"""In-memory session that loads a reconstructed lattice and turns per-knot
latent-code edits into meshes for the latent-edit GUI.

The heavy lifting (model + B-spline of latent codes + mesh extraction) is reused
from :mod:`deepshapeopt.reconstruction` and ``DeepSDFStruct`` so this module only
adds: locating/loading (or fitting) the reconstructed control-point codes,
exposing the control net as a list of editable knots, and re-meshing after an
edit.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np
import torch

from deepshapeopt.config import ExperimentSpecifications, make_experiment_paths
from deepshapeopt.reconstruction import (
    build_reconstruction_lattice,
    fit_lattice_to_sdf,
    with_float32_lattice,
)

logger = logging.getLogger(__name__)


def _case_name(mesh_path: Path, tiling) -> str:
    """Mirror ``reconstruct_shape``'s auto case name so we look in the same
    directory it (or the optimization phase) saved ``rec_parameters.pt`` to."""
    tiling_str = "x".join(map(str, tiling))
    return f"{mesh_path.stem}_tiling_{tiling_str}"


def _resolve_param_file(experiment_path: Path, specs, mesh_path: Path, tiling) -> Path:
    """Path of the persisted control-point codes for this experiment.

    Matches the ``heavy_dir`` used by ``reconstruct_shape``: heavy-data dir when
    ``heavy_data_output_path`` is configured, otherwise the local results dir.
    """
    rec_cfg = specs["reconstruction"]
    paths = make_experiment_paths(
        experiment_path,
        results_name=specs.get("results_name", "results"),
        heavy_data_output_path=rec_cfg.get("heavy_data_output_path"),
    )
    case = _case_name(mesh_path, tiling)
    if paths.heavy_data is not None:
        heavy_dir = paths.heavy_data / "reconstruction" / case
    else:
        heavy_dir = paths.reconstruction / case
    return heavy_dir / "rec_parameters.pt"


def _greville_positions(spline) -> tuple[np.ndarray, np.ndarray]:
    """Greville abscissae (parametric/world positions of the control points)
    row-aligned with ``spline.control_points`` (and hence with
    ``torch_spline.control_points``), plus their (i, j, k) grid indices.

    Prefers splinepy's own ``greville_abscissae`` (guaranteed same ordering as
    the control points); falls back to the tensor product of the per-dimension
    Greville points in splinepy's first-axis-fastest ("F") ordering.
    """
    from DeepSDFStruct.export_knot_grid import _greville_1d

    degrees = np.asarray(spline.degrees, dtype=int)
    kvs = [np.asarray(kv, dtype=float) for kv in spline.knot_vectors]
    per_dim = [_greville_1d(kvs[i], int(degrees[i])) for i in range(len(kvs))]
    n_per_dim = [len(g) for g in per_dim]
    n_total = int(np.prod(n_per_dim))

    # (i, j, k) indices in splinepy's first-axis-fastest ordering.
    ids = np.arange(n_total)
    idx = np.unravel_index(ids, n_per_dim, order="F")

    positions = None
    grev = getattr(spline, "greville_abscissae", None)
    if grev is not None:
        try:
            arr = np.asarray(grev() if callable(grev) else grev, dtype=float)
            if arr.shape[0] == n_total:
                positions = arr
        except Exception:  # pragma: no cover - defensive
            positions = None
    if positions is None:
        positions = np.column_stack(
            [per_dim[d][idx[d]] for d in range(len(per_dim))]
        )

    ijk = np.column_stack(idx) if len(idx) else np.zeros((n_total, 0), dtype=int)
    return positions, ijk


class LatentEditSession:
    """Owns a reconstructed lattice and re-meshes it after latent-code edits.

    Thread-safe: every public method that touches the lattice takes an internal
    lock, so the stdlib threaded HTTP server can serve concurrent requests
    without corrupting the shared control-point tensor.
    """

    def __init__(
        self,
        config_path: str | Path,
        device: str | None = None,
        fit_if_missing: bool = True,
        mesh_n: int | None = None,
        source: str = "reconstruction",
        params_file: str | Path | None = None,
    ):
        if source not in ("reconstruction", "optimization"):
            raise ValueError(
                f"Unknown latent-GUI source {source!r}; use 'reconstruction' or 'optimization'."
            )
        self.config_path = Path(config_path).resolve()
        self.experiment_path = self.config_path.parent
        self.specs = ExperimentSpecifications(str(self.config_path))
        self.rec_cfg = self.specs["reconstruction"]
        self.source = source
        self.params_file = Path(params_file).resolve() if params_file else None
        self._lock = threading.RLock()

        logger.info("Building %s lattice for %s", source, self.config_path)
        if source == "optimization":
            self._build_optimization_lattice(device=device)
        else:
            self._build_reconstruction_lattice(device=device)

        # FlexiCubes meshes a lattice on an ``N_base * tiling`` grid, so the
        # reconstruction's ``create_mesh_N`` can blow up for heavily-tiled
        # cases (e.g. tiling [1,8,8] -> 32*256*256 points). For *live* editing
        # we mesh at a lower per-cell resolution so each edit stays responsive;
        # the heuristic keeps the largest tiled dimension near ~64 cells.
        max_tile = max(int(t) for t in self.tiling) if len(self.tiling) else 1
        default_n = max(4, min(self.create_mesh_N, 64 // max(max_tile, 1)))
        self.mesh_n = int(mesh_n) if mesh_n else int(default_n)

        self._control_points = self.param_spline.torch_spline.control_points
        self.n_knots = int(self._control_points.shape[0])

        positions, ijk = _greville_positions(self.param_spline_sp)
        self._knot_positions = positions.astype(float)
        self._knot_ijk = ijk.astype(int)

        if source == "optimization":
            self._load_optimization_codes()
        else:
            self._load_or_fit_codes(fit_if_missing=fit_if_missing)
        self._original_codes = self._control_points.detach().clone()

    # -- lattice construction ---------------------------------------------
    def _build_reconstruction_lattice(self, device: str | None = None) -> None:
        """Whole-input-mesh reconstruction lattice (the default GUI source)."""
        built = build_reconstruction_lattice(self.specs, device=device)
        self.lattice_struct = built["lattice_struct"]
        self.param_spline = built["param_spline"]
        self.param_spline_sp = built["param_spline_sp"]
        self.bounds = built["bounds"]
        self.mesh_norm = built["mesh_norm"]
        self.latent_dim = int(built["latent_dim"])
        self.tiling = built["tiling"]
        self.create_mesh_N = int(built["create_mesh_N"])
        self.device = built["device"]
        self.mesh_path = built["mesh_path"]
        self.code_bound = float(self.rec_cfg.get("code_bound", 1.0))

    def _build_optimization_lattice(self, device: str | None = None) -> None:
        """Rebuild the *optimization* design-domain lattice exactly as the optimizer
        does: a B-spline of latent codes over the physical ``design_domain`` normalized
        by the DomainFrame, rather than over the whole input mesh. The meshed surface is
        therefore only the inner design region the optimizer actually moves."""
        from deepshapeopt.config import ensure_experiment_dirs, make_experiment_paths
        from deepshapeopt.shape_optimization import build_lattice, setup_model_and_domain

        if device is not None:
            self.rec_cfg = {**self.rec_cfg, "device": device}
        opt_cfg = self.specs.get("optimization", {})
        self._paths = make_experiment_paths(
            self.experiment_path,
            results_name=self.specs.get("results_name", "results"),
            heavy_data_output_path=opt_cfg.get("heavy_data_output_path"),
        )
        ensure_experiment_dirs(self._paths)

        ms = setup_model_and_domain(self.rec_cfg, self._paths.reconstruction)
        ls = build_lattice(self.rec_cfg, ms.model, ms.sdf, ms.frame)
        self.frame = ms.frame
        self.lattice_struct = ls.lattice_struct
        self.param_spline = ls.param_spline
        self.param_spline_sp = ls.param_spline_sp
        self.bounds = ms.frame.box_norm
        self.mesh_norm = None
        self.latent_dim = int(ms.model._trained_latent_vectors[0].shape[0])
        self.tiling = self.rec_cfg["tiling"]
        self.create_mesh_N = int(
            self.rec_cfg.get("create_mesh_N", opt_cfg.get("mesh_resolution", 32))
        )
        self.device = self.rec_cfg["device"]
        self.mesh_path = Path(self.rec_cfg["mesh_path"]).resolve()
        # Slider range: use the optimizer's design-variable bounds when present.
        opt_bounds = opt_cfg.get("bounds")
        self.code_bound = (
            float(max(abs(float(b)) for b in opt_bounds))
            if opt_bounds
            else float(self.rec_cfg.get("code_bound", 1.0))
        )

    # -- loading ----------------------------------------------------------
    def _load_optimization_codes(self) -> None:
        """Load an optimization run's saved latent control points and adopt them as
        the editable shape. Never fits: in optimization mode the GUI inspects an
        *existing* optimized design, so a missing parameter file is an error."""
        expected = tuple(self._control_points.shape)
        if self.params_file is not None:
            candidates = [self.params_file]
        else:
            candidates = [
                self._paths.optimization / "updated_parameters.pt",
                self._paths.reconstruction / "rec_parameters.pt",
            ]
        for f in candidates:
            if f.exists():
                data = torch.load(f, map_location=self.device)
                codes = data[0] if isinstance(data, (list, tuple)) else data
                if tuple(codes.shape) != expected:
                    raise ValueError(
                        f"Saved latents at {f} have shape {tuple(codes.shape)} but this "
                        f"configuration expects {expected}. The config's tiling/model must "
                        f"match the run that produced the parameters."
                    )
                self.param_spline.set_param(
                    codes.to(device=self._control_points.device, dtype=self._control_points.dtype)
                )
                logger.info("Loaded optimization latents from %s", f)
                return
        searched = "\n  ".join(str(c) for c in candidates)
        raise FileNotFoundError(
            "No optimization latent parameters found to load (optimization-mode GUI does "
            "not fit on the fly). Run the optimization for this experiment first, or pass "
            "--params-file explicitly. Looked for:\n  " + searched
        )

    # -- reconstruction loading ------------------------------------------
    def _load_or_fit_codes(self, fit_if_missing: bool) -> None:
        param_file = _resolve_param_file(
            self.experiment_path, self.specs, self.mesh_path, self.tiling
        )
        expected = tuple(self._control_points.shape)

        if param_file.exists():
            recon_param = torch.load(param_file, map_location=self.device)
            codes = recon_param[0] if isinstance(recon_param, (list, tuple)) else recon_param
            if tuple(codes.shape) != expected:
                raise ValueError(
                    f"Saved rec_parameters.pt has shape {tuple(codes.shape)} but "
                    f"this configuration expects {expected}. Delete the stale file "
                    f"or fix the config: {param_file}"
                )
            logger.info("Loaded reconstructed codes from %s", param_file)
            self.param_spline.set_param(
                codes.to(device=self._control_points.device, dtype=self._control_points.dtype)
            )
            return

        if not fit_if_missing:
            raise FileNotFoundError(
                f"No saved reconstruction found at {param_file}. Run "
                f"`python scripts/reconstruct.py --config {self.config_path}` first, "
                f"or launch the GUI without --no-fit to fit on the fly."
            )

        logger.info(
            "No saved codes at %s — running reconstruction fit (this is a one-off; "
            "the result is cached for next launch).",
            param_file,
        )
        param_file.parent.mkdir(parents=True, exist_ok=True)
        result = fit_lattice_to_sdf(
            self.lattice_struct,
            self.mesh_norm,
            self.bounds,
            self.rec_cfg,
            output_dir=param_file.parent,
            save_vtp=False,
            box_constrained=False,
        )
        self.param_spline.set_param(result["params"][0])
        torch.save(result["params"], param_file)
        logger.info("Saved reconstructed codes to %s", param_file)

    # -- queries ----------------------------------------------------------
    def knots(self) -> list[dict]:
        """One entry per control point: index, (i, j, k) and world position."""
        out = []
        for idx in range(self.n_knots):
            pos = self._knot_positions[idx]
            ijk = self._knot_ijk[idx] if self._knot_ijk.shape[1] else []
            out.append(
                {
                    "idx": idx,
                    "ijk": [int(v) for v in ijk],
                    "pos": [float(v) for v in pos],
                }
            )
        return out

    def codes(self) -> list[list[float]]:
        with self._lock:
            arr = self._control_points.detach().cpu().numpy()
        return arr.astype(float).tolist()

    def code(self, knot_idx: int) -> list[float]:
        with self._lock:
            return self._control_points[knot_idx].detach().cpu().numpy().astype(float).tolist()

    # -- edits ------------------------------------------------------------
    def set_value(self, knot_idx: int, dim: int, value: float) -> None:
        if not (0 <= knot_idx < self.n_knots):
            raise IndexError(f"knot_idx {knot_idx} out of range [0, {self.n_knots})")
        if not (0 <= dim < self.latent_dim):
            raise IndexError(f"dim {dim} out of range [0, {self.latent_dim})")
        with self._lock:
            new_codes = self._control_points.detach().clone()
            new_codes[knot_idx, dim] = float(value)
            self.param_spline.set_param(new_codes)

    def reset(self) -> None:
        with self._lock:
            self.param_spline.set_param(self._original_codes.clone())

    # -- meshing ----------------------------------------------------------
    def mesh(self) -> dict:
        """Extract the current zero-level-set surface as flat vertex/face arrays
        (param space — same frame as the knot positions, so they overlay)."""
        from DeepSDFStruct.mesh import create_3D_mesh

        with self._lock:
            def _make(bounds_f32):
                surf, _ = create_3D_mesh(
                    self.lattice_struct,
                    self.mesh_n,
                    mesh_type="surface",
                    differentiate=False,
                    device=self.device,
                    bounds=bounds_f32,
                    extend_bounds=True,
                )
                return surf

            surf = with_float32_lattice(self.lattice_struct, self.bounds, _make)

        verts = surf.vertices.detach().cpu().numpy().astype(np.float32)
        faces = surf.faces.detach().cpu().numpy().astype(np.int32)
        return {
            "vertices": verts.reshape(-1).tolist(),
            "faces": faces.reshape(-1).tolist(),
            "n_vertices": int(verts.shape[0]),
            "n_faces": int(faces.shape[0]),
        }

    # -- combined ---------------------------------------------------------
    def state(self) -> dict:
        """Everything the frontend needs on first load."""
        return {
            "latent_dim": self.latent_dim,
            "code_bound": self.code_bound,
            "n_knots": self.n_knots,
            "mesh_n": self.mesh_n,
            "tiling": list(self.tiling),
            "bounds": self.bounds.detach().cpu().numpy().astype(float).tolist(),
            "knots": self.knots(),
            "codes": self.codes(),
            "mesh": self.mesh(),
        }

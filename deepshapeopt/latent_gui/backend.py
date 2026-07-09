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
        mesh_n: int | None = None,
        source: str = "auto",
        params_file: str | Path | None = None,
    ):
        self.config_path = Path(config_path).resolve()
        self.experiment_path = self.config_path.parent
        self.specs = ExperimentSpecifications(str(self.config_path))
        self.rec_cfg = self.specs["reconstruction"]
        if source == "auto":
            # Inspect the optimization result when the config drives an
            # optimization; otherwise show the whole-input-mesh reconstruction.
            source = "optimization" if self.specs.get("optimization") else "reconstruction"
        if source not in ("reconstruction", "optimization"):
            raise ValueError(
                f"Unknown latent-GUI source {source!r}; use 'auto', 'reconstruction' "
                "or 'optimization'."
            )
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

        self._load_codes()
        self._original_codes = self._control_points.detach().clone()
        self._setup_pca()

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
    def _load_codes(self) -> None:
        """Load the saved latent control points to display and adopt them as the
        editable shape. The GUI never fits on the fly: it always inspects an
        *existing* saved design, so a missing parameter file is an error.

        The file is located from ``--config`` (overridable with ``--params-file``):
        the optimization source prefers the optimizer's ``parameters.pt`` and falls
        back to the reconstruction's ``rec_parameters.pt``; the reconstruction source
        uses the reconstruction's ``rec_parameters.pt``.
        """
        if self.params_file is not None:
            candidates = [self.params_file]
        elif self.source == "optimization":
            candidates = [
                #self._paths.optimization / "parameters.pt",
                self._paths.reconstruction / "rec_parameters.pt",
            ]
        else:
            candidates = [
                _resolve_param_file(
                    self.experiment_path, self.specs, self.mesh_path, self.tiling
                )
            ]

        for f in candidates:
            if f.exists():
                self._apply_codes(f)
                return

        searched = "\n  ".join(str(c) for c in candidates)
        raise FileNotFoundError(
            f"No saved latent parameters found to load for the {self.source} source. The "
            "latent GUI never fits on the fly; run the reconstruction/optimization for "
            "this experiment first (e.g. "
            f"`python scripts/reconstruct.py --config {self.config_path}`), or pass "
            "--params-file explicitly. Looked for:\n  " + searched
        )

    def _apply_codes(self, param_file: Path) -> None:
        """Load latent control points from *param_file* and set them on the spline,
        validating their shape against this configuration's control net."""
        expected = tuple(self._control_points.shape)
        data = torch.load(param_file, map_location=self.device)
        codes = data[0] if isinstance(data, (list, tuple)) else data
        if tuple(codes.shape) != expected:
            raise ValueError(
                f"Saved latents at {param_file} have shape {tuple(codes.shape)} but this "
                f"configuration expects {expected}. The config's tiling/model must match "
                "the run that produced the parameters (delete the stale file or fix the "
                "config)."
            )
        self.param_spline.set_param(
            codes.to(device=self._control_points.device, dtype=self._control_points.dtype)
        )
        logger.info("Loaded latents from %s", param_file)

    # -- PCA reduced basis ------------------------------------------------
    def _setup_pca(self) -> None:
        """When the config enables PCA latent reduction (an ``optimization.pca`` block),
        rebuild the same whitened PCA basis the optimizer uses so the GUI can also edit
        each knot's PCA coefficients. No-op otherwise (``pca_basis`` stays ``None``).

        Fits the basis directly from the model's training latents (rather than via
        ``build_pca_basis``) so we can also keep the per-component explained-variance
        ratio for display; the mean/components/scale are identical to the optimizer's.
        """
        self.pca_basis = None
        self.pca_bounds: list[float] | None = None
        self.pca_explained: list[float] = []
        self.n_components = 0
        if self.source != "optimization":
            return
        pca_cfg = self.specs.get("optimization", {}).get("pca") or {}
        if not pca_cfg.get("enabled", False):
            return

        from deepshapeopt.latent_pca import (
            PCALatentBasis,
            compute_latent_pca,
            gather_training_latents,
        )

        dev = self._control_points.device
        dtype = self._control_points.dtype
        latents = gather_training_latents(
            self.rec_cfg["model_path"],
            self.rec_cfg.get("model_checkpoint", "latest"),
            device=dev,
        )
        mean, components, explained, scale = compute_latent_pca(
            latents, int(pca_cfg["n_components"])
        )
        self.pca_basis = PCALatentBasis(mean, components, scale).to(device=dev, dtype=dtype)
        # Delta parametrization: measure coefficients as deltas around the reconstruction
        # lambda^0 (same origin the optimizer uses), so the sliders match the optimizer's
        # design variables and truncation keeps the reconstruction detail.
        self.pca_basis.set_reference(self._pca_reference_field())
        self.pca_explained = [float(e) for e in explained]
        b = pca_cfg.get("bounds", [-2.5, 2.5])
        self.pca_bounds = [float(b[0]), float(b[1])]
        self.n_components = int(self.pca_basis.n_components)
        logger.info(
            "PCA panel enabled: %d modes (%.1f%% variance), coefficient box %s "
            "(deltas around the reconstruction)",
            self.n_components,
            100.0 * sum(self.pca_explained),
            self.pca_bounds,
        )

    def _pca_reference_field(self) -> torch.Tensor:
        """The reconstruction lambda^0 the PCA deltas are measured from -- the same origin
        the optimizer uses. Loads the run's ``rec_parameters.pt``; falls back to the loaded
        field (deltas measured from what's shown, all zero at load) if it is unavailable."""
        cp = self._control_points
        rec_file = self._paths.reconstruction / "rec_parameters.pt"
        if rec_file.exists():
            data = torch.load(rec_file, map_location=cp.device)
            ref = data[0] if isinstance(data, (list, tuple)) else data
            if tuple(ref.shape) == tuple(cp.shape):
                logger.info("PCA delta reference: reconstruction %s", rec_file)
                return ref.to(device=cp.device, dtype=cp.dtype)
            logger.warning(
                "rec_parameters.pt shape %s != %s; measuring PCA deltas from the loaded field.",
                tuple(ref.shape), tuple(cp.shape),
            )
        else:
            logger.warning(
                "No rec_parameters.pt at %s; measuring PCA deltas from the loaded field.",
                rec_file,
            )
        return cp.detach().clone()

    def coeffs(self) -> list[list[float]] | None:
        """Current whitened PCA coefficients ``(n_knots, k)`` derived from the live
        latent control points, or ``None`` when PCA is not enabled."""
        if self.pca_basis is None:
            return None
        with self._lock:
            c = self.pca_basis.to_coeff(self._control_points.detach())
        return c.cpu().numpy().astype(float).tolist()

    def coeff(self, knot_idx: int) -> list[float] | None:
        """One knot's whitened PCA coefficients, or ``None`` when PCA is not enabled."""
        if self.pca_basis is None:
            return None
        with self._lock:
            row = self._control_points[knot_idx : knot_idx + 1].detach()
            c = self.pca_basis.to_coeff(row)
        return c[0].cpu().numpy().astype(float).tolist()

    def set_coeff(self, knot_idx: int, comp: int, value: float) -> None:
        """Move a single knot along one PCA direction: ``z += (value - c_old) * scale_j
        * V[:, j]``. Editing the coefficient in place leaves the orthogonal complement
        (the part of the latent not captured by the k components) and every other knot
        untouched, so it composes cleanly with raw-latent edits."""
        if self.pca_basis is None:
            raise RuntimeError("PCA editing requested but no PCA basis is loaded.")
        if not (0 <= knot_idx < self.n_knots):
            raise IndexError(f"knot_idx {knot_idx} out of range [0, {self.n_knots})")
        if not (0 <= comp < self.n_components):
            raise IndexError(f"comp {comp} out of range [0, {self.n_components})")
        with self._lock:
            z = self._control_points.detach().clone()
            old = float(self.pca_basis.to_coeff(z[knot_idx : knot_idx + 1])[0, comp])
            delta = (float(value) - old) * self.pca_basis.scale[comp]
            z[knot_idx] = z[knot_idx] + delta * self.pca_basis.components[:, comp]
            self.param_spline.set_param(z)

    def truncate_to_k(self, k: int) -> None:
        """Zero every knot's PCA coefficients from component ``k`` on, i.e. project the
        whole design onto its first ``k`` principal directions. This reproduces the
        starting design of a reduced ``n_components = k`` optimization run (which drops
        the tail at init), so you can preview what a smaller basis has to work with.
        Lossy: the discarded tail cannot be recovered except via ``reset``."""
        if self.pca_basis is None:
            raise RuntimeError("PCA truncation requested but no PCA basis is loaded.")
        if not (1 <= k <= self.n_components):
            raise IndexError(f"k {k} out of range [1, {self.n_components}]")
        with self._lock:
            c = self.pca_basis.to_coeff(self._control_points.detach()).clone()
            c[:, k:] = 0.0
            self.param_spline.set_param(self.pca_basis.to_latent(c))

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
        st = {
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
        if self.pca_basis is not None:
            st["pca"] = {
                "enabled": True,
                "n_components": self.n_components,
                "bounds": self.pca_bounds,
                "explained": self.pca_explained,
            }
            st["coeffs"] = self.coeffs()
        else:
            st["pca"] = {"enabled": False}
        return st

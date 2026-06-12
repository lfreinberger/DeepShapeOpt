"""Per-iteration driver of the SDF-driven hex mesh pipeline (``sdf_hex``).

Replaces the FlexiCubes-STL + snappyHexMesh path: the body-fitted hex mesh
for the OpenFOAM forward/adjoint run is generated directly from the
(differentiable) SDF.  The mesh outside the design-space ``mesh_box`` is
built once and is identical across all iterations; only the castellation
inside the box changes, so shape topology changes are supported.  OpenFOAM
point sensitivities map back onto the snapped wall points by index (we wrote
the points file), and flow through the differentiable snap step into the
lattice parameters.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from .constraints import volume_centroid_from_wall
from .foamwriter import checkmesh_log_ok, write_polymesh
from .lattice import CellSet, Lattice
from .octree import (
    Castellation,
    MeshBox,
    build_inner_castellation,
    build_static_octree,
)
from .polymesh import PolyMeshData, build_polymesh, face_pyramid_volumes
from .sdf_field import PhysicalSDF
from .snap import SnapHandle, snap_wall_points

logger = logging.getLogger(__name__)

_DEFAULTS = {
    "domain": [[-5.0, -5.0, -5.0], [15.0, 5.0, 5.0]],
    "base_cell_size": 1.0,
    "mesh_box": [[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]],
    "interface_level": 2,
    "max_level": 5,
    "refine_band_beta": 1.0,
    "snap_iters": 4,
    "max_disp_frac": 0.7,
    "smooth_iters": 2,
    "min_pyr_frac": 0.02,
    "quality_max_rounds": 5,
    "enable_fast_path": True,
    "check_sign": True,
    "sign_probe_point": None,
    # checkMesh gate: failed checks whose description matches one of these
    # substrings only log a warning instead of aborting the iteration.
    "checkmesh_ignore": ["skew"],
}


@dataclasses.dataclass
class HexMeshResult:
    mesh: PolyMeshData  # with snapped (detached) point coordinates
    surface_point_ids: np.ndarray  # indices into mesh.points == OpenFOAM ids
    surface_points: torch.Tensor  # [P, 3] differentiable snapped positions
    snap_lambda: np.ndarray  # [P] displacement scaling (1 = fully snapped)
    wall_tris_local: torch.Tensor  # [T, 3] triangulated wall, local indices
    castellation_hash: str
    reused_connectivity: bool
    stats: dict


class SdfHexMeshPipeline:
    """Builds and writes the OpenFOAM polyMesh directly from the SDF."""

    def __init__(self, lattice_struct, model_setup, opt_cfg: dict, results_dir: Path):
        self.lattice_struct = lattice_struct
        self.model_setup = model_setup
        self.cfg = {**_DEFAULTS, **opt_cfg.get("sdf_hex", {})}
        self.results_dir = Path(results_dir)
        self.device = model_setup.design_domain.device

        domain = np.asarray(self.cfg["domain"], dtype=np.float64)
        h0 = float(self.cfg["base_cell_size"])
        root_dims = (domain[1] - domain[0]) / h0
        if not np.allclose(root_dims, np.rint(root_dims)):
            raise ValueError("domain extents must be multiples of base_cell_size")

        self.lattice = Lattice(
            origin=domain[0],
            h0=h0,
            root_dims=np.rint(root_dims).astype(np.int64),
            max_depth=int(self.cfg["max_level"]),
        )
        self.mesh_box = MeshBox.from_physical(self.lattice, self.cfg["mesh_box"])
        self.interface_level = int(self.cfg["interface_level"])
        self.max_level = int(self.cfg["max_level"])

        self._validate_margins()
        self.static_cells = self._load_or_build_static()

        self._sign_checked = False
        self._last_hash: str | None = None
        self._last_mesh: PolyMeshData | None = None
        self._last_castellation: Castellation | None = None
        self._last_result: HexMeshResult | None = None

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _validate_margins(self) -> None:
        dd = self.model_setup.design_domain.detach().cpu().numpy().astype(np.float64)
        box = np.asarray(self.cfg["mesh_box"], dtype=np.float64)
        margin = float(min((dd[0] - box[0]).min(), (box[1] - dd[1]).min()))
        w_iface = self.lattice.h0 / (1 << self.interface_level)
        if margin < 2.0 * w_iface:
            raise ValueError(
                f"Margin between design_domain and mesh_box is {margin:.3f} but "
                f"must be >= 2 interface cells ({2 * w_iface:.3f}). Enlarge "
                "mesh_box or raise interface_level."
            )
        # Cells at the design-domain edge may be refined to max_level; the
        # frozen shell plus the 2:1 grading chain must fit into the margin
        # (with one max_level cell of slack for band cells straddling the
        # design-domain boundary).
        from .octree import grading_distance

        w_max = self.lattice.h0 / (1 << self.max_level)
        required = grading_distance(
            self.lattice, self.interface_level, self.max_level
        ) * self.lattice.h_fine
        if required > margin - w_max:
            raise ValueError(
                f"Refinement to max_level needs {required + w_max:.3f} of "
                f"margin between design_domain and mesh_box but only "
                f"{margin:.3f} is available. Enlarge mesh_box, raise "
                "interface_level, or lower max_level."
            )

    def _static_cache_path(self) -> Path:
        key = hashlib.sha256(
            repr(
                (
                    self.cfg["domain"],
                    self.cfg["base_cell_size"],
                    self.cfg["mesh_box"],
                    self.interface_level,
                    self.max_level,
                )
            ).encode()
        ).hexdigest()[:16]
        return self.results_dir / f"static_octree_{key}.npz"

    def _load_or_build_static(self) -> CellSet:
        path = self._static_cache_path()
        if path.exists():
            data = np.load(path)
            cells = CellSet(levels=data["levels"], anchors=data["anchors"])
            logger.info("Loaded static outer octree from %s (%d cells)", path, len(cells))
            return cells
        cells = build_static_octree(self.lattice, self.mesh_box, self.interface_level)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, levels=cells.levels, anchors=cells.anchors)
        return cells

    def _make_sdf(self) -> PhysicalSDF:
        sdf = PhysicalSDF(
            sdf_norm_fn=self.lattice_struct,
            norm_fn=self.model_setup.norm_fn,
            design_domain=self.model_setup.design_domain.detach(),
            dist_scale=1.0 / float(self.model_setup.scale),
            device=self.device,
        )
        if self.cfg["check_sign"] and not self._sign_checked:
            probe = self.cfg["sign_probe_point"]
            if probe is None:
                dd = self.model_setup.design_domain.detach().cpu().numpy()
                probe = 0.5 * (dd[0] + dd[1])
            sdf.check_sign_convention(probe)
            self._sign_checked = True
        return sdf

    def _with_float32(self, fn):
        from deepshapeopt.reconstruction import with_float32_lattice

        return with_float32_lattice(self.lattice_struct, self.model_setup.box_norm, lambda _b: fn())

    # ------------------------------------------------------------------
    # Per-iteration build
    # ------------------------------------------------------------------

    def build(self, reuse_castellation: bool = False) -> HexMeshResult:
        """Castellate + snap + assemble; returns the differentiable handle.

        With ``reuse_castellation=True`` the previous castellation/connectivity
        is reused and only the snap is recomputed (boundary-movement mode,
        also used for finite-difference gradient checks).
        """
        return self._with_float32(lambda: self._build_impl(reuse_castellation))

    def _build_impl(self, reuse_castellation: bool) -> HexMeshResult:
        t0 = time.time()
        sdf = self._make_sdf()

        if reuse_castellation:
            if self._last_castellation is None:
                raise RuntimeError("No previous castellation to reuse")
            cast = self._last_castellation
        else:
            cast = build_inner_castellation(
                self.lattice,
                self.mesh_box,
                self.interface_level,
                self.max_level,
                sdf.phi_ext_np,
                beta=float(self.cfg["refine_band_beta"]),
            )

        cast_hash = hashlib.sha256(
            cast.cells.levels.tobytes() + cast.cells.anchors.tobytes()
        ).hexdigest()

        reused = (
            bool(self.cfg["enable_fast_path"])
            and self._last_mesh is not None
            and cast_hash == self._last_hash
        )
        if reused:
            mesh = self._last_mesh
            logger.info("Castellation unchanged - reusing mesh connectivity")
        else:
            cells = CellSet.concat(self.static_cells, cast.cells)
            mesh = build_polymesh(self.lattice, cells, self.mesh_box)

        from .lattice import unpack_keys

        wall_ids = mesh.wall_point_ids()
        points0 = self.lattice.point_coords(unpack_keys(mesh.point_keys))
        x0 = points0[wall_ids]
        h_local = self._wall_h_local(mesh, wall_ids)

        handle = snap_wall_points(
            sdf,
            x0,
            h_local,
            snap_iters=int(self.cfg["snap_iters"]),
            max_disp_frac=float(self.cfg["max_disp_frac"]),
            smooth_iters=int(self.cfg["smooth_iters"]),
            wall_edges=self._wall_edges_local(mesh, wall_ids),
        )
        final_points, n_relaxed = self._quality_guard(mesh, points0, wall_ids, handle)

        x_star = handle.x_star()
        snapped_mesh = dataclasses.replace(mesh, points=final_points)

        result = HexMeshResult(
            mesh=snapped_mesh,
            surface_point_ids=wall_ids,
            surface_points=x_star,
            snap_lambda=handle.lam.detach().cpu().numpy(),
            wall_tris_local=self._wall_tris_local(mesh, wall_ids),
            castellation_hash=cast_hash,
            reused_connectivity=reused,
            stats={
                "n_cells": mesh.n_cells,
                "n_points": len(mesh.points),
                "n_wall_faces": mesh.patch_by_name("dragObject").n_faces,
                "n_wall_points": len(wall_ids),
                "n_relaxed_points": n_relaxed,
                "snap_residual_max": float(handle.residuals().max()),
                "build_seconds": time.time() - t0,
            },
        )
        logger.info("Hex mesh build: %s", result.stats)

        self._last_hash = cast_hash
        self._last_mesh = mesh
        self._last_castellation = cast
        self._last_result = result
        return result

    def _wall_h_local(self, mesh: PolyMeshData, wall_ids: np.ndarray) -> np.ndarray:
        """Per wall point: finest adjacent wall face size (physical)."""
        wall_faces = mesh.faces[mesh.wall_face_slice()]
        from .lattice import unpack_keys

        corners = unpack_keys(mesh.point_keys[wall_faces.ravel()]).reshape(-1, 4, 3)
        width_fine = (corners.max(axis=1) - corners.min(axis=1)).max(axis=1)  # [Fw]
        h_face = width_fine.astype(np.float64) * self.lattice.h_fine

        h_local = np.full(len(wall_ids), np.inf)
        local = np.searchsorted(wall_ids, wall_faces.ravel()).reshape(-1, 4)
        for k in range(4):
            np.minimum.at(h_local, local[:, k], h_face)
        return h_local

    def _wall_tris_local(self, mesh: PolyMeshData, wall_ids: np.ndarray) -> torch.Tensor:
        """Fan-triangulated wall faces with indices local to ``wall_ids``."""
        wall_faces = mesh.faces[mesh.wall_face_slice()]
        local = np.searchsorted(wall_ids, wall_faces)
        tris = np.concatenate([local[:, [0, 1, 2]], local[:, [0, 2, 3]]], axis=0)
        return torch.as_tensor(tris, dtype=torch.long)

    def _wall_edges_local(self, mesh: PolyMeshData, wall_ids: np.ndarray) -> np.ndarray:
        """Unique wall-surface edges with indices local to ``wall_ids``."""
        wall_faces = mesh.faces[mesh.wall_face_slice()]
        local = np.searchsorted(wall_ids, wall_faces)  # [Fw, 4]
        edges = np.concatenate(
            [local[:, [0, 1]], local[:, [1, 2]], local[:, [2, 3]], local[:, [3, 0]]],
            axis=0,
        )
        edges = np.sort(edges, axis=1)
        return np.unique(edges, axis=0)

    def _quality_guard(
        self,
        mesh: PolyMeshData,
        points0: np.ndarray,
        wall_ids: np.ndarray,
        handle: SnapHandle,
    ) -> tuple[np.ndarray, int]:
        """Scale back snap displacements that would degenerate cells."""
        ref_owner, ref_neigh = face_pyramid_volumes(mesh, points0)
        frac = float(self.cfg["min_pyr_frac"])
        floor = 1e-13
        n_int = mesh.n_internal_faces
        relaxed_total = np.zeros(len(wall_ids), dtype=bool)

        for round_idx in range(int(self.cfg["quality_max_rounds"]) + 1):
            pts = points0.copy()
            pts[wall_ids] = handle.x_star().detach().double().cpu().numpy()
            pyr_o, pyr_n = face_pyramid_volumes(mesh, pts)
            bad_face = pyr_o < np.maximum(frac * ref_owner, floor)
            bad_face[:n_int] |= pyr_n < np.maximum(frac * ref_neigh, floor)
            if not np.any(bad_face):
                return pts, int(relaxed_total.sum())

            bad_pts = np.unique(mesh.faces[bad_face].ravel())
            member = np.isin(bad_pts, wall_ids)
            mask = np.zeros(len(wall_ids), dtype=bool)
            mask[np.searchsorted(wall_ids, bad_pts[member])] = True
            relaxed_total |= mask

            if round_idx < int(self.cfg["quality_max_rounds"]):
                handle.reduce_lambda(mask, factor=0.75)
                logger.debug(
                    "Quality guard round %d: %d bad faces, relaxing %d points",
                    round_idx, int(bad_face.sum()), int(mask.sum()),
                )
            else:
                handle.zero_lambda(mask)
                logger.warning(
                    "Quality guard: %d wall points left unsnapped (lambda=0)",
                    int(mask.sum()),
                )

        pts = points0.copy()
        pts[wall_ids] = handle.x_star().detach().double().cpu().numpy()
        return pts, int(relaxed_total.sum())

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    def volume_centroid(self, no_grad: bool = False):
        """Volume and centroid of the meshed body, differentiable through the snap.

        Divergence theorem over the snapped wall triangulation of the last
        :meth:`build` — measures exactly the body the CFD sees.
        """
        if self._last_result is None:
            raise RuntimeError("build() must be called before volume_centroid()")
        points = self._last_result.surface_points
        tris = self._last_result.wall_tris_local
        if no_grad:
            with torch.no_grad():
                return volume_centroid_from_wall(points.detach(), tris)
        return volume_centroid_from_wall(points, tris)

    # ------------------------------------------------------------------
    # OpenFOAM coupling
    # ------------------------------------------------------------------

    def run_case(self, case_dir: Path, verbose: bool = False):
        """Clean the case, write the polyMesh, run Allrun, gate on checkMesh."""
        import contextlib
        import io as _io

        from foamlib import FoamCase

        import deepshapeopt.foam_utils as foam_utils

        if self._last_result is None:
            raise RuntimeError("build() must be called before run_case()")

        case_dir = Path(case_dir)
        foam_case = FoamCase(case_dir)
        if verbose:
            foam_case.clean()
        else:
            with contextlib.redirect_stdout(_io.StringIO()), contextlib.redirect_stderr(
                _io.StringIO()
            ):
                foam_case.clean()

        write_polymesh(self._last_result.mesh, case_dir)
        foam_utils.run_openfoam_case(case_dir, verbose=verbose, clean=False)

        log_path = case_dir / "log.checkMesh"
        if not log_path.exists():
            raise RuntimeError(f"checkMesh log not found: {log_path}")
        ok, failures = checkmesh_log_ok(log_path.read_text())
        if not ok:
            ignore = [str(s).lower() for s in self.cfg["checkmesh_ignore"]]
            fatal = [
                f for f in failures
                if not any(pattern in f.lower() for pattern in ignore)
            ]
            soft = [f for f in failures if f not in fatal]
            if soft:
                logger.warning("checkMesh quality warnings (ignored): %s", soft)
            if fatal or not failures:
                archive = self.results_dir / "failed_mesh"
                archive.mkdir(parents=True, exist_ok=True)
                shutil.copytree(
                    case_dir / "constant" / "polyMesh", archive / "polyMesh", dirs_exist_ok=True
                )
                shutil.copy2(log_path, archive / "log.checkMesh")
                raise RuntimeError(
                    f"checkMesh failed: {failures}; mesh archived to {archive}"
                )
        return foam_case

    def load_sensitivities(
        self,
        case_dir: Path,
        foam_case,
        field_name: str = "pointSensVecadjS1ESI",
        objective_path: str = "optimisation/objective/0/dragadjS1",
        time_index: int = -1,
    ) -> tuple[np.ndarray, float]:
        """Read point sensitivities directly by index (no projection)."""
        from foamlib import FoamFieldFile

        import deepshapeopt.foam_utils as foam_utils

        if self._last_result is None:
            raise RuntimeError("build() must be called before load_sensitivities()")

        time_value = Path(str(foam_case[time_index])).name
        field_path = Path(case_dir) / time_value / field_name
        sens_all = np.asarray(FoamFieldFile(field_path).internal_field, dtype=np.float64)

        n_points = len(self._last_result.mesh.points)
        if sens_all.shape[0] != n_points:
            raise ValueError(
                f"Sensitivity field has {sens_all.shape[0]} points but the "
                f"written mesh has {n_points} - mesh/field mismatch."
            )
        sens = sens_all[self._last_result.surface_point_ids]
        J = foam_utils.read_objective(Path(case_dir), objective_path=objective_path)
        logger.debug(
            "Loaded %d wall point sensitivities directly (|sens| max %.3e)",
            len(sens), float(np.abs(sens).max()),
        )
        return sens, J

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
    build_static_castellation,
    build_static_octree,
    parse_refine_regions,
)
from .polymesh import PatchPlan, PolyMeshData, build_polymesh, face_pyramid_volumes
from .sdf_field import CompositeSDF, PhysicalSDF
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
    # --- internal flow ----------------------------------------------------
    # "external": fluid box around a solid object (drag); "internal": the
    # fluid is a channel through fixed geometry (extrusion die).
    "flow": "external",
    "geometry_stl": None,  # path; None -> model_setup.mesh_orig
    "fluid_side": "outside",  # "inside" for channels
    "cap_axis": "x",  # axis of the geometry's inlet/outlet cap planes
    "cap_tol": 1e-4,
    "seed_point": None,  # fluid point in the static region (required internal)
    "sign_probe_expect": None,  # None -> "solid" external / "fluid" internal
    # Boundary layout: domain faces -> patch names, plus the wall and the
    # design-surface (sensitivity) patch names.  None -> drag default.
    "patches": None,
    "refinement_regions": [],
    "band_max_level": None,  # design-surface band cap (default: max_level)
    "static_max_level": None,  # default: max_level
    "static_band_max_level": None,  # wall-band cap in the static region
    "static_refine_band_beta": None,  # default: refine_band_beta
    "write_scale": 1.0,  # point scale on write (0.001: mm config, m case)
    "outlet_interior": None,  # outletInterior carve (snappy-config schema)
}

_AXES = {"x": 0, "y": 1, "z": 2}


def resolve_sdf_hex_cfg(opt_cfg: dict, base_dir: Path) -> dict:
    """Resolve ``optimization.sdf_hex``: inline dict, or a JSON file path
    relative to the experiment directory (separate meshing config)."""
    import json

    raw = opt_cfg.get("sdf_hex", {})
    if isinstance(raw, str):
        path = Path(raw)
        if not path.is_absolute():
            path = Path(base_dir) / path
        with open(path) as f:
            raw = json.load(f)
        logger.info("Loaded sdf_hex mesh config from %s", path)
    return raw


class _HangingCorrector:
    """Applies hanging-node midpoint constraints (see
    :meth:`SdfHexMeshPipeline._hanging_constraints`).

    Works in the global point frame: lattice points stay bit-exact float64
    in :meth:`full_points`; only wall points and corrected midpoints move.
    """

    def __init__(self, mid_ids, parents, wall_ids, points0, n_passes, device):
        self.n_hanging = len(mid_ids)
        self.wall_ids = wall_ids
        self._mid_ids = mid_ids
        if self.n_hanging == 0:
            return
        self._parents = parents
        self.device = torch.device(device)
        self.n_passes = n_passes
        self._mid_t = torch.as_tensor(mid_ids, dtype=torch.long, device=device)
        self._p0_t = torch.as_tensor(parents[:, 0], dtype=torch.long, device=device)
        self._p1_t = torch.as_tensor(parents[:, 1], dtype=torch.long, device=device)
        self._wall_t = torch.as_tensor(wall_ids, dtype=torch.long, device=device)
        self._base = torch.as_tensor(points0, dtype=torch.float32, device=device)

    def _global(self, x_wall: torch.Tensor) -> torch.Tensor:
        xg = self._base.index_put((self._wall_t,), x_wall)
        for _ in range(self.n_passes):
            xg = xg.index_put(
                (self._mid_t,), 0.5 * (xg[self._p0_t] + xg[self._p1_t])
            )
        return xg

    def wall(self, x_wall: torch.Tensor) -> torch.Tensor:
        """Corrected wall-local positions (differentiable)."""
        if self.n_hanging == 0:
            return x_wall
        return self._global(x_wall)[self._wall_t]

    def full_points(self, points0: np.ndarray, x_wall: torch.Tensor) -> np.ndarray:
        """Global float64 point array: lattice points bit-exact from
        ``points0``, wall points from the snap, midpoints re-averaged in
        float64 (checkMesh's cell-openness tolerance is below float32
        round-off for large domains)."""
        pts = points0.copy()
        pts[self.wall_ids] = x_wall.detach().double().cpu().numpy()
        if self.n_hanging == 0:
            return pts
        mid, p0, p1 = self._mid_ids, self._parents[:, 0], self._parents[:, 1]
        for _ in range(self.n_passes):
            pts[mid] = 0.5 * (pts[p0] + pts[p1])
        return pts


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

    def patch_tris_local(self, name: str) -> torch.Tensor:
        """Fan-triangulated faces of one wall-type patch, with indices local
        to ``surface_point_ids`` (e.g. the sensitivity_region subset for
        penalties that must only see the design surface)."""
        faces = self.mesh.faces[self.mesh.patch_face_slice(name)]
        local = np.searchsorted(self.surface_point_ids, faces)
        if np.any(self.surface_point_ids[local] != faces):
            raise ValueError(f"Patch {name!r} is not a wall-type patch")
        tris = np.concatenate([local[:, [0, 1, 2]], local[:, [0, 2, 3]]], axis=0)
        return torch.as_tensor(tris, dtype=torch.long)


class SdfHexMeshPipeline:
    """Builds and writes the OpenFOAM polyMesh directly from the SDF."""

    def __init__(self, lattice_struct, model_setup, opt_cfg: dict, results_dir: Path):
        self.lattice_struct = lattice_struct
        self.model_setup = model_setup
        raw_cfg = opt_cfg.get("sdf_hex", {})
        if isinstance(raw_cfg, str):
            raise TypeError(
                "optimization.sdf_hex is a file path; resolve it with "
                "resolve_sdf_hex_cfg(opt_cfg, experiment_dir) before "
                "constructing the pipeline."
            )
        self.cfg = {**_DEFAULTS, **raw_cfg}
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

        self.flow = str(self.cfg["flow"])
        if self.flow not in ("external", "internal"):
            raise ValueError(f"sdf_hex.flow must be 'external' or 'internal', got {self.flow!r}")
        self.regions = parse_refine_regions(
            self.cfg["refinement_regions"], self.lattice, self.max_level
        )
        self.geom = None
        self.patch_plan: PatchPlan | None = None
        if self.flow == "internal":
            if self.cfg["seed_point"] is None:
                raise ValueError(
                    "sdf_hex.seed_point (a fluid point in the static region) "
                    "is required for internal flow"
                )
            self.geom = self._make_geometry()
            self.patch_plan = self._make_patch_plan()

        self._validate_margins()
        self.static_cells = self._load_or_build_static()

        self._sign_checked = False
        self._shell_hash: str | None = None
        self._last_hash: str | None = None
        self._last_hanging: tuple | None = None
        self._last_mesh: PolyMeshData | None = None
        self._last_castellation: Castellation | None = None
        self._last_result: HexMeshResult | None = None

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _make_geometry(self):
        from .trimesh_sdf import TriMeshSDF

        path = self.cfg["geometry_stl"]
        if path is not None:
            import trimesh

            mesh = trimesh.load(str(path), force="mesh")
        else:
            mesh = getattr(self.model_setup, "mesh_orig", None)
            if mesh is None:
                raise ValueError(
                    "sdf_hex.geometry_stl is not set and model_setup has no "
                    "mesh_orig to take the fixed geometry from"
                )
        return TriMeshSDF.from_trimesh(
            mesh,
            cap_axis=_AXES[str(self.cfg["cap_axis"])],
            cap_tol=float(self.cfg["cap_tol"]),
            fluid_side=str(self.cfg["fluid_side"]),
            device=self.device,
        )

    def _make_patch_plan(self) -> PatchPlan:
        pcfg = dict(self.cfg["patches"] or {})
        wall_name = pcfg.pop("wall", "walls")
        sens_name = pcfg.pop("sensitivity", None)
        domain_faces = {
            "x_min": "inlet", "x_max": "outlet",
            "y_min": "sides", "y_max": "sides",
            "z_min": "sides", "z_max": "sides",
        }
        unknown = set(pcfg) - set(domain_faces)
        if unknown:
            raise ValueError(f"sdf_hex.patches has unknown keys: {sorted(unknown)}")
        domain_faces.update(pcfg)

        subpatch_name = None
        subpatch_source = None
        face_subpatch = None
        oi = self.cfg["outlet_interior"]
        if oi and oi.get("enabled", True):
            from .patches import outlet_strip_classifier

            subpatch_source = str(oi.get("source_patch", "outlet"))
            subpatch_name = str(oi.get("patch_name", "outletInterior"))
            domain = np.asarray(self.cfg["domain"], dtype=np.float64)
            source_faces = [k for k, v in domain_faces.items() if v == subpatch_source]
            if len(source_faces) != 1:
                raise ValueError(
                    f"outlet_interior.source_patch {subpatch_source!r} must map "
                    f"to exactly one domain face, found {source_faces}"
                )
            axis = _AXES[source_faces[0][0]]
            side = 0 if source_faces[0].endswith("min") else 1
            debug_dir = (
                self.results_dir / "debug_outlet_interior"
                if oi.get("debug") else None
            )
            face_subpatch = outlet_strip_classifier(
                self.geom,
                plane_axis=axis,
                plane_value=float(domain[side, axis]),
                plane_tol=float(self.cfg["cap_tol"]),
                outlet_interior_cfg=oi,
                debug_dir=debug_dir,
            )

        dd = self.model_setup.design_domain.detach().cpu().numpy().astype(np.float64)
        return PatchPlan(
            domain_faces=domain_faces,
            wall_name=wall_name,
            wall_groups=(),
            sensitivity_name=sens_name,
            sensitivity_box=dd if sens_name is not None else None,
            subpatch_name=subpatch_name,
            subpatch_source=subpatch_source,
            face_subpatch=face_subpatch,
            drop_empty=True,
        )

    def _validate_margins(self) -> None:
        dd = self.model_setup.design_domain.detach().cpu().numpy().astype(np.float64)
        box = np.asarray(self.cfg["mesh_box"], dtype=np.float64)
        if np.any(dd[0] < box[0]) or np.any(dd[1] > box[1]):
            raise ValueError("design_domain must lie inside mesh_box")

        # Margins only matter toward interface faces; a box face coinciding
        # with the domain boundary has no static mesh behind it.
        ifc = self.mesh_box.interface_faces(self.lattice)
        ring = np.concatenate([(dd[0] - box[0])[~ifc[0]], (box[1] - dd[1])[~ifc[1]]])
        if len(ring):
            logger.info(
                "Mesh box faces on the domain boundary; fixed ring width(s): %s",
                np.round(ring, 4).tolist(),
            )
        margins = np.concatenate([(dd[0] - box[0])[ifc[0]], (box[1] - dd[1])[ifc[1]]])
        if len(margins) == 0:
            return
        margin = float(margins.min())
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
        parts = (
            self.cfg["domain"],
            self.cfg["base_cell_size"],
            self.cfg["mesh_box"],
            self.interface_level,
            self.max_level,
        )
        if self.flow == "internal":
            parts += (
                self.geom.content_hash(),
                self.cfg["seed_point"],
                self.cfg["refinement_regions"],
                self.cfg["static_max_level"],
                self.cfg["static_band_max_level"],
                self.cfg["static_refine_band_beta"],
            )
        key = hashlib.sha256(repr(parts).encode()).hexdigest()[:16]
        return self.results_dir / f"static_octree_{key}.npz"

    def _load_or_build_static(self) -> CellSet:
        path = self._static_cache_path()
        if path.exists():
            data = np.load(path)
            cells = CellSet(levels=data["levels"], anchors=data["anchors"])
            logger.info("Loaded static outer octree from %s (%d cells)", path, len(cells))
            return cells
        if self.flow == "internal":
            static_max = self.cfg["static_max_level"]
            band_beta = self.cfg["static_refine_band_beta"]
            cast = build_static_castellation(
                self.lattice,
                self.mesh_box,
                self.interface_level,
                self.max_level if static_max is None else int(static_max),
                self.geom.phi_np,
                beta=float(self.cfg["refine_band_beta"] if band_beta is None else band_beta),
                seed_point=self.cfg["seed_point"],
                regions=self.regions,
                band_max_level=self.cfg["static_band_max_level"],
            )
            cells = cast.cells
        else:
            cells = build_static_octree(self.lattice, self.mesh_box, self.interface_level)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, levels=cells.levels, anchors=cells.anchors)
        return cells

    def _make_sdf(self) -> PhysicalSDF:
        sdf = self.model_setup.frame.physical_sdf(
            self.lattice_struct,
            sign=-1.0 if self.cfg["fluid_side"] == "inside" else 1.0,
            device=self.device,
        )
        if self.cfg["check_sign"] and not self._sign_checked:
            probe = self.cfg["sign_probe_point"]
            if probe is None:
                dd = self.model_setup.design_domain.detach().cpu().numpy()
                probe = 0.5 * (dd[0] + dd[1])
            expect = self.cfg["sign_probe_expect"]
            if expect is None:
                expect = "fluid" if self.flow == "internal" else "solid"
            sdf.check_sign_convention(probe, expect=expect)
            self._sign_checked = True
        return sdf

    def _with_float32(self, fn):
        from deepshapeopt.reconstruction import with_float32_lattice

        return with_float32_lattice(self.lattice_struct, self.model_setup.frame.box_norm, lambda _b: fn())

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

    def _check_shell_frozen(self, cast: Castellation) -> None:
        """Assert the interface-shell fluid pattern is identical across
        builds (the precise meaning of the frozen interface when the margin
        ring contains wall geometry)."""
        shell_hash = hashlib.sha256(
            cast.cells.anchors[cast.shell_mask].tobytes()
            + cast.cells.levels[cast.shell_mask].tobytes()
        ).hexdigest()
        if self._shell_hash is None:
            self._shell_hash = shell_hash
        elif shell_hash != self._shell_hash:
            raise RuntimeError(
                "The interface-shell fluid pattern changed between builds: "
                "the shape reaches the mesh_box boundary. Enlarge the margin "
                "between design_domain and mesh_box, or check that the "
                "boundary control points are locked."
            )

    def _build_impl(self, reuse_castellation: bool) -> HexMeshResult:
        t0 = time.time()
        sdf = self._make_sdf()
        field = CompositeSDF(sdf, self.geom) if self.flow == "internal" else sdf

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
                field.phi_ext_np,
                beta=float(self.cfg["refine_band_beta"]),
                regions=self.regions,
                band_max_level=self.cfg["band_max_level"],
            )
            self._check_shell_frozen(cast)

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
            mesh = build_polymesh(self.lattice, cells, patch_plan=self.patch_plan)

        from .lattice import unpack_keys

        wall_ids = mesh.wall_point_ids()
        point_keys_xyz = unpack_keys(mesh.point_keys)
        points0 = self.lattice.point_coords(point_keys_xyz)
        x0 = points0[wall_ids]
        h_local = self._wall_h_local(mesh, wall_ids)

        # Wall points on a domain boundary plane (the inlet/outlet rim of
        # internal flows) may only slide within that plane.
        wall_keys = point_keys_xyz[wall_ids]
        lock_axes = (wall_keys == 0) | (wall_keys == self.lattice.fine_dims[None, :])
        if not np.any(lock_axes):
            lock_axes = None

        if reused and self._last_hanging is not None:
            corrector = self._last_hanging
        else:
            corrector = self._hanging_constraints(mesh, wall_ids, points0)
        self._last_hanging = corrector

        handle = snap_wall_points(
            field,
            x0,
            h_local,
            snap_iters=int(self.cfg["snap_iters"]),
            max_disp_frac=float(self.cfg["max_disp_frac"]),
            smooth_iters=int(self.cfg["smooth_iters"]),
            wall_edges=self._wall_edges_local(mesh, wall_ids),
            lock_axes=lock_axes,
        )
        final_points, n_relaxed = self._quality_guard(
            mesh, points0, wall_ids, handle, corrector
        )

        x_star = corrector.wall(handle.x_star())
        snapped_mesh = dataclasses.replace(mesh, points=final_points)

        wall_slice = mesh.wall_face_slice()
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
                "n_wall_faces": wall_slice.stop - wall_slice.start,
                "n_wall_points": len(wall_ids),
                "n_relaxed_points": n_relaxed,
                "n_hanging_points": corrector.n_hanging,
                "snap_residual_max": float(handle.residuals().max()),
                "patch_faces": {p.name: p.n_faces for p in mesh.patches},
                "build_seconds": time.time() - t0,
            },
        )
        if self.patch_plan is not None and self.patch_plan.subpatch_name is not None:
            n_sub = result.stats["patch_faces"].get(self.patch_plan.subpatch_name, 0)
            if n_sub == 0:
                logger.warning(
                    "Sub-patch %s received no faces; the outlet faces are "
                    "probably coarser than the interior region (add a "
                    "refinement_regions entry for the outlet plane).",
                    self.patch_plan.subpatch_name,
                )
        logger.info("Hex mesh build: %s", result.stats)

        self._last_hash = cast_hash
        self._last_mesh = mesh
        self._last_castellation = cast
        self._last_result = result
        return result

    def _hanging_constraints(
        self, mesh: PolyMeshData, wall_ids: np.ndarray, points0: np.ndarray
    ) -> "_HangingCorrector":
        """Midpoint constraints for hanging nodes near snapped walls.

        At a 2:1 transition the fine faces introduce midpoints on the edges
        of adjacent coarse faces.  Unsnapped, every midpoint lies exactly on
        the straight edge, so the coarse cell is geometrically closed; once
        snapping moves the midpoint or its edge endpoints, the midpoint
        leaves the (new) line and the cell opens (checkMesh "open cells").
        This happens wherever the wall crosses a level transition --
        internal-flow margin rings, never the drag case (its wall sits
        inside the uniform refinement band).

        Fix, like hexRef8's point constraints: every hanging midpoint is
        forced to the mean of its edge endpoints -- differentiably, so a
        wall midpoint's shape gradient flows through both parents.  This
        must cover midpoints that are not wall points themselves but whose
        edge endpoints are snapped wall points (internal faces hanging off
        a wall-face edge).
        """
        from .lattice import pack_keys, unpack_keys

        edges = np.concatenate(
            [mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]],
             mesh.faces[:, [2, 3]], mesh.faces[:, [3, 0]]],
            axis=0,
        )
        edges = np.unique(np.sort(edges, axis=1), axis=0)
        key_xyz = unpack_keys(mesh.point_keys)
        ksum = key_xyz[edges[:, 0]] + key_xyz[edges[:, 1]]
        even = np.all(ksum % 2 == 0, axis=1)  # finest (length-1) edges drop out
        edges = edges[even]
        mid_keys = pack_keys(ksum[even] // 2)

        pos = np.searchsorted(mesh.point_keys, mid_keys)
        pos = np.minimum(pos, len(mesh.point_keys) - 1)
        hit = mesh.point_keys[pos] == mid_keys
        mid_ids = pos[hit]
        parents = edges[hit]

        # 2:1 balance limits hanging chains to one level per face pair; a
        # few fixed-point passes resolve mid-of-mid dependencies.
        n_passes = max(2, self.max_level - self.interface_level + 1)
        return _HangingCorrector(
            mid_ids, parents, wall_ids, points0, n_passes, self.device
        )

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
        corrector: "_HangingCorrector",
    ) -> tuple[np.ndarray, int]:
        """Scale back snap displacements that would degenerate cells."""
        ref_owner, ref_neigh = face_pyramid_volumes(mesh, points0)
        frac = float(self.cfg["min_pyr_frac"])
        floor = 1e-13
        n_int = mesh.n_internal_faces
        relaxed_total = np.zeros(len(wall_ids), dtype=bool)

        for round_idx in range(int(self.cfg["quality_max_rounds"]) + 1):
            pts = corrector.full_points(points0, handle.x_star())
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

        pts = corrector.full_points(points0, handle.x_star())
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

        write_polymesh(
            self._last_result.mesh, case_dir, scale=float(self.cfg["write_scale"])
        )
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
        time_value: str | None = None,
    ) -> tuple[np.ndarray, float]:
        """Read point sensitivities directly by index (no projection).

        ``time_value`` overrides the time directory (the extrusion-die
        driver passes the per-solver adjoint time, since as1 and as2 write
        to different times); default is the case's ``time_index``-th time.
        """
        from foamlib import FoamFieldFile

        import deepshapeopt.foam_utils as foam_utils

        if self._last_result is None:
            raise RuntimeError("build() must be called before load_sensitivities()")

        if time_value is None:
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

"""Body-fitted hex mesh from the design SDF (the ``sdf_hex`` pipeline) for one run."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .hexmesh.foamwriter import checkmesh_log_ok, write_polymesh
from .hexmesh.pipeline import SdfHexMeshPipeline
from .hexmesh.polymesh import PolyMeshData

logger = logging.getLogger(__name__)


@dataclass
class MeshResult:
    verts: torch.Tensor          # [P, 3] snapped wall points, differentiable, config length unit
    faces: torch.Tensor          # [T, 3] wall triangles (local point ids)
    faces_design: torch.Tensor   # triangles of the design surface only
    polymesh: PolyMeshData
    wall_point_ids: np.ndarray   # indices of the wall points in the polyMesh
    snap_lambda: np.ndarray
    reused_connectivity: bool

    @property
    def n_points(self) -> int:
        return len(self.polymesh.points)

    @property
    def verts_design(self) -> torch.Tensor:
        return self.verts[torch.unique(self.faces_design)]


class SdfHexMesher:
    """Wraps :class:`SdfHexMeshPipeline` with the design SDF of a parametrization."""

    def __init__(self, design_sdf, frame, mesh_orig, sdf_hex_cfg: dict, *, flow: str,
                 unit_to_metre: float, results_dir: Path):
        cfg = dict(sdf_hex_cfg)
        cfg["flow"] = flow
        cfg["write_scale"] = float(unit_to_metre)
        self.unit_to_metre = float(unit_to_metre)
        self.flow = flow
        self.design_patch = (cfg.get("patches") or {}).get("sensitivity")
        setup = SimpleNamespace(frame=frame, design_domain=frame.design_domain, mesh_orig=mesh_orig)
        self.pipeline = SdfHexMeshPipeline(design_sdf, setup, {"sdf_hex": cfg}, results_dir)
        self.results_dir = Path(results_dir)
        self.checkmesh_ignore = [str(s).lower() for s in self.pipeline.cfg["checkmesh_ignore"]]
        self.last: MeshResult | None = None

    def build(self, reuse_castellation: bool = False) -> MeshResult:
        res = self.pipeline.build(reuse_castellation=reuse_castellation)
        verts = res.surface_points
        faces = res.wall_tris_local.to(verts.device)
        if self.flow == "internal" and self.design_patch:
            faces_design = res.patch_tris_local(self.design_patch).to(verts.device)
        else:
            faces_design = faces
        self.last = MeshResult(
            verts=verts, faces=faces, faces_design=faces_design, polymesh=res.mesh,
            wall_point_ids=res.surface_point_ids, snap_lambda=res.snap_lambda,
            reused_connectivity=res.reused_connectivity,
        )
        return self.last

    def write_polymesh(self, case_dir: Path) -> Path:
        if self.last is None:
            raise RuntimeError("build() must run before write_polymesh()")
        return write_polymesh(self.last.polymesh, Path(case_dir), scale=self.unit_to_metre)

    def check_mesh(self, case_dir: Path) -> None:
        """Gate on the checkMesh log of the case; archives a failed mesh."""
        log_path = Path(case_dir) / "log.checkMesh"
        if not log_path.exists():
            raise RuntimeError(f"checkMesh log not found: {log_path}")
        ok, failures = checkmesh_log_ok(log_path.read_text())
        if ok:
            return
        fatal = [f for f in failures if not any(p in f.lower() for p in self.checkmesh_ignore)]
        soft = [f for f in failures if f not in fatal]
        if soft:
            logger.warning("checkMesh quality warnings (ignored): %s", soft)
        if fatal or not failures:
            archive = self.results_dir / "failed_mesh"
            archive.mkdir(parents=True, exist_ok=True)
            shutil.copytree(Path(case_dir) / "constant" / "polyMesh", archive / "polyMesh", dirs_exist_ok=True)
            shutil.copy2(log_path, archive / "log.checkMesh")
            raise RuntimeError(f"checkMesh failed: {failures}; mesh archived to {archive}")

    def wall_displacement(self, verts_old: torch.Tensor) -> np.ndarray:
        return np.asarray(self.pipeline.wall_displacement(verts_old))

    def volume_centroid(self, no_grad: bool = False):
        return self.pipeline.volume_centroid(no_grad=no_grad)

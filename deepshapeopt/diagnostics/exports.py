"""Debug exports for ParaView: sensitivity VTPs, point clouds, control lattices."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
import torch


def save_mesh_with_sensitivities(verts, faces, sensitivities, out_path: Path):
    """Write a VTP with sensitivities as point data for ParaView checks."""
    verts_np = verts.detach().cpu().numpy()
    faces_np = faces.detach().cpu().numpy()
    n_faces = faces_np.shape[0]
    faces_pv = np.hstack(
        np.c_[np.full(n_faces, 3, dtype=np.int64), faces_np.astype(np.int64)]
    ).ravel()

    mesh_orig = pv.PolyData(verts_np, faces_pv)
    mesh_orig.point_data["sens"] = sensitivities
    mesh_orig.save(out_path)

def save_points_with_vectors(points, vectors, out_path: Path, scalars=None):
    """Write a point-cloud VTP carrying a vector field (+ optional scalars) for ParaView.

    Useful for glyphing per-face quantities: pass face centroids as ``points`` and the
    associated vectors (e.g. surface normals) as ``vectors``; in ParaView apply a Glyph
    filter oriented by the vector array. ``scalars`` is an optional ``{name: array}`` dict.
    """
    cloud = pv.PolyData(np.asarray(points, dtype=float))
    cloud["vector"] = np.asarray(vectors, dtype=float)
    for name, arr in (scalars or {}).items():
        cloud[name] = np.asarray(arr, dtype=float)
    cloud.save(out_path)


def export_control_net(spline_sp, out_dir: Path, locked_idx=None) -> None:
    """Knot grid, control lattice and design volume of a B-spline parametrization (paramspace)."""
    from DeepSDFStruct.export_knot_grid import (
        export_control_lattice_paramspace,
        export_design_volume_paramspace,
        export_knot_grid_paramspace,
    )

    out_dir = Path(out_dir)
    export_knot_grid_paramspace(spline_sp, out_dir / "knot_grid_paramspace.vtp")
    export_design_volume_paramspace(spline_sp, out_dir / "design_volume_paramspace.vts")
    if locked_idx is not None and len(locked_idx) == 0:
        locked_idx = None
    export_control_lattice_paramspace(
        spline_sp, out_dir / "control_lattice_paramspace.vtp", locked_idx=locked_idx,
    )


def export_wall_stl(verts: torch.Tensor, faces: torch.Tensor, path: Path) -> None:
    """Triangulated wall (or design) surface as STL."""
    import trimesh

    mesh = trimesh.Trimesh(
        vertices=verts.detach().cpu().numpy(), faces=faces.detach().cpu().numpy(), process=False,
    )
    mesh.remove_unreferenced_vertices()
    mesh.export(str(path))

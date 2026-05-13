"""
Reusable analysis utilities for reconstruction quality assessment.

Provides functions for computing and exporting mesh reconstruction errors
(vertex-wise SDF error between a reconstructed mesh and ground truth).
"""
import csv
from pathlib import Path

import numpy as np
import trimesh
import pyvista as pv
from matplotlib import cm, colors

from DeepSDFStruct.SDF import SDFfromMesh


def compute_bbox_normalization(mesh: trimesh.Trimesh):
    """Compute translation and scale so that the mesh fits into [-1, 1]^3."""
    bounds = mesh.bounds
    bbox_min = bounds[0]
    bbox_max = bounds[1]

    center = 0.5 * (bbox_min + bbox_max)
    extent = bbox_max - bbox_min
    max_extent = np.max(extent)

    if max_extent <= 0:
        raise ValueError("Mesh has invalid bounding box extent.")

    scale = 2.0 / max_extent
    return center, scale


def apply_normalization(
    mesh: trimesh.Trimesh,
    center: np.ndarray,
    scale: float,
) -> trimesh.Trimesh:
    """Apply normalization to a mesh: (x - center) * scale."""
    mesh_norm = mesh.copy()
    mesh_norm.vertices = (mesh_norm.vertices - center) * scale
    return mesh_norm


def compute_vertex_sdf_error(
    gt_mesh: trimesh.Trimesh,
    reconstructed_mesh: trimesh.Trimesh,
) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Normalize both meshes using GT bounding box and compute SDF error.

    Returns the normalized reconstructed mesh and the signed distance error
    at each of its vertices (queried against the GT mesh SDF).
    """
    norm_center, norm_scale = compute_bbox_normalization(gt_mesh)

    gt_mesh_norm = apply_normalization(gt_mesh, norm_center, norm_scale)
    reconstructed_mesh_norm = apply_normalization(
        reconstructed_mesh, norm_center, norm_scale
    )

    gt_sdf = SDFfromMesh(gt_mesh_norm, scale=False)

    vertices = reconstructed_mesh_norm.vertices
    sdf_values = gt_sdf.forward(vertices)
    sdf_error = np.asarray(sdf_values).reshape(-1)

    return reconstructed_mesh_norm, sdf_error


def trimesh_to_pyvista(mesh: trimesh.Trimesh) -> pv.PolyData:
    """Convert a trimesh mesh to a PyVista PolyData mesh."""
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)

    faces_pv = np.hstack(
        [np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]
    ).ravel()

    return pv.PolyData(vertices, faces_pv)


def add_vertex_colors_from_scalar(
    poly: pv.PolyData,
    scalar_name: str = "sdf_error",
    cmap_name: str = "viridis",
    log_scale: bool = False,
):
    """Create RGB vertex colors from a scalar array and store them in the mesh."""
    values = np.asarray(poly.point_data[scalar_name])

    if log_scale:
        positive_values = values[values > 0]
        eps = max(np.min(positive_values) if positive_values.size > 0 else 1e-12, 1e-12)
        vmin = max(values.min(), eps)
        vmax = max(values.max(), eps)
        if np.isclose(vmin, vmax):
            vmax = vmin + 1e-12
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        vmin = values.min()
        vmax = values.max()
        if np.isclose(vmin, vmax):
            vmax = vmin + 1e-12
        norm = colors.Normalize(vmin=vmin, vmax=vmax)

    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(norm(values))
    rgb_uint8 = (rgba[:, :3] * 255).astype(np.uint8)

    poly.point_data["RGB"] = rgb_uint8


def write_error_stats_csv(csv_path: Path, stats: dict):
    """Write error statistics to a CSV file."""
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in stats.items():
            writer.writerow([key, value])


def export_mesh_with_sdf_error(
    gt_mesh_path: str | Path,
    reconstructed_mesh_path: str | Path,
    output_dir: str | Path | None = None,
    add_rgb_colors: bool = True,
):
    """Load GT and reconstructed meshes, compute vertex SDF error, and export.

    Outputs a VTP mesh with sdf_error scalar field and a CSV with
    error statistics.

    Parameters
    ----------
    gt_mesh_path : path
        Path to ground truth mesh.
    reconstructed_mesh_path : path
        Path to reconstructed mesh.
    output_dir : path or None
        Output directory. If None, creates a subdirectory next to
        the reconstructed mesh.
    add_rgb_colors : bool
        Whether to add RGB vertex colors from the error field.
    """
    gt_mesh_path = Path(gt_mesh_path)
    reconstructed_mesh_path = Path(reconstructed_mesh_path)

    gt_mesh = trimesh.load_mesh(str(gt_mesh_path), force="mesh")
    reconstructed_mesh = trimesh.load_mesh(str(reconstructed_mesh_path), force="mesh")

    reconstructed_mesh_norm, sdf_error = compute_vertex_sdf_error(
        gt_mesh,
        reconstructed_mesh,
    )

    poly = trimesh_to_pyvista(reconstructed_mesh_norm)
    poly.point_data["sdf_error"] = sdf_error

    if add_rgb_colors:
        add_vertex_colors_from_scalar(poly, scalar_name="sdf_error", cmap_name="turbo")

    if output_dir is None:
        output_dir = reconstructed_mesh_path.parent / f"{reconstructed_mesh_path.stem}_sdf_error"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vtp_path = output_dir / f"{reconstructed_mesh_path.stem}_with_sdf_error.vtp"
    csv_path = output_dir / f"{reconstructed_mesh_path.stem}_error_stats.csv"

    poly.save(str(vtp_path))

    abs_error = np.abs(sdf_error)
    stats = {
        "num_vertices": int(reconstructed_mesh_norm.vertices.shape[0]),
        "min_signed_error": float(sdf_error.min()),
        "max_signed_error": float(sdf_error.max()),
        "mean_signed_error": float(sdf_error.mean()),
        "min_abs_error": float(abs_error.min()),
        "max_abs_error": float(abs_error.max()),
        "mean_abs_error": float(abs_error.mean()),
    }

    write_error_stats_csv(csv_path, stats)

    print(f"Saved VTP mesh to: {vtp_path}")
    print(f"Saved CSV stats to: {csv_path}")
    print(f"Number of vertices: {stats['num_vertices']}")
    print(f"Min signed sdf_error: {stats['min_signed_error']:.8e}")
    print(f"Max signed sdf_error: {stats['max_signed_error']:.8e}")
    print(f"Mean abs sdf_error:   {stats['mean_abs_error']:.8e}")

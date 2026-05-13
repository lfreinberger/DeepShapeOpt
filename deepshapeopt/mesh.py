from __future__ import annotations

import numpy as np
import trimesh
import torch
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List
from stl import mesh as stl_mesh
from DeepSDFStruct.optimization import tet_signed_vol


from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import trimesh
import numpy as np
import trimesh
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal


def compute_tet_mesh_volume_centroid(
    verts: torch.Tensor,
    tets: torch.Tensor,
    return_diagnostics: bool = False,
):
    """Compute volume and centroid from a tetrahedral mesh.

    Fixes inconsistent tet orientation by swapping vertices on
    negative-volume tets before summing.

    Parameters
    ----------
    verts : (n_verts, 3) torch.Tensor
        Vertex coordinates (can require grad).
    tets : (n_tet, 4) torch.Tensor
        Tet connectivity (integer indices).

    Returns
    -------
    total_vol : torch.Tensor
        Scalar volume.
    centroid : torch.Tensor
        (3,) centroid.
    diag : dict, optional
        Only when return_diagnostics=True.
    """
    # Fix orientation: swap vertices 1 and 2 on negative-volume tets
    perm = torch.tensor([0, 2, 1, 3], device=tets.device)
    tets = tets[:, perm]

    vols = tet_signed_vol(verts, tets)  # (n_tet,)

    #warning if some vols are negative including percentage
    n_negative = (vols < 0).sum().item()
    # if n_negative > 0:
    #     print(f"WARNING: {n_negative} / {len(vols)} tets have negative volume ({vols[vols<0].sum().item()/vols.sum().item()*100:.2f}%volume)!")


    total_vol = vols.sum()

    a = verts[tets[:, 0]]
    b = verts[tets[:, 1]]
    c = verts[tets[:, 2]]
    d = verts[tets[:, 3]]
    tet_centroids = (a + b + c + d) / 4.0
    centroid = (vols[:, None] * tet_centroids).sum(dim=0) / total_vol

    if return_diagnostics:
        diag = {
            "tets_oriented": tets,
            "vols": vols,
            "n_reoriented": n_negative,
        }
        return total_vol, centroid, diag

    return total_vol, centroid

def make_classifier(
    inlet_x: float,
    outlet_x: float,
    sensitivity_min: float,
    sensitivity_max: float,
) -> Callable[[np.ndarray], str]:

    def classify_face(centroid: np.ndarray) -> str:
        x, y, z = centroid

        if x > inlet_x:
            return "inlet"
        elif x < outlet_x:
            return "outlet"
        elif sensitivity_min < x < sensitivity_max:
            return "sensitivity_region"
        else:
            return "walls"

    return classify_face


def export_tet_signed_volume_vtu(
    verts: torch.Tensor,
    tets: torch.Tensor,
    vols_signed: torch.Tensor,
    out_path,
    export_only_negative: bool = True,
) -> None:
    """Export a tetrahedral VTU for ParaView diagnostics.

    Writes an UnstructuredGrid (tet cells) with cell-data arrays:
    - vol_signed: signed tet volume (negative indicates inverted / inconsistent orientation)
    - vol_abs: absolute tet volume
    - is_negative: 1 for vol_signed < 0 else 0

    Notes
    -----
    This is intended for visualization/diagnostics. Tensors are detached
    before writing so the autograd graph is not retained.
    """

    import pyvista as pv

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    verts_np = verts.detach().cpu().numpy()
    tets_np = tets.detach().cpu().numpy().astype(np.int64)
    vols_np = vols_signed.detach().cpu().numpy()

    if tets_np.ndim != 2 or tets_np.shape[1] != 4:
        raise ValueError(f"Expected tets of shape (n_tet, 4), got {tets_np.shape}")

    if export_only_negative:
        mask = vols_np < 0
        if not np.any(mask):
            return
        tets_np = tets_np[mask]
        vols_np = vols_np[mask]

    n_tet = int(tets_np.shape[0])
    cells = np.hstack([np.full((n_tet, 1), 4, dtype=np.int64), tets_np]).ravel()
    cell_types = np.full(n_tet, pv.CellType.TETRA, dtype=np.uint8)

    grid = pv.UnstructuredGrid(cells, cell_types, verts_np)
    grid.cell_data["vol_signed"] = vols_np
    grid.cell_data["vol_abs"] = np.abs(vols_np)
    grid.cell_data["is_negative"] = (vols_np < 0).astype(np.uint8)
    grid.save(out_path)


def split_stl_into_patches(
    input_stl,
    output_dir,
    classify_face: Callable[[np.ndarray], str],
    write_multi_region: bool = True,
    multi_region_name: str = "multi_region.stl",
    outlet_interior: dict[str, Any] | None = None,
) -> Dict[str, int]:
    """
    Split an STL into patches according to a user-provided classify_face function.

    Parameters
    ----------
    input_stl : str or Path
        Path to the input STL file.
    output_dir : str or Path
        Directory where per-patch STLs (and optionally the multi-region STL) are written.
    classify_face : callable
        Function centroid -> patch_name (str). Signature: (centroid: np.ndarray) -> str
    write_multi_region : bool, optional
        If True, also writes one ASCII STL with multiple 'solid <patch_name>' regions.
    multi_region_name : str, optional
        File name for the multi-region STL inside output_dir.

    Returns
    -------
    Dict[str, int]
        Mapping patch_name -> number of faces in that patch.
    """

    input_stl = Path(input_stl)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {input_stl} ...")
    original_mesh = stl_mesh.Mesh.from_file(input_stl.as_posix())

    # Triangle vertices: shape (n_triangles, 3, 3)
    triangles = original_mesh.vectors  # (n, 3, 3)

    # Group triangles by patch
    patch_faces: Dict[str, List[np.ndarray]] = {}

    for tri_idx, tri in enumerate(triangles):
        # centroid = mean of 3 vertices
        centroid = tri.mean(axis=0)
        patch_name = classify_face(centroid)

        if patch_name not in patch_faces:
            patch_faces[patch_name] = []
        patch_faces[patch_name].append(tri)

    # Remove empty entries if any
    patch_faces = {
        name: tris for name, tris in patch_faces.items() if len(tris) > 0
    }
    debug_patch_names: list[str] = []

    if outlet_interior and outlet_interior.get("enabled", False):
        source_patch = outlet_interior.get("source_patch", "outlet")
        interior_patch = outlet_interior.get("patch_name", "outletInterior")
        debug_patch_names = [source_patch, interior_patch]
        if source_patch not in patch_faces:
            raise ValueError(f"Cannot create outlet interior patch: missing source patch '{source_patch}'.")

        method = outlet_interior.get("method", "local_inset")
        if method == "local_inset":
            interior_regions = _build_local_inset_outlet_regions(
                patch_faces[source_patch],
                source_patch_name=source_patch,
                interior_patch_name=interior_patch,
                inset_distance=float(outlet_interior.get("inset_distance", 0.10)),
            )
        elif method == "centerline_band":
            interior_regions = _build_centerline_band_outlet_regions(
                patch_faces[source_patch],
                source_patch_name=source_patch,
                interior_patch_name=interior_patch,
                exclusion_fraction=float(outlet_interior.get("inset_fraction", 0.20)),
                station_count=int(outlet_interior.get("station_count", 240)),
            )
        elif method == "local_distance":
            interior_regions = _build_local_distance_outlet_regions(
                patch_faces[source_patch],
                source_patch_name=source_patch,
                interior_patch_name=interior_patch,
                threshold=float(outlet_interior.get("inset_fraction", 0.10)),
                grid_resolution=int(outlet_interior.get("grid_resolution", 180)),
                ray_directions=int(outlet_interior.get("ray_directions", 16)),
            )
        else:
            raise ValueError(f"Unknown outlet_interior method: {method}")
        patch_faces[source_patch] = interior_regions[source_patch]
        patch_faces[interior_patch] = interior_regions[interior_patch]
        print(
            "Outlet interior patch split stats: "
            f"{source_patch}={len(patch_faces[source_patch])} triangles, "
            f"{interior_patch}={len(patch_faces[interior_patch])} triangles"
        )
        if outlet_interior.get("debug", True):
            _write_patch_debug_stls(output_dir, patch_faces, debug_patch_names)

    # Optionally: write a single multi-region STL (ASCII)
    if write_multi_region:
        multi_path = output_dir / multi_region_name
        with open(multi_path, "w") as f:
            for patch_name, tris in patch_faces.items():
                tris_array = np.array(tris)
                patch_mesh = stl_mesh.Mesh(
                    np.zeros(tris_array.shape[0], dtype=stl_mesh.Mesh.dtype)
                )
                patch_mesh.vectors[:] = tris_array

                f.write(f"solid {patch_name}\n")
                for i in range(len(patch_mesh.vectors)):
                    n = patch_mesh.normals[i]
                    v = patch_mesh.vectors[i]

                    f.write(f"  facet normal {n[0]} {n[1]} {n[2]}\n")
                    f.write("    outer loop\n")
                    f.write(f"      vertex {v[0,0]} {v[0,1]} {v[0,2]}\n")
                    f.write(f"      vertex {v[1,0]} {v[1,1]} {v[1,2]}\n")
                    f.write(f"      vertex {v[2,0]} {v[2,1]} {v[2,2]}\n")
                    f.write("    endloop\n")
                    f.write("  endfacet\n")
                f.write(f"endsolid {patch_name}\n")

        print(f"Saved multi-region STL to {multi_path}")

    # Return stats in case you want to log/assert in tests
    return {name: len(tris) for name, tris in patch_faces.items()}


# ---------------------------------------------------------------------------
# Mesh stitching at box boundaries
# ---------------------------------------------------------------------------

def _triangle_normal(tri: np.ndarray) -> np.ndarray:
    normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    norm = np.linalg.norm(normal)
    if norm < 1e-30:
        return np.zeros(3)
    return normal / norm


def _write_ascii_stl(path: Path, solid_name: str, triangles: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(f"solid {solid_name}\n")
        for tri in triangles:
            tri = np.asarray(tri, dtype=float)
            n = _triangle_normal(tri)
            f.write(f"  facet normal {n[0]} {n[1]} {n[2]}\n")
            f.write("    outer loop\n")
            for v in tri:
                f.write(f"      vertex {v[0]} {v[1]} {v[2]}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write(f"endsolid {solid_name}\n")


def _write_patch_debug_stls(
    output_dir: Path,
    patch_faces: dict[str, list[np.ndarray]],
    patch_names: list[str],
) -> None:
    debug_dir = output_dir / "debug_outlet_interior"
    debug_dir.mkdir(parents=True, exist_ok=True)

    for patch_name in patch_names:
        if patch_name not in patch_faces:
            continue
        path = debug_dir / f"{patch_name}.stl"
        _write_ascii_stl(path, patch_name, patch_faces[patch_name])
        print(f"Saved outlet debug STL: {path}")

    combined_path = debug_dir / "outlet_split_debug.stl"
    with open(combined_path, "w") as f:
        for patch_name in patch_names:
            tris = patch_faces.get(patch_name)
            if not tris:
                continue
            f.write(f"solid {patch_name}\n")
            for tri in tris:
                tri = np.asarray(tri, dtype=float)
                n = _triangle_normal(tri)
                f.write(f"  facet normal {n[0]} {n[1]} {n[2]}\n")
                f.write("    outer loop\n")
                for v in tri:
                    f.write(f"      vertex {v[0]} {v[1]} {v[2]}\n")
                f.write("    endloop\n")
                f.write("  endfacet\n")
            f.write(f"endsolid {patch_name}\n")
    print(f"Saved outlet split debug STL: {combined_path}")


def _polygon_area_2d(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _points_in_polygon_2d(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    x0 = polygon[:, 0]
    y0 = polygon[:, 1]
    x1 = np.roll(x0, -1)
    y1 = np.roll(y0, -1)
    for ax, ay, bx, by in zip(x0, y0, x1, y1):
        crosses = ((ay > y) != (by > y)) & (
            x < (bx - ax) * (y - ay) / (by - ay + 1e-300) + ax
        )
        inside ^= crosses
    return inside


def _distance_to_polygon_segments_2d(points: np.ndarray, polygon: np.ndarray, chunk_size: int = 4096) -> np.ndarray:
    seg_a = polygon
    seg_b = np.roll(polygon, -1, axis=0)
    seg = seg_b - seg_a
    seg_len2 = np.einsum("ij,ij->i", seg, seg)
    seg_len2 = np.where(seg_len2 < 1e-30, 1.0, seg_len2)

    out = np.empty(len(points), dtype=float)
    for start in range(0, len(points), chunk_size):
        p = points[start:start + chunk_size]
        ap = p[:, None, :] - seg_a[None, :, :]
        t = np.clip(np.einsum("nsi,si->ns", ap, seg) / seg_len2[None, :], 0.0, 1.0)
        closest = seg_a[None, :, :] + t[:, :, None] * seg[None, :, :]
        dist2 = np.sum((p[:, None, :] - closest) ** 2, axis=2)
        out[start:start + chunk_size] = np.sqrt(dist2.min(axis=1))
    return out


def _ray_polygon_distance_2d(point: np.ndarray, direction: np.ndarray, polygon: np.ndarray) -> float:
    seg_a = polygon
    seg_b = np.roll(polygon, -1, axis=0)
    seg = seg_b - seg_a
    cross = direction[0] * seg[:, 1] - direction[1] * seg[:, 0]
    valid = np.abs(cross) > 1e-14
    if not np.any(valid):
        return np.inf

    delta = seg_a - point
    t = np.full(len(seg), np.inf, dtype=float)
    u = np.full(len(seg), np.inf, dtype=float)
    t[valid] = (
        delta[valid, 0] * seg[valid, 1] - delta[valid, 1] * seg[valid, 0]
    ) / cross[valid]
    u[valid] = (
        delta[valid, 0] * direction[1] - delta[valid, 1] * direction[0]
    ) / cross[valid]
    hits = (t > 1e-12) & (u >= -1e-12) & (u <= 1.0 + 1e-12)
    if not np.any(hits):
        return np.inf
    return float(np.min(t[hits]))


def _local_half_width_2d(points: np.ndarray, polygon: np.ndarray, n_directions: int) -> np.ndarray:
    angles = np.linspace(0.0, np.pi, n_directions, endpoint=False)
    directions = np.column_stack([np.cos(angles), np.sin(angles)])
    half_width = np.empty(len(points), dtype=float)

    for i, p in enumerate(points):
        widths = []
        for direction in directions:
            d_pos = _ray_polygon_distance_2d(p, direction, polygon)
            d_neg = _ray_polygon_distance_2d(p, -direction, polygon)
            if np.isfinite(d_pos) and np.isfinite(d_neg):
                widths.append(0.5 * (d_pos + d_neg))
        half_width[i] = min(widths) if widths else np.nan
    return half_width


def _order_boundary_loop_from_triangles(triangles: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    vertex_ids: dict[tuple[int, int, int], int] = {}
    vertices: list[np.ndarray] = []

    def vid(point: np.ndarray) -> int:
        key = tuple(np.round(point / tol).astype(np.int64))
        if key not in vertex_ids:
            vertex_ids[key] = len(vertices)
            vertices.append(np.asarray(point, dtype=float))
        return vertex_ids[key]

    edge_counts: dict[tuple[int, int], int] = defaultdict(int)
    for tri in triangles:
        ids = [vid(p) for p in tri]
        for a, b in ((ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0])):
            edge_counts[tuple(sorted((a, b)))] += 1

    adjacency: dict[int, list[int]] = defaultdict(list)
    for (a, b), count in edge_counts.items():
        if count == 1:
            adjacency[a].append(b)
            adjacency[b].append(a)

    if not adjacency:
        raise ValueError("Outlet patch has no boundary edges.")

    loops: list[list[int]] = []
    visited_edges: set[tuple[int, int]] = set()
    for start in adjacency:
        for first_next in adjacency[start]:
            edge = tuple(sorted((start, first_next)))
            if edge in visited_edges:
                continue
            loop = [start]
            prev, curr = start, first_next
            while True:
                visited_edges.add(tuple(sorted((prev, curr))))
                loop.append(curr)
                candidates = [n for n in adjacency[curr] if n != prev]
                if not candidates:
                    break
                next_id = candidates[0]
                if next_id == start:
                    visited_edges.add(tuple(sorted((curr, next_id))))
                    break
                prev, curr = curr, next_id
                if len(loop) > len(adjacency) + 2:
                    raise ValueError("Could not order outlet boundary loop.")
            if len(loop) >= 3:
                loops.append(loop)

    if not loops:
        raise ValueError("Could not find a closed outlet boundary loop.")

    verts = np.asarray(vertices)
    return verts[max(loops, key=len)]


def _make_plane_basis(points: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    normal = np.zeros(3, dtype=float)
    for tri in triangles:
        normal += np.cross(tri[1] - tri[0], tri[2] - tri[0])
    n_norm = np.linalg.norm(normal)
    if n_norm < 1e-14:
        centered = points - points.mean(axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal = vh[-1]
        n_norm = np.linalg.norm(normal)
    normal = normal / max(n_norm, 1e-30)

    origin = points.mean(axis=0)
    axis_u = points[1] - points[0]
    axis_u = axis_u - np.dot(axis_u, normal) * normal
    if np.linalg.norm(axis_u) < 1e-14:
        axis_u = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(axis_u, normal)) > 0.9:
            axis_u = np.array([0.0, 1.0, 0.0])
        axis_u = axis_u - np.dot(axis_u, normal) * normal
    axis_u = axis_u / np.linalg.norm(axis_u)
    axis_v = np.cross(normal, axis_u)
    axis_v = axis_v / np.linalg.norm(axis_v)
    return origin, normal, axis_u, axis_v


def _project_to_plane(points: np.ndarray, origin: np.ndarray, axis_u: np.ndarray, axis_v: np.ndarray) -> np.ndarray:
    rel = points - origin
    return np.column_stack([rel @ axis_u, rel @ axis_v])


def _unproject_from_plane(points_2d: np.ndarray, origin: np.ndarray, axis_u: np.ndarray, axis_v: np.ndarray) -> np.ndarray:
    return origin + points_2d[:, 0:1] * axis_u + points_2d[:, 1:2] * axis_v


def _orient_triangle(tri: np.ndarray, normal: np.ndarray) -> np.ndarray:
    tri_normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    if np.dot(tri_normal, normal) < 0:
        return tri[[0, 2, 1]]
    return tri


def _clip_scalar_polygon(
    points: list[np.ndarray],
    values: list[float],
    keep_positive: bool,
) -> tuple[list[np.ndarray], list[float]]:
    if not points:
        return [], []

    out_points: list[np.ndarray] = []
    out_values: list[float] = []

    def keep(value: float) -> bool:
        return value >= 0.0 if keep_positive else value <= 0.0

    for i, curr_point in enumerate(points):
        prev_point = points[i - 1]
        curr_value = values[i]
        prev_value = values[i - 1]
        curr_keep = keep(curr_value)
        prev_keep = keep(prev_value)

        if curr_keep != prev_keep:
            denom = prev_value - curr_value
            t = 0.5 if abs(denom) < 1e-30 else prev_value / denom
            t = float(np.clip(t, 0.0, 1.0))
            out_points.append(prev_point + t * (curr_point - prev_point))
            out_values.append(0.0)
        if curr_keep:
            out_points.append(curr_point)
            out_values.append(curr_value)

    return out_points, out_values


def _triangulate_planar_polygon(points_3d: list[np.ndarray], normal: np.ndarray) -> list[np.ndarray]:
    if len(points_3d) < 3:
        return []
    p0 = points_3d[0]
    triangles = []
    for i in range(1, len(points_3d) - 1):
        tri = np.array([p0, points_3d[i], points_3d[i + 1]])
        triangles.append(_orient_triangle(tri, normal))
    return triangles


def _triangulate_polygon_2d(points: np.ndarray) -> np.ndarray:
    order = list(range(len(points)))
    if _polygon_area_2d(points) < 0:
        order = list(reversed(order))
        points = points[order]

    def cross2(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        ab = b - a
        ac = c - a
        return float(ab[0] * ac[1] - ab[1] * ac[0])

    def point_in_triangle(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
        v0 = c - a
        v1 = b - a
        v2 = p - a
        den = v0[0] * v1[1] - v1[0] * v0[1]
        if abs(den) < 1e-20:
            return False
        u = (v2[0] * v1[1] - v1[0] * v2[1]) / den
        v = (v0[0] * v2[1] - v2[0] * v0[1]) / den
        return u > 1e-12 and v > 1e-12 and (u + v) < 1.0 - 1e-12

    remaining = list(range(len(points)))
    triangles: list[list[int]] = []
    guard = 0
    while len(remaining) > 3:
        clipped = False
        for pos, idx in enumerate(remaining):
            prev_idx = remaining[(pos - 1) % len(remaining)]
            next_idx = remaining[(pos + 1) % len(remaining)]
            a, b, c = points[prev_idx], points[idx], points[next_idx]
            if cross2(a, b, c) <= 1e-14:
                continue
            if any(
                point_in_triangle(points[other], a, b, c)
                for other in remaining
                if other not in (prev_idx, idx, next_idx)
            ):
                continue
            triangles.append([order[prev_idx], order[idx], order[next_idx]])
            del remaining[pos]
            clipped = True
            break
        guard += 1
        if not clipped or guard > len(points) * len(points):
            raise ValueError("Ear clipping failed to triangulate inset outlet polygon.")
    triangles.append([order[idx] for idx in remaining])
    return np.asarray(triangles, dtype=np.int64)


def _inward_vertex_directions_2d(polygon: np.ndarray) -> np.ndarray:
    if _polygon_area_2d(polygon) < 0:
        polygon = polygon[::-1]
    dirs = []
    centroid = polygon.mean(axis=0)
    for i in range(len(polygon)):
        prev_p = polygon[(i - 1) % len(polygon)]
        curr_p = polygon[i]
        next_p = polygon[(i + 1) % len(polygon)]
        prev_dir = curr_p - prev_p
        next_dir = next_p - curr_p
        prev_dir = prev_dir / max(np.linalg.norm(prev_dir), 1e-30)
        next_dir = next_dir / max(np.linalg.norm(next_dir), 1e-30)
        n_prev = np.array([-prev_dir[1], prev_dir[0]])
        n_next = np.array([-next_dir[1], next_dir[0]])
        candidates = [
            centroid - curr_p,
            n_prev + n_next,
            n_prev,
            n_next,
            0.5 * (prev_p + next_p) - curr_p,
        ]
        scale = max(np.ptp(polygon[:, 0]), np.ptp(polygon[:, 1]), 1.0)
        direction = None
        for candidate in candidates:
            if np.linalg.norm(candidate) < 1e-14:
                continue
            cand = candidate / np.linalg.norm(candidate)
            if _points_in_polygon_2d((curr_p + 1e-6 * scale * cand)[None, :], polygon)[0]:
                direction = cand
                break
        if direction is None:
            best_dir = None
            best_dist = -np.inf
            for angle in np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False):
                cand = np.array([np.cos(angle), np.sin(angle)])
                if not _points_in_polygon_2d((curr_p + 1e-6 * scale * cand)[None, :], polygon)[0]:
                    continue
                dist = _ray_polygon_distance_2d(curr_p + 1e-6 * scale * cand, cand, polygon)
                if np.isfinite(dist) and dist > best_dist:
                    best_dist = dist
                    best_dir = cand
            direction = best_dir
        if direction is None:
            direction = candidates[0] / max(np.linalg.norm(candidates[0]), 1e-30)
        dirs.append(direction)
    return np.asarray(dirs)


def _smooth_periodic(values: np.ndarray, passes: int = 2) -> np.ndarray:
    out = values.astype(float).copy()
    for _ in range(passes):
        padded = np.r_[out[-2:], out, out[:2]]
        out = np.array([
            np.median(padded[i:i + 5])
            for i in range(len(out))
        ])
    return out


def _resample_closed_polyline(points: np.ndarray, n_samples: int) -> np.ndarray:
    edges = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(edges, axis=1)
    perimeter = float(lengths.sum())
    if perimeter <= 1e-30:
        raise ValueError("Cannot resample degenerate outlet boundary.")

    cumulative = np.r_[0.0, np.cumsum(lengths)]
    samples = np.linspace(0.0, perimeter, int(n_samples), endpoint=False)
    out = np.empty((len(samples), points.shape[1]), dtype=float)
    edge_idx = np.searchsorted(cumulative, samples, side="right") - 1
    edge_idx = np.clip(edge_idx, 0, len(points) - 1)
    local = samples - cumulative[edge_idx]
    t = local / np.maximum(lengths[edge_idx], 1e-30)
    out = points[edge_idx] + t[:, None] * edges[edge_idx]
    return out


def _smooth_closed_points(points: np.ndarray, passes: int, keep_inside: np.ndarray | None = None) -> np.ndarray:
    out = points.copy()
    for _ in range(passes):
        candidate = (
            0.25 * np.roll(out, 1, axis=0)
            + 0.5 * out
            + 0.25 * np.roll(out, -1, axis=0)
        )
        if keep_inside is not None:
            inside = _points_in_polygon_2d(candidate, keep_inside)
            candidate = np.where(inside[:, None], candidate, out)
        out = candidate
    return out


def _build_local_inset_outlet_regions(
    outlet_triangles: list[np.ndarray],
    source_patch_name: str,
    interior_patch_name: str,
    inset_distance: float,
) -> dict[str, list[np.ndarray]]:
    triangles = np.asarray(outlet_triangles, dtype=float)
    outer_3d = _order_boundary_loop_from_triangles(triangles)
    origin, normal, axis_u, axis_v = _make_plane_basis(outer_3d, triangles)
    outer_2d = _project_to_plane(outer_3d, origin, axis_u, axis_v)

    if _polygon_area_2d(outer_2d) < 0:
        outer_2d = outer_2d[::-1]
        outer_3d = outer_3d[::-1]

    sample_count = max(160, min(800, len(outer_2d) * 8))
    outer_sample_2d = _resample_closed_polyline(outer_2d, sample_count)
    outer_sample_3d = _unproject_from_plane(outer_sample_2d, origin, axis_u, axis_v)

    extent = max(float(np.ptp(outer_sample_2d[:, 0])), float(np.ptp(outer_sample_2d[:, 1])))
    eps = max(extent * 1e-8, 1e-12)
    directions = _inward_vertex_directions_2d(outer_sample_2d)
    inset_distance = max(float(inset_distance), eps)
    outer_area = abs(_polygon_area_2d(outer_2d))
    last_error: Exception | None = None
    for scale in (1.0, 0.75, 0.5, 0.35, 0.25, 0.15, 0.1):
        margins = np.full(len(outer_sample_2d), scale * inset_distance, dtype=float)
        inner_2d = outer_sample_2d + margins[:, None] * directions

        try:
            for i in range(len(inner_2d)):
                if _points_in_polygon_2d(inner_2d[i:i + 1], outer_2d)[0]:
                    continue
                margin = margins[i]
                while margin > eps:
                    margin *= 0.5
                    candidate = outer_sample_2d[i] + margin * directions[i]
                    if _points_in_polygon_2d(candidate[None, :], outer_2d)[0]:
                        inner_2d[i] = candidate
                        margins[i] = margin
                        break
                else:
                    inner_2d[i] = outer_sample_2d[i] + eps * directions[i]
                    margins[i] = eps

            inner_2d = _smooth_closed_points(inner_2d, passes=4, keep_inside=outer_2d)
            inner_area = abs(_polygon_area_2d(inner_2d))
            if inner_area <= 1e-20 or inner_area >= outer_area:
                raise ValueError("Local inset outlet polygon has invalid area.")
            inner_tri_idx = _triangulate_polygon_2d(inner_2d)
            break
        except Exception as exc:
            last_error = exc
    else:
        raise ValueError(f"Could not create valid local inset outlet polygon: {last_error}")

    inner_3d = _unproject_from_plane(inner_2d, origin, axis_u, axis_v)
    interior_tris = [
        _orient_triangle(inner_3d[idx], normal)
        for idx in inner_tri_idx
    ]

    rim_tris: list[np.ndarray] = []
    for i in range(len(outer_sample_3d)):
        j = (i + 1) % len(outer_sample_3d)
        rim_tris.append(_orient_triangle(np.array([outer_sample_3d[i], outer_sample_3d[j], inner_3d[j]]), normal))
        rim_tris.append(_orient_triangle(np.array([outer_sample_3d[i], inner_3d[j], inner_3d[i]]), normal))

    print(
        f"Created {interior_patch_name} with local inset: "
        f"inset_distance={inset_distance:.6e}, applied_scale={scale:.2f}, "
        f"margin_range=[{margins.min():.6e}, {margins.max():.6e}], "
        f"samples={sample_count}, "
        f"outer_area={outer_area:.6e}, inner_area={inner_area:.6e}, "
        f"interior_tris={len(interior_tris)}, {source_patch_name}_rim_tris={len(rim_tris)}"
    )
    return {
        source_patch_name: rim_tris,
        interior_patch_name: interior_tris,
    }


def _build_local_distance_outlet_regions(
    outlet_triangles: list[np.ndarray],
    source_patch_name: str,
    interior_patch_name: str,
    threshold: float,
    grid_resolution: int = 180,
    ray_directions: int = 16,
) -> dict[str, list[np.ndarray]]:
    triangles = np.asarray(outlet_triangles, dtype=float)
    outer_3d = _order_boundary_loop_from_triangles(triangles)
    origin, normal, axis_u, axis_v = _make_plane_basis(outer_3d, triangles)
    outer_2d = _project_to_plane(outer_3d, origin, axis_u, axis_v)

    if _polygon_area_2d(outer_2d) < 0:
        outer_2d = outer_2d[::-1]

    bbox_min = outer_2d.min(axis=0)
    bbox_max = outer_2d.max(axis=0)
    span = bbox_max - bbox_min
    max_span = float(np.max(span))
    if max_span <= 0.0:
        raise ValueError("Outlet boundary has invalid 2D extent.")

    cell_size = max_span / int(grid_resolution)
    nx = max(1, int(np.ceil(span[0] / cell_size)))
    ny = max(1, int(np.ceil(span[1] / cell_size)))
    x0 = bbox_min[0]
    y0 = bbox_min[1]

    xs = x0 + (np.arange(nx) + 0.5) * cell_size
    ys = y0 + (np.arange(ny) + 0.5) * cell_size
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    centers = np.column_stack([xx.ravel(), yy.ravel()])
    inside = _points_in_polygon_2d(centers, outer_2d)
    inside_flat_ids = np.flatnonzero(inside)
    if len(inside_flat_ids) == 0:
        raise ValueError("Rasterized outlet has no cells inside the boundary.")

    grid_x = x0 + np.arange(nx + 1) * cell_size
    grid_y = y0 + np.arange(ny + 1) * cell_size
    gx, gy = np.meshgrid(grid_x, grid_y, indexing="ij")
    grid_points = np.column_stack([gx.ravel(), gy.ravel()])
    grid_inside = _points_in_polygon_2d(grid_points, outer_2d)
    inside_points = grid_points[grid_inside]
    if len(inside_points) == 0:
        raise ValueError("Rasterized outlet has no grid vertices inside the boundary.")

    d_wall = _distance_to_polygon_segments_2d(inside_points, outer_2d)
    local_half_width = _local_half_width_2d(
        inside_points,
        outer_2d,
        n_directions=int(ray_directions),
    )
    finite = np.isfinite(local_half_width) & (local_half_width > 0.0)
    fallback = np.nanmedian(local_half_width[finite]) if np.any(finite) else np.nan
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = max(float(np.nanmax(d_wall)), cell_size)
    local_half_width = np.where(finite, local_half_width, fallback)

    normalized_grid = np.full(len(grid_points), -np.inf, dtype=float)
    normalized_grid[grid_inside] = d_wall / np.maximum(local_half_width, 1e-30)
    phi_grid = normalized_grid.reshape((nx + 1, ny + 1)) - float(threshold)

    def point_3d(i: int, j: int) -> np.ndarray:
        p2 = np.array([x0 + i * cell_size, y0 + j * cell_size])
        return _unproject_from_plane(p2[None, :], origin, axis_u, axis_v)[0]

    source_tris: list[np.ndarray] = []
    interior_tris: list[np.ndarray] = []
    interior_cells = 0
    rim_cells = 0
    mixed_cells = 0
    for flat_id in inside_flat_ids:
        i = flat_id // ny
        j = flat_id % ny
        cell_points = [
            point_3d(i, j),
            point_3d(i + 1, j),
            point_3d(i + 1, j + 1),
            point_3d(i, j + 1),
        ]
        cell_phi = [
            float(phi_grid[i, j]),
            float(phi_grid[i + 1, j]),
            float(phi_grid[i + 1, j + 1]),
            float(phi_grid[i, j + 1]),
        ]

        interior_poly, _ = _clip_scalar_polygon(cell_points, cell_phi, keep_positive=True)
        rim_poly, _ = _clip_scalar_polygon(cell_points, cell_phi, keep_positive=False)

        if len(interior_poly) >= 3:
            interior_cells += 1
            interior_tris.extend(_triangulate_planar_polygon(interior_poly, normal))
        if len(rim_poly) >= 3:
            rim_cells += 1
            source_tris.extend(_triangulate_planar_polygon(rim_poly, normal))
        if len(interior_poly) >= 3 and len(rim_poly) >= 3:
            mixed_cells += 1

    if not interior_tris:
        raise ValueError(
            f"No outletInterior triangles for normalized threshold {threshold}. "
            "Lower outlet_interior.inset_fraction or increase grid_resolution."
        )
    if not source_tris:
        raise ValueError(
            f"No outlet rim triangles for normalized threshold {threshold}. "
            "Increase outlet_interior.inset_fraction."
        )

    print(
        f"Created {interior_patch_name} with local distance split: "
        f"threshold={threshold:.3f}, grid={nx}x{ny}, cell_size={cell_size:.6e}, "
        f"{interior_patch_name}_cells={interior_cells}, "
        f"{source_patch_name}_rim_cells={rim_cells}, mixed_cells={mixed_cells}"
    )

    return {
        source_patch_name: source_tris,
        interior_patch_name: interior_tris,
    }


def _scanline_intervals_2d(polygon: np.ndarray, axis: int, value: float) -> list[tuple[float, float]]:
    cross_axis = 1 - axis
    coords = []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        av = a[axis]
        bv = b[axis]
        if abs(av - bv) < 1e-14:
            continue
        if (av <= value < bv) or (bv <= value < av):
            t = (value - av) / (bv - av)
            coords.append(float(a[cross_axis] + t * (b[cross_axis] - a[cross_axis])))

    coords = sorted(coords)
    intervals = []
    for i in range(0, len(coords) - 1, 2):
        lo, hi = coords[i], coords[i + 1]
        if hi - lo > 1e-12:
            intervals.append((lo, hi))
    return intervals


def _station_point_3d(
    station: float,
    transverse: float,
    axis: int,
    origin: np.ndarray,
    axis_u: np.ndarray,
    axis_v: np.ndarray,
) -> np.ndarray:
    p2 = np.zeros(2)
    p2[axis] = station
    p2[1 - axis] = transverse
    return _unproject_from_plane(p2[None, :], origin, axis_u, axis_v)[0]


def _add_scanline_quad(
    target: list[np.ndarray],
    s0: float,
    s1: float,
    a0: float,
    b0: float,
    a1: float,
    b1: float,
    axis: int,
    origin: np.ndarray,
    axis_u: np.ndarray,
    axis_v: np.ndarray,
    normal: np.ndarray,
) -> None:
    if min(abs(b0 - a0), abs(b1 - a1), abs(s1 - s0)) < 1e-12:
        return
    p00 = _station_point_3d(s0, a0, axis, origin, axis_u, axis_v)
    p01 = _station_point_3d(s0, b0, axis, origin, axis_u, axis_v)
    p11 = _station_point_3d(s1, b1, axis, origin, axis_u, axis_v)
    p10 = _station_point_3d(s1, a1, axis, origin, axis_u, axis_v)
    target.append(_orient_triangle(np.array([p00, p10, p11]), normal))
    target.append(_orient_triangle(np.array([p00, p11, p01]), normal))


def _build_centerline_band_outlet_regions(
    outlet_triangles: list[np.ndarray],
    source_patch_name: str,
    interior_patch_name: str,
    exclusion_fraction: float,
    station_count: int = 240,
) -> dict[str, list[np.ndarray]]:
    triangles = np.asarray(outlet_triangles, dtype=float)
    outer_3d = _order_boundary_loop_from_triangles(triangles)
    origin, normal, axis_u, axis_v = _make_plane_basis(outer_3d, triangles)
    outer_2d = _project_to_plane(outer_3d, origin, axis_u, axis_v)

    if _polygon_area_2d(outer_2d) < 0:
        outer_2d = outer_2d[::-1]

    span = outer_2d.max(axis=0) - outer_2d.min(axis=0)
    axis = int(np.argmax(span))
    axis_min = float(outer_2d[:, axis].min())
    axis_max = float(outer_2d[:, axis].max())
    if axis_max <= axis_min:
        raise ValueError("Outlet boundary has invalid 2D extent.")

    exclusion_fraction = float(np.clip(exclusion_fraction, 0.0, 0.49))
    band_fraction = 1.0 - 2.0 * exclusion_fraction
    if band_fraction <= 0.0:
        raise ValueError("Centerline band is empty. Lower outlet_interior.inset_fraction.")

    eps = 1e-9 * (axis_max - axis_min)
    stations = np.linspace(axis_min + eps, axis_max - eps, int(station_count))
    station_intervals = [_scanline_intervals_2d(outer_2d, axis, s) for s in stations]

    source_tris: list[np.ndarray] = []
    interior_tris: list[np.ndarray] = []
    skipped = 0
    strips = 0

    for k in range(len(stations) - 1):
        intervals0 = station_intervals[k]
        intervals1 = station_intervals[k + 1]
        if not intervals0 or not intervals1:
            continue
        if len(intervals0) != len(intervals1):
            skipped += 1
            continue

        s0 = float(stations[k])
        s1 = float(stations[k + 1])
        for (lo0, hi0), (lo1, hi1) in zip(intervals0, intervals1):
            c0 = 0.5 * (lo0 + hi0)
            c1 = 0.5 * (lo1 + hi1)
            h0 = 0.5 * (hi0 - lo0) * band_fraction
            h1 = 0.5 * (hi1 - lo1) * band_fraction
            blo0, bhi0 = c0 - h0, c0 + h0
            blo1, bhi1 = c1 - h1, c1 + h1

            _add_scanline_quad(source_tris, s0, s1, lo0, blo0, lo1, blo1, axis, origin, axis_u, axis_v, normal)
            _add_scanline_quad(interior_tris, s0, s1, blo0, bhi0, blo1, bhi1, axis, origin, axis_u, axis_v, normal)
            _add_scanline_quad(source_tris, s0, s1, bhi0, hi0, bhi1, hi1, axis, origin, axis_u, axis_v, normal)
            strips += 1

    if not interior_tris:
        raise ValueError("Centerline outletInterior produced no triangles.")
    if not source_tris:
        raise ValueError("Centerline outlet rim produced no triangles.")

    print(
        f"Created {interior_patch_name} with centerline band: "
        f"axis={'uv'[axis]}, exclusion_fraction={exclusion_fraction:.3f}, "
        f"band_fraction={band_fraction:.3f}, stations={len(stations)}, "
        f"strips={strips}, skipped_branch_transitions={skipped}"
    )

    return {
        source_patch_name: source_tris,
        interior_patch_name: interior_tris,
    }


def _remove_box_caps(
    mesh: trimesh.Trimesh,
    box_bounds: np.ndarray,
    tol: float = 1e-6,
) -> trimesh.Trimesh:
    """Remove faces that lie flat on box boundary planes (caps from boolean)."""
    box_min, box_max = box_bounds[0], box_bounds[1]
    face_verts = mesh.vertices[mesh.faces]  # (n_faces, 3, 3)

    is_cap = np.zeros(len(mesh.faces), dtype=bool)
    for axis in range(3):
        for val in [box_min[axis], box_max[axis]]:
            on_plane = np.all(np.abs(face_verts[:, :, axis] - val) < tol, axis=1)
            n = on_plane.sum()
            if n > 0:
                print(f"[_remove_box_caps] axis={axis} val={val:.4f}: {n} cap faces (tol={tol})")
            is_cap |= on_plane

    print(f"[_remove_box_caps] total caps removed: {is_cap.sum()} / {len(mesh.faces)} faces")
    if is_cap.any():
        mesh = mesh.copy()
        mesh.update_faces(~is_cap)
        mesh.remove_unreferenced_vertices()
    return mesh

#!/usr/bin/env python3

"""Stitch two open-boundary meshes into a single watertight mesh.

Greedy boundary zippering: extract the open boundary loop of each mesh,
pair them by centroid proximity, align their traversal direction via a
Newell-normal dot-product test, pick the closest starting pair, then walk
both loops forward -- advancing whichever cursor produces the shorter new
bridge edge -- emitting one triangle per step. Handles loops with
different vertex counts and makes no assumption about a shared axis.
"""




def _get_boundary_edges(mesh: trimesh.Trimesh) -> np.ndarray:
    """Return undirected boundary edges as an (N, 2) array of sorted index pairs."""
    faces = np.asarray(mesh.faces)
    if faces.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    edges = np.sort(edges, axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    return unique[counts == 1].astype(np.int64)


def _chain_edges_to_loops(edges: np.ndarray) -> List[List[int]]:
    """Chain boundary edges into closed vertex loops.

    Each boundary vertex should connect to exactly two boundary edges,
    forming one or more closed polygons.
    """
    adj: Dict[int, List[int]] = defaultdict(list)
    for v0, v1 in edges:
        adj[int(v0)].append(int(v1))
        adj[int(v1)].append(int(v0))

    visited: set = set()
    loops: List[List[int]] = []

    for start in list(adj.keys()):
        if start in visited:
            continue
        loop: List[int] = []
        current = start
        prev = -1
        while current not in visited:
            visited.add(current)
            loop.append(current)
            neighbors = [n for n in adj[current] if n != prev]
            if not neighbors:
                break
            next_v = neighbors[0]
            if next_v == start:
                break
            prev = current
            current = next_v
        if len(loop) >= 3:
            loops.append(loop)

    return loops


def _loop_normal(vertices: np.ndarray, loop: List[int]) -> np.ndarray:
    """Unit normal of a (possibly non-planar) loop.

    Sum of cross(p_i, p_{i+1}) over a closed loop is translation-invariant,
    so the direction depends only on shape and traversal order.
    """
    pts = vertices[np.asarray(loop, dtype=np.int64)]
    n = np.cross(pts, np.roll(pts, -1, axis=0)).sum(axis=0)
    norm = float(np.linalg.norm(n))
    if norm < 1e-30:
        return np.array([0.0, 0.0, 1.0])
    return n / norm


def _loop_centroid(vertices: np.ndarray, loop: List[int]) -> np.ndarray:
    return vertices[np.asarray(loop, dtype=np.int64)].mean(axis=0)


def _pair_all_loops(
    verts_a: np.ndarray,
    loops_a: List[List[int]],
    verts_b: np.ndarray,
    loops_b: List[List[int]],
) -> List[Tuple[List[int], List[int]]]:
    """Greedy 1-to-1 matching of A-loops to B-loops by centroid distance.

    Returns min(len(loops_a), len(loops_b)) pairs; any surplus loops on the
    larger side stay unpaired (and thus unstitched) so the caller can see
    them in the diagnostics.
    """
    if not loops_a or not loops_b:
        return []
    ca = np.asarray([_loop_centroid(verts_a, l) for l in loops_a])
    cb = np.asarray([_loop_centroid(verts_b, l) for l in loops_b])
    dists = np.linalg.norm(ca[:, None, :] - cb[None, :, :], axis=2)

    flat = sorted(
        ((float(dists[i, j]), i, j) for i in range(len(loops_a)) for j in range(len(loops_b))),
        key=lambda x: x[0],
    )
    pairs: List[Tuple[List[int], List[int]]] = []
    used_a: set = set()
    used_b: set = set()
    k = min(len(loops_a), len(loops_b))
    for _, i, j in flat:
        if i in used_a or j in used_b:
            continue
        pairs.append((loops_a[i], loops_b[j]))
        used_a.add(i)
        used_b.add(j)
        if len(pairs) == k:
            break
    return pairs


def _orient_loops(
    verts_a: np.ndarray,
    loop_a: List[int],
    verts_b: np.ndarray,
    loop_b: List[int],
) -> Tuple[List[int], List[int]]:
    """Make both loops traverse the gap in the same rotational sense.

    Two loops on meshes facing each other naturally have anti-parallel
    Newell normals. When that's the case, reverse B so both walk the gap
    the same way -- the zipper needs this to emit consistently-wound
    bridge triangles.
    """
    n_a = _loop_normal(verts_a, loop_a)
    n_b = _loop_normal(verts_b, loop_b)
    if float(np.dot(n_a, n_b)) < 0.0:
        loop_b = list(reversed(loop_b))
    return loop_a, loop_b


def _find_start_pair(
    verts_a: np.ndarray,
    loop_a: List[int],
    verts_b: np.ndarray,
    loop_b: List[int],
) -> Tuple[int, int]:
    """Loop-index pair (i, j) minimizing ||A[i] - B[j]||."""
    pts_a = verts_a[np.asarray(loop_a, dtype=np.int64)]
    pts_b = verts_b[np.asarray(loop_b, dtype=np.int64)]
    d2 = np.sum((pts_a[:, None, :] - pts_b[None, :, :]) ** 2, axis=2)
    i, j = np.unravel_index(int(np.argmin(d2)), d2.shape)
    return int(i), int(j)


def _zipper(
    verts_a: np.ndarray,
    loop_a: List[int],
    verts_b: np.ndarray,
    loop_b: List[int],
    start_a: int,
    start_b: int,
    offset_b: int,
) -> np.ndarray:
    """Greedy bridge between two oriented loops.

    B-side indices are returned already shifted by ``offset_b`` so the
    resulting triangles index directly into the concatenated
    (verts_a, verts_b) array.
    """
    na = len(loop_a)
    nb = len(loop_b)
    i, j = start_a, start_b
    steps_a, steps_b = 0, 0
    tris: List[Tuple[int, int, int]] = []

    while steps_a < na or steps_b < nb:
        a_curr = loop_a[i]
        b_curr_local = loop_b[j]
        i_next = (i + 1) % na
        j_next = (j + 1) % nb
        a_next = loop_a[i_next]
        b_next_local = loop_b[j_next]

        if steps_a >= na:
            advance_a = False
        elif steps_b >= nb:
            advance_a = True
        else:
            d_adv_a = float(np.linalg.norm(verts_a[a_next] - verts_b[b_curr_local]))
            d_adv_b = float(np.linalg.norm(verts_a[a_curr] - verts_b[b_next_local]))
            advance_a = d_adv_a <= d_adv_b

        if advance_a:
            tris.append((a_curr, a_next, b_curr_local + offset_b))
            i = i_next
            steps_a += 1
        else:
            tris.append((a_curr, b_next_local + offset_b, b_curr_local + offset_b))
            j = j_next
            steps_b += 1

    return np.asarray(tris, dtype=np.int64)


def stitch_meshes(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh) -> trimesh.Trimesh:
    """Stitch two meshes along their closest open boundary loops.

    Returns a single triangle mesh combining mesh_a, mesh_b, and the
    bridge band connecting their paired boundaries.
    """
    edges_a = _get_boundary_edges(mesh_a)
    edges_b = _get_boundary_edges(mesh_b)
    if edges_a.size == 0:
        raise ValueError("mesh_a has no open boundary to stitch.")
    if edges_b.size == 0:
        raise ValueError("mesh_b has no open boundary to stitch.")

    loops_a = _chain_edges_to_loops(edges_a)
    loops_b = _chain_edges_to_loops(edges_b)
    if not loops_a or not loops_b:
        raise ValueError("Failed to chain boundary edges into closed loops.")

    pairs = _pair_all_loops(mesh_a.vertices, loops_a, mesh_b.vertices, loops_b)
    if not pairs:
        raise ValueError("No boundary loop pairs could be formed.")

    offset_b = int(mesh_a.vertices.shape[0])
    bridges: List[np.ndarray] = []
    for loop_a, loop_b in pairs:
        loop_a, loop_b = _orient_loops(mesh_a.vertices, loop_a, mesh_b.vertices, loop_b)
        start_a, start_b = _find_start_pair(mesh_a.vertices, loop_a, mesh_b.vertices, loop_b)
        bridges.append(
            _zipper(
                mesh_a.vertices,
                loop_a,
                mesh_b.vertices,
                loop_b,
                start_a,
                start_b,
                offset_b,
            )
        )

    combined_verts = np.vstack([np.asarray(mesh_a.vertices), np.asarray(mesh_b.vertices)])
    combined_faces = np.vstack([
        np.asarray(mesh_a.faces, dtype=np.int64),
        np.asarray(mesh_b.faces, dtype=np.int64) + offset_b,
        *bridges,
    ])

    stitched = trimesh.Trimesh(vertices=combined_verts, faces=combined_faces, process=False)
    stitched.process(validate=False)
    stitched.fix_normals()
    return stitched


def mesh_diagnostics(mesh: trimesh.Trimesh) -> Dict[str, object]:
    """Per-mesh boundary-detection report for debugging stitching inputs."""
    edges = _get_boundary_edges(mesh)
    loops = _chain_edges_to_loops(edges)
    return {
        "n_vertices": int(mesh.vertices.shape[0]),
        "n_faces": int(mesh.faces.shape[0]),
        "n_boundary_edges": int(edges.shape[0]),
        "n_loops": len(loops),
        "loop_lengths": [len(loop) for loop in loops],
        "loop_centroids": [
            tuple(float(c) for c in _loop_centroid(mesh.vertices, loop)) for loop in loops
        ],
    }


def stitch_diagnostics(mesh: trimesh.Trimesh) -> Dict[str, object]:
    """Quick quality report for a stitched mesh."""
    watertight = bool(mesh.is_watertight)
    return {
        "n_vertices": int(mesh.vertices.shape[0]),
        "n_faces": int(mesh.faces.shape[0]),
        "n_boundary_edges": int(_get_boundary_edges(mesh).shape[0]),
        "is_watertight": watertight,
        "is_winding_consistent": bool(mesh.is_winding_consistent),
        "euler_number": int(mesh.euler_number),
        "volume": float(mesh.volume) if watertight else float("nan"),
    }


def make_boundary_edge_marker_mesh(
    mesh: trimesh.Trimesh,
    boundary_edges: np.ndarray | None = None,
    radius: float | None = None,
    sections: int = 6,
) -> trimesh.Trimesh:
    """Create a visualization mesh that highlights boundary edges.

    ParaView cannot directly show "selected edges" when loading an STL.
    This helper converts boundary edges into thin cylinders and returns
    them as a separate triangle mesh which can be exported as STL.

    Notes
    -----
    - The output is meant for debugging/visual inspection only.
    - Use a small *radius* relative to your geometry scale.
    """
    if boundary_edges is None:
        boundary_edges = _get_boundary_edges(mesh)

    if len(boundary_edges) == 0:
        return trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64), process=False)

    if radius is None:
        # Small but visible tube radius relative to mesh size
        radius = float(mesh.scale) * 0.002
        if radius <= 0:
            radius = 1e-3

    cylinders: List[trimesh.Trimesh] = []
    verts = mesh.vertices
    for v0, v1 in boundary_edges:
        p0 = verts[int(v0)]
        p1 = verts[int(v1)]
        if not np.all(np.isfinite(p0)) or not np.all(np.isfinite(p1)):
            continue
        if np.linalg.norm(p1 - p0) < 1e-12:
            continue
        cyl = trimesh.creation.cylinder(
            radius=radius,
            sections=int(sections),
            segment=np.stack([p0, p1], axis=0),
        )
        cylinders.append(cyl)

    if not cylinders:
        return trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64), process=False)

    return trimesh.util.concatenate(cylinders)


def export_boundary_edge_markers_stl(
    mesh: trimesh.Trimesh,
    out_path: str | Path,
    boundary_edges: np.ndarray | None = None,
    radius: float | None = None,
    sections: int = 6,
) -> Path:
    """Export boundary edges as a separate STL for ParaView visualization."""
    out_path = Path(out_path)
    marker = make_boundary_edge_marker_mesh(
        mesh,
        boundary_edges=boundary_edges,
        radius=radius,
        sections=sections,
    )
    marker.export(str(out_path))
    return out_path


if __name__ == "__main__":
    mesh_a = trimesh.load("open_cylinder_1.stl", process=False)
    mesh_b = trimesh.load("open_cylinder_2.stl", process=False)

    stitched = stitch_meshes(mesh_a, mesh_b)
    stitched.export("stitched.stl")

    print("stitch diagnostics:")
    for key, value in stitch_diagnostics(stitched).items():
        print(f"  {key}: {value}")

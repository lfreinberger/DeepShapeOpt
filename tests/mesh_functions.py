# import numpy as np
# import trimesh
# import torch
# from collections import defaultdict
# from pathlib import Path
# from typing import Callable, Dict, List, Literal

# # ---------------------------------------------------------------------------
# # Mesh stitching at box boundaries
# # ---------------------------------------------------------------------------

# def _get_boundary_edges(mesh: trimesh.Trimesh) -> np.ndarray:
#     """Return boundary edges (edges in exactly one face) as (N, 2) array."""
#     edge_face_count = np.bincount(
#         mesh.edges_unique_inverse, minlength=len(mesh.edges_unique),
#     )
#     return mesh.edges_unique[edge_face_count == 1]


# def make_boundary_edge_marker_mesh(
#     mesh: trimesh.Trimesh,
#     boundary_edges: np.ndarray | None = None,
#     radius: float | None = None,
#     sections: int = 6,
# ) -> trimesh.Trimesh:
#     """Create a visualization mesh that highlights boundary edges.

#     ParaView cannot directly show "selected edges" when loading an STL.
#     This helper converts boundary edges into thin cylinders and returns
#     them as a separate triangle mesh which can be exported as STL.

#     Notes
#     -----
#     - The output is meant for debugging/visual inspection only.
#     - Use a small *radius* relative to your geometry scale.
#     """
#     if boundary_edges is None:
#         boundary_edges = _get_boundary_edges(mesh)

#     if len(boundary_edges) == 0:
#         return trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64), process=False)

#     if radius is None:
#         # Small but visible tube radius relative to mesh size
#         radius = float(mesh.scale) * 0.002
#         if radius <= 0:
#             radius = 1e-3

#     cylinders: List[trimesh.Trimesh] = []
#     verts = mesh.vertices
#     for v0, v1 in boundary_edges:
#         p0 = verts[int(v0)]
#         p1 = verts[int(v1)]
#         if not np.all(np.isfinite(p0)) or not np.all(np.isfinite(p1)):
#             continue
#         if np.linalg.norm(p1 - p0) < 1e-12:
#             continue
#         cyl = trimesh.creation.cylinder(
#             radius=radius,
#             sections=int(sections),
#             segment=np.stack([p0, p1], axis=0),
#         )
#         cylinders.append(cyl)

#     if not cylinders:
#         return trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64), process=False)

#     return trimesh.util.concatenate(cylinders)


# def export_boundary_edge_markers_stl(
#     mesh: trimesh.Trimesh,
#     out_path: str | Path,
#     boundary_edges: np.ndarray | None = None,
#     radius: float | None = None,
#     sections: int = 6,
# ) -> Path:
#     """Export boundary edges as a separate STL for ParaView visualization."""
#     out_path = Path(out_path)
#     marker = make_boundary_edge_marker_mesh(
#         mesh,
#         boundary_edges=boundary_edges,
#         radius=radius,
#         sections=sections,
#     )
#     marker.export(str(out_path))
#     return out_path


# def _order_edges_into_loops(edges: np.ndarray) -> List[List[int]]:
#     """Order boundary edges into closed loops of vertex indices."""
#     adj: Dict[int, List[int]] = defaultdict(list)
#     for v0, v1 in edges:
#         adj[v0].append(v1)
#         adj[v1].append(v0)

#     visited_edges: set = set()
#     loops: List[List[int]] = []

#     for start in sorted(adj.keys()):
#         if all(
#             (min(start, n), max(start, n)) in visited_edges
#             for n in adj[start]
#         ):
#             continue
#         loop = [start]
#         current = start
#         while True:
#             next_v = None
#             for n in adj[current]:
#                 ek = (min(current, n), max(current, n))
#                 if ek not in visited_edges:
#                     next_v = n
#                     visited_edges.add(ek)
#                     break
#             if next_v is None or next_v == start:
#                 break
#             loop.append(next_v)
#             current = next_v
#         if len(loop) >= 3:
#             loops.append(loop)
#     return loops


# def _find_boundary_loops_at_plane(
#     mesh: trimesh.Trimesh,
#     boundary_edges: np.ndarray,
#     axis: int,
#     value: float,
#     tol: float,
#     vertices_override: np.ndarray | None = None,
#     index_offset: int = 0,
# ) -> List[List[int]]:
#     """Find ordered boundary loops whose vertices lie near a coordinate plane.

#     Parameters
#     ----------
#     vertices_override : array or None
#         If provided, read vertex coordinates from this array (with
#         ``index_offset`` added to edge indices) instead of ``mesh.vertices``.
#     index_offset : int
#         Added to edge vertex indices when using *vertices_override*.
#     """
#     if len(boundary_edges) == 0:
#         return []
#     if vertices_override is not None:
#         v0c = vertices_override[boundary_edges[:, 0] + index_offset, axis]
#         v1c = vertices_override[boundary_edges[:, 1] + index_offset, axis]
#     else:
#         v0c = mesh.vertices[boundary_edges[:, 0], axis]
#         v1c = mesh.vertices[boundary_edges[:, 1], axis]
#     near = (np.abs(v0c - value) < tol) & (np.abs(v1c - value) < tol)
#     if near.sum() == 0:
#         return []
#     return _order_edges_into_loops(boundary_edges[near])


# def _remove_box_caps(
#     mesh: trimesh.Trimesh,
#     box_bounds: np.ndarray,
#     tol: float = 1e-6,
# ) -> trimesh.Trimesh:
#     """Remove faces that lie flat on box boundary planes (caps from boolean)."""
#     box_min, box_max = box_bounds[0], box_bounds[1]
#     face_verts = mesh.vertices[mesh.faces]  # (n_faces, 3, 3)

#     is_cap = np.zeros(len(mesh.faces), dtype=bool)
#     for axis in range(3):
#         for val in [box_min[axis], box_max[axis]]:
#             on_plane = np.all(np.abs(face_verts[:, :, axis] - val) < tol, axis=1)
#             n = on_plane.sum()
#             if n > 0:
#                 print(f"[_remove_box_caps] axis={axis} val={val:.4f}: {n} cap faces (tol={tol})")
#             is_cap |= on_plane

#     print(f"[_remove_box_caps] total caps removed: {is_cap.sum()} / {len(mesh.faces)} faces")
#     if is_cap.any():
#         mesh = mesh.copy()
#         mesh.update_faces(~is_cap)
#         mesh.remove_unreferenced_vertices()
#     return mesh

import numpy as np
import trimesh
import triangle as tr
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Literal

# ---------------------------------------------------------------------------
# Mesh stitching at box boundaries
# ---------------------------------------------------------------------------

def _get_boundary_edges(mesh: trimesh.Trimesh) -> np.ndarray:
    """Return boundary edges (edges in exactly one face) as (N, 2) array."""
    edge_face_count = np.bincount(
        mesh.edges_unique_inverse, minlength=len(mesh.edges_unique),
    )
    return mesh.edges_unique[edge_face_count == 1]


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


def _signed_area_2d(pts: np.ndarray) -> float:
    """Signed area of a 2D polygon via the shoelace formula."""
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _split_edges_at_boundary_vertices(
    vertices: np.ndarray,
    faces_list: List[List[int]],
    tol: float = 1e-8,
) -> None:
    """Split faces where a boundary vertex lies on another boundary edge.

    After CDT fills, the CDT may split an original boundary segment at an
    interior vertex (e.g. where an inscribed circle touches the bounding
    rectangle).  The original mesh still has the full unsplit edge, causing
    both the full edge and its sub-edges to remain as boundary edges.

    This helper detects such cases and splits the face containing the full
    edge, making the sub-edges match.  Modifies *faces_list* in place.
    """
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces_list), process=False)
    boundary = _get_boundary_edges(mesh)
    if len(boundary) == 0:
        return

    bv_set: set = set()
    for e in boundary:
        bv_set.add(int(e[0]))
        bv_set.add(int(e[1]))

    for edge in boundary:
        v0, v1 = int(edge[0]), int(edge[1])
        p0, p1 = vertices[v0], vertices[v1]
        edge_vec = p1 - p0
        edge_len = np.linalg.norm(edge_vec)
        if edge_len < tol:
            continue
        edge_dir = edge_vec / edge_len

        # Boundary vertices that lie strictly inside this edge
        on_edge: List[tuple] = []
        for vm in bv_set:
            if vm == v0 or vm == v1:
                continue
            pm = vertices[vm]
            t = float(np.dot(pm - p0, edge_dir))
            if t <= tol or t >= edge_len - tol:
                continue
            if np.linalg.norm(pm - (p0 + t * edge_dir)) < tol:
                on_edge.append((t, vm))

        if not on_edge:
            continue
        on_edge.sort()
        chain = [v0] + [vm for _, vm in on_edge] + [v1]

        # Find the face containing the full edge (v0, v1) and split it
        for fi in range(len(faces_list)):
            fl = [int(x) for x in faces_list[fi]]
            if v0 in fl and v1 in fl:
                i0, i1 = fl.index(v0), fl.index(v1)
                third = fl[3 - i0 - i1]
                if (i1 - i0) % 3 == 1:
                    split = [[chain[k], chain[k + 1], third]
                             for k in range(len(chain) - 1)]
                else:
                    rev = list(reversed(chain))
                    split = [[rev[k], rev[k + 1], third]
                             for k in range(len(rev) - 1)]
                faces_list[fi] = split[0]
                faces_list.extend(split[1:])
                break


def _find_stitching_planes(
    vertices: np.ndarray,
    boundary_edges: np.ndarray,
    tol: float = 1e-6,
) -> Dict[tuple, np.ndarray]:
    """Detect axis-aligned planes that contain boundary edges forming loops.

    Returns a dict mapping ``(axis, value)`` to the subset of boundary
    edges that lie on that plane.
    """
    # Collect candidate (axis, rounded_value) for every boundary edge
    plane_candidates: Dict[tuple, List[int]] = defaultdict(list)
    for idx, (v0, v1) in enumerate(boundary_edges):
        p0, p1 = vertices[v0], vertices[v1]
        for ax in range(3):
            if abs(p0[ax] - p1[ax]) < tol:
                val = round(float(p0[ax] + p1[ax]) / 2, 6)
                plane_candidates[(ax, val)].append(idx)

    # Keep only planes whose edges form at least one closed loop
    result: Dict[tuple, np.ndarray] = {}
    processed: set = set()
    # Process largest groups first so shared edges land in the right plane
    for key in sorted(plane_candidates, key=lambda k: -len(plane_candidates[k])):
        idxs = [i for i in plane_candidates[key] if i not in processed]
        if len(idxs) < 3:
            continue
        edges = boundary_edges[idxs]
        loops = _chain_edges_to_loops(edges)
        if loops:
            result[key] = edges
            processed.update(idxs)

    return result


def stitch_meshes_at_box(
    outer: trimesh.Trimesh,
    inner: trimesh.Trimesh,
    box_bounds: np.ndarray,
    tol: float = 1e-6,
) -> trimesh.Trimesh:
    """Stitch outer and inner meshes along axis-aligned boundary planes.

    After a box boolean-difference and cap removal, both meshes have open
    boundary edges.  This function auto-detects the axis-aligned planes
    that contain those boundary edges and fills the gaps between the
    outer and inner boundary loops with new triangles via Constrained
    Delaunay Triangulation (CDT).

    Parameters
    ----------
    outer, inner : trimesh.Trimesh
        Meshes to join.  Each should already have caps removed so that
        boundary edges are exposed.
    box_bounds : (2, 3) array
        ``[[x_min, y_min, z_min], [x_max, y_max, z_max]]`` — kept for
        API context but plane detection is fully automatic.
    tol : float
        Tolerance for coplanarity checks.
    """
    combined = trimesh.util.concatenate([outer, inner])
    boundary_edges = _get_boundary_edges(combined)

    if len(boundary_edges) == 0:
        return combined

    planes = _find_stitching_planes(combined.vertices, boundary_edges, tol)
    if not planes:
        return combined

    extra_verts: List[np.ndarray] = []
    extra_faces: List[np.ndarray] = []

    for (axis, val), plane_edges in planes.items():
        loops = _chain_edges_to_loops(plane_edges)
        if not loops:
            continue

        # 2D projection axes (drop the constant axis)
        axes_2d = [i for i in range(3) if i != axis]

        # Unique vertex list and local ↔ global index mapping
        seen: set = set()
        unique_global: List[int] = []
        for loop in loops:
            for v in loop:
                if v not in seen:
                    seen.add(v)
                    unique_global.append(v)
        g2l = {g: l for l, g in enumerate(unique_global)}

        pts_2d = combined.vertices[unique_global][:, axes_2d]

        # Constrained segments for every loop
        segments: List[List[int]] = []
        for loop in loops:
            n = len(loop)
            for i in range(n):
                segments.append([g2l[loop[i]], g2l[loop[(i + 1) % n]]])

        # Largest-area loop is the outer boundary; the rest are holes
        abs_areas = [
            abs(_signed_area_2d(combined.vertices[lp][:, axes_2d]))
            for lp in loops
        ]
        outer_idx = int(np.argmax(abs_areas))

        hole_pts: List[List[float]] = []
        for i, lp in enumerate(loops):
            if i != outer_idx:
                centroid = combined.vertices[lp][:, axes_2d].mean(axis=0)
                hole_pts.append(centroid.tolist())

        # --- CDT ---
        cdt_in: Dict = dict(
            vertices=pts_2d,
            segments=np.array(segments, dtype=np.int32),
        )
        if hole_pts:
            cdt_in["holes"] = np.array(hole_pts)

        cdt_out = tr.triangulate(cdt_in, "p")
        out_verts = cdt_out["vertices"]
        out_tris  = cdt_out["triangles"]

        if len(out_tris) == 0:
            continue

        # Handle possible Steiner points added by Triangle
        n_orig = len(unique_global)
        n_out  = len(out_verts)
        steiner_base = len(combined.vertices) + sum(len(a) for a in extra_verts)

        if n_out > n_orig:
            st_2d = out_verts[n_orig:]
            st_3d = np.zeros((len(st_2d), 3))
            st_3d[:, axes_2d[0]] = st_2d[:, 0]
            st_3d[:, axes_2d[1]] = st_2d[:, 1]
            st_3d[:, axis] = val
            extra_verts.append(st_3d)

        # Map CDT-local indices → global mesh indices
        new_faces = np.empty(out_tris.shape, dtype=np.int64)
        for i, tri_local in enumerate(out_tris):
            for j, v in enumerate(tri_local):
                if v < n_orig:
                    new_faces[i, j] = unique_global[v]
                else:
                    new_faces[i, j] = steiner_base + (v - n_orig)

        # Fix winding: the shared boundary edge must be traversed in
        # opposite directions in the existing face and the new fill face.
        ref_v0, ref_v1 = int(plane_edges[0, 0]), int(plane_edges[0, 1])

        existing_fwd = None
        for face in combined.faces:
            fl = [int(x) for x in face]
            if ref_v0 in fl and ref_v1 in fl:
                i0, i1 = fl.index(ref_v0), fl.index(ref_v1)
                existing_fwd = (i1 - i0) % 3 == 1
                break

        if existing_fwd is not None:
            for tri_row in new_faces:
                tl = [int(x) for x in tri_row]
                if ref_v0 in tl and ref_v1 in tl:
                    i0, i1 = tl.index(ref_v0), tl.index(ref_v1)
                    fill_fwd = (i1 - i0) % 3 == 1
                    if fill_fwd == existing_fwd:
                        new_faces = new_faces[:, ::-1]
                    break

        extra_faces.append(new_faces)

    if extra_faces:
        all_verts = combined.vertices
        if extra_verts:
            all_verts = np.concatenate([all_verts] + extra_verts)
        all_faces_arr = np.concatenate([combined.faces] + extra_faces)

        # Post-process: split original faces where CDT sub-divided a
        # boundary segment at an interior vertex (e.g. inscribed-circle
        # touching the bounding rectangle).
        faces_list = [list(f) for f in all_faces_arr]
        _split_edges_at_boundary_vertices(all_verts, faces_list, tol)

        combined = trimesh.Trimesh(
            vertices=all_verts, faces=np.array(faces_list), process=False,
        )
        combined.remove_unreferenced_vertices()

    return combined
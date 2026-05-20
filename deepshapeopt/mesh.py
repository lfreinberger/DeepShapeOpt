from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import trimesh
import torch
from stl import mesh as stl_mesh


logger = logging.getLogger(__name__)


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
    from DeepSDFStruct.optimization import tet_signed_vol

    # Fix orientation: swap vertices 1 and 2 on negative-volume tets
    perm = torch.tensor([0, 2, 1, 3], device=tets.device)
    tets = tets[:, perm]

    vols = tet_signed_vol(verts, tets)  # (n_tet,)

    #warning if some vols are negative including percentage
    n_negative = (vols < 0).sum().item()
    # if n_negative > 0:


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

    logger.debug("Loading %s", input_stl)
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

        method = str(outlet_interior.get("method", "polygon_offset"))
        debug_dir = (output_dir / "debug_outlet_interior") if outlet_interior.get("debug", True) else None
        if method == "polygon_offset":
            interior_regions = _build_polygon_offset_outlet_regions(
                patch_faces[source_patch],
                source_patch_name=source_patch,
                interior_patch_name=interior_patch,
                inset_distance=float(outlet_interior.get("inset_distance", 1.0)),
                quad_segs=int(outlet_interior.get("quad_segs", 8)),
                join_style=str(outlet_interior.get("join_style", "round")),
                triangulation_engine=str(outlet_interior.get("triangulation_engine", "triangle")),
                debug_dir=debug_dir,
            )
        elif method == "medial_axis":
            interior_regions = _build_medial_axis_outlet_regions(
                patch_faces[source_patch],
                source_patch_name=source_patch,
                interior_patch_name=interior_patch,
                boundary_sample_ds=float(outlet_interior.get("boundary_sample_ds", 0.05)),
                strip_half_width=float(outlet_interior.get("strip_half_width", 0.1)),
                min_dist_from_boundary=float(outlet_interior.get("min_dist_from_boundary", 0.3)),
                prune_branch_len=float(outlet_interior.get("prune_branch_len", 1.0)),
                triangulation_engine=str(outlet_interior.get("triangulation_engine", "triangle")),
                debug_dir=debug_dir,
            )
        else:
            raise ValueError(
                f"Unknown outlet_interior.method '{method}' "
                "(expected 'polygon_offset' or 'medial_axis')."
            )
        patch_faces[source_patch] = interior_regions[source_patch]
        patch_faces[interior_patch] = interior_regions[interior_patch]
        logger.info(
            "Outlet interior patch split: %s=%d triangles, %s=%d triangles",
            source_patch,
            len(patch_faces[source_patch]),
            interior_patch,
            len(patch_faces[interior_patch]),
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

        logger.debug("Saved multi-region STL to %s", multi_path)

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
        logger.debug("Saved outlet debug STL: %s", path)

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
    logger.debug("Saved outlet split debug STL: %s", combined_path)


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


def _all_boundary_loops_from_triangles(triangles: np.ndarray, tol: float = 1e-8) -> list[np.ndarray]:
    """Return every closed boundary loop of the given triangle set as a 3D point array.

    Unlike _order_boundary_loop_from_triangles (which returns only the longest loop),
    this preserves disconnected outlet components — required for multi-channel cross
    sections where each channel has its own perimeter.
    """
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

    verts = np.asarray(vertices)
    loops: list[np.ndarray] = []
    visited_edges: set[tuple[int, int]] = set()
    for start in list(adjacency.keys()):
        for first_next in adjacency[start]:
            edge = tuple(sorted((start, first_next)))
            if edge in visited_edges:
                continue
            loop_ids = [start]
            prev, curr = start, first_next
            closed = False
            while True:
                visited_edges.add(tuple(sorted((prev, curr))))
                loop_ids.append(curr)
                candidates = [n for n in adjacency[curr] if n != prev and tuple(sorted((curr, n))) not in visited_edges]
                if not candidates:
                    # Try to close on start if possible
                    if start in adjacency[curr]:
                        visited_edges.add(tuple(sorted((curr, start))))
                        closed = True
                    break
                next_id = candidates[0]
                if next_id == start:
                    visited_edges.add(tuple(sorted((curr, next_id))))
                    closed = True
                    break
                prev, curr = curr, next_id
                if len(loop_ids) > len(adjacency) + 2:
                    raise ValueError("Could not chain outlet boundary loop.")
            if closed and len(loop_ids) >= 3:
                loops.append(verts[loop_ids])

    if not loops:
        raise ValueError("Could not find any closed outlet boundary loop.")
    return loops


def _build_shapely_multipolygon(loops_2d: list[np.ndarray]):
    """Build a shapely (Multi)Polygon from a list of 2D loops, classifying outer rings vs holes by containment depth.

    Even depth (0, 2, ...) -> outer ring of a new polygon; odd depth -> hole of the enclosing ring.
    All input loops must be closed (first != last is acceptable; shapely closes implicitly).
    """
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.geometry.polygon import orient

    rings = [np.asarray(l, dtype=float) for l in loops_2d]
    polys_test = [Polygon(r) for r in rings]
    # Containment graph: ring i is enclosed by ring j iff a vertex of ring i is
    # inside ring j's polygon. A vertex of ring i can never coincide with the
    # interior of ring i itself (so witness is unambiguous), and rings cannot
    # cross (well-formed boundary), so any single vertex suffices.
    from shapely.geometry import Point
    depth = [0] * len(rings)
    parent = [-1] * len(rings)
    for i, ri in enumerate(rings):
        witness = Point(ri[0, 0], ri[0, 1])
        candidates = []
        for j, pj in enumerate(polys_test):
            if i == j:
                continue
            if pj.contains(witness):
                candidates.append(j)
        depth[i] = len(candidates)
        if candidates:
            # Direct parent = smallest enclosing polygon by area.
            parent[i] = min(candidates, key=lambda j: polys_test[j].area)

    # Assemble: outer rings have even depth. Each outer ring's holes are
    # children (parent == this ring) with odd depth.
    outer_indices = [i for i, d in enumerate(depth) if d % 2 == 0]
    polygons = []
    for oi in outer_indices:
        holes = [rings[j].tolist() for j in range(len(rings)) if parent[j] == oi and depth[j] % 2 == 1]
        poly = Polygon(rings[oi].tolist(), holes=holes)
        polygons.append(orient(poly, sign=1.0))  # CCW exterior, CW holes

    if not polygons:
        raise ValueError("polygon_offset: no outer rings detected after containment classification.")
    if len(polygons) == 1:
        return polygons[0]
    return MultiPolygon(polygons)


def _triangulate_shapely_to_2d(poly, engine: str = "triangle") -> list[tuple[np.ndarray, np.ndarray]]:
    """Triangulate a shapely Polygon or MultiPolygon (with optional holes) into 2D triangles.

    Returns a list of (vertices_2d, faces) tuples, one per polygon component. faces indexes vertices_2d.
    Empty geometries return an empty list.
    """
    from shapely.geometry import MultiPolygon, Polygon
    from trimesh.creation import triangulate_polygon

    if poly.is_empty:
        return []
    if isinstance(poly, MultiPolygon):
        components = list(poly.geoms)
    elif isinstance(poly, Polygon):
        components = [poly]
    else:
        # GeometryCollection: keep only polygonal pieces
        components = [g for g in getattr(poly, "geoms", []) if isinstance(g, Polygon) and not g.is_empty]

    out: list[tuple[np.ndarray, np.ndarray]] = []
    for comp in components:
        if comp.is_empty or comp.area <= 0:
            continue
        verts_2d, faces = triangulate_polygon(comp, engine=engine)
        if len(faces) == 0:
            continue
        out.append((np.asarray(verts_2d, dtype=float), np.asarray(faces, dtype=np.int64)))
    return out


def _lift_2d_triangles_to_3d(
    triangulations: list[tuple[np.ndarray, np.ndarray]],
    origin: np.ndarray,
    axis_u: np.ndarray,
    axis_v: np.ndarray,
    normal: np.ndarray,
) -> list[np.ndarray]:
    """Convert per-component 2D triangulations into a flat list of oriented 3D triangles."""
    out: list[np.ndarray] = []
    for verts_2d, faces in triangulations:
        verts_3d = _unproject_from_plane(verts_2d, origin, axis_u, axis_v)
        for tri in verts_3d[faces]:
            out.append(_orient_triangle(tri, normal))
    return out


def _build_polygon_offset_outlet_regions(
    outlet_triangles: list[np.ndarray],
    source_patch_name: str,
    interior_patch_name: str,
    inset_distance: float,
    quad_segs: int = 8,
    join_style: str = "round",
    triangulation_engine: str = "triangle",
    debug_dir: Path | None = None,
) -> dict[str, list[np.ndarray]]:
    """Split the outlet into a smooth interior patch and a wall-margin rim using a true 2D Minkowski offset.

    The outlet boundary is extracted as one or more 2D loops in the outlet plane, assembled into a
    shapely (Multi)Polygon, and shrunk inward via ``buffer(-inset_distance, ...)``. Both the inset
    polygon and its ring complement are re-triangulated so the resulting STL patch boundary is the
    smooth offset curve itself, not the original triangle edges nearest to it. Channels narrower than
    ``2 * inset_distance`` collapse cleanly to empty interior — which is the correct geometric result.
    """
    import shapely  # local import: keeps shapely optional for users not invoking this method

    join_style_map = {"round": "round", "mitre": "mitre", "bevel": "bevel"}
    if join_style not in join_style_map:
        raise ValueError(f"polygon_offset: unknown join_style '{join_style}' (expected one of {sorted(join_style_map)}).")
    if inset_distance <= 0:
        raise ValueError(f"polygon_offset: inset_distance must be positive, got {inset_distance}.")

    triangles = np.asarray(outlet_triangles, dtype=float)
    loops_3d = _all_boundary_loops_from_triangles(triangles)

    all_loop_pts = np.vstack(loops_3d)
    origin, normal, axis_u, axis_v = _make_plane_basis(all_loop_pts, triangles)
    loops_2d = [_project_to_plane(l, origin, axis_u, axis_v) for l in loops_3d]

    poly_outlet = _build_shapely_multipolygon(loops_2d)
    poly_in = poly_outlet.buffer(
        -float(inset_distance),
        quad_segs=int(quad_segs),
        join_style=join_style_map[join_style],
    )
    if poly_in.is_empty or poly_in.area <= 0:
        raise ValueError(
            f"polygon_offset: inset_distance={inset_distance} erased the entire outlet; reduce it."
        )
    poly_ring = poly_outlet.difference(poly_in)

    interior_tris_3d = _lift_2d_triangles_to_3d(
        _triangulate_shapely_to_2d(poly_in, engine=triangulation_engine),
        origin, axis_u, axis_v, normal,
    )
    rim_tris_3d = _lift_2d_triangles_to_3d(
        _triangulate_shapely_to_2d(poly_ring, engine=triangulation_engine),
        origin, axis_u, axis_v, normal,
    )

    logger.info(
        "Created %s via polygon_offset: inset_distance=%.6e, loops=%d, "
        "outlet_area=%.6e, inset_area=%.6e (%.1f%%), ring_area=%.6e, "
        "%s_tris=%d, %s_tris=%d",
        interior_patch_name,
        inset_distance,
        len(loops_2d),
        poly_outlet.area,
        poly_in.area,
        100.0 * poly_in.area / poly_outlet.area,
        poly_ring.area,
        interior_patch_name,
        len(interior_tris_3d),
        source_patch_name,
        len(rim_tris_3d),
    )

    if debug_dir is not None:
        _write_polygon_offset_debug(
            debug_dir, loops_2d, poly_outlet, poly_in, poly_ring,
            origin, axis_u, axis_v,
        )

    return {
        source_patch_name: rim_tris_3d,
        interior_patch_name: interior_tris_3d,
    }


def _write_polygon_offset_debug(
    debug_dir: Path,
    loops_2d: list[np.ndarray],
    poly_outlet,
    poly_in,
    poly_ring,
    origin: np.ndarray,
    axis_u: np.ndarray,
    axis_v: np.ndarray,
) -> None:
    """Write 2D-in-3D polyline OBJs for each stage of the polygon offset; small format, ParaView-readable."""
    from shapely.geometry import MultiPolygon, Polygon

    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    def collect_rings(geom) -> list[np.ndarray]:
        rings: list[np.ndarray] = []
        if geom.is_empty:
            return rings
        if isinstance(geom, MultiPolygon):
            for g in geom.geoms:
                rings.extend(collect_rings(g))
            return rings
        if isinstance(geom, Polygon):
            rings.append(np.asarray(geom.exterior.coords, dtype=float)[:, :2])
            for hole in geom.interiors:
                rings.append(np.asarray(hole.coords, dtype=float)[:, :2])
            return rings
        for g in getattr(geom, "geoms", []):
            if isinstance(g, Polygon):
                rings.extend(collect_rings(g))
        return rings

    def write_obj(path: Path, rings: list[np.ndarray]) -> None:
        with open(path, "w") as f:
            v_offset = 1
            for ring in rings:
                if len(ring) == 0:
                    continue
                ring_3d = _unproject_from_plane(np.asarray(ring, dtype=float), origin, axis_u, axis_v)
                for v in ring_3d:
                    f.write(f"v {v[0]} {v[1]} {v[2]}\n")
                idxs = list(range(v_offset, v_offset + len(ring_3d)))
                f.write("l " + " ".join(str(i) for i in idxs) + f" {v_offset}\n")
                v_offset += len(ring_3d)

    write_obj(debug_dir / "polygon_offset_outlet_boundary.obj", [r for r in loops_2d])
    write_obj(debug_dir / "polygon_offset_inset.obj", collect_rings(poly_in))
    write_obj(debug_dir / "polygon_offset_ring.obj", collect_rings(poly_ring))
    logger.debug("Wrote polygon_offset debug curves to %s", debug_dir)


# ---------------------------------------------------------------------------
# Medial-axis outlet split
# ---------------------------------------------------------------------------

def _resample_loop_2d(loop_2d: np.ndarray, ds: float) -> np.ndarray:
    """Uniformly resample a closed 2D polyline at spacing ~ds (min 16 points)."""
    pts = np.asarray(loop_2d, float)
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0:1]])
    seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg_len)])
    n = max(int(round(s[-1] / ds)), 16)
    s_new = np.linspace(0.0, s[-1], n, endpoint=False)
    xs = np.interp(s_new, s, pts[:, 0])
    ys = np.interp(s_new, s, pts[:, 1])
    return np.column_stack([xs, ys])


def _voronoi_medial_axis(poly, ds: float, min_dist: float):
    """Compute a Voronoi-based medial axis of a shapely (Multi)Polygon.

    Dense-samples the boundary at spacing ``ds``, builds the Voronoi diagram,
    keeps edges interior to ``poly`` whose endpoints are at least ``min_dist``
    from the boundary (drops the dense same-side fringe), and merges into one
    or more polylines. Returns a shapely MultiLineString (possibly empty).
    """
    import shapely
    from shapely.geometry import LineString, MultiLineString, MultiPoint, Point
    from shapely.ops import linemerge, unary_union, voronoi_diagram

    polys = list(poly.geoms) if isinstance(poly, shapely.MultiPolygon) else [poly]
    sample_pts: list[np.ndarray] = []
    for p in polys:
        for ring in [p.exterior, *p.interiors]:
            coords = np.asarray(ring.coords, float)[:, :2]
            sample_pts.append(_resample_loop_2d(coords, ds))
    if not sample_pts:
        return MultiLineString()
    all_pts = np.vstack(sample_pts)
    mp = MultiPoint([(x, y) for x, y in all_pts])

    vd = voronoi_diagram(mp, envelope=poly.envelope, edges=True)
    # shapely returns a GeometryCollection containing a single MultiLineString
    # of all Voronoi edges; clip it to the polygon directly.
    clipped = vd.intersection(poly)
    if clipped.is_empty:
        return MultiLineString()
    if isinstance(clipped, LineString):
        inside = [clipped]
    elif isinstance(clipped, MultiLineString):
        inside = list(clipped.geoms)
    else:
        inside = [g for g in getattr(clipped, "geoms", []) if isinstance(g, LineString)]
    if not inside:
        return MultiLineString()

    # Fringe filter: keep only edges whose endpoints are well inside the polygon.
    boundary = poly.boundary
    kept = [
        ls for ls in inside
        if all(boundary.distance(Point(c)) >= min_dist for c in ls.coords)
    ]
    if not kept:
        return MultiLineString()

    merged = linemerge(unary_union(kept))
    if isinstance(merged, LineString):
        merged = MultiLineString([merged])
    return merged


def _prune_short_polylines(ml, min_len: float):
    """Drop polylines shorter than min_len (returns empty MultiLineString if all removed)."""
    from shapely.geometry import MultiLineString

    if ml.is_empty:
        return ml
    kept = [g for g in ml.geoms if g.length >= min_len]
    return MultiLineString(kept)


def _build_medial_axis_outlet_regions(
    outlet_triangles: list[np.ndarray],
    source_patch_name: str,
    interior_patch_name: str,
    boundary_sample_ds: float,
    strip_half_width: float,
    min_dist_from_boundary: float,
    prune_branch_len: float = 1.0,
    triangulation_engine: str = "triangle",
    debug_dir: Path | None = None,
) -> dict[str, list[np.ndarray]]:
    """Split the outlet using a buffered medial-axis strip as the interior patch.

    The medial axis (Voronoi-based) of the outlet polygon is extracted, pruned,
    buffered by ``strip_half_width`` and clipped to the outlet to form the
    interior patch; the rim is the outlet minus the strip. Both regions are
    re-triangulated to the smooth strip boundary.

    Use ``min_dist_from_boundary`` slightly below the channel's narrowest local
    half-width (otherwise the medial line gets cut into disconnected pieces).
    """
    import shapely

    if strip_half_width <= 0:
        raise ValueError(f"medial_axis: strip_half_width must be positive, got {strip_half_width}.")
    if boundary_sample_ds <= 0:
        raise ValueError(f"medial_axis: boundary_sample_ds must be positive, got {boundary_sample_ds}.")
    if min_dist_from_boundary <= 0:
        raise ValueError(f"medial_axis: min_dist_from_boundary must be positive, got {min_dist_from_boundary}.")

    triangles = np.asarray(outlet_triangles, dtype=float)
    loops_3d = _all_boundary_loops_from_triangles(triangles)
    if not loops_3d:
        raise ValueError("medial_axis: no boundary loops on outlet patch.")

    all_loop_pts = np.vstack(loops_3d)
    origin, normal, axis_u, axis_v = _make_plane_basis(all_loop_pts, triangles)
    loops_2d = [_project_to_plane(l, origin, axis_u, axis_v) for l in loops_3d]
    poly_outlet = _build_shapely_multipolygon(loops_2d)

    medial = _voronoi_medial_axis(poly_outlet, ds=boundary_sample_ds, min_dist=min_dist_from_boundary)
    medial = _prune_short_polylines(medial, prune_branch_len)
    if medial.is_empty:
        raise ValueError(
            f"medial_axis: extraction produced an empty medial axis "
            f"(min_dist_from_boundary={min_dist_from_boundary} too large, or channel too narrow)."
        )

    strip_2d = medial.buffer(strip_half_width, join_style="round", cap_style="round")
    strip_2d = strip_2d.intersection(poly_outlet)
    if strip_2d.is_empty or strip_2d.area <= 0:
        raise ValueError(
            f"medial_axis: buffered strip is empty (strip_half_width={strip_half_width})."
        )
    ring_2d = poly_outlet.difference(strip_2d)

    interior_tris_3d = _lift_2d_triangles_to_3d(
        _triangulate_shapely_to_2d(strip_2d, engine=triangulation_engine),
        origin, axis_u, axis_v, normal,
    )
    rim_tris_3d = _lift_2d_triangles_to_3d(
        _triangulate_shapely_to_2d(ring_2d, engine=triangulation_engine),
        origin, axis_u, axis_v, normal,
    )

    total_medial_len = sum(g.length for g in medial.geoms)
    try:
        clearance = float(strip_2d.boundary.distance(poly_outlet.boundary))
    except Exception:
        clearance = float("nan")
    logger.info(
        "Created %s via medial_axis: medial polylines=%d (len=%.3e), "
        "outlet_area=%.6e, strip_area=%.6e (%.1f%%), rim_area=%.6e, "
        "wall_clearance=%.4e, %s_tris=%d, %s_tris=%d",
        interior_patch_name,
        len(medial.geoms),
        total_medial_len,
        poly_outlet.area,
        strip_2d.area,
        100.0 * strip_2d.area / poly_outlet.area,
        ring_2d.area,
        clearance,
        interior_patch_name,
        len(interior_tris_3d),
        source_patch_name,
        len(rim_tris_3d),
    )

    if debug_dir is not None:
        _write_medial_axis_debug(
            debug_dir, poly_outlet, medial, strip_2d,
            origin, axis_u, axis_v,
            params=dict(
                boundary_sample_ds=boundary_sample_ds,
                strip_half_width=strip_half_width,
                min_dist_from_boundary=min_dist_from_boundary,
                prune_branch_len=prune_branch_len,
            ),
            clearance=clearance,
            medial_length=total_medial_len,
        )

    return {
        source_patch_name: rim_tris_3d,
        interior_patch_name: interior_tris_3d,
    }


def _write_medial_axis_debug(
    debug_dir: Path,
    poly_outlet,
    medial,
    strip_2d,
    origin: np.ndarray,
    axis_u: np.ndarray,
    axis_v: np.ndarray,
    params: dict[str, float],
    clearance: float,
    medial_length: float,
) -> None:
    """Write the medial-axis polyline (OBJ in 3D) and a PNG overview+zoom."""
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Medial axis as a 3D OBJ polyline
    obj_path = debug_dir / "medial_axis.obj"
    with open(obj_path, "w") as f:
        v_offset = 1
        for g in medial.geoms:
            coords_2d = np.asarray(g.coords, dtype=float)
            pts_3d = _unproject_from_plane(coords_2d, origin, axis_u, axis_v)
            for v in pts_3d:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for i in range(len(pts_3d) - 1):
                f.write(f"l {v_offset + i} {v_offset + i + 1}\n")
            v_offset += len(pts_3d)

    _render_medial_axis_png(
        debug_dir / "medial_axis.png",
        poly_outlet, medial, strip_2d,
        params=params, clearance=clearance, medial_length=medial_length,
    )


def _render_medial_axis_png(
    path: Path,
    poly,
    medial,
    strip,
    params: dict[str, float],
    clearance: float,
    medial_length: float,
    zoom_size_mm: float = 4.0,
) -> None:
    """Render an overview + zoom of the outlet split (matplotlib, Agg backend)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import PathPatch, Rectangle
        from matplotlib.path import Path as MplPath
    except ImportError:
        logger.warning("matplotlib not available; skipping medial_axis.png debug render.")
        return

    def polygon_to_patch(p, **kw):
        verts, codes = [], []
        def add_ring(coords):
            cs = list(coords)
            verts.extend(cs)
            codes.append(MplPath.MOVETO)
            codes.extend([MplPath.LINETO] * (len(cs) - 2))
            codes.append(MplPath.CLOSEPOLY)
        polys = list(p.geoms) if p.geom_type == "MultiPolygon" else [p]
        for q in polys:
            add_ring(q.exterior.coords)
            for h in q.interiors:
                add_ring(h.coords)
        return PathPatch(MplPath(verts, codes), **kw)

    def draw_scene(ax, lw_scale=1.0):
        ax.add_patch(polygon_to_patch(poly, facecolor="#dddddd", edgecolor="none", zorder=0))
        if not strip.is_empty:
            ax.add_patch(polygon_to_patch(
                strip, facecolor="#ffc8a8", edgecolor="#d2691e",
                lw=0.6 * lw_scale, zorder=1, label="strip",
            ))
        polys = list(poly.geoms) if poly.geom_type == "MultiPolygon" else [poly]
        for q in polys:
            ex = np.asarray(q.exterior.coords)
            ax.plot(ex[:, 0], ex[:, 1], "-", color="#1f77b4", lw=0.8 * lw_scale, label="outer wall")
            for h in q.interiors:
                hc = np.asarray(h.coords)
                ax.plot(hc[:, 0], hc[:, 1], "-", color="#2ca02c", lw=0.8 * lw_scale, label="inner wall")
        for g in medial.geoms:
            c = np.asarray(g.coords)
            ax.plot(c[:, 0], c[:, 1], "-", color="#d62728", lw=1.2 * lw_scale, label="medial axis")

    minx, miny, maxx, maxy = poly.bounds
    cx, cy = 0.5 * (minx + maxx), miny + 0.45 * (maxy - miny)
    medial_pts = np.vstack([np.asarray(g.coords) for g in medial.geoms])
    nearest = medial_pts[np.argmin(np.hypot(medial_pts[:, 0] - cx, medial_pts[:, 1] - cy))]
    half = 0.5 * zoom_size_mm

    title = (
        f"ds={params['boundary_sample_ds']}, min_dist={params['min_dist_from_boundary']}, "
        f"strip_w={params['strip_half_width']}, prune<{params['prune_branch_len']}  |  "
        f"medial {medial_length:.1f} mm, strip {100 * strip.area / poly.area:.1f}%, "
        f"clearance {clearance:.3f} mm"
    )

    fig, (ax, axz) = plt.subplots(
        2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [1.0, 1.4]}
    )
    draw_scene(ax)
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    ax.legend(seen.values(), seen.keys(), loc="lower right", fontsize=8)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("u [mm]")
    ax.set_ylabel("v [mm]")
    ax.add_patch(Rectangle(
        (nearest[0] - half, nearest[1] - half), zoom_size_mm, zoom_size_mm,
        fill=False, edgecolor="0.2", lw=1.0, zorder=3,
    ))

    draw_scene(axz, lw_scale=1.5)
    axz.set_xlim(nearest[0] - half, nearest[0] + half)
    axz.set_ylim(nearest[1] - half, nearest[1] + half)
    axz.set_aspect("equal")
    axz.set_title(f"zoom: {zoom_size_mm:.1f} x {zoom_size_mm:.1f} mm window", fontsize=10)
    axz.set_xlabel("u [mm]")
    axz.set_ylabel("v [mm]")
    axz.grid(True, linestyle=":", color="0.7", lw=0.4)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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
            is_cap |= on_plane

    logger.debug("_remove_box_caps removed %d / %d faces (tol=%s)", is_cap.sum(), len(mesh.faces), tol)
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

    logger.info("stitch diagnostics:")
    for key, value in stitch_diagnostics(stitched).items():
        logger.info("  %s: %s", key, value)

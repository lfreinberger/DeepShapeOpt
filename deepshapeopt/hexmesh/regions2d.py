"""2-D regions of a planar cap cross-section: plane basis, boundary loops, medial axis.

Shared by the outlet-interior carve (:mod:`deepshapeopt.hexmesh.patches`), the cap
handling (:mod:`deepshapeopt.hexmesh.caps`) and the undercut silhouette gate.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def make_plane_basis(points: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

def project_to_plane(points: np.ndarray, origin: np.ndarray, axis_u: np.ndarray, axis_v: np.ndarray) -> np.ndarray:
    rel = points - origin
    return np.column_stack([rel @ axis_u, rel @ axis_v])

def unproject_from_plane(points_2d: np.ndarray, origin: np.ndarray, axis_u: np.ndarray, axis_v: np.ndarray) -> np.ndarray:
    return origin + points_2d[:, 0:1] * axis_u + points_2d[:, 1:2] * axis_v

def inplane_world_projector(origin: np.ndarray, axis_u: np.ndarray, axis_v: np.ndarray):
    """Projector for rendering debug plots in the patch's own plane.

    Returns ``(project, xlabel, ylabel)`` where ``project(coords_uv)`` lifts plane
    ``(u, v)`` coordinates to 3D and returns the two world coordinates that span the
    patch plane (the axes other than the plane normal, in ascending index order).

    This keeps the plot in the patch's own axis plane whatever its orientation --
    an x-normal outlet renders in world (y, z), a z-normal cap in world (x, y) --
    instead of a hardcoded world (y, z) view that collapses for non-x-normal patches.
    """
    normal = np.cross(np.asarray(axis_u, dtype=float), np.asarray(axis_v, dtype=float))
    k = int(np.argmax(np.abs(normal)))
    i, j = (a for a in (0, 1, 2) if a != k)
    names = "xyz"

    def project(coords_uv: np.ndarray) -> np.ndarray:
        pts3d = unproject_from_plane(np.asarray(coords_uv, dtype=float), origin, axis_u, axis_v)
        return np.column_stack([pts3d[:, i], pts3d[:, j]])

    return project, f"{names[i]} [mm]", f"{names[j]} [mm]"

def all_boundary_loops_from_triangles(triangles: np.ndarray, tol: float = 1e-8) -> list[np.ndarray]:
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

def build_shapely_multipolygon(loops_2d: list[np.ndarray]):
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

def write_polygon_offset_debug(
    debug_dir: Path,
    loops_2d: list[np.ndarray],
    poly_outlet,
    poly_in,
    poly_ring,
    origin: np.ndarray,
    axis_u: np.ndarray,
    axis_v: np.ndarray,
    inset_distance: float,
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
                ring_3d = unproject_from_plane(np.asarray(ring, dtype=float), origin, axis_u, axis_v)
                for v in ring_3d:
                    f.write(f"v {v[0]} {v[1]} {v[2]}\n")
                idxs = list(range(v_offset, v_offset + len(ring_3d)))
                f.write("l " + " ".join(str(i) for i in idxs) + f" {v_offset}\n")
                v_offset += len(ring_3d)

    write_obj(debug_dir / "polygon_offset_outlet_boundary.obj", [r for r in loops_2d])
    write_obj(debug_dir / "polygon_offset_inset.obj", collect_rings(poly_in))
    write_obj(debug_dir / "polygon_offset_ring.obj", collect_rings(poly_ring))
    render_polygon_offset_png(
        debug_dir / "polygon_offset.png",
        poly_outlet, poly_in, poly_ring,
        origin=origin, axis_u=axis_u, axis_v=axis_v,
        inset_distance=float(inset_distance),
    )
    logger.debug("Wrote polygon_offset debug curves to %s", debug_dir)

def render_polygon_offset_png(
    path: Path,
    poly_outlet,
    poly_in,
    poly_ring,
    origin: np.ndarray,
    axis_u: np.ndarray,
    axis_v: np.ndarray,
    inset_distance: float,
) -> None:
    """Render overview of the outlet interior split in the patch's own axis plane."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import PathPatch
        from matplotlib.path import Path as MplPath
    except ImportError:
        logger.warning("matplotlib not available; skipping polygon_offset.png debug render.")
        return

    to_world_yz, xlabel, ylabel = inplane_world_projector(origin, axis_u, axis_v)

    def polygon_to_patch(p, **kw):
        verts, codes = [], []
        def add_ring(coords):
            cs = list(to_world_yz(np.asarray(coords)))
            verts.extend(cs)
            codes.append(MplPath.MOVETO)
            codes.extend([MplPath.LINETO] * (len(cs) - 2))
            codes.append(MplPath.CLOSEPOLY)
        polys = list(p.geoms) if p.geom_type == "MultiPolygon" else [p]
        for q in polys:
            if q.is_empty:
                continue
            add_ring(q.exterior.coords)
            for h in q.interiors:
                add_ring(h.coords)
        if not verts:
            return None
        return PathPatch(MplPath(verts, codes), **kw)

    fig, ax = plt.subplots(figsize=(14, 6))
    ring_patch = polygon_to_patch(
        poly_ring, facecolor="#dddddd", edgecolor="none", zorder=0, label="outlet",
    )
    if ring_patch is not None:
        ax.add_patch(ring_patch)
    in_patch = polygon_to_patch(
        poly_in, facecolor="#ffc8a8", edgecolor="#d2691e",
        lw=0.6, zorder=1, label="outletInterior",
    )
    if in_patch is not None:
        ax.add_patch(in_patch)
    polys = list(poly_outlet.geoms) if poly_outlet.geom_type == "MultiPolygon" else [poly_outlet]
    for q in polys:
        ex = to_world_yz(np.asarray(q.exterior.coords))
        ax.plot(ex[:, 0], ex[:, 1], "-", color="#1f77b4", lw=0.8)
        for h in q.interiors:
            hc = to_world_yz(np.asarray(h.coords))
            ax.plot(hc[:, 0], hc[:, 1], "-", color="#2ca02c", lw=0.8)
    in_polys = list(poly_in.geoms) if poly_in.geom_type == "MultiPolygon" else [poly_in]
    for q in in_polys:
        if q.is_empty:
            continue
        ex = to_world_yz(np.asarray(q.exterior.coords))
        ax.plot(ex[:, 0], ex[:, 1], "-", color="#d62728", lw=1.0)
        for h in q.interiors:
            hc = to_world_yz(np.asarray(h.coords))
            ax.plot(hc[:, 0], hc[:, 1], "-", color="#d62728", lw=1.0)

    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    ax.legend(seen.values(), seen.keys(), loc="lower right", fontsize=8)
    ax.set_aspect("equal")
    title = (
        f"offset = {inset_distance:g} mm"
        if np.isfinite(inset_distance)
        else "outlet interior split"
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def resample_loop_2d(loop_2d: np.ndarray, ds: float) -> np.ndarray:
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

def voronoi_medial_axis(poly, ds: float, min_dist: float):
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
            sample_pts.append(resample_loop_2d(coords, ds))
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

def prune_short_polylines(ml, min_len: float):
    """Drop polylines shorter than min_len (returns empty MultiLineString if all removed)."""
    from shapely.geometry import MultiLineString

    if ml.is_empty:
        return ml
    kept = [g for g in ml.geoms if g.length >= min_len]
    return MultiLineString(kept)

def write_medial_axis_debug(
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
            pts_3d = unproject_from_plane(coords_2d, origin, axis_u, axis_v)
            for v in pts_3d:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for i in range(len(pts_3d) - 1):
                f.write(f"l {v_offset + i} {v_offset + i + 1}\n")
            v_offset += len(pts_3d)

    render_medial_axis_pngs(
        debug_dir,
        poly_outlet, medial, strip_2d,
        origin=origin, axis_u=axis_u, axis_v=axis_v,
        strip_half_width=float(params["strip_half_width"]),
    )

def render_medial_axis_pngs(
    debug_dir: Path,
    poly,
    medial,
    strip,
    origin: np.ndarray,
    axis_u: np.ndarray,
    axis_v: np.ndarray,
    strip_half_width: float,
) -> None:
    """Render three PNGs (outlet-only, medial-axis-only, combined) in the patch's own axis plane."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import PathPatch
        from matplotlib.path import Path as MplPath
    except ImportError:
        logger.warning("matplotlib not available; skipping medial_axis PNG debug renders.")
        return

    to_world_yz, xlabel, ylabel = inplane_world_projector(origin, axis_u, axis_v)

    def polygon_to_patch(p, **kw):
        verts, codes = [], []
        def add_ring(coords):
            cs = list(to_world_yz(np.asarray(coords)))
            verts.extend(cs)
            codes.append(MplPath.MOVETO)
            codes.extend([MplPath.LINETO] * (len(cs) - 2))
            codes.append(MplPath.CLOSEPOLY)
        polys = list(p.geoms) if p.geom_type == "MultiPolygon" else [p]
        for q in polys:
            if q.is_empty:
                continue
            add_ring(q.exterior.coords)
            for h in q.interiors:
                add_ring(h.coords)
        if not verts:
            return None
        return PathPatch(MplPath(verts, codes), **kw)

    def draw_outlet(ax) -> None:
        rim_patch = polygon_to_patch(
            poly, facecolor="#dddddd", edgecolor="none", zorder=0, label="outlet",
        )
        if rim_patch is not None:
            ax.add_patch(rim_patch)
        polys = list(poly.geoms) if poly.geom_type == "MultiPolygon" else [poly]
        for q in polys:
            ex = to_world_yz(np.asarray(q.exterior.coords))
            ax.plot(ex[:, 0], ex[:, 1], "-", color="#1f77b4", lw=0.8)
            for h in q.interiors:
                hc = to_world_yz(np.asarray(h.coords))
                ax.plot(hc[:, 0], hc[:, 1], "-", color="#2ca02c", lw=0.8)

    def draw_medial(ax) -> None:
        first = True
        for g in medial.geoms:
            c = to_world_yz(np.asarray(g.coords))
            ax.plot(
                c[:, 0], c[:, 1], "-", color="#d62728", lw=1.2,
                label="medial axis" if first else None,
            )
            first = False

    def draw_strip(ax) -> None:
        strip_patch = polygon_to_patch(
            strip, facecolor="#ffc8a8", edgecolor="#d2691e",
            lw=0.6, zorder=1, label="outletInterior",
        )
        if strip_patch is not None:
            ax.add_patch(strip_patch)

    # Shared frame: bounds and decoration applied to every figure.
    polys = list(poly.geoms) if poly.geom_type == "MultiPolygon" else [poly]
    outlet_pts = np.vstack([to_world_yz(np.asarray(q.exterior.coords)) for q in polys])
    xmin, ymin = outlet_pts.min(axis=0)
    xmax, ymax = outlet_pts.max(axis=0)
    pad = 0.02 * max(xmax - xmin, ymax - ymin, 1e-9)

    def finalize(fig, ax, title: str, out: Path) -> None:
        handles, labels = ax.get_legend_handles_labels()
        seen = {}
        for h, l in zip(handles, labels):
            seen.setdefault(l, h)
        if seen:
            ax.legend(seen.values(), seen.keys(), loc="lower right", fontsize=8)
        ax.set_aspect("equal")
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)

    title = f"strip half width = {strip_half_width:g} mm"

    fig, ax = plt.subplots(figsize=(14, 6))
    draw_outlet(ax)
    finalize(fig, ax, "outlet", debug_dir / "medial_axis_outlet.png")

    fig, ax = plt.subplots(figsize=(14, 6))
    draw_outlet(ax)
    draw_medial(ax)
    finalize(fig, ax, "medial axis", debug_dir / "medial_axis_only.png")

    fig, ax = plt.subplots(figsize=(14, 6))
    draw_outlet(ax)
    draw_strip(ax)
    draw_medial(ax)
    finalize(fig, ax, title, debug_dir / "medial_axis.png")

"""Outlet sub-patch (outletInterior) carve for the hex mesh pipeline.

Computes the interior region of the outlet cross-section once from the
original STL's outlet cap -- using the same medial-axis / polygon-offset
machinery as the snappy pipeline (:mod:`deepshapeopt.mesh`) -- and returns
a vectorized centroid classifier for :class:`~deepshapeopt.hexmesh.polymesh.
PatchPlan.face_subpatch`.  The classifier runs on the final face set of
every build, so patch membership follows refinement changes even though
the cross-section itself is fixed (the design domain does not reach the
outlet plane).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np

from deepshapeopt.mesh import (
    _all_boundary_loops_from_triangles,
    _build_shapely_multipolygon,
    _make_plane_basis,
    _project_to_plane,
    _prune_short_polylines,
    _voronoi_medial_axis,
    _write_medial_axis_debug,
    _write_polygon_offset_debug,
)

logger = logging.getLogger(__name__)


def outlet_cap_triangles(
    mesh_orig, plane_axis: int, plane_value: float, plane_tol: float
) -> np.ndarray:
    """Triangles [T, 3, 3] of ``mesh_orig`` lying on the outlet plane."""
    vertices = np.asarray(mesh_orig.vertices, dtype=np.float64)
    faces = np.asarray(mesh_orig.faces, dtype=np.int64)
    tris = vertices[faces]
    on_plane = np.abs(tris[:, :, plane_axis].mean(axis=1) - plane_value) < plane_tol
    if not np.any(on_plane):
        raise ValueError(
            f"No outlet cap triangles found on axis {plane_axis} at "
            f"{plane_value} (tol {plane_tol})."
        )
    return tris[on_plane]


def outlet_strip_classifier(
    mesh_orig,
    plane_axis: int,
    plane_value: float,
    plane_tol: float,
    outlet_interior_cfg: dict,
    debug_dir: Path | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build the outletInterior centroid classifier from the STL outlet cap.

    ``outlet_interior_cfg`` uses the same schema as the snappy pipeline's
    ``optimization.outlet_interior`` block (``method``, ``strip_half_width``,
    ``boundary_sample_ds``, ``min_dist_from_boundary``, ``prune_branch_len``
    for ``medial_axis``; ``inset_distance``, ``quad_segs``, ``join_style``
    for ``polygon_offset``).

    Returns a callable mapping face centroids [N, 3] (physical) to a bool
    mask of faces inside the interior region.
    """
    triangles = outlet_cap_triangles(mesh_orig, plane_axis, plane_value, plane_tol)
    return strip_classifier_from_triangles(
        triangles, outlet_interior_cfg, debug_dir=debug_dir
    )


def strip_classifier_from_triangles(
    triangles: np.ndarray,
    outlet_interior_cfg: dict,
    debug_dir: Path | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """As :func:`outlet_strip_classifier`, but from pre-selected cap
    triangles [T, 3, 3] (also used per plane for interior-cap sources)."""
    import shapely

    interior_2d, origin, axis_u, axis_v = interior_region_2d(
        triangles, outlet_interior_cfg, debug_dir=debug_dir
    )
    shapely.prepare(interior_2d)

    def classify(centroids: np.ndarray) -> np.ndarray:
        centroids = np.asarray(centroids, dtype=np.float64).reshape(-1, 3)
        uv = _project_to_plane(centroids, origin, axis_u, axis_v)
        return shapely.contains_xy(interior_2d, uv[:, 0], uv[:, 1])

    return classify


def interior_region_2d(
    triangles: np.ndarray,
    outlet_interior_cfg: dict,
    debug_dir: Path | None = None,
):
    """2D interior region of a cap cross-section plus its plane basis.

    Returns ``(interior_2d, origin, axis_u, axis_v)``.
    """
    cfg = outlet_interior_cfg
    method = str(cfg.get("method", "medial_axis"))

    loops_3d = _all_boundary_loops_from_triangles(triangles)
    if not loops_3d:
        raise ValueError("outlet_strip_classifier: no boundary loops on the outlet cap.")
    all_loop_pts = np.vstack(loops_3d)
    origin, normal, axis_u, axis_v = _make_plane_basis(all_loop_pts, triangles)
    loops_2d = [_project_to_plane(l, origin, axis_u, axis_v) for l in loops_3d]
    poly_outlet = _build_shapely_multipolygon(loops_2d)

    if method == "medial_axis":
        ds = float(cfg.get("boundary_sample_ds", 0.05))
        half_width = float(cfg["strip_half_width"])
        min_dist = float(cfg.get("min_dist_from_boundary", 0.3))
        prune_len = float(cfg.get("prune_branch_len", 1.0))
        medial = _voronoi_medial_axis(poly_outlet, ds=ds, min_dist=min_dist)
        medial = _prune_short_polylines(medial, prune_len)
        if medial.is_empty:
            raise ValueError(
                "outlet_strip_classifier: empty medial axis "
                f"(min_dist_from_boundary={min_dist} too large, or channel too narrow)."
            )
        interior_2d = medial.buffer(
            half_width, join_style="round", cap_style="round"
        ).intersection(poly_outlet)
        if interior_2d.is_empty or interior_2d.area <= 0:
            raise ValueError(
                f"outlet_strip_classifier: empty strip (strip_half_width={half_width})."
            )
        logger.info(
            "outletInterior strip (medial_axis): outlet_area=%.6e, "
            "strip_area=%.6e (%.1f%%)",
            poly_outlet.area, interior_2d.area,
            100.0 * interior_2d.area / poly_outlet.area,
        )
        if debug_dir is not None:
            _write_medial_axis_debug(
                Path(debug_dir), poly_outlet, medial, interior_2d,
                origin, axis_u, axis_v,
                params=dict(
                    boundary_sample_ds=ds,
                    strip_half_width=half_width,
                    min_dist_from_boundary=min_dist,
                    prune_branch_len=prune_len,
                ),
                clearance=float("nan"),
                medial_length=sum(g.length for g in medial.geoms),
            )
    elif method == "polygon_offset":
        inset = float(cfg["inset_distance"])
        interior_2d = poly_outlet.buffer(
            -inset,
            quad_segs=int(cfg.get("quad_segs", 8)),
            join_style=str(cfg.get("join_style", "round")),
        )
        if interior_2d.is_empty or interior_2d.area <= 0:
            raise ValueError(
                f"outlet_strip_classifier: inset_distance={inset} erased the outlet."
            )
        logger.info(
            "outletInterior strip (polygon_offset): outlet_area=%.6e, "
            "inset_area=%.6e (%.1f%%)",
            poly_outlet.area, interior_2d.area,
            100.0 * interior_2d.area / poly_outlet.area,
        )
        if debug_dir is not None:
            _write_polygon_offset_debug(
                Path(debug_dir), loops_2d, poly_outlet, interior_2d,
                poly_outlet.difference(interior_2d),
                origin, axis_u, axis_v, inset_distance=inset,
            )
    else:
        raise ValueError(
            f"outlet_interior.method {method!r} not supported by the hex "
            "pipeline (expected 'medial_axis' or 'polygon_offset')."
        )

    return interior_2d, origin, axis_u, axis_v

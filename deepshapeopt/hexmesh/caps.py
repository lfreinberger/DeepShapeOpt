"""Cap-plane resolution, validation and carve classifiers for ``sdf_hex``.

A *cap* is a flat, axis-aligned plane where the fluid channel of an
internal-flow geometry ends: an inlet/outlet cross-section of the closed
(watertight) fluid solid.  Historically the pipeline supported a single
``cap_axis`` whose min/max planes were dropped from the wall SDF and had
to coincide with domain box faces.  The explicit ``sdf_hex.caps`` config
generalizes this to any number of planes on any axes, each mapped to a
named patch, and the planes need not lie on the domain box ("interior
caps", e.g. outlet slots where a channel discharges sideways).

Treatment is chosen automatically per cap:

- **domain-face cap** (``|value - domain plane| <= tol``): the cap
  triangles are *dropped* from the wall SDF (no zero level set, no
  refinement band, no snapping at the plane); the domain clip terminates
  the mesh and ``_classify_boundary`` names the faces.
- **interior cap**: the triangles are *kept* in the wall SDF, which seals
  the channel flush at the plane; the castellated boundary faces on the
  plane are reassigned to the cap's patch by a wall-carve classifier
  (:mod:`deepshapeopt.hexmesh.polymesh`) and their points participate in
  the snap, landing exactly on the (planar) cap surface.
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np

from deepshapeopt.mesh import (
    _all_boundary_loops_from_triangles,
    _build_shapely_multipolygon,
    _make_plane_basis,
    _project_to_plane,
)

logger = logging.getLogger(__name__)

_AXES = {"x": 0, "y": 1, "z": 2}
_AXIS_NAMES = "xyz"
_NORMAL_ALIGN = 0.99
_CAP_KEYS = {"axis", "value", "patch", "tol"}

# OpenFOAM constraint patch types through which the geometry may
# intentionally be truncated by the domain box (mirror-symmetric halves).
CONSTRAINT_TYPES = ("symmetry", "symmetryPlane")


@dataclasses.dataclass
class CapSpec:
    """One resolved cap plane."""

    axis: int
    value: float
    patch: str
    tol: float
    treatment: str  # "drop" (domain-face cap) | "keep" (interior cap)
    triangles: np.ndarray  # [T, 3, 3] STL cap triangles on the plane
    domain_side: int | None = None  # 0/1 for domain-face caps, None interior
    _polygon: object = dataclasses.field(default=None, repr=False)
    _basis: tuple | None = dataclasses.field(default=None, repr=False)

    def polygon(self):
        """Shapely (multi)polygon of the cap cross-section and its plane
        basis ``(origin, axis_u, axis_v)``, built lazily from the STL cap
        triangles."""
        if self._polygon is None:
            loops_3d = _all_boundary_loops_from_triangles(self.triangles)
            if not loops_3d:
                raise ValueError(
                    f"Cap {self.patch!r} on axis {self.axis} at {self.value}: "
                    "no boundary loops on the cap triangles."
                )
            all_pts = np.vstack(loops_3d)
            origin, _normal, axis_u, axis_v = _make_plane_basis(all_pts, self.triangles)
            loops_2d = [_project_to_plane(l, origin, axis_u, axis_v) for l in loops_3d]
            self._polygon = _build_shapely_multipolygon(loops_2d)
            self._basis = (origin, axis_u, axis_v)
        return self._polygon, self._basis


def _face_normals(tris: np.ndarray) -> np.ndarray:
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)


def planar_clusters(
    vertices: np.ndarray, faces: np.ndarray, axis: int,
    min_tris: int = 2, max_clusters: int = 10,
) -> list[tuple[float, int]]:
    """Diagnostic: (plane value, triangle count) of axis-aligned planar
    clusters -- the candidate cap planes of the geometry along ``axis``.
    The ``max_clusters`` largest clusters, sorted by count descending."""
    tris = vertices[faces]
    aligned = np.abs(_face_normals(tris)[:, axis]) > _NORMAL_ALIGN
    c = tris[:, :, axis].mean(axis=1)[aligned]
    vals, counts = np.unique(np.round(c, 4), return_counts=True)
    keep = counts >= min_tris
    order = np.argsort(-counts[keep])[:max_clusters]
    return list(zip(vals[keep][order].tolist(), counts[keep][order].tolist()))


def resolve_caps(
    caps_cfg: list[dict],
    domain: np.ndarray,
    design_domain: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    default_tol: float,
    interface_width: float,
) -> list[CapSpec]:
    """Parse and validate ``sdf_hex.caps`` against the STL geometry.

    ``interface_width`` is the interface cell size ``h0 / 2**interface_level``;
    an interior cap closer than two interface cells to a parallel domain
    face is rejected (the castellation cannot resolve the sliver band, and
    the user almost certainly meant a domain-face cap).
    """
    domain = np.asarray(domain, dtype=np.float64)
    design_domain = np.asarray(design_domain, dtype=np.float64)
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    tris = vertices[faces]
    centroids = tris.mean(axis=1)
    normals = _face_normals(tris)

    caps: list[CapSpec] = []
    for i, entry in enumerate(caps_cfg):
        unknown = set(entry) - _CAP_KEYS
        if unknown:
            raise ValueError(
                f"caps[{i}] has unknown keys {sorted(unknown)}. "
                f"Valid: {sorted(_CAP_KEYS)}"
            )
        axis_name = str(entry["axis"])
        if axis_name not in _AXES:
            raise ValueError(f"caps[{i}].axis must be one of x/y/z, got {axis_name!r}")
        axis = _AXES[axis_name]
        value = float(entry["value"])
        patch = str(entry["patch"])
        tol = float(entry.get("tol", default_tol))

        on_plane = (np.abs(centroids[:, axis] - value) < tol) & (
            np.abs(normals[:, axis]) > _NORMAL_ALIGN
        )
        if not np.any(on_plane):
            clusters = planar_clusters(vertices, faces, axis)
            found = (
                ", ".join(f"{axis_name}={v:g} ({n} tris)" for v, n in clusters)
                or "none"
            )
            raise ValueError(
                f"caps[{i}] ({patch!r}): no cap triangles on axis {axis_name} at "
                f"{value:g} (tol {tol:g}). Planar clusters found on this axis: "
                f"{found}. Correct the cap value or widen its tol."
            )

        d_min = abs(value - domain[0, axis])
        d_max = abs(value - domain[1, axis])
        domain_side: int | None = None
        if min(d_min, d_max) <= tol:
            treatment = "drop"
            domain_side = 0 if d_min <= d_max else 1
        else:
            treatment = "keep"
            if min(d_min, d_max) < 2.0 * interface_width:
                # Not fatal (the sliver beyond the cap is solid, so no cells
                # exist there), but a domain-face cap is cheaper: no zero
                # level set, no refinement band along the plane.
                side = "min" if d_min < d_max else "max"
                plane = domain[0 if side == "min" else 1, axis]
                logger.warning(
                    "caps[%d] (%r): interior cap at %s=%g is only %g inside "
                    "the domain face %s_%s=%g; consider moving the domain "
                    "face onto the cap plane to make it a (cheaper) "
                    "domain-face cap.",
                    i, patch, axis_name, value, min(d_min, d_max),
                    axis_name, side, plane,
                )
            # Interior caps must live in the fixed (STL) region: inside the
            # design domain the geometry is the differentiable DeepSDF and a
            # static sealing plane is meaningless.
            if design_domain[0, axis] < value < design_domain[1, axis]:
                other = [a for a in range(3) if a != axis]
                bb_lo = tris[on_plane].reshape(-1, 3).min(axis=0)
                bb_hi = tris[on_plane].reshape(-1, 3).max(axis=0)
                overlaps = all(
                    bb_hi[a] > design_domain[0, a] and bb_lo[a] < design_domain[1, a]
                    for a in other
                )
                if overlaps:
                    raise ValueError(
                        f"caps[{i}] ({patch!r}): cap plane {axis_name}={value:g} "
                        "lies inside the design domain -- the geometry there is "
                        "differentiable; caps must sit in the fixed region."
                    )

        logger.info(
            "Cap %r: axis %s at %.6g, %d triangles, treatment=%s",
            patch, axis_name, value, int(on_plane.sum()), treatment,
        )
        caps.append(
            CapSpec(
                axis=axis, value=value, patch=patch, tol=tol,
                treatment=treatment, triangles=tris[on_plane],
                domain_side=domain_side,
            )
        )

    # Warn about parallel wall triangles hugging an interior cap plane inside
    # its footprint: the carve predicate selects boundary faces within one
    # local cell of the plane and could pick up such wall faces.
    for cap in caps:
        if cap.treatment != "keep":
            continue
        near = (
            (np.abs(centroids[:, cap.axis] - cap.value) < 4.0 * interface_width)
            & (np.abs(centroids[:, cap.axis] - cap.value) >= cap.tol)
            & (np.abs(normals[:, cap.axis]) > _NORMAL_ALIGN)
        )
        if np.any(near):
            logger.warning(
                "Cap %r: %d wall triangles parallel to the cap plane within "
                "%g of it; if they overlap the cap footprint the carve may "
                "misclassify faces there.",
                cap.patch, int(near.sum()), 4.0 * interface_width,
            )
    return caps


def validate_geometry_containment(
    vertices: np.ndarray,
    domain: np.ndarray,
    exempt_faces: set[tuple[int, int]],
    tol: float = 1e-6,
) -> None:
    """Raise if the STL protrudes past the domain through a non-exempt face.

    A protruding side silently truncates the fluid with an open ``sides``
    boundary.  ``exempt_faces`` is a set of ``(axis, side)`` tuples
    (side 0 = min, 1 = max) that may legitimately truncate the geometry:
    domain-face caps and faces whose patch type is a constraint type
    (symmetry -- the cut plane of a mirror-symmetric half model).
    """
    domain = np.asarray(domain, dtype=np.float64)
    lo = np.asarray(vertices, dtype=np.float64).min(axis=0)
    hi = np.asarray(vertices, dtype=np.float64).max(axis=0)
    for axis in range(3):
        if lo[axis] < domain[0, axis] - tol and (axis, 0) not in exempt_faces:
            raise ValueError(
                f"Geometry protrudes past domain {_AXIS_NAMES[axis]}_min="
                f"{domain[0, axis]:g} (STL reaches {lo[axis]:g}): the fluid "
                "would be truncated with an open 'sides' boundary. Enlarge "
                "the domain, put a domain-face cap there, or declare the "
                "face's patch type as symmetry."
            )
        if hi[axis] > domain[1, axis] + tol and (axis, 1) not in exempt_faces:
            raise ValueError(
                f"Geometry protrudes past domain {_AXIS_NAMES[axis]}_max="
                f"{domain[1, axis]:g} (STL reaches {hi[axis]:g}): the fluid "
                "would be truncated with an open 'sides' boundary. Enlarge "
                "the domain, put a domain-face cap there, or declare the "
                "face's patch type as symmetry."
            )


def cap_subpatch_classifier(caps: list[CapSpec], oi_cfg: dict, debug_dir=None):
    """outletInterior centroid classifier for an interior-cap source patch.

    Builds the 2D interior region per cap plane (same machinery/config
    schema as the domain-face path, :func:`deepshapeopt.hexmesh.patches.
    strip_classifier_from_triangles`) and unions them; each centroid is
    tested against its nearest plane.  Candidate faces already belong to
    the carved cap patch, so no distance gate beyond nearest-plane
    selection is needed.
    """
    import shapely
    from pathlib import Path

    from .patches import interior_region_2d

    planes = []
    for i, cap in enumerate(caps):
        dbg = None
        if debug_dir is not None:
            dbg = Path(debug_dir) if len(caps) == 1 else Path(debug_dir) / f"plane_{i}"
        interior_2d, origin, axis_u, axis_v = interior_region_2d(
            cap.triangles, oi_cfg, debug_dir=dbg
        )
        shapely.prepare(interior_2d)
        planes.append((cap, interior_2d, origin, axis_u, axis_v))

    def classify(centroids: np.ndarray) -> np.ndarray:
        centroids = np.asarray(centroids, dtype=np.float64).reshape(-1, 3)
        dist = np.stack(
            [np.abs(centroids[:, cap.axis] - cap.value) for cap, *_ in planes],
            axis=1,
        )
        nearest = np.argmin(dist, axis=1)
        out = np.zeros(len(centroids), dtype=bool)
        for j, (cap, interior_2d, origin, axis_u, axis_v) in enumerate(planes):
            sel = nearest == j
            if not np.any(sel):
                continue
            uv = _project_to_plane(centroids[sel], origin, axis_u, axis_v)
            out[sel] = shapely.contains_xy(interior_2d, uv[:, 0], uv[:, 1])
        return out

    return classify


def cap_carve_classifier(caps: list[CapSpec], h_fine: float):
    """Wall-carve classifier for one cap patch (one or more planes).

    Contract (see ``PatchPlan.wall_carves``): ``classify(centroids [N, 3],
    corners [N, 4, 3] physical) -> bool mask`` over candidate wall faces.
    Three gates per plane:

    1. coplanarity: all four corners equal along the cap axis (castellated
       boundary faces are axis-normal quads; rejects the perpendicular
       staircase side faces whose centroids also sit near the plane);
    2. distance: ``|corner - value| <= 0.75 * h_face`` with ``h_face`` the
       face's own in-plane extent -- exposed sealing faces lie within half
       a local cell of the plane (a cell is fluid iff its center is beyond
       it), 0.75 adds slack;
    3. footprint: the centroid, projected into the cap's plane basis, lies
       in the cap polygon buffered outward by ``0.5 * h_fine`` (castellated
       rim faces can stick out up to half a fine cell before snapping).
    """
    import shapely

    prepared = []
    for cap in caps:
        poly, (origin, axis_u, axis_v) = cap.polygon()
        buffered = poly.buffer(0.5 * h_fine)
        shapely.prepare(buffered)
        prepared.append((cap, buffered, origin, axis_u, axis_v))

    def classify(centroids: np.ndarray, corners: np.ndarray) -> np.ndarray:
        centroids = np.asarray(centroids, dtype=np.float64)
        corners = np.asarray(corners, dtype=np.float64)
        out = np.zeros(len(centroids), dtype=bool)
        for cap, buffered, origin, axis_u, axis_v in prepared:
            ax = cap.axis
            c_ax = corners[:, :, ax]
            coplanar = np.all(
                np.abs(c_ax - c_ax[:, 0:1]) < 0.25 * h_fine, axis=1
            )
            other = [a for a in range(3) if a != ax]
            ext = corners.max(axis=1) - corners.min(axis=1)
            h_face = np.maximum(ext[:, other[0]], ext[:, other[1]])
            near = np.abs(c_ax[:, 0] - cap.value) <= 0.75 * h_face
            cand = coplanar & near & ~out
            if not np.any(cand):
                continue
            uv = _project_to_plane(centroids[cand], origin, axis_u, axis_v)
            inside = shapely.contains_xy(buffered, uv[:, 0], uv[:, 1])
            idx = np.nonzero(cand)[0]
            out[idx[inside]] = True
        return out

    return classify

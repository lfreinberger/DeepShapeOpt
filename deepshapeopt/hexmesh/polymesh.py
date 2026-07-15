"""polyMesh assembly from a 2:1-balanced octree cell set.

Produces the arrays OpenFOAM needs (points/faces/owner/neighbour/boundary)
with the ordering rules checkMesh enforces:

- internal faces first, sorted by (owner, neighbour) ascending
  ("upper-triangular order"), with owner < neighbour;
- face winding such that the right-hand-rule normal points owner ->
  neighbour (internal) or out of the domain (boundary);
- boundary faces in contiguous per-patch blocks matching the boundary file;
- no unused points.

At 2:1 level transitions the coarse quad is never emitted; the (up to) four
fine quads are, each owned by the coarse cell on one side -- the coarse cell
becomes a polyhedron, exactly like snappyHexMesh/hexRef8 output.  Where a
refined neighbour region is only partially present (cut by the shape), the
missing quadrants are emitted as quarter wall faces so the mesh stays closed.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Callable

import numpy as np

from .lattice import (
    CellIndex,
    CellSet,
    Lattice,
    face_corner_keys,
    face_sample_points_doubled,
    pack_keys,
    unpack_keys,
)

logger = logging.getLogger(__name__)

_INTERNAL = -1

# Default (external flow / drag) patch layout, kept for backward
# compatibility with existing configs and tests.
PATCH_NAMES = ["inlet", "outlet", "sides", "dragObject"]

_FACE_KEYS = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")


_PATCH_TYPES = ("patch", "wall", "symmetry", "symmetryPlane")


@dataclasses.dataclass
class WallCarve:
    """A named patch carved out of the wall faces by a face classifier.

    ``classify(centroids [N, 3], corners [N, 4, 3] physical) -> bool mask``
    runs over the current wall-code faces of every build (an interior cap
    plane of an internal-flow geometry).  Carved patches are boundary type
    "patch" (not wall) but their points participate in the snap.
    """

    name: str
    classify: Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclasses.dataclass
class PatchPlan:
    """Config-driven boundary-patch layout.

    ``domain_faces`` maps each domain box face to a patch name (faces may
    share a name, e.g. all four lateral faces -> "sides").  Faces not on a
    domain plane are walls; ``wall_carves`` reassign wall faces on interior
    cap planes to named patches; wall faces whose centroid lies inside
    ``sensitivity_box`` move to the ``sensitivity_name`` patch (the design
    surface the adjoint differentiates).  ``face_subpatch`` carves a
    sub-patch out of ``subpatch_source`` by face centroid (the
    outletInterior of the extrusion die); it runs on the final per-build
    face set, so membership follows refinement changes.  ``subpatch_source``
    may be a domain-face patch or a wall-carve patch.

    Patch order in the boundary file: domain-face patches (in
    x/y/z-min/max order), then a domain-face-sourced sub-patch, then the
    carve patches, then a carve-sourced sub-patch, then the wall-type
    patches.  The tail from the first carve patch onward is the contiguous
    *snapped* block (``snap_face_slice``); the wall-type patches stay last
    and contiguous (``wall_face_slice``).

    ``patch_types`` optionally overrides the OpenFOAM boundary type of
    non-wall patches (e.g. ``{"symmetry": "symmetry"}`` for the cut plane
    of a mirror-symmetric half model); wall-type patches are always "wall".

    Patches listed in ``required_nonempty`` raise when they receive no
    faces instead of being silently dropped by ``drop_empty``.
    """

    domain_faces: dict[str, str]
    wall_name: str = "dragObject"
    wall_groups: tuple[str, ...] = ("dragObjectGroup",)
    sensitivity_name: str | None = None
    sensitivity_box: np.ndarray | None = None  # (2, 3) physical
    subpatch_name: str | None = None
    subpatch_source: str | None = None
    face_subpatch: Callable[[np.ndarray], np.ndarray] | None = None
    drop_empty: bool = False
    wall_carves: tuple[WallCarve, ...] = ()
    patch_types: dict[str, str] | None = None
    required_nonempty: tuple[str, ...] = ()

    @staticmethod
    def default_external() -> "PatchPlan":
        """The hardcoded drag-case layout (inlet at x_min, outlet at x_max)."""
        return PatchPlan(
            domain_faces={
                "x_min": "inlet", "x_max": "outlet",
                "y_min": "sides", "y_max": "sides",
                "z_min": "sides", "z_max": "sides",
            },
        )

    def __post_init__(self):
        missing = [k for k in _FACE_KEYS if k not in self.domain_faces]
        if missing:
            raise ValueError(f"PatchPlan.domain_faces missing faces: {missing}")
        if (self.subpatch_name is not None) != (self.face_subpatch is not None):
            raise ValueError("subpatch_name and face_subpatch must come together")
        if self.subpatch_name is not None and self.subpatch_source is None:
            raise ValueError("subpatch_source is required with subpatch_name")
        if self.sensitivity_box is not None:
            self.sensitivity_box = np.asarray(self.sensitivity_box, dtype=np.float64)
        reserved = set(self.domain_faces.values()) | set(self.wall_patch_names())
        for carve in self.wall_carves:
            if carve.name in reserved:
                raise ValueError(
                    f"wall-carve patch {carve.name!r} collides with a "
                    "domain-face or wall-type patch name"
                )
        for name, ptype in (self.patch_types or {}).items():
            if ptype not in _PATCH_TYPES:
                raise ValueError(
                    f"patch_types[{name!r}] = {ptype!r}; valid: {_PATCH_TYPES}"
                )
        if self.subpatch_source is not None:
            known = set(self.domain_faces.values()) | self.carve_names()
            if self.subpatch_source not in known:
                raise ValueError(
                    f"subpatch_source {self.subpatch_source!r} is neither a "
                    f"domain-face patch nor a carve patch (known: {sorted(known)})"
                )

    def carve_names(self) -> set[str]:
        return {c.name for c in self.wall_carves}

    def _subpatch_after_carves(self) -> bool:
        return (
            self.subpatch_name is not None
            and self.subpatch_source in self.carve_names()
        )

    def ordered_names(self) -> list[str]:
        """Patch names in boundary-file order (snapped patches last, with
        the wall-type patches at the very end)."""
        names: list[str] = []
        for key in _FACE_KEYS:
            nm = self.domain_faces[key]
            if nm not in names:
                names.append(nm)
        if self.subpatch_name is not None and not self._subpatch_after_carves():
            names.append(self.subpatch_name)
        for carve in self.wall_carves:
            if carve.name not in names:
                names.append(carve.name)
        if self._subpatch_after_carves():
            names.append(self.subpatch_name)
        names.append(self.wall_name)
        if self.sensitivity_name is not None:
            names.append(self.sensitivity_name)
        return names

    def wall_patch_names(self) -> list[str]:
        names = [self.wall_name]
        if self.sensitivity_name is not None:
            names.append(self.sensitivity_name)
        return names

    def snap_patch_names(self) -> list[str]:
        """Patches whose points snap onto the SDF surface: the wall-type
        patches plus interior-cap carve patches (and a carve-sourced
        sub-patch) -- contiguous and last in the boundary file."""
        names: list[str] = []
        for carve in self.wall_carves:
            if carve.name not in names:
                names.append(carve.name)
        if self._subpatch_after_carves():
            names.append(self.subpatch_name)
        names.extend(self.wall_patch_names())
        return names


@dataclasses.dataclass
class PatchSpec:
    name: str
    start: int
    n_faces: int
    type: str = "patch"
    in_groups: tuple[str, ...] = ()


@dataclasses.dataclass
class PolyMeshData:
    points: np.ndarray  # [P, 3] float64
    point_keys: np.ndarray  # [P] packed int64 (sorted ascending)
    faces: np.ndarray  # [F, 4] int64 point ids
    owner: np.ndarray  # [F] int64
    neighbour: np.ndarray  # [n_internal] int64
    patches: list[PatchSpec]
    n_cells: int
    wall_patch_names: tuple[str, ...] = ("dragObject",)
    # Patches whose points snap onto the SDF surface: wall-type patches plus
    # interior-cap carve patches.  Empty -> same as wall_patch_names.
    snap_patch_names: tuple[str, ...] = ()

    @property
    def n_internal_faces(self) -> int:
        return len(self.neighbour)

    def patch_by_name(self, name: str) -> PatchSpec:
        for p in self.patches:
            if p.name == name:
                return p
        raise KeyError(name)

    def patch_face_slice(self, name: str) -> slice:
        p = self.patch_by_name(name)
        return slice(p.start, p.start + p.n_faces)

    def _contiguous_slice(self, names: tuple[str, ...], what: str) -> slice:
        present = [p for p in self.patches if p.name in names]
        if not present:
            raise KeyError(f"No {what} patches {names} in mesh")
        start = min(p.start for p in present)
        end = max(p.start + p.n_faces for p in present)
        if sum(p.n_faces for p in present) != end - start:
            raise RuntimeError(f"{what} patches are not contiguous".capitalize())
        return slice(start, end)

    def wall_face_slice(self) -> slice:
        """Single contiguous slice over all wall-type patches (they are
        written last and adjacent in the boundary file)."""
        return self._contiguous_slice(self.wall_patch_names, "wall")

    def wall_point_ids(self) -> np.ndarray:
        """Sorted unique point ids over all wall-type patches."""
        return np.unique(self.faces[self.wall_face_slice()].ravel())

    def snap_face_slice(self) -> slice:
        """Single contiguous slice over all snapped patches (wall-type
        patches plus interior-cap carve patches, adjacent at the end of
        the boundary file)."""
        names = self.snap_patch_names or self.wall_patch_names
        return self._contiguous_slice(names, "snapped")

    def snap_point_ids(self) -> np.ndarray:
        """Sorted unique point ids over all snapped patches."""
        return np.unique(self.faces[self.snap_face_slice()].ravel())


def _classify_boundary(
    lattice: Lattice,
    direction: int,
    plane_fine: np.ndarray,
    face_codes: dict[str, int],
    wall_code: int,
) -> np.ndarray:
    """Patch code per boundary face given its plane position (fine units).

    Faces on the domain boundary planes get the configured patch; all other
    boundary faces are walls (the geometry surface) -- legal both inside the
    mesh box (design surface) and outside (fixed walls of internal flows).
    """
    axis = direction // 2
    positive = direction % 2 == 0
    codes = np.full(len(plane_fine), wall_code, dtype=np.int64)
    if positive:
        on_domain = plane_fine == lattice.fine_dims[axis]
    else:
        on_domain = plane_fine == 0
    key = f"{'xyz'[axis]}_{'max' if positive else 'min'}"
    codes[on_domain] = face_codes[key]
    return codes


def build_polymesh(
    lattice: Lattice,
    cells: CellSet,
    mesh_box=None,
    patch_plan: PatchPlan | None = None,
) -> PolyMeshData:
    """Assemble polyMesh arrays from the combined (outer + inner fluid) cells.

    ``patch_plan`` controls the boundary layout; ``None`` reproduces the
    drag-case default (inlet/outlet/sides/dragObject).  ``mesh_box`` is
    accepted for backward compatibility and no longer used.
    """
    plan = patch_plan if patch_plan is not None else PatchPlan.default_external()
    ordered_names = plan.ordered_names()
    name_code = {name: i for i, name in enumerate(ordered_names)}
    face_codes = {key: name_code[plan.domain_faces[key]] for key in _FACE_KEYS}
    wall_code = name_code[plan.wall_name]

    index = CellIndex(lattice, cells)
    widths = cells.widths(lattice)
    levels = cells.levels
    n_cells = len(cells)

    corner_batches: list[np.ndarray] = []  # each [M, 4, 3] fine-unit keys
    emitter_batches: list[np.ndarray] = []
    nbr_batches: list[np.ndarray] = []
    patch_batches: list[np.ndarray] = []

    for direction in range(6):
        samples = face_sample_points_doubled(cells.anchors, widths, direction)
        nbr = index.locate_doubled(samples.reshape(-1, 3)).reshape(-1, 4)

        all_same = np.all(nbr == nbr[:, 0:1], axis=1)
        full_found = all_same & (nbr[:, 0] >= 0)
        full_bound = all_same & (nbr[:, 0] == -1)
        mixed = ~all_same

        # --- full faces against an existing neighbour ---------------------
        if np.any(full_found):
            idx = np.nonzero(full_found)[0]
            nbr_ids = nbr[idx, 0]
            nbr_lvl = levels[nbr_ids]
            same_lvl = nbr_lvl == levels[idx]
            coarser = nbr_lvl == levels[idx] - 1
            if np.any(~same_lvl & ~coarser):
                raise RuntimeError("2:1 balance violated during face build")

            # Same level: emit once (positive directions only).
            # Fine -> coarse: the fine side emits, in any direction.
            emit = (same_lvl & (direction % 2 == 0)) | coarser
            idx = idx[emit]
            if len(idx) > 0:
                corner_batches.append(
                    face_corner_keys(cells.anchors[idx], widths[idx], direction)
                )
                emitter_batches.append(idx)
                nbr_batches.append(nbr[idx, 0])
                patch_batches.append(np.full(len(idx), _INTERNAL, dtype=np.int64))

        # --- full boundary faces ------------------------------------------
        if np.any(full_bound):
            idx = np.nonzero(full_bound)[0]
            axis = direction // 2
            plane = cells.anchors[idx, axis] + (widths[idx] if direction % 2 == 0 else 0)
            codes = _classify_boundary(lattice, direction, plane, face_codes, wall_code)
            corner_batches.append(
                face_corner_keys(cells.anchors[idx], widths[idx], direction)
            )
            emitter_batches.append(idx)
            nbr_batches.append(np.full(len(idx), -1, dtype=np.int64))
            patch_batches.append(codes)

        # --- mixed: refined neighbour region, partially absent -------------
        if np.any(mixed):
            m_idx = np.nonzero(mixed)[0]
            m_nbr = nbr[m_idx]  # [M, 4]
            found = m_nbr >= 0
            lvl_ok = np.zeros_like(found)
            lvl_ok[found] = levels[m_nbr[found]] == np.repeat(levels[m_idx], 4).reshape(-1, 4)[found] + 1
            if np.any(found & ~lvl_ok):
                raise RuntimeError("Mixed face with non-finer neighbour")

            # Quadrants without a neighbour become quarter wall faces.
            for quadrant in range(4):
                q_missing = m_idx[~found[:, quadrant]]
                if len(q_missing) == 0:
                    continue
                corner_batches.append(
                    face_corner_keys(
                        cells.anchors[q_missing],
                        widths[q_missing],
                        direction,
                        quadrant=np.full(len(q_missing), quadrant, dtype=np.int64),
                    )
                )
                emitter_batches.append(q_missing)
                nbr_batches.append(np.full(len(q_missing), -1, dtype=np.int64))
                patch_batches.append(np.full(len(q_missing), wall_code, dtype=np.int64))

    corners = np.concatenate(corner_batches, axis=0)  # [F, 4, 3]
    emitter = np.concatenate(emitter_batches)
    neighbour_raw = np.concatenate(nbr_batches)
    patch_code = np.concatenate(patch_batches)

    # --- centroid-based reassignment (carves, sensitivity region, sub-patch)
    if (
        plan.wall_carves
        or plan.sensitivity_name is not None
        or plan.subpatch_name is not None
    ):
        centroids = lattice.point_coords(corners.mean(axis=1))
        eps = 0.25 * lattice.h_fine

        # Interior-cap carves first: they take faces out of the wall set
        # before the sensitivity-box test (cap planes are validated to lie
        # outside the design domain, so the order is defensive only).
        for carve in plan.wall_carves:
            cand = np.nonzero(patch_code == wall_code)[0]
            if len(cand) == 0:
                break
            corner_phys = lattice.point_coords(
                corners[cand].reshape(-1, 3)
            ).reshape(-1, 4, 3)
            sel = np.asarray(carve.classify(centroids[cand], corner_phys), dtype=bool)
            patch_code[cand[sel]] = name_code[carve.name]

        if plan.sensitivity_name is not None:
            box = plan.sensitivity_box
            if box is None:
                raise ValueError("sensitivity_name requires sensitivity_box")
            inside = np.all(
                (centroids >= box[0][None, :] - eps)
                & (centroids <= box[1][None, :] + eps),
                axis=1,
            )
            move = (patch_code == wall_code) & inside
            patch_code[move] = name_code[plan.sensitivity_name]

        if plan.subpatch_name is not None:
            src = name_code[plan.subpatch_source]
            cand = np.nonzero(patch_code == src)[0]
            if len(cand) > 0:
                sel = np.asarray(plan.face_subpatch(centroids[cand]), dtype=bool)
                patch_code[cand[sel]] = name_code[plan.subpatch_name]

    # --- points: unique corner keys, sorted ascending ----------------------
    corner_packed = pack_keys(corners.reshape(-1, 3)).reshape(-1, 4)
    point_keys, faces_pt = np.unique(corner_packed, return_inverse=True)
    faces_pt = faces_pt.reshape(-1, 4).astype(np.int64)
    points = lattice.point_coords(unpack_keys(point_keys))

    # --- owner/neighbour + winding ----------------------------------------
    internal = patch_code == _INTERNAL
    owner = emitter.copy()
    neighbour = neighbour_raw.copy()
    flip = internal & (neighbour_raw < emitter)
    owner[flip] = neighbour_raw[flip]
    neighbour[flip] = emitter[flip]
    faces_pt[flip] = faces_pt[flip][:, ::-1]

    # --- ordering: internal upper-triangular, then per-patch blocks --------
    int_idx = np.nonzero(internal)[0]
    int_order = int_idx[np.lexsort((neighbour[int_idx], owner[int_idx]))]

    wall_names = set(plan.wall_patch_names())
    type_overrides = plan.patch_types or {}
    patches: list[PatchSpec] = []
    boundary_order = []
    start = len(int_order)
    for code, name in enumerate(ordered_names):
        p_idx = np.nonzero(patch_code == code)[0]
        if len(p_idx) == 0:
            if name in plan.required_nonempty:
                raise ValueError(
                    f"Patch {name!r} received no faces but is required "
                    "nonempty: check the cap plane value/tol, the domain "
                    "alignment, and that the region behind the cap is "
                    "reachable from the seed point."
                )
            if plan.drop_empty:
                continue
        p_order = p_idx[np.lexsort((corner_packed[p_idx, 0], owner[p_idx]))]
        boundary_order.append(p_order)
        is_wall = name in wall_names
        patches.append(
            PatchSpec(
                name=name,
                start=start,
                n_faces=len(p_order),
                type="wall" if is_wall else type_overrides.get(name, "patch"),
                in_groups=plan.wall_groups if is_wall else (),
            )
        )
        start += len(p_order)

    order = np.concatenate([int_order] + boundary_order)
    faces_pt = faces_pt[order]
    owner = owner[order]
    neighbour = neighbour[order][: len(int_order)]

    if np.any(owner[: len(int_order)] >= neighbour):
        raise RuntimeError("owner < neighbour violated")

    data = PolyMeshData(
        points=points,
        point_keys=point_keys,
        faces=faces_pt,
        owner=owner,
        neighbour=neighbour,
        patches=patches,
        n_cells=n_cells,
        wall_patch_names=tuple(plan.wall_patch_names()),
        snap_patch_names=tuple(plan.snap_patch_names()),
    )
    n_wall = data.wall_face_slice()
    logger.info(
        "polyMesh: %d cells, %d points, %d faces (%d internal, wall %d)",
        n_cells, len(points), len(faces_pt), len(neighbour),
        n_wall.stop - n_wall.start,
    )
    return data


# ---------------------------------------------------------------------------
# Geometry checks (used by the snap quality guard and by tests)
# ---------------------------------------------------------------------------

def cell_centers_approx(mesh: PolyMeshData, points: np.ndarray | None = None) -> np.ndarray:
    """Approximate cell centers: mean of the corner points of each cell's faces."""
    pts = mesh.points if points is None else points
    sums = np.zeros((mesh.n_cells, 3), dtype=np.float64)
    counts = np.zeros(mesh.n_cells, dtype=np.int64)
    face_pt_sum = pts[mesh.faces].sum(axis=1)  # [F, 3]

    np.add.at(sums, mesh.owner, face_pt_sum)
    np.add.at(counts, mesh.owner, 4)
    n_int = mesh.n_internal_faces
    np.add.at(sums, mesh.neighbour, face_pt_sum[:n_int])
    np.add.at(counts, mesh.neighbour, 4)
    return sums / counts[:, None]


def face_pyramid_volumes(
    mesh: PolyMeshData, points: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Signed pyramid volumes between each face and its owner / neighbour.

    Both must be positive for a valid mesh (checkMesh minPyrVol).  Returns
    (pyr_owner [F], pyr_neighbour [n_internal]).
    """
    pts = mesh.points if points is None else points
    centers = cell_centers_approx(mesh, pts)

    fp = pts[mesh.faces]  # [F, 4, 3]
    c_f = fp.mean(axis=1)  # [F, 3]
    n_int = mesh.n_internal_faces

    pyr_owner = np.zeros(len(mesh.faces), dtype=np.float64)
    pyr_neigh = np.zeros(n_int, dtype=np.float64)
    c_o = centers[mesh.owner]
    c_n = centers[mesh.neighbour]

    for k in range(4):
        p0 = fp[:, k]
        p1 = fp[:, (k + 1) % 4]
        area_vec = 0.5 * np.cross(p1 - p0, c_f - p0)  # oriented with winding
        c_t = (p0 + p1 + c_f) / 3.0
        pyr_owner += np.einsum("ij,ij->i", area_vec, c_t - c_o) / 3.0
        pyr_neigh += np.einsum("ij,ij->i", -area_vec[:n_int], c_t[:n_int] - c_n) / 3.0

    return pyr_owner, pyr_neigh


def cell_volumes(mesh: PolyMeshData, points: np.ndarray | None = None) -> np.ndarray:
    """Cell volumes via the divergence theorem (exact for the face geometry)."""
    pts = mesh.points if points is None else points
    fp = pts[mesh.faces]
    c_f = fp.mean(axis=1)
    n_int = mesh.n_internal_faces

    vols = np.zeros(mesh.n_cells, dtype=np.float64)
    for k in range(4):
        p0 = fp[:, k]
        p1 = fp[:, (k + 1) % 4]
        area_vec = 0.5 * np.cross(p1 - p0, c_f - p0)
        c_t = (p0 + p1 + c_f) / 3.0
        contrib = np.einsum("ij,ij->i", area_vec, c_t) / 3.0
        np.add.at(vols, mesh.owner, contrib)
        np.subtract.at(vols, mesh.neighbour, contrib[:n_int])
    return vols

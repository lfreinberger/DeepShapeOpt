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

# Patch codes (order == patch order in the boundary file; wall patch last so
# the static patches keep their position regardless of the shape).
PATCH_INLET = 0
PATCH_OUTLET = 1
PATCH_SIDES = 2
PATCH_WALL = 3
_INTERNAL = -1

PATCH_NAMES = ["inlet", "outlet", "sides", "dragObject"]


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

    @property
    def n_internal_faces(self) -> int:
        return len(self.neighbour)

    def patch_by_name(self, name: str) -> PatchSpec:
        for p in self.patches:
            if p.name == name:
                return p
        raise KeyError(name)

    def wall_face_slice(self) -> slice:
        p = self.patch_by_name("dragObject")
        return slice(p.start, p.start + p.n_faces)

    def wall_point_ids(self) -> np.ndarray:
        """Sorted unique point ids of the wall (dragObject) patch."""
        return np.unique(self.faces[self.wall_face_slice()].ravel())


def _classify_boundary(
    lattice: Lattice, direction: int, plane_fine: np.ndarray, in_box: np.ndarray
) -> np.ndarray:
    """Patch code per boundary face given its plane position (fine units)."""
    axis = direction // 2
    positive = direction % 2 == 0
    codes = np.full(len(plane_fine), PATCH_WALL, dtype=np.int64)
    if positive:
        on_domain = plane_fine == lattice.fine_dims[axis]
    else:
        on_domain = plane_fine == 0
    if axis == 0:
        codes[on_domain] = PATCH_OUTLET if positive else PATCH_INLET
    else:
        codes[on_domain] = PATCH_SIDES
    if np.any(on_domain & in_box):
        raise RuntimeError("Domain-boundary face inside the mesh box")
    if np.any(~on_domain & ~in_box):
        raise RuntimeError(
            "Wall face outside the mesh box: a cell is missing a neighbour "
            "in the static outer region"
        )
    return codes


def build_polymesh(lattice: Lattice, cells: CellSet, mesh_box) -> PolyMeshData:
    """Assemble polyMesh arrays from the combined (outer + inner fluid) cells.

    ``mesh_box`` is the :class:`~deepshapeopt.hexmesh.octree.MeshBox`; it is
    only used for sanity-classifying boundary faces.
    """
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
            in_box = np.all(
                (cells.anchors[idx] >= mesh_box.lo[None, :])
                & (cells.anchors[idx] + widths[idx][:, None] <= mesh_box.hi[None, :]),
                axis=1,
            )
            codes = _classify_boundary(lattice, direction, plane, in_box)
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
                patch_batches.append(np.full(len(q_missing), PATCH_WALL, dtype=np.int64))

    corners = np.concatenate(corner_batches, axis=0)  # [F, 4, 3]
    emitter = np.concatenate(emitter_batches)
    neighbour_raw = np.concatenate(nbr_batches)
    patch_code = np.concatenate(patch_batches)

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

    patches: list[PatchSpec] = []
    boundary_order = []
    start = len(int_order)
    for code, name in enumerate(PATCH_NAMES):
        p_idx = np.nonzero(patch_code == code)[0]
        p_order = p_idx[np.lexsort((corner_packed[p_idx, 0], owner[p_idx]))]
        boundary_order.append(p_order)
        patches.append(
            PatchSpec(
                name=name,
                start=start,
                n_faces=len(p_order),
                type="wall" if code == PATCH_WALL else "patch",
                in_groups=("dragObjectGroup",) if code == PATCH_WALL else (),
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
    )
    logger.info(
        "polyMesh: %d cells, %d points, %d faces (%d internal, wall %d)",
        n_cells, len(points), len(faces_pt), len(neighbour),
        data.patch_by_name("dragObject").n_faces,
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

"""Integer-lattice utilities for the SDF-driven hex mesh pipeline.

All octree cells live on a single global integer lattice: the domain is
divided into ``root_dims`` root cells of size ``h0`` (level 0); a cell at
level ``l`` has side ``2**(max_depth - l)`` *fine units*, where one fine
unit is ``h0 / 2**max_depth``.  Cells are stored as ``(level, anchor)``
with the anchor being the minimum corner in fine units.  Because every
coordinate is an integer, point coordinates are bit-identical across
iterations (basis of the fixed non-design-space mesh guarantee).

Point-location queries use *doubled* fine coordinates so that cell- and
face-center sample points are exact integers.
"""

from __future__ import annotations

import dataclasses

import numpy as np

# 21 bits per axis -> up to 2**21 fine units per axis, packed into one int64.
_BITS = 21
_MASK = (1 << _BITS) - 1


def pack_keys(keys: np.ndarray) -> np.ndarray:
    """Pack non-negative integer triplets [N, 3] into a single int64 key."""
    keys = np.asarray(keys, dtype=np.int64)
    if keys.ndim == 1:
        keys = keys[None, :]
    return (keys[:, 0] << (2 * _BITS)) | (keys[:, 1] << _BITS) | keys[:, 2]


def unpack_keys(packed: np.ndarray) -> np.ndarray:
    """Inverse of :func:`pack_keys`; returns [N, 3] int64."""
    packed = np.asarray(packed, dtype=np.int64)
    return np.stack(
        [(packed >> (2 * _BITS)) & _MASK, (packed >> _BITS) & _MASK, packed & _MASK],
        axis=1,
    )


@dataclasses.dataclass(frozen=True)
class Lattice:
    """Global integer lattice covering the whole CFD domain."""

    origin: np.ndarray  # (3,) float64, physical min corner of the domain
    h0: float  # root cell size (== blockMesh background cell size)
    root_dims: np.ndarray  # (3,) int64, number of root cells per axis
    max_depth: int  # finest refinement level relative to the root

    def __post_init__(self):
        object.__setattr__(self, "origin", np.asarray(self.origin, dtype=np.float64))
        object.__setattr__(self, "root_dims", np.asarray(self.root_dims, dtype=np.int64))
        if np.any(self.fine_dims >= (1 << _BITS)):
            raise ValueError(
                f"Lattice too fine for key packing: fine_dims={self.fine_dims}"
            )

    @property
    def fine_per_root(self) -> int:
        return 1 << self.max_depth

    @property
    def fine_dims(self) -> np.ndarray:
        return self.root_dims * self.fine_per_root

    @property
    def h_fine(self) -> float:
        return self.h0 / self.fine_per_root

    def width(self, level) -> np.ndarray:
        """Cell side length in fine units for the given level(s)."""
        return np.int64(1) << (self.max_depth - np.asarray(level, dtype=np.int64))

    def cell_size(self, level) -> np.ndarray:
        """Physical cell side length for the given level(s)."""
        return self.h0 / (1 << np.asarray(level, dtype=np.int64)).astype(np.float64)

    def point_coords(self, keys: np.ndarray) -> np.ndarray:
        """Physical float64 coordinates of fine-unit integer points [N, 3]."""
        return self.origin[None, :] + np.asarray(keys, dtype=np.float64) * self.h_fine

    def to_fine_units(self, coords) -> np.ndarray:
        """Convert physical coordinates to (rounded) fine units; must be exact."""
        rel = (np.asarray(coords, dtype=np.float64) - self.origin) / self.h_fine
        fine = np.rint(rel).astype(np.int64)
        if not np.allclose(rel, fine, atol=1e-9):
            raise ValueError(f"Coordinates {coords} are not lattice-aligned")
        return fine


@dataclasses.dataclass
class CellSet:
    """A set of octree cells: per-cell level and anchor (min corner, fine units)."""

    levels: np.ndarray  # [N] int64
    anchors: np.ndarray  # [N, 3] int64

    def __post_init__(self):
        self.levels = np.asarray(self.levels, dtype=np.int64)
        self.anchors = np.asarray(self.anchors, dtype=np.int64).reshape(-1, 3)

    def __len__(self) -> int:
        return len(self.levels)

    def widths(self, lattice: Lattice) -> np.ndarray:
        return lattice.width(self.levels)

    def centers_phys(self, lattice: Lattice) -> np.ndarray:
        w = self.widths(lattice)
        return lattice.point_coords(self.anchors + 0.5 * w[:, None])

    def corners_phys(self, lattice: Lattice) -> np.ndarray:
        """[N, 8, 3] physical corner coordinates."""
        w = self.widths(lattice)
        offsets = np.array(
            [[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)],
            dtype=np.int64,
        )  # [8, 3]
        corners = self.anchors[:, None, :] + offsets[None, :, :] * w[:, None, None]
        return lattice.point_coords(corners.reshape(-1, 3)).reshape(-1, 8, 3)

    def sort_key(self) -> np.ndarray:
        """Deterministic ordering: by packed anchor (anchors are unique)."""
        return np.argsort(pack_keys(self.anchors), kind="stable")

    @staticmethod
    def concat(a: "CellSet", b: "CellSet") -> "CellSet":
        return CellSet(
            levels=np.concatenate([a.levels, b.levels]),
            anchors=np.concatenate([a.anchors, b.anchors], axis=0),
        )


def split_cells(cells: CellSet, mask: np.ndarray, lattice: Lattice) -> CellSet:
    """Replace masked cells by their 8 children, keep the rest unchanged."""
    if not np.any(mask):
        return cells
    keep_levels = cells.levels[~mask]
    keep_anchors = cells.anchors[~mask]

    parents_levels = cells.levels[mask]
    parents_anchors = cells.anchors[mask]
    if np.any(parents_levels >= lattice.max_depth):
        raise ValueError("Cannot split cells already at max_depth")

    half = lattice.width(parents_levels + 1)  # child width in fine units
    offsets = np.array(
        [[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)],
        dtype=np.int64,
    )  # [8, 3]
    child_anchors = (
        parents_anchors[:, None, :] + offsets[None, :, :] * half[:, None, None]
    ).reshape(-1, 3)
    child_levels = np.repeat(parents_levels + 1, 8)

    return CellSet(
        levels=np.concatenate([keep_levels, child_levels]),
        anchors=np.concatenate([keep_anchors, child_anchors], axis=0),
    )


class CellIndex:
    """Point-location index over a :class:`CellSet`.

    Queries take *doubled* fine coordinates (integers), so face/cell centers
    and quadrant sample points are representable exactly.
    """

    def __init__(self, lattice: Lattice, cells: CellSet):
        self.lattice = lattice
        self._levels = np.unique(cells.levels)[::-1]  # finest first
        self._per_level: list[tuple[int, np.ndarray, np.ndarray]] = []
        for level in self._levels:
            sel = np.nonzero(cells.levels == level)[0]
            packed = pack_keys(cells.anchors[sel])
            order = np.argsort(packed)
            self._per_level.append((int(level), packed[order], sel[order]))
        self._fine_dims2 = lattice.fine_dims * 2

    def locate_doubled(self, q: np.ndarray) -> np.ndarray:
        """Locate cells containing doubled-fine-coordinate points [M, 3].

        Returns global cell indices, -1 where no cell contains the point
        (outside the domain or in a removed region).  Points must lie
        strictly inside a cell (odd coordinate in at least the relevant
        axes); face-boundary points are ambiguous and must be avoided by
        the caller.
        """
        q = np.asarray(q, dtype=np.int64).reshape(-1, 3)
        out = np.full(len(q), -1, dtype=np.int64)
        inside = np.all((q >= 0) & (q < self._fine_dims2[None, :]), axis=1)
        if not np.any(inside):
            return out
        idx_inside = np.nonzero(inside)[0]
        q_in = q[idx_inside]
        remaining = np.arange(len(q_in))
        for level, packed_sorted, global_ids in self._per_level:
            if len(remaining) == 0:
                break
            w = int(self.lattice.width(level))
            anchor = (q_in[remaining] // (2 * w)) * w
            keys = pack_keys(anchor)
            pos = np.searchsorted(packed_sorted, keys)
            pos_clipped = np.minimum(pos, len(packed_sorted) - 1)
            hit = (len(packed_sorted) > 0) & (packed_sorted[pos_clipped] == keys)
            if np.any(hit):
                out[idx_inside[remaining[hit]]] = global_ids[pos_clipped[hit]]
                remaining = remaining[~hit]
        return out


def face_sample_points_doubled(
    anchors: np.ndarray, widths: np.ndarray, direction: int
) -> np.ndarray:
    """Sample points just outside one face of each cell, in doubled fine coords.

    Four quadrant samples per face (so that all up-to-4 finer neighbours are
    found); for finest-level cells (width 1) the four samples coincide at the
    face center.

    Parameters
    ----------
    anchors : [N, 3] int64 (fine units)
    widths : [N] int64 (fine units)
    direction : int in 0..5 -> +x, -x, +y, -y, +z, -z

    Returns
    -------
    [N, 4, 3] int64 doubled fine coordinates.
    """
    n = len(anchors)
    axis = direction // 2
    positive = direction % 2 == 0
    t_axes = [a for a in range(3) if a != axis]

    q = np.empty((n, 4, 3), dtype=np.int64)
    # Normal-axis coordinate: just outside the face plane.
    if positive:
        q[:, :, axis] = (2 * (anchors[:, axis] + widths) + 1)[:, None]
    else:
        q[:, :, axis] = (2 * anchors[:, axis] - 1)[:, None]

    # Transverse quadrant centers (doubled units): w/2 and 3w/2 for w >= 2,
    # both collapse to the face center (offset w == 1) for w == 1.
    lo = np.where(widths >= 2, widths // 2, 1)
    hi = np.where(widths >= 2, 3 * (widths // 2), 1)
    u, v = t_axes
    base_u = 2 * anchors[:, u]
    base_v = 2 * anchors[:, v]
    # Quadrant order: (lo,lo), (lo,hi), (hi,lo), (hi,hi)
    q[:, 0, u] = base_u + lo
    q[:, 0, v] = base_v + lo
    q[:, 1, u] = base_u + lo
    q[:, 1, v] = base_v + hi
    q[:, 2, u] = base_u + hi
    q[:, 2, v] = base_v + lo
    q[:, 3, u] = base_u + hi
    q[:, 3, v] = base_v + hi
    return q


# Face corner templates: for a cell (anchor a, width w) and direction d, the
# four corners (in fine units) wound so the right-hand-rule normal points in
# direction d (outward from the cell).  Each entry: [4, 3] multiples of w
# added to the face base corner.
_FACE_TEMPLATES = {
    # +x at x = a_x + w
    0: (np.array([1, 0, 0]), np.array([[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]])),
    # -x at x = a_x (reverse winding of +x)
    1: (np.array([0, 0, 0]), np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]])),
    # +y at y = a_y + w
    2: (np.array([0, 1, 0]), np.array([[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]])),
    # -y at y = a_y
    3: (np.array([0, 0, 0]), np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]])),
    # +z at z = a_z + w
    4: (np.array([0, 0, 1]), np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])),
    # -z at z = a_z
    5: (np.array([0, 0, 0]), np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]])),
}


def face_corner_keys(
    anchors: np.ndarray,
    widths: np.ndarray,
    direction: int,
    quadrant: np.ndarray | None = None,
) -> np.ndarray:
    """Corner point keys [N, 4, 3] (fine units) of cell faces in a direction.

    With ``quadrant`` (int array [N], values 0..3 matching the quadrant order
    of :func:`face_sample_points_doubled`), quarter faces of the half width
    are produced instead (used when a coarse cell borders a refined region
    that is only partially present).
    """
    base_shift, template = _FACE_TEMPLATES[direction]
    anchors = np.asarray(anchors, dtype=np.int64).reshape(-1, 3)
    widths = np.asarray(widths, dtype=np.int64).reshape(-1)

    if quadrant is None:
        w = widths
        base = anchors + base_shift[None, :] * w[:, None]
    else:
        if np.any(widths < 2):
            raise ValueError("Quarter faces require width >= 2")
        w = widths // 2
        axis = direction // 2
        t_axes = [a for a in range(3) if a != axis]
        base = anchors + base_shift[None, :] * widths[:, None]
        # Quadrant order (lo,lo), (lo,hi), (hi,lo), (hi,hi) in (u, v).
        u_sel = (quadrant >= 2).astype(np.int64)
        v_sel = (quadrant % 2).astype(np.int64)
        base = base.copy()
        base[:, t_axes[0]] += u_sel * w
        base[:, t_axes[1]] += v_sel * w

    return base[:, None, :] + template[None, :, :] * w[:, None, None]

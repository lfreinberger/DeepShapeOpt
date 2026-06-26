"""Octree castellation for the SDF-driven hex mesh pipeline.

Two builders:

- :func:`build_static_octree`: one-time outer mesh covering the domain minus
  the mesh box, statically graded (2:1) so every outer cell touching the box
  is exactly at ``interface_level``.  Identical across all iterations.
- :func:`build_inner_castellation`: per-iteration octree inside the mesh box,
  refined toward the SDF zero level set, with a frozen one-cell-thick shell
  at ``interface_level`` along the box boundary so the interface to the
  static mesh never changes.

Levels are absolute: level 0 == root cells of size ``h0``.
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np

from .lattice import (
    CellIndex,
    CellSet,
    Lattice,
    face_sample_points_doubled,
    split_cells,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class MeshBox:
    """The design-space mesh box in fine lattice units (root-cell aligned)."""

    lo: np.ndarray  # (3,) int64 fine units
    hi: np.ndarray  # (3,) int64 fine units

    @staticmethod
    def from_physical(lattice: Lattice, box_phys) -> "MeshBox":
        box_phys = np.asarray(box_phys, dtype=np.float64)
        lo = lattice.to_fine_units(box_phys[0])
        hi = lattice.to_fine_units(box_phys[1])
        fpr = lattice.fine_per_root
        if np.any(lo % fpr != 0) or np.any(hi % fpr != 0):
            raise ValueError(
                f"mesh_box {box_phys.tolist()} is not aligned to root cells "
                f"(h0={lattice.h0})"
            )
        if np.any(lo < 0) or np.any(hi > lattice.fine_dims) or np.any(lo >= hi):
            raise ValueError(f"mesh_box {box_phys.tolist()} not inside the domain")
        return MeshBox(lo=lo, hi=hi)

    def contains_cells(self, cells: CellSet, lattice: Lattice) -> np.ndarray:
        """Cells fully inside the box (cells never straddle the box boundary)."""
        w = cells.widths(lattice)
        return np.all(
            (cells.anchors >= self.lo[None, :])
            & (cells.anchors + w[:, None] <= self.hi[None, :]),
            axis=1,
        )

    def interface_faces(self, lattice: Lattice) -> np.ndarray:
        """(2, 3) bool: which box faces ([lo/hi side, axis]) border the static
        mesh.  A box face coinciding with the domain boundary (internal-flow
        outlet/inlet plane) is *not* an interface: there is no static mesh
        behind it, so no frozen shell or grading cap applies there.
        """
        return np.stack([self.lo > 0, self.hi < lattice.fine_dims])

    def touches_boundary(self, cells: CellSet, lattice: Lattice) -> np.ndarray:
        """Cells inside the box sharing a face patch with an *interface* face."""
        w = cells.widths(lattice)
        ifc = self.interface_faces(lattice)
        at_lo = np.any((cells.anchors == self.lo[None, :]) & ifc[0][None, :], axis=1)
        at_hi = np.any(
            (cells.anchors + w[:, None] == self.hi[None, :]) & ifc[1][None, :], axis=1
        )
        return self.contains_cells(cells, lattice) & (at_lo | at_hi)

    def touches_domain_faces(self, cells: CellSet, lattice: Lattice) -> np.ndarray:
        """Cells inside the box touching a box face that lies on the domain
        boundary (the complement of :meth:`touches_boundary` contact)."""
        w = cells.widths(lattice)
        ifc = self.interface_faces(lattice)
        at_lo = np.any((cells.anchors == self.lo[None, :]) & ~ifc[0][None, :], axis=1)
        at_hi = np.any(
            (cells.anchors + w[:, None] == self.hi[None, :]) & ~ifc[1][None, :], axis=1
        )
        return self.contains_cells(cells, lattice) & (at_lo | at_hi)


def root_cells(lattice: Lattice, mask_fn=None) -> CellSet:
    """All root cells of the domain, optionally filtered by mask_fn(anchors)."""
    nx, ny, nz = (int(d) for d in lattice.root_dims)
    fpr = lattice.fine_per_root
    grid = np.stack(
        np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"),
        axis=-1,
    ).reshape(-1, 3).astype(np.int64) * fpr
    if mask_fn is not None:
        grid = grid[mask_fn(grid)]
    return CellSet(levels=np.zeros(len(grid), dtype=np.int64), anchors=grid)


# ---------------------------------------------------------------------------
# 2:1 balancing
# ---------------------------------------------------------------------------

def balance_2to1(
    lattice: Lattice,
    cells: CellSet,
    pinned: np.ndarray | None = None,
    max_rounds: int = 64,
) -> CellSet:
    """Refine cells until no face-adjacent pair differs by more than 1 level.

    ``pinned`` cells must never be refined; if balancing demands it, the
    configuration is infeasible (e.g. max_level too high for the margin
    between design domain and mesh box) and an error is raised.  Pinned
    status is tracked through rounds via cell anchors (a pinned cell that is
    not refined keeps its identity).
    """
    # Cells are disjoint, so the packed anchor uniquely identifies a cell as
    # long as it is never split (pinned cells never are).
    pinned_keys = None
    if pinned is not None and np.any(pinned):
        from .lattice import pack_keys

        pinned_keys = np.sort(pack_keys(cells.anchors[pinned]))

    for round_idx in range(max_rounds):
        index = CellIndex(lattice, cells)
        w = cells.widths(lattice)
        need_refine = np.zeros(len(cells), dtype=bool)

        for direction in range(6):
            samples = face_sample_points_doubled(cells.anchors, w, direction)
            nbr = index.locate_doubled(samples.reshape(-1, 3)).reshape(-1, 4)
            nbr_flat = nbr[nbr >= 0]
            if len(nbr_flat) == 0:
                continue
            src_levels = np.repeat(cells.levels, 4)[(nbr >= 0).reshape(-1)]
            too_coarse = cells.levels[nbr_flat] <= src_levels - 2
            need_refine[nbr_flat[too_coarse]] = True

        if not np.any(need_refine):
            if round_idx > 0:
                logger.debug("2:1 balance converged after %d rounds", round_idx)
            return cells

        if pinned_keys is not None:
            from .lattice import pack_keys

            cand = pack_keys(cells.anchors[need_refine])
            hit = np.searchsorted(pinned_keys, cand)
            hit = np.minimum(hit, len(pinned_keys) - 1)
            if np.any(pinned_keys[hit] == cand):
                raise RuntimeError(
                    "2:1 balancing requires refining a pinned interface-shell "
                    "cell. Increase the margin between design_domain and "
                    "mesh_box, raise interface_level, or lower max_level."
                )

        cells = split_cells(cells, need_refine, lattice)

    raise RuntimeError("2:1 balancing did not converge")


# ---------------------------------------------------------------------------
# Static outer octree
# ---------------------------------------------------------------------------

def build_static_octree(
    lattice: Lattice, box: MeshBox, interface_level: int
) -> CellSet:
    """Outer mesh: domain minus mesh box, graded to interface_level at the box.

    Built by refining every cell that intersects-or-touches the *closed* mesh
    box down to ``interface_level``, balancing, then dropping the cells inside
    the box.  The remaining outer cells adjacent to the box are exactly at
    ``interface_level`` and the whole set is deterministic.
    """
    cells = root_cells(lattice)

    for _ in range(interface_level):
        w = cells.widths(lattice)
        touches = np.all(
            (cells.anchors <= box.hi[None, :])
            & (cells.anchors + w[:, None] >= box.lo[None, :]),
            axis=1,
        )
        refine = touches & (cells.levels < interface_level)
        if not np.any(refine):
            break
        cells = split_cells(cells, refine, lattice)

    cells = balance_2to1(lattice, cells)
    outer = ~box.contains_cells(cells, lattice)
    cells = CellSet(levels=cells.levels[outer], anchors=cells.anchors[outer])

    order = cells.sort_key()
    cells = CellSet(levels=cells.levels[order], anchors=cells.anchors[order])
    logger.info(
        "Static outer octree: %d cells (levels %d..%d)",
        len(cells), cells.levels.min(), cells.levels.max(),
    )
    return cells


def _static_level_cap(
    cells: CellSet,
    lattice: Lattice,
    box: MeshBox,
    interface_level: int,
    max_level: int,
) -> np.ndarray:
    """Outward mirror of :func:`_level_cap`: per-cell maximum level by
    L-infinity distance from the (closed) mesh box, so the 2:1 grading from
    any refined static cell down to ``interface_level`` always completes
    before the box.  Cells touching the box are capped at exactly
    ``interface_level`` -- both sides of the static/inner interface then
    meet at the same level by construction."""
    w = cells.widths(lattice)
    gap_lo = box.lo[None, :] - (cells.anchors + w[:, None])
    gap_hi = cells.anchors - box.hi[None, :]
    gap = np.maximum(np.maximum(gap_lo, gap_hi), 0).max(axis=1)  # fine units

    cap = np.full(len(cells), interface_level, dtype=np.int64)
    for level in range(interface_level + 1, max_level + 1):
        cap[gap >= grading_distance(lattice, interface_level, level)] = level
    return cap


def build_static_castellation(
    lattice: Lattice,
    box: MeshBox,
    interface_level: int,
    max_level: int,
    phi_np,
    beta: float,
    seed_point,
    regions: list["RefineRegion"] | None = None,
    band_max_level: int | None = None,
) -> "Castellation":
    """One-time castellation of the outer region against fixed geometry.

    Unlike :func:`build_static_octree` (external flow: the outer region is
    pure fluid), internal-flow cases have most of their walls in the static
    region, so the outer mesh needs the full treatment: band refinement
    toward the geometry surface, fluid classification, and flood fill from
    a known fluid seed point.  Built once and cached by the pipeline.

    Parameters
    ----------
    phi_np : callable
        Static geometry evaluator, numpy [N, 3] physical -> [N] float64
        (positive in the fluid), e.g. ``TriMeshSDF.phi_np``.
    seed_point : (3,) physical coordinates of a known fluid point in the
        static region (the flood-fill seed).
    band_max_level : int, optional
        Cap for the surface-band refinement (default ``max_level``).  The
        fixed walls usually need less resolution than the design surface or
        the user refinement regions; regions still refine up to their own
        level.
    """
    if band_max_level is None:
        band_max_level = max_level
    cells = root_cells(lattice)

    for level in range(max_level):
        at_level = cells.levels == level
        if not np.any(at_level):
            continue
        w = cells.widths(lattice)
        touches_closed = np.all(
            (cells.anchors <= box.hi[None, :])
            & (cells.anchors + w[:, None] >= box.lo[None, :]),
            axis=1,
        )
        inside = box.contains_cells(cells, lattice)
        cap = _static_level_cap(cells, lattice, box, interface_level, max_level)

        forced = at_level & touches_closed & (cells.levels < interface_level)
        candidates = at_level & ~inside & (cells.levels < cap)
        band_cand = candidates & (cells.levels < band_max_level)
        sdf_refine = np.zeros(len(cells), dtype=bool)
        cand_idx = np.nonzero(band_cand)[0]
        if len(cand_idx) > 0:
            sub = CellSet(cells.levels[cand_idx], cells.anchors[cand_idx])
            sdf_refine[cand_idx] = _near_surface(sub, lattice, phi_np, beta)

        region_refine = np.zeros(len(cells), dtype=bool)
        if regions:
            req = np.minimum(region_min_level(cells, lattice, regions), cap)
            region_refine = candidates & (cells.levels < req)

        refine = forced | sdf_refine | region_refine
        if not np.any(refine):
            break
        cells = split_cells(cells, refine, lattice)

    w = cells.widths(lattice)
    touches_closed = np.all(
        (cells.anchors <= box.hi[None, :])
        & (cells.anchors + w[:, None] >= box.lo[None, :]),
        axis=1,
    )
    if np.any(cells.levels[touches_closed] != interface_level):
        raise RuntimeError(
            "Static cells at the mesh box did not all reach interface_level "
            f"(levels {np.unique(cells.levels[touches_closed])})."
        )
    cells = balance_2to1(lattice, cells, pinned=touches_closed)

    inside = box.contains_cells(cells, lattice)
    outer = np.nonzero(~inside)[0]
    cells = CellSet(cells.levels[outer], cells.anchors[outer])

    # --- fluid classification + flood fill from the seed point ------------
    phi_center = phi_np(cells.centers_phys(lattice))
    fluid = phi_center > 0.0
    n_solid = int(np.sum(~fluid))
    keep = np.nonzero(fluid)[0]
    cells = CellSet(cells.levels[keep], cells.anchors[keep])

    seed = np.asarray(seed_point, dtype=np.float64).reshape(3)
    q = np.rint(2.0 * (seed - lattice.origin) / lattice.h_fine).astype(np.int64)
    seed_idx = CellIndex(lattice, cells).locate_doubled(q[None, :])[0]
    if seed_idx < 0:
        raise ValueError(
            f"sdf_hex.seed_point {seed.tolist()} is not inside a fluid cell of "
            "the static region; pick a point in the flow channel outside the "
            "mesh box."
        )
    reachable = _flood_fill(lattice, cells, seeds=np.array([seed_idx]))
    n_pockets = int(np.sum(~reachable))
    if n_pockets:
        logger.info("Static region: removing %d unreachable fluid cells", n_pockets)
        keep = np.nonzero(reachable)[0]
        cells = CellSet(cells.levels[keep], cells.anchors[keep])

    # Fluid ring adjacent to the box (the static side of the interface).
    w = cells.widths(lattice)
    shell = np.all(
        (cells.anchors <= box.hi[None, :])
        & (cells.anchors + w[:, None] >= box.lo[None, :]),
        axis=1,
    )

    order = cells.sort_key()
    cells = CellSet(cells.levels[order], cells.anchors[order])
    shell = shell[order]

    logger.info(
        "Static castellation: %d fluid cells (levels %d..%d), "
        "%d solid removed, %d unreachable removed",
        len(cells), cells.levels.min(), cells.levels.max(), n_solid, n_pockets,
    )
    return Castellation(
        cells=cells,
        shell_mask=shell,
        n_solid_removed=n_solid,
        n_pocket_removed=n_pockets,
    )


# ---------------------------------------------------------------------------
# Inner castellation (per iteration)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Castellation:
    """Result of the inner castellation: fluid cells only."""

    cells: CellSet
    shell_mask: np.ndarray  # frozen interface-shell cells (within `cells`)
    n_solid_removed: int
    n_pocket_removed: int


def grading_distance(lattice: Lattice, interface_level: int, level: int) -> int:
    """Minimum distance (fine units) from the box boundary at which a cell may
    be refined to ``level``: the frozen shell plus one cell of every
    intermediate level of the 2:1 grading chain must fit in between.
    """
    req = int(lattice.width(interface_level))
    for l in range(interface_level + 1, level):
        req += int(lattice.width(l))
    return req


def _level_cap(
    cells: CellSet,
    lattice: Lattice,
    box: MeshBox,
    interface_level: int,
    max_level: int,
) -> np.ndarray:
    """Per-cell maximum refinement level by distance to the box *interface*
    faces, such that 2:1 grading down to the frozen shell always stays
    feasible.  Box faces on the domain boundary impose no cap (there is no
    static mesh behind them), so e.g. an outlet plane stays refinable."""
    big = np.iinfo(np.int64).max // 2
    w = cells.widths(lattice)
    ifc = box.interface_faces(lattice)
    dist_lo = np.where(ifc[0][None, :], cells.anchors - box.lo[None, :], big)
    dist_hi = np.where(
        ifc[1][None, :], box.hi[None, :] - (cells.anchors + w[:, None]), big
    )
    dist = np.minimum(dist_lo, dist_hi).min(axis=1)  # fine units, >= 0

    cap = np.full(len(cells), interface_level, dtype=np.int64)
    for level in range(interface_level + 1, max_level + 1):
        cap[dist >= grading_distance(lattice, interface_level, level)] = level
    return cap


# ---------------------------------------------------------------------------
# User-specified refinement regions
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class RefineRegion:
    """Axis-aligned box (physical coords) whose cells are refined to ``level``."""

    box: np.ndarray  # (2, 3) float64 physical
    level: int


def parse_refine_regions(cfg_list, lattice: Lattice, max_level: int) -> list[RefineRegion]:
    """Parse the ``refinement_regions`` config list.

    Two entry forms:
      - ``{"box": [[lo], [hi]], "level": L}`` -- explicit physical box;
      - ``{"face": "x_min", "distance": d, "level": L}`` -- a slab of
        thickness ``d`` against that domain face.
    """
    regions: list[RefineRegion] = []
    domain_lo = lattice.origin
    domain_hi = lattice.origin + lattice.root_dims * lattice.h0
    for entry in cfg_list or []:
        level = int(entry["level"])
        if level > max_level:
            raise ValueError(
                f"refinement region level {level} exceeds max_level {max_level}"
            )
        if "box" in entry:
            box = np.asarray(entry["box"], dtype=np.float64)
        elif "face" in entry:
            axis = {"x": 0, "y": 1, "z": 2}[entry["face"][0]]
            side = entry["face"][2:]
            d = float(entry["distance"])
            box = np.stack([domain_lo.copy(), domain_hi.copy()])
            if side == "min":
                box[1, axis] = domain_lo[axis] + d
            elif side == "max":
                box[0, axis] = domain_hi[axis] - d
            else:
                raise ValueError(f"Unknown face {entry['face']!r}")
        else:
            raise ValueError(f"refinement region needs 'box' or 'face': {entry}")
        regions.append(RefineRegion(box=box, level=level))
    return regions


def region_min_level(
    cells: CellSet, lattice: Lattice, regions: list[RefineRegion]
) -> np.ndarray:
    """Per-cell required refinement level from overlapping regions (0 if none)."""
    req = np.zeros(len(cells), dtype=np.int64)
    if not regions:
        return req
    w = cells.widths(lattice)
    cell_lo = lattice.point_coords(cells.anchors)
    cell_hi = lattice.point_coords(cells.anchors + w[:, None])
    for region in regions:
        overlap = np.all(
            (cell_lo < region.box[1][None, :]) & (cell_hi > region.box[0][None, :]),
            axis=1,
        )
        req[overlap] = np.maximum(req[overlap], region.level)
    return req


def build_inner_castellation(
    lattice: Lattice,
    box: MeshBox,
    interface_level: int,
    max_level: int,
    phi_ext_np,
    beta: float = 1.0,
    regions: list[RefineRegion] | None = None,
    band_max_level: int | None = None,
) -> Castellation:
    """SDF-driven octree of the mesh box interior, fluid cells only.

    Parameters
    ----------
    phi_ext_np : callable
        Extended SDF evaluator, numpy [N, 3] physical -> [N] float64
        (positive in the fluid).
    beta : float
        Refinement band width in units of the cell half-diagonal.
    regions : list of :class:`RefineRegion`, optional
        User-specified refinement boxes (clamped by the grading cap, so
        they can never violate the frozen interface shell).
    band_max_level : int, optional
        Cap for the surface-band refinement (default ``max_level``).
        Decouples the general design-surface resolution from refinement
        regions: e.g. walls at level 2 while the outlet slab still
        refines to level 3 (regions refine all their cells to their own
        level regardless of this cap).
    """
    if band_max_level is None:
        band_max_level = max_level
    cells = root_cells(
        lattice,
        mask_fn=lambda anchors: np.all(
            (anchors >= box.lo[None, :])
            & (anchors + lattice.fine_per_root <= box.hi[None, :]),
            axis=1,
        ),
    )
    if len(cells) == 0:
        raise ValueError("mesh_box contains no root cells")

    for level in range(max_level):
        at_level = cells.levels == level
        if not np.any(at_level):
            continue
        touches = box.touches_boundary(cells, lattice)
        cap = _level_cap(cells, lattice, box, interface_level, max_level)

        forced = at_level & touches & (cells.levels < interface_level)
        candidates = at_level & ~touches & (cells.levels < cap)
        band_cand = candidates & (cells.levels < band_max_level)
        sdf_refine = np.zeros(len(cells), dtype=bool)
        cand_idx = np.nonzero(band_cand)[0]
        if len(cand_idx) > 0:
            sub = CellSet(cells.levels[cand_idx], cells.anchors[cand_idx])
            sdf_refine[cand_idx] = _near_surface(sub, lattice, phi_ext_np, beta)

        region_refine = np.zeros(len(cells), dtype=bool)
        if regions:
            req = np.minimum(region_min_level(cells, lattice, regions), cap)
            region_refine = candidates & (cells.levels < req)

        refine = forced | sdf_refine | region_refine
        if not np.any(refine):
            break
        cells = split_cells(cells, refine, lattice)

    shell = box.touches_boundary(cells, lattice)
    if np.any(cells.levels[shell] != interface_level):
        raise RuntimeError(
            "Interface shell cells did not all reach interface_level "
            f"(levels {np.unique(cells.levels[shell])}); mesh_box may be "
            "thinner than 2 root cells along some axis."
        )

    cells = balance_2to1(lattice, cells, pinned=shell)
    shell = box.touches_boundary(cells, lattice)

    # --- fluid classification -------------------------------------------
    phi_center = phi_ext_np(cells.centers_phys(lattice))
    fluid = phi_center > 0.0

    # Shell cells in the solid are legal for internal flow (the margin ring
    # contains fixed wall geometry); their classification comes from the
    # static field there and is identical every iteration, which the
    # pipeline asserts via the shell pattern hash.
    n_shell_solid = int(np.sum(shell & ~fluid))
    if n_shell_solid:
        logger.debug("%d interface-shell cells are solid (wall in the margin ring)",
                     n_shell_solid)
    n_solid = int(np.sum(~fluid))

    domain_face = box.touches_domain_faces(cells, lattice)

    keep = np.nonzero(fluid)[0]
    cells = CellSet(cells.levels[keep], cells.anchors[keep])
    shell = shell[keep]
    domain_face = domain_face[keep]

    # --- flood fill: drop sealed fluid pockets ----------------------------
    # Seeds: fluid shell cells plus fluid cells on domain-coinciding box
    # faces (outlet/inlet planes of internal-flow cases).
    seeds = np.nonzero(shell | domain_face)[0]
    if len(seeds) == 0:
        raise RuntimeError(
            "No fluid cells at the mesh-box interface or domain faces; the "
            "shape seals off the mesh box entirely."
        )
    reachable = _flood_fill(lattice, cells, seeds=seeds)
    n_pockets = int(np.sum(~reachable))
    if n_pockets:
        logger.info("Removing %d enclosed fluid pocket cells", n_pockets)
        keep = np.nonzero(reachable)[0]
        cells = CellSet(cells.levels[keep], cells.anchors[keep])
        shell = shell[keep]

    order = cells.sort_key()
    cells = CellSet(cells.levels[order], cells.anchors[order])
    shell = shell[order]

    logger.info(
        "Inner castellation: %d fluid cells (levels %d..%d), "
        "%d solid removed, %d pocket cells removed",
        len(cells), cells.levels.min(), cells.levels.max(), n_solid, n_pockets,
    )
    return Castellation(
        cells=cells,
        shell_mask=shell,
        n_solid_removed=n_solid,
        n_pocket_removed=n_pockets,
    )


def _near_surface(
    cells: CellSet, lattice: Lattice, phi_ext_np, beta: float
) -> np.ndarray:
    """Cells whose center/corner SDF samples indicate the surface band."""
    centers = cells.centers_phys(lattice)
    corners = cells.corners_phys(lattice)
    pts = np.concatenate([centers[:, None, :], corners], axis=1)  # [N, 9, 3]
    phi = phi_ext_np(pts.reshape(-1, 3)).reshape(len(cells), 9)

    half_diag = 0.5 * np.sqrt(3.0) * lattice.cell_size(cells.levels)
    in_band = np.abs(phi).min(axis=1) < beta * half_diag
    sign_change = np.any(phi[:, 1:] > 0, axis=1) & np.any(phi[:, 1:] < 0, axis=1)
    return in_band | sign_change


def _flood_fill(lattice: Lattice, cells: CellSet, seeds: np.ndarray) -> np.ndarray:
    """Face-adjacency reachability from seed cells within the cell set."""
    index = CellIndex(lattice, cells)
    w = cells.widths(lattice)
    visited = np.zeros(len(cells), dtype=bool)
    visited[seeds] = True
    frontier = np.unique(seeds)

    while len(frontier) > 0:
        nbrs = []
        fa = cells.anchors[frontier]
        fw = w[frontier]
        for direction in range(6):
            samples = face_sample_points_doubled(fa, fw, direction)
            nbr = index.locate_doubled(samples.reshape(-1, 3))
            nbrs.append(nbr[nbr >= 0])
        nbrs = np.unique(np.concatenate(nbrs))
        new = nbrs[~visited[nbrs]]
        visited[new] = True
        frontier = new

    return visited

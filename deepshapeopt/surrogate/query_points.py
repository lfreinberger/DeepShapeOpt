"""Query-point cloud construction for the Transolver flow surrogate.

The surrogate predicts (U, p) on a point cloud assembled from
- the differentiable wall surface points (role 0),
- two near-wall probe shells offset along the fluid-side vertex normals
  (roles 1 and 2, at ``delta`` and ``2 * delta``) used for the one-sided
  wall-shear finite difference in :mod:`deepshapeopt.surrogate.drag`,
- a fixed, seeded Sobol volume cloud (role 3) whose shape dependence enters
  only through the SDF feature; points inside the solid are dropped.

Per-point features follow the Transolver ShapeNetCar convention:
``[pos(3), sdf(1), unit_normal(3)]`` with zero normals off-surface.

Contract: ``sdf_fn(x_phys) -> phi`` returns the SDF in PHYSICAL distance
units with ``phi > 0`` in the fluid, ``phi < 0`` inside the solid, and must
be differentiable w.r.t. the lattice parameters (and, where applicable, x).
"""

from __future__ import annotations

import dataclasses
import logging

import torch

from ..foam_utils import (
    compute_area_weighted_vertex_normals,
    compute_vertex_normals,
)

logger = logging.getLogger(__name__)

ROLE_SURFACE = 0
ROLE_SHELL1 = 1
ROLE_SHELL2 = 2
ROLE_VOLUME = 3

# Cache of the deterministic base volume clouds, keyed by
# (seed, n_points, domain-bytes, mesh_box-bytes). CPU float32 tensors.
_VOLUME_CLOUD_CACHE: dict = {}


@dataclasses.dataclass
class QueryCloud:
    """Assembled query cloud; ``feats`` carries the autograd graph to the
    lattice parameters (via surface points, normals and the SDF channel)."""

    feats: torch.Tensor  # [N, 7] float32
    roles: torch.Tensor  # [N] int8
    n_surface: int
    unit_normals: torch.Tensor  # [P, 3] into-fluid unit vertex normals
    area_normals: torch.Tensor  # [P, 3] A_v * n_hat_v (vertex quadrature weights)
    delta: torch.Tensor  # [P] per-vertex shell offset actually used (detached)

    @property
    def n_points(self) -> int:
        return int(self.feats.shape[0])


def _sobol_box(engine: torch.quasirandom.SobolEngine, n: int, lo, hi) -> torch.Tensor:
    u = engine.draw(n)
    lo = torch.as_tensor(lo, dtype=torch.float32)
    hi = torch.as_tensor(hi, dtype=torch.float32)
    return lo + u * (hi - lo)


def base_volume_cloud(cfg: dict) -> torch.Tensor:
    """Deterministic shape-independent volume points [N_v, 3] (CPU float32).

    50% inside ``mesh_box`` (near-body), 25% in a wake strip downstream of it,
    25% across the full flow ``domain``.
    """
    n_total = int(cfg.get("n_volume_points", 8192))
    seed = int(cfg.get("volume_seed", 0))
    domain = cfg.get("domain", [[-5.0, -5.0, -5.0], [15.0, 5.0, 5.0]])
    mesh_box = cfg.get("mesh_box", [[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]])

    key = (seed, n_total, str(domain), str(mesh_box))
    cached = _VOLUME_CLOUD_CACHE.get(key)
    if cached is not None:
        return cached

    n_body = n_total // 2
    n_wake = n_total // 4
    n_far = n_total - n_body - n_wake
    engine = torch.quasirandom.SobolEngine(3, scramble=True, seed=seed)

    (dlo, dhi), (mlo, mhi) = domain, mesh_box
    wake_lo = [mhi[0], mlo[1], mlo[2]]
    wake_hi = [min(dhi[0], mhi[0] + 8.0), mhi[1], mhi[2]]

    pts = torch.cat(
        [
            _sobol_box(engine, n_body, mlo, mhi),
            _sobol_box(engine, n_wake, wake_lo, wake_hi),
            _sobol_box(engine, n_far, dlo, dhi),
        ],
        dim=0,
    )
    _VOLUME_CLOUD_CACHE[key] = pts
    return pts


def build_query_cloud(
    surface_points: torch.Tensor,
    wall_tris: torch.Tensor,
    sdf_fn,
    cfg: dict,
) -> QueryCloud:
    """Assemble the surrogate query cloud for one shape.

    Parameters
    ----------
    surface_points : [P, 3] float32 torch tensor, autograd-connected to the
        lattice parameters (snapped wall points or FlexiCubes vertices).
    wall_tris : [F, 3] integer triangle connectivity over ``surface_points``.
    sdf_fn : physical-unit SDF callable (see module docstring).
    cfg : surrogate config block (``shell_offsets``, ``n_volume_points``,
        ``volume_seed``, ``domain``, ``mesh_box``).
    """
    verts = surface_points
    device, dtype = verts.device, verts.dtype
    faces = wall_tris.to(device)
    P = verts.shape[0]

    # invert_normals=True is the exterior-drag convention in foam_utils; the
    # SDF majority vote below corrects for any winding difference.
    normals = compute_vertex_normals(verts, faces, invert_normals=True)
    area_normals = compute_area_weighted_vertex_normals(verts, faces, invert_normals=True)

    shell_offsets = cfg.get("shell_offsets", [0.015625, 0.03125])
    delta0 = float(shell_offsets[0])

    # The shell offsets are absolute distances tied to the wall cell size the
    # model was trained on (h_fine = base_cell_size / 2^max_level). Running a
    # differently refined mesh silently rescales the wall-gradient finite
    # difference -- at one level coarser the viscous drag more than doubles.
    with torch.no_grad():
        e = torch.stack([
            (verts[faces[:, 1]] - verts[faces[:, 0]]).norm(dim=1),
            (verts[faces[:, 2]] - verts[faces[:, 1]]).norm(dim=1),
            (verts[faces[:, 0]] - verts[faces[:, 2]]).norm(dim=1),
        ])
        h_wall = float(e.median())
    ratio = h_wall / (2.0 * delta0)
    if not 0.5 < ratio < 2.0:
        logger.warning(
            "Wall spacing %.4g does not match shell_offsets %s (ratio %.2f): the "
            "surrogate was trained at one wall refinement; check sdf_hex.max_level "
            "or rescale shell_offsets.",
            h_wall, list(shell_offsets), ratio,
        )

    with torch.no_grad():
        probe = verts + delta0 * normals
        frac_fluid = (sdf_fn(probe) > 0).float().mean().item()
    if frac_fluid < 0.5:
        logger.warning(
            "Vertex normals point into the solid for %.0f%% of vertices; flipping.",
            100 * (1 - frac_fluid),
        )
        normals = -normals
        area_normals = -area_normals

    # Concave-pocket guard: shrink the offset where the outer probe would sit
    # too close to (or inside) another wall. Detached: delta carries no graph.
    delta = torch.full((P,), delta0, device=device, dtype=dtype)
    with torch.no_grad():
        for _ in range(3):
            phi2 = sdf_fn(verts.detach() + (2.0 * delta)[:, None] * normals.detach())
            bad = phi2 < 0.5 * delta
            if not bool(bad.any()):
                break
            delta = torch.where(bad, delta * 0.5, delta)
        n_bad = int((phi2 < 0.5 * delta).sum())
        if n_bad:
            logger.warning("%d shell probes remain close to the wall after shrinking.", n_bad)

    shell1 = verts + delta[:, None] * normals
    shell2 = verts + (2.0 * delta)[:, None] * normals

    vol_base = base_volume_cloud(cfg).to(device=device, dtype=dtype)
    with torch.no_grad():
        keep = sdf_fn(vol_base) > 0.0
    vol_pts = vol_base[keep]

    pos = torch.cat([verts, shell1, shell2, vol_pts], dim=0)
    phi = sdf_fn(pos).reshape(-1, 1)

    nrm_feat = torch.zeros_like(pos)
    nrm_feat[:P] = normals

    feats = torch.cat([pos, phi, nrm_feat], dim=1)
    roles = torch.cat(
        [
            torch.full((P,), ROLE_SURFACE, dtype=torch.int8),
            torch.full((P,), ROLE_SHELL1, dtype=torch.int8),
            torch.full((P,), ROLE_SHELL2, dtype=torch.int8),
            torch.full((int(vol_pts.shape[0]),), ROLE_VOLUME, dtype=torch.int8),
        ]
    ).to(device)

    return QueryCloud(
        feats=feats,
        roles=roles,
        n_surface=P,
        unit_normals=normals,
        area_normals=area_normals,
        delta=delta.detach(),
    )

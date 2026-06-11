"""Differentiable volume/centroid constraints from the SDF (no mesh needed).

Replaces the FlexiCubes-tet-mesh volume computation in the ``sdf_hex``
pipeline: the solid volume is a smoothed-Heaviside quadrature of the SDF on
a fixed grid over the design domain, fully differentiable to the lattice
parameters.  The initial volume must be computed with the same function so
the constraint is bias-consistent.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from .sdf_field import PhysicalSDF

logger = logging.getLogger(__name__)


def _smoothed_heaviside(t: torch.Tensor, eps: float) -> torch.Tensor:
    """C1 smoothed Heaviside: 0 for t < -eps, 1 for t > eps."""
    x = torch.clamp(t / eps, min=-1.0, max=1.0)
    return 0.5 * (1.0 + x + torch.sin(np.pi * x) / np.pi)


def volume_centroid_from_sdf(
    sdf: PhysicalSDF,
    grid_res: int = 64,
    eps_cells: float = 1.5,
    chunk: int = 65536,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solid volume and centroid, differentiable to the SDF parameters.

    Parameters
    ----------
    grid_res : int
        Number of quadrature cells along the longest design-domain axis;
        other axes are scaled for near-cubic quadrature cells.
    eps_cells : float
        Heaviside half-width in units of the quadrature cell size.
    """
    lo = sdf.design_domain[0].detach().cpu().numpy().astype(np.float64)
    hi = sdf.design_domain[1].detach().cpu().numpy().astype(np.float64)
    extent = hi - lo
    n = np.maximum(1, np.round(grid_res * extent / extent.max()).astype(int))
    dx = extent / n

    axes = [lo[i] + (np.arange(n[i]) + 0.5) * dx[i] for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    w = float(np.prod(dx))

    # Heaviside half-width in SDF value units.
    eps = float(eps_cells * dx.mean() * sdf.dist_scale)

    V = None
    Cx = None
    for start in range(0, len(grid), chunk):
        x = torch.as_tensor(
            grid[start : start + chunk], dtype=torch.float32, device=sdf.device
        )
        # Solid where phi < 0.
        H = _smoothed_heaviside(-sdf.phi(x), eps)
        v_chunk = H.sum()
        c_chunk = (H[:, None] * x).sum(dim=0)
        V = v_chunk if V is None else V + v_chunk
        Cx = c_chunk if Cx is None else Cx + c_chunk

    volume = w * V
    centroid = Cx / V.clamp_min(1e-30)
    return volume, centroid

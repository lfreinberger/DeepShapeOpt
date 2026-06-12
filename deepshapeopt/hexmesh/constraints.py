"""Differentiable volume/centroid constraints from the wall triangulation.

The solid volume and centroid are computed with the divergence theorem over
the (closed) triangulated wall surface of the hex mesh, so the constraint
measures exactly the body the CFD sees.  The gradient flows through the
differentiable snapped wall points into the lattice parameters.

An SDF-quadrature variant (smoothed Heaviside on a fixed grid) was used
before but turned out to be exploitable: the optimizer can flatten |phi|
in the fluid region — invisible to the mesher — and inflate the reported
volume without growing the body.
"""

from __future__ import annotations

import torch


def volume_centroid_from_wall(
    points: torch.Tensor, tris: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solid volume and centroid enclosed by a closed triangulated surface.

    Signed-tetrahedra sum with respect to the origin (divergence theorem).
    Orientation-agnostic: the volume is returned as an absolute value and
    the centroid is invariant under flipping the triangle orientation.
    Differentiable with respect to ``points``.

    Parameters
    ----------
    points : torch.Tensor
        [P, 3] vertex positions (the snapped wall points).
    tris : torch.Tensor
        [T, 3] integer triangle indices into ``points``; must form a
        closed (watertight) surface.
    """
    tris = tris.to(points.device)
    v0 = points[tris[:, 0]]
    v1 = points[tris[:, 1]]
    v2 = points[tris[:, 2]]
    tet_vol = (v0 * torch.linalg.cross(v1, v2)).sum(dim=1) / 6.0
    vol_signed = tet_vol.sum()
    centroid = (tet_vol[:, None] * (v0 + v1 + v2)).sum(dim=0) / (4.0 * vol_signed)
    return vol_signed.abs(), centroid

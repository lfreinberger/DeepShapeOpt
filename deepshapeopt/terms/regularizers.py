"""Parameter-space regularizers: lattice smoothness and proximity to the start design."""
from __future__ import annotations

import torch


def control_lattice_smoothness_penalty(param, n_ctrl_per_dim, latent_dim, param_ref=None):
    """Spatial smoothness (graph-Dirichlet energy) over the B-spline latent control lattice.

    Unlike the other penalties here, this acts purely in *latent design-variable space* (not the
    SDF/mesh): it penalizes large differences between neighboring control-point latent vectors, so a
    single control point cannot jump into a strange decoder region while its neighbors stay normal.
    Use it as a reconstruction-loss term and/or an optimization penalty to keep the latent field
    spatially coherent (a soft, manifold-respecting regularizer complementary to the PCA basis).

    With ``param_ref`` the energy acts on the *update field* ``delta = param - param_ref`` instead
    of the absolute latent field. Locked control points keep delta = 0, so minimizing the Dirichlet
    energy of delta tapers the shape change smoothly to zero at locked boundaries (e.g. the frozen
    outlet face) instead of collapsing within one control-point spacing -- without penalizing the
    legitimate spatial variation already present in ``param_ref`` (the initial reconstruction).

    ``param`` is the flattened control points (n_ctrl_total * latent_dim,) or (n_ctrl_total,
    latent_dim). splinepy orders control points x-fastest, then y, then z, so the C-order reshape is
    ``(nz, ny, nx, latent_dim)`` -- x-neighbors lie along axis 2, y along axis 1, z along axis 0.

    Intensive form: mean squared difference per adjacent (pair, component), so the value is grid- and
    latent-dim-independent and a tuned weight transfers across tilings. Returns
    ``(P_value_detached, dP_param)`` with ``dP_param`` = dP/dparam (same flat shape as ``param``).
    """
    nx, ny, nz = (int(n) for n in n_ctrl_per_dim)
    field = param if param_ref is None else (param - param_ref.detach())
    flat = field.reshape(-1)
    x = flat.reshape(nz, ny, nx, int(latent_dim))

    sq, cnt = 0.0, 0
    if nx > 1:
        dx = x[:, :, 1:, :] - x[:, :, :-1, :]
        sq = sq + dx.pow(2).sum(); cnt += dx.numel()
    if ny > 1:
        dy = x[:, 1:, :, :] - x[:, :-1, :, :]
        sq = sq + dy.pow(2).sum(); cnt += dy.numel()
    if nz > 1:
        dz = x[1:, :, :, :] - x[:-1, :, :, :]
        sq = sq + dz.pow(2).sum(); cnt += dz.numel()

    if cnt == 0:  # degenerate 1x1x1 lattice: nothing to smooth
        return (param.sum() * 0.0).detach(), torch.zeros_like(param)

    P = sq / cnt
    g = torch.autograd.grad(P, param, retain_graph=False, allow_unused=True)[0]
    if g is None:
        g = torch.zeros_like(param)
    return P.detach().to(param.dtype), g.to(param.dtype)

"""Differentiable drag integration from predicted (U, p) fields.

Kinematic convention matching the OpenFOAM ``force`` objective of the drag
case (incompressible, rho = 1, ``normalise false``): ``J`` is the raw force
integral in the given direction,

    J = sum_v A_v * ( -p_v * n_hat_v + nu * dU_t/dn |_v ) . e_dir

with ``n_hat`` the into-fluid unit vertex normal and ``p`` the kinematic
pressure. The wall-shear term uses a second-order one-sided difference of the
predicted velocity at the two probe shells (no-slip gives U(0) = 0):

    dU/dn ~= (4 U(delta) - U(2 delta)) / (2 delta)

projected onto the tangent plane. Signs are validated against OpenFOAM's own
fields and the Stokes-sphere anchor in phase P2 (see plan).
"""

from __future__ import annotations

import torch

from .query_points import QueryCloud


def drag_from_fields(
    U: torch.Tensor,
    p: torch.Tensor,
    cloud: QueryCloud,
    nu: float,
    direction=(1.0, 0.0, 0.0),
) -> tuple[torch.Tensor, dict]:
    """Integrate the drag force from per-point predictions.

    Parameters
    ----------
    U : [N, 3] predicted velocity (de-normalized, physical units).
    p : [N] or [N, 1] predicted kinematic pressure.
    cloud : the query cloud the predictions were made on (defines ordering:
        surface [0:P], shell1 [P:2P], shell2 [2P:3P], volume rest).
    nu : kinematic viscosity (physical units, from the case config).
    direction : drag direction, defaults to +x.

    Returns
    -------
    (J, diagnostics) with ``J`` a scalar tensor carrying the autograd graph
    and ``diagnostics`` containing the per-vertex wall traction [P, 3]
    (graph-carrying) plus detached scalar terms.
    """
    P = cloud.n_surface
    p = p.reshape(-1)
    if U.shape[0] != cloud.n_points or p.shape[0] != cloud.n_points:
        raise ValueError(
            f"prediction size mismatch: U {tuple(U.shape)}, p {tuple(p.shape)}, "
            f"cloud has {cloud.n_points} points"
        )

    n_hat = cloud.unit_normals
    e_dir = torch.as_tensor(direction, dtype=U.dtype, device=U.device)

    U1 = U[P : 2 * P]
    U2 = U[2 * P : 3 * P]
    dUdn = (4.0 * U1 - U2) / (2.0 * cloud.delta)[:, None]
    # Tangential projection removes the spurious normal component.
    dUdn_t = dUdn - (dUdn * n_hat).sum(dim=1, keepdim=True) * n_hat
    traction = -p[:P, None] * n_hat + nu * dUdn_t  # force per area on the body

    area = cloud.area_normals.norm(dim=1)  # A_v (unit normals => |A_v n_hat| = A_v)
    # Pressure term uses the area-weighted normal directly (exact quadrature),
    # the viscous term the scalar vertex area.
    J_p = -(p[:P] * (cloud.area_normals @ e_dir)).sum()
    J_visc = ((nu * dUdn_t @ e_dir) * area).sum()
    J = J_p + J_visc

    diagnostics = {
        "J_p": float(J_p.detach()),
        "J_visc": float(J_visc.detach()),
        "traction": traction,
    }
    return J, diagnostics

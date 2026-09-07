"""Differentiable drag integration from predicted fields.

Convention matching the OpenFOAM adjoint ``force`` objective (see
``objectiveForce.C``: ``J = F / (0.5 * UInf^2 * Aref)``, i.e. a force
coefficient; with the drag case's ``UInf = Aref = 1`` this is ``2 F``).
The raw force integral in the given direction is

    J = sum_v A_v * ( -p_v * n_hat_v + t_v ) . e_dir

with ``n_hat`` the into-fluid unit vertex normal, ``p`` the kinematic
pressure and ``t_v`` the viscous traction on the body. Two sources for ``t_v``:

- ``tau_w`` given ("tau" mode, 7-channel checkpoints): the predicted wall
  shear stress in OpenFOAM's ``wallShearStress`` convention, whose vector
  points opposite to the traction on the body, so ``t_v = -tau_w,v``. It is
  used as the FULL vector, deliberately not projected onto the tangent
  plane: the objective integrates ``devReff & Sf`` including its normal
  component (~20% rms on the point-interpolated patch data); projecting
  moves the reproduction of ``dragadjS1`` from 1.01 to 1.09-1.15 (measured
  on the stored training targets). Do not "fix" this for symmetry with the
  FD path.
- ``tau_w`` None ("fd" mode, 4-channel checkpoints): a second-order
  one-sided difference of the predicted velocity at the two probe shells
  (no-slip gives U(0) = 0), ``dU/dn ~= (4 U(delta) - U(2 delta)) / (2 delta)``,
  projected onto the tangent plane, ``t_v = nu * dU_t/dn``.

Signs were validated against OpenFOAM's own fields and the Stokes-sphere
anchor in phase P2; the tau convention on 60 stored samples (phase P7).
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
    u_inf: float = 1.0,
    a_ref: float = 1.0,
    visc_scale: float = 1.0,
    pressure_scale: float = 1.0,
    tau_w: torch.Tensor | None = None,
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
    visc_scale, pressure_scale : calibration factors absorbing the
        quadrature bias of the FD path and the trained network's systematic
        offsets (see ``scripts/calibrate_surrogate_bias.py``); 1.0 for
        ground-truth fields.
    tau_w : optional [P, 3] or [N, 3] predicted wall shear stress
        (OpenFOAM convention, wall rows first). Selects "tau" mode; the
        probe shells are then not read at all.

    Returns
    -------
    (J, diagnostics) with ``J`` a scalar tensor carrying the autograd graph
    and ``diagnostics`` containing the per-vertex wall traction [P, 3]
    (graph-carrying, same scales as J, not clamped) plus detached scalars.
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

    if tau_w is None:
        viscous_mode = "fd"
        U1 = U[P : 2 * P]
        U2 = U[2 * P : 3 * P]
        dUdn = (4.0 * U1 - U2) / (2.0 * cloud.delta)[:, None]
        # Tangential projection removes the spurious normal component.
        dUdn_t = dUdn - (dUdn * n_hat).sum(dim=1, keepdim=True) * n_hat
        visc_vec = nu * dUdn_t
    else:
        viscous_mode = "tau"
        if tau_w.ndim != 2 or tau_w.shape[1] != 3 or tau_w.shape[0] not in (P, cloud.n_points):
            raise ValueError(
                f"tau_w must be [P, 3] or [N, 3] (P={P}, N={cloud.n_points}), "
                f"got {tuple(tau_w.shape)}"
            )
        # OpenFOAM's wallShearStress points opposite to the traction on the
        # body; full vector, no tangential projection (see module docstring).
        visc_vec = -tau_w[:P]

    area = cloud.area_normals.norm(dim=1)  # A_v (unit normals => |A_v n_hat| = A_v)
    # Pressure term uses the area-weighted normal directly (exact quadrature),
    # the viscous term the scalar vertex area.
    denom = 0.5 * u_inf**2 * a_ref  # objectiveForce.C: rhoInf NOT in denom
    J_p = -pressure_scale * (p[:P] * (cloud.area_normals @ e_dir)).sum() / denom
    J_visc_raw = visc_scale * ((visc_vec @ e_dir) * area).sum() / denom
    # Physics guard: total viscous drag cannot be negative for an external
    # body. Clamping removes the optimizer's incentive to exploit unphysical
    # predictions far from the training distribution (the failure mode
    # observed in the first pure-surrogate trial run).
    J_visc = J_visc_raw.clamp_min(0.0)
    J = J_p + J_visc

    # Force per area on the body with the same scales as J; a diagnostic and
    # the stand-in for the adjoint sensitivity field, deliberately unclamped.
    traction = -pressure_scale * p[:P, None] * n_hat + visc_scale * visc_vec

    diagnostics = {
        "J_p": float(J_p.detach()),
        "J_visc": float(J_visc.detach()),
        "J_visc_raw": float(J_visc_raw.detach()),
        "visc_clamped": bool(J_visc_raw.detach() < 0),
        "viscous_mode": viscous_mode,
        "force": float(J.detach()) * denom,
        "traction": traction,
    }
    return J, diagnostics

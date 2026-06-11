"""Differentiable snapping of castellated wall points onto the SDF zero set.

The projection runs a few detached Newton steps ``x <- x - phi * grad/|grad|^2``
followed by one final step that stays in the autograd graph of the lattice
parameters.  With the spatial gradient detached, the snapped position is

    x*(lam) = x0 + lam * (x_d - x0) - lam * phi(x_d; param) * n_hat / |grad phi|

so ``d x*/d param = -lam * (d phi/d param) * n_hat / |grad phi|`` -- exactly
the Hadamard shape-derivative form, without noisy second derivatives of the
network.

The per-point ``lam`` scales the *total* displacement from the castellated
position (not just the final Newton step), so the quality guard can pull
degenerate points back toward the staircase while preserving the gradient
direction; ``lam = 0`` leaves a point unsnapped (and gradient-free).
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
import torch

from .sdf_field import PhysicalSDF

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class SnapHandle:
    """Snapped wall points with a re-evaluable differentiable final step."""

    x0: torch.Tensor  # [P, 3] original castellated positions (detached)
    x_d: torch.Tensor  # [P, 3] after detached Newton steps
    f_param: torch.Tensor  # [P] phi(x_d) carrying the parameter graph
    dir_vec: torch.Tensor  # [P, 3] n_hat / |grad phi| (detached)
    lam: torch.Tensor  # [P] displacement scaling (detached)

    def x_star(self) -> torch.Tensor:
        """Differentiable snapped positions for the current ``lam``."""
        blended = self.x0 + self.lam[:, None] * (self.x_d - self.x0)
        return blended - (self.lam * self.f_param)[:, None] * self.dir_vec

    def residuals(self) -> np.ndarray:
        """|phi| at the fully-snapped detached positions (Newton residual)."""
        return self.f_param.detach().abs().cpu().numpy()

    def reduce_lambda(self, point_mask: np.ndarray, factor: float = 0.75) -> None:
        mask = torch.as_tensor(point_mask, device=self.lam.device, dtype=torch.bool)
        self.lam = torch.where(mask, self.lam * factor, self.lam)

    def zero_lambda(self, point_mask: np.ndarray) -> None:
        mask = torch.as_tensor(point_mask, device=self.lam.device, dtype=torch.bool)
        self.lam = torch.where(mask, torch.zeros_like(self.lam), self.lam)


def snap_wall_points(
    sdf: PhysicalSDF,
    x0_phys: np.ndarray,
    h_local: np.ndarray,
    snap_iters: int = 4,
    max_disp_frac: float = 0.7,
    grad_eps: float = 1e-10,
) -> SnapHandle:
    """Project wall points onto the zero level set.

    Parameters
    ----------
    x0_phys : [P, 3] float64
        Castellated wall point positions.
    h_local : [P] float64
        Local cell size per point (finest adjacent wall face); caps the total
        displacement at ``max_disp_frac * h_local``.
    snap_iters : int
        Total Newton steps; the last one is the differentiable step.
    """
    device = sdf.device
    x0 = torch.as_tensor(np.asarray(x0_phys, dtype=np.float32), device=device)
    h = torch.as_tensor(np.asarray(h_local, dtype=np.float32), device=device)
    max_disp = max_disp_frac * h

    x = x0.clone()
    with torch.no_grad():
        for _ in range(max(0, snap_iters - 1)):
            f, g = sdf.phi_and_grad(x)
            g_norm2 = (g * g).sum(dim=1).clamp_min(grad_eps)
            x_new = x - (f / g_norm2)[:, None] * g
            disp = x_new - x0
            disp_norm = disp.norm(dim=1).clamp_min(1e-30)
            scale = torch.clamp(max_disp / disp_norm, max=1.0)
            x = x0 + scale[:, None] * disp

    x_d = x.detach()
    f_d, g_d = sdf.phi_and_grad(x_d)
    g_norm = g_d.norm(dim=1).clamp_min(np.sqrt(grad_eps))
    # n_hat / |grad phi| == grad phi / |grad phi|^2
    dir_vec = (g_d / (g_norm**2)[:, None]).detach()

    # Differentiable phi at the detached evaluation points.
    f_param = sdf.phi_ext(x_d)

    # Cap the total displacement (Newton path + final step) at max_disp.
    with torch.no_grad():
        total = (x_d - x0).norm(dim=1) + f_d.abs() / g_norm
        lam = torch.clamp(max_disp / total.clamp_min(1e-30), max=1.0)

    handle = SnapHandle(x0=x0, x_d=x_d, f_param=f_param, dir_vec=dir_vec, lam=lam)

    res = handle.residuals()
    logger.debug(
        "Snap: %d points, residual |phi| median %.3e max %.3e, "
        "lam<1 for %d points",
        len(x0), float(np.median(res)), float(res.max()),
        int((lam < 1.0).sum().item()),
    )
    return handle

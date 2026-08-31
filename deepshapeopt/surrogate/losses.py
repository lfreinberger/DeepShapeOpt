"""Losses and metrics for the flow surrogate."""

from __future__ import annotations

import torch


def rel_l2(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Relative L2 norm over masked rows: ||pred - target|| / ||target||."""
    if not bool(mask.any()):
        return pred.sum() * 0.0
    diff = (pred[mask] - target[mask]).norm()
    denom = target[mask].norm().clamp_min(1e-8)
    return diff / denom


def field_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    role: torch.Tensor,
    valid: torch.Tensor,
    w_p: float = 1.0,
    w_U: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """Combined loss on normalized fields.

    Pressure is supervised on the wall surface (role 0, where the drag
    integral needs it) plus all valid off-surface points at lower implicit
    weight through the shared denominator; velocity only off-surface (it is
    identically zero on the wall by no-slip).
    """
    surf = role == 0
    off = (~surf) & valid
    p_pred, p_tgt = pred[:, 3], target[:, 3]
    U_pred, U_tgt = pred[:, :3], target[:, :3]

    l_p_surf = rel_l2(p_pred, p_tgt, surf & valid)
    l_p_off = rel_l2(p_pred, p_tgt, off)
    l_U = rel_l2(U_pred, U_tgt, off)

    loss = w_p * (l_p_surf + 0.25 * l_p_off) + w_U * l_U
    parts = {
        "p_surf": float(l_p_surf.detach()),
        "p_off": float(l_p_off.detach()),
        "U_off": float(l_U.detach()),
    }
    return loss, parts

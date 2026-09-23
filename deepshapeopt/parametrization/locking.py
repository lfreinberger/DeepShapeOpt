"""Locked control points: Greville-point boxes, masks and gradient masking.

Control points whose Greville abscissa lies inside a lock box keep their values; their
gradient entries are zeroed before every optimizer step. Boxes come from the
``parametrization.lock`` config block (design-box faces, physical or normalized boxes, or a
named layout).
"""
from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

FACE_AXES = {"x": 0, "y": 1, "z": 2}


def make_locked_masks(param: torch.Tensor, locked_idx: torch.Tensor):
    """Return masks and fixed values for locked control points."""
    n_ctrl, latent_dim = param.shape

    mask_locked_cp = torch.zeros(n_ctrl, dtype=torch.bool, device=param.device)
    if locked_idx.numel() > 0:
        mask_locked_cp[locked_idx] = True

    mask_locked_flat = mask_locked_cp[:, None].expand(n_ctrl, latent_dim).reshape(-1)
    locked_values = param[mask_locked_cp].detach().clone()
    return mask_locked_cp, mask_locked_flat, locked_values

def mask_locked_gradient(grad: torch.Tensor, mask_locked_flat: torch.Tensor):
    """Mask a single gradient tensor for MMA (no constraint gradient)."""
    grad_flat = grad.reshape(-1, 1).clone()
    mask_flat = mask_locked_flat.reshape(-1)

    if grad_flat.shape[0] != mask_flat.shape[0]:
        raise ValueError(
            f"Mask and gradient size mismatch: "
            f"grad_flat has {grad_flat.shape[0]} entries, "
            f"mask has {mask_flat.shape[0]} entries."
        )

    if mask_flat.any():
        grad_flat[mask_flat] = 0.0
    return grad_flat

def _greville_1d(U, p):
    n = len(U) - p - 1
    if n <= 0:
        raise ValueError(f"Invalid knot vector length {len(U)} for degree {p}")
    if p == 0:
        return 0.5 * (U[:n] + U[1:n+1])
    return np.array([np.sum(U[i+1:i+p+1]) / p for i in range(n)], dtype=float)

def greville_points_3d(spline, order):
    """Return pts (N,3) Greville points in parameter space and (n0,n1,n2)."""
    degrees = np.array(spline.degrees, dtype=int)
    kvs = [np.asarray(kv, dtype=float) for kv in spline.knot_vectors]

    g0 = _greville_1d(kvs[0], degrees[0])
    g1 = _greville_1d(kvs[1], degrees[1])
    g2 = _greville_1d(kvs[2], degrees[2])

    n0, n1, n2 = len(g0), len(g1), len(g2)
    ids = np.arange(n0 * n1 * n2)
    I, J, K = np.unravel_index(ids, (n0, n1, n2), order=order)

    pts = np.column_stack([g0[I], g1[J], g2[K]])  # (N,3)
    return pts, (n0, n1, n2)

def locked_indices_from_bboxes(spline, bboxes, device="cuda", order="F"):
    pts, _ = greville_points_3d(spline, order=order)
    if not bboxes:
        return torch.empty(0, dtype=torch.long, device=device)

    bboxes = np.array([
        [
            [x.detach().cpu().item() if torch.is_tensor(x) else float(x) for x in bmin],
            [x.detach().cpu().item() if torch.is_tensor(x) else float(x) for x in bmax],
        ]
        for bmin, bmax in bboxes
    ], dtype=float)

    bmin = bboxes[:, 0]
    bmax = bboxes[:, 1]

    inside = np.all((pts[:, None, :] >= bmin) & (pts[:, None, :] <= bmax), axis=2)
    lock = np.any(inside, axis=1)

    return torch.as_tensor(np.nonzero(lock)[0], dtype=torch.long, device=device)

def outlet_face_and_inlet_rim_boxes(box_norm, safety):
    (x_min, y_min, z_min), (x_max, y_max, z_max) = box_norm

    lock_domain = [
        # 1) full face at x = x_min
        (
            [x_min, y_min - safety, z_min - safety],
            [x_min + safety, y_max + safety, z_max + safety],
        ),

        # 2) strip at x = x_max, z = z_min
        (
            [x_max - safety, y_min - safety, z_min - safety],
            [x_max, y_max + safety, z_min + safety],
        ),

        # 3) strip at x = x_max, z = z_max
        (
            [x_max - safety, y_min - safety, z_max - safety],
            [x_max, y_max + safety, z_max + safety],
        ),

        # 4) strip at x = x_max, y = y_min
        (
            [x_max - safety, y_min - safety, z_min - safety],
            [x_max, y_min + safety, z_max + safety],
        ),

        # 5) strip at x = x_max, y = y_max
        (
            [x_max - safety, y_max - safety, z_min - safety],
            [x_max, y_max + safety, z_max + safety],
        ),
    ]

    return lock_domain


def lock_boxes_from_config(lock_cfg, frame) -> list:
    """Lock boxes in normalized coordinates from a ``parametrization.lock`` config.

    ``faces`` locks whole design-box faces (``"x_min"`` ... ``"z_max"``), ``boxes_physical``
    and ``boxes_norm`` arbitrary boxes, ``layout`` a named layout
    (``"outlet_face_and_inlet_rim"``: the x_min face plus the four edge strips of the x_max
    face). Boxes are unioned; an empty configuration locks nothing.
    """
    safety = float(lock_cfg.safety)
    boxes = []
    if lock_cfg.layout == "outlet_face_and_inlet_rim":
        boxes += outlet_face_and_inlet_rim_boxes(frame.box_norm, safety=safety)

    lo = [float(v) for v in frame.box_norm[0]]
    hi = [float(v) for v in frame.box_norm[1]]
    for face in lock_cfg.faces:
        axis_name, _, side = str(face).partition("_")
        if axis_name not in FACE_AXES or side not in ("min", "max"):
            raise ValueError(f"Invalid lock face {face!r}; expected e.g. 'x_min' or 'z_max'")
        i = FACE_AXES[axis_name]
        bmin = [c - safety for c in lo]
        bmax = [c + safety for c in hi]
        if side == "min":
            bmax[i] = lo[i] + safety
        else:
            bmin[i] = hi[i] - safety
        boxes.append((bmin, bmax))

    def to_norm(corner):
        pt = torch.tensor([float(v) for v in corner], dtype=frame.center.dtype, device=frame.center.device)
        return frame.to_norm(pt).tolist()

    for bmin, bmax in lock_cfg.boxes_physical:
        bmin, bmax = to_norm(bmin), to_norm(bmax)
        boxes.append(([v - safety for v in bmin], [v + safety for v in bmax]))
    for bmin, bmax in lock_cfg.boxes_norm:
        boxes.append(([float(v) - safety for v in bmin], [float(v) + safety for v in bmax]))

    if not boxes:
        logger.warning("lock: no boxes configured; every control point is free to move")
    else:
        logger.info("lock: %d boxes", len(boxes))
    return boxes

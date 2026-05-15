import numpy as np
import torch


def make_locked_masks(param: torch.Tensor, locked_idx: torch.Tensor):
    """Return masks and fixed values for locked control points."""
    n_ctrl, latent_dim = param.shape

    mask_locked_cp = torch.zeros(n_ctrl, dtype=torch.bool, device=param.device)
    if locked_idx.numel() > 0:
        mask_locked_cp[locked_idx] = True

    mask_locked_flat = mask_locked_cp[:, None].expand(n_ctrl, latent_dim).reshape(-1)
    locked_values = param[mask_locked_cp].detach().clone()
    return mask_locked_cp, mask_locked_flat, locked_values


def mask_grads_for_mma(dJ: torch.Tensor, dV: torch.Tensor, mask_locked_flat: torch.Tensor):
    """Flatten two gradients for MMA and zero locked entries."""
    dJ_flat = dJ.reshape(-1, 1).clone()
    dV_flat = dV.reshape(-1, 1).clone()
    mask_flat = mask_locked_flat.reshape(-1)
    if mask_flat.any():
        dJ_flat[mask_flat] = 0.0
        dV_flat[mask_flat] = 0.0
    return dJ_flat, dV_flat


def mask_single_grad_for_mma(grad: torch.Tensor, mask_locked_flat: torch.Tensor):
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


def make_lock_domain(box_param_sapce, safety):
    (x_min, y_min, z_min), (x_max, y_max, z_max) = box_param_sapce

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

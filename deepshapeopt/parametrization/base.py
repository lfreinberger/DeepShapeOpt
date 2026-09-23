"""Design parametrization protocol and the design space the optimizer moves in."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from deepshapeopt.config.schema import LockConfig, OptimizerConfig, PCAConfig

from .locking import lock_boxes_from_config, locked_indices_from_bboxes, make_locked_masks, mask_locked_gradient

logger = logging.getLogger(__name__)


class Parametrization(Protocol):
    """A design parametrization: control points, their spline, and the design SDF."""

    frame: Any
    mesh_orig: Any
    spline_sp: Any
    control_dims: list[int]

    @property
    def param(self) -> torch.nn.Parameter: ...

    @property
    def design_sdf(self): ...

    def save(self, path: Path) -> None: ...

    def load(self, path: Path) -> None: ...

    def export_debug(self, out_dir: Path, locked_idx) -> None: ...

    def export_iteration(self, out_dir: Path, iteration: int, series_dir: Path | None) -> None: ...


def load_parameter_file(path: Path, like: torch.Tensor) -> torch.Tensor:
    """Read a saved parameter tensor (``[tensor]`` list or bare tensor) shaped like ``like``."""
    loaded = torch.load(Path(path), map_location=like.device, weights_only=False)
    loaded = loaded[0] if isinstance(loaded, (list, tuple)) else loaded
    if tuple(loaded.shape) != tuple(like.shape):
        raise ValueError(f"{path} holds shape {tuple(loaded.shape)}, the design parameters have {tuple(like.shape)}")
    return loaded.detach().to(device=like.device, dtype=like.dtype)


class DesignSpace:
    """Maps between the control points and the vector the MMA moves.

    Locked control points (from ``parametrization.lock``) keep their values and get zero
    gradient. With PCA enabled the optimizer moves whitened coefficients of deltas around the
    reconstruction (``z = z0 + (c * scale) @ V_k.T``); otherwise it moves the control points.
    """

    def __init__(self, param: torch.nn.Parameter, spline_sp, frame, lock_cfg: LockConfig,
                 opt_cfg: OptimizerConfig, pca_cfg: PCAConfig | None = None,
                 model_path: str | None = None, checkpoint: str = "latest"):
        self.param = param
        device = param.device
        boxes = lock_boxes_from_config(lock_cfg, frame)
        locked_idx = locked_indices_from_bboxes(spline_sp, boxes, device=device, order="F")
        if locked_idx.numel() > 0:
            logger.info("Locked control points: %d", locked_idx.numel())
        self.locked_idx = locked_idx
        self.mask_locked_cp, self.mask_locked_flat, self.locked_values = make_locked_masks(param, locked_idx)
        self.n_free = int((~self.mask_locked_flat.reshape(-1).bool()).sum().item())

        self.pca = None
        self.coeffs = None
        self.mask_locked_coeff = None
        if pca_cfg is not None and pca_cfg.enabled:
            from .pca import build_pca_basis

            k = int(pca_cfg.n_components)
            basis = build_pca_basis(model_path, checkpoint, k, device=device, cache_path=pca_cfg.cache_path)
            basis = basis.to(device=device, dtype=param.dtype)
            basis.set_reference(param.detach().clone())
            n_ctrl = param.shape[0]
            self.pca = basis
            self.coeffs = torch.zeros((n_ctrl, k), device=device, dtype=param.dtype)
            self.mask_locked_coeff = self.mask_locked_cp[:, None].expand(n_ctrl, k).reshape(-1)
            self.bounds = np.full((self.coeffs.numel(), 2), pca_cfg.bounds)
            self.max_step = float(pca_cfg.max_step) if pca_cfg.max_step is not None else float(opt_cfg.max_step)
            logger.info("PCA delta reduction: %d control points x %d modes = %d design variables (was %d)",
                        n_ctrl, k, self.coeffs.numel(), param.numel())
        else:
            self.bounds = np.full((param.numel(), 2), opt_cfg.bounds)
            self.max_step = float(opt_cfg.max_step)

    @property
    def vector(self) -> torch.Tensor:
        """The tensor the MMA optimizes in place (control points or PCA coefficients)."""
        return self.coeffs if self.pca is not None else self.param

    @property
    def n_vars(self) -> int:
        return int(self.vector.numel())

    def grad_to_vector(self, grad_param: torch.Tensor) -> torch.Tensor:
        """Parameter-shaped gradient -> masked flat gradient of the optimizer vector, (n, 1)."""
        if self.pca is not None:
            return mask_locked_gradient(self.pca.project_grad(grad_param), self.mask_locked_coeff)
        return mask_locked_gradient(grad_param, self.mask_locked_flat)

    def sync_param(self) -> None:
        """After an optimizer step: rebuild the control points from the PCA coefficients."""
        if self.pca is None:
            return
        with torch.no_grad():
            z = self.pca.to_latent(self.coeffs)
            if self.mask_locked_cp.any():
                z[self.mask_locked_cp] = self.locked_values
            self.param.copy_(z)

    def write_vector(self, x_np) -> None:
        """Set the design from a flat optimizer vector (candidate of a callback)."""
        with torch.no_grad():
            if self.pca is not None:
                c = torch.as_tensor(np.asarray(x_np).reshape(self.coeffs.shape), dtype=self.param.dtype,
                                    device=self.param.device)
                z = self.pca.to_latent(c)
                z[self.mask_locked_cp] = self.locked_values
                self.param.copy_(z)
            else:
                self.param.copy_(torch.as_tensor(np.asarray(x_np).reshape(self.param.shape),
                                                 dtype=self.param.dtype, device=self.param.device))

    def free_direction(self, seed: int) -> torch.Tensor:
        """Random direction of the free control points, max-norm 1 (noise probe)."""
        if self.pca is not None:
            raise ValueError("the noise probe perturbs the control points directly; disable PCA")
        gen = torch.Generator().manual_seed(int(seed))
        d = torch.randn(self.param.numel(), generator=gen, dtype=self.param.dtype).to(self.param.device)
        d[self.mask_locked_flat.reshape(-1).bool()] = 0.0
        return (d / d.abs().max()).reshape(self.param.shape)

    @property
    def mask_free_flat(self) -> torch.Tensor:
        return ~self.mask_locked_flat.reshape(-1).bool()

"""MMA step on the design space, with the cheap-term callbacks of GCMMA and restoration."""

from __future__ import annotations

import logging
from typing import Callable, Sequence

import numpy as np
import torch
from DeepSDFStruct.optimization import MMA

from deepshapeopt.config.schema import OptimizerConfig
from deepshapeopt.parametrization.base import DesignSpace
from deepshapeopt.terms.base import Row

logger = logging.getLogger(__name__)


class MMAOptimizer:
    def __init__(self, design_space: DesignSpace, cfg: OptimizerConfig, n_constraints: int):
        self.space = design_space
        self.cfg = cfg
        self.mma = MMA(design_space.vector.reshape(-1, 1), design_space.bounds,
                       max_step=design_space.max_step, n_constraints=max(1, int(n_constraints)))
        self.gcmma = dict(cfg.gcmma)
        self.restoration = dict(cfg.feasibility_restoration)
        if self.gcmma.get("enabled", False):
            logger.info("Hybrid GCMMA: max_inner=%d feas_tol=%.3g", int(self.gcmma.get("max_inner", 15)),
                        float(self.gcmma.get("feas_tol", 0.05)))
        if self.restoration.get("enabled", False):
            logger.info("Feasibility restoration: tol=%.3g max_steps=%d", float(self.restoration.get("tol", 0.005)),
                        int(self.restoration.get("max_steps", 8)))

    # -- rows ------------------------------------------------------------------------
    def assemble(self, rows: Sequence[Row], n_vars: int, device, dtype):
        """(G, dG, scales) in optimizer space; a single inactive dummy row when there is none."""
        if not rows:
            G = torch.zeros((1, 1), device=device, dtype=dtype)
            dG = torch.zeros((1, n_vars), device=device, dtype=dtype)
            return G, dG, [None]
        G = torch.tensor([[r.value] for r in rows], device=device, dtype=dtype)
        dG = torch.cat([self.space.grad_to_vector(r.grad).reshape(1, -1) for r in rows], dim=0)
        return G, dG, [r.scale for r in rows]

    def step(self, J_total: float, dJ_total: torch.Tensor, rows: Sequence[Row],
             cheap_eval: Callable[[np.ndarray, bool], tuple[np.ndarray, np.ndarray | None]] | None = None) -> None:
        """One MMA step. ``dJ_total`` is the parameter-shaped objective gradient.

        ``cheap_eval(x_np, with_grad)`` re-evaluates the cheap rows (in the order they appear
        in ``rows``) at a candidate design and returns raw ``value - target`` rows (and their
        optimizer-space gradients); it feeds the GCMMA inner loop and the restoration.
        """
        dJ = self.space.grad_to_vector(dJ_total)
        G, dG, scales = self.assemble(rows, dJ.shape[0], dJ.device, dJ.dtype)
        F = torch.tensor([[float(J_total)]], dtype=dJ.dtype)
        cheap_rows = [i for i, r in enumerate(rows) if r.cheap]
        kwargs = {}
        if cheap_rows and cheap_eval is not None:
            if self.gcmma.get("enabled", False):
                kwargs.update(
                    geom_eval=lambda x: cheap_eval(x, False)[0], geom_rows=cheap_rows,
                    max_inner=int(self.gcmma.get("max_inner", 15)), feas_tol=float(self.gcmma.get("feas_tol", 0.05)),
                )
            if self.restoration.get("enabled", False):
                limit = self.restoration.get("step_limit")
                kwargs.update(
                    restore_eval=lambda x: cheap_eval(x, True), geom_rows=cheap_rows,
                    restore_tol=float(self.restoration.get("tol", 0.005)),
                    restore_max_steps=int(self.restoration.get("max_steps", 8)),
                    restore_step_limit=float(limit) if limit is not None else None,
                )
        self.mma.step(F, dJ, G, dG, G_scale=scales, **kwargs)
        self.space.sync_param()
        self.last_dJ = dJ
        self.last_dG = dG
        self.last_G = G

    # -- state of the last step ---------------------------------------------------------
    @property
    def x(self) -> np.ndarray:
        return self.mma.x

    @property
    def xold1(self) -> np.ndarray:
        return self.mma.xold1

    @property
    def step_vector(self) -> np.ndarray:
        return (self.mma.x - self.mma.xold1).reshape(-1, 1)

    @property
    def ch(self) -> float:
        return float(self.mma.ch)

    @property
    def kkt_norm(self) -> float:
        return float(self.mma.kkt_norm)

    @property
    def lam(self) -> np.ndarray:
        return np.asarray(self.mma.lam, dtype=float).reshape(-1)

    @property
    def F0(self) -> float:
        return float(np.asarray(self.mma.F0).reshape(-1)[0])

    @property
    def G_scale(self) -> np.ndarray:
        return np.asarray(self.mma.G_scale, dtype=float).reshape(-1)

    @property
    def max_step(self) -> float:
        return float(self.mma.max_step)

    @max_step.setter
    def max_step(self, value: float) -> None:
        self.mma.max_step = float(value)

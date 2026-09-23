"""Trust-region-style move-limit control from the realized vs. predicted merit change.

The first-order model predicts the change of the scaled merit ``J/F0 + sum_i lam_i G_i/scale_i``
along the committed step; the next evaluation gives the realized change. Their ratio shrinks
the move limit when the gradient stops predicting the merit and grows it back otherwise. The
run stops when the move limit collapses or the predicted decrease stays negligible.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class StepControl:
    def __init__(self, cfg: dict, max_step0: float):
        self.enabled = bool(cfg.get("enabled", False))
        self.rho_shrink = float(cfg.get("rho_shrink", 0.25))
        self.rho_grow = float(cfg.get("rho_grow", 0.75))
        self.shrink = float(cfg.get("shrink", 0.5))
        self.grow = float(cfg.get("grow", 2.0))
        self.max_step0 = float(max_step0)
        self.min_step = self.max_step0 * float(cfg.get("min_step_factor", 1.0 / 16.0))
        self.pred_tol_rel = float(cfg.get("pred_tol_rel", 1e-4))
        self.patience = int(cfg.get("patience", 3))
        self.pred = None       # (predicted merit change, merit before the step, multipliers)
        self.rho = float("nan")
        self.small_count = 0
        self.stop_reason: str | None = None
        if self.enabled:
            logger.info("step_control: rho_shrink=%.2f rho_grow=%.2f shrink=%.2f grow=%.2f max_step_0=%.4f "
                        "min_step=%.5f pred_tol_rel=%.1e patience=%d", self.rho_shrink, self.rho_grow,
                        self.shrink, self.grow, self.max_step0, self.min_step, self.pred_tol_rel, self.patience)

    def observe(self, optimizer, J_total: float, G: np.ndarray) -> None:
        """Realized merit change of the last step; adapts the move limit."""
        if not self.enabled or self.pred is None:
            return
        pred, merit_prev, lam_prev = self.pred
        merit_now = float(J_total) / optimizer.F0
        scales = optimizer.G_scale
        for i in range(min(len(lam_prev), len(G), len(scales))):
            merit_now += lam_prev[i] * float(G[i]) / scales[i]
        real = merit_now - merit_prev
        self.rho = real / pred if pred != 0.0 else float("nan")
        old = optimizer.max_step
        if not (pred < 0.0) or self.rho < self.rho_shrink:
            optimizer.max_step = max(old * self.shrink, 0.0)
        elif self.rho > self.rho_grow:
            optimizer.max_step = min(old * self.grow, self.max_step0)
        self.small_count = self.small_count + 1 if abs(pred) < self.pred_tol_rel else 0
        logger.info("step_control: predicted %+.3e realized %+.3e rho %+.3f -> max_step %.5f (small streak %d/%d)",
                    pred, real, self.rho, optimizer.max_step, self.small_count, self.patience)
        if optimizer.max_step < self.min_step:
            self.stop_reason = (f"move limit {optimizer.max_step:.5f} fell below {self.min_step:.5f}: "
                                "the gradient no longer predicts the merit")
        elif self.small_count >= self.patience:
            self.stop_reason = (f"predicted merit decrease below {self.pred_tol_rel:.1e} for "
                                f"{self.patience} consecutive iterations")

    def predict(self, optimizer, J_total: float) -> None:
        """First-order prediction of the merit change along the step just taken."""
        if not self.enabled:
            return
        dx = optimizer.step_vector
        F0 = optimizer.F0
        g = optimizer.last_dJ.detach().cpu().numpy().reshape(1, -1)
        pred = (g @ dx).item() / F0
        merit_prev = float(J_total) / F0
        lam = optimizer.lam.tolist()
        scales = optimizer.G_scale
        dG = optimizer.last_dG.detach().cpu().numpy()
        G0 = optimizer.last_G.detach().cpu().numpy().reshape(-1)
        for i in range(min(len(lam), dG.shape[0], len(scales))):
            pred += lam[i] * (dG[i].reshape(1, -1) @ dx).item() / scales[i]
            merit_prev += lam[i] * G0[i] / scales[i]
        self.pred = (pred, merit_prev, lam)

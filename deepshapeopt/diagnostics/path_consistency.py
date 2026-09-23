"""Path consistency: does the gradient predict the realized objective change along the step?

For every committed step ``k -> k+1`` the first-order prediction ``g_k . dx`` and the trapezoid
``0.5 (g_k + g_k+1) . dx`` (exact for a quadratic) are compared with the realized change. Both
ratios near 1: the gradient predicts J. Forward ratio small or negative while the trapezoid
ratio is near 1: overshoot (step too long). Both near 0: the gradient is wrong (error floor).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class PathConsistency:
    def __init__(self):
        self.prev = None   # (dx, J_total, g.dx, G0, dG0.dx)
        self.scale = None  # |J(x_0)|

    def observe(self, dJ_vec: np.ndarray, J_total: float, G: np.ndarray, dG: np.ndarray) -> dict:
        """Values of the step that led to the current design (all None before the first step)."""
        if self.scale is None:
            self.scale = abs(float(J_total)) if np.isfinite(J_total) and J_total != 0.0 else 1.0
        out = {k: None for k in ("path_pred", "path_real", "path_rho_fwd", "path_rho_trap",
                                 "path_rho_fwd_con", "path_rho_trap_con")}
        if self.prev is None:
            return out
        dx, J_prev, gdx, G_prev, dgdx = self.prev
        gdx_end = (dJ_vec.reshape(1, -1) @ dx).item()
        real = float(J_total) - J_prev
        trap = 0.5 * (gdx + gdx_end)
        out["path_pred"] = gdx / self.scale
        out["path_real"] = real / self.scale
        out["path_rho_fwd"] = real / gdx if gdx != 0.0 else float("nan")
        out["path_rho_trap"] = real / trap if trap != 0.0 else float("nan")
        msg = (f"path consistency: predicted {out['path_pred']:+.3e} (end of step {gdx_end / self.scale:+.3e}) "
               f"realized {out['path_real']:+.3e} of |J0| -> rho_fwd {out['path_rho_fwd']:+.3f} "
               f"rho_trap {out['path_rho_trap']:+.3f}")
        if G_prev is not None and len(G) > 0:
            dgdx_end = (dG[0].reshape(1, -1) @ dx).item()
            real_con = float(G[0]) - G_prev
            trap_con = 0.5 * (dgdx + dgdx_end)
            out["path_rho_fwd_con"] = real_con / dgdx if dgdx != 0.0 else float("nan")
            out["path_rho_trap_con"] = real_con / trap_con if trap_con != 0.0 else float("nan")
            msg += f" | constraint rho_fwd {out['path_rho_fwd_con']:+.3f} rho_trap {out['path_rho_trap_con']:+.3f}"
        logger.info(msg)
        return out

    def record_step(self, dx: np.ndarray, J_total: float, dJ_vec: np.ndarray, G: np.ndarray, dG: np.ndarray) -> None:
        gdx = (dJ_vec.reshape(1, -1) @ dx).item()
        if len(G) > 0:
            self.prev = (dx, float(J_total), gdx, float(G[0]), (dG[0].reshape(1, -1) @ dx).item())
        else:
            self.prev = (dx, float(J_total), gdx, None, None)

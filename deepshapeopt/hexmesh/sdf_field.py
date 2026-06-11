"""Differentiable SDF evaluation in physical coordinates.

Wraps the normalized-space SDF (``LatticeSDFStruct`` or any callable with the
same convention) so the hex mesh pipeline can query it at physical points,
with an extension outside the design domain and autograd access to the
spatial gradient.

Sign convention (verified at init): ``phi > 0`` in the fluid (outside the
solid), ``phi < 0`` inside the solid.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import torch

logger = logging.getLogger(__name__)


class PhysicalSDF:
    """SDF queryable at physical coordinates.

    Parameters
    ----------
    sdf_norm_fn : callable
        Maps normalized coordinates [N, 3] (float32 tensor) to SDF values
        [N] or [N, 1].  For the optimization pipeline this is the
        ``LatticeSDFStruct`` itself; tests inject synthetic functions.
    norm_fn : callable
        Physical -> normalized coordinate map (e.g. from
        ``fit_box_to_unit_cube``).  Identity for tests.
    design_domain : (2, 3) array-like
        Physical bounding box that contains the zero level set.  Queries
        outside are clamped to it (see :meth:`phi_ext`).
    dist_scale : float
        Factor converting physical distances to SDF value units (the
        normalization scale, ``2 / L``).  Used by the clamped extension.
    device : torch device for SDF evaluation.
    """

    def __init__(
        self,
        sdf_norm_fn: Callable,
        norm_fn: Callable,
        design_domain,
        dist_scale: float = 1.0,
        device: str | torch.device = "cpu",
    ):
        self._fn = sdf_norm_fn
        self._norm_fn = norm_fn
        self.device = torch.device(device)
        self.design_domain = torch.as_tensor(
            np.asarray(design_domain, dtype=np.float32), device=self.device
        )
        self.dist_scale = float(dist_scale)

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def phi(self, x_phys: torch.Tensor) -> torch.Tensor:
        """SDF at physical points [N, 3] -> [N].  Differentiable."""
        x_norm = self._norm_fn(x_phys)
        out = self._fn(x_norm)
        return out.reshape(-1)

    def phi_ext(self, x_phys: torch.Tensor) -> torch.Tensor:
        """SDF extended outside the design domain.

        Queries are clamped to the design domain; the distance to the clamp
        point (converted to SDF units) is added so values grow outward.
        This keeps the zero level set strictly inside the design domain and
        marks the entire margin band as fluid.
        """
        lo = self.design_domain[0][None, :]
        hi = self.design_domain[1][None, :]
        x_cl = torch.clamp(x_phys, min=lo, max=hi)
        dist = torch.linalg.norm(x_phys - x_cl, dim=1)
        return self.phi(x_cl) + self.dist_scale * dist

    def phi_and_grad(self, x_phys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """SDF value and spatial gradient at physical points (no param graph)."""
        with torch.enable_grad():
            x = x_phys.detach().requires_grad_(True)
            f = self.phi(x)
            (g,) = torch.autograd.grad(f.sum(), x, create_graph=False)
        return f.detach(), g.detach()

    # ------------------------------------------------------------------
    # Bulk numpy evaluation (castellation, no gradients)
    # ------------------------------------------------------------------

    def phi_ext_np(self, points: np.ndarray, chunk: int = 65536) -> np.ndarray:
        """Extended SDF at numpy points [N, 3] -> float64 [N], chunked, no grad."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        out = np.empty(len(points), dtype=np.float64)
        with torch.no_grad():
            for start in range(0, len(points), chunk):
                x = torch.as_tensor(
                    points[start : start + chunk], dtype=torch.float32, device=self.device
                )
                out[start : start + chunk] = (
                    self.phi_ext(x).detach().cpu().numpy().astype(np.float64)
                )
        return out

    # ------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------

    def check_sign_convention(self, probe_point) -> None:
        """Assert that the SDF is negative (solid) at a known interior point."""
        x = torch.as_tensor(
            np.asarray(probe_point, dtype=np.float32).reshape(1, 3), device=self.device
        )
        with torch.no_grad():
            val = float(self.phi(x).item())
        if not val < 0.0:
            raise RuntimeError(
                f"SDF sign convention check failed: phi({probe_point}) = {val:.4e} "
                "expected < 0 (solid). The hex mesh pipeline assumes phi > 0 in "
                "the fluid. Set sdf_hex.sign_probe_point to a point inside the "
                "solid, or disable the check with sdf_hex.check_sign: false."
            )
        logger.debug("SDF sign convention OK: phi(%s) = %.4e < 0", probe_point, val)

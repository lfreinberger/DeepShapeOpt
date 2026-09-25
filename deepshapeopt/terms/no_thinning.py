"""Protected regions that may gain but never lose solid (no-thinning constraint).

Inside a user-defined region (a :class:`ProjectedMask` and/or boxes in mm) every grid
point that is solid in the start design must stay solid: the level set may move outward
(thicken, lengthen) but not inward. With the solid-positive SDF ``phi`` and the start
field ``phi_0`` the pointwise condition is

    phi(x) >= min(phi_0(x), cap) - tolerance      for x in region, phi_0(x) > 0,

where the ``cap`` keeps deep-interior points from pinning the (inexact) DeepSDF interior
values. The margin ``s = (min(phi_0, cap) - phi) / tolerance`` (``s = 1``: the surface
retreated by ``tolerance`` mm) is aggregated with the plain, unnormalized KS
``M = log(sum exp(rho s)) / rho >= max s``, so one thinning feature is never averaged
away; the row is ``M <= 1`` and starts at or below ``log(N) / rho`` (capped points count
negative). The candidate set and the reference are fixed on the first evaluation (the
start design of the run), so the gradient is exact.
"""
from __future__ import annotations

import logging
import math

import torch
from DeepSDFStruct.utils import with_float32_lattice

from deepshapeopt.geometry.projected_mask import ProjectedMask

from .base import Budget, ConstraintTerm, State, TermValue, box_grid, exclude_boxes, known_keys
from .ks import KSStream

logger = logging.getLogger(__name__)

_KEYS = {"type", "budget", "mask", "boxes", "grid_spacing", "tolerance", "cap", "ks_rho"}
CHUNK_GRID = 262144
CHUNK_CAND = 65536


class NoThinningConstraint(ConstraintTerm):
    cheap = True
    unscaled = True

    def __init__(self, cfg: dict):
        known_keys(cfg, _KEYS, "no_thinning")
        super().__init__(Budget(cfg.get("budget"), "ks_bound", name="no_thinning"))
        if self.budget.mode != "ks_bound":
            raise ValueError("no_thinning: budget.mode must be 'ks_bound' (the row is M <= 1)")
        self.name = "no_thinning"
        self.mask = ProjectedMask.load(cfg["mask"]) if cfg.get("mask") else None
        self.boxes = exclude_boxes(cfg.get("boxes")) or []
        if self.mask is None and not self.boxes:
            raise ValueError("no_thinning needs a 'mask' file and/or 'boxes'")
        self.grid_spacing = float(cfg.get("grid_spacing", 0.25))
        self.tolerance = float(cfg.get("tolerance", 0.1))
        self.cap = float(cfg.get("cap", 1.0))
        self.ks_rho = float(cfg.get("ks_rho", 50.0))
        if min(self.grid_spacing, self.tolerance, self.cap, self.ks_rho) <= 0.0:
            raise ValueError("no_thinning: grid_spacing, tolerance, cap and ks_rho must be > 0")
        self._points: torch.Tensor | None = None   # normalized candidate points (N, 3)
        self._ref: torch.Tensor | None = None      # min(phi_0, cap), normalized units (N,)

    def ks_bound(self) -> float:
        return 1.0

    def in_region(self, points_mm: torch.Tensor) -> torch.Tensor:
        inside = torch.zeros(points_mm.shape[0], dtype=torch.bool, device=points_mm.device)
        if self.mask is not None:
            inside |= self.mask.contains(points_mm)
        for lo, hi in self.boxes:
            lo_t = torch.as_tensor(lo, dtype=points_mm.dtype, device=points_mm.device)
            hi_t = torch.as_tensor(hi, dtype=points_mm.dtype, device=points_mm.device)
            inside |= ((points_mm >= torch.minimum(lo_t, hi_t)) & (points_mm <= torch.maximum(lo_t, hi_t))).all(dim=1)
        return inside

    def _setup(self, lattice, frame, device) -> None:
        scale = float(frame.scale)
        box = frame.box_norm.to(device=device, dtype=torch.float32)
        sp = scale * self.grid_spacing
        grid = box_grid(box[0], box[1], sp, 0.5 * sp)
        center = frame.center.to(device=device, dtype=torch.float32)

        def _compute(_bounds):
            pts, ref = [], []
            with torch.no_grad():
                for i in range(0, grid.shape[0], CHUNK_GRID):
                    g = grid[i:i + CHUNK_GRID]
                    g = g[self.in_region(g / scale + center)]
                    if g.shape[0] == 0:
                        continue
                    phi = lattice(g).reshape(-1)
                    solid = phi > 0.0
                    pts.append(g[solid])
                    ref.append(torch.clamp(phi[solid], max=scale * self.cap))
            if not pts:
                return grid[:0], grid.new_zeros(0)
            return torch.cat(pts), torch.cat(ref)

        self._points, self._ref = with_float32_lattice(lattice, frame.box_norm, _compute)
        if self._points.shape[0] == 0:
            logger.warning("no_thinning: the region holds no solid grid point of the start design")
        else:
            logger.info("no_thinning: %d protected solid points (M_0 <= ln(N)/rho = %.3f)",
                        self._points.shape[0], math.log(self._points.shape[0]) / self.ks_rho)

    def evaluate(self, state: State) -> TermValue:
        lattice = getattr(state.parametrization, "lattice_struct", None)
        if lattice is None:
            raise ValueError("no_thinning needs the DeepSDF lattice parametrization")
        frame = state.parametrization.frame
        param = state.param
        if self._points is None:
            self._setup(lattice, frame, param.device)
        n = int(self._points.shape[0])
        if n == 0:
            return TermValue(value=-1.0, grad=torch.zeros_like(param), debug={"candidates": 0})

        scale = float(frame.scale)
        inv = 1.0 / (scale * self.tolerance)
        rho = self.ks_rho

        def _compute(_bounds):
            ks = KSStream(param)
            retreat = []
            for i in range(0, n, CHUNK_CAND):
                X = self._points[i:i + CHUNK_CAND]
                r = self._ref[i:i + CHUNK_CAND]
                phi = lattice(X).reshape(-1)
                ks.add(rho * (r - phi) * inv)
                retreat.append(((r - phi.detach()) / scale).detach())
            logZ, dlogZ = ks.finalize()
            return logZ, dlogZ, torch.cat(retreat)

        logZ, dlogZ, retreat = with_float32_lattice(lattice, frame.box_norm, _compute)
        value = logZ / rho
        grad = (dlogZ / rho).to(param.dtype).detach()
        pts_mm = (self._points / scale + frame.center.to(self._points)).cpu().numpy()
        retreat_np = retreat.cpu().numpy()
        debug = {
            "candidates": n,
            "reading": f"max retreat {float(retreat_np.max()):+.3f} mm",
            "cloud": (pts_mm, 0.0 * pts_mm,
                      {"retreat_mm": retreat_np, "ref_mm": (self._ref / scale).cpu().numpy()}),
        }
        return TermValue(value=float(value), grad=grad, debug=debug)


"""Fold-over guard of the FFD: KS lower bound of det(dT/dx) over the design domain."""

from __future__ import annotations

import torch

from deepshapeopt.parametrization.ffd import min_jacobian_det_ks

from .base import Budget, ConstraintTerm, State, TermValue, known_keys


class FfdJacobianTerm(ConstraintTerm):
    """``threshold - KS_min(det J) <= 0``, scaled by the threshold."""

    def __init__(self, cfg: dict):
        known_keys(cfg, {"type", "threshold", "samples", "ks_rho"}, "ffd_jacobian")
        self.threshold = float(cfg.get("threshold", 0.1))
        super().__init__(Budget({"mode": "absolute", "value": 0.0}, name="ffd_jacobian"))
        self.name = "ffd_jacobian"
        self.samples = int(cfg.get("samples", 6))
        self.ks_rho = float(cfg.get("ks_rho", 50.0))

    def evaluate(self, state: State) -> TermValue:
        deformation = state.parametrization.deformation
        ks, det_min = min_jacobian_det_ks(deformation, n_samples_per_dim=self.samples, ks_rho=self.ks_rho)
        g = self.threshold - ks
        (grad,) = torch.autograd.grad(g, state.param)
        return TermValue(value=float(g.detach()), grad=grad.detach().to(state.param.dtype),
                         debug={"jacobian_ks": float(ks.detach()), "jacobian_det_min": det_min})

    def row(self, tv: TermValue):
        row = super().row(tv)
        row.scale = self.threshold
        return row

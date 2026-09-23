"""Volume and centroid of the meshed body (drag optimization constraints)."""

from __future__ import annotations

import torch

from .base import Budget, ConstraintTerm, State, TermValue, known_keys


class VolumeTerm(ConstraintTerm):
    """``V_0 - V <= 0``: the body may not lose volume. Raw row (legacy scaling)."""

    unscaled = True

    def __init__(self, cfg: dict):
        known_keys(cfg, {"type"}, "volume")
        super().__init__(Budget({"mode": "absolute", "value": 0.0}, name="volume"))
        self.name = "volume"
        self.initial_volume: float | None = None
        self.volume: float | None = None

    def set_initial(self, volume: float, centroid) -> None:
        self.initial_volume = float(volume)

    def evaluate(self, state: State) -> TermValue:
        volume, _ = state.mesher.volume_centroid()
        if self.initial_volume is None:
            self.initial_volume = float(volume.item())
        constraint = self.initial_volume - volume
        (grad,) = torch.autograd.grad(constraint, state.param, retain_graph=True)
        self.volume = float(volume.item())
        return TermValue(value=float(constraint.item()), grad=grad.detach(), debug={"volume": self.volume})


class CentroidTerm(ConstraintTerm):
    """``|c - c_0|^2 - tol^2 <= 0``: the centroid stays within ``tol`` of the start. Raw row."""

    unscaled = True

    def __init__(self, cfg: dict):
        known_keys(cfg, {"type", "tol"}, "centroid")
        super().__init__(Budget({"mode": "absolute", "value": 0.0}, name="centroid"))
        self.name = "centroid"
        self.tol = float(cfg.get("tol", 0.0))
        self.initial_centroid = None

    def set_initial(self, volume, centroid) -> None:
        self.initial_centroid = centroid.detach().clone()

    def evaluate(self, state: State) -> TermValue:
        _, centroid = state.mesher.volume_centroid()
        if self.initial_centroid is None:
            self.initial_centroid = centroid.detach().clone()
        constraint = ((centroid - self.initial_centroid) ** 2).sum() - self.tol ** 2
        (grad,) = torch.autograd.grad(constraint, state.param, retain_graph=True)
        return TermValue(value=float(constraint.item()), grad=grad.detach())

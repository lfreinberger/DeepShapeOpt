"""CFD metrics from the forward solver as objective or constraint."""

from __future__ import annotations

import torch

from deepshapeopt.gradient import shape_gradient, vertex_normals
from deepshapeopt.solvers.metrics import Metric

from .base import Budget, ConstraintTerm, State, Term, TermValue


def _evaluate_metric(metric: Metric, state: State) -> TermValue:
    sens = state.solver.sensitivities[metric.name]
    verts_solver = state.mesh.verts * state.unit_to_metre
    grad = shape_gradient(state.param, verts_solver, sens)
    debug = {}
    if state.debug:
        normals = vertex_normals(state.mesh.verts, state.mesh.faces, invert_normals=False)
        s = torch.as_tensor(sens, dtype=normals.dtype, device=normals.device)
        debug["sens_normal"] = (s * normals).sum(dim=1).detach().cpu().numpy()
        debug["sens_norm"] = float(s.norm())
    return TermValue(value=float(state.solver.values[metric.name]), grad=grad, debug=debug)


class CfdObjective(Term):
    def __init__(self, metric: Metric):
        self.metric = metric
        self.name = metric.name

    def evaluate(self, state: State) -> TermValue:
        return _evaluate_metric(self.metric, state)


class CfdConstraint(ConstraintTerm):
    def __init__(self, metric: Metric, budget_cfg: dict | None):
        super().__init__(Budget(budget_cfg, "relative_to_initial", name=metric.name))
        if self.budget.mode not in ("relative_to_initial", "absolute"):
            raise ValueError(f"{metric.name}: a CFD constraint budget is relative_to_initial or absolute")
        self.metric = metric
        self.name = metric.name

    def evaluate(self, state: State) -> TermValue:
        return _evaluate_metric(self.metric, state)

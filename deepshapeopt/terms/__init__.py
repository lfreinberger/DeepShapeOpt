"""Objective and constraint terms of the optimization problem."""

from __future__ import annotations

from deepshapeopt.config.schema import Config
from deepshapeopt.solvers.metrics import get_metric

from .base import Budget, ConstraintTerm, PenaltyTerm, Row, State, Term, TermValue
from .cfd import CfdConstraint, CfdObjective
from .ffd_jacobian import FfdJacobianTerm
from .regularizers import LatticeSmoothnessPenalty, ProximityPenalty
from .steg_length import MinStegLengthConstraint, MinStegLengthPenalty
from .undercut import UndercutConstraint, UndercutPenalty
from .volume import CentroidTerm, VolumeTerm

CONSTRAINT_TYPES = {
    "metric": None,  # handled separately (needs the metric registry)
    "volume": VolumeTerm,
    "centroid": CentroidTerm,
    "ffd_jacobian": FfdJacobianTerm,
    "undercut": UndercutConstraint,
    "min_steg_length": MinStegLengthConstraint,
}
PENALTY_TYPES = {
    "proximity": ProximityPenalty,
    "lattice_smoothness": LatticeSmoothnessPenalty,
    "undercut": UndercutPenalty,
    "min_steg_length": MinStegLengthPenalty,
}


def build_objective(cfg: Config) -> CfdObjective:
    return CfdObjective(get_metric(cfg.objective.metric))


def build_penalties(cfg: Config) -> list[PenaltyTerm]:
    out = []
    for entry in cfg.objective.penalties:
        kind = entry["type"]
        if kind not in PENALTY_TYPES:
            raise ValueError(f"objective.penalties: unknown type {kind!r}; valid {sorted(PENALTY_TYPES)}")
        out.append(PENALTY_TYPES[kind](entry))
    return out


def build_constraints(cfg: Config) -> list[ConstraintTerm]:
    out = []
    for entry in cfg.constraints:
        kind = entry["type"]
        if kind == "metric":
            metric = get_metric(entry["metric"])
            if metric.name == cfg.objective.metric:
                raise ValueError("objective and constraint must be different metrics")
            out.append(CfdConstraint(metric, entry.get("budget")))
        elif kind in CONSTRAINT_TYPES:
            out.append(CONSTRAINT_TYPES[kind](entry))
        else:
            raise ValueError(f"constraints: unknown type {kind!r}; valid {sorted(CONSTRAINT_TYPES)}")
    return out


__all__ = [
    "Budget", "ConstraintTerm", "PenaltyTerm", "Row", "State", "Term", "TermValue",
    "build_objective", "build_penalties", "build_constraints",
]

"""Terms of the optimization problem: objective, penalties and constraint rows.

A term turns the current state (design parameters, mesh, solver result) into a value and
its gradient with respect to the design parameters. Constraint terms carry a
:class:`Budget` that fixes their target on the first evaluation and turn into MMA rows
``value - target <= 0``; ``cheap`` terms need no CFD and can be re-evaluated by the
GCMMA inner loop and the feasibility restoration.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass
class State:
    """Everything a term may read during one evaluation."""

    iteration: int
    param: torch.Tensor
    parametrization: Any
    design_space: Any
    mesher: Any
    unit_to_metre: float
    debug: bool
    results_dir: Path
    heavy_dir: Path | None = None
    mesh: Any = None            # MeshResult of this iteration
    solver: Any = None          # SolverResult of this iteration
    objective_scale: float | None = None  # |J(x_0)| for dimensionless penalty weights


@dataclass
class TermValue:
    value: float
    grad: torch.Tensor | None          # d value / d param, param-shaped, detached
    debug: dict = field(default_factory=dict)


@dataclass
class Row:
    """One MMA constraint row: ``value <= 0`` is feasible."""

    name: str
    value: float
    grad: torch.Tensor
    scale: float | None      # divisor applied by MMA (the target); None = already dimensionless
    cheap: bool
    raw_value: float         # the term's value before subtracting the target
    target: float


class Budget:
    """Target of a constraint term.

    ``relative_to_initial``: target = initial value * factor; ``absolute``: target = value;
    ``ks_bound``: the bound of a signed KS margin (set by the term, e.g. ``-sin(draft)``;
    ``shortfall`` for the steg length); ``absolute_shortfall``: the squared relative shortfall
    of the steg-length penalty form.
    """

    MODES = ("relative_to_initial", "absolute", "ks_bound", "absolute_shortfall")

    def __init__(self, cfg: dict | None, default_mode: str = "relative_to_initial", name: str = "constraint"):
        cfg = dict(cfg or {})
        self.mode = str(cfg.get("mode", default_mode))
        if self.mode not in self.MODES:
            raise ValueError(f"{name}: budget.mode must be one of {self.MODES}, got {self.mode!r}")
        self.factor = float(cfg.get("factor", 1.0))
        self.value = cfg.get("value")
        self.shortfall = cfg.get("shortfall")
        self.name = name
        if self.mode == "absolute" and self.value is None:
            raise ValueError(f"{name}: budget.mode 'absolute' needs budget.value")

    def target(self, initial: float, *, bound: float | None = None, length_scale: float | None = None) -> float:
        if self.mode == "relative_to_initial":
            return float(initial) * self.factor
        if self.mode == "absolute":
            return float(self.value)
        if self.mode == "ks_bound":
            if bound is None:
                raise ValueError(f"{self.name}: budget.mode 'ks_bound' needs a KS formulation")
            return float(bound)
        if self.shortfall is None or not length_scale:
            raise ValueError(f"{self.name}: budget.mode 'absolute_shortfall' needs budget.shortfall and a length")
        return (float(self.shortfall) / float(length_scale)) ** 2


class Term(ABC):
    name: str = "term"
    cheap: bool = False   # evaluable without the solver (SDF-grid terms)

    @abstractmethod
    def evaluate(self, state: State) -> TermValue: ...


class ConstraintTerm(Term):
    """A term with a budget that becomes one MMA row."""

    budget: Budget
    unscaled: bool = False   # KS margins are O(1) and enter MMA without a scale

    def __init__(self, budget: Budget):
        self.budget = budget
        self.target: float | None = None
        self.initial: float | None = None

    def ks_bound(self) -> float | None:
        return None

    def length_scale(self) -> float | None:
        return None

    def resolve_target(self, value: float) -> float:
        if self.target is None:
            self.initial = float(value)
            self.target = self.budget.target(value, bound=self.ks_bound(), length_scale=self.length_scale())
            logger.info("%s: initial %.6e, target %.6e", self.name, self.initial, self.target)
        return self.target

    def row(self, tv: TermValue) -> Row:
        target = self.resolve_target(tv.value)
        return Row(name=self.name, value=float(tv.value) - target, grad=tv.grad,
                   scale=None if self.unscaled else target, cheap=self.cheap,
                   raw_value=float(tv.value), target=target)


class PenaltyTerm(Term):
    """An objective-mode term ``weight * |J(x_0)| * value`` (dimensionless weight)."""

    def __init__(self, weight: float):
        self.weight = float(weight)

    def contribution(self, tv: TermValue, objective_scale: float) -> tuple[float, torch.Tensor]:
        w = self.weight * float(objective_scale)
        return w * float(tv.value), w * tv.grad


def exclude_boxes(cfg_value):
    """Normalize ``exclude_region`` (one box or a list of boxes) to a list of boxes or None."""
    if cfg_value is None:
        return None
    return cfg_value if hasattr(cfg_value[0][0], "__len__") else [cfg_value]


def known_keys(cfg: dict, allowed: set[str], name: str) -> None:
    unknown = sorted(k for k in cfg if not k.startswith("_") and k not in allowed)
    if unknown:
        raise ValueError(f"{name}: unknown keys {unknown}; allowed {sorted(allowed)}")

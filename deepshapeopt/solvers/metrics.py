"""Registry of the CFD metrics an objective or constraint can refer to.

A metric names the OpenFOAM adjoint solver and objective entry that compute it, and the
DAFoam output that plays the same role with the discrete adjoint.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Metric:
    name: str
    of_solver: str          # adjoint solver entry in optimisationDict (as1, as2, adjS1)
    of_objective: str       # objective entry of that solver (its file is <objective><solver>)
    dafoam_output: str | None  # name in solver.dafoam.outputs, None when DAFoam has no counterpart

    def field_name(self, suffix: str = "ESI") -> str:
        """Integrated point-sensitivity field written by the adjoint (surfacePoints)."""
        return f"pointSensVec{self.of_solver}{suffix}"

    @property
    def objective_file(self) -> str:
        return f"optimisation/objective/0/{self.of_objective}{self.of_solver}"


METRICS = {
    "drag": Metric("drag", "adjS1", "drag", "drag"),
    "uniformity": Metric("uniformity", "as1", "uniformity", "uniformity"),
    # DAFoam has no directional uniformity; its plain variance stands in (logged when used).
    "uniformity_directional": Metric("uniformity_directional", "as1", "uniformityDirectional", "uniformity"),
    "losses": Metric("losses", "as2", "losses", "losses"),
}


def get_metric(name: str) -> Metric:
    try:
        return METRICS[name]
    except KeyError:
        raise ValueError(f"Unknown metric {name!r}; valid: {sorted(METRICS)}") from None

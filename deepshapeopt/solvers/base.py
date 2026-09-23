"""What the optimization loop needs from a forward solver."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from .metrics import Metric


@dataclass
class SolverResult:
    values: dict[str, float]                 # metric name -> J
    sensitivities: dict[str, np.ndarray]     # metric name -> dJ/dx of the wall points [P, 3]
    case_dir: Path
    time_s: float
    foam_case: Any = None                    # foamlib case (OpenFOAM) for field exports
    adjoint_times: dict[str, str] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)


class ForwardSolver(Protocol):
    """Runs primal + adjoint on the current mesh and returns values and wall sensitivities.

    The sensitivities are the integrated ``dJ/dx`` of every wall point in the solver's length
    unit (metres); :func:`deepshapeopt.gradient.shape_gradient` pulls them back to the design
    parameters.
    """

    case_dir: Path

    def evaluate(self, mesh, metrics: Sequence[Metric]) -> SolverResult: ...

    def export_fields(self, result: SolverResult, iteration: int, out_dir: Path) -> None: ...

    def clean_iteration(self, result: SolverResult) -> None: ...

    def close(self) -> None: ...

"""Metric -> DAFoam output mapping.

DAFoam evaluates ``functions`` (force, variance, totalPressure, ...) and ``outputs`` (linear
combinations of functions, one adjoint solve each); both are written in the experiment
config. A metric is served by the output of the same name given in the registry.
"""

from __future__ import annotations

import logging
from typing import Sequence

from ..metrics import Metric

logger = logging.getLogger(__name__)


def resolve_outputs(metrics: Sequence[Metric], dafoam_cfg: dict) -> dict[str, str]:
    """``{metric name: DAFoam output name}``; raises for metrics DAFoam cannot provide."""
    outputs = dafoam_cfg.get("outputs") or {}
    mapping = {}
    for metric in metrics:
        name = metric.dafoam_output
        if name is None:
            raise ValueError(f"metric {metric.name!r} has no DAFoam counterpart")
        if name not in outputs:
            raise ValueError(f"metric {metric.name!r} needs solver.dafoam.outputs[{name!r}]; have {sorted(outputs)}")
        if name != metric.name:
            logger.warning("DAFoam serves metric %r with output %r (no directional variant exists)", metric.name, name)
        mapping[metric.name] = name
    return mapping

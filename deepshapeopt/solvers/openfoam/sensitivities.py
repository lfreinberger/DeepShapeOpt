"""Read objective values and point sensitivities back from an adjoint run."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from foamlib import FoamFieldFile

logger = logging.getLogger(__name__)


def resolve_adjoint_time(case_dir: Path, field_name: str, expected_time: str | None) -> str:
    """Name of the (reconstructed) time directory that actually holds ``field_name``.

    ``expected_time`` is the prediction from :func:`configure_foam_runtime`. It is wrong
    whenever an adjoint solver stops before its ``nIters`` (``residualControl``): every
    later solver then writes at an earlier time. If the field is not at the predicted
    time, the latest time directory containing it is used and a warning names the shift.
    """
    case_dir = Path(case_dir)
    if expected_time is not None and (case_dir / str(expected_time) / field_name).is_file():
        return str(expected_time)

    candidates = []
    for d in case_dir.iterdir():
        if not d.is_dir() or not (d / field_name).is_file():
            continue
        try:
            candidates.append((float(d.name), d.name))
        except ValueError:
            continue
    if not candidates:
        # Distinguish "the solver never wrote it" from "it was written but not
        # reconstructed" -- the second is a reconstructPar time-range problem and
        # the message should say so instead of sending the reader to the solver log.
        decomposed = sorted(
            d.name for d in (case_dir / "processor0").iterdir()
            if d.is_dir() and (d / field_name).is_file()
        ) if (case_dir / "processor0").is_dir() else []
        if decomposed:
            raise FileNotFoundError(
                f"{field_name} exists in processor0 at time(s) {', '.join(decomposed)} "
                f"but was not reconstructed into {case_dir} (predicted time: "
                f"{expected_time}). reconstructPar's -time range missed it."
            )
        raise FileNotFoundError(
            f"{field_name} not found in any time directory of {case_dir} "
            f"(predicted time: {expected_time}). Check log.adjointOptimisationFoam: "
            "was the solver active, and did reconstructPar cover its final time?"
        )
    _, found = max(candidates)
    if expected_time is not None:
        logger.warning(
            "Adjoint field %s is not at the predicted time %s; using %s "
            "(an adjoint solver stopped before nIters, e.g. via residualControl).",
            field_name, expected_time, found,
        )
    return found

def read_objective(case_path: Path, objective_path, with_start_time: bool = False):
    """Objective value from ``optimisation/objective/0/<name>``.

    Column 0 is the time at which the row was written, which is the moment the
    objective manager updated the value -- i.e. the START time of that adjoint
    solver, equal to the END of the solver before it (verified: the as1 file
    starts at the primal's end, the as2 file at as1's end). With
    ``with_start_time`` it is returned alongside J, which makes it a free
    detector for a primal that stopped early.
    """
    path = case_path / objective_path
    data = np.atleast_2d(np.loadtxt(path, comments="#"))[-1]
    J_last = data[1]
    return (J_last, int(round(float(data[0])))) if with_start_time else J_last


def read_point_sensitivities(
    case_dir: Path, field_name: str, time_value: str, point_ids: np.ndarray, n_points: int,
) -> np.ndarray:
    """Point sensitivity vectors of the wall points, read by index from the written mesh."""
    time_value = resolve_adjoint_time(Path(case_dir), field_name, time_value)
    field_path = Path(case_dir) / time_value / field_name
    sens_all = np.asarray(FoamFieldFile(field_path).internal_field, dtype=np.float64)
    if sens_all.shape[0] != n_points:
        raise ValueError(
            f"{field_name} has {sens_all.shape[0]} points but the written mesh has {n_points}"
        )
    sens = sens_all[np.asarray(point_ids)]
    logger.debug("Loaded %d wall point sensitivities (|s| max %.3e)", len(sens), float(np.abs(sens).max()))
    return sens

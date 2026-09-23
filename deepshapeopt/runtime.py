"""Small runtime helpers for optimization scripts."""

from __future__ import annotations

import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np


def is_debug_enabled(specs: dict[str, Any]) -> bool:
    opt_cfg = specs.get("optimization", {})
    return bool(specs.get("debug", opt_cfg.get("debug", False)))


def configure_logging(debug: bool, log_file: Path | None = None) -> None:
    """Install a clean stdout (+ optional file) handler on the root logger.

    Third-party libraries (notably DeepSDFStruct) attach their own handlers
    to named loggers with a timestamped formatter; without intervention
    those messages get emitted twice — once by their handler and once via
    propagation through ours. Here we detach existing handlers, install
    ours with a bare ``%(message)s`` format, strip the named loggers' own
    handlers and let them propagate through ours (so e.g. the MMA / GCMMA
    back-off lines reach run.log, not just the console). In non-debug mode
    we also raise their level to ``WARNING`` and silence
    ``DeepSDFStruct``-origin ``UserWarning``s (PyTorch tensor noise).
    """
    level = logging.DEBUG if debug else logging.INFO

    formatter = logging.Formatter("%(message)s")
    handlers: list[logging.Handler] = []

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    handlers.append(stream_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()
    for h in handlers:
        root.addHandler(h)
    root.setLevel(level)

    # Matplotlib's font_manager logs a findfont score line per installed
    # font at DEBUG; keep third-party debug chatter out even in debug mode.
    for noisy in ("matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    for name, lg in list(logging.Logger.manager.loggerDict.items()):
        if not isinstance(lg, logging.Logger) or not lg.handlers:
            continue
        # Strip their own handlers and route them through ours instead: single
        # emission, and their messages (e.g. the MMA iteration / GCMMA back-off
        # lines from DeepSDFStruct) land in run.log too, not just on the console.
        for h in list(lg.handlers):
            lg.removeHandler(h)
        lg.propagate = True
        if not debug:
            lg.setLevel(logging.WARNING)

    # Third-party libraries emit huge volumes of DEBUG/INFO records that
    # drown out our own output. Pin them to WARNING regardless of mode so
    # turning on debug for deepshapeopt code doesn't unleash matplotlib
    # font-scoring, gustaf getter/setter traces, etc.
    for noisy in (
        "matplotlib",
        "PIL",
        "fontTools",
        "gustaf",
        "trimesh",
        "h5py",
        "asyncio",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if not debug:
        warnings.filterwarnings("ignore", category=UserWarning, module=r"DeepSDFStruct\..*")
        warnings.filterwarnings("ignore", category=UserWarning, module=r"torch\..*")


def log_iteration_summary(logger: logging.Logger, **values: Any) -> None:
    fields = []
    for key, value in values.items():
        if value is None:
            continue
        if isinstance(value, float):
            fields.append(f"{key}={value:.6e}")
        else:
            fields.append(f"{key}={value}")
    logger.info("  " + " | ".join(fields))


def log_timing(
    logger: logging.Logger,
    iter_start: float,
    run_start: float,
    iteration_times: list[float],
    total_iters: int,
    current_iter: int,
) -> None:
    iter_time = time.time() - iter_start
    iteration_times.append(iter_time)
    avg_time = sum(iteration_times) / len(iteration_times)
    remaining = max(0, total_iters - current_iter - 1)
    eta = avg_time * remaining
    elapsed = time.time() - run_start
    logger.info(
        "  time=%.2fs | avg=%.2fs | elapsed=%.2fmin | eta=%.2fmin",
        iter_time,
        avg_time,
        elapsed / 60,
        eta / 60,
    )


def is_feasible(rows, scales, tol: float) -> bool:
    """All raw constraint rows ``value - target`` within ``tol`` of their scale (the target)."""
    for row, scale in zip(rows, scales):
        if float(row) > tol * max(abs(float(scale)), 1e-300):
            return False
    return True


def progress_gain(objectives, window: int, feasible=None) -> float:
    """Relative decrease of the best feasible objective over the last ``window`` evaluations.

    ``(best[k - window] - best[k]) / |best[k]|`` with ``best`` the running minimum over the
    feasible evaluations; NaN while fewer than ``window + 1`` evaluations exist or no feasible
    one is in range. The window should span one MMA overshoot cycle (10-15 iterations on the
    cylinder, die and feed channel), and the best-so-far makes both endpoints insensitive to an
    overshoot landing there. Noise and re-castellation jumps only DELAY the stop (a spuriously
    low value counts as gain).
    """
    vals = np.asarray(objectives, dtype=float)
    if feasible is not None:
        vals = np.where(np.asarray(feasible, dtype=bool), vals, np.nan)
    if vals.size <= window:
        return float("nan")
    best = np.fmin.accumulate(vals)
    now, then = best[-1], best[-1 - window]
    if not (np.isfinite(now) and np.isfinite(then)) or now == 0.0:
        return float("nan")
    return float((then - now) / abs(now))


def has_converged(gains, tolerance: float | None, patience: int = 1) -> bool:
    """Progress-based stop: the last ``patience`` values of ``progress_gain`` are all below
    ``tolerance`` (``None`` disables the criterion).

    Reading: the run improved by less than ``tolerance`` (relative to the current best) over
    the last window, so the next window is not worth its evaluations. It is a marginal-gain
    criterion, not an optimality test -- the MMA never satisfies one in a flat valley, where it
    keeps walking at the move limit with a constant gradient norm. ``tolerance`` is the one
    problem-dependent number: the relative gain below which two valid meshes of the same
    geometry can no longer be told apart (die at 0.25 mm cells: 3 %). Backtested 2026-09-22 on
    17 runs (cylinder, die, feed channel, drag cube): fires where the remaining gain is below
    1 % on the only run with a converged reference, never fires spuriously in an overshoot
    phase; the earlier single-step form (|dJ|/J0 < 2e-4 for 3 iterations) gave up 12-14 % on
    the die because J0-normalization makes the tolerance shrink with the achieved reduction,
    and never fired at the cylinder's zigzag error floor.
    """
    if tolerance is None or len(gains) < patience:
        return False
    recent = gains[-patience:]
    return all(g == g and g < tolerance for g in recent)

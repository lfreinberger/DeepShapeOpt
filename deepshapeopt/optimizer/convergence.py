"""Progress-based stop: gain of the best feasible objective over one MMA cycle."""
from __future__ import annotations

import numpy as np


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

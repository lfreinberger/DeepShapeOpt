"""Estimating the numerical noise of expensive evaluations.

The CFD pipeline is deterministic but not smooth: re-castellating the hex mesh flips cells in
or out of the fluid for arbitrarily small design changes, so J and its adjoint gradient carry a
"deterministic noise" component. Re-evaluating the same design reproduces it exactly, so the
noise has to be read off evaluations at a few nearby designs instead.

:func:`ecnoise` is a port of ECnoise (Moré & Wild, "Estimating computational noise",
SIAM J. Sci. Comput. 33(3), 2011): evaluate ``f(x + t_i h p)`` on equally spaced points along
one direction, build the difference table and take the first order whose scaled level has
stabilized. For white noise of standard deviation sigma, ``E[(Δ^k e)^2] = sigma^2 (2k)!/(k!)^2``,
so ``sqrt(gamma_k mean((Δ^k f)^2))`` with ``gamma_k = (k!)^2/(2k)!`` estimates sigma once the
smooth part of ``f`` no longer contributes to ``Δ^k f``.

:func:`ecnoise_vector` applies the same table to a vector-valued evaluation (a gradient) and
returns the expected L2 norm of the noise vector, the quantity to compare an L2 stationarity
residual against.

``inform`` codes follow ECnoise: 1 noise detected, 2 not detected because ``h`` is too small
(the evaluations barely change), 3 not detected because ``h`` is too large (the smooth part
dominates every difference order).
"""

from __future__ import annotations

import numpy as np

NOISE_DETECTED = 1
H_TOO_SMALL = 2
H_TOO_LARGE = 3


def _first_stable_level(levels: np.ndarray, sign_change: np.ndarray) -> int | None:
    """First order k whose levels k..k+2 agree within a factor 4 and whose differences change sign."""
    for k in range(len(levels) - 2):
        window = levels[k:k + 3]
        if window.max() <= 4.0 * window.min() and sign_change[k]:
            return k
    return None


def ecnoise(fvals) -> tuple[float, np.ndarray, int]:
    """Noise level of scalar evaluations on equally spaced points along one direction.

    Parameters
    ----------
    fvals : array-like, shape (n,)
        ``f(x + t_i h p)`` in order of ``t_i``; ``n >= 4`` (ECnoise uses about 7 to 9).

    Returns
    -------
    noise : float
        Estimated standard deviation of the noise (0.0 unless ``inform == 1``).
    levels : ndarray, shape (n-1,)
        Scaled level of every difference order, for inspection.
    inform : int
        1 detected, 2 ``h`` too small, 3 ``h`` too large.
    """
    f = np.asarray(fvals, dtype=float).reshape(-1)
    nf = f.size
    if nf < 4:
        raise ValueError(f"ecnoise needs at least 4 evaluations, got {nf}")
    levels = np.zeros(nf - 1)
    sign_change = np.zeros(nf - 1, dtype=bool)

    fmin, fmax = f.min(), f.max()
    scale = max(abs(fmin), abs(fmax))
    if scale > 0.0 and (fmax - fmin) / scale > 0.1:
        return 0.0, levels, H_TOO_LARGE

    d = f.copy()
    gamma = 1.0
    for j in range(1, nf):
        d = np.diff(d)
        if j == 1 and np.count_nonzero(d == 0.0) >= nf / 2:
            return 0.0, levels, H_TOO_SMALL
        gamma *= 0.5 * j / (2 * j - 1)  # gamma_j = (j!)^2 / (2j)!
        levels[j - 1] = np.sqrt(gamma * np.mean(d ** 2))
        sign_change[j - 1] = d.min() * d.max() < 0.0

    k = _first_stable_level(levels, sign_change)
    if k is None:
        return 0.0, levels, H_TOO_LARGE
    return float(levels[k]), levels, NOISE_DETECTED


def ecnoise_vector(G) -> tuple[float, np.ndarray, int]:
    """Expected L2 norm of the noise of vector evaluations along one direction.

    Parameters
    ----------
    G : array-like, shape (n_points, n_vars)
        One vector evaluation (e.g. a gradient) per row, in order of ``t_i``.

    Returns
    -------
    noise_norm : float
        Estimated ``sqrt(E ||e||^2)`` of the noise vector (0.0 unless ``inform == 1``).
    levels : ndarray, shape (n_points-1,)
        Scaled level of every difference order.
    inform : int
        1 detected, 2 ``h`` too small, 3 ``h`` too large.

    The level of order k is ``sqrt(gamma_k mean_i ||Δ^k G_i||^2)``. The sign-change test asks
    that the k-th differences change sign in the majority of the components that vary at all.
    """
    G = np.asarray(G, dtype=float)
    if G.ndim != 2:
        raise ValueError(f"ecnoise_vector expects shape (n_points, n_vars), got {G.shape}")
    nf = G.shape[0]
    if nf < 4:
        raise ValueError(f"ecnoise_vector needs at least 4 evaluations, got {nf}")
    levels = np.zeros(nf - 1)
    sign_change = np.zeros(nf - 1, dtype=bool)

    D = G.copy()
    gamma = 1.0
    varying = None
    for j in range(1, nf):
        D = np.diff(D, axis=0)
        if j == 1:
            varying = np.any(D != 0.0, axis=0)
            unchanged_rows = np.count_nonzero(np.all(D == 0.0, axis=1))
            if not varying.any() or unchanged_rows >= nf / 2:
                return 0.0, levels, H_TOO_SMALL
        gamma *= 0.5 * j / (2 * j - 1)
        levels[j - 1] = np.sqrt(gamma * np.mean(np.sum(D ** 2, axis=1)))
        if D.shape[0] >= 2:
            comp = D[:, varying]
            sign_change[j - 1] = np.mean(comp.min(axis=0) * comp.max(axis=0) < 0.0) > 0.5

    k = _first_stable_level(levels, sign_change)
    if k is None:
        return 0.0, levels, H_TOO_LARGE
    return float(levels[k]), levels, NOISE_DETECTED

"""Streaming Kreisselmeier-Steinhauser (logsumexp) aggregation."""
from __future__ import annotations

import math

import torch


class KSStream:
    """Streaming logsumexp accumulator for chunk-wise KS (smooth-max) aggregation.

    Accumulates ``logZ = log sum_k exp(a_k)`` and (when ``param`` is given)
    ``dlogZ/dparam`` over grad-connected chunks WITHOUT keeping more than one chunk's
    autograd graph alive: each :meth:`add` reduces its chunk to a (sum, grad) pair
    immediately and rescales the running pair when the running max shifts -- exact and
    overflow-safe (every exponent evaluated is <= 0). This is what lets the ks_margin
    formulation of :func:`min_steg_length_penalty_sdf` aggregate over an unbounded
    candidate count at the same peak memory as the chunked penalty form (unlike
    :func:`undercut_penalty_sdf`, whose band fits one logsumexp). Also reused by
    scripts/check_min_steg_fd.py for the frozen-weights FD gate.
    """

    def __init__(self, param=None):
        self.param = param          # None -> value-only (no gradient accumulation)
        self.m = -math.inf          # running max exponent
        self.S = 0.0                # running sum of exp(a - m)
        self.G = None if param is None else torch.zeros_like(param)

    def add(self, a):
        """Fold in a 1D chunk of (grad-connected) exponents; empty chunks are no-ops."""
        if a.numel() == 0:
            return
        m_c = float(a.detach().max())
        if m_c == -math.inf:
            return
        m_new = max(self.m, m_c)
        S_c = torch.exp(a - m_new).sum()
        r = math.exp(self.m - m_new)  # rescale of the running pair; 0.0 on the first add
        if self.param is not None:
            g_c = torch.autograd.grad(S_c, self.param, retain_graph=False, allow_unused=True)[0]
            self.G = self.G * r + (g_c if g_c is not None else 0.0)
        self.S = self.S * r + float(S_c.detach())
        self.m = m_new

    def finalize(self):
        """Return ``(logZ, dlogZ/dparam)``; ``(-inf, zeros)`` when nothing accumulated."""
        if self.S <= 0.0:
            return -math.inf, (None if self.param is None else torch.zeros_like(self.param))
        g = None if self.param is None else self.G / self.S
        return self.m + math.log(self.S), g

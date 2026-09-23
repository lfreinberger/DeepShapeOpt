"""Geometry terms without a solver: KS streaming, undercut margin, steg length.

The undercut gates use a synthetic frustum channel with an analytic drawability margin; the
steg-length gates use an analytic box SDF standing in for the lattice, so both the
"thicken" and the "lengthen" remedies have autograd paths.
"""

import math

import numpy as np
import pytest
import torch

from deepshapeopt.geometry.frame import DomainFrame
from deepshapeopt.terms.ks import KSStream
from deepshapeopt.terms.steg_length import min_steg_length_penalty_sdf
from deepshapeopt.terms.undercut import build_outlet_silhouette, undercut_penalty


# ---------------------------------------------------------------------------
# KSStream
# ---------------------------------------------------------------------------

def test_ksstream_matches_logsumexp_value_and_gradient():
    torch.manual_seed(0)
    p = torch.randn(7, requires_grad=True)
    a_full = torch.cat([p * 3.0, p ** 2 - 10.0, p.abs() * 20.0, p * 0.0 - 500.0])
    ref = torch.logsumexp(a_full, dim=0)
    g_ref = torch.autograd.grad(ref, p, retain_graph=True)[0]

    ks = KSStream(p)
    ks.add(p ** 2 - 10.0)
    ks.add(p.new_zeros(0))
    ks.add(torch.full((3,), -math.inf))
    ks.add(p * 3.0)
    ks.add(p.abs() * 20.0)
    ks.add(p * 0.0 - 500.0)
    logZ, g = ks.finalize()
    assert abs(logZ - ref.item()) < 1e-6
    assert (g - g_ref).norm().item() < 1e-6 * max(g_ref.norm().item(), 1.0)

    logZ0, g0 = KSStream(p).finalize()
    assert logZ0 == -math.inf and g0.abs().sum() == 0


# ---------------------------------------------------------------------------
# Undercut: silhouette gate and mesh KS margin
# ---------------------------------------------------------------------------

def ring_triangles(outer: float, inner: float, x0: float = 0.0) -> np.ndarray:
    def sq(h):
        return np.array([[x0, -h, -h], [x0, h, -h], [x0, h, h], [x0, -h, h]])

    O, I = sq(outer), sq(inner)
    tris = []
    for k in range(4):
        a, b = O[k], O[(k + 1) % 4]
        c, d = I[(k + 1) % 4], I[k]
        tris += [[a, b, c], [a, c, d]]
    return np.array(tris)


def test_outlet_silhouette_geometry():
    tris = ring_triangles(outer=10.0, inner=3.0)
    d = [-1.0, 0.0, 0.0]
    sil = build_outlet_silhouette(tris, d, margin=0.0)
    inside = sil.inside(np.array([[5.0, 0.0, 5.0], [5.0, 0.0, 0.0], [5.0, 0.0, 12.0]]))
    assert inside.tolist() == [True, False, False]
    assert abs(sil.area - (20.0 ** 2 - 6.0 ** 2)) < 1e-6

    sil_m = build_outlet_silhouette(tris, d, margin=2.5)
    assert sil_m.inside(np.array([[5.0, 0.0, 11.5], [5.0, 0.0, 1.0]])).all()

    d_obl = np.array([-1.0, 0.3, 0.2])
    d_obl /= np.linalg.norm(d_obl)
    p = np.array([0.0, 0.0, 5.0]) - 7.0 * d_obl
    assert build_outlet_silhouette(tris, d_obl).inside(p[None, :])[0]


def frustum_channel(n_seg=64, n_ax=24, L=20.0, r0=5.0, slope=0.15, bump_amp=0.0, bump_x0=4.0, bump_w=1.2):
    xs = np.linspace(0.0, L, n_ax)
    th = np.linspace(0.0, 2 * np.pi, n_seg, endpoint=False)
    r = r0 + slope * xs + bump_amp * np.exp(-((xs - bump_x0) / bump_w) ** 2)
    verts = np.array([[x, ri * np.cos(t), ri * np.sin(t)] for x, ri in zip(xs, r) for t in th])
    faces = []
    for i in range(n_ax - 1):
        for j in range(n_seg):
            a = i * n_seg + j
            b = i * n_seg + (j + 1) % n_seg
            c = (i + 1) * n_seg + (j + 1) % n_seg
            dd = (i + 1) * n_seg + j
            faces += [[a, b, c], [a, c, dd]]
    return torch.tensor(verts, dtype=torch.float64), torch.tensor(faces, dtype=torch.long)


def _frozen_ks(verts, faces, d_vec, gs, keep, w, rho):
    f = faces
    v0, v1, v2 = verts[f[:, 0]], verts[f[:, 1]], verts[f[:, 2]]
    fn = torch.cross(v1 - v0, v2 - v0, dim=1)
    n = fn / fn.norm(dim=1).clamp_min(1e-20).unsqueeze(1)
    nd = (n * d_vec).sum(dim=1) * gs
    return torch.logsumexp(rho * nd[keep] + torch.log(w), dim=0) / rho


def test_undercut_mesh_ks_margin_frustum():
    rho, slope = 50.0, 0.15
    sin_a = math.sin(math.atan(slope))
    d_draw = [-1.0, 0.0, 0.0]
    verts, faces = frustum_channel(slope=slope)
    M, *_ = undercut_penalty(verts, faces, d_draw, surface="cavity", formulation="ks_margin", ks_rho=rho)
    assert abs(M.item() + sin_a) < 2e-3
    M_flip, *_ = undercut_penalty(verts, faces, [1.0, 0.0, 0.0], surface="cavity", formulation="ks_margin", ks_rho=rho)
    assert abs(M_flip.item() - sin_a) < 2e-3

    verts_b, faces_b = frustum_channel(slope=slope, bump_amp=0.8)
    M_all, *_ = undercut_penalty(verts_b, faces_b, d_draw, surface="cavity", formulation="ks_margin", ks_rho=rho)
    th = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    ring = np.stack([np.zeros_like(th), 7.0 * np.cos(th), 7.0 * np.sin(th)], axis=1)
    disk = np.array([[np.zeros(3), ring[k], ring[(k + 1) % 64]] for k in range(64)])
    sil = build_outlet_silhouette(disk, d_draw)
    M_out, *_ = undercut_penalty(verts_b, faces_b, d_draw, surface="cavity", formulation="ks_margin",
                                 ks_rho=rho, silhouette=sil)
    assert M_all.item() > 0.2
    assert abs(M_out.item() + sin_a) < 5e-3

    # gradient against a frozen re-evaluation (masks, weights and sign detached as in production)
    verts_g = verts_b.clone().requires_grad_(True)
    M_g, _, _, _, n_or, _ = undercut_penalty(verts_g, faces_b, d_draw, surface="cavity", formulation="ks_margin",
                                             ks_rho=rho, silhouette=sil)
    g = torch.autograd.grad(M_g, verts_g)[0].reshape(-1)
    d_vec = torch.tensor(d_draw, dtype=torch.float64)
    v0, v1, v2 = verts_b[faces_b[:, 0]], verts_b[faces_b[:, 1]], verts_b[faces_b[:, 2]]
    fn = torch.cross(v1 - v0, v2 - v0, dim=1)
    area = 0.5 * fn.norm(dim=1)
    n = fn / fn.norm(dim=1).clamp_min(1e-20).unsqueeze(1)
    gs = torch.sign((n[0] * n_or[0]).sum())
    keep = (n * d_vec).sum(dim=1).abs() <= math.cos(math.radians(30.0))
    keep &= torch.as_tensor(~sil.inside(((v0 + v1 + v2) / 3.0).numpy()), dtype=torch.bool)
    w = area[keep] / area[keep].sum()
    assert abs(_frozen_ks(verts_b, faces_b, d_vec, gs, keep, w, rho).item() - M_g.item()) < 1e-12

    eps, flat = 1e-6, verts_b.reshape(-1)
    for idx in torch.argsort(g.abs(), descending=True)[:6].tolist():
        orig = flat[idx].item()
        vals = []
        for s in (+eps, -eps):
            flat[idx] = orig + s
            vals.append(_frozen_ks(verts_b, faces_b, d_vec, gs, keep, w, rho).item())
        flat[idx] = orig
        fd = (vals[0] - vals[1]) / (2 * eps)
        assert abs(g[idx].item() - fd) < 1e-3 * max(abs(fd), 1e-30)


# ---------------------------------------------------------------------------
# Minimum steg length on an analytic box SDF
# ---------------------------------------------------------------------------

DESIGN_DOMAIN = [[-20.0, -20.0, -20.0], [20.0, 20.0, 20.0]]
FLOW_DIR = [1.0, 0.0, 0.0]
T_MIN, L_MIN = 1.0, 10.0
COMMON = dict(grid_spacing=0.5, ray_step_mm=0.5, tau_mm=0.1, slab_margin=0.5, formulation="ks_margin", ks_rho=50.0)


class BoxesSDF(torch.nn.Module):
    """Solid-positive SDF of a union of boxes with the lattice interface the penalty uses."""

    def __init__(self, frame, make_boxes, t_half, length):
        super().__init__()
        self.param = torch.nn.Parameter(torch.tensor([t_half, length], dtype=torch.float32))
        self.parametrization = self
        self.bounds = frame.box_norm.clone()
        self.frame = frame
        self.make_boxes = make_boxes

    def forward(self, x_norm):
        x = x_norm / self.frame.scale + self.frame.center
        phi = None
        for lo, hi in self.make_boxes(self.param):
            c, h = 0.5 * (lo + hi), 0.5 * (hi - lo)
            q = (x - c).abs() - h
            outside = q.clamp(min=0.0).pow(2).sum(dim=1).clamp_min(1e-24).sqrt()
            inside = q.max(dim=1).values.clamp(max=0.0)
            p = -(outside + inside)
            phi = p if phi is None else torch.maximum(phi, p)
        if phi is None:
            phi = x.new_full((x.shape[0],), -5.0)
        return phi * self.frame.scale


def _slab(param, x_lo, x_hi, z_half):
    t = param[0]
    c = lambda v: torch.as_tensor(float(v), dtype=torch.float32) if not torch.is_tensor(v) else v
    return (torch.stack([c(x_lo), -t, c(-z_half)]), torch.stack([c(x_hi), t, c(z_half)]))


def cantilever(param):
    return [_slab(param, -10.0, -10.0 + param[1], 8.0)]


def backed(param):
    plate = _slab(param, -10.0, -10.0 + param[1], 8.0)
    x_join = -10.0 + param[1] - 0.5
    block = (torch.stack([x_join, torch.tensor(-6.0), torch.tensor(-10.0)]), torch.tensor([12.0, 6.0, 10.0]))
    return [plate, block]


def past_box(param):
    return [_slab(param, 6.0, 6.0 + param[1], 8.0)]


def evaluate(ls, frame, length_mode, formulation="ks_margin"):
    kw = dict(COMMON, formulation=formulation)
    return min_steg_length_penalty_sdf(ls, frame, ls.param, FLOW_DIR, T_MIN, L_MIN, length_mode=length_mode, **kw)


@pytest.fixture(scope="module")
def frame():
    return DomainFrame.from_design_domain(DESIGN_DOMAIN)


def test_steg_length_semantics_downstream_reach(frame):
    M_cant, g_cant, n_cand, n_flag, *_ = evaluate(BoxesSDF(frame, cantilever, 0.4, 6.0), frame, "downstream_reach")
    assert M_cant.item() > 0.3 and n_flag > 0
    assert g_cant[1].item() < 0.0  # lengthening is the remedy
    M_back, *_ = evaluate(BoxesSDF(frame, backed, 0.4, 6.0), frame, "downstream_reach")
    assert M_back.item() < -0.05
    M_thick, *_ = evaluate(BoxesSDF(frame, cantilever, 3.0, 6.0), frame, "downstream_reach")
    assert abs(M_thick.item() + 1.0) < 1e-6
    M_empty, _, n_cand_e, *_ = evaluate(BoxesSDF(frame, lambda p: [], 0.4, 6.0), frame, "downstream_reach")
    assert abs(M_empty.item() + 1.0) < 1e-6 and n_cand_e == 0
    M_past, *_ = evaluate(BoxesSDF(frame, past_box, 0.4, 20.0), frame, "downstream_reach")
    assert M_past.item() > 0.3  # the void past the box face is air, not support


def test_steg_length_penalty_formulation(frame):
    W_cant, *_ = evaluate(BoxesSDF(frame, cantilever, 0.4, 6.0), frame, "downstream_reach", formulation="penalty")
    W_back, *_ = evaluate(BoxesSDF(frame, backed, 0.4, 6.0), frame, "downstream_reach", formulation="penalty")
    assert W_cant.item() > 0.05
    assert W_back.item() < 1e-3


def test_steg_length_thin_band_mode(frame):
    short = BoxesSDF(frame, lambda p: [_slab(p, -3.0, -3.0 + p[1], 8.0)], 0.35, 6.0)
    long = BoxesSDF(frame, lambda p: [_slab(p, -15.0, -15.0 + p[1], 8.0)], 0.35, 30.0)
    M_short, *_ = evaluate(short, frame, "thin_band")
    M_long, *_ = evaluate(long, frame, "thin_band")
    assert M_short.item() > 0.15
    assert M_long.item() < -0.01


def test_steg_length_gradient_matches_finite_difference(frame):
    """Central FD of the margin in the length parameter (the direction with signal)."""
    ls = BoxesSDF(frame, cantilever, 0.4, 6.0)
    M0, g0, *_ = evaluate(ls, frame, "downstream_reach")
    eps = 5e-3
    vals = []
    for s in (+eps, -eps):
        ls.param.data[1] += s
        vals.append(float(evaluate(ls, frame, "downstream_reach")[0]))
        ls.param.data[1] -= s
    fd = (vals[0] - vals[1]) / (2 * eps)
    assert fd < 0.0
    assert abs(g0[1].item() - fd) < 0.1 * abs(fd)

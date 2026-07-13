"""Offline gates for the min-steg-length ks_margin formulation (no OpenFOAM required).

Three parts, most-synthetic first:

1. ksstream    -- pure-math self-test of :class:`KSStream` (the streaming chunked
   logsumexp): value and gradient must match a single torch.logsumexp over the
   concatenated chunks. Always runs, no model needed.
2. synthetic   -- ks_margin gates on an analytic box-SDF "lattice" (a thin plate steg
   whose half-thickness and length are the two entries of param, so both remedy
   directions have autograd paths):
     * cantilever (thin, short, free downstream end)      -> M >> 0 (infeasible)
     * same plate backed by a thick block downstream      -> M < 0  (real slack)
     * thick plate (no thin material)                     -> M = -1 (sentinel)
     * empty box (no solid at all)                        -> M = -1 (early return)
     * plate extending past the box face                  -> M >> 0 (OUT-OF-BOX = AIR:
       the void past the outlet face must not count as support)
     * penalty-formulation regression (existing path untouched)
     * thin_band mode: short plate infeasible, long plate feasible
   Gradient gates: the frozen re-evaluation (base thin points + base normalized KS
   weights, only Lx recomputed -- exactly the function production autograd
   differentiates, since the weights are detached there) must match production in
   value AND gradient; central FD of the frozen function must match its autograd.
   The production-vs-frozen gradient match IS the lengthen-only check: the frozen
   function contains no thin-gate (tw) path at all, so equality proves the
   production gradient has none either. dM/d(length) < 0 is checked explicitly.
3. sdf         -- the same frozen-FD gate on the real lattice (needs --config and an
   existing reconstruction): analytic dM/dparam vs central FD over the top-k |g|
   latent components, frozen (thin points + ks_weight from the base evaluation's
   debug cloud) and raw (informational: contains the DESIGNED-IN drift from the
   detached weights).

Usage:
    uv run python check_min_steg_fd.py                       # parts 1 + 2 (fast)
    uv run python check_min_steg_fd.py \
        --config <app-repo>/experiments/optimization/<experiment>/config.json \
        --results-name results \
        [--n-components 4] [--eps 1e-4]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

from deepshapeopt.geometry_constraints import KSStream, min_steg_length_penalty_sdf

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# Part 1: KSStream self-test
# ---------------------------------------------------------------------------

def part1_ksstream() -> None:
    print("\n=== Part 1: KSStream (streaming chunked logsumexp) ===")
    torch.manual_seed(0)
    p = torch.randn(7, requires_grad=True)
    # Exponents spread over many orders of magnitude and hostile chunk order (the
    # running max shifts up mid-stream), plus an empty and an all -inf chunk.
    a_full = torch.cat([p * 3.0, p ** 2 - 10.0, p.abs() * 20.0, p * 0.0 - 500.0])
    ref = torch.logsumexp(a_full, dim=0)
    g_ref = torch.autograd.grad(ref, p, retain_graph=True)[0]

    ks = KSStream(p)
    ks.add((p ** 2 - 10.0))
    ks.add(p.new_zeros(0))                       # empty chunk: no-op
    ks.add(torch.full((3,), -math.inf))          # all -inf chunk: no-op
    ks.add(p * 3.0)
    ks.add(p.abs() * 20.0)                       # max shifts up here
    ks.add(p * 0.0 - 500.0)                      # negligible tail chunk
    logZ, g = ks.finalize()
    check("value matches torch.logsumexp", abs(logZ - ref.item()) < 1e-6,
          f"diff={logZ - ref.item():.2e}")
    check("gradient matches torch.logsumexp",
          (g - g_ref).norm().item() < 1e-6 * max(g_ref.norm().item(), 1.0),
          f"rel={(g - g_ref).norm().item() / max(g_ref.norm().item(), 1e-30):.2e}")

    ks0 = KSStream(p)
    logZ0, g0 = ks0.finalize()
    check("empty stream -> (-inf, zeros)", logZ0 == -math.inf and g0.abs().sum() == 0)


# ---------------------------------------------------------------------------
# Part 2: synthetic box-SDF plate steg
# ---------------------------------------------------------------------------

DESIGN_DOMAIN = [[-20.0, -20.0, -20.0], [20.0, 20.0, 20.0]]
FLOW_DIR = [1.0, 0.0, 0.0]          # downstream = +x
T_MIN_MM = 1.0                       # "thin" below this cross-flow thickness
L_MIN_MM = 10.0                      # required streamwise extent
COMMON = dict(grid_spacing=0.5, ray_step_mm=0.5, tau_mm=0.1, slab_margin=0.5,
              formulation="ks_margin", ks_rho=50.0)


class BoxesSDF(torch.nn.Module):
    """Analytic solid-positive SDF of a union of axis-aligned boxes -- the synthetic
    "lattice". Mimics the LatticeSDFStruct interface used by the penalty
    (callable on normalized points; ``.parametrization`` / ``.bounds`` for
    with_float32_lattice). ``param = (plate_half_thickness_mm, plate_length_mm)``
    feeds ``make_boxes(param) -> [(lo_mm, hi_mm), ...]``, so both the "thicken" and
    the "lengthen" remedies have real autograd paths into phi.
    """

    def __init__(self, frame, make_boxes, t_half_mm, len_mm):
        super().__init__()
        self.param = torch.nn.Parameter(
            torch.tensor([t_half_mm, len_mm], dtype=torch.float32))
        self.parametrization = self          # .parameters() -> param
        self.bounds = frame.box_norm.clone()
        self.frame = frame
        self.make_boxes = make_boxes

    def forward(self, x_norm):
        x_mm = x_norm / self.frame.scale + self.frame.center
        phi_mm = None
        for lo, hi in self.make_boxes(self.param):
            c, h = 0.5 * (lo + hi), 0.5 * (hi - lo)
            q = (x_mm - c).abs() - h
            # clamp_min before sqrt keeps the gradient finite (0) at interior points,
            # where the outside term is exactly 0 and d sqrt(0) would be NaN.
            outside = q.clamp(min=0.0).pow(2).sum(dim=1).clamp_min(1e-24).sqrt()
            inside = q.max(dim=1).values.clamp(max=0.0)
            p = -(outside + inside)          # solid-positive box SDF (mm)
            phi_mm = p if phi_mm is None else torch.maximum(phi_mm, p)
        if phi_mm is None:                   # empty geometry: everything fluid
            phi_mm = x_mm.new_full((x_mm.shape[0],), -5.0)
        return phi_mm * self.frame.scale     # normalized SDF units, like the lattice


def _box(param, x_lo, x_hi, z_half):
    """Plate slab |y| <= t_half, x in [x_lo, x_hi], |z| <= z_half (any grad-connected)."""
    t = param[0]
    c = lambda v: torch.as_tensor(float(v), dtype=torch.float32) if not torch.is_tensor(v) else v
    return (torch.stack([c(x_lo), -t, c(-z_half)]), torch.stack([c(x_hi), t, c(z_half)]))


def cantilever_boxes(param):
    # Thin plate x in [-10, -10 + L], free downstream end in open fluid.
    return [_box(param, -10.0, -10.0 + param[1], 8.0)]


def backed_boxes(param):
    # Same plate, but merged into a thick block downstream (>= 1.15*L_min long):
    # every thin point is backed, the margin must show real negative slack.
    plate = _box(param, -10.0, -10.0 + param[1], 8.0)
    x_join = -10.0 + param[1] - 0.5          # 0.5 mm overlap keeps the union contiguous
    block = (torch.stack([x_join, torch.tensor(-6.0), torch.tensor(-10.0)]),
             torch.tensor([12.0, 6.0, 10.0]))
    return [plate, block]


def pastbox_boxes(param):
    # Plate runs THROUGH the +x box face (the "outlet"): the analytic SDF keeps
    # reading solid past the face, so only the OUT-OF-BOX = AIR override in the
    # penalty can stop the downstream run there.
    return [_box(param, 6.0, 6.0 + param[1], 8.0)]


def make_frozen(ls, frame, param, pts_phys, ks_w, *, flow_dir, min_length_mm,
                grid_spacing, ray_step_mm=None, tau_mm=None,
                thickness_threshold_mm=1.0, length_mode="downstream_reach",
                ks_rho=50.0, chunk=8192):
    """Frozen re-evaluation of the ks_margin: base thin-candidate points + base
    NORMALIZED KS weights fixed, only the streamwise reach Lx (hence the margin
    s = 1 - Lx/L_min) recomputed -- exactly the function production autograd
    differentiates (weights/selection are detached there). Constants are derived
    exactly as in min_steg_length_penalty_sdf; the value-match check below catches
    any drift between this reimplementation and production.
    """
    from deepshapeopt.reconstruction import with_float32_lattice

    device = param.device
    scale = float(frame.scale)
    L_min = scale * float(min_length_mm)
    ray_step = scale * (float(ray_step_mm) if ray_step_mm is not None else float(grid_spacing))
    tau = scale * (float(tau_mm) if tau_mm is not None else 0.4 * float(grid_spacing))
    h = 0.5 * scale * float(thickness_threshold_mm)
    n_axial = max(2, int(math.ceil(1.15 * L_min / ray_step)))
    b_thr, tau_gate = 0.5, 0.12
    phi_stop = -2.0 * tau
    tau_stop = 0.5 * tau
    phi_air = phi_stop - 4.0 * tau_stop
    box_norm = frame.box_norm.to(device=device, dtype=torch.float32)
    lo, hi = box_norm[0], box_norm[1]
    d = torch.as_tensor(flow_dir, dtype=torch.float32, device=device)
    d = d / d.norm()
    rho = float(ks_rho)

    Xb = frame.to_norm(
        torch.as_tensor(pts_phys, dtype=param.dtype, device=device)).to(torch.float32)
    w = torch.as_tensor(ks_w, dtype=torch.float32, device=device)
    pos = w > 0
    Xp, logw = Xb[pos], torch.log(w[pos])

    def frozen(want_grad=False):
        def _compute(_b):
            def q(xx):
                return ls(xx).reshape(-1)

            def march(Xc):
                nc = Xc.shape[0]
                Lx = torch.zeros(nc, device=device)
                if length_mode == "downstream_reach":
                    m = torch.full((nc,), float("inf"), device=device)
                    for k in range(1, n_axial + 1):
                        Pk = Xc + (k * ray_step) * d
                        phi_k = q(Pk)
                        out = ((Pk < lo) | (Pk > hi)).any(dim=1)
                        phi_k = torch.where(out, torch.full_like(phi_k, phi_air), phi_k)
                        m = torch.minimum(m, phi_k)
                        Lx = Lx + ray_step * torch.sigmoid((m - phi_stop) / tau_stop)
                else:  # thin_band
                    for sgn in (1.0, -1.0):
                        m = torch.ones(nc, device=device)
                        for k in range(1, n_axial + 1):
                            phi_k = q(Xc + (sgn * k * ray_step) * d)
                            bk = torch.sigmoid(phi_k / tau) * torch.sigmoid((h - phi_k) / tau)
                            m = torch.minimum(m, bk)
                            Lx = Lx + ray_step * torch.sigmoid((m - b_thr) / tau_gate)
                return Lx

            ks = KSStream(param if want_grad else None)
            for ci in range(0, Xp.shape[0], chunk):
                if want_grad:
                    s = 1.0 - march(Xp[ci:ci + chunk]) / L_min
                else:
                    with torch.no_grad():
                        s = 1.0 - march(Xp[ci:ci + chunk]) / L_min
                ks.add(rho * s + logw[ci:ci + chunk])
            logZ, dlogZ = ks.finalize()
            # logw is normalized (sums to 1), so M = logZ/rho directly.
            return logZ / rho, (None if dlogZ is None else dlogZ / rho)
        return with_float32_lattice(ls, frame.box_norm, _compute)

    return frozen


def eval_prod(ls, frame, length_mode, formulation="ks_margin"):
    kw = dict(COMMON)
    kw["formulation"] = formulation
    return min_steg_length_penalty_sdf(
        ls, frame, ls.param, FLOW_DIR, T_MIN_MM, L_MIN_MM,
        length_mode=length_mode, **kw)


def part2_synthetic(eps=5e-3, fd_modes=("downstream_reach", "thin_band")) -> None:
    print("\n=== Part 2: synthetic plate steg (analytic box SDF) ===")
    from deepshapeopt.domain_frame import DomainFrame

    frame = DomainFrame.from_design_domain(DESIGN_DOMAIN)

    # (a) value semantics, downstream_reach (the production config's mode).
    ls_cant = BoxesSDF(frame, cantilever_boxes, t_half_mm=0.4, len_mm=6.0)
    M_cant, g_cant, n_cand, n_flag, pts_c, sc_c, _ = eval_prod(ls_cant, frame, "downstream_reach")
    s_max = float((1.0 - np.asarray(sc_c["Lx_mm"]) / L_MIN_MM).max())
    print(f"  cantilever: M={M_cant.item():+.4f} (true worst s={s_max:+.4f}), "
          f"cand={n_cand}, thin&short={n_flag}")
    check("cantilever infeasible (M > 0.3)", M_cant.item() > 0.3, f"M={M_cant.item():+.4f}")

    ls_back = BoxesSDF(frame, backed_boxes, t_half_mm=0.4, len_mm=6.0)
    M_back, *_ = eval_prod(ls_back, frame, "downstream_reach")
    check("backed plate feasible (M < -0.05)", M_back.item() < -0.05,
          f"M={M_back.item():+.4f}")

    ls_thick = BoxesSDF(frame, cantilever_boxes, t_half_mm=3.0, len_mm=6.0)
    M_thick, _, n_cand_t, *_ = eval_prod(ls_thick, frame, "downstream_reach")
    check("thick plate: no thin material -> sentinel -1",
          abs(M_thick.item() + 1.0) < 1e-6, f"M={M_thick.item():+.4f} cand={n_cand_t}")

    ls_empty = BoxesSDF(frame, lambda p: [], t_half_mm=0.4, len_mm=6.0)
    M_empty, _, n_cand_e, *_ = eval_prod(ls_empty, frame, "downstream_reach")
    check("empty box: no candidates -> sentinel -1",
          abs(M_empty.item() + 1.0) < 1e-6 and n_cand_e == 0,
          f"M={M_empty.item():+.4f} cand={n_cand_e}")

    # Plate running through the +x box face: the analytic SDF reads solid out there,
    # so a feasible M would mean the penalty counted the void past the "outlet" as
    # support -- the OUT-OF-BOX = AIR override must keep this infeasible.
    ls_past = BoxesSDF(frame, pastbox_boxes, t_half_mm=0.4, len_mm=20.0)
    M_past, *_ = eval_prod(ls_past, frame, "downstream_reach")
    check("plate past box face: out-of-box counts as air (M > 0.3)",
          M_past.item() > 0.3, f"M={M_past.item():+.4f}")

    # (b) penalty-formulation regression (existing path must be untouched).
    W_cant, *_ = eval_prod(ls_cant, frame, "downstream_reach", formulation="penalty")
    W_back, *_ = eval_prod(ls_back, frame, "downstream_reach", formulation="penalty")
    check("penalty regression: cantilever W > 0.05", W_cant.item() > 0.05,
          f"W={W_cant.item():.3e}")
    check("penalty regression: backed W < 1e-3", W_back.item() < 1e-3,
          f"W={W_back.item():.3e}")

    # (c) thin_band mode semantics (measures the steg's OWN extent, both directions).
    ls_short = BoxesSDF(frame, lambda p: [_box(p, -3.0, -3.0 + p[1], 8.0)],
                        t_half_mm=0.35, len_mm=6.0)
    M_short, *_ = eval_prod(ls_short, frame, "thin_band")
    ls_long = BoxesSDF(frame, lambda p: [_box(p, -15.0, -15.0 + p[1], 8.0)],
                       t_half_mm=0.35, len_mm=30.0)
    M_long, *_ = eval_prod(ls_long, frame, "thin_band")
    check("thin_band: short plate infeasible (M > 0.15)", M_short.item() > 0.15,
          f"M={M_short.item():+.4f}")
    check("thin_band: long plate feasible (M < -0.01)", M_long.item() < -0.01,
          f"M={M_long.item():+.4f}")

    # (d) frozen value/gradient/FD gates per length_mode, on the violating geometry
    # (nontrivial gradient). The frozen function has NO thin-gate (tw) path at all,
    # so "frozen grad == production grad" proves production's gradient is
    # lengthen-only (the tw path is detached).
    cases = {"downstream_reach": (ls_cant, M_cant, g_cant, pts_c, sc_c),
             "thin_band": (ls_short, M_short, None, None, None)}
    for mode in fd_modes:
        ls, M0, g0, pts, sc = cases[mode]
        if g0 is None:
            M0, g0, _, _, pts, sc, _ = eval_prod(ls, frame, mode)
        print(f"  -- frozen/FD gates, length_mode={mode} --")
        frozen = make_frozen(
            ls, frame, ls.param, pts, sc["ks_weight"],
            flow_dir=FLOW_DIR, min_length_mm=L_MIN_MM,
            grid_spacing=COMMON["grid_spacing"], ray_step_mm=COMMON["ray_step_mm"],
            tau_mm=COMMON["tau_mm"], thickness_threshold_mm=T_MIN_MM,
            length_mode=mode, ks_rho=COMMON["ks_rho"])
        M_fro, g_fro = frozen(want_grad=True)
        check(f"[{mode}] frozen re-eval matches production value",
              abs(M_fro - M0.item()) < 2e-3, f"diff={M_fro - M0.item():.2e}")
        rel_g = (g_fro - g0).norm().item() / max(g0.norm().item(), 1e-30)
        check(f"[{mode}] frozen grad == production grad (lengthen-only proof, <1%)",
              rel_g < 1e-2, f"rel diff={100 * rel_g:.3f}%")
        check(f"[{mode}] dM/d(length) < 0 (lengthening is the remedy)",
              g0[1].item() < 0.0,
              f"dM/dL={g0[1].item():+.4e}, dM/dt={g0[0].item():+.4e}")

        print(f"  {'param':>7} {'autograd':>14} {'FD frozen':>14} {'rel err':>9} {'FD raw':>14}")
        max_rel = 0.0
        flat = ls.param.data
        for idx, name in ((1, "len"), (0, "t")):
            orig = flat[idx].item()
            vals_fro, vals_raw = [], []
            for s in (+eps, -eps):
                flat[idx] = orig + s
                vals_fro.append(frozen(want_grad=False)[0])
                vals_raw.append(float(eval_prod(ls, frame, mode)[0]))
            flat[idx] = orig
            fd_fro = (vals_fro[0] - vals_fro[1]) / (2 * eps)
            fd_raw = (vals_raw[0] - vals_raw[1]) / (2 * eps)
            a = g_fro[idx].item()
            # Gate relative where the signal is real; tiny components (the thickness
            # one can be near zero by design) are gated absolutely instead.
            if abs(fd_fro) > 1e-4:
                rel = abs(a - fd_fro) / abs(fd_fro)
                max_rel = max(max_rel, rel)
            else:
                rel = float("nan")
                check(f"[{mode}] near-zero {name}-component matches absolutely",
                      abs(a - fd_fro) < 5e-4, f"a={a:.2e} fd={fd_fro:.2e}")
            print(f"  {name:>7} {a:>14.6e} {fd_fro:>14.6e} {100 * rel:>8.3f}% {fd_raw:>14.6e}")
        check(f"[{mode}] frozen-FD gate (max rel err < 2%)", max_rel < 2e-2,
              f"max={100 * max_rel:.3f}%")


# ---------------------------------------------------------------------------
# Part 3: sdf ks margin on the real lattice
# ---------------------------------------------------------------------------

def part3_sdf(args) -> None:
    print("\n=== Part 3: sdf ks_margin (real lattice) ===")
    import deepshapeopt.config as dso_config
    from deepshapeopt.config import ExperimentSpecifications
    from deepshapeopt.shape_optimization import build_lattice, setup_model_and_domain

    from deepshapeopt.config import make_setup_name

    specs = ExperimentSpecifications(args.config)
    rec_cfg = specs["reconstruction"]
    opt_cfg = specs["optimization"]
    experiment_path = Path(args.config).resolve().parent

    con_cfg = opt_cfg.get("constraint") or {}
    setup_name = make_setup_name(
        opt_cfg["objective_name"], bool(con_cfg.get("enabled", False)),
        con_cfg.get("name"), con_cfg,
    )
    results_name = args.results_name or specs.get("results_name", "results")
    paths = dso_config.make_experiment_paths(
        experiment_path, results_name=results_name, run_subdir=setup_name)
    rec_file = paths.reconstruction / "rec_parameters.pt"
    if not rec_file.exists():
        sys.exit(f"No reconstruction at {rec_file}; run the driver once or pass --results-name.")

    model_setup = setup_model_and_domain(rec_cfg, paths.reconstruction)
    lattice = build_lattice(rec_cfg, model_setup.model, model_setup.sdf, model_setup.frame)
    recon = torch.load(rec_file, map_location=rec_cfg["device"])
    lattice.lattice_struct.parametrization.set_param(
        recon[0].to(device=model_setup.model.device, dtype=torch.float32))
    param = next(lattice.lattice_struct.parametrization.parameters())
    frame = model_setup.frame
    ls = lattice.lattice_struct

    sl = opt_cfg.get("min_steg_length") or {}
    flow_dir = sl.get("flow_direction", [1.0, 0.0, 0.0])
    t_mm = float(sl.get("thickness_threshold_mm", 1.0))
    L_mm = float(sl.get("min_length_mm", 10.0))
    kw = dict(
        grid_spacing=float(sl.get("grid_spacing", 0.25)),
        ray_step_mm=sl.get("ray_step_mm"), tau_mm=sl.get("tau_mm"),
        slab_margin=float(sl.get("slab_margin", 0.5)),
        exclude_region=sl.get("exclude_region"),
        length_mode=sl.get("length_mode", "thin_band"),
        formulation="ks_margin", ks_rho=float(sl.get("ks_rho", 50.0)),
    )
    M0, g, n_cand, n_flag, pts_phys, scalars, _ = min_steg_length_penalty_sdf(
        ls, frame, param, flow_dir, t_mm, L_mm, **kw)
    print(f"  base: M={M0.item():+.6e} (worst shortfall {M0.item() * L_mm:+.3f} mm), "
          f"cand={n_cand}, thin&short={n_flag}, |g|={g.norm().item():.3e}")
    if pts_phys is None:
        print("  no thin candidates on this design: frozen-FD gate has nothing to hold "
              "onto (sentinel path already covered by part 2); skipping.")
        return

    frozen = make_frozen(
        ls, frame, param, pts_phys, scalars["ks_weight"],
        flow_dir=flow_dir, min_length_mm=L_mm, grid_spacing=kw["grid_spacing"],
        ray_step_mm=kw["ray_step_mm"], tau_mm=kw["tau_mm"],
        thickness_threshold_mm=t_mm, length_mode=kw["length_mode"],
        ks_rho=kw["ks_rho"])
    M_fro, g_fro = frozen(want_grad=True)
    g_fro = g_fro.reshape(-1)
    check("sdf frozen re-eval matches production value",
          abs(M_fro - M0.item()) < 1e-3, f"diff={M_fro - M0.item():.2e}")
    rel_g = (g_fro - g.reshape(-1)).norm().item() / max(g.norm().item(), 1e-30)
    check("sdf frozen grad == production grad (lengthen-only proof, <1%)",
          rel_g < 1e-2, f"rel diff={100 * rel_g:.3f}%")

    top = torch.argsort(g.reshape(-1).abs(), descending=True)[: args.n_components]
    flat = param.data.reshape(-1)
    eps = args.eps
    print(f"  {'comp':>6} {'autograd':>14} {'FD frozen':>14} {'rel err':>9} {'FD raw':>14}")
    max_rel, rows = 0.0, []
    for idx in top.tolist():
        orig = flat[idx].item()
        vals_fro, vals_raw = [], []
        for s in (+eps, -eps):
            flat[idx] = orig + s
            vals_fro.append(frozen(want_grad=False)[0])
            vals_raw.append(float(min_steg_length_penalty_sdf(
                ls, frame, param, flow_dir, t_mm, L_mm, **kw)[0]))
        flat[idx] = orig
        fd_fro = (vals_fro[0] - vals_fro[1]) / (2 * eps)
        fd_raw = (vals_raw[0] - vals_raw[1]) / (2 * eps)
        a = g_fro[idx].item()
        rel = abs(a - fd_fro) / max(abs(fd_fro), 1e-30)
        max_rel = max(max_rel, rel)
        rows.append((a, fd_fro))
        print(f"  {idx:>6} {a:>14.6e} {fd_fro:>14.6e} {100 * rel:>8.3f}% {fd_raw:>14.6e}")
    av = torch.tensor([r[0] for r in rows])
    fv = torch.tensor([r[1] for r in rows])
    cos = torch.nn.functional.cosine_similarity(av, fv, dim=0).item()
    check("sdf frozen-FD gate (max rel err < 2%)", max_rel < 2e-2, f"max={100 * max_rel:.3f}%")
    check("sdf frozen-FD cosine > 0.999", cos > 0.999, f"cos={cos:.5f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="Experiment JSON; enables part 3 (sdf gate)")
    p.add_argument("--results-name", help="Existing results dir with a reconstruction "
                                          "(defaults to the config's results_name)")
    p.add_argument("--n-components", type=int, default=4)
    # KS with rho ~ 50 is strongly curved (softmax reweighting); like the undercut
    # gate, latent-component FD needs a small eps (1e-4) to sit below the 2% gate.
    p.add_argument("--eps", type=float, default=1e-4)
    args = p.parse_args()

    part1_ksstream()
    part2_synthetic()
    if args.config:
        part3_sdf(args)
    else:
        print("\n(no --config: skipping part 3, the sdf gate on the real lattice)")

    print()
    if FAILURES:
        sys.exit(f"FAILED: {len(FAILURES)} gate(s): " + ", ".join(FAILURES))
    print("All gates passed.")


if __name__ == "__main__":
    main()

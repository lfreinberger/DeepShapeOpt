"""Offline gates for the undercut-penalty variants (no OpenFOAM required).

Three parts, most-synthetic first:

1. silhouette  -- pure-geometry self-test of :func:`build_outlet_silhouette` /
   ``OutletSilhouette.inside``: square-ring outlet polygon (profile hole), margin
   buffer, oblique draw direction. Always runs, no model needed.
2. mesh        -- KS-margin gate on a synthetic frustum channel (cavity surface):
   the analytic margin is known exactly (M = -sin(taper angle) for the drawable
   orientation, +sin for the flipped draw direction, so the cavity orientation
   vote is exercised too); a Gaussian bump adds a genuine undercut INSIDE a disk
   silhouette, so scope=outside_outlet must flip M from infeasible to feasible.
   Gradient gate: autograd dM/dverts vs central FD of a FROZEN re-evaluation
   (base keep-mask/weights/global sign, exactly the detached quantities of the
   production formulation -- must match to FD truncation), plus the raw FD of the
   full production function (informational: contains the DESIGNED-IN drift from
   the detached area weights, damped by 1/rho).
3. sdf         -- KS-margin gate on the real lattice (needs --config and an
   existing reconstruction): analytic dM/dparam vs central FD over the top-k
   |g| latent components, again frozen (band points + weights from the base
   evaluation's debug cloud) and raw. Optionally with a synthetic rectangle
   silhouette in the outlet plane (--with-silhouette).

Usage:
    uv run python check_undercut_fd.py                       # parts 1 + 2 (fast)
    uv run python check_undercut_fd.py \
        --config <app-repo>/experiments/optimization/<experiment>/config.json \
        --results-name results_simpleDie_debug \
        [--n-components 4] [--eps 1e-3] [--with-silhouette]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

from deepshapeopt.geometry_constraints import (
    build_outlet_silhouette,
    undercut_penalty,
    undercut_penalty_sdf,
)

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# Part 1: silhouette geometry self-test
# ---------------------------------------------------------------------------

def ring_triangles(outer: float, inner: float, x0: float = 0.0) -> np.ndarray:
    """Square ring (outer square minus inner-square hole) in the x=x0 plane, 8 triangles."""
    def sq(h):
        return np.array([[x0, -h, -h], [x0, h, -h], [x0, h, h], [x0, -h, h]])

    O, I = sq(outer), sq(inner)
    tris = []
    for k in range(4):
        a, b = O[k], O[(k + 1) % 4]
        c, d = I[(k + 1) % 4], I[k]
        tris += [[a, b, c], [a, c, d]]
    return np.array(tris)


def part1_silhouette() -> None:
    print("\n=== Part 1: silhouette geometry ===")
    tris = ring_triangles(outer=10.0, inner=3.0)
    d = [-1.0, 0.0, 0.0]
    sil = build_outlet_silhouette(tris, d, margin=0.0)

    pts = np.array([
        [5.0, 0.0, 5.0],    # over the ring           -> inside
        [5.0, 0.0, 0.0],    # over the hole           -> outside (profile hole = solid)
        [5.0, 0.0, 12.0],   # laterally past the ring -> outside
    ])
    ins = sil.inside(pts)
    check("ring point inside", bool(ins[0]))
    check("hole point outside", not bool(ins[1]))
    check("far point outside", not bool(ins[2]))
    check("polygon area = outer^2 - hole^2",
          abs(sil.area - (20.0 ** 2 - 6.0 ** 2)) < 1e-6, f"area={sil.area:.6f}")

    # Margin buffers BOTH rims leniently: outer rim grows, hole shrinks.
    sil_m = build_outlet_silhouette(tris, d, margin=2.5)
    ins_m = sil_m.inside(np.array([[5.0, 0.0, 11.5], [5.0, 0.0, 1.0]]))
    check("margin widens outer rim", bool(ins_m[0]))
    check("margin shrinks hole", bool(ins_m[1]))

    # Oblique draw direction: a point offset along -d must project back into the ring.
    d_obl = np.array([-1.0, 0.3, 0.2])
    d_obl = d_obl / np.linalg.norm(d_obl)
    sil_o = build_outlet_silhouette(tris, d_obl)
    p = np.array([0.0, 0.0, 5.0]) - 7.0 * d_obl  # projects back to (0, 0, 5) = ring
    check("oblique projection", bool(sil_o.inside(p[None, :])[0]))


# ---------------------------------------------------------------------------
# Part 2: mesh KS margin on a synthetic frustum channel
# ---------------------------------------------------------------------------

def frustum_channel(n_seg=64, n_ax=24, L=20.0, r0=5.0, slope=0.15,
                    bump_amp=0.0, bump_x0=4.0, bump_w=1.2):
    """Side wall of a channel r(x) = r0 + slope*x (+ optional Gaussian bump), float64."""
    xs = np.linspace(0.0, L, n_ax)
    th = np.linspace(0.0, 2 * np.pi, n_seg, endpoint=False)
    r = r0 + slope * xs + bump_amp * np.exp(-((xs - bump_x0) / bump_w) ** 2)
    verts = np.array([
        [x, ri * np.cos(t), ri * np.sin(t)] for x, ri in zip(xs, r) for t in th
    ])
    faces = []
    for i in range(n_ax - 1):
        for j in range(n_seg):
            a = i * n_seg + j
            b = i * n_seg + (j + 1) % n_seg
            c = (i + 1) * n_seg + (j + 1) % n_seg
            dd = (i + 1) * n_seg + j
            faces += [[a, b, c], [a, c, dd]]
    return (
        torch.tensor(verts, dtype=torch.float64),
        torch.tensor(faces, dtype=torch.long),
    )


def mesh_keep_w_gs(verts, faces, d_vec, exclude_axial_deg, silhouette, n_oriented_ref):
    """Recompute the DETACHED parts of undercut_penalty's ks_margin at the base design:
    keep mask, normalized area weights, global orientation sign (from the reference
    oriented normals). This is the frozen state for the frozen-FD evaluation."""
    f = faces
    v0, v1, v2 = verts[f[:, 0]], verts[f[:, 1]], verts[f[:, 2]]
    fn = torch.cross(v1 - v0, v2 - v0, dim=1)
    area = 0.5 * fn.norm(dim=1)
    n = fn / fn.norm(dim=1).clamp_min(1e-20).unsqueeze(1)
    gs = torch.sign((n[0] * n_oriented_ref[0]).sum())  # winding vs production orientation
    nd = (n * d_vec).sum(dim=1)
    keep = nd.abs() <= math.cos(math.radians(exclude_axial_deg))
    if silhouette is not None:
        cent = ((v0 + v1 + v2) / 3.0).detach().cpu().numpy()
        keep = keep & torch.as_tensor(~silhouette.inside(cent), dtype=torch.bool)
    w = area.detach()[keep]
    w = w / w.sum()
    return keep, w, gs


def mesh_ks_frozen(verts, faces, d_vec, gs, keep, w, rho):
    """KS aggregation with frozen keep/w/gs -- the exact function the analytic
    gradient differentiates (weights and masks are detached in production)."""
    f = faces
    v0, v1, v2 = verts[f[:, 0]], verts[f[:, 1]], verts[f[:, 2]]
    fn = torch.cross(v1 - v0, v2 - v0, dim=1)
    n = fn / fn.norm(dim=1).clamp_min(1e-20).unsqueeze(1)
    nd = (n * d_vec).sum(dim=1) * gs
    return torch.logsumexp(rho * nd[keep] + torch.log(w), dim=0) / rho


def part2_mesh(rho=50.0, n_fd=6, eps=1e-6) -> None:
    print("\n=== Part 2: mesh ks_margin (synthetic frustum, cavity) ===")
    slope = 0.15
    alpha = math.degrees(math.atan(slope))
    sin_a = math.sin(math.atan(slope))
    d_draw = [-1.0, 0.0, 0.0]

    # (a) clean frustum: analytic margin -sin(alpha) when drawable, +sin(alpha) flipped.
    verts, faces = frustum_channel(slope=slope)
    M, _, _, _, n_or, _ = undercut_penalty(
        verts, faces, d_draw, surface="cavity", formulation="ks_margin", ks_rho=rho)
    check("drawable frustum: M = -sin(taper)",
          abs(M.item() + sin_a) < 2e-3, f"M={M.item():+.5f} vs {-sin_a:+.5f} (alpha={alpha:.2f}deg)")
    M_flip, *_ = undercut_penalty(
        verts, faces, [1.0, 0.0, 0.0], surface="cavity", formulation="ks_margin", ks_rho=rho)
    check("flipped draw dir: M = +sin(taper)",
          abs(M_flip.item() - sin_a) < 2e-3, f"M={M_flip.item():+.5f}")

    # (b) bump inside a disk silhouette: scope must flip feasibility.
    verts_b, faces_b = frustum_channel(slope=slope, bump_amp=0.8)
    M_all, _, _, _, n_or_b, _ = undercut_penalty(
        verts_b, faces_b, d_draw, surface="cavity", formulation="ks_margin", ks_rho=rho)
    th = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    ring = np.stack([np.zeros_like(th), 7.0 * np.cos(th), 7.0 * np.sin(th)], axis=1)
    disk = np.array([[np.zeros(3), ring[k], ring[(k + 1) % 64]] for k in range(64)])
    sil = build_outlet_silhouette(disk, d_draw)
    M_out, _, _, _, _, _ = undercut_penalty(
        verts_b, faces_b, d_draw, surface="cavity", formulation="ks_margin",
        ks_rho=rho, silhouette=sil)
    check("bump w/o scope: infeasible (M > 0.2)", M_all.item() > 0.2, f"M_all={M_all.item():+.4f}")
    check("bump inside silhouette exempt: M ~ -sin(taper)",
          abs(M_out.item() + sin_a) < 5e-3, f"M_outside={M_out.item():+.5f}")

    # (c) gradient gate on the bumped + silhouetted case (the hardest path).
    verts_g = verts_b.clone().requires_grad_(True)
    M_g, _, _, _, n_or_g, _ = undercut_penalty(
        verts_g, faces_b, d_draw, surface="cavity", formulation="ks_margin",
        ks_rho=rho, silhouette=sil)
    g = torch.autograd.grad(M_g, verts_g)[0].reshape(-1)

    d_vec = torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float64)
    keep, w, gs = mesh_keep_w_gs(verts_b, faces_b, d_vec, 30.0, sil, n_or_g)
    M_frozen0 = mesh_ks_frozen(verts_b, faces_b, d_vec, gs, keep, w, rho)
    check("frozen re-eval matches production", abs(M_frozen0.item() - M_g.item()) < 1e-12,
          f"diff={M_frozen0.item() - M_g.item():.2e}")

    top = torch.argsort(g.abs(), descending=True)[:n_fd]
    flat = verts_b.reshape(-1)
    print(f"  {'coord':>7} {'autograd':>14} {'FD frozen':>14} {'rel err':>9} {'FD raw':>14}")
    max_rel = 0.0
    for idx in top.tolist():
        orig = flat[idx].item()
        vals_fro, vals_raw = [], []
        for s in (+eps, -eps):
            flat[idx] = orig + s
            vals_fro.append(mesh_ks_frozen(verts_b, faces_b, d_vec, gs, keep, w, rho).item())
            vals_raw.append(undercut_penalty(
                verts_b, faces_b, d_draw, surface="cavity", formulation="ks_margin",
                ks_rho=rho, silhouette=sil)[0].item())
        flat[idx] = orig
        fd_fro = (vals_fro[0] - vals_fro[1]) / (2 * eps)
        fd_raw = (vals_raw[0] - vals_raw[1]) / (2 * eps)
        rel = abs(g[idx].item() - fd_fro) / max(abs(fd_fro), 1e-30)
        max_rel = max(max_rel, rel)
        print(f"  {idx:>7} {g[idx].item():>14.6e} {fd_fro:>14.6e} {100 * rel:>8.3f}% {fd_raw:>14.6e}")
    check("mesh frozen-FD gate (max rel err < 0.1%)", max_rel < 1e-3, f"max={100 * max_rel:.4f}%")


# ---------------------------------------------------------------------------
# Part 3: sdf KS margin on the real lattice
# ---------------------------------------------------------------------------

def part3_sdf(args) -> None:
    print("\n=== Part 3: sdf ks_margin (real lattice) ===")
    import deepshapeopt.config as dso_config
    from deepshapeopt.config import ExperimentSpecifications
    from deepshapeopt.reconstruction import with_float32_lattice
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

    uc = opt_cfg.get("no_undercut") or {}
    draw_dir = uc.get("draw_direction", [-1.0, 0.0, 0.0])
    threshold = -math.sin(math.radians(float(uc.get("draft_angle_deg", 0.0))))
    excl_deg = float(uc.get("exclude_axial_deg", 30.0))
    grid_spacing = float(uc.get("grid_spacing", 0.5))
    band_factor = float(uc.get("band_factor", 1.5))
    excl_region = uc.get("exclude_region")
    rho = float(uc.get("ks_rho", 50.0))

    sil = None
    if args.with_silhouette:
        # Synthetic rectangle covering the central 50% (y,z) of the design domain in the
        # outlet plane (min-x face for draw dir -x, max-x otherwise) -- exercises the
        # scope gate on the real band without needing a hex mesh build.
        dd = np.asarray(rec_cfg["design_domain"], dtype=float)
        x_pl = dd[0][0] if float(draw_dir[0]) < 0 else dd[1][0]
        cy, cz = dd.mean(axis=0)[1], dd.mean(axis=0)[2]
        hy, hz = 0.25 * (dd[1][1] - dd[0][1]), 0.25 * (dd[1][2] - dd[0][2])
        quad = np.array([
            [x_pl, cy - hy, cz - hz], [x_pl, cy + hy, cz - hz],
            [x_pl, cy + hy, cz + hz], [x_pl, cy - hy, cz + hz],
        ])
        sil = build_outlet_silhouette(
            np.array([[quad[0], quad[1], quad[2]], [quad[0], quad[2], quad[3]]]), draw_dir)
        print(f"  synthetic silhouette at x={x_pl:g}, area={sil.area:.4e} mm^2")

    common = dict(
        exclude_axial_deg=excl_deg, grid_spacing=grid_spacing,
        band_factor=band_factor, exclude_region=excl_region,
        formulation="ks_margin", ks_rho=rho, silhouette=sil,
    )
    M0, g, n_band, n_uc, pts_phys, scalars, _ = undercut_penalty_sdf(
        ls, frame, param, draw_dir, threshold, collect_debug=True, **common)
    print(f"  base: M={M0.item():+.6e} (worst angle "
          f"{math.degrees(math.asin(max(-1.0, min(1.0, M0.item())))):+.2f} deg), "
          f"band_pts={n_band}, undercut_pts={n_uc}, |g|={g.norm().item():.3e}")

    # Frozen re-evaluation: base band points + base normalized weights, only the
    # level-set normals (ndotd_out) are recomputed -- exactly the function the
    # production autograd differentiates (weights/selection are detached there).
    device = param.device
    Xb = frame.to_norm(
        torch.as_tensor(pts_phys, dtype=param.dtype, device=device)).to(torch.float32)
    w = torch.as_tensor(scalars["ks_weight"], dtype=torch.float32, device=device)
    pos = w > 0
    logw = torch.log(w[pos])
    d_t = torch.as_tensor(draw_dir, dtype=torch.float32, device=device)
    d_t = d_t / d_t.norm()
    sp = float(frame.scale) * grid_spacing
    fd_step = 0.25 * sp
    exs = [torch.tensor(v, device=device) for v in
           ([fd_step, 0, 0], [0, fd_step, 0], [0, 0, fd_step])]

    def frozen():
        def _compute(_b):
            q = lambda x: ls(x).reshape(-1)
            gp = [(q(Xb + e) - q(Xb - e)) / (2 * fd_step) for e in exs]
            gnorm = torch.sqrt(gp[0] ** 2 + gp[1] ** 2 + gp[2] ** 2).clamp_min(1e-12)
            ndotd_out = -(gp[0] * d_t[0] + gp[1] * d_t[1] + gp[2] * d_t[2]) / gnorm
            return torch.logsumexp(rho * ndotd_out[pos] + logw, dim=0) / rho
        return with_float32_lattice(ls, frame.box_norm, _compute)

    M_fro = frozen()
    g_fro = torch.autograd.grad(M_fro, param)[0].reshape(-1)
    check("sdf frozen re-eval matches production value",
          abs(M_fro.item() - M0.item()) < 1e-3, f"diff={M_fro.item() - M0.item():.2e}")
    rel_g = (g_fro - g.reshape(-1)).norm().item() / max(g.norm().item(), 1e-30)
    check("sdf frozen autograd matches production gradient (<1%)",
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
            vals_fro.append(float(frozen().detach()))
            vals_raw.append(float(undercut_penalty_sdf(
                ls, frame, param, draw_dir, threshold, **common)[0]))
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
    # KS with rho ~ 50 is strongly curved (softmax reweighting): eps 1e-3 already shows
    # ~20% FD truncation error on single components, 2e-4 up to ~2.4%; 1e-4 lands well
    # below the 2% gate (float32 noise floor is still ~2 orders below the signal there).
    p.add_argument("--eps", type=float, default=1e-4)
    p.add_argument("--with-silhouette", action="store_true",
                   help="Part 3: also gate with a synthetic outlet-plane silhouette")
    args = p.parse_args()

    part1_silhouette()
    part2_mesh()
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

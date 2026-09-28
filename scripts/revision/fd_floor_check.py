"""Extended float64 FD-vs-AD sweep: locate the true floor (reviewer comment 7).

The response-letter table stops at h = 1e-5, where the float64 column reads
5.8e-5 and is still descending -- the floor is not yet visible. This script
repeats fd_check_geometry.py's experiment B (mesh-free sub-chain
f(d) = sum_k w_k s(x_k; d), same config, seed, reconstruction, sample points,
and the same 20 design-variable components) in float64 only, with the step
sweep extended down to 1e-9.

What to expect and how to read the result: the composed chain is piecewise
multilinear (ReLU decoder), so f(d) has kinks in design space wherever a
perturbation flips an activation at one of the 10^4 sample points. Central
differences across such crossings contribute an error that vanishes with h but
follows crossing statistics (~h^(1/2..1)) rather than the smooth h^2 law, so
the curve may keep falling without a flat truncation floor until it hits the
float64 round-off wall ~ eps64 |f| / h. The per-step log-log slope is printed
to identify the regimes. Two steps (1e-4, 1e-5) overlap with the archived
fd_check_geometry_float64.json as a consistency check.

Usage:
    uv run python scripts/revision/fd_floor_check.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import deepshapeopt.config as config
from deepshapeopt.config import ExperimentSpecifications
from deepshapeopt.shape_optimization import (
    build_lattice,
    run_reconstruction,
    setup_model_and_domain,
)

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "experiments/drag_cube/config_latent_cube.json"
SEED = 0
N_POINTS = 10_000
N_COMPONENTS = 20
STEPS = [1e-4, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7, 3e-8, 1e-8, 1e-9]


def _flat_param(lattice):
    (p,) = list(lattice.parametrization.parameters())
    return p


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--out-dir", type=Path, default=REPO / "revision_artifacts")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    specs = ExperimentSpecifications(str(args.config))
    rec_cfg, opt_cfg = specs["reconstruction"], specs["optimization"]
    experiment_path = args.config.resolve().parent
    paths = config.make_experiment_paths(
        experiment_path, results_name="revision_fdfloor",
        heavy_data_output_path=opt_cfg.get("heavy_data_output_path"))
    config.ensure_experiment_dirs(paths)

    model_setup = setup_model_and_domain(rec_cfg, paths.reconstruction)
    lattice_setup = build_lattice(rec_cfg, model_setup.model, model_setup.sdf,
                                  model_setup.box_norm)
    run_reconstruction(
        lattice_setup.lattice_struct, model_setup.mesh_norm, model_setup.box_norm,
        rec_cfg, paths.reconstruction, model_setup.model, model_setup.scaling,
        opt_cfg, debug=False)
    lattice = lattice_setup.lattice_struct
    box_norm = model_setup.box_norm
    device = box_norm.device

    # identical points and component subset to fd_check_geometry.py
    g = torch.Generator(device="cpu").manual_seed(SEED)
    lo, hi = box_norm[0].cpu(), box_norm[1].cpu()
    pts = (lo + (hi - lo) * torch.rand(N_POINTS, 3, generator=g)).to(device)
    w = torch.ones(N_POINTS, device=device) / N_POINTS

    p = _flat_param(lattice)
    n_total = p.numel()
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(n_total, size=min(N_COMPONENTS, n_total), replace=False))

    saved_p, saved_b = p.data.clone(), lattice.bounds.data.clone()
    saved_default = torch.get_default_dtype()
    decoder = lattice.microtile.model._decoder
    saved_decoder_dtype = next(decoder.parameters()).dtype
    try:
        torch.set_default_dtype(torch.float64)
        p.data = p.data.to(torch.float64)
        lattice.bounds.data = lattice.bounds.data.to(torch.float64)
        decoder.to(torch.float64)
        pts_d, w_d = pts.double(), w.double()
        base = p.data.clone()

        def f() -> torch.Tensor:
            return (lattice(pts_d).reshape(-1) * w_d).sum()

        p.grad = None
        val = f()
        (g_ad,) = torch.autograd.grad(val, p)
        g_ad = g_ad.detach().reshape(-1)[idx].cpu().numpy()
        f_val = float(val.item())
        print(f"|f| = {abs(f_val):.4e}; predicted round-off wall "
              f"eps64*|f|/h reaches the AD gradient scale near "
              f"h ~ {2.2e-16 * abs(f_val) / max(np.abs(g_ad)):.1e}")

        rows = []
        for h in STEPS:
            g_fd = np.zeros(len(idx))
            for k, j in enumerate(idx):
                for sign in (+1, -1):
                    p.data = base.clone()
                    p.data.reshape(-1)[j] += sign * h
                    with torch.no_grad():
                        g_fd[k] += sign * float(f().item())
                g_fd[k] /= 2 * h
            p.data = base.clone()
            rel_l2 = float(np.linalg.norm(g_fd - g_ad) / np.linalg.norm(g_ad))
            denom = float(np.max(np.abs(g_ad)))
            rows.append({"h": h, "rel_l2": rel_l2,
                         "max_rel_diff": float(np.abs(g_fd - g_ad).max() / denom)})
            slope = ""
            if len(rows) > 1:
                r0, r1 = rows[-2], rows[-1]
                s = (np.log(r1["rel_l2"]) - np.log(r0["rel_l2"])) / \
                    (np.log(r1["h"]) - np.log(r0["h"]))
                slope = f"   slope {s:+.2f}"
            print(f"    h={h:<8.0e} rel_L2 {rel_l2:.3e}   "
                  f"max {rows[-1]['max_rel_diff']:.3e}{slope}", flush=True)
    finally:
        torch.set_default_dtype(saved_default)
        p.data = saved_p
        lattice.bounds.data = saved_b
        decoder.to(saved_decoder_dtype)

    best = min(rows, key=lambda r: r["rel_l2"])
    print(f"\n  floor: rel_L2 {best['rel_l2']:.3e} at h={best['h']:.0e}")

    out = {"config": str(args.config.relative_to(REPO)),
           "n_design_variables": int(n_total), "n_points": N_POINTS,
           "components": idx.tolist(), "functional_f64": f_val,
           "grad_ad_f64_subset": g_ad.tolist(), "sweep": rows,
           "best_rel_l2": best["rel_l2"], "best_h": best["h"]}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "fd_floor_check.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out_dir}/fd_floor_check.json")


if __name__ == "__main__":
    main()

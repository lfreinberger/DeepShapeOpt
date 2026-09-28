"""Direct float32-vs-float64 comparison of the AD gradient (reviewer comment 7).

Companion to fd_check_geometry.py, which verifies the AD gradient against
central differences per precision. Finite differences cannot verify the float32
chain below ~eps32*|s|/h (the 5.5e-4 floor in the response-letter table is a
limitation of the ESTIMATOR, not of the gradient): the noise sits in the
float32 function values, so no choice of step size or difference-quotient
precision gets past it.

The reviewer's actual question -- does float32 on the neural side harm the
composed gradient? -- has a direct, h-free answer: evaluate the same AD
gradient of the mesh-free sub-chain

    f(d) = sum_k w_k s(x_k; d)        (identical to fd_check experiment B)

once in float32 (the production state) and once with the identical weights,
parameters and points cast to float64, and report ||g32 - g64|| / ||g64||.
Combined with the existing FD64-vs-AD64 agreement of 5.8e-5 (which certifies
the float64 gradient as reference), this closes the argument chain.

Setup (config, seed, reconstruction, sample points) is identical to
fd_check_geometry.py so the numbers are directly comparable with the
response-letter table.

Usage:
    uv run python scripts/revision/ad_precision_check.py
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
N_COMPONENTS = 20  # the subset reported in the fd_check table, for comparison


def _flat_param(lattice):
    (p,) = list(lattice.parametrization.parameters())
    return p


def _grad(lattice, pts: torch.Tensor, w: torch.Tensor) -> np.ndarray:
    p = _flat_param(lattice)
    p.grad = None
    val = (lattice(pts).reshape(-1) * w).sum()
    (g,) = torch.autograd.grad(val, p)
    return float(val.item()), g.detach().reshape(-1).double().cpu().numpy()


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
        experiment_path, results_name="revision_adcheck",
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

    # Identical sample points to fd_check_geometry.experiment_b: generated in
    # float32 (the production dtype), then embedded exactly into float64.
    g = torch.Generator(device="cpu").manual_seed(SEED)
    lo, hi = box_norm[0].cpu(), box_norm[1].cpu()
    pts32 = (lo + (hi - lo) * torch.rand(N_POINTS, 3, generator=g)).to(device)
    w32 = torch.ones(N_POINTS, device=device) / N_POINTS

    # -- float32 gradient (the production state)
    f32, g32 = _grad(lattice, pts32, w32)

    # -- float64 gradient: identical values, higher working precision.
    # Same casting pattern as fd_check_geometry.experiment_b; the decoder is a
    # plain object, not a registered submodule, and must be cast separately.
    p = _flat_param(lattice)
    saved_p, saved_b = p.data.clone(), lattice.bounds.data.clone()
    saved_default = torch.get_default_dtype()
    decoder = lattice.microtile.model._decoder
    saved_decoder_dtype = next(decoder.parameters()).dtype
    try:
        torch.set_default_dtype(torch.float64)
        p.data = p.data.to(torch.float64)
        lattice.bounds.data = lattice.bounds.data.to(torch.float64)
        decoder.to(torch.float64)
        f64, g64 = _grad(lattice, pts32.double(), w32.double())
    finally:
        torch.set_default_dtype(saved_default)
        p.data = saved_p
        lattice.bounds.data = saved_b
        decoder.to(saved_decoder_dtype)

    n_total = g64.size
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(n_total, size=min(N_COMPONENTS, n_total), replace=False))

    diff = g32 - g64
    denom = np.max(np.abs(g64)) or 1.0
    cos = float(np.dot(g32, g64) / (np.linalg.norm(g32) * np.linalg.norm(g64)))
    out = {
        "config": str(args.config.relative_to(REPO)),
        "n_design_variables": int(n_total),
        "n_points": N_POINTS,
        "functional_f32": f32,
        "functional_f64": f64,
        "functional_rel_diff": abs(f32 - f64) / abs(f64),
        "rel_l2": float(np.linalg.norm(diff) / np.linalg.norm(g64)),
        "max_rel_diff": float(np.abs(diff).max() / denom),
        "mean_rel_diff": float(np.abs(diff).mean() / denom),
        "cosine_similarity": cos,
        "norm_g64": float(np.linalg.norm(g64)),
        "subset_components": idx.tolist(),
        "subset_rel_l2": float(np.linalg.norm(diff[idx]) / np.linalg.norm(g64[idx])),
        "grad_f32": g32.tolist(),
        "grad_f64": g64.tolist(),
    }

    print("\n=== AD gradient, float32 vs float64 (mesh-free sub-chain d -> lambda -> s) ===")
    print(f"  design variables:            {n_total} (full gradient compared)")
    print(f"  functional value rel. diff:  {out['functional_rel_diff']:.3e}")
    print(f"  rel_l2  ||g32-g64||/||g64||: {out['rel_l2']:.3e}")
    print(f"  max component (vs max |g64|): {out['max_rel_diff']:.3e}")
    print(f"  cosine similarity:           {1.0 - cos:.3e} below 1")
    print(f"  20-component subset rel_l2:  {out['subset_rel_l2']:.3e} "
          "(the subset of the fd_check table)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "ad_precision_check.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out_dir}/ad_precision_check.json")


if __name__ == "__main__":
    main()

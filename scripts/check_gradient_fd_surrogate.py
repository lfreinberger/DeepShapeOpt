"""Finite-difference check of the surrogate gradient chain.

Compares the end-to-end autograd gradient dJ/dz (Transolver fields -> drag
integral -> differentiable snap -> latent code) against central finite
differences of the surrogate objective itself. Each FD evaluation is a
surrogate call (~1 s), so many components are affordable. The castellation
is frozen (``build(reuse_castellation=True)``) so J(z) is smooth.

This validates the *plumbing* (query cloud, drag integral, snap VJP) even
with a mediocre model; run it once per trained checkpoint.

Usage:
    uv run python scripts/check_gradient_fd_surrogate.py \
        --config tests/drag_latent_surrogate/config.json --n-components 20
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

import deepshapeopt.config as config
from deepshapeopt.config import ExperimentSpecifications
from deepshapeopt.hexmesh import SdfHexMeshPipeline
from deepshapeopt.runtime import configure_logging
from deepshapeopt.shape_optimization import (
    build_lattice,
    run_reconstruction,
    setup_model_and_domain,
)
from deepshapeopt.surrogate.predictor import TransolverSurrogate

LOGGER = logging.getLogger(__name__)


def objective(hex_pipeline, surrogate, reuse_castellation: bool, need_grad: bool):
    result = hex_pipeline.build(reuse_castellation=reuse_castellation)
    ctx = torch.enable_grad() if need_grad else torch.no_grad()
    with ctx:
        J, _ = surrogate.objective(
            result.surface_points, result.wall_tris_local, hex_pipeline.sdf_at_phys,
            latent_fn=hex_pipeline.latent_at_phys,
        )
    return J


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--n-components", type=int, default=20)
    parser.add_argument("--eps", type=float, default=2e-3)
    args = parser.parse_args()

    experiment_path = Path(args.config).resolve().parent
    specs = ExperimentSpecifications(args.config)
    rec_cfg, opt_cfg = specs["reconstruction"], specs["optimization"]
    if opt_cfg.get("mesh_pipeline") != "sdf_hex":
        raise SystemExit("requires mesh_pipeline 'sdf_hex'")

    paths = config.make_experiment_paths(
        experiment_path, results_name=specs.get("results_name", "results")
    )
    config.ensure_experiment_dirs(paths)
    configure_logging(False, paths.optimization / "fd_check_surrogate.log")

    model_setup = setup_model_and_domain(rec_cfg, paths.reconstruction)
    lattice = build_lattice(rec_cfg, model_setup.model, model_setup.sdf, model_setup.frame)
    run_reconstruction(
        lattice.lattice_struct, model_setup.frame, model_setup.mesh_orig,
        rec_cfg, paths.reconstruction, model_setup.model, opt_cfg,
    )
    param = next(lattice.lattice_struct.parametrization.parameters())
    hex_pipeline = SdfHexMeshPipeline(
        lattice.lattice_struct, model_setup, opt_cfg, paths.optimization
    )
    surrogate = TransolverSurrogate.from_config(opt_cfg["surrogate"])

    # Reference evaluation with gradient (fresh castellation).
    J0 = objective(hex_pipeline, surrogate, reuse_castellation=False, need_grad=True)
    (dJ,) = torch.autograd.grad(J0, param)
    dJ_flat = dJ.reshape(-1)
    print(f"J0 = {float(J0):.6f}, |dJ| = {float(dJ_flat.norm()):.4e}")

    order = torch.argsort(dJ_flat.abs(), descending=True)[: args.n_components]
    rows, fd_vals, ad_vals = [], [], []
    for k, idx in enumerate(order.tolist()):
        with torch.no_grad():
            orig = float(param.reshape(-1)[idx])
            param.reshape(-1)[idx] = orig + args.eps
        J_plus = float(objective(hex_pipeline, surrogate, True, False))
        with torch.no_grad():
            param.reshape(-1)[idx] = orig - args.eps
        J_minus = float(objective(hex_pipeline, surrogate, True, False))
        with torch.no_grad():
            param.reshape(-1)[idx] = orig

        fd = (J_plus - J_minus) / (2 * args.eps)
        ad = float(dJ_flat[idx])
        rel = abs(fd - ad) / max(abs(fd), abs(ad), 1e-12)
        fd_vals.append(fd)
        ad_vals.append(ad)
        print(f"[{k:2d}] idx={idx:4d}  autograd={ad:+.5e}  FD={fd:+.5e}  rel={100 * rel:6.2f}%")
        rows.append(rel)

    fd_t, ad_t = torch.tensor(fd_vals), torch.tensor(ad_vals)
    cos = float((fd_t @ ad_t) / (fd_t.norm() * ad_t.norm()).clamp_min(1e-12))
    med = float(torch.tensor(rows).median())
    print(f"\ncosine(FD, autograd) over {len(rows)} components: {cos:.5f}")
    print(f"median rel. error: {100 * med:.2f}%")
    if cos < 0.99:
        raise SystemExit("FD gate FAILED (cosine < 0.99)")
    print("FD gate PASSED")


if __name__ == "__main__":
    main()

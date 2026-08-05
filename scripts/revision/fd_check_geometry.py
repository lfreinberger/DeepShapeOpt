"""Gradient verification and the float32/float64 question (reviewer comment 7).

The reviewer asks how much the float32 neural side versus float64 OpenFOAM side
matters for the composed gradient of Eq. (21) and for the convergence criterion
eps_D = 2e-4.

Two experiments, deliberately excluding CFD:

(A) FULL GEOMETRY CHAIN, float32.
    f(d) = V(d), the body volume, which traverses the entire differentiable
    geometry path d -> lambda(x) -> s -> Gamma (FlexiCubes) -> V. This is the
    same quantity and the same autograd call the production loop already
    differentiates (optimize_drag_latent.py:187). Central differences on a fixed
    random subset of design variables, swept over step size.

    Topology guard: FlexiCubes changes vertex count when a perturbation moves
    the zero level set across a grid cell, which makes V(d) non-smooth and can
    masquerade as a precision floor. Vertex counts are recorded for every
    evaluation and any step that changes them is flagged.

(B) MESH-FREE SUB-CHAIN, float32 vs float64.
    f(d) = sum_k w_k s(x_k; d) at fixed sample points -- spline + decoder only,
    no meshing. This isolates arithmetic precision cleanly and, unlike (A), can
    be run in float64: the production path forces float32 through
    with_float32_lattice (reconstruction.py:340) because FlexiCubes requires it,
    but the sub-chain has no such constraint.

Interpretation for eps_D (state this explicitly in the rebuttal): eps_D bounds
obj_change = |J_k - J_{k-1}| / |J_0| (optimize_drag_latent.py:265-268), a
criterion on the OBJECTIVE, not on the gradient. J is read from OpenFOAM ASCII
at writePrecision 6 (controlDict:37), i.e. a relative representation error of
order 1e-6 -- two orders of magnitude below eps_D = 2e-4. So ASCII precision is
not the binding constraint; the discretization uncertainty of J is, and that is
what scripts/revision/mesh_study.py quantifies. Reviewer comments 7 and 8 share
one deliverable.

Usage:
    uv run python scripts/revision/fd_check_geometry.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import deepshapeopt.config as config
from deepshapeopt.config import ExperimentSpecifications
from deepshapeopt.mesh import compute_tet_mesh_volume_centroid
from deepshapeopt.shape_optimization import (
    build_lattice,
    generate_mesh,
    run_reconstruction,
    setup_model_and_domain,
)

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "experiments/drag_cube/config_latent_cube.json"
STEPS = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 1e-5]
N_COMPONENTS = 20
SEED = 0


def _discrepancy(g_fd: np.ndarray, g_ad: np.ndarray) -> dict:
    """Agreement between a finite-difference and an analytic gradient.

    Several statistics, because a single one is easy to misread:

    * ``max_rel_diff``  -- worst component, normalized by the largest analytic
      component. Conservative, but dominated by whichever component is worst.
    * ``mean_rel_diff`` / ``median_rel_diff`` -- same normalization, averaged.
      Less sensitive to a single outlier.
    * ``rel_l2``        -- ||g_fd - g_ad||_2 / ||g_ad||_2, the usual relative
      error of the gradient vector. Not dominated by one component and not
      sensitive to near-zero components, so this is the fairest summary.

    A single scalar denominator is used throughout rather than per-component
    normalization, which would blow up wherever an analytic component is near
    zero. The raw per-component differences are stored so that any other
    statistic can be formed later without re-running.
    """
    diff = np.abs(g_fd - g_ad)
    denom = float(np.max(np.abs(g_ad))) or 1.0
    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "median_abs_diff": float(np.median(diff)),
        "max_rel_diff": float(diff.max() / denom),
        "mean_rel_diff": float(diff.mean() / denom),
        "median_rel_diff": float(np.median(diff) / denom),
        "rel_l2": float(np.linalg.norm(g_fd - g_ad) / np.linalg.norm(g_ad)),
        "max_abs_grad": denom,
        "per_component_abs_diff": diff.tolist(),
        "grad_analytic": g_ad.tolist(),
    }


def _flat_param(lattice):
    (p,) = list(lattice.parametrization.parameters())
    return p


def volume_of(lattice, opt_cfg, rec_cfg, box_norm, scaling) -> tuple[torch.Tensor, int]:
    mesh, _ = generate_mesh(lattice, opt_cfg, rec_cfg, box_norm, scaling,
                            mesh_type="volume", extend_bounds=True)
    vol, _ = compute_tet_mesh_volume_centroid(mesh.vertices, mesh.volumes)
    return vol, int(mesh.vertices.shape[0])


def experiment_a(lattice, opt_cfg, rec_cfg, box_norm, scaling, idx) -> dict:
    """Full geometry chain, float32: autograd vs central differences."""
    p = _flat_param(lattice)
    base = p.data.clone()

    p.grad = None
    vol, n_ref = volume_of(lattice, opt_cfg, rec_cfg, box_norm, scaling)
    (g_ad,) = torch.autograd.grad(vol, p)
    g_ad = g_ad.detach().reshape(-1)[idx].cpu().numpy()

    rows = []
    for h in STEPS:
        g_fd, topo = np.zeros(len(idx)), 0
        for k, j in enumerate(idx):
            for sign in (+1, -1):
                p.data = base.clone()
                p.data.reshape(-1)[j] += sign * h
                with torch.no_grad():
                    v, n = volume_of(lattice, opt_cfg, rec_cfg, box_norm, scaling)
                topo += int(n != n_ref)
                g_fd[k] += sign * float(v.item())
            g_fd[k] /= 2 * h
        p.data = base.clone()

        rows.append({"h": h, "topology_changes": topo, "n_evals": 2 * len(idx),
                     **_discrepancy(g_fd, g_ad)})
        r = rows[-1]
        print(f"    h={h:<8.0e} max {r['max_rel_diff']:.3e}   mean {r['mean_rel_diff']:.3e}"
              f"   rel_L2 {r['rel_l2']:.3e}   topology changes {topo}/{2 * len(idx)}", flush=True)

    return {"reference_vertex_count": n_ref, "grad_autograd_subset": g_ad.tolist(),
            "sweep": rows}


def experiment_b(lattice, box_norm, idx, dtype: torch.dtype, n_points=10_000) -> dict:
    """Mesh-free sub-chain d -> lambda -> s at fixed points, in the given dtype."""
    device = box_norm.device
    g = torch.Generator(device="cpu").manual_seed(SEED)
    lo, hi = box_norm[0].cpu(), box_norm[1].cpu()
    pts = (lo + (hi - lo) * torch.rand(n_points, 3, generator=g)).to(device=device)
    w = torch.ones(n_points, device=device) / n_points

    p = _flat_param(lattice)
    saved_p, saved_b = p.data.clone(), lattice.bounds.data.clone()
    saved_default = torch.get_default_dtype()

    # The DECODER WEIGHTS must be cast too. DeepSDFModel is a plain class, not an
    # nn.Module (deep_sdf/models.py), so the decoder is not a registered submodule
    # of the lattice and is not reached by casting the lattice parameters. Missing
    # this fails with "mat1 and mat2 must have the same dtype, but got Double and
    # Float" as soon as the first linear layer is hit.
    decoder = lattice.microtile.model._decoder
    saved_decoder_dtype = next(decoder.parameters()).dtype

    try:
        torch.set_default_dtype(dtype)
        p.data = p.data.to(dtype)
        lattice.bounds.data = lattice.bounds.data.to(dtype)
        decoder.to(dtype)
        pts_d, w_d = pts.to(dtype), w.to(dtype)
        base = p.data.clone()

        def f() -> torch.Tensor:
            return (lattice(pts_d).reshape(-1) * w_d).sum()

        p.grad = None
        val = f()
        (g_ad,) = torch.autograd.grad(val, p)
        g_ad = g_ad.detach().reshape(-1)[idx].double().cpu().numpy()

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
            rows.append({"h": h, **_discrepancy(g_fd, g_ad)})
            r = rows[-1]
            print(f"    h={h:<8.0e} max {r['max_rel_diff']:.3e}   mean {r['mean_rel_diff']:.3e}"
                  f"   rel_L2 {r['rel_l2']:.3e}", flush=True)
        best = min(rows, key=lambda r: r["max_rel_diff"])
        best_l2 = min(rows, key=lambda r: r["rel_l2"])
        return {"dtype": str(dtype), "n_points": n_points, "sweep": rows,
                "best_rel_diff": best["max_rel_diff"], "best_h": best["h"],
                "best_rel_l2": best_l2["rel_l2"], "best_h_l2": best_l2["h"]}
    finally:
        torch.set_default_dtype(saved_default)
        p.data = saved_p
        lattice.bounds.data = saved_b
        decoder.to(saved_decoder_dtype)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--n-components", type=int, default=N_COMPONENTS)
    parser.add_argument("--out-dir", type=Path, default=REPO / "revision_artifacts")
    parser.add_argument("--skip-a", action="store_true",
                        help="skip the full-chain experiment (it is the expensive one)")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    specs = ExperimentSpecifications(str(args.config))
    rec_cfg, opt_cfg = specs["reconstruction"], specs["optimization"]
    experiment_path = args.config.resolve().parent
    paths = config.make_experiment_paths(
        experiment_path, results_name="revision_fdcheck",
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

    n_total = _flat_param(lattice).numel()
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(n_total, size=min(args.n_components, n_total), replace=False))
    print(f"design variables: {n_total}; probing {len(idx)} random components\n")

    out: dict = {"config": str(args.config.relative_to(REPO)),
                 "n_design_variables": int(n_total),
                 "components": idx.tolist(), "steps": STEPS}

    if not args.skip_a:
        print("=== (A) full geometry chain d -> lambda -> s -> Gamma -> V, float32 ===")
        out["experiment_a_full_chain_float32"] = experiment_a(
            lattice, opt_cfg, rec_cfg, model_setup.box_norm, model_setup.scaling, idx)

    print("\n=== (B) mesh-free sub-chain d -> lambda -> s, float32 ===")
    out["experiment_b_subchain_float32"] = experiment_b(
        lattice, model_setup.box_norm, idx, torch.float32)

    print("\n=== (B) mesh-free sub-chain d -> lambda -> s, float64 ===")
    try:
        out["experiment_b_subchain_float64"] = experiment_b(
            lattice, model_setup.box_norm, idx, torch.float64)
    except Exception as exc:  # noqa: BLE001 - record why rather than lose the float32 result
        print(f"    float64 arm failed: {exc}")
        out["experiment_b_subchain_float64"] = {"error": str(exc)}

    b32 = out["experiment_b_subchain_float32"]
    b64 = out.get("experiment_b_subchain_float64", {})
    print("\n=== summary ===")
    print(f"  sub-chain float32 best agreement: {b32['best_rel_diff']:.3e} at h={b32['best_h']:.0e}")
    if "best_rel_diff" in b64:
        print(f"  sub-chain float64 best agreement: {b64['best_rel_diff']:.3e} "
              f"at h={b64['best_h']:.0e}")
    if "experiment_a_full_chain_float32" in out:
        a = out["experiment_a_full_chain_float32"]
        best = min(a["sweep"], key=lambda r: r["max_rel_diff"])
        print(f"  full chain float32 best agreement: {best['max_rel_diff']:.3e} "
              f"at h={best['h']:.0e}  (topology changes at that step: "
              f"{best['topology_changes']}/{best['n_evals']})")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "fd_check_geometry.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out_dir}/fd_check_geometry.json")


if __name__ == "__main__":
    main()

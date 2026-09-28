"""float32-vs-float64 AD comparison through the FULL geometry chain (comment 7).

Extends ad_precision_check.py (mesh-free sub-chain) to the complete
differentiable path

    d -> lambda(x) -> s -> Gamma (FlexiCubes) -> V,

i.e. including the differentiable surface extraction that the finite-difference
check must exclude. The FD exclusion argument (connectivity changes under
almost every perturbation) does NOT apply here: both precisions are evaluated
at the SAME design point, so if the extracted connectivity agrees, gradients
and vertex positions are comparable elementwise.

The one failure mode is checked explicitly: connectivity is decided by the
sign of s at the voxel-grid corners, and a corner with |s| below the float32
rounding scale (~1e-7) could flip between precisions. The script reports the
minimum |s| over all grid corners and the number of sign disagreements; with
~1e6 corners and O(1) SDF range, zero flips are expected but not guaranteed.

Setup (config, seed, reconstruction) is identical to ad_precision_check.py /
fd_check_geometry.py. The float32 arm uses the production path
(generate_mesh -> with_float32_lattice); the float64 arm calls create_3D_mesh
directly under a float64 context (default dtype + decoder + lattice + scaling
casts) -- the FlexiCubes fork follows torch.get_default_dtype() internally.

Usage:
    uv run python scripts/revision/ad_precision_check_fullchain.py
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
from DeepSDFStruct.mesh import (
    _prepare_flexicubes_querypoints,
    create_3D_mesh,
    process_N_base_input,
)

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "experiments/drag_cube/config_latent_cube.json"
SEED = 0


def _flat_param(lattice):
    (p,) = list(lattice.parametrization.parameters())
    return p


def _grid_signs(lattice, box_norm, resolution, device):
    """SDF values at the FlexiCubes voxel-grid corners (current dtype)."""
    bounds = box_norm.clone()
    off = (bounds[1] - bounds[0]) * 0.05
    bounds[0] -= off
    bounds[1] += off
    n = process_N_base_input(resolution, torch.tensor(lattice.tiling))
    _, samples, _ = _prepare_flexicubes_querypoints(n, device=device, bounds=bounds)
    with torch.no_grad():
        s = lattice(samples).reshape(-1)
    return s


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--resolution", type=int, default=None,
                        help="override opt_cfg mesh_resolution (both arms)")
    parser.add_argument("--out-dir", type=Path, default=REPO / "revision_artifacts")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    specs = ExperimentSpecifications(str(args.config))
    rec_cfg, opt_cfg = specs["reconstruction"], specs["optimization"]
    if args.resolution is not None:
        opt_cfg = dict(opt_cfg)
        opt_cfg["mesh_resolution"] = args.resolution
    experiment_path = args.config.resolve().parent
    paths = config.make_experiment_paths(
        experiment_path, results_name="revision_adfull",
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
    scaling = model_setup.scaling
    device = rec_cfg["device"]
    res = opt_cfg["mesh_resolution"]

    p = _flat_param(lattice)

    # ---- float32 arm: the production path -----------------------------
    p.grad = None
    mesh32, _ = generate_mesh(lattice, opt_cfg, rec_cfg, box_norm, scaling,
                              mesh_type="volume", extend_bounds=True)
    vol32, _ = compute_tet_mesh_volume_centroid(mesh32.vertices, mesh32.volumes)
    (g32,) = torch.autograd.grad(vol32, p)
    g32 = g32.detach().reshape(-1).double().cpu().numpy()
    v32 = mesh32.vertices.detach().double().cpu()
    t32 = mesh32.volumes.detach().cpu()
    s32 = _grid_signs(lattice, box_norm, res, device).detach().double().cpu()
    vol32_f = float(vol32.item())
    print(f"float32 arm: V = {vol32_f:.10e}, {v32.shape[0]} vertices, "
          f"{t32.shape[0]} tets")

    # ---- float64 arm ---------------------------------------------------
    saved_p, saved_b = p.data.clone(), lattice.bounds.data.clone()
    saved_default = torch.get_default_dtype()
    decoder = lattice.microtile.model._decoder
    saved_decoder_dtype = next(decoder.parameters()).dtype
    try:
        torch.set_default_dtype(torch.float64)
        p.data = p.data.to(torch.float64)
        lattice.bounds.data = lattice.bounds.data.to(torch.float64)
        decoder.to(torch.float64)
        scaling.double()
        box64 = box_norm.double()

        p.grad = None
        mesh64, _ = create_3D_mesh(
            lattice, res, mesh_type="volume", differentiate=False,
            device=device, bounds=box64, deformation_function=scaling,
            extend_bounds=True)
        vol64, _ = compute_tet_mesh_volume_centroid(mesh64.vertices, mesh64.volumes)
        (g64,) = torch.autograd.grad(vol64, p)
        g64 = g64.detach().reshape(-1).cpu().numpy()
        v64 = mesh64.vertices.detach().cpu()
        t64 = mesh64.volumes.detach().cpu()
        s64 = _grid_signs(lattice, box_norm.double(), res, device).detach().cpu()
        vol64_f = float(vol64.item())
    finally:
        torch.set_default_dtype(saved_default)
        p.data = saved_p
        lattice.bounds.data = saved_b
        decoder.to(saved_decoder_dtype)
        scaling.float()
    print(f"float64 arm: V = {vol64_f:.10e}, {v64.shape[0]} vertices, "
          f"{t64.shape[0]} tets")

    # ---- connectivity and occupancy checks -----------------------------
    same_counts = (v32.shape[0] == v64.shape[0]) and (t32.shape[0] == t64.shape[0])
    same_connectivity = bool(same_counts and torch.equal(t32, t64))
    sign_flips = int((torch.signbit(s32) != torch.signbit(s64)).sum().item())
    min_abs_s = float(s64.abs().min().item())

    out = {
        "config": str(args.config.relative_to(REPO)),
        "mesh_resolution": res,
        "n_design_variables": int(p.numel()),
        "n_grid_corners": int(s64.numel()),
        "min_abs_s_at_corners": min_abs_s,
        "occupancy_sign_flips": sign_flips,
        "n_vertices": [int(v32.shape[0]), int(v64.shape[0])],
        "n_tets": [int(t32.shape[0]), int(t64.shape[0])],
        "same_connectivity": same_connectivity,
        "volume_f32": vol32_f,
        "volume_f64": vol64_f,
        "volume_rel_diff": abs(vol32_f - vol64_f) / abs(vol64_f),
    }

    print("\n=== full chain d -> lambda -> s -> Gamma -> V, float32 vs float64 ===")
    print(f"  grid corners: {out['n_grid_corners']}, min |s| = {min_abs_s:.3e}, "
          f"occupancy sign flips: {sign_flips}")
    print(f"  connectivity identical: {same_connectivity} "
          f"(verts {out['n_vertices']}, tets {out['n_tets']})")
    print(f"  volume rel. diff:            {out['volume_rel_diff']:.3e}")

    if same_connectivity:
        diff = g32 - g64
        vdiff = (v32 - v64).abs()
        bbox = float((box_norm[1] - box_norm[0]).max().item())
        out.update({
            "grad_rel_l2": float(np.linalg.norm(diff) / np.linalg.norm(g64)),
            "grad_max_rel_diff": float(np.abs(diff).max() / np.max(np.abs(g64))),
            "grad_cosine_similarity": float(
                np.dot(g32, g64) / (np.linalg.norm(g32) * np.linalg.norm(g64))),
            "vertex_max_abs_diff": float(vdiff.max().item()),
            "vertex_max_rel_bbox": float(vdiff.max().item() / bbox),
            "grad_f32": g32.tolist(),
            "grad_f64": g64.tolist(),
        })
        print(f"  gradient rel_l2 ||g32-g64||/||g64||: {out['grad_rel_l2']:.3e}")
        print(f"  gradient max component:      {out['grad_max_rel_diff']:.3e}")
        print(f"  vertex positions, max |dv|:  {out['vertex_max_abs_diff']:.3e} "
              f"({out['vertex_max_rel_bbox']:.3e} of bbox)")
    else:
        print("  connectivity differs -> elementwise gradient/vertex comparison "
              "skipped; only the scalar volume is compared")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "ad_precision_check_fullchain.json").write_text(
        json.dumps(out, indent=2))
    print(f"\nwrote {args.out_dir}/ad_precision_check_fullchain.json")


if __name__ == "__main__":
    main()

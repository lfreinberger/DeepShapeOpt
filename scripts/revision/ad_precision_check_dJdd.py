"""float32-vs-float64 comparison of the COMPOSED gradient dJ/dd (comment 7).

The most direct answer to the reviewer's question about "the accuracy of the
composed gradient in Equation (21)": Eq. (21) contracts the float64 OpenFOAM
surface sensitivity dJ/dGamma with the float32 neural vertex Jacobian
dGamma/dd. This script evaluates exactly that contraction -- the quantity MMA
receives -- in both precisions, with the adjoint seed held FIXED:

  1. fit + surface mesh in float32 (production path), export STL,
  2. ONE local OpenFOAM run (primal + adjoint, adjointOptimisationFoam) on
     that surface; load the integrated vertex sensitivities s = dJ/dGamma
     (loading_method "conservative", as in production),
  3. dJ/dd = s^T (dGamma/dd) via the production VJP
     (foam_utils.compute_shape_gradient) once on the float32 mesh graph and
     once on the float64 mesh graph, with the SAME seed vector s (float64
     external data, matched by vertex index -- valid because the extracted
     connectivity is identical between precisions, which is verified).

Holding the seed fixed isolates the neural side: re-running the CFD per
precision would bury the ~1e-5 effect under adjoint solver noise (the adjoint
terminates on max iterations, not on its residual tolerance).

Setup identical to ad_precision_check_fullchain.py (config, seed,
reconstruction). Requires a local OpenFOAM environment (WM_PROJECT_DIR);
runs WITHOUT Slurm via foam_utils.run_openfoam_case.

Usage:
    uv run python scripts/revision/ad_precision_check_dJdd.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import deepshapeopt.config as config
import deepshapeopt.foam_utils as foam_utils
from deepshapeopt.config import ExperimentSpecifications
from deepshapeopt.shape_optimization import (
    build_lattice,
    generate_mesh,
    load_sensitivities,
    run_foam_case,
    run_reconstruction,
    setup_model_and_domain,
)
from DeepSDFStruct.mesh import create_3D_mesh

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "experiments/drag_cube/config_latent_cube.json"
TEMPLATE = REPO / "experiments/drag_cube/foam_case"
SEED = 0


def _flat_param(lattice):
    (p,) = list(lattice.parametrization.parameters())
    return p


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--case-dir", type=Path, default=None,
                        help="reuse an existing, already-run foam case "
                             "(skips the OpenFOAM run)")
    parser.add_argument("--out-dir", type=Path, default=REPO / "revision_artifacts")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    specs = ExperimentSpecifications(str(args.config))
    rec_cfg, opt_cfg = specs["reconstruction"], specs["optimization"]
    sens_cfg = opt_cfg.get("sensitivity", {})
    experiment_path = args.config.resolve().parent
    paths = config.make_experiment_paths(
        experiment_path, results_name="revision_dJdd",
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
    p = _flat_param(lattice)

    # ---- float32 arm: production surface mesh (graph attached to p) ----
    mesh32, deriv32 = generate_mesh(lattice, opt_cfg, rec_cfg, box_norm, scaling,
                                    mesh_type="surface", extend_bounds=True)
    verts32, faces32 = mesh32.vertices, mesh32.faces
    print(f"float32 surface: {verts32.shape[0]} vertices, {faces32.shape[0]} faces")

    # ---- one OpenFOAM run on the float32 surface (or reuse an existing one)
    if args.case_dir is None:
        case_dir = foam_utils.prepare_foam_runtime(TEMPLATE, "adprecision")
        (case_dir / "constant" / "triSurface").mkdir(parents=True, exist_ok=True)
        print(f"running OpenFOAM locally in {case_dir} ...")
        foam_case = run_foam_case(case_dir, mesh32, deriv32, paths.optimization,
                                  plot_residuals=False)
    else:
        case_dir = args.case_dir
        from foamlib import FoamCase
        foam_case = FoamCase(case_dir)
        print(f"reusing foam case {case_dir}")

    loading_method = sens_cfg.get("loading_method", "interpolate")
    sens_on_orig, J_raw = load_sensitivities(
        case_dir, foam_case, verts32,
        field_name=sens_cfg.get("field_name", "pointSensVecadjS1ESI"),
        objective_path=sens_cfg.get("objective_path",
                                    "optimisation/objective/0/dragadjS1"),
        loading_method=loading_method,
        patch_name=sens_cfg.get("patch_name", "dragObject"),
        faces=faces32,
    )
    sens_fixed = np.asarray(
        sens_on_orig.detach().cpu().numpy()
        if torch.is_tensor(sens_on_orig) else sens_on_orig, dtype=np.float64)
    integrated = loading_method == "conservative"
    invert_normals = sens_cfg.get("invert_normals", True)
    print(f"J_raw = {float(np.atleast_1d(J_raw).ravel()[0]):.6e}, "
          f"seed |s| range [{np.abs(sens_fixed).min():.3e}, "
          f"{np.abs(sens_fixed).max():.3e}], method={loading_method}")

    dJ32 = foam_utils.compute_shape_gradient(
        p, verts32, faces32, sens_fixed,
        invert_normals=invert_normals, integrated=integrated)
    dJ32 = dJ32.detach().reshape(-1).double().cpu().numpy()

    # ---- float64 arm: same seed, same design point, float64 chain ---------
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

        mesh64, _ = create_3D_mesh(
            lattice, opt_cfg["mesh_resolution"], mesh_type="surface",
            differentiate=False, device=device, bounds=box_norm.double(),
            deformation_function=scaling, extend_bounds=True)
        verts64, faces64 = mesh64.vertices, mesh64.faces
        same_connectivity = bool(
            verts64.shape[0] == verts32.shape[0]
            and torch.equal(faces64.detach().cpu(), faces32.detach().cpu()))
        print(f"float64 surface: {verts64.shape[0]} vertices; "
              f"connectivity identical: {same_connectivity}")
        dJ64 = None
        if same_connectivity:
            dJ64 = foam_utils.compute_shape_gradient(
                p, verts64, faces64, sens_fixed,
                invert_normals=invert_normals, integrated=integrated)
            dJ64 = dJ64.detach().reshape(-1).cpu().numpy()
        vdiff_max = float((verts64.detach().cpu()
                           - verts32.detach().double().cpu()).abs().max().item())
    finally:
        torch.set_default_dtype(saved_default)
        p.data = saved_p
        lattice.bounds.data = saved_b
        decoder.to(saved_decoder_dtype)
        scaling.float()

    out = {
        "config": str(args.config.relative_to(REPO)),
        "n_design_variables": int(p.numel()),
        "n_vertices": [int(verts32.shape[0]), int(verts64.shape[0])],
        "same_connectivity": same_connectivity,
        "vertex_max_abs_diff": vdiff_max,
        "J_raw": float(np.atleast_1d(J_raw).ravel()[0]),
        "loading_method": loading_method,
        "case_dir": str(case_dir),
    }
    print("\n=== composed gradient dJ/dd (Eq. 21), float32 vs float64, "
          "fixed adjoint seed ===")
    if dJ64 is not None:
        diff = dJ32 - dJ64
        out.update({
            "dJ_rel_l2": float(np.linalg.norm(diff) / np.linalg.norm(dJ64)),
            "dJ_max_rel_diff": float(np.abs(diff).max() / np.max(np.abs(dJ64))),
            "dJ_cosine_similarity": float(
                np.dot(dJ32, dJ64) / (np.linalg.norm(dJ32) * np.linalg.norm(dJ64))),
            "norm_dJ64": float(np.linalg.norm(dJ64)),
            "dJ_f32": dJ32.tolist(),
            "dJ_f64": dJ64.tolist(),
        })
        print(f"  rel_l2 ||dJ32-dJ64||/||dJ64||: {out['dJ_rel_l2']:.3e}")
        print(f"  max component:                 {out['dJ_max_rel_diff']:.3e}")
        print(f"  cosine similarity:             {1.0 - out['dJ_cosine_similarity']:.3e} below 1")
        print(f"  vertex max |dv|:               {vdiff_max:.3e}")
    else:
        print("  connectivity differs between precisions -- comparison aborted")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "ad_precision_check_dJdd.json").write_text(
        json.dumps(out, indent=2))
    print(f"\nwrote {args.out_dir}/ad_precision_check_dJdd.json")


if __name__ == "__main__":
    main()

"""Drag optimization of a cube shape using DeepSDF + B-spline parametrization."""
import argparse
import math
import shutil
import time
from pathlib import Path

import torch
from foamlib import FoamCase

from DeepSDFStruct.export_knot_grid import (
    export_knot_grid_paramspace,
    export_control_lattice_paramspace,
    export_design_volume_paramspace,
)

import deepshapeopt.config as config
import deepshapeopt.foam_utils as foam_utils
from deepshapeopt.config import ExperimentSpecifications
from deepshapeopt.logging import OptimizationLogger
from deepshapeopt.mesh import compute_tet_mesh_volume_centroid
from deepshapeopt.plotting_utils import plot_optimization_history, plot_convergence_diagnostics, save_shape_snapshot
from deepshapeopt.shape_optimization import (
    DTYPE,
    setup_model_and_domain,
    build_lattice,
    run_reconstruction,
    setup_optimizer,
    generate_mesh,
    run_foam_case,
    load_sensitivities,
    mask_gradients,
    log_iteration_time,
)


def optimize_shape(experiment_path: Path, specs):
    rec_cfg = specs["reconstruction"]
    opt_cfg = specs["optimization"]
    sens_cfg = opt_cfg.get("sensitivity", {})

    results_name = specs.get("results_name", "results")
    paths = config.make_experiment_paths(
        experiment_path, results_name=results_name,
        heavy_data_output_path=opt_cfg.get("heavy_data_output_path"),
    )
    config.ensure_experiment_dirs(paths)
    if paths.heavy_data is not None:
        (paths.heavy_data / "optimization").mkdir(parents=True, exist_ok=True)
        for subdir in ("vtk_series", "stl_series"):
            d = paths.heavy_data / subdir
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)

    use_center_constraint = opt_cfg.get("use_center_constraint", False)

    # Phase 1: Model & domain
    model_setup = setup_model_and_domain(rec_cfg, paths.reconstruction)

    # Phase 2: Lattice
    lattice = build_lattice(rec_cfg, model_setup.model, model_setup.sdf, model_setup.box_norm)

    # Phase 3: Reconstruction
    run_reconstruction(
        lattice.lattice_struct, model_setup.mesh_norm, model_setup.box_norm,
        rec_cfg, paths.reconstruction, model_setup.model, model_setup.scaling, opt_cfg,
    )

    # Initial volume & centroid from tet mesh.
    # We compute this once and treat it as a constant baseline.
    with torch.no_grad():
        mesh_init, _ = generate_mesh(
            lattice.lattice_struct,
            opt_cfg,
            rec_cfg,
            model_setup.box_norm,
            model_setup.scaling,
            mesh_type="volume",
        )
        init_volume, initial_centroid = compute_tet_mesh_volume_centroid(
            mesh_init.vertices,
            mesh_init.volumes,
        )
    print(f"Initial volume: {init_volume.item():.6f}")
    print(f"Initial centroid: {initial_centroid}")

    # Phase 4: Optimizer
    n_constraints = 2 if use_center_constraint else 1
    opt_setup = setup_optimizer(
        lattice.lattice_struct, lattice.param_spline_sp,
        opt_cfg, rec_cfg,
        lock_bboxes=[([-10, -10, -10], [-9, -9, -9])],
        n_constraints=n_constraints,
    )

    # Parametric-space control-lattice visualization (static, once per run)
    export_knot_grid_paramspace(
        lattice.param_spline_sp,
        paths.reconstruction / "knot_grid_paramspace.vtp",
    )
    export_design_volume_paramspace(
        lattice.param_spline_sp,
        paths.reconstruction / "design_volume_paramspace.vts",
    )
    locked_idx = torch.nonzero(opt_setup.mask_locked_cp, as_tuple=True)[0]
    export_control_lattice_paramspace(
        lattice.param_spline_sp,
        paths.reconstruction / "control_lattice_paramspace.vtp",
        locked_idx=locked_idx if locked_idx.numel() > 0 else None,
    )

    # Phase 5: Optimization loop
    case_dir = foam_utils.prepare_foam_runtime(experiment_path / "foam_case", run_name=results_name)
    snapshot_dir = paths.optimization / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    logger = OptimizationLogger(paths.optimization, specs, total_iters=opt_cfg["num_iter"])

    convergence_obj_tol = opt_cfg.get("convergence_obj_tol", opt_cfg.get("convergence_ch_tol", None))
    convergence_window = opt_cfg.get("convergence_window", 3)

    history_constraint, history_objective = [], []
    history_grad_norm, history_obj_change, history_mma_ch = [], [], []
    history_vol_constraint, history_sens_norm = [], []
    start_time = time.time()
    iteration_times = []
    torch.set_default_dtype(DTYPE)

    for e in range(1, opt_cfg["num_iter"]):
        print(f"=== Optimization Iteration {e}/{opt_cfg['num_iter']} ===")
        iter_start = time.time()
        logger.start_iteration(e)

        # Generate current meshes (surface for CFD, volume for constraints)
        mesh, derivative = generate_mesh(
            lattice.lattice_struct, opt_cfg, rec_cfg,
            model_setup.box_norm, model_setup.scaling,
        )
        verts, faces = mesh.vertices, mesh.faces

        vol_mesh, _ = generate_mesh(
            lattice.lattice_struct, opt_cfg, rec_cfg,
            model_setup.box_norm, model_setup.scaling,
            mesh_type="volume",
        )

        from DeepSDFStruct.mesh import export_surface_mesh
        export_surface_mesh(paths.optimization / "current_shape.stl", mesh.to_gus(), derivative)

        save_shape_snapshot(
            verts=verts, faces=faces, design_domain=model_setup.design_domain,
            out_path=snapshot_dir / f"shape_{e:04d}.png", view_axis="z", title=f"Iteration {e}",
        )

        # Volume constraint
        volume, current_centroid = compute_tet_mesh_volume_centroid(
            vol_mesh.vertices,
            vol_mesh.volumes,
        )
        print(f"Current volume: {volume.item():.6f}")

        vol_constraint = float(init_volume.item()) - volume
        dV = torch.autograd.grad(vol_constraint, opt_setup.param, retain_graph=True)[0]

        # Run OpenFOAM
        foam_case = run_foam_case(case_dir, mesh, derivative, paths.optimization)

        # Load sensitivities
        sens_on_orig, J_raw = load_sensitivities(
            case_dir, foam_case, verts,
            field_name=sens_cfg.get("field_name", "pointSensVecadjS1ESI"),
            objective_path=sens_cfg.get("objective_path", "optimisation/objective/0/dragadjS1"),
            loading_method=sens_cfg.get("loading_method", "interpolate"),
            patch_name=sens_cfg.get("patch_name", "dragObject"),
            faces=faces,
        )

        normals = foam_utils.compute_vertex_normals(verts, faces, invert_normals=True)
        sens_normal = (torch.as_tensor(sens_on_orig, dtype=verts.dtype, device=verts.device) * normals).sum(dim=1)
        foam_utils.save_mesh_with_sensitivities(
            verts,
            faces,
            sens_normal.detach().cpu().numpy(),
            paths.optimization /"check_originalMesh_withSens.vtp",
        )

        # Shape gradient
        loading_method = sens_cfg.get("loading_method", "interpolate")
        dJ = foam_utils.compute_shape_gradient(
            opt_setup.param, verts, faces, sens_on_orig,
            invert_normals=sens_cfg.get("invert_normals", True),
            integrated=(loading_method == "conservative"),
        )
        J = torch.as_tensor(J_raw, device=rec_cfg["device"]).view(-1, 1)

        # Build constraints
        grad_list = [dJ, dV]
        constraints = [vol_constraint.reshape(())]

        if use_center_constraint:
            centroid_shift = current_centroid - initial_centroid
            center_constraint = (centroid_shift ** 2).sum() - opt_cfg["center_tol"] ** 2
            dC = torch.autograd.grad(center_constraint, opt_setup.param, retain_graph=True)[0]
            grad_list.append(dC)
            constraints.append(center_constraint.reshape(()))

        G = torch.stack(constraints)
        masked = mask_gradients(grad_list, opt_setup.mask_locked_flat)
        dJ_mma = masked[0]
        dG_mma = torch.stack(masked[1:], dim=0)

        print(f"dJ/dx range MMA: [{sens_on_orig.min():.3e}, {sens_on_orig.max():.3e}]")
        print(f"J: {J.item():.6e}, dJ/dz range: [{dJ.min():.3e}, {dJ.max():.3e}]")
        print(f"Max latent variable: {opt_setup.param.abs().max().item():.6f}")

        # VTK & STL export
        if paths.heavy_data is not None:
            foam_utils.export_vtk_for_iteration(foam_case, case_dir, paths.heavy_data, e)
            shutil.copy2(
                paths.optimization / "current_shape.stl",
                paths.heavy_data / "stl_series" / f"shape_{e:04d}.stl",
            )

        history_constraint.append(volume.item())
        history_objective.append(J.item())
        history_grad_norm.append(dJ.norm().item())
        history_vol_constraint.append(vol_constraint.item())
        history_sens_norm.append(torch.as_tensor(sens_on_orig).norm().item())
        plot_optimization_history(history_objective, history_constraint, init_volume, paths.optimization)

        # MMA step
        opt_setup.optimizer.step(J, dJ_mma, G, dG_mma)
        history_mma_ch.append(float(opt_setup.optimizer.ch))
        if len(history_objective) >= 2:
            history_obj_change.append(abs(history_objective[-1] - history_objective[-2]) / abs(history_objective[0]))
        else:
            history_obj_change.append(float('nan'))

        plot_convergence_diagnostics(
            {
                "obj_change": history_obj_change,
                "grad_norm": history_grad_norm,
                "mma_ch": history_mma_ch,
                "vol_constraint": history_vol_constraint,
                "sens_norm": history_sens_norm,
            },
            paths.optimization,
        )
        torch.save(
            list(lattice.lattice_struct.parametrization.parameters()),
            paths.optimization / "parameters.pt",
        )

        log_iteration_time(iter_start, start_time, iteration_times, opt_cfg["num_iter"], e)

        logger.log_iteration(
            iteration=e,
            objective=J.item(),
            vol_constraint=vol_constraint.item(),
            volume=volume.item(),
            center_constraint=float((current_centroid - initial_centroid).pow(2).sum() - opt_cfg["center_tol"] ** 2) if use_center_constraint else None,
            grad_norm=history_grad_norm[-1],
            sens_norm=history_sens_norm[-1],
            mma_ch=history_mma_ch[-1],
            obj_change=history_obj_change[-1],
            max_param=opt_setup.param.abs().max().item(),
        )

        # Convergence check
        if convergence_obj_tol is not None and len(history_obj_change) >= convergence_window:
            recent = history_obj_change[-convergence_window:]
            if all(math.isfinite(ch) and ch < convergence_obj_tol for ch in recent):
                print(
                    f"Converged: relative objective change < {convergence_obj_tol} for "
                    f"{convergence_window} consecutive iterations."
                )
                break

    logger.close()
    FoamCase(case_dir).clean()
    shutil.rmtree(case_dir, ignore_errors=True)

    return {
        "init_volume": float(init_volume.item()),
        "final_volume": float(volume.item()),
        "final_objective": float(J.item()),
        "results_dir": str(paths.results),
    }


if __name__ == "__main__":
    experiment_path = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(experiment_path / "config_cube_with_holes.json"),
        help="Path to an experiment JSON config.",
    )
    args = parser.parse_args()

    specs = ExperimentSpecifications(args.config)
    optimize_shape(experiment_path, specs)

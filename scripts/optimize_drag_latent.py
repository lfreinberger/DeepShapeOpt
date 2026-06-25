"""Drag optimization with a DeepSDF latent B-spline parametrization."""

from __future__ import annotations

import argparse
import logging
import math
import shutil
import time
from pathlib import Path

import torch
from foamlib import FoamCase

from DeepSDFStruct.export_knot_grid import (
    export_control_lattice_paramspace,
    export_design_volume_paramspace,
    export_knot_grid_paramspace,
)
from DeepSDFStruct.mesh import export_surface_mesh

import deepshapeopt.config as config
import deepshapeopt.foam_utils as foam_utils
from deepshapeopt.config import ExperimentSpecifications
from deepshapeopt.logging import OptimizationLogger
from deepshapeopt.mesh import compute_tet_mesh_volume_centroid
from deepshapeopt.plotting_utils import (
    plot_convergence_diagnostics,
    plot_optimization_history,
    save_shape_snapshot,
)
from deepshapeopt.runtime import (
    configure_logging,
    has_converged,
    is_debug_enabled,
    log_iteration_summary,
    log_timing,
)
from deepshapeopt.shape_optimization import (
    build_lattice,
    generate_mesh,
    load_sensitivities,
    mask_gradients,
    run_foam_case,
    run_reconstruction,
    setup_model_and_domain,
    setup_optimizer,
)


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_PATH = PROJECT_ROOT / "experiments" / "drag_cube"


def export_wall_stl(verts: torch.Tensor, faces: torch.Tensor, path: Path) -> None:
    """Export the triangulated wall surface of the hex mesh as STL."""
    import trimesh

    trimesh.Trimesh(
        vertices=verts.detach().cpu().numpy(),
        faces=faces.detach().cpu().numpy(),
        process=False,
    ).export(path)


def optimize_shape(experiment_path: Path, specs):
    rec_cfg = specs["reconstruction"]
    opt_cfg = specs["optimization"]
    sens_cfg = opt_cfg.get("sensitivity", {})
    debug = is_debug_enabled(specs)

    results_name = specs.get("results_name", "results")
    paths = config.make_experiment_paths(
        experiment_path,
        results_name=results_name,
        heavy_data_output_path=opt_cfg.get("heavy_data_output_path"),
    )
    config.ensure_experiment_dirs(paths)
    configure_logging(debug, paths.optimization / "run.log")

    if debug and paths.heavy_data is not None:
        for subdir in ("vtk_series", "stl_series"):
            out_dir = paths.heavy_data / subdir
            if out_dir.exists():
                shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True)

    use_center_constraint = opt_cfg.get("use_center_constraint", False)
    model_setup = setup_model_and_domain(rec_cfg, paths.reconstruction)
    lattice = build_lattice(rec_cfg, model_setup.model, model_setup.sdf, model_setup.box_norm)

    run_reconstruction(
        lattice.lattice_struct,
        model_setup.mesh_norm,
        model_setup.box_norm,
        rec_cfg,
        paths.reconstruction,
        model_setup.model,
        model_setup.scaling,
        opt_cfg,
        debug=debug,
        heavy_data=paths.heavy_data,
    )

    mesh_pipeline = opt_cfg.get("mesh_pipeline", "snappy")
    hex_pipeline = None
    if mesh_pipeline == "sdf_hex":
        from deepshapeopt.hexmesh import SdfHexMeshPipeline

        hex_pipeline = SdfHexMeshPipeline(
            lattice.lattice_struct, model_setup, opt_cfg, paths.optimization
        )
        hex_pipeline.build()
        init_volume, initial_centroid = hex_pipeline.volume_centroid(no_grad=True)
    elif mesh_pipeline == "snappy":
        with torch.no_grad():
            mesh_init, _ = generate_mesh(
                lattice.lattice_struct,
                opt_cfg,
                rec_cfg,
                model_setup.box_norm,
                model_setup.scaling,
                mesh_type="volume",
                extend_bounds=True
            )
            init_volume, initial_centroid = compute_tet_mesh_volume_centroid(
                mesh_init.vertices,
                mesh_init.volumes,
            )
    else:
        raise ValueError(f"Unknown mesh_pipeline: {mesh_pipeline!r}")

    LOGGER.info("Initial volume: %.6f", init_volume.item())
    LOGGER.info("Initial centroid: %s", initial_centroid)

    n_constraints = 2 if use_center_constraint else 1
    opt_setup = setup_optimizer(
        lattice.lattice_struct,
        lattice.param_spline_sp,
        opt_cfg,
        rec_cfg,
        lock_bboxes=None,
        n_constraints=n_constraints,
    )

    if debug:
        export_knot_grid_paramspace(lattice.param_spline_sp, paths.reconstruction / "knot_grid_paramspace.vtp")
        export_design_volume_paramspace(lattice.param_spline_sp, paths.reconstruction / "design_volume_paramspace.vts")
        locked_idx = torch.nonzero(opt_setup.mask_locked_cp, as_tuple=True)[0]
        export_control_lattice_paramspace(
            lattice.param_spline_sp,
            paths.reconstruction / "control_lattice_paramspace.vtp",
            locked_idx=locked_idx if locked_idx.numel() > 0 else None,
        )

    case_dir = foam_utils.prepare_foam_runtime(experiment_path / "foam_case", run_name=results_name)
    foam_utils.select_allrun(case_dir, mesh_pipeline)
    snapshot_dir = paths.optimization / "snapshots"
    if debug:
        snapshot_dir.mkdir(parents=True, exist_ok=True)

    logger = OptimizationLogger(paths.optimization, specs, total_iters=opt_cfg["num_iter"])
    convergence_obj_tol = opt_cfg.get("convergence_obj_tol", opt_cfg.get("convergence_ch_tol"))
    convergence_window = opt_cfg.get("convergence_window", 3)

    history_constraint, history_objective = [], []
    history_grad_norm, history_obj_change, history_mma_ch = [], [], []
    history_vol_constraint, history_sens_norm = [], []
    start_time = time.time()
    iteration_times = []
    volume = init_volume
    J = torch.zeros((1, 1), device=rec_cfg["device"], dtype=opt_setup.param.dtype)


    for iteration in range(opt_cfg["num_iter"]):
        LOGGER.info("=== Optimization iteration %d/%d ===", iteration, opt_cfg["num_iter"] - 1)
        iter_start = time.time()
        logger.start_iteration(iteration)

        if mesh_pipeline == "sdf_hex":
            hex_result = hex_pipeline.build()
            verts = hex_result.surface_points
            faces = hex_result.wall_tris_local.to(verts.device)
            export_wall_stl(verts, faces, paths.optimization / "current_shape.stl")

            volume, current_centroid = hex_pipeline.volume_centroid()
            vol_constraint = float(init_volume.item()) - volume
            dV = torch.autograd.grad(vol_constraint, opt_setup.param, retain_graph=True)[0]

            foam_case = hex_pipeline.run_case(case_dir, verbose=debug)
            sens_on_orig, J_raw = hex_pipeline.load_sensitivities(
                case_dir,
                foam_case,
                field_name=sens_cfg.get("field_name", "pointSensVecadjS1ESI"),
                objective_path=sens_cfg.get("objective_path", "optimisation/objective/0/dragadjS1"),
            )
            dJ = foam_utils.compute_shape_gradient(
                opt_setup.param,
                verts,
                faces,
                sens_on_orig,
                integrated=True,
            )
        else:
            mesh, derivative = generate_mesh(
                lattice.lattice_struct,
                opt_cfg,
                rec_cfg,
                model_setup.box_norm,
                model_setup.scaling,
                extend_bounds=True
            )
            verts, faces = mesh.vertices, mesh.faces
            vol_mesh, _ = generate_mesh(
                lattice.lattice_struct,
                opt_cfg,
                rec_cfg,
                model_setup.box_norm,
                model_setup.scaling,
                mesh_type="volume",
                extend_bounds=True
            )
            export_surface_mesh(paths.optimization / "current_shape.stl", mesh.to_gus(), derivative)

            volume, current_centroid = compute_tet_mesh_volume_centroid(vol_mesh.vertices, vol_mesh.volumes)
            vol_constraint = float(init_volume.item()) - volume
            dV = torch.autograd.grad(vol_constraint, opt_setup.param, retain_graph=True)[0]

            foam_case = run_foam_case(case_dir, mesh, derivative, paths.optimization, plot_residuals=debug)
            loading_method = sens_cfg.get("loading_method", "interpolate")
            sens_on_orig, J_raw = load_sensitivities(
                case_dir,
                foam_case,
                verts,
                field_name=sens_cfg.get("field_name", "pointSensVecadjS1ESI"),
                objective_path=sens_cfg.get("objective_path", "optimisation/objective/0/dragadjS1"),
                loading_method=loading_method,
                patch_name=sens_cfg.get("patch_name", "dragObject"),
                faces=faces,
            )

            dJ = foam_utils.compute_shape_gradient(
                opt_setup.param,
                verts,
                faces,
                sens_on_orig,
                invert_normals=sens_cfg.get("invert_normals", True),
                integrated=(loading_method == "conservative"),
            )

        if debug:
            save_shape_snapshot(
                verts=verts,
                faces=faces,
                design_domain=model_setup.design_domain,
                out_path=snapshot_dir / f"shape_{iteration:04d}.png",
                view_axis="z",
                title=f"Iteration {iteration}",
            )

        normals = foam_utils.compute_vertex_normals(verts, faces, invert_normals=True)
        sens_tensor = torch.as_tensor(sens_on_orig, dtype=verts.dtype, device=verts.device)
        if debug:
            sens_normal = (sens_tensor * normals).sum(dim=1)
            foam_utils.save_mesh_with_sensitivities(
                verts,
                faces,
                sens_normal.detach().cpu().numpy(),
                paths.optimization / "check_original_mesh_with_sens.vtp",
            )

        J = torch.as_tensor(J_raw, device=rec_cfg["device"], dtype=opt_setup.param.dtype).view(-1, 1)

        grad_list = [dJ, dV]
        constraints = [vol_constraint.reshape(())]
        center_constraint = None
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

        LOGGER.debug("Sensitivity range: [%.3e, %.3e]", sens_on_orig.min(), sens_on_orig.max())
        LOGGER.debug("dJ range: [%.3e, %.3e]", dJ.min(), dJ.max())
        LOGGER.debug("Max latent variable: %.6f", opt_setup.param.abs().max().item())

        if debug and paths.heavy_data is not None:
            foam_utils.export_vtk_for_iteration(foam_case, case_dir, paths.heavy_data, iteration)
            shutil.copy2(
                paths.optimization / "current_shape.stl",
                paths.heavy_data / "stl_series" / f"shape_{iteration:04d}.stl",
            )

        history_constraint.append(volume.item())
        history_objective.append(J.item())
        history_grad_norm.append(dJ.norm().item())
        history_vol_constraint.append(vol_constraint.item())
        history_sens_norm.append(sens_tensor.norm().item())
        plot_optimization_history(
            history_objective,
            history_constraint,
            init_volume,
            paths.optimization,
            obj_label="Drag objective",
            con_label="Volume",
        )

        opt_setup.optimizer.step(J, dJ_mma, G, dG_mma)
        history_mma_ch.append(float(opt_setup.optimizer.ch))
        if len(history_objective) >= 2 and history_objective[0] != 0:
            history_obj_change.append(abs(history_objective[-1] - history_objective[-2]) / abs(history_objective[0]))
        else:
            history_obj_change.append(float("nan"))

        if debug:
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
        torch.save(list(lattice.lattice_struct.parametrization.parameters()), paths.optimization / "parameters.pt")

        log_iteration_summary(
            LOGGER,
            objective=J.item(),
            volume=volume.item(),
            vol_constraint=vol_constraint.item(),
            center_constraint=float(center_constraint.detach()) if center_constraint is not None else None,
            mma_ch=history_mma_ch[-1],
        )
        log_timing(LOGGER, iter_start, start_time, iteration_times, opt_cfg["num_iter"], iteration)

        logger.log_iteration(
            iteration=iteration,
            objective=J.item(),
            vol_constraint=vol_constraint.item(),
            volume=volume.item(),
            center_constraint=float(center_constraint.detach()) if center_constraint is not None else None,
            grad_norm=history_grad_norm[-1],
            sens_norm=history_sens_norm[-1],
            mma_ch=history_mma_ch[-1],
            obj_change=history_obj_change[-1],
            max_param=opt_setup.param.abs().max().item(),
        )

        if has_converged(history_obj_change, convergence_obj_tol, convergence_window):
            LOGGER.info(
                "Converged: relative objective change < %s for %d consecutive iterations",
                convergence_obj_tol,
                convergence_window,
            )
            break

    logger.close()
    if case_dir.exists():
        FoamCase(case_dir).clean()
        shutil.rmtree(case_dir, ignore_errors=True)

    return {
        "init_volume": float(init_volume.item()),
        "final_volume": float(volume.item()),
        "final_objective": float(J.item()),
        "results_dir": str(paths.results),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(DEFAULT_EXPERIMENT_PATH / "config_latent_cube.json"),
        help="Path to an experiment JSON config.",
    )
    args = parser.parse_args()

    experiment_path = Path(args.config).resolve().parent
    specs = ExperimentSpecifications(args.config)
    result = optimize_shape(experiment_path, specs)
    print(f"Results written to: {result['results_dir']}")
    print(f"Final objective: {result['final_objective']:.6e}")


if __name__ == "__main__":
    main()

"""Drag optimization with Free-Form Deformation."""

from __future__ import annotations

import argparse
import logging
import math
import shutil
import time
from pathlib import Path

import numpy as np
import splinepy
import torch
import trimesh
from foamlib import FoamCase

from DeepSDFStruct.SDF import SDFfromMesh
from DeepSDFStruct.export_knot_grid import (
    export_control_lattice_physical,
    export_control_volume_physical,
)
from DeepSDFStruct.mesh import TorchSpline, create_3D_mesh, export_surface_mesh
from DeepSDFStruct.optimization import MMA

import deepshapeopt.config as config
import deepshapeopt.foam_utils as foam_utils
from deepshapeopt.config import ExperimentSpecifications
from deepshapeopt.logging import OptimizationLogger
from deepshapeopt.mesh import compute_tet_mesh_volume_centroid
from deepshapeopt.plotting_utils import (
    plot_convergence_diagnostics,
    plot_optimization_history,
    plot_residuals_from_log,
    save_shape_snapshot,
)
from deepshapeopt.runtime import (
    configure_logging,
    has_converged,
    is_debug_enabled,
    log_iteration_summary,
    log_timing,
)
from deepshapeopt.shape_optimization import load_sensitivities, mask_gradients


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_PATH = PROJECT_ROOT / "experiments" / "drag_cube"


class FFDDeformation(torch.nn.Module):
    """Map points through a displacement spline over the design domain."""

    def __init__(self, disp_spline_sp, design_domain, device="cpu", dtype=torch.float32):
        super().__init__()
        self.disp = TorchSpline(disp_spline_sp, device=device, dtype=dtype)
        mins = torch.tensor(design_domain[0], device=device, dtype=dtype)
        maxs = torch.tensor(design_domain[1], device=device, dtype=dtype)
        self.register_buffer("domain_min", mins)
        self.register_buffer("domain_max", maxs)
        self.register_buffer("domain_size", maxs - mins)

    @property
    def control_points(self):
        return self.disp.control_points

    def forward(self, queries: torch.Tensor):
        queries = queries.to(device=self.control_points.device, dtype=self.control_points.dtype)
        xi = (queries - self.domain_min) / self.domain_size
        return queries + self.disp(xi)


def make_clamped_knots_unit(degree, n_control_points):
    n_internal = n_control_points - degree - 1
    parts = [torch.zeros(degree + 1)]
    if n_internal > 0:
        parts.append(torch.linspace(0.0, 1.0, n_internal + 2)[1:-1])
    parts.append(torch.ones(degree + 1))
    return torch.cat(parts)


def define_control_points(design_domain, n_control_points, device="cpu", dtype=torch.float32):
    mins = torch.as_tensor(design_domain[0], device=device, dtype=dtype)
    maxs = torch.as_tensor(design_domain[1], device=device, dtype=dtype)
    ncp = torch.as_tensor(n_control_points, dtype=dtype, device=device)
    xs = torch.linspace(mins[0], maxs[0], int(ncp[0].item()), device=device, dtype=dtype)
    ys = torch.linspace(mins[1], maxs[1], int(ncp[1].item()), device=device, dtype=dtype)
    zs = torch.linspace(mins[2], maxs[2], int(ncp[2].item()), device=device, dtype=dtype)
    X, Y, Z = torch.meshgrid(xs, ys, zs, indexing="ij")
    return torch.stack(
        [
            X.permute(2, 1, 0).reshape(-1),
            Y.permute(2, 1, 0).reshape(-1),
            Z.permute(2, 1, 0).reshape(-1),
        ],
        dim=1,
    )


def compute_min_jacobian_det(deformation_ffd, n_samples_per_dim=6, ks_rho=50.0):
    """Return a smooth lower-bound surrogate for the minimum FFD Jacobian."""
    device = deformation_ffd.control_points.device
    dtype = deformation_ffd.control_points.dtype

    lin = torch.linspace(0.0, 1.0, n_samples_per_dim, device=device, dtype=dtype)
    gx, gy, gz = torch.meshgrid(lin, lin, lin, indexing="ij")
    xi = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=1).requires_grad_(True)
    displacement = deformation_ffd.disp(xi)
    jac_disp = torch.stack(
        [torch.autograd.grad(displacement[:, i].sum(), xi, create_graph=True)[0] for i in range(3)],
        dim=1,
    )
    inv_size = (1.0 / deformation_ffd.domain_size).view(1, 1, 3)
    jacobian = torch.eye(3, device=device, dtype=dtype).unsqueeze(0) + jac_disp * inv_size
    det_j = torch.linalg.det(jacobian)
    LOGGER.debug("Sampled min det(J_FFD): %.4f", det_j.min().item())
    return -torch.logsumexp(-ks_rho * det_j, dim=0) / ks_rho


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
        for subdir in ("vtk_series", "stl_series", "control_lattice_series"):
            out_dir = paths.heavy_data / subdir
            if out_dir.exists():
                shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True)
        lattice_series_dir = paths.heavy_data / "control_lattice_series"
    else:
        lattice_series_dir = paths.optimization / "control_lattice_series"

    dtype = torch.float32
    use_center_constraint = opt_cfg.get("use_center_constraint", False)
    center_tol = opt_cfg.get("center_tol", 0.0)
    use_jacobian_constraint = opt_cfg.get("use_jacobian_constraint", False)
    jacobian_threshold = opt_cfg.get("jacobian_threshold", 0.01)
    jacobian_samples = opt_cfg.get("jacobian_samples", 6)
    design_domain = torch.tensor(rec_cfg["design_domain"], device=rec_cfg["device"], dtype=dtype)

    mesh_orig = trimesh.load(rec_cfg["mesh_path"])
    sdf_mesh = SDFfromMesh(mesh_orig, scale=False)
    spline_degree = rec_cfg["spline_degree"]
    n_control_points = rec_cfg["n_control_points"]
    knot_vectors = [make_clamped_knots_unit(spline_degree[i], n_control_points[i]) for i in range(3)]
    control_point_coords = define_control_points(design_domain, n_control_points, device=rec_cfg["device"], dtype=dtype)

    disp_spline_sp = splinepy.BSpline(
        degrees=spline_degree,
        knot_vectors=[kv.tolist() for kv in knot_vectors],
        control_points=np.zeros_like(control_point_coords.cpu().numpy()),
    )
    deformation_ffd = FFDDeformation(disp_spline_sp, rec_cfg["design_domain"], device=rec_cfg["device"], dtype=dtype)
    with torch.no_grad():
        deformation_ffd.control_points.zero_()
    param = deformation_ffd.control_points

    mesh_resolution = opt_cfg["mesh_resolution"]
    sdf_bounds = torch.tensor(sdf_mesh._get_domain_bounds(), device=rec_cfg["device"], dtype=torch.float32)
    mesh_base, _ = create_3D_mesh(
        sdf_mesh,
        mesh_resolution,
        mesh_type="surface",
        differentiate=False,
        device=rec_cfg["device"],
        bounds=sdf_bounds,
    )
    verts_base = mesh_base.vertices.detach()
    faces = mesh_base.faces
    vol_mesh_base, _ = create_3D_mesh(
        sdf_mesh,
        mesh_resolution,
        mesh_type="volume",
        differentiate=False,
        device=rec_cfg["device"],
        bounds=sdf_bounds,
    )
    vol_verts_base = vol_mesh_base.vertices.detach()
    tets = vol_mesh_base.volumes
    export_surface_mesh(paths.reconstruction / "reconstructed_mesh.stl", mesh_base.to_gus(), None)

    with torch.no_grad():
        vol_verts_init = deformation_ffd(vol_verts_base)
        init_volume, initial_centroid = compute_tet_mesh_volume_centroid(vol_verts_init, tets)
    LOGGER.info("Initial volume: %.6f", init_volume.item())
    LOGGER.info("Initial centroid: %s", initial_centroid)

    # Optional node-local scratch root for the transient OpenFOAM case (see
    # optimize_drag_latent.py). Absent -> case lives next to the template.
    foam_runtime_root = opt_cfg.get("foam_runtime_root")
    if foam_runtime_root:
        LOGGER.info("Foam runtime root (scratch): %s", foam_runtime_root)
    case_dir = foam_utils.prepare_foam_runtime(
        experiment_path / "foam_case", run_name=results_name,
        runtime_root=Path(foam_runtime_root) if foam_runtime_root else None,
    )
    bounds = np.full((param.reshape(-1, 1).shape[0], 2), opt_cfg["bounds"])
    n_constraints = 1 + int(use_center_constraint) + int(use_jacobian_constraint)
    optimizer = MMA(param.reshape(-1, 1), bounds, max_step=opt_cfg["max_step"], n_constraints=n_constraints)

    snapshot_dir = paths.optimization / "snapshots"
    if debug:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        lattice_series_dir.mkdir(parents=True, exist_ok=True)

    logger = OptimizationLogger(paths.optimization, specs, total_iters=opt_cfg["num_iter"])
    convergence_obj_tol = opt_cfg.get("convergence_obj_tol", opt_cfg.get("convergence_ch_tol"))
    convergence_window = opt_cfg.get("convergence_window", 3)

    history_constraint, history_objective = [], []
    history_grad_norm, history_obj_change, history_mma_ch = [], [], []
    history_vol_constraint, history_sens_norm, history_sens_to_grad_ratio = [], [], []
    history_cons_max_proj_dist = []
    start_time = time.time()
    iteration_times = []
    volume = init_volume
    J = torch.zeros((1, 1), device=rec_cfg["device"], dtype=param.dtype)

    for iteration in range(opt_cfg["num_iter"]):
        LOGGER.info("=== Optimization iteration %d/%d ===", iteration, opt_cfg["num_iter"] - 1)
        iter_start = time.time()
        logger.start_iteration(iteration)

        verts = deformation_ffd(verts_base)
        current_mesh = trimesh.Trimesh(vertices=verts.detach().cpu().numpy(), faces=faces.cpu().numpy())
        current_mesh.export(paths.optimization / "current_shape.stl")

        cp_current = control_point_coords + deformation_ffd.control_points
        export_control_lattice_physical(cp_current, n_control_points, paths.optimization / "control_lattice.vtp")
        export_control_volume_physical(
            cp_current,
            n_control_points,
            paths.optimization / "control_volume.vts",
            undeformed=control_point_coords,
        )

        if debug:
            export_control_lattice_physical(
                cp_current,
                n_control_points,
                lattice_series_dir / f"control_lattice_{iteration:04d}.vtp",
            )
            export_control_volume_physical(
                cp_current,
                n_control_points,
                lattice_series_dir / f"control_volume_{iteration:04d}.vts",
                undeformed=control_point_coords,
            )
            save_shape_snapshot(
                verts=verts,
                faces=faces,
                design_domain=design_domain,
                out_path=snapshot_dir / f"shape_{iteration:04d}.png",
                view_axis="z",
                title=f"Iteration {iteration}",
            )

        vol_verts = deformation_ffd(vol_verts_base)
        volume, current_centroid, vol_diag = compute_tet_mesh_volume_centroid(
            vol_verts,
            tets,
            return_diagnostics=True,
        )
        vol_constraint = float(init_volume.item()) - volume
        dV = torch.autograd.grad(vol_constraint, param, retain_graph=True)[0]

        current_mesh.export(case_dir / "constant/triSurface/shape.stl")
        foam_case = foam_utils.run_openfoam_case(case_dir, verbose=False)
        if debug:
            log_path = case_dir / "log.adjointOptimisationFoam"
            if log_path.exists():
                plot_residuals_from_log(log_path, output_dir=str(paths.optimization))

        loading_method = sens_cfg.get("loading_method", "project")
        sens_diag = {}
        sens_kwargs = dict(
            field_name=sens_cfg.get("field_name", "pointSensVecadjS1ESI"),
            objective_path=sens_cfg.get("objective_path", "optimisation/objective/0/dragadjS1"),
            loading_method=loading_method,
            patch_name=sens_cfg.get("patch_name", "dragObject"),
            faces=faces,
        )
        if loading_method == "conservative":
            sens_on_orig, J_raw, sens_diag = load_sensitivities(
                case_dir,
                foam_case,
                verts,
                return_diagnostics=True,
                **sens_kwargs,
            )
        else:
            sens_on_orig, J_raw = load_sensitivities(case_dir, foam_case, verts, **sens_kwargs)

        dJ = foam_utils.compute_shape_gradient(
            param,
            verts,
            faces,
            sens_on_orig,
            integrated=(loading_method == "conservative"),
        )
        J = torch.as_tensor(J_raw, device=rec_cfg["device"], dtype=param.dtype).view(-1, 1)

        grad_list = [dJ, dV]
        constraints = [vol_constraint.reshape(())]
        center_constraint = None
        jac_constraint = None
        if use_center_constraint:
            centroid_shift = current_centroid - initial_centroid
            center_constraint = (centroid_shift ** 2).sum() - center_tol ** 2
            dC = torch.autograd.grad(center_constraint, param, retain_graph=True)[0]
            grad_list.append(dC)
            constraints.append(center_constraint.reshape(()))

        if use_jacobian_constraint:
            min_det_ks = compute_min_jacobian_det(deformation_ffd, n_samples_per_dim=jacobian_samples)
            jac_constraint = jacobian_threshold - min_det_ks
            dJac = torch.autograd.grad(jac_constraint, param, retain_graph=True)[0]
            grad_list.append(dJac)
            constraints.append(jac_constraint.reshape(()))
            LOGGER.debug("min det(J_FFD) [KS]: %.4f threshold=%.4f", min_det_ks.item(), jacobian_threshold)

        G = torch.stack(constraints)
        no_locked_params = torch.zeros(param.numel(), dtype=torch.bool, device=param.device)
        masked = mask_gradients(grad_list, no_locked_params)
        dJ_mma = masked[0]
        dG_mma = torch.stack(masked[1:], dim=0)

        LOGGER.debug("dJ range: [%.3e, %.3e]", dJ.min(), dJ.max())
        LOGGER.debug("Max control point displacement: %.6f", param.abs().max().item())

        if debug and paths.heavy_data is not None:
            foam_utils.export_vtk_for_iteration(foam_case, case_dir, paths.heavy_data, iteration)
            shutil.copy2(
                paths.optimization / "current_shape.stl",
                paths.heavy_data / "stl_series" / f"shape_{iteration:04d}.stl",
            )

        history_constraint.append(volume.item())
        history_objective.append(J.item())
        grad_norm = float(dJ.norm().item())
        sens_norm = float(torch.as_tensor(sens_on_orig).norm().item())
        history_grad_norm.append(grad_norm)
        history_vol_constraint.append(vol_constraint.item())
        history_sens_norm.append(sens_norm)
        history_sens_to_grad_ratio.append(grad_norm / (sens_norm + 1e-12))
        history_cons_max_proj_dist.append(float(sens_diag.get("conservative_max_proj_dist", float("nan"))))
        plot_optimization_history(
            history_objective,
            history_constraint,
            init_volume,
            paths.optimization,
            obj_label="Drag objective",
            con_label="Volume",
        )

        optimizer.step(J, dJ_mma, G, dG_mma)
        history_mma_ch.append(float(optimizer.ch))
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
                    "sens_to_grad_ratio": history_sens_to_grad_ratio,
                    "conservative_max_proj_dist": history_cons_max_proj_dist,
                },
                paths.optimization,
            )

        log_iteration_summary(
            LOGGER,
            objective=J.item(),
            volume=volume.item(),
            vol_constraint=vol_constraint.item(),
            center_constraint=float(center_constraint.detach()) if center_constraint is not None else None,
            jacobian_constraint=float(jac_constraint.detach()) if jac_constraint is not None else None,
            mma_ch=history_mma_ch[-1],
        )
        log_timing(LOGGER, iter_start, start_time, iteration_times, opt_cfg["num_iter"], iteration)

        logger.log_iteration(
            iteration=iteration,
            objective=J.item(),
            vol_constraint=vol_constraint.item(),
            volume=volume.item(),
            center_constraint=float(center_constraint.detach()) if center_constraint is not None else None,
            jacobian_constraint=float(jac_constraint.detach()) if jac_constraint is not None else None,
            grad_norm=grad_norm,
            sens_norm=sens_norm,
            mma_ch=history_mma_ch[-1],
            obj_change=history_obj_change[-1],
            max_param=param.abs().max().item(),
            sens_to_grad_ratio=history_sens_to_grad_ratio[-1],
            n_reoriented_tets=vol_diag["n_reoriented"],
            conservative_max_proj_dist=sens_diag.get("conservative_max_proj_dist"),
            conservative_l1_ratio=sens_diag.get("conservative_l1_ratio"),
            conservative_vec_norm_ratio=sens_diag.get("conservative_vec_norm_ratio"),
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
        default=str(DEFAULT_EXPERIMENT_PATH / "config_ffd_cube_with_cylinders_7x7x7.json"),
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

"""
Shared functions for shape optimization experiments.

These functions extract the common phases from experiment scripts so each
experiment can be written as a short sequence of calls with only
experiment-specific logic inline.
"""
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import splinepy
import torch
import trimesh

from DeepSDFStruct.lattice_structure import LatticeSDFStruct
from DeepSDFStruct.mesh import TorchScaling, create_3D_mesh, export_surface_mesh, export_sdf_grid_vtk
from DeepSDFStruct.optimization import MMA
from DeepSDFStruct.parametrization import SplineParametrization
from DeepSDFStruct.pretrained_models import get_model
from DeepSDFStruct.SDF import SDFfromDeepSDF

from deepshapeopt.config import ExperimentPaths, make_experiment_paths, ensure_experiment_dirs
from deepshapeopt.parameters import locked_indices_from_bboxes, make_locked_masks
from deepshapeopt.reconstruction import fit_box_to_unit_cube, init_spline_parameters, build_parameter_spline, run_sdf_reconstruction
import deepshapeopt.foam_utils as foam_utils
from deepshapeopt.plotting_utils import (
    plot_optimization_history,
    plot_residuals_from_log,
    save_shape_snapshot,
)

DTYPE = torch.float32


# ---------------------------------------------------------------------------
# Phase 1: Model & domain setup
# ---------------------------------------------------------------------------

@dataclass
class ModelSetup:
    model: Any
    sdf: Any
    design_domain: torch.Tensor
    scaling: TorchScaling
    scale: torch.Tensor
    center: torch.Tensor
    norm_fn: Any
    denorm_fn: Any
    box_norm: torch.Tensor
    mesh_orig: trimesh.Trimesh
    mesh_norm: trimesh.Trimesh


def setup_model_and_domain(rec_cfg: dict, rec_results_path: Path) -> ModelSetup:
    """Load DeepSDF model, normalize the design domain, prepare reference mesh."""
    model = get_model(
        model=rec_cfg["model_path"],
        checkpoint=rec_cfg["model_checkpoint"],
        device=rec_cfg["device"],
    )

    design_domain = torch.tensor(rec_cfg["design_domain"], device=rec_cfg["device"], dtype=DTYPE)
    scale, center, norm_fn, denorm_fn, box_norm = fit_box_to_unit_cube(design_domain)

    print("scale:", scale.item())
    print("center:", center)
    print("normalized box:", box_norm)

    scaling = TorchScaling(
        scale_factors=scale, translation=center, bounds=design_domain, device=rec_cfg["device"]
    )

    mesh_orig = trimesh.load(rec_cfg["mesh_path"])
    V = torch.from_numpy(mesh_orig.vertices).to(device=model.device, dtype=DTYPE)
    V_norm = norm_fn(V)
    mesh_norm = mesh_orig.copy()
    mesh_norm.vertices = V_norm.detach().cpu().numpy()
    mesh_norm.export(rec_results_path / "gt_mesh_normalized.stl")

    sdf = SDFfromDeepSDF(model)

    return ModelSetup(
        model=model, sdf=sdf, design_domain=design_domain,
        scaling=scaling, scale=scale, center=center,
        norm_fn=norm_fn, denorm_fn=denorm_fn, box_norm=box_norm,
        mesh_orig=mesh_orig, mesh_norm=mesh_norm,
    )


# ---------------------------------------------------------------------------
# Phase 2: Lattice / spline construction
# ---------------------------------------------------------------------------

@dataclass
class LatticeSetup:
    lattice_struct: LatticeSDFStruct
    param_spline_sp: splinepy.BSpline
    param_spline: SplineParametrization


def build_lattice(rec_cfg: dict, model, sdf, box_norm: torch.Tensor) -> LatticeSetup:
    """Build the B-spline parametrization and LatticeSDFStruct."""

    mins = box_norm[0].detach().cpu().numpy()
    maxs = box_norm[1].detach().cpu().numpy()
    degree = rec_cfg["spline_degree"]
    ctrl_dim = model._trained_latent_vectors[0].shape[0]

    param_spline_sp = build_parameter_spline(
        spline_degrees=degree,
        tiling=rec_cfg["tiling"],
        latent_dim=ctrl_dim,
        bounds=np.stack([mins, maxs]),
    )

    param_spline = SplineParametrization(param_spline_sp, device=rec_cfg["device"])
    init_spline_parameters(param_spline, mean=0.0, std=0.001)

    lattice_struct = LatticeSDFStruct(
        tiling=rec_cfg["tiling"],
        microtile=sdf,
        parametrization=param_spline,
        bounds=box_norm,
    )

    return LatticeSetup(
        lattice_struct=lattice_struct,
        param_spline_sp=param_spline_sp,
        param_spline=param_spline,
    )


# ---------------------------------------------------------------------------
# Phase 3: Reconstruction
# ---------------------------------------------------------------------------

def _with_float32_lattice(lattice_struct, box_norm, fn):
    """Run *fn* with the lattice temporarily cast to float32.

    The DeepSDF decoder weights are float32 and FlexiCubes creates
    internal buffers in float32 (and uses ``torch.get_default_dtype()``
    for grid construction), so all SDF evaluation / mesh generation
    must happen in float32.  The MMA optimizer, however, works in
    float64 (DTYPE).  This helper casts the parametrization, the
    lattice bounds buffer, and the default dtype to float32, calls *fn*,
    then restores everything.
    """
    # Save and cast parametrization params
    params = list(lattice_struct.parametrization.parameters())
    saved_params = [p.data for p in params]
    for p in params:
        p.data = p.data.float()

    # Save and cast the lattice bounds buffer (used for coordinate
    # normalization inside LatticeSDFStruct._compute)
    saved_bounds = lattice_struct.bounds.data
    lattice_struct.bounds.data = lattice_struct.bounds.data.float()

    # FlexiCubes uses torch.get_default_dtype() to create grid points
    saved_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)

    try:
        return fn(box_norm.float())
    finally:
        torch.set_default_dtype(saved_default_dtype)
        for p, s in zip(params, saved_params):
            p.data = s
        lattice_struct.bounds.data = saved_bounds


def run_reconstruction(
    lattice_struct: LatticeSDFStruct,
    mesh_norm: trimesh.Trimesh,
    box_norm: torch.Tensor,
    rec_cfg: dict,
    rec_results_path: Path,
    model,
    scaling: TorchScaling,
    opt_cfg: dict,
    extend_bounds: bool = True,
):
    """Run or load reconstruction, export visualization files, return parameters."""
    use_parameter = rec_cfg["reuse_parameter"]
    recon_parameter_file = rec_results_path / "rec_parameters.pt"

    if recon_parameter_file.exists() and use_parameter:
        print("Loading existing reconstruction parameters...")
        recon_param = torch.load(recon_parameter_file, map_location=rec_cfg["device"])
        print("Loaded parameters:", recon_param[0].shape, recon_param[0].min().item(), recon_param[0].max().item())
    else:
        print("Running reconstruction...")

        # Run reconstruction in float32 — the DeepSDF decoder and
        # lattice bounds must match the decoder's float32 weights.
        saved_bounds = lattice_struct.bounds.data
        lattice_struct.bounds.data = lattice_struct.bounds.data.float()

        recon_result = run_sdf_reconstruction(
            lattice_struct,
            mesh_norm,
            box_norm.float(),
            rec_cfg,
            output_dir=rec_results_path,
            save_vtp=True,
            box_constrained=True,
        )

        recon_param = recon_result["params"]

        lattice_struct.bounds.data = saved_bounds

        print("Reconstructed parameters:", recon_param[0].shape, recon_param[0].min().item(), recon_param[0].max().item())
        torch.save(recon_param, recon_parameter_file)

    # Export SDF grid and reconstructed meshes in float32
    # (decoder weights, FlexiCubes internals, and FlexiCubes grid
    # construction via torch.get_default_dtype() all require float32)
    mesh_res = opt_cfg["mesh_resolution"]
    device = rec_cfg["device"]

    def _export_meshes(box_norm_f32):
        export_sdf_grid_vtk(
            lattice_struct, N=64,
            filename=rec_results_path / "reconstructed_sdf_grid.vtk",
            bounds=box_norm_f32,
        )

        mesh_ps, deriv_ps = create_3D_mesh(
            lattice_struct, mesh_res,
            mesh_type="surface", differentiate=False,
            device=device, bounds=box_norm_f32,
            extend_bounds=extend_bounds
        )
        export_surface_mesh(rec_results_path / "reconstructed_mesh_parameterspace.stl", mesh_ps.to_gus(), deriv_ps)

        mesh_phys, deriv_phys = create_3D_mesh(
            lattice_struct, mesh_res,
            mesh_type="surface", differentiate=False,
            device=device, bounds=box_norm_f32,
            deformation_function=scaling,
            extend_bounds=extend_bounds
        )
        export_surface_mesh(rec_results_path / "reconstructed_mesh.stl", mesh_phys.to_gus(), deriv_phys)
        print(f"Reconstructed mesh saved to {rec_results_path / 'reconstructed_mesh.stl'}")

    lattice_struct.parametrization.set_param(
        recon_param[0].to(device=model.device, dtype=torch.float32)
    )
    _with_float32_lattice(lattice_struct, box_norm, _export_meshes)

    # Restore float64 parameters on the lattice for the MMA optimizer
    recon_param = [p.to(device=model.device, dtype=DTYPE) for p in recon_param]
    lattice_struct.parametrization.set_param(recon_param[0])
    return recon_param


# ---------------------------------------------------------------------------
# Phase 4: Optimizer setup
# ---------------------------------------------------------------------------

@dataclass
class OptSetup:
    param: torch.nn.Parameter
    mask_locked_cp: torch.Tensor
    mask_locked_flat: torch.Tensor
    locked_values: torch.Tensor
    optimizer: MMA


def setup_optimizer(
    lattice_struct: LatticeSDFStruct,
    param_spline_sp: splinepy.BSpline,
    opt_cfg: dict,
    rec_cfg: dict,
    lock_bboxes: list,
    n_constraints: int = 1,
) -> OptSetup:
    """Create the MMA optimizer with locked control points."""
    param = next(lattice_struct.parametrization.parameters())

    locked_idx = locked_indices_from_bboxes(
        param_spline_sp, lock_bboxes,
        device=rec_cfg["device"], order="F",
    )
    print(f"Locked control points: {locked_idx.numel()}")

    mask_locked_cp, mask_locked_flat, locked_values = make_locked_masks(param, locked_idx)

    bounds = np.full((param.reshape(-1, 1).shape[0], 2), opt_cfg["bounds"])
    optimizer = MMA(
        param.reshape(-1, 1), bounds,
        max_step=opt_cfg["max_step"],
        n_constraints=n_constraints,
    )

    return OptSetup(
        param=param,
        mask_locked_cp=mask_locked_cp,
        mask_locked_flat=mask_locked_flat,
        locked_values=locked_values,
        optimizer=optimizer,
    )


# ---------------------------------------------------------------------------
# Phase 5: Optimization loop helpers
# ---------------------------------------------------------------------------

def generate_mesh(lattice_struct, opt_cfg, rec_cfg, box_norm, scaling, mesh_type="surface", extend_bounds=True):
    """Generate a mesh from lattice parameters.

    Temporarily casts the lattice parametrization to float32 because the
    DeepSDF decoder and FlexiCubes mesh constructor require float32.
    The float64 parameters are restored afterward for the MMA optimizer.

    Parameters
    ----------
    mesh_type : str
        ``"surface"`` for triangle mesh or ``"volume"`` for tet mesh.
    """
    def _create(bounds_f32):
        return create_3D_mesh(
            lattice_struct, opt_cfg["mesh_resolution"],
            mesh_type=mesh_type, differentiate=False,
            device=rec_cfg["device"], bounds=bounds_f32,
            deformation_function=scaling,
            extend_bounds=extend_bounds
        )

    return _with_float32_lattice(lattice_struct, box_norm, _create)


def run_foam_case(case_dir: Path, mesh, derivative, opt_results_path: Path):
    """Export STL to foam case and run OpenFOAM."""
    export_surface_mesh(
        case_dir / "constant/triSurface/shape.stl",
        mesh.to_gus(), derivative,
    )

    foam_case = foam_utils.run_openfoam_case(case_dir, verbose=False)

    log_path = case_dir / "log.adjointOptimisationFoam"
    if log_path.exists():
        plot_residuals_from_log(log_path, output_dir=str(opt_results_path))

    return foam_case


def load_sensitivities(
    case_dir: Path,
    foam_case,
    verts: torch.Tensor,
    field_name: str = "pointSensVecadjS1ESI",
    objective_path: str = "optimisation/objective/0/dragadjS1",
    loading_method: str = "interpolate",
    time_index: int = -1,
    **kwargs,
):
    """Load sensitivities from OpenFOAM results and map to mesh vertices.

    loading_method: "interpolate", "map", "project", or "conservative"
    time_index: which OpenFOAM time step to use (default: -1 = last)

    For "project" and "conservative", extra kwargs:
        patch_name (str): boundary patch to read faces from (default "dragObject")
    For "conservative", extra kwargs:
        faces (torch.Tensor): STL face connectivity, required for scatter
    """
    time_step = foam_case[time_index]
    return_diagnostics = bool(kwargs.get("return_diagnostics", False))
    sens_diag: dict[str, Any] = {}

    if loading_method == "interpolate":
        coords, sensitivities, _, _ = foam_utils.load_mesh_coords_and_sensitivities(
            case_dir, field_name=field_name, time_step=time_step,
        )
        sens_on_verts = foam_utils.interpolate_sensitivities_to_vertices(
            coords, sensitivities, verts,
            warn_tol=kwargs.get("warn_tol", 1e-2),
        )
    elif loading_method == "project":
        coords, sensitivities, _, _ = foam_utils.load_mesh_coords_and_sensitivities(
            case_dir, field_name=field_name, time_step=time_step,
        )
        patch_faces = foam_utils.load_boundary_patch_faces(
            case_dir, patch_name=kwargs.get("patch_name", "dragObject"),
        )
        sens_on_verts = foam_utils.project_sensitivities_to_vertices(
            coords, patch_faces, sensitivities, verts,
            warn_tol=kwargs.get("warn_tol", 1e-2),
        )
    elif loading_method == "conservative":
        coords, sensitivities, _, _ = foam_utils.load_mesh_coords_and_sensitivities(
            case_dir, field_name=field_name, time_step=time_step,
        )
        patch_faces = foam_utils.load_boundary_patch_faces(
            case_dir, patch_name=kwargs.get("patch_name", "dragObject"),
        )
        faces = kwargs.get("faces")
        if faces is None:
            raise ValueError("loading_method='conservative' requires faces kwarg")
        if return_diagnostics:
            sens_on_verts, sens_diag = foam_utils.conservative_sensitivity_transfer(
                coords, patch_faces, sensitivities, verts, faces,
                warn_tol=kwargs.get("warn_tol", 1e-2),
                return_diagnostics=True,
            )
        else:
            sens_on_verts = foam_utils.conservative_sensitivity_transfer(
                coords, patch_faces, sensitivities, verts, faces,
                warn_tol=kwargs.get("warn_tol", 1e-2),
            )
    elif loading_method == "map":
        coords, sensitivities, _, _ = foam_utils.load_sampled_surface(
            case_dir, field_name=field_name, time_step=time_step,
        )
        sens_on_verts = foam_utils.map_sensitivities_to_vertices(
            coords, sensitivities, verts,
            tol=kwargs.get("tol", 1e-6),
        )
    else:
        raise ValueError(f"Unknown loading_method: {loading_method}")

    J = foam_utils.read_objective(case_dir, objective_path=objective_path)
    if return_diagnostics:
        return sens_on_verts, J, sens_diag
    return sens_on_verts, J


def mask_gradients(grads: list[torch.Tensor], mask_locked_flat: torch.Tensor):
    """Zero out gradient entries for locked control points.

    grads: list of gradient tensors (dJ, dV, dC, ...)
    Returns list of masked flat tensors.
    """
    masked = []
    for g in grads:
        g_flat = g.reshape(-1).clone()
        g_flat[mask_locked_flat] = 0.0
        masked.append(g_flat)
    return masked


def log_iteration_time(iter_start: float, start_time: float, iteration_times: list, total_iters: int, current_iter: int):
    """Print timing info for the current iteration."""
    iter_time = time.time() - iter_start
    iteration_times.append(iter_time)

    avg_time = sum(iteration_times) / len(iteration_times)
    remaining = total_iters - (current_iter + 1)
    eta = avg_time * remaining
    elapsed = time.time() - start_time

    print(f"[TIME] Iteration took: {iter_time:.2f}s")
    print(f"[TIME] Avg/iter: {avg_time:.2f}s")
    print(f"[TIME] Elapsed: {elapsed/60:.2f} min")
    print(f"[TIME] ETA remaining: {eta/60:.2f} min ({eta/3600:.2f} h)")

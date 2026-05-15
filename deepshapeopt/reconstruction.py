from __future__ import annotations

"""
Reconstruction utilities and standalone reconstruction pipeline.

Provides:
- Building blocks for B-spline parametrized SDF reconstruction
  (used by both standalone reconstruction experiments and the optimization pipeline)
- A complete `reconstruct_shape` function for standalone reconstruction experiments
"""
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import splinepy
import torch

if TYPE_CHECKING:
    import trimesh
    from DeepSDFStruct.lattice_structure import LatticeSDFStruct
    from deepshapeopt.config import ExperimentSpecifications


# ---------------------------------------------------------------------------
# Building blocks (used by both standalone reconstruction and optimization)
# ---------------------------------------------------------------------------

def build_parameter_spline(
    spline_degrees: list[int],
    tiling: tuple[int, int, int],
    latent_dim: int,
    bounds: np.ndarray | None = None,
) -> splinepy.BSpline:
    """Build a parameter BSpline with specified degrees and tiling.

    Parameters
    ----------
    spline_degrees : list of 3 ints
        Polynomial degree per spatial dimension.
    tiling : tuple of 3 ints
        Number of knot spans (boxes) per dimension. Knots are inserted
        uniformly within each dimension.
    latent_dim : int
        Dimensionality of the control point vectors (e.g. latent vector size).
    bounds : (2, 3) array-like or None
        [[xmin, ymin, zmin], [xmax, ymax, zmax]]. Defines the physical
        domain of the spline. If None, defaults to [0, 1]^3.
    """
    if bounds is not None:
        bounds = np.asarray(bounds)
        mins = bounds[0]
        maxs = bounds[1]
    else:
        mins = np.array([0.0, 0.0, 0.0])
        maxs = np.array([1.0, 1.0, 1.0])

    knot_vectors = [
        [mins[i]] * (spline_degrees[i] + 1) + [maxs[i]] * (spline_degrees[i] + 1)
        for i in range(3)
    ]

    n_ctrl_per_dim = [len(knot_vectors[i]) - spline_degrees[i] - 1 for i in range(3)]
    n_ctrl_total = int(np.prod(n_ctrl_per_dim))
    control_points = np.zeros((n_ctrl_total, latent_dim))

    param_spline_sp = splinepy.BSpline(
        spline_degrees,
        knot_vectors,
        control_points,
    )

    for i_box, n_box in enumerate(tiling):
        knots = np.linspace(mins[i_box], maxs[i_box], n_box + 1)[1:-1]
        if len(knots) == 0:
            continue
        print(f"Inserting {n_box - 1} knots at {knots} into spline dim {i_box}")
        param_spline_sp.insert_knots(i_box, knots)

    return param_spline_sp


def init_spline_parameters(param_spline, mean=0.0, std=0.001):
    """Initialize all trainable parameters of the spline."""
    for p in param_spline.parameters():
        torch.nn.init.normal_(p, mean=mean, std=std)


def run_sdf_reconstruction(
    lattice_struct: LatticeSDFStruct,
    mesh: trimesh.Trimesh,
    bounds: torch.Tensor,
    rec_cfg: dict,
    output_dir: Path,
    lightweight_output_dir: Path | None = None,
    save_vtp: bool = True,
    use_mlflow: bool = False,
    mlflow_metric_prefix: str | None = None,
    mlflow_log_every_n_steps: int = 10,
    box_constrained: bool = True,
):
    """Core SDF reconstruction: sample ground truth, fit, and export.

    This is the single reconstruction entry point used by both the standalone
    reconstruction pipeline (`reconstruct_shape`) and the optimization
    pipeline (`shape_optimization.run_reconstruction`).

    Parameters
    ----------
    lattice_struct : LatticeSDFStruct
        Lattice structure with parametrization already initialized.
    mesh : trimesh.Trimesh
        Ground truth mesh (already in the target coordinate system).
    bounds : torch.Tensor
        (2, 3) bounding box for sampling.
    rec_cfg : dict
        Reconstruction config section. Expected keys: ``lr``,
        ``num_iterations``, ``batch_size``, ``n_uniform_samples``,
        ``n_surface_samples``. Optional: ``samples_surface_stds``,
        ``code_reg_lambda``, ``code_bound``, ``grad_clip``,
        ``eikonal_lambda``, ``device``.
    output_dir : Path
        Directory for heavy data exports (VTP files).
    lightweight_output_dir : Path or None
        Directory for lightweight outputs (loss plot, loss CSV).
        If None, defaults to ``output_dir``.
    save_vtp : bool
        Whether to export VTP sample files.
    use_mlflow : bool
        Whether to log metrics to MLflow.
    mlflow_metric_prefix : str or None
        Prefix for MLflow metric names.
    mlflow_log_every_n_steps : int
        How often to log reconstruction loss to MLflow.
    box_constrained : bool
        If True, clamp surface samples to bounds before fitting.

    Returns
    -------
    dict with keys: ``params``, ``final_loss``, ``num_steps``,
    ``surface_samples`` (SampledSDF, only if save_vtp).
    """
    from DeepSDFStruct.deep_sdf.reconstruction import reconstruct_from_samples
    from DeepSDFStruct.sampling import save_points_to_vtp

    device = rec_cfg.get("device", "cuda")
    stds = rec_cfg.get("samples_surface_stds", [0.025, 0.0001])
    n_uniform_samples = int(rec_cfg.get("n_uniform_samples", 100000))
    n_surface_samples = int(rec_cfg.get("n_surface_samples", 500000))
    lr = float(rec_cfg["lr"])
    num_iterations = int(rec_cfg["num_iterations"])
    batch_size = int(rec_cfg["batch_size"])
    code_reg_lambda = float(rec_cfg.get("code_reg_lambda", 0.0))
    code_bound = rec_cfg.get("code_bound", None)
    grad_clip = rec_cfg.get("grad_clip", None)
    eikonal_lambda = float(rec_cfg.get("eikonal_lambda", 0.0))

    output_dir = Path(output_dir)
    lightweight_output_dir = Path(lightweight_output_dir) if lightweight_output_dir is not None else output_dir

    # --- Sample ground truth SDF ---
    sdf_samples = sample_sdf(
        mesh, bounds,
        n_uniform_samples=n_uniform_samples,
        n_surface_samples=n_surface_samples,
        stds=stds,
        device=device,
        box_constrained=box_constrained,
    )

    surface_samples = None
    if save_vtp:
        gt_points_all = torch.hstack(
            (sdf_samples.samples.detach(), sdf_samples.distances.detach())
        )
        save_points_to_vtp(output_dir / "gt_sdf_samples.vtp", gt_points_all)

    # --- Run fitting ---
    recon_result = reconstruct_from_samples(
        lattice_struct,
        sdf_samples,
        lr=lr,
        loss_fn="ClampedL1",
        num_iterations=num_iterations,
        batch_size=batch_size,
        use_tanh_on_gt=False,
        loss_plot_path=lightweight_output_dir / "loss_plot.png",
        loss_csv_path=lightweight_output_dir / "loss_history.csv",
        optimizer_name="adam",
        deformation_function=None,
        use_mlflow=use_mlflow,
        mlflow_metric_prefix=mlflow_metric_prefix,
        mlflow_log_every_n_steps=mlflow_log_every_n_steps,
        code_reg_lambda=code_reg_lambda,
        code_bound=code_bound,
        grad_clip=grad_clip,
        eikonal_lambda=eikonal_lambda,
    )

    # --- Export reconstructed SDF samples ---
    if save_vtp:
        samples_ps = sdf_samples.samples.detach()
        rec_dist = lattice_struct(samples_ps)
        rec_points = torch.hstack((samples_ps, rec_dist.detach()))
        save_points_to_vtp(output_dir / "rec_sdf_samples.vtp", rec_points)

    return recon_result


def sample_sdf(mesh, bounds, n_uniform_samples, n_surface_samples, device, stds, box_constrained=True):
    """Sample points and SDF values from the ground truth mesh.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Ground truth mesh (should already be in the target coordinate system).
    bounds : torch.Tensor
        (2, 3) bounding box for uniform sampling.
    n_uniform_samples : int
        Number of uniform random samples within bounds.
    n_surface_samples : int
        Number of near-surface samples.
    device : str
        Torch device.
    stds : list of 2 floats
        Standard deviations for surface noise perturbation.
    box_constrained : bool
        If True, use box-constrained surface sampling (clips samples to bounds).
        If False, use unconstrained surface sampling.
    """
    from DeepSDFStruct.SDF import SDFfromMesh
    from DeepSDFStruct.sampling import SampledSDF, random_sample_sdf, sample_mesh_surface

    gt_sdf = SDFfromMesh(mesh, scale=False)

    uniform_samples = random_sample_sdf(
        gt_sdf,
        bounds,
        n_samples=n_uniform_samples,
        type="uniform",
        device=device,
    )

    surface_samples = sample_mesh_surface(
        gt_sdf,
        mesh,
        n_samples=n_surface_samples,
        stds=stds,
        device=device,
    )

    if box_constrained:
        # Rejection sampling: keep only surface samples inside bounds
        bmin = bounds[0].to(surface_samples.samples.device)
        bmax = bounds[1].to(surface_samples.samples.device)

        inside = ((surface_samples.samples >= bmin) & (surface_samples.samples <= bmax)).all(dim=1)
        n_accepted = inside.sum().item()
        acceptance_rate = n_accepted / surface_samples.samples.shape[0] if n_accepted > 0 else 0.0

        all_samples = [surface_samples.samples[inside]]
        all_distances = [surface_samples.distances[inside]]
        n_collected = n_accepted

        max_rounds = 10
        for _ in range(max_rounds):
            if n_collected >= n_surface_samples:
                break
            n_needed = n_surface_samples - n_collected
            oversample_factor = max(1.0 / max(acceptance_rate, 0.01), 2.0)
            n_to_sample = int(n_needed * oversample_factor * 1.5)

            extra = sample_mesh_surface(gt_sdf, mesh, n_samples=n_to_sample, stds=stds, device=device)
            inside_extra = ((extra.samples >= bmin) & (extra.samples <= bmax)).all(dim=1)
            all_samples.append(extra.samples[inside_extra])
            all_distances.append(extra.distances[inside_extra])

            n_new = inside_extra.sum().item()
            n_collected += n_new
            if n_new > 0:
                acceptance_rate = n_new / extra.samples.shape[0]

        final_samples = torch.cat(all_samples, dim=0)[:n_surface_samples]
        final_distances = torch.cat(all_distances, dim=0)[:n_surface_samples]
        print(
            f"Box-constrained sampling (rejection): {final_samples.shape[0]}/{n_surface_samples} "
            f"surface samples inside bounds (acceptance rate ~{acceptance_rate:.1%})"
        )
        if final_samples.shape[0] < n_surface_samples:
            print(f"WARNING: only collected {final_samples.shape[0]} of {n_surface_samples} requested surface samples inside bounds")
        surface_samples = SampledSDF(samples=final_samples, distances=final_distances)

    return uniform_samples + surface_samples


def fit_box_to_unit_cube(box_bounds: torch.Tensor, eps: float = 1e-12):
    """
    box_bounds: (2,3) tensor [[xmin,ymin,zmin],[xmax,ymax,zmax]]
    Returns:
      scale: float tensor scalar
      center: (3,) tensor
      normalize_fn(points): applies (points-center)*scale
      denormalize_fn(points): applies points/scale + center
      box_bounds_norm: (2,3) tensor
    """

    bmin, bmax = box_bounds[0], box_bounds[1]

    center = 0.5 * (bmin + bmax)
    size = (bmax - bmin).clamp_min(eps)  # avoid zero-length issues
    L = torch.max(size)                  # uniform reference length

    scale = 2.0 / L

    def normalize_fn(points: torch.Tensor) -> torch.Tensor:
        return (points - center) * scale

    def denormalize_fn(points: torch.Tensor) -> torch.Tensor:
        return points / scale + center

    box_bounds_norm = torch.stack([normalize_fn(bmin), normalize_fn(bmax)], dim=0)
    return 1/scale, center, normalize_fn, denormalize_fn, box_bounds_norm


# ---------------------------------------------------------------------------
# Private helpers for reconstruct_shape
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Standalone reconstruction pipeline
# ---------------------------------------------------------------------------

def reconstruct_shape(
    experiment_path: Path,
    specs: ExperimentSpecifications,
    case_name: str | None = None,
    use_mlflow: bool = False,
    mlflow_metric_prefix: str | None = None,
    mlflow_log_every_n_steps: int = 10,
    verbose: bool = True,
    save_vtp: bool = True,
):
    """Run a standalone shape reconstruction experiment.

    Loads a DeepSDF model, samples the ground truth SDF, builds a
    parametrized B-spline lattice, and runs reconstruction optimization
    to fit the lattice to the SDF samples.

    Parameters
    ----------
    experiment_path : Path
        Root directory of the experiment (contains config.json).
    specs : ExperimentSpecifications
        Experiment configuration with nested ``reconstruction`` section.
    case_name : str or None
        Identifier for this run. Auto-generated from mesh name and tiling
        if not provided.
    """
    import trimesh
    from DeepSDFStruct.pretrained_models import get_model
    from DeepSDFStruct.SDF import SDFfromDeepSDF, SDFfromMesh, normalize_mesh_to_unit_cube
    from DeepSDFStruct.mesh import create_3D_mesh, export_surface_mesh, export_sdf_grid_vtk
    from DeepSDFStruct.lattice_structure import LatticeSDFStruct
    from DeepSDFStruct.parametrization import SplineParametrization
    from DeepSDFStruct.torch_spline import TorchScaling
    from DeepSDFStruct.sampling import sample_mesh_surface, save_points_to_vtp
    from DeepSDFStruct.export_knot_grid import (
        export_knot_grid_paramspace,
        export_control_lattice_paramspace,
    )
    from DeepSDFStruct.deep_sdf.metrics.error_metrics import compute_metrics_from_vtp
    from deepshapeopt.analysis import (
        add_vertex_colors_from_scalar,
        compute_vertex_sdf_error,
        trimesh_to_pyvista,
    )
    from deepshapeopt.config import make_experiment_paths, ensure_experiment_dirs

    experiment_path = Path(experiment_path).resolve()
    rec_cfg = specs["reconstruction"]

    # --- Config fields ---
    device = rec_cfg.get("device", "cuda")
    mesh_device = rec_cfg.get("mesh_device", device)
    tiling = rec_cfg["tiling"]
    spline_degree = rec_cfg.get("spline_degree", [1, 1, 1])
    create_mesh_N = int(rec_cfg["create_mesh_N"])
    n_surface_samples = int(rec_cfg.get("n_surface_samples", 500000))
    samples_surface_stds = rec_cfg.get("samples_surface_stds", [0.025, 0.0001])

    # --- Paths ---
    results_name = specs.get("results_name", "results")
    heavy_data_output_path = rec_cfg.get("heavy_data_output_path")
    paths = make_experiment_paths(
        experiment_path,
        results_name=results_name,
        heavy_data_output_path=heavy_data_output_path,
    )
    ensure_experiment_dirs(paths)

    # --- Case name ---
    mesh_path = Path(rec_cfg["mesh_path"]).resolve()
    model_path = str(rec_cfg["model_path"])
    checkpoint = str(rec_cfg.get("model_checkpoint", "latest"))

    if case_name is None:
        mesh_stem = mesh_path.stem
        tiling_str = "x".join(map(str, tiling))
        case_name = f"{mesh_stem}_tiling_{tiling_str}"

    # --- Output directories ---
    local_dir = paths.reconstruction / case_name
    local_dir.mkdir(parents=True, exist_ok=True)

    if paths.heavy_data is not None:
        heavy_dir = paths.heavy_data / "reconstruction" / case_name
        heavy_dir.mkdir(parents=True, exist_ok=True)
    else:
        heavy_dir = local_dir

    # Save config snapshot
    _write_json(local_dir / "specs_reconstruction.json", dict(specs))

    run_info = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "experiment_dir": str(experiment_path),
        "case_name": case_name,
        "resolved_mesh_path": str(mesh_path),
        "status": "started",
    }
    _write_json(local_dir / "specs_summary.json", run_info)

    # --- CUDA check ---
    if "cuda" in str(device) or "cuda" in str(mesh_device):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested in config, but torch.cuda.is_available() is False."
            )

    # --- Load and normalize mesh ---
    mesh_orig = trimesh.load_mesh(str(mesh_path))
    mesh, scale, shift = normalize_mesh_to_unit_cube(mesh_orig.copy())
    if verbose:
        mesh.export(str(heavy_dir / "input_mesh_normalized.stl"))

    bounds = torch.tensor(mesh.bounds, device=device, dtype=torch.float32)
    scaling = TorchScaling(
        scale_factors=scale,
        translation=shift,
        bounds=bounds,
        device=device,
    )

    # --- Load model ---
    model = get_model(model_path, checkpoint=checkpoint)
    sdf = SDFfromDeepSDF(model)

    # --- Build spline ---
    mins = bounds[0].detach().cpu().numpy()
    maxs = bounds[1].detach().cpu().numpy()
    latent_dim = model._trained_latent_vectors[0].shape[0]

    param_spline_sp = build_parameter_spline(
        spline_degrees=spline_degree,
        tiling=tiling,
        latent_dim=latent_dim,
        bounds=np.stack([mins, maxs]),
    )

    if verbose:
        export_knot_grid_paramspace(
            param_spline_sp, filename=str(heavy_dir / "knot_grid.vtp")
        )
        export_control_lattice_paramspace(
            param_spline_sp,
            filename=str(heavy_dir / "control_lattice_paramspace.vtp"),
            order="F",
        )

    param_spline = SplineParametrization(param_spline_sp, device=model.device)
    init_spline_parameters(param_spline, mean=0.0, std=0.001)

    lattice_struct = LatticeSDFStruct(
        tiling=tiling, microtile=sdf, parametrization=param_spline, bounds=bounds
    )

    # --- Reconstruct ---
    metric_prefix = mlflow_metric_prefix or f"reconstruction_{case_name}"

    recon_result = run_sdf_reconstruction(
        lattice_struct,
        mesh,
        bounds,
        rec_cfg,
        output_dir=heavy_dir,
        lightweight_output_dir=local_dir,
        save_vtp=save_vtp,
        use_mlflow=use_mlflow,
        mlflow_metric_prefix=metric_prefix,
        mlflow_log_every_n_steps=mlflow_log_every_n_steps,
        box_constrained=False,
    )

    lattice_struct.parametrization.set_param(recon_result["params"][0])

    # --- Post-reconstruction exports (standalone-specific) ---
    metrics = None
    if save_vtp:
        # Surface-only samples for error metrics
        gt_sdf_obj = SDFfromMesh(mesh, scale=False)
        surface_samples = sample_mesh_surface(
            gt_sdf_obj, mesh,
            n_samples=n_surface_samples,
            stds=samples_surface_stds,
            device=device,
        )
        gt_points_surface = torch.hstack(
        (surface_samples.samples.detach(), surface_samples.distances.detach())
        )
        save_points_to_vtp(heavy_dir / "gt_sdf_samples_surface.vtp", gt_points_surface)

        samples_surface = surface_samples.samples.detach()
        rec_dist_surface = lattice_struct(samples_surface)
        rec_points_surface = torch.hstack(
        (samples_surface, rec_dist_surface.detach())
        )
        save_points_to_vtp(
            heavy_dir / "rec_sdf_samples_surface.vtp", rec_points_surface
        )

        # Compute reconstruction error metrics on SDF samples (neural SDF error)
        error_cutoff = rec_cfg.get("error_cutoff", 0.1)
        metrics = compute_metrics_from_vtp(
            gt_vtp_path=heavy_dir / "gt_sdf_samples_surface.vtp",
            pred_vtp_path=heavy_dir / "rec_sdf_samples_surface.vtp",
            cutoff=error_cutoff,
            output_json_path=None,
        )

        export_sdf_grid_vtk(
            lattice_struct,
            N=64,
            filename=str(heavy_dir / "reconstructed_sdf_grid.vtk"),
            bounds=bounds,
        )

    # --- Generate output mesh ---
    lattice_struct = lattice_struct.to(mesh_device)
    scaling = scaling.to(mesh_device)
    lattice_struct.microtile.model.device = mesh_device
    lattice_struct.microtile.model._decoder = (
        lattice_struct.microtile.model._decoder.to(mesh_device)
    )

    surf_mesh, derivative = create_3D_mesh(
        lattice_struct,
        create_mesh_N,
        differentiate=False,
        device=mesh_device,
        mesh_type="surface",
        deformation_function=scaling,
    )

    reconstructed_mesh_filename = f"{mesh_path.stem}_reconstructed.stl"
    reconstructed_mesh_path = heavy_dir / reconstructed_mesh_filename
    export_surface_mesh(str(reconstructed_mesh_path), surf_mesh.to_gus(), derivative)

    # Create reconstruted 3D mesh in parameter space for visualization
    param_surf_mesh, _ = create_3D_mesh(
        lattice_struct,
        create_mesh_N,
        differentiate=False,
        device=mesh_device,
        mesh_type="surface",
        deformation_function=None,  # no scaling, keep in param space
    )
    export_surface_mesh(
        str(heavy_dir / f"{mesh_path.stem}_reconstructed_param_space.stl"),
        param_surf_mesh.to_gus(),
        derivative,
    )

    # --- Mesh-vs-mesh vertex SDF error (explicit reconstructed mesh vs GT) ---
    mesh_metrics = None
    if save_vtp:
        reconstructed_mesh_trimesh = trimesh.load_mesh(
            str(reconstructed_mesh_path), force="mesh"
        )
        reconstructed_mesh_norm, mesh_sdf_error = compute_vertex_sdf_error(
            mesh_orig, reconstructed_mesh_trimesh
        )

        mesh_abs_error = np.abs(mesh_sdf_error)
        mesh_metrics = {
            "num_vertices": int(reconstructed_mesh_norm.vertices.shape[0]),
            "mae": float(mesh_abs_error.mean()),
            "rmse": float(np.sqrt(np.mean(mesh_sdf_error ** 2))),
            "median": float(np.median(mesh_abs_error)),
            "p95": float(np.quantile(mesh_abs_error, 0.95)),
            "max": float(mesh_abs_error.max()),
            "min_signed_error": float(mesh_sdf_error.min()),
            "max_signed_error": float(mesh_sdf_error.max()),
            "mean_signed_error": float(mesh_sdf_error.mean()),
        }

        mesh_error_poly = trimesh_to_pyvista(reconstructed_mesh_norm)
        mesh_error_poly.point_data["sdf_error"] = mesh_sdf_error
        add_vertex_colors_from_scalar(
            mesh_error_poly, scalar_name="sdf_error", cmap_name="turbo"
        )
        mesh_error_vtp_path = heavy_dir / f"{mesh_path.stem}_mesh_sdf_error.vtp"
        mesh_error_poly.save(str(mesh_error_vtp_path))

        print("\nMesh-vertex SDF error (explicit mesh vs GT mesh)")
        print("------------------------------------------------")
        print(f"num vertices: {mesh_metrics['num_vertices']}")
        print(f"mae:          {mesh_metrics['mae']:.8e}")
        print(f"rmse:         {mesh_metrics['rmse']:.8e}")
        print(f"median:       {mesh_metrics['median']:.8e}")
        print(f"p95:          {mesh_metrics['p95']:.8e}")
        print(f"Saved mesh-error VTP to: {mesh_error_vtp_path}")

    # --- Write unified error metrics JSON ---
    if metrics is not None or mesh_metrics is not None:
        combined_error_metrics = {
            "sdf_sample_error": metrics,
            "mesh_vertex_error": mesh_metrics,
        }
        _write_json(local_dir / "error_metrics.json", combined_error_metrics)
        print(f"\nSaved unified error metrics to: {local_dir / 'error_metrics.json'}")

    # --- MLflow logging ---
    if use_mlflow:
        import mlflow

    if use_mlflow and mlflow.active_run() is not None:
        mlflow.log_param(f"{metric_prefix}_mesh_path", str(mesh_path))
        mlflow.log_param(f"{metric_prefix}_tiling", str(tiling))
        mlflow.log_param(f"{metric_prefix}_checkpoint", checkpoint)

        if recon_result["final_loss"] is not None:
            mlflow.log_metric(f"{metric_prefix}_final_loss", recon_result["final_loss"])

        mlflow.log_artifact(
            str(local_dir / "specs_reconstruction.json"),
            artifact_path=f"reconstruction/{case_name}/config",
        )
        mlflow.log_artifact(
            str(local_dir / "loss_plot.png"),
            artifact_path=f"reconstruction/{case_name}",
        )
        mlflow.log_artifact(
            str(reconstructed_mesh_path),
            artifact_path=f"reconstruction/{case_name}",
        )

        if metrics is not None:
            for key in ("mae", "rmse", "median", "p95"):
                mlflow.log_metric(f"{metric_prefix}_sdf_{key}", metrics[key])

        if mesh_metrics is not None:
            for key in ("mae", "rmse", "median", "p95"):
                mlflow.log_metric(f"{metric_prefix}_mesh_{key}", mesh_metrics[key])

        if metrics is not None or mesh_metrics is not None:
            mlflow.log_artifact(
                str(local_dir / "error_metrics.json"),
                artifact_path=f"reconstruction/{case_name}",
            )

    # --- Save summary ---
    run_info["status"] = "finished"
    run_info["mesh_bounds"] = bounds.tolist()
    run_info["final_loss"] = recon_result["final_loss"]
    run_info["num_steps"] = recon_result["num_steps"]
    run_info["reconstructed_mesh_path"] = str(reconstructed_mesh_path)
    _write_json(local_dir / "specs_summary.json", run_info)

    print("Done.")
    print("Results in:", local_dir)
    if heavy_dir != local_dir:
        print("Heavy data in:", heavy_dir)

    return {
        "case_name": case_name,
        "results_dir": str(local_dir),
        "heavy_data_dir": str(heavy_dir),
        "reconstructed_mesh_path": str(reconstructed_mesh_path),
        "final_loss": recon_result["final_loss"],
        "num_steps": recon_result["num_steps"],
        "metrics": metrics if save_vtp else None,
        "mesh_metrics": mesh_metrics if save_vtp else None,
    }

from __future__ import annotations

"""
Reconstruction utilities and the standalone reconstruction pipeline.

Public API:
- ``fit_lattice_to_sdf``: core fit step — sample the ground-truth SDF,
  call ``reconstruct_from_samples``, and export sample VTPs. Shared by both
  the standalone pipeline and the in-optimization phase.
- ``reconstruct_shape``: standalone pipeline used by ``scripts/reconstruct.py``.
- ``export_reconstructed_artifacts`` / ``with_float32_lattice``: post-fit
  export helpers shared with ``shape_optimization.run_reconstruction``
  (the in-optimization reconstruction phase, which lives there because it
  also handles ``reuse_parameter`` caching).
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import splinepy
import torch

logger = logging.getLogger(__name__)

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
        logger.debug("Inserting %d knots at %s into spline dim %d", n_box - 1, knots, i_box)
        param_spline_sp.insert_knots(i_box, knots)

    return param_spline_sp


def init_spline_parameters(param_spline, mean=0.0, std=0.001):
    """Initialize all trainable parameters of the spline."""
    for p in param_spline.parameters():
        torch.nn.init.normal_(p, mean=mean, std=std)


def fit_lattice_to_sdf(
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
    samples_series_dir: Path | None = None,
):
    """Core fit step: sample ground-truth SDF, fit the lattice, export VTPs.

    Shared by both the standalone pipeline (`reconstruct_shape`) and the
    in-optimization phase (`shape_optimization.run_reconstruction`).

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
    samples_series_dir : Path or None
        If set and ``rec_cfg["export_rec_samples_series"]`` is truthy, export a
        contiguously numbered ``rec_sdf_samples_{frame:04d}.vtp`` series here
        (frame 0000 = initial field). By default one frame per epoch; set
        ``rec_cfg["export_rec_samples_fine_until_epoch"]`` > 0 to also export
        between batch steps for those first epochs (every
        ``export_rec_samples_fine_every`` batches, default 1).

    Returns
    -------
    dict with keys: ``params``, ``final_loss``, ``num_steps``,
    ``surface_samples`` (SampledSDF, only if save_vtp).
    """
    from DeepSDFStruct.deep_sdf.reconstruction import reconstruct_from_samples
    from DeepSDFStruct.sampling import save_points_to_vtp

    def _export_rec_samples(samples_ps, path):
        """Evaluate the fitted lattice at ``samples_ps`` and write a points VTP."""
        rec_dist = lattice_struct(samples_ps)
        rec_points = torch.hstack((samples_ps, rec_dist.detach()))
        save_points_to_vtp(path, rec_points)

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

    # --- Per-epoch reconstructed-sample series (heavy debug, off by default) ---
    export_series = (
        bool(rec_cfg.get("export_rec_samples_series", False))
        and samples_series_dir is not None
    )
    step_callback = None
    if export_series:
        samples_series_dir = Path(samples_series_dir)
        samples_series_dir.mkdir(parents=True, exist_ok=True)
        series_samples = sdf_samples.samples.detach()
        # Fine (per-batch) export for the first ``fine_until_epoch`` epochs, then
        # coarse (per-epoch) afterwards. Frames are numbered contiguously so the
        # series loads as a single ParaView time sequence.
        fine_until_epoch = int(rec_cfg.get("export_rec_samples_fine_until_epoch", 0))
        fine_every = max(1, int(rec_cfg.get("export_rec_samples_fine_every", 1)))
        frame = 0

        def _write_frame():
            nonlocal frame
            with torch.no_grad():
                _export_rec_samples(
                    series_samples,
                    samples_series_dir / f"rec_sdf_samples_{frame:04d}.vtp",
                )
            frame += 1

        # Frame 0000: the initial field, before any fitting step.
        _write_frame()

        def step_callback(e, batch_idx, n_batches):
            is_epoch_end = batch_idx == n_batches - 1
            if e < fine_until_epoch:
                do_export = (batch_idx % fine_every == 0) or is_epoch_end
            else:
                do_export = is_epoch_end
            if do_export:
                _write_frame()

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
        step_callback=step_callback,
    )

    # --- Export reconstructed SDF samples ---
    if save_vtp:
        _export_rec_samples(
            sdf_samples.samples.detach(), output_dir / "rec_sdf_samples.vtp"
        )

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
        logger.debug(
            "Box-constrained sampling (rejection): %d/%d surface samples inside bounds (acceptance rate ~%.1f%%)",
            final_samples.shape[0], n_surface_samples, acceptance_rate * 100,
        )
        if final_samples.shape[0] < n_surface_samples:
            logger.warning(
                "Only collected %d of %d requested surface samples inside bounds",
                final_samples.shape[0], n_surface_samples,
            )
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
# Post-fit export helpers (shared by both pipelines)
# ---------------------------------------------------------------------------

def with_float32_lattice(lattice_struct, bounds, fn):
    """Run *fn(bounds_f32)* with the lattice temporarily cast to float32.

    DeepSDF/FlexiCubes mesh generation requires float32; the outer
    optimization keeps parameters in float64. This helper performs the
    local cast around ``fn`` and restores the original dtype afterwards.
    """
    params = list(lattice_struct.parametrization.parameters())
    saved_params = [p.data for p in params]
    for p in params:
        p.data = p.data.float()

    saved_bounds = lattice_struct.bounds.data
    lattice_struct.bounds.data = lattice_struct.bounds.data.float()

    saved_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)

    try:
        return fn(bounds.float())
    finally:
        torch.set_default_dtype(saved_default_dtype)
        for p, s in zip(params, saved_params):
            p.data = s
        lattice_struct.bounds.data = saved_bounds


def export_reconstructed_artifacts(
    lattice_struct,
    output_dir: Path,
    *,
    mesh_resolution: int,
    bounds: torch.Tensor,
    device,
    scaling=None,
    extend_bounds: bool = True,
    sdf_grid_N: int = 64,
    sdf_grid_name: str = "reconstructed_sdf_grid.vtk",
    param_mesh_name: str = "reconstructed_mesh_parameterspace.stl",
    physical_mesh_name: str = "reconstructed_mesh.stl",
    export_sdf_grid: bool = True,
    export_param_mesh: bool = True,
) -> Path:
    """Export SDF grid + param-space STL + (if scaling) physical-space STL.

    Wraps generation in :func:`with_float32_lattice`. Returns the path of the
    physical-space mesh (or the param-space mesh if no ``scaling`` was given).
    """
    from DeepSDFStruct.mesh import create_3D_mesh, export_surface_mesh, export_sdf_grid_vtk

    output_dir = Path(output_dir)

    def _export(bounds_f32):
        if export_sdf_grid:
            export_sdf_grid_vtk(
                lattice_struct, N=sdf_grid_N,
                filename=str(output_dir / sdf_grid_name),
                bounds=bounds_f32,
            )

        if export_param_mesh:
            ps_mesh, ps_deriv = create_3D_mesh(
                lattice_struct, mesh_resolution,
                mesh_type="surface", differentiate=False,
                device=device, bounds=bounds_f32,
                extend_bounds=extend_bounds,
            )
            export_surface_mesh(
                str(output_dir / param_mesh_name), ps_mesh.to_gus(), ps_deriv,
            )

        if scaling is None:
            return output_dir / param_mesh_name

        phys_mesh, phys_deriv = create_3D_mesh(
            lattice_struct, mesh_resolution,
            mesh_type="surface", differentiate=False,
            device=device, bounds=bounds_f32,
            deformation_function=scaling,
            extend_bounds=extend_bounds,
        )
        export_surface_mesh(
            str(output_dir / physical_mesh_name), phys_mesh.to_gus(), phys_deriv,
        )
        return output_dir / physical_mesh_name

    return with_float32_lattice(lattice_struct, bounds, _export)


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


def _compute_reconstruction_metrics(
    *,
    lattice_struct,
    mesh,
    mesh_orig,
    reconstructed_mesh_path: Path,
    heavy_dir: Path,
    mesh_stem: str,
    n_surface_samples: int,
    samples_surface_stds,
    error_cutoff: float,
    device,
):
    """Compute SDF-sample and mesh-vertex error metrics; save accompanying VTPs."""
    import trimesh
    from DeepSDFStruct.SDF import SDFfromMesh
    from DeepSDFStruct.sampling import sample_mesh_surface, save_points_to_vtp
    from DeepSDFStruct.deep_sdf.metrics.error_metrics import compute_metrics_from_vtp
    from deepshapeopt.analysis import (
        add_vertex_colors_from_scalar,
        compute_vertex_sdf_error,
        trimesh_to_pyvista,
    )

    # Sample GT SDF on the surface and save GT + reconstructed sample VTPs.
    gt_sdf_obj = SDFfromMesh(mesh, scale=False)
    surface_samples = sample_mesh_surface(
        gt_sdf_obj, mesh,
        n_samples=n_surface_samples, stds=samples_surface_stds, device=device,
    )
    gt_points = torch.hstack(
        (surface_samples.samples.detach(), surface_samples.distances.detach())
    )
    save_points_to_vtp(heavy_dir / "gt_sdf_samples_surface.vtp", gt_points)

    samples = surface_samples.samples.detach()
    rec_dist = lattice_struct(samples)
    rec_points = torch.hstack((samples, rec_dist.detach()))
    save_points_to_vtp(heavy_dir / "rec_sdf_samples_surface.vtp", rec_points)

    sdf_metrics = compute_metrics_from_vtp(
        gt_vtp_path=heavy_dir / "gt_sdf_samples_surface.vtp",
        pred_vtp_path=heavy_dir / "rec_sdf_samples_surface.vtp",
        cutoff=error_cutoff,
        output_json_path=None,
    )

    # Mesh-vertex SDF error: reconstructed mesh evaluated against the GT mesh.
    reconstructed_trimesh = trimesh.load_mesh(str(reconstructed_mesh_path), force="mesh")
    reconstructed_norm, mesh_sdf_error = compute_vertex_sdf_error(
        mesh_orig, reconstructed_trimesh
    )
    abs_err = np.abs(mesh_sdf_error)
    mesh_metrics = {
        "num_vertices": int(reconstructed_norm.vertices.shape[0]),
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt(np.mean(mesh_sdf_error ** 2))),
        "median": float(np.median(abs_err)),
        "p95": float(np.quantile(abs_err, 0.95)),
        "max": float(abs_err.max()),
        "min_signed_error": float(mesh_sdf_error.min()),
        "max_signed_error": float(mesh_sdf_error.max()),
        "mean_signed_error": float(mesh_sdf_error.mean()),
    }

    mesh_error_poly = trimesh_to_pyvista(reconstructed_norm)
    mesh_error_poly.point_data["sdf_error"] = mesh_sdf_error
    add_vertex_colors_from_scalar(mesh_error_poly, scalar_name="sdf_error", cmap_name="turbo")
    mesh_error_vtp_path = heavy_dir / f"{mesh_stem}_mesh_sdf_error.vtp"
    mesh_error_poly.save(str(mesh_error_vtp_path))

    logger.debug(
        "Mesh-vertex SDF error: n=%d mae=%.6e rmse=%.6e median=%.6e p95=%.6e (VTP: %s)",
        mesh_metrics["num_vertices"], mesh_metrics["mae"], mesh_metrics["rmse"],
        mesh_metrics["median"], mesh_metrics["p95"], mesh_error_vtp_path,
    )

    return sdf_metrics, mesh_metrics


def _log_reconstruction_to_mlflow(
    *,
    metric_prefix: str,
    case_name: str,
    mesh_path: Path,
    tiling,
    checkpoint: str,
    recon_result: dict,
    metrics: dict | None,
    mesh_metrics: dict | None,
    local_dir: Path,
    reconstructed_mesh_path: Path,
) -> None:
    """Log reconstruction params, metrics, and artifacts to the active MLflow run."""
    import mlflow

    if mlflow.active_run() is None:
        return

    mlflow.log_param(f"{metric_prefix}_mesh_path", str(mesh_path))
    mlflow.log_param(f"{metric_prefix}_tiling", str(tiling))
    mlflow.log_param(f"{metric_prefix}_checkpoint", checkpoint)

    if recon_result["final_loss"] is not None:
        mlflow.log_metric(f"{metric_prefix}_final_loss", recon_result["final_loss"])

    artifact_path = f"reconstruction/{case_name}"
    mlflow.log_artifact(str(local_dir / "specs_reconstruction.json"), artifact_path=f"{artifact_path}/config")
    mlflow.log_artifact(str(local_dir / "loss_plot.png"), artifact_path=artifact_path)
    mlflow.log_artifact(str(reconstructed_mesh_path), artifact_path=artifact_path)

    for prefix, m in (("sdf", metrics), ("mesh", mesh_metrics)):
        if m is None:
            continue
        for key in ("mae", "rmse", "median", "p95"):
            mlflow.log_metric(f"{metric_prefix}_{prefix}_{key}", m[key])

    if metrics is not None or mesh_metrics is not None:
        mlflow.log_artifact(str(local_dir / "error_metrics.json"), artifact_path=artifact_path)


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
    from DeepSDFStruct.SDF import SDFfromDeepSDF, normalize_mesh_to_unit_cube
    from DeepSDFStruct.lattice_structure import LatticeSDFStruct
    from DeepSDFStruct.parametrization import SplineParametrization
    from DeepSDFStruct.torch_spline import TorchScaling
    from DeepSDFStruct.export_knot_grid import (
        export_knot_grid_paramspace,
        export_control_lattice_paramspace,
    )
    from deepshapeopt.config import make_experiment_paths, ensure_experiment_dirs
    from deepshapeopt.runtime import is_debug_enabled

    experiment_path = Path(experiment_path).resolve()
    rec_cfg = specs["reconstruction"]
    debug = is_debug_enabled(specs)

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

    if debug:
        export_knot_grid_paramspace(
            param_spline_sp, filename=str(heavy_dir / "knot_grid.vtp")
        )
        export_control_lattice_paramspace(
            param_spline_sp,
            filename=str(heavy_dir / "control_lattice_paramspace.vtp"),
            order="F",
        )

    param_spline = SplineParametrization(param_spline_sp, device=model.device)
    # Initialize every lattice control point to the mean of the trained latent
    # vectors. The decoder is only well-conditioned inside the learned latent
    # manifold; starting from ~0 lands in a flat/degenerate region where the
    # inference-time code optimization stalls (pronounced at small latent dims).
    control_points = param_spline.torch_spline.control_points
    trained_codes = torch.stack(list(model._trained_latent_vectors), dim=0)
    mean_code = trained_codes.mean(dim=0).to(
        device=control_points.device, dtype=control_points.dtype
    )
    param_spline.set_param(mean_code.expand(control_points.shape))

    lattice_struct = LatticeSDFStruct(
        tiling=tiling, microtile=sdf, parametrization=param_spline, bounds=bounds
    )

    # --- Reconstruct ---
    metric_prefix = mlflow_metric_prefix or f"reconstruction_{case_name}"

    recon_result = fit_lattice_to_sdf(
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
        samples_series_dir=heavy_dir / "rec_sdf_samples_series",
    )

    lattice_struct.parametrization.set_param(recon_result["params"][0])

    # --- Mesh + SDF-grid exports ---
    physical_mesh_name = f"{mesh_path.stem}_reconstructed.stl"
    reconstructed_mesh_path = export_reconstructed_artifacts(
        lattice_struct,
        heavy_dir,
        mesh_resolution=create_mesh_N,
        bounds=bounds,
        device=mesh_device,
        scaling=scaling,
        physical_mesh_name=physical_mesh_name,
        param_mesh_name=f"{mesh_path.stem}_reconstructed_param_space.stl",
    )

    # --- Error metrics ---
    metrics, mesh_metrics = None, None
    if save_vtp:
        metrics, mesh_metrics = _compute_reconstruction_metrics(
            lattice_struct=lattice_struct,
            mesh=mesh,
            mesh_orig=mesh_orig,
            reconstructed_mesh_path=reconstructed_mesh_path,
            heavy_dir=heavy_dir,
            mesh_stem=mesh_path.stem,
            n_surface_samples=n_surface_samples,
            samples_surface_stds=samples_surface_stds,
            error_cutoff=rec_cfg.get("error_cutoff", 0.1),
            device=device,
        )
        _write_json(
            local_dir / "error_metrics.json",
            {"sdf_sample_error": metrics, "mesh_vertex_error": mesh_metrics},
        )
        logger.debug("Saved unified error metrics to: %s", local_dir / "error_metrics.json")

    # --- MLflow logging ---
    if use_mlflow:
        _log_reconstruction_to_mlflow(
            metric_prefix=metric_prefix,
            case_name=case_name,
            mesh_path=mesh_path,
            tiling=tiling,
            checkpoint=checkpoint,
            recon_result=recon_result,
            metrics=metrics,
            mesh_metrics=mesh_metrics,
            local_dir=local_dir,
            reconstructed_mesh_path=reconstructed_mesh_path,
        )

    # --- Save summary ---
    run_info["status"] = "finished"
    run_info["mesh_bounds"] = bounds.tolist()
    run_info["final_loss"] = recon_result["final_loss"]
    run_info["num_steps"] = recon_result["num_steps"]
    run_info["reconstructed_mesh_path"] = str(reconstructed_mesh_path)
    _write_json(local_dir / "specs_summary.json", run_info)

    logger.info("Reconstruction done. Results in %s", local_dir)
    if heavy_dir != local_dir:
        logger.debug("Heavy data in: %s", heavy_dir)

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

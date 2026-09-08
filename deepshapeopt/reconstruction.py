from __future__ import annotations

"""
Config-driven wrapper around ``DeepSDFStruct.geom_reconstruction``.

The reconstruction itself -- normalizing the mesh, building the B-spline of
latent codes, sampling the ground-truth SDF and fitting the lattice -- lives in
:class:`DeepSDFStruct.geom_reconstruction.LocalShapesReconstructor`. What stays here
is everything specific to running one as a *DeepShapeOpt experiment*:

- ``reconstruct_shape``: the standalone pipeline behind ``scripts/reconstruct.py``
  -- experiment paths, spec snapshots, error metrics, MLflow.
- ``fit_lattice_to_sdf``: the fit step plus this repo's debug VTP exports.
  Shared with the in-optimization phase (``shape_optimization.run_reconstruction``,
  which lives there because it also handles ``reuse_parameter`` caching).
- ``build_reconstruction_lattice``: config-dict front end to
  ``LocalShapesReconstructor.build_struct``, shared with the latent-edit GUI.

Reusable helpers that used to live here now come from DeepSDFStruct:
``build_parameter_spline`` and ``sample_gt_sdf`` from
``DeepSDFStruct.geom_reconstruction``, ``export_reconstructed_artifacts`` from
``DeepSDFStruct.mesh``, and ``with_float32_lattice`` from ``DeepSDFStruct.utils``.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import trimesh
    from DeepSDFStruct.lattice_structure import LatticeSDFStruct
    from deepshapeopt.config import ExperimentSpecifications


# ---------------------------------------------------------------------------
# Building blocks used by the optimization phase
# ---------------------------------------------------------------------------

def init_spline_parameters(param_spline, mean=0.0, std=0.001):
    """Initialize all trainable parameters of the spline.

    Used by the optimization workflow, which deliberately starts from small
    random codes rather than the mean trained code that
    ``LocalShapesReconstructor.build_struct`` uses.
    """
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
    box_constrained: bool = True,
    samples_series_dir: Path | None = None,
):
    """Core fit step: sample ground-truth SDF, fit the lattice, export VTPs.

    A config-dict front end to
    :meth:`DeepSDFStruct.geom_reconstruction.LocalShapesReconstructor.fit_samples`
    that adds this repo's debug exports. Shared by both the standalone pipeline
    (`reconstruct_shape`) and the in-optimization phase
    (`shape_optimization.run_reconstruction`).

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
        ``num_iterations``, ``batch_size``. Optional: ``n_uniform_samples``,
        ``n_surface_samples``, ``samples_surface_stds``, ``code_reg_lambda``,
        ``code_bound``, ``grad_clip``, ``eikonal_lambda``, ``device``.
    output_dir : Path
        Directory for heavy data exports (VTP files).
    lightweight_output_dir : Path or None
        Directory for lightweight outputs (loss plot, loss CSV).
        If None, defaults to ``output_dir``.
    save_vtp : bool
        Whether to export VTP sample files.
    box_constrained : bool
        If True, reject surface samples outside ``bounds`` before fitting.
    samples_series_dir : Path or None
        If set and ``rec_cfg["export_rec_samples_series"]`` is truthy, export a
        contiguously numbered ``rec_sdf_samples_{frame:04d}.vtp`` series here
        (frame 0000 = initial field). By default one frame per epoch; set
        ``rec_cfg["export_rec_samples_fine_until_epoch"]`` > 0 to also export
        between batch steps for those first epochs (every
        ``export_rec_samples_fine_every`` batches, default 1).

    Returns
    -------
    dict with keys: ``params``, ``loss_history``, ``final_loss``, ``num_steps``.
    """
    from DeepSDFStruct.geom_reconstruction import LocalShapesReconstructor, sample_gt_sdf
    from DeepSDFStruct.SDF import SDFfromMesh
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

    output_dir = Path(output_dir)
    lightweight_output_dir = (
        Path(lightweight_output_dir)
        if lightweight_output_dir is not None
        else output_dir
    )

    # --- Sample ground truth SDF ---
    gt_sdf = SDFfromMesh(mesh, scale=False)
    sdf_samples = sample_gt_sdf(
        gt_sdf,
        mesh,
        bounds,
        n_uniform=n_uniform_samples,
        n_surface=n_surface_samples,
        stds=stds,
        device=device,
        box_constrained=box_constrained,
    )

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
    # The lattice is already built, so drive the fit step directly rather than
    # going through fit_mesh (which would rebuild the structure from scratch).
    recon_result = LocalShapesReconstructor.fit_samples(
        lattice_struct,
        sdf_samples,
        lr=float(rec_cfg["lr"]),
        num_iterations=int(rec_cfg["num_iterations"]),
        batch_size=int(rec_cfg["batch_size"]),
        code_reg_lambda=float(rec_cfg.get("code_reg_lambda", 0.0)),
        code_bound=rec_cfg.get("code_bound", None),
        grad_clip=rec_cfg.get("grad_clip", None),
        eikonal_lambda=float(rec_cfg.get("eikonal_lambda", 0.0)),
        loss_fn=rec_cfg.get("loss_fn", "ClampedL1"),
        clamp_val=float(rec_cfg.get("clamp_val", 0.1)),
        loss_plot_path=lightweight_output_dir / "loss_plot.png",
        loss_csv_path=lightweight_output_dir / "loss_history.csv",
        step_callback=step_callback,
    )

    # --- Export reconstructed SDF samples ---
    if save_vtp:
        _export_rec_samples(
            sdf_samples.samples.detach(), output_dir / "rec_sdf_samples.vtp"
        )

    return recon_result


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

def build_reconstruction_lattice(specs: ExperimentSpecifications, device=None):
    """Build the DeepSDF lattice for a reconstruction experiment.

    Config-dict front end to
    :meth:`DeepSDFStruct.geom_reconstruction.LocalShapesReconstructor.build_struct`.
    Shared by :func:`reconstruct_shape` and the interactive latent-edit GUI, so
    the two always agree on tiling, bounds, and ordering.

    Parameters
    ----------
    specs : ExperimentSpecifications
        Experiment configuration with a nested ``reconstruction`` section.
    device : str or torch.device, optional
        Override for the compute device. Defaults to ``reconstruction.device``.

    Returns
    -------
    dict
        Keys: ``lattice_struct``, ``param_spline`` (SplineParametrization),
        ``param_spline_sp`` (splinepy.BSpline), ``scaling`` (TorchScaling),
        ``bounds``, ``model``, ``sdf``, ``mesh_orig``, ``mesh_norm``,
        ``scale``, ``shift``, ``latent_dim``, ``tiling``, ``spline_degree``,
        ``create_mesh_N``, ``mesh_path``, ``device``.
    """
    import trimesh
    from DeepSDFStruct.geom_reconstruction import LocalShapesReconstructor

    rec_cfg = specs["reconstruction"]
    device = device if device is not None else rec_cfg.get("device", "cuda")
    tiling = rec_cfg["tiling"]
    spline_degree = rec_cfg.get("spline_degree", [1, 1, 1])
    create_mesh_N = int(rec_cfg["create_mesh_N"])
    mesh_path = Path(rec_cfg["mesh_path"]).resolve()

    recon = LocalShapesReconstructor(
        str(rec_cfg["model_path"]),
        checkpoint=str(rec_cfg.get("model_checkpoint", "latest")),
        device=device,
    )

    mesh_orig = trimesh.load_mesh(str(mesh_path))
    built = recon.build_struct(mesh_orig, tiling, spline_degree=spline_degree)

    return {
        "lattice_struct": built.struct,
        "param_spline": built.param_spline,
        "param_spline_sp": built.param_spline_sp,
        "scaling": built.scaling,
        "bounds": built.bounds,
        "model": recon.model,
        "sdf": recon.microtile,
        "mesh_orig": mesh_orig,
        "mesh_norm": built.mesh_norm,
        "scale": built.scale,
        "shift": built.shift,
        "latent_dim": recon.latent_dim,
        "tiling": tiling,
        "spline_degree": spline_degree,
        "create_mesh_N": create_mesh_N,
        "mesh_path": mesh_path,
        "device": device,
    }


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
    from DeepSDFStruct.export_knot_grid import (
        export_knot_grid_paramspace,
        export_control_lattice_paramspace,
    )
    from DeepSDFStruct.mesh import export_reconstructed_artifacts
    from deepshapeopt.config import make_experiment_paths, ensure_experiment_dirs
    from deepshapeopt.runtime import is_debug_enabled

    experiment_path = Path(experiment_path).resolve()
    rec_cfg = specs["reconstruction"]
    debug = is_debug_enabled(specs)

    # --- Config fields ---
    device = rec_cfg.get("device", "cuda")
    mesh_device = rec_cfg.get("mesh_device", device)
    tiling = rec_cfg["tiling"]
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

    # --- CUDA check (covers the separate mesh_device too) ---
    if "cuda" in str(device) or "cuda" in str(mesh_device):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested in config, but torch.cuda.is_available() is False."
            )

    # --- Build model + lattice (shared with the latent-edit GUI) ---
    built = build_reconstruction_lattice(specs, device=device)
    mesh_orig = built["mesh_orig"]
    mesh = built["mesh_norm"]
    bounds = built["bounds"]
    scaling = built["scaling"]
    param_spline_sp = built["param_spline_sp"]
    lattice_struct = built["lattice_struct"]

    if verbose:
        mesh.export(str(heavy_dir / "input_mesh_normalized.stl"))

    if debug:
        export_knot_grid_paramspace(
            param_spline_sp, filename=str(heavy_dir / "knot_grid.vtp")
        )
        export_control_lattice_paramspace(
            param_spline_sp,
            filename=str(heavy_dir / "control_lattice_paramspace.vtp"),
            order="F",
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
        box_constrained=False,
        samples_series_dir=heavy_dir / "rec_sdf_samples_series",
    )

    lattice_struct.parametrization.set_param(recon_result["params"][0])

    # Persist the optimized control-point latent codes so the reconstruction
    # can be reloaded later (e.g. by the interactive latent-edit GUI). Same
    # format as shape_optimization.run_reconstruction's rec_parameters.pt.
    torch.save(recon_result["params"], heavy_dir / "rec_parameters.pt")

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

"""Fit of the latent lattice to a mesh, and the standalone reconstruction workflow.

The fit itself lives in :class:`DeepSDFStruct.geom_reconstruction.LocalShapesReconstructor`;
this module adds the config front end, the debug exports and the error metrics of a
DeepShapeOpt reconstruction run.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from deepshapeopt.config import Config
    from deepshapeopt.config.schema import ReconstructionConfig

logger = logging.getLogger(__name__)


def init_spline_parameters(param_spline, mean=0.0, std=0.001):
    """Small random start for the latent control points of the optimization lattice."""
    for p in param_spline.parameters():
        torch.nn.init.normal_(p, mean=mean, std=std)


def fit_lattice_to_sdf(
    lattice_struct,
    mesh,
    bounds: torch.Tensor,
    recon: "ReconstructionConfig",
    device,
    output_dir: Path,
    lightweight_output_dir: Path | None = None,
    save_vtp: bool = True,
    box_constrained: bool = True,
    samples_series_dir: Path | None = None,
):
    """Sample the ground-truth SDF of ``mesh`` and fit ``lattice_struct`` to it.

    Returns the dict of ``LocalShapesReconstructor.fit_samples`` (``params``, ``loss_history``,
    ``final_loss``, ``num_steps``). ``output_dir`` takes the heavy VTP exports,
    ``lightweight_output_dir`` (default: the same) the loss plot and CSV. With
    ``recon.export_samples_series`` and ``samples_series_dir`` a per-epoch series of the
    reconstructed samples is written (fine per-batch frames for the first
    ``export_samples_fine_until_epoch`` epochs).
    """
    from DeepSDFStruct.geom_reconstruction import LocalShapesReconstructor, sample_gt_sdf
    from DeepSDFStruct.SDF import SDFfromMesh
    from DeepSDFStruct.sampling import save_points_to_vtp

    output_dir = Path(output_dir)
    lightweight_output_dir = Path(lightweight_output_dir) if lightweight_output_dir is not None else output_dir

    def export_samples(samples_ps, path):
        rec_dist = lattice_struct(samples_ps)
        save_points_to_vtp(path, torch.hstack((samples_ps, rec_dist.detach())))

    gt_sdf = SDFfromMesh(mesh, scale=False)
    sdf_samples = sample_gt_sdf(
        gt_sdf, mesh, bounds,
        n_uniform=int(recon.n_uniform_samples), n_surface=int(recon.n_surface_samples),
        stds=list(recon.samples_surface_stds), device=device, box_constrained=box_constrained,
    )
    if save_vtp:
        save_points_to_vtp(
            output_dir / "gt_sdf_samples.vtp",
            torch.hstack((sdf_samples.samples.detach(), sdf_samples.distances.detach())),
        )

    step_callback = None
    if recon.export_samples_series and samples_series_dir is not None:
        samples_series_dir = Path(samples_series_dir)
        samples_series_dir.mkdir(parents=True, exist_ok=True)
        series_samples = sdf_samples.samples.detach()
        fine_until = int(recon.export_samples_fine_until_epoch)
        fine_every = max(1, int(recon.export_samples_fine_every))
        frame = 0

        def write_frame():
            nonlocal frame
            with torch.no_grad():
                export_samples(series_samples, samples_series_dir / f"rec_sdf_samples_{frame:04d}.vtp")
            frame += 1

        write_frame()

        def step_callback(e, batch_idx, n_batches):
            end = batch_idx == n_batches - 1
            if (e < fine_until and (batch_idx % fine_every == 0 or end)) or (e >= fine_until and end):
                write_frame()

    result = LocalShapesReconstructor.fit_samples(
        lattice_struct, sdf_samples,
        lr=float(recon.lr), num_iterations=int(recon.num_iterations), batch_size=int(recon.batch_size),
        code_reg_lambda=float(recon.code_reg_lambda), code_bound=recon.code_bound,
        grad_clip=recon.grad_clip, eikonal_lambda=float(recon.eikonal_lambda),
        loss_fn=recon.loss_fn, clamp_val=float(recon.clamp_val),
        loss_plot_path=lightweight_output_dir / "loss_plot.png",
        loss_csv_path=lightweight_output_dir / "loss_history.csv",
        step_callback=step_callback,
    )
    if save_vtp:
        export_samples(sdf_samples.samples.detach(), output_dir / "rec_sdf_samples.vtp")
    return result


# ---------------------------------------------------------------------------
# Standalone reconstruction (scripts/reconstruct.py, latent GUI)
# ---------------------------------------------------------------------------

def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def build_reconstruction_lattice(cfg: "Config", device=None) -> dict:
    """Model, mesh and lattice of a reconstruction experiment (shared with the latent GUI)."""
    import trimesh
    from DeepSDFStruct.geom_reconstruction import LocalShapesReconstructor

    ds = cfg.parametrization.deepsdf
    device = device if device is not None else cfg.run.device
    mesh_path = Path(cfg.geometry.mesh_path).resolve()
    recon = LocalShapesReconstructor(str(ds.model_path), checkpoint=str(ds.checkpoint), device=device)
    mesh_orig = trimesh.load_mesh(str(mesh_path))
    built = recon.build_struct(mesh_orig, ds.tiling, spline_degree=ds.spline_degree)
    return {
        "lattice_struct": built.struct, "param_spline": built.param_spline,
        "param_spline_sp": built.param_spline_sp, "scaling": built.scaling, "bounds": built.bounds,
        "model": recon.model, "sdf": recon.microtile, "mesh_orig": mesh_orig, "mesh_norm": built.mesh_norm,
        "scale": built.scale, "shift": built.shift, "latent_dim": recon.latent_dim,
        "tiling": ds.tiling, "spline_degree": ds.spline_degree,
        "export_resolution": int(ds.reconstruction.export_resolution), "mesh_path": mesh_path, "device": device,
    }


def _reconstruction_metrics(*, lattice_struct, mesh, mesh_orig, reconstructed_mesh_path, heavy_dir,
                            mesh_stem, recon, device):
    """SDF-sample and mesh-vertex error metrics; writes the accompanying VTPs."""
    import trimesh
    from DeepSDFStruct.SDF import SDFfromMesh
    from DeepSDFStruct.deep_sdf.metrics.error_metrics import compute_metrics_from_vtp
    from DeepSDFStruct.sampling import sample_mesh_surface, save_points_to_vtp

    from .analysis import add_vertex_colors_from_scalar, compute_vertex_sdf_error, trimesh_to_pyvista

    gt = SDFfromMesh(mesh, scale=False)
    surface = sample_mesh_surface(gt, mesh, n_samples=int(recon.n_surface_samples),
                                  stds=list(recon.samples_surface_stds), device=device)
    save_points_to_vtp(heavy_dir / "gt_sdf_samples_surface.vtp",
                       torch.hstack((surface.samples.detach(), surface.distances.detach())))
    samples = surface.samples.detach()
    save_points_to_vtp(heavy_dir / "rec_sdf_samples_surface.vtp",
                       torch.hstack((samples, lattice_struct(samples).detach())))
    sdf_metrics = compute_metrics_from_vtp(
        gt_vtp_path=heavy_dir / "gt_sdf_samples_surface.vtp",
        pred_vtp_path=heavy_dir / "rec_sdf_samples_surface.vtp",
        cutoff=float(recon.error_cutoff), output_json_path=None,
    )
    reconstructed = trimesh.load_mesh(str(reconstructed_mesh_path), force="mesh")
    reconstructed_norm, err = compute_vertex_sdf_error(mesh_orig, reconstructed)
    abs_err = np.abs(err)
    mesh_metrics = {
        "num_vertices": int(reconstructed_norm.vertices.shape[0]),
        "mae": float(abs_err.mean()), "rmse": float(np.sqrt(np.mean(err ** 2))),
        "median": float(np.median(abs_err)), "p95": float(np.quantile(abs_err, 0.95)),
        "max": float(abs_err.max()), "min_signed_error": float(err.min()),
        "max_signed_error": float(err.max()), "mean_signed_error": float(err.mean()),
    }
    poly = trimesh_to_pyvista(reconstructed_norm)
    poly.point_data["sdf_error"] = err
    add_vertex_colors_from_scalar(poly, scalar_name="sdf_error", cmap_name="turbo")
    poly.save(str(heavy_dir / f"{mesh_stem}_mesh_sdf_error.vtp"))
    return sdf_metrics, mesh_metrics


def reconstruct_shape(cfg: "Config", experiment_dir: Path, case_name: str | None = None,
                      save_vtp: bool = True) -> dict:
    """Standalone reconstruction: fit the lattice to the input mesh, export mesh and metrics."""
    from DeepSDFStruct.export_knot_grid import export_control_lattice_paramspace, export_knot_grid_paramspace
    from DeepSDFStruct.mesh import export_reconstructed_artifacts

    from deepshapeopt.config import make_run_paths

    ds = cfg.parametrization.deepsdf
    recon = ds.reconstruction
    device = cfg.run.device
    mesh_device = recon.mesh_device or device
    paths = make_run_paths(cfg, experiment_dir).ensure()
    mesh_path = Path(cfg.geometry.mesh_path).resolve()
    if case_name is None:
        case_name = f"{mesh_path.stem}_tiling_{'x'.join(map(str, ds.tiling))}"

    local_dir = paths.reconstruction / case_name
    local_dir.mkdir(parents=True, exist_ok=True)
    heavy_dir = paths.heavy_data / "reconstruction" / case_name if paths.heavy_data is not None else local_dir
    heavy_dir.mkdir(parents=True, exist_ok=True)
    _write_json(local_dir / "config_log.json", cfg.raw)
    run_info = {"timestamp": datetime.now().isoformat(timespec="seconds"), "experiment_dir": str(experiment_dir),
                "case_name": case_name, "resolved_mesh_path": str(mesh_path), "status": "started"}
    _write_json(local_dir / "run_summary.json", run_info)

    if ("cuda" in str(device) or "cuda" in str(mesh_device)) and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested in the config, but torch.cuda.is_available() is False")

    built = build_reconstruction_lattice(cfg, device=device)
    mesh, bounds, lattice_struct = built["mesh_norm"], built["bounds"], built["lattice_struct"]
    mesh.export(str(heavy_dir / "input_mesh_normalized.stl"))
    if cfg.run.debug:
        export_knot_grid_paramspace(built["param_spline_sp"], filename=str(heavy_dir / "knot_grid.vtp"))
        export_control_lattice_paramspace(built["param_spline_sp"],
                                          filename=str(heavy_dir / "control_lattice_paramspace.vtp"), order="F")

    result = fit_lattice_to_sdf(
        lattice_struct, mesh, bounds, recon, device, output_dir=heavy_dir, lightweight_output_dir=local_dir,
        save_vtp=save_vtp, box_constrained=False, samples_series_dir=heavy_dir / "rec_sdf_samples_series",
    )
    lattice_struct.parametrization.set_param(result["params"][0])
    torch.save(result["params"], heavy_dir / "rec_parameters.pt")

    reconstructed_mesh_path = export_reconstructed_artifacts(
        lattice_struct, heavy_dir, mesh_resolution=int(recon.export_resolution), bounds=bounds,
        device=mesh_device, scaling=built["scaling"],
        physical_mesh_name=f"{mesh_path.stem}_reconstructed.stl",
        param_mesh_name=f"{mesh_path.stem}_reconstructed_param_space.stl",
    )

    metrics = mesh_metrics = None
    if save_vtp:
        metrics, mesh_metrics = _reconstruction_metrics(
            lattice_struct=lattice_struct, mesh=mesh, mesh_orig=built["mesh_orig"],
            reconstructed_mesh_path=reconstructed_mesh_path, heavy_dir=heavy_dir,
            mesh_stem=mesh_path.stem, recon=recon, device=device,
        )
        _write_json(local_dir / "error_metrics.json", {"sdf_sample_error": metrics, "mesh_vertex_error": mesh_metrics})

    run_info.update(status="finished", mesh_bounds=bounds.tolist(), final_loss=result["final_loss"],
                    num_steps=result["num_steps"], reconstructed_mesh_path=str(reconstructed_mesh_path))
    _write_json(local_dir / "run_summary.json", run_info)
    logger.info("Reconstruction done: %s", local_dir)
    return {"case_name": case_name, "results_dir": str(local_dir), "heavy_data_dir": str(heavy_dir),
            "reconstructed_mesh_path": str(reconstructed_mesh_path), "final_loss": result["final_loss"],
            "num_steps": result["num_steps"], "metrics": metrics, "mesh_metrics": mesh_metrics}

"""Post-hoc PCA projection study for a reconstruction experiment.

Fits a DeepSDF B-spline lattice to a target shape once at the full latent
dimension ``D``, then projects the fitted per-control-point latent codes onto the
top-``k`` principal directions of the model's *training* latents for a range of
``k`` and measures how the SDF reconstruction error grows as ``k`` shrinks. This
answers "how many latent variables does this shape actually need?" with no CFD.

The setup (mesh normalization, spline build, fit) mirrors
``deepshapeopt.reconstruction.reconstruct_shape`` so the fitted shape matches a
normal reconstruction run.

Usage:
    uv run python scripts/pca_reconstruction_study.py \
        --config experiments/reconstruction/feed_channel/config.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from deepshapeopt.config import (
    ExperimentSpecifications,
    ensure_experiment_dirs,
    make_experiment_paths,
)
from deepshapeopt.latent_pca import (
    PCALatentBasis,
    compute_latent_pca,
    gather_training_latents,
)
from deepshapeopt.reconstruction import (
    build_parameter_spline,
    export_reconstructed_artifacts,
    fit_lattice_to_sdf,
    sample_sdf,
)

LOGGER = logging.getLogger("pca_reconstruction_study")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT / "experiments" / "reconstruction" / "feed_channel" / "config.json"
)


def _sdf_errors(lattice_struct, samples, gt_dist, clamp=0.1):
    """SDF error of the current lattice against ground-truth distances."""
    with torch.no_grad():
        pred = lattice_struct(samples).reshape(-1)
    gt = gt_dist.reshape(-1)
    err = pred - gt
    rmse = float(torch.sqrt((err ** 2).mean()))
    mae = float(err.abs().mean())
    # Near-surface (clamped) error: where reconstruction fidelity matters most.
    pc, gc = pred.clamp(-clamp, clamp), gt.clamp(-clamp, clamp)
    rmse_clamped = float(torch.sqrt(((pc - gc) ** 2).mean()))
    return rmse, mae, rmse_clamped


def _plot(rows, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = [r["k"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(ks, [r["sdf_rmse"] for r in rows], "o-", color="C0", label="SDF RMSE")
    ax1.plot(ks, [r["sdf_rmse_clamped"] for r in rows], "s--", color="C3",
             label="SDF RMSE (clamped ±0.1)")
    ax1.set_xlabel("number of PCA components k")
    ax1.set_ylabel("reconstruction SDF error")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    ax2 = ax1.twinx()
    ax2.plot(ks, [r["explained_variance"] for r in rows], "^:", color="C2",
             label="cumulative explained variance")
    ax2.set_ylabel("cumulative explained variance")
    ax2.set_ylim(0, 1.02)
    ax2.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--k-list", default=None,
                    help="Comma-separated k values (default: 1..latent_dim)")
    ap.add_argument("--mesh-resolution", type=int, default=None)
    ap.add_argument("--export-meshes", action="store_true",
                    help="Export a reconstructed STL per k (slower)")
    ap.add_argument("--eval-uniform", type=int, default=50000)
    ap.add_argument("--eval-surface", type=int, default=200000)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config_path = Path(args.config).resolve()
    experiment_path = config_path.parent
    specs = ExperimentSpecifications(str(config_path))
    rec_cfg = specs["reconstruction"]
    device = rec_cfg.get("device", "cpu")

    # --- Setup mirrors reconstruct_shape -----------------------------------
    import trimesh
    from DeepSDFStruct.lattice_structure import LatticeSDFStruct
    from DeepSDFStruct.parametrization import SplineParametrization
    from DeepSDFStruct.pretrained_models import get_model
    from DeepSDFStruct.SDF import SDFfromDeepSDF, normalize_mesh_to_unit_cube
    from DeepSDFStruct.torch_spline import TorchScaling

    results_name = specs.get("results_name", "results")
    paths = make_experiment_paths(
        experiment_path, results_name=results_name,
        heavy_data_output_path=rec_cfg.get("heavy_data_output_path"),
    )
    ensure_experiment_dirs(paths)
    out_dir = paths.reconstruction / "pca_study"
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh_path = Path(rec_cfg["mesh_path"]).resolve()
    model_path = str(rec_cfg["model_path"])
    checkpoint = str(rec_cfg.get("model_checkpoint", "latest"))
    tiling = rec_cfg["tiling"]
    spline_degree = rec_cfg.get("spline_degree", [1, 1, 1])
    mesh_res = args.mesh_resolution or int(rec_cfg.get("create_mesh_N", 32))

    mesh_orig = trimesh.load_mesh(str(mesh_path))
    mesh, scale, shift = normalize_mesh_to_unit_cube(mesh_orig.copy())
    bounds = torch.tensor(mesh.bounds, device=device, dtype=torch.float32)
    scaling = TorchScaling(scale_factors=scale, translation=shift, bounds=bounds, device=device)

    model = get_model(model_path, checkpoint=checkpoint)
    sdf = SDFfromDeepSDF(model)
    latent_dim = model._trained_latent_vectors[0].shape[0]

    mins, maxs = bounds[0].cpu().numpy(), bounds[1].cpu().numpy()
    param_spline_sp = build_parameter_spline(
        spline_degrees=spline_degree, tiling=tiling,
        latent_dim=latent_dim, bounds=np.stack([mins, maxs]),
    )
    param_spline = SplineParametrization(param_spline_sp, device=model.device)
    # Initialize control points to the mean trained code (as reconstruct_shape).
    cps = param_spline.torch_spline.control_points
    trained = torch.stack(list(model._trained_latent_vectors), dim=0)
    mean_code = trained.mean(0).to(device=cps.device, dtype=cps.dtype)
    param_spline.set_param(mean_code.expand(cps.shape))
    lattice_struct = LatticeSDFStruct(
        tiling=tiling, microtile=sdf, parametrization=param_spline, bounds=bounds
    )

    # --- Fit once at full latent dimension ---------------------------------
    LOGGER.info("Fitting full-rank reconstruction (latent_dim=%d, tiling=%s)...",
                latent_dim, tiling)
    recon = fit_lattice_to_sdf(
        lattice_struct, mesh, bounds, rec_cfg,
        output_dir=out_dir, lightweight_output_dir=out_dir,
        save_vtp=False, box_constrained=False,
    )
    P = recon["params"][0].detach().clone()  # (n_cp, D) fitted control points
    n_cp = P.shape[0]
    LOGGER.info("Fitted %d control points x %d latent dims; final fit loss=%.5e",
                n_cp, latent_dim, recon["final_loss"])

    # --- Fixed held-out evaluation sample set (shared across all k) --------
    torch.manual_seed(0)
    eval_sdf = sample_sdf(
        mesh, bounds,
        n_uniform_samples=args.eval_uniform, n_surface_samples=args.eval_surface,
        device=device, stds=rec_cfg.get("samples_surface_stds", [0.005, 0.0001]),
        box_constrained=False,
    )
    samples = eval_sdf.samples.detach()
    gt_dist = eval_sdf.distances.detach()

    # --- PCA basis on the model's training latents -------------------------
    latents = gather_training_latents(model_path, checkpoint, device=device)
    mean, comps, expl, _scale = compute_latent_pca(latents, latent_dim)
    mean, comps = mean.to(P), comps.to(P)
    cum_evr = torch.cumsum(expl, 0)

    ks = ([int(x) for x in args.k_list.split(",")] if args.k_list
          else list(range(1, latent_dim + 1)))

    LOGGER.info("\n%-4s %-10s %-12s %-12s %-12s %-14s",
                "k", "EVR", "cp_resid", "sdf_rmse", "sdf_mae", "rmse@clamp")
    rows = []
    for k in ks:
        basis = PCALatentBasis(mean, comps[:, :k].contiguous())
        with torch.no_grad():
            P_k = basis.to_latent(basis.to_coeff(P))
            param_spline.set_param(P_k)
            cp_resid = float((P_k - P).pow(2).mean().sqrt())
        rmse, mae, rmse_c = _sdf_errors(lattice_struct, samples, gt_dist)
        evr = float(cum_evr[k - 1])
        rows.append({
            "k": k, "explained_variance": evr, "cp_residual_rmse": cp_resid,
            "sdf_rmse": rmse, "sdf_mae": mae, "sdf_rmse_clamped": rmse_c,
        })
        LOGGER.info("%-4d %-10.4f %-12.5f %-12.6f %-12.6f %-14.6f",
                    k, evr, cp_resid, rmse, mae, rmse_c)

        if args.export_meshes:
            param_spline.set_param(P_k)
            try:
                export_reconstructed_artifacts(
                    lattice_struct, out_dir, mesh_resolution=mesh_res,
                    bounds=bounds, device=device, scaling=scaling,
                    export_sdf_grid=False, export_param_mesh=False,
                    physical_mesh_name=f"recon_k{k:02d}.stl",
                )
            except Exception as exc:  # meshing is optional/diagnostic
                LOGGER.warning("mesh export failed for k=%d: %s", k, exc)

    param_spline.set_param(P)  # restore full-rank fit

    # --- Persist results ---------------------------------------------------
    summary = {
        "config": str(config_path),
        "mesh": str(mesh_path),
        "model": model_path,
        "latent_dim": latent_dim,
        "n_control_points": n_cp,
        "full_fit_loss": recon["final_loss"],
        "rows": rows,
    }
    (out_dir / "pca_study.json").write_text(json.dumps(summary, indent=2))
    csv_lines = ["k,explained_variance,cp_residual_rmse,sdf_rmse,sdf_mae,sdf_rmse_clamped"]
    csv_lines += [
        f"{r['k']},{r['explained_variance']:.6f},{r['cp_residual_rmse']:.6f},"
        f"{r['sdf_rmse']:.6f},{r['sdf_mae']:.6f},{r['sdf_rmse_clamped']:.6f}"
        for r in rows
    ]
    (out_dir / "pca_study.csv").write_text("\n".join(csv_lines) + "\n")
    try:
        _plot(rows, out_dir / "pca_study.png")
    except Exception as exc:
        LOGGER.warning("plot failed: %s", exc)

    LOGGER.info("\nResults written to %s", out_dir)


if __name__ == "__main__":
    main()

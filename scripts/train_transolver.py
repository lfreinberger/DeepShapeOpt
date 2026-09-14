"""Train the Transolver flow surrogate on the generated OpenFOAM dataset.

Usage:
    uv run python scripts/train_transolver.py --config experiments/transolver/config_train.json

Checkpoints (``latest.pt`` / ``best.pt``) are self-contained: model config +
weights + normalization statistics + surrogate query-cloud config, so the
optimizer only needs the checkpoint path. Set ``training.overfit_n`` to a
small integer for the overfitting sanity check (P3 gate).

``model.out_dim`` selects the target set: 4 = ``[U, p]`` (viscous drag from
the probe-shell finite difference, "fd" mode), 7 = ``[U, p, tau_w]`` (wall
shear stress predicted directly, "tau" mode; no FD calibration needed).

``features.set`` selects the per-point input (default ``"geo"`` =
``[pos, sdf, normal]``; ``"lat"`` = ``[pos, z(x)]``; ``"geo+lat"`` = both),
where ``z(x)`` is the lattice's B-spline latent code evaluated from the
``param`` array of every sample with the lattice layout in
``features.latent`` (``design_domain``, ``tiling``, ``spline_degree`` of the
dataset's reconstruction block). ``model.in_dim`` must match the feature set
(7 / 35 / 39 for 32-dim codes) and is filled in when absent.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from deepshapeopt.config import ExperimentSpecifications
from deepshapeopt.surrogate.dataset import (
    FlowFieldDataset,
    Normalizer,
    calibrate_visc_scale,
    cloud_from_batch,
    compute_norm_stats,
    split_files,
)
from deepshapeopt.surrogate.features import (
    DEFAULT_FEATURE_SET,
    feature_dim,
    make_latent_spec,
    needs_latent,
)
from deepshapeopt.surrogate.losses import field_loss, rel_l2
from deepshapeopt.surrogate.predictor import TransolverSurrogate
from deepshapeopt.surrogate.transolver import Transolver

logger = logging.getLogger("train_transolver")


def drag_metric(batch: dict, pred_raw: torch.Tensor, surrogate: TransolverSurrogate) -> float:
    """Relative drag error of one sample from de-normalized predictions."""
    J, _ = surrogate.drag_from_prediction(pred_raw, cloud_from_batch(batch))
    J_ref = batch["drag_foam"]
    return float(abs(float(J) - J_ref) / (abs(J_ref) + 1e-12))


def evaluate(model, loader, surrogate: TransolverSurrogate, device) -> dict:
    tau_mode = surrogate.viscous_mode == "tau"
    model.eval()
    m = {"p_surf": [], "U_off": [], "drag_rel": []}
    if tau_mode:
        m["tau_surf"] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            pred = model(batch["x"][None])[0]
            role, valid = batch["role"], batch["valid"]
            surf = role == 0
            off = (~surf) & valid
            m["p_surf"].append(float(rel_l2(pred[:, 3], batch["y"][:, 3], surf & valid)))
            m["U_off"].append(float(rel_l2(pred[:, :3], batch["y"][:, :3], off)))
            if tau_mode:
                m["tau_surf"].append(
                    float(rel_l2(pred[:, 4:7], batch["y"][:, 4:7], surf & valid))
                )
            m["drag_rel"].append(drag_metric(batch, surrogate.norm.denorm_y(pred), surrogate))
    return {k: float(np.mean(v)) for k, v in m.items()}


def collate_single(items):
    (item,) = items
    return item


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = ExperimentSpecifications(args.config)
    tr_cfg, model_cfg, sur_cfg = cfg["training"], dict(cfg["model"]), cfg["surrogate"]
    device = torch.device(cfg.get("device", "cuda"))
    torch.manual_seed(int(cfg.get("seed", 0)))

    out_dim = int(model_cfg.get("out_dim", 4))
    tau_mode = out_dim >= 7
    logger.info("targets: %s (viscous mode %s)",
                "[U, p, tau_w]" if tau_mode else "[U, p]", "tau" if tau_mode else "fd")

    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        f for pat in ("sample_*.npz", "traj_*.npz")
        for f in Path(cfg["data_dir"]).glob(pat)
    )
    if not files:
        raise SystemExit(f"no samples found in {cfg['data_dir']}")

    feat_cfg = cfg.get("features") or {}
    feature_set = feat_cfg.get("set", DEFAULT_FEATURE_SET)
    latent_spec = None
    latent_dim = 0
    if needs_latent(feature_set):
        with np.load(files[0]) as z0:
            latent_dim = int(z0["param"].shape[1])
        lat_cfg = feat_cfg.get("latent")
        if not lat_cfg:
            raise SystemExit(
                f"features.set {feature_set!r} needs features.latent "
                "(design_domain, tiling, spline_degree of the dataset lattice)"
            )
        latent_spec = make_latent_spec(
            lat_cfg["design_domain"], lat_cfg["tiling"], lat_cfg["spline_degree"], latent_dim
        )
    in_dim = feature_dim(feature_set, latent_dim)
    if "in_dim" in model_cfg and int(model_cfg["in_dim"]) != in_dim:
        raise SystemExit(
            f"model.in_dim {model_cfg['in_dim']} does not match feature set "
            f"{feature_set!r} ({in_dim} channels)"
        )
    model_cfg["in_dim"] = in_dim
    logger.info("features: %s (%d channels)%s", feature_set, in_dim,
                f", latent spec {latent_spec}" if latent_spec else "")

    train_files, val_files = split_files(
        files,
        val_fraction=cfg["split"].get("val_fraction", 0.1),
        holdout_family=cfg["split"].get("holdout_family"),
        seed=int(cfg.get("seed", 0)),
        min_surface_points=int(cfg["split"].get("min_surface_points", 0)),
        require_tau=tau_mode,
    )
    overfit_n = tr_cfg.get("overfit_n")
    if overfit_n:
        train_files = train_files[: int(overfit_n)]
        val_files = train_files
    logger.info("train=%d val=%d samples", len(train_files), len(val_files))

    stats_path = results_dir / "dataset_stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else None
    if stats is not None and len(stats["mean_y"]) != out_dim:
        logger.warning(
            "cached %s has %d target channels, model has %d: recomputing",
            stats_path.name, len(stats["mean_y"]), out_dim,
        )
        stats = None
    if stats is not None and (
        len(stats["mean_x"]) != in_dim
        or stats.get("feature_set", DEFAULT_FEATURE_SET) != feature_set
        or stats.get("latent_spec") != latent_spec
    ):
        logger.warning(
            "cached %s was computed for feature set %r (%d channels), config wants "
            "%r (%d): recomputing",
            stats_path.name, stats.get("feature_set", DEFAULT_FEATURE_SET),
            len(stats["mean_x"]), feature_set, in_dim,
        )
        stats = None
    if stats is None:
        stats = compute_norm_stats(
            train_files, with_tau=tau_mode, feature_set=feature_set, latent_spec=latent_spec
        )
        stats_path.write_text(json.dumps(stats, indent=2))
    logger.info("norm stats over %d points from %d files (%d feature / %d target channels)",
                stats["n_points"], stats["n_files"], len(stats["mean_x"]), len(stats["mean_y"]))

    train_ds = FlowFieldDataset(train_files, stats)
    val_ds = FlowFieldDataset(val_files, stats)
    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True, num_workers=2, collate_fn=collate_single
    )
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0, collate_fn=collate_single)

    # Separate device-side normalizer for evaluation; the dataset's own
    # normalizer must stay on CPU (it runs inside dataloader workers).
    eval_norm = Normalizer(stats, device=device)

    model = Transolver(**model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Transolver with %.2fM parameters on %s", n_params / 1e6, device)

    epochs = int(tr_cfg["epochs"])
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(tr_cfg["lr"]), weight_decay=float(tr_cfg["weight_decay"])
    )
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=float(tr_cfg["lr"]), total_steps=epochs * max(1, len(train_loader))
    )

    nu = float(sur_cfg.get("nu", 1.0))
    direction = tuple(sur_cfg.get("drag_direction", (1.0, 0.0, 0.0)))

    calib_path = results_dir / "visc_calibration.json"
    if tau_mode:
        # The viscous drag comes from the predicted wall shear stress; the
        # FD quadrature factor does not apply.
        calib = {"visc_scale": 1.0, "per_sample_mean": 1.0, "per_sample_std": 0.0,
                 "n": 0, "viscous_mode": "tau"}
        calib_path.write_text(json.dumps(calib, indent=2))
        logger.info("tau mode: visc_scale fixed to 1.0 (no FD calibration)")
    else:
        if calib_path.exists():
            calib = json.loads(calib_path.read_text())
        else:
            calib = calibrate_visc_scale(train_files, nu=nu, direction=direction)
            calib_path.write_text(json.dumps(calib, indent=2))
        logger.info(
            "viscous FD calibration: scale %.4f (per-sample %.4f +/- %.4f, n=%d)",
            calib["visc_scale"], calib["per_sample_mean"], calib["per_sample_std"], calib["n"],
        )
    sur_cfg = {**sur_cfg, "visc_scale": calib["visc_scale"],
               "viscous_mode": "tau" if tau_mode else "fd"}
    # Evaluation-side surrogate view of the model under training: owns the
    # drag-integral calibration; model.train() is re-entered every epoch.
    eval_surrogate = TransolverSurrogate(model, eval_norm, {**sur_cfg, "device": str(device)})

    best = float("inf")
    history = []

    def save(name: str, epoch: int, val_metrics: dict) -> None:
        torch.save(
            {
                "model_cfg": dict(model_cfg),
                "model_state": model.state_dict(),
                "norm_stats": stats,
                "surrogate_cfg": dict(sur_cfg),
                "train_cfg": dict(tr_cfg),
                "epoch": epoch,
                "val_metric": val_metrics,
            },
            results_dir / name,
        )

    for epoch in range(epochs):
        model.train()
        t0, losses = time.time(), []
        for batch in train_loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            pred = model(batch["x"][None])[0]
            loss, parts = field_loss(
                pred, batch["y"], batch["role"], batch["valid"],
                w_p=float(tr_cfg["w_p"]), w_U=float(tr_cfg["w_U"]),
                w_tau=float(tr_cfg.get("w_tau", 1.0)),
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(tr_cfg["grad_clip"]))
            opt.step()
            sched.step()
            losses.append(float(loss.detach()))

        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "t": time.time() - t0}
        if epoch % 5 == 0 or epoch == epochs - 1:
            row.update(evaluate(model, val_loader, eval_surrogate, device))
            tau_txt = f" tau_surf {row['tau_surf']:.4f}" if tau_mode else ""
            logger.info(
                "epoch %4d loss %.4f | val p_surf %.4f U_off %.4f%s drag_rel %.4f [%.1f s]",
                epoch, row["train_loss"], row["p_surf"], row["U_off"], tau_txt,
                row["drag_rel"], row["t"],
            )
            metric = row["p_surf"] + row["U_off"] + (row["tau_surf"] if tau_mode else 0.0)
            if metric < best:
                best = metric
                save("best.pt", epoch, row)
        history.append(row)
        if epoch % int(tr_cfg.get("checkpoint_every", 25)) == 0 or epoch == epochs - 1:
            save("latest.pt", epoch, row)
            (results_dir / "history.json").write_text(json.dumps(history, indent=2))

    save("latest.pt", epochs - 1, history[-1])
    (results_dir / "history.json").write_text(json.dumps(history, indent=2))
    logger.info("done; best val metric %.4f", best)


if __name__ == "__main__":
    main()

"""Train the Transolver flow surrogate on the generated OpenFOAM dataset.

Usage:
    uv run python scripts/train_transolver.py --config experiments/transolver/config_train.json

Checkpoints (``latest.pt`` / ``best.pt``) are self-contained: model config +
weights + normalization statistics + surrogate query-cloud config, so the
optimizer only needs the checkpoint path. Set ``training.overfit_n`` to a
small integer for the overfitting sanity check (P3 gate).
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
    compute_norm_stats,
    split_files,
)
from deepshapeopt.surrogate.drag import drag_from_fields
from deepshapeopt.surrogate.losses import field_loss, rel_l2
from deepshapeopt.surrogate.query_points import QueryCloud
from deepshapeopt.surrogate.transolver import Transolver

logger = logging.getLogger("train_transolver")


def drag_metric(batch: dict, pred_raw: torch.Tensor, nu: float, direction) -> float:
    """Relative drag error of one sample from de-normalized predictions."""
    P = int(batch["n_surface"])
    role = batch["role"]
    n_hat = torch.nn.functional.normalize(batch["area_normals"], dim=1)
    cloud = QueryCloud(
        feats=batch["x"],
        roles=role.to(torch.int8),
        n_surface=P,
        unit_normals=n_hat,
        area_normals=batch["area_normals"],
        delta=batch["delta"],
    )
    J, _ = drag_from_fields(pred_raw[:, :3], pred_raw[:, 3], cloud, nu=nu, direction=direction)
    J_ref = batch["drag_foam"]
    return float(abs(float(J) - J_ref) / (abs(J_ref) + 1e-12))


def evaluate(model, loader, norm, nu, direction, device) -> dict:
    model.eval()
    m = {"p_surf": [], "U_off": [], "drag_rel": []}
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            pred = model(batch["x"][None])[0]
            role, valid = batch["role"], batch["valid"]
            surf = role == 0
            off = (~surf) & valid
            m["p_surf"].append(float(rel_l2(pred[:, 3], batch["y"][:, 3], surf & valid)))
            m["U_off"].append(float(rel_l2(pred[:, :3], batch["y"][:, :3], off)))
            m["drag_rel"].append(
                drag_metric(batch, norm.denorm_y(pred), nu, direction)
            )
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
    tr_cfg, model_cfg, sur_cfg = cfg["training"], cfg["model"], cfg["surrogate"]
    device = torch.device(cfg.get("device", "cuda"))
    torch.manual_seed(int(cfg.get("seed", 0)))

    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(Path(cfg["data_dir"]).glob("sample_*.npz"))
    if not files:
        raise SystemExit(f"no samples found in {cfg['data_dir']}")
    train_files, val_files = split_files(
        files,
        val_fraction=cfg["split"].get("val_fraction", 0.1),
        holdout_family=cfg["split"].get("holdout_family"),
        seed=int(cfg.get("seed", 0)),
        min_surface_points=int(cfg["split"].get("min_surface_points", 0)),
    )
    overfit_n = tr_cfg.get("overfit_n")
    if overfit_n:
        train_files = train_files[: int(overfit_n)]
        val_files = train_files
    logger.info("train=%d val=%d samples", len(train_files), len(val_files))

    stats_path = results_dir / "dataset_stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
    else:
        stats = compute_norm_stats(train_files)
        stats_path.write_text(json.dumps(stats, indent=2))
    logger.info("norm stats over %d points from %d files", stats["n_points"], stats["n_files"])

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
    if calib_path.exists():
        calib = json.loads(calib_path.read_text())
    else:
        calib = calibrate_visc_scale(train_files, nu=nu, direction=direction)
        calib_path.write_text(json.dumps(calib, indent=2))
    sur_cfg = {**sur_cfg, "visc_scale": calib["visc_scale"]}
    logger.info(
        "viscous FD calibration: scale %.4f (per-sample %.4f +/- %.4f, n=%d)",
        calib["visc_scale"], calib["per_sample_mean"], calib["per_sample_std"], calib["n"],
    )
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
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(tr_cfg["grad_clip"]))
            opt.step()
            sched.step()
            losses.append(float(loss.detach()))

        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "t": time.time() - t0}
        if epoch % 5 == 0 or epoch == epochs - 1:
            row.update(evaluate(model, val_loader, eval_norm, nu, direction, device))
            logger.info(
                "epoch %4d loss %.4f | val p_surf %.4f U_off %.4f drag_rel %.4f [%.1f s]",
                epoch, row["train_loss"], row["p_surf"], row["U_off"], row["drag_rel"], row["t"],
            )
            metric = row["p_surf"] + row["U_off"]
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

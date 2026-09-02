"""Post-training bias calibration of the surrogate drag integral.

``train_transolver.py`` fits ``visc_scale`` on the *training targets*, i.e. it
corrects the probe-shell finite-difference quadrature only. The trained
network adds its own systematic offset (v2: predicted drag was 12% low across
every family and variant, with tiny scatter -- Spearman 0.995). This script
measures that residual bias on the *predictions* over the training split and
folds it into the checkpoint's ``visc_scale`` / a new ``pressure_scale``, so
the surrogate objective is unbiased without retraining.

Least squares over the training samples: minimize
    sum_i ( J_foam,i - (a * J_p,i + b * J_visc,i) )^2
with a (pressure_scale) and b (extra viscous factor) -- both terms are scaled
separately because the network's pressure and velocity errors differ.

Usage:
    uv run python scripts/calibrate_surrogate_bias.py \
        --config experiments/transolver/config_train.json [--checkpoint best.pt]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from deepshapeopt.config import ExperimentSpecifications
from deepshapeopt.surrogate.dataset import FlowFieldDataset, split_files
from deepshapeopt.surrogate.drag import drag_from_fields
from deepshapeopt.surrogate.predictor import TransolverSurrogate
from deepshapeopt.surrogate.query_points import QueryCloud

logger = logging.getLogger("calibrate_surrogate_bias")


def drag_terms(surrogate, batch, device):
    """(J_p, J_visc) of one sample from the network's predictions."""
    pred = surrogate.model(batch["x"][None].to(device))[0]
    y_raw = surrogate.norm.denorm_y(pred)
    an = batch["area_normals"].to(device)
    cloud = QueryCloud(
        feats=batch["x"].to(device),
        roles=batch["role"].to(torch.int8).to(device),
        n_surface=int(batch["n_surface"]),
        unit_normals=torch.nn.functional.normalize(an, dim=1),
        area_normals=an,
        delta=batch["delta"].to(device),
    )
    _, d = drag_from_fields(
        y_raw[:, :3], y_raw[:, 3], cloud,
        nu=surrogate.nu, direction=surrogate.direction,
        u_inf=surrogate.u_inf, a_ref=surrogate.a_ref,
        visc_scale=surrogate.visc_scale,
    )
    return d["J_p"], d["J_visc"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = ExperimentSpecifications(args.config)
    results_dir = Path(cfg["results_dir"])
    ckpt_path = results_dir / args.checkpoint
    surrogate = TransolverSurrogate.from_config(
        {**cfg["surrogate"], "checkpoint": str(ckpt_path), "device": cfg.get("device", "cuda")}
    )
    device = surrogate.device

    files = sorted(
        f for pat in ("sample_*.npz", "traj_*.npz")
        for f in Path(cfg["data_dir"]).glob(pat)
    )
    train_files, _ = split_files(
        files,
        val_fraction=cfg["split"].get("val_fraction", 0.1),
        holdout_family=cfg["split"].get("holdout_family"),
        seed=int(cfg.get("seed", 0)),
        min_surface_points=int(cfg["split"].get("min_surface_points", 0)),
    )
    stats = json.loads((results_dir / "dataset_stats.json").read_text())
    ds = FlowFieldDataset(train_files, stats)
    logger.info("Calibrating on %d training samples", len(ds))

    Jp, Jv, Jf = [], [], []
    with torch.no_grad():
        for i in range(len(ds)):
            b = ds[i]
            p_term, v_term = drag_terms(surrogate, b, device)
            Jp.append(p_term)
            Jv.append(v_term)
            Jf.append(float(b["drag_foam"]))

    A = np.stack([np.array(Jp), np.array(Jv)], axis=1)
    y = np.array(Jf)
    (a, b_), *_ = np.linalg.lstsq(A, y, rcond=None)
    pred_before = A.sum(axis=1)
    pred_after = A @ np.array([a, b_])
    err_before = np.abs(pred_before - y) / np.abs(y)
    err_after = np.abs(pred_after - y) / np.abs(y)

    logger.info("pressure_scale = %.4f, viscous factor = %.4f", a, b_)
    logger.info(
        "train drag error: median %.1f%% -> %.1f%% | bias %+.1f%% -> %+.1f%%",
        100 * np.median(err_before), 100 * np.median(err_after),
        100 * (pred_before / y - 1).mean(), 100 * (pred_after / y - 1).mean(),
    )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sur_cfg = dict(ckpt.get("surrogate_cfg", {}))
    sur_cfg["pressure_scale"] = float(a)
    sur_cfg["visc_scale"] = float(surrogate.visc_scale * b_)
    sur_cfg["bias_calibration"] = {
        "n_train": len(ds),
        "median_err_before": float(np.median(err_before)),
        "median_err_after": float(np.median(err_after)),
    }
    ckpt["surrogate_cfg"] = sur_cfg
    torch.save(ckpt, ckpt_path)
    logger.info("checkpoint updated: %s", ckpt_path)


if __name__ == "__main__":
    main()

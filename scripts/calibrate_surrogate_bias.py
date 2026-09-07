"""Post-training bias calibration of the surrogate drag integral.

``train_transolver.py`` fits ``visc_scale`` on the *training targets* (FD
mode only), i.e. it corrects the probe-shell finite-difference quadrature.
The trained network adds its own systematic offset (v2: predicted drag was
12% low across every family and variant, with tiny scatter -- Spearman
0.995). This script measures that residual bias on the *predictions* over the
training split and folds it into the checkpoint's ``pressure_scale`` and
``visc_scale``, so the surrogate objective is unbiased without retraining.
In tau mode the viscous term is the predicted wall shear stress and the
fitted viscous factor is expected to sit close to 1.

Least squares over the training samples: minimize
    sum_i ( J_foam,i - (a * J_p,i + b * J_visc,i) )^2
where J_p / J_visc are evaluated with the checkpoint's CURRENT scales, so the
fitted factors compound onto them (a second run is a no-op up to noise).

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
from deepshapeopt.surrogate.dataset import FlowFieldDataset, cloud_from_batch, split_files
from deepshapeopt.surrogate.predictor import TransolverSurrogate

logger = logging.getLogger("calibrate_surrogate_bias")


def drag_terms(surrogate, batch, device):
    """(J_p, J_visc) of one sample from the network's predictions."""
    b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    pred = surrogate.model(b["x"][None])[0]
    y_raw = surrogate.norm.denorm_y(pred)
    _, d = surrogate.drag_from_prediction(y_raw, cloud_from_batch(b))
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
        require_tau=surrogate.viscous_mode == "tau",
    )
    stats = json.loads((results_dir / "dataset_stats.json").read_text())
    ds = FlowFieldDataset(train_files, stats)
    logger.info("Calibrating on %d training samples (viscous mode %s)",
                len(ds), surrogate.viscous_mode)

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

    logger.info("pressure factor = %.4f, viscous factor = %.4f (on top of the current scales)", a, b_)
    logger.info(
        "train drag error: median %.1f%% -> %.1f%% | bias %+.1f%% -> %+.1f%%",
        100 * np.median(err_before), 100 * np.median(err_after),
        100 * (pred_before / y - 1).mean(), 100 * (pred_after / y - 1).mean(),
    )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sur_cfg = dict(ckpt.get("surrogate_cfg", {}))
    sur_cfg["pressure_scale"] = float(surrogate.pressure_scale * a)
    sur_cfg["visc_scale"] = float(surrogate.visc_scale * b_)
    sur_cfg["bias_calibration"] = {
        "n_train": len(ds),
        "viscous_mode": surrogate.viscous_mode,
        "pressure_factor": float(a),
        "viscous_factor": float(b_),
        "median_err_before": float(np.median(err_before)),
        "median_err_after": float(np.median(err_after)),
    }
    ckpt["surrogate_cfg"] = sur_cfg
    torch.save(ckpt, ckpt_path)
    logger.info("checkpoint updated: %s (pressure_scale %.4f, visc_scale %.4f)",
                ckpt_path, sur_cfg["pressure_scale"], sur_cfg["visc_scale"])


if __name__ == "__main__":
    main()

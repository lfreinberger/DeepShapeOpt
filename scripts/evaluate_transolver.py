"""Evaluate a trained Transolver surrogate on the held-out validation split.

Reports per-sample relative L2 errors for p (surface) and U (off-surface),
the drag error of the differentiable drag integral on predicted fields vs the
OpenFOAM objective, and the Spearman rank correlation of predicted vs true
drag (what shape *ranking* the optimizer would see). Writes ``evaluation.csv``
into the training results dir.

Usage:
    uv run python scripts/evaluate_transolver.py \
        --config experiments/transolver/config_train.json [--checkpoint best.pt]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
import torch

from deepshapeopt.config import ExperimentSpecifications
from deepshapeopt.surrogate.dataset import FlowFieldDataset, split_files
from deepshapeopt.surrogate.drag import drag_from_fields
from deepshapeopt.surrogate.losses import rel_l2
from deepshapeopt.surrogate.predictor import TransolverSurrogate
from deepshapeopt.surrogate.query_points import QueryCloud

logger = logging.getLogger("evaluate_transolver")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = ExperimentSpecifications(args.config)
    results_dir = Path(cfg["results_dir"])
    ckpt = results_dir / args.checkpoint
    surrogate = TransolverSurrogate.from_config(
        {**cfg["surrogate"], "checkpoint": str(ckpt), "device": cfg.get("device", "cuda")}
    )
    device = surrogate.device

    files = sorted(
        f for pat in ("sample_*.npz", "traj_*.npz")
        for f in Path(cfg["data_dir"]).glob(pat)
    )
    _, val_files = split_files(
        files,
        val_fraction=cfg["split"].get("val_fraction", 0.1),
        holdout_family=cfg["split"].get("holdout_family"),
        seed=int(cfg.get("seed", 0)),
        min_surface_points=int(cfg["split"].get("min_surface_points", 0)),
    )
    stats = json.loads((results_dir / "dataset_stats.json").read_text())
    ds = FlowFieldDataset(val_files, stats)
    logger.info("Evaluating %d validation samples", len(ds))

    rows = []
    with torch.no_grad():
        for i in range(len(ds)):
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in ds[i].items()}
            pred = surrogate.model(b["x"][None])[0]
            y = b["y"]
            role, valid = b["role"], b["valid"]
            surf = role == 0
            off = (~surf) & valid

            an = b["area_normals"]
            cloud = QueryCloud(
                feats=b["x"], roles=role.to(torch.int8), n_surface=int(b["n_surface"]),
                unit_normals=torch.nn.functional.normalize(an, dim=1),
                area_normals=an, delta=b["delta"],
            )
            y_raw = surrogate.norm.denorm_y(pred)
            J, parts = drag_from_fields(
                y_raw[:, :3], y_raw[:, 3], cloud,
                nu=surrogate.nu, direction=surrogate.direction,
                u_inf=surrogate.u_inf, a_ref=surrogate.a_ref,
            )
            rows.append({
                "sample": Path(b["path"]).name,
                "family": b["family"],
                "p_surf_relL2": float(rel_l2(pred[:, 3], y[:, 3], surf & valid)),
                "U_off_relL2": float(rel_l2(pred[:, :3], y[:, :3], off)),
                "J_pred": float(J),
                "J_foam": float(b["drag_foam"]),
            })

    for r in rows:
        r["drag_rel_err"] = abs(r["J_pred"] - r["J_foam"]) / max(abs(r["J_foam"]), 1e-12)

    out_csv = results_dir / "evaluation.csv"
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    from scipy.stats import spearmanr

    jp = np.array([r["J_pred"] for r in rows])
    jf = np.array([r["J_foam"] for r in rows])
    rho = float(spearmanr(jp, jf).statistic) if len(rows) > 2 else float("nan")

    print(f"\n{'sample':<22} {'family':<12} {'p_relL2':>8} {'U_relL2':>8} {'J_pred':>9} {'J_foam':>9} {'dJ_rel':>7}")
    for r in rows:
        print(f"{r['sample']:<22} {r['family']:<12} {r['p_surf_relL2']:8.4f} "
              f"{r['U_off_relL2']:8.4f} {r['J_pred']:9.4f} {r['J_foam']:9.4f} "
              f"{100 * r['drag_rel_err']:6.1f}%")
    print(f"\nmean p relL2 {np.mean([r['p_surf_relL2'] for r in rows]):.4f} | "
          f"mean U relL2 {np.mean([r['U_off_relL2'] for r in rows]):.4f} | "
          f"drag rel err mean {100 * np.mean([r['drag_rel_err'] for r in rows]):.1f}% "
          f"median {100 * np.median([r['drag_rel_err'] for r in rows]):.1f}% | "
          f"Spearman rho {rho:.4f}")
    print(f"written: {out_csv}")


if __name__ == "__main__":
    main()

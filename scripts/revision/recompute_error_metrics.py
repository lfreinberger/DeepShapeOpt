"""Recompute Table 3 / Figure 9 reconstruction errors from archived VTPs.

No reconstruction is re-run. ``error_metrics.json`` was never written for the
manuscript runs, but the inputs it would have been computed from survive:

* ``gt_sdf_samples_surface.vtp`` / ``rec_sdf_samples_surface.vtp`` -> MAE_SDF,
  the near-surface signed-distance error of Eq. (22).
* ``<stem>_mesh_sdf_error.vtp``                                   -> MAE_geom,
  the mesh-vertex geometric error of Eq. (23).

READ-ONLY BY CONSTRUCTION.  The archive is deposited manuscript data, so this
script never writes into it.  In particular it deliberately does *not* call
``compute_metrics_from_vtp``: that function unconditionally saves an
``sdf_error.vtp`` next to its input (error_metrics.py, ``save_error_vtp``).
Instead it reuses the pure reduction ``compute_near_surface_metrics`` and reads
the mesh error array directly.

Two different run families back the manuscript, and they must not be mixed:

* Figure 9 (tiling x code-length sweep) -> ``..._primitives_cl{08,16,32}_tiling_*``
* Table 3 (one row per shape)           -> the untagged production runs

Usage:
    uv run python scripts/revision/recompute_error_metrics.py
    uv run python scripts/revision/recompute_error_metrics.py --json-out out.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pyvista as pv

from DeepSDFStruct.deep_sdf.metrics.error_metrics import compute_near_surface_metrics

ARCHIVE = Path(
    "/storage/lfrei/Archive/DeepShapeOpt_archive_paper/experiments/reconstruction"
)
CUTOFF = 0.1  # tau = delta, Section 3.1.2

# Table 3: (label, d_lat, tiling, directory, printed MAE_SDF, printed MAE_geom) in 1e-4
TABLE3 = [
    ("Flow channel", 32, "1x8x8",
     "feed_channel/results/reconstruction/angusskanal_10x2_vereinfacht_tiling_1x8x8", 5.73, 5.31),
    ("Propeller", 32, "12x12x4",
     "ship_propeller/results/reconstruction/ship_propeller_tiling_12x12x4", 4.95, 5.01),
    ("Rim", 32, "6x12x12",
     "rim/results/reconstruction/rim_tiling_6x12x12", 6.59, 8.43),
    ("Dog", 32, "4x12x12",
     "shiba/results/reconstruction/shiba_tiling_4x12x12", 4.20, 4.15),
]

# Figure 9: the flow-channel sweep.
FIG9_TILINGS = ["1x2x2", "1x4x4", "1x6x6", "1x8x8", "1x10x10", "1x12x12"]
FIG9_CODES = ["cl08", "cl16", "cl32"]
FIG9_DIR = "feed_channel/results/reconstruction"
FIG9_STEM = "angusskanal_10x2_vereinfacht"


def mae_sdf(case_dir: Path) -> dict | None:
    """Near-surface SDF error (Eq. 22), reduced in memory."""
    gt = case_dir / "gt_sdf_samples_surface.vtp"
    rec = case_dir / "rec_sdf_samples_surface.vtp"
    if not (gt.is_file() and rec.is_file()):
        return None
    gt_sdf = np.asarray(pv.read(gt).point_data["SDF"]).reshape(-1)
    rec_sdf = np.asarray(pv.read(rec).point_data["SDF"]).reshape(-1)
    if gt_sdf.shape != rec_sdf.shape:
        raise SystemExit(f"{case_dir}: gt/rec sample counts differ")
    return compute_near_surface_metrics(gt_sdf, rec_sdf, CUTOFF)


def mae_geom(case_dir: Path) -> dict | None:
    """Mesh-vertex geometric error (Eq. 23), reduced in memory."""
    hits = sorted(glob.glob(os.path.join(case_dir, "*_mesh_sdf_error.vtp")))
    if not hits:
        return None
    err = np.asarray(pv.read(hits[0]).point_data["sdf_error"]).reshape(-1)
    abs_err = np.abs(err)
    return {
        "num_vertices": int(err.size),
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "median": float(np.median(abs_err)),
        "p95": float(np.quantile(abs_err, 0.95)),
        "max": float(abs_err.max()),
        "mean_signed_error": float(err.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--latex", action="store_true")
    args = parser.parse_args()

    if not args.archive.is_dir():
        raise SystemExit(f"archive not found: {args.archive}")

    out: dict = {"cutoff": CUTOFF, "table3": [], "figure9": []}

    # ---- Table 3, with the reproduction gate -------------------------------
    print("=== Table 3 (per-shape reconstruction error) ===")
    print(f"{'shape':<14}{'tiling':<10}{'MAE_SDF':>12}{'printed':>10}"
          f"{'MAE_geom':>12}{'printed':>10}  gate")
    n_ok = 0
    for label, d_lat, tiling, rel, p_sdf, p_geom in TABLE3:
        case = args.archive / rel
        if not case.is_dir():
            print(f"{label:<14}{tiling:<10}  MISSING {case}")
            continue
        s, g = mae_sdf(case), mae_geom(case)
        if s is None or g is None:
            print(f"{label:<14}{tiling:<10}  incomplete artifacts")
            continue
        s4, g4 = s["mae"] * 1e4, g["mae"] * 1e4
        ok = abs(s4 - p_sdf) < 0.005 and abs(g4 - p_geom) < 0.005
        n_ok += ok
        print(f"{label:<14}{tiling:<10}{s4:>12.2f}{p_sdf:>10.2f}"
              f"{g4:>12.2f}{p_geom:>10.2f}  {'OK' if ok else 'MISMATCH'}")
        out["table3"].append(
            {"shape": label, "latent_dim": d_lat, "tiling": tiling,
             "dir": rel, "sdf_sample_error": s, "mesh_vertex_error": g,
             "printed_mae_sdf_e4": p_sdf, "printed_mae_geom_e4": p_geom,
             "reproduces": bool(ok)}
        )

    # Hard gate: the plan forbids publishing recomputed numbers unless the
    # convention is proven against the printed table.
    if n_ok == 0:
        raise SystemExit(
            "GATE FAILED: no Table 3 row reproduced its printed value. "
            "The cutoff or run-directory convention is wrong -- do not publish."
        )
    print(f"\ngate: {n_ok}/{len(TABLE3)} printed rows reproduced to printed precision")

    # ---- Figure 9 sweep ---------------------------------------------------
    print("\n=== Figure 9 (flow-channel sweep, MAE_SDF x 1e-4) ===")
    print(f"{'tiling':<10}" + "".join(f"{c:>10}" for c in FIG9_CODES))
    for tiling in FIG9_TILINGS:
        row = [f"{tiling:<10}"]
        for code in FIG9_CODES:
            case = args.archive / FIG9_DIR / f"{FIG9_STEM}_primitives_{code}_tiling_{tiling}"
            s = mae_sdf(case) if case.is_dir() else None
            row.append(f"{s['mae'] * 1e4:>10.2f}" if s else f"{'--':>10}")
            if s:
                out["figure9"].append(
                    {"tiling": tiling, "code_length": int(code[2:]),
                     "sdf_sample_error": s}
                )
        print("".join(row))

    if args.latex:
        print("\n=== LaTeX (Table 3, with design-variable column) ===")
        for r in out["table3"]:
            til = r["tiling"].replace("x", r" \times ")
            print(f"{r['shape']} & {r['latent_dim']} & ${til}$ & "
                  f"{r['sdf_sample_error']['mae'] * 1e4:.2f} & "
                  f"{r['mesh_vertex_error']['mae'] * 1e4:.2f} \\\\")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()

"""Rebuild Table 4 (and the Fig. 14 / Fig. 20 headline numbers) from stored histories.

No optimization is re-run: every manuscript optimization run survives as an
``optimization_history.csv`` next to the ``config_log.json`` that produced it.

Conventions, read off the stored files:

* Row 1 is the *initial* design (``objective_normalized == 1.0`` exactly), so the
  number of MMA updates is ``len(rows) - 1``.
* Drag reduction is ``1 - objective_normalized`` of the last row.
* Design variables: latent runs use ``d_lat * prod(tiling_i + degree_i)``,
  FFD runs use ``3 * prod(n_control_points_i)``.

Every run directory found is reported, not just the five in the manuscript, so
that a run matching a disputed iteration count cannot be silently missed.

Usage:
    uv run python scripts/revision/rebuild_table4.py
    uv run python scripts/revision/rebuild_table4.py --latex
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

ARCHIVE = Path(
    "/storage/lfrei/Archive/DeepShapeOpt-old-private-backup/experiments/optimization"
    "/drag_optimization_cube"
)

# Which stored run backs which manuscript entry, and what the manuscript printed.
# (run_dir, manuscript label, printed n_design, printed iterations, printed reduction %)
MANUSCRIPT = [
    ("results_ffd_cube_with_cylinders_5x5x5", "FFD (5x5x5 control points)", 375, 46, 14.2),
    ("results_ffd_cube_with_cylinders_7x7x7", "FFD (7x7x7 control points)", 1029, 32, 15.7),
    ("results_cube_with_cylinders", "Proposed neural SDF method", 864, 33, 16.7),
    # Not part of Table 4, but the same forensic treatment is worth having.
    ("results_cube", "Experiment 1 (cube, Fig. 14)", 864, 30, 15.0),
    ("results_cube_with_holes", "Experiment 3 (perforated, Fig. 20)", 864, 21, 25.4),
]


def design_variables(cfg: dict) -> int | None:
    """Design-variable count implied by a stored config_log.json."""
    rec = cfg.get("reconstruction", {})
    if "n_control_points" in rec:  # FFD: displacement of each control point, 3 DOF
        return 3 * math.prod(rec["n_control_points"])
    if "tiling" in rec:  # latent field: see build_parameter_spline
        degree = rec.get("spline_degree", [1, 1, 1])
        n_cp = math.prod(t + p for t, p in zip(rec["tiling"], degree))
        model = str(rec.get("model_path", ""))
        for tag, d_lat in (("cl08", 8), ("cl16", 16), ("cl32", 32)):
            if tag in model:
                return n_cp * d_lat
        return None
    return None


def summarize(run_dir: Path) -> dict | None:
    hist = run_dir / "optimization" / "optimization_history.csv"
    if not hist.is_file():
        return None
    with hist.open() as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None

    norm = [float(r["objective_normalized"]) for r in rows]
    if abs(norm[0] - 1.0) > 1e-9:
        raise SystemExit(
            f"{run_dir.name}: first row is not the baseline "
            f"(objective_normalized={norm[0]}); the rows-1 convention does not hold"
        )

    cfg_path = run_dir / "optimization" / "config_log.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.is_file() else {}
    opt = cfg.get("optimization", {})

    best = min(range(len(norm)), key=lambda i: norm[i])
    return {
        "run": run_dir.name,
        "n_rows": len(rows),
        "n_updates": len(rows) - 1,
        "n_design": design_variables(cfg),
        "reduction_final_pct": (1.0 - norm[-1]) * 100.0,
        "reduction_best_pct": (1.0 - norm[best]) * 100.0,
        "best_update": best,
        "num_iter_cap": opt.get("num_iter"),
        "hit_cap": opt.get("num_iter") is not None and len(rows) - 1 >= opt["num_iter"] - 1,
        "conv_tol": opt.get("convergence_obj_tol"),
        "conv_window": opt.get("convergence_window"),
        "elapsed_s": float(rows[-1]["elapsed_s"]) if rows[-1].get("elapsed_s") else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latex", action="store_true", help="emit LaTeX rows for Table 4")
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--json-out", type=Path, default=None, help="write the results as JSON")
    args = parser.parse_args()

    if not args.archive.is_dir():
        raise SystemExit(f"archive not found: {args.archive}")

    found = {}
    for run_dir in sorted(args.archive.iterdir()):
        if not run_dir.is_dir():
            continue
        s = summarize(run_dir)
        if s:
            found[run_dir.name] = s

    print("=== all stored runs in the archive ===")
    hdr = f"{'run':<44} {'upd':>4} {'n_des':>6} {'final%':>7} {'best%':>7} {'best@':>6} {'cap':>4}"
    print(hdr)
    print("-" * len(hdr))
    for s in found.values():
        print(
            f"{s['run']:<44} {s['n_updates']:>4} "
            f"{(s['n_design'] if s['n_design'] else -1):>6} "
            f"{s['reduction_final_pct']:>7.2f} {s['reduction_best_pct']:>7.2f} "
            f"{s['best_update']:>6} {'yes' if s['hit_cap'] else 'no':>4}"
        )

    print("\n=== comparison with the submitted manuscript ===")
    for run, label, m_des, m_it, m_red in MANUSCRIPT:
        s = found.get(run)
        if s is None:
            print(f"{label:<38} MISSING run directory {run}")
            continue
        des_ok = "ok" if s["n_design"] == m_des else f"MISMATCH (stored {s['n_design']})"
        it_ok = "ok" if s["n_updates"] == m_it else f"MISMATCH (stored {s['n_updates']})"
        red_ok = (
            "ok"
            if abs(s["reduction_final_pct"] - m_red) < 0.05
            else f"MISMATCH (stored {s['reduction_final_pct']:.2f})"
        )
        print(f"{label}")
        print(f"    design vars : printed {m_des:<6} {des_ok}")
        print(f"    iterations  : printed {m_it:<6} {it_ok}")
        print(f"    reduction   : printed {m_red:<6} {red_ok}")
        if s["hit_cap"]:
            print(
                f"    NOTE: run reached its num_iter cap ({s['num_iter_cap']}); the convergence "
                f"criterion (tol={s['conv_tol']}, window={s['conv_window']}) was never met"
            )

    if args.latex:
        print("\n=== LaTeX (corrected Table 4) ===")
        for run, label, _m_des, _m_it, _m_red in MANUSCRIPT[:3]:
            s = found[run]
            print(
                f"{label} & {s['n_design']} & {s['n_updates']} "
                f"& {s['reduction_final_pct']:.1f} \\\\"
            )

    if args.json_out:
        payload = {
            "archive": str(args.archive),
            "convention": "row 1 is the initial design; n_updates = n_rows - 1",
            "all_runs": list(found.values()),
            "manuscript_comparison": [
                {
                    "label": label,
                    "run": run,
                    "printed": {"n_design": m_des, "iterations": m_it, "reduction_pct": m_red},
                    "stored": found.get(run),
                }
                for run, label, m_des, m_it, m_red in MANUSCRIPT
            ],
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()

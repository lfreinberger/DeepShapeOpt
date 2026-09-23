"""Path consistency: does the adjoint gradient predict J along the optimization path?

Works on a finished (or running) optimization that was run with ``debug: true`` and a
``heavy_data_output_path``: every iteration k then left the wall surface with the normal
component of the conservative point sensitivity (``sens_series/check_sens_*_{k}.vtp``). For
every committed step k -> k+1 the script compares the realized objective change with the
gradients at BOTH ends of the step, projected onto the wall displacement d (distance of the
wall points to the other surface along their normal):

    pred_k    = sum_i sens_k,i   * d_i        first-order prediction at the start design
    pred_k+1  = the same directional derivative, evaluated at the end design
    rho_fwd   = (J_k+1 - J_k) / pred_k
    rho_trap  = (J_k+1 - J_k) / (0.5 * (pred_k + pred_k+1))      exact for a quadratic J

This is a finite-difference check along the actual path, remeshing included, at no extra CFD
cost. Reading the result:

    rho_fwd ~ rho_trap ~ 1                     the gradient predicts J
    pred_k < 0 < pred_k+1, rho_trap ~ 1        overshoot: the directional derivative changed
                                               sign inside the step (step too long for the
                                               curvature), the gradient itself is fine
    pred_k ~ pred_k+1 < 0, both rho ~ 0        error floor: the gradient keeps promising a
                                               decrease that is not realized
    |realized - trapezoid| of the steps without overshoot bounds the evaluation noise of J

Only the normal component of the sensitivity is stored and d is a closest-point distance, so
the numbers carry a few percent of projection error. Newer runs log the exact latent-space
values (``path_rho_fwd`` / ``path_rho_trap`` in ``optimization_history.csv``); the script
prints them next to its own when present.

Usage, from an application repo root (each argument is the ``optimization`` directory of a
run or any directory above it, e.g. the results folder):

    uv run ../DeepShapeOpt/scripts/check_path_consistency.py \\
        experiments/optimization/simpleExtrusionDie/results_simpleDie_dafoam
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv

HISTORY = "optimization_history.csv"
WINDOW_MIN_STEPS = 60  # longer runs are summarized per window instead of listed per step


def find_history_files(path: Path) -> list[Path]:
    if (path / HISTORY).is_file():
        return [path / HISTORY]
    files = sorted(path.glob(f"**/{HISTORY}"))
    if not files:
        raise FileNotFoundError(f"no {HISTORY} at or below {path}")
    return files


def sens_series_dir(opt_dir: Path) -> Path:
    """Heavy-data ``sens_series`` of a run, derived like ``config.make_experiment_paths``."""
    config = json.loads((opt_dir / "config_log.json").read_text())
    heavy_root = config["optimization"].get("heavy_data_output_path")
    if heavy_root is None:
        raise FileNotFoundError(f"{opt_dir}: no heavy_data_output_path, pass --sens-series")
    setup_dir = opt_dir.parent.resolve()
    project_root = setup_dir
    while project_root != project_root.parent and not (project_root / "pyproject.toml").exists():
        project_root = project_root.parent
    return Path(os.path.expandvars(heavy_root)) / setup_dir.relative_to(project_root) / "sens_series"


def load_surface(path: Path):
    """Wall surface with unit vertex normals oriented like the driver's (face winding)."""
    mesh = pv.read(path)
    points = np.asarray(mesh.points, dtype=np.float64)
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    face_normals = np.cross(points[faces[:, 1]] - points[faces[:, 0]],
                            points[faces[:, 2]] - points[faces[:, 0]])
    normals = np.zeros_like(points)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)
    return mesh, points, normals, np.asarray(mesh.point_data["sens"], dtype=np.float64)


def normal_displacement(points, normals, target) -> np.ndarray:
    """Signed distance of ``points`` to the surface ``target`` along their own normals."""
    _, closest = target.find_closest_cell(points, return_closest_point=True)
    return ((closest - points) * normals).sum(axis=1)


def path_table(history: pd.DataFrame, files: list[Path]) -> pd.DataFrame:
    J = history["objective"].to_numpy()
    n = min(len(files), len(J))
    rows = []
    current = load_surface(files[0])
    for k in range(n - 1):
        following = load_surface(files[k + 1])
        d_fwd = normal_displacement(current[1], current[2], following[0])
        d_back = normal_displacement(following[1], following[2], current[0])
        pred_start = float((current[3] * d_fwd).sum())
        pred_end = -float((following[3] * d_back).sum())
        real = J[k + 1] - J[k]
        trap = 0.5 * (pred_start + pred_end)
        rows.append({
            "k": k, "J/J0": J[k] / J[0], "real": real / abs(J[0]),
            "pred_k": pred_start / abs(J[0]), "pred_k+1": pred_end / abs(J[0]),
            "rho_fwd": real / pred_start if pred_start else np.nan,
            "rho_trap": real / trap if trap else np.nan,
            "d_max_mm": 1e3 * np.abs(d_fwd).max(),
        })
        current = following
    table = pd.DataFrame(rows)
    # Exact latent-space values of newer runs: row k+1 holds the step k -> k+1.
    for column in ("path_rho_fwd", "path_rho_trap"):
        if column in history.columns:
            table[column.replace("path_", "driver_")] = history[column].to_numpy()[1:n]
    return table


def summarize(table: pd.DataFrame) -> None:
    overshoot = table["pred_k+1"] > 0
    residual = (table["real"] - table["real"] / table["rho_trap"]).abs()
    for name, part in (("no overshoot", table[~overshoot]), ("overshoot", table[overshoot])):
        if part.empty:
            continue
        q = part["rho_trap"].quantile([0.25, 0.5, 0.75]).to_numpy()
        print(f"  {name:12s} n={len(part):3d}  rho_fwd median {part['rho_fwd'].median():+.2f}  "
              f"rho_trap q25/50/75 {q[0]:+.2f} {q[1]:+.2f} {q[2]:+.2f}  "
              f"sum realized {part['real'].sum():+.3e}")
    quiet = residual[~overshoot]
    if not quiet.empty:
        print(f"  |realized - trapezoid| without overshoot, median: {quiet.median():.2e} of |J0| "
              f"(upper bound of the evaluation noise of J)")
    floor = (~overshoot) & (table["pred_k"] < 0) & (table["rho_trap"].abs() < 0.25)
    print(f"  steps: {overshoot.mean():.0%} overshoot, {floor.mean():.0%} error-floor-like "
          f"(consistent negative prediction, |rho_trap| < 0.25)")


def report(history_file: Path, sens_dir: Path | None, write_csv: bool) -> None:
    opt_dir = history_file.parent
    sens_dir = sens_dir or sens_series_dir(opt_dir)
    files = sorted(sens_dir.glob("*.vtp"))
    if len(files) < 2:
        raise FileNotFoundError(f"fewer than two sensitivity surfaces in {sens_dir}")
    table = path_table(pd.read_csv(history_file), files)
    print(f"\n=== {opt_dir}\n    {len(table)} steps, surfaces from {sens_dir}")
    fmt = lambda x: f"{x:+.4g}"  # noqa: E731
    if len(table) <= WINDOW_MIN_STEPS:
        print(table.to_string(index=False, float_format=fmt))
    else:
        edges = np.unique(np.r_[0, 10, 30, np.arange(60, len(table), 60), len(table)])
        for a, b in zip(edges[:-1], edges[1:]):
            w = table[(table["k"] >= a) & (table["k"] < b)]
            print(f"  k {a:3d}-{b:3d}: J/J0 {w['J/J0'].iloc[0]:.4f} -> {w['J/J0'].iloc[-1]:.4f}  "
                  f"sum realized {w['real'].sum():+.2e} predicted {w['pred_k'].sum():+.2e}  "
                  f"median rho_fwd {w['rho_fwd'].median():+.2f} rho_trap {w['rho_trap'].median():+.2f}  "
                  f"overshoot {(w['pred_k+1'] > 0).mean():.0%}")
    summarize(table)
    if write_csv:
        out = opt_dir / "path_consistency.csv"
        table.to_csv(out, index=False)
        print(f"  written: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="+", type=Path,
                        help="optimization directory of a run, or any directory above it")
    parser.add_argument("--sens-series", type=Path, default=None,
                        help="directory of the check_sens_*.vtp series (default: derived from "
                             "heavy_data_output_path in config_log.json); single run only")
    parser.add_argument("--csv", action="store_true",
                        help="write the per-step table as path_consistency.csv next to the history")
    args = parser.parse_args()
    history_files = [f for path in args.paths for f in find_history_files(path)]
    if args.sens_series is not None and len(history_files) != 1:
        parser.error("--sens-series needs exactly one run")
    for history_file in history_files:
        report(history_file, args.sens_series, args.csv)


if __name__ == "__main__":
    main()

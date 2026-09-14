"""Static out-of-distribution test of surrogate checkpoints on the FOAM-verified
anchor shapes of a finished optimization run (zero OpenFOAM calls).

A surrogate-driven run with ``reanchor_every`` logs the true objective at
every anchor iteration (``Re-anchor @ iter k: J_foam=...``) and stores the
design parameters of every iteration in ``parameters_series/``. This script
rebuilds those anchor shapes, evaluates one or more checkpoints on the
identical query clouds and reports the drift ``J_surr / J_foam`` per anchor:
how each checkpoint extrapolates along the optimizer's path. Pass the
checkpoints of an ablation (e.g. geo / lat / geo+lat) side by side.

Usage:
    uv run python scripts/evaluate_anchor_shapes.py \
        --config experiments/drag_cube/config_latent_cube_transolver_nopca_tau_l1.json \
        --checkpoints geo=experiments/transolver/results_v4_tau/best.pt \
                      lat=experiments/transolver/results_v5_lat/best.pt \
                      geolat=experiments/transolver/results_v5_geolat/best.pt \
        [--iterations 0 10 20 30] [--out anchor_eval.csv]

Without ``--checkpoints`` the run config's own surrogate is evaluated (its
drift must reproduce the ``scale`` values of the run log). Without
``--iterations`` the anchors are taken from the LAST run segment of
``run.log`` (the log is appended across restarts; ``parameters_series`` holds
the latest run). ``param_k.pt`` is written AFTER the MMA step of iteration
``k``, i.e. it is the design of iteration ``k + 1``: anchor ``k`` therefore
uses ``param_{k-1}.pt``, and anchor 0 the reconstruction itself.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from pathlib import Path

import torch

import deepshapeopt.config as config
from deepshapeopt.config import ExperimentSpecifications
from deepshapeopt.hexmesh import SdfHexMeshPipeline
from deepshapeopt.runtime import configure_logging
from deepshapeopt.shape_optimization import (
    build_lattice,
    run_reconstruction,
    setup_model_and_domain,
)
from deepshapeopt.surrogate.predictor import TransolverSurrogate

LOGGER = logging.getLogger(__name__)

_ANCHOR_RE = re.compile(r"Re-anchor @ iter (\d+): J_foam=([-\d.eE+]+) J_surr=([-\d.eE+]+)")


def anchors_from_log(run_log: Path) -> dict[int, tuple[float, float]]:
    """``{iteration: (J_foam, J_surr)}`` of the last run segment in ``run.log``."""
    anchors: dict[int, tuple[float, float]] = {}
    for line in run_log.read_text().splitlines():
        m = _ANCHOR_RE.search(line)
        if not m:
            continue
        it = int(m.group(1))
        if it == 0 and anchors:
            anchors = {}  # a new run segment starts: keep only the latest
        anchors[it] = (float(m.group(2)), float(m.group(3)))
    return anchors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoints", nargs="*", default=None,
                        metavar="NAME=PATH", help="checkpoints to compare")
    parser.add_argument("--iterations", nargs="*", type=int, default=None)
    parser.add_argument("--out", default=None, help="CSV path (default <optimization>/anchor_eval.csv)")
    args = parser.parse_args()

    experiment_path = Path(args.config).resolve().parent
    specs = ExperimentSpecifications(args.config)
    rec_cfg, opt_cfg = specs["reconstruction"], specs["optimization"]
    if opt_cfg.get("mesh_pipeline") != "sdf_hex" or "surrogate" not in opt_cfg:
        raise SystemExit("needs an sdf_hex config with an optimization.surrogate block")

    paths = config.make_experiment_paths(
        experiment_path, results_name=specs.get("results_name", "results")
    )
    configure_logging(False, paths.optimization / "evaluate_anchor_shapes.log")

    anchors = anchors_from_log(paths.optimization / "run.log")
    iterations = args.iterations if args.iterations is not None else sorted(anchors)
    if not iterations:
        raise SystemExit("no anchor iterations found (pass --iterations)")
    series = paths.optimization / "parameters_series"

    def snapshot(it: int) -> Path | None:
        """Design of iteration ``it`` (None = the reconstruction)."""
        return series / f"param_{it - 1:04d}.pt" if it > 0 else None

    missing = [it for it in iterations if it > 0 and not snapshot(it).exists()]
    if missing:
        raise SystemExit(f"missing parameter snapshots for iterations {missing} in {series}")

    ckpt_specs = args.checkpoints or [f"config={opt_cfg['surrogate']['checkpoint']}"]
    surrogates = {}
    for item in ckpt_specs:
        name, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--checkpoints entries must be NAME=PATH, got {item!r}")
        surrogates[name] = TransolverSurrogate.from_config(
            {**opt_cfg["surrogate"], "checkpoint": path}
        )

    model_setup = setup_model_and_domain(rec_cfg, paths.reconstruction)
    lattice = build_lattice(rec_cfg, model_setup.model, model_setup.sdf, model_setup.frame)
    run_reconstruction(
        lattice.lattice_struct, model_setup.frame, model_setup.mesh_orig,
        rec_cfg, paths.reconstruction, model_setup.model, opt_cfg,
    )
    hex_pipeline = SdfHexMeshPipeline(
        lattice.lattice_struct, model_setup, opt_cfg, paths.optimization
    )

    recon_param = next(lattice.lattice_struct.parametrization.parameters()).detach().clone()
    rows = []
    for it in iterations:
        snap = snapshot(it)
        if snap is None:
            params = recon_param
        else:
            params = torch.load(snap, map_location=rec_cfg["device"])[0].detach()
        lattice.lattice_struct.parametrization.set_param(params.to(torch.float32))
        result = hex_pipeline.build()
        J_foam = anchors.get(it, (float("nan"), float("nan")))[0]
        row = {"iteration": it, "J_foam": J_foam, "n_wall": int(result.surface_points.shape[0])}
        for name, sur in surrogates.items():
            with torch.no_grad():
                J, diag = sur.objective(
                    result.surface_points, result.wall_tris_local,
                    hex_pipeline.sdf_at_phys, latent_fn=hex_pipeline.latent_at_phys,
                )
            J = float(J)
            row[f"J_{name}"] = J
            row[f"Jp_{name}"] = float(diag["J_p"])
            row[f"Jv_{name}"] = float(diag["J_visc"])
            row[f"drift_{name}"] = J / J_foam if J_foam == J_foam else float("nan")
        rows.append(row)
        LOGGER.info("iteration %d: %s", it, row)

    names = list(surrogates)
    header = f"{'iter':>5} {'J_foam':>9}" + "".join(f" {n + ' J':>11} {n + ' drift':>11}" for n in names)
    print("\n" + header)
    for row in rows:
        line = f"{row['iteration']:>5} {row['J_foam']:>9.4f}"
        for n in names:
            line += f" {row[f'J_{n}']:>11.4f} {row[f'drift_{n}']:>11.4f}"
        print(line)
    if anchors:
        print("\nrun-log reference (scale = J_foam / J_surr of the run's own surrogate):")
        for it in iterations:
            if it in anchors:
                jf, js = anchors[it]
                print(f"  iter {it:>3}: J_foam {jf:.4f}  J_surr {js:.4f}  drift {js / jf:.4f}")

    out = Path(args.out) if args.out else paths.optimization / "anchor_eval.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"written: {out}")


if __name__ == "__main__":
    main()

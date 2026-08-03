"""Cost of the initialization stage relative to the online loop (reviewer comment 11).

The reviewer asks what the initialization stage -- reconstructing the initial
CAD geometry in latent space -- costs relative to the online optimization loop,
since it is the one step a practitioner cannot avoid.

The manuscript runs cannot answer this from their logs: in
scripts/optimize_drag_latent.py the OptimizationLogger is constructed at line
137 and ``start_time`` is set at line 144, both *after* ``run_reconstruction``
at lines 82-92, and configure_logging installs a bare "%(message)s" formatter
(runtime.py:32) so run.log carries no timestamps either. The initialization
stage was therefore never inside any timer.

This script re-measures it by replicating exactly what optimize_drag_latent.py
does before its clock starts (lines 79-107):

    setup_model_and_domain  -> load decoder, normalize design domain
    build_lattice           -> B-spline parametrization + LatticeSDFStruct
    run_reconstruction      -> fit latent control vectors to the initial CAD
    generate_mesh(volume)   -> initial volume mesh, volume + centroid

No CFD is involved, so this is cheap and does not touch the optimization runs.
The online-loop cost is read from the archived optimization_history.csv of the
corresponding manuscript run (its ``elapsed_s`` column), so the ratio compares
a fresh measurement against the recorded loop time on the same hardware class --
state that caveat when quoting it.

Usage:
    uv run python scripts/revision/time_initialization.py
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch

import deepshapeopt.config as config
from deepshapeopt.config import ExperimentSpecifications
from deepshapeopt.mesh import compute_tet_mesh_volume_centroid
from deepshapeopt.shape_optimization import (
    build_lattice,
    generate_mesh,
    run_reconstruction,
    setup_model_and_domain,
)

REPO = Path(__file__).resolve().parents[2]
ARCHIVE = Path(
    "/storage/lfrei/Archive/DeepShapeOpt-old-private-backup/experiments/optimization"
    "/drag_optimization_cube"
)

# (config, archived run whose loop time is the comparison)
CASES = [
    ("experiments/drag_cube/config_latent_cube.json", "results_cube"),
    ("experiments/drag_cube/config_latent_cube_with_cylinders.json",
     "results_cube_with_cylinders"),
    ("experiments/drag_cube/config_latent_cube_with_holes.json", "results_cube_with_holes"),
]


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# Warm-up note: the FIRST volume-meshing call in a process pays a one-off
# CUDA/FlexiCubes kernel-compilation cost of order 20 s, while subsequent calls
# at identical mesh_resolution take well under a second. That is a property of
# the framework, not a cost of the initialization stage, so the meshing step is
# warmed up (untimed) before it is measured, and the warm-up cost is reported
# separately.


def loop_cost(run: str) -> dict | None:
    """Online-loop wall clock from the archived history."""
    hist = ARCHIVE / run / "optimization" / "optimization_history.csv"
    if not hist.is_file():
        return None
    with hist.open() as fh:
        rows = list(csv.DictReader(fh))
    times = [float(r["iter_time_s"]) for r in rows if r.get("iter_time_s")]
    total = float(rows[-1]["elapsed_s"]) if rows[-1].get("elapsed_s") else None
    return {
        "run": run,
        "n_updates": len(rows) - 1,
        "total_elapsed_s": total,
        "mean_iter_s": sum(times) / len(times) if times else None,
    }


def time_initialization(cfg_path: Path, results_name: str) -> dict:
    specs = ExperimentSpecifications(str(cfg_path))
    rec_cfg, opt_cfg = specs["reconstruction"], specs["optimization"]
    experiment_path = cfg_path.resolve().parent

    paths = config.make_experiment_paths(
        experiment_path,
        results_name=results_name,
        heavy_data_output_path=opt_cfg.get("heavy_data_output_path"),
    )
    config.ensure_experiment_dirs(paths)

    # Force a fresh fit: reuse_parameter would load cached latent vectors and
    # measure nothing (shape_optimization.py:154).
    if rec_cfg.get("reuse_parameter"):
        rec_cfg["reuse_parameter"] = False

    timings: dict[str, float] = {}

    t = time.perf_counter(); _sync()
    model_setup = setup_model_and_domain(rec_cfg, paths.reconstruction)
    _sync(); timings["setup_model_and_domain_s"] = time.perf_counter() - t

    t = time.perf_counter()
    lattice = build_lattice(rec_cfg, model_setup.model, model_setup.sdf, model_setup.box_norm)
    _sync(); timings["build_lattice_s"] = time.perf_counter() - t

    t = time.perf_counter()
    run_reconstruction(
        lattice.lattice_struct, model_setup.mesh_norm, model_setup.box_norm,
        rec_cfg, paths.reconstruction, model_setup.model, model_setup.scaling,
        opt_cfg, debug=False,
    )
    _sync(); timings["run_reconstruction_s"] = time.perf_counter() - t

    def _volume_mesh():
        with torch.no_grad():
            m, _ = generate_mesh(
                lattice.lattice_struct, opt_cfg, rec_cfg, model_setup.box_norm,
                model_setup.scaling, mesh_type="volume", extend_bounds=True,
            )
            return m, compute_tet_mesh_volume_centroid(m.vertices, m.volumes)[0]

    t = time.perf_counter(); _sync()
    _volume_mesh()  # untimed: absorbs one-off kernel compilation, see note above
    _sync(); warmup_s = time.perf_counter() - t

    t = time.perf_counter()
    mesh_init, init_volume = _volume_mesh()
    _sync(); timings["initial_volume_mesh_s"] = time.perf_counter() - t
    timings_warmup = warmup_s

    timings["initialization_total_s"] = sum(timings.values())
    return {
        "config": str(cfg_path.relative_to(REPO)),
        "device": rec_cfg.get("device"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "first_meshing_call_s": timings_warmup,
        "reconstruction_iterations": rec_cfg.get("num_iterations"),
        "batch_size": rec_cfg.get("batch_size"),
        "init_volume": float(init_volume.item()),
        "timings": timings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-name", default="revision_timing",
                        help="output dir name; kept separate from the manuscript results")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    out = []
    for rel, run in CASES:
        cfg_path = REPO / rel
        if not cfg_path.is_file():
            print(f"### {rel}  MISSING")
            continue
        print(f"\n### {rel}")
        r = time_initialization(cfg_path, args.results_name)
        r["online_loop"] = loop_cost(run)
        out.append(r)

        t = r["timings"]
        print(f"  device {r['device']} ({r['gpu']})")
        for k in ("setup_model_and_domain_s", "build_lattice_s", "run_reconstruction_s",
                  "initial_volume_mesh_s"):
            print(f"    {k:<28} {t[k]:8.2f} s")
        print(f"    {'initialization TOTAL':<28} {t['initialization_total_s']:8.2f} s")
        print(f"    ({'first meshing call':<26} {r['first_meshing_call_s']:8.2f} s, untimed "
              f"warm-up -- excluded)")

        loop = r["online_loop"]
        if loop and loop["total_elapsed_s"]:
            frac = t["initialization_total_s"] / loop["total_elapsed_s"] * 100
            print(f"  online loop ({loop['run']}): {loop['n_updates']} updates, "
                  f"{loop['total_elapsed_s']:.0f} s total, "
                  f"{loop['mean_iter_s']:.1f} s/iteration")
            print(f"  => initialization is {frac:.1f}% of the online loop, "
                  f"i.e. {t['initialization_total_s'] / loop['mean_iter_s']:.2f} "
                  f"equivalent optimization iterations")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()

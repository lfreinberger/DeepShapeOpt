"""Harvest surrogate-optimization trajectories as FOAM training samples.

Takes the per-iteration parameter snapshots (``parameters_series/``) of a
finished surrogate-driven optimization run, re-solves every visited shape
with OpenFOAM and writes ``traj_<tag>_NNNN.npz`` samples into the dataset
directory. These carry exactly the shape distribution the optimizer explores
(including its out-of-distribution excursions), which generic jitter cannot
imitate. The training split treats ``traj_*`` files as train-only.

Usage:
    uv run python scripts/generate_trajectory_dataset.py \
        --run-config experiments/drag_cube/config_traj_sphere.json \
        --dataset-config experiments/flow_dataset/config_dataset.json \
        [--stride 1]
"""

from __future__ import annotations

import argparse
import logging
import shutil
import time
import traceback
from pathlib import Path

import torch

import deepshapeopt.config as config
import deepshapeopt.foam_utils as foam_utils
from deepshapeopt.config import ExperimentSpecifications
from deepshapeopt.hexmesh import SdfHexMeshPipeline
from deepshapeopt.shape_optimization import (
    build_lattice,
    run_reconstruction,
    setup_model_and_domain,
)
from generate_flow_dataset import generate_sample

logger = logging.getLogger("generate_trajectory_dataset")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", required=True, help="config of the finished optimization run")
    parser.add_argument("--dataset-config", required=True, help="flow-dataset config (foam template, output dir)")
    parser.add_argument("--stride", type=int, default=1, help="take every n-th iteration")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("deepshapeopt").setLevel(logging.WARNING)

    run_specs = ExperimentSpecifications(args.run_config)
    ds_specs = ExperimentSpecifications(args.dataset_config)
    ds_cfg = dict(ds_specs["dataset"])
    rec_cfg = dict(run_specs["reconstruction"])
    opt_cfg = dict(run_specs["optimization"])

    run_dir = Path(args.run_config).resolve().parent
    tag = run_specs.get("results_name", "run").removeprefix("results_")
    paths = config.make_experiment_paths(run_dir, results_name=run_specs["results_name"])
    series_dir = paths.optimization / "parameters_series"
    param_files = sorted(series_dir.glob("param_*.pt"))[:: args.stride]
    if not param_files:
        raise SystemExit(f"no parameter snapshots in {series_dir}")

    out_dir = Path(ds_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Same lattice/reconstruction setup as the run (loads its rec_parameters.pt).
    model_setup = setup_model_and_domain(rec_cfg, paths.reconstruction)
    lattice = build_lattice(rec_cfg, model_setup.model, model_setup.sdf, model_setup.frame)
    run_reconstruction(
        lattice.lattice_struct, model_setup.frame, model_setup.mesh_orig,
        rec_cfg, paths.reconstruction, model_setup.model, opt_cfg,
    )
    hex_pipeline = SdfHexMeshPipeline(
        lattice.lattice_struct, model_setup, opt_cfg, paths.optimization
    )

    ds_dir = Path(args.dataset_config).resolve().parent
    runtime_root = opt_cfg.get("foam_runtime_root") or ds_specs["optimization"].get("foam_runtime_root")
    case_dir = foam_utils.prepare_foam_runtime(
        ds_dir / "foam_case", run_name=f"traj_{tag}",
        runtime_root=Path(runtime_root) if runtime_root else None,
    )
    foam_utils.select_allrun(case_dir, "sdf_hex")
    logger.info("Harvesting %d shapes from %s", len(param_files), series_dir)

    n_done = n_failed = 0
    try:
        for pf in param_files:
            it = int(pf.stem.split("_")[1])
            npz_path = out_dir / f"traj_{tag}_{it:04d}.npz"
            if npz_path.exists():
                continue
            t0 = time.time()
            params = torch.load(pf, map_location=rec_cfg["device"])
            try:
                info = generate_sample(
                    seed=0,
                    variant=0,  # use the snapshot as-is, no jitter
                    base_param=params[0].to(dtype=torch.float32),
                    lattice_struct=lattice.lattice_struct,
                    hex_pipeline=hex_pipeline,
                    case_dir=case_dir,
                    ds_cfg=ds_cfg,
                    npz_path=npz_path,
                    meta={
                        "family": f"traj_{tag}",
                        "seed": -1,
                        "variant": 0,
                        "geometry": {"trajectory_iteration": it, "run": str(run_dir)},
                        "nu": ds_cfg["nu"],
                    },
                )
                n_done += 1
                logger.info(
                    "traj_%s_%04d  J=%9.4f pts=%6d (surf %5d) [%.0f s]",
                    tag, it, info["J"], info["n_points"], info["n_surface"], time.time() - t0,
                )
            except Exception:
                n_failed += 1
                logger.error("traj_%s_%04d FAILED:\n%s", tag, it, traceback.format_exc(limit=6))
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)

    logger.info("Done: %d harvested, %d failed.", n_done, n_failed)
    if n_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

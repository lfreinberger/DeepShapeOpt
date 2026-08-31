"""One-shot OpenFOAM verification of an optimized shape (``parameters.pt``).

Loads the design parameters, rebuilds the sdf_hex mesh, runs one full
adjointOptimisationFoam case and reports the FOAM drag next to the Transolver
surrogate prediction (when a surrogate block is configured). Use after a
surrogate-driven optimization to confirm the drag reduction with real CFD.

Usage:
    uv run python scripts/evaluate_shape_foam.py \
        --config experiments/drag_cube/config_latent_cube_transolver.json \
        [--parameters <path/to/parameters.pt>]
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import torch

import deepshapeopt.config as config
import deepshapeopt.foam_utils as foam_utils
from deepshapeopt.config import ExperimentSpecifications
from deepshapeopt.hexmesh import SdfHexMeshPipeline
from deepshapeopt.runtime import configure_logging
from deepshapeopt.shape_optimization import (
    build_lattice,
    run_reconstruction,
    setup_model_and_domain,
)

LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--parameters",
        default=None,
        help="parameters.pt to evaluate (default: <results>/optimization/parameters.pt)",
    )
    args = parser.parse_args()

    experiment_path = Path(args.config).resolve().parent
    specs = ExperimentSpecifications(args.config)
    rec_cfg, opt_cfg = specs["reconstruction"], specs["optimization"]
    sens_cfg = opt_cfg.get("sensitivity", {})
    if opt_cfg.get("mesh_pipeline") != "sdf_hex":
        raise SystemExit("evaluate_shape_foam requires mesh_pipeline 'sdf_hex'")

    results_name = specs.get("results_name", "results")
    paths = config.make_experiment_paths(experiment_path, results_name=results_name)
    config.ensure_experiment_dirs(paths)
    configure_logging(False, paths.optimization / "evaluate_shape_foam.log")

    model_setup = setup_model_and_domain(rec_cfg, paths.reconstruction)
    lattice = build_lattice(rec_cfg, model_setup.model, model_setup.sdf, model_setup.frame)
    run_reconstruction(
        lattice.lattice_struct, model_setup.frame, model_setup.mesh_orig,
        rec_cfg, paths.reconstruction, model_setup.model, opt_cfg,
    )

    param_path = Path(args.parameters) if args.parameters else paths.optimization / "parameters.pt"
    if param_path.exists():
        params = torch.load(param_path, map_location=rec_cfg["device"])
        lattice.lattice_struct.parametrization.set_param(
            params[0].to(dtype=torch.float32)
        )
        LOGGER.info("Loaded design parameters from %s", param_path)
    else:
        LOGGER.warning("%s not found; evaluating the reconstruction itself", param_path)

    hex_pipeline = SdfHexMeshPipeline(
        lattice.lattice_struct, model_setup, opt_cfg, paths.optimization
    )
    hex_result = hex_pipeline.build()

    surrogate_J = None
    if "surrogate" in opt_cfg:
        from deepshapeopt.surrogate.predictor import TransolverSurrogate

        surrogate = TransolverSurrogate.from_config(opt_cfg["surrogate"])
        with torch.no_grad():
            J_t, diag = surrogate.objective(
                hex_result.surface_points, hex_result.wall_tris_local,
                hex_pipeline.sdf_at_phys,
            )
        surrogate_J = float(J_t)
        LOGGER.info(
            "Surrogate: J=%.6f (pressure %.6f, viscous %.6f)",
            surrogate_J, diag["J_p"], diag["J_visc"],
        )

    case_dir = foam_utils.prepare_foam_runtime(
        experiment_path / "foam_case", run_name=f"{results_name}_verify",
        runtime_root=Path(opt_cfg["foam_runtime_root"]) if opt_cfg.get("foam_runtime_root") else None,
    )
    foam_utils.select_allrun(case_dir, "sdf_hex")
    try:
        foam_case = hex_pipeline.run_case(case_dir, verbose=False)
        _, J_foam = hex_pipeline.load_sensitivities(
            case_dir, foam_case,
            objective_path=sens_cfg.get("objective_path", "optimisation/objective/0/dragadjS1"),
        )
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)

    print(f"OpenFOAM drag J = {float(J_foam):.6f}")
    if surrogate_J is not None:
        rel = abs(surrogate_J - float(J_foam)) / max(abs(float(J_foam)), 1e-12)
        print(f"Surrogate drag J = {surrogate_J:.6f}  (rel. error {100 * rel:.2f}%)")


if __name__ == "__main__":
    main()

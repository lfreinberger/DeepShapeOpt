"""MLflow parameter study for reconstruction regularization hyperparameters.

Sweeps over code_reg_lambda, code_bound, grad_clip, and eikonal_lambda.
Each combination is tracked as a nested MLflow run with error metrics and
reconstructed STL logged as artifacts for visual comparison.
"""
import itertools
import json
import traceback
from pathlib import Path

import mlflow

from deepshapeopt.config import ExperimentSpecifications
from deepshapeopt.mlflow_runner import set_nested, value_to_name
from deepshapeopt.reconstruction import reconstruct_shape

# ---------------------------------------------------------------------------
# Parameter grid — edit these to change the sweep
# ---------------------------------------------------------------------------
PARAM_GRID = {
    "reconstruction.code_reg_lambda": [0, 1e-4],
    "reconstruction.eikonal_lambda": [0, 0.005, 0.05],
    "reconstruction.code_bound": [None, 1.0],
    "reconstruction.grad_clip": [None, 1.0],
}

TRACKING_URI = "mlruns"
EXPERIMENT_NAME = "feed_channel_regularization_study"
# ---------------------------------------------------------------------------


def _short_name(key: str, value) -> str:
    tag = key.rsplit(".", 1)[-1]
    if isinstance(value, float) and value == 0:
        return f"{tag}=0"
    return f"{tag}={value_to_name(value)}"


def main():
    experiment_path = Path(__file__).resolve().parent
    base_specs = ExperimentSpecifications(str(experiment_path / "config.json"))

    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combos = list(itertools.product(*values))

    print(f"Parameter study: {len(combos)} runs")

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    results_summary = []

    with mlflow.start_run(run_name="regularization_sweep"):
        mlflow.log_param("num_runs", len(combos))
        mlflow.log_param("parameter_grid", json.dumps(PARAM_GRID, default=str))

        for i, combo in enumerate(combos, 1):
            name_parts = [_short_name(k, v) for k, v in zip(keys, combo)]
            run_name = "__".join(name_parts)
            results_name = "study__" + run_name

            print(f"\n{'='*70}")
            print(f"Run {i}/{len(combos)}: {run_name}")
            print(f"{'='*70}")

            specs = base_specs.copy()
            for key, value in zip(keys, combo):
                set_nested(specs, key, value)
            specs["results_name"] = results_name

            with mlflow.start_run(run_name=run_name, nested=True):
                mlflow.log_params({
                    key.replace(".", "__"): json.dumps(value)
                    for key, value in zip(keys, combo)
                })

                try:
                    result = reconstruct_shape(
                        experiment_path,
                        specs,
                        use_mlflow=True,
                        save_vtp=True,
                    )

                    mlflow.log_metric("final_loss", result["final_loss"])
                    if result["metrics"] is not None:
                        for key in ("mae", "rmse", "median", "p95"):
                            mlflow.log_metric(key, result["metrics"][key])

                    # Log reconstructed STL for visual comparison
                    mlflow.log_artifact(result["reconstructed_mesh_path"])

                    results_summary.append((run_name, "OK", result))
                    loss = result["final_loss"]
                    mae = result["metrics"]["mae"] if result["metrics"] else None
                    print(f"Run {i} finished: loss={loss:.6e}, MAE={mae:.8f}")

                except Exception:
                    traceback.print_exc()
                    mlflow.set_tag("status", "FAILED")
                    results_summary.append((run_name, "FAILED", None))
                    print(f"Run {i} FAILED — continuing with next run")

    # Print summary
    print(f"\n{'='*70}")
    print("PARAMETER STUDY SUMMARY")
    print(f"{'='*70}")
    for name, status, result in results_summary:
        if status == "OK" and result["metrics"]:
            m = result["metrics"]
            print(
                f"  OK     {name}  "
                f"loss={result['final_loss']:.6e}  "
                f"MAE={m['mae']:.8f}  RMSE={m['rmse']:.8f}  P95={m['p95']:.8f}"
            )
        elif status == "OK":
            print(f"  OK     {name}  loss={result['final_loss']:.6e}")
        else:
            print(f"  FAILED {name}")

    n_ok = sum(1 for _, s, _ in results_summary if s == "OK")
    print(f"\n{n_ok}/{len(results_summary)} runs completed successfully")


if __name__ == "__main__":
    main()

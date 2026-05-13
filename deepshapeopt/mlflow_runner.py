"""Generic MLflow parameter study runner for shape optimization experiments."""
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable

import mlflow

from deepshapeopt.config import ExperimentSpecifications


def set_nested(config_obj, dotted_key: str, value: Any) -> None:
    """Set a nested config entry using dot notation."""
    keys = dotted_key.split(".")
    current = config_obj
    for key in keys[:-1]:
        current = current[key]
    current[keys[-1]] = value


def get_nested(config_obj, dotted_key: str) -> Any:
    """Read a nested config entry using dot notation."""
    keys = dotted_key.split(".")
    current = config_obj
    for key in keys:
        current = current[key]
    return current


def value_to_name(value: Any) -> str:
    """Convert parameter value to a compact string for run names."""
    if isinstance(value, (list, tuple)):
        return "x".join(map(str, value))
    if isinstance(value, Path):
        return value.name
    if isinstance(value, str):
        p = Path(value)
        return p.name if "/" in value else value
    return str(value)


def run_parameter_study(
    experiment_path: Path,
    optimize_fn: Callable,
    tracking_uri: str,
    experiment_name: str,
    parameter_path: str,
    parameter_values: list,
    config_file: str = "config.json",
    storage_base: str | None = None,
):
    """Run a single-parameter sweep with MLflow tracking."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("mlflow_parameter_study")
    if storage_base is None:
        storage_base = os.environ.get("DEEPSHAPEOPT_RESULTS_DIR", "results/mlflow")

    base_specs = ExperimentSpecifications(str(experiment_path / config_file))

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="parameter_study"):
        mlflow.log_param("parameter_path", parameter_path)
        mlflow.log_param("num_values", len(parameter_values))
        mlflow.log_param("parameter_values", json.dumps(parameter_values))
        mlflow.set_tag("study_type", "single_parameter_sweep")
        mlflow.set_tag("project", "DeepShapeOpt")

        for value in parameter_values:
            specs = base_specs.copy()
            set_nested(specs, parameter_path, value)

            param_name = parameter_path.replace(".", "__")
            run_name = f"{param_name}__{value_to_name(value)}"

            specs["optimization"]["heavy_data_output_path"] = str(
                Path(storage_base) / run_name
            )

            with mlflow.start_run(run_name=run_name, nested=True):
                mlflow.log_params(specs.flatten())
                mlflow.log_param("swept_parameter", parameter_path)
                mlflow.log_param("swept_value", json.dumps(value))
                mlflow.log_dict(dict(specs), "config/effective_config.json")

                logger.info("Starting run: %s", run_name)

                results = optimize_fn(
                    experiment_path=experiment_path,
                    specs=specs,
                )

                for key, result_value in results.items():
                    if isinstance(result_value, (int, float)):
                        mlflow.log_metric(key, float(result_value))

                mlflow.set_tag("status", "finished")
                logger.info("Finished run: %s", run_name)

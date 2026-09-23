"""Experiment configuration: typed schema, loader and result paths."""

from .loader import expand_env_vars, load_config, load_config_dict  # noqa: F401
from .paths import RunPaths, make_run_paths
from .schema import Config, ConfigError

__all__ = [
    "Config",
    "ConfigError",
    "RunPaths",
    "expand_env_vars",
    "load_config",
    "load_config_dict",
    "make_run_paths",
]

# Legacy (v1) API, kept importable until the old drivers are removed.
from .legacy import (  # noqa: E402
    ExperimentSpecifications,
    ExperimentPaths,
    ensure_experiment_dirs,
    make_experiment_paths,
    make_setup_name,
)

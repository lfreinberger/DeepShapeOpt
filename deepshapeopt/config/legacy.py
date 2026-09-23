import copy
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union


_UNRESOLVED_ENV_RE = re.compile(r"\$\{[^}]+\}|\$[A-Za-z_][A-Za-z0-9_]*")


class ExperimentSpecifications(dict):
    def __init__(self, specs: Union[Dict[str, Any], str, None] = None):
        if isinstance(specs, str):
            loaded_specs = self._load_from_file(specs)
            super().__init__(expand_env_vars(copy.deepcopy(loaded_specs)))
        else:
            super().__init__(expand_env_vars(copy.deepcopy(specs)) if specs else {})

    @staticmethod
    def _load_from_file(filename: str) -> Dict[str, Any]:
        with open(filename, "r") as f:
            if filename.endswith(".json"):
                return json.load(f)
            raise ValueError("Unsupported file format. Use .json")

    def save(self, filename: str) -> None:
        with open(filename, "w") as f:
            json.dump(self, f, indent=4)

    def update(self, updates: Dict[str, Any]) -> None:
        def _recursive_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and isinstance(d.get(k), dict):
                    _recursive_update(d[k], v)
                else:
                    d[k] = v

        _recursive_update(self, updates)

    def copy(self):
        return ExperimentSpecifications(copy.deepcopy(self))

    def flatten(self, parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
        items = {}
        for k, v in self.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(ExperimentSpecifications(v).flatten(new_key, sep))
            elif isinstance(v, list):
                if all(isinstance(i, dict) for i in v):
                    for idx, elem in enumerate(v):
                        items.update(
                            ExperimentSpecifications(elem).flatten(
                                f"{new_key}.{idx}", sep
                            )
                        )
                else:
                    items[new_key] = str(v)
            else:
                items[new_key] = v
        return items


@dataclass(frozen=True)
class ExperimentPaths:
    base: Path
    results: Path
    reconstruction: Path
    optimization: Path
    heavy_data: Union[Path, None] = None

def make_experiment_paths(
    base_dir: Union[str, Path],
    results_name: str = "results",
    heavy_data_output_path: Union[str, Path, None] = None,
    run_subdir: Union[str, Path, None] = None,
) -> ExperimentPaths:
    base = Path(base_dir).resolve()
    results = base / results_name
    if run_subdir is not None:
        results = results / Path(run_subdir)
    heavy_data = None
    heavy_data_output_path = expand_env_vars(heavy_data_output_path)
    if heavy_data_output_path is not None:
        project_root = base
        while project_root != project_root.parent:
            if (project_root / "pyproject.toml").exists():
                break
            project_root = project_root.parent
        try:
            rel = base.relative_to(project_root)
        except ValueError:
            rel = Path(base.name)
        heavy_data = Path(heavy_data_output_path) / rel / results_name
        if run_subdir is not None:
            heavy_data = heavy_data / Path(run_subdir)
    return ExperimentPaths(
        base=base,
        results=results,
        reconstruction=results / "reconstruction",
        optimization=results / "optimization",
        heavy_data=heavy_data,
    )

def make_setup_name(objective_name, constraint_enabled=False, constraint_name=None, constraint_cfg=None):
    """Encode the objective/constraint setup into a run_subdir leaf name.

    Two runs that differ only by objective/constraint then live side-by-side under the same
    results dir without overwriting (e.g. ``obj_uniformityas1__con_lossesas2__rel_1.0``).
    Feed the result to :func:`make_experiment_paths` as ``run_subdir``.
    """
    if not constraint_enabled:
        return f"obj_{objective_name}__no_constraint"
    target_mode = constraint_cfg.get("target_mode")
    if target_mode == "relative_to_initial":
        target_txt = f"rel_{constraint_cfg.get('target_factor')}"
    elif target_mode == "absolute":
        target_txt = f"abs_{constraint_cfg.get('target_value')}"
    else:
        target_txt = "target_unknown"
    return f"obj_{objective_name}__con_{constraint_name}__{target_txt}"


def ensure_experiment_dirs(paths: ExperimentPaths) -> None:
    for p in (paths.results, paths.reconstruction, paths.optimization):
        p.mkdir(parents=True, exist_ok=True)
    if paths.heavy_data is not None:
        paths.heavy_data.mkdir(parents=True, exist_ok=True)


def expand_env_vars(value: Any) -> Any:
    """Recursively expand environment variables in experiment specs.

    This keeps paper configs portable. For example, a JSON value such as
    ``"${DEEPSHAPEOPT_MODEL_DIR}/primitives_cl32"`` resolves at load time.
    """
    if isinstance(value, str):
        expanded = os.path.expanduser(os.path.expandvars(value))
        if _UNRESOLVED_ENV_RE.search(expanded):
            raise ValueError(
                f"Unresolved environment variable in path or config value: {value!r}. "
                "Check the environment variables used by the experiment config."
            )
        return expanded
    if isinstance(value, list):
        return [expand_env_vars(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_env_vars(item) for key, item in value.items()}
    return value

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, Union
import copy
import json


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
) -> ExperimentPaths:
    base = Path(base_dir).resolve()
    results = base / results_name
    heavy_data = None
    heavy_data_output_path = expand_env_vars(heavy_data_output_path)
    if heavy_data_output_path is not None:
        # Mirror the local directory structure (experiments/reconstruction/...)
        # under the heavy data root by computing the path relative to the
        # project root (identified by pyproject.toml).
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
    return ExperimentPaths(
        base=base,
        results=results,
        reconstruction=results / "reconstruction",
        optimization=results / "optimization",
        heavy_data=heavy_data,
    )

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
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [expand_env_vars(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_env_vars(item) for key, item in value.items()}
    return value

"""Primitive shape reconstruction experiment."""
from pathlib import Path

from deepshapeopt.config import ExperimentSpecifications
from deepshapeopt.reconstruction import reconstruct_shape

if __name__ == "__main__":
    experiment_path = Path(__file__).resolve().parent
    specs = ExperimentSpecifications(str(experiment_path / "config.json"))
    reconstruct_shape(experiment_path, specs)

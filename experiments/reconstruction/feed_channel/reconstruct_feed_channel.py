"""Flow-channel reconstruction experiment."""
import argparse
from pathlib import Path

from deepshapeopt.config import ExperimentSpecifications
from deepshapeopt.reconstruction import reconstruct_shape

if __name__ == "__main__":
    experiment_path = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(experiment_path / "config.json"),
        help="Path to an experiment JSON config.",
    )
    args = parser.parse_args()
    specs = ExperimentSpecifications(args.config)
    reconstruct_shape(experiment_path, specs)

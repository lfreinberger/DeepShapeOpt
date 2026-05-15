"""Standalone DeepSDF shape reconstruction."""

from __future__ import annotations

import argparse
from pathlib import Path

from deepshapeopt.config import ExperimentSpecifications
from deepshapeopt.reconstruction import reconstruct_shape


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to an experiment JSON config.",
    )
    args = parser.parse_args()

    experiment_path = Path(args.config).resolve().parent
    specs = ExperimentSpecifications(args.config)
    reconstruct_shape(experiment_path, specs)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train a DeepSDFStruct latent-field decoder for DeepShapeOpt.

The DeepSDFStruct training routine expects an experiment directory containing
`specs.json`. This wrapper keeps DeepShapeOpt configs portable: it reads a JSON
config, expands environment variables, writes `specs.json` into the requested
output directory, and then calls
`DeepSDFStruct.deep_sdf.training_latent_field.train`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

def expand_env_vars(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [expand_env_vars(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_env_vars(item) for key, item in value.items()}
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a DeepSDFStruct latent-field decoder from a DeepShapeOpt config."
    )
    parser.add_argument(
        "--config",
        default="experiments/training/primitives_cl32/specs.json",
        help="Training specs JSON. Defaults to the primitives_cl32 experiment specs.",
    )
    parser.add_argument(
        "--output",
        default="${DEEPSHAPEOPT_MODEL_DIR}/primitives_cl32",
        help="Model output directory. A specs.json file is written here.",
    )
    parser.add_argument(
        "--data-source",
        default=None,
        help="Override the DataSource entry from the config.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override training device, e.g. cpu or cuda. If omitted, DeepSDFStruct chooses.",
    )
    parser.add_argument(
        "--continue-from",
        default=None,
        help="Optional checkpoint name passed to DeepSDFStruct training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from DeepSDFStruct.deep_sdf.training_latent_field import train

    config_path = Path(args.config).resolve()
    output_dir = Path(os.path.expanduser(os.path.expandvars(args.output))).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with config_path.open("r", encoding="utf-8") as f:
        specs = expand_env_vars(json.load(f))

    if args.data_source is not None:
        specs["DataSource"] = os.path.expanduser(os.path.expandvars(args.data_source))

    specs_path = output_dir / "specs.json"
    with specs_path.open("w", encoding="utf-8") as f:
        json.dump(specs, f, indent=2)
        f.write("\n")

    train(
        str(output_dir),
        data_source=specs.get("DataSource"),
        continue_from=args.continue_from,
        device=args.device,
    )


if __name__ == "__main__":
    main()

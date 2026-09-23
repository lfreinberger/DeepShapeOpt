"""Run one shape optimization: ``python scripts/optimize.py --config <experiment>/config.json``."""

import sys

from deepshapeopt.cli import main

if __name__ == "__main__":
    sys.exit(main(["optimize", *sys.argv[1:]]))

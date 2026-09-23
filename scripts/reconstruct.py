"""Standalone reconstruction: ``python scripts/reconstruct.py --config <experiment>/config.json``."""

import sys

from deepshapeopt.cli import main

if __name__ == "__main__":
    sys.exit(main(["reconstruct", *sys.argv[1:]]))

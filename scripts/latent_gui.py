"""Latent-code editor: ``python scripts/latent_gui.py --config <experiment>/config.json``."""

import sys

from deepshapeopt.cli import main

if __name__ == "__main__":
    sys.exit(main(["latent-gui", *sys.argv[1:]]))

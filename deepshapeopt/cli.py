"""Command line: ``deepshapeopt optimize|reconstruct|latent-gui|migrate-config``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _config_parser(prog: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=prog)
    ap.add_argument("--config", required=True, help="experiment JSON config (schema v2)")
    return ap


def cmd_optimize(argv) -> int:
    args = _config_parser("deepshapeopt optimize").parse_args(argv)
    from .config import load_config
    from .driver import run

    path = Path(args.config).resolve()
    result = run(load_config(path), path.parent)
    print(f"Results written to: {result['results_dir']}")
    print(f"Final objective: {result['final_objective']:.6e}")
    return 0


def cmd_reconstruct(argv) -> int:
    args = _config_parser("deepshapeopt reconstruct").parse_args(argv)
    from .config import load_config
    from .geometry.reconstruction import reconstruct_shape
    from .logging_setup import configure_logging

    path = Path(args.config).resolve()
    cfg = load_config(path)
    configure_logging(cfg.run.debug)
    result = reconstruct_shape(cfg, path.parent)
    print(f"Reconstruction results: {result['results_dir']}")
    return 0


def cmd_latent_gui(argv) -> int:
    from .latent_gui.server import main as gui_main

    return gui_main(argv) or 0


def cmd_migrate(argv) -> int:
    from .config.migrate import main as migrate_main

    return migrate_main(argv)


COMMANDS = {
    "optimize": cmd_optimize,
    "reconstruct": cmd_reconstruct,
    "latent-gui": cmd_latent_gui,
    "migrate-config": cmd_migrate,
}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: deepshapeopt {" + ",".join(COMMANDS) + "} [options]")
        return 0 if argv else 2
    command, rest = argv[0], argv[1:]
    if command not in COMMANDS:
        print(f"unknown command {command!r}; choose from {', '.join(COMMANDS)}", file=sys.stderr)
        return 2
    return COMMANDS[command](rest)


if __name__ == "__main__":
    sys.exit(main())

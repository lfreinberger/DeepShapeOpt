"""Launch the interactive local-latent-code editor for a saved shape.

Loads an experiment config.json (the same one used by ``scripts/reconstruct.py``
or ``scripts/optimize_drag_latent.py``), rebuilds its B-spline control net of
local latent codes, loads that run's *saved* latent parameters, and serves a
small web GUI where you click a knot and drag sliders to change its latent code
while the geometry updates live.

The GUI never fits on the fly: it always loads an existing parameter file. Point
it at the run via ``--config`` (the source and parameter file are located
automatically) or give an explicit ``--source`` / ``--params-file``; a missing
file raises an error.

By default ``--source auto`` inspects the optimization result when the config has
an ``optimization`` block and the whole-input-mesh reconstruction otherwise.

Examples
--------
    # inspect a reconstruction (rec_parameters.pt under the config's results):
    python scripts/latent_gui.py \
        --config experiments/reconstruction/feed_channel/config.json

    # inspect an optimization result (auto-detected; parameters.pt, else
    # rec_parameters.pt):
    python scripts/latent_gui.py \
        --config experiments/drag_cube/config_latent_cube.json

    # quick non-interactive sanity check (no browser/server):
    python scripts/latent_gui.py \
        --config experiments/reconstruction/feed_channel/config.json --smoke
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from deepshapeopt.latent_gui.backend import LatentEditSession
from deepshapeopt.latent_gui.server import serve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT / "experiments" / "reconstruction" / "feed_channel" / "config.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Experiment config.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default=None, help="Override compute device (e.g. cpu, cuda)")
    parser.add_argument(
        "--mesh-n",
        type=int,
        default=None,
        help="Per-unit-cell mesh resolution for live editing (grid = mesh_n * tiling). "
        "Lower = faster updates, coarser surface. Defaults to a responsive heuristic.",
    )
    parser.add_argument(
        "--source",
        choices=("auto", "reconstruction", "optimization"),
        default="auto",
        help="Which lattice/parameters to load. auto (default): optimization when the "
        "config has an 'optimization' block, else reconstruction. reconstruction: "
        "rebuild the whole-input-mesh lattice and load its rec_parameters.pt. "
        "optimization: rebuild the optimizer's design-domain lattice and load that "
        "run's saved latents (parameters.pt, else rec_parameters.pt). The GUI never "
        "fits: the file must already exist or it errors.",
    )
    parser.add_argument(
        "--params-file",
        default=None,
        help="Explicit .pt latent file to load, overriding the automatic lookup from "
        "--config. Must exist.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Load, edit one code, re-mesh, print counts and exit (no server).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    session = LatentEditSession(
        args.config,
        device=args.device,
        mesh_n=args.mesh_n,
        source=args.source,
        params_file=args.params_file,
    )

    if args.smoke:
        before = session.mesh()
        knot = session.n_knots // 2
        session.set_value(knot, 0, session.code_bound)
        after = session.mesh()
        print(
            f"OK: {session.n_knots} knots, latent_dim={session.latent_dim}, "
            f"tiling={session.tiling}\n"
            f"    mesh before edit: {before['n_vertices']} verts / {before['n_faces']} faces\n"
            f"    edited knot #{knot} dim 0 -> {session.code_bound}\n"
            f"    mesh after  edit: {after['n_vertices']} verts / {after['n_faces']} faces"
        )
        if before["n_vertices"] == 0:
            raise SystemExit("Smoke test FAILED: empty initial mesh.")
        if (before["n_vertices"], before["n_faces"]) == (
            after["n_vertices"],
            after["n_faces"],
        ) and before["vertices"] == after["vertices"]:
            print("    WARNING: mesh did not change after the edit.")
        return

    serve(session, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

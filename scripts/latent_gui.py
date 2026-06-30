"""Launch the interactive local-latent-code editor for a reconstructed shape.

Loads a reconstruction experiment (the same config.json used by
``scripts/reconstruct.py``), exposes its B-spline control net of local latent
codes in a small web GUI, and lets you click a knot and drag sliders to change
that knot's latent code while the geometry updates live.

Examples
--------
    python scripts/latent_gui.py \
        --config experiments/reconstruction/feed_channel/config.json

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
        "--no-fit",
        action="store_true",
        help="Require a saved rec_parameters.pt instead of fitting on launch.",
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
        fit_if_missing=not args.no_fit,
        mesh_n=args.mesh_n,
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

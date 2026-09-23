"""Tiny stdlib HTTP server exposing a :class:`LatentEditSession` as a JSON API
plus a static Three.js viewer. No third-party web dependencies.

Endpoints
---------
GET  /              -> static/index.html
GET  /api/state        -> full initial state (latent_dim, knots, codes, mesh, pca, ...)
POST /api/update       -> {"knot_idx", "dim", "value"}  -> {"mesh", "code"[, "coeff"]}
POST /api/update_coeff -> {"knot_idx", "comp", "value"} -> {"mesh", "code", "coeff"}
POST /api/pca_truncate -> {"k"} project onto first k PCs -> {"mesh", "codes"[, "coeffs"]}
POST /api/reset        -> restore the loaded codes      -> {"mesh", "codes"[, "coeffs"]}
"""

from __future__ import annotations

import argparse
import json
import logging
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from deepshapeopt.latent_gui.backend import LatentEditSession

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"


class _Handler(BaseHTTPRequestHandler):
    # ``session`` is injected via the subclass created in ``serve``.
    session: LatentEditSession = None  # type: ignore[assignment]

    # Quieter logging through the project logger.
    def log_message(self, fmt, *args):  # noqa: N802
        logger.debug("%s - %s", self.address_string(), fmt % args)

    # -- helpers ----------------------------------------------------------
    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str):
        try:
            body = path.read_bytes()
        except FileNotFoundError:
            self._send_json({"error": f"not found: {path.name}"}, status=404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _knot_edit_payload(self, idx: int) -> dict:
        """Mesh plus the edited knot's latent code (and PCA coefficients when enabled),
        so either editor tab can refresh its sliders after an edit made through the
        other (a coeff edit changes the raw latents and vice versa)."""
        payload = {"mesh": self.session.mesh(), "code": self.session.code(idx)}
        coeff = self.session.coeff(idx)
        if coeff is not None:
            payload["coeff"] = coeff
        return payload

    def _all_knots_payload(self) -> dict:
        """Mesh + all latent codes (and all PCA coefficients when enabled), for edits
        that move every knot at once (reset, PCA truncation)."""
        payload = {"mesh": self.session.mesh(), "codes": self.session.codes()}
        coeffs = self.session.coeffs()
        if coeffs is not None:
            payload["coeffs"] = coeffs
        return payload

    def _serve_static(self, url_path: str):
        """Serve a file from STATIC_DIR (e.g. the vendored three.js), guarding
        against path traversal outside the static root."""
        rel = unquote(url_path).lstrip("/") or "index.html"
        target = (STATIC_DIR / rel).resolve()
        if not target.is_relative_to(STATIC_DIR) or not target.is_file():
            self._send_json({"error": "not found"}, status=404)
            return
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if target.suffix in (".js", ".mjs"):
            ctype = "text/javascript"
        if target.suffix == ".html":
            ctype = "text/html; charset=utf-8"
        self._send_file(target, ctype)

    # -- routes -----------------------------------------------------------
    def do_GET(self):  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
        elif path == "/api/state":
            try:
                self._send_json(self.session.state())
            except Exception as exc:  # pragma: no cover - surfaced to client
                logger.exception("state failed")
                self._send_json({"error": str(exc)}, status=500)
        elif path.startswith("/api/"):
            self._send_json({"error": "not found"}, status=404)
        else:
            self._serve_static(path)

    def do_POST(self):  # noqa: N802
        try:
            if self.path == "/api/update":
                data = self._read_json()
                idx = int(data["knot_idx"])
                self.session.set_value(idx, int(data["dim"]), float(data["value"]))
                self._send_json(self._knot_edit_payload(idx))
            elif self.path == "/api/update_coeff":
                data = self._read_json()
                idx = int(data["knot_idx"])
                self.session.set_coeff(idx, int(data["comp"]), float(data["value"]))
                self._send_json(self._knot_edit_payload(idx))
            elif self.path == "/api/pca_truncate":
                data = self._read_json()
                self.session.truncate_to_k(int(data["k"]))
                self._send_json(self._all_knots_payload())
            elif self.path == "/api/reset":
                self.session.reset()
                self._send_json(self._all_knots_payload())
            else:
                self._send_json({"error": "not found"}, status=404)
        except (KeyError, ValueError, IndexError) as exc:
            self._send_json({"error": str(exc)}, status=400)
        except Exception as exc:  # pragma: no cover - surfaced to client
            logger.exception("POST %s failed", self.path)
            self._send_json({"error": str(exc)}, status=500)


def serve(session: LatentEditSession, host: str = "127.0.0.1", port: int = 8000):
    """Block serving ``session`` over HTTP until interrupted."""

    handler = type("BoundHandler", (_Handler,), {"session": session})
    httpd = ThreadingHTTPServer((host, port), handler)
    logger.info("Latent-edit GUI serving at http://%s:%d", host, port)
    print(f"Latent-edit GUI ready: http://{host}:{port}  (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


CLI_HELP = '''Launch the interactive local-latent-code editor for a saved shape.

Loads an experiment config.json (the same one used by ``deepshapeopt reconstruct``
or ``deepshapeopt optimize``), rebuilds its B-spline control net of
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
    deepshapeopt latent-gui \
        --config experiments/reconstruction/feed_channel/config.json

    # inspect an optimization result (auto-detected; parameters.pt, else
    # rec_parameters.pt):
    deepshapeopt latent-gui \
        --config experiments/drag_cube/config_latent_cube.json

    # quick non-interactive sanity check (no browser/server):
    deepshapeopt latent-gui \
        --config experiments/reconstruction/feed_channel/config.json --smoke
'''


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="deepshapeopt latent-gui", description=CLI_HELP)
    parser.add_argument("--config", required=True, help="Experiment config.json")
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
    args = parser.parse_args(argv)

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
        return 0

    serve(session, host=args.host, port=args.port)
    return 0



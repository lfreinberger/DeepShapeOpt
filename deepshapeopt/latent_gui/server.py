"""Tiny stdlib HTTP server exposing a :class:`LatentEditSession` as a JSON API
plus a static Three.js viewer. No third-party web dependencies.

Endpoints
---------
GET  /              -> static/index.html
GET  /api/state     -> full initial state (latent_dim, knots, codes, mesh, ...)
POST /api/update    -> {"knot_idx", "dim", "value"} -> {"mesh", "code"}
POST /api/reset     -> restore the reconstructed codes -> {"mesh"}
"""

from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

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

    # -- routes -----------------------------------------------------------
    def do_GET(self):  # noqa: N802
        if self.path in ("/", "/index.html"):
            self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
        elif self.path == "/api/state":
            try:
                self._send_json(self.session.state())
            except Exception as exc:  # pragma: no cover - surfaced to client
                logger.exception("state failed")
                self._send_json({"error": str(exc)}, status=500)
        else:
            self._send_json({"error": "not found"}, status=404)

    def do_POST(self):  # noqa: N802
        try:
            if self.path == "/api/update":
                data = self._read_json()
                self.session.set_value(
                    int(data["knot_idx"]), int(data["dim"]), float(data["value"])
                )
                self._send_json(
                    {"mesh": self.session.mesh(), "code": self.session.code(int(data["knot_idx"]))}
                )
            elif self.path == "/api/reset":
                self.session.reset()
                self._send_json({"mesh": self.session.mesh(), "codes": self.session.codes()})
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

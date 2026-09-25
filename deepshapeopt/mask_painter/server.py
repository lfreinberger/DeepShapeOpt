"""Web painter for the protected-region mask of the ``no_thinning`` constraint.

Endpoints
---------
GET  /           -> static/index.html
GET  /api/state  -> raster size, extent, background and current mask (PNG data URLs)
POST /api/save   -> {"mask": <PNG data URL>, "axis_range_mm": null | [lo, hi]}
"""
from __future__ import annotations

import argparse
import logging
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from deepshapeopt.latent_gui.server import _Handler
from deepshapeopt.mask_painter.backend import MaskPaintSession

logger = logging.getLogger(__name__)

INDEX = Path(__file__).resolve().parent / "static" / "index.html"


class _PaintHandler(_Handler):
    session: MaskPaintSession = None  # type: ignore[assignment]

    def do_GET(self):  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send_file(INDEX, "text/html; charset=utf-8")
        elif path == "/api/state":
            self._send_json(self.session.state())
        else:
            self._send_json({"error": "not found"}, status=404)

    def do_POST(self):  # noqa: N802
        try:
            if self.path == "/api/save":
                data = self._read_json()
                self._send_json(self.session.save(data["mask"], data.get("axis_range_mm")))
            else:
                self._send_json({"error": "not found"}, status=404)
        except (KeyError, ValueError) as exc:
            self._send_json({"error": str(exc)}, status=400)
        except Exception as exc:  # pragma: no cover - surfaced to client
            logger.exception("POST %s failed", self.path)
            self._send_json({"error": str(exc)}, status=500)


def serve(session: MaskPaintSession, host: str = "127.0.0.1", port: int = 8001):
    handler = type("BoundPaintHandler", (_PaintHandler,), {"session": session})
    httpd = ThreadingHTTPServer((host, port), handler)
    print(f"Mask painter ready: http://{host}:{port}  (Ctrl-C to stop; mask file {session.out})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


CLI_HELP = """Paint the protected region of the no_thinning constraint.

Renders the start design of a config (design-domain lattice with the reconstruction's
rec_parameters.pt) projected along one axis and serves a brush editor for a 2-D mask
that is extruded along that axis. Saving writes <out>.json and <out>.png; an existing
<out>.json is loaded for further editing.

Example
-------
    deepshapeopt paint-mask --config experiments/channel/config.json --out data/masks/channel.json
    deepshapeopt paint-mask --config ... --out ... --export-template   # background PNG only
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="deepshapeopt paint-mask", description=CLI_HELP,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="experiment config (schema v2)")
    ap.add_argument("--out", required=True, help="mask JSON to write (the PNG goes next to it)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--device", default=None, help="override run.device (e.g. cpu)")
    ap.add_argument("--params-file", default=None, help="latents to render instead of rec_parameters.pt")
    ap.add_argument("--pixel", type=float, default=None,
                    help="pixel size in mm of a new mask (default: the no_thinning grid_spacing, else 0.25)")
    ap.add_argument("--axis", choices=("x", "y", "z"), default="x", help="projection/extrusion axis of a new mask")
    ap.add_argument("--axis-samples", type=int, default=40, help="sections along the axis for the background")
    ap.add_argument("--export-template", action="store_true",
                    help="write <out>_background.png (and an empty mask) and exit without a server")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    session = MaskPaintSession(args.config, args.out, device=args.device, params_file=args.params_file,
                               pixel_mm=args.pixel, axis="xyz".index(args.axis),
                               axis_samples=args.axis_samples)
    if args.export_template:
        background, mask = session.export_template()
        print(f"Background: {background}\nMask:       {mask}")
        return 0
    serve(session, args.host, args.port)
    return 0

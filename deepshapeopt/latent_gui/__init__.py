"""Small web GUI for interactively editing the local latent codes of a
reconstructed DeepSDF B-spline lattice and watching the geometry update live.

Public API:
- :class:`~deepshapeopt.latent_gui.backend.LatentEditSession` — owns the
  in-memory lattice and turns latent-code edits into meshes.
- :func:`~deepshapeopt.latent_gui.server.serve` — stdlib HTTP server exposing
  the session as a tiny JSON API plus a static Three.js viewer.
"""

from deepshapeopt.latent_gui.backend import LatentEditSession

__all__ = ["LatentEditSession"]

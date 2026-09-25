"""A 2-D raster region extruded along one coordinate axis.

The mask is painted in the plane of the two remaining axes (for ``axis=0``: y to the
right, z up) and covers every point whose projection falls on a set pixel, optionally
only within ``axis_range_mm`` along the extrusion axis. On disk it is a JSON sidecar
next to a PNG::

    {"image": "<name>.png", "axis": 0, "extent_mm": [[u_lo, u_hi], [v_lo, v_hi]],
     "axis_range_mm": null}

Row 0 of the PNG is the top edge (``v_hi``). A pixel is set when its alpha is at least
128 and its brightest colour channel is at least 128, so white or any saturated colour
on black or on a transparent canvas marks the region.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


def plane_axes(axis: int) -> tuple[int, int]:
    """The in-plane axes ``(u, v)`` of the projection along ``axis``."""
    return tuple(i for i in range(3) if i != axis)


@dataclass
class ProjectedMask:
    image: np.ndarray                       # bool (H, W); row 0 at v_hi
    extent_mm: tuple[tuple[float, float], tuple[float, float]]
    axis: int = 0
    axis_range_mm: tuple[float, float] | None = None

    def __post_init__(self):
        self.image = np.asarray(self.image, dtype=bool)
        if self.image.ndim != 2:
            raise ValueError(f"mask image must be 2-D, got shape {self.image.shape}")
        if self.axis not in (0, 1, 2):
            raise ValueError(f"mask axis must be 0, 1 or 2, got {self.axis}")
        (u_lo, u_hi), (v_lo, v_hi) = self.extent_mm
        if not (u_hi > u_lo and v_hi > v_lo):
            raise ValueError(f"mask extent_mm must be [[lo, hi], [lo, hi]] with hi > lo, got {self.extent_mm}")
        self.extent_mm = ((float(u_lo), float(u_hi)), (float(v_lo), float(v_hi)))
        if self.axis_range_mm is not None:
            lo, hi = self.axis_range_mm
            self.axis_range_mm = (float(min(lo, hi)), float(max(lo, hi)))

    @property
    def shape(self) -> tuple[int, int]:
        return self.image.shape

    @classmethod
    def empty(cls, extent_mm, pixel_mm: float, axis: int = 0) -> "ProjectedMask":
        (u_lo, u_hi), (v_lo, v_hi) = extent_mm
        width = max(1, int(round((u_hi - u_lo) / pixel_mm)))
        height = max(1, int(round((v_hi - v_lo) / pixel_mm)))
        return cls(np.zeros((height, width), dtype=bool), extent_mm, axis)

    def pixel_centres(self) -> tuple[np.ndarray, np.ndarray]:
        """Physical ``u`` of every column and ``v`` of every row (row 0 = ``v_hi``)."""
        (u_lo, u_hi), (v_lo, v_hi) = self.extent_mm
        h, w = self.image.shape
        u = u_lo + (np.arange(w) + 0.5) * (u_hi - u_lo) / w
        v = v_hi - (np.arange(h) + 0.5) * (v_hi - v_lo) / h
        return u, v

    def contains(self, points_mm: torch.Tensor) -> torch.Tensor:
        """Boolean tensor: does each physical point ``(N, 3)`` lie in the extruded region."""
        (u_lo, u_hi), (v_lo, v_hi) = self.extent_mm
        h, w = self.image.shape
        ia, ib = plane_axes(self.axis)
        col = torch.floor((points_mm[:, ia] - u_lo) * (w / (u_hi - u_lo))).long()
        row = torch.floor((v_hi - points_mm[:, ib]) * (h / (v_hi - v_lo))).long()
        inside = (col >= 0) & (col < w) & (row >= 0) & (row < h)
        if self.axis_range_mm is not None:
            a = points_mm[:, self.axis]
            inside = inside & (a >= self.axis_range_mm[0]) & (a <= self.axis_range_mm[1])
        img = torch.as_tensor(self.image, device=points_mm.device)
        hit = torch.zeros_like(inside)
        hit[inside] = img[row[inside], col[inside]]
        return hit

    # -- files ------------------------------------------------------------------
    @staticmethod
    def image_from_rgba(rgba: np.ndarray) -> np.ndarray:
        """Set pixels of an ``(H, W, 4)`` RGBA array (the rule of the module docstring)."""
        return (rgba[..., 3] >= 128) & (rgba[..., :3].max(axis=2) >= 128)

    @classmethod
    def load(cls, path: str | Path) -> "ProjectedMask":
        from PIL import Image

        path = Path(path)
        meta = json.loads(path.read_text())
        rgba = np.asarray(Image.open(path.parent / meta["image"]).convert("RGBA"))
        return cls(cls.image_from_rgba(rgba), meta["extent_mm"], int(meta.get("axis", 0)), meta.get("axis_range_mm"))

    def save(self, path: str | Path) -> Path:
        """Write ``<path>`` (JSON) and ``<path stem>.png`` (white on black); returns the PNG."""
        from PIL import Image

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        png = path.with_suffix(".png")
        Image.fromarray(np.where(self.image, 255, 0).astype(np.uint8)).save(png)
        meta = {
            "image": png.name,
            "axis": self.axis,
            "extent_mm": [list(self.extent_mm[0]), list(self.extent_mm[1])],
            "axis_range_mm": list(self.axis_range_mm) if self.axis_range_mm is not None else None,
        }
        path.write_text(json.dumps(meta, indent=2) + "\n")
        return png

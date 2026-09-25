"""Start design seen along one axis, and the protected-region mask painted on top of it.

The background is rendered from the design-domain lattice with the reconstruction latents
(the start design the ``no_thinning`` constraint protects): the grey level is the solid
fraction along the projection axis, the coloured outlines are the solid sections at the
low (blue) and the high (orange) end of the design domain.
"""
from __future__ import annotations

import base64
import io
import logging
import threading
from pathlib import Path

import numpy as np
import torch
from DeepSDFStruct.utils import with_float32_lattice

from deepshapeopt.config import load_config, make_run_paths
from deepshapeopt.geometry.projected_mask import ProjectedMask, plane_axes

logger = logging.getLogger(__name__)

AXIS_NAMES = ("x", "y", "z")
LOW_END = np.array([47, 125, 224], dtype=np.uint8)
HIGH_END = np.array([224, 140, 40], dtype=np.uint8)
CHUNK = 262144


def _png_data_url(array: np.ndarray) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _rgba_from_data_url(url: str) -> np.ndarray:
    from PIL import Image

    raw = base64.b64decode(url.split(",", 1)[1])
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGBA"))


def _outline(solid: np.ndarray) -> np.ndarray:
    """Solid pixels with a fluid 4-neighbour."""
    pad = np.pad(solid, 1, mode="edge")
    fluid_nb = ~pad[:-2, 1:-1] | ~pad[2:, 1:-1] | ~pad[1:-1, :-2] | ~pad[1:-1, 2:]
    return solid & fluid_nb


def _default_pixel(cfg) -> float:
    for con in cfg.constraints:
        if con.get("type") == "no_thinning" and con.get("grid_spacing"):
            return float(con["grid_spacing"])
    return 0.25


class MaskPaintSession:
    def __init__(self, config_path: str | Path, out: str | Path, *, device: str | None = None,
                 params_file: str | Path | None = None, pixel_mm: float | None = None,
                 axis: int = 0, axis_samples: int = 40):
        from deepshapeopt.parametrization.deepsdf import DeepSDFLattice

        self.config_path = Path(config_path).resolve()
        self.out = Path(out).resolve()
        cfg = load_config(self.config_path)
        if device is not None:
            cfg.run.device = device
        paths = make_run_paths(cfg, self.config_path.parent).ensure()
        params = Path(params_file).resolve() if params_file else paths.reconstruction / "rec_parameters.pt"
        if not params.exists():
            raise FileNotFoundError(
                f"{params} does not exist; run the reconstruction of this config first "
                f"(deepshapeopt reconstruct / optimize) or pass --params-file")
        lattice = DeepSDFLattice(cfg, paths)
        lattice.load(params)
        logger.info("Loaded start design from %s", params)
        self.lattice_struct = lattice.lattice_struct
        self.frame = lattice.frame
        self.domain = self.frame.design_domain.detach().cpu().numpy().astype(float)
        self._lock = threading.Lock()

        if self.out.exists():
            self.mask = ProjectedMask.load(self.out)
            logger.info("Editing the existing mask %s", self.out)
        else:
            ia, ib = plane_axes(axis)
            extent = [[self.domain[0, ia], self.domain[1, ia]], [self.domain[0, ib], self.domain[1, ib]]]
            self.mask = ProjectedMask.empty(extent, pixel_mm or _default_pixel(cfg), axis)
        self.background = self._render_background(int(axis_samples))

    def _render_background(self, n_samples: int) -> np.ndarray:
        mask = self.mask
        axis = mask.axis
        ia, ib = plane_axes(axis)
        u, v = mask.pixel_centres()
        h, w = mask.shape
        lo, hi = self.domain[0, axis], self.domain[1, axis]
        step = (hi - lo) / n_samples
        levels = lo + (np.arange(n_samples) + 0.5) * step
        device = self.frame.center.device
        plane = torch.zeros((h * w, 3), dtype=torch.float32, device=device)
        vv, uu = np.meshgrid(v, u, indexing="ij")
        plane[:, ia] = torch.as_tensor(uu.reshape(-1), dtype=torch.float32, device=device)
        plane[:, ib] = torch.as_tensor(vv.reshape(-1), dtype=torch.float32, device=device)

        def _compute(_bounds):
            frac = torch.zeros(h * w, device=device)
            ends = []
            with torch.no_grad():
                for k, level in enumerate(levels):
                    plane[:, axis] = float(level)
                    pts = self.frame.to_norm(plane)
                    solid = torch.cat([self.lattice_struct(pts[i:i + CHUNK]).reshape(-1) > 0.0
                                       for i in range(0, pts.shape[0], CHUNK)])
                    frac += solid.float()
                    if k in (0, n_samples - 1):
                        ends.append(solid.reshape(h, w).cpu().numpy())
            return (frac / n_samples).reshape(h, w).cpu().numpy(), ends

        logger.info("Rendering the %s-projection of the start design (%dx%d px, %d samples)",
                    AXIS_NAMES[axis], w, h, n_samples)
        frac, (low, high) = with_float32_lattice(self.lattice_struct, self.frame.box_norm, _compute)
        grey = (255.0 - 190.0 * frac).astype(np.uint8)
        rgb = np.repeat(grey[..., None], 3, axis=2)
        rgb[_outline(high)] = HIGH_END
        rgb[_outline(low)] = LOW_END
        return rgb

    # -- API ----------------------------------------------------------------
    def state(self) -> dict:
        mask = self.mask
        h, w = mask.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[mask.image] = (230, 60, 60, 255)
        ia, ib = plane_axes(mask.axis)
        return {
            "out": str(self.out),
            "width": w,
            "height": h,
            "extent_mm": [list(mask.extent_mm[0]), list(mask.extent_mm[1])],
            "axis": AXIS_NAMES[mask.axis],
            "plane_axes": [AXIS_NAMES[ia], AXIS_NAMES[ib]],
            "axis_domain_mm": [float(self.domain[0, mask.axis]), float(self.domain[1, mask.axis])],
            "axis_range_mm": list(mask.axis_range_mm) if mask.axis_range_mm is not None else None,
            "background": _png_data_url(self.background),
            "mask": _png_data_url(rgba),
        }

    def save(self, mask_data_url: str, axis_range_mm=None) -> dict:
        image = ProjectedMask.image_from_rgba(_rgba_from_data_url(mask_data_url))
        with self._lock:
            if image.shape != self.mask.shape:
                raise ValueError(f"painted mask has shape {image.shape}, expected {self.mask.shape}")
            self.mask.image = image
            self.mask.axis_range_mm = tuple(sorted(float(a) for a in axis_range_mm)) if axis_range_mm else None
            png = self.mask.save(self.out)
        (u_lo, u_hi), (v_lo, v_hi) = self.mask.extent_mm
        area = float(image.sum()) * (u_hi - u_lo) * (v_hi - v_lo) / image.size
        logger.info("Saved %s (%d px, %.1f mm2)", self.out, int(image.sum()), area)
        return {"saved": str(self.out), "png": str(png), "pixels": int(image.sum()), "area_mm2": area}

    def export_template(self) -> tuple[Path, Path]:
        """Background PNG next to the mask, plus an empty mask when none exists yet."""
        from PIL import Image

        background = self.out.with_name(self.out.stem + "_background.png")
        background.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.background).save(background)
        if not self.out.exists():
            self.mask.save(self.out)
        return background, self.out

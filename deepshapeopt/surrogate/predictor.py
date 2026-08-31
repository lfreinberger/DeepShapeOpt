"""Inference-side wrapper: Transolver surrogate as differentiable drag objective.

``TransolverSurrogate.objective`` is the drop-in replacement for the OpenFOAM
forward step in the optimization loop: it assembles the query cloud from the
(differentiable) wall surface, predicts (U, p), integrates the drag and
returns a scalar objective carrying the autograd graph back to the design
parameters.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from .dataset import Normalizer
from .drag import drag_from_fields
from .query_points import build_query_cloud
from .transolver import Transolver

logger = logging.getLogger(__name__)


class TransolverSurrogate:
    def __init__(self, model: Transolver, normalizer: Normalizer, cfg: dict):
        self.model = model
        self.norm = normalizer
        self.cfg = cfg
        self.device = torch.device(cfg.get("device", "cuda"))
        self.nu = float(cfg.get("nu", 1.0))
        self.u_inf = float(cfg.get("u_inf", 1.0))
        self.a_ref = float(cfg.get("a_ref", 1.0))
        self.visc_scale = float(cfg.get("visc_scale", 1.0))
        self.direction = tuple(cfg.get("drag_direction", (1.0, 0.0, 0.0)))
        self.model.to(self.device).eval()
        self.norm.to(self.device)

    @classmethod
    def from_config(cls, cfg: dict) -> "TransolverSurrogate":
        """Load from an ``optimization.surrogate`` config block.

        Required key ``checkpoint``: path to a ``train_transolver.py``
        checkpoint (self-contained: model_cfg + state dict + norm stats).
        """
        ckpt_path = Path(cfg["checkpoint"])
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = Transolver(**ckpt["model_cfg"])
        model.load_state_dict(ckpt["model_state"])
        normalizer = Normalizer(ckpt["norm_stats"])
        merged = {**ckpt.get("surrogate_cfg", {}), **cfg}
        logger.info(
            "Loaded Transolver surrogate from %s (epoch %s, val %s)",
            ckpt_path, ckpt.get("epoch"), ckpt.get("val_metric"),
        )
        return cls(model, normalizer, merged)

    def predict(self, cloud) -> tuple[torch.Tensor, torch.Tensor]:
        """De-normalized predictions on a query cloud: ``(U [N,3], p [N])``.

        Differentiable w.r.t. the cloud features (model weights stay frozen).
        """
        feats = self.norm.norm_x(cloud.feats.to(self.device))
        out = self.model(feats[None])[0]
        y = self.norm.denorm_y(out)
        return y[:, :3], y[:, 3]

    def objective(
        self, surface_points: torch.Tensor, wall_tris: torch.Tensor, sdf_fn
    ) -> tuple[torch.Tensor, dict]:
        """Differentiable drag objective for the current design.

        Returns ``(J, diagnostics)``; diagnostics contain the per-vertex wall
        traction (detachable stand-in for the adjoint sensitivity field) and
        the pressure/viscous split.
        """
        cloud = build_query_cloud(surface_points, wall_tris, sdf_fn, self.cfg)
        U, p = self.predict(cloud)
        J, diag = drag_from_fields(
            U, p, cloud, nu=self.nu, direction=self.direction,
            u_inf=self.u_inf, a_ref=self.a_ref, visc_scale=self.visc_scale,
        )
        diag["n_points"] = cloud.n_points
        diag["n_surface"] = cloud.n_surface
        return J, diag

"""Inference-side wrapper: Transolver surrogate as differentiable drag objective.

``TransolverSurrogate.objective`` is the drop-in replacement for the OpenFOAM
forward step in the optimization loop: it assembles the query cloud from the
(differentiable) wall surface, predicts the fields, integrates the drag and
returns a scalar objective carrying the autograd graph back to the design
parameters.

The viscous drag source follows the checkpoint: 7 target channels
(``[U, p, tau_w]``) -> "tau" mode (predicted wall shear stress), 4 channels
(``[U, p]``) -> "fd" mode (probe-shell finite difference). The mode is
derived from the checkpoint and cannot be overridden by a run config. The
same holds for the input feature set (``geo`` / ``lat`` / ``geo+lat``, see
:mod:`deepshapeopt.surrogate.features`): it comes with the normalization
statistics; a latent feature set needs a ``latent_fn`` from a lattice design.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from .dataset import Normalizer
from .drag import drag_from_fields
from .features import LatentCodeField, needs_latent
from .query_points import build_query_cloud, model_input
from .transolver import Transolver

logger = logging.getLogger(__name__)


class TransolverSurrogate:
    def __init__(self, model: Transolver, normalizer: Normalizer, cfg: dict):
        self.model = model
        self.norm = normalizer
        self.device = torch.device(cfg.get("device", "cuda"))
        self.nu = float(cfg.get("nu", 1.0))
        self.u_inf = float(cfg.get("u_inf", 1.0))
        self.a_ref = float(cfg.get("a_ref", 1.0))
        self.visc_scale = float(cfg.get("visc_scale", 1.0))
        self.pressure_scale = float(cfg.get("pressure_scale", 1.0))
        self.direction = tuple(cfg.get("drag_direction", (1.0, 0.0, 0.0)))
        mode = "tau" if normalizer.n_y >= 7 else "fd"
        requested = cfg.get("viscous_mode")
        if requested and requested != mode:
            logger.warning(
                "viscous_mode %r requested, but the checkpoint has %d target "
                "channels; using %r", requested, normalizer.n_y, mode,
            )
        self.viscous_mode = mode
        self.feature_set = normalizer.feature_set
        self.latent_spec = normalizer.latent_spec
        self._latent_checked: set[int] = set()
        self.cfg = {**cfg, "viscous_mode": mode}
        self.model.to(self.device).eval()
        self.norm.to(self.device)

    @property
    def needs_latent(self) -> bool:
        return needs_latent(self.feature_set)

    @classmethod
    def from_config(cls, cfg: dict) -> "TransolverSurrogate":
        """Load from an ``optimization.surrogate`` config block.

        Required key ``checkpoint``: path to a ``train_transolver.py``
        checkpoint (self-contained: model_cfg + state dict + norm stats).
        """
        ckpt_path = Path(cfg["checkpoint"])
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        out_dim = int(ckpt["model_cfg"].get("out_dim", 4))
        n_y = len(ckpt["norm_stats"]["mean_y"])
        if out_dim != n_y:
            raise ValueError(
                f"{ckpt_path}: model out_dim {out_dim} != {n_y} normalization "
                "channels (stale dataset_stats.json at training time?)"
            )
        in_dim = int(ckpt["model_cfg"].get("in_dim", 7))
        n_x = len(ckpt["norm_stats"]["mean_x"])
        if in_dim != n_x:
            raise ValueError(
                f"{ckpt_path}: model in_dim {in_dim} != {n_x} normalization "
                "channels (stale dataset_stats.json at training time?)"
            )
        model = Transolver(**ckpt["model_cfg"])
        model.load_state_dict(ckpt["model_state"])
        normalizer = Normalizer(ckpt["norm_stats"])
        merged = {**ckpt.get("surrogate_cfg", {}), **cfg}
        surrogate = cls(model, normalizer, merged)
        logger.info(
            "Loaded Transolver surrogate from %s (epoch %s, features %s [%d], "
            "viscous_mode %s, val %s)",
            ckpt_path, ckpt.get("epoch"), surrogate.feature_set, n_x,
            surrogate.viscous_mode, ckpt.get("val_metric"),
        )
        return surrogate

    def _check_latent_fn(self, latent_fn) -> None:
        if self.needs_latent and latent_fn is None:
            raise ValueError(
                f"checkpoint uses feature set {self.feature_set!r} (needs z(x)); pass a "
                "latent_fn -- only the DeepSDF lattice parametrization provides one"
            )
        if latent_fn is not None and id(latent_fn) not in self._latent_checked:
            if isinstance(latent_fn, LatentCodeField):
                latent_fn.check_spec(self.latent_spec)
            self._latent_checked.add(id(latent_fn))

    def predict_raw(self, cloud) -> torch.Tensor:
        """De-normalized prediction ``[N, n_y]`` on a query cloud.

        Differentiable w.r.t. the cloud features (model weights stay frozen).
        """
        feats = self.norm.norm_x(model_input(cloud, self.feature_set).to(self.device))
        out = self.model(feats[None])[0]
        return self.norm.denorm_y(out)

    def predict(self, cloud) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """``(U [N,3], p [N], tau_w [N,3] | None)`` -- tau only in "tau" mode."""
        y = self.predict_raw(cloud)
        tau = y[:, 4:7] if self.viscous_mode == "tau" else None
        return y[:, :3], y[:, 3], tau

    def drag_from_prediction(self, y_raw: torch.Tensor, cloud) -> tuple[torch.Tensor, dict]:
        """Drag integral of a de-normalized prediction with this surrogate's
        calibration; the single place that knows the ``drag_from_fields``
        arguments (used by the optimizer, training metric, evaluation,
        bias calibration and field export)."""
        return drag_from_fields(
            y_raw[:, :3], y_raw[:, 3], cloud,
            nu=self.nu, direction=self.direction,
            u_inf=self.u_inf, a_ref=self.a_ref,
            visc_scale=self.visc_scale, pressure_scale=self.pressure_scale,
            tau_w=y_raw[:, 4:7] if self.viscous_mode == "tau" else None,
        )

    def build_cloud(
        self, surface_points: torch.Tensor, wall_tris: torch.Tensor, sdf_fn, latent_fn=None
    ):
        """Query cloud for this checkpoint (latent block only when needed)."""
        self._check_latent_fn(latent_fn)
        return build_query_cloud(
            surface_points, wall_tris, sdf_fn, self.cfg,
            latent_fn=latent_fn if self.needs_latent else None,
        )

    def objective(
        self, surface_points: torch.Tensor, wall_tris: torch.Tensor, sdf_fn, latent_fn=None
    ) -> tuple[torch.Tensor, dict]:
        """Differentiable drag objective for the current design.

        ``latent_fn`` (``z(x_phys)``, e.g. ``SdfHexMeshPipeline.latent_at_phys``)
        is required for latent feature sets and ignored otherwise. Returns
        ``(J, diagnostics)``; diagnostics contain the per-vertex wall traction
        (detachable stand-in for the adjoint sensitivity field) and the
        pressure/viscous split.
        """
        cloud = self.build_cloud(surface_points, wall_tris, sdf_fn, latent_fn)
        y = self.predict_raw(cloud)
        J, diag = self.drag_from_prediction(y, cloud)
        diag["n_points"] = cloud.n_points
        diag["n_surface"] = cloud.n_surface
        return J, diag

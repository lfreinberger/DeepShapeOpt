"""Per-point feature sets of the Transolver surrogate and the latent-code field.

The surrogate's per-point input is assembled from named blocks:

- ``pos``     [3]  physical query position (always first)
- ``sdf``     [1]  design SDF in physical units
- ``normal``  [3]  into-fluid unit normal on the wall, zero elsewhere
- ``latent``  [L]  ``z(x)``: the B-spline-interpolated latent code of the
                   DeepSDF lattice at the query position -- exactly the latent
                   half of the decoder input the geometry was generated from.

``FEATURE_SETS`` names the three ablation variants. The feature set is a
property of a trained checkpoint (stored with the normalization statistics,
like the target channels); the same :func:`assemble_features` builds the
model input from stored dataset arrays and from the live query cloud.

:class:`LatentCodeField` evaluates ``z(x)`` in physical coordinates. Queries
are clamped to the design domain first: the torch spline extrapolates
linearly outside its knot span, and the surrogate's far-field points lie up
to ten design-box lengths away. Outside the box the lattice SDF is the
distance to the box anyway (independent of the codes), so the clamped value,
the code of the nearest design-box point, is the natural extension.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "geo": ("pos", "sdf", "normal"),
    "lat": ("pos", "latent"),
    "geo+lat": ("pos", "sdf", "normal", "latent"),
}
DEFAULT_FEATURE_SET = "geo"
_BLOCK_DIMS = {"pos": 3, "sdf": 1, "normal": 3}
_BLOCK_CHANNELS = {
    "pos": ["x", "y", "z"],
    "sdf": ["sdf"],
    "normal": ["nx", "ny", "nz"],
}


def feature_blocks(feature_set: str) -> tuple[str, ...]:
    try:
        return FEATURE_SETS[feature_set]
    except KeyError:
        raise ValueError(
            f"unknown feature set {feature_set!r}; expected one of {sorted(FEATURE_SETS)}"
        ) from None


def needs_latent(feature_set: str) -> bool:
    return "latent" in feature_blocks(feature_set)


def feature_dim(feature_set: str, latent_dim: int = 0) -> int:
    """Model ``in_dim`` of a feature set (``latent_dim`` only matters with ``latent``)."""
    n = 0
    for block in feature_blocks(feature_set):
        if block == "latent":
            if latent_dim <= 0:
                raise ValueError(f"feature set {feature_set!r} needs latent_dim > 0")
            n += int(latent_dim)
        else:
            n += _BLOCK_DIMS[block]
    return n


def feature_channels(feature_set: str, latent_dim: int = 0) -> list[str]:
    """Channel names in model-input order (stored in ``dataset_stats.json``)."""
    names: list[str] = []
    for block in feature_blocks(feature_set):
        if block == "latent":
            names += [f"z{i}" for i in range(int(latent_dim))]
        else:
            names += _BLOCK_CHANNELS[block]
    return names


def assemble_features(blocks: dict[str, Any], feature_set: str):
    """Concatenate the named blocks of ``feature_set`` in canonical order.

    Accepts torch tensors (any device; the autograd graph is kept) or numpy
    arrays, but not mixed. 1-D blocks are treated as one channel.
    """
    parts = []
    for name in feature_blocks(feature_set):
        block = blocks.get(name)
        if block is None:
            raise ValueError(f"feature set {feature_set!r} needs the {name!r} block")
        if block.ndim == 1:
            block = block[:, None]
        parts.append(block)
    if torch.is_tensor(parts[0]):
        return torch.cat(parts, dim=1)
    return np.concatenate(parts, axis=1)


# ---------------------------------------------------------------------------
# z(x): the latent-code field of the lattice parametrization
# ---------------------------------------------------------------------------


def make_latent_spec(design_domain, tiling, spline_degree, latent_dim: int) -> dict:
    """JSON-serializable description of a latent B-spline parametrization."""
    dd = np.asarray(design_domain, dtype=np.float64)
    if dd.shape != (2, 3):
        raise ValueError(f"design_domain must be (2, 3), got {dd.shape}")
    return {
        "design_domain": dd.tolist(),
        "tiling": [int(t) for t in tiling],
        "spline_degree": [int(d) for d in spline_degree],
        "latent_dim": int(latent_dim),
    }


def latent_spec_from_rec_cfg(rec_cfg: dict, latent_dim: int) -> dict:
    """Latent spec of the lattice a ``reconstruction`` config block builds."""
    return make_latent_spec(
        rec_cfg["design_domain"], rec_cfg["tiling"], rec_cfg["spline_degree"], latent_dim
    )


def _spec_from_spline(spline_sp, design_domain, latent_dim: int) -> dict:
    tiling = [len(np.unique(np.asarray(kv, dtype=np.float64))) - 1 for kv in spline_sp.knot_vectors]
    return make_latent_spec(design_domain, tiling, list(spline_sp.degrees), latent_dim)


class LatentCodeField:
    """``z(x_phys) -> [N, latent_dim]``, differentiable w.r.t. the control
    points and the query positions.

    ``param_spline`` is the lattice's :class:`SplineParametrization` (knot
    vectors span ``frame.box_norm``); ``frame`` the :class:`DomainFrame` that
    maps physical to normalized coordinates. ``spec`` documents the
    parametrization so a checkpoint trained on one lattice layout can refuse
    another (:meth:`check_spec`).
    """

    def __init__(self, param_spline, frame, spec: dict, scope=None):
        self.param_spline = param_spline
        self.frame = frame
        self.spec = spec
        # Optional evaluation context, e.g. the design's float32 scope.
        self.scope = scope

    @classmethod
    def from_spec(cls, spec: dict, param, device="cpu") -> "LatentCodeField":
        """Stand-alone field from a spec and its control values ``[n_ctrl, latent_dim]``
        (dataset preprocessing: the ``param`` array stored in every sample)."""
        from DeepSDFStruct.geom_reconstruction import build_parameter_spline
        from DeepSDFStruct.parametrization import SplineParametrization

        from ..domain_frame import DomainFrame

        frame = DomainFrame.from_design_domain(spec["design_domain"], device=device)
        box_norm = frame.box_norm.detach().cpu().numpy().astype(np.float64)
        spline_sp = build_parameter_spline(
            spline_degrees=list(spec["spline_degree"]),
            tiling=list(spec["tiling"]),
            latent_dim=int(spec["latent_dim"]),
            bounds=box_norm,
        )
        param = torch.as_tensor(np.asarray(param), dtype=torch.float32)
        if tuple(param.shape) != (spline_sp.control_points.shape[0], int(spec["latent_dim"])):
            raise ValueError(
                f"param shape {tuple(param.shape)} does not match the latent spec "
                f"({spline_sp.control_points.shape[0]} control points x {spec['latent_dim']})"
            )
        param_spline = SplineParametrization(spline_sp, device=device)
        param_spline.set_param(param)
        return cls(param_spline, frame, dict(spec))

    @classmethod
    def from_lattice(cls, lattice_struct, frame, scope=None) -> "LatentCodeField":
        """Live field of a ``LatticeSDFStruct`` (shares its control points, so
        the autograd graph reaches the design parameters). ``scope(fn)`` wraps
        every evaluation (the design's float32 scope)."""
        param_spline = lattice_struct.parametrization
        torch_spline = getattr(param_spline, "torch_spline", None)
        if torch_spline is None:
            raise TypeError(
                "latent features need a SplineParametrization lattice, got "
                f"{type(param_spline).__name__}"
            )
        latent_dim = int(torch_spline.control_points.shape[1])
        spec = _spec_from_spline(
            torch_spline.spline, frame.design_domain.detach().cpu().numpy(), latent_dim
        )
        return cls(param_spline, frame, spec, scope=scope)

    @property
    def latent_dim(self) -> int:
        return int(self.spec["latent_dim"])

    def _evaluate(self, x_phys: torch.Tensor) -> torch.Tensor:
        dd = self.frame.design_domain.to(device=x_phys.device, dtype=x_phys.dtype)
        x_cl = torch.clamp(x_phys, min=dd[0][None, :], max=dd[1][None, :])
        return self.param_spline(self.frame.to_norm(x_cl))

    def __call__(self, x_phys: torch.Tensor) -> torch.Tensor:
        if self.scope is None:
            return self._evaluate(x_phys)
        return self.scope(lambda: self._evaluate(x_phys))

    def check_spec(self, expected: dict | None, atol: float = 1e-6) -> None:
        """Raise if this field's parametrization differs from ``expected``
        (the spec a checkpoint was trained with)."""
        if expected is None:
            return
        mine = self.spec
        for key in ("tiling", "spline_degree", "latent_dim"):
            if list(np.atleast_1d(mine[key])) != list(np.atleast_1d(expected[key])):
                raise ValueError(
                    f"latent parametrization mismatch: {key} {mine[key]} != "
                    f"checkpoint {expected[key]}"
                )
        if not np.allclose(mine["design_domain"], expected["design_domain"], atol=atol):
            raise ValueError(
                f"latent parametrization mismatch: design_domain {mine['design_domain']} "
                f"!= checkpoint {expected['design_domain']}"
            )

"""PCA reduced basis for the latent design variables in shape optimization.

The drag optimization uses the B-spline lattice control points as design
variables, where each control point is a ``latent_dim`` (e.g. 32) DeepSDF latent
code. That dimensionality exists so the *decoder* reconstructs the whole training
set well; a single optimization run only needs to move along the directions in
which real shapes actually vary. This module fits a linear principal-component
basis on the latent codes learned during DeepSDF training so the optimizer can
work in a small ``k``-dimensional coefficient space instead.

The reparametrization is linear::

    z = mean + c @ V_k.T          (coefficients -> latent control points)
    c = (z - mean) @ V_k          (latent control points -> coefficients)

so gradients of the objective/constraints w.r.t. the latent control points
``dz`` project to coefficient space as ``dc = dz @ V_k``.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def _resolve_model_dir(model_path) -> Path:
    """Resolve a model identifier (resolved str path or PretrainedModels enum)."""
    if isinstance(model_path, (str, Path)):
        return Path(model_path)
    # Enum form: look it up in the DeepSDFStruct registry.
    from DeepSDFStruct.pretrained_models import _MODEL_REGISTRY

    return Path(_MODEL_REGISTRY[model_path])


def gather_training_latents(
    model_path, checkpoint: str = "latest", device="cpu"
) -> torch.Tensor:
    """Stack the latent codes learned during DeepSDF training into an ``(N, D)`` matrix.

    For a spline latent-field model the learned values are the per-scene B-spline
    control points stored under ``latent_fields_state_dict`` (keys ending in
    ``control_points``). For a classic auto-decoder the per-shape ``latent_codes``
    tensor is used instead.
    """
    latent_file = _resolve_model_dir(model_path) / "LatentCodes" / f"{checkpoint}.pth"
    if not latent_file.is_file():
        raise FileNotFoundError(f"No latent code file at {latent_file}")

    data = torch.load(latent_file, map_location=device, weights_only=False)

    samples = []
    state = data.get("latent_fields_state_dict") if isinstance(data, dict) else None
    if state:
        for key, value in state.items():
            if key.endswith("control_points"):
                samples.append(value.reshape(-1, value.shape[-1]).to(device))

    if samples:
        latents = torch.cat(samples, dim=0)
    else:
        codes = data["latent_codes"] if isinstance(data, dict) else data
        if isinstance(codes, dict):  # nn.Embedding state dict
            codes = codes["weight"]
        codes = torch.as_tensor(codes, device=device)
        latents = codes.reshape(-1, codes.shape[-1])

    latents = latents.float()
    if float(latents.std()) == 0.0:
        raise ValueError(
            f"Training latents at {latent_file} have zero variance; cannot fit PCA. "
            "Check that the checkpoint stores learned latent fields (not just the "
            "dummy compatibility codes)."
        )
    logger.info(
        "Gathered %d training latent samples of dim %d",
        latents.shape[0],
        latents.shape[1],
    )
    return latents


def compute_latent_pca(latent_samples: torch.Tensor, k: int):
    """Fit PCA on ``(N, D)`` samples.

    Returns ``(mean (D,), components (D, k), explained_variance_ratio (k,))`` where
    the columns of ``components`` are the top-``k`` principal directions ``V_k``.
    """
    x = latent_samples.float()
    d = x.shape[1]
    if not 1 <= k <= d:
        raise ValueError(f"n_components={k} must be in [1, {d}]")

    mean = x.mean(dim=0)
    xc = x - mean
    # Centered data SVD: xc = U @ diag(s) @ Vh; principal directions are rows of Vh.
    _, s, vh = torch.linalg.svd(xc, full_matrices=False)
    components = vh[:k].T.contiguous()  # (D, k)

    var = s ** 2
    explained = (var[:k] / var.sum()).contiguous()
    logger.info(
        "PCA: k=%d captures %.4f of variance (per-component: %s)",
        k,
        float(explained.sum()),
        ", ".join(f"{float(e):.3f}" for e in explained),
    )
    return mean, components, explained


class PCALatentBasis:
    """Linear PCA reparametrization of the per-control-point latent codes."""

    def __init__(self, mean: torch.Tensor, components: torch.Tensor):
        # mean: (D,), components V_k: (D, k)
        self.mean = mean
        self.components = components

    @property
    def latent_dim(self) -> int:
        return self.components.shape[0]

    @property
    def n_components(self) -> int:
        return self.components.shape[1]

    def to(self, device=None, dtype=None) -> "PCALatentBasis":
        self.mean = self.mean.to(device=device, dtype=dtype)
        self.components = self.components.to(device=device, dtype=dtype)
        return self

    def to_coeff(self, z: torch.Tensor) -> torch.Tensor:
        """``(n_cp, D)`` latent control points -> ``(n_cp, k)`` coefficients."""
        return (z - self.mean) @ self.components

    def to_latent(self, c: torch.Tensor) -> torch.Tensor:
        """``(n_cp, k)`` coefficients -> ``(n_cp, D)`` latent control points."""
        return self.mean + c @ self.components.T

    def project_grad(self, dz: torch.Tensor) -> torch.Tensor:
        """Project a latent-space gradient ``(n_cp, D)`` to coefficient space.

        Returns the flattened ``(n_cp * k,)`` gradient so it lines up with the
        flattened MMA design vector.
        """
        dz = dz.reshape(-1, self.latent_dim)
        return (dz @ self.components).reshape(-1)

    def save(self, path) -> None:
        torch.save(
            {"mean": self.mean.detach().cpu(), "components": self.components.detach().cpu()},
            path,
        )

    @classmethod
    def load(cls, path, device="cpu") -> "PCALatentBasis":
        data = torch.load(path, map_location=device, weights_only=False)
        return cls(data["mean"].to(device), data["components"].to(device))


def build_pca_basis(
    model_path, checkpoint: str, k: int, device="cpu", cache_path=None
) -> PCALatentBasis:
    """Build (or load from cache) a :class:`PCALatentBasis` fit on a model's latents."""
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.is_file():
            basis = PCALatentBasis.load(cache_path, device=device)
            if basis.n_components == k:
                logger.info("Loaded cached PCA basis from %s", cache_path)
                return basis
            logger.info(
                "Cached PCA basis has k=%d != requested %d; refitting",
                basis.n_components,
                k,
            )

    latents = gather_training_latents(model_path, checkpoint, device=device)
    mean, components, _ = compute_latent_pca(latents, k)
    basis = PCALatentBasis(mean, components)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        basis.save(cache_path)
        logger.info("Saved PCA basis to %s", cache_path)
    return basis

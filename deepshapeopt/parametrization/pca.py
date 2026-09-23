"""PCA reduced basis for the latent design variables in shape optimization.

The drag optimization uses the B-spline lattice control points as design
variables, where each control point is a ``latent_dim`` (e.g. 32) DeepSDF latent
code. That dimensionality exists so the *decoder* reconstructs the whole training
set well; a single optimization run only needs to move along the directions in
which real shapes actually vary. This module fits a linear principal-component
basis on the latent codes learned during DeepSDF training so the optimizer can
work in a small ``k``-dimensional coefficient space instead.

The reparametrization is linear, taken around a reference point ``r`` -- the
training ``mean`` by default (classic PCA), or the reconstructed latent field
``lambda^0`` when the shape optimizer sets it, so the coefficients become
*deltas around the reconstruction*::

    z = r + (c * scale) @ V_k.T      (whitened coefficients -> latent control points)
    c = ((z - r) @ V_k) / scale      (latent control points -> whitened coefficients)

so gradients of the objective/constraints w.r.t. the latent control points
``dz`` project to coefficient space as ``dc = (dz @ V_k) * scale`` (independent
of the reference).
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

    Returns ``(mean (D,), components (D, k), explained_variance_ratio (k,), scale (k,))``
    where the columns of ``components`` are the top-``k`` principal directions ``V_k`` and
    ``scale`` is the per-component standard deviation (sqrt eigenvalue) used for whitening.
    """
    x = latent_samples.float()
    n, d = x.shape
    if not 1 <= k <= d:
        raise ValueError(f"n_components={k} must be in [1, {d}]")

    mean = x.mean(dim=0)
    xc = x - mean
    # Centered data SVD: xc = U @ diag(s) @ Vh; principal directions are rows of Vh.
    _, s, vh = torch.linalg.svd(xc, full_matrices=False)
    components = vh[:k].T.contiguous()  # (D, k)

    var = s ** 2
    explained = (var[:k] / var.sum()).contiguous()
    # Per-component std (sqrt eigenvalue). Whitening by this makes each coefficient
    # ~unit variance, so a single coefficient box [-b, b] means +-b*sigma uniformly
    # (high- and low-variance directions are no longer bounded by the same raw box).
    scale = (s[:k] / max(n - 1, 1) ** 0.5).clamp_min(1e-8).contiguous()
    logger.info(
        "PCA: k=%d captures %.4f of variance (per-component: %s)",
        k,
        float(explained.sum()),
        ", ".join(f"{float(e):.3f}" for e in explained),
    )
    return mean, components, explained, scale


class PCALatentBasis:
    """Linear PCA reparametrization of the per-control-point latent codes."""

    def __init__(
        self,
        mean: torch.Tensor,
        components: torch.Tensor,
        scale: torch.Tensor | None = None,
        reference: torch.Tensor | None = None,
    ):
        # mean: (D,), components V_k: (D, k), scale: (k,) per-component std for whitening.
        # scale=None -> unit scale (no whitening; raw PCA coefficients).
        self.mean = mean
        self.components = components
        self.scale = scale if scale is not None else torch.ones(
            components.shape[1], device=components.device, dtype=components.dtype
        )
        # Point coefficients are expressed relative to. Defaults to the training ``mean``
        # (classic PCA). The shape optimizer calls ``set_reference`` with the reconstructed
        # field lambda^0 (shape (n_cp, D)) to switch to the delta parametrization: then
        # c=0 is exactly the reconstruction, its detail outside span(V_k) is preserved, and
        # the box bounds the perturbation rather than the absolute code.
        self.reference = reference if reference is not None else self.mean

    @property
    def latent_dim(self) -> int:
        return self.components.shape[0]

    @property
    def n_components(self) -> int:
        return self.components.shape[1]

    def to(self, device=None, dtype=None) -> "PCALatentBasis":
        ref_is_mean = self.reference is self.mean
        self.mean = self.mean.to(device=device, dtype=dtype)
        self.components = self.components.to(device=device, dtype=dtype)
        self.scale = self.scale.to(device=device, dtype=dtype)
        # Preserve the mean-is-reference link so a later set_reference/query is consistent.
        self.reference = self.mean if ref_is_mean else self.reference.to(device=device, dtype=dtype)
        return self

    def set_reference(self, reference: torch.Tensor) -> "PCALatentBasis":
        """Express coefficients as deltas around ``reference`` (e.g. the reconstructed
        field lambda^0, shape ``(n_cp, D)``) instead of the training mean. The reference is
        a fixed origin, so it is detached (never carries gradient)."""
        self.reference = reference.detach().to(
            device=self.components.device, dtype=self.components.dtype
        )
        return self

    def to_coeff(self, z: torch.Tensor) -> torch.Tensor:
        """``(n_cp, D)`` latent control points -> ``(n_cp, k)`` *whitened* coefficients
        (relative to ``self.reference``)."""
        return ((z - self.reference) @ self.components) / self.scale

    def to_latent(self, c: torch.Tensor) -> torch.Tensor:
        """``(n_cp, k)`` whitened coefficients -> ``(n_cp, D)`` latent control points
        (added onto ``self.reference``)."""
        return self.reference + (c * self.scale) @ self.components.T

    def project_grad(self, dz: torch.Tensor) -> torch.Tensor:
        """Project a latent-space gradient ``(n_cp, D)`` to whitened-coefficient space.

        Chain rule through ``z = mean + (c * scale) @ V_k.T`` gives
        ``dJ/dc = (dJ/dz @ V_k) * scale``. Returns the flattened ``(n_cp * k,)`` gradient
        so it lines up with the flattened MMA design vector.
        """
        dz = dz.reshape(-1, self.latent_dim)
        return ((dz @ self.components) * self.scale).reshape(-1)

    def save(self, path) -> None:
        torch.save(
            {"mean": self.mean.detach().cpu(),
             "components": self.components.detach().cpu(),
             "scale": self.scale.detach().cpu()},
            path,
        )

    @classmethod
    def load(cls, path, device="cpu") -> "PCALatentBasis":
        data = torch.load(path, map_location=device, weights_only=False)
        scale = data.get("scale")
        return cls(
            data["mean"].to(device),
            data["components"].to(device),
            scale.to(device) if scale is not None else None,
        )


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
    mean, components, _, scale = compute_latent_pca(latents, k)
    basis = PCALatentBasis(mean, components, scale)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        basis.save(cache_path)
        logger.info("Saved PCA basis to %s", cache_path)
    return basis

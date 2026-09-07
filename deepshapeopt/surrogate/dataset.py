"""Flow-field dataset for Transolver training.

Each item is one ``sample_SSSSS_V.npz`` produced by
``scripts/generate_flow_dataset.py``. Variable point counts -> use
``batch_size=1`` with the identity collate (the ShapeNetCar practice).

Features ``x = [pos(3), sdf(1), normal(3)]`` and targets
``y = [U(3), p(1), tau_w(3)]`` are standardized channel-wise with dataset
statistics computed over the training split (:func:`compute_norm_stats`).
``tau_w`` (OpenFOAM ``wallShearStress``) is defined on the wall points only;
off-wall rows carry zero padding that is excluded from the statistics and
from every loss/metric. A 4-channel checkpoint (``[U, p]``, probe-shell FD
for the viscous drag) keeps working: the dataset slices ``y`` to the
normalizer's channel count.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .query_points import ROLE_SURFACE, QueryCloud

logger = logging.getLogger(__name__)

Y_CHANNELS_4 = ["Ux", "Uy", "Uz", "p"]
Y_CHANNELS_7 = Y_CHANNELS_4 + ["tau_x", "tau_y", "tau_z"]


def load_sample(path: Path) -> dict:
    with np.load(path) as z:
        x = np.concatenate([z["pos"], z["sdf"][:, None], z["normal"]], axis=1)
        tau_w = z["tau_w"].astype(np.float32)
        n = z["U"].shape[0]
        tau_full = np.zeros((n, 3), dtype=np.float32)
        tau_full[: tau_w.shape[0]] = tau_w
        y = np.concatenate([z["U"], z["p"][:, None], tau_full], axis=1)
        return {
            "x": x.astype(np.float32),
            "y": y.astype(np.float32),
            "role": z["role"].astype(np.int8),
            "valid": z["valid"].astype(bool),
            "n_surface": int(z["n_surface"]),
            "delta": z["delta"].astype(np.float32),
            "area_normals": z["area_normals"].astype(np.float32),
            "tau_w": tau_w,
            "tau_missing": bool(not np.any(tau_w)),
            "drag_foam": float(z["drag_foam"]),
            "meta": json.loads(str(z["meta"])),
            "path": str(path),
        }


def tau_missing(path: Path) -> bool:
    """True when the stored wall shear stress is the all-zero fallback."""
    with np.load(path) as z:
        return bool(not np.any(z["tau_w"]))


def compute_norm_stats(files: list[Path], with_tau: bool = False) -> dict:
    """Channel-wise mean/std of features and targets over ``files``.

    ``[U, p]`` statistics use every valid point; the ``tau_w`` statistics
    (only with ``with_tau``) use the wall rows only, never the zero padding.
    Returned lists are JSON-serializable and stored both in
    ``dataset_stats.json`` and inside training checkpoints.
    """
    sx = sx2 = sy = sy2 = None
    st = st2 = None
    n = n_wall = 0
    for f in files:
        s = load_sample(Path(f))
        m = s["valid"]
        x, y = s["x"][m].astype(np.float64), s["y"][m][:, :4].astype(np.float64)
        if sx is None:
            sx, sx2 = x.sum(0), (x**2).sum(0)
            sy, sy2 = y.sum(0), (y**2).sum(0)
        else:
            sx += x.sum(0)
            sx2 += (x**2).sum(0)
            sy += y.sum(0)
            sy2 += (y**2).sum(0)
        n += len(x)
        if with_tau:
            wall = (s["role"] == ROLE_SURFACE) & m
            t = s["y"][wall][:, 4:7].astype(np.float64)
            if st is None:
                st, st2 = t.sum(0), (t**2).sum(0)
            else:
                st += t.sum(0)
                st2 += (t**2).sum(0)
            n_wall += len(t)
    mean_x, mean_y = sx / n, sy / n
    std_x = np.sqrt(np.maximum(sx2 / n - mean_x**2, 1e-12))
    std_y = np.sqrt(np.maximum(sy2 / n - mean_y**2, 1e-12))
    stats = {
        "mean_x": mean_x.tolist(),
        "std_x": std_x.tolist(),
        "mean_y": mean_y.tolist(),
        "std_y": std_y.tolist(),
        "y_channels": list(Y_CHANNELS_4),
        "n_points": int(n),
        "n_files": len(files),
    }
    if with_tau:
        mean_t = st / n_wall
        std_t = np.sqrt(np.maximum(st2 / n_wall - mean_t**2, 1e-12))
        stats["mean_y"] += mean_t.tolist()
        stats["std_y"] += std_t.tolist()
        stats["y_channels"] = list(Y_CHANNELS_7)
        stats["n_wall_points"] = int(n_wall)
    return stats


class Normalizer:
    """Applies / inverts the channel-wise standardization on torch tensors."""

    def __init__(self, stats: dict, device=None, dtype=torch.float32):
        self.mean_x = torch.tensor(stats["mean_x"], device=device, dtype=dtype)
        self.std_x = torch.tensor(stats["std_x"], device=device, dtype=dtype)
        self.mean_y = torch.tensor(stats["mean_y"], device=device, dtype=dtype)
        self.std_y = torch.tensor(stats["std_y"], device=device, dtype=dtype)
        self.stats = stats
        self.n_y = len(stats["mean_y"])

    def to(self, device):
        for name in ("mean_x", "std_x", "mean_y", "std_y"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def norm_x(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean_x) / (self.std_x + 1e-8)

    def norm_y(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.mean_y) / (self.std_y + 1e-8)

    def denorm_y(self, y: torch.Tensor) -> torch.Tensor:
        return y * (self.std_y + 1e-8) + self.mean_y


class FlowFieldDataset(Dataset):
    """Per-shape flow samples; returns normalized torch tensors.

    ``y`` is sliced to the normalizer's channel count, so 4-channel
    statistics/checkpoints see exactly the ``[U, p]`` targets they were
    trained on.
    """

    def __init__(self, files: list, stats: dict):
        self.files = [Path(f) for f in files]
        self.norm = Normalizer(stats)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int) -> dict:
        s = load_sample(self.files[i])
        x = torch.from_numpy(s["x"])
        y = torch.from_numpy(s["y"])[:, : self.norm.n_y]
        return {
            "x": self.norm.norm_x(x),
            "y": self.norm.norm_y(y),
            "y_raw": y,
            "role": torch.from_numpy(s["role"].astype(np.int64)),
            "valid": torch.from_numpy(s["valid"]),
            "n_surface": s["n_surface"],
            "delta": torch.from_numpy(s["delta"]),
            "area_normals": torch.from_numpy(s["area_normals"]),
            "tau_missing": s["tau_missing"],
            "drag_foam": s["drag_foam"],
            "family": s["meta"]["family"],
            "path": s["path"],
        }


def cloud_from_batch(batch: dict) -> QueryCloud:
    """Rebuild the query cloud of a dataset item (device follows the batch).

    The single canonical version used by the training metric, the evaluation
    and the bias calibration; ``feats`` may be the normalized features, the
    drag integral only reads ``n_surface``, normals, areas and ``delta``.
    """
    an = batch["area_normals"]
    return QueryCloud(
        feats=batch["x"],
        roles=batch["role"].to(torch.int8),
        n_surface=int(batch["n_surface"]),
        unit_normals=torch.nn.functional.normalize(an, dim=1),
        area_normals=an,
        delta=batch["delta"],
    )


def split_files(
    files: list[Path],
    val_fraction: float = 0.1,
    holdout_family: str | None = None,
    seed: int = 0,
    min_surface_points: int = 0,
    require_tau: bool = False,
) -> tuple[list[Path], list[Path]]:
    """Random split plus (optionally) an entire held-out geometry family.

    ``min_surface_points`` drops degenerate samples (jitter-collapsed shapes
    with a handful of wall points) from both splits; ``require_tau`` drops
    samples whose stored wall shear stress is the all-zero fallback.
    """
    files = sorted(Path(f) for f in files)
    # Trajectory-harvested samples (traj_*.npz) go to training only: they
    # come from surrogate-driven optimization runs and must never leak into
    # the validation metrics.
    forced_train = [f for f in files if f.name.startswith("traj_")]
    files = [f for f in files if not f.name.startswith("traj_")]
    if min_surface_points > 0:
        forced_train = [
            f for f in forced_train if int(np.load(f)["n_surface"]) >= min_surface_points
        ]
        kept = [f for f in files if int(np.load(f)["n_surface"]) >= min_surface_points]
        if len(kept) < len(files):
            logger.warning(
                "split_files: dropped %d degenerate samples (< %d wall points)",
                len(files) - len(kept), min_surface_points,
            )
        files = kept
    if require_tau:
        n_before = len(files) + len(forced_train)
        forced_train = [f for f in forced_train if not tau_missing(f)]
        files = [f for f in files if not tau_missing(f)]
        n_dropped = n_before - len(files) - len(forced_train)
        logger.log(
            logging.WARNING if n_dropped else logging.INFO,
            "split_files: dropped %d samples without wallShearStress", n_dropped,
        )
    if holdout_family:
        fam = {f: json.loads(str(np.load(f)["meta"]))["family"] for f in files}
        held = [f for f in files if fam[f] == holdout_family]
        rest = [f for f in files if fam[f] != holdout_family]
    else:
        held, rest = [], list(files)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(rest))
    n_val = max(1, int(round(val_fraction * len(rest)))) if rest else 0
    val = [rest[i] for i in perm[:n_val]] + held
    train = [rest[i] for i in perm[n_val:]] + forced_train
    return train, val


def calibrate_visc_scale(files: list, nu: float = 1.0, direction=(1.0, 0.0, 0.0)) -> dict:
    """Least-squares factor c minimizing |J_foam - (J_p + c * J_visc_FD)|.

    FD mode only. Computed from the STORED targets (not predictions):
    quantifies how much the one-sided probe-shell difference underestimates
    the wall gradient of the interpolation-smoothed velocity field. Stored in
    the checkpoint and applied by the predictor at inference.
    """
    from .drag import drag_from_fields as _dff

    num = den = 0.0
    per_sample = []
    for f in files:
        s = load_sample(Path(f))
        an = torch.from_numpy(s["area_normals"])
        cloud = QueryCloud(
            feats=torch.from_numpy(s["x"]),
            roles=torch.from_numpy(s["role"]),
            n_surface=s["n_surface"],
            unit_normals=torch.nn.functional.normalize(an, dim=1),
            area_normals=an,
            delta=torch.from_numpy(s["delta"]),
        )
        _, d = _dff(
            torch.from_numpy(s["y"][:, :3]), torch.from_numpy(s["y"][:, 3]),
            cloud, nu=nu, direction=direction,
        )
        target_visc = s["drag_foam"] - d["J_p"]
        num += target_visc * d["J_visc"]
        den += d["J_visc"] ** 2
        per_sample.append(target_visc / d["J_visc"] if abs(d["J_visc"]) > 1e-12 else float("nan"))
    c = num / max(den, 1e-12)
    arr = np.asarray(per_sample, dtype=float)
    return {"visc_scale": float(c), "per_sample_mean": float(np.nanmean(arr)),
            "per_sample_std": float(np.nanstd(arr)), "n": len(files)}

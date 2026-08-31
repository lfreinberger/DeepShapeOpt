"""Flow-field dataset for Transolver training.

Each item is one ``sample_SSSSS_V.npz`` produced by
``scripts/generate_flow_dataset.py``. Variable point counts -> use
``batch_size=1`` with the identity collate (the ShapeNetCar practice).

Features ``x = [pos(3), sdf(1), normal(3)]`` and targets ``y = [U(3), p(1)]``
are standardized channel-wise with dataset statistics computed over the
training split (:func:`compute_norm_stats`).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .query_points import ROLE_SURFACE


def load_sample(path: Path) -> dict:
    with np.load(path) as z:
        x = np.concatenate([z["pos"], z["sdf"][:, None], z["normal"]], axis=1)
        y = np.concatenate([z["U"], z["p"][:, None]], axis=1)
        return {
            "x": x.astype(np.float32),
            "y": y.astype(np.float32),
            "role": z["role"].astype(np.int8),
            "valid": z["valid"].astype(bool),
            "n_surface": int(z["n_surface"]),
            "delta": z["delta"].astype(np.float32),
            "area_normals": z["area_normals"].astype(np.float32),
            "tau_w": z["tau_w"].astype(np.float32),
            "drag_foam": float(z["drag_foam"]),
            "meta": json.loads(str(z["meta"])),
            "path": str(path),
        }


def compute_norm_stats(files: list[Path]) -> dict:
    """Channel-wise mean/std of features and targets over ``files``.

    Only valid points contribute. Returned lists are JSON-serializable and
    stored both in ``dataset_stats.json`` and inside training checkpoints.
    """
    sx = sx2 = sy = sy2 = None
    n = 0
    for f in files:
        s = load_sample(Path(f))
        m = s["valid"]
        x, y = s["x"][m].astype(np.float64), s["y"][m].astype(np.float64)
        if sx is None:
            sx, sx2 = x.sum(0), (x**2).sum(0)
            sy, sy2 = y.sum(0), (y**2).sum(0)
        else:
            sx += x.sum(0)
            sx2 += (x**2).sum(0)
            sy += y.sum(0)
            sy2 += (y**2).sum(0)
        n += len(x)
    mean_x, mean_y = sx / n, sy / n
    std_x = np.sqrt(np.maximum(sx2 / n - mean_x**2, 1e-12))
    std_y = np.sqrt(np.maximum(sy2 / n - mean_y**2, 1e-12))
    return {
        "mean_x": mean_x.tolist(),
        "std_x": std_x.tolist(),
        "mean_y": mean_y.tolist(),
        "std_y": std_y.tolist(),
        "n_points": int(n),
        "n_files": len(files),
    }


class Normalizer:
    """Applies / inverts the channel-wise standardization on torch tensors."""

    def __init__(self, stats: dict, device=None, dtype=torch.float32):
        self.mean_x = torch.tensor(stats["mean_x"], device=device, dtype=dtype)
        self.std_x = torch.tensor(stats["std_x"], device=device, dtype=dtype)
        self.mean_y = torch.tensor(stats["mean_y"], device=device, dtype=dtype)
        self.std_y = torch.tensor(stats["std_y"], device=device, dtype=dtype)
        self.stats = stats

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
    """Per-shape flow samples; returns normalized torch tensors."""

    def __init__(self, files: list, stats: dict):
        self.files = [Path(f) for f in files]
        self.norm = Normalizer(stats)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int) -> dict:
        s = load_sample(self.files[i])
        x = torch.from_numpy(s["x"])
        y = torch.from_numpy(s["y"])
        return {
            "x": self.norm.norm_x(x),
            "y": self.norm.norm_y(y),
            "y_raw": y,
            "role": torch.from_numpy(s["role"].astype(np.int64)),
            "valid": torch.from_numpy(s["valid"]),
            "n_surface": s["n_surface"],
            "delta": torch.from_numpy(s["delta"]),
            "area_normals": torch.from_numpy(s["area_normals"]),
            "drag_foam": s["drag_foam"],
            "family": s["meta"]["family"],
            "path": s["path"],
        }


def split_files(
    files: list[Path], val_fraction: float = 0.1, holdout_family: str | None = None, seed: int = 0
) -> tuple[list[Path], list[Path]]:
    """Random split plus (optionally) an entire held-out geometry family."""
    files = sorted(Path(f) for f in files)
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
    train = [rest[i] for i in perm[n_val:]]
    return train, val

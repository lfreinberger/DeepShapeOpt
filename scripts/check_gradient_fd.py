"""Gradient check: adjoint directional derivative vs finite differences of J.

Works on the samples of a noise-probe run (``optimization.noise_probe`` in the driver): the
design was evaluated at ``x_base + t_i*h*p`` along one direction ``p``, and every evaluation
stored J, the constraint and both adjoint gradients w.r.t. the latent parameters (locked
entries masked).

For every interior point the adjoint derivative along one probe step, ``dJ_i . (h p)``, is
compared with the central finite difference ``(J[i+1] - J[i-1]) / 2``. The summary compares
the mean adjoint derivative with the slope of a least-squares line through all samples. The
fit averages out evaluation noise, and for a quadratic J its slope is the derivative at the
centre point.

Reading the result:
    ratio ~ 1                  adjoint gradient consistent with J along p
    ratio far from 1 or < 0    the gradient does not describe J there
    "unresolved"               J barely changes along p compared to its noise: increase h

One random direction only tests one projection of the gradient.

Usage, from an application repo root (each argument is the ``optimization`` directory of a
probe run or any directory above it, e.g. the results folder):

    uv run ../DeepShapeOpt/scripts/check_gradient_fd.py \\
        experiments/optimization/cylinder/results_cos_p2_noise_probe_start \\
        experiments/optimization/cylinder/results_cos_p2_noise_probe
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

SAMPLES = "noise_probe_samples.npz"
CONSISTENT = (0.8, 1.25)  # accepted adjoint/FD ratio
MIN_SIGNIFICANCE = 3.0     # |FD slope| / its standard error


def find_sample_files(path: Path) -> list[Path]:
    if (path / SAMPLES).is_file():
        return [path / SAMPLES]
    files = sorted(path.glob(f"**/{SAMPLES}"))
    if not files:
        raise FileNotFoundError(f"no {SAMPLES} at or below {path}")
    return files


def probe_direction(samples, opt_dir: Path) -> tuple[np.ndarray, str]:
    """Flat probe direction p (max-norm 1).

    Newer probe runs store it. For older runs it is rebuilt from the seed exactly like the
    driver does, except for the max-norm scale: the driver zeroed the LOCKED entries before
    scaling, and the lock mask is not stored. Entries whose gradient is zero everywhere stand
    in for it; they do not change dJ.p, only the scale can differ by a few percent.
    """
    if "direction" in samples.files:
        return samples["direction"].astype(float), "stored"

    probe_cfg = json.loads((opt_dir / "config_log.json").read_text())["optimization"]["noise_probe"]
    dtype = torch.float32
    start = Path(probe_cfg["start_parameters"])
    if not start.is_absolute():
        start = opt_dir.parents[2] / start  # optimization/ -> setup/ -> results/ -> experiment
    if start.is_file():
        loaded = torch.load(start, map_location="cpu")
        dtype = (loaded[0] if isinstance(loaded, (list, tuple)) else loaded).dtype

    n_vars = samples["dJ"].shape[1]
    gen = torch.Generator().manual_seed(int(probe_cfg.get("seed", 0)))
    p = torch.randn(n_vars, generator=gen, dtype=dtype).numpy().astype(float)
    active = np.any(samples["dJ"] != 0.0, axis=0)
    if samples["dc"].size:
        active |= np.any(samples["dc"] != 0.0, axis=0)
    p[~active] = 0.0
    return p / np.abs(p).max(), "rebuilt from seed (scale approx.)"


def check(values: np.ndarray, grads: np.ndarray, t: np.ndarray, step: np.ndarray, label: str):
    adjoint = grads @ step  # adjoint derivative per probe step at every point

    # Least-squares slope in units of one probe step, with its standard error from the
    # residuals of a quadratic fit (so curvature is not mistaken for noise).
    slope = np.polyfit(t, values, 1)[0]
    resid = values - np.polyval(np.polyfit(t, values, 2), t)
    dof = max(len(t) - 3, 1)
    se = np.sqrt(np.sum(resid ** 2) / dof) / np.sqrt(np.sum((t - t.mean()) ** 2))
    significance = abs(slope) / se if se > 0 else np.inf
    ratio = adjoint.mean() / slope if slope != 0 else np.nan

    # Pointwise comparison for a curved J along p: regress the adjoint derivative on the
    # central FD over the interior points (slope through the origin). When the derivative
    # changes sign inside the probe range the least-squares slope above is ~0 and its
    # ratio meaningless; the regression uses the variation of the derivative instead.
    fd = 0.5 * (values[2:] - values[:-2])
    adj_in = adjoint[1:-1]
    ratio_reg = float(adj_in @ fd / (fd @ fd)) if fd @ fd > 0 else np.nan
    resid_reg = adj_in - ratio_reg * fd
    r2 = 1.0 - float(resid_reg @ resid_reg) / float(((adj_in - adj_in.mean()) ** 2).sum() or np.inf)
    fd_range_sig = (fd.max() - fd.min()) / (np.sqrt(2.0) * se * np.sqrt(np.sum((t - t.mean()) ** 2)) / 1.0) if se > 0 else np.inf
    curved = fd_range_sig > 3.0 * MIN_SIGNIFICANCE and abs(fd.max() - fd.min()) > abs(fd.mean())

    if curved:
        ratio_used, method = ratio_reg, "pointwise regression (J curved along p)"
        resolved = True
    else:
        ratio_used, method = ratio, "fit slope"
        resolved = significance >= MIN_SIGNIFICANCE
    if not resolved:
        verdict = "unresolved (increase h)"
    elif CONSISTENT[0] <= ratio_used <= CONSISTENT[1]:
        verdict = "consistent"
    else:
        verdict = "INCONSISTENT"

    print(f"  {label}: mean value {values.mean():.4e}, |mean gradient| "
          f"{np.linalg.norm(grads.mean(axis=0)):.3e}")
    print(f"    adjoint derivative {adjoint.mean():+.3e} | FD slope {slope:+.3e} "
          f"(significance {significance:.1f}) | fit ratio {ratio:+.3f} | pointwise "
          f"regression ratio {ratio_reg:+.3f} (R^2 {r2:.2f})")
    print(f"    -> {verdict} [{method}]")
    print("      i   adjoint        central FD     ratio")
    for i in range(1, len(values) - 1):
        fd = 0.5 * (values[i + 1] - values[i - 1])
        r = adjoint[i] / fd if fd != 0 else np.nan
        print(f"      {i}   {adjoint[i]:+.3e}     {fd:+.3e}     {r:+.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="+", type=Path,
                        help="probe run directories (optimization dir or a parent of it)")
    args = parser.parse_args()

    for path in args.paths:
        for file in find_sample_files(path):
            opt_dir = file.parent
            samples = np.load(file)
            h = float(samples["h"])
            t = np.asarray(samples["offsets"], dtype=float)
            p, source = probe_direction(samples, opt_dir)
            print(f"\n== {opt_dir}")
            print(f"   h={h}, {len(t)} points, direction: {source}")
            if len(t) < 4:
                print("   too few points for a check")
                continue
            check(samples["J"], samples["dJ"], t, h * p, "objective")
            if samples["con"].size:
                check(samples["con"], samples["dc"], t, h * p, "constraint")


if __name__ == "__main__":
    main()

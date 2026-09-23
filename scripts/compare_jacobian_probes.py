"""Compare two (or more) jacobian_probe.npz files: spectrum, column norms and the principal
angles between the dominant right singular subspaces -- how much the geometric map of the
parametrization changed between the designs (``deepshapeopt.latent_metric``).

    uv run ../DeepShapeOpt/scripts/compare_jacobian_probes.py <results_a> <results_b> [...]
"""

import argparse
from pathlib import Path

import numpy as np

from deepshapeopt import latent_metric as lm

NAME = "jacobian_probe.npz"


def find(path: Path) -> Path:
    if path.name == NAME:
        return path
    files = sorted(path.glob(f"**/{NAME}"))
    if len(files) != 1:
        raise FileNotFoundError(f"expected one {NAME} below {path}, found {len(files)}")
    return files[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    probes = [(p, np.load(find(p))) for p in args.paths]
    for p, z in probes:
        s = z["sigma"]; col = z["col_norm"]; nz = col > 0
        print(f"\n=== {p} ({z['start_parameters']})")
        print(f"  {len(z['surface_ids'])} surface points, {int(z['mask_free'].sum())} free variables, "
              f"area {float(z['area_total']):.1f} mm^2, max_step {float(z['max_step']):g}")
        print("  sigma_max %.3e; above 1e-1/1e-2/1e-3/1e-4/1e-6: %s" % (
            s[0], " / ".join(str(int((s / s[0] > t).sum())) for t in (1e-1, 1e-2, 1e-3, 1e-4, 1e-6))))
        print("  column norms q05/q50/q95/max: %.3e / %.3e / %.3e / %.3e" % tuple(
            np.quantile(col[nz], q) for q in (0.05, 0.5, 0.95, 1.0)))
        cm = z["cos_max"]
        print("  max column cosine: median %.3f, q90 %.3f, fraction > 0.9: %.1f%%, > 0.99: %.1f%%" % (
            np.nanmedian(cm), np.nanquantile(cm, 0.9), 100 * np.nanmean(cm > 0.9), 100 * np.nanmean(cm > 0.99)))
    if len(probes) >= 2:
        (pa, a), (pb, b) = probes[0], probes[1]
        print(f"\n=== {pa.name} -> {pb.name}")
        for line in lm.compare_probes(a, b):
            print("  " + line)


if __name__ == "__main__":
    main()

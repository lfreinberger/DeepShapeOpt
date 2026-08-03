"""Design-variable counts per tiling (reviewer editorial item: Table 3 column).

Counts are obtained by actually building the latent-field spline with the
production function ``build_parameter_spline`` and reading its control-point
count, so the numbers cannot drift from the code that produced the manuscript.

The closed form implied by that function is

    n_design = latent_dim * prod_i (tiling_i + degree_i)

because the clamped knot vector starts with ``degree_i + 1`` control points per
axis and ``insert_knots`` adds ``tiling_i - 1`` more.

Usage:
    uv run python scripts/revision/design_var_counts.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from deepshapeopt.reconstruction import build_parameter_spline

REPO = Path(__file__).resolve().parents[2]

# (label, tiling, degree, latent_dim) for every configuration the manuscript reports.
CASES = [
    ("Flow channel", [1, 2, 2], [1, 1, 1], 32),
    ("Flow channel", [1, 4, 4], [1, 1, 1], 32),
    ("Flow channel", [1, 6, 6], [1, 1, 1], 32),
    ("Flow channel", [1, 8, 8], [1, 1, 1], 32),
    ("Flow channel", [1, 10, 10], [1, 1, 1], 32),
    ("Flow channel", [1, 12, 12], [1, 1, 1], 32),
    ("Propeller", [12, 12, 4], [1, 1, 1], 32),
    ("Rim", [6, 12, 12], [1, 1, 1], 32),
    ("Dog", [4, 12, 12], [1, 1, 1], 32),
    ("Cube (Exp. 1-3)", [2, 2, 2], [1, 1, 1], 32),
    # Reviewer 4 suggests a higher-degree latent field; report its cost.
    ("Flow channel, p=2", [1, 8, 8], [2, 2, 2], 32),
    ("Cube, p=2", [2, 2, 2], [2, 2, 2], 32),
]


def n_control_points(tiling: list[int], degree: list[int], latent_dim: int) -> int:
    """Build the real spline and count its control points."""
    spline = build_parameter_spline(degree, tuple(tiling), latent_dim)
    return int(spline.control_points.shape[0])


def closed_form(tiling: list[int], degree: list[int]) -> int:
    return math.prod(t + p for t, p in zip(tiling, degree))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--latex", action="store_true", help="emit LaTeX table rows instead of a plain table"
    )
    parser.add_argument(
        "--json-out", type=Path, default=None, help="also write the results as JSON"
    )
    args = parser.parse_args()

    rows = []
    for label, tiling, degree, latent_dim in CASES:
        n_cp = n_control_points(tiling, degree, latent_dim)
        n_cp_formula = closed_form(tiling, degree)
        if n_cp != n_cp_formula:
            raise SystemExit(
                f"{label} {tiling}: spline gives {n_cp} control points, "
                f"closed form gives {n_cp_formula} -- the formula in the paper would be wrong"
            )
        rows.append(
            {
                "shape": label,
                "tiling": tiling,
                "degree": degree,
                "latent_dim": latent_dim,
                "n_control_points": n_cp,
                "n_design_variables": n_cp * latent_dim,
            }
        )

    if args.latex:
        for r in rows:
            til = r"$" + r" \times ".join(str(t) for t in r["tiling"]) + r"$"
            print(f"{r['shape']} & {r['latent_dim']} & {til} & {r['n_design_variables']} \\\\")
    else:
        print(f"{'shape':<20} {'tiling':<12} {'p':<10} {'d_lat':>5} {'n_cp':>7} {'n_design':>9}")
        print("-" * 68)
        for r in rows:
            til = "x".join(str(t) for t in r["tiling"])
            deg = "x".join(str(p) for p in r["degree"])
            print(
                f"{r['shape']:<20} {til:<12} {deg:<10} {r['latent_dim']:>5} "
                f"{r['n_control_points']:>7} {r['n_design_variables']:>9}"
            )

    # Cross-check against the one count the manuscript prints (864, Section 3.2.1).
    cube = next(r for r in rows if r["shape"] == "Cube (Exp. 1-3)")
    status = "OK" if cube["n_design_variables"] == 864 else "MISMATCH"
    print(f"\ncheck: cube 2x2x2, d_lat=32 -> {cube['n_design_variables']} design variables "
          f"(manuscript says 864) [{status}]")
    if status == "MISMATCH":
        raise SystemExit("cube count does not reproduce the manuscript -- do not use these numbers")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()

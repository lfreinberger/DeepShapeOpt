"""Mesh-study table using the manuscript's normalization (reviewer comment 8).

The manuscript reports each method's drag reduction relative to ITS OWN initial
geometry, and the two pipelines do not share one:

    latent-field run   J0 = 37.1499   (reconstructed surface)
    FFD runs           J0 = 36.8845   (CAD surface)

To reproduce those reductions at every refinement level, the initial geometry of
*each* pipeline must be evaluated on the *same* mesh as the corresponding final
design. Numerator and denominator are therefore always taken at matched level; a
baseline measured at one level is never combined with a final design at another.

At level 6 this must reproduce the published 16.71% and 15.69% exactly, which is
the check built in below.

Usage:
    uv run python scripts/revision/mesh_study_table.py
    uv run python scripts/revision/mesh_study_table.py --latex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "revision_artifacts"

# (label, initial geometry key, final geometry key, published reduction at L6)
PAIRS = [
    ("Neural SDF", "cyl_init", "cyl_final_neural", 16.71),
    ("FFD 7x7x7", "cyl_init_ffd", "cyl_final_ffd7", 15.69),
]
LEVELS = [5, 6, 7]


def load() -> dict:
    runs = {}
    for path in (ART / "mesh_study" / "mesh_study.json",
                 ART / "mesh_study_ffdinit" / "mesh_study.json"):
        if path.is_file():
            for r in json.loads(path.read_text()):
                runs[(r["geometry"], r["level"])] = r
    return runs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--json-out", type=Path, default=ART / "mesh_study_table.json")
    args = ap.parse_args()

    runs = load()
    missing = [(g, l) for _, gi, gf, _ in PAIRS for g in (gi, gf) for l in LEVELS
               if (g, l) not in runs]
    if missing:
        print("MISSING runs (table will be incomplete):")
        for g, l in sorted(set(missing)):
            print(f"   {g} at L{l}")
        print()

    rows, out = [], []
    for label, gi, gf, published in PAIRS:
        for lvl in LEVELS:
            ri, rf = runs.get((gi, lvl)), runs.get((gf, lvl))
            if not (ri and rf):
                continue
            red = 100.0 * (ri["drag"] - rf["drag"]) / ri["drag"]
            rows.append((label, lvl, ri["cells"], rf["cells"], ri["drag"], rf["drag"], red))
            out.append({"method": label, "level": lvl,
                        "cells_initial": ri["cells"], "cells_final": rf["cells"],
                        "drag_initial": ri["drag"], "drag_final": rf["drag"],
                        "reduction_pct": red,
                        "published_reduction_pct_L6": published if lvl == 6 else None})

    hdr = (f"{'method':<12}{'level':>6}{'cells init':>12}{'cells final':>13}"
           f"{'J_initial':>12}{'J_final':>11}{'reduction':>11}")
    print(hdr); print("-" * len(hdr))
    for label, lvl, ci, cf, ji, jf, red in rows:
        print(f"{label:<12}{lvl:>6}{ci:>12,}{cf:>13,}{ji:>12.4f}{jf:>11.4f}{red:>10.2f}%"
              .replace(",", " "))

    # Gate: level 6 must reproduce the published reductions.
    print()
    ok = True
    for label, gi, gf, published in PAIRS:
        r = next((x for x in rows if x[0] == label and x[1] == 6), None)
        if r is None:
            print(f"  {label:<12} L6 not available -- cannot check against the manuscript")
            ok = False
            continue
        good = abs(r[6] - published) < 0.005
        ok &= good
        print(f"  {label:<12} L6 recomputed {r[6]:.2f}%  vs published {published:.2f}%  "
              f"[{'OK' if good else 'MISMATCH'}]")
    if not ok:
        print("\n  NOTE: a mismatch here means the normalization convention differs from"
              "\n  the manuscript's -- resolve before using these numbers.")

    if args.latex:
        print("\n=== LaTeX ===")
        print(r"\begin{tabular}{llrrrrr}")
        print(r"\hline")
        print(r"Method & Level & Cells (init.) & Cells (final) & $J_{0}$ & $J_{\mathrm{final}}$"
              r" & Reduction \\")
        print(r"\hline")
        for label, lvl, ci, cf, ji, jf, red in rows:
            print(f"{label} & {lvl} & {ci:,} & {cf:,} & {ji:.4f} & {jf:.4f} & "
                  f"{red:.2f}\\% \\\\".replace(",", r"\,"))
        print(r"\hline")
        print(r"\end{tabular}")

    args.json_out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()

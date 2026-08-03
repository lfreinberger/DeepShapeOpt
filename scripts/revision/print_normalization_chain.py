"""Make the training/optimization normalization mapping explicit (reviewer comment 6).

The reviewer asks how the training normalization (Omega_box = [-1,1]^3) relates
to the optimization setup, whose design domain is not a unit cube.

There are TWO stages, and only the first is isotropic:

  Stage A  physical -> normalized      deepshapeopt/reconstruction.py:307
           fit_box_to_unit_cube: uniform scale by 2/L with L = max(extent),
           a single scalar for all three axes. The design box is therefore NOT
           stretched to fill [-1,1]^3; only its longest axis reaches +-1.

  Stage B  normalized -> decoder input DeepSDFStruct/lattice_structure.py:280
           transform(): each axis is independently rescaled so that every tile
           spans exactly [-1,1] in the decoder's input. This is PER-AXIS.

Net effect: Stage B undoes the isotropy Stage A established. The decoder is
queried on tiles whose physical aspect ratio is generally not 1:1:1, i.e.
slightly off the distribution it was trained on ([-1,1]^3 primitive scenes).

This script derives the numbers with the production functions and verifies the
tile mapping numerically. It is pure: no decoder is loaded, no file is written
by the library, CPU only.

Usage:
    uv run python scripts/revision/print_normalization_chain.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from DeepSDFStruct.lattice_structure import transform
from deepshapeopt.reconstruction import fit_box_to_unit_cube

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIGS = [
    REPO / "experiments/drag_cube/config_latent_cube.json",
    REPO / "experiments/drag_cube/config_latent_cube_with_cylinders.json",
    REPO / "experiments/drag_cube/config_latent_cube_with_holes.json",
    REPO / "experiments/reconstruction/feed_channel/config.json",
]
AXES = "xyz"


def analyse(cfg_path: Path) -> dict | None:
    cfg = json.loads(cfg_path.read_text())
    rec = cfg.get("reconstruction", {})
    if "tiling" not in rec:
        return {"config": str(cfg_path), "skipped": "no latent tiling (FFD configuration)"}
    if "design_domain" not in rec:
        # The standalone reconstruction path does not use fit_box_to_unit_cube at
        # all: it calls normalize_mesh_to_unit_cube on the mesh's own bounding
        # box (reconstruction.py:670) with box_constrained=False (:729). Stage A
        # therefore differs between the two pipelines; only Stage B is shared.
        return {
            "config": str(cfg_path),
            "skipped": "no design_domain -- standalone reconstruction path normalizes to the "
                       "mesh bounding box (normalize_mesh_to_unit_cube), not the design box",
        }

    domain = torch.tensor(rec["design_domain"], dtype=torch.float64)
    tiling = list(rec["tiling"])

    # --- Stage A -----------------------------------------------------------
    # NOTE: fit_box_to_unit_cube returns 1/scale as its first value (a length
    # L/2), despite the name; the factor actually applied by normalize_fn is
    # 2/L. Report the factor, not the returned length.
    inv_scale, center, norm_fn, _denorm_fn, box_norm = fit_box_to_unit_cube(domain)
    extent = (domain[1] - domain[0]).tolist()
    L = max(extent)
    factor_A = 2.0 / L

    # --- Stage B -----------------------------------------------------------
    # Per axis: the normalized tile width, and the factor that maps it to [-1,1].
    norm_extent = (box_norm[1] - box_norm[0]).tolist()
    tile_norm = [w / t for w, t in zip(norm_extent, tiling)]
    tile_phys = [e / t for e, t in zip(extent, tiling)]
    factor_B = [2.0 / w for w in tile_norm]
    aspect = [f / min(factor_B) for f in factor_B]

    # --- numerical verification of the tile mapping -------------------------
    # T is a triangular wave of period TWO tiles: with v = t * x_norm,
    #   v = 0 -> -1,  v = 0.5 -> 0,  v = 1 -> +1,  v = 1.5 -> 0,  v = 2 -> -1.
    # So successive tiles are mirrored, tile boundaries alternate -1/+1, and
    # tile midpoints map to 0. The sign of dT/dx flips at every integer v,
    # i.e. exactly at the tile boundaries -- these are the kink planes.
    expected = [(0.0, -1.0), (0.5, 0.0), (1.0, 1.0), (1.5, 0.0), (2.0, -1.0)]
    checks = []
    for i, t in enumerate(tiling):
        lo, hi = box_norm[0, i].item(), box_norm[1, i].item()
        span = hi - lo
        for v, want in expected:
            if v > t:  # that many tiles do not exist on this axis
                continue
            x = torch.tensor([lo + span * v / t], dtype=torch.float64)
            got = float(transform(x, t, [lo, hi]).item())
            checks.append(
                {"axis": AXES[i], "v": v, "expected": want, "got": got,
                 "ok": abs(got - want) < 1e-9}
            )

    return {
        "config": str(cfg_path.relative_to(REPO)),
        "tiling": tiling,
        "spline_degree": rec.get("spline_degree"),
        "design_domain": rec["design_domain"],
        "physical_extent": extent,
        "stage_A": {
            "reference_length_L": L,
            "factor": factor_A,
            "isotropic": True,
            "center": center.tolist(),
            "normalized_box": box_norm.tolist(),
            "returned_inv_scale": float(inv_scale),
        },
        "stage_B": {
            "normalized_tile_width": tile_norm,
            "physical_tile_size": tile_phys,
            "per_axis_factor": factor_B,
            "isotropic": max(aspect) - min(aspect) < 1e-9,
        },
        "decoder_sees_aspect_ratio": aspect,
        "tile_mapping_checks": checks,
    }


def report(a: dict) -> None:
    print(f"\n### {a['config']}")
    ext = a["physical_extent"]
    print(f"  design domain           {ext[0]:.4g} x {ext[1]:.4g} x {ext[2]:.4g}"
          f"   tiling {'x'.join(map(str, a['tiling']))}")

    sa = a["stage_A"]
    print(f"  Stage A (physical -> normalized)   uniform, factor 2/L = {sa['factor']:.6g} "
          f"(L = {sa['reference_length_L']:.4g})")
    lo, hi = sa["normalized_box"]
    print(f"      normalized box      "
          f"[{lo[0]:+.4f}, {lo[1]:+.4f}, {lo[2]:+.4f}] .. "
          f"[{hi[0]:+.4f}, {hi[1]:+.4f}, {hi[2]:+.4f}]")
    if any(abs(abs(v) - 1.0) > 1e-9 for v in hi):
        print("      -> only the longest axis reaches +-1; the box does NOT fill [-1,1]^3")

    sb = a["stage_B"]
    tp = sb["physical_tile_size"]
    fb = sb["per_axis_factor"]
    print(f"  Stage B (normalized -> decoder)    per-axis, factors "
          f"[{fb[0]:.4g}, {fb[1]:.4g}, {fb[2]:.4g}]")
    print(f"      one tile, physical  {tp[0]:.4g} x {tp[1]:.4g} x {tp[2]:.4g}"
          f"   -> each mapped to [-1,1]^3")

    tile_aspect = [s / min(tp) for s in tp]
    iso = max(tile_aspect) - min(tile_aspect) < 1e-9
    print(f"  => physical tile aspect {tile_aspect[0]:.4g} : {tile_aspect[1]:.4g} : "
          f"{tile_aspect[2]:.4g}   [{'isotropic' if iso else 'ANISOTROPIC'}]")
    if not iso:
        long_axis = AXES[max(range(3), key=lambda i: tile_aspect[i])]
        r = max(tile_aspect) / min(tile_aspect)
        print(f"     each tile is mapped onto the cube [-1,1]^3, so as seen by the decoder")
        print(f"     distances along {long_axis} are compressed by {r:.4g}x relative to the "
              f"other axes")
        print(f"     -> Stage B undoes the isotropy of Stage A; the decoder is queried "
              f"off its [-1,1]^3 training distribution")

    for c in a["tile_mapping_checks"]:
        if not c["ok"]:
            print(f"     WARNING axis {c['axis']}: T(v={c['v']}) = {c['got']:+.6f}, "
                  f"expected {c['expected']:+.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, nargs="*", default=DEFAULT_CONFIGS)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    print("Normalization chain: physical -> normalized -> decoder input")
    print("(Stage A uniform, Stage B per-axis; see module docstring for file refs)")

    results = []
    for cfg_path in args.config:
        if not cfg_path.is_file():
            print(f"\n### {cfg_path}  MISSING")
            continue
        a = analyse(cfg_path)
        if "skipped" in a:
            print(f"\n### {cfg_path.relative_to(REPO)}\n  skipped: {a['skipped']}")
            continue
        report(a)
        results.append(a)

    # All tile-mapping checks must hold, or the description above is wrong.
    bad = [(a["config"], c) for a in results for c in a["tile_mapping_checks"] if not c["ok"]]
    if bad:
        raise SystemExit(f"tile mapping does not behave as described: {bad}")
    n_checks = sum(len(a["tile_mapping_checks"]) for a in results)
    print(f"\nverified ({n_checks} checks over {len(results)} configurations): T is a "
          f"triangular wave of period two tiles,")
    print("  T(v=0)=-1, T(v=0.5)=0, T(v=1)=+1, T(v=1.5)=0, T(v=2)=-1  with v = tiling * x_norm.")
    print("  Successive tiles are mirrored; dT/dx flips sign at every integer v, i.e. at the")
    print("  tile boundaries -- these are the kink planes probed for reviewer comment 4.")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()

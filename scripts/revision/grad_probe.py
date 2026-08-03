"""Gradient behaviour of the neural SDF at tile interfaces (reviewer comment 4).

The reviewer observes that the coordinate transform T(x) is only C0, with
piecewise-constant derivatives at the voxel interfaces, and asks whether the
resulting kinks are visible in the extracted normals or the sensitivity field,
and whether an H1/eikonal loss or a higher-degree (p >= 2) latent field would
help.

Where the kinks are, analytically. With v = tiling * x_norm the transform
(DeepSDFStruct/lattice_structure.py:280) is a triangular wave of period two
tiles; torch.floor contributes zero gradient and torch.abs the sign, so

    dT/dx = +- 2 * tiling / (b1 - b0)   a.e.,

with the sign flipping at every integer v -- i.e. exactly at the tile
boundaries. Those planes are computed here, not searched for, and are verified
independently by scripts/revision/print_normalization_chain.py.

What this experiment can and cannot show. The kink lives in T(x), UPSTREAM of
the decoder. Neither a higher-degree latent field nor an eikonal penalty can
remove it; they can only reduce the magnitude of the jump by making lambda(x)
vary less abruptly near the interface. Moreover the shipped decoders were
trained with EikonalLambda = 0.0, so switching eikonal_lambda on here
regularizes only the latent codes during reconstruction -- it does NOT produce
an eikonal-trained decoder. Report both caveats with the numbers.

Design matrix (same geometry, same decoder, same tiling, fixed seed):

    A  degree [1,1,1]  eikonal 0.0   manuscript baseline
    B  degree [2,2,2]  eikonal 0.0
    C  degree [1,1,1]  eikonal 0.05
    D  degree [2,2,2]  eikonal 0.05

Usage:
    uv run python scripts/revision/grad_probe.py
    uv run python scripts/revision/grad_probe.py --variants A B
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import trimesh

from DeepSDFStruct.SDF import SDFfromDeepSDF, normalize_mesh_to_unit_cube
from DeepSDFStruct.lattice_structure import LatticeSDFStruct
from DeepSDFStruct.parametrization import SplineParametrization
from DeepSDFStruct.pretrained_models import get_model
from DeepSDFStruct.torch_spline import TorchScaling

from deepshapeopt.config import ExperimentSpecifications
from deepshapeopt.reconstruction import (
    build_parameter_spline,
    fit_lattice_to_sdf,
    init_spline_parameters,
)

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "experiments/reconstruction/feed_channel/config.json"
SEED = 0

VARIANTS = {
    "A": {"spline_degree": [1, 1, 1], "eikonal_lambda": 0.0},
    "B": {"spline_degree": [2, 2, 2], "eikonal_lambda": 0.0},
    "C": {"spline_degree": [1, 1, 1], "eikonal_lambda": 0.05},
    "D": {"spline_degree": [2, 2, 2], "eikonal_lambda": 0.05},
}


def kink_planes(bounds: torch.Tensor, tiling: list[int], axis: int) -> np.ndarray:
    """Tile-boundary coordinates on one axis -- where dT/dx flips sign."""
    lo = bounds[0, axis].item()
    hi = bounds[1, axis].item()
    return lo + (hi - lo) * np.arange(tiling[axis] + 1) / tiling[axis]


def build_and_fit(variant: str, out_dir: Path) -> tuple[LatticeSDFStruct, torch.Tensor, dict]:
    """Replicate reconstruct_shape's setup (reconstruction.py:668-732) for one variant."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    specs = ExperimentSpecifications(str(CONFIG))
    rec_cfg = dict(specs["reconstruction"])
    rec_cfg.update(VARIANTS[variant])

    device = rec_cfg["device"]
    mesh_orig = trimesh.load_mesh(str(rec_cfg["mesh_path"]))
    mesh, scale, shift = normalize_mesh_to_unit_cube(mesh_orig.copy())
    bounds = torch.tensor(mesh.bounds, device=device, dtype=torch.float32)
    TorchScaling(scale_factors=scale, translation=shift, bounds=bounds, device=device)

    model = get_model(rec_cfg["model_path"], checkpoint=rec_cfg["model_checkpoint"])
    sdf = SDFfromDeepSDF(model)
    latent_dim = model._trained_latent_vectors[0].shape[0]

    spline_sp = build_parameter_spline(
        spline_degrees=rec_cfg["spline_degree"],
        tiling=rec_cfg["tiling"],
        latent_dim=latent_dim,
        bounds=np.stack([bounds[0].cpu().numpy(), bounds[1].cpu().numpy()]),
    )
    param_spline = SplineParametrization(spline_sp, device=model.device)
    init_spline_parameters(param_spline, mean=0.0, std=0.001)

    lattice = LatticeSDFStruct(
        tiling=rec_cfg["tiling"], microtile=sdf, parametrization=param_spline, bounds=bounds
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    result = fit_lattice_to_sdf(
        lattice, mesh, bounds, rec_cfg,
        output_dir=out_dir, lightweight_output_dir=out_dir,
        save_vtp=False, box_constrained=False,
    )
    lattice.parametrization.set_param(result["params"][0])

    meta = {
        "variant": variant,
        "spline_degree": rec_cfg["spline_degree"],
        "eikonal_lambda": rec_cfg["eikonal_lambda"],
        "tiling": rec_cfg["tiling"],
        "n_control_points": int(spline_sp.control_points.shape[0]),
        "n_design_variables": int(spline_sp.control_points.shape[0]) * latent_dim,
        "final_reconstruction_loss": float(result["final_loss"]),
    }
    return lattice, bounds, meta


def grad_norm_at(lattice: LatticeSDFStruct, pts: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """|grad s| and s at the given points, via autograd (same pattern as the eikonal term)."""
    x = pts.clone().requires_grad_(True)
    s = lattice(x)
    g = torch.autograd.grad(s.sum(), x, create_graph=False)[0]
    return g.norm(dim=-1).detach().cpu().numpy(), s.detach().cpu().numpy().reshape(-1)


def line_probe(lattice, bounds, tiling, axis=1, n=2001, device="cuda") -> dict:
    """|grad s| along a line crossing several tile interfaces."""
    lo, hi = bounds[0, axis].item(), bounds[1, axis].item()
    t = torch.linspace(lo, hi, n, device=device)
    pts = torch.zeros(n, 3, device=device)
    for i in range(3):
        pts[:, i] = 0.5 * (bounds[0, i] + bounds[1, i])  # domain centre on the other axes
    pts[:, axis] = t
    gn, s = grad_norm_at(lattice, pts)
    return {
        "axis": axis,
        "coord": t.cpu().numpy().tolist(),
        "grad_norm": gn.tolist(),
        "sdf": s.tolist(),
        "kink_planes": kink_planes(bounds, tiling, axis).tolist(),
    }


def _grad_at(lattice, pts: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Full gradient vector and s at the given points."""
    x = pts.clone().requires_grad_(True)
    s = lattice(x)
    g = torch.autograd.grad(s.sum(), x, create_graph=False)[0]
    return g.detach().cpu().numpy(), s.detach().cpu().numpy().reshape(-1)


def jump_statistics(lattice, bounds, tiling, axis=1, n=64, band=0.02,
                    eps_frac=0.01, seed=0, device="cuda") -> dict:
    """Discontinuity of grad s ACROSS tile interfaces, against a mid-tile control.

    The C0 kink is a SIGN FLIP of dT/dx at the interface, so the right
    observable is the jump in grad s between points straddling the plane -- not
    a band-averaged eikonal defect, which mixes the kink in with the decoder's
    general failure to satisfy |grad s| = 1.

    Crucially the measurement is restricted to the near-surface band |s| < band.
    The training loss is clamped at delta = 0.1 (Eq. 13), so beyond that the
    decoder is unconstrained and its gradients carry no information; including
    those points produces large jitter that has nothing to do with the kink.

    Control: the same measurement on MID-TILE planes, where the transform is
    smooth. Interface vs. control is the falsifiable comparison.
    """
    rng = np.random.default_rng(seed)
    lo = bounds[0].cpu().numpy()
    hi = bounds[1].cpu().numpy()
    width = (hi[axis] - lo[axis]) / tiling[axis]
    eps = eps_frac * width

    planes_if = kink_planes(bounds, tiling, axis)[1:-1]      # interior interfaces only
    planes_ct = planes_if + 0.5 * width                       # mid-tile control planes
    planes_ct = planes_ct[planes_ct < hi[axis] - eps]
    if len(planes_if) == 0:
        return {"error": f"axis {axis} has tiling {tiling[axis]}, no interior interface"}

    def measure(planes: np.ndarray) -> dict:
        rel_jump, angle, kept = [], [], 0
        for plane in planes:
            # Random points on the plane, then straddle it along `axis`.
            pts = rng.uniform(lo, hi, size=(n * n, 3))
            pts[:, axis] = plane
            p = torch.tensor(pts, dtype=torch.float32, device=device)
            minus, plus = p.clone(), p.clone()
            minus[:, axis] -= eps
            plus[:, axis] += eps

            g_m, s_m = _grad_at(lattice, minus)
            g_p, s_p = _grad_at(lattice, plus)

            near = (np.abs(s_m) < band) & (np.abs(s_p) < band)
            if near.sum() == 0:
                continue
            kept += int(near.sum())
            nm, np_ = np.linalg.norm(g_m[near], axis=1), np.linalg.norm(g_p[near], axis=1)
            ok = (nm > 1e-8) & (np_ > 1e-8)
            rel_jump.append(np.abs(np_[ok] - nm[ok]) / (0.5 * (np_[ok] + nm[ok])))
            cos = np.sum(g_m[near][ok] * g_p[near][ok], axis=1) / (nm[ok] * np_[ok])
            angle.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))

        if not rel_jump:
            return {"n": 0}
        rel_jump = np.concatenate(rel_jump); angle = np.concatenate(angle)
        return {
            "n": kept,
            "median_rel_jump": float(np.median(rel_jump)),
            "p95_rel_jump": float(np.quantile(rel_jump, 0.95)),
            "median_normal_angle_deg": float(np.median(angle)),
            "p95_normal_angle_deg": float(np.quantile(angle, 0.95)),
        }

    return {
        "axis": axis, "band": band, "eps_frac": eps_frac,
        "n_interface_planes": int(len(planes_if)),
        "n_control_planes": int(len(planes_ct)),
        "interface": measure(planes_if),
        "control_midtile": measure(planes_ct),
    }


def plot(results: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    CLAMP = 0.1  # training-loss truncation distance delta (Eq. 13)
    YMAX = 4.0   # variant C spikes to ~14; clipping keeps A, B and D legible
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    # Shade where the decoder is UNCONSTRAINED by the clamped training loss.
    # Gradients there carry no information and must not be read as artifacts.
    # Union over variants: shade wherever AT LEAST ONE curve is outside the
    # trust region, so no curve is silently interpreted where it is untrained.
    ref = results[0]["line_probe"]
    coord = np.asarray(ref["coord"])
    untrusted = np.zeros_like(coord, dtype=bool)
    for r in results:
        untrusted |= np.abs(np.asarray(r["line_probe"]["sdf"])) > CLAMP
    for ax in axes:
        ax.fill_between(coord, 0, 1, where=untrusted, transform=ax.get_xaxis_transform(),
                        color="0.9", zorder=0, linewidth=0)

    for r in results:
        lp = r["line_probe"]
        axes[0].plot(lp["coord"], lp["grad_norm"], lw=1.0,
                     label=f"{r['meta']['variant']}: p={r['meta']['spline_degree'][0]}, "
                           f"eik={r['meta']['eikonal_lambda']}")
        axes[1].plot(lp["coord"], lp["sdf"], lw=1.0)
    for p in ref["kink_planes"]:
        for ax in axes:
            ax.axvline(p, color="tab:red", lw=0.7, ls="--", zorder=1, alpha=0.7)
    axes[0].axhline(1.0, color="k", lw=0.6, ls=":")
    axes[1].axhline(CLAMP, color="k", lw=0.6, ls=":")
    axes[1].axhline(-CLAMP, color="k", lw=0.6, ls=":")
    axes[0].set_ylim(0, YMAX)
    axes[0].set_ylabel(r"$|\nabla s|$")
    axes[1].set_ylabel(r"$s$")
    axes[1].set_xlabel("coordinate along probe line "
                       r"(red dashed = tile interfaces; grey = $|s|>\delta$ for some variant)")
    axes[0].legend(fontsize=8, loc="upper left", ncol=2)
    axes[0].set_title(r"Gradient magnitude across tile interfaces "
                      f"($|\\nabla s|$ clipped at {YMAX:g}; variant C peaks near 14)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def print_summary(results: list[dict]) -> None:
    print("\n=== jump in grad s across tile interfaces vs mid-tile control ===")
    print("(near-surface only, |s| < band; the kink is a sign flip of dT/dx, so it shows")
    print(" up in the DIRECTION of grad s -- the surface normal -- not its magnitude)")
    print(f"{'variant':<8}{'p':<4}{'eik':<7}{'rec.loss':>10}"
          f"{'|g| jump if':>12}{'ctl':>8}"
          f"{'angle if':>10}{'ctl':>8}{'ratio':>7}")
    for r in results:
        m, j = r["meta"], r["jump"]
        if "error" in j:
            continue
        i, c = j["interface"], j["control_midtile"]
        ratio = (i["median_normal_angle_deg"] / c["median_normal_angle_deg"]
                 if c["median_normal_angle_deg"] else float("nan"))
        print(f"{m['variant']:<8}{m['spline_degree'][0]:<4}{m['eikonal_lambda']:<7}"
              f"{m['final_reconstruction_loss']:>10.2e}"
              f"{i['median_rel_jump']:>12.4f}{c['median_rel_jump']:>8.4f}"
              f"{i['median_normal_angle_deg']:>9.2f}d{c['median_normal_angle_deg']:>7.2f}d"
              f"{ratio:>7.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS))
    parser.add_argument("--out-dir", type=Path,
                        default=REPO / "revision_artifacts" / "grad_probe")
    parser.add_argument("--grid-n", type=int, default=96)
    parser.add_argument("--replot", action="store_true",
                        help="re-render the figure from a previous run's JSON without "
                             "refitting (each variant costs ~2.5 min on GPU)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []

    if args.replot:
        cached = args.out_dir / "grad_probe.json"
        if not cached.is_file():
            raise SystemExit(f"--replot needs {cached}, which does not exist")
        results = json.loads(cached.read_text())
        plot(results, args.out_dir / "grad_across_interfaces.png")
        print(f"replotted {len(results)} variants from {cached}")
        print_summary(results)
        return

    for v in args.variants:
        print(f"\n=== variant {v}: {VARIANTS[v]} ===")
        lattice, bounds, meta = build_and_fit(v, args.out_dir / f"fit_{v}")
        tiling = meta["tiling"]
        print(f"  fitted, final loss {meta['final_reconstruction_loss']:.6e}, "
              f"{meta['n_design_variables']} design variables")

        lp = line_probe(lattice, bounds, tiling, axis=1, device=device)
        js = jump_statistics(lattice, bounds, tiling, axis=1, n=args.grid_n, device=device)
        results.append({"meta": meta, "line_probe": lp, "jump": js})

        if "error" in js:
            print(f"  jump statistics: {js['error']}")
        else:
            i, c = js["interface"], js["control_midtile"]
            print(f"  median relative jump in |grad s|:  interface {i['median_rel_jump']:.4f}"
                  f"   mid-tile control {c['median_rel_jump']:.4f}   (n={i['n']}/{c['n']})")
            print(f"  median normal-direction change:    interface "
                  f"{i['median_normal_angle_deg']:.2f} deg   mid-tile control "
                  f"{c['median_normal_angle_deg']:.2f} deg")

    plot(results, args.out_dir / "grad_across_interfaces.png")
    (args.out_dir / "grad_probe.json").write_text(json.dumps(results, indent=2))

    print_summary(results)
    print(f"\nwrote {args.out_dir}/grad_probe.json and grad_across_interfaces.png")


if __name__ == "__main__":
    main()

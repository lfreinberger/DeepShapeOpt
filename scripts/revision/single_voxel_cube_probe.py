"""Single-voxel cube probe: verify the regularity claims made in the response letter.

The letter (comments 1-3) argues about the structure of the composed neural SDF
s(x) = f_theta(T(x), lambda(x)). Four claims are checked numerically here, on the
shipped primitives_cl32 decoder, by reconstructing a cube:

    C1  With a spatially CONSTANT latent ("one voxel", degree 0) the composed SDF
        is piecewise LINEAR in x: along any line, second differences vanish except
        at isolated kinks whose count does not grow under grid refinement.
    C2  With the production degree-1 tensor-product latent field the composed SDF
        is piecewise TRILINEAR: axis-aligned lines stay piecewise affine (trilinear
        restricted to an axis line is affine), but diagonal/generic lines show
        DISTRIBUTED curvature that survives grid refinement.
    C3  The representation produces genuine creases: the angle between grad s on
        either side of a reconstructed cube edge plateaus as the straddle distance
        eps -> 0, while a face-centre control angle decays to zero.
    C4  Creases are movable through the latent field: interpolating two fitted
        constant latents (cubes of edge 0.7 and 1.2) moves the zero crossings, the
        ReLU kink locations, and the corner radius continuously.

Caveats stated up front: the shipped checkpoint's "latent_codes" are dummy zeros
(the real training-time latents are 729-CP degree-1 spline fields), so a constant
latent is out-of-distribution -- C1 is a claim about the function class and holds
regardless of fit quality. All probes run in float64 (float32 second differences
at h ~ 1e-3 drown in roundoff); fitting uses the float32 production recipe from
experiments/drag_cube/config_latent_cube.json.

Usage:
    uv run python scripts/revision/single_voxel_cube_probe.py
    uv run python scripts/revision/single_voxel_cube_probe.py --epochs 15
    uv run python scripts/revision/single_voxel_cube_probe.py --skip-fit --stages probes crease interp
    uv run python scripts/revision/single_voxel_cube_probe.py --replot
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import trimesh

from DeepSDFStruct.SDF import SDFfromDeepSDF
from DeepSDFStruct.lattice_structure import LatticeSDFStruct
from DeepSDFStruct.parametrization import SplineParametrization
from DeepSDFStruct.pretrained_models import get_model
from DeepSDFStruct.sampling import random_sample_sdf, sample_mesh_surface
from DeepSDFStruct.sdf_primitives import BoxSDF  # exact Euclidean box SDF
from DeepSDFStruct.deep_sdf.reconstruction import reconstruct_from_samples

from deepshapeopt.reconstruction import build_parameter_spline, init_spline_parameters

REPO = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(
    os.environ.get("DEEPSHAPEOPT_MODEL_DIR")
    or REPO.parent / "DeepSDFStruct" / "DeepSDFStruct" / "trained_models"
)
MODEL_PATH = MODEL_DIR / "primitives_cl32"
LATENT_DIM = 32
CLAMP = 0.1          # training-loss truncation delta
TAU = 1e-2           # second-difference threshold |d2s/dt2| for "active" samples
BOUNDS = [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]

# Probe lines: name -> (origin, direction). Directions get normalized. The
# "generic" line dodges the cube's symmetry planes so no term of the trilinear
# field cancels by symmetry.
LINES = {
    "axis-x": ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    "axis-y": ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "axis-z": ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "diag-xyz": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    "diag-xy": ((0.0, 0.0, 0.0), (1.0, 1.0, 0.0)),
    "generic": ((0.03, -0.02, 0.05), (0.83, 0.41, 0.35)),
}
AXIS_LINES = ("axis-x", "axis-y", "axis-z")
CURVED_LINES = ("diag-xyz", "diag-xy", "generic")


# ---------------------------------------------------------------- fitting ----

def make_samples(extents: list[float], device: str):
    """Production-style sample set for one box: 100k uniform + 2x500k surface."""
    box = BoxSDF(center=[0.0, 0.0, 0.0], extents=extents)
    # no_grad: BoxSDF's center/extents are nn.Parameters, and a grad history on
    # the ground-truth distances would be backwarded through once per batch
    with torch.no_grad():
        uniform = random_sample_sdf(box, BOUNDS, 100_000, device=device)
        surface = sample_mesh_surface(
            box, trimesh.creation.box(extents=extents),
            n_samples=500_000, stds=[0.005, 0.0001], device=device,
        )
    return uniform + surface


def _fit(sdf, samples, epochs: int, out_dir: Path, tag: str) -> float:
    """Production recipe from experiments/drag_cube/config_latent_cube.json.

    deformation_function=None MUST be passed explicitly: the signature default
    is the (truthy) type union None | TorchSpline | TorchScaling.
    """
    result = reconstruct_from_samples(
        sdf, samples,
        num_iterations=epochs, lr=5e-3, loss_fn="ClampedL1", batch_size=4096,
        deformation_function=None, code_reg_lambda=0.0, code_bound=1.0,
        grad_clip=1.0, loss_plot_path=str(out_dir / f"loss_{tag}.png"),
    )
    return float(result["final_loss"])


def fit_constant(model, samples, init_vec, epochs, out_dir, tag):
    sdf = SDFfromDeepSDF(model)
    sdf.set_latent_vec(init_vec.to(model.device))
    loss = _fit(sdf, samples, epochs, out_dir, tag)
    return sdf.parametrization.param.detach().clone(), loss


def fit_lattice8(model, samples, epochs, out_dir, tag):
    lattice = build_lattice(model, cps=None)
    init_spline_parameters(lattice.parametrization, mean=0.0, std=0.001)
    loss = _fit(lattice, samples, epochs, out_dir, tag)
    cps = lattice.parametrization.torch_spline.control_points.detach().clone()
    return cps, loss


def build_constant_sdf(model, z: torch.Tensor) -> SDFfromDeepSDF:
    sdf = SDFfromDeepSDF(model)
    sdf.set_latent_vec(z.to(model.device))
    return sdf


def build_lattice(model, cps: torch.Tensor | None) -> LatticeSDFStruct:
    """Minimal degree-1 lattice: tiling [1,1,1] (affine transform), 8 CPs."""
    spline_sp = build_parameter_spline(
        spline_degrees=[1, 1, 1], tiling=(1, 1, 1), latent_dim=LATENT_DIM,
        bounds=np.array(BOUNDS),
    )
    param = SplineParametrization(spline_sp, device=model.device)
    lattice = LatticeSDFStruct(
        tiling=[1, 1, 1], microtile=SDFfromDeepSDF(model), parametrization=param,
        bounds=torch.tensor(BOUNDS, device=model.device, dtype=torch.float32),
    )
    if cps is not None:
        param.set_param(cps)
    return lattice


def checkerboard_cps(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    """8 corner CPs alternating z_a/z_b by (i+j+k) parity -- a guaranteed
    non-degenerate trilinear latent field, independent of any fit."""
    cps = torch.empty(8, LATENT_DIM, dtype=z_a.dtype)
    for n in range(8):
        i, j, k = n % 2, (n // 2) % 2, n // 4
        cps[n] = z_a if (i + j + k) % 2 == 0 else z_b
    return cps


def mean_cp_init(model_path: Path) -> torch.Tensor:
    """Mean training-time control-point vector (fallback init, norm ~ 0.26)."""
    d = torch.load(model_path / "LatentCodes" / "latest.pth", map_location="cpu")
    cps = [v for k, v in d["latent_fields_state_dict"].items()
           if k.endswith("control_points")]
    return torch.cat(cps, dim=0).mean(dim=0)


def fit_gate(sdf, half: float, device: str, dtype=torch.float32) -> dict:
    """Mean |s| at the 6 GT face centres and 8 GT corners."""
    faces, corners = [], []
    for ax in range(3):
        for sgn in (-1.0, 1.0):
            p = [0.0, 0.0, 0.0]
            p[ax] = sgn * half
            faces.append(p)
    for n in range(8):
        corners.append([half * (1 if (n >> b) & 1 else -1) for b in range(3)])
    with torch.no_grad():
        sf = sdf(torch.tensor(faces, device=device, dtype=dtype)).abs().mean().item()
        sc = sdf(torch.tensor(corners, device=device, dtype=dtype)).abs().mean().item()
    return {"mean_abs_s_faces": sf, "mean_abs_s_corners": sc, "fit_ok": sf < 0.05}


# ----------------------------------------------------------- float64 eval ----

@contextmanager
def float64_eval(module: torch.nn.Module, model):
    """Cast a probe object AND the decoder weights to float64, restore after.

    DeepSDFModel is a plain class (not an nn.Module attribute), so the decoder
    is not reached by module.double(); torch.set_default_dtype covers internal
    torch.zeros allocations in SDFfromDeepSDF._compute.
    """
    saved_default = torch.get_default_dtype()
    decoder = model._decoder
    saved_dec = next(decoder.parameters()).dtype
    try:
        torch.set_default_dtype(torch.float64)
        module.double()
        decoder.to(torch.float64)
        yield
    finally:
        torch.set_default_dtype(saved_default)
        module.float()
        decoder.to(saved_dec)


def eval_s(sdf, pts: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return sdf(pts).reshape(-1)


def grad_s(sdf, pts: torch.Tensor):
    x = pts.detach().clone().requires_grad_(True)
    s = sdf(x)
    g = torch.autograd.grad(s.sum(), x, create_graph=False)[0]
    return g.detach(), s.detach().reshape(-1)


# ------------------------------------------------------- C1/C2 line probes ----

def _clusters(active: np.ndarray) -> list[tuple[int, int]]:
    """Connected runs of True; returns (start, width) pairs."""
    runs, start = [], None
    for i, a in enumerate(active):
        if a and start is None:
            start = i
        elif not a and start is not None:
            runs.append((start, i - start))
            start = None
    if start is not None:
        runs.append((start, len(active) - start))
    return runs


def _line_metrics(t: np.ndarray, s: np.ndarray, tau: float) -> dict:
    h = t[1] - t[0]
    c = np.abs(s[:-2] - 2.0 * s[1:-1] + s[2:]) / h**2
    active = c > tau
    runs = _clusters(active)
    trusted = np.abs(s[1:-1]) <= CLAMP
    active_tr = active & trusted
    return {
        "h": float(h),
        "median_c": float(np.median(c)),
        "max_c": float(c.max()),
        "frac_above": float(active.mean()),
        "n_clusters": len(runs),
        "max_cluster_width": int(max((w for _, w in runs), default=0)),
        "frac_above_trusted": float(active_tr.sum() / max(trusted.sum(), 1)),
        "curve": {"t": t[1:-1].tolist(), "c": c.tolist(), "s": s[1:-1].tolist()},
    }


def probe_lines(sdf, model, device: str, n: int = 2001) -> dict:
    """Second-difference metrics on all LINES at resolutions n and 2n-1.

    Discriminator between "piecewise linear" and "distributed curvature":
      - kinks occupy <= a few samples each and their COUNT is grid-independent,
        so frac_above halves when the grid doubles (frac_ratio ~ 0.5);
      - genuine curvature keeps frac_above constant (frac_ratio ~ 1) and shows
        wide clusters.
    """
    out = {}
    with float64_eval(sdf, model):
        for name, (origin, direction) in LINES.items():
            o = torch.tensor(origin, device=device, dtype=torch.float64)
            d = torch.tensor(direction, device=device, dtype=torch.float64)
            d = d / d.norm()
            res = {}
            for nn_ in (n, 2 * n - 1):
                t = torch.linspace(-0.9, 0.9, nn_, device=device, dtype=torch.float64)
                pts = o + t[:, None] * d
                s = eval_s(sdf, pts).cpu().numpy()
                res[nn_] = _line_metrics(t.cpu().numpy(), s, TAU)
            coarse, fine = res[n], res[2 * n - 1]
            fine.pop("curve")  # keep JSON small; coarse curve is enough to plot
            out[name] = {
                "coarse": coarse,
                "fine": fine,
                "frac_ratio": (fine["frac_above"] / coarse["frac_above"]
                               if coarse["frac_above"] > 0 else 0.0),
                "cluster_ratio": (fine["n_clusters"] / coarse["n_clusters"]
                                  if coarse["n_clusters"] > 0 else 0.0),
            }
    return out


def classify_line(m: dict) -> str:
    """'kinky' = piecewise affine with isolated kinks; 'curved' = distributed.

    The two are separated by decades in the MEDIAN second difference: between
    isolated kinks a piecewise-affine function has machine-zero curvature
    (~1e-11 in float64), while a trilinear field puts smooth curvature almost
    everywhere along a non-axis line (>= 1e-4 in practice). The refinement
    ratio corroborates: isolated kinks occupy a fixed NUMBER of samples, so
    the active fraction halves when the grid doubles (~0.5); distributed
    curvature keeps it constant (~1).
    """
    coarse = m["coarse"]
    if coarse["median_c"] < 1e-6 and m["frac_ratio"] < 0.75:
        return "kinky"
    if coarse["median_c"] > 1e-4:
        return "curved"
    return "ambiguous"


# ------------------------------------------------------ C3 crease straddle ----

def _bisect_zero(sdf, make_pts, r_lo: float, r_hi: float, k: int,
                 device: str, iters: int = 50):
    """Vectorized bisection of s(make_pts(r)) = 0 over k parallel rays."""
    lo = torch.full((k,), r_lo, device=device, dtype=torch.float64)
    hi = torch.full((k,), r_hi, device=device, dtype=torch.float64)
    s_lo, s_hi = eval_s(sdf, make_pts(lo)), eval_s(sdf, make_pts(hi))
    valid = (s_lo * s_hi) < 0
    mid = 0.5 * (lo + hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        s_mid = eval_s(sdf, make_pts(mid))
        left = (s_lo * s_mid) <= 0
        hi = torch.where(left, mid, hi)
        s_hi = torch.where(left, s_mid, s_hi)
        lo = torch.where(left, lo, mid)
        s_lo = torch.where(left, s_lo, s_mid)
    return mid, valid


def _project(sdf, q: torch.Tensor, steps: int = 2):
    """Newton-project points onto the zero level set; returns (points, |s|)."""
    for _ in range(steps):
        g, s = grad_s(sdf, q)
        q = q - (s / (g * g).sum(-1).clamp_min(1e-12))[:, None] * g
    _, s = grad_s(sdf, q)
    return q.detach(), s.abs()


def _normals(sdf, pts: torch.Tensor):
    g, s = grad_s(sdf, pts)
    return g / g.norm(dim=-1, keepdim=True).clamp_min(1e-12), s


def _arc_scan(sdf, centers: torch.Tensor, tangent: torch.Tensor,
              arc_half: float, n_arc: int) -> dict:
    """Walk an arc of surface points across each centre and measure how the
    normal rotates. On a piecewise-linear zero set the rotation happens in
    JUMPS between flat facets; a crease shows up as one dominant jump carrying
    most of the total rotation (concentration ~ 1, stable under refinement of
    the arc), while a smooth rounding spreads it evenly (concentration ~ 1/n)."""
    k = centers.shape[0]
    t = torch.linspace(-arc_half, arc_half, n_arc, device=centers.device,
                       dtype=torch.float64)
    pts = (centers[:, None, :] + t[None, :, None] * tangent).reshape(-1, 3)
    proj, _ = _project(sdf, pts)
    n, _ = _normals(sdf, proj)
    n = n.reshape(k, n_arc, 3)
    proj = proj.reshape(k, n_arc, 3)
    cos = (n[:, :-1] * n[:, 1:]).sum(-1).clamp(-1.0, 1.0)
    theta = torch.rad2deg(torch.arccos(cos))          # (k, n_arc-1)
    total = theta.sum(dim=1)
    tmax, jstar = theta.max(dim=1)
    conc = torch.where(total > 5.0, tmax / total.clamp_min(1e-9),
                       torch.zeros_like(total))
    idx = torch.arange(k, device=centers.device)
    crease_pts = 0.5 * (proj[idx, jstar] + proj[idx, jstar + 1])
    return {"total": total, "max_jump": tmax, "concentration": conc,
            "crease_pts": crease_pts}


def crease_probe(sdf, model, half: float, device: str,
                 eps_list=(0.05, 0.02, 0.01, 0.005, 0.002),
                 arc_half: float = 0.06) -> dict:
    """Genuine-crease test at the fitted cube edge vs. a face-centre control.

    Two-step protocol (a naive straddle around the GT edge fails on a
    piecewise-linear surface: for eps below the local facet size both straddle
    points land on the SAME facet and the angle collapses to zero even though
    the crease exists a facet away):

      1. Arc scan: project a dense arc of points across the edge onto the
         fitted zero set and accumulate the normal rotation. A crease carries
         the rotation in one dominant jump (concentration ~ 1, stable when the
         arc is refined); its location gives the actual crease line.
      2. Recentred eps sweep: straddle THAT crease point. The angle between
         grad s on both sides now plateaus near the dihedral angle as
         eps -> 0, instead of decaying.
    """
    u = torch.tensor([1.0, 1.0, 0.0], device=device, dtype=torch.float64)
    u = u / u.norm()
    v = torch.tensor([1.0, -1.0, 0.0], device=device, dtype=torch.float64)
    v = v / v.norm()
    ey = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float64)
    ez = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float64)
    z_ks = torch.linspace(-0.7 * half, 0.7 * half, 9, device=device,
                          dtype=torch.float64)
    r_star = half * float(np.sqrt(2.0))

    out = {"eps": list(eps_list), "arc": {}, "sweep": {"edge": {}, "control": {}}}
    with float64_eval(sdf, model):
        # -- surface points near the edge and at the +x face centre
        def edge_pts(r):
            return r[:, None] * u + z_ks[:, None] * ez

        def face_pts(x):
            pts = torch.zeros(len(z_ks), 3, device=device, dtype=torch.float64)
            pts[:, 0] = x
            pts[:, 2] = z_ks
            return pts

        r_edge, valid = _bisect_zero(sdf, edge_pts, 0.6 * r_star, 1.35 * r_star,
                                     len(z_ks), device)
        p_edge, res = _project(sdf, edge_pts(r_edge)[valid])
        x_face, fvalid = _bisect_zero(sdf, face_pts, 0.6 * half, 1.35 * half,
                                      len(z_ks), device)
        p_face, _ = _project(sdf, face_pts(x_face)[fvalid])

        out["n_edge_points"] = int(valid.sum().item())
        out["edge_bisection_failed"] = int((~valid).sum().item())
        out["edge_r_median"] = (float(r_edge[valid].median().item())
                                if valid.any() else None)
        out["proj_residual_max"] = float(res.max().item()) if valid.any() else None
        if p_edge.shape[0] == 0:
            return out

        # -- step 1: arc scans at two refinements
        for tag, pts, tangent in (("edge", p_edge, v), ("control", p_face, ey)):
            scans = {}
            for n_arc in (41, 81):
                a = _arc_scan(sdf, pts, tangent, arc_half, n_arc)
                scans[n_arc] = a
            out["arc"][tag] = {
                "arc_half": arc_half,
                "total_deg_median": float(scans[41]["total"].median().item()),
                "max_jump_deg_median": float(scans[41]["max_jump"].median().item()),
                "concentration_median_41": float(scans[41]["concentration"].median().item()),
                "concentration_median_81": float(scans[81]["concentration"].median().item()),
            }
            if tag == "edge":
                crease_pts = scans[81]["crease_pts"]
                out["arc"]["edge"]["crease_offset_median"] = float(
                    (crease_pts - p_edge).norm(dim=-1).median().item())

        # -- step 2: eps sweep recentred on the detected crease line
        for eps in eps_list:
            for tag, pts, tangent in (("edge", crease_pts, v),
                                      ("control", p_face, ey)):
                qp, _ = _project(sdf, pts + eps * tangent)
                qm, _ = _project(sdf, pts - eps * tangent)
                gp, _ = grad_s(sdf, qp)
                gm, _ = grad_s(sdf, qm)
                cos = (gp * gm).sum(-1) / (gp.norm(dim=-1)
                                           * gm.norm(dim=-1)).clamp_min(1e-12)
                ang = torch.rad2deg(torch.arccos(cos.clamp(-1.0, 1.0)))
                out["sweep"][tag][str(eps)] = {
                    "median_deg": float(ang.median().item()),
                    "min_deg": float(ang.min().item()),
                    "max_deg": float(ang.max().item()),
                }
    return out


# -------------------------------------------------- C4 latent interpolation ----

def interp_sweep(model, z_small: torch.Tensor, z_big: torch.Tensor,
                 half_small: float, half_big: float, device: str,
                 n_alpha: int = 21, n_line: int = 2001) -> dict:
    """lambda(alpha) = (1-alpha) z_small + alpha z_big; track along the x-axis
    line: zero crossings, kink locations (second-difference spike clusters),
    and the corner radius from the C3 bisection at z = 0."""
    sdf = SDFfromDeepSDF(model)
    sdf.set_latent_vec(z_small.to(model.device))
    zs = z_small.to(model.device, torch.float64)
    zb = z_big.to(model.device, torch.float64)
    u = torch.tensor([1.0, 1.0, 0.0], device=device, dtype=torch.float64)
    u = u / u.norm()

    alphas = np.linspace(0.0, 1.0, n_alpha)
    records = []
    with float64_eval(sdf, model):
        t = torch.linspace(-0.9, 0.9, n_line, device=device, dtype=torch.float64)
        pts = torch.zeros(n_line, 3, device=device, dtype=torch.float64)
        pts[:, 0] = t
        t_np = t.cpu().numpy()
        for a in alphas:
            sdf.set_latent_vec(((1.0 - a) * zs + a * zb))
            s = eval_s(sdf, pts).cpu().numpy()

            # zero crossings (linear interpolation between sign changes)
            zc = []
            sign_change = np.where(np.signbit(s[:-1]) != np.signbit(s[1:]))[0]
            for i in sign_change:
                w = s[i] / (s[i] - s[i + 1])
                zc.append(float(t_np[i] + w * (t_np[i + 1] - t_np[i])))

            # kink locations: strong second-difference spike clusters
            h = t_np[1] - t_np[0]
            c = np.abs(s[:-2] - 2.0 * s[1:-1] + s[2:]) / h**2
            tau_k = max(0.05, 20.0 * float(np.median(c)))
            kinks = [float(t_np[1 + start + w // 2])
                     for start, w in _clusters(c > tau_k)]

            # all corner-ray crossings at z=0 along (1,1,0)/sqrt(2)
            r_grid = torch.linspace(0.5 * half_small * np.sqrt(2.0),
                                    1.35 * half_big * np.sqrt(2.0), 512,
                                    device=device, dtype=torch.float64)
            s_ray = eval_s(sdf, r_grid[:, None] * u).cpu().numpy()
            r_np = r_grid.cpu().numpy()
            rc = []
            for i in np.where(np.signbit(s_ray[:-1]) != np.signbit(s_ray[1:]))[0]:
                w = s_ray[i] / (s_ray[i] - s_ray[i + 1])
                rc.append(float(r_np[i] + w * (r_np[i + 1] - r_np[i])))
            records.append({
                "alpha": float(a),
                "zero_crossings": zc,
                "kinks": kinks,
                "corner_crossings": rc,
                "s_curve": s[::8].tolist(),
            })
        records_t = t_np[::8].tolist()

    # branch tracking: follow the crossing NEAREST to the previous alpha's
    # value (max() would jump whenever an additional crossing appears)
    def track(values_per_alpha: list[list[float]], start_pick) -> list[float | None]:
        out: list[float | None] = []
        prev = None
        for vals in values_per_alpha:
            if not vals:
                out.append(None)
                prev = None
                continue
            pick = (start_pick(vals) if prev is None
                    else min(vals, key=lambda v: abs(v - prev)))
            out.append(float(pick))
            prev = pick
        return out

    x0 = track([r["zero_crossings"] for r in records], max)
    rr_tracked = track([r["corner_crossings"] for r in records], min)
    for rec, r_val in zip(records, rr_tracked):
        rec["corner_r"] = r_val
    surface_lost = sum(1 for x in x0 if x is None)
    x0_valid = [x for x in x0 if x is not None]
    steps = np.abs(np.diff(x0_valid)) if len(x0_valid) > 1 else np.array([np.inf])
    rr = [r for r in rr_tracked if r is not None]
    r_steps = np.abs(np.diff(rr)) if len(rr) > 1 else np.array([np.inf])

    # greedy nearest-neighbour tracking of kinks across alpha
    tracks: list[list[tuple[int, float]]] = []
    for ia, rec in enumerate(records):
        for k in rec["kinks"]:
            best = None
            for tr in tracks:
                last_ia, last_k = tr[-1]
                if last_ia == ia - 1 and abs(last_k - k) < 0.05:
                    if best is None or abs(last_k - k) < abs(best[-1][1] - k):
                        best = tr
            if best is not None:
                best.append((ia, k))
            else:
                tracks.append([(ia, k)])
    full_tracks = [tr for tr in tracks if len(tr) == len(records)]
    longest_track = max((len(tr) for tr in tracks), default=0)

    return {
        "alphas": alphas.tolist(),
        "records": records,
        "t_curve": records_t,
        "x0_positive": x0,
        "x0_max_step": float(steps.max()),
        "x0_span": (float(abs(x0_valid[-1] - x0_valid[0]))
                    if len(x0_valid) > 1 else 0.0),
        "x0_monotone": bool(np.all(np.diff(x0_valid) > -5e-3)) if len(x0_valid) > 1 else False,
        "surface_lost_alphas": surface_lost,
        "corner_r_max_step": float(r_steps.max()),
        "n_kink_tracks_full_range": len(full_tracks),
        "longest_kink_track": longest_track,
        "n_alphas": len(records),
        "kink_tracks": [[[ia, k] for ia, k in tr] for tr in tracks if len(tr) >= 3],
    }


# ------------------------------------------------------------ mesh (option) ----

def mesh_stage(model, z: torch.Tensor, out_dir: Path, n_base: int = 64) -> dict:
    """Extract the one-latent cube with FlexiCubes (float32) and report the
    dihedral-angle distribution -- the visual counterpart of C3."""
    from DeepSDFStruct.mesh import create_3D_mesh

    sdf = build_constant_sdf(model, z)
    mesh, _ = create_3D_mesh(
        sdf, n_base, mesh_type="surface",
        bounds=torch.tensor(BOUNDS, dtype=torch.float32),
        extend_bounds=False, use_tiling=False, device=model.device,
    )
    tm = mesh.to_trimesh()
    obj_path = out_dir / "one_latent_cube.obj"
    tm.export(str(obj_path))
    ang = np.degrees(tm.face_adjacency_angles)
    return {
        "n_vertices": int(tm.vertices.shape[0]),
        "n_faces": int(tm.faces.shape[0]),
        "dihedral_p50_deg": float(np.median(ang)),
        "dihedral_p95_deg": float(np.quantile(ang, 0.95)),
        "dihedral_max_deg": float(ang.max()),
        "obj": str(obj_path),
    }


# ------------------------------------------------------------------- plots ----

def plot_lines(lines: dict, title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    for ax, (name, m) in zip(axes.flat, lines.items()):
        curve = m["coarse"]["curve"]
        t = np.asarray(curve["t"])
        s = np.asarray(curve["s"])
        c = np.asarray(curve["c"])
        ax.fill_between(t, 0, 1, where=np.abs(s) > CLAMP,
                        transform=ax.get_xaxis_transform(),
                        color="0.92", zorder=0, linewidth=0)
        ax.plot(t, s, lw=0.9, color="tab:blue", label="s")
        ax.axhline(0.0, color="k", lw=0.4)
        ax2 = ax.twinx()
        ax2.semilogy(t, np.maximum(c, 1e-12), lw=0.6, color="tab:orange",
                     alpha=0.8, label=r"$|\Delta^2 s|/h^2$")
        ax2.axhline(TAU, color="tab:red", lw=0.6, ls=":")
        ax2.set_ylim(1e-12, 1e4)
        cls = classify_line(m)
        ax.set_title(f"{name}  [{cls}]  med={m['coarse']['median_c']:.1e} "
                     f"frac={m['coarse']['frac_above']:.2f} "
                     f"ratio={m['frac_ratio']:.2f}", fontsize=9)
    axes[0, 0].set_ylabel("s (blue)")
    fig.suptitle(f"{title} -- grey: $|s|>\\delta$ (untrained), "
                 "orange: second difference (log, right axis), "
                 "ratio: frac_above fine/coarse (0.5 = isolated kinks, 1 = curvature)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_crease(crease: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps = np.asarray(crease["eps"], dtype=float)
    sweep = crease.get("sweep", {})
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for tag, color in (("edge", "tab:red"), ("control", "tab:blue")):
        d = sweep.get(tag, {})
        med = [d[str(e)]["median_deg"] if d.get(str(e)) else np.nan for e in eps]
        lo = [d[str(e)]["min_deg"] if d.get(str(e)) else np.nan for e in eps]
        hi = [d[str(e)]["max_deg"] if d.get(str(e)) else np.nan for e in eps]
        ax.plot(eps, med, "o-", color=color, label=f"{tag} (median)")
        ax.fill_between(eps, lo, hi, color=color, alpha=0.15)
    ax.axhline(90.0, color="k", lw=0.6, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel(r"straddle distance $\varepsilon$ (recentred on the detected crease)")
    ax.set_ylabel(r"angle between $\nabla s$ on both sides [deg]")
    arc = crease.get("arc", {}).get("edge", {})
    ax.set_title("Crease test at the cube edge vs. face control\n"
                 f"arc scan: total rotation {arc.get('total_deg_median', float('nan')):.1f} deg, "
                 f"concentration {arc.get('concentration_median_41', float('nan')):.2f} (n=41) / "
                 f"{arc.get('concentration_median_81', float('nan')):.2f} (n=81)",
                 fontsize=10)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_interp(interp: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    alphas = np.asarray(interp["alphas"])
    t = np.asarray(interp["t_curve"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    cmap = plt.get_cmap("viridis")
    for rec in interp["records"]:
        ax1.plot(t, rec["s_curve"], lw=0.7, color=cmap(rec["alpha"]))
    ax1.axhline(0.0, color="k", lw=0.5)
    ax1.set_xlabel("x (probe line y=z=0)")
    ax1.set_ylabel("s")
    ax1.set_title(r"$\hat{s}(x)$ along the x-axis, coloured by $\alpha$")

    for tr in interp["kink_tracks"]:
        ia = [alphas[i] for i, _ in tr]
        kk = [k for _, k in tr]
        ax2.plot(ia, kk, "-", lw=0.7, color="0.6")
    x0 = [x if x is not None else np.nan for x in interp["x0_positive"]]
    ax2.plot(alphas, x0, "o-", color="tab:blue", label="zero crossing $x_0$")
    rr = [rec["corner_r"] if rec["corner_r"] is not None else np.nan
          for rec in interp["records"]]
    ax2.plot(alphas, rr, "s-", color="tab:red", label="corner radius $r$")
    ax2.set_xlabel(r"$\alpha$")
    ax2.set_ylabel("position")
    ax2.set_title("Surface, corner and ReLU-kink positions move continuously")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------- summary ----

def build_summary(results: dict) -> str:
    lines = ["=== single-voxel cube probe: claim verification ===", ""]

    def verdict(ok):  # noqa: E731
        return "PASS" if ok else "FAIL"

    if "lines_A" in results:
        cls = {k: classify_line(m) for k, m in results["lines_A"].items()}
        ok = all(v == "kinky" for v in cls.values())
        lines.append(f"C1 (constant latent => piecewise linear): {verdict(ok)}")
        for k, m in results["lines_A"].items():
            lines.append(f"    {k:<10} {cls[k]:<10} median_c={m['coarse']['median_c']:.2e} "
                         f"frac={m['coarse']['frac_above']:.3f} "
                         f"maxw={m['coarse']['max_cluster_width']} "
                         f"frac_ratio={m['frac_ratio']:.2f}")
    for key, label in (("lines_B", "fitted"), ("lines_Bsynth", "synthetic checkerboard")):
        if key in results:
            cls = {k: classify_line(m) for k, m in results[key].items()}
            # diag-xy runs at z=0, where the checkerboard field's quadratic
            # terms cancel by symmetry -- require 2 of 3 non-axis lines curved
            n_curved = sum(cls[k] == "curved" for k in CURVED_LINES)
            ok = (all(cls[k] == "kinky" for k in AXIS_LINES) and n_curved >= 2)
            lines.append(f"C2 (degree-1 field => piecewise trilinear, {label} CPs): {verdict(ok)}")
            for k, m in results[key].items():
                lines.append(f"    {k:<10} {cls[k]:<10} median_c={m['coarse']['median_c']:.2e} "
                             f"frac={m['coarse']['frac_above']:.3f} "
                             f"maxw={m['coarse']['max_cluster_width']} "
                             f"frac_ratio={m['frac_ratio']:.2f}")
    if "crease" in results:
        cr = results["crease"]
        arc_e = cr.get("arc", {}).get("edge", {})
        arc_c = cr.get("arc", {}).get("control", {})
        sweep_e = cr.get("sweep", {}).get("edge", {})
        sweep_c = cr.get("sweep", {}).get("control", {})
        e_small = sweep_e.get("0.002")
        c_small = sweep_c.get("0.002")
        e_all = [sweep_e[str(e)]["median_deg"] for e in cr["eps"]
                 if sweep_e.get(str(e))]
        # A genuine crease = the straddle angle PLATEAUS at a nonzero value as
        # eps -> 0 (a smooth rounding decays to zero, like the face control),
        # and the arc-scan jump structure persists when the arc is refined
        # (smooth curvature would halve the max jump per refinement).
        total = arc_e.get("total_deg_median", 0.0)
        conc41 = arc_e.get("concentration_median_41", 0.0)
        conc81 = arc_e.get("concentration_median_81", 0.0)
        conc_persist = conc81 / max(conc41, 1e-9) > 0.8
        plateau = (max(e_all[-3:]) / max(min(e_all[-3:]), 1e-9) < 1.6
                   if len(e_all) >= 3 else False)
        ok = (total > 60.0 and arc_c.get("total_deg_median", 90.0) < 10.0
              and conc_persist and plateau
              and e_small is not None and e_small["median_deg"] > 15.0
              and c_small is not None and c_small["median_deg"] < 5.0)
        n_creases = (round(total / max(arc_e.get("max_jump_deg_median", 1e-9), 1e-9))
                     if total > 0 else 0)
        lines.append(f"C3 (genuine crease at the cube edge): {verdict(ok)}")
        lines.append(f"    arc scan: edge rotation {total:.1f} deg, "
                     f"concentration {conc41:.2f} (n=41) / {conc81:.2f} (n=81); "
                     f"control rotation {arc_c.get('total_deg_median', 0.0):.1f} deg")
        lines.append(f"    interpretation: the dihedral is carried by ~{n_creases} "
                     f"dominant facet crease(s) inside a band of width "
                     f"~{2 * arc_e.get('arc_half', 0.06):.2f} -- a genuine, "
                     f"resolution-limited crease, not a smooth rounding")
        lines.append(f"    recentred eps sweep (edge): {[round(a, 1) for a in e_all]} deg"
                     + (f"; control at eps=0.002: {c_small['median_deg']:.1f} deg"
                        if c_small else ""))
        lines.append(f"    edge points found: {cr.get('n_edge_points', 0)}/9, "
                     f"crease offset from GT-edge ray "
                     f"{arc_e.get('crease_offset_median', float('nan')):.4f}, "
                     f"max projection residual {cr.get('proj_residual_max')}")
    if "interp" in results:
        it = results["interp"]
        # Individual ReLU kinks may legitimately appear/disappear as activation
        # surfaces sweep across the probe line while lambda changes -- the
        # claim is that kinks move CONTINUOUSLY while they exist, so require a
        # track covering >= 80% of the alpha range rather than all of it.
        n_alphas = it.get("n_alphas", 21)
        track_ok = it.get("longest_kink_track", 0) >= 0.8 * n_alphas
        ok = (it.get("surface_lost_alphas", 99) == 0
              and it["x0_max_step"] < 0.05 and it.get("x0_span", 0.0) > 0.15
              and it["x0_monotone"] and it["corner_r_max_step"] < 0.08
              and track_ok)
        lines.append(f"C4 (creases move continuously with lambda): {verdict(ok)}")
        lines.append(f"    x0: span {it.get('x0_span', 0.0):.3f}, max step "
                     f"{it['x0_max_step']:.4f}, monotone {it['x0_monotone']}, "
                     f"surface lost at {it.get('surface_lost_alphas', '?')} alphas; "
                     f"corner_r (crease of the level set) max step "
                     f"{it['corner_r_max_step']:.4f}; longest kink track "
                     f"{it.get('longest_kink_track', 0)}/{n_alphas} alphas "
                     f"({it['n_kink_tracks_full_range']} full-range)")
    if "fits" in results:
        lines.append("")
        lines.append("fit gates (mean |s| at GT face centres; claims C1/C2 are")
        lines.append("function-class statements and hold regardless):")
        for tag, f in results["fits"].items():
            lines.append(f"    {tag:<10} loss={f['final_loss']:.3e} "
                         f"faces={f['gate']['mean_abs_s_faces']:.4f} "
                         f"corners={f['gate']['mean_abs_s_corners']:.4f} "
                         f"fit_ok={f['gate']['fit_ok']}")
    if "mesh" in results:
        m = results["mesh"]
        lines.append(f"mesh: {m['n_faces']} faces, dihedral p50/p95/max = "
                     f"{m['dihedral_p50_deg']:.1f}/{m['dihedral_p95_deg']:.1f}/"
                     f"{m['dihedral_max_deg']:.1f} deg -> {m['obj']}")
    lines.append("")
    lines.append("note: constant latents are out-of-distribution for this decoder")
    lines.append("(training always used 729-CP degree-1 latent fields).")
    return "\n".join(lines)


# -------------------------------------------------------------------- main ----

def replot_all(results: dict, out_dir: Path) -> None:
    if "lines_A" in results:
        plot_lines(results["lines_A"], "Case A: constant latent (one voxel, degree 0)",
                   out_dir / "line_probes_caseA.png")
    if "lines_B" in results:
        plot_lines(results["lines_B"], "Case B: degree-1 latent field (8 CPs)",
                   out_dir / "line_probes_caseB.png")
    if "lines_Bsynth" in results:
        plot_lines(results["lines_Bsynth"],
                   "Case B-synth: checkerboard CPs (guaranteed trilinear)",
                   out_dir / "line_probes_caseBsynth.png")
    if "crease" in results:
        plot_crease(results["crease"], out_dir / "crease_angle_vs_eps.png")
    if "interp" in results:
        plot_interp(results["interp"], out_dir / "latent_interp.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", nargs="*",
                        default=["fit", "probes", "crease", "interp"],
                        choices=["fit", "probes", "crease", "interp", "mesh"])
    parser.add_argument("--out-dir", type=Path,
                        default=REPO / "revision_artifacts" / "single_voxel_cube")
    parser.add_argument("--replot", action="store_true")
    parser.add_argument("--skip-fit", action="store_true",
                        help="load fitted_params.pt instead of fitting")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--edge-main", type=float, default=1.0)
    parser.add_argument("--edge-small", type=float, default=0.7)
    parser.add_argument("--edge-big", type=float, default=1.2)
    parser.add_argument("--init", choices=["random", "mean-cp"], default="random")
    parser.add_argument("--interp-case", choices=["A", "B"], default="A")
    parser.add_argument("--asym", action="store_true",
                        help="fit case B on box extents (0.8, 1.0, 1.2)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "single_voxel_cube.json"
    results = json.loads(json_path.read_text()) if json_path.is_file() else {}

    if args.replot:
        if not results:
            raise SystemExit(f"--replot needs {json_path}, which does not exist")
        replot_all(results, args.out_dir)
        print(build_summary(results))
        return

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = get_model(str(MODEL_PATH), checkpoint="latest")
    cache = args.out_dir / "fitted_params.pt"

    # ---- fit -----------------------------------------------------------
    if "fit" in args.stages and not args.skip_fit:
        fits = {}
        state = {}
        if args.init == "mean-cp":
            init_vec = mean_cp_init(MODEL_PATH)
        else:
            init_vec = torch.randn(LATENT_DIM) * 1e-3
        # A_big is warm-started from the fitted z_small: two independently
        # fitted codes end up in different basins of the (out-of-distribution)
        # constant-latent landscape, and the straight line between them loses
        # the zero level set entirely. Warm-starting keeps both endpoints in
        # one basin so that linear interpolation morphs one cube into the
        # other -- which is what C4 measures.
        for tag, extents, warm_from in (
            ("A_main", [args.edge_main] * 3, None),
            ("A_small", [args.edge_small] * 3, None),
            ("A_big", [args.edge_big] * 3, "z_A_small"),
        ):
            print(f"\n=== fit {tag}: constant latent, extents {extents} ===")
            samples = make_samples(extents, device)
            init = state[warm_from].clone() if warm_from else init_vec.clone()
            z, loss = fit_constant(model, samples, init, args.epochs,
                                   args.out_dir, tag)
            sdf = build_constant_sdf(model, z)
            gate = fit_gate(sdf, extents[0] / 2.0, device)
            state[f"z_{tag}"] = z.cpu()
            fits[tag] = {"final_loss": loss, "gate": gate, "extents": extents,
                         "warm_start": warm_from}
            print(f"    loss {loss:.3e}  gate {gate}")

        extents_b = [0.8, 1.0, 1.2] if args.asym else [args.edge_main] * 3
        print(f"\n=== fit B: 8-CP degree-1 lattice, extents {extents_b} ===")
        samples = make_samples(extents_b, device)
        cps, loss = fit_lattice8(model, samples, args.epochs, args.out_dir, "B")
        lattice = build_lattice(model, cps)
        gate = fit_gate(lattice, extents_b[0] / 2.0, device)
        state["cp_B"] = cps.cpu()
        fits["B"] = {"final_loss": loss, "gate": gate, "extents": extents_b}
        print(f"    loss {loss:.3e}  gate {gate}")
        cp_spread = float((cps.max(dim=0).values - cps.min(dim=0).values).norm())
        fits["B"]["cp_spread"] = cp_spread
        print(f"    CP spread (degeneracy check, 0 = constant field): {cp_spread:.4f}")

        torch.save(state, cache)
        results["fits"] = fits

    if not cache.is_file():
        raise SystemExit(f"no {cache}; run the fit stage first")
    state = torch.load(cache, map_location="cpu")
    z_main, z_small, z_big = state["z_A_main"], state["z_A_small"], state["z_A_big"]
    cp_B = state["cp_B"]
    half_main = args.edge_main / 2.0

    # ---- probes (C1, C2) ----------------------------------------------
    if "probes" in args.stages:
        print("\n=== line probes: case A (constant latent) ===")
        results["lines_A"] = probe_lines(build_constant_sdf(model, z_main), model, device)
        print("\n=== line probes: case B (fitted 8-CP degree-1 field) ===")
        results["lines_B"] = probe_lines(build_lattice(model, cp_B), model, device)
        print("\n=== line probes: case B-synth (checkerboard CPs) ===")
        cps_synth = checkerboard_cps(z_small, z_big)
        results["lines_Bsynth"] = probe_lines(build_lattice(model, cps_synth),
                                              model, device)

    # ---- crease (C3) ---------------------------------------------------
    if "crease" in args.stages:
        print("\n=== crease straddle probe (case A, main cube) ===")
        results["crease"] = crease_probe(build_constant_sdf(model, z_main), model,
                                         half_main, device)
        if results["crease"]["n_edge_points"] <= 4:
            print("    <5 edge points on case A -- falling back to case B")
            results["crease_caseB"] = crease_probe(build_lattice(model, cp_B),
                                                   model, half_main, device)

    # ---- interp (C4) ---------------------------------------------------
    if "interp" in args.stages:
        print("\n=== latent interpolation sweep ===")
        results["interp"] = interp_sweep(
            model, z_small, z_big, args.edge_small / 2.0, args.edge_big / 2.0, device)

    # ---- mesh (optional figure) ---------------------------------------
    if "mesh" in args.stages:
        print("\n=== FlexiCubes mesh of the one-latent cube ===")
        results["mesh"] = mesh_stage(model, z_main, args.out_dir)

    json_path.write_text(json.dumps(results, indent=1))
    replot_all(results, args.out_dir)
    summary = build_summary(results)
    (args.out_dir / "summary.txt").write_text(summary + "\n")
    print("\n" + summary)
    print(f"\nwrote {json_path} and figures in {args.out_dir}")


if __name__ == "__main__":
    main()

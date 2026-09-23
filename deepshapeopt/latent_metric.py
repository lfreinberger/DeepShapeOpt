"""Geometric metric of a latent parametrization: the Jacobian of the wall's normal
displacement with respect to the free design variables, and what its spectrum says.

The design surface moves along its normal when the latent parameters change (the snap's
differentiable final step is a normal displacement, see ``hexmesh/snap.py``). The geometric
size of a parameter change is therefore the L2(dA) norm of the normal displacement field,

    |delta x|_geo^2 = sum_i a_i (n_i . delta x_i)^2 ,

with a_i the vertex area and n_i the unit normal of the design surface. ``wall_jacobian``
returns J = d y / d lambda_free for y_i = sqrt(a_i) n_i . x_i (units mm^2 per latent unit), so
that J^T J is the Gram matrix of that metric in latent space. ``analyze_jacobian`` reduces it to
the three questions about a parametrization:

1. anisotropy  -- column norms |J_j|: geometric change per unit step of each variable;
2. redundancy  -- singular value spectrum (effective rank) and the cosines between columns:
                  two variables with cosine ~1 move the wall the same way;
3. drift       -- compare two probes (different designs) with ``compare_probes``.

Rows of J are computed with reverse-mode VJPs of the scalar field y (batched through
``is_grads_batched`` where the graph allows it), so no forward-mode support is needed.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import torch

logger = logging.getLogger(__name__)


def design_surface_field(verts: torch.Tensor, faces: torch.Tensor):
    """Area-weighted normal displacement field of a triangulated design surface.

    Returns ``(y, ids, area)``: ``y[i] = sqrt(a_i) * n_i . verts[ids[i]]`` (differentiable
    through ``verts``; normals and areas detached), the vertex ids of the surface and the
    vertex areas (mm^2, a third of every adjacent triangle).
    """
    faces = faces.to(torch.long).to(verts.device)
    ids = torch.unique(faces)
    v = verts.detach()
    fn = torch.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]], dim=1)
    vn = torch.zeros_like(v)
    for c in range(3):
        vn.index_add_(0, faces[:, c], fn)
    # |cross| = 2 * area, a third of each triangle per vertex -> vertex area = |sum| / 6
    # (exact for a flat fan, a lower bound where adjacent normals disagree)
    area_all = vn.norm(dim=1) / 6.0
    n_all = vn / vn.norm(dim=1, keepdim=True).clamp_min(1e-30)
    area, n = area_all[ids], n_all[ids]
    y = (area.sqrt()[:, None] * n * verts[ids]).sum(dim=1)
    return y, ids, area


def wall_jacobian(y: torch.Tensor, param: torch.Tensor, mask_free: torch.Tensor,
                  batch: int = 64) -> torch.Tensor:
    """d y / d param[mask_free] as an (m, n_free) tensor, one VJP per row."""
    m = int(y.numel())
    free = torch.as_tensor(mask_free, device=param.device, dtype=torch.bool).reshape(-1)
    n_free = int(free.sum())
    J = torch.empty((m, n_free), device=param.device, dtype=param.dtype)
    t0 = time.time()
    batched = True
    i = 0
    while i < m:
        b = min(batch, m - i)
        if batched:
            try:
                eye = torch.zeros((b, m), device=y.device, dtype=y.dtype)
                eye[torch.arange(b), torch.arange(i, i + b)] = 1.0
                (rows,) = torch.autograd.grad(
                    y, param, grad_outputs=eye, retain_graph=True, is_grads_batched=True,
                )
                J[i:i + b] = rows.reshape(b, -1)[:, free]
                i += b
                continue
            except RuntimeError as err:  # an op without a vmap rule: fall back to a loop
                logger.warning("batched VJP unsupported (%s); falling back to row-wise VJPs",
                               str(err).splitlines()[0][:120])
                batched = False
        e = torch.zeros(m, device=y.device, dtype=y.dtype)
        e[i] = 1.0
        (row,) = torch.autograd.grad(y, param, grad_outputs=e, retain_graph=True)
        J[i] = row.reshape(-1)[free]
        i += 1
        if i % 1000 == 0:
            logger.info("wall_jacobian: %d / %d rows, %.0f s", i, m, time.time() - t0)
    logger.info("wall_jacobian: %d x %d in %.1f s (%s)", m, n_free, time.time() - t0,
                "batched" if batched else "row-wise")
    return J


def analyze_jacobian(J: torch.Tensor, n_latent_per_cp: int, keep_vectors: int = 50,
                     max_step: float | None = None, area_total: float | None = None) -> dict:
    """Spectrum, per-variable norms and column cosines of the geometric Jacobian."""
    Jd = J.detach().to(torch.float64).cpu()
    m, n = Jd.shape
    # SVD of the (m, n) matrix; n_free <= m is the usual case
    U, S, Vh = torch.linalg.svd(Jd, full_matrices=False)
    s = S.numpy()
    s_rel = s / s[0]
    col = Jd.norm(dim=0).numpy()
    nz = col > 0
    # Column cosines (redundancy): normalized Gram, largest off-diagonal magnitude per column.
    G = (Jd.T @ Jd).numpy()
    d = np.sqrt(np.clip(np.diag(G), 1e-300, None))
    C = G / np.outer(d, d)
    np.fill_diagonal(C, 0.0)
    cos_max = np.abs(C).max(axis=1)
    cos_partner = np.abs(C).argmax(axis=1)
    cos_max[~nz] = np.nan
    p = s ** 2 / (s ** 2).sum()
    p = p[p > 0]
    out = {
        "m": m, "n_free": n,
        "sigma": s, "sigma_rel": s_rel,
        "n_above": {f"{t:g}": int((s_rel > t).sum()) for t in (1e-1, 1e-2, 1e-3, 1e-4, 1e-6)},
        "rank_participation": float(s.sum() ** 2 / (s ** 2).sum()),
        "rank_entropy": float(np.exp(-(p * np.log(p)).sum())),
        "col_norm": col,
        "col_norm_quantiles": {q: float(np.quantile(col[nz], q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95, 1.0)},
        "col_norm_zero": int((~nz).sum()),
        "cos_max": cos_max,
        "cos_max_quantiles": {q: float(np.nanquantile(cos_max, q)) for q in (0.5, 0.9, 0.99)},
        "frac_cos_above": {f"{t:g}": float(np.nanmean(cos_max > t)) for t in (0.9, 0.99, 0.999)},
        "V_top": Vh[:keep_vectors].T.numpy(),
        "U_top": U[:, :keep_vectors].numpy(),
    }
    # Per control point (a block of n_latent_per_cp columns): RMS geometric change per unit
    # step, whether a variable's near-duplicate sits in its own block, and the block's own
    # rank -- how many of the n_latent_per_cp code directions act on the geometry at all.
    if n % n_latent_per_cp == 0:
        n_cp = n // n_latent_per_cp
        out["cp_norm"] = np.sqrt((col.reshape(n_cp, n_latent_per_cp) ** 2).mean(axis=1))
        same_cp = (cos_partner // n_latent_per_cp) == (np.arange(n) // n_latent_per_cp)
        out["frac_partner_same_cp"] = float(np.mean(same_cp[nz]))
        block_rank = np.full(n_cp, np.nan)
        for c in range(n_cp):
            blk = Jd[:, c * n_latent_per_cp:(c + 1) * n_latent_per_cp]
            if float(blk.abs().max()) == 0.0:
                continue
            sb = torch.linalg.svdvals(blk).numpy()
            block_rank[c] = int((sb / sb[0] > 1e-2).sum())
        out["cp_block_rank"] = block_rank
        out["cp_block_rank_quantiles"] = {q: float(np.nanquantile(block_rank, q)) for q in (0.1, 0.5, 0.9)}
    if max_step is not None and area_total:
        # RMS normal displacement over the surface for a full move-limit step of ONE variable.
        out["rms_disp_per_max_step_mm"] = col * max_step / np.sqrt(area_total)
        out["rms_disp_top_singular_mm"] = s[0] * max_step / np.sqrt(area_total)
    return out


def summary_lines(a: dict) -> list[str]:
    q = a["col_norm_quantiles"]
    lines = [
        f"Jacobian {a['m']} surface points x {a['n_free']} free variables",
        f"singular values: sigma_max {a['sigma'][0]:.3e}; count above 1e-1/1e-2/1e-3/1e-4/1e-6 of sigma_max: "
        + " / ".join(str(a["n_above"][k]) for k in ("0.1", "0.01", "0.001", "0.0001", "1e-06")),
        f"effective rank: participation {a['rank_participation']:.1f}, entropy {a['rank_entropy']:.1f} of {a['n_free']}",
        f"column norms (geometric change per unit latent step): q05/q50/q95/max "
        f"{q[0.05]:.3e} / {q[0.5]:.3e} / {q[0.95]:.3e} / {q[1.0]:.3e}  (ratio q95/q05 {q[0.95] / max(q[0.05], 1e-300):.1f}, "
        f"{a['col_norm_zero']} zero columns)",
        f"redundancy: max |cos| between a variable's column and any other, median {a['cos_max_quantiles'][0.5]:.3f}, "
        f"q90 {a['cos_max_quantiles'][0.9]:.3f}; fraction of variables with a near-duplicate (cos > 0.9 / 0.99): "
        f"{a['frac_cos_above']['0.9']:.1%} / {a['frac_cos_above']['0.99']:.1%}",
    ]
    if "cp_block_rank" in a:
        r = a["cp_block_rank_quantiles"]
        lines.append(
            f"per control point ({int(a['n_free'] // len(a['cp_norm']))} code dims each): code directions with a geometric "
            f"effect above 1 % of the block's largest, q10/q50/q90 {r[0.1]:.0f} / {r[0.5]:.0f} / {r[0.9]:.0f}; "
            f"{a['frac_partner_same_cp']:.0%} of the near-duplicates sit in the same control point"
        )
    if "rms_disp_per_max_step_mm" in a:
        r = a["rms_disp_per_max_step_mm"]
        nz = r > 0
        lines.append(
            f"RMS normal displacement for one full move-limit step of a single variable: median "
            f"{np.median(r[nz]):.2e} mm, max {r.max():.2e} mm; along the top singular direction {a['rms_disp_top_singular_mm']:.2e} mm"
        )
    return lines


def compare_probes(a: np.lib.npyio.NpzFile, b: np.lib.npyio.NpzFile, ks=(5, 20, 50)) -> list[str]:
    """How much the geometric map changed between two designs (principal angles of the
    dominant right singular subspaces, spectrum and column-norm agreement)."""
    lines = []
    sa, sb = a["sigma"], b["sigma"]
    n = min(len(sa), len(sb))
    for t in (0.1, 0.01, 0.001):
        lines.append(f"count above {t:g} sigma_max: {int((sa / sa[0] > t).sum())} vs {int((sb / sb[0] > t).sum())}")
    ratio = sb[:n] / sa[:n]
    lines.append(f"sigma_b / sigma_a: top {ratio[0]:.2f}, median over the first 100 {np.median(ratio[:100]):.2f}")
    ca, cb = a["col_norm"], b["col_norm"]
    nz = (ca > 0) & (cb > 0)
    lines.append(f"column norms b/a: median {np.median(cb[nz] / ca[nz]):.2f}, q05 {np.quantile(cb[nz] / ca[nz], 0.05):.2f}, "
                 f"q95 {np.quantile(cb[nz] / ca[nz], 0.95):.2f}; correlation of log norms {np.corrcoef(np.log(ca[nz]), np.log(cb[nz]))[0, 1]:.3f}")
    Va, Vb = a["V_top"], b["V_top"]
    for k in ks:
        k = min(k, Va.shape[1], Vb.shape[1])
        cosines = np.linalg.svd(Va[:, :k].T @ Vb[:, :k], compute_uv=False)
        lines.append(f"top-{k} right singular subspaces: principal-angle cosines min {cosines.min():.3f}, "
                     f"median {np.median(cosines):.3f} (1 = same subspace)")
    return lines

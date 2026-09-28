"""Mirror symmetry of the latent lattice: control-point permutations and their checks.

A mirror at the centre of the lattice box along axis ``a`` maps the lattice onto itself when the
knot vector is symmetric and the tile count along ``a`` is even (the tiling map satisfies
``u(1 - x) = (-1)^t u(x)``). Then ``phi_c(M x) = phi_{P c}(x)`` with ``P`` the pure control-point
permutation ``i_a -> n_a - 1 - i_a``, and a code field with ``c = P c`` gives a mirror-symmetric
shape. Symmetrizing parameters and gradients with ``(t + t[P]) / 2`` keeps a fit or an optimizer
step exactly on that subspace.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from .locking import FACE_AXES, _greville_1d

logger = logging.getLogger(__name__)


class SymmetryError(ValueError):
    """The lattice or the target geometry is not mirror-symmetric as configured."""


class MirrorSymmetry:
    """Control-point permutations of mirror planes through the lattice-box centre."""

    def __init__(self, axes: list[str], perms: list[torch.Tensor]):
        self.axes = list(axes)
        self.perms = perms

    @classmethod
    def from_spline(cls, spline_sp, axes, *, order: str = "F", device="cpu") -> "MirrorSymmetry":
        """Permutations from the Greville abscissae of ``spline_sp`` (control-point order ``order``)."""
        degrees = [int(p) for p in spline_sp.degrees]
        kvs = [np.asarray(kv, dtype=float) for kv in spline_sp.knot_vectors]
        greville = [_greville_1d(kv, p) for kv, p in zip(kvs, degrees)]
        shape = tuple(len(g) for g in greville)
        ids = np.arange(int(np.prod(shape)))
        grid = np.unravel_index(ids, shape, order=order)

        perms = []
        for axis in axes:
            a = FACE_AXES[axis]
            kv, g = kvs[a], greville[a]
            span = kv[-1] - kv[0]
            if not np.allclose(kv, (kv[0] + kv[-1]) - kv[::-1], rtol=0.0, atol=1e-6 * span):
                raise SymmetryError(f"knot vector along {axis} is not symmetric about its centre")
            if not np.allclose(g, (kv[0] + kv[-1]) - g[::-1], rtol=0.0, atol=1e-6 * span):
                raise SymmetryError(f"Greville points along {axis} are not symmetric about the box centre")
            mirrored = list(grid)
            mirrored[a] = shape[a] - 1 - grid[a]
            perm = np.ravel_multi_index(tuple(mirrored), shape, order=order)
            perms.append(torch.as_tensor(perm, dtype=torch.long, device=device))
        return cls(axes, perms)

    def __bool__(self) -> bool:
        return bool(self.perms)

    def symmetrize(self, t: torch.Tensor) -> torch.Tensor:
        """Orbit average over the control-point axis (dim 0) of ``t``."""
        for perm in self.perms:
            t = 0.5 * (t + t[perm.to(t.device)])
        return t

    @torch.no_grad()
    def symmetrize_(self, t: torch.Tensor) -> torch.Tensor:
        t.copy_(self.symmetrize(t))
        return t

    @torch.no_grad()
    def residual(self, param: torch.Tensor) -> float:
        """Largest ``|c - P c| / |c|`` over the mirror axes."""
        norm = float(param.norm())
        if norm == 0.0:
            return 0.0
        return max(float((param - param[perm.to(param.device)]).norm()) / norm for perm in self.perms)

    def check_index_set(self, idx: torch.Tensor, n_ctrl: int, what: str = "locked control points") -> None:
        """Raise unless the index set ``idx`` is mapped onto itself by every mirror."""
        mask = torch.zeros(n_ctrl, dtype=torch.bool)
        mask[idx.cpu()] = True
        for axis, perm in zip(self.axes, self.perms):
            if not torch.equal(mask, mask[perm.cpu()]):
                raise SymmetryError(f"the {what} are not mirror-symmetric in {axis}")

    def mirror_points(self, x: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        """Mirror points about the centre of ``box`` ((2, 3)) along every configured axis."""
        x = x.clone()
        for axis in self.axes:
            a = FACE_AXES[axis]
            x[:, a] = (box[0, a] + box[1, a]).to(x.dtype) - x[:, a]
        return x

    @torch.no_grad()
    def check_lattice_equivariance(self, lattice_struct, param: torch.Tensor, *, n_points: int = 8192,
                                   code_std: float = 0.2, tol: float = 1e-4, seed: int = 0) -> float:
        """Evaluate a random symmetric code field at ``x`` and at every mirror image of ``x``.

        Catches a control-point order, box or tiling map that breaks the equivariance. The
        parameters are restored afterwards. Returns the largest SDF difference (normalized units).
        """
        gen = torch.Generator().manual_seed(int(seed))
        saved = param.detach().clone()
        box = lattice_struct.bounds.detach()
        try:
            codes = saved + code_std * torch.randn(saved.shape, generator=gen, dtype=saved.dtype).to(saved.device)
            param.copy_(self.symmetrize(codes.clamp(-1.0, 1.0)))
            u = torch.rand((n_points, 3), generator=gen, dtype=torch.float32).to(param.device)
            lo, hi = box[0].to(u), box[1].to(u)
            x = lo + u * (hi - lo)
            ref = lattice_struct(x)
            err = 0.0
            for i in range(len(self.axes)):
                single = MirrorSymmetry([self.axes[i]], [self.perms[i]])
                err = max(err, float((lattice_struct(single.mirror_points(x, box)) - ref).abs().max()))
        finally:
            param.copy_(saved)
        if err > tol:
            raise SymmetryError(
                f"the lattice is not mirror-equivariant in {self.axes}: max SDF difference {err:.2e} "
                f"(tolerance {tol:.0e}); check the tile count, the knot vectors and the lattice box"
            )
        logger.info("symmetry: lattice mirror-equivariant in %s (max SDF difference %.1e)", self.axes, err)
        return err

    @torch.no_grad()
    def check_ground_truth(self, gt_sdf, mesh, box: torch.Tensor, *, dist_to_phys: float, tolerance: float,
                           n_points: int = 200000) -> dict:
        """Mirror mismatch of the target geometry about the lattice-box centre, in physical units.

        Surface points of ``mesh`` inside ``box`` are mirrored and evaluated in ``gt_sdf``; the
        p95 of ``|sdf|`` must not exceed ``tolerance``. The error message estimates where the
        geometry's own mirror plane is (area-weighted surface centroid), so an off-centre design
        domain shows up as an offset.
        """
        import trimesh

        points, _ = trimesh.sample.sample_surface(mesh, int(n_points), seed=0)
        box_np = box.detach().cpu().numpy().astype(float)
        points = points[np.all((points >= box_np[0]) & (points <= box_np[1]), axis=1)]
        if len(points) == 0:
            raise SymmetryError("no surface of the target mesh lies inside the lattice box")
        x = torch.as_tensor(points, dtype=torch.float32)
        mismatch = {}
        for axis in self.axes:
            a = FACE_AXES[axis]
            xm = x.clone()
            xm[:, a] = float(box_np[0, a] + box_np[1, a]) - xm[:, a]
            d = gt_sdf(xm).detach().abs().cpu().numpy().ravel() * float(dist_to_phys)
            centroid = float(np.average(mesh.triangles_center[:, a], weights=mesh.area_faces))
            offset = (centroid - 0.5 * float(box_np[0, a] + box_np[1, a])) * float(dist_to_phys)
            stats = {"mean": float(d.mean()), "p95": float(np.quantile(d, 0.95)), "max": float(d.max()),
                     "centroid_offset": offset}
            mismatch[axis] = stats
            logger.info("symmetry: target mirror mismatch in %s: mean %.3g, p95 %.3g, max %.3g "
                        "(surface centroid %.3g off the box centre)",
                        axis, stats["mean"], stats["p95"], stats["max"], offset)
            if stats["p95"] > tolerance:
                raise SymmetryError(
                    f"the target geometry is not mirror-symmetric about the design-domain centre in {axis}: "
                    f"p95 mismatch {stats['p95']:.3g} > tolerance {tolerance:.3g} (mean {stats['mean']:.3g}, "
                    f"max {stats['max']:.3g}); its surface centroid lies {offset:.3g} off the centre. Centre the "
                    "design domain on the geometry's mirror plane, or raise symmetry.tolerance to fit the "
                    "mirror average on purpose"
                )
        return mismatch

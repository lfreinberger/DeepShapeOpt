"""Free-form deformation (FFD) parametrization.

A displacement B-spline ``D`` over the design domain, evaluated in the normalized
coordinates of a :class:`~deepshapeopt.geometry.frame.DomainFrame` (its knot vectors
span ``frame.box_norm``) with control values in *physical* units:

    T(x) = x + D(frame.to_norm(x)),    x inside the design domain.

The control-point displacements are the design variables: an ``nn.Parameter`` of
shape ``(N, 3)`` in splinepy / Fortran ordering (first parametric axis fastest), the
same layout as the latent lattice spline, so the locking helpers in
:mod:`deepshapeopt.parametrization.locking` and the lattice exports in
``DeepSDFStruct.export_knot_grid`` apply unchanged.  The spline is built with
``build_parameter_spline`` (one knot span per "tile", ``n_control_points = tiling +
degree``), i.e. clamped knot vectors with uniformly spaced interior knots.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import splinepy
import torch

from DeepSDFStruct.export_knot_grid import (
    export_control_lattice_physical,
    export_control_volume_physical,
)
from DeepSDFStruct.geom_reconstruction import build_parameter_spline
from DeepSDFStruct.torch_spline import TorchSpline

from deepshapeopt.geometry.frame import DomainFrame
from deepshapeopt.parametrization.locking import greville_points_3d

logger = logging.getLogger(__name__)

FACE_NAMES = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")


def _parse_face(face: str) -> tuple[int, int]:
    """``"z_min"`` -> ``(axis=2, side=0)``."""
    axis_name, _, side = str(face).partition("_")
    if axis_name not in "xyz" or len(axis_name) != 1 or side not in ("min", "max"):
        raise ValueError(f"Invalid design-domain face {face!r}; expected one of {FACE_NAMES}")
    return "xyz".index(axis_name), 0 if side == "min" else 1


class FFDDeformation(torch.nn.Module):
    """``x -> x + D(frame.to_norm(x))`` with a :class:`TorchSpline` displacement field.

    The spline's knot vectors must span ``frame.box_norm`` (see
    :func:`build_ffd_deformation`); queries are expected inside the design domain
    (a B-spline silently extrapolates outside its knot range).  Control values are
    displacements in physical units and start at zero (identity map).
    """

    def __init__(
        self,
        disp_spline_sp: splinepy.BSpline,
        frame: DomainFrame,
        device="cpu",
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        self.disp = TorchSpline(disp_spline_sp, device=device, dtype=dtype)
        with torch.no_grad():
            self.disp.control_points.zero_()
        as_buf = lambda t: torch.as_tensor(t).detach().clone().to(device=device, dtype=dtype)
        self.register_buffer("center", as_buf(frame.center))
        self.register_buffer("scale", as_buf(frame.scale))
        self.register_buffer("design_domain", as_buf(frame.design_domain))

    @property
    def control_points(self) -> torch.nn.Parameter:
        """Displacement control values, ``(N, 3)`` physical units, F-order."""
        return self.disp.control_points

    @property
    def dtype(self) -> torch.dtype:
        return self.control_points.dtype

    @property
    def device(self) -> torch.device:
        return self.control_points.device

    def to_norm(self, x_phys: torch.Tensor) -> torch.Tensor:
        return (x_phys - self.center) * self.scale

    def displacement(self, x_phys: torch.Tensor) -> torch.Tensor:
        """``D(x)`` at physical points ``(P, 3)`` (cast to the control-point dtype)."""
        x = x_phys.to(device=self.device, dtype=self.dtype)
        return self.disp(self.to_norm(x))

    def forward(self, x_phys: torch.Tensor) -> torch.Tensor:
        x = x_phys.to(device=self.device, dtype=self.dtype)
        return x + self.disp(self.to_norm(x))


@dataclass
class FFDSetup:
    deformation: FFDDeformation
    disp_spline_sp: splinepy.BSpline
    n_control_points: list[int]
    spline_degree: list[int]
    greville_phys: torch.Tensor  # (N, 3) undeformed control lattice (Greville abscissae), F-order

    def control_lattice_physical(self) -> torch.Tensor:
        """Current control lattice in physical space: Greville points + displacements."""
        return self.greville_phys + self.deformation.control_points.detach()

    def export_lattice(self, vtp_path: Path | str, vts_path: Path | str | None = None) -> None:
        """Write the deformed control lattice (.vtp polyline cage) and, optionally,
        the structured control volume (.vts, with ``displacement`` arrays)."""
        cp = self.control_lattice_physical().cpu().numpy()
        export_control_lattice_physical(cp, self.n_control_points, str(vtp_path), order="F")
        if vts_path is not None:
            export_control_volume_physical(
                cp, self.n_control_points, str(vts_path),
                undeformed=self.greville_phys.cpu().numpy(),
            )


def build_ffd_deformation(
    ffd_cfg: dict,
    frame: DomainFrame,
    device="cpu",
    dtype: torch.dtype = torch.float64,
) -> FFDSetup:
    """Build the FFD displacement spline over the design domain from a config block.

    ``ffd_cfg`` keys: ``n_control_points`` (3 ints, > degree) and ``spline_degree``
    (3 ints, default ``[2, 2, 2]``).
    """
    n_cp = [int(n) for n in ffd_cfg["n_control_points"]]
    degree = [int(p) for p in ffd_cfg.get("spline_degree", [2, 2, 2])]
    if len(n_cp) != 3 or len(degree) != 3:
        raise ValueError("ffd.n_control_points and ffd.spline_degree must have 3 entries")
    for n, p in zip(n_cp, degree):
        if p < 1 or n <= p:
            raise ValueError(
                f"ffd: need n_control_points > spline_degree >= 1 per axis, got {n_cp} / {degree}"
            )
    tiling = [n - p for n, p in zip(n_cp, degree)]

    box_norm = frame.box_norm.detach().cpu().numpy().astype(np.float64)
    disp_spline_sp = build_parameter_spline(degree, tiling, latent_dim=3, bounds=box_norm)
    greville_norm, dims = greville_points_3d(disp_spline_sp, order="F")
    if list(dims) != n_cp or disp_spline_sp.control_points.shape[0] != int(np.prod(n_cp)):
        raise RuntimeError(
            f"FFD spline has {disp_spline_sp.control_points.shape[0]} control points "
            f"(dims {tuple(dims)}), expected {n_cp}"
        )

    deformation = FFDDeformation(disp_spline_sp, frame, device=device, dtype=dtype)
    greville_phys = frame.to_phys(
        torch.as_tensor(greville_norm, dtype=frame.center.dtype, device=frame.center.device)
    ).to(device=device, dtype=dtype)
    logger.info(
        "FFD: %s control points (degree %s) over the design domain -> %d displacement DOFs",
        n_cp, degree, 3 * int(np.prod(n_cp)),
    )
    return FFDSetup(
        deformation=deformation,
        disp_spline_sp=disp_spline_sp,
        n_control_points=n_cp,
        spline_degree=degree,
        greville_phys=greville_phys,
    )


def min_jacobian_det_ks(
    deformation: FFDDeformation,
    n_samples_per_dim: int = 6,
    ks_rho: float = 50.0,
) -> tuple[torch.Tensor, float]:
    """Smooth lower bound of ``min det(dT/dx)`` over the design domain (fold-over guard).

    Samples ``det(I + dD/dxi * scale)`` on a uniform grid of the normalized design box
    and aggregates with a Kreisselmeier-Steinhauser minimum
    (``-logsumexp(-rho * det) / rho``, always <= the true sampled minimum).  Returns
    the differentiable KS value and the raw sampled minimum for logging.
    """
    dev, dt = deformation.device, deformation.dtype
    box = deformation.to_norm(deformation.design_domain)
    axes = [
        torch.linspace(float(box[0, i]), float(box[1, i]), n_samples_per_dim, device=dev, dtype=dt)
        for i in range(3)
    ]
    gx, gy, gz = torch.meshgrid(*axes, indexing="ij")
    xi = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=1).requires_grad_(True)
    disp = deformation.disp(xi)
    jac = torch.stack(
        [torch.autograd.grad(disp[:, i].sum(), xi, create_graph=True)[0] for i in range(3)],
        dim=1,
    )  # (P, 3, 3): d disp_i / d xi_j
    jacobian = torch.eye(3, device=dev, dtype=dt)[None] + jac * deformation.scale
    det = torch.linalg.det(jacobian)
    ks = -torch.logsumexp(-ks_rho * det, dim=0) / ks_rho
    return ks, float(det.detach().min())


def crossed_design_faces(vertices, faces, design_domain, tol: float = 1e-9) -> list[str]:
    """Design-box faces that the triangle mesh passes through.

    A face is crossed when triangles meeting the face plane inside the face
    rectangle (straddling it, or touching it with a vertex / edge on the plane)
    continue on *both* sides of the plane.  Those faces need their FFD boundary
    control layer locked, otherwise the deformed design surface tears away from
    the fixed geometry outside the box.
    """
    verts = np.asarray(vertices, dtype=np.float64)
    tris = verts[np.asarray(faces, dtype=np.int64)]  # (T, 3, 3)
    dd = np.asarray(design_domain, dtype=np.float64).reshape(2, 3)
    crossed = []
    for axis in range(3):
        others = [i for i in range(3) if i != axis]
        for side in range(2):
            value = dd[side, axis]
            s = tris[:, :, axis] - value
            smin, smax = s.min(axis=1), s.max(axis=1)
            touch = (smin <= tol) & (smax >= -tol)
            if not np.any(touch):
                continue
            t, st = tris[touch], s[touch]
            ids = np.arange(len(t))
            # Intersection points with the plane: on-plane vertices + crossing edges,
            # each tagged with the triangle it belongs to.
            pts, owner = [], []
            for v in range(3):
                on = np.abs(st[:, v]) <= tol
                pts.append(t[on, v])
                owner.append(ids[on])
            for a, b in ((0, 1), (1, 2), (2, 0)):
                sa, sb = st[:, a], st[:, b]
                cross = (sa * sb) < 0.0
                w = sa[cross] / (sa[cross] - sb[cross])
                pts.append(t[cross, a] + w[:, None] * (t[cross, b] - t[cross, a]))
                owner.append(ids[cross])
            pts = np.concatenate(pts, axis=0)
            owner = np.concatenate(owner, axis=0)
            inside = np.all(
                (pts[:, others] >= dd[0, others] - tol) & (pts[:, others] <= dd[1, others] + tol),
                axis=1,
            )
            hit = np.unique(owner[inside])
            if hit.size == 0:
                continue
            has_plus = np.any(smax[touch][hit] > tol)
            has_minus = np.any(smin[touch][hit] < -tol)
            if has_plus and has_minus:
                crossed.append(f"{'xyz'[axis]}_{'min' if side == 0 else 'max'}")
    return crossed


def face_layer_indices(spline_sp: splinepy.BSpline, face: str) -> np.ndarray:
    """F-order control-point ids of the boundary layer on one design-box face."""
    axis, side = _parse_face(face)
    _, dims = greville_points_3d(spline_sp, order="F")
    dims = tuple(int(d) for d in dims)
    ids = np.arange(int(np.prod(dims)))
    ijk = np.stack(np.unravel_index(ids, dims, order="F"), axis=1)
    layer = 0 if side == 0 else dims[axis] - 1
    return ids[ijk[:, axis] == layer]


def assert_face_layers_locked(faces, spline_sp: splinepy.BSpline, mask_locked_cp) -> None:
    """Raise unless every control point on each of ``faces`` is locked."""
    mask = mask_locked_cp.detach().cpu().numpy() if torch.is_tensor(mask_locked_cp) else mask_locked_cp
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    for face in faces:
        ids = face_layer_indices(spline_sp, face)
        n_free = int((~mask[ids]).sum())
        if n_free:
            raise ValueError(
                f"The wall geometry crosses the design-domain face '{face}' but {n_free} of its "
                f"{len(ids)} boundary control points are free: the FFD would tear the wall off "
                f"the fixed geometry at that face. Add '{face}' to parametrization.lock.faces."
            )

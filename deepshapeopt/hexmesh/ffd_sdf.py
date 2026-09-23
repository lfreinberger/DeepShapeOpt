"""FFD-deformed wall geometry as the design SDF of the hex mesh pipeline.

The design surface is the original wall mesh with its vertices inside the design
domain moved by a free-form deformation (``deepshapeopt.parametrization.ffd.FFDDeformation``):
``V_def = V + D(V)``.  :class:`FFDMeshSDF` is a :class:`TriMeshSDF` on that
deformed mesh (all detached queries -- castellation, Newton snap steps, sign --
run igl on the deformed numpy vertices) plus a ``phi_ext`` whose *value* is the
exact signed distance and whose *autograd graph* reaches the control points:

    phi(x; c) = s |x - C(c)|,   C = closest wall point = sum_i b_i V_def[tri_i](c)

so ``d phi / dc = -n_hat . dC/dc`` (envelope theorem: tangential sliding of the
closest point drops out to first order), the Hadamard normal velocity the snap
expects.  Near the surface the direction ``(x - C)/|x - C|`` is round-off noise,
so within ``near_eps`` the oriented face normal of the closest triangle is used
instead (and the sign is taken from that normal, not from the winding number,
which is ambiguous on the surface).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import trimesh

from .sdf_field import check_sign_convention
from .trimesh_sdf import TriMeshSDF

logger = logging.getLogger(__name__)


def barycentric_coordinates(points: np.ndarray, triangles: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    """Barycentric weights ``(n, 3)`` of ``points (n, 3)`` w.r.t. ``triangles (n, 3, 3)``.

    Points are assumed to lie in their triangle (closest points from igl).
    Degenerate triangles fall back to the nearest vertex.
    """
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    v0, v1, v2 = b - a, c - a, points - a
    d00 = (v0 * v0).sum(1)
    d01 = (v0 * v1).sum(1)
    d11 = (v1 * v1).sum(1)
    d20 = (v2 * v0).sum(1)
    d21 = (v2 * v1).sum(1)
    denom = d00 * d11 - d01 * d01
    ok = denom > eps
    safe = np.where(ok, denom, 1.0)
    v = np.where(ok, (d11 * d20 - d01 * d21) / safe, 0.0)
    w = np.where(ok, (d00 * d21 - d01 * d20) / safe, 0.0)
    bary = np.stack([1.0 - v - w, v, w], axis=1)
    if not np.all(ok):
        bad = ~ok
        dist = np.linalg.norm(triangles[bad] - points[bad][:, None, :], axis=2)
        nearest = np.zeros((int(bad.sum()), 3))
        nearest[np.arange(len(nearest)), dist.argmin(axis=1)] = 1.0
        bary[bad] = nearest
    return bary


class FFDMeshSDF(TriMeshSDF):
    """Signed distance to the FFD-deformed wall mesh, differentiable in the control points.

    Parameters
    ----------
    base : TriMeshSDF
        The undeformed geometry (faces, wall-face subset, fluid side, device).
    vertices_def : torch.Tensor (V, 3)
        Deformed vertex positions carrying the autograd graph to the design
        parameters.
    design_domain : (2, 3)
        Physical design box (needed by :class:`CompositeSDF` for routing).
    near_eps : float
        Distance (mesh units) below which a point counts as on the surface: the
        gradient direction and the sign then come from the closest triangle's
        oriented normal.
    """

    def __init__(
        self,
        base: TriMeshSDF,
        vertices_def: torch.Tensor,
        design_domain,
        near_eps: float = 1e-3,
    ):
        super().__init__(
            vertices_def.detach().cpu().numpy(),
            base.faces,
            wall_faces=base.wall_faces,
            fluid_side=base.fluid_side,
            device=base.device,
        )
        self._v_def = vertices_def
        self.near_eps = float(near_eps)
        dd = design_domain.detach().cpu().numpy() if torch.is_tensor(design_domain) else design_domain
        self.design_domain = torch.as_tensor(
            np.asarray(dd, dtype=np.float32).reshape(2, 3), device=self.device
        )
        # grad phi points into the fluid: -outward normal when the fluid is inside.
        self._n_sign = -1.0 if self.fluid_side == "inside" else 1.0
        self._wall_normals: np.ndarray | None = None

    # ------------------------------------------------------------------

    def _wall_face_normals(self) -> np.ndarray:
        """Unit normals of the (deformed) wall triangles, mesh winding orientation."""
        if self._wall_normals is None:
            t = self.vertices[self.wall_faces]
            n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
            n /= np.maximum(np.linalg.norm(n, axis=1), 1e-300)[:, None]
            self._wall_normals = n
        return self._wall_normals

    def _grad_np(self, points: np.ndarray):
        """Signed distance, gradient, closest points and closest wall-face ids."""
        points = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 3)
        d, closest, fidx = self._closest(points)
        s = self._sign(points)
        g = s[:, None] * (points - closest) / np.maximum(d, 1e-300)[:, None]
        near = d <= self.near_eps
        if np.any(near):
            n_face = self._n_sign * self._wall_face_normals()[fidx[near]]
            side = ((points[near] - closest[near]) * n_face).sum(axis=1)
            s[near] = np.where(side >= 0.0, 1.0, -1.0)
            g[near] = n_face
        return s * d, g, closest, fidx

    def phi_and_grad_np(self, points: np.ndarray, grad_eps: float = 1e-12):
        f, g, _, _ = self._grad_np(points)
        return f, g

    def phi_ext(self, x_phys: torch.Tensor) -> torch.Tensor:
        """Exact signed distance whose graph reaches the control points (see module doc)."""
        f_det, g, closest, fidx = self._grad_np(x_phys.detach().cpu().numpy())
        tri = self.wall_faces[fidx]
        bary = barycentric_coordinates(closest, self.vertices[tri])
        v_def = self._v_def
        tri_t = torch.as_tensor(tri, dtype=torch.long, device=v_def.device)
        b_t = torch.as_tensor(bary, dtype=v_def.dtype, device=v_def.device)
        x_param = (b_t[:, :, None] * v_def[tri_t]).sum(dim=1)  # (P, 3), carries the graph
        g_t = torch.as_tensor(g, dtype=v_def.dtype, device=v_def.device)
        f = torch.as_tensor(f_det, dtype=v_def.dtype, device=v_def.device)
        f = f - (g_t * (x_param - x_param.detach())).sum(dim=1)
        return f.to(dtype=x_phys.dtype, device=x_phys.device)

    def check_sign_convention(self, probe_point, expect: str = "solid") -> None:
        check_sign_convention(self, probe_point, expect=expect)


class FFDDesignSDF:
    """:class:`~deepshapeopt.hexmesh.design.DesignSDF` for the FFD parametrization.

    Deforms a triangle mesh with ``deformation`` inside the design domain and returns an
    :class:`FFDMeshSDF` on the result. For internal flow the mesh is the pipeline's fixed
    outer geometry (``outer``, the :class:`TriMeshSDF` of the wall mesh); for external flow
    it is the design object itself (``base_mesh``, the input mesh, fluid outside). The mesh
    must be consistently oriented with outward normals (checked once).
    """

    def __init__(self, deformation, frame, base_mesh=None):
        self.deformation = deformation
        self.frame = frame
        self.base_mesh = base_mesh
        self._object: TriMeshSDF | None = None
        self._base_key: int | None = None
        self._v0: torch.Tensor | None = None
        self._idx_in: torch.Tensor | None = None

    def _object_sdf(self, device) -> TriMeshSDF:
        """The design object as a closed mesh with the fluid outside (external flow)."""
        if self._object is None:
            if self.base_mesh is None:
                raise ValueError("FFD external flow needs the input mesh as base_mesh")
            self._object = TriMeshSDF(
                np.asarray(self.base_mesh.vertices, dtype=np.float64),
                np.asarray(self.base_mesh.faces, dtype=np.int64),
                fluid_side="outside", device=device,
            )
        return self._object

    def _base_vertices(self, outer: TriMeshSDF) -> tuple[torch.Tensor, torch.Tensor]:
        if self._base_key != id(outer):
            mesh = trimesh.Trimesh(outer.vertices, outer.faces, process=False)
            if not mesh.is_winding_consistent or mesh.volume <= 0.0:
                raise ValueError(
                    "FFD design geometry needs a consistently oriented wall mesh with "
                    f"outward normals (winding_consistent={mesh.is_winding_consistent}, "
                    f"volume={mesh.volume:.4g})."
                )
            v0 = torch.as_tensor(
                outer.vertices, dtype=self.deformation.dtype, device=self.deformation.device
            )
            dd = torch.as_tensor(self.frame.design_domain).detach().to(v0)
            inside = torch.all((v0 >= dd[0]) & (v0 <= dd[1]), dim=1)
            self._idx_in = inside.nonzero(as_tuple=True)[0]
            self._v0 = v0
            self._base_key = id(outer)
            logger.info(
                "FFD design geometry: %d of %d wall-mesh vertices inside the design domain",
                int(self._idx_in.numel()), int(v0.shape[0]),
            )
        return self._v0, self._idx_in

    def make_sdf(self, *, sign: float, device, outer: Any | None) -> FFDMeshSDF:
        if outer is None:
            outer = self._object_sdf(device)
        expected = -1.0 if outer.fluid_side == "inside" else 1.0
        if float(sign) != expected:
            raise ValueError(
                f"sign {sign} does not match the outer geometry's fluid_side {outer.fluid_side!r}"
            )
        v0, idx_in = self._base_vertices(outer)
        disp = torch.zeros_like(v0).index_put(
            (idx_in,), self.deformation.displacement(v0[idx_in])
        )
        return FFDMeshSDF(outer, v0 + disp, self.frame.design_domain)

    def float32_scope(self, fn):
        return fn()

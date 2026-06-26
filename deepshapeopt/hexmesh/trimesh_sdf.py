"""Static signed-distance adapter to a watertight triangle mesh.

Provides the fixed (non-design) geometry of internal-flow cases with the
same query interface as :class:`~deepshapeopt.hexmesh.sdf_field.PhysicalSDF`,
so castellation and snapping run unchanged against an STL.  There is no
SDF-grid approximation: every query is an exact signed distance to the
triangle mesh (winding-number sign, closest-point distance), and the Newton
step of the snap lands on the exact closest surface point.

Sign convention matches the pipeline: ``phi > 0`` in the fluid.  For
internal flow (``fluid_side="inside"``) the mesh interior is fluid.

Distances (and snap projections) are measured against a *wall-only* face
subset: the inlet/outlet cap triangles are excluded, so mid-channel points
near a cap plane see the (distant) channel wall rather than the cap -- no
spurious refinement band at the caps and no snapping onto them.  The
inside/outside sign still uses the full closed mesh.
"""

from __future__ import annotations

import hashlib
import logging

import igl
import numpy as np
import torch

logger = logging.getLogger(__name__)


class TriMeshSDF:
    """Exact signed distance to a fixed triangle mesh (``phi > 0`` = fluid).

    Parameters
    ----------
    vertices : [V, 3] float
        Mesh vertex positions (physical coordinates).
    faces : [F, 3] int
        Triangle indices of the full closed mesh (used for the sign).
    wall_faces : [Fw, 3] int, optional
        Triangle subset used for distances and closest points; defaults to
        all faces.  Use :meth:`from_trimesh` to drop the cap triangles.
    fluid_side : "inside" | "outside"
        Which side of the surface is fluid (``phi > 0``).
    device : torch device for the tensor wrappers.
    """

    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        wall_faces: np.ndarray | None = None,
        fluid_side: str = "inside",
        device: str | torch.device = "cpu",
    ):
        if fluid_side not in ("inside", "outside"):
            raise ValueError(f"fluid_side must be 'inside' or 'outside', got {fluid_side!r}")
        self.vertices = np.ascontiguousarray(vertices, dtype=np.float64)
        self.faces = np.ascontiguousarray(faces, dtype=np.int64)
        self.wall_faces = (
            self.faces
            if wall_faces is None
            else np.ascontiguousarray(wall_faces, dtype=np.int64)
        )
        if len(self.wall_faces) == 0:
            raise ValueError("wall_faces is empty")
        self.fluid_side = fluid_side
        self.device = torch.device(device)

    @staticmethod
    def from_trimesh(
        mesh,
        cap_axis: int = 0,
        cap_tol: float = 1e-4,
        fluid_side: str = "inside",
        device: str | torch.device = "cpu",
    ) -> "TriMeshSDF":
        """Build from a ``trimesh.Trimesh``, dropping the cap triangles.

        Caps are the triangles whose centroid lies within ``cap_tol`` of the
        mesh's min/max plane along ``cap_axis`` (the inlet/outlet planes of
        an extrusion channel).
        """
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        centroids = vertices[faces].mean(axis=1)
        lo = vertices[:, cap_axis].min()
        hi = vertices[:, cap_axis].max()
        is_cap = (np.abs(centroids[:, cap_axis] - lo) < cap_tol) | (
            np.abs(centroids[:, cap_axis] - hi) < cap_tol
        )
        wall_faces = faces[~is_cap]
        logger.info(
            "TriMeshSDF: %d triangles (%d wall, %d cap along axis %d), fluid %s",
            len(faces), len(wall_faces), int(is_cap.sum()), cap_axis, fluid_side,
        )
        return TriMeshSDF(
            vertices, faces, wall_faces=wall_faces, fluid_side=fluid_side, device=device
        )

    # ------------------------------------------------------------------
    # Core numpy evaluation
    # ------------------------------------------------------------------

    def _sign(self, points: np.ndarray) -> np.ndarray:
        """+1 in the fluid, -1 in the solid (winding number on the closed mesh)."""
        s_full, _, _, _ = igl.signed_distance(
            points, self.vertices, self.faces,
            sign_type=igl.SignedDistanceType.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER,
        )
        inside = s_full <= 0.0
        if self.fluid_side == "inside":
            return np.where(inside, 1.0, -1.0)
        return np.where(inside, -1.0, 1.0)

    def _dist_and_closest(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Unsigned distance to the wall triangles and the closest points."""
        sq_d, _, closest = igl.point_mesh_squared_distance(
            points, self.vertices, self.wall_faces
        )
        return np.sqrt(np.maximum(sq_d, 0.0)), closest

    def phi_np(self, points: np.ndarray, chunk: int = 262144) -> np.ndarray:
        """Signed wall distance at numpy points [N, 3] -> float64 [N]."""
        points = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 3)
        out = np.empty(len(points), dtype=np.float64)
        for start in range(0, len(points), chunk):
            p = points[start : start + chunk]
            d, _ = self._dist_and_closest(p)
            out[start : start + chunk] = self._sign(p) * d
        return out

    def phi_and_grad_np(
        self, points: np.ndarray, grad_eps: float = 1e-12
    ) -> tuple[np.ndarray, np.ndarray]:
        """Signed wall distance and its (unit) spatial gradient.

        ``grad phi = s * (x - C) / |x - C|`` with ``C`` the closest wall
        point; one Newton step ``x - phi * g / |g|^2`` lands exactly on
        ``C``.  Points on the surface get a zero gradient (no movement).
        """
        points = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 3)
        d, closest = self._dist_and_closest(points)
        s = self._sign(points)
        g = (points - closest) / np.maximum(d, grad_eps)[:, None] * s[:, None]
        g[d <= grad_eps] = 0.0
        return s * d, g

    # ------------------------------------------------------------------
    # Tensor wrappers (PhysicalSDF-compatible, no autograd graph)
    # ------------------------------------------------------------------

    def phi(self, x_phys: torch.Tensor) -> torch.Tensor:
        vals = self.phi_np(x_phys.detach().cpu().numpy())
        return torch.as_tensor(vals, dtype=x_phys.dtype, device=x_phys.device)

    # The zero level set IS the geometry everywhere; no extension needed.
    phi_ext = phi

    def phi_and_grad(self, x_phys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f, g = self.phi_and_grad_np(x_phys.detach().cpu().numpy())
        return (
            torch.as_tensor(f, dtype=x_phys.dtype, device=x_phys.device),
            torch.as_tensor(g, dtype=x_phys.dtype, device=x_phys.device),
        )

    def phi_ext_np(self, points: np.ndarray, chunk: int = 262144) -> np.ndarray:
        return self.phi_np(points, chunk=chunk)

    # ------------------------------------------------------------------

    def content_hash(self) -> str:
        """Identifies the geometry for static-mesh cache keys."""
        h = hashlib.sha256()
        h.update(self.vertices.tobytes())
        h.update(self.faces.tobytes())
        h.update(self.wall_faces.tobytes())
        h.update(self.fluid_side.encode())
        return h.hexdigest()[:16]

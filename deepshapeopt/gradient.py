"""Shape gradient: wall-point sensitivities pulled back to the design parameters."""
from __future__ import annotations

import torch


def vertex_normals(verts: torch.Tensor, faces: torch.Tensor, invert_normals: bool) -> torch.Tensor:
    """Return unit vertex normals (direction only, no area weighting)."""
    V = verts.shape[0]
    device = verts.device
    faces = faces.to(torch.long).to(device)
    v0 = verts[faces[:,0]]
    v1 = verts[faces[:,1]]
    v2 = verts[faces[:,2]]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)

    vn = torch.zeros((V,3), device=device, dtype=verts.dtype)
    vn.index_add_(0, faces[:,0], face_normals)
    vn.index_add_(0, faces[:,1], face_normals)
    vn.index_add_(0, faces[:,2], face_normals)

    n = torch.nn.functional.normalize(vn, dim=1, eps=1e-12)
    if invert_normals: # true for exterior drag optimization, false for internal-flow objectives
        return -n
    else:
        return n


def shape_gradient(param: torch.Tensor, verts: torch.Tensor, sensitivities) -> torch.Tensor:
    """``dJ/dparam`` from integrated wall-point sensitivities ``dJ/dx`` ([N, 3]).

    ``verts`` must carry the autograd graph from ``param`` (the snapped wall points in the
    solver's length unit); the product is one vector-Jacobian product through that graph.
    """
    s = torch.as_tensor(sensitivities, dtype=verts.dtype, device=verts.device).reshape(-1)
    (grad,) = torch.autograd.grad(
        outputs=verts.reshape(-1), inputs=param, grad_outputs=s, retain_graph=True,
    )
    return grad

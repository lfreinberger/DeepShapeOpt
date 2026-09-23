"""FFD parametrization on the SDF hex mesh pipeline (no OpenFOAM, no DeepSDF).

Synthetic internal-flow layout shared with test_hexmesh_internal.py (straight
square channel, fluid inside, caps on the domain x faces).  The design surface
is the channel's wall mesh inside the design domain moved by an FFD
displacement spline.  The channel crosses the box's x faces, so the i = 0 and
i = n-1 control layers are never moved here (what ``lock_domain.faces:
["x_min", "x_max"]`` enforces in the driver).
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import trimesh

from deepshapeopt.domain_frame import DomainFrame
from deepshapeopt.ffd import (
    assert_face_layers_locked,
    build_ffd_deformation,
    crossed_design_faces,
    face_layer_indices,
    min_jacobian_det_ks,
)
from deepshapeopt.hexmesh.ffd_sdf import FFDDesignSDF, FFDMeshSDF, barycentric_coordinates
from deepshapeopt.hexmesh.pipeline import SdfHexMeshPipeline
from deepshapeopt.hexmesh.polymesh import face_pyramid_volumes
from deepshapeopt.hexmesh.sdf_field import CompositeSDF
from deepshapeopt.hexmesh.trimesh_sdf import TriMeshSDF
from deepshapeopt.parameters import greville_points_3d
from test_hexmesh_internal import (
    DESIGN_DOMAIN,
    DOMAIN,
    HALF_WIDTH,
    IFACE,
    MAX_LEVEL,
    MESH_BOX,
    make_tube,
)

N_CP = [4, 4, 4]
DEGREE = [2, 2, 2]


def make_fine_tube(levels: int = 5) -> trimesh.Trimesh:
    """The square channel with walls tessellated finely enough that the design
    domain contains vertices for the FFD to move (uniform midpoint subdivision
    keeps the box watertight; 5 levels -> 1.75 mm along x, 0.25 mm across)."""
    mesh = make_tube()
    for _ in range(levels):
        mesh = mesh.subdivide()
    mesh.merge_vertices()
    assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0
    return mesh


def make_setup(device="cpu"):
    frame = DomainFrame.from_design_domain(DESIGN_DOMAIN, device=device)
    ffd = build_ffd_deformation(
        {"n_control_points": N_CP, "spline_degree": DEGREE}, frame, device=device
    )
    return frame, ffd


def cp_index(i: int, j: int, k: int) -> int:
    """F-order control-point id (first parametric axis fastest)."""
    return int(np.ravel_multi_index((i, j, k), tuple(N_CP), order="F"))


def make_ffd_pipeline(tmp_path, mesh, frame, ffd, **cfg_overrides) -> SdfHexMeshPipeline:
    model_setup = SimpleNamespace(
        design_domain=frame.design_domain, frame=frame, mesh_orig=mesh
    )
    sdf_hex = {
        "flow": "internal",
        "domain": DOMAIN,
        "base_cell_size": 1.0,
        "mesh_box": MESH_BOX,
        "interface_level": IFACE,
        "max_level": MAX_LEVEL,
        "static_band_max_level": 2,
        "fluid_side": "inside",
        "seed_point": [10.0, 0.0, 0.0],
        "sign_probe_point": [0.0, 0.0, 0.0],
        "sign_probe_expect": "fluid",
        "patches": {
            "x_min": "outlet", "x_max": "inlet",
            "wall": "walls", "sensitivity": "sensitivity_region",
        },
        "refinement_regions": [{"face": "x_min", "distance": 1.0, "level": 3}],
    }
    sdf_hex.update(cfg_overrides)
    design = FFDDesignSDF(ffd.deformation, frame)
    return SdfHexMeshPipeline(design, model_setup, {"sdf_hex": sdf_hex}, tmp_path)


# ---------------------------------------------------------------------------
# FFD SDF
# ---------------------------------------------------------------------------

def test_barycentric_coordinates_roundtrip():
    rng = np.random.default_rng(0)
    tri = rng.normal(size=(50, 3, 3))
    b = rng.dirichlet(np.ones(3), size=50)
    p = (b[:, :, None] * tri).sum(axis=1)
    assert np.allclose(barycentric_coordinates(p, tri), b, atol=1e-10)


def test_ffd_sdf_zero_displacement_matches_trimesh():
    frame, ffd = make_setup()
    outer = TriMeshSDF.from_trimesh(make_fine_tube(), cap_axis=0, fluid_side="inside")
    sdf = FFDDesignSDF(ffd.deformation, frame).make_sdf(sign=-1.0, device="cpu", outer=outer)
    assert isinstance(sdf, FFDMeshSDF)

    rng = np.random.default_rng(1)
    pts = rng.uniform(DOMAIN[0], DOMAIN[1], size=(500, 3))
    assert np.array_equal(sdf.phi_np(pts), outer.phi_np(pts))

    # Off the surface both gradients agree ...
    near = np.array([[0.0, 3.9, 0.0]])
    f, g = sdf.phi_and_grad_np(near)
    f0, g0 = outer.phi_and_grad_np(near)
    assert f[0] == pytest.approx(0.1) and np.allclose(g, g0)
    # ... on the surface the FFD SDF supplies the oriented (into-the-fluid) normal
    # where TriMeshSDF returns a zero gradient.
    on = np.array([
        [0.0, HALF_WIDTH, 0.0], [0.0, -HALF_WIDTH, 0.0], [0.0, 0.0, HALF_WIDTH],
    ])
    f, g = sdf.phi_and_grad_np(on)
    assert np.allclose(f, 0.0, atol=1e-12)
    assert np.allclose(g, [[0, -1, 0], [0, 1, 0], [0, 0, -1]])

    # Differentiable phi_ext: exact values (in the query dtype) with a graph.
    x = torch.tensor(pts[:20], dtype=torch.float32)
    fe = sdf.phi_ext(x)
    assert fe.dtype == torch.float32 and fe.requires_grad
    assert np.allclose(fe.detach().numpy(), outer.phi_np(pts[:20]), atol=1e-5)


def test_ffd_sdf_parameter_gradient_fd():
    """d phi / d control point from phi_ext == central differences of the
    signed distance to the deformed mesh."""
    frame, ffd = make_setup()
    outer = TriMeshSDF.from_trimesh(make_fine_tube(), cap_axis=0, fluid_side="inside")
    design = FFDDesignSDF(ffd.deformation, frame)
    cp = ffd.deformation.control_points
    with torch.no_grad():
        cp[cp_index(2, 2, 1), 2] = 0.3  # a generic (non-identity) state
    idx, comp = cp_index(1, 1, 2), 1

    # Fluid points near the +y wall inside the design box, one far outside it.
    x = torch.tensor(
        [[-2.0, 3.5, 0.5], [-3.0, 3.0, -1.0], [0.0, 3.8, 2.0], [20.0, 3.5, 0.0]],
        dtype=torch.float32,
    )
    f = design.make_sdf(sign=-1.0, device="cpu", outer=outer).phi_ext(x)
    dphi = torch.stack([
        torch.autograd.grad(f[i], cp, retain_graph=True)[0][idx, comp] for i in range(len(x))
    ]).numpy()

    eps = 1e-3

    def phi_at(delta):
        with torch.no_grad():
            cp[idx, comp] += delta
        try:
            sdf = design.make_sdf(sign=-1.0, device="cpu", outer=outer)
            return sdf.phi_np(x.numpy().astype(np.float64))
        finally:
            with torch.no_grad():
                cp[idx, comp] -= delta

    fd = (phi_at(eps) - phi_at(-eps)) / (2 * eps)
    assert np.abs(dphi[:3]).max() > 1e-3  # the control point does move these walls
    assert np.allclose(dphi, fd, atol=1e-6, rtol=2e-2), f"autograd {dphi} vs FD {fd}"
    assert dphi[3] == 0.0 and fd[3] == 0.0  # outside the box: undeformed


def test_crossed_faces_and_lock_assertion():
    frame, ffd = make_setup()
    mesh = make_fine_tube()
    assert crossed_design_faces(mesh.vertices, mesh.faces, DESIGN_DOMAIN) == ["x_min", "x_max"]

    spline = ffd.disp_spline_sp
    greville, _ = greville_points_3d(spline, order="F")
    ids = face_layer_indices(spline, "x_min")
    assert np.allclose(greville[ids, 0], frame.box_norm[0, 0].item())
    ids = face_layer_indices(spline, "x_max")
    assert np.allclose(greville[ids, 0], frame.box_norm[1, 0].item())

    n = int(np.prod(N_CP))
    free = torch.zeros(n, dtype=torch.bool)
    with pytest.raises(ValueError, match="x_min"):
        assert_face_layers_locked(["x_min", "x_max"], spline, free)
    locked = free.clone()
    for face in ("x_min", "x_max"):
        locked[face_layer_indices(spline, face)] = True
    assert int(locked.sum()) == 2 * N_CP[1] * N_CP[2]
    assert_face_layers_locked(["x_min", "x_max"], spline, locked)


def test_min_jacobian_det_ks():
    frame, ffd = make_setup()
    cp = ffd.deformation.control_points
    ks, det_min = min_jacobian_det_ks(ffd.deformation, n_samples_per_dim=4, ks_rho=50.0)
    assert det_min == pytest.approx(1.0)
    assert ks.item() == pytest.approx(1.0 - np.log(64) / 50.0, abs=1e-6)
    (dks,) = torch.autograd.grad(ks, cp)
    assert torch.isfinite(dks).all()

    with torch.no_grad():
        cp[cp_index(1, 1, 1), 0] = -30.0  # squash the box along x: fold-over
    ks2, det_min2 = min_jacobian_det_ks(ffd.deformation, n_samples_per_dim=6, ks_rho=50.0)
    assert det_min2 < 0.0 and ks2.item() < 0.2


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def test_ffd_pipeline_build_and_deform(tmp_path):
    frame, ffd = make_setup()
    cp = ffd.deformation.control_points
    pipe = make_ffd_pipeline(tmp_path, make_fine_tube(), frame, ffd)
    res = pipe.build()

    names = [p.name for p in res.mesh.patches]
    assert names == ["outlet", "inlet", "walls", "sensitivity_region"]
    assert all(p.n_faces > 0 for p in res.mesh.patches)
    pyr_o, pyr_n = face_pyramid_volumes(res.mesh)
    assert pyr_o.min() > 0 and pyr_n.min() > 0

    g = torch.randn_like(res.surface_points)
    (dJ,) = torch.autograd.grad(
        res.surface_points, cp, grad_outputs=g, retain_graph=True
    )
    assert torch.isfinite(dJ).all() and dJ.abs().sum() > 0

    # Moving an interior control point moves the design wall; points well outside
    # the box (beyond the smoothing stencil) are untouched.
    x0 = res.surface_points.detach().numpy()
    with torch.no_grad():
        cp[cp_index(1, 2, 1), 1] = 0.4
    res2 = pipe.build(reuse_castellation=True)
    x1 = res2.surface_points.detach().numpy()
    dd = np.asarray(DESIGN_DOMAIN)
    inside = np.all((x0 >= dd[0] + 0.5) & (x0 <= dd[1] - 0.5), axis=1)
    far_outside = ~np.all((x0 >= dd[0] - 3.0) & (x0 <= dd[1] + 3.0), axis=1)
    assert inside.sum() > 0 and far_outside.sum() > 0
    assert np.abs(x1 - x0)[inside].max() > 0.05
    assert np.abs(x1 - x0)[far_outside].max() == 0.0


def test_ffd_pipeline_gradient_fd(tmp_path):
    # Central-difference check through the differentiable snap (see the
    # matching DeepSDF test in test_hexmesh_internal.py for the rationale of
    # the normal-directed functional and the excluded points).
    torch.manual_seed(0)
    frame, ffd = make_setup()
    cp = ffd.deformation.control_points
    pipe = make_ffd_pipeline(tmp_path, make_fine_tube(), frame, ffd)
    pipe.build()  # castellation at the reference state
    idx, comp, eps = cp_index(1, 1, 2), 1, 1e-2

    def build_at(delta):
        with torch.no_grad():
            cp[idx, comp] += delta
        try:
            return pipe.build(reuse_castellation=True)
        finally:
            with torch.no_grad():
                cp[idx, comp] -= delta

    res_p, res_m, res = build_at(eps), build_at(-eps), build_at(0.0)

    x0 = res.surface_points.detach()
    x0_np = x0.numpy()
    dd = np.asarray(DESIGN_DOMAIN)
    in_dd = np.all((x0_np >= dd[0] + 0.3) & (x0_np <= dd[1] - 0.3), axis=1)
    off_corner = np.abs(np.abs(x0_np[:, 1]) - np.abs(x0_np[:, 2])) > 0.5
    full = (
        in_dd
        & off_corner
        & (res.snap_lambda >= 0.999)
        & (res_p.snap_lambda >= 0.999)
        & (res_m.snap_lambda >= 0.999)
    )
    assert full.sum() > 100

    comp_sdf = CompositeSDF(
        pipe.design.make_sdf(sign=-1.0, device="cpu", outer=pipe.geom), pipe.geom
    )
    _, gphi = comp_sdf.phi_and_grad(x0)
    n_hat = gphi / gphi.norm(dim=1, keepdim=True).clamp_min(1e-12)
    g = torch.randn(len(x0))[:, None] * n_hat
    g[torch.as_tensor(~full)] = 0.0

    (dJ,) = torch.autograd.grad((g * res.surface_points).sum(), cp, retain_graph=True)
    fd = (
        (g * res_p.surface_points).sum() - (g * res_m.surface_points).sum()
    ).item() / (2 * eps)
    adj = dJ[idx, comp].item()
    assert abs(adj - fd) / abs(fd) < 0.05, f"adjoint {adj} vs FD {fd}"

    # Points clearly outside the design domain are STL-routed: exactly zero
    # parameter gradient.
    near_dd = np.all((x0_np >= dd[0] - 0.5) & (x0_np <= dd[1] + 0.5), axis=1)
    assert np.any(~near_dd)
    g_stl = torch.randn_like(res.surface_points)
    g_stl[torch.as_tensor(near_dd)] = 0.0
    (dJ_stl,) = torch.autograd.grad(
        (g_stl * res.surface_points).sum(), cp, retain_graph=True
    )
    assert dJ_stl.abs().sum().item() == 0.0

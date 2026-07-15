"""Internal-flow tests for the SDF hex mesh pipeline (no OpenFOAM, no DeepSDF).

Synthetic setup mirroring the extrusion-die layout: a straight square-channel
STL (fluid inside) whose outlet/inlet cap planes coincide with the domain
x faces, and an analytic "DeepSDF" insert in the design domain that matches
the channel at the design-domain faces (mimicking locked boundary control
points) and bulges in between.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import trimesh

from deepshapeopt.hexmesh.lattice import CellIndex, Lattice
from deepshapeopt.hexmesh.octree import (
    MeshBox,
    build_inner_castellation,
    build_static_castellation,
    parse_refine_regions,
)
from deepshapeopt.hexmesh.patches import outlet_strip_classifier
from deepshapeopt.hexmesh.pipeline import SdfHexMeshPipeline
from deepshapeopt.hexmesh.polymesh import face_pyramid_volumes
from deepshapeopt.hexmesh.sdf_field import CompositeSDF, PhysicalSDF
from deepshapeopt.hexmesh.trimesh_sdf import TriMeshSDF

# Domain/channel layout (lengths in "mm"): channel x in [-6, 50], square
# cross-section half-width 4; design domain strictly inside the mesh box,
# whose x_min face coincides with the domain (outlet) plane.
DOMAIN = [[-6.0, -21.0, -21.0], [50.0, 21.0, 21.0]]
MESH_BOX = [[-6.0, -13.0, -13.0], [2.0, 13.0, 13.0]]
DESIGN_DOMAIN = [[-5.5, -12.0, -12.0], [1.0, 12.0, 12.0]]
IFACE = 2
MAX_LEVEL = 3
HALF_WIDTH = 4.0


def make_lattice() -> Lattice:
    return Lattice(origin=DOMAIN[0], h0=1.0, root_dims=[56, 42, 42], max_depth=MAX_LEVEL)


def make_tube() -> trimesh.Trimesh:
    tube = trimesh.creation.box(extents=[56.0, 2 * HALF_WIDTH, 2 * HALF_WIDTH])
    tube.apply_translation([22.0, 0.0, 0.0])
    return tube


def make_tube_span(x0: float, x1: float) -> trimesh.Trimesh:
    """Square channel spanning [x0, x1] in x (ends off the domain box make
    interior caps)."""
    tube = trimesh.creation.box(extents=[x1 - x0, 2 * HALF_WIDTH, 2 * HALF_WIDTH])
    tube.apply_translation([0.5 * (x0 + x1), 0.0, 0.0])
    return tube


def make_insert_sdf(r: torch.Tensor) -> PhysicalSDF:
    """Square channel of half-width ``r`` with a bulge vanishing at the
    design-domain x faces; raw DeepSDF convention (negative inside the
    channel), so sign=-1 makes the fluid positive."""

    def fn(x):
        bulge = 0.3 * torch.exp(-(((x[:, 0] + 2.25) / 1.5) ** 2))
        return torch.maximum(x[:, 1].abs(), x[:, 2].abs()) - (r + bulge)

    return PhysicalSDF(fn, lambda x: x, DESIGN_DOMAIN, sign=-1.0)


def _check_balance(lattice, cells):
    from deepshapeopt.hexmesh.lattice import face_sample_points_doubled

    index = CellIndex(lattice, cells)
    w = cells.widths(lattice)
    for direction in range(6):
        samples = face_sample_points_doubled(cells.anchors, w, direction)
        nbr = index.locate_doubled(samples.reshape(-1, 3)).reshape(-1, 4)
        src_lvl = np.repeat(cells.levels[:, None], 4, axis=1)
        found = nbr >= 0
        assert np.all(np.abs(cells.levels[nbr[found]] - src_lvl[found]) <= 1)


# ---------------------------------------------------------------------------
# TriMeshSDF
# ---------------------------------------------------------------------------

def test_trimesh_sdf_sign_and_caps():
    g = TriMeshSDF.from_trimesh(make_tube(), cap_axis=0, fluid_side="inside")

    pts = np.array([
        [0.0, 0.0, 0.0],     # mid-channel: fluid, 4 to the nearest wall
        [0.0, 3.9, 0.0],     # near +y wall, inside
        [0.0, 4.5, 0.0],     # outside the channel: solid
        [49.9, 0.0, 0.0],    # mid-channel next to the inlet cap plane
    ])
    phi = g.phi_np(pts)
    assert phi[0] == pytest.approx(4.0)
    assert phi[1] == pytest.approx(0.1)
    assert phi[2] == pytest.approx(-0.5)
    # Cap triangles are excluded from distances: the nearest *wall* is 4
    # away, so there is no spurious zero level set at the cap plane.
    assert phi[3] == pytest.approx(4.0)

    # One Newton step lands exactly on the wall.
    f, grad = g.phi_and_grad_np(pts[1:2])
    x_new = pts[1] - f[0] * grad[0] / (grad[0] @ grad[0])
    assert g.phi_np(x_new[None])[0] == pytest.approx(0.0, abs=1e-12)

    g_out = TriMeshSDF.from_trimesh(make_tube(), fluid_side="outside")
    assert g_out.phi_np(pts[:1])[0] == pytest.approx(-4.0)


def test_trimesh_sdf_torch_wrappers():
    g = TriMeshSDF.from_trimesh(make_tube(), fluid_side="inside")
    x = torch.tensor([[0.0, 3.5, 0.0]], dtype=torch.float32)
    f, grad = g.phi_and_grad(x)
    assert f.item() == pytest.approx(0.5, abs=1e-6)
    assert torch.allclose(grad, torch.tensor([[0.0, -1.0, 0.0]]), atol=1e-6)
    assert g.phi_ext(x).item() == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# Static castellation
# ---------------------------------------------------------------------------

def test_static_castellation_tube():
    lattice = make_lattice()
    box = MeshBox.from_physical(lattice, MESH_BOX)
    g = TriMeshSDF.from_trimesh(make_tube(), fluid_side="inside")

    cast = build_static_castellation(
        lattice, box, IFACE, MAX_LEVEL, g.phi_np,
        beta=1.0, seed_point=[10.0, 0.0, 0.0], band_max_level=2,
    )
    cells = cast.cells
    _check_balance(lattice, cells)

    # All fluid cells lie inside the channel and outside the mesh box.
    centers = cells.centers_phys(lattice)
    assert np.all(np.abs(centers[:, 1]) < HALF_WIDTH)
    assert np.all(np.abs(centers[:, 2]) < HALF_WIDTH)
    assert np.all(centers[:, 0] > MESH_BOX[1][0])  # outer region is x > box

    # The ring adjacent to the box is exactly at interface level.
    assert np.any(cast.shell_mask)
    assert np.all(cells.levels[cast.shell_mask] == IFACE)


def test_static_castellation_bad_seed():
    lattice = make_lattice()
    box = MeshBox.from_physical(lattice, MESH_BOX)
    g = TriMeshSDF.from_trimesh(make_tube(), fluid_side="inside")
    with pytest.raises(ValueError, match="seed_point"):
        build_static_castellation(
            lattice, box, IFACE, MAX_LEVEL, g.phi_np,
            beta=1.0, seed_point=[10.0, 15.0, 15.0],  # in the solid
        )


def test_inner_castellation_internal_regions():
    lattice = make_lattice()
    box = MeshBox.from_physical(lattice, MESH_BOX)
    # x_min coincides with the domain boundary -> not an interface face.
    assert box.interface_faces(lattice).tolist() == [
        [False, True, True], [True, True, True]
    ]

    comp = CompositeSDF(
        make_insert_sdf(torch.tensor(4.0)),
        TriMeshSDF.from_trimesh(make_tube(), fluid_side="inside"),
    )
    regions = parse_refine_regions(
        [{"face": "x_min", "distance": 1.0, "level": 3}], lattice, MAX_LEVEL
    )
    cast = build_inner_castellation(
        lattice, box, IFACE, MAX_LEVEL, comp.phi_ext_np, beta=1.0, regions=regions
    )
    cells = cast.cells
    _check_balance(lattice, cells)
    assert np.all(cells.levels[cast.shell_mask] == IFACE)

    # The refinement region forces level 3 down to the outlet plane.
    centers = cells.centers_phys(lattice)
    slab = centers[:, 0] < DOMAIN[0][0] + 1.0
    assert np.any(slab)
    assert np.all(cells.levels[slab] == 3)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def make_internal_pipeline(
    tmp_path,
    r: torch.Tensor,
    mesh: trimesh.Trimesh | None = None,
    design_domain=None,
    **cfg_overrides,
) -> SdfHexMeshPipeline:
    dd = DESIGN_DOMAIN if design_domain is None else design_domain

    def fn(x):
        bulge = 0.3 * torch.exp(-(((x[:, 0] + 2.25) / 1.5) ** 2))
        return torch.maximum(x[:, 1].abs(), x[:, 2].abs()) - (r + bulge)

    model_setup = SimpleNamespace(
        design_domain=torch.tensor(dd, dtype=torch.float32),
        frame=SimpleNamespace(
            physical_sdf=lambda lattice_struct, sign=1.0, device="cpu": PhysicalSDF(
                lattice_struct, lambda x: x, dd, sign=sign, device=device
            ),
            box_norm=torch.tensor(dd, dtype=torch.float32),
        ),
        mesh_orig=make_tube() if mesh is None else mesh,
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
        "outlet_interior": {
            "enabled": True,
            "method": "polygon_offset",
            "inset_distance": 1.0,
        },
    }
    sdf_hex.update(cfg_overrides)
    pipe = SdfHexMeshPipeline(fn, model_setup, {"sdf_hex": sdf_hex}, tmp_path)
    pipe._with_float32 = lambda f: f()  # no DeepSDFStruct lattice in tests
    return pipe


def test_internal_pipeline_build(tmp_path):
    r = torch.tensor(4.0, requires_grad=True)
    pipe = make_internal_pipeline(tmp_path, r)
    res = pipe.build()

    # Patch layout: empty "sides" dropped, wall-type patches last+contiguous.
    names = [p.name for p in res.mesh.patches]
    assert names == ["outlet", "inlet", "outletInterior", "walls", "sensitivity_region"]
    assert all(p.n_faces > 0 for p in res.mesh.patches)
    types = {p.name: p.type for p in res.mesh.patches}
    assert types["walls"] == "wall" and types["sensitivity_region"] == "wall"
    assert types["outlet"] == "patch"

    # Snapped mesh stays valid.
    pyr_o, pyr_n = face_pyramid_volumes(res.mesh)
    assert pyr_o.min() > 0 and pyr_n.min() > 0

    # Outlet (and its sub-patch) faces lie exactly on the domain x_min plane,
    # and the rim points were not pulled off it by the snap.
    for name in ("outlet", "outletInterior"):
        sl = res.mesh.patch_face_slice(name)
        pts = res.mesh.points[np.unique(res.mesh.faces[sl].ravel())]
        assert np.all(pts[:, 0] == DOMAIN[0][0])

    # Parameter gradient flows through the snap (insert region only).
    g = torch.randn_like(res.surface_points)
    (dJ,) = torch.autograd.grad(
        res.surface_points, r, grad_outputs=g, retain_graph=True
    )
    assert torch.isfinite(dJ)

    # patch_tris_local gives the design-surface subset; all referenced
    # points must be inside the design domain box (up to float32 snap
    # round-off at the box faces).
    tris = res.patch_tris_local("sensitivity_region")
    pts = res.surface_points.detach().numpy()[np.unique(tris.numpy().ravel())]
    dd = np.asarray(DESIGN_DOMAIN)
    assert np.all(pts >= dd[0] - 1e-3) and np.all(pts <= dd[1] + 1e-3)


def test_internal_pipeline_frozen_static_and_shell(tmp_path):
    r = torch.tensor(4.0, requires_grad=True)
    pipe = make_internal_pipeline(tmp_path, r)
    res1 = pipe.build()
    static1 = (pipe.static_cells.levels.copy(), pipe.static_cells.anchors.copy())
    shell_hash1 = pipe._shell_hash

    with torch.no_grad():
        r -= 0.4  # different insert shape
    res2 = pipe.build()

    # Static cells and the interface-shell pattern are bit-identical; the
    # interior castellation changed with the shape.
    assert np.array_equal(static1[0], pipe.static_cells.levels)
    assert np.array_equal(static1[1], pipe.static_cells.anchors)
    assert pipe._shell_hash == shell_hash1
    assert res2.castellation_hash != res1.castellation_hash

    # Boundary faces of the static region are identical across the builds.
    def static_walls(res):
        sl = res.mesh.patch_face_slice("walls")
        faces = res.mesh.faces[sl]
        c = res.mesh.points[faces].mean(axis=1)
        keep = c[:, 0] > MESH_BOX[1][0] + 1.0
        return np.sort(res.mesh.point_keys[faces[keep]].ravel())

    assert np.array_equal(static_walls(res1), static_walls(res2))


def test_outlet_interior_partition(tmp_path):
    r = torch.tensor(4.0, requires_grad=True)
    res_carved = make_internal_pipeline(tmp_path / "a", r).build()
    res_plain = make_internal_pipeline(tmp_path / "b", r, outlet_interior=None).build()

    n_out = res_carved.mesh.patch_by_name("outlet").n_faces
    n_int = res_carved.mesh.patch_by_name("outletInterior").n_faces
    assert n_int > 0 and n_out > 0
    # Carved outlet + interior == plain outlet.
    assert n_out + n_int == res_plain.mesh.patch_by_name("outlet").n_faces

    # The interior faces are strictly inside the cross-section (inset 1.0
    # from the half-width-4 channel walls).
    sl = res_carved.mesh.patch_face_slice("outletInterior")
    c = res_carved.mesh.points[res_carved.mesh.faces[sl]].mean(axis=1)
    assert np.all(np.abs(c[:, 1]) < HALF_WIDTH - 0.9)
    assert np.all(np.abs(c[:, 2]) < HALF_WIDTH - 0.9)


def test_outlet_strip_classifier_medial_axis():
    cls = outlet_strip_classifier(
        make_tube(), plane_axis=0, plane_value=-6.0, plane_tol=1e-4,
        outlet_interior_cfg={
            "method": "medial_axis", "boundary_sample_ds": 0.05,
            "strip_half_width": 0.5, "min_dist_from_boundary": 0.3,
            "prune_branch_len": 1.0,
        },
    )
    pts = np.array([
        [-6.0, 0.0, 0.0],   # on the medial axis
        [-6.0, 3.9, 0.0],   # at the wall
        [-6.0, 0.0, 3.9],
    ])
    assert cls(pts).tolist() == [True, False, False]


def test_internal_pipeline_gradient_fd(tmp_path):
    # Central-difference check through the differentiable snap.  The
    # Hadamard-form autograd derivative captures the *normal* motion of the
    # wall under a parameter change (which is what adjoint sensitivities
    # pair with); tangential point drift (Newton path + smoothing re-run at
    # the perturbed radius) is intentionally not in the graph.  So the test
    # functional must be normal-directed, and points near the channel's
    # corner kinks (discontinuous normal of the max-norm SDF) are excluded.
    torch.manual_seed(0)
    r = torch.tensor(4.0, requires_grad=True)
    pipe = make_internal_pipeline(tmp_path, r)
    res = pipe.build()

    eps = 2e-3
    with torch.no_grad():
        r += eps
    res_p = pipe.build(reuse_castellation=True)
    with torch.no_grad():
        r -= 2 * eps
    res_m = pipe.build(reuse_castellation=True)
    with torch.no_grad():
        r += eps

    x0 = res.surface_points.detach()
    x0_np = x0.numpy()
    dd = np.asarray(DESIGN_DOMAIN)
    in_dd = np.all((x0_np >= dd[0] + 0.3) & (x0_np <= dd[1] - 0.3), axis=1)
    off_corner = (
        np.abs(np.abs(x0_np[:, 1]) - np.abs(x0_np[:, 2])) > 0.5
    )
    full = (
        in_dd
        & off_corner
        & (res.snap_lambda >= 0.999)
        & (res_p.snap_lambda >= 0.999)
        & (res_m.snap_lambda >= 0.999)
    )
    assert full.sum() > 100

    # Normal-directed functional: random weight times the wall normal.
    comp = CompositeSDF(
        make_insert_sdf(r.detach().clone()),
        TriMeshSDF.from_trimesh(make_tube(), fluid_side="inside"),
    )
    _, gphi = comp.phi_and_grad(x0)
    n_hat = gphi / gphi.norm(dim=1, keepdim=True).clamp_min(1e-12)
    g = torch.randn(len(x0))[:, None] * n_hat
    g[torch.as_tensor(~full)] = 0.0

    J0 = (g * res.surface_points).sum()
    (dJ,) = torch.autograd.grad(J0, r, retain_graph=True)
    fd = (
        (g * res_p.surface_points).sum() - (g * res_m.surface_points).sum()
    ).item() / (2 * eps)
    assert abs(dJ.item() - fd) / abs(fd) < 0.05, f"adjoint {dJ.item()} vs FD {fd}"

    # Points clearly outside the design domain (beyond any Newton-iterate
    # drift into it) are STL-routed: exactly zero parameter gradient.
    near_dd = np.all((x0_np >= dd[0] - 0.5) & (x0_np <= dd[1] + 0.5), axis=1)
    assert np.any(~near_dd)
    g_stl = torch.randn_like(res.surface_points)
    g_stl[torch.as_tensor(near_dd)] = 0.0
    (dJ_stl,) = torch.autograd.grad(
        (g_stl * res.surface_points).sum(), r, retain_graph=True, allow_unused=False
    )
    assert dJ_stl.item() == 0.0


# ---------------------------------------------------------------------------
# Explicit caps: interior cap planes, carves, strict patches, validation
# ---------------------------------------------------------------------------

INTERIOR_CAPS = [
    {"axis": "x", "value": -5.8, "patch": "outlet"},
    {"axis": "x", "value": 50.0, "patch": "inlet"},
]
CAPS_PATCHES = {"wall": "walls", "sensitivity": "sensitivity_region"}


def test_trimesh_sdf_drop_planes():
    tube = make_tube()
    legacy = TriMeshSDF.from_trimesh(tube, cap_axis=0, fluid_side="inside")
    explicit = TriMeshSDF.from_trimesh(
        tube, fluid_side="inside",
        drop_planes=[(0, -6.0, 1e-4), (0, 50.0, 1e-4)],
    )
    # Equivalent drop set -> bit-identical wall faces and cache hash.
    assert np.array_equal(legacy.wall_faces, explicit.wall_faces)
    assert legacy.content_hash() == explicit.content_hash()

    # Keeping a cap changes the hash and seals the channel there.
    keep = TriMeshSDF.from_trimesh(
        tube, fluid_side="inside", drop_planes=[(0, 50.0, 1e-4)]
    )
    assert keep.content_hash() != legacy.content_hash()
    p = np.array([[-5.9, 0.0, 0.0]])
    assert keep.phi_np(p)[0] == pytest.approx(0.1)
    assert legacy.phi_np(p)[0] == pytest.approx(4.0)

    with pytest.raises(ValueError, match="No cap triangles"):
        TriMeshSDF.from_trimesh(tube, drop_planes=[(0, -5.0, 1e-4)])


def test_interior_cap_carve_and_snap(tmp_path):
    r = torch.tensor(4.0, requires_grad=True)
    pipe = make_internal_pipeline(
        tmp_path, r,
        mesh=make_tube_span(-5.8, 50.0),
        caps=INTERIOR_CAPS,
        patches=CAPS_PATCHES,
        outlet_interior=None,
    )
    res = pipe.build()

    # x_max cap fills "inlet"; interior cap carves "outlet" before the
    # wall-type patches; empty sides dropped.
    names = [p.name for p in res.mesh.patches]
    assert names == ["inlet", "outlet", "walls", "sensitivity_region"]
    types = {p.name: p.type for p in res.mesh.patches}
    assert types["outlet"] == "patch" and types["inlet"] == "patch"
    assert types["walls"] == "wall"

    # All outlet faces snapped exactly onto the interior cap plane, and the
    # patch covers the full 8x8 cross-section.
    sl = res.mesh.patch_face_slice("outlet")
    out_faces = res.mesh.faces[sl]
    pts = res.mesh.points[np.unique(out_faces.ravel())]
    assert np.all(np.abs(pts[:, 0] - (-5.8)) < 1e-6)
    fp = res.mesh.points[out_faces]
    area = (
        0.5 * np.linalg.norm(np.cross(fp[:, 1] - fp[:, 0], fp[:, 2] - fp[:, 0]), axis=1)
        + 0.5 * np.linalg.norm(np.cross(fp[:, 2] - fp[:, 0], fp[:, 3] - fp[:, 0]), axis=1)
    ).sum()
    assert abs(area - 64.0) / 64.0 < 0.02

    # Cap points are snapped (in surface_point_ids) but not wall-typed.
    out_pts = np.unique(out_faces.ravel())
    assert np.all(np.isin(out_pts, res.surface_point_ids))
    pure_cap = np.setdiff1d(out_pts, res.mesh.wall_point_ids())
    assert len(pure_cap) > 0

    # Snapped mesh stays valid.
    pyr_o, pyr_n = face_pyramid_volumes(res.mesh)
    assert pyr_o.min() > 0 and pyr_n.min() > 0

    # Cap points are STL-routed: exactly zero parameter gradient.
    sel = np.isin(res.surface_point_ids, out_pts)
    g = torch.randn_like(res.surface_points)
    g[torch.as_tensor(~sel)] = 0.0
    (dJ,) = torch.autograd.grad(
        (g * res.surface_points).sum(), r, retain_graph=True
    )
    assert dJ.item() == 0.0

    # patch_tris_local works for the carved (snapped, non-wall) patch.
    tris = res.patch_tris_local("outlet")
    assert len(tris) == 2 * (sl.stop - sl.start)


def test_two_caps_merge_and_static_carve(tmp_path):
    r = torch.tensor(4.0, requires_grad=True)
    pipe = make_internal_pipeline(
        tmp_path, r,
        mesh=make_tube_span(-5.8, 49.3),
        caps=[
            {"axis": "x", "value": -5.8, "patch": "outlet"},
            {"axis": "x", "value": 49.3, "patch": "outlet"},
        ],
        patches=CAPS_PATCHES,
        outlet_interior=None,
    )
    res = pipe.build()

    # Same-name caps merge into one patch; every domain-face patch is empty.
    names = [p.name for p in res.mesh.patches]
    assert names == ["outlet", "walls", "sensitivity_region"]

    sl = res.mesh.patch_face_slice("outlet")
    cx = res.mesh.points[res.mesh.faces[sl]].mean(axis=1)[:, 0]
    near_lo = np.abs(cx - (-5.8)) < 0.3
    near_hi = np.abs(cx - 49.3) < 0.3
    assert np.all(near_lo | near_hi)
    # Both planes are populated; the x=49.3 one lies in the static region.
    assert near_lo.sum() > 0 and near_hi.sum() > 0
    assert np.all(cx[near_hi] > MESH_BOX[1][0])

    pts_hi = res.mesh.points[np.unique(res.mesh.faces[sl][near_hi].ravel())]
    assert np.all(np.abs(pts_hi[:, 0] - 49.3) < 1e-6)


def test_outlet_interior_on_cap_source(tmp_path):
    r = torch.tensor(4.0, requires_grad=True)
    kwargs = dict(
        mesh=make_tube_span(-5.8, 50.0),
        caps=INTERIOR_CAPS,
        patches=CAPS_PATCHES,
    )
    oi = {"enabled": True, "method": "polygon_offset", "inset_distance": 1.0}
    res_carved = make_internal_pipeline(
        tmp_path / "a", r, outlet_interior=oi, **kwargs
    ).build()
    res_plain = make_internal_pipeline(
        tmp_path / "b", r, outlet_interior=None, **kwargs
    ).build()

    names = [p.name for p in res_carved.mesh.patches]
    assert names == ["inlet", "outlet", "outletInterior", "walls", "sensitivity_region"]

    n_out = res_carved.mesh.patch_by_name("outlet").n_faces
    n_int = res_carved.mesh.patch_by_name("outletInterior").n_faces
    assert n_int > 0 and n_out > 0
    assert n_out + n_int == res_plain.mesh.patch_by_name("outlet").n_faces

    # Interior faces strictly inside the inset cross-section, on the plane.
    sl = res_carved.mesh.patch_face_slice("outletInterior")
    fp = res_carved.mesh.points[res_carved.mesh.faces[sl]]
    c = fp.mean(axis=1)
    assert np.all(np.abs(c[:, 1]) < HALF_WIDTH - 0.9)
    assert np.all(np.abs(c[:, 2]) < HALF_WIDTH - 0.9)
    assert np.all(np.abs(fp[..., 0] - (-5.8)) < 1e-6)


def test_strict_patches_and_required(tmp_path):
    r = torch.tensor(4.0, requires_grad=True)

    # patches block present: unspecified faces default to "sides", never the
    # drag inlet/outlet (the tube crosses x_max, so those faces get "sides").
    pipe = make_internal_pipeline(
        tmp_path / "a", r,
        patches={"x_min": "outlet", "wall": "walls",
                 "sensitivity": "sensitivity_region"},
        outlet_interior=None,
    )
    res = pipe.build()
    names = [p.name for p in res.mesh.patches]
    assert "inlet" not in names
    assert "sides" in names  # x_max faces of the tube
    assert res.mesh.patch_by_name("sides").n_faces > 0

    # An explicitly named face patch that receives no faces raises instead
    # of being silently dropped.
    pipe = make_internal_pipeline(
        tmp_path / "b", r,
        patches={"x_min": "outlet", "x_max": "inlet", "y_min": "bogusInlet",
                 "wall": "walls", "sensitivity": "sensitivity_region"},
        outlet_interior=None,
    )
    with pytest.raises(ValueError, match="received no faces"):
        pipe.build()


def test_cap_validation_errors(tmp_path):
    r = torch.tensor(4.0, requires_grad=True)

    # Wrong plane value: the error lists the planar clusters actually found.
    with pytest.raises(ValueError, match=r"-5\.8"):
        make_internal_pipeline(
            tmp_path / "a", r,
            mesh=make_tube_span(-5.8, 50.0),
            caps=[{"axis": "x", "value": -5.0, "patch": "outlet"}],
            patches=CAPS_PATCHES, outlet_interior=None,
        )

    # caps + explicit cap_axis are mutually exclusive.
    with pytest.raises(ValueError, match="mutually exclusive"):
        make_internal_pipeline(
            tmp_path / "b", r,
            mesh=make_tube_span(-5.8, 50.0),
            caps=INTERIOR_CAPS, cap_axis="x",
            patches=CAPS_PATCHES, outlet_interior=None,
        )

    # Cap plane inside the design domain (differentiable region).
    with pytest.raises(ValueError, match="design domain"):
        make_internal_pipeline(
            tmp_path / "d", r,
            mesh=make_tube_span(-5.0, 50.0),
            caps=[{"axis": "x", "value": -5.0, "patch": "outlet"},
                  {"axis": "x", "value": 50.0, "patch": "inlet"}],
            patches=CAPS_PATCHES, outlet_interior=None,
        )

    # Interior cap patch name colliding with a wall-type patch.
    with pytest.raises(ValueError, match="collides"):
        make_internal_pipeline(
            tmp_path / "e", r,
            mesh=make_tube_span(-5.8, 50.0),
            caps=[{"axis": "x", "value": -5.8, "patch": "walls"},
                  {"axis": "x", "value": 50.0, "patch": "inlet"}],
            patches=CAPS_PATCHES, outlet_interior=None,
        )

    # Geometry protruding past the domain without an exempt face.
    wide = trimesh.creation.box(extents=[56.0, 50.0, 2 * HALF_WIDTH])
    wide.apply_translation([22.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="protrudes"):
        make_internal_pipeline(
            tmp_path / "f", r,
            mesh=wide,
            caps=[{"axis": "x", "value": -6.0, "patch": "outlet"},
                  {"axis": "x", "value": 50.0, "patch": "inlet"}],
            patches=CAPS_PATCHES, outlet_interior=None,
        )


# Symmetric half model: the z_min domain face cuts the geometry; declaring
# its patch type as "symmetry" exempts the protrusion and types the patch.
DOMAIN_SYM = [[-6.0, -21.0, 0.0], [50.0, 21.0, 21.0]]
MESH_BOX_SYM = [[-6.0, -13.0, 0.0], [2.0, 13.0, 13.0]]
DESIGN_DOMAIN_SYM = [[-5.5, -12.0, 0.0], [1.0, 12.0, 12.0]]


def _sym_kwargs():
    return dict(
        mesh=make_tube(),
        design_domain=DESIGN_DOMAIN_SYM,
        domain=DOMAIN_SYM,
        mesh_box=MESH_BOX_SYM,
        caps=[{"axis": "x", "value": -6.0, "patch": "outlet"},
              {"axis": "x", "value": 50.0, "patch": "inlet"}],
        patches={"z_min": "symmetry", "wall": "walls",
                 "sensitivity": "sensitivity_region"},
        seed_point=[10.0, 0.0, 1.0],
        sign_probe_point=[0.0, 0.0, 1.0],
        outlet_interior=None,
    )


def test_symmetry_half_model(tmp_path):
    r = torch.tensor(4.0, requires_grad=True)
    pipe = make_internal_pipeline(
        tmp_path, r, patch_types={"symmetry": "symmetry"}, **_sym_kwargs()
    )
    res = pipe.build()

    names = {p.name: p for p in res.mesh.patches}
    assert "symmetry" in names
    assert names["symmetry"].type == "symmetry"
    assert names["outlet"].type == "patch"

    # Symmetry faces lie exactly on the cut plane; the fluid is the upper
    # half of the channel only.
    sl = res.mesh.patch_face_slice("symmetry")
    pts = res.mesh.points[np.unique(res.mesh.faces[sl].ravel())]
    assert np.all(pts[:, 2] == 0.0)
    assert res.mesh.points[:, 2].min() >= 0.0

    pyr_o, pyr_n = face_pyramid_volumes(res.mesh)
    assert pyr_o.min() > 0 and pyr_n.min() > 0


def test_symmetry_requires_patch_type(tmp_path):
    r = torch.tensor(4.0, requires_grad=True)
    with pytest.raises(ValueError, match="protrudes"):
        make_internal_pipeline(tmp_path, r, **_sym_kwargs())


def test_cap_carve_predicate():
    from deepshapeopt.hexmesh.caps import CapSpec, cap_carve_classifier

    h_fine = 0.25

    def square_cap(axis, value, lo=0.0, hi=4.0):
        other = [a for a in range(3) if a != axis]
        p = np.zeros((4, 3))
        p[:, axis] = value
        p[:, other[0]] = [lo, hi, hi, lo]
        p[:, other[1]] = [lo, lo, hi, hi]
        tris = np.stack([p[[0, 1, 2]], p[[0, 2, 3]]])
        return CapSpec(
            axis=axis, value=value, patch="outlet", tol=1e-4,
            treatment="keep", triangles=tris,
        )

    def quad(axis, offset, center_u, center_v, width, perpendicular=False):
        """Corners [1, 4, 3] of a boundary quad near the cap plane."""
        other = [a for a in range(3) if a != axis]
        c = np.zeros((4, 3))
        du = np.array([-0.5, 0.5, 0.5, -0.5]) * width
        dv = np.array([-0.5, -0.5, 0.5, 0.5]) * width
        if perpendicular:
            # Varies along the cap axis instead of lying in the plane.
            c[:, axis] = offset + du
            c[:, other[0]] = center_u + dv
            c[:, other[1]] = center_v
        else:
            c[:, axis] = offset
            c[:, other[0]] = center_u + du
            c[:, other[1]] = center_v + dv
        return c[None]

    for axis in (1, 2):
        cap = square_cap(axis, 5.0)
        classify = cap_carve_classifier([cap], h_fine)

        cases = np.concatenate([
            quad(axis, 5.2, 2.0, 2.0, 0.5),    # parallel, near, inside -> in
            quad(axis, 5.5, 2.0, 2.0, 0.5),    # too far (0.5 > 0.375)
            quad(axis, 5.2, 2.0, 2.0, 0.5, perpendicular=True),  # not coplanar
            quad(axis, 5.2, 10.0, 10.0, 0.5),  # outside the footprint
        ])
        centroids = cases.mean(axis=1)
        got = classify(centroids, cases).tolist()
        assert got == [True, False, False, False], f"axis {axis}: {got}"

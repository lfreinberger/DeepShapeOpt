"""Tests for the SDF-driven hex mesh pipeline (synthetic sphere SDF).

These run without OpenFOAM / DeepSDFStruct: the SDF is an analytic sphere
with a differentiable radius parameter, standing in for the lattice network.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from deepshapeopt.hexmesh.constraints import volume_centroid_from_sdf
from deepshapeopt.hexmesh.foamwriter import checkmesh_log_ok, write_polymesh
from deepshapeopt.hexmesh.lattice import (
    CellIndex,
    CellSet,
    Lattice,
    face_corner_keys,
    pack_keys,
    split_cells,
    unpack_keys,
)
from deepshapeopt.hexmesh.octree import (
    MeshBox,
    balance_2to1,
    build_inner_castellation,
    build_static_octree,
    root_cells,
)
from deepshapeopt.hexmesh.pipeline import SdfHexMeshPipeline
from deepshapeopt.hexmesh.polymesh import (
    build_polymesh,
    cell_volumes,
    face_pyramid_volumes,
)
from deepshapeopt.hexmesh.sdf_field import PhysicalSDF
from deepshapeopt.hexmesh.snap import snap_wall_points


# ---------------------------------------------------------------------------
# Fixtures: small domain with a sphere SDF
# ---------------------------------------------------------------------------

DOMAIN = [[-4.0, -4.0, -4.0], [4.0, 4.0, 4.0]]
MESH_BOX = [[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]]
DESIGN_DOMAIN = [[-1.4, -1.4, -1.4], [1.4, 1.4, 1.4]]
IFACE = 2
MAX_LEVEL = 4


def make_lattice(max_depth=MAX_LEVEL) -> Lattice:
    return Lattice(origin=DOMAIN[0], h0=1.0, root_dims=[8, 8, 8], max_depth=max_depth)


def make_sphere_sdf(radius: float | torch.Tensor = 0.8) -> tuple[PhysicalSDF, torch.Tensor]:
    r = radius if isinstance(radius, torch.Tensor) else torch.tensor(float(radius))

    def fn(x):
        return torch.linalg.norm(x, dim=1) - r

    sdf = PhysicalSDF(
        sdf_norm_fn=fn,
        norm_fn=lambda x: x,
        design_domain=DESIGN_DOMAIN,
        dist_scale=1.0,
        device="cpu",
    )
    return sdf, r


def build_fluid_cells(radius=0.8):
    lattice = make_lattice()
    box = MeshBox.from_physical(lattice, MESH_BOX)
    sdf, r = make_sphere_sdf(radius)
    static = build_static_octree(lattice, box, IFACE)
    cast = build_inner_castellation(lattice, box, IFACE, MAX_LEVEL, sdf.phi_ext_np)
    return lattice, box, static, cast, sdf, r


# ---------------------------------------------------------------------------
# Lattice
# ---------------------------------------------------------------------------

def test_pack_unpack_roundtrip():
    keys = np.array([[0, 0, 0], [5, 123, 99], [2**20, 7, 2**21 - 1]], dtype=np.int64)
    assert np.array_equal(unpack_keys(pack_keys(keys)), keys)


def test_face_templates_normals():
    # Right-hand-rule normal of each template must point in its direction.
    expected = {
        0: [1, 0, 0], 1: [-1, 0, 0], 2: [0, 1, 0],
        3: [0, -1, 0], 4: [0, 0, 1], 5: [0, 0, -1],
    }
    anchors = np.array([[4, 4, 4]])
    widths = np.array([2])
    for direction, n_exp in expected.items():
        corners = face_corner_keys(anchors, widths, direction)[0].astype(float)
        normal = np.cross(corners[1] - corners[0], corners[2] - corners[1])
        normal /= np.linalg.norm(normal)
        assert np.allclose(normal, n_exp), f"direction {direction}"


def test_locate_doubled():
    lattice = make_lattice(max_depth=2)
    cells = root_cells(lattice)
    cells = split_cells(cells, np.arange(len(cells)) == 0, lattice)  # refine one
    index = CellIndex(lattice, cells)

    # Center of a child of the refined root cell (doubled fine units).
    q = np.array([[1, 1, 1], [2 * 4 + 4, 4, 4], [-1, 0, 0]])
    found = index.locate_doubled(q)
    assert found[0] >= 0 and cells.levels[found[0]] == 1
    assert found[1] >= 0 and cells.levels[found[1]] == 0
    assert found[2] == -1


# ---------------------------------------------------------------------------
# Octree
# ---------------------------------------------------------------------------

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


def test_static_octree():
    lattice = make_lattice()
    box = MeshBox.from_physical(lattice, MESH_BOX)
    static = build_static_octree(lattice, box, IFACE)

    # Partition: total volume == domain volume - box volume.
    vols = lattice.cell_size(static.levels) ** 3
    assert np.isclose(vols.sum(), 8.0**3 - 4.0**3)

    # No cell inside the box; cells adjacent to the box are at IFACE level.
    assert not np.any(box.contains_cells(static, lattice))
    w = static.widths(lattice)
    adjacent = np.all(
        (static.anchors + w[:, None] >= box.lo[None, :])
        & (static.anchors <= box.hi[None, :]),
        axis=1,
    ) & np.any(
        (static.anchors + w[:, None] == box.lo[None, :])
        | (static.anchors == box.hi[None, :]),
        axis=1,
    )
    assert np.any(adjacent)
    assert np.all(static.levels[adjacent] == IFACE)
    _check_balance(lattice, static)

    # Deterministic.
    static2 = build_static_octree(lattice, box, IFACE)
    assert np.array_equal(static.anchors, static2.anchors)
    assert np.array_equal(static.levels, static2.levels)


def test_inner_castellation_sphere():
    lattice, box, static, cast, sdf, _ = build_fluid_cells()
    cells = cast.cells

    # Solid removed: no fluid cell center inside the sphere.
    centers = cells.centers_phys(lattice)
    assert np.all(np.linalg.norm(centers, axis=1) > 0.8 - 1e-9)
    assert cast.n_solid_removed > 0

    # Shell cells are at the interface level and the union is balanced.
    assert np.all(cells.levels[cast.shell_mask] == IFACE)
    _check_balance(lattice, CellSet.concat(static, cells))

    # Near-surface cells are at max_level.
    near = np.abs(np.linalg.norm(centers, axis=1) - 0.8) < 0.5 * lattice.cell_size(MAX_LEVEL)
    assert np.all(cells.levels[near] == MAX_LEVEL)


# ---------------------------------------------------------------------------
# polyMesh assembly
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sphere_mesh():
    lattice, box, static, cast, sdf, r = build_fluid_cells()
    cells = CellSet.concat(static, cast.cells)
    mesh = build_polymesh(lattice, cells, box)
    return lattice, box, static, cast, sdf, r, cells, mesh


def test_polymesh_topology(sphere_mesh):
    lattice, box, static, cast, sdf, r, cells, mesh = sphere_mesh
    n_int = mesh.n_internal_faces

    # owner < neighbour and upper-triangular ordering.
    assert np.all(mesh.owner[:n_int] < mesh.neighbour)
    order = np.lexsort((mesh.neighbour, mesh.owner[:n_int]))
    assert np.array_equal(order, np.arange(n_int))

    # All points used.
    assert len(np.unique(mesh.faces.ravel())) == len(mesh.points)

    # Patch blocks contiguous and complete.
    assert mesh.patches[-1].name == "dragObject"
    total = n_int + sum(p.n_faces for p in mesh.patches)
    assert total == len(mesh.faces)
    assert mesh.patch_by_name("dragObject").n_faces > 0
    # Inlet plane (x = domain min) is far from the box: 8x8 root cell faces.
    assert mesh.patch_by_name("inlet").n_faces == 64
    _ = mesh.wall_point_ids()


def test_polymesh_geometry_closed(sphere_mesh):
    lattice, box, static, cast, sdf, r, cells, mesh = sphere_mesh

    # Unsnapped cells are perfect cubes: divergence-theorem volume must match.
    vols = cell_volumes(mesh)
    expected = lattice.cell_size(cells.levels) ** 3
    assert np.allclose(vols, expected, rtol=1e-9), (
        f"max err {np.abs(vols - expected).max()}"
    )

    # All face pyramids positive.
    pyr_o, pyr_n = face_pyramid_volumes(mesh)
    assert pyr_o.min() > 0
    assert pyr_n.min() > 0


def test_static_region_fixed_across_shapes():
    # R2: with different shapes, the mesh outside the box must be identical.
    lattice = make_lattice()
    box = MeshBox.from_physical(lattice, MESH_BOX)
    static = build_static_octree(lattice, box, IFACE)

    meshes = []
    for radius in (0.7, 0.95):
        sdf, _ = make_sphere_sdf(radius)
        cast = build_inner_castellation(lattice, box, IFACE, MAX_LEVEL, sdf.phi_ext_np)
        cells = CellSet.concat(static, cast.cells)
        meshes.append(build_polymesh(lattice, cells, box))

    n_static = len(static)
    for m in meshes:
        assert m.n_cells > n_static

    # Outer cells occupy ids [0, n_static) in both meshes; the faces they own
    # (incl. interface faces) must be geometrically identical.
    def outer_face_set(mesh):
        sel = mesh.owner < n_static
        return set(map(tuple, np.sort(mesh.point_keys[mesh.faces[sel]], axis=1)))

    assert outer_face_set(meshes[0]) == outer_face_set(meshes[1])

    # All points on or outside the box boundary are identical.
    def outer_points(mesh):
        keys = unpack_keys(mesh.point_keys)
        outside = np.any((keys <= box.lo[None, :]) | (keys >= box.hi[None, :]), axis=1)
        return set(mesh.point_keys[outside].tolist())

    assert outer_points(meshes[0]) == outer_points(meshes[1])


# ---------------------------------------------------------------------------
# Snap
# ---------------------------------------------------------------------------

def test_snap_residual_and_gradient(sphere_mesh):
    lattice, box, static, cast, sdf, r0, cells, mesh = sphere_mesh
    r = torch.tensor(0.8, requires_grad=True)
    sdf, _ = make_sphere_sdf(r)

    wall_ids = mesh.wall_point_ids()
    x0 = mesh.points[wall_ids]
    h = np.full(len(wall_ids), lattice.cell_size(MAX_LEVEL))

    handle = snap_wall_points(sdf, x0, h, snap_iters=4, max_disp_frac=0.9)
    x_star = handle.x_star()

    radii = torch.linalg.norm(x_star, dim=1).detach().numpy()
    full = handle.lam.numpy() >= 0.999
    assert full.mean() > 0.5  # most points snap fully
    assert np.abs(radii[full] - 0.8).max() < 1e-4

    # Gradient: x* depends on r via -lam * dphi/dr * n_hat / |grad| = +lam*n_hat.
    g = torch.randn_like(x_star)
    (dJ_dr,) = torch.autograd.grad(x_star, r, grad_outputs=g, retain_graph=True)
    n_hat = (x_star / torch.linalg.norm(x_star, dim=1, keepdim=True)).detach()
    expected = (g * n_hat * handle.lam[:, None]).sum()
    assert torch.isclose(dJ_dr, expected, rtol=1e-3, atol=1e-5)


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def test_foamwriter_roundtrip(tmp_path, sphere_mesh):
    *_, mesh = sphere_mesh
    out = write_polymesh(mesh, tmp_path)

    owner_txt = (out / "owner").read_text()
    m = re.search(r"\n(\d+)\n\(\n", owner_txt)
    assert int(m.group(1)) == len(mesh.faces)
    assert f"nCells:{mesh.n_cells}" in owner_txt

    faces_txt = (out / "faces").read_text()
    first_face = re.search(r"\(\n4\((\d+) (\d+) (\d+) (\d+)\)", faces_txt)
    assert [int(x) for x in first_face.groups()] == mesh.faces[0].tolist()

    boundary_txt = (out / "boundary").read_text()
    assert "dragObject" in boundary_txt
    assert "inGroups        1(dragObjectGroup)" in boundary_txt
    assert f"startFace       {mesh.n_internal_faces};" in boundary_txt

    try:
        from foamlib import FoamFile
    except ImportError:
        pytest.skip("foamlib not installed")
    pts = np.asarray(FoamFile(out / "points")[None], dtype=np.float64)
    assert pts.shape == mesh.points.shape
    assert np.allclose(pts, mesh.points)


def test_checkmesh_log_parse():
    ok, _ = checkmesh_log_ok("... \nMesh OK.\nEnd\n")
    assert ok
    ok, failed = checkmesh_log_ok(
        "***Max skewness = 8, 3 highly skew faces.\nFailed 1 mesh checks.\n"
    )
    assert not ok and len(failed) == 1
    ok, _ = checkmesh_log_ok("crashed before verdict")
    assert not ok


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------

def test_volume_centroid_sphere():
    r = torch.tensor(0.8, requires_grad=True)
    sdf, _ = make_sphere_sdf(r)
    V, C = volume_centroid_from_sdf(sdf, grid_res=64, eps_cells=1.5)
    V_exact = 4.0 / 3.0 * np.pi * 0.8**3
    assert abs(V.item() - V_exact) / V_exact < 0.01
    assert torch.allclose(C, torch.zeros(3), atol=0.01)

    # dV/dr = 4 pi r^2
    (dV,) = torch.autograd.grad(V, r)
    assert abs(dV.item() - 4 * np.pi * 0.8**2) / (4 * np.pi * 0.8**2) < 0.02


# ---------------------------------------------------------------------------
# Full pipeline (no OpenFOAM)
# ---------------------------------------------------------------------------

def make_pipeline(tmp_path, r: torch.Tensor) -> SdfHexMeshPipeline:
    def fn(x):
        return torch.linalg.norm(x, dim=1) - r

    model_setup = SimpleNamespace(
        design_domain=torch.tensor(DESIGN_DOMAIN, dtype=torch.float32),
        norm_fn=lambda x: x,
        scale=torch.tensor(1.0),
        box_norm=torch.tensor(DESIGN_DOMAIN, dtype=torch.float32),
    )
    opt_cfg = {
        "sdf_hex": {
            "domain": DOMAIN,
            "base_cell_size": 1.0,
            "mesh_box": MESH_BOX,
            "interface_level": IFACE,
            "max_level": MAX_LEVEL,
        }
    }
    pipe = SdfHexMeshPipeline(fn, model_setup, opt_cfg, tmp_path)
    # No DeepSDFStruct lattice here: bypass the float32 cast wrapper.
    pipe._with_float32 = lambda f: f()
    return pipe


def test_pipeline_build_and_quality(tmp_path):
    r = torch.tensor(0.8, requires_grad=True)
    pipe = make_pipeline(tmp_path, r)
    result = pipe.build()

    # Snapped mesh must still have positive pyramids everywhere.
    pyr_o, pyr_n = face_pyramid_volumes(result.mesh)
    assert pyr_o.min() > 0 and pyr_n.min() > 0

    # Sensitivity gradient path: random point forces -> dJ/dr.
    g = torch.randn_like(result.surface_points)
    (dJ,) = torch.autograd.grad(
        result.surface_points, r, grad_outputs=g, retain_graph=True
    )
    assert torch.isfinite(dJ)

    # Volume + gradient.
    V, C = pipe.volume_centroid()
    (dV,) = torch.autograd.grad(V, r)
    assert abs(dV.item() - 4 * np.pi * 0.8**2) / (4 * np.pi * 0.8**2) < 0.02


def test_pipeline_gradient_finite_difference(tmp_path):
    # End-to-end dJ/dr through the differentiable snap vs central differences
    # with frozen castellation (the sphere projection is purely radial, so
    # the Hadamard-form autograd derivative is the full derivative).  The
    # comparison is restricted to fully snapped points: displacement-capped
    # points (lam < 1) intentionally carry only the scaled Hadamard term.
    r = torch.tensor(0.8, requires_grad=True)
    pipe = make_pipeline(tmp_path, r)
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

    full = (
        (res.snap_lambda >= 0.999)
        & (res_p.snap_lambda >= 0.999)
        & (res_m.snap_lambda >= 0.999)
    )
    assert full.mean() > 0.3
    g = torch.randn_like(res.surface_points)
    g[torch.as_tensor(~full)] = 0.0

    J0 = (g * res.surface_points).sum()
    (dJ,) = torch.autograd.grad(J0, r)
    fd = ((g * res_p.surface_points).sum() - (g * res_m.surface_points).sum()).item() / (2 * eps)
    assert abs(dJ.item() - fd) / abs(fd) < 0.05, f"adjoint {dJ.item()} vs FD {fd}"


def test_pipeline_fast_path_and_topology_change(tmp_path):
    r = torch.tensor(0.8, requires_grad=True)
    pipe = make_pipeline(tmp_path, r)
    res1 = pipe.build()
    assert not res1.reused_connectivity

    # Same shape -> identical castellation -> connectivity reuse.
    res2 = pipe.build()
    assert res2.reused_connectivity
    assert res2.castellation_hash == res1.castellation_hash

    # Frozen-castellation rebuild (for FD checks) also reuses connectivity.
    res3 = pipe.build(reuse_castellation=True)
    assert res3.reused_connectivity

    # Different radius -> different castellation -> rebuild.
    with torch.no_grad():
        r += -0.15
    res4 = pipe.build()
    assert res4.castellation_hash != res1.castellation_hash
    pyr_o, pyr_n = face_pyramid_volumes(res4.mesh)
    assert pyr_o.min() > 0 and pyr_n.min() > 0

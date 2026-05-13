"""Visual test for mesh stitching at box boundaries.

Creates synthetic tube geometries (solid and hollow), stitches them
with different vertex counts, and exports STLs for inspection.

Run:
    uv run tests/test_mesh_stitching.py
"""
import numpy as np
import trimesh
from trimesh.transformations import rotation_matrix
from mesh_functions import (
    stitch_meshes_at_box,
    _remove_box_caps,
    _get_boundary_edges,
    export_boundary_edge_markers_stl,
)

R = rotation_matrix(np.pi / 2, [0, 1, 0])  # align cylinders along x


def check_mesh(mesh, name):
    """Check connectivity, unreferenced verts, and internal partition faces."""
    # Unreferenced vertices
    n_before = len(mesh.vertices)
    clean = mesh.copy()
    clean.remove_unreferenced_vertices()
    n_unreferenced = n_before - len(clean.vertices)

    # Connected components
    components = mesh.split()
    n_components = len(components)

    # Internal faces: both sides of the face are "inside" the mesh.
    # Outer faces have one side inside, one outside.
    # Skip near-degenerate faces (area < 1e-3 * median) — these are thin
    # bridge slivers from boundary projection that confuse the ray test.
    centroids = mesh.triangles_center
    normals = mesh.face_normals
    areas = mesh.area_faces
    area_threshold = float(np.median(areas)) * 1e-3
    worth_checking = areas > area_threshold
    eps = mesh.scale * 1e-4
    inside_pos = np.zeros(len(mesh.faces), dtype=bool)
    inside_neg = np.zeros(len(mesh.faces), dtype=bool)
    if worth_checking.any():
        inside_pos[worth_checking] = mesh.contains(
            centroids[worth_checking] + eps * normals[worth_checking])
        inside_neg[worth_checking] = mesh.contains(
            centroids[worth_checking] - eps * normals[worth_checking])
    internal = inside_pos & inside_neg
    n_internal = int(internal.sum())
    n_skipped = int((~worth_checking).sum())

    print(f"  Connectivity: {n_unreferenced} unreferenced verts, "
          f"{n_components} component(s), {n_internal} internal faces"
          f" ({n_skipped} degenerate faces skipped)")
    assert n_unreferenced == 0, f"{name}: {n_unreferenced} unreferenced vertices!"
    assert n_components == 1, f"{name}: {n_components} components (expected 1)!"
    assert n_internal == 0, f"{name}: {n_internal} internal partition faces detected!"


def test_cylinder_with_box_cut(out_dir):
    """Coarse cylinder with box cut, stitched with a fine hollow cylinder.

    Mimics a generic internal-channel stitching problem:
    - Base cylinder is coarse (8 sections) → outer boundary has few edges
    - Box is smaller than the cylinder → rectangular cut at one face
    - Inner shape is a fine hollow cylinder (32 sections) with open ends
    - Tests both subdivision of coarse outer boundary and stitching with
      mismatched vertex counts / boundary shapes.
    """
    print("=== Cylinder with box cut (coarse outer, fine inner) ===")

    # Design domain
    box_bounds = np.array([[-0.5, -0.5, -0.5], [0.2, 0.5, 0.5]])
    box = trimesh.creation.box(bounds=box_bounds)

    # Outer: coarse cylinder with box cut
    base_height = 2.0
    base_radius = 1.0
    base = trimesh.creation.cylinder(radius=base_radius, height=base_height, sections=16)
    base.apply_transform(R)
    base.apply_translation([base_height/2.0, 0, 0])  # center at x=0, extend in +x
    base.export(str(out_dir / "boxcut_original.stl"))

    outer = trimesh.boolean.difference([base, box], engine="manifold", check_volume=False)
    outer = _remove_box_caps(outer, box_bounds)

    be_out = _get_boundary_edges(outer)
    print(f"  Outer: {len(outer.faces)} faces, {len(be_out)} boundary edges")
    export_boundary_edge_markers_stl(
        outer,
        out_dir / "boxcut_outer_boundary_edges.stl",
        boundary_edges=be_out,
    )
    outer.export(str(out_dir / "boxcut_outer_uncapped.stl"))

    # Inner: fine hollow cylinder, open end
    inner_height = 0.5
    inner = trimesh.creation.cylinder(radius=0.5, height=inner_height, sections=32)
    inner.apply_transform(R)  # along x, centered at origin
    inner.apply_translation([-inner_height/2.0, 0, 0])  # shift to x=[-3, 1]
    axis = R[:3, :3] @ np.array([0, 0, 1])
    cap_normals = np.dot(inner.face_normals, axis) > 0.9
    inner.update_faces(~cap_normals)
    inner.remove_unreferenced_vertices()

    be_in = _get_boundary_edges(inner)
    print(f"  Inner: {len(inner.faces)} faces, {len(be_in)} boundary edges")
    export_boundary_edge_markers_stl(
        inner,
        out_dir / "boxcut_inner_boundary_edges.stl",
        boundary_edges=be_in,
    )

    inner.export(str(out_dir / "boxcut_inner.stl"))

    # Stitched result: should be a single watertight mesh with no boundary edges
    full = stitch_meshes_at_box(outer, inner, box_bounds)
    be_full = _get_boundary_edges(full)
    print(f"  Result: {len(full.faces)} faces, watertight={full.is_watertight}, boundary={len(be_full)}")

    export_boundary_edge_markers_stl(
        full,
        out_dir / "boxcut_stitched_boundary_edges.stl",
        boundary_edges=be_full,
    )
    full.export(str(out_dir / "boxcut_stitched.stl"))

    assert full.is_watertight, "Box-cut cylinder result is not watertight!"
    assert len(be_full) == 0, f"Box-cut cylinder has {len(be_full)} remaining boundary edges!"
    check_mesh(full, "Box-cut cylinder")
    print("  PASSED\n")


if __name__ == "__main__":
    from pathlib import Path

    out_dir = Path(__file__).resolve().parent / "test_stitch_output"
    out_dir.mkdir(exist_ok=True)
    print(f"Output directory: {out_dir}\n")

    test_cylinder_with_box_cut(out_dir)

    print("All tests passed. Inspect STLs in ParaView:")
    print(f"  {out_dir}")

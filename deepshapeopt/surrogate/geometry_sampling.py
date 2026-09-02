"""Seeded generation of primitive training geometries for the flow surrogate.

Produces watertight trimesh primitives (sphere, ellipsoid, box, rounded box,
cylinder along x/y/z, capsule) with randomized size, aspect ratio and
orientation, constrained to fit inside the design domain with a safety
margin. Each sample is deterministic in its integer seed, so slurm-array
shards and re-runs generate identical geometries.

The STLs are fed through the standard reconstruction phase
(:func:`deepshapeopt.shape_optimization.run_reconstruction`), which places
every training shape in exactly the lattice parametrization + snap
distribution that the optimizer explores.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import trimesh

FAMILIES = (
    "sphere",
    "ellipsoid",
    "box",
    "rounded_box",
    "cylinder_x",
    "cylinder_y",
    "cylinder_z",
    "capsule",
)


@dataclasses.dataclass
class GeometrySample:
    mesh: trimesh.Trimesh
    family: str
    seed: int
    meta: dict


def _random_rotation(rng: np.random.Generator, max_angle_deg: float) -> np.ndarray:
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = np.deg2rad(rng.uniform(0.0, max_angle_deg))
    return trimesh.transformations.rotation_matrix(angle, axis)


def _fit_into_box(mesh: trimesh.Trimesh, half_extents: np.ndarray) -> float:
    """Uniformly scale the mesh (about the origin) to fit the centered box."""
    ext = np.abs(mesh.bounds).max(axis=0)  # symmetric half-extents after centering
    scale = float(np.min(half_extents / ext))
    if scale < 1.0:
        mesh.apply_scale(scale)
    return min(scale, 1.0)


def sample_geometry(
    seed: int,
    design_domain,
    margin: float = 0.15,
    families=FAMILIES,
    max_rotation_deg: float = 30.0,
    volume_target_range=(0.7, 1.4),
    small_body_fraction: float = 0.25,
    small_volume_range=(0.15, 0.7),
) -> GeometrySample:
    """Generate one deterministic primitive geometry at a targeted volume.

    The shape is built at a nominal size, rotated, then scaled uniformly to
    hit a sampled target volume (the DeepSDF reconstruction preserves volume
    to ~1%, so the target carries through to the training sample). Volume
    targeting matters because the drag optimization runs under a volume
    constraint at V ~ 1: radius-uniform sampling put only 9% of shapes above
    V = 0.9, leaving the optimizer's operating regime nearly uncovered.

    A ``small_body_fraction`` of samples is drawn from ``small_volume_range``
    instead, to keep small-body coverage. Shapes that would leave the design
    box after scaling are clipped back (the achieved volume is recorded in
    ``meta["volume"]``).

    Parameters
    ----------
    seed : integer sample seed (one per dataset sample).
    design_domain : [[xmin, ymin, zmin], [xmax, ymax, zmax]] physical box.
    margin : physical clearance kept to every design-domain face (must cover
        the interface refinement band of the hex mesh pipeline).
    volume_target_range : (lo, hi) target volume of the main population.
    small_body_fraction : share of samples drawn from ``small_volume_range``.
    """
    rng = np.random.default_rng(seed)
    family = families[int(rng.integers(len(families)))]

    dd = np.asarray(design_domain, dtype=float)
    center = 0.5 * (dd[0] + dd[1])
    half = 0.5 * (dd[1] - dd[0]) - margin
    if np.any(half <= 0):
        raise ValueError(f"margin {margin} leaves no room in domain {design_domain}")

    # Characteristic radius: relative to the smallest half-extent so even the
    # largest draw fits the flat directions before rotation.
    r = float(rng.uniform(0.45, 0.95) * half.min())
    aspect = rng.uniform(0.6, 1.8, size=3)

    if family == "sphere":
        mesh = trimesh.creation.icosphere(subdivisions=4, radius=r)
        meta = {"radius": r}
    elif family == "ellipsoid":
        mesh = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
        semi = r * aspect
        mesh.apply_scale(semi)
        meta = {"semi_axes": semi.tolist()}
    elif family == "box":
        extents = 2.0 * r * aspect
        mesh = trimesh.creation.box(extents=extents)
        meta = {"extents": extents.tolist()}
    elif family == "rounded_box":
        extents = 2.0 * r * aspect
        radius = float(rng.uniform(0.1, 0.3) * extents.min())
        mesh = trimesh.creation.box(extents=extents - 2 * radius)
        # Approximate Minkowski rounding: subdivide, offset along vertex
        # normals, re-hull (convex, so this is safe and watertight).
        for _ in range(3):
            mesh = mesh.subdivide()
        verts = mesh.vertices + radius * mesh.vertex_normals
        mesh = trimesh.Trimesh(vertices=verts, faces=mesh.faces).convex_hull
        meta = {"extents": extents.tolist(), "corner_radius": radius}
    elif family in ("cylinder_x", "cylinder_y", "cylinder_z"):
        radius = r * float(rng.uniform(0.5, 0.9))
        height = 2.0 * r * float(rng.uniform(0.8, 1.6))
        mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=64)
        axis = {"cylinder_x": [0, 1, 0], "cylinder_y": [1, 0, 0], "cylinder_z": None}[family]
        if axis is not None:
            mesh.apply_transform(
                trimesh.transformations.rotation_matrix(np.pi / 2, axis)
            )
        meta = {"radius": radius, "height": height}
    elif family == "capsule":
        radius = r * float(rng.uniform(0.4, 0.7))
        height = 2.0 * r * float(rng.uniform(0.6, 1.2))
        mesh = trimesh.creation.capsule(radius=radius, height=height, count=[32, 32])
        mesh.apply_translation(-mesh.bounds.mean(axis=0))  # capsule() is not centered
        mesh.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
        )
        meta = {"radius": radius, "height": height}
    else:  # pragma: no cover
        raise ValueError(family)

    mesh.apply_transform(_random_rotation(rng, max_rotation_deg))

    # Scale to the target volume, then clip back into the design box.
    if rng.random() < small_body_fraction:
        v_target = float(rng.uniform(*small_volume_range))
    else:
        v_target = float(rng.uniform(*volume_target_range))
    mesh.apply_scale((v_target / mesh.volume) ** (1.0 / 3.0))
    scale_applied = _fit_into_box(mesh, half)
    # The design box is elongated (2.7 x 1.3 x 1.3 for the drag case), so
    # isotropic growth hits the y/z faces long before the target volume is
    # reached. Recover the deficit by stretching along the roomy axes -- a
    # property of the box, independent of what the optimum looks like.
    for axis in np.argsort(-(half / np.abs(mesh.bounds).max(axis=0))):
        deficit = v_target / mesh.volume
        if deficit <= 1.001:
            break
        room = half[axis] / np.abs(mesh.bounds).max(axis=0)[axis]
        factor = min(deficit, room)
        if factor <= 1.001:
            continue
        stretch = np.ones(3)
        stretch[axis] = factor
        mesh.apply_scale(stretch)
    mesh.apply_translation(center)

    if not mesh.is_watertight:
        raise RuntimeError(f"sample {seed} ({family}) is not watertight")

    meta.update(
        {
            "aspect": aspect.tolist(),
            "scale_clip": scale_applied,
            "volume_target": v_target,
            "volume": float(mesh.volume),
            "bounds": mesh.bounds.tolist(),
        }
    )
    return GeometrySample(mesh=mesh, family=family, seed=seed, meta=meta)

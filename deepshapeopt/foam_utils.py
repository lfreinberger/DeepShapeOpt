from pathlib import Path
import contextlib
import io
import logging
import os
import re
import time
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
import torch
import subprocess
import shutil
from foamlib import FoamCase, FoamFile, FoamFieldFile
import warnings
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


logger = logging.getLogger(__name__)


def prepare_foam_runtime(template_dir: Path, run_name: str) -> Path:
    """Copy foam_case template to an isolated runtime directory for concurrent execution."""
    runtime_dir = template_dir.parent / f"foam_run_{run_name}"
    if runtime_dir.exists():
        shutil.rmtree(runtime_dir)
    shutil.copytree(template_dir, runtime_dir)
    return runtime_dir


def configure_foam_runtime(case_dir: Path, constraint_enabled: bool) -> dict[str, str]:
    """Derive adjoint-time directories from optimisationDict and patch the runtime case.

    Reads primal/adjoint ``nIters`` from ``system/optimisationDict`` and computes the
    time directory each adjoint solver will write to:
    ``t_as1 = p_n + as1_n`` and ``t_as2 = t_as1 + as2_n``.

    Also mutates the runtime copy:
      - ``optimisationDict``: sets ``am1.as2.active`` to ``constraint_enabled``
      - ``controlDict``: forces ``purgeWrite = 0`` so no needed time dir is purged
      - ``Allrun``: replaces the ``__ADJOINT_TIMES__`` marker with the derived times

    Should be called once after ``prepare_foam_runtime`` copies the template.
    Operating on the template directly would dirty the git-tracked files.

    Returns a mapping ``{"as1": "300"}`` or ``{"as1": "300", "as2": "500"}``.
    """
    opt_path = case_dir / "system" / "optimisationDict"
    ctrl_path = case_dir / "system" / "controlDict"
    allrun_path = case_dir / "Allrun"

    opt = FoamFile(opt_path)
    p_n = int(opt["primalSolvers", "p1", "solutionControls", "nIters"])
    as1_n = int(opt["adjointManagers", "am1", "adjointSolvers", "as1", "solutionControls", "nIters"])
    as2_n = int(opt["adjointManagers", "am1", "adjointSolvers", "as2", "solutionControls", "nIters"])

    opt["adjointManagers", "am1", "adjointSolvers", "as2", "active"] = bool(constraint_enabled)

    t_as1 = p_n + as1_n
    adjoint_times = {"as1": str(t_as1)}
    if constraint_enabled:
        adjoint_times["as2"] = str(t_as1 + as2_n)

    ctrl = FoamFile(ctrl_path)
    ctrl["purgeWrite"] = 0

    marker = "__ADJOINT_TIMES__"
    text = allrun_path.read_text()
    if marker not in text:
        raise RuntimeError(f"{allrun_path}: missing {marker} marker in Allrun template")
    allrun_path.write_text(text.replace(marker, ",".join(adjoint_times.values())))

    return adjoint_times


def run_openfoam_case(case_dir: Path, verbose: bool = True):
    if not case_dir.exists():
        raise FileNotFoundError(f"Foam case path does not exist: {case_dir}")

    foam_case = FoamCase(case_dir)
    if verbose:
        foam_case.clean()
    else:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            foam_case.clean()

    paraview_file = case_dir / "open.foam"
    paraview_file.touch(exist_ok=True)

    cmd = "source $WM_PROJECT_DIR/etc/bashrc && ./Allrun"

    start = time.time()
    logger.info("Running OpenFOAM")
    if verbose:
        subprocess.run(
            cmd,
            cwd=case_dir,
            shell=True,
            executable="/bin/bash",
            check=False,
        )
    else:
        subprocess.run(
            cmd,
            cwd=case_dir,
            shell=True,
            executable="/bin/bash",
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    end = time.time()
    logger.debug("OpenFOAM finished in %.2f seconds", end - start)

    return foam_case

def load_mesh_coords_and_sensitivities(case_dir: Path, field_name: str, time_step):
    time_value = Path(str(time_step)).name

    points_path = case_dir / "constant" / "polyMesh" / "points"
    field_path = case_dir / time_value / field_name

    coords = np.asarray(FoamFile(points_path)[None], dtype=np.float64)
    sensitivities = np.asarray(FoamFieldFile(field_path).internal_field, dtype=np.float64)

    if coords.shape[0] != sensitivities.shape[0]:
        raise ValueError(
            f"Mismatch: {coords.shape[0]} points, {sensitivities.shape[0]} sensitivities."
        )

    return coords, sensitivities, None, time_value


def load_sampled_surface(case_dir: Path, field_name: str, time_step):
    """Run OpenFOAM case, load STLsurface.vtp, return coords + sensitivities."""
    time_value = os.path.basename(time_step)
    vtp_path = case_dir / f"postProcessing/sampleDict/{time_value}/STLsurface.vtp"

    sampled_mesh = pv.read(vtp_path)

    sensitivities = sampled_mesh.point_data[field_name]
    if sensitivities.max() > 1e9:
        raise RuntimeError("Unusually high sensitivity values detected from OpenFOAM: {}.".format(sensitivities.max()))

    coords = sampled_mesh.points
    return coords, sensitivities, sampled_mesh, time_value


def map_sensitivities_to_vertices(sampled_coords, sensitivities, verts, tol=1e-9):
    """
    Map sensitivities to verts by coordinate matching.
    - Vertices without a matching coordinate get sensitivity = 0
    - Negative sensitivities are clipped to 0
    """

    verts_np = verts.detach().cpu().numpy()

    tree = cKDTree(sampled_coords)
    dist, idx = tree.query(verts_np, k=1)

    # geometric match
    mask = dist <= tol

    n_total = len(verts_np)
    n_matched = np.sum(mask)

    if n_matched < n_total:
        raise ValueError(
            f"Only {n_matched} out of {n_total} vertices matched within tolerance {tol}. "
        )
    logger.debug("Matched vertices: %d / %d with max distance %.2e", n_matched, n_total, dist.max())

    # initialize with zeros
    sens_on_verts = np.zeros(n_total, dtype=sensitivities.dtype)

    # assign matched sensitivities
    sens_on_verts[mask] = sensitivities[idx[mask]]

    return sens_on_verts

def interpolate_sensitivities_to_vertices(
    sampled_coords,
    sensitivities,
    verts,
    warn_tol=None,
):
    verts_np = verts.detach().cpu().numpy()

    n_verts = len(verts_np)
    if sensitivities.ndim > 1:
        nonzero_mask = np.any(sensitivities != 0, axis=1)
    else:
        nonzero_mask = sensitivities != 0
    n_nonzero = int(np.sum(nonzero_mask))
    logger.debug(
        "Interpolation: %d vertices, %d/%d nonzero source points (%.1f%%)",
        n_verts,
        n_nonzero,
        len(sampled_coords),
        n_nonzero / len(verts_np) * 100,
    )

    src_coords = sampled_coords[nonzero_mask]
    src_sens = sensitivities[nonzero_mask]

    linear_interp = LinearNDInterpolator(src_coords, src_sens)
    sens_on_verts = linear_interp(verts_np)

    # Fall back to nearest-neighbor for vertices outside the convex hull
    if sens_on_verts.ndim > 1:
        nan_rows = np.any(np.isnan(sens_on_verts), axis=1)
    else:
        nan_rows = np.isnan(sens_on_verts)
    n_fallback = int(np.sum(nan_rows))
    if n_fallback > 0:
        nearest_interp = NearestNDInterpolator(src_coords, src_sens)
        sens_on_verts[nan_rows] = nearest_interp(verts_np[nan_rows])
        logger.debug("Interpolation: %d vertices outside convex hull, filled with nearest-neighbor", n_fallback)

    if warn_tol is not None:
        tree = cKDTree(src_coords)
        dist, _ = tree.query(verts_np, k=1)

        max_dist = dist.max()
        n_large = np.sum(dist > warn_tol)

        if n_large > 0:
            warnings.warn(
                f"{n_large} vertices have nearest-neighbor distance larger than warn_tol={warn_tol}. "
                f"Max distance = {max_dist:.6e}. Interpolated values may be unreliable.",
                RuntimeWarning
            )

    return sens_on_verts


def load_boundary_patch_faces(case_dir, patch_name="dragObject"):
    """Read triangulated face connectivity for a boundary patch from OpenFOAM polyMesh.

    Returns an [F, 3] int array of vertex indices (into the full polyMesh points
    array).  Non-triangular faces are fan-triangulated.

    Only the relevant faces are parsed (not the entire faces file), so this is
    fast even for large meshes.
    """
    boundary_path = case_dir / "constant" / "polyMesh" / "boundary"
    faces_path = case_dir / "constant" / "polyMesh" / "faces"

    # --- read boundary to find patch offset and count ---
    boundary_entries = FoamFile(boundary_path)[None]  # list of (name, info_dict)

    patch_info = None
    for name, info in boundary_entries:
        if name == patch_name:
            patch_info = info
            break
    if patch_info is None:
        available = [name for name, _ in boundary_entries]
        raise ValueError(
            f"Patch '{patch_name}' not found in boundary file. "
            f"Available patches: {available}"
        )

    start_face = int(patch_info["startFace"])
    n_faces = int(patch_info["nFaces"])

    # --- parse only the needed face lines from the faces file ---
    # The file has a header, then a count line, then '(', then face lines.
    # Face i corresponds to the i-th face line after '('.
    face_pattern = re.compile(r"\d+\(([^)]+)\)")

    patch_faces_raw = []
    with open(faces_path, "r") as f:
        # Skip header until we find the opening '('
        line_idx = -1
        for line in f:
            line = line.strip()
            if line == "(":
                line_idx = 0
                break

        # Now read face lines, only keep those in [start_face, start_face + n_faces)
        end_face = start_face + n_faces
        for line in f:
            if line_idx >= end_face:
                break
            if line_idx >= start_face:
                m = face_pattern.match(line.strip())
                if m:
                    verts = [int(x) for x in m.group(1).split()]
                    patch_faces_raw.append(verts)
            line_idx += 1

    if len(patch_faces_raw) != n_faces:
        raise RuntimeError(
            f"Expected {n_faces} faces for patch '{patch_name}', "
            f"but parsed {len(patch_faces_raw)}"
        )

    # --- fan-triangulate each face ---
    triangles = []
    for face in patch_faces_raw:
        for j in range(1, len(face) - 1):
            triangles.append([face[0], face[j], face[j + 1]])

    logger.debug("Boundary patch %s: %d faces -> %d triangles", patch_name, n_faces, len(triangles))
    return np.array(triangles, dtype=np.int64)


def _barycentric_coords(points, v0, v1, v2):
    """Vectorized barycentric coordinates for points w.r.t. triangles.

    All inputs [N, 3].  Returns [N, 3] weights (w0, w1, w2) for (v0, v1, v2).
    Coordinates are clamped to the triangle for robustness.
    """
    e0 = v1 - v0
    e1 = v2 - v0
    v = points - v0

    d00 = np.einsum("ij,ij->i", e0, e0)
    d01 = np.einsum("ij,ij->i", e0, e1)
    d11 = np.einsum("ij,ij->i", e1, e1)
    d20 = np.einsum("ij,ij->i", v, e0)
    d21 = np.einsum("ij,ij->i", v, e1)

    denom = d00 * d11 - d01 * d01
    denom = np.where(np.abs(denom) < 1e-30, 1e-30, denom)

    w1 = (d11 * d20 - d01 * d21) / denom
    w2 = (d00 * d21 - d01 * d20) / denom
    w0 = 1.0 - w1 - w2

    # Clamp to triangle and renormalize
    w0 = np.maximum(w0, 0.0)
    w1 = np.maximum(w1, 0.0)
    w2 = np.maximum(w2, 0.0)
    total = w0 + w1 + w2
    total = np.where(total < 1e-30, 1.0, total)

    return np.column_stack([w0 / total, w1 / total, w2 / total])


def project_sensitivities_to_vertices(
    all_coords,
    patch_faces,
    all_sens,
    target_verts,
    warn_tol=None,
):
    """Map sensitivities from a source boundary mesh to target vertices via
    nearest-face projection with barycentric interpolation.

    Parameters
    ----------
    all_coords : ndarray [N_total, 3]
        All polyMesh points.
    patch_faces : ndarray [F, 3]
        Triangulated boundary patch faces (indices into all_coords).
    all_sens : ndarray [N_total, ...]
        Sensitivity field on all polyMesh points (zeros away from patch).
    target_verts : torch.Tensor [N_target, 3]
        Target mesh vertices to interpolate onto.
    warn_tol : float, optional
        Warn if any projection distance exceeds this value.
    """
    target_np = target_verts.detach().cpu().numpy()

    # Extract the local patch sub-mesh
    unique_vert_ids = np.unique(patch_faces.ravel())
    global_to_local = np.full(len(all_coords), -1, dtype=np.int64)
    global_to_local[unique_vert_ids] = np.arange(len(unique_vert_ids))

    local_coords = all_coords[unique_vert_ids]
    local_sens = all_sens[unique_vert_ids]
    local_faces = global_to_local[patch_faces]

    n_src = len(local_coords)
    n_target = len(target_np)
    logger.debug(
        "Projection: %d target vertices, %d source vertices, %d source triangles",
        n_target,
        n_src,
        len(local_faces),
    )

    # Build pyvista surface mesh for cell locator
    faces_pv = np.column_stack([
        np.full(len(local_faces), 3, dtype=np.int64),
        local_faces,
    ]).ravel()
    source_mesh = pv.PolyData(local_coords, faces_pv)

    # Find closest cell (triangle) for each target vertex
    cell_ids, closest_pts = source_mesh.find_closest_cell(
        target_np, return_closest_point=True
    )

    # Triangle vertex indices for each closest cell
    tri_verts = local_faces[cell_ids]  # [N_target, 3]
    v0 = local_coords[tri_verts[:, 0]]
    v1 = local_coords[tri_verts[:, 1]]
    v2 = local_coords[tri_verts[:, 2]]

    # Barycentric interpolation at the projected points
    bary = _barycentric_coords(closest_pts, v0, v1, v2)  # [N_target, 3]

    s0 = local_sens[tri_verts[:, 0]]
    s1 = local_sens[tri_verts[:, 1]]
    s2 = local_sens[tri_verts[:, 2]]

    if local_sens.ndim > 1:
        sens_on_verts = (
            bary[:, 0:1] * s0 + bary[:, 1:2] * s1 + bary[:, 2:3] * s2
        )
    else:
        sens_on_verts = bary[:, 0] * s0 + bary[:, 1] * s1 + bary[:, 2] * s2

    # Distance diagnostics
    dists = np.linalg.norm(target_np - closest_pts, axis=1)
    max_dist = dists.max()
    logger.debug("Projection: max distance = %.6e", max_dist)

    if warn_tol is not None:
        n_large = int(np.sum(dists > warn_tol))
        if n_large > 0:
            warnings.warn(
                f"{n_large} target vertices have projection distance > {warn_tol}. "
                f"Max distance = {max_dist:.6e}.",
                RuntimeWarning,
            )
    logger.debug("abs(local_sens).mean(): %.6e", abs(local_sens).mean())
    logger.debug("abs(sens_on_verts).mean(): %.6e", abs(sens_on_verts).mean())
    return sens_on_verts


def conservative_sensitivity_transfer(
    all_coords,
    patch_faces,
    all_sens,
    target_verts,
    target_faces,
    warn_tol=None,
    return_diagnostics: bool = False,
):
    """Conservative scatter of integrated OF forces onto STL vertices.

    Requires ``includeSurfaceArea true`` in optimisationDict so that OF writes
    the already-integrated dJ/dx (not a per-unit-area density).

    Each OF boundary vertex is projected onto the closest STL triangle, and its
    force vector is distributed to the three triangle vertices via barycentric
    weights.  This is the transpose of consistent interpolation and guarantees
    ``sum(F_STL) == sum(F_OF)`` regardless of the mesh ratio.

    Returns
    -------
    F_STL : ndarray [N_target, 3]
        Integrated force per STL vertex.  Pass to ``compute_shape_gradient``
        with ``integrated=True``.
    """
    target_np = target_verts.detach().cpu().numpy()
    target_faces_np = target_faces.detach().cpu().numpy().astype(np.int64)

    # --- extract local OF boundary sub-mesh ---
    unique_vert_ids = np.unique(patch_faces.ravel())
    global_to_local = np.full(len(all_coords), -1, dtype=np.int64)
    global_to_local[unique_vert_ids] = np.arange(len(unique_vert_ids))

    local_coords = all_coords[unique_vert_ids]
    local_sens = all_sens[unique_vert_ids]       # integrated force [N_local, 3]
    local_faces = global_to_local[patch_faces]

    n_src = len(local_coords)
    n_target = len(target_np)
    logger.debug(
        "Conservative transfer: %d source OF vertices, %d target STL vertices",
        n_src,
        n_target,
    )

    # --- build pyvista STL mesh for cell locator ---
    stl_faces_pv = np.column_stack([
        np.full(len(target_faces_np), 3, dtype=np.int64),
        target_faces_np,
    ]).ravel()
    stl_mesh = pv.PolyData(target_np, stl_faces_pv)

    # --- project each OF vertex onto closest STL triangle ---
    cell_ids, closest_pts = stl_mesh.find_closest_cell(
        local_coords, return_closest_point=True,
    )

    tri_verts = target_faces_np[cell_ids]  # [N_local, 3]
    tv0 = target_np[tri_verts[:, 0]]
    tv1 = target_np[tri_verts[:, 1]]
    tv2 = target_np[tri_verts[:, 2]]
    bary = _barycentric_coords(closest_pts, tv0, tv1, tv2)  # [N_local, 3]

    # --- scatter forces to STL vertices (transpose of consistent interp) ---
    F_STL = np.zeros((n_target, 3), dtype=np.float64)
    for k in range(3):
        np.add.at(F_STL, tri_verts[:, k], bary[:, k:k+1] * local_sens)

    # --- diagnostics ---
    total_force_of = float(np.abs(local_sens).sum())
    total_force_stl = float(np.abs(F_STL).sum())
    dists = np.linalg.norm(local_coords - closest_pts, axis=1)
    max_dist = float(dists.max())

    force_vec_of = local_sens.sum(axis=0)
    force_vec_stl = F_STL.sum(axis=0)
    force_vec_of_norm = float(np.linalg.norm(force_vec_of))
    force_vec_stl_norm = float(np.linalg.norm(force_vec_stl))

    diag = {
        "conservative_max_proj_dist": max_dist,
        "conservative_l1_ratio": total_force_stl / (total_force_of + 1e-30),
        "conservative_vec_norm_ratio": force_vec_stl_norm / (force_vec_of_norm + 1e-30),
        "conservative_n_src": int(n_src),
        "conservative_n_target": int(n_target),
    }
    logger.debug("Conservative transfer: max projection distance = %.6e", max_dist)
    logger.debug(
        "Conservative transfer: sum|F_OF|=%.6e, sum|F_STL|=%.6e, ratio=%.6f",
        total_force_of,
        total_force_stl,
        total_force_stl / total_force_of,
    )

    if warn_tol is not None:
        n_large = int(np.sum(dists > warn_tol))
        if n_large > 0:
            warnings.warn(
                f"{n_large} OF vertices have projection distance > {warn_tol}. "
                f"Max distance = {max_dist:.6e}.",
                RuntimeWarning,
            )

    if return_diagnostics:
        return F_STL, diag
    return F_STL


def save_mesh_with_sensitivities(verts, faces, sens_on_orig, out_path: Path):
    """Write a VTP with sensitivities as point data for ParaView checks."""
    verts_np = verts.detach().cpu().numpy()
    faces_np = faces.detach().cpu().numpy()
    n_faces = faces_np.shape[0]
    faces_pv = np.hstack(
        np.c_[np.full(n_faces, 3, dtype=np.int64), faces_np.astype(np.int64)]
    ).ravel()

    mesh_orig = pv.PolyData(verts_np, faces_pv)
    mesh_orig.point_data["sens"] = sens_on_orig
    mesh_orig.save(out_path)

def read_objective(case_path: Path, objective_path):
    path = case_path / objective_path
    data = np.loadtxt(path, comments="#")
    J_last = data[1]   # last iteration's J
    return J_last


def compute_vertex_normals(verts: torch.Tensor, faces: torch.Tensor, invert_normals: bool) -> torch.Tensor:
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


def compute_area_weighted_vertex_normals(verts: torch.Tensor, faces: torch.Tensor, invert_normals: bool) -> torch.Tensor:
    """
    Return area-weighted vertex normals for surface integral quadrature.

    Each vertex accumulates the cross-product contributions of its adjacent
    faces.  cross(e1, e2) has magnitude 2*area_f and direction n_f, so the
    accumulated vector at vertex j encodes both normal direction and the sum of
    adjacent face areas — suitable for discretising ∫ s · δx_n dA without a
    separate vertex-area term.

    Use this (not unit normals) in compute_shape_gradient when OpenFOAM
    sensitivities are computed with includeSurfaceArea false.
    """
    V = verts.shape[0]
    device = verts.device
    faces = faces.to(torch.long).to(device)
    v0 = verts[faces[:,0]]
    v1 = verts[faces[:,1]]
    v2 = verts[faces[:,2]]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)  # magnitude = 2*area_f

    vn = torch.zeros((V, 3), device=device, dtype=verts.dtype)
    vn.index_add_(0, faces[:,0], face_normals)
    vn.index_add_(0, faces[:,1], face_normals)
    vn.index_add_(0, faces[:,2], face_normals)

    # Divide by 6: factor 2 from cross product (parallelogram vs triangle area),
    # factor 3 from each face being shared by 3 vertices.
    # Result: vn[j] ≈ A_j * n̂_j, the correct vertex quadrature weight.
    vn = vn / 6.0

    if invert_normals:
        return -vn
    return vn


def compute_vertex_magSf(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Sum of adjacent face areas per vertex, matching OpenFOAM's pointMagSf.

    OpenFOAM's surfacePoints sensitivity divides the integrated dJ/dx by
    pointMagSf (= sum of full adjacent face areas) when includeSurfaceArea is
    false.  To recover dJ/dx from the written density, multiply by the same
    quantity on the STL mesh.
    """
    faces = faces.to(torch.long).to(verts.device)
    v0, v1, v2 = verts[faces[:,0]], verts[faces[:,1]], verts[faces[:,2]]
    face_areas = torch.cross(v1 - v0, v2 - v0, dim=1).norm(dim=1) / 2.0
    vertex_magSf = torch.zeros(verts.shape[0], device=verts.device, dtype=verts.dtype)
    vertex_magSf.index_add_(0, faces[:,0], face_areas)
    vertex_magSf.index_add_(0, faces[:,1], face_areas)
    vertex_magSf.index_add_(0, faces[:,2], face_areas)
    return vertex_magSf


def compute_shape_gradient(param, verts, faces, sens_on_orig, invert_normals=False, integrated=False):
    """Compute dJ/dparam given vertex sensitivities.

    Parameters
    ----------
    integrated : bool
        If True, *sens_on_orig* already contains integrated forces [N, 3]
        (e.g. from ``conservative_sensitivity_transfer`` with
        ``includeSurfaceArea true``).  No area multiplication is applied.
        If False (default), sens_on_orig is treated as a density that must be
        multiplied by STL pointMagSf (legacy ``includeSurfaceArea false`` path).

    Modes depending on sens_on_orig shape:
    - [N, 3] full vector (pointSensVecadjS1ESI):
      * integrated=False: density path, multiplies by STL pointMagSf
      * integrated=True:  already integrated, used directly
    - [N] scalar (legacy, pointSensNormaladjS1ESI): uses unit normals.
    """
    s = torch.as_tensor(sens_on_orig, dtype=verts.dtype, device=verts.device)

    if s.ndim == 2:  # full 3D vector path
        if integrated:
            g = s.reshape(-1)
        else:
            vertex_magSf = compute_vertex_magSf(verts, faces)
            g = (s * vertex_magSf[:, None]).reshape(-1)
    else:  # legacy scalar path
        normals = compute_vertex_normals(verts, faces, invert_normals)
        g = (s[:, None] * normals).reshape(-1)

    dJ, = torch.autograd.grad(
        outputs=verts.reshape(-1),
        inputs=param,
        grad_outputs=g.reshape(-1),
        retain_graph=True,
        create_graph=False,
        allow_unused=False,
    )
    return dJ

def export_vtk_for_iteration(case, case_dir: Path, vtk_series_dir: Path, e: int):
    """
    Export VTK files for the latest OpenFOAM time step for a given optimization iteration.
    """

    case.run(["foamToVTK", "-latestTime"])

    vtk_series_dir = str(vtk_series_dir / "vtk_series")
    os.makedirs(vtk_series_dir, exist_ok=True)

    vtk_root = case_dir / "VTK"
    if not vtk_root.exists():
        raise RuntimeError(f"VTK directory not found: {vtk_root}")

    vtk_dirs = [d for d in vtk_root.iterdir() if d.is_dir()]
    if not vtk_dirs:
        raise RuntimeError(f"No VTK subdirectories in {vtk_root}")

    latest_vtk_dir = max(vtk_dirs, key=lambda p: p.stat().st_mtime)

    # ---------- volume ----------
    vtu_files = list(latest_vtk_dir.glob("internal*.vtu"))
    if not vtu_files:
        raise RuntimeError(f"No internal*.vtu found in {latest_vtk_dir}")

    src_internal = max(vtu_files, key=lambda p: p.stat().st_mtime)
    dst_internal = f"{vtk_series_dir}/internal_{e:04d}.vtu"

    shutil.copy2(src_internal, dst_internal)
    logger.debug("Saved volume VTK for ParaView: %s", dst_internal)

    # ---------- boundary (optional) ----------
    boundary_dir = latest_vtk_dir / "boundary"
    src_boundary = boundary_dir / "dragObject.vtp"

    if boundary_dir.exists() and src_boundary.exists():
        dst_boundary = f"{vtk_series_dir}/dragObject_{e:04d}.vtp"
        shutil.copy2(src_boundary, dst_boundary)
        logger.debug("Saved boundary VTK for ParaView: %s", dst_boundary)
    else:
        logger.debug("Boundary VTK not found for iteration %s (expected %s)", e, src_boundary)

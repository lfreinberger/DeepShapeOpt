"""Generate the OpenFOAM flow-field training dataset for the Transolver surrogate.

Per sample: seeded primitive geometry -> DeepSDF lattice reconstruction ->
sdf_hex mesh -> adjointOptimisationFoam (primal + adjoint) -> field extraction
on the surrogate query cloud -> one ``sample_SSSSS_V.npz`` in the dataset
output directory. Variant 0 is the reconstruction itself; variants >= 1 add
seeded Gaussian jitter to the latent control points (local coverage around
feasible shapes, matching optimizer step statistics).

Sharding for slurm arrays: ``--start`` / ``--count`` select the shape-seed
range; completed samples are skipped, so re-runs resume.

Usage:
    uv run python scripts/generate_flow_dataset.py \
        --config experiments/flow_dataset/config_dataset.json --start 0 --count 10
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from DeepSDFStruct.pretrained_models import get_model
from DeepSDFStruct.SDF import SDFfromDeepSDF

import deepshapeopt.foam_utils as foam_utils
from deepshapeopt.config import ExperimentSpecifications
from deepshapeopt.hexmesh import SdfHexMeshPipeline
from deepshapeopt.shape_optimization import (
    ModelSetup,
    build_lattice,
    run_reconstruction,
    setup_domain,
)
from deepshapeopt.surrogate.geometry_sampling import sample_geometry
from deepshapeopt.surrogate.query_points import ROLE_SURFACE, build_query_cloud

logger = logging.getLogger("generate_flow_dataset")


def _run_in_case(case_dir: Path, command: str) -> None:
    """Run a shell command inside the case with the OpenFOAM env sourced."""
    result = subprocess.run(
        f"source $WM_PROJECT_DIR/etc/bashrc >/dev/null 2>&1; {command}",
        cwd=case_dir,
        shell=True,
        executable="/bin/bash",
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"'{command}' failed in {case_dir}:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
        )


def extract_targets(case_dir: Path, cloud, surface_points: torch.Tensor) -> dict:
    """Read (U, p) targets for the query cloud from the reconstructed case.

    Wall values come from the foamToVTK boundary patch (point-interpolated),
    matched to the wall vertices by exact coordinates; shell and volume values
    are probed from ``internal.vtu`` with VTK's cell interpolation.
    """
    import pyvista as pv
    from scipy.spatial import cKDTree

    vtk_root = case_dir / "VTK"
    if vtk_root.exists():
        shutil.rmtree(vtk_root)
    # The adjoint solver's latestTime write lacks wallShearStress (the function
    # object writes at primal write times only); recompute it in place.
    _run_in_case(
        case_dir,
        "simpleFoam -postProcess -func wallShearStress -latestTime > log.wallShearStress 2>&1",
    )
    _run_in_case(case_dir, "foamToVTK -latestTime > log.foamToVTK 2>&1")

    vtk_dirs = [d for d in vtk_root.iterdir() if d.is_dir()]
    latest = max(vtk_dirs, key=lambda p: p.stat().st_mtime)

    patch = pv.read(latest / "boundary" / "dragObject.vtp")
    for name in ("p", "wallShearStress"):
        if name not in patch.point_data and name in patch.cell_data:
            patch = patch.cell_data_to_point_data(pass_cell_data=True)
            break

    pts_np = surface_points.detach().cpu().numpy().astype(np.float64)
    dist, idx = cKDTree(np.asarray(patch.points, dtype=np.float64)).query(pts_np)
    if dist.max() > 1e-4:
        raise RuntimeError(f"wall vertex match failed: max dist {dist.max():.3e}")
    p_wall = np.asarray(patch.point_data["p"], dtype=np.float32)[idx]
    if "wallShearStress" in patch.point_data:
        tau_w = np.asarray(patch.point_data["wallShearStress"], dtype=np.float32)[idx]
    else:
        logger.warning("wallShearStress missing on patch; storing zeros")
        tau_w = np.zeros((len(idx), 3), dtype=np.float32)

    pos = cloud.feats[:, :3].detach().cpu().numpy().astype(np.float64)
    roles = cloud.roles.cpu().numpy()
    P = cloud.n_surface
    off_surface = pos[P:]

    grid = pv.read(latest / "internal.vtu")
    probe = pv.PolyData(off_surface).sample(grid)
    U_probe = np.asarray(probe["U"], dtype=np.float32)
    p_probe = np.asarray(probe["p"], dtype=np.float32)
    valid_probe = np.asarray(probe["vtkValidPointMask"]) > 0.5

    N = pos.shape[0]
    U = np.zeros((N, 3), dtype=np.float32)  # no-slip: exact zeros on the wall
    p = np.empty(N, dtype=np.float32)
    valid = np.ones(N, dtype=bool)
    p[:P] = p_wall
    U[P:] = U_probe
    p[P:] = p_probe
    valid[P:] = valid_probe

    shutil.rmtree(vtk_root)
    return {
        "U": U,
        "p": p,
        "valid": valid,
        "tau_w": tau_w,
        "roles": roles,
        "n_invalid": int((~valid).sum()),
    }


def generate_sample(
    seed: int,
    variant: int,
    base_param: torch.Tensor,
    lattice_struct,
    hex_pipeline: SdfHexMeshPipeline,
    case_dir: Path,
    ds_cfg: dict,
    npz_path: Path,
    meta: dict,
) -> dict:
    """Mesh + solve + extract one (possibly jittered) variant; write the npz."""
    if variant == 0:
        param = base_param
    else:
        gen = torch.Generator(device="cpu").manual_seed(seed * 100 + variant)
        eps = float(ds_cfg["jitter_std"]) * torch.randn(
            base_param.shape, generator=gen
        ).to(base_param.device, base_param.dtype)
        param = base_param + eps
    lattice_struct.parametrization.set_param(param)

    hex_result = hex_pipeline.build()
    verts = hex_result.surface_points
    min_pts = int(ds_cfg.get("min_surface_points", 200))
    if verts.shape[0] < min_pts:
        raise RuntimeError(
            f"degenerate geometry: only {verts.shape[0]} wall points "
            f"(< {min_pts}); jitter likely collapsed the shape"
        )
    foam_case = hex_pipeline.run_case(case_dir, verbose=False)
    sens, J = hex_pipeline.load_sensitivities(
        case_dir, foam_case, objective_path=ds_cfg["objective_path"]
    )

    cloud = build_query_cloud(
        verts, hex_result.wall_tris_local, hex_pipeline.sdf_at_phys, ds_cfg["surrogate"]
    )
    targets = extract_targets(case_dir, cloud, verts)

    feats = cloud.feats.detach().cpu().numpy().astype(np.float32)
    np.savez_compressed(
        npz_path,
        pos=feats[:, :3],
        sdf=feats[:, 3],
        normal=feats[:, 4:7],
        role=targets["roles"].astype(np.int8),
        n_surface=np.int64(cloud.n_surface),
        delta=cloud.delta.cpu().numpy().astype(np.float32),
        area_normals=cloud.area_normals.detach().cpu().numpy().astype(np.float32),
        U=targets["U"],
        p=targets["p"],
        valid=targets["valid"],
        tau_w=targets["tau_w"],
        sens=sens.astype(np.float32),
        drag_foam=np.float64(J),
        param=param.detach().cpu().numpy().astype(np.float32),
        meta=json.dumps(meta),
    )
    return {
        "J": float(J),
        "n_points": int(feats.shape[0]),
        "n_surface": int(cloud.n_surface),
        "n_invalid": targets["n_invalid"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--start", type=int, default=0, help="first shape seed")
    parser.add_argument("--count", type=int, default=None, help="number of shape seeds")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("deepshapeopt").setLevel(logging.WARNING)

    specs = ExperimentSpecifications(args.config)
    rec_cfg = dict(specs["reconstruction"])
    opt_cfg = dict(specs["optimization"])
    ds_cfg = dict(specs["dataset"])
    config_dir = Path(args.config).resolve().parent

    n_shapes = int(ds_cfg["n_shapes"]) if args.count is None else args.count
    start = args.start
    n_variants = 1 + int(ds_cfg.get("jitter_per_shape", 0))

    out_dir = Path(ds_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    work_root = out_dir / "work"
    work_root.mkdir(exist_ok=True)

    device = rec_cfg["device"]
    model = get_model(
        model=rec_cfg["model_path"], checkpoint=rec_cfg["model_checkpoint"], device=device
    )
    sdf = SDFfromDeepSDF(model)

    runtime_root = opt_cfg.get("foam_runtime_root")
    case_dir = foam_utils.prepare_foam_runtime(
        config_dir / "foam_case",
        run_name=f"flow_dataset_s{start}",
        runtime_root=Path(runtime_root) if runtime_root else None,
    )
    foam_utils.select_allrun(case_dir, "sdf_hex")
    logger.info("Foam runtime case: %s", case_dir)

    n_done = n_failed = 0
    try:
        for seed in range(start, start + n_shapes):
            todo = [
                v for v in range(n_variants)
                if not (out_dir / f"sample_{seed:05d}_{v}.npz").exists()
            ]
            if not todo:
                continue

            sdir = work_root / f"sample_{seed:05d}"
            rec_dir = sdir / "reconstruction"
            rec_dir.mkdir(parents=True, exist_ok=True)

            geo = sample_geometry(
                seed,
                rec_cfg["design_domain"],
                margin=float(ds_cfg.get("geometry_margin", 0.15)),
            )
            stl_path = sdir / "shape.stl"
            geo.mesh.export(stl_path)

            rec_cfg_s = dict(rec_cfg)
            rec_cfg_s["mesh_path"] = str(stl_path)
            domain = setup_domain(rec_cfg_s, rec_dir)
            model_setup = ModelSetup(
                model=model, sdf=sdf, frame=domain.frame, mesh_orig=domain.mesh_orig
            )
            lattice = build_lattice(rec_cfg_s, model, sdf, model_setup.frame)
            recon_param = run_reconstruction(
                lattice.lattice_struct,
                model_setup.frame,
                model_setup.mesh_orig,
                rec_cfg_s,
                rec_dir,
                model,
                opt_cfg,
                debug=False,
            )
            base_param = recon_param[0].detach().clone()
            hex_pipeline = SdfHexMeshPipeline(
                lattice.lattice_struct, model_setup, opt_cfg, sdir
            )

            for variant in todo:
                name = f"sample_{seed:05d}_{variant}"
                t0 = time.time()
                try:
                    info = generate_sample(
                        seed,
                        variant,
                        base_param,
                        lattice.lattice_struct,
                        hex_pipeline,
                        case_dir,
                        ds_cfg,
                        out_dir / f"{name}.npz",
                        meta={
                            "family": geo.family,
                            "seed": seed,
                            "variant": variant,
                            "geometry": geo.meta,
                            "nu": ds_cfg["nu"],
                        },
                    )
                    n_done += 1
                    logger.info(
                        "%s family=%-12s J=%9.4f pts=%6d (surf %5d, invalid %d) [%.0f s]",
                        name, geo.family, info["J"], info["n_points"],
                        info["n_surface"], info["n_invalid"], time.time() - t0,
                    )
                except Exception:
                    n_failed += 1
                    logger.error("%s FAILED:\n%s", name, traceback.format_exc(limit=8))
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)

    logger.info("Done: %d samples written, %d failed.", n_done, n_failed)
    if n_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Export the surrogate's predicted (U, p) fields of one optimization
iteration for ParaView.

Rebuilds the sdf_hex wall mesh for a parameter snapshot from
``<results>/optimization/parameters_series/``, assembles the surrogate query
cloud, predicts (U, p) with the Transolver checkpoint of the run config and
writes into ``<results>/optimization/surrogate_fields/``:

- ``wall_iterNNNN.vtp``   -- wall surface (triangles) with p_pred, traction
  (pressure + viscous, force per area on the body), its drag component,
  and the unit normals
- ``cloud_iterNNNN.vtp``  -- every query point (wall, two probe shells,
  volume) with U_pred, |U_pred|, p_pred, role (0 wall, 1/2 shells, 3 volume)
  and the SDF feature; view as points / glyphs or Delaunay3D it in ParaView

With ``--with-foam`` the same shape is additionally solved with OpenFOAM
(one primal+adjoint run), the FOAM fields are probed onto the identical
query points (U_foam, p_foam, and the differences U_err, p_err) and the
foamToVTK output is copied next to it as ``foam_iterNNNN/``.

Usage:
    uv run python scripts/export_surrogate_fields.py \
        --config experiments/drag_cube/config_latent_cube_transolver_nopca.json \
        --iteration 59 [--with-foam]
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import pyvista as pv
import torch

import deepshapeopt.config as config
import deepshapeopt.foam_utils as foam_utils
from deepshapeopt.config import ExperimentSpecifications
from deepshapeopt.hexmesh import SdfHexMeshPipeline
from deepshapeopt.runtime import configure_logging
from deepshapeopt.shape_optimization import (
    build_lattice,
    run_reconstruction,
    setup_model_and_domain,
)
from deepshapeopt.surrogate.drag import drag_from_fields
from deepshapeopt.surrogate.predictor import TransolverSurrogate
from deepshapeopt.surrogate.query_points import build_query_cloud

LOGGER = logging.getLogger(__name__)


def _polydata_surface(verts: np.ndarray, faces: np.ndarray) -> pv.PolyData:
    n = faces.shape[0]
    cells = np.hstack([np.full((n, 1), 3, dtype=np.int64), faces.astype(np.int64)]).ravel()
    return pv.PolyData(verts.astype(np.float64), cells)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--iteration", type=int, default=None,
                        help="snapshot index (default: latest in parameters_series)")
    parser.add_argument("--with-foam", action="store_true",
                        help="also run OpenFOAM on the shape and export/probe the true fields")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    experiment_path = Path(args.config).resolve().parent
    specs = ExperimentSpecifications(args.config)
    rec_cfg, opt_cfg = specs["reconstruction"], specs["optimization"]
    if opt_cfg.get("mesh_pipeline") != "sdf_hex" or "surrogate" not in opt_cfg:
        raise SystemExit("needs an sdf_hex config with an optimization.surrogate block")

    paths = config.make_experiment_paths(
        experiment_path, results_name=specs.get("results_name", "results")
    )
    configure_logging(False, paths.optimization / "export_surrogate_fields.log")

    series = sorted((paths.optimization / "parameters_series").glob("param_*.pt"))
    if not series:
        raise SystemExit(f"no parameter snapshots in {paths.optimization / 'parameters_series'}")
    if args.iteration is None:
        snapshot = series[-1]
    else:
        snapshot = paths.optimization / "parameters_series" / f"param_{args.iteration:04d}.pt"
        if not snapshot.exists():
            raise SystemExit(f"{snapshot} not found ({len(series)} snapshots available)")
    it = int(snapshot.stem.split("_")[1])
    out_dir = Path(args.out_dir) if args.out_dir else paths.optimization / "surrogate_fields"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- geometry of that iteration --------------------------------------
    model_setup = setup_model_and_domain(rec_cfg, paths.reconstruction)
    lattice = build_lattice(rec_cfg, model_setup.model, model_setup.sdf, model_setup.frame)
    run_reconstruction(
        lattice.lattice_struct, model_setup.frame, model_setup.mesh_orig,
        rec_cfg, paths.reconstruction, model_setup.model, opt_cfg,
    )
    params = torch.load(snapshot, map_location=rec_cfg["device"])
    lattice.lattice_struct.parametrization.set_param(params[0].detach().to(torch.float32))
    hex_pipeline = SdfHexMeshPipeline(
        lattice.lattice_struct, model_setup, opt_cfg, paths.optimization
    )
    result = hex_pipeline.build()
    verts_t, faces_t = result.surface_points, result.wall_tris_local

    # --- surrogate prediction ---------------------------------------------
    surrogate = TransolverSurrogate.from_config(opt_cfg["surrogate"])
    with torch.no_grad():
        cloud = build_query_cloud(verts_t, faces_t, hex_pipeline.sdf_at_phys, surrogate.cfg)
        U, p = surrogate.predict(cloud)
        J, diag = drag_from_fields(
            U, p, cloud, nu=surrogate.nu, direction=surrogate.direction,
            u_inf=surrogate.u_inf, a_ref=surrogate.a_ref,
            visc_scale=surrogate.visc_scale, pressure_scale=surrogate.pressure_scale,
        )
    P = cloud.n_surface
    pts = cloud.feats[:, :3].detach().cpu().numpy()
    sdf = cloud.feats[:, 3].detach().cpu().numpy()
    U_np, p_np = U.cpu().numpy(), p.cpu().numpy()
    roles = cloud.roles.cpu().numpy().astype(np.int32)
    normals = cloud.unit_normals.detach().cpu().numpy()
    traction = diag["traction"].detach().cpu().numpy()
    e_dir = np.asarray(surrogate.direction, dtype=np.float64)

    wall = _polydata_surface(verts_t.detach().cpu().numpy(), faces_t.cpu().numpy())
    wall.point_data["p_pred"] = p_np[:P]
    wall.point_data["traction"] = traction
    wall.point_data["traction_drag"] = traction @ e_dir
    wall.point_data["normal"] = normals
    wall.field_data["J_surrogate"] = [float(J)]
    wall.field_data["J_pressure"] = [diag["J_p"]]
    wall.field_data["J_viscous"] = [diag["J_visc"]]

    cloud_pd = pv.PolyData(pts.astype(np.float64))
    cloud_pd.point_data["U_pred"] = U_np
    cloud_pd.point_data["U_mag_pred"] = np.linalg.norm(U_np, axis=1)
    cloud_pd.point_data["p_pred"] = p_np
    cloud_pd.point_data["role"] = roles
    cloud_pd.point_data["sdf"] = sdf

    # --- optional OpenFOAM reference on the same points --------------------
    if args.with_foam:
        from generate_flow_dataset import extract_targets  # scripts/ is on sys.path

        case_dir = foam_utils.prepare_foam_runtime(
            experiment_path / "foam_case", run_name=f"{specs['results_name']}_fields",
            runtime_root=Path(opt_cfg["foam_runtime_root"]) if opt_cfg.get("foam_runtime_root") else None,
        )
        foam_utils.select_allrun(case_dir, "sdf_hex")
        try:
            foam_case = hex_pipeline.run_case(case_dir, verbose=False)
            _, J_foam = hex_pipeline.load_sensitivities(case_dir, foam_case)
            # extract_targets runs foamToVTK; copy its output before it is removed
            targets = extract_targets(case_dir, cloud, verts_t)
        finally:
            vtk_root = case_dir / "VTK"
            if vtk_root.exists():
                latest = max((d for d in vtk_root.iterdir() if d.is_dir()), key=lambda d: d.stat().st_mtime)
                dst = out_dir / f"foam_iter{it:04d}"
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(latest, dst)
            shutil.rmtree(case_dir, ignore_errors=True)

        valid = targets["valid"]
        cloud_pd.point_data["U_foam"] = targets["U"]
        cloud_pd.point_data["p_foam"] = targets["p"]
        cloud_pd.point_data["valid_foam"] = valid.astype(np.int32)
        cloud_pd.point_data["U_err"] = np.where(valid[:, None], U_np - targets["U"], np.nan)
        cloud_pd.point_data["p_err"] = np.where(valid, p_np - targets["p"], np.nan)
        wall.point_data["p_foam"] = targets["p"][:P]
        wall.point_data["p_err"] = p_np[:P] - targets["p"][:P]
        wall.point_data["wallShearStress_foam"] = targets["tau_w"]
        wall.field_data["J_foam"] = [float(J_foam)]
        LOGGER.info("J_foam = %.4f", float(J_foam))

    wall_path = out_dir / f"wall_iter{it:04d}.vtp"
    cloud_path = out_dir / f"cloud_iter{it:04d}.vtp"
    wall.save(wall_path)
    cloud_pd.save(cloud_path)

    print(f"iteration {it}: J_surrogate = {float(J):.4f} "
          f"(pressure {diag['J_p']:.3f}, viscous {diag['J_visc']:.3f}), "
          f"{cloud.n_points} query points ({P} wall)")
    if args.with_foam:
        print(f"              J_foam      = {float(J_foam):.4f}")
    print(f"written: {wall_path}\n         {cloud_path}")
    if args.with_foam:
        print(f"         {out_dir / f'foam_iter{it:04d}'}/ (foamToVTK: internal.vtu + boundary/)")


if __name__ == "__main__":
    main()

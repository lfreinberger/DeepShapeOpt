"""Finite-difference check of the full sdf_hex gradient chain.

Compares the adjoint gradient dJ/dz (OpenFOAM point sensitivities ->
differentiable snap -> latent code) against central finite differences of
the actual CFD objective, for the latent components with the largest
adjoint gradient (best signal-to-noise).

Each FD evaluation is a full OpenFOAM run (primal + adjoint, ~1 min), so a
check of k components costs 2k + 1 runs.  The castellation is frozen
(``build(reuse_castellation=True)``) for the perturbed evaluations so that
J(z) is smooth: only the snapped wall points move, the mesh topology is
identical.

Expected agreement is a few percent, not machine precision: the FD moves
only the boundary points of a fixed interior mesh, while the ESI adjoint
sensitivities assume the interior deforms with the boundary; both converge
to the same continuous shape derivative under refinement.  If the mismatch
is large, rerunning with ``includeMeshMovement false`` (SI sensitivities)
in optimisationDict is a useful diagnostic.

Usage:
    uv run python scripts/check_gradient_fd.py \
        --config experiments/drag_cube/config_sdfhex_validation.json \
        --n-components 3 --eps 2e-3
"""

from __future__ import annotations

import argparse
import logging
import shutil
import time
from pathlib import Path

import torch
from foamlib import FoamCase

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

LOGGER = logging.getLogger(__name__)


def evaluate(hex_pipeline, case_dir, sens_cfg, reuse_castellation):
    """One pipeline evaluation: mesh -> OpenFOAM -> (result, sens, J)."""
    result = hex_pipeline.build(reuse_castellation=reuse_castellation)
    foam_case = hex_pipeline.run_case(case_dir, verbose=False)
    sens, J = hex_pipeline.load_sensitivities(
        case_dir,
        foam_case,
        field_name=sens_cfg.get("field_name", "pointSensVecadjS1ESI"),
        objective_path=sens_cfg.get("objective_path", "optimisation/objective/0/dragadjS1"),
    )
    return result, sens, float(J)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Experiment JSON (sdf_hex)")
    parser.add_argument("--n-components", type=int, default=3,
                        help="Number of latent components to check (largest |dJ|)")
    parser.add_argument("--eps", type=float, default=2e-3,
                        help="Central-difference step per component")
    args = parser.parse_args()

    experiment_path = Path(args.config).resolve().parent
    specs = ExperimentSpecifications(args.config)
    rec_cfg = specs["reconstruction"]
    opt_cfg = specs["optimization"]
    sens_cfg = opt_cfg.get("sensitivity", {})
    if opt_cfg.get("mesh_pipeline") != "sdf_hex":
        raise SystemExit("This check requires mesh_pipeline: 'sdf_hex' in the config")

    results_name = specs.get("results_name", "results")
    paths = config.make_experiment_paths(experiment_path, results_name=results_name)
    config.ensure_experiment_dirs(paths)
    configure_logging(False, paths.optimization / "fd_check.log")

    # --- setup identical to the optimization script -----------------------
    model_setup = setup_model_and_domain(rec_cfg, paths.reconstruction)
    lattice = build_lattice(rec_cfg, model_setup.model, model_setup.sdf, model_setup.frame)
    run_reconstruction(
        lattice.lattice_struct, model_setup.frame, model_setup.mesh_orig,
        rec_cfg, paths.reconstruction, model_setup.model, opt_cfg,
    )
    param = next(lattice.lattice_struct.parametrization.parameters())
    hex_pipeline = SdfHexMeshPipeline(
        lattice.lattice_struct, model_setup, opt_cfg, paths.optimization
    )
    foam_runtime_root = opt_cfg.get("foam_runtime_root")
    case_dir = foam_utils.prepare_foam_runtime(
        experiment_path / "foam_case", run_name=f"{results_name}_fd_check",
        runtime_root=Path(foam_runtime_root) if foam_runtime_root else None,
    )
    foam_utils.select_allrun(case_dir, "sdf_hex")

    try:
        # --- base evaluation: adjoint gradient -----------------------------
        t0 = time.time()
        result, sens, J0 = evaluate(hex_pipeline, case_dir, sens_cfg, reuse_castellation=False)
        dJ = foam_utils.compute_shape_gradient(
            param, result.surface_points, result.wall_tris_local, sens, integrated=True
        ).reshape(-1)
        LOGGER.info("Base run: J = %.6e (%.0f s)", J0, time.time() - t0)

        # When PCA latent reduction is enabled, validate the *projected* gradient
        # dJ/dc against FD in coefficient space: perturbing coefficient (p, j)
        # moves z[p] along the principal direction V[:, j].
        pca_cfg = opt_cfg.get("pca", {})
        if pca_cfg.get("enabled", False):
            from deepshapeopt.latent_pca import build_pca_basis

            basis = build_pca_basis(
                rec_cfg["model_path"], rec_cfg.get("model_checkpoint", "latest"),
                int(pca_cfg["n_components"]),
                device=param.device, cache_path=pca_cfg.get("cache_path"),
            ).to(device=param.device, dtype=param.dtype)
            k = basis.n_components
            dc = basis.project_grad(dJ).reshape(-1, k)  # (n_cp, k) adjoint dJ/dc
            param2d = param.data.reshape(-1, basis.latent_dim)
            top = torch.argsort(dc.reshape(-1).abs(), descending=True)[: args.n_components]
            rows = []
            for flat_idx in top.tolist():
                p, j = divmod(flat_idx, k)
                adj = float(dc[p, j])
                direction = basis.components[:, j]  # z-direction for coeff (p, j)
                LOGGER.info("Coeff (cp %d, comp %d): adjoint dJ/dc = %.6e", p, j, adj)
                param2d[p] += args.eps * direction
                _, _, J_plus = evaluate(hex_pipeline, case_dir, sens_cfg, reuse_castellation=True)
                param2d[p] -= 2 * args.eps * direction
                _, _, J_minus = evaluate(hex_pipeline, case_dir, sens_cfg, reuse_castellation=True)
                param2d[p] += args.eps * direction  # restore

                fd = (J_plus - J_minus) / (2 * args.eps)
                rel = abs(adj - fd) / max(abs(fd), 1e-30)
                rows.append((flat_idx, adj, fd, rel))
                LOGGER.info(
                    "Coeff (cp %d, comp %d): adjoint %.6e vs FD %.6e -> rel err %.2f%%",
                    p, j, adj, fd, 100 * rel,
                )
        else:
            top = torch.argsort(dJ.abs(), descending=True)[: args.n_components]
            flat = param.data.reshape(-1)
            rows = []
            for idx in top.tolist():
                adj = float(dJ[idx])
                LOGGER.info(
                    "Component %d: adjoint dJ = %.6e, predicted dJ*2eps = %.3e",
                    idx, adj, abs(adj) * 2 * args.eps,
                )
                flat[idx] += args.eps
                _, _, J_plus = evaluate(hex_pipeline, case_dir, sens_cfg, reuse_castellation=True)
                flat[idx] -= 2 * args.eps
                _, _, J_minus = evaluate(hex_pipeline, case_dir, sens_cfg, reuse_castellation=True)
                flat[idx] += args.eps  # restore

                fd = (J_plus - J_minus) / (2 * args.eps)
                rel = abs(adj - fd) / max(abs(fd), 1e-30)
                rows.append((idx, adj, fd, rel))
                LOGGER.info(
                    "Component %d: adjoint %.6e vs FD %.6e (J+ %.6e, J- %.6e) -> rel err %.2f%%",
                    idx, adj, fd, J_plus, J_minus, 100 * rel,
                )

        print(f"\nBase objective J = {J0:.6e}, eps = {args.eps:g}")
        print(f"{'comp':>6} {'adjoint dJ/dz':>16} {'FD dJ/dz':>16} {'rel err':>9}")
        for idx, adj, fd, rel in rows:
            print(f"{idx:>6} {adj:>16.6e} {fd:>16.6e} {100 * rel:>8.2f}%")
        adj_v = torch.tensor([r[1] for r in rows])
        fd_v = torch.tensor([r[2] for r in rows])
        cos = torch.nn.functional.cosine_similarity(adj_v, fd_v, dim=0).item()
        print(f"cosine over checked components: {cos:.4f}")
    finally:
        if case_dir.exists():
            FoamCase(case_dir).clean()
            shutil.rmtree(case_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

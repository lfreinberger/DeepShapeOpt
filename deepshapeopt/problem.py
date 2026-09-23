"""Assemble the building blocks of one optimization run from its config."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config, RunPaths, make_run_paths
from .config.loader import archive_config
from .logging_setup import configure_logging
from .mesher import SdfHexMesher
from .parametrization.base import DesignSpace
from .solvers.metrics import Metric, get_metric
from .terms import build_constraints, build_objective, build_penalties
from .terms.base import ConstraintTerm, PenaltyTerm
from .terms.cfd import CfdConstraint, CfdObjective

logger = logging.getLogger(__name__)


@dataclass
class Problem:
    cfg: Config
    paths: RunPaths
    parametrization: object
    space: DesignSpace
    mesher: SdfHexMesher
    solver: object
    objective: CfdObjective
    penalties: list[PenaltyTerm]
    constraints: list[ConstraintTerm]
    metrics: list[Metric] = field(default_factory=list)

    @property
    def param(self):
        return self.parametrization.param


def run_name(cfg: Config) -> str:
    """Name of the transient solver case: ``run.name`` without a leading ``results`` prefix."""
    name = cfg.run.name
    for prefix in ("results_", "results"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name.strip("_") or "run"


def build_problem(cfg: Config, experiment_dir: Path) -> Problem:
    cfg.require_optimization()
    paths = make_run_paths(cfg, experiment_dir).ensure()
    configure_logging(cfg.run.debug, paths.optimization / "run.log")
    archive_config(cfg.raw, paths.optimization / "config_log.json")
    debug = cfg.run.debug
    logger.info("Run %s: %s parametrization, %s solver, objective %s, %d constraint(s)",
                cfg.run.name, cfg.parametrization.type, cfg.solver.type, cfg.objective.metric, len(cfg.constraints))

    if cfg.parametrization.type == "deepsdf":
        from .parametrization.deepsdf import DeepSDFLattice

        parametrization = DeepSDFLattice(cfg, paths)
        parametrization.reconstruct(paths, debug)
        model_path, checkpoint = cfg.parametrization.deepsdf.model_path, cfg.parametrization.deepsdf.checkpoint
        pca_cfg = cfg.parametrization.deepsdf.pca
    else:
        from .parametrization.ffd import FFDParametrization

        parametrization = FFDParametrization(cfg, paths)
        model_path, checkpoint, pca_cfg = None, "latest", None

    space = DesignSpace(parametrization.param, parametrization.spline_sp, parametrization.frame,
                        cfg.parametrization.lock, cfg.optimizer, pca_cfg, model_path, checkpoint)
    if cfg.parametrization.type == "ffd":
        parametrization.check_locked_faces(space.mask_locked_cp)
    if debug:
        parametrization.export_debug(paths.reconstruction, space.locked_idx if space.locked_idx.numel() else None)

    sdf_hex = dict(cfg.mesh.sdf_hex)
    if "outlet_interior" in sdf_hex:
        sdf_hex["outlet_interior"] = {**sdf_hex["outlet_interior"], "debug": debug}
    mesher = SdfHexMesher(parametrization.design_sdf, parametrization.frame, parametrization.mesh_orig, sdf_hex,
                          flow=cfg.geometry.flow, unit_to_metre=cfg.geometry.unit_to_metre,
                          results_dir=paths.optimization)

    objective = build_objective(cfg)
    penalties = build_penalties(cfg)
    constraints = build_constraints(cfg)
    metrics = [objective.metric] + [c.metric for c in constraints if isinstance(c, CfdConstraint)]

    template = paths.experiment / cfg.solver.case
    if cfg.solver.type == "dafoam":
        from .solvers.dafoam.solver import DAFoamSolver

        solver = DAFoamSolver(cfg.solver.dafoam, template, run_name(cfg), paths.scratch, metrics, mesher, verbose=debug)
    else:
        from .solvers.openfoam.solver import OpenFoamAdjointSolver

        solver = OpenFoamAdjointSolver(cfg.solver.openfoam, template, run_name(cfg), paths.scratch, metrics,
                                       mesher, verbose=debug)
    return Problem(cfg=cfg, paths=paths, parametrization=parametrization, space=space, mesher=mesher,
                   solver=solver, objective=objective, penalties=penalties, constraints=constraints, metrics=metrics)


__all__ = ["Problem", "build_problem", "get_metric", "run_name"]

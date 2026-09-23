"""OpenFOAM continuous adjoint (``adjointOptimisationFoam``) as forward solver."""

from __future__ import annotations

import contextlib
import io
import logging
import time
from pathlib import Path
from typing import Sequence

from foamlib import FoamCase

from deepshapeopt.config.schema import OpenFoamConfig
from deepshapeopt.mesher import MeshResult, SdfHexMesher

from ..base import SolverResult
from ..metrics import Metric
from . import runtime
from .export import export_vtk_for_iteration
from .sensitivities import read_objective, read_point_sensitivities

logger = logging.getLogger(__name__)


class OpenFoamAdjointSolver:
    """Copies the case template once per run; per evaluation writes the mesh, runs Allrun
    (decomposePar, checkMesh, adjointOptimisationFoam, reconstructPar) and reads the
    objective files and integrated point sensitivities of the metrics."""

    def __init__(self, cfg: OpenFoamConfig, template: Path, run_name: str, scratch_root: Path | None,
                 metrics: Sequence[Metric], mesher: SdfHexMesher, verbose: bool = False):
        self.cfg = cfg
        self.mesher = mesher
        self.verbose = verbose
        self.metrics = list(metrics)
        self.case_dir = runtime.prepare_foam_runtime(Path(template), run_name=run_name, runtime_root=scratch_root)
        runtime.select_allrun(self.case_dir)
        self.adjoint_times = runtime.configure_foam_runtime(
            self.case_dir, active_solvers={m.of_solver for m in self.metrics},
            solver_convergence=cfg.solver_convergence,
        )
        runtime.apply_foam_dict_overrides(self.case_dir, cfg.dict_overrides)
        if cfg.solver_convergence:
            logger.info("Solver convergence: %s", cfg.solver_convergence)
        logger.info("OpenFOAM case %s, adjoint write times %s", self.case_dir, self.adjoint_times)

    def evaluate(self, mesh: MeshResult, metrics: Sequence[Metric]) -> SolverResult:
        t0 = time.time()
        foam_case = FoamCase(self.case_dir)
        self._quiet(foam_case.clean)
        self.mesher.write_polymesh(self.case_dir)
        runtime.run_openfoam_case(self.case_dir, verbose=self.verbose, clean=False)
        self.mesher.check_mesh(self.case_dir)

        log_path = self.case_dir / "log.adjointOptimisationFoam"
        iterations = runtime.parse_solver_iterations(log_path)
        if iterations:
            logger.info("Solver iterations: %s", runtime.format_solver_iterations(iterations))

        values, sens = {}, {}
        for metric in metrics:
            field = metric.field_name(self.cfg.sensitivity.field_suffix)
            sens[metric.name] = read_point_sensitivities(
                self.case_dir, field, self.adjoint_times[metric.of_solver],
                mesh.wall_point_ids, mesh.n_points,
            )
            values[metric.name] = float(read_objective(self.case_dir, metric.objective_file))
        return SolverResult(
            values=values, sensitivities=sens, case_dir=self.case_dir, time_s=time.time() - t0,
            foam_case=foam_case, adjoint_times=dict(self.adjoint_times),
            info={"solver_iterations": iterations, "log": log_path},
        )

    def export_fields(self, result: SolverResult, iteration: int, out_dir: Path) -> None:
        time_value = next(iter(result.adjoint_times.values()), None)
        export_vtk_for_iteration(result.foam_case, self.case_dir, out_dir, iteration, time_value=time_value)

    def clean_iteration(self, result: SolverResult) -> None:
        self._quiet(result.foam_case.clean)

    def close(self) -> None:
        if self.case_dir.exists():
            self._quiet(FoamCase(self.case_dir).clean)
            import shutil

            shutil.rmtree(self.case_dir, ignore_errors=True)

    def _quiet(self, fn) -> None:
        if self.verbose:
            fn()
            return
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            fn()

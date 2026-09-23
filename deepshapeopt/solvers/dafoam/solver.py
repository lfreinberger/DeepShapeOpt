"""DAFoam discrete adjoint as forward solver (runs inside the Apptainer image)."""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Sequence

from deepshapeopt.mesher import MeshResult, SdfHexMesher

from ..base import SolverResult
from ..metrics import Metric
from .functions import resolve_outputs
from .runner import DAFoamRunner, prepare_dafoam_case

logger = logging.getLogger(__name__)


class DAFoamSolver:
    def __init__(self, dafoam_cfg: dict, template: Path, run_name: str, scratch_root: Path | None,
                 metrics: Sequence[Metric], mesher: SdfHexMesher, verbose: bool = False):
        self.mesher = mesher
        self.outputs = resolve_outputs(metrics, dafoam_cfg)
        self.case_dir = prepare_dafoam_case(Path(template), run_name=run_name, runtime_root=scratch_root)
        self.runner = DAFoamRunner(dafoam_cfg, self.case_dir, verbose=verbose)
        logger.info("DAFoam case %s (%s, %d procs)", self.case_dir, self.runner.dcfg.container, self.runner.dcfg.n_procs)

    def evaluate(self, mesh: MeshResult, metrics: Sequence[Metric]) -> SolverResult:
        t0 = time.time()
        self.mesher.write_polymesh(self.case_dir)
        wanted = [self.outputs[m.name] for m in metrics]
        result = self.runner.evaluate(mesh.n_points, wanted)
        logger.info(
            "DAFoam: %.1f s (primal %.1f s, mesh %s), outputs %s",
            result["time_total_s"], result.get("time_primal_s", float("nan")),
            "OK" if result.get("mesh_ok") == 1 else "flagged", result["outputs"],
        )
        values, sens = {}, {}
        for metric in metrics:
            s, J = self.runner.load_sensitivities(self.outputs[metric.name], mesh.wall_point_ids, mesh.n_points)
            sens[metric.name], values[metric.name] = s, float(J)
        return SolverResult(values=values, sensitivities=sens, case_dir=self.case_dir,
                            time_s=time.time() - t0, info={"dafoam": result})

    def export_fields(self, result: SolverResult, iteration: int, out_dir: Path) -> None:
        logger.debug("DAFoam writes no OpenFOAM time directories; no field export for iteration %d", iteration)

    def clean_iteration(self, result: SolverResult) -> None:
        self.runner.clean()

    def close(self) -> None:
        shutil.rmtree(self.case_dir, ignore_errors=True)

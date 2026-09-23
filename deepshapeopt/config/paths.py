"""Result and scratch locations of one run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .schema import Config


@dataclass(frozen=True)
class RunPaths:
    experiment: Path      # directory of the config (case templates live here)
    results: Path         # <experiment>/<run.name>
    reconstruction: Path  # results/reconstruction
    optimization: Path    # results/optimization
    heavy_data: Path | None  # mirrored heavy exports (VTK/STL series), or None
    scratch: Path | None  # node-local root for the transient solver case, or None

    def ensure(self) -> "RunPaths":
        for p in (self.results, self.reconstruction, self.optimization):
            p.mkdir(parents=True, exist_ok=True)
        if self.heavy_data is not None:
            self.heavy_data.mkdir(parents=True, exist_ok=True)
        return self


def _project_root(start: Path) -> Path:
    root = start
    while root != root.parent:
        if (root / "pyproject.toml").exists():
            return root
        root = root.parent
    return start


def make_run_paths(cfg: Config, experiment_dir: str | Path) -> RunPaths:
    """Results next to the config; heavy data mirrors the experiment path under
    ``run.heavy_data_dir`` (only when debug exports are on)."""
    experiment = Path(experiment_dir).resolve()
    results = experiment / cfg.run.name
    heavy = None
    if cfg.run.heavy_data_dir and cfg.run.debug:
        root = _project_root(experiment)
        try:
            rel = experiment.relative_to(root)
        except ValueError:
            rel = Path(experiment.name)
        heavy = Path(cfg.run.heavy_data_dir) / rel / cfg.run.name
    scratch = Path(cfg.run.scratch_dir) if cfg.run.scratch_dir else None
    return RunPaths(
        experiment=experiment,
        results=results,
        reconstruction=results / "reconstruction",
        optimization=results / "optimization",
        heavy_data=heavy,
        scratch=scratch,
    )

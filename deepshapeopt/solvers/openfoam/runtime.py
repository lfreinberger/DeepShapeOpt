"""Runtime copy of an OpenFOAM case template and its per-run configuration."""
from __future__ import annotations

import contextlib
import io
import logging
import re
import shutil
import subprocess
import time
from pathlib import Path

from foamlib import FoamCase, FoamFile

logger = logging.getLogger(__name__)


def prepare_foam_runtime(
    template_dir: Path, run_name: str, runtime_root: Path | None = None
) -> Path:
    """Copy foam_case template to an isolated runtime directory for concurrent execution.

    By default the runtime case is created next to the template
    (``template_dir.parent / foam_run_<run_name>``). Pass ``runtime_root`` to place
    it elsewhere -- e.g. node-local scratch (``/work``) instead of the backed-up
    file server -- which keeps the per-iteration decomposePar/solver/reconstructPar
    I/O off NFS. The root is created if missing; only the transient case moves, all
    callers reference the returned ``case_dir`` directly.
    """
    if runtime_root is not None:
        runtime_root = Path(runtime_root)
        runtime_root.mkdir(parents=True, exist_ok=True)
        runtime_dir = runtime_root / f"foam_run_{run_name}"
    else:
        runtime_dir = template_dir.parent / f"foam_run_{run_name}"
    if runtime_dir.exists():
        shutil.rmtree(runtime_dir)
    shutil.copytree(template_dir, runtime_dir)
    return runtime_dir

SOLVER_PATHS = {
    "p1": ("primalSolvers", "p1", "solutionControls"),
    "as1": ("adjointManagers", "am1", "adjointSolvers", "as1", "solutionControls"),
    "as2": ("adjointManagers", "am1", "adjointSolvers", "as2", "solutionControls"),
}

PRIMAL_FIELDS = {"simple": {"p", "U"}, "simpleHeatTransfer": {"p", "U", "T"}}

WRITE_NEVER = 10 ** 9

def _apply_solver_convergence(opt, cfg: dict, case_dir: Path) -> None:
    """Write the ``solver_convergence`` config block into the runtime dicts.

    Modes: ``as_template`` (leave everything alone), ``fixed`` (write nIters and
    DELETE residualControl, so an early exit is structurally impossible instead
    of numerically improbable), ``residual`` (nIters as a cap plus thresholds).
    Must run BEFORE the nIters are read for the time prediction.
    """
    mode = cfg.get("mode", "as_template")
    if mode == "as_template":
        return
    if mode not in ("fixed", "residual"):
        raise ValueError(f"solver_convergence.mode must be 'as_template', 'fixed' "
                         f"or 'residual', got {mode!r}")

    for name, spec in (cfg.get("solvers") or {}).items():
        if name not in SOLVER_PATHS:
            raise ValueError(f"solver_convergence: unknown solver {name!r}; "
                             f"valid: {sorted(SOLVER_PATHS)}")
        base = SOLVER_PATHS[name]
        if spec.get("n_iters") is not None:
            opt[base + ("nIters",)] = int(spec["n_iters"])

        if mode == "fixed":
            try:
                del opt[base + ("residualControl",)]
            except (KeyError, TypeError):
                pass
            continue

        residuals = spec.get("residuals")
        if not residuals:
            raise ValueError(f"solver_convergence: mode 'residual' needs "
                             f"'residuals' for solver {name!r}")
        if name == "p1":
            solver = str(opt["primalSolvers", "p1", "solver"])
            expected = PRIMAL_FIELDS.get(solver, {"p", "U"})
            missing = expected - set(residuals)
            if missing:
                raise ValueError(
                    f"solver_convergence: primal solver {solver!r} also solves "
                    f"{sorted(missing)}; without a threshold those equations are "
                    "not part of the convergence criterion and the run would "
                    "report convergence with them unconverged")
        # Regex keys must carry literal quotes -- an unquoted key is a literal
        # keyword to OpenFOAM and would never match a field.
        opt[base + ("residualControl",)] = {
            f'"{field}.*"': float(value) for field, value in residuals.items()
        }

    if cfg.get("write_only_end_states", mode == "residual"):
        # Only the forced end-of-solver writes remain, so the time directories
        # are an unambiguous record of where each solver stopped -- and
        # reconstructPar has three of them instead of one per writeInterval.
        FoamFile(case_dir / "system" / "controlDict")["writeInterval"] = WRITE_NEVER

def configure_foam_runtime(
    case_dir: Path,
    constraint_enabled: bool,
    section_patches: list[tuple[str, float]] | None = None,
    as1_active: bool = True,
    solver_convergence: dict | None = None,
) -> dict[str, str]:
    """Derive adjoint-time directories from optimisationDict and patch the runtime case.

    Reads primal/adjoint ``nIters`` from ``system/optimisationDict`` and computes the
    time directory each *active* adjoint solver will write to: the solvers run back to
    back, so ``t_as1 = p_n + as1_n`` and ``t_as2 = t_as1 + as2_n`` (or ``p_n + as2_n``
    when as1 is inactive). These are predictions: an adjoint solver that satisfies its
    ``residualControl`` before ``nIters`` shifts every later write time -- use
    :func:`resolve_adjoint_time` when reading the fields back.

    Also mutates the runtime copy:
      - ``optimisationDict``: sets ``am1.as1.active`` to ``as1_active`` and
        ``am1.as2.active`` to ``constraint_enabled`` (an inactive solver is skipped
        entirely, which also avoids solving a degenerate adjoint -- e.g. a uniformity
        objective on a flow that is already uniform stops after one step)
      - ``controlDict``: forces ``purgeWrite = 0`` so no needed time dir is purged
      - ``Allrun``: replaces the ``__ADJOINT_TIMES__`` marker with the open time range
        ``"1:"``, so ``reconstructPar`` covers every write whatever the solvers do.
        It must NOT start at ``p_n``: when the primal exits early on
        ``residualControl`` the adjoint end times move down with it and can fall
        BELOW that bound, in which case reconstructPar produces nothing and
        :func:`resolve_adjoint_time` -- which can only find what was reconstructed --
        raises. ``deltaT`` is 1 and iteration 1 can never satisfy the criterion, so
        the earliest possible write is time 2 and ``1:`` cannot clip a real one.

    When ``section_patches`` is provided (list of ``(patch_name, target_fraction)`` pairs),
    additionally:
      - ``optimisationDict``: replaces the ``as1`` objectiveNames block with a single
        ``flowBalance`` objective of type ``flowRatePartition`` over the listed sections.
      - ``snappyHexMeshDict``: removes ``outlet``/``outletInterior`` region entries from
        ``geometry.shape.stl.regions`` and ``castellatedMeshControls.refinementSurfaces.
        extrusion_die.regions``, replacing them with one entry per section (inheriting
        the prior outlet refinement level).

    ``solver_convergence`` is the experiment's config block; it is applied to the
    runtime dicts FIRST, so the predicted times use the values that will actually
    run. See :func:`_apply_solver_convergence`.

    Should be called once after ``prepare_foam_runtime`` copies the template.
    Operating on the template directly would dirty the git-tracked files.

    Returns a mapping of the active solvers to their predicted write times, e.g.
    ``{"as1": "300"}``, ``{"as1": "300", "as2": "500"}`` or ``{"as2": "500"}``.
    """
    opt_path = case_dir / "system" / "optimisationDict"
    ctrl_path = case_dir / "system" / "controlDict"
    allrun_path = case_dir / "Allrun"

    if not (as1_active or constraint_enabled):
        raise ValueError("configure_foam_runtime: at least one adjoint solver must be active")

    opt = FoamFile(opt_path)
    if solver_convergence:
        _apply_solver_convergence(opt, solver_convergence, case_dir)
    p_n = int(opt["primalSolvers", "p1", "solutionControls", "nIters"])
    as1_n = int(opt["adjointManagers", "am1", "adjointSolvers", "as1", "solutionControls", "nIters"])
    as2_n = int(opt["adjointManagers", "am1", "adjointSolvers", "as2", "solutionControls", "nIters"])

    opt["adjointManagers", "am1", "adjointSolvers", "as1", "active"] = bool(as1_active)
    opt["adjointManagers", "am1", "adjointSolvers", "as2", "active"] = bool(constraint_enabled)

    if section_patches is not None:
        _inject_section_partition_into_runtime(case_dir, opt, section_patches)

    t = p_n
    adjoint_times: dict[str, str] = {}
    if as1_active:
        t += as1_n
        adjoint_times["as1"] = str(t)
    if constraint_enabled:
        t += as2_n
        adjoint_times["as2"] = str(t)

    ctrl = FoamFile(ctrl_path)
    ctrl["purgeWrite"] = 0

    marker = "__ADJOINT_TIMES__"
    text = allrun_path.read_text()
    if marker not in text:
        raise RuntimeError(f"{allrun_path}: missing {marker} marker in Allrun template")
    allrun_path.write_text(text.replace(marker, "1:"))

    return adjoint_times

RX_CONVERGED = re.compile(
    r"^(\w+) solution converged in ([0-9.eE+-]+) iterations", re.M)

RX_MAXITERS = re.compile(
    r"^(\w+) solution reached max\. number of iterations ([0-9]+)", re.M)

def parse_solver_iterations(log_path: Path) -> dict[str, dict]:
    """How many iterations each solver in ``log.adjointOptimisationFoam`` ran.

    Returns ``{"p1": {"iters": int, "end_time": int, "reason": str}, ...}``;
    solvers that were inactive print nothing and are simply absent. Nothing else
    in the pipeline records this -- the predicted times in
    :func:`configure_foam_runtime` are only right while every solver runs its
    full ``nIters``.
    """
    log_path = Path(log_path)
    if not log_path.is_file():
        return {}
    text = log_path.read_text(errors="ignore")

    events = []
    for m in RX_CONVERGED.finditer(text):
        events.append((m.start(), m.group(1), float(m.group(2)), "converged"))
    for m in RX_MAXITERS.finditer(text):
        events.append((m.start(), m.group(1), float(m.group(2)), "max_iters"))
    events.sort()

    out, start = {}, 0.0
    for _pos, solver, value, reason in events:
        end = value if reason == "converged" else start + value
        out[solver] = {"iters": int(round(end - start)),
                       "end_time": int(round(end)), "reason": reason}
        start = end
    return out

def format_solver_iterations(info: dict[str, dict], caps: dict[str, int] | None = None) -> str:
    """One-line summary for the run log, e.g. ``p1 412/2000 converged, as1 ...``."""
    caps = caps or {}
    parts = []
    for name, d in info.items():
        cap = f"/{caps[name]}" if name in caps else ""
        parts.append(f"{name} {d['iters']}{cap} {d['reason']}")
    return ", ".join(parts) if parts else "(no termination message in the log)"

def apply_foam_dict_overrides(case_dir: Path, overrides: dict | None) -> None:
    """Set individual entries of OpenFOAM dictionaries in the RUNTIME case copy.

    ``overrides`` maps a case-relative file (``"system/optimisationDict"``) to a mapping
    of slash-separated entry paths to values, e.g.::

        {"system/optimisationDict":
            {"adjointManagers/am1/adjointSolvers/as1/ATCModel/ATCModel": "cancel"}}

    Values are written by foamlib as given (a str becomes a bare word, numbers and
    bools their OpenFOAM spelling). Meant for one-off studies that vary a solver
    setting without forking the experiment's ``foam_case`` template; call it AFTER
    ``configure_foam_runtime`` so the override wins over the derived settings.
    """
    if not overrides:
        return
    for rel_file, entries in overrides.items():
        path = case_dir / rel_file
        if not path.is_file():
            raise FileNotFoundError(f"foam_dict_overrides: {path} does not exist")
        foam_file = FoamFile(path)
        for entry_path, value in entries.items():
            key = tuple(entry_path.strip("/").split("/"))
            old = foam_file.get(key, None)
            foam_file[key] = value
            logger.info(
                "foam_dict_overrides: %s %s: %r -> %r", rel_file, entry_path, old, value
            )

def run_openfoam_case(case_dir: Path, verbose: bool = True, clean: bool = True):
    """Run the case's Allrun script.

    With ``clean=False`` the case is not cleaned first -- used by the
    ``sdf_hex`` pipeline, which cleans, then writes ``constant/polyMesh``
    (which cleaning would delete), then runs.
    """
    if not case_dir.exists():
        raise FileNotFoundError(f"Foam case path does not exist: {case_dir}")

    foam_case = FoamCase(case_dir)
    if not clean:
        pass
    elif verbose:
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

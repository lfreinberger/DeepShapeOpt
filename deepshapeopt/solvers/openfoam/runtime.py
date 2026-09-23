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

# Fields each primal solver solves. simpleControl::criteriaSatisfied only checks fields that
# are BOTH solved and matched by a listed regex, so a missing entry silently drops that
# equation from the criterion (an unconverged energy equation would report "converged").
PRIMAL_FIELDS = {"simple": {"p", "U"}, "simpleHeatTransfer": {"p", "U", "T"}}
WRITE_NEVER = 10 ** 9


def dict_layout(opt: FoamFile) -> tuple[str, str, list[str]]:
    """``(primal solver, adjoint manager, adjoint solver names)`` of an optimisationDict."""
    primal = list(opt["primalSolvers"].keys())[0]
    managers = list(opt["adjointManagers"].keys())
    if len(managers) != 1:
        raise ValueError(f"expected one adjoint manager, found {managers}")
    solvers = list(opt["adjointManagers"][managers[0]]["adjointSolvers"].keys())
    return str(primal), str(managers[0]), [str(k) for k in solvers]


def _solution_controls_path(opt: FoamFile, name: str) -> tuple:
    primal, manager, solvers = dict_layout(opt)
    if name in ("p1", primal):
        return ("primalSolvers", primal, "solutionControls")
    if name not in solvers:
        raise ValueError(f"unknown solver {name!r}; valid: {primal}, {solvers}")
    return ("adjointManagers", manager, "adjointSolvers", name, "solutionControls")


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
        base = _solution_controls_path(opt, name)
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
        if base[0] == "primalSolvers":
            solver = str(opt["primalSolvers", base[1], "solver"])
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
    active_solvers: set[str],
    solver_convergence: dict | None = None,
) -> dict[str, str]:
    """Activate the adjoint solvers that carry a metric and predict their write times.

    Reads primal/adjoint ``nIters`` from ``system/optimisationDict``. The solvers run back
    to back, so an active solver writes at ``p1.nIters + sum(nIters of the active solvers
    before it)``. Predictions only hold while every solver runs its full ``nIters``; use
    :func:`~deepshapeopt.solvers.openfoam.sensitivities.resolve_adjoint_time` on readback.

    Also mutates the runtime copy: every adjoint solver's ``active`` flag, ``purgeWrite 0``
    in controlDict, and the ``__ADJOINT_TIMES__`` marker of the Allrun (when present) becomes
    the open time range ``"1:"`` so reconstructPar covers every write.
    ``solver_convergence`` is applied first so the predicted times use the values that run.
    """
    opt_path = case_dir / "system" / "optimisationDict"
    ctrl_path = case_dir / "system" / "controlDict"
    allrun_path = case_dir / "Allrun"

    opt = FoamFile(opt_path)
    if solver_convergence:
        _apply_solver_convergence(opt, solver_convergence, case_dir)
    primal, manager, names = dict_layout(opt)
    unknown = set(active_solvers) - set(names)
    if unknown:
        raise ValueError(f"metrics refer to adjoint solvers {sorted(unknown)} missing from {opt_path} ({names})")
    if not active_solvers:
        raise ValueError("configure_foam_runtime: at least one adjoint solver must be active")

    t = int(opt["primalSolvers", primal, "solutionControls", "nIters"])
    adjoint_times: dict[str, str] = {}
    for name in names:
        active = name in active_solvers
        opt["adjointManagers", manager, "adjointSolvers", name, "active"] = bool(active)
        if active:
            t += int(opt["adjointManagers", manager, "adjointSolvers", name, "solutionControls", "nIters"])
            adjoint_times[name] = str(t)

    FoamFile(ctrl_path)["purgeWrite"] = 0

    marker = "__ADJOINT_TIMES__"
    if allrun_path.is_file():
        text = allrun_path.read_text()
        if marker in text:
            allrun_path.write_text(text.replace(marker, "1:"))
    return adjoint_times


def select_allrun(case_dir: Path) -> None:
    """Use the ``Allrun.sdf_hex`` variant of a template as its Allrun when it exists."""
    variant = case_dir / "Allrun.sdf_hex"
    if variant.exists():
        allrun = case_dir / "Allrun"
        shutil.copy2(variant, allrun)
        allrun.chmod(0o755)


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

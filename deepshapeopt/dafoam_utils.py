"""DAFoam (discrete adjoint) as forward solver for the sdf_hex pipeline.

DAFoam lives in an Apptainer image (``dafoam/opt-packages``, OpenFOAM-v2506 + CoDiPack AD
builds). The coupling is file based, like the OpenFOAM path:

1. the driver copies the experiment's ``dafoam_case/`` template (``0.orig``, ``constant``,
   ``system``) to a runtime directory (:func:`prepare_dafoam_case`) -- the in-container
   driver :mod:`deepshapeopt.dafoam_runscript` is copied next to it;
2. per evaluation the hex pipeline writes ``constant/polyMesh`` (metres) into it and
   :meth:`DAFoamRunner.evaluate` writes ``dafoam_options.json`` and runs
   ``apptainer exec ... mpirun -np N python dafoam_runscript.py``;
3. :meth:`DAFoamRunner.load_sensitivities` reads the output value and the total derivative
   ``dJ/dX`` of every mesh point (``dafoam_output/dFdXv_<name>.npy``, global point order)
   and returns the rows of the snapped wall points -- the exact analogue of the
   ``pointSensVec*`` readback, ready for ``foam_utils.compute_shape_gradient(...,
   integrated=True)``.

Only the wall points move between designs in the sdf_hex pipeline (interior points stay
on the castellation grid), so ``dJ/dX`` restricted to the wall points IS the total
derivative w.r.t. the pipeline's design motion; no mesh-warping derivative is involved.

Config block (``optimization.dafoam``)::

    "forward_solver": "dafoam",
    "dafoam": {
      "container": "/usr2/lfrei/containers/dafoam.sif",
      "n_procs": 20,
      "binds": ["/work", "/usr2", "/workdisk"],
      "template": "dafoam_case",                    # relative to the experiment dir
      "daOptions": {...},                           # pyDAFoam options (tolerances, ...)
      "functions": {                                # DAFoam "function" dicts
        "uniformity": {"type": "variance", ..., "reference": {"field": "UData",
                        "patch": "outlet", "patch_mean_of": "U"}},
        "TPin": {...}, "TPout": {...}
      },
      "outputs": {"uniformity": {"uniformity": 1.0}, "losses": {"TPin": 1.0, "TPout": -1.0}}
    }

``outputs`` are linear combinations of DAFoam functions; each gets one adjoint solve. A
function's optional ``reference`` block asks for reference data (DAFoam ``variance``
reads ``0/<field>``) equal to the area-weighted patch mean of a field. ``"at"`` selects the
design that mean belongs to:

``"current"`` (default)
    the design being evaluated, i.e. the pure variance at every design, at no extra cost.
    With area weights A_f and n components DAFoam computes
    V(r) = scale/(n*A) * sum_c sum_f A_f (U_cf - r_c)^2, so V(mean) = V(0) - scale/n * |mean|^2.
    The run therefore uses a zero reference and ``patchMean`` functions for the mean; the
    run script subtracts the quadratic term and feeds its linearization (-2*scale/n*mean_c)
    into the same single adjoint solve. dV/dr vanishes at r = mean, so this is also the
    exact gradient of the pure variance. Needs ``mode: surface`` and ``useGeoWeight: 1``.
``"first"`` (legacy)
    the FIRST evaluated design; computed once with an extra primal and then kept fixed.
    The function is then variance + scale/n * |mean - mean_0|^2, NOT the pure variance:
    the offset was 0.2 % of J on the cylinder but ~15 % at the end of the die runs
    (measured 2026-09-21: J/J0 0.0300 vs 0.0432 after 40 iterations on the die).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field, replace as dc_replace
from pathlib import Path

import numpy as np

import deepshapeopt.foam_utils as foam_utils

logger = logging.getLogger(__name__)

RUNSCRIPT = Path(__file__).with_name("dafoam_runscript.py")
DEFAULT_LOAD_SCRIPT = "/home/dafoamuser/dafoam/loadDAFoam.sh"
OPTIONS_FILE = "dafoam_options.json"
OUTPUT_DIR = "dafoam_output"
# Mount points inside the container for a published patched build (see
# scripts/publish_dafoam_build.sh). CONTAINER_LIBS is the stock shared-library directory,
# already on LD_LIBRARY_PATH through loadDAFoam.sh; CONTAINER_PATCH only has to be a path
# that exists in the image and is not otherwise used.
CONTAINER_LIBS = "/home/dafoamuser/dafoam/OpenFOAM/sharedLibs"
CONTAINER_PATCH = "/opt/dafoam-patched"


@dataclass
class DAFoamConfig:
    container: Path
    n_procs: int = 1
    binds: list[str] = field(default_factory=lambda: ["/work", "/usr2", "/workdisk"])
    template: str = "dafoam_case"
    da_options: dict = field(default_factory=dict)
    functions: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    load_script: str = DEFAULT_LOAD_SCRIPT
    apptainer: str = "apptainer"
    fail_on_mesh_check: bool = False
    keep_coloring: bool = False
    # "coloring" (stock DAFoam) or "fvmatrix" (preconditioner straight from the fvMatrix
    # coefficients -- no coloring, ~10x faster per evaluation, but it needs the patched
    # build, see external/dafoam_patches/)
    pc_mode: str = "coloring"
    # Root of the published patched build: a directory holding dafoam/ (the python
    # package) and sharedLibs/ (with the patched libDASolver.so), produced by
    # scripts/publish_dafoam_build.sh. Defaults to $DAFOAM_BUILD_ROOT so configs stay
    # portable -- a host path baked into a config breaks every other machine, and in
    # particular every slurm job, because node-local scratch is not visible there.
    build_root: Path | None = None

    @classmethod
    def from_dict(cls, cfg: dict) -> "DAFoamConfig":
        known = {
            "container", "n_procs", "binds", "template", "daOptions", "functions",
            "outputs", "load_script", "apptainer", "fail_on_mesh_check", "keep_coloring",
            "pc_mode", "build_source", "build_overlay",
        }
        unknown = set(cfg) - known
        if unknown:
            raise ValueError(f"Unknown optimization.dafoam keys {sorted(unknown)}; valid: {sorted(known)}")
        if "container" not in cfg:
            raise ValueError("optimization.dafoam.container (path to the .sif image) is required")
        container = Path(cfg["container"]).expanduser()
        if not container.is_file():
            raise FileNotFoundError(f"DAFoam container image not found: {container}")
        for fn, spec in cfg.get("functions", {}).items():
            if "type" not in spec:
                raise ValueError(f"dafoam.functions.{fn}: missing DAFoam 'type'")
        for name, terms in cfg.get("outputs", {}).items():
            missing = set(terms) - set(cfg.get("functions", {}))
            if missing:
                raise ValueError(f"dafoam.outputs.{name} references unknown functions {sorted(missing)}")
        pc_mode = str(cfg.get("pc_mode", "coloring"))
        if pc_mode not in ("coloring", "fvmatrix"):
            raise ValueError(f"dafoam.pc_mode {pc_mode!r} invalid; use 'coloring' or 'fvmatrix'")
        raw_root = cfg.get("build_root") or os.environ.get("DAFOAM_BUILD_ROOT")
        build_root = Path(raw_root).expanduser() if raw_root else None
        if pc_mode == "fvmatrix":
            build_root = _check_build_root(build_root, from_config="build_root" in cfg)
        return cls(
            container=container,
            n_procs=int(cfg.get("n_procs", 1)),
            binds=list(cfg.get("binds", ["/work", "/usr2", "/workdisk"])),
            template=str(cfg.get("template", "dafoam_case")),
            da_options=dict(cfg.get("daOptions", {})),
            functions=dict(cfg.get("functions", {})),
            outputs=dict(cfg.get("outputs", {})),
            load_script=str(cfg.get("load_script", DEFAULT_LOAD_SCRIPT)),
            apptainer=str(cfg.get("apptainer", "apptainer")),
            fail_on_mesh_check=bool(cfg.get("fail_on_mesh_check", False)),
            keep_coloring=bool(cfg.get("keep_coloring", False)),
            pc_mode=pc_mode,
            build_root=build_root,
        )


def _check_build_root(root: Path | None, from_config: bool) -> Path:
    """Validate the published patched build, with a message that says what to do.

    Checked eagerly at config load, long before the reconstruction and the meshing, so a
    missing build costs a second rather than several minutes.
    """
    hint = (
        "pc_mode 'fvmatrix' needs the patched DAFoam build (the stock container has no "
        "initializePCMatFvMatrix). Either\n"
        "  - publish it:  DeepShapeOpt/scripts/publish_dafoam_build.sh   (see "
        "DeepShapeOpt/external/dafoam_patches/README.md), or\n"
        "  - set pc_mode to 'coloring' to run on the stock container (~10x slower per "
        "evaluation)."
    )
    if root is None:
        raise ValueError(
            "dafoam.build_root is not set and $DAFOAM_BUILD_ROOT is empty.\n" + hint
        )
    where = "dafoam.build_root" if from_config else "$DAFOAM_BUILD_ROOT"
    if not root.is_dir():
        raise FileNotFoundError(
            f"{where} is not a directory: {root}\n"
            "On a slurm job this usually means the build sits on node-local scratch "
            "(e.g. /workdisk), which the compute node cannot see -- publish it to shared "
            "storage instead.\n" + hint
        )
    for sub in ("dafoam", "sharedLibs"):
        if not (root / sub).is_dir():
            raise FileNotFoundError(
                f"{where} = {root} has no {sub}/ -- it does not look like a published "
                f"DAFoam build.\n" + hint
            )
    lib = root / "sharedLibs" / "libDASolver.so"
    if not lib.is_file():
        raise FileNotFoundError(f"{where} = {root}: missing sharedLibs/libDASolver.so\n" + hint)
    return root


def prepare_dafoam_case(template_dir: Path, run_name: str, runtime_root: Path | None = None) -> Path:
    """Copy the ``dafoam_case`` template to ``foam_run_<run_name>`` and add the run script."""
    template_dir = Path(template_dir)
    for required in ("0.orig", "constant", "system"):
        if not (template_dir / required).is_dir():
            raise FileNotFoundError(f"DAFoam case template {template_dir} lacks {required}/")
    case_dir = foam_utils.prepare_foam_runtime(template_dir, run_name=run_name, runtime_root=runtime_root)
    shutil.copy2(RUNSCRIPT, case_dir / RUNSCRIPT.name)
    return case_dir


def container_command(dcfg: DAFoamConfig, n_procs: int | None = None) -> list[str]:
    """The ``apptainer exec`` command line that runs the run script in the case directory.

    ``--cleanenv`` keeps the host's OpenFOAM environment (v2506 login shell) out of the
    container; the DAFoam environment comes from ``loadDAFoam.sh`` inside.
    """
    n = int(dcfg.n_procs if n_procs is None else n_procs)
    python_cmd = f"python {RUNSCRIPT.name} {OPTIONS_FILE}"
    if n > 1:
        python_cmd = f"mpirun -np {n} -x PYTHONPATH {python_cmd}"
    prefix = ""
    binds = list(dcfg.binds)
    if dcfg.build_root:
        # Patched DAFoam, mounted over the stock install -- no overlay, because
        # fuse-overlayfs on NFS is fragile and the build has to live on shared storage
        # for cluster nodes to see it. sharedLibs/ carries the patched libDASolver.so
        # (it is already on LD_LIBRARY_PATH via loadDAFoam.sh); PYTHONPATH makes
        # `import dafoam` resolve to the patched package instead of site-packages.
        binds.append(f"{dcfg.build_root}/sharedLibs:{CONTAINER_LIBS}")
        binds.append(f"{dcfg.build_root}:{CONTAINER_PATCH}")
        prefix = f"export PYTHONPATH={CONTAINER_PATCH}:$PYTHONPATH; "
    # Apptainer aborts container creation when a bind SOURCE does not exist, so drop the
    # ones that are absent instead of failing. The default list is written for the
    # workstation; a slurm job on a compute node has no /workdisk, for instance. Binding a
    # path that is not there cannot help anyway, and the case directory itself lives under
    # a path the driver already resolved.
    kept = []
    for b in binds:
        src = b.split(":", 1)[0]
        if Path(src).exists():
            kept.append(b)
        else:
            logger.info("Skipping bind %s: not present on this host", src)

    inner = f"unset DISPLAY; source {dcfg.load_script} >/dev/null 2>&1 && {prefix}{python_cmd}"
    cmd = [dcfg.apptainer, "exec", "--cleanenv"]
    if kept:
        cmd += ["--bind", ",".join(kept)]
    cmd += [str(dcfg.container), "bash", "-c", inner]
    return cmd


def _strip_private_keys(functions: dict) -> dict:
    """DAFoam function dicts without our own ``reference`` block."""
    return {name: {k: v for k, v in spec.items() if k != "reference"} for name, spec in functions.items()}


def write_dafoam_options(
    case_dir: Path,
    dcfg: DAFoamConfig,
    functions: dict,
    outputs: dict,
    sensitivities: list[str],
    reference_fields: dict | None = None,
    n_points: int | None = None,
    perturb: dict | None = None,
    output_corrections: dict | None = None,
) -> Path:
    opts = {
        "daOptions": dcfg.da_options,
        "functions": _strip_private_keys(functions),
        "outputs": outputs,
        "output_corrections": output_corrections or {},
        "sensitivities": list(sensitivities),
        "reference_fields": reference_fields or {},
        "n_points": int(n_points) if n_points else 0,
        "output_dir": OUTPUT_DIR,
        "fail_on_mesh_check": dcfg.fail_on_mesh_check,
        "keep_coloring": dcfg.keep_coloring,
        "pc_mode": dcfg.pc_mode,
    }
    if perturb:
        opts["perturb"] = perturb
    path = Path(case_dir) / OPTIONS_FILE
    path.write_text(json.dumps(opts, indent=2) + "\n")
    return path


def run_dafoam_case(
    case_dir: Path, dcfg: DAFoamConfig, log_name: str = "log.dafoam", verbose: bool = False,
    n_procs: int | None = None,
) -> dict:
    """Run the container; return the parsed ``dafoam_output/functions.json``.

    Raises ``RuntimeError`` (with the log tail) when the run, the primal or an adjoint
    solve fails -- there is no silent fallback.
    """
    case_dir = Path(case_dir)
    cmd = container_command(dcfg, n_procs)
    log_path = case_dir / log_name
    logger.info("Running DAFoam (%d procs): %s", n_procs or dcfg.n_procs, case_dir)
    start = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, cwd=case_dir, stdout=log, stderr=subprocess.STDOUT, check=False)
    elapsed = time.time() - start
    result_path = case_dir / OUTPUT_DIR / "functions.json"
    result = json.loads(result_path.read_text()) if result_path.is_file() else None
    if proc.returncode != 0 or result is None:
        tail = "".join(log_path.read_text().splitlines(keepends=True)[-40:])
        raise RuntimeError(
            f"DAFoam run failed (exit {proc.returncode}) in {case_dir}; log tail:\n{tail}"
        )
    if result.get("primal_fail"):
        raise RuntimeError(f"DAFoam primal did not converge in {case_dir} (see {log_path})")
    for name, adj in result.get("adjoint", {}).items():
        if adj.get("fail"):
            raise RuntimeError(f"DAFoam adjoint for {name!r} failed in {case_dir} (see {log_path})")
    result["time_total_s"] = elapsed
    if verbose:
        logger.info("DAFoam finished in %.1f s: outputs %s", elapsed, result.get("outputs"))
    else:
        logger.debug("DAFoam finished in %.1f s: %s", elapsed, result.get("outputs"))
    return result


def load_dafoam_sensitivities(
    case_dir: Path, name: str, surface_point_ids: np.ndarray, n_points: int
) -> tuple[np.ndarray, float]:
    """``(dJ/dX on the wall points [N, 3] in 1/m, J)`` of output ``name``."""
    case_dir = Path(case_dir)
    result = json.loads((case_dir / OUTPUT_DIR / "functions.json").read_text())
    if name not in result["outputs"]:
        raise KeyError(f"DAFoam output {name!r} not in {list(result['outputs'])}")
    sens_all = np.load(case_dir / OUTPUT_DIR / f"dFdXv_{name}.npy")
    if sens_all.shape != (n_points, 3):
        raise ValueError(
            f"dFdXv_{name} has shape {sens_all.shape}, the written mesh has {n_points} points"
        )
    sens = sens_all[np.asarray(surface_point_ids)]
    logger.debug("Loaded DAFoam sensitivities %r: %d wall points, |dJ/dX| max %.3e",
                 name, len(sens), float(np.abs(sens).max()))
    return sens, float(result["outputs"][name])


class DAFoamRunner:
    """Per-run state: config, runtime case, resolved reference fields."""

    def __init__(self, cfg: dict, case_dir: Path, verbose: bool = False):
        self.dcfg = DAFoamConfig.from_dict(cfg)
        self.case_dir = Path(case_dir)
        self.verbose = verbose
        self.reference_fields: dict | None = None
        self.last_result: dict | None = None

    # -- reference data (variance-type functions) ----------------------------------
    def _reference_specs(self, at: str) -> dict:
        specs = {}
        for fn, spec in self.dcfg.functions.items():
            ref = spec.get("reference")
            if ref is None:
                continue
            mode = ref.get("at", "current")
            if mode not in ("first", "current"):
                raise ValueError(f"dafoam.functions.{fn}.reference.at must be 'first' or 'current', got {mode!r}")
            if mode == at:
                specs[fn] = ref
        return specs

    def _current_mean_plan(self) -> tuple[dict, dict, dict]:
        """Functions, zero reference fields and output corrections of the ``at: current``
        references (pure variance at every design, see the module docstring)."""
        functions, fields, corrections = {}, {}, {}
        for fn, ref in self._reference_specs("current").items():
            spec = self.dcfg.functions[fn]
            if spec.get("type") != "variance" or spec.get("mode") != "surface" or not spec.get("useGeoWeight"):
                raise ValueError(
                    f"dafoam.functions.{fn}: reference.at 'current' needs type 'variance', "
                    "mode 'surface' and useGeoWeight 1 (the identity uses DAFoam's area weights)"
                )
            fld, patch, var = ref["field"], ref["patch"], ref.get("patch_mean_of", "U")
            vector = spec.get("varType", "vector") == "vector"
            indices = list(spec["indices"]) if vector else [0]
            names = []
            for i in indices:
                name = f"ref_{fld}_{i}"
                functions[name] = {
                    "type": "patchMean", "source": "patchToFace", "patches": [patch],
                    "varName": var, "varType": spec.get("varType", "vector"), "index": i, "scale": 1.0,
                }
                names.append(name)
            fields[fld] = {"patch": patch, "value": [0.0] * (3 if vector else 1)}
            for out, terms in self.dcfg.outputs.items():
                if fn in terms:
                    corrections.setdefault(out, []).append({
                        "functions": names,
                        "factor": float(terms[fn]) * float(spec.get("scale", 1.0)) / len(indices),
                    })
        return functions, fields, corrections

    def _resolve_reference_fields(self, n_points: int) -> dict:
        specs = self._reference_specs("first")
        if not specs:
            return {}
        logger.warning(
            "DAFoam: reference of %s is frozen at the first evaluated design (reference.at "
            "'first'): the function is the variance plus a mean-drift term, not the pure "
            "variance. Set reference.at to 'current' for the pure variance.", sorted(specs),
        )
        functions, wanted = {}, {}
        for fn, ref in specs.items():
            fld, patch, var = ref["field"], ref["patch"], ref.get("patch_mean_of", "U")
            key = (fld, patch, var)
            if key in wanted:
                continue
            names = []
            for i in range(3):
                fname = f"ref_{fld}_{i}"
                functions[fname] = {
                    "type": "patchMean", "source": "patchToFace", "patches": [patch],
                    "varName": var, "varType": "vector", "index": i, "scale": 1.0,
                }
                names.append(fname)
            wanted[key] = names
        # This extra primal runs with its OWN function set (the ref_* patch means), so a
        # daOptions.primalFuncStdTol naming the evaluation's functions would abort with
        # "functionName ... not found". Disable it here and let the reference primal use
        # the plain residual criterion -- it runs once and sets UData for the whole run.
        ref_dcfg = self.dcfg
        if dict(self.dcfg.da_options).get("primalFuncStdTol", {}).get("stdTol", -1.0) > 0:
            ref_opts = {k: v for k, v in self.dcfg.da_options.items() if k != "primalFuncStdTol"}
            ref_dcfg = dc_replace(self.dcfg, da_options=ref_opts)
        write_dafoam_options(
            self.case_dir, ref_dcfg, functions, outputs={}, sensitivities=[], n_points=n_points,
        )
        logger.info("DAFoam: extra primal for the reference patch means of %s", sorted(wanted))
        result = run_dafoam_case(self.case_dir, self.dcfg, log_name="log.dafoam_reference",
                                 verbose=self.verbose)
        fields = {}
        for (fld, patch, var), names in wanted.items():
            value = [result["functions"][n] for n in names]
            fields[fld] = {"patch": patch, "value": value}
            logger.info("DAFoam reference %s: area mean of %s on %s = %s", fld, var, patch, value)
        return fields

    # -- evaluation -----------------------------------------------------------------
    def evaluate(self, n_points: int, sensitivities: list[str], perturb: dict | None = None) -> dict:
        """Primal + one adjoint per requested output on the mesh currently in the case."""
        for name in sensitivities:
            if name not in self.dcfg.outputs:
                raise KeyError(f"DAFoam output {name!r} not defined in dafoam.outputs")
        if self.reference_fields is None:
            self.reference_fields = self._resolve_reference_fields(n_points)
        mean_functions, zero_fields, corrections = self._current_mean_plan()
        write_dafoam_options(
            self.case_dir, self.dcfg, {**self.dcfg.functions, **mean_functions}, self.dcfg.outputs,
            sensitivities, reference_fields={**self.reference_fields, **zero_fields},
            n_points=n_points, perturb=perturb, output_corrections=corrections,
        )
        self.last_result = run_dafoam_case(self.case_dir, self.dcfg, verbose=self.verbose)
        return self.last_result

    def load_sensitivities(self, name: str, surface_point_ids: np.ndarray, n_points: int):
        return load_dafoam_sensitivities(self.case_dir, name, surface_point_ids, n_points)

    def clean(self) -> None:
        """Remove solver output of the last evaluation (the run script cleans again on start)."""
        for entry in self.case_dir.iterdir():
            if entry.is_dir() and (entry.name.startswith("processor") or entry.name == OUTPUT_DIR):
                shutil.rmtree(entry)

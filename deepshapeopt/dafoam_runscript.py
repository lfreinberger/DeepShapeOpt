#!/usr/bin/env python
"""DAFoam driver script -- runs INSIDE the DAFoam container (pyDAFoam, numpy, mpi4py,
petsc4py only; no deepshapeopt imports).

One call = one design evaluation with DAFoam's discrete adjoint:

    mpirun -np N python dafoam_runscript.py dafoam_options.json

The options file is written by ``deepshapeopt.dafoam_utils`` next to the case:

    {
      "daOptions":     {...},            # DAFoam options (solverName, tolerances, ...)
      "functions":     {"name": {...}},  # DAFoam "function" dicts (type, patches, scale, ...)
      "outputs":       {"J": {"name": coef, ...}},  # linear combinations of functions
      "sensitivities": ["J", ...],       # outputs whose dJ/dXv is computed (reverse AD)
      "reference_fields": {"UData": {"patch": "outlet", "value": [ux, uy, uz]}},  # optional
      "perturb":       {"point": k, "coord": j, "delta": h},   # optional (serial FD checks)
      "n_points":      N,                # global point count of constant/polyMesh (optional)
      "output_dir":    "dafoam_output"
    }

Outputs (in ``output_dir``, written by rank 0):

    functions.json          function values, output values, primal/adjoint status, timings
    dFdXv_<output>.npy      (N, 3) total derivative dJ/dX of the output w.r.t. ALL mesh
                            points, in the GLOBAL (serial constant/polyMesh) point order and
                            in the mesh's length unit (metres). Parallel runs are assembled
                            through processor*/constant/polyMesh/pointProcAddressing; points
                            shared by several processors sum their local contributions.

Total derivative (states W, mesh points X, residuals R(W, X) = 0):

    dJ/dX = dJ/dX|_W - psi^T dR/dX,   (dR/dW)^T psi = (dJ/dW)^T

evaluated matrix-free with DAFoam's reverse-mode AD (``calcJacTVecProduct``) exactly as
``dafoam.mphys.mphys_dafoam.DAFoamSolver`` does inside OpenMDAO. The adjoint of a linear
combination of functions is solved once with the combined right-hand side.
"""

import gzip
import json
import os
import re
import shutil
import sys
import time

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dafoam import PYDAFOAM

STATE_NAME = "aero_states"
RESIDUAL_NAME = "aero_residuals"
VOL_COORD_NAME = "aero_vol_coords"


def info(msg):
    if MPI.COMM_WORLD.rank == 0:
        print(f"[dafoam_runscript] {msg}", flush=True)


def read_label_list(path):
    """Read an OpenFOAM ascii labelList (e.g. pointProcAddressing), plain or gzipped."""
    if not os.path.exists(path) and os.path.exists(path + ".gz"):
        path = path + ".gz"
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        text = f.read()
    body = text[text.index("}") + 1:]  # drop the FoamFile header
    m = re.search(r"(\d+)\s*\(", body)
    if m is None:
        raise ValueError(f"cannot parse labelList {path}")
    n = int(m.group(1))
    start = body.index("(", m.end() - 1) + 1
    end = body.index(")", start)
    values = np.array(body[start:end].split(), dtype=np.int64)
    if len(values) != n:
        raise ValueError(f"{path}: expected {n} labels, found {len(values)}")
    return values


def read_point_count(path):
    """Number of points in constant/polyMesh/points (plain or gzipped)."""
    if not os.path.exists(path) and os.path.exists(path + ".gz"):
        path = path + ".gz"
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        text = f.read(4096)
    body = text[text.index("}") + 1:]
    m = re.search(r"(\d+)\s*\(", body)
    if m is None:
        raise ValueError(f"cannot parse point count from {path}")
    return int(m.group(1))


def write_reference_field(name, patch, value, time_dir="0"):
    """Write a uniform volVectorField/volScalarField ``name`` whose ``patch`` boundary
    value is ``value`` (the reference data of DAFoam's ``variance`` function)."""
    value = [float(v) for v in np.atleast_1d(np.asarray(value, dtype=float))]
    if len(value) == 3:
        cls, val = "volVectorField", "(%.17g %.17g %.17g)" % tuple(value)
        dims = "[0 1 -1 0 0 0 0]"
        zero = "(0 0 0)"
    elif len(value) == 1:
        cls, val = "volScalarField", "%.17g" % value[0]
        dims = "[0 0 0 0 0 0 0]"
        zero = "0"
    else:
        raise ValueError(f"reference value must have 1 or 3 components, got {len(value)}")
    text = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    location    "{time_dir}";
    object      {name};
}}

dimensions      {dims};

internalField   uniform {zero};

boundaryField
{{
    {patch}
    {{
        type            fixedValue;
        value           uniform {val};
    }}
    ".*"
    {{
        type            zeroGradient;
    }}
}}
"""
    os.makedirs(time_dir, exist_ok=True)
    with open(os.path.join(time_dir, name), "w") as f:
        f.write(text)


def prepare_case(opts, comm):
    """Fresh 0/ from 0.orig, drop old processor*/time dirs, write reference fields."""
    if comm.rank == 0:
        for entry in os.listdir("."):
            is_time_dir = re.fullmatch(r"[0-9.e+-]+", entry) is not None
            if os.path.isdir(entry) and (entry.startswith("processor") or is_time_dir):
                shutil.rmtree(entry)
            # The dRdW coloring depends only on the sparsity pattern (connectivity), not on
            # the point coordinates. It is dropped by default and kept with keep_coloring
            # when the caller guarantees unchanged connectivity (noise probe with
            # reuse_castellation), which saves a full coloring pass per evaluation.
            if entry.startswith("dRdWColoring_") and not opts.get("keep_coloring", False):
                os.remove(entry)
        shutil.copytree("0.orig", "0")
        for name, spec in (opts.get("reference_fields") or {}).items():
            write_reference_field(name, spec["patch"], spec["value"])
        os.makedirs(opts.get("output_dir", "dafoam_output"), exist_ok=True)
    comm.Barrier()


def local_to_global_index(comm):
    """Global (serial polyMesh) point index of every local point."""
    if comm.size == 1:
        return None
    return read_label_list(f"processor{comm.rank}/constant/polyMesh/pointProcAddressing")


def assemble_global(local, l2g, n_global, comm):
    """Gather per-rank (nLocal*3,) products into a (n_global, 3) array on rank 0."""
    local = np.asarray(local, dtype=float).reshape(-1, 3)
    if comm.size == 1:
        return local
    parts = comm.gather((l2g, local), root=0)
    if comm.rank != 0:
        return None
    out = np.zeros((n_global, 3))
    for idx, vals in parts:
        np.add.at(out, idx, vals)
    return out


def main():
    comm = MPI.COMM_WORLD
    opts_path = sys.argv[1] if len(sys.argv) > 1 else "dafoam_options.json"
    with open(opts_path) as f:
        opts = json.load(f)
    out_dir = opts.get("output_dir", "dafoam_output")
    t0 = time.time()

    prepare_case(opts, comm)

    da_options = dict(opts["daOptions"])
    da_options["function"] = opts["functions"]
    da_options.setdefault("inputInfo", {})
    da_options["inputInfo"].setdefault(
        VOL_COORD_NAME, {"type": "volCoord", "components": ["solver", "function"]}
    )
    da_options.setdefault("useAD", {"mode": "reverse"})

    DASolver = PYDAFOAM(options=da_options, comm=comm)
    if da_options.get("adjEqnSolMethod", "Krylov") == "Krylov":
        # matrix-free dRdW^T operator of the reverse-AD solver (mphys does this in setup)
        DASolver.solverAD.initializedRdWTMatrixFree()
    n_local_points = DASolver.solver.getNLocalPoints()
    n_global = int(opts.get("n_points") or 0)
    if n_global == 0:
        n_global = read_point_count("constant/polyMesh/points")

    xv = np.zeros(n_local_points * 3)
    DASolver.solver.getOFMeshPoints(xv)

    perturb = opts.get("perturb")
    if perturb:
        if comm.size != 1:
            raise RuntimeError("perturb (FD check) is only supported in serial runs")
        k, j, h = int(perturb["point"]), int(perturb["coord"]), float(perturb["delta"])
        xv[3 * k + j] += h
        DASolver.setVolCoords(xv)
        info(f"perturbed point {k} coord {j} by {h:+.3e}")

    mesh_ok = int(DASolver.solver.checkMesh())
    info(f"checkMesh: {'OK' if mesh_ok == 1 else 'FAILED (thresholds in checkMeshThreshold)'}")
    if mesh_ok != 1 and opts.get("fail_on_mesh_check", False):
        raise RuntimeError("DAFoam checkMesh failed")

    # ---- primal --------------------------------------------------------------------
    t_primal = time.time()
    DASolver()
    primal_fail = int(DASolver.primalFail)
    DASolver.solver.calcPrimalResidualStatistics("print")
    states = DASolver.getStates()
    DASolver.setStates(states)
    DASolver.solverAD.calcPrimalResidualStatistics("calc")
    t_primal = time.time() - t_primal

    funcs = {}
    DASolver.evalFunctions(funcs)
    funcs = {k: float(v) for k, v in funcs.items()}
    outputs = {}
    for name, terms in (opts.get("outputs") or {}).items():
        outputs[name] = float(sum(coef * funcs[fn] for fn, coef in terms.items()))
    info(f"primal done in {t_primal:.1f} s, fail={primal_fail}; functions {funcs}; outputs {outputs}")

    result = {
        "primal_fail": primal_fail,
        "mesh_ok": mesh_ok,
        "functions": funcs,
        "outputs": outputs,
        "n_points": n_global,
        "n_procs": comm.size,
        "adjoint": {},
        "time_primal_s": t_primal,
    }
    if comm.rank == 0:
        with open(os.path.join(out_dir, "functions.json"), "w") as f:
            json.dump(result, f, indent=2)
    if primal_fail:
        if opts.get("fail_on_primal", True):
            info("primal FAILED (residual above primalMinResTol*primalMinResTolDiff) -- no sensitivities")
            sys.exit(1)
        info("primal flagged as not converged; continuing (fail_on_primal=false)")

    sens_names = list(opts.get("sensitivities") or [])
    if not sens_names:
        info(f"done in {time.time() - t0:.1f} s (no sensitivities requested)")
        return

    # ---- adjoint setup (as in mphys_dafoam.DAFoamSolver.solve_linear) -------------
    # "Krylov" (DAFoam default): GMRES preconditioned by an explicit dRdW^T matrix, which is
    # assembled from colored residual evaluations -- accurate but the coloring and the PC
    # assembly dominate the cost and are redone whenever the mesh connectivity changes.
    # "fixedPoint": DASimpleFoam's own SIMPLE-like adjoint iteration, matrix free -- no
    # coloring, no PC matrix, so roughly the cost of a primal. It forbids normalizeStates.
    t_setup = time.time()
    adj_method = DASolver.getOption("adjEqnSolMethod")
    local_adj_size = DASolver.getNLocalAdjointStates()
    ksp = None
    if adj_method == "Krylov":
        if DASolver.getOption("adjUseColoring"):
            DASolver.solver.runColoring()
        dRdWTPC = PETSc.Mat().create(comm)
        DASolver.solver.calcdRdWT(1, dRdWTPC)
        ksp = PETSc.KSP().create(comm)
        DASolver.solverAD.createMLRKSPMatrixFree(dRdWTPC, ksp)
    elif adj_method != "fixedPoint":
        raise ValueError(f"adjEqnSolMethod {adj_method!r} not supported; use Krylov or fixedPoint")
    psi = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
    psi.setSizes((local_adj_size, PETSc.DECIDE), bsize=1)
    psi.setFromOptions()
    l2g = local_to_global_index(comm)
    info(f"adjoint setup ({adj_method}) done in {time.time() - t_setup:.1f} s")

    for name in sens_names:
        t_adj = time.time()
        terms = opts["outputs"][name]
        dFdW = np.zeros(local_adj_size)
        dFdXv = np.zeros(len(xv))
        for fn, coef in terms.items():
            seed = np.array([float(coef)])
            product = np.zeros(local_adj_size)
            DASolver.solverAD.calcJacTVecProduct(
                STATE_NAME, "stateVar", states, fn, "function", seed, product
            )
            dFdW += product
            product = np.zeros(len(xv))
            DASolver.solverAD.calcJacTVecProduct(
                VOL_COORD_NAME, "volCoord", xv, fn, "function", seed, product
            )
            dFdXv += product

        psi.set(0)
        rhs = DASolver.array2Vec(dFdW)
        if adj_method == "Krylov":
            fail = int(DASolver.solverAD.solveLinearEqn(ksp, rhs, psi))
        else:
            fail = int(DASolver.solverAD.runFPAdj(rhs, psi))
        psi_arr = DASolver.vec2Array(psi)

        dRdXvT_psi = np.zeros(len(xv))
        DASolver.solverAD.calcJacTVecProduct(
            VOL_COORD_NAME, "volCoord", xv, RESIDUAL_NAME, "residual", psi_arr, dRdXvT_psi
        )
        total_local = dFdXv - dRdXvT_psi
        total = assemble_global(total_local, l2g, n_global, comm)
        t_adj = time.time() - t_adj
        result["adjoint"][name] = {"fail": fail, "time_s": t_adj}
        if comm.rank == 0:
            np.save(os.path.join(out_dir, f"dFdXv_{name}.npy"), total)
            with open(os.path.join(out_dir, "functions.json"), "w") as f:
                json.dump(result, f, indent=2)
        info(
            f"adjoint '{name}': fail={fail}, {t_adj:.1f} s, |dFdXv| max "
            f"{float(np.abs(total).max()) if total is not None else float('nan'):.3e}"
        )
        if fail:
            sys.exit(2)

    info(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()

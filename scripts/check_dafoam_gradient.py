"""Finite-difference gate of the DAFoam discrete-adjoint shape derivative dJ/dX.

Builds the start geometry of a ``solver.type: dafoam`` config exactly as the driver does
(reconstruction, sdf_hex mesh, DAFoam case), then

  1. runs primal + adjoint once: J and dJ/dX of one output on all mesh points,
  2. for each test direction g (on the wall points the driver uses, max |g| = 1):
     runs two more primals with the mesh points moved by +-eps*g and compares the
     central difference (J+ - J-)/(2 eps) with sum(g . dJ/dX).

Directions: ``grad`` (dJ/dX itself on the wall points, the most sensitive magnitude check)
and ``random`` (seeded Gaussian on the wall points, checks that the direction is right
and not just the norm).

The primal must run a FIXED number of iterations, otherwise J(X) is piecewise in X and
the difference quotient meaningless: the check sets primalMinResTol so small that it is
never met (and the fail threshold, primalMinResTol*primalMinResTolDiff, to 1) and drops
primalFuncStdTol, so every primal runs endTime iterations (``--niters`` overrides endTime
in system/controlDict of the runtime case). Use a coarse mesh (small max_level): the
check is about the adjoint, not the discretization.

    uv run python scripts/check_dafoam_gradient.py --config <config.json> --output uniformity
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import replace as dc_replace
from pathlib import Path

import numpy as np

from deepshapeopt.config import load_config
from deepshapeopt.problem import build_problem

LOGGER = logging.getLogger("check_dafoam_gradient")
DISP_FILE = "fd_displacement.npy"


def _set_end_time(case_dir: Path, n_iters: int) -> None:
    path = case_dir / "system" / "controlDict"
    text = re.sub(r"(\n\s*endTime\s+)[^;]+;", rf"\g<1>{n_iters};", path.read_text())
    path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path, help="Experiment JSON with solver.type dafoam")
    parser.add_argument("--output", default="uniformity", help="DAFoam output (key of solver.dafoam.outputs)")
    parser.add_argument("--eps", type=float, default=1e-6, help="Largest point displacement [m]")
    parser.add_argument("--niters", type=int, default=None, help="Fixed primal iterations (endTime)")
    parser.add_argument("--directions", default="grad,random", help="Comma list of grad, random")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tol", type=float, default=1e-2, help="Pass if |ratio - 1| <= tol for all directions")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    if cfg.solver.type != "dafoam":
        raise SystemExit("check_dafoam_gradient needs solver.type 'dafoam'")
    problem = build_problem(cfg, args.config.parent)
    solver = problem.solver
    runner = solver.runner
    case = solver.case_dir

    # fixed iteration count, see the module docstring
    da = dict(runner.dcfg.da_options)
    da.pop("primalFuncStdTol", None)
    da["primalMinResTol"] = 1e-300
    da["primalMinResTolDiff"] = 1e300
    runner.dcfg = dc_replace(runner.dcfg, da_options=da)
    if args.niters:
        _set_end_time(case, args.niters)

    mesh = problem.mesher.build()
    problem.mesher.write_polymesh(case)
    n_points = int(mesh.n_points)
    wall = np.asarray(mesh.wall_point_ids, dtype=np.int64)

    base = runner.evaluate(n_points, [args.output])
    J0 = float(base["outputs"][args.output])
    sens = np.load(case / "dafoam_output" / f"dFdXv_{args.output}.npy")
    LOGGER.info("J0 = %.10e, |dJ/dX| on %d wall points: max %.3e", J0, wall.size, np.abs(sens[wall]).max())

    rng = np.random.default_rng(args.seed)
    report = {"config": str(args.config), "output": args.output, "eps": args.eps, "J0": J0,
              "n_points": n_points, "n_wall_points": int(wall.size), "directions": {}}
    ok = True
    for direction in [d.strip() for d in args.directions.split(",") if d.strip()]:
        g = np.zeros((n_points, 3))
        if direction == "grad":
            g[wall] = sens[wall]
        elif direction == "random":
            g[wall] = rng.standard_normal((wall.size, 3))
        else:
            raise SystemExit(f"unknown direction {direction!r}")
        g /= np.abs(g).max()
        np.save(case / DISP_FILE, g)
        predicted = float(np.sum(g * sens))
        J = {}
        for sign in (+1, -1):
            res = runner.evaluate(n_points, [], perturb={"displacement": DISP_FILE, "scale": sign * args.eps})
            J[sign] = float(res["outputs"][args.output])
        measured = (J[+1] - J[-1]) / (2.0 * args.eps)
        ratio = measured / predicted if predicted != 0.0 else float("nan")
        passed = abs(ratio - 1.0) <= args.tol
        ok &= passed
        report["directions"][direction] = {"J+": J[+1], "J-": J[-1], "measured": measured,
                                           "predicted": predicted, "ratio": ratio, "pass": passed}
        LOGGER.info("%-6s FD %.6e  adjoint %.6e  ratio %.6f  %s", direction, measured, predicted, ratio,
                    "PASS" if passed else "FAIL")

    out = problem.paths.optimization / f"check_dafoam_gradient_{args.output}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    LOGGER.info("report: %s", out)
    solver.close()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

"""Solver-termination handling: log parsing and the solver_convergence block.

No OpenFOAM needed -- these cover exactly the two things that are easy to get
wrong and expensive to discover on the cluster:

  * the two termination messages carry different quantities (global time vs the
    solver's own iteration count), so a naive parse mixes them up;
  * regex keys in `residualControl` must keep their literal quotes, otherwise
    OpenFOAM reads them as a plain keyword that never matches any field.
"""
import textwrap

import pytest
from foamlib import FoamFile

from deepshapeopt.foam_utils import (
    _apply_solver_convergence,
    format_solver_iterations,
    parse_solver_iterations,
)

LOG = """\
Time = 1
Solving for Ux, Initial residual = 1
p1 solution converged in 412 iterations

Time = 413
Solving for Uaas1x, Initial residual = 1
as1 solution reached max. number of iterations 1500

Time = 1913
Solving for Uaas2x, Initial residual = 1
as2 solution converged in 2400 iterations
"""

DICT = textwrap.dedent("""\
    FoamFile { version 2.0; format ascii; class dictionary; object optimisationDict; }

    primalSolvers
    {
        p1
        {
            solver simple;
            solutionControls
            {
                nIters 150;
                residualControl { "p.*" 1.e-7; "U.*" 1.e-7; }
            }
        }
    }
    adjointManagers
    {
        am1
        {
            adjointSolvers
            {
                as1
                {
                    solutionControls
                    {
                        nIters 300;
                        residualControl { "pa.*" 1.e-7; "Ua.*" 1.e-7; }
                    }
                }
                as2
                {
                    solutionControls
                    {
                        nIters 300;
                        residualControl { "pa.*" 1.e-7; "Ua.*" 1.e-7; }
                    }
                }
            }
        }
    }
    """)


def _dict_file(tmp_path):
    case = tmp_path / "case"
    (case / "system").mkdir(parents=True)
    (case / "system" / "optimisationDict").write_text(DICT)
    (case / "system" / "controlDict").write_text(
        "FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }\n"
        "writeControl timeStep;\nwriteInterval 100;\npurgeWrite 5;\n")
    return case


def test_parse_mixes_global_and_local_counts(tmp_path):
    """`converged in T` is a global time, `reached max ... N` a local count."""
    log = tmp_path / "log.adjointOptimisationFoam"
    log.write_text(LOG)
    info = parse_solver_iterations(log)

    assert info["p1"] == {"iters": 412, "end_time": 412, "reason": "converged"}
    # 1500 is the solver's OWN count -> it ends at 412 + 1500
    assert info["as1"] == {"iters": 1500, "end_time": 1912, "reason": "max_iters"}
    # 2400 is a global time -> 2400 - 1912 = 488 of its own iterations
    assert info["as2"] == {"iters": 488, "end_time": 2400, "reason": "converged"}
    assert "as1 1500/1500 max_iters" in format_solver_iterations(
        info, {"as1": 1500})


def test_parse_tolerates_inactive_solver_and_missing_log(tmp_path):
    log = tmp_path / "log.adjointOptimisationFoam"
    log.write_text(LOG.split("Time = 1913")[0])       # as2 never ran
    info = parse_solver_iterations(log)
    assert set(info) == {"p1", "as1"}
    assert parse_solver_iterations(tmp_path / "nope.log") == {}


def test_fixed_mode_removes_residual_control(tmp_path):
    """`fixed` must DELETE the block, not set an unreachable value."""
    case = _dict_file(tmp_path)
    opt = FoamFile(case / "system" / "optimisationDict")
    _apply_solver_convergence(
        opt, {"mode": "fixed", "solvers": {"p1": {"n_iters": 2000},
                                           "as1": {"n_iters": 1500}}}, case)

    text = (case / "system" / "optimisationDict").read_text()
    assert "residualControl" not in text.split("as2")[0]
    assert int(opt["primalSolvers", "p1", "solutionControls", "nIters"]) == 2000
    assert int(opt["adjointManagers", "am1", "adjointSolvers", "as1",
                   "solutionControls", "nIters"]) == 1500


def test_residual_mode_writes_quoted_regex_keys(tmp_path):
    case = _dict_file(tmp_path)
    opt = FoamFile(case / "system" / "optimisationDict")
    _apply_solver_convergence(
        opt, {"mode": "residual", "write_only_end_states": True,
              "solvers": {"p1": {"n_iters": 2000,
                                 "residuals": {"p": 1e-6, "U": 1e-6}}}}, case)

    written = (case / "system" / "optimisationDict").read_text()
    assert '"p.*"' in written and '"U.*"' in written
    rc = opt["primalSolvers", "p1", "solutionControls", "residualControl"]
    assert pytest.approx(float(rc['"p.*"'])) == 1e-6
    # write_only_end_states leaves just the forced end-of-solver writes
    assert int(FoamFile(case / "system" / "controlDict")["writeInterval"]) >= 10 ** 6


def test_residual_mode_rejects_unconverged_energy_equation(tmp_path):
    """simpleHeatTransfer also solves T; without a T threshold the criterion
    would ignore the energy equation and report a false convergence."""
    case = _dict_file(tmp_path)
    p = case / "system" / "optimisationDict"
    p.write_text(p.read_text().replace("solver simple;", "solver simpleHeatTransfer;"))
    opt = FoamFile(p)
    with pytest.raises(ValueError, match="also solves"):
        _apply_solver_convergence(
            opt, {"mode": "residual",
                  "solvers": {"p1": {"residuals": {"p": 1e-6, "U": 1e-6}}}}, case)


def test_config_errors(tmp_path):
    case = _dict_file(tmp_path)
    opt = FoamFile(case / "system" / "optimisationDict")
    with pytest.raises(ValueError, match="unknown solver"):
        _apply_solver_convergence(opt, {"mode": "fixed",
                                        "solvers": {"as3": {"n_iters": 10}}}, case)
    with pytest.raises(ValueError, match="needs 'residuals'"):
        _apply_solver_convergence(opt, {"mode": "residual",
                                        "solvers": {"as1": {"n_iters": 10}}}, case)
    with pytest.raises(ValueError, match="mode must be"):
        _apply_solver_convergence(opt, {"mode": "sometimes"}, case)


def test_as_template_is_a_no_op(tmp_path):
    case = _dict_file(tmp_path)
    before = (case / "system" / "optimisationDict").read_text()
    _apply_solver_convergence(FoamFile(case / "system" / "optimisationDict"),
                              {"mode": "as_template"}, case)
    assert (case / "system" / "optimisationDict").read_text() == before

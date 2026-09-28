"""Validation of solver.dafoam: solverName against pc_mode and the patched build."""

import pytest

from deepshapeopt.solvers.dafoam.runner import DAFoamConfig


@pytest.fixture
def sif(tmp_path):
    path = tmp_path / "dafoam.sif"
    path.write_text("")
    return path


@pytest.fixture
def build_root(tmp_path):
    root = tmp_path / "build"
    (root / "dafoam").mkdir(parents=True)
    (root / "sharedLibs").mkdir()
    (root / "sharedLibs" / "libDASolver.so").write_text("")
    return root


def _cfg(sif, solver, pc_mode="coloring", **extra):
    return {"container": str(sif), "pc_mode": pc_mode, "daOptions": {"solverName": solver}, **extra}


def test_fvmatrix_rejected_for_heat_transfer_solver(sif, build_root):
    with pytest.raises(ValueError, match="fvmatrix"):
        DAFoamConfig.from_dict(_cfg(sif, "DASimpleHeatTransferFoam", "fvmatrix", build_root=str(build_root)))


def test_heat_transfer_solver_needs_patched_build(sif, monkeypatch):
    monkeypatch.delenv("DAFOAM_BUILD_ROOT", raising=False)
    with pytest.raises(ValueError, match="only in the patched DAFoam build"):
        DAFoamConfig.from_dict(_cfg(sif, "DASimpleHeatTransferFoam"))


def test_heat_transfer_solver_with_build_root(sif, build_root, monkeypatch):
    monkeypatch.delenv("DAFOAM_BUILD_ROOT", raising=False)
    dcfg = DAFoamConfig.from_dict(_cfg(sif, "DASimpleHeatTransferFoam", build_root=str(build_root)))
    assert dcfg.build_root == build_root
    assert dcfg.pc_mode == "coloring"


def test_stock_solver_with_coloring_needs_no_build(sif, monkeypatch):
    monkeypatch.delenv("DAFOAM_BUILD_ROOT", raising=False)
    assert DAFoamConfig.from_dict(_cfg(sif, "DASimpleFoam")).build_root is None

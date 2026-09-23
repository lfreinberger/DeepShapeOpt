"""Config schema v2 and the v1 -> v2 migration."""

import json
import sys
from pathlib import Path

import pytest

from deepshapeopt.config import Config, ConfigError, load_config

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "v1"
from deepshapeopt.config import migrate as migrate_config  # noqa: E402


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("DEEPSHAPEOPT_DATA_DIR", "/data")
    monkeypatch.setenv("DEEPSHAPEOPT_MODEL_DIR", "/models")
    monkeypatch.setenv("DEEPSHAPEOPT_RESULTS_DIR", "/results")
    monkeypatch.setenv("DEEPSHAPEOPT_SCRATCH_DIR", "/scratch")
    monkeypatch.setenv("DAFOAM_SIF", "/containers/dafoam.sif")


def _v1(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def _migrated(name: str, **kw) -> Config:
    v2, _ = migrate_config.migrate(_v1(name), **kw)
    return Config.from_dict(migrate_config._env_free(v2))


def test_v1_config_is_rejected_with_hint():
    with pytest.raises(ConfigError, match="migrate-config"):
        Config.from_dict(_v1("drag_deepsdf"))


def test_unknown_key_names_replacement():
    v2, _ = migrate_config.migrate(_v1("drag_deepsdf"))
    v2["run"]["heavy_data_output_path"] = "/x"
    with pytest.raises(ConfigError, match="run.heavy_data_dir"):
        Config.from_dict(migrate_config._env_free(v2))


def test_write_scale_in_mesh_block_is_rejected():
    v2, _ = migrate_config.migrate(_v1("drag_deepsdf"))
    v2["mesh"]["sdf_hex"]["write_scale"] = 0.001
    with pytest.raises(ConfigError, match="unit_to_metre"):
        Config.from_dict(migrate_config._env_free(v2))


@pytest.mark.parametrize("name", ["drag_deepsdf", "drag_deepsdf_dafoam", "drag_ffd", "channel_terms"])
def test_every_v1_key_is_known(name):
    assert migrate_config.unknown_v1_keys(_v1(name)) == []


def test_drag_migration():
    cfg = _migrated("drag_deepsdf")
    assert cfg.is_optimization
    assert cfg.objective.metric == "drag"
    assert [c["type"] for c in cfg.constraints] == ["volume", "centroid"]
    assert cfg.parametrization.type == "deepsdf"
    assert cfg.parametrization.deepsdf.pca.enabled is True
    assert cfg.parametrization.deepsdf.reconstruction.export_resolution == 48
    assert cfg.geometry.unit_to_metre == 1.0
    assert cfg.solver.type == "openfoam" and cfg.solver.case == "foam_case"
    assert cfg.optimizer.convergence.obj_tol == 0.03


def test_dafoam_migration_moves_template_and_container():
    cfg = _migrated("drag_deepsdf_dafoam")
    assert cfg.solver.type == "dafoam"
    assert cfg.solver.case == "dafoam_case"
    assert "template" not in cfg.solver.dafoam
    assert "DAFOAM_SIF" in cfg.solver.dafoam["container"]


def test_ffd_migration():
    cfg = _migrated("drag_ffd")
    assert cfg.parametrization.type == "ffd"
    assert cfg.parametrization.ffd.n_control_points == [5, 5, 5]
    assert [c["type"] for c in cfg.constraints] == ["volume", "centroid", "ffd_jacobian"]


def test_internal_flow_migration():
    cfg = _migrated("channel_terms")
    assert cfg.geometry.flow == "internal"
    assert cfg.geometry.unit_to_metre == 0.001
    assert "write_scale" not in cfg.mesh.sdf_hex and "flow" not in cfg.mesh.sdf_hex
    assert "outlet_interior" in cfg.mesh.sdf_hex
    assert cfg.objective.metric == "uniformity_directional"
    kinds = [c["type"] for c in cfg.constraints]
    assert kinds == ["metric", "undercut", "min_steg_length"]
    steg = cfg.constraints[2]
    assert steg["budget"]["mode"] == "absolute_shortfall" and "thickness_threshold" in steg
    assert [p["type"] for p in cfg.objective.penalties] == ["proximity", "lattice_smoothness"]
    assert cfg.optimizer.gcmma["enabled"] and cfg.optimizer.feasibility_restoration["enabled"]
    assert cfg.optimizer.step_control["enabled"]
    assert cfg.parametrization.lock.faces == ["x_min"]


def test_lock_layout_option():
    cfg = _migrated("channel_terms", lock_layout="outlet_face_and_inlet_rim")
    # an explicit lock block wins over the default layout
    assert cfg.parametrization.lock.layout is None
    v1 = _v1("channel_terms")
    del v1["optimization"]["lock_domain"]
    v2, _ = migrate_config.migrate(v1, lock_layout="outlet_face_and_inlet_rim")
    cfg = Config.from_dict(migrate_config._env_free(v2))
    assert cfg.parametrization.lock.layout == "outlet_face_and_inlet_rim"


def test_dropped_feature_aborts():
    v1 = _v1("channel_terms")
    v1["optimization"]["taper"]["enabled"] = True
    with pytest.raises(migrate_config.MigrationError, match="taper"):
        migrate_config.migrate(v1)


def test_reconstruction_only_config():
    cfg = _migrated("reconstruction")
    assert not cfg.is_optimization
    assert cfg.parametrization.deepsdf.reconstruction.export_resolution == 12
    with pytest.raises(ConfigError):
        cfg.require_optimization()


def test_load_config_expands_env(tmp_path):
    v2, _ = migrate_config.migrate(_v1("drag_deepsdf"))
    path = tmp_path / "config.json"
    path.write_text(json.dumps(v2))
    cfg = load_config(path)
    assert cfg.geometry.mesh_path == "/data/shapes/cube_l1.stl"
    assert cfg.run.heavy_data_dir == "/results"


def test_unresolved_env_var_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSHAPEOPT_DATA_DIR")
    v2, _ = migrate_config.migrate(_v1("drag_deepsdf"))
    path = tmp_path / "config.json"
    path.write_text(json.dumps(v2))
    with pytest.raises(ConfigError, match="DEEPSHAPEOPT_DATA_DIR"):
        load_config(path)

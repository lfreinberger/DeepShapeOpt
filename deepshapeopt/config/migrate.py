#!/usr/bin/env python
"""Convert a v1 experiment config (``reconstruction`` / ``optimization`` blocks) to schema v2.

    uv run deepshapeopt migrate-config experiments/.../config.json [--write] \
        [--lock-layout outlet_face_and_inlet_rim] [--keep-paths]

Without ``--write`` the converted config is printed. With ``--write`` the file is replaced in
place and free-form ``_note`` strings are collected into ``<experiment>/notes/config_notes.md``.
Host-specific absolute paths are replaced by the environment variables the runtime expands
(``DEEPSHAPEOPT_*``, ``DAFOAM_SIF``) unless ``--keep-paths`` is given. Features that were
dropped from the library (taper, minimum wall thickness, section partition, snappy) abort the
conversion when they are enabled in the source config.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .schema import Config, ConfigError

METRIC_NAMES = {
    "uniformityas1": "uniformity",
    "uniformityDirectionalas1": "uniformity_directional",
    "lossesas2": "losses",
}
DROPPED_BLOCKS = ("taper", "feature_size", "min_wall_thickness", "section_partition")
HOST_PATHS = (
    ("/usr2/lfrei/ProjectsPhD/DeepSDFStruct/DeepSDFStruct/trained_models", "${DEEPSHAPEOPT_MODEL_DIR}"),
    ("/usr2/lfrei/containers/dafoam.sif", "${DAFOAM_SIF}"),
)


class MigrationError(Exception):
    pass


# Every v1 leaf key the tool understands (mapped or deliberately dropped). The migration
# test asserts that no kept config carries a key outside this set, so an option cannot be
# lost silently.
KNOWN_V1_KEYS = {
    "results_name", "debug",
    "reconstruction.mesh_path", "reconstruction.model_path", "reconstruction.model_checkpoint",
    "reconstruction.tiling", "reconstruction.spline_degree", "reconstruction.tiling_map",
    "reconstruction.design_domain", "reconstruction.n_uniform_samples",
    "reconstruction.n_surface_samples", "reconstruction.samples_surface_stds", "reconstruction.lr",
    "reconstruction.num_iterations", "reconstruction.batch_size", "reconstruction.loss_fn",
    "reconstruction.clamp_val", "reconstruction.code_bound", "reconstruction.code_reg_lambda",
    "reconstruction.grad_clip", "reconstruction.eikonal_lambda", "reconstruction.device",
    "reconstruction.mesh_device", "reconstruction.reuse_parameter", "reconstruction.create_mesh_N",
    "reconstruction.export_rec_samples_series", "reconstruction.export_rec_samples_fine_until_epoch",
    "reconstruction.export_rec_samples_fine_every", "reconstruction.error_cutoff",
    "reconstruction.heavy_data_output_path", "reconstruction.n_control_points",
    "optimization.debug", "optimization.num_iter", "optimization.start_iter",
    "optimization.mesh_resolution", "optimization.max_step", "optimization.bounds",
    "optimization.sensitivity_region_x", "optimization.objective_name", "optimization.mesh_pipeline",
    "optimization.parametrization", "optimization.forward_solver", "optimization.heavy_data_output_path",
    "optimization.foam_runtime_root", "optimization.use_center_constraint", "optimization.center_tol",
    "optimization.use_jacobian_constraint", "optimization.jacobian_threshold",
    "optimization.jacobian_samples", "optimization.convergence_obj_tol", "optimization.convergence_window",
    "optimization.convergence_min_iter", "optimization.convergence_feasibility_tol",
    "optimization.convergence_patience",
    "optimization.sensitivity.loading_method", "optimization.sensitivity.scaling_to_mm",
    "optimization.sensitivity.warn_tol", "optimization.sensitivity.field_suffix",
    "optimization.sensitivity.field_name", "optimization.sensitivity.objective_path",
    "optimization.sensitivity.invert_normals", "optimization.sensitivity.dafoam_output",
    "optimization.sensitivity.patch_name",
    "optimization.constraint.enabled", "optimization.constraint.name", "optimization.constraint.target_mode",
    "optimization.constraint.target_factor", "optimization.constraint.target_value",
    "optimization.regularization.enabled", "optimization.regularization.type",
    "optimization.regularization.weight",
    "optimization.lattice_smoothness.enabled", "optimization.lattice_smoothness.weight",
    "optimization.lattice_smoothness.reference",
    "optimization.pca.enabled", "optimization.pca.n_components", "optimization.pca.bounds",
    "optimization.pca.max_step", "optimization.pca.cache_path",
    "optimization.ffd.n_control_points", "optimization.ffd.spline_degree",
    "optimization.ffd.jacobian_constraint.enabled", "optimization.ffd.jacobian_constraint.threshold",
    "optimization.ffd.jacobian_constraint.samples", "optimization.ffd.jacobian_constraint.ks_rho",
    "optimization.lock_domain.faces", "optimization.lock_domain.boxes_physical",
    "optimization.lock_domain.boxes_norm", "optimization.lock_domain.safety", "optimization.lock_domain.preset",
    "optimization.no_undercut.enabled", "optimization.no_undercut.mode", "optimization.no_undercut.method",
    "optimization.no_undercut.draw_direction", "optimization.no_undercut.draft_angle_deg",
    "optimization.no_undercut.exclude_axial_deg", "optimization.no_undercut.grid_spacing",
    "optimization.no_undercut.band_factor", "optimization.no_undercut.weight",
    "optimization.no_undercut.target_mode", "optimization.no_undercut.target_value",
    "optimization.no_undercut.target_factor", "optimization.no_undercut.exclude_region",
    "optimization.no_undercut.formulation", "optimization.no_undercut.ks_rho", "optimization.no_undercut.scope",
    "optimization.no_undercut.silhouette_margin", "optimization.no_undercut.outlet_patch",
    "optimization.min_steg_length.enabled", "optimization.min_steg_length.mode",
    "optimization.min_steg_length.formulation", "optimization.min_steg_length.ks_rho",
    "optimization.min_steg_length.length_mode", "optimization.min_steg_length.flow_direction",
    "optimization.min_steg_length.thickness_threshold_mm", "optimization.min_steg_length.min_length_mm",
    "optimization.min_steg_length.grid_spacing", "optimization.min_steg_length.n_dirs",
    "optimization.min_steg_length.ray_step_mm", "optimization.min_steg_length.tau_mm",
    "optimization.min_steg_length.slab_margin", "optimization.min_steg_length.weight",
    "optimization.min_steg_length.target_mode", "optimization.min_steg_length.target_value",
    "optimization.min_steg_length.target_factor", "optimization.min_steg_length.target_shortfall_mm",
    "optimization.min_steg_length.exclude_region",
    "optimization.gcmma.enabled", "optimization.gcmma.max_inner", "optimization.gcmma.feas_tol",
    "optimization.feasibility_restoration.enabled", "optimization.feasibility_restoration.tol",
    "optimization.feasibility_restoration.max_steps", "optimization.feasibility_restoration.step_limit",
    "optimization.step_control.enabled", "optimization.step_control.rho_shrink",
    "optimization.step_control.rho_grow", "optimization.step_control.shrink", "optimization.step_control.grow",
    "optimization.step_control.min_step_factor", "optimization.step_control.pred_tol_rel",
    "optimization.step_control.patience",
    "optimization.noise_probe.enabled", "optimization.noise_probe.start_parameters",
    "optimization.noise_probe.h", "optimization.noise_probe.n_points", "optimization.noise_probe.seed",
    "optimization.noise_probe.reuse_castellation",
    "optimization.jacobian_probe.enabled", "optimization.jacobian_probe.start_parameters",
    "optimization.jacobian_probe.batch", "optimization.jacobian_probe.keep_vectors",
    "optimization.solver_convergence.mode", "optimization.solver_convergence.solvers",
    "optimization.solver_convergence.write_only_end_states",
    "optimization.foam_dict_overrides", "optimization.outlet_interior", "optimization.sdf_hex",
    "optimization.dafoam", "optimization.taper", "optimization.min_wall_thickness",
    "optimization.section_partition", "optimization.feature_size", "optimization.diagnostics",
}
# Blocks whose content is passed through as a whole (their inner keys are not enumerated).
PASSTHROUGH_PREFIXES = (
    "optimization.sdf_hex.", "optimization.dafoam.", "optimization.outlet_interior.",
    "optimization.foam_dict_overrides.", "optimization.solver_convergence.solvers.",
    "optimization.taper.", "optimization.min_wall_thickness.", "optimization.section_partition.",
    "optimization.diagnostics.",
)


def leaf_keys(node, prefix: str = ""):
    """Dotted paths of every leaf in a v1 config (notes excluded, pass-through blocks kept whole)."""
    out = []
    if isinstance(node, dict):
        for k, v in node.items():
            if k.startswith("_"):
                continue
            path = f"{prefix}{k}"
            passthrough = any((path + ".") == p or (path + ".").startswith(p) for p in PASSTHROUGH_PREFIXES)
            if isinstance(v, dict) and v and not passthrough:
                out.extend(leaf_keys(v, path + "."))
            else:
                out.append(path)
    return out


def unknown_v1_keys(v1: dict) -> list[str]:
    return sorted(k for k in leaf_keys(v1) if k not in KNOWN_V1_KEYS)


def _env_path(value, kind: str, keep: bool):
    """Replace host paths by environment variables."""
    if value is None or keep or not isinstance(value, str):
        return value
    for prefix, var in HOST_PATHS:
        if value.startswith(prefix):
            return var + value[len(prefix):]
    if kind == "heavy" and value.startswith("/storage/lfrei/"):
        return "${DEEPSHAPEOPT_RESULTS_DIR}"
    if kind == "scratch" and value.startswith("/work/lfrei/"):
        return "${DEEPSHAPEOPT_SCRATCH_DIR}"
    if kind == "mesh" and value.startswith("data/"):
        return "${DEEPSHAPEOPT_DATA_DIR}/" + value[len("data/"):]
    return value


def _collect_notes(node, prefix, out):
    if isinstance(node, dict):
        for k, v in node.items():
            if k.startswith("_") and isinstance(v, str):
                out.append((f"{prefix}{k}", v))
            else:
                _collect_notes(v, f"{prefix}{k}.", out)


def _strip_notes(node):
    if isinstance(node, dict):
        return {k: _strip_notes(v) for k, v in node.items() if not k.startswith("_")}
    if isinstance(node, list):
        return [_strip_notes(v) for v in node]
    return node


def _budget(cfg: dict, formulation: str, kind: str) -> dict:
    """Constraint budget of a v1 geometry block (target_mode & co.)."""
    if formulation == "ks_margin":
        budget = {"mode": "ks_bound"}
        if kind == "min_steg_length" and cfg.get("target_shortfall_mm") is not None:
            budget["shortfall"] = float(cfg["target_shortfall_mm"])
        return budget
    mode = cfg.get("target_mode", "relative_to_initial")
    if mode == "relative_to_initial":
        return {"mode": "relative_to_initial", "factor": float(cfg.get("target_factor", 1.0))}
    if mode == "absolute":
        if cfg.get("target_value") is None:
            raise MigrationError(f"{kind}: target_mode 'absolute' without target_value")
        return {"mode": "absolute", "value": float(cfg["target_value"])}
    if mode == "absolute_shortfall_mm":
        return {"mode": "absolute_shortfall", "shortfall": float(cfg["target_shortfall_mm"])}
    raise MigrationError(f"{kind}: unknown target_mode {mode!r}")


def _undercut_params(uc: dict) -> dict:
    out = {"type": "undercut"}
    for src, dst in (
        ("method", "method"), ("formulation", "formulation"), ("draw_direction", "draw_direction"),
        ("draft_angle_deg", "draft_angle_deg"), ("exclude_axial_deg", "exclude_axial_deg"),
        ("grid_spacing", "grid_spacing"), ("band_factor", "band_factor"),
        ("exclude_region", "exclude_region"), ("ks_rho", "ks_rho"), ("scope", "scope"),
        ("silhouette_margin", "silhouette_margin"), ("outlet_patch", "outlet_patch"),
    ):
        if uc.get(src) is not None:
            out[dst] = uc[src]
    return out


def _steg_params(sl: dict) -> dict:
    out = {"type": "min_steg_length"}
    for src, dst in (
        ("formulation", "formulation"), ("length_mode", "length_mode"),
        ("flow_direction", "flow_direction"), ("thickness_threshold_mm", "thickness_threshold"),
        ("min_length_mm", "min_length"), ("grid_spacing", "grid_spacing"), ("n_dirs", "n_dirs"),
        ("ray_step_mm", "ray_step"), ("tau_mm", "tau"), ("slab_margin", "slab_margin"),
        ("ks_rho", "ks_rho"), ("exclude_region", "exclude_region"),
    ):
        if sl.get(src) is not None:
            out[dst] = sl[src]
    return out


def migrate(v1: dict, *, lock_layout: str | None = None, keep_paths: bool = False) -> tuple[dict, list[str]]:
    """Return ``(v2 config dict, list of remarks)``."""
    if "run" in v1 and "geometry" in v1:
        raise MigrationError("already a v2 config")
    rec = dict(v1.get("reconstruction") or {})
    opt = dict(v1.get("optimization") or {})
    remarks: list[str] = []
    is_opt = bool(opt)

    for block in DROPPED_BLOCKS:
        blk = opt.get(block)
        if blk and blk.get("enabled", False):
            raise MigrationError(f"optimization.{block} is enabled but the feature was removed")
        if blk is not None:
            remarks.append(f"dropped disabled block optimization.{block}")
    if opt.get("mesh_pipeline", "sdf_hex") != "sdf_hex" and is_opt:
        raise MigrationError("mesh_pipeline 'snappy' was removed; only sdf_hex configs can be migrated")

    debug = bool(v1.get("debug", opt.get("debug", False)))
    run = {
        "name": v1.get("results_name", "results"),
        "device": rec.get("device", "cuda"),
        "debug": debug,
    }
    if is_opt:
        run["num_iter"] = int(opt.get("num_iter", 1))
    heavy = opt.get("heavy_data_output_path", rec.get("heavy_data_output_path"))
    if heavy:
        run["heavy_data_dir"] = _env_path(heavy, "heavy", keep_paths)
    if opt.get("foam_runtime_root"):
        run["scratch_dir"] = _env_path(opt["foam_runtime_root"], "scratch", keep_paths)

    sens = dict(opt.get("sensitivity") or {})
    sdf_hex = dict(opt.get("sdf_hex") or {})
    unit = float(sens.get("scaling_to_mm", sdf_hex.get("write_scale", 1.0)))
    if "write_scale" in sdf_hex and abs(float(sdf_hex["write_scale"]) - unit) > 1e-15:
        raise MigrationError("sensitivity.scaling_to_mm and sdf_hex.write_scale disagree")
    sdf_hex.pop("write_scale", None)
    flow = sdf_hex.pop("flow", "external")
    geometry = {"mesh_path": _env_path(rec["mesh_path"], "mesh", keep_paths), "unit_to_metre": unit}
    if rec.get("design_domain") is not None:
        geometry["design_domain"] = rec["design_domain"]
    if is_opt:
        geometry["flow"] = flow

    # --- parametrization ---------------------------------------------------------
    if opt.get("parametrization") == "ffd" or ("n_control_points" in rec and "model_path" not in rec):
        kind = "ffd"
    else:
        kind = "deepsdf"
    param: dict = {"type": kind}
    if kind == "deepsdf":
        recon = {}
        for src, dst in (
            ("n_uniform_samples", "n_uniform_samples"), ("n_surface_samples", "n_surface_samples"),
            ("samples_surface_stds", "samples_surface_stds"), ("lr", "lr"),
            ("num_iterations", "num_iterations"), ("batch_size", "batch_size"), ("loss_fn", "loss_fn"),
            ("clamp_val", "clamp_val"), ("code_bound", "code_bound"), ("code_reg_lambda", "code_reg_lambda"),
            ("grad_clip", "grad_clip"), ("eikonal_lambda", "eikonal_lambda"), ("reuse_parameter", "reuse"),
            ("export_rec_samples_series", "export_samples_series"),
            ("export_rec_samples_fine_until_epoch", "export_samples_fine_until_epoch"),
            ("export_rec_samples_fine_every", "export_samples_fine_every"),
            ("error_cutoff", "error_cutoff"), ("mesh_device", "mesh_device"),
        ):
            if src in rec:
                recon[dst] = rec[src]
        if "mesh_resolution" in opt:
            recon["export_resolution"] = int(opt["mesh_resolution"])
        elif "create_mesh_N" in rec:
            recon["export_resolution"] = int(rec["create_mesh_N"])
        param["deepsdf"] = {
            "model_path": _env_path(rec["model_path"], "model", keep_paths),
            "checkpoint": rec.get("model_checkpoint", "latest"),
            "tiling": rec["tiling"],
            "spline_degree": rec.get("spline_degree", [1, 1, 1]),
            "tiling_map": rec.get("tiling_map", "hat"),
            "reconstruction": recon,
        }
        pca = _strip_notes(opt.get("pca") or {})
        if pca:
            param["deepsdf"]["pca"] = pca
    else:
        ffd = dict(opt.get("ffd") or {})
        param["ffd"] = {
            "n_control_points": ffd.get("n_control_points", rec.get("n_control_points")),
            "spline_degree": ffd.get("spline_degree", rec.get("spline_degree", [2, 2, 2])),
        }
    lock = {}
    lock_v1 = dict(opt.get("lock_domain") or {})
    for key in ("faces", "boxes_physical", "boxes_norm", "safety"):
        if key in lock_v1:
            lock[key] = lock_v1[key]
    if lock_v1.get("preset") or (lock_layout and not lock_v1):
        lock["layout"] = lock_layout or "outlet_face_and_inlet_rim"
    if lock:
        param["lock"] = lock

    v2 = {"run": run, "geometry": geometry, "parametrization": param}
    if not is_opt:
        return v2, remarks

    # --- mesh --------------------------------------------------------------------
    oi = opt.get("outlet_interior")
    if oi is not None and "outlet_interior" not in sdf_hex:
        oi = dict(oi)
        oi.pop("debug", None)
        if oi.get("enabled", True):
            oi.pop("enabled", None)
            sdf_hex["outlet_interior"] = oi
    v2["mesh"] = {"sdf_hex": _strip_notes(sdf_hex)}

    # --- solver ------------------------------------------------------------------
    solver_type = opt.get("forward_solver", "openfoam")
    solver: dict = {"type": solver_type}
    openfoam = {}
    if opt.get("solver_convergence"):
        openfoam["solver_convergence"] = _strip_notes(opt["solver_convergence"])
    if opt.get("foam_dict_overrides"):
        openfoam["dict_overrides"] = opt["foam_dict_overrides"]
    sensitivity = {}
    if sens.get("field_suffix"):
        sensitivity["field_suffix"] = sens["field_suffix"]
    if sens.get("warn_tol") is not None:
        sensitivity["warn_tol"] = sens["warn_tol"]
    if sensitivity:
        openfoam["sensitivity"] = sensitivity
    if openfoam:
        solver["openfoam"] = openfoam
    if solver_type == "dafoam":
        dafoam = _strip_notes(dict(opt.get("dafoam") or {}))
        solver["case"] = dafoam.pop("template", "dafoam_case")
        if "container" in dafoam:
            dafoam["container"] = _env_path(dafoam["container"], "container", keep_paths)
        solver["dafoam"] = dafoam
    else:
        solver["case"] = "foam_case"
    for key in ("loading_method", "field_name", "objective_path", "invert_normals", "dafoam_output", "patch_name"):
        if key in sens:
            remarks.append(f"dropped sensitivity.{key} (derived from the metric registry now)")
    v2["solver"] = solver

    # --- objective and constraints ---------------------------------------------------
    constraints: list[dict] = []
    penalties: list[dict] = []
    if "objective_name" in opt:
        name = opt["objective_name"]
        if name not in METRIC_NAMES:
            raise MigrationError(f"objective {name!r} has no v2 metric (flowBalance was removed)")
        metric = METRIC_NAMES[name]
        con = opt.get("constraint") or {}
        if con.get("enabled", False):
            if con["name"] not in METRIC_NAMES:
                raise MigrationError(f"constraint {con['name']!r} has no v2 metric")
            budget = {"mode": con.get("target_mode", "relative_to_initial")}
            if budget["mode"] == "relative_to_initial":
                budget["factor"] = float(con.get("target_factor", 1.0))
            else:
                budget["value"] = float(con["target_value"])
            constraints.append({"type": "metric", "metric": METRIC_NAMES[con["name"]], "budget": budget})
    else:
        metric = "drag"
        constraints.append({"type": "volume"})
        if opt.get("use_center_constraint", False):
            constraints.append({"type": "centroid", "tol": float(opt.get("center_tol", 0.0))})
    jac = (opt.get("ffd") or {}).get("jacobian_constraint") or {}
    if jac.get("enabled", False) or opt.get("use_jacobian_constraint", False):
        constraints.append({
            "type": "ffd_jacobian",
            "threshold": float(jac.get("threshold", opt.get("jacobian_threshold", 0.1))),
            "samples": int(jac.get("samples", opt.get("jacobian_samples", 6))),
            "ks_rho": float(jac.get("ks_rho", 50.0)),
        })
    reg = opt.get("regularization") or {}
    if reg.get("enabled", False):
        penalties.append({"type": "proximity", "weight": float(reg.get("weight", 0.0))})
    sm = opt.get("lattice_smoothness") or {}
    if sm.get("enabled", False):
        penalties.append({"type": "lattice_smoothness", "weight": float(sm.get("weight", 0.0)),
                          "reference": sm.get("reference", "none")})
    uc = opt.get("no_undercut") or {}
    if uc.get("enabled", False):
        entry = _undercut_params(uc)
        if uc.get("mode", "objective") == "objective":
            entry["weight"] = float(uc.get("weight", 0.0))
            penalties.append(entry)
        else:
            entry["budget"] = _budget(uc, uc.get("formulation", "penalty"), "undercut")
            constraints.append(entry)
    sl = opt.get("min_steg_length") or {}
    if sl.get("enabled", False):
        entry = _steg_params(sl)
        if sl.get("mode", "constraint") == "objective":
            entry["weight"] = float(sl.get("weight", 1.0))
            penalties.append(entry)
        else:
            entry["budget"] = _budget(sl, sl.get("formulation", "penalty"), "min_steg_length")
            constraints.append(entry)
    objective = {"metric": metric}
    if penalties:
        objective["penalties"] = penalties
    v2["objective"] = objective
    v2["constraints"] = constraints

    # --- optimizer ---------------------------------------------------------------
    optimizer: dict = {"max_step": opt["max_step"], "bounds": opt["bounds"]}
    for block in ("gcmma", "feasibility_restoration", "step_control"):
        if opt.get(block):
            optimizer[block] = _strip_notes(opt[block])
    convergence = {}
    for src, dst in (("convergence_obj_tol", "obj_tol"), ("convergence_window", "window"),
                     ("convergence_min_iter", "min_iter"), ("convergence_feasibility_tol", "feasibility_tol"),
                     ("convergence_patience", "patience")):
        if src in opt:
            convergence[dst] = opt[src]
    if convergence:
        optimizer["convergence"] = convergence
    v2["optimizer"] = optimizer

    # --- diagnostics -------------------------------------------------------------
    diagnostics: dict = {}
    probe = opt.get("noise_probe") or {}
    if probe.get("enabled", False):
        diagnostics["mode"] = "noise_probe"
        diagnostics["noise_probe"] = {k: v for k, v in probe.items() if k != "enabled" and not k.startswith("_")}
    jp = opt.get("jacobian_probe") or {}
    if jp.get("enabled", False):
        diagnostics["mode"] = "jacobian_probe"
        diagnostics["jacobian_probe"] = {k: v for k, v in jp.items() if k != "enabled" and not k.startswith("_")}
    if diagnostics:
        v2["diagnostics"] = diagnostics
    for key in ("start_iter", "sensitivity_region_x", "_sensitivity_region_x"):
        if key in opt:
            remarks.append(f"dropped optimization.{key}")
    return v2, remarks


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs", nargs="+", type=Path)
    ap.add_argument("--write", action="store_true", help="replace the files in place")
    ap.add_argument("--lock-layout", default=None,
                    help="named lock layout for configs whose driver locked control points in code")
    ap.add_argument("--keep-paths", action="store_true", help="do not replace host paths by env vars")
    args = ap.parse_args(argv)

    rc = 0
    for path in args.configs:
        raw = json.loads(path.read_text())
        try:
            v2, remarks = migrate(raw, lock_layout=args.lock_layout, keep_paths=args.keep_paths)
            Config.from_dict(_env_free(v2))
        except (MigrationError, ConfigError, KeyError) as exc:
            print(f"FAILED {path}: {exc}", file=sys.stderr)
            rc = 1
            continue
        notes: list[tuple[str, str]] = []
        _collect_notes(raw, "", notes)
        if args.write:
            path.write_text(json.dumps(v2, indent=2) + "\n")
            if notes:
                notes_dir = path.parent / "notes"
                notes_dir.mkdir(exist_ok=True)
                out = notes_dir / "config_notes.md"
                with out.open("a") as f:
                    f.write(f"\n## Notes migrated from {path.name}\n\n")
                    for key, text in notes:
                        f.write(f"- `{key}`: {text}\n")
            print(f"converted {path}" + (f" ({len(notes)} notes -> notes/config_notes.md)" if notes else ""))
        else:
            print(json.dumps(v2, indent=2))
        for r in remarks:
            print(f"  note: {r}")
    return rc


def _env_free(node):
    """Replace ``${VAR}`` placeholders by dummies so the schema can be validated offline."""
    if isinstance(node, str):
        return node.replace("${", "/env/").replace("}", "")
    if isinstance(node, dict):
        return {k: _env_free(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_env_free(v) for v in node]
    return node


if __name__ == "__main__":
    sys.exit(main())

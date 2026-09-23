"""Typed experiment configuration (schema v2).

The sections mirror the building blocks of the optimization: ``run``, ``geometry``,
``parametrization``, ``mesh``, ``solver``, ``objective``, ``constraints``, ``optimizer``,
``diagnostics``. Every key a code path reads is declared here; unknown keys are rejected
with a message that names the legacy key's replacement where one exists. Keys starting with
an underscore are free-form notes and are ignored.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any


class ConfigError(ValueError):
    """Raised for a malformed or legacy (v1) configuration."""


# Legacy (v1) keys and where their content lives now. Used only for error messages.
LEGACY_KEYS = {
    "results_name": "run.name",
    "reconstruction": "geometry / parametrization.deepsdf (run deepshapeopt migrate-config)",
    "optimization": "the v2 sections (run deepshapeopt migrate-config)",
    "mesh_pipeline": "removed (sdf_hex is the only mesher)",
    "forward_solver": "solver.type",
    "objective_name": "objective.metric",
    "constraint": "constraints[] entry of type 'metric'",
    "sensitivity": "solver.openfoam.sensitivity and geometry.unit_to_metre",
    "scaling_to_mm": "geometry.unit_to_metre",
    "write_scale": "geometry.unit_to_metre",
    "mesh_resolution": "parametrization.deepsdf.reconstruction.export_resolution",
    "create_mesh_N": "parametrization.deepsdf.reconstruction.export_resolution",
    "heavy_data_output_path": "run.heavy_data_dir",
    "foam_runtime_root": "run.scratch_dir",
    "lock_domain": "parametrization.lock",
    "use_center_constraint": "constraints[] entry of type 'centroid'",
    "use_jacobian_constraint": "constraints[] entry of type 'ffd_jacobian'",
    "no_undercut": "constraints[] entry of type 'undercut' or objective.penalties[]",
    "min_steg_length": "constraints[] entry of type 'min_steg_length' or objective.penalties[]",
    "regularization": "objective.penalties[] entry of type 'proximity'",
    "lattice_smoothness": "objective.penalties[] entry of type 'lattice_smoothness'",
    "convergence_obj_tol": "optimizer.convergence.obj_tol",
    "start_iter": "removed",
    "taper": "removed",
    "min_wall_thickness": "removed",
    "section_partition": "removed",
    "reuse_parameter": "parametrization.deepsdf.reconstruction.reuse",
    "model_checkpoint": "parametrization.deepsdf.checkpoint",
    "export_rec_samples_series": "parametrization.deepsdf.reconstruction.export_samples_series",
}


def _check_keys(section: str, raw: dict, allowed: set[str]) -> None:
    unknown = [k for k in raw if not k.startswith("_") and k not in allowed]
    if unknown:
        hints = [f"{k!r}" + (f" -> {LEGACY_KEYS[k]}" if k in LEGACY_KEYS else "") for k in unknown]
        raise ConfigError(
            f"Unknown key(s) in '{section}': {', '.join(hints)}. Allowed: {sorted(allowed)}."
        )


def _require(section: str, raw: dict, key: str):
    if key not in raw:
        raise ConfigError(f"'{section}' needs the key {key!r}.")
    return raw[key]


def _dataclass_from_dict(cls, section: str, raw: dict | None, required: tuple[str, ...] = ()):
    raw = dict(raw or {})
    names = {f.name for f in fields(cls)}
    _check_keys(section, raw, names)
    for key in required:
        _require(section, raw, key)
    kwargs = {k: v for k, v in raw.items() if k in names}
    return cls(**kwargs)


def _box(section: str, value) -> list[list[float]]:
    try:
        lo, hi = value
        lo = [float(v) for v in lo]
        hi = [float(v) for v in hi]
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"'{section}' must be [[x0, y0, z0], [x1, y1, z1]], got {value!r}") from exc
    if len(lo) != 3 or len(hi) != 3:
        raise ConfigError(f"'{section}' must have two corners with three coordinates each")
    return [lo, hi]


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    name: str
    device: str = "cuda"
    num_iter: int = 1
    debug: bool = False
    heavy_data_dir: str | None = None
    scratch_dir: str | None = None

    @classmethod
    def from_dict(cls, raw):
        return _dataclass_from_dict(cls, "run", raw, required=("name",))


@dataclass
class GeometryConfig:
    mesh_path: str
    unit_to_metre: float = 1.0
    design_domain: list[list[float]] | None = None
    flow: str = "external"

    @classmethod
    def from_dict(cls, raw):
        cfg = _dataclass_from_dict(cls, "geometry", raw, required=("mesh_path",))
        if cfg.design_domain is not None:
            cfg.design_domain = _box("geometry.design_domain", cfg.design_domain)
        if cfg.flow not in ("internal", "external"):
            raise ConfigError(f"geometry.flow must be 'internal' or 'external', got {cfg.flow!r}")
        cfg.unit_to_metre = float(cfg.unit_to_metre)
        return cfg


@dataclass
class ReconstructionConfig:
    n_uniform_samples: int = 100000
    n_surface_samples: int = 500000
    samples_surface_stds: list[float] = field(default_factory=lambda: [0.025, 0.0001])
    lr: float = 0.005
    num_iterations: int = 10
    batch_size: int = 4096
    loss_fn: str = "ClampedL1"
    clamp_val: float = 0.1
    code_bound: float | None = 1.0
    code_reg_lambda: float = 0.0
    grad_clip: float | None = None
    eikonal_lambda: float = 0.0
    reuse: bool = True
    export_resolution: int = 16
    export_samples_series: bool = False
    export_samples_fine_until_epoch: int = 0
    export_samples_fine_every: int = 1
    error_cutoff: float = 0.1
    mesh_device: str | None = None

    @classmethod
    def from_dict(cls, raw):
        cfg = _dataclass_from_dict(cls, "parametrization.deepsdf.reconstruction", raw)
        if cfg.loss_fn not in ("ClampedL1", "L1", "MSE"):
            raise ConfigError(f"reconstruction.loss_fn must be ClampedL1, L1 or MSE, got {cfg.loss_fn!r}")
        return cfg


@dataclass
class PCAConfig:
    enabled: bool = False
    n_components: int = 16
    bounds: list[float] = field(default_factory=lambda: [-3.5, 3.5])
    max_step: float | None = None
    cache_path: str | None = None

    @classmethod
    def from_dict(cls, raw):
        return _dataclass_from_dict(cls, "parametrization.deepsdf.pca", raw)


@dataclass
class DeepSDFConfig:
    model_path: str
    tiling: list[int]
    checkpoint: str = "latest"
    spline_degree: list[int] = field(default_factory=lambda: [1, 1, 1])
    tiling_map: str = "hat"
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    pca: PCAConfig = field(default_factory=PCAConfig)

    @classmethod
    def from_dict(cls, raw):
        raw = dict(raw or {})
        _check_keys("parametrization.deepsdf", raw, {f.name for f in fields(cls)})
        for key in ("model_path", "tiling"):
            _require("parametrization.deepsdf", raw, key)
        raw["reconstruction"] = ReconstructionConfig.from_dict(raw.get("reconstruction"))
        raw["pca"] = PCAConfig.from_dict(raw.get("pca"))
        cfg = cls(**{k: v for k, v in raw.items() if not k.startswith("_")})
        cfg.tiling = [int(t) for t in cfg.tiling]
        cfg.spline_degree = [int(p) for p in cfg.spline_degree]
        if len(cfg.tiling) != 3 or len(cfg.spline_degree) != 3:
            raise ConfigError("parametrization.deepsdf.tiling and spline_degree need three entries")
        if cfg.tiling_map not in ("hat", "cosine"):
            raise ConfigError(f"parametrization.deepsdf.tiling_map must be 'hat' or 'cosine', got {cfg.tiling_map!r}")
        return cfg


@dataclass
class FFDConfig:
    n_control_points: list[int]
    spline_degree: list[int] = field(default_factory=lambda: [2, 2, 2])

    @classmethod
    def from_dict(cls, raw):
        cfg = _dataclass_from_dict(cls, "parametrization.ffd", raw, required=("n_control_points",))
        cfg.n_control_points = [int(n) for n in cfg.n_control_points]
        cfg.spline_degree = [int(p) for p in cfg.spline_degree]
        return cfg


@dataclass
class LockConfig:
    faces: list[str] = field(default_factory=list)
    boxes_physical: list = field(default_factory=list)
    boxes_norm: list = field(default_factory=list)
    layout: str | None = None
    safety: float = 0.01

    @classmethod
    def from_dict(cls, raw):
        cfg = _dataclass_from_dict(cls, "parametrization.lock", raw)
        if cfg.layout is not None and cfg.layout != "outlet_face_and_inlet_rim":
            raise ConfigError(
                f"parametrization.lock.layout {cfg.layout!r} unknown; valid: 'outlet_face_and_inlet_rim'"
            )
        return cfg


@dataclass
class ParametrizationConfig:
    type: str
    deepsdf: DeepSDFConfig | None = None
    ffd: FFDConfig | None = None
    lock: LockConfig = field(default_factory=LockConfig)

    @classmethod
    def from_dict(cls, raw):
        raw = dict(raw or {})
        _check_keys("parametrization", raw, {f.name for f in fields(cls)})
        kind = _require("parametrization", raw, "type")
        if kind not in ("deepsdf", "ffd"):
            raise ConfigError(f"parametrization.type must be 'deepsdf' or 'ffd', got {kind!r}")
        deepsdf = DeepSDFConfig.from_dict(raw["deepsdf"]) if raw.get("deepsdf") is not None else None
        ffd = FFDConfig.from_dict(raw["ffd"]) if raw.get("ffd") is not None else None
        if kind == "deepsdf" and deepsdf is None:
            raise ConfigError("parametrization.type 'deepsdf' needs a parametrization.deepsdf block")
        if kind == "ffd" and ffd is None:
            raise ConfigError("parametrization.type 'ffd' needs a parametrization.ffd block")
        return cls(type=kind, deepsdf=deepsdf, ffd=ffd, lock=LockConfig.from_dict(raw.get("lock")))


@dataclass
class MeshConfig:
    sdf_hex: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw):
        raw = dict(raw or {})
        _check_keys("mesh", raw, {"sdf_hex"})
        sdf_hex = raw.get("sdf_hex")
        if not isinstance(sdf_hex, dict):
            raise ConfigError("mesh.sdf_hex must be a dictionary (the hex mesher settings)")
        if "write_scale" in sdf_hex:
            raise ConfigError("mesh.sdf_hex.write_scale is set by geometry.unit_to_metre; remove it")
        return cls(sdf_hex=dict(sdf_hex))


@dataclass
class SensitivityConfig:
    field_suffix: str = "ESI"
    warn_tol: float = 1e-3

    @classmethod
    def from_dict(cls, raw):
        cfg = _dataclass_from_dict(cls, "solver.openfoam.sensitivity", raw)
        if cfg.field_suffix not in ("ESI", "SI"):
            raise ConfigError("solver.openfoam.sensitivity.field_suffix must be 'ESI' or 'SI'")
        return cfg


@dataclass
class OpenFoamConfig:
    solver_convergence: dict | None = None
    dict_overrides: dict = field(default_factory=dict)
    sensitivity: SensitivityConfig = field(default_factory=SensitivityConfig)

    @classmethod
    def from_dict(cls, raw):
        raw = dict(raw or {})
        _check_keys("solver.openfoam", raw, {f.name for f in fields(cls)})
        return cls(
            solver_convergence=raw.get("solver_convergence"),
            dict_overrides=dict(raw.get("dict_overrides") or {}),
            sensitivity=SensitivityConfig.from_dict(raw.get("sensitivity")),
        )


@dataclass
class SolverConfig:
    type: str = "openfoam"
    case: str | None = None
    openfoam: OpenFoamConfig = field(default_factory=OpenFoamConfig)
    dafoam: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw):
        raw = dict(raw or {})
        _check_keys("solver", raw, {f.name for f in fields(cls)})
        kind = str(raw.get("type", "openfoam"))
        if kind not in ("openfoam", "dafoam"):
            raise ConfigError(f"solver.type must be 'openfoam' or 'dafoam', got {kind!r}")
        case = raw.get("case") or ("dafoam_case" if kind == "dafoam" else "foam_case")
        dafoam = dict(raw.get("dafoam") or {})
        if kind == "dafoam" and not dafoam:
            raise ConfigError("solver.type 'dafoam' needs a solver.dafoam block")
        return cls(type=kind, case=str(case), openfoam=OpenFoamConfig.from_dict(raw.get("openfoam")), dafoam=dafoam)


@dataclass
class ObjectiveConfig:
    metric: str
    penalties: list[dict] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw):
        raw = dict(raw or {})
        _check_keys("objective", raw, {"metric", "penalties"})
        metric = _require("objective", raw, "metric")
        penalties = list(raw.get("penalties") or [])
        for i, pen in enumerate(penalties):
            if not isinstance(pen, dict) or "type" not in pen:
                raise ConfigError(f"objective.penalties[{i}] needs a 'type'")
        return cls(metric=str(metric), penalties=penalties)


@dataclass
class ConvergenceConfig:
    obj_tol: float | None = None
    window: int = 15
    min_iter: int | None = None
    feasibility_tol: float = 1e-3
    patience: int = 1

    @classmethod
    def from_dict(cls, raw):
        cfg = _dataclass_from_dict(cls, "optimizer.convergence", raw)
        if cfg.min_iter is None:
            cfg.min_iter = int(cfg.window) + 5
        return cfg


@dataclass
class OptimizerConfig:
    max_step: float
    bounds: list[float]
    gcmma: dict = field(default_factory=lambda: {"enabled": False})
    feasibility_restoration: dict = field(default_factory=lambda: {"enabled": False})
    step_control: dict = field(default_factory=lambda: {"enabled": False})
    convergence: ConvergenceConfig = field(default_factory=ConvergenceConfig)

    @classmethod
    def from_dict(cls, raw):
        raw = dict(raw or {})
        _check_keys("optimizer", raw, {f.name for f in fields(cls)})
        for key in ("max_step", "bounds"):
            _require("optimizer", raw, key)
        for block, allowed in (
            ("gcmma", {"enabled", "max_inner", "feas_tol"}),
            ("feasibility_restoration", {"enabled", "tol", "max_steps", "step_limit"}),
            ("step_control", {"enabled", "rho_shrink", "rho_grow", "shrink", "grow",
                              "min_step_factor", "pred_tol_rel", "patience"}),
        ):
            _check_keys(f"optimizer.{block}", raw.get(block) or {}, allowed)
        return cls(
            max_step=float(raw["max_step"]),
            bounds=[float(b) for b in raw["bounds"]],
            gcmma=dict(raw.get("gcmma") or {"enabled": False}),
            feasibility_restoration=dict(raw.get("feasibility_restoration") or {"enabled": False}),
            step_control=dict(raw.get("step_control") or {"enabled": False}),
            convergence=ConvergenceConfig.from_dict(raw.get("convergence")),
        )


@dataclass
class ExportsConfig:
    vtk_series: bool = True
    stl_series: bool = True
    sens_series: bool = True
    snapshots: bool = False

    @classmethod
    def from_dict(cls, raw):
        return _dataclass_from_dict(cls, "diagnostics.exports", raw)


@dataclass
class DiagnosticsConfig:
    mode: str = "optimize"
    noise_probe: dict = field(default_factory=dict)
    jacobian_probe: dict = field(default_factory=dict)
    exports: ExportsConfig = field(default_factory=ExportsConfig)

    @classmethod
    def from_dict(cls, raw):
        raw = dict(raw or {})
        _check_keys("diagnostics", raw, {f.name for f in fields(cls)})
        mode = str(raw.get("mode", "optimize"))
        if mode not in ("optimize", "noise_probe", "jacobian_probe"):
            raise ConfigError(f"diagnostics.mode must be optimize, noise_probe or jacobian_probe, got {mode!r}")
        _check_keys("diagnostics.noise_probe", raw.get("noise_probe") or {},
                    {"start_parameters", "h", "n_points", "seed", "reuse_castellation"})
        _check_keys("diagnostics.jacobian_probe", raw.get("jacobian_probe") or {},
                    {"start_parameters", "batch", "keep_vectors"})
        return cls(
            mode=mode,
            noise_probe=dict(raw.get("noise_probe") or {}),
            jacobian_probe=dict(raw.get("jacobian_probe") or {}),
            exports=ExportsConfig.from_dict(raw.get("exports")),
        )


# ---------------------------------------------------------------------------
# Whole config
# ---------------------------------------------------------------------------

TOP_LEVEL = {"run", "geometry", "parametrization", "mesh", "solver", "objective",
             "constraints", "optimizer", "diagnostics"}
OPTIMIZATION_SECTIONS = ("mesh", "solver", "objective", "optimizer")


@dataclass
class Config:
    run: RunConfig
    geometry: GeometryConfig
    parametrization: ParametrizationConfig
    mesh: MeshConfig | None = None
    solver: SolverConfig | None = None
    objective: ObjectiveConfig | None = None
    constraints: list[dict] = field(default_factory=list)
    optimizer: OptimizerConfig | None = None
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def is_optimization(self) -> bool:
        return all(getattr(self, s) is not None for s in OPTIMIZATION_SECTIONS)

    @classmethod
    def from_dict(cls, raw: dict) -> "Config":
        if not isinstance(raw, dict):
            raise ConfigError("the configuration must be a JSON object")
        if "reconstruction" in raw or "optimization" in raw or "results_name" in raw:
            raise ConfigError(
                "This is a v1 configuration (top-level 'reconstruction'/'optimization'). "
                "Convert it with `deepshapeopt migrate-config`."
            )
        _check_keys("config", raw, TOP_LEVEL)
        for key in ("run", "geometry", "parametrization"):
            _require("config", raw, key)
        constraints = list(raw.get("constraints") or [])
        for i, con in enumerate(constraints):
            if not isinstance(con, dict) or "type" not in con:
                raise ConfigError(f"constraints[{i}] needs a 'type'")
        cfg = cls(
            run=RunConfig.from_dict(raw["run"]),
            geometry=GeometryConfig.from_dict(raw["geometry"]),
            parametrization=ParametrizationConfig.from_dict(raw["parametrization"]),
            mesh=MeshConfig.from_dict(raw["mesh"]) if raw.get("mesh") is not None else None,
            solver=SolverConfig.from_dict(raw["solver"]) if raw.get("solver") is not None else None,
            objective=ObjectiveConfig.from_dict(raw["objective"]) if raw.get("objective") is not None else None,
            constraints=constraints,
            optimizer=OptimizerConfig.from_dict(raw["optimizer"]) if raw.get("optimizer") is not None else None,
            diagnostics=DiagnosticsConfig.from_dict(raw.get("diagnostics")),
            raw=raw,
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        present = [s for s in OPTIMIZATION_SECTIONS if getattr(self, s) is not None]
        if present and len(present) != len(OPTIMIZATION_SECTIONS):
            missing = [s for s in OPTIMIZATION_SECTIONS if getattr(self, s) is None]
            raise ConfigError(f"an optimization config needs all of {OPTIMIZATION_SECTIONS}; missing {missing}")
        if self.is_optimization:
            if self.geometry.design_domain is None:
                raise ConfigError("geometry.design_domain is required for an optimization")
            if self.parametrization.type == "ffd" and self.parametrization.deepsdf is None and self.geometry.flow is None:
                raise ConfigError("geometry.flow is required")

    def require_optimization(self) -> None:
        if not self.is_optimization:
            missing = [s for s in OPTIMIZATION_SECTIONS if getattr(self, s) is None]
            raise ConfigError(f"this command needs an optimization config; missing sections {missing}")

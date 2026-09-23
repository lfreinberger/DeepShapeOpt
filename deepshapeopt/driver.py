"""The optimization loop: build the mesh, solve, assemble the rows, step, record."""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from .diagnostics.exports import export_wall_stl, save_mesh_with_sensitivities, save_points_with_vectors
from .diagnostics.history import RunHistory
from .diagnostics.path_consistency import PathConsistency
from .diagnostics.plots import plot_convergence_diagnostics, plot_optimization_history, plot_residuals_from_log, save_shape_snapshot
from .diagnostics.probes import JacobianProbe, NoiseProbe
from .logging_setup import log_iteration_summary, log_timing
from .optimizer.convergence import has_converged, is_feasible, progress_gain
from .optimizer.mma import MMAOptimizer
from .optimizer.step_control import StepControl
from .problem import Problem
from .terms.base import Row, State
from .terms.cfd import CfdConstraint
from .terms.volume import CentroidTerm, VolumeTerm

logger = logging.getLogger(__name__)

SERIES = ("vtk_series", "stl_series", "sens_series", "control_lattice_series")


class OptimizationLoop:
    def __init__(self, problem: Problem):
        self.p = problem
        cfg = problem.cfg
        self.cfg = cfg
        self.debug = cfg.run.debug
        self.paths = problem.paths
        self.out = self.paths.optimization
        self.heavy = self.paths.heavy_data
        self.exports = cfg.diagnostics.exports
        if self.heavy is not None:
            for name in SERIES:
                d = self.heavy / name
                if d.exists():
                    shutil.rmtree(d)
                d.mkdir(parents=True)
        self.snapshots = self.out / "snapshots"

        self.optimizer = MMAOptimizer(problem.space, cfg.optimizer, n_constraints=len(problem.constraints))
        self.step_control = StepControl(cfg.optimizer.step_control, self.optimizer.max_step)
        self.path = PathConsistency()
        self.convergence = cfg.optimizer.convergence
        self.history = RunHistory(self.out, cfg.raw, total_iters=cfg.run.num_iter)
        self.mode = cfg.diagnostics.mode
        self.num_iter = int(cfg.run.num_iter)

        self.noise_probe = None
        self.jacobian_probe = None
        if self.mode == "noise_probe":
            self.noise_probe = NoiseProbe(cfg.diagnostics.noise_probe, problem.space, self.paths.experiment, self.out)
            self.num_iter = self.noise_probe.n_points
        elif self.mode == "jacobian_probe":
            self.jacobian_probe = JacobianProbe(cfg.diagnostics.jacobian_probe, problem.space, self.paths.experiment,
                                                self.out, max_step=self.optimizer.max_step)
            self.num_iter = 1
        self.step_control.enabled = self.step_control.enabled and self.noise_probe is None

        self.h_objective: list[float] = []
        self.h_total: list[float] = []
        self.h_penalty: dict[str, list[float]] = {t.name: [] for t in problem.penalties}
        self.h_constraint: dict[str, list[float]] = {t.name: [] for t in problem.constraints}
        self.h_feasible: list[bool] = []
        self.h_gain: list[float] = []
        self.diag: dict[str, list[float]] = {}
        self.objective_scale: float | None = None
        self.final_objective = float("nan")

    # ------------------------------------------------------------------ helpers
    def _state(self, iteration: int, mesh=None, result=None) -> State:
        return State(iteration=iteration, param=self.p.param, parametrization=self.p.parametrization,
                     design_space=self.p.space, mesher=self.p.mesher, unit_to_metre=self.cfg.geometry.unit_to_metre,
                     debug=self.debug, results_dir=self.out, heavy_dir=self.heavy, mesh=mesh, solver=result,
                     objective_scale=self.objective_scale)

    def _initial_geometry(self) -> None:
        """Volume / centroid baseline from the start design (one extra mesh build)."""
        geo = [t for t in self.p.constraints if isinstance(t, (VolumeTerm, CentroidTerm))]
        if not geo:
            return
        self.p.mesher.build()
        volume, centroid = self.p.mesher.volume_centroid(no_grad=True)
        logger.info("Initial volume %.6f, centroid %s", float(volume), centroid.detach().cpu().numpy().tolist())
        for t in geo:
            t.set_initial(volume, centroid)

    def _cheap_eval(self, state: State, rows: list[Row]):
        cheap = [(i, t) for i, t in enumerate(self.p.constraints) if t.cheap]
        if not cheap:
            return None
        space = self.p.space

        def evaluate(x_np, with_grad: bool):
            space.write_vector(x_np)
            vals, grads = [], []
            for _, term in cheap:
                tv = term.evaluate(state)
                vals.append(tv.value - term.target)
                if with_grad:
                    grads.append(space.grad_to_vector(tv.grad).reshape(1, -1))
            vals = np.asarray(vals, dtype=float)
            if not with_grad:
                return vals, None
            return vals, torch.cat(grads, dim=0).detach().cpu().numpy().astype(float)

        return evaluate

    def _export_term_debug(self, name: str, tv, mesh) -> None:
        cloud = tv.debug.get("cloud")
        cloud_path = self.out / f"{name}_points.vtp"
        if cloud is not None and len(cloud[0]):
            save_points_with_vectors(cloud[0], cloud[1], cloud_path, scalars=cloud[2])
        elif cloud_path.exists():
            cloud_path.unlink()
        faces = tv.debug.get("faces")
        if faces is not None:
            import trimesh

            tris, centroids, normals, ndotd = faces
            stl, vtp = self.out / f"{name}_faces.stl", self.out / f"{name}_normals.vtp"
            if len(tris):
                m = trimesh.Trimesh(vertices=mesh.verts.detach().cpu().numpy(), faces=tris, process=False)
                m.remove_unreferenced_vertices()
                m.export(str(stl))
                save_points_with_vectors(centroids, normals, vtp, scalars={"n_dot_d": ndotd})
            else:
                for pth in (stl, vtp):
                    if pth.exists():
                        pth.unlink()

    def _export_sensitivities(self, name: str, tv, mesh, iteration: int) -> None:
        sens_normal = tv.debug.get("sens_normal")
        if sens_normal is None:
            return
        verts = mesh.verts * self.cfg.geometry.unit_to_metre
        vtp = self.out / f"check_sens_{name}.vtp"
        save_mesh_with_sensitivities(verts, mesh.faces, sens_normal, vtp)
        if self.heavy is not None and self.exports.sens_series:
            shutil.copy2(vtp, self.heavy / "sens_series" / f"check_sens_{name}_{iteration:04d}.vtp")

    # ------------------------------------------------------------------ loop
    def run(self) -> dict:
        try:
            return self._run()
        finally:
            self.history.close()
            self.p.solver.close()

    def _run(self) -> dict:
        p, cfg = self.p, self.cfg
        start_time = time.time()
        iteration_times: list[float] = []
        stop_reason = None
        if self.mode == "optimize":
            self._initial_geometry()
        if self.debug and self.exports.snapshots:
            self.snapshots.mkdir(parents=True, exist_ok=True)

        for it in range(self.num_iter):
            logger.info("=== Optimization iteration %d/%d ===", it, self.num_iter - 1)
            iter_start = time.time()
            self.history.start_iteration(it)

            reuse = self.noise_probe is not None and self.noise_probe.reuse_castellation and it > 0
            mesh = p.mesher.build(reuse_castellation=reuse)
            export_wall_stl(mesh.verts, mesh.faces_design, self.out / "current_shape.stl")
            p.parametrization.export_iteration(
                self.out, it, self.heavy / "control_lattice_series" if self.heavy is not None else None)
            if self.debug and self.exports.snapshots:
                save_shape_snapshot(verts=mesh.verts, faces=mesh.faces, design_domain=p.parametrization.frame.design_domain,
                                    out_path=self.snapshots / f"shape_{it:04d}.png", view_axis="z", title=f"Iteration {it}")

            if self.jacobian_probe is not None:
                self.jacobian_probe.run(mesh.verts, mesh.faces_design)
                return {"results_dir": str(self.paths.results), "final_objective": float("nan")}

            result = p.solver.evaluate(mesh, p.metrics)
            if self.debug and result.info.get("log") and Path(result.info["log"]).exists():
                plot_residuals_from_log(result.info["log"], output_dir=str(self.out))

            state = self._state(it, mesh, result)
            tv_obj = p.objective.evaluate(state)
            J, dJ = tv_obj.value, tv_obj.grad
            if self.objective_scale is None:
                self.objective_scale = abs(J) if np.isfinite(J) and J != 0.0 else 1.0
                state.objective_scale = self.objective_scale
            logger.info("%s: J=%.6e |dJ/dx|=%.3e |dJ/dp|=%.3e", p.objective.name, J,
                        float(np.linalg.norm(result.sensitivities[p.objective.name])), float(dJ.norm()))
            self._export_sensitivities(p.objective.name, tv_obj, mesh, it)

            J_total, dJ_total = J, dJ
            contributions = {}
            for term in p.penalties:
                tv = term.evaluate(state)
                value, grad = term.contribution(tv, self.objective_scale)
                J_total = J_total + value
                dJ_total = dJ_total + grad
                contributions[term.name] = value
                logger.info("%s (penalty): value=%.3e contribution=%.3e |w*grad|=%.3e", term.name, tv.value, value,
                            float(grad.norm()))
                if self.debug:
                    self._export_term_debug(term.name, tv, mesh)

            rows: list[Row] = []
            for term in p.constraints:
                tv = term.evaluate(state)
                row = term.row(tv)
                rows.append(row)
                logger.info("%s (constraint): value=%.6e target=%.6e |grad|=%.3e %s", term.name, tv.value,
                            row.target, float(tv.grad.norm()), tv.debug.get("reading", ""))
                if self.debug:
                    self._export_term_debug(term.name, tv, mesh)
                if isinstance(term, CfdConstraint):
                    self._export_sensitivities(term.name, tv, mesh, it)

            if self.heavy is not None:
                if self.exports.vtk_series:
                    p.solver.export_fields(result, it, self.heavy)
                if self.exports.stl_series:
                    shutil.copy2(self.out / "current_shape.stl", self.heavy / "stl_series" / f"shape_{it:04d}.stl")

            # history plot
            self.h_objective.append(J)
            self.h_total.append(J_total)
            for name, value in contributions.items():
                self.h_penalty[name].append(value)
            for row in rows:
                self.h_constraint[row.name].append(row.raw_value)
            plot_optimization_history(
                self.h_objective, result_dir=self.out, obj_label=p.objective.name,
                history_obj_total=self.h_total if p.penalties else None,
                obj_total_label=" + ".join([p.objective.name] + [t.name for t in p.penalties]),
                extra_series=[(f"{n} (weighted)", v) for n, v in self.h_penalty.items()],
                constraints=[(r.name, self.h_constraint[r.name], r.target) for r in rows],
            )

            verts_design_old = mesh.verts_design.detach()
            dJ_vec = p.space.grad_to_vector(dJ_total).detach().cpu().numpy().reshape(-1)
            G_raw = np.asarray([r.value for r in rows], dtype=float)
            dG_vec = np.stack([p.space.grad_to_vector(r.grad).detach().cpu().numpy().reshape(-1) for r in rows]) \
                if rows else np.zeros((0, dJ_vec.size))
            path_info = self.path.observe(dJ_vec, J_total, G_raw, dG_vec)
            self.step_control.observe(self.optimizer, J_total, G_raw)

            if self.noise_probe is not None:
                cfd_rows = [(r, t) for r, t in zip(rows, p.constraints) if isinstance(t, CfdConstraint)]
                dJ_cfd = p.space.grad_to_vector(dJ)
                if cfd_rows:
                    self.noise_probe.record(J, dJ_cfd, cfd_rows[0][0].raw_value,
                                            p.space.grad_to_vector(cfd_rows[0][0].grad))
                else:
                    self.noise_probe.record(J, dJ_cfd, None, None)
                p.solver.clean_iteration(result)
                iteration_times.append(time.time() - iter_start)
                continue

            self.optimizer.step(J_total, dJ_total, rows, self._cheap_eval(state, rows))
            dx = self.optimizer.step_vector
            self.path.record_step(dx, J_total, dJ_vec, G_raw, dG_vec)
            self.step_control.predict(self.optimizer, J_total)
            p.parametrization.save(self.out / "parameters.pt")

            # convergence diagnostics
            step_inf = float(np.abs(dx).max())
            step_ratio = step_inf / self.optimizer.max_step if self.optimizer.max_step > 0 else float("nan")
            grad_norm = float(np.linalg.norm(dJ_vec))
            obj_change = (abs(self.h_objective[-1] - self.h_objective[-2]) / max(abs(self.h_objective[-2]), 1e-300)
                          if len(self.h_objective) >= 2 else float("nan"))
            wall_disp = p.mesher.wall_displacement(verts_design_old)
            wall_max, wall_mean = float(np.abs(wall_disp).max()), float(wall_disp.mean())
            kkt_norm = self.optimizer.kkt_norm
            lam0 = float(self.optimizer.lam[0]) if rows else None
            diag_vals = {"obj_change": obj_change, "grad_norm": grad_norm, "mma_ch": self.optimizer.ch,
                         "step_inf": step_inf, "step_ratio": step_ratio, "wall_disp_max": wall_max,
                         "wall_disp_mean": wall_mean, "kkt_norm": kkt_norm,
                         "path_rho_fwd": _nan(path_info["path_rho_fwd"]), "path_rho_trap": _nan(path_info["path_rho_trap"])}
            if self.step_control.enabled:
                diag_vals.update(step_control_rho=self.step_control.rho, max_step=self.optimizer.max_step)
            for key, val in diag_vals.items():
                self.diag.setdefault(key, []).append(val)
            plot_convergence_diagnostics(self.diag, self.out)

            p.solver.clean_iteration(result)

            feasible = is_feasible([r.value for r in rows], [1.0 if r.scale is None else r.scale for r in rows],
                                   self.convergence.feasibility_tol) if rows else True
            self.h_feasible.append(feasible)
            gain = (progress_gain(self.h_objective, self.convergence.window, self.h_feasible)
                    if len(self.h_objective) > self.convergence.min_iter else float("nan"))
            self.h_gain.append(gain)
            if gain == gain:
                logger.info("progress: best feasible objective gained %.3f %% over the last %d iterations",
                            100.0 * gain, self.convergence.window)
            if stop_reason is None and has_converged(self.h_gain, self.convergence.obj_tol, self.convergence.patience):
                stop_reason = (f"best feasible objective gained < {100 * float(self.convergence.obj_tol):.2f} % "
                               f"over {self.convergence.window} iterations")
            if self.step_control.stop_reason:
                stop_reason = self.step_control.stop_reason

            summary = {"objective": J}
            if p.penalties:
                summary["objective_total"] = J_total
            for row in rows:
                summary[row.name] = row.raw_value
                summary[f"{row.name}_target"] = row.target
            summary.update(mma_ch=self.optimizer.ch, step_ratio=step_ratio, grad_norm=grad_norm, kkt_norm=kkt_norm,
                           lam=lam0, wall_disp_max=wall_max, wall_disp_mean=wall_mean)
            log_iteration_summary(logger, **summary)
            log_timing(logger, iter_start, start_time, iteration_times, self.num_iter, it)

            record = dict(summary)
            record.update({f"{n}_contribution": v for n, v in contributions.items()})
            record.update(obj_change=obj_change, step_inf=step_inf, max_param=float(p.param.abs().max()),
                          max_step=self.optimizer.max_step, progress_gain=gain, solver_time_s=result.time_s,
                          path_pred=path_info["path_pred"], path_real=path_info["path_real"],
                          path_rho_fwd=path_info["path_rho_fwd"], path_rho_trap=path_info["path_rho_trap"],
                          path_rho_fwd_con=path_info["path_rho_fwd_con"], path_rho_trap_con=path_info["path_rho_trap_con"],
                          step_control_rho=self.step_control.rho if self.step_control.enabled else None)
            for name, value in result.values.items():
                record[f"metric_{name}"] = value
            self.history.log_iteration(iteration=it, **record)
            self.final_objective = J
            if stop_reason is not None:
                logger.info("converged at iteration %d (%s); stopping", it, stop_reason)
                break

        if self.noise_probe is not None:
            self.noise_probe.finalize()
        return {"results_dir": str(self.paths.results), "final_objective": float(self.final_objective)}


def _nan(value):
    return float("nan") if value is None else value


def run(cfg, experiment_dir: Path) -> dict:
    from .problem import build_problem

    return OptimizationLoop(build_problem(cfg, experiment_dir)).run()

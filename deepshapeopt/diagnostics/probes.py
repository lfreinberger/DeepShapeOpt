"""Run modes that measure instead of optimizing: the noise probe and the jacobian probe."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from deepshapeopt.parametrization.base import DesignSpace, load_parameter_file

logger = logging.getLogger(__name__)


def _resolve_start(path_value, experiment_dir: Path) -> Path:
    from deepshapeopt.config import expand_env_vars

    p = Path(expand_env_vars(str(path_value))).expanduser()
    return p if p.is_absolute() else experiment_dir / p


class NoiseProbe:
    """Evaluate the pipeline at ``x_base + t_i h p`` along one random free direction ``p``
    (max-norm 1) and estimate the numerical noise of J, the constraint and both gradients with
    ECnoise. Writes ``noise_probe_samples.npz`` (input of tools/check_gradient_fd.py) and
    ``noise_estimate.json``."""

    def __init__(self, cfg: dict, space: DesignSpace, experiment_dir: Path, out_dir: Path):
        self.space = space
        self.out_dir = Path(out_dir)
        self.n_points = int(cfg.get("n_points", 9))
        self.h = float(cfg.get("h", 0.002))
        self.reuse_castellation = bool(cfg.get("reuse_castellation", False))
        start = _resolve_start(cfg["start_parameters"], experiment_dir)
        param = space.param
        self.x_base = load_parameter_file(start, param).clone()
        self.direction = space.free_direction(int(cfg.get("seed", 0)))
        self.offsets = [i - 0.5 * (self.n_points - 1) for i in range(self.n_points)]
        self.J, self.con, self.dJ, self.dc = [], [], [], []
        self.set_point(0)
        logger.info("NOISE PROBE: %d evaluations along a random direction, h=%.3e (max-norm), start %s",
                    self.n_points, self.h, start)

    def set_point(self, i: int) -> None:
        with torch.no_grad():
            self.space.param.copy_(self.x_base + self.offsets[i] * self.h * self.direction)

    def record(self, J: float, dJ_vec: torch.Tensor, con: float | None, dc_vec: torch.Tensor | None) -> None:
        self.J.append(float(J))
        self.dJ.append(dJ_vec.reshape(-1).detach().cpu().numpy())
        if con is not None:
            self.con.append(float(con))
            self.dc.append(dc_vec.reshape(-1).detach().cpu().numpy())
        n = len(self.J)
        np.savez(
            self.out_dir / "noise_probe_samples.npz",
            offsets=np.asarray(self.offsets[:n]), h=self.h, direction=self.direction.reshape(-1).cpu().numpy(),
            J=np.asarray(self.J), con=np.asarray(self.con), dJ=np.stack(self.dJ),
            dc=np.stack(self.dc) if self.dc else np.zeros((0, 0)),
        )
        logger.info("probe point %d/%d: J=%.9e", n, self.n_points, J)
        if n < self.n_points:
            self.set_point(n)

    def finalize(self) -> None:
        from .noise import ecnoise, ecnoise_vector

        if len(self.J) < 4:
            logger.info("NOISE PROBE: %d evaluation(s), too few for a noise estimate; samples written", len(self.J))
            return
        J, dJ = np.asarray(self.J), np.stack(self.dJ)
        eps, lev, inf = ecnoise(J)
        eps_g, lev_g, inf_g = ecnoise_vector(dJ)
        g_mean = float(np.linalg.norm(dJ.mean(axis=0)))
        est = {
            "h": self.h, "n_points": int(J.size),
            "inform_legend": "1 noise detected, 2 h too small, 3 h too large",
            "eps_obj": eps, "inform_obj": inf, "levels_obj": lev.tolist(), "mean_obj": float(J.mean()),
            "rel_eps_obj": eps / max(abs(J.mean()), 1e-300),
            "eps_grad_obj": eps_g, "inform_grad_obj": inf_g, "levels_grad_obj": lev_g.tolist(),
            "mean_grad_obj_norm": g_mean, "rel_eps_grad_obj": eps_g / max(g_mean, 1e-300),
        }
        if self.con:
            con, dc = np.asarray(self.con), np.stack(self.dc)
            eps_c, lev_c, inf_c = ecnoise(con)
            eps_gc, lev_gc, inf_gc = ecnoise_vector(dc)
            gc_mean = float(np.linalg.norm(dc.mean(axis=0)))
            est.update({
                "eps_con": eps_c, "inform_con": inf_c, "levels_con": lev_c.tolist(), "mean_con": float(con.mean()),
                "rel_eps_con": eps_c / max(abs(con.mean()), 1e-300),
                "eps_grad_con": eps_gc, "inform_grad_con": inf_gc, "levels_grad_con": lev_gc.tolist(),
                "mean_grad_con_norm": gc_mean, "rel_eps_grad_con": eps_gc / max(gc_mean, 1e-300),
            })
        (self.out_dir / "noise_estimate.json").write_text(json.dumps(est, indent=2) + "\n")
        logger.info("NOISE PROBE: eps_obj=%.3e (inform %d, rel %.2e) | eps_grad_obj=%.3e (inform %d, rel %.2e)",
                    eps, inf, est["rel_eps_obj"], eps_g, inf_g, est["rel_eps_grad_obj"])


class JacobianProbe:
    """Geometric metric of the parametrization at one design: the Jacobian of the design
    surface's area-weighted normal displacement w.r.t. the free variables, its spectrum,
    per-variable norms and column cosines. Writes ``jacobian_probe.npz``; runs no solver."""

    def __init__(self, cfg: dict, space: DesignSpace, experiment_dir: Path, out_dir: Path, max_step: float):
        self.space = space
        self.out_dir = Path(out_dir)
        self.batch = int(cfg.get("batch", 64))
        self.keep_vectors = int(cfg.get("keep_vectors", 50))
        self.max_step = float(max_step)
        self.start = cfg.get("start_parameters")
        if self.start:
            path = _resolve_start(self.start, experiment_dir)
            with torch.no_grad():
                space.param.copy_(load_parameter_file(path, space.param))
            logger.info("JACOBIAN PROBE at the design of %s", path)
        else:
            logger.info("JACOBIAN PROBE at the reconstruction")

    def run(self, verts: torch.Tensor, faces_design: torch.Tensor) -> None:
        from . import latent_metric as lm

        y, ids, area = lm.design_surface_field(verts, faces_design)
        mask_free = self.space.mask_free_flat
        J = lm.wall_jacobian(y, self.space.param, mask_free, batch=self.batch)
        param = self.space.param
        n_lat = int(param.shape[1]) if param.ndim == 2 else 1
        area_total = float(area.sum())
        analysis = lm.analyze_jacobian(J, n_lat, keep_vectors=self.keep_vectors, max_step=self.max_step,
                                       area_total=area_total)
        for line in lm.summary_lines(analysis):
            logger.info("JACOBIAN PROBE: %s", line)
        np.savez(
            self.out_dir / "jacobian_probe.npz",
            sigma=analysis["sigma"], col_norm=analysis["col_norm"], cos_max=analysis["cos_max"],
            V_top=analysis["V_top"], U_top=analysis["U_top"],
            cp_norm=analysis.get("cp_norm", np.zeros(0)), cp_block_rank=analysis.get("cp_block_rank", np.zeros(0)),
            mask_free=mask_free.cpu().numpy(), surface_ids=ids.cpu().numpy(),
            surface_area=area.detach().cpu().numpy(), area_total=area_total,
            max_step=self.max_step, n_latent_per_cp=n_lat, start_parameters=str(self.start or "reconstruction"),
        )
        logger.info("JACOBIAN PROBE: written %s; no solver run", self.out_dir / "jacobian_probe.npz")

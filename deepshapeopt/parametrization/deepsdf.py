"""DeepSDF latent lattice: a B-spline of latent codes over the design domain."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import trimesh
from DeepSDFStruct.geom_reconstruction import build_parameter_spline
from DeepSDFStruct.lattice_structure import LatticeSDFStruct
from DeepSDFStruct.mesh import export_reconstructed_artifacts
from DeepSDFStruct.parametrization import SplineParametrization
from DeepSDFStruct.pretrained_models import get_model
from DeepSDFStruct.SDF import SDFfromDeepSDF

from deepshapeopt.config import Config, RunPaths
from deepshapeopt.geometry.frame import DomainFrame
from deepshapeopt.geometry.reconstruction import fit_lattice_to_sdf, init_spline_parameters
from deepshapeopt.hexmesh.design import LatticeDesignSDF

from .base import load_parameter_file

logger = logging.getLogger(__name__)


class DeepSDFLattice:
    """Decoder + lattice of latent control points fitted to the input mesh."""

    def __init__(self, cfg: Config, paths: RunPaths):
        ds = cfg.parametrization.deepsdf
        device = cfg.run.device
        self.cfg = ds
        self.device = device
        self.model = get_model(model=ds.model_path, checkpoint=ds.checkpoint, device=device)
        self.sdf = SDFfromDeepSDF(self.model)
        self.frame = DomainFrame.from_design_domain(cfg.geometry.design_domain, device=device)
        self.mesh_orig = trimesh.load(cfg.geometry.mesh_path)
        self.frame.normalize_mesh(self.mesh_orig).export(paths.reconstruction / "gt_mesh_normalized.stl")

        box_norm = self.frame.box_norm
        mins = box_norm[0].detach().cpu().numpy()
        maxs = box_norm[1].detach().cpu().numpy()
        self.latent_dim = int(self.model._trained_latent_vectors[0].shape[0])
        self.spline_sp = build_parameter_spline(
            spline_degrees=ds.spline_degree, tiling=ds.tiling, latent_dim=self.latent_dim,
            bounds=np.stack([mins, maxs]),
        )
        self.param_spline = SplineParametrization(self.spline_sp, device=device)
        init_spline_parameters(self.param_spline, mean=0.0, std=0.001)
        self.lattice_struct = LatticeSDFStruct(
            tiling=ds.tiling, microtile=self.sdf, parametrization=self.param_spline,
            bounds=box_norm, tiling_map=ds.tiling_map,
        )
        self.extend_bounds = cfg.geometry.flow == "external"

    @property
    def param(self) -> torch.nn.Parameter:
        return next(self.lattice_struct.parametrization.parameters())

    @property
    def design_sdf(self) -> LatticeDesignSDF:
        return LatticeDesignSDF(self.lattice_struct, self.frame)

    @property
    def control_dims(self) -> list[int]:
        return [len(kv) - d - 1 for kv, d in zip(self.spline_sp.knot_vectors, self.spline_sp.degrees)]

    def reconstruct(self, paths: RunPaths, debug: bool) -> None:
        """Fit the lattice to the input mesh, or reuse ``rec_parameters.pt`` of the run."""
        recon = self.cfg.reconstruction
        mesh_norm = self.frame.normalize_mesh(self.mesh_orig)
        box_norm = self.frame.box_norm
        rec_file = paths.reconstruction / "rec_parameters.pt"
        if rec_file.exists() and recon.reuse:
            logger.info("Reusing reconstruction parameters from %s", rec_file)
            params = torch.load(rec_file, map_location=self.device, weights_only=False)
        else:
            logger.info("Running reconstruction")
            saved_bounds = self.lattice_struct.bounds.data
            self.lattice_struct.bounds.data = saved_bounds.float()
            result = fit_lattice_to_sdf(
                self.lattice_struct, mesh_norm, box_norm.float(), recon, self.device,
                output_dir=paths.reconstruction, save_vtp=debug, box_constrained=True,
                samples_series_dir=(paths.heavy_data / "reconstruction" / "rec_sdf_samples_series"
                                    if paths.heavy_data is not None else None),
            )
            params = result["params"]
            self.lattice_struct.bounds.data = saved_bounds
            torch.save(params, rec_file)
        self.lattice_struct.parametrization.set_param(params[0].to(device=self.device, dtype=torch.float32))
        export_reconstructed_artifacts(
            self.lattice_struct, paths.reconstruction, mesh_resolution=int(recon.export_resolution),
            bounds=box_norm, device=self.device, scaling=self.frame.torch_scaling(self.device),
            extend_bounds=self.extend_bounds, export_sdf_grid=debug, export_param_mesh=debug,
        )
        self.lattice_struct.parametrization.set_param(params[0].to(device=self.device))

    def save(self, path: Path) -> None:
        torch.save(list(self.lattice_struct.parametrization.parameters()), path)

    def load(self, path: Path) -> None:
        with torch.no_grad():
            self.param.copy_(load_parameter_file(path, self.param))

    def export_debug(self, out_dir: Path, locked_idx) -> None:
        from deepshapeopt.diagnostics.exports import export_control_net

        export_control_net(self.spline_sp, out_dir, locked_idx)

    def export_iteration(self, out_dir: Path, iteration: int, series_dir: Path | None) -> None:
        return None

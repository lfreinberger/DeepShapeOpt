import numpy as np
import pytest
import torch
import trimesh
from DeepSDFStruct.geom_reconstruction import build_parameter_spline
from DeepSDFStruct.lattice_structure import LatticeSDFStruct
from DeepSDFStruct.parametrization import SplineParametrization
from DeepSDFStruct.SDF import SDFBase

from deepshapeopt.config.schema import ConfigError, DeepSDFConfig, ReconstructionConfig
from deepshapeopt.geometry.reconstruction import fit_lattice_to_sdf
from deepshapeopt.parametrization.symmetry import MirrorSymmetry, SymmetryError

LATENT_DIM = 4
BOX = [[0.0, 1.0, 2.0], [1.0, 3.0, 2.5]]


class ToyTile(SDFBase):
    """Random latent-conditioned MLP, deliberately without any mirror symmetry in u."""

    def __init__(self):
        super().__init__(geometric_dim=3)
        gen = torch.Generator().manual_seed(1)
        self.register_buffer("w1", torch.randn(3 + LATENT_DIM, 16, generator=gen))
        self.register_buffer("b1", torch.randn(16, generator=gen))
        self.register_buffer("w2", torch.randn(16, 1, generator=gen) / 4)
        self.latvec = None

    def _set_param(self, parameters):
        self.latvec = parameters

    def _get_domain_bounds(self):
        return torch.tensor([[-1.0] * 3, [1.0] * 3])

    def _compute(self, queries):
        h = torch.tanh(torch.cat([queries, self.latvec.to(queries.dtype)], dim=1) @ self.w1 + self.b1)
        return h @ self.w2 - 0.1


def make_lattice(tiling, degrees=(1, 1, 1), tiling_map="hat", box=BOX):
    spline_sp = build_parameter_spline(list(degrees), list(tiling), LATENT_DIM, bounds=np.asarray(box))
    param_spline = SplineParametrization(spline_sp)
    lattice = LatticeSDFStruct(tiling=list(tiling), microtile=ToyTile(), parametrization=param_spline,
                               bounds=torch.tensor(box), tiling_map=tiling_map)
    param = next(lattice.parametrization.parameters())
    with torch.no_grad():
        param.copy_(0.1 * torch.randn(param.shape, generator=torch.Generator().manual_seed(2)))
    return spline_sp, lattice, param


@pytest.mark.parametrize("degrees", [(1, 1, 1), (2, 2, 2)])
def test_permutation_is_mirror_involution(degrees):
    spline_sp, _, param = make_lattice((2, 4, 3), degrees)
    sym = MirrorSymmetry.from_spline(spline_sp, ["x", "y"])
    for perm in sym.perms:
        assert sorted(perm.tolist()) == list(range(param.shape[0]))
        assert torch.equal(perm[perm], torch.arange(param.shape[0]))
    c = sym.symmetrize(param.detach())
    assert sym.residual(c) == 0.0
    assert sym.residual(param) > 0.1


@pytest.mark.parametrize("tiling_map", ["hat", "cosine"])
def test_lattice_equivariance_even_tiling(tiling_map):
    spline_sp, lattice, param = make_lattice((3, 4, 2), (2, 2, 2), tiling_map)
    saved = param.detach().clone()
    sym = MirrorSymmetry.from_spline(spline_sp, ["y", "z"])
    assert sym.check_lattice_equivariance(lattice, param) < 1e-5
    assert torch.equal(param.detach(), saved)


@pytest.mark.parametrize("tiling_map", ["hat", "cosine"])
def test_lattice_equivariance_fails_for_odd_tiling(tiling_map):
    spline_sp, lattice, param = make_lattice((2, 3, 2), (1, 1, 1), tiling_map)
    sym = MirrorSymmetry.from_spline(spline_sp, ["y"])
    with pytest.raises(SymmetryError, match="not mirror-equivariant"):
        sym.check_lattice_equivariance(lattice, param)


def test_config_rejects_odd_tile_count():
    raw = {"model_path": "m", "tiling": [4, 9, 12], "symmetry": {"axes": ["y"]}}
    with pytest.raises(ConfigError, match="even tile count"):
        DeepSDFConfig.from_dict(raw)
    raw["symmetry"]["axes"] = ["z"]
    assert DeepSDFConfig.from_dict(raw).symmetry.axes == ["z"]
    with pytest.raises(ConfigError, match="needs at least one axis"):
        DeepSDFConfig.from_dict({"model_path": "m", "tiling": [2, 2, 2],
                                 "symmetry": {"enforce_in_optimization": True}})


def test_index_set_must_be_symmetric():
    spline_sp, _, param = make_lattice((2, 4, 2))
    sym = MirrorSymmetry.from_spline(spline_sp, ["y"])
    n = param.shape[0]
    i = 0
    sym.check_index_set(torch.tensor([i, int(sym.perms[0][i])]), n)
    with pytest.raises(SymmetryError, match="not mirror-symmetric in y"):
        sym.check_index_set(torch.tensor([i]), n)


def _recon_cfg():
    return ReconstructionConfig(n_uniform_samples=512, n_surface_samples=512, num_iterations=6,
                                batch_size=128, lr=0.01, loss_fn="L1")


def test_fit_keeps_codes_symmetric(tmp_path):
    spline_sp, lattice, param = make_lattice((2, 4, 2), (2, 2, 2), "cosine")
    sym = MirrorSymmetry.from_spline(spline_sp, ["y"])
    mesh = trimesh.creation.box(extents=[0.6, 1.2, 0.3])
    mesh.apply_translation([0.5, 2.0, 2.25])
    result = fit_lattice_to_sdf(lattice, mesh, lattice.bounds, _recon_cfg(), "cpu", tmp_path, save_vtp=False,
                                symmetry=sym, symmetry_tolerance=1e-3)
    assert result["symmetry_residual"] == 0.0
    assert sym.residual(param) == 0.0
    assert not param._backward_hooks


def test_fit_rejects_off_centre_target(tmp_path):
    spline_sp, lattice, _ = make_lattice((2, 4, 2))
    sym = MirrorSymmetry.from_spline(spline_sp, ["y"])
    mesh = trimesh.creation.box(extents=[0.6, 1.2, 0.3])
    mesh.apply_translation([0.5, 2.1, 2.25])
    with pytest.raises(SymmetryError, match=r"surface centroid lies 0\.1"):
        fit_lattice_to_sdf(lattice, mesh, lattice.bounds, _recon_cfg(), "cpu", tmp_path, save_vtp=False,
                           symmetry=sym, symmetry_tolerance=1e-3)

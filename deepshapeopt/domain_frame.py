"""Single source of truth for the physical <-> normalized (parameter) frame.

A :class:`DomainFrame` owns the isotropic, centered affine transform that maps a
physical design domain into the DeepSDF decoder's trained coordinate range
``[-1, 1]`` (the longest axis fills it).  Because a signed-distance field is a
Euclidean distance, the *same* scalar ``scale = 2 / L`` (``L`` = longest design-
domain extent) also rescales the SDF *values*; coordinate normalization and
SDF-value normalization are therefore one knob, not two.

Previously that one transform was smeared across several carriers that had to be
kept consistent by hand (``scale``/``center``/``norm_fn``/``denorm_fn``/
``box_norm`` plus a ``TorchScaling`` for mesh generation and a ``PhysicalSDF``
for optimization queries, the latter re-deriving the transform from three of
those fields).  ``DomainFrame`` exposes the transform once and builds those
downstream objects from itself, so they can no longer drift apart.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from deepshapeopt.reconstruction import fit_box_to_unit_cube


@dataclass(frozen=True)
class DomainFrame:
    center: torch.Tensor          # (3,) physical center of the design domain
    scale: torch.Tensor           # scalar 2/L: physical distance -> normalized distance
    design_domain: torch.Tensor   # (2, 3) physical bounds [[min...], [max...]]

    @classmethod
    def from_design_domain(
        cls, design_domain, *, device=None, dtype=torch.float32
    ) -> "DomainFrame":
        """Build a frame from a physical design-domain box (2x3, [[min],[max]])."""
        if torch.is_tensor(design_domain):
            dd = design_domain.to(dtype=dtype)
            if device is not None:
                dd = dd.to(device=device)
        else:
            dd = torch.tensor(design_domain, dtype=dtype, device=device)
        # fit_box_to_unit_cube returns 1/scale (= L/2) as its first element;
        # invert it back to the fundamental phys->norm factor (2/L).
        denorm_scale, center, _norm_fn, _denorm_fn, _box_norm = fit_box_to_unit_cube(dd)
        scale = 1.0 / denorm_scale
        return cls(center=center, scale=scale, design_domain=dd)

    # -- coordinate transforms -------------------------------------------------
    def to_norm(self, points: torch.Tensor) -> torch.Tensor:
        """Physical -> normalized coordinates: ``(x - center) * (2/L)``."""
        return (points - self.center) * self.scale

    def to_phys(self, points: torch.Tensor) -> torch.Tensor:
        """Normalized -> physical coordinates: ``x / (2/L) + center``."""
        return points / self.scale + self.center

    # -- derived quantities ----------------------------------------------------
    @property
    def box_norm(self) -> torch.Tensor:
        """Design-domain bounds in normalized coordinates, shape (2, 3)."""
        return torch.stack(
            [self.to_norm(self.design_domain[0]), self.to_norm(self.design_domain[1])],
            dim=0,
        )

    @property
    def dist_norm_to_phys(self) -> torch.Tensor:
        """Factor converting a normalized distance back to physical units (L/2)."""
        return 1.0 / self.scale

    def normalize_mesh(self, mesh):
        """Return a copy of *mesh* with its vertices mapped into normalized space."""
        verts = torch.as_tensor(
            mesh.vertices, dtype=self.center.dtype, device=self.center.device
        )
        out = mesh.copy()
        out.vertices = self.to_norm(verts).detach().cpu().numpy()
        return out

    # -- builders for downstream consumers ------------------------------------
    def torch_scaling(self, device="cpu"):
        """Build the normalized->physical ``TorchScaling`` used at mesh generation."""
        from DeepSDFStruct.mesh import TorchScaling

        return TorchScaling(
            scale_factors=self.dist_norm_to_phys,
            translation=self.center,
            bounds=self.design_domain,
            device=device,
        )

    def physical_sdf(self, lattice_struct, *, sign=1.0, device="cpu"):
        """Wrap a normalized-space SDF as a physical-coordinate ``PhysicalSDF``."""
        from deepshapeopt.hexmesh.sdf_field import PhysicalSDF

        return PhysicalSDF(
            sdf_norm_fn=lattice_struct,
            norm_fn=self.to_norm,
            design_domain=self.design_domain.detach(),
            dist_scale=float(self.scale),
            sign=sign,
            device=device,
        )

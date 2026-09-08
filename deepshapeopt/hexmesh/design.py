"""Design-geometry adapters for the hex mesh pipeline.

:class:`SdfHexMeshPipeline` needs two things from the design parametrization:
a physical-coordinate SDF whose values carry the autograd graph to the design
parameters (built fresh for every mesh build), and a dtype scope in which the
SDF can be evaluated.  :class:`DesignSDF` names that contract;
:class:`LatticeDesignSDF` implements it for the DeepSDF lattice (the original
behaviour), :class:`~deepshapeopt.hexmesh.ffd_sdf.FFDDesignSDF` for the
free-form-deformation parametrization.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, TypeVar

T = TypeVar("T")


class DesignSDF(Protocol):
    """What the hex mesh pipeline needs from a design parametrization."""

    def make_sdf(self, *, sign: float, device, outer: Any | None) -> Any:
        """Return the design-domain SDF for the current parameters.

        The result must offer the :class:`~deepshapeopt.hexmesh.sdf_field.PhysicalSDF`
        query interface (``phi``, ``phi_ext`` with the parameter graph,
        ``phi_and_grad``, ``phi_ext_np``, ``design_domain``, ``device``).
        ``sign`` is ``-1`` when the fluid is inside the zero level set;
        ``outer`` is the pipeline's fixed outer geometry
        (:class:`~deepshapeopt.hexmesh.trimesh_sdf.TriMeshSDF`) for internal
        flows, ``None`` otherwise.
        """

    def float32_scope(self, fn: Callable[[], T]) -> T:
        """Run ``fn`` in the dtype context required by the SDF evaluation."""


class LatticeDesignSDF:
    """DeepSDF lattice (``LatticeSDFStruct``) wrapped through the domain frame."""

    def __init__(self, lattice_struct, frame):
        self.lattice_struct = lattice_struct
        self.frame = frame

    def make_sdf(self, *, sign: float, device, outer: Any | None = None):
        return self.frame.physical_sdf(self.lattice_struct, sign=sign, device=device)

    def float32_scope(self, fn: Callable[[], T]) -> T:
        from DeepSDFStruct.utils import with_float32_lattice

        return with_float32_lattice(
            self.lattice_struct, self.frame.box_norm, lambda _b: fn()
        )


def as_design_sdf(obj, frame) -> DesignSDF:
    """Accept a :class:`DesignSDF` as is; wrap anything else (a
    ``LatticeSDFStruct`` or a normalized-space SDF callable) as a
    :class:`LatticeDesignSDF` for backward compatibility."""
    if hasattr(obj, "make_sdf") and hasattr(obj, "float32_scope"):
        return obj
    return LatticeDesignSDF(obj, frame)

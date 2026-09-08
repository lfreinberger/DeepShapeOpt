"""Direct SDF-to-hex-mesh pipeline (``mesh_pipeline: "sdf_hex"``).

Generates the body-fitted OpenFOAM polyMesh directly from the differentiable
SDF (octree castellation + differentiable snap), with a fixed mesh outside
the design-space box and direct index-based sensitivity transfer.
"""

from .design import DesignSDF, LatticeDesignSDF
from .ffd_sdf import FFDDesignSDF, FFDMeshSDF
from .pipeline import HexMeshResult, SdfHexMeshPipeline, resolve_sdf_hex_cfg

__all__ = [
    "DesignSDF",
    "FFDDesignSDF",
    "FFDMeshSDF",
    "HexMeshResult",
    "LatticeDesignSDF",
    "SdfHexMeshPipeline",
    "resolve_sdf_hex_cfg",
]

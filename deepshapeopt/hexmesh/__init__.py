"""Direct SDF-to-hex-mesh pipeline (``mesh_pipeline: "sdf_hex"``).

Generates the body-fitted OpenFOAM polyMesh directly from the differentiable
SDF (octree castellation + differentiable snap), with a fixed mesh outside
the design-space box and direct index-based sensitivity transfer.
"""

from .pipeline import HexMeshResult, SdfHexMeshPipeline, resolve_sdf_hex_cfg

__all__ = ["HexMeshResult", "SdfHexMeshPipeline", "resolve_sdf_hex_cfg"]

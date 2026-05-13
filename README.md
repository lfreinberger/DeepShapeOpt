# DeepShapeOpt

DeepShapeOpt is a research codebase for neural implicit geometry representation
and gradient-based shape optimization.

The repository contains the reusable method implementation and public examples
used to reproduce the results from *Shape Optimization Using a Neural Implicit
Geometry Representation Framework*. It is also intended as the future development
home for the DeepShapeOpt method.

## Dependencies

DeepShapeOpt uses `DeepSDFStruct` as a separate dependency. The dependency is
currently pinned to commit `0fcd508`; tag that commit as `v1.0.0-paper` before
the final archived release.

The intended citable releases are:

- `DeepShapeOpt v1.0.0-paper`
- `DeepSDFStruct v1.0.0-paper`

After creating GitHub releases, archive both releases with Zenodo and replace the
placeholder citation metadata with the Zenodo version DOIs.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For full optimization runs, install and source a compatible OpenFOAM environment
before running the optimization scripts.

Set the data/model/result locations:

```bash
export DEEPSHAPEOPT_DATA_DIR=/path/to/deepshapeopt-data
export DEEPSHAPEOPT_MODEL_DIR=$DEEPSHAPEOPT_DATA_DIR/models
export DEEPSHAPEOPT_RESULTS_DIR=/path/to/deepshapeopt-results
```

The data package is distributed separately through the TU Wien cloud. See
`data_manifest.json` for the expected layout and checksums.

## Reproduce The Paper Pipeline

The pretrained primitive decoder used by the examples is available in the data
package under:

```text
DeepSDFStruct/trained_models/primitives_cl32
```

Place or symlink that directory at:

```text
${DEEPSHAPEOPT_MODEL_DIR}/primitives_cl32
```

Alternatively, train the primitive decoder from the provided training data:

```bash
python scripts/train_latent_field.py \
  --config experiments/training/primitives_cl32/specs.json \
  --output "${DEEPSHAPEOPT_MODEL_DIR}/primitives_cl32"
```

Reconstruct the flow-channel geometry:

```bash
python experiments/reconstruction/feed_channel/reconstruct_feed_channel.py
```

Run the cube optimization with the neural SDF parameterization:

```bash
python experiments/optimization/drag_optimization_cube/optimization_cube.py \
  --config experiments/optimization/drag_optimization_cube/config_cube_with_holes.json
```

Run the cube optimization with FFD:

```bash
python experiments/optimization/drag_optimization_cube/optimization_cube_ffd.py \
  --config experiments/optimization/drag_optimization_cube/config_ffd_cube_with_cylinders_7x7x7.json
```

## Release Checklist

Before tagged releases, maintainers should confirm that all included datasets and
meshes have publication clearance and that `data_manifest.json` contains final
URLs, checksums, and license information.

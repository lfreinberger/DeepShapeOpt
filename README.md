# DeepShapeOpt

DeepShapeOpt is a research codebase for neural implicit geometry representation
and gradient-based shape optimization. The repository contains the reusable
method implementation and examples used to reproduce the results from
*Shape Optimization Using a Neural Implicit Geometry Representation Framework*.

## Highlights

- Neural implicit geometry (DeepSDF) parameterization with B-spline lattices.
- Free-form deformation (FFD) optimization baseline.
- Reconstruction pipelines with configurable sampling, tiling, and loss settings.
- OpenFOAM-driven shape optimization with adjoint sensitivities.
- Reproducible, config-first experiments and standardized output structure.

## Dependencies

- Python >= 3.11.
- Core deps are defined in pyproject.toml, including `DeepSDFStruct` and
  scientific tooling (torchmesh, gmsh, foamlib, mlflow, etc.).
- A compatible OpenFOAM environment is required for optimization experiments.

DeepShapeOpt uses `DeepSDFStruct` as a separate dependency. The dependency is
currently pinned to commit `0fcd508`; tag that commit as `v1.0.0-paper` before
the final archived release.

The intended citable releases are:

- DeepShapeOpt v1.0.0-paper
- DeepSDFStruct v1.0.0-paper

After creating GitHub releases, archive both releases with Zenodo and replace the
placeholder citation metadata with the Zenodo version DOIs.

## Setup

Create a Python environment and install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For full optimization runs, install and source a compatible OpenFOAM
environment before running the optimization scripts.

### Data and Results Locations

The data package is distributed separately through the TU Wien cloud. See
data_manifest.json for the expected layout and checksums.

Set the data/model/result locations so experiment configs resolve correctly:

```bash
export DEEPSHAPEOPT_DATA_DIR=/path/to/deepshapeopt-data
export DEEPSHAPEOPT_MODEL_DIR=$DEEPSHAPEOPT_DATA_DIR/models
export DEEPSHAPEOPT_RESULTS_DIR=/path/to/deepshapeopt-results
```

Experiment configs expand environment variables at load time, so paths like
${DEEPSHAPEOPT_MODEL_DIR}/primitives_cl32 work directly in JSON files.

## Quickstart

Make sure the pretrained primitive decoder is available in the data package
under:

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

Run a reconstruction example:

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

## Experiments Overview

Reconstruction experiments live under experiments/reconstruction and are driven
by JSON configs (mesh path, spline tiling, sampling counts, and training
iterations). Example entry points:

- feed_channel/reconstruct_feed_channel.py
- primitive_shape/reconstruct_primitive_shape.py
- rim/reconstruct_rim.py
- shiba/reconstruct_shiba.py
- ship_propeller/reconstruct_ship_propeller.py

Optimization experiments live under experiments/optimization and wrap
reconstruction, CFD runs, sensitivity extraction, and MMA optimization. The
configs include reconstruction parameters, optimization hyperparameters, and
OpenFOAM sensitivity extraction settings.

## Output Structure

Each experiment writes results under its local results directory (e.g.
experiments/optimization/drag_optimization_cube/results_cube/). If
DEEPSHAPEOPT_RESULTS_DIR is set, heavy data (VTK series, STL series, control
lattices) is mirrored to that external location while a lightweight summary
stays alongside the experiment.

The common folder structure is:

```text
results*/
  reconstruction/
  optimization/
```

## Reproducibility Notes

- The paper workflow includes: (1) primitive decoder training, (2) flow-channel
  reconstruction, (3) cube drag optimization with the neural SDF
  parameterization, (4) cube drag optimization with FFD.

## Citation

CITATION.cff will be updated with Zenodo DOIs after the v1.0.0-paper releases
are archived. Please cite both DeepShapeOpt and DeepSDFStruct when referencing
the method and experiments.

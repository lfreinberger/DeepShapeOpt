# DeepShapeOpt

[![DOI](https://zenodo.org/badge/1237823553.svg)](https://doi.org/10.5281/zenodo.20210464)

DeepShapeOpt is a research codebase for shape optimization with neural implicit geometry representations and OpenFOAM-based adjoint sensitivities. The public repository focuses on drag optimization of cube-like benchmark geometries.

## Reproducing the paper

<!-- Replace this paragraph with the full paper citation once it is published. -->
The results reported in the companion paper (citation TBA) were produced with the following archived artefacts:

| Artefact | DOI |
| --- | --- |
| DeepShapeOpt source (this repository) | [10.5281/zenodo.20210464](https://doi.org/10.5281/zenodo.20210464) |
| DeepSDFStruct (editable dependency, fork) | [10.5281/zenodo.20205817](https://doi.org/10.5281/zenodo.20205817) |
| Training data for the neural decoder | [10.48436/12y18-j6236](https://doi.org/10.48436/12y18-j6236) |

To reproduce the optimization results, the training data is **not required** — the pretrained decoder checkpoints are shipped inside the archived DeepSDFStruct release under `trained_models/`. The training data is only needed to retrain the decoder from scratch.

## Installation

The project uses Python 3.11+ and `uv`.

```bash
uv sync
```

`DeepSDFStruct` is configured as an editable local dependency in `pyproject.toml`. Make sure the path in `[tool.uv.sources]` matches your checkout.

## Environment

Set these variables before running an optimization:

```bash
export DEEPSHAPEOPT_DATA_DIR=/path/to/deepshapeopt-data
export DEEPSHAPEOPT_MODEL_DIR=/path/to/DeepSDFStruct/trained_models
export DEEPSHAPEOPT_RESULTS_DIR=/path/to/deepshapeopt-results
```

For local development, `.vscode/launch.json` contains matching debug configurations.

## Repository layout

- `scripts/` — pipeline entry points (`reconstruct.py`, `optimize_drag_latent.py`, `optimize_drag_ffd.py`); each takes a `--config` JSON.
- `deepshapeopt/` — library package: config loading, lattice/spline parametrization, SDF reconstruction, the outer optimization loop, OpenFOAM case handling, meshing, and logging/plotting utilities.
- `experiments/drag_cube/` — drag-optimization configs (`config_latent_*.json`, `config_ffd_*.json`) and the OpenFOAM `foam_case/` template that the scripts copy per run.
- `experiments/reconstruction/{rim,shiba,ship_propeller,feed_channel}/` — public reconstruction cases, one `config.json` each.
- `data/shapes/` — input STL meshes referenced by the configs via `${DEEPSHAPEOPT_DATA_DIR}`.

## Workflows

### Shape Reconstruction

The reconstruction workflow fits a DeepSDF microtile lattice to an input STL without running any flow solver.

```bash
uv run python scripts/reconstruct.py \
  --config experiments/reconstruction/rim/config.json
```

Other public reconstruction configs:

```text
experiments/reconstruction/shiba/config.json
experiments/reconstruction/ship_propeller/config.json
experiments/reconstruction/feed_channel/config.json
```

### Latent Drag Optimization

The latent workflow reconstructs the shape with a DeepSDF microtile and optimizes B-spline latent parameters.

```bash
uv run python scripts/optimize_drag_latent.py \
  --config experiments/drag_cube/config_latent_cube.json
```

Other public latent configs:

```text
experiments/drag_cube/config_latent_cube_with_cylinders.json
experiments/drag_cube/config_latent_cube_with_holes.json
```

### FFD Drag Optimization

The FFD workflow optimizes a free-form deformation of the input mesh directly.

```bash
uv run python scripts/optimize_drag_ffd.py \
  --config experiments/drag_cube/config_ffd_cube_with_cylinders_7x7x7.json
```

The smaller FFD lattice config is available at:

```text
experiments/drag_cube/config_ffd_cube_with_cylinders_5x5x5.json
```

## Outputs

Each run writes lightweight results next to the experiment config, for example:

```text
experiments/drag_cube/results_cube/
  reconstruction/
  optimization/
```

The optimization folder contains the current shape, optimized parameters, CSV/text history, residual plots, convergence diagnostics, and optimization history plots.

Large debug exports are written only when debug mode is enabled. Enable debug mode with either:

```json
{
  "debug": true
}
```

or:

```json
{
  "optimization": {
    "debug": true
  }
}
```

When `heavy_data_output_path` is set, debug exports are mirrored under `DEEPSHAPEOPT_RESULTS_DIR` while preserving the experiment path layout.

## Notes

- Iteration numbering starts at `0` in logs, files, and plots.
- Normal console output is intentionally concise: iteration, objective, constraints, MMA change, and timing.
- Detailed sensitivity-transfer diagnostics and heavy visualization series are debug-only.
- OpenFOAM residual plots are generated whenever solver logs are available.

## Citation

If you use this code in academic work, please cite the archived release:

- DeepShapeOpt v0.1.0, Zenodo, <https://doi.org/10.5281/zenodo.20210464>

<!-- Add the paper citation here once it is published. -->

A machine-readable citation is provided in `CITATION.cff`.

## Data sources

The reconstruction test cases use the following geometries from Thingiverse:

- *My little shiba* by layerone, <https://www.thingiverse.com/thing:1308678>.
- *Parametric Ship Propeller / Rc propeller / Marine Propeller* by Fysik_klubben, <https://www.thingiverse.com/thing:6906448>.
- *RC Rim* by Attila_d, <https://www.thingiverse.com/thing:1328760>.

## License

See `LICENSE`.

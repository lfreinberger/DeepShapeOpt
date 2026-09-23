# DeepShapeOpt

[![DOI](https://zenodo.org/badge/1237823553.svg)](https://doi.org/10.5281/zenodo.20210464)

Shape optimization with neural implicit geometry and adjoint CFD sensitivities. A design is a
DeepSDF latent lattice (or a free-form deformation) inside a design box; every iteration the
body-fitted hex mesh is generated directly from the signed distance field, OpenFOAM's continuous
adjoint or DAFoam's discrete adjoint returns the wall sensitivities, and the method of moving
asymptotes (MMA) takes a step under CFD and geometric constraints.

Installation and machine setup: [INSTALL.md](INSTALL.md).

## One driver, one config

```sh
uv run deepshapeopt optimize --config experiments/drag_cube/config_latent_cube.json
```

The config sections mirror the building blocks of a run:

| Section | Block | Options |
|---|---|---|
| `run` | name, device, iterations, debug, heavy-data and scratch directories | |
| `geometry` | input mesh, length unit, design domain, flow type | `internal` / `external` |
| `parametrization` | design variables | `deepsdf` (latent lattice, optional PCA reduction), `ffd`; locked control points |
| `mesh` | the `sdf_hex` mesher: octree castellation and differentiable snap onto the SDF | caps, patches, refinement |
| `solver` | forward solver and adjoint | `openfoam` (continuous, ESI), `dafoam` (discrete) |
| `objective` | CFD metric plus penalties | `drag`, `uniformity`, `uniformity_directional`, `losses`; proximity, lattice smoothness |
| `constraints` | MMA rows with budgets | CFD metric, volume, centroid, FFD fold-over guard, undercut, minimum steg length |
| `optimizer` | MMA settings | move limit, bounds, GCMMA inner loop, feasibility restoration, step control, convergence stop |
| `diagnostics` | run mode and exports | `optimize`, `noise_probe`, `jacobian_probe`; VTK / STL series |

Older configs (two blocks `reconstruction` / `optimization`) convert with
`uv run deepshapeopt migrate-config --write <config>`.

## Repository layout

```
deepshapeopt/
  driver.py, problem.py, cli.py     the optimization loop, its assembly from a config, the command line
  config/                           schema v2, loader, run paths, v1 migration
  geometry/                         domain frame, lattice reconstruction, reconstruction analysis
  parametrization/                  DeepSDF lattice, PCA basis, FFD, locked control points, design space
  hexmesh/                          sdf_hex: octree, snap, polyMesh writer, caps and outlet sub-patches
  mesher.py                         MeshResult and the mesher wrapper the loop uses
  solvers/                          metric registry; openfoam/ (runtime, sensitivities, export); dafoam/ (runner, in-container script)
  terms/                            objective, penalties and constraint rows (CFD, volume, centroid, FFD jacobian, undercut, steg length)
  optimizer/                        MMA wrapper (DeepSDFStruct.optimization.MMA), step control, convergence stop
  diagnostics/                      history CSV, plots, exports, probes, path consistency, noise estimate, latent metric
  latent_gui/                       web editor for the latent codes of a saved design
openfoam/                           OpenFOAM user libraries (directional uniformity objective, powerLaw-Arrhenius material model)
external/dafoam_patches/            DAFoam patch (fvMatrix preconditioner) and its build notes
experiments/                        drag_cube/ (external flow), channel/ (internal flow), reconstruction/ (rim, shiba, propeller, channel)
tests/                              pytest (CPU only) and tests/smoke/ end-to-end configs
scripts/                            optimize.py, reconstruct.py, latent_gui.py (thin wrappers around the CLI), publish_dafoam_build.sh
slurm/                              generic SLURM template, config as argument
```

## Experiments

- `experiments/drag_cube/`: drag of a cube-like body (DeepSDF lattice with and without PCA, FFD,
  OpenFOAM and DAFoam), the case template `foam_case/` and `dafoam_case/`.
- `experiments/channel/`: internal flow through a branching channel, outlet velocity uniformity
  under a pressure-loss constraint (DeepSDF lattice and FFD).
- `experiments/reconstruction/`: DeepSDF reconstructions of public geometries (rim, shiba, ship
  propeller from Thingiverse, and the channel).

Each run writes `<experiment>/<run.name>/{reconstruction,optimization}/` with `current_shape.stl`,
`parameters.pt`, `optimization_history.csv`, plots and `run.log`; with `run.debug` the VTK / STL
series go to `run.heavy_data_dir`. Iteration numbering starts at 0.

## Paper

The results of the companion paper (citation to follow) were produced with the archived
snapshots DeepShapeOpt 10.5281/zenodo.20210465 and DeepSDFStruct 10.5281/zenodo.20210456
(concept DOIs 10.5281/zenodo.20210464 and 10.5281/zenodo.20205817). Those snapshots contain the
snappyHexMesh path used for the paper runs, which this version replaced by the SDF hex mesher.
Training data for the decoder: 10.48436/12y18-j6236 (only needed to retrain).

## Citation

Cite the archived release (DeepShapeOpt v0.1.0, <https://doi.org/10.5281/zenodo.20210464>); a
machine-readable citation is in `CITATION.cff`.

Reconstruction geometries: *My little shiba* by layerone (thing:1308678), *Parametric Ship
Propeller* by Fysik_klubben (thing:6906448), *RC Rim* by Attila_d (thing:1328760), all Thingiverse.

## License

See `LICENSE`.

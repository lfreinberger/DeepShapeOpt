# Setting up DeepShapeOpt on a new machine

DeepShapeOpt is the library and driver; the application repositories (private) hold the
experiments and depend on it as an editable path dependency. Clone them side by side:

```
ProjectsPhD/
  DeepSDFStruct/      neural implicit geometry (fork), pretrained decoders in DeepSDFStruct/trained_models
  DeepShapeOpt/       this repository
  <application>/     private experiment repositories (configs, case templates, geometries)
```

## 1. Prerequisites

| Component | Version | Used for |
|---|---|---|
| Linux, gcc | | building the OpenFOAM user libraries |
| [uv](https://docs.astral.sh/uv/) | >= 0.9 | Python environments (`uv sync`, `uv run`) |
| Python | 3.11 (pinned in `.python-version`) | |
| CUDA-capable GPU | optional | the DeepSDF lattice runs on `run.device: "cuda"`; `"cpu"` works and is slower |
| OpenFOAM | v2506 (ESI) | continuous adjoint (`adjointOptimisationFoam`) and the case tools |
| Apptainer | | DAFoam discrete adjoint (image `dafoam/opt-packages`, DAFoam v5.1.1 on OpenFOAM-v2506) |

## 2. Python environments

```sh
git clone git@github.com:lfreinberger/DeepSDFStruct.git
git clone git@github.com:lfreinberger/DeepShapeOpt.git
cd DeepShapeOpt && uv sync            # installs DeepSDFStruct from ../DeepSDFStruct (editable)
cd ../<application> && uv sync        # installs DeepShapeOpt from ../DeepShapeOpt (editable)
```

`pyproject.toml` of every repository names the sibling path under `[tool.uv.sources]`; keep the
directory layout above or edit those paths.

## 3. Environment variables

Configs never contain machine paths; they expand these variables at load time:

| Variable | Meaning |
|---|---|
| `DEEPSHAPEOPT_DATA_DIR` | input geometries (`<repo>/data`, relative to the repository you run from) |
| `DEEPSHAPEOPT_MODEL_DIR` | DeepSDF checkpoints, normally `DeepSDFStruct/DeepSDFStruct/trained_models` |
| `DEEPSHAPEOPT_RESULTS_DIR` | heavy debug exports (VTK / STL series), a large disk |
| `DEEPSHAPEOPT_SCRATCH_DIR` | transient solver cases, a fast node-local disk |
| `DAFOAM_SIF` | the DAFoam Apptainer image |
| `DAFOAM_BUILD_ROOT` | published patched DAFoam build (`pc_mode: fvmatrix`), see section 6 |

Copy `env.example.sh` (bash) or `env.example.csh` (tcsh) to `env.sh` / `env.csh` in each repository,
adjust the paths and source the file before running. The SLURM template and the VS Code launch
configurations read `env.sh`.

## 4. OpenFOAM user libraries

The internal-flow objective and the shear-thinning material model are user libraries under
`openfoam/` (see `openfoam/README.md`). Build them once per OpenFOAM installation:

```sh
source /path/to/OpenFOAM-v2506/etc/bashrc
openfoam/Allwmake                       # -> $FOAM_USER_LIBBIN
```

## 5. DeepSDF checkpoints

The decoders (`primitives_cl32` and others) ship inside the archived DeepSDFStruct release
(`trained_models/`, Zenodo DOI 10.5281/zenodo.20205817) and in the fork. Training data is only
needed to retrain a decoder (DOI 10.48436/12y18-j6236).

## 6. DAFoam (discrete adjoint, optional)

```sh
apptainer pull $DAFOAM_SIF docker://dafoam/opt-packages:latest
```

The stock image runs with `solver.dafoam.pc_mode: "coloring"`. The default `"fvmatrix"` mode
needs the patched build described in `external/dafoam_patches/README.md`: apply the patch
inside the container, build on local scratch, then publish the two run-time pieces to shared
storage with `scripts/publish_dafoam_build.sh` and point `DAFOAM_BUILD_ROOT` at the result.
A config asking for `fvmatrix` without a published build fails at load time with instructions.

## 7. Verify

```sh
uv run pytest tests -q                  # unit tests, CPU only, no solver (about 6 minutes)
source env.sh
tests/smoke/run_all.sh drag_deepsdf_openfoam        # one end-to-end run, 2 iterations, OpenFOAM + GPU
tests/smoke/run_all.sh                              # all smoke cases (OpenFOAM, DAFoam, GPU)
```

If `deepshapeopt` fails with `ModuleNotFoundError` (for `deepshapeopt` itself or for a
dependency such as `torchfem.materials`) although `uv sync` reports no changes, the environment
was installed with `UV_LINK_MODE=symlink`: every file in `.venv` is then a symlink into
`~/.cache/uv/archive-v0`, and `uv cache clean` / `uv cache prune` leaves the whole environment
dangling (`find .venv -xtype l | wc -l` shows the count). Repair with

```sh
uv sync --reinstall          # in every affected repository
```

and prefer the default link mode (hardlinks; `unset UV_LINK_MODE` or `UV_LINK_MODE=copy`) so a
cache clean cannot break the environments again. Symlink mode only saves the disk space of one
copy per package.

## 8. Running

```sh
uv run deepshapeopt optimize --config experiments/drag_cube/config_latent_cube.json
uv run deepshapeopt reconstruct --config experiments/reconstruction/rim/config.json
uv run deepshapeopt latent-gui --config experiments/reconstruction/rim/config.json
uv run deepshapeopt migrate-config --write <old config>.json     # v1 -> v2 configs
```

`scripts/optimize.py`, `scripts/reconstruct.py` and `scripts/latent_gui.py` are the same entry
points for the VS Code play button (`.vscode/launch.json` has one entry per experiment).

Cluster: submit from the repository root of the experiment,

```sh
sbatch [-J name] ../DeepShapeOpt/slurm/optimize.slrm experiments/<case>/config.json
```

`--cpus-per-task` must match `numberOfSubdomains` of the case (`system/decomposeParDict*`) or
`solver.dafoam.n_procs`; on a CPU partition set `run.device: "cpu"` in the config.

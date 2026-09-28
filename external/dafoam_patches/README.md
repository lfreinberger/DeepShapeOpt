# DAFoam patches

| patch | what |
|---|---|
| `0001-fvmatrix-preconditioner-for-DASimpleFoam.patch` | adjoint preconditioner from the fvMatrix coefficients (below) |
| `0002-DASimpleHeatTransferFoam.patch` | new solver: generalized-Newtonian melt with energy equation, fully coupled adjoint (section at the end) |

Both apply in order to the same base. `build_dafoam.sh` builds plain, ADR and ADF;
`scripts/publish_dafoam_build.sh` publishes all three.

# 0001: fvMatrix preconditioner for `DASimpleFoam`

Builds the adjoint preconditioner directly from OpenFOAM's assembled fvMatrix
coefficients instead of the dRdW graph colouring. That removes the colouring and the one
AD sweep per colour, which together are ~95 % of a stock evaluation.

Measured on the drag cube (51405 cells, 16 procs): a full evaluation goes from **280 s to
27.5 s**. GMRES needs 235 instead of 50 iterations, but the preconditioner assembly drops
from ~260 s to **0.05 s**. The gradient is unchanged — relative 7.9e-8 against the
colouring path, cosine 1.000000000000, J bit-identical. Cross-checked on the cylinder
(internal flow, two objectives): gradients agree to 4.5e-9 and 2.0e-9.

## Base

`0001-fvmatrix-preconditioner-for-DASimpleFoam.patch` applies to DAFoam v5.1.1 at commit
`a941594dc3d0a85b885074f8aabcf2806ad4604d` ("Added the line search for findFeasible
util"), the revision shipped in `/usr2/lfrei/containers/dafoam.sif`.

## What it changes

| file | change |
|---|---|
| `DAResidual/DAResidualSimpleFoam.{H,C}` | new `calcPCMatWithFvMatrix`. Only the unsteady solvers had one, so `DASimpleFoam` hit the base class's `FatalErrorIn("Child class not implemented!")`. Mirrors `calcResiduals` one-to-one (same `div(pc)`, same normalisations) and adds the `dR_phi/dphi` diagonal, which the Pimple template omits. |
| `DASolver/DASolver.{H,C}` | new `initializePCMatFvMatrix`: sizes and preallocates the block-diagonal matrix **without** the colouring. Stock DAFoam always creates this matrix through `calcdRdWT`, i.e. through the colouring, and only refreshes its values with `calcPCMatWithFvMatrix`. Off-processor preallocation is zero because the assembly only ever touches internal faces. |
| `DAModel/DATurbulenceModel/DASpalartAllmaras.C` | drop `nuTildaEqn.relax()` before reading the coefficients (see below). |
| `pyDASolvers/DASolvers.H`, `pyDASolvers.pyx` | expose `initializePCMatFvMatrix` to Python. |

## Two corrections that took the iteration count from 864 to 235

Both were found by comparing the assembled matrix block by block against the AD-built one.

**phi diagonal was off by a factor 642.** `phiRes` is divided by `magSf` when normalised,
but `DAPartDeriv.C` *also* overrides `normalizeStates["phi"]` with `magSf` for the state
scaling. The two cancel, so the normalised diagonal is exactly -1.

**U and nuTilda diagonals were ~40 % too large.** The AD differentiates through the
under-relaxation: `relax(alpha)` scales the diagonal by `1/alpha` but puts
`(1-alpha)/alpha*A*U` back into the source, and that `U` is the state itself, so the two
cancel. The exact block is the **unrelaxed** matrix. `DAResidualPimpleFoam` already gets
this right with an explicit `relax(1.0)`; the SA turbulence model does not. The relaxed
equation is still used for `rAU` and the pressure block — the p block matches the AD
matrix to 1e-10 that way.

Afterwards the deviation from the AD matrix is U 2.9 %, p 0.13 %, nuTilda 0.096 %,
phi 0.005 %.

## Build

The container is read-only, so the sources are bind-mounted from the host and the rebuilt
library is captured in an Apptainer overlay. This needs no root and no new image, and
dropping the `--overlay`/`--bind` flags gives stock DAFoam back.

```bash
B=/work/lfrei/dafoam-build          # $B/dafoam = sources, $B/overlay = writable layer
apptainer exec --cleanenv --overlay $B/overlay \
  --bind /work,/usr2,/workdisk,$B/dafoam:/home/dafoamuser/dafoam/repos/dafoam \
  /usr2/lfrei/containers/dafoam.sif bash -c '
    source /home/dafoamuser/dafoam/loadDAFoam.sh >/dev/null 2>&1
    cd /home/dafoamuser/dafoam/repos/dafoam/src/adjoint && WM_QUIET=true wmake -j 8'
```

A change-and-rebuild cycle is ~21 s. After touching `pyDASolvers.pyx`, also run
`src/pyDASolvers/Allmake` and copy `src/pyDASolvers/pyDASolvers.so` over
`dafoam/libs/pyDASolvers.cpython-312-x86_64-linux-gnu.so`; `PYTHONPATH` then has to point
at the repo so `import dafoam` resolves to the patched package. `dafoam_utils` sets that
up automatically when `build_source` is given.

0001 alone would only need the **plain** build (neither new function is virtual and both are
only ever called on `DASolver.solver`). 0002 adds a solver that pyDAFoam instantiates in every
mode, so all three are rebuilt now: `build_dafoam.sh [build-tree] [plain ADR ADF]` (~3 min per
mode; the environment scripts return non-zero, so `set -e` only comes after sourcing them).

## Publish, then use

Build on local scratch (fast), then publish the two things a *run* needs onto shared
storage — otherwise a slurm job fails, because node-local scratch is invisible from the
compute nodes:

```bash
DeepShapeOpt/scripts/publish_dafoam_build.sh    # -> $DAFOAM_BUILD_ROOT
```

That writes `<root>/sharedLibs/` (the image's shared libs with our patched
`libDASolver.so`) and `<root>/dafoam/` (the patched python package), ~90 MB in total. At
run time both are bind-mounted over the stock install and `PYTHONPATH` points at the root,
so `import dafoam` resolves to the patched package. No overlay is involved — fuse-overlayfs
on NFS is fragile, and the build has to be on NFS for the cluster.

Configs only carry the switch; the location comes from `$DAFOAM_BUILD_ROOT`
(set in `env.sh`), so they stay portable:

```json
"dafoam": { "pc_mode": "fvmatrix" }
```

`build_root` in the config overrides the variable. If neither resolves to a published
build, the config load fails immediately — before the reconstruction and meshing — with a
message naming the two ways out (publish, or fall back to `pc_mode: "coloring"`). There is
deliberately no silent fallback: it would make a run ~10x slower without anyone noticing.

## Open

- The U diagonal still deviates by 2.9 %, most likely because `fvMatrix::D()` uses the
  component average of the boundary internal coefficients rather than the per-component
  value.
- `DASpalartAllmaras.C` is upstream code, not an addition. The change is right for the
  preconditioner but affects every other caller of `getFvMatrixFields`.
- Raising `pcFillLevel` does nothing here (864 -> 862 -> 859 at levels 1/2/3): the missing
  coupling is absent from the matrix, and no factorisation can invent it. Only better
  values help, which is what the two corrections above did.

# 0002: `DASimpleHeatTransferFoam`

The discrete counterpart of the `simpleHeatTransfer` primal (`openfoam/primalSolvers/`):
SIMPLE for U and p as in `DASimpleFoam`, plus

    div(phi,T) - laplacian(DT,T) == (1/cp)*(tau && grad(U)),   tau = nu*(grad(U) + grad(U)^T)

with `DT` and `cp` from `constant/transportProperties`, and `transportModel powerLawArrhenius`
(or `Newtonian`). DAFoam's own T equation in `DASimpleFoam` is passive (no dissipation, nu
never depends on T) and would not do.

| file | role |
|---|---|
| `DASolver/DASimpleHeatTransferFoam/` | derives from `DASimpleFoam`; reads T **before** the transport model (the viscosity model looks T up in its constructor), `DT`, `cp`; primal loop U -> p -> T -> `laminarTransport.correct()`. The fixed-point adjoint is refused. |
| `DAResidual/DAResidualSimpleHeatTransferFoam.{H,C}` | U/p/phi residuals of `DASimpleFoam`, plus TRes of the energy equation. `updateIntermediateVariables` overwrites the registered `nu` with `powerLawArrhenius::calcNu(coeffs, U, T)`, **unrelaxed**. DASolver calls it before every residual evaluation, also inside the AD tape, so dR/dW carries dnu/dU and dnu/dT and dR/dXv the grad(U) of nu. `calcPCMatWithFvMatrix` is refused (no T block yet). |
| `DAStateInfo/DAStateInfoSimpleHeatTransferFoam.{H,C}` | states p, T, U, phi, nut; connectivity widened for nu(grad(U), T). It only shapes the coloring preconditioner. |
| `Make/files`, `Make/options` | the three classes and `transportModels/powerLawArrhenius/powerLawArrhenius.C`, a symlink to `openfoam/viscosityModels/powerLawArrhenius` that `build_dafoam.sh` creates. The model is compiled into `libDASolver$(WM_AD_MODE)`, so each mode has its own AD-typed copy and no `libs` entry is needed. |
| `dafoam/pyDAFoam.py` | `DASimpleHeatTransferFoam` in `solverRegistry["Incompressible"]`. |

Why the residual cannot just call `laminarTransport.correct()`: the model under-relaxes
(`viscosityRelaxation`), which would scale dnu/dW by the relaxation factor, and
`singlePhaseTransportModel` gives no access to the model. Hence the static `calcNu`: one formula
for the ESI library and the residual.

Laminar flow is still modelled as Spalart-Allmaras with nuTilda = 0 (DASimpleFoam needs a
registered `nut`); `Pr` and `Prt` stay in `transportProperties` because `DATurbulenceModel` and
the `DASimpleFoam` residual read them. They have no effect.

Unlike the ESI continuous adjoint (`adjointLaminarPowerLaw`, T frozen), this gives the gradient of
the fully coupled U-p-T-nu system. `scripts/check_dafoam_gradient.py` is its FD gate.

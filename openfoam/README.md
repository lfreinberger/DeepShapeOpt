# OpenFOAM user libraries

Custom libraries for the adjoint shape optimization, built into `$FOAM_USER_LIBBIN` and loaded
through `libs (...)` in a case's `controlDict`. The shared OpenFOAM installation is never modified.

```sh
source /programs/shared/openfoam/OpenFOAM-v2506/etc/bashrc   # or your OpenFOAM-v2506 (ESI) install
./Allwmake                                                   # from this directory
```

| Directory | Library | Provides | Selected via |
|---|---|---|---|
| `adjointObjectives/objectiveUniformityPatchDirectional` | `libcustomAdjointObjectives` | objective `uniformityPatchDirectional`: velocity uniform along `uniformDirection` on `parallelPatches`, transverse components driven to zero on `transversePatches` (`transverseWeight`) | `system/optimisationDict` (`objectiveNames`) |
| `viscosityModels/powerLawArrhenius` | `libpowerLawArrhenius` | `transportModel powerLawArrhenius`: `nu = clamp(k * gammaDot^(n-1) * exp(E/(R T)), nuMin, nuMax)`, looks up the registry field `T` | `constant/transportProperties` |
| `primalSolvers/simpleHeatTransfer` | `libcustomPrimalSolvers` | primal solver `simpleHeatTransfer`: SIMPLE plus the energy equation `div(phi,T) - laplacian(DT,T) = (1/cp) (tau : grad U)`; registers `T` before the transport model | `system/optimisationDict` (`primalSolvers.p1.solver`) |
| `adjointRASModels/adjointLaminarPowerLaw` | `libcustomAdjointRAS` | adjoint model `adjointLaminarPowerLaw`: frozen-viscosity adjoint plus the shear-rate linearization `dnu/dgammaDot` (`adjointMeanFlowSource`) and the direct grid-metric term (`FISensitivityTerm`); temperature stays frozen in the adjoint | `constant/adjointRASProperties` |

The directional uniformity objective is the metric `uniformity_directional` of the library; every
internal-flow case template loads it. The three material-model libraries make up the
shear-thinning, temperature-dependent melt rheology ("Level 1" adjoint: frozen viscosity plus
the shear-rate coupling, complete with both linearization routes). The Newtonian control of the
2-D gate reproduces the exact gradient to 0.03 %; Level 1 lands within 1.5 % at production-like
Arrhenius coupling while the frozen-viscosity adjoint is 30 % off. The coefficients of a
material (`k`, `n`, `nuMin`, `nuMax`, `Eactive`, `DT`, `cp`) live in the case's
`transportProperties`, not here. The 2-D bump-case gate suite that measured these numbers
stays with the application repository that owns the material data.

Notes for the models:

- `adjointLaminarPowerLaw` needs about 1.2 to 1.8 times the adjoint iterations of
  `adjointLaminar` for the same residual level (geometry dependent).
- The powerLaw primal converges only algebraically under the Arrhenius feedback; use fixed
  iteration counts (`solver.openfoam.solver_convergence.mode: fixed`).
- `FISensitivityTranspose` is not resolved by the 2-D gate (shear-dominated flow); it may matter
  in strongly three-dimensional flows.

# Revision results — DeepShapeOpt manuscript

Everything needed to write the response letter and update the manuscript is in this
folder. **No code or repository access is required**: every number quoted below is also
stored in the accompanying data files, and the headline figures are repeated inline here
so they can be used directly.

Manuscript: *Shape Optimization Using a Neural Implicit Geometry Representation*
(Freinberger, Key, Kofler, Breinl, Drossel, Büttner, Roder, Elgeti), AIMS.

Produced with the exact archived code behind the submission: DeepShapeOpt `864385b`,
DeepSDFStruct `v0.1.0`.

---

## Start here

| File | What it is |
|---|---|
| **`RESPONSE_LETTER.md`** | Full point-by-point draft response, all 11 comments + 5 editorial items. **Read this first.** |
| `00_README.md` | This file — index plus all headline numbers. |

The draft letter contains `[AUTHOR: …]` markers where a fact could not be verified from
the code or archived data. **Four remain open** and are listed at the end of this file.

---

## Headline results, by reviewer comment

### Comment 4 — C⁰ transform kinks at tile interfaces
Files: `grad_probe/summary.txt`, `grad_probe/grad_probe.json`,
`grad_probe/grad_across_interfaces.png`

Gradient measured either side of each tile interface, restricted to the near-surface
band |s| < 0.02, against a mid-tile control where the transform is smooth.
Same geometry (flow channel), decoder (d_lat = 32), tiling (1×8×8), fixed seed.

| variant | p | eikonal λ | recon. loss | \|∇s\| jump if/ctl | normal turn if/ctl |
|---|---|---|---|---|---|
| **A (as submitted)** | 1 | 0 | 6.31e-4 | 0.0742 / 0.0772 | **6.21° / 1.71°** |
| B | 2 | 0 | 5.16e-4 | 0.0755 / 0.0781 | 3.85° / 1.92° |
| C | 1 | 0.05 | 7.67e-3 | 0.1368 / 0.1483 | 17.35° / 6.54° |
| D | 2 | 0.05 | 3.67e-3 | 0.1003 / 0.1037 | 5.12° / 2.30° |

- The kink appears in the **direction** of ∇s (the surface normal), **not** its magnitude
  (|∇s| jump ratio 0.96, i.e. indistinguishable from the control).
- Manuscript configuration: normal turns **6.21° across an interface vs 1.71° mid-tile**
  (factor 3.6). Real but modest.
- **p = 2 halves it** (3.85°, factor 2.0) at 1.85× the design variables (5184 → 9600).
- **Eikonal at reconstruction time is counterproductive**: 17.35°, reconstruction loss
  12× worse. The decoders were trained with `EikonalLambda = 0`, so imposing |∇s| = 1 on
  the latent codes fights a fixed decoder.

### Comment 6 — Normalization chain
Files: `normalization_chain.txt`, `normalization_chain.json`

Two stages, only the first isotropic:
- **Stage A** uniform scale 2/L, L = max extent. The design box does **not** fill [−1,1]³.
- **Stage B** per-axis: each tile mapped independently onto [−1,1].

Net for Experiment 1: a tile is physically **1.5 × 0.8 × 0.8** mapped onto the cube
[−1,1]³, so the decoder sees x compressed by **1.875×** relative to y,z (**2.0×** for the
cube-with-cylinders domain, 3.2 × 1.6 × 1.6). The decoder is therefore queried slightly
off its isotropic training distribution.

**Correction:** the design domain is **3 × 1.6 × 1.6**, not the 3 × 1.5 × 1.5 printed in
Section 3.2.1.

Also verified: T is a triangular wave of period **two** tiles — T(v=0) = −1, T(v=0.5) = 0,
T(v=1) = +1 — so successive tiles are mirrored and dT/dx flips sign at every tile
boundary. Those are the kink planes of Comment 4.

### Comment 7 — float32 vs float64 in the composed gradient
Files: `fd_check_geometry.json` (experiment B), `fd_check_experiment_a.json` (experiment A)

**(B) mesh-free sub-chain d → λ(x) → s**, 864 design variables, 20 random components:

| step h | float32 | float64 |
|---|---|---|
| 1e-1 | 1.677e-2 | 1.676e-2 |
| 1e-2 | 1.723e-3 | 1.715e-3 |
| 1e-3 | **1.242e-3** | 1.207e-3 |
| 3e-4 | 1.418e-3 | 8.044e-4 |
| 1e-4 | 5.040e-3 | 5.557e-4 |
| 1e-5 | 3.909e-2 | **2.231e-4** |

- **Indistinguishable for h ≥ 1e-3**, where truncation error dominates.
- float32 floors at **~1.2e-3 relative** (≈3 significant digits); float64 keeps converging.
- MMA's move limit is `max_step = 0.02`, two orders above where precision matters, so
  single precision is **not limiting**.
- In the float64 run the decoder weights, latent vectors, spline and coordinates are
  **all** promoted. The weights were trained in float32, so this widens the same values —
  both columns differentiate the **identical function**, isolating arithmetic precision.

**(A) full chain including FlexiCubes, float32** — did **not** converge at any step size:

| h | 1e-1 | 1e-2 | 1e-3 | 1e-4 | 1e-5 |
|---|---|---|---|---|---|
| rel. diff | 9.39e-2 | 3.31e-2 | 3.14e-2 | 3.62e-2 | 3.20e-2 |
| topology changes | 40/40 | 40/40 | 34/40 | 23/40 | 9/40 |

Not a precision floor: the FlexiCubes connectivity changes under perturbation, so
**V(d) is only piecewise smooth** in the design variables. A finer extraction grid would
make this worse, not better.

**On ε_D = 2e-4:** it bounds the change in the *objective*, not the gradient. J is read
from OpenFOAM ASCII at `writePrecision 6` (~1e-6 relative), two orders below ε_D, so
arithmetic precision is not the binding uncertainty — discretization is (Comment 8).

### Comment 8 — Mesh independence, residuals, and the FFD comparison
Files: `mesh_study/analysis.txt` (**read this**), `mesh_study/mesh_study.json`,
`mesh_study/meshstudy_*/log.*` (raw OpenFOAM logs), `cfd_setup.json`

Nine runs: three fixed geometries × three refinement levels. **Level 6 is the shipped
configuration behind every manuscript result.** The gate run reproduced the stored
iteration-1 drag to **0.000%**, so the harness matches the manuscript pipeline exactly.

| geometry | L5 cells / drag | L6 cells / drag | L7 cells / drag |
|---|---|---|---|
| initial | 77 280 / 36.9013 | 275 951 / 37.1499 | 1 416 870 / 37.2557 |
| optimized, neural SDF | 70 831 / 30.9071 | 204 951 / 30.9404 | 1 056 701 / 30.9999 |
| optimized, FFD 7×7×7 | 71 289 / 31.0600 | 207 264 / 31.0976 | 1 079 245 / 31.1562 |

Discretization uncertainty L6 → L7: **0.285% / 0.192% / 0.188%**.
Mesh quality: all pass `checkMesh`; max non-orthogonality 41–63°, max skewness < 1.3.

**The ranking survives.** On a common mesh the neural-SDF design has lower drag at every
level, by a stable margin: **0.492% (L5), 0.506% (L6), 0.502% (L7)** — larger than the
discretization uncertainty, and since refinement moves both designs the same way, the
*difference* is better converged than either absolute value.

**⚠ But the headline overstates it — different baselines.**

| run | J₀ | J_final | reported reduction | FlexiCubes res |
|---|---|---|---|---|
| neural SDF | **37.1499** | 30.9404 | 16.71% | 48 |
| FFD 7×7×7 | **36.8845** | 31.0976 | 15.69% | 64 |
| FFD 5×5×5 | **36.8845** | 31.6468 | 14.20% | 64 |

The neural run starts from the **reconstructed** surface, the FFD runs from the **CAD**
surface, 0.72% lower. The reductions are therefore measured from different starting
points, so part of the 16.71 vs 15.69 gap is that offset rather than optimizer
performance. **The defensible claim is absolute final drag on a common mesh: ~0.5%, not a
full percentage point.**

**Adjoint convergence.** In all nine runs the primal converged (606–924 SIMPLE
iterations) but the **adjoint reached its 1000-iteration cap** without meeting
`residualControl 1e-5`. This affects the sensitivity field (the descent direction), not
the tabulated drags, which come from the converged primal.

**Parallel decomposition:** 16 subdomains, hierarchical, from `decomposeParDict.20`
(filename misleading). Identical in the manuscript-era runs (`nProcs : 16` in both).
Worth stating: OpenFOAM's linear solvers are not decomposition-invariant, but since all
runs used the same decomposition this cancels in the comparisons.

### Comment 10 — Figure 18
Files: `fig18_cross_sections.png`, `.pdf`, `.json`

Redrawn with labelled axes, a free-stream arrow, and 3-D sketches showing each cutting
plane. Both cuts contain the flow direction: x–y plane (normal z) and x–z plane
(normal y). Both geometries sectioned with the same origin (centroid of the neural-SDF
result). The two centroids agree to 7e-3, i.e. the centroid constraint held:
FFD [0.0498, 0.0007, −0.0028] vs neural [0.0480, 0.0020, −0.0093].

### Comment 11 — Initialization cost
Files: `initialization_cost_mp_summary.txt` (RTX 4090 — **primary**),
`initialization_cost_summary.txt` (RTX 4000 SFF Ada — independent check),
plus the corresponding `.json`

On the RTX 4090 (same model as used for training):

| case | initialization | per iteration | share | equivalent iterations |
|---|---|---|---|---|
| cube | 210 s | 179.6 s | 3.8% | 1.2 |
| cube with cylinders | 285 s | 199.1 s | 4.2% | 1.4 |
| perforated cube | 173 s | 173.4 s | 4.5% | 1.0 |

**Initialization ≈ one optimization iteration, about 4% of the total.** Dominated
entirely by the latent-code fit; decoder load + lattice construction < 0.5 s.

Independent check on an RTX 4000 SFF Ada: 169 / 238 / 151 s → 3.0–4.0%. The conclusion is
insensitive to hardware. Note the workload does not saturate either GPU (4096 points per
step through a small decoder), so timings are launch/host-bound — which is why the
nominally faster card is *not* faster here.

### Editorial — Table 3 design variables
Files: `design_var_counts.txt`, `design_var_counts.json`

n_design = d_lat · Π(tiling_i + degree_i). Verified against the printed 864 for the cube.
Flow channel 1×8×8 → **5184**; propeller and dog → 27 040; rim → 37 856.
At p = 2 the flow channel would need 9600 (1.85×).

### Editorial — Table 2, dropout / epochs / training cost
File: `training_setup.json`

- **Dropout was INACTIVE.** `NetworkSpecs.dropout_prob = 0.0` in all three models; the
  `0.2` is a stray top-level key in `specs.json` that no code path reads. Confirmed
  against the original MLflow training runs (all log `NetworkSpecs.dropout_prob = 0.0`,
  despite run *names* containing `do=0.2`). **Table 2 must be corrected.**
- **"Epoch"** = one pass over the N = 100 training *scenes*; `ScenesPerBatch = 1`,
  `SamplesPerScene = 4096` resampled per draw. So 1 epoch = 100 optimizer steps × 4096
  points, and 10 000 epochs = **10⁶ optimizer steps** — not 10⁴ passes over all 1.1e6
  samples per scene.
- **Learning rate never decayed**: step schedule interval 12 000 epochs > 10 000 trained.
- **Training time**: 4:24:51 / 4:26:20 / 4:26:40 for d_lat = 8 / 16 / 32, each on a single
  **NVIDIA GeForce RTX 4090 (24 GB)**, host `mp` (driver 590.48.01).

### Table 3 / Figure 9 reconstruction errors
Files: `reconstruction_error_metrics.txt`, `.json`

Recomputed from archived per-sample error fields. **All four printed Table 3 rows
reproduce exactly** (Flow channel 5.73/5.31, Propeller 4.95/5.01, Rim 6.59/8.43,
Dog 4.20/4.15, ×1e-4). Full Figure 9 sweep (6 tilings × 3 code lengths) regenerated.

⚠ **Figure 9 and the Table 3 flow-channel row come from different runs** of nominally the
same configuration (6.99e-4 vs 5.73e-4 at 1×8×8, cl32). Needs a footnote if a reader
cross-references them.

### Table 4
Files: `table4_rebuilt.txt`, `.json`

Rebuilt from the stored optimization histories, convention: row 1 is the initial design,
so iterations = rows − 1. Four of five entries reproduce exactly.

**Correction: FFD 5×5×5 ran 59 design updates, not the printed 46.** It reached its
`num_iter = 60` cap without meeting the convergence criterion (tol 2e-4, window 3); at
iteration 46 the reduction was 13.84%, and the reported 14.20% is the end-of-run value.

**Correction: Experiment 1 is 14.49%**, described in the text as "approximately 15%".

---

## Manuscript corrections — consolidated

1. **Table 2** — dropout was inactive (`dropout_prob = 0.0`), not 0.2.
2. **Table 4** — FFD 5×5×5 is 59 iterations, not 46; note it did not meet the convergence
   criterion.
3. **Section 3.2.1** — design domain is 3 × 1.6 × 1.6, not 3 × 1.5 × 1.5.
4. **Section 3.2.1** — Experiment 1 reduction is 14.49%, not "approximately 15%".
5. **Section 3.2.2 / Table 4** — restate the FFD comparison in terms of absolute final
   drag on a common mesh; the two runs used different baselines.
6. **Section 3.2** — state the parallel decomposition (16 subdomains, hierarchical) and
   that the adjoint did not meet its residual tolerance.
7. **Table 2** — add the epoch definition, training wall-clock and GPU model.
8. **Table 3** — add the design-variable column.
9. **Equation (24)** — adopt σ = −pI + 2ρν·½(∇u + ∇uᵀ) and fix "symmetric part" wording.
10. **Running head** — replace the "LAST-NAME" placeholder.
11. **Section 2.2, Step 5** — "criteria" → "criterion" (twice).

---

## Still open — needs the authors

1. **Comment 9 citation.** A reference is needed for the claim that smoother shapes
   reduce drag in low-Reynolds-number external flow. Pironneau, *On optimum profiles in
   Stokes flow*, JFM **59** (1973) 117–128 is the natural candidate, **but its details
   were not verified** — please check before use.
2. **Comment 2 scope.** The edge-vs-curvature distinction in Figure 10d is answered
   *qualitatively only*; the two contributions were not separated numerically.
3. **Comment 11 caveat.** Per-iteration times come from the original runs, whose hardware
   is unrecorded (`config_log.json` stores only `device: "cuda"`). Decide whether to state
   this.
4. **Zenodo.** Decide whether the new scripts, results and raw solver logs join the
   updated deposit.

**Not attempted, by explicit decision:** retraining the decoder with an eikonal/H¹ loss;
a dropout ablation (unnecessary — dropout was off); and a numerical defence of the
turbulence setup. On the last: the primal uses Spalart-Allmaras while the adjoint is
laminar/frozen (`adjointLaminar`, `adjointTurbulence off`). At Re ≈ 2 the eddy viscosity
should be negligible so this is consistent, **but it was not verified numerically** — no
run was made with turbulence disabled and max(nut)/nu was not measured. A reader opening
the case files will see the mismatch. See `cfd_setup.json`.

---

## File index

| File | Contents |
|---|---|
| `RESPONSE_LETTER.md` | Full draft response letter |
| `00_README.md` | This index and summary |
| `cfd_setup.json` | Solver, decomposition, mesh, physics, turbulence; per-run mesh/residual data |
| `training_setup.json` | Dropout evidence, epoch definition, training times, GPU |
| `design_var_counts.{txt,json}` | Design variables per tiling |
| `reconstruction_error_metrics.{txt,json}` | Table 3 + Figure 9 errors |
| `table4_rebuilt.{txt,json}` | Table 4 rebuilt from stored histories |
| `normalization_chain.{txt,json}` | Two-stage normalization, anisotropy |
| `grad_probe/summary.txt`, `grad_probe.json`, `*.png` | Comment 4 |
| `fd_check_geometry.json` | Comment 7, experiment B (float32 vs float64) |
| `fd_check_experiment_a.json` | Comment 7, experiment A (full chain, topology) |
| `mesh_study/analysis.txt` | **Comment 8 written analysis** |
| `mesh_study/mesh_study.json` | Comment 8 per-run data |
| `mesh_study/meshstudy_*/log.*` | Raw OpenFOAM logs, 9 runs |
| `initialization_cost_mp_summary.txt` | Comment 11, RTX 4090 (primary) |
| `initialization_cost_summary.txt` | Comment 11, RTX 4000 SFF Ada (check) |
| `fig18_cross_sections.{png,pdf,json}` | Comment 10 |
| `slurm-*.out` | Raw job logs |
| `*_cost.txt`, `grad_probe.log` | Raw run logs (large, mostly progress bars — prefer the `_summary` files) |

`fd_check_geometry_float64.json` duplicates `fd_check_geometry.json`; both contain
experiment B only. Experiment A is in `fd_check_experiment_a.json`, recovered from
`slurm-80130-fd_check.out` after a later run overwrote the original file.

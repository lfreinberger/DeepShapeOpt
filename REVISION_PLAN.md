# Revision plan — "Shape Optimization Using a Neural Implicit Geometry Representation"

## Context

The manuscript was submitted and reviewed. The review is favourable (it commends the DOI-deposited
code/data) and asks for 11 numbered clarifications plus 5 editorial fixes. Most are text-only; a
handful need numbers the manuscript never produced.

Work happens on branch `revision/manuscript` in two git worktrees pinned to the exact submitted code:

| worktree | commit | role |
|---|---|---|
| `/usr2/lfrei/ProjectsPhD/revision/DeepShapeOpt` | `864385b` | optimization app |
| `/usr2/lfrei/ProjectsPhD/revision/DeepSDFStruct` | `06f5767` = tag `v0.1.0` | decoder / SDF / spline library |

Environment already synced (`uv sync --frozen`, torch 2.12+cu130, CUDA available). Set
`DEEPSHAPEOPT_MODEL_DIR=/usr2/lfrei/ProjectsPhD/revision/DeepSDFStruct/DeepSDFStruct/trained_models`
and a fresh `DEEPSHAPEOPT_RESULTS_DIR` — the checked-in `.vscode/launch.json` points at the drifted
live repo, whose `specs.json` files differ (weights are identical).

**Guiding constraint:** nothing in `deepshapeopt/` or `DeepSDFStruct/` changes semantics. All new work
lands in new files under `scripts/revision/`, so the branch stays diff-clean against `864385b` for
everything the manuscript reports.

---

## 1. Findings that change the manuscript, found before any new work

These came out of code and archive inspection. Three are corrections the reviewer did not ask for but
that must be made, because answering their questions honestly exposes them.

**1a. Table 2 "Dropout probability 0.2" is wrong — dropout was inactive.**
The decoder is instantiated only from `experiment_specs["NetworkSpecs"]`
(`DeepSDFStruct/deep_sdf/workspace.py:244`). In all three shipped models,
`NetworkSpecs.dropout_prob = 0.0`; the `"dropout_prob": 0.2` sitting at the top level of `specs.json`
is read by nothing. Verified against the original training runs
(`/storage/lfrei/DeepSDFStruct/mlruns/222816595078334609/*/params/NetworkSpecs.dropout_prob` = `0.0`,
and `artifacts/mlflow_specs_dump.json`). Independently, `pretrained_models.py:55` calls
`decoder.eval()`, so dropout would be inactive at reconstruction regardless.
**This converts reviewer comment 5 from a retraining ablation into a one-line table correction.**

**1b. The design domain is not 3 × 1.5 × 1.5.**
`config_latent_cube.json` gives `[[-1.5,-0.8,-0.8],[1.5,0.8,0.8]]` = **3 × 1.6 × 1.6**, and the
cylinder case uses **3.2 × 1.6 × 1.6**. Section 3.2.1 must be corrected.

**1c. Table 4, FFD 5×5×5 — restate as 60 iterations / 14.20%.** *(decided)*
The archived run ran the full `num_iter=60`; at iteration 46 it was at 13.84%, reaching 14.20% only at
iteration 60, and the convergence criterion (`obj_tol=2e-4`, `window=3`) was never met. The other four
runs match the manuscript exactly under the `rows−1` convention. Restate from
`results_ffd_cube_with_cylinders_5x5x5/optimization/optimization_history.csv` and re-check that the
Table 4 ranking still holds before writing the reviewer-8 response.

**1d. The adjoint never met its residual tolerance.** Every surviving log of this case family shows
`adjS1 solution reached max. number of iterations 1000` while the primal converged in ~565. This must
be reported honestly in the reviewer-8 answer.

---

## 2. Triage of the review

### Answerable now — text only, no code, no runs (10 of 16 items)

| # | Item | Basis for the answer |
|---|---|---|
| 1 | Regularity assumptions on Γ | Standing-assumption paragraph in §2.1.1 |
| 2 | Fig. 10a: edge normals vs. curvature | Discussion sentence; the flow channel's edges are visible in Fig. 10a |
| 3 | Fig. 2 caption → regularity link | Caption edit |
| 5 | Dropout justification | **Finding 1a** — correct Table 2, state dropout was inactive |
| 6 | Normalization / domain mapping | Code-derived, see §3 item 0.6; plus **finding 1b** |
| 9 | Citation for low-Re smooth shapes | Literature |
| 11 | Cost of initialization stage | Needs one cheap measurement — see item 1.1 |
| E1 | Running head placeholder | Typo |
| E2 | Eq. (24) factor-of-2 / "symmetric part" | Adopt `σ = −pI + 2ρν·½(∇u+∇uᵀ)` and fix the loose wording |
| E3 | "criteria" → "criterion" | Typo, two places |
| E4 | Definition of "epoch" + wall-clock + hardware | Mostly recoverable, see item 0.5 |

**Definition of "epoch" (E4):** `SDFSamples.__len__` returns the number of *scenes*; the DataLoader
uses `ScenesPerBatch=1` with `SamplesPerScene=4096` resampled at each draw. So one epoch = 100
optimizer steps × 4096 points, and 10 000 epochs = 10⁶ Adam steps — **not** a pass over all 1.1 M
samples per scene. Also worth stating: the LR schedule interval is 12 000 > 10 000 epochs, so the
learning rate never decayed.

**Wall-clock (E4):** recorded in `trained_models/primitives_cl{08,16,32}/training_summary.json` —
4:24:51 / 4:26:20 / 4:26:40 on host `mp`. The **GPU model is not recorded** (only the string `"cuda"`,
`training_latent_field.py:911`); recover it from host `mp` out-of-band rather than guessing.

### Needs new code or new runs (6 items)

| # | Item | What is required |
|---|---|---|
| 4 | C⁰ transform → gradient kinks; H¹/eikonal; p≥2 | `\|∇s\|` probe + 4 reconstructions (item 1.2) |
| 7 | float32 vs float64 precision in the composed gradient | FD gradient check (item 1.3) |
| 8 | Mesh resolution, residuals, mesh independence | 9-run OpenFOAM study (item 2.1) |
| 10 | Fig. 18 axes / cutting-plane inset | New plotting script (item 3.1) |
| E5 | Table 3: add design-variable counts | Formula script (item 0.2) |
| — | Table 3 / Fig. 9 MAE values | Not stored; recomputable from archived VTPs (item 0.3) |

---

## 3. Work plan

### Wave 0 — CPU, hours, no runs. Unblocks the rebuttal prose.

**0.2 Design-variable counts (E5).** New `scripts/revision/design_var_counts.py` (~20 lines).
From `deepshapeopt/reconstruction.py:60-88`: clamped knot vectors give `p_i+1` control points per
axis, `insert_knots` adds `t_i−1`, hence

```
n_design = latent_dim · Π_i (tiling_i + degree_i)
```

Check: cube `[2,2,2]`, cl32 → 32·3³ = **864** ✓ matches the manuscript. Flow channel `[1,8,8]` →
32·(2·9·9) = **5184**. Useful side result for reviewer 4: p=2 on `[1,8,8]` gives 9600, i.e. **1.85×
more design variables** — a real cost of the reviewer's own suggestion.

**0.3 Recompute Table 3 / Fig. 9 MAEs from archived VTPs.** New
`scripts/revision/recompute_error_metrics.py` (~80 lines). `error_metrics.json` was never written for
the manuscript runs, but the per-sample and per-vertex error fields survive as `sdf_error.vtp` and
`*_mesh_sdf_error.vtp` (Float32 point array `sdf_error`) under
`/storage/lfrei/Archive/DeepShapeOpt_archive_paper/experiments/reconstruction/<case>/`. Read with
`pyvista` (already a dependency), reduce with the same statistics as `_compute_reconstruction_metrics`
(`deepshapeopt/reconstruction.py:747-766`), respecting `error_cutoff=0.1`. Schema template:
`/usr2/lfrei/ProjectsPhD/DeepShapeOpt/experiments/reconstruction/feed_channel/results/reconstruction/flow_channel_tiling_1x8x8/error_metrics.json`.

> **Gate:** at least one recomputed MAE must reproduce a printed manuscript value to printed
> precision. If none do, the cutoff/reduction convention is wrong — stop, do not publish recomputed
> numbers.

**0.4 Rebuild Table 4.** New `scripts/revision/rebuild_table4.py` (~60 lines). Reads all five
`optimization_history.csv` + `config_log.json` from
`/storage/lfrei/Archive/DeepShapeOpt-old-private-backup/experiments/optimization/drag_optimization_cube/`,
applies the `rows−1` convention, emits the LaTeX table. Implements decision **1c**.
**Must be settled before the reviewer-8 text is written** — it changes the percentages being compared.

**0.6 Normalization chain (reviewer 6).** New `scripts/revision/print_normalization_chain.py` (~30
lines) using the existing `setup_model_and_domain` / `build_lattice`
(`shape_optimization.py:57,105`). The mapping is **two stages**:

- **Stage A** `fit_box_to_unit_cube` (`reconstruction.py:307-333`) — *uniform isotropic* scale, one
  scalar `L = max(size)`.
- **Stage B** `transform` (`lattice_structure.py:280-284`) — *per-axis*: each tile is independently
  mapped to [−1,1] via `x_norm = (x−b0)/(b1−b0)`.

Net for the cube case with tiling `[2,2,2]`: the decoder sees the geometry at aspect ratio
**1.875 : 1 : 1** relative to its isotropic `[−1,1]³` training box. Stage B undoes Stage A's isotropy,
so the decoder is queried slightly off-distribution — state this rather than let the reviewer find it.
Deliverable: a small commutative-diagram figure with the numbers filled in.
Note `DomainFrame` does **not** exist in this version — do not cite it.

### Wave 1 — GPU, CFD-free, about a day.

**1.1 Initialization cost (reviewer 11).** Reconstruction was never timed: `OptimizationLogger` is
built at `optimize_drag_latent.py:137` and `start_time` set at line 144, both *after*
`run_reconstruction` at lines 82-92; `run.log` has no timestamps
(`logging.Formatter("%(message)s")`, `runtime.py:32`). Reconstruction is CFD-free, so simply re-time
it standalone: `scripts/reconstruct.py --config ...` under `/usr/bin/time -v`, once per case
(~1-3 min each). Optionally add 3 lines around `optimize_drag_latent.py:82-92` to record it in
`config_log.json` for reproducibility. Combine with `elapsed_s` from the stored histories to state the
ratio.

**1.2 |∇s| / normal-artifact diagnostic (reviewer 4).** *Diagnostic only — decided.*

*Where the kinks are, analytically.* With `v = t·(x−b0)/(b1−b0)`, `torch.floor` contributes zero
gradient and `torch.abs` the sign, so `dT/dx = ±2t/(b1−b0)` a.e. with the sign flipping at every
integer `v` — i.e. exactly at the planes `x_i = b0_i + (k/t_i)(b1_i−b0_i)`, `k = 0…t_i`. No search
needed; for `[1,8,8]` that is 9 planes in y and 9 in z.

*Design matrix* — same geometry (flow channel), same decoder (`primitives_cl32`), same tiling
`[1,8,8]`, fixed RNG seed. Four **config copies only, no new reconstruction code** (`spline_degree` is
read at `reconstruction.py:614`, `eikonal_lambda` at `:164`):

| run | `spline_degree` | `eikonal_lambda` |
|---|---|---|
| A | `[1,1,1]` | 0.0 (manuscript baseline) |
| B | `[2,2,2]` | 0.0 |
| C | `[1,1,1]` | 0.05 |
| D | `[2,2,2]` | 0.05 |

*New file* `scripts/revision/grad_probe.py` (~120 lines), import-only:
1. Restore the lattice from saved reconstruction parameters (same path as `reconstruction.py:731`).
2. **Line probe** across three consecutive interface planes: `x.requires_grad_(True)`,
   `torch.autograd.grad(s.sum(), x)` — reuse the pattern already in the eikonal block
   (`deep_sdf/reconstruction.py:121-131`); do not hand-roll finite differences. Plot `|∇s|` and
   `∂s/∂n` for A-D with the analytic kink planes as vertical rules. **This is the money figure.**
3. **Slab statistics** via `SDF.get_equidistant_grid_sample` (`SDF.py:76-127`, has no `no_grad`):
   restrict to `|s| < δ`, bin by distance to the nearest kink plane, report `RMS(||∇s|−1|)` in the
   interface bin vs. the interior bin → a 4×2 table.
4. **Extracted-normal jump:** per-vertex normals via the existing `foam_utils.compute_vertex_normals`
   on the FlexiCubes mesh; report the 95th-percentile dihedral angle, interface vs. interior bin.
   Shows whether the kink actually reaches a surface a reader would see.
5. Do **not** call `mesh.export_sdf_grid_vtk` — it wraps evaluation in `torch.no_grad()`
   (`mesh.py:1019`), killing the gradient. Write the grid with `pyvista.ImageData` instead (~10 lines).

*What to state honestly in the rebuttal:* the kink lives in `T(x)`, **upstream of the decoder**.
Neither p≥2 nor an eikonal penalty removes it — they only shrink the magnitude of the jump by making
`λ(x)` vary less abruptly. And eikonal-on *reconstruction* regularizes only the latent codes; the
shipped decoders were trained with `EikonalLambda = 0.0`. Say this explicitly.

**1.3 Gradient verification, float32 vs float64 (reviewer 7).** New
`scripts/revision/fd_check_geometry.py` (~150 lines).

*Scope:* exclude CFD. The volume/centroid constraints already traverse the entire geometry chain
`d → λ(x) → s → Γ(FlexiCubes) → {V,c}`, are differentiated by autograd in the production loop
(`optimize_drag_latent.py:187,229`), and are cheap. Reuse `generate_mesh`
(`shape_optimization.py:270`) unchanged.

- Central differences on a **fixed random 20-component subset** of the 864 design variables (not the
  full Jacobian), step sweep `h ∈ {1e-1 … 1e-5}` → the classic V-curve.
- **Topology-change guard, not optional:** FlexiCubes changes vertex count when a perturbation moves
  the zero level set across a grid cell, making `V(d)` non-smooth. Record
  `mesh.vertices.shape[0]` at every evaluation and flag any step where it changes; if it changes at
  all step sizes, raise `mesh_resolution` above 48 and report the value used.
- **float64 arm:** try `decoder.double()` + float64 lattice, bypassing `with_float32_lattice`
  (`reconstruction.py:340`). Time-box to ~1 h — FlexiCubes may have float32-hardcoded kernels. Fallback
  that always works: run the FD test on the smooth sub-chain only, `f(d) = Σ_k w_k s(x_k; d)` at ~10⁴
  fixed points, no meshing.

*Then connect to `ε_D = 2e-4` explicitly* — the reviewer's actual question, and a different one:
`ε_D` bounds `obj_change = |J_k − J_{k−1}|/|J_0|`, a criterion on the **objective**, not the gradient.
`J` is read from OpenFOAM ASCII at `writePrecision 6` (`controlDict:37`), i.e. relative representation
error ~1e-6, **two orders below `ε_D`** — so ASCII precision is not the binding constraint. The binding
constraint is the discretization uncertainty of `J`, quantified by item 2.1.
**Reviewers 7 and 8 therefore share one deliverable; do not answer them independently.**

### Wave 2 — OpenFOAM, overnight. The long pole.

**2.1 Mesh-independence study on fixed geometries (reviewer 8).** *Full 9-run grid, no defensive
extras — decided.* New `scripts/revision/mesh_study.py` (~200 lines).

No mesh machinery exists: `snappyHexMeshDict`/`blockMeshDict` are copied verbatim by
`prepare_foam_runtime` (`foam_utils.py:22-28`) and no config key touches any CFD mesh setting. (The
config key `mesh_resolution` is the FlexiCubes extraction resolution, not the CFD mesh.)
`configure_foam_runtime` is dead code *and* incompatible with the shipped `optimisationDict` — leave it
alone; repairing it would modify the manuscript code path.

*Approach:* one template, patched **only on the runtime copy**, with anchored `re.sub` asserting
exactly one match per pattern (fail loudly otherwise). Text substitution rather than a `FoamFile`
round-trip — snappy's `(6 6)` / `((0.05 6))` tuple syntax is where reformatting risk is highest.

| entry | line | L5 | L6 (baseline) | L7 |
|---|---|---|---|---|
| `features … level` | `:94` | 5 | 6 | 7 |
| `refinementSurfaces … level` | `:114` | `(5 5)` | `(6 6)` | `(7 7)` |
| `refinementRegions … levels` | `:148` | 5 | 6 | 7 |
| `maxGlobalCells` | `:63` | 2e6 | 2e6 | 8e6 |
| `maxLocalCells` | `:56` | 1e5 | 1e5 | 5e5 |

Leave `refinementBox level 3`, `nCellsBetweenLevels 2`, layers and blockMesh untouched so one thing
varies.

*Geometries* — stored STLs, no torch/lattice/reconstruction in this harness at all. From
`/storage/lfrei/Archive/DeepShapeOpt_archive_paper/experiments/optimization/drag_optimization_cube/`:
`results_cube_with_cylinders/stl_series/shape_0001.stl` (initial),
`results_cube_with_cylinders/optimization/current_shape.stl` (neural-SDF final),
`results_ffd_cube_with_cylinders_7x7x7/optimization/current_shape.stl` (FFD final) — the three shapes
behind the contested 16.71% vs 15.69% comparison. Note `shape_0000.stl` was not archived, so
"initial" is the shape after one MMA step.

*Run* via the existing `foam_utils.run_openfoam_case(case_dir, verbose=False)`. **Do not `rmtree` the
case** the way `optimize_drag_latent.py:315-317` does, and copy `log.*` out *before* the next run —
`FoamCase.clean()` at `foam_utils.py:82-85` destroys them.

*Two new parsers in the same file:*
- `parse_checkmesh` → cells / hexahedra / polyhedra / Mesh OK / max non-orthogonality. Reference format
  from the surviving Apr-2026 log of this case family (258 447 cells = 205 227 hex + 48 238 poly).
- `parse_solver_log` → per-field final residuals plus `converged in (\d+) iterations` /
  `reached max\. number of iterations (\d+)`. **Reuse the regex dictionary from
  `plotting_utils.plot_residuals_from_log` (`plotting_utils.py:151-170`) verbatim** — it already covers
  the `Uaas1*`/`paas1` adjoint fields — returning last values instead of plotting.
- Objective from `optimisation/objective/0/dragadjS1`, the path `load_sensitivities` already uses.
- The shipped `Allrun` already prints `Finished: <step> in M min S s` — free per-stage wall clock.

One CSV row per run: `geometry, level, cells, hex, poly, mesh_ok, max_nonortho, drag, primal_iters,
primal_converged, adjoint_iters, adjoint_converged, res_p, res_Ux, res_paas1, res_Uaas1x, t_snappy_s,
t_solver_s`.

> **Gate — run this one first, alone:** `cyl_init` at L6, checked against the `objective` column at
> iteration 1 of `results_cube_with_cylinders/optimization/optimization_history.csv`. If it matches to
> printed precision, the standalone harness reproduces the manuscript pipeline and every later number
> is trustworthy. If not, stop and find out why before burning the remaining 8 runs.

*Cost:* ~20-40 min per run at L6 (258k cells, 20 cores); L5 ≈ ¼, L7 ≈ 4-8×. Total roughly **6-12 h
serial** — run overnight, sequentially. If time-constrained, drop `cyl_init` at L7 first.
*Risk:* if `maxGlobalCells` is not raised for L7, snappy silently stops refining and the row is a lie.
Assert the achieved cell count is ≈4× L6.

*Report the adjoint cap honestly* (**finding 1d**): give the achieved adjoint residual per level, and
note that it affects the *sensitivity* field rather than the drag values in the table (which come from
the converged primal), and that an inexact descent direction is a plausible partial explanation of the
slow tails in the convergence histories.

**Deliverables:** drag-vs-cell-count table (3×3); the sentence the reviewer is fishing for — "drag
varies by X% between L6 and L7, so the [corrected] neural-SDF vs FFD difference is / is not resolved by
the mesh"; a residual table including the cap; a `checkMesh` quality row. **Depends on 0.4.**

### Wave 3 — CPU, hours. Good filler while OpenFOAM runs.

**3.1 Figure 18 with axes and cutting-plane inset (reviewer 10).** New
`scripts/revision/fig18_cross_section.py` (~80 lines). No such code exists, but the inputs are plain
STLs and `trimesh` is already a dependency: `trimesh.load_mesh(...).section(plane_origin, plane_normal)`
→ `.to_planar()` → matplotlib overlay. **Copy the `ax.quiver` + `ax.text` axis-arrow block from
`plotting_utils.save_shape_snapshot` (`plotting_utils.py:296-328`)** so styling matches the other
figures. No CFD, no GPU.

---

## 4. Ordering

```
Day 1 CPU:    0.2  0.3  0.4  0.6  +  all text-only items      → rebuttal prose unblocked
Day 1 GPU:    1.1  1.2 (A-D + probe)  1.3
Day 2 night:  2.1 gate run → 9-run grid                        [needs 0.4]
Day 3 CPU:    3.1
```

Only two hard orderings: **0.4 before the reviewer-8 text** (the compared percentages change), and
**0.3's gate before any recomputed MAE is published**. Everything else is independent.

## 5. Files

| file | status | ~LOC | needs |
|---|---|---|---|
| `scripts/revision/design_var_counts.py` | new | 20 | CPU |
| `scripts/revision/recompute_error_metrics.py` | new | 80 | CPU |
| `scripts/revision/rebuild_table4.py` | new | 60 | CPU |
| `scripts/revision/print_normalization_chain.py` | new | 30 | GPU |
| `scripts/revision/grad_probe.py` | new | 120 | GPU |
| `scripts/revision/fd_check_geometry.py` | new | 150 | GPU |
| `scripts/revision/mesh_study.py` | new | 200 | OpenFOAM |
| `scripts/revision/fig18_cross_section.py` | new | 80 | CPU |
| `experiments/reconstruction/feed_channel/config_{A,B,C,D}.json` | new | — | 2 keys changed each |
| `scripts/optimize_drag_latent.py` | modify | +3 | reconstruction timing, lines 82-92 |
| `deepshapeopt/shape_optimization.py:275`, `reconstruction.py:344` | modify | docstrings | both claim float64; the chain is float32 |

~740 new lines, all in new files. No library semantics change.

## 6. Verification

- **0.3 gate:** a recomputed MAE reproduces a printed manuscript value.
- **2.1 gate:** the standalone harness reproduces iteration-1 drag from the stored history.
- **1.3 guard:** FlexiCubes vertex count constant across FD perturbations.
- **2.1 assert:** L7 achieves ≈4× the L6 cell count (else `maxGlobalCells` bound the mesh).
- **Overall:** `git diff 864385b` on `revision/manuscript` touches only `scripts/revision/`, the four
  new configs, 3 lines of `optimize_drag_latent.py`, and two docstrings. Commit one reviewer point per
  commit with the point number in the message — that diff is most of a response letter.

## 7. Deliberately out of scope

- **Decoder retrain with eikonal/H¹ loss** (~4.5 h/model). The kink is upstream of the decoder, so
  retraining cannot remove it; a retrained decoder invalidates every other number unless everything is
  re-run; and the dataset stores only `(x,y,z,sdf)` — no normals — so H¹ supervision would also need
  regenerated data.
- **Dropout ablation.** Provably inactive (finding 1a).
- **Re-running any optimization loop.** All five histories survive as CSV.
- **Full 864-component FD Jacobian.** A 20-component subset with a step sweep is standard and more
  informative per unit cost.
- **Adding residuals/cell counts to the per-iteration CSV** (`logging.py:75-134`). Would change the
  production code path for no deliverable.
- **Repairing `configure_foam_runtime`.** Dead and doubly incompatible; leave it.
- **Turbulence-model defence.** *Chosen exposure, flagged:* `turbulenceProperties` sets
  `RASModel SpalartAllmaras; turbulence on` while the adjoint is `adjointLaminar; adjointTurbulence
  off`. At `nu = 1`, `U = 1`, `L ≈ 1.8` the flow is Re ≈ 2, where SA should give `nut ≈ 0` and the
  frozen-turbulence adjoint is exact — but a reviewer who opens the case files sees the inconsistency
  without doing that arithmetic. A single extra L6 run with `RASModel laminar` would settle it in
  ~30 min if the second round asks.

---
---

# Outcomes — what actually happened

*Appended 2026-08-04, after executing the plan above. Everything before this line
is the plan as approved, unedited. Several of its predictions were wrong; they are
corrected here rather than silently edited above, so the record stays honest.*

All 11 numbered reviewer comments and 5 editorial items are now answerable.
13 commits on `revision/manuscript`; `git log 864385b..HEAD` carries the findings
in its commit bodies.

## Results by reviewer point

**R4 (C⁰ kink).** The planned metric — band-averaged `RMS(||∇s|−1|)` — was the
**wrong observable** and initially reported "no effect" (ratio 0.99). Two flaws: it
averaged over regions with `|s|` up to 0.34, far outside the clamped training loss
(δ=0.1) where the decoder is unconstrained and its gradients meaningless; and it
measured gradient *magnitude* when the kink is a *sign flip*. Replaced with the jump
in `∇s` across interface planes versus a mid-tile control, restricted to `|s|<0.02`.

Result: the kink is real and shows in the **normal direction**, not the magnitude.
Manuscript config: median normal turn 6.21° at interfaces vs 1.71° mid-tile (3.6×);
`|∇s|` jump ratio 0.96, i.e. no magnitude effect. p=2 halves it (3.85°, ratio 2.0)
at 1.85× the design variables. Eikonal-on at reconstruction time is
**counterproductive** (17.35°, reconstruction loss 12× worse) because the decoders
were trained with `EikonalLambda=0`.

**R6 (normalization).** Stage A is uniform, Stage B is per-axis and undoes it: a tile
is physically 1.5×0.8×0.8 mapped onto `[-1,1]³`, so the decoder sees x compressed
**1.875×** (2× for the cylinder case). Also: the design domain is **3 × 1.6 × 1.6**,
not the 3 × 1.5 × 1.5 printed in §3.2.1.

**R7 (precision).** float32 and float64 FD curves are **indistinguishable for h ≥ 1e-3**;
below that float32 floors at 1.2e-3 relative while float64 continues to 2.2e-4.
MMA's move limit is `max_step = 0.02`, two orders above where precision bites, so
single precision is not limiting. Separately, `ε_D` bounds the *objective*, and `J`
is read at `writePrecision 6` (~1e-6), two orders below `ε_D`.

The full-chain arm (through FlexiCubes) **did not converge at any step size** — 3-4%
error from h=1e-1 to 1e-5 with the topology guard firing throughout. `V(d)` is only
*piecewise* smooth in the design variables. The plan's contingency (raise
`mesh_resolution`) would make this **worse**, not better: a finer grid puts more cell
corners near the zero level set. Reported as a property of the representation.

**R8 (mesh independence).** 9 runs, gate reproduced iteration-1 drag to 0.000%.
L6→L7 uncertainty 0.19-0.29%. The neural design beats FFD 7×7×7 on absolute drag at
**every** level by a stable 0.492/0.506/0.502%, so the ranking survives.

**But the headline overstates it:** the two optimizations used different baselines —
neural starts from the *reconstructed* surface (J₀=37.1499), FFD from the *CAD*
surface (J₀=36.8845), 0.72% lower — and different FlexiCubes resolutions (48 vs 64).
Part of the 16.71 vs 15.69 gap is that offset, not optimizer performance. The
defensible claim is absolute final drag on a common mesh: ~0.5%, not a full point.

**R11 (initialization cost).** 169/238/151 s versus 179.6/199.1/173.4 s per iteration
→ **3.0-4.0% of the loop, about one iteration**. Measured by replicating
`optimize_drag_latent.py:79-107`, not `reconstruct.py` as the plan said — that is the
*standalone* path with different normalization.

**Table 3 / Fig 9.** Gate passed 4/4 exactly. Note Fig. 9 and the Table 3 flow-channel
row come from **different run families** (6.99e-4 vs 5.73e-4 for nominally the same
configuration) — needs a footnote.

**Table 4.** FFD 5×5×5 is **59** updates — not the printed 46, and not 60 as first
stated; 59 is what the `rows−1` convention used by every other row gives. Experiment 1
is **14.49%**, described as "approximately 15%".

**Table 2.** Dropout was **inactive** (`NetworkSpecs.dropout_prob = 0.0`); the 0.2 is a
stray top-level key nothing reads. Table correction, not an ablation.

## Where the plan's estimates were wrong

- Mesh study: predicted 6-12 h serial, actual **1 h 40** (per run 4 min at L6, not 20-40).
  L7 dominates at ~22-31 min.
- The plan assumed `tests/env.sh` for Slurm; it **does not exist** in this worktree
  (`tests/` is gitignored at 864385b). Environment set inline instead.
- Three latent job-script bugs the plan could not have predicted: OpenFOAM's bashrc
  fails under `set -e` *and* under `set -u`, and `uv` is not on PATH in a batch shell.

## Process notes

- **Slurm**: mid-session the user required GPU/large studies to go through Slurm with
  the job script reviewed before `sbatch`. Adopted; `sbatch` is blocked for the agent,
  so submissions are the user's. Jobs 80129 (gate), 80130/80132 (FD), 80131 (grid).
- **Archive write incident**: `compute_metrics_from_vtp` unconditionally saves an
  `sdf_error.vtp` next to its input, which overwrote one file in the deposited archive.
  Verified content-identical to the canonical regeneration (bit-identical on an
  untouched sibling), inputs untouched. `recompute_error_metrics.py` now uses the pure
  reduction `compute_near_surface_metrics` instead.
- Results live in `revision_artifacts/`, **not** `revision_results/` — `.gitignore` has
  `results*/`. Note `**/log.*` and `slurm-*.out` are also ignored, so raw OpenFOAM and
  Slurm logs are **not** committed; their content is extracted into the JSON artifacts.

## Still open

- No response-letter prose has been written.
- Nothing pushed; all 13 commits are local.
- Everything in §7 above (decoder retrain, dropout ablation, turbulence-model defence)
  remains deliberately out of scope. The Re≈2 / SpalartAllmaras inconsistency is still
  an undefended exposure by explicit choice.

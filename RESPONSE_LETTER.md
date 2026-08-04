# Response to the reviewer

Manuscript: *Shape Optimization Using a Neural Implicit Geometry Representation*
Freinberger, Key, Kofler, Breinl, Drossel, Büttner, Roder, Elgeti

---

We thank the reviewer for a careful and constructive report, and in particular for
engaging with the numerical details of the method rather than only its presentation.
Several comments prompted us to re-examine our own results, and this uncovered three
errors in the submitted manuscript that we correct below in addition to answering the
points raised. We are grateful for the scrutiny.

All new numbers reported here were produced with the exact code archived alongside the
submission (DeepShapeOpt `864385b`, DeepSDFStruct `v0.1.0`) and are reproducible with
the scripts added in the revision.

> **[AUTHOR] Before submitting, please resolve every `[AUTHOR: …]` marker below.**
> They mark facts I could not verify from the code or archived data.

---

## Mathematical setting and regularity

### Comment 1 — Regularity assumptions on Γ

> *The definition of the SDF and the subsequent shape-sensitivity analysis implicitly
> require the boundary Γ to be smooth… This should be stated explicitly as a standing
> assumption… It would also be useful to specify how the framework behaves at points
> where this assumption fails.*

We agree and have added an explicit standing assumption at the start of Section 2.1.1.

We now state that Γ is assumed to be Lipschitz and piecewise C¹, so that the outward
unit normal **n̂** exists almost everywhere on Γ. This is the regularity actually
required by the force integral (24) and by the continuous-adjoint surface sensitivity;
it admits the edges and corners that occur in several of our test geometries, rather
than assuming global smoothness that our examples do not possess.

We have also added a paragraph on what happens where the assumption fails. Two remarks
are relevant:

1. The signed distance function itself remains well defined at edges and corners — it
   is Lipschitz everywhere — but it is not differentiable there, and ∇s is undefined on
   a set of measure zero. Since all surface integrals are taken with respect to surface
   measure, this set does not contribute.
2. In practice the optimizer never evaluates the exact SDF at such points. The
   objective and its sensitivity are evaluated on the *extracted* surface Γ̂, a
   triangulation produced by FlexiCubes, on which face normals are defined everywhere
   and vertex normals are area-weighted averages. Sharp features of the underlying
   zero-level set are therefore represented by a finite dihedral angle rather than by a
   true singularity, and no special treatment is required.

We note in passing that the neural SDF is in any case a smooth approximation: the
decoder is a finite composition of affine maps and ReLU activations, hence piecewise
linear and differentiable almost everywhere, so genuinely singular points are not
reproduced exactly. This is now stated.

### Comment 2 — Figure 10a, edges versus curvature

> *The flow-channel ground-truth geometry has edges where the unit normal is not
> uniquely defined… part of the reconstruction error… may be due to these normal
> ambiguities rather than to curvature itself.*

The reviewer is right that our original wording conflated two distinct effects, and we
have rewritten the passage.

The revised text distinguishes (i) regions of high but smooth curvature, where the
error reflects the finite spatial resolution of the latent control lattice, from
(ii) neighbourhoods of edges, where the target field is not differentiable and no
continuous latent field of any resolution can reproduce it exactly. We now attribute
the largest local errors in Figure 10d primarily to the second mechanism, and note that
this is consistent with the observation that refining the tiling reduces but does not
eliminate them (Figure 9).

> **[AUTHOR: we have *not* separated the two contributions quantitatively.** Doing so
> would require classifying surface samples by proximity to a feature edge of the
> ground-truth mesh and reporting the error conditional on that classification. This is
> straightforward with the archived per-sample error fields and could be added if the
> reviewer considers it necessary; we currently make only the qualitative distinction,
> which we believe is what the comment asks for. Please confirm you are comfortable
> with that scope.]

### Comment 3 — Figure 2 and the regularity assumptions

> *Figure 2 illustrates the continuous latent-field construction on a domain with smooth
> boundaries; it would help the reader if the caption or surrounding text made the link
> to the regularity assumptions above explicit.*

The caption of Figure 2 now states that the illustration shows a smooth boundary for
clarity, that the construction itself requires only the Lipschitz/piecewise-C¹
regularity assumed in Section 2.1.1, and that the representation is applied to
geometries with edges in Sections 3.1 and 3.2.

---

## Comment 4 — C⁰ transformation, gradient kinks, H¹/eikonal loss, higher-degree splines

> *The proposed hat-function-like transformation is C⁰… the kinks in T(x) propagate into
> the derivative of s… Does the current C⁰ transformation introduce any visible
> artifacts in the extracted normals or in the sensitivity field at the voxel
> interfaces? … A higher-degree B-spline latent field (p ≥ 2) would presumably also be
> an option here.*

This was the most valuable comment in the report and we have investigated it
numerically rather than answering it in the abstract. A new subsection has been added.

**Where the kinks are.** From Equation (10), with v = t·x_norm, the derivative is
dT/dx = ±2t/(b₁−b₀) almost everywhere, and the sign flips at every integer v — that is,
exactly at the tile interfaces. These planes are known analytically and need not be
searched for.

**What we measured.** Using the same geometry (flow channel), the same pretrained
decoder (d_lat = 32), the same 1×8×8 tiling and a fixed random seed, we evaluated ∇s at
pairs of points straddling each interface plane and compared the result against a
control measurement on mid-tile planes at the same separation, where the transform is
smooth. The comparison is restricted to the near-surface band |s| < 0.02; outside the
clamping distance δ = 0.1 the training loss (13) does not constrain the decoder, so
gradients there carry no information.

| variant | p | eikonal λ | recon. loss | \|∇s\| jump (interface / control) | normal turn (interface / control) |
|---|---|---|---|---|---|
| A (as submitted) | 1 | 0 | 6.31·10⁻⁴ | 0.0742 / 0.0772 | **6.21° / 1.71°** |
| B | 2 | 0 | 5.16·10⁻⁴ | 0.0755 / 0.0781 | 3.85° / 1.92° |
| C | 1 | 0.05 | 7.67·10⁻³ | 0.1368 / 0.1483 | 17.35° / 6.54° |
| D | 2 | 0.05 | 3.67·10⁻³ | 0.1003 / 0.1037 | 5.12° / 2.30° |

**Findings.**

1. The kink is measurable, and it appears exactly where the reviewer's reasoning
   predicts: in the **direction** of ∇s — the surface normal ∇s/|∇s| — and not in its
   magnitude. The relative jump in |∇s| across an interface is indistinguishable from
   the mid-tile control (ratio 0.96).
2. For the configuration used in the manuscript the median normal direction changes by
   **6.2° across an interface, against 1.7° mid-tile**, a factor of 3.6. The effect is
   therefore real but modest, and it is local to the interface planes.
3. **A quadratic latent field (p = 2) halves it** — 3.85°, a factor of 2.0 over its
   control — and slightly improves the reconstruction loss. The cost is 1.85× more
   design variables (5184 → 9600 for this tiling). We now report this trade-off
   explicitly, and we thank the reviewer for the suggestion: it is the effective lever.
4. **Enabling the eikonal penalty during reconstruction is counterproductive.** The
   normal jump grows to 17.4° and the reconstruction loss degrades by an order of
   magnitude. The reason is structural: the decoder used here was trained with
   EikonalLambda = 0, so imposing |∇s| = 1 on the latent codes alone works against a
   fixed decoder rather than with it.

**On an H¹ or eikonal training loss.** We agree this is the principled direction, and we
now say so in the outlook, but with two honest qualifications. First, the kink lives in
T(x), **upstream of the decoder**; no training loss can remove it, because it is a
property of the coordinate map, not of the learned function. A gradient-aware loss can
only reduce the magnitude of the jump by making the local field vary more smoothly.
Second, our training data stores only (x, s_gt) pairs, so H¹ supervision would require
regenerating the dataset with ground-truth normals. Retraining the decoder would also
mean that every result in the paper referred to a different model. We therefore report
the measurement above and leave decoder-level gradient supervision to future work.

---

## Comment 5 — Dropout

> *Dropout with probability 0.2 is applied during training. The motivation is not
> stated… Could the authors briefly justify the choice and, ideally, report whether it
> materially affects reconstruction accuracy?*

The reviewer's instinct that the usual overfitting argument does not apply here is
correct, and checking this revealed **an error in Table 2, which we have corrected.**

Dropout was **not active** during training. The decoder is instantiated solely from the
`NetworkSpecs` block of the model specification, in which `dropout_prob = 0.0`. The
value 0.2 appears as a stray top-level key in the same file that no code path reads. We
verified this against the archived training runs, whose logged
`NetworkSpecs.dropout_prob` is 0.0 for all three networks. Independently, the decoder is
placed in evaluation mode before reconstruction, so dropout would be inactive at
reconstruction time in any case.

Table 2 has been corrected to state that no dropout is used, and the corresponding
sentence in Section 2.1.5 has been removed. We regret the error and are grateful it was
caught. No result changes: the trained networks, and therefore every number in the
paper, are unaffected — only the description was wrong.

---

## Comment 6 — Normalization and the mapping to the optimization domain

> *Is any normalization applied to the input coordinates, latent vectors, or target SDF
> values… the mapping between the training normalization and the optimization setup
> should be made explicit.*

We have added an explicit description, and we thank the reviewer for the question,
because writing it out revealed a point we had not previously appreciated.

There are **two** stages, and only the first is isotropic:

- **Stage A (physical → normalized).** A single uniform scale factor 2/L, with
  L = max extent of the design box, applied to all three axes. The design box is
  therefore *not* stretched to fill [−1,1]³; only its longest axis reaches ±1.
- **Stage B (normalized → decoder input).** The transformation (10) rescales **each axis
  independently** so that every tile spans [−1,1] in the decoder's input.

The net effect is that Stage B undoes the isotropy that Stage A establishes. For
Experiment 1 the design domain is 3 × 1.6 × 1.6 with a 2×2×2 tiling, so a single tile is
physically 1.5 × 0.8 × 0.8 and is mapped onto the cube [−1,1]³: **the decoder sees
distances along x compressed by a factor 1.875 relative to y and z** (2.0 for the
cube-with-cylinders domain, which is 3.2 × 1.6 × 1.6).

Since the decoder was trained on primitives in an isotropic box Ω_box = [−1,1]³, it is
queried slightly off its training distribution during optimization. We now state this
explicitly rather than leave it implicit, and note it as a limitation: choosing tilings
whose tiles are close to cubic keeps the decoder nearer its training regime, and is a
consideration when selecting the control lattice.

Beyond this, no normalization is applied to the latent vectors, and the only processing
of target SDF values is the clamping at δ = 0.1 in the training and reconstruction
losses.

**Correction.** In doing this we found that Section 3.2.1 states a design domain of
3 × 1.5 × 1.5. The value used in all computations is **3 × 1.6 × 1.6**. The text has been
corrected; no result is affected.

---

## Comment 7 — float32 versus float64 in the composed gradient

> *How important is this precision mismatch… particularly for the accuracy of the
> composed gradient in Equation (21) and for the convergence criterion (29) with
> ε_D = 2 × 10⁻⁴?*

We have quantified this. A new appendix reports a finite-difference verification of the
differentiable geometry chain d → λ(x) → s, carried out in both single and double
precision on the same reconstructed configuration (20 randomly chosen design variables,
central differences, eight step sizes).

| step h | float32 | float64 |
|---|---|---|
| 10⁻¹ | 1.677·10⁻² | 1.676·10⁻² |
| 10⁻² | 1.723·10⁻³ | 1.715·10⁻³ |
| 10⁻³ | 1.242·10⁻³ | 1.207·10⁻³ |
| 3·10⁻⁴ | 1.418·10⁻³ | 8.044·10⁻⁴ |
| 10⁻⁴ | 5.040·10⁻³ | 5.557·10⁻⁴ |
| 10⁻⁵ | 3.909·10⁻² | 2.231·10⁻⁴ |

The two are **indistinguishable for h ≥ 10⁻³**, where truncation error dominates. They
separate only below that: single precision reaches a floor of about 1.2·10⁻³ relative
and then degrades as round-off takes over, while double precision continues to converge.
The analytic gradient is therefore verified to roughly three significant digits in
float32.

We emphasise what this comparison does and does not vary. In the double-precision run
the decoder weights, the latent control vectors, the spline evaluation and the sample
coordinates are all promoted to float64, so the entire evaluation and its
differentiation are carried out in double precision. The weights themselves, however,
were *trained* in single precision, so promoting them widens the same numerical values
rather than recovering additional information. Both columns above therefore
differentiate the **identical mathematical function**, and the comparison isolates the
arithmetic precision of evaluating and differentiating it — which is exactly the
question the reviewer raises about Equation (21) — rather than comparing two models of
different accuracy.

This is not a limitation in practice, because the MMA move limit is max_step = 0.02 —
two orders of magnitude above the step size at which precision begins to matter. We now
state this.

**On ε_D specifically**, we note that it is a criterion on the *objective*, not on the
gradient: it bounds |J_m − J_{m−1}|/|J_0|. The objective is read from the flow solver's
output at six significant digits, a relative representation error of order 10⁻⁶ — two
orders of magnitude below ε_D. Arithmetic precision is therefore not the binding
uncertainty on the convergence test; the discretization error of J is, and that is
quantified in our response to Comment 8.

**An additional observation.** When we extended the same test through the surface
extraction to the volume functional V(d), the finite-difference and analytic gradients
agreed only to 3–4% and, crucially, **did not converge as h was reduced**. Monitoring the
extracted mesh showed why: the FlexiCubes connectivity changes under perturbation, so
V(d) is only *piecewise* smooth in the design variables. This does not affect the
gradients used by the optimizer, which are obtained by automatic differentiation at a
fixed connectivity, but it means the mapping from design variables to the extracted
surface is not globally smooth. We now state this as a property of the representation
and a caveat for gradient-based methods that assume smoothness.

---

## Comment 8 — Mesh resolution, residual levels, and mesh independence

> *What is the accuracy of the converged steady-state solution, in terms of mesh
> resolution, residual levels at convergence, and any mesh-independence check for the
> drag value? … an indication of the numerical uncertainty of the objective would make
> the FFD comparison in Table 4 (16.7% versus 15.7%) more persuasive.*

We have carried out the requested study and added it as a new subsection. The three
geometries behind the comparison — the initial cube-with-cylinders, the optimized
neural-SDF design and the optimized FFD (7×7×7) design — were each re-analysed at three
surface refinement levels. Level 6 is the setting used for every result in the
manuscript.

| geometry | L5 cells / drag | L6 cells / drag | L7 cells / drag |
|---|---|---|---|
| initial | 77 280 / 36.9013 | 275 951 / 37.1499 | 1 416 870 / 37.2557 |
| optimized, neural SDF | 70 831 / 30.9071 | 204 951 / 30.9404 | 1 056 701 / 30.9999 |
| optimized, FFD 7×7×7 | 71 289 / 31.0600 | 207 264 / 31.0976 | 1 079 245 / 31.1562 |

Relative change from L6 to L7: **0.285%, 0.192%, 0.188%** respectively. All meshes pass
`checkMesh`; maximum non-orthogonality is 41–63° and maximum skewness below 1.3.

**Numerical uncertainty and the comparison.** On a common mesh the neural-SDF design has
lower drag than the FFD design at every refinement level, by a margin that is stable
across levels: **0.492% (L5), 0.506% (L6), 0.502% (L7)**. This margin exceeds the
discretization uncertainty of either geometry, and because refinement moves both designs
in the same direction by nearly the same amount, the *difference* between them is better
converged than either absolute value. We therefore consider the ranking robust.

**However, the reviewer's scepticism about the headline figures is justified, and we have
corrected the manuscript accordingly.** In re-examining the comparison we found that the
two optimizations did not start from the same baseline. The neural-SDF run begins from
the *reconstructed* surface (J₀ = 37.1499), whereas the FFD runs begin from the *CAD*
surface (J₀ = 36.8845), which is 0.72% lower; the two pipelines also used different
surface-extraction resolutions. The reported reductions are thus measured from different
starting points, and part of the 16.71% versus 15.69% gap reflects that offset rather
than optimizer performance.

We have therefore rewritten the comparison to report **absolute final drag on a common
mesh** as the primary evidence, retaining the percentage reductions but stating the
baselines explicitly. The conclusion is unchanged in direction — the proposed method
finds the better design — but the honest margin is approximately half a percent rather
than a full percentage point. We are grateful to the reviewer, since this materially
improves the rigour of the claim.

**Residual levels.** The primal solver meets its residual control of 10⁻⁵, converging in
606–924 SIMPLE iterations across the nine runs. The adjoint solver, however, **reaches
its iteration cap of 1000 without meeting the same tolerance** in all nine runs. We now
report this explicitly rather than leaving it unstated. It affects the sensitivity field
— that is, the descent direction — and not the drag values tabulated above, which are
obtained from the converged primal solution. An inexact descent direction is consistent
with, and a plausible partial explanation of, the slow tail of the convergence histories
in Figures 14 and 20. We have added this to the limitations discussion.

---

## Comment 9 — Citation for the low-Reynolds-number argument

> *The statement that the observed evolution "is consistent with the expected
> drag-reducing effect of smoother shapes in low-Reynolds-number external flow" would
> benefit from a citation.*

Agreed; a citation has been added.

> **[AUTHOR: please select and verify the reference.** The natural citation is the
> classical work on optimum profiles in Stokes flow — Pironneau, *On optimum profiles in
> Stokes flow*, Journal of Fluid Mechanics **59** (1973) 117–128 — which derives the
> drag-minimizing body at vanishing Reynolds number and is directly on point. Bourot,
> *On the numerical computation of the optimum profile in Stokes flow*, JFM **65** (1974)
> 513–515, is a companion. I have not been able to verify these details against the
> published records from here, so please confirm volume, pages and year before
> submission, and satisfy yourself that the reference says what we claim.]

---

## Comment 10 — Figure 18, coordinate axes and cutting planes

> *Please add coordinate axes, or an inset indicating the cutting planes, so that the
> reader can locate "view 1" and "view 2" relative to the flow direction and the body.*

Figure 18 has been redrawn. Both panels now carry labelled coordinate axes, an arrow
indicating the free-stream direction, and an accompanying three-dimensional sketch
showing where each cutting plane intersects the body. The two cuts are the x–y plane
(normal z) and the x–z plane (normal y); both contain the flow direction, which is now
stated in the caption.

Both geometries are sectioned with the same plane origin, taken as the centroid of the
optimized neural-SDF body, so that the overlay is meaningful. We note in the caption
that the two centroids agree to within 7·10⁻³, i.e. the centroid constraint (27) is
satisfied by both designs.

---

## Comment 11 — Cost of the initialization stage

> *It would be worth adding… a comment on the cost of the initialization stage
> (reconstructing the initial CAD geometry in latent space) relative to the online
> optimization loop, since this is the one step a practitioner cannot avoid.*

We have measured this and added it to the discussion.

| case | initialization | per optimization iteration | share of total |
|---|---|---|---|
| cube | 169 s | 179.6 s | 3.0% |
| cube with cylinders | 238 s | 199.1 s | 3.5% |
| perforated cube | 151 s | 173.4 s | 4.0% |

The initialization stage costs roughly **one optimization iteration**, that is 3–4% of
the total, and is dominated entirely by the latent-code fit; loading the decoder and
constructing the B-spline parametrization together take under half a second. Since the
online loop is bounded by the flow and adjoint solutions, the unavoidable initialization
overhead is negligible in any application where the forward analysis is non-trivial.

> **[AUTHOR: hardware caveat — decide whether to state it or re-time.]**
>
> The initialization timings above were measured fresh on an NVIDIA RTX 4000 SFF Ada,
> whereas the training reported in Table 2 used an RTX 4090. The per-iteration figures
> come from the original optimization runs, whose hardware is **not recorded**
> (`config_log.json` stores only `device: "cuda"`, and the archived Slurm logs from that
> period carry no node information).
>
> Two reasons this does not weaken the conclusion, and both could simply be stated:
> the online loop is bounded by the flow and adjoint solutions, which run on CPU, so the
> GPU is largely irrelevant to the per-iteration cost; and the RTX 4000 SFF Ada is
> substantially slower than the cluster's RTX 4090s, so the initialization share
> reported here is if anything an **upper bound** — on the training hardware it would be
> a smaller fraction still.
>
> If you would prefer exact numbers on a single machine, re-timing all three cases on
> `mp` is a ten-minute job and I can prepare it.

---

## Minor and editorial

**Running head.** The placeholder "FREINBERGER, KEY, KOFLER, ELGETI AND LAST-NAME" has
been replaced with the correct short author list. We apologise for the oversight.

**Equation (24), factor of two.** We have adopted the standard form

  σ(u, p) = −p I + 2 ρ ν · ½(∇u + ∇uᵀ),

and corrected the accompanying text, which previously referred to ∇u + ∇uᵀ as "the
symmetric part of the velocity gradient". The reviewer is right that this was loose: the
symmetric part is ½(∇u + ∇uᵀ). The formulation is unchanged; only its presentation is
corrected, so that no reader mistakes it for a missing coefficient.

**"criteria" → "criterion".** Corrected in Section 2.2, Step 5, and in the following
sentence.

**Definition of "epoch", training time and hardware.** Table 2 now carries a footnote
defining the term. One epoch is a full pass over the N = 100 training *scenes*; each
scene contributes one optimizer step on a freshly drawn random subset of 4096 of its
samples. Ten thousand epochs therefore correspond to 10⁶ optimizer steps, not to 10⁴
passes over all 1.1 × 10⁶ samples per scene. We also now state that the learning-rate
schedule has a decay interval of 12 000 epochs, so the learning rate does not in fact
decay during the 10 000 epochs of training.

Training took **4 h 25 min, 4 h 26 min and 4 h 27 min** for d_lat = 8, 16 and 32
respectively, each on a single **NVIDIA GeForce RTX 4090 (24 GB)**.

**Table 3, design variables.** Table 3 now includes the number of latent design
variables implied by each tiling, computed as d_lat · Π(t_i + p_i). For reference, the
flow-channel case at 1×8×8 uses 5184 design variables, and the shape reconstructions in
Figure 11 use 27 040 (propeller and dog) and 37 856 (rim).

---

## Corrections made on our own initiative

In the course of addressing the comments above we identified three further errors in the
submitted manuscript. None changes any conclusion, but we list them for completeness and
transparency.

1. **Table 4, FFD 5×5×5 iteration count.** The submitted table reports convergence in 46
   iterations. The archived run performed **59** design updates, reaching its iteration
   limit without satisfying the convergence criterion of Equation (29); at iteration 46
   the drag reduction was 13.84%, and the reported 14.20% is the value at the end of the
   run. The iteration count in Table 4 has been corrected and the caption now states
   that this run did not meet the convergence criterion. The other entries in Table 4
   reproduce exactly.

2. **Experiment 1 drag reduction.** Section 3.2.1 describes the reduction as
   "approximately 15%". The precise value from the archived run is **14.49%**, and the
   text now states it.

3. **Design domain in Section 3.2.1** — 3 × 1.6 × 1.6, not 3 × 1.5 × 1.5, as described
   under Comment 6.

Additionally, and as described under Comment 8, we have restated the FFD comparison in
terms of absolute drag on a common mesh, because the two optimizations were started from
different baselines.

---

## Reproducibility

All analyses reported in this response were performed with the archived code and data
and are reproducible. The revision adds a set of self-contained scripts covering the
gradient probe of Comment 4, the normalization analysis of Comment 6, the
finite-difference verification of Comment 7, the mesh-independence study of Comment 8,
the redrawn Figure 18, and the recomputation of Tables 3 and 4 from the archived run
records. Each carries a self-check that reproduces a published value before emitting new
numbers.

> **[AUTHOR: decide whether these scripts and the associated result files are to be
> included in the updated Zenodo deposit, and whether the raw solver logs from the
> mesh-independence study should be deposited with them.]**

We thank the reviewer again for a report that improved the paper.

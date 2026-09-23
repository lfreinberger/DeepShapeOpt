"""Minimum streamwise length of thin solid webs (level-set penalty)."""
from __future__ import annotations

import math

import torch

from .ks import KSStream


def min_steg_length_penalty_sdf(
    lattice_struct, frame, param, flow_dir,
    thickness_threshold_mm, min_length_mm,
    grid_spacing=0.25, n_dirs=8, ray_step_mm=None, tau_mm=None,
    slab_margin=0.5, weight=1.0, exclude_region=None,
    length_mode="thin_band",
    formulation="penalty", ks_rho=50.0,
):
    """SDF penalty for THIN solid webs (stegs) that are too SHORT in the flow direction -- smooth in latents.

    Motivation: in internal-flow shape optimization the optimizer can carve the (intended,
    tapering) fluid distribution channels so aggressively that the SOLID webs left between them
    taper to thin, acute wedge tips. Those break off under the flow loading. They slip past both
    existing geometry constraints: :func:`taper_penalty_sdf` (the walls are acute -- nearly
    parallel to the flow -- so ``|n.d|`` barely violates) and :func:`min_wall_thickness_penalty_sdf`
    (whose medial gate ``relu(grad_cutoff-|grad phi|)`` is tuned for parallel-wall pinch-off and
    deliberately exempts tapering wedges). The criterion that separates the bad webs from
    everything legitimate is the CONJUNCTION ``thin AND short-in-x``: thick material (bulk walls)
    is fine no matter how short in x; thin material is fine as long as it is long enough in x
    (long conical inlets, long thin ribs). Only solid that is thin in cross-section AND short
    along the flow is forbidden.

    Sign convention (raw lattice, ``fluid_side: inside``): fluid ``phi<0``, solid wall ``phi>0``.
    Working on ``phi>0`` only means the tapering fluid channels are ignored automatically.

    Two smooth measures, both differentiable functions of the latents, evaluated on a FIXED
    normalized-coordinate grid (no remeshing), over candidate thin-solid points (``0<phi<phi_hi``):

    * Cross-flow thickness ``t_cf`` -- the solid chord ALONG THE SURFACE NORMAL (``grad phi``),
      projected into the plane perpendicular to the flow ``d``. The normal is by construction
      perpendicular to the local surface, so its chord spans the feature thickness (near-dist +
      far-dist) regardless of how off-centre the candidate sits in the band. (A min over arbitrary
      in-plane directions is WRONG: for a near-surface point one direction always points at the
      nearest surface, so the min collapses to ~2*phi and every skin point reads as thin -- an
      over-flagging bug that flagged ~100% of a *valid* part.) Projecting out the flow component keeps
      a thick-but-streamwise-short feature (a blunt cap, the taper constraint's job) from being
      misread as thin. This is a TRUE thickness -- large for thick bodies incl. their skin, small
      only for genuinely thin webs -- and like a medial measure it also catches acute wedges. The
      thin weight is ``tw = relu(1 - t_cf/t_min)`` in [0,1]. (``n_dirs`` is retained for config
      compatibility but no longer used.)
    * Streamwise length ``Lx`` -- selectable via ``length_mode`` (both via a running EXACT min, a.e.
      differentiable, so the length does not decay along a continuous run -- an earlier smooth-min
      with ``smin_eps=0.1`` drifted down ``~0.05*sqrt(k)`` per step even in perfect solid, so an
      infinitely-backed point read Lx ~ 8mm < L_min=10mm and EVERY thin point carried a false
      violation floor; ``viol = relu(1 - Lx/L_min)``):
        - ``"thin_band"`` (default): how far the THIN-solid band extends along ``+-d`` (both
          directions); indicator ``b = sigmoid(phi/tau)*sigmoid((h-phi)/tau)`` (solid AND near a
          surface), running min gated at ``b_thr``. Sign-independent. Measures the web's own extent.
        - ``"downstream_reach"`` (Option B): contiguous SOLID run DOWNSTREAM (along ``+d``, the flow
          direction, toward the outlet) until the first fluid. Tracks the running min of ``phi``
          ITSELF and gates it in phi units, ``Lx += ray_step * sigmoid((min_phi - phi_stop)/tau_stop)``
          with ``phi_stop = -2*tau`` (a graze past a surface, min_phi ~ 0, must NOT end the run --
          only a genuine fluid crossing does), ``tau_stop = 0.5*tau``: a thin web (phi small but
          > 0) still reads fully solid -- an occupancy ``sigmoid(phi/tau)`` would sit mid-range
          inside thin material and conflate THIN with SHORT (long thin webs, which are allowed,
          read short). The ray marches 15% past L_min so a healthy backed run saturates viol to 0.
          Penalizes thin material with a free downstream end (a cantilever); thin material backed by
          solid downstream (into the bulk) is fine. DIRECTIONAL -- ``flow_dir`` must point
          downstream. The gradient routes to the argmin sample = the first fluid crossing, exactly
          where "extend solid downstream / thicken" applies. OUT-OF-BOX = AIR: the part ENDS at the
          outlet (nothing but air past it), so downstream ray samples leaving the design box are
          overridden to a hard air value (``phi_air``) and the run stops at the box face. (The
          lattice itself returns positive distance-to-box there -- "solid" -- which would falsely
          count the void beyond the outlet as support.) Consequence: thin material near the outlet
          is supported only by what lies between it and the outlet face; legitimately thin outlet
          regions must be exempted via ``exclude_region`` (or accepted as flagged).
      The ``tw`` gate restricts the penalty to genuinely thin points in both modes.

        W = weight * sum_k wphi_k * tw_k * viol_k^2 * dV / (sum_k wphi_k * tw_k * dV)
                                                       (over thin-solid candidates)

    i.e. the thin-weighted MEAN squared length violation over the thin-candidate band
    (``wphi`` = the phi-window) -- intensive (O(violation^2) in [0,~1], grid- and
    candidate-count-independent), not the extensive sum; the denominator (detached) rescales
    value and gradient by the same constant. The denominator is floored at 1% of the FULL
    phi-band measure (``sum_k wphi_k``) so W cannot jump UP through mean-concentration when the
    optimizer thickens most webs and the thin-eligible set shrinks to a handful of points.
    With ``t_min = scale*thickness_threshold_mm``, ``h = t_min/2``, ``L_min = scale*min_length_mm``
    (normalized units; ``scale = 2/L``). ``exclude_region`` is a physical-mm box mapped to normalized
    coordinates (e.g. the cylinder->outlet shoulder).

    ``sqrt(W/weight)*min_length_mm`` is the interpretable diagnostic: the RMS shortfall (mm) of
    the flagged thin material w.r.t. the required streamwise length.

    ``formulation="ks_margin"`` (constraint mode; mirrors :func:`undercut_penalty_sdf`):
    instead of the mean penalty W, return the SIGNED worst-case streamwise-shortfall margin
    ``M = KS-max_k s_k`` with ``s = 1 - Lx/L_min`` (positive = too short; ``s <= 1`` since
    ``Lx >= 0``, so the KS exponents are bounded) -- a smooth weighted max (logsumexp,
    sharpness ``ks_rho``, streamed chunk-wise via :class:`KSStream`) over the thin
    candidates, weights ``wphi*tw*dV`` DETACHED and normalized. The constraint is
    ``M <= budget`` in shortfall-fraction units (``M * min_length_mm`` = worst-case
    shortfall in mm). Vs. the mean penalty: real negative slack when feasible (saturating
    at ~-0.15 from the 15% ray-march overshoot; ~-1.3 in thin_band mode), a
    never-vanishing softmax gradient at the boundary, worst-case semantics (one deep
    violation cannot hide in a mean over many healthy points), and no mean-concentration
    artifact (the 1%-floor below is penalty-only). Detaching the weights is deliberate
    POLICY, not KS bookkeeping: the thin gate ``tw`` carries the "thicken the web"
    remedy, so with ``w`` detached the only gradient path is through ``Lx`` -- the
    optimizer can fix a violation ONLY by lengthening/backing the web, never by
    fattening it (material that thickens anyway still fades out of the weights between
    iterations, so healthy webs are not fought over). ``weight`` is ignored (a signed
    margin is not weighted); no thin candidate anywhere returns the fully-feasible
    sentinel ``-1`` (undercut's empty-band convention). The debug scalars gain
    ``ks_weight`` (normalized aggregation weight per thin point) -- scripts/check_min_steg_fd.py
    freezes exactly these for its FD gate.

    Returns ``(W_value, dW_param, n_cand, n_flagged, thin_pts_phys, thin_scalars, thin_normal)``
    where the point cloud is the WHOLE thin-candidate set (``tw > 0``), so a web-clean iteration
    still yields a debug VTP (with ``viol ~ 0`` everywhere) rather than no file. ``thin_scalars``
    is a dict of per-point diagnostics (``Lx_mm``, ``viol``, ``t_cf_mm``) matching the
    points/vectors/scalars signature of the debug VTP writer -- threshold on ``viol > 0.05`` in
    ParaView to isolate the offenders. ``n_flagged`` counts thin AND short (``viol > 0.05``; the
    tolerance keeps epsilon violations from counting).
    """
    from DeepSDFStruct.utils import with_float32_lattice

    if formulation not in ("penalty", "ks_margin"):
        raise ValueError(
            f"min_steg_length_penalty_sdf formulation must be 'penalty' or 'ks_margin', "
            f"got '{formulation}'"
        )

    device = param.device
    box_norm = frame.box_norm.to(device=device, dtype=torch.float32)
    scale = float(frame.scale)
    sp = scale * float(grid_spacing)                       # grid spacing (normalized units)
    t_min = scale * float(thickness_threshold_mm)          # full cross-thickness threshold
    h = 0.5 * t_min                                        # half-thickness (phi candidate band)
    phi_hi = h * (1.0 + float(slab_margin))                # candidate upper bound on phi
    L_min = scale * float(min_length_mm)                   # required streamwise extent
    ray_step = scale * (float(ray_step_mm) if ray_step_mm is not None else float(grid_spacing))
    tau = scale * (float(tau_mm) if tau_mm is not None else 0.4 * float(grid_spacing))
    fd = 0.25 * sp                                         # central-difference step (normal glyph)
    lo = box_norm[0]
    hi = box_norm[1]
    inset = sp + fd
    n_cross = max(3, int(math.ceil(3.0 * t_min / ray_step)))  # reach ~3x threshold so thick walls read thick
    # March 15% past L_min: the stop gate is < 1 by a few % wherever the ray merely grazes a
    # surface (min_phi ~ 0), so without headroom a fully-backed run tops out at ~0.95*L_min and
    # every healthy thin point carries a small false violation. With the overshoot a healthy run
    # saturates Lx past L_min and viol = relu(1 - Lx/L_min) lands at exactly 0.
    n_axial = max(2, int(math.ceil(1.15 * L_min / ray_step)))  # steps to cover the length threshold
    b_thr = 0.5                                            # thin_band: "still thin-solid" gate level
    tau_gate = 0.12                                        # thin_band: gate softness on the running min
    # downstream_reach stop level: clearly BELOW "grazing the surface". Candidates live in the
    # skin band, so rays along a wavy web routinely pass within ~0.1mm of a surface (min_phi ~ 0);
    # with phi_stop = -tau such a graze gated only 0.88-0.95 per step and healthy long webs read
    # Lx ~ 8.6-9.5mm < L_min (marginally flagged all over the part). At -2*tau a graze reads ~0.98
    # and only a genuine fluid crossing (phi < -2*tau within one step) kills the run.
    phi_stop = -2.0 * tau                                  # downstream_reach: solid-run stop level (phi units)
    tau_stop = 0.5 * tau                                   # downstream_reach: gate softness (phi units)
    # Hard air value for downstream ray samples that leave the design box: the part ENDS at the
    # outlet, so past the box face there is no support -- gate reads ~0 (sigmoid(-4)) and the
    # run stops. 4 gate-widths below phi_stop; detached constant (the box is fixed geometry).
    phi_air = phi_stop - 4.0 * tau_stop
    viol_flag_tol = 0.05                                   # debug flag threshold on viol (not the penalty)
    CHUNK_GRID = 262144                                    # no-grad full-box query chunk (memory bound)
    CHUNK_CAND = 8192                                      # grad ray-eval chunk over candidates (memory bound)

    # Flow direction (normalized) and an orthonormal cross-plane basis (u, v) perpendicular to it.
    d = torch.as_tensor(flow_dir, dtype=torch.float32, device=device)
    d = d / d.norm().clamp_min(1e-20)
    ex_fd = torch.tensor([fd, 0.0, 0.0], device=device)    # central-difference basis for grad phi
    ey_fd = torch.tensor([0.0, fd, 0.0], device=device)
    ez_fd = torch.tensor([0.0, 0.0, fd], device=device)

    # exclude_region: a single [[lo],[hi]] mm box, OR a list of such boxes (so e.g. the inlet
    # shoulder AND a legitimately thin outlet region can both be exempted).
    excl_boxes = []
    if exclude_region is not None:
        boxes = exclude_region if hasattr(exclude_region[0][0], "__len__") else [exclude_region]
        for bx in boxes:
            elo = frame.to_norm(torch.as_tensor(bx[0], dtype=torch.float32, device=device))
            ehi = frame.to_norm(torch.as_tensor(bx[1], dtype=torch.float32, device=device))
            excl_boxes.append((torch.minimum(elo, ehi), torch.maximum(elo, ehi)))

    def _query(x):
        return lattice_struct(x).reshape(-1)

    def _steg_terms(Xb):
        """Grad-connected per-candidate terms for a chunk Xb: phi-window, thin weight, violation, Lx."""
        nc = Xb.shape[0]
        # Compact-support phi-window (raised cosine) over the candidate band (0, phi_hi): it -> 0 with
        # zero slope at BOTH edges, so a point entering/leaving the hard band as the latents move
        # contributes ~0 at the boundary -> the band selection adds no gradient discontinuity (the
        # integrand tw*viol^2 is otherwise nonzero at the edges, unlike taper's delta_eps).
        phi = _query(Xb)
        wphi = torch.where(
            (phi > 0.0) & (phi < phi_hi),
            0.5 * (1.0 - torch.cos(2.0 * math.pi * phi / phi_hi)),
            torch.zeros_like(phi),
        )

        # (a) cross-flow thickness t_cf = solid chord along the surface normal, projected into the
        # plane perpendicular to the flow. The normal n = grad phi is by construction perpendicular
        # to the local surface, so the chord along it spans the feature THICKNESS (near-dist +
        # far-dist) independent of how off-centre the candidate sits in the band. (A min over
        # ARBITRARY directions is wrong: for a near-surface point one direction always points at the
        # nearest surface, so the min collapses to ~2*phi and EVERY skin point reads as thin -- the
        # over-flagging bug.) Projecting out the flow component restricts "thin" to the cross-section,
        # so a thick-but-streamwise-short feature (a blunt cap) is NOT misread as thin (its cross-flow
        # chord is large); such caps are the taper constraint's job, not this one.
        gpx = (_query(Xb + ex_fd) - _query(Xb - ex_fd)) / (2 * fd)
        gpy = (_query(Xb + ey_fd) - _query(Xb - ey_fd)) / (2 * fd)
        gpz = (_query(Xb + ez_fd) - _query(Xb - ez_fd)) / (2 * fd)
        gd = gpx * d[0] + gpy * d[1] + gpz * d[2]                 # normal . flow
        nx, ny, nz = gpx - gd * d[0], gpy - gd * d[1], gpz - gd * d[2]   # normal projected to cross-plane
        gnp = torch.sqrt(nx ** 2 + ny ** 2 + nz ** 2).clamp_min(1e-9)
        e_thin = torch.stack([nx / gnp, ny / gnp, nz / gnp], dim=1)     # unit thin direction (in y-z)
        t_cf = torch.zeros(nc, device=device)
        for sgn in (1.0, -1.0):
            run = torch.ones(nc, device=device)                  # contiguous-solid cumulative product
            for k in range(1, n_cross + 1):
                run = run * torch.sigmoid(_query(Xb + (sgn * k * ray_step) * e_thin) / tau)
                t_cf = t_cf + run * ray_step
        tw = torch.clamp(1.0 - t_cf / t_min, min=0.0)            # thin weight in [0,1]

        # (b) streamwise length measure -- two selectable modes (length_mode). Both accumulate a
        # contiguous run with a *running EXACT min* (torch.minimum, a.e. differentiable; NOT a
        # cumulative product, which decays geometrically, and NOT a smooth-min, whose eps bias
        # drifts down ~eps/2*sqrt(k) per step even in perfect solid -- with the previous
        # smin_eps=0.1 an infinitely-backed point read Lx ~ 8mm < L_min=10mm, so every thin point
        # carried a false violation floor). The running min stays exactly flat along a continuous
        # run and only collapses at the first genuine dip, so the length faithfully reaches L_min
        # and is small only for a genuinely short feature.
        if length_mode == "downstream_reach":
            # Option B: contiguous SOLID run DOWNSTREAM (along +d = flow direction = toward the outlet)
            # until the first FLUID. Penalizes thin material with a FREE downstream end (a cantilever
            # the flow rips off); thin material backed by solid downstream (merging into the bulk or
            # reaching the outlet -- or leaving the design box, where the lattice SDF reads positive =
            # attached to the fixed outer geometry) is fine. Track the running min of phi ITSELF and gate
            # in phi units at phi_stop = -tau (slightly fluid-tolerant): inside a thin web phi is
            # small (< h ~ 2.5*tau) but positive, so the gate still reads ~1 -- an occupancy
            # sigmoid(phi/tau) would sit mid-range there and misread LONG thin webs (allowed) as
            # short. Stopping is a single event (first fluid crossing), and torch.minimum routes the
            # gradient to exactly that sample ("extend solid downstream OR thicken"). DIRECTIONAL:
            # flow_dir must point downstream (set flow_direction to the true flow sign).
            Lx = torch.zeros(nc, device=device)
            m = torch.full((nc,), float("inf"), device=device)  # running min of phi along the ray
            for k in range(1, n_axial + 1):
                Pk = Xb + (k * ray_step) * d
                phi_k = _query(Pk)
                # Past the design box there is only AIR (the part ends at the outlet): override the
                # lattice's positive distance-to-box reading so the run stops at the box face.
                outside = ((Pk < lo) | (Pk > hi)).any(dim=1)
                phi_k = torch.where(outside, torch.full_like(phi_k, phi_air), phi_k)
                m = torch.minimum(m, phi_k)
                Lx = Lx + ray_step * torch.sigmoid((m - phi_stop) / tau_stop)
            viol = torch.clamp(1.0 - Lx / L_min, min=0.0)
        else:
            # "thin_band" (default): how far the THIN-solid band extends along +-d (both directions).
            # thin-solid indicator b = sigmoid(phi/tau)*sigmoid((h-phi)/tau) (~1 inside a thin web,
            # -> 0 at the fluid edge phi->0 and at the thickening edge phi->h). Sign-independent.
            Lx = torch.zeros(nc, device=device)
            for sgn in (1.0, -1.0):
                m = torch.ones(nc, device=device)       # running min of b along the ray
                for k in range(1, n_axial + 1):
                    phi_k = _query(Xb + (sgn * k * ray_step) * d)
                    bk = torch.sigmoid(phi_k / tau) * torch.sigmoid((h - phi_k) / tau)
                    m = torch.minimum(m, bk)
                    Lx = Lx + ray_step * torch.sigmoid((m - b_thr) / tau_gate)
            viol = torch.clamp(1.0 - Lx / L_min, min=0.0)
        return wphi, tw, viol, Lx, t_cf

    def _compute(_bounds_f32):
        axes = []
        for i in range(3):
            n_i = max(2, int(round((hi[i].item() - lo[i].item() - 2 * inset) / sp)) + 1)
            axes.append(torch.linspace(lo[i].item() + inset, hi[i].item() - inset, n_i, device=device))
        gx, gy, gz = torch.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
        grid = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=1).float()
        dV = sp ** 3

        # Pass 1 (no grad, chunked): candidate = potentially-thin solid points (small phi, solid),
        # outside exclude_region. Chunked so the full-box query never allocates the whole grid at once.
        keep_parts = []
        with torch.no_grad():
            for gi in range(0, grid.shape[0], CHUNK_GRID):
                gch = grid[gi:gi + CHUNK_GRID]
                p0 = _query(gch)
                kk = (p0 > 0.0) & (p0 < phi_hi)
                if excl_boxes:
                    inside = torch.zeros(gch.shape[0], dtype=torch.bool, device=device)
                    for blo, bhi in excl_boxes:
                        inside = inside | ((gch >= blo) & (gch <= bhi)).all(dim=1)
                    kk = kk & (~inside)
                keep_parts.append(kk)
        Xb_all = grid[torch.cat(keep_parts)]
        n_cand = int(Xb_all.shape[0])
        if n_cand == 0:
            # ks_margin: no candidates = fully feasible margin (-1); penalty: 0.
            empty_val = -1.0 if formulation == "ks_margin" else 0.0
            return (
                torch.tensor(empty_val, device=device, dtype=torch.float32),
                torch.zeros_like(param), 0, 0,
                None, None, None, None, None, None,
            )

        # Pass 2 (grad, chunked): accumulate one candidate chunk at a time so the autograd
        # graph for the ray queries stays bounded no matter how many candidates there are.
        # penalty: extensive numerator + measures for the intensive mean below. ks_margin:
        # streaming KS/logsumexp of the signed margin (KSStream), same peak memory.
        W_num = 0.0
        g_acc = torch.zeros_like(param)
        measure = 0.0       # penalty: detached eligibility measure (thin*phi-window integral)
        band_measure = 0.0  # penalty: detached FULL band measure (phi-window integral)
        ks = KSStream(param) if formulation == "ks_margin" else None
        ks_wsum = 0.0       # ks_margin: total detached aggregation weight sum(wphi*tw)*dV
        f_pts_l, f_Lx_l, f_viol_l, f_tcf_l, f_w_l, n_flag = [], [], [], [], [], 0
        for ci in range(0, n_cand, CHUNK_CAND):
            Xb = Xb_all[ci:ci + CHUNK_CAND]
            wphi, tw, viol, Lx, t_cf = _steg_terms(Xb)
            if ks is not None:
                # Signed worst-case margin: smooth weighted max of the streamwise-shortfall
                # margin s = 1 - Lx/L_min over the thin candidates, weights = the DETACHED
                # eligibility measure wphi*tw*dV. Detaching tw is deliberate policy (see
                # docstring): the only gradient path left is through Lx, so the optimizer
                # fixes a violation ONLY by lengthening/backing the web, never by
                # fattening it. The chunk's exponents feed the streaming logsumexp and the
                # chunk graph is freed immediately (KSStream.add).
                s = 1.0 - Lx / L_min
                w = (wphi * tw).detach() * dV
                pos = w > 0
                if bool(pos.any()):
                    ks.add(float(ks_rho) * s[pos] + torch.log(w[pos]))
                ks_wsum += float(w.sum())
            else:
                num_c = float(weight) * (tw * viol ** 2 * wphi).sum() * dV   # extensive numerator (chunk)
                gc = torch.autograd.grad(num_c, param, retain_graph=False, allow_unused=True)[0]
                if gc is not None:
                    g_acc = g_acc + gc
                W_num += float(num_c.detach())
                measure += float((tw * wphi).sum().detach()) * dV           # eligibility measure (chunk)
                band_measure += float(wphi.sum().detach()) * dV             # full band measure (chunk)
            # Export the WHOLE thin-candidate cloud (tw>0), not just the violating points: a
            # web-clean iteration then still produces a VTP (viol ~ 0 everywhere) instead of no
            # file, so "clean" and "constraint broken" are distinguishable, and the user can watch
            # thin regions before they become short. n_flag keeps counting thin AND short.
            thin = (tw > 0).detach()
            n_flag += int((thin & (viol > viol_flag_tol)).sum())
            if bool(thin.any()):
                f_pts_l.append(Xb[thin].detach())
                f_Lx_l.append((Lx[thin] / scale).detach())
                f_viol_l.append(viol[thin].detach())
                f_tcf_l.append((t_cf[thin] / scale).detach())
                if ks is not None:
                    f_w_l.append(((wphi * tw).detach() * dV)[thin])

        if ks is not None:
            # M = (log sum_k w_k e^{rho s_k} - log sum_k w_k) / rho: the weight-normalized
            # KS max, dM/dparam = dlogZ/dparam / rho (the weight sum is detached). No thin
            # candidate anywhere (all tw = 0): fully feasible sentinel -1.
            logZ, dlogZ = ks.finalize()
            if ks_wsum <= 0.0 or logZ == -math.inf:
                W_val = -1.0
                g_acc = torch.zeros_like(param)
            else:
                W_val = (logZ - math.log(ks_wsum)) / float(ks_rho)
                g_acc = dlogZ / float(ks_rho)
        else:
            # Intensive (mean) form: divide the extensive thin-AND-short violation sum by the
            # eligibility measure (the thin*phi-window weight integral). Detached denominator ->
            # pure rescale (W_num and g_acc by the same constant) independent of grid_spacing and
            # the candidate count, so W lands at O(violation^2) in [0, ~1] and a tuned weight/
            # target transfers across configs. Floor at 1% of the FULL band measure: as the
            # optimizer thickens most webs the eligible set shrinks, and a mean over the few
            # remaining worst points would jump UP while the geometry actually improves.
            M = max(measure, 0.01 * band_measure, 1e-12)
            W_val = W_num / M
            g_acc = g_acc / M

        f_pts = torch.cat(f_pts_l) if f_pts_l else Xb_all[:0]
        f_Lx = torch.cat(f_Lx_l) if f_Lx_l else Xb_all.new_zeros(0)
        f_viol = torch.cat(f_viol_l) if f_viol_l else Xb_all.new_zeros(0)
        f_tcf = torch.cat(f_tcf_l) if f_tcf_l else Xb_all.new_zeros(0)
        f_w = None
        if ks is not None:
            # Normalized per-point KS aggregation weight (sums to 1 over the thin cloud,
            # modulo the wphi=0 band-edge points that carry weight 0).
            f_w = (torch.cat(f_w_l) / max(ks_wsum, 1e-30)) if f_w_l else Xb_all.new_zeros(0)
        with torch.no_grad():
            if f_pts.shape[0] > 0:
                ex = torch.tensor([fd, 0.0, 0.0], device=device)
                ey = torch.tensor([0.0, fd, 0.0], device=device)
                ez = torch.tensor([0.0, 0.0, fd], device=device)
                nrm = torch.stack([
                    _query(f_pts + ex) - _query(f_pts - ex),
                    _query(f_pts + ey) - _query(f_pts - ey),
                    _query(f_pts + ez) - _query(f_pts - ez),
                ], dim=1)
                nrm = nrm / nrm.norm(dim=1, keepdim=True).clamp_min(1e-12)
            else:
                nrm = torch.zeros((0, 3), device=device)
        W_tensor = torch.tensor(W_val, device=device, dtype=torch.float32)
        return W_tensor, g_acc, n_cand, n_flag, f_pts, f_Lx, f_viol, f_tcf, f_w, nrm.detach()

    W, g, n_cand, n_flag, f_pts, f_Lx, f_viol, f_tcf, f_w, f_dir = with_float32_lattice(
        lattice_struct, frame.box_norm, _compute
    )

    pts_phys = scalars = grad_np = None
    if f_pts is not None and f_pts.shape[0] > 0:
        pts_phys = frame.to_phys(f_pts.to(param.dtype)).cpu().numpy()
        scalars = {
            "Lx_mm": f_Lx.to(param.dtype).cpu().numpy(),
            "viol": f_viol.to(param.dtype).cpu().numpy(),
            "t_cf_mm": f_tcf.to(param.dtype).cpu().numpy(),
        }
        if f_w is not None:
            scalars["ks_weight"] = f_w.to(param.dtype).cpu().numpy()
        grad_np = f_dir.to(param.dtype).cpu().numpy()
    return (
        W.detach().to(param.dtype), g.to(param.dtype), n_cand, n_flag,
        pts_phys, scalars, grad_np,
    )

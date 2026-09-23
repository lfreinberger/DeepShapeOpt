"""No-undercut (drawability) penalty: mesh and level-set kernels, outlet silhouette gate."""
from __future__ import annotations

import math

import torch


class OutletSilhouette:
    """Projected-outlet silhouette gate for the undercut ``scope: outside_outlet`` option.

    Holds the 2D outlet polygon (shapely, prepared; may carry holes for interior islands
    of the profile) plus the outlet-plane basis and the draw direction. A point is
    "inside" when its OBLIQUE projection along the draw axis onto the outlet plane lands
    inside the polygon -- i.e. the point is laterally covered by the outlet opening and
    therefore reachable/drawable through it. Undercut penalties exempt inside points;
    only pockets growing OUTSIDE the silhouette (not covered by the opening) are penalized.

    Build via :func:`build_outlet_silhouette`. The gate is geometry-fixed (the outlet lies
    on the locked lattice boundary), detached, and orientation-free -- it only masks which
    faces/points enter a penalty; gradients still flow through the kept ones.
    """

    def __init__(self, poly, origin, normal, axis_u, axis_v, d, d_dot_n, margin, area):
        self.poly = poly
        self.origin = origin
        self.normal = normal
        self.axis_u = axis_u
        self.axis_v = axis_v
        self.d = d
        self.d_dot_n = d_dot_n
        self.margin = margin
        self.area = area

    def inside(self, points):
        """Boolean mask [N] for physical-mm points [N,3]: True = covered by the outlet."""
        import numpy as np
        import shapely

        p = np.asarray(points, dtype=float).reshape(-1, 3)
        # Oblique projection along the draw axis onto the outlet plane:
        # solve (p + t*d - origin) . n = 0.
        t = ((self.origin - p) @ self.normal) / self.d_dot_n
        rel = p + t[:, None] * self.d - self.origin
        return shapely.contains_xy(self.poly, rel @ self.axis_u, rel @ self.axis_v)

def build_outlet_silhouette(outlet_triangles, draw_dir, margin=0.0):
    """Build the projected-outlet silhouette gate from the outlet patch triangles.

    ``outlet_triangles``: (N,3,3) triangle vertex array in the physical (mm) frame --
    e.g. the polyMesh outlet patch faces (sdf_hex) or the classifier's outlet faces
    (snappy). The boundary loops of that face set are projected into the outlet plane and
    assembled into a shapely (Multi)Polygon (holes -- interior islands of the profile -- classified
    by containment depth, exactly the machinery the outlet_interior carve-out uses), so
    profile holes correctly count as OUTSIDE the silhouette.

    ``margin`` (mm) buffers the polygon: positive = more lenient (faces within ``margin``
    of the silhouette rim are still exempt -- use ~1 cell size against membership flicker
    of mesh faces near the rim), negative = stricter. ``draw_dir`` must not be (near-)
    parallel to the outlet plane.
    """
    import numpy as np
    import shapely
    from deepshapeopt.hexmesh.regions2d import (
        all_boundary_loops_from_triangles,
        build_shapely_multipolygon,
        make_plane_basis,
        project_to_plane,
    )

    tris = np.asarray(outlet_triangles, dtype=float)
    if tris.ndim != 3 or tris.shape[0] == 0:
        raise ValueError(
            f"build_outlet_silhouette: expected (N,3,3) outlet triangles, got shape {tris.shape}."
        )
    loops_3d = all_boundary_loops_from_triangles(tris)
    origin, normal, axis_u, axis_v = make_plane_basis(np.vstack(loops_3d), tris)
    loops_2d = [project_to_plane(l, origin, axis_u, axis_v) for l in loops_3d]
    poly = build_shapely_multipolygon(loops_2d)
    if margin:
        poly = poly.buffer(float(margin))
        if poly.is_empty or poly.area <= 0:
            raise ValueError(
                f"build_outlet_silhouette: margin={margin} erased the outlet polygon."
            )
    d = np.asarray(draw_dir, dtype=float)
    d = d / max(float(np.linalg.norm(d)), 1e-20)
    d_dot_n = float(d @ normal)
    if abs(d_dot_n) < 0.2:
        raise ValueError(
            "build_outlet_silhouette: draw_direction is nearly parallel to the outlet "
            f"plane (|d.n| = {abs(d_dot_n):.3f}); the oblique projection is ill-posed."
        )
    shapely.prepare(poly)
    return OutletSilhouette(
        poly=poly, origin=origin, normal=normal, axis_u=axis_u, axis_v=axis_v,
        d=d, d_dot_n=d_dot_n, margin=float(margin), area=float(poly.area),
    )

def undercut_penalty(verts, faces, draw_dir, threshold=0.0, exclude_axial_deg=30.0,
                     exclude_region=None, surface="solid",
                     formulation="penalty", ks_rho=50.0, silhouette=None):
    """Differentiable area-weighted draft/undercut penalty for a draw direction.

    ``exclude_region`` is an optional ``[[lo],[hi]]`` box in the mesh's physical (mm)
    frame -- or a LIST of such boxes -- dropping faces whose centroid lies inside from
    the penalty, its gradient, and the diagnostics (same semantics as
    :func:`undercut_penalty_sdf`; use it for the legitimate
    fluid-opening / shoulder zones).

    ``surface`` tells the global orientation vote what "outward" means -- it cannot be
    inferred from the face set alone: ``"solid"`` (default, historic) for the outer
    surface of a solid piece (outward normals point radially AWAY from the centroid,
    e.g. the snappy current_shape.stl); ``"cavity"`` for an internal channel/cavity
    wall (the solid's outward normal points INTO the channel = radially TOWARD the
    axis, e.g. the sdf_hex design-surface triangles). Feeding a channel wall with
    ``surface="solid"`` silently NEGATES ndotd everywhere -- the penalty, history
    value and flagged faces all become the exact mirror set (verified against the SDF
    normal field: correlation -0.999).

    Manufacturability: the surface must be drawable along ``draw_dir`` (a 3-vector,
    normalized here) without re-entrant features ("Hinterschneidungen"). Only **side-wall**
    faces are considered: faces whose normal is within ``exclude_axial_deg`` of the +/-draw
    axis (channel inlet/outlet openings, caps, and the flare where the channel meets the
    fixed outer geometry) are excluded -- those are openings, not undercuts. On the kept walls,
    with outward normals ``n``, the channel is drawable where ``n·d <= threshold``; an
    undercut is a face with ``n·d > threshold``. ``threshold = -sin(draft angle)`` (0 forbids
    only walls tilting toward the draw direction; a negative threshold additionally requires a
    positive draft so walls taper open).

    ``formulation`` selects what the returned scalar U is:

    * ``"penalty"`` (historic): area-weighted MEAN of relu(n_f·d - threshold)^2 -- one-sided,
      >= 0, and both value and gradient vanish once feasible. Good as a weighted objective
      term; as a hard constraint with target ~0 it has NO feasible interior (no budget).
    * ``"ks_margin"``: SIGNED worst-case drawability margin -- the smooth area-weighted
      maximum (Kreisselmeier-Steinhauser / logsumexp) of ``n·d`` over the kept side walls,
      ``M = (1/rho) * log( sum_f w_f * exp(rho * n_f·d) )`` with detached normalized area
      weights ``w_f = area_f / sum area`` (gradient flows through the normals only, not a
      perverse "shrink the bad face" path). Since sum w = 1, M lies in [min n·d, max n·d]
      subset [-1, 1] (sine units); the constraint is ``M <= threshold`` and M is strictly
      NEGATIVE of the bound when every wall has draft margin -- a real budget, with a
      non-vanishing softmax gradient concentrated on the worst faces. ``ks_rho`` sets the
      smoothing (~1/rho in sine units; 50 resolves ~1-2 deg near zero). Note the
      area-normalized KS UNDER-estimates the hard max by up to ln(1/w_f)/rho for a
      small-area offender -- compensate with draft_angle/margin, not by huge rho.
      With no kept faces M = -1 (fully feasible).

    ``silhouette`` (an :class:`OutletSilhouette` or None) optionally exempts every face
    whose centroid projects INSIDE the outlet polygon along the draw axis
    (``scope: outside_outlet``): those are covered by the outlet opening; only pockets
    growing outside it are penalized. Applied AFTER the orientation vote (the vote stays on
    the full side-wall set so a mostly-inside wall cannot destabilize the global sign) and
    affects the penalty, its gradient, and the diagnostics.

    Returns:

      U             scalar (autograd-connected): the penalty (intensive, O(violation^2) in
                    [0,~1]) or, for ``ks_margin``, the signed margin M in [-1, 1]
      undercut_area detached float, kept-wall area where n·d > threshold (interpretable diagnostic)
      undercut_mask detached bool tensor [F], True for the offending (undercut) faces
      face_centroid detached [F,3] face centroids
      n_oriented    detached [F,3] outward-oriented unit face normals
      ndotd         detached [F]   n·d per face (the drawability test quantity)

    Face normals use the mesh's consistent winding oriented outward by a single global sign
    voted from the side-wall faces (near-axial openings/flares are excluded from the vote so a
    flare cannot flip it). Validated to read ~0 on a monotonically-drawable channel and ~100% on a
    converging one, and to keep the normal field consistent (no faces flipped into the fluid).
    """
    f = faces.to(torch.long)
    v0, v1, v2 = verts[f[:, 0]], verts[f[:, 1]], verts[f[:, 2]]
    fn = torch.cross(v1 - v0, v2 - v0, dim=1)
    two_area = fn.norm(dim=1)
    area = 0.5 * two_area
    n = fn / two_area.clamp_min(1e-20).unsqueeze(1)

    d = torch.as_tensor(draw_dir, dtype=verts.dtype, device=verts.device)
    d = d / d.norm().clamp_min(1e-20)
    nd = (n * d).sum(dim=1)

    # Side-wall mask (orientation-independent, uses |n·d|): keep faces more than
    # exclude_axial_deg away from the draw axis; drop openings/caps/flare facing +/-d.
    keep_bool = nd.abs() <= math.cos(math.radians(exclude_axial_deg))
    face_centroid = (v0 + v1 + v2) / 3.0
    # Optional spatial exclusion (single box or list of boxes, physical mm): drop faces
    # whose centroid lies inside (legitimate opening/shoulder zones). Excluded faces also
    # do not vote on the global orientation sign below.
    if exclude_region is not None:
        boxes = exclude_region if hasattr(exclude_region[0][0], "__len__") else [exclude_region]
        inside = torch.zeros(keep_bool.shape[0], dtype=torch.bool, device=verts.device)
        for bx in boxes:
            blo = torch.as_tensor(bx[0], dtype=verts.dtype, device=verts.device)
            bhi = torch.as_tensor(bx[1], dtype=verts.dtype, device=verts.device)
            lo_b, hi_b = torch.minimum(blo, bhi), torch.maximum(blo, bhi)
            inside = inside | ((face_centroid >= lo_b) & (face_centroid <= hi_b)).all(dim=1)
        keep_bool = keep_bool & (~inside)
    keep = keep_bool.to(verts.dtype)

    # Outward orientation: the mesh winding is globally consistent, so orient the whole field
    # with a SINGLE global sign rather than per face (a per-face flip toward the axis corrupts
    # the consistent winding on non-convex / re-entrant geometry, leaving ~40% of normals
    # pointing into the fluid and mis-classifying those faces). The global sign is voted from
    # the kept side-wall faces only -- they point radially so they vote cleanly, while near-axial
    # openings/flares (which can dominate and flip a naive vote) are excluded by the mask.
    rel = face_centroid - verts.mean(dim=0)
    rho = rel - (rel * d).sum(dim=1, keepdim=True) * d  # radial component (perp to draw axis)
    vote = (area * keep * (rho * n).sum(dim=1)).sum()
    gs = torch.sign(vote)
    gs = torch.where(gs == 0, torch.ones_like(gs), gs)
    if surface == "cavity":
        # Internal channel wall: solid-outward = radially INWARD, so the "radially
        # outward" vote result must be inverted to yield solid-outward normals.
        gs = -gs
    elif surface != "solid":
        raise ValueError(f"undercut_penalty surface must be 'solid' or 'cavity', got '{surface}'")
    n_oriented = n * gs  # outward-oriented unit face normals (consistent winding * global sign)
    ndotd = nd * gs

    # Outlet-silhouette scope gate (scope: outside_outlet): exempt faces whose centroid
    # projects INSIDE the outlet polygon along the draw axis -- they are covered by the
    # outlet opening. Applied AFTER the vote above so the global sign stays stable even
    # when most of the wall lies inside the silhouette. Detached hard mask (like
    # exclude_region); the silhouette itself is fixed geometry (locked outlet).
    if silhouette is not None:
        inside_sil = torch.as_tensor(
            silhouette.inside(face_centroid.detach().cpu().numpy()),
            dtype=torch.bool, device=verts.device,
        )
        keep_bool = keep_bool & (~inside_sil)
        keep = keep_bool.to(verts.dtype)

    viol = torch.clamp(ndotd - threshold, min=0.0) * keep
    if formulation == "ks_margin":
        # Signed worst-case margin (see docstring): smooth area-weighted max of n·d over
        # the kept side walls. Weights are DETACHED normalized areas -- the gradient flows
        # through the face normals only, mirroring the detached denominator of the mean
        # form (no "resize the face to change its vote" path); the softmax weights sum to
        # 1, so the gradient never vanishes at the constraint boundary.
        if bool(keep_bool.any()):
            w = area.detach()[keep_bool]
            w = w / w.sum().clamp_min(1e-20)
            rho = float(ks_rho)
            U = torch.logsumexp(rho * ndotd[keep_bool] + torch.log(w), dim=0) / rho
        else:
            # No eligible faces: fully feasible (margin at the sine floor), autograd-connected.
            U = verts.sum() * 0.0 - 1.0
    elif formulation == "penalty":
        # Intensive (mean) form: area-weighted MEAN squared violation over the kept side-wall
        # area, not the extensive sum. The detached denominator makes this a pure rescale (value
        # and autograd gradient divided by the same constant -- no perverse "grow the area to
        # dilute" gradient) that is independent of mesh resolution and physical part size, so the
        # penalty lands at O(violation^2) in [0, ~1] and a tuned weight/target transfers across
        # configs. clamp_min guards the no-kept-faces case (numerator is then ~0 too -> U ~ 0).
        kept_area = (area * keep).sum().detach().clamp_min(1e-12)
        U = (area * viol ** 2).sum() / kept_area
    else:
        raise ValueError(
            f"undercut_penalty formulation must be 'penalty' or 'ks_margin', got '{formulation}'"
        )
    undercut_mask = keep_bool & (ndotd > threshold)
    undercut_area = area[undercut_mask].sum().detach()
    # Also return per-face centroids, oriented normals and n·d so callers can visualize
    # the offending faces' normals (e.g. a glyph VTP).
    return (
        U, undercut_area, undercut_mask.detach(),
        face_centroid.detach(), n_oriented.detach(), ndotd.detach(),
    )

def undercut_penalty_sdf(
    lattice_struct, frame, param, draw_dir, threshold,
    exclude_axial_deg=30.0, grid_spacing=0.5, band_factor=1.5, exclude_region=None,
    collect_debug=False, formulation="penalty", ks_rho=50.0, silhouette=None,
):
    """SDF-level-set draft / undercut penalty -- smooth in the latent params.

    The SDF sibling of the mesh-based :func:`undercut_penalty`, 
    The mesh version reads face normals off the surface the sdf_hex pipeline *re-meshes*
    every iteration (a noisy, discontinuous function of the latents whose autograd
    gradient does not match a finite difference), and it must *vote a single global sign*
    to orient those normals -- fragile on non-convex / re-entrant geometry. This version
    never touches the mesh: it reads ``phi`` and its spatial gradient on a FIXED grid and
    integrates the drawability violation over a thin band around the zero level set with a
    smoothed surface delta (compact-support raised cosine):

        U = sum_k delta_eps(phi_k)*|grad phi_k|*keep_k*relu(ndotd_out_k - threshold)^2 * dV
            ----------------------------------------------------------------------------------
                          sum_k delta_eps(phi_k)*|grad phi_k|*keep_k * dV

    ``delta_eps(phi)|grad phi| dV`` is the surface-area element, so the numerator is the
    area-weighted drawability violation and the denominator is the kept SIDE-WALL band area:
    U is their ratio, the area-weighted MEAN squared violation -- intensive (O(violation^2)
    in [0,~1], grid- and size-independent), not the extensive sum. The denominator is
    detached, so value and gradient rescale by the same constant.

    Orientation is automatic: ``grad phi`` points toward increasing phi, and in this convention
    ``phi > 0`` is solid (see :func:`min_wall_thickness_penalty_sdf`), so the **outward**
    normal (solid->fluid) is ``n_out = -grad phi / |grad phi|`` and the drawability quantity
    is ``ndotd_out = n_out . d``. No global-sign vote is needed -- the level-set gradient is
    globally consistent by construction. The surface is drawable along ``draw_dir`` where
    ``ndotd_out <= threshold``; an undercut has ``ndotd_out > threshold`` (with
    ``threshold = -sin(draft angle)``, identical to the mesh version).

    Only **side-wall** points are penalized: ``keep_k = (|n.d| <= cos(exclude_axial_deg))``
    drops near-axial openings/caps/flares (orientation-independent, so it needs no sign).
    Sampling is in NORMALIZED coordinates (see :func:`taper_penalty_sdf` for why
    ``|n.d|`` and the band width are frame-invariant). ``exclude_region`` is a physical-mm
    box or a LIST of boxes (e.g. one per fluid opening).

    ``formulation`` / ``ks_rho`` / ``silhouette`` mirror :func:`undercut_penalty` exactly:
    ``"ks_margin"`` returns the SIGNED worst-case margin M = smooth area-weighted max of
    ``ndotd_out`` over the side-wall band (weights ``delta*|grad phi|*dV*sidewall``,
    detached, normalized to sum 1 -- so M is in [-1, 1] sine units, the constraint is
    ``M <= threshold``, and feasibility shows as genuinely negative slack with a
    never-vanishing softmax gradient). ``silhouette`` exempts band points whose oblique
    projection along the draw axis lands inside the outlet polygon
    (``scope: outside_outlet``); both the grid points and the silhouette are FIXED, so the
    gate's membership is constant across iterations and adds no new discontinuity in the
    latents. In ks_margin mode the debug cloud carries an extra ``ks_weight`` scalar (the
    normalized aggregation weight per band point).

    Returns ``(U_value_tensor, dU_param, n_band, n_undercut, pts_phys, scalars, normals)``
    where ``U_value_tensor`` is detached and ``dU_param`` is ``dU/dparam`` (both ready to
    feed the optimizer). The last three are the debug cloud (only when ``collect_debug``,
    else ``None``): the WHOLE band point cloud in physical mm -- exclude_region already
    removed, i.e. exactly the points the penalty integrated -- with outward normals and
    per-point scalars ``n_dot_d_out`` / ``viol`` (side-wall-gated) / ``sidewall`` (gate
    weight). Threshold ``viol > 0`` in ParaView to isolate the offending points; a clean
    iteration still yields a cloud (viol ~ 0 everywhere), so "clean" and "export broken"
    stay distinguishable.
    """
    from DeepSDFStruct.utils import with_float32_lattice
    import math as _math

    device = param.device
    box_norm = frame.box_norm.to(device=device, dtype=torch.float32)
    scale = float(frame.scale)
    sp = scale * float(grid_spacing)        # grid spacing in normalized units
    eps = band_factor * sp                  # band half-width (normalized SDF units)
    fd = 0.25 * sp                          # central-difference step (normalized)
    lo = box_norm[0]
    hi = box_norm[1]
    inset = sp + fd                         # keep grid + finite-diff stencil inside box_norm
    cos_excl = _math.cos(_math.radians(float(exclude_axial_deg)))  # side-wall gate on |n·d|

    # exclude_region: a single [[lo],[hi]] mm box, OR a list of such boxes (so e.g. the
    # inlet and outlet opening zones can both be exempted -- same convention as min_steg).
    excl_boxes = []
    if exclude_region is not None:
        boxes = exclude_region if hasattr(exclude_region[0][0], "__len__") else [exclude_region]
        for bx in boxes:
            elo = frame.to_norm(torch.as_tensor(bx[0], dtype=torch.float32, device=device))
            ehi = frame.to_norm(torch.as_tensor(bx[1], dtype=torch.float32, device=device))
            excl_boxes.append((torch.minimum(elo, ehi), torch.maximum(elo, ehi)))

    def _query(x):
        return lattice_struct(x).reshape(-1)

    def _compute(_bounds_f32):
        axes = []
        for i in range(3):
            n_i = max(2, int(round((hi[i].item() - lo[i].item() - 2 * inset) / sp)) + 1)
            axes.append(torch.linspace(lo[i].item() + inset, hi[i].item() - inset, n_i, device=device))
        gx, gy, gz = torch.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
        grid = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=1).float()
        dV = sp ** 3

        # Pass 1 (no grad): keep only the narrow band around the surface (and outside
        # exclude_region). Band points at |phi|=eps have delta_eps=0, so this hard
        # selection adds no discontinuity as points enter/leave the band.
        with torch.no_grad():
            phi0 = _query(grid)
        keep = phi0.abs() < eps
        if excl_boxes:
            inside = torch.zeros(grid.shape[0], dtype=torch.bool, device=device)
            for blo, bhi in excl_boxes:
                inside = inside | ((grid >= blo) & (grid <= bhi)).all(dim=1)
            keep = keep & (~inside)
        Xb = grid[keep]
        if silhouette is not None and Xb.shape[0] > 0:
            # Outlet-silhouette scope gate (scope: outside_outlet), tested in PHYSICAL mm.
            # Grid points and silhouette are both fixed, so this hard selection is constant
            # across iterations (no discontinuity in the latents) -- points covered by the
            # outlet opening leave numerator, denominator and the debug cloud alike.
            pts_mm = frame.to_phys(Xb.to(param.dtype)).detach().cpu().numpy()
            outside = torch.as_tensor(
                ~silhouette.inside(pts_mm), dtype=torch.bool, device=device
            )
            Xb = Xb[outside]
        n_band = int(Xb.shape[0])
        if n_band == 0:
            # ks_margin: empty region = fully feasible margin (-1); penalty: 0. Both
            # autograd-connected so the caller's grad() finds param in the graph.
            base = param.sum() * 0.0
            return (base - 1.0 if formulation == "ks_margin" else base), 0, 0, None

        d = torch.as_tensor(draw_dir, dtype=torch.float32, device=device)
        d = d / d.norm().clamp_min(1e-20)
        ex = torch.tensor([fd, 0.0, 0.0], device=device)
        ey = torch.tensor([0.0, fd, 0.0], device=device)
        ez = torch.tensor([0.0, 0.0, fd], device=device)

        phi = _query(Xb)
        gpx = (_query(Xb + ex) - _query(Xb - ex)) / (2 * fd)
        gpy = (_query(Xb + ey) - _query(Xb - ey)) / (2 * fd)
        gpz = (_query(Xb + ez) - _query(Xb - ez)) / (2 * fd)
        gnorm = torch.sqrt(gpx ** 2 + gpy ** 2 + gpz ** 2).clamp_min(1e-12)
        nd = (gpx * d[0] + gpy * d[1] + gpz * d[2]) / gnorm   # (grad phi / |grad phi|) . d
        ndotd_out = -nd                                       # outward normal is -grad phi
        # Side-wall gate (orientation-independent): drop near-axial openings/caps/flares.
        # SMOOTH raised-cosine ramp instead of a hard boolean: with a binary gate, a band
        # point whose normal sits at the cone edge flips discretely in/out of numerator
        # AND denominator as the latents move, so U has finite jumps (observed ~4e-4 from
        # single high-violation points at |n.d| ~ cos(excl)). The MMA linearization cannot
        # see such a jump, the GCMMA back-off then converges INTO the cliff edge and the
        # optimization stalls riding it. The ramp (1 -> 0 over +-gate_w around cos_excl,
        # ~2 deg at excl=30 deg) keeps gate membership differentiable, so crossing the
        # cone edge becomes a steep-but-smooth trade the optimizer can navigate.
        gate_w = 0.03
        tt = ((nd.abs() - (cos_excl - gate_w)) / (2.0 * gate_w)).clamp(0.0, 1.0)
        sidewall = 0.5 * (1.0 + torch.cos(_math.pi * tt))
        viol = torch.clamp(ndotd_out - threshold, min=0.0) * sidewall
        delta = torch.where(
            phi.abs() < eps,
            (1.0 / (2.0 * eps)) * (1.0 + torch.cos(_math.pi * phi / eps)),
            torch.zeros_like(phi),
        )
        area_elem = delta * gnorm * dV
        ks_w = None
        if formulation == "ks_margin":
            # Signed worst-case margin (see undercut_penalty): smooth area-weighted max of
            # ndotd_out over the side-wall band. Weights = detached normalized area
            # elements gated by the smooth sidewall ramp (points entering the band or the
            # sidewall cone fade in with weight -> 0, so membership stays smooth); the
            # gradient flows through ndotd_out (the level-set normals) only.
            w = (area_elem * sidewall).detach()
            ks_w = w / w.sum().clamp_min(1e-30)
            pos = ks_w > 0
            if bool(pos.any()):
                rho = float(ks_rho)
                U = torch.logsumexp(rho * ndotd_out[pos] + torch.log(ks_w[pos]), dim=0) / rho
            else:
                U = param.sum() * 0.0 - 1.0
        elif formulation == "penalty":
            # Intensive (mean) form: divide the surface integral by the SIDE-WALL band area
            # (delta*gnorm*dV is the area element; sidewall gates to the eligible faces) ->
            # area-weighted MEAN squared drawability violation. Detached denominator -> pure,
            # grid- and size-independent rescale to O(violation^2) in [0, ~1] (see
            # taper_penalty_sdf for the full rationale).
            U = (area_elem * viol ** 2).sum() / (area_elem * sidewall).sum().detach().clamp_min(1e-12)
        else:
            raise ValueError(
                f"undercut_penalty_sdf formulation must be 'penalty' or 'ks_margin', got '{formulation}'"
            )
        n_undercut = int((viol > 0).sum())
        dbg = None
        if collect_debug:
            with torch.no_grad():
                n_out = -torch.stack([gpx, gpy, gpz], dim=1) / gnorm.unsqueeze(1)
                scal = {"n_dot_d_out": ndotd_out.detach(), "viol": viol.detach(),
                        "sidewall": sidewall.detach()}
                if ks_w is not None:
                    scal["ks_weight"] = ks_w.detach()
                dbg = (Xb.detach(), n_out.detach(), scal)
        return U, n_band, n_undercut, dbg

    U, n_band, n_undercut, dbg = with_float32_lattice(lattice_struct, frame.box_norm, _compute)
    g = torch.autograd.grad(U, param, retain_graph=False, allow_unused=True)[0]
    if g is None:
        g = torch.zeros_like(param)
    pts_phys = scalars = normals = None
    if dbg is not None and dbg[0].shape[0] > 0:
        pts_phys = frame.to_phys(dbg[0].to(param.dtype)).detach().cpu().numpy()
        normals = dbg[1].cpu().numpy()
        scalars = {k: v.cpu().numpy() for k, v in dbg[2].items()}
    return (
        U.detach().to(param.dtype), g.to(param.dtype), n_band, n_undercut,
        pts_phys, scalars, normals,
    )


# ---------------------------------------------------------------------------
# Term
# ---------------------------------------------------------------------------

from .base import Budget, ConstraintTerm, PenaltyTerm, State, TermValue, exclude_boxes, known_keys  # noqa: E402

_UNDERCUT_KEYS = {"type", "weight", "budget", "method", "formulation", "draw_direction", "draft_angle_deg",
                  "exclude_axial_deg", "grid_spacing", "band_factor", "exclude_region", "ks_rho", "scope",
                  "silhouette_margin", "outlet_patch"}


class _UndercutEvaluator:
    """Shared evaluation of the drawability measure; the term classes add budget or weight."""

    def __init__(self, cfg: dict):
        known_keys(cfg, _UNDERCUT_KEYS, "undercut")
        self.method = str(cfg.get("method", "sdf"))
        self.formulation = str(cfg.get("formulation", "penalty"))
        if self.method not in ("sdf", "mesh"):
            raise ValueError(f"undercut.method must be 'sdf' or 'mesh', got {self.method!r}")
        if self.formulation not in ("penalty", "ks_margin"):
            raise ValueError(f"undercut.formulation must be 'penalty' or 'ks_margin', got {self.formulation!r}")
        self.draw_dir = cfg.get("draw_direction", [1.0, 0.0, 0.0])
        self.draft_angle_deg = float(cfg.get("draft_angle_deg", 0.0))
        self.threshold = -math.sin(math.radians(self.draft_angle_deg))
        self.exclude_axial_deg = float(cfg.get("exclude_axial_deg", 30.0))
        self.grid_spacing = float(cfg.get("grid_spacing", 0.5))
        self.band_factor = float(cfg.get("band_factor", 1.5))
        self.exclude_region = exclude_boxes(cfg.get("exclude_region"))
        self.ks_rho = float(cfg.get("ks_rho", 50.0))
        self.scope = str(cfg.get("scope", "all"))
        if self.scope not in ("all", "outside_outlet"):
            raise ValueError(f"undercut.scope must be 'all' or 'outside_outlet', got {self.scope!r}")
        self.silhouette_margin = float(cfg.get("silhouette_margin", 0.0))
        self.outlet_patch = str(cfg.get("outlet_patch", "outlet"))
        self.silhouette = None

    def _ensure_silhouette(self, state: State) -> None:
        if self.scope != "outside_outlet" or self.silhouette is not None:
            return
        import numpy as np

        pm = state.mesh.polymesh
        quads = pm.faces[pm.patch_face_slice(self.outlet_patch)]
        pts = np.asarray(pm.points, dtype=float)
        tris = np.concatenate([pts[quads[:, [0, 1, 2]]], pts[quads[:, [0, 2, 3]]]], axis=0)
        self.silhouette = build_outlet_silhouette(tris, self.draw_dir, margin=self.silhouette_margin)

    def measure(self, state: State) -> TermValue:
        self._ensure_silhouette(state)
        param = state.param
        if self.method == "sdf":
            lattice = getattr(state.parametrization, "lattice_struct", None)
            if lattice is None:
                raise ValueError("undercut.method 'sdf' needs the DeepSDF lattice parametrization")
            U, dU, n_band, n_uc, pts, scalars, normals = undercut_penalty_sdf(
                lattice, state.parametrization.frame, param, self.draw_dir, self.threshold,
                exclude_axial_deg=self.exclude_axial_deg, grid_spacing=self.grid_spacing,
                band_factor=self.band_factor, exclude_region=self.exclude_region,
                collect_debug=state.debug, formulation=self.formulation, ks_rho=self.ks_rho,
                silhouette=self.silhouette,
            )
            debug = {"band_points": n_band, "undercut_points": n_uc}
            if pts is not None:
                debug["cloud"] = (pts, normals, scalars)
            return TermValue(value=float(U.item()), grad=dU, debug=debug)
        faces = state.mesh.faces_design
        U, area, mask, centroids, normals, ndotd = undercut_penalty(
            state.mesh.verts, faces, self.draw_dir, self.threshold, self.exclude_axial_deg,
            exclude_region=self.exclude_region, surface="cavity",
            formulation=self.formulation, ks_rho=self.ks_rho, silhouette=self.silhouette,
        )
        (dU,) = torch.autograd.grad(U, param, retain_graph=True)
        debug = {"undercut_area": float(area.item()), "faces": (faces[mask].detach().cpu().numpy(),
                 centroids[mask].cpu().numpy(), (-normals[mask]).cpu().numpy(), ndotd[mask].cpu().numpy())}
        return TermValue(value=float(U.item()), grad=dU.detach(), debug=debug)


class UndercutConstraint(_UndercutEvaluator, ConstraintTerm):
    def __init__(self, cfg: dict):
        _UndercutEvaluator.__init__(self, cfg)
        default = "ks_bound" if self.formulation == "ks_margin" else "relative_to_initial"
        ConstraintTerm.__init__(self, Budget(cfg.get("budget"), default, name="undercut"))
        self.name = "undercut"
        self.cheap = self.method == "sdf"
        self.unscaled = self.formulation == "ks_margin"

    def ks_bound(self):
        return self.threshold if self.formulation == "ks_margin" else None

    def evaluate(self, state: State) -> TermValue:
        return self.measure(state)


class UndercutPenalty(_UndercutEvaluator, PenaltyTerm):
    def __init__(self, cfg: dict):
        _UndercutEvaluator.__init__(self, cfg)
        if self.formulation == "ks_margin":
            raise ValueError("undercut: the ks_margin formulation is a signed margin; use it as a constraint")
        PenaltyTerm.__init__(self, cfg.get("weight", 0.0))
        self.name = "undercut"
        self.cheap = self.method == "sdf"

    def evaluate(self, state: State) -> TermValue:
        return self.measure(state)

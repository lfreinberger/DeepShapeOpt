"""Differentiable geometry / manufacturability penalties for SDF/mesh shape optimization.

Each penalty is a smooth function of the B-spline latent parameters and returns a value plus its
gradient w.r.t. those parameters, ready to feed the MMA optimizer. The SDF-level-set variants never
touch the mesh: they sample the LatticeSDFStruct on a fixed normalized-coordinate grid via
with_float32_lattice.
"""
import math

import torch


def control_lattice_smoothness_penalty(param, n_ctrl_per_dim, latent_dim, param_ref=None):
    """Spatial smoothness (graph-Dirichlet energy) over the B-spline latent control lattice.

    Unlike the other penalties here, this acts purely in *latent design-variable space* (not the
    SDF/mesh): it penalizes large differences between neighboring control-point latent vectors, so a
    single control point cannot jump into a strange decoder region while its neighbors stay normal.
    Use it as a reconstruction-loss term and/or an optimization penalty to keep the latent field
    spatially coherent (a soft, manifold-respecting regularizer complementary to the PCA basis).

    With ``param_ref`` the energy acts on the *update field* ``delta = param - param_ref`` instead
    of the absolute latent field. Locked control points keep delta = 0, so minimizing the Dirichlet
    energy of delta tapers the shape change smoothly to zero at locked boundaries (e.g. the frozen
    outlet face) instead of collapsing within one control-point spacing -- without penalizing the
    legitimate spatial variation already present in ``param_ref`` (the initial reconstruction).

    ``param`` is the flattened control points (n_ctrl_total * latent_dim,) or (n_ctrl_total,
    latent_dim). splinepy orders control points x-fastest, then y, then z, so the C-order reshape is
    ``(nz, ny, nx, latent_dim)`` -- x-neighbors lie along axis 2, y along axis 1, z along axis 0.

    Intensive form: mean squared difference per adjacent (pair, component), so the value is grid- and
    latent-dim-independent and a tuned weight transfers across tilings. Returns
    ``(P_value_detached, dP_param)`` with ``dP_param`` = dP/dparam (same flat shape as ``param``).
    """
    nx, ny, nz = (int(n) for n in n_ctrl_per_dim)
    field = param if param_ref is None else (param - param_ref.detach())
    flat = field.reshape(-1)
    x = flat.reshape(nz, ny, nx, int(latent_dim))

    sq, cnt = 0.0, 0
    if nx > 1:
        dx = x[:, :, 1:, :] - x[:, :, :-1, :]
        sq = sq + dx.pow(2).sum(); cnt += dx.numel()
    if ny > 1:
        dy = x[:, 1:, :, :] - x[:, :-1, :, :]
        sq = sq + dy.pow(2).sum(); cnt += dy.numel()
    if nz > 1:
        dz = x[1:, :, :, :] - x[:-1, :, :, :]
        sq = sq + dz.pow(2).sum(); cnt += dz.numel()

    if cnt == 0:  # degenerate 1x1x1 lattice: nothing to smooth
        return (param.sum() * 0.0).detach(), torch.zeros_like(param)

    P = sq / cnt
    g = torch.autograd.grad(P, param, retain_graph=False, allow_unused=True)[0]
    if g is None:
        g = torch.zeros_like(param)
    return P.detach().to(param.dtype), g.to(param.dtype)


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
    from deepshapeopt.mesh import (
        _all_boundary_loops_from_triangles,
        _build_shapely_multipolygon,
        _make_plane_basis,
        _project_to_plane,
    )

    tris = np.asarray(outlet_triangles, dtype=float)
    if tris.ndim != 3 or tris.shape[0] == 0:
        raise ValueError(
            f"build_outlet_silhouette: expected (N,3,3) outlet triangles, got shape {tris.shape}."
        )
    loops_3d = _all_boundary_loops_from_triangles(tris)
    origin, normal, axis_u, axis_v = _make_plane_basis(np.vstack(loops_3d), tris)
    loops_2d = [_project_to_plane(l, origin, axis_u, axis_v) for l in loops_3d]
    poly = _build_shapely_multipolygon(loops_2d)
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
    :func:`taper_penalty` / :func:`undercut_penalty_sdf`; use it for the legitimate
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


def taper_penalty(verts, faces, flow_dir, sin_threshold, exclude_region=None):
    """Differentiable area-weighted taper (max surface angle vs. flow) penalty.

    Manufacturability: short, blunt ("stumpfe") features pointing in the flow direction break
    off under the material flow and must be forbidden, while long, conically tapering thin
    features are fine. What separates the two is the surface slope relative to the flow axis,
    not the absolute thickness: for a face with unit normal ``n`` and unit flow axis ``d`` the
    surface makes angle ``theta`` with the flow axis where ``sin(theta) = |n·d|``. A long taper
    has shallow slopes (``|n·d|`` small); a short stub has steep walls / a blunt cap
    (``|n·d| -> 1``). We penalize faces steeper than the allowed taper ``max_taper_angle_deg``,
    i.e. where ``|n·d| > sin_threshold`` with ``sin_threshold = sin(max_taper_angle_deg)``.

    This is the two-sided sibling of :func:`undercut_penalty` (undercut: ``n·d > threshold``,
    one-sided demoldability; taper: ``|n·d| > sin_threshold``, two-sided streamlining).
    Because it uses ``|n·d|`` it is orientation-independent -- no global normal-sign vote needed.

    ``exclude_region`` is an optional ``[[xmin,ymin,zmin],[xmax,ymax,zmax]]`` box in the same
    (physical, mm) frame as the mesh; faces whose centroid lies inside are dropped from the
    penalty, its gradient, and the diagnostics. Use it for walls that are *legitimately*
    perpendicular to the flow, e.g. the unavoidable shoulder where a large inlet cylinder
    steps down to a much smaller outlet region. Returns:

      P            scalar penalty  area-weighted MEAN of relu(|n_f·d| - sin_threshold)^2 over kept
                   faces, i.e. sum_f area_f*relu(.)^2 / sum_f area_f (autograd-connected, intensive:
                   O(violation^2) in [0,~1], resolution- and size-independent)
      blunt_area   detached float, area of the offending (too-blunt) faces
      blunt_mask   detached bool tensor [F], True for the offending faces
      face_centroid detached [F,3] face centroids
      n_unit       detached [F,3] unit face normals (mesh winding, un-reoriented)
      absnd        detached [F]   |n·d| per face (the bluntness test quantity)
    """
    f = faces.to(torch.long)
    v0, v1, v2 = verts[f[:, 0]], verts[f[:, 1]], verts[f[:, 2]]
    fn = torch.cross(v1 - v0, v2 - v0, dim=1)
    two_area = fn.norm(dim=1)
    area = 0.5 * two_area
    n = fn / two_area.clamp_min(1e-20).unsqueeze(1)

    d = torch.as_tensor(flow_dir, dtype=verts.dtype, device=verts.device)
    d = d / d.norm().clamp_min(1e-20)
    absnd = (n * d).sum(dim=1).abs()  # |n·d|: orientation-independent (two-sided)

    face_centroid = (v0 + v1 + v2) / 3.0

    # Optional spatial exclusion: drop faces whose centroid lies inside the box (e.g. the
    # unavoidable cylinder->outlet shoulder). keep zeroes both penalty and gradient there.
    keep = torch.ones_like(absnd)
    if exclude_region is not None:
        lo = torch.as_tensor(exclude_region[0], dtype=verts.dtype, device=verts.device)
        hi = torch.as_tensor(exclude_region[1], dtype=verts.dtype, device=verts.device)
        inside = ((face_centroid >= lo) & (face_centroid <= hi)).all(dim=1)
        keep = (~inside).to(verts.dtype)

    viol = torch.clamp(absnd - sin_threshold, min=0.0) * keep
    # Intensive (mean) form: area-weighted MEAN squared violation over the kept area (see
    # undercut_penalty for the rationale -- detached denominator gives a pure, resolution- and
    # size-independent rescale to O(violation^2) in [0, ~1]).
    kept_area = (area * keep).sum().detach().clamp_min(1e-12)
    P = (area * viol ** 2).sum() / kept_area
    blunt_mask = (absnd > sin_threshold) & keep.bool()
    blunt_area = area[blunt_mask].sum().detach()
    return (
        P, blunt_area, blunt_mask.detach(),
        face_centroid.detach(), n.detach(), absnd.detach(),
    )


def taper_penalty_sdf(
    lattice_struct, frame, param, flow_dir, sin_threshold,
    grid_spacing=0.5, band_factor=1.5, exclude_region=None,
):
    """SDF-level-set taper penalty -- smooth in the latent params.

    The mesh-based :func:`taper_penalty` reads face normals off the surface
    that the sdf_hex pipeline *re-meshes* from scratch every iteration. That makes
    the penalty a noisy, discontinuous function of the latents: a tiny latent step
    changes the castellation/snapping, flips which faces exist and which are blunt,
    and the autograd gradient (which assumes the surface deforms smoothly) does not
    match a finite difference -- validated to disagree even in sign. The optimizer
    then follows a gradient that does not correspond to the true penalty landscape.

    This version never touches the mesh. It reads the SDF ``phi`` and its spatial
    gradient -- both smooth functions of the latents -- on a FIXED grid, and
    integrates the taper violation over a thin band around the zero level set with a
    smoothed surface delta (compact-support raised cosine):

        P = sum_k delta_eps(phi_k)*|grad phi_k|*relu(|n_k.d| - sin_threshold)^2 * dV
            -------------------------------------------------------------------------
                              sum_k delta_eps(phi_k)*|grad phi_k| * dV

    with ``n_k = grad phi_k / |grad phi_k|``. ``delta_eps(phi)|grad phi| dV`` is the
    surface-area element, so the numerator is the surface integral of the taper violation
    and the denominator is the band surface area: P is their ratio, the area-weighted MEAN
    squared violation -- intensive (O(violation^2) in [0,~1], grid- and size-independent),
    not the extensive sum. The denominator is detached, so value and gradient rescale by the
    same constant. (Same quantity the mesh version estimates, but differentiable.)

    Sampling is done in NORMALIZED coordinates: the frame is isotropic (single
    ``scale = 2/L``), so normal directions -- and therefore ``|n.d|`` -- are
    identical in normalized and physical space. The SDF value is likewise in
    normalized units (``phi ~ scale * d_phys``), so the band width is set in
    normalized units from a physical ``grid_spacing`` (mm). ``exclude_region`` is a
    physical-mm box, mapped to normalized coordinates here.

    Both the full-grid band selection and the gradient evaluation are chunked
    (``CHUNK_GRID`` / ``CHUNK_BAND``, same scheme as :func:`min_steg_length_penalty_sdf`)
    so peak GPU memory stays bounded at fine ``grid_spacing`` -- the graph of the seven
    lattice queries per band point is built and freed one chunk at a time.

    Returns ``(P_value_tensor, dP_param, n_band, n_blunt_band)`` where ``P_value_tensor``
    is detached and ``dP_param`` is ``dP/dparam`` (both ready to feed the optimizer).
    """
    from deepshapeopt.reconstruction import with_float32_lattice
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

    excl_n = None
    if exclude_region is not None:
        elo = frame.to_norm(torch.as_tensor(exclude_region[0], dtype=torch.float32, device=device))
        ehi = frame.to_norm(torch.as_tensor(exclude_region[1], dtype=torch.float32, device=device))
        excl_n = (torch.minimum(elo, ehi), torch.maximum(elo, ehi))

    CHUNK_GRID = 262144   # no-grad full-box query chunk (memory bound)
    CHUNK_BAND = 16384    # grad band-eval chunk (memory bound; 7 grad queries per point)

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

        # Pass 1 (no grad, chunked): keep only the narrow band around the surface (and
        # outside exclude_region). Chunked so the full-box query never allocates the whole
        # grid at once. Band points at |phi|=eps have delta_eps=0, so this hard selection
        # adds no discontinuity as points enter/leave the band.
        keep_parts = []
        with torch.no_grad():
            for gi in range(0, grid.shape[0], CHUNK_GRID):
                gch = grid[gi:gi + CHUNK_GRID]
                kk = _query(gch).abs() < eps
                if excl_n is not None:
                    inside = ((gch >= excl_n[0]) & (gch <= excl_n[1])).all(dim=1)
                    kk = kk & (~inside)
                keep_parts.append(kk)
        Xb_all = grid[torch.cat(keep_parts)]
        n_band = int(Xb_all.shape[0])
        if n_band == 0:
            return (param.sum() * 0.0).detach(), torch.zeros_like(param), 0, 0

        d = torch.as_tensor(flow_dir, dtype=torch.float32, device=device)
        d = d / d.norm().clamp_min(1e-20)
        ex = torch.tensor([fd, 0.0, 0.0], device=device)
        ey = torch.tensor([0.0, fd, 0.0], device=device)
        ez = torch.tensor([0.0, 0.0, fd], device=device)

        # Pass 2 (grad, chunked): accumulate the violation surface integral, its gradient
        # and the (detached) band area one chunk at a time, calling autograd.grad per chunk
        # with retain_graph=False so the graph of the 7 lattice queries per point is freed
        # immediately -- peak memory is bounded by one chunk regardless of band size.
        P_num = 0.0
        area_sum = 0.0
        g_acc = torch.zeros_like(param)
        n_blunt = 0
        for ci in range(0, n_band, CHUNK_BAND):
            Xb = Xb_all[ci:ci + CHUNK_BAND]
            phi = _query(Xb)
            gpx = (_query(Xb + ex) - _query(Xb - ex)) / (2 * fd)
            gpy = (_query(Xb + ey) - _query(Xb - ey)) / (2 * fd)
            gpz = (_query(Xb + ez) - _query(Xb - ez)) / (2 * fd)
            gnorm = torch.sqrt(gpx ** 2 + gpy ** 2 + gpz ** 2).clamp_min(1e-12)
            absnd = (gpx * d[0] + gpy * d[1] + gpz * d[2]).abs() / gnorm
            viol = torch.clamp(absnd - sin_threshold, min=0.0)
            delta = torch.where(
                phi.abs() < eps,
                (1.0 / (2.0 * eps)) * (1.0 + torch.cos(_math.pi * phi / eps)),
                torch.zeros_like(phi),
            )
            area_elem = delta * gnorm * dV
            num_c = (area_elem * viol ** 2).sum()
            gc = torch.autograd.grad(num_c, param, retain_graph=False, allow_unused=True)[0]
            if gc is not None:
                g_acc = g_acc + gc
            P_num += float(num_c.detach())
            area_sum += float(area_elem.sum().detach())
            n_blunt += int((viol > 0).sum())

        # Intensive (mean) form: divide the surface integral of the taper violation by the
        # band SURFACE AREA (delta*gnorm*dV is the area element) -> area-weighted MEAN squared
        # violation. The detached denominator makes this a pure rescale (value and autograd
        # gradient by the same constant -- no perverse "grow the surface to dilute" gradient)
        # that is independent of grid_spacing and physical part size, so P lands at
        # O(violation^2) in [0, ~1] and a tuned target transfers across configs. Dividing
        # the accumulated numerator and gradient by the accumulated (already-detached) area
        # is identical to the unchunked P = num.sum() / area.sum().detach().
        A = max(area_sum, 1e-12)
        P_t = torch.tensor(P_num / A, device=device, dtype=torch.float32)
        return P_t, g_acc / A, n_band, n_blunt

    P, g, n_band, n_blunt = with_float32_lattice(lattice_struct, frame.box_norm, _compute)
    return P.detach().to(param.dtype), g.to(param.dtype), n_band, n_blunt


def undercut_penalty_sdf(
    lattice_struct, frame, param, draw_dir, threshold,
    exclude_axial_deg=30.0, grid_spacing=0.5, band_factor=1.5, exclude_region=None,
    collect_debug=False, formulation="penalty", ks_rho=50.0, silhouette=None,
):
    """SDF-level-set draft / undercut penalty -- smooth in the latent params.

    The SDF sibling of the mesh-based :func:`undercut_penalty`, exactly as
    :func:`taper_penalty_sdf` is the SDF sibling of :func:`taper_penalty`.
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
    from deepshapeopt.reconstruction import with_float32_lattice
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


def min_wall_thickness_penalty_sdf(
    lattice_struct, frame, param, min_thickness_mm,
    grid_spacing=0.2, grad_cutoff=0.5, slab_margin=0.25, weight=1.0,
    exclude_region=None,
):
    """SDF min-wall-thickness penalty -- smooth in the latent params, ramps *before* pinch-off.

    The taper :func:`taper_penalty_sdf` is a *cliff* w.r.t. wall thickness: a thinning
    solid wall produces no taper violation until it pinches into a hole, then the hole's rim
    spikes the penalty discontinuously. By then the optimizer has already driven into the
    infeasible region with a meaningless gradient. This penalty instead measures *thickness*
    and grows smoothly as a wall thins toward pinch-off, so MMA gets a real gradient that
    pushes the thickness back up before any hole forms.

    Sign convention (raw lattice, ``fluid_side: inside``): fluid is ``phi < 0`` and the solid
    wall is ``phi > 0``. A thin solid wall is a thin slab of ``phi > 0`` between two
    ``phi < 0`` channels; on its midline (medial axis) the distance contributions of the two
    opposing surfaces cancel, so ``phi`` is small *and* ``|grad phi|`` is small. Both are
    smooth functions of the latents. We integrate, over the solid slab, a thickness violation
    weighted by a medial indicator:

        W = weight * sum_k m_k * relu(1 - phi_k/h)^2 * dV / (sum_k m_k * dV)
                                                     (over solid points 0 < phi < h*(1+margin))

    i.e. the medial-weighted MEAN squared thickness violation -- intensive (O(violation^2) in
    [0,~1], grid- and size-independent), not the extensive sum; the denominator (detached
    medial-weighted volume) rescales value and gradient by the same constant. With
    half-thickness threshold ``h = scale * min_thickness_mm/2`` (normalized SDF units),
    medial weight ``m = relu(grad_cutoff - |grad phi|)^2 / grad_cutoff^2`` (=1 on a ridge where
    ``|grad phi| -> 0``, =0 near a single clean surface where ``|grad phi| ~ 1``), and
    ``viol_frac = relu(1 - phi/h)`` in [0, 1]. The medial weight is essential: without it every
    surface's near-skin (small phi) would be penalized -- including thick walls -- and MMA would
    just inflate the whole design / shrink the channel everywhere. ``m`` localizes the penalty to
    genuine thin walls (small phi AND cancelled gradient).

    ``grad_cutoff`` is effectively a TAPER-ANGLE tolerance: at the midline of a feature whose
    surfaces taper at half-angle ``alpha`` the medial gradient is ``|grad phi| ~ sin(alpha)`` --
    a flat/parallel wall (alpha -> 0, the geometry that pinches into a hole) has ``|grad phi| -> 0``
    and is caught, while a cone/taper steeper than ``arcsin(grad_cutoff)`` keeps a residual
    gradient and is exempt. So this penalty fires on features that are *thin AND wall-like*
    (a hole precursor) but leaves legitimate steep tapers alone. ``grad_cutoff`` must be tight
    enough to exclude tapers yet loose enough that the discrete grid still samples the
    near-critical pinch point: ~0.2-0.3 (i.e. ~12-17 deg) works well in practice; values
    >=0.5 over-flag tapering tips, values <=0.1 start missing the saddle between grid nodes.

    Sampling mirrors :func:`taper_penalty_sdf`: a FIXED grid in NORMALIZED coordinates,
    a no-grad pass to select the thin solid slab, then a grad pass with central-difference
    spatial gradients (kept connected to ``param``). ``exclude_region`` is a physical-mm box
    (e.g. the cylinder->outlet shoulder) mapped to normalized coordinates here.

    Returns ``(W_value_tensor, dW_param, n_solid, n_thin, thin_pts_phys, thin_thickness_mm,
    thin_grad_dir)`` where ``W_value_tensor`` is detached, ``dW_param`` is ``dW/dparam``, and the
    trailing arrays (detached numpy) describe the flagged thin-wall points for a debug VTP.
    """
    from deepshapeopt.reconstruction import with_float32_lattice

    device = param.device
    box_norm = frame.box_norm.to(device=device, dtype=torch.float32)
    scale = float(frame.scale)
    sp = scale * float(grid_spacing)            # grid spacing in normalized units
    h = scale * 0.5 * float(min_thickness_mm)   # half-thickness threshold (normalized SDF units)
    phi_hi = h * (1.0 + float(slab_margin))     # solid-slab upper bound on phi
    fd = 0.25 * sp                              # central-difference step (normalized)
    gc = float(grad_cutoff)
    lo = box_norm[0]
    hi = box_norm[1]
    inset = sp + fd                             # keep grid + finite-diff stencil inside box_norm

    excl_n = None
    if exclude_region is not None:
        elo = frame.to_norm(torch.as_tensor(exclude_region[0], dtype=torch.float32, device=device))
        ehi = frame.to_norm(torch.as_tensor(exclude_region[1], dtype=torch.float32, device=device))
        excl_n = (torch.minimum(elo, ehi), torch.maximum(elo, ehi))

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

        # Pass 1 (no grad): keep only the thin solid slab 0 < phi < phi_hi (and outside
        # exclude_region). At phi=phi_hi the integrand (relu(1-phi/h)^2) is already 0, so the
        # slab's upper face adds no discontinuity; the phi=0 face is the pinch event itself.
        with torch.no_grad():
            phi0 = _query(grid)
        keep = (phi0 > 0.0) & (phi0 < phi_hi)
        if excl_n is not None:
            inside = ((grid >= excl_n[0]) & (grid <= excl_n[1])).all(dim=1)
            keep = keep & (~inside)
        Xb = grid[keep]
        n_solid = int(Xb.shape[0])
        if n_solid == 0:
            return param.sum() * 0.0, 0, 0, None, None, None

        ex = torch.tensor([fd, 0.0, 0.0], device=device)
        ey = torch.tensor([0.0, fd, 0.0], device=device)
        ez = torch.tensor([0.0, 0.0, fd], device=device)

        phi = _query(Xb)
        gpx = (_query(Xb + ex) - _query(Xb - ex)) / (2 * fd)
        gpy = (_query(Xb + ey) - _query(Xb - ey)) / (2 * fd)
        gpz = (_query(Xb + ez) - _query(Xb - ez)) / (2 * fd)
        gnorm = torch.sqrt(gpx ** 2 + gpy ** 2 + gpz ** 2).clamp_min(1e-12)
        viol = torch.clamp(1.0 - phi / h, min=0.0)                  # thickness violation fraction
        medial = torch.clamp(gc - gnorm, min=0.0) ** 2 / (gc ** 2)  # 1 on a ridge, 0 near a clean surface
        # Intensive (mean) form: medial-weighted MEAN squared thickness violation (divide by
        # the medial-weighted volume), not the extensive sum. Detached denominator -> pure
        # rescale (value and gradient by the same constant) independent of grid_spacing and the
        # physical extent of the thin region, so W lands at O(violation^2) in [0, ~1] and a
        # tuned weight transfers across configs. The denominator is the medial region only, so
        # the mean is not diluted by the bulk solid; clamp_min guards the no-thin-wall case.
        medial_vol = medial.sum().detach().clamp_min(1e-12) * dV
        W = float(weight) * (medial * viol ** 2).sum() * dV / medial_vol

        flag = ((gnorm < gc) & (viol > 0)).detach()
        n_thin = int(flag.sum())
        thin_pts = Xb[flag].detach()
        thin_thk = (2.0 / scale) * phi[flag].detach()              # local wall thickness estimate (mm)
        thin_dir = (torch.stack([gpx, gpy, gpz], dim=1)[flag] / gnorm[flag].unsqueeze(1)).detach()
        return W, n_solid, n_thin, thin_pts, thin_thk, thin_dir

    W, n_solid, n_thin, thin_pts, thin_thk, thin_dir = with_float32_lattice(
        lattice_struct, frame.box_norm, _compute
    )
    g = torch.autograd.grad(W, param, retain_graph=False, allow_unused=True)[0]
    if g is None:
        g = torch.zeros_like(param)

    thin_pts_phys = thin_thk_mm = thin_grad = None
    if thin_pts is not None and thin_pts.shape[0] > 0:
        thin_pts_phys = frame.to_phys(thin_pts.to(param.dtype)).cpu().numpy()
        thin_thk_mm = thin_thk.to(param.dtype).cpu().numpy()
        thin_grad = thin_dir.to(param.dtype).cpu().numpy()
    return (
        W.detach().to(param.dtype), g.to(param.dtype), n_solid, n_thin,
        thin_pts_phys, thin_thk_mm, thin_grad,
    )


class KSStream:
    """Streaming logsumexp accumulator for chunk-wise KS (smooth-max) aggregation.

    Accumulates ``logZ = log sum_k exp(a_k)`` and (when ``param`` is given)
    ``dlogZ/dparam`` over grad-connected chunks WITHOUT keeping more than one chunk's
    autograd graph alive: each :meth:`add` reduces its chunk to a (sum, grad) pair
    immediately and rescales the running pair when the running max shifts -- exact and
    overflow-safe (every exponent evaluated is <= 0). This is what lets the ks_margin
    formulation of :func:`min_steg_length_penalty_sdf` aggregate over an unbounded
    candidate count at the same peak memory as the chunked penalty form (unlike
    :func:`undercut_penalty_sdf`, whose band fits one logsumexp). Also reused by
    scripts/check_min_steg_fd.py for the frozen-weights FD gate.
    """

    def __init__(self, param=None):
        self.param = param          # None -> value-only (no gradient accumulation)
        self.m = -math.inf          # running max exponent
        self.S = 0.0                # running sum of exp(a - m)
        self.G = None if param is None else torch.zeros_like(param)

    def add(self, a):
        """Fold in a 1D chunk of (grad-connected) exponents; empty chunks are no-ops."""
        if a.numel() == 0:
            return
        m_c = float(a.detach().max())
        if m_c == -math.inf:
            return
        m_new = max(self.m, m_c)
        S_c = torch.exp(a - m_new).sum()
        r = math.exp(self.m - m_new)  # rescale of the running pair; 0.0 on the first add
        if self.param is not None:
            g_c = torch.autograd.grad(S_c, self.param, retain_graph=False, allow_unused=True)[0]
            self.G = self.G * r + (g_c if g_c is not None else 0.0)
        self.S = self.S * r + float(S_c.detach())
        self.m = m_new

    def finalize(self):
        """Return ``(logZ, dlogZ/dparam)``; ``(-inf, zeros)`` when nothing accumulated."""
        if self.S <= 0.0:
            return -math.inf, (None if self.param is None else torch.zeros_like(self.param))
        g = None if self.param is None else self.G / self.S
        return self.m + math.log(self.S), g


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
    from deepshapeopt.reconstruction import with_float32_lattice

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



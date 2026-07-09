# Local latent-code editor

A small web GUI that loads a reconstructed shape, lets you click a knot of the
B-spline control net, and drag sliders to change that knot's **local latent
code** while the geometry updates live.

A "reconstructed shape" here is a `LatticeSDFStruct`: a DeepSDF decoder driven
by a tensor-product B-spline whose control points *are* per-knot latent codes.
Editing a knot writes one value of `control_points[knot, dim]`, re-evaluates the
SDF, and re-extracts the surface with FlexiCubes.

## Run

```bash
# env vars the experiment configs expand (same as scripts/reconstruct.py):
export DEEPSHAPEOPT_DATA_DIR=/path/to/data        # contains shapes/*.stl
export DEEPSHAPEOPT_MODEL_DIR=/path/to/models     # contains primitives_cl08, ...
export DEEPSHAPEOPT_RESULTS_DIR=/path/to/results

python scripts/latent_gui.py \
    --config experiments/reconstruction/feed_channel/config.json
# -> open http://127.0.0.1:8000
```

On first launch, if no reconstruction has been saved yet, the server fits the
lattice once (~seconds) and caches the result to `rec_parameters.pt`; subsequent
launches load instantly. Run `scripts/reconstruct.py` first to skip this.

### Useful flags

- `--mesh-n N` — per-unit-cell mesh resolution for live editing (grid is
  `N * tiling`). Lower = snappier updates, coarser surface. Defaults to a
  responsive heuristic; raise it for a finer live preview.
- `--device cpu|cuda` — override the config's device.
- `--no-fit` — require an existing `rec_parameters.pt` instead of fitting.
- `--smoke` — load, edit one code, re-mesh, print vertex/face counts, exit (no
  browser). Quick end-to-end check.

## How it works

- `backend.LatentEditSession` — owns the lattice; reuses
  `reconstruction.build_reconstruction_lattice`, loads/fits the codes,
  exposes the control net (Greville positions, row-aligned to the control
  points) and re-meshes after edits.
- `server.py` — stdlib `ThreadingHTTPServer` with `/api/state`, `/api/update`,
  `/api/reset` and static serving. No third-party web dependencies.
- `static/index.html` — self-contained Three.js viewer (CDN importmap):
  renders the surface plus a sphere per knot; click a sphere to pick a knot,
  then drag any of its `latent_dim` sliders.

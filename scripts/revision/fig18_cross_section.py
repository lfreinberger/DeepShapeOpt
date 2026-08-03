"""Figure 18 with coordinate axes and cutting-plane insets (reviewer comment 10).

The reviewer asks for coordinate axes, or an inset indicating the cutting
planes, so that "view 1" and "view 2" can be located relative to the flow
direction and the body.

Both cuts contain the flow direction x (inflow is (1,0,0), see
foam_case/0.orig/include/initialConditions):

    view 1: plane normal z -> the x-y plane   (top view)
    view 2: plane normal y -> the x-z plane   (side view)

Both geometries are cut with the SAME plane origin so the overlay is
meaningful; the origin is the centroid of the reference (neural-SDF) shape,
and both centroids are reported so the reader can see they coincide to the
tolerance imposed by the centroid constraint.

Inputs are the final surface meshes stored with the manuscript runs. No CFD,
no reconstruction, no GPU -- this is a pure replot.

Usage:
    uv run python scripts/revision/fig18_cross_section.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

REPO = Path(__file__).resolve().parents[2]
ARCHIVE = Path(
    "/storage/lfrei/Archive/DeepShapeOpt-old-private-backup/experiments/optimization"
    "/drag_optimization_cube"
)

# (label, run dir, colour) -- colours follow the manuscript caption:
# "The FFD result is shown in blue, ... neural SDF ... in orange."
GEOMETRIES = [
    ("FFD (7x7x7)", "results_ffd_cube_with_cylinders_7x7x7", "tab:blue"),
    ("Neural SDF", "results_cube_with_cylinders", "tab:orange"),
]
REFERENCE = "Neural SDF"  # whose centroid defines the cutting-plane origin

AXIS_ID = {"x": 0, "y": 1, "z": 2}
# (view label, plane normal, horizontal axis, vertical axis)
VIEWS = [
    ("View 1: $x$-$y$ plane (normal $z$)", "z", "x", "y"),
    ("View 2: $x$-$z$ plane (normal $y$)", "y", "x", "z"),
]


def to_2d_transform(normal: str, horiz: str, vert: str, origin: np.ndarray) -> np.ndarray:
    """World -> plane coordinates, with `horiz` mapped to the plot's x axis.

    trimesh's default to_planar() picks an arbitrary in-plane basis, which would
    make the two panels inconsistent with each other and with the flow
    direction. Build the basis explicitly instead.
    """
    e1 = np.zeros(3); e1[AXIS_ID[horiz]] = 1.0
    e2 = np.zeros(3); e2[AXIS_ID[vert]] = 1.0
    n = np.zeros(3); n[AXIS_ID[normal]] = 1.0
    plane_to_world = np.eye(4)
    plane_to_world[:3, 0], plane_to_world[:3, 1], plane_to_world[:3, 2] = e1, e2, n
    plane_to_world[:3, 3] = origin
    return np.linalg.inv(plane_to_world)


def section_polylines(mesh: trimesh.Trimesh, normal: str, horiz: str, vert: str,
                      origin: np.ndarray) -> list[np.ndarray]:
    n = np.zeros(3); n[AXIS_ID[normal]] = 1.0
    sec = mesh.section(plane_origin=origin, plane_normal=n)
    if sec is None:
        return []
    planar, _ = sec.to_planar(to_2D=to_2d_transform(normal, horiz, vert, origin))
    return [np.asarray(planar.vertices[e.points]) for e in planar.entities]


def draw_plane_indicator(ax, mesh: trimesh.Trimesh, normal: str, origin: np.ndarray) -> None:
    """3D sketch showing where the cut plane sits on the body.

    Drawn into its OWN axes in a dedicated figure row. An earlier version placed
    these as inset_axes over the section plots; their opaque background painted
    over part of the section curve, which read as a gap in the geometry. The
    sections are single closed loops -- keep the indicators out of the data area.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # Subsample faces: the neural mesh has ~33k triangles and this sketch is small.
    idx = np.linspace(0, len(mesh.faces) - 1, min(3000, len(mesh.faces))).astype(int)
    ax.add_collection3d(
        Poly3DCollection(mesh.vertices[mesh.faces[idx]], facecolor="0.62",
                         edgecolor="none", alpha=0.85)
    )

    lo, hi = mesh.bounds
    pad = 0.12 * (hi - lo)
    ni = AXIS_ID[normal]
    others = [i for i in range(3) if i != ni]
    a = np.linspace(lo[others[0]] - pad[others[0]], hi[others[0]] + pad[others[0]], 2)
    b = np.linspace(lo[others[1]] - pad[others[1]], hi[others[1]] + pad[others[1]], 2)
    A, B = np.meshgrid(a, b)
    corners = np.zeros((4, 3))
    corners[:, others[0]] = [A[0, 0], A[0, 1], A[1, 1], A[1, 0]]
    corners[:, others[1]] = [B[0, 0], B[0, 1], B[1, 1], B[1, 0]]
    corners[:, ni] = origin[ni]
    ax.add_collection3d(
        Poly3DCollection([corners], facecolor="tab:red", edgecolor="tab:red", alpha=0.25)
    )

    for i, lbl in enumerate("xyz"):
        v = np.zeros(3); v[i] = 1.0
        ax.quiver(*lo, *(v * 0.45 * (hi - lo)), color="k", lw=0.8, arrow_length_ratio=0.18)
        ax.text(*(lo + v * 0.55 * (hi - lo)), lbl, fontsize=7)

    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_zlim(lo[2] - pad[2], hi[2] + pad[2])
    ax.set_box_aspect((hi - lo) + 2 * pad)
    ax.set_axis_off()
    ax.view_init(elev=22, azim=-58)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--out-dir", type=Path, default=REPO / "revision_artifacts")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meshes, meta = {}, {}
    for label, run, colour in GEOMETRIES:
        path = args.archive / run / "optimization" / "current_shape.stl"
        if not path.is_file():
            raise SystemExit(f"missing geometry: {path}")
        m = trimesh.load_mesh(str(path), force="mesh")
        meshes[label] = (m, colour)
        meta[label] = {
            "run": run, "path": str(path), "n_faces": int(len(m.faces)),
            "watertight": bool(m.is_watertight),
            "centroid": [float(v) for v in m.center_mass],
            "volume": float(m.volume),
        }

    origin = np.asarray(meshes[REFERENCE][0].center_mass, dtype=float)

    # Two rows: plane indicators on top, sections below. Nothing overlaps the data.
    fig = plt.figure(figsize=(11, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.85, 3.0], hspace=-0.08, wspace=0.22)
    axes = [fig.add_subplot(gs[1, i]) for i in range(2)]

    ref_mesh = meshes[REFERENCE][0]
    for i, (title, normal, _h, _v) in enumerate(VIEWS):
        ax3d = fig.add_subplot(gs[0, i], projection="3d")
        draw_plane_indicator(ax3d, ref_mesh, normal, origin)
        ax3d.set_title(title, fontsize=10, pad=0)

    for ax, (title, normal, horiz, vert) in zip(axes, VIEWS):
        for label, (mesh, colour) in meshes.items():
            polys = section_polylines(mesh, normal, horiz, vert, origin)
            if not polys:
                raise SystemExit(f"{label}: empty section for normal {normal}")
            for k, p in enumerate(polys):
                ax.plot(p[:, 0], p[:, 1], color=colour, lw=1.4,
                        label=label if k == 0 else None)
        ax.set_xlabel(f"${horiz}$", labelpad=1)
        ax.set_ylabel(f"${vert}$")
        ax.set_aspect("equal")
        ax.grid(alpha=0.25, lw=0.5)
        ax.axhline(0, color="0.85", lw=0.6, zorder=0)
        ax.axvline(0, color="0.85", lw=0.6, zorder=0)
        # Flow direction: inflow is +x and both views contain x.
        ax.annotate("", xy=(0.155, 0.95), xytext=(0.02, 0.95), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color="0.35", lw=1.2))
        ax.text(0.17, 0.935, r"flow $v_\infty$", transform=ax.transAxes,
                fontsize=8, color="0.35")
        # Legend outside the data area: the sections fill the panel.
        ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.19),
                  ncol=2, frameon=False)
    out_png = args.out_dir / "fig18_cross_sections.png"
    out_pdf = args.out_dir / "fig18_cross_sections.pdf"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    plt.close(fig)

    meta_out = {
        "cut_plane_origin": origin.tolist(),
        "origin_source": f"centroid of '{REFERENCE}'",
        "views": [{"title": t, "normal": n, "horizontal": h, "vertical": v}
                  for t, n, h, v in VIEWS],
        "geometries": meta,
    }
    (args.out_dir / "fig18_cross_sections.json").write_text(json.dumps(meta_out, indent=2))

    print("cut-plane origin (centroid of "
          f"{REFERENCE}): [{origin[0]:.4f}, {origin[1]:.4f}, {origin[2]:.4f}]")
    for label, m in meta.items():
        c = m["centroid"]
        print(f"  {label:<14} faces {m['n_faces']:>6}  watertight {m['watertight']}  "
              f"volume {m['volume']:.4f}  centroid [{c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f}]")
    print(f"wrote {out_png}\n      {out_pdf}")


if __name__ == "__main__":
    main()

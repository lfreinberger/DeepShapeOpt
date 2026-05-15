from __future__ import annotations

import os
import re
import numpy as np
from pathlib import Path


def _get_pyplot():
    import matplotlib.pyplot as plt

    return plt


def _get_poly3d_collection():
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    return Poly3DCollection


def to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return x


def _plot_obj_con_axes(
    ax1,
    iterations,
    history_obj,
    history_con,
    limit_con,
    obj_label="Objective",
    con_label="Constraint",
):
    """Plot objective on left axis and constraint/volume on right axis."""
    l1, = ax1.plot(
        iterations,
        history_obj,
        marker="o",
        color="tab:blue",
        label=obj_label,
    )

    ax1.set_ylabel(obj_label, color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, linestyle="--", alpha=0.4)

    lines = [l1]

    if history_con is not None and not np.all(np.isnan(history_con)):
        ax2 = ax1.twinx()

        l2, = ax2.plot(
            iterations,
            history_con,
            marker="s",
            color="tab:orange",
            label=con_label,
        )

        ax2.set_ylabel(con_label, color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")
        finite_con = history_con[np.isfinite(history_con)]
        finite_limit = np.asarray([limit_con], dtype=float) if limit_con is not None else np.asarray([])
        finite_values = np.concatenate([finite_con, finite_limit[np.isfinite(finite_limit)]])
        if finite_values.size and np.nanmin(finite_values) >= 0:
            ax2.set_ylim(bottom=0)

        if limit_con is not None:
            ax2.axhline(
                limit_con,
                linestyle=":",
                color="tab:orange",
                alpha=0.7,
                label=f"{con_label} limit",
            )

        lines.append(l2)

    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="best")


def plot_optimization_history(
    history_obj,
    history_con=None,
    limit_con=None,
    result_dir=None,
    obj_label="Objective",
    con_label="Constraint",
):
    plt = _get_pyplot()

    history_obj = np.asarray(to_numpy(history_obj), dtype=float)

    if history_con is not None:
        history_con = np.asarray(to_numpy(history_con), dtype=float)

    if limit_con is not None:
        limit_con = float(to_numpy(limit_con))

    iterations = np.arange(len(history_obj))

    has_con = history_con is not None and not np.all(np.isnan(history_con))

    obj_abs = history_obj.copy()
    con_abs = history_con.copy() if has_con else None
    limit_abs = limit_con

    obj0 = history_obj[0] if history_obj[0] != 0 else 1.0
    obj_norm = history_obj / obj0

    con_norm = None
    limit_norm = None
    if has_con:
        con0 = history_con[0] if history_con[0] != 0 else 1.0
        con_norm = history_con / con0
        if limit_con is not None:
            limit_norm = limit_con / con0

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    _plot_obj_con_axes(
        ax_top, iterations, obj_abs, con_abs, limit_abs,
        obj_label=obj_label,
        con_label=con_label,
    )
    ax_top.set_title("Absolute values")

    _plot_obj_con_axes(
        ax_bot, iterations, obj_norm, con_norm, limit_norm,
        obj_label="Objective (J/J\u2080)",
        con_label=f"{con_label} / initial",
    )
    ax_bot.set_xlabel("Iteration")
    ax_bot.set_title("Normalized values")

    plt.tight_layout()
    plt.savefig(result_dir / "optimization_history.png", dpi=150)
    plt.close()


def plot_residuals_from_log(logfile, output_dir="."):
    plt = _get_pyplot()

    patterns = {
        "Ux": re.compile(r"Solving for Ux, Initial residual = ([0-9eE+.\-]+)"),
        "Uy": re.compile(r"Solving for Uy, Initial residual = ([0-9eE+.\-]+)"),
        "Uz": re.compile(r"Solving for Uz, Initial residual = ([0-9eE+.\-]+)"),
        "p": re.compile(r"Solving for p, Initial residual = ([0-9eE+.\-]+)"),
        "Uaas1x": re.compile(r"Solving for Uaas1x, Initial residual = ([0-9eE+.\-]+)"),
        "Uaas1y": re.compile(r"Solving for Uaas1y, Initial residual = ([0-9eE+.\-]+)"),
        "Uaas1z": re.compile(r"Solving for Uaas1z, Initial residual = ([0-9eE+.\-]+)"),
        "paas1": re.compile(r"Solving for paas1, Initial residual = ([0-9eE+.\-]+)"),
        "Uaas2x": re.compile(r"Solving for Uaas2x, Initial residual = ([0-9eE+.\-]+)"),
        "Uaas2y": re.compile(r"Solving for Uaas2y, Initial residual = ([0-9eE+.\-]+)"),
        "Uaas2z": re.compile(r"Solving for Uaas2z, Initial residual = ([0-9eE+.\-]+)"),
        "paas2": re.compile(r"Solving for paas2, Initial residual = ([0-9eE+.\-]+)"),
        "continuity_global": re.compile(
            r"time step continuity errors : sum local = [0-9eE+.\-]+, global = ([0-9eE+.\-]+), cumulative = [0-9eE+.\-]+"
        ),
        "continuity_cumulative": re.compile(
            r"time step continuity errors : sum local = [0-9eE+.\-]+, global = [0-9eE+.\-]+, cumulative = ([0-9eE+.\-]+)"
        ),
    }

    data = {k: [] for k in patterns}

    with open(logfile, "r") as f:
        for line in f:
            for key, pat in patterns.items():
                m = pat.search(line)
                if m:
                    data[key].append(float(m.group(1)))

    os.makedirs(output_dir, exist_ok=True)

    def add_legend_if_needed():
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles:
            plt.legend()

    # Skip first outer iteration: OpenFOAM reports an artificially low
    # initial residual on iter 1 (uniform fields => RHS ~ 0).
    skip = 0

    # Primal residuals
    fig1 = plt.figure(figsize=(9, 5))
    for key in ["Ux", "Uy", "Uz", "p"]:
        if len(data[key]) > skip:
            plt.semilogy(range(skip + 1, len(data[key]) + 1), data[key][skip:], label=key)
    plt.xlabel("Iteration")
    plt.ylabel("Initial residual")
    plt.title("Primal residuals")
    plt.grid(True)
    add_legend_if_needed()
    plt.tight_layout()
    fig1.savefig(os.path.join(output_dir, "residuals_primal.png"), dpi=150)
    plt.close(fig1)

    # Adjoint residuals 1
    fig2 = plt.figure(figsize=(9, 5))
    for key in ["Uaas1x", "Uaas1y", "Uaas1z", "paas1"]:
        if len(data[key]) > skip:
            plt.semilogy(range(skip + 1, len(data[key]) + 1), data[key][skip:], label=key)
    plt.xlabel("Iteration")
    plt.ylabel("Initial residual")
    plt.title("Adjoint residuals")
    plt.grid(True)
    add_legend_if_needed()
    plt.tight_layout()
    fig2.savefig(os.path.join(output_dir, "residuals_adjoint.png"), dpi=150)
    plt.close(fig2)

    # Adjoint residuals 2
    fig3 = plt.figure(figsize=(9, 5))
    for key in ["Uaas2x", "Uaas2y", "Uaas2z", "paas2"]:
        if len(data[key]) > skip:
            plt.semilogy(range(skip + 1, len(data[key]) + 1), data[key][skip:], label=key)
    plt.xlabel("Iteration")
    plt.ylabel("Initial residual")
    plt.title("Adjoint residuals 2")
    plt.grid(True)
    add_legend_if_needed()
    plt.tight_layout()
    fig3.savefig(os.path.join(output_dir, "residuals_adjoint_2.png"), dpi=150)
    plt.close(fig3)

    # Continuity
    fig4 = plt.figure(figsize=(9, 5))
    if len(data["continuity_global"]) > skip:
        vals = [abs(x) for x in data["continuity_global"][skip:]]
        plt.semilogy(range(skip + 1, skip + 1 + len(vals)), vals, label="|global continuity|")
    if len(data["continuity_cumulative"]) > skip:
        vals = [abs(x) for x in data["continuity_cumulative"][skip:]]
        plt.semilogy(range(skip + 1, skip + 1 + len(vals)), vals, label="|cumulative continuity|")
    plt.xlabel("Iteration")
    plt.ylabel("Absolute value")
    plt.title("Continuity error")
    plt.grid(True)
    add_legend_if_needed()
    plt.tight_layout()
    fig4.savefig(os.path.join(output_dir, "residuals_continuity.png"), dpi=150)
    plt.close(fig4)


def save_shape_snapshot(
    verts: torch.Tensor,
    faces: torch.Tensor,
    design_domain: torch.Tensor,
    out_path: Path,
    view_axis: str = "x",
    figsize=(6, 6),
    dpi=200,
    title: str | None = None,
    show_axes: bool = True,
):
    plt = _get_pyplot()
    Poly3DCollection = _get_poly3d_collection()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    v = verts.detach().cpu().numpy()
    f = faces.detach().cpu().numpy()

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # --- mesh ---
    triangles = v[f]
    mesh = Poly3DCollection(
        triangles,
        facecolor="lightgray",
        edgecolor="black",
        linewidth=0.05,
        alpha=1.0,
    )
    ax.add_collection3d(mesh)

    # --- equal scaling ---
    mins = design_domain[0].cpu().numpy()
    maxs = design_domain[1].cpu().numpy()

    center = 0.5 * (mins + maxs)
    max_range = 0.5 * (maxs - mins).max()

    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)

    # --- coordinate axes ---
    if show_axes:
        axis_len = max_range * 1.2

        # X axis (red)
        ax.quiver(
            center[0], center[1], center[2],
            axis_len, 0, 0,
            arrow_length_ratio=0.1
        )
        ax.text(center[0] + axis_len, center[1], center[2], "X")

        # Y axis (green)
        ax.quiver(
            center[0], center[1], center[2],
            0, axis_len, 0,
            arrow_length_ratio=0.1
        )
        ax.text(center[0], center[1] + axis_len, center[2], "Y")

        # Z axis (blue)
        ax.quiver(
            center[0], center[1], center[2],
            0, 0, axis_len,
            arrow_length_ratio=0.1
        )
        ax.text(center[0], center[1], center[2] + axis_len, "Z")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.grid(True)

    else:
        ax.set_axis_off()

    # --- view direction ---
    view_map = {
        "x":   (0, 0),
        "-x":  (0, 180),
        "y":   (0, 90),
        "-y":  (0, -90),
        "z":   (90, -90),
        "-z":  (-90, -90),
    }

    if view_axis not in view_map:
        raise ValueError(f"Unsupported view_axis '{view_axis}'")

    elev, azim = view_map[view_axis]
    ax.view_init(elev=elev, azim=azim)

    if title is not None:
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_center_constraint_history(
    history_objective_raw,
    history_objective_total,
    history_center_penalty,
    history_center_penalty_weighted,
    output_dir,
):
    plt = _get_pyplot()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    iters = range(len(history_objective_raw))

    # Plot 1: objective comparison
    plt.figure(figsize=(8, 5))
    plt.plot(iters, history_objective_raw, label="J raw")
    plt.plot(iters, history_objective_total, label="J total = J raw + lambda * penalty")
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.title("Objective with and without center penalty")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "center_constraint_objectives.png", dpi=200)
    plt.close()

    # Plot 2: unweighted penalty
    plt.figure(figsize=(8, 5))
    plt.plot(iters, history_center_penalty_weighted, label="lambda * center penalty")
    plt.xlabel("Iteration")
    plt.ylabel("Penalty")
    plt.title("Center penalty history")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "center_penalty.png", dpi=200)
    plt.close()


def plot_convergence_diagnostics(diagnostics: dict, output_dir: Path):
    """Plot convergence diagnostic quantities over optimization iterations.

    Parameters
    ----------
    diagnostics : dict
        Keys: "obj_change", "grad_norm", "mma_ch", "vol_constraint", "sens_norm".
        Each value is a list of per-iteration scalars.
    output_dir : Path
        Directory where ``convergence_diagnostics.png`` is saved.
    """
    plt = _get_pyplot()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    panels = [
        ("obj_change",      "Relative objective change",                 True),
        ("grad_norm",       "Gradient norm ||dJ/dp||",                   True),
        ("mma_ch",          "MMA design change (ch)",                    True),
        ("vol_constraint",  "Volume constraint value",                   False),
        ("sens_norm",       "Sensitivity norm ||s||",                    True),
        ("sens_to_grad_ratio",         "||dJ/dp|| / ||s||",               True),
        ("conservative_max_proj_dist", "Conservative max projection dist", True),
        ("conservative_l1_ratio",      "Conservative L1 ratio (STL/OF)",  False),
        ("conservative_vec_norm_ratio", "Conservative vec-norm ratio",   False),
    ]

    # Only plot panels that have data
    panels = [(k, label, log) for k, label, log in panels if k in diagnostics and len(diagnostics[k]) > 0]
    if not panels:
        return

    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(8, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (key, label, use_log) in zip(axes, panels):
        vals = np.asarray(diagnostics[key], dtype=float)
        iters = np.arange(1, len(vals) + 1)

        if use_log:
            ax.semilogy(iters, np.abs(vals), marker="o", markersize=3)
        else:
            ax.plot(iters, vals, marker="o", markersize=3)
            ax.axhline(0, color="k", linewidth=0.5, linestyle="--")

        ax.set_ylabel(label)
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("Iteration")
    fig.suptitle("Convergence diagnostics", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "convergence_diagnostics.png", dpi=150)
    plt.close(fig)

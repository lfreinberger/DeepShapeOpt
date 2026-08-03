"""Mesh-independence study on fixed geometries (reviewer comment 8).

The reviewer asks for the mesh resolution, the residual levels at convergence,
and a mesh-independence check on the drag, noting that the reported reductions
(14-25%) and the FFD comparison (16.7 vs 15.7%) are only as meaningful as the
numerical uncertainty of the objective.

Design decisions:

* FIXED geometries, no optimization loop. The reviewer wants a drag-vs-mesh
  table; the stored final STLs answer that at a fraction of the cost. No torch,
  no decoder, no lattice is involved here at all.
* The shipped ``foam_case`` template is NEVER modified. ``prepare_foam_runtime``
  (foam_utils.py:22) copies it, and only the copy is patched.
* ``configure_foam_runtime`` is deliberately not used: it is dead code and is
  incompatible with the shipped optimisationDict (it looks for
  primalSolvers.p1 / adjointSolvers.as1 while the dict defines op1 / adjS1, and
  it needs an __ADJOINT_TIMES__ marker the shipped Allrun does not contain).
* Refinement is patched by anchored regex with an exact match count, rather than
  through a FoamFile round-trip: snappy's nested dicts and its ``(6 6)`` /
  ``((0.05 6))`` tuple syntax are where a dictionary rewrite is most likely to
  reformat something silently.
* Logs are copied out before the next run. ``run_openfoam_case`` calls
  ``FoamCase.clean()`` on entry (foam_utils.py:82-85) and the optimization
  scripts rmtree the runtime dir at the end (optimize_drag_latent.py:315-317);
  neither must be allowed to destroy the evidence here.

Gate: run ``--only cyl_init:6`` first and check the reported drag against the
``objective`` column of iteration 1 of the corresponding stored history. If that
reproduces, the standalone harness matches the manuscript pipeline and the rest
of the grid is trustworthy.

Requires a sourced OpenFOAM environment ($WM_PROJECT_DIR); see
scripts/revision/mesh_study.slrm.

Usage:
    uv run python scripts/revision/mesh_study.py --only cyl_init:6      # gate
    uv run python scripts/revision/mesh_study.py                        # full grid
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import time
from pathlib import Path

import numpy as np

from deepshapeopt import foam_utils

REPO = Path(__file__).resolve().parents[2]
TEMPLATE = REPO / "experiments/drag_cube/foam_case"
ARCHIVE_HEAVY = Path(
    "/storage/lfrei/Archive/DeepShapeOpt_archive_paper/experiments/optimization"
    "/drag_optimization_cube"
)
ARCHIVE_LIGHT = Path(
    "/storage/lfrei/Archive/DeepShapeOpt-old-private-backup/experiments/optimization"
    "/drag_optimization_cube"
)
OBJECTIVE_PATH = "optimisation/objective/0/dragadjS1"

# The three shapes behind the contested 16.71% vs 15.69% comparison.
# NOTE: shape_0000.stl was not archived (stl_series starts at 0001), so the
# "initial" geometry is the shape after one MMA update. That is what the gate
# compares against, so the comparison is still exact.
GEOMETRIES = {
    "cyl_init": ARCHIVE_HEAVY / "results_cube_with_cylinders/stl_series/shape_0001.stl",
    "cyl_final_neural": ARCHIVE_LIGHT
    / "results_cube_with_cylinders/optimization/current_shape.stl",
    "cyl_final_ffd7": ARCHIVE_LIGHT
    / "results_ffd_cube_with_cylinders_7x7x7/optimization/current_shape.stl",
}
# Gate reference: iteration 1 of this run's stored history.
GATE = ("cyl_init", 6, ARCHIVE_LIGHT / "results_cube_with_cylinders/optimization"
        / "optimization_history.csv")

LEVELS = [5, 6, 7]  # 6 is the shipped baseline

# (regex with one capture group for the value, replacement template) per level.
# Each pattern MUST match exactly once, or the patch is rejected.
# Anchors chosen from the shipped dict so each matches exactly once:
#   features:            file "shape.eMesh";  /  level 6;   // was 7
#   refinementSurfaces:  level (6 6);   // was (5 6)      <- only "level (" in the file
#   refinementRegions:   levels ((0.05 6));               <- anchored on the 0.05 distance
# The refinementBox entry `levels ((1E15 3))` is deliberately NOT matched: it stays
# at level 3 so the study varies one thing only.
PATCH = {
    "features_level": (
        re.compile(r"(file\s+\"shape\.eMesh\"\s*;\s*level\s+)(\d+)(\s*;)"),
        lambda lvl: rf"\g<1>{lvl}\g<3>",
    ),
    "refinementSurfaces_level": (
        re.compile(r"(level\s*\(\s*)(\d+)(\s+)(\d+)(\s*\)\s*;)"),
        lambda lvl: rf"\g<1>{lvl}\g<3>{lvl}\g<5>",
    ),
    "refinementRegions_distance": (
        re.compile(r"(levels\s*\(\s*\(\s*0\.05\s+)(\d+)(\s*\)\s*\))"),
        lambda lvl: rf"\g<1>{lvl}\g<3>",
    ),
}
# Cell caps must grow with the level or snappy silently stops refining.
CAPS = {5: (100_000, 2_000_000), 6: (100_000, 2_000_000), 7: (500_000, 8_000_000)}

RESIDUAL_PATTERNS = {
    name: re.compile(rf"Solving for {name}, Initial residual = ([0-9eE+.\-]+)")
    for name in ("Ux", "Uy", "Uz", "p", "Uaas1x", "Uaas1y", "Uaas1z", "paas1")
}
CONV_OK = re.compile(r"(\w+) solution converged in (\d+) iterations")
CONV_CAP = re.compile(r"(\w+) solution reached max\. number of iterations (\d+)")


def patch_refinement(case_dir: Path, level: int) -> dict:
    """Patch snappyHexMeshDict on the RUNTIME COPY only. Fails loudly."""
    path = case_dir / "system" / "snappyHexMeshDict"
    text = path.read_text()
    applied = {}

    for name, (pattern, repl) in PATCH.items():
        text, n = pattern.subn(repl(level), text)
        if n != 1:
            raise SystemExit(
                f"{path}: pattern '{name}' matched {n} times, expected exactly 1. "
                "Refusing to patch -- the resulting mesh would not be what the table claims."
            )
        applied[name] = level

    max_local, max_global = CAPS[level]
    for key, val in (("maxLocalCells", max_local), ("maxGlobalCells", max_global)):
        text, n = re.subn(rf"({key}\s+)(\d+)(\s*;)", rf"\g<1>{val}\g<3>", text)
        if n != 1:
            raise SystemExit(f"{path}: '{key}' matched {n} times, expected 1")
        applied[key] = val

    path.write_text(text)
    return applied


def parse_checkmesh(log: Path) -> dict:
    if not log.is_file():
        return {"error": "log.checkMesh missing"}
    text = log.read_text()
    out: dict = {"mesh_ok": "Mesh OK" in text}
    for key, pat in (
        ("cells", r"^\s*cells:\s+(\d+)"),
        ("hexahedra", r"^\s*hexahedra:\s+(\d+)"),
        ("polyhedra", r"^\s*polyhedra:\s+(\d+)"),
    ):
        m = re.search(pat, text, re.M)
        out[key] = int(m.group(1)) if m else None
    # OpenFOAM writes "Mesh non-orthogonality Max: 55.06 average: 9.25", not
    # "Max non-orthogonality = ...". Getting this wrong silently yields None.
    m = re.search(r"Mesh non-orthogonality Max:\s*([0-9.eE+-]+)\s+average:\s*([0-9.eE+-]+)", text)
    out["max_nonortho"] = float(m.group(1)) if m else None
    out["avg_nonortho"] = float(m.group(2)) if m else None
    m = re.search(r"Max skewness = ([0-9.eE+-]+)", text)
    out["max_skewness"] = float(m.group(1)) if m else None
    m = re.search(r"Max aspect ratio = ([0-9.eE+-]+)", text)
    out["max_aspect_ratio"] = float(m.group(1)) if m else None
    out["nonortho_check_ok"] = "Non-orthogonality check OK" in text
    m = re.search(r"OPENFOAM=(\S+)", text)
    out["openfoam_version"] = m.group(1) if m else None
    m = re.search(r"Build\s*:\s*(\S+)", text)
    out["openfoam_build"] = m.group(1) if m else None
    return out


def parse_solver_log(log: Path) -> dict:
    """Final residual per field, plus per-solver convergence status."""
    if not log.is_file():
        return {"error": "log.adjointOptimisationFoam missing"}
    last: dict[str, float] = {}
    converged: dict[str, dict] = {}
    with log.open(errors="ignore") as fh:
        for line in fh:
            for name, pat in RESIDUAL_PATTERNS.items():
                m = pat.search(line)
                if m:
                    last[name] = float(m.group(1))
            m = CONV_OK.search(line)
            if m:
                converged[m.group(1)] = {"converged": True, "iterations": int(m.group(2))}
            m = CONV_CAP.search(line)
            if m:
                converged[m.group(1)] = {"converged": False, "iterations": int(m.group(2))}
    return {"final_residuals": last, "solvers": converged}


def gate_reference(history_csv: Path) -> float | None:
    """Drag at iteration 1 of a stored manuscript run."""
    if not history_csv.is_file():
        return None
    with history_csv.open() as fh:
        rows = list(csv.DictReader(fh))
    return float(rows[0]["objective"]) if rows else None


def run_one(geom: str, level: int, out_dir: Path, keep_case: bool) -> dict:
    stl = GEOMETRIES[geom]
    if not stl.is_file():
        raise SystemExit(f"missing geometry: {stl}")

    run_name = f"meshstudy_{geom}_L{level}"
    case = foam_utils.prepare_foam_runtime(TEMPLATE, run_name=run_name)
    applied = patch_refinement(case, level)

    tri = case / "constant" / "triSurface"
    tri.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(stl, tri / "shape.stl")

    t0 = time.time()
    foam_utils.run_openfoam_case(case, verbose=False)
    wall = time.time() - t0

    # Copy the evidence out BEFORE anything can clean the case.
    log_dir = out_dir / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    for log in sorted(case.glob("log.*")):
        shutil.copyfile(log, log_dir / log.name)

    rec: dict = {
        "geometry": geom, "level": level, "run_name": run_name,
        "stl": str(stl), "patched": applied, "wall_clock_s": wall,
    }
    rec.update(parse_checkmesh(log_dir / "log.checkMesh"))
    rec.update(parse_solver_log(log_dir / "log.adjointOptimisationFoam"))

    try:
        rec["drag"] = float(foam_utils.read_objective(case, objective_path=OBJECTIVE_PATH))
    except Exception as exc:  # noqa: BLE001 - record and continue; one failure must not lose the grid
        rec["drag"] = None
        rec["drag_error"] = str(exc)

    if not keep_case:
        shutil.rmtree(case, ignore_errors=True)
    else:
        rec["case_dir"] = str(case)
    return rec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=None,
                        help="run a subset as geom:level, e.g. cyl_init:6 (the gate)")
    parser.add_argument("--out-dir", type=Path,
                        default=REPO / "revision_artifacts" / "mesh_study")
    parser.add_argument("--keep-cases", action="store_true",
                        help="keep the runtime case directories (large)")
    parser.add_argument("--reparse", action="store_true",
                        help="re-derive results from logs already copied into --out-dir, "
                             "without re-running OpenFOAM (drag is taken from the stored JSON)")
    args = parser.parse_args()

    if args.reparse:
        prev_path = args.out_dir / "mesh_study.json"
        if not prev_path.is_file():
            raise SystemExit(f"--reparse needs {prev_path}, which does not exist")
        prev = json.loads(prev_path.read_text())
        by_run = {r["run_name"]: r for r in prev}
        results = []
        for log_dir in sorted(p for p in args.out_dir.iterdir() if p.is_dir()):
            old = by_run.get(log_dir.name, {})
            rec = {k: old.get(k) for k in
                   ("geometry", "level", "run_name", "stl", "patched", "wall_clock_s", "drag")}
            rec["run_name"] = log_dir.name
            rec.update(parse_checkmesh(log_dir / "log.checkMesh"))
            rec.update(parse_solver_log(log_dir / "log.adjointOptimisationFoam"))
            results.append(rec)
        prev_path.write_text(json.dumps(results, indent=2))
        for r in results:
            print(f"{r['run_name']}: cells {r.get('cells')}  drag {r.get('drag')}  "
                  f"max_nonortho {r.get('max_nonortho')}  avg {r.get('avg_nonortho')}  "
                  f"max_skew {r.get('max_skewness')}  aspect {r.get('max_aspect_ratio')}")
        print(f"\nreparsed {len(results)} run(s) -> {prev_path}")
        return

    if not TEMPLATE.is_dir():
        raise SystemExit(f"missing foam_case template: {TEMPLATE}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.only:
        jobs = []
        for spec in args.only:
            geom, _, lvl = spec.partition(":")
            if geom not in GEOMETRIES:
                raise SystemExit(f"unknown geometry '{geom}'; have {list(GEOMETRIES)}")
            jobs.append((geom, int(lvl)))
    else:
        jobs = [(g, lvl) for g in GEOMETRIES for lvl in LEVELS]

    results = []
    for geom, level in jobs:
        print(f"\n=== {geom} at refinement level {level} ===", flush=True)
        rec = run_one(geom, level, args.out_dir, args.keep_cases)
        results.append(rec)

        cells = rec.get("cells")
        print(f"  cells {cells}  mesh_ok {rec.get('mesh_ok')}  "
              f"max_nonortho {rec.get('max_nonortho')}", flush=True)
        print(f"  drag  {rec.get('drag')}   wall {rec['wall_clock_s'] / 60:.1f} min", flush=True)
        for soln, st in rec.get("solvers", {}).items():
            flag = "converged" if st["converged"] else "REACHED ITERATION CAP"
            print(f"    {soln}: {flag} at {st['iterations']} iterations", flush=True)

        (args.out_dir / "mesh_study.json").write_text(json.dumps(results, indent=2))

    # Gate check, if the gate run is part of this invocation.
    geom_g, lvl_g, hist = GATE
    gate_rec = next((r for r in results
                     if r["geometry"] == geom_g and r["level"] == lvl_g), None)
    if gate_rec and gate_rec.get("drag") is not None:
        ref = gate_reference(hist)
        if ref is not None:
            rel = abs(gate_rec["drag"] - ref) / abs(ref)
            print(f"\n=== GATE ===\n  harness drag {gate_rec['drag']:.6g} vs stored "
                  f"iteration-1 drag {ref:.6g}  ->  relative difference {rel:.3%}")
            print("  PASS: harness reproduces the manuscript pipeline" if rel < 0.01
                  else "  FAIL: do NOT trust the remaining rows until this is understood")

    # Table
    print("\n=== drag vs mesh ===")
    print(f"{'geometry':<20}{'level':>6}{'cells':>10}{'drag':>14}{'primal':>18}{'adjoint':>20}")
    for r in results:
        s = r.get("solvers", {})
        p = next((v for k, v in s.items() if k.startswith("op")), None)
        a = next((v for k, v in s.items() if k.startswith("adjS")), None)
        fmt = lambda v: (f"{v['iterations']}{'' if v['converged'] else ' (CAP)'}" if v else "-")
        drag = f"{r['drag']:.6g}" if r.get("drag") is not None else "-"
        print(f"{r['geometry']:<20}{r['level']:>6}{str(r.get('cells')):>10}{drag:>14}"
              f"{fmt(p):>18}{fmt(a):>20}")

    # Mesh-independence summary per geometry.
    print("\n=== relative change in drag between levels ===")
    for g in GEOMETRIES:
        rows = sorted([r for r in results if r["geometry"] == g and r.get("drag")],
                      key=lambda r: r["level"])
        for a, b in zip(rows, rows[1:]):
            rel = abs(b["drag"] - a["drag"]) / abs(a["drag"])
            print(f"  {g:<20} L{a['level']} -> L{b['level']}: {rel:.3%}")

    print(f"\nwrote {args.out_dir}/mesh_study.json")


if __name__ == "__main__":
    main()

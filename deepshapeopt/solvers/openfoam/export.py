"""Export one OpenFOAM time step as a per-iteration VTK multiblock."""
from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def export_vtk_for_iteration(case, case_dir: Path, vtk_series_dir: Path, e: int,
                             time_value: str | None = None):
    """
    Export one OpenFOAM time step as a per-iteration VTK multiblock.

    ``time_value`` should be the time the fields were actually read from (see
    :func:`resolve_adjoint_time`). Without it the export falls back to
    ``-latestTime``, which silently exports time 0 -- the initial fields, dressed
    up as this design iteration -- whenever reconstructPar missed the real write.

    Copies foamToVTK's whole time directory (internal.vtu + boundary/<patch>.vtp)
    to vtk_series/iter_NNNN/ and rewrites the accompanying .vtm so every boundary
    patch stays a named block ParaView can toggle in the Multi-block Inspector.
    An iterations.vtm.series index maps ParaView time to the iteration number.
    """

    # Run with the OpenFOAM environment sourced (foamlib's case.run needs
    # foamToVTK already on PATH, which only holds in fully-loaded shells).
    selector = f"-time {time_value}" if time_value is not None else "-latestTime"
    subprocess.run(
        "source $WM_PROJECT_DIR/etc/bashrc >/dev/null 2>&1; "
        f"foamToVTK {selector} > log.foamToVTK 2>&1",
        cwd=case_dir,
        shell=True,
        executable="/bin/bash",
        check=True,
    )

    vtk_series_dir = Path(vtk_series_dir) / "vtk_series"
    vtk_series_dir.mkdir(parents=True, exist_ok=True)

    vtk_root = case_dir / "VTK"
    if not vtk_root.exists():
        raise RuntimeError(f"VTK directory not found: {vtk_root}")

    vtk_dirs = [d for d in vtk_root.iterdir() if d.is_dir()]
    if not vtk_dirs:
        raise RuntimeError(f"No VTK subdirectories in {vtk_root}")

    latest_vtk_dir = max(vtk_dirs, key=lambda p: p.stat().st_mtime)
    if latest_vtk_dir.name.endswith("_0"):
        logger.warning(
            "export_vtk_for_iteration: exporting time 0 for iteration %d -- the "
            "requested time (%s) was not in the case, so this is the INITIAL "
            "field, not the solution.", e, time_value)
    src_vtm = latest_vtk_dir.with_suffix(".vtm")
    if not src_vtm.exists():
        raise RuntimeError(f"Multiblock file not found: {src_vtm}")

    iter_name = f"iter_{e:04d}"
    dst_dir = vtk_series_dir / iter_name
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    shutil.copytree(latest_vtk_dir, dst_dir)

    # Re-point the block references at the copied directory and stamp the
    # iteration number as TimeValue — the solver's own latestTime is identical
    # every iteration and would collapse the series onto a single time step.
    vtm_text = src_vtm.read_text()
    vtm_text = vtm_text.replace(f"{latest_vtk_dir.name}/", f"{iter_name}/")
    vtm_text = re.sub(r"<!-- time='[^']*' -->", f"<!-- time='{e}' -->", vtm_text)
    vtm_text = re.sub(
        r"(Name='TimeValue'[^>]*>\s*)[-+0-9.eE]+", rf"\g<1>{e}", vtm_text
    )
    dst_vtm = vtk_series_dir / f"{iter_name}.vtm"
    dst_vtm.write_text(vtm_text)
    logger.debug("Saved multiblock VTK for ParaView: %s", dst_vtm)

    # Regenerate the series index from what is on disk so reruns stay consistent.
    entries = [
        {"name": p.name, "time": int(p.stem.split("_")[1])}
        for p in sorted(vtk_series_dir.glob("iter_*.vtm"))
    ]
    series = {"file-series-version": "1.0", "files": entries}
    (vtk_series_dir / "iterations.vtm.series").write_text(json.dumps(series, indent=2))

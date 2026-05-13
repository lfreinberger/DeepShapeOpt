"""Optimization logging utilities.

Provides structured logging for optimization experiments:
- Config dump (JSON copy of experiment settings)
- Per-iteration text log with key values and timing
- CSV file with objective, constraint, and diagnostic values
"""
import csv
import json
import time
from io import TextIOWrapper
from pathlib import Path
from typing import Any, Union


class OptimizationLogger:
    """Manages per-iteration text log, CSV history, and timing for an optimization run.

    Usage::

        logger = OptimizationLogger(output_dir, specs, total_iters=30)
        for e in range(1, 31):
            logger.start_iteration(e)
            # ... compute objective, constraints, etc. ...
            logger.log_iteration(
                iteration=e,
                objective=J.item(),
                vol_constraint=vol_constraint.item(),
                grad_norm=dJ.norm().item(),
            )
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        specs: dict,
        total_iters: int,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.total_iters = total_iters

        # Timing state
        self._start_time = time.time()
        self._iter_start: float | None = None
        self._iteration_times: list[float] = []

        # CSV state
        self._csv_path = self.output_dir / "optimization_history.csv"
        self._csv_file: TextIOWrapper | None = None
        self._csv_writer: csv.DictWriter | None = None
        self._csv_columns: list[str] | None = None

        # Text log
        self._log_path = self.output_dir / "optimization_log.txt"
        self._log_file = open(self._log_path, "w")

        # Initial values for normalization (set on first log_iteration call)
        self._initial_objective: float | None = None
        self._initial_volume: float | None = None

        # Dump config
        self._dump_config(specs)

    def _dump_config(self, specs: dict):
        config_path = self.output_dir / "config_log.json"
        with open(config_path, "w") as f:
            json.dump(dict(specs), f, indent=4, default=str)
        self._write_log(f"Config written to {config_path}")

    def start_iteration(self, iteration: int):
        """Call at the beginning of each iteration to start the timer."""
        self._iter_start = time.time()

    def log_iteration(self, *, iteration: int, objective: float, **kwargs: Any):
        """Log one iteration's data to both the text log and CSV.

        Required:
            iteration: iteration number
            objective: raw objective value

        Optional (pass as keyword arguments, e.g.):
            vol_constraint, center_constraint, jacobian_constraint,
            cfd_constraint, cfd_constraint_name, constraint_target,
            volume, grad_norm, sens_norm, mma_ch, max_param,
            obj_change, sens_to_grad_ratio,
            conservative_max_proj_dist, conservative_l1_ratio,
            conservative_vec_norm_ratio,
            n_reoriented_tets, warnings (str)

        Values that are None are silently dropped.
        """
        # Drop None values
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        # Normalization
        if self._initial_objective is None:
            self._initial_objective = objective
        obj_normalized = (
            objective / self._initial_objective
            if self._initial_objective != 0
            else float("nan")
        )

        volume = kwargs.get("volume")
        if volume is not None:
            if self._initial_volume is None:
                self._initial_volume = volume
            vol_normalized = (
                volume / self._initial_volume
                if self._initial_volume != 0
                else float("nan")
            )
            kwargs["volume_normalized"] = vol_normalized

        # Timing
        iter_time, elapsed, avg_time, eta = self._compute_timing()

        # Build row
        row: dict[str, Any] = {
            "iteration": iteration,
            "objective": objective,
            "objective_normalized": obj_normalized,
        }
        row.update(kwargs)
        row["iter_time_s"] = iter_time
        row["elapsed_s"] = elapsed
        row["eta_s"] = eta

        # Write text log
        self._write_iteration_log(row)

        # Write CSV row
        self._write_csv_row(row)

    def _compute_timing(self) -> tuple[float, float, float, float]:
        """Returns (iter_time, elapsed, avg_time, eta)."""
        now = time.time()
        if self._iter_start is not None:
            iter_time = now - self._iter_start
            self._iteration_times.append(iter_time)
        else:
            iter_time = float("nan")

        elapsed = now - self._start_time
        avg_time = (
            sum(self._iteration_times) / len(self._iteration_times)
            if self._iteration_times
            else 0.0
        )
        completed = len(self._iteration_times)
        remaining = max(0, self.total_iters - completed)
        eta = avg_time * remaining
        return iter_time, elapsed, avg_time, eta

    def _write_iteration_log(self, row: dict[str, Any]):
        iteration = row["iteration"]
        lines = [
            f"=== Iteration {iteration}/{self.total_iters} ===",
            f"  Objective:            {row['objective']:.6e}",
            f"  Objective (normed):   {row['objective_normalized']:.6e}",
        ]

        # Constraints
        if "vol_constraint" in row:
            lines.append(f"  Volume constraint:    {row['vol_constraint']:.6e}")
        if "volume" in row:
            lines.append(f"  Volume:               {row['volume']:.6f}")
        if "volume_normalized" in row:
            lines.append(f"  Volume (normed):      {row['volume_normalized']:.6e}")
        if "center_constraint" in row:
            lines.append(f"  Center constraint:    {row['center_constraint']:.6e}")
        if "jacobian_constraint" in row:
            lines.append(f"  Jacobian constraint:  {row['jacobian_constraint']:.6e}")
        if "cfd_constraint" in row:
            name = row.get("cfd_constraint_name", "CFD constraint")
            lines.append(f"  {name}:  {row['cfd_constraint']:.6e}")
        if "constraint_target" in row:
            lines.append(f"  Constraint target:    {row['constraint_target']:.6e}")

        # Diagnostics
        if "grad_norm" in row:
            lines.append(f"  Gradient norm:        {row['grad_norm']:.6e}")
        if "sens_norm" in row:
            lines.append(f"  Sensitivity norm:     {row['sens_norm']:.6e}")
        if "mma_ch" in row:
            lines.append(f"  MMA step (ch):        {row['mma_ch']:.6e}")
        if "max_param" in row:
            lines.append(f"  Max param value:      {row['max_param']:.6f}")
        if "obj_change" in row:
            lines.append(f"  Obj change (rel):     {row['obj_change']:.6e}")
        if "sens_to_grad_ratio" in row:
            lines.append(f"  Sens/grad ratio:      {row['sens_to_grad_ratio']:.6e}")

        # Conservative transfer diagnostics
        if "conservative_max_proj_dist" in row:
            lines.append(f"  Cons. max proj dist:  {row['conservative_max_proj_dist']:.6e}")
        if "conservative_l1_ratio" in row:
            lines.append(f"  Cons. L1 ratio:       {row['conservative_l1_ratio']:.6e}")
        if "conservative_vec_norm_ratio" in row:
            lines.append(f"  Cons. vec norm ratio: {row['conservative_vec_norm_ratio']:.6e}")

        # Warnings
        if "n_reoriented_tets" in row and row["n_reoriented_tets"] > 0:
            lines.append(f"  WARNING: {row['n_reoriented_tets']} tets with negative volume")
        if "warnings" in row and row["warnings"]:
            lines.append(f"  WARNING: {row['warnings']}")

        # Timing
        lines.append(f"  Iter time:            {row.get('iter_time_s', 0):.2f}s")
        lines.append(f"  Elapsed:              {row.get('elapsed_s', 0) / 60:.2f} min")
        lines.append(f"  ETA:                  {row.get('eta_s', 0) / 60:.2f} min")
        lines.append("")

        text = "\n".join(lines)
        self._write_log(text)

    def _write_log(self, text: str):
        self._log_file.write(text + "\n")
        self._log_file.flush()

    def _write_csv_row(self, row: dict[str, Any]):
        # Exclude non-numeric/meta fields from CSV
        exclude = {"warnings", "cfd_constraint_name"}
        csv_row = {k: v for k, v in row.items() if k not in exclude}

        if self._csv_writer is None:
            self._csv_columns = list(csv_row.keys())
            self._csv_file = open(self._csv_path, "w", newline="")
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=self._csv_columns, extrasaction="ignore",
            )
            self._csv_writer.writeheader()

        # If new columns appear in later iterations, rewrite the file
        new_keys = [k for k in csv_row if k not in self._csv_columns]
        if new_keys:
            self._csv_columns.extend(new_keys)
            self._rewrite_csv_with_new_columns()

        self._csv_writer.writerow(csv_row)
        self._csv_file.flush()

    def _rewrite_csv_with_new_columns(self):
        """Re-open CSV with updated column set (happens rarely, e.g. constraint added mid-run)."""
        self._csv_file.close()
        # Read existing rows
        rows = []
        with open(self._csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        # Rewrite with new columns
        self._csv_file = open(self._csv_path, "w", newline="")
        self._csv_writer = csv.DictWriter(
            self._csv_file, fieldnames=self._csv_columns, extrasaction="ignore",
        )
        self._csv_writer.writeheader()
        for r in rows:
            self._csv_writer.writerow(r)

    def close(self):
        """Flush and close log and CSV files."""
        if self._log_file and not self._log_file.closed:
            self._log_file.close()
        if self._csv_file and not self._csv_file.closed:
            self._csv_file.close()

    def __del__(self):
        self.close()

"""Per-iteration record of a run: ``optimization_history.csv`` and ``optimization_log.txt``."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any


class RunHistory:
    """Appends one row per iteration; new columns appearing later rewrite the CSV header."""

    def __init__(self, out_dir: Path, config: dict, total_iters: int):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.total_iters = int(total_iters)
        self._start = time.time()
        self._iter_start: float | None = None
        self._times: list[float] = []
        self._csv_path = self.out_dir / "optimization_history.csv"
        self._csv_file = None
        self._writer = None
        self._columns: list[str] = []
        self._log = open(self.out_dir / "optimization_log.txt", "w")
        self._initial_objective: float | None = None
        with open(self.out_dir / "config_log.json", "w") as f:
            json.dump(config, f, indent=2, default=str)

    def start_iteration(self, iteration: int) -> None:
        self._iter_start = time.time()

    def log_iteration(self, *, iteration: int, objective: float, **values: Any) -> None:
        values = {k: v for k, v in values.items() if v is not None}
        if self._initial_objective is None:
            self._initial_objective = float(objective)
        normed = objective / self._initial_objective if self._initial_objective else float("nan")
        now = time.time()
        iter_time = now - self._iter_start if self._iter_start is not None else float("nan")
        if self._iter_start is not None:
            self._times.append(iter_time)
        avg = sum(self._times) / len(self._times) if self._times else 0.0
        eta = avg * max(0, self.total_iters - len(self._times))

        row: dict[str, Any] = {"iteration": iteration, "objective": objective, "objective_normalized": normed}
        row.update(values)
        row.update(iter_time_s=iter_time, elapsed_s=now - self._start, eta_s=eta)
        self._write_text(row)
        self._write_csv(row)

    def _write_text(self, row: dict[str, Any]) -> None:
        lines = [f"=== Iteration {row['iteration']}/{self.total_iters} ==="]
        for key, value in row.items():
            if key == "iteration":
                continue
            if isinstance(value, float):
                lines.append(f"  {key:28s} {value:.6e}")
            else:
                lines.append(f"  {key:28s} {value}")
        self._log.write("\n".join(lines) + "\n\n")
        self._log.flush()

    def _write_csv(self, row: dict[str, Any]) -> None:
        numeric = {k: v for k, v in row.items() if isinstance(v, (int, float, str))}
        if self._writer is None:
            self._columns = list(numeric)
            self._csv_file = open(self._csv_path, "w", newline="")
            self._writer = csv.DictWriter(self._csv_file, fieldnames=self._columns, extrasaction="ignore")
            self._writer.writeheader()
        new = [k for k in numeric if k not in self._columns]
        if new:
            self._columns.extend(new)
            self._csv_file.close()
            with open(self._csv_path, newline="") as f:
                rows = list(csv.DictReader(f))
            self._csv_file = open(self._csv_path, "w", newline="")
            self._writer = csv.DictWriter(self._csv_file, fieldnames=self._columns, extrasaction="ignore")
            self._writer.writeheader()
            for r in rows:
                self._writer.writerow(r)
        self._writer.writerow(numeric)
        self._csv_file.flush()

    def close(self) -> None:
        if not self._log.closed:
            self._log.close()
        if self._csv_file is not None and not self._csv_file.closed:
            self._csv_file.close()

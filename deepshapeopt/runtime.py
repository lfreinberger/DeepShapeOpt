"""Small runtime helpers for optimization scripts."""

from __future__ import annotations

import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Any


def is_debug_enabled(specs: dict[str, Any]) -> bool:
    opt_cfg = specs.get("optimization", {})
    return bool(specs.get("debug", opt_cfg.get("debug", False)))


def configure_logging(debug: bool, log_file: Path | None = None) -> None:
    """Install a clean stdout (+ optional file) handler on the root logger.

    Third-party libraries (notably DeepSDFStruct) attach their own handlers
    to named loggers with a timestamped formatter; without intervention
    those messages get emitted twice — once by their handler and once via
    propagation through ours. Here we detach existing handlers, install
    ours with a bare ``%(message)s`` format, and stop named loggers that
    own handlers from propagating into the root. In non-debug mode we also
    raise their level to ``WARNING`` and silence ``DeepSDFStruct``-origin
    ``UserWarning``s (PyTorch tensor-construction noise).
    """
    level = logging.DEBUG if debug else logging.INFO

    formatter = logging.Formatter("%(message)s")
    handlers: list[logging.Handler] = []

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    handlers.append(stream_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()
    for h in handlers:
        root.addHandler(h)
    root.setLevel(level)

    for name, lg in list(logging.Logger.manager.loggerDict.items()):
        if not isinstance(lg, logging.Logger) or not lg.handlers:
            continue
        lg.propagate = False
        if not debug:
            lg.setLevel(logging.WARNING)

    if not debug:
        warnings.filterwarnings("ignore", category=UserWarning, module=r"DeepSDFStruct\..*")
        warnings.filterwarnings("ignore", category=UserWarning, module=r"torch\..*")


def log_iteration_summary(logger: logging.Logger, **values: Any) -> None:
    fields = []
    for key, value in values.items():
        if value is None:
            continue
        if isinstance(value, float):
            fields.append(f"{key}={value:.6e}")
        else:
            fields.append(f"{key}={value}")
    logger.info("  " + " | ".join(fields))


def log_timing(
    logger: logging.Logger,
    iter_start: float,
    run_start: float,
    iteration_times: list[float],
    total_iters: int,
    current_iter: int,
) -> None:
    iter_time = time.time() - iter_start
    iteration_times.append(iter_time)
    avg_time = sum(iteration_times) / len(iteration_times)
    remaining = max(0, total_iters - current_iter - 1)
    eta = avg_time * remaining
    elapsed = time.time() - run_start
    logger.info(
        "  time=%.2fs | avg=%.2fs | elapsed=%.2fmin | eta=%.2fmin",
        iter_time,
        avg_time,
        elapsed / 60,
        eta / 60,
    )


def has_converged(changes: list[float], tolerance: float | None, window: int) -> bool:
    if tolerance is None or len(changes) < window:
        return False
    recent = changes[-window:]
    return all(value == value and value < tolerance for value in recent)

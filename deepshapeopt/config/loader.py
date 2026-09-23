"""Load a JSON experiment config: environment expansion, validation, archiving."""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any

from .schema import Config, ConfigError

_UNRESOLVED_ENV_RE = re.compile(r"\$\{[^}]+\}|\$[A-Za-z_][A-Za-z0-9_]*")


def expand_env_vars(value: Any) -> Any:
    """Recursively expand ``${VAR}`` / ``~`` in strings; unresolved variables raise.

    Keeps configs portable: ``"${DEEPSHAPEOPT_MODEL_DIR}/primitives_cl32"`` resolves at
    load time on every machine that sets the variable.
    """
    if isinstance(value, str):
        expanded = os.path.expanduser(os.path.expandvars(value))
        if _UNRESOLVED_ENV_RE.search(expanded):
            raise ConfigError(
                f"Unresolved environment variable in config value {value!r}. "
                "Source env.sh (see env.example.sh) before running."
            )
        return expanded
    if isinstance(value, list):
        return [expand_env_vars(v) for v in value]
    if isinstance(value, dict):
        return {k: expand_env_vars(v) for k, v in value.items()}
    return value


def load_config_dict(path: str | Path) -> dict:
    """The raw JSON with environment variables expanded (no validation)."""
    path = Path(path)
    if path.suffix != ".json":
        raise ConfigError(f"config must be a .json file, got {path}")
    with path.open() as f:
        raw = json.load(f)
    return expand_env_vars(copy.deepcopy(raw))


def load_config(path: str | Path) -> Config:
    """Load and validate a v2 config file."""
    return Config.from_dict(load_config_dict(path))


def archive_config(raw: dict, path: str | Path) -> None:
    """Write the resolved config next to the results (``config_log.json``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(raw, f, indent=2, default=str)
        f.write("\n")


def set_nested(raw: dict, dotted_key: str, value: Any) -> None:
    """Set ``a.b.c`` in a nested dict (used by parameter studies)."""
    keys = dotted_key.split(".")
    node = raw
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def get_nested(raw: dict, dotted_key: str) -> Any:
    node = raw
    for key in dotted_key.split("."):
        node = node[key]
    return node

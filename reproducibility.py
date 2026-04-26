"""Utilities for deterministic and auditable experiment runs.

This module centralises reproducibility controls used across scripts.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


CORE_PACKAGES: tuple[str, ...] = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "xgboost",
    "statsmodels",
)


def set_global_determinism(seed: int, *, force_single_thread: bool = True) -> None:
    """Set global deterministic controls for Python and NumPy.

    Notes
    -----
    ``PYTHONHASHSEED`` is read at interpreter startup. We still set it here
    for metadata/audit visibility and child-process consistency.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    if force_single_thread:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    random.seed(seed)
    np.random.seed(seed)


def sha256_file(path: Path) -> str:
    """Return SHA256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _get_installed_version(package_name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(package_name)
    except Exception:
        return None


def _get_git_commit(project_root: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return proc.stdout.strip() or None
    except Exception:
        return None


def _json_safe(value: Any) -> Any:
    """Convert values into JSON-serialisable primitives."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def collect_runtime_metadata(
    *,
    project_root: Path,
    seed: int,
    argv: Iterable[str],
    args_dict: dict[str, Any],
    force_single_thread: bool,
) -> dict[str, Any]:
    """Collect machine- and run-level metadata for reproducibility audit."""
    packages = {
        name: _get_installed_version(name)
        for name in CORE_PACKAGES
    }
    env_vars = {
        name: os.environ.get(name)
        for name in (
            "PYTHONHASHSEED",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    }
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "force_single_thread": bool(force_single_thread),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "argv": [_json_safe(v) for v in argv],
        "args": _json_safe(args_dict),
        "git_commit": _get_git_commit(project_root),
        "package_versions": packages,
        "env": env_vars,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write pretty-printed JSON with stable key ordering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)

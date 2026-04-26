"""Feature-engineering modules for WEC 2026.

Each event table in the contest has a sibling module here that exposes a
``build_*_features(events_df, main_df) -> pandas.DataFrame`` entry-point
producing the curated feature manifest established in the corresponding
EDA notebook.

Domain modules
--------------
- :mod:`features.shots`    - shots manifest (~16 cols)
- :mod:`features.runs`     - runs manifest (~12 cols)
- :mod:`features.pressure` - pressure manifest (~13 cols)
- :mod:`features.passes`   - passes manifest (~9 cols)

Cross modules
-------------
- :mod:`features.cross` - three cross-table entry points:
    * ``build_cross_features(main, runs, shots)`` (16 cols)
    * ``build_press_cross_features(main, runs, shots, press)`` (16 cols)
    * ``build_full_cross_features(main, runs, shots, press, passes)`` (20 cols)

Unified pipeline
----------------
- :func:`build_features` - assemble all manifests into a single panel,
  toggling each feature group on/off. Designed for ablation experiments
  (RQ3, RQ4).
"""
from __future__ import annotations

from features.cross import (
    CROSS_FEATURE_COLUMNS,
    PRESS_CROSS_FEATURE_COLUMNS,
    FULL_CROSS_FEATURE_COLUMNS,
    build_cross_features,
    build_press_cross_features,
    build_full_cross_features,
    save_cross_features,
)
from features.passes import build_pass_features, save_pass_features
from features.pipeline import VALID_GROUPS, build_features
from features.pressure import build_pressure_features, save_pressure_features
from features.runs import build_run_features, save_run_features
from features.shots import (
    ShotFeaturePipeline,
    build_shot_features,
    save_shot_features,
)


__all__ = [
    # shots
    "ShotFeaturePipeline",
    "build_shot_features",
    "save_shot_features",
    # runs
    "build_run_features",
    "save_run_features",
    # pressure
    "build_pressure_features",
    "save_pressure_features",
    # passes
    "build_pass_features",
    "save_pass_features",
    # cross
    "CROSS_FEATURE_COLUMNS",
    "PRESS_CROSS_FEATURE_COLUMNS",
    "FULL_CROSS_FEATURE_COLUMNS",
    "build_cross_features",
    "build_press_cross_features",
    "build_full_cross_features",
    "save_cross_features",
    # unified
    "build_features",
    "VALID_GROUPS",
]

"""features/pipeline.py - unified feature-assembly entry point.

A single ``build_features(main, ..., include={...})`` call composes any
combination of the four single-domain manifests and the three cross-table
manifests into one panel keyed on (player_appearance_id, checkpoint).

Designed for ablation experiments answering RQ3 ("are sprints + shots
sufficient?") and RQ4 ("does pass / pressure data add to the model?")
in a single line of code.

Usage
-----
    from features import build_features

    # Full feature space
    panel = build_features(main, runs=runs, shots=shots,
                           press=press, passes=passes,
                           include="all")

    # Ablation: sprints + shots only (RQ3 baseline)
    panel = build_features(main, runs=runs, shots=shots,
                           include={"runs", "shots"})

    # Ablation: + cross-table (RQ4)
    panel = build_features(main, runs=runs, shots=shots,
                           press=press, passes=passes,
                           include={"runs", "shots", "cross", "press_cross"})
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd

from features.cross import (
    build_cross_features,
    build_full_cross_features,
    build_press_cross_features,
)
from features.passes import build_pass_features
from features.pressure import build_pressure_features
from features.runs import build_run_features
from features.shots import build_shot_features


# ---------------------------------------------------------------------------
# Group definitions
# ---------------------------------------------------------------------------

VALID_GROUPS: frozenset[str] = frozenset({
    "shots", "runs", "pressure", "passes",
    "cross", "press_cross", "full_cross",
})

# Identifiers shared across every feature frame; these are the merge keys
# rather than features in their own right.
JOIN_COLS: tuple[str, ...] = ("player_appearance_id", "checkpoint")


def _resolve_groups(include: Iterable[str] | str) -> frozenset[str]:
    """Normalise the ``include`` argument into a set of group names."""
    if isinstance(include, str):
        if include == "all":
            return VALID_GROUPS
        include = (include,)
    groups = frozenset(include)
    unknown = groups - VALID_GROUPS
    if unknown:
        raise ValueError(
            f"unknown feature group(s): {sorted(unknown)}. "
            f"valid groups are {sorted(VALID_GROUPS)}"
        )
    return groups


def _drop_redundant_ids(features: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that duplicate identifiers already in the base frame."""
    redundant = ("fixture_id", "is_home", "position")
    keep = [c for c in features.columns if c not in redundant]
    return features[keep]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_features(
    main: pd.DataFrame,
    *,
    runs: pd.DataFrame | None = None,
    shots: pd.DataFrame | None = None,
    press: pd.DataFrame | None = None,
    passes: pd.DataFrame | None = None,
    include: Iterable[str] | str = "all",
    drop_zero_ratios: bool = False,
) -> pd.DataFrame:
    """Assemble the full feature panel by merging selected manifests.

    Parameters
    ----------
    main
        Main panel (`players_quarters_final.csv`).
    runs, shots, press, passes
        The four event tables. Each is required only if a feature group
        depending on it is included; otherwise pass ``None``.
    include
        Which feature groups to compose. Either ``"all"`` (default) or
        any iterable subset of
        ``{"shots", "runs", "pressure", "passes",
        "cross", "press_cross", "full_cross"}``.
    drop_zero_ratios
        Forwarded to each domain builder. If True, ratio features become
        0 where the denominator collapses; if False (default) they
        remain NaN for downstream tree models.

    Returns
    -------
    pandas.DataFrame
        Panel keyed on `(player_appearance_id, checkpoint)`. Includes the
        full main-table columns (target `scored_after`, position, time,
        etc.) plus all engineered features from the included groups.
    """
    groups = _resolve_groups(include)

    requires_runs = groups & {"runs", "cross", "press_cross", "full_cross"}
    requires_shots = groups & {"shots", "cross", "press_cross", "full_cross"}
    requires_press = groups & {"pressure", "press_cross", "full_cross"}
    requires_passes = groups & {"passes", "full_cross"}

    if requires_runs and runs is None:
        raise ValueError("`runs` is required for the requested feature groups")
    if requires_shots and shots is None:
        raise ValueError("`shots` is required for the requested feature groups")
    if requires_press and press is None:
        raise ValueError("`press` is required for the requested feature groups")
    if requires_passes and passes is None:
        raise ValueError("`passes` is required for the requested feature groups")

    panel = main.copy()

    if "shots" in groups:
        feats = build_shot_features(shots, main, drop_zero_shot_ratios=drop_zero_ratios)
        panel = panel.merge(_drop_redundant_ids(feats), on=list(JOIN_COLS), how="left")

    if "runs" in groups:
        feats = build_run_features(runs, main, drop_zero_run_ratios=drop_zero_ratios)
        panel = panel.merge(_drop_redundant_ids(feats), on=list(JOIN_COLS), how="left")

    if "pressure" in groups:
        feats = build_pressure_features(press, main, drop_zero_press_ratios=drop_zero_ratios)
        panel = panel.merge(_drop_redundant_ids(feats), on=list(JOIN_COLS), how="left")

    if "passes" in groups:
        feats = build_pass_features(passes, main, drop_zero_pass_ratios=drop_zero_ratios)
        panel = panel.merge(_drop_redundant_ids(feats), on=list(JOIN_COLS), how="left")

    if "cross" in groups:
        feats = build_cross_features(main, runs, shots)
        panel = panel.merge(feats, on=list(JOIN_COLS), how="left")

    if "press_cross" in groups:
        feats = build_press_cross_features(main, runs, shots, press)
        panel = panel.merge(feats, on=list(JOIN_COLS), how="left")

    if "full_cross" in groups:
        feats = build_full_cross_features(main, runs, shots, press, passes)
        panel = panel.merge(feats, on=list(JOIN_COLS), how="left")

    # Drop any duplicate columns produced by overlap between manifests.
    panel = panel.loc[:, ~panel.columns.duplicated()]

    return panel.reset_index(drop=True)


__all__ = ["build_features", "VALID_GROUPS"]

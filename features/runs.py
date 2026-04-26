"""features/runs.py - run-feature engineering pipeline.

Implements the curated manifest from `eda_player_appearance_run.ipynb`
Section F. Output is keyed on (player_appearance_id, checkpoint) and
intended to merge cleanly with the main panel.

Manifest (all 7 features survived BH q=0.05 + at least 1.4x baseline rate):

| # | Feature                  | Source            | r vs target |
|---|--------------------------|-------------------|------------:|
| 1 | last15_mean_max_speed    | main (carry)      | +0.062      |
| 2 | last15_peak_speed        | main (carry)      | +0.069      |
| 3 | last15_hsr               | main (carry)      | +0.053      |
| 4 | cumul_mean_max_speed     | main (carry)      | +0.038      |
| 5 | top_third_run_share      | engineered        | +0.104      |
| 6 | top_third_last15_share   | engineered        | (recency)   |
| 7 | runs_per_minute_played   | engineered        | +0.067      |

Usage
-----
    from features.runs import build_run_features

    features = build_run_features(runs_df, main_df)
"""
from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants - mirror the windowing rules from the run-EDA / cross module.
# ---------------------------------------------------------------------------

PERIOD_ORDER: Final[dict[str, int]] = {
    "half_1": 0, "half_2": 1, "extra_time_1": 2, "extra_time_2": 3,
}

CHECKPOINTS: Final[tuple[tuple[str, str, int], ...]] = (
    ("H1_15", "half_1", 15),
    ("H1_30", "half_1", 30),
    ("H1_45", "half_1", 45),
    ("H2_15", "half_2", 15),
    ("H2_30", "half_2", 30),
    ("H2_45", "half_2", 45),
    ("ET1_15", "extra_time_1", 15),
)

MATCH_MINUTE: Final[dict[str, int]] = {
    "H1_15": 15, "H1_30": 30, "H1_45": 45,
    "H2_15": 60, "H2_30": 75, "H2_45": 90, "ET1_15": 105,
}


REQUIRED_RUN_COLUMNS: Final[tuple[str, ...]] = (
    "player_appearance_id", "period", "stage", "minute",
    "min_speed", "max_speed", "distance", "run_type",
)

REQUIRED_MAIN_COLUMNS: Final[tuple[str, ...]] = (
    "player_appearance_id", "checkpoint", "fixture_id", "is_home", "position",
    "minute_in", "minute_out",
    "cumul_sprints", "cumul_hsr",
    "last15_hsr", "last15_mean_max_speed", "last15_peak_speed",
    "cumul_mean_max_speed",
)


FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    # identifiers
    "player_appearance_id", "checkpoint", "fixture_id", "is_home", "position",
    # main-table carry-throughs
    "last15_mean_max_speed", "last15_peak_speed",
    "last15_hsr", "cumul_mean_max_speed",
    # engineered
    "top_third_run_share", "top_third_last15_share",
    "runs_per_minute_played",
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_inputs(runs: pd.DataFrame, main: pd.DataFrame) -> None:
    missing_runs = set(REQUIRED_RUN_COLUMNS) - set(runs.columns)
    if missing_runs:
        raise ValueError(f"`runs` is missing required columns: {sorted(missing_runs)}")
    missing_main = set(REQUIRED_MAIN_COLUMNS) - set(main.columns)
    if missing_main:
        raise ValueError(f"`main` is missing required columns: {sorted(missing_main)}")


def _coerce_dtypes(runs: pd.DataFrame) -> pd.DataFrame:
    """Cast min_speed / max_speed / distance from string to float."""
    runs = runs.copy()
    for col in ("min_speed", "max_speed", "distance"):
        runs[col] = pd.to_numeric(runs[col], errors="coerce")
    return runs


def _filter_orphans(runs: pd.DataFrame, main: pd.DataFrame) -> pd.DataFrame:
    """Drop runs whose player_appearance_id is absent from main.

    Mirrors the run-EDA Section A7 finding (108 orphan appearances /
    ~1010 rows in the contest dataset).
    """
    return runs.loc[runs["player_appearance_id"].isin(main["player_appearance_id"])].copy()


def _aggregate_window(window_df: pd.DataFrame) -> pd.DataFrame:
    """Per-appearance aggregates inside a windowed slice of the run table."""
    g = window_df.groupby("player_appearance_id")
    return pd.DataFrame({
        "n_runs": g.size(),
        "n_top_runs": g.apply(lambda x: (x["stage"] == "top").sum()),
    })


def _build_panel(runs_clean: pd.DataFrame) -> pd.DataFrame:
    """Build a (player_appearance_id, checkpoint) panel.

    For each checkpoint, applies the strict windowing rules established
    in shots-EDA F2 / run-EDA Section B:

        cumul:  (period_order, minute) <= (cp_period_order, cp_minute)
        last15: same period AND cp_minute - 15 < minute <= cp_minute

    Returns columns:
        player_appearance_id, checkpoint,
        cumul_n_runs, cumul_n_top_runs,
        last15_n_runs, last15_n_top_runs
    """
    rows: list[pd.DataFrame] = []
    for cp, period, cp_min in CHECKPOINTS:
        cp_ord = PERIOD_ORDER[period]
        cumul_mask = (
            (runs_clean["period_order"] < cp_ord)
            | (
                (runs_clean["period_order"] == cp_ord)
                & (runs_clean["minute"] <= cp_min)
            )
        )
        last15_mask = (
            (runs_clean["period"] == period)
            & (runs_clean["minute"] > cp_min - 15)
            & (runs_clean["minute"] <= cp_min)
        )

        cumul = _aggregate_window(runs_clean.loc[cumul_mask]).add_prefix("cumul_")
        last15 = _aggregate_window(runs_clean.loc[last15_mask]).add_prefix("last15_")

        out = cumul.join(last15, how="outer").fillna(0).astype(int)
        out["checkpoint"] = cp
        rows.append(out.reset_index())

    return pd.concat(rows, ignore_index=True)


def _ratio(num: pd.Series, denom: pd.Series) -> pd.Series:
    """NaN-preserving ratio (NaN where denominator is zero)."""
    return num / denom.replace(0, np.nan)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_run_features(
    runs: pd.DataFrame,
    main: pd.DataFrame,
    *,
    drop_zero_run_ratios: bool = False,
) -> pd.DataFrame:
    """Engineer the curated run-feature manifest at the panel grain.

    Parameters
    ----------
    runs
        Raw run-event table (`player_appearance_run.csv`).
    main
        Main panel (`players_quarters_final.csv`).
    drop_zero_run_ratios
        If True, ratio features (`top_third_run_share`,
        `top_third_last15_share`) become 0 on rows where the player has
        recorded no runs in the window. If False (default) they remain
        NaN - leaving the imputation policy to downstream models
        (recommended for tree models).

    Returns
    -------
    pandas.DataFrame
        One row per main-panel `(player_appearance_id, checkpoint)`,
        columns listed in :data:`FEATURE_COLUMNS`.
    """
    _validate_inputs(runs, main)

    runs = _coerce_dtypes(runs)
    runs = _filter_orphans(runs, main)
    runs["period_order"] = runs["period"].map(PERIOD_ORDER)

    panel = _build_panel(runs)

    base_cols = list(REQUIRED_MAIN_COLUMNS)
    out = main[base_cols].merge(
        panel, on=["player_appearance_id", "checkpoint"], how="left",
    )
    count_cols = ["cumul_n_runs", "cumul_n_top_runs", "last15_n_runs", "last15_n_top_runs"]
    out[count_cols] = out[count_cols].fillna(0).astype(int)

    # Time / exposure context for the per-minute feature.
    out["match_minute_at_cp"] = out["checkpoint"].map(MATCH_MINUTE)
    out["mins_played_so_far"] = (
        out[["match_minute_at_cp", "minute_out"]].min(axis=1)
        - out["minute_in"].clip(lower=1) + 1
    ).clip(lower=1)

    # Engineered features.
    cumul_total = out["cumul_sprints"] + out["cumul_hsr"]
    out["top_third_run_share"] = _ratio(out["cumul_n_top_runs"], cumul_total)
    out["top_third_last15_share"] = _ratio(out["last15_n_top_runs"], out["last15_n_runs"])
    out["runs_per_minute_played"] = cumul_total / out["mins_played_so_far"]

    if drop_zero_run_ratios:
        ratio_cols = ["top_third_run_share", "top_third_last15_share"]
        out[ratio_cols] = out[ratio_cols].fillna(0.0)

    return out[list(FEATURE_COLUMNS)].reset_index(drop=True)


def save_run_features(features: pd.DataFrame, path: str | Path) -> Path:
    """Write engineered run features to ``path`` as CSV."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_path, index=False)
    return out_path


__all__ = [
    "FEATURE_COLUMNS",
    "build_run_features",
    "save_run_features",
]

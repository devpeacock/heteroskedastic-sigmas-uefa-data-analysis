"""Shot-derived feature engineering for the WEC 2026 modelling panel.

This module turns the raw shot-event table
(``player_appearance_shot_limited.csv``) and the main panel
(``players_quarters_final.csv``) into a tidy frame of shot-derived
features at the ``(player_appearance_id, checkpoint)`` grain — ready to
join into a modelling matrix.

Design follows the manifest established in Sections F4, F7, F9 and F10
of ``eda/eda_player_appearance_shot_limited.ipynb``:

* **Strict windowing rule** for both `last15_*` and `cumul_*`
  aggregates (F2 verdict — 99.94 % match against the main-table values).
* **Orphan filtering** — shot rows whose `player_appearance_id` is
  absent from the main panel are dropped before aggregation (F1).
* **Leakage-safe time feature** — `match_minute_at_cp` (current match
  minute) instead of `minutes_remaining` (which would leak the
  post-hoc fact that a fixture reached extra time).
* **Outcome-leaking columns** (`own_goal_player_appearance_id`,
  `block_player_appearance_id`) are never used.

Public API
----------
``build_shot_features(shots, main) -> pandas.DataFrame``
    Stateless function. The standard entry point for one-shot use.

``ShotFeaturePipeline``
    Class wrapper. Use when running the same engineering against
    multiple feature subsets, or when integrating into a larger
    pipeline that benefits from cached intermediates.

``save_shot_features(features, path)``
    Persist the engineered frame to CSV (no index column).

Examples
--------
>>> import pandas as pd
>>> from features.shots import build_shot_features, save_shot_features
>>> shots = pd.read_csv("data/player_appearance_shot_limited.csv")
>>> main = pd.read_csv("data/players_quarters_final.csv")
>>> features = build_shot_features(shots, main)
>>> save_shot_features(features, "features/shots_features.csv")
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants — derived from the EDA, see Sections A6, B3, F2, F7c.
# ---------------------------------------------------------------------------

#: Lexicographic ordering for ``period`` columns (event-side). Used to
#: implement the strict cumulative-window rule (F2).
PERIOD_ORDER: Final[dict[str, int]] = {
    "half_1": 1,
    "half_2": 2,
    "extra_time_1": 3,
    "extra_time_2": 4,
}

#: Lexicographic ordering for the main-panel ``checkpoint_period``.
CHECKPOINT_PERIOD_ORDER: Final[dict[str, int]] = {
    "half_1": 1,
    "half_2": 2,
    "extra_time_1": 3,
}

#: Local minute offset to convert a checkpoint into a continuous match
#: minute (`match_minute_at_cp`).
PERIOD_OFFSET: Final[dict[str, int]] = {
    "half_1": 0,
    "half_2": 45,
    "extra_time_1": 90,
}

#: Last regulation-time minute for each period — used to detect the
#: period-cap checkpoints (`H1_45`, `H2_45`, `ET1_15`).
PERIOD_CAP: Final[dict[str, int]] = {
    "half_1": 45,
    "half_2": 45,
    "extra_time_1": 15,
}

#: Set-piece play patterns (B3).
SET_PIECE_PATTERNS: Final[frozenset[str]] = frozenset(
    {"corner_kick", "indirect_free_kick", "direct_free_kick", "throw_in"}
)

#: Numerical guard against divide-by-zero in intensity ratios (F4).
RATIO_EPS: Final[float] = 1e-6

#: Columns the main-table input must provide.
REQUIRED_MAIN_COLUMNS: Final[tuple[str, ...]] = (
    "player_appearance_id",
    "checkpoint",
    "checkpoint_period",
    "checkpoint_min",
    "fixture_id",
    "is_home",
    "position",
    "minute_in",
    "minute_out",
    "cumul_shots",
    "cumul_shots_on_target",
    "cumul_shots_top_third",
    "cumul_shots_under_press",
    "last15_shots",
    "last15_shots_on_target",
    "last15_shots_top_third",
    "last15_shots_under_press",
)

#: Columns the shot-table input must provide.
REQUIRED_SHOT_COLUMNS: Final[tuple[str, ...]] = (
    "player_appearance_id",
    "period",
    "minute",
    "body_part",
    "play_pattern",
    "stage",
    "under_pressure",
)

#: Final column order of the engineered feature frame.
FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    # identifiers
    "player_appearance_id", "checkpoint", "fixture_id", "is_home", "position",
    # main-table existing (carried through)
    "cumul_shots", "cumul_shots_on_target",
    "last15_shots", "last15_shots_on_target",
    # time / context
    "match_minute_at_cp",
    # composition
    "shot_accuracy",
    "share_left_foot", "share_right_foot",
    "dominant_foot_strength", "set_piece_share", "is_penalty_taker",
    # intensity
    "last15_intensity", "cumul_intensity", "intensity_uplift",
    # team-level
    "player_shot_share",
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_inputs(shots: pd.DataFrame, main: pd.DataFrame) -> None:
    """Raise if either input is missing required columns."""
    missing_main = set(REQUIRED_MAIN_COLUMNS) - set(main.columns)
    if missing_main:
        raise ValueError(f"`main` is missing required columns: {sorted(missing_main)}")
    missing_shots = set(REQUIRED_SHOT_COLUMNS) - set(shots.columns)
    if missing_shots:
        raise ValueError(f"`shots` is missing required columns: {sorted(missing_shots)}")


def _drop_orphan_shots(shots: pd.DataFrame, main: pd.DataFrame) -> pd.DataFrame:
    """Filter shots to those whose appearance has at least one main-panel row.

    Implements the F1 finding: 15 appearances appear in the shot table but
    not in the main panel (late H2 substitutes). Their 19 shots cannot be
    joined to a checkpoint and would otherwise contaminate aggregates.
    """
    valid_appearances = set(main["player_appearance_id"].unique())
    return shots.loc[shots["player_appearance_id"].isin(valid_appearances)].copy()


def _build_shot_x_checkpoint_join(
    shots: pd.DataFrame, main: pd.DataFrame
) -> pd.DataFrame:
    """Cross-join every shot to every checkpoint of its appearance.

    Adds membership columns:
      ``in_cumul_window`` — strict ``(period, minute) ≤ (cp_period, cp_min)``,
      with the documented period-cap exception
      (`F2`'s ``cumul_inclusive`` rule was rejected; the strict rule wins
      with 99.94 % main-table match).
      ``in_last15_window`` — same period AND minute in
      ``[cp_min - 14, cp_min]``.

    Returns a long-format frame: one row per (shot, checkpoint) pair.
    """
    panel = main[["player_appearance_id", "checkpoint", "checkpoint_period",
                  "checkpoint_min"]].copy()
    panel["c_po"] = panel["checkpoint_period"].map(CHECKPOINT_PERIOD_ORDER)

    events = shots.copy()
    events["e_po"] = events["period"].map(PERIOD_ORDER)

    joined = events.merge(panel, on="player_appearance_id", how="inner")

    # Strict cumulative window — F2 verdict.
    joined["in_cumul_window"] = (
        (joined["e_po"] < joined["c_po"]) |
        ((joined["e_po"] == joined["c_po"]) &
         (joined["minute"] <= joined["checkpoint_min"]))
    )
    # Strict 15-minute window — F2 verdict (note the 18-row stoppage slack
    # noted in the EDA is accepted).
    same_period = joined["e_po"] == joined["c_po"]
    joined["in_last15_window"] = same_period & joined["minute"].between(
        joined["checkpoint_min"] - 14, joined["checkpoint_min"], inclusive="both"
    )
    return joined


def _aggregate_player_features(joined: pd.DataFrame) -> pd.DataFrame:
    """Per-(appearance, checkpoint) sums for body-part, play-pattern, etc.

    Aggregates the cross-join over the strict cumulative window only —
    the existing main-table ``last15_*`` and ``cumul_*shots`` are
    carried through unchanged in :func:`build_shot_features`.
    """
    work = joined.loc[joined["in_cumul_window"]].copy()
    work["is_left_foot"] = (work["body_part"] == "left_foot").astype(int)
    work["is_right_foot"] = (work["body_part"] == "right_foot").astype(int)
    work["is_head"] = (work["body_part"] == "head").astype(int)
    work["is_set_piece"] = work["play_pattern"].isin(SET_PIECE_PATTERNS).astype(int)
    work["is_penalty"] = (work["play_pattern"] == "penalty").astype(int)

    agg = (
        work.groupby(["player_appearance_id", "checkpoint"], observed=True)
            .agg(
                n_shots=("minute", "size"),
                n_left_foot=("is_left_foot", "sum"),
                n_right_foot=("is_right_foot", "sum"),
                n_head=("is_head", "sum"),
                n_set_piece=("is_set_piece", "sum"),
                n_penalty=("is_penalty", "sum"),
            )
    )
    return agg


def _aggregate_team_features(
    shots: pd.DataFrame, main: pd.DataFrame
) -> pd.DataFrame:
    """Per-(appearance, checkpoint) team-mate shot count via possession.

    Implements the team-in-match aggregation established in Section E1:
    `possession` identifies a team within a fixture (31 fixtures × 2 = 62
    distinct possession ids). For each main row we count every
    same-team shot taken before the checkpoint.
    """
    # Map appearance → (fixture, is_home) once.
    apr = main[["player_appearance_id", "fixture_id", "is_home"]].drop_duplicates(
        subset="player_appearance_id"
    )
    shots_with_team = shots.merge(apr, on="player_appearance_id", how="left")

    panel = main[["player_appearance_id", "fixture_id", "is_home", "checkpoint",
                  "checkpoint_period", "checkpoint_min"]].copy()
    panel["c_po"] = panel["checkpoint_period"].map(CHECKPOINT_PERIOD_ORDER)

    # Cross-join on fixture → every panel row × every same-fixture shot.
    cross = panel.merge(
        shots_with_team.rename(columns={
            "player_appearance_id": "shooter_appearance_id",
            "is_home": "shooter_is_home",
        }),
        on="fixture_id",
    )
    cross["e_po"] = cross["period"].map(PERIOD_ORDER)
    cross["in_cumul_window"] = (
        (cross["e_po"] < cross["c_po"]) |
        ((cross["e_po"] == cross["c_po"]) &
         (cross["minute"] <= cross["checkpoint_min"]))
    )
    cross["same_team"] = cross["is_home"] == cross["shooter_is_home"]

    counts = (
        cross.loc[cross["in_cumul_window"] & cross["same_team"]]
             .groupby(["player_appearance_id", "checkpoint"], observed=True)
             .size()
             .rename("team_n_shots")
             .to_frame()
    )
    return counts


def _ratio(num: pd.Series, denom: pd.Series) -> pd.Series:
    """NaN-preserving safe division (numerator / max(denominator, eps))."""
    out = num / denom.where(denom > 0)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_shot_features(
    shots: pd.DataFrame,
    main: pd.DataFrame,
    *,
    drop_zero_shot_ratios: bool = False,
) -> pd.DataFrame:
    """Engineer the curated shot-feature manifest at the panel grain.

    Parameters
    ----------
    shots
        Raw shot-event table (``player_appearance_shot_limited.csv``).
    main
        Main panel (``players_quarters_final.csv``).
    drop_zero_shot_ratios
        If ``True``, ratio features (``shot_accuracy`` etc.) become 0 on
        rows where the player has taken no shots so far. If ``False``
        (default) they remain NaN — leaving the imputation policy to the
        downstream model (recommended for tree models).

    Returns
    -------
    pandas.DataFrame
        One row per main-panel ``(player_appearance_id, checkpoint)`` pair,
        with columns listed in :data:`FEATURE_COLUMNS`. Index is the
        default RangeIndex.

    Notes
    -----
    Only the **strict** windowing rule is implemented for `cumul_*` and
    `last15_*` aggregates; the F2 EDA established this rule reproduces
    the main-table values 99.94 % of the time.
    """
    _validate_inputs(shots, main)

    shots_aligned = _drop_orphan_shots(shots, main)
    joined = _build_shot_x_checkpoint_join(shots_aligned, main)

    player_agg = _aggregate_player_features(joined)
    team_agg = _aggregate_team_features(shots_aligned, main)

    base_columns = list(REQUIRED_MAIN_COLUMNS)
    out = (
        main[base_columns]
        .merge(player_agg, on=["player_appearance_id", "checkpoint"], how="left")
        .merge(team_agg, on=["player_appearance_id", "checkpoint"], how="left")
    )
    # Fill aggregated counts with 0 (no shots) and team_n_shots with 0.
    count_cols = [
        "n_shots", "n_left_foot", "n_right_foot", "n_head",
        "n_set_piece", "n_penalty", "team_n_shots",
    ]
    out[count_cols] = out[count_cols].fillna(0).astype(int)

    # Time features.
    out["match_minute_at_cp"] = (
        out["checkpoint_period"].map(PERIOD_OFFSET) + out["checkpoint_min"]
    )

    # Minutes the player spent on the pitch up to (and including) the cp.
    out["minutes_played"] = (
        out[["match_minute_at_cp", "minute_out"]].min(axis=1)
        - out["minute_in"] + 1
    ).clip(lower=1)

    # Composition ratios — leakage-safe because they use main-table aggregates.
    out["shot_accuracy"] = _ratio(out["cumul_shots_on_target"], out["cumul_shots"])
    out["share_left_foot"] = _ratio(out["n_left_foot"], out["n_shots"])
    out["share_right_foot"] = _ratio(out["n_right_foot"], out["n_shots"])
    out["dominant_foot_strength"] = out[["share_left_foot", "share_right_foot"]].max(axis=1)
    out["set_piece_share"] = _ratio(out["n_set_piece"], out["n_shots"])
    out["is_penalty_taker"] = (out["n_penalty"] > 0).astype(int)

    # Intensity features.
    out["last15_intensity"] = out["last15_shots"] / 15.0
    out["cumul_intensity"] = out["cumul_shots"] / out["minutes_played"]
    out["intensity_uplift"] = (
        out["last15_intensity"] / (out["cumul_intensity"] + RATIO_EPS)
    )

    # Team-level — `possession` ↔ team-in-match (E1).
    out["player_shot_share"] = _ratio(out["cumul_shots"], out["team_n_shots"])

    if drop_zero_shot_ratios:
        ratio_cols = [
            "shot_accuracy", "share_left_foot", "share_right_foot",
            "dominant_foot_strength", "set_piece_share", "player_shot_share",
        ]
        out[ratio_cols] = out[ratio_cols].fillna(0.0)

    # Reorder to the canonical column list and drop intermediates.
    return out[list(FEATURE_COLUMNS)].reset_index(drop=True)


def save_shot_features(features: pd.DataFrame, path: str | Path) -> Path:
    """Write the engineered features to a CSV at ``path``.

    Parameters
    ----------
    features
        DataFrame returned by :func:`build_shot_features`.
    path
        Destination CSV path. Parent directories are created if missing.

    Returns
    -------
    pathlib.Path
        The resolved destination path.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------------
# Class wrapper for stateful / pipeline-style use
# ---------------------------------------------------------------------------


@dataclass
class ShotFeaturePipeline:
    """Stateful wrapper around :func:`build_shot_features`.

    Useful when running the same engineering against multiple feature
    subsets, or when integrating into a larger sklearn-style pipeline.

    Attributes
    ----------
    drop_zero_shot_ratios
        Forwarded to :func:`build_shot_features`.

    Examples
    --------
    >>> pipeline = ShotFeaturePipeline()
    >>> features = pipeline.transform(shots, main)
    >>> pipeline.save(features, "features/shots_features.csv")
    """

    drop_zero_shot_ratios: bool = False

    def transform(self, shots: pd.DataFrame, main: pd.DataFrame) -> pd.DataFrame:
        """Return the engineered feature frame."""
        return build_shot_features(
            shots, main, drop_zero_shot_ratios=self.drop_zero_shot_ratios
        )

    @staticmethod
    def save(features: pd.DataFrame, path: str | Path) -> Path:
        """Persist a feature frame to disk (delegates to :func:`save_shot_features`)."""
        return save_shot_features(features, path)

    @staticmethod
    def feature_names() -> list[str]:
        """Names of the feature columns the pipeline produces, in order."""
        return list(FEATURE_COLUMNS)

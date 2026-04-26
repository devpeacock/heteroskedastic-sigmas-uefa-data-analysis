"""features/cross.py — cross-table engineered features (runs × shots × main).

Implements the prior-ranked top-8 cross-table candidates from the EDA series:

1. shots_per_top_third_run        clinical-finisher proxy
2. position × top_third_run_share spatial role amplification (4 columns)
3. position × last15_sprints      attacker-fatigue inversion recovery (4 cols)
4. fatigue_gap                    cumul vs last15 sprint intensity
5. top_third_run_share × set_piece_share         target-striker profile
6. top_third_run_share × shot_accuracy           clinical attacking presence
7. shot_intent_under_fatigue      last15_shots × fatigue_gap
8. speed_above_team_avg           cumul_mean_max_speed - team mean (same fixture × side × cp)

Output is keyed on ``(player_appearance_id, checkpoint)`` and intended to be
merged with the main panel + shot/run feature frames in modelling.

Usage
-----
    from features.cross import build_cross_features

    cross = build_cross_features(main_df, runs_df, shots_df)
"""
from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from features.shots import build_shot_features


# ---------------------------------------------------------------------------
# Constants — match the windowing rules used in features/shots.py and the
# eda_player_appearance_run notebook.
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

# Continuous match-minute representation of each checkpoint, mirroring
# features/shots.py PERIOD_OFFSET (0 / 45 / 90 / 105).
MATCH_MINUTE: Final[dict[str, int]] = {
    "H1_15": 15, "H1_30": 30, "H1_45": 45,
    "H2_15": 60, "H2_30": 75, "H2_45": 90, "ET1_15": 105,
}

EPS: Final[float] = 1e-6

POSITIONS: Final[tuple[str, ...]] = ("A", "M", "D", "G")

CROSS_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    # identifiers
    "player_appearance_id", "checkpoint",
    # 1
    "shots_per_top_third_run",
    # 2 — position × top_third_run_share (4 cols, only one is non-zero per row)
    "top_third_run_share_A", "top_third_run_share_M",
    "top_third_run_share_D", "top_third_run_share_G",
    # 3 — position × last15_sprints
    "last15_sprints_A", "last15_sprints_M",
    "last15_sprints_D", "last15_sprints_G",
    # 4
    "fatigue_gap",
    # 5
    "top_third_run_share_x_set_piece_share",
    # 6
    "top_third_run_share_x_shot_accuracy",
    # 7
    "shot_intent_under_fatigue",
    # 8
    "speed_above_team_avg",
)


FULL_CROSS_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    # identifiers
    "player_appearance_id", "checkpoint",
    # Theme 1 — 4-domain spatial concentration
    "four_domain_top_third_avg",
    "top_third_consistency_count",
    "top_third_pass_x_top_third_run",
    # Theme 2 — pass × position (4 dummies each)
    "cumul_passes_A", "cumul_passes_M", "cumul_passes_D", "cumul_passes_G",
    "passes_received_share_A", "passes_received_share_M",
    "passes_received_share_D", "passes_received_share_G",
    # Theme 3 — reception → goal pipeline
    "shots_per_pass_received",
    # Theme 4 — composure (pressure vs general)
    "composure_under_pressure_ratio",
    # Theme 6 — off-the-ball reception
    "last15_passes_received_x_last15_sprints",
    # Theme 7 — team-relative reception
    "passes_received_share_above_team_avg",
    # Theme 9 — substitute effects
    "subbed_x_last15_intensity_shots",
    # Theme 11 — role-fingerprint ratios
    "shot_to_pass_ratio",
    "shot_to_press_ratio",
)


PRESS_CROSS_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    # identifiers
    "player_appearance_id", "checkpoint",
    # 1 — multiplicative gate of the two strongest spatial signals
    "top_third_presence_joint",
    # 2 — extends the strongest shots-EDA interaction with pressure spatial axis
    "top_third_press_share_x_set_piece_share",
    # 3 — tactical-role disambiguator
    "pressing_minus_pressed_x_cumul_shots",
    # 4 — press_turnover_rate × position
    "press_turnover_rate_A", "press_turnover_rate_M",
    "press_turnover_rate_D", "press_turnover_rate_G",
    # 5 — forward_pass_share × position
    "forward_pass_share_A", "forward_pass_share_M",
    "forward_pass_share_D", "forward_pass_share_G",
    # 6 — team-relative spatial pressure
    "top_third_press_share_above_team_avg",
    # 7 — strongest cross-domain recency surge
    "last15_press_events_x_last15_intensity_shots",
    # 8 — additive complement to the joint gate
    "top_third_presence_avg",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce_run_dtypes(runs: pd.DataFrame) -> pd.DataFrame:
    """Cast the three numeric run columns from string to float.

    The source CSV writes ``min_speed`` / ``max_speed`` / ``distance`` as
    quoted strings; pandas reads them as ``object``. Coerce explicitly so all
    later numeric ops behave.
    """
    runs = runs.copy()
    for col in ("min_speed", "max_speed", "distance"):
        runs[col] = pd.to_numeric(runs[col], errors="coerce")
    return runs


def _filter_to_main(runs: pd.DataFrame, main: pd.DataFrame) -> pd.DataFrame:
    """Drop run rows whose `player_appearance_id` is absent from ``main``.

    Mirrors the orphan-filter from the run EDA Section A7 (108 appearances /
    1010 rows in the contest dataset).
    """
    return runs.loc[runs["player_appearance_id"].isin(main["player_appearance_id"])].copy()


def _aggregate_top_runs(window_df: pd.DataFrame) -> pd.DataFrame:
    """Per-appearance count of runs in the attacking-third (`stage == "top"`)."""
    g = window_df.groupby("player_appearance_id")
    return pd.DataFrame({
        "n_top_runs": g.apply(lambda x: (x["stage"] == "top").sum()),
    })


def _runs_top_third_panel(runs_clean: pd.DataFrame) -> pd.DataFrame:
    """Build a (player_appearance_id, checkpoint) panel of cumul_n_top_runs.

    Uses the strict windowing rule established in the shots-EDA F2:

        cumul:  (period_order, minute) <= (cp_period_order, cp_minute)
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
        agg = _aggregate_top_runs(runs_clean.loc[cumul_mask]).add_prefix("cumul_")
        agg["checkpoint"] = cp
        rows.append(agg.reset_index())
    return pd.concat(rows, ignore_index=True)


def _select_shot_columns(shots: pd.DataFrame, main: pd.DataFrame) -> pd.DataFrame:
    """Pull `set_piece_share` and `shot_accuracy` from features/shots.py.

    Reusing the productionised module guarantees the same windowing /
    leakage discipline as the shots feature pipeline.
    """
    shot_features = build_shot_features(shots, main)
    keep = ["player_appearance_id", "checkpoint", "set_piece_share", "shot_accuracy"]
    return shot_features[keep].copy()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_cross_features(
    main: pd.DataFrame,
    runs: pd.DataFrame,
    shots: pd.DataFrame,
) -> pd.DataFrame:
    """Compose the top-8 cross-table feature recommendations from the EDA.

    Parameters
    ----------
    main, runs, shots
        The three contest CSVs.

    Returns
    -------
    pandas.DataFrame
        One row per ``(player_appearance_id, checkpoint)`` pair from the main
        panel, with columns listed in :data:`CROSS_FEATURE_COLUMNS`.

    Notes
    -----
    Ratios that hit a zero denominator (e.g. ``cumul_n_top_runs == 0`` for
    rows where the player has not yet recorded any attacking-third run) are
    left as ``NaN`` to preserve the imputation choice for downstream models.
    Tree models can split on the missingness; for linear models, the
    recommended policy is ``fillna(0)`` paired with a
    ``has_top_third_run_history`` indicator.
    """
    # --- 1. Cleanup runs (cast + orphan filter + period_order). --------------
    runs = _coerce_run_dtypes(runs)
    runs = _filter_to_main(runs, main)
    runs["period_order"] = runs["period"].map(PERIOD_ORDER)

    # --- 2. Run-derived aggregates not present in main: cumul_n_top_runs.
    top_runs_panel = _runs_top_third_panel(runs)

    # --- 3. Shot composition features we depend on (set_piece_share, shot_accuracy).
    shot_feats = _select_shot_columns(shots, main)

    # --- 4. Base frame: identifiers + main columns we need + merges. ---------
    needed_main_cols = [
        "player_appearance_id", "checkpoint", "fixture_id", "is_home", "position",
        "minute_in", "minute_out",
        "cumul_shots", "last15_shots",
        "cumul_sprints", "cumul_hsr", "last15_sprints",
        "cumul_mean_max_speed",
    ]
    base = main[needed_main_cols].copy()

    base = base.merge(top_runs_panel, on=["player_appearance_id", "checkpoint"], how="left")
    base["cumul_n_top_runs"] = base["cumul_n_top_runs"].fillna(0).astype(int)

    base = base.merge(shot_feats, on=["player_appearance_id", "checkpoint"], how="left")

    # --- 5. Time / exposure features. ----------------------------------------
    base["match_minute_at_cp"] = base["checkpoint"].map(MATCH_MINUTE)
    base["mins_played_so_far"] = (
        base[["match_minute_at_cp", "minute_out"]].min(axis=1)
        - base["minute_in"].clip(lower=1) + 1
    ).clip(lower=1)

    # --- 6. Building blocks for the 8 features. ------------------------------
    # Total cumulative runs for the share denominator.
    cumul_total_runs = base["cumul_sprints"] + base["cumul_hsr"]

    # top_third_run_share — used in features 2, 5, 6.
    base["_top_third_run_share"] = (
        base["cumul_n_top_runs"] / cumul_total_runs.replace(0, np.nan)
    )

    # cumul / last15 sprint intensity — used in feature 4 (and indirectly 7).
    cumul_intensity_runs = base["cumul_sprints"] / base["mins_played_so_far"]
    last15_intensity_runs = base["last15_sprints"] / 15.0

    # --- 7. Compose the 8 features. ------------------------------------------
    out = base[["player_appearance_id", "checkpoint"]].copy()

    # 1. shots_per_top_third_run
    out["shots_per_top_third_run"] = (
        base["cumul_shots"] / base["cumul_n_top_runs"].replace(0, np.nan)
    )

    # 2. position × top_third_run_share — one column per position, zero on mismatch.
    for pos in POSITIONS:
        out[f"top_third_run_share_{pos}"] = np.where(
            base["position"] == pos, base["_top_third_run_share"], 0.0
        )

    # 3. position × last15_sprints
    for pos in POSITIONS:
        out[f"last15_sprints_{pos}"] = np.where(
            base["position"] == pos, base["last15_sprints"].astype(float), 0.0
        )

    # 4. fatigue_gap — positive = slowing down (cumul rate exceeds recent).
    out["fatigue_gap"] = cumul_intensity_runs - last15_intensity_runs

    # 5. top_third_run_share × set_piece_share
    out["top_third_run_share_x_set_piece_share"] = (
        base["_top_third_run_share"] * base["set_piece_share"]
    )

    # 6. top_third_run_share × shot_accuracy
    out["top_third_run_share_x_shot_accuracy"] = (
        base["_top_third_run_share"] * base["shot_accuracy"]
    )

    # 7. shot_intent_under_fatigue — captures "tired but still shooting".
    out["shot_intent_under_fatigue"] = base["last15_shots"] * out["fatigue_gap"]

    # 8. speed_above_team_avg — per-checkpoint, per-team-side mean as baseline.
    team_speed = base.groupby(
        ["fixture_id", "is_home", "checkpoint"]
    )["cumul_mean_max_speed"].transform("mean")
    out["speed_above_team_avg"] = base["cumul_mean_max_speed"] - team_speed

    # Normalise NaN inf etc.
    out = out.replace([np.inf, -np.inf], np.nan)

    return out[list(CROSS_FEATURE_COLUMNS)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Pressure-cross helpers + entry point
# ---------------------------------------------------------------------------


def _filter_press_to_main(press: pd.DataFrame, main: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split press into pressed-side and pressing-side views, both filtered.

    Returns
    -------
    pressed, pressing
        Two frames; ``pressed`` is filtered on ``player_appearance_id``,
        ``pressing`` is filtered on ``pressing_player_appearance_id``. Both
        carry a ``period_order`` column for fast windowing.
    """
    main_apps = set(main["player_appearance_id"].unique())

    pressed = press.loc[press["player_appearance_id"].isin(main_apps)].copy()
    pressed["period_order"] = pressed["period"].map(PERIOD_ORDER)

    pressing = press.loc[press["pressing_player_appearance_id"].isin(main_apps)].copy()
    pressing["period_order"] = pressing["period"].map(PERIOD_ORDER)

    return pressed, pressing


def _press_aggregates_panel(
    pressed: pd.DataFrame, pressing: pd.DataFrame
) -> pd.DataFrame:
    """Per (player_appearance_id, checkpoint) pressure aggregates.

    Uses the strict windowing rule:

        cumul:  (period_order, minute) <= (cp_period_order, cp_minute)
        last15: same period AND cp_minute - 15 < minute <= cp_minute

    Returns columns:
        player_appearance_id, checkpoint,
        cumul_press_events, last15_press_events,
        cumul_press_top_third, cumul_press_turnovers,
        cumul_forward_passes, cumul_dir_passes,
        cumul_pressing_others
    """
    rows: list[pd.DataFrame] = []
    for cp, period, cp_min in CHECKPOINTS:
        cp_ord = PERIOD_ORDER[period]

        # --- Pressed-side aggregates (this player was the one being pressed).
        cumul_p1 = (
            (pressed["period_order"] < cp_ord)
            | (
                (pressed["period_order"] == cp_ord)
                & (pressed["minute"] <= cp_min)
            )
        )
        last15_p1 = (
            (pressed["period"] == period)
            & (pressed["minute"] > cp_min - 15)
            & (pressed["minute"] <= cp_min)
        )

        c_df = pressed.loc[cumul_p1]
        l_df = pressed.loc[last15_p1]

        gc = c_df.groupby("player_appearance_id")
        cumul_pressed = pd.DataFrame({
            "cumul_press_events": gc.size(),
            "cumul_press_top_third": gc.apply(
                lambda x: (x["stage"] == "top").sum()
            ),
            "cumul_press_turnovers": gc.apply(
                lambda x: (x["press_induced_outcome"] == "turnover").sum()
            ),
            "cumul_forward_passes": gc.apply(
                lambda x: (x["press_induced_outcome"] == "forward_pass").sum()
            ),
            "cumul_dir_passes": gc.apply(
                lambda x: x["press_induced_outcome"].isin(
                    ("forward_pass", "backward_pass", "lateral_pass")
                ).sum()
            ),
            # Used by build_full_cross_features → composure_under_pressure_ratio.
            # Restrict to directional-pass rows so the numerator is a proper
            # subset of `cumul_dir_passes` (the denominator). Without this
            # filter, `accurate=True` on ball_carry rows inflates the count
            # and the ratio can artefactually exceed 1.
            "cumul_press_accurate": gc.apply(
                lambda x: (
                    (x["accurate"] == True)
                    & x["press_induced_outcome"].isin(
                        ("forward_pass", "backward_pass", "lateral_pass")
                    )
                ).sum()
            ),
        })

        gl = l_df.groupby("player_appearance_id")
        last15_pressed = pd.DataFrame({"last15_press_events": gl.size()})

        # --- Pressing-side aggregate (this player applied pressure).
        cumul_p2 = (
            (pressing["period_order"] < cp_ord)
            | (
                (pressing["period_order"] == cp_ord)
                & (pressing["minute"] <= cp_min)
            )
        )
        gp = pressing.loc[cumul_p2].groupby("pressing_player_appearance_id")
        pressing_agg = pd.DataFrame({"cumul_pressing_others": gp.size()})
        pressing_agg.index.name = "player_appearance_id"

        out = (
            cumul_pressed
            .join(last15_pressed, how="outer")
            .join(pressing_agg, how="outer")
            .fillna(0)
            .astype(int)
        )
        out["checkpoint"] = cp
        rows.append(out.reset_index())

    return pd.concat(rows, ignore_index=True)


def _runs_top_third_share(
    main: pd.DataFrame, runs: pd.DataFrame
) -> pd.DataFrame:
    """Per (player_appearance_id, checkpoint) `top_third_run_share`.

    Computed identically to :func:`build_cross_features` (denominator is
    ``cumul_sprints + cumul_hsr`` from the main panel). Returned columns:
    ``player_appearance_id``, ``checkpoint``, ``top_third_run_share``.
    """
    runs = _coerce_run_dtypes(runs)
    runs = _filter_to_main(runs, main)
    runs["period_order"] = runs["period"].map(PERIOD_ORDER)

    panel = _runs_top_third_panel(runs)
    merged = main[[
        "player_appearance_id", "checkpoint", "cumul_sprints", "cumul_hsr",
    ]].merge(panel, on=["player_appearance_id", "checkpoint"], how="left")
    merged["cumul_n_top_runs"] = merged["cumul_n_top_runs"].fillna(0).astype(int)

    total = (merged["cumul_sprints"] + merged["cumul_hsr"]).replace(0, np.nan)
    merged["top_third_run_share"] = merged["cumul_n_top_runs"] / total
    return merged[["player_appearance_id", "checkpoint", "top_third_run_share"]]


def build_press_cross_features(
    main: pd.DataFrame,
    runs: pd.DataFrame,
    shots: pd.DataFrame,
    press: pd.DataFrame,
) -> pd.DataFrame:
    """Compose the prior-ranked top-8 pressure-cross features.

    Parameters
    ----------
    main, runs, shots, press
        The four contest CSVs.

    Returns
    -------
    pandas.DataFrame
        One row per ``(player_appearance_id, checkpoint)`` pair from the
        main panel, with columns listed in
        :data:`PRESS_CROSS_FEATURE_COLUMNS`.

    Notes
    -----
    Ratio features are NaN where their denominators collapse to zero
    (e.g. a checkpoint with zero cumulative pressure events for a player).
    The same NaN policy as :func:`build_cross_features` applies downstream:
    tree models can split on missingness; for linear models, pair with a
    ``has_press_history`` indicator.
    """
    # --- 1. Run-side prerequisite: top_third_run_share. ----------------------
    run_share = _runs_top_third_share(main, runs)

    # --- 2. Press-side aggregates (pressed + pressing views). ----------------
    pressed, pressing = _filter_press_to_main(press, main)
    press_panel = _press_aggregates_panel(pressed, pressing)

    # --- 3. Shot-side prerequisite: set_piece_share. -------------------------
    shot_features = build_shot_features(shots, main)
    shot_keep = shot_features[
        ["player_appearance_id", "checkpoint", "set_piece_share"]
    ].copy()

    # --- 4. Assemble base from main + merges. --------------------------------
    base = main[[
        "player_appearance_id", "checkpoint", "fixture_id", "is_home", "position",
        "cumul_shots", "last15_shots",
    ]].copy()

    base = base.merge(run_share, on=["player_appearance_id", "checkpoint"], how="left")
    base = base.merge(press_panel, on=["player_appearance_id", "checkpoint"], how="left")
    base = base.merge(shot_keep, on=["player_appearance_id", "checkpoint"], how="left")

    # Press counts: 0 = "no event yet", which is true info, not missingness.
    press_count_cols = (
        "cumul_press_events", "last15_press_events",
        "cumul_press_top_third", "cumul_press_turnovers",
        "cumul_forward_passes", "cumul_dir_passes",
        "cumul_press_accurate", "cumul_pressing_others",
    )
    base[list(press_count_cols)] = base[list(press_count_cols)].fillna(0).astype(int)

    # --- 5. Derived pressure feature blocks. ---------------------------------
    base["_top_third_press_share"] = (
        base["cumul_press_top_third"]
        / base["cumul_press_events"].replace(0, np.nan)
    )
    base["_press_turnover_rate"] = (
        base["cumul_press_turnovers"]
        / base["cumul_press_events"].replace(0, np.nan)
    )
    base["_forward_pass_share"] = (
        base["cumul_forward_passes"]
        / base["cumul_dir_passes"].replace(0, np.nan)
    )
    base["_pressing_minus_pressed"] = (
        base["cumul_pressing_others"] - base["cumul_press_events"]
    )
    base["_last15_intensity_shots"] = base["last15_shots"] / 15.0

    # --- 6. Compose the 8 features. ------------------------------------------
    out = base[["player_appearance_id", "checkpoint"]].copy()

    # 1. Joint top-third presence (multiplicative gate).
    out["top_third_presence_joint"] = (
        base["top_third_run_share"] * base["_top_third_press_share"]
    )

    # 2. top_third_press_share × set_piece_share.
    out["top_third_press_share_x_set_piece_share"] = (
        base["_top_third_press_share"] * base["set_piece_share"]
    )

    # 3. pressing_minus_pressed × cumul_shots.
    out["pressing_minus_pressed_x_cumul_shots"] = (
        base["_pressing_minus_pressed"] * base["cumul_shots"]
    )

    # 4. press_turnover_rate × position (4 dummies).
    for pos in POSITIONS:
        out[f"press_turnover_rate_{pos}"] = np.where(
            base["position"] == pos,
            base["_press_turnover_rate"],
            0.0,
        )

    # 5. forward_pass_share × position (4 dummies).
    for pos in POSITIONS:
        out[f"forward_pass_share_{pos}"] = np.where(
            base["position"] == pos,
            base["_forward_pass_share"],
            0.0,
        )

    # 6. top_third_press_share - team mean (per fixture × side × checkpoint).
    team_press_share = base.groupby(
        ["fixture_id", "is_home", "checkpoint"]
    )["_top_third_press_share"].transform("mean")
    out["top_third_press_share_above_team_avg"] = (
        base["_top_third_press_share"] - team_press_share
    )

    # 7. last15_press_events × last15_intensity_shots.
    out["last15_press_events_x_last15_intensity_shots"] = (
        base["last15_press_events"] * base["_last15_intensity_shots"]
    )

    # 8. Additive complement to the joint gate. NaN preserved only when
    # both inputs are NaN; otherwise treats a missing share as "no presence
    # yet" = 0, which is the meaningful default for the additive form.
    run_share_filled = base["top_third_run_share"].fillna(0.0)
    press_share_filled = base["_top_third_press_share"].fillna(0.0)
    out["top_third_presence_avg"] = (run_share_filled + press_share_filled) / 2.0
    both_nan = (
        base["top_third_run_share"].isna() & base["_top_third_press_share"].isna()
    )
    out.loc[both_nan, "top_third_presence_avg"] = np.nan

    out = out.replace([np.inf, -np.inf], np.nan)
    return out[list(PRESS_CROSS_FEATURE_COLUMNS)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Pass-side helpers + full cross entry point
# ---------------------------------------------------------------------------


def _filter_pass_to_main(
    passes: pd.DataFrame, main: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split passes into sender-side and receiver-side views, both filtered.

    Returns
    -------
    sender, receiver
        Two frames; ``sender`` is filtered on ``player_appearance_id``,
        ``receiver`` is filtered on ``addressee_player_appearance_id``
        (NULL adresses dropped). Both carry a ``period_order`` column.
    """
    main_apps = set(main["player_appearance_id"].unique())

    sender = passes.loc[passes["player_appearance_id"].isin(main_apps)].copy()
    sender["period_order"] = sender["period"].map(PERIOD_ORDER)

    receiver_mask = passes["addressee_player_appearance_id"].notna()
    receiver = passes.loc[receiver_mask].copy()
    receiver["addressee_player_appearance_id"] = (
        receiver["addressee_player_appearance_id"].astype(int)
    )
    receiver = receiver.loc[
        receiver["addressee_player_appearance_id"].isin(main_apps)
    ].copy()
    receiver["period_order"] = receiver["period"].map(PERIOD_ORDER)

    return sender, receiver


def _pass_aggregates_panel(
    sender: pd.DataFrame, receiver: pd.DataFrame
) -> pd.DataFrame:
    """Per (player_appearance_id, checkpoint) pass aggregates.

    Returns columns:
        player_appearance_id, checkpoint,
        cumul_passes, cumul_passes_accurate, cumul_passes_top_third,
        last15_passes, cumul_passes_received, last15_passes_received
    """
    rows: list[pd.DataFrame] = []
    for cp, period, cp_min in CHECKPOINTS:
        cp_ord = PERIOD_ORDER[period]

        # --- sender side --------------------------------------------------
        cumul_s = (
            (sender["period_order"] < cp_ord)
            | (
                (sender["period_order"] == cp_ord)
                & (sender["minute"] <= cp_min)
            )
        )
        last15_s = (
            (sender["period"] == period)
            & (sender["minute"] > cp_min - 15)
            & (sender["minute"] <= cp_min)
        )

        gc = sender.loc[cumul_s].groupby("player_appearance_id")
        cumul_send = pd.DataFrame({
            "cumul_passes": gc.size(),
            "cumul_passes_accurate": gc.apply(
                lambda x: (x["accurate"] == True).sum()
            ),
            "cumul_passes_top_third": gc.apply(
                lambda x: (x["stage"] == "top").sum()
            ),
        })

        gl = sender.loc[last15_s].groupby("player_appearance_id")
        last15_send = pd.DataFrame({"last15_passes": gl.size()})

        # --- receiver side ------------------------------------------------
        cumul_r = (
            (receiver["period_order"] < cp_ord)
            | (
                (receiver["period_order"] == cp_ord)
                & (receiver["minute"] <= cp_min)
            )
        )
        last15_r = (
            (receiver["period"] == period)
            & (receiver["minute"] > cp_min - 15)
            & (receiver["minute"] <= cp_min)
        )

        gp_c = receiver.loc[cumul_r].groupby("addressee_player_appearance_id")
        cumul_recv = pd.DataFrame({"cumul_passes_received": gp_c.size()})
        cumul_recv.index.name = "player_appearance_id"

        gp_l = receiver.loc[last15_r].groupby("addressee_player_appearance_id")
        last15_recv = pd.DataFrame({"last15_passes_received": gp_l.size()})
        last15_recv.index.name = "player_appearance_id"

        out = (
            cumul_send
            .join(last15_send, how="outer")
            .join(cumul_recv, how="outer")
            .join(last15_recv, how="outer")
            .fillna(0)
            .astype(int)
        )
        out["checkpoint"] = cp
        rows.append(out.reset_index())

    return pd.concat(rows, ignore_index=True)


def _to_int_bool(series: pd.Series) -> pd.Series:
    """Coerce a TRUE/FALSE column (string or bool) to int 0/1."""
    if series.dtype == bool:
        return series.astype(int)
    return series.astype(str).str.upper().eq("TRUE").astype(int)


def build_full_cross_features(
    main: pd.DataFrame,
    runs: pd.DataFrame,
    shots: pd.DataFrame,
    press: pd.DataFrame,
    passes: pd.DataFrame,
) -> pd.DataFrame:
    """Compose the prior-ranked tier-A 12 cross-table features (4-table joint).

    Parameters
    ----------
    main, runs, shots, press, passes
        All five contest CSVs.

    Returns
    -------
    pandas.DataFrame
        One row per ``(player_appearance_id, checkpoint)`` from the main
        panel, with columns listed in :data:`FULL_CROSS_FEATURE_COLUMNS`.

    Notes
    -----
    Re-uses the windowing / orphan / NaN policies established across the
    earlier helpers. The 12 logical features expand to 18 columns because
    ``cumul_passes × position`` and ``passes_received_share × position``
    each produce 4 position dummies.
    """
    # --- 1. Run-side prerequisite ------------------------------------------------
    run_share = _runs_top_third_share(main, runs)

    # --- 2. Press-side prerequisites --------------------------------------------
    pressed, pressing = _filter_press_to_main(press, main)
    press_panel = _press_aggregates_panel(pressed, pressing)

    # --- 3. Pass-side prerequisites ---------------------------------------------
    sender, receiver = _filter_pass_to_main(passes, main)
    pass_panel = _pass_aggregates_panel(sender, receiver)

    # --- 4. Assemble base from main + merges ------------------------------------
    base = main[[
        "player_appearance_id", "checkpoint", "fixture_id", "is_home",
        "position", "subbed",
        "cumul_shots", "last15_shots", "cumul_shots_top_third",
        "last15_sprints",
    ]].copy()
    base["subbed_int"] = _to_int_bool(base["subbed"])

    base = base.merge(run_share, on=["player_appearance_id", "checkpoint"], how="left")
    base = base.merge(press_panel, on=["player_appearance_id", "checkpoint"], how="left")
    base = base.merge(pass_panel, on=["player_appearance_id", "checkpoint"], how="left")

    # Press counts: 0 = "no event yet", true information.
    press_count_cols = (
        "cumul_press_events", "last15_press_events",
        "cumul_press_top_third", "cumul_press_turnovers",
        "cumul_forward_passes", "cumul_dir_passes",
        "cumul_press_accurate", "cumul_pressing_others",
    )
    base[list(press_count_cols)] = base[list(press_count_cols)].fillna(0).astype(int)

    pass_count_cols = (
        "cumul_passes", "cumul_passes_accurate", "cumul_passes_top_third",
        "last15_passes", "cumul_passes_received", "last15_passes_received",
    )
    base[list(pass_count_cols)] = base[list(pass_count_cols)].fillna(0).astype(int)

    # --- 5. Derived shares & accuracies (NaN where denominator collapses) -------
    base["_top_third_press_share"] = (
        base["cumul_press_top_third"]
        / base["cumul_press_events"].replace(0, np.nan)
    )
    base["_top_third_pass_share"] = (
        base["cumul_passes_top_third"]
        / base["cumul_passes"].replace(0, np.nan)
    )
    base["_share_shots_top_third"] = (
        base["cumul_shots_top_third"]
        / base["cumul_shots"].replace(0, np.nan)
    )
    base["_pass_accuracy"] = (
        base["cumul_passes_accurate"]
        / base["cumul_passes"].replace(0, np.nan)
    )
    # `_press_accuracy` = fraction of pressure events the player survived
    # (i.e. did NOT lose possession). This is the proper analogue of
    # `pass_accuracy` for the pressure domain — the raw `accurate` flag in
    # the press table is degenerate for directional passes (always True),
    # so we use the inverse turnover rate instead.
    base["_press_accuracy"] = 1.0 - (
        base["cumul_press_turnovers"]
        / base["cumul_press_events"].replace(0, np.nan)
    )
    base["_passes_received_share"] = (
        base["cumul_passes_received"]
        / (base["cumul_passes"] + base["cumul_passes_received"]).replace(0, np.nan)
    )

    # --- 6. Compose the 12 features ---------------------------------------------
    out = base[["player_appearance_id", "checkpoint"]].copy()

    # 1. four_domain_top_third_avg — mean of 4 share-features.
    spatial_cols = [
        "top_third_run_share", "_top_third_press_share",
        "_top_third_pass_share", "_share_shots_top_third",
    ]
    spatial_filled = base[spatial_cols].fillna(0.0)
    out["four_domain_top_third_avg"] = spatial_filled.mean(axis=1)
    all_spatial_nan = base[spatial_cols].isna().all(axis=1)
    out.loc[all_spatial_nan, "four_domain_top_third_avg"] = np.nan

    # 2. top_third_consistency_count — discrete count of domains with share > 0.3.
    out["top_third_consistency_count"] = (
        (base["top_third_run_share"].fillna(0.0) > 0.3).astype(int)
        + (base["_top_third_press_share"].fillna(0.0) > 0.3).astype(int)
        + (base["_top_third_pass_share"].fillna(0.0) > 0.3).astype(int)
        + (base["_share_shots_top_third"].fillna(0.0) > 0.3).astype(int)
    )

    # 3. top_third_pass × top_third_run — passing AND running in attacking third.
    out["top_third_pass_x_top_third_run"] = (
        base["_top_third_pass_share"] * base["top_third_run_share"]
    )

    # 4. cumul_passes × position — 4 dummies (recovers the negative pooled coef).
    for pos in POSITIONS:
        out[f"cumul_passes_{pos}"] = np.where(
            base["position"] == pos,
            base["cumul_passes"].astype(float),
            0.0,
        )

    # 5. passes_received_share × position — target-by-role identifier.
    for pos in POSITIONS:
        out[f"passes_received_share_{pos}"] = np.where(
            base["position"] == pos,
            base["_passes_received_share"],
            0.0,
        )

    # 6. shots_per_pass_received — clean target-striker proxy.
    out["shots_per_pass_received"] = (
        base["cumul_shots"]
        / base["cumul_passes_received"].replace(0, np.nan)
    )

    # 7. composure_under_pressure_ratio — relative composure under pressure.
    raw_ratio = base["_press_accuracy"] / (base["_pass_accuracy"] + EPS)
    out["composure_under_pressure_ratio"] = raw_ratio.clip(upper=10)

    # 8. last15_passes_received × last15_sprints — breakaway target signal.
    out["last15_passes_received_x_last15_sprints"] = (
        base["last15_passes_received"].astype(float)
        * base["last15_sprints"].astype(float)
    )

    # 9. passes_received_share_above_team_avg — team's main reception target.
    team_recv = base.groupby(
        ["fixture_id", "is_home", "checkpoint"]
    )["_passes_received_share"].transform("mean")
    out["passes_received_share_above_team_avg"] = (
        base["_passes_received_share"] - team_recv
    )

    # 10. subbed × last15_intensity_shots — impact-sub signal.
    out["subbed_x_last15_intensity_shots"] = (
        base["subbed_int"].astype(float) * (base["last15_shots"] / 15.0)
    )

    # 11. shot_to_pass_ratio — striker vs distributor fingerprint.
    out["shot_to_pass_ratio"] = (
        base["cumul_shots"] / base["cumul_passes"].replace(0, np.nan)
    )

    # 12. shot_to_press_ratio — clinical conversion under pressure.
    out["shot_to_press_ratio"] = (
        base["cumul_shots"] / base["cumul_press_events"].replace(0, np.nan)
    )

    out = out.replace([np.inf, -np.inf], np.nan)
    return out[list(FULL_CROSS_FEATURE_COLUMNS)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def save_cross_features(features: pd.DataFrame, path: str | Path) -> Path:
    """Write engineered features to ``path`` as CSV.

    Works for both :func:`build_cross_features` and
    :func:`build_press_cross_features` outputs.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_path, index=False)
    return out_path


__all__ = [
    "CROSS_FEATURE_COLUMNS",
    "PRESS_CROSS_FEATURE_COLUMNS",
    "FULL_CROSS_FEATURE_COLUMNS",
    "build_cross_features",
    "build_press_cross_features",
    "build_full_cross_features",
    "save_cross_features",
]

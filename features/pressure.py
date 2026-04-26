"""features/pressure.py - pressure-feature engineering pipeline.

Implements the curated manifest from `eda_player_appearance_behaviour_under_pressure.ipynb`.
The pressure table contributes ZERO columns to the main panel - so every
feature here is genuinely new and directly addresses RQ4 (does
pressure data improve goal-scoring prediction?).

Manifest:

| # | Feature                  | Verdict | r vs target |
|---|--------------------------|---------|------------:|
| 1 | top_third_press_share    | KEEP    | +0.108      |
| 2 | press_turnover_rate      | KEEP    | +0.069      |
| 3 | last15_press_events      | KEEP    | +0.052      |
| 4 | cumul_press_events       | cautious| -0.026      |
| 5 | forward_pass_share       | cautious| -0.038      |
| 6 | mean_abs_pass_angle      | cautious| -0.042      |
| 7 | cumul_pressing_others    | tactical| +0.002      |
| 8 | pressing_minus_pressed   | tactical| +0.029      |

Note on `press_accuracy`: the raw `accurate=True` flag is True for **every**
directional pass in this dataset by construction (only ball_carry and
turnover have variable accuracy). The proper composure measure is therefore
`1 - press_turnover_rate` rather than the bare accurate-rate.

Usage
-----
    from features.pressure import build_pressure_features

    features = build_pressure_features(press_df, main_df)
"""
from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
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


REQUIRED_PRESS_COLUMNS: Final[tuple[str, ...]] = (
    "id", "period", "player_appearance_id", "addressee_player_appearance_id",
    "accurate", "pressing_player_appearance_id", "press_induced_outcome",
    "pass_angle", "minute", "stage",
)

REQUIRED_MAIN_COLUMNS: Final[tuple[str, ...]] = (
    "player_appearance_id", "checkpoint", "fixture_id", "is_home", "position",
)


FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    # identifiers
    "player_appearance_id", "checkpoint", "fixture_id", "is_home", "position",
    # volume
    "cumul_press_events", "last15_press_events",
    # quality / composure
    "press_turnover_rate",
    # spatial
    "top_third_press_share",
    # tactical role
    "forward_pass_share",
    "mean_abs_pass_angle",
    "cumul_pressing_others",
    "pressing_minus_pressed",
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_inputs(press: pd.DataFrame, main: pd.DataFrame) -> None:
    missing_press = set(REQUIRED_PRESS_COLUMNS) - set(press.columns)
    if missing_press:
        raise ValueError(f"`press` is missing required columns: {sorted(missing_press)}")
    missing_main = set(REQUIRED_MAIN_COLUMNS) - set(main.columns)
    if missing_main:
        raise ValueError(f"`main` is missing required columns: {sorted(missing_main)}")


def _coerce_dtypes(press: pd.DataFrame) -> pd.DataFrame:
    """Cast pass_angle from string to float (NULL preserved as NaN)."""
    press = press.copy()
    press["pass_angle"] = pd.to_numeric(press["pass_angle"], errors="coerce")
    return press


def _split_views(
    press: pd.DataFrame, main: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two filtered views: pressed-side (player) and pressing-side (presser)."""
    main_apps = set(main["player_appearance_id"].unique())

    pressed = press.loc[press["player_appearance_id"].isin(main_apps)].copy()
    pressed["period_order"] = pressed["period"].map(PERIOD_ORDER)

    pressing = press.loc[press["pressing_player_appearance_id"].isin(main_apps)].copy()
    pressing["period_order"] = pressing["period"].map(PERIOD_ORDER)

    return pressed, pressing


def _aggregate_pressed(window_df: pd.DataFrame) -> pd.DataFrame:
    g = window_df.groupby("player_appearance_id")
    return pd.DataFrame({
        "events": g.size(),
        "top_third": g.apply(lambda x: (x["stage"] == "top").sum()),
        "turnovers": g.apply(lambda x: (x["press_induced_outcome"] == "turnover").sum()),
        "forward_passes": g.apply(
            lambda x: (x["press_induced_outcome"] == "forward_pass").sum()
        ),
        "dir_passes": g.apply(
            lambda x: x["press_induced_outcome"].isin(
                ("forward_pass", "backward_pass", "lateral_pass")
            ).sum()
        ),
        "abs_angle_sum": g.apply(lambda x: x["pass_angle"].abs().sum()),
        "abs_angle_n":   g.apply(lambda x: x["pass_angle"].notna().sum()),
    })


def _aggregate_pressing(window_df: pd.DataFrame) -> pd.DataFrame:
    g = window_df.groupby("pressing_player_appearance_id")
    out = pd.DataFrame({"pressing_others": g.size()})
    out.index.name = "player_appearance_id"
    return out


def _build_panel(pressed: pd.DataFrame, pressing: pd.DataFrame) -> pd.DataFrame:
    """Per (player_appearance_id, checkpoint) pressure aggregates with strict windowing."""
    rows: list[pd.DataFrame] = []
    for cp, period, cp_min in CHECKPOINTS:
        cp_ord = PERIOD_ORDER[period]

        cumul_p1 = (
            (pressed["period_order"] < cp_ord)
            | ((pressed["period_order"] == cp_ord) & (pressed["minute"] <= cp_min))
        )
        last15_p1 = (
            (pressed["period"] == period)
            & (pressed["minute"] > cp_min - 15)
            & (pressed["minute"] <= cp_min)
        )

        cumul_pressed = _aggregate_pressed(pressed.loc[cumul_p1]).add_prefix("cumul_")
        last15_pressed = _aggregate_pressed(pressed.loc[last15_p1]).add_prefix("last15_")

        cumul_p2 = (
            (pressing["period_order"] < cp_ord)
            | ((pressing["period_order"] == cp_ord) & (pressing["minute"] <= cp_min))
        )
        cumul_pressing_df = _aggregate_pressing(pressing.loc[cumul_p2]).add_prefix("cumul_")

        out = (
            cumul_pressed
            .join(last15_pressed, how="outer")
            .join(cumul_pressing_df, how="outer")
            .fillna(0)
        )
        out["checkpoint"] = cp
        rows.append(out.reset_index())

    return pd.concat(rows, ignore_index=True)


def _ratio(num: pd.Series, denom: pd.Series) -> pd.Series:
    return num / denom.replace(0, np.nan)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_pressure_features(
    press: pd.DataFrame,
    main: pd.DataFrame,
    *,
    drop_zero_press_ratios: bool = False,
) -> pd.DataFrame:
    """Engineer the curated pressure-feature manifest.

    Parameters
    ----------
    press
        Raw under-pressure event table.
    main
        Main panel.
    drop_zero_press_ratios
        If True, ratio features (`press_turnover_rate`,
        `top_third_press_share`, `forward_pass_share`) become 0 when the
        denominator collapses; if False (default) they remain NaN.

    Returns
    -------
    pandas.DataFrame
        One row per main-panel `(player_appearance_id, checkpoint)`,
        columns listed in :data:`FEATURE_COLUMNS`.
    """
    _validate_inputs(press, main)

    press = _coerce_dtypes(press)
    pressed, pressing = _split_views(press, main)
    panel = _build_panel(pressed, pressing)

    out = main[list(REQUIRED_MAIN_COLUMNS)].merge(
        panel, on=["player_appearance_id", "checkpoint"], how="left",
    )

    # Press counts: 0 = "no event yet", true info, fillna with 0.
    int_cols = [
        c for c in panel.columns
        if c not in ("player_appearance_id", "checkpoint")
        and c not in ("cumul_abs_angle_sum",)  # float
    ]
    for c in int_cols:
        if c in out.columns:
            out[c] = out[c].fillna(0)
    if "cumul_abs_angle_sum" in out.columns:
        out["cumul_abs_angle_sum"] = out["cumul_abs_angle_sum"].fillna(0.0)

    # Engineered features.
    out["press_turnover_rate"] = _ratio(out["cumul_turnovers"], out["cumul_events"])
    out["top_third_press_share"] = _ratio(out["cumul_top_third"], out["cumul_events"])
    out["forward_pass_share"] = _ratio(out["cumul_forward_passes"], out["cumul_dir_passes"])
    out["mean_abs_pass_angle"] = _ratio(out["cumul_abs_angle_sum"], out["cumul_abs_angle_n"])
    out["pressing_minus_pressed"] = out["cumul_pressing_others"] - out["cumul_events"]

    # Rename count columns to canonical names.
    out = out.rename(columns={
        "cumul_events": "cumul_press_events",
        "last15_events": "last15_press_events",
    })

    if drop_zero_press_ratios:
        ratio_cols = ["press_turnover_rate", "top_third_press_share",
                      "forward_pass_share", "mean_abs_pass_angle"]
        out[ratio_cols] = out[ratio_cols].fillna(0.0)

    return out[list(FEATURE_COLUMNS)].reset_index(drop=True)


def save_pressure_features(features: pd.DataFrame, path: str | Path) -> Path:
    """Write engineered pressure features to ``path`` as CSV."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_path, index=False)
    return out_path


__all__ = [
    "FEATURE_COLUMNS",
    "build_pressure_features",
    "save_pressure_features",
]

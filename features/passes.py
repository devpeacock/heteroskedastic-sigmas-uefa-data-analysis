"""features/passes.py - pass-feature engineering pipeline.

Implements the curated manifest from `eda_player_appearance_pass.ipynb`.
The pass table contributes ZERO columns to the main panel - so every
feature here is genuinely new and directly addresses RQ4.

Manifest:

| # | Feature                | Verdict | r vs target |
|---|------------------------|---------|------------:|
| 1 | top_third_pass_share   | KEEP    | +0.115      |
| 2 | passes_received_share  | KEEP    | +0.071      |
| 3 | cumul_passes           | KEEP*   | -0.102      |
| 4 | last15_passes          | cautious| -0.050      |

\\* `cumul_passes` is **negative pooled** (distributors pass more, score
less). The signal flips for attackers - downstream models must include
a `cumul_passes x position` interaction (already provided by
`features.cross.build_full_cross_features`). Kept here for completeness.

Dropped features (failed BH q=0.05 + flat / collinear):
    pass_accuracy (r=-0.010, p_bh=0.79 - flattest feature in dataset),
    cumul_passes_received (rho=0.98 with cumul_passes),
    passes_per_minute_played (rho=0.84 with last15_passes, rho=0.68 with cumul_passes),
    cumul_top_third_passes (parent flat),
    cumul_unspecified_passes (rare event),
    unspecified_pass_share (flat),
    last15_pass_uplift (RQ6 anchor failed).

Usage
-----
    from features.passes import build_pass_features

    features = build_pass_features(passes_df, main_df)
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


REQUIRED_PASS_COLUMNS: Final[tuple[str, ...]] = (
    "id", "period", "player_appearance_id", "addressee_player_appearance_id",
    "accurate", "minute", "stage",
)

REQUIRED_MAIN_COLUMNS: Final[tuple[str, ...]] = (
    "player_appearance_id", "checkpoint", "fixture_id", "is_home", "position",
)


FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    # identifiers
    "player_appearance_id", "checkpoint", "fixture_id", "is_home", "position",
    # volume
    "cumul_passes", "last15_passes",
    # spatial concentration
    "top_third_pass_share",
    # tactical role
    "passes_received_share",
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_inputs(passes: pd.DataFrame, main: pd.DataFrame) -> None:
    missing_pass = set(REQUIRED_PASS_COLUMNS) - set(passes.columns)
    if missing_pass:
        raise ValueError(f"`passes` is missing required columns: {sorted(missing_pass)}")
    missing_main = set(REQUIRED_MAIN_COLUMNS) - set(main.columns)
    if missing_main:
        raise ValueError(f"`main` is missing required columns: {sorted(missing_main)}")


def _split_views(
    passes: pd.DataFrame, main: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sender-side and receiver-side views, both filtered to main appearances."""
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


def _aggregate_sender(window_df: pd.DataFrame) -> pd.DataFrame:
    g = window_df.groupby("player_appearance_id")
    return pd.DataFrame({
        "passes": g.size(),
        "top_third": g.apply(lambda x: (x["stage"] == "top").sum()),
    })


def _aggregate_receiver(window_df: pd.DataFrame) -> pd.DataFrame:
    g = window_df.groupby("addressee_player_appearance_id")
    out = pd.DataFrame({"received": g.size()})
    out.index.name = "player_appearance_id"
    return out


def _build_panel(sender: pd.DataFrame, receiver: pd.DataFrame) -> pd.DataFrame:
    """Per (player_appearance_id, checkpoint) pass aggregates with strict windowing."""
    rows: list[pd.DataFrame] = []
    for cp, period, cp_min in CHECKPOINTS:
        cp_ord = PERIOD_ORDER[period]

        cumul_s = (
            (sender["period_order"] < cp_ord)
            | ((sender["period_order"] == cp_ord) & (sender["minute"] <= cp_min))
        )
        last15_s = (
            (sender["period"] == period)
            & (sender["minute"] > cp_min - 15)
            & (sender["minute"] <= cp_min)
        )
        cumul_r = (
            (receiver["period_order"] < cp_ord)
            | ((receiver["period_order"] == cp_ord) & (receiver["minute"] <= cp_min))
        )

        cumul_send = _aggregate_sender(sender.loc[cumul_s]).add_prefix("cumul_")
        last15_send = _aggregate_sender(sender.loc[last15_s]).add_prefix("last15_")
        cumul_recv = _aggregate_receiver(receiver.loc[cumul_r]).add_prefix("cumul_")

        out = (
            cumul_send
            .join(last15_send, how="outer")
            .join(cumul_recv, how="outer")
            .fillna(0)
            .astype(int)
        )
        out["checkpoint"] = cp
        rows.append(out.reset_index())

    return pd.concat(rows, ignore_index=True)


def _ratio(num: pd.Series, denom: pd.Series) -> pd.Series:
    return num / denom.replace(0, np.nan)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_pass_features(
    passes: pd.DataFrame,
    main: pd.DataFrame,
    *,
    drop_zero_pass_ratios: bool = False,
) -> pd.DataFrame:
    """Engineer the curated pass-feature manifest.

    Parameters
    ----------
    passes
        Raw pass-event table (`player_appearance_pass.csv`).
    main
        Main panel.
    drop_zero_pass_ratios
        If True, ratio features become 0 when their denominator
        collapses; if False (default) they remain NaN.

    Returns
    -------
    pandas.DataFrame
        One row per main-panel `(player_appearance_id, checkpoint)`,
        columns listed in :data:`FEATURE_COLUMNS`.
    """
    _validate_inputs(passes, main)

    sender, receiver = _split_views(passes, main)
    panel = _build_panel(sender, receiver)

    out = main[list(REQUIRED_MAIN_COLUMNS)].merge(
        panel, on=["player_appearance_id", "checkpoint"], how="left",
    )

    # Pass counts: 0 = "no pass yet".
    count_cols = [
        c for c in panel.columns
        if c not in ("player_appearance_id", "checkpoint")
    ]
    for c in count_cols:
        if c in out.columns:
            out[c] = out[c].fillna(0).astype(int)

    # Engineered features.
    out = out.rename(columns={
        "cumul_passes": "cumul_passes",
        "last15_passes": "last15_passes",
    })  # no-op rename for clarity
    out["top_third_pass_share"] = _ratio(out["cumul_top_third"], out["cumul_passes"])
    out["passes_received_share"] = _ratio(
        out["cumul_received"], out["cumul_passes"] + out["cumul_received"]
    )

    if drop_zero_pass_ratios:
        ratio_cols = ["top_third_pass_share", "passes_received_share"]
        out[ratio_cols] = out[ratio_cols].fillna(0.0)

    return out[list(FEATURE_COLUMNS)].reset_index(drop=True)


def save_pass_features(features: pd.DataFrame, path: str | Path) -> Path:
    """Write engineered pass features to ``path`` as CSV."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_path, index=False)
    return out_path


__all__ = [
    "FEATURE_COLUMNS",
    "build_pass_features",
    "save_pass_features",
]

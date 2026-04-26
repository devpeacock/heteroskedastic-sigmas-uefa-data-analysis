"""Rank every engineered feature against `scored_after` and narrow the set.

Pipeline
--------
1. Load `features/all_engineered_features.csv` and merge onto main to get
   the target.
2. For each numeric feature compute:
     - Pearson r vs target
     - Best non-baseline binned rate (with Wilson 95% CI)
     - Cluster-robust logistic-regression p-value (cluster on fixture_id)
     - BH-FDR adjusted p
3. Apply a verdict policy:
     - tier S : top by |r| AND BH-significant (5-8 features)
     - tier A : BH q=0.05 + best-bin rate above baseline (the manifest)
     - tier B : Pearson borderline OR cautious cross-domain
     - tier C : drop (flat / collinear / artefact)
4. Greedy collinearity pruning: drop any feature with |Pearson| >= 0.85
   against an already-selected higher-ranked feature.
5. Save:
     - features/feature_ranking.csv     (full table)
     - features/curated_features.csv    (narrow set after pruning)

Usage
-----
    python scripts/rank_features.py
    python scripts/rank_features.py --rho-threshold 0.90
    python scripts/rank_features.py --max-features 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--features", type=Path,
                   default=Path("features/all_engineered_features.csv"))
    p.add_argument("--main", type=Path,
                   default=Path("data/players_quarters_final.csv"))
    p.add_argument("--output-ranking", type=Path,
                   default=Path("features/feature_ranking.csv"))
    p.add_argument("--output-curated", type=Path,
                   default=Path("features/curated_features_final.csv"))
    p.add_argument("--rho-threshold", type=float, default=0.85,
                   help="Drop a feature collinear at |rho| >= this with any "
                        "higher-ranked feature already selected.")
    p.add_argument("--max-features", type=int, default=35,
                   help="Cap the curated set at this many features.")
    p.add_argument("--bh-q", type=float, default=0.05,
                   help="BH-FDR q-value cutoff for tier-A.")
    p.add_argument("--seed", type=int, default=42,
                   help="Global random seed used for deterministic behaviour.")
    p.add_argument("--single-thread", action="store_true",
                   help="Force single-threaded numeric backends for stable outputs.")
    p.add_argument("--metadata-out", type=Path, default=None,
                   help="Optional metadata JSON output path. Defaults to <output_curated>.metadata.json")
    return p.parse_args()


def wilson_ci(s: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    from scipy import stats as _stats
    p = s / n
    z = _stats.norm.ppf(1.0 - alpha / 2.0)
    denom = 1.0 + (z * z) / n
    centre = (p + (z * z) / (2.0 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + (z * z) / (4 * n * n)) / denom
    return float(centre - half), float(centre + half)


def extreme_quantile_rate(
    series: pd.Series, target: pd.Series, baseline: float, q: float = 0.1,
) -> tuple[float, float, float, int, str]:
    """Return the most extreme target rate observed in either tail of the feature.

    Computes rate at top-`q` and bottom-`q` quantile cut-offs, picks the
    tail whose rate is furthest from `baseline`. This is robust to
    heavy-zero distributions (position dummies, intensity uplifts) where
    quartile-style binning collapses.

    Returns
    -------
    rate, ci_lo, ci_hi, n, direction
        ``direction`` is "top" or "bottom" indicating which tail won.
    """
    s = series.dropna()
    t = target.loc[s.index]
    n = len(s)
    if s.std() == 0 or n < 30:
        return float("nan"), float("nan"), float("nan"), 0, "n/a"

    # Take the TOP q-fraction by value. For features with a heavy zero
    # mass we may end up with a cluster of ties at 0; fall back to using
    # all non-zero rows in that case.
    n_top = max(int(n * q), 30)
    top_threshold = s.nlargest(n_top).iloc[-1]
    top_idx = s.index[s >= top_threshold]
    if len(top_idx) > n * 0.5:  # too many ties – use strict > 0
        top_idx = s.index[s > 0]
    if len(top_idx) < 30:
        top_idx = s.index[s >= top_threshold]
    top_n = len(top_idx)
    top_succ = int(t.loc[top_idx].sum()) if top_n else 0
    top_rate = top_succ / top_n if top_n else float("nan")

    # Bottom q-fraction.
    n_bot = max(int(n * q), 30)
    bot_threshold = s.nsmallest(n_bot).iloc[-1]
    bot_idx = s.index[s <= bot_threshold]
    if len(bot_idx) > n * 0.5:
        bot_idx = s.index[s == 0]
    if len(bot_idx) < 30:
        bot_idx = s.index[s <= bot_threshold]
    bot_n = len(bot_idx)
    bot_succ = int(t.loc[bot_idx].sum()) if bot_n else 0
    bot_rate = bot_succ / bot_n if bot_n else float("nan")

    # Pick the tail furthest from baseline.
    top_dist = abs(top_rate - baseline) if not pd.isna(top_rate) else 0
    bot_dist = abs(bot_rate - baseline) if not pd.isna(bot_rate) else 0

    if top_dist >= bot_dist:
        n_w, succ, direction = top_n, top_succ, "top"
        rate = top_rate
    else:
        n_w, succ, direction = bot_n, bot_succ, "bottom"
        rate = bot_rate

    lo, hi = wilson_ci(succ, n_w)
    return float(rate), lo, hi, int(n_w), direction


def cluster_robust_glm(
    df: pd.DataFrame, feature: str
) -> tuple[float, float, int]:
    """One-feature logistic regression with fixture-clustered SEs."""
    sub = df[[feature, "scored_after", "fixture_id"]].dropna()
    if sub[feature].std() == 0 or len(sub) < 30:
        return float("nan"), float("nan"), len(sub)
    try:
        import statsmodels.formula.api as smf
        m = smf.logit(f"scored_after ~ {feature}", data=sub).fit(
            disp=False, maxiter=200,
            cov_type="cluster", cov_kwds={"groups": sub["fixture_id"]},
        )
        return float(m.params[feature]), float(m.pvalues[feature]), len(sub)
    except Exception:
        return float("nan"), float("nan"), len(sub)


def assign_tier(row: pd.Series, baseline: float) -> str:
    """Verdict policy.

    Tiers (in priority order):
    * **C - drop (separation)** spurious BH p from perfect target separation
      (e.g. goalkeeper dummies where the target is identically zero on a
      large mass of the column). Detected via near-zero ``best_rate`` AND
      tiny ``p_bh`` (<1e-30) AND moderate Pearson r.
    * **S - top**     |r| >= 0.10 AND BH-significant
    * **A - keep**    BH-significant AND best_rate off-baseline
    * **B - cautious** borderline
    * **C - drop**    flat / no signal
    """
    if pd.isna(row["best_rate"]):
        return "C - drop (no data)"

    abs_r = abs(row["pearson_r"]) if not pd.isna(row["pearson_r"]) else 0.0
    bh_sig = (not pd.isna(row["p_bh"])) and row["p_bh"] < 0.05

    # Perfect-separation guard: an attribute that perfectly identifies a
    # zero-target subgroup (typically goalkeeper dummies) returns p_bh
    # essentially equal to zero from logistic regression, which is a
    # numerical artefact rather than real signal. Demote them.
    if bh_sig and (not pd.isna(row["p_bh"])) and row["p_bh"] < 1e-30:
        if not pd.isna(row["best_rate"]) and row["best_rate"] <= 0.005:
            return "C - drop (separation)"

    rate_above = row["best_rate"] > baseline * 1.3
    rate_below = row["best_rate"] < baseline * 0.7
    rate_off_baseline = rate_above or rate_below
    ci_off = (
        (not pd.isna(row["ci_lo"]) and row["ci_lo"] > baseline)
        or (not pd.isna(row["ci_hi"]) and row["ci_hi"] < baseline)
    )

    if abs_r >= 0.10 and bh_sig:
        return "S - top"
    if bh_sig and (rate_off_baseline or ci_off):
        return "A - keep"
    if abs_r >= 0.07 and bh_sig:
        return "A - keep"
    if ci_off or rate_off_baseline:
        return "B - cautious (rate)"
    if abs_r >= 0.05:
        return "B - cautious (r)"
    return "C - drop (flat)"


def collinearity_prune(
    ranking: pd.DataFrame, features_df: pd.DataFrame, rho_threshold: float
) -> tuple[list[str], list[tuple[str, str, float]]]:
    """Greedy collinearity pruning by descending tier + |r|.

    Walks the ranking top-down. A feature is kept iff its |Pearson| with
    every already-selected feature is < ``rho_threshold``.

    Returns
    -------
    kept
        Feature names in selection order.
    dropped
        ``(dropped_feature, conflicting_kept_feature, rho)`` triples.
    """
    feature_cols = [c for c in features_df.columns
                    if c not in ("player_appearance_id", "checkpoint")]
    corr = features_df[feature_cols].corr(method="pearson").abs()

    tier_order = {"S - top": 0, "A - keep": 1,
                  "B - cautious (CI)": 2, "B - cautious (rate)": 3,
                  "B - cautious (r)": 4, "C - drop (flat)": 5,
                  "C - drop (separation)": 6, "C - drop (no data)": 7}
    ordered = (
        ranking.assign(_tier=ranking["tier"].map(tier_order))
               .assign(_abs_r=ranking["pearson_r"].abs())
               .sort_values(["_tier", "_abs_r"], ascending=[True, False])
               ["feature"]
               .tolist()
    )

    kept: list[str] = []
    dropped: list[tuple[str, str, float]] = []
    for f in ordered:
        if f not in corr.columns:
            continue
        conflict = None
        for k in kept:
            if k not in corr.columns:
                continue
            r = corr.at[f, k]
            if pd.notna(r) and r >= rho_threshold:
                conflict = (k, float(r))
                break
        if conflict is None:
            kept.append(f)
        else:
            dropped.append((f, conflict[0], conflict[1]))
    return kept, dropped


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from reproducibility import (
        collect_runtime_metadata,
        set_global_determinism,
        sha256_file,
        write_json,
    )

    set_global_determinism(args.seed, force_single_thread=args.single_thread)

    main_df = pd.read_csv(args.main)
    feats = pd.read_csv(args.features)
    panel = main_df[["player_appearance_id", "checkpoint",
                     "scored_after", "fixture_id"]].merge(
        feats, on=["player_appearance_id", "checkpoint"], how="left",
    )
    baseline = panel["scored_after"].mean()
    print(f"baseline scored_after rate: {baseline:.4f}")
    print(f"features to rank: "
          f"{len(feats.columns) - 2} engineered + 0 main")
    print()

    feature_cols = [c for c in feats.columns
                    if c not in ("player_appearance_id", "checkpoint")]

    # --- Compute stats ------------------------------------------------------
    print("scoring features (pearson + best-bin + cluster-robust GLM)...")
    rows: list[dict] = []
    for i, f in enumerate(feature_cols, 1):
        s = panel[f].dropna()
        n = len(s)
        if s.std() == 0 or n < 30:
            r = float("nan")
        else:
            r = float(np.corrcoef(s, panel.loc[s.index, "scored_after"])[0, 1])

        rate, lo, hi, n_bin, direction = extreme_quantile_rate(
            panel[f], panel["scored_after"], baseline,
        )
        coef, p_raw, n_sig = cluster_robust_glm(panel, f)

        rows.append({
            "feature": f, "n": n, "pearson_r": r,
            "best_rate": rate, "ci_lo": lo, "ci_hi": hi, "n_bin": n_bin,
            "best_direction": direction,
            "coef": coef, "p_raw": p_raw,
        })
        if i % 15 == 0 or i == len(feature_cols):
            print(f"  {i}/{len(feature_cols)} done")

    ranking = pd.DataFrame(rows)

    # --- BH-FDR adjustment --------------------------------------------------
    from statsmodels.stats.multitest import multipletests
    mask = ranking["p_raw"].notna()
    p_adj = np.full(len(ranking), np.nan)
    if mask.sum():
        _, adj, _, _ = multipletests(ranking.loc[mask, "p_raw"].values, method="fdr_bh")
        p_adj[mask.values] = adj
    ranking["p_bh"] = p_adj
    ranking["bh_sig_q05"] = ranking["p_bh"] < args.bh_q

    # --- Tier assignment ----------------------------------------------------
    ranking["tier"] = ranking.apply(lambda r: assign_tier(r, baseline), axis=1)

    # --- Sort by signal strength -------------------------------------------
    ranking["_abs_r"] = ranking["pearson_r"].abs()
    ranking = ranking.sort_values(
        ["bh_sig_q05", "_abs_r"], ascending=[False, False]
    ).reset_index(drop=True).drop(columns="_abs_r")

    print()
    print("tier counts:")
    print(ranking["tier"].value_counts().to_string())
    print()
    print("top 20:")
    print(ranking.head(20)[["feature", "n", "pearson_r", "best_rate",
                            "best_direction", "p_bh", "tier"]].to_string(index=False))
    print()

    args.output_ranking.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(args.output_ranking, index=False)
    print(f"wrote ranking: {args.output_ranking}")

    # --- Collinearity pruning ----------------------------------------------
    print()
    print(f"collinearity pruning at |rho| >= {args.rho_threshold}...")
    kept, dropped = collinearity_prune(
        ranking, feats, rho_threshold=args.rho_threshold,
    )
    print(f"  kept: {len(kept)}, dropped: {len(dropped)}")
    if dropped[:8]:
        print("  examples (dropped, kept-conflict, rho):")
        for d, k, r in dropped[:8]:
            print(f"    {d:55s} <- {k:40s} rho={r:.3f}")
    print()

    # Cap at max-features
    capped = kept[: args.max_features]
    print(f"after cap @ {args.max_features}: {len(capped)} features")

    curated_cols = ["player_appearance_id", "checkpoint"] + capped
    curated = feats[curated_cols]
    args.output_curated.parent.mkdir(parents=True, exist_ok=True)
    curated.to_csv(args.output_curated, index=False)
    print(f"wrote curated set: {args.output_curated}")
    metadata_out = (
        args.metadata_out
        if args.metadata_out is not None
        else args.output_curated.with_suffix(args.output_curated.suffix + ".metadata.json")
    )
    metadata = {
        **collect_runtime_metadata(
            project_root=project_root,
            seed=args.seed,
            argv=sys.argv,
            args_dict=vars(args),
            force_single_thread=args.single_thread,
        ),
        "inputs": {
            "features": {
                "path": str(args.features.resolve()),
                "sha256": sha256_file(args.features),
            },
            "main": {
                "path": str(args.main.resolve()),
                "sha256": sha256_file(args.main),
            },
        },
        "outputs": {
            "ranking_csv": {
                "path": str(args.output_ranking.resolve()),
                "sha256": sha256_file(args.output_ranking),
            },
            "curated_csv": {
                "path": str(args.output_curated.resolve()),
                "sha256": sha256_file(args.output_curated),
            },
        },
    }
    write_json(metadata_out, metadata)
    print(f"wrote {metadata_out}")
    print(f"  shape: {curated.shape}")

    # --- Final tier breakdown of curated set --------------------------------
    print()
    print("curated set tier breakdown:")
    cur_tiers = ranking.set_index("feature").loc[capped, "tier"]
    print(cur_tiers.value_counts().to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

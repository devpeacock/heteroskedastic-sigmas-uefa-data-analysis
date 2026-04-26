"""Render the 9 confusion matrices in a compact 2-row x 5-col grid."""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ART = PROJECT_ROOT / "models"


def load_test_pred(model_dir: str, pred_col: str = "test_pred"):
    df = pd.read_csv(ART / model_dir / "test_predictions.csv")
    y = df["scored_after"].values
    pred = df[pred_col].values
    return y, pred


def top_k_per_match(proba, fixture_ids, checkpoints, position_arr,
                    k=5, threshold=0.07, allowed=("A", "M")):
    """02f-style: top-K + position filter + threshold."""
    df = pd.DataFrame({
        "proba": proba, "fixture_id": fixture_ids, "checkpoint": checkpoints,
        "position": position_arr, "_idx": np.arange(len(proba)),
    })
    eligible = df["position"].isin(allowed) & (df["proba"] >= threshold)
    df["rank"] = df.loc[eligible].groupby(
        ["fixture_id", "checkpoint"]
    )["proba"].rank(method="first", ascending=False)
    df["pred"] = (eligible & (df["rank"] <= k)).fillna(False).astype(int)
    return df.sort_values("_idx").reset_index(drop=True)["pred"].values


def main():
    main_panel = pd.read_csv(PROJECT_ROOT / "data" / "players_quarters_final.csv")
    pos_lookup = main_panel[
        ["player_appearance_id", "checkpoint", "position"]
    ].drop_duplicates()

    def get_pos(art):
        df = pd.read_csv(art).merge(
            pos_lookup, on=["player_appearance_id", "checkpoint"], how="left"
        )
        return df["position"].values

    # Load all 02b_AP test predictions (has all variants).
    df_ap = pd.read_csv(ART / "kitchen_sink_AP" / "test_predictions.csv")
    y = df_ap["scored_after"].values

    # Pred 1: 02b kitchen sink (global BA thr).
    df_b = pd.read_csv(ART / "kitchen_sink" / "test_predictions.csv")
    cfg_b = json.loads((ART / "kitchen_sink" / "config.json").read_text())
    pred_02b = (df_b["test_cal_global"].values >= cfg_b["global_threshold"]).astype(int)

    # Pred 2-5: 02b_AP variants from the saved CSV.
    pred_02bAP_g_BA = df_ap["pred_g_ba"].values
    pred_02bAP_g_F1 = df_ap["pred_g_f1"].values
    pred_02bAP_p_F1 = df_ap["pred_pcp_pcpf1"].values
    pred_02bAP_p_BA = df_ap["pred_pcp_pcpba"].values

    # Pred 6: 02e advanced (per-pos BA thr).
    df_e = pd.read_csv(ART / "advanced" / "test_predictions.csv")
    cfg_e = json.loads((ART / "advanced" / "config.json").read_text())
    pos_e = get_pos(ART / "advanced" / "test_predictions.csv")
    pred_02e = np.zeros(len(df_e), dtype=int)
    for p, t in cfg_e["percp_thresholds"].items():
        m = pos_e == p
        pred_02e[m] = (df_e["test_proba_calibrated"].values[m] >= t).astype(int)

    # Pred 7: 02e + 02f (top-5 + A+M + F1).
    cfg_f = json.loads((ART / "precision" / "config.json").read_text())
    strat = cfg_f["recommended_strategy"]
    pred_02e_02f = top_k_per_match(
        df_e["test_proba_calibrated"].values,
        df_e["fixture_id"].values, df_e["checkpoint"].values, pos_e,
        k=strat["k"], threshold=strat["threshold"],
        allowed=tuple(strat["allowed_positions"]),
    )

    # Pred 8: 02g AP-first (per-pos F1 thr).
    df_g = pd.read_csv(ART / "precision_first" / "test_predictions.csv")
    cfg_g = json.loads((ART / "precision_first" / "config.json").read_text())
    pos_g = get_pos(ART / "precision_first" / "test_predictions.csv")
    pred_02g = np.zeros(len(df_g), dtype=int)
    for p, t in cfg_g["percp_thresholds_f1"].items():
        m = pos_g == p
        pred_02g[m] = (df_g["test_proba_calibrated"].values[m] >= t).astype(int)

    # Pred 9: 02g + 02f.
    pred_02g_02f = top_k_per_match(
        df_g["test_proba_calibrated"].values,
        df_g["fixture_id"].values, df_g["checkpoint"].values, pos_g,
        k=5, threshold=cfg_g["f1_threshold_global"],
        allowed=("A", "M"),
    )

    strategies = [
        ("02b kitchen sink", pred_02b),
        ("02b_AP global (BA thr)", pred_02bAP_g_BA),
        ("02b_AP global (F1 thr)", pred_02bAP_g_F1),
        ("02b_AP per-cp (per-cp F1 thr)", pred_02bAP_p_F1),
        ("02b_AP per-cp (per-cp BA thr)", pred_02bAP_p_BA),
        ("02e advanced (per-pos BA thr)", pred_02e),
        ("02e + 02f (top-5 + A+M + F1)", pred_02e_02f),
        ("02g AP-first (per-pos F1 thr)", pred_02g),
        ("02g + 02f (top-5 + A+M + F1)", pred_02g_02f),
    ]

    # Render in 2 rows x 5 cols (one empty slot).
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    axes = axes.flatten()

    for ax, (name, pred) in zip(axes, strategies):
        cm = confusion_matrix(y, pred)
        ax.imshow(cm, cmap="Blues", aspect="auto")
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=15, fontweight="bold",
                )
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["pred 0", "pred 1"], fontsize=9)
        ax.set_yticklabels(["actual 0", "actual 1"], fontsize=9)
        p = precision_score(y, pred, zero_division=0)
        r = recall_score(y, pred)
        ax.set_title(f"{name}\nP={p:.2f}, R={r:.2f}", fontsize=10)

    # Hide unused subplot slot.
    for ax in axes[len(strategies):]:
        ax.axis("off")

    plt.tight_layout()
    out = ART / "confusion_matrices_grid.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()

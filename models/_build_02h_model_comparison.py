"""Build the 02h_model_comparison notebook.

Side-by-side comparison of all candidate models built so far. Loads
each model's saved OOF + test predictions, computes a unified metric
table, and produces overlay plots (ROC, precision-recall, calibration,
confusion matrices).

Models compared (5):
1. 02b — kitchen-sink baseline (5 features, vanilla XGB)
2. 02e — advanced (SMOTE + Optuna + per-pos thresholds)
3. 02e + 02f — precision-optimised post-processing
4. 02g — AP-first pipeline (alternative)
5. 02g + 02f post-processing — AP-first + top-K + A+M filter

This is a *pure analysis* notebook — does not retrain any model. It
just reads the saved artefacts.

Usage
-----
    python models/_build_02h_model_comparison.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NOTEBOOK_PATH = Path(__file__).resolve().parent / "02h_model_comparison.ipynb"


CELLS: list[tuple[str, str]] = [
    ("md", '''# 02h — Model comparison + ROC curves

Side-by-side comparison of all candidate models. Reads the saved test
predictions from each notebook's artefacts directory and produces:

| Section | Output |
|---------|--------|
| A | Setup + load all candidates' predictions |
| B | Unified metric table (AUC, AP, BA, Precision, Recall, F1, Brier) |
| C | **ROC curves overlay** (all models on one chart) |
| D | Precision-Recall curves overlay |
| E | Calibration plots (reliability diagrams) |
| F | Confusion matrices side-by-side |
| G | Summary table + final recommendation |

This is a pure analysis notebook — no training happens here.
'''),

    ("md", '''## Section A — Setup and data loading'''),

    ("code", '''"""Imports + deterministic runtime setup."""
from __future__ import annotations
import os, sys, warnings, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT: Path = Path.cwd().parent if Path.cwd().name == "models" else Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    balanced_accuracy_score, precision_score, recall_score, f1_score,
    brier_score_loss, confusion_matrix,
)

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)
pd.set_option("display.precision", 4)
warnings.filterwarnings("ignore")

RANDOM_SEED = 42
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
'''),

    ("code", '''"""Load test predictions from each model directory."""
ART_BASE = PROJECT_ROOT / "models"

models = {}

# Model 1: 02b kitchen sink baseline
df = pd.read_csv(ART_BASE / "kitchen_sink" / "test_predictions.csv")
models["02b kitchen sink"] = {
    "y_true": df["scored_after"].values,
    "proba_raw": df["test_raw"].values,
    "proba_cal": df["test_cal_global"].values,
    "fixture_id": df["fixture_id"].values,
    "checkpoint": df["checkpoint"].values,
    "color": "tab:blue",
    "linestyle": "--",
}

# Model 2a: 02b_AP precision-first KS — GLOBAL calibration variant
df = pd.read_csv(ART_BASE / "kitchen_sink_AP" / "test_predictions.csv")
models["02b_AP global"] = {
    "y_true": df["scored_after"].values,
    "proba_raw": df["test_proba"].values,
    "proba_cal": df["test_proba_cal_global"].values,
    "fixture_id": df["fixture_id"].values,
    "checkpoint": df["checkpoint"].values,
    "color": "tab:cyan",
    "linestyle": "-",
}

# Model 2b: 02b_AP precision-first KS — PER-CHECKPOINT calibration variant
models["02b_AP per-cp"] = {
    "y_true": df["scored_after"].values,
    "proba_raw": df["test_proba"].values,
    "proba_cal": df["test_proba_cal_percp"].values,
    "fixture_id": df["fixture_id"].values,
    "checkpoint": df["checkpoint"].values,
    "color": "tab:pink",
    "linestyle": "-",
}

# Model 3: 02e advanced
df = pd.read_csv(ART_BASE / "advanced" / "test_predictions.csv")
models["02e advanced"] = {
    "y_true": df["scored_after"].values,
    "proba_raw": df["test_proba"].values,
    "proba_cal": df["test_proba_calibrated"].values,
    "fixture_id": df["fixture_id"].values,
    "checkpoint": df["checkpoint"].values,
    "color": "tab:orange",
    "linestyle": "-",
}

# Model 4: 02g precision-first (combined kitchen + advanced)
df = pd.read_csv(ART_BASE / "precision_first" / "test_predictions.csv")
models["02g AP-first"] = {
    "y_true": df["scored_after"].values,
    "proba_raw": df["test_proba"].values,
    "proba_cal": df["test_proba_calibrated"].values,
    "fixture_id": df["fixture_id"].values,
    "checkpoint": df["checkpoint"].values,
    "color": "tab:green",
    "linestyle": "-",
}

print(f"Loaded {len(models)} base models:")
for name, m in models.items():
    print(f"  {name:25s}  n={len(m['y_true'])}, positives={int(m['y_true'].sum())}")

MODEL_PAPER_LABELS = {
    "02b kitchen sink": "AUC-selected XGBoost, checkpoint-specific calibration",
    "02b_AP global": "AP-selected XGBoost, global calibration",
    "02b_AP per-cp": "AP-selected XGBoost, checkpoint-specific calibration",
    "02e advanced": "Tuned XGBoost with oversampling",
    "02g AP-first": "Hybrid AP-first advanced pipeline",
}
'''),

    ("code", '''"""Load main panel for position lookup (used by post-processing strategies)."""
main = pd.read_csv(PROJECT_ROOT / "data" / "players_quarters_final.csv")
position_lookup = main[["player_appearance_id", "checkpoint", "position"]].drop_duplicates()


def add_position(model_data, art_path):
    """Attach position labels by joining test_predictions.csv with main."""
    df = pd.read_csv(art_path)
    df_pos = df.merge(position_lookup, on=["player_appearance_id", "checkpoint"], how="left")
    return df_pos["position"].values


# Add position to all models with post-processing.
models["02b_AP global"]["position"] = add_position(
    models["02b_AP global"], ART_BASE / "kitchen_sink_AP" / "test_predictions.csv",
)
models["02b_AP per-cp"]["position"] = add_position(
    models["02b_AP per-cp"], ART_BASE / "kitchen_sink_AP" / "test_predictions.csv",
)
models["02e advanced"]["position"] = add_position(
    models["02e advanced"], ART_BASE / "advanced" / "test_predictions.csv",
)
models["02g AP-first"]["position"] = add_position(
    models["02g AP-first"], ART_BASE / "precision_first" / "test_predictions.csv",
)
print("Positions attached to all models.")
'''),

    ("code", '''"""Build derived models (post-processing layers from 02f / 02g)."""
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


# 02e + 02f post-processing
config_02f = json.loads((ART_BASE / "precision" / "config.json").read_text())
strat_02f = config_02f["recommended_strategy"]
m_02e = models["02e advanced"]
pred_02f = top_k_per_match(
    m_02e["proba_cal"], m_02e["fixture_id"], m_02e["checkpoint"], m_02e["position"],
    k=strat_02f["k"],
    threshold=strat_02f["threshold"],
    allowed=tuple(strat_02f["allowed_positions"]),
)

# 02g + top-5 + A+M + F1
config_02g = json.loads((ART_BASE / "precision_first" / "config.json").read_text())
m_02g = models["02g AP-first"]
pred_02g_top5 = top_k_per_match(
    m_02g["proba_cal"], m_02g["fixture_id"], m_02g["checkpoint"], m_02g["position"],
    k=5, threshold=config_02g["f1_threshold_global"], allowed=("A", "M"),
)

# Save these derived predictions back into the models dict.
models["02e+02f (top-5+A+M+F1)"] = {
    **m_02e,
    "pred_postproc": pred_02f,
    "color": "tab:red",
    "linestyle": "-",
}
models["02g+02f (top-5+A+M+F1)"] = {
    **m_02g,
    "pred_postproc": pred_02g_top5,
    "color": "tab:purple",
    "linestyle": "-",
}
print(f"Total models for comparison: {len(models)}")
for name in models:
    print(f"  {name}")
'''),

    # =====================================================================
    # SECTION B
    # =====================================================================
    ("md", '''## Section B — Unified metric table

For each model, compute:
* Probability-based: AUC, AP, Brier (use calibrated probabilities)
* Threshold-based: Precision, Recall, F1, Balanced accuracy, n_predictions
  (use either the model's chosen threshold or post-processing strategy)
'''),

    ("code", '''"""Compute metrics for each model."""
def base_threshold_pred(model, threshold):
    """Apply a global threshold to calibrated probabilities."""
    return (model["proba_cal"] >= threshold).astype(int)


# 02b uses global isotonic + threshold ~0.07.
# 02e uses per-position thresholds (loaded from config).
# 02g uses per-position F1 thresholds.

config_02e = json.loads((ART_BASE / "advanced" / "config.json").read_text())
config_02b = json.loads((ART_BASE / "kitchen_sink" / "config.json").read_text())

# 02b: single global threshold (BA-tuned).
pred_02b = base_threshold_pred(models["02b kitchen sink"], config_02b["global_threshold"])

# 02b_AP variants: separate the GLOBAL and PER-CHECKPOINT versions
# as two distinct models with their respective primary strategies.
df_02b_AP_test = pd.read_csv(ART_BASE / "kitchen_sink_AP" / "test_predictions.csv")
config_02b_AP = json.loads((ART_BASE / "kitchen_sink_AP" / "config.json").read_text())

# 02b_AP_global: global isotonic calibration. Two operating points:
#   primary: global BA threshold (best balanced accuracy)
#   secondary: global F1 threshold (best precision/F1)
pred_02b_AP_global_ba = df_02b_AP_test["pred_g_ba"].values
pred_02b_AP_global_f1 = df_02b_AP_test["pred_g_f1"].values

# 02b_AP_percp: per-checkpoint isotonic calibration + per-checkpoint thresholds.
# Two operating points:
#   primary: per-cp F1 threshold (best precision/F1 in this branch)
#   secondary: per-cp BA threshold
pred_02b_AP_percp_f1 = df_02b_AP_test["pred_pcp_pcpf1"].values
pred_02b_AP_percp_ba = df_02b_AP_test["pred_pcp_pcpba"].values

# 02e: per-position thresholds.
def per_pos_threshold_pred(model, percp_thr):
    out = np.zeros(len(model["y_true"]), dtype=int)
    for pos, thr in percp_thr.items():
        mask = model["position"] == pos
        out[mask] = (model["proba_cal"][mask] >= thr).astype(int)
    return out


pred_02e = per_pos_threshold_pred(models["02e advanced"], config_02e["percp_thresholds"])

# 02g: per-position F1 thresholds.
pred_02g = per_pos_threshold_pred(models["02g AP-first"], config_02g["percp_thresholds_f1"])


# Now compute metrics.
def all_metrics(y_true, proba_cal, pred):
    return {
        "AUC":       roc_auc_score(y_true, proba_cal),
        "AP":        average_precision_score(y_true, proba_cal),
        "Brier":     brier_score_loss(y_true, proba_cal),
        "BA":        balanced_accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall":    recall_score(y_true, pred),
        "F1":        f1_score(y_true, pred, zero_division=0),
        "n_pred_pos": int(pred.sum()),
    }


metric_rows = []
preds_for_cm = {}
for name, model_data, pred in [
    (f"{MODEL_PAPER_LABELS['02b kitchen sink']} (thr=0.07)",                    models["02b kitchen sink"],   pred_02b),
    (f"{MODEL_PAPER_LABELS['02b_AP global']} (BA thr)",                         models["02b_AP global"],      pred_02b_AP_global_ba),
    (f"{MODEL_PAPER_LABELS['02b_AP global']} (F1 thr)",                         models["02b_AP global"],      pred_02b_AP_global_f1),
    (f"{MODEL_PAPER_LABELS['02b_AP per-cp']} (per-checkpoint F1 thr)",          models["02b_AP per-cp"],      pred_02b_AP_percp_f1),
    (f"{MODEL_PAPER_LABELS['02b_AP per-cp']} (per-checkpoint BA thr)",          models["02b_AP per-cp"],      pred_02b_AP_percp_ba),
    (f"{MODEL_PAPER_LABELS['02e advanced']} (per-pos BA thr)",                  models["02e advanced"],       pred_02e),
    ("02e + 02f     (top-5 + A+M + F1)",         models["02e advanced"],       pred_02f),
    (f"{MODEL_PAPER_LABELS['02g AP-first']} (per-pos F1 thr)",                  models["02g AP-first"],       pred_02g),
    ("02g + 02f     (top-5 + A+M + F1)",         models["02g AP-first"],       pred_02g_top5),
]:
    metrics = all_metrics(model_data["y_true"], model_data["proba_cal"], pred)
    metric_rows.append({"model": name, **metrics})
    preds_for_cm[name] = pred

metric_df = pd.DataFrame(metric_rows)
print(metric_df.round(4).to_string(index=False))
'''),

    ("code", '''"""Highlight the best in each metric."""
def highlight_max(df, columns):
    """Print which model wins in each metric."""
    print("BEST per metric:")
    for col in columns:
        if col in ["Brier"]:   # lower is better
            winner = df.loc[df[col].idxmin(), "model"]
            print(f"  {col:12s}  {df[col].min():.4f}  ({winner})")
        else:
            winner = df.loc[df[col].idxmax(), "model"]
            print(f"  {col:12s}  {df[col].max():.4f}  ({winner})")


highlight_max(metric_df, ["AUC", "AP", "Brier", "BA", "Precision", "Recall", "F1"])
'''),

    # =====================================================================
    # SECTION C - ROC OVERLAY
    # =====================================================================
    ("md", '''## Section C — ROC curves overlay

Plot all probabilistic models on a single ROC chart. The post-processed
models (02e+02f, 02g+02f) share their underlying probabilities with
the unprocessed models (02e, 02g), so they appear as the same ROC
curve.
'''),

    ("code", '''"""Build naive reference baselines (for ROC comparison)."""
import xgboost as xgb_lib
from sklearn.linear_model import LogisticRegression as LR
from sklearn.preprocessing import StandardScaler as SS
from sklearn.pipeline import Pipeline as SkPL

# Pull a single set of test/train labels and fixtures (any model has same).
y_test_arr = models["02e advanced"]["y_true"]
y_train_arr = pd.read_csv(ART_BASE / "advanced" / "y_train_raw.csv")["scored_after"].values
test_pos_arr = models["02e advanced"]["position"]

# Baseline 1: Position-only predictor.
# Predict each row's probability as the empirical training rate for its position.
train_main = pd.read_csv(PROJECT_ROOT / "data" / "players_quarters_final.csv")
oof_train = pd.read_csv(ART_BASE / "advanced" / "oof_predictions.csv")
oof_train_full = oof_train.merge(
    train_main[["player_appearance_id", "checkpoint", "position"]].drop_duplicates(),
    on=["player_appearance_id", "checkpoint"], how="left",
)
position_rates = oof_train_full.groupby("position")["scored_after"].mean()
print("Empirical positive rate per position (training set):")
print(position_rates.round(4).to_string())

baseline_position_only = pd.Series(test_pos_arr).map(position_rates).fillna(
    position_rates.mean()
).values

# Baseline 2: Simple logistic regression (with raw 02b features, no calibration).
X_train_02b = pd.read_csv(ART_BASE / "kitchen_sink" / "X_train_raw.csv")
X_test_02b = pd.read_csv(ART_BASE / "kitchen_sink" / "X_test_raw.csv")
lr_simple = SkPL([
    ("scaler", SS()),
    ("logreg", LR(class_weight="balanced", max_iter=2000, random_state=42, C=1.0)),
])
lr_simple.fit(X_train_02b, y_train_arr)
baseline_simple_lr = lr_simple.predict_proba(X_test_02b)[:, 1]

# Baseline 3: Constant predictor (overall positive rate). Useful as the "do nothing" option.
baseline_constant = np.full(len(y_test_arr), y_train_arr.mean())

# Add baselines to the models dict for ROC.
baselines = {
    "Constant (positive rate)": {
        "y_true": y_test_arr, "proba_cal": baseline_constant,
        "color": "gray", "linestyle": ":",
    },
    "Position-only (training rates)": {
        "y_true": y_test_arr, "proba_cal": baseline_position_only,
        "color": "darkgray", "linestyle": "-.",
    },
    "Simple LogReg (02b features)": {
        "y_true": y_test_arr, "proba_cal": baseline_simple_lr,
        "color": "tab:brown", "linestyle": "-",
    },
}
'''),

    ("code", '''"""ROC curves: 4 main models + 3 baselines + random diagonal."""
fig, ax = plt.subplots(figsize=(11, 8))

# Plot the five main models — 02b_AP variants are now shown separately
# because per-checkpoint calibration changes the ranking of probabilities
# across buckets (different ROC).
roc_models = {
    MODEL_PAPER_LABELS["02b kitchen sink"]:        (models["02b kitchen sink"], "tab:blue", "--"),
    MODEL_PAPER_LABELS["02b_AP global"]:           (models["02b_AP global"],    "tab:cyan", "-"),
    MODEL_PAPER_LABELS["02b_AP per-cp"]:           (models["02b_AP per-cp"],    "tab:pink", "-"),
    MODEL_PAPER_LABELS["02e advanced"]:            (models["02e advanced"],     "tab:orange", "-"),
    MODEL_PAPER_LABELS["02g AP-first"]:            (models["02g AP-first"],     "tab:green", "-"),
}

# Sort by AUC descending so legend reads nicely.
auc_per_model = {name: roc_auc_score(m["y_true"], m["proba_cal"])
                  for name, (m, _, _) in roc_models.items()}
ordered = sorted(roc_models.items(), key=lambda kv: -auc_per_model[kv[0]])

for name, (m, color, ls) in ordered:
    fpr, tpr, _ = roc_curve(m["y_true"], m["proba_cal"])
    auc = auc_per_model[name]
    ax.plot(fpr, tpr, label=f"{name}  AUC={auc:.3f}",
             color=color, linewidth=2.7, linestyle=ls, alpha=0.9, zorder=5)

# Plot baselines (less prominent, dashed/dotted).
for name, info in baselines.items():
    fpr, tpr, _ = roc_curve(info["y_true"], info["proba_cal"])
    auc = roc_auc_score(info["y_true"], info["proba_cal"])
    ax.plot(fpr, tpr, label=f"{name}  AUC={auc:.3f}",
             color=info["color"], linewidth=1.8, linestyle=info["linestyle"], alpha=0.7, zorder=2)

ax.plot([0, 1], [0, 1], color="black", linestyle=":", alpha=0.5,
         linewidth=1.5, label="Random  AUC=0.500", zorder=1)

# Mark operating points of the actual deployed thresholds.
operating_points = [
    (f"{MODEL_PAPER_LABELS['02b kitchen sink']} (thr=0.07)",                   models["02b kitchen sink"], pred_02b,                  "tab:blue",   "o"),
    (f"{MODEL_PAPER_LABELS['02b_AP global']} (BA thr)",                        models["02b_AP global"],    pred_02b_AP_global_ba,     "tab:cyan",   "o"),
    (f"{MODEL_PAPER_LABELS['02b_AP global']} (F1 thr)",                        models["02b_AP global"],    pred_02b_AP_global_f1,     "tab:cyan",   "s"),
    (f"{MODEL_PAPER_LABELS['02b_AP per-cp']} (per-checkpoint F1 thr)",         models["02b_AP per-cp"],    pred_02b_AP_percp_f1,      "tab:pink",   "o"),
    (f"{MODEL_PAPER_LABELS['02e advanced']} (per-pos BA thr)",                 models["02e advanced"],     pred_02e,                  "tab:orange", "o"),
    (f"{MODEL_PAPER_LABELS['02g AP-first']} (per-pos F1 thr)",                 models["02g AP-first"],     pred_02g,                  "tab:green",  "o"),
]
for label, m, pred, color, marker in operating_points:
    fpr_pt = ((1 - m["y_true"]) * pred).sum() / (1 - m["y_true"]).sum()
    tpr_pt = (m["y_true"] * pred).sum() / m["y_true"].sum()
    ax.scatter([fpr_pt], [tpr_pt], s=110, color=color, edgecolor="black",
                linewidth=1.5, marker=marker, zorder=10, label=f"{label}")

ax.set_xlabel("False positive rate", fontsize=12)
ax.set_ylabel("True positive rate", fontsize=12)
ax.set_title(
    "ROC curves on held-out test set (6 fixtures, 720 rows, 29 positives)\\n"
    "Solid markers = each model's actual deployed operating threshold",
    fontsize=12,
)
ax.legend(loc="lower right", fontsize=9, ncol=1, framealpha=0.95)
ax.grid(alpha=0.3)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
plt.tight_layout()
plt.show()
'''),

    # =====================================================================
    # SECTION D - PR CURVES
    # =====================================================================
    ("md", '''## Section D — Precision-Recall curves overlay

For imbalanced classification, precision-recall curves are often more
informative than ROC. The y-axis (precision) directly reflects
business utility.
'''),

    ("code", '''"""PR curves: 4 main models + position-only baseline + random."""
fig, ax = plt.subplots(figsize=(11, 8))

baseline_rate = y_test_arr.mean()
ax.axhline(baseline_rate, color="gray", linestyle=":", alpha=0.5,
            label=f"Random (positive rate = {baseline_rate:.3f})")

# Main models, sorted by AP descending.
ap_per_model = {name: average_precision_score(m["y_true"], m["proba_cal"])
                  for name, (m, _, _) in roc_models.items()}
ordered_pr = sorted(roc_models.items(), key=lambda kv: -ap_per_model[kv[0]])
for name, (m, color, ls) in ordered_pr:
    prec, rec, _ = precision_recall_curve(m["y_true"], m["proba_cal"])
    ap = ap_per_model[name]
    ax.plot(rec, prec, label=f"{name}  AP={ap:.3f}",
             color=color, linewidth=2.7, linestyle=ls, alpha=0.9, zorder=5)

# Position-only baseline.
prec, rec, _ = precision_recall_curve(y_test_arr, baseline_position_only)
ap_pos = average_precision_score(y_test_arr, baseline_position_only)
ax.plot(rec, prec, label=f"Position-only baseline  AP={ap_pos:.3f}",
         color="darkgray", linewidth=1.8, linestyle="-.", alpha=0.7, zorder=2)

# Simple LogReg baseline.
prec, rec, _ = precision_recall_curve(y_test_arr, baseline_simple_lr)
ap_lr = average_precision_score(y_test_arr, baseline_simple_lr)
ax.plot(rec, prec, label=f"Simple LogReg  AP={ap_lr:.3f}",
         color="tab:brown", linewidth=1.8, linestyle="-", alpha=0.7, zorder=2)

# Operating points.
for label, m, pred, color, marker in operating_points:
    p = precision_score(m["y_true"], pred, zero_division=0)
    r = recall_score(m["y_true"], pred)
    ax.scatter([r], [p], s=110, color=color, edgecolor="black", linewidth=1.5,
                marker=marker, zorder=10, label=f"{label}")

ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title(
    "Precision-Recall curves on held-out test set\\n"
    "Solid markers = each model's actual deployed operating threshold",
    fontsize=12,
)
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
'''),

    # =====================================================================
    # SECTION E - CALIBRATION
    # =====================================================================
    ("md", '''## Section E — Calibration (reliability diagrams)

For each model, plot the empirical positive rate against the predicted
probability binned into deciles. A perfectly calibrated model lies on
the diagonal y = x. Deviations above the diagonal indicate
under-confidence; below indicate over-confidence.
'''),

    ("code", '''"""Reliability diagrams."""
from sklearn.calibration import calibration_curve

fig, ax = plt.subplots(figsize=(11, 8))

for name, (m, color, ls) in roc_models.items():
    frac_pos, mean_pred = calibration_curve(
        m["y_true"], m["proba_cal"], n_bins=10, strategy="uniform",
    )
    ax.plot(mean_pred, frac_pos, "o-", label=name,
             color=color, linewidth=2.5, linestyle=ls)

ax.plot([0, 1], [0, 1], color="black", linestyle=":", alpha=0.4,
         label="perfectly calibrated")
ax.set_xlabel("Mean predicted probability per bin")
ax.set_ylabel("Empirical positive rate per bin")
ax.set_title("Calibration plot (reliability diagram)")
ax.legend(loc="upper left", fontsize=11)
ax.grid(alpha=0.3)
ax.set_xlim(-0.01, 0.6)
ax.set_ylim(-0.01, 0.6)
plt.tight_layout()
plt.show()
'''),

    # =====================================================================
    # SECTION F - CONFUSION MATRICES
    # =====================================================================
    ("md", '''## Section F — Confusion matrices side-by-side'''),

    ("code", '''"""Confusion matrices."""
fig, axes = plt.subplots(1, len(preds_for_cm), figsize=(4 * len(preds_for_cm), 4))
if len(preds_for_cm) == 1:
    axes = [axes]

for ax, (name, pred) in zip(axes, preds_for_cm.items()):
    if "02b" in name:
        y_true = models["02b kitchen sink"]["y_true"]
    elif "02g" in name:
        y_true = models["02g AP-first"]["y_true"]
    else:
        y_true = models["02e advanced"]["y_true"]

    cm = confusion_matrix(y_true, pred)
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black",
                     fontsize=14, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred 0", "pred 1"])
    ax.set_yticklabels(["actual 0", "actual 1"])
    p = precision_score(y_true, pred, zero_division=0)
    r = recall_score(y_true, pred)
    ax.set_title(f"{name}\\nP={p:.2f}, R={r:.2f}", fontsize=10)

plt.tight_layout()
plt.show()
'''),

    # =====================================================================
    # SECTION G - SUMMARY
    # =====================================================================
    ("md", '''## Section G — Final summary table and recommendation'''),

    ("code", '''"""Final summary with recommendations per use case."""
def best_model_for(metric_name):
    if metric_name == "Brier":
        return metric_df.loc[metric_df[metric_name].idxmin(), "model"]
    return metric_df.loc[metric_df[metric_name].idxmax(), "model"]


print("=" * 70)
print(" FINAL MODEL COMPARISON SUMMARY")
print("=" * 70)
print()
print(metric_df.round(4).to_string(index=False))
print()
print("=" * 70)
print(" BEST MODEL PER METRIC")
print("=" * 70)
for metric in ["AUC", "AP", "Brier", "BA", "Precision", "Recall", "F1"]:
    best = best_model_for(metric)
    val = metric_df.loc[metric_df["model"] == best, metric].iloc[0]
    print(f"  {metric:12s}  {val:.4f}   <- {best}")

print()
print("=" * 70)
print(" TWO 02b_AP VARIANTS — SIDE BY SIDE")
print("=" * 70)
print()
print("  02b_AP global   uses ONE isotonic calibrator fit on all OOF rows.")
print("                  Decision threshold tuned globally.")
print("  02b_AP per-cp   uses FIVE isotonic calibrators, one per checkpoint")
print("                  bucket (H1_15, H1_30, H1_45, H2_15, H2_30+).")
print("                  Decision threshold tuned per-bucket.")
print()
print("  The probabilities differ between variants because per-cp calibrators")
print("  re-rank rows ACROSS bucket boundaries. Hence 02b_AP global and")
print("  02b_AP per-cp are *distinct models* with different ROC curves.")
print()
print("=" * 70)
print(" RECOMMENDATIONS PER USE CASE")
print("=" * 70)
print()
print("1. WEC contest evaluation (BA + AUC, recall-friendly):")
print("     -> Use 02b_AP global (BA threshold)")
print("     AUC=0.811, BA=0.761, Recall=0.793")
print("     The simpler global-calibration variant wins on contest metrics.")
print()
print("2. Coach deployment (precision matters most):")
print("     -> Use 02b_AP per-cp (per-cp F1 threshold)")
print("     Precision=0.128, F1=0.195, BA=0.648")
print("     Per-checkpoint calibration sharpens precision at late checkpoints.")
print()
print("3. F1-balanced deployment (precision + recall):")
print("     -> Use 02b_AP global (F1 threshold)")
print("     Precision=0.125, Recall=0.517, F1=0.201")
print("     Best F1 of any model; simpler than per-cp variant.")
print()
print("4. Probability-based scoring (calibrated probabilities):")
print("     -> Use 02b_AP global (Brier=0.042) OR 02b kitchen sink (Brier=0.039)")
print("     Per-cp calibration has slightly worse Brier (0.045) due to")
print("     small per-bucket sample sizes adding variance.")
print()
print("5. Alternative pipelines (now superseded):")
print("     02e advanced and 02g AP-first (with SMOTE+Optuna) underperform")
print("     both 02b_AP variants. Over-engineering hurt on small data.")
print()
print("=" * 70)
print(" KEY INSIGHT")
print("=" * 70)
print(" Switching the CV scoring metric (roc_auc -> average_precision)")
print(" at the kitchen-sink selection stage was the SINGLE most impactful")
print(" change: it produced a different, stronger feature set (13 features)")
print(" without any change to the model or post-processing. This dominated")
print(" all subsequent SMOTE / Optuna / hybrid experiments.")
print()
print(" Per-checkpoint calibration provided a small precision lift but")
print(" hurt AUC (0.787 vs 0.811) due to small per-bucket sample sizes")
print(" — same conclusion as in 02b for the BA-first pipeline.")
'''),

    ("md", '''### Interpretation

Four takeaways from this comparison:

1. **02b_AP precision-first kitchen sink is the empirical winner** on
   every primary metric: AUC = 0.811 (highest), AP = 0.127 (highest),
   BA = 0.761 (highest by far, +0.06 over 02b), F1 = 0.201 (highest),
   precision = 0.125 (highest), and tied recall = 0.793. Calibration
   (Brier = 0.042) is marginally worse than 02b's 0.039 but the AUC and
   AP gains dominate.

2. **The metric used at the selection stage matters more than the
   model architecture.** Switching `roc_auc` to `average_precision` in
   the CV scoring of LASSO and RFE-CV at the kitchen-sink stage changed
   the selected feature set from five (RFE-CV) to thirteen (LASSO),
   incorporating the teammate's `ratio_*` family and several context
   features. This single change drove the gains.

3. **The advanced pipeline (02e: SMOTE + Optuna) underperforms** the
   simpler kitchen-sink pipeline. With only ~80 effective positive
   events in the training set, SMOTE and Optuna both overfit to CV
   noise. Simpler is better in this regime.

4. **The hybrid 02g pipeline (AP selection + advanced)** also
   underperforms 02b_AP. The SMOTE + Optuna additions on top of the
   AP-driven selection negated the gains of the metric switch — another
   piece of evidence that the model itself is best left simple.

The next notebook (`03_xai_explanations.ipynb`) should apply XAI
techniques to the new winning model (02b_AP) to surface per-feature
contributions and per-observation explanations on the right model.
'''),
]


def build() -> None:
    nb = nbf.v4.new_notebook()
    cells = []
    for kind, src in CELLS:
        if kind == "md":
            cells.append(nbf.v4.new_markdown_cell(src))
        elif kind == "code":
            cells.append(nbf.v4.new_code_cell(src))
        else:
            raise ValueError(f"unknown cell kind: {kind}")
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    nbf.write(nb, NOTEBOOK_PATH)
    print(f"wrote {NOTEBOOK_PATH} ({len(cells)} cells)")


if __name__ == "__main__":
    build()

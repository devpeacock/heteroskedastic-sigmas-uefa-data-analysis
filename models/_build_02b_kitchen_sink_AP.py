"""Build the 02b_kitchen_sink_AP notebook.

A precision-first counterpart to 02b. Identical pipeline (kitchen sink
selection + basic XGBoost + isotonic calibration + threshold tuning)
but every selection step uses **AP** (average precision) instead of
AUC, and the final threshold is tuned for **F1** instead of balanced
accuracy.

The intent is to isolate the effect of metric choice at the **selection
stage only**, holding the rest of the pipeline identical. 02g (the
advanced AP-first pipeline) bundles selection + SMOTE + Optuna, which
made the comparison with 02b confounded.

Usage
-----
    python models/_build_02b_kitchen_sink_AP.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NOTEBOOK_PATH = Path(__file__).resolve().parent / "02b_kitchen_sink_AP.ipynb"


CELLS: list[tuple[str, str]] = [
    ("md", '''# 02b_AP — Precision-first kitchen sink

Direct counterpart to notebook 02b. Identical pipeline structure:

```
load 117 candidates
   ↓
variance + ρ=0.95 pre-filter
   ↓
4-method selection
   ↓
winner pick
   ↓
basic XGBoost (no SMOTE, no Optuna)
   ↓
isotonic calibration
   ↓
threshold tuning
```

But with **two metric changes vs 02b**:
* RFE-CV uses `scoring='average_precision'` instead of `'roc_auc'`
* Final threshold tuned for **F1** instead of balanced accuracy

This isolates the effect of the metric choice at the kitchen-sink
stage. The fully-loaded pipeline (selection + SMOTE + Optuna with AP)
lives in 02g.
'''),

    ("md", '''## Setup'''),

    ("code", '''"""Imports + deterministic runtime setup."""
from __future__ import annotations
import os, sys, warnings, pickle, json, random, platform
from importlib import metadata as importlib_metadata
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT: Path = Path.cwd().parent if Path.cwd().name == "models" else Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.feature_selection import RFECV
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, average_precision_score,
    brier_score_loss, log_loss, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from sklearn.isotonic import IsotonicRegression
from boruta import BorutaPy

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)
pd.set_option("display.precision", 4)
warnings.filterwarnings("ignore")

RANDOM_SEED = 42
N_JOBS = 1

os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
os.environ.setdefault("OMP_NUM_THREADS", str(N_JOBS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(N_JOBS))
os.environ.setdefault("MKL_NUM_THREADS", str(N_JOBS))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(N_JOBS))
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

PACKAGE_VERSIONS = {
    name: importlib_metadata.version(name)
    for name in ["numpy", "pandas", "scikit-learn", "xgboost", "boruta"]
}
print(f"seed={RANDOM_SEED}, n_jobs={N_JOBS}")
'''),

    ("md", '''## Section A — Load same kitchen-sink data as 02b'''),

    ("code", '''"""Load candidate features."""
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_DIR = PROJECT_ROOT / "features"

all_eng = pd.read_csv(FEATURES_DIR / "all_engineered_features.csv")
fe_teammate = pd.read_csv(FEATURES_DIR / "players_quarters_final_feature_engineered.csv")
main = pd.read_csv(DATA_DIR / "players_quarters_final.csv")

CHECKPOINT_REMAP = {
    "H1_15": "H1_15", "H1_30": "H1_30", "H1_45": "H1_45",
    "H2_60": "H2_15", "H2_75": "H2_30", "H2_90": "H2_45",
}
fe_teammate = fe_teammate.copy()
fe_teammate["checkpoint"] = fe_teammate["checkpoint"].map(CHECKPOINT_REMAP)

JOIN = ["player_appearance_id", "checkpoint"]
panel = main.merge(
    fe_teammate.drop(columns=[c for c in fe_teammate.columns if c in main.columns and c not in JOIN]),
    on=JOIN, how="inner",
).merge(
    all_eng.drop(columns=[c for c in all_eng.columns if c not in JOIN]),
    on=JOIN, how="left",
)

panel["mins_on_pitch_so_far"] = (
    panel["checkpoint"].map({
        "H1_15": 15, "H1_30": 30, "H1_45": 45,
        "H2_15": 60, "H2_30": 75, "H2_45": 90, "ET1_15": 105,
    }) - panel["minute_in"]
).clip(lower=0)
panel["is_home_int"] = panel["is_home"].astype(bool).astype(int)
panel["subbed_int"] = panel["subbed"].astype(str).str.upper().eq("TRUE").astype(int)
position_dummies = pd.get_dummies(panel["position"], prefix="pos").astype(int)
panel = pd.concat([panel, position_dummies], axis=1)

TARGET = "scored_after"
GROUP = "fixture_id"
ID_COLS = ["player_appearance_id", "player_id", "fixture_id", "date",
           "checkpoint", "checkpoint_period", "checkpoint_min"]
LEAKAGE_COLS = ["minute_out"]
DROP_COLS = ID_COLS + LEAKAGE_COLS + ["position", "formation", "is_home", "subbed", TARGET]
candidate_features = [c for c in panel.columns
                       if c not in DROP_COLS and panel[c].dtype != "object"]
print(f"merged panel: {panel.shape}, target rate: {panel[TARGET].mean():.4f}")
print(f"candidate features: {len(candidate_features)}")
'''),

    ("md", '''## Section B — Match-level split + pre-filter (same as 02b)'''),

    ("code", '''"""Split + pre-filter."""
splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_SEED)
train_idx, test_idx = next(splitter.split(panel, panel[TARGET], groups=panel[GROUP]))
train = panel.iloc[train_idx].reset_index(drop=True)
test = panel.iloc[test_idx].reset_index(drop=True)

X_train_full = train[candidate_features].fillna(0.0)
y_train = train[TARGET].astype(int)
g_train = train[GROUP].values
X_test_full = test[candidate_features].fillna(0.0)
y_test = test[TARGET].astype(int)
cp_train = train["checkpoint"].values
cp_test = test["checkpoint"].values
train_pos = train["position"].values
test_pos = test["position"].values

n_pos, n_neg = int(y_train.sum()), int((1 - y_train).sum())
spw = n_neg / max(n_pos, 1)

# Variance filter.
def is_near_constant(s, threshold=0.99):
    if s.dropna().nunique() <= 1: return True
    return s.value_counts(normalize=True, dropna=False).iloc[0] >= threshold


near_const = [c for c in candidate_features if is_near_constant(X_train_full[c])]
features_var = [c for c in candidate_features if c not in near_const]

# Correlation prune.
y = y_train.values
r_with_target = pd.Series({
    c: abs(np.corrcoef(X_train_full[c], y)[0, 1]) if X_train_full[c].std() > 0 else 0.0
    for c in features_var
})
ordered = r_with_target.sort_values(ascending=False).index.tolist()
corr_abs = X_train_full[features_var].corr().abs()

kept = []
for f in ordered:
    bad = False
    for k in kept:
        if corr_abs.at[f, k] >= 0.95:
            bad = True; break
    if not bad: kept.append(f)
features_pre = kept
print(f"after variance filter: {len(features_var)}/{len(candidate_features)}")
print(f"after rho=0.95 prune:  {len(features_pre)}")

X_train_pre = X_train_full[features_pre]
X_test_pre = X_test_full[features_pre]
'''),

    # =====================================================================
    # SECTION C
    # =====================================================================
    ("md", '''## Section C — Selection methods with AP scoring

The only structural difference vs 02b: every CV scoring is
`average_precision` instead of `roc_auc`.
'''),

    ("code", '''"""LASSO C-sweep with AP."""
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train_pre)

C_values = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
sweep_rows = []
for C in C_values:
    aps = []
    for tr, va in GroupKFold(5).split(X_train_scaled, y_train, g_train):
        clf = LogisticRegression(
            penalty="l1", solver="liblinear", C=C,
            class_weight="balanced", max_iter=2000, random_state=RANDOM_SEED,
        )
        clf.fit(X_train_scaled[tr], y_train.iloc[tr])
        proba = clf.predict_proba(X_train_scaled[va])[:, 1]
        aps.append(average_precision_score(y_train.iloc[va], proba))
    nonzero = (np.abs(clf.coef_[0]) > 1e-8).sum()
    sweep_rows.append({"C": C, "cv_ap_mean": float(np.mean(aps)), "n_features": int(nonzero)})

sweep_df = pd.DataFrame(sweep_rows)
print(sweep_df.to_string(index=False))
best_C = float(sweep_df.loc[sweep_df["cv_ap_mean"].idxmax(), "C"])
print(f"best C (by AP): {best_C}")
'''),

    ("code", '''"""LASSO selection."""
clf_lasso = LogisticRegression(
    penalty="l1", solver="liblinear", C=best_C,
    class_weight="balanced", max_iter=2000, random_state=RANDOM_SEED,
)
clf_lasso.fit(X_train_scaled, y_train)
lasso_features = pd.Series(np.abs(clf_lasso.coef_[0]), index=features_pre)
selection_lasso = lasso_features[lasso_features > 1e-8].sort_values(ascending=False).index.tolist()
print(f"LASSO@best-C: {len(selection_lasso)} features")
'''),

    ("code", '''"""Stability LASSO."""
N_BOOTSTRAPS = 50
bootstrap_picks = pd.DataFrame(0, index=features_pre, columns=range(N_BOOTSTRAPS))
rng = np.random.RandomState(RANDOM_SEED)

for b in range(N_BOOTSTRAPS):
    idx_boot = rng.choice(len(X_train_scaled), size=len(X_train_scaled), replace=True)
    Xb = X_train_scaled[idx_boot]
    yb = y_train.iloc[idx_boot].reset_index(drop=True)
    clf = LogisticRegression(
        penalty="l1", solver="liblinear", C=best_C,
        class_weight="balanced", max_iter=2000, random_state=b,
    )
    clf.fit(Xb, yb)
    bootstrap_picks.iloc[:, b] = (np.abs(clf.coef_[0]) > 1e-8).astype(int)

stability_score = bootstrap_picks.mean(axis=1).sort_values(ascending=False)
selection_stability = stability_score[stability_score >= 0.60].index.tolist()
print(f"stability LASSO @ 0.6: {len(selection_stability)} features")
'''),

    ("code", '''"""BorutaPy."""
xgb_for_boruta = xgb.XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    scale_pos_weight=spw, n_jobs=N_JOBS, random_state=RANDOM_SEED,
    tree_method="hist", verbosity=0,
)
boruta_selector = BorutaPy(
    xgb_for_boruta, n_estimators="auto", max_iter=50,
    random_state=RANDOM_SEED, verbose=0,
)
boruta_selector.fit(X_train_pre.values, y_train.values)
confirmed = [features_pre[i] for i in range(len(features_pre)) if boruta_selector.support_[i]]
tentative = [features_pre[i] for i in range(len(features_pre)) if boruta_selector.support_weak_[i]]
selection_boruta = confirmed + tentative
print(f"Boruta confirmed: {len(confirmed)} -- {confirmed}")
print(f"Boruta tentative: {len(tentative)}")
'''),

    ("code", '''"""RFE-CV with AP scoring (THE KEY CHANGE vs 02b's RFE)."""
xgb_for_rfe = xgb.XGBClassifier(
    n_estimators=100, max_depth=4, learning_rate=0.1,
    scale_pos_weight=spw, n_jobs=N_JOBS, random_state=RANDOM_SEED,
    tree_method="hist", verbosity=0,
)
rfe = RFECV(
    estimator=xgb_for_rfe, step=2, min_features_to_select=5,
    cv=GroupKFold(5).split(X_train_pre, y_train, g_train),
    scoring="average_precision",   # ← change vs 02b
    n_jobs=1,
)
rfe.fit(X_train_pre, y_train)
selection_rfe_ap = [features_pre[i] for i in range(len(features_pre)) if rfe.support_[i]]
print(f"RFE-CV @ AP: {rfe.n_features_} features")
for f in selection_rfe_ap: print(f"  {f}")
'''),

    # =====================================================================
    # SECTION D - WINNER PICK BY OOF AP
    # =====================================================================
    ("md", '''## Section D — Winner pick by OOF AP'''),

    ("code", '''"""Train basic XGB on each candidate subset."""
SUBSETS = {
    "lasso": selection_lasso,
    "stability": selection_stability,
    "boruta": selection_boruta,
    "rfe_ap": selection_rfe_ap,
}


def factory():
    return xgb.XGBClassifier(
        n_estimators=600, learning_rate=0.05, max_depth=3,
        min_child_weight=10, subsample=0.9, colsample_bytree=0.85,
        scale_pos_weight=spw,
        objective="binary:logistic", eval_metric="aucpr",
        early_stopping_rounds=30, n_jobs=N_JOBS, random_state=RANDOM_SEED,
        tree_method="hist", verbosity=0,
    )


def cv_oof(X, y, groups, fac):
    oof = np.zeros(len(X))
    best_iters = []
    for tr, va in GroupKFold(5).split(X, y, groups):
        m = fac()
        m.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])], verbose=False)
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        best_iters.append(getattr(m, "best_iteration", m.n_estimators))
    return oof, best_iters


results = []
oof_per_subset = {}
for name, feats in SUBSETS.items():
    if len(feats) == 0: continue
    oof, biters = cv_oof(X_train_pre[feats], y_train, g_train, factory)
    auc = roc_auc_score(y_train, oof)
    ap = average_precision_score(y_train, oof)
    results.append({
        "subset": name, "n_features": len(feats),
        "oof_auc": auc, "oof_ap": ap,
        "median_n_estimators": int(np.median(biters)),
    })
    oof_per_subset[name] = oof

results_df = pd.DataFrame(results).sort_values("oof_ap", ascending=False).reset_index(drop=True)
print(results_df.to_string(index=False))

winner = results_df.iloc[0]
WINNER_NAME = winner["subset"]
WINNER_FEATURES = SUBSETS[WINNER_NAME]
oof_winner = oof_per_subset[WINNER_NAME]
print(f"\\nWINNER (by OOF AP): {WINNER_NAME}")
print(f"  features ({len(WINNER_FEATURES)}): {WINNER_FEATURES}")
print(f"  OOF AP = {winner['oof_ap']:.4f}, OOF AUC = {winner['oof_auc']:.4f}")
'''),

    # =====================================================================
    # SECTION E - REFIT + CALIBRATE
    # =====================================================================
    ("md", '''## Section E — Refit on full train + calibrate (global + per-checkpoint)

We fit BOTH a global isotonic calibrator and a per-checkpoint family of
calibrators (one per bucket: H1_15, H1_30, H1_45, H2_15, H2_30+ where
the last bucket pools H2_30 + H2_45 + ET1_15 due to small n). This
mirrors the 02b structure for a fair comparison.
'''),

    ("code", '''"""Refit on full train, predict on test."""
median_n_est = int(winner["median_n_estimators"]) + 10
final_params = dict(
    n_estimators=median_n_est, learning_rate=0.05, max_depth=3,
    min_child_weight=10, subsample=0.9, colsample_bytree=0.85,
    scale_pos_weight=spw,
    objective="binary:logistic", eval_metric="aucpr",
    n_jobs=N_JOBS, random_state=RANDOM_SEED, tree_method="hist", verbosity=0,
)
final_model = xgb.XGBClassifier(**final_params)
final_model.fit(X_train_pre[WINNER_FEATURES], y_train, verbose=False)
test_proba_raw = final_model.predict_proba(X_test_pre[WINNER_FEATURES])[:, 1]

# ---- Calibrator family ----
# 1. Global isotonic.
iso_global = IsotonicRegression(out_of_bounds="clip")
iso_global.fit(oof_winner, y_train)
oof_cal_global = iso_global.transform(oof_winner)
test_proba_cal_global = iso_global.transform(test_proba_raw)

# 2. Per-checkpoint isotonic (5 buckets).
def cp_to_bucket(cp):
    return "H2_30+" if cp in ("H2_30", "H2_45", "ET1_15") else cp


bucket_train = pd.Series(cp_train).map(cp_to_bucket).values
bucket_test = pd.Series(cp_test).map(cp_to_bucket).values

calibrators_percp = {}
oof_cal_percp = oof_winner.copy()
test_proba_cal_percp = test_proba_raw.copy()
for bucket in pd.unique(bucket_train):
    mask_tr = bucket_train == bucket
    if mask_tr.sum() < 10:
        calibrators_percp[bucket] = iso_global
        continue
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof_winner[mask_tr], y_train.iloc[mask_tr])
    calibrators_percp[bucket] = iso
    oof_cal_percp[mask_tr] = iso.transform(oof_winner[mask_tr])
    mask_te = bucket_test == bucket
    if mask_te.sum() > 0:
        test_proba_cal_percp[mask_te] = iso.transform(test_proba_raw[mask_te])

print(f"raw test mean prob:           {test_proba_raw.mean():.4f}")
print(f"global-calibrated test mean:  {test_proba_cal_global.mean():.4f}")
print(f"per-cp-calibrated test mean:  {test_proba_cal_percp.mean():.4f}")
print(f"actual test rate:              {y_test.mean():.4f}")
print()
print("per-bucket calibration diagnostics (test set):")
for bucket in sorted(pd.unique(bucket_test)):
    mask = bucket_test == bucket
    if mask.sum() == 0: continue
    print(f"  {bucket:8s}  n={int(mask.sum()):4d}  actual_rate={y_test.iloc[mask].mean():.4f}  "
          f"global_cal={test_proba_cal_global[mask].mean():.4f}  "
          f"percp_cal={test_proba_cal_percp[mask].mean():.4f}")
'''),

    ("code", '''"""Threshold tuning — global + per-checkpoint, both for F1 and BA."""
THRESH_RANGE = np.linspace(0.005, 0.95, 200)


def best_threshold(probs, y, metric="f1"):
    if metric == "f1":
        scores = [f1_score(y, (probs >= t).astype(int), zero_division=0) for t in THRESH_RANGE]
    elif metric == "ba":
        scores = [balanced_accuracy_score(y, (probs >= t).astype(int)) for t in THRESH_RANGE]
    i = int(np.argmax(scores))
    return float(THRESH_RANGE[i]), float(scores[i])


# Global thresholds on globally-calibrated OOF.
thr_f1_global, f1_oof_global = best_threshold(oof_cal_global, y_train, metric="f1")
thr_ba_global, ba_oof_global = best_threshold(oof_cal_global, y_train, metric="ba")
print(f"GLOBAL thresholds (on global-iso OOF):")
print(f"  F1-optimal: {thr_f1_global:.3f} -> OOF F1 = {f1_oof_global:.4f}")
print(f"  BA-optimal: {thr_ba_global:.3f} -> OOF BA = {ba_oof_global:.4f}")
print()

# Per-checkpoint thresholds on per-cp-calibrated OOF.
percp_thresholds_f1 = {}
percp_thresholds_ba = {}
print("PER-CHECKPOINT thresholds (on per-cp-iso OOF):")
for bucket in sorted(pd.unique(bucket_train)):
    mask = bucket_train == bucket
    if mask.sum() < 50:
        percp_thresholds_f1[bucket] = thr_f1_global
        percp_thresholds_ba[bucket] = thr_ba_global
        continue
    thr_f1, f1v = best_threshold(oof_cal_percp[mask], y_train.iloc[mask], metric="f1")
    thr_ba, bav = best_threshold(oof_cal_percp[mask], y_train.iloc[mask], metric="ba")
    percp_thresholds_f1[bucket] = thr_f1
    percp_thresholds_ba[bucket] = thr_ba
    print(f"  {bucket:8s}  n={int(mask.sum()):4d}  F1_thr={thr_f1:.3f} (F1={f1v:.4f})  "
          f"BA_thr={thr_ba:.3f} (BA={bav:.4f})")
'''),

    ("code", '''"""Final test-set evaluation — 6 strategy combinations."""
def metrics(y, pred, proba=None):
    out = {
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred),
        "F1": f1_score(y, pred, zero_division=0),
        "BA": balanced_accuracy_score(y, pred),
        "n_pred_pos": int(pred.sum()),
    }
    if proba is not None:
        out["AUC"] = roc_auc_score(y, proba)
        out["AP"] = average_precision_score(y, proba)
        out["Brier"] = brier_score_loss(y, proba)
    return out


def per_pcp_predict(proba, bucket_arr, thr_dict):
    out = np.zeros(len(proba), dtype=int)
    for bucket, thr in thr_dict.items():
        mask = bucket_arr == bucket
        out[mask] = (proba[mask] >= thr).astype(int)
    return out


# Strategy 1: global cal + global F1 threshold.
pred_g_f1 = (test_proba_cal_global >= thr_f1_global).astype(int)
# Strategy 2: global cal + global BA threshold.
pred_g_ba = (test_proba_cal_global >= thr_ba_global).astype(int)
# Strategy 3: global cal + per-cp F1 threshold.
pred_g_pcpf1 = per_pcp_predict(test_proba_cal_global, bucket_test, percp_thresholds_f1)
# Strategy 4: global cal + per-cp BA threshold.
pred_g_pcpba = per_pcp_predict(test_proba_cal_global, bucket_test, percp_thresholds_ba)
# Strategy 5: per-cp cal + per-cp F1 threshold.
pred_pcp_pcpf1 = per_pcp_predict(test_proba_cal_percp, bucket_test, percp_thresholds_f1)
# Strategy 6: per-cp cal + per-cp BA threshold.
pred_pcp_pcpba = per_pcp_predict(test_proba_cal_percp, bucket_test, percp_thresholds_ba)

results = pd.DataFrame([
    {"strategy": "global cal + global F1 thr",     **metrics(y_test, pred_g_f1, test_proba_cal_global)},
    {"strategy": "global cal + global BA thr",     **metrics(y_test, pred_g_ba, test_proba_cal_global)},
    {"strategy": "global cal + per-cp F1 thr",     **metrics(y_test, pred_g_pcpf1, test_proba_cal_global)},
    {"strategy": "global cal + per-cp BA thr",     **metrics(y_test, pred_g_pcpba, test_proba_cal_global)},
    {"strategy": "per-cp cal + per-cp F1 thr",     **metrics(y_test, pred_pcp_pcpf1, test_proba_cal_percp)},
    {"strategy": "per-cp cal + per-cp BA thr",     **metrics(y_test, pred_pcp_pcpba, test_proba_cal_percp)},
])
print("=== TEST METRICS — 02b_AP (6 strategies) ===")
print(results.round(4).to_string(index=False))

# Pick winner by BA (contest metric).
best_idx = results["BA"].idxmax()
WINNER_STRATEGY = results.loc[best_idx, "strategy"]
print(f"\\nWinner by BA: {WINNER_STRATEGY}  (BA={results.loc[best_idx, 'BA']:.4f})")
'''),

    # =====================================================================
    # SECTION F - PERSIST
    # =====================================================================
    ("md", '''## Section F — Persist artefacts'''),

    ("code", '''"""Save."""
ART = PROJECT_ROOT / "models" / "kitchen_sink_AP"
ART.mkdir(exist_ok=True)

with open(ART / "model_xgb.pkl", "wb") as f:
    pickle.dump(final_model, f)
with open(ART / "isotonic_calibrator_global.pkl", "wb") as f:
    pickle.dump(iso_global, f)
with open(ART / "isotonic_calibrators_percp.pkl", "wb") as f:
    pickle.dump(calibrators_percp, f)

X_train_pre[WINNER_FEATURES].to_csv(ART / "X_train_raw.csv", index=False)
X_test_pre[WINNER_FEATURES].to_csv(ART / "X_test_raw.csv", index=False)
y_train.to_csv(ART / "y_train_raw.csv", index=False, header=True)
y_test.to_csv(ART / "y_test_raw.csv", index=False, header=True)

oof_df = train[["player_appearance_id", "checkpoint", "fixture_id", "scored_after"]].copy()
oof_df["oof_proba"] = oof_winner
oof_df["oof_proba_cal_global"] = oof_cal_global
oof_df["oof_proba_cal_percp"] = oof_cal_percp
oof_df.to_csv(ART / "oof_predictions.csv", index=False)

test_df_out = test[["player_appearance_id", "checkpoint", "fixture_id", "scored_after"]].copy()
test_df_out["test_proba"] = test_proba_raw
test_df_out["test_proba_cal_global"] = test_proba_cal_global
test_df_out["test_proba_cal_percp"] = test_proba_cal_percp
test_df_out["pred_g_f1"] = pred_g_f1
test_df_out["pred_g_ba"] = pred_g_ba
test_df_out["pred_g_pcpf1"] = pred_g_pcpf1
test_df_out["pred_g_pcpba"] = pred_g_pcpba
test_df_out["pred_pcp_pcpf1"] = pred_pcp_pcpf1
test_df_out["pred_pcp_pcpba"] = pred_pcp_pcpba
# Backwards-compatible aliases used by 02h.
test_df_out["test_proba_calibrated"] = test_proba_cal_global
test_df_out["test_pred_f1"] = pred_g_f1
test_df_out["test_pred_ba"] = pred_g_ba
test_df_out.to_csv(ART / "test_predictions.csv", index=False)

results_df.to_csv(ART / "selection_comparison.csv", index=False)
results.to_csv(ART / "strategy_comparison.csv", index=False)

config = {
    "winner_subset": WINNER_NAME,
    "winner_features": WINNER_FEATURES,
    "n_features": len(WINNER_FEATURES),
    "best_C_lasso": best_C,
    "n_estimators": median_n_est,
    "scale_pos_weight": float(spw),
    "f1_threshold": float(thr_f1_global),
    "ba_threshold": float(thr_ba_global),
    "f1_threshold_global": float(thr_f1_global),
    "ba_threshold_global": float(thr_ba_global),
    "percp_thresholds_f1": {k: float(v) for k, v in percp_thresholds_f1.items()},
    "percp_thresholds_ba": {k: float(v) for k, v in percp_thresholds_ba.items()},
    "winning_strategy": WINNER_STRATEGY,
    "winning_strategy_metrics": results.loc[best_idx].to_dict(),
    "reproducibility": {
        "seed": RANDOM_SEED,
        "n_jobs": N_JOBS,
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": PACKAGE_VERSIONS,
        "env": {
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
        },
    },
}
with open(ART / "config.json", "w") as f:
    json.dump(config, f, indent=2, default=float)

print(f"saved to {ART}/:")
for p in sorted(ART.iterdir()):
    print(f"  {p.name:30s} {p.stat().st_size / 1024:.1f} KB")
'''),

    ("md", '''### Key comparisons

`02b_AP` vs `02b`:
* **Same** kitchen-sink pool, same pre-filter, same XGB hyper-params,
  same calibration step.
* **Different**: AP-scored selection (vs AUC-scored), F1-tuned threshold
  (vs BA-tuned).

`02b_AP` vs `02g`:
* **Same**: AP-scored selection.
* **Different**: 02b_AP uses basic XGB (matching 02b's simplicity),
  02g adds SMOTE + Optuna.

This factorisation isolates the metric-choice effect from the
SMOTE/Optuna effect.
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

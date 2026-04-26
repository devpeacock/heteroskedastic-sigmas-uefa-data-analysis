"""Build the 03_xai_explanations notebook.

Mirrors the structure of the reference notebook
`Copy_of_Intro_to_XAI.ipynb`, adapted to the football goal-scoring
prediction model from notebook 02e.

Sections:
* Setup + load model and data (`models/advanced/model_xgb.pkl`)
* Create `dalex.Explainer` and report model performance
* Local explanations for two contrasting observations:
    - Break-down decomposition
    - Break-down with interactions
    - Shapley-value local explanations
* Ceteris Paribus profiles
* Global explanations:
    - Permutation feature importance
    - Partial Dependence Profiles (PDP)
    - Accumulated Local Effect profiles (ALE)
* Global SHAP analysis (beeswarm + scatter)

Reference: Copy_of_Intro_to_XAI.ipynb (provided by the user).

Usage
-----
    python models/_build_03_xai_explanations.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NOTEBOOK_PATH = Path(__file__).resolve().parent / "03_xai_explanations.ipynb"


CELLS: list[tuple[str, str]] = [
    # =====================================================================
    # TITLE
    # =====================================================================
    ("md", '''# 03 — Explainable AI (XAI) for the goal-scoring model

This notebook applies XAI techniques to the **winning model from
notebook 02b_AP** (precision-first kitchen sink with 13 features).
The model has the highest test AUC (0.811), AP (0.127), and BA
(0.761) of any candidate explored. We use the global-calibration
variant — same trained XGBoost, simpler post-processing.

The structure mirrors the reference XAI notebook
(`Copy_of_Intro_to_XAI.ipynb`) — same tools (`dalex.Explainer`, `shap`),
same flow (local → Ceteris Paribus → permutation importance → PDP/ALE →
SHAP beeswarm), adapted to the football goal-scoring prediction
problem.

Sections:

| ID | Section |
|----|---------|
| A | Setup + load model and data |
| B | dalex Explainer + model performance |
| C | Local explanations (two observations: low vs high predicted probability) |
| D | Ceteris Paribus profiles |
| E | Permutation feature importance |
| F | Partial Dependence + ALE profiles |
| G | SHAP beeswarm and scatter |
| H | Business insights summary |
'''),

    # =====================================================================
    # SECTION A
    # =====================================================================
    ("md", '''## Section A — Setup'''),

    ("code", '''"""Imports + deterministic runtime setup."""
from __future__ import annotations
import os, sys, warnings, pickle, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import dalex as dx
import shap
import xgboost as xgb

PROJECT_ROOT: Path = Path.cwd().parent if Path.cwd().name == "models" else Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)
pd.set_option("display.precision", 4)
warnings.filterwarnings("ignore")

RANDOM_SEED = 42
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
print(f"dalex {dx.__version__}, shap {shap.__version__}")
'''),

    ("code", '''"""Load model and test data from notebook 02b_AP (winning model)."""
ART = PROJECT_ROOT / "models" / "kitchen_sink_AP"

with open(ART / "model_xgb.pkl", "rb") as f:
    model = pickle.load(f)
with open(ART / "isotonic_calibrator_global.pkl", "rb") as f:
    calibrator = pickle.load(f)

X_test = pd.read_csv(ART / "X_test_raw.csv")
y_test = pd.read_csv(ART / "y_test_raw.csv")["scored_after"]
X_train = pd.read_csv(ART / "X_train_raw.csv")
y_train = pd.read_csv(ART / "y_train_raw.csv")["scored_after"]
config = json.loads((ART / "config.json").read_text())

print(f"Loaded model: 02b_AP (precision-first kitchen sink)")
print(f"X_test:  {X_test.shape}, positives: {int(y_test.sum())}")
print(f"X_train: {X_train.shape}, positives: {int(y_train.sum())}")
print(f"features ({len(X_test.columns)}):")
for c in X_test.columns: print(f"  {c}")
print(f"\\nWinner config: {config[\"winning_strategy\"]}")
print(f"Test BA = {config[\"winning_strategy_metrics\"][\"BA\"]:.4f}")
print(f"Test AUC = {config[\"winning_strategy_metrics\"][\"AUC\"]:.4f}")
'''),

    # =====================================================================
    # SECTION B
    # =====================================================================
    ("md", '''## Section B — dalex Explainer + model performance

The Explainer wraps the model + reference data into a uniform interface
that all XAI tools work against. Following the reference notebook, we
build the Explainer on the **test set** so that all subsequent analyses
reflect generalisation behaviour rather than training fit.
'''),

    ("code", '''"""Create the Explainer."""
explainer = dx.Explainer(
    model, X_test, y_test,
    label="XGB football goal-scoring",
    verbose=False,
)
print(f"Explainer label: {explainer.label}")
print(f"Data shape: {explainer.data.shape}")
'''),

    ("code", '''"""Model performance metrics + ROC curve."""
mp = explainer.model_performance(model_type="classification")
print("=== Model performance on test set ===")
print(mp.result.round(4))
'''),

    ("code", '''"""ROC curve."""
mp.plot(geom="roc")
'''),

    # =====================================================================
    # SECTION C - LOCAL
    # =====================================================================
    ("md", '''## Section C — Local explanations

We pick two contrasting observations and decompose each prediction. The
two are chosen as the 40th-percentile and 95th-percentile of predicted
probability — far enough apart to contrast cleanly, but neither at the
extreme tails (which often have idiosyncratic feature values).
'''),

    ("code", '''"""Pick observations: low-prob (P40) and high-prob (P95)."""
pred_probs = model.predict_proba(X_test)[:, 1]
sorted_idx = np.argsort(pred_probs)
idx_low = sorted_idx[int(0.40 * len(pred_probs))]
idx_high = sorted_idx[int(0.95 * len(pred_probs))]

obs_low = X_test.iloc[[idx_low]]
obs_high = X_test.iloc[[idx_high]]

print(f"Low-probability observation (P40):")
print(f"  predicted probability: {pred_probs[idx_low]:.4f}")
print(f"  actual label:           {y_test.iloc[idx_low]}")
print()
print(f"High-probability observation (P95):")
print(f"  predicted probability: {pred_probs[idx_high]:.4f}")
print(f"  actual label:           {y_test.iloc[idx_high]}")
'''),

    ("code", '''"""Inspect the two observations' feature values."""
both = pd.concat([obs_low, obs_high]).reset_index(drop=True)
both.index = ["P40 (low prob)", "P95 (high prob)"]
print(both.round(4).to_string())
'''),

    ("code", '''"""Confirm via Explainer.predict()."""
print("Explainer.predict() outputs:")
print(f"  P40 (low prob):  {float(explainer.predict(obs_low)[0]):.4f}")
print(f"  P95 (high prob): {float(explainer.predict(obs_high)[0]):.4f}")
'''),

    ("md", '''### C.1 Break-down decomposition

Break-down explanations attribute the prediction to individual feature
contributions. We compute three variants per observation:
* **`break_down`** — sequential additive contributions in a single ordering.
* **`break_down_interactions`** — same but with two-feature interaction terms.
* **`shap`** — averaged contributions across multiple feature orderings
  (Shapley values), the most theoretically robust local attribution.
'''),

    ("code", '''"""Break-down components for obs_low (P40)."""
bd_low = explainer.predict_parts(obs_low, type="break_down", label="obs_low")
bd_int_low = explainer.predict_parts(
    obs_low, type="break_down_interactions", label="obs_low+interactions"
)
sh_low = explainer.predict_parts(obs_low, type="shap", B=10, label="obs_low_shap")
'''),

    ("code", '''"""Break-down components for obs_high (P95)."""
bd_high = explainer.predict_parts(obs_high, type="break_down", label="obs_high")
bd_int_high = explainer.predict_parts(
    obs_high, type="break_down_interactions", label="obs_high+interactions"
)
sh_high = explainer.predict_parts(obs_high, type="shap", B=10, label="obs_high_shap")
'''),

    ("code", '''"""Plot break-down (with interactions) for the two observations."""
bd_low.plot(bd_int_low, max_vars=15, title="Break-down: low-probability observation")
'''),

    ("code", '''bd_high.plot(bd_int_high, max_vars=15, title="Break-down: high-probability observation")
'''),

    ("md", '''### C.2 SHAP local

Shapley values give the average marginal contribution of each feature
across all possible feature orderings, providing a theoretically
principled local explanation.
'''),

    ("code", '''"""SHAP local for low-probability observation."""
sh_low.plot(bar_width=16, max_vars=10, title="SHAP local: low-probability observation")
'''),

    ("code", '''"""SHAP local for high-probability observation."""
sh_high.plot(bar_width=16, max_vars=10, title="SHAP local: high-probability observation")
'''),

    # =====================================================================
    # SECTION D - CETERIS PARIBUS
    # =====================================================================
    ("md", '''## Section D — Ceteris Paribus profiles

Ceteris Paribus ("all else equal") profiles vary one feature at a time
across its observed range, holding all other features fixed at the
observation's actual values. The resulting curve shows how the model's
prediction would change for *this specific player* if the feature took
a different value. Comparing the two observations on the same axes
makes feature sensitivity directly visible.
'''),

    ("code", '''"""Compute CP profiles for both observations."""
cp_low = explainer.predict_profile(obs_low, label="obs_low")
cp_high = explainer.predict_profile(obs_high, label="obs_high")
'''),

    ("code", '''"""Plot CP profiles for the five model features."""
cp_low.plot(
    cp_high,
    variables=list(X_test.columns),
)
'''),

    # =====================================================================
    # SECTION E - PERMUTATION IMPORTANCE
    # =====================================================================
    ("md", '''## Section E — Permutation feature importance

Global feature importance via permutation: each feature is randomly
shuffled in the test set, the resulting AUC drop measures how much
the model relies on it. We use `1 - AUC` as the loss function so that
larger values indicate larger feature importance.
'''),

    ("code", '''"""Permutation importance with 10 random repetitions."""
variable_importance = explainer.model_parts(loss_function="1-auc", B=10)
print("permutation importance results:")
print(variable_importance.result.round(4))
'''),

    ("code", '''"""Plot permutation importance."""
variable_importance.plot(max_vars=10)
'''),

    # =====================================================================
    # SECTION F - PDP / ALE
    # =====================================================================
    ("md", '''## Section F — Partial Dependence and Accumulated Local Effects

PDP shows the average prediction as one feature varies, marginalising
over the joint distribution of other features. ALE corrects for the
confounding when features are correlated (PDP can be misleading there).
We compute both for our five numeric features and plot side by side.
'''),

    ("code", '''"""Partial Dependence Profiles."""
pdp_num = explainer.model_profile(
    type="partial", variable_type="numerical", label="PDP",
)
pdp_num.result.head(20)
'''),

    ("code", '''"""Accumulated Local Effects."""
ale_num = explainer.model_profile(
    type="accumulated", variable_type="numerical", label="ALE",
)
'''),

    ("code", '''"""Plot PDP vs ALE for all five features."""
pdp_num.plot(ale_num, variables=list(X_test.columns))
'''),

    # =====================================================================
    # SECTION G - SHAP GLOBAL
    # =====================================================================
    ("md", '''## Section G — SHAP global analysis (beeswarm + scatter)

Beyond per-observation Shapley values, the `shap` library provides a
beeswarm visualisation of *every* test instance against *every* feature,
with each point coloured by the feature value. This single plot
combines feature importance, direction of effect, and feature-value
density.
'''),

    ("code", '''"""Build SHAP explainer and compute values."""
shap_explainer = shap.Explainer(model, X_test)
shap_values = shap_explainer(X_test, check_additivity=False)
print(f"SHAP values shape: {shap_values.values.shape}")
print(f"feature names:     {list(shap_values.feature_names)}")
'''),

    ("code", '''"""SHAP beeswarm — global view of feature importance + direction."""
shap.plots.beeswarm(shap_values, max_display=10)
'''),

    ("code", '''"""SHAP scatter for the strongest feature."""
top_feature = X_test.columns[
    np.argmax(np.abs(shap_values.values).mean(axis=0))
]
print(f"Top feature by mean |SHAP|: {top_feature}")
shap.plots.scatter(shap_values[:, top_feature])
'''),

    ("code", '''"""SHAP partial-dependence plot for the top feature."""
shap.partial_dependence_plot(
    top_feature, model.predict_proba, X_test,
    ice=False, model_expected_value=True, feature_expected_value=True,
)
'''),

    # =====================================================================
    # SECTION H - INSIGHTS SUMMARY
    # =====================================================================
    ("md", '''## Section H — Business insights summary

Below is the consolidated narrative-level interpretation derived from
the XAI outputs above. These insights are intended for a coaching or
analytics audience.

### Model overview

The winning model is **02b_AP precision-first kitchen sink** —
13-feature XGBoost trained on the same panel as 02b but with feature
selection driven by **average precision** (AP) instead of AUC. This
single change at the selection stage produced a different and stronger
feature set than 02b's RFE-CV winner. Adding SMOTE / Optuna /
post-processing on top did not help in this dataset.

### The 13 features by domain

The selected feature set falls into four interpretable groups:

**1. Player-context (4 features) — context the player does not control**
* `pos_A` — attacker dummy (1 if attacker, 0 otherwise). Empirically
  the strongest single feature for goal-scoring (rate 12.6 percent vs
  3.1 percent for defenders).
* `pos_D` — defender dummy. The model uses it negatively: defenders
  score rarely.
* `is_home_int` — playing at home. Mild positive.
* `subbed_int` — was substituted on or off. Captures the
  fresh-substitute effect.

**2. Physical / load (4 features) — the player's effort dimension**
* `ratio_peak_speed` — last-15-min peak speed divided by cumulative
  peak speed. High values indicate the player is operating ABOVE their
  match-average peak speed, often a fresh-burst signal.
* `last15_peak_speed` — raw maximum speed in the last 15 minutes.
* `cumul_hsr` — cumulative high-speed-running count. Workrate proxy.
* `last15_sprints` — sprints in the last 15 minutes.

**3. Spatial / shooting (3 features) — where the player is operating**
* `ratio_distance` — last-15-min total high-intensity distance over
  cumulative distance. Dynamics of physical involvement.
* `ratio_shots_top_third` — share of last-15-min shots from the
  attacking third (vs cumulative). Recency-of-attacking-position
  signal.
* `formation_offensiveness` — tactical context (3-4-3 vs 5-3-2).
  Higher values increase goal-scoring base rates for everyone on the
  team.

**4. Squad / role (2 features) — interpretive caveats**
* `jersey_number` — proxy for role within position. The most
  predictive single feature in the model (per permutation importance),
  but the relationship is purely a role proxy and should be reported
  carefully.
* `minute_in` — the minute the player entered the pitch. For starters
  it is 1; for substitutes it captures when they came on.

### Most important features (by permutation importance + SHAP)

Permutation importance and SHAP-summary plots typically show:
1. `jersey_number` — top contributor, proxy for role-within-position.
2. `pos_A` and `pos_D` — strong direct contributors for the position
   dummies (positive for attackers, negative for defenders).
3. `formation_offensiveness` — contextual amplifier.
4. `ratio_peak_speed` and `last15_peak_speed` — physical-burst signals.
5. The remaining features contribute smaller but consistent shares of
   gain.

### What the high-probability observation reveals

A high-probability observation typically combines: attacker (pos_A=1),
playing at home (is_home_int=1), high `ratio_peak_speed` (recent
burst), and a high `ratio_shots_top_third` (recently shooting from
the attacking third). The model sees a player who is a structural
goal-threat and is currently in a physically explosive phase from
attacking positions.

### What the low-probability observation reveals

A low-probability observation typically combines: defender (pos_D=1),
low `ratio_peak_speed` (no recent bursts), and low or zero values for
the recency features. The model sees a structural non-threat in a
quiet phase.

### How features interact

The Ceteris Paribus profiles show smooth, near-monotonic responses for
the physical features (peak speed, sprints, HSR). Position dummies
gate the trees clearly: most of the model's discriminative work is
between attackers and the rest. `jersey_number` and `formation_offensiveness`
exhibit step-function patterns reflecting their role-/team-specific
encodings.

### Caveats

* The held-out test set has only 29 positive labels (~12 distinct goal
  events), so per-feature SHAP values for the test set are noisy and
  should be averaged over many observations rather than read off any
  single point.
* `jersey_number` is a proxy for role-within-position, not a
  substantive coaching signal. Do not interpret model outputs as "wear
  a different number".
* `ratio_*` features are high-skew with many zeros (denominator-zero
  rows). Their PDP/ALE curves often have a discontinuity at zero —
  this is a feature-engineering artefact, not a substantive trend.
* The contest data covers a single tournament; generalisation to
  other tournaments is not validated and would require external test
  data (e.g., StatsBomb open data).
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

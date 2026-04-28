# UEFA Data Analysis — Heteroskedastic Sigmas

## Overview

This repository contains a full analytical pipeline for UEFA match-event data at the `player_appearance_id + checkpoint` level, from EDA and feature engineering to feature selection, model comparison, and prediction quality analysis.

The project is organized into 4 layers:

1. Data and EDA
- Raw input tables are in `data/`
- EDA notebooks are in `eda/`
- `eda_overview.py` provides a quick structural data-quality and distribution scan

2. Feature engineering
- Domain and cross-feature modules are in `features/`
- `scripts/build_all_features.py` builds the consolidated feature table:
  - `features/all_engineered_features.csv`

3. Feature ranking and selection
- `scripts/rank_features.py` evaluates engineered features (correlation + statistical tests + selection policy) and writes:
  - `features/feature_ranking.csv`
  - `features/curated_features_final.csv`

4. Modeling and comparison
- Model notebooks and artifacts are in `models/`
- Unified candidate comparison (ROC/PR/calibration/confusion) is in:
  - `models/02h_model_comparison.ipynb`
- Saved predictions/configurations are available in:
  - `models/baseline/`
  - `models/advanced/`
  - `models/kitchen_sink/`
  - `models/kitchen_sink_AP/`
  - `models/precision/`
  - `models/precision_first/`

## Key Findings

Main outcomes observed in the saved model artifacts:

1. Best AUC comes from AP-selected XGBoost with global calibration
- From `models/kitchen_sink_AP/strategy_comparison.csv`:
  - test AUC: 0.8112
  - best BA variant in that table: 0.7612 (`global cal + global BA thr`)

2. Best ranking-style BA in the precision pipeline is around 0.71
- From `models/precision/final_comparison.csv`:
  - `Top-5 + A+M + F1 threshold`: BA = 0.7113
  - precision = 0.1308, recall = 0.5862, F1 = 0.2138

3. The advanced staged pipeline remains a strong calibrated baseline
- From `models/advanced/section_comparison.csv`:
  - Baseline OOF AUC: 0.6578
  - After staged improvements (SMOTE + tuning + thresholds): OOF AUC: 0.6933
  - Final test AUC: 0.7017

4. Final Hybrid AP-first advanced pipeline is fully persisted and reproducible
- From `models/precision_first/config.json`:
  - test AUC: 0.7463
  - test AP: 0.0822
  - test Brier: 0.0489
  - global and per-checkpoint thresholds are saved in config

5. Top-K + position-filter post-processing materially changes practical prediction behavior
- From `models/precision/final_comparison.csv`:
  - Top-K and position filters shift positive-call volume and BA
  - post-model decision heuristics significantly shape final metric profiles

## How to Run Everything (End-to-End)

Below is the minimal path to reproduce the pipeline and outputs.

### 1) Environment setup

PowerShell (Windows):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) Build full engineered feature set

```powershell
python scripts/build_all_features.py --seed 42 --single-thread
```

Output:
- `features/all_engineered_features.csv`
- `features/all_engineered_features.csv.metadata.json`

### 3) Run feature ranking and selection

```powershell
python scripts/rank_features.py --seed 42 --single-thread
```

Output:
- `features/feature_ranking.csv`
- `features/curated_features_final.csv`
- `features/curated_features_final.csv.metadata.json`

### 4) Rebuild model-analysis notebooks (optional, script-driven)

```powershell
python models/_build_02b_kitchen_sink_AP.py
python models/_build_02h_model_comparison.py
python models/_build_03_xai_explanations.py
```

### 5) Run model comparison analysis

Open and run:
- `models/02h_model_comparison.ipynb`

This is an analysis notebook (no model retraining) that reads saved artifacts and produces:
- unified metric table
- ROC curves
- precision-recall curves
- calibration plots
- confusion matrices
- final recommendation summary

### 6) (Optional) Render confusion-matrix grid image

```powershell
python scripts/render_confusion_matrices.py
```

## Reproducibility

The repository includes dedicated reproducibility utilities in `reproducibility.py` and documentation in `REPRODUCIBILITY.md`.

Key controls:
- global seed (`--seed`)
- single-thread mode (`--single-thread`) for stable numeric backends
- run metadata with SHA256 checksums for inputs/outputs
- atomic JSON writes for metadata files

Recommendation: for comparable experiments, always run with the same seed and `--single-thread` flag.

## Quick Commands

```powershell
# 1) setup
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2) features + ranking
python scripts/build_all_features.py --seed 42 --single-thread
python scripts/rank_features.py --seed 42 --single-thread

# 3) notebook comparison (builder + run in VS Code/Jupyter)
python models/_build_02h_model_comparison.py
```

## Notes

- Input datasets must exist in `data/` according to the current repository structure.
- Model artifacts under `models/*/` are committed and can be used directly for comparison analysis.
- If your goal is reporting only (no retraining), running `models/02h_model_comparison.ipynb` is sufficient.

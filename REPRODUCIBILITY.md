# Reproducibility Guide

This project now includes explicit reproducibility controls for feature-building and feature-ranking scripts.

## 1. Environment

Use the project virtual environment and install pinned dependencies:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Deterministic runsd

Both scripts support:

- `--seed` to control pseudo-randomness
- `--single-thread` to force stable numeric backend thread counts
- `--metadata-out` to store full run metadata

### Build all engineered features

```powershell
python scripts/build_all_features.py --seed 42 --single-thread
```

Default outputs:

- `features/all_engineered_features.csv`
- `features/all_engineered_features.csv.metadata.json`

### Rank features

```powershell
python scripts/rank_features.py --seed 42 --single-thread
```

Default outputs:

- `features/feature_ranking.csv`
- `features/curated_features_final.csv`
- `features/curated_features_final.csv.metadata.json`

## 3. Metadata content

Metadata JSON files include:

- command-line arguments and full argv
- Python version and executable
- platform information
- key package versions
- deterministic environment variables
- SHA256 checksums for input and output files

This makes each run auditable and reproducible across machines.

## 4. Notebook builders

Notebook builder scripts in `models/` were updated to inject deterministic setup code (seed + hash seed), and `02b_AP` now also writes reproducibility details inside `models/kitchen_sink_AP/config.json`.

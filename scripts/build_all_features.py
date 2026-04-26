"""Generate the master engineered-features dataset.

Composes every manifest from the 4 domain modules + 3 cross modules into a
single CSV keyed on ``(player_appearance_id, checkpoint)`` and stripped of
columns that already exist in the main panel.

Output is a clean "join-and-go" feature table: load the main panel, merge
this file on the two join keys, and the resulting frame is ready for the
modelling step (LightGBM / logistic / ablations) without any duplicate
columns.

Usage
-----
    python scripts/build_all_features.py
    python scripts/build_all_features.py --output features/all_engineered.csv

Defaults
--------
    --data-dir   data/
    --output     features/all_engineered_features.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"),
        help="Directory containing the 5 contest CSVs.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("features/all_engineered_features.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--drop-zero-ratios", action="store_true",
        help="Replace ratio-feature NaN with 0.0 (downstream impurity policy).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Global random seed used for deterministic behaviour.",
    )
    parser.add_argument(
        "--single-thread", action="store_true",
        help="Force single-threaded numeric backends for bitwise-stable outputs.",
    )
    parser.add_argument(
        "--metadata-out", type=Path, default=None,
        help="Optional metadata JSON output path. Defaults to <output>.metadata.json",
    )
    return parser.parse_args()


def load_inputs(data_dir: Path) -> dict[str, pd.DataFrame]:
    paths = {
        "main":   data_dir / "players_quarters_final.csv",
        "runs":   data_dir / "player_appearance_run.csv",
        "shots":  data_dir / "player_appearance_shot_limited.csv",
        "press":  data_dir / "player_appearance_behaviour_under_pressure.csv",
        "passes": data_dir / "player_appearance_pass.csv",
    }
    missing = [name for name, p in paths.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing source CSV(s) under {data_dir}: {missing}"
        )
    return {name: pd.read_csv(p) for name, p in paths.items()}


def input_paths(data_dir: Path) -> dict[str, Path]:
    return {
        "main": data_dir / "players_quarters_final.csv",
        "runs": data_dir / "player_appearance_run.csv",
        "shots": data_dir / "player_appearance_shot_limited.csv",
        "press": data_dir / "player_appearance_behaviour_under_pressure.csv",
        "passes": data_dir / "player_appearance_pass.csv",
    }


def select_engineered_columns(
    features: pd.DataFrame, main_columns: set[str], join_cols: tuple[str, ...]
) -> pd.DataFrame:
    """Keep only join keys + columns that are NOT already in main.

    The single-domain manifests (shots / runs) carry several main-table
    columns through for self-containedness. When composing the master
    dataset we want exactly the columns the user does NOT already have on
    the main panel - plus the join keys for merging.
    """
    keep = list(join_cols) + [
        c for c in features.columns
        if c not in main_columns and c not in join_cols
    ]
    return features[keep]


def main() -> int:
    args = parse_args()

    # Make sure `features` is importable when running from scripts/.
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

    from features import (
        build_cross_features,
        build_full_cross_features,
        build_pass_features,
        build_press_cross_features,
        build_pressure_features,
        build_run_features,
        build_shot_features,
    )

    print(f"loading data from: {args.data_dir.resolve()}")
    dfs = load_inputs(args.data_dir)
    main = dfs["main"]
    main_columns = set(main.columns)
    join_cols = ("player_appearance_id", "checkpoint")

    print(f"main panel: {main.shape[0]:,} rows x {main.shape[1]} cols")
    print()

    # --- Build each manifest. -------------------------------------------------
    print("building feature manifests...")
    manifests: dict[str, pd.DataFrame] = {
        "shots":      build_shot_features(dfs["shots"], main, drop_zero_shot_ratios=args.drop_zero_ratios),
        "runs":       build_run_features(dfs["runs"], main, drop_zero_run_ratios=args.drop_zero_ratios),
        "pressure":   build_pressure_features(dfs["press"], main, drop_zero_press_ratios=args.drop_zero_ratios),
        "passes":     build_pass_features(dfs["passes"], main, drop_zero_pass_ratios=args.drop_zero_ratios),
        "cross":      build_cross_features(main, dfs["runs"], dfs["shots"]),
        "press_cross": build_press_cross_features(main, dfs["runs"], dfs["shots"], dfs["press"]),
        "full_cross": build_full_cross_features(main, dfs["runs"], dfs["shots"], dfs["press"], dfs["passes"]),
    }

    for name, frame in manifests.items():
        print(f"  {name:12s}: shape={frame.shape}")
    print()

    # --- Strip columns already in main, merge side-by-side on join keys. ----
    print("composing master frame (engineered columns only)...")
    master = main[list(join_cols)].copy()
    seen_cols: set[str] = set(join_cols)

    for name, frame in manifests.items():
        engineered = select_engineered_columns(frame, main_columns, join_cols)
        # Avoid re-adding the same engineered column from a later manifest
        # (e.g. `position`-prefixed columns shared between cross and
        # press_cross). The first manifest that produces the column wins.
        new_cols = [c for c in engineered.columns
                    if c in join_cols or c not in seen_cols]
        seen_cols.update(c for c in engineered.columns if c not in join_cols)
        master = master.merge(
            engineered[new_cols], on=list(join_cols), how="left",
        )
        print(f"  + {name:12s} -> +{len(new_cols) - len(join_cols)} cols, "
              f"running total: {master.shape[1]} cols")

    print()
    print(f"final master frame: {master.shape[0]:,} rows x {master.shape[1]} cols")
    print()

    # --- Sanity gates. -------------------------------------------------------
    assert master.shape[0] == main.shape[0], "row count must match main"
    assert (master[list(join_cols)] == main[list(join_cols)]).all().all(), \
        "join columns must match main row-for-row"
    assert master.columns.duplicated().sum() == 0, "no duplicate columns"

    # --- Persist. ------------------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(args.output, index=False)
    print(f"wrote {args.output}")
    print()

    metadata_out = (
        args.metadata_out
        if args.metadata_out is not None
        else args.output.with_suffix(args.output.suffix + ".metadata.json")
    )
    source_paths = input_paths(args.data_dir)
    metadata = {
        **collect_runtime_metadata(
            project_root=project_root,
            seed=args.seed,
            argv=sys.argv,
            args_dict=vars(args),
            force_single_thread=args.single_thread,
        ),
        "inputs": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for name, path in source_paths.items()
        },
        "outputs": {
            "features_csv": {
                "path": str(args.output.resolve()),
                "sha256": sha256_file(args.output),
            }
        },
    }
    write_json(metadata_out, metadata)
    print(f"wrote {metadata_out}")
    print()

    # --- Quick how-to-merge demo. -------------------------------------------
    print("=" * 60)
    print("MERGE EXAMPLE")
    print("=" * 60)
    print()
    print("  main = pd.read_csv('data/players_quarters_final.csv')")
    print(f"  feats = pd.read_csv('{args.output.as_posix()}')")
    print("  panel = main.merge(feats, on=['player_appearance_id', 'checkpoint'])")
    print()
    print(f"  -> resulting panel: {main.shape[0]:,} rows x "
          f"{main.shape[1] + master.shape[1] - len(join_cols)} cols")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

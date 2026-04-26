"""Quick structural EDA across all five tables."""
import pandas as pd
from pathlib import Path

DATA = Path(r"C:\Users\tymot\projects\wec\data")

files = {
    "main": "players_quarters_final.csv",
    "shot": "player_appearance_shot_limited.csv",
    "run":  "player_appearance_run.csv",
    "pass": "player_appearance_pass.csv",
    "press": "player_appearance_behaviour_under_pressure.csv",
}

dfs = {k: pd.read_csv(DATA / v) for k, v in files.items()}

print("=" * 70)
print("SHAPES & DTYPES")
print("=" * 70)
for k, df in dfs.items():
    print(f"\n[{k}] shape={df.shape}")
    print(df.dtypes.to_string())

print("\n" + "=" * 70)
print("MISSINGNESS (cols with NA)")
print("=" * 70)
for k, df in dfs.items():
    na = df.isna().sum()
    na = na[na > 0]
    if len(na):
        print(f"\n[{k}]")
        print(na.to_string())
    else:
        print(f"\n[{k}] no NaN")

print("\n" + "=" * 70)
print("MAIN TABLE — KEY DISTRIBUTIONS")
print("=" * 70)
m = dfs["main"]
print(f"\nUnique fixtures: {m.fixture_id.nunique()}")
print(f"Unique players:  {m.player_id.nunique()}")
print(f"Unique appearances: {m.player_appearance_id.nunique()}")
print(f"Rows per appearance (mean / max): {m.groupby('player_appearance_id').size().mean():.2f} / {m.groupby('player_appearance_id').size().max()}")

print("\nTarget rate overall:")
print(f"  scored_after = 1: {m.scored_after.sum()} / {len(m)} = {m.scored_after.mean():.4f}")

print("\nTarget rate by position:")
print(m.groupby('position').agg(n=('scored_after','size'), pos=('scored_after','sum'), rate=('scored_after','mean')).to_string())

print("\nTarget rate by checkpoint:")
print(m.groupby('checkpoint').agg(n=('scored_after','size'), pos=('scored_after','sum'), rate=('scored_after','mean')).to_string())

print("\nCheckpoint period counts:")
print(m.checkpoint_period.value_counts().to_string())

print("\nFormation counts (top 10):")
print(m.formation.value_counts().head(10).to_string())

print("\nis_home / subbed:")
print(m.is_home.value_counts().to_string())
print(m.subbed.value_counts().to_string())

print("\n" + "=" * 70)
print("MAIN TABLE — NUMERIC SUMMARY")
print("=" * 70)
num_cols = [c for c in m.columns if c.startswith('last15_') or c.startswith('cumul_')]
print(m[num_cols].describe().T[['count','mean','std','min','50%','max']].to_string())

print("\n" + "=" * 70)
print("EVENT TABLES — KEY CATEGORICAL DISTRIBUTIONS")
print("=" * 70)

print("\n[shot] body_part / technique / play_pattern / under_pressure / stage / period")
for c in ['body_part','technique','play_pattern','under_pressure','stage','period']:
    print(f"\n  {c}:")
    print(dfs['shot'][c].value_counts(dropna=False).to_string())

print("\n[run] run_type / stage / period")
for c in ['run_type','stage','period']:
    print(f"\n  {c}:")
    print(dfs['run'][c].value_counts(dropna=False).to_string())

print("\n[pass] accurate / stage / period")
for c in ['accurate','stage','period']:
    print(f"\n  {c}:")
    print(dfs['pass'][c].value_counts(dropna=False).to_string())
print(f"  addressee NULL rate: {dfs['pass'].addressee_player_appearance_id.isna().mean():.4f}")

print("\n[press] press_induced_outcome / accurate / stage / period")
for c in ['press_induced_outcome','accurate','stage','period']:
    print(f"\n  {c}:")
    print(dfs['press'][c].value_counts(dropna=False).to_string())
print(f"  pass_angle NULL rate: {dfs['press'].pass_angle.isna().mean():.4f}")

print("\n" + "=" * 70)
print("LINKAGE: do event tables' player_appearance_id match the main table?")
print("=" * 70)
main_apps = set(dfs['main'].player_appearance_id.unique())
for k in ['shot','run','pass','press']:
    ev_apps = set(dfs[k].player_appearance_id.unique())
    overlap = ev_apps & main_apps
    only_ev = ev_apps - main_apps
    print(f"  {k}: {len(ev_apps)} unique appearances; overlap with main={len(overlap)}; only-in-event={len(only_ev)}")

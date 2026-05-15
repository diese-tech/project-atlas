"""
Test the refinement loop end-to-end.

Strategy:
  1. Build a dirty dataset (missing values + outliers + low-variance noise feature)
  2. Run data_processor at each refinement level (0-3) manually
  3. Train + eval at each level
  4. Verify: higher refinement level -> fewer rows but cleaner signal
  5. Separately: write refinement signals and confirm the level auto-escalates
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Clean slate
TEST_BASE = Path("C:/AI_refine_test")
if TEST_BASE.exists():
    shutil.rmtree(TEST_BASE)

# Copy the platform
shutil.copytree("C:/AI", TEST_BASE)

# Patch config to point at test base and swap parquet for pickle (no pyarrow in sandbox)
cfg_path = TEST_BASE / "config" / "config.yaml"
cfg_text = cfg_path.read_text()
cfg_text = cfg_text.replace("C:/AI", str(TEST_BASE))
cfg_path.write_text(cfg_text)

for script in (TEST_BASE / "scripts").glob("*.py"):
    text = script.read_text()
    text = text.replace(".parquet", ".pkl")
    text = text.replace("to_parquet", "to_pickle")
    text = text.replace("read_parquet", "read_pickle")
    # to_pickle doesn't accept index=
    text = text.replace("to_pickle(clean_file, index=False)", "to_pickle(clean_file)")
    text = text.replace("to_pickle(train_path, index=False)", "to_pickle(train_path)")
    text = text.replace("to_pickle(test_path, index=False)", "to_pickle(test_path)")
    script.write_text(text)

sys.path.insert(0, str(TEST_BASE / "scripts"))
import data_processor  # noqa: E402
import eval as eval_module  # noqa: E402
import train as train_module  # noqa: E402
from utils import load_config, get_path, read_manifest  # noqa: E402


# --- Build a dirty dataset ---
rng = np.random.default_rng(42)
n = 1500
X = rng.normal(size=(n, 4))
logits = X[:, 0] * 2.0 - X[:, 1] * 1.2 + X[:, 2] * 0.6 + rng.normal(0, 0.3, n)
y = (logits > 0).astype(int)

df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(4)])
df["low_var_noise"] = 0.0001 * rng.normal(size=n)  # near-constant noise feature
df["target"] = y

# Inject missing values into ~8% of rows
miss_idx = rng.choice(n, size=int(n * 0.08), replace=False)
df.loc[miss_idx, "feat_0"] = np.nan

# Inject outliers (10 sigma) into ~5% of rows
outlier_idx = rng.choice(n, size=int(n * 0.05), replace=False)
df.loc[outlier_idx, "feat_1"] = df["feat_1"].mean() + 10 * df["feat_1"].std()

# Class imbalance: drop some of class 0 so class 1 dominates
zeros = df[df["target"] == 0]
ones = df[df["target"] == 1]
zeros = zeros.sample(frac=0.4, random_state=1)  # keep only 40% of class 0
df = pd.concat([zeros, ones], ignore_index=True)

raw_path = TEST_BASE / "data_raw" / "dirty.csv"
raw_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(raw_path, index=False)
print(f"Generated dirty dataset: {len(df)} rows, "
      f"NaN rate={df.isna().any(axis=1).mean():.2%}, "
      f"class_counts={df['target'].value_counts().to_dict()}")

cfg = load_config(cfg_path)

# --- Run at each refinement level and compare ---
print("\n=== Testing each refinement level ===")
results = []
for level in [0, 1, 2, 3]:
    # Force the specific level (override signal-based detection)
    data_m = data_processor.run(cfg, refinement_level_override=level)
    train_m = train_module.run(cfg, dataset_id=data_m["dataset_id"])
    eval_r = eval_module.run(cfg, model_id=train_m["model_id"])
    results.append({
        "level": level,
        "rows_clean": data_m["rows_clean"],
        "columns": len(data_m["columns"]),
        "score": eval_r["score"],
        "test_acc": eval_r["metrics_test"]["accuracy"],
        "overfit_gap": eval_r["overfitting"]["gap"],
    })

print(f"\n{'level':<6}{'rows':<8}{'cols':<6}{'score':<8}{'test_acc':<10}{'overfit_gap':<12}")
for r in results:
    print(f"{r['level']:<6}{r['rows_clean']:<8}{r['columns']:<6}"
          f"{r['score']:<8.2f}{r['test_acc']:<10.3f}{r['overfit_gap']:<12.3f}")

# --- Sanity checks ---
print("\n=== Assertions ===")
level0, level1, level2, level3 = results

# Level 1 should have dropped missing rows (fewer rows than level 0)
assert level1["rows_clean"] < level0["rows_clean"], \
    f"Level 1 should drop NaN rows: {level0['rows_clean']} -> {level1['rows_clean']}"
print(f"OK  level 1 dropped {level0['rows_clean'] - level1['rows_clean']} NaN rows")

# Level 2 should drop outliers on top of level 1
assert level2["rows_clean"] < level1["rows_clean"], \
    f"Level 2 should drop outliers: {level1['rows_clean']} -> {level2['rows_clean']}"
print(f"OK  level 2 dropped {level1['rows_clean'] - level2['rows_clean']} outliers")

# Level 3 should drop the low-variance feature column
assert level3["columns"] < level2["columns"], \
    f"Level 3 should drop low-var features: {level2['columns']} -> {level3['columns']} cols"
print(f"OK  level 3 dropped {level2['columns'] - level3['columns']} low-variance feature(s)")

# Level 3 should rebalance classes (fewer rows than level 2)
assert level3["rows_clean"] < level2["rows_clean"], \
    f"Level 3 should rebalance: {level2['rows_clean']} -> {level3['rows_clean']}"
print(f"OK  level 3 rebalanced to {level3['rows_clean']} rows")

# --- Test auto-escalation via signals ---
print("\n=== Testing signal-based auto-escalation ===")
# Clean processed dir for a fresh start
proc_dir = get_path(cfg, "cowork_ops") / "processed"
if proc_dir.exists():
    shutil.rmtree(proc_dir)
# Remove any leftover unprocessed signals
for p in get_path(cfg, "cowork_ops").glob("refine_request_*.json"):
    p.unlink()

# No signals: should be level 0
m = data_processor.run(cfg)
assert m["refinement_level"] == 0, f"Expected level 0, got {m['refinement_level']}"
print(f"OK  zero signals -> level 0")

# Write one signal: next run should be level 1
from utils import write_manifest  # noqa: E402
write_manifest(get_path(cfg, "cowork_ops") / "refine_request_test1.json",
               {"reason": "test"})
m = data_processor.run(cfg)
assert m["refinement_level"] == 1, f"Expected level 1, got {m['refinement_level']}"
print(f"OK  one signal -> level 1 (signal was consumed)")

# Signal should now be in processed/, so next run is level 0 again
m = data_processor.run(cfg)
assert m["refinement_level"] == 0, f"Expected level 0 after consumption, got {m['refinement_level']}"
print(f"OK  processed signals don't re-trigger -> back to level 0")

# Write 3 signals: should be level 3
for i in range(3):
    write_manifest(get_path(cfg, "cowork_ops") / f"refine_request_test_{i}.json",
                   {"reason": "test"})
m = data_processor.run(cfg)
assert m["refinement_level"] == 3, f"Expected level 3, got {m['refinement_level']}"
print(f"OK  three signals -> level 3 (capped)")

# Write 10 signals: should cap at level 3
for i in range(10):
    write_manifest(get_path(cfg, "cowork_ops") / f"refine_request_test_cap_{i}.json",
                   {"reason": "test"})
m = data_processor.run(cfg)
assert m["refinement_level"] == 3, f"Level should cap at 3, got {m['refinement_level']}"
print(f"OK  ten signals -> level 3 (still capped)")

print("\nAll refinement tests passed.")

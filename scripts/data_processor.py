"""
data_processor.py
-----------------
Raw -> Clean -> Curated pipeline, with escalating refinement.

Responsibilities:
  * Read from <base>/data_raw (READ-ONLY from our side)
  * Dedupe, normalize formatting, validate schema
  * Apply refinement operations when the orchestrator signals regression
  * Write cleaned data to <base>/data_clean
  * Write curated (train/test split, versioned) data to <base>/data_curated
  * Log transformation operations to <base>/cowork_ops

Refinement levels (escalate when the orchestrator keeps signaling):
  0  baseline     - dedupe + normalize + schema validate
  1  mild         - + drop rows with any NaN in features
  2  moderate     - + remove outliers (>3 sigma on any feature)
  3  aggressive   - + drop low-variance features + class rebalance (undersample majority)

The orchestrator escalates by writing refine_request files. Each processed
request is moved to cowork_ops/processed/ so it can't trigger twice.

Invariants:
  * Raw data is NEVER modified. Fingerprinted before and after.
  * Every curated dataset is versioned with a dataset_id.
  * refinement_level is recorded in the manifest for traceability.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (  # noqa: E402
    dir_fingerprint,
    ensure_dirs,
    get_logger,
    get_path,
    halt,
    load_config,
    new_run_id,
    read_manifest,
    utc_now_iso,
    write_manifest,
)


# ---------------------------------------------------------------
# LOADING
# ---------------------------------------------------------------

def load_raw_files(raw_dir: Path) -> pd.DataFrame:
    """Load and concatenate all CSVs in raw_dir. Extend for other formats as needed."""
    csvs = sorted(raw_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")
    frames = []
    for p in csvs:
        df = pd.read_csv(p)
        df["__source_file"] = p.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------
# BASELINE CLEANING (level 0)
# ---------------------------------------------------------------

def normalize_formatting(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace, lowercase column names, coerce obvious types."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
    return df


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    return df, before - len(df)


def validate_schema(df: pd.DataFrame, schema_cfg: dict) -> None:
    """Halt if required columns are missing or mistyped."""
    required = schema_cfg.get("required_columns", [])
    for col_spec in required:
        name = col_spec["name"]
        expected = col_spec["type"]
        if name not in df.columns:
            raise ValueError(f"Schema violation: missing required column '{name}'")
        actual_dtype = df[name].dtype
        compat = {
            "int": np.issubdtype(actual_dtype, np.integer) or np.issubdtype(actual_dtype, np.floating),
            "float": np.issubdtype(actual_dtype, np.number),
            "str": actual_dtype == object,
            "bool": actual_dtype == bool or np.issubdtype(actual_dtype, np.integer),
        }
        if not compat.get(expected, True):
            raise ValueError(
                f"Schema violation: column '{name}' expected {expected}, got {actual_dtype}"
            )


# ---------------------------------------------------------------
# REFINEMENT OPERATIONS (levels 1-3)
# ---------------------------------------------------------------

def drop_missing(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, int]:
    """Level 1: drop any row with NaN in feature columns."""
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    return df, before - len(df)


def remove_outliers(df: pd.DataFrame, target_col: str, sigma: float = 3.0
                    ) -> tuple[pd.DataFrame, int]:
    """Level 2: drop rows where any numeric feature is >sigma std from its mean."""
    feature_cols = [c for c in df.columns
                    if c != target_col and c != "__source_file"
                    and np.issubdtype(df[c].dtype, np.number)]
    if not feature_cols:
        return df, 0
    before = len(df)
    stds = df[feature_cols].std(ddof=0).replace(0, 1)  # avoid div-by-zero on constant cols
    z = np.abs((df[feature_cols] - df[feature_cols].mean()) / stds)
    mask = (z <= sigma).all(axis=1)
    df = df[mask].reset_index(drop=True)
    return df, before - len(df)


def drop_low_variance_features(df: pd.DataFrame, target_col: str,
                               min_variance: float = 0.01) -> tuple[pd.DataFrame, list[str]]:
    """Level 3a: drop numeric features with variance below threshold."""
    feature_cols = [c for c in df.columns
                    if c != target_col and c != "__source_file"
                    and np.issubdtype(df[c].dtype, np.number)]
    dropped = [c for c in feature_cols if df[c].var() < min_variance]
    if dropped:
        df = df.drop(columns=dropped)
    return df, dropped


def rebalance_classes(df: pd.DataFrame, target_col: str, seed: int
                      ) -> tuple[pd.DataFrame, dict]:
    """Level 3b: undersample majority class to match minority count."""
    counts = df[target_col].value_counts()
    if len(counts) < 2:
        return df, {"action": "skipped_single_class"}
    min_count = int(counts.min())
    rng = np.random.default_rng(seed)
    pieces = []
    for cls, grp in df.groupby(target_col):
        if len(grp) > min_count:
            idx = rng.choice(len(grp), size=min_count, replace=False)
            pieces.append(grp.iloc[idx])
        else:
            pieces.append(grp)
    balanced = pd.concat(pieces, ignore_index=True).sample(
        frac=1, random_state=seed).reset_index(drop=True)
    return balanced, {
        "before": {str(k): int(v) for k, v in counts.items()},
        "after": {str(k): int(v) for k, v in balanced[target_col].value_counts().items()},
    }


# ---------------------------------------------------------------
# REFINEMENT SIGNAL HANDLING
# ---------------------------------------------------------------

def read_pending_refinements(cfg: dict) -> list[dict]:
    """Find unprocessed refine_request files in cowork_ops/ (not in processed/)."""
    cowork = get_path(cfg, "cowork_ops")
    pending = []
    for p in sorted(cowork.glob("refine_request_*.json")):
        if "processed" in p.parts:
            continue
        try:
            pending.append({"path": p, "data": read_manifest(p)})
        except Exception:
            continue
    return pending


def determine_refinement_level(pending: list[dict]) -> int:
    """
    Escalation: level = min(N_pending_signals, 3).
    Zero pending = level 0 (baseline).
    """
    return min(len(pending), 3)


def mark_refinements_processed(pending: list[dict], dataset_id: str,
                               cfg: dict, logger) -> None:
    """Move processed refine_request files to cowork_ops/processed/."""
    if not pending:
        return
    processed_dir = get_path(cfg, "cowork_ops") / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    for req in pending:
        src: Path = req["path"]
        dst = processed_dir / f"{dataset_id}__{src.name}"
        shutil.move(str(src), str(dst))
        logger.info(f"marked refinement processed: {src.name} -> {dst.name}")


# ---------------------------------------------------------------
# CURATION
# ---------------------------------------------------------------

def curate(df: pd.DataFrame, target_col: str, test_split: float, seed: int
           ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic train/test split."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_test = int(len(df) * test_split)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


# ---------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------

def run(cfg: dict, run_id: str | None = None,
        refinement_level_override: int | None = None) -> dict:
    """
    Execute the full data pipeline.

    Args:
      refinement_level_override: force level 0-3. If None, derived from
        pending refine_request signals.
    """
    ensure_dirs(cfg)
    run_id = run_id or new_run_id("data")
    logger = get_logger("data_processor", cfg)
    logger.info(f"[{run_id}] data pipeline start")

    raw_dir = get_path(cfg, "raw")
    clean_dir = get_path(cfg, "clean")
    curated_dir = get_path(cfg, "curated")
    cowork_dir = get_path(cfg, "cowork_ops")

    # --- SAFETY: fingerprint raw before ---
    raw_fp_before = dir_fingerprint(raw_dir)
    if not raw_fp_before:
        halt(logger, f"data_raw is empty: {raw_dir}")

    # --- REFINEMENT LEVEL ---
    pending = read_pending_refinements(cfg)
    level = (refinement_level_override if refinement_level_override is not None
             else determine_refinement_level(pending))
    logger.info(f"[{run_id}] refinement_level={level} ({len(pending)} pending signals)")

    target_col = cfg["data"]["target_column"]
    seed = cfg["system"]["random_seed"]
    ops_log: list[dict] = []

    # --- LOAD ---
    df = load_raw_files(raw_dir)
    rows_raw = len(df)
    ops_log.append({"op": "load", "rows": rows_raw})
    logger.info(f"[{run_id}] loaded {rows_raw} rows from raw")

    # --- LEVEL 0 (baseline) ---
    df = normalize_formatting(df)
    ops_log.append({"op": "normalize_formatting"})

    df, dupes_removed = remove_duplicates(df)
    ops_log.append({"op": "remove_duplicates", "removed": dupes_removed})

    validate_schema(df, cfg["data"]["schema"])
    ops_log.append({"op": "validate_schema", "result": "ok"})

    if target_col not in df.columns:
        halt(logger, f"target column '{target_col}' not in data")

    # --- LEVEL 1 ---
    if level >= 1:
        df, dropped = drop_missing(df, target_col)
        ops_log.append({"op": "drop_missing", "rows_dropped": dropped})
        logger.info(f"[{run_id}] level 1: dropped {dropped} rows with NaN")

    # --- LEVEL 2 ---
    if level >= 2:
        df, outliers = remove_outliers(df, target_col, sigma=3.0)
        ops_log.append({"op": "remove_outliers", "rows_dropped": outliers, "sigma": 3.0})
        logger.info(f"[{run_id}] level 2: dropped {outliers} outlier rows")

    # --- LEVEL 3 ---
    if level >= 3:
        df, dropped_feats = drop_low_variance_features(df, target_col)
        ops_log.append({"op": "drop_low_variance_features", "features_dropped": dropped_feats})
        logger.info(f"[{run_id}] level 3a: dropped features {dropped_feats}")

        df, rebalance_info = rebalance_classes(df, target_col, seed)
        ops_log.append({"op": "rebalance_classes", **rebalance_info})
        logger.info(f"[{run_id}] level 3b: rebalanced classes {rebalance_info}")

    rows_clean = len(df)
    if rows_clean < cfg["data"]["min_rows_after_clean"]:
        halt(logger, f"only {rows_clean} rows survived level-{level} cleaning "
                     f"(min {cfg['data']['min_rows_after_clean']})")

    # --- WRITE CLEAN ---
    dataset_id = run_id.replace("data_", "ds_")
    clean_file = clean_dir / f"{dataset_id}.parquet"
    df.to_parquet(clean_file, index=False)
    logger.info(f"[{run_id}] wrote clean dataset: {clean_file} ({rows_clean} rows)")

    # --- CURATE ---
    train_df, test_df = curate(
        df.drop(columns=["__source_file"], errors="ignore"),
        target_col,
        cfg["data"]["test_split"],
        seed,
    )

    curated_subdir = curated_dir / dataset_id
    curated_subdir.mkdir(parents=True, exist_ok=True)
    train_path = curated_subdir / "train.parquet"
    test_path = curated_subdir / "test.parquet"
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    # --- SAFETY: fingerprint raw after ---
    raw_fp_after = dir_fingerprint(raw_dir)
    if raw_fp_after != raw_fp_before:
        halt(logger, "raw data was modified during pipeline — integrity violation")

    # --- MANIFEST ---
    manifest = {
        "dataset_id": dataset_id,
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "refinement_level": level,
        "refinement_signals_processed": [str(r["path"].name) for r in pending],
        "rows_raw": rows_raw,
        "rows_clean": rows_clean,
        "duplicates_removed": dupes_removed,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "target_column": target_col,
        "columns": list(train_df.columns),
        "raw_fingerprint": raw_fp_before,
        "paths": {
            "clean": str(clean_file),
            "train": str(train_path),
            "test": str(test_path),
        },
    }
    write_manifest(curated_subdir / "manifest.json", manifest)

    # --- OP LOG ---
    op_log_path = cowork_dir / f"data_ops_{run_id}.json"
    ops_log.append({"op": "split", "train": len(train_df), "test": len(test_df)})
    write_manifest(op_log_path, {
        "run_id": run_id,
        "stage": "data",
        "refinement_level": level,
        "ops": ops_log,
        "output_dataset_id": dataset_id,
    })

    # --- MARK SIGNALS PROCESSED ---
    mark_refinements_processed(pending, dataset_id, cfg, logger)

    logger.info(f"[{run_id}] data pipeline done — dataset_id={dataset_id} "
                f"level={level} rows={rows_clean}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--refinement-level", type=int, default=None,
                        help="Force refinement level 0-3 (override signal detection)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    m = run(cfg, refinement_level_override=args.refinement_level)
    print(f"DATASET_ID={m['dataset_id']} LEVEL={m['refinement_level']}")

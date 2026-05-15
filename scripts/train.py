"""
train.py
--------
Training engine.

Reads curated datasets from <base>/data_curated/<dataset_id>/
Writes model checkpoints to <base>/models/<model_id>/
Writes training run metadata to <base>/training_runs/<run_id>/

Supports swappable backends via config.training.model_type:
  * sklearn_rf       - RandomForestClassifier
  * sklearn_logreg   - LogisticRegression
  * pytorch_mlp      - simple MLP (requires torch)

NEVER reads from data_raw or data_clean.
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (  # noqa: E402
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
# DETERMINISM
# ---------------------------------------------------------------

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------
# DATA LOADING (curated ONLY)
# ---------------------------------------------------------------

def load_curated(cfg: dict, dataset_id: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    curated_dir = get_path(cfg, "curated") / dataset_id
    if not curated_dir.exists():
        raise FileNotFoundError(f"dataset_id not found in curated: {dataset_id}")
    manifest_path = curated_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"dataset manifest missing: {manifest_path}")
    manifest = read_manifest(manifest_path)
    train = pd.read_parquet(curated_dir / "train.parquet")
    test = pd.read_parquet(curated_dir / "test.parquet")
    return train, test, manifest["target_column"]


def latest_dataset_id(cfg: dict) -> str:
    curated_dir = get_path(cfg, "curated")
    subdirs = [p for p in curated_dir.iterdir() if p.is_dir()]
    if not subdirs:
        raise FileNotFoundError("no curated datasets available")
    return sorted(subdirs, key=lambda p: p.name)[-1].name


# ---------------------------------------------------------------
# MODEL BACKENDS
# ---------------------------------------------------------------

def train_sklearn(model_type: str, hparams: dict, X_train, y_train) -> Any:
    if model_type == "sklearn_rf":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(random_state=42, **hparams)
    elif model_type == "sklearn_logreg":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=42, **hparams)
    else:
        raise ValueError(f"unknown sklearn model: {model_type}")
    model.fit(X_train, y_train)
    return model


def train_pytorch_mlp(hparams: dict, X_train, y_train, seed: int) -> Any:
    """Simple MLP classifier in PyTorch. Returns a dict (state_dict + meta)."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    set_seeds(seed)
    n_classes = int(np.max(y_train)) + 1
    n_features = X_train.shape[1]

    layers: list[nn.Module] = []
    prev = n_features
    for h in hparams["hidden_dims"]:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(hparams["dropout"])]
        prev = h
    layers.append(nn.Linear(prev, n_classes))
    model = nn.Sequential(*layers)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    loader = DataLoader(TensorDataset(X_t, y_t),
                        batch_size=hparams["batch_size"], shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=hparams["learning_rate"])
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(hparams["epochs"]):
        for xb, yb in loader:
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
    return {"model": model, "n_classes": n_classes, "n_features": n_features}


# ---------------------------------------------------------------
# SAVE / LOAD CHECKPOINT
# ---------------------------------------------------------------

def save_checkpoint(model_dir: Path, model: Any, model_type: str) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    if model_type.startswith("sklearn"):
        path = model_dir / "model.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f)
    elif model_type == "pytorch_mlp":
        import torch
        path = model_dir / "model.pt"
        torch.save({
            "state_dict": model["model"].state_dict(),
            "n_classes": model["n_classes"],
            "n_features": model["n_features"],
        }, path)
    else:
        raise ValueError(f"cannot save unknown model_type: {model_type}")
    return path


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------

def run(cfg: dict, dataset_id: str | None = None,
        hparam_override: dict | None = None,
        run_id: str | None = None) -> dict:
    ensure_dirs(cfg)
    run_id = run_id or new_run_id("train")
    logger = get_logger("train", cfg)

    seed = cfg["system"]["random_seed"]
    set_seeds(seed)

    dataset_id = dataset_id or latest_dataset_id(cfg)
    logger.info(f"[{run_id}] training on dataset {dataset_id}")

    train_df, test_df, target_col = load_curated(cfg, dataset_id)
    feature_cols = [c for c in train_df.columns if c != target_col]
    X_train = train_df[feature_cols].to_numpy()
    y_train = train_df[target_col].to_numpy()
    X_test = test_df[feature_cols].to_numpy()
    y_test = test_df[target_col].to_numpy()

    model_type = cfg["training"]["model_type"]
    hparams = dict(cfg["training"]["hyperparameters"][model_type])
    if hparam_override:
        hparams.update(hparam_override)
    logger.info(f"[{run_id}] model_type={model_type} hparams={hparams}")

    t0 = time.time()
    if model_type.startswith("sklearn"):
        model = train_sklearn(model_type, hparams, X_train, y_train)
    elif model_type == "pytorch_mlp":
        model = train_pytorch_mlp(hparams, X_train, y_train, seed)
    else:
        halt(logger, f"unsupported model_type: {model_type}")
    duration = time.time() - t0

    # Model ID + save
    model_id = run_id.replace("train_", "model_")
    model_dir = get_path(cfg, "models") / model_id
    ckpt_path = save_checkpoint(model_dir, model, model_type)

    # Training run record
    train_run_dir = get_path(cfg, "training_runs") / run_id
    train_run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "model_id": model_id,
        "dataset_id": dataset_id,
        "model_type": model_type,
        "hyperparameters": hparams,
        "seed": seed,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "duration_s": round(duration, 3),
        "checkpoint": str(ckpt_path),
        "created_at": utc_now_iso(),
    }
    write_manifest(train_run_dir / "manifest.json", manifest)
    write_manifest(model_dir / "manifest.json", manifest)

    # Stash feature columns alongside the model for eval/inference
    with open(model_dir / "feature_cols.json", "w", encoding="utf-8") as f:
        json.dump({"feature_cols": feature_cols, "target_col": target_col}, f)

    logger.info(f"[{run_id}] trained in {duration:.2f}s -> {ckpt_path}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dataset-id", type=str, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    m = run(cfg, dataset_id=args.dataset_id)
    print(f"MODEL_ID={m['model_id']}")

"""
eval.py
-------
Evaluation engine.

Runs after every training cycle. For a given model_id:
  * Loads model + its curated test set
  * Computes accuracy, f1_macro, log_loss
  * Compares against previous model (if any) on same target column
  * Detects:
      - overfitting       (train-test gap)
      - confidence drift  (tabular hallucination proxy)
      - instruction drift (prediction class distribution shift)
  * Returns a 0-100 composite score
  * Writes JSON to <base>/evals/<eval_id>.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
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
# MODEL LOADING
# ---------------------------------------------------------------

def load_model(model_dir: Path, model_type: str) -> Any:
    if model_type.startswith("sklearn"):
        with open(model_dir / "model.pkl", "rb") as f:
            return pickle.load(f)
    elif model_type == "pytorch_mlp":
        import torch
        import torch.nn as nn
        ckpt = torch.load(model_dir / "model.pt", map_location="cpu")
        # We need to know the hidden dims — reload them from the training manifest
        train_manifest = read_manifest(model_dir / "manifest.json")
        hidden_dims = train_manifest["hyperparameters"]["hidden_dims"]
        dropout = train_manifest["hyperparameters"]["dropout"]
        layers: list[nn.Module] = []
        prev = ckpt["n_features"]
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, ckpt["n_classes"]))
        model = nn.Sequential(*layers)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model
    raise ValueError(f"unknown model_type: {model_type}")


def predict(model: Any, X, model_type: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (predicted_labels, predicted_probabilities)."""
    if model_type.startswith("sklearn"):
        preds = model.predict(X)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
        else:
            # fallback: one-hot of predictions
            n_classes = len(np.unique(preds))
            probs = np.eye(n_classes)[preds]
        return preds, probs
    elif model_type == "pytorch_mlp":
        import torch
        with torch.no_grad():
            logits = model(torch.tensor(X, dtype=torch.float32))
            probs = torch.softmax(logits, dim=1).numpy()
            preds = probs.argmax(axis=1)
        return preds, probs
    raise ValueError(f"unknown model_type: {model_type}")


# ---------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------

def compute_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, log_loss
    # Guard log_loss against single-class edge cases
    try:
        ll = float(log_loss(y_true, y_proba, labels=sorted(np.unique(y_true))))
    except Exception:
        ll = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "log_loss": ll,
    }


# ---------------------------------------------------------------
# DRIFT DETECTION
# ---------------------------------------------------------------

def detect_overfit(train_acc: float, test_acc: float, threshold: float) -> dict:
    gap = train_acc - test_acc
    return {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "gap": gap,
        "overfit_flag": bool(gap > threshold),
    }


def detect_confidence_drift(curr_probs: np.ndarray, prev_probs: np.ndarray | None,
                            curr_correct: np.ndarray, threshold: float) -> dict:
    """
    'Hallucination' proxy for classifiers: model becoming more confident on
    the wrong answers. We compare mean confidence on *incorrect* predictions.
    Higher = worse.
    """
    curr_wrong_conf = float(curr_probs.max(axis=1)[~curr_correct].mean()) \
        if (~curr_correct).any() else 0.0
    result: dict = {"curr_wrong_conf": curr_wrong_conf}
    if prev_probs is None:
        result["delta"] = 0.0
        result["drift_flag"] = False
        return result
    # prev_probs comes from previous eval — we only have summary stats, not raw
    # so we compare current to the stored summary.
    prev_wrong_conf = float(prev_probs) if isinstance(prev_probs, (int, float)) else 0.0
    delta = curr_wrong_conf - prev_wrong_conf
    result.update({
        "prev_wrong_conf": prev_wrong_conf,
        "delta": delta,
        "drift_flag": bool(delta > threshold),
    })
    return result


def detect_instruction_drift(curr_pred_dist: dict, prev_pred_dist: dict | None,
                             threshold: float) -> dict:
    """
    Shift in the model's output class distribution vs previous model.
    Measured as max absolute difference across classes.
    """
    if prev_pred_dist is None:
        return {"max_shift": 0.0, "drift_flag": False, "distribution": curr_pred_dist}
    all_classes = set(curr_pred_dist) | set(prev_pred_dist)
    max_shift = max(
        abs(curr_pred_dist.get(c, 0.0) - prev_pred_dist.get(c, 0.0))
        for c in all_classes
    )
    return {
        "max_shift": max_shift,
        "drift_flag": bool(max_shift > threshold),
        "distribution": curr_pred_dist,
        "previous_distribution": prev_pred_dist,
    }


# ---------------------------------------------------------------
# SCORING 0-100
# ---------------------------------------------------------------

def composite_score(metrics: dict, overfit: dict, conf_drift: dict,
                    inst_drift: dict) -> float:
    """
    Composite score out of 100.
    Base = test_accuracy * 60 + f1_macro * 40
    Penalties for drift flags.
    """
    base = metrics["accuracy"] * 60 + metrics["f1_macro"] * 40
    penalty = 0.0
    if overfit["overfit_flag"]:
        penalty += 10 * min(1.0, overfit["gap"] / 0.3)
    if conf_drift.get("drift_flag"):
        penalty += 10
    if inst_drift.get("drift_flag"):
        penalty += 10
    return max(0.0, min(100.0, base - penalty))


# ---------------------------------------------------------------
# PREVIOUS MODEL LOOKUP
# ---------------------------------------------------------------

def find_previous_eval(cfg: dict, current_model_id: str) -> dict | None:
    evals_dir = get_path(cfg, "evals")
    files = sorted(evals_dir.glob("eval_*.json"))
    # Return most recent eval that isn't for current_model_id
    for f in reversed(files):
        try:
            data = read_manifest(f)
            if data.get("model_id") != current_model_id:
                return data
        except Exception:
            continue
    return None


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------

def run(cfg: dict, model_id: str | None = None, run_id: str | None = None) -> dict:
    ensure_dirs(cfg)
    run_id = run_id or new_run_id("eval")
    logger = get_logger("eval", cfg)

    # Resolve model
    models_dir = get_path(cfg, "models")
    if model_id is None:
        dirs = [p for p in models_dir.iterdir() if p.is_dir()]
        if not dirs:
            halt(logger, "no trained models available to evaluate")
        model_id = sorted(dirs, key=lambda p: p.name)[-1].name
    model_dir = models_dir / model_id
    if not model_dir.exists():
        halt(logger, f"model not found: {model_id}")

    train_manifest = read_manifest(model_dir / "manifest.json")
    model_type = train_manifest["model_type"]
    dataset_id = train_manifest["dataset_id"]
    feature_cols = train_manifest["feature_cols"]
    target_col = train_manifest["target_col"]

    # Load model and curated split
    model = load_model(model_dir, model_type)
    curated_dir = get_path(cfg, "curated") / dataset_id
    if not (curated_dir / "manifest.json").exists():
        halt(logger, f"dataset manifest missing for {dataset_id}")

    train_df = pd.read_parquet(curated_dir / "train.parquet")
    test_df = pd.read_parquet(curated_dir / "test.parquet")
    X_train, y_train = train_df[feature_cols].to_numpy(), train_df[target_col].to_numpy()
    X_test, y_test = test_df[feature_cols].to_numpy(), test_df[target_col].to_numpy()

    # Predict
    train_pred, train_proba = predict(model, X_train, model_type)
    test_pred, test_proba = predict(model, X_test, model_type)

    # Metrics
    test_metrics = compute_metrics(y_test, test_pred, test_proba)
    train_metrics = compute_metrics(y_train, train_pred, train_proba)

    # Drift vs previous
    previous = find_previous_eval(cfg, model_id)
    prev_wrong_conf = previous["confidence_drift"]["curr_wrong_conf"] if previous else None
    prev_pred_dist = previous["instruction_drift"]["distribution"] if previous else None

    overfit = detect_overfit(
        train_metrics["accuracy"], test_metrics["accuracy"],
        cfg["evaluation"]["overfitting_threshold"],
    )
    conf_drift = detect_confidence_drift(
        test_proba, prev_wrong_conf, test_pred == y_test,
        cfg["evaluation"]["confidence_drift_threshold"],
    )
    # prediction class distribution
    unique, counts = np.unique(test_pred, return_counts=True)
    curr_pred_dist = {str(int(u)): float(c / len(test_pred)) for u, c in zip(unique, counts)}
    inst_drift = detect_instruction_drift(
        curr_pred_dist, prev_pred_dist,
        cfg["evaluation"]["instruction_drift_threshold"],
    )

    score = composite_score(test_metrics, overfit, conf_drift, inst_drift)

    result = {
        "eval_id": run_id,
        "model_id": model_id,
        "dataset_id": dataset_id,
        "evaluated_at": utc_now_iso(),
        "metrics_test": test_metrics,
        "metrics_train": train_metrics,
        "overfitting": overfit,
        "confidence_drift": conf_drift,
        "instruction_drift": inst_drift,
        "score": round(score, 2),
        "previous_eval_id": previous["eval_id"] if previous else None,
        "previous_score": previous["score"] if previous else None,
    }

    # Validate output structure before writing (safety rule)
    required_keys = {"eval_id", "model_id", "score", "metrics_test"}
    if not required_keys.issubset(result.keys()):
        halt(logger, "eval produced invalid structure")

    out = get_path(cfg, "evals") / f"{run_id}.json"
    write_manifest(out, result)
    logger.info(f"[{run_id}] eval complete — model={model_id} score={score:.2f}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model-id", type=str, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    r = run(cfg, model_id=args.model_id)
    print(f"EVAL_ID={r['eval_id']} SCORE={r['score']}")

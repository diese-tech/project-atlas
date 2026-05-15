"""
orchestrator.py
---------------
The system brain. Runs the autonomous loop:

    DATA -> TRAIN -> EVAL -> DECIDE -> LOOP

Decision rules (from config):
  * score improves      -> PROMOTE model (write promoted.json pointer)
  * score regresses     -> REFINE_DATA (flag dataset, trigger re-pipeline)
  * score plateaus      -> TUNE_HYPERPARAMS (cycle through config sweep)

Safety:
  * retry once on transient failure, then HALT
  * honors SafetyViolation from any stage -> halt loop
  * versions every artifact
  * supports rollback to last promoted model
"""
from __future__ import annotations

import argparse
import itertools
import sys
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data_processor  # noqa: E402
import eval as eval_module  # noqa: E402
import train as train_module  # noqa: E402
from utils import (  # noqa: E402
    SafetyViolation,
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
# PROMOTION & ROLLBACK
# ---------------------------------------------------------------

def promoted_pointer_path(cfg: dict) -> Path:
    return get_path(cfg, "orchestration").parent / "orchestration" / "promoted.json"


def current_promoted(cfg: dict) -> dict | None:
    p = promoted_pointer_path(cfg)
    if not p.exists():
        return None
    try:
        return read_manifest(p)
    except Exception:
        return None


def promote(cfg: dict, eval_result: dict, logger) -> None:
    ptr = promoted_pointer_path(cfg)
    prev = current_promoted(cfg)
    record = {
        "model_id": eval_result["model_id"],
        "eval_id": eval_result["eval_id"],
        "score": eval_result["score"],
        "promoted_at": utc_now_iso(),
        "previous_promoted": prev,
    }
    write_manifest(ptr, record)
    logger.info(f"PROMOTED model {record['model_id']} (score {record['score']})")


def rollback(cfg: dict, logger) -> dict | None:
    """Restore the previous promoted model as active."""
    curr = current_promoted(cfg)
    if not curr or not curr.get("previous_promoted"):
        logger.warning("rollback requested but no previous promoted model exists")
        return None
    prev = curr["previous_promoted"]
    write_manifest(promoted_pointer_path(cfg), prev)
    logger.info(f"ROLLED BACK to model {prev['model_id']}")
    return prev


# ---------------------------------------------------------------
# HYPERPARAM SWEEP STATE
# ---------------------------------------------------------------

def hparam_combinations(cfg: dict) -> list[dict]:
    """Generate the cartesian product of tuning options."""
    model_type = cfg["training"]["model_type"]
    sweep = cfg["orchestrator"]["hyperparam_tuning"].get(model_type, {})
    if not sweep:
        return []
    keys = list(sweep.keys())
    values = [sweep[k] for k in keys]
    combos = []
    for combo in itertools.product(*values):
        combos.append(dict(zip(keys, combo)))
    return combos


# ---------------------------------------------------------------
# DECISION LOGIC
# ---------------------------------------------------------------

def decide(cfg: dict, curr_score: float, prev_score: float | None,
           plateau_counter: int) -> tuple[str, int]:
    """Return (action, new_plateau_counter)."""
    delta = cfg["orchestrator"]["improvement_delta"]
    tol = cfg["orchestrator"]["plateau_tolerance"]

    if prev_score is None:
        return ("promote" if curr_score >= cfg["evaluation"]["min_score_to_promote"]
                else "refine_data", 0)

    if curr_score > prev_score + delta:
        return "promote", 0
    if curr_score < prev_score - delta:
        return "refine_data", 0
    # plateau territory
    plateau_counter += 1
    if plateau_counter >= tol:
        return "tune_hyperparams", 0
    return "hold", plateau_counter


# ---------------------------------------------------------------
# STAGE RUNNERS WITH RETRY
# ---------------------------------------------------------------

def run_with_retry(fn, logger, stage: str, retries: int, **kwargs):
    last_exc = None
    for attempt in range(retries + 1):
        try:
            return fn(**kwargs)
        except SafetyViolation:
            raise  # safety violations never retry
        except Exception as e:
            last_exc = e
            logger.warning(f"[{stage}] attempt {attempt + 1} failed: {e}")
            logger.warning(traceback.format_exc())
    halt(logger, f"[{stage}] exhausted retries ({retries + 1}): {last_exc}")


# ---------------------------------------------------------------
# ORCHESTRATOR MAIN LOOP
# ---------------------------------------------------------------

def run(cfg: dict, max_iters: int | None = None) -> dict:
    ensure_dirs(cfg)
    master_id = new_run_id("orch")
    logger = get_logger("orchestrator", cfg)
    logger.info(f"[{master_id}] orchestrator start")

    retries = cfg["orchestrator"]["retry_on_failure"]
    max_iters = max_iters or cfg["system"]["max_loop_iterations"]
    sweep = hparam_combinations(cfg)
    sweep_idx = 0

    history: list[dict] = []
    plateau_counter = 0
    regression_streak = 0
    prev_score: float | None = None
    halted_reason: str | None = None

    try:
        for i in range(max_iters):
            iter_id = f"{master_id}_iter{i:02d}"
            logger.info(f"[{iter_id}] === loop iteration {i + 1}/{max_iters} ===")

            # --- DATA ---
            data_manifest = run_with_retry(
                data_processor.run, logger, "data", retries,
                cfg=cfg, run_id=new_run_id("data"),
            )

            # --- TRAIN ---
            hparam_override = sweep[sweep_idx % len(sweep)] if sweep else None
            train_manifest = run_with_retry(
                train_module.run, logger, "train", retries,
                cfg=cfg, dataset_id=data_manifest["dataset_id"],
                hparam_override=hparam_override, run_id=new_run_id("train"),
            )

            # --- EVAL ---
            eval_result = run_with_retry(
                eval_module.run, logger, "eval", retries,
                cfg=cfg, model_id=train_manifest["model_id"], run_id=new_run_id("eval"),
            )
            score = eval_result["score"]

            # --- DECIDE ---
            action, plateau_counter = decide(cfg, score, prev_score, plateau_counter)
            logger.info(f"[{iter_id}] score={score:.2f} prev={prev_score} action={action}")

            if action == "promote":
                promote(cfg, eval_result, logger)
                prev_score = score
                regression_streak = 0
            elif action == "refine_data":
                # Write a refinement signal. data_processor will escalate
                # its cleaning level based on how many pending signals exist.
                refine_flag = get_path(cfg, "cowork_ops") / f"refine_request_{iter_id}.json"
                write_manifest(refine_flag, {
                    "iter_id": iter_id, "reason": "score_regression",
                    "prev_score": prev_score, "curr_score": score,
                })
                regression_streak += 1
                logger.info(f"[{iter_id}] refine_data requested "
                            f"(regression_streak={regression_streak})")
                # Only rollback after refinement has had chances to help.
                # Two consecutive regressions despite refinement -> revert.
                if regression_streak >= 2:
                    logger.warning(f"[{iter_id}] refinement not recovering — rolling back")
                    rollback(cfg, logger)
                    regression_streak = 0
            elif action == "tune_hyperparams":
                sweep_idx += 1
                logger.info(f"[{iter_id}] advancing sweep to idx {sweep_idx}")
            # 'hold' does nothing beyond logging

            history.append({
                "iter_id": iter_id,
                "dataset_id": data_manifest["dataset_id"],
                "refinement_level": data_manifest.get("refinement_level", 0),
                "model_id": train_manifest["model_id"],
                "eval_id": eval_result["eval_id"],
                "score": score,
                "action": action,
                "hparam_override": hparam_override,
            })

    except SafetyViolation as e:
        halted_reason = f"safety_violation: {e}"
        logger.error(halted_reason)
    except Exception as e:
        halted_reason = f"unexpected: {e}"
        logger.error(halted_reason)
        logger.error(traceback.format_exc())

    # --- FINAL REPORT ---
    report = {
        "master_run_id": master_id,
        "started_at": history[0]["iter_id"] if history else None,
        "completed_at": utc_now_iso(),
        "iterations": history,
        "final_promoted": current_promoted(cfg),
        "halted_reason": halted_reason,
    }
    write_manifest(get_path(cfg, "orchestration") / f"{master_id}.json", report)
    logger.info(f"[{master_id}] orchestrator end — halted={halted_reason}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--rollback", action="store_true",
                        help="Rollback to previous promoted model and exit.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.rollback:
        logger = get_logger("orchestrator", cfg)
        rollback(cfg, logger)
        sys.exit(0)
    report = run(cfg, max_iters=args.max_iters)
    print(f"MASTER_RUN_ID={report['master_run_id']}")
    print(f"FINAL_PROMOTED={report['final_promoted']}")
    if report["halted_reason"]:
        sys.exit(2)

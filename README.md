# Autonomous Local ML Training Loop

A lightweight, file-driven, deterministic ML platform that runs the full loop:

```
DATA -> TRAIN -> EVAL -> DECIDE -> LOOP
```

Base directory: `C:\AI`

## Directory map

```
C:\AI
├── data_raw          # Drop your CSVs here (read-only from the system's POV)
├── data_clean        # Deduped + normalized parquet files
├── data_curated      # Versioned train/test splits + manifest.json
├── models            # Versioned model checkpoints + manifests
├── training_runs     # Training run metadata
├── evals             # JSON eval reports (one per model)
├── orchestration     # Master run reports + promoted.json pointer
├── cowork_ops        # Transformation op-logs + refinement signals
├── logs              # Structured JSON log files
├── config
│   └── config.yaml   # All tunables live here
└── scripts
    ├── utils.py
    ├── data_processor.py
    ├── train.py
    ├── eval.py
    └── orchestrator.py
```

## Install

```powershell
cd C:\AI
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`pytorch` is optional — only needed if you set `training.model_type: pytorch_mlp`.

## Prepare data

Drop one or more CSVs into `C:\AI\data_raw`. They must contain the target
column defined in `config.yaml` (default: `target`, integer-valued).

## Run a single stage (useful for debugging)

```powershell
# Just the data pipeline
python scripts\data_processor.py

# Train on the latest curated dataset
python scripts\train.py

# Evaluate the latest model
python scripts\eval.py
```

Each prints the resulting ID on stdout (e.g. `MODEL_ID=model_...`).

## Run the autonomous loop

```powershell
python scripts\orchestrator.py
```

Optional flags:
- `--max-iters N` — override `system.max_loop_iterations`
- `--rollback`    — revert `promoted.json` to the previous model and exit

The orchestrator writes a full report to `orchestration\<master_run_id>.json`.

## How the decision logic works

After each eval, the orchestrator compares the current score to the previous
promoted score:

| Condition                                   | Action             |
|--------------------------------------------|--------------------|
| `score > prev + improvement_delta`          | **promote**        |
| `score < prev - improvement_delta`          | **refine_data** (escalate cleaning; rollback only after 2 consecutive regressions) |
| Within delta, `plateau_counter < tolerance` | hold               |
| Within delta, `plateau_counter >= tolerance`| **tune_hyperparams** (advance sweep) |

All thresholds live in `config.yaml` under `orchestrator` and `evaluation`.

## Refinement — what "refine_data" actually does

When the orchestrator signals regression, it writes a `refine_request_*.json`
file into `cowork_ops/`. The data pipeline reads these signals on the next run
and escalates its cleaning level based on how many are pending:

| Level | Operations |
|-------|------------|
| 0 (baseline) | dedupe, normalize formatting, schema validate |
| 1 (mild)     | + drop rows with any NaN in features |
| 2 (moderate) | + remove outliers (>3σ on any numeric feature) |
| 3 (aggressive) | + drop low-variance features + undersample majority class |

The level is capped at 3. Once a signal is consumed, `data_processor` moves it
to `cowork_ops/processed/` so it never triggers twice. The refinement level is
recorded in each dataset manifest for traceability.

You can also force a level manually:

```powershell
python scripts\data_processor.py --refinement-level 2
```

## Safety stops

The system halts (raising `SafetyViolation`) if:

1. `data_raw` fingerprint changes between pipeline start and end
2. A curated dataset manifest is missing
3. An evaluation produces an invalid JSON structure
4. A model checkpoint would be silently overwritten (we archive instead)
5. Any stage fails more than `orchestrator.retry_on_failure` times

On halt, the current promoted model is untouched and the master run report
records the halt reason.

## Swapping the model backend

In `config.yaml`:

```yaml
training:
  model_type: sklearn_logreg   # or pytorch_mlp
```

No code changes required.

## Extending

- **New data formats**: extend `data_processor.load_raw_files` (JSONL, parquet ingest, etc.)
- **New model type**: add a branch in `train.train_*` and `eval.load_model`/`eval.predict`
- **New drift metric**: add a detector to `eval.py` and fold it into `composite_score`
- **New refinement level**: add a function in `data_processor.py` and gate it on `level >= N`

## Reproducibility

Seeds are set for `random`, `numpy`, and `torch` (when used). The same raw
data + config will produce identical curated splits, models, and eval scores.
Dataset fingerprints are stored in every manifest so you can prove lineage.

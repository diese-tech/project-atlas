# Project Atlas

Two systems, one repo: a **local AI chat assistant** and an **autonomous ML training loop**, both running entirely on your machine with no cloud dependencies.

| System | Entry point | What it does |
|---|---|---|
| Chat App | `start.bat` | FastAPI + Ollama chat UI with persistent session history |
| ML Loop | `start_pipeline.bat` | Autonomous data → train → eval → decide pipeline |

---

## Directory map

```
C:\AI\                        ← source code (git repo)
├── main.py                   # Chat server (FastAPI + SQLite + Ollama)
├── index.html                # Chat frontend (single-page browser app)
├── models.js                 # Ollama model registry
├── start.bat                 # Start the chat app
├── start_pipeline.bat        # Start the ML pipeline
├── .env                      # Local config — gitignored, copy from .env.example
├── .env.example              # Config template
├── config/
│   └── config.yaml           # All ML pipeline tunables
├── data_raw/                 # Drop training CSVs here (source-controlled)
├── scripts/
│   ├── orchestrator.py
│   ├── data_processor.py
│   ├── train.py
│   ├── eval.py
│   └── utils.py
└── tests/
    └── test_refinement.py

D:\AI\                        ← generated output (written at runtime)
├── data_clean/
├── data_curated/
├── models/
├── training_runs/
├── evals/
├── orchestration/
├── cowork_ops/
└── logs/

D:\chat.db                    ← SQLite chat session database
D:\ollama\                    ← Ollama model files
```

---

## Install

```powershell
cd C:\AI
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and confirm your paths:

```
CHAT_DB_PATH=D:/chat.db
OLLAMA_URL=http://localhost:11434
```

`pytorch` is optional — only needed if `config.yaml` sets `training.model_type: pytorch_mlp`.

---

## Chat App

### Requirements

- [Ollama](https://ollama.com) running locally
- At least one model pulled, e.g. `ollama pull qwen2.5-coder:14b`

### Start

```powershell
.\start.bat
```

Opens a browser to `http://127.0.0.1:5500`. Backend runs on port 8000.

### Environment variables (`.env`)

| Variable | Default | Description |
|---|---|---|
| `CHAT_DB_PATH` | `chat.db` | Path to SQLite session database |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama base URL |

### Models

Edit `models.js` to add or remove models from the UI dropdown. The first entry is the default.

---

## ML Training Loop

### Prepare data

Drop one or more CSVs into `C:\AI\data_raw`. They must contain the target column defined in `config.yaml` (default: `target`, integer-valued).

### Run a single stage (useful for debugging)

```powershell
python scripts\data_processor.py
python scripts\train.py
python scripts\eval.py
```

Each prints its output ID to stdout (e.g. `MODEL_ID=model_...`).

### Run the autonomous loop

```powershell
python scripts\orchestrator.py
```

Optional flags:
- `--max-iters N` — override `system.max_loop_iterations`
- `--rollback`    — revert `promoted.json` to the previous model and exit

The orchestrator writes a full report to `D:\AI\orchestration\<master_run_id>.json`.

### How the decision logic works

After each eval, the orchestrator compares the current score to the previous promoted score:

| Condition | Action |
|---|---|
| `score > prev + improvement_delta` | **promote** |
| `score < prev - improvement_delta` | **refine_data** (escalate cleaning; rollback after 2 consecutive regressions) |
| Within delta, `plateau_counter < tolerance` | hold |
| Within delta, `plateau_counter >= tolerance` | **tune_hyperparams** (advance sweep) |

All thresholds live in `config.yaml` under `orchestrator` and `evaluation`.

### Refinement levels

When the orchestrator signals regression, it writes a `refine_request_*.json` file into `D:\AI\cowork_ops\`. The data pipeline reads these signals on the next run and escalates cleaning based on how many are pending:

| Level | Operations |
|---|---|
| 0 (baseline) | dedupe, normalize formatting, schema validate |
| 1 (mild) | + drop rows with any NaN in features |
| 2 (moderate) | + remove outliers (>3σ on any numeric feature) |
| 3 (aggressive) | + drop low-variance features + undersample majority class |

The level is capped at 3. Once a signal is consumed it moves to `cowork_ops/processed/` so it never triggers twice.

Force a level manually:

```powershell
python scripts\data_processor.py --refinement-level 2
```

### Safety stops

The system halts (raising `SafetyViolation`) if:

1. `data_raw` fingerprint changes between pipeline start and end
2. A curated dataset manifest is missing
3. An evaluation produces an invalid JSON structure
4. A model checkpoint would be silently overwritten (archived instead)
5. Any stage fails more than `orchestrator.retry_on_failure` times

On halt, the current promoted model is untouched and the master run report records the halt reason.

### Swapping the model backend

In `config.yaml`:

```yaml
training:
  model_type: sklearn_logreg   # or pytorch_mlp
```

No code changes required.

### Extending

- **New data formats**: extend `data_processor.load_raw_files`
- **New model type**: add a branch in `train.train_*` and update `eval.load_model` / `eval.predict`
- **New drift metric**: add a detector to `eval.py` and fold it into `composite_score`
- **New refinement level**: add a function in `data_processor.py` and gate it on `level >= N`

### Reproducibility

Seeds are set for `random`, `numpy`, and `torch` (when used). The same raw data + config produces identical curated splits, models, and eval scores. Dataset fingerprints are stored in every manifest for lineage tracing.

# Project Atlas

Project Atlas is a **local-first AI workspace** for experimenting with two complementary systems: a persistent Ollama-backed chat assistant and an autonomous machine-learning refinement loop.

Everything runs on the local machine. Cloud services are not required for the core workflows.

> **Status:** Experimental but functional. Atlas is a lab for local AI orchestration, persistent assistant state, and repeatable model-training workflows.

## What’s Included

| System | Entry point | Purpose |
| --- | --- | --- |
| Local Chat | `start.bat` | FastAPI + Ollama chat UI with persistent SQLite history |
| ML Pipeline | `start_pipeline.bat` / `scripts/orchestrator.py` | Data → train → evaluate → decide refinement loop |

## Local Chat

The chat application provides:

- local Ollama model access;
- browser-based chat UI;
- persistent SQLite session history;
- configurable model registry through `models.js`;
- no required cloud inference dependency.

### Requirements

- Python environment from `requirements.txt`
- Ollama running locally
- At least one compatible model pulled into Ollama

### Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
.\start.bat
```

The browser UI runs locally and connects to the FastAPI backend.

## Autonomous ML Loop

Atlas also contains a repeatable training pipeline that processes raw data, trains a model, evaluates the result, and decides whether to promote, refine, or tune the next run.

Core properties:

- deterministic seeds for reproducibility;
- dataset fingerprints and lineage manifests;
- configurable refinement levels;
- promotion/rollback decisions based on evaluation thresholds;
- bounded retry behavior and explicit safety stops;
- support for the current scikit-learn and optional PyTorch model paths.

### Run the pipeline

```powershell
python scripts\orchestrator.py
```

Individual stages can also be run directly for debugging:

```powershell
python scripts\data_processor.py
python scripts\train.py
python scripts\eval.py
```

Configuration lives in `config/config.yaml`.

## Repository Layout

```text
main.py                 Local chat server
index.html              Browser chat UI
models.js               Ollama model registry
config/                  ML pipeline configuration
scripts/                 Processing, training, evaluation, orchestration
tests/                   Pipeline tests
data_raw/                Input datasets
```

Generated runtime artifacts are intentionally separated from source-controlled project files. Paths are configurable through `.env` and `config/config.yaml`.

## Safety and Reproducibility

The training loop is designed to halt instead of silently continuing when core assumptions break. Examples include unexpected raw-data changes, missing manifests, invalid evaluation output, overwrite risk, or repeated stage failure.

A promoted model is left untouched when the pipeline halts unexpectedly.

## Extending Atlas

Atlas is intentionally modular. Common extension points include:

- new dataset loaders or cleaning strategies;
- additional model backends;
- new evaluation/drift metrics;
- additional refinement policies;
- richer local-agent capabilities layered over the existing chat runtime.

Keep experiments local-first, observable, and reversible. New automation should preserve explicit boundaries around filesystem access, model execution, and durable state.

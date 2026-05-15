# Logging Format Specification

All logs are written as **JSON Lines** (one JSON object per line).

ML pipeline components write to:
```
D:\AI\logs\<component>.log
```

Components: `data_processor`, `train`, `eval`, `orchestrator`.

The chat server (`chat_server`) logs to **stdout** in the same JSON format.

## Schema

Every log line conforms to:

```json
{
  "ts":        "2026-04-19T18:30:00.123456+00:00",
  "level":     "INFO",
  "component": "orchestrator",
  "message":   "loop iteration 1/10",
  "run_id":    "orch_20260419T183000Z_a1b2c3",
  "stage":     "train",
  "event":     "iter_start",
  "score":     82.4,
  "duration_s": 3.14
}
```

### Required fields
| Field       | Type   | Meaning |
|-------------|--------|---------|
| `ts`        | string | UTC ISO-8601 timestamp |
| `level`     | string | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` |
| `component` | string | Logger name (module) |
| `message`   | string | Human-readable message |

### Optional fields (attached via `extra=`)
| Field        | Type    | Meaning |
|--------------|---------|---------|
| `run_id`     | string  | Unique ID for the run producing the line |
| `stage`      | string  | `data` \| `train` \| `eval` \| `decide` |
| `event`      | string  | Canonical event name (e.g. `promote`, `halt`, `retry`) |
| `score`      | number  | Evaluation score (0–100) |
| `duration_s` | number  | Wall-clock seconds for the operation |
| `exception`  | string  | Full traceback when an exception is attached |

## Artifact manifests (JSON sidecars, not logs)

| Path | Purpose |
|------|---------|
| `data_curated/<dataset_id>/manifest.json` | Dataset lineage, row counts, raw fingerprint |
| `models/<model_id>/manifest.json`         | Model type, hyperparameters, checkpoint path |
| `training_runs/<run_id>/manifest.json`    | Training run metadata (mirror of model manifest) |
| `evals/<eval_id>.json`                    | Metrics, drift detections, score, previous comparison |
| `orchestration/<master_run_id>.json`      | Full loop report with per-iteration decisions |
| `orchestration/promoted.json`             | Current promoted model pointer (+ previous for rollback) |
| `cowork_ops/data_ops_<run_id>.json`       | Data transformation op-log |
| `cowork_ops/refine_request_<iter_id>.json`| Signal written when regression triggers data refinement |

## Retention

`config.yaml -> logging.retain_runs` controls how many eval JSONs and training run folders
are kept. Cleanup is not automatic in this build — add a scheduled task or cron job
that deletes the oldest manifests beyond the retention limit.

## Reading logs

On Windows PowerShell:
```powershell
Get-Content D:\AI\logs\orchestrator.log -Wait | ConvertFrom-Json
```

Any JSON-aware tool (jq, Python, Pandas) will parse these directly.

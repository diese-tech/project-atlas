"""
Shared utilities for the ML platform.
Provides: path resolution, structured logging, run IDs, manifest I/O.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------
# CONFIG LOADING
# ---------------------------------------------------------------

def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load the YAML config. Defaults to <base>/config/config.yaml."""
    if config_path is None:
        # Look for config relative to this file
        here = Path(__file__).resolve().parent
        config_path = here.parent / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_base_dir(cfg: dict[str, Any]) -> Path:
    """Return the base directory as a pathlib.Path."""
    return Path(cfg["system"]["base_dir"])


def get_path(cfg: dict[str, Any], key: str) -> Path:
    """
    Resolve a directory from config to an absolute Path.
    Looks up `data.<key>_dir` or `logging.<key>_dir` style keys.
    """
    base = get_base_dir(cfg)
    mapping = {
        "raw": cfg["data"]["raw_dir"],
        "clean": cfg["data"]["clean_dir"],
        "curated": cfg["data"]["curated_dir"],
        "models": "models",
        "training_runs": "training_runs",
        "evals": "evals",
        "scripts": "scripts",
        "orchestration": "orchestration",
        "cowork_ops": cfg["logging"]["cowork_ops_dir"],
        "logs": cfg["logging"]["log_dir"],
    }
    if key not in mapping:
        raise KeyError(f"Unknown path key: {key}")
    return base / mapping[key]


def ensure_dirs(cfg: dict[str, Any]) -> None:
    """Create all required directories if missing."""
    for key in ["raw", "clean", "curated", "models", "training_runs",
                "evals", "scripts", "orchestration", "cowork_ops", "logs"]:
        get_path(cfg, key).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------
# RUN IDS & VERSIONING
# ---------------------------------------------------------------

def new_run_id(prefix: str = "run") -> str:
    """Generate a timestamped, unique run ID."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    short = uuid.uuid4().hex[:6]
    return f"{prefix}_{ts}_{short}"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    """Compute SHA-256 of a file — used for raw data integrity checks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_fingerprint(directory: Path) -> dict[str, str]:
    """
    Build a {filename: sha256} map for all files in a directory.
    Used to detect raw-data tampering.
    """
    if not directory.exists():
        return {}
    fp: dict[str, str] = {}
    for p in sorted(directory.rglob("*")):
        if p.is_file():
            fp[str(p.relative_to(directory))] = file_sha256(p)
    return fp


# ---------------------------------------------------------------
# MANIFEST I/O
# ---------------------------------------------------------------

def write_manifest(path: Path, data: dict[str, Any]) -> None:
    """Write a JSON manifest atomically (no silent overwrite)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        # Never silently overwrite — archive first
        archive = path.with_suffix(path.suffix + f".bak_{uuid.uuid4().hex[:6]}")
        path.rename(archive)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    tmp.replace(path)


def read_manifest(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------

class JsonFormatter(logging.Formatter):
    """Structured JSON log line — matches the format spec."""
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "component": record.name,
            "message": record.getMessage(),
        }
        # Attach any extra fields
        for k, v in record.__dict__.items():
            if k in ("run_id", "stage", "event", "score", "duration_s"):
                payload[k] = v
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def get_logger(name: str, cfg: dict[str, Any], run_id: str | None = None) -> logging.Logger:
    """
    Return a configured logger. Writes JSON lines to <logs>/<name>.log
    and also to stderr for visibility.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    level = getattr(logging, cfg["logging"]["level"].upper(), logging.INFO)
    logger.setLevel(level)

    log_dir = get_path(cfg, "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name}.log"

    fmt = JsonFormatter() if cfg["logging"]["format"] == "json" else logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if run_id:
        logger = logging.LoggerAdapter(logger, {"run_id": run_id})  # type: ignore

    return logger


# ---------------------------------------------------------------
# SAFETY HALT
# ---------------------------------------------------------------

class SafetyViolation(Exception):
    """Raised when a safety rule is violated. Orchestrator must halt."""


def halt(logger: logging.Logger, reason: str) -> None:
    """Log a halt and raise SafetyViolation."""
    logger.error(f"SAFETY HALT: {reason}")
    raise SafetyViolation(reason)

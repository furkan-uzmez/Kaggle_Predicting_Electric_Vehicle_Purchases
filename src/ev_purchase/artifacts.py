import json
import math
import os
import re
from pathlib import Path
from typing import Any
from uuid import uuid4

MANDATORY_METRIC_KEYS = {
    "run_id",
    "status",
    "config_sha256",
    "data_sha256",
    "split_sha256",
    "model",
    "seed",
    "fold_metrics",
    "aggregate_metrics",
    "runtime_seconds",
    "artifact_paths",
}

SHA256_HEX_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


def validate_relative_path(path_str: str) -> None:
    """Validate that path_str is a non-empty relative path without '..' traversal."""
    if not isinstance(path_str, str) or not path_str.strip():
        raise ValueError("Artifact path must be a non-empty string")

    p = Path(path_str)
    if p.is_absolute() or os.path.isabs(path_str):
        raise ValueError(f"Artifact path must be relative, got absolute path: {path_str}")

    if ".." in p.parts:
        raise ValueError(f"Artifact path must not contain '..' traversal: {path_str}")


def _validate_numeric_finite(val: Any, val_name: str) -> float:
    """Validate that val is a finite number (int or float, not bool)."""
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise TypeError(f"Value '{val_name}' must be numeric (int/float), got {type(val).__name__}")
    f_val = float(val)
    if not math.isfinite(f_val):
        raise ValueError(f"Value '{val_name}' must be finite, got {val}")
    return f_val


def validate_run_metrics(payload: dict[str, Any]) -> None:
    """Validate run metrics payload contract and structure."""
    if not isinstance(payload, dict):
        raise TypeError("Payload must be a dictionary")

    missing = MANDATORY_METRIC_KEYS - set(payload.keys())
    if missing:
        raise ValueError(f"Missing mandatory metric keys: {sorted(missing)}")

    # run_id
    run_id = payload["run_id"]
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a non-empty string")

    # status
    status = payload["status"]
    if not isinstance(status, str) or not status.strip():
        raise ValueError("status must be a non-empty string")

    # SHA256 hashes
    for sha_key in ["config_sha256", "data_sha256", "split_sha256"]:
        val = payload[sha_key]
        if not isinstance(val, str) or not SHA256_HEX_PATTERN.match(val):
            raise ValueError(
                f"{sha_key} must be a 64-character hexadecimal SHA-256 string, got {val!r}"
            )

    # model
    model = payload["model"]
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty string")

    # seed
    seed = payload["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be an integer, got {type(seed).__name__}")

    # runtime_seconds
    runtime = payload["runtime_seconds"]
    f_runtime = _validate_numeric_finite(runtime, "runtime_seconds")
    if f_runtime < 0:
        raise ValueError(f"runtime_seconds must be non-negative, got {f_runtime}")

    # fold_metrics
    fold_metrics = payload["fold_metrics"]
    if not isinstance(fold_metrics, (list, tuple, dict)):
        raise TypeError("fold_metrics must be a list, tuple, or dict")
    if isinstance(fold_metrics, (list, tuple)):
        for idx, item in enumerate(fold_metrics):
            if not isinstance(item, dict):
                raise TypeError(f"fold_metrics[{idx}] must be a dict")
            for k, v in item.items():
                _validate_numeric_finite(v, f"fold_metrics[{idx}][{k}]")
    elif isinstance(fold_metrics, dict):
        for fold_key, item in fold_metrics.items():
            if not isinstance(item, dict):
                raise TypeError(f"fold_metrics[{fold_key!r}] must be a dict")
            for k, v in item.items():
                _validate_numeric_finite(v, f"fold_metrics[{fold_key!r}][{k}]")

    # aggregate_metrics
    agg_metrics = payload["aggregate_metrics"]
    if not isinstance(agg_metrics, dict):
        raise TypeError("aggregate_metrics must be a dict")
    for k, v in agg_metrics.items():
        _validate_numeric_finite(v, f"aggregate_metrics[{k}]")

    # artifact_paths
    art_paths = payload["artifact_paths"]
    if not isinstance(art_paths, dict):
        raise TypeError("artifact_paths must be a dict mapping names to relative path strings")
    for _k, p in art_paths.items():
        validate_relative_path(p)

    # Any top-level keys ending with '_path'
    for k, v in payload.items():
        if k.endswith("_path") and k != "artifact_paths":
            validate_relative_path(str(v))

    # JSON serializability check with strict allow_nan=False
    try:
        json.dumps(payload, allow_nan=False)
    except (TypeError, OverflowError, ValueError) as exc:
        raise ValueError(f"Payload is not strictly JSON serializable: {exc}") from exc


def save_run_metrics(payload: dict[str, Any], output_path: str | Path) -> Path:
    """Validate and atomically save run metrics JSON payload."""
    validate_run_metrics(payload)

    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = target_path.with_name(f".{target_path.name}.tmp_{uuid4().hex}")
    try:
        formatted_json = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(formatted_json)

        os.replace(temp_path, target_path)
    except Exception:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise

    return target_path


def load_run_metrics(input_path: str | Path) -> dict[str, Any]:
    """Load and validate run metrics JSON payload from file."""
    p = Path(input_path)
    if not p.is_file():
        raise FileNotFoundError(f"Metrics file not found: {p}")

    with open(p, encoding="utf-8") as f:
        data = json.load(f)

    validate_run_metrics(data)
    return data

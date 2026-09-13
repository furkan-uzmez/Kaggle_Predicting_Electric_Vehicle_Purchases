"""Run metric analysis and comparability verification module."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ev_purchase.artifacts import load_run_metrics


def _extract_fold_count(run: dict[str, Any]) -> int | None:
    if "effective_n_splits" in run:
        return int(run["effective_n_splits"])
    fold_metrics = run.get("fold_metrics")
    if isinstance(fold_metrics, list) and fold_metrics:
        folds = {item["fold"] for item in fold_metrics if isinstance(item, dict) and "fold" in item}
        if folds:
            return len(folds)
    return None


def _extract_repeat_count(run: dict[str, Any]) -> int | None:
    if "effective_n_repeats" in run:
        return int(run["effective_n_repeats"])
    fold_metrics = run.get("fold_metrics")
    if isinstance(fold_metrics, list) and fold_metrics:
        repeats = {
            item["repeat"] for item in fold_metrics if isinstance(item, dict) and "repeat" in item
        }
        if repeats:
            return len(repeats)
    return None


def _extract_target(run: dict[str, Any]) -> str | None:
    if "target" in run:
        return str(run["target"])
    if "target_column" in run:
        return str(run["target_column"])
    return None


def assert_comparable_runs(runs: Sequence[dict[str, Any]]) -> None:
    """Verify that a sequence of run metric payloads are comparable.

    Requires identical dataset hash (data_sha256), split hash (split_sha256),
    target, metric implementation version (or metric key set), fold count,
    repeat count, and smoke execution status.

    Note on metric implementation version:
        Currently, training.py payloads do not record an explicit 'metrics_version' field.
        Comparability falls back to matching the key set of aggregate_metrics unless
        an explicit 'metrics_version' is present in the payload.

    Args:
        runs: Sequence of run metric dictionary payloads.

    Raises:
        ValueError: If runs list is empty or if any run differs in comparability criteria.
    """
    if not runs:
        raise ValueError("Cannot compare empty list of runs")

    if len(runs) == 1:
        return

    ref = runs[0]
    ref_run_id = ref.get("run_id", "run[0]")

    ref_data_sha = ref.get("data_sha256")
    ref_split_sha = ref.get("split_sha256")
    ref_smoke = bool(ref.get("smoke", False))
    ref_fold_count = _extract_fold_count(ref)
    ref_repeat_count = _extract_repeat_count(ref)
    ref_target = _extract_target(ref)
    ref_metrics_ver = ref.get("metrics_version")
    ref_metric_keys = (
        set(ref.get("aggregate_metrics", {}).keys())
        if isinstance(ref.get("aggregate_metrics"), dict)
        else set()
    )

    for idx, run in enumerate(runs[1:], start=1):
        run_id = run.get("run_id", f"run[{idx}]")

        run_data_sha = run.get("data_sha256")
        if run_data_sha != ref_data_sha:
            raise ValueError(
                f"Inconsistent data_sha256 between {ref_run_id} ({ref_data_sha}) "
                f"and {run_id} ({run_data_sha})"
            )

        run_split_sha = run.get("split_sha256")
        if run_split_sha != ref_split_sha:
            raise ValueError(
                f"Inconsistent split_sha256 between {ref_run_id} ({ref_split_sha}) "
                f"and {run_id} ({run_split_sha})"
            )

        run_smoke = bool(run.get("smoke", False))
        if run_smoke != ref_smoke:
            raise ValueError(
                f"Cannot mix smoke ({run_smoke}) and non-smoke ({ref_smoke}) runs "
                f"between {ref_run_id} and {run_id}"
            )

        run_fold_count = _extract_fold_count(run)
        if run_fold_count != ref_fold_count:
            raise ValueError(
                f"Inconsistent fold count (effective_n_splits) between {ref_run_id} "
                f"({ref_fold_count}) and {run_id} ({run_fold_count})"
            )

        run_repeat_count = _extract_repeat_count(run)
        if run_repeat_count != ref_repeat_count:
            raise ValueError(
                f"Inconsistent repeat count (effective_n_repeats) between {ref_run_id} "
                f"({ref_repeat_count}) and {run_id} ({run_repeat_count})"
            )

        run_target = _extract_target(run)
        if run_target != ref_target:
            raise ValueError(
                f"Inconsistent target between {ref_run_id} ({ref_target}) "
                f"and {run_id} ({run_target})"
            )

        run_metrics_ver = run.get("metrics_version")
        if ref_metrics_ver is not None or run_metrics_ver is not None:
            if run_metrics_ver != ref_metrics_ver:
                raise ValueError(
                    f"Inconsistent metrics_version between {ref_run_id} ({ref_metrics_ver}) "
                    f"and {run_id} ({run_metrics_ver})"
                )
        else:
            run_metric_keys = (
                set(run.get("aggregate_metrics", {}).keys())
                if isinstance(run.get("aggregate_metrics"), dict)
                else set()
            )
            if run_metric_keys != ref_metric_keys:
                raise ValueError(
                    f"Inconsistent aggregate_metrics keys between {ref_run_id} "
                    f"({ref_metric_keys}) and {run_id} ({run_metric_keys})"
                )


def load_all_runs(runs_dir: str | Path = "outputs/runs") -> list[dict[str, Any]]:
    """Load and validate all metrics.json run payloads from a runs directory.

    Args:
        runs_dir: Path to directory containing run subdirectories.

    Returns:
        List of validated run metric dictionary payloads.
    """
    p = Path(runs_dir)
    if not p.is_dir():
        return []

    runs: list[dict[str, Any]] = []
    for metrics_file in sorted(p.glob("*/metrics.json")):
        payload = load_run_metrics(metrics_file)
        runs.append(payload)

    return runs

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    seed: int


@dataclass(frozen=True)
class DataConfig:
    train: Path
    test: Path
    sample_submission: Path
    id_column: str
    target_column: str


@dataclass(frozen=True)
class ValidationConfig:
    strategy: str
    n_splits: int
    n_repeats: int
    random_state: int


@dataclass(frozen=True)
class MetricsConfig:
    primary: str
    guardrails: tuple[str, ...]


@dataclass(frozen=True)
class OutputsConfig:
    root: Path


@dataclass(frozen=True)
class Config:
    project: ProjectConfig
    data: DataConfig
    validation: ValidationConfig
    metrics: MetricsConfig
    outputs: OutputsConfig


def _resolve_and_validate_path(path_val: Any, project_root: Path, field_name: str) -> Path:
    if not isinstance(path_val, (str, Path)):
        raise ValueError("path fields must be string or Path")
    if isinstance(path_val, str) and not path_val.strip():
        raise ValueError(f"{field_name} path must be a non-empty string")
    p = Path(path_val)
    if p.is_absolute():
        raise ValueError("Configured paths must be relative")
    resolved = (project_root / p).resolve()
    root_resolved = project_root.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError:
        raise ValueError("Configured path escapes project root") from None
    return resolved


def load_config(config_path: Path | str, project_root: Path | str | None = None) -> Config:
    """Load and validate typed YAML configuration with root-relative path resolution."""
    if project_root is None:
        raise ValueError("project_root must be explicitly provided")

    config_path = Path(config_path)
    root_path = Path(project_root)

    with config_path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError("Invalid configuration format")

    required_sections = {"project", "data", "validation", "metrics", "outputs"}
    missing = required_sections - raw.keys()
    if missing:
        raise ValueError(f"missing required section: {', '.join(sorted(missing))}")

    for sec in ["project", "data", "validation", "metrics", "outputs"]:
        if not isinstance(raw[sec], Mapping):
            raise ValueError(f"section '{sec}' must be a mapping")

    proj_dict: Mapping[str, Any] = raw["project"]
    data_dict: Mapping[str, Any] = raw["data"]
    val_dict: Mapping[str, Any] = raw["validation"]
    met_dict: Mapping[str, Any] = raw["metrics"]
    out_dict: Mapping[str, Any] = raw["outputs"]

    seed = proj_dict.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")

    id_col = data_dict.get("id_column")
    target_col = data_dict.get("target_column")
    if (
        not isinstance(id_col, str)
        or not id_col.strip()
        or not isinstance(target_col, str)
        or not target_col.strip()
    ):
        raise ValueError("id_column and target_column must be non-empty strings")

    strategy = val_dict.get("strategy")
    if not isinstance(strategy, str) or strategy != "repeated_stratified_kfold":
        raise ValueError("invalid validation strategy")

    n_splits = val_dict.get("n_splits")
    n_repeats = val_dict.get("n_repeats")
    random_state = val_dict.get("random_state")

    invalid_cv = (
        not isinstance(n_splits, int)
        or isinstance(n_splits, bool)
        or n_splits < 2
        or not isinstance(n_repeats, int)
        or isinstance(n_repeats, bool)
        or n_repeats < 1
    )
    if invalid_cv:
        raise ValueError("n_splits must be integer >= 2 and n_repeats must be integer >= 1")

    if not isinstance(random_state, int) or isinstance(random_state, bool):
        raise ValueError("random_state must be an integer")

    primary = met_dict.get("primary")
    guardrails = met_dict.get("guardrails")
    if not isinstance(primary, str) or not primary.strip():
        raise ValueError("primary metric must be a non-empty string")
    if (
        not isinstance(guardrails, list)
        or len(guardrails) == 0
        or not all(isinstance(g, str) and g.strip() for g in guardrails)
    ):
        raise ValueError("guardrails must be a list of non-empty strings")

    project = ProjectConfig(seed=seed)
    data = DataConfig(
        train=_resolve_and_validate_path(data_dict.get("train"), root_path, "train"),
        test=_resolve_and_validate_path(data_dict.get("test"), root_path, "test"),
        sample_submission=_resolve_and_validate_path(
            data_dict.get("sample_submission"), root_path, "sample_submission"
        ),
        id_column=id_col,
        target_column=target_col,
    )
    validation = ValidationConfig(
        strategy=strategy,
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    metrics = MetricsConfig(
        primary=primary,
        guardrails=tuple(guardrails),
    )
    outputs = OutputsConfig(
        root=_resolve_and_validate_path(out_dict.get("root"), root_path, "root"),
    )

    return Config(
        project=project,
        data=data,
        validation=validation,
        metrics=metrics,
        outputs=outputs,
    )

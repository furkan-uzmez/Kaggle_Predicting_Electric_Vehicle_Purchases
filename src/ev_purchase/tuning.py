"""Bounded hyperparameter optimization engine using Optuna and multi-fidelity protocol."""

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import yaml
from optuna.trial import FrozenTrial, TrialState

from ev_purchase.config import load_config
from ev_purchase.data import DatasetContract, encode_target, load_frames, validate_frames
from ev_purchase.features import apply_feature_policy
from ev_purchase.metrics import classification_metrics, validate_probabilities
from ev_purchase.models import build_model
from ev_purchase.splits import build_split_manifest, validate_split_manifest


@dataclass(frozen=True)
class FidelityPlan:
    search_fold_ids: tuple[int, ...]
    finalist_fold_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        s_folds = tuple(self.search_fold_ids)
        f_folds = tuple(self.finalist_fold_ids)

        object.__setattr__(self, "search_fold_ids", s_folds)
        object.__setattr__(self, "finalist_fold_ids", f_folds)

        self._validate_folds(s_folds, "search_fold_ids")
        self._validate_folds(f_folds, "finalist_fold_ids")

        if not set(s_folds).issubset(set(f_folds)):
            raise ValueError("search_fold_ids must be a subset of finalist_fold_ids")

    @staticmethod
    def _validate_folds(folds: tuple[int, ...], name: str) -> None:
        if not folds:
            raise ValueError(f"{name} must not be empty")
        for f in folds:
            if isinstance(f, bool) or not isinstance(f, int):
                raise TypeError(f"{name} elements must be integers, got {type(f).__name__}")
            if f < 0:
                raise ValueError(f"{name} elements must be non-negative, got {f}")
        if len(folds) != len(set(folds)):
            raise ValueError(f"{name} must contain unique fold IDs")


@dataclass(frozen=True)
class HPOConfig:
    n_trials: int
    timeout_seconds: int
    search_fold_ids: tuple[int, ...]
    finalist_fold_ids: tuple[int, ...]
    finalist_count: int
    sampler_seed: int
    pruner: str
    plateau_patience_trials: int | None = None
    roc_auc_guardrail_floor: float | None = None
    pr_auc_guardrail_floor: float | None = None
    log_loss_guardrail_ceiling: float | None = None

    def __post_init__(self) -> None:
        if self.pruner.lower() not in ("median", "none"):
            raise ValueError(f"Invalid pruner '{self.pruner}'. Must be 'median' or 'none'")
        if self.n_trials <= 0:
            raise ValueError("n_trials must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.finalist_count <= 0:
            raise ValueError("finalist_count must be positive")
        if self.plateau_patience_trials is not None and self.plateau_patience_trials <= 0:
            raise ValueError("plateau_patience_trials must be positive if specified")
        if self.roc_auc_guardrail_floor is not None and not (
            0.0 < self.roc_auc_guardrail_floor < 1.0
        ):
            raise ValueError("roc_auc_guardrail_floor must be between 0.0 and 1.0 if specified")
        if self.pr_auc_guardrail_floor is not None and not (
            0.0 < self.pr_auc_guardrail_floor < 1.0
        ):
            raise ValueError("pr_auc_guardrail_floor must be between 0.0 and 1.0 if specified")
        if self.log_loss_guardrail_ceiling is not None and self.log_loss_guardrail_ceiling <= 0.0:
            raise ValueError("log_loss_guardrail_ceiling must be positive if specified")


@dataclass(frozen=True)
class TuningConfig:
    model_name: str
    baseline_config: Path
    split_manifest: Path
    study_name: str
    hpo: HPOConfig


def _compute_file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def _compute_combined_sha256(paths: list[Path]) -> str:
    hasher = hashlib.sha256()
    for path in sorted(paths):
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
    return hasher.hexdigest()


def _compute_search_space_fingerprint(model_name: str) -> str:
    """Compute deterministic SHA-256 fingerprint for model parameter search space specification."""
    clean_name = model_name.strip().lower()
    specs = {
        "catboost": {
            "iterations": ("int", 200, 1000, 50),
            "depth": ("int", 4, 8),
            "learning_rate": ("float", 0.01, 0.2, "log"),
            "l2_leaf_reg": ("float", 1.0, 10.0, "log"),
            "subsample": ("float", 0.6, 1.0),
            "random_strength": ("float", 1e-3, 10.0, "log"),
        },
        "lightgbm": {
            "n_estimators": ("int", 200, 1000, 50),
            "learning_rate": ("float", 0.01, 0.2, "log"),
            "num_leaves": ("int", 15, 127),
            "max_depth": ("int", 3, 10),
            "min_child_samples": ("int", 10, 100),
            "subsample": ("float", 0.6, 1.0),
            "colsample_bytree": ("float", 0.6, 1.0),
            "reg_alpha": ("float", 1e-8, 10.0, "log"),
            "reg_lambda": ("float", 1e-8, 10.0, "log"),
        },
        "dummy": {},
    }
    space_spec = specs.get(clean_name, {"name": clean_name})
    spec_bytes = json.dumps(space_spec, sort_keys=True).encode("utf-8")
    return hashlib.sha256(spec_bytes).hexdigest()


def _canonicalize_provenance(prov: dict[str, Any]) -> dict[str, Any]:
    return {
        "config_sha256": str(prov.get("config_sha256", "")),
        "data_sha256": str(prov.get("data_sha256", "")),
        "split_sha256": str(prov.get("split_sha256", "")),
        "model_name": str(prov.get("model_name", "")),
        "smoke": bool(prov.get("smoke", False)),
        "effective_n_rows": int(prov.get("effective_n_rows", 0)),
        "effective_search_fold_ids": sorted(
            int(x) for x in prov.get("effective_search_fold_ids", [])
        ),
        "effective_finalist_fold_ids": sorted(
            int(x) for x in prov.get("effective_finalist_fold_ids", [])
        ),
        "sampler_seed": int(prov.get("sampler_seed", 0)),
        "search_space_fingerprint": str(prov.get("search_space_fingerprint", "")),
        "search_space_version": str(prov.get("search_space_version", "1.0")),
    }


def get_study_system_attrs(study: optuna.Study) -> dict[str, Any]:
    """Helper to read study system attributes from Optuna storage."""
    return study._storage.get_study_system_attrs(study._study_id)


def set_study_system_attr(study: optuna.Study, key: str, value: Any) -> None:
    """Helper to set study system attribute in Optuna storage."""
    study._storage.set_study_system_attr(study._study_id, key, value)


def load_tuning_config(
    config_path: Path | str, project_root: Path | str | None = None
) -> TuningConfig:
    cfg_p = Path(config_path)
    root_p = Path(project_root) if project_root is not None else Path.cwd()

    if not cfg_p.is_absolute():
        cfg_p = (root_p / cfg_p).resolve()

    with open(cfg_p, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError("Tuning config must be a dictionary")

    model_name = str(raw.get("model_name", "")).strip()
    if not model_name:
        raise ValueError("model_name is required in tuning config")

    base_cfg_str = str(raw.get("baseline_config", "")).strip()
    split_man_str = str(raw.get("split_manifest", "")).strip()
    study_name = str(raw.get("study_name", "")).strip()

    if not base_cfg_str or not split_man_str or not study_name:
        raise ValueError("baseline_config, split_manifest, and study_name are required")

    base_cfg = Path(base_cfg_str)
    if not base_cfg.is_absolute():
        base_cfg = (root_p / base_cfg).resolve()

    split_man = Path(split_man_str)
    if not split_man.is_absolute():
        split_man = (root_p / split_man).resolve()

    hpo_dict = raw.get("hpo", {})
    if not isinstance(hpo_dict, dict):
        raise ValueError("hpo section must be a dictionary")

    s_folds = tuple(hpo_dict.get("search_fold_ids", []))
    f_folds = tuple(hpo_dict.get("finalist_fold_ids", []))

    plan = FidelityPlan(search_fold_ids=s_folds, finalist_fold_ids=f_folds)

    patience = hpo_dict.get("plateau_patience_trials")
    flr_roc = hpo_dict.get("roc_auc_guardrail_floor")
    flr_pr = hpo_dict.get("pr_auc_guardrail_floor")
    ceil_log = hpo_dict.get("log_loss_guardrail_ceiling")

    hpo_config = HPOConfig(
        n_trials=int(hpo_dict.get("n_trials", 40)),
        timeout_seconds=int(hpo_dict.get("timeout_seconds", 10800)),
        search_fold_ids=plan.search_fold_ids,
        finalist_fold_ids=plan.finalist_fold_ids,
        finalist_count=int(hpo_dict.get("finalist_count", 3)),
        sampler_seed=int(hpo_dict.get("sampler_seed", 42)),
        pruner=str(hpo_dict.get("pruner", "median")),
        plateau_patience_trials=int(patience) if patience is not None else None,
        roc_auc_guardrail_floor=float(flr_roc) if flr_roc is not None else None,
        pr_auc_guardrail_floor=float(flr_pr) if flr_pr is not None else None,
        log_loss_guardrail_ceiling=float(ceil_log) if ceil_log is not None else None,
    )

    return TuningConfig(
        model_name=model_name,
        baseline_config=base_cfg,
        split_manifest=split_man,
        study_name=study_name,
        hpo=hpo_config,
    )


def map_flat_folds(manifest: pd.DataFrame) -> dict[int, tuple[int, int]]:
    """Map flat fold IDs 0..N-1 deterministically to sorted (repeat, fold) tuples."""
    grouped = sorted(manifest.groupby(["repeat", "fold"]).groups.keys())
    return {idx: (r, f) for idx, (r, f) in enumerate(grouped)}


def sample_params(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    clean_name = model_name.strip().lower()
    if clean_name == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 200, 1000, step=50),
            "depth": trial.suggest_int("depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
            "thread_count": 4,
        }
    if clean_name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "n_jobs": 4,
        }
    if clean_name == "dummy":
        return {}

    raise ValueError(f"Unsupported model family for tuning: {model_name}")


def evaluate_model_folds(
    X_train: pd.DataFrame,
    y_encoded: np.ndarray,
    manifest: pd.DataFrame,
    flat_folds_map: dict[int, tuple[int, int]],
    fold_ids: tuple[int, ...],
    model_name: str,
    params: dict[str, Any],
    base_seed: int = 42,
    trial: optuna.Trial | None = None,
    fit_callback: Callable[[int, np.ndarray, np.ndarray], None] | None = None,
) -> dict[str, Any]:
    all_indices = np.arange(len(X_train))
    fold_metrics: list[dict[str, float | int]] = []
    roc_aucs: list[float] = []
    pr_aucs: list[float] = []
    log_losses: list[float] = []
    completed_fold_ids: list[int] = []

    for idx, fold_id in enumerate(fold_ids):
        if fold_id not in flat_folds_map:
            raise ValueError(
                f"Fold ID {fold_id} is out of range for manifest (max {len(flat_folds_map) - 1})"
            )

        repeat, fold = flat_folds_map[fold_id]
        fold_mask = (manifest["repeat"] == repeat) & (manifest["fold"] == fold)
        val_idx = manifest.loc[fold_mask, "row_index"].to_numpy()
        train_idx = np.setdiff1d(all_indices, val_idx)

        # Exact complement assertion
        expected_train_idx = np.setdiff1d(all_indices, val_idx)
        if not np.array_equal(train_idx, expected_train_idx):
            raise AssertionError("Train indices must strictly complement validation indices")

        if fit_callback is not None:
            fit_callback(fold_id, train_idx, val_idx)

        X_tr = X_train.iloc[train_idx]
        y_tr = y_encoded[train_idx]
        X_va = X_train.iloc[val_idx]
        y_va = y_encoded[val_idx]

        fold_seed = base_seed + int(repeat) * 100 + int(fold)
        fold_start_time = time.time()
        model = build_model(model_name=model_name, seed=fold_seed, sample_frame=X_tr, params=params)
        model.fit(X_tr, y_tr)

        val_p = model.predict_proba(X_va)[:, 1]
        validate_probabilities(val_p)

        metrics = classification_metrics(y_va, val_p)
        fold_runtime = time.time() - fold_start_time

        fold_record: dict[str, float | int] = {
            "fold_id": fold_id,
            "repeat": repeat,
            "fold": fold,
            "runtime_seconds": round(fold_runtime, 4),
            **metrics,
        }
        fold_metrics.append(fold_record)
        completed_fold_ids.append(fold_id)

        # Incremental trial user_attrs update after EVERY successful fold
        if trial is not None:
            trial.set_user_attr("evaluated_fold_ids", list(completed_fold_ids))
            trial.set_user_attr("fold_metrics", fold_metrics)

        roc_aucs.append(metrics["roc_auc"])
        pr_aucs.append(metrics["average_precision"])
        log_losses.append(metrics["log_loss"])

        if trial is not None:
            intermediate_val = float(np.mean(roc_aucs))
            trial.report(intermediate_val, step=idx)
            if trial.should_prune():
                msg = f"Pruned at fold step {idx} with mean ROC-AUC {intermediate_val:.4f}"
                raise optuna.TrialPruned(msg)

    return {
        "mean_roc_auc": float(np.mean(roc_aucs)),
        "std_roc_auc": float(np.std(roc_aucs)),
        "mean_average_precision": float(np.mean(pr_aucs)),
        "std_average_precision": float(np.std(pr_aucs)),
        "mean_log_loss": float(np.mean(log_losses)),
        "std_log_loss": float(np.std(log_losses)),
        "fold_metrics": fold_metrics,
        "evaluated_fold_ids": tuple(completed_fold_ids),
    }


class OptunaStoppingCallback:
    """True Optuna study stopping callback using study.stop()."""

    def __init__(self, hpo_config: HPOConfig, existing_trials: list[FrozenTrial]):
        self.cfg = hpo_config
        self.stop_reason = "Completed target trial budget or timeout"
        self.stopped_early = False
        self.best_search_score = -1.0
        self.no_improve_trials = 0

        # Reconstruct plateau history deterministically from complete trial history
        for t in sorted(existing_trials, key=lambda x: x.number):
            if t.state == TrialState.COMPLETE and t.value is not None:
                if t.value >= self.best_search_score + 1e-4:
                    self.best_search_score = t.value
                    self.no_improve_trials = 0
                else:
                    self.no_improve_trials += 1

                # Check if reconstructed history already violates patience
                if (
                    self.cfg.plateau_patience_trials is not None
                    and self.no_improve_trials >= self.cfg.plateau_patience_trials
                ):
                    self.stopped_early = True
                    pat = self.cfg.plateau_patience_trials
                    self.stop_reason = (
                        f"Plateau reached: no improvement >= 1e-4 in {pat} consecutive trials"
                    )

    def __call__(self, study: optuna.Study, trial: FrozenTrial) -> None:
        if trial.state != TrialState.COMPLETE or trial.value is None:
            return

        val = trial.value

        # 1. ROC-AUC Floor
        flr_roc = self.cfg.roc_auc_guardrail_floor
        if flr_roc is not None and val < flr_roc:
            self.stopped_early = True
            self.stop_reason = (
                f"Guardrail ROC-AUC floor violated: trial ROC-AUC {val:.4f} < floor {flr_roc}"
            )
            set_study_system_attr(study, "stop_reason", self.stop_reason)
            set_study_system_attr(study, "stopped_early", True)
            study.stop()
            return

        # 2. PR-AUC Floor
        flr_pr = self.cfg.pr_auc_guardrail_floor
        pr_val = trial.user_attrs.get("pr_auc_mean")
        if flr_pr is not None and pr_val is not None and pr_val < flr_pr:
            self.stopped_early = True
            self.stop_reason = (
                f"Guardrail PR-AUC floor violated: trial PR-AUC {pr_val:.4f} < floor {flr_pr}"
            )
            set_study_system_attr(study, "stop_reason", self.stop_reason)
            set_study_system_attr(study, "stopped_early", True)
            study.stop()
            return

        # 3. LogLoss Ceiling
        ceil_log = self.cfg.log_loss_guardrail_ceiling
        log_val = trial.user_attrs.get("log_loss_mean")
        if ceil_log is not None and log_val is not None and log_val > ceil_log:
            self.stopped_early = True
            self.stop_reason = (
                f"Guardrail LogLoss ceiling violated: {log_val:.4f} > ceiling {ceil_log:.4f}"
            )
            set_study_system_attr(study, "stop_reason", self.stop_reason)
            set_study_system_attr(study, "stopped_early", True)
            study.stop()
            return

        # 4. Plateau Patience
        if self.cfg.plateau_patience_trials is not None:
            if val >= self.best_search_score + 1e-4:
                self.best_search_score = val
                self.no_improve_trials = 0
            else:
                self.no_improve_trials += 1
                if self.no_improve_trials >= self.cfg.plateau_patience_trials:
                    self.stopped_early = True
                    pat = self.cfg.plateau_patience_trials
                    self.stop_reason = (
                        f"Plateau reached: no improvement >= 1e-4 in {pat} consecutive trials"
                    )
                    set_study_system_attr(study, "stop_reason", self.stop_reason)
                    set_study_system_attr(study, "stopped_early", True)
                    study.stop()
                    return


def run_hpo_study(
    config_path: Path | str,
    project_root: Path | str | None = None,
    n_trials: int | None = None,
    smoke: bool = False,
    custom_objective: Callable[[optuna.Trial], float] | None = None,
) -> dict[str, Any]:
    if n_trials is not None and n_trials <= 0:
        raise ValueError("n_trials must be positive")

    root_p = Path(project_root) if project_root is not None else Path.cwd()
    tuning_cfg = load_tuning_config(config_path, project_root=root_p)
    base_cfg = load_config(tuning_cfg.baseline_config, project_root=root_p)

    train_df, test_df = load_frames(base_cfg.data.train, base_cfg.data.test)
    contract = DatasetContract(
        id_column=base_cfg.data.id_column, target_column=base_cfg.data.target_column
    )
    validate_frames(train_df, test_df, contract)
    y_encoded = encode_target(train_df[base_cfg.data.target_column], contract)

    manifest = pd.read_parquet(tuning_cfg.split_manifest)
    validate_split_manifest(
        manifest=manifest,
        n_rows=len(train_df),
        n_splits=base_cfg.validation.n_splits,
        n_repeats=base_cfg.validation.n_repeats,
        y=y_encoded,
    )

    study_name = f"{tuning_cfg.study_name}__smoke" if smoke else tuning_cfg.study_name

    if smoke:
        smoke_n_rows = min(5000, len(train_df))
        train_df_use = train_df.iloc[:smoke_n_rows].reset_index(drop=True)
        y_encoded_use = y_encoded[:smoke_n_rows]

        smoke_manifest = build_split_manifest(
            y=y_encoded_use,
            n_splits=base_cfg.validation.n_splits,
            n_repeats=1,
            random_state=base_cfg.validation.random_state,
        )
        manifest_use = smoke_manifest
        search_folds_use = tuple(
            f for f in tuning_cfg.hpo.search_fold_ids if f < base_cfg.validation.n_splits
        )
        finalist_folds_use = tuple(
            f for f in tuning_cfg.hpo.finalist_fold_ids if f < base_cfg.validation.n_splits
        )
        if not search_folds_use:
            search_folds_use = (0, 1)
        if not finalist_folds_use:
            finalist_folds_use = (0, 1, 2)
    else:
        train_df_use = train_df
        y_encoded_use = y_encoded
        manifest_use = manifest
        search_folds_use = tuning_cfg.hpo.search_fold_ids
        finalist_folds_use = tuning_cfg.hpo.finalist_fold_ids

    X_train_raw = train_df_use.drop(columns=[base_cfg.data.id_column, base_cfg.data.target_column])
    X_train = apply_feature_policy("base", X_train_raw)

    flat_map = map_flat_folds(manifest_use)

    max_flat_fold = len(flat_map) - 1
    for f in search_folds_use:
        if f > max_flat_fold:
            raise ValueError(f"Search fold ID {f} out of range for manifest (max {max_flat_fold})")
    for f in finalist_folds_use:
        if f > max_flat_fold:
            raise ValueError(
                f"Finalist fold ID {f} out of range for manifest (max {max_flat_fold})"
            )

    out_dir = root_p / "outputs" / "tuning" / study_name
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_dir / "metadata.json"

    cfg_p = Path(config_path)
    if not cfg_p.is_absolute():
        cfg_p = (root_p / cfg_p).resolve()

    config_sha256 = _compute_file_sha256(cfg_p)
    data_sha256 = _compute_combined_sha256([base_cfg.data.train, base_cfg.data.test])
    if smoke:
        split_sha256 = hashlib.sha256(smoke_manifest.to_parquet()).hexdigest()
    else:
        split_sha256 = _compute_file_sha256(tuning_cfg.split_manifest)

    search_space_fp = _compute_search_space_fingerprint(tuning_cfg.model_name)

    canonical_prov = _canonicalize_provenance(
        {
            "config_sha256": config_sha256,
            "data_sha256": data_sha256,
            "split_sha256": split_sha256,
            "model_name": tuning_cfg.model_name,
            "smoke": smoke,
            "effective_n_rows": len(train_df_use),
            "effective_search_fold_ids": list(search_folds_use),
            "effective_finalist_fold_ids": list(finalist_folds_use),
            "sampler_seed": tuning_cfg.hpo.sampler_seed,
            "search_space_fingerprint": search_space_fp,
            "search_space_version": "1.0",
        }
    )

    db_path = out_dir / "study.db"
    storage_url = f"sqlite:///{db_path.resolve()}"

    sampler = optuna.samplers.TPESampler(seed=tuning_cfg.hpo.sampler_seed)
    if tuning_cfg.hpo.pruner.lower() == "median":
        pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner()
    else:
        pruner = optuna.pruners.NopPruner()

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=True,
    )

    # CRITICAL 1: System Attr Provenance Validation
    sys_attrs = get_study_system_attrs(study)
    existing_sys_prov = sys_attrs.get("provenance")
    if existing_sys_prov:
        existing_canonical = _canonicalize_provenance(existing_sys_prov)
        if existing_canonical != canonical_prov:
            mismatches = [
                f"{k}: expected {canonical_prov.get(k)!r}, got {existing_canonical.get(k)!r}"
                for k in canonical_prov
                if existing_canonical.get(k) != canonical_prov.get(k)
            ]
            msg = (
                f"Incompatible resume attempt for study '{study_name}'. "
                f"System attr provenance mismatches: {'; '.join(mismatches)}"
            )
            raise ValueError(msg)
    elif len(study.trials) > 0:
        msg = (
            f"Incompatible resume attempt for study '{study_name}': "
            "existing study DB lacks provenance system attribute."
        )
        raise ValueError(msg)
    else:
        set_study_system_attr(study, "provenance", canonical_prov)

    # Check if study is already stopped (via system attr or reconstructed state)
    stopping_cb = OptunaStoppingCallback(tuning_cfg.hpo, study.trials)
    sys_attrs = get_study_system_attrs(study)

    is_already_stopped = bool(sys_attrs.get("stopped_early", stopping_cb.stopped_early))

    if is_already_stopped:
        n_trials_to_run = 0
        stop_reason = sys_attrs.get("stop_reason", stopping_cb.stop_reason)
        stopped_early = True
        set_study_system_attr(study, "stop_reason", stop_reason)
        set_study_system_attr(study, "stopped_early", True)
    else:
        n_trials_target = n_trials if n_trials is not None else tuning_cfg.hpo.n_trials
        existing_count = len(study.trials)
        n_trials_to_run = max(0, n_trials_target - existing_count)
        stop_reason = stopping_cb.stop_reason
        stopped_early = stopping_cb.stopped_early

    existing_count = len(study.trials)

    def default_objective(trial: optuna.Trial) -> float:
        params = sample_params(trial, tuning_cfg.model_name)
        res = evaluate_model_folds(
            X_train=X_train,
            y_encoded=y_encoded_use,
            manifest=manifest_use,
            flat_folds_map=flat_map,
            fold_ids=search_folds_use,
            model_name=tuning_cfg.model_name,
            params=params,
            base_seed=base_cfg.project.seed,
            trial=trial,
        )
        trial.set_user_attr("roc_auc_mean", res["mean_roc_auc"])
        trial.set_user_attr("roc_auc_std", res["std_roc_auc"])
        trial.set_user_attr("pr_auc_mean", res["mean_average_precision"])
        trial.set_user_attr("log_loss_mean", res["mean_log_loss"])
        return res["mean_roc_auc"]

    objective_func = custom_objective if custom_objective is not None else default_objective

    def wrapped_objective(trial: optuna.Trial) -> float:
        start_time = time.time()
        try:
            val = objective_func(trial)
            runtime = time.time() - start_time
            trial.set_user_attr("runtime_seconds", runtime)
            trial.set_user_attr("state", "COMPLETED")
            trial.set_user_attr("failure_reason", "")
            return val
        except optuna.TrialPruned as e:
            runtime = time.time() - start_time
            trial.set_user_attr("runtime_seconds", runtime)
            trial.set_user_attr("state", "PRUNED")
            trial.set_user_attr("failure_reason", str(e))
            raise e
        except Exception as e:
            runtime = time.time() - start_time
            trial.set_user_attr("runtime_seconds", runtime)
            trial.set_user_attr("state", "FAIL")
            trial.set_user_attr("failure_reason", str(e))
            raise e

    if n_trials_to_run > 0:
        study.optimize(
            wrapped_objective,
            n_trials=n_trials_to_run,
            timeout=tuning_cfg.hpo.timeout_seconds,
            callbacks=[stopping_cb],
            catch=(Exception,),
        )

    trials_executed_run = len(study.trials) - existing_count
    final_sys_attrs = get_study_system_attrs(study)
    stop_reason = final_sys_attrs.get("stop_reason", stopping_cb.stop_reason)
    stopped_early = bool(final_sys_attrs.get("stopped_early", stopping_cb.stopped_early))
    n_trials_target = n_trials if n_trials is not None else tuning_cfg.hpo.n_trials

    # Build and export trials.parquet
    trials_records: list[dict[str, Any]] = []
    for t in study.get_trials(deepcopy=False):
        t_val = t.value if t.value is not None else np.nan
        t_dur = t.duration.total_seconds() if t.duration else np.nan
        t_eval_folds = t.user_attrs.get("evaluated_fold_ids", [])
        is_complete = bool(t.state == TrialState.COMPLETE)
        prov_complete = bool(is_complete and set(t_eval_folds) == set(search_folds_use))

        record: dict[str, Any] = {
            "number": t.number,
            "state": t.user_attrs.get("state", t.state.name),
            "value": t_val,
            "roc_auc_mean": t.user_attrs.get("roc_auc_mean", t_val),
            "roc_auc_std": t.user_attrs.get("roc_auc_std", np.nan),
            "pr_auc_mean": t.user_attrs.get("pr_auc_mean", np.nan),
            "log_loss_mean": t.user_attrs.get("log_loss_mean", np.nan),
            "runtime_seconds": t.user_attrs.get("runtime_seconds", t_dur),
            "failure_reason": t.user_attrs.get("failure_reason", ""),
            "evaluated_fold_ids": str(t_eval_folds),
            "provenance_complete": prov_complete,
        }
        for k, v in t.params.items():
            record[f"param_{k}"] = v
        trials_records.append(record)

    trials_df = pd.DataFrame(trials_records)
    trials_path = out_dir / "trials.parquet"
    trials_df.to_parquet(trials_path)

    # Shortlist top candidates based on search score
    completed_trials = [t for t in study.get_trials() if t.state == TrialState.COMPLETE]
    completed_trials.sort(key=lambda t: t.value if t.value is not None else -1.0, reverse=True)
    top_candidates = completed_trials[: tuning_cfg.hpo.finalist_count]

    candidate_evaluations: list[dict[str, Any]] = []

    for s_rank, trial in enumerate(top_candidates, start=1):
        params = trial.params.copy()
        if tuning_cfg.model_name in ("catboost", "lightgbm"):
            if tuning_cfg.model_name == "catboost":
                params["thread_count"] = 4
            else:
                params["n_jobs"] = 4
                params["subsample_freq"] = 1

        res = evaluate_model_folds(
            X_train=X_train,
            y_encoded=y_encoded_use,
            manifest=manifest_use,
            flat_folds_map=flat_map,
            fold_ids=finalist_folds_use,
            model_name=tuning_cfg.model_name,
            params=params,
            base_seed=base_cfg.project.seed,
        )

        candidate_evaluations.append(
            {
                "search_rank": s_rank,
                "trial": trial,
                "rerun_result": res,
            }
        )

    # Re-rank finalists by full-protocol rerun ROC-AUC descending
    candidate_evaluations.sort(key=lambda item: item["rerun_result"]["mean_roc_auc"], reverse=True)

    finalist_records: list[dict[str, Any]] = []
    finalist_fold_records: list[dict[str, Any]] = []

    for f_rank, item in enumerate(candidate_evaluations, start=1):
        s_rank = item["search_rank"]
        trial = item["trial"]
        res = item["rerun_result"]

        fin_record: dict[str, Any] = {
            "finalist_rank": f_rank,
            "search_rank": s_rank,
            "trial_number": trial.number,
            "roc_auc_mean": res["mean_roc_auc"],
            "roc_auc_std": res["std_roc_auc"],
            "pr_auc_mean": res["mean_average_precision"],
            "log_loss_mean": res["mean_log_loss"],
            "search_roc_auc_mean": trial.value,
            "evaluated_fold_ids": str(finalist_folds_use),
        }
        for k, v in trial.params.items():
            fin_record[f"param_{k}"] = v
        finalist_records.append(fin_record)

        for fold_rec in res["fold_metrics"]:
            finalist_fold_records.append(
                {
                    "trial_number": trial.number,
                    "finalist_rank": f_rank,
                    "search_rank": s_rank,
                    "fold_id": fold_rec["fold_id"],
                    "repeat": fold_rec["repeat"],
                    "fold": fold_rec["fold"],
                    "roc_auc": fold_rec["roc_auc"],
                    "average_precision": fold_rec["average_precision"],
                    "log_loss": fold_rec["log_loss"],
                    "runtime_seconds": fold_rec["runtime_seconds"],
                }
            )

    # Stable empty parquet schemas
    if len(finalist_records) > 0:
        finalists_df = pd.DataFrame(finalist_records)
    else:
        finalists_df = pd.DataFrame(
            columns=[
                "finalist_rank",
                "search_rank",
                "trial_number",
                "roc_auc_mean",
                "roc_auc_std",
                "pr_auc_mean",
                "log_loss_mean",
                "search_roc_auc_mean",
                "evaluated_fold_ids",
            ]
        ).astype(
            {
                "finalist_rank": "int64",
                "search_rank": "int64",
                "trial_number": "int64",
                "roc_auc_mean": "float64",
                "roc_auc_std": "float64",
                "pr_auc_mean": "float64",
                "log_loss_mean": "float64",
                "search_roc_auc_mean": "float64",
                "evaluated_fold_ids": "object",
            }
        )

    finalists_path = out_dir / "finalists.parquet"
    finalists_df.to_parquet(finalists_path)

    if len(finalist_fold_records) > 0:
        finalist_fold_df = pd.DataFrame(finalist_fold_records)
    else:
        finalist_fold_df = pd.DataFrame(
            columns=[
                "trial_number",
                "finalist_rank",
                "search_rank",
                "fold_id",
                "repeat",
                "fold",
                "roc_auc",
                "average_precision",
                "log_loss",
                "runtime_seconds",
            ]
        ).astype(
            {
                "trial_number": "int64",
                "finalist_rank": "int64",
                "search_rank": "int64",
                "fold_id": "int64",
                "repeat": "int64",
                "fold": "int64",
                "roc_auc": "float64",
                "average_precision": "float64",
                "log_loss": "float64",
                "runtime_seconds": "float64",
            }
        )

    finalist_fold_path = out_dir / "finalist_fold_metrics.parquet"
    finalist_fold_df.to_parquet(finalist_fold_path)

    # In full mode, assert every finalist has every configured finalist fold exactly once
    if not smoke and len(finalist_fold_df) > 0:
        expected_folds_count = len(finalist_folds_use)
        for _, group in finalist_fold_df.groupby("trial_number"):
            if len(group) != expected_folds_count:
                msg = f"Finalist missing folds: expected {expected_folds_count}, got {len(group)}"
                raise AssertionError(msg)

    metadata = {
        "study_name": study_name,
        "base_study_name": tuning_cfg.study_name,
        "model_name": tuning_cfg.model_name,
        "provenance": canonical_prov,
        "n_trials_target": n_trials_target,
        "n_trials_executed_total": len(study.trials),
        "n_trials_executed_run": trials_executed_run,
        "timeout_seconds": tuning_cfg.hpo.timeout_seconds,
        "finalist_count_configured": tuning_cfg.hpo.finalist_count,
        "effective_finalist_count": len(finalist_records),
        "stop_reason": stop_reason,
        "stopped_early": stopped_early,
        "stopping_policy": {
            "plateau_patience_trials": tuning_cfg.hpo.plateau_patience_trials,
            "roc_auc_guardrail_floor": tuning_cfg.hpo.roc_auc_guardrail_floor,
            "pr_auc_guardrail_floor": tuning_cfg.hpo.pr_auc_guardrail_floor,
            "log_loss_guardrail_ceiling": tuning_cfg.hpo.log_loss_guardrail_ceiling,
        },
        "smoke": smoke,
        "output_dir": f"outputs/tuning/{study_name}",
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "study_name": study_name,
        "trials_path": trials_path,
        "finalists_path": finalists_path,
        "finalist_fold_metrics_path": finalist_fold_path,
        "metadata_path": meta_path,
        "metadata": metadata,
        "study": study,
    }

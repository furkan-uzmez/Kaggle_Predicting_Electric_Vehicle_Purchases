import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import pytest

from ev_purchase.splits import build_split_manifest
from ev_purchase.tuning import (
    FidelityPlan,
    HPOConfig,
    evaluate_model_folds,
    get_study_system_attrs,
    map_flat_folds,
    run_hpo_study,
)


def test_fidelity_plan_reserves_full_protocol_for_finalists() -> None:
    plan = FidelityPlan(search_fold_ids=(0, 1, 2, 3, 4), finalist_fold_ids=tuple(range(15)))
    assert len(plan.search_fold_ids) == 5
    assert len(plan.finalist_fold_ids) == 15
    assert set(plan.search_fold_ids) <= set(plan.finalist_fold_ids)


def test_hpo_config_validation_positive_counts_and_bounds() -> None:
    with pytest.raises(ValueError, match="n_trials must be positive"):
        HPOConfig(
            n_trials=0,
            timeout_seconds=100,
            search_fold_ids=(0,),
            finalist_fold_ids=(0,),
            finalist_count=1,
            sampler_seed=42,
            pruner="median",
        )

    with pytest.raises(ValueError, match="timeout_seconds must be positive"):
        HPOConfig(
            n_trials=10,
            timeout_seconds=0,
            search_fold_ids=(0,),
            finalist_fold_ids=(0,),
            finalist_count=1,
            sampler_seed=42,
            pruner="median",
        )

    with pytest.raises(ValueError, match="finalist_count must be positive"):
        HPOConfig(
            n_trials=5,
            timeout_seconds=100,
            search_fold_ids=(0,),
            finalist_fold_ids=(0,),
            finalist_count=0,
            sampler_seed=42,
            pruner="median",
        )

    with pytest.raises(ValueError, match="[Ii]nvalid pruner"):
        HPOConfig(
            n_trials=5,
            timeout_seconds=100,
            search_fold_ids=(0,),
            finalist_fold_ids=(0,),
            finalist_count=1,
            sampler_seed=42,
            pruner="invalid_pruner",
        )


def test_fidelity_plan_validation_invalid_duplicate_bool_out_of_range() -> None:
    with pytest.raises(ValueError, match="search_fold_ids must not be empty"):
        FidelityPlan(search_fold_ids=(), finalist_fold_ids=(0, 1))

    with pytest.raises(ValueError, match="finalist_fold_ids must not be empty"):
        FidelityPlan(search_fold_ids=(0,), finalist_fold_ids=())

    with pytest.raises(TypeError, match="elements must be integers"):
        FidelityPlan(search_fold_ids=(True, 1), finalist_fold_ids=(0, 1))

    with pytest.raises(TypeError, match="elements must be integers"):
        FidelityPlan(search_fold_ids=(0, 1), finalist_fold_ids=(0, False))

    with pytest.raises(TypeError, match="elements must be integers"):
        FidelityPlan(search_fold_ids=(0.5, 1), finalist_fold_ids=(0, 1))  # type: ignore

    with pytest.raises(ValueError, match="elements must be non-negative"):
        FidelityPlan(search_fold_ids=(-1, 0), finalist_fold_ids=(0, 1))

    with pytest.raises(ValueError, match="must contain unique fold IDs"):
        FidelityPlan(search_fold_ids=(0, 0), finalist_fold_ids=(0, 1))

    with pytest.raises(ValueError, match="must contain unique fold IDs"):
        FidelityPlan(search_fold_ids=(0, 1), finalist_fold_ids=(0, 1, 1))

    with pytest.raises(ValueError, match="search_fold_ids must be a subset of finalist_fold_ids"):
        FidelityPlan(search_fold_ids=(0, 5), finalist_fold_ids=(0, 1, 2, 3, 4))


def test_flat_fold_mapping_deterministic_and_exact() -> None:
    y = np.array([0, 1] * 20)
    manifest = build_split_manifest(y=y, n_splits=5, n_repeats=3, random_state=42)

    mapping = map_flat_folds(manifest)
    assert len(mapping) == 15

    expected_pairs = [(r, f) for r in range(3) for f in range(5)]
    for idx, expected in enumerate(expected_pairs):
        assert mapping[idx] == expected


def test_objective_touches_only_search_folds_and_fit_excludes_validation_rows() -> None:
    np.random.seed(42)
    n = 100
    df = pd.DataFrame(
        {
            "feature1": np.random.randn(n),
            "feature2": np.random.randn(n),
            "cat1": np.random.choice(["A", "B"], size=n),
        }
    )
    y = np.random.choice([0, 1], size=n)

    manifest = build_split_manifest(y=y, n_splits=5, n_repeats=3, random_state=42)
    flat_map = map_flat_folds(manifest)

    search_folds = (0, 1)
    fit_call_logs: list[tuple[int, set[int]]] = []

    res = evaluate_model_folds(
        X_train=df,
        y_encoded=y,
        manifest=manifest,
        flat_folds_map=flat_map,
        fold_ids=search_folds,
        model_name="dummy",
        params={},
        base_seed=42,
        fit_callback=lambda fold_id, train_indices, val_indices: fit_call_logs.append(
            (fold_id, set(val_indices).intersection(set(train_indices)))
        ),
    )

    assert len(res["fold_metrics"]) == 2
    assert res["evaluated_fold_ids"] == (0, 1)

    for fold_id, intersection in fit_call_logs:
        assert len(intersection) == 0, f"Validation rows leaked into fit for fold {fold_id}"


def test_finalist_rerun_touches_all_finalist_folds() -> None:
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({"feat": np.random.randn(n)})
    y = np.random.choice([0, 1], size=n)

    manifest = build_split_manifest(y=y, n_splits=5, n_repeats=3, random_state=42)
    flat_map = map_flat_folds(manifest)

    finalist_folds = tuple(range(15))
    res = evaluate_model_folds(
        X_train=df,
        y_encoded=y,
        manifest=manifest,
        flat_folds_map=flat_map,
        fold_ids=finalist_folds,
        model_name="dummy",
        params={},
        base_seed=42,
    )

    assert len(res["fold_metrics"]) == 15
    assert res["evaluated_fold_ids"] == finalist_folds


def _setup_synthetic_environment(tmp_path: Path) -> tuple[Path, Path, Path]:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_csv = data_dir / "train.csv"
    test_csv = data_dir / "test.csv"

    df_tr = pd.DataFrame(
        {"id": range(100), "feat": np.random.randn(100), "target": ["No", "Yes"] * 50}
    )
    df_te = pd.DataFrame({"id": range(100, 120), "feat": np.random.randn(20)})
    df_tr.to_csv(train_csv, index=False)
    df_te.to_csv(test_csv, index=False)

    manifest = build_split_manifest(y=[0, 1] * 50, n_splits=5, n_repeats=3)
    split_path = data_dir / "splits.parquet"
    manifest.to_parquet(split_path)

    exp_cfg = tmp_path / "exp.yaml"
    exp_cfg.write_text(
        f"""
project:
  seed: 42
data:
  train: {train_csv.relative_to(tmp_path)}
  test: {test_csv.relative_to(tmp_path)}
  sample_submission: {test_csv.relative_to(tmp_path)}
  id_column: id
  target_column: target
validation:
  strategy: repeated_stratified_kfold
  n_splits: 5
  n_repeats: 3
  random_state: 42
metrics:
  primary: roc_auc
  guardrails:
    - average_precision
    - log_loss
outputs:
  root: outputs
""",
        encoding="utf-8",
    )

    tune_cfg = tmp_path / "tune.yaml"
    tune_cfg.write_text(
        f"""
model_name: dummy
baseline_config: {exp_cfg.relative_to(tmp_path)}
split_manifest: {split_path.relative_to(tmp_path)}
study_name: test_study
hpo:
  n_trials: 3
  timeout_seconds: 300
  search_fold_ids: [0, 1]
  finalist_fold_ids: [0, 1, 2]
  finalist_count: 2
  sampler_seed: 42
  pruner: median
  plateau_patience_trials: null
  roc_auc_guardrail_floor: null
  pr_auc_guardrail_floor: null
  log_loss_guardrail_ceiling: null
""",
        encoding="utf-8",
    )
    return tune_cfg, split_path, exp_cfg


def test_smoke_and_full_mode_isolation_and_paths() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        tune_cfg, _, _ = _setup_synthetic_environment(tmp_path)

        res_smoke = run_hpo_study(config_path=tune_cfg, project_root=tmp_path, smoke=True)
        assert res_smoke["study_name"] == "test_study__smoke"
        assert res_smoke["metadata"]["smoke"] is True
        assert res_smoke["metadata_path"].parent.name == "test_study__smoke"

        res_full = run_hpo_study(config_path=tune_cfg, project_root=tmp_path, smoke=False)
        assert res_full["study_name"] == "test_study"
        assert res_full["metadata"]["smoke"] is False
        assert res_full["metadata_path"].parent.name == "test_study"


def test_system_attr_provenance_persisted_and_validated_on_resume() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        tune_cfg, _, _ = _setup_synthetic_environment(tmp_path)

        res1 = run_hpo_study(config_path=tune_cfg, project_root=tmp_path, smoke=True)
        study1 = res1["study"]
        sys_attrs1 = get_study_system_attrs(study1)
        assert "provenance" in sys_attrs1
        assert sys_attrs1["provenance"]["sampler_seed"] == 42
        assert "search_space_fingerprint" in sys_attrs1["provenance"]

        # 1. Modify config to change sampler seed -> resume must fail
        tune_cfg.write_text(
            tune_cfg.read_text().replace("sampler_seed: 42", "sampler_seed: 99"),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="[Ii]ncompatible resume"):
            run_hpo_study(config_path=tune_cfg, project_root=tmp_path, smoke=True)

        # 2. Existing DB lacking provenance system_attrs -> resume must fail
        db_path = res1["metadata_path"].parent / "study.db"
        storage_url = f"sqlite:///{db_path.resolve()}"
        study_raw = optuna.load_study(study_name="test_study__smoke", storage=storage_url)
        storage = optuna.storages.RDBStorage(storage_url)
        storage.set_study_system_attr(study_raw._study_id, "provenance", {})

        tune_cfg.write_text(
            tune_cfg.read_text().replace("sampler_seed: 99", "sampler_seed: 42"),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="[Ii]ncompatible resume"):
            run_hpo_study(config_path=tune_cfg, project_root=tmp_path, smoke=True)


def test_early_stopped_resume_executes_zero_trials_and_preserves_stop_reason() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        tune_cfg, _, _ = _setup_synthetic_environment(tmp_path)

        tune_cfg.write_text(
            tune_cfg.read_text().replace(
                "plateau_patience_trials: null", "plateau_patience_trials: 2"
            ),
            encoding="utf-8",
        )

        def flat_obj(trial: optuna.Trial) -> float:
            return 0.80

        res1 = run_hpo_study(
            config_path=tune_cfg,
            project_root=tmp_path,
            n_trials=5,
            smoke=True,
            custom_objective=flat_obj,
        )

        study1 = res1["study"]
        sys_attrs1 = get_study_system_attrs(study1)
        assert len(study1.trials) == 3
        assert sys_attrs1.get("stopped_early") is True
        original_reason = sys_attrs1.get("stop_reason")

        # Resume study with larger target budget (n_trials=10)
        res2 = run_hpo_study(
            config_path=tune_cfg,
            project_root=tmp_path,
            n_trials=10,
            smoke=True,
            custom_objective=flat_obj,
        )

        meta2 = res2["metadata"]
        assert meta2["n_trials_executed_run"] == 0
        assert meta2["n_trials_executed_total"] == 3
        assert meta2["stopped_early"] is True
        assert meta2["stop_reason"] == original_reason


def test_true_optuna_stopping_via_callback() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        tune_cfg, _, _ = _setup_synthetic_environment(tmp_path)

        tune_cfg.write_text(
            tune_cfg.read_text().replace(
                "plateau_patience_trials: null", "plateau_patience_trials: 2"
            ),
            encoding="utf-8",
        )

        def flat_obj(trial: optuna.Trial) -> float:
            return 0.80

        res = run_hpo_study(
            config_path=tune_cfg,
            project_root=tmp_path,
            n_trials=10,
            smoke=True,
            custom_objective=flat_obj,
        )

        study = res["study"]
        sys_attrs = get_study_system_attrs(study)
        assert len(study.trials) == 3
        assert sys_attrs.get("stopped_early") is True
        assert "Plateau" in sys_attrs.get("stop_reason", "")


def test_pr_auc_guardrail_floor_stopping() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        tune_cfg, _, _ = _setup_synthetic_environment(tmp_path)

        tune_cfg.write_text(
            tune_cfg.read_text().replace(
                "pr_auc_guardrail_floor: null", "pr_auc_guardrail_floor: 0.90"
            ),
            encoding="utf-8",
        )

        def low_pr_obj(trial: optuna.Trial) -> float:
            trial.set_user_attr("roc_auc_mean", 0.95)
            trial.set_user_attr("pr_auc_mean", 0.50)
            trial.set_user_attr("log_loss_mean", 0.20)
            return 0.95

        res = run_hpo_study(
            config_path=tune_cfg,
            project_root=tmp_path,
            n_trials=5,
            smoke=True,
            custom_objective=low_pr_obj,
        )
        study = res["study"]
        sys_attrs = get_study_system_attrs(study)
        assert len(study.trials) == 1
        assert sys_attrs.get("stopped_early") is True
        assert "PR-AUC" in sys_attrs.get("stop_reason", "")


def test_log_loss_guardrail_ceiling_stopping() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        tune_cfg, _, _ = _setup_synthetic_environment(tmp_path)

        tune_cfg.write_text(
            tune_cfg.read_text().replace(
                "log_loss_guardrail_ceiling: null", "log_loss_guardrail_ceiling: 0.25"
            ),
            encoding="utf-8",
        )

        def high_loss_obj(trial: optuna.Trial) -> float:
            trial.set_user_attr("roc_auc_mean", 0.90)
            trial.set_user_attr("pr_auc_mean", 0.80)
            trial.set_user_attr("log_loss_mean", 0.35)
            return 0.90

        res = run_hpo_study(
            config_path=tune_cfg,
            project_root=tmp_path,
            n_trials=5,
            smoke=True,
            custom_objective=high_loss_obj,
        )
        study = res["study"]
        sys_attrs = get_study_system_attrs(study)
        assert len(study.trials) == 1
        assert sys_attrs.get("stopped_early") is True
        assert "LogLoss" in sys_attrs.get("stop_reason", "")


def test_true_incremental_failure_provenance_preserves_completed_folds() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        tune_cfg, _, _ = _setup_synthetic_environment(tmp_path)

        def fail_on_fold_1_obj(trial: optuna.Trial) -> float:
            df_tr = pd.DataFrame({"feat": np.random.randn(100)})
            y = np.array([0, 1] * 50)
            manifest = build_split_manifest(y=y, n_splits=5, n_repeats=1)
            flat_map = map_flat_folds(manifest)

            def fit_cb(fold_id: int, tr: np.ndarray, va: np.ndarray) -> None:
                if fold_id == 1:
                    raise RuntimeError("Fail on fold 1")

            try:
                evaluate_model_folds(
                    X_train=df_tr,
                    y_encoded=y,
                    manifest=manifest,
                    flat_folds_map=flat_map,
                    fold_ids=(0, 1, 2),
                    model_name="dummy",
                    params={},
                    base_seed=42,
                    trial=trial,
                    fit_callback=fit_cb,
                )
            except RuntimeError as exc:
                raise exc
            return 0.80

        res = run_hpo_study(
            config_path=tune_cfg,
            project_root=tmp_path,
            n_trials=1,
            smoke=True,
            custom_objective=fail_on_fold_1_obj,
        )

        trials_df = pd.read_parquet(res["trials_path"])
        t0 = trials_df.iloc[0]
        assert t0["state"] in ("FAIL", "FAILED")
        assert t0["evaluated_fold_ids"] == "[0]"
        assert bool(t0["provenance_complete"]) is False


def test_true_rerank_order_and_ranks_by_full_protocol_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        tune_cfg, _, _ = _setup_synthetic_environment(tmp_path)

        def mock_eval_folds(
            X_train: Any,
            y_encoded: Any,
            manifest: Any,
            flat_folds_map: Any,
            fold_ids: tuple[int, ...],
            model_name: str,
            params: dict[str, Any],
            base_seed: int = 42,
            trial: Any = None,
            fit_callback: Any = None,
        ) -> dict[str, Any]:
            # Distinct search vs finalist rerun scores
            if len(fold_ids) == 2:
                # Search phase: Trial 0 gets 0.90, Trial 1 gets 0.80
                if params.get("iterations") == 500:
                    roc = 0.90
                else:
                    roc = 0.80
            else:
                # Finalist rerun phase: Trial 0 gets 0.70, Trial 1 gets 0.95 (REVERSED!)
                if params.get("iterations") == 500:
                    roc = 0.70
                else:
                    roc = 0.95

            fold_recs = [
                {
                    "fold_id": f,
                    "repeat": 0,
                    "fold": f,
                    "runtime_seconds": 0.1,
                    "roc_auc": roc,
                    "average_precision": 0.80,
                    "log_loss": 0.25,
                }
                for f in fold_ids
            ]
            if trial is not None:
                trial.set_user_attr("evaluated_fold_ids", list(fold_ids))
                trial.set_user_attr("fold_metrics", fold_recs)

            return {
                "mean_roc_auc": roc,
                "std_roc_auc": 0.01,
                "mean_average_precision": 0.80,
                "std_average_precision": 0.01,
                "mean_log_loss": 0.25,
                "std_log_loss": 0.01,
                "fold_metrics": fold_recs,
                "evaluated_fold_ids": fold_ids,
            }

        import ev_purchase.tuning

        monkeypatch.setattr(ev_purchase.tuning, "evaluate_model_folds", mock_eval_folds)

        def custom_obj(trial: optuna.Trial) -> float:
            iters = 500 if trial.number == 0 else 200
            trial.suggest_int("iterations", iters, iters)
            params = {"iterations": iters}
            res = mock_eval_folds(
                X_train=None,
                y_encoded=None,
                manifest=None,
                flat_folds_map={},
                fold_ids=(0, 1),
                model_name="dummy",
                params=params,
                trial=trial,
            )
            trial.set_user_attr("roc_auc_mean", res["mean_roc_auc"])
            trial.set_user_attr("pr_auc_mean", res["mean_average_precision"])
            trial.set_user_attr("log_loss_mean", res["mean_log_loss"])
            return res["mean_roc_auc"]

        res = run_hpo_study(
            config_path=tune_cfg,
            project_root=tmp_path,
            n_trials=2,
            smoke=True,
            custom_objective=custom_obj,
        )

        finalists_df = pd.read_parquet(res["finalists_path"])
        assert len(finalists_df) == 2
        # Trial 1 MUST be finalist_rank 1 (rerun ROC-AUC 0.95, search rank 2)
        # Trial 0 MUST be finalist_rank 2 (rerun ROC-AUC 0.70, search rank 1)
        assert list(finalists_df["trial_number"]) == [1, 0]
        assert list(finalists_df["finalist_rank"]) == [1, 2]
        assert list(finalists_df["search_rank"]) == [2, 1]
        assert list(finalists_df["roc_auc_mean"]) == [0.95, 0.70]
        assert list(finalists_df["search_roc_auc_mean"]) == [0.80, 0.90]


def test_stable_empty_parquet_schemas() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        tune_cfg, _, _ = _setup_synthetic_environment(tmp_path)

        def all_fail_obj(trial: optuna.Trial) -> float:
            raise ValueError("All trials fail")

        res = run_hpo_study(
            config_path=tune_cfg,
            project_root=tmp_path,
            n_trials=2,
            smoke=True,
            custom_objective=all_fail_obj,
        )

        finalists_df = pd.read_parquet(res["finalists_path"])
        ffm_df = pd.read_parquet(res["finalist_fold_metrics_path"])

        assert len(finalists_df) == 0
        assert len(ffm_df) == 0

        req_fin_cols = {
            "finalist_rank",
            "search_rank",
            "trial_number",
            "roc_auc_mean",
            "roc_auc_std",
            "pr_auc_mean",
            "log_loss_mean",
            "search_roc_auc_mean",
            "evaluated_fold_ids",
        }
        assert req_fin_cols.issubset(set(finalists_df.columns))

        req_ffm_cols = {
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
        }
        assert req_ffm_cols.issubset(set(ffm_df.columns))


def test_cli_and_runner_n_trials_validation_and_effective_finalists() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        tune_cfg, _, _ = _setup_synthetic_environment(tmp_path)

        with pytest.raises(ValueError, match="n_trials must be positive"):
            run_hpo_study(config_path=tune_cfg, project_root=tmp_path, n_trials=0, smoke=True)

        res = run_hpo_study(config_path=tune_cfg, project_root=tmp_path, n_trials=1, smoke=True)
        finalists_df = pd.read_parquet(res["finalists_path"])
        meta = res["metadata"]

        assert len(finalists_df) == 1
        assert meta["finalist_count_configured"] == 2
        assert meta["effective_finalist_count"] == 1

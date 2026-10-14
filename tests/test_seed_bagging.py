"""Unit and integration tests for the LightGBM 3-seed bagging experiment harness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ev_purchase.data import DatasetContract
from ev_purchase.seed_bagging import (
    MODEL_SEEDS,
    SEED_ALIASES,
    THREAT_MODEL_STATEMENT,
    build_bagging_training_provenance,
    calculate_bagging_diversity_metrics,
    calculate_bagging_experiment_metrics,
    configure_seed_aliases,
    generate_bagging_report_markdown,
    make_gate_decision,
    run_seed_bagging_experiment,
    train_seed_bagged_cv,
)


def test_model_seeds_and_aliases_constants() -> None:
    """Model seeds must be exactly [42, 137, 2026] and stochastic aliases complete."""
    assert list(MODEL_SEEDS) == [42, 137, 2026]
    expected_aliases = {
        "random_state",
        "bagging_seed",
        "feature_fraction_seed",
        "data_random_seed",
        "extra_seed",
    }
    assert expected_aliases.issubset(set(SEED_ALIASES))


def test_configure_seed_aliases_injects_all_aliases() -> None:
    """Configuring seed aliases must inject all aliases consistently with the seed."""
    base_params = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
    }
    configured = configure_seed_aliases(base_params, seed=137)

    # Base parameters preserved
    assert configured["n_estimators"] == 100
    assert configured["learning_rate"] == 0.1
    # All aliases set to 137
    assert configured["random_state"] == 137
    assert configured["bagging_seed"] == 137
    assert configured["feature_fraction_seed"] == 137
    assert configured["data_random_seed"] == 137
    assert configured["extra_seed"] == 137


def test_train_seed_bagged_cv_averages_probabilities() -> None:
    """CV training must train 3 models per fold and average validation probabilities."""
    # Synthetic dataset
    np.random.seed(0)
    n_samples = 60
    train_df = pd.DataFrame(
        {
            "id": np.arange(n_samples),
            "num_feat1": np.random.randn(n_samples),
            "num_feat2": np.random.randn(n_samples),
            "cat_feat": np.random.choice(["A", "B", "C"], size=n_samples),
            "Will_Buy_EV": np.random.choice([0, 1], size=n_samples),
        }
    )
    # Manifest with 2 folds, 1 repeat
    manifest = pd.DataFrame(
        {
            "row_index": np.arange(n_samples),
            "repeat": np.zeros(n_samples, dtype=int),
            "fold": np.repeat([0, 1], n_samples // 2),
        }
    )
    contract = DatasetContract(id_column="id", target_column="Will_Buy_EV")
    params = {"n_estimators": 5, "min_child_samples": 2, "subsample_freq": 1}

    candidate_oof, per_seed_oof = train_seed_bagged_cv(
        train_df=train_df,
        manifest=manifest,
        contract=contract,
        model_params=params,
        model_seeds=[42, 137, 2026],
        folds=[(0, 0)],  # run fold (0, 0)
    )

    val_count = n_samples // 2
    assert len(candidate_oof) == val_count
    assert len(per_seed_oof) == val_count
    assert list(candidate_oof.columns) == ["row_index", "repeat", "fold", "probability"]
    assert "prob_seed_42" in per_seed_oof.columns
    assert "prob_seed_137" in per_seed_oof.columns
    assert "prob_seed_2026" in per_seed_oof.columns

    # Check that candidate probability is exact average of the 3 seed probabilities
    avg_prob = (
        per_seed_oof["prob_seed_42"]
        + per_seed_oof["prob_seed_137"]
        + per_seed_oof["prob_seed_2026"]
    ) / 3.0
    np.testing.assert_allclose(candidate_oof["probability"].to_numpy(), avg_prob.to_numpy())


def test_calculate_bagging_diversity_metrics() -> None:
    """Diversity metrics must compute pairwise Pearson/Spearman correlations, MAE, and per-seed AUCs."""
    y_true = np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 0])
    per_seed_df = pd.DataFrame(
        {
            "prob_seed_42": np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.6, 0.4, 0.85, 0.15]),
            "prob_seed_137": np.array([0.12, 0.88, 0.22, 0.78, 0.28, 0.72, 0.65, 0.38, 0.82, 0.18]),
            "prob_seed_2026": np.array([0.08, 0.92, 0.18, 0.82, 0.32, 0.68, 0.62, 0.42, 0.88, 0.12]),
        }
    )
    div_metrics = calculate_bagging_diversity_metrics(
        y_true=y_true,
        per_seed_df=per_seed_df,
        model_seeds=[42, 137, 2026],
    )

    assert "pairwise_pearson" in div_metrics
    assert "pairwise_spearman" in div_metrics
    assert "pairwise_mae" in div_metrics
    assert "seed_metrics" in div_metrics
    assert "mean_pairwise_pearson" in div_metrics
    assert "mean_prediction_std" in div_metrics

    # Verify correlations are high and positive
    assert div_metrics["mean_pairwise_pearson"] > 0.95
    assert len(div_metrics["seed_metrics"]) == 3
    for s in [42, 137, 2026]:
        assert f"seed_{s}" in div_metrics["seed_metrics"]
        assert div_metrics["seed_metrics"][f"seed_{s}"]["roc_auc"] > 0.9


def test_gate_decision_boundaries() -> None:
    """Gate decision matches preregistered criteria: GO, BORDERLINE, NO-GO."""
    # 1. GO condition
    dec_go = make_gate_decision(
        global_auc_delta=0.00012,
        repeat_auc_deltas=[0.00010, 0.00011, 0.00015],
        nb_ci_lower=0.00002,
        global_log_loss_delta=0.00005,
        slice_regressions=[],
    )
    assert dec_go.decision == "GO"

    # 2. GO failed due to non-positive repeat delta -> NO-GO
    dec_neg_rep = make_gate_decision(
        global_auc_delta=0.00015,
        repeat_auc_deltas=[0.00010, -0.00001, 0.00020],
        nb_ci_lower=0.00003,
        global_log_loss_delta=0.00005,
        slice_regressions=[],
    )
    assert dec_neg_rep.decision == "NO-GO"

    # 3. GO failed due to NB CI lower <= 0 -> NO-GO
    dec_nb_fail = make_gate_decision(
        global_auc_delta=0.00012,
        repeat_auc_deltas=[0.00010, 0.00011, 0.00015],
        nb_ci_lower=-0.00001,
        global_log_loss_delta=0.00005,
        slice_regressions=[],
    )
    assert dec_nb_fail.decision == "NO-GO"

    # 4. GO failed due to log loss degradation > 0.00020 -> NO-GO
    dec_ll_fail = make_gate_decision(
        global_auc_delta=0.00015,
        repeat_auc_deltas=[0.00010, 0.00011, 0.00015],
        nb_ci_lower=0.00003,
        global_log_loss_delta=0.00025,
        slice_regressions=[],
    )
    assert dec_ll_fail.decision == "NO-GO"

    # 5. Borderline condition [0.00005, 0.00010)
    dec_borderline = make_gate_decision(
        global_auc_delta=0.00008,
        repeat_auc_deltas=[0.00006, 0.00009, 0.00008],
        nb_ci_lower=-0.00002,
        global_log_loss_delta=0.00002,
        slice_regressions=[],
    )
    assert dec_borderline.decision == "BORDERLINE"

    # 6. Insufficient gain < 0.00005 -> NO-GO
    dec_low = make_gate_decision(
        global_auc_delta=0.00003,
        repeat_auc_deltas=[0.00002, 0.00004, 0.00003],
        nb_ci_lower=-0.00002,
        global_log_loss_delta=-0.00001,
        slice_regressions=[],
    )
    assert dec_low.decision == "NO-GO"


def test_build_bagging_training_provenance() -> None:
    """Provenance record captures exact model seeds, aliases, parameters, and hashes."""
    prov = build_bagging_training_provenance(
        model_params={"n_estimators": 450},
        model_seeds=[42, 137, 2026],
        data_hash="dummy_data_hash",
        train_file_sha256="dummy_train_hash",
        split_hash="dummy_split_hash",
        config_hash="dummy_config_hash",
        candidate_oof_path=Path("dummy_path.parquet"),
        candidate_oof_sha="dummy_oof_sha",
    )
    assert prov["experiment_name"] == "lightgbm_3seed"
    assert prov["model_seeds"] == [42, 137, 2026]
    assert prov["feature_policy"] == "base"
    assert prov["baseline_candidate_id"] == "lightgbm_hpo_trial35"
    assert "seed_aliases" in prov
    assert prov["data_hash"] == "dummy_data_hash"
    assert prov["threat_model"] == THREAT_MODEL_STATEMENT

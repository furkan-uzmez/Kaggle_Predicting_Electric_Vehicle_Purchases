"""Tests for Feature Search, Hill Climbing, and Stacking Kaggle Notebook Generator.

Validates notebook builder, structure, AST syntax, self-contained execution,
the binding Signal-vs-Noise statistical contract, and submission export contracts.
"""

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import nbformat
import numpy as np
import pandas as pd
import pytest
from scipy import stats

NOTEBOOK_PATH = Path("notebooks/kaggle_ev_purchase_feature_search_and_ensemble.ipynb")
BUILDER_SCRIPT = Path("scripts/build_feature_search_and_ensemble_notebook.py")
FOLD_SCORES_PATH = Path("outputs/leaderboard/fold_scores.parquet")

# Reference CIs recorded in leaderboard.md for lightgbm_hpo_trial35 vs catboost_hpo_trial22.
REFERENCE_NAIVE_CI = (0.0000542, 0.0001547)
REFERENCE_NB_CI = (-0.0000051, 0.0002140)


def test_builder_script_exists() -> None:
    """Builder script must exist in scripts/."""
    assert BUILDER_SCRIPT.is_file(), f"Missing builder script: {BUILDER_SCRIPT}"


def test_builder_runs_and_generates_notebook() -> None:
    """Builder script should execute cleanly and create valid notebook."""
    cmd = [sys.executable, str(BUILDER_SCRIPT)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    msg = f"Builder failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    assert result.returncode == 0, msg
    assert NOTEBOOK_PATH.is_file(), f"Notebook was not created at {NOTEBOOK_PATH}"


def test_checked_in_notebook_matches_generator() -> None:
    """Checked-in notebook sources and metadata must match the generator output."""
    spec = importlib.util.spec_from_file_location("builder", BUILDER_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    generated = module.build_notebook()
    checked_in = nbformat.read(NOTEBOOK_PATH, as_version=4)

    def canonicalize(value: nbformat.NotebookNode) -> dict:
        payload = json.loads(nbformat.writes(value))
        for cell in payload["cells"]:
            cell.pop("id", None)
        return payload

    assert canonicalize(checked_in) == canonicalize(generated)


@pytest.fixture(scope="module")
def notebook() -> nbformat.NotebookNode:
    """Loads notebook node."""
    if not NOTEBOOK_PATH.is_file():
        cmd = [sys.executable, str(BUILDER_SCRIPT)]
        subprocess.run(cmd, check=True)
    return nbformat.read(NOTEBOOK_PATH, as_version=4)


def test_all_code_cells_parse_ast(notebook: nbformat.NotebookNode) -> None:
    """Every code cell must be valid Python syntax."""
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert len(code_cells) >= 8, f"Expected at least 8 code cells, found {len(code_cells)}"

    for idx, cell in enumerate(code_cells):
        try:
            ast.parse(cell.source)
        except SyntaxError as err:
            pytest.fail(f"Code cell {idx} failed AST parse:\n{err}\n\nCell source:\n{cell.source}")


def test_notebook_is_self_contained(notebook: nbformat.NotebookNode) -> None:
    """Notebook must NOT import from src, ev_purchase, or local scripts."""
    all_code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")

    forbidden = ["from src", "import src", "from ev_purchase", "import ev_purchase"]
    for pattern in forbidden:
        assert pattern not in all_code, f"Forbidden import found: {pattern}"


def test_no_machine_absolute_paths(notebook: nbformat.NotebookNode) -> None:
    """Notebook must NOT contain machine-absolute developer home paths."""
    all_content = "\n".join(cell.source for cell in notebook.cells)
    assert "/home/" not in all_content, "Found '/home/' in notebook source!"
    assert "/Users/" not in all_content, "Found '/Users/' in notebook source!"


def test_required_pipeline_sections_present(notebook: nbformat.NotebookNode) -> None:
    """Notebook must implement all 5 phases."""
    all_code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")

    required_keywords = [
        "FoldLocalCategoricalPrep",
        "run_hill_climbing",
        "RUN_STACKING",
        "outer_folds",
        "itertools.combinations",
        "TOP_K_VERIFY",
        "paired_fold_ci",
        "holm_bonferroni",
        "submission_hill_climbing.csv",
        "submission_stacking.csv",
        "submission.csv",
    ]
    for kw in required_keywords:
        assert kw in all_code, f"Missing required pipeline component: {kw}"


# ---------------------------------------------------------------------------
# Signal-vs-Noise statistical contract (binding)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def stat_functions(notebook: nbformat.NotebookNode) -> dict[str, Any]:
    """Extracts paired_fold_ci and holm_bonferroni definitions from the notebook."""
    source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    tree = ast.parse(source)
    wanted = {"paired_fold_ci", "holm_bonferroni"}
    namespace: dict[str, Any] = {"np": np, "stats": stats}

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            module = ast.Module(body=[node], type_ignores=[])
            exec(compile(module, "<notebook>", "exec"), namespace)  # noqa: S102

    missing = wanted - set(namespace)
    assert not missing, f"Notebook is missing statistical functions: {missing}"
    return namespace


def test_ci_uses_t_quantile_not_hardcoded_normal(stat_functions: dict[str, Any]) -> None:
    """Interval width must follow the t quantile for the actual fold count."""
    paired_fold_ci = stat_functions["paired_fold_ci"]
    rng = np.random.default_rng(0)
    scores_a = rng.normal(0.94, 0.001, size=15)
    scores_b = scores_a + rng.normal(0.0002, 0.00008, size=15)

    result = paired_fold_ci(scores_a, scores_b, test_train_ratio=None)
    half_width = (result["hi"] - result["lo"]) / 2.0
    expected_t = stats.t.ppf(0.975, 14)

    assert result["se"] > 0.0, "fixture must produce non-degenerate spread"
    assert half_width == pytest.approx(expected_t * result["se"], rel=1e-9)
    assert expected_t == pytest.approx(2.144787, abs=1e-5)
    assert half_width != pytest.approx(1.96 * result["se"], rel=1e-6)
    assert half_width != pytest.approx(2.0 * result["se"], rel=1e-6)


def test_ci_uses_sample_std_ddof_one(stat_functions: dict[str, Any]) -> None:
    """Standard deviation must use ddof=1, never the population form."""
    paired_fold_ci = stat_functions["paired_fold_ci"]
    scores_a = np.zeros(15)
    scores_b = np.arange(15, dtype=float) / 1000.0

    result = paired_fold_ci(scores_a, scores_b, test_train_ratio=None)
    diffs = scores_b - scores_a
    expected_se = diffs.std(ddof=1) / np.sqrt(15)

    assert result["se"] == pytest.approx(expected_se, rel=1e-12)
    assert result["se"] != pytest.approx(diffs.std(ddof=0) / np.sqrt(15), rel=1e-9)


def test_nadeau_bengio_correction_widens_interval(stat_functions: dict[str, Any]) -> None:
    """The Nadeau-Bengio form must be wider than the naive paired interval."""
    paired_fold_ci = stat_functions["paired_fold_ci"]
    rng = np.random.default_rng(7)
    scores_a = rng.normal(0.94, 0.001, size=15)
    scores_b = scores_a + rng.normal(0.0001, 0.00005, size=15)

    naive = paired_fold_ci(scores_a, scores_b, test_train_ratio=None)
    corrected = paired_fold_ci(scores_a, scores_b, test_train_ratio=0.25)

    diffs = scores_b - scores_a
    expected_se = diffs.std(ddof=1) * np.sqrt(1.0 / 15 + 0.25)

    assert corrected["se"] == pytest.approx(expected_se, rel=1e-12)
    assert (corrected["hi"] - corrected["lo"]) > (naive["hi"] - naive["lo"])


@pytest.mark.skipif(not FOLD_SCORES_PATH.is_file(), reason="fold_scores.parquet unavailable")
def test_ci_reproduces_validated_leaderboard_reference(stat_functions: dict[str, Any]) -> None:
    """Notebook CI must reproduce the project's recorded leaderboard intervals."""
    paired_fold_ci = stat_functions["paired_fold_ci"]
    fold_scores = pd.read_parquet(FOLD_SCORES_PATH)
    pivot = fold_scores.pivot_table(
        index=["repeat", "fold"],
        columns="model",
        values="roc_auc",
    ).sort_index()

    catboost = pivot["catboost_hpo_trial22"].to_numpy()
    lightgbm = pivot["lightgbm_hpo_trial35"].to_numpy()

    naive = paired_fold_ci(catboost, lightgbm, test_train_ratio=None)
    corrected = paired_fold_ci(catboost, lightgbm, test_train_ratio=0.25)

    assert naive["lo"] == pytest.approx(REFERENCE_NAIVE_CI[0], abs=1e-7)
    assert naive["hi"] == pytest.approx(REFERENCE_NAIVE_CI[1], abs=1e-7)
    assert corrected["lo"] == pytest.approx(REFERENCE_NB_CI[0], abs=1e-7)
    assert corrected["hi"] == pytest.approx(REFERENCE_NB_CI[1], abs=1e-7)

    assert not naive["covers_zero"], "naive interval should exclude zero for this pair"
    assert corrected["covers_zero"], "corrected interval must flip this pair to inconclusive"


def test_holm_bonferroni_is_monotone_and_bounded(stat_functions: dict[str, Any]) -> None:
    """Step-down adjustment must be non-decreasing, bounded, and never below raw."""
    holm_bonferroni = stat_functions["holm_bonferroni"]
    raw = [0.001, 0.02, 0.04, 0.3]
    adjusted = holm_bonferroni(raw)

    assert adjusted[0] == pytest.approx(0.004, abs=1e-12)
    assert all(adj >= p for adj, p in zip(adjusted, raw, strict=True))
    assert all(adjusted[i] <= adjusted[i + 1] for i in range(len(adjusted) - 1))
    assert all(0.0 <= adj <= 1.0 for adj in adjusted)


def test_verdict_tiers_cover_all_binding_outcomes(notebook: nbformat.NotebookNode) -> None:
    """All four verdict tiers plus the minimum-lift gate must be present."""
    all_code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")

    assert "Signal (Beats Baseline)" in all_code
    assert "Detectable but not worth adopting" in all_code
    assert "Noise / Inconclusive" in all_code
    assert "Within run-to-run variation" in all_code
    assert "MIN_LIFT = 5e-5" in all_code
    assert "ALPHA = 0.05" in all_code
    assert "covers_zero" in all_code


def test_verdict_requires_family_wise_significance(notebook: nbformat.NotebookNode) -> None:
    """Promotion must gate on the Holm-adjusted p-value, not the raw one."""
    all_code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")

    assert "family_wise_significant" in all_code, "Missing family-wise significance gate"
    assert "p_adj < ALPHA" in all_code, "Verdict must consult the Holm-adjusted p-value"
    assert "if not family_wise_significant:" in all_code


def test_both_noise_sources_are_reported(notebook: nbformat.NotebookNode) -> None:
    """Split noise (paired CI) and run-to-run noise (seed spread) must both appear."""
    all_code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")

    assert "seed_spread" in all_code, "Run-to-run noise (seed spread) is not measured"
    assert "ddof=1" in all_code
    assert "within_seed_noise" in all_code, "Seed spread must feed the verdict"
    assert "Split noise source" in all_code
    assert "Run-to-run noise" in all_code


def test_mandatory_caveats_are_emitted(notebook: nbformat.NotebookNode) -> None:
    """The four binding interpretation caveats must be printed."""
    all_code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")

    assert "MANDATORY CAVEATS" in all_code
    assert "NOT proven equivalence" in all_code
    assert "selection bias" in all_code
    assert "never justifies adoption" in all_code
    assert "neither substitutes for the other" in all_code


def test_xgboost_params_exclude_random_state(notebook: nbformat.NotebookNode) -> None:
    """random_state must be supplied per fold, never duplicated in the params dict."""
    source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    tree = ast.parse(source)

    xgb_params: dict[str, Any] | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if "XGBOOST_PARAMS" in targets and isinstance(node.value, ast.Dict):
                xgb_params = {key.value for key in node.value.keys if isinstance(key, ast.Constant)}

    assert xgb_params is not None, "XGBOOST_PARAMS dict not found"
    assert "random_state" not in xgb_params, (
        "random_state in XGBOOST_PARAMS collides with the per-fold seed at fit time"
    )


def test_torch_import_is_optional(notebook: nbformat.NotebookNode) -> None:
    """Notebook must run without torch installed; GPU detection is best-effort."""
    all_code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")

    assert "except ImportError:" in all_code, "torch import must be guarded"
    assert "if torch is not None:" in all_code, "seed_everything must tolerate missing torch"


# ---------------------------------------------------------------------------
# Artifact persistence contract
# ---------------------------------------------------------------------------

EXPECTED_ARTIFACTS = [
    # Phase 1 evidence
    "fold_metrics.parquet",
    "oof_predictions.parquet",
    "test_predictions.parquet",
    "diversity_correlation.csv",
    "standalone_metrics.csv",
    # Phase 2 evidence
    "hill_climbing_history.csv",
    "hill_climbing_weights.csv",
    # Phase 3 evidence
    "stacking_meta_coefficients.csv",
    "stacking_outer_fold_metrics.csv",
    # Phase 4 evidence
    "feature_search_full_scan.parquet",
    "screen_checkpoint.parquet",
    "top_feature_subsets.parquet",
    # Phase 5 decision evidence
    "decision_ledger.csv",
    "statistical_comparisons.parquet",
    "ensemble_oof_predictions.parquet",
    "run_manifest.json",
    # Submissions — one per evaluated experiment plus the promoted canonical file
    "submission.csv",
    "submission_lightgbm_trial35.csv",
    "submission_hill_climbing.csv",
    "submission_stacking.csv",
    "submission_best_feature_subset.csv",
]


def test_all_expected_artifacts_are_written(notebook: nbformat.NotebookNode) -> None:
    """Every phase must persist its evidence, not only the submissions."""
    all_code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")

    missing = [name for name in EXPECTED_ARTIFACTS if name not in all_code]
    assert not missing, f"Notebook never writes these artifacts: {missing}"


def test_full_feature_scan_is_persisted(notebook: nbformat.NotebookNode) -> None:
    """The complete screening scan must be saved, not only the verified Top-K."""
    all_code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")

    assert "feature_search_full_scan.parquet" in all_code, (
        "The full 8,191-subset scan must be persisted; Top-K alone discards the work"
    )
    assert "df_screen_out" in all_code
    assert "CHECKPOINT_INTERVAL" in all_code, "Periodic checkpointing guards session timeouts"


def test_every_candidate_carries_test_predictions(notebook: nbformat.NotebookNode) -> None:
    """Each evaluated candidate must ship test predictions and a submission file."""
    all_code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")

    assert "'test_preds'" in all_code, "Candidates must carry test predictions"
    assert "'submission'" in all_code, "Candidates must declare a submission filename"
    assert "'policy'" in all_code, "Candidates must declare a policy identifier"
    assert "for cand in candidate_comparisons:\n    save_verified_submission(" in all_code, (
        "Submissions must be exported by iterating candidates, not hard-coded names"
    )


def test_feature_subset_generates_test_predictions(notebook: nbformat.NotebookNode) -> None:
    """The winning feature subset must be refit to produce submittable predictions."""
    all_code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")

    assert "collect_test" in all_code, "eval_subset_per_fold must support test collection"
    assert "test_best_subset" in all_code, "Winning subset must produce test predictions"
    assert "refit_auc" in all_code, "Refit must reproduce the verified AUC"
    assert "abs(refit_auc - best_subset_auc) < 1e-9" in all_code, (
        "Refit AUC must be asserted against the verified 15-fold value"
    )


def test_promotion_uses_strongest_verified_lower_bound(
    notebook: nbformat.NotebookNode,
) -> None:
    """Promotion must pick the candidate with the highest CI lower bound, not a name."""
    all_code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")

    assert "promoted_ci_lower" in all_code, "Promotion must track the CI lower bound"
    assert "lo > promoted_ci_lower" in all_code, "Promotion must compare lower bounds"
    assert "promoted_ci_lower = -np.inf" in all_code, "Baseline must start unbeatable-by-noise"
    assert "cand['name'] == 'Nested Stacking Ensemble'" not in all_code, (
        "Promotion must not branch on hard-coded candidate names"
    )


def test_run_manifest_records_protocol_and_gates(notebook: nbformat.NotebookNode) -> None:
    """Manifest must capture the protocol, gates, decision, and environment."""
    all_code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")

    for key in (
        "'protocol'",
        "'gates'",
        "'decision'",
        "'search'",
        "'environment'",
        "'seed_formula'",
        "'multiplicity_correction'",
        "'promoted_policy'",
    ):
        assert key in all_code, f"run_manifest is missing {key}"
    assert "json.dump(manifest" in all_code


# ---------------------------------------------------------------------------
# GPU-portability regression tests (CatBoostError: Bayesian + subsample)
# ---------------------------------------------------------------------------


def _extract_cfg_dict(notebook: nbformat.NotebookNode, attr: str) -> dict[str, Any]:
    """Reads a CFG class attribute dict literal straight out of the notebook."""
    source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "CFG":
            for stmt in node.body:
                if isinstance(stmt, ast.Assign):
                    targets = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
                    if attr in targets and isinstance(stmt.value, ast.Dict):
                        return {
                            key.value: ast.literal_eval(value)
                            for key, value in zip(stmt.value.keys, stmt.value.values, strict=True)
                        }
    raise AssertionError(f"CFG.{attr} dict literal not found in notebook")


def test_catboost_params_pin_gpu_compatible_bootstrap(
    notebook: nbformat.NotebookNode,
) -> None:
    """subsample requires Bernoulli/MVS/Poisson; the GPU-default Bayesian rejects it.

    Regression test for: CatBoostError ... default bootstrap type (bayesian)
    doesn't support 'subsample' option.
    """
    params = _extract_cfg_dict(notebook, "CATBOOST_PARAMS")

    if "subsample" in params:
        compatible = {"Bernoulli", "MVS", "Poisson"}
        bootstrap = params.get("bootstrap_type")
        assert bootstrap in compatible, (
            f"CFG.CATBOOST_PARAMS uses subsample={params['subsample']} with "
            f"bootstrap_type={bootstrap!r}; GPU CatBoost defaults to Bayesian, "
            "which rejects subsample. Pin Bernoulli (GPU-safe) explicitly."
        )


def test_categorical_prep_is_pandas_version_portable(
    notebook: nbformat.NotebookNode,
) -> None:
    """Category detection must work on pandas 2.x (object) and 3.x (StringDtype).

    A bare `dtype == object` check silently detects zero categorical columns
    on pandas 3, which would feed raw strings into the GBDT models.
    """
    all_code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")

    assert "is_numeric_dtype" in all_code, (
        "FoldLocalCategoricalPrep must use pd.api.types.is_numeric_dtype"
    )
    assert "df[c].dtype == object" not in all_code, (
        "dtype == object misses StringDtype columns on pandas 3.x"
    )


def test_candidates_scored_with_per_repeat_predictions(
    notebook: nbformat.NotebookNode,
) -> None:
    """Fold (r, f) must be scored on repeat r's own predictions, like the baseline.

    Regression test: slicing the repeat-averaged OOF vector gives candidates a
    variance-reduction advantage the baseline never received, producing a
    phantom +0.000189 lift that made an identity blend look like 'Signal'.
    """
    all_code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")

    assert "oof_hill_per_repeat" in all_code, "Hill climbing must build per-repeat blends"
    assert "oof_stacked_per_repeat" in all_code, "Stacking must build per-repeat outputs"
    assert "oof_hill_per_repeat[va_idx, r]" in all_code, (
        "Hill fold scores must index the per-repeat matrix, not the pooled vector"
    )
    assert "oof_stacked_per_repeat[va_idx, r]" in all_code, (
        "Stacking fold scores must index the per-repeat matrix, not the pooled vector"
    )
    assert "roc_auc_score(y_true[va_idx], oof_stacked[va_idx])" not in all_code, (
        "Scoring the pooled stacked vector per fold reintroduces the phantom lift"
    )


def test_measurement_convention_assert_is_present(
    notebook: nbformat.NotebookNode,
) -> None:
    """A runtime assert must block deltas that are measurement artifacts."""
    all_code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")

    assert "max_convention_drift" in all_code, "Convention drift must be measured"
    assert "identity_check" in all_code, "Baseline must be re-scored from per-repeat OOF"
    assert "assert max_convention_drift < 1e-12" in all_code, (
        "Notebook must hard-fail when baseline and candidates use different conventions"
    )


def test_identity_blend_measures_zero_delta(stat_functions: dict[str, Any]) -> None:
    """A candidate identical to the baseline must yield an exactly zero interval."""
    paired_fold_ci = stat_functions["paired_fold_ci"]
    rng = np.random.default_rng(11)
    baseline_scores = rng.normal(0.942, 0.0009, size=15)

    result = paired_fold_ci(baseline_scores, baseline_scores.copy(), test_train_ratio=0.25)

    assert result["mean"] == pytest.approx(0.0, abs=1e-15)
    assert result["se"] == pytest.approx(0.0, abs=1e-15)
    assert result["lo"] == pytest.approx(0.0, abs=1e-15)
    assert result["hi"] == pytest.approx(0.0, abs=1e-15)


def test_categorical_prep_detects_all_six_categoricals(
    notebook: nbformat.NotebookNode,
) -> None:
    """End-to-end check: the notebook's own prep finds the 6 string columns."""
    source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    tree = ast.parse(source)
    namespace: dict[str, Any] = {"np": np, "pd": pd}

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "FoldLocalCategoricalPrep":
            module = ast.Module(body=[node], type_ignores=[])
            exec(compile(module, "<notebook>", "exec"), namespace)  # noqa: S102

    prep_cls = namespace.get("FoldLocalCategoricalPrep")
    assert prep_cls is not None, "FoldLocalCategoricalPrep not found in notebook"

    frame = pd.DataFrame(
        {
            "Age": [25, 40],
            "Annual_Income_USD": [50000.0, 90000.0],
            "Gender": ["Male", "Female"],
            "City_Type": ["Urban", "Rural"],
            "Current_Car_Type": ["Sedan", "SUV"],
            "Home_Charging_Possible": ["Yes", "No"],
            "Subsidy_Available": ["Yes", "No"],
            "Range_Anxiety_Level": ["Low", "High"],
        }
    )
    detected = prep_cls().fit(frame).cat_cols_
    expected = {
        "Gender",
        "City_Type",
        "Current_Car_Type",
        "Home_Charging_Possible",
        "Subsidy_Available",
        "Range_Anxiety_Level",
    }
    assert set(detected) == expected, f"detected {detected}, expected {sorted(expected)}"

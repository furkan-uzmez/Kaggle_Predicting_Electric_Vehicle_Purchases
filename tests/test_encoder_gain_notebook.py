"""Contract tests for the value-level encoding gain notebook generator."""

import ast
import importlib.util
from pathlib import Path

import nbformat as nbf
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = PROJECT_ROOT / "scripts" / "build_encoder_gain_notebook.py"
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "kaggle_ev_purchase_encoder_gain.ipynb"


def _load_generator():
    """Import the generator module directly from its path."""
    spec = importlib.util.spec_from_file_location("build_encoder_gain_notebook", GENERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator():
    return _load_generator()


@pytest.fixture(scope="module")
def notebook(generator):
    return generator.build_notebook()


@pytest.fixture(scope="module")
def code_source(notebook) -> str:
    return "\n".join(c.source for c in notebook.cells if c.cell_type == "code")


class TestGeneratorOutput:
    def test_generator_file_exists(self):
        assert GENERATOR_PATH.exists()

    def test_notebook_is_valid_nbformat(self, notebook):
        nbf.validate(notebook)

    def test_every_code_cell_parses(self, notebook):
        cells = [c for c in notebook.cells if c.cell_type == "code"]
        assert len(cells) >= 10
        for cell in cells:
            ast.parse(cell.source)

    def test_has_markdown_narrative(self, notebook):
        markdown = [c for c in notebook.cells if c.cell_type == "markdown"]
        assert len(markdown) >= 10

    def test_committed_notebook_matches_generator(self, notebook):
        """The checked-in notebook must not drift from its generator."""
        if not NOTEBOOK_PATH.exists():
            pytest.skip("notebook not generated yet")
        on_disk = nbf.read(NOTEBOOK_PATH, as_version=4)
        generated_src = [c.source for c in notebook.cells if c.cell_type == "code"]
        disk_src = [c.source for c in on_disk.cells if c.cell_type == "code"]
        assert generated_src == disk_src, "run scripts/build_encoder_gain_notebook.py to resync"


class TestProtocolContract:
    def test_uses_repeated_stratified_kfold_with_fixed_seed(self, code_source):
        assert "RepeatedStratifiedKFold" in code_source
        assert "random_state=CFG.seed" in code_source

    def test_asserts_per_repeat_coverage(self, code_source):
        assert "coverage == N_REPEATS" in code_source

    def test_asserts_no_train_valid_overlap(self, code_source):
        assert "intersect1d" in code_source
        assert "train/valid overlap" in code_source

    def test_fold_seed_formula_varies_by_repeat_and_fold(self, code_source):
        assert "CFG.seed + repeat * 100 + fold" in code_source

    def test_scores_folds_with_per_repeat_predictions(self, code_source):
        """Guards the defect that produced a spurious lift in an earlier study."""
        assert "oof_per_repeat[repeat, va]" in code_source
        assert "fold scoring must use per-repeat predictions" in code_source

    def test_declares_gates_before_running_arms(self, code_source):
        cfg_pos = code_source.index("MIN_LIFT")
        run_pos = code_source.index("RESULTS['A'] = run_arm")
        assert cfg_pos < run_pos, "gates must be declared before any arm executes"


class TestStatisticalContract:
    def test_uses_ddof_one(self, code_source):
        assert "ddof=1" in code_source

    def test_uses_real_t_quantile_not_hardcoded_constant(self, code_source):
        assert "stats.t.ppf" in code_source
        assert "1.96" not in code_source

    def test_applies_nadeau_bengio_correction(self, code_source):
        assert "se_nb" in code_source
        assert "1.0 / n + ratio" in code_source

    def test_applies_holm_bonferroni(self, code_source):
        assert "holm_adjust" in code_source
        assert "holm-bonferroni" in code_source

    def test_measures_and_uses_seed_spread(self, code_source):
        assert "measure_seed_spread" in code_source
        assert "seed_spread" in code_source
        assert "within run-to-run variation" in code_source

    def test_never_claims_equivalence_on_zero_covering_interval(self, code_source):
        assert "noise / inconclusive" in code_source
        assert "equivalent" not in code_source.lower()

    def test_minimum_lift_gate_is_enforced(self, code_source):
        assert "below minimum lift" in code_source


class TestLeakageContract:
    def test_encoders_are_fitted_on_training_rows_only(self, code_source):
        assert ".fit(base_tr)" in code_source
        assert ".fit(base_va" not in code_source
        assert ".fit(base_te" not in code_source

    def test_target_encoder_transform_takes_no_labels(self, generator):
        """transform must not accept y, making validation-label leakage impossible."""
        source = generator.ENCODER_CODE
        assert "def transform(self, frame):" in source
        assert "def transform(self, frame, y" not in source

    def test_target_encoder_uses_inner_cross_fitting(self, code_source):
        assert "fit_transform(base_tr, y_tr)" in code_source
        assert "StratifiedKFold" in code_source

    def test_validation_and_test_use_full_fold_mapping(self, code_source):
        assert "te.transform(base_va)" in code_source
        assert "te.transform(base_te)" in code_source

    def test_transductive_arm_is_opt_in_and_documented(self, generator, code_source):
        assert "USE_UNLABELED" in code_source
        assert "transductive" in generator.ENCODER_CODE.lower()
        assert "use_unlabeled_transductive" in code_source

    def test_leaderboard_feedback_is_not_a_selection_signal(self, generator):
        header = generator.HEADER_MD.lower()
        assert "never as a selection" in header

    def test_blend_weight_chosen_on_oof_only(self, code_source):
        assert "roc_auc_score(y_true, blended)" in code_source


class TestEncoderSelfTests:
    def test_notebook_self_tests_run_before_training(self, code_source):
        selftest_pos = code_source.index("Encoder self-tests passed")
        train_pos = code_source.index("RESULTS['A'] = run_arm")
        assert selftest_pos < train_pos

    def test_asserts_unseen_value_fallback(self, code_source):
        assert "unseen value must fall back to 0" in code_source

    def test_asserts_leave_one_out_target_encoding(self, code_source):
        assert "positive row leaked its own label" in code_source

    def test_asserts_smoothing_arithmetic(self, code_source):
        assert "smoothing arithmetic drifted" in code_source

    def test_asserts_float_safe_digit_extraction(self, code_source):
        assert "digit extraction hit float error" in code_source


class TestResourceContract:
    def test_smoke_mode_defaults_on(self, generator):
        assert "SMOKE = True" in generator.CFG_CODE

    def test_smoke_results_are_labelled_invalid(self, code_source):
        assert "NOT valid evidence" in code_source

    def test_gpu_path_is_configurable_and_reported(self, code_source):
        assert "USE_GPU" in code_source
        assert "device'] = 'cuda'" in code_source
        assert "gpu_active" in code_source

    def test_wall_clock_budget_guards_expensive_arm(self, code_source):
        assert "budget_remaining()" in code_source
        assert "SKIPPED_ARMS" in code_source

    def test_records_runtime_per_arm(self, code_source):
        assert "per_arm_seconds" in code_source


class TestArtifactContract:
    @pytest.mark.parametrize(
        "artifact",
        [
            "fold_metrics.parquet",
            "decision_ledger.csv",
            "arm_correlations.csv",
            "arm_metrics.csv",
            "oof_predictions.parquet",
            "run_manifest.json",
            "submission.csv",
        ],
    )
    def test_writes_required_artifact(self, code_source, artifact):
        assert artifact in code_source

    def test_writes_one_submission_per_arm(self, code_source):
        assert "submission_arm_{arm}.csv" in code_source

    def test_validates_submission_ranges(self, code_source):
        assert "out-of-range scores" in code_source
        assert "missing scores" in code_source

    def test_manifest_records_gates_and_encoder_settings(self, code_source):
        for key in ("min_lift", "seed_spread", "te_smoothing_m", "digit_k_range"):
            assert key in code_source

    def test_retaining_reference_floor_is_a_valid_outcome(self, code_source):
        assert "the reference floor is retained" in code_source

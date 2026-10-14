"""Contract tests for the diversity stack notebook generator."""

import ast
import importlib.util
from pathlib import Path

import nbformat as nbf
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = PROJECT_ROOT / "scripts" / "build_diversity_stack_notebook.py"
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "kaggle_ev_purchase_diversity_stack.ipynb"


def _load_generator():
    spec = importlib.util.spec_from_file_location("build_diversity_stack_notebook", GENERATOR_PATH)
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
    def test_notebook_is_valid_nbformat(self, notebook):
        nbf.validate(notebook)

    def test_every_code_cell_parses(self, notebook):
        cells = [c for c in notebook.cells if c.cell_type == "code"]
        assert len(cells) >= 10
        for cell in cells:
            ast.parse(cell.source)

    def test_has_markdown_narrative(self, notebook):
        assert len([c for c in notebook.cells if c.cell_type == "markdown"]) >= 10

    def test_committed_notebook_matches_generator(self, notebook):
        if not NOTEBOOK_PATH.exists():
            pytest.skip("notebook not generated yet")
        on_disk = nbf.read(NOTEBOOK_PATH, as_version=4)
        generated = [c.source for c in notebook.cells if c.cell_type == "code"]
        disk = [c.source for c in on_disk.cells if c.cell_type == "code"]
        assert generated == disk, "run scripts/build_diversity_stack_notebook.py to resync"


class TestProtocolContract:
    def test_uses_frozen_repeated_stratified_manifest(self, code_source):
        assert "RepeatedStratifiedKFold" in code_source
        assert "random_state=CFG.seed" in code_source

    def test_asserts_per_repeat_coverage_and_no_overlap(self, code_source):
        assert "coverage == N_REPEATS" in code_source
        assert "train/valid overlap" in code_source

    def test_fold_seed_varies_by_repeat_and_fold(self, code_source):
        assert "CFG.seed + spec['repeat'] * 100 + spec['fold']" in code_source

    def test_scores_folds_with_per_repeat_predictions(self, code_source):
        assert "oof_per_repeat[s['repeat'], s['valid']]" in code_source

    def test_gates_declared_before_training(self, code_source):
        assert code_source.index("MIN_LIFT") < code_source.index("run_candidate(recipe_name")

    def test_reference_is_the_adopted_arm_not_raw_baseline(self, code_source):
        assert "REFERENCE_KEY = 'te_m20__lgbm'" in code_source
        assert "reference" in code_source.lower()


class TestDiversityContract:
    def test_level1_varies_feature_recipes(self, code_source):
        assert "RECIPES = {" in code_source
        for recipe in ("'raw'", "'freq'", "'te_m20'", "'digits'"):
            assert recipe in code_source

    def test_recipe_grid_spans_multiple_smoothing_strengths(self, code_source):
        for recipe in ("te_m5", "te_m100"):
            assert recipe in code_source

    def test_selects_members_by_correlation_ceiling(self, code_source):
        assert "MAX_PAIR_CORR" in code_source
        assert "select_diverse_members" in code_source
        assert "exceeds" in code_source

    def test_warns_when_pool_is_homogeneous(self, code_source):
        # The warning text is split across string literals in the generator, so
        # match the distinctive word rather than a contiguous phrase.
        assert "homogeneous" in code_source
        assert "fewer than two decorrelated members" in code_source

    def test_reports_pairwise_correlations(self, code_source):
        assert "level1_correlations.csv" in code_source
        assert "level1_pair_correlations.csv" in code_source

    def test_member_selection_uses_oof_only(self, code_source):
        assert "pooled_roc_auc" in code_source
        assert "public" not in code_source.lower()


class TestStackingContract:
    def test_meta_learner_scored_under_outer_folds(self, code_source):
        assert "never scored on the folds that produced its own inputs" in code_source
        assert "meta.fit(meta_tr, y_true[spec['train']])" in code_source

    def test_uses_logit_transformed_inputs(self, code_source):
        assert "def logit(" in code_source

    def test_offers_consensus_and_dispersion_meta_features(self, code_source):
        assert "meta_consensus" in code_source
        assert "meta_dispersion" in code_source
        assert "USE_META_FEATURES" in code_source

    def test_meta_features_come_from_out_of_fold_predictions(self, code_source):
        assert "LEVEL1[k]['oof_per_repeat'][repeat, rows]" in code_source

    def test_level3_is_a_simple_combiner_only(self, code_source):
        assert "simple combiner" in code_source
        assert "prohibited" in code_source

    def test_level3_does_not_train_a_flexible_learner(self, generator):
        source = generator.LEVEL2_CODE
        level3 = source[source.index("RUN_LEVEL3") :]
        for flexible in ("LGBMClassifier", "XGBClassifier", "CatBoostClassifier"):
            assert flexible not in level3

    def test_records_meta_coefficients(self, code_source):
        assert "level2_coefficients.csv" in code_source


class TestLeakageContract:
    def test_encoders_fitted_on_training_rows_only(self, code_source):
        assert ".fit(base_tr)" in code_source
        assert ".fit(base_va" not in code_source
        assert ".fit(base_te" not in code_source

    def test_target_encoder_transform_takes_no_labels(self, generator):
        assert "def transform(self, frame):" in generator.ENCODER_CODE
        assert "def transform(self, frame, y" not in generator.ENCODER_CODE

    def test_target_encoder_uses_inner_cross_fitting(self, code_source):
        assert "te.fit_transform(base_tr, y_tr)" in code_source
        assert "te.transform(base_va)" in code_source

    def test_transductive_counting_stays_disabled(self, code_source):
        assert "USE_UNLABELED = False" in code_source

    def test_leaderboard_never_selects(self, generator):
        assert "never selects anything" in generator.HEADER_MD


class TestStatisticalContract:
    def test_uses_ddof_one_and_real_t_quantile(self, code_source):
        assert "ddof=1" in code_source
        assert "stats.t.ppf" in code_source
        assert "1.96" not in code_source

    def test_applies_nadeau_bengio_and_holm(self, code_source):
        assert "1.0 / n + ratio" in code_source
        assert "holm_adjust" in code_source

    def test_measures_seed_spread(self, code_source):
        assert "measure_seed_spread" in code_source
        assert "within run-to-run variation" in code_source

    def test_promotion_requires_minimum_lift_not_just_significance(self, code_source):
        assert "verdicts[n]['mean_delta'] >= CFG.MIN_LIFT" in code_source

    def test_never_claims_equivalence(self, generator):
        """A zero-covering interval must read as inconclusive, never as equivalence.

        Checked against the verdict strings themselves. The word "equivalent"
        appears elsewhere in the notebook only to say that the reference recipe
        is the Arm C equivalent, which is a naming statement, not a claim about
        statistical equivalence.
        """
        import re

        verdicts = re.findall(r"verdict = '([^']+)'", generator.STATS_CODE)
        assert "noise / inconclusive" in verdicts
        assert not any("equivalent" in v.lower() for v in verdicts)

    def test_retaining_reference_is_valid(self, code_source):
        assert "complete and valid result" in code_source


class TestResourceContract:
    def test_smoke_defaults_on_and_is_labelled_invalid(self, generator, code_source):
        assert "SMOKE = True" in generator.CFG_CODE
        assert "NOT valid evidence" in code_source

    def test_gpu_path_configurable(self, code_source):
        assert "USE_GPU" in code_source
        assert "device'] = 'cuda'" in code_source

    def test_budget_guard_skips_rather_than_dies(self, code_source):
        assert "budget_remaining()" in code_source
        assert "budget exhausted" in code_source

    def test_failed_candidate_does_not_abort_run(self, code_source):
        assert "except Exception as exc:" in code_source
        assert "FAILED" in code_source

    def test_catboost_is_optional(self, code_source):
        assert "HAS_CATBOOST" in code_source


class TestArtifactContract:
    @pytest.mark.parametrize(
        "artifact",
        [
            "decision_ledger.csv",
            "level1_metrics.csv",
            "level1_correlations.csv",
            "oof_predictions.parquet",
            "run_manifest.json",
            "submission.csv",
        ],
    )
    def test_writes_required_artifact(self, code_source, artifact):
        assert artifact in code_source

    def test_validates_submission_ranges(self, code_source):
        assert "out-of-range scores" in code_source
        assert "missing scores" in code_source

    def test_manifest_records_diversity_screen(self, code_source):
        for key in ("selected_members", "min_pair_corr_observed", "max_pair_corr"):
            assert key in code_source

    def test_manifest_records_skipped_candidates(self, code_source):
        assert "'skipped': SKIPPED" in code_source

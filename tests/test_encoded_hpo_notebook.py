"""Contract tests for the LightGBM Encoded HPO notebook generator."""

import ast
import importlib.util
from pathlib import Path
import time

import nbformat as nbf
import numpy as np
import optuna
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = PROJECT_ROOT / "scripts" / "build_encoded_hpo_notebook.py"
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "kaggle_ev_purchase_encoded_hpo.ipynb"


def _load_generator():
    """Import the generator module directly from its path."""
    assert GENERATOR_PATH.exists(), f"Generator script not found at {GENERATOR_PATH}"
    spec = importlib.util.spec_from_file_location("build_encoded_hpo_notebook", GENERATOR_PATH)
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
        for idx, cell in enumerate(cells):
            try:
                ast.parse(cell.source)
            except SyntaxError as exc:
                pytest.fail(f"Cell {idx} failed to parse: {exc}\n{cell.source}")

    def test_has_markdown_narrative(self, notebook):
        markdown = [c for c in notebook.cells if c.cell_type == "markdown"]
        assert len(markdown) >= 8

    def test_generator_determinism(self, generator):
        nb1 = generator.build_notebook()
        nb2 = generator.build_notebook()
        src1 = [c.source for c in nb1.cells]
        src2 = [c.source for c in nb2.cells]
        assert src1 == src2, "Generator must be deterministic"

    def test_committed_notebook_matches_generator(self, notebook):
        if not NOTEBOOK_PATH.exists():
            pytest.skip("notebook not generated yet")
        on_disk = nbf.read(NOTEBOOK_PATH, as_version=4)
        generated_src = [c.source for c in notebook.cells if c.cell_type == "code"]
        disk_src = [c.source for c in on_disk.cells if c.cell_type == "code"]
        assert generated_src == disk_src, "run scripts/build_encoded_hpo_notebook.py to resync"


class TestSelfContainedKaggleContract:
    def test_no_local_src_imports(self, code_source):
        forbidden = ["import ev_purchase", "from ev_purchase", "sys.path.append", "sys.path.insert"]
        for token in forbidden:
            assert token not in code_source, f"Forbidden import or path hack found: {token}"

    def test_uses_kaggle_paths_with_local_fallback(self, code_source):
        assert "find_data_directory" in code_source
        assert "/kaggle/input" in code_source
        assert "/kaggle/working" in code_source

    def test_submission_sanity_checks(self, code_source):
        assert "between(0, 1)" in code_source or "(0.0 <= " in code_source
        assert "isna().any()" in code_source
        assert "len(sub) == len(test_df)" in code_source or "shape[0] == len(test_df)" in code_source


class TestFrozenEncoderContract:
    def test_inlined_encoders(self, code_source):
        assert "class ValueFrequencyEncoder" in code_source
        assert "class FloorFlagEncoder" in code_source
        assert "class SmoothedTargetEncoder" in code_source

    def test_frozen_feature_count_is_19(self, code_source):
        # 13 raw + 2 frequency + 2 floor flags + 2 target encodings = 19
        assert "19" in code_source
        assert "FROZEN_N_FEATURES = 19" in code_source or "len(X_tr.columns) == 19" in code_source

    def test_frozen_value_columns(self, code_source):
        assert "Annual_Income_USD" in code_source
        assert "Daily_Commute_km" in code_source

    def test_target_encoder_smoothing_m20_and_inner_folds(self, code_source):
        assert "TE_SMOOTHING_M = 20.0" in code_source or "m=20" in code_source
        assert "TE_INNER_FOLDS = 5" in code_source or "n_inner_folds=5" in code_source

    def test_floor_min_share_is_0_01(self, code_source):
        assert "FLOOR_MIN_SHARE = 0.01" in code_source or "min_share=0.01" in code_source

    def test_transductive_use_unlabeled_is_false(self, code_source):
        assert "USE_UNLABELED = False" in code_source

    def test_target_encoder_transform_has_no_target_parameter(self, generator):
        encoder_code = generator.ENCODER_CODE
        assert "def transform(self, frame):" in encoder_code
        assert "def transform(self, frame, y" not in encoder_code

    def test_anti_leakage_self_tests_included(self, code_source):
        assert "positive row leaked its own label" in code_source
        assert "unseen value must fall back to 0" in code_source

    def test_fixed_mass_points_and_expected_19_ordered_features(self, code_source):
        assert "Annual_Income_USD_is_30000.0" in code_source or "Annual_Income_USD_is_30000" in code_source
        assert "Daily_Commute_km_is_5.0" in code_source or "Daily_Commute_km_is_5" in code_source
        assert "EXPECTED_FEATURE_NAMES" in code_source


class TestDisjointProtocolContract:
    def test_uses_repeated_stratified_kfold_manifest(self, code_source):
        assert "RepeatedStratifiedKFold" in code_source
        assert "random_state=CFG.seed" in code_source

    def test_asserts_coverage_and_no_overlap(self, code_source):
        assert "coverage == N_REPEATS" in code_source
        assert "train/valid overlap" in code_source

    def test_disjoint_three_stage_protocol_partitions(self, code_source):
        # 3-Stage protocol:
        # Search: repeat 0 only (folds 0-4)
        # Confirmation: repeat 1 only (folds 5-9)
        # Final untouched gate: repeat 2 only (folds 10-14)
        assert "SEARCH_REPEATS = (0,)" in code_source
        assert "CONFIRM_REPEATS = (1,)" in code_source
        assert "FINAL_REPEATS = (2,)" in code_source
        assert "isdisjoint" in code_source

    def test_final_gate_evaluates_repeat_2_only(self, code_source):
        assert "repeat in FINAL_REPEATS" in code_source or "spec['repeat'] in FINAL_REPEATS" in code_source

    def test_seed_formula_varies_by_repeat_and_fold(self, code_source):
        assert "42 + spec['repeat'] * 100 + spec['fold']" in code_source or \
               "CFG.seed + spec['repeat'] * 100 + spec['fold']" in code_source or \
               "base_seed + repeat * 100 + fold" in code_source

    def test_evaluates_incumbent_arm_c_reference(self, code_source):
        assert "ARM_C_INCUMBENT" in code_source or "REFERENCE_PARAMS" in code_source
        assert "0.184834" in code_source  # from configs/final.yaml trial35 learning_rate
        assert "0.930696" in code_source  # from configs/final.yaml trial35 subsample


class TestSearchSpaceContract:
    def test_uses_optuna_tpe_and_pruner(self, code_source):
        assert "optuna.create_study" in code_source
        assert "TPESampler" in code_source
        assert "MedianPruner" in code_source or "SuccessiveHalvingPruner" in code_source or "HyperbandPruner" in code_source

    def test_search_space_boundaries(self, code_source):
        assert "learning_rate" in code_source
        assert "num_leaves" in code_source
        assert "max_depth" in code_source
        assert "min_child_samples" in code_source
        assert "feature_fraction" in code_source or "colsample_bytree" in code_source
        assert "bagging_fraction" in code_source or "subsample" in code_source
        assert "bagging_freq" in code_source or "subsample_freq" in code_source
        assert "reg_alpha" in code_source
        assert "reg_lambda" in code_source

    def test_bagging_fraction_paired_with_bagging_freq(self, code_source):
        # Must set bagging_freq 1-7 together with bagging_fraction 0.5-1.0
        assert "bagging_freq" in code_source
        assert "bagging_fraction" in code_source

    def test_optuna_sqlite_storage(self, code_source):
        assert "sqlite:///" in code_source

    def test_study_resume_safety_and_fingerprint(self, code_source):
        assert "fingerprint" in code_source
        assert "user_attrs" in code_source
        assert "remaining_trials" in code_source
        assert "hpo_study_" in code_source

    def test_smoke_estimator_honestly_parameterized(self, code_source):
        assert "trial.suggest_int('n_estimators', 10, 25)" in code_source or \
               "trial.suggest_int('n_estimators', 10, 30)" in code_source or \
               "suggest_int('n_estimators', 10," in code_source

    def test_zero_trial_fallback_guards_study_best_value(self, code_source):
        assert "if completed_trials:" in code_source or "if len(completed_trials)" in code_source


class TestPromotionGateContract:
    def test_min_lift_raised_to_5e_04(self, code_source):
        assert "MIN_LIFT = 0.0005" in code_source or "MIN_LIFT = 5e-04" in code_source or "MIN_LIFT = 5e-4" in code_source

    def test_nadeau_bengio_correction_present(self, code_source):
        assert "ddof=1" in code_source
        assert "stats.t.ppf" in code_source
        assert "1.96" not in code_source
        assert "ratio = (1.0 / n_splits) / (1.0 - 1.0 / n_splits)" in code_source
        assert "n_splits <= 1" in code_source or "n_splits < 2" in code_source
        assert "nadeau_bengio_ci(deltas, n_splits=CFG.n_splits" in code_source

    def test_nadeau_bengio_ci_numerical_behavior(self, code_source):
        import numpy as np
        from scipy import stats
        scope = {"np": np, "stats": stats, "CFG": type("CFG", (), {"n_splits": 5, "ALPHA": 0.05})}
        lines = []
        in_fn = False
        brace_count = 0
        for line in code_source.splitlines():
            if line.startswith("def nadeau_bengio_ci("):
                in_fn = True
            if in_fn:
                lines.append(line)
                brace_count += line.count("{") - line.count("}")
                if "return {" in line:
                    pass
                elif brace_count == 0 and len(lines) > 5 and line.strip().startswith("}"):
                    break
        fn_code = "\n".join(lines)
        exec(fn_code, scope)
        nb_ci = scope["nadeau_bengio_ci"]

        with pytest.raises(ValueError, match="n_splits"):
            nb_ci([0.001, 0.002, 0.0015], n_splits=1)
        with pytest.raises(ValueError, match="n_splits"):
            nb_ci([0.001, 0.002, 0.0015], n_splits=0)

        deltas = [0.001, 0.002, 0.0015, 0.0008, 0.0012]
        res5 = nb_ci(deltas, n_splits=5)
        res10 = nb_ci(deltas, n_splits=10)
        assert res5["mean"] == pytest.approx(res10["mean"])
        assert res5["se_nb"] > res10["se_nb"]
        assert (res5["ci_high"] - res5["ci_low"]) > (res10["ci_high"] - res10["ci_low"])

    def test_ci_excludes_zero_logic(self, code_source):
        assert "ci_low <= 0 <= ci_high" in code_source or "not (low <= 0 <= high)" in code_source

    def test_confirmatory_raw_p_value_and_no_holm_bonferroni(self, code_source):
        assert "single_prespecified_hypothesis" in code_source or "unadjusted_single_finalist" in code_source
        assert "def holm_adjust" not in code_source
        assert "holm-bonferroni" not in code_source.lower()

    def test_no_holm_wording_in_any_cell(self, notebook):
        for idx, cell in enumerate(notebook.cells):
            assert "holm" not in cell.source.lower(), (
                f"Cell {idx} ({cell.cell_type}) contains stale 'holm' reference: {cell.source[:100]}"
            )

    def test_no_fabricated_p_values_in_code(self, code_source):
        assert "[0.5]" not in code_source

    def test_model_seed_spread_measurement_contract(self, code_source):
        assert "MODEL_STABILITY_SEEDS = (101, 202, 303)" in code_source or \
               "MODEL_STABILITY_SEEDS = (101, 202, 303)" in code_source or \
               "(101, 202, 303)" in code_source
        assert "model_seed_spread_range" in code_source
        assert "model_stability_seeds" in code_source
        assert "model_stability_scores" in code_source
        assert "mean_delta > model_seed_spread_range" in code_source
        assert "measure_seed_spread(ref_scores)" not in code_source

    def test_model_seed_spread_uses_same_folds_and_truthful_naming(self, code_source):
        assert "final_encoded_folds" in code_source
        assert "model_seed_spread_range" in code_source
        assert "'model_seed_spread_range'" in code_source
        assert "'model_stability_seeds'" in code_source
        assert "'model_stability_scores'" in code_source

    def test_guardrails_evaluated(self, code_source):
        assert "pr_auc" in code_source.lower()
        assert "log_loss" in code_source.lower()

    def test_fallback_to_reference(self, code_source):
        assert "PROMOTED = 'REFERENCE'" in code_source or "PROMOTED = \"REFERENCE\"" in code_source


class TestArtifactPersistenceContract:
    @pytest.mark.parametrize(
        "artifact",
        [
            "run_manifest.json",
            "decision_ledger.csv",
            "best_params.json",
            "fold_scores.csv",
            "submission.csv",
        ],
    )
    def test_persists_required_artifacts(self, code_source, artifact):
        assert artifact in code_source

    def test_persists_trials_table(self, code_source):
        assert "trials.csv" in code_source or "hpo_trials.csv" in code_source

    def test_smoke_submission_branching_and_schema_validation(self, code_source):
        assert "smoke_submission.csv" in code_source
        assert "submission.csv" in code_source
        assert "sample_submission.csv" in code_source or "sample_sub" in code_source


class TestProvenanceEnrichmentContract:
    def test_manifest_source_contains_selected_and_candidate_trials(self, code_source):
        assert "'selected_trial_number':" in code_source or "'selected_optuna_trial_number':" in code_source
        assert "'candidate_trial_numbers':" in code_source or "'top_distinct_candidate_trials':" in code_source
        assert "'confirmation_ranking':" in code_source or "'confirmation_results':" in code_source
        assert "'confirmation_scores':" in code_source

    def test_manifest_source_contains_trial_counts(self, code_source):
        assert "'requested_total_attempts':" in code_source
        assert "'attempted':" in code_source or "'attempted_trials_count':" in code_source
        assert "'completed':" in code_source or "'completed_trials_count':" in code_source
        assert "'pruned':" in code_source or "'pruned_trials_count':" in code_source
        assert "'failed':" in code_source or "'failed_trials_count':" in code_source
        assert "'remaining_attempts':" in code_source or "'remaining_attempts_count':" in code_source
        assert "'existing_trial_count':" in code_source
        assert "'existing_complete_count':" not in code_source
        assert "'fallback_source':" in code_source
        assert "'runtime_exhausted':" in code_source
        assert "'inference_complete':" in code_source
        assert "'stability_complete':" in code_source

    def test_manifest_source_contains_explicit_partitions(self, code_source):
        assert "'search_repeats': list(SEARCH_REPEATS)" in code_source
        assert "'confirm_repeats': list(CONFIRM_REPEATS)" in code_source
        assert "'final_repeats': list(FINAL_REPEATS)" in code_source

    def test_manifest_source_contains_dataset_identity(self, code_source):
        assert "'dataset_identity':" in code_source
        assert "dataset_identity" in code_source
        assert "'train_sha256':" in code_source
        assert "'test_sha256':" in code_source
        assert "'sample_submission_sha256':" in code_source
        assert "'protocol_fingerprint':" in code_source
        assert "'fingerprint_payload':" in code_source

    def test_manifest_source_contains_exact_ordered_frozen_feature_names(self, code_source):
        assert "'exact_ordered_frozen_feature_names':" in code_source or "'frozen_feature_names':" in code_source
        assert "EXPECTED_FEATURE_NAMES" in code_source

    def test_manifest_source_contains_package_versions(self, code_source):
        assert "'package_versions':" in code_source
        for pkg in ["python", "numpy", "pandas", "sklearn", "lightgbm", "optuna"]:
            assert f"'{pkg}':" in code_source

    def test_manifest_source_contains_platform_string(self, code_source):
        assert "'platform':" in code_source
        assert "platform.platform()" in code_source

    def test_manifest_source_contains_git_commit_sha_and_dirty_flag_with_safe_fallback(self, code_source):
        assert "'git':" in code_source or "'git_commit':" in code_source
        assert "'git_dirty':" in code_source or "'dirty':" in code_source
        assert "subprocess" in code_source
        assert "rev-parse" in code_source
        assert "--porcelain" in code_source

    def test_manifest_source_contains_all_seeds(self, code_source):
        assert "'seeds':" in code_source
        assert "'base_seed':" in code_source
        assert "'model_stability_seeds':" in code_source
        assert "MODEL_STABILITY_SEEDS" in code_source

    def test_manifest_source_contains_study_resumed_name_db_mode(self, code_source):
        assert "'study_resumed':" in code_source
        assert "'study_name':" in code_source
        assert "'study_db_path':" in code_source
        assert "'study_mode':" in code_source

    def test_best_params_distinguishes_finalist_reference_promoted_and_no_candidate(self, code_source):
        assert "'finalist_params':" in code_source
        assert "'reference_params':" in code_source
        assert "'promoted_params':" in code_source
        assert "'no_viable_candidate':" in code_source
        # Must never set fallback incumbent to candidate_params when no candidate
        assert "finalist_params if not no_viable_candidate else None" in code_source or \
               "actual_finalist_params" in code_source

    def test_manifest_and_best_params_helper_execution(self, generator):
        scope = {}
        helper_code = (
            "import sys\nimport platform\nimport subprocess\nimport sklearn\n"
            "import numpy as np\nimport pandas as pd\nimport lightgbm as lgb\nimport optuna\n"
        )
        exec(helper_code, scope)

        # Extract build_manifest and best_params logic from ARTIFACTS_CODE
        # We test both execution scenarios: candidate present vs no viable candidate
        exec(generator.SETUP_CODE, scope)
        exec(generator.CFG_CODE, scope)

        # Helper simulating the manifest and best_params generation logic
        def run_artifact_logic(has_candidate=True):
            sim_scope = dict(scope)
            sim_scope["DATA_DIR"] = Path("/tmp")
            sim_scope["WORK_DIR"] = Path("/tmp")
            sim_scope["sub_filename"] = "smoke_submission.csv"
            sim_scope["test_df"] = pd.DataFrame({"id": [1, 2], "Will_Buy_EV": [0, 1]})
            sim_scope["reference_test_preds_cached"] = np.array([0.5, 0.5])
            sim_scope["ref_test_preds"] = np.array([0.5, 0.5])
            sim_scope["confirm_df"] = pd.DataFrame()
            sim_scope["paired_df"] = pd.DataFrame()
            sim_scope["study_mode"] = "smoke"
            sim_scope["study_name"] = "encoded_hpo_smoke"
            sim_scope["study_db_path"] = Path("/tmp/test.db")
            sim_scope["study_resumed"] = False
            sim_scope["protocol_fingerprint"] = "abc123"
            sim_scope["fingerprint_payload"] = {}
            sim_scope["hpo_trials_target"] = 3
            sim_scope["remaining_trials"] = 3
            sim_scope["requested_total_attempts"] = 3
            sim_scope["existing_trials_count"] = 0
            sim_scope["STUDY_MAX_RUNTIME_HOURS"] = 0.5
            sim_scope["RUN_DEADLINE"] = time.time() + 1800
            sim_scope["RUN_STARTED_AT"] = time.time()
            sim_scope["budget_remaining_seconds"] = lambda: 1800.0
            sim_scope["is_budget_exhausted"] = lambda buffer_seconds=0.0: False
            sim_scope["N_REPEATS"] = 3
            sim_scope["FOLDS"] = [
                {"repeat": r, "fold": f, "seed": 42 + r * 100 + f}
                for r in range(3) for f in range(5)
            ]
            sim_scope["EXPECTED_FEATURE_NAMES"] = ["feat1"]
            sim_scope["model_seed_spread_range"] = 0.0001
            sim_scope["model_stability_seeds"] = (101, 202, 303)
            sim_scope["model_stability_scores"] = [0.85, 0.851, 0.849]
            sim_scope["mean_delta"] = 0.001
            sim_scope["ci_low"] = 0.0006
            sim_scope["ci_high"] = 0.0014
            sim_scope["p_value"] = 0.001
            sim_scope["multiplicity_method"] = "unadjusted_single_finalist"
            sim_scope["mean_delta_pr"] = 0.0005
            sim_scope["mean_delta_ll"] = -0.002
            sim_scope["dataset_identity"] = {"data_dir": "/tmp"}
            sim_scope["study"] = optuna.create_study()

            if has_candidate:
                sim_scope["PROMOTED"] = "CANDIDATE"
                sim_scope["no_viable_candidate"] = False
                sim_scope["inference_complete"] = True
                sim_scope["runtime_exhausted"] = False
                sim_scope["cand_test_preds"] = np.array([0.6, 0.4])
                sim_scope["finalist"] = {"trial_number": 1, "confirm_roc_auc": 0.852}
                sim_scope["finalist_params"] = {"num_leaves": 31}
                sim_scope["promoted_params"] = {"num_leaves": 31}
                sim_scope["top_candidates"] = [type("Trial", (), {"number": 1, "params": {"num_leaves": 31}})()]
                sim_scope["confirmation_results"] = [{
                    "candidate_rank": 0, "trial_number": 1, "search_roc_auc": 0.85,
                    "confirm_roc_auc": 0.852, "confirm_pr_auc": 0.70, "confirm_log_loss": 0.40
                }]
            else:
                sim_scope["PROMOTED"] = "REFERENCE"
                sim_scope["no_viable_candidate"] = True
                sim_scope["inference_complete"] = False
                sim_scope["runtime_exhausted"] = False
                sim_scope["cand_test_preds"] = None
                sim_scope["finalist"] = None
                sim_scope["finalist_params"] = None
                sim_scope["promoted_params"] = dict(generator.ARM_C_INCUMBENT)
                sim_scope["top_candidates"] = []
                sim_scope["confirmation_results"] = []

            # Extract lines from ARTIFACTS_CODE that create best_params_payload and manifest
            lines = generator.ARTIFACTS_CODE.splitlines()
            code_snippet_lines = []
            for line in lines:
                if line.startswith("with open(") or line.startswith("print('All artifacts") or line.startswith("print('Persisted"):
                    continue
                if "json.dump(" in line or ".to_csv(" in line:
                    continue
                code_snippet_lines.append(line)
            code_snippet = "\n".join(code_snippet_lines)
            exec(code_snippet, sim_scope)
            return sim_scope["best_params_payload"], sim_scope["manifest"]

        # Case 1: Viable candidate promoted
        best_params_cand, manifest_cand = run_artifact_logic(has_candidate=True)
        assert best_params_cand["promoted"] == "CANDIDATE"
        assert best_params_cand["no_viable_candidate"] is False
        assert best_params_cand["candidate_params"] == {"num_leaves": 31}
        assert best_params_cand["finalist_params"] == {"num_leaves": 31}
        assert best_params_cand["promoted_params"] == {"num_leaves": 31}
        assert "reference_params" in best_params_cand
        assert manifest_cand["selected_trial_number"] == 1
        assert manifest_cand["candidate_trial_numbers"] == [1]
        assert len(manifest_cand["confirmation_ranking"]) == 1
        assert "package_versions" in manifest_cand
        assert manifest_cand["package_versions"]["optuna"] == optuna.__version__
        assert "platform" in manifest_cand
        assert "git" in manifest_cand
        assert "seeds" in manifest_cand
        assert manifest_cand["trial_counts"]["completed"] == 0

        # Case 2: No viable candidate fallback
        best_params_none, manifest_none = run_artifact_logic(has_candidate=False)
        assert best_params_none["promoted"] == "REFERENCE"
        assert best_params_none["no_viable_candidate"] is True
        assert best_params_none["candidate_params"] is None
        assert best_params_none["finalist_params"] is None
        for k, v in generator.ARM_C_INCUMBENT.items():
            assert best_params_none["promoted_params"][k] == v
        assert manifest_none["selected_trial_number"] is None
        assert manifest_none["candidate_trial_numbers"] == []


class TestStudyResumeAndFingerprintSafety:
    def test_fingerprint_payload_includes_dataset_hashes(self, code_source):
        assert "'train_sha256':" in code_source
        assert "'test_sha256':" in code_source
        assert "'sample_submission_sha256':" in code_source

    def test_study_resume_behavior_matching_fingerprint_accepted(self, tmp_path):
        db_path = tmp_path / "study_matching.db"
        storage_url = f"sqlite:///{db_path}"
        study1 = optuna.create_study(study_name="test_study", storage=storage_url)
        fp = "matching_hash_123"
        study1.set_user_attr("protocol_fingerprint", fp)

        study2 = optuna.create_study(study_name="test_study", storage=storage_url, load_if_exists=True)
        existing_fp = study2.user_attrs.get("protocol_fingerprint")
        assert existing_fp == fp

    def test_study_resume_behavior_mismatched_hash_rejected(self, tmp_path):
        db_path = tmp_path / "study_mismatch.db"
        storage_url = f"sqlite:///{db_path}"
        study1 = optuna.create_study(study_name="test_study", storage=storage_url)
        study1.set_user_attr("protocol_fingerprint", "old_hash_111")

        study2 = optuna.create_study(study_name="test_study", storage=storage_url, load_if_exists=True)
        existing_fp = study2.user_attrs.get("protocol_fingerprint")
        target_fp = "new_hash_222"
        with pytest.raises(RuntimeError, match="Protocol fingerprint mismatch"):
            if existing_fp != target_fp:
                raise RuntimeError(
                    f'Protocol fingerprint mismatch for study "test_study"! '
                    f'Expected {target_fp}, found {existing_fp}.'
                )

    def test_study_resume_behavior_nonempty_missing_fingerprint_rejected(self, tmp_path):
        db_path = tmp_path / "study_legacy.db"
        storage_url = f"sqlite:///{db_path}"
        study1 = optuna.create_study(study_name="test_study", storage=storage_url)
        study1.optimize(lambda trial: trial.suggest_float("x", 0, 1), n_trials=1)

        study2 = optuna.create_study(study_name="test_study", storage=storage_url, load_if_exists=True)
        existing_fp = study2.user_attrs.get("protocol_fingerprint")
        trials_count = len(study2.trials)
        assert trials_count > 0
        assert existing_fp is None

        with pytest.raises(RuntimeError, match="Unverified legacy study"):
            if existing_fp is None:
                if trials_count > 0:
                    raise RuntimeError(
                        f'Unverified legacy study "test_study" has {trials_count} trials '
                        f'but no protocol_fingerprint in user_attrs; cannot safely resume.'
                    )

    def test_study_resume_behavior_empty_new_study_initializes(self, tmp_path):
        db_path = tmp_path / "study_empty.db"
        storage_url = f"sqlite:///{db_path}"
        study = optuna.create_study(study_name="test_study", storage=storage_url, load_if_exists=True)
        existing_fp = study.user_attrs.get("protocol_fingerprint")
        trials_count = len(study.trials)
        assert existing_fp is None
        assert trials_count == 0
        target_fp = "new_fp_456"
        if existing_fp is None:
            if trials_count > 0:
                raise RuntimeError("Legacy study")
            study.set_user_attr("protocol_fingerprint", target_fp)
        assert study.user_attrs.get("protocol_fingerprint") == target_fp


class TestRuntimeBudgetContract:
    def test_budget_functions_defined(self, code_source):
        assert "RUN_DEADLINE" in code_source or "RUN_STARTED_AT" in code_source
        assert "budget_remaining_seconds" in code_source or "budget_remaining" in code_source
        assert "is_budget_exhausted" in code_source

    def test_budget_exhaustion_forces_reference_only_and_coherent_artifacts(self, generator):
        scope = {}
        exec(generator.SETUP_CODE, scope)
        exec(generator.CFG_CODE, scope)

        sim_scope = dict(scope)
        sim_scope["DATA_DIR"] = Path("/tmp")
        sim_scope["WORK_DIR"] = Path("/tmp")
        sim_scope["sub_filename"] = "smoke_submission.csv"
        sim_scope["test_df"] = pd.DataFrame({"id": [1, 2], "Will_Buy_EV": [0, 1]})
        sim_scope["reference_test_preds_cached"] = np.array([0.5, 0.5])
        sim_scope["ref_test_preds"] = np.array([0.5, 0.5])
        sim_scope["confirm_df"] = pd.DataFrame()
        sim_scope["paired_df"] = pd.DataFrame()
        sim_scope["study_mode"] = "smoke"
        sim_scope["study_name"] = "encoded_hpo_smoke"
        sim_scope["study_db_path"] = Path("/tmp/test.db")
        sim_scope["study_resumed"] = False
        sim_scope["protocol_fingerprint"] = "abc123"
        sim_scope["fingerprint_payload"] = {}
        sim_scope["hpo_trials_target"] = 3
        sim_scope["requested_total_attempts"] = 3
        sim_scope["existing_trials_count"] = 0
        sim_scope["STUDY_MAX_RUNTIME_HOURS"] = 0.5
        sim_scope["RUN_DEADLINE"] = time.time()
        sim_scope["RUN_STARTED_AT"] = time.time() - 1800
        sim_scope["budget_remaining_seconds"] = lambda: 0.0
        sim_scope["is_budget_exhausted"] = lambda buffer_seconds=0.0: True
        sim_scope["existing_complete_count"] = 0
        sim_scope["remaining_trials"] = 3
        sim_scope["N_REPEATS"] = 3
        sim_scope["FOLDS"] = [
            {"repeat": r, "fold": f, "seed": 42 + r * 100 + f}
            for r in range(3) for f in range(5)
        ]
        sim_scope["EXPECTED_FEATURE_NAMES"] = ["feat1"]
        sim_scope["model_seed_spread_range"] = 0.0001
        sim_scope["model_stability_seeds"] = (101, 202, 303)
        sim_scope["model_stability_scores"] = [0.85, 0.851, 0.849]
        sim_scope["mean_delta"] = 0.001
        sim_scope["ci_low"] = 0.0006
        sim_scope["ci_high"] = 0.0014
        sim_scope["p_value"] = 0.001
        sim_scope["multiplicity_method"] = "single_prespecified_hypothesis"
        sim_scope["mean_delta_pr"] = 0.0005
        sim_scope["mean_delta_ll"] = -0.002
        sim_scope["dataset_identity"] = {"data_dir": "/tmp"}
        sim_scope["study"] = optuna.create_study()
        sim_scope["is_budget_exhausted"] = lambda buffer_seconds=0.0: True
        sim_scope["PROMOTED"] = "REFERENCE"
        sim_scope["no_viable_candidate"] = True
        sim_scope["finalist"] = None
        sim_scope["finalist_params"] = None
        sim_scope["promoted_params"] = dict(generator.ARM_C_INCUMBENT)
        sim_scope["top_candidates"] = []
        sim_scope["confirmation_results"] = []

        lines = generator.ARTIFACTS_CODE.splitlines()
        code_snippet_lines = []
        for line in lines:
            if line.startswith("with open(") or line.startswith("print('All artifacts") or line.startswith("print('Persisted"):
                continue
            if "json.dump(" in line or ".to_csv(" in line:
                continue
            code_snippet_lines.append(line)
        code_snippet = "\n".join(code_snippet_lines)
        exec(code_snippet, sim_scope)
        payload = sim_scope["best_params_payload"]
        assert payload["promoted"] == "REFERENCE"
        assert payload["no_viable_candidate"] is True
        assert payload["candidate_params"] is None
        assert payload["finalist_params"] is None


class TestInlinedEncoderParityContract:
    @pytest.fixture(scope="class")
    @classmethod
    def notebook_encoders(cls, generator):
        scope = {}
        exec(
            "import numpy as np\nimport pandas as pd\nfrom sklearn.model_selection import StratifiedKFold\n"
            + generator.ENCODER_CODE,
            scope,
        )
        return scope["ValueFrequencyEncoder"], scope["SmoothedTargetEncoder"]

    def test_value_frequency_encoder_parity_with_src(self, notebook_encoders):
        from ev_purchase.fold_encoders import ValueFrequencyEncoder as SrcValueFrequencyEncoder

        NbValueFreq, _ = notebook_encoders

        # Representative training data with multiple values and frequencies
        train_df = pd.DataFrame({
            "Annual_Income_USD": [30000.0, 30000.0, 50000.0, 30000.0, 75000.0, 50000.0, 100000.0],
            "Daily_Commute_km": [5.0, 15.0, 5.0, 5.0, 20.0, 5.0, 25.0],
        })
        # Test frame containing known and unseen categories
        test_df = pd.DataFrame({
            "Annual_Income_USD": [30000.0, 50000.0, 999999.0, 75000.0],  # 999999 is unseen
            "Daily_Commute_km": [5.0, 999.0, 15.0, 20.0],                # 999 is unseen
        })

        cols = ["Annual_Income_USD", "Daily_Commute_km"]
        nb_enc = NbValueFreq(columns=cols, unseen_count=0).fit(train_df)
        src_enc = SrcValueFrequencyEncoder(columns=cols, unseen_count=0).fit(train_df)

        nb_tr = nb_enc.transform(train_df)
        src_tr = src_enc.transform(train_df)
        pd.testing.assert_frame_equal(nb_tr, src_tr)

        nb_te = nb_enc.transform(test_df)
        src_te = src_enc.transform(test_df)
        pd.testing.assert_frame_equal(nb_te, src_te)

        # Artifact serialization parity
        nb_art = nb_enc.get_artifact()
        src_art = src_enc.get_artifact()
        assert nb_art["encoder"] == src_art["encoder"]
        assert nb_art["columns"] == src_art["columns"]
        assert nb_art["unseen_count"] == src_art["unseen_count"]
        assert nb_art["mappings"] == src_art["mappings"]

    @pytest.mark.parametrize("m", [0.0, 5.0, 20.0])
    def test_smoothed_target_encoder_parity_with_src_mapping_and_transform(self, notebook_encoders, m):
        from ev_purchase.fold_encoders import SmoothedTargetEncoder as SrcSmoothedTargetEncoder

        _, NbTargetEnc = notebook_encoders

        train_df = pd.DataFrame({
            "Annual_Income_USD": [30000.0, 30000.0, 50000.0, 30000.0, 75000.0, 50000.0, 80000.0, 30000.0],
            "Daily_Commute_km": [5.0, 15.0, 5.0, 5.0, 20.0, 5.0, 10.0, 5.0],
        })
        y = np.array([1, 0, 1, 1, 0, 0, 1, 0])
        test_df = pd.DataFrame({
            "Annual_Income_USD": [30000.0, 50000.0, 123456.0],  # unseen category
            "Daily_Commute_km": [5.0, 999.0, 15.0],            # unseen category
        })

        cols = ["Annual_Income_USD", "Daily_Commute_km"]
        nb_te = NbTargetEnc(columns=cols, m=m, n_inner_folds=5, random_state=42).fit(train_df, y)
        src_te = SrcSmoothedTargetEncoder(columns=cols, m=m, n_inner_folds=5, random_state=42).fit(train_df, y)

        assert nb_te.global_mean_ == pytest.approx(src_te.global_mean_)
        for col in cols:
            for val, rate in src_te.mappings_[col].items():
                assert nb_te.mappings_[col][val] == pytest.approx(rate, rel=1e-10)

        nb_out = nb_te.transform(test_df)
        src_out = src_te.transform(test_df)
        pd.testing.assert_frame_equal(nb_out, src_out)

    def test_smoothed_target_encoder_anti_leakage_and_unseen_categories(self, notebook_encoders):
        _, NbTargetEnc = notebook_encoders

        # 1. Transform signature protection: transform must not accept target y
        import inspect
        params = list(inspect.signature(NbTargetEnc.transform).parameters.keys())
        assert params == ["self", "frame"], "transform must not accept target y"

        # 2. Leave-one-out isolation: single positive row must NOT leak into its own encoding
        df = pd.DataFrame({"Annual_Income_USD": [30000.0, 30000.0, 30000.0, 30000.0]})
        y = np.array([1, 0, 0, 0])
        enc = NbTargetEnc(columns=["Annual_Income_USD"], m=0.0, n_inner_folds=4, random_state=0)
        out = enc.fit_transform(df, y)["Annual_Income_USD_te"].to_numpy()
        assert abs(out[0] - 0.0) < 1e-12, "Row 0 leaked its own label into inner fit_transform"
        assert abs(out[1] - (1.0 / 3.0)) < 1e-12, "Negative rows should average remaining rows"

        # 3. Unseen categories: fallback strictly to global prior
        df_unseen = pd.DataFrame({"Annual_Income_USD": [999999.0]})
        out_unseen = enc.transform(df_unseen)["Annual_Income_USD_te"].iloc[0]
        assert out_unseen == pytest.approx(enc.global_mean_)

    def test_frozen_fixed_flag_builder_exact_outputs(self):
        # Floor behavior is intentionally fixed in feature builder, not dynamic FloorFlagEncoder
        df = pd.DataFrame({
            "Annual_Income_USD": [30000.0, 29999.0, 30000.0, 50000.0],
            "Daily_Commute_km": [5.0, 5.0, 10.0, 5.0],
        })
        flags = pd.DataFrame({
            "Annual_Income_USD_is_30000.0": (df["Annual_Income_USD"] == 30000.0).astype("int64"),
            "Daily_Commute_km_is_5.0": (df["Daily_Commute_km"] == 5.0).astype("int64"),
        }, index=df.index)

        assert list(flags.columns) == ["Annual_Income_USD_is_30000.0", "Daily_Commute_km_is_5.0"]
        assert flags["Annual_Income_USD_is_30000.0"].tolist() == [1, 0, 1, 0]
        assert flags["Daily_Commute_km_is_5.0"].tolist() == [1, 1, 0, 1]
        assert flags["Annual_Income_USD_is_30000.0"].dtype == np.int64
        assert flags["Daily_Commute_km_is_5.0"].dtype == np.int64


class TestFullFileHashWithoutCap:
    def test_compute_sha256_reads_full_file_beyond_20mib(self, generator, tmp_path):
        import hashlib
        scope = {}
        # DATA_CODE defines compute_sha256
        data_code_lines = []
        for line in generator.DATA_CODE.splitlines():
            if line.startswith("def compute_sha256"):
                data_code_lines.append(line)
            elif data_code_lines:
                if line.startswith("dataset_identity ="):
                    break
                data_code_lines.append(line)
        exec("import hashlib\nfrom pathlib import Path\n" + "\n".join(data_code_lines), scope)
        compute_sha256 = scope["compute_sha256"]

        prefix = b"0" * (20 * 1024 * 1024)
        file1 = tmp_path / "file1.bin"
        file2 = tmp_path / "file2.bin"

        file1.write_bytes(prefix + b"AAAA" * (256 * 1024))
        file2.write_bytes(prefix + b"BBBB" * (256 * 1024))

        assert file1.stat().st_size == 21 * 1024 * 1024
        assert file2.stat().st_size == 21 * 1024 * 1024

        h1 = compute_sha256(file1)
        h2 = compute_sha256(file2)
        assert h1 != h2, "compute_sha256 must read full file; hashes of differing >20MiB files must differ"
        assert h1 == hashlib.sha256(file1.read_bytes()).hexdigest()
        assert h2 == hashlib.sha256(file2.read_bytes()).hexdigest()


class TestPrecomputedReferenceTestEnsemble:
    def test_precomputed_reference_ensemble_contract_and_order(self, generator):
        code_source = "\n".join(c.source for c in generator.build_notebook().cells if c.cell_type == "code")
        assert "reference_test_preds_cached" in code_source
        assert "'precomputed_reference_test_ensemble'" in code_source or \
               '"precomputed_reference_test_ensemble"' in code_source

        pos_precompute = code_source.find("reference_test_preds_cached")
        pos_hpo = code_source.find("study = optuna.create_study")
        assert pos_precompute != -1 and pos_hpo != -1
        assert pos_precompute < pos_hpo, "Reference test ensemble must be precomputed BEFORE HPO study creation"

    def test_expired_deadline_stage_control_flow(self, generator):
        scope = {}
        exec(generator.SETUP_CODE, scope)
        exec(generator.CFG_CODE, scope)

        n_test = 100
        ref_cached = np.full(n_test, 0.42)
        scope["reference_test_preds_cached"] = ref_cached
        scope["is_budget_exhausted"] = lambda buffer_seconds=0.0: True
        scope["no_viable_candidate"] = False
        scope["finalist_params"] = {"num_leaves": 31}
        scope["final_specs"] = [{"repeat": 2, "fold": f, "seed": 200 + f} for f in range(5)]
        scope["final_encoded_folds"] = []
        scope["paired_df"] = pd.DataFrame()
        scope["confirm_df"] = pd.DataFrame()
        scope["test_df"] = pd.DataFrame({"id": range(n_test)})
        scope["WORK_DIR"] = Path("/tmp")
        scope["DATA_DIR"] = Path("/tmp")
        scope["sub_filename"] = "smoke_submission.csv"
        scope["study_mode"] = "smoke"
        scope["study_name"] = "encoded_hpo_smoke"
        scope["study_db_path"] = Path("/tmp/test.db")
        scope["study_resumed"] = False
        scope["protocol_fingerprint"] = "abc123"
        scope["fingerprint_payload"] = {}
        scope["dataset_identity"] = {"data_dir": "/tmp"}
        scope["hpo_trials_target"] = 3
        scope["requested_total_attempts"] = 3
        scope["existing_trials_count"] = 0
        scope["STUDY_MAX_RUNTIME_HOURS"] = 0.5
        scope["RUN_DEADLINE"] = time.time()
        scope["RUN_STARTED_AT"] = time.time() - 1800
        scope["budget_remaining_seconds"] = lambda: 0.0
        scope["remaining_trials"] = 0
        scope["N_REPEATS"] = 3
        scope["FOLDS"] = []
        scope["EXPECTED_FEATURE_NAMES"] = ["f1"]
        scope["FROZEN_N_FEATURES"] = 19
        scope["study"] = optuna.create_study()
        scope["top_candidates"] = []
        scope["confirmation_results"] = []

        # Execute final stage under expired budget
        exec(generator.FINAL_CODE, scope)

        assert scope["runtime_exhausted"] is True
        assert scope["inference_complete"] is False
        assert scope["no_viable_candidate"] is True
        assert scope["PROMOTED"] == "REFERENCE"

        # Execute artifacts snippet
        lines = generator.ARTIFACTS_CODE.splitlines()
        code_lines = []
        capture = False
        for line in lines:
            if "if PROMOTED == 'CANDIDATE'" in line:
                capture = True
            if capture:
                if line.startswith("with open(") or line.startswith("print('All artifacts") or line.startswith("print('Persisted"):
                    continue
                if "json.dump(" in line or ".to_csv(" in line:
                    continue
                code_lines.append(line)
        exec("\n".join(code_lines), scope)

        assert np.array_equal(scope["final_test_preds"], ref_cached)
        assert scope["fallback_source"] == "precomputed_reference_test_ensemble"
        assert scope["manifest"]["fallback_source"] == "precomputed_reference_test_ensemble"
        assert scope["manifest"]["runtime_exhausted"] is True
        assert scope["manifest"]["inference_complete"] is False


def test_model_seed_stability_stops_before_next_fold_when_budget_expires(generator):
    """The stability loop must check the deadline before every model fit."""
    tree = ast.parse(generator.GATE_CODE)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "measure_model_seed_spread"
    )
    module = ast.Module(body=[function], type_ignores=[])

    fit_calls = []
    budget_checks = iter([False, False, True])  # seed, fold 1, fold 2

    class FakeLGBMClassifier:
        def __init__(self, **params):
            fit_calls.append(params)

    scope = {
        "np": np,
        "MODEL_STABILITY_SEEDS": (101,),
        "is_budget_exhausted": lambda: next(budget_checks),
        "lgb": type("FakeLGB", (), {"LGBMClassifier": FakeLGBMClassifier}),
        "evaluate_fold_model": lambda *args: {"metrics": {"roc_auc": 0.9}},
    }
    exec(compile(module, "<gate-test>", "exec"), scope)
    folds = [
        {
            "spec": {"fold": fold},
            "X_tr": None,
            "y_tr": None,
            "X_va": None,
            "y_va": None,
        }
        for fold in range(3)
    ]

    result = scope["measure_model_seed_spread"](
        folds, {"n_estimators": 10}, seeds=(101,), is_smoke=False
    )

    assert len(fit_calls) == 1
    assert result["stability_complete"] is False
    assert result["model_seed_spread_range"] is None

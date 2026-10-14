"""Tests for LightGBM ablation, HPO, and bagging notebook generator.

Validates the notebook builder, structure, leakage safety, search space,
behavioral transformer properties, locked holdout isolation, and submission contracts.
"""

import ast
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import nbformat
import pandas as pd
import pytest

NOTEBOOK_PATH = Path("notebooks/kaggle_ev_purchase_lgbm_ablation_hpo_bagging.ipynb")
BUILDER_SCRIPT = Path("scripts/build_lgbm_ablation_hpo_bagging_notebook.py")


def test_builder_script_exists() -> None:
    """Builder script must exist in scripts/."""
    assert BUILDER_SCRIPT.is_file(), f"Missing builder script: {BUILDER_SCRIPT}"


def test_checked_in_notebook_matches_generator() -> None:
    """Checked-in notebook sources and metadata must match the generator output."""
    spec = importlib.util.spec_from_file_location("lgbm_notebook_builder", BUILDER_SCRIPT)
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


def test_builder_runs_and_generates_notebook() -> None:
    """Builder script should execute cleanly and create valid notebook."""
    cmd = [sys.executable, str(BUILDER_SCRIPT)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    msg = f"Builder failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    assert result.returncode == 0, msg
    assert NOTEBOOK_PATH.is_file(), f"Notebook was not created at {NOTEBOOK_PATH}"


@pytest.fixture(scope="module")
def notebook() -> nbformat.NotebookNode:
    """Loads notebook node after ensuring builder has run."""
    if not NOTEBOOK_PATH.is_file():
        cmd = [sys.executable, str(BUILDER_SCRIPT)]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    return nbformat.read(NOTEBOOK_PATH, as_version=4)


def test_notebook_validates(notebook: nbformat.NotebookNode) -> None:
    """Notebook must pass standard nbformat schema validation."""
    nbformat.validate(notebook)


def test_notebook_metadata_python3(notebook: nbformat.NotebookNode) -> None:
    """Notebook metadata must specify python3 kernel."""
    kernelspec = notebook.metadata.get("kernelspec", {})
    assert kernelspec.get("name") == "python3" or "python" in kernelspec.get("language", "")


def test_every_code_cell_has_preceding_markdown(notebook: nbformat.NotebookNode) -> None:
    """Every code cell must have a preceding non-empty markdown cell."""
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            assert index > 0, "First cell cannot be code"
            prev_cell = notebook.cells[index - 1]
            err_msg = f"Code cell {index} is preceded by {prev_cell.cell_type}"
            assert prev_cell.cell_type == "markdown", err_msg
            err_empty = f"Preceding markdown cell for code cell {index} is empty"
            assert prev_cell.source.strip(), err_empty


def test_notebook_does_not_contain_hardcoded_user_paths(notebook: nbformat.NotebookNode) -> None:
    """Notebook must not contain hardcoded user home paths or internal agent tokens."""
    content = json.dumps(notebook)
    forbidden = ["/home/", "/Users/", ".agent/skills", "subagent", "Task ID"]
    found = [token for token in forbidden if token in content]
    assert not found, f"Forbidden tokens found in notebook: {found}"


def test_docs_wording_no_target_group_means(notebook: nbformat.NotebookNode) -> None:
    """Notebook docs must say 'numeric group means' and not 'target group means'."""
    content = json.dumps(notebook)
    assert "target group means" not in content.lower()
    assert "numeric group means" in content.lower()


def test_all_code_cells_compile(notebook: nbformat.NotebookNode) -> None:
    """Every code cell source must compile as valid Python."""
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert len(code_cells) >= 6, "Expected at least 6 code cells for end-to-end pipeline"
    for i, cell in enumerate(code_cells):
        try:
            compile(cell.source, f"<cell_{i}>", "exec")
        except SyntaxError as e:
            pytest.fail(f"Code cell {i} failed to compile: {e}\nSource:\n{cell.source}")


def test_dataset_path_resolution_and_target_validation(notebook: nbformat.NotebookNode) -> None:
    """Notebook must strictly validate binary target and split locked holdout."""
    code_cells = [c.source for c in notebook.cells if c.cell_type == "code"]
    assert len(code_cells) >= 2
    cell1 = code_cells[1]

    # Path resolution and target column
    assert "find_data_directory" in cell1 or "/kaggle/input" in cell1
    assert "Will_Buy_EV" in cell1
    # Strict target validation: no nulls, exact set {'No', 'Yes'}, class counts >= 5
    assert ".isna().sum() == 0" in cell1 or "isna().any()" in cell1
    assert "{'No', 'Yes'}" in cell1 or '{"No", "Yes"}' in cell1
    assert ">= 5" in cell1

    # Locked holdout split via train_test_split before any policy/HPO
    assert "train_test_split" in cell1
    assert "test_size=0.2" in cell1 or "test_size=0.20" in cell1
    assert "random_state=2026" in cell1
    assert "stratify=" in cell1
    assert "holdout" in cell1


def test_three_feature_policies_defined(notebook: nbformat.NotebookNode) -> None:
    """Notebook must define raw, stateless, and fold-local feature policies."""
    all_code = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    assert "raw" in all_code
    assert "stateless" in all_code
    assert "fold_local" in all_code or "FoldLocal" in all_code

    assert "EV_Infrastructure_Score" in all_code
    assert "Affordability_Index" in all_code
    assert "Anxiety_Stations_Ratio" in all_code
    assert "Charging_Home_Subsidy_Combo" in all_code or "Combo" in all_code


def test_fold_local_transformer_leakage_safety(notebook: nbformat.NotebookNode) -> None:
    """Fold-local transformer must fit ONLY on train folds and fallback safely."""
    all_code = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")

    assert "FoldLocalFeatureTransformer" in all_code or "FoldLocalTransformer" in all_code

    fit_pattern = re.findall(r"\.fit\s*\(\s*([A-Za-z0-9_]+)", all_code)
    for fitted_var in fit_pattern:
        var_lower = fitted_var.lower()
        assert "val" not in var_lower and "test" not in var_lower and "holdout" not in var_lower, (
            f"Leakage violation: .fit() called on validation/test/holdout variable: {fitted_var}"
        )

    assert "City_Type" in all_code
    assert "Current_Car_Type" in all_code
    assert "Range_Anxiety_Level" in all_code
    assert "Annual_Income_USD" in all_code
    assert "Daily_Commute_km" in all_code

    assert "fillna(0" in all_code or "fillna(0.0" in all_code
    assert "mean" in all_code


def test_fold_local_transformer_behavior_with_synthetic_data(
    notebook: nbformat.NotebookNode,
) -> None:
    """Executes transformer cell with synthetic frames and tests fallback/isolation."""
    # Find the feature cell defining FoldLocalFeatureTransformer
    transformer_cell_code = None
    for cell in notebook.cells:
        if cell.cell_type == "code" and "FoldLocalFeatureTransformer" in cell.source:
            transformer_cell_code = cell.source
            break

    assert transformer_cell_code is not None, "Transformer code cell not found"

    # Execute cell in isolated namespace
    ns = {"pd": pd, "Path": Path}
    exec(transformer_cell_code, ns)
    TransformerClass = ns.get("FoldLocalFeatureTransformer")
    assert TransformerClass is not None

    train_data = pd.DataFrame({
        "City_Type": ["Metro", "Metro", "Rural"],
        "Current_Car_Type": ["Sedan", "SUV", "Sedan"],
        "Range_Anxiety_Level": ["High", "Low", "High"],
        "Annual_Income_USD": [100000.0, 120000.0, 50000.0],
        "Daily_Commute_km": [20.0, 40.0, 60.0],
    })

    test_data = pd.DataFrame({
        "City_Type": ["UnseenCity", "Metro"],
        "Current_Car_Type": ["FlyingCar", "SUV"],
        "Range_Anxiety_Level": ["Extreme", "Low"],
        "Annual_Income_USD": [90000.0, 110000.0],
        "Daily_Commute_km": [30.0, 35.0],
    })

    transformer = TransformerClass()
    transformer.fit(train_data)

    initial_city_map = dict(transformer.city_income_mean_map)
    transformed_test = transformer.transform(test_data)

    # 1. Unseen frequency must fall back to 0.0
    assert transformed_test["City_Type_Freq"].iloc[0] == 0.0
    assert transformed_test["Current_Car_Type_Freq"].iloc[0] == 0.0
    assert transformed_test["Range_Anxiety_Level_Freq"].iloc[0] == 0.0

    # 2. Unseen category income mean must fall back to train global mean
    expected_global_inc = float(train_data["Annual_Income_USD"].mean())
    assert transformed_test["City_Type_Income_Mean"].iloc[0] == expected_global_inc
    assert transformed_test["Current_Car_Type_Income_Mean"].iloc[0] == expected_global_inc

    # 3. Test transform must not mutate fitted mappings
    assert transformer.city_income_mean_map == initial_city_map
    assert "UnseenCity" not in transformer.city_income_mean_map


def test_gpu_probe_once_with_lightgbm_error(notebook: nbformat.NotebookNode) -> None:
    """LightGBM GPU probe must run once on small train sample catching LightGBMError."""
    all_code = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    tree = ast.parse(all_code)

    lightgbm_handlers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ExceptHandler)
        and isinstance(node.type, ast.Name)
        and node.type.id == "LightGBMError"
    ]
    assert len(lightgbm_handlers) == 1
    gpu_probe_tries = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Try) and lightgbm_handlers[0] in node.handlers
    ]

    assert len(gpu_probe_tries) == 1
    probe_calls = [
        node
        for node in ast.walk(gpu_probe_tries[0])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "LGBMClassifier"
    ]
    assert len(probe_calls) == 1
    probe_kwargs = {kw.arg: kw.value for kw in probe_calls[0].keywords if kw.arg is not None}
    assert ast.literal_eval(probe_kwargs["device"]) == "gpu"
    assert ast.literal_eval(probe_kwargs["max_bin"]) == 255

    def exception_type_names(node: ast.expr | None) -> set[str]:
        if node is None:
            return {"bare"}
        if isinstance(node, ast.Name):
            return {node.id}
        if isinstance(node, ast.Attribute):
            return {node.attr}
        if isinstance(node, ast.Tuple):
            return set().union(*(exception_type_names(item) for item in node.elts))
        return set()

    broad_names = {"bare", "Exception", "BaseException"}
    broad_handlers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ExceptHandler)
        and exception_type_names(node.type) & broad_names
    ]
    assert not broad_handlers

    max_bin_assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "MAX_BIN_HIGH"
            for target in node.targets
        )
    ]
    assert len(max_bin_assignments) == 1
    bound = max_bin_assignments[0].value
    assert isinstance(bound, ast.IfExp)
    assert ast.literal_eval(bound.body) == 255
    assert ast.literal_eval(bound.orelse) == 512

    max_bin_suggestions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "suggest_int"
        and node.args
        and ast.literal_eval(node.args[0]) == "max_bin"
    ]
    assert len(max_bin_suggestions) == 1
    max_bin_args = max_bin_suggestions[0].args
    assert ast.literal_eval(max_bin_args[1]) == 128
    assert isinstance(max_bin_args[2], ast.Name) and max_bin_args[2].id == "MAX_BIN_HIGH"


def test_phases_isolate_development_subset(notebook: nbformat.NotebookNode) -> None:
    """Phase 1 ablation, Phase 2 HPO, and Phase 3 repeated CV must run ONLY on dev subset."""
    all_code = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    # Verify split usage: dev_df and y_dev are split
    assert "dev_df" in all_code and "y_dev" in all_code
    # Holdout is isolated
    assert "holdout_df" in all_code and "y_holdout" in all_code


def test_development_split_retains_minimum_class_counts(
    notebook: nbformat.NotebookNode,
) -> None:
    """The post-split development subset must retain at least five rows per class."""
    all_code = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    tree = ast.parse(all_code)

    split_assignment = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "train_test_split"
    )
    count_assignment = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "dev_target_counts"
            for target in node.targets
        )
    )
    count_guard = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(child, ast.Name) and child.id == "dev_target_counts"
            for child in ast.walk(node.test)
        )
    )

    assert split_assignment.lineno < count_assignment.lineno < count_guard.lineno
    assert any(
        isinstance(node, ast.Constant)
        and node.value == "Development split must contain >= 5 rows per class: "
        for node in ast.walk(count_guard)
    )

    guard_module = ast.fix_missing_locations(
        ast.Module(body=[count_assignment, count_guard], type_ignores=[])
    )
    with pytest.raises(ValueError, match="Development split must contain >= 5 rows per class"):
        invalid_namespace = {"pd": pd, "y_dev": [0] * 5 + [1] * 4}
        exec(compile(guard_module, "<dev_class_guard>", "exec"), invalid_namespace)

    valid_namespace = {"pd": pd, "y_dev": [0] * 5 + [1] * 5}
    exec(compile(guard_module, "<dev_class_guard>", "exec"), valid_namespace)
    assert valid_namespace["dev_target_counts"].to_dict() == {0: 5, 1: 5}


def test_phase_1_ablation_spec(notebook: nbformat.NotebookNode) -> None:
    """Phase 1 must run StratifiedKFold(5, shuffle=True, random_state=42) on development data."""
    all_code = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    assert "StratifiedKFold" in all_code
    assert "random_state=42" in all_code or "random_state = 42" in all_code
    assert "roc_auc_score" in all_code
    assert "log_loss" in all_code
    assert "feature_ablation.csv" in all_code


def test_phase_2_hpo_spec(notebook: nbformat.NotebookNode) -> None:
    """Phase 2 Optuna HPO must search the 7 required parameters with 15 trials on dev data."""
    all_code = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    assert "optuna" in all_code
    assert "TPESampler" in all_code
    assert "seed=42" in all_code or "seed = 42" in all_code
    assert "n_trials=15" in all_code or "n_trials = 15" in all_code

    assert "num_leaves" in all_code
    assert "min_child_samples" in all_code
    assert "colsample_bytree" in all_code or "feature_fraction" in all_code
    assert "subsample" in all_code or "bagging_fraction" in all_code
    assert "subsample_freq" in all_code
    assert "reg_alpha" in all_code or "lambda_l1" in all_code
    assert "reg_lambda" in all_code or "lambda_l2" in all_code
    assert "max_bin" in all_code
    assert "hpo_trials.csv" in all_code


def test_phase_3_rskf_and_seeds(notebook: nbformat.NotebookNode) -> None:
    """Phase 3 must evaluate 5x3 on dev data with seeds [42, 137, 2026] and export dev OOF."""
    all_code = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    assert "42" in all_code and "137" in all_code and "2026" in all_code
    assert "oof_predictions.csv" in all_code
    assert "fold_metrics.csv" in all_code


def test_holdout_acceptance_against_baseline(notebook: nbformat.NotebookNode) -> None:
    """Acceptance requires holdout candidate vs holdout baseline gate comparison."""
    all_code = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    assert "baseline" in all_code.lower()
    assert "holdout" in all_code.lower()
    # Baseline comparison gate
    assert "0.00010" in all_code or "0.0001" in all_code
    assert "0.001" in all_code
    # Historical 0.94224 is descriptive only
    assert "0.94224" in all_code


def test_hpo_and_final_params_consistency_and_full_retrain(
    notebook: nbformat.NotebookNode,
) -> None:
    """Fixed params (n_estimators=450, lr=0.04) match, retrain on full data with 3 seeds."""
    all_code = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    # Exact fixed params across HPO and candidate
    assert "n_estimators" in all_code
    assert "450" in all_code
    assert "0.04" in all_code
    # No silent 600 / 0.035 mismatch
    assert "600" not in all_code
    assert "0.035" not in all_code

    # Seed-bagged full-data models averaged over 3 seeds
    assert "/ 3" in all_code or "len(SEEDS)" in all_code or "np.mean(" in all_code


def test_submission_contract_and_manifest_exports(notebook: nbformat.NotebookNode) -> None:
    """Notebook must perform submission contract checks and export required artifacts."""
    all_code = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")

    assert "submission.csv" in all_code
    assert "experiment_manifest.json" in all_code
    assert "sha256" in all_code or "SHA-256" in all_code or "hashlib" in all_code

    required_exports = [
        "submission.csv",
        "oof_predictions.csv",
        "feature_ablation.csv",
        "hpo_trials.csv",
        "fold_metrics.csv",
        "experiment_manifest.json",
    ]
    for exp in required_exports:
        assert exp in all_code, f"Missing export reference: {exp}"


def test_exact_locked_evaluation_and_full_retrain_protocol(
    notebook: nbformat.NotebookNode,
) -> None:
    """Critical split, device, acceptance, retrain, and manifest contracts are explicit."""
    all_code = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")

    assert all_code.count("except LightGBMError:") == 1
    assert all_code.count("device='gpu'") == 1
    assert all_code.count("'device': MODEL_DEVICE") >= 5

    assert all_code.count("split(dev_df, y_dev)") == 3
    assert "split(train_df, y_train_all)" not in all_code
    assert "candidate_holdout_auc >= baseline_holdout_auc + AUC_IMPROVEMENT_GATE" in all_code
    assert "candidate_holdout_logloss <= baseline_holdout_logloss + LOGLOSS_TOLERANCE" in all_code

    assert "for final_seed in SEEDS:" in all_code
    assert "assert len(full_data_test_predictions) == 3" in all_code
    assert "test_final = np.mean(full_data_test_predictions, axis=0)" in all_code

    required_manifest_fields = [
        "fixed_model_params",
        "tuned_model_params",
        "final_model_params",
        "model_device",
        "splits",
        "locked_holdout_metrics",
        "acceptance",
        "submission_ensemble",
        "data_hashes",
        "versions",
    ]
    for field in required_manifest_fields:
        assert f"'{field}'" in all_code

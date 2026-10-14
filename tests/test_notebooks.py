from pathlib import Path

import nbformat
import pytest

NOTEBOOKS = sorted(Path("notebooks").glob("*.ipynb"))


@pytest.mark.parametrize("path", NOTEBOOKS)
def test_every_code_cell_has_preceding_markdown(path: Path) -> None:
    notebook = nbformat.read(path, as_version=4)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            assert index > 0
            assert notebook.cells[index - 1].cell_type == "markdown"
            assert notebook.cells[index - 1].source.strip()


@pytest.mark.parametrize("path", NOTEBOOKS)
def test_notebooks_do_not_contain_agent_meta_or_home_paths(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    forbidden = ["/home/", "/Users/", ".agent/skills", "subagent", "Task ID"]
    assert not [token for token in forbidden if token in text]


def test_build_notebooks_targeted_mode() -> None:
    import sys
    sys.path.insert(0, str(Path.cwd() / "scripts"))
    from build_notebooks import NB04, _notebook
    nb = _notebook(NB04)
    assert len(nb.cells) == 36
    assert nb.cells[0].cell_type == "markdown"
    assert nb.cells[1].cell_type == "code"


def test_final_notebook_does_not_fit_or_tune() -> None:
    path = Path("notebooks/05_final_results.ipynb")
    assert path.exists(), "Notebook 05 does not exist"
    notebook = nbformat.read(path, as_version=4)
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    forbidden = [
        ".fit(",
        "train(",
        "run_cross_validation(",
        "cross_validate(",
        "build_model(",
        "study.optimize(",
        "suggest_float(",
        "suggest_int(",
    ]
    found = [token for token in forbidden if token in code]
    assert not found, f"Forbidden training/fitting token found in code cells: {found}"


def test_final_notebook_exact_14_headings() -> None:
    path = Path("notebooks/05_final_results.ipynb")
    assert path.exists(), "Notebook 05 does not exist"
    notebook = nbformat.read(path, as_version=4)
    md_text = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "markdown")
    expected_headings = [
        "## 1. Executive Summary",
        "## 2. Task and Data Contract",
        "## 3. Fixed Split and Leakage Policy",
        "## 4. Baseline Comparison",
        "## 5. Selected Pipeline and Source Run IDs",
        "## 6. 15-Fold Cross-Validation Metric Evidence",
        "## 7. Repeat and Fold Stability",
        "## 8. Error, Prediction Decile, and Feature Cohort Findings",
        "## 9. Advanced Experiments, HPO, and Blending Decision",
        "## 10. Runtime and Model Complexity Tradeoffs",
        "## 11. Model Limitations and Operational Boundary Conditions",
        "## 12. Final Submission File Validation and Checksum",
        "## 13. Reproducibility Commands and Handoff",
        "## 14. Kaggle Public Leaderboard and Submission Ledger",
    ]
    for h in expected_headings:
        assert h in md_text, f"Missing expected heading: {h}"


def test_final_notebook_every_major_pair_has_oia() -> None:
    path = Path("notebooks/05_final_results.ipynb")
    assert path.exists(), "Notebook 05 does not exist"
    notebook = nbformat.read(path, as_version=4)
    oia_cells = [
        c
        for c in notebook.cells
        if c.cell_type == "markdown" and "### Observation / Interpretation / Action" in c.source
    ]
    assert len(oia_cells) == 14, f"Expected 14 OIA markdown cells, found {len(oia_cells)}"


def test_notebook05_section14_reads_persisted_leaderboard_artifacts() -> None:
    import sys
    sys.path.insert(0, str(Path.cwd() / "scripts"))
    from build_notebooks import NB05, _notebook
    nb = _notebook(NB05)
    code = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "code")
    required_artifacts = [
        "public_leaderboard.parquet",
        "paired_comparisons.parquet",
        "model_summary.parquet",
        "leaderboard_manifest.json",
        "outputs/leaderboard",
    ]
    for artifact in required_artifacts:
        assert artifact in code, f"Missing required leaderboard artifact reference: {artifact}"


def test_notebook05_section14_distinguishes_locked_model_from_best_public() -> None:
    import sys
    sys.path.insert(0, str(Path.cwd() / "scripts"))
    from build_notebooks import NB05, _notebook
    nb = _notebook(NB05)
    md_text = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "markdown")
    required_phrases = ["Locked Production Model", "Best Public Submission"]
    for phrase in required_phrases:
        assert phrase in md_text, f"Missing required distinction phrase: {phrase}"


def test_notebook05_section14_no_superiority_overclaim() -> None:
    import sys
    sys.path.insert(0, str(Path.cwd() / "scripts"))
    from build_notebooks import NB05, _notebook
    nb = _notebook(NB05)
    sec14_md = ""
    for idx, cell in enumerate(nb.cells):
        if cell.cell_type == "markdown" and "## 14." in cell.source:
            sec14_md += cell.source + "\n"
            if idx + 2 < len(nb.cells) and nb.cells[idx + 2].cell_type == "markdown":
                sec14_md += nb.cells[idx + 2].source + "\n"

    forbidden = ["statistically significant", "proves", "definitively better", "is superior"]
    sec14_md_lower = sec14_md.lower()
    for phrase in forbidden:
        assert (
            phrase not in sec14_md_lower
        ), f"Forbidden over-claim phrase found in Section 14: '{phrase}'"

    msg = "Section 14 markdown must contain explicit uncertainty caveat 'within fold-level noise'"
    assert "within fold-level noise" in sec14_md, msg


def test_notebook05_section14_no_fitting_tokens() -> None:
    import sys
    sys.path.insert(0, str(Path.cwd() / "scripts"))
    from build_notebooks import NB05
    forbidden = [
        ".fit(",
        "train(",
        "run_cross_validation(",
        "cross_validate(",
        "build_model(",
        "study.optimize(",
        "suggest_float(",
        "suggest_int(",
    ]
    sec14_code = ""
    for md, code in NB05:
        if "## 14." in md or "leaderboard" in md.lower():
            sec14_code += code + "\n"

    found = [token for token in forbidden if token in sec14_code]
    assert not found, f"Forbidden training/fitting token found in Section 14 code: {found}"


def _nb05_section14_sources() -> tuple[str, str]:
    """Return (markdown, code) concatenated for Section 14 and its OIA cell."""
    import sys
    sys.path.insert(0, str(Path.cwd() / "scripts"))
    from build_notebooks import NB05, _notebook
    nb = _notebook(NB05)
    md_parts: list[str] = []
    code_parts: list[str] = []
    for idx, cell in enumerate(nb.cells):
        if cell.cell_type == "markdown" and "## 14." in cell.source:
            md_parts.append(cell.source)
            if idx + 1 < len(nb.cells) and nb.cells[idx + 1].cell_type == "code":
                code_parts.append(nb.cells[idx + 1].source)
            if idx + 2 < len(nb.cells) and nb.cells[idx + 2].cell_type == "markdown":
                md_parts.append(nb.cells[idx + 2].source)
    return "\n".join(md_parts), "\n".join(code_parts)


def test_notebook05_section13_includes_leaderboard_artifact_command() -> None:
    """Section 13 reproducibility list must regenerate the leaderboard artifacts."""
    import sys
    sys.path.insert(0, str(Path.cwd() / "scripts"))
    from build_notebooks import NB05, _notebook
    nb = _notebook(NB05)
    code = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "code")
    assert "scripts/build_leaderboard_artifacts.py" in code, (
        "Section 13 repro_commands must include scripts/build_leaderboard_artifacts.py"
    )


def test_notebook05_section14_verifies_leaderboard_md_staleness() -> None:
    """Section 14 must verify leaderboard.md sha256 against the persisted manifest."""
    _, code = _nb05_section14_sources()
    assert "leaderboard_md_sha256" in code, "Section 14 must read leaderboard_md_sha256"
    assert "sha256" in code and "hexdigest()" in code, (
        "Section 14 must recompute leaderboard.md sha256"
    )
    assert "assert" in code and "build_leaderboard_artifacts" in code, (
        "Section 14 must assert freshness and name the regeneration script on failure"
    )


def test_notebook05_section14_explains_two_distinct_local_metrics() -> None:
    """Section 14 must distinguish per-fold mean ROC-AUC from pooled OOF ROC-AUC."""
    md, _ = _nb05_section14_sources()
    assert "mean of per-fold ROC-AUC" in md, (
        "Section 14 must define the per-fold mean metric explicitly"
    )
    assert "repeat-averaged out-of-fold ROC-AUC" in md, (
        "Section 14 must define the repeat-averaged OOF metric explicitly"
    )
    assert "bagging effect" in md, (
        "Section 14 must explain the gap as a bagging effect, not rounding noise"
    )


def test_notebook05_section14_abs_ratio_non_significance_caveat() -> None:
    """abs_ratio must be labelled descriptive-only, never a significance test."""
    md, _ = _nb05_section14_sources()
    assert "not a significance test" in md.lower(), (
        "Section 14 must state abs_ratio is not a significance test"
    )
    assert "correlated" in md.lower(), (
        "Section 14 must note repeated-CV folds are correlated"
    )
    assert "descriptive" in md.lower(), (
        "Section 14 must label the close-call threshold as descriptive"
    )


def test_final_notebook_executed_outputs_present() -> None:
    path = Path("notebooks/05_final_results.ipynb")
    assert path.exists(), "Notebook 05 does not exist"
    notebook = nbformat.read(path, as_version=4)
    code_cells = [c for c in notebook.cells if c.cell_type == "code"]
    for idx, c in enumerate(code_cells):
        assert c.outputs, f"Code cell {idx} is missing execution outputs"


def test_final_notebook_generator_source_sync() -> None:
    import sys
    sys.path.insert(0, str(Path.cwd() / "scripts"))
    from build_notebooks import NB05, _notebook
    nb_gen = _notebook(NB05)
    path = Path("notebooks/05_final_results.ipynb")
    nb_file = nbformat.read(path, as_version=4)
    assert len(nb_gen.cells) == len(nb_file.cells)
    for idx in range(len(nb_gen.cells)):
        assert nb_gen.cells[idx].source.strip() == nb_file.cells[idx].source.strip()


def test_notebooks_do_not_contain_escaped_newline_prints() -> None:
    """Verify code cells do not contain literal print('\\\\n...') escaped strings."""
    import sys
    sys.path.insert(0, str(Path.cwd() / "scripts"))
    from build_notebooks import NB01, NB02, NB03, NB04, NB05

    nb_lists = [("NB01", NB01), ("NB02", NB02), ("NB03", NB03), ("NB04", NB04), ("NB05", NB05)]
    for nb_name, nb_list in nb_lists:
        for _md, code in nb_list:
            assert r'print("\\n' not in code, f"Literal escaped newline in print found in {nb_name}"


def test_final_notebook_section5_source_ids_and_assertions() -> None:
    path = Path("notebooks/05_final_results.ipynb")
    assert path.exists(), "Notebook 05 does not exist"
    notebook = nbformat.read(path, as_version=4)
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    required_tokens = [
        "source_run_ids",
        "selected_run_id",
        "selected_candidate_id",
        "candidate_metadata_path",
        "assert set(source_run_ids) ==",
        "assert selected_run_id == selected_candidate_id",
        "assert meta_run_id == selected_run_id",
    ]
    for token in required_tokens:
        assert token in code, f"Missing required token in Section 5 code: {token}"


def test_notebook01_contains_adversarial_validation() -> None:
    import sys
    sys.path.insert(0, str(Path.cwd() / "scripts"))
    from build_notebooks import NB01, _notebook
    nb = _notebook(NB01)
    md_text = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "markdown")
    code_text = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "code")

    # Contract headings
    assert "## Adversarial validation" in md_text
    assert "### Observation / Interpretation / Action" in md_text

    # Reusable module invocation and wiring
    assert "run_adversarial_validation(" in code_text
    assert "from ev_purchase.adversarial_validation import" in code_text
    assert "oof_domain_preds" in code_text
    assert "drop(columns=[config.data.id_column, config.data.target_column])" in code_text

    # Provenance: dataset SHA-256 computation and pass-through
    assert "compute_file_sha256" in code_text
    assert "data_sha256" in code_text

    # Current-result dynamic interpretation and action recommendation (not hardcoded)
    assert "adversarial_res.interpretation" in code_text
    assert "action" in code_text.lower()

    # Anti-leakage checks: no global preprocessing/fitting outside folds
    adversarial_code = ""
    for md, code in NB01:
        if "## Adversarial validation" in md:
            adversarial_code = code
            break
    assert "fit_transform(" not in adversarial_code, (
        "Forbidden global fit_transform in adversarial validation cell"
    )
    assert "adversarial_validation.json" in adversarial_code
    assert "adversarial_fold_metrics.csv" in adversarial_code
    assert "adversarial_feature_importance.csv" in adversarial_code

    # Generator must not contain import-time artifact reads or hardcoded fallback metrics
    build_script = (Path.cwd() / "scripts" / "build_notebooks.py").read_text(encoding="utf-8")
    assert "_load_adversarial_diagnostics" not in build_script
    assert "_adv_diag" not in build_script

    # Markdown contract: scan entire NB01 markdown and reject premature/unconditional claims
    md_text_lower = md_text.lower()
    forbidden_claims = [
        "same generating process",
        "track the leaderboard",
        "trust cross-validation",
        "trust cv",
        "0.4988",
    ]
    for claim in forbidden_claims:
        assert claim not in md_text_lower, (
            f"Forbidden premature claim found in NB01 markdown: {claim}"
        )

    # Neutral screening wording in univariate shift section
    assert "screening only" in md_text_lower
    assert "adversarial validation below determines" in md_text_lower

    # Adversarial validation markdown explains reading bands rather than claiming observed outcome
    adv_md = ""
    for idx, (md, _code) in enumerate(NB01):
        if "## Adversarial validation" in md:
            if idx + 1 < len(NB01):
                adv_md = NB01[idx + 1][0]
            break
    assert "interpretation bands" in adv_md.lower()
    assert "little detectable shift" in adv_md.lower()
    assert "material diagnostic shift" in adv_md.lower()
    assert "yields" not in adv_md.lower()


def test_notebook03_contains_required_diagnostic_sections() -> None:
    import sys
    sys.path.insert(0, str(Path.cwd() / "scripts"))
    from build_notebooks import NB03, _notebook
    nb = _notebook(NB03)
    md_text = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "markdown")

    required_topics = [
        "OOF hardest examples",
        "Label-quality diagnostics",
        "Cross-fold SHAP",
        "Cross-fold permutation importance",
        "Evidence-driven proposed experiments",
    ]
    for topic in required_topics:
        assert topic.lower() in md_text.lower(), f"Missing topic in NB03 markdown: {topic}"

    oia_count = md_text.count("### Observation / Interpretation / Action")
    assert oia_count >= 13, f"Expected at least 13 OIA sections in NB03 markdown, found {oia_count}"


def test_notebook03_generator_source_sync() -> None:
    import sys
    sys.path.insert(0, str(Path.cwd() / "scripts"))
    from build_notebooks import NB03, _notebook
    nb_gen = _notebook(NB03)
    path = Path("notebooks/03_model_analysis.ipynb")
    assert path.exists(), "Notebook 03 does not exist"
    nb_file = nbformat.read(path, as_version=4)
    assert len(nb_gen.cells) == len(nb_file.cells), (
        f"Cell count mismatch: gen {len(nb_gen.cells)} vs file {len(nb_file.cells)}"
    )
    for idx in range(len(nb_gen.cells)):
        assert nb_gen.cells[idx].source.strip() == nb_file.cells[idx].source.strip(), (
            f"Mismatch at cell {idx}"
        )


def test_notebook03_oof_keyed_aggregation_log_loss_behavioral() -> None:
    import sys
    sys.path.insert(0, str(Path.cwd() / "scripts"))
    from build_notebooks import NB03, _notebook
    nb = _notebook(NB03)
    code_text = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "code")

    required_tokens = [
        "groupby(\"row_index\")",
        "assert len(mean_oof) == len(train_df)",
        "np.isclose(recomputed[\"log_loss\"], logged[\"log_loss\"]",
        "0.226",
    ]
    for token in required_tokens:
        assert token in code_text, f"Missing OOF keyed aggregation token in NB03 code: {token}"


def test_notebook03_shap_same_model_cohort_alignment() -> None:
    import sys
    sys.path.insert(0, str(Path.cwd() / "scripts"))
    from build_notebooks import NB03, _notebook
    nb = _notebook(NB03)
    code_text = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "code")
    md_text = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "markdown")

    assert "pred_binary_lgb" in code_text, (
        "SHAP FP/FN mask must use same-model LightGBM predictions"
    )
    assert "max_abs_err" in code_text or "np.max(np.abs(" in code_text, (
        "SHAP must compute maximum absolute reconstruction error"
    )
    assert "100% exact match" not in md_text, "SHAP language must avoid '100% exact match'"


def test_notebook03_label_diagnostics_sensitivity_calibration_wording() -> None:
    import sys
    sys.path.insert(0, str(Path.cwd() / "scripts"))
    from build_notebooks import NB03, _notebook
    nb = _notebook(NB03)
    code_text = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "code")
    md_text = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "markdown")

    assert "brier_score" in code_text, (
        "Label diagnostics must include Brier score calibration metric"
    )
    assert (
        "0.6/0.4" in code_text
        or "0.6/0.4" in md_text
        or "threshold_sensitivity" in code_text
    ), "Missing threshold sensitivity table"

    forbidden_phrases = ["confirmed error", "confirmed label error", "proven error"]
    for phrase in forbidden_phrases:
        assert phrase not in md_text.lower(), f"Forbidden label error phrase in NB03: {phrase}"


def test_notebook03_permutation_reporting_counts_and_negative_drop() -> None:
    import sys
    sys.path.insert(0, str(Path.cwd() / "scripts"))
    from build_notebooks import NB03, _notebook
    nb = _notebook(NB03)
    md_text = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "markdown")
    code_text = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "code")

    assert (
        "65 permutations" in md_text
        or "13 features x 5 folds = 65" in code_text
        or "65 total permutations" in code_text
    ), "Missing exact permutation count (13 features x 5 folds = 65)"
    assert "negative" in md_text.lower(), "Missing explanation of negative permutation drops"


def test_notebook03_proposed_experiments_anti_leakage_constraints() -> None:
    import sys
    sys.path.insert(0, str(Path.cwd() / "scripts"))
    from build_notebooks import NB03, _notebook
    nb = _notebook(NB03)
    code_text = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "code")
    md_text = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "markdown")

    assert (
        "outer validation" in md_text.lower()
        or "outer validation" in code_text.lower()
    ), "Missing explicit anti-leakage outer validation statement"
    assert "sample weights set to 0.5" not in code_text, (
        "Proposed experiments must not use row-keyed OOF sample reweighting"
    )


def test_notebook03_artifact_schemas_and_ranges() -> None:
    import numpy as np
    import pandas as pd

    hardest_path = Path("outputs/diagnostics/oof_hardest_examples.csv")
    label_path = Path("outputs/diagnostics/label_quality_candidates.csv")
    shap_path = Path("outputs/diagnostics/shap_feature_importance.csv")
    perm_path = Path("outputs/diagnostics/permutation_importance.csv")
    prop_path = Path("outputs/diagnostics/proposed_experiments.csv")

    for p in [hardest_path, label_path, shap_path, perm_path, prop_path]:
        assert p.exists(), f"Missing diagnostic artifact: {p}"
        df = pd.read_csv(p)
        assert len(df) > 0, f"Diagnostic artifact is empty: {p}"

    df_hard = pd.read_csv(hardest_path)
    assert "id" in df_hard.columns
    assert "y_true" in df_hard.columns
    assert "p_mean" in df_hard.columns
    assert "log_loss" in df_hard.columns
    assert np.isfinite(df_hard["p_mean"]).all()
    assert (df_hard["p_mean"] >= 0.0).all() and (df_hard["p_mean"] <= 1.0).all()
    assert (df_hard["log_loss"] >= 0.0).all()

    df_prop = pd.read_csv(prop_path)
    text_prop = " ".join(df_prop.to_string().split()).lower()
    assert "outer validation" in text_prop or "nested" in text_prop, (
        "Proposed experiments artifact must document outer/nested validation constraints"
    )


def test_ft_transformer_notebook_compiles_and_validates_schema() -> None:
    """Verify FT-Transformer notebook schema and valid Python syntax in all code cells."""
    path = Path("notebooks/kaggle_ev_purchase_ft_transformer.ipynb")
    assert path.exists(), f"FT-Transformer notebook does not exist at {path}"
    notebook = nbformat.read(path, as_version=4)

    nbformat.validate(notebook)

    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert code_cells, "No code cells found in FT-Transformer notebook"

    compile_errors: list[str] = []
    for idx, cell in enumerate(code_cells):
        try:
            compile(cell.source, f"<ft_transformer_code_cell_{idx}>", "exec")
        except SyntaxError as exc:
            compile_errors.append(
                f"code_cell_{idx}:line_{exc.lineno}: {exc.msg} -> "
                f"{exc.text.strip() if exc.text else ''}"
            )

    assert not compile_errors, (
        f"Found {len(compile_errors)} code cell compilation error(s):\n"
        + "\n".join(compile_errors)
    )

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
    assert len(nb.cells) == 33
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


def test_final_notebook_exact_13_headings() -> None:
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
    assert len(oia_cells) == 13, f"Expected 13 OIA markdown cells, found {len(oia_cells)}"


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

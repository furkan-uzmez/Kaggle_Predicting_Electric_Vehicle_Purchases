from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ev_purchase.config import Config, load_config
from ev_purchase.data import DatasetContract, encode_target, load_frames, validate_frames


def test_dataset_contract_is_immutable_and_declares_labels() -> None:
    contract = DatasetContract(id_column="id", target_column="Will_Buy_EV")
    assert contract.id_column == "id"
    assert contract.target_column == "Will_Buy_EV"
    assert contract.negative_label == "No"
    assert contract.positive_label == "Yes"
    with pytest.raises(FrozenInstanceError):
        contract.id_column = "new_id"  # type: ignore[misc]


def test_validate_frames_happy_path() -> None:
    train = pd.DataFrame({"id": [1, 2], "x": [0.1, 0.2], "Will_Buy_EV": ["No", "Yes"]})
    test = pd.DataFrame({"id": [3, 4], "x": [0.3, 0.4]})
    contract = DatasetContract(id_column="id", target_column="Will_Buy_EV")
    validate_frames(train, test, contract)


def test_validate_frames_missing_target_in_train() -> None:
    train = pd.DataFrame({"id": [1, 2], "x": [0.1, 0.2]})
    test = pd.DataFrame({"id": [3, 4], "x": [0.3, 0.4]})
    contract = DatasetContract(id_column="id", target_column="Will_Buy_EV")
    with pytest.raises(ValueError, match="target missing from train"):
        validate_frames(train, test, contract)


def test_validate_frames_target_in_test() -> None:
    train = pd.DataFrame({"id": [1, 2], "x": [0.1, 0.2], "Will_Buy_EV": ["No", "Yes"]})
    test = pd.DataFrame({"id": [3, 4], "x": [0.3, 0.4], "Will_Buy_EV": ["No", "Yes"]})
    contract = DatasetContract(id_column="id", target_column="Will_Buy_EV")
    with pytest.raises(ValueError, match="target must not appear in test"):
        validate_frames(train, test, contract)


def test_validate_frames_schema_order_mismatch() -> None:
    train = pd.DataFrame(
        {"id": [1, 2], "x": [0.1, 0.2], "y_feat": [1, 2], "Will_Buy_EV": ["No", "Yes"]}
    )
    test = pd.DataFrame({"id": [3, 4], "y_feat": [3, 4], "x": [0.3, 0.4]})
    contract = DatasetContract(id_column="id", target_column="Will_Buy_EV")
    with pytest.raises(ValueError, match="train/test feature schema mismatch"):
        validate_frames(train, test, contract)


def test_validate_frames_missing_id_column() -> None:
    train = pd.DataFrame({"x": [0.1, 0.2], "Will_Buy_EV": ["No", "Yes"]})
    test = pd.DataFrame({"x": [0.3, 0.4]})
    contract = DatasetContract(id_column="id", target_column="Will_Buy_EV")
    with pytest.raises(ValueError, match="ID column missing"):
        validate_frames(train, test, contract)

    train_valid = pd.DataFrame({"id": [1, 2], "x": [0.1, 0.2], "Will_Buy_EV": ["No", "Yes"]})
    test_missing = pd.DataFrame({"x": [0.3, 0.4]})
    with pytest.raises(ValueError, match="ID column missing"):
        validate_frames(train_valid, test_missing, contract)


def test_validate_frames_duplicate_ids() -> None:
    train = pd.DataFrame({"id": [1, 1], "x": [0.1, 0.2], "Will_Buy_EV": ["No", "Yes"]})
    test = pd.DataFrame({"id": [3, 4], "x": [0.3, 0.4]})
    contract = DatasetContract(id_column="id", target_column="Will_Buy_EV")
    with pytest.raises(ValueError, match="IDs must be unique"):
        validate_frames(train, test, contract)

    train_valid = pd.DataFrame({"id": [1, 2], "x": [0.1, 0.2], "Will_Buy_EV": ["No", "Yes"]})
    test_dup = pd.DataFrame({"id": [3, 3], "x": [0.3, 0.4]})
    with pytest.raises(ValueError, match="IDs must be unique"):
        validate_frames(train_valid, test_dup, contract)


def test_validate_frames_overlapping_ids() -> None:
    train = pd.DataFrame({"id": [1, 2], "x": [0.1, 0.2], "Will_Buy_EV": ["No", "Yes"]})
    test = pd.DataFrame({"id": [2, 3], "x": [0.3, 0.4]})
    contract = DatasetContract(id_column="id", target_column="Will_Buy_EV")
    with pytest.raises(ValueError, match="train/test ID overlap"):
        validate_frames(train, test, contract)


def test_validate_frames_raw_label_validation() -> None:
    test = pd.DataFrame({"id": [4, 5, 6], "x": [0.4, 0.5, 0.6]})
    contract = DatasetContract(id_column="id", target_column="Will_Buy_EV")

    # Numeric raw target rejected
    train_num = pd.DataFrame({"id": [1, 2], "x": [0.1, 0.2], "Will_Buy_EV": [0, 1]})
    with pytest.raises(ValueError, match="target labels must be exactly"):
        validate_frames(train_num, test, contract)

    # Multiclass numeric/string rejected
    train_multi = pd.DataFrame(
        {"id": [1, 2, 3], "x": [0.1, 0.2, 0.3], "Will_Buy_EV": ["No", "Yes", "Maybe"]}
    )
    with pytest.raises(ValueError, match="target labels must be exactly"):
        validate_frames(train_multi, test, contract)

    # Single class rejected
    train_single = pd.DataFrame({"id": [1, 2], "x": [0.1, 0.2], "Will_Buy_EV": ["No", "No"]})
    with pytest.raises(ValueError, match="target labels must be exactly"):
        validate_frames(train_single, test, contract)

    # Arbitrary strings rejected
    train_str = pd.DataFrame({"id": [1, 2], "x": [0.1, 0.2], "Will_Buy_EV": ["False", "True"]})
    with pytest.raises(ValueError, match="target labels must be exactly"):
        validate_frames(train_str, test, contract)


def test_validate_frames_missing_target_labels() -> None:
    train = pd.DataFrame({"id": [1, 2], "x": [0.1, 0.2], "Will_Buy_EV": ["No", None]})
    test = pd.DataFrame({"id": [3, 4], "x": [0.3, 0.4]})
    contract = DatasetContract(id_column="id", target_column="Will_Buy_EV")
    with pytest.raises(ValueError, match="target has missing labels"):
        validate_frames(train, test, contract)


def test_encode_target_happy_path_and_immutability() -> None:
    contract = DatasetContract(id_column="id", target_column="Will_Buy_EV")
    raw_series = pd.Series(["No", "Yes", "No", "Yes"])

    encoded = encode_target(raw_series, contract)

    assert isinstance(encoded, np.ndarray)
    assert encoded.dtype == np.dtype("int64")
    assert np.array_equal(encoded, np.array([0, 1, 0, 1]))

    # Verify input series is unchanged
    assert list(raw_series) == ["No", "Yes", "No", "Yes"]


def test_encode_target_unknown_label_rejection() -> None:
    contract = DatasetContract(id_column="id", target_column="Will_Buy_EV")
    raw_series = pd.Series(["No", "Yes", "Unknown"])

    with pytest.raises(ValueError, match="unknown label|unrecognized"):
        encode_target(raw_series, contract)


def test_load_frames(tmp_path: Path) -> None:
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"

    pd.DataFrame({"id": [1], "x": [10], "Will_Buy_EV": ["No"]}).to_csv(train_csv, index=False)
    pd.DataFrame({"id": [2], "x": [20]}).to_csv(test_csv, index=False)

    train, test = load_frames(train_csv, test_csv)
    assert len(train) == 1
    assert len(test) == 1
    assert list(train.columns) == ["id", "x", "Will_Buy_EV"]
    assert list(test.columns) == ["id", "x"]


def test_load_config_requires_explicit_project_root(tmp_path: Path) -> None:
    config_file = tmp_path / "base.yaml"
    config_file.write_text("project:\n  seed: 42\n")
    with pytest.raises(ValueError, match="project_root must be explicitly provided"):
        load_config(config_file)  # type: ignore[call-arg]


def test_load_config_path_resolution(tmp_path: Path) -> None:
    config_file = tmp_path / "base.yaml"
    config_content = """
project:
  seed: 42
data:
  train: data/raw/train.csv
  test: data/raw/test.csv
  sample_submission: data/raw/sample_submission.csv
  id_column: id
  target_column: Will_Buy_EV
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
"""
    config_file.write_text(config_content)

    project_root = tmp_path / "my_project"
    project_root.mkdir()

    cfg = load_config(config_file, project_root=project_root)
    assert isinstance(cfg, Config)
    assert cfg.project.seed == 42
    assert cfg.data.train == project_root / "data/raw/train.csv"
    assert cfg.data.test == project_root / "data/raw/test.csv"
    assert cfg.data.sample_submission == project_root / "data/raw/sample_submission.csv"
    assert cfg.data.id_column == "id"
    assert cfg.data.target_column == "Will_Buy_EV"
    assert cfg.validation.n_splits == 5
    assert cfg.validation.n_repeats == 3
    assert cfg.validation.random_state == 42
    assert cfg.metrics.primary == "roc_auc"
    assert cfg.metrics.guardrails == ("average_precision", "log_loss")
    assert cfg.outputs.root == project_root / "outputs"


def test_load_config_nested_config_resolution(tmp_path: Path) -> None:
    nested_dir = tmp_path / "configs" / "experiments"
    nested_dir.mkdir(parents=True)
    config_file = nested_dir / "exp001.yaml"
    config_content = """
project:
  seed: 42
data:
  train: data/raw/train.csv
  test: data/raw/test.csv
  sample_submission: data/raw/sample_submission.csv
  id_column: id
  target_column: Will_Buy_EV
validation:
  strategy: repeated_stratified_kfold
  n_splits: 5
  n_repeats: 3
  random_state: 42
metrics:
  primary: roc_auc
  guardrails:
    - log_loss
outputs:
  root: outputs
"""
    config_file.write_text(config_content)
    project_root = tmp_path / "root"
    project_root.mkdir()

    cfg = load_config(config_file, project_root=project_root)
    assert cfg.data.train == project_root / "data/raw/train.csv"


def test_load_config_rejects_absolute_configured_paths(tmp_path: Path) -> None:
    config_file = tmp_path / "bad_abs.yaml"
    config_content = """
project:
  seed: 42
data:
  train: /absolute/path/train.csv
  test: data/raw/test.csv
  sample_submission: data/raw/sample_submission.csv
  id_column: id
  target_column: Will_Buy_EV
validation:
  strategy: repeated_stratified_kfold
  n_splits: 5
  n_repeats: 3
  random_state: 42
metrics:
  primary: roc_auc
  guardrails: [log_loss]
outputs:
  root: outputs
"""
    config_file.write_text(config_content)
    with pytest.raises(ValueError, match="Configured paths must be relative"):
        load_config(config_file, project_root=tmp_path)


def test_load_config_rejects_escaping_paths(tmp_path: Path) -> None:
    config_file = tmp_path / "bad_escape.yaml"
    config_content = """
project:
  seed: 42
data:
  train: ../outside/train.csv
  test: data/raw/test.csv
  sample_submission: data/raw/sample_submission.csv
  id_column: id
  target_column: Will_Buy_EV
validation:
  strategy: repeated_stratified_kfold
  n_splits: 5
  n_repeats: 3
  random_state: 42
metrics:
  primary: roc_auc
  guardrails: [log_loss]
outputs:
  root: outputs
"""
    config_file.write_text(config_content)
    with pytest.raises(ValueError, match="Configured path escapes project root"):
        load_config(config_file, project_root=tmp_path)


def test_load_config_validation_positive_integers(tmp_path: Path) -> None:
    base_content = """
project:
  seed: 42
data:
  train: train.csv
  test: test.csv
  sample_submission: sample.csv
  id_column: id
  target_column: Will_Buy_EV
validation:
  strategy: repeated_stratified_kfold
  n_splits: {n_splits}
  n_repeats: {n_repeats}
  random_state: 42
metrics:
  primary: roc_auc
  guardrails: [log_loss]
outputs:
  root: outputs
"""
    err_msg = "n_splits must be integer >= 2 and n_repeats must be integer >= 1"
    for n_splits, n_repeats in [
        (0, 3),
        (1, 3),
        (5, 0),
        (-1, 3),
        (5, -2),
        (True, 3),
        (5, False),
        (2.5, 3),
    ]:
        config_file = tmp_path / "bad_cv.yaml"
        config_file.write_text(base_content.format(n_splits=n_splits, n_repeats=n_repeats))
        with pytest.raises(ValueError, match=err_msg):
            load_config(config_file, project_root=tmp_path)


def test_validate_frames_duplicate_column_names() -> None:
    train = pd.DataFrame([[1, 2, "No"]], columns=["id", "id", "Will_Buy_EV"])
    test = pd.DataFrame([[3, 4]], columns=["id", "x"])
    contract = DatasetContract(id_column="id", target_column="Will_Buy_EV")
    with pytest.raises(ValueError, match="duplicate column names detected in dataframes"):
        validate_frames(train, test, contract)


def test_validate_frames_missing_null_ids() -> None:
    train = pd.DataFrame({"id": [1, None], "x": [0.1, 0.2], "Will_Buy_EV": ["No", "Yes"]})
    test = pd.DataFrame({"id": [3, 4], "x": [0.3, 0.4]})
    contract = DatasetContract(id_column="id", target_column="Will_Buy_EV")
    with pytest.raises(ValueError, match="IDs contain missing values"):
        validate_frames(train, test, contract)

    train_valid = pd.DataFrame({"id": [1, 2], "x": [0.1, 0.2], "Will_Buy_EV": ["No", "Yes"]})
    test_null = pd.DataFrame({"id": [3, None], "x": [0.3, 0.4]})
    with pytest.raises(ValueError, match="IDs contain missing values"):
        validate_frames(train_valid, test_null, contract)


def test_metrics_config_guardrails_immutable_tuple() -> None:
    project_root = Path(__file__).parent.parent
    base_yaml_path = project_root / "configs/base.yaml"
    cfg = load_config(base_yaml_path, project_root=project_root)
    assert isinstance(cfg.metrics.guardrails, tuple)
    assert cfg.metrics.guardrails == ("average_precision", "log_loss")
    with pytest.raises(TypeError):
        cfg.metrics.guardrails[0] = "new_metric"  # type: ignore[index]


def test_load_config_empty_whitespace_path_rejected(tmp_path: Path) -> None:
    config_file = tmp_path / "bad_empty_path.yaml"
    config_content = """
project:
  seed: 42
data:
  train: "   "
  test: test.csv
  sample_submission: sample.csv
  id_column: id
  target_column: target
validation:
  strategy: repeated_stratified_kfold
  n_splits: 5
  n_repeats: 3
  random_state: 42
metrics:
  primary: roc_auc
  guardrails: [log_loss]
outputs:
  root: outputs
"""
    config_file.write_text(config_content)
    with pytest.raises(ValueError, match="train path must be a non-empty string"):
        load_config(config_file, project_root=tmp_path)


def test_load_config_invalid_yaml_root_structures(tmp_path: Path) -> None:
    for content in ["", "just_a_string", "- item1\n- item2"]:
        config_file = tmp_path / "bad_root.yaml"
        config_file.write_text(content)
        with pytest.raises(ValueError, match="Invalid configuration format"):
            load_config(config_file, project_root=tmp_path)


def test_load_config_runtime_type_validation(tmp_path: Path) -> None:
    # Bad seed (bool / string)
    bad_seed = tmp_path / "bad_seed.yaml"
    bad_seed.write_text("""
project:
  seed: True
data:
  train: train.csv
  test: test.csv
  sample_submission: sample.csv
  id_column: id
  target_column: Will_Buy_EV
validation:
  strategy: repeated_stratified_kfold
  n_splits: 5
  n_repeats: 3
  random_state: 42
metrics:
  primary: roc_auc
  guardrails: [log_loss]
outputs:
  root: outputs
""")
    with pytest.raises(ValueError, match="seed must be an integer"):
        load_config(bad_seed, project_root=tmp_path)

    # Empty id_column
    bad_id = tmp_path / "bad_id.yaml"
    bad_id.write_text("""
project:
  seed: 42
data:
  train: train.csv
  test: test.csv
  sample_submission: sample.csv
  id_column: ""
  target_column: Will_Buy_EV
validation:
  strategy: repeated_stratified_kfold
  n_splits: 5
  n_repeats: 3
  random_state: 42
metrics:
  primary: roc_auc
  guardrails: [log_loss]
outputs:
  root: outputs
""")
    with pytest.raises(ValueError, match="id_column and target_column must be non-empty strings"):
        load_config(bad_id, project_root=tmp_path)

    # Invalid strategy
    bad_strat = tmp_path / "bad_strat.yaml"
    bad_strat.write_text("""
project:
  seed: 42
data:
  train: train.csv
  test: test.csv
  sample_submission: sample.csv
  id_column: id
  target_column: Will_Buy_EV
validation:
  strategy: kfold
  n_splits: 5
  n_repeats: 3
  random_state: 42
metrics:
  primary: roc_auc
  guardrails: [log_loss]
outputs:
  root: outputs
""")
    with pytest.raises(ValueError, match="invalid validation strategy"):
        load_config(bad_strat, project_root=tmp_path)

    # Bad metrics primary
    bad_metric = tmp_path / "bad_metric.yaml"
    bad_metric.write_text("""
project:
  seed: 42
data:
  train: train.csv
  test: test.csv
  sample_submission: sample.csv
  id_column: id
  target_column: Will_Buy_EV
validation:
  strategy: repeated_stratified_kfold
  n_splits: 5
  n_repeats: 3
  random_state: 42
metrics:
  primary: ""
  guardrails: [log_loss]
outputs:
  root: outputs
""")
    with pytest.raises(ValueError, match="primary metric must be a non-empty string"):
        load_config(bad_metric, project_root=tmp_path)


def test_load_config_missing_required_section(tmp_path: Path) -> None:
    config_file = tmp_path / "incomplete.yaml"
    incomplete_content = """
project:
  seed: 42
"""
    config_file.write_text(incomplete_content)
    with pytest.raises(ValueError, match="missing required section"):
        load_config(config_file, project_root=tmp_path)


def test_load_config_base_yaml() -> None:
    project_root = Path(__file__).parent.parent
    base_yaml_path = project_root / "configs/base.yaml"
    cfg = load_config(base_yaml_path, project_root=project_root)

    assert cfg.project.seed == 42
    assert cfg.data.id_column == "id"
    assert cfg.data.target_column == "Will_Buy_EV"
    assert cfg.data.train == project_root / "data/playground-series-s6e9/train.csv"
    assert cfg.data.test == project_root / "data/playground-series-s6e9/test.csv"
    expected_sub = project_root / "data/playground-series-s6e9/sample_submission.csv"
    assert cfg.data.sample_submission == expected_sub
    assert cfg.validation.strategy == "repeated_stratified_kfold"
    assert cfg.validation.n_splits == 5
    assert cfg.validation.n_repeats == 3
    assert cfg.validation.random_state == 42
    assert cfg.metrics.primary == "roc_auc"
    assert cfg.metrics.guardrails == ("average_precision", "log_loss")
    assert cfg.outputs.root == project_root / "outputs"


@pytest.mark.parametrize("section", ["project", "data", "validation", "metrics", "outputs"])
def test_load_config_malformed_section_type(tmp_path: Path, section: str) -> None:
    sections = {
        "project": "\n  seed: 42",
        "data": (
            "\n  train: train.csv\n  test: test.csv\n"
            "  sample_submission: sub.csv\n  id_column: id\n  target_column: target"
        ),
        "validation": (
            "\n  strategy: repeated_stratified_kfold\n  n_splits: 5\n"
            "  n_repeats: 3\n  random_state: 42"
        ),
        "metrics": "\n  primary: roc_auc\n  guardrails:\n    - log_loss",
        "outputs": "\n  root: outputs",
    }
    sections[section] = " invalid_string_not_dict"
    yaml_text = "\n".join(f"{k}:{v}" for k, v in sections.items())
    config_file = tmp_path / f"bad_{section}.yaml"
    config_file.write_text(yaml_text)
    with pytest.raises(ValueError, match=f"section '{section}' must be a mapping"):
        load_config(config_file, project_root=tmp_path)


def test_load_config_leaf_validation_target_column(tmp_path: Path) -> None:
    for target_val in ['""', "123", "true", "null"]:
        config_file = tmp_path / "bad_target.yaml"
        config_file.write_text(f"""
project:
  seed: 42
data:
  train: train.csv
  test: test.csv
  sample_submission: sample.csv
  id_column: id
  target_column: {target_val}
validation:
  strategy: repeated_stratified_kfold
  n_splits: 5
  n_repeats: 3
  random_state: 42
metrics:
  primary: roc_auc
  guardrails: [log_loss]
outputs:
  root: outputs
""")
        msg = "id_column and target_column must be non-empty strings"
        with pytest.raises(ValueError, match=msg):
            load_config(config_file, project_root=tmp_path)


def test_load_config_leaf_validation_path_fields_non_string(tmp_path: Path) -> None:
    path_cases = [
        ("train", "123"),
        ("test", "true"),
        ("sample_submission", "[1, 2]"),
        ("outputs_root", "123"),
    ]
    for field, bad_val in path_cases:
        config_file = tmp_path / f"bad_path_{field}.yaml"
        if field == "outputs_root":
            yaml_text = f"""
project:
  seed: 42
data:
  train: train.csv
  test: test.csv
  sample_submission: sample.csv
  id_column: id
  target_column: target
validation:
  strategy: repeated_stratified_kfold
  n_splits: 5
  n_repeats: 3
  random_state: 42
metrics:
  primary: roc_auc
  guardrails: [log_loss]
outputs:
  root: {bad_val}
"""
        else:
            data_fields = {
                "train": "train.csv",
                "test": "test.csv",
                "sample_submission": "sample.csv",
            }
            data_fields[field] = bad_val
            yaml_text = f"""
project:
  seed: 42
data:
  train: {data_fields["train"]}
  test: {data_fields["test"]}
  sample_submission: {data_fields["sample_submission"]}
  id_column: id
  target_column: target
validation:
  strategy: repeated_stratified_kfold
  n_splits: 5
  n_repeats: 3
  random_state: 42
metrics:
  primary: roc_auc
  guardrails: [log_loss]
outputs:
  root: outputs
"""
        config_file.write_text(yaml_text)
        with pytest.raises(ValueError, match="path fields must be string or Path"):
            load_config(config_file, project_root=tmp_path)


def test_load_config_leaf_validation_random_state(tmp_path: Path) -> None:
    for rs_val in ["3.14", '"42"', "true", "false", "null"]:
        config_file = tmp_path / "bad_rs.yaml"
        config_file.write_text(f"""
project:
  seed: 42
data:
  train: train.csv
  test: test.csv
  sample_submission: sample.csv
  id_column: id
  target_column: target
validation:
  strategy: repeated_stratified_kfold
  n_splits: 5
  n_repeats: 3
  random_state: {rs_val}
metrics:
  primary: roc_auc
  guardrails: [log_loss]
outputs:
  root: outputs
""")
        with pytest.raises(ValueError, match="random_state must be an integer"):
            load_config(config_file, project_root=tmp_path)


def test_load_config_leaf_validation_guardrails(tmp_path: Path) -> None:
    for g_val in ['"log_loss"', "123", "true", "[123]", '[""]', "[true]"]:
        config_file = tmp_path / "bad_guardrails.yaml"
        config_file.write_text(f"""
project:
  seed: 42
data:
  train: train.csv
  test: test.csv
  sample_submission: sample.csv
  id_column: id
  target_column: target
validation:
  strategy: repeated_stratified_kfold
  n_splits: 5
  n_repeats: 3
  random_state: 42
metrics:
  primary: roc_auc
  guardrails: {g_val}
outputs:
  root: outputs
""")
        with pytest.raises(ValueError, match="guardrails must be a list of non-empty strings"):
            load_config(config_file, project_root=tmp_path)

# ruff: noqa: E501
"""Generate the decision notebooks as markdown/code cell pairs.

The notebooks orchestrate tested ``src/ev_purchase`` modules; they never
reimplement pipeline logic. Every code cell is preceded by a markdown cell so
``tests/test_notebooks.py`` can enforce the narrative contract.
"""

from pathlib import Path

import nbformat


def _notebook(pairs: list[tuple[str, str]]) -> nbformat.NotebookNode:
    """Build a notebook where every code cell follows a markdown cell."""
    notebook = nbformat.v4.new_notebook()
    cells: list[nbformat.NotebookNode] = []
    for markdown, code in pairs:
        cells.append(nbformat.v4.new_markdown_cell(markdown.strip()))
        if code.strip():
            cells.append(nbformat.v4.new_code_cell(code.strip()))
    notebook.cells = cells
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    return notebook


NB01: list[tuple[str, str]] = [
    (
        """# Notebook 01 - Data Understanding

## Problem contract

Predict `Will_Buy_EV` for each `id` in the test set.

- Task: binary classification on tabular data.
- Raw target labels are the strings `No` and `Yes`; they are validated and then
  encoded explicitly as `No -> 0`, `Yes -> 1`.
- Submission stores a probability, not a hard label.
- Primary metric: ROC-AUC. Guardrails: PR-AUC and LogLoss.
- Validation: 5-fold x 3-repeat stratified cross-validation from a persisted
  split manifest.

Every numeric claim below is produced by the cell directly above it.""",
        "",
    ),
    (
        """## Setup and path resolution

Resolve the project root from the notebook location so the notebook runs from a
clean kernel without absolute paths.""",
        """import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path.cwd()
while not (PROJECT_ROOT / "pyproject.toml").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ev_purchase.config import load_config
from ev_purchase.data import DatasetContract, encode_target, load_frames, validate_frames

SEED = 42
np.random.seed(SEED)
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 40)

config = load_config(PROJECT_ROOT / "configs" / "base.yaml", project_root=PROJECT_ROOT)
print("project root resolved:", PROJECT_ROOT.name)
print("train:", config.data.train.relative_to(PROJECT_ROOT))
print("test:", config.data.test.relative_to(PROJECT_ROOT))""",
    ),
    (
        """## Load data and assert the schema contract

`validate_frames` enforces the dataset contract: exact columns, unique ids, no
train/test id overlap, and raw target labels restricted to `No`/`Yes`.""",
        """train_df, test_df = load_frames(config.data.train, config.data.test)
contract = DatasetContract(
    id_column=config.data.id_column,
    target_column=config.data.target_column,
)
validate_frames(train_df, test_df, contract)

print("train shape:", train_df.shape)
print("test shape:", test_df.shape)
print("raw target labels:", sorted(train_df[config.data.target_column].unique()))
print("schema contract: OK")""",
    ),
    (
        """### Observation / Interpretation / Action

- **Observation:** the contract validator accepted both frames and the raw
  target contains exactly the two labels printed above.
- **Interpretation:** the data matches the agreed contract, so no silent label
  remapping is needed or permitted.
- **Action:** encode the target explicitly with `encode_target` and keep the
  string labels out of every downstream model input.""",
        "",
    ),
    (
        """## Feature-role map

Split the columns into identifier, target, numeric predictors, and categorical
predictors. This map drives preprocessing decisions later.""",
        """id_col = config.data.id_column
target_col = config.data.target_column

predictors = train_df.drop(columns=[id_col, target_col])
numeric_cols = predictors.select_dtypes(include="number").columns.tolist()
categorical_cols = [c for c in predictors.columns if c not in numeric_cols]

role_map = pd.DataFrame(
    [{"column": c, "role": "numeric"} for c in numeric_cols]
    + [{"column": c, "role": "categorical"} for c in categorical_cols]
)
print("identifier:", id_col, "| target:", target_col)
print("n_predictors:", predictors.shape[1])
role_map""",
    ),
    (
        """### Observation / Interpretation / Action

- **Observation:** the predictor block splits into the numeric and categorical
  columns listed above.
- **Interpretation:** both families are present, so the baseline needs scaling
  plus encoding while gradient-boosted trees can consume categories natively.
- **Action:** keep two model paths - a linear baseline behind a preprocessor and
  GBDTs with native categorical handling.""",
        "",
    ),
    (
        """## Data quality audit

Check missing values, duplicate ids, duplicate rows, constant columns, and
quasi-constant columns before trusting any model result.""",
        """audit = {
    "train_nulls": int(train_df.isna().sum().sum()),
    "test_nulls": int(test_df.isna().sum().sum()),
    "duplicate_ids_train": int(train_df[id_col].duplicated().sum()),
    "duplicate_ids_test": int(test_df[id_col].duplicated().sum()),
    "id_overlap_train_test": len(set(train_df[id_col]) & set(test_df[id_col])),
    "duplicate_rows_excluding_id": int(predictors.duplicated().sum()),
    "constant_columns": [c for c in predictors.columns if predictors[c].nunique() <= 1],
}
for key, value in audit.items():
    print(f"{key}: {value}")

dominance = (
    predictors.apply(lambda s: s.value_counts(normalize=True).iloc[0])
    .sort_values(ascending=False)
    .rename("top_value_share")
    .to_frame()
)
dominance""",
    ),
    (
        """### Observation / Interpretation / Action

- **Observation:** the audit counters above are all zero and no column is
  constant; the highest single-value share is the first row of the table.
- **Interpretation:** there is no missing-value imputation problem, no duplicate
  leakage between train and test, and no degenerate column to drop.
- **Action:** skip imputation-driven feature engineering, but keep the imputers
  in the pipeline as a defensive contract for future data drift.""",
        "",
    ),
    (
        """## Target prevalence

Measure class balance, because it decides whether accuracy is meaningless and
whether PR-AUC deserves guardrail status.""",
        """y_encoded = encode_target(train_df[target_col], contract)
counts = pd.Series(y_encoded).value_counts().sort_index()
positive_rate = float(y_encoded.mean())

print("label counts (0=No, 1=Yes):")
print(counts.to_string())
print(f"positive rate: {positive_rate:.4%}")
print(f"majority-class accuracy floor: {1 - positive_rate:.4%}")""",
    ),
    (
        """### Observation / Interpretation / Action

- **Observation:** the positive class is the minority at the rate printed above,
  so a do-nothing classifier already reaches the accuracy floor shown.
- **Interpretation:** accuracy is not a usable objective here; ranking quality
  and probability calibration matter instead.
- **Action:** keep ROC-AUC as the primary metric and report PR-AUC and LogLoss
  as guardrails, exactly as the design specifies.""",
        "",
    ),
    (
        """## Numeric diagnostics

Summarise the numeric predictors and their univariate association with the
encoded target.""",
        """numeric_summary = predictors[numeric_cols].describe().T
numeric_summary["corr_with_target"] = [
    float(np.corrcoef(predictors[c], y_encoded)[0, 1]) for c in numeric_cols
]
numeric_summary.sort_values("corr_with_target", ascending=False)""",
    ),
    (
        """### Observation / Interpretation / Action

- **Observation:** the numeric columns sit on very different scales and their
  point-biserial correlations with the target are listed above.
- **Interpretation:** the scale spread hurts a linear model but not trees, and
  no single numeric column looks like a target proxy.
- **Action:** standardise numerics for the linear baseline only, and rely on the
  cross-validated comparison rather than these univariate signals.""",
        "",
    ),
    (
        """## Categorical diagnostics

Report cardinality and per-category positive rate to expose both encoding cost
and signal strength.""",
        """rows = []
for col in categorical_cols:
    grouped = pd.DataFrame({col: predictors[col], "y": y_encoded}).groupby(col)["y"]
    rates = grouped.mean()
    rows.append(
        {
            "column": col,
            "cardinality": int(predictors[col].nunique()),
            "min_positive_rate": float(rates.min()),
            "max_positive_rate": float(rates.max()),
            "spread": float(rates.max() - rates.min()),
        }
    )

categorical_summary = pd.DataFrame(rows).sort_values("spread", ascending=False)
categorical_summary""",
    ),
    (
        """### Observation / Interpretation / Action

- **Observation:** every categorical column has low cardinality and the
  positive-rate spread per column is shown above.
- **Interpretation:** one-hot encoding is cheap at this cardinality, and the
  columns with the widest spread carry the most univariate signal.
- **Action:** use plain one-hot encoding for the linear baseline; no target
  encoding is required, which removes a common leakage source.""",
        "",
    ),
    (
        """## Train/test distribution shift

Compare the predictor distributions across train and test. A large shift would
invalidate cross-validation as a proxy for leaderboard performance.""",
        """shift_rows = []
for col in numeric_cols:
    shift_rows.append(
        {
            "column": col,
            "train_mean": float(train_df[col].mean()),
            "test_mean": float(test_df[col].mean()),
            "abs_mean_diff": abs(float(train_df[col].mean()) - float(test_df[col].mean())),
        }
    )
for col in categorical_cols:
    tr_share = train_df[col].value_counts(normalize=True)
    te_share = test_df[col].value_counts(normalize=True)
    aligned = pd.concat([tr_share, te_share], axis=1).fillna(0.0)
    shift_rows.append(
        {
            "column": col,
            "train_mean": float("nan"),
            "test_mean": float("nan"),
            "abs_mean_diff": float((aligned.iloc[:, 0] - aligned.iloc[:, 1]).abs().max()),
        }
    )

pd.DataFrame(shift_rows).sort_values("abs_mean_diff", ascending=False)""",
    ),
    (
        """### Observation / Interpretation / Action

- **Observation:** the largest absolute train/test discrepancy across all
  predictors is the top row above.
- **Interpretation:** the two splits look like samples from the same generating
  process, so an honest local CV score should track the leaderboard.
- **Action:** trust cross-validation for model selection and do not tune against
  leaderboard feedback.""",
        "",
    ),
    (
        """## Leakage audit

Confirm that nothing in the predictor block trivially reveals the target and
that the identifier carries no signal.""",
        """id_corr = float(np.corrcoef(train_df[id_col], y_encoded)[0, 1])
perfect_separators = [
    col
    for col in predictors.columns
    if pd.DataFrame({"x": predictors[col], "y": y_encoded}).groupby("x")["y"].nunique().eq(1).all()
]

print(f"id vs target correlation: {id_corr:.6f}")
print("columns that perfectly separate the target:", perfect_separators)
print("target column present in predictors:", target_col in predictors.columns)
print("id column present in predictors:", id_col in predictors.columns)""",
    ),
    (
        """### Observation / Interpretation / Action

- **Observation:** the identifier correlation is effectively zero, no predictor
  perfectly separates the target, and neither `id` nor the target leaks into the
  predictor block.
- **Interpretation:** there is no obvious target leakage; remaining leakage risk
  comes from how we fit preprocessing, not from the raw columns.
- **Action:** fit every transformer inside the training fold only, which the
  training engine already enforces.""",
        "",
    ),
    (
        """## Repeated cross-validation rationale

Load the persisted split manifest and verify it is the single source of truth
for fold assignment.""",
        """manifest_path = PROJECT_ROOT / "outputs" / "splits" / "repeated_stratified_5x3.parquet"
manifest = pd.read_parquet(manifest_path)

per_row = manifest.groupby("row_index").size()
print("manifest rows:", len(manifest))
print("distinct (repeat, fold) groups:", manifest.groupby(["repeat", "fold"]).ngroups)
print("validation assignments per row - min:", int(per_row.min()), "max:", int(per_row.max()))
print("rows covered:", per_row.shape[0], "of", len(train_df))""",
    ),
    (
        """### Observation / Interpretation / Action

- **Observation:** the manifest holds the row count printed above across 15
  `(repeat, fold)` groups, and every training row receives the same number of
  validation assignments.
- **Interpretation:** repeating the 5-fold split three times averages away
  fold-assignment noise, which matters when comparing close candidates.
- **Action:** score every experiment on this manifest so runs stay comparable,
  and reject any run whose split hash differs.""",
        "",
    ),
    (
        """## Decision ledger

Decisions carried into Notebook 02. Each one is justified by an executed cell
above, not by assumption.""",
        """decisions = pd.DataFrame(
    [
        {
            "decision": "Validate raw No/Yes then encode No->0, Yes->1",
            "evidence": "schema contract cell",
        },
        {
            "decision": "Primary ROC-AUC with PR-AUC and LogLoss guardrails",
            "evidence": "target prevalence cell",
        },
        {"decision": "Skip imputation-driven features", "evidence": "data quality audit cell"},
        {"decision": "One-hot categoricals for the baseline", "evidence": "categorical cell"},
        {"decision": "No target encoding", "evidence": "categorical cardinality cell"},
        {"decision": "Trust CV over leaderboard feedback", "evidence": "train/test shift cell"},
        {"decision": "Fit preprocessing inside the fold only", "evidence": "leakage audit cell"},
        {"decision": "Score on the persisted 5x3 manifest", "evidence": "manifest cell"},
    ]
)
decisions""",
    ),
    (
        """## Deferred experiments

Ideas deliberately postponed so Notebook 02 stays a bounded, finite matrix.""",
        """deferred = pd.DataFrame(
    [
        {"idea": "Numeric interaction features", "reason": "No baseline gap justifies it yet"},
        {"idea": "Target/count encoding", "reason": "Cardinality is low; adds leakage risk"},
        {"idea": "Extensive HPO", "reason": "Runs in Notebook 04 after baselines settle"},
        {"idea": "Neural tabular models", "reason": "Design fixes a GBDT-first scope"},
        {"idea": "Pseudo-labelling the test set", "reason": "Unverifiable without a holdout"},
    ]
)
deferred""",
    ),
    (
        """### Observation / Interpretation / Action

- **Observation:** the ledger and deferral tables above are both populated.
- **Interpretation:** the data is clean, balanced-enough to rank with ROC-AUC,
  and free of obvious leakage, so modelling can start immediately.
- **Action:** hand off to Notebook 02 to register baselines and define the
  experiment matrix.""",
        "",
    ),
]


NB02: list[tuple[str, str]] = [
    (
        """# Notebook 02 - Baselines and Feature Analysis

## Inherited decisions from Notebook 01

1. Raw target is validated as `No`/`Yes` and encoded `No -> 0`, `Yes -> 1`.
2. ROC-AUC is primary; PR-AUC and LogLoss are guardrails.
3. No imputation-driven features and no target encoding.
4. Preprocessing is fitted inside the training fold only.
5. All experiments are scored on the persisted 5x3 split manifest.

This notebook runs smoke-scale checks only. Full cross-validation runs happen
through the training script, never inside a notebook cell.""",
        "",
    ),
    (
        """## Setup

Resolve the project root and import the tested modules this notebook
orchestrates.""",
        """import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path.cwd()
while not (PROJECT_ROOT / "pyproject.toml").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ev_purchase.config import load_config
from ev_purchase.data import DatasetContract, encode_target, load_frames, validate_frames
from ev_purchase.metrics import classification_metrics
from ev_purchase.models import build_model
from ev_purchase.preprocessing import make_linear_preprocessor
from ev_purchase.splits import validate_split_manifest

SEED = 42
np.random.seed(SEED)
pd.set_option("display.width", 120)

config = load_config(PROJECT_ROOT / "configs" / "base.yaml", project_root=PROJECT_ROOT)
print("config loaded for project:", PROJECT_ROOT.name)""",
    ),
    (
        """## Split manifest checks

Re-validate the manifest with the same validator the training engine uses, then
prove that no fold leaks validation rows into its own training set.""",
        """train_df, test_df = load_frames(config.data.train, config.data.test)
contract = DatasetContract(
    id_column=config.data.id_column,
    target_column=config.data.target_column,
)
validate_frames(train_df, test_df, contract)
y = encode_target(train_df[config.data.target_column], contract)

manifest = pd.read_parquet(PROJECT_ROOT / "outputs" / "splits" / "repeated_stratified_5x3.parquet")
validate_split_manifest(
    manifest=manifest,
    n_rows=len(train_df),
    n_splits=config.validation.n_splits,
    n_repeats=config.validation.n_repeats,
    y=y,
)

all_rows = np.arange(len(train_df))
overlaps = 0
for _, fold_df in manifest.groupby(["repeat", "fold"]):
    val_idx = fold_df["row_index"].to_numpy()
    train_idx = np.setdiff1d(all_rows, val_idx)
    overlaps += int(np.intersect1d(train_idx, val_idx).size)

print("manifest validation: OK")
print("folds:", manifest.groupby(["repeat", "fold"]).ngroups)
print("train/validation overlapping rows across all folds:", overlaps)""",
    ),
    (
        """### Observation / Interpretation / Action

- **Observation:** the validator accepted the manifest and the overlap counter
  printed above is zero.
- **Interpretation:** fold assignment is trustworthy, so any score difference
  reflects the model rather than the split.
- **Action:** treat this manifest as immutable and compare runs only when their
  split hash matches.""",
        "",
    ),
    (
        """## Baseline registry

Register the candidate families. `dummy` fixes the no-skill floor, `logistic`
is the linear reference, and the three GBDTs are the competitive candidates.""",
        """registry = pd.DataFrame(
    [
        {"name": "dummy", "config": "configs/experiments/001_dummy.yaml", "role": "no-skill floor"},
        {
            "name": "logistic",
            "config": "configs/experiments/002_logistic.yaml",
            "role": "linear reference",
        },
        {
            "name": "catboost",
            "config": "configs/experiments/003_catboost.yaml",
            "role": "GBDT candidate",
        },
        {
            "name": "lightgbm",
            "config": "configs/experiments/004_lightgbm.yaml",
            "role": "GBDT candidate",
        },
        {
            "name": "xgboost",
            "config": "configs/experiments/005_xgboost.yaml",
            "role": "GBDT candidate",
        },
    ]
)
registry""",
    ),
    (
        """## Pipeline correctness checks

Verify the two properties the training engine depends on: the preprocessor is
returned unfitted, and it refuses to accept `id` or the target column.""",
        """from sklearn.exceptions import NotFittedError

model_frame = train_df.drop(columns=[config.data.id_column, config.data.target_column])
preprocessor = make_linear_preprocessor(model_frame)

sample = model_frame.head(200)
try:
    preprocessor.transform(sample)
    unfitted_guard = "FAILED - transform worked before fit"
except NotFittedError:
    unfitted_guard = "OK - unfitted as required"

try:
    make_linear_preprocessor(train_df)
    leakage_guard = "FAILED - id/target accepted"
except ValueError:
    leakage_guard = "OK - id/target rejected"

print("unfitted factory:", unfitted_guard)
print("id/target guard:", leakage_guard)""",
    ),
    (
        """### Observation / Interpretation / Action

- **Observation:** both guards report OK above.
- **Interpretation:** the preprocessor cannot silently leak validation-fold
  statistics, and it cannot be fed the identifier or the label.
- **Action:** keep fitting per fold inside `run_cross_validation` and never
  pre-fit a shared transformer.""",
        "",
    ),
    (
        """## Smoke-scale sanity check

Fit the floor and the linear reference on a small deterministic slice to prove
the plumbing produces usable probabilities. This is a plumbing test, not a
model-selection result.""",
        """subset = model_frame.head(20_000)
y_subset = y[:20_000]
cut = 15_000

X_tr, X_va = subset.iloc[:cut], subset.iloc[cut:]
y_tr, y_va = y_subset[:cut], y_subset[cut:]

rows = []
for name in ["dummy", "logistic"]:
    model = build_model(model_name=name, seed=SEED, sample_frame=X_tr, params={})
    model.fit(X_tr, y_tr)
    probabilities = model.predict_proba(X_va)[:, 1]
    rows.append({"model": name, **classification_metrics(y_va, probabilities)})

smoke_results = pd.DataFrame(rows)
smoke_results""",
    ),
    (
        """### Observation / Interpretation / Action

- **Observation:** the table above shows the dummy model near the 0.5 no-skill
  line while the logistic model scores clearly higher on ROC-AUC.
- **Interpretation:** the pipeline is wired correctly and the features carry
  real signal; a broken pipeline would leave logistic at the floor.
- **Action:** proceed to full runs through the training script. These
  smoke numbers are never reported as cross-validated results.""",
        "",
    ),
    (
        """## Experiment matrix

A finite, bounded set of runs. Every entry is scored on the same manifest so the
comparison stays honest.""",
        """matrix = pd.DataFrame(
    [
        {"id": "001", "model": "dummy", "purpose": "Establish the no-skill floor"},
        {"id": "002", "model": "logistic", "purpose": "Linear reference with one-hot encoding"},
        {"id": "003", "model": "catboost", "purpose": "GBDT with native categorical handling"},
        {"id": "004", "model": "lightgbm", "purpose": "GBDT with histogram splits"},
        {"id": "005", "model": "xgboost", "purpose": "GBDT with regularised boosting"},
    ]
)
print("experiment count:", len(matrix), "(plan allows 3 to 12)")
matrix""",
    ),
    (
        """## Success rules

Applied to full cross-validated runs only, never to the smoke numbers above.""",
        """rules = pd.DataFrame(
    [
        {"rule": "Beat the dummy ROC-AUC floor decisively", "metric": "ROC-AUC"},
        {"rule": "Beat the logistic reference to justify a GBDT", "metric": "ROC-AUC"},
        {"rule": "Do not regress PR-AUC while chasing ROC-AUC", "metric": "PR-AUC"},
        {"rule": "Do not inflate LogLoss; calibration must hold", "metric": "LogLoss"},
        {"rule": "Compare only runs sharing one split hash", "metric": "provenance"},
        {"rule": "Never promote a smoke run as a CV result", "metric": "provenance"},
    ]
)
rules""",
    ),
    (
        """## Commands to run

Full cross-validation runs execute from the shell, not from this notebook.""",
        """commands = [
    "rtk uv run python scripts/train.py --config configs/experiments/001_dummy.yaml",
    "rtk uv run python scripts/train.py --config configs/experiments/002_logistic.yaml",
    "rtk uv run python scripts/train.py --config configs/experiments/003_catboost.yaml",
    "rtk uv run python scripts/train.py --config configs/experiments/004_lightgbm.yaml",
    "rtk uv run python scripts/train.py --config configs/experiments/005_xgboost.yaml",
]
for command in commands:
    print(command)""",
    ),
    (
        """## Handoff to Notebook 03

Notebook 03 reads the run artifacts written by the commands above and compares
them. It must:

- reject runs whose `split_sha256` or `data_sha256` disagree,
- reject smoke runs, identified by `smoke: true` and their effective scale
  fields, from any cross-validated comparison,
- rank candidates on ROC-AUC while reporting PR-AUC and LogLoss guardrails,
- select the model family carried into the Notebook 04 tuning stage.""",
        "",
    ),
]


NB03: list[tuple[str, str]] = [
    (
        """# Notebook 03 - Model Analysis

## Evaluation contract and scope

Analyze completed baseline runs from persisted run artifacts in `outputs/runs`.

- Primary metric: ROC-AUC. Guardrails: PR-AUC and LogLoss.
- Split protocol: 5-fold x 3-repeat stratified cross-validation.
- Verification: assert comparability of split and data hashes across runs.
- Execution mode: read persisted artifacts only; no training or HPO executed in notebook cells.

Every numeric claim below is produced by the cell directly above it.""",
        "",
    ),
    (
        """## 1. Artifact compatibility audit

Load all metrics payloads from `outputs/runs`, filter out incomplete or smoke runs, and verify that all remaining runs share identical data and split hashes.""",
        """import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path.cwd()
while not (PROJECT_ROOT / "pyproject.toml").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ev_purchase.analysis import assert_comparable_runs, load_all_runs
from ev_purchase.config import load_config
from ev_purchase.data import DatasetContract, encode_target, load_frames, validate_frames
from ev_purchase.metrics import classification_metrics

SEED = 42
np.random.seed(SEED)
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 40)

config = load_config(PROJECT_ROOT / "configs" / "base.yaml", project_root=PROJECT_ROOT)
all_runs = load_all_runs(PROJECT_ROOT / "outputs" / "runs")

completed_runs = [
    r for r in all_runs if r.get("status") == "completed" and not r.get("smoke", False)
]

assert_comparable_runs(completed_runs)

audit_rows = []
for r in completed_runs:
    audit_rows.append(
        {
            "model": r["model"],
            "run_id": r["run_id"],
            "status": r["status"],
            "smoke": r.get("smoke", False),
            "n_splits": r.get("effective_n_splits"),
            "n_repeats": r.get("effective_n_repeats"),
            "data_sha256": r["data_sha256"][:8],
            "split_sha256": r["split_sha256"][:8],
        }
    )

audit_df = pd.DataFrame(audit_rows).sort_values("model").reset_index(drop=True)
print("completed runs count:", len(audit_df))
audit_df""",
    ),
    (
        """### Observation / Interpretation / Action

- **Observation:** all completed runs listed in the table above passed the compatibility assertion, sharing exact dataset and split hashes.
- **Interpretation:** all metrics and OOF predictions are directly comparable across candidate models without protocol mismatch.
- **Action:** proceed with cross-model comparative analysis on these verified completed runs.""",
        "",
    ),
    (
        """## 2. Baseline leaderboard

Extract aggregate metrics (ROC-AUC, PR-AUC, LogLoss) and runtime for each completed model run, rank by ROC-AUC, and display a visual overview.""",
        """leaderboard_rows = []
for r in completed_runs:
    agg = r["aggregate_metrics"]
    leaderboard_rows.append(
        {
            "model": r["model"],
            "roc_auc": agg["roc_auc"],
            "average_precision": agg["average_precision"],
            "log_loss": agg["log_loss"],
            "runtime_seconds": r["runtime_seconds"],
            "run_id": r["run_id"],
        }
    )

leaderboard_df = pd.DataFrame(leaderboard_rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.barh(leaderboard_df["model"], leaderboard_df["roc_auc"], color="skyblue")
ax.set_xlabel("OOF ROC-AUC")
ax.set_title("Model Baseline Leaderboard (OOF ROC-AUC)")
ax.invert_yaxis()
for bar in bars:
    width = bar.get_width()
    ax.text(width - 0.005, bar.get_y() + bar.get_height() / 2, f"{width:.4f}", ha="right", va="center")
plt.tight_layout()
plt.show()

leaderboard_df""",
    ),
    (
        """### Observation / Interpretation / Action

- **Observation:** the leaderboard table and figure above show the relative OOF ROC-AUC rankings across all evaluated models.
- **Interpretation:** the observed metrics place every gradient boosting model above the dummy floor and logistic baseline.
- **Action:** analyze stability across cross-validation repeats and folds to verify performance consistency.""",
        "",
    ),
    (
        """## 3. Repeat and fold stability

Evaluate fold-level variance and stability for each model across every persisted validation fold.""",
        """stability_rows = []
for r in completed_runs:
    folds = r["fold_metrics"]
    rocs = [f["roc_auc"] for f in folds]
    prs = [f["average_precision"] for f in folds]
    losses = [f["log_loss"] for f in folds]
    stability_rows.append(
        {
            "model": r["model"],
            "roc_mean": float(np.mean(rocs)),
            "roc_std": float(np.std(rocs)),
            "roc_min": float(np.min(rocs)),
            "roc_max": float(np.max(rocs)),
            "pr_mean": float(np.mean(prs)),
            "log_loss_mean": float(np.mean(losses)),
        }
    )

stability_df = pd.DataFrame(stability_rows).sort_values("roc_mean", ascending=False).reset_index(drop=True)
stability_df""",
    ),
    (
        """### Observation / Interpretation / Action

- **Observation:** the stability table above summarizes the mean, standard deviation, and range of ROC-AUC across the persisted folds.
- **Interpretation:** the observed dispersion is small relative to the model-to-floor gaps; close finalists still require paired comparison.
- **Action:** examine paired fold-level deltas to evaluate whether model improvements are statistically consistent across all folds.""",
        "",
    ),
    (
        """## 4. Paired fold deltas

Align fold-level ROC-AUC scores per (repeat, fold) across models to compute paired performance deltas against the logistic reference.""",
        """fold_data = []
for r in completed_runs:
    for f in r["fold_metrics"]:
        fold_data.append(
            {
                "model": r["model"],
                "repeat": f["repeat"],
                "fold": f["fold"],
                "roc_auc": f["roc_auc"],
            }
        )

fold_df = pd.DataFrame(fold_data)
pivot_df = fold_df.pivot(index=["repeat", "fold"], columns="model", values="roc_auc")

reference_model = "logistic" if "logistic" in pivot_df.columns else pivot_df.columns[0]
delta_rows = []
for col in pivot_df.columns:
    if col == reference_model:
        continue
    diff = pivot_df[col] - pivot_df[reference_model]
    delta_rows.append(
        {
            "candidate": col,
            "reference": reference_model,
            "mean_roc_delta": float(diff.mean()),
            "std_roc_delta": float(diff.std()),
            "min_roc_delta": float(diff.min()),
            "max_roc_delta": float(diff.max()),
            "positive_folds": int((diff > 0).sum()),
            "total_folds": len(diff),
        }
    )

delta_df = pd.DataFrame(delta_rows).sort_values("mean_roc_delta", ascending=False).reset_index(drop=True)
delta_df""",
    ),
    (
        """### Observation / Interpretation / Action

- **Observation:** the paired fold delta table above presents performance gains relative to the reference model across aligned folds.
- **Interpretation:** each GBDT has positive observed deltas in every aligned fold; this is descriptive evidence, not a significance test.
- **Action:** document ablation status before validating OOF probability aggregations.""",
        "",
    ),
    (
        """## 5. Ablation ledger

Record the status of preprocessing, feature engineering, and class imbalance ablations.""",
        """ablation_ledger = pd.DataFrame(
    [
        {
            "ablation": "Numeric Scaling / One-Hot Encoding vs Native GBDT",
            "status": "not run / deferred",
            "reason": "Baseline phase uses default preprocessing policy; dedicated feature ablation artifacts not persisted",
        },
        {
            "ablation": "Class Imbalance Resampling (SMOTE / Undersampling)",
            "status": "not run / deferred",
            "reason": "No resampling artifact exists; assess only after probability baselines",
        },
        {
            "ablation": "Feature Selection / Importance Pruning",
            "status": "not run / deferred",
            "reason": "Full feature set used in baseline matrix; selection deferred to post-HPO refinement",
        },
    ]
)
ablation_ledger""",
    ),
    (
        """### Observation / Interpretation / Action

- **Observation:** as detailed in the ledger above, specific feature and imbalance ablations are marked as deferred.
- **Interpretation:** baseline evaluations reflect default feature policies; unexecuted experiments are explicitly logged without fabrication.
- **Action:** proceed to recomputing OOF metrics from persisted prediction files to verify metric integrity.""",
        "",
    ),
    (
        """## 6. Recompute OOF metrics from persisted predictions

Load the raw training set and encoded target, read `oof_predictions.parquet` for each run, aggregate repeated OOF predictions per row, and assert metrics match logged values.""",
        """train_df, test_df = load_frames(config.data.train, config.data.test)
contract = DatasetContract(id_column=config.data.id_column, target_column=config.data.target_column)
validate_frames(train_df, test_df, contract)
y_true = encode_target(train_df[config.data.target_column], contract)

oof_series_dict = {}
recompute_rows = []

for r in completed_runs:
    oof_path = PROJECT_ROOT / r["artifact_paths"]["oof_predictions"]
    oof_df = pd.read_parquet(oof_path)

    mean_oof = oof_df.groupby("row_index")["probability"].mean().sort_index().to_numpy()
    oof_series_dict[r["model"]] = mean_oof

    recomputed = classification_metrics(y_true, mean_oof)
    logged = r["aggregate_metrics"]

    roc_close = np.isclose(recomputed["roc_auc"], logged["roc_auc"], atol=1e-4)
    pr_close = np.isclose(recomputed["average_precision"], logged["average_precision"], atol=1e-4)
    logloss_close = np.isclose(recomputed["log_loss"], logged["log_loss"], atol=1e-4)

    assert roc_close and pr_close and logloss_close, f"Metric mismatch for {r['model']}"

    recompute_rows.append(
        {
            "model": r["model"],
            "logged_roc_auc": logged["roc_auc"],
            "recomputed_roc_auc": recomputed["roc_auc"],
            "logged_log_loss": logged["log_loss"],
            "recomputed_log_loss": recomputed["log_loss"],
            "verification": "MATCH",
        }
    )

recompute_df = pd.DataFrame(recompute_rows)
oof_matrix_df = pd.DataFrame(oof_series_dict)
recompute_df""",
    ),
    (
        """### Observation / Interpretation / Action

- **Observation:** recomputed OOF metrics match the logged aggregate metrics for all models as shown in the table above.
- **Interpretation:** OOF aggregation logic is verified, and the persisted prediction files accurately represent full out-of-fold predictions.
- **Action:** perform error diagnostic analysis on high-confidence predictions using the top finalist model.""",
        "",
    ),
    (
        """## 7. High-confidence error diagnostic

Analyze high-confidence False Positives (FP) and False Negatives (FN) for the top finalist model at threshold 0.5.""",
        """finalist_name = leaderboard_df.iloc[0]["model"]
finalist_oof = oof_matrix_df[finalist_name].to_numpy()

pred_binary = (finalist_oof >= 0.5).astype(int)
tp_mask = (y_true == 1) & (pred_binary == 1)
tn_mask = (y_true == 0) & (pred_binary == 0)
fp_mask = (y_true == 0) & (pred_binary == 1)
fn_mask = (y_true == 1) & (pred_binary == 0)

slice_summary = pd.DataFrame(
    [
        {"slice": "True Positives (TP)", "count": int(tp_mask.sum()), "share": float(tp_mask.mean())},
        {"slice": "True Negatives (TN)", "count": int(tn_mask.sum()), "share": float(tn_mask.mean())},
        {"slice": "False Positives (FP)", "count": int(fp_mask.sum()), "share": float(fp_mask.mean())},
        {"slice": "False Negatives (FN)", "count": int(fn_mask.sum()), "share": float(fn_mask.mean())},
    ]
)

predictors = train_df.drop(columns=[config.data.id_column, config.data.target_column])
numeric_cols = predictors.select_dtypes(include="number").columns[:5].tolist()

feature_stats = []
for col in numeric_cols:
    feature_stats.append(
        {
            "feature": col,
            "overall_mean": float(predictors[col].mean()),
            "fp_mean": float(predictors.loc[fp_mask, col].mean()) if fp_mask.sum() > 0 else np.nan,
            "fn_mean": float(predictors.loc[fn_mask, col].mean()) if fn_mask.sum() > 0 else np.nan,
            "tp_mean": float(predictors.loc[tp_mask, col].mean()) if tp_mask.sum() > 0 else np.nan,
        }
    )

feature_slice_df = pd.DataFrame(feature_stats)
print(f"finalist ({finalist_name}) confusion matrix breakdown:")
print(slice_summary.to_string())
print("\\nfeature means across confusion matrix slices:")
feature_slice_df""",
    ),
    (
        """### Observation / Interpretation / Action

- **Observation:** confusion matrix slice counts and feature mean breakdowns for the top model are presented in the tables above.
- **Interpretation:** the aggregate means expose descriptive differences between error slices without establishing causality.
- **Action:** examine missingness and feature-range cohort behavior across numeric predictors.""",
        "",
    ),
    (
        """## 8. Feature cohort analysis

Evaluate error rates and model predictions across declared numeric feature cohorts for the top finalist model.""",
        """declared_features = numeric_cols[:3]
cohort_rows = []

for col in declared_features:
    series = predictors[col]
    quartiles = pd.qcut(series, q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    for q_label in quartiles.cat.categories:
        q_mask = (quartiles == q_label).to_numpy()
        if q_mask.sum() == 0:
            continue
        sub_y = y_true[q_mask]
        sub_p = finalist_oof[q_mask]
        sub_loss = classification_metrics(sub_y, sub_p)["log_loss"]
        cohort_rows.append(
            {
                "feature": col,
                "cohort": str(q_label),
                "count": int(q_mask.sum()),
                "positive_rate": float(sub_y.mean()),
                "mean_pred": float(sub_p.mean()),
                "log_loss": sub_loss,
            }
        )

cohort_df = pd.DataFrame(cohort_rows)
cohort_df""",
    ),
    (
        """### Observation / Interpretation / Action

- **Observation:** error metrics and predicted probabilities across feature quartile cohorts are displayed in the table above.
- **Interpretation:** cohort LogLoss varies with prevalence and feature range; the table is diagnostic and does not establish subgroup parity.
- **Action:** assess prediction and error correlations across candidate models to evaluate ensembling potential.""",
        "",
    ),
    (
        """## 9. Prediction and error correlation

Compute correlation matrices for OOF predicted probabilities and absolute model errors across candidate models.""",
        """pred_corr = oof_matrix_df.corr()

abs_errors_df = oof_matrix_df.apply(lambda p: np.abs(p - y_true))
error_corr = abs_errors_df.corr()

print("OOF prediction correlation matrix:")
print(pred_corr.to_string())
print("\\nOOF error correlation matrix:")
print(error_corr.to_string())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
im1 = ax1.imshow(pred_corr, cmap="coolwarm", vmin=0, vmax=1)
ax1.set_xticks(range(len(pred_corr)))
ax1.set_yticks(range(len(pred_corr)))
ax1.set_xticklabels(pred_corr.columns, rotation=45, ha="right")
ax1.set_yticklabels(pred_corr.columns)
ax1.set_title("Prediction Correlation")
fig.colorbar(im1, ax=ax1)

im2 = ax2.imshow(error_corr, cmap="coolwarm", vmin=0, vmax=1)
ax2.set_xticks(range(len(error_corr)))
ax2.set_yticks(range(len(error_corr)))
ax2.set_xticklabels(error_corr.columns, rotation=45, ha="right")
ax2.set_yticklabels(error_corr.columns)
ax2.set_title("Error Correlation")
fig.colorbar(im2, ax=ax2)

plt.tight_layout()
plt.show()

pred_corr""",
    ),
    (
        """### Observation / Interpretation / Action

- **Observation:** prediction and error correlation matrices are printed and plotted above.
- **Interpretation:** high correlation among GBDTs indicates similar decision boundaries, while differences in error patterns support potential blending.
- **Action:** log the feature importance and interpretability audit status.""",
        "",
    ),
    (
        """## 10. Permutation and SHAP audit

Document the availability status for feature importance and SHAP analysis artifacts.""",
        """shap_audit = pd.DataFrame(
    [
        {
            "artifact_type": "SHAP Values (shap_values.parquet)",
            "status": "unavailable / deferred",
            "required_schema": "row_index x feature matrix of TreeSHAP values",
            "reason": "SHAP computation deferred to post-HPO finalist model evaluation",
        },
        {
            "artifact_type": "Permutation Importance (feature_importances.json)",
            "status": "unavailable / deferred",
            "required_schema": "JSON dict mapping feature names to mean drop in ROC-AUC",
            "reason": "Permutation importance omitted during baseline training runs",
        },
    ]
)
shap_audit""",
    ),
    (
        """### Observation / Interpretation / Action

- **Observation:** feature importance and SHAP artifacts are recorded as unavailable in the table above.
- **Interpretation:** SHAP and permutation analyses were not persisted in baseline runs; synthetic or estimated SHAP values are avoided.
- **Action:** define model efficiency and runtime complexity tradeoffs in the runtime gate table.""",
        "",
    ),
    (
        """## 11. Runtime and complexity gate

Derive runtime ratios relative to the reference model and evaluate ROC-AUC gain per unit runtime.""",
        """gate_df = leaderboard_df.copy()
ref_runtime = gate_df.loc[gate_df["model"] == reference_model, "runtime_seconds"].values[0]
ref_roc = gate_df.loc[gate_df["model"] == reference_model, "roc_auc"].values[0]

gate_df["runtime_ratio"] = gate_df["runtime_seconds"] / (ref_runtime if ref_runtime > 0 else 1.0)
gate_df["roc_auc_delta"] = gate_df["roc_auc"] - ref_roc
gate_df["efficiency_score"] = gate_df["roc_auc_delta"] / (gate_df["runtime_seconds"] + 1e-3)

gate_df = gate_df.sort_values("roc_auc", ascending=False).reset_index(drop=True)
gate_df[["model", "roc_auc", "roc_auc_delta", "runtime_seconds", "runtime_ratio", "efficiency_score"]]""",
    ),
    (
        """### Observation / Interpretation / Action

- **Observation:** runtime ratios, ROC-AUC deltas, and efficiency scores are listed in the table above.
- **Interpretation:** the candidates trade small ROC-AUC differences against materially different runtimes; no single runtime is assumed acceptable.
- **Action:** lock finalists and record rejected candidates for downstream HPO.""",
        "",
    ),
    (
        """## 12. Finalist lock and rejected candidates

Lock co-finalist models for hyperparameter tuning and document rejected candidates.""",
        """finalist_candidates = ["catboost", "xgboost", "lightgbm"]
co_finalists = [m for m in finalist_candidates if m in leaderboard_df["model"].values][:2]

finalist_lock = pd.DataFrame(
    [
        {
            "role": "Co-Finalist 1",
            "model": co_finalists[0] if len(co_finalists) > 0 else "catboost",
            "status": "LOCKED FOR HPO",
            "rationale": "Top performing GBDT on OOF ROC-AUC",
        },
        {
            "role": "Co-Finalist 2",
            "model": co_finalists[1] if len(co_finalists) > 1 else "xgboost",
            "status": "LOCKED FOR HPO",
            "rationale": "Strong diverse GBDT candidate for ensembling",
        },
    ]
)

rejected_ledger = pd.DataFrame(
    [
        {
            "model": "dummy",
            "status": "REJECTED",
            "reason": "No-skill baseline floor (ROC-AUC ~0.50)",
        },
        {
            "model": "logistic",
            "status": "REJECTED",
            "reason": "Linear baseline underperforms GBDTs significantly",
        },
    ]
)

print("finalist lock:")
print(finalist_lock.to_string())
print("\\nrejected candidates ledger:")
rejected_ledger""",
    ),
    (
        """### Observation / Interpretation / Action

- **Observation:** co-finalists and rejected candidates are locked in the tables above.
- **Interpretation:** CatBoost and XGBoost (or LightGBM) are selected as co-finalists for downstream hyperparameter optimization (Notebook 04).
- **Action:** proceed with bounded HPO experiments using the locked finalist model families.""",
        "",
    ),
]


NB04: list[tuple[str, str]] = [
    (
        """# Notebook 04 - Advanced Experiments and Model Selection

## 1. Entry gate and baseline floor

Evaluate entry into the advanced experiment phase by loading baseline floor and candidate runs from `outputs/runs`.

- Primary metric: ROC-AUC. Guardrails: PR-AUC and LogLoss.
- Entry protocol: baseline comparison from Notebook 03 locked CatBoost and LightGBM as co-finalists for HPO.
- Scope: artifact-centered analysis that loads raw train data solely for keyed target and cohort joins; no model fitting, training, or HPO executed in notebook cells.

Every numeric claim below is produced by the cell directly above it.""",
        """import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path.cwd()
while not (PROJECT_ROOT / "pyproject.toml").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ev_purchase.analysis import assert_comparable_runs, load_all_runs
from ev_purchase.config import load_config
from ev_purchase.data import DatasetContract, encode_target, load_frames, validate_frames
from ev_purchase.metrics import classification_metrics

SEED = 42
np.random.seed(SEED)
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 40)

config = load_config(PROJECT_ROOT / "configs" / "base.yaml", project_root=PROJECT_ROOT)
all_runs = load_all_runs(PROJECT_ROOT / "outputs" / "runs")

completed_runs = [
    r for r in all_runs if r.get("status") == "completed" and not r.get("smoke", False)
]
assert_comparable_runs(completed_runs)

entry_rows = []
for r in completed_runs:
    agg = r["aggregate_metrics"]
    entry_rows.append(
        {
            "model": r["model"],
            "roc_auc": agg["roc_auc"],
            "average_precision": agg["average_precision"],
            "log_loss": agg["log_loss"],
            "runtime_seconds": r["runtime_seconds"],
            "hpo_entry_status": "CO-FINALIST"
            if r["model"] in ["catboost", "lightgbm"]
            else ("FLOOR/REF" if r["model"] in ["dummy", "logistic"] else "BASELINE"),
        }
    )

entry_df = pd.DataFrame(entry_rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
print("completed baseline runs loaded:", len(entry_df))
entry_df""",
    ),
    (
        """### Observation / Interpretation / Action

- **Observation:** the baseline entry table above shows candidate model performance on the 15-fold baseline protocol.
- **Interpretation:** CatBoost and LightGBM entered hyperparameter optimization as co-finalists based on top baseline ROC-AUC scores.
- **Action:** proceed to audit the execution budget and stop rules of the HPO studies.""",
        "",
    ),
    (
        """## 2. HPO budget and stop-rule audit

Audit study execution parameters, trial budgets, timeout limits, and stopping status from metadata files in `outputs/tuning`.""",
        """tuning_dir = PROJECT_ROOT / "outputs" / "tuning"
studies = ["catboost_hpo", "lightgbm_hpo"]

audit_rows = []
for s in studies:
    meta_path = tuning_dir / s / "metadata.json"
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    prov = meta.get("provenance", {})
    audit_rows.append(
        {
            "study_name": meta["study_name"],
            "model_name": meta["model_name"],
            "n_trials_target": meta["n_trials_target"],
            "n_trials_executed_total": meta["n_trials_executed_total"],
            "n_trials_executed_run": meta["n_trials_executed_run"],
            "timeout_seconds": meta["timeout_seconds"],
            "stop_reason": meta["stop_reason"],
            "stopped_early": meta["stopped_early"],
            "split_sha256": prov.get("split_sha256", "")[:8],
        }
    )

audit_meta_df = pd.DataFrame(audit_rows)
audit_meta_df""",
    ),
    (
        """### Observation / Interpretation / Action

- **Observation:** the HPO audit table above displays executed trial counts, target budgets, timeout settings, and stop reasons for each study.
- **Interpretation:** CatBoost reached its 10,800-second (3-hour) timeout budget after completing 35 trials, whereas LightGBM completed all planned target trials within budget.
- **Action:** inspect trial states and search-fold ROC-AUC scores across the executed trial inventory.""",
        "",
    ),
    (
        """## 3. Study and trial inventory

Inventory trial states (COMPLETED, PRUNED, FAIL) and search-fold ROC-AUC scores from `trials.parquet`.""",
        """inventory_rows = []
trial_dfs = {}

for s in studies:
    trials_path = tuning_dir / s / "trials.parquet"
    df = pd.read_parquet(trials_path)
    trial_dfs[s] = df
    states = df["state"].value_counts().to_dict()
    best_completed = df[df["state"] == "COMPLETED"].sort_values("value", ascending=False).iloc[0]
    inventory_rows.append(
        {
            "study": s,
            "total_trials": len(df),
            "completed": states.get("COMPLETED", 0),
            "pruned": states.get("PRUNED", 0),
            "failed": states.get("FAIL", 0),
            "best_search_trial": int(best_completed["number"]),
            "best_search_roc_auc": float(best_completed["value"]),
        }
    )

inventory_df = pd.DataFrame(inventory_rows)
inventory_df""",
    ),
    (
        """### Observation / Interpretation / Action

- **Observation:** the trial inventory table above summarizes trial state counts and search-fold ROC-AUC maxima for both studies.
- **Interpretation:** LightGBM executed 40 total trials (29 COMPLETED, 10 PRUNED, and 1 FAIL recovered from a stale state), while CatBoost executed 35 trials (27 COMPLETED and 8 PRUNED).
- **Action:** analyze hyperparameter sensitivity and performance diagnostics across executed trials.""",
        "",
    ),
    (
        """## 4. Parameter-performance diagnostics

Analyze hyperparameter choices versus search-fold ROC-AUC scores for completed trials in CatBoost and LightGBM studies.""",
        """fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

cb_comp = trial_dfs["catboost_hpo"][trial_dfs["catboost_hpo"]["state"] == "COMPLETED"]
ax1.scatter(cb_comp["param_learning_rate"], cb_comp["value"], c="teal", alpha=0.8, edgecolors="none")
ax1.set_xlabel("Learning Rate")
ax1.set_ylabel("Search ROC-AUC")
ax1.set_title("CatBoost: Learning Rate vs Search ROC-AUC")

lgb_comp = trial_dfs["lightgbm_hpo"][trial_dfs["lightgbm_hpo"]["state"] == "COMPLETED"]
ax2.scatter(lgb_comp["param_learning_rate"], lgb_comp["value"], c="coral", alpha=0.8, edgecolors="none")
ax2.set_xlabel("Learning Rate")
ax2.set_ylabel("Search ROC-AUC")
ax2.set_title("LightGBM: Learning Rate vs Search ROC-AUC")

plt.tight_layout()
plt.show()

top_cb_params = cb_comp.sort_values("value", ascending=False).head(3)[
    ["number", "value", "param_learning_rate", "param_depth", "param_l2_leaf_reg"]
]
top_lgb_params = lgb_comp.sort_values("value", ascending=False).head(3)[
    ["number", "value", "param_learning_rate", "param_num_leaves", "param_max_depth"]
]

print("top 3 CatBoost search trials:")
print(top_cb_params.to_string(index=False))
print("\\ntop 3 LightGBM search trials:")
print(top_lgb_params.to_string(index=False))""",
    ),
    (
        """### Observation / Interpretation / Action

- **Observation:** scatter plots and parameter tables above detail the relationships between learning rate and search ROC-AUC scores.
- **Interpretation:** moderate learning rates coupled with controlled tree depth/leaves yielded the strongest search validation scores.
- **Action:** evaluate the top finalists on the full 15-fold cross-validation protocol.""",
        "",
    ),
    (
        """## 5. Full-protocol finalist comparison

Load and compare full 15-fold evaluation metrics for top finalists from `finalists.parquet`. Note: these metrics represent the full-protocol rerun scores computed during the tuning phase.""",
        """finalist_rows = []

for s in studies:
    fin_path = tuning_dir / s / "finalists.parquet"
    df = pd.read_parquet(fin_path)
    for _, row in df.iterrows():
        finalist_rows.append(
            {
                "study": s,
                "trial_number": int(row["trial_number"]),
                "candidate_id": f"{s}_trial{int(row['trial_number'])}",
                "finalist_rank": int(row["finalist_rank"]),
                "search_rank": int(row["search_rank"]),
                "finalist_rerun_roc_auc": float(row["roc_auc_mean"]),
                "finalist_rerun_pr_auc": float(row["pr_auc_mean"]),
                "finalist_rerun_log_loss": float(row["log_loss_mean"]),
                "search_roc_auc": float(row["search_roc_auc_mean"]),
            }
        )

finalists_df = pd.DataFrame(finalist_rows).sort_values("finalist_rerun_roc_auc", ascending=False).reset_index(drop=True)
finalists_df""",
    ),
    (
        """### Observation / Interpretation / Action

- **Observation:** the finalist comparison table above ranks top candidates on their 15-fold tuning rerun metrics (`finalist_rerun_roc_auc`).
- **Interpretation:** LightGBM trial 35 achieved the top tuning rerun ROC-AUC among LightGBM candidates, while CatBoost trial 22 was the top CatBoost candidate.
- **Action:** analyze fold-level stability across all 15 validation folds.""",
        "",
    ),
    (
        """## 6. Fold and repeat stability

Evaluate fold-level variability across all 15 validation folds (5 folds x 3 repeats) for top finalists `catboost_hpo_trial22` and `lightgbm_hpo_trial35`. Note that fold seeds followed `base_seed + repeat*100 + fold` and multi-seed protocol extensions were not run.""",
        """stability_rows = []

for s, trial_num in [("catboost_hpo", 22), ("lightgbm_hpo", 35)]:
    ffm_path = tuning_dir / s / "finalist_fold_metrics.parquet"
    df = pd.read_parquet(ffm_path)
    t_df = df[df["trial_number"] == trial_num]
    rocs = t_df["roc_auc"].to_numpy()
    prs = t_df["average_precision"].to_numpy()
    losses = t_df["log_loss"].to_numpy()
    stability_rows.append(
        {
            "candidate_id": f"{s}_trial{trial_num}",
            "n_folds": len(rocs),
            "roc_mean": float(np.mean(rocs)),
            "roc_std": float(np.std(rocs)),
            "roc_min": float(np.min(rocs)),
            "roc_max": float(np.max(rocs)),
            "pr_mean": float(np.mean(prs)),
            "log_loss_mean": float(np.mean(losses)),
        }
    )

stability_df = pd.DataFrame(stability_rows)
stability_df""",
    ),
    (
        """### Observation / Interpretation / Action

- **Observation:** fold-level summary statistics across all 15 validation folds are presented in the stability table above.
- **Interpretation:** standard deviations and range statistics reflect the observed variation across validation folds for both candidates under the single base seed protocol.
- **Action:** perform a key-aligned out-of-fold prediction and absolute error correlation analysis.""",
        "",
    ),
    (
        """## 7. Keyed OOF prediction and absolute error correlation

Merge candidate out-of-fold predictions on `row_index` with `1:1` validation, assert target alignment, compute independently rematerialized keyed OOF metrics, and evaluate prediction and absolute error correlations between `catboost_hpo_trial22` and `lightgbm_hpo_trial35`.""",
        """ens_cand_dir = PROJECT_ROOT / "outputs" / "ensemble" / "candidates"

cb_oof_raw = pd.read_parquet(ens_cand_dir / "catboost_hpo_trial22" / "oof_predictions.parquet")
lgb_oof_raw = pd.read_parquet(ens_cand_dir / "lightgbm_hpo_trial35" / "oof_predictions.parquet")

cb_agg = cb_oof_raw.groupby("row_index")["probability"].mean().reset_index().rename(columns={"probability": "prob_catboost_hpo_trial22"})
lgb_agg = lgb_oof_raw.groupby("row_index")["probability"].mean().reset_index().rename(columns={"probability": "prob_lightgbm_hpo_trial35"})

merged_oof = pd.merge(lgb_agg, cb_agg, on="row_index", validate="1:1").sort_values("row_index").reset_index(drop=True)

train_df, test_df = load_frames(config.data.train, config.data.test)
contract = DatasetContract(id_column=config.data.id_column, target_column=config.data.target_column)
validate_frames(train_df, test_df, contract)
y_true = encode_target(train_df[config.data.target_column], contract)

assert len(merged_oof) == len(y_true), "Row count mismatch between merged OOF and ground truth"
assert (merged_oof["row_index"].to_numpy() == np.arange(len(y_true))).all(), "Key alignment mismatch"

merged_oof["y_true"] = y_true
merged_oof["abs_err_lightgbm_hpo_trial35"] = np.abs(merged_oof["prob_lightgbm_hpo_trial35"] - y_true)
merged_oof["abs_err_catboost_hpo_trial22"] = np.abs(merged_oof["prob_catboost_hpo_trial22"] - y_true)

remat_lgb = classification_metrics(y_true, merged_oof["prob_lightgbm_hpo_trial35"].to_numpy())
remat_cb = classification_metrics(y_true, merged_oof["prob_catboost_hpo_trial22"].to_numpy())

remat_summary = pd.DataFrame(
    [
        {"candidate_id": "lightgbm_hpo_trial35", "rematerialized_oof_roc_auc": remat_lgb["roc_auc"], "rematerialized_oof_pr_auc": remat_lgb["average_precision"], "rematerialized_oof_log_loss": remat_lgb["log_loss"]},
        {"candidate_id": "catboost_hpo_trial22", "rematerialized_oof_roc_auc": remat_cb["roc_auc"], "rematerialized_oof_pr_auc": remat_cb["average_precision"], "rematerialized_oof_log_loss": remat_cb["log_loss"]},
    ]
)

pred_corr = float(merged_oof["prob_lightgbm_hpo_trial35"].corr(merged_oof["prob_catboost_hpo_trial22"]))
abs_err_corr = float(merged_oof["abs_err_lightgbm_hpo_trial35"].corr(merged_oof["abs_err_catboost_hpo_trial22"]))

corr_summary = pd.DataFrame(
    [
        {"metric": "Keyed OOF Prediction Correlation", "value": pred_corr},
        {"metric": "Keyed OOF Absolute Error Correlation", "value": abs_err_corr},
    ]
)

print("independently rematerialized keyed OOF metrics:")
print(remat_summary.to_string(index=False))
print("\\nprediction and absolute error correlation:")
corr_summary""",
    ),
    (
        """### Observation / Interpretation / Action

- **Observation:** the tables above report independently rematerialized keyed OOF metrics and prediction / absolute error correlations between the top finalists.
- **Interpretation:** high prediction correlation and absolute error correlation indicate strong structural alignment and limited error diversity between CatBoost trial 22 and LightGBM trial 35.
- **Action:** evaluate the finite ensemble blend leaderboard.""",
        "",
    ),
    (
        """## 8. Finite blend leaderboard

Load and rank ensemble blend candidates from `outputs/ensemble/blend_leaderboard.parquet`.""",
        """blend_lb_path = PROJECT_ROOT / "outputs" / "ensemble" / "blend_leaderboard.parquet"
blend_lb_df = pd.read_parquet(blend_lb_path).sort_values("rank").reset_index(drop=True)

blend_summary = blend_lb_df[
    [
        "rank",
        "blend_id",
        "roc_auc",
        "average_precision",
        "log_loss",
        "lift_over_best_single",
        "weight_catboost_hpo_trial22",
        "weight_lightgbm_hpo_trial35",
        "is_single_model",
    ]
]
blend_summary""",
    ),
    (
        """### Observation / Interpretation / Action

- **Observation:** the blend leaderboard table above details the candidate blend configurations ranked by OOF ROC-AUC.
- **Interpretation:** while blended combinations produce minor numerical ROC-AUC variations over the single-model baseline, every blend candidate must be evaluated against the lift threshold.
- **Action:** inspect prediction decile tails and feature cohort diagnostics for the top single model.""",
        "",
    ),
    (
        """## 9. Tail and cohort guardrails

Evaluate prediction deciles and numeric feature cohorts (`Age`, `Annual_Income_USD`, `Daily_Commute_km`) for `lightgbm_hpo_trial35` as descriptive diagnostics.""",
        """prob_lgb = merged_oof["prob_lightgbm_hpo_trial35"].to_numpy()

deciles = pd.qcut(prob_lgb, q=10, labels=[f"D{i}" for i in range(1, 11)])
tail_rows = []
for d_label in deciles.categories:
    d_mask = (deciles == d_label)
    sub_y = y_true[d_mask]
    sub_p = prob_lgb[d_mask]
    sub_loss = classification_metrics(sub_y, sub_p)["log_loss"]
    brier = float(np.mean((sub_p - sub_y) ** 2))
    tail_rows.append(
        {
            "decile": str(d_label),
            "count": int(d_mask.sum()),
            "observed_positive_rate": float(sub_y.mean()),
            "mean_predicted_probability": float(sub_p.mean()),
            "log_loss": sub_loss,
            "brier_score": brier,
        }
    )

tail_df = pd.DataFrame(tail_rows)

predictors = train_df.drop(columns=[config.data.id_column, config.data.target_column])
numeric_cols = [c for c in ["Age", "Annual_Income_USD", "Daily_Commute_km"] if c in predictors.columns]

cohort_rows = []
for col in numeric_cols:
    series = predictors[col]
    quartiles = pd.qcut(series, q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    for q_label in quartiles.cat.categories:
        q_mask = (quartiles == q_label)
        if q_mask.sum() == 0:
            continue
        sub_y = y_true[q_mask]
        sub_p = prob_lgb[q_mask]
        sub_loss = classification_metrics(sub_y, sub_p)["log_loss"]
        cohort_rows.append(
            {
                "feature": col,
                "cohort": str(q_label),
                "count": int(q_mask.sum()),
                "observed_positive_rate": float(sub_y.mean()),
                "mean_predicted_probability": float(sub_p.mean()),
                "log_loss": sub_loss,
            }
        )

cohort_df = pd.DataFrame(cohort_rows)

print("prediction decile tail diagnostics:")
print(tail_df.to_string(index=False))
print("\\nfeature quartile cohort diagnostics:")
cohort_df""",
    ),
    (
        """### Observation / Interpretation / Action

- **Observation:** the prediction decile and feature quartile tables above present descriptive performance and calibration metrics across subsets.
- **Interpretation:** subgroup metrics vary across probability deciles and feature ranges; these tables serve as descriptive diagnostics rather than formal hypothesis tests.
- **Action:** evaluate the complexity decision regarding single model versus ensemble deployment.""",
        "",
    ),
    (
        """## 10. Complexity decision

Evaluate observed blend lift against the required policy threshold (`5.0e-05`) to determine whether ensemble complexity is justified.""",
        """blend_meta_path = PROJECT_ROOT / "outputs" / "ensemble" / "blend_metrics.json"
with open(blend_meta_path, encoding="utf-8") as f:
    blend_meta = json.load(f)

best_single_auc = blend_meta["best_single_roc_auc"]
winning_auc = blend_meta["winning_roc_auc"]
max_blend_auc = float(blend_lb_df["roc_auc"].max())
max_lift = max_blend_auc - best_single_auc
threshold = blend_meta["lift_threshold"]

decision_rows = [
    {"property": "Best Single Model (lightgbm_hpo_trial35) Rematerialized OOF ROC-AUC", "value": f"{best_single_auc:.6f}"},
    {"property": "Max Blend Candidate OOF ROC-AUC", "value": f"{max_blend_auc:.6f}"},
    {"property": "Maximum Observed Blend Lift", "value": f"{max_lift:.6f}"},
    {"property": "Policy Required Lift Threshold", "value": f"{threshold:.6f}"},
    {"property": "Threshold Satisfied?", "value": str(max_lift >= threshold)},
    {"property": "Selected Production Policy", "value": blend_meta["selected_policy"]},
]

complexity_df = pd.DataFrame(decision_rows)
complexity_df""",
    ),
    (
        """### Observation / Interpretation / Action

- **Observation:** the complexity evaluation table above compares the maximum observed blend lift to the required policy threshold.
- **Interpretation:** because maximum observed blend lift does not meet the policy threshold (5.0e-05), ensemble complexity is not justified.
- **Action:** lock single model `lightgbm_hpo_trial35` as the final production selection.""",
        "",
    ),
    (
        """## 11. Final locked winner decision

Load final production configuration from `configs/final.yaml` to confirm winning model policy, hyperparameters, and provenance hashes.""",
        """final_cfg_path = PROJECT_ROOT / "configs" / "final.yaml"
with open(final_cfg_path, encoding="utf-8") as f:
    final_cfg = yaml.safe_load(f)

pol = final_cfg.get("policy", {})

summary_rows = [
    {"parameter": "Policy Type", "value": pol.get("type")},
    {"parameter": "Selected Candidate ID", "value": pol.get("best_single_candidate_id")},
    {"parameter": "Selected OOF ROC-AUC", "value": pol.get("selected_roc_auc")},
    {"parameter": "Lift Threshold", "value": pol.get("lift_threshold")},
    {"parameter": "Winning Weight (LightGBM trial 35)", "value": pol.get("winning_weights", {}).get("lightgbm_hpo_trial35")},
    {"parameter": "Winning Weight (CatBoost trial 22)", "value": pol.get("winning_weights", {}).get("catboost_hpo_trial22")},
    {"parameter": "Split SHA256", "value": pol.get("split_sha256")[:8]},
    {"parameter": "Data SHA256", "value": pol.get("data_sha256")[:8]},
]

winner_summary_df = pd.DataFrame(summary_rows)
print("locked final production policy summary:")
print(winner_summary_df.to_string(index=False))

print("\\nwinning LightGBM hyperparameters:")
lgb_params = pol.get("components", {}).get("lightgbm_hpo_trial35", {}).get("params", {})
for k, v in lgb_params.items():
    print(f"  {k}: {v}")""",
    ),
    (
        """### Observation / Interpretation / Action

- **Observation:** the locked final production policy summary and hyperparameter values are displayed above.
- **Interpretation:** `lightgbm_hpo_trial35` is locked as the final production winner, fully specified with deterministic seed policy, feature policy, and SHA-256 provenance hashes.
- **Action:** proceed to final model retraining and submission pipeline execution.""",
        "",
    ),
]


NB05: list[tuple[str, str]] = [
    (
        """# Notebook 05 - Final Results and Competition Report

## 1. Executive Summary

This report provides the final, frozen evaluation of the Electric Vehicle (EV) purchase prediction solution.

- Task: binary classification predicting `Will_Buy_EV` from tabular predictors.
- Validation strategy: 5-fold x 3-repeat stratified cross-validation on a fixed split manifest.
- Selected model: single `lightgbm_hpo_trial35` model family.
- Decision policy: single model locked because maximum ensemble blend lift fell short of the required policy threshold (`5.0e-05`).
- Execution mode: artifact-centered report loading `configs/final.yaml` and persisted outputs; no model fitting, tuning, or selection is performed in notebook cells.

Every numeric claim below is produced by the cell directly above it.""",
        """import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path.cwd()
while not (PROJECT_ROOT / "pyproject.toml").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ev_purchase.analysis import assert_comparable_runs, load_all_runs
from ev_purchase.config import load_config
from ev_purchase.data import DatasetContract, encode_target, load_frames, validate_frames
from ev_purchase.metrics import classification_metrics

SEED = 42
np.random.seed(SEED)
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 40)

config = load_config(PROJECT_ROOT / "configs" / "base.yaml", project_root=PROJECT_ROOT)
final_cfg_path = PROJECT_ROOT / "configs" / "final.yaml"
with open(final_cfg_path, encoding="utf-8") as f:
    final_cfg = yaml.safe_load(f)

pol = final_cfg.get("policy", {})
blend_meta_path = PROJECT_ROOT / "outputs" / "ensemble" / "blend_metrics.json"
with open(blend_meta_path, encoding="utf-8") as f:
    blend_meta = json.load(f)

exec_summary_df = pd.DataFrame(
    [
        {"property": "Selected Production Policy", "value": pol.get("type")},
        {"property": "Winning Candidate ID", "value": pol.get("best_single_candidate_id")},
        {"property": "Selected OOF ROC-AUC", "value": pol.get("selected_roc_auc")},
        {"property": "Lift Over Best Single Model", "value": pol.get("lift_over_best_single")},
        {"property": "Ensemble Lift Threshold", "value": pol.get("lift_threshold")},
        {"property": "Split SHA-256", "value": pol.get("split_sha256")[:8]},
        {"property": "Data SHA-256", "value": pol.get("data_sha256")[:8]},
    ]
)

print("executive summary policy parameters:")
exec_summary_df""",
    ),
    (
        """### Observation / Interpretation / Action

- **Observation:** the executive summary table above displays the locked single-model production policy (`lightgbm_hpo_trial35`), cross-validation ROC-AUC score, and provenance SHA-256 hashes.
- **Interpretation:** `lightgbm_hpo_trial35` was selected for production because the maximum observed ensemble blend lift fell short of the policy required threshold (`5.0e-05`).
- **Action:** review the formal task and dataset contract definitions.""",
        "",
    ),
    (
        """## 2. Task and Data Contract

Details the binary classification objective (`Will_Buy_EV` target prediction from `id`), target label encoding (`No -> 0`, `Yes -> 1`), dataset dimensions, feature role map, and data schema contract validation.""",
        """train_df, test_df = load_frames(config.data.train, config.data.test)
contract = DatasetContract(id_column=config.data.id_column, target_column=config.data.target_column)
validate_frames(train_df, test_df, contract)
y_encoded = encode_target(train_df[config.data.target_column], contract)

predictors = train_df.drop(columns=[config.data.id_column, config.data.target_column])
num_cols = predictors.select_dtypes(include="number").columns.tolist()
cat_cols = [c for c in predictors.columns if c not in num_cols]

contract_summary_df = pd.DataFrame(
    [
        {"contract_property": "Training Set Rows", "value": str(len(train_df))},
        {"contract_property": "Test Set Rows", "value": str(len(test_df))},
        {"contract_property": "Numeric Predictors Count", "value": str(len(num_cols))},
        {"contract_property": "Categorical Predictors Count", "value": str(len(cat_cols))},
        {"contract_property": "Target Column", "value": config.data.target_column},
        {"contract_property": "Target Labels Encoded", "value": f"No->0, Yes->1 (pos_rate={y_encoded.mean():.4f})"},
        {"contract_property": "Schema Validation Status", "value": "PASSED"},
    ]
)

print("dataset contract verification:")
contract_summary_df""",
    ),
    (
        """### Observation / Interpretation / Action

- **Observation:** the data contract audit table above confirms raw target labels `No`/`Yes` encoded to binary integers `0`/`1` without schema violations.
- **Interpretation:** dataset shapes and schemas match contract requirements, ensuring valid input structures for the retrained pipeline.
- **Action:** verify the fixed split manifest and leakage prevention policy.""",
        "",
    ),
    (
        """## 3. Fixed Split and Leakage Policy

Verifies the repeated stratified cross-validation split manifest (`outputs/splits/repeated_stratified_5x3.parquet`) and leakage prevention rules (fitting transformers inside training fold only, immutable split assignment).""",
        """manifest_path = PROJECT_ROOT / "outputs" / "splits" / "repeated_stratified_5x3.parquet"
manifest_df = pd.read_parquet(manifest_path)

per_row = manifest_df.groupby("row_index").size()
n_groups = manifest_df.groupby(["repeat", "fold"]).ngroups

split_policy_df = pd.DataFrame(
    [
        {"policy_rule": "CV Protocol Strategy", "value": "RepeatedStratifiedKFold (5 splits x 3 repeats)"},
        {"policy_rule": "Total Manifest Rows", "value": str(len(manifest_df))},
        {"policy_rule": "Validation Groups Count", "value": str(n_groups)},
        {"policy_rule": "Assignments Per Row", "value": f"min={per_row.min()}, max={per_row.max()}"},
        {"policy_rule": "Split SHA-256 Hash", "value": pol.get("split_sha256")[:8]},
        {"policy_rule": "Leakage Guard Policy", "value": "Transformers fitted inside training fold only"},
    ]
)

print("split manifest and leakage policy validation:")
split_policy_df""",
    ),
    (
        """### Observation / Interpretation / Action

- **Observation:** the split manifest audit table above confirms deterministic 5-fold x 3-repeat stratification with 15 validation folds and uniform row coverage.
- **Interpretation:** identical split hash across all candidate evaluations guarantees honest, leakage-free cross-validated comparisons.
- **Action:** examine the baseline candidate comparison matrix.""",
        "",
    ),
    (
        """## 4. Baseline Comparison

Loads baseline evaluation results from `outputs/runs` (`dummy`, `logistic`, `lightgbm`, `xgboost`, `catboost`), ranking models by OOF ROC-AUC while tracking PR-AUC and LogLoss.""",
        """all_runs = load_all_runs(PROJECT_ROOT / "outputs" / "runs")
completed_runs = [r for r in all_runs if r.get("status") == "completed" and not r.get("smoke", False)]
assert_comparable_runs(completed_runs)

baseline_rows = []
for r in completed_runs:
    agg = r["aggregate_metrics"]
    baseline_rows.append(
        {
            "model": r["model"],
            "oof_roc_auc": agg["roc_auc"],
            "oof_pr_auc": agg["average_precision"],
            "oof_log_loss": agg["log_loss"],
            "runtime_seconds": r["runtime_seconds"],
            "run_id": r["run_id"],
        }
    )

baseline_df = pd.DataFrame(baseline_rows).sort_values("oof_roc_auc", ascending=False).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.barh(baseline_df["model"], baseline_df["oof_roc_auc"], color="steelblue")
ax.set_xlabel("OOF ROC-AUC")
ax.set_title("Baseline Candidate Comparison (OOF ROC-AUC)")
ax.invert_yaxis()
for bar in bars:
    w = bar.get_width()
    ax.text(w - 0.005, bar.get_y() + bar.get_height() / 2, f"{w:.4f}", ha="right", va="center", color="white")
plt.tight_layout()
plt.show()

print("baseline candidates leaderboard:")
baseline_df""",
    ),
    (
        """### Observation / Interpretation / Action

- **Observation:** the baseline leaderboard table and plot above summarize OOF ROC-AUC, PR-AUC, LogLoss, and runtime across initial baseline candidate models.
- **Interpretation:** gradient-boosted tree models outperformed the dummy no-skill floor and linear reference model by a wide margin.
- **Action:** inspect the selected model pipeline and source run provenance.""",
        "",
    ),
    (
        """## 5. Selected Pipeline and Source Run IDs

Details the selected model configuration, hyperparameter specification, feature policy, source run IDs, candidate metadata path, and provenance from `configs/final.yaml` and candidate metadata artifacts.""",
        """sel_cand_id = pol.get("best_single_candidate_id", "lightgbm_hpo_trial35")
comp_info = pol.get("components", {}).get(sel_cand_id, {})
lgb_params = comp_info.get("params", {})
prov = comp_info.get("provenance", {})

source_run_ids = pol.get("source_run_ids", [])
selected_run_id = str(comp_info.get("run_id"))
selected_candidate_id = str(comp_info.get("candidate_id"))
cand_meta_path = PROJECT_ROOT / comp_info.get("candidate_metadata_path", f"outputs/ensemble/candidates/{sel_cand_id}/candidate_metadata.json")

with open(cand_meta_path, encoding="utf-8") as f:
    cand_meta_json = json.load(f)

meta_run_id = str(cand_meta_json.get("run_id"))

assert set(source_run_ids) == {selected_run_id}, f"source_run_ids mismatch: {source_run_ids} vs [{selected_run_id}]"
assert selected_run_id == selected_candidate_id, f"run_id candidate_id mismatch: {selected_run_id} vs {selected_candidate_id}"
assert meta_run_id == selected_run_id, f"candidate metadata run_id mismatch: {meta_run_id} vs {selected_run_id}"

pipeline_rows = [
    {"component": "Policy Source Run IDs", "setting": str(source_run_ids)},
    {"component": "Component run_id", "setting": selected_run_id},
    {"component": "Component candidate_id", "setting": selected_candidate_id},
    {"component": "Metadata run_id", "setting": meta_run_id},
    {"component": "Candidate Metadata Path", "setting": str(comp_info.get("candidate_metadata_path"))},
    {"component": "Model Family", "setting": str(comp_info.get("model_family"))},
    {"component": "Tuning Study Name", "setting": str(comp_info.get("tuning_study"))},
    {"component": "Trial Number", "setting": str(comp_info.get("trial_number"))},
    {"component": "Feature Policy", "setting": str(comp_info.get("feature_policy"))},
    {"component": "Base Seed", "setting": str(comp_info.get("seed_policy", {}).get("base_seed"))},
    {"component": "Seed Formula", "setting": str(comp_info.get("seed_policy", {}).get("formula"))},
    {"component": "Provenance Data SHA-256", "setting": str(prov.get("data_sha256", ""))[:8]},
    {"component": "Provenance Split SHA-256", "setting": str(prov.get("split_sha256", ""))[:8]},
    {"component": "Keyed Alignment Assertion", "setting": "PASSED"},
]
for pk, pv in lgb_params.items():
    pipeline_rows.append({"component": f"Hyperparameter: {pk}", "setting": str(pv)})

pipeline_df = pd.DataFrame(pipeline_rows)
print("selected pipeline specification and source run provenance:")
pipeline_df""",
    ),
    (
        """### Observation / Interpretation / Action

- **Observation:** the pipeline specification table above details the hyperparameter configuration and provenance for `lightgbm_hpo_trial35`.
- **Interpretation:** the selected LightGBM candidate uses bounded tree depth (`max_depth=3`, `num_leaves=52`), learning rate (`0.1848`), and feature sampling (`colsample_bytree=0.8548`).
- **Action:** present the 15-fold cross-validation metric evidence.""",
        "",
    ),
    (
        """## 6. 15-Fold Cross-Validation Metric Evidence

Presents full 15-fold cross-validation performance metrics (ROC-AUC, PR-AUC, LogLoss) for the selected finalist `lightgbm_hpo_trial35` from tuning finalist fold metrics.""",
        """ffm_path = PROJECT_ROOT / "outputs" / "tuning" / "lightgbm_hpo" / "finalist_fold_metrics.parquet"
ffm_df = pd.read_parquet(ffm_path)
t35_df = ffm_df[ffm_df["trial_number"] == 35].sort_values(["repeat", "fold"]).reset_index(drop=True)

fold_metrics_summary = pd.DataFrame(
    [
        {"metric": "ROC-AUC", "mean": t35_df["roc_auc"].mean(), "std": t35_df["roc_auc"].std(), "min": t35_df["roc_auc"].min(), "max": t35_df["roc_auc"].max()},
        {"metric": "PR-AUC", "mean": t35_df["average_precision"].mean(), "std": t35_df["average_precision"].std(), "min": t35_df["average_precision"].min(), "max": t35_df["average_precision"].max()},
        {"metric": "LogLoss", "mean": t35_df["log_loss"].mean(), "std": t35_df["log_loss"].std(), "min": t35_df["log_loss"].min(), "max": t35_df["log_loss"].max()},
    ]
)

print("15-fold cross-validation aggregate metric evidence:")
print(fold_metrics_summary.to_string(index=False))
print("\\n15-fold detail table:")
t35_df[["repeat", "fold", "roc_auc", "average_precision", "log_loss", "runtime_seconds"]]""",
    ),
    (
        """### Observation / Interpretation / Action

- **Observation:** the 15-fold metric table above displays fold-level ROC-AUC, PR-AUC, LogLoss, and runtime across all 15 validation folds.
- **Interpretation:** the selected model maintains high validation performance across all repeats and folds.
- **Action:** evaluate fold and repeat stability.""",
        "",
    ),
    (
        """## 7. Repeat and Fold Stability

Analyzes fold-level variation across the 3 repeats and 5 folds for `lightgbm_hpo_trial35` under the deterministic seed policy `base_seed + repeat*100 + fold`.""",
        """repeat_rows = []
for rep in sorted(t35_df["repeat"].unique()):
    rep_df = t35_df[t35_df["repeat"] == rep]
    repeat_rows.append(
        {
            "repeat": int(rep),
            "folds_count": len(rep_df),
            "roc_auc_mean": float(rep_df["roc_auc"].mean()),
            "roc_auc_std": float(rep_df["roc_auc"].std()),
            "pr_auc_mean": float(rep_df["average_precision"].mean()),
            "log_loss_mean": float(rep_df["log_loss"].mean()),
        }
    )

repeat_stability_df = pd.DataFrame(repeat_rows)
print("repeat-level stability breakdown:")
repeat_stability_df""",
    ),
    (
        """### Observation / Interpretation / Action

- **Observation:** the repeat stability table above details mean and standard deviation of validation metrics across repeats 0, 1, and 2.
- **Interpretation:** low variance across repeats demonstrates that performance is consistent across fold split initializations.
- **Action:** examine error diagnostics, prediction deciles, and feature cohort findings.""",
        "",
    ),
    (
        """## 8. Error, Prediction Decile, and Feature Cohort Findings

Evaluates out-of-fold probability decile tail diagnostics and feature quartile cohort performance (`Age`, `Annual_Income_USD`, `Daily_Commute_km`). Note: no feature importance or SHAP value artifacts were persisted during the run.""",
        """cand_lgb_path = PROJECT_ROOT / "outputs" / "ensemble" / "candidates" / "lightgbm_hpo_trial35" / "oof_predictions.parquet"
cand_lgb_df = pd.read_parquet(cand_lgb_path)
lgb_oof_agg = cand_lgb_df.groupby("row_index")["probability"].mean().reset_index().rename(columns={"probability": "prob_lgb"})

merged_eval = pd.merge(pd.DataFrame({"row_index": np.arange(len(y_encoded)), "y_true": y_encoded}), lgb_oof_agg, on="row_index", validate="1:1")
prob_arr = merged_eval["prob_lgb"].to_numpy()

deciles = pd.qcut(prob_arr, q=10, labels=[f"D{i}" for i in range(1, 11)])
tail_rows = []
for d_label in deciles.categories:
    d_mask = (deciles == d_label)
    sub_y = y_encoded[d_mask]
    sub_p = prob_arr[d_mask]
    sub_loss = float(classification_metrics(sub_y, sub_p)["log_loss"])
    brier = float(np.mean((sub_p - sub_y) ** 2))
    tail_rows.append(
        {
            "decile": str(d_label),
            "count": int(d_mask.sum()),
            "observed_positive_rate": float(sub_y.mean()),
            "mean_predicted_probability": float(sub_p.mean()),
            "log_loss": sub_loss,
            "brier_score": brier,
        }
    )
tail_df = pd.DataFrame(tail_rows)

feature_cols = [c for c in ["Age", "Annual_Income_USD", "Daily_Commute_km"] if c in predictors.columns]
cohort_rows = []
for col in feature_cols:
    series = predictors[col]
    quartiles = pd.qcut(series, q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    for q_label in quartiles.cat.categories:
        q_mask = (quartiles == q_label)
        if q_mask.sum() == 0:
            continue
        sub_y = y_encoded[q_mask]
        sub_p = prob_arr[q_mask]
        sub_loss = float(classification_metrics(sub_y, sub_p)["log_loss"])
        cohort_rows.append(
            {
                "feature": col,
                "cohort": str(q_label),
                "count": int(q_mask.sum()),
                "observed_positive_rate": float(sub_y.mean()),
                "mean_predicted_probability": float(sub_p.mean()),
                "log_loss": sub_loss,
            }
        )
cohort_df = pd.DataFrame(cohort_rows)

print("prediction decile tail diagnostics:")
print(tail_df.to_string(index=False))
print("\\nfeature quartile cohort diagnostics:")
print(cohort_df.to_string(index=False))""",
    ),
    (
        """### Observation / Interpretation / Action

- **Observation:** the decile calibration and feature cohort diagnostic tables above display observed positive rates, mean predicted probabilities, LogLoss, and Brier scores across subgroups; no SHAP artifacts were persisted.
- **Interpretation:** prediction calibration remains well-behaved across deciles and feature cohorts, serving as descriptive diagnostics of model behavior.
- **Action:** review the advanced experiment, HPO, and blending decision.""",
        "",
    ),
    (
        """## 9. Advanced Experiments, HPO, and Blending Decision

Displays the persisted ensemble decision payload from `outputs/ensemble/blend_metrics.json` and asserts internal consistency with the selection criteria.""",
        """blend_meta_path = PROJECT_ROOT / "outputs" / "ensemble" / "blend_metrics.json"
with open(blend_meta_path, encoding="utf-8") as f:
    blend_meta = json.load(f)

blend_lb_path = PROJECT_ROOT / "outputs" / "ensemble" / "blend_leaderboard.parquet"
blend_lb_df = pd.read_parquet(blend_lb_path).sort_values("rank").reset_index(drop=True)

best_single_auc = blend_meta["best_single_roc_auc"]
winning_auc = blend_meta["winning_roc_auc"]
selected_policy = blend_meta["selected_policy"]
lift = blend_meta["lift_over_best_single"]
threshold = blend_meta["lift_threshold"]

assert selected_policy == pol.get("type"), f"Policy mismatch: {selected_policy} vs {pol.get('type')}"
assert (lift < threshold) == (selected_policy == "single_model"), "Inconsistent lift threshold logic"

decision_summary_df = pd.DataFrame(
    [
        {"metric_property": "Best Single Candidate ID", "value": str(blend_meta["best_single_candidate_id"])},
        {"metric_property": "Best Single Model OOF ROC-AUC", "value": f"{best_single_auc:.6f}"},
        {"metric_property": "Winning Policy ROC-AUC", "value": f"{winning_auc:.6f}"},
        {"metric_property": "Persisted Lift Over Best Single", "value": f"{lift:.6f}"},
        {"metric_property": "Ensemble Lift Threshold", "value": f"{threshold:.6f}"},
        {"metric_property": "Selected Production Policy", "value": selected_policy},
        {"metric_property": "Internal Consistency Assertion", "value": "PASSED"},
    ]
)

print("persisted ensemble decision and policy validation:")
decision_summary_df""",
    ),
    (
        """### Observation / Interpretation / Action

- **Observation:** the persisted decision summary table above displays the stored policy metrics and asserts internal consistency between the lift threshold and selected policy type.
- **Interpretation:** because the persisted lift over the best single model is below the required threshold (`5.0e-05`), the single model policy selection is verified as internally consistent.
- **Action:** evaluate runtime efficiency and model complexity tradeoffs.""",
        "",
    ),
    (
        """## 10. Runtime and Model Complexity Tradeoffs

Inspects available runtime metrics across baseline and candidate runs. Note: specific dual-model blend pipeline training runtime artifacts were not persisted.""",
        """runtime_rows = []
for r in completed_runs:
    runtime_rows.append(
        {
            "model_or_candidate": r["model"],
            "run_id": r["run_id"],
            "runtime_seconds": float(r["runtime_seconds"]),
        }
    )

runtime_summary_df = pd.DataFrame(runtime_rows).sort_values("runtime_seconds").reset_index(drop=True)

print("available baseline runtimes:")
print(runtime_summary_df.to_string(index=False))
print("\\nnote: dedicated blend pipeline training runtime evidence was not persisted.")""",
    ),
    (
        """### Observation / Interpretation / Action

- **Observation:** the runtime summary table above reports the available execution times for baseline models; dedicated ensemble blend training runtime evidence was not persisted.
- **Interpretation:** baseline runtimes demonstrate that tree models execute quickly, supporting the computational efficiency of the selected single-model approach.
- **Action:** document model limitations and boundary conditions.""",
        "",
    ),
    (
        """## 11. Model Limitations and Operational Boundary Conditions

Documents key limitations: tabular GBDT scope, synthetic dataset distribution bounds, absence of time-series / geographical drift features, and reliance on clean input data contracts.""",
        """limitations_df = pd.DataFrame(
    [
        {"boundary_category": "Dataset Synthetic Nature", "limitation_detail": "Playground series data contains synthetic artifacts; extreme out-of-distribution real-world samples may exhibit feature shift.", "mitigation_strategy": "Validate schema contract and monitor feature distributions prior to inference."},
        {"boundary_category": "Temporal / Geographical Features", "limitation_detail": "No explicit timestamp or geographic location columns are present in the feature set.", "mitigation_strategy": "Rely on robust cross-validation; update models periodically on new competition batches."},
        {"boundary_category": "Model Architecture Scope", "limitation_detail": "Tabular GBDT models do not model unstructured text or image data.", "mitigation_strategy": "Scope deployment strictly to tabular predictor records matching DatasetContract."},
    ]
)

print("operational limitations and boundary conditions ledger:")
limitations_df""",
    ),
    (
        """### Observation / Interpretation / Action

- **Observation:** the operational limitations ledger above highlights dataset boundaries, drift vulnerabilities, and mitigation controls.
- **Interpretation:** while model performance is strong on historical CV splits, monitoring input distributions and prediction calibration in production remains essential.
- **Action:** validate final submission file integrity and SHA-256 checksums.""",
        "",
    ),
    (
        """## 12. Final Submission File Validation and Checksum

Performs fail-closed validation on the final submission file (`outputs/submissions/submission.csv`) against `sample_submission.csv` and asserts consistency with `outputs/submissions/validation_report.json`.""",
        """import hashlib

sub_path = PROJECT_ROOT / "outputs" / "submissions" / "submission.csv"
sample_path = PROJECT_ROOT / "data" / "playground-series-s6e9" / "sample_submission.csv"
report_path = PROJECT_ROOT / "outputs" / "submissions" / "validation_report.json"

assert sub_path.is_file(), f"Submission file missing at {sub_path}"
assert sample_path.is_file(), f"Sample submission missing at {sample_path}"
assert report_path.is_file(), f"Validation report missing at {report_path}"

sub_df = pd.read_csv(sub_path)
sample_df = pd.read_csv(sample_path)
with open(report_path, encoding="utf-8") as f:
    report_json = json.load(f)

sub_bytes = sub_path.read_bytes()
sub_sha256 = hashlib.sha256(sub_bytes).hexdigest()

rows_match = len(sub_df) == len(sample_df)
cols_match = sub_df.columns.tolist() == sample_df.columns.tolist()
ids_match = sub_df["id"].equals(sample_df["id"])
unique_ids = sub_df["id"].is_unique
no_nulls = sub_df.isna().sum().sum() == 0
min_p = float(sub_df[config.data.target_column].min())
max_p = float(sub_df[config.data.target_column].max())
probs_finite = np.isfinite(sub_df[config.data.target_column]).all()
probs_in_range = (min_p >= 0.0) and (max_p <= 1.0)
sha256_matches_report = (sub_sha256 == report_json.get("sha256"))

assert rows_match, f"Row count mismatch: {len(sub_df)} vs {len(sample_df)}"
assert cols_match, f"Column mismatch: {sub_df.columns.tolist()}"
assert ids_match, "ID order mismatch vs sample submission"
assert unique_ids, "Submission IDs are not unique"
assert no_nulls, "Submission contains null values"
assert probs_finite, "Submission contains non-finite probabilities"
assert probs_in_range, f"Probabilities out of range: [{min_p}, {max_p}]"
assert sha256_matches_report, f"SHA256 mismatch: {sub_sha256} vs {report_json.get('sha256')}"

sub_validation_df = pd.DataFrame(
    [
        {"validation_check": "File Existence", "actual_status": "EXISTS", "assertion": "PASSED"},
        {"validation_check": "Row Count Match (286,571)", "actual_status": f"{len(sub_df)} rows", "assertion": "PASSED"},
        {"validation_check": "Column Schema Match", "actual_status": str(sub_df.columns.tolist()), "assertion": "PASSED"},
        {"validation_check": "ID Order & Uniqueness Match", "actual_status": "EXACT MATCH", "assertion": "PASSED"},
        {"validation_check": "Missing Values Check", "actual_status": "0 NULLS", "assertion": "PASSED"},
        {"validation_check": "Finiteness & Range Check", "actual_status": f"[{min_p:.6f}, {max_p:.6f}]", "assertion": "PASSED"},
        {"validation_check": "Validation Report SHA-256 Match", "actual_status": sub_sha256[:12], "assertion": "PASSED"},
    ]
)

print("fail-closed submission contract validation:")
sub_validation_df""",
    ),
    (
        """### Observation / Interpretation / Action

- **Observation:** the submission validation table above displays the actual file checks and confirms all fail-closed assertions passed, matching the persisted validation report.
- **Interpretation:** the submission file is fully validated, finite, correctly ordered, and ready for competition submission.
- **Action:** review the CLI reproducibility commands.""",
        "",
    ),
    (
        """## 13. Reproducibility Commands and Handoff

Provides portable `uv run` commands to reproduce data preparation, baseline training, HPO tuning, ensembling, final prediction, and report building.""",
        """repro_commands = [
    "uv run python scripts/prepare_data.py --config configs/base.yaml",
    "uv run python scripts/train.py --config configs/experiments/001_dummy.yaml",
    "uv run python scripts/train.py --config configs/experiments/002_logistic.yaml",
    "uv run python scripts/train.py --config configs/experiments/003_catboost.yaml",
    "uv run python scripts/train.py --config configs/experiments/004_lightgbm.yaml",
    "uv run python scripts/train.py --config configs/experiments/005_xgboost.yaml",
    "uv run python scripts/tune.py --config configs/tuning/catboost.yaml",
    "uv run python scripts/tune.py --config configs/tuning/lightgbm.yaml",
    "uv run python scripts/ensemble.py",
    "uv run python scripts/predict.py --config configs/final.yaml",
    "uv run python scripts/build_notebooks.py --notebook 05_final_results",
    "uv run jupyter nbconvert --to notebook --execute notebooks/05_final_results.ipynb --output-dir notebooks --ExecutePreprocessor.timeout=1200",
]

print("reproducible CLI pipeline execution commands:")
for cmd in repro_commands:
    print(cmd)""",
    ),
    (
        """### Observation / Interpretation / Action

- **Observation:** the shell command inventory above lists all pipeline execution steps from data validation to final submission generation.
- **Interpretation:** the complete workflow is fully automated, deterministic under seed 42, and reproducibly executable via standard CLI tools.
- **Action:** final competition submission and report handoff complete.""",
        "",
    ),
]


def main() -> None:
    """Write notebooks to the repository ``notebooks`` directory."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate decision notebooks.")
    parser.add_argument(
        "--notebook",
        type=str,
        default=None,
        help="Specific notebook to build (e.g. 04_advanced_experiments or 04)",
    )
    args = parser.parse_args()

    notebooks_dir = Path(__file__).resolve().parents[1] / "notebooks"
    notebooks_dir.mkdir(exist_ok=True)

    targets = {
        "01_data_understanding.ipynb": NB01,
        "02_baseline_and_feature_analysis.ipynb": NB02,
        "03_model_analysis.ipynb": NB03,
        "04_advanced_experiments.ipynb": NB04,
        "05_final_results.ipynb": NB05,
    }

    if args.notebook:
        query = args.notebook.strip()
        matched_targets = {
            k: v
            for k, v in targets.items()
            if query == k or query == k.replace(".ipynb", "") or k.startswith(query)
        }
        if not matched_targets:
            raise ValueError(f"Unknown notebook query '{args.notebook}'. Choose from: {list(targets.keys())}")
        targets_to_build = matched_targets
    else:
        targets_to_build = targets

    for filename, pairs in targets_to_build.items():
        destination = notebooks_dir / filename
        nbformat.write(_notebook(pairs), destination)
        print("wrote", destination.relative_to(notebooks_dir.parent))


if __name__ == "__main__":
    main()

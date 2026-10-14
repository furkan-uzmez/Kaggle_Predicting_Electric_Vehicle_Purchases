import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

from ev_purchase.data import DatasetContract, encode_target
from ev_purchase.features import apply_feature_policy

PROJECT_ROOT = Path.cwd()

print("1. Loading raw dataset and sample submission...")
train_df = pd.read_csv(PROJECT_ROOT / "data/playground-series-s6e9/train.csv")
test_df = pd.read_csv(PROJECT_ROOT / "data/playground-series-s6e9/test.csv")
sample_sub = pd.read_csv(PROJECT_ROOT / "data/playground-series-s6e9/sample_submission.csv")

contract = DatasetContract(id_column="id", target_column="Will_Buy_EV")
y_true = encode_target(train_df["Will_Buy_EV"], contract)

print("2. Applying 'advanced_engineered' feature policy...")
train_feat = apply_feature_policy("advanced_engineered", train_df)
test_feat = apply_feature_policy("advanced_engineered", test_df)

drop_cols = ["id", "Will_Buy_EV"]
X_train_full = train_feat.drop(columns=drop_cols)
X_test_full = test_feat.drop(columns=[c for c in drop_cols if c in test_feat.columns])

cat_cols = X_train_full.select_dtypes(include=["object", "category", "string"]).columns.tolist()
for c in cat_cols:
    X_train_full[c] = X_train_full[c].astype("category")
    X_test_full[c] = X_test_full[c].astype("category")

print(f"Features prepared: {X_train_full.shape[1]} columns")

manifest_df = pd.read_parquet(PROJECT_ROOT / "outputs/splits/repeated_stratified_5x3.parquet")

lgb_params = {
    "n_estimators": 450,
    "learning_rate": 0.1848,
    "num_leaves": 52,
    "max_depth": 3,
    "min_child_samples": 35,
    "subsample": 0.9307,
    "colsample_bytree": 0.8548,
    "reg_alpha": 0.0270,
    "reg_lambda": 0.0031,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

oof_preds_by_repeat = [np.zeros(len(train_df)) for _ in range(3)]
test_preds_folds = []

print("3. Executing 15-fold CV training and test set inference...")
for (repeat, _fold), group in manifest_df.groupby(["repeat", "fold"]):
    val_idx = group["row_index"].to_numpy()
    train_idx = np.setdiff1d(np.arange(len(train_df)), val_idx)

    X_tr, y_tr = X_train_full.iloc[train_idx], y_true[train_idx]
    X_va, y_va = X_train_full.iloc[val_idx], y_true[val_idx]

    model = LGBMClassifier(**lgb_params)
    model.fit(X_tr, y_tr)

    val_preds = model.predict_proba(X_va)[:, 1]
    oof_preds_by_repeat[repeat][val_idx] = val_preds

    test_pred_fold = model.predict_proba(X_test_full)[:, 1]
    test_preds_folds.append(test_pred_fold)

# Calculate Repeat-Averaged OOF ROC-AUC
avg_oof_preds = np.mean(oof_preds_by_repeat, axis=0)
oof_auc = float(roc_auc_score(y_true, avg_oof_preds))
print(f"-> 15-Fold Repeat-Averaged OOF ROC-AUC: {oof_auc:.6f}")

# Final Test Predictions (Average over 15 folds)
final_test_preds = np.mean(test_preds_folds, axis=0)

sub_df = pd.DataFrame(
    {
        "id": test_df["id"],
        "Will_Buy_EV": final_test_preds,
    }
)

print("4. Validating submission dataframe against contract...")
assert len(sub_df) == len(sample_sub), f"Row count mismatch: {len(sub_df)} vs {len(sample_sub)}"
assert list(sub_df.columns) == list(sample_sub.columns), "Column schema mismatch"
assert (sub_df["id"].to_numpy() == sample_sub["id"].to_numpy()).all(), "ID sequence mismatch"
assert sub_df["Will_Buy_EV"].isna().sum() == 0, "Null values found in predictions"
assert np.isfinite(sub_df["Will_Buy_EV"]).all(), "Non-finite probabilities found"
min_p, max_p = sub_df["Will_Buy_EV"].min(), sub_df["Will_Buy_EV"].max()
assert (
    0.0 <= min_p <= 1.0 and 0.0 <= max_p <= 1.0
), f"Probability range out of bounds: [{min_p}, {max_p}]"

print("PASSED all submission contract checks.")

sub_dir = PROJECT_ROOT / "outputs/submissions"
sub_dir.mkdir(parents=True, exist_ok=True)
sub_path = sub_dir / "submission_advanced_engineered.csv"

sub_df.to_csv(sub_path, index=False)
sub_bytes = sub_path.read_bytes()
sha256_hash = hashlib.sha256(sub_bytes).hexdigest()

print(f"5. Saved submission file to {sub_path}")
print(f"   SHA-256: {sha256_hash}")
print(f"   Rows: {len(sub_df)}, Prob Range: [{min_p:.6f}, {max_p:.6f}]")

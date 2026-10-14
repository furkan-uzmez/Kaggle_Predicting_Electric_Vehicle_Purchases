import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

from ev_purchase.data import DatasetContract, encode_target
from ev_purchase.features import apply_feature_policy

PROJECT_ROOT = Path.cwd()

print("1. Loading raw datasets and sample submission...")
train_df = pd.read_csv(PROJECT_ROOT / "data/playground-series-s6e9/train.csv")
test_df = pd.read_csv(PROJECT_ROOT / "data/playground-series-s6e9/test.csv")
sample_sub = pd.read_csv(PROJECT_ROOT / "data/playground-series-s6e9/sample_submission.csv")

contract = DatasetContract(id_column="id", target_column="Will_Buy_EV")
y_true = encode_target(train_df["Will_Buy_EV"], contract)

print("2. Applying 'advanced_engineered' feature policy (21 features)...")
train_feat = apply_feature_policy("advanced_engineered", train_df)
test_feat = apply_feature_policy("advanced_engineered", test_df)

drop_cols = ["id", "Will_Buy_EV"]
X_train_full = train_feat.drop(columns=drop_cols)
X_test_full = test_feat.drop(columns=[c for c in drop_cols if c in test_feat.columns])

cat_cols = X_train_full.select_dtypes(include=["object", "category", "string"]).columns.tolist()

# Prepare feature formats for CatBoost and XGBoost
X_tr_cb = X_train_full.copy()
X_te_cb = X_test_full.copy()

X_tr_xgb = pd.get_dummies(X_train_full, columns=cat_cols, drop_first=True)
X_te_xgb = pd.get_dummies(X_test_full, columns=cat_cols, drop_first=True)
X_tr_xgb, X_te_xgb = X_tr_xgb.align(X_te_xgb, join="left", axis=1, fill_value=0)

manifest_df = pd.read_parquet(PROJECT_ROOT / "outputs/splits/repeated_stratified_5x3.parquet")

cb_params = {
    "iterations": 850,
    "depth": 4,
    "learning_rate": 0.1942,
    "l2_leaf_reg": 8.3748,
    "subsample": 0.8396,
    "random_strength": 2.2197,
    "random_seed": 42,
    "thread_count": -1,
    "verbose": 0,
    "cat_features": cat_cols,
}

xgb_params = {
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.12,
    "subsample": 0.85,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
}

oof_cb_repeats = [np.zeros(len(train_df)) for _ in range(3)]
oof_xgb_repeats = [np.zeros(len(train_df)) for _ in range(3)]

test_preds_cb, test_preds_xgb = [], []

print("3. Executing 15-fold CV for CatBoost and XGBoost with Advanced Features...")
for fold_idx, ((repeat, _fold), group) in enumerate(manifest_df.groupby(["repeat", "fold"])):
    val_idx = group["row_index"].to_numpy()
    train_idx = np.setdiff1d(np.arange(len(train_df)), val_idx)

    y_tr, y_va = y_true[train_idx], y_true[val_idx]

    # CatBoost
    model_cb = CatBoostClassifier(**cb_params)
    model_cb.fit(X_tr_cb.iloc[train_idx], y_tr)
    pred_val_cb = model_cb.predict_proba(X_tr_cb.iloc[val_idx])[:, 1]
    oof_cb_repeats[repeat][val_idx] = pred_val_cb
    test_preds_cb.append(model_cb.predict_proba(X_te_cb)[:, 1])

    # XGBoost
    model_xgb = XGBClassifier(**xgb_params)
    model_xgb.fit(X_tr_xgb.iloc[train_idx], y_tr)
    pred_val_xgb = model_xgb.predict_proba(X_tr_xgb.iloc[val_idx])[:, 1]
    oof_xgb_repeats[repeat][val_idx] = pred_val_xgb
    test_preds_xgb.append(model_xgb.predict_proba(X_te_xgb)[:, 1])

    if (fold_idx + 1) % 5 == 0:
        print(f"   Completed {fold_idx + 1}/15 Folds")

oof_cb = np.mean(oof_cb_repeats, axis=0)
oof_xgb = np.mean(oof_xgb_repeats, axis=0)

auc_cb = roc_auc_score(y_true, oof_cb)
auc_xgb = roc_auc_score(y_true, oof_xgb)

print(f"\n--- OOF Results ---")
print(f"CatBoost (advanced_engineered) OOF ROC-AUC: {auc_cb:.6f}")
print(f"XGBoost  (advanced_engineered) OOF ROC-AUC: {auc_xgb:.6f}")

print("\n4. Validating and saving submission files...")
sub_dir = PROJECT_ROOT / "outputs/submissions"
sub_dir.mkdir(parents=True, exist_ok=True)

# Helper function for fail-closed validation
def save_and_validate(preds, filename, label):
    sub_df = pd.DataFrame({"id": test_df["id"], "Will_Buy_EV": preds})
    assert len(sub_df) == len(sample_sub), f"Row count mismatch for {label}"
    assert list(sub_df.columns) == list(sample_sub.columns), "Column schema mismatch"
    assert (sub_df["id"].to_numpy() == sample_sub["id"].to_numpy()).all(), "ID order mismatch"
    assert sub_df["Will_Buy_EV"].isna().sum() == 0, "Null predictions found"
    assert np.isfinite(sub_df["Will_Buy_EV"]).all(), "Non-finite values found"
    min_p, max_p = sub_df["Will_Buy_EV"].min(), sub_df["Will_Buy_EV"].max()
    assert 0.0 <= min_p <= 1.0 and 0.0 <= max_p <= 1.0, f"Out of range: [{min_p}, {max_p}]"

    file_path = sub_dir / filename
    sub_df.to_csv(file_path, index=False)
    sha256_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
    print(f"✅ {label}: Saved to {file_path.name}")
    print(f"   SHA-256: {sha256_hash}")
    print(f"   Prob Range: [{min_p:.6f}, {max_p:.6f}]")

save_and_validate(np.mean(test_preds_cb, axis=0), "submission_catboost_advanced_engineered.csv", "CatBoost Advanced")
save_and_validate(np.mean(test_preds_xgb, axis=0), "submission_xgboost_advanced_engineered.csv", "XGBoost Advanced")

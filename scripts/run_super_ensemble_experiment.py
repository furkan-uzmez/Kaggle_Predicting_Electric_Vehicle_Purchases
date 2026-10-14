import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from scipy.optimize import minimize
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

print("2. Applying 'advanced_v2' feature policy (24 features)...")
train_feat = apply_feature_policy("advanced_v2", train_df)
test_feat = apply_feature_policy("advanced_v2", test_df)

drop_cols = ["id", "Will_Buy_EV"]
X_train_full = train_feat.drop(columns=drop_cols)
X_test_full = test_feat.drop(columns=[c for c in drop_cols if c in test_feat.columns])

cat_cols = X_train_full.select_dtypes(include=["object", "category", "string"]).columns.tolist()
print(f"Categorical columns identified: {cat_cols}")

# Prepare pandas category types for LightGBM/CatBoost and string/category for XGBoost
X_tr_lgb = X_train_full.copy()
X_te_lgb = X_test_full.copy()
for c in cat_cols:
    X_tr_lgb[c] = X_tr_lgb[c].astype("category")
    X_te_lgb[c] = X_te_lgb[c].astype("category")

X_tr_xgb = pd.get_dummies(X_train_full, columns=cat_cols, drop_first=True)
X_te_xgb = pd.get_dummies(X_test_full, columns=cat_cols, drop_first=True)
X_tr_xgb, X_te_xgb = X_tr_xgb.align(X_te_xgb, join="left", axis=1, fill_value=0)

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

cb_params = {
    "iterations": 600,
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

print(f"Features prepared: LGB/CB={X_tr_lgb.shape[1]}, XGB={X_tr_xgb.shape[1]}")

oof_lgb_repeats = [np.zeros(len(train_df)) for _ in range(3)]
oof_cb_repeats = [np.zeros(len(train_df)) for _ in range(3)]
oof_xgb_repeats = [np.zeros(len(train_df)) for _ in range(3)]

test_preds_lgb, test_preds_cb, test_preds_xgb = [], [], []

print("\n3. Executing 15-fold CV training for LightGBM, CatBoost, and XGBoost...")
for (repeat, fold), group in manifest_df.groupby(["repeat", "fold"]):
    val_idx = group["row_index"].to_numpy()
    train_idx = np.setdiff1d(np.arange(len(train_df)), val_idx)

    y_tr, y_va = y_true[train_idx], y_true[val_idx]

    # LightGBM
    model_lgb = LGBMClassifier(**lgb_params)
    model_lgb.fit(X_tr_lgb.iloc[train_idx], y_tr)
    pred_val_lgb = model_lgb.predict_proba(X_tr_lgb.iloc[val_idx])[:, 1]
    oof_lgb_repeats[repeat][val_idx] = pred_val_lgb
    test_preds_lgb.append(model_lgb.predict_proba(X_te_lgb)[:, 1])

    # CatBoost
    model_cb = CatBoostClassifier(**cb_params)
    model_cb.fit(X_train_full.iloc[train_idx], y_tr)
    pred_val_cb = model_cb.predict_proba(X_train_full.iloc[val_idx])[:, 1]
    oof_cb_repeats[repeat][val_idx] = pred_val_cb
    test_preds_cb.append(model_cb.predict_proba(X_test_full)[:, 1])

    # XGBoost
    model_xgb = XGBClassifier(**xgb_params)
    model_xgb.fit(X_tr_xgb.iloc[train_idx], y_tr)
    pred_val_xgb = model_xgb.predict_proba(X_tr_xgb.iloc[val_idx])[:, 1]
    oof_xgb_repeats[repeat][val_idx] = pred_val_xgb
    test_preds_xgb.append(model_xgb.predict_proba(X_te_xgb)[:, 1])

oof_lgb = np.mean(oof_lgb_repeats, axis=0)
oof_cb = np.mean(oof_cb_repeats, axis=0)
oof_xgb = np.mean(oof_xgb_repeats, axis=0)

auc_lgb = roc_auc_score(y_true, oof_lgb)
auc_cb = roc_auc_score(y_true, oof_cb)
auc_xgb = roc_auc_score(y_true, oof_xgb)

print(f"\n--- OOF Individual Model Performance (Repeat-Averaged ROC-AUC) ---")
print(f"LightGBM (advanced_v2): {auc_lgb:.6f}")
print(f"CatBoost (advanced_v2): {auc_cb:.6f}")
print(f"XGBoost  (advanced_v2): {auc_xgb:.6f}")

print("\n4. Optimizing Super Ensemble weights (SLSQP)...")
def loss_func(weights):
    w_lgb, w_cb, w_xgb = weights
    blend = w_lgb * oof_lgb + w_cb * oof_cb + w_xgb * oof_xgb
    return -roc_auc_score(y_true, blend)

init_weights = [0.4, 0.4, 0.2]
bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
constraints = {"type": "eq", "fun": lambda w: sum(w) - 1.0}

opt_res = minimize(loss_func, init_weights, method="SLSQP", bounds=bounds, constraints=constraints)
w_lgb_opt, w_cb_opt, w_xgb_opt = opt_res.x
super_oof_auc = -opt_res.fun

print(f"Optimal Weights -> LightGBM: {w_lgb_opt:.4f}, CatBoost: {w_cb_opt:.4f}, XGBoost: {w_xgb_opt:.4f}")
print(f"--> Super Ensemble Repeat-Averaged OOF ROC-AUC: {super_oof_auc:.6f}")

print("\n5. Generating final Super Ensemble test predictions...")
test_lgb = np.mean(test_preds_lgb, axis=0)
test_cb = np.mean(test_preds_cb, axis=0)
test_xgb = np.mean(test_preds_xgb, axis=0)

final_super_preds = w_lgb_opt * test_lgb + w_cb_opt * test_cb + w_xgb_opt * test_xgb

sub_df = pd.DataFrame({"id": test_df["id"], "Will_Buy_EV": final_super_preds})

print("6. Validating submission dataframe against contract...")
assert len(sub_df) == len(sample_sub), f"Row count mismatch: {len(sub_df)} vs {len(sample_sub)}"
assert list(sub_df.columns) == list(sample_sub.columns), "Column schema mismatch"
assert (sub_df["id"].to_numpy() == sample_sub["id"].to_numpy()).all(), "ID sequence mismatch"
assert sub_df["Will_Buy_EV"].isna().sum() == 0, "Null values found in predictions"
assert np.isfinite(sub_df["Will_Buy_EV"]).all(), "Non-finite probabilities found"
min_p, max_p = sub_df["Will_Buy_EV"].min(), sub_df["Will_Buy_EV"].max()
assert 0.0 <= min_p <= 1.0 and 0.0 <= max_p <= 1.0, f"Probability range out of bounds: [{min_p}, {max_p}]"

print("PASSED all submission contract checks.")

sub_dir = PROJECT_ROOT / "outputs/submissions"
sub_dir.mkdir(parents=True, exist_ok=True)
sub_path = sub_dir / "submission_super_ensemble.csv"

sub_df.to_csv(sub_path, index=False)
sub_bytes = sub_path.read_bytes()
sha256_hash = hashlib.sha256(sub_bytes).hexdigest()

print(f"\n7. Saved Super Ensemble submission file to {sub_path}")
print(f"   SHA-256: {sha256_hash}")
print(f"   Rows: {len(sub_df)}, Prob Range: [{min_p:.6f}, {max_p:.6f}]")

# Save diagnostic summary
diag_payload = {
    "super_oof_auc": float(super_oof_auc),
    "optimal_weights": {
        "lightgbm": float(w_lgb_opt),
        "catboost": float(w_cb_opt),
        "xgboost": float(w_xgb_opt),
    },
    "individual_oof_auc": {
        "lightgbm": float(auc_lgb),
        "catboost": float(auc_cb),
        "xgboost": float(auc_xgb),
    },
    "sha256": sha256_hash,
}

diag_path = PROJECT_ROOT / "outputs/diagnostics/super_ensemble_metrics.json"
with open(diag_path, "w") as f:
    json.dump(diag_payload, f, indent=2)
print("Saved super ensemble diagnostic payload to", diag_path)

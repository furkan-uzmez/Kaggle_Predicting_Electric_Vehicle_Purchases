import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

from ev_purchase.data import DatasetContract, encode_target
from ev_purchase.features import apply_feature_policy

PROJECT_ROOT = Path.cwd()
train_df = pd.read_csv(PROJECT_ROOT / "data/playground-series-s6e9/train.csv")
contract = DatasetContract(id_column="id", target_column="Will_Buy_EV")
y_true = encode_target(train_df["Will_Buy_EV"], contract)

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

results = {}
permutation_importance_results = {}

policies_to_test = ["base", "engineered", "selected", "advanced_engineered"]

for policy_name in policies_to_test:
    df_feat = apply_feature_policy(policy_name, train_df)
    drop_cols = ["id", "Will_Buy_EV"]
    X = df_feat.drop(columns=drop_cols)
    cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    for c in cat_cols:
        X[c] = X[c].astype("category")

    oof_preds_by_repeat = [np.zeros(len(df_feat)) for _ in range(3)]
    fold_aucs = []

    print(
        f"\n--- Running 15-fold CV for feature policy: {policy_name} (Features: {X.shape[1]}) ---"
    )

    perm_diffs = {col: [] for col in X.columns}

    for (repeat, fold), group in manifest_df.groupby(["repeat", "fold"]):
        val_idx = group["row_index"].to_numpy()
        train_idx = np.setdiff1d(np.arange(len(df_feat)), val_idx)

        X_train, y_train = X.iloc[train_idx], y_true[train_idx]
        X_val, y_val = X.iloc[val_idx], y_true[val_idx]

        model = LGBMClassifier(**lgb_params)
        model.fit(X_train, y_train)

        preds = model.predict_proba(X_val)[:, 1]
        oof_preds_by_repeat[repeat][val_idx] = preds

        base_auc = roc_auc_score(y_val, preds)
        fold_aucs.append(base_auc)

        if policy_name == "engineered" and repeat == 0:
            rng = np.random.default_rng(42 + fold)
            for col in X.columns:
                X_val_perm = X_val.copy()
                shuffled = rng.permutation(X_val_perm[col].to_numpy())
                if isinstance(X_val[col].dtype, pd.CategoricalDtype):
                    X_val_perm[col] = pd.Categorical(
                        shuffled,
                        categories=X_val[col].cat.categories,
                        ordered=X_val[col].cat.ordered,
                    )
                else:
                    X_val_perm[col] = shuffled
                perm_preds = model.predict_proba(X_val_perm)[:, 1]
                perm_auc = roc_auc_score(y_val, perm_preds)
                perm_diffs[col].append(base_auc - perm_auc)

    # Repeat-averaged OOF ROC-AUC
    avg_oof_preds = np.mean(oof_preds_by_repeat, axis=0)
    repeat_avg_oof_auc = roc_auc_score(y_true, avg_oof_preds)
    mean_fold_auc = np.mean(fold_aucs)
    std_fold_auc = np.std(fold_aucs, ddof=1)

    print(
        f"Policy: {policy_name:10s} | Repeat-Avg OOF ROC-AUC: {repeat_avg_oof_auc:.6f} | "
        f"Mean Fold ROC-AUC: {mean_fold_auc:.6f} +/- {std_fold_auc:.6f}"
    )

    results[policy_name] = {
        "repeat_avg_oof_auc": float(repeat_avg_oof_auc),
        "mean_fold_roc_auc": float(mean_fold_auc),
        "std_fold_roc_auc": float(std_fold_auc),
        "n_features": X.shape[1],
        "feature_names": X.columns.tolist(),
    }

    if policy_name == "engineered":
        perm_summary = [
            {"feature": col, "mean_importance_drop": float(np.mean(diffs))}
            for col, diffs in perm_diffs.items()
        ]
        perm_summary.sort(key=lambda x: x["mean_importance_drop"], reverse=True)
        permutation_importance_results = perm_summary

output_path = PROJECT_ROOT / "outputs/diagnostics/feature_engineering_experiment.json"
output_path.parent.mkdir(parents=True, exist_ok=True)
payload = {"summary": results, "permutation_importance": permutation_importance_results}
with open(output_path, "w") as f:
    json.dump(payload, f, indent=2)

print("\nSaved feature experiment results to", output_path)

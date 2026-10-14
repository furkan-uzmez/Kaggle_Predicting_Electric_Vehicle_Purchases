import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell("""# 🚀 Kaggle EV Purchase Prediction - Super Ensemble (25-Fold GPU)

An end-to-end solution for **Predicting Electric Vehicle Purchases (Kaggle s6e9)**.

### 🌟 Pipeline Highlights:
1. **Robust Auto-Data Finder:** Resolves Kaggle competition inputs or local paths automatically.
2. **5x5 Stratified CV (25-Fold Bagging):** `RepeatedStratifiedKFold` (5 splits x 5 repeats).
3. **GPU Acceleration & Fallback:** Auto-detects CUDA/GPU environment for 10-15x faster training.
4. **Advanced Feature Engineering (`advanced_v2`):**
   - **Group Aggregations:** Income & commute differences from city and car-type averages.
   - **Frequency Encodings:** Categorical representation based on category frequency.
   - **Categorical Interactions:** `Charging_Home_Subsidy_Combo` & `Car_City_Combo`.
   - **Domain Indicators:** `EV_Infrastructure_Score`, `Affordability_Index`.
5. **Multi-Model Diverse Architecture:** LightGBM + CatBoost + XGBoost.
6. **Optimal Weighted Super Ensembling:** Scipy SLSQP constrained weight optimization.
7. **OOF & Submission Export:** Saves `submission.csv` AND `oof_predictions.csv`.
"""),
    nbf.v4.new_code_cell("""import os
import sys
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import RepeatedStratifiedKFold

# GBDT Models
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

# GPU Detection
import torch
HAS_GPU = torch.cuda.is_available()
print(f"GPU Hardware Acceleration Detected: {HAS_GPU}")
if HAS_GPU:
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
"""),
    nbf.v4.new_markdown_cell("""## 1. Robust Environment & Data Path Resolution

Finds `train.csv` dynamically across competition inputs or local paths."""),
    nbf.v4.new_code_cell("""def find_data_directory():
    possible_paths = [
        Path("/kaggle/input/competitions/playground-series-s6e9"),
        Path("/kaggle/input/playground-series-s6e9"),
        Path("data/playground-series-s6e9"),
        Path("data/raw"),
        Path("data"),
        Path("."),
    ]

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for root, dirs, files in os.walk(kaggle_input):
            if "train.csv" in files:
                found_path = Path(root)
                possible_paths.insert(0, found_path)
                break

    for p in possible_paths:
        if (p / "train.csv").exists():
            print(f"✅ Found dataset at: {p.resolve()}")
            return p

    msg = "❌ Could not locate train.csv. Attach dataset to Kaggle notebook."
    raise FileNotFoundError(msg)

DATA_DIR = find_data_directory()

if os.path.exists("/kaggle/working"):
    OUTPUT_DIR = Path("/kaggle/working")
else:
    OUTPUT_DIR = Path("outputs/submissions")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train_path = DATA_DIR / "train.csv"
test_path = DATA_DIR / "test.csv"
sample_sub_path = DATA_DIR / "sample_submission.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_sub_path)

print(f"Train set shape: {train_df.shape}")
print(f"Test set shape:  {test_df.shape}")

target_col = "Will_Buy_EV"
id_col = "id"

y_true = (train_df[target_col] == "Yes").astype(int).to_numpy()
print(f"Target distribution: Positive rate = {y_true.mean():.4f}")
"""),
    nbf.v4.new_markdown_cell("""## 2. Advanced Feature Engineering (`advanced_v2`)

Derives high-signal group aggregations, frequency encodings, combos, and domain indices."""),
    nbf.v4.new_code_cell("""def build_advanced_features(df_in):
    df = df_in.copy()

    if "Annual_Income_USD" in df.columns and "City_Type" in df.columns:
        city_inc_mean = df.groupby("City_Type")["Annual_Income_USD"].transform("mean")
        df["Income_Diff_From_City_Avg"] = df["Annual_Income_USD"] - city_inc_mean

    if "Daily_Commute_km" in df.columns and "City_Type" in df.columns:
        city_com_mean = df.groupby("City_Type")["Daily_Commute_km"].transform("mean")
        df["Commute_Diff_From_City_Avg"] = df["Daily_Commute_km"] - city_com_mean

    if "Annual_Income_USD" in df.columns and "Current_Car_Type" in df.columns:
        car_inc_mean = df.groupby("Current_Car_Type")["Annual_Income_USD"].transform("mean")
        df["Income_Diff_From_CarType_Avg"] = df["Annual_Income_USD"] - car_inc_mean

    for cat_col in ["City_Type", "Current_Car_Type", "Range_Anxiety_Level"]:
        if cat_col in df.columns:
            freq = df[cat_col].map(df[cat_col].value_counts(normalize=True))
            df[f"{cat_col}_Freq"] = freq

    if "Home_Charging_Possible" in df.columns and "Subsidy_Available" in df.columns:
        df["Charging_Home_Subsidy_Combo"] = (
            df["Home_Charging_Possible"].astype(str) + "_" + df["Subsidy_Available"].astype(str)
        )

    if "Current_Car_Type" in df.columns and "City_Type" in df.columns:
        df["Car_City_Combo"] = (
            df["Current_Car_Type"].astype(str) + "_" + df["City_Type"].astype(str)
        )

    has_home = "Home_Charging_Possible" in df.columns
    is_home = (df["Home_Charging_Possible"] == "Yes").astype(float) if has_home else 0.0
    has_near_home = "Charging_Stations_Near_Home" in df.columns
    home_st = df["Charging_Stations_Near_Home"].astype(float) if has_near_home else 0.0
    has_near_work = "Charging_Stations_Near_Work" in df.columns
    work_st = df["Charging_Stations_Near_Work"].astype(float) if has_near_work else 0.0
    df["EV_Infrastructure_Score"] = (is_home * 5.0) + (home_st * 2.0) + work_st

    has_income = "Annual_Income_USD" in df.columns
    has_age = "Age" in df.columns
    has_cars = "Number_of_Cars_Owned" in df.columns
    if has_income and has_age and has_cars:
        cars = df["Number_of_Cars_Owned"].clip(lower=1)
        age_factor = (df["Age"] / 10.0).clip(lower=1.0)
        df["Affordability_Index"] = df["Annual_Income_USD"] / (cars * age_factor)

    if "Range_Anxiety_Level" in df.columns:
        anxiety_map = {"Low": 1.0, "Medium": 2.0, "High": 3.0}
        anxiety_val = df["Range_Anxiety_Level"].map(anxiety_map).fillna(2.0)
        df["Anxiety_Stations_Ratio"] = (home_st + work_st + 1.0) / anxiety_val

    return df

print("Applying feature engineering pipeline...")
train_feat = build_advanced_features(train_df)
test_feat = build_advanced_features(test_df)

drop_cols = [id_col, target_col]
X_train = train_feat.drop(columns=[c for c in drop_cols if c in train_feat.columns])
X_test = test_feat.drop(columns=[c for c in drop_cols if c in test_feat.columns])

cat_cols = X_train.select_dtypes(include=["object", "category", "string"]).columns.tolist()

X_train_lgb = X_train.copy()
X_test_lgb = X_test.copy()
for c in cat_cols:
    X_train_lgb[c] = X_train_lgb[c].astype("category")
    X_test_lgb[c] = X_test_lgb[c].astype("category")

X_train_xgb = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
X_test_xgb = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)
X_train_xgb, X_test_xgb = X_train_xgb.align(X_test_xgb, join="left", axis=1, fill_value=0)

print(f"Engineered Features Count: {X_train.shape[1]}")
"""),
    nbf.v4.new_markdown_cell("""## 3. 25-Fold (5x5) Multi-Model Training with GPU

Runs 5-Fold x 5-Repeat Stratified CV (25 Folds Total) across LightGBM, CatBoost, and XGBoost."""),
    nbf.v4.new_code_cell("""N_SPLITS = 5
N_REPEATS = 5

rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=42)

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
    "device": "gpu" if HAS_GPU else "cpu",
}

cb_params = {
    "iterations": 850,
    "depth": 4,
    "learning_rate": 0.1942,
    "l2_leaf_reg": 8.3748,
    "subsample": 0.8396,
    "bootstrap_type": "Bernoulli",
    "random_strength": 2.2197,
    "random_seed": 42,
    "task_type": "GPU" if HAS_GPU else "CPU",
    "verbose": 0,
    "cat_features": cat_cols,
}

xgb_params = {
    "n_estimators": 450,
    "max_depth": 4,
    "learning_rate": 0.12,
    "subsample": 0.85,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
    "device": "cuda" if HAS_GPU else "cpu",
}

print(f"Strategy: {N_SPLITS} Splits x {N_REPEATS} Repeats = {N_SPLITS * N_REPEATS} Folds Total")
print("Configured Model Execution Mode:")
print(f"  LightGBM Device: {lgb_params['device']}")
print(f"  CatBoost Task:   {cb_params['task_type']} (Bootstrap: Bernoulli)")
print(f"  XGBoost Device:  {xgb_params['device']}")

oof_lgb_repeats = [np.zeros(len(train_df)) for _ in range(N_REPEATS)]
oof_cb_repeats = [np.zeros(len(train_df)) for _ in range(N_REPEATS)]
oof_xgb_repeats = [np.zeros(len(train_df)) for _ in range(N_REPEATS)]

test_preds_lgb, test_preds_cb, test_preds_xgb = [], [], []

print("\nExecuting 25-Fold CV training...")
for fold_idx, (train_idx, val_idx) in enumerate(rskf.split(X_train, y_true)):
    repeat_idx = fold_idx // N_SPLITS

    y_tr, y_va = y_true[train_idx], y_true[val_idx]

    try:
        model_lgb = LGBMClassifier(**lgb_params)
        model_lgb.fit(X_train_lgb.iloc[train_idx], y_tr)
    except Exception:
        lgb_params["device"] = "cpu"
        model_lgb = LGBMClassifier(**lgb_params)
        model_lgb.fit(X_train_lgb.iloc[train_idx], y_tr)

    pred_val_lgb = model_lgb.predict_proba(X_train_lgb.iloc[val_idx])[:, 1]
    oof_lgb_repeats[repeat_idx][val_idx] = pred_val_lgb
    test_preds_lgb.append(model_lgb.predict_proba(X_test_lgb)[:, 1])

    model_cb = CatBoostClassifier(**cb_params)
    model_cb.fit(X_train.iloc[train_idx], y_tr)
    pred_val_cb = model_cb.predict_proba(X_train.iloc[val_idx])[:, 1]
    oof_cb_repeats[repeat_idx][val_idx] = pred_val_cb
    test_preds_cb.append(model_cb.predict_proba(X_test)[:, 1])

    model_xgb = XGBClassifier(**xgb_params)
    model_xgb.fit(X_train_xgb.iloc[train_idx], y_tr)
    pred_val_xgb = model_xgb.predict_proba(X_train_xgb.iloc[val_idx])[:, 1]
    oof_xgb_repeats[repeat_idx][val_idx] = pred_val_xgb
    test_preds_xgb.append(model_xgb.predict_proba(X_test_xgb)[:, 1])

    if (fold_idx + 1) % 5 == 0:
        rep_num = repeat_idx + 1
        f_start = (repeat_idx * 5) + 1
        f_end = (repeat_idx + 1) * 5
        print(f"Completed Repeat {rep_num}/{N_REPEATS} (Folds {f_start}-{f_end})")

oof_lgb = np.mean(oof_lgb_repeats, axis=0)
oof_cb = np.mean(oof_cb_repeats, axis=0)
oof_xgb = np.mean(oof_xgb_repeats, axis=0)

print("\n--- OOF Individual Model Scores ---")
print(f"LightGBM OOF ROC-AUC: {roc_auc_score(y_true, oof_lgb):.6f}")
print(f"CatBoost OOF ROC-AUC: {roc_auc_score(y_true, oof_cb):.6f}")
print(f"XGBoost  OOF ROC-AUC: {roc_auc_score(y_true, oof_xgb):.6f}")
"""),
    nbf.v4.new_markdown_cell("""## 4. Super Ensemble Weight Optimization

Uses Scipy SLSQP constrained optimization to find optimal ensemble weights."""),
    nbf.v4.new_code_cell("""def ensemble_loss(weights):
    w_lgb, w_cb, w_xgb = weights
    blend = w_lgb * oof_lgb + w_cb * oof_cb + w_xgb * oof_xgb
    return -roc_auc_score(y_true, blend)

init_weights = [0.4, 0.4, 0.2]
bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
constraints = {"type": "eq", "fun": lambda w: sum(w) - 1.0}

opt_res = minimize(
    ensemble_loss, init_weights, method="SLSQP", bounds=bounds, constraints=constraints
)
w_lgb_opt, w_cb_opt, w_xgb_opt = opt_res.x
super_oof_auc = -opt_res.fun

print(f"Optimal Super Ensemble Weights:")
print(f"  LightGBM Weight: {w_lgb_opt:.4f}")
print(f"  CatBoost Weight: {w_cb_opt:.4f}")
print(f"  XGBoost Weight:  {w_xgb_opt:.4f}")
print(f"🚀 Super Ensemble OOF ROC-AUC: {super_oof_auc:.6f}")
"""),
    nbf.v4.new_markdown_cell("""## 5. Submission Generation & OOF Export & Contract Validation

Saves `submission.csv` AND `oof_predictions.csv` with full contract checks."""),
    nbf.v4.new_code_cell("""test_lgb = np.mean(test_preds_lgb, axis=0)
test_cb = np.mean(test_preds_cb, axis=0)
test_xgb = np.mean(test_preds_xgb, axis=0)

final_super_preds = w_lgb_opt * test_lgb + w_cb_opt * test_cb + w_xgb_opt * test_xgb
final_super_oof = w_lgb_opt * oof_lgb + w_cb_opt * oof_cb + w_xgb_opt * oof_xgb

# 1. Export Submission File
sub_df = pd.DataFrame({"id": test_df[id_col], target_col: final_super_preds})

# 2. Export Out-Of-Fold (OOF) Predictions File
oof_df = pd.DataFrame({
    "id": train_df[id_col],
    "oof_lgb": oof_lgb,
    "oof_cb": oof_cb,
    "oof_xgb": oof_xgb,
    "oof_super_ensemble": final_super_oof,
    "target": y_true,
})

# Contract Checks
assert len(sub_df) == len(sample_sub), f"Row count mismatch: {len(sub_df)} vs {len(sample_sub)}"
assert list(sub_df.columns) == list(sample_sub.columns), "Column schema mismatch"
assert (sub_df["id"].to_numpy() == sample_sub["id"].to_numpy()).all(), "ID sequence mismatch"
assert sub_df[target_col].isna().sum() == 0, "Null values found in predictions"
assert np.isfinite(sub_df[target_col]).all(), "Non-finite probabilities found"

min_p, max_p = sub_df[target_col].min(), sub_df[target_col].max()
assert (
    0.0 <= min_p <= 1.0 and 0.0 <= max_p <= 1.0
), f"Probability range out of bounds: [{min_p}, {max_p}]"

print("✅ ALL SUBMISSION CONTRACT CHECKS PASSED!")

sub_file_path = OUTPUT_DIR / "submission.csv"
oof_file_path = OUTPUT_DIR / "oof_predictions.csv"

sub_df.to_csv(sub_file_path, index=False)
oof_df.to_csv(oof_file_path, index=False)

sub_bytes = sub_file_path.read_bytes()
sha256_hash = hashlib.sha256(sub_bytes).hexdigest()

print(f"\nFinal submission generated: {sub_file_path}")
print(f"OOF predictions exported:  {oof_file_path}")
print(f"SHA-256 Checksum: {sha256_hash}")
print(f"Total Test Rows: {len(sub_df)}, OOF Train Rows: {len(oof_df)}")
print(f"Prediction Probability Range: [{min_p:.6f}, {max_p:.6f}]")
""")
]

nb.cells = cells

output_nb_path = Path("notebooks/kaggle_ev_purchase_super_ensemble.ipynb")
output_nb_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_nb_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Generated Kaggle Notebook with OOF Export at:", output_nb_path)

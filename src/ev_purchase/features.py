"""Feature policy definition and registry for EV purchase prediction."""

from collections.abc import Callable

import pandas as pd

FeaturePolicy = Callable[[pd.DataFrame], pd.DataFrame]


def base_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Base feature policy: returns an unmutated copy of valid raw model columns."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"frame must be a pandas DataFrame, got {type(frame).__name__}")
    return frame.copy()


def engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Engineered feature policy: derives domain ratios and interactions."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"frame must be a pandas DataFrame, got {type(frame).__name__}")
    df = frame.copy()
    if "Annual_Income_USD" in df.columns and "Age" in df.columns:
        df["Income_per_Age"] = df["Annual_Income_USD"] / (df["Age"] + 1)
    if "Charging_Stations_Near_Home" in df.columns and "Charging_Stations_Near_Work" in df.columns:
        df["Total_Charging_Stations"] = (
            df["Charging_Stations_Near_Home"] + df["Charging_Stations_Near_Work"]
        )
    if "Annual_Income_USD" in df.columns and "Daily_Commute_km" in df.columns:
        df["Income_to_Commute_Ratio"] = df["Annual_Income_USD"] / (df["Daily_Commute_km"] + 1)
    if "Daily_Commute_km" in df.columns and "Age" in df.columns:
        df["Commute_per_Age"] = df["Daily_Commute_km"] / (df["Age"] + 1)
    return df


def selected_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Selected feature policy: drops noisy/low-importance features."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"frame must be a pandas DataFrame, got {type(frame).__name__}")
    df = frame.copy()
    noisy_cols = ["Gender", "City_Type", "Number_of_Cars_Owned"]
    cols_to_drop = [c for c in noisy_cols if c in df.columns]
    return df.drop(columns=cols_to_drop)


def advanced_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Advanced engineered feature policy: group aggs, frequency encodings, and category combos."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"frame must be a pandas DataFrame, got {type(frame).__name__}")
    df = frame.copy()

    # 1. Group Aggregations (Diff from Group Mean)
    if "Annual_Income_USD" in df.columns and "City_Type" in df.columns:
        city_inc_mean = df.groupby("City_Type")["Annual_Income_USD"].transform("mean")
        df["Income_Diff_From_City_Avg"] = df["Annual_Income_USD"] - city_inc_mean

    if "Daily_Commute_km" in df.columns and "City_Type" in df.columns:
        city_com_mean = df.groupby("City_Type")["Daily_Commute_km"].transform("mean")
        df["Commute_Diff_From_City_Avg"] = df["Daily_Commute_km"] - city_com_mean

    if "Annual_Income_USD" in df.columns and "Current_Car_Type" in df.columns:
        car_inc_mean = df.groupby("Current_Car_Type")["Annual_Income_USD"].transform("mean")
        df["Income_Diff_From_CarType_Avg"] = df["Annual_Income_USD"] - car_inc_mean

    # 2. Frequency Encodings
    for cat_col in ["City_Type", "Current_Car_Type", "Range_Anxiety_Level"]:
        if cat_col in df.columns:
            freq = df[cat_col].map(df[cat_col].value_counts(normalize=True))
            df[f"{cat_col}_Freq"] = freq

    # 3. Categorical Combinations
    if "Home_Charging_Possible" in df.columns and "Subsidy_Available" in df.columns:
        df["Charging_Home_Subsidy_Combo"] = (
            df["Home_Charging_Possible"].astype(str) + "_" + df["Subsidy_Available"].astype(str)
        )

    if "Current_Car_Type" in df.columns and "City_Type" in df.columns:
        df["Car_City_Combo"] = (
            df["Current_Car_Type"].astype(str) + "_" + df["City_Type"].astype(str)
        )

    return df


def advanced_v2_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Advanced v2 feature policy: Hata analizi bulgularına dayalı etkileşim ve altyapı skorları."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"frame must be a pandas DataFrame, got {type(frame).__name__}")
    df = advanced_engineered_features(frame)

    # 1. Altyapı ve Hazırlık Skoru (EV Infrastructure Score)
    home_charge = (df["Home_Charging_Possible"] == "Yes").astype(float) if "Home_Charging_Possible" in df.columns else 0.0
    home_st = df["Charging_Stations_Near_Home"].astype(float) if "Charging_Stations_Near_Home" in df.columns else 0.0
    work_st = df["Charging_Stations_Near_Work"].astype(float) if "Charging_Stations_Near_Work" in df.columns else 0.0
    df["EV_Infrastructure_Score"] = (home_charge * 5.0) + (home_st * 2.0) + work_st

    # 2. Alım Gücü ve Gelir İndeksi (Affordability Index)
    if "Annual_Income_USD" in df.columns and "Age" in df.columns and "Number_of_Cars_Owned" in df.columns:
        cars = df["Number_of_Cars_Owned"].clip(lower=1)
        age_factor = (df["Age"] / 10.0).clip(lower=1.0)
        df["Affordability_Index"] = df["Annual_Income_USD"] / (cars * age_factor)

    # 3. Menzil Endişesi vs İstasyon Oranı (Anxiety vs Infrastructure)
    if "Range_Anxiety_Level" in df.columns:
        anxiety_map = {"Low": 1.0, "Medium": 2.0, "High": 3.0}
        anxiety_val = df["Range_Anxiety_Level"].map(anxiety_map).fillna(2.0)
        df["Anxiety_Stations_Ratio"] = (home_st + work_st + 1.0) / anxiety_val

    return df


BUILTIN_FEATURE_POLICIES: set[str] = {
    "base",
    "engineered",
    "selected",
    "advanced_engineered",
    "advanced_v2",
}

_FEATURE_POLICIES: dict[str, FeaturePolicy] = {
    "base": base_features,
    "engineered": engineered_features,
    "selected": selected_features,
    "advanced_engineered": advanced_engineered_features,
    "advanced_v2": advanced_v2_features,
}


def register_feature_policy(name: str, policy: FeaturePolicy, overwrite: bool = False) -> None:
    """Register a new feature policy.

    Args:
        name: Unique string identifier for the policy.
        policy: Callable that accepts a DataFrame and returns a DataFrame.
        overwrite: If True, allow replacing an existing policy with the same name.

    Raises:
        TypeError: If policy is not callable.
        ValueError: If name is not a non-empty string or policy is already registered
            without overwrite=True.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("policy name must be a non-empty string")
    if not callable(policy):
        raise TypeError("policy must be a callable accepting and returning a DataFrame")

    clean_name = name.strip()
    if clean_name in _FEATURE_POLICIES and not overwrite:
        raise ValueError(
            f"Policy '{clean_name}' is already registered. Set overwrite=True to replace it."
        )

    _FEATURE_POLICIES[clean_name] = policy


def unregister_feature_policy(name: str) -> None:
    """Unregister a non-builtin feature policy by name.

    Args:
        name: Policy name to unregister.

    Raises:
        ValueError: If name is not a non-empty string, is a built-in policy, or is not registered.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("policy name must be a non-empty string")

    clean_name = name.strip()
    if clean_name in BUILTIN_FEATURE_POLICIES:
        raise ValueError(f"Cannot unregister built-in policy '{clean_name}'")

    if clean_name not in _FEATURE_POLICIES:
        raise ValueError(f"Policy '{clean_name}' is not registered")

    del _FEATURE_POLICIES[clean_name]


def reset_feature_policies() -> None:
    """Reset feature policy registry back to built-in defaults."""
    _FEATURE_POLICIES.clear()
    _FEATURE_POLICIES["base"] = base_features
    _FEATURE_POLICIES["engineered"] = engineered_features
    _FEATURE_POLICIES["selected"] = selected_features
    _FEATURE_POLICIES["advanced_engineered"] = advanced_engineered_features
    _FEATURE_POLICIES["advanced_v2"] = advanced_v2_features


def get_feature_policy(name: str) -> FeaturePolicy:
    """Look up a feature policy by name.

    Args:
        name: Name of the registered feature policy.

    Returns:
        The policy callable.

    Raises:
        ValueError: If name is not a non-empty string or policy is not found.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("policy name must be a non-empty string")

    clean_name = name.strip()
    if clean_name not in _FEATURE_POLICIES:
        available = ", ".join(sorted(_FEATURE_POLICIES.keys()))
        raise ValueError(
            f"Unknown feature policy '{clean_name}'. Available policies: [{available}]"
        )
    return _FEATURE_POLICIES[clean_name]


def list_feature_policies() -> list[str]:
    """Return a sorted list of registered feature policy names."""
    return sorted(_FEATURE_POLICIES.keys())


def apply_feature_policy(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Apply a named feature policy to a DataFrame without mutating the input.

    Args:
        name: Policy name.
        frame: Input DataFrame.

    Returns:
        Transformed DataFrame.

    Raises:
        TypeError: If frame is not a DataFrame or policy result is not a DataFrame.
        RuntimeError: If policy mutates the input DataFrame.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"frame must be a pandas DataFrame, got {type(frame).__name__}")

    policy_fn = get_feature_policy(name)
    original_copy = frame.copy(deep=True)
    result = policy_fn(frame)

    if not isinstance(result, pd.DataFrame):
        raise TypeError(
            f"Feature policy '{name}' must return a pandas DataFrame, got {type(result).__name__}"
        )

    if not frame.equals(original_copy):
        raise RuntimeError(f"Feature policy '{name}' mutated its input DataFrame")

    return result

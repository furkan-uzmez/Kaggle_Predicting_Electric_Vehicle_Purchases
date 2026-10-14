import pandas as pd
import pytest

from ev_purchase.features import (
    engineered_features,
    get_feature_policy,
    list_feature_policies,
)


def test_engineered_features_adds_expected_columns() -> None:
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "Age": [30, 50],
            "Annual_Income_USD": [60000.0, 100000.0],
            "Daily_Commute_km": [20.0, 50.0],
            "Charging_Stations_Near_Home": [2, 0],
            "Charging_Stations_Near_Work": [3, 1],
            "Will_Buy_EV": ["No", "Yes"],
        }
    )
    res = engineered_features(df)
    expected_new_cols = [
        "Income_per_Age",
        "Total_Charging_Stations",
        "Income_to_Commute_Ratio",
        "Commute_per_Age",
    ]
    for col in expected_new_cols:
        assert col in res.columns, f"Missing feature: {col}"

    assert res["Total_Charging_Stations"].tolist() == [5, 1]
    assert res["Income_per_Age"].iloc[0] == pytest.approx(60000.0 / 31.0)


def test_engineered_policy_registration() -> None:
    assert "engineered" in list_feature_policies()
    policy_fn = get_feature_policy("engineered")
    assert policy_fn == engineered_features


def test_selected_features_removes_low_importance_columns() -> None:
    from ev_purchase.features import selected_features

    df = pd.DataFrame(
        {
            "id": [1, 2],
            "Age": [30, 50],
            "Gender": ["Male", "Female"],
            "City_Type": ["Urban", "Suburban"],
            "Number_of_Cars_Owned": [1, 2],
            "Annual_Income_USD": [60000.0, 100000.0],
            "Will_Buy_EV": ["No", "Yes"],
        }
    )
    res = selected_features(df)
    assert "Gender" not in res.columns
    assert "City_Type" not in res.columns
    assert "Number_of_Cars_Owned" not in res.columns
    assert "Annual_Income_USD" in res.columns


def test_advanced_engineered_features_adds_group_aggs_freq_and_combos() -> None:
    from ev_purchase.features import advanced_engineered_features

    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "Age": [30, 50, 40],
            "Annual_Income_USD": [60000.0, 100000.0, 80000.0],
            "Daily_Commute_km": [20.0, 50.0, 30.0],
            "Charging_Stations_Near_Home": [2, 0, 1],
            "Charging_Stations_Near_Work": [3, 1, 0],
            "Number_of_Cars_Owned": [1, 2, 1],
            "City_Type": ["Urban", "Urban", "Suburban"],
            "Current_Car_Type": ["SUV", "Sedan", "SUV"],
            "Home_Charging_Possible": ["Yes", "No", "Yes"],
            "Subsidy_Available": ["Yes", "No", "No"],
            "Range_Anxiety_Level": ["Low", "High", "Medium"],
            "Will_Buy_EV": ["No", "Yes", "No"],
        }
    )
    res = advanced_engineered_features(df)
    expected_cols = [
        "Income_Diff_From_City_Avg",
        "Commute_Diff_From_City_Avg",
        "Income_Diff_From_CarType_Avg",
        "City_Type_Freq",
        "Current_Car_Type_Freq",
        "Range_Anxiety_Level_Freq",
        "Charging_Home_Subsidy_Combo",
        "Car_City_Combo",
    ]
    for col in expected_cols:
        assert col in res.columns, f"Missing column: {col}"

    assert res["Charging_Home_Subsidy_Combo"].tolist() == ["Yes_Yes", "No_No", "Yes_No"]
    assert res["Car_City_Combo"].tolist() == ["SUV_Urban", "Sedan_Urban", "SUV_Suburban"]


def test_advanced_v2_features_adds_infrastructure_and_affordability_scores() -> None:
    from ev_purchase.features import advanced_v2_features

    df = pd.DataFrame(
        {
            "id": [1, 2],
            "Age": [30, 50],
            "Annual_Income_USD": [60000.0, 100000.0],
            "Daily_Commute_km": [20.0, 50.0],
            "Charging_Stations_Near_Home": [2, 0],
            "Charging_Stations_Near_Work": [3, 1],
            "Number_of_Cars_Owned": [1, 2],
            "City_Type": ["Urban", "Suburban"],
            "Current_Car_Type": ["SUV", "Sedan"],
            "Home_Charging_Possible": ["Yes", "No"],
            "Subsidy_Available": ["Yes", "No"],
            "Range_Anxiety_Level": ["Low", "High"],
            "Will_Buy_EV": ["No", "Yes"],
        }
    )
    res = advanced_v2_features(df)
    expected_cols = [
        "EV_Infrastructure_Score",
        "Affordability_Index",
        "Anxiety_Stations_Ratio",
    ]
    for col in expected_cols:
        assert col in res.columns, f"Missing column in advanced_v2: {col}"

    assert res["EV_Infrastructure_Score"].tolist() == [12.0, 1.0]

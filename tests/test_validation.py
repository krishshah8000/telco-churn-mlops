import pandas as pd
import pytest

from src.validate import validate_data


def create_valid_dataframe():
    return pd.DataFrame(
        {
            "customerID": ["001", "002"],
            "gender": ["Male", "Female"],
            "SeniorCitizen": [0, 1],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "Yes"],
            "tenure": [1, 24],
            "PhoneService": ["No", "Yes"],
            "MultipleLines": ["No phone service", "No"],
            "InternetService": ["DSL", "Fiber optic"],
            "OnlineSecurity": ["No", "Yes"],
            "OnlineBackup": ["Yes", "No"],
            "DeviceProtection": ["No", "Yes"],
            "TechSupport": ["No", "Yes"],
            "StreamingTV": ["No", "Yes"],
            "StreamingMovies": ["No", "Yes"],
            "Contract": ["Month-to-month", "One year"],
            "PaperlessBilling": ["Yes", "No"],
            "PaymentMethod": [
                "Electronic check",
                "Credit card (automatic)",
            ],
            "MonthlyCharges": [29.85, 70.35],
            "TotalCharges": ["29.85", "1680.85"],
            "Churn": ["No", "Yes"],
        }
    )


def test_valid_data_passes():
    df = create_valid_dataframe()

    assert validate_data(df) is True


def test_empty_dataset_fails():
    df = pd.DataFrame()

    with pytest.raises(ValueError, match="Dataset is empty"):
        validate_data(df)


def test_missing_required_column_fails():
    df = create_valid_dataframe()

    df = df.drop(columns=["gender"])

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        validate_data(df)


def test_duplicate_customer_id_fails():
    df = create_valid_dataframe()

    df.loc[1, "customerID"] = "001"

    with pytest.raises(
        ValueError,
        match="duplicate customer IDs",
    ):
        validate_data(df)


def test_invalid_category_fails():
    df = create_valid_dataframe()

    df.loc[0, "gender"] = "Unknown"

    with pytest.raises(
        ValueError,
        match="Unexpected categories",
    ):
        validate_data(df)

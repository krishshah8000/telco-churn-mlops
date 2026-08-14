import joblib
import pandas as pd

from src.preprocess import create_preprocessor
from src.transform_data import transform_data


def create_test_dataframe():
    return pd.DataFrame(
        {
            "customerID": ["001", "002"],
            "gender": ["Male", "Female"],
            "SeniorCitizen": [0, 1],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "Yes"],
            "tenure": [1, 24],
            "PhoneService": ["No", "Yes"],
            "MultipleLines": [
                "No phone service",
                "No",
            ],
            "InternetService": [
                "DSL",
                "Fiber optic",
            ],
            "OnlineSecurity": ["No", "Yes"],
            "OnlineBackup": ["Yes", "No"],
            "DeviceProtection": ["No", "Yes"],
            "TechSupport": ["No", "Yes"],
            "StreamingTV": ["No", "Yes"],
            "StreamingMovies": ["No", "Yes"],
            "Contract": [
                "Month-to-month",
                "One year",
            ],
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


def fit_test_preprocessor(df):
    """
    Create and fit the project's actual preprocessor
    using the test feature data.
    """

    X = df.drop(
        columns=["Churn", "customerID"]
    ).copy()

    # Match the project's preprocessing logic.
    X["TotalCharges"] = (
        X["TotalCharges"]
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
    )

    X["TotalCharges"] = pd.to_numeric(
        X["TotalCharges"],
        errors="coerce",
    )

    preprocessor = create_preprocessor()

    preprocessor.fit(X)

    return preprocessor


def test_transform_data_creates_processed_file(
    tmp_path,
):
    input_file = tmp_path / "input.csv"
    output_file = tmp_path / "processed.csv"
    preprocessor_file = (
        tmp_path / "preprocessor.joblib"
    )

    df = create_test_dataframe()

    df.to_csv(
        input_file,
        index=False,
    )

    # Create and fit the project's actual preprocessor.
    preprocessor = fit_test_preprocessor(df)

    joblib.dump(
        preprocessor,
        preprocessor_file,
    )

    transform_data(
        input_path=str(input_file),
        output_path=str(output_file),
        preprocessor_path=str(preprocessor_file),
    )

    assert output_file.exists()


def test_transform_data_preserves_target(
    tmp_path,
):
    input_file = tmp_path / "input.csv"
    output_file = tmp_path / "processed.csv"
    preprocessor_file = (
        tmp_path / "preprocessor.joblib"
    )

    df = create_test_dataframe()

    df.to_csv(
        input_file,
        index=False,
    )

    # Create and fit the project's actual preprocessor.
    preprocessor = fit_test_preprocessor(df)

    joblib.dump(
        preprocessor,
        preprocessor_file,
    )

    transform_data(
        input_path=str(input_file),
        output_path=str(output_file),
        preprocessor_path=str(preprocessor_file),
    )

    processed_df = pd.read_csv(
        output_file
    )

    assert "Churn" in processed_df.columns

    # Original target:
    # No -> 0
    # Yes -> 1
    assert processed_df["Churn"].tolist() == [
        0,
        1,
    ]

    assert len(processed_df) == len(df)

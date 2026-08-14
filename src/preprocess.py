import os

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

NUMERICAL_COLUMNS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_COLUMNS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

PREPROCESSOR_PATH = (
    "artifacts/preprocessing/preprocessor.joblib"
)


def create_preprocessor():
    """
    Create the preprocessing pipeline.
    """

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                NUMERICAL_COLUMNS,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_COLUMNS,
            ),
        ]
    )

    return preprocessor


def prepare_dataframe(df):
    """
    Prepare raw dataframe before applying the ML preprocessor.
    """

    df = df.copy()

    # Convert blank TotalCharges to NaN.
    df["TotalCharges"] = (
        df["TotalCharges"]
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
    )

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce",
    )

    # Remove customer ID because it is an identifier,
    # not a useful predictive feature.
    df = df.drop(columns=[ID_COLUMN])

    return df


def fit_and_transform(
    input_path,
    output_path,
    preprocessor_path=PREPROCESSOR_PATH,
):
    """
    Fit preprocessing on training data and transform it.
    """

    df = pd.read_csv(input_path)

    y = (
        df[TARGET_COLUMN]
        .map({"No": 0, "Yes": 1})
        .astype(int)
    )

    X = df.drop(columns=[TARGET_COLUMN])

    X = prepare_dataframe(X)

    preprocessor = create_preprocessor()

    X_processed = preprocessor.fit_transform(X)

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    processed_df = pd.DataFrame(
        X_processed,
        columns=feature_names,
    )

    processed_df[TARGET_COLUMN] = y.values

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True,
    )

    os.makedirs(
        os.path.dirname(preprocessor_path),
        exist_ok=True,
    )

    processed_df.to_csv(
        output_path,
        index=False,
    )

    joblib.dump(
        preprocessor,
        preprocessor_path,
    )

    print("\n===== PREPROCESSING COMPLETE =====")
    print(f"Input rows       : {len(df)}")
    print(f"Processed rows   : {len(processed_df)}")
    print(f"Processed columns: {len(feature_names)}")
    print(f"Saved data       : {output_path}")
    print(f"Saved preprocessor: {preprocessor_path}")

    return processed_df, preprocessor


if __name__ == "__main__":

    fit_and_transform(
        input_path="data/initial/initial_data.csv",
        output_path="artifacts/preprocessing/initial_processed.csv",
    )

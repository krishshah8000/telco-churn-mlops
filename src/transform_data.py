import os

import joblib
import pandas as pd


TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

PREPROCESSOR_PATH = (
    "artifacts/preprocessing/preprocessor.joblib"
)


def prepare_dataframe(df):
    """
    Prepare raw dataframe before applying
    the saved preprocessing pipeline.
    """

    df = df.copy()

    # Convert blank TotalCharges to missing values
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

    # Remove customer ID
    df = df.drop(columns=[ID_COLUMN])

    return df


def transform_data(
    input_path,
    output_path,
    preprocessor_path=PREPROCESSOR_PATH,
):
    """
    Transform data using the already-fitted preprocessor.
    """

    # Load raw data
    df = pd.read_csv(input_path)

    # Separate target
    y = (
        df[TARGET_COLUMN]
        .map({"No": 0, "Yes": 1})
        .astype(int)
    )

    X = df.drop(columns=[TARGET_COLUMN])

    # Prepare features
    X = prepare_dataframe(X)

    # Load saved preprocessor
    preprocessor = joblib.load(
        preprocessor_path
    )

    # Transform without fitting again
    X_processed = preprocessor.transform(X)

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

    processed_df.to_csv(
        output_path,
        index=False,
    )

    print("\n===== DATA TRANSFORMATION COMPLETE =====")
    print(f"Input rows       : {len(df)}")
    print(f"Processed rows   : {len(processed_df)}")
    print(f"Processed columns: {len(feature_names)}")
    print(f"Saved data       : {output_path}")


if __name__ == "__main__":

    # Transform new data
    transform_data(
        input_path="data/new/new_data.csv",
        output_path=(
            "artifacts/preprocessing/new_processed.csv"
        ),
    )

    # Transform held-out data
    transform_data(
        input_path="data/held_out/held_out_data.csv",
        output_path=(
            "artifacts/preprocessing/held_out_processed.csv"
        ),
    )

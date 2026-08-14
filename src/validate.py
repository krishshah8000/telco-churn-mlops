import pandas as pd


REQUIRED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
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
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]

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
    "Churn",
]


# Expected categories from the Telco Customer Churn dataset
EXPECTED_CATEGORIES = {
    "gender": {"Male", "Female"},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": {"Yes", "No", "No internet service"},
    "OnlineBackup": {"Yes", "No", "No internet service"},
    "DeviceProtection": {"Yes", "No", "No internet service"},
    "TechSupport": {"Yes", "No", "No internet service"},
    "StreamingTV": {"Yes", "No", "No internet service"},
    "StreamingMovies": {"Yes", "No", "No internet service"},
    "Contract": {
        "Month-to-month",
        "One year",
        "Two year",
    },
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    },
    "Churn": {"Yes", "No"},
}


def validate_data(df: pd.DataFrame) -> bool:
    """Validate the Telco Customer Churn dataset."""

    print("\n===== STARTING DATA VALIDATION =====")

    # 1. Check whether dataset is empty
    if df.empty:
        raise ValueError("Dataset is empty.")

    print("✓ Dataset is not empty")

    # 2. Check required columns
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print("✓ All required columns are present")

    # 3. Check for unexpected columns
    unexpected_columns = [
        column
        for column in df.columns
        if column not in REQUIRED_COLUMNS
    ]

    if unexpected_columns:
        raise ValueError(
            f"Unexpected columns found: {unexpected_columns}"
        )

    print("✓ No unexpected columns found")

    # 4. Check duplicate customer IDs
    duplicate_ids = df["customerID"].duplicated().sum()

    if duplicate_ids > 0:
        raise ValueError(
            f"Found {duplicate_ids} duplicate customer IDs."
        )

    print("✓ No duplicate customer IDs")

    # 5. Check pandas missing values
    missing_values = df.isnull().sum()

    if missing_values.any():
        missing = missing_values[
            missing_values > 0
        ].to_dict()

        raise ValueError(
            f"Missing values found: {missing}"
        )

    print("✓ No pandas missing values")

    # 6. Check blank values
    blank_values = {}

    for column in df.columns:
        blank_count = (
            df[column]
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        )

        if blank_count > 0:
            blank_values[column] = blank_count

    # TotalCharges blanks are known in this dataset.
    # They will be handled during preprocessing.
    allowed_blank_columns = {"TotalCharges"}

    unexpected_blank_columns = (
        set(blank_values.keys())
        - allowed_blank_columns
    )

    if unexpected_blank_columns:
        raise ValueError(
            "Unexpected blank values found: "
            f"{unexpected_blank_columns}"
        )

    if "TotalCharges" in blank_values:
        print(
            f"✓ TotalCharges contains "
            f"{blank_values['TotalCharges']} blank values "
            "and will be handled during preprocessing"
        )
    else:
        print("✓ No blank values")

    # 7. Validate categorical columns
    for column in CATEGORICAL_COLUMNS:

        actual_values = set(
            df[column]
            .astype(str)
            .str.strip()
            .unique()
        )

        expected_values = EXPECTED_CATEGORIES[column]

        unexpected_values = (
            actual_values - expected_values
        )

        if unexpected_values:
            raise ValueError(
                f"Unexpected categories in '{column}': "
                f"{unexpected_values}"
            )

        print(f"✓ Categories valid: {column}")

    # 8. Validate numerical columns
    for column in NUMERICAL_COLUMNS:

        # TotalCharges may contain blanks.
        # Those blanks are handled during preprocessing.
        if column == "TotalCharges":
            values = df[column].astype(str).str.strip()
            values = values.replace("", pd.NA)
            values = pd.to_numeric(
                values,
                errors="coerce"
            )

            invalid_values = (
                values.isna()
                & df[column].astype(str).str.strip().ne("")
            ).sum()

            if invalid_values > 0:
                raise ValueError(
                    f"Column '{column}' contains "
                    "invalid numerical values."
                )

        else:
            try:
                pd.to_numeric(
                    df[column],
                    errors="raise"
                )
            except (ValueError, TypeError):
                raise ValueError(
                    f"Column '{column}' contains "
                    "non-numerical values."
                )

        print(f"✓ Numerical values valid: {column}")

    # 9. SeniorCitizen must be 0 or 1
    senior_values = set(
        pd.to_numeric(df["SeniorCitizen"]).unique()
    )

    if not senior_values.issubset({0, 1}):
        raise ValueError(
            f"Invalid SeniorCitizen values: "
            f"{senior_values}"
        )

    print("✓ SeniorCitizen values are valid")

    # 10. Tenure cannot be negative
    tenure = pd.to_numeric(df["tenure"])

    if (tenure < 0).any():
        raise ValueError(
            "Negative tenure values found."
        )

    print("✓ Tenure values are valid")

    # 11. MonthlyCharges cannot be negative
    monthly_charges = pd.to_numeric(
        df["MonthlyCharges"]
    )

    if (monthly_charges < 0).any():
        raise ValueError(
            "Negative MonthlyCharges values found."
        )

    print("✓ MonthlyCharges values are valid")

    # 12. TotalCharges cannot be negative
    total_charges = (
        df["TotalCharges"]
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
    )

    total_charges = pd.to_numeric(
        total_charges,
        errors="coerce"
    )

    if (total_charges.dropna() < 0).any():
        raise ValueError(
            "Negative TotalCharges values found."
        )

    print("✓ TotalCharges values are valid")

    print("\n===== VALIDATION PASSED =====")

    return True


if __name__ == "__main__":

    initial_data = pd.read_csv(
        "data/initial/initial_data.csv"
    )

    validate_data(initial_data)

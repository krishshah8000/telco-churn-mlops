import os
import pandas as pd

from zenml import pipeline, step

from src.validate import validate_data
from src.transform_data import transform_data
from src.retrain import retrain
from src.evaluate import evaluate_candidate
from src.promote import compare_and_promote


# ============================================================
# CONFIGURATION
# ============================================================

NEW_DATA = (
    "data/new/new_data.csv"
)

PREPROCESSOR_PATH = (
    "artifacts/preprocessing/preprocessor.joblib"
)


# ============================================================
# STEP 1 — INGEST NEW DATA
# ============================================================

@step
def ingest_new_data(
    new_data_path: str,
) -> str:

    print("\n==========================================")
    print("ZENML INGEST STEP")
    print("==========================================")

    if not os.path.exists(
        new_data_path
    ):
        raise FileNotFoundError(
            f"New data not found: "
            f"{new_data_path}"
        )

    df = pd.read_csv(
        new_data_path
    )

    if df.empty:
        raise ValueError(
            "New data is empty."
        )

    print(
        f"New data rows: {len(df)}"
    )

    print(
        f"New data columns: {len(df.columns)}"
    )

    return new_data_path


# ============================================================
# STEP 2 — VALIDATE NEW DATA
# ============================================================

@step
def validate_new_data(
    new_data_path: str,
) -> str:

    print("\n==========================================")
    print("ZENML VALIDATION STEP")
    print("==========================================")

    df = pd.read_csv(
        new_data_path
    )

    # Reuse existing validation logic
    validate_data(
        df
    )

    print(
        f"✓ Validation passed for {len(df)} rows"
    )

    return new_data_path


# ============================================================
# STEP 3 — TRANSFORM NEW DATA
# ============================================================

@step
def transform_new_data(
    validated_data_path: str,
) -> str:

    print("\n==========================================")
    print("ZENML TRANSFORMATION STEP")
    print("==========================================")

    output_path = (
        "artifacts/preprocessing/"
        "new_processed.csv"
    )

    os.makedirs(
        os.path.dirname(
            output_path
        ),
        exist_ok=True,
    )

    # Reuse existing transformation implementation
    transform_data(
        input_path=validated_data_path,
        output_path=output_path,
        preprocessor_path=PREPROCESSOR_PATH,
    )

    # --------------------------------------------------------
    # Verify processed data
    # --------------------------------------------------------

    processed = pd.read_csv(
        output_path
    )

    expected_features = 45

    if "Churn" not in processed.columns:
        raise ValueError(
            "Churn target missing after transformation."
        )

    feature_count = (
        len(processed.columns) - 1
    )

    if feature_count != expected_features:
        raise ValueError(
            f"Expected {expected_features} features "
            f"but found {feature_count}."
        )

    print(
        f"✓ Processed rows: {len(processed)}"
    )

    print(
        f"✓ Features: {feature_count}"
    )

    print(
        "✓ Churn target present"
    )

    print(
        f"✓ Saved: {output_path}"
    )

    return output_path


# ============================================================
# STEP 4 — RETRAIN MODEL
# ============================================================

@step
def retrain_model(
    new_processed_path: str,
) -> str:

    print("\n==========================================")
    print("ZENML RETRAINING STEP")
    print("==========================================")

    # Pass the processed new-data path directly
    # to the reusable retraining function.
    candidate_path = retrain(
        new_processed_path
    )

    if not os.path.exists(
        candidate_path
    ):
        raise FileNotFoundError(
            "Candidate model was not created."
        )

    print(
        f"✓ Candidate model: {candidate_path}"
    )

    return candidate_path


# ============================================================
# STEP 5 — EVALUATE CANDIDATE
# ============================================================

@step
def evaluate_candidate_model(
    candidate_model_path: str,
) -> dict:

    print("\n==========================================")
    print("ZENML EVALUATION STEP")
    print("==========================================")

    if not os.path.exists(
        candidate_model_path
    ):
        raise FileNotFoundError(
            f"Candidate model not found: "
            f"{candidate_model_path}"
        )

    # evaluate.py performs the complete evaluation:
    #
    # Candidate model
    #        +
    # Production model
    #        +
    # Held-out dataset
    #
    # It returns the evaluation results.

    evaluation_results = evaluate_candidate()

    print(
        "\n✓ Candidate and production evaluation completed."
    )

    return evaluation_results


# ============================================================
# STEP 6 — PROMOTE OR REJECT
# ============================================================

@step
def promote_candidate(
    evaluation_results: dict,
) -> str:

    print("\n==========================================")
    print("ZENML PROMOTION STEP")
    print("==========================================")

    # promote.py makes the final decision using
    # the evaluation results.
    decision = compare_and_promote(
        evaluation_results
    )

    print(
        f"\n✓ Final promotion decision: {decision}"
    )

    return decision


# ============================================================
# ZENML PIPELINE
# ============================================================

@pipeline
def retraining_pipeline(
    new_data_path: str = NEW_DATA,
):

    # --------------------------------------------------------
    # 1. INGEST
    # --------------------------------------------------------

    ingested_data = ingest_new_data(
        new_data_path
    )

    # --------------------------------------------------------
    # 2. VALIDATE
    # --------------------------------------------------------

    validated_data = validate_new_data(
        ingested_data
    )

    # --------------------------------------------------------
    # 3. TRANSFORM
    # --------------------------------------------------------

    transformed_data = transform_new_data(
        validated_data
    )

    # --------------------------------------------------------
    # 4. RETRAIN + OPTUNA
    # --------------------------------------------------------

    candidate_model = retrain_model(
        transformed_data
    )

    # --------------------------------------------------------
    # 5. EVALUATE
    # --------------------------------------------------------

    evaluation_results = (
        evaluate_candidate_model(
            candidate_model
        )
    )

    # --------------------------------------------------------
    # 6. PROMOTE OR REJECT
    # --------------------------------------------------------

    promote_candidate(
        evaluation_results
    )


# ============================================================
# RUN PIPELINE
# ============================================================

if __name__ == "__main__":

    print(
        "\nStarting ZenML retraining pipeline..."
    )

    retraining_pipeline(
        new_data_path=NEW_DATA
    )

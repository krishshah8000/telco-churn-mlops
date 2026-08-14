import os
import json
import joblib
import mlflow
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

CANDIDATE_MODEL_PATH = (
    "models/candidate_model.joblib"
)

PRODUCTION_MODEL_PATH = (
    "models/production_model.joblib"
)

HELD_OUT_PATH = (
    "artifacts/preprocessing/held_out_processed.csv"
)

EVALUATION_OUTPUT_PATH = (
    "models/candidate_evaluation_metrics.json"
)

MLFLOW_EXPERIMENT = (
    "telco-churn-production-evaluation"
)


# ============================================================
# MLFLOW
# ============================================================

mlflow.set_experiment(
    MLFLOW_EXPERIMENT
)


# ============================================================
# LOAD HELD-OUT DATA
# ============================================================

def load_held_out_data():

    if not os.path.exists(
        HELD_OUT_PATH
    ):
        raise FileNotFoundError(
            f"Held-out data not found: "
            f"{HELD_OUT_PATH}"
        )

    df = pd.read_csv(
        HELD_OUT_PATH
    )

    if "Churn" not in df.columns:
        raise ValueError(
            "Churn target not found in held-out data."
        )

    X = df.drop(
        columns=["Churn"]
    )

    y = df["Churn"].astype(int)

    return X, y


# ============================================================
# GENERIC MODEL EVALUATION
# ============================================================

def evaluate_model(
    model,
    X,
    y,
):

    predictions = model.predict(
        X
    )

    probabilities = model.predict_proba(
        X
    )[:, 1]

    metrics = {

        "accuracy": accuracy_score(
            y,
            predictions,
        ),

        "precision": precision_score(
            y,
            predictions,
            zero_division=0,
        ),

        "recall": recall_score(
            y,
            predictions,
            zero_division=0,
        ),

        "f1_score": f1_score(
            y,
            predictions,
            zero_division=0,
        ),

        "roc_auc": roc_auc_score(
            y,
            probabilities,
        ),
    }

    return metrics


# ============================================================
# PRINT METRICS
# ============================================================

def print_metrics(
    title,
    metrics,
):

    print(
        f"\n===== {title} ====="
    )

    for metric, value in metrics.items():

        print(
            f"{metric}: {value:.4f}"
        )


# ============================================================
# EVALUATE CANDIDATE AND PRODUCTION
# ============================================================

def evaluate_candidate():

    print("\n==========================================")
    print("HELD-OUT EVALUATION")
    print("==========================================")

    # --------------------------------------------------------
    # Load held-out data
    # --------------------------------------------------------

    X_test, y_test = load_held_out_data()

    print(
        f"Held-out rows: {len(X_test)}"
    )

    # --------------------------------------------------------
    # Check candidate
    # --------------------------------------------------------

    if not os.path.exists(
        CANDIDATE_MODEL_PATH
    ):
        raise FileNotFoundError(
            f"Candidate model not found: "
            f"{CANDIDATE_MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Check production
    # --------------------------------------------------------

    if not os.path.exists(
        PRODUCTION_MODEL_PATH
    ):
        raise FileNotFoundError(
            f"Production model not found: "
            f"{PRODUCTION_MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    candidate_model = joblib.load(
        CANDIDATE_MODEL_PATH
    )

    production_model = joblib.load(
        PRODUCTION_MODEL_PATH
    )

    print(
        f"Candidate model loaded: "
        f"{CANDIDATE_MODEL_PATH}"
    )

    print(
        f"Production model loaded: "
        f"{PRODUCTION_MODEL_PATH}"
    )

    # --------------------------------------------------------
    # Evaluate candidate
    # --------------------------------------------------------

    candidate_metrics = evaluate_model(
        candidate_model,
        X_test,
        y_test,
    )

    print_metrics(
        "CANDIDATE - HELD-OUT METRICS",
        candidate_metrics,
    )

    # --------------------------------------------------------
    # Evaluate production
    # --------------------------------------------------------

    production_metrics = evaluate_model(
        production_model,
        X_test,
        y_test,
    )

    print_metrics(
        "PRODUCTION - HELD-OUT METRICS",
        production_metrics,
    )

    # --------------------------------------------------------
    # Compare
    # --------------------------------------------------------

    candidate_f1 = (
        candidate_metrics["f1_score"]
    )

    production_f1 = (
        production_metrics["f1_score"]
    )

    improvement = (
        candidate_f1
        - production_f1
    )

    candidate_better = (
        candidate_f1 > production_f1
    )

    print(
        "\n=========================================="
    )

    print(
        "HELD-OUT F1 COMPARISON"
    )

    print(
        "=========================================="
    )

    print(
        f"Production F1 : {production_f1:.4f}"
    )

    print(
        f"Candidate F1  : {candidate_f1:.4f}"
    )

    print(
        f"Improvement    : {improvement:+.4f}"
    )

    if candidate_better:

        print(
            "\n✓ Candidate has better held-out F1."
        )

        print(
            "→ Candidate is eligible for promotion."
        )

    else:

        print(
            "\n✗ Candidate does not have better "
            "held-out F1."
        )

        print(
            "→ Candidate should NOT be promoted."
        )

    # --------------------------------------------------------
    # Create evaluation result
    # --------------------------------------------------------

    evaluation = {

        "candidate_metrics": candidate_metrics,

        "production_metrics": production_metrics,

        "f1_improvement": improvement,

        "candidate_better": candidate_better,
    }

    # --------------------------------------------------------
    # Save evaluation result
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(
            EVALUATION_OUTPUT_PATH
        ),
        exist_ok=True,
    )

    with open(
        EVALUATION_OUTPUT_PATH,
        "w",
    ) as file:

        json.dump(
            evaluation,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # MLflow
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name="candidate_held_out_evaluation"
    ):

        mlflow.set_tag(
            "run_type",
            "held_out_evaluation",
        )

        mlflow.set_tag(
            "model_type",
            "XGBClassifier",
        )

        mlflow.log_metrics(
            {
                f"candidate_{key}": value
                for key, value
                in candidate_metrics.items()
            }
        )

        mlflow.log_metrics(
            {
                f"production_{key}": value
                for key, value
                in production_metrics.items()
            }
        )

        mlflow.log_metric(
            "f1_improvement",
            improvement,
        )

        mlflow.log_param(
            "candidate_better",
            candidate_better,
        )

        mlflow.log_artifact(
            EVALUATION_OUTPUT_PATH
        )

    print(
        "\n✓ Evaluation results saved:"
    )

    print(
        EVALUATION_OUTPUT_PATH
    )

    print(
        "\n✓ Production model was NOT modified."
    )

    return evaluation


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    evaluate_candidate()

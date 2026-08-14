import os
import json
import shutil
import mlflow


# ============================================================
# CONFIGURATION
# ============================================================

CANDIDATE_MODEL_PATH = (
    "models/candidate_model.joblib"
)

PRODUCTION_MODEL_PATH = (
    "models/production_model.joblib"
)

PRODUCTION_METRICS_PATH = (
    "models/production_metrics.json"
)

PROMOTION_RECORD_PATH = (
    "models/promotion_record.json"
)

MLFLOW_EXPERIMENT = (
    "telco-churn-model-promotion"
)


# ============================================================
# MLFLOW
# ============================================================

mlflow.set_experiment(
    MLFLOW_EXPERIMENT
)


# ============================================================
# SAVE PROMOTION RECORD
# ============================================================

def save_promotion_record(
    candidate_metrics,
    production_metrics,
    decision,
):

    if production_metrics is None:

        improvement = None

    else:

        improvement = (
            candidate_metrics["f1_score"]
            - production_metrics["f1_score"]
        )

    record = {
        "decision": decision,
        "candidate_metrics": candidate_metrics,
        "production_metrics": production_metrics,
        "f1_improvement": improvement,
    }

    with open(
        PROMOTION_RECORD_PATH,
        "w",
    ) as file:

        json.dump(
            record,
            file,
            indent=4,
        )


# ============================================================
# PROMOTION LOGIC
# ============================================================

def compare_and_promote(
    evaluation_results,
):

    print("\n==========================================")
    print("MODEL PROMOTION")
    print("==========================================")

    # --------------------------------------------------------
    # Extract evaluation results
    # --------------------------------------------------------

    candidate_metrics = evaluation_results[
        "candidate_metrics"
    ]

    production_metrics = evaluation_results[
        "production_metrics"
    ]

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

    print(
        f"\nCandidate F1  : {candidate_f1:.4f}"
    )

    print(
        f"Production F1 : {production_f1:.4f}"
    )

    print(
        f"Improvement   : {improvement:+.4f}"
    )

    # ========================================================
    # DECISION
    # ========================================================

    if candidate_f1 > production_f1:

        decision = "PROMOTE"

        print(
            "\n✓ Candidate F1 is better."
        )

        print(
            "✓ Candidate will be promoted."
        )

    else:

        decision = "REJECT"

        print(
            "\n✗ Candidate F1 is not better."
        )

        print(
            "✗ Existing production model "
            "will be retained."
        )

    # ========================================================
    # PROMOTION
    # ========================================================

    if decision == "PROMOTE":

        if not os.path.exists(
            CANDIDATE_MODEL_PATH
        ):

            raise FileNotFoundError(
                f"Candidate model not found: "
                f"{CANDIDATE_MODEL_PATH}"
            )

        # ----------------------------------------------------
        # Replace production model
        # ----------------------------------------------------

        shutil.copy2(
            CANDIDATE_MODEL_PATH,
            PRODUCTION_MODEL_PATH,
        )

        # ----------------------------------------------------
        # Save production metrics
        # ----------------------------------------------------

        with open(
            PRODUCTION_METRICS_PATH,
            "w",
        ) as file:

            json.dump(
                {
                    "model_name": "production_model",
                    "metrics": candidate_metrics,
                },
                file,
                indent=4,
            )

        print(
            "\n=========================================="
        )

        print(
            "✓ MODEL PROMOTED TO PRODUCTION"
        )

        print(
            "=========================================="
        )

    else:

        print(
            "\n=========================================="
        )

        print(
            "✗ CANDIDATE REJECTED"
        )

        print(
            "✓ PRODUCTION MODEL RETAINED"
        )

        print(
            "=========================================="
        )

    # ========================================================
    # SAVE PROMOTION RECORD
    # ========================================================

    save_promotion_record(
        candidate_metrics,
        production_metrics,
        decision,
    )

    # ========================================================
    # MLFLOW
    # ========================================================

    with mlflow.start_run(
        run_name="model_promotion_decision"
    ):

        mlflow.set_tag(
            "run_type",
            "model_promotion",
        )

        mlflow.set_tag(
            "decision",
            decision,
        )

        mlflow.set_tag(
            "comparison_metric",
            "f1_score",
        )

        # Candidate metrics
        mlflow.log_metrics(
            {
                f"candidate_{key}": value
                for key, value
                in candidate_metrics.items()
            }
        )

        # Production metrics
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
            "promotion_decision",
            decision,
        )

        mlflow.log_artifact(
            PROMOTION_RECORD_PATH
        )

        if decision == "PROMOTE":

            mlflow.log_artifact(
                PRODUCTION_MODEL_PATH
            )

    print(
        f"\n✓ Promotion decision: {decision}"
    )

    return decision


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    print(
        "promote.py is designed to be called "
        "from the ZenML pipeline."
    )

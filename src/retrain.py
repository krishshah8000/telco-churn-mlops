import os
import json
import joblib
import mlflow
import optuna
import pandas as pd

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
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

INITIAL_DATA_PATH = (
    "artifacts/preprocessing/initial_processed.csv"
)

MODEL_OUTPUT_PATH = (
    "models/candidate_model.joblib"
)

OPTUNA_DIR = (
    "artifacts/optuna_retraining"
)

MLFLOW_EXPERIMENT = (
    "telco-churn-xgboost-retraining"
)

N_TRIALS = 20
RANDOM_STATE = 42


# ============================================================
# DIRECTORIES
# ============================================================

os.makedirs(
    OPTUNA_DIR,
    exist_ok=True,
)

os.makedirs(
    "models",
    exist_ok=True,
)


# ============================================================
# MLFLOW
# ============================================================

mlflow.set_experiment(
    MLFLOW_EXPERIMENT
)


# ============================================================
# LOAD TRAINING DATA
# ============================================================

def load_training_data(
    new_processed_path,
):

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not os.path.exists(
        INITIAL_DATA_PATH
    ):
        raise FileNotFoundError(
            f"Initial processed data not found: "
            f"{INITIAL_DATA_PATH}"
        )

    if not os.path.exists(
        new_processed_path
    ):
        raise FileNotFoundError(
            f"New processed data not found: "
            f"{new_processed_path}"
        )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    initial_df = pd.read_csv(
        INITIAL_DATA_PATH
    )

    new_df = pd.read_csv(
        new_processed_path
    )

    print("\n===== RETRAINING DATA =====")

    print(
        f"Initial rows : {len(initial_df)}"
    )

    print(
        f"New rows     : {len(new_df)}"
    )

    # --------------------------------------------------------
    # Combine initial + new data
    # --------------------------------------------------------

    combined_df = pd.concat(
        [
            initial_df,
            new_df,
        ],
        ignore_index=True,
    )

    print(
        f"Combined rows: {len(combined_df)}"
    )

    return combined_df


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(
    df,
):

    if "Churn" not in df.columns:
        raise ValueError(
            "Churn column not found in processed data."
        )

    X = df.drop(
        columns=["Churn"]
    )

    y = df["Churn"].astype(int)

    return X, y


# ============================================================
# OPTUNA OBJECTIVE
# ============================================================

def create_objective(
    X_train,
    X_valid,
    y_train,
    y_valid,
):

    def objective(
        trial,
    ):

        params = {

            "n_estimators": trial.suggest_int(
                "n_estimators",
                100,
                500,
            ),

            "max_depth": trial.suggest_int(
                "max_depth",
                3,
                10,
            ),

            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.01,
                0.30,
                log=True,
            ),

            "subsample": trial.suggest_float(
                "subsample",
                0.6,
                1.0,
            ),

            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                0.6,
                1.0,
            ),

            "min_child_weight": trial.suggest_int(
                "min_child_weight",
                1,
                10,
            ),

            "gamma": trial.suggest_float(
                "gamma",
                0.0,
                5.0,
            ),

            "reg_alpha": trial.suggest_float(
                "reg_alpha",
                0.0,
                5.0,
            ),

            "reg_lambda": trial.suggest_float(
                "reg_lambda",
                0.1,
                10.0,
                log=True,
            ),

            "scale_pos_weight": trial.suggest_float(
                "scale_pos_weight",
                1.0,
                5.0,
            ),
        }

        # ----------------------------------------------------
        # Create model
        # ----------------------------------------------------

        model = XGBClassifier(
            **params,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        model.fit(
            X_train,
            y_train,
        )

        # ----------------------------------------------------
        # Validation prediction
        # ----------------------------------------------------

        predictions = model.predict(
            X_valid
        )

        f1 = f1_score(
            y_valid,
            predictions,
            zero_division=0,
        )

        # ----------------------------------------------------
        # MLflow — one run per Optuna trial
        # ----------------------------------------------------

        with mlflow.start_run(
            run_name=(
                f"retraining_trial_"
                f"{trial.number:02d}"
            )
        ):

            mlflow.set_tag(
                "run_type",
                "retraining_optuna_trial",
            )

            mlflow.set_tag(
                "model_type",
                "XGBClassifier",
            )

            mlflow.set_tag(
                "trial_number",
                trial.number,
            )

            mlflow.log_params(
                params
            )

            mlflow.log_metric(
                "validation_f1",
                f1,
            )

            mlflow.log_param(
                "training_rows",
                len(X_train),
            )

            mlflow.log_param(
                "validation_rows",
                len(X_valid),
            )

        # ----------------------------------------------------
        # Store trial information
        # ----------------------------------------------------

        trial.set_user_attr(
            "validation_f1",
            f1,
        )

        return f1

    return objective


# ============================================================
# MAIN RETRAINING
# ============================================================

def retrain(
    new_processed_path,
):

    print("\n==========================================")
    print("STARTING NEW-DATA RETRAINING")
    print("==========================================")

    # --------------------------------------------------------
    # Load initial + new processed data
    # --------------------------------------------------------

    df = load_training_data(
        new_processed_path
    )

    X, y = prepare_features(
        df
    )

    print(
        f"Feature columns: {X.shape[1]}"
    )

    # --------------------------------------------------------
    # Train / validation split
    # --------------------------------------------------------

    X_train, X_valid, y_train, y_valid = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    print(
        f"Training rows   : {len(X_train)}"
    )

    print(
        f"Validation rows : {len(X_valid)}"
    )

    # --------------------------------------------------------
    # Optuna
    # --------------------------------------------------------

    print("\n==========================================")
    print("STARTING OPTUNA RETRAINING")
    print("==========================================")

    study = optuna.create_study(
        direction="maximize",
        study_name=(
            "telco_churn_retraining_study"
        ),
    )

    objective = create_objective(
        X_train,
        X_valid,
        y_train,
        y_valid,
    )

    study.optimize(
        objective,
        n_trials=N_TRIALS,
    )

    # --------------------------------------------------------
    # Best trial
    # --------------------------------------------------------

    best_trial = study.best_trial

    print("\n==========================================")
    print("BEST RETRAINING TRIAL")
    print("==========================================")

    print(
        f"Best trial : {best_trial.number}"
    )

    print(
        f"Best F1    : {best_trial.value:.4f}"
    )

    print(
        "Best parameters:"
    )

    for key, value in study.best_params.items():

        print(
            f"  {key}: {value}"
        )

    # --------------------------------------------------------
    # Save best parameters
    # --------------------------------------------------------

    best_params_path = os.path.join(
        OPTUNA_DIR,
        "best_params.json",
    )

    with open(
        best_params_path,
        "w",
    ) as file:

        json.dump(
            study.best_params,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # Train final candidate model
    # --------------------------------------------------------

    print("\n==========================================")
    print("TRAINING FINAL CANDIDATE MODEL")
    print("==========================================")

    final_model = XGBClassifier(
        **study.best_params,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    final_model.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # Final validation predictions
    # --------------------------------------------------------

    predictions = final_model.predict(
        X_valid
    )

    probabilities = final_model.predict_proba(
        X_valid
    )[:, 1]

    # --------------------------------------------------------
    # Final validation metrics
    # --------------------------------------------------------

    metrics = {

        "accuracy": accuracy_score(
            y_valid,
            predictions,
        ),

        "precision": precision_score(
            y_valid,
            predictions,
            zero_division=0,
        ),

        "recall": recall_score(
            y_valid,
            predictions,
            zero_division=0,
        ),

        "f1_score": f1_score(
            y_valid,
            predictions,
            zero_division=0,
        ),

        "roc_auc": roc_auc_score(
            y_valid,
            probabilities,
        ),
    }

    # --------------------------------------------------------
    # Save candidate model
    # --------------------------------------------------------

    joblib.dump(
        final_model,
        MODEL_OUTPUT_PATH,
    )

    # --------------------------------------------------------
    # Save retraining summary
    # --------------------------------------------------------

    initial_df = pd.read_csv(
        INITIAL_DATA_PATH
    )

    new_df = pd.read_csv(
        new_processed_path
    )

    summary = {

        "best_trial": best_trial.number,

        "best_params": study.best_params,

        "metrics": metrics,

        "initial_rows": len(
            initial_df
        ),

        "new_rows": len(
            new_df
        ),

        "combined_rows": len(df),
    }

    summary_path = os.path.join(
        OPTUNA_DIR,
        "retraining_summary.json",
    )

    with open(
        summary_path,
        "w",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # MLflow parent run
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name="retraining_candidate_model"
    ):

        mlflow.set_tag(
            "run_type",
            "retraining_candidate",
        )

        mlflow.set_tag(
            "model_status",
            "candidate",
        )

        mlflow.set_tag(
            "model_type",
            "XGBClassifier",
        )

        mlflow.set_tag(
            "data",
            "initial_plus_new",
        )

        mlflow.log_params(
            study.best_params
        )

        mlflow.log_metrics(
            metrics
        )

        mlflow.log_param(
            "initial_rows",
            summary["initial_rows"],
        )

        mlflow.log_param(
            "new_rows",
            summary["new_rows"],
        )

        mlflow.log_param(
            "combined_rows",
            summary["combined_rows"],
        )

        mlflow.log_param(
            "best_trial",
            best_trial.number,
        )

        mlflow.log_artifact(
            MODEL_OUTPUT_PATH
        )

        mlflow.log_artifact(
            best_params_path
        )

        mlflow.log_artifact(
            summary_path
        )

        mlflow.xgboost.log_model(
            final_model,
            name="candidate_model_retrained",
        )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\n==========================================")
    print("RETRAINING COMPLETE")
    print("==========================================")

    print(
        f"Candidate model: "
        f"{MODEL_OUTPUT_PATH}"
    )

    print(
        f"Best trial: "
        f"{best_trial.number}"
    )

    print("\n===== FINAL VALIDATION METRICS =====")

    for metric, value in metrics.items():

        print(
            f"{metric}: {value:.4f}"
        )

    print(
        f"\n✓ {N_TRIALS} Optuna trials completed."
    )

    print(
        "✓ New candidate model created."
    )

    print(
        "✓ Production model was NOT changed."
    )

    # --------------------------------------------------------
    # Return candidate path to ZenML
    # --------------------------------------------------------

    return MODEL_OUTPUT_PATH


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    # Direct execution is still supported.
    # The default path is the normal transformed
    # new-data location.

    retrain(
        "artifacts/preprocessing/new_processed.csv"
    )

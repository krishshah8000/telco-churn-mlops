import os
import json
import joblib
import mlflow
import mlflow.xgboost
import optuna
import pandas as pd

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "artifacts/preprocessing/initial_processed.csv"

MODEL_DIR = "models"
OPTUNA_DIR = "artifacts/optuna"

MODEL_PATH = "models/candidate_model.joblib"

BEST_PARAMS_PATH = "artifacts/optuna/best_params.json"

STUDY_PATH = "artifacts/optuna/optuna_study.joblib"

TRIALS_CSV_PATH = "artifacts/optuna/trials_summary.csv"

MLFLOW_EXPERIMENT = "telco-churn-xgboost-optuna"

N_TRIALS = 20

RANDOM_STATE = 42


# ============================================================
# MLflow
# ============================================================

mlflow.set_experiment(MLFLOW_EXPERIMENT)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["Churn"])

    y = df["Churn"].astype(int)

    return X, y


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    y_probability,
):

    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),

        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),

        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),

        "f1_score": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),

        "roc_auc": roc_auc_score(
            y_true,
            y_probability,
        ),
    }


# ============================================================
# OPTUNA OBJECTIVE
# ============================================================

def objective(trial):

    X, y = load_data()

    # --------------------------------------------------------
    # Stratified split
    # --------------------------------------------------------

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # --------------------------------------------------------
    # XGBoost hyperparameter search space
    # --------------------------------------------------------

    params = {
        "n_estimators": trial.suggest_int(
            "n_estimators",
            100,
            500,
            step=50,
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

        "min_child_weight": trial.suggest_int(
            "min_child_weight",
            1,
            10,
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

        "gamma": trial.suggest_float(
            "gamma",
            0.0,
            5.0,
        ),

        "reg_alpha": trial.suggest_float(
            "reg_alpha",
            1e-8,
            10.0,
            log=True,
        ),

        "reg_lambda": trial.suggest_float(
            "reg_lambda",
            1e-8,
            10.0,
            log=True,
        ),

        # Important for our imbalanced churn dataset
        "scale_pos_weight": trial.suggest_float(
            "scale_pos_weight",
            1.0,
            5.0,
        ),

        "objective": "binary:logistic",

        "eval_metric": "logloss",

        "random_state": RANDOM_STATE,

        "n_jobs": -1,
    }

    # --------------------------------------------------------
    # Individual MLflow run
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name=f"optuna_trial_{trial.number:02d}",
        nested=True,
    ):

        mlflow.set_tag(
            "run_type",
            "optuna_trial",
        )

        mlflow.set_tag(
            "model_type",
            "XGBClassifier",
        )

        mlflow.set_tag(
            "trial_number",
            str(trial.number),
        )

        mlflow.set_tag(
            "optimization_metric",
            "f1_score",
        )

        # ----------------------------------------------------
        # Log hyperparameters
        # ----------------------------------------------------

        mlflow.log_params(params)

        # ----------------------------------------------------
        # Train XGBoost
        # ----------------------------------------------------

        model = XGBClassifier(
            **params
        )

        model.fit(
            X_train,
            y_train,
        )

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        y_pred = model.predict(
            X_valid
        )

        y_probability = model.predict_proba(
            X_valid
        )[:, 1]

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        metrics = calculate_metrics(
            y_valid,
            y_pred,
            y_probability,
        )

        mlflow.log_metrics(
            metrics
        )

        # ----------------------------------------------------
        # Optuna objective
        # ----------------------------------------------------

        objective_value = metrics["f1_score"]

        trial.set_user_attr(
            "accuracy",
            metrics["accuracy"],
        )

        trial.set_user_attr(
            "precision",
            metrics["precision"],
        )

        trial.set_user_attr(
            "recall",
            metrics["recall"],
        )

        trial.set_user_attr(
            "f1_score",
            metrics["f1_score"],
        )

        trial.set_user_attr(
            "roc_auc",
            metrics["roc_auc"],
        )

        # ----------------------------------------------------
        # Save individual trial information
        # ----------------------------------------------------

        trial_info = {
            "trial_number": trial.number,
            "parameters": params,
            "metrics": metrics,
            "objective": objective_value,
        }

        os.makedirs(
            OPTUNA_DIR,
            exist_ok=True,
        )

        trial_file = os.path.join(
            OPTUNA_DIR,
            f"trial_{trial.number:02d}.json",
        )

        with open(
            trial_file,
            "w",
        ) as file:

            json.dump(
                trial_info,
                file,
                indent=4,
                default=str,
            )

        mlflow.log_artifact(
            trial_file
        )

        print(
            f"Trial {trial.number:02d} | "
            f"F1 = {objective_value:.4f} | "
            f"ROC-AUC = {metrics['roc_auc']:.4f}"
        )

        return objective_value


# ============================================================
# FINAL MODEL TRAINING
# ============================================================

def train_final_model(best_params):

    X, y = load_data()

    # Use the same stratified validation split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = XGBClassifier(
        **best_params,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    # Validation metrics
    y_pred = model.predict(
        X_valid
    )

    y_probability = model.predict_proba(
        X_valid
    )[:, 1]

    metrics = calculate_metrics(
        y_valid,
        y_pred,
        y_probability,
    )

    # Save candidate model
    os.makedirs(
        MODEL_DIR,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    return model, metrics


# ============================================================
# MAIN TUNING PIPELINE
# ============================================================

def run_tuning():

    os.makedirs(
        MODEL_DIR,
        exist_ok=True,
    )

    os.makedirs(
        OPTUNA_DIR,
        exist_ok=True,
    )

    print("\n==========================================")
    print("STARTING XGBOOST + OPTUNA HYPERPARAMETER")
    print("TUNING")
    print("==========================================")

    print(
        f"Number of trials: {N_TRIALS}"
    )

    print(
        "Optimization metric: F1-score"
    )

    # --------------------------------------------------------
    # Parent MLflow run
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name="optuna_parent_run",
    ):

        mlflow.set_tag(
            "run_type",
            "optuna_parent",
        )

        mlflow.set_tag(
            "model_type",
            "XGBClassifier",
        )

        mlflow.set_tag(
            "dataset",
            "Telco Customer Churn",
        )

        mlflow.set_tag(
            "class_imbalance_handling",
            "scale_pos_weight",
        )

        mlflow.log_param(
            "n_trials",
            N_TRIALS,
        )

        mlflow.log_param(
            "optimization_metric",
            "f1_score",
        )

        mlflow.log_param(
            "random_state",
            RANDOM_STATE,
        )

        # ----------------------------------------------------
        # Create Optuna study
        # ----------------------------------------------------

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                seed=RANDOM_STATE,
            ),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
            ),
        )

        # ----------------------------------------------------
        # Run 20 trials
        # ----------------------------------------------------

        study.optimize(
            objective,
            n_trials=N_TRIALS,
            show_progress_bar=True,
        )

        # ----------------------------------------------------
        # Best trial
        # ----------------------------------------------------

        best_trial = study.best_trial

        print("\n==========================================")
        print("OPTUNA TUNING COMPLETE")
        print("==========================================")

        print(
            f"Best trial: {best_trial.number}"
        )

        print(
            f"Best F1 score: "
            f"{best_trial.value:.4f}"
        )

        print("\nBest parameters:")

        for key, value in best_trial.params.items():

            print(
                f"{key}: {value}"
            )

        # ----------------------------------------------------
        # Log best trial information
        # ----------------------------------------------------

        mlflow.log_metric(
            "best_f1_score",
            best_trial.value,
        )

        mlflow.log_param(
            "best_trial_number",
            best_trial.number,
        )

        mlflow.log_params(
            {
                f"best_{key}": value
                for key, value
                in best_trial.params.items()
            }
        )

        # ----------------------------------------------------
        # Save best parameters
        # ----------------------------------------------------

        with open(
            BEST_PARAMS_PATH,
            "w",
        ) as file:

            json.dump(
                best_trial.params,
                file,
                indent=4,
            )

        mlflow.log_artifact(
            BEST_PARAMS_PATH
        )

        # ----------------------------------------------------
        # Save complete Optuna study
        # ----------------------------------------------------

        joblib.dump(
            study,
            STUDY_PATH,
        )

        mlflow.log_artifact(
            STUDY_PATH
        )

        # ----------------------------------------------------
        # Save all trial results
        # ----------------------------------------------------

        trials_df = study.trials_dataframe()

        trials_df.to_csv(
            TRIALS_CSV_PATH,
            index=False,
        )

        mlflow.log_artifact(
            TRIALS_CSV_PATH
        )

        # ----------------------------------------------------
        # Train final candidate model
        # ----------------------------------------------------

        print("\n==========================================")
        print("TRAINING FINAL CANDIDATE MODEL")
        print("==========================================")

        model, final_metrics = train_final_model(
            best_trial.params
        )

        print("\nFinal candidate metrics:")

        for metric, value in final_metrics.items():

            print(
                f"{metric}: {value:.4f}"
            )

        # ----------------------------------------------------
        # Log final candidate metrics
        # ----------------------------------------------------

        mlflow.log_metrics(
            {
                f"candidate_{key}": value
                for key, value
                in final_metrics.items()
            }
        )

        # ----------------------------------------------------
        # Log candidate model
        # ----------------------------------------------------

        mlflow.xgboost.log_model(
            model,
            artifact_path="candidate_model",
        )

        # ----------------------------------------------------
        # Log model file
        # ----------------------------------------------------

        mlflow.log_artifact(
            MODEL_PATH
        )

        mlflow.set_tag(
            "candidate_model",
            "true",
        )

        mlflow.set_tag(
            "best_trial",
            str(best_trial.number),
        )

        print("\n==========================================")
        print("TRAINING COMPLETE")
        print("==========================================")

        print(
            f"Model saved: {MODEL_PATH}"
        )

        print(
            f"Best trial: {best_trial.number}"
        )

        print(
            f"Best F1: {best_trial.value:.4f}"
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_tuning()

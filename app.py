from pathlib import Path
import time

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import Response, JSONResponse

from pydantic import BaseModel, Field
from typing import Literal

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "production_model.joblib"
)

PREPROCESSOR_PATH = (
    BASE_DIR
    / "artifacts"
    / "preprocessing"
    / "preprocessor.joblib"
)


# ============================================================
# Load model and preprocessor
# ============================================================

try:

    model = joblib.load(MODEL_PATH)

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

except Exception as e:

    raise RuntimeError(
        f"Failed to load model or preprocessor: {e}"
    )


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description=(
        "API for predicting customer churn "
        "using the production XGBoost model."
    ),
    version="1.0.0",
)


# ============================================================
# Prometheus Metrics
# ============================================================

# 1. Total prediction requests
PREDICTION_REQUESTS = Counter(
    "prediction_requests_total",
    "Total number of prediction requests",
)


# 2. Successful predictions
SUCCESSFUL_PREDICTIONS = Counter(
    "successful_predictions_total",
    "Total number of successful predictions",
)


# 3. Failed predictions
FAILED_PREDICTIONS = Counter(
    "failed_predictions_total",
    "Total number of failed prediction requests",
)


# 4. Predictions by churn result
PREDICTION_RESULTS = Counter(
    "prediction_results_total",
    "Total predictions by churn result",
    ["churn"],
)


# 5. Prediction latency
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction request latency in seconds",
)


# 6. API health
API_UP = Gauge(
    "api_up",
    "Indicates whether the API is running",
)


# 7. Production model health
MODEL_LOADED = Gauge(
    "production_model_loaded",
    "Indicates whether the production model is loaded",
)


# 8. Prediction probability
PREDICTION_PROBABILITY = Histogram(
    "prediction_probability",
    "Distribution of predicted churn probabilities",
    buckets=(
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ),
)


# 9. Churn predictions
CHURN_PREDICTIONS = Counter(
    "churn_predictions_total",
    "Total number of customers predicted to churn",
)


# 10. Non-churn predictions
NON_CHURN_PREDICTIONS = Counter(
    "non_churn_predictions_total",
    "Total number of customers predicted not to churn",
)


# ============================================================
# Initial metric state
# ============================================================

API_UP.set(1)

if model is not None and preprocessor is not None:

    MODEL_LOADED.set(1)

else:

    MODEL_LOADED.set(0)


# ============================================================
# Count every /predict request
# ============================================================

@app.middleware("http")
async def count_prediction_requests(
    request: Request,
    call_next,
):

    if request.url.path == "/predict":

        PREDICTION_REQUESTS.inc()

    response = await call_next(request)

    return response


# ============================================================
# Validation error handler
# ============================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):

    if request.url.path == "/predict":

        FAILED_PREDICTIONS.inc()

    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
        },
    )


# ============================================================
# Request Schema
# ============================================================

class CustomerData(BaseModel):

    customerID: str

    SeniorCitizen: Literal[0, 1]

    tenure: int = Field(
        ge=0
    )

    MonthlyCharges: float = Field(
        ge=0
    )

    TotalCharges: float | None = Field(
        default=None,
        ge=0,
    )

    gender: Literal[
        "Male",
        "Female",
    ]

    Partner: Literal[
        "Yes",
        "No",
    ]

    Dependents: Literal[
        "Yes",
        "No",
    ]

    PhoneService: Literal[
        "Yes",
        "No",
    ]

    MultipleLines: Literal[
        "Yes",
        "No",
        "No phone service",
    ]

    InternetService: Literal[
        "DSL",
        "Fiber optic",
        "No",
    ]

    OnlineSecurity: Literal[
        "Yes",
        "No",
        "No internet service",
    ]

    OnlineBackup: Literal[
        "Yes",
        "No",
        "No internet service",
    ]

    DeviceProtection: Literal[
        "Yes",
        "No",
        "No internet service",
    ]

    TechSupport: Literal[
        "Yes",
        "No",
        "No internet service",
    ]

    StreamingTV: Literal[
        "Yes",
        "No",
        "No internet service",
    ]

    StreamingMovies: Literal[
        "Yes",
        "No",
        "No internet service",
    ]

    Contract: Literal[
        "Month-to-month",
        "One year",
        "Two year",
    ]

    PaperlessBilling: Literal[
        "Yes",
        "No",
    ]

    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]


# ============================================================
# Root endpoint
# ============================================================

@app.get("/")
def root():

    return {
        "message": (
            "Telco Customer Churn "
            "Prediction API"
        ),
        "status": "running",
        "model": "production_model.joblib",
        "monitoring": "Prometheus enabled",
    }


# ============================================================
# Prometheus Metrics Endpoint
# ============================================================

@app.get("/metrics")
def metrics():

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


# ============================================================
# Prediction Endpoint
# ============================================================

@app.post("/predict")
def predict(customer: CustomerData):

    start_time = time.time()

    try:

        # ----------------------------------------------------
        # Convert request into DataFrame
        # ----------------------------------------------------

        data = customer.model_dump()

        df = pd.DataFrame([data])


        # ----------------------------------------------------
        # Remove customer ID
        # ----------------------------------------------------

        df = df.drop(
            columns=["customerID"]
        )


        # ----------------------------------------------------
        # Clean TotalCharges
        # ----------------------------------------------------

        df["TotalCharges"] = (
            df["TotalCharges"]
            .astype(str)
            .str.strip()
            .replace(
                ["None", "nan"],
                pd.NA,
            )
        )

        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce",
        )


        # ----------------------------------------------------
        # Apply fitted preprocessing
        # ----------------------------------------------------

        X_processed = (
            preprocessor.transform(df)
        )


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = int(
            model.predict(
                X_processed
            )[0]
        )


        # ----------------------------------------------------
        # Prediction probability
        # ----------------------------------------------------

        probability = float(
            model.predict_proba(
                X_processed
            )[0][1]
        )


        # ----------------------------------------------------
        # Convert prediction to label
        # ----------------------------------------------------

        churn_label = (
            "Yes"
            if prediction == 1
            else "No"
        )


        # ----------------------------------------------------
        # Update successful prediction metrics
        # ----------------------------------------------------

        SUCCESSFUL_PREDICTIONS.inc()

        PREDICTION_RESULTS.labels(
            churn=churn_label
        ).inc()

        PREDICTION_PROBABILITY.observe(
            probability
        )


        if prediction == 1:

            CHURN_PREDICTIONS.inc()

        else:

            NON_CHURN_PREDICTIONS.inc()


        # ----------------------------------------------------
        # Record prediction latency
        # ----------------------------------------------------

        PREDICTION_LATENCY.observe(
            time.time() - start_time
        )


        # ----------------------------------------------------
        # API response
        # ----------------------------------------------------

        return {

            "customerID":
                customer.customerID,

            "prediction":
                prediction,

            "churn":
                churn_label,

            "churn_probability":
                round(
                    probability,
                    4,
                ),
        }


    except Exception as e:

        # ----------------------------------------------------
        # Application-level prediction failure
        # ----------------------------------------------------

        FAILED_PREDICTIONS.inc()

        PREDICTION_LATENCY.observe(
            time.time() - start_time
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Prediction failed: {str(e)}"
            ),
        )

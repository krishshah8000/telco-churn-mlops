import json
import os
import joblib

from src import promote


def test_promote_when_candidate_f1_is_better(tmp_path, monkeypatch):
    """
    Candidate should be promoted when its F1 score
    is higher than the production model.
    """

    candidate_path = tmp_path / "candidate_model.joblib"
    production_path = tmp_path / "production_model.joblib"
    production_metrics_path = tmp_path / "production_metrics.json"
    promotion_record_path = tmp_path / "promotion_record.json"

    # Dummy models
    candidate_model = {"model": "candidate"}
    production_model = {"model": "production"}

    joblib.dump(candidate_model, candidate_path)
    joblib.dump(production_model, production_path)

    # Redirect promote.py paths to temporary files
    monkeypatch.setattr(
        promote,
        "CANDIDATE_MODEL_PATH",
        str(candidate_path),
    )

    monkeypatch.setattr(
        promote,
        "PRODUCTION_MODEL_PATH",
        str(production_path),
    )

    monkeypatch.setattr(
        promote,
        "PRODUCTION_METRICS_PATH",
        str(production_metrics_path),
    )

    monkeypatch.setattr(
        promote,
        "PROMOTION_RECORD_PATH",
        str(promotion_record_path),
    )

    evaluation_results = {
        "candidate_metrics": {
            "accuracy": 0.80,
            "precision": 0.70,
            "recall": 0.75,
            "f1_score": 0.72,
            "roc_auc": 0.85,
        },
        "production_metrics": {
            "accuracy": 0.78,
            "precision": 0.65,
            "recall": 0.70,
            "f1_score": 0.68,
            "roc_auc": 0.82,
        },
    }

    decision = promote.compare_and_promote(
        evaluation_results
    )

    assert decision == "PROMOTE"

    # Production model should now contain candidate model
    promoted_model = joblib.load(
        production_path
    )

    assert promoted_model == candidate_model

    # Production metrics should be updated
    with open(
        production_metrics_path,
        "r",
    ) as file:
        metrics_data = json.load(file)

    assert (
        metrics_data["metrics"]["f1_score"]
        == 0.72
    )


def test_reject_when_candidate_f1_is_not_better(
    tmp_path,
    monkeypatch,
):
    """
    Candidate should be rejected when its F1 score
    is equal to or lower than production.
    """

    candidate_path = tmp_path / "candidate_model.joblib"
    production_path = tmp_path / "production_model.joblib"
    production_metrics_path = tmp_path / "production_metrics.json"
    promotion_record_path = tmp_path / "promotion_record.json"

    # Dummy models
    candidate_model = {"model": "candidate"}
    production_model = {"model": "production"}

    joblib.dump(candidate_model, candidate_path)
    joblib.dump(production_model, production_path)

    # Save original production model bytes
    with open(
        production_path,
        "rb",
    ) as file:
        original_production_bytes = file.read()

    # Redirect paths
    monkeypatch.setattr(
        promote,
        "CANDIDATE_MODEL_PATH",
        str(candidate_path),
    )

    monkeypatch.setattr(
        promote,
        "PRODUCTION_MODEL_PATH",
        str(production_path),
    )

    monkeypatch.setattr(
        promote,
        "PRODUCTION_METRICS_PATH",
        str(production_metrics_path),
    )

    monkeypatch.setattr(
        promote,
        "PROMOTION_RECORD_PATH",
        str(promotion_record_path),
    )

    evaluation_results = {
        "candidate_metrics": {
            "accuracy": 0.75,
            "precision": 0.60,
            "recall": 0.65,
            "f1_score": 0.62,
            "roc_auc": 0.80,
        },
        "production_metrics": {
            "accuracy": 0.78,
            "precision": 0.65,
            "recall": 0.70,
            "f1_score": 0.68,
            "roc_auc": 0.82,
        },
    }

    decision = promote.compare_and_promote(
        evaluation_results
    )

    assert decision == "REJECT"

    # Production model must remain unchanged
    with open(
        production_path,
        "rb",
    ) as file:
        current_production_bytes = file.read()

    assert (
        current_production_bytes
        == original_production_bytes
    )

    # Production metrics must NOT be created/changed
    assert not production_metrics_path.exists()

    # Promotion record should still be created
    assert promotion_record_path.exists()

    with open(
        promotion_record_path,
        "r",
    ) as file:
        record = json.load(file)

    assert record["decision"] == "REJECT"
    assert (
        record["f1_improvement"]
        == 0.62 - 0.68
    )

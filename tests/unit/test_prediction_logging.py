import pandas as pd
import pytest

from app.core.config import settings
from app.services.prediction_logger import PredictionLogger
from tests.schemas.requests import LoanApplicationRequest


@pytest.mark.unit
@pytest.mark.mlops
def test_prediction_logger_writes_csv_with_required_columns(tmp_path, monkeypatch):
    log_path = tmp_path / "prediction_log.csv"
    monkeypatch.setattr(settings, "prediction_log_path", log_path)
    monkeypatch.setattr(settings, "inference_log_dir", tmp_path)

    input_df = pd.DataFrame(
        [
            {
                "no_of_dependents": 2,
                "education": "Graduate",
                "self_employed": "No",
                "income_annum": 9600000,
                "loan_amount": 29900000,
                "loan_term": 12,
                "cibil_score": 778,
            }
        ]
    )
    predictions = [
        {
            "prediction": 1,
            "prediction_label": "Approved",
            "approval_probability": 0.97,
        }
    ]

    PredictionLogger().log(input_df, predictions)

    assert log_path.exists(), "Prediction log file was not created"
    logged_df = pd.read_csv(log_path)
    assert len(logged_df) == 1

    for col in ["prediction", "prediction_label", "approval_probability", "logged_at_utc"]:
        assert col in logged_df.columns, f"Missing column: {col}"

    assert logged_df.loc[0, "prediction"] == 1
    assert logged_df.loc[0, "prediction_label"] == "Approved"
    assert logged_df.loc[0, "approval_probability"] == pytest.approx(0.97)


@pytest.mark.unit
@pytest.mark.mlops
def test_prediction_logger_appends_to_existing_log(tmp_path, monkeypatch):
    log_path = tmp_path / "prediction_log.csv"
    monkeypatch.setattr(settings, "prediction_log_path", log_path)
    monkeypatch.setattr(settings, "inference_log_dir", tmp_path)

    input_df = pd.DataFrame([{"income_annum": 500000, "loan_amount": 100000}])
    pred1 = [{"prediction": 1, "prediction_label": "Approved", "approval_probability": 0.9}]
    pred2 = [{"prediction": 0, "prediction_label": "Rejected", "approval_probability": 0.2}]

    logger = PredictionLogger()
    logger.log(input_df, pred1)
    logger.log(input_df, pred2)

    logged_df = pd.read_csv(log_path)
    assert len(logged_df) == 2, "Logger should append rows, not overwrite"
    assert logged_df.loc[0, "prediction"] == 1
    assert logged_df.loc[1, "prediction"] == 0


@pytest.mark.docker
@pytest.mark.mlops
def test_prediction_creates_valid_loggable_response(loan_service, valid_loan_payload):
    request_model = LoanApplicationRequest(**valid_loan_payload)

    result = loan_service.predict(request_model)

    assert result.prediction in [0, 1]
    assert result.prediction_label in ["Approved", "Rejected"]
    assert 0 <= result.approval_probability <= 1

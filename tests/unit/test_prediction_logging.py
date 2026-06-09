from pathlib import Path

import pandas as pd
import pytest
from _pytest.monkeypatch import MonkeyPatch

from app.core.config import settings
from app.services.prediction_logger import PredictionLogger
from tests.schemas.requests import LoanApplicationRequest
from tests.schemas.responses import LoanPredictionResponse


@pytest.mark.unit
@pytest.mark.mlops
def test_prediction_logger_writes_csv_with_required_columns(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
):
    log_path = tmp_path / "prediction_log.csv"

    monkeypatch.setattr(settings, "prediction_log_path", log_path)
    monkeypatch.setattr(settings, "inference_log_dir", tmp_path)

    request = LoanApplicationRequest(
        no_of_dependents=2,
        education="Graduate",
        self_employed="No",
        income_annum=9600000,
        loan_amount=29900000,
        loan_term=12,
        cibil_score=778,
    )

    input_df = pd.DataFrame([request.model_dump(exclude_none=True)])

    prediction = LoanPredictionResponse(
        prediction=1,
        prediction_label="Approved",
        approval_probability=0.97,
    )

    PredictionLogger().log(
        input_df,
        predictions=[prediction.model_dump()],
    )

    assert log_path.exists(), "Prediction log file was not created"

    logged_df = pd.read_csv(log_path)

    assert len(logged_df) == 1

    for col in [
        "prediction",
        "prediction_label",
        "approval_probability",
        "logged_at_utc",
    ]:
        assert col in logged_df.columns, f"Missing column: {col}"

    assert logged_df.loc[0, "prediction"] == prediction.prediction
    assert logged_df.loc[0, "prediction_label"] == prediction.prediction_label
    assert logged_df.loc[0, "approval_probability"] == pytest.approx(
        prediction.approval_probability
    )


@pytest.mark.unit
@pytest.mark.mlops
def test_prediction_logger_appends_to_existing_log(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
):
    log_path = tmp_path / "prediction_log.csv"

    monkeypatch.setattr(settings, "prediction_log_path", log_path)
    monkeypatch.setattr(settings, "inference_log_dir", tmp_path)

    input_df = pd.DataFrame(
        [
            {
                "income_annum": 500000,
                "loan_amount": 100000,
            }
        ]
    )

    pred_approved = LoanPredictionResponse(
        prediction=1,
        prediction_label="Approved",
        approval_probability=0.9,
    )

    pred_rejected = LoanPredictionResponse(
        prediction=0,
        prediction_label="Rejected",
        approval_probability=0.2,
    )

    logger = PredictionLogger()

    logger.log(
        input_df,
        predictions=[pred_approved.model_dump()],
    )

    logger.log(
        input_df,
        predictions=[pred_rejected.model_dump()],
    )

    logged_df = pd.read_csv(log_path)

    assert len(logged_df) == 2, "Logger should append rows, not overwrite"
    assert logged_df.loc[0, "prediction"] == pred_approved.prediction
    assert logged_df.loc[1, "prediction"] == pred_rejected.prediction


@pytest.mark.docker
@pytest.mark.mlops
def test_prediction_creates_valid_loggable_response(
    loan_service,
    valid_loan_payload,
):
    request = LoanApplicationRequest(**valid_loan_payload)

    result = loan_service.predict(request)

    assert result.prediction in [0, 1]
    assert result.prediction_label in ["Approved", "Rejected"]
    assert 0 <= result.approval_probability <= 1
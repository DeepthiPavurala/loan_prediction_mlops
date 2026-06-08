"""Business logic layer for the Loan Prediction service.

Uses LoanPredictionClient for HTTP calls, adds validation,
assertions, and returns typed Pydantic response objects.
"""

from __future__ import annotations

from typing import Any

from tests.loan_prediction.loan_prediction_client import LoanPredictionClient
from tests.loan_prediction.request_client import Response
from tests.schemas.requests import LoanApplicationRequest
from tests.schemas.responses import (
    BatchPredictionResponse,
    ExpectedError,
    HealthResponse,
    LoanPredictionResponse,
)


def validate_expected_error(expected: ExpectedError, response: Response) -> None:
    """Assert that a Response matches the expected error status, detail, and structure."""
    assert response.status_code == expected.status_code, (
        f"Expected status {expected.status_code}, got {response.status_code}"
    )

    if not response.body:
        return

    if expected.detail:
        actual_detail = str(response.body.get("detail", ""))
        assert expected.detail.lower() in actual_detail.lower(), (
            f"Expected detail containing '{expected.detail}', got {actual_detail}"
        )

    if expected.status_code == 422 and isinstance(response.body.get("detail"), list):
        errors = response.body["detail"]
        assert len(errors) > 0, "422 response should contain at least one validation error"

        for error in errors:
            assert "type" in error, f"Validation error missing 'type': {error}"
            assert "loc" in error, f"Validation error missing 'loc': {error}"
            assert "msg" in error, f"Validation error missing 'msg': {error}"

        if expected.error_fields:
            actual_fields = {err["loc"][-1] for err in errors if err.get("loc")}
            for field in expected.error_fields:
                assert field in actual_fields, f"Expected field '{field}' in validation errors, got {actual_fields}"

        if expected.error_type:
            actual_types = {err.get("type") for err in errors}
            assert expected.error_type in actual_types, f"Expected error type '{expected.error_type}', got {actual_types}"


class LoanPredictionService:
    def __init__(self, transport) -> None:
        self._client = LoanPredictionClient(transport)

    def health(self, expected_error: ExpectedError | None = None) -> HealthResponse | Response:
        response = self._client.health()

        if expected_error:
            validate_expected_error(expected_error, response)
            return response

        assert response.status_code == 200
        return HealthResponse.model_validate(response.body)

    def root(self) -> dict:
        response = self._client.root()
        assert response.status_code == 200
        return response.body

    def predict(
        self,
        payload: LoanApplicationRequest | dict[str, Any] | list,
        expected_error: ExpectedError | None = None,
    ) -> LoanPredictionResponse | Response:
        json_payload = payload.model_dump(exclude_none=True) if isinstance(payload, LoanApplicationRequest) else payload
        response = self._client.predict(json_payload)

        if expected_error:
            validate_expected_error(expected_error, response)
            return response

        assert response.status_code == 200
        return LoanPredictionResponse.model_validate(response.body)

    def predict_batch(
        self,
        payload: list[LoanApplicationRequest] | list[dict] | Any,
        expected_error: ExpectedError | None = None,
    ) -> BatchPredictionResponse | Response:
        if payload and isinstance(payload, list) and isinstance(payload[0], LoanApplicationRequest):
            json_payload = [r.model_dump(exclude_none=True) for r in payload]
        else:
            json_payload = payload

        response = self._client.predict_batch(json_payload)

        if expected_error:
            validate_expected_error(expected_error, response)
            return response

        assert response.status_code == 200
        return BatchPredictionResponse.model_validate(response.body)

    def run_pipeline_step(self, step: str) -> dict:
        response = self._client.run_pipeline_step(step)
        assert response.status_code == 200
        return response.body

    def run_full_pipeline(self) -> dict:
        response = self._client.run_full_pipeline()
        assert response.status_code == 200
        return response.body

    def get_drift_report(self) -> dict:
        response = self._client.get_drift_report()
        assert response.status_code == 200
        return response.body

    def get_method_predict(self, expected_error: ExpectedError | None = None) -> Response:
        response = self._client.predict_get()

        if expected_error:
            validate_expected_error(expected_error, response)

        return response

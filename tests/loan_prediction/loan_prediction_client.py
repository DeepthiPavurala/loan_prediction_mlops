"""Thin HTTP client for the Loan Prediction service.

Calls endpoints and returns raw Response objects.
No assertions, no validation -- that's the service layer's job.
"""

from __future__ import annotations

from typing import Any

from tests.loan_prediction.request_client import RequestClient, Response


class LoanPredictionClient:
    def __init__(self, client: RequestClient) -> None:
        self.client = client

    def health(self) -> Response:
        return self.client.get(path="/health")

    def root(self) -> Response:
        return self.client.get(path="/")

    def predict(self, payload: dict[str, Any] | list) -> Response:
        return self.client.post(path="/predict", json=payload)

    def predict_batch(self, payload: list[dict[str, Any]]) -> Response:
        return self.client.post(path="/predict/batch", json=payload)

    def predict_get(self) -> Response:
        return self.client.get(path="/predict")

    def run_pipeline_step(self, step: str) -> Response:
        return self.client.post(path=f"/pipeline/{step}")

    def run_full_pipeline(self) -> Response:
        return self.client.post(path="/pipeline/full")

    def get_drift_report(self) -> Response:
        return self.client.get(path="/monitoring/drift")

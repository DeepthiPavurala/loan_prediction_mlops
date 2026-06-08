"""End-to-end test: ingest → prepare → train → evaluate → predict → drift.

Runs the complete MLOps lifecycle in a single test to verify all stages
integrate correctly. Uses in-process client so no Docker is required,
but needs the raw dataset to exist.
"""

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from tests.data.loan_payloads import guaranteed_approval_payload, guaranteed_rejection_payload

RAW_DATA_EXISTS = settings.raw_data_path.exists()


@pytest.mark.e2e
@pytest.mark.skipif(not RAW_DATA_EXISTS, reason="Raw dataset not found at data/raw/")
def test_full_lifecycle(tmp_path, monkeypatch):
    """Complete MLOps lifecycle: train → predict → monitor."""
    monkeypatch.setattr(settings, "processed_dir", tmp_path / "processed")
    monkeypatch.setattr(settings, "reference_dir", tmp_path / "reference")
    monkeypatch.setattr(settings, "model_dir", tmp_path / "models")
    monkeypatch.setattr(settings, "metrics_dir", tmp_path / "metrics")
    monkeypatch.setattr(settings, "inference_log_dir", tmp_path / "inference_logs")
    monkeypatch.setattr(settings, "reports_dir", tmp_path / "reports")

    monkeypatch.setattr(settings, "model_path", tmp_path / "models" / "loan_model_pipeline.joblib")
    monkeypatch.setattr(settings, "model_metadata_path", tmp_path / "models" / "model_metadata.json")
    monkeypatch.setattr(settings, "latest_metrics_path", tmp_path / "metrics" / "latest_metrics.json")
    monkeypatch.setattr(settings, "prediction_log_path", tmp_path / "inference_logs" / "prediction_log.csv")
    monkeypatch.setattr(settings, "reference_data_path", tmp_path / "reference" / "reference_data.csv")

    with TestClient(app) as client:
        # Step 1: Ingest
        resp = client.post("/pipeline/ingest")
        assert resp.status_code == 200, f"Ingest failed: {resp.json()}"
        assert resp.json()["status"] == "success"
        assert resp.json()["rows"] > 0

        # Step 2: Prepare
        resp = client.post("/pipeline/prepare")
        assert resp.status_code == 200, f"Prepare failed: {resp.json()}"
        assert resp.json()["status"] == "success"
        assert resp.json()["train_rows"] > 0
        assert resp.json()["test_rows"] > 0

        # Step 3: Train
        resp = client.post("/pipeline/train")
        assert resp.status_code == 200, f"Train failed: {resp.json()}"
        assert resp.json()["status"] == "success"
        assert settings.model_path.exists(), "Model artifact was not created"

        # Step 4: Evaluate
        resp = client.post("/pipeline/evaluate")
        assert resp.status_code == 200, f"Evaluate failed: {resp.json()}"
        assert resp.json()["roc_auc"] > 0.7, "Model quality is below threshold"

        # Step 5: Health check shows model ready
        resp = client.get("/health")
        assert resp.status_code == 200
        health = resp.json()
        assert health["status"] == "healthy"
        assert health["model_ready"] is True

        # Step 6: Predict (approval)
        resp = client.post("/predict", json=guaranteed_approval_payload())
        assert resp.status_code == 200
        result = resp.json()
        assert result["prediction"] == 1
        assert result["prediction_label"] == "Approved"
        assert result["approval_probability"] > 0.5
        assert result["loan_eligibility"] is not None

        # Step 7: Predict (rejection)
        resp = client.post("/predict", json=guaranteed_rejection_payload())
        assert resp.status_code == 200
        result = resp.json()
        assert result["prediction"] == 0
        assert result["prediction_label"] == "Rejected"
        assert result["approval_probability"] < 0.5

        # Step 8: Prediction log exists
        assert settings.prediction_log_path.exists(), "Prediction log was not created"

        # Step 9: Drift monitoring
        resp = client.get("/monitoring/drift")
        assert resp.status_code == 200
        drift = resp.json()
        assert drift["status"] == "success"
        assert "drift_detected" in drift
        assert "feature_results" in drift
        assert "threshold" in drift

        # Step 10: Performance monitoring
        resp = client.get("/monitoring/performance")
        assert resp.status_code == 200
        perf = resp.json()
        assert perf["status"] == "success"
        assert perf["total_predictions_served"] == 2
        assert perf["training_accuracy"] is not None

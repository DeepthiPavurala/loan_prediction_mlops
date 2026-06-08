import pandas as pd
import pytest

from app.monitoring.drift_monitor import DriftMonitor

monitor = DriftMonitor()


@pytest.mark.unit
@pytest.mark.mlops
def test_no_numeric_drift_when_distributions_are_similar():
    reference = pd.Series([700, 710, 720, 730, 740])
    current = pd.Series([702, 711, 719, 731, 739])

    result = monitor._numeric_drift(reference, current)

    assert result["test"] == "ks"
    assert result["drift_detected"] is False
    assert result["p_value"] > 0.10


@pytest.mark.unit
@pytest.mark.mlops
def test_numeric_drift_detected_when_distributions_differ():
    reference = pd.Series([700, 710, 720, 730, 740])
    current = pd.Series([400, 410, 420, 430, 440])

    result = monitor._numeric_drift(reference, current)

    assert result["test"] == "ks"
    assert result["drift_detected"] is True
    assert result["p_value"] < 0.10


@pytest.mark.unit
@pytest.mark.mlops
def test_no_categorical_drift_when_distributions_match():
    reference = pd.Series(["Graduate", "Graduate", "Not Graduate", "Graduate", "Not Graduate"])
    current = pd.Series(["Graduate", "Graduate", "Not Graduate", "Graduate", "Not Graduate"])

    result = monitor._categorical_drift(reference, current)

    assert result["test"] == "chi_square"
    assert result["drift_detected"] is False


@pytest.mark.unit
@pytest.mark.mlops
def test_categorical_drift_detected_when_distributions_differ():
    reference = pd.Series(["Graduate"] * 50 + ["Not Graduate"] * 50)
    current = pd.Series(["Graduate"] * 95 + ["Not Graduate"] * 5)

    result = monitor._categorical_drift(reference, current)

    assert result["test"] == "chi_square"
    assert result["drift_detected"] is True


@pytest.mark.docker
@pytest.mark.mlops
def test_drift_monitoring_endpoint(loan_service):
    result = loan_service.get_drift_report()

    assert result["status"] == "success"
    assert "drift_detected" in result
    assert "feature_results" in result
    assert "threshold" in result

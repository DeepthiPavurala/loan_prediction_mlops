import json
from pathlib import Path

import pytest

ARTIFACT_DIR = Path("artifacts/models")
METRICS_DIR = Path("artifacts/metrics")

POSSIBLE_METRICS_FILES = [
    METRICS_DIR / "model_metrics.json",
    METRICS_DIR / "latest_metrics.json",
    Path("reports/model_metrics.json"),
]

MINIMUM_THRESHOLDS = {
    "accuracy": 0.80,
    "precision": 0.80,
    "recall": 0.80,
    "f1_score": 0.80,
}


@pytest.mark.model
@pytest.mark.docker
def test_model_artifacts_directory_exists():
    assert ARTIFACT_DIR.exists(), "artifacts/models directory is missing"


@pytest.mark.model
@pytest.mark.docker
def test_at_least_one_model_artifact_exists():
    model_files = list(ARTIFACT_DIR.glob("*.pkl")) + list(ARTIFACT_DIR.glob("*.joblib"))

    assert model_files, "No model artifact found in artifacts/models"


@pytest.mark.model
@pytest.mark.docker
def test_metrics_directory_exists():
    assert METRICS_DIR.exists(), "artifacts/metrics directory is missing"


@pytest.mark.model
@pytest.mark.docker
def test_model_metrics_file_exists():
    assert any(path.exists() for path in POSSIBLE_METRICS_FILES), (
        "No model metrics file found. Expected one of: " + ", ".join(str(p) for p in POSSIBLE_METRICS_FILES)
    )


@pytest.mark.model
@pytest.mark.docker
def test_model_quality_metrics_meet_minimum_thresholds():
    metrics_file = next((p for p in POSSIBLE_METRICS_FILES if p.exists()), None)

    assert metrics_file is not None, "Metrics file not found"

    metrics = json.loads(metrics_file.read_text())

    for metric_name, minimum_value in MINIMUM_THRESHOLDS.items():
        if metric_name in metrics:
            assert metrics[metric_name] >= minimum_value, (
                f"{metric_name}={metrics[metric_name]} is below threshold {minimum_value}"
            )

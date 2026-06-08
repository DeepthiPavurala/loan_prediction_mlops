"""Golden output regression tests.

These tests load committed golden outputs (tests/data/golden_outputs.json)
and verify the model produces the exact same predictions. If any test fails,
it means the model's behavior has changed -- either intentionally (retrain)
or accidentally.

To regenerate golden outputs after retraining:
    python scripts/analyze_model.py
"""

import json
from pathlib import Path

import pytest

from tests.conftest import requires_model

GOLDEN_PATH = Path(__file__).parent / "data" / "golden_outputs.json"


def _load_golden_cases() -> list[dict]:
    if not GOLDEN_PATH.exists():
        pytest.skip("Golden outputs file not found. Run: python scripts/analyze_model.py")
    data = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    return data["test_cases"]


def _golden_ids() -> list[str]:
    if not GOLDEN_PATH.exists():
        return []
    data = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    return [case["name"] for case in data["test_cases"]]


GOLDEN_CASES = _load_golden_cases() if GOLDEN_PATH.exists() else []
GOLDEN_IDS = _golden_ids()


@requires_model
@pytest.mark.regression
@pytest.mark.model
@pytest.mark.parametrize("case", GOLDEN_CASES, ids=GOLDEN_IDS)
def test_golden_prediction_matches(case):
    """Verify model output exactly matches the golden expectation."""
    import joblib
    import pandas as pd

    from app.core.config import settings
    from app.pipelines.data_preparation import DataPreparationPipeline

    pipeline = joblib.load(settings.model_path)
    prep = DataPreparationPipeline()

    df = pd.DataFrame([case["input"]])
    features = prep.prepare_features(df).drop(columns=[settings.target_column], errors="ignore")

    prediction = int(pipeline.predict(features)[0])
    probability = round(float(pipeline.predict_proba(features)[0, 1]), 4)

    expected = case["expected_output"]

    assert prediction == expected["prediction"], (
        f"[{case['name']}] Expected prediction={expected['prediction']} ({expected['prediction_label']}), got {prediction}"
    )
    assert abs(probability - expected["approval_probability"]) <= 0.01, (
        f"[{case['name']}] Expected probability={expected['approval_probability']}, got {probability} (tolerance=0.01)"
    )


@requires_model
@pytest.mark.regression
@pytest.mark.model
def test_approved_cases_have_high_probability():
    """All golden approved cases should have probability > 0.5."""
    for case in GOLDEN_CASES:
        if case["expected_output"]["prediction"] == 1:
            prob = case["expected_output"]["approval_probability"]
            assert prob > 0.5, f"[{case['name']}] Approved case has low probability: {prob}"


@requires_model
@pytest.mark.regression
@pytest.mark.model
def test_rejected_cases_have_low_probability():
    """All golden rejected cases should have probability < 0.5."""
    for case in GOLDEN_CASES:
        if case["expected_output"]["prediction"] == 0:
            prob = case["expected_output"]["approval_probability"]
            assert prob < 0.5, f"[{case['name']}] Rejected case has high probability: {prob}"


@requires_model
@pytest.mark.regression
@pytest.mark.model
def test_golden_file_has_both_outcomes():
    """Golden file must contain at least one approved and one rejected case."""
    predictions = {case["expected_output"]["prediction"] for case in GOLDEN_CASES}
    assert 0 in predictions, "No rejected cases in golden file"
    assert 1 in predictions, "No approved cases in golden file"


@requires_model
@pytest.mark.regression
@pytest.mark.model
@pytest.mark.parametrize("case", GOLDEN_CASES, ids=GOLDEN_IDS)
def test_golden_eligibility_fields(case):
    """Verify golden cases include loan eligibility and review flag data."""
    expected = case["expected_output"]
    assert "needs_review" in expected, f"[{case['name']}] Missing needs_review field"
    assert "loan_eligibility" in expected, f"[{case['name']}] Missing loan_eligibility field"

    elig = expected["loan_eligibility"]
    assert elig["min_loan_amount"] > 0
    assert elig["max_loan_amount"] > 0
    assert elig["loan_income_ratio"] >= 0
    assert elig["max_allowed_ratio"] > 0

    if expected["prediction"] == 1 and elig["loan_income_ratio"] > 4.0:
        assert expected["needs_review"] is True, (
            f"[{case['name']}] Approved with ratio {elig['loan_income_ratio']} should need review"
        )


@requires_model
@pytest.mark.regression
@pytest.mark.model
def test_golden_review_flag_consistency():
    """Rejected cases should never have needs_review=True."""
    for case in GOLDEN_CASES:
        expected = case["expected_output"]
        if expected["prediction"] == 0:
            assert expected["needs_review"] is False, f"[{case['name']}] Rejected case should not need review"

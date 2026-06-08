"""Field sensitivity tests -- how individual fields affect model decisions.

These parametrized tests document the model's decision boundaries by
systematically varying one field at a time.

Requires the trained model artifact to be present.
"""

import pytest

from tests.conftest import requires_model


def _predict(payload: dict) -> tuple[int, float]:
    """Run a single payload through the model, return (prediction, probability)."""
    import joblib
    import pandas as pd

    from app.core.config import settings
    from app.pipelines.data_preparation import DataPreparationPipeline

    pipeline = joblib.load(settings.model_path)
    prep = DataPreparationPipeline()

    df = pd.DataFrame([payload])
    features = prep.prepare_features(df).drop(columns=[settings.target_column], errors="ignore")

    prediction = int(pipeline.predict(features)[0])
    probability = round(float(pipeline.predict_proba(features)[0, 1]), 4)
    return prediction, probability


def _base_approved_payload() -> dict:
    """A payload known to be approved -- strong across all dimensions."""
    return {
        "no_of_dependents": 2,
        "education": "Graduate",
        "self_employed": "No",
        "income_annum": 9600000,
        "loan_amount": 29900000,
        "loan_term": 12,
        "cibil_score": 778,
        "residential_assets_value": 2400000,
        "commercial_assets_value": 17600000,
        "luxury_assets_value": 22700000,
        "bank_asset_value": 8000000,
    }


# ---------------------------------------------------------------------------
# CIBIL Score Impact
# ---------------------------------------------------------------------------
@requires_model
@pytest.mark.sensitivity
@pytest.mark.parametrize(
    "cibil_score, expected_label",
    [
        (778, "Approved"),
        (700, "Approved"),
        (600, "Approved"),
        (500, "Rejected"),
        (400, "Rejected"),
        (350, "Rejected"),
    ],
    ids=[
        "cibil_778_approved",
        "cibil_700_approved",
        "cibil_600_approved",
        "cibil_500_rejected",
        "cibil_400_rejected",
        "cibil_350_rejected",
    ],
)
def test_cibil_score_impact(cibil_score, expected_label):
    """Varying CIBIL score on a strong base payload -- shows the CIBIL threshold."""
    payload = _base_approved_payload()
    payload["cibil_score"] = cibil_score
    prediction, _ = _predict(payload)
    actual_label = "Approved" if prediction == 1 else "Rejected"
    assert actual_label == expected_label, f"CIBIL {cibil_score}: expected {expected_label}, got {actual_label}"


# ---------------------------------------------------------------------------
# High CIBIL Does NOT Guarantee Approval (Ratio Impact)
# ---------------------------------------------------------------------------
@requires_model
@pytest.mark.sensitivity
@pytest.mark.parametrize(
    "income, loan_amount, total_assets, expected_label",
    [
        (9600000, 29900000, 50700000, "Approved"),
        (100000, 10000000, 0, "Rejected"),
        (100000, 700000, 0, "Rejected"),
        (100000, 70000, 0, "Rejected"),
    ],
    ids=[
        "good_ratios_approved",
        "100x_income_rejected",
        "7x_income_no_assets_rejected",
        "0.7x_income_no_assets_rejected",
    ],
)
def test_high_cibil_ratio_override(income, loan_amount, total_assets, expected_label):
    """CIBIL 800 with varying loan/income ratios -- proves ratios can override high CIBIL."""
    payload = {
        "no_of_dependents": 0,
        "education": "Graduate",
        "self_employed": "No",
        "income_annum": income,
        "loan_amount": loan_amount,
        "loan_term": 12,
        "cibil_score": 800,
        "residential_assets_value": total_assets // 4 if total_assets else 0,
        "commercial_assets_value": total_assets // 4 if total_assets else 0,
        "luxury_assets_value": total_assets // 4 if total_assets else 0,
        "bank_asset_value": total_assets // 4 if total_assets else 0,
    }
    prediction, _ = _predict(payload)
    actual_label = "Approved" if prediction == 1 else "Rejected"
    assert actual_label == expected_label, (
        f"Income={income:,}, Loan={loan_amount:,}, Assets={total_assets:,}: expected {expected_label}, got {actual_label}"
    )


# ---------------------------------------------------------------------------
# Assets Flip the Decision (same CIBIL, same ratios)
# ---------------------------------------------------------------------------
@requires_model
@pytest.mark.sensitivity
def test_assets_flip_decision():
    """Same CIBIL (610) and loan/income ratio -- adding assets flips rejection to approval."""
    base = {
        "no_of_dependents": 1,
        "education": "Graduate",
        "self_employed": "No",
        "income_annum": 4000000,
        "loan_amount": 6000000,
        "loan_term": 10,
        "cibil_score": 610,
    }

    no_assets = {
        **base,
        "residential_assets_value": 0,
        "commercial_assets_value": 0,
        "luxury_assets_value": 0,
        "bank_asset_value": 0,
    }
    with_assets = {
        **base,
        "residential_assets_value": 3000000,
        "commercial_assets_value": 2000000,
        "luxury_assets_value": 1000000,
        "bank_asset_value": 1000000,
    }

    pred_no, prob_no = _predict(no_assets)
    pred_with, prob_with = _predict(with_assets)

    assert pred_no == 0, f"Expected rejection without assets, got Approved (prob={prob_no})"
    assert pred_with == 1, f"Expected approval with assets, got Rejected (prob={prob_with})"
    assert prob_with > prob_no, "Probability should increase when assets are added"


# ---------------------------------------------------------------------------
# Low CIBIL Cannot Be Saved by Assets Alone
# ---------------------------------------------------------------------------
@requires_model
@pytest.mark.sensitivity
def test_low_cibil_not_saved_by_assets():
    """Even massive assets cannot override a very low CIBIL score (400)."""
    payload = {
        "no_of_dependents": 0,
        "education": "Graduate",
        "self_employed": "No",
        "income_annum": 8000000,
        "loan_amount": 1000000,
        "loan_term": 6,
        "cibil_score": 400,
        "residential_assets_value": 10000000,
        "commercial_assets_value": 5000000,
        "luxury_assets_value": 3000000,
        "bank_asset_value": 2000000,
    }
    prediction, probability = _predict(payload)
    assert prediction == 0, f"Low CIBIL (400) should be rejected even with massive assets, got Approved (prob={probability})"


# ---------------------------------------------------------------------------
# Education and Self-Employment Have Minimal Impact
# ---------------------------------------------------------------------------
@requires_model
@pytest.mark.sensitivity
@pytest.mark.parametrize(
    "education, self_employed",
    [
        ("Graduate", "No"),
        ("Graduate", "Yes"),
        ("Not Graduate", "No"),
        ("Not Graduate", "Yes"),
    ],
    ids=["grad_employed", "grad_self_employed", "not_grad_employed", "not_grad_self_employed"],
)
def test_education_employment_minimal_impact(education, self_employed):
    """Education and self-employment should not flip the decision on a strong payload."""
    payload = _base_approved_payload()
    payload["education"] = education
    payload["self_employed"] = self_employed
    prediction, _ = _predict(payload)
    assert prediction == 1, (
        f"education={education}, self_employed={self_employed} should not flip a strong approval payload to rejection"
    )


# ---------------------------------------------------------------------------
# Loan Eligibility and Review Flag Tests
# ---------------------------------------------------------------------------
def _predict_with_review(payload: dict) -> dict:
    """Run a payload through ModelService.predict() and return full response dict."""
    from app.services.model_service import ModelService

    svc = ModelService()
    responses = svc.predict(payload)
    return responses[0].model_dump()


@requires_model
@pytest.mark.sensitivity
def test_review_flag_triggered_for_high_ratio_approval():
    """Approved loan with ratio > 4x income should trigger needs_review."""
    payload = _base_approved_payload()
    payload["income_annum"] = 5000000
    payload["loan_amount"] = 25000000
    result = _predict_with_review(payload)
    assert result["prediction"] == 1, "Expected approval for this payload"
    assert result["needs_review"] is True, f"Loan at {25000000 / 5000000}x income should trigger review"
    assert result["review_reason"] is not None


@requires_model
@pytest.mark.sensitivity
def test_review_flag_not_triggered_for_low_ratio_approval():
    """Approved loan with ratio < 4x income should NOT trigger needs_review."""
    payload = _base_approved_payload()
    payload["income_annum"] = 9600000
    payload["loan_amount"] = 10000000
    result = _predict_with_review(payload)
    assert result["prediction"] == 1, "Expected approval for this payload"
    assert result["needs_review"] is False
    assert result["review_reason"] is None


@requires_model
@pytest.mark.sensitivity
def test_review_flag_never_on_rejected():
    """Rejected loans should never have needs_review=True regardless of ratio."""
    from tests.data.loan_payloads import guaranteed_rejection_payload

    result = _predict_with_review(guaranteed_rejection_payload())
    assert result["prediction"] == 0
    assert result["needs_review"] is False


@requires_model
@pytest.mark.sensitivity
def test_loan_eligibility_limits_calculated():
    """Verify eligibility limits are correctly computed from income."""
    payload = _base_approved_payload()
    payload["income_annum"] = 5000000
    payload["loan_amount"] = 10000000
    result = _predict_with_review(payload)

    elig = result["loan_eligibility"]
    assert elig["max_loan_amount"] == 25000000.0
    assert elig["min_loan_amount"] == 50000.0
    assert elig["loan_income_ratio"] == 2.0
    assert elig["max_allowed_ratio"] == 5.0
    assert elig["within_limits"] is True


@requires_model
@pytest.mark.sensitivity
def test_loan_outside_limits():
    """Loan exceeding max allowed ratio should show within_limits=False."""
    payload = _base_approved_payload()
    payload["income_annum"] = 1000000
    payload["loan_amount"] = 6000000
    result = _predict_with_review(payload)

    elig = result["loan_eligibility"]
    assert elig["max_loan_amount"] == 5000000.0
    assert elig["loan_income_ratio"] == 6.0
    assert elig["within_limits"] is False

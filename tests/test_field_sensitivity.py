import pytest

from tests.conftest import requires_model
from tests.schemas.requests import LoanApplicationRequest
from tests.schemas.responses import LoanPredictionResponse


def _predict(request: LoanApplicationRequest) -> LoanPredictionResponse:
    """Run a single request through the model, return validated response."""

    import joblib
    import pandas as pd

    from app.core.config import settings
    from app.pipelines.data_preparation import DataPreparationPipeline

    pipeline = joblib.load(settings.model_path)
    prep = DataPreparationPipeline()

    df = pd.DataFrame([request.model_dump(exclude_none=True)])

    features = prep.prepare_features(df).drop(
        columns=[settings.target_column],
        errors="ignore",
    )

    prediction = int(pipeline.predict(features)[0])
    probability = round(
        float(pipeline.predict_proba(features)[0, 1]),
        4,
    )

    label = "Approved" if prediction == 1 else "Rejected"

    return LoanPredictionResponse(
        prediction=prediction,
        prediction_label=label,
        approval_probability=probability,
    )


def _predict_with_review(
    request: LoanApplicationRequest,
) -> LoanPredictionResponse:
    """Run a request through ModelService.predict() and return validated response."""

    from app.services.model_service import ModelService

    svc = ModelService()

    responses = svc.predict(request.model_dump(exclude_none=True))

    return LoanPredictionResponse.model_validate(responses[0].model_dump())


def _base_approved_payload() -> LoanApplicationRequest:
    """A payload known to be approved -- strong across all dimensions."""

    return LoanApplicationRequest(
        no_of_dependents=2,
        education="Graduate",
        self_employed="No",
        income_annum=9600000,
        loan_amount=29900000,
        loan_term=12,
        cibil_score=778,
        residential_assets_value=2400000,
        commercial_assets_value=17600000,
        luxury_assets_value=22700000,
        bank_asset_value=8000000,
    )


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
def test_cibil_score_impact(
    cibil_score: int,
    expected_label: str,
):
    """Varying CIBIL score on a strong base payload -- shows the CIBIL threshold."""

    request = _base_approved_payload().model_copy(update={"cibil_score": cibil_score})

    result = _predict(request)

    assert result.prediction_label == expected_label, (
        f"CIBIL {cibil_score}: expected {expected_label}, got {result.prediction_label}"
    )


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
def test_high_cibil_ratio_override(
    income: int,
    loan_amount: int,
    total_assets: int,
    expected_label: str,
):
    """CIBIL 800 with varying loan/income ratios -- proves ratios can override high CIBIL."""

    request = LoanApplicationRequest(
        no_of_dependents=0,
        education="Graduate",
        self_employed="No",
        income_annum=income,
        loan_amount=loan_amount,
        loan_term=12,
        cibil_score=800,
        residential_assets_value=total_assets // 4 if total_assets else 0,
        commercial_assets_value=total_assets // 4 if total_assets else 0,
        luxury_assets_value=total_assets // 4 if total_assets else 0,
        bank_asset_value=total_assets // 4 if total_assets else 0,
    )

    result = _predict(request)

    assert result.prediction_label == expected_label, (
        f"Income={income:,}, "
        f"Loan={loan_amount:,}, "
        f"Assets={total_assets:,}: "
        f"expected {expected_label}, "
        f"got {result.prediction_label}"
    )


@requires_model
@pytest.mark.sensitivity
def test_assets_flip_decision():
    """Same CIBIL (610) and loan/income ratio -- adding assets flips rejection to approval."""

    base = LoanApplicationRequest(
        no_of_dependents=1,
        education="Graduate",
        self_employed="No",
        income_annum=4000000,
        loan_amount=6000000,
        loan_term=10,
        cibil_score=610,
        residential_assets_value=0,
        commercial_assets_value=0,
        luxury_assets_value=0,
        bank_asset_value=0,
    )

    no_assets_result = _predict(base)

    with_assets = base.model_copy(
        update={
            "residential_assets_value": 3000000,
            "commercial_assets_value": 2000000,
            "luxury_assets_value": 1000000,
            "bank_asset_value": 1000000,
        }
    )

    with_assets_result = _predict(with_assets)

    assert no_assets_result.prediction == 0, (
        f"Expected rejection without assets, got Approved (prob={no_assets_result.approval_probability})"
    )

    assert with_assets_result.prediction == 1, (
        f"Expected approval with assets, got Rejected (prob={with_assets_result.approval_probability})"
    )

    assert with_assets_result.approval_probability > no_assets_result.approval_probability, (
        "Probability should increase when assets are added"
    )


@requires_model
@pytest.mark.sensitivity
def test_low_cibil_not_saved_by_assets():
    """Even massive assets cannot override a very low CIBIL score (400)."""

    request = LoanApplicationRequest(
        no_of_dependents=0,
        education="Graduate",
        self_employed="No",
        income_annum=8000000,
        loan_amount=1000000,
        loan_term=6,
        cibil_score=400,
        residential_assets_value=10000000,
        commercial_assets_value=5000000,
        luxury_assets_value=3000000,
        bank_asset_value=2000000,
    )

    result = _predict(request)

    assert result.prediction == 0, (
        f"Low CIBIL (400) should be rejected even with massive assets, got Approved (prob={result.approval_probability})"
    )


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
    ids=[
        "grad_employed",
        "grad_self_employed",
        "not_grad_employed",
        "not_grad_self_employed",
    ],
)
def test_education_employment_minimal_impact(
    education: str,
    self_employed: str,
):
    """Education and self-employment should not flip the decision on a strong payload."""

    request = _base_approved_payload().model_copy(
        update={
            "education": education,
            "self_employed": self_employed,
        }
    )

    result = _predict(request)

    assert result.prediction == 1, (
        f"Education={education}, self_employed={self_employed} should not flip a strong approval payload to rejection"
    )


@requires_model
@pytest.mark.sensitivity
def test_review_flag_triggered_for_high_ratio_approval():
    """Approved loan with ratio > 4x income should trigger needs_review."""

    request = _base_approved_payload().model_copy(
        update={
            "income_annum": 500000,
            "loan_amount": 2500000,
        }
    )

    result = _predict_with_review(request)

    assert result.prediction == 1
    assert result.needs_review is True
    assert result.review_reason is not None


@requires_model
@pytest.mark.sensitivity
def test_review_flag_not_triggered_for_low_ratio_approval():
    """Approved loan with ratio < 4x income should NOT trigger needs_review."""

    request = _base_approved_payload().model_copy(
        update={
            "income_annum": 9600000,
            "loan_amount": 10000000,
        }
    )

    result = _predict_with_review(request)

    assert result.prediction == 1
    assert result.needs_review is False
    assert result.review_reason is None


@requires_model
@pytest.mark.sensitivity
def test_review_flag_never_on_rejected():
    """Rejected loans should never have needs_review=True regardless of ratio."""

    from tests.data.loan_payloads import guaranteed_rejection_payload

    request = LoanApplicationRequest(**guaranteed_rejection_payload())

    result = _predict_with_review(request)

    assert result.prediction == 0
    assert result.needs_review is False


@requires_model
@pytest.mark.sensitivity
def test_loan_eligibility_limits_calculated():
    """Verify eligibility limits are correctly computed from income."""

    request = _base_approved_payload().model_copy(
        update={
            "income_annum": 500000,
            "loan_amount": 1000000,
        }
    )

    result = _predict_with_review(request)

    assert result.loan_eligibility is not None

    elig = result.loan_eligibility

    assert elig.max_loan_amount == 2500000.0
    assert elig.min_loan_amount == 50000.0
    assert elig.loan_income_ratio == 2.0
    assert elig.max_allowed_ratio == 5.0
    assert elig.within_limits is True


@requires_model
@pytest.mark.sensitivity
def test_loan_outside_limits():
    """Loan exceeding max allowed ratio should show within_limits=False."""

    request = _base_approved_payload().model_copy(
        update={
            "income_annum": 100000,
            "loan_amount": 600000,
        }
    )

    result = _predict_with_review(request)

    assert result.loan_eligibility is not None

    elig = result.loan_eligibility

    assert elig.max_loan_amount == 500000.0
    assert elig.loan_income_ratio == 6.0
    assert elig.within_limits is False

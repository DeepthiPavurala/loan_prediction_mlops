import pytest

from tests.conftest import requires_model
from tests.data.loan_payloads import (
    invalid_boundary_payload,
    invalid_category_payload,
    missing_required_field_payload,
    second_loan_payload,
)
from tests.data.loan_payloads import (
    valid_loan_payload as get_valid_payload,
)
from tests.schemas.requests import LoanApplicationRequest
from tests.schemas.responses import ExpectedError


@requires_model
@pytest.mark.api
@pytest.mark.smoke
def test_predict_returns_success_for_valid_payload(loan_service, valid_loan_payload):
    request_model = LoanApplicationRequest(**valid_loan_payload)

    result = loan_service.predict(request_model)

    assert result.prediction in [0, 1]
    assert result.prediction_label in ["Approved", "Rejected"]
    assert 0 <= result.approval_probability <= 1


@requires_model
@pytest.mark.api
@pytest.mark.smoke
def test_batch_predict_returns_predictions(loan_service):
    first = LoanApplicationRequest(**get_valid_payload())
    second = LoanApplicationRequest(**second_loan_payload())

    result = loan_service.predict_batch([first, second])

    assert result.total_records == 2
    assert len(result.predictions) == 2

    for prediction in result.predictions:
        assert prediction.prediction in [0, 1]
        assert prediction.prediction_label in ["Approved", "Rejected"]
        assert 0 <= prediction.approval_probability <= 1


@pytest.mark.api
@pytest.mark.negative
def test_predict_rejects_empty_payload(loan_service):
    loan_service.predict(
        payload={},
        expected_error=ExpectedError(status_code=422, error_type="missing"),
    )


@pytest.mark.api
@pytest.mark.negative
def test_predict_rejects_list_payload(loan_service):
    loan_service.predict(
        payload=[],
        expected_error=ExpectedError(status_code=422),
    )


@pytest.mark.api
@pytest.mark.negative
def test_predict_get_method_not_allowed(loan_service):
    loan_service.get_method_predict(
        expected_error=ExpectedError(status_code=405),
    )


@pytest.mark.api
@pytest.mark.negative
@pytest.mark.parametrize(
    ("payload", "expected_error"),
    [
        (
            missing_required_field_payload(),
            ExpectedError(status_code=422, error_fields=["cibil_score"], error_type="missing"),
        ),
        (
            invalid_category_payload(),
            ExpectedError(status_code=422, error_fields=["education"], error_type="string_pattern_mismatch"),
        ),
        (
            invalid_boundary_payload(),
            ExpectedError(status_code=422, error_fields=["cibil_score"], error_type="less_than_equal"),
        ),
    ],
)
def test_prediction_rejects_invalid_payloads(loan_service, payload, expected_error):
    loan_service.predict(payload=payload, expected_error=expected_error)

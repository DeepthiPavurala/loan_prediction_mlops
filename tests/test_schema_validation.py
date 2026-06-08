import pytest

from tests.conftest import requires_model
from tests.data.invalid_loan_payloads import (
    EXTRA_FIELD_CASES,
    INVALID_FIELD_VALUE_CASES,
    INVALID_TYPE_CASES,
    VALID_FIELD_VALUE_CASES,
)
from tests.data.loan_payloads import MANDATORY_FIELDS, OPTIONAL_FIELDS
from tests.schemas.requests import LoanApplicationRequest
from tests.schemas.responses import ExpectedError


@requires_model
@pytest.mark.smoke
def test_predict_accepts_valid_complete_request_schema(loan_service, valid_loan_payload):
    request_model = LoanApplicationRequest(**valid_loan_payload)

    result = loan_service.predict(request_model)

    assert result.prediction in [0, 1]
    assert result.prediction_label in ["Approved", "Rejected"]
    assert 0 <= result.approval_probability <= 1


@pytest.mark.negative
@pytest.mark.parametrize("missing_field", MANDATORY_FIELDS)
def test_predict_rejects_missing_mandatory_fields(loan_service, valid_loan_payload, missing_field):
    payload = valid_loan_payload.copy()
    payload.pop(missing_field)

    loan_service.predict(
        payload=payload,
        expected_error=ExpectedError(
            status_code=422,
            error_fields=[missing_field],
            error_type="missing",
        ),
    )


@requires_model
def test_predict_accepts_missing_optional_fields(loan_service, valid_loan_payload):
    for optional_field in OPTIONAL_FIELDS:
        payload = valid_loan_payload.copy()
        payload.pop(optional_field)

        request_model = LoanApplicationRequest(**payload)
        result = loan_service.predict(request_model)

        assert result.prediction in [0, 1]


@requires_model
def test_predict_accepts_all_optional_fields_missing(loan_service, valid_loan_payload):
    payload = valid_loan_payload.copy()

    for optional_field in OPTIONAL_FIELDS:
        payload.pop(optional_field, None)

    request_model = LoanApplicationRequest(**payload)
    result = loan_service.predict(request_model)

    assert result.prediction in [0, 1]


@pytest.mark.negative
@pytest.mark.parametrize("case", EXTRA_FIELD_CASES)
def test_predict_rejects_extra_fields(loan_service, valid_loan_payload, case):
    payload = valid_loan_payload.copy()
    payload.update(case["extra_fields"])

    loan_service.predict(
        payload=payload,
        expected_error=ExpectedError(
            status_code=case["expected_status"],
            detail=case.get("expected_detail"),
            error_type=case.get("expected_error_type"),
        ),
    )


@requires_model
@pytest.mark.boundary
@pytest.mark.parametrize("case", VALID_FIELD_VALUE_CASES)
def test_predict_accepts_valid_field_values(loan_service, valid_loan_payload, case):
    payload = valid_loan_payload.copy()
    payload[case["field"]] = case["value"]

    request_model = LoanApplicationRequest(**payload)
    result = loan_service.predict(request_model)

    assert result.prediction in [0, 1]


@pytest.mark.negative
@pytest.mark.parametrize("case", INVALID_FIELD_VALUE_CASES)
def test_predict_rejects_invalid_field_values(loan_service, valid_loan_payload, case):
    payload = valid_loan_payload.copy()
    payload[case["field"]] = case["value"]

    loan_service.predict(
        payload=payload,
        expected_error=ExpectedError(
            status_code=case["expected_status"],
            error_fields=[case["field"]],
            error_type=case.get("expected_error_type"),
        ),
    )


@pytest.mark.negative
@pytest.mark.parametrize("case", INVALID_TYPE_CASES)
def test_predict_rejects_invalid_data_types(loan_service, valid_loan_payload, case):
    payload = valid_loan_payload.copy()
    payload[case["field"]] = case["value"]

    loan_service.predict(
        payload=payload,
        expected_error=ExpectedError(
            status_code=case["expected_status"],
            error_fields=[case["field"]],
            error_type=case.get("expected_error_type"),
        ),
    )

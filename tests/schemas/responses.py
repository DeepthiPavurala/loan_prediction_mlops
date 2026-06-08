from typing import Any

from pydantic import BaseModel, Field


class LoanEligibility(BaseModel):
    min_loan_amount: float
    max_loan_amount: float
    loan_income_ratio: float
    max_allowed_ratio: float
    within_limits: bool


class LoanPredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    approval_probability: float = Field(..., ge=0, le=1)
    model_version: str | None = None
    needs_review: bool = False
    review_reason: str | None = None
    loan_eligibility: LoanEligibility | None = None


class BatchPredictionResponse(BaseModel):
    predictions: list[LoanPredictionResponse]
    total_records: int = Field(..., ge=0)


class HealthResponse(BaseModel):
    status: str
    service: str | None = None
    version: str | None = None
    model_ready: bool = False
    model_path: str | None = None


class ValidationErrorDetail(BaseModel):
    type: str | None = None
    loc: list[str | int] | None = None
    msg: str | None = None
    input: Any | None = None


class ValidationErrorResponse(BaseModel):
    detail: list[ValidationErrorDetail] | str | dict


class ErrorResponse(BaseModel):
    detail: Any


class ExpectedError(BaseModel):
    status_code: int
    detail: str | None = None
    error_fields: list[str] | None = None
    error_type: str | None = None

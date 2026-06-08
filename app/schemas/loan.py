from pydantic import BaseModel, ConfigDict, Field


class LoanApplication(BaseModel):
    model_config = ConfigDict(extra="forbid")

    loan_id: int | None = None
    no_of_dependents: int = Field(..., ge=0)
    education: str = Field(..., pattern="^(Graduate|Not Graduate)$")
    self_employed: str = Field(..., pattern="^(Yes|No)$")
    income_annum: float = Field(..., gt=0)
    loan_amount: float = Field(..., gt=0)
    loan_term: int = Field(..., gt=0)
    cibil_score: int = Field(..., ge=300, le=900)

    residential_assets_value: float | None = Field(default=0, ge=0)
    commercial_assets_value: float | None = Field(default=0, ge=0)
    luxury_assets_value: float | None = Field(default=0, ge=0)
    bank_asset_value: float | None = Field(default=0, ge=0)


class LoanEligibility(BaseModel):
    min_loan_amount: float
    max_loan_amount: float
    loan_income_ratio: float
    max_allowed_ratio: float
    within_limits: bool


class LoanPredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    approval_probability: float
    model_version: str | None = None
    needs_review: bool = False
    review_reason: str | None = None
    loan_eligibility: LoanEligibility | None = None


class BatchPredictionResponse(BaseModel):
    predictions: list[LoanPredictionResponse]
    total_records: int

from pydantic import BaseModel, ConfigDict, Field


class LoanApplicationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    loan_id: int | None = Field(default=None, ge=1)

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


class BatchLoanApplicationRequest(BaseModel):
    records: list[LoanApplicationRequest]

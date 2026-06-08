from pydantic import BaseModel, Field

from app.schemas.loan import LoanApplication


class ExplainRequest(LoanApplication):
    pass


class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float


class ExplainResponse(BaseModel):
    prediction: int
    prediction_label: str
    approval_probability: float
    feature_contributions: list[FeatureImportanceItem] = Field(default_factory=list)

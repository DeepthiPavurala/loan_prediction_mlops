import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from app.core.config import settings
from app.pipelines.data_preparation import DataPreparationPipeline
from app.schemas.loan import LoanApplication, LoanEligibility, LoanPredictionResponse
from app.services.prediction_logger import PredictionLogger
from app.utils.file_reader import read_csv, read_json


class ModelService:
    def __init__(self):
        self.preparation = DataPreparationPipeline()
        self.prediction_logger = PredictionLogger()
        self._model = None
        self._metadata = None

    def _load_model(self):
        if self._model is None:
            if not settings.model_path.exists():
                raise FileNotFoundError("Model artifact not found. Run /pipeline/full first or scripts/run_pipeline.py")
            self._model = joblib.load(settings.model_path)
        return self._model

    def _load_metadata(self) -> dict:
        if self._metadata is None:
            if settings.model_metadata_path.exists():
                self._metadata = json.loads(settings.model_metadata_path.read_text(encoding="utf-8"))
            else:
                self._metadata = {"model_name": "unknown"}
        return self._metadata

    def to_dataframe(self, payload: Any) -> pd.DataFrame:
        if isinstance(payload, LoanApplication):
            return pd.DataFrame([payload.model_dump()])
        if isinstance(payload, dict):
            return pd.DataFrame([LoanApplication(**payload).model_dump()])
        if isinstance(payload, list):
            records = []
            for item in payload:
                if isinstance(item, LoanApplication):
                    records.append(item.model_dump())
                elif isinstance(item, dict):
                    records.append(LoanApplication(**item).model_dump())
                else:
                    raise TypeError("Batch payload must contain dictionaries or LoanApplication objects")
            return pd.DataFrame(records)
        if isinstance(payload, pd.DataFrame):
            return payload.copy()
        if isinstance(payload, (str, Path)):
            path = Path(payload)
            if path.suffix.lower() == ".csv":
                return read_csv(path)
            if path.suffix.lower() == ".json":
                return read_json(path)
            raise ValueError("Only CSV and JSON files are supported")
        raise TypeError("Unsupported prediction input type")

    def prepare_prediction_features(self, input_df: pd.DataFrame) -> pd.DataFrame:
        prepared = self.preparation.prepare_features(input_df)
        return prepared.drop(columns=[settings.target_column], errors="ignore")

    def _build_eligibility(self, income: float, loan_amount: float) -> LoanEligibility:
        max_loan = income * settings.max_loan_income_ratio
        min_loan = settings.min_loan_amount
        ratio = round(loan_amount / income, 4) if income > 0 else float("inf")
        return LoanEligibility(
            min_loan_amount=min_loan,
            max_loan_amount=round(max_loan, 2),
            loan_income_ratio=ratio,
            max_allowed_ratio=settings.max_loan_income_ratio,
            within_limits=min_loan <= loan_amount <= max_loan,
        )

    def _check_review(self, prediction: int, loan_income_ratio: float) -> tuple[bool, str | None]:
        if prediction == 1 and loan_income_ratio > settings.review_threshold_ratio:
            return True, (f"Loan amount exceeds {settings.review_threshold_ratio}x income (ratio: {loan_income_ratio})")
        return False, None

    def predict(self, payload: Any) -> list[LoanPredictionResponse]:
        model = self._load_model()
        metadata = self._load_metadata()
        input_df = self.to_dataframe(payload)
        features = self.prepare_prediction_features(input_df)
        predictions = model.predict(features)
        probabilities = model.predict_proba(features)[:, 1]

        response_dicts = []
        responses = []
        for idx, (pred, prob) in enumerate(zip(predictions, probabilities, strict=True)):
            pred = int(pred)
            row = input_df.iloc[idx]
            income = float(row.get("income_annum", 0))
            loan_amount = float(row.get("loan_amount", 0))

            eligibility = self._build_eligibility(income, loan_amount)
            needs_review, review_reason = self._check_review(pred, eligibility.loan_income_ratio)

            item = {
                "prediction": pred,
                "prediction_label": "Approved" if pred == 1 else "Rejected",
                "approval_probability": round(float(prob), 4),
                "model_version": metadata.get("trained_at_utc"),
                "needs_review": needs_review,
                "review_reason": review_reason,
                "loan_eligibility": eligibility.model_dump(),
            }
            response_dicts.append(item)
            responses.append(LoanPredictionResponse(**item))

        self.prediction_logger.log(input_df, response_dicts)
        return responses

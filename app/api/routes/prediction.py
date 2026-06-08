import numpy as np
from fastapi import APIRouter, Body, HTTPException

from app.schemas.loan import BatchPredictionResponse, LoanApplication, LoanPredictionResponse
from app.schemas.model import ExplainRequest, ExplainResponse, FeatureImportanceItem
from app.services.model_service import ModelService

router = APIRouter(prefix="/predict", tags=["prediction"])
model_service = ModelService()


@router.post("", response_model=LoanPredictionResponse)
def predict(application: LoanApplication = Body(...)):
    try:
        return model_service.predict(application)[0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/batch", response_model=BatchPredictionResponse)
def predict_batch(applications: list[LoanApplication] = Body(...)):
    try:
        predictions = model_service.predict(applications)
        return BatchPredictionResponse(predictions=predictions, total_records=len(predictions))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/explain", response_model=ExplainResponse)
def predict_explain(application: ExplainRequest = Body(...)):
    try:
        result = model_service.predict(LoanApplication(**application.model_dump()))[0]

        pipeline = model_service._load_model()
        classifier = pipeline.named_steps["model"]
        preprocessor = pipeline.named_steps["preprocessor"]

        input_df = model_service.to_dataframe(application.model_dump())
        features = model_service.prepare_prediction_features(input_df)
        preprocessor.transform(features)
        feature_names = preprocessor.get_feature_names_out()

        if hasattr(classifier, "feature_importances_"):
            importances = classifier.feature_importances_
        elif hasattr(classifier, "coef_"):
            importances = np.abs(classifier.coef_[0])
        else:
            importances = np.zeros(len(feature_names))

        contributions = sorted(
            [
                FeatureImportanceItem(feature=name, importance=round(float(imp), 6))
                for name, imp in zip(feature_names, importances, strict=True)
            ],
            key=lambda x: x.importance,
            reverse=True,
        )
        return ExplainResponse(
            prediction=result.prediction,
            prediction_label=result.prediction_label,
            approval_probability=result.approval_probability,
            feature_contributions=contributions,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

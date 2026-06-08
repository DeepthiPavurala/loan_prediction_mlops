from fastapi import APIRouter, HTTPException

from app.pipelines.data_ingestion import DataIngestionPipeline
from app.pipelines.data_preparation import DataPreparationPipeline
from app.pipelines.model_evaluation import ModelEvaluationPipeline
from app.pipelines.model_training import ModelTrainingPipeline

router = APIRouter(prefix="/pipeline", tags=["ml-pipeline"])


@router.post("/ingest")
def ingest_data():
    try:
        return DataIngestionPipeline().run()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/prepare")
def prepare_data():
    try:
        return DataPreparationPipeline().run()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/train")
def train_model():
    try:
        return ModelTrainingPipeline().run()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/evaluate")
def evaluate_model():
    try:
        return ModelEvaluationPipeline().run()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/full")
def run_full_pipeline():
    try:
        ingestion = DataIngestionPipeline().run()
        preparation = DataPreparationPipeline().run()
        training = ModelTrainingPipeline().run()
        evaluation = ModelEvaluationPipeline().run()
        return {
            "status": "success",
            "steps": {
                "ingestion": ingestion,
                "preparation": preparation,
                "training": training,
                "evaluation": evaluation,
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

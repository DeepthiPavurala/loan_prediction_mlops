#!/bin/bash
set -e

MODEL_PATH="artifacts/models/loan_model_pipeline.joblib"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Model not found. Running training pipeline..."
    python -c "
from app.pipelines.data_ingestion import DataIngestionPipeline
from app.pipelines.data_preparation import DataPreparationPipeline
from app.pipelines.model_training import ModelTrainingPipeline
from app.pipelines.model_evaluation import ModelEvaluationPipeline

DataIngestionPipeline().run()
DataPreparationPipeline().run()
ModelTrainingPipeline().run()
ModelEvaluationPipeline().run()
print('Pipeline completed. Model is ready.')
"
else
    echo "Model already exists at $MODEL_PATH. Skipping training."
fi

exec uvicorn app.main:app --host 0.0.0.0 --port 9000

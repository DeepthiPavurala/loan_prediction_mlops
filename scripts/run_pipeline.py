import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.pipelines.data_ingestion import DataIngestionPipeline
from app.pipelines.data_preparation import DataPreparationPipeline
from app.pipelines.model_evaluation import ModelEvaluationPipeline
from app.pipelines.model_training import ModelTrainingPipeline


def main():
    print("1. Data ingestion")
    print(DataIngestionPipeline().run())

    print("2. Data preparation")
    print(DataPreparationPipeline().run())

    print("3. Model training")
    print(ModelTrainingPipeline().run())

    print("4. Model evaluation")
    print(ModelEvaluationPipeline().run())


if __name__ == "__main__":
    main()

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.core.config import settings
from app.core.logging import get_logger
from app.utils.file_reader import read_csv

logger = get_logger(__name__)


class ModelTrainingPipeline:
    """Trains candidate models and registers the best one locally."""

    categorical_features: ClassVar[list[str]] = ["education", "self_employed"]

    def _build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        numeric_features = [col for col in X.columns if col not in self.categorical_features]
        return ColumnTransformer(
            transformers=[
                ("numeric", StandardScaler(), numeric_features),
                ("categorical", OneHotEncoder(handle_unknown="ignore"), self.categorical_features),
            ]
        )

    def _candidate_models(self) -> dict:
        return {
            "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
            "random_forest": RandomForestClassifier(
                n_estimators=200, random_state=settings.random_state, class_weight="balanced"
            ),
            "gradient_boosting": GradientBoostingClassifier(random_state=settings.random_state),
        }

    def _evaluate(self, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        y_pred = pipeline.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
        }
        if hasattr(pipeline, "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = roc_auc_score(y_test, y_prob)
        else:
            metrics["roc_auc"] = 0.0
        return {key: round(float(value), 4) for key, value in metrics.items()}

    def run(
        self,
        train_path: str | Path | None = None,
        test_path: str | Path | None = None,
    ) -> dict:
        train_path = Path(train_path) if train_path else settings.processed_dir / "train_data.csv"
        test_path = Path(test_path) if test_path else settings.processed_dir / "test_data.csv"

        train_df = read_csv(train_path)
        test_df = read_csv(test_path)
        X_train = train_df.drop(columns=[settings.target_column])
        y_train = train_df[settings.target_column]
        X_test = test_df.drop(columns=[settings.target_column])
        y_test = test_df[settings.target_column]

        results = []
        best_name = None
        best_pipeline = None
        best_metrics = None
        best_score = -1.0

        for model_name, model in self._candidate_models().items():
            pipeline = Pipeline(
                [
                    ("preprocessor", self._build_preprocessor(X_train)),
                    ("model", model),
                ]
            )
            pipeline.fit(X_train, y_train)
            metrics = self._evaluate(pipeline, X_test, y_test)
            results.append({"model_name": model_name, **metrics})

            # F1 is selected because approval/rejection errors are more business-sensitive than raw accuracy.
            if metrics["f1"] > best_score:
                best_score = metrics["f1"]
                best_name = model_name
                best_pipeline = pipeline
                best_metrics = metrics

        settings.model_dir.mkdir(parents=True, exist_ok=True)
        settings.metrics_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_pipeline, settings.model_path)

        metadata = {
            "model_name": best_name,
            "model_path": str(settings.model_path),
            "trained_at_utc": datetime.now(UTC).isoformat(),
            "selection_metric": "f1",
            "metrics": best_metrics,
            "candidate_results": results,
            "features": list(X_train.columns),
            "target_column": settings.target_column,
        }
        settings.model_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        settings.latest_metrics_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        logger.info("Training completed. Best model=%s f1=%s", best_name, best_score)
        return {"status": "success", **metadata}

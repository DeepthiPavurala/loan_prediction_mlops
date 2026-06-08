import json
from pathlib import Path

import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from app.core.config import settings
from app.core.logging import get_logger
from app.utils.file_reader import read_csv

logger = get_logger(__name__)

class ModelEvaluationPipeline:
    """Evaluates the registered model against a holdout test set."""

    def run(self, test_path: str | Path | None = None) -> dict:
        test_path = Path(test_path) if test_path else settings.processed_dir / "test_data.csv"
        if not settings.model_path.exists():
            raise FileNotFoundError("Model artifact not found. Run training first.")

        model = joblib.load(settings.model_path)
        test_df = read_csv(test_path)
        X_test = test_df.drop(columns=[settings.target_column])
        y_test = test_df[settings.target_column]
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        matrix = confusion_matrix(y_test, y_pred).tolist()
        evaluation = {
            "status": "success",
            "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
            "classification_report": report,
            "confusion_matrix": matrix,
        }

        settings.metrics_dir.mkdir(parents=True, exist_ok=True)
        output_path = settings.metrics_dir / "evaluation_report.json"
        output_path.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
        evaluation["output_path"] = str(output_path)
        return evaluation

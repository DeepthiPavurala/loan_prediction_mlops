from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from app.core.config import settings


class PredictionLogger:
    def log(self, input_df: pd.DataFrame, predictions: list[dict]) -> None:
        settings.inference_log_dir.mkdir(parents=True, exist_ok=True)
        rows = input_df.copy()
        rows["prediction"] = [item["prediction"] for item in predictions]
        rows["prediction_label"] = [item["prediction_label"] for item in predictions]
        rows["approval_probability"] = [item["approval_probability"] for item in predictions]
        rows["logged_at_utc"] = datetime.now(UTC).isoformat()

        output_path = Path(settings.prediction_log_path)
        header = not output_path.exists()
        rows.to_csv(output_path, mode="a", header=header, index=False)

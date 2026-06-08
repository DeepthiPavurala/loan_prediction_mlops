import json
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

from app.core.config import settings
from app.pipelines.data_preparation import DataPreparationPipeline
from app.utils.file_reader import read_csv


class DriftMonitor:
    """Simple backend drift monitor using KS test for numeric columns and chi-square for categorical columns."""

    categorical_columns: ClassVar[list[str]] = ["education", "self_employed"]

    def _numeric_drift(self, reference: pd.Series, current: pd.Series) -> dict:
        reference = pd.to_numeric(reference, errors="coerce").dropna()
        current = pd.to_numeric(current, errors="coerce").dropna()
        if len(reference) == 0 or len(current) == 0:
            return {"test": "ks", "p_value": None, "drift_detected": False}
        statistic, p_value = ks_2samp(reference, current)
        return {
            "test": "ks",
            "statistic": round(float(statistic), 4),
            "p_value": round(float(p_value), 4),
            "drift_detected": bool(p_value < settings.drift_threshold),
        }

    def _categorical_drift(self, reference: pd.Series, current: pd.Series) -> dict:
        categories = sorted(set(reference.dropna().unique()).union(set(current.dropna().unique())))
        if not categories:
            return {"test": "chi_square", "p_value": None, "drift_detected": False}
        ref_counts = reference.value_counts().reindex(categories, fill_value=0)
        cur_counts = current.value_counts().reindex(categories, fill_value=0)
        table = np.array([ref_counts.values, cur_counts.values])
        if (table.sum(axis=0) == 0).any():
            return {"test": "chi_square", "p_value": None, "drift_detected": False}
        statistic, p_value, _, _ = chi2_contingency(table)
        return {
            "test": "chi_square",
            "statistic": round(float(statistic), 4),
            "p_value": round(float(p_value), 4),
            "drift_detected": bool(p_value < settings.drift_threshold),
        }

    def run(
        self,
        current_path: str | Path | None = None,
        reference_path: str | Path | None = None,
    ) -> dict:
        reference_path = Path(reference_path) if reference_path else settings.reference_data_path
        current_path = Path(current_path) if current_path else settings.prediction_log_path

        if not reference_path.exists():
            raise FileNotFoundError("Reference data not found. Run data preparation first.")
        if not current_path.exists():
            raise FileNotFoundError("Current prediction log not found. Call /predict or pass current_path.")

        reference_df = read_csv(reference_path)
        current_df = read_csv(current_path)
        prepared_current = (
            DataPreparationPipeline().prepare_features(current_df).drop(columns=[settings.target_column], errors="ignore")
        )

        results = {}
        for col in reference_df.columns:
            if col not in prepared_current.columns:
                continue
            if col in self.categorical_columns:
                results[col] = self._categorical_drift(reference_df[col], prepared_current[col])
            else:
                results[col] = self._numeric_drift(reference_df[col], prepared_current[col])

        drifted_features = [col for col, result in results.items() if result.get("drift_detected")]
        report = {
            "status": "success",
            "drift_detected": bool(drifted_features),
            "drifted_features": drifted_features,
            "threshold": settings.drift_threshold,
            "feature_results": results,
        }
        settings.reports_dir.mkdir(parents=True, exist_ok=True)
        output_path = settings.reports_dir / "drift_report.json"
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["output_path"] = str(output_path)
        return report

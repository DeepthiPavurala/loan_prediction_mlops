import json

from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.monitoring.drift_monitor import DriftMonitor

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.get("/drift")
def run_drift_monitoring():
    try:
        return DriftMonitor().run()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/performance")
def model_performance():
    eval_path = settings.metrics_dir / "evaluation_report.json"
    if not eval_path.exists():
        raise HTTPException(status_code=404, detail="Evaluation report not found. Run /pipeline/evaluate first.")

    report = json.loads(eval_path.read_text(encoding="utf-8"))
    training_metrics = report.get("classification_report", {})

    result = {
        "status": "success",
        "training_accuracy": training_metrics.get("accuracy"),
        "training_roc_auc": report.get("roc_auc"),
    }

    if settings.prediction_log_path.exists():
        from app.utils.file_reader import read_csv

        log_df = read_csv(settings.prediction_log_path)
        result["total_predictions_served"] = len(log_df)
        result["live_approval_rate"] = round(float((log_df["prediction"] == 1).mean()), 4)
        result["avg_confidence"] = round(float(log_df["approval_probability"].mean()), 4)
    else:
        result["total_predictions_served"] = 0
        result["live_approval_rate"] = None
        result["avg_confidence"] = None

    drift_report_path = settings.reports_dir / "drift_report.json"
    if drift_report_path.exists():
        drift = json.loads(drift_report_path.read_text(encoding="utf-8"))
        result["drift_detected"] = drift.get("drift_detected", False)
        result["drifted_features"] = drift.get("drifted_features", [])
    else:
        result["drift_detected"] = None
        result["drifted_features"] = []

    return result

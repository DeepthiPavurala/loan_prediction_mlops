from pathlib import Path

from pydantic import BaseModel


class Settings(BaseModel):
    app_name: str = "Loan Approval MLOps Backend"
    version: str = "2.0.0"

    base_dir: Path = Path(__file__).resolve().parents[2]
    raw_data_path: Path = base_dir / "data" / "raw" / "loan_approval_dataset.csv"
    processed_dir: Path = base_dir / "data" / "processed"
    reference_dir: Path = base_dir / "data" / "reference"
    inference_log_dir: Path = base_dir / "data" / "inference_logs"
    reports_dir: Path = base_dir / "reports"
    model_dir: Path = base_dir / "artifacts" / "models"
    metrics_dir: Path = base_dir / "artifacts" / "metrics"

    model_path: Path = model_dir / "loan_model_pipeline.joblib"
    model_metadata_path: Path = model_dir / "model_metadata.json"
    latest_metrics_path: Path = metrics_dir / "latest_metrics.json"
    prediction_log_path: Path = inference_log_dir / "prediction_log.csv"
    reference_data_path: Path = reference_dir / "reference_data.csv"

    target_column: str = "loan_status"
    approval_label: str = "Approved"
    rejection_label: str = "Rejected"
    test_size: float = 0.2
    random_state: int = 42
    drift_threshold: float = 0.10

    max_loan_income_ratio: float = 5.0
    min_loan_amount: float = 50000
    review_threshold_ratio: float = 4.0

    raw_required_columns: list[str] = [
        "loan_id",
        "no_of_dependents",
        "education",
        "self_employed",
        "income_annum",
        "loan_amount",
        "loan_term",
        "cibil_score",
        "residential_assets_value",
        "commercial_assets_value",
        "luxury_assets_value",
        "bank_asset_value",
        "loan_status",
    ]

    prediction_required_columns: list[str] = [
        "no_of_dependents",
        "education",
        "self_employed",
        "income_annum",
        "loan_amount",
        "loan_term",
        "cibil_score",
    ]

    optional_asset_columns: list[str] = [
        "residential_assets_value",
        "commercial_assets_value",
        "luxury_assets_value",
        "bank_asset_value",
    ]


settings = Settings()

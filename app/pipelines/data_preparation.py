from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from app.core.config import settings
from app.core.logging import get_logger
from app.utils.file_reader import read_csv

logger = get_logger(__name__)


class DataPreparationPipeline:
    """Cleans raw data, creates mortgage-style risk features, and splits train/test/reference data."""

    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = (
            df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True).str.replace(r"_+", "_", regex=True)
        )
        return df

    def clean_string_values(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in df.select_dtypes(include=["object", "string"]).columns:
            df[col] = df[col].astype(str).str.strip()
        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        required = [*settings.prediction_required_columns, settings.target_column]
        missing_required = [col for col in required if col in df.columns and df[col].isna().any()]
        if missing_required:
            raise ValueError(f"Missing values found in required columns: {missing_required}")

        for col in settings.optional_asset_columns:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        return df

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in settings.optional_asset_columns:
            if col not in df.columns:
                df[col] = 0

        df["loan_income_ratio"] = df["loan_amount"] / df["income_annum"]
        df["total_assets"] = df[settings.optional_asset_columns].sum(axis=1)
        df["asset_coverage_ratio"] = df["total_assets"] / df["loan_amount"]
        df["net_worth"] = df["total_assets"] - df["loan_amount"]
        return df

    def encode_target(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if settings.target_column in df.columns:
            df[settings.target_column] = df[settings.target_column].map(
                {
                    settings.rejection_label: 0,
                    settings.approval_label: 1,
                    0: 0,
                    1: 1,
                }
            )
            if df[settings.target_column].isna().any():
                raise ValueError("Target column has invalid values after encoding")
        return df

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.clean_column_names(df)
        df = self.clean_string_values(df)
        df = self.handle_missing_values(df)
        df = df.drop(columns=["loan_id"], errors="ignore")
        df = self.create_features(df)
        df = self.encode_target(df)
        return df

    def run(self, input_path: str | Path | None = None) -> dict:
        input_path = Path(input_path) if input_path else settings.processed_dir / "ingested_loan_data.csv"
        if not input_path.exists():
            input_path = settings.raw_data_path

        raw_df = read_csv(input_path)
        prepared_df = self.prepare_features(raw_df)

        train_df, test_df = train_test_split(
            prepared_df,
            test_size=settings.test_size,
            random_state=settings.random_state,
            stratify=prepared_df[settings.target_column],
        )

        settings.processed_dir.mkdir(parents=True, exist_ok=True)
        settings.reference_dir.mkdir(parents=True, exist_ok=True)

        prepared_path = settings.processed_dir / "prepared_loan_data.csv"
        train_path = settings.processed_dir / "train_data.csv"
        test_path = settings.processed_dir / "test_data.csv"

        prepared_df.to_csv(prepared_path, index=False)
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        train_df.drop(columns=[settings.target_column]).to_csv(settings.reference_data_path, index=False)

        logger.info("Data preparation completed: train=%s test=%s", len(train_df), len(test_df))
        return {
            "status": "success",
            "prepared_rows": len(prepared_df),
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "prepared_path": str(prepared_path),
            "train_path": str(train_path),
            "test_path": str(test_path),
            "reference_data_path": str(settings.reference_data_path),
        }

import shutil
from pathlib import Path

from app.core.config import settings
from app.core.logging import get_logger
from app.utils.file_reader import read_csv

logger = get_logger(__name__)


class DataIngestionPipeline:
    """Loads raw loan data and stores an immutable/raw copy for the pipeline."""

    def run(self, source_path: str | Path | None = None) -> dict:
        source = Path(source_path) if source_path else settings.raw_data_path
        df = read_csv(source)
        df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

        missing = sorted(set(settings.raw_required_columns) - set(df.columns))
        if missing:
            raise ValueError(f"Raw data missing required columns: {missing}")

        settings.processed_dir.mkdir(parents=True, exist_ok=True)
        ingested_path = settings.processed_dir / "ingested_loan_data.csv"
        df.to_csv(ingested_path, index=False)

        snapshot_path = settings.processed_dir / f"raw_snapshot_rows_{len(df)}.csv"
        shutil.copyfile(ingested_path, snapshot_path)

        logger.info("Data ingestion completed: %s rows", len(df))
        return {
            "status": "success",
            "rows": len(df),
            "columns": len(df.columns),
            "output_path": str(ingested_path),
            "snapshot_path": str(snapshot_path),
        }

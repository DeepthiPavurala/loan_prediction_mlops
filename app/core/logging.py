import logging
import sys

from app.core.config import settings


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(settings.reports_dir / "app.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

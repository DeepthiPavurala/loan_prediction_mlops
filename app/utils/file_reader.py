import json
from pathlib import Path

import pandas as pd


def read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError("Only CSV files are supported")
    return pd.read_csv(path)


def read_json(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() != ".json":
        raise ValueError("Only JSON files are supported")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict):
        return pd.DataFrame([data])
    if isinstance(data, list):
        return pd.DataFrame(data)
    raise ValueError("JSON must contain an object or a list of objects")

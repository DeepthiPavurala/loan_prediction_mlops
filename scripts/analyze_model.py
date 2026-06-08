"""Introspect the trained model and generate golden output fixtures.

Run after training to produce:
  - tests/data/golden_outputs.json  (deterministic input→output map)
  - Feature importance ranking printed to stdout

Usage:
    python scripts/analyze_model.py
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import settings
from app.pipelines.data_preparation import DataPreparationPipeline

GOLDEN_OUTPUT_PATH = ROOT / "tests" / "data" / "golden_outputs.json"

TEST_CASES = [
    {
        "name": "strong_all_around",
        "description": "High CIBIL, high income, good ratios, large assets -- clear approval",
        "input": {
            "no_of_dependents": 2,
            "education": "Graduate",
            "self_employed": "No",
            "income_annum": 9600000,
            "loan_amount": 29900000,
            "loan_term": 12,
            "cibil_score": 778,
            "residential_assets_value": 2400000,
            "commercial_assets_value": 17600000,
            "luxury_assets_value": 22700000,
            "bank_asset_value": 8000000,
        },
    },
    {
        "name": "low_cibil_low_income",
        "description": "Low CIBIL (350), low income, high loan -- clear rejection",
        "input": {
            "no_of_dependents": 3,
            "education": "Not Graduate",
            "self_employed": "No",
            "income_annum": 300000,
            "loan_amount": 5000000,
            "loan_term": 12,
            "cibil_score": 350,
            "residential_assets_value": 0,
            "commercial_assets_value": 0,
            "luxury_assets_value": 0,
            "bank_asset_value": 0,
        },
    },
    {
        "name": "high_cibil_terrible_ratios",
        "description": "CIBIL 800 but loan is 100x income with zero assets -- rejection despite high CIBIL",
        "input": {
            "no_of_dependents": 0,
            "education": "Graduate",
            "self_employed": "No",
            "income_annum": 100000,
            "loan_amount": 10000000,
            "loan_term": 4,
            "cibil_score": 800,
            "residential_assets_value": 0,
            "commercial_assets_value": 0,
            "luxury_assets_value": 0,
            "bank_asset_value": 0,
        },
    },
    {
        "name": "high_cibil_moderate_loan",
        "description": "CIBIL 800, moderate loan (7x income), zero assets",
        "input": {
            "no_of_dependents": 0,
            "education": "Graduate",
            "self_employed": "No",
            "income_annum": 100000,
            "loan_amount": 700000,
            "loan_term": 4,
            "cibil_score": 800,
            "residential_assets_value": 0,
            "commercial_assets_value": 0,
            "luxury_assets_value": 0,
            "bank_asset_value": 0,
        },
    },
    {
        "name": "high_cibil_small_loan",
        "description": "CIBIL 800, small loan (0.7x income), zero assets",
        "input": {
            "no_of_dependents": 0,
            "education": "Graduate",
            "self_employed": "No",
            "income_annum": 100000,
            "loan_amount": 70000,
            "loan_term": 4,
            "cibil_score": 800,
            "residential_assets_value": 0,
            "commercial_assets_value": 0,
            "luxury_assets_value": 0,
            "bank_asset_value": 0,
        },
    },
    {
        "name": "mid_cibil_good_ratios",
        "description": "Mid CIBIL (600), but excellent income-to-loan ratio and strong assets",
        "input": {
            "no_of_dependents": 1,
            "education": "Graduate",
            "self_employed": "No",
            "income_annum": 5000000,
            "loan_amount": 2000000,
            "loan_term": 12,
            "cibil_score": 600,
            "residential_assets_value": 3000000,
            "commercial_assets_value": 2000000,
            "luxury_assets_value": 1000000,
            "bank_asset_value": 1000000,
        },
    },
    {
        "name": "low_cibil_great_assets",
        "description": "Low CIBIL (400) but massive assets and low loan -- tests if assets can override CIBIL",
        "input": {
            "no_of_dependents": 0,
            "education": "Graduate",
            "self_employed": "No",
            "income_annum": 8000000,
            "loan_amount": 1000000,
            "loan_term": 6,
            "cibil_score": 400,
            "residential_assets_value": 10000000,
            "commercial_assets_value": 5000000,
            "luxury_assets_value": 3000000,
            "bank_asset_value": 2000000,
        },
    },
    {
        "name": "not_graduate_self_employed_high_cibil",
        "description": "Not Graduate + self-employed but high CIBIL and good financials",
        "input": {
            "no_of_dependents": 4,
            "education": "Not Graduate",
            "self_employed": "Yes",
            "income_annum": 4000000,
            "loan_amount": 8000000,
            "loan_term": 10,
            "cibil_score": 750,
            "residential_assets_value": 5000000,
            "commercial_assets_value": 3000000,
            "luxury_assets_value": 2000000,
            "bank_asset_value": 1000000,
        },
    },
    {
        "name": "graduate_high_dependents",
        "description": "Graduate with many dependents, moderate financials",
        "input": {
            "no_of_dependents": 5,
            "education": "Graduate",
            "self_employed": "Yes",
            "income_annum": 3000000,
            "loan_amount": 10000000,
            "loan_term": 15,
            "cibil_score": 650,
            "residential_assets_value": 2000000,
            "commercial_assets_value": 1000000,
            "luxury_assets_value": 500000,
            "bank_asset_value": 500000,
        },
    },
    {
        "name": "borderline_approved",
        "description": "CIBIL 610 with decent assets and 1.5x loan/income -- just over the approval line",
        "input": {
            "no_of_dependents": 1,
            "education": "Graduate",
            "self_employed": "No",
            "income_annum": 4000000,
            "loan_amount": 6000000,
            "loan_term": 10,
            "cibil_score": 610,
            "residential_assets_value": 3000000,
            "commercial_assets_value": 2000000,
            "luxury_assets_value": 1000000,
            "bank_asset_value": 1000000,
        },
    },
    {
        "name": "approved_needs_review",
        "description": "High CIBIL (750) with strong assets but loan is 5x income -- approved but flagged for review",
        "input": {
            "no_of_dependents": 1,
            "education": "Graduate",
            "self_employed": "No",
            "income_annum": 5000000,
            "loan_amount": 25000000,
            "loan_term": 12,
            "cibil_score": 750,
            "residential_assets_value": 10000000,
            "commercial_assets_value": 8000000,
            "luxury_assets_value": 5000000,
            "bank_asset_value": 5000000,
        },
    },
    {
        "name": "low_income_needs_review",
        "description": "Low income (155K) but high CIBIL and assets -- approved at 4.84x ratio, flagged for review",
        "input": {
            "no_of_dependents": 0,
            "education": "Graduate",
            "self_employed": "No",
            "income_annum": 155000,
            "loan_amount": 750000,
            "loan_term": 12,
            "cibil_score": 780,
            "residential_assets_value": 500000,
            "commercial_assets_value": 300000,
            "luxury_assets_value": 200000,
            "bank_asset_value": 100000,
        },
    },
    {
        "name": "borderline_rejected",
        "description": "CIBIL 610 with zero assets -- same CIBIL but no assets flips to rejection",
        "input": {
            "no_of_dependents": 1,
            "education": "Graduate",
            "self_employed": "No",
            "income_annum": 4000000,
            "loan_amount": 6000000,
            "loan_term": 10,
            "cibil_score": 610,
            "residential_assets_value": 0,
            "commercial_assets_value": 0,
            "luxury_assets_value": 0,
            "bank_asset_value": 0,
        },
    },
]


def extract_feature_importances(pipeline) -> list[dict]:
    """Pull feature names and importances from the trained pipeline."""
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    clean_names = [name.split("__", 1)[-1] for name in feature_names]

    importances = model.feature_importances_
    ranked = sorted(
        zip(clean_names, importances, strict=True),
        key=lambda x: x[1],
        reverse=True,
    )
    return [{"feature": name, "importance": round(float(imp), 6)} for name, imp in ranked]


def run_prediction(pipeline, preparation, payload: dict) -> dict:
    """Run a single payload through the full prediction pipeline."""
    df = pd.DataFrame([payload])
    features = preparation.prepare_features(df)
    features = features.drop(columns=[settings.target_column], errors="ignore")

    prediction = int(pipeline.predict(features)[0])
    probability = round(float(pipeline.predict_proba(features)[0, 1]), 4)

    income = payload.get("income_annum", 0)
    loan_amount = payload.get("loan_amount", 0)
    max_loan = income * settings.max_loan_income_ratio
    ratio = round(loan_amount / income, 4) if income > 0 else float("inf")

    needs_review = prediction == 1 and ratio > settings.review_threshold_ratio
    review_reason = (
        f"Loan amount exceeds {settings.review_threshold_ratio}x income (ratio: {ratio})" if needs_review else None
    )

    return {
        "prediction": prediction,
        "prediction_label": "Approved" if prediction == 1 else "Rejected",
        "approval_probability": probability,
        "needs_review": needs_review,
        "review_reason": review_reason,
        "loan_eligibility": {
            "min_loan_amount": settings.min_loan_amount,
            "max_loan_amount": round(max_loan, 2),
            "loan_income_ratio": ratio,
            "max_allowed_ratio": settings.max_loan_income_ratio,
            "within_limits": settings.min_loan_amount <= loan_amount <= max_loan,
        },
    }


def main():
    if not settings.model_path.exists():
        print(f"ERROR: Model not found at {settings.model_path}")
        print("Run the training pipeline first: python scripts/run_pipeline.py")
        sys.exit(1)

    pipeline = joblib.load(settings.model_path)
    preparation = DataPreparationPipeline()

    print("=" * 70)
    print("FEATURE IMPORTANCES (Random Forest)")
    print("=" * 70)
    importances = extract_feature_importances(pipeline)
    for i, fi in enumerate(importances, 1):
        bar = "#" * int(fi["importance"] * 50)
        print(f"  {i:2d}. {fi['feature']:<30s} {fi['importance']:.6f}  {bar}")

    print("\n" + "=" * 70)
    print("GOLDEN TEST CASE PREDICTIONS")
    print("=" * 70)

    golden = {
        "description": "Golden output fixtures for model regression tests. Regenerate with: python scripts/analyze_model.py",
        "feature_importances": importances,
        "test_cases": [],
    }

    for case in TEST_CASES:
        result = run_prediction(pipeline, preparation, case["input"])

        engineered = {
            "loan_income_ratio": round(case["input"]["loan_amount"] / case["input"]["income_annum"], 4),
            "total_assets": sum(case["input"].get(col, 0) for col in settings.optional_asset_columns),
        }
        engineered["asset_coverage_ratio"] = round(engineered["total_assets"] / case["input"]["loan_amount"], 4)
        engineered["net_worth"] = engineered["total_assets"] - case["input"]["loan_amount"]

        golden_case = {
            "name": case["name"],
            "description": case["description"],
            "input": case["input"],
            "engineered_features": engineered,
            "expected_output": result,
        }
        golden["test_cases"].append(golden_case)

        status = "APPROVED" if result["prediction"] == 1 else "REJECTED"
        print(f"\n  {case['name']}:")
        print(f"    {case['description']}")
        print(f"    Result: {status}  (probability: {result['approval_probability']})")
        print(
            f"    Ratios: loan/income={engineered['loan_income_ratio']}, "
            f"asset_coverage={engineered['asset_coverage_ratio']}, "
            f"net_worth={engineered['net_worth']:,.0f}"
        )

    GOLDEN_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_OUTPUT_PATH.write_text(json.dumps(golden, indent=2) + "\n", encoding="utf-8")
    print(f"\nGolden outputs written to: {GOLDEN_OUTPUT_PATH}")


if __name__ == "__main__":
    main()

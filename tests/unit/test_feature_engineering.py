import pandas as pd
import pytest

from app.pipelines.data_preparation import DataPreparationPipeline

prep = DataPreparationPipeline()

FEATURE_ENGINEERING_CASES = [
    {
        "case_id": "standard_values",
        "input_row": {
            "residential_assets_value": 100000,
            "commercial_assets_value": 200000,
            "luxury_assets_value": 300000,
            "bank_asset_value": 400000,
            "loan_amount": 500000,
            "income_annum": 250000,
        },
        "expected_features": {
            "total_assets": 1000000,
            "loan_income_ratio": 2.0,
            "asset_coverage_ratio": 2.0,
            "net_worth": 500000,
        },
    },
    {
        "case_id": "zero_assets",
        "input_row": {
            "residential_assets_value": 0,
            "commercial_assets_value": 0,
            "luxury_assets_value": 0,
            "bank_asset_value": 0,
            "loan_amount": 1000000,
            "income_annum": 500000,
        },
        "expected_features": {
            "total_assets": 0,
            "loan_income_ratio": 2.0,
            "asset_coverage_ratio": 0.0,
            "net_worth": -1000000,
        },
    },
    {
        "case_id": "high_assets",
        "input_row": {
            "residential_assets_value": 1000000,
            "commercial_assets_value": 500000,
            "luxury_assets_value": 300000,
            "bank_asset_value": 200000,
            "loan_amount": 1200000,
            "income_annum": 500000,
        },
        "expected_features": {
            "total_assets": 2000000,
            "loan_income_ratio": 2.4,
            "asset_coverage_ratio": 2000000 / 1200000,
            "net_worth": 800000,
        },
    },
]


@pytest.mark.unit
@pytest.mark.parametrize(
    "case",
    FEATURE_ENGINEERING_CASES,
    ids=[c["case_id"] for c in FEATURE_ENGINEERING_CASES],
)
def test_feature_engineering_outputs_expected_features(case):
    input_df = pd.DataFrame([case["input_row"]])

    output_df = prep.create_features(input_df)

    for feature_name, expected_value in case["expected_features"].items():
        assert feature_name in output_df.columns
        assert output_df.loc[0, feature_name] == pytest.approx(expected_value)

import pandas as pd
import pytest

from app.core.config import settings
from app.pipelines.data_preparation import DataPreparationPipeline
from tests.schemas.requests import LoanApplicationRequest

prep = DataPreparationPipeline()


@pytest.mark.unit
def test_optional_asset_fields_are_filled_with_zero():
    request = LoanApplicationRequest(
        no_of_dependents=0,
        education="Graduate",
        self_employed="No",
        income_annum=500000,
        loan_amount=100000,
        loan_term=12,
        cibil_score=700,
        residential_assets_value=None,
        commercial_assets_value=None,
        luxury_assets_value=None,
        bank_asset_value=None,
    )

    df = pd.DataFrame([request.model_dump()])

    result = prep.handle_missing_values(df)

    for field in settings.optional_asset_columns:
        assert result.loc[0, field] == 0


@pytest.mark.unit
def test_string_fields_are_trimmed():
    df = pd.DataFrame(
        [
            {
                "education": " Graduate ",
                "self_employed": " No ",
            }
        ]
    )

    result = prep.clean_string_values(df)

    assert result.loc[0, "education"] == "Graduate"
    assert result.loc[0, "self_employed"] == "No"


@pytest.mark.unit
def test_column_names_are_normalized():
    df = pd.DataFrame(
        [
            {
                "Income Annum": 500000,
                " Loan Amount ": 100000,
                "CIBIL__Score": 700,
            }
        ]
    )

    result = prep.clean_column_names(df)

    assert "income_annum" in result.columns
    assert "loan_amount" in result.columns
    assert "cibil_score" in result.columns


@pytest.mark.unit
def test_handle_missing_values_raises_on_required_field():
    df = pd.DataFrame(
        [
            {
                "no_of_dependents": 0,
                "education": "Graduate",
                "self_employed": "No",
                "income_annum": None,
                "loan_amount": 100000,
                "loan_term": 12,
                "cibil_score": 700,
                "loan_status": 1,
            }
        ]
    )

    with pytest.raises(
        ValueError,
        match="Missing values found in required columns",
    ):
        prep.handle_missing_values(df)

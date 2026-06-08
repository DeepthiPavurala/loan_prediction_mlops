from tests.schemas.requests import LoanApplicationRequest


def valid_loan_payload() -> dict:
    return {
        "loan_id": 1001,
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
    }


def valid_loan_request() -> LoanApplicationRequest:
    return LoanApplicationRequest(**valid_loan_payload())


def second_loan_payload() -> dict:
    payload = valid_loan_payload()
    payload["loan_id"] = 1002
    payload["education"] = "Not Graduate"
    payload["self_employed"] = "Yes"
    payload["cibil_score"] = 650
    return payload


def missing_required_field_payload() -> dict:
    payload = valid_loan_payload()
    payload.pop("cibil_score")
    return payload


def invalid_category_payload() -> dict:
    payload = valid_loan_payload()
    payload["education"] = "Masters"
    return payload


def invalid_boundary_payload() -> dict:
    payload = valid_loan_payload()
    payload["cibil_score"] = 1000
    return payload


def guaranteed_approval_payload() -> dict:
    """High CIBIL (778), high income, strong assets, positive net worth -- always approved (prob ~0.97)."""
    return {
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
    }


def guaranteed_rejection_payload() -> dict:
    """Low CIBIL (350), low income, high loan, zero assets -- always rejected (prob ~0.085)."""
    return {
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
    }


def high_cibil_bad_ratios_payload() -> dict:
    """CIBIL 800 but loan is 100x income with zero assets -- rejected despite high CIBIL (prob ~0.07)."""
    return {
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
    }


def borderline_approval_payload() -> dict:
    """CIBIL 610 with decent assets and 1.5x loan/income -- just over the approval line (prob ~0.94)."""
    return {
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
    }


def borderline_rejection_payload() -> dict:
    """Same as borderline_approval but with zero assets -- flips to rejection (prob ~0.245)."""
    return {
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
    }


MANDATORY_FIELDS = [
    "no_of_dependents",
    "education",
    "self_employed",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
]

OPTIONAL_FIELDS = [
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]

ALLOWED_FEATURES = {
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
}

ENGINEERED_FEATURES = [
    "total_assets",
    "loan_income_ratio",
    "asset_coverage_ratio",
    "net_worth",
]

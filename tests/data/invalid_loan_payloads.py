INVALID_FIELD_VALUE_CASES = [
    {
        "field": "education",
        "value": "Masters",
        "expected_status": 422,
        "expected_error_type": "string_pattern_mismatch",
    },
    {
        "field": "self_employed",
        "value": "Maybe",
        "expected_status": 422,
        "expected_error_type": "string_pattern_mismatch",
    },
    {
        "field": "cibil_score",
        "value": 250,
        "expected_status": 422,
        "expected_error_type": "greater_than_equal",
    },
    {
        "field": "cibil_score",
        "value": 950,
        "expected_status": 422,
        "expected_error_type": "less_than_equal",
    },
    {
        "field": "loan_amount",
        "value": -1000,
        "expected_status": 422,
        "expected_error_type": "greater_than",
    },
    {
        "field": "income_annum",
        "value": -5000,
        "expected_status": 422,
        "expected_error_type": "greater_than",
    },
    {
        "field": "no_of_dependents",
        "value": -1,
        "expected_status": 422,
        "expected_error_type": "greater_than_equal",
    },
    {
        "field": "loan_term",
        "value": 0,
        "expected_status": 422,
        "expected_error_type": "greater_than",
    },
]

INVALID_TYPE_CASES = [
    {
        "field": "cibil_score",
        "value": "not_a_number",
        "expected_status": 422,
        "expected_error_type": "int_parsing",
    },
    {
        "field": "income_annum",
        "value": "high",
        "expected_status": 422,
        "expected_error_type": "float_parsing",
    },
    {
        "field": "no_of_dependents",
        "value": "two",
        "expected_status": 422,
        "expected_error_type": "int_parsing",
    },
]

EXTRA_FIELD_CASES = [
    {
        "extra_fields": {"unknown_field": "unexpected"},
        "expected_status": 422,
        "expected_detail": "extra",
        "expected_error_type": "extra_forbidden",
    },
]

VALID_FIELD_VALUE_CASES = [
    {"field": "cibil_score", "value": 300, "expected_status": 200},
    {"field": "cibil_score", "value": 900, "expected_status": 200},
    {"field": "education", "value": "Graduate", "expected_status": 200},
    {"field": "education", "value": "Not Graduate", "expected_status": 200},
    {"field": "self_employed", "value": "Yes", "expected_status": 200},
    {"field": "self_employed", "value": "No", "expected_status": 200},
]

BOUNDARY_CASES = [
    # (field, value, expected_status)
    ("cibil_score", 300, 200),
    ("cibil_score", 900, 200),
    ("cibil_score", 299, 422),
    ("cibil_score", 901, 422),
    ("loan_amount", 0.01, 200),
    ("loan_amount", 0, 422),
    ("loan_amount", -1, 422),
    ("income_annum", 0.01, 200),
    ("income_annum", 0, 422),
    ("loan_term", 1, 200),
    ("loan_term", 0, 422),
]

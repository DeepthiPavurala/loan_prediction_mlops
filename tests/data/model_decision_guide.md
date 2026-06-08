# Loan Prediction Model -- Decision Guide for SDETs

## How the Model Decides

The model is a **Random Forest Classifier** trained on historical loan data.
It does NOT use simple rules like "CIBIL > 650 = Approved." Instead, it considers
**15 features together** and votes across 200 decision trees.

### Feature Importance Ranking

| Rank | Feature                  | Importance | Notes                                    |
|------|--------------------------|------------|------------------------------------------|
| 1    | **cibil_score**          | **81.0%**  | Dominant feature -- but NOT sufficient alone |
| 2    | loan_term                | 6.5%       | Loan duration                            |
| 3    | loan_income_ratio        | 3.6%       | Engineered: loan_amount / income_annum   |
| 4    | asset_coverage_ratio     | 1.7%       | Engineered: total_assets / loan_amount   |
| 5    | net_worth                | 1.3%       | Engineered: total_assets - loan_amount   |
| 6    | loan_amount              | 1.0%       |                                          |
| 7-11 | Asset columns            | ~0.7% each | residential, commercial, luxury, bank    |
| 12   | income_annum             | 0.7%       |                                          |
| 13   | no_of_dependents         | 0.3%       | Minimal impact                           |
| 14-17| education, self_employed | <0.1% each | Negligible impact                        |

### Key Decision Rules (learned from data, not hardcoded)

1. **CIBIL score is necessary but NOT sufficient.**
   - CIBIL >= 600 with good ratios and assets => likely Approved
   - CIBIL >= 800 with terrible ratios (high loan/income, zero assets) => **Rejected**
   - CIBIL < 500 => almost always Rejected, regardless of other factors

2. **Loan-to-income ratio matters heavily.**
   - ratio < 3x with assets => tends toward Approval
   - ratio > 7x with no assets => tends toward Rejection even with high CIBIL

3. **Assets can flip the decision at the boundary.**
   - CIBIL 610 + assets => Approved
   - CIBIL 610 + zero assets => Rejected (same CIBIL, same income, same loan)

4. **Education and self-employment have negligible impact.**
   - Graduate vs Not Graduate does not meaningfully change predictions
   - Self-employed vs employed does not meaningfully change predictions

## Quick Reference: Known Outcomes

### Guaranteed Approvals

| Scenario                    | CIBIL | Loan/Income | Assets     | Probability |
|-----------------------------|-------|-------------|------------|-------------|
| Strong all-around           | 778   | 3.1x        | 50.7M      | 0.97        |
| Mid CIBIL, great ratios     | 600   | 0.4x        | 7M         | 0.955       |
| Not Graduate + self-employed| 750   | 2.0x        | 11M        | 0.97        |
| Borderline (with assets)    | 610   | 1.5x        | 7M         | 0.94        |

### Guaranteed Rejections

| Scenario                    | CIBIL | Loan/Income | Assets     | Probability |
|-----------------------------|-------|-------------|------------|-------------|
| Low CIBIL, low income       | 350   | 16.7x       | 0          | 0.085       |
| High CIBIL, terrible ratios | 800   | 100x        | 0          | 0.07        |
| High CIBIL, moderate loan   | 800   | 7x          | 0          | 0.095       |
| High CIBIL, small loan      | 800   | 0.7x        | 0          | 0.255       |
| Low CIBIL, great assets     | 400   | 0.125x      | 20M        | 0.035       |
| High dependents, bad ratios | 650   | 3.3x        | 4M         | 0.14        |
| Borderline (no assets)      | 610   | 1.5x        | 0          | 0.245       |

## How to Verify

```bash
# Run all regression tests (exact output matching)
pytest tests/test_model_regression.py -v

# Run sensitivity tests (decision boundary exploration)
pytest tests/test_field_sensitivity.py -v

# Run both
pytest -m "regression or sensitivity" -v
```

## When Golden Outputs Need Regeneration

If the model is retrained (new training data or changed hyperparameters),
golden outputs must be regenerated:

```bash
python scripts/analyze_model.py
```

Review the diff in `tests/data/golden_outputs.json` before committing.
If predictions changed, update this guide accordingly.

## Source of Truth Chain

```
Training Data (committed) → Trained Model (.joblib) → Golden Outputs (committed) → Regression Tests
```

If regression tests fail, something in this chain changed. Check:
1. Was the model retrained? => Regenerate golden outputs
2. Was training data modified? => Retrain and regenerate
3. Was code changed? => Debug the code change

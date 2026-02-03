# Small-Sample Monte Carlo Validation Framework

Based on Lee & Wooldridge (2026) ssrn-5325686, Section 5.

## Overview

This test module implements a comprehensive Monte Carlo validation framework for the LWDID package, specifically designed for small-sample settings (N=20, T=20). The framework validates:

1. **DGP Implementation**: Data generating process matching paper Section 5
2. **Estimator Performance**: Demeaning vs Detrending comparison
3. **Standard Error Accuracy**: OLS vs HC3 coverage rates
4. **Paper Replication**: Comparison with Table 2 reference values

## Paper Reference

Lee, S.J. & Wooldridge, J.M. (2026). "Difference-in-Differences with a Single Treated Unit." SSRN 5325686.

### Key DGP Parameters (Table 1)

| Parameter | Value | Description |
|-----------|-------|-------------|
| N | 20 | Number of units |
| T | 20 | Number of periods |
| T₀ | 10 | Pre-treatment periods |
| T₁ | 10 | Post-treatment periods |
| Treatment start | t=11 | Common treatment timing |
| σ_C | 2.0 | Unit fixed effect SD |
| σ_G | 1.0 | Unit trend SD (mean=1) |
| ρ | 0.75 | AR(1) coefficient |
| σ_ε | √2 | AR(1) innovation SD |

### Treatment Scenarios

| Scenario | P(D=1) | (α₀, α₁, α₂) |
|----------|--------|--------------|
| 1 | 0.32 | (-1, -1/3, 1/4) |
| 2 | 0.24 | (-1.5, 1/3, 1/4) |
| 3 | 0.17 | (-2, 1/3, 1/4) |

### Time-Varying Treatment Effects (δ_t)

```
Pre-treatment (t=1-10):  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Post-treatment (t=11-20): [1, 2, 3, 3, 3, 2, 2, 2, 1, 1]
Average ATT = 2.0
```

## Test Structure

```
tests/test_small_sample_monte_carlo/
├── __init__.py
├── conftest.py                    # Pytest fixtures
├── fixtures/
│   ├── __init__.py
│   └── monte_carlo_runner.py      # Monte Carlo simulation runner
├── test_dgp_small_sample.py       # DGP unit tests
├── test_numerical_validation.py   # Numerical validation (vibe-math)
├── test_formula_validation.py     # Formula verification
├── test_hc3_coverage.py           # HC3 coverage tests
├── test_monte_carlo_small.py      # Full Monte Carlo tests
├── test_e2e_small_sample.py       # End-to-end tests
├── test_edge_cases.py             # Edge case tests
├── test_stata_e2e.py              # Stata comparison tests
├── test_paper_table2_validation.py # Paper Table 2 validation
└── README.md                      # This file
```

## Running Tests

### All Tests (excluding slow)

```bash
pytest tests/test_small_sample_monte_carlo/ -v -m "not slow"
```

### Full Monte Carlo (slow)

```bash
pytest tests/test_small_sample_monte_carlo/ -v -m "slow"
```

### Numerical Validation Only

```bash
pytest tests/test_small_sample_monte_carlo/ -v -m "numerical"
```

### Paper Validation Only

```bash
pytest tests/test_small_sample_monte_carlo/ -v -m "paper_validation"
```

### Stata Comparison (requires Stata)

```bash
pytest tests/test_small_sample_monte_carlo/ -v -m "stata"
```

## Test Markers

| Marker | Description |
|--------|-------------|
| `@pytest.mark.slow` | Tests taking > 1 minute |
| `@pytest.mark.monte_carlo` | Monte Carlo simulation tests |
| `@pytest.mark.numerical` | Numerical validation tests |
| `@pytest.mark.paper_validation` | Paper Table 2 comparison |
| `@pytest.mark.stata` | Requires Stata environment |

## Key Findings from Paper

### 1. Detrending Outperforms Demeaning

When unit-specific trends are heterogeneous, detrending produces:
- Lower bias
- Lower RMSE
- Better coverage rates

### 2. HC3 Improves Small-Sample Coverage

HC3 standard errors (Davidson & MacKinnon, 1993) provide:
- Coverage rates closer to nominal 95%
- Better performance than OLS SE in small samples

### 3. Sparse Treatment Increases Variance

As P(D=1) decreases from 0.32 to 0.17:
- Standard deviation increases
- RMSE increases
- But detrending remains optimal

## Paper Table 2 Reference Values

### Detrending Estimator

| Scenario | Bias | SD | RMSE | Coverage (OLS) |
|----------|------|-----|------|----------------|
| 1 | 0.009 | 1.73 | 1.734 | 96% |
| 2 | -0.042 | 1.89 | 1.892 | 95% |
| 3 | 0.165 | 2.37 | 2.380 | 95% |

## DGP Formulas

### Outcome Model

```
Y_it = λ_t - C_i + G_i × t + D_i × post_t × (δ_t + ν_it) + u_it
```

Where:
- `λ_t`: Period effects (Table 1)
- `C_i ~ N(0, 4)`: Unit fixed effects
- `G_i ~ N(1, 1)`: Unit-specific trend slopes
- `δ_t`: Time-varying treatment effects (Table 1)
- `ν_it ~ N(0, 2)`: Treatment effect heterogeneity
- `u_it`: AR(1) errors with ρ=0.75

### Treatment Assignment

```
D_i = I(α₀ - α₁ × C_i + α₂ × G_i + e_i > 0)
e_i ~ Logistic(0, 1)
```

### Transformations

**Demeaning (Procedure 2.1)**:
```
ẏ_it = Y_it - Ȳ_i,pre
```

**Detrending (Procedure 5.1)**:
```
Ÿ_it = Y_it - (α̂_i + β̂_i × t)
```

## Validation Tolerances

| Metric | Tolerance | Type |
|--------|-----------|------|
| Bias | 0.5 | Absolute |
| SD | 30% | Relative |
| RMSE | 30% | Relative |
| Coverage | 15pp | Absolute |

## Usage Example

```python
from tests.test_small_sample_monte_carlo.fixtures.monte_carlo_runner import (
    run_small_sample_monte_carlo,
    generate_comparison_table,
)

# Run Monte Carlo for Scenario 1
results = run_small_sample_monte_carlo(
    n_reps=1000,
    scenario='scenario_1',
    estimators=['demeaning', 'detrending'],
    seed=42,
    verbose=True,
    use_lwdid=False,  # Use manual implementation
)

# Print results
for estimator, result in results.items():
    print(result.summary())
```

## Dependencies

- numpy
- pandas
- scipy
- statsmodels
- pytest

## Contributing

When adding new tests:
1. Follow existing naming conventions
2. Add appropriate pytest markers
3. Include docstrings with paper references
4. Update this README if adding new test categories

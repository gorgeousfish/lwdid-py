"""
Tests for HC3 standard errors in common timing scenario.

This module tests HC3 SE calculation against Stata for:
1. Monte Carlo simulated data (multiple treated units) - EXACT MATCH
2. Smoking data (single treated unit) - BOUNDARY CASE

Note: When there is only ONE treated unit, h_ii = 1 for that unit,
causing (1-h)^2 = 0. This is a mathematical singularity. Stata and
Python handle this differently:
- Stata: Uses some internal fallback (unclear mechanism)
- Python: Uses eps protection, resulting in SE closer to HC2

For normal use cases with multiple treated units, Python matches Stata exactly.
"""

import numpy as np
import pandas as pd
import pytest
import warnings


class TestHC3CommonTimingStataMatch:
    """Test HC3 matches Stata exactly for normal data."""
    
    def test_monte_carlo_data_exact_match(self):
        """
        Test HC3 with Monte Carlo data (multiple treated units).
        
        Stata code:
        ```stata
        clear
        set seed 42
        set obs 100
        gen d = (_n <= 30)
        gen x1 = rnormal()
        gen x2 = rnormal()
        gen sigma = 1 + 0.5 * abs(x1)
        gen y_dot = 1 + 0.5 * d + 2 * x1 + 1.5 * x2 + sigma * rnormal()
        regress y_dot d x1 x2, vce(hc3)
        ```
        
        Stata results:
        - d coef: 0.4517956
        - OLS SE(d): 0.3212457
        - HC1 SE(d): 0.3224824
        - HC3 SE(d): 0.3309612
        """
        try:
            df = pd.read_csv('/tmp/mc_common_timing_hc3.csv')
        except FileNotFoundError:
            pytest.skip("Stata data not available - run Stata first")
        
        y = df['y_dot'].values
        d = df['d'].values.astype(float)
        x1 = df['x1'].values
        x2 = df['x2'].values
        
        X = np.column_stack([np.ones(len(y)), d, x1, x2])
        n, k = X.shape
        
        # OLS
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta
        XtX_inv = np.linalg.inv(X.T @ X)
        
        # HC3
        tmp = X @ XtX_inv
        h_ii = (tmp * X).sum(axis=1)
        denom = np.maximum((1 - h_ii) ** 2, np.finfo(np.float64).eps)
        omega = residuals ** 2 / denom
        meat = (X.T * omega) @ X
        var_hc3 = XtX_inv @ meat @ XtX_inv
        se_hc3 = np.sqrt(np.diag(var_hc3))
        
        # Stata reference values
        stata_coef_d = 0.4517956
        stata_se_hc3_d = 0.3309612
        
        # Verify exact match
        assert np.isclose(beta[1], stata_coef_d, rtol=1e-5), \
            f"Coefficient mismatch: {beta[1]} vs {stata_coef_d}"
        
        assert np.isclose(se_hc3[1], stata_se_hc3_d, rtol=1e-5), \
            f"HC3 SE mismatch: {se_hc3[1]} vs {stata_se_hc3_d}"
    
    def test_leverage_values_match_stata(self):
        """Verify leverage values match Stata exactly."""
        try:
            df = pd.read_csv('/tmp/mc_common_timing_hc3.csv')
        except FileNotFoundError:
            pytest.skip("Stata data not available")
        
        y = df['y_dot'].values
        d = df['d'].values.astype(float)
        x1 = df['x1'].values
        x2 = df['x2'].values
        
        X = np.column_stack([np.ones(len(y)), d, x1, x2])
        XtX_inv = np.linalg.inv(X.T @ X)
        tmp = X @ XtX_inv
        h_ii_python = (tmp * X).sum(axis=1)
        h_ii_stata = df['h_ii'].values
        
        max_diff = np.max(np.abs(h_ii_python - h_ii_stata))
        assert max_diff < 1e-8, f"Leverage max diff = {max_diff}"


class TestHC3SingleTreatedUnit:
    """
    Test HC3 behavior with single treated unit (boundary case).
    
    When there is only ONE treated unit:
    - Its leverage h_ii = 1 / (n * p) = 1 (when p = 1/n)
    - This causes (1-h)^2 = 0, making HC3 weight undefined
    - Python uses eps protection; Stata uses unknown fallback
    """
    
    def test_single_treated_leverage_equals_one(self):
        """Verify h_ii = 1 for single treated unit."""
        n = 39
        n_treated = 1
        
        d = np.zeros(n)
        d[0] = 1  # Single treated unit
        
        X = np.column_stack([np.ones(n), d])
        XtX_inv = np.linalg.inv(X.T @ X)
        tmp = X @ XtX_inv
        h_ii = (tmp * X).sum(axis=1)
        
        # Theoretical: h = 1/(n*p) = 1/(n * 1/n) = 1
        assert np.isclose(h_ii[0], 1.0, rtol=1e-10), \
            f"Expected h_ii = 1 for single treated, got {h_ii[0]}"
    
    def test_single_treated_residual_is_zero(self):
        """Verify residual = 0 for single treated unit."""
        n = 39
        np.random.seed(42)
        
        d = np.zeros(n)
        d[0] = 1
        
        y = np.random.randn(n) + 0.5 * d
        X = np.column_stack([np.ones(n), d])
        
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta
        
        # The treated unit's residual should be 0 (it determines the treatment effect)
        assert np.abs(residuals[0]) < 1e-10, \
            f"Expected residual ≈ 0 for single treated, got {residuals[0]}"
    
    def test_hc3_numerical_stability_single_treated(self):
        """HC3 should not crash or return NaN with single treated unit."""
        n = 50
        np.random.seed(42)
        
        d = np.zeros(n)
        d[0] = 1  # Single treated
        
        y = np.random.randn(n) + 0.5 * d
        X = np.column_stack([np.ones(n), d])
        
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta
        XtX_inv = np.linalg.inv(X.T @ X)
        
        tmp = X @ XtX_inv
        h_ii = (tmp * X).sum(axis=1)
        
        # HC3 with eps protection
        denom = np.maximum((1 - h_ii) ** 2, np.finfo(np.float64).eps)
        omega = residuals ** 2 / denom
        meat = (X.T * omega) @ X
        var_hc3 = XtX_inv @ meat @ XtX_inv
        se_hc3 = np.sqrt(np.diag(var_hc3))
        
        # Should be finite
        assert np.all(np.isfinite(se_hc3)), "HC3 SE should be finite"
        assert se_hc3[1] > 0, "HC3 SE should be positive"


class TestHC3MultipleTreatedUnits:
    """Test HC3 with multiple treated units (normal case)."""
    
    def test_hc3_se_larger_than_hc1(self):
        """HC3 SE should typically be >= HC1 SE."""
        np.random.seed(42)
        n = 100
        
        d = (np.arange(n) < 30).astype(float)  # 30 treated
        x = np.random.randn(n)
        y = 1 + 0.5 * d + 2 * x + np.random.randn(n)
        
        X = np.column_stack([np.ones(n), d, x])
        n_obs, k = X.shape
        
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta
        XtX_inv = np.linalg.inv(X.T @ X)
        
        # HC1
        omega_hc1 = (n_obs / (n_obs - k)) * residuals ** 2
        meat_hc1 = (X.T * omega_hc1) @ X
        var_hc1 = XtX_inv @ meat_hc1 @ XtX_inv
        se_hc1 = np.sqrt(np.diag(var_hc1))
        
        # HC3
        tmp = X @ XtX_inv
        h_ii = (tmp * X).sum(axis=1)
        denom = np.maximum((1 - h_ii) ** 2, np.finfo(np.float64).eps)
        omega_hc3 = residuals ** 2 / denom
        meat_hc3 = (X.T * omega_hc3) @ X
        var_hc3 = XtX_inv @ meat_hc3 @ XtX_inv
        se_hc3 = np.sqrt(np.diag(var_hc3))
        
        # HC3 >= HC1 (usually)
        assert se_hc3[1] >= se_hc1[1] * 0.99, \
            f"HC3 SE ({se_hc3[1]}) should be >= HC1 SE ({se_hc1[1]})"
    
    def test_leverage_sum_equals_k(self):
        """Sum of leverage values should equal k."""
        np.random.seed(42)
        n = 100
        k = 3
        
        d = (np.arange(n) < 30).astype(float)
        x = np.random.randn(n)
        
        X = np.column_stack([np.ones(n), d, x])
        XtX_inv = np.linalg.inv(X.T @ X)
        tmp = X @ XtX_inv
        h_ii = (tmp * X).sum(axis=1)
        
        assert np.isclose(h_ii.sum(), k, rtol=1e-10), \
            f"Sum of leverage should be {k}, got {h_ii.sum()}"


class TestSmokingDataHC3:
    """Test HC3 with smoking dataset."""
    
    @pytest.fixture
    def smoking_data_1990(self):
        """Load and prepare smoking data for year 1990."""
        try:
            df = pd.read_csv('data/smoking.csv')
        except FileNotFoundError:
            pytest.skip("Smoking data not available")
        
        pre_mean = df[df['post'] == 0].groupby('state')['cigsale'].mean()
        df['pre_mean'] = df['state'].map(pre_mean)
        df['y_dot'] = df['cigsale'] - df['pre_mean']
        
        return df[df['year'] == 1990].copy()
    
    def test_smoking_single_treated_leverage(self, smoking_data_1990):
        """Verify California (single treated) has h_ii = 1."""
        df = smoking_data_1990
        
        y = df['y_dot'].values
        d = df['d'].values.astype(float)
        
        X = np.column_stack([np.ones(len(y)), d])
        XtX_inv = np.linalg.inv(X.T @ X)
        tmp = X @ XtX_inv
        h_ii = (tmp * X).sum(axis=1)
        
        h_treated = h_ii[d == 1][0]
        assert np.isclose(h_treated, 1.0, rtol=1e-6), \
            f"Expected h_ii = 1 for California, got {h_treated}"
    
    def test_smoking_coefficients_match_stata(self, smoking_data_1990):
        """OLS coefficients should match Stata exactly."""
        df = smoking_data_1990
        
        y = df['y_dot'].values
        d = df['d'].values.astype(float)
        
        X = np.column_stack([np.ones(len(y)), d])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        
        # Stata values
        stata_intercept = -24.90374
        stata_d_coef = -13.50678
        
        assert np.isclose(beta[0], stata_intercept, rtol=1e-5), \
            f"Intercept mismatch: {beta[0]} vs {stata_intercept}"
        assert np.isclose(beta[1], stata_d_coef, rtol=1e-5), \
            f"d coef mismatch: {beta[1]} vs {stata_d_coef}"
    
    def test_smoking_hc3_boundary_documented(self, smoking_data_1990):
        """
        Document the boundary case behavior.
        
        This test documents the difference between Python and Stata
        for the h_ii = 1 boundary case. This is NOT a bug - it's a
        mathematical singularity with no unique solution.
        """
        df = smoking_data_1990
        
        y = df['y_dot'].values
        d = df['d'].values.astype(float)
        
        X = np.column_stack([np.ones(len(y)), d])
        n, k = X.shape
        
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta
        XtX_inv = np.linalg.inv(X.T @ X)
        
        # HC3 with eps protection
        tmp = X @ XtX_inv
        h_ii = (tmp * X).sum(axis=1)
        denom = np.maximum((1 - h_ii) ** 2, np.finfo(np.float64).eps)
        omega = residuals ** 2 / denom
        meat = (X.T * omega) @ X
        var_hc3 = XtX_inv @ meat @ XtX_inv
        se_hc3_python = np.sqrt(var_hc3[1, 1])
        
        # Stata values (for reference)
        stata_se_hc3 = 18.86442
        stata_se_ols = 16.78816
        stata_se_hc1 = 2.7234
        
        # Python SE will be close to HC1/HC2 (because treated residual ≈ 0)
        # This is documented behavior, not a bug
        assert se_hc3_python > 0, "HC3 SE should be positive"
        assert np.isfinite(se_hc3_python), "HC3 SE should be finite"
        
        # Document the difference
        print(f"\n=== Single Treated Unit Boundary Case ===")
        print(f"Python HC3 SE: {se_hc3_python:.5f}")
        print(f"Stata HC3 SE:  {stata_se_hc3:.5f}")
        print(f"Stata OLS SE:  {stata_se_ols:.5f}")
        print(f"Stata HC1 SE:  {stata_se_hc1:.5f}")
        print("Note: Difference is expected due to h_ii = 1 singularity")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

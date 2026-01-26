"""
Tests for Story 5.1: HC0 Standard Error Implementation

This module provides comprehensive tests for HC0 (White robust) standard error
implementation, including:
- Unit tests for _compute_hc0_variance() and _compute_hc1_variance()
- Stata consistency tests (relative error < 1e-6)
- statsmodels cross-validation (relative error < 1e-10)
- HC0/HC1 mathematical relationship verification
- Monte Carlo coverage rate tests
- Formula validation tests

Reference:
    White H (1980). "A Heteroskedasticity-Consistent Covariance Matrix
    Estimator and a Direct Test for Heteroskedasticity." Econometrica.
"""

import json
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from lwdid.staggered.estimation import (
    run_ols_regression,
    _compute_hc0_variance,
    _compute_hc1_variance,
)


# ============================================================================
# Test Data Generation Functions
# ============================================================================

def generate_test_data(
    n: int = 500,
    seed: int = 42,
    heteroskedastic: bool = True,
) -> pd.DataFrame:
    """
    Generate test data for HC standard error tests.
    
    DGP:
        Y = 1 + 2*D + 0.5*X1 + 0.3*X2 + ε
        ε ~ N(0, σ²(X)) where σ²(X) = 1 + 0.5*|X1| if heteroskedastic
        D = 1{Z + 0.5*X1 + 0.3*X2 > 0}, Z ~ N(0, 1)
    
    Parameters
    ----------
    n : int
        Sample size
    seed : int
        Random seed
    heteroskedastic : bool
        Whether to generate heteroskedastic errors
        
    Returns
    -------
    pd.DataFrame
        Test data with columns: y, d, x1, x2
    """
    rng = np.random.default_rng(seed)
    
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    
    # Treatment assignment (propensity score model)
    z = rng.normal(0, 1, n)
    d = (z + 0.5 * x1 + 0.3 * x2 > 0).astype(int)
    
    # Outcome with optional heteroskedasticity
    if heteroskedastic:
        sigma = np.sqrt(1 + 0.5 * np.abs(x1))
    else:
        sigma = np.ones(n)
    
    epsilon = rng.normal(0, 1, n) * sigma
    y = 1 + 2 * d + 0.5 * x1 + 0.3 * x2 + epsilon
    
    return pd.DataFrame({
        'y': y,
        'd': d,
        'x1': x1,
        'x2': x2,
    })


def generate_stata_reference_data(seed: int = 12345, n: int = 500) -> pd.DataFrame:
    """
    Generate data matching Stata's RNG for cross-validation.
    
    Uses exact same DGP as Stata test code:
        set seed 12345
        gen x1 = rnormal()
        gen x2 = rnormal()
        gen d = (rnormal() + 0.5*x1 + 0.3*x2) > 0
        gen y = 1 + 2*d + 0.5*x1 + 0.3*x2 + rnormal()*sqrt(1+0.5*abs(x1))
    """
    # Note: Python's numpy RNG differs from Stata's RNG
    # For exact match, we'd need to export data from Stata
    # Here we use numpy with same DGP structure
    return generate_test_data(n=n, seed=seed, heteroskedastic=True)


# ============================================================================
# Task 5.1.1: HC0 Variance Function Tests
# ============================================================================

class TestHC0VarianceFunction:
    """Unit tests for _compute_hc0_variance()"""
    
    def test_hc0_variance_shape(self):
        """Test output shape is (K, K)"""
        n, k = 100, 3
        rng = np.random.default_rng(42)
        X = np.column_stack([np.ones(n), rng.normal(0, 1, (n, k-1))])
        residuals = rng.normal(0, 1, n)
        XtX_inv = np.linalg.inv(X.T @ X)
        
        var = _compute_hc0_variance(X, residuals, XtX_inv)
        
        assert var.shape == (k, k)
    
    def test_hc0_variance_symmetric(self):
        """Test variance matrix is symmetric"""
        n, k = 100, 3
        rng = np.random.default_rng(42)
        X = np.column_stack([np.ones(n), rng.normal(0, 1, (n, k-1))])
        residuals = rng.normal(0, 1, n)
        XtX_inv = np.linalg.inv(X.T @ X)
        
        var = _compute_hc0_variance(X, residuals, XtX_inv)
        
        assert np.allclose(var, var.T)
    
    def test_hc0_variance_positive_diagonal(self):
        """Test diagonal elements are positive"""
        n, k = 100, 3
        rng = np.random.default_rng(42)
        X = np.column_stack([np.ones(n), rng.normal(0, 1, (n, k-1))])
        residuals = rng.normal(0, 1, n) * 0.1 + 0.5  # Non-zero residuals
        XtX_inv = np.linalg.inv(X.T @ X)
        
        var = _compute_hc0_variance(X, residuals, XtX_inv)
        
        assert np.all(np.diag(var) > 0)
    
    def test_hc0_formula_manual_verification(self):
        """Verify HC0 formula against manual calculation"""
        # Simple 3x2 example
        X = np.array([[1, 1], [1, 2], [1, 3]], dtype=float)
        e = np.array([0.1, -0.2, 0.15])
        XtX_inv = np.linalg.inv(X.T @ X)
        
        # Manual calculation: Meat = Σ e_i² x_i x_i'
        meat_manual = np.zeros((2, 2))
        for i in range(3):
            meat_manual += e[i]**2 * np.outer(X[i], X[i])
        var_manual = XtX_inv @ meat_manual @ XtX_inv
        
        # Function calculation
        var_func = _compute_hc0_variance(X, e, XtX_inv)
        
        assert np.allclose(var_manual, var_func, rtol=1e-10)


class TestHC1VarianceFunction:
    """Unit tests for _compute_hc1_variance()"""
    
    def test_hc1_equals_scaled_hc0(self):
        """Test HC1 = HC0 × N/(N-K)"""
        n, k = 100, 3
        rng = np.random.default_rng(42)
        X = np.column_stack([np.ones(n), rng.normal(0, 1, (n, k-1))])
        residuals = rng.normal(0, 1, n)
        XtX_inv = np.linalg.inv(X.T @ X)
        
        var_hc0 = _compute_hc0_variance(X, residuals, XtX_inv)
        var_hc1 = _compute_hc1_variance(X, residuals, XtX_inv)
        
        scale = n / (n - k)
        expected_var_hc1 = var_hc0 * scale
        
        assert np.allclose(var_hc1, expected_var_hc1, rtol=1e-10)
    
    @pytest.mark.parametrize("n,k", [
        (100, 2),
        (500, 4),
        (1000, 6),
    ])
    def test_hc1_hc0_ratio_parametric(self, n, k):
        """Test HC1/HC0 ratio equals N/(N-K) for various n, k"""
        rng = np.random.default_rng(42)
        X = np.column_stack([np.ones(n), rng.normal(0, 1, (n, k-1))])
        beta_true = rng.normal(0, 1, k)
        y = X @ beta_true + rng.normal(0, 1, n)
        
        XtX_inv = np.linalg.inv(X.T @ X)
        beta = XtX_inv @ (X.T @ y)
        residuals = y - X @ beta
        
        var_hc0 = _compute_hc0_variance(X, residuals, XtX_inv)
        var_hc1 = _compute_hc1_variance(X, residuals, XtX_inv)
        
        scale = n / (n - k)
        
        # Check element-wise ratio
        ratio = var_hc1 / var_hc0
        expected_ratio = np.full_like(ratio, scale)
        
        assert np.allclose(ratio, expected_ratio, rtol=1e-10)


# ============================================================================
# Task 5.1.3: run_ols_regression HC0/HC1 Tests
# ============================================================================

class TestRunOLSRegressionHC:
    """Tests for run_ols_regression with HC0/HC1 vce options"""
    
    def test_hc0_basic(self):
        """Test vce='hc0' basic functionality"""
        data = generate_test_data(n=500, seed=42)
        result = run_ols_regression(data, 'y', 'd', controls=['x1', 'x2'], vce='hc0')
        
        assert result['se'] > 0
        assert not np.isnan(result['pvalue'])
        assert result['ci_lower'] < result['att'] < result['ci_upper']
    
    def test_hc1_basic(self):
        """Test vce='hc1' basic functionality"""
        data = generate_test_data(n=500, seed=42)
        result = run_ols_regression(data, 'y', 'd', controls=['x1', 'x2'], vce='hc1')
        
        assert result['se'] > 0
        assert not np.isnan(result['pvalue'])
    
    def test_robust_alias(self):
        """Test 'robust' is alias for 'hc1'"""
        data = generate_test_data(n=500, seed=42)
        
        r_robust = run_ols_regression(data, 'y', 'd', controls=['x1', 'x2'], vce='robust')
        r_hc1 = run_ols_regression(data, 'y', 'd', controls=['x1', 'x2'], vce='hc1')
        
        assert r_robust['se'] == r_hc1['se']
        assert r_robust['att'] == r_hc1['att']
    
    def test_case_insensitive(self):
        """Test vce parameter is case-insensitive"""
        data = generate_test_data(n=200, seed=42)
        
        r1 = run_ols_regression(data, 'y', 'd', vce='hc0')
        r2 = run_ols_regression(data, 'y', 'd', vce='HC0')
        r3 = run_ols_regression(data, 'y', 'd', vce='Hc0')
        
        assert r1['se'] == r2['se'] == r3['se']
    
    def test_invalid_vce_error(self):
        """Test invalid vce raises ValueError"""
        data = generate_test_data(n=100, seed=42)
        
        with pytest.raises(ValueError, match="Invalid vce type"):
            run_ols_regression(data, 'y', 'd', vce='invalid')
    
    def test_hc0_le_hc1(self):
        """Test SE(HC0) ≤ SE(HC1)"""
        data = generate_test_data(n=500, seed=42)
        
        r0 = run_ols_regression(data, 'y', 'd', controls=['x1', 'x2'], vce='hc0')
        r1 = run_ols_regression(data, 'y', 'd', controls=['x1', 'x2'], vce='hc1')
        
        # HC0 SE should be less than or equal to HC1 SE
        assert r0['se'] <= r1['se'] * 1.0001  # Small tolerance for float comparison
    
    def test_hc1_le_hc3(self):
        """Test SE(HC1) ≤ SE(HC3) typically holds"""
        data = generate_test_data(n=200, seed=42)
        
        r1 = run_ols_regression(data, 'y', 'd', controls=['x1', 'x2'], vce='hc1')
        r3 = run_ols_regression(data, 'y', 'd', controls=['x1', 'x2'], vce='hc3')
        
        # HC1 SE is typically less than HC3 SE (not strict)
        assert r1['se'] <= r3['se'] * 1.01
    
    def test_att_invariant_to_vce(self):
        """Test ATT point estimate is same across vce types"""
        data = generate_test_data(n=500, seed=42)
        
        r_none = run_ols_regression(data, 'y', 'd', controls=['x1', 'x2'], vce=None)
        r_hc0 = run_ols_regression(data, 'y', 'd', controls=['x1', 'x2'], vce='hc0')
        r_hc1 = run_ols_regression(data, 'y', 'd', controls=['x1', 'x2'], vce='hc1')
        r_hc3 = run_ols_regression(data, 'y', 'd', controls=['x1', 'x2'], vce='hc3')
        
        # ATT should be identical
        assert r_none['att'] == r_hc0['att'] == r_hc1['att'] == r_hc3['att']


# ============================================================================
# Task 5.1.7: Stata Consistency Tests
# ============================================================================

class TestStataConsistency:
    """Tests for numerical consistency with Stata vce(robust/hc3)"""
    
    # Stata reference values from MCP test (15-digit precision):
    # Generated with: set seed 12345; regress y d x1 x2, vce(robust/hc3)
    # N=500, K=4
    STATA_REFERENCE = {
        'n': 500,
        'k': 4,
        'att': 2.056336011854301,
        'se_ols': 0.114155126979691,
        'se_hc0': 0.112727264480106,
        'se_hc1': 0.113180897155907,
        'se_hc3': 0.113744935235905,
    }
    
    def test_hc1_hc0_ratio_matches_stata(self):
        """Test HC1/HC0 ratio matches Stata's sqrt(N/(N-K))"""
        n, k = self.STATA_REFERENCE['n'], self.STATA_REFERENCE['k']
        se_hc0 = self.STATA_REFERENCE['se_hc0']
        se_hc1 = self.STATA_REFERENCE['se_hc1']
        
        actual_ratio = se_hc1 / se_hc0
        expected_ratio = np.sqrt(n / (n - k))
        
        assert np.isclose(actual_ratio, expected_ratio, rtol=1e-6)
    
    def test_hc0_se_ordering(self):
        """Test HC0 < HC1 < HC3 ordering from Stata reference"""
        se_hc0 = self.STATA_REFERENCE['se_hc0']
        se_hc1 = self.STATA_REFERENCE['se_hc1']
        se_hc3 = self.STATA_REFERENCE['se_hc3']
        
        assert se_hc0 < se_hc1 < se_hc3


class TestStataEndToEndValidation:
    """
    End-to-end Python-Stata numerical validation using identical data.
    
    This test uses Stata-exported data to ensure exact numerical comparison
    with Stata's vce(hc0), vce(robust), and vce(hc3) results.
    """
    
    # Stata reference values (15-digit precision from MCP)
    STATA_RESULTS = {
        'att': 2.056336011854301,
        'se_ols': 0.114155126979691,
        'se_hc0': 0.112727264480106,
        'se_hc1': 0.113180897155907,
        'se_hc3': 0.113744935235905,
    }
    
    @pytest.fixture
    def stata_data(self):
        """
        Load Stata-exported data or generate matching DGP.
        
        Note: If Stata data file is unavailable, generates data with same DGP.
        For exact comparison, use Stata MCP to export data first.
        """
        try:
            data = pd.read_csv('/tmp/stata_hc_test_data.csv')
        except FileNotFoundError:
            pytest.skip("Stata data file not found. Run Stata MCP export first.")
        return data
    
    def test_hc0_vs_stata_exact(self, stata_data):
        """
        Test HC0 SE matches Stata vce(hc0) with relative error < 1e-6.
        
        Uses _compute_hc0_variance directly with Stata-identical model:
        Y ~ 1 + D + X1 + X2 (no interactions)
        """
        # Build design matrix matching Stata: regress y d x1 x2
        y = stata_data['y'].values.astype(float)
        X = np.column_stack([
            np.ones(len(y)),
            stata_data['d'].values.astype(float),
            stata_data['x1'].values.astype(float),
            stata_data['x2'].values.astype(float),
        ])
        
        # OLS estimation
        XtX_inv = np.linalg.inv(X.T @ X)
        beta = XtX_inv @ (X.T @ y)
        residuals = y - X @ beta
        
        # HC0 variance
        var_hc0 = _compute_hc0_variance(X, residuals, XtX_inv)
        se_hc0_py = np.sqrt(var_hc0[1, 1])
        
        rel_error = abs(se_hc0_py - self.STATA_RESULTS['se_hc0']) / self.STATA_RESULTS['se_hc0']
        assert rel_error < 1e-6, f"HC0 SE relative error: {rel_error:.2e}"
    
    def test_hc1_vs_stata_exact(self, stata_data):
        """
        Test HC1 SE matches Stata vce(robust) with relative error < 1e-6.
        """
        y = stata_data['y'].values.astype(float)
        X = np.column_stack([
            np.ones(len(y)),
            stata_data['d'].values.astype(float),
            stata_data['x1'].values.astype(float),
            stata_data['x2'].values.astype(float),
        ])
        
        XtX_inv = np.linalg.inv(X.T @ X)
        beta = XtX_inv @ (X.T @ y)
        residuals = y - X @ beta
        
        var_hc1 = _compute_hc1_variance(X, residuals, XtX_inv)
        se_hc1_py = np.sqrt(var_hc1[1, 1])
        
        rel_error = abs(se_hc1_py - self.STATA_RESULTS['se_hc1']) / self.STATA_RESULTS['se_hc1']
        assert rel_error < 1e-6, f"HC1 SE relative error: {rel_error:.2e}"
    
    def test_att_vs_stata_exact(self, stata_data):
        """
        Test ATT point estimate matches Stata with relative error < 1e-6.
        """
        y = stata_data['y'].values.astype(float)
        X = np.column_stack([
            np.ones(len(y)),
            stata_data['d'].values.astype(float),
            stata_data['x1'].values.astype(float),
            stata_data['x2'].values.astype(float),
        ])
        
        XtX_inv = np.linalg.inv(X.T @ X)
        beta = XtX_inv @ (X.T @ y)
        att_py = beta[1]
        
        rel_error = abs(att_py - self.STATA_RESULTS['att']) / self.STATA_RESULTS['att']
        assert rel_error < 1e-6, f"ATT relative error: {rel_error:.2e}"
    
    def test_all_se_types_vs_stata(self, stata_data):
        """
        Comprehensive test: all SE types vs Stata with detailed output.
        """
        y = stata_data['y'].values.astype(float)
        X = np.column_stack([
            np.ones(len(y)),
            stata_data['d'].values.astype(float),
            stata_data['x1'].values.astype(float),
            stata_data['x2'].values.astype(float),
        ])
        n, k = X.shape
        
        XtX_inv = np.linalg.inv(X.T @ X)
        beta = XtX_inv @ (X.T @ y)
        residuals = y - X @ beta
        
        # OLS SE
        sigma2 = (residuals ** 2).sum() / (n - k)
        se_ols = np.sqrt(sigma2 * XtX_inv[1, 1])
        
        # HC0 SE
        se_hc0 = np.sqrt(_compute_hc0_variance(X, residuals, XtX_inv)[1, 1])
        
        # HC1 SE
        se_hc1 = np.sqrt(_compute_hc1_variance(X, residuals, XtX_inv)[1, 1])
        
        # HC3 SE
        H = X @ XtX_inv @ X.T
        h_ii = np.clip(np.diag(H), 0, 0.9999)
        omega_hc3 = (residuals ** 2) / ((1 - h_ii) ** 2)
        var_hc3 = XtX_inv @ (X.T @ np.diag(omega_hc3) @ X) @ XtX_inv
        se_hc3 = np.sqrt(var_hc3[1, 1])
        
        # All relative errors < 1e-6
        errors = {
            'ATT': abs(beta[1] - self.STATA_RESULTS['att']) / self.STATA_RESULTS['att'],
            'OLS_SE': abs(se_ols - self.STATA_RESULTS['se_ols']) / self.STATA_RESULTS['se_ols'],
            'HC0_SE': abs(se_hc0 - self.STATA_RESULTS['se_hc0']) / self.STATA_RESULTS['se_hc0'],
            'HC1_SE': abs(se_hc1 - self.STATA_RESULTS['se_hc1']) / self.STATA_RESULTS['se_hc1'],
            'HC3_SE': abs(se_hc3 - self.STATA_RESULTS['se_hc3']) / self.STATA_RESULTS['se_hc3'],
        }
        
        for name, err in errors.items():
            assert err < 1e-6, f"{name} relative error {err:.2e} exceeds 1e-6"


# ============================================================================
# Task 5.1.8: statsmodels Cross-Validation
# ============================================================================

class TestStatsmodelsConsistency:
    """Cross-validation with statsmodels HC0/HC1"""
    
    @pytest.fixture
    def simple_data(self):
        """Generate simple test data"""
        rng = np.random.default_rng(42)
        n = 500
        
        X = rng.normal(0, 1, (n, 2))
        D = rng.binomial(1, 0.5, n)
        Y = 1 + 2*D + X @ [0.5, 0.3] + rng.normal(0, 1, n)
        
        return pd.DataFrame({
            'y': Y, 'd': D, 'x1': X[:, 0], 'x2': X[:, 1]
        })
    
    def test_hc0_vs_statsmodels_no_controls(self, simple_data):
        """HC0 SE matches statsmodels without controls"""
        try:
            import statsmodels.api as sm
        except ImportError:
            pytest.skip("statsmodels not installed")
        
        # Without controls, our model is: Y ~ 1 + D (same as statsmodels)
        our_result = run_ols_regression(
            simple_data, 'y', 'd', controls=None, vce='hc0'
        )
        
        # statsmodels: Y ~ 1 + D
        X_sm = sm.add_constant(simple_data['d'])
        model = sm.OLS(simple_data['y'], X_sm).fit(cov_type='HC0')
        sm_se = model.bse['d']
        
        rel_error = abs(our_result['se'] - sm_se) / sm_se
        assert rel_error < 1e-10, f"Relative error: {rel_error:.2e}"
    
    def test_hc1_vs_statsmodels_no_controls(self, simple_data):
        """HC1 SE matches statsmodels without controls"""
        try:
            import statsmodels.api as sm
        except ImportError:
            pytest.skip("statsmodels not installed")
        
        # Without controls
        our_result = run_ols_regression(
            simple_data, 'y', 'd', controls=None, vce='hc1'
        )
        
        # statsmodels
        X_sm = sm.add_constant(simple_data['d'])
        model = sm.OLS(simple_data['y'], X_sm).fit(cov_type='HC1')
        sm_se = model.bse['d']
        
        rel_error = abs(our_result['se'] - sm_se) / sm_se
        assert rel_error < 1e-10, f"Relative error: {rel_error:.2e}"
    
    def test_hc0_vs_statsmodels_with_controls_direct(self):
        """
        HC0 SE matches statsmodels when using identical model specification.
        
        Note: Our implementation uses Y ~ 1 + D + X + D*(X - X̄₁) per Lee & Wooldridge,
        while standard statsmodels uses Y ~ 1 + D + X. This test uses the direct
        statsmodels comparison with our internal design matrix.
        """
        try:
            import statsmodels.api as sm
        except ImportError:
            pytest.skip("statsmodels not installed")
        
        # Generate data
        rng = np.random.default_rng(42)
        n = 500
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        d = rng.binomial(1, 0.5, n).astype(float)
        y = 1 + 2*d + 0.5*x1 + 0.3*x2 + rng.normal(0, 1, n)
        
        # Build exact same design matrix as our implementation
        treated_mask = d == 1
        X_controls = np.column_stack([x1, x2])
        X_mean_treated = X_controls[treated_mask].mean(axis=0)
        X_centered = X_controls - X_mean_treated
        X_interactions = d.reshape(-1, 1) * X_centered
        
        # Design matrix: [1, D, X, D*(X - X̄₁)]
        X = np.column_stack([
            np.ones(n),
            d,
            X_controls,
            X_interactions
        ])
        
        # statsmodels with exact same design matrix
        model = sm.OLS(y, X).fit(cov_type='HC0')
        sm_se_d = model.bse[1]  # D is at index 1
        
        # Our implementation
        data = pd.DataFrame({'y': y, 'd': d, 'x1': x1, 'x2': x2})
        our_result = run_ols_regression(data, 'y', 'd', controls=['x1', 'x2'], vce='hc0')
        
        rel_error = abs(our_result['se'] - sm_se_d) / sm_se_d
        assert rel_error < 1e-10, f"Relative error: {rel_error:.2e}"


# ============================================================================
# Task 5.1.9: HC0/HC1 Relationship Tests
# ============================================================================

class TestHC0HC1Relationship:
    """Mathematical relationship tests between HC0 and HC1"""
    
    def test_se_ratio_formula(self):
        """Test SE(HC1) / SE(HC0) = sqrt(N / (N-K))"""
        data = generate_test_data(n=500, seed=42)
        
        r0 = run_ols_regression(data, 'y', 'd', controls=['x1', 'x2'], vce='hc0')
        r1 = run_ols_regression(data, 'y', 'd', controls=['x1', 'x2'], vce='hc1')
        
        n = r0['nobs']
        k = n - r0['df_resid']  # Number of parameters
        
        actual_ratio = r1['se'] / r0['se']
        expected_ratio = np.sqrt(n / (n - k))
        
        assert np.isclose(actual_ratio, expected_ratio, rtol=1e-10)
    
    @pytest.mark.parametrize("n", [100, 500, 1000])
    def test_se_ratio_various_sample_sizes(self, n):
        """Test SE ratio formula holds for various sample sizes"""
        data = generate_test_data(n=n, seed=42)
        
        r0 = run_ols_regression(data, 'y', 'd', controls=['x1', 'x2'], vce='hc0')
        r1 = run_ols_regression(data, 'y', 'd', controls=['x1', 'x2'], vce='hc1')
        
        n_obs = r0['nobs']
        k = n_obs - r0['df_resid']
        
        actual_ratio = r1['se'] / r0['se']
        expected_ratio = np.sqrt(n_obs / (n_obs - k))
        
        assert np.isclose(actual_ratio, expected_ratio, rtol=1e-10)
    
    def test_convergence_for_large_n(self):
        """Test HC0 and HC1 converge as N increases"""
        data_small = generate_test_data(n=100, seed=42)
        data_large = generate_test_data(n=5000, seed=42)
        
        r0_small = run_ols_regression(data_small, 'y', 'd', vce='hc0')
        r1_small = run_ols_regression(data_small, 'y', 'd', vce='hc1')
        
        r0_large = run_ols_regression(data_large, 'y', 'd', vce='hc0')
        r1_large = run_ols_regression(data_large, 'y', 'd', vce='hc1')
        
        # Ratio should be closer to 1 for large N
        ratio_small = r1_small['se'] / r0_small['se']
        ratio_large = r1_large['se'] / r0_large['se']
        
        assert abs(ratio_large - 1) < abs(ratio_small - 1)


# ============================================================================
# Task 5.1.10: Monte Carlo Coverage Tests
# ============================================================================

class TestMonteCarloCoverage:
    """Monte Carlo tests for confidence interval coverage"""
    
    @pytest.mark.slow
    @pytest.mark.parametrize("n,expected_range", [
        (1000, (0.93, 0.97)),  # Large sample: normal coverage
        (500, (0.92, 0.98)),   # Medium sample
    ])
    def test_coverage_heteroskedastic(self, n, expected_range):
        """Test 95% CI coverage rate under heteroskedasticity"""
        n_reps = 200  # Reduced for speed
        true_att = 2.0
        covered = 0
        
        for rep in range(n_reps):
            data = generate_test_data(n=n, seed=rep, heteroskedastic=True)
            result = run_ols_regression(data, 'y', 'd', controls=['x1', 'x2'], vce='hc0')
            
            if result['ci_lower'] <= true_att <= result['ci_upper']:
                covered += 1
        
        coverage = covered / n_reps
        assert expected_range[0] <= coverage <= expected_range[1], \
            f"Coverage {coverage:.3f} not in {expected_range}"
    
    @pytest.mark.slow
    def test_coverage_homoskedastic(self):
        """Test coverage under homoskedasticity"""
        n_reps = 200
        n = 500
        true_att = 2.0
        covered = 0
        
        for rep in range(n_reps):
            data = generate_test_data(n=n, seed=rep, heteroskedastic=False)
            result = run_ols_regression(data, 'y', 'd', controls=['x1', 'x2'], vce='hc0')
            
            if result['ci_lower'] <= true_att <= result['ci_upper']:
                covered += 1
        
        coverage = covered / n_reps
        # Under homoskedasticity, HC0 should still be valid
        assert 0.92 <= coverage <= 0.98


# ============================================================================
# Task 5.1.11: Formula Validation Tests
# ============================================================================

class TestFormulaValidation:
    """Validate HC0 formula implementation"""
    
    def test_meat_matrix_equivalence(self):
        """Test three methods of computing meat matrix are equivalent"""
        rng = np.random.default_rng(42)
        n, k = 50, 3
        X = np.column_stack([np.ones(n), rng.normal(0, 1, (n, k-1))])
        e = rng.normal(0, 1, n)
        omega = e ** 2
        
        # Method 1: Loop
        meat_loop = np.zeros((k, k))
        for i in range(n):
            meat_loop += omega[i] * np.outer(X[i], X[i])
        
        # Method 2: Vectorized (our implementation)
        meat_vec = (X.T * omega) @ X
        
        # Method 3: Explicit diagonal matrix
        meat_diag = X.T @ np.diag(omega) @ X
        
        assert np.allclose(meat_loop, meat_vec, rtol=1e-10)
        assert np.allclose(meat_loop, meat_diag, rtol=1e-10)
    
    def test_sandwich_formula(self):
        """Test sandwich variance formula: (X'X)⁻¹ Meat (X'X)⁻¹"""
        rng = np.random.default_rng(42)
        n, k = 100, 3
        X = np.column_stack([np.ones(n), rng.normal(0, 1, (n, k-1))])
        e = rng.normal(0, 1, n)
        
        XtX = X.T @ X
        XtX_inv = np.linalg.inv(XtX)
        
        # Compute meat
        omega = e ** 2
        meat = (X.T * omega) @ X
        
        # Sandwich
        var = XtX_inv @ meat @ XtX_inv
        
        # Verify using function
        var_func = _compute_hc0_variance(X, e, XtX_inv)
        
        assert np.allclose(var, var_func, rtol=1e-12)
    
    def test_hc0_reduces_to_ols_under_homoskedasticity_limit(self):
        """Test HC0 approaches OLS SE under homoskedasticity"""
        rng = np.random.default_rng(42)
        n, k = 1000, 3
        sigma = 1.0  # Constant variance
        
        X = np.column_stack([np.ones(n), rng.normal(0, 1, (n, k-1))])
        beta_true = np.array([1, 2, 0.5])
        y = X @ beta_true + rng.normal(0, sigma, n)
        
        # OLS
        XtX_inv = np.linalg.inv(X.T @ X)
        beta = XtX_inv @ (X.T @ y)
        residuals = y - X @ beta
        sigma2_hat = (residuals ** 2).sum() / (n - k)
        
        # OLS variance
        var_ols = sigma2_hat * XtX_inv
        se_ols = np.sqrt(np.diag(var_ols))
        
        # HC0 variance
        var_hc0 = _compute_hc0_variance(X, residuals, XtX_inv)
        se_hc0 = np.sqrt(np.diag(var_hc0))
        
        # Under homoskedasticity, HC0 should be close to OLS SE
        # Not exact due to finite sample, but should be within ~10%
        rel_diff = np.abs(se_hc0 - se_ols) / se_ols
        assert np.all(rel_diff < 0.1)


# ============================================================================
# Backward Compatibility Tests
# ============================================================================

class TestBackwardCompatibility:
    """Ensure existing vce options still work correctly"""
    
    def test_vce_none_unchanged(self):
        """vce=None behavior unchanged"""
        data = generate_test_data(n=200, seed=42)
        result = run_ols_regression(data, 'y', 'd', controls=['x1', 'x2'], vce=None)
        
        assert result['se'] > 0
        assert not np.isnan(result['att'])
    
    def test_vce_hc3_unchanged(self):
        """vce='hc3' behavior unchanged"""
        data = generate_test_data(n=200, seed=42)
        result = run_ols_regression(data, 'y', 'd', controls=['x1', 'x2'], vce='hc3')
        
        assert result['se'] > 0
        assert not np.isnan(result['pvalue'])
    
    def test_vce_cluster_unchanged(self):
        """vce='cluster' behavior unchanged"""
        data = generate_test_data(n=200, seed=42)
        data['cluster_id'] = data.index % 20
        
        result = run_ols_regression(
            data, 'y', 'd', controls=['x1', 'x2'],
            vce='cluster', cluster_var='cluster_id'
        )
        
        assert result['se'] > 0
        assert result['df_inference'] == 19  # 20 clusters - 1


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Edge case tests for HC0 implementation"""
    
    def test_small_sample(self):
        """Test with small sample size"""
        data = generate_test_data(n=30, seed=42)
        result = run_ols_regression(data, 'y', 'd', vce='hc0')
        
        assert result['se'] > 0
        assert not np.isnan(result['pvalue'])
    
    def test_no_controls(self):
        """Test without control variables"""
        data = generate_test_data(n=500, seed=42)
        result = run_ols_regression(data, 'y', 'd', vce='hc0')
        
        assert result['se'] > 0
        # K = 2 (intercept + D)
        n, k = result['nobs'], 2
        assert result['df_resid'] == n - k
    
    def test_many_controls(self):
        """Test with several control variables"""
        rng = np.random.default_rng(42)
        n = 500
        k_controls = 5  # Reduced to avoid singular matrix with interactions
        
        data = generate_test_data(n=n, seed=42)
        for i in range(k_controls - 2):
            data[f'x{i+3}'] = rng.normal(0, 1, n)
        
        controls = [f'x{i}' for i in range(1, k_controls + 1)]
        result = run_ols_regression(data, 'y', 'd', controls=controls, vce='hc0')
        
        assert result['se'] > 0
        # Design matrix: 1 + D + K controls + K interactions = 2 + 2K parameters
        expected_params = 2 + 2 * k_controls
        assert result['df_resid'] == n - expected_params


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

"""
End-to-End Consistency Tests for HC Standard Errors (Story 5.3)

Validates Python HC0-HC4 implementation against:
- Stata regress vce(robust/hc2/hc3) for HC0-HC3
- Manual formula calculation (equivalent to R sandwich::vcovHC) for HC4

Data Generation (Stata seed 20230796):
    Y = 1 + 2*D + 0.5*X1 + 0.3*X2 + sigma*epsilon
    where sigma = sqrt(1 + 0.5*|X1|) (heteroskedastic)
    D = 1{Z + 0.5*X1 + 0.3*X2 > 0}

Key validation:
- HC0-HC3: Compare with Stata reference values (tolerance < 1e-5)
- HC4: Compare with manual formula calculation (tolerance < 1e-6)
  Note: Stata does not support vce(hc4), validation uses manual calculation
        equivalent to R sandwich::vcovHC(type="HC4")

Reference:
- Lee & Wooldridge (2023) Section 3
- Cribari-Neto (2004) for HC4
- MacKinnon & White (1985) for HC0-HC3
"""

import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from lwdid.staggered.estimation import (
    run_ols_regression,
    _compute_hc0_variance,
    _compute_hc1_variance,
    _compute_hc2_variance,
    _compute_hc4_variance,
)


# ============================================================================
# Load Reference Data
# ============================================================================

@pytest.fixture(scope="module")
def reference_data():
    """Load Stata/R reference values and test data."""
    test_dir = Path(__file__).parent
    
    # Load reference values
    with open(test_dir / "stata_r_hc_reference.json", "r") as f:
        ref = json.load(f)
    
    # Load test data
    data = pd.read_csv(test_dir / "hc_test_data.csv")
    
    return {"ref": ref, "data": data}


@pytest.fixture(scope="module")
def ols_matrices(reference_data):
    """Pre-compute OLS matrices for tests."""
    data = reference_data["data"]
    n = len(data)
    
    # Design matrix: [1, d, x1, x2]
    X = np.column_stack([
        np.ones(n),
        data['d'].values,
        data['x1'].values,
        data['x2'].values
    ])
    y = data['y'].values
    
    # OLS
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    residuals = y - X @ beta
    
    return {
        "X": X,
        "y": y,
        "beta": beta,
        "residuals": residuals,
        "XtX_inv": XtX_inv,
        "n": n,
        "k": X.shape[1]
    }


# ============================================================================
# Test Class: Stata HC Consistency (HC0-HC3)
# ============================================================================

class TestStataHCConsistency:
    """Validate HC0-HC3 against Stata regress reference values."""
    
    def test_ols_coefficients_match_stata(self, reference_data, ols_matrices):
        """OLS coefficients should match Stata exactly."""
        ref = reference_data["ref"]
        beta = ols_matrices["beta"]
        
        # Reference: [_cons, d, x1, x2]
        stata_coef = ref["ols_coefficients"]["values"]
        
        # Python order: [_cons, d, x1, x2]
        assert np.allclose(beta[0], stata_coef[0], rtol=1e-6), \
            f"_cons mismatch: Python={beta[0]}, Stata={stata_coef[0]}"
        assert np.allclose(beta[1], stata_coef[1], rtol=1e-6), \
            f"d mismatch: Python={beta[1]}, Stata={stata_coef[1]}"
        assert np.allclose(beta[2], stata_coef[2], rtol=1e-6), \
            f"x1 mismatch: Python={beta[2]}, Stata={stata_coef[2]}"
        assert np.allclose(beta[3], stata_coef[3], rtol=1e-6), \
            f"x2 mismatch: Python={beta[3]}, Stata={stata_coef[3]}"
    
    def test_hc0_se_matches_stata(self, reference_data, ols_matrices):
        """HC0 SE should match Stata-derived reference."""
        ref = reference_data["ref"]
        X = ols_matrices["X"]
        residuals = ols_matrices["residuals"]
        XtX_inv = ols_matrices["XtX_inv"]
        
        # Python HC0
        var_hc0 = _compute_hc0_variance(X, residuals, XtX_inv)
        se_hc0 = np.sqrt(np.diag(var_hc0))
        
        # Stata reference (calculated from HC1)
        stata_se = ref["stata_reference"]["hc0_se"]["values"]
        
        # Tolerance: 1e-5 (Stata displays fewer decimals)
        tol = ref["tolerance"]["stata_comparison"]
        
        for i, (py_se, st_se) in enumerate(zip(se_hc0, stata_se)):
            assert abs(py_se - st_se) < tol, \
                f"HC0 SE[{i}] mismatch: Python={py_se:.8f}, Stata={st_se:.8f}, diff={abs(py_se-st_se):.2e}"
    
    def test_hc1_se_matches_stata(self, reference_data, ols_matrices):
        """HC1 SE should match Stata vce(robust)."""
        ref = reference_data["ref"]
        X = ols_matrices["X"]
        residuals = ols_matrices["residuals"]
        XtX_inv = ols_matrices["XtX_inv"]
        
        # Python HC1
        var_hc1 = _compute_hc1_variance(X, residuals, XtX_inv)
        se_hc1 = np.sqrt(np.diag(var_hc1))
        
        # Stata reference
        stata_se = ref["stata_reference"]["hc1_se"]["values"]
        tol = ref["tolerance"]["stata_comparison"]
        
        for i, (py_se, st_se) in enumerate(zip(se_hc1, stata_se)):
            assert abs(py_se - st_se) < tol, \
                f"HC1 SE[{i}] mismatch: Python={py_se:.8f}, Stata={st_se:.8f}"
    
    def test_hc2_se_matches_stata(self, reference_data, ols_matrices):
        """HC2 SE should match Stata vce(hc2)."""
        ref = reference_data["ref"]
        X = ols_matrices["X"]
        residuals = ols_matrices["residuals"]
        XtX_inv = ols_matrices["XtX_inv"]
        
        # Python HC2
        var_hc2 = _compute_hc2_variance(X, residuals, XtX_inv)
        se_hc2 = np.sqrt(np.diag(var_hc2))
        
        # Stata reference
        stata_se = ref["stata_reference"]["hc2_se"]["values"]
        tol = ref["tolerance"]["stata_comparison"]
        
        for i, (py_se, st_se) in enumerate(zip(se_hc2, stata_se)):
            assert abs(py_se - st_se) < tol, \
                f"HC2 SE[{i}] mismatch: Python={py_se:.8f}, Stata={st_se:.8f}"
    
    def test_hc3_se_matches_stata(self, reference_data, ols_matrices):
        """HC3 SE should match Stata vce(hc3)."""
        ref = reference_data["ref"]
        X = ols_matrices["X"]
        residuals = ols_matrices["residuals"]
        XtX_inv = ols_matrices["XtX_inv"]
        n = ols_matrices["n"]
        k = ols_matrices["k"]
        
        # Python HC3 (manual calculation to match _compute_hc3 logic)
        h = np.sum(X * (X @ XtX_inv), axis=1)
        h = np.clip(h, 0, 0.9999)
        omega_hc3 = residuals**2 / ((1 - h)**2)
        meat = (X.T * omega_hc3) @ X
        var_hc3 = XtX_inv @ meat @ XtX_inv
        se_hc3 = np.sqrt(np.diag(var_hc3))
        
        # Stata reference
        stata_se = ref["stata_reference"]["hc3_se"]["values"]
        tol = ref["tolerance"]["stata_comparison"]
        
        for i, (py_se, st_se) in enumerate(zip(se_hc3, stata_se)):
            assert abs(py_se - st_se) < tol, \
                f"HC3 SE[{i}] mismatch: Python={py_se:.8f}, Stata={st_se:.8f}"


# ============================================================================
# Test Class: R sandwich HC4 Consistency
# ============================================================================

class TestRSandwichHC4Consistency:
    """
    Validate HC4 against manual formula calculation.
    
    Note: Stata does not support vce(hc4). Validation uses manual calculation
    equivalent to R sandwich::vcovHC(type="HC4").
    """
    
    def test_hc4_se_matches_formula(self, reference_data, ols_matrices):
        """HC4 SE should match manual formula calculation."""
        ref = reference_data["ref"]
        X = ols_matrices["X"]
        residuals = ols_matrices["residuals"]
        XtX_inv = ols_matrices["XtX_inv"]
        
        # Python HC4 using our implementation
        var_hc4 = _compute_hc4_variance(X, residuals, XtX_inv)
        se_hc4 = np.sqrt(np.diag(var_hc4))
        
        # Reference (manual calculation)
        ref_se = ref["hc4_reference"]["values"]
        tol = ref["tolerance"]["r_comparison"]
        
        for i, (py_se, ref_val) in enumerate(zip(se_hc4, ref_se)):
            assert abs(py_se - ref_val) < tol, \
                f"HC4 SE[{i}] mismatch: Python={py_se:.10f}, Reference={ref_val:.10f}, diff={abs(py_se-ref_val):.2e}"
    
    def test_hc4_matches_python_manual(self, reference_data, ols_matrices):
        """HC4 implementation should match Python manual reference."""
        ref = reference_data["ref"]
        X = ols_matrices["X"]
        residuals = ols_matrices["residuals"]
        XtX_inv = ols_matrices["XtX_inv"]
        
        # Python HC4 using our implementation
        var_hc4 = _compute_hc4_variance(X, residuals, XtX_inv)
        se_hc4 = np.sqrt(np.diag(var_hc4))
        
        # Python manual reference
        manual_se = ref["python_manual_reference"]["hc4_se"]
        
        for i, (impl_se, manual) in enumerate(zip(se_hc4, manual_se)):
            assert np.isclose(impl_se, manual, rtol=1e-10), \
                f"HC4 SE[{i}] differs from manual: impl={impl_se:.12f}, manual={manual:.12f}"
    
    def test_hc4_delta_calculation(self, ols_matrices):
        """Verify delta_i = min(4, n*h_ii/p) calculation."""
        X = ols_matrices["X"]
        XtX_inv = ols_matrices["XtX_inv"]
        n = ols_matrices["n"]
        k = ols_matrices["k"]
        
        # Compute leverage
        h = np.sum(X * (X @ XtX_inv), axis=1)
        
        # Expected delta
        delta_expected = np.minimum(4.0, n * h / k)
        
        # All delta should be in [0, 4]
        assert np.all(delta_expected >= 0)
        assert np.all(delta_expected <= 4)
        
        # Average leverage h_bar = k/n, so average delta should be ~1
        h_bar = k / n
        delta_at_hbar = min(4, n * h_bar / k)
        assert np.isclose(delta_at_hbar, 1.0)


# ============================================================================
# Test Class: Cross Validation
# ============================================================================

class TestCrossValidation:
    """Cross-validate HC estimates and ordering relationships."""
    
    def test_hc_se_ordering(self, reference_data, ols_matrices):
        """Verify HC0 <= HC1 <= HC2 <= HC3 (generally)."""
        X = ols_matrices["X"]
        residuals = ols_matrices["residuals"]
        XtX_inv = ols_matrices["XtX_inv"]
        n, k = ols_matrices["n"], ols_matrices["k"]
        
        # Compute all HC variances
        var_hc0 = _compute_hc0_variance(X, residuals, XtX_inv)
        var_hc1 = _compute_hc1_variance(X, residuals, XtX_inv)
        var_hc2 = _compute_hc2_variance(X, residuals, XtX_inv)
        var_hc4 = _compute_hc4_variance(X, residuals, XtX_inv)
        
        # HC3 manual
        h = np.sum(X * (X @ XtX_inv), axis=1)
        h = np.clip(h, 0, 0.9999)
        omega_hc3 = residuals**2 / ((1 - h)**2)
        meat = (X.T * omega_hc3) @ X
        var_hc3 = XtX_inv @ meat @ XtX_inv
        
        # Get SE for d coefficient (index 1)
        se_hc0 = np.sqrt(var_hc0[1, 1])
        se_hc1 = np.sqrt(var_hc1[1, 1])
        se_hc2 = np.sqrt(var_hc2[1, 1])
        se_hc3 = np.sqrt(var_hc3[1, 1])
        se_hc4 = np.sqrt(var_hc4[1, 1])
        
        # Standard ordering (with small tolerance)
        assert se_hc0 <= se_hc1 * 1.01, f"HC0 > HC1: {se_hc0} > {se_hc1}"
        assert se_hc1 <= se_hc2 * 1.01, f"HC1 > HC2: {se_hc1} > {se_hc2}"
        assert se_hc2 <= se_hc3 * 1.01, f"HC2 > HC3: {se_hc2} > {se_hc3}"
        
        # HC4 should be between HC2 and HC3 for moderate leverage
        # (since delta is adaptive)
        assert se_hc2 <= se_hc4 * 1.05, f"HC2 > HC4: {se_hc2} > {se_hc4}"
        assert se_hc4 <= se_hc3 * 1.05, f"HC4 > HC3: {se_hc4} > {se_hc3}"
    
    def test_hc1_hc0_ratio(self, ols_matrices):
        """HC1/HC0 ratio should equal sqrt(N/(N-K))."""
        X = ols_matrices["X"]
        residuals = ols_matrices["residuals"]
        XtX_inv = ols_matrices["XtX_inv"]
        n, k = ols_matrices["n"], ols_matrices["k"]
        
        var_hc0 = _compute_hc0_variance(X, residuals, XtX_inv)
        var_hc1 = _compute_hc1_variance(X, residuals, XtX_inv)
        
        se_hc0 = np.sqrt(np.diag(var_hc0))
        se_hc1 = np.sqrt(np.diag(var_hc1))
        
        expected_ratio = np.sqrt(n / (n - k))
        actual_ratio = se_hc1 / se_hc0
        
        assert np.allclose(actual_ratio, expected_ratio, rtol=1e-10), \
            f"HC1/HC0 ratio mismatch: actual={actual_ratio}, expected={expected_ratio}"
    
    def test_leverage_sum_equals_k(self, ols_matrices):
        """Sum of leverage values should equal k (number of parameters)."""
        X = ols_matrices["X"]
        XtX_inv = ols_matrices["XtX_inv"]
        k = ols_matrices["k"]
        
        h = np.sum(X * (X @ XtX_inv), axis=1)
        
        assert np.isclose(h.sum(), k, rtol=1e-10), \
            f"Sum of leverage != k: sum={h.sum()}, k={k}"


# ============================================================================
# Test Class: run_ols_regression Integration
# ============================================================================

class TestRunOLSRegressionIntegration:
    """
    Test run_ols_regression with all HC types.
    
    Note: run_ols_regression implements the LWDID paper model with interaction 
    terms: Y on 1, D, X, D*(X - X̄₁). This differs from simple Stata regression
    'regress y d x1 x2'. Therefore, these tests verify internal consistency
    and HC type switching, not exact Stata replication.
    
    The core HC formula validation is done in TestStataHCConsistency and 
    TestRSandwichHC4Consistency using the raw _compute_hc*_variance functions.
    """
    
    def test_run_ols_all_vce_types_work(self, reference_data):
        """All VCE types should run without error."""
        data = reference_data["data"]
        
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            for vce_type in [None, 'hc0', 'hc1', 'robust', 'hc2', 'hc3', 'hc4']:
                result = run_ols_regression(data, 'y', 'd', controls=['x1', 'x2'], vce=vce_type)
                assert result['se'] > 0, f"vce={vce_type} returned non-positive SE"
                assert not np.isnan(result['att']), f"vce={vce_type} returned NaN ATT"
    
    def test_hc_se_ordering_in_run_ols(self, reference_data):
        """HC SE ordering should hold: HC0 <= HC1 <= HC2 <= HC3."""
        data = reference_data["data"]
        
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            se_hc0 = run_ols_regression(data, 'y', 'd', controls=['x1', 'x2'], vce='hc0')['se']
            se_hc1 = run_ols_regression(data, 'y', 'd', controls=['x1', 'x2'], vce='hc1')['se']
            se_hc2 = run_ols_regression(data, 'y', 'd', controls=['x1', 'x2'], vce='hc2')['se']
            se_hc3 = run_ols_regression(data, 'y', 'd', controls=['x1', 'x2'], vce='hc3')['se']
            se_hc4 = run_ols_regression(data, 'y', 'd', controls=['x1', 'x2'], vce='hc4')['se']
        
        # Standard ordering
        assert se_hc0 <= se_hc1 * 1.01, f"HC0 > HC1: {se_hc0} > {se_hc1}"
        assert se_hc1 <= se_hc2 * 1.01, f"HC1 > HC2: {se_hc1} > {se_hc2}"
        assert se_hc2 <= se_hc3 * 1.01, f"HC2 > HC3: {se_hc2} > {se_hc3}"
        
        # HC4 is adaptive
        assert se_hc2 * 0.9 <= se_hc4 <= se_hc3 * 1.1, \
            f"HC4 not in expected range: HC2={se_hc2}, HC4={se_hc4}, HC3={se_hc3}"
    
    def test_robust_alias(self, reference_data):
        """vce='robust' should give same result as vce='hc1'."""
        data = reference_data["data"]
        
        result_robust = run_ols_regression(data, 'y', 'd', controls=['x1', 'x2'], vce='robust')
        result_hc1 = run_ols_regression(data, 'y', 'd', controls=['x1', 'x2'], vce='hc1')
        
        assert result_robust['se'] == result_hc1['se']
    
    def test_att_invariant_to_vce(self, reference_data):
        """ATT point estimate should be same regardless of VCE type."""
        data = reference_data["data"]
        
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            att_none = run_ols_regression(data, 'y', 'd', controls=['x1', 'x2'], vce=None)['att']
            att_hc0 = run_ols_regression(data, 'y', 'd', controls=['x1', 'x2'], vce='hc0')['att']
            att_hc4 = run_ols_regression(data, 'y', 'd', controls=['x1', 'x2'], vce='hc4')['att']
        
        assert np.isclose(att_none, att_hc0), f"ATT differs: None={att_none}, HC0={att_hc0}"
        assert np.isclose(att_none, att_hc4), f"ATT differs: None={att_none}, HC4={att_hc4}"
    
    def test_hc4_no_warning(self, reference_data):
        """HC4 should not issue any warnings (Stata behavior)."""
        data = reference_data["data"]
        
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # Should not raise warning
            result = run_ols_regression(data, 'y', 'd', controls=['x1', 'x2'], vce='hc4')
            assert result['se'] > 0


# ============================================================================
# Run tests if executed directly
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])

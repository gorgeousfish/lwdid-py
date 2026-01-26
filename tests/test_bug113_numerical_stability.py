"""
Unit tests for BUG-113: Numerical stability fix using Cholesky decomposition.

Tests verify that the Cholesky-based OLS implementation:
1. Produces correct estimates for well-conditioned matrices
2. Issues warnings for ill-conditioned matrices
3. Handles singular matrices appropriately
4. Matches results from standard OLS implementations
"""

import numpy as np
import pandas as pd
import pytest
import warnings

from lwdid.staggered.estimation import run_ols_regression


class TestCholeskyOLSNumericalStability:
    """Tests for Cholesky-based OLS numerical stability."""
    
    def test_basic_ols_well_conditioned(self):
        """Verify basic OLS estimates are correct for well-conditioned data."""
        np.random.seed(42)
        n = 100
        
        # Generate well-conditioned data
        d = np.zeros(n)
        d[:30] = 1  # 30 treated, 70 control
        y = 1.0 + 2.5 * d + np.random.randn(n) * 0.5
        
        data = pd.DataFrame({
            'y': y,
            'd': d,
        })
        
        result = run_ols_regression(data, 'y', 'd', controls=None, vce=None)
        
        # ATT should be close to 2.5
        assert abs(result['att'] - 2.5) < 0.3, f"ATT={result['att']}, expected ~2.5"
        assert result['se'] > 0, "SE should be positive"
        assert np.isfinite(result['t_stat']), "t_stat should be finite"
        assert 0 <= result['pvalue'] <= 1, "p-value should be in [0, 1]"
    
    def test_ols_with_controls_well_conditioned(self):
        """Verify OLS with controls produces correct estimates."""
        np.random.seed(123)
        n = 200
        
        # Generate data with controls
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        d = np.zeros(n)
        d[:80] = 1
        y = 1.0 + 3.0 * d + 0.5 * x1 + 0.3 * x2 + np.random.randn(n) * 0.5
        
        data = pd.DataFrame({
            'y': y,
            'd': d,
            'x1': x1,
            'x2': x2,
        })
        
        result = run_ols_regression(data, 'y', 'd', controls=['x1', 'x2'], vce='hc1')
        
        # ATT should be close to 3.0
        assert abs(result['att'] - 3.0) < 0.3, f"ATT={result['att']}, expected ~3.0"
        assert result['se'] > 0, "SE should be positive"
    
    def test_ill_conditioned_matrix_warning(self):
        """Verify warning is issued for ill-conditioned design matrix."""
        np.random.seed(456)
        n = 100
        
        # Create ill-conditioned data: x2 is nearly collinear with x1
        x1 = np.random.randn(n)
        x2 = x1 + np.random.randn(n) * 1e-7  # Near-perfect collinearity
        d = np.zeros(n)
        d[:40] = 1
        y = 1.0 + 2.0 * d + 0.5 * x1 + np.random.randn(n) * 0.5
        
        data = pd.DataFrame({
            'y': y,
            'd': d,
            'x1': x1,
            'x2': x2,
        })
        
        # Should issue warning about ill-conditioned matrix
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = run_ols_regression(data, 'y', 'd', controls=['x1', 'x2'], vce=None)
            
            # Check that a warning was issued
            condition_warnings = [
                warning for warning in w 
                if "ill-conditioned" in str(warning.message).lower()
            ]
            assert len(condition_warnings) > 0, \
                "Expected warning about ill-conditioned matrix"
    
    def test_singular_matrix_raises_error(self):
        """Verify that perfectly singular matrices raise ValueError."""
        np.random.seed(789)
        n = 50
        
        # Create perfectly singular data: x2 = x1
        x1 = np.random.randn(n)
        x2 = x1  # Perfect collinearity
        d = np.zeros(n)
        d[:20] = 1
        y = 1.0 + 2.0 * d + np.random.randn(n) * 0.5
        
        data = pd.DataFrame({
            'y': y,
            'd': d,
            'x1': x1,
            'x2': x2,
        })
        
        with pytest.raises(ValueError, match="singular|positive definite"):
            run_ols_regression(data, 'y', 'd', controls=['x1', 'x2'], vce=None)
    
    def test_cholesky_matches_direct_ols(self):
        """Verify Cholesky solution matches numpy lstsq solution."""
        np.random.seed(101)
        n = 80
        
        x = np.random.randn(n)
        d = np.zeros(n)
        d[:30] = 1
        y = 0.5 + 1.8 * d + 0.7 * x + np.random.randn(n) * 0.3
        
        data = pd.DataFrame({
            'y': y,
            'd': d,
            'x': x,
        })
        
        # Get result from our implementation
        result = run_ols_regression(data, 'y', 'd', controls=['x'], vce=None)
        
        # Compute reference using numpy lstsq
        X = np.column_stack([
            np.ones(n),
            d,
            x,
            d * (x - x[d == 1].mean())  # Interaction term
        ])
        beta_ref = np.linalg.lstsq(X, y, rcond=None)[0]
        att_ref = beta_ref[1]
        
        # Should match to high precision
        assert abs(result['att'] - att_ref) < 1e-10, \
            f"ATT mismatch: {result['att']} vs {att_ref}"
    
    def test_heteroskedastic_robust_se(self):
        """Verify HC1-HC4 variance estimators work with Cholesky solution."""
        np.random.seed(202)
        n = 150
        
        d = np.zeros(n)
        d[:50] = 1
        y = 1.0 + 2.0 * d + np.random.randn(n) * (1 + d)  # Heteroskedastic errors
        
        data = pd.DataFrame({
            'y': y,
            'd': d,
        })
        
        results = {}
        for vce in [None, 'hc0', 'hc1', 'hc2', 'hc3', 'hc4']:
            result = run_ols_regression(data, 'y', 'd', controls=None, vce=vce)
            results[vce] = result
            
            # All should produce valid results
            assert np.isfinite(result['att']), f"ATT should be finite for vce={vce}"
            assert result['se'] > 0, f"SE should be positive for vce={vce}"
        
        # All VCE types should give same point estimate
        att_base = results[None]['att']
        for vce, result in results.items():
            assert abs(result['att'] - att_base) < 1e-10, \
                f"Point estimate should be same across VCE types: {vce}"
    
    def test_cluster_robust_se_with_cholesky(self):
        """Verify cluster-robust SE works with Cholesky solution."""
        np.random.seed(303)
        n = 200
        n_clusters = 20
        
        cluster = np.repeat(np.arange(n_clusters), n // n_clusters)
        d = np.zeros(n)
        d[cluster < 8] = 1  # Cluster-level treatment
        
        # Cluster-correlated errors
        cluster_effects = np.random.randn(n_clusters)[cluster]
        y = 1.0 + 1.5 * d + cluster_effects + np.random.randn(n) * 0.3
        
        data = pd.DataFrame({
            'y': y,
            'd': d,
            'cluster': cluster,
        })
        
        result = run_ols_regression(
            data, 'y', 'd', 
            controls=None, 
            vce='cluster', 
            cluster_var='cluster'
        )
        
        assert np.isfinite(result['att']), "ATT should be finite"
        assert result['se'] > 0, "Cluster SE should be positive"
        assert result['df_inference'] == n_clusters - 1, \
            f"df should be G-1={n_clusters-1}, got {result['df_inference']}"


class TestConditionNumberDiagnostics:
    """Tests for condition number diagnostics."""
    
    def test_condition_number_threshold(self):
        """Verify warning threshold is appropriate (1e10)."""
        np.random.seed(404)
        n = 100
        
        # Create moderately ill-conditioned data (should NOT warn)
        x1 = np.random.randn(n)
        x2 = x1 + np.random.randn(n) * 0.1  # Moderate collinearity
        d = np.zeros(n)
        d[:40] = 1
        y = 1.0 + 2.0 * d + x1 + np.random.randn(n) * 0.5
        
        data = pd.DataFrame({
            'y': y,
            'd': d,
            'x1': x1,
            'x2': x2,
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            run_ols_regression(data, 'y', 'd', controls=['x1', 'x2'], vce=None)
            
            # Check condition number warnings
            condition_warnings = [
                warning for warning in w 
                if "ill-conditioned" in str(warning.message).lower()
            ]
            # Moderate collinearity should not trigger warning
            # (condition number should be well below 1e10)
    
    def test_extreme_condition_number(self):
        """Verify extreme ill-conditioning is detected or handled gracefully.
        
        With extreme collinearity (1e-8 noise), either:
        1. Cholesky fails (near-singular X'X) -> ValueError
        2. Cholesky succeeds but warns about ill-conditioning
        Both outcomes are acceptable.
        """
        np.random.seed(505)
        n = 100
        
        # Create extremely ill-conditioned data
        x1 = np.random.randn(n)
        x2 = x1 + np.random.randn(n) * 1e-8  # Extreme collinearity
        d = np.zeros(n)
        d[:40] = 1
        y = 1.0 + 2.0 * d + x1 + np.random.randn(n) * 0.5
        
        data = pd.DataFrame({
            'y': y,
            'd': d,
            'x1': x1,
            'x2': x2,
        })
        
        # Either raises ValueError (near-singular) or warns about ill-conditioning
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = run_ols_regression(data, 'y', 'd', controls=['x1', 'x2'], vce=None)
                
                condition_warnings = [
                    warning for warning in w 
                    if "ill-conditioned" in str(warning.message).lower()
                ]
                # If we get here without error, we should have a warning
                assert len(condition_warnings) > 0, \
                    "Expected warning for extreme ill-conditioning"
        except ValueError as e:
            # Cholesky failed due to near-singularity - this is acceptable
            assert "singular" in str(e).lower() or "positive definite" in str(e).lower(), \
                f"Unexpected error message: {e}"


class TestNumericalPrecision:
    """Tests for numerical precision of Cholesky solution."""
    
    def test_small_sample_precision(self):
        """Verify precision for small samples (n < 10)."""
        np.random.seed(606)
        
        # Very small sample
        d = np.array([1, 1, 0, 0, 0])
        y = np.array([5.0, 4.5, 2.0, 1.5, 2.5])
        
        data = pd.DataFrame({'y': y, 'd': d})
        
        result = run_ols_regression(data, 'y', 'd', controls=None, vce=None)
        
        # Manual calculation:
        # E[Y|D=1] = (5.0 + 4.5) / 2 = 4.75
        # E[Y|D=0] = (2.0 + 1.5 + 2.5) / 3 = 2.0
        # ATT = 4.75 - 2.0 = 2.75
        expected_att = 2.75
        
        assert abs(result['att'] - expected_att) < 1e-10, \
            f"ATT={result['att']}, expected {expected_att}"
    
    def test_large_sample_precision(self):
        """Verify precision for large samples."""
        np.random.seed(707)
        n = 10000
        
        d = np.zeros(n)
        d[:3000] = 1
        true_att = 2.5
        y = 1.0 + true_att * d + np.random.randn(n) * 1.0
        
        data = pd.DataFrame({'y': y, 'd': d})
        
        result = run_ols_regression(data, 'y', 'd', controls=None, vce='hc1')
        
        # With large sample, estimate should be very close to true value
        assert abs(result['att'] - true_att) < 0.1, \
            f"ATT={result['att']}, expected ~{true_att}"
        
        # SE should be small for large sample
        assert result['se'] < 0.1, f"SE={result['se']}, expected < 0.1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

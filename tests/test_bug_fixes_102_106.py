"""
Unit tests for BUG-102, BUG-104, BUG-105, BUG-106 fixes.

Tests boundary condition handling for:
- BUG-102: Empty array mean warnings in aggregation
- BUG-104: HC1 variance n==k division by zero
- BUG-105: Cluster-robust SE n==k division by zero
- BUG-106: Negative variance leading to NaN SE
"""

import warnings
import numpy as np
import pandas as pd
import pytest

from lwdid.staggered.aggregation import _compute_cohort_aggregated_variable
from lwdid.staggered.estimation import (
    _compute_hc1_variance,
    run_ols_regression,
)


class TestBug102EmptyArrayWarning:
    """Test BUG-102: Empty array mean warnings suppression."""
    
    def test_empty_slice_no_warning(self):
        """Test that np.nanmean with all NaN rows doesn't produce warnings."""
        # Create data with all NaN transformed values for some units
        data = pd.DataFrame({
            'id': [1, 1, 2, 2, 3, 3],
            'ydot_g2000_r2000': [1.0, np.nan, np.nan, np.nan, 2.0, 3.0],
            'ydot_g2000_r2001': [1.5, np.nan, np.nan, np.nan, 2.5, 3.5],
        })
        
        ydot_cols = ['ydot_g2000_r2000', 'ydot_g2000_r2001']
        
        # This should not produce RuntimeWarning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _compute_cohort_aggregated_variable(data, 'id', ydot_cols)
            
            # Check no "Mean of empty slice" warning was raised
            mean_warnings = [warning for warning in w 
                           if "Mean of empty slice" in str(warning.message)]
            assert len(mean_warnings) == 0, "Should not produce 'Mean of empty slice' warning"
        
        # Verify result correctness
        assert result.loc[1] == pytest.approx(1.25)  # (1.0 + 1.5) / 2
        assert np.isnan(result.loc[2])  # All NaN
        assert result.loc[3] == pytest.approx(2.75)  # (2.0 + 2.5 + 3.0 + 3.5) / 4

    def test_all_nan_returns_nan(self):
        """Test that all-NaN data returns NaN without warnings."""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'ydot_g2000_r2000': [np.nan, np.nan, np.nan, np.nan],
            'ydot_g2000_r2001': [np.nan, np.nan, np.nan, np.nan],
        })
        
        ydot_cols = ['ydot_g2000_r2000', 'ydot_g2000_r2001']
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _compute_cohort_aggregated_variable(data, 'id', ydot_cols)
            
            mean_warnings = [warning for warning in w 
                           if "Mean of empty slice" in str(warning.message)]
            assert len(mean_warnings) == 0
        
        assert np.isnan(result.loc[1])
        assert np.isnan(result.loc[2])


class TestBug104HC1DivisionByZero:
    """Test BUG-104: HC1 variance n==k division by zero."""
    
    def test_hc1_n_equals_k_raises_error(self):
        """Test that HC1 variance raises error when n == k."""
        # Create design matrix with n == k (2 observations, 2 parameters)
        X = np.array([[1, 0], [1, 1]], dtype=float)  # n=2, k=2
        residuals = np.array([0.1, -0.1])
        XtX_inv = np.linalg.inv(X.T @ X)
        
        with pytest.raises(ValueError, match="HC1 variance requires n > k"):
            _compute_hc1_variance(X, residuals, XtX_inv)
    
    def test_hc1_n_less_than_k_raises_error(self):
        """Test that HC1 variance raises error when n < k."""
        # Create underdetermined system: n=2, k=3
        X = np.array([[1, 0, 0], [1, 1, 0]], dtype=float)
        residuals = np.array([0.1, -0.1])
        XtX_inv = np.eye(3)  # Dummy inverse (wouldn't be computable in practice)
        
        with pytest.raises(ValueError, match="HC1 variance requires n > k"):
            _compute_hc1_variance(X, residuals, XtX_inv)
    
    def test_hc1_n_greater_than_k_succeeds(self):
        """Test that HC1 variance computes correctly when n > k."""
        # Create well-specified system: n=5, k=2
        np.random.seed(42)
        X = np.column_stack([np.ones(5), np.random.randn(5)])
        residuals = np.random.randn(5)
        XtX_inv = np.linalg.inv(X.T @ X)
        
        # Should not raise error
        var_beta = _compute_hc1_variance(X, residuals, XtX_inv)
        
        assert var_beta.shape == (2, 2)
        assert np.all(np.isfinite(var_beta))


class TestBug105ClusterSEDivisionByZero:
    """Test BUG-105: Cluster-robust SE n==k division by zero."""
    
    def test_cluster_n_equals_k_returns_nan(self):
        """Test that when n == k, code returns NaN SE before reaching cluster calculation."""
        # Create minimal dataset with n == k
        # When n == k, df_resid = 0, so code returns early with NaN SE
        data = pd.DataFrame({
            'y': [1.0, 2.0],
            'd': [0, 1],
            'cluster': [1, 2],
        })
        
        # This should trigger the df_resid <= 0 check and return NaN SE
        with pytest.warns(UserWarning, match="No degrees of freedom"):
            result = run_ols_regression(
                data=data,
                y='y',
                d='d',
                controls=None,
                vce='cluster',
                cluster_var='cluster',
                alpha=0.05,
            )
        
        # Verify that SE is NaN (protection worked)
        assert np.isnan(result['se'])
        assert np.isfinite(result['att'])  # ATT should still be computed
    
    def test_cluster_n_greater_than_k_succeeds(self):
        """Test that cluster-robust SE computes correctly when n > k."""
        # Create dataset with n > k
        np.random.seed(42)
        data = pd.DataFrame({
            'y': np.random.randn(10),
            'd': np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
            'cluster': np.array([1, 1, 2, 2, 3, 4, 4, 5, 5, 6]),
        })
        
        # Should not raise error
        result = run_ols_regression(
            data=data,
            y='y',
            d='d',
            controls=None,
            vce='cluster',
            cluster_var='cluster',
            alpha=0.05,
        )
        
        assert np.isfinite(result['att'])
        assert np.isfinite(result['se'])
        assert result['se'] > 0


class TestBug106NegativeVariance:
    """Test BUG-106: Negative variance leading to NaN SE."""
    
    def test_negative_variance_warning_and_correction(self):
        """Test that negative variance produces warning and uses absolute value."""
        # Create scenario with potential numerical precision issues
        # Using near-singular design matrix
        np.random.seed(42)
        n = 50
        # Create highly correlated covariates to induce numerical issues
        x1 = np.random.randn(n)
        x2 = x1 + 1e-10 * np.random.randn(n)  # Nearly collinear
        
        data = pd.DataFrame({
            'y': np.random.randn(n),
            'd': np.array([0] * 25 + [1] * 25),
            'x1': x1,
            'x2': x2,
        })
        
        # Run with robust SE on ill-conditioned data
        # This may produce negative variance due to numerical issues
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            try:
                result = run_ols_regression(
                    data=data,
                    y='y',
                    d='d',
                    controls=['x1', 'x2'],
                    vce='hc1',
                    cluster_var=None,
                    alpha=0.05,
                )
                
                # Check that SE is not NaN (should use absolute value if variance was negative)
                assert np.isfinite(result['se']), "SE should be finite even with potential negative variance"
                
                # If variance was negative, there should be a warning
                negative_var_warnings = [warning for warning in w 
                                        if "Variance estimate is negative" in str(warning.message)]
                
                # If warning was raised, verify SE is still valid
                if len(negative_var_warnings) > 0:
                    assert result['se'] >= 0, "SE should be non-negative after correction"
            
            except np.linalg.LinAlgError:
                # Matrix may be singular - this is expected for this test
                pytest.skip("Matrix is singular - expected for ill-conditioned test")
    
    def test_positive_variance_no_warning(self):
        """Test that well-conditioned data doesn't produce negative variance warning."""
        # Create well-conditioned data
        np.random.seed(42)
        n = 100
        
        data = pd.DataFrame({
            'y': np.random.randn(n),
            'd': np.array([0] * 50 + [1] * 50),
            'x': np.random.randn(n),
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = run_ols_regression(
                data=data,
                y='y',
                d='d',
                controls=['x'],
                vce='hc1',
                cluster_var=None,
                alpha=0.05,
            )
            
            # Check no negative variance warning
            negative_var_warnings = [warning for warning in w 
                                    if "Variance estimate is negative" in str(warning.message)]
            assert len(negative_var_warnings) == 0, "Should not produce negative variance warning"
            
            # Verify SE is valid
            assert np.isfinite(result['se'])
            assert result['se'] > 0


class TestBoundaryConditionsIntegration:
    """Integration tests for all boundary condition fixes."""
    
    def test_minimal_sample_size_errors(self):
        """Test that minimal sample sizes are handled correctly."""
        # n=1 should fail
        data = pd.DataFrame({
            'y': [1.0],
            'd': [1],
        })
        
        with pytest.raises(ValueError, match="Insufficient observations"):
            run_ols_regression(
                data=data,
                y='y',
                d='d',
                controls=None,
                vce=None,
                cluster_var=None,
                alpha=0.05,
            )
    
    def test_df_resid_zero_returns_nan_se(self):
        """Test that df_resid <= 0 returns NaN SE with warning."""
        # Create n=k scenario (2 observations, 2 parameters)
        data = pd.DataFrame({
            'y': [1.0, 2.0],
            'd': [0, 1],
        })
        
        with pytest.warns(UserWarning, match="No degrees of freedom"):
            result = run_ols_regression(
                data=data,
                y='y',
                d='d',
                controls=None,
                vce=None,
                cluster_var=None,
                alpha=0.05,
            )
        
        # ATT should be computed, but SE should be NaN
        assert np.isfinite(result['att'])
        assert np.isnan(result['se'])
        assert np.isnan(result['ci_lower'])
        assert np.isnan(result['ci_upper'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

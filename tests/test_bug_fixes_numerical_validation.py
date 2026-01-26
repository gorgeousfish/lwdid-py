"""
Numerical validation tests for bug fixes in Round 87-89.

This module performs numerical validation to ensure:
1. ATT estimates are consistent with manual calculations
2. Standard errors are correctly computed
3. HC4 variance calculations produce real (not complex) numbers
4. Edge cases are handled correctly
"""

import warnings
import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lwdid import lwdid
from lwdid.estimation import _compute_hc4_variance
import statsmodels.api as sm


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def known_att_data():
    """Create panel data with known ATT for validation."""
    np.random.seed(12345)
    n_units = 100
    n_periods = 6
    true_att = 2.5  # Known treatment effect
    
    data = []
    for i in range(1, n_units + 1):
        treated = 1 if i <= 50 else 0
        for t in range(2000, 2000 + n_periods):
            post = 1 if t >= 2003 else 0
            # y = base + unit_fe + time_fe + ATT*treated*post + noise
            y = 10 + (i % 10) * 0.5 + t * 0.1 + true_att * treated * post + np.random.normal(0, 0.5)
            data.append({
                'unit': i,
                'year': t,
                'treated': treated,
                'post': post,
                'outcome': y,
            })
    
    return pd.DataFrame(data), true_att


@pytest.fixture
def staggered_known_att():
    """Create staggered panel with known cohort-specific effects."""
    np.random.seed(54321)
    n_units = 90
    n_periods = 8
    
    # Known effects: cohort 2003 -> ATT=2.0, cohort 2005 -> ATT=3.0
    cohort_effects = {2003: 2.0, 2005: 3.0}
    
    data = []
    for i in range(1, n_units + 1):
        if i <= 30:
            gvar = 0  # never treated
        elif i <= 60:
            gvar = 2003
        else:
            gvar = 2005
            
        for t in range(2000, 2000 + n_periods):
            treated = 1 if gvar > 0 and t >= gvar else 0
            effect = cohort_effects.get(gvar, 0) if treated else 0
            y = 10 + (i % 5) * 0.3 + effect + np.random.normal(0, 0.3)
            data.append({
                'unit': i,
                'year': t,
                'gvar': gvar,
                'outcome': y,
            })
    
    return pd.DataFrame(data), cohort_effects


# =============================================================================
# ATT Estimate Validation
# =============================================================================

class TestATTEstimateValidation:
    """Validate ATT estimates against known values."""
    
    def test_common_timing_att_close_to_true(self, known_att_data):
        """Common timing ATT should be close to true effect."""
        df, true_att = known_att_data
        
        result = lwdid(
            data=df,
            y='outcome',
            d='treated',
            ivar='unit',
            tvar='year',
            post='post',
            rolling='demean'
        )
        
        # ATT should be within 0.5 of true value (allowing for sampling variability)
        assert abs(result.att - true_att) < 0.5, \
            f"ATT={result.att:.4f} differs too much from true ATT={true_att}"
    
    def test_staggered_cohort_effects_reasonable(self, staggered_known_att):
        """Staggered cohort effects should be close to true values."""
        df, cohort_effects = staggered_known_att
        
        result = lwdid(
            data=df,
            y='outcome',
            ivar='unit',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            aggregate='cohort'
        )
        
        # Check overall ATT is reasonable (between cohort effects)
        min_effect = min(cohort_effects.values())
        max_effect = max(cohort_effects.values())
        
        assert min_effect - 0.5 < result.att < max_effect + 0.5, \
            f"ATT={result.att:.4f} outside expected range [{min_effect-0.5}, {max_effect+0.5}]"


# =============================================================================
# Standard Error Validation
# =============================================================================

class TestStandardErrorValidation:
    """Validate standard error calculations."""
    
    def test_se_positive(self, known_att_data):
        """Standard errors should be positive."""
        df, _ = known_att_data
        
        result = lwdid(
            data=df,
            y='outcome',
            d='treated',
            ivar='unit',
            tvar='year',
            post='post',
            rolling='demean'
        )
        
        assert result.se_att > 0, "SE should be positive"
        assert not np.isnan(result.se_att), "SE should not be NaN"
        assert not np.isinf(result.se_att), "SE should not be infinite"
    
    def test_confidence_interval_contains_estimate(self, known_att_data):
        """CI should contain the point estimate."""
        df, _ = known_att_data
        
        result = lwdid(
            data=df,
            y='outcome',
            d='treated',
            ivar='unit',
            tvar='year',
            post='post',
            rolling='demean'
        )
        
        assert result.ci_lower <= result.att <= result.ci_upper, \
            f"ATT={result.att} not in CI=[{result.ci_lower}, {result.ci_upper}]"
    
    def test_ci_width_reasonable(self, known_att_data):
        """CI width should be reasonable (not too wide or narrow)."""
        df, _ = known_att_data
        
        result = lwdid(
            data=df,
            y='outcome',
            d='treated',
            ivar='unit',
            tvar='year',
            post='post',
            rolling='demean'
        )
        
        ci_width = result.ci_upper - result.ci_lower
        # For n=100 with low noise, CI width should be < 2
        assert 0 < ci_width < 2, f"CI width={ci_width:.4f} seems unreasonable"


# =============================================================================
# HC4 Variance Validation
# =============================================================================

class TestHC4VarianceValidation:
    """Validate HC4 variance calculations."""
    
    def test_hc4_produces_real_values(self):
        """HC4 variance should produce real positive values."""
        np.random.seed(42)
        n = 50
        
        # Create data
        X = np.column_stack([
            np.ones(n),
            np.random.randn(n),
            np.random.randn(n)
        ])
        beta_true = np.array([1, 2, -0.5])
        y = X @ beta_true + np.random.randn(n) * 0.5
        
        # Fit OLS to get residuals
        model = sm.OLS(y, X).fit()
        
        # Get our HC4 variance
        residuals = y - X @ model.params
        XtX_inv = np.linalg.inv(X.T @ X)
        var_beta_ours = _compute_hc4_variance(X, residuals, XtX_inv)
        
        # Check that all variances are real, positive, and finite
        var_diag = np.diag(var_beta_ours)
        assert np.isrealobj(var_diag), "HC4 variance should be real"
        assert np.all(var_diag > 0), "HC4 variances should be positive"
        assert np.all(np.isfinite(var_diag)), "HC4 variances should be finite"
    
    def test_hc4_no_nan_inf(self):
        """HC4 should not produce NaN or Inf."""
        np.random.seed(42)
        
        # Create challenging data with some collinearity
        n = 30
        x1 = np.random.randn(n)
        x2 = x1 + np.random.randn(n) * 0.1  # Nearly collinear
        X = np.column_stack([np.ones(n), x1, x2])
        residuals = np.random.randn(n)
        
        # Add regularization to make invertible
        XtX = X.T @ X
        XtX_inv = np.linalg.inv(XtX + np.eye(3) * 1e-6)
        
        var_beta = _compute_hc4_variance(X, residuals, XtX_inv)
        
        assert not np.isnan(var_beta).any(), "HC4 produced NaN"
        assert not np.isinf(var_beta).any(), "HC4 produced Inf"


# =============================================================================
# Numerical Stability Tests
# =============================================================================

class TestNumericalStability:
    """Test numerical stability of calculations."""
    
    def test_large_year_values(self):
        """Should handle large year values without overflow."""
        np.random.seed(42)
        
        # Create data with large year values
        data = []
        for i in range(1, 51):
            treated = 1 if i <= 25 else 0
            for t in range(2020, 2026):
                post = 1 if t >= 2023 else 0
                y = 10 + 2 * treated * post + np.random.normal(0, 1)
                data.append({
                    'unit': i,
                    'year': t,
                    'treated': treated,
                    'post': post,
                    'outcome': y,
                })
        
        df = pd.DataFrame(data)
        
        result = lwdid(
            data=df,
            y='outcome',
            d='treated',
            ivar='unit',
            tvar='year',
            post='post',
            rolling='demean'
        )
        
        assert not np.isnan(result.att)
        assert not np.isinf(result.att)
    
    def test_small_sample(self):
        """Should handle small samples gracefully."""
        np.random.seed(42)
        
        # Create minimal valid data
        data = []
        for i in range(1, 11):  # 10 units
            treated = 1 if i <= 5 else 0
            for t in range(2000, 2004):  # 4 periods
                post = 1 if t >= 2002 else 0
                y = 10 + 2 * treated * post + np.random.normal(0, 1)
                data.append({
                    'unit': i,
                    'year': t,
                    'treated': treated,
                    'post': post,
                    'outcome': y,
                })
        
        df = pd.DataFrame(data)
        
        result = lwdid(
            data=df,
            y='outcome',
            d='treated',
            ivar='unit',
            tvar='year',
            post='post',
            rolling='demean'
        )
        
        assert result is not None
        assert not np.isnan(result.att)


# =============================================================================
# Run tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

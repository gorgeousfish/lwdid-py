"""
Regression tests for BUG-266: p-value numerical stability fix.

This module tests the fix that changes p-value calculation from:
  2 * scipy.stats.t.cdf(-abs(t_stat), df)
to:
  2 * scipy.stats.t.sf(abs(t_stat), df)

The sf() (survival function) approach:
1. Matches Stata's ttail() function exactly
2. Provides better numerical stability for extreme t-statistics
3. Is consistent with the rest of the codebase
"""

import inspect
import numpy as np
import pandas as pd
import pytest
from scipy import stats

import sys
sys.path.insert(0, '/Users/cxy/Desktop/大样本lwdid/lwdid-py_v0.1.0/src')

from lwdid import estimation


class TestBUG266PValueMethod:
    """Test BUG-266: p-value should use sf() not cdf()."""
    
    def test_estimate_att_uses_sf(self):
        """Verify estimate_att uses scipy.stats.t.sf() for p-value calculation."""
        source = inspect.getsource(estimation.estimate_att)
        
        # Should use sf(), not cdf()
        assert 'scipy.stats.t.sf(abs(t_stat)' in source, \
            "BUG-266: estimate_att should use sf(abs(t_stat), df)"
        assert 'scipy.stats.t.cdf(-abs(t_stat)' not in source, \
            "BUG-266: estimate_att should not use cdf(-abs(t_stat), df)"
    
    def test_estimate_period_effects_uses_sf(self):
        """Verify estimate_period_effects uses scipy.stats.t.sf() for p-value."""
        source = inspect.getsource(estimation.estimate_period_effects)
        
        # Should use sf(), not cdf()
        assert 'scipy.stats.t.sf(abs(t_stat)' in source, \
            "BUG-266: estimate_period_effects should use sf(abs(t_stat), df)"
        assert 'scipy.stats.t.cdf(-abs(t_stat)' not in source, \
            "BUG-266: estimate_period_effects should not use cdf(-abs(t_stat), df)"


class TestBUG266NumericalStability:
    """Test numerical stability of sf() vs cdf() methods."""
    
    @pytest.mark.parametrize("t_stat,df", [
        (0, 10),
        (1, 10),
        (2, 30),
        (3, 100),
        (5, 100),
        (10, 100),
        (20, 100),
        (30, 100),
        (40, 100),
        (50, 100),
    ])
    def test_sf_equals_cdf_mathematically(self, t_stat, df):
        """Verify sf(|t|) = cdf(-|t|) mathematically."""
        p_sf = 2 * stats.t.sf(abs(t_stat), df)
        p_cdf = 2 * stats.t.cdf(-abs(t_stat), df)
        
        # Should be equal within machine precision
        np.testing.assert_allclose(p_sf, p_cdf, rtol=1e-14, atol=1e-300,
            err_msg=f"sf and cdf methods should give same result for t={t_stat}, df={df}")


class TestBUG266StataConsistency:
    """Test consistency with Stata's ttail() function."""
    
    # Stata reference values from: 2 * ttail(df, abs(t))
    STATA_REFERENCE = {
        (0, 10): 1.0,
        (1, 10): 3.4089313230206e-01,
        (2, 10): 7.3388034770740e-02,
        (3, 10): 1.3343655022570e-02,
        (5, 10): 5.3733360275650e-04,
        (10, 10): 1.5895531756000e-06,
        (1, 30): 3.2530861542603e-01,
        (5, 30): 2.3296685467000e-05,
        (10, 30): 4.5752514082300e-11,
        (1, 100): 3.1972415578412e-01,
        (5, 100): 2.4501734135000e-06,
        (10, 100): 9.9016889845900e-17,
    }
    
    @pytest.mark.parametrize("t_df,stata_p", list(STATA_REFERENCE.items()))
    def test_matches_stata_ttail(self, t_df, stata_p):
        """Verify Python sf() matches Stata ttail() within tolerance."""
        t_stat, df = t_df
        python_p = 2 * stats.t.sf(abs(t_stat), df)
        
        # Allow small relative difference due to precision
        if stata_p > 0:
            rel_diff = abs(python_p - stata_p) / stata_p
            assert rel_diff < 1e-10, \
                f"Python p-value differs from Stata: t={t_stat}, df={df}, " \
                f"Python={python_p:.16e}, Stata={stata_p:.16e}, RelDiff={rel_diff:.2e}"
        else:
            assert python_p == 0, f"Expected p=0 for t={t_stat}, df={df}"


class TestBUG266PValueFormula:
    """Test the p-value formula itself."""
    
    def test_pvalue_formula_two_sided(self):
        """Verify the formula p = 2 * sf(|t|, df) gives two-sided p-value."""
        # For t=2, df=100, the two-sided p-value should be ~0.048
        t_stat = 2.0
        df = 100
        
        p_value = 2 * stats.t.sf(abs(t_stat), df)
        
        # Reference: two-sided p-value from symmetry
        p_ref = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        
        np.testing.assert_allclose(p_value, p_ref, rtol=1e-14)
        
        # Should be approximately 0.048
        assert 0.04 < p_value < 0.05, f"Expected p ~ 0.048, got {p_value}"
    
    def test_pvalue_at_critical_values(self):
        """Test p-values at common critical t-values."""
        df = 100
        
        # t = 1.984 is approximately the 0.05 critical value for df=100
        t_crit_05 = stats.t.ppf(0.975, df)  # Two-sided
        p_at_crit = 2 * stats.t.sf(t_crit_05, df)
        
        # Allow slightly looser tolerance due to floating-point round-trip
        np.testing.assert_allclose(p_at_crit, 0.05, rtol=1e-9,
            err_msg="p-value at critical value should equal alpha")
    
    def test_pvalue_symmetry(self):
        """Test that p-value is same for positive and negative t-stats."""
        df = 50
        t_positive = 2.5
        t_negative = -2.5
        
        p_pos = 2 * stats.t.sf(abs(t_positive), df)
        p_neg = 2 * stats.t.sf(abs(t_negative), df)
        
        np.testing.assert_equal(p_pos, p_neg,
            err_msg="p-value should be symmetric around zero")


class TestBUG266ExtremeValues:
    """Test handling of extreme t-statistics."""
    
    def test_extreme_t_stat_pvalue_not_zero(self):
        """Test that extreme t-stats don't produce exactly zero p-values prematurely."""
        # For t=30, df=100, p-value should be ~8e-52, not 0
        t_stat = 30
        df = 100
        
        p_value = 2 * stats.t.sf(abs(t_stat), df)
        
        assert p_value > 0, \
            f"p-value for t={t_stat}, df={df} should be > 0, got {p_value}"
        assert p_value < 1e-50, \
            f"p-value for t={t_stat}, df={df} should be very small"
    
    def test_very_extreme_t_stat(self):
        """Test very extreme t-statistics (|t| > 100)."""
        # For extremely large t, p-value may underflow to 0
        # This is expected behavior, not an error
        t_stat = 200
        df = 100
        
        p_value = 2 * stats.t.sf(abs(t_stat), df)
        
        # Should be 0 or very small positive
        assert p_value >= 0, "p-value should never be negative"
        assert p_value <= 1e-100, "p-value for t=200 should be essentially 0"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

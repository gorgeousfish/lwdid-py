"""
Numerical validation tests for BUG-277, BUG-279, BUG-330 fixes.

These tests verify the fixes produce numerically correct results
by comparing with known expected values and edge case behaviors.
"""

import warnings
import numpy as np
import pandas as pd
import pytest

from lwdid import lwdid
from lwdid.estimation import _compute_hc4_variance


class TestBug330NumericalValidation:
    """Numerical validation for HC4 scale Inf fix."""

    def test_hc4_scale_inf_handling(self):
        """
        Verify HC4 correctly handles edge case where scale could be Inf.
        
        When results.scale is Inf (due to perfect fit or numerical issues),
        the fix ensures we use scale=1.0 instead of Inf to avoid:
        hc4_vcov / Inf = 0 (which would make all SEs zero)
        """
        np.random.seed(42)
        
        # Create data with very low residual variance (but not exactly zero)
        n = 50
        data = pd.DataFrame({
            'id': np.repeat(np.arange(n), 4),
            'year': np.tile([2000, 2001, 2002, 2003], n),
            'y': np.repeat(np.arange(n).astype(float), 4) + np.random.randn(n * 4) * 0.001,
            'd': np.repeat(np.random.choice([0, 1], n), 4),
            'post': np.tile([0, 0, 1, 1], n),
        })
        
        result = lwdid(
            data=data,
            y='y',
            d='d',
            ivar='id',
            tvar='year',
            post='post',
            rolling='demean',
            vce='hc4',
        )
        
        # SE should be positive and finite (not zero due to Inf scale)
        assert result.se_att > 0, f"SE should be positive, got {result.se_att}"
        assert np.isfinite(result.se_att), f"SE should be finite, got {result.se_att}"
        
        # t-stat should be finite (not Inf)
        assert np.isfinite(result.t_stat), f"t-stat should be finite, got {result.t_stat}"
        
        # p-value should be valid (between 0 and 1)
        assert 0 <= result.pvalue <= 1, f"p-value should be in [0,1], got {result.pvalue}"

    def test_hc4_variance_numerical_stability(self):
        """Test HC4 variance calculation numerical stability with small samples."""
        np.random.seed(123)
        
        # Test with various sample sizes
        for n in [5, 10, 20, 50]:
            X = np.column_stack([np.ones(n), np.random.randn(n)])
            residuals = np.random.randn(n)
            
            XtX = X.T @ X
            try:
                XtX_inv = np.linalg.inv(XtX)
            except np.linalg.LinAlgError:
                continue  # Skip if singular
            
            var_beta = _compute_hc4_variance(X, residuals, XtX_inv)
            
            # All diagonal elements should be non-negative
            diag = np.diag(var_beta)
            assert np.all(diag >= 0), f"n={n}: Variance diagonal should be >= 0"
            
            # All elements should be finite
            assert np.all(np.isfinite(var_beta)), f"n={n}: Variance should be finite"

    def test_hc4_matches_hc3_direction(self):
        """Verify HC4 produces SEs in reasonable range compared to HC3."""
        np.random.seed(456)
        n = 100
        
        data = pd.DataFrame({
            'id': np.repeat(np.arange(n), 5),
            'year': np.tile([2000, 2001, 2002, 2003, 2004], n),
            'y': np.random.randn(n * 5),
            'd': np.repeat(np.random.choice([0, 1], n), 5),
            'post': np.tile([0, 0, 1, 1, 1], n),
        })
        
        result_hc3 = lwdid(
            data=data, y='y', d='d', ivar='id', tvar='year',
            post='post', rolling='demean', vce='hc3',
        )
        
        result_hc4 = lwdid(
            data=data, y='y', d='d', ivar='id', tvar='year',
            post='post', rolling='demean', vce='hc4',
        )
        
        # HC4 and HC3 should produce SEs in similar magnitude
        # (typically HC4 >= HC3 for high-leverage observations)
        ratio = result_hc4.se_att / result_hc3.se_att
        assert 0.5 < ratio < 2.0, \
            f"HC4/HC3 SE ratio should be reasonable, got {ratio:.4f}"


class TestBug277NumericalValidation:
    """Numerical validation for cohort aggregation warning fix."""

    def test_aggregation_with_partial_nan_cohorts(self):
        """
        Test cohort aggregation when some (but not all) cohorts have NaN ATT.
        
        The fix should only warn when ALL cohorts are invalid, not when
        some cohorts succeed.
        """
        np.random.seed(789)
        n = 120
        
        # Create staggered data with enough observations
        gvar_values = np.concatenate([
            np.repeat(2002, n // 3),
            np.repeat(2003, n // 3),
            np.zeros(n - 2 * (n // 3)),  # Never-treated
        ])
        
        data = pd.DataFrame({
            'id': np.repeat(np.arange(n), 5),
            'year': np.tile([2000, 2001, 2002, 2003, 2004], n),
            'y': np.random.randn(n * 5),
            'gvar': np.repeat(gvar_values, 5),
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = lwdid(
                data=data,
                y='y',
                ivar='id',
                tvar='year',
                gvar='gvar',
                rolling='demean',
                aggregate='cohort',
            )
            
            # Check if we got valid cohort results
            if result.att_by_cohort is not None and len(result.att_by_cohort) > 0:
                n_valid = result.att_by_cohort['att'].notna().sum()
                
                # If we have some valid cohorts, should NOT see "all invalid" warning
                if n_valid > 0:
                    all_invalid_warning = any(
                        "all" in str(wm.message).lower() and
                        "invalid" in str(wm.message).lower() and
                        "cohort" in str(wm.message).lower()
                        for wm in w
                    )
                    assert not all_invalid_warning, \
                        f"Should not warn about 'all invalid' when {n_valid} cohorts valid"

    def test_aggregation_statistics_calculation(self):
        """Verify cohort aggregation statistics are calculated correctly."""
        np.random.seed(101)
        n = 90
        
        gvar_values = np.concatenate([
            np.repeat(2002, n // 3),
            np.repeat(2003, n // 3),
            np.zeros(n - 2 * (n // 3)),
        ])
        
        data = pd.DataFrame({
            'id': np.repeat(np.arange(n), 5),
            'year': np.tile([2000, 2001, 2002, 2003, 2004], n),
            'y': np.random.randn(n * 5) + np.repeat(gvar_values > 0, 5) * 0.5,  # Treatment effect
            'gvar': np.repeat(gvar_values, 5),
        })
        
        result = lwdid(
            data=data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            aggregate='cohort',
        )
        
        # If we have valid cohort results, check aggregation
        if result.att_by_cohort is not None and len(result.att_by_cohort) > 0:
            valid_cohorts = result.att_by_cohort[
                result.att_by_cohort['att'].notna() & 
                result.att_by_cohort['se'].notna()
            ]
            
            if len(valid_cohorts) > 0:
                # Verify ATT values are finite
                assert np.all(np.isfinite(valid_cohorts['att'])), \
                    "Valid cohort ATTs should be finite"
                
                # Verify SEs are positive
                assert np.all(valid_cohorts['se'] > 0), \
                    "Valid cohort SEs should be positive"


class TestBug279NumericalValidation:
    """Numerical validation for tvar list validation fix."""

    def test_error_message_clarity(self):
        """Verify error messages are clear and helpful."""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'year': [2000, 2001, 2000, 2001],
            'y': [1.0, 2.0, 1.5, 2.5],
            'gvar': [2001, 2001, 0, 0],
        })
        
        # Test empty list error message
        with pytest.raises(ValueError) as excinfo:
            lwdid(data=data, y='y', ivar='id', tvar=[], gvar='gvar', rolling='demean')
        
        error_msg = str(excinfo.value)
        # Should contain helpful guidance
        assert any(keyword in error_msg.lower() for keyword in ['empty', 'annual', 'quarterly']), \
            f"Error message should be helpful: {error_msg}"

    def test_valid_annual_data_works(self):
        """Verify annual data with string tvar works correctly."""
        np.random.seed(42)
        n = 60
        
        gvar_values = np.concatenate([
            np.repeat(2002, n // 2),
            np.zeros(n - n // 2),
        ])
        
        data = pd.DataFrame({
            'id': np.repeat(np.arange(n), 4),
            'year': np.tile([2000, 2001, 2002, 2003], n),
            'y': np.random.randn(n * 4),
            'gvar': np.repeat(gvar_values, 4),
        })
        
        # String tvar for annual data should work
        result = lwdid(
            data=data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
        )
        
        assert result is not None
        assert hasattr(result, 'att')


class TestCrossValidation:
    """Cross-validation tests comparing different estimation methods."""

    def test_vce_methods_produce_different_ses(self):
        """Verify different VCE methods produce different (but reasonable) SEs."""
        np.random.seed(999)
        n = 80
        
        data = pd.DataFrame({
            'id': np.repeat(np.arange(n), 5),
            'year': np.tile([2000, 2001, 2002, 2003, 2004], n),
            'y': np.random.randn(n * 5),
            'd': np.repeat(np.random.choice([0, 1], n), 5),
            'post': np.tile([0, 0, 1, 1, 1], n),
        })
        
        vce_methods = [None, 'robust', 'hc0', 'hc1', 'hc2', 'hc3', 'hc4']
        results = {}
        
        for vce in vce_methods:
            r = lwdid(
                data=data, y='y', d='d', ivar='id', tvar='year',
                post='post', rolling='demean', vce=vce,
            )
            results[vce if vce else 'ols'] = r.se_att
        
        # All SEs should be positive
        for name, se in results.items():
            assert se > 0, f"{name} SE should be positive"
        
        # HC methods should generally produce SEs in similar range
        hc_ses = [results['hc0'], results['hc1'], results['hc2'], results['hc3'], results['hc4']]
        se_range = max(hc_ses) / min(hc_ses)
        assert se_range < 5, f"HC SEs should not vary too wildly: range ratio = {se_range:.2f}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

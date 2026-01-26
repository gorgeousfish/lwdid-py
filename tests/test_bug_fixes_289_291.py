"""
Regression tests for bug fixes BUG-289, BUG-290, BUG-291 (Code Review Round 86).

This module tests the fixes for:
- BUG-289 (P2): build_subsample_for_ps_estimation uses tolerance-based cohort comparison
- BUG-290 (P2): validate_quarter_diversity handles NaN post values correctly
- BUG-291 (P2): validate_staggered_data raises error for mixed NaN/valid gvar values
"""

import warnings
import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, '/Users/cxy/Desktop/大样本lwdid/lwdid-py_v0.1.0/src')

from lwdid.validation import (
    validate_quarter_diversity,
    validate_quarter_coverage,
    validate_staggered_data,
    COHORT_FLOAT_TOLERANCE
)
from lwdid.exceptions import (
    InvalidStaggeredDataError,
    InsufficientQuarterDiversityError
)
from lwdid.staggered.estimators import build_subsample_for_ps_estimation


class TestBUG289ToleranceBasedCohortComparison:
    """Test BUG-289: build_subsample_for_ps_estimation should use tolerance comparison."""
    
    def test_code_uses_tolerance_comparison(self):
        """Verify that the code uses tolerance-based comparison, not exact equality."""
        from lwdid.staggered import estimators
        import inspect
        source = inspect.getsource(estimators.build_subsample_for_ps_estimation)
        # The fix uses np.abs(...) <= COHORT_FLOAT_TOLERANCE
        assert 'COHORT_FLOAT_TOLERANCE' in source, \
            "BUG-289: Should use COHORT_FLOAT_TOLERANCE for cohort comparison"
        assert 'np.abs' in source, \
            "BUG-289: Should use np.abs for tolerance comparison"
    
    def test_float_cohort_treated_identification(self):
        """Test that floating-point cohort values are correctly identified."""
        # Create test data with floating-point cohort value
        data = pd.DataFrame({
            'id': [1, 1, 2, 2, 3, 3],
            'time': [2004, 2005, 2004, 2005, 2004, 2005],
            'gvar': [2005.0 + 1e-10, 2005.0 + 1e-10, 0.0, 0.0, 2006.0, 2006.0],  # Unit 1 has floating point
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        })
        
        # Test that build_subsample_for_ps_estimation correctly identifies treated units
        result = build_subsample_for_ps_estimation(
            data=data,
            gvar_col='gvar',
            ivar_col='id',
            cohort_g=2005.0,  # Exact integer value
            period_r=2005,
            control_group='not_yet_treated'
        )
        
        # Unit 1 (gvar=2005.0+1e-10) should be identified as treated
        # n_treated counts rows in subsample, not unique units (unit 1 has 2 rows)
        assert result.n_treated == 2, \
            f"BUG-289: Unit with gvar=2005.0+1e-10 should be treated for cohort_g=2005.0, got {result.n_treated}"
        
        # Verify only unit 1 is in treated group by checking unique ids
        treated_ids = result.subsample[result.subsample['gvar'].apply(
            lambda x: np.abs(x - 2005.0) <= COHORT_FLOAT_TOLERANCE
        )]['id'].unique()
        assert len(treated_ids) == 1 and treated_ids[0] == 1, \
            f"BUG-289: Only unit 1 should be treated, got {treated_ids}"
    
    def test_consistency_with_create_binary_treatment(self):
        """Ensure build_subsample uses same tolerance as _create_binary_treatment."""
        from lwdid.staggered.estimators import _create_binary_treatment
        
        # Create test data
        data = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'time': [2004, 2005, 2004, 2005],
            'gvar': [2005.0 + COHORT_FLOAT_TOLERANCE * 0.5, 2005.0 + COHORT_FLOAT_TOLERANCE * 0.5, 
                     0.0, 0.0],
            'y': [1.0, 2.0, 3.0, 4.0]
        })
        
        # Both functions should identify unit 1 as treated
        result = build_subsample_for_ps_estimation(
            data=data,
            gvar_col='gvar',
            ivar_col='id',
            cohort_g=2005.0,
            period_r=2005,
            control_group='never_treated'
        )
        
        D_ig = _create_binary_treatment(result.subsample, 'gvar', 2005.0)
        
        # The subsample should have treated units identified
        assert result.n_treated > 0, \
            "BUG-289: Tolerance comparison should identify treated units"
        assert D_ig.sum() > 0, \
            "BUG-289: _create_binary_treatment should find treated units"


class TestBUG290PostNaNHandling:
    """Test BUG-290: validate_quarter_diversity should handle NaN post values."""
    
    def test_code_excludes_nan_from_post_binary(self):
        """Verify that the code properly handles NaN post values."""
        from lwdid import validation
        import inspect
        source = inspect.getsource(validation.validate_quarter_diversity)
        # The fix uses notna() to exclude NaN
        assert 'notna()' in source, \
            "BUG-290: Should use notna() to exclude NaN from post binarization"
    
    def test_nan_post_not_treated_as_post_period(self):
        """Test that NaN post values are not incorrectly treated as post period."""
        # Create data where post column has NaN
        data = pd.DataFrame({
            'id': [1, 1, 1, 1, 1, 1, 1, 1],
            'quarter': [1, 2, 3, 4, 1, 2, 3, 4],
            'post': [0, 0, np.nan, 0, 1, 1, 1, 1],  # NaN should be treated as pre-period
            'y': range(8)
        })
        
        # Should not raise error because we have quarters 1,2,4 in pre-period
        # and quarters 1,2,3,4 in post-period (NaN excluded from post)
        # After fix: NaN post is treated as pre-period (0)
        try:
            validate_quarter_diversity(data, 'id', 'quarter', 'post')
            # If we get here, the fix is working - NaN is treated as pre-period
        except InsufficientQuarterDiversityError as e:
            # This should not happen after the fix
            pytest.fail(f"BUG-290: NaN post value should not be treated as post period: {e}")
    
    def test_validate_quarter_coverage_nan_handling(self):
        """Test that validate_quarter_coverage also handles NaN post."""
        from lwdid import validation
        import inspect
        source = inspect.getsource(validation.validate_quarter_coverage)
        # The fix uses notna() to exclude NaN
        assert 'notna()' in source, \
            "BUG-290: validate_quarter_coverage should also use notna()"
    
    def test_nan_comparison_behavior(self):
        """Verify that np.nan != 0 returns True (the bug source)."""
        # This test documents the bug source
        assert np.nan != 0, "np.nan != 0 should return True (Python float semantics)"
        # The fix ensures NaN is explicitly excluded


class TestBUG291GvarMixedValuesError:
    """Test BUG-291: validate_staggered_data should raise error for mixed gvar values."""
    
    def test_code_raises_error_not_warning(self):
        """Verify that the code raises InvalidStaggeredDataError, not warning."""
        from lwdid import validation
        import inspect
        source = inspect.getsource(validation.validate_staggered_data)
        # The fix changes warnings.warn to raise InvalidStaggeredDataError
        # Check that the mixed units section raises error
        assert 'raise InvalidStaggeredDataError' in source, \
            "BUG-291: Should raise InvalidStaggeredDataError for mixed gvar values"
    
    def test_mixed_nan_valid_gvar_raises_error(self):
        """Test that mixed NaN and valid gvar values raise error."""
        # Create data where unit 1 has mixed NaN and valid gvar values
        data = pd.DataFrame({
            'id': [1, 1, 1, 2, 2, 2],
            'time': [2003, 2004, 2005, 2003, 2004, 2005],
            'gvar': [np.nan, 2005, 2005, 2006, 2006, 2006],  # Unit 1 has mixed NaN/valid
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        })
        
        # Should raise InvalidStaggeredDataError
        with pytest.raises(InvalidStaggeredDataError) as excinfo:
            validate_staggered_data(data, 'gvar', 'id', 'time', 'y')
        
        assert 'mixed NaN and valid gvar values' in str(excinfo.value), \
            "BUG-291: Error message should mention mixed NaN and valid values"
    
    def test_all_nan_gvar_allowed(self):
        """Test that units with all-NaN gvar (never-treated) are allowed."""
        # Create data where unit 2 has all NaN gvar (never-treated)
        data = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'time': [2004, 2005, 2004, 2005],
            'gvar': [2005, 2005, np.nan, np.nan],  # Unit 2 is never-treated
            'y': [1.0, 2.0, 3.0, 4.0]
        })
        
        # Should NOT raise error - all-NaN is valid (never-treated)
        try:
            validate_staggered_data(data, 'gvar', 'id', 'time', 'y')
        except InvalidStaggeredDataError as e:
            if 'mixed NaN' in str(e):
                pytest.fail("BUG-291: All-NaN gvar should be allowed (never-treated)")
            # Other errors are OK (e.g., insufficient cohorts)
    
    def test_all_valid_gvar_allowed(self):
        """Test that units with all-valid gvar values are allowed."""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2, 3, 3],
            'time': [2004, 2005, 2004, 2005, 2004, 2005],
            'gvar': [2005, 2005, 2006, 2006, 0, 0],  # All units have consistent gvar
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        })
        
        # Should NOT raise error for mixed values
        try:
            validate_staggered_data(data, 'gvar', 'id', 'time', 'y')
        except InvalidStaggeredDataError as e:
            if 'mixed NaN' in str(e):
                pytest.fail("BUG-291: All-valid gvar should not trigger mixed values error")
    
    def test_error_message_provides_fix_guidance(self):
        """Test that error message provides helpful fix suggestions."""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'time': [2004, 2005, 2004, 2005],
            'gvar': [np.nan, 2005, 2006, 2006],  # Unit 1 has mixed
            'y': [1.0, 2.0, 3.0, 4.0]
        })
        
        with pytest.raises(InvalidStaggeredDataError) as excinfo:
            validate_staggered_data(data, 'gvar', 'id', 'time', 'y')
        
        error_msg = str(excinfo.value)
        assert 'How to fix' in error_msg, \
            "BUG-291: Error should provide fix guidance"
        assert 'fillna' in error_msg or 'dropna' in error_msg, \
            "BUG-291: Error should suggest fillna or dropna"


class TestIntegrationBUG289290291:
    """Integration tests for all three bug fixes together."""
    
    def test_staggered_estimation_with_float_cohorts(self):
        """Test that staggered estimation works with floating-point cohorts."""
        from lwdid import lwdid
        
        # Create larger test data with floating-point precision issues
        np.random.seed(42)
        n_units = 50
        n_periods = 6
        
        data = []
        for i in range(n_units):
            # Some units treated in 2004, some in 2005, some never
            if i < 15:
                gvar = 2004.0 + np.random.uniform(-1e-10, 1e-10)  # Floating point noise
            elif i < 30:
                gvar = 2005.0 + np.random.uniform(-1e-10, 1e-10)
            else:
                gvar = 0.0  # Never treated
            
            for t in range(2002, 2002 + n_periods):
                post = 1 if (gvar > 0 and t >= gvar) else 0
                y = np.random.normal(0, 1) + post * 2.0  # Treatment effect of 2
                data.append({'id': i, 'time': t, 'gvar': gvar, 'y': y})
        
        df = pd.DataFrame(data)
        
        # Should work without errors
        try:
            result = lwdid(df, y='y', ivar='id', tvar='time', gvar='gvar',
                          estimator='ra', control_group='not_yet_treated')
            assert result is not None, "BUG-289: Should produce result with float cohorts"
        except Exception as e:
            pytest.fail(f"BUG-289: Float cohort estimation failed: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

"""
Test suite for DESIGN-043-L: Large integer gvar range validation.

This module tests the validation of gvar (first treatment period) values to ensure
that unreasonably large values (e.g., 1e20) trigger appropriate warnings, as such
values are likely data entry errors rather than valid treatment periods.
"""

import numpy as np
import pandas as pd
import pytest
import warnings

from lwdid.validation import (
    validate_staggered_data,
    MAX_REASONABLE_YEAR,
    safe_int_cohort,
)


class TestMaxReasonableYearConstant:
    """Tests for the MAX_REASONABLE_YEAR constant definition."""
    
    def test_constant_exists_and_is_reasonable(self):
        """MAX_REASONABLE_YEAR should be defined and have a reasonable value."""
        assert MAX_REASONABLE_YEAR is not None
        assert isinstance(MAX_REASONABLE_YEAR, int)
        assert MAX_REASONABLE_YEAR >= 2100  # At least beyond current century
        assert MAX_REASONABLE_YEAR <= 10000  # Not excessively high
    
    def test_constant_value(self):
        """MAX_REASONABLE_YEAR should be 3000 as specified in design."""
        assert MAX_REASONABLE_YEAR == 3000


class TestNormalGvarValues:
    """Tests verifying that normal gvar values do not trigger warnings."""
    
    @pytest.fixture
    def normal_data(self):
        """Create panel data with normal cohort values."""
        return pd.DataFrame({
            'id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
            'year': [2003, 2004, 2005] * 3,
            'gvar': [2004]*3 + [2005]*3 + [0]*3,
            'y': np.random.randn(9)
        })
    
    def test_normal_cohorts_no_warning(self, normal_data):
        """Normal year values should not trigger gvar range warnings."""
        result = validate_staggered_data(
            normal_data, gvar='gvar', ivar='id', tvar='year', y='y'
        )
        range_warnings = [w for w in result['warnings'] if 'exceed reasonable year range' in w]
        assert len(range_warnings) == 0
        assert result['cohorts'] == [2004, 2005]
    
    def test_boundary_value_no_warning(self):
        """gvar = MAX_REASONABLE_YEAR should not trigger warning."""
        data = pd.DataFrame({
            'id': [1, 1, 1, 2, 2, 2],
            'year': [2999, 3000, 3001] * 2,
            'gvar': [3000]*3 + [0]*3,
            'y': np.random.randn(6)
        })
        result = validate_staggered_data(
            data, gvar='gvar', ivar='id', tvar='year', y='y'
        )
        range_warnings = [w for w in result['warnings'] if 'exceed reasonable year range' in w]
        assert len(range_warnings) == 0


class TestLargeGvarWarnings:
    """Tests verifying that unreasonably large gvar values trigger warnings."""
    
    def test_extremely_large_gvar_warning(self):
        """gvar = 1e20 should trigger a warning about unreasonable year range."""
        data = pd.DataFrame({
            'id': [1, 1, 1, 2, 2, 2],
            'year': [2003, 2004, 2005] * 2,
            'gvar': [1e20]*3 + [0]*3,
            'y': np.random.randn(6)
        })
        result = validate_staggered_data(
            data, gvar='gvar', ivar='id', tvar='year', y='y'
        )
        range_warnings = [w for w in result['warnings'] if 'exceed reasonable year range' in w]
        
        assert len(range_warnings) == 1
        assert '>' + str(MAX_REASONABLE_YEAR) in range_warnings[0]
        assert 'data entry errors' in range_warnings[0]
    
    def test_slightly_over_limit_warning(self):
        """gvar = MAX_REASONABLE_YEAR + 1 should trigger warning."""
        over_limit = MAX_REASONABLE_YEAR + 1
        data = pd.DataFrame({
            'id': [1, 1, 1, 2, 2, 2],
            'year': [3000, 3001, 3002] * 2,
            'gvar': [over_limit]*3 + [0]*3,
            'y': np.random.randn(6)
        })
        result = validate_staggered_data(
            data, gvar='gvar', ivar='id', tvar='year', y='y'
        )
        range_warnings = [w for w in result['warnings'] if 'exceed reasonable year range' in w]
        
        assert len(range_warnings) == 1
        assert str(over_limit) in range_warnings[0]
    
    def test_multiple_large_gvar_values(self):
        """Multiple large gvar values should be reported in a single warning."""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2, 3, 3, 4, 4],
            'year': [2003, 2004] * 4,
            'gvar': [5000]*2 + [6000]*2 + [7000]*2 + [0]*2,
            'y': np.random.randn(8)
        })
        result = validate_staggered_data(
            data, gvar='gvar', ivar='id', tvar='year', y='y'
        )
        range_warnings = [w for w in result['warnings'] if 'exceed reasonable year range' in w]
        
        assert len(range_warnings) == 1
        # Should report sample of values
        assert '5000' in range_warnings[0] or '6000' in range_warnings[0] or '7000' in range_warnings[0]
        # Should report affected unit count
        assert '3 unit(s)' in range_warnings[0]
    
    def test_warning_includes_helpful_message(self):
        """Warning message should include helpful context."""
        data = pd.DataFrame({
            'id': [1, 1, 1, 2, 2, 2],
            'year': [2003, 2004, 2005] * 2,
            'gvar': [10000]*3 + [0]*3,
            'y': np.random.randn(6)
        })
        result = validate_staggered_data(
            data, gvar='gvar', ivar='id', tvar='year', y='y'
        )
        range_warnings = [w for w in result['warnings'] if 'exceed reasonable year range' in w]
        
        assert len(range_warnings) == 1
        warning_msg = range_warnings[0]
        
        # Should mention data entry errors
        assert 'data entry errors' in warning_msg.lower() or 'data entry' in warning_msg
        # Should mention that warning can be ignored for non-calendar indices
        assert 'non-calendar' in warning_msg or 'quarterly' in warning_msg or 'ignored' in warning_msg


class TestGvarValidationStillWorks:
    """Tests ensuring that gvar range check doesn't break other validation."""
    
    def test_cohort_extraction_still_works(self):
        """Cohorts should still be correctly extracted despite warning."""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2, 3, 3],
            'year': [2003, 2004] * 3,
            'gvar': [5000]*2 + [6000]*2 + [0]*2,
            'y': np.random.randn(6)
        })
        result = validate_staggered_data(
            data, gvar='gvar', ivar='id', tvar='year', y='y'
        )
        
        assert result['cohorts'] == [5000, 6000]
        assert result['n_cohorts'] == 2
        assert result['cohort_sizes'] == {5000: 1, 6000: 1}
    
    def test_other_warnings_still_present(self):
        """Other relevant warnings should still be generated."""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'year': [2003, 2004] * 2,
            'gvar': [10000]*2 + [0]*2,  # Large but only 1 NT unit
            'y': np.random.randn(4)
        })
        result = validate_staggered_data(
            data, gvar='gvar', ivar='id', tvar='year', y='y'
        )
        
        # Should have range warning
        range_warnings = [w for w in result['warnings'] if 'exceed reasonable year range' in w]
        assert len(range_warnings) == 1
        
        # Should also have warning about few NT units
        nt_warnings = [w for w in result['warnings'] if 'never-treated' in w.lower()]
        assert len(nt_warnings) >= 1


class TestSafeIntCohortWithLargeValues:
    """Tests for safe_int_cohort function with large values."""
    
    def test_safe_int_cohort_handles_large_values(self):
        """safe_int_cohort should handle large but valid integer-like values."""
        # This should succeed (not raise)
        result = safe_int_cohort(1e10)
        assert result == 10000000000
        
        result = safe_int_cohort(1e20)
        assert result == 100000000000000000000
    
    def test_safe_int_cohort_rejects_non_integers(self):
        """safe_int_cohort should reject non-integer values."""
        with pytest.raises(ValueError, match="not close to an integer"):
            safe_int_cohort(2000.5)
        
        with pytest.raises(ValueError, match="not close to an integer"):
            safe_int_cohort(2000.1234)
        
        # Note: Very large floats like 1e20 + 0.1 will pass due to floating point
        # precision limitations (1e20 + 0.1 == 1e20 in float representation).
        # This is expected and acceptable behavior.


class TestEdgeCases:
    """Edge case tests for gvar range validation."""
    
    def test_mixed_normal_and_large_gvar(self):
        """Mix of normal and large gvar values should only warn about large ones."""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2, 3, 3, 4, 4],
            'year': [2003, 2004] * 4,
            'gvar': [2004]*2 + [2005]*2 + [5000]*2 + [0]*2,
            'y': np.random.randn(8)
        })
        result = validate_staggered_data(
            data, gvar='gvar', ivar='id', tvar='year', y='y'
        )
        
        # Only 5000 should trigger warning, not 2004 or 2005
        range_warnings = [w for w in result['warnings'] if 'exceed reasonable year range' in w]
        assert len(range_warnings) == 1
        assert '5000' in range_warnings[0]
        assert '2004' not in range_warnings[0]
        assert '2005' not in range_warnings[0]
    
    def test_inf_gvar_not_treated_as_large(self):
        """np.inf gvar (never-treated) should not trigger large value warning."""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'year': [2003, 2004] * 2,
            'gvar': [2004]*2 + [np.inf]*2,  # inf = never treated
            'y': np.random.randn(4)
        })
        result = validate_staggered_data(
            data, gvar='gvar', ivar='id', tvar='year', y='y'
        )
        
        range_warnings = [w for w in result['warnings'] if 'exceed reasonable year range' in w]
        assert len(range_warnings) == 0
    
    def test_nan_gvar_not_treated_as_large(self):
        """NaN gvar (never-treated) should not trigger large value warning."""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'year': [2003, 2004] * 2,
            'gvar': [2004]*2 + [np.nan]*2,  # NaN = never treated
            'y': np.random.randn(4)
        })
        result = validate_staggered_data(
            data, gvar='gvar', ivar='id', tvar='year', y='y'
        )
        
        range_warnings = [w for w in result['warnings'] if 'exceed reasonable year range' in w]
        assert len(range_warnings) == 0


class TestRealWorldScenarios:
    """Tests simulating real-world data entry error scenarios."""
    
    def test_typo_1e20_instead_of_2020(self):
        """Common typo: 1e20 instead of 2020."""
        data = pd.DataFrame({
            'id': [1, 1, 1, 2, 2, 2],
            'year': [2018, 2019, 2020] * 2,
            'gvar': [1e20]*3 + [0]*3,  # Typo: should be 2020
            'y': np.random.randn(6)
        })
        result = validate_staggered_data(
            data, gvar='gvar', ivar='id', tvar='year', y='y'
        )
        
        range_warnings = [w for w in result['warnings'] if 'exceed reasonable year range' in w]
        assert len(range_warnings) == 1
        assert '1e20 instead of 2020' in range_warnings[0] or 'data entry errors' in range_warnings[0]
    
    def test_quarterly_tq_encoding_may_exceed_limit(self):
        """
        Quarterly tq encoding (year - 1960) * 4 + (quarter - 1) can produce large values.
        For year 3000, tq = (3000 - 1960) * 4 + 0 = 4160, which is > 3000.
        This is expected and the warning should mention it can be ignored.
        """
        # Simulate a dataset where gvar uses quarterly tq encoding
        # tq for year 2500 Q1 = (2500 - 1960) * 4 + 0 = 2160
        # tq for year 3100 Q1 = (3100 - 1960) * 4 + 0 = 4560
        tq_3100 = (3100 - 1960) * 4  # = 4560 > MAX_REASONABLE_YEAR
        
        data = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'year': [3099, 3100] * 2,
            'gvar': [tq_3100]*2 + [0]*2,
            'y': np.random.randn(4)
        })
        result = validate_staggered_data(
            data, gvar='gvar', ivar='id', tvar='year', y='y'
        )
        
        range_warnings = [w for w in result['warnings'] if 'exceed reasonable year range' in w]
        assert len(range_warnings) == 1
        # Warning should mention that non-calendar indices can be ignored
        assert 'non-calendar' in range_warnings[0].lower() or 'ignored' in range_warnings[0]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

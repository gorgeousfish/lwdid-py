"""Tests for DESIGN-043: Non-integer gvar value validation.

This module tests that validate_staggered_data() properly validates gvar values
are integers and rejects non-integer values with clear error messages.

DESIGN-043 Fix: Use safe_int_cohort() to validate gvar values instead of
direct int() conversion, preventing silent truncation of non-integer values.
"""

import numpy as np
import pandas as pd
import pytest

from lwdid.validation import (
    validate_staggered_data,
    safe_int_cohort,
    is_never_treated,
    get_cohort_mask,
)
from lwdid.exceptions import InvalidStaggeredDataError


class TestSafeIntCohort:
    """Tests for the safe_int_cohort() helper function."""
    
    def test_integer_values(self):
        """Integer values should pass through without error."""
        assert safe_int_cohort(2005) == 2005
        assert safe_int_cohort(2010) == 2010
        assert safe_int_cohort(1) == 1
    
    def test_float_integer_values(self):
        """Float values that are exactly integers should pass."""
        assert safe_int_cohort(2005.0) == 2005
        assert safe_int_cohort(2010.0) == 2010
        assert safe_int_cohort(1.0) == 1
    
    def test_near_integer_values_within_tolerance(self):
        """Float values very close to integers (within 1e-9) should pass.
        
        DESIGN-045: Tolerance was changed from 1e-6 to 1e-9 to be consistent
        with get_cohort_mask() and other cohort comparison functions.
        """
        # These are within tolerance (1e-9 = COHORT_FLOAT_TOLERANCE)
        assert safe_int_cohort(2005.0000000001) == 2005  # 1e-10 away
        assert safe_int_cohort(2004.9999999999) == 2005  # 1e-10 away
        assert safe_int_cohort(2010.0000000005) == 2010  # 5e-10 away
    
    def test_non_integer_values_rejected(self):
        """Non-integer values should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            safe_int_cohort(2005.5)
        
        error_msg = str(exc_info.value)
        assert "2005.5" in error_msg
        assert "not close to an integer" in error_msg
        assert "Why this matters:" in error_msg
        assert "How to fix:" in error_msg
    
    def test_clearly_non_integer_values(self):
        """Various clearly non-integer values should be rejected."""
        non_integer_values = [2005.5, 2004.9, 2010.1, 2000.25, 1999.75]
        
        for val in non_integer_values:
            with pytest.raises(ValueError) as exc_info:
                safe_int_cohort(val)
            assert str(val) in str(exc_info.value)
    
    def test_numpy_integer_types(self):
        """NumPy integer types should pass."""
        assert safe_int_cohort(np.int32(2005)) == 2005
        assert safe_int_cohort(np.int64(2010)) == 2010
    
    def test_numpy_float_integer_values(self):
        """NumPy float values that are integers should pass."""
        assert safe_int_cohort(np.float64(2005.0)) == 2005


class TestValidateStaggeredDataNonIntegerGvar:
    """Tests for validate_staggered_data() with non-integer gvar values."""
    
    @pytest.fixture
    def base_data(self):
        """Create base test data."""
        return pd.DataFrame({
            'id': [1, 1, 2, 2, 3, 3, 4, 4],
            'year': [2000, 2001, 2000, 2001, 2000, 2001, 2000, 2001],
            'y': [1.0, 2.0, 1.5, 2.5, 1.2, 2.2, 1.3, 2.3],
            'gvar': [2001, 2001, 0, 0, 0, 0, 0, 0]  # One cohort, three NT
        })
    
    def test_integer_gvar_accepted(self, base_data):
        """Integer gvar values should be accepted."""
        result = validate_staggered_data(base_data, 'gvar', 'id', 'year', 'y')
        
        assert result['cohorts'] == [2001]
        assert result['n_treated'] == 1
        assert result['n_never_treated'] == 3
    
    def test_float_integer_gvar_accepted(self, base_data):
        """Float gvar values that are exactly integers should be accepted."""
        data = base_data.copy()
        data['gvar'] = data['gvar'].astype(float)  # 2001.0, 0.0
        
        result = validate_staggered_data(data, 'gvar', 'id', 'year', 'y')
        
        assert result['cohorts'] == [2001]
        assert result['n_treated'] == 1
    
    def test_non_integer_gvar_rejected(self, base_data):
        """Non-integer gvar values should be rejected with InvalidStaggeredDataError."""
        data = base_data.copy()
        # Set a non-integer gvar value
        data.loc[data['id'] == 1, 'gvar'] = 2001.5
        
        with pytest.raises(InvalidStaggeredDataError) as exc_info:
            validate_staggered_data(data, 'gvar', 'id', 'year', 'y')
        
        error_msg = str(exc_info.value)
        assert "2001.5" in error_msg
        assert "not close to an integer" in error_msg
    
    def test_fractional_gvar_rejected(self, base_data):
        """Fractional gvar values (like 2005.9) should be rejected."""
        data = base_data.copy()
        data.loc[data['id'] == 1, 'gvar'] = 2000.9
        
        with pytest.raises(InvalidStaggeredDataError) as exc_info:
            validate_staggered_data(data, 'gvar', 'id', 'year', 'y')
        
        error_msg = str(exc_info.value)
        assert "2000.9" in error_msg
    
    def test_multiple_non_integer_gvar_reports_first(self, base_data):
        """Multiple non-integer gvar values - should report first encountered."""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2, 3, 3],
            'year': [2000, 2001, 2000, 2001, 2000, 2001],
            'y': [1.0, 2.0, 1.5, 2.5, 1.2, 2.2],
            'gvar': [2001.5, 2001.5, 2002.7, 2002.7, 0, 0]
        })
        
        with pytest.raises(InvalidStaggeredDataError):
            validate_staggered_data(data, 'gvar', 'id', 'year', 'y')
    
    def test_near_integer_gvar_within_tolerance_accepted(self, base_data):
        """Gvar values very close to integers should be accepted.
        
        DESIGN-045: Tolerance was changed from 1e-6 to 1e-9 to be consistent
        with get_cohort_mask() and other cohort comparison functions.
        """
        data = base_data.copy()
        # Set gvar to be very close to integer (within 1e-9 tolerance)
        # Using 1e-10 away from 2001
        data.loc[data['id'] == 1, 'gvar'] = 2001.0000000001
        
        result = validate_staggered_data(data, 'gvar', 'id', 'year', 'y')
        
        # Should be rounded to 2001
        assert 2001 in result['cohorts']
    
    def test_error_message_has_fix_instructions(self, base_data):
        """Error message should contain fix instructions."""
        data = base_data.copy()
        data.loc[data['id'] == 1, 'gvar'] = 2001.5
        
        with pytest.raises(InvalidStaggeredDataError) as exc_info:
            validate_staggered_data(data, 'gvar', 'id', 'year', 'y')
        
        error_msg = str(exc_info.value)
        # Should contain guidance on how to fix
        assert "Why this matters:" in error_msg
        assert "How to fix:" in error_msg
        assert "round" in error_msg.lower() or "astype" in error_msg.lower()


class TestNeverTreatedValues:
    """Tests for never-treated value handling."""
    
    def test_zero_is_never_treated(self):
        """Zero should be recognized as never-treated."""
        assert is_never_treated(0) is True
        assert is_never_treated(0.0) is True
    
    def test_inf_is_never_treated(self):
        """Positive infinity should be recognized as never-treated."""
        assert is_never_treated(np.inf) is True
    
    def test_nan_is_never_treated(self):
        """NaN should be recognized as never-treated."""
        assert is_never_treated(np.nan) is True
    
    def test_positive_integer_is_treated(self):
        """Positive integers should be recognized as treated (cohort)."""
        assert is_never_treated(2005) is False
        assert is_never_treated(2010) is False
        assert is_never_treated(1) is False
    
    def test_near_zero_is_never_treated(self):
        """Values very close to zero should be treated as never-treated."""
        assert is_never_treated(1e-10) is True
        assert is_never_treated(-1e-10) is True  # Tolerance for floating point


class TestCohortMaskFloatingPoint:
    """Tests for cohort mask with floating point tolerance."""
    
    def test_exact_match(self):
        """Exact integer matches should work."""
        gvar = pd.Series([2005, 2006, 2005, 0])
        mask = get_cohort_mask(gvar, 2005)
        
        assert mask.tolist() == [True, False, True, False]
    
    def test_floating_point_tolerance(self):
        """Floating point near-matches should be handled."""
        # get_cohort_mask uses 1e-9 tolerance (very strict)
        # 2005.0000001 differs by 1e-7 from 2005, which is > 1e-9, so it won't match
        # 2005.000000001 differs by 1e-9, which is at the boundary
        gvar = pd.Series([2005.000000001, 2005.9999999, 2006.0, 0])
        
        mask = get_cohort_mask(gvar, 2005)
        # 2005.000000001 is within 1e-9 tolerance of 2005
        assert mask[0] == True  # Very close to 2005
        
        # 2005.9999999 should NOT match 2005 (it's closer to 2006)
        # |2005.9999999 - 2005| = 0.9999999 > 1e-9
        assert mask[1] == False  # Far from 2005


class TestIntegrationNonIntegerGvar:
    """Integration tests for non-integer gvar validation."""
    
    def test_full_workflow_with_valid_integer_gvar(self):
        """Full validation workflow should work with valid integer gvar."""
        data = pd.DataFrame({
            'id': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            'year': [2000, 2001, 2002, 2000, 2001, 2002, 2000, 2001, 2002, 2000, 2001, 2002],
            'y': np.random.randn(12),
            'gvar': [2001, 2001, 2001, 2002, 2002, 2002, 0, 0, 0, 0, 0, 0]
        })
        
        result = validate_staggered_data(data, 'gvar', 'id', 'year', 'y')
        
        assert set(result['cohorts']) == {2001, 2002}
        assert result['n_treated'] == 2
        assert result['n_never_treated'] == 2
    
    def test_full_workflow_rejects_non_integer_gvar(self):
        """Full validation workflow should reject non-integer gvar."""
        data = pd.DataFrame({
            'id': [1, 1, 1, 2, 2, 2],
            'year': [2000, 2001, 2002, 2000, 2001, 2002],
            'y': np.random.randn(6),
            'gvar': [2001.5, 2001.5, 2001.5, 0, 0, 0]  # Non-integer cohort
        })
        
        with pytest.raises(InvalidStaggeredDataError):
            validate_staggered_data(data, 'gvar', 'id', 'year', 'y')
    
    def test_mixed_integer_types(self):
        """Should handle mixed integer types (int, float that is integer)."""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2, 3, 3],
            'year': [2000, 2001, 2000, 2001, 2000, 2001],
            'y': [1.0, 2.0, 1.5, 2.5, 1.2, 2.2],
            'gvar': [2001, 2001.0, 0, 0.0, np.nan, np.nan]  # Mixed types
        })
        
        result = validate_staggered_data(data, 'gvar', 'id', 'year', 'y')
        
        assert result['cohorts'] == [2001]
        assert result['n_treated'] == 1
        assert result['n_never_treated'] == 2


class TestNumericalValidation:
    """Numerical validation tests for the DESIGN-043 fix.
    
    DESIGN-045: Tolerance was changed from 1e-6 to 1e-9 to be consistent
    with get_cohort_mask() and other cohort comparison functions.
    """
    
    def test_tolerance_boundary_accept(self):
        """Values at tolerance boundary should be accepted."""
        # 1e-10 is within 1e-9 tolerance (COHORT_FLOAT_TOLERANCE)
        data = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'year': [2000, 2001, 2000, 2001],
            'y': [1.0, 2.0, 1.5, 2.5],
            'gvar': [2005.0000000001, 2005.0000000001, 0, 0]
        })
        
        result = validate_staggered_data(data, 'gvar', 'id', 'year', 'y')
        assert 2005 in result['cohorts']
    
    def test_tolerance_boundary_reject(self):
        """Values outside tolerance boundary should be rejected."""
        # 0.1 is definitely outside 1e-9 tolerance
        data = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'year': [2000, 2001, 2000, 2001],
            'y': [1.0, 2.0, 1.5, 2.5],
            'gvar': [2005.1, 2005.1, 0, 0]
        })
        
        with pytest.raises(InvalidStaggeredDataError):
            validate_staggered_data(data, 'gvar', 'id', 'year', 'y')
    
    def test_no_silent_truncation(self):
        """Verify 2005.9 is NOT silently truncated to 2005."""
        data = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'year': [2000, 2001, 2000, 2001],
            'y': [1.0, 2.0, 1.5, 2.5],
            'gvar': [2005.9, 2005.9, 0, 0]
        })
        
        # Before the fix, this would silently truncate 2005.9 to 2005
        # After the fix, it should raise an error
        with pytest.raises(InvalidStaggeredDataError) as exc_info:
            validate_staggered_data(data, 'gvar', 'id', 'year', 'y')
        
        assert "2005.9" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

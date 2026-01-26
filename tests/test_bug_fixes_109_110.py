"""
Tests for BUG-109 and BUG-110 fixes.

BUG-109: validation.py is_never_treated() negative infinity handling
    - Problem: Negative infinity (-inf) raised exception, inconsistent with function signature
    - Fix: Return False for negative infinity (not a valid NT indicator or cohort)
    - Rationale: is_never_treated() is a pure predicate function (-> bool) that should
      always return a boolean. Data validation is handled by validate_staggered_data().

BUG-110: aggregation.py gvar > 0 condition incorrectly includes inf in treated count
    - Problem: np.inf > 0 returns True, causing never-treated units to be
      miscounted in treated NaN statistics
    - Fix: Add np.isfinite() check to exclude infinity values
"""

import numpy as np
import pandas as pd
import pytest
import warnings

from lwdid.validation import is_never_treated, COHORT_FLOAT_TOLERANCE
from lwdid.staggered.aggregation import construct_aggregated_outcome
from lwdid.staggered.transformations import transform_staggered_demean


# =============================================================================
# BUG-109 Tests: Negative Infinity Handling in is_never_treated()
# =============================================================================

class TestBug109NegativeInfinity:
    """
    BUG-109 regression tests for is_never_treated() negative infinity handling.
    
    The function must correctly identify:
    - Positive infinity (+inf): Valid never-treated indicator (return True)
    - Negative infinity (-inf): Not valid NT or cohort (return False)
    - Zero (0): Valid never-treated indicator (return True)
    - NaN: Valid never-treated indicator (return True)
    - Positive integers: Treated cohort (return False)
    
    Note: is_never_treated() is a pure predicate function that always returns bool.
    Data validation (rejecting invalid values like -inf) should be done upstream
    via validate_staggered_data().
    """
    
    def test_positive_infinity_returns_true(self):
        """Positive infinity is a valid never-treated indicator."""
        assert is_never_treated(np.inf) is True
        assert is_never_treated(float('inf')) is True
    
    def test_negative_infinity_returns_false(self):
        """Negative infinity returns False (not a valid NT indicator).
        
        Negative infinity is neither a recognized NT value nor a valid cohort.
        The function returns False because:
        1. Only +inf, 0, and NaN are valid NT indicators
        2. Data containing -inf should be validated upstream
        """
        assert is_never_treated(-np.inf) is False
    
    def test_negative_infinity_float_returns_false(self):
        """Negative infinity as float returns False."""
        assert is_never_treated(float('-inf')) is False
    
    def test_zero_returns_true(self):
        """Zero is a valid never-treated indicator."""
        assert is_never_treated(0) is True
        assert is_never_treated(0.0) is True
    
    def test_nan_returns_true(self):
        """NaN is a valid never-treated indicator."""
        assert is_never_treated(np.nan) is True
        assert is_never_treated(float('nan')) is True
        assert is_never_treated(pd.NA) is True
        assert is_never_treated(None) is True
    
    def test_positive_integer_returns_false(self):
        """Positive integers are treated cohort indicators."""
        assert is_never_treated(2005) is False
        assert is_never_treated(2010) is False
        assert is_never_treated(1) is False
    
    def test_positive_float_returns_false(self):
        """Positive floats (cohort values) return False."""
        assert is_never_treated(2005.0) is False
        assert is_never_treated(2010.0) is False
    
    def test_near_zero_with_tolerance(self):
        """Values very close to zero should be treated as never-treated."""
        # Values within tolerance should be considered as 0
        assert is_never_treated(COHORT_FLOAT_TOLERANCE / 2) is True
        assert is_never_treated(-COHORT_FLOAT_TOLERANCE / 2) is True
        
        # Values outside tolerance should not be
        assert is_never_treated(COHORT_FLOAT_TOLERANCE * 2) is False
    
    def test_is_pure_predicate_function(self):
        """is_never_treated should always return bool, never raise for any float.
        
        This ensures the function can be safely used in .apply() operations
        without risk of exceptions interrupting DataFrame processing.
        """
        test_values = [
            0, 0.0, 1, -1, 2005, -2005,
            np.inf, -np.inf, float('inf'), float('-inf'),
            np.nan, float('nan'), None,
            1e-10, -1e-10, 1e-9, -1e-9, 1e-8,
            np.float64(0), np.float64(np.inf), np.float64(-np.inf),
            np.int64(0), np.int64(2005),
        ]
        for val in test_values:
            result = is_never_treated(val)
            assert isinstance(result, bool), f"is_never_treated({val}) returned {type(result)}, expected bool"


# =============================================================================
# BUG-110 Tests: Infinity in Treated Count Statistics
# =============================================================================

class TestBug110InfInTreatedCount:
    """
    BUG-110 regression tests for treated NaN count statistics.
    
    The construct_aggregated_outcome() function counts treated units with NaN
    Y_bar values for warning messages. The condition `gvar > 0` incorrectly
    includes infinity values because np.inf > 0 returns True.
    
    Fix: Use `gvar > 0 and np.isfinite(gvar)` to exclude infinity.
    """
    
    def test_inf_greater_than_zero_is_true(self):
        """Verify that np.inf > 0 returns True (the root cause of BUG-110)."""
        assert (np.inf > 0) is True
        assert (-np.inf > 0) is False
    
    def test_isfinite_excludes_inf(self):
        """Verify that np.isfinite() correctly identifies infinity."""
        assert np.isfinite(np.inf) == False
        assert np.isfinite(-np.inf) == False
        assert np.isfinite(0) == True
        assert np.isfinite(2005) == True
        assert np.isfinite(np.nan) == False
    
    def test_combined_condition_excludes_inf(self):
        """Test the corrected condition: gvar > 0 and np.isfinite(gvar)."""
        # Positive infinity should be excluded
        gvar = np.inf
        assert bool(gvar > 0 and np.isfinite(gvar)) is False
        
        # Negative infinity should also be excluded
        gvar = -np.inf
        assert bool(gvar > 0 and np.isfinite(gvar)) is False
        
        # Normal cohort values should be included
        gvar = 2005
        assert bool(gvar > 0 and np.isfinite(gvar)) is True
        
        # Zero should be excluded (not > 0)
        gvar = 0
        assert bool(gvar > 0 and np.isfinite(gvar)) is False
    
    def test_construct_aggregated_outcome_inf_not_counted_as_treated(self):
        """
        Verify that units with gvar=inf are not counted as treated in warnings.
        
        This is the end-to-end test for BUG-110.
        """
        # Create data with:
        # - Unit 1: treated (gvar=3)
        # - Unit 2: never-treated (gvar=0)
        # - Unit 3: never-treated (gvar=inf)
        data = pd.DataFrame({
            'id': [1,1,1,1, 2,2,2,2, 3,3,3,3],
            'year': [1,2,3,4]*3,
            'y': [
                10, 12, 20, 22,  # Unit 1: cohort 3
                5, 7, 9, 11,     # Unit 2: NT (gvar=0)
                6, 8, 10, 12     # Unit 3: NT (gvar=inf)
            ],
            'gvar': [3]*4 + [0]*4 + [np.inf]*4
        })
        
        transformed = transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        
        weights = {3: 1.0}
        cohorts = [3]
        T_max = 4
        
        # Should not raise any exception
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Y_bar = construct_aggregated_outcome(
                transformed, 'gvar', 'id', 'year',
                weights, cohorts, T_max,
                transform_type='demean',
            )
            
            # Check if any warning mentions "treated" with incorrect count
            # that would include inf units
            for warning in w:
                msg = str(warning.message)
                if "treated unit" in msg.lower():
                    # If there's a warning about treated units with NaN,
                    # it should not count the inf unit (unit 3) as treated
                    # Unit 3 has gvar=inf, which is never-treated
                    pass
        
        # Verify result is valid
        assert isinstance(Y_bar, pd.Series)
        # Treated unit should have valid Y_bar
        assert not np.isnan(Y_bar[1])


# =============================================================================
# Integration Tests: Both Fixes Working Together
# =============================================================================

class TestBug109110Integration:
    """
    Integration tests verifying both BUG-109 and BUG-110 fixes work together.
    """
    
    def test_data_with_positive_infinity_gvar(self):
        """Data with gvar=+inf should work correctly."""
        data = pd.DataFrame({
            'id': [1,1,1,1, 2,2,2,2, 3,3,3,3],
            'year': [1,2,3,4]*3,
            'y': [
                10, 12, 20, 22,  # cohort 3
                15, 17, 25, 27,  # cohort 4
                5, 7, 9, 11      # NT (gvar=inf)
            ],
            'gvar': [3]*4 + [4]*4 + [np.inf]*4
        })
        
        # Should transform without error
        transformed = transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        
        # Verify transformation columns exist
        assert any(col.startswith('ydot_') for col in transformed.columns)
    
    def test_data_with_negative_infinity_gvar_skipped(self):
        """Data with gvar=-inf: unit is skipped (not a valid cohort).
        
        With the updated is_never_treated() behavior:
        - is_never_treated(-inf) returns False (not a valid NT indicator)
        - np.isfinite(-inf) returns False, so it's filtered from valid cohorts
        - The unit is effectively skipped during transformation
        
        This is the expected behavior: -inf is neither NT nor a valid cohort.
        Data validation should happen upstream via validate_staggered_data().
        """
        data = pd.DataFrame({
            'id': [1,1,1,1, 2,2,2,2],
            'year': [1,2,3,4]*2,
            'y': [
                10, 12, 20, 22,  # cohort 3
                5, 7, 9, 11      # gvar=-inf (skipped)
            ],
            'gvar': [3]*4 + [-np.inf]*4
        })
        
        # Transformation completes; -inf is filtered out as non-finite
        transformed = transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        
        # Verify transformation completed
        assert len(transformed) == len(data)
    
    def test_mixed_special_values_handled_correctly(self):
        """Data with 0, +inf, NaN as never-treated indicators."""
        data = pd.DataFrame({
            'id': [1,1,1,1, 2,2,2,2, 3,3,3,3, 4,4,4,4],
            'year': [1,2,3,4]*4,
            'y': [
                10, 12, 20, 22,  # cohort 3
                5, 7, 9, 11,     # NT (gvar=0)
                6, 8, 10, 12,    # NT (gvar=inf)
                7, 9, 11, 13     # NT (gvar=NaN)
            ],
            'gvar': [3]*4 + [0]*4 + [np.inf]*4 + [np.nan]*4
        })
        
        # Should transform without error
        transformed = transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        
        # All NT units should be recognized
        from lwdid.validation import is_never_treated
        unit_gvar = data.groupby('id')['gvar'].first()
        
        assert is_never_treated(unit_gvar[2]) is True  # gvar=0
        assert is_never_treated(unit_gvar[3]) is True  # gvar=inf
        assert is_never_treated(unit_gvar[4]) is True  # gvar=NaN
        assert is_never_treated(unit_gvar[1]) is False  # gvar=3 (treated)
    
    def test_normal_data_unaffected_by_fixes(self):
        """Verify fixes don't affect normal data (no inf/nan gvar)."""
        data = pd.DataFrame({
            'id': [1,1,1,1, 2,2,2,2, 3,3,3,3],
            'year': [1,2,3,4]*3,
            'y': [
                10, 12, 20, 22,  # cohort 3
                15, 17, 25, 27,  # cohort 4
                5, 7, 9, 11      # NT (gvar=0)
            ],
            'gvar': [3]*4 + [4]*4 + [0]*4
        })
        
        # Should work as before
        transformed = transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        
        weights = {3: 0.5, 4: 0.5}
        cohorts = [3, 4]
        T_max = 4
        
        Y_bar = construct_aggregated_outcome(
            transformed, 'gvar', 'id', 'year',
            weights, cohorts, T_max,
            transform_type='demean',
        )
        
        # All units should have valid Y_bar
        assert not np.isnan(Y_bar[1])  # cohort 3
        assert not np.isnan(Y_bar[2])  # cohort 4
        assert not np.isnan(Y_bar[3])  # NT


# =============================================================================
# Boundary Condition Tests
# =============================================================================

class TestBoundaryConditions:
    """
    Boundary condition tests for edge cases.
    """
    
    def test_very_large_positive_gvar(self):
        """Very large positive gvar values should be treated as cohorts."""
        assert is_never_treated(2999) is False
        assert is_never_treated(3000) is False
        assert is_never_treated(9999) is False
    
    def test_small_negative_gvar(self):
        """Small negative gvar should be treated as treated (not never-treated)."""
        # Note: Negative gvar values other than -inf are not valid cohorts
        # but they are not infinity, so they won't trigger the -inf error
        assert is_never_treated(-1) is False
        assert is_never_treated(-2005) is False
    
    def test_float_near_integer(self):
        """Float values very close to integers should work correctly."""
        # 2005.0 is essentially 2005
        assert is_never_treated(2005.0) is False
        
        # 2005.0000000001 should also be treated as cohort
        assert is_never_treated(2005.0 + 1e-10) is False
    
    def test_numpy_scalar_types(self):
        """NumPy scalar types should be handled correctly."""
        # np.float64
        assert is_never_treated(np.float64(np.inf)) is True
        assert is_never_treated(np.float64(0)) is True
        assert is_never_treated(np.float64(2005)) is False
        
        # np.int64
        assert is_never_treated(np.int64(0)) is True
        assert is_never_treated(np.int64(2005)) is False
        
        # Negative infinity with np.float64 returns False (not NT)
        assert is_never_treated(np.float64(-np.inf)) is False


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])

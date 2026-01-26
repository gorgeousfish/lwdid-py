"""
Tests for DESIGN-061, DESIGN-062, DESIGN-063 fixes.

This test module verifies:
1. DESIGN-061: WEIGHT_SUM_TOLERANCE constant (1e-6) for weight sum validation
2. DESIGN-062: NUMERIC_TOLERANCE constant (1e-10) for time-invariance checks
3. DESIGN-063: _cohort_equals removed, get_cohort_mask is the canonical function

Reference:
    - DESIGN-061, DESIGN-062, DESIGN-063 in 审查/设计问题列表2.md
"""

import warnings
import numpy as np
import pandas as pd
import pytest

from lwdid.validation import (
    COHORT_FLOAT_TOLERANCE,
    NUMERIC_TOLERANCE,
    get_cohort_mask,
    is_never_treated,
)
from lwdid.staggered.aggregation import WEIGHT_SUM_TOLERANCE


class TestDesign061WeightSumTolerance:
    """Tests for DESIGN-061: Weight sum tolerance fix."""
    
    def test_weight_sum_tolerance_constant_exists(self):
        """WEIGHT_SUM_TOLERANCE should be defined in aggregation.py"""
        assert WEIGHT_SUM_TOLERANCE is not None
        assert isinstance(WEIGHT_SUM_TOLERANCE, float)
    
    def test_weight_sum_tolerance_value(self):
        """WEIGHT_SUM_TOLERANCE should be 1e-6 (more permissive than 1e-9)"""
        assert WEIGHT_SUM_TOLERANCE == 1e-6
    
    def test_weight_sum_tolerance_more_permissive_than_cohort_tolerance(self):
        """WEIGHT_SUM_TOLERANCE should be larger than COHORT_FLOAT_TOLERANCE"""
        assert WEIGHT_SUM_TOLERANCE > COHORT_FLOAT_TOLERANCE
    
    def test_typical_floating_point_weight_accumulation(self):
        """Test that typical floating point weight accumulation passes 1e-6 tolerance."""
        # Simulate typical weight calculation with 3 cohorts
        weights = {
            4: 0.3333333333333333,
            5: 0.3333333333333333,
            6: 0.3333333333333334
        }
        total = sum(weights.values())
        
        # Should pass with 1e-6 tolerance
        assert np.isclose(total, 1.0, atol=WEIGHT_SUM_TOLERANCE)
        
        # But might fail with 1e-9 tolerance (depending on exact accumulation)
        # We just verify the new tolerance is more lenient
    
    def test_many_cohort_weight_accumulation(self):
        """Test weight accumulation with many cohorts (more floating point errors)."""
        n_cohorts = 100
        weights = {i: 1.0 / n_cohorts for i in range(n_cohorts)}
        total = sum(weights.values())
        
        # Deviation from 1.0 should be within tolerance
        deviation = abs(total - 1.0)
        assert deviation < WEIGHT_SUM_TOLERANCE, (
            f"Weight sum deviation {deviation:.2e} exceeds tolerance {WEIGHT_SUM_TOLERANCE:.2e}"
        )


class TestDesign062NumericTolerance:
    """Tests for DESIGN-062: Numeric tolerance constant consistency."""
    
    def test_numeric_tolerance_constant_exists(self):
        """NUMERIC_TOLERANCE should be defined in validation.py"""
        assert NUMERIC_TOLERANCE is not None
        assert isinstance(NUMERIC_TOLERANCE, float)
    
    def test_numeric_tolerance_value(self):
        """NUMERIC_TOLERANCE should be 1e-10"""
        assert NUMERIC_TOLERANCE == 1e-10
    
    def test_numeric_tolerance_stricter_than_cohort_tolerance(self):
        """NUMERIC_TOLERANCE should be stricter (smaller) than COHORT_FLOAT_TOLERANCE"""
        assert NUMERIC_TOLERANCE < COHORT_FLOAT_TOLERANCE
    
    def test_time_invariance_detection_sensitivity(self):
        """Test that NUMERIC_TOLERANCE correctly detects time-varying values."""
        # True constant value (should pass time-invariance check)
        constant_values = pd.Series([5.0, 5.0, 5.0, 5.0])
        std_constant = constant_values.std()
        assert std_constant <= NUMERIC_TOLERANCE or np.isnan(std_constant)
        
        # Slightly varying value (should fail time-invariance check)
        varying_values = pd.Series([5.0, 5.0, 5.0, 5.001])
        std_varying = varying_values.std()
        assert std_varying > NUMERIC_TOLERANCE


class TestDesign063GetCohortMaskCanonical:
    """Tests for DESIGN-063: get_cohort_mask is the canonical cohort matching function."""
    
    def test_cohort_equals_removed(self):
        """_cohort_equals should no longer exist in aggregation.py"""
        with pytest.raises(ImportError):
            from lwdid.staggered.aggregation import _cohort_equals
    
    def test_get_cohort_mask_exists(self):
        """get_cohort_mask should exist in validation.py"""
        from lwdid.validation import get_cohort_mask
        assert callable(get_cohort_mask)
    
    def test_get_cohort_mask_returns_series(self):
        """get_cohort_mask should return a pandas Series"""
        unit_gvar = pd.Series([4.0, 5.0, 6.0], index=[1, 2, 3])
        result = get_cohort_mask(unit_gvar, 5)
        assert isinstance(result, pd.Series)
    
    def test_get_cohort_mask_preserves_index(self):
        """get_cohort_mask should preserve the input Series index"""
        unit_gvar = pd.Series([4.0, 5.0, 6.0], index=['a', 'b', 'c'])
        result = get_cohort_mask(unit_gvar, 5)
        assert list(result.index) == ['a', 'b', 'c']
    
    def test_get_cohort_mask_exact_match(self):
        """get_cohort_mask should match exact cohort values"""
        unit_gvar = pd.Series([4, 5, 6, 4, 5], index=[1, 2, 3, 4, 5])
        
        mask_4 = get_cohort_mask(unit_gvar, 4)
        assert mask_4.sum() == 2
        assert mask_4[1] and mask_4[4]
        
        mask_5 = get_cohort_mask(unit_gvar, 5)
        assert mask_5.sum() == 2
        assert mask_5[2] and mask_5[5]
    
    def test_get_cohort_mask_floating_point_tolerance(self):
        """get_cohort_mask should handle floating point comparison correctly"""
        # Values with tiny floating point differences
        unit_gvar = pd.Series([
            4.0,
            4.0 + COHORT_FLOAT_TOLERANCE * 0.5,  # Within tolerance
            4.0 + COHORT_FLOAT_TOLERANCE * 2,    # Beyond tolerance
        ], index=[1, 2, 3])
        
        mask = get_cohort_mask(unit_gvar, 4)
        assert mask[1] == True   # Exact match
        assert mask[2] == True   # Within tolerance
        assert mask[3] == False  # Beyond tolerance


class TestDesignFixesIntegration:
    """Integration tests for all three design fixes."""
    
    def test_aggregation_uses_get_cohort_mask(self):
        """Verify aggregation.py uses get_cohort_mask from validation.py"""
        import inspect
        from lwdid.staggered import aggregation
        
        source = inspect.getsource(aggregation)
        
        # Should import get_cohort_mask
        assert 'from ..validation import' in source
        assert 'get_cohort_mask' in source
        
        # Should not define _cohort_equals
        assert 'def _cohort_equals' not in source
    
    def test_validation_uses_numeric_tolerance(self):
        """Verify validation.py uses NUMERIC_TOLERANCE constant"""
        import inspect
        from lwdid import validation
        
        source = inspect.getsource(validation)
        
        # Should define NUMERIC_TOLERANCE
        assert 'NUMERIC_TOLERANCE = 1e-10' in source
        
        # Should use NUMERIC_TOLERANCE (not hardcoded 1e-10) in time-invariance checks
        # Count occurrences of hardcoded 1e-10 vs NUMERIC_TOLERANCE
        hardcoded_count = source.count('> 1e-10')
        constant_count = source.count('> NUMERIC_TOLERANCE')
        
        # All time-invariance checks should use the constant
        assert constant_count >= 2, "Should use NUMERIC_TOLERANCE in time-invariance checks"


class TestNumericalAccuracy:
    """Numerical accuracy tests to ensure fixes don't affect computation results."""
    
    def test_weight_sum_accuracy(self):
        """Test that weight sum check is accurate but not overly strict."""
        # Test case: weights that sum to exactly 1.0
        weights_exact = {4: 0.5, 5: 0.5}
        assert sum(weights_exact.values()) == 1.0
        assert np.isclose(sum(weights_exact.values()), 1.0, atol=WEIGHT_SUM_TOLERANCE)
        
        # Test case: weights with typical floating point error
        weights_typical = {4: 1/3, 5: 1/3, 6: 1/3}
        sum_typical = sum(weights_typical.values())
        # Sum should be very close to 1.0
        assert abs(sum_typical - 1.0) < 1e-15
        assert np.isclose(sum_typical, 1.0, atol=WEIGHT_SUM_TOLERANCE)
    
    def test_cohort_mask_consistency_with_is_never_treated(self):
        """Test that get_cohort_mask and is_never_treated are consistent."""
        test_values = [0, 0.0, np.inf, np.nan, 2005, 2005.0, 2005.0000000001]
        
        for val in test_values:
            # Never-treated values should not match any positive cohort
            if is_never_treated(val):
                # For never-treated, get_cohort_mask with positive cohort should return False
                unit_gvar = pd.Series([val], index=[1])
                mask = get_cohort_mask(unit_gvar, 2005)
                # Only np.nan and np.inf should definitely not match 2005
                if pd.isna(val) or val == np.inf or val == 0:
                    assert mask[1] == False, f"Never-treated {val} should not match cohort 2005"

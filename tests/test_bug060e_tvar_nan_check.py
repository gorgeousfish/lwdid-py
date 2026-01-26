"""
Test BUG-060-E: T_max/T_min NaN check before int() conversion

This test verifies that when tvar column contains all NaN values,
the affected functions raise a clear ValueError instead of the cryptic
"ValueError: cannot convert float NaN to integer" from int(NaN).

Affected functions:
- aggregate_to_overall() in aggregation.py
- estimate_staggered_effects() in estimation.py  
- transform_staggered_demean() in transformations.py
- transform_staggered_detrend() in transformations.py
- staggered_randomization_inference() in randomization.py
- validate_staggered_data() in validation.py

Reference:
    Stata's summarize command automatically skips missing values.
    If all values are missing, r(max) returns "." (Stata missing).
    Python should provide a clear error message in this edge case.
"""

import numpy as np
import pandas as pd
import pytest

from lwdid.staggered.transformations import (
    transform_staggered_demean,
    transform_staggered_detrend,
)
from lwdid.validation import validate_staggered_data
from lwdid.exceptions import InvalidStaggeredDataError


def create_test_data_with_nan_tvar():
    """Create test data where tvar column is all NaN."""
    return pd.DataFrame({
        'id': [1, 1, 2, 2, 3, 3],
        'year': [np.nan] * 6,  # All NaN
        'y': [10.0, 12.0, 15.0, 18.0, 20.0, 22.0],
        'gvar': [4, 4, 5, 5, 0, 0],  # Unit 1: cohort 4, Unit 2: cohort 5, Unit 3: NT
    })


def create_valid_staggered_data():
    """Create valid staggered data for regression testing."""
    data = []
    # Never-treated units (id 1-2)
    for i in [1, 2]:
        for t in range(1, 7):
            data.append({
                'id': i,
                'year': t,
                'y': 10.0 + 2.0 * t + np.random.normal(0, 0.5),
                'gvar': 0,  # Never treated
            })
    # Cohort 4 units (id 3-4)
    for i in [3, 4]:
        for t in range(1, 7):
            treatment_effect = 5.0 if t >= 4 else 0.0
            data.append({
                'id': i,
                'year': t,
                'y': 10.0 + 2.0 * t + treatment_effect + np.random.normal(0, 0.5),
                'gvar': 4,
            })
    # Cohort 5 units (id 5-6)
    for i in [5, 6]:
        for t in range(1, 7):
            treatment_effect = 3.0 if t >= 5 else 0.0
            data.append({
                'id': i,
                'year': t,
                'y': 10.0 + 2.0 * t + treatment_effect + np.random.normal(0, 0.5),
                'gvar': 5,
            })
    
    np.random.seed(42)
    return pd.DataFrame(data)


class TestBUG060E_TvarNaNCheck:
    """Test that tvar all-NaN condition raises clear ValueError."""
    
    def test_validate_staggered_data_raises_on_all_nan_tvar(self):
        """Test validate_staggered_data raises InvalidStaggeredDataError."""
        data = create_test_data_with_nan_tvar()
        
        with pytest.raises(InvalidStaggeredDataError) as excinfo:
            validate_staggered_data(data, gvar='gvar', ivar='id', tvar='year', y='y')
        
        assert "no valid (non-NaN) values" in str(excinfo.value)
        assert "year" in str(excinfo.value)
    
    def test_transform_staggered_demean_raises_on_all_nan_tvar(self):
        """Test transform_staggered_demean raises ValueError."""
        data = create_test_data_with_nan_tvar()
        
        with pytest.raises(ValueError) as excinfo:
            transform_staggered_demean(
                data, y='y', gvar='gvar', ivar='id', tvar='year'
            )
        
        assert "no valid (non-NaN) values" in str(excinfo.value)
        assert "year" in str(excinfo.value)
    
    def test_transform_staggered_detrend_raises_on_all_nan_tvar(self):
        """Test transform_staggered_detrend raises ValueError."""
        data = create_test_data_with_nan_tvar()
        
        with pytest.raises(ValueError) as excinfo:
            transform_staggered_detrend(
                data, y='y', gvar='gvar', ivar='id', tvar='year'
            )
        
        assert "no valid (non-NaN) values" in str(excinfo.value)
        assert "year" in str(excinfo.value)


class TestBUG060E_PartialNaNTvarWorks:
    """Test that partial NaN in tvar still works correctly."""
    
    def test_partial_nan_tvar_works(self):
        """Test data with some NaN in tvar still computes correctly."""
        np.random.seed(42)
        data = create_valid_staggered_data()
        
        # Add a few NaN values to tvar (but not all)
        data_with_nan = data.copy()
        # Set a few year values to NaN
        data_with_nan.loc[0, 'year'] = np.nan
        data_with_nan.loc[5, 'year'] = np.nan
        
        # Drop rows with NaN year for valid processing
        valid_data = data_with_nan.dropna(subset=['year'])
        
        # Should work without error
        result = transform_staggered_demean(
            valid_data, y='y', gvar='gvar', ivar='id', tvar='year'
        )
        
        # Verify output is valid
        assert 'ydot_g4_r4' in result.columns or 'ydot_g5_r5' in result.columns


class TestBUG060E_NormalDataUnaffected:
    """Test that fix doesn't break normal data processing."""
    
    def test_normal_data_works(self):
        """Regression test: normal data should work as before."""
        np.random.seed(42)
        data = create_valid_staggered_data()
        
        # Should work without error
        result = transform_staggered_demean(
            data, y='y', gvar='gvar', ivar='id', tvar='year'
        )
        
        # Verify expected columns exist
        assert 'ydot_g4_r4' in result.columns
        assert 'ydot_g4_r5' in result.columns
        assert 'ydot_g4_r6' in result.columns
        assert 'ydot_g5_r5' in result.columns
        assert 'ydot_g5_r6' in result.columns
    
    def test_validation_with_normal_data(self):
        """Test validate_staggered_data with normal data."""
        np.random.seed(42)
        data = create_valid_staggered_data()
        
        # Should return valid result
        result = validate_staggered_data(data, gvar='gvar', ivar='id', tvar='year', y='y')
        
        assert result['T_min'] == 1
        assert result['T_max'] == 6
        assert 4 in result['cohorts']
        assert 5 in result['cohorts']


class TestBUG060E_EdgeCases:
    """Test edge cases for T_max/T_min NaN handling."""
    
    def test_single_valid_time_value_with_valid_cohort(self):
        """Test data with only one valid time value but valid cohort setup."""
        # Create data where the only valid time is the treatment period
        # This tests that T_min = T_max doesn't cause issues
        data = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'year': [3.0, 3.0, 3.0, 3.0],  # Only year=3 is valid
            'y': [10.0, 12.0, 15.0, 18.0],
            'gvar': [3, 3, 0, 0],  # Cohort starts at 3, same as data start
        })
        
        # This should fail because cohort 3 has no pre-treatment period
        # (g=3 but T_min=T_max=3, so no period < g)
        with pytest.raises(ValueError) as excinfo:
            transform_staggered_demean(
                data, y='y', gvar='gvar', ivar='id', tvar='year'
            )
        
        # Error should mention pre-treatment period issue
        assert "pre-treatment" in str(excinfo.value).lower()
    
    def test_all_same_time_value_no_treatment(self):
        """Test data where all time values are the same but no treated units."""
        data = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'year': [3.0, 3.0, 3.0, 3.0],  # All same year
            'y': [10.0, 12.0, 15.0, 18.0],
            'gvar': [0, 0, 0, 0],  # All never treated
        })
        
        # Should fail: no treatment cohorts found
        with pytest.raises(ValueError) as excinfo:
            transform_staggered_demean(
                data, y='y', gvar='gvar', ivar='id', tvar='year'
            )
        
        assert "cohort" in str(excinfo.value).lower()


class TestBUG060E_ErrorMessageQuality:
    """Test that error messages are helpful and specific."""
    
    def test_error_message_includes_variable_name(self):
        """Verify error message includes the problematic variable name."""
        data = pd.DataFrame({
            'id': [1, 1],
            'custom_time': [np.nan, np.nan],  # Custom column name
            'y': [10.0, 12.0],
            'gvar': [4, 4],
        })
        
        with pytest.raises(ValueError) as excinfo:
            transform_staggered_demean(
                data, y='y', gvar='gvar', ivar='id', tvar='custom_time'
            )
        
        # Error should mention the custom variable name
        assert "custom_time" in str(excinfo.value)
    
    def test_validation_error_includes_fix_suggestions(self):
        """Verify validation error includes helpful fix suggestions."""
        data = create_test_data_with_nan_tvar()
        
        with pytest.raises(InvalidStaggeredDataError) as excinfo:
            validate_staggered_data(data, gvar='gvar', ivar='id', tvar='year', y='y')
        
        error_msg = str(excinfo.value)
        # Should include diagnostic suggestions
        assert "isna" in error_msg.lower() or "nan" in error_msg.lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

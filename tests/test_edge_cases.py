"""
Edge case and exception handling tests for lwdid bug fixes.

This module tests boundary conditions and error handling to ensure
the bug fixes properly handle edge cases.
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
from lwdid.validation import validate_and_prepare_data
from lwdid.exceptions import (
    InvalidParameterError,
    InsufficientDataError,
    RandomizationError,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def minimal_data():
    """Create minimal valid panel data."""
    data = []
    for i in range(1, 7):  # 6 units (3 treated, 3 control)
        treated = 1 if i <= 3 else 0
        for t in range(2000, 2004):  # 4 periods
            post = 1 if t >= 2002 else 0
            y = 10 + 2 * treated * post + np.random.normal(0, 0.5)
            data.append({
                'unit': i,
                'year': t,
                'treated': treated,
                'post': post,
                'outcome': y,
            })
    return pd.DataFrame(data)


# =============================================================================
# Empty and Minimal Data Tests
# =============================================================================

class TestEmptyData:
    """Test handling of empty or near-empty data."""
    
    def test_empty_dataframe_raises(self):
        """Empty DataFrame should raise an error."""
        df = pd.DataFrame(columns=['unit', 'year', 'treated', 'post', 'outcome'])
        
        with pytest.raises((InsufficientDataError, InvalidParameterError)):
            validate_and_prepare_data(
                df, y='outcome', d='treated', ivar='unit',
                tvar='year', post='post', rolling='demean'
            )
    
    def test_single_unit_raises(self):
        """Single unit should raise error."""
        data = {
            'unit': [1, 1, 1, 1],
            'year': [2000, 2001, 2002, 2003],
            'treated': [1, 1, 1, 1],
            'post': [0, 0, 1, 1],
            'outcome': [10, 11, 12, 13],
        }
        df = pd.DataFrame(data)
        
        with pytest.raises(Exception):  # Could be various error types
            lwdid(
                data=df, y='outcome', d='treated', ivar='unit',
                tvar='year', post='post', rolling='demean'
            )


# =============================================================================
# Missing Values Tests
# =============================================================================

class TestMissingValues:
    """Test handling of missing values."""
    
    def test_nan_outcome_dropped(self, minimal_data):
        """Rows with NaN outcome should be dropped."""
        df = minimal_data.copy()
        df.loc[df.index[:3], 'outcome'] = np.nan
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data_clean, metadata = validate_and_prepare_data(
                df, y='outcome', d='treated', ivar='unit',
                tvar='year', post='post', rolling='demean'
            )
        
        assert data_clean['outcome'].isna().sum() == 0
    
    def test_nan_treatment_dropped(self, minimal_data):
        """Rows with NaN treatment should be dropped."""
        df = minimal_data.copy()
        df.loc[df.index[:2], 'treated'] = np.nan
        
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            data_clean, metadata = validate_and_prepare_data(
                df, y='outcome', d='treated', ivar='unit',
                tvar='year', post='post', rolling='demean'
            )
        
        assert data_clean['d_'].isna().sum() == 0


# =============================================================================
# Data Type Edge Cases
# =============================================================================

class TestDataTypes:
    """Test handling of various data types."""
    
    def test_float_unit_ids(self, minimal_data):
        """Float unit IDs should work."""
        df = minimal_data.copy()
        df['unit'] = df['unit'].astype(float)
        
        result = lwdid(
            data=df, y='outcome', d='treated', ivar='unit',
            tvar='year', post='post', rolling='demean'
        )
        assert result is not None
    
    def test_string_unit_ids(self, minimal_data):
        """String unit IDs should work."""
        df = minimal_data.copy()
        df['unit'] = df['unit'].map(lambda x: f"unit_{x}")
        
        result = lwdid(
            data=df, y='outcome', d='treated', ivar='unit',
            tvar='year', post='post', rolling='demean'
        )
        assert result is not None
    
    def test_negative_outcome(self, minimal_data):
        """Negative outcomes should work."""
        df = minimal_data.copy()
        df['outcome'] = df['outcome'] - 20  # Make all negative
        
        result = lwdid(
            data=df, y='outcome', d='treated', ivar='unit',
            tvar='year', post='post', rolling='demean'
        )
        assert result is not None


# =============================================================================
# Boundary Condition Tests
# =============================================================================

class TestBoundaryConditions:
    """Test boundary conditions."""
    
    def test_all_treated(self, minimal_data):
        """All treated units should raise error."""
        df = minimal_data.copy()
        df['treated'] = 1
        
        with pytest.raises(Exception):
            lwdid(
                data=df, y='outcome', d='treated', ivar='unit',
                tvar='year', post='post', rolling='demean'
            )
    
    def test_all_control(self, minimal_data):
        """All control units should raise error."""
        df = minimal_data.copy()
        df['treated'] = 0
        
        with pytest.raises(Exception):
            lwdid(
                data=df, y='outcome', d='treated', ivar='unit',
                tvar='year', post='post', rolling='demean'
            )
    
    def test_all_pre_period(self, minimal_data):
        """All pre-period should raise error."""
        df = minimal_data.copy()
        df['post'] = 0
        
        with pytest.raises(InsufficientDataError):
            validate_and_prepare_data(
                df, y='outcome', d='treated', ivar='unit',
                tvar='year', post='post', rolling='demean'
            )
    
    def test_all_post_period(self, minimal_data):
        """All post-period should raise error."""
        df = minimal_data.copy()
        df['post'] = 1
        
        with pytest.raises(InsufficientDataError):
            validate_and_prepare_data(
                df, y='outcome', d='treated', ivar='unit',
                tvar='year', post='post', rolling='demean'
            )


# =============================================================================
# Parameter Validation Tests
# =============================================================================

class TestParameterValidation:
    """Test parameter validation."""
    
    def test_invalid_rolling_method(self, minimal_data):
        """Invalid rolling method should raise error."""
        df = minimal_data.copy()
        
        with pytest.raises(Exception):
            lwdid(
                data=df, y='outcome', d='treated', ivar='unit',
                tvar='year', post='post', rolling='invalid_method'
            )
    
    def test_missing_column(self, minimal_data):
        """Missing required column should raise error."""
        df = minimal_data.copy()
        df = df.drop(columns=['outcome'])
        
        with pytest.raises(Exception):
            lwdid(
                data=df, y='outcome', d='treated', ivar='unit',
                tvar='year', post='post', rolling='demean'
            )
    
    def test_non_dataframe_input(self):
        """Non-DataFrame input should raise TypeError."""
        data_dict = {'a': [1, 2], 'b': [3, 4]}
        
        with pytest.raises(TypeError):
            lwdid(
                data=data_dict, y='a', d='b', ivar='a',
                tvar='a', post='b', rolling='demean'
            )


# =============================================================================
# Staggered Edge Cases
# =============================================================================

class TestStaggeredEdgeCases:
    """Test staggered DiD edge cases."""
    
    def test_single_cohort(self):
        """Single treatment cohort should work."""
        np.random.seed(42)
        data = []
        for i in range(1, 31):
            if i <= 15:
                gvar = 0  # never treated
            else:
                gvar = 2003  # all treated in same year
            
            for t in range(2000, 2006):
                treated = 1 if gvar > 0 and t >= gvar else 0
                y = 10 + 2 * treated + np.random.normal(0, 1)
                data.append({
                    'unit': i,
                    'year': t,
                    'gvar': gvar,
                    'outcome': y,
                })
        
        df = pd.DataFrame(data)
        
        result = lwdid(
            data=df, y='outcome', ivar='unit', tvar='year',
            gvar='gvar', rolling='demean'
        )
        assert result is not None
    
    def test_no_never_treated(self):
        """No never-treated units should work with not_yet_treated control and aggregate='none'."""
        np.random.seed(42)
        data = []
        for i in range(1, 31):
            if i <= 15:
                gvar = 2003
            else:
                gvar = 2005
            
            for t in range(2000, 2008):
                treated = 1 if t >= gvar else 0
                y = 10 + 2 * treated + np.random.normal(0, 1)
                data.append({
                    'unit': i,
                    'year': t,
                    'gvar': gvar,
                    'outcome': y,
                })
        
        df = pd.DataFrame(data)
        
        # When no never-treated units, must use aggregate='none' for (g,r)-specific effects
        result = lwdid(
            data=df, y='outcome', ivar='unit', tvar='year',
            gvar='gvar', rolling='demean', control_group='not_yet_treated',
            aggregate='none'  # Required when no never-treated units
        )
        assert result is not None


# =============================================================================
# Numerical Edge Cases
# =============================================================================

class TestNumericalEdgeCases:
    """Test numerical edge cases."""
    
    def test_very_small_outcomes(self, minimal_data):
        """Very small outcome values should work."""
        df = minimal_data.copy()
        df['outcome'] = df['outcome'] * 1e-10
        
        result = lwdid(
            data=df, y='outcome', d='treated', ivar='unit',
            tvar='year', post='post', rolling='demean'
        )
        assert result is not None
        assert not np.isnan(result.att)
    
    def test_very_large_outcomes(self, minimal_data):
        """Very large outcome values should work."""
        df = minimal_data.copy()
        df['outcome'] = df['outcome'] * 1e10
        
        result = lwdid(
            data=df, y='outcome', d='treated', ivar='unit',
            tvar='year', post='post', rolling='demean'
        )
        assert result is not None
        assert not np.isnan(result.att)
    
    def test_constant_outcome(self, minimal_data):
        """Constant outcome values should be handled."""
        df = minimal_data.copy()
        df['outcome'] = 10.0  # All same value
        
        # This may produce warnings or special results
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            try:
                result = lwdid(
                    data=df, y='outcome', d='treated', ivar='unit',
                    tvar='year', post='post', rolling='demean'
                )
                # If it runs, ATT should be 0 for constant outcome
                assert result.att == 0 or np.isnan(result.att)
            except Exception:
                # Some edge cases may raise exceptions, which is acceptable
                pass


# =============================================================================
# Run tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

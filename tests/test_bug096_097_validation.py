"""
Unit tests for BUG-096 and BUG-097 fixes.

BUG-096: trim_threshold parameter validation in estimate_ipwra/ipw/psm
BUG-097: controls numeric type validation in validate_staggered_data

These tests verify that:
1. Invalid trim_threshold values raise ValueError
2. Non-numeric controls raise InvalidParameterError in staggered validation
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lwdid.staggered.estimators import (
    estimate_ipwra,
    estimate_ipw,
    estimate_psm,
)
from lwdid.validation import validate_staggered_data
from lwdid.exceptions import InvalidParameterError


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def simple_cross_sectional_data():
    """Create simple cross-sectional data for IPW/IPWRA/PSM tests."""
    np.random.seed(42)
    n = 200
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    # True propensity: logit(p) = -0.5 + 0.5*x1 + 0.3*x2
    logit_p = -0.5 + 0.5 * x1 + 0.3 * x2
    p_true = 1 / (1 + np.exp(-logit_p))
    d = (np.random.uniform(0, 1, n) < p_true).astype(int)
    # Outcome: y = 1.0 + 0.5*d + 0.3*x1 + 0.2*x2 + epsilon
    y = 1.0 + 0.5 * d + 0.3 * x1 + 0.2 * x2 + np.random.normal(0, 0.5, n)
    
    return pd.DataFrame({
        'y': y,
        'd': d,
        'x1': x1,
        'x2': x2,
    })


@pytest.fixture
def simple_staggered_data():
    """Create simple staggered panel data for validation tests."""
    np.random.seed(42)
    n_units = 50
    n_periods = 10
    
    # Create panel structure
    units = np.repeat(np.arange(n_units), n_periods)
    periods = np.tile(np.arange(1, n_periods + 1), n_units)
    
    # Assign cohorts (30 treated, 20 never-treated)
    cohorts = np.zeros(n_units)
    cohorts[:10] = 5  # Cohort 1: treated at period 5
    cohorts[10:20] = 7  # Cohort 2: treated at period 7
    cohorts[20:30] = 9  # Cohort 3: treated at period 9
    cohorts[30:] = 0  # Never treated
    gvar = np.repeat(cohorts, n_periods)
    
    # Generate covariates
    x1 = np.random.normal(0, 1, n_units * n_periods)
    x2 = np.random.normal(0, 1, n_units * n_periods)
    
    # Generate outcome
    y = 1.0 + 0.3 * x1 + 0.2 * x2 + np.random.normal(0, 0.5, n_units * n_periods)
    
    return pd.DataFrame({
        'ivar': units,
        'tvar': periods,
        'gvar': gvar,
        'y': y,
        'x1': x1,
        'x2': x2,
    })


# =============================================================================
# BUG-096: trim_threshold Parameter Validation Tests
# =============================================================================

class TestTrimThresholdValidation:
    """Test trim_threshold parameter validation (BUG-096)."""
    
    # -------------------------------------------------------------------------
    # estimate_ipwra tests
    # -------------------------------------------------------------------------
    
    def test_ipwra_trim_threshold_negative_raises(self, simple_cross_sectional_data):
        """estimate_ipwra should raise ValueError for negative trim_threshold."""
        with pytest.raises(ValueError, match="trim_threshold must be in range"):
            estimate_ipwra(
                simple_cross_sectional_data,
                y='y',
                d='d',
                controls=['x1', 'x2'],
                trim_threshold=-0.1,
            )
    
    def test_ipwra_trim_threshold_zero_raises(self, simple_cross_sectional_data):
        """estimate_ipwra should raise ValueError for trim_threshold=0."""
        with pytest.raises(ValueError, match="trim_threshold must be in range"):
            estimate_ipwra(
                simple_cross_sectional_data,
                y='y',
                d='d',
                controls=['x1', 'x2'],
                trim_threshold=0,
            )
    
    def test_ipwra_trim_threshold_half_raises(self, simple_cross_sectional_data):
        """estimate_ipwra should raise ValueError for trim_threshold=0.5."""
        with pytest.raises(ValueError, match="trim_threshold must be in range"):
            estimate_ipwra(
                simple_cross_sectional_data,
                y='y',
                d='d',
                controls=['x1', 'x2'],
                trim_threshold=0.5,
            )
    
    def test_ipwra_trim_threshold_large_raises(self, simple_cross_sectional_data):
        """estimate_ipwra should raise ValueError for trim_threshold>=0.5."""
        with pytest.raises(ValueError, match="trim_threshold must be in range"):
            estimate_ipwra(
                simple_cross_sectional_data,
                y='y',
                d='d',
                controls=['x1', 'x2'],
                trim_threshold=0.6,
            )
    
    def test_ipwra_trim_threshold_valid_values(self, simple_cross_sectional_data):
        """estimate_ipwra should accept valid trim_threshold values."""
        for trim in [0.01, 0.05, 0.1, 0.2, 0.49]:
            result = estimate_ipwra(
                simple_cross_sectional_data,
                y='y',
                d='d',
                controls=['x1', 'x2'],
                trim_threshold=trim,
            )
            assert result is not None
            assert not np.isnan(result.att)
    
    # -------------------------------------------------------------------------
    # estimate_ipw tests
    # -------------------------------------------------------------------------
    
    def test_ipw_trim_threshold_negative_raises(self, simple_cross_sectional_data):
        """estimate_ipw should raise ValueError for negative trim_threshold."""
        with pytest.raises(ValueError, match="trim_threshold must be in range"):
            estimate_ipw(
                simple_cross_sectional_data,
                y='y',
                d='d',
                propensity_controls=['x1', 'x2'],
                trim_threshold=-0.1,
            )
    
    def test_ipw_trim_threshold_zero_raises(self, simple_cross_sectional_data):
        """estimate_ipw should raise ValueError for trim_threshold=0."""
        with pytest.raises(ValueError, match="trim_threshold must be in range"):
            estimate_ipw(
                simple_cross_sectional_data,
                y='y',
                d='d',
                propensity_controls=['x1', 'x2'],
                trim_threshold=0,
            )
    
    def test_ipw_trim_threshold_half_raises(self, simple_cross_sectional_data):
        """estimate_ipw should raise ValueError for trim_threshold=0.5."""
        with pytest.raises(ValueError, match="trim_threshold must be in range"):
            estimate_ipw(
                simple_cross_sectional_data,
                y='y',
                d='d',
                propensity_controls=['x1', 'x2'],
                trim_threshold=0.5,
            )
    
    def test_ipw_trim_threshold_valid_values(self, simple_cross_sectional_data):
        """estimate_ipw should accept valid trim_threshold values."""
        for trim in [0.01, 0.05, 0.1]:
            result = estimate_ipw(
                simple_cross_sectional_data,
                y='y',
                d='d',
                propensity_controls=['x1', 'x2'],
                trim_threshold=trim,
            )
            assert result is not None
            assert not np.isnan(result.att)
    
    # -------------------------------------------------------------------------
    # estimate_psm tests
    # -------------------------------------------------------------------------
    
    def test_psm_trim_threshold_negative_raises(self, simple_cross_sectional_data):
        """estimate_psm should raise ValueError for negative trim_threshold."""
        with pytest.raises(ValueError, match="trim_threshold must be in range"):
            estimate_psm(
                simple_cross_sectional_data,
                y='y',
                d='d',
                propensity_controls=['x1', 'x2'],
                trim_threshold=-0.1,
            )
    
    def test_psm_trim_threshold_zero_raises(self, simple_cross_sectional_data):
        """estimate_psm should raise ValueError for trim_threshold=0."""
        with pytest.raises(ValueError, match="trim_threshold must be in range"):
            estimate_psm(
                simple_cross_sectional_data,
                y='y',
                d='d',
                propensity_controls=['x1', 'x2'],
                trim_threshold=0,
            )
    
    def test_psm_trim_threshold_half_raises(self, simple_cross_sectional_data):
        """estimate_psm should raise ValueError for trim_threshold=0.5."""
        with pytest.raises(ValueError, match="trim_threshold must be in range"):
            estimate_psm(
                simple_cross_sectional_data,
                y='y',
                d='d',
                propensity_controls=['x1', 'x2'],
                trim_threshold=0.5,
            )
    
    def test_psm_trim_threshold_valid_values(self, simple_cross_sectional_data):
        """estimate_psm should accept valid trim_threshold values."""
        for trim in [0.01, 0.05, 0.1]:
            result = estimate_psm(
                simple_cross_sectional_data,
                y='y',
                d='d',
                propensity_controls=['x1', 'x2'],
                trim_threshold=trim,
                n_neighbors=1,
            )
            assert result is not None
            assert not np.isnan(result.att)


# =============================================================================
# BUG-097: Controls Numeric Type Validation Tests
# =============================================================================

class TestControlsNumericValidation:
    """Test controls numeric type validation in validate_staggered_data (BUG-097)."""
    
    def test_string_control_raises_error(self, simple_staggered_data):
        """validate_staggered_data should raise error for string control columns."""
        # Add a string column as control
        data = simple_staggered_data.copy()
        data['region'] = np.random.choice(['North', 'South', 'East', 'West'], len(data))
        
        with pytest.raises(InvalidParameterError, match="must be numeric"):
            validate_staggered_data(
                data,
                gvar='gvar',
                ivar='ivar',
                tvar='tvar',
                y='y',
                controls=['x1', 'region'],
            )
    
    def test_categorical_control_raises_error(self, simple_staggered_data):
        """validate_staggered_data should raise error for categorical control columns."""
        data = simple_staggered_data.copy()
        data['category'] = pd.Categorical(
            np.random.choice(['A', 'B', 'C'], len(data))
        )
        
        with pytest.raises(InvalidParameterError, match="must be numeric"):
            validate_staggered_data(
                data,
                gvar='gvar',
                ivar='ivar',
                tvar='tvar',
                y='y',
                controls=['x1', 'category'],
            )
    
    def test_object_dtype_control_raises_error(self, simple_staggered_data):
        """validate_staggered_data should raise error for object dtype control columns."""
        data = simple_staggered_data.copy()
        data['obj_col'] = ['item_' + str(i % 10) for i in range(len(data))]
        
        with pytest.raises(InvalidParameterError, match="must be numeric"):
            validate_staggered_data(
                data,
                gvar='gvar',
                ivar='ivar',
                tvar='tvar',
                y='y',
                controls=['x1', 'obj_col'],
            )
    
    def test_numeric_controls_accepted(self, simple_staggered_data):
        """validate_staggered_data should accept numeric control columns."""
        data = simple_staggered_data.copy()
        
        # Add various numeric types
        data['x3_int'] = np.random.randint(0, 10, len(data))
        data['x4_float32'] = np.random.normal(0, 1, len(data)).astype(np.float32)
        data['x5_int64'] = np.random.randint(-100, 100, len(data)).astype(np.int64)
        
        # Should not raise
        validate_staggered_data(
            data,
            gvar='gvar',
            ivar='ivar',
            tvar='tvar',
            y='y',
            controls=['x1', 'x2', 'x3_int', 'x4_float32', 'x5_int64'],
        )
    
    def test_no_controls_accepted(self, simple_staggered_data):
        """validate_staggered_data should accept None or empty controls."""
        # None controls
        validate_staggered_data(
            simple_staggered_data,
            gvar='gvar',
            ivar='ivar',
            tvar='tvar',
            y='y',
            controls=None,
        )
        
        # Empty list controls
        validate_staggered_data(
            simple_staggered_data,
            gvar='gvar',
            ivar='ivar',
            tvar='tvar',
            y='y',
            controls=[],
        )
    
    def test_inf_in_controls_raises_error(self, simple_staggered_data):
        """validate_staggered_data should raise error for infinite values in controls."""
        data = simple_staggered_data.copy()
        data.loc[0, 'x1'] = np.inf  # Add infinite value
        
        with pytest.raises(InvalidParameterError, match="infinite values"):
            validate_staggered_data(
                data,
                gvar='gvar',
                ivar='ivar',
                tvar='tvar',
                y='y',
                controls=['x1', 'x2'],
            )
    
    def test_multiple_non_numeric_controls_raises_error(self, simple_staggered_data):
        """validate_staggered_data should list all non-numeric controls in error message."""
        data = simple_staggered_data.copy()
        data['str_col'] = ['a'] * len(data)
        data['cat_col'] = pd.Categorical(['x', 'y'] * (len(data) // 2))
        
        with pytest.raises(InvalidParameterError) as exc_info:
            validate_staggered_data(
                data,
                gvar='gvar',
                ivar='ivar',
                tvar='tvar',
                y='y',
                controls=['x1', 'str_col', 'cat_col'],
            )
        
        # Error message should mention both non-numeric columns
        error_msg = str(exc_info.value)
        assert 'str_col' in error_msg
        assert 'cat_col' in error_msg


# =============================================================================
# Error Message Quality Tests
# =============================================================================

class TestErrorMessageQuality:
    """Test that error messages are informative and helpful."""
    
    def test_trim_threshold_error_suggests_common_values(self, simple_cross_sectional_data):
        """Error message should suggest common trim_threshold values."""
        with pytest.raises(ValueError) as exc_info:
            estimate_ipwra(
                simple_cross_sectional_data,
                y='y',
                d='d',
                controls=['x1', 'x2'],
                trim_threshold=-0.1,
            )
        
        error_msg = str(exc_info.value)
        assert '0.01' in error_msg or 'default' in error_msg.lower()
    
    def test_controls_error_explains_why(self, simple_staggered_data):
        """Error message should explain why controls must be numeric."""
        data = simple_staggered_data.copy()
        data['region'] = np.random.choice(['A', 'B'], len(data))
        
        with pytest.raises(InvalidParameterError) as exc_info:
            validate_staggered_data(
                data,
                gvar='gvar',
                ivar='ivar',
                tvar='tvar',
                y='y',
                controls=['region'],
            )
        
        error_msg = str(exc_info.value)
        # Should explain the reason
        assert 'numeric' in error_msg.lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

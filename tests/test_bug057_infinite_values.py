"""Tests for BUG-057: Infinite value validation in outcome and control variables.

This module tests the fix for BUG-057 which adds validation for infinite values
(inf/-inf) in outcome variables and control variables.

Test Categories:
1. Unit tests for _validate_outcome_dtype
2. Unit tests for _validate_controls_dtype
3. Integration tests with validate_and_prepare_data
4. Edge cases and boundary conditions
5. Error message quality verification
"""

import numpy as np
import pandas as pd
import pytest

from lwdid.validation import (
    validate_and_prepare_data,
    _validate_outcome_dtype,
    _validate_controls_dtype,
)
from lwdid.exceptions import InvalidParameterError


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def basic_panel_data():
    """Basic valid panel data for testing."""
    return pd.DataFrame({
        'id': [1, 1, 2, 2, 3, 3],
        'year': [2000, 2001, 2000, 2001, 2000, 2001],
        'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        'd': [1, 1, 0, 0, 0, 0],
        'post': [0, 1, 0, 1, 0, 1],
    })


@pytest.fixture
def data_with_controls():
    """Panel data with control variables."""
    return pd.DataFrame({
        'id': [1, 1, 2, 2, 3, 3],
        'year': [2000, 2001, 2000, 2001, 2000, 2001],
        'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        'd': [1, 1, 0, 0, 0, 0],
        'post': [0, 1, 0, 1, 0, 1],
        'x1': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        'x2': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    })


# =============================================================================
# Unit Tests: Outcome Variable Infinite Value Detection
# =============================================================================

class TestOutcomeInfiniteValueDetection:
    """Unit tests for infinite value detection in outcome variable."""
    
    def test_outcome_with_positive_inf(self, basic_panel_data):
        """Positive infinity in outcome should raise InvalidParameterError."""
        data = basic_panel_data.copy()
        data.loc[0, 'y'] = np.inf
        
        with pytest.raises(InvalidParameterError) as exc_info:
            _validate_outcome_dtype(data, 'y')
        
        error_msg = str(exc_info.value)
        assert "infinite" in error_msg.lower()
        assert "1" in error_msg  # Should report count
        assert "+inf" in error_msg or "Positive infinity" in error_msg
    
    def test_outcome_with_negative_inf(self, basic_panel_data):
        """Negative infinity in outcome should raise InvalidParameterError."""
        data = basic_panel_data.copy()
        data.loc[0, 'y'] = -np.inf
        
        with pytest.raises(InvalidParameterError) as exc_info:
            _validate_outcome_dtype(data, 'y')
        
        error_msg = str(exc_info.value)
        assert "infinite" in error_msg.lower()
        assert "-inf" in error_msg or "Negative infinity" in error_msg
    
    def test_outcome_with_mixed_inf(self, basic_panel_data):
        """Mixed positive and negative infinity should raise with detailed counts."""
        data = basic_panel_data.copy()
        data.loc[0, 'y'] = np.inf
        data.loc[1, 'y'] = -np.inf
        data.loc[2, 'y'] = np.inf
        
        with pytest.raises(InvalidParameterError) as exc_info:
            _validate_outcome_dtype(data, 'y')
        
        error_msg = str(exc_info.value)
        # Should report 3 total infinite values
        assert "3" in error_msg
        # Should breakdown by type
        assert "+inf" in error_msg and "-inf" in error_msg
    
    def test_outcome_no_inf_passes(self, basic_panel_data):
        """Outcome without infinite values should pass validation."""
        # Should not raise any exception
        _validate_outcome_dtype(basic_panel_data, 'y')
    
    def test_outcome_with_nan_passes(self, basic_panel_data):
        """NaN values should still pass (handled separately by dropna)."""
        data = basic_panel_data.copy()
        data.loc[0, 'y'] = np.nan
        
        # NaN is handled by dropna, not by inf check
        _validate_outcome_dtype(data, 'y')  # Should not raise
    
    def test_outcome_inf_all_values(self, basic_panel_data):
        """All infinite values should be caught."""
        data = basic_panel_data.copy()
        data['y'] = np.inf
        
        with pytest.raises(InvalidParameterError) as exc_info:
            _validate_outcome_dtype(data, 'y')
        
        error_msg = str(exc_info.value)
        # All 6 observations are inf
        assert "6" in error_msg
        assert "100.00%" in error_msg


# =============================================================================
# Unit Tests: Control Variables Infinite Value Detection
# =============================================================================

class TestControlsInfiniteValueDetection:
    """Unit tests for infinite value detection in control variables."""
    
    def test_control_with_positive_inf(self, data_with_controls):
        """Positive infinity in control should raise InvalidParameterError."""
        data = data_with_controls.copy()
        data.loc[0, 'x1'] = np.inf
        
        with pytest.raises(InvalidParameterError) as exc_info:
            _validate_controls_dtype(data, ['x1', 'x2'])
        
        error_msg = str(exc_info.value)
        assert "infinite" in error_msg.lower()
        assert "x1" in error_msg
    
    def test_control_with_negative_inf(self, data_with_controls):
        """Negative infinity in control should raise InvalidParameterError."""
        data = data_with_controls.copy()
        data.loc[0, 'x2'] = -np.inf
        
        with pytest.raises(InvalidParameterError) as exc_info:
            _validate_controls_dtype(data, ['x1', 'x2'])
        
        error_msg = str(exc_info.value)
        assert "infinite" in error_msg.lower()
        assert "x2" in error_msg
    
    def test_multiple_controls_with_inf(self, data_with_controls):
        """Multiple controls with infinity should list all problematic columns."""
        data = data_with_controls.copy()
        data.loc[0, 'x1'] = np.inf
        data.loc[1, 'x2'] = -np.inf
        
        with pytest.raises(InvalidParameterError) as exc_info:
            _validate_controls_dtype(data, ['x1', 'x2'])
        
        error_msg = str(exc_info.value)
        # Both controls should be mentioned
        assert "x1" in error_msg
        assert "x2" in error_msg
    
    def test_controls_no_inf_passes(self, data_with_controls):
        """Controls without infinite values should pass validation."""
        _validate_controls_dtype(data_with_controls, ['x1', 'x2'])
    
    def test_empty_controls_passes(self, data_with_controls):
        """Empty controls list should pass validation."""
        _validate_controls_dtype(data_with_controls, [])
        _validate_controls_dtype(data_with_controls, None)


# =============================================================================
# Integration Tests: Full Pipeline
# =============================================================================

class TestInfiniteValueIntegration:
    """Integration tests with validate_and_prepare_data."""
    
    def test_pipeline_catches_outcome_inf(self, basic_panel_data):
        """Full pipeline should catch infinite values in outcome."""
        data = basic_panel_data.copy()
        data.loc[0, 'y'] = np.inf
        
        with pytest.raises(InvalidParameterError) as exc_info:
            validate_and_prepare_data(
                data, y='y', d='d', ivar='id', tvar='year',
                post='post', rolling='demean'
            )
        
        assert "infinite" in str(exc_info.value).lower()
    
    def test_pipeline_catches_control_inf(self, data_with_controls):
        """Full pipeline should catch infinite values in controls."""
        data = data_with_controls.copy()
        data.loc[0, 'x1'] = np.inf
        
        with pytest.raises(InvalidParameterError) as exc_info:
            validate_and_prepare_data(
                data, y='y', d='d', ivar='id', tvar='year',
                post='post', rolling='demean', controls=['x1', 'x2']
            )
        
        assert "infinite" in str(exc_info.value).lower()


# =============================================================================
# Error Message Quality Tests
# =============================================================================

class TestErrorMessageQuality:
    """Verify error messages contain required sections (DESIGN-025 compliance)."""
    
    def _verify_message_sections(self, error_msg: str):
        """Helper to verify error message contains required sections."""
        assert "Why this matters:" in error_msg, \
            f"Error message should contain 'Why this matters' section:\n{error_msg}"
        assert "How to fix:" in error_msg, \
            f"Error message should contain 'How to fix' section:\n{error_msg}"
    
    def test_outcome_inf_error_message_quality(self, basic_panel_data):
        """Outcome infinite error message should have Why/How sections."""
        data = basic_panel_data.copy()
        data.loc[0, 'y'] = np.inf
        
        with pytest.raises(InvalidParameterError) as exc_info:
            _validate_outcome_dtype(data, 'y')
        
        error_msg = str(exc_info.value)
        self._verify_message_sections(error_msg)
        
        # Should include specific guidance
        assert "np.isinf" in error_msg or "isinf" in error_msg
        assert "replace" in error_msg.lower()
    
    def test_control_inf_error_message_quality(self, data_with_controls):
        """Control infinite error message should have Why/How sections."""
        data = data_with_controls.copy()
        data.loc[0, 'x1'] = np.inf
        
        with pytest.raises(InvalidParameterError) as exc_info:
            _validate_controls_dtype(data, ['x1', 'x2'])
        
        error_msg = str(exc_info.value)
        self._verify_message_sections(error_msg)


# =============================================================================
# Edge Cases and Boundary Conditions
# =============================================================================

class TestEdgeCases:
    """Edge cases for infinite value validation."""
    
    def test_single_inf_at_boundary(self, basic_panel_data):
        """Single inf at first/last row should be caught."""
        data = basic_panel_data.copy()
        data.loc[len(data) - 1, 'y'] = np.inf  # Last row
        
        with pytest.raises(InvalidParameterError):
            _validate_outcome_dtype(data, 'y')
    
    def test_inf_with_nan_mixed(self, basic_panel_data):
        """Inf and NaN mixed should catch inf (NaN handled separately)."""
        data = basic_panel_data.copy()
        data.loc[0, 'y'] = np.inf
        data.loc[1, 'y'] = np.nan
        
        with pytest.raises(InvalidParameterError) as exc_info:
            _validate_outcome_dtype(data, 'y')
        
        # Should only report 1 inf (not 2)
        error_msg = str(exc_info.value)
        assert "1 infinite" in error_msg.lower() or "1)" in error_msg
    
    def test_very_large_finite_passes(self, basic_panel_data):
        """Very large but finite values should pass."""
        data = basic_panel_data.copy()
        data.loc[0, 'y'] = 1e308  # Near max float but finite
        
        # Should not raise
        _validate_outcome_dtype(data, 'y')
    
    def test_zero_and_negative_values_pass(self, basic_panel_data):
        """Zero and negative values should pass (they are finite)."""
        data = basic_panel_data.copy()
        data.loc[0, 'y'] = 0.0
        data.loc[1, 'y'] = -1000.0
        
        # Should not raise
        _validate_outcome_dtype(data, 'y')
    
    def test_integer_dtype_no_inf(self, basic_panel_data):
        """Integer dtype cannot contain inf (only float can)."""
        data = basic_panel_data.copy()
        data['y'] = data['y'].astype(int)
        
        # Should not raise (integers cannot be inf)
        _validate_outcome_dtype(data, 'y')


# =============================================================================
# Numerical Verification Tests
# =============================================================================

class TestNumericalVerification:
    """Verify the numerical correctness of inf detection."""
    
    def test_isinf_correctness(self):
        """Verify np.isinf correctly identifies infinite values."""
        values = np.array([1.0, np.inf, -np.inf, np.nan, 0.0, -1e308])
        
        inf_mask = np.isinf(values)
        expected = np.array([False, True, True, False, False, False])
        
        np.testing.assert_array_equal(inf_mask, expected)
    
    def test_inf_count_accuracy(self):
        """Verify accurate counting of inf values."""
        data = pd.DataFrame({
            'y': [1.0, np.inf, -np.inf, np.inf, 5.0, -np.inf, 7.0]
        })
        
        y_values = data['y'].values
        n_inf = np.sum(np.isinf(y_values))
        n_pos_inf = np.sum(y_values == np.inf)
        n_neg_inf = np.sum(y_values == -np.inf)
        
        assert n_inf == 4
        assert n_pos_inf == 2
        assert n_neg_inf == 2


# =============================================================================
# Monte Carlo Test: Impact of Inf on Calculations
# =============================================================================

class TestInfImpactOnCalculations:
    """Demonstrate why inf detection is important."""
    
    def test_inf_corrupts_mean(self):
        """Show that inf corrupts mean calculation."""
        values = np.array([1.0, 2.0, 3.0, np.inf])
        
        # Mean becomes inf
        assert np.mean(values) == np.inf
    
    def test_inf_corrupts_regression(self):
        """Show that inf corrupts regression results."""
        X = np.array([[1, 1], [1, 2], [1, 3], [1, np.inf]])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        
        # X'X will have inf
        XtX = X.T @ X
        assert np.any(np.isinf(XtX))
    
    def test_negative_inf_corrupts_min(self):
        """Show that -inf corrupts min calculation."""
        values = np.array([1.0, 2.0, -np.inf, 4.0])
        
        # Min becomes -inf
        assert np.min(values) == -np.inf


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

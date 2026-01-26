"""
Tests for BUG-117/BUG-164: alpha parameter validation in estimator functions.

BUG-164 Fix: Verifies that all estimator functions (estimate_ra, estimate_ipw,
estimate_ipwra, estimate_psm) properly validate the alpha parameter to be in
the open interval (0, 1). Invalid alpha values (0, 1, negative, or > 1) should
raise ValueError.

The validation is implemented directly in each estimator function.
"""

import pytest
import numpy as np
import pandas as pd

from lwdid.staggered.estimators import (
    estimate_ra,
    estimate_ipw,
    estimate_ipwra,
    estimate_psm,
)


class TestEstimateRAAlphaValidation:
    """Test alpha validation in estimate_ra function."""
    
    @pytest.fixture
    def simple_data(self):
        """Create simple cross-sectional data for RA testing."""
        np.random.seed(42)
        n = 100
        return pd.DataFrame({
            'y': np.random.randn(n),
            'd': np.array([0] * 50 + [1] * 50),
            'x1': np.random.randn(n),
            'x2': np.random.randn(n),
        })
    
    def test_estimate_ra_alpha_zero_raises(self, simple_data):
        """estimate_ra with alpha=0 should raise ValueError."""
        with pytest.raises(ValueError, match="alpha must be in range"):
            estimate_ra(simple_data, y='y', d='d', controls=['x1', 'x2'], alpha=0)
    
    def test_estimate_ra_alpha_one_raises(self, simple_data):
        """estimate_ra with alpha=1 should raise ValueError."""
        with pytest.raises(ValueError, match="alpha must be in range"):
            estimate_ra(simple_data, y='y', d='d', controls=['x1', 'x2'], alpha=1)
    
    def test_estimate_ra_alpha_negative_raises(self, simple_data):
        """estimate_ra with negative alpha should raise ValueError."""
        with pytest.raises(ValueError, match="alpha must be in range"):
            estimate_ra(simple_data, y='y', d='d', controls=['x1', 'x2'], alpha=-0.05)
    
    def test_estimate_ra_alpha_valid(self, simple_data):
        """estimate_ra with valid alpha should work."""
        result = estimate_ra(simple_data, y='y', d='d', controls=['x1', 'x2'], alpha=0.05)
        assert result is not None
        assert hasattr(result, 'att')


class TestEstimateIPWAlphaValidation:
    """Test alpha validation in estimate_ipw function."""
    
    @pytest.fixture
    def simple_data(self):
        """Create simple cross-sectional data for IPW testing."""
        np.random.seed(42)
        n = 100
        return pd.DataFrame({
            'y': np.random.randn(n),
            'd': np.array([0] * 50 + [1] * 50),
            'x1': np.random.randn(n),
            'x2': np.random.randn(n),
        })
    
    def test_estimate_ipw_alpha_zero_raises(self, simple_data):
        """estimate_ipw with alpha=0 should raise ValueError."""
        with pytest.raises(ValueError, match="alpha must be in range"):
            estimate_ipw(simple_data, y='y', d='d', propensity_controls=['x1', 'x2'], alpha=0)
    
    def test_estimate_ipw_alpha_one_raises(self, simple_data):
        """estimate_ipw with alpha=1 should raise ValueError."""
        with pytest.raises(ValueError, match="alpha must be in range"):
            estimate_ipw(simple_data, y='y', d='d', propensity_controls=['x1', 'x2'], alpha=1)
    
    def test_estimate_ipw_alpha_negative_raises(self, simple_data):
        """estimate_ipw with negative alpha should raise ValueError."""
        with pytest.raises(ValueError, match="alpha must be in range"):
            estimate_ipw(simple_data, y='y', d='d', propensity_controls=['x1', 'x2'], alpha=-0.05)
    
    def test_estimate_ipw_alpha_valid(self, simple_data):
        """estimate_ipw with valid alpha should work."""
        result = estimate_ipw(simple_data, y='y', d='d', propensity_controls=['x1', 'x2'], alpha=0.05)
        assert result is not None
        assert hasattr(result, 'att')


class TestEstimateIPWRAAlphaValidation:
    """Test alpha validation in estimate_ipwra function."""
    
    @pytest.fixture
    def simple_data(self):
        """Create simple cross-sectional data for IPWRA testing."""
        np.random.seed(42)
        n = 100
        return pd.DataFrame({
            'y': np.random.randn(n),
            'd': np.array([0] * 50 + [1] * 50),
            'x1': np.random.randn(n),
            'x2': np.random.randn(n),
        })
    
    def test_estimate_ipwra_alpha_zero_raises(self, simple_data):
        """estimate_ipwra with alpha=0 should raise ValueError."""
        with pytest.raises(ValueError, match="alpha must be in range"):
            estimate_ipwra(simple_data, y='y', d='d', controls=['x1', 'x2'], alpha=0)
    
    def test_estimate_ipwra_alpha_one_raises(self, simple_data):
        """estimate_ipwra with alpha=1 should raise ValueError."""
        with pytest.raises(ValueError, match="alpha must be in range"):
            estimate_ipwra(simple_data, y='y', d='d', controls=['x1', 'x2'], alpha=1)
    
    def test_estimate_ipwra_alpha_negative_raises(self, simple_data):
        """estimate_ipwra with negative alpha should raise ValueError."""
        with pytest.raises(ValueError, match="alpha must be in range"):
            estimate_ipwra(simple_data, y='y', d='d', controls=['x1', 'x2'], alpha=-0.05)
    
    def test_estimate_ipwra_alpha_valid(self, simple_data):
        """estimate_ipwra with valid alpha should work."""
        result = estimate_ipwra(simple_data, y='y', d='d', controls=['x1', 'x2'], alpha=0.05)
        assert result is not None
        assert hasattr(result, 'att')


class TestEstimatePSMAlphaValidation:
    """Test alpha validation in estimate_psm function."""
    
    @pytest.fixture
    def simple_data(self):
        """Create simple cross-sectional data for PSM testing."""
        np.random.seed(42)
        n = 100
        return pd.DataFrame({
            'y': np.random.randn(n),
            'd': np.array([0] * 50 + [1] * 50),
            'x1': np.random.randn(n),
            'x2': np.random.randn(n),
        })
    
    def test_estimate_psm_alpha_zero_raises(self, simple_data):
        """estimate_psm with alpha=0 should raise ValueError."""
        with pytest.raises(ValueError, match="alpha must be in range"):
            estimate_psm(simple_data, y='y', d='d', propensity_controls=['x1', 'x2'], alpha=0)
    
    def test_estimate_psm_alpha_one_raises(self, simple_data):
        """estimate_psm with alpha=1 should raise ValueError."""
        with pytest.raises(ValueError, match="alpha must be in range"):
            estimate_psm(simple_data, y='y', d='d', propensity_controls=['x1', 'x2'], alpha=1)
    
    def test_estimate_psm_alpha_negative_raises(self, simple_data):
        """estimate_psm with negative alpha should raise ValueError."""
        with pytest.raises(ValueError, match="alpha must be in range"):
            estimate_psm(simple_data, y='y', d='d', propensity_controls=['x1', 'x2'], alpha=-0.05)
    
    def test_estimate_psm_alpha_valid(self, simple_data):
        """estimate_psm with valid alpha should work."""
        result = estimate_psm(simple_data, y='y', d='d', propensity_controls=['x1', 'x2'], alpha=0.05)
        assert result is not None
        assert hasattr(result, 'att')


class TestAlphaBoundaryBehavior:
    """Test behavior at boundary values and edge cases."""
    
    @pytest.fixture
    def simple_data(self):
        """Create simple cross-sectional data for testing."""
        np.random.seed(42)
        n = 100
        return pd.DataFrame({
            'y': np.random.randn(n),
            'd': np.array([0] * 50 + [1] * 50),
            'x1': np.random.randn(n),
            'x2': np.random.randn(n),
        })
    
    def test_alpha_very_small_is_valid(self, simple_data):
        """Test alpha very close to 0 is valid."""
        result = estimate_ra(simple_data, y='y', d='d', controls=['x1', 'x2'], alpha=1e-10)
        assert result is not None
    
    def test_alpha_very_large_is_valid(self, simple_data):
        """Test alpha very close to 1 is valid."""
        result = estimate_ra(simple_data, y='y', d='d', controls=['x1', 'x2'], alpha=1 - 1e-10)
        assert result is not None
    
    def test_alpha_exactly_zero_is_invalid(self, simple_data):
        """Test alpha exactly 0 is invalid."""
        with pytest.raises(ValueError, match="alpha must be in range"):
            estimate_ra(simple_data, y='y', d='d', controls=['x1', 'x2'], alpha=0.0)
    
    def test_alpha_exactly_one_is_invalid(self, simple_data):
        """Test alpha exactly 1 is invalid."""
        with pytest.raises(ValueError, match="alpha must be in range"):
            estimate_ra(simple_data, y='y', d='d', controls=['x1', 'x2'], alpha=1.0)
    
    def test_error_message_contains_value(self, simple_data):
        """Test error message includes the invalid alpha value."""
        with pytest.raises(ValueError, match="got 0"):
            estimate_ra(simple_data, y='y', d='d', controls=['x1', 'x2'], alpha=0)
        
        with pytest.raises(ValueError, match="got 1"):
            estimate_ra(simple_data, y='y', d='d', controls=['x1', 'x2'], alpha=1)
        
        with pytest.raises(ValueError, match="got -0.5"):
            estimate_ra(simple_data, y='y', d='d', controls=['x1', 'x2'], alpha=-0.5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

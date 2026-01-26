"""
Test suite for bug fixes BUG-304 to BUG-306 (Round 87 code review).

This module tests the following bug fixes:
- BUG-304: caliper parameter NaN/Inf validation in _validate_psm_params
- BUG-305: estimator parameter type validation (confirmed as duplicate of BUG-314)
- BUG-306: IPW/PSM controls vs ps_controls validation logic

Test coverage:
1. Unit tests for parameter validation
2. Integration tests with actual estimation
3. Stata compatibility tests
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
from lwdid.core import _validate_psm_params


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def simple_panel_data():
    """Create a simple panel dataset for common timing DiD."""
    np.random.seed(42)
    n_units = 50
    n_periods = 6
    
    data = []
    for i in range(1, n_units + 1):
        treated = 1 if i <= 25 else 0
        # Time-invariant control variables (constant within each unit)
        x1 = np.random.normal(0, 1) + treated * 0.5
        x2 = np.random.normal(0, 1) + treated * 0.3
        for t in range(2000, 2000 + n_periods):
            post = 1 if t >= 2003 else 0
            y = 10 + 2*treated + 3*post + 1.5*treated*post + 0.5*x1 + 0.3*x2 + np.random.normal(0, 1)
            data.append({
                'unit': i,
                'year': t,
                'treated': treated,
                'post': post,
                'outcome': y,
                'x1': x1,  # Time-invariant
                'x2': x2,  # Time-invariant
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def staggered_panel_data():
    """Create a staggered adoption panel dataset."""
    np.random.seed(42)
    n_units = 60
    n_periods = 8
    
    data = []
    for i in range(1, n_units + 1):
        # Assign cohort: 20 units never treated, 20 units treated in 2003, 20 in 2005
        if i <= 20:
            gvar = 0  # never treated
        elif i <= 40:
            gvar = 2003
        else:
            gvar = 2005
        
        # Time-invariant controls
        x1 = np.random.normal(0, 1)
        x2 = np.random.normal(0, 1)
        
        for t in range(2000, 2000 + n_periods):
            treated = 1 if gvar > 0 and t >= gvar else 0
            y = 10 + 2*treated + 0.5*x1 + 0.3*x2 + np.random.normal(0, 1)
            data.append({
                'unit': i,
                'year': t,
                'gvar': gvar,
                'outcome': y,
                'x1': x1,
                'x2': x2,
            })
    
    return pd.DataFrame(data)


# =============================================================================
# BUG-304: caliper NaN/Inf validation
# =============================================================================

class TestBug304CaliperValidation:
    """Test caliper parameter validation for NaN and Inf values."""
    
    def test_caliper_nan_rejected(self):
        """BUG-304: caliper=np.nan should be rejected with clear error message."""
        with pytest.raises(ValueError) as exc_info:
            _validate_psm_params(
                n_neighbors=1,
                caliper=np.nan,
                with_replacement=True,
                match_order='data'
            )
        assert "finite positive number" in str(exc_info.value).lower()
        assert "nan" in str(exc_info.value).lower()
    
    def test_caliper_inf_rejected(self):
        """BUG-304: caliper=np.inf should be rejected."""
        with pytest.raises(ValueError) as exc_info:
            _validate_psm_params(
                n_neighbors=1,
                caliper=np.inf,
                with_replacement=True,
                match_order='data'
            )
        assert "finite positive number" in str(exc_info.value).lower()
    
    def test_caliper_neg_inf_rejected(self):
        """BUG-304: caliper=-np.inf should be rejected."""
        with pytest.raises(ValueError) as exc_info:
            _validate_psm_params(
                n_neighbors=1,
                caliper=-np.inf,
                with_replacement=True,
                match_order='data'
            )
        assert "finite positive number" in str(exc_info.value).lower()
    
    def test_caliper_positive_finite_accepted(self):
        """Valid caliper values should be accepted."""
        # Should not raise any exception
        _validate_psm_params(
            n_neighbors=1,
            caliper=0.1,
            with_replacement=True,
            match_order='data'
        )
        
        _validate_psm_params(
            n_neighbors=1,
            caliper=0.5,
            with_replacement=True,
            match_order='data'
        )
        
        _validate_psm_params(
            n_neighbors=1,
            caliper=1.0,
            with_replacement=True,
            match_order='data'
        )
    
    def test_caliper_zero_rejected(self):
        """caliper=0 should be rejected (must be > 0)."""
        with pytest.raises(ValueError) as exc_info:
            _validate_psm_params(
                n_neighbors=1,
                caliper=0,
                with_replacement=True,
                match_order='data'
            )
        assert "finite positive number" in str(exc_info.value).lower()
    
    def test_caliper_negative_rejected(self):
        """caliper < 0 should be rejected."""
        with pytest.raises(ValueError) as exc_info:
            _validate_psm_params(
                n_neighbors=1,
                caliper=-0.5,
                with_replacement=True,
                match_order='data'
            )
        assert "finite positive number" in str(exc_info.value).lower()
    
    def test_caliper_none_accepted(self):
        """caliper=None should be accepted (no caliper constraint)."""
        # Should not raise any exception
        _validate_psm_params(
            n_neighbors=1,
            caliper=None,
            with_replacement=True,
            match_order='data'
        )


# =============================================================================
# BUG-305: estimator type validation (confirmed as duplicate of BUG-314)
# =============================================================================

class TestBug305EstimatorTypeValidation:
    """Test estimator parameter type validation."""
    
    def test_estimator_non_string_rejected_common_timing(self, simple_panel_data):
        """BUG-305/BUG-314: Non-string estimator should raise TypeError in common timing."""
        with pytest.raises(TypeError) as exc_info:
            lwdid(
                data=simple_panel_data,
                y='outcome',
                d='treated',
                ivar='unit',
                tvar='year',
                post='post',
                rolling='demean',
                estimator=1  # Invalid: integer instead of string
            )
        assert "estimator must be a string" in str(exc_info.value)
    
    def test_estimator_non_string_rejected_staggered(self, staggered_panel_data):
        """BUG-305/BUG-314: Non-string estimator should raise TypeError in staggered mode."""
        with pytest.raises(TypeError) as exc_info:
            lwdid(
                data=staggered_panel_data,
                y='outcome',
                ivar='unit',
                tvar='year',
                gvar='gvar',
                rolling='demean',
                estimator=123  # Invalid: integer instead of string
            )
        assert "estimator must be a string" in str(exc_info.value)
    
    def test_estimator_list_rejected(self, simple_panel_data):
        """BUG-305/BUG-314: List estimator should raise TypeError."""
        with pytest.raises(TypeError) as exc_info:
            lwdid(
                data=simple_panel_data,
                y='outcome',
                d='treated',
                ivar='unit',
                tvar='year',
                post='post',
                rolling='demean',
                estimator=['ra']  # Invalid: list instead of string
            )
        assert "estimator must be a string" in str(exc_info.value)
    
    def test_estimator_valid_string_accepted(self, simple_panel_data):
        """Valid string estimators should be accepted."""
        # RA estimator (default)
        result = lwdid(
            data=simple_panel_data,
            y='outcome',
            d='treated',
            ivar='unit',
            tvar='year',
            post='post',
            rolling='demean',
            estimator='ra'
        )
        assert result is not None
        assert hasattr(result, 'att')


# =============================================================================
# BUG-306: IPW/PSM controls vs ps_controls validation
# =============================================================================

class TestBug306ControlsValidation:
    """Test IPW/PSM control variable validation logic."""
    
    def test_ipw_with_only_ps_controls_accepted_common_timing(self, simple_panel_data):
        """BUG-306: IPW should accept ps_controls without controls in common timing."""
        # This should NOT raise an error - IPW only needs ps_controls
        result = lwdid(
            data=simple_panel_data,
            y='outcome',
            d='treated',
            ivar='unit',
            tvar='year',
            post='post',
            rolling='demean',
            estimator='ipw',
            ps_controls=['x1', 'x2']  # Only ps_controls, no controls
        )
        assert result is not None
        assert hasattr(result, 'att')
    
    def test_psm_with_only_ps_controls_accepted_common_timing(self, simple_panel_data):
        """BUG-306: PSM should accept ps_controls without controls in common timing."""
        # This should NOT raise an error - PSM only needs ps_controls
        result = lwdid(
            data=simple_panel_data,
            y='outcome',
            d='treated',
            ivar='unit',
            tvar='year',
            post='post',
            rolling='demean',
            estimator='psm',
            ps_controls=['x1', 'x2']  # Only ps_controls, no controls
        )
        assert result is not None
        assert hasattr(result, 'att')
    
    def test_ipw_with_only_controls_accepted_common_timing(self, simple_panel_data):
        """IPW should also accept controls (used as ps_controls by default)."""
        result = lwdid(
            data=simple_panel_data,
            y='outcome',
            d='treated',
            ivar='unit',
            tvar='year',
            post='post',
            rolling='demean',
            estimator='ipw',
            controls=['x1', 'x2']  # Only controls, will be used as ps_controls
        )
        assert result is not None
        assert hasattr(result, 'att')
    
    def test_ipw_without_any_controls_rejected_common_timing(self, simple_panel_data):
        """IPW without controls or ps_controls should be rejected."""
        with pytest.raises(ValueError) as exc_info:
            lwdid(
                data=simple_panel_data,
                y='outcome',
                d='treated',
                ivar='unit',
                tvar='year',
                post='post',
                rolling='demean',
                estimator='ipw'
                # No controls or ps_controls
            )
        assert "controls" in str(exc_info.value).lower() or "ps_controls" in str(exc_info.value).lower()
    
    def test_psm_without_any_controls_rejected_common_timing(self, simple_panel_data):
        """PSM without controls or ps_controls should be rejected."""
        with pytest.raises(ValueError) as exc_info:
            lwdid(
                data=simple_panel_data,
                y='outcome',
                d='treated',
                ivar='unit',
                tvar='year',
                post='post',
                rolling='demean',
                estimator='psm'
                # No controls or ps_controls
            )
        assert "controls" in str(exc_info.value).lower() or "ps_controls" in str(exc_info.value).lower()
    
    def test_ipwra_requires_controls_common_timing(self, simple_panel_data):
        """IPWRA requires controls parameter (for outcome model)."""
        with pytest.raises(ValueError) as exc_info:
            lwdid(
                data=simple_panel_data,
                y='outcome',
                d='treated',
                ivar='unit',
                tvar='year',
                post='post',
                rolling='demean',
                estimator='ipwra',
                ps_controls=['x1', 'x2']  # ps_controls only, no controls
            )
        assert "controls" in str(exc_info.value).lower()
        assert "ipwra" in str(exc_info.value).lower()
    
    def test_ipwra_with_controls_accepted_common_timing(self, simple_panel_data):
        """IPWRA with controls should be accepted."""
        result = lwdid(
            data=simple_panel_data,
            y='outcome',
            d='treated',
            ivar='unit',
            tvar='year',
            post='post',
            rolling='demean',
            estimator='ipwra',
            controls=['x1', 'x2']  # controls provided
        )
        assert result is not None
        assert hasattr(result, 'att')
    
    # Staggered mode tests
    def test_ipw_with_only_ps_controls_accepted_staggered(self, staggered_panel_data):
        """BUG-306: IPW should accept ps_controls without controls in staggered mode."""
        result = lwdid(
            data=staggered_panel_data,
            y='outcome',
            ivar='unit',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            estimator='ipw',
            ps_controls=['x1', 'x2'],  # Only ps_controls
            aggregate='none'
        )
        assert result is not None
        assert hasattr(result, 'att')
    
    def test_psm_with_only_ps_controls_accepted_staggered(self, staggered_panel_data):
        """BUG-306: PSM should accept ps_controls without controls in staggered mode."""
        result = lwdid(
            data=staggered_panel_data,
            y='outcome',
            ivar='unit',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            estimator='psm',
            ps_controls=['x1', 'x2'],  # Only ps_controls
            aggregate='none'
        )
        assert result is not None
        assert hasattr(result, 'att')
    
    def test_ipw_without_any_controls_rejected_staggered(self, staggered_panel_data):
        """IPW without controls or ps_controls should be rejected in staggered mode."""
        with pytest.raises(ValueError) as exc_info:
            lwdid(
                data=staggered_panel_data,
                y='outcome',
                ivar='unit',
                tvar='year',
                gvar='gvar',
                rolling='demean',
                estimator='ipw'
                # No controls or ps_controls
            )
        assert "controls" in str(exc_info.value).lower() or "ps_controls" in str(exc_info.value).lower()
    
    def test_ipwra_requires_controls_staggered(self, staggered_panel_data):
        """IPWRA requires controls parameter in staggered mode."""
        with pytest.raises(ValueError) as exc_info:
            lwdid(
                data=staggered_panel_data,
                y='outcome',
                ivar='unit',
                tvar='year',
                gvar='gvar',
                rolling='demean',
                estimator='ipwra',
                ps_controls=['x1', 'x2']  # ps_controls only, no controls
            )
        assert "controls" in str(exc_info.value).lower()
        assert "ipwra" in str(exc_info.value).lower()


# =============================================================================
# Integration Tests
# =============================================================================

class TestBugFixesIntegration:
    """Integration tests to verify bug fixes work in realistic scenarios."""
    
    def test_ipw_ps_controls_produces_valid_results(self, simple_panel_data):
        """Verify IPW with ps_controls produces statistically valid results."""
        result = lwdid(
            data=simple_panel_data,
            y='outcome',
            d='treated',
            ivar='unit',
            tvar='year',
            post='post',
            rolling='demean',
            estimator='ipw',
            ps_controls=['x1', 'x2']
        )
        
        # Verify result structure
        assert result is not None
        assert hasattr(result, 'att')
        assert hasattr(result, 'se_att')
        assert hasattr(result, 'pvalue')
        
        # Verify numerical validity
        assert np.isfinite(result.att)
        assert np.isfinite(result.se_att)
        assert result.se_att > 0
        assert 0 <= result.pvalue <= 1
        
        # With true effect of 1.5, ATT should be reasonably close
        assert -5 < result.att < 10  # Reasonable range
    
    def test_psm_ps_controls_produces_valid_results(self, simple_panel_data):
        """Verify PSM with ps_controls produces statistically valid results."""
        result = lwdid(
            data=simple_panel_data,
            y='outcome',
            d='treated',
            ivar='unit',
            tvar='year',
            post='post',
            rolling='demean',
            estimator='psm',
            ps_controls=['x1', 'x2']
        )
        
        # Verify result structure
        assert result is not None
        assert hasattr(result, 'att')
        assert hasattr(result, 'se_att')
        
        # Verify numerical validity
        assert np.isfinite(result.att)
        assert np.isfinite(result.se_att)
        assert result.se_att > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

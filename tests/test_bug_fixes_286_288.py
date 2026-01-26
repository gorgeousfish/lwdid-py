"""
Test suite for bug fixes BUG-286 to BUG-288 (Round 85-86 code review).

This module tests the following bug fixes:
- BUG-286: results.py summary_staggered t_stat handling for SE=0/Inf
- BUG-287: core.py ri_method parameter string type validation
- BUG-288: core.py trim_threshold range validation in common timing mode
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
from lwdid.results import _compute_t_stat, LWDIDResults


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def common_timing_data():
    """Create a common timing panel dataset for testing."""
    np.random.seed(42)
    n_units = 40
    n_periods = 6
    
    data = []
    for i in range(1, n_units + 1):
        # Half treated, half control
        d = 1 if i <= 20 else 0
        # Time-invariant controls (constant within unit)
        x1_unit = np.random.normal(0, 1)
        x2_unit = np.random.normal(0, 1)
        
        for t in range(2000, 2000 + n_periods):
            post = 1 if t >= 2003 else 0
            treated = d * post
            y = 10 + 2 * treated + 0.5 * x1_unit + np.random.normal(0, 1)
            data.append({
                'unit': i,
                'year': t,
                'd': d,
                'post': post,
                'y': y,
                'x1': x1_unit,  # Time-invariant
                'x2': x2_unit,  # Time-invariant
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def staggered_panel_data():
    """Create a staggered adoption panel dataset for testing."""
    np.random.seed(42)
    n_units = 30
    n_periods = 6
    
    data = []
    for i in range(1, n_units + 1):
        # Assign cohort: 10 units never treated, 10 treated in 2003, 10 in 2005
        if i <= 10:
            gvar = 0  # never treated
        elif i <= 20:
            gvar = 2003
        else:
            gvar = 2005
        
        # Time-invariant controls (constant within unit)
        x1_unit = np.random.normal(0, 1)
        x2_unit = np.random.normal(0, 1)
            
        for t in range(2000, 2000 + n_periods):
            treated = 1 if gvar > 0 and t >= gvar else 0
            y = 10 + 2 * treated + 0.5 * x1_unit + np.random.normal(0, 1)
            data.append({
                'unit': i,
                'year': t,
                'gvar': gvar,
                'y': y,
                'x1': x1_unit,  # Time-invariant
                'x2': x2_unit,  # Time-invariant
            })
    
    return pd.DataFrame(data)


# =============================================================================
# BUG-286: results.py t_stat handling for SE=0/Inf
# =============================================================================

class TestBug286TStatBoundaryHandling:
    """Test that t_stat computation handles SE=0 and SE=Inf correctly."""
    
    def test_compute_t_stat_normal_case(self):
        """Test normal t_stat computation."""
        t_stat = _compute_t_stat(att=2.0, se=0.5, warn_on_boundary=False)
        assert t_stat == 4.0
    
    def test_compute_t_stat_se_zero_returns_nan(self):
        """Test that SE=0 returns NaN with warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            t_stat = _compute_t_stat(att=2.0, se=0.0, warn_on_boundary=True)
            
            assert np.isnan(t_stat)
            assert len(w) == 1
            assert "zero" in str(w[0].message).lower()
            assert "undefined" in str(w[0].message).lower()
    
    def test_compute_t_stat_se_inf_returns_zero(self):
        """Test that SE=Inf returns 0.0 with warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            t_stat = _compute_t_stat(att=2.0, se=np.inf, warn_on_boundary=True)
            
            assert t_stat == 0.0
            assert len(w) == 1
            assert "infinite" in str(w[0].message).lower()
    
    def test_compute_t_stat_se_negative_returns_nan(self):
        """Test that negative SE returns NaN with warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            t_stat = _compute_t_stat(att=2.0, se=-0.5, warn_on_boundary=True)
            
            assert np.isnan(t_stat)
            assert len(w) == 1
            assert "negative" in str(w[0].message).lower()
    
    def test_compute_t_stat_att_nan_returns_nan(self):
        """Test that ATT=NaN returns NaN without warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            t_stat = _compute_t_stat(att=np.nan, se=0.5, warn_on_boundary=True)
            
            assert np.isnan(t_stat)
            # No warning for NaN input (expected behavior)
            assert len(w) == 0
    
    def test_compute_t_stat_se_nan_returns_nan(self):
        """Test that SE=NaN returns NaN without warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            t_stat = _compute_t_stat(att=2.0, se=np.nan, warn_on_boundary=True)
            
            assert np.isnan(t_stat)
            assert len(w) == 0
    
    def test_compute_t_stat_no_warning_when_disabled(self):
        """Test that warnings can be disabled."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _compute_t_stat(att=2.0, se=0.0, warn_on_boundary=False)
            _compute_t_stat(att=2.0, se=np.inf, warn_on_boundary=False)
            
            # No warnings when disabled
            assert len(w) == 0
    
    def test_compute_t_stat_cohort_in_warning_message(self):
        """Test that cohort identifier appears in warning message."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _compute_t_stat(att=2.0, se=0.0, cohort=2003, warn_on_boundary=True)
            
            assert "cohort 2003" in str(w[0].message)


# =============================================================================
# BUG-287: core.py ri_method parameter type validation
# =============================================================================

class TestBug287RiMethodTypeValidation:
    """Test that ri_method parameter validates string type."""
    
    def test_ri_method_integer_raises_type_error_common_timing(self, common_timing_data):
        """Test that ri_method=123 raises TypeError in common timing mode."""
        with pytest.raises(TypeError) as excinfo:
            lwdid(
                data=common_timing_data,
                y='y',
                d='d',
                ivar='unit',
                tvar='year',
                post='post',
                rolling='demean',
                ri=True,
                rireps=10,
                ri_method=123,  # Invalid: should be string
            )
        
        assert "ri_method" in str(excinfo.value)
        assert "string" in str(excinfo.value).lower()
        assert "int" in str(excinfo.value).lower()
    
    def test_ri_method_list_raises_type_error_common_timing(self, common_timing_data):
        """Test that ri_method=['bootstrap'] raises TypeError."""
        with pytest.raises(TypeError) as excinfo:
            lwdid(
                data=common_timing_data,
                y='y',
                d='d',
                ivar='unit',
                tvar='year',
                post='post',
                rolling='demean',
                ri=True,
                rireps=10,
                ri_method=['bootstrap'],  # Invalid: should be string
            )
        
        assert "ri_method" in str(excinfo.value)
        assert "list" in str(excinfo.value).lower()
    
    def test_ri_method_integer_raises_type_error_staggered(self, staggered_panel_data):
        """Test that ri_method=123 raises TypeError in staggered mode."""
        with pytest.raises(TypeError) as excinfo:
            lwdid(
                data=staggered_panel_data,
                y='y',
                gvar='gvar',
                ivar='unit',
                tvar='year',
                rolling='demean',
                ri=True,
                rireps=10,
                ri_method=123,  # Invalid: should be string
            )
        
        assert "ri_method" in str(excinfo.value)
        assert "string" in str(excinfo.value).lower()
    
    def test_ri_method_valid_string_works_common_timing(self, common_timing_data):
        """Test that valid ri_method string works."""
        # Should not raise
        result = lwdid(
            data=common_timing_data,
            y='y',
            d='d',
            ivar='unit',
            tvar='year',
            post='post',
            rolling='demean',
            ri=True,
            rireps=10,
            ri_method='bootstrap',
            seed=42,
        )
        assert result is not None
    
    def test_ri_method_none_uses_default(self, common_timing_data):
        """Test that ri_method=None uses default 'bootstrap'."""
        # Should not raise
        result = lwdid(
            data=common_timing_data,
            y='y',
            d='d',
            ivar='unit',
            tvar='year',
            post='post',
            rolling='demean',
            ri=True,
            rireps=10,
            ri_method=None,  # Should default to 'bootstrap'
            seed=42,
        )
        assert result is not None


# =============================================================================
# BUG-288: core.py trim_threshold range validation in common timing mode
# =============================================================================

class TestBug288TrimThresholdValidation:
    """Test that trim_threshold range is validated in common timing mode."""
    
    def test_trim_threshold_negative_raises_error(self, common_timing_data):
        """Test that negative trim_threshold raises ValueError."""
        with pytest.raises(ValueError) as excinfo:
            lwdid(
                data=common_timing_data,
                y='y',
                d='d',
                ivar='unit',
                tvar='year',
                post='post',
                rolling='demean',
                estimator='ipw',
                controls=['x1', 'x2'],
                trim_threshold=-0.1,  # Invalid: must be > 0
            )
        
        assert "trim_threshold" in str(excinfo.value)
        assert "-0.1" in str(excinfo.value)
    
    def test_trim_threshold_zero_raises_error(self, common_timing_data):
        """Test that trim_threshold=0 raises ValueError."""
        with pytest.raises(ValueError) as excinfo:
            lwdid(
                data=common_timing_data,
                y='y',
                d='d',
                ivar='unit',
                tvar='year',
                post='post',
                rolling='demean',
                estimator='ipw',
                controls=['x1', 'x2'],
                trim_threshold=0.0,  # Invalid: must be > 0
            )
        
        assert "trim_threshold" in str(excinfo.value)
    
    def test_trim_threshold_greater_than_half_raises_error(self, common_timing_data):
        """Test that trim_threshold > 0.5 raises ValueError."""
        with pytest.raises(ValueError) as excinfo:
            lwdid(
                data=common_timing_data,
                y='y',
                d='d',
                ivar='unit',
                tvar='year',
                post='post',
                rolling='demean',
                estimator='ipw',
                controls=['x1', 'x2'],
                trim_threshold=0.6,  # Invalid: must be < 0.5
            )
        
        assert "trim_threshold" in str(excinfo.value)
        assert "0.6" in str(excinfo.value)
    
    def test_trim_threshold_exactly_half_raises_error(self, common_timing_data):
        """Test that trim_threshold=0.5 raises ValueError (exclusive boundary)."""
        with pytest.raises(ValueError) as excinfo:
            lwdid(
                data=common_timing_data,
                y='y',
                d='d',
                ivar='unit',
                tvar='year',
                post='post',
                rolling='demean',
                estimator='ipw',
                controls=['x1', 'x2'],
                trim_threshold=0.5,  # Invalid: boundary is exclusive
            )
        
        assert "trim_threshold" in str(excinfo.value)
        assert "0.5" in str(excinfo.value)
    
    def test_trim_threshold_valid_values_work(self, common_timing_data):
        """Test that valid trim_threshold values work."""
        # Test near-boundary value (0.5 is exclusive, so use 0.49)
        result = lwdid(
            data=common_timing_data,
            y='y',
            d='d',
            ivar='unit',
            tvar='year',
            post='post',
            rolling='demean',
            estimator='ipw',
            controls=['x1', 'x2'],
            trim_threshold=0.49,  # Valid: just below 0.5 boundary
        )
        assert result is not None
        
        # Test typical value
        result = lwdid(
            data=common_timing_data,
            y='y',
            d='d',
            ivar='unit',
            tvar='year',
            post='post',
            rolling='demean',
            estimator='ipw',
            controls=['x1', 'x2'],
            trim_threshold=0.05,  # Valid: typical value
        )
        assert result is not None
    
    def test_trim_threshold_not_validated_for_ra_estimator(self, common_timing_data):
        """Test that trim_threshold validation is skipped for RA estimator."""
        # RA estimator doesn't use propensity scores, so invalid trim_threshold
        # should be ignored (with a warning), not raise an error
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = lwdid(
                data=common_timing_data,
                y='y',
                d='d',
                ivar='unit',
                tvar='year',
                post='post',
                rolling='demean',
                estimator='ra',  # RA doesn't use trim_threshold
                trim_threshold=0.6,  # Would be invalid for IPW
            )
        
        # Should complete successfully (RA ignores trim_threshold)
        assert result is not None
        # Should warn about ignored parameter
        assert any("trim_threshold" in str(w_i.message) for w_i in w)
    
    def test_trim_threshold_validated_for_ipwra(self, common_timing_data):
        """Test that trim_threshold is validated for IPWRA estimator."""
        with pytest.raises(ValueError) as excinfo:
            lwdid(
                data=common_timing_data,
                y='y',
                d='d',
                ivar='unit',
                tvar='year',
                post='post',
                rolling='demean',
                estimator='ipwra',
                controls=['x1', 'x2'],
                trim_threshold=-0.1,  # Invalid
            )
        
        assert "trim_threshold" in str(excinfo.value)
    
    def test_trim_threshold_validated_for_psm(self, common_timing_data):
        """Test that trim_threshold is validated for PSM estimator."""
        with pytest.raises(ValueError) as excinfo:
            lwdid(
                data=common_timing_data,
                y='y',
                d='d',
                ivar='unit',
                tvar='year',
                post='post',
                rolling='demean',
                estimator='psm',
                controls=['x1', 'x2'],
                trim_threshold=0.7,  # Invalid
            )
        
        assert "trim_threshold" in str(excinfo.value)


# =============================================================================
# Integration Tests
# =============================================================================

class TestBugFixesIntegration:
    """Integration tests ensuring all bug fixes work together."""
    
    def test_all_fixes_common_timing(self, common_timing_data):
        """Test that all bug fixes work correctly in common timing mode."""
        # Run a complete estimation to ensure no regressions
        result = lwdid(
            data=common_timing_data,
            y='y',
            d='d',
            ivar='unit',
            tvar='year',
            post='post',
            rolling='demean',
            estimator='ipw',
            controls=['x1', 'x2'],
            trim_threshold=0.05,  # Valid value (BUG-288)
        )
        
        assert result is not None
        # Common timing mode uses 'att' and 'se_att' attributes
        assert result.att is not None
        assert result.se_att is not None
    
    def test_all_fixes_staggered(self, staggered_panel_data):
        """Test that all bug fixes work correctly in staggered mode."""
        # Run a complete estimation to ensure no regressions
        result = lwdid(
            data=staggered_panel_data,
            y='y',
            gvar='gvar',
            ivar='unit',
            tvar='year',
            rolling='demean',
            aggregate='overall',
        )
        
        assert result is not None
        assert result.att_overall is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

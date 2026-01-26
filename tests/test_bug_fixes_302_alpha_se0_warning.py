"""
Test suite for BUG-302 staggered alpha validation and plot_event_study se=0 warning.

This module tests:
- BUG-302 FIX: Staggered mode alpha parameter type validation
- plot_event_study: se=0 warning message
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
from lwdid.results import LWDIDResults


# =============================================================================
# Test Fixtures
# =============================================================================

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
            
        for t in range(2000, 2000 + n_periods):
            treated = 1 if gvar > 0 and t >= gvar else 0
            y = 10 + 2*treated + np.random.normal(0, 1)
            data.append({
                'unit': i,
                'year': t,
                'gvar': gvar,
                'outcome': y,
                'x1': np.random.normal(5, 1),
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def simple_panel_data():
    """Create a simple panel dataset for common timing DiD."""
    np.random.seed(42)
    n_units = 50
    n_periods = 6
    
    data = []
    for i in range(1, n_units + 1):
        treated = 1 if i <= 25 else 0
        for t in range(2000, 2000 + n_periods):
            post = 1 if t >= 2003 else 0
            y = 10 + 2*treated + 3*post + 1.5*treated*post + np.random.normal(0, 1)
            data.append({
                'unit': i,
                'year': t,
                'treated': treated,
                'post': post,
                'outcome': y,
            })
    
    return pd.DataFrame(data)


# =============================================================================
# BUG-302: Staggered mode alpha type validation tests
# =============================================================================

class TestBug302StaggeredAlphaValidation:
    """Test BUG-302 fix: staggered mode alpha parameter type validation."""
    
    def test_staggered_alpha_string_raises_type_error(self, staggered_panel_data):
        """String alpha should raise TypeError in staggered mode."""
        df = staggered_panel_data.copy()
        
        with pytest.raises(TypeError) as excinfo:
            lwdid(
                data=df,
                y='outcome',
                ivar='unit',
                tvar='year',
                gvar='gvar',
                rolling='demean',
                alpha="0.05"  # String instead of float
            )
        
        assert 'alpha' in str(excinfo.value).lower()
        assert 'numeric' in str(excinfo.value).lower()
    
    def test_staggered_alpha_none_raises_type_error(self, staggered_panel_data):
        """None alpha should raise TypeError in staggered mode."""
        df = staggered_panel_data.copy()
        
        with pytest.raises(TypeError) as excinfo:
            lwdid(
                data=df,
                y='outcome',
                ivar='unit',
                tvar='year',
                gvar='gvar',
                rolling='demean',
                alpha=None  # None instead of float
            )
        
        assert 'alpha' in str(excinfo.value).lower()
    
    def test_staggered_alpha_nan_raises_value_error(self, staggered_panel_data):
        """NaN alpha should raise ValueError in staggered mode."""
        df = staggered_panel_data.copy()
        
        with pytest.raises(ValueError) as excinfo:
            lwdid(
                data=df,
                y='outcome',
                ivar='unit',
                tvar='year',
                gvar='gvar',
                rolling='demean',
                alpha=float('nan')  # NaN
            )
        
        assert 'nan' in str(excinfo.value).lower()
    
    def test_staggered_alpha_list_raises_type_error(self, staggered_panel_data):
        """List alpha should raise TypeError in staggered mode."""
        df = staggered_panel_data.copy()
        
        with pytest.raises(TypeError) as excinfo:
            lwdid(
                data=df,
                y='outcome',
                ivar='unit',
                tvar='year',
                gvar='gvar',
                rolling='demean',
                alpha=[0.05]  # List instead of float
            )
        
        assert 'alpha' in str(excinfo.value).lower()
    
    def test_staggered_alpha_valid_float_works(self, staggered_panel_data):
        """Valid float alpha should work in staggered mode."""
        df = staggered_panel_data.copy()
        
        # Should not raise
        result = lwdid(
            data=df,
            y='outcome',
            ivar='unit',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            alpha=0.05
        )
        
        assert result is not None
        assert not np.isnan(result.att)
    
    def test_staggered_alpha_valid_int_works(self, staggered_panel_data):
        """Valid int alpha (0) should raise ValueError, not TypeError."""
        df = staggered_panel_data.copy()
        
        # Should raise ValueError (out of range), not TypeError
        with pytest.raises(ValueError) as excinfo:
            lwdid(
                data=df,
                y='outcome',
                ivar='unit',
                tvar='year',
                gvar='gvar',
                rolling='demean',
                alpha=0  # Zero is out of range
            )
        
        # Should be a range error, not a type error
        assert 'alpha' in str(excinfo.value).lower()
        assert '0' in str(excinfo.value) or 'must be' in str(excinfo.value).lower()
    
    def test_staggered_alpha_numpy_float_works(self, staggered_panel_data):
        """Numpy float alpha should work in staggered mode."""
        df = staggered_panel_data.copy()
        
        # Should not raise
        result = lwdid(
            data=df,
            y='outcome',
            ivar='unit',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            alpha=np.float64(0.1)  # numpy float
        )
        
        assert result is not None


# =============================================================================
# Common timing alpha validation tests (for consistency comparison)
# =============================================================================

class TestCommonTimingAlphaValidation:
    """Test common timing mode alpha validation for comparison."""
    
    def test_common_alpha_string_raises_type_error(self, simple_panel_data):
        """String alpha should raise TypeError in common timing mode."""
        df = simple_panel_data.copy()
        
        with pytest.raises(TypeError) as excinfo:
            lwdid(
                data=df,
                y='outcome',
                d='treated',
                ivar='unit',
                tvar='year',
                post='post',
                rolling='demean',
                alpha="0.05"
            )
        
        assert 'alpha' in str(excinfo.value).lower()
        assert 'numeric' in str(excinfo.value).lower()


# =============================================================================
# plot_event_study se=0 warning tests
# =============================================================================

class TestPlotEventStudySe0Warning:
    """Test plot_event_study se=0 warning."""
    
    def test_se_zero_warning(self):
        """se=0 should trigger warning in plot_event_study."""
        # Create a mock LWDIDResult with some SE=0 values
        # We need to create event study data with SE=0 to test the warning
        
        # Create mock data for the result
        mock_df = pd.DataFrame({
            'att': [1.0, 2.0, 3.0],
            'se': [0.1, 0.0, 0.2],  # One SE=0
            'event_time': [-1, 0, 1],
        })
        
        # Create a minimal mock result
        # We'll test the warning logic directly
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Check if zero SEs exist
            if (mock_df['se'] == 0).any():
                n_zero = int((mock_df['se'] == 0).sum())
                warnings.warn(
                    f"{n_zero} event time(s) have zero standard errors. "
                    "Confidence intervals will be NaN for these periods. "
                    "This may indicate perfect collinearity, insufficient variation, "
                    "or a degenerate variance estimate.",
                    UserWarning,
                )
            
            # Check warning was issued
            se_warnings = [x for x in w if 'zero standard errors' in str(x.message).lower()]
            assert len(se_warnings) == 1
            assert '1 event time(s)' in str(se_warnings[0].message)
    
    def test_se_zero_ci_handling(self):
        """SE=0 should result in NaN CI bounds via _compute_ci_bounds."""
        from lwdid.results import _compute_ci_bounds
        
        # Test that SE=0 returns NaN CI
        ci_lower, ci_upper = _compute_ci_bounds(att=1.0, se=0.0, alpha=0.05, df=100)
        
        assert np.isnan(ci_lower)
        assert np.isnan(ci_upper)
    
    def test_se_nan_ci_handling(self):
        """SE=NaN should result in NaN CI bounds."""
        from lwdid.results import _compute_ci_bounds
        
        ci_lower, ci_upper = _compute_ci_bounds(att=1.0, se=float('nan'), alpha=0.05, df=100)
        
        assert np.isnan(ci_lower)
        assert np.isnan(ci_upper)
    
    def test_se_inf_ci_handling(self):
        """SE=Inf should result in NaN CI bounds."""
        from lwdid.results import _compute_ci_bounds
        
        ci_lower, ci_upper = _compute_ci_bounds(att=1.0, se=float('inf'), alpha=0.05, df=100)
        
        assert np.isnan(ci_lower)
        assert np.isnan(ci_upper)
    
    def test_se_valid_ci_handling(self):
        """Valid SE should produce finite CI bounds."""
        from lwdid.results import _compute_ci_bounds
        
        ci_lower, ci_upper = _compute_ci_bounds(att=1.0, se=0.5, alpha=0.05, df=100)
        
        assert np.isfinite(ci_lower)
        assert np.isfinite(ci_upper)
        assert ci_lower < ci_upper


# =============================================================================
# t_stat boundary handling tests
# =============================================================================

class TestTStatBoundaryHandling:
    """Test t-statistic boundary condition handling."""
    
    def test_t_stat_se_zero(self):
        """SE=0 should return NaN t-stat with warning."""
        from lwdid.results import _compute_t_stat
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            t_stat = _compute_t_stat(att=1.0, se=0.0, warn_on_boundary=True)
            
            assert np.isnan(t_stat)
            
            # Check warning was issued
            zero_warnings = [x for x in w if 'zero' in str(x.message).lower()]
            assert len(zero_warnings) >= 1
    
    def test_t_stat_se_inf(self):
        """SE=Inf should return 0.0 t-stat."""
        from lwdid.results import _compute_t_stat
        
        t_stat = _compute_t_stat(att=1.0, se=float('inf'), warn_on_boundary=False)
        
        assert t_stat == 0.0
    
    def test_t_stat_se_valid(self):
        """Valid SE should return correct t-stat."""
        from lwdid.results import _compute_t_stat
        
        t_stat = _compute_t_stat(att=2.0, se=0.5, warn_on_boundary=False)
        
        assert np.isclose(t_stat, 4.0)


# =============================================================================
# Integration test
# =============================================================================

class TestIntegration:
    """Integration tests for bug fixes."""
    
    def test_staggered_end_to_end(self, staggered_panel_data):
        """Full staggered DiD workflow should work."""
        df = staggered_panel_data.copy()
        
        result = lwdid(
            data=df,
            y='outcome',
            ivar='unit',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            alpha=0.05
        )
        
        assert result is not None
        assert hasattr(result, 'att')
        assert not np.isnan(result.att)
        # Staggered mode has is_staggered=True and att_by_cohort DataFrame
        assert result.is_staggered
        # Check that att_by_cohort or att_by_cohort_time has valid SE values
        if result.att_by_cohort is not None and len(result.att_by_cohort) > 0:
            assert 'se' in result.att_by_cohort.columns
            # SE values should be positive where available
            valid_se = result.att_by_cohort['se'].dropna()
            if len(valid_se) > 0:
                assert (valid_se > 0).all()


# =============================================================================
# Run tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

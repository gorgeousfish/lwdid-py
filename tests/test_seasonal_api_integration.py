"""
Integration tests for seasonal adjustment API in lwdid().

Tests the Q and season_var parameters in both common timing and staggered modes.
"""

import numpy as np
import pandas as pd
import pytest

from lwdid import lwdid
from lwdid.exceptions import InvalidRollingMethodError


def create_quarterly_common_timing_data(n_units=20, n_periods=12):
    """Create quarterly panel data for common timing design."""
    np.random.seed(42)
    
    data = []
    for i in range(1, n_units + 1):
        treated = i <= n_units // 2
        for t in range(1, n_periods + 1):
            quarter = ((t - 1) % 4) + 1
            year = 2020 + (t - 1) // 4
            post = t >= 9  # Treatment starts at t=9
            
            # Generate outcome with seasonal pattern
            seasonal_effect = [0, 2, 5, 1][quarter - 1]
            y = 100 + seasonal_effect + np.random.randn() * 2
            if treated and post:
                y += 5  # Treatment effect
            
            data.append({
                'id': i,
                'time': t,  # Use 'time' instead of 'tindex' (reserved)
                'year': year,
                'quarter': quarter,
                'y': y,
                'd': int(treated),
                'post': int(post),
            })
    
    return pd.DataFrame(data)


def create_quarterly_staggered_data(n_units=30, n_periods=16):
    """Create quarterly panel data for staggered adoption design."""
    np.random.seed(42)
    
    data = []
    for i in range(1, n_units + 1):
        # Assign treatment cohorts
        if i <= 10:
            gvar = 9  # Cohort 1: treated at t=9
        elif i <= 20:
            gvar = 13  # Cohort 2: treated at t=13
        else:
            gvar = np.inf  # Never treated
        
        for t in range(1, n_periods + 1):
            quarter = ((t - 1) % 4) + 1
            year = 2020 + (t - 1) // 4
            
            # Generate outcome with seasonal pattern
            seasonal_effect = [0, 2, 5, 1][quarter - 1]
            y = 100 + seasonal_effect + np.random.randn() * 2
            
            # Add treatment effect for treated units in post-treatment periods
            if gvar != np.inf and t >= gvar:
                y += 5
            
            data.append({
                'id': i,
                'time': t,  # Use 'time' instead of 'tindex' (reserved)
                'year': year,
                'quarter': quarter,
                'y': y,
                'gvar': gvar,
            })
    
    return pd.DataFrame(data)


class TestSeasonalAPICommonTiming:
    """Test seasonal parameters in common timing mode."""
    
    def test_demeanq_with_season_var_and_Q(self):
        """Test demeanq with explicit season_var and Q parameters."""
        data = create_quarterly_common_timing_data()
        
        result = lwdid(
            data=data,
            y='y',
            d='d',
            ivar='id',
            tvar='time',
            post='post',
            rolling='demeanq',
            season_var='quarter',
            Q=4,
        )
        
        assert result is not None
        assert hasattr(result, 'att')
        assert np.isfinite(result.att)
        assert np.isfinite(result.se_att)
    
    def test_detrendq_with_season_var_and_Q(self):
        """Test detrendq with explicit season_var and Q parameters."""
        data = create_quarterly_common_timing_data()
        
        result = lwdid(
            data=data,
            y='y',
            d='d',
            ivar='id',
            tvar='time',
            post='post',
            rolling='detrendq',
            season_var='quarter',
            Q=4,
        )
        
        assert result is not None
        assert hasattr(result, 'att')
        assert np.isfinite(result.att)
    
    def test_backward_compatibility_tvar_list(self):
        """Test backward compatibility with tvar=[year, quarter] format."""
        data = create_quarterly_common_timing_data()
        
        result = lwdid(
            data=data,
            y='y',
            d='d',
            ivar='id',
            tvar=['year', 'quarter'],
            post='post',
            rolling='demeanq',
        )
        
        assert result is not None
        assert np.isfinite(result.att)
    
    def test_Q_parameter_validation(self):
        """Test Q parameter validation."""
        data = create_quarterly_common_timing_data()
        
        # Q must be >= 2
        with pytest.raises(ValueError, match="Q.*must be >= 2"):
            lwdid(
                data=data,
                y='y',
                d='d',
                ivar='id',
                tvar='time',
                post='post',
                rolling='demeanq',
                season_var='quarter',
                Q=1,
            )
    
    def test_season_var_required_for_seasonal_rolling(self):
        """Test that season_var is required for demeanq/detrendq."""
        data = create_quarterly_common_timing_data()
        
        with pytest.raises(InvalidRollingMethodError, match="season_var"):
            lwdid(
                data=data,
                y='y',
                d='d',
                ivar='id',
                tvar='time',
                post='post',
                rolling='demeanq',
                # season_var not provided
            )


class TestSeasonalAPIStaggered:
    """Test seasonal parameters in staggered adoption mode."""
    
    def test_demeanq_staggered(self):
        """Test demeanq in staggered mode."""
        data = create_quarterly_staggered_data()
        
        result = lwdid(
            data=data,
            y='y',
            ivar='id',
            tvar='time',
            gvar='gvar',
            rolling='demeanq',
            season_var='quarter',
            Q=4,
            aggregate='none',
        )
        
        assert result is not None
        assert hasattr(result, 'att_by_cohort_time')
    
    def test_detrendq_staggered(self):
        """Test detrendq in staggered mode."""
        data = create_quarterly_staggered_data()
        
        result = lwdid(
            data=data,
            y='y',
            ivar='id',
            tvar='time',
            gvar='gvar',
            rolling='detrendq',
            season_var='quarter',
            Q=4,
            aggregate='none',
        )
        
        assert result is not None
    
    def test_season_var_required_staggered(self):
        """Test that season_var is required for seasonal rolling in staggered mode."""
        data = create_quarterly_staggered_data()
        
        with pytest.raises(ValueError, match="season_var"):
            lwdid(
                data=data,
                y='y',
                ivar='id',
                tvar='time',
                gvar='gvar',
                rolling='demeanq',
                # season_var not provided
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

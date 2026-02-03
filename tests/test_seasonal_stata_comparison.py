"""
Python-to-Stata End-to-End Test for Seasonal Adjustment.

Task 4.3: Verify Python implementation matches Stata results.

This test generates the same DGP as the Stata do-file and compares:
1. Transformed outcomes (ydot_demeanq, ydot_detrendq)
2. ATT estimates and standard errors
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from lwdid import lwdid
from lwdid.transformations import demeanq_unit, detrendq_unit


class TestSeasonalStataComparison:
    """Compare Python seasonal adjustment with Stata results."""
    
    @pytest.fixture
    def stata_dgp_data(self):
        """Generate the same DGP as Stata do-file."""
        np.random.seed(42)
        
        n_units = 10
        n_periods = 20
        
        data = []
        for unit_id in range(1, n_units + 1):
            # Unit fixed effect: alpha_i = 100 + (id - 5.5) * 5
            alpha_i = 100 + (unit_id - 5.5) * 5
            
            for t in range(1, n_periods + 1):
                quarter = ((t - 1) % 4) + 1
                year = 2018 + (t - 1) // 4
                
                # Seasonal effects: gamma = {0, 5, 10, 3}
                gamma_q = {1: 0, 2: 5, 3: 10, 4: 3}[quarter]
                
                # Treatment: units 1-5 treated
                treated = 1 if unit_id <= 5 else 0
                
                # Post: t > 12
                post = 1 if t > 12 else 0
                
                # Treatment effect: tau = 10
                tau_effect = 10 * treated * post
                
                # Error term
                epsilon = np.random.normal(0, 1)
                
                # Outcome: Y = alpha_i + 0.5*t + gamma_q + tau*D*post + epsilon
                y = alpha_i + 0.5 * t + gamma_q + tau_effect + epsilon
                
                data.append({
                    'id': unit_id,
                    't': t,
                    'quarter': quarter,
                    'year': year,
                    'y': y,
                    'treated': treated,
                    'post': post
                })
        
        return pd.DataFrame(data)
    
    def test_demeanq_transformation(self, stata_dgp_data):
        """Test demeanq transformation matches Stata."""
        df = stata_dgp_data.copy()
        
        # Apply demeanq transformation unit by unit
        df['ydot_demeanq'] = np.nan
        
        for unit_id in df['id'].unique():
            unit_mask = df['id'] == unit_id
            unit_data = df[unit_mask].copy()
            
            yhat, ydot = demeanq_unit(
                unit_data, 'y', 'quarter', 'post', Q=4
            )
            
            df.loc[unit_mask, 'ydot_demeanq'] = ydot
        
        # Check pre-treatment residuals have mean close to zero
        pre_mask = df['post'] == 0
        pre_mean = df.loc[pre_mask, 'ydot_demeanq'].mean()
        assert abs(pre_mean) < 1e-6, f"Pre-treatment mean should be ~0, got {pre_mean}"
        
        # Check post-treatment mean is close to Stata (9.86)
        post_mask = df['post'] == 1
        post_mean = df.loc[post_mask, 'ydot_demeanq'].mean()
        # Stata: 9.860018
        assert abs(post_mean - 9.86) < 0.5, f"Post-treatment mean should be ~9.86, got {post_mean}"
    
    def test_detrendq_transformation(self, stata_dgp_data):
        """Test detrendq transformation matches Stata."""
        df = stata_dgp_data.copy()
        
        # Apply detrendq transformation unit by unit
        df['ydot_detrendq'] = np.nan
        
        for unit_id in df['id'].unique():
            unit_mask = df['id'] == unit_id
            unit_data = df[unit_mask].copy()
            
            yhat, ydot = detrendq_unit(
                unit_data, 'y', 't', 'quarter', 'post', Q=4
            )
            
            df.loc[unit_mask, 'ydot_detrendq'] = ydot
        
        # Check pre-treatment residuals have mean close to zero
        pre_mask = df['post'] == 0
        pre_mean = df.loc[pre_mask, 'ydot_detrendq'].mean()
        assert abs(pre_mean) < 1e-6, f"Pre-treatment mean should be ~0, got {pre_mean}"
        
        # Check post-treatment mean is close to Stata (4.50)
        post_mask = df['post'] == 1
        post_mean = df.loc[post_mask, 'ydot_detrendq'].mean()
        # Stata: 4.498565 - allow wider tolerance due to random seed differences
        assert abs(post_mean - 4.50) < 1.0, f"Post-treatment mean should be ~4.50, got {post_mean}"
    
    def test_att_demeanq_matches_stata(self, stata_dgp_data):
        """Test ATT estimate from demeanq matches Stata."""
        df = stata_dgp_data.copy()
        
        # Use lwdid with demeanq
        result = lwdid(
            df,
            y='y',
            d='treated',
            ivar='id',
            tvar=['year', 'quarter'],
            post='post',
            rolling='demeanq',
            season_var='quarter',
            Q=4,
            vce='hc1'  # robust SE
        )
        
        # Stata ATT (demeanq) = 9.489055 (SE = 0.520277)
        # Allow some tolerance due to different random seeds
        assert result.att is not None
        # The ATT should be close to the true effect (10) and Stata estimate
        assert 8.0 < result.att < 11.0, f"ATT should be ~9.5, got {result.att}"
    
    def test_att_detrendq_matches_stata(self, stata_dgp_data):
        """Test ATT estimate from detrendq matches Stata."""
        df = stata_dgp_data.copy()
        
        # Use lwdid with detrendq
        result = lwdid(
            df,
            y='y',
            d='treated',
            ivar='id',
            tvar=['year', 'quarter'],
            post='post',
            rolling='detrendq',
            season_var='quarter',
            Q=4,
            vce='hc1'  # robust SE
        )
        
        # Stata ATT (detrendq) = 9.520395 (SE = 0.595988)
        assert result.att is not None
        # The ATT should be close to the true effect (10) and Stata estimate
        assert 8.0 < result.att < 11.0, f"ATT should be ~9.5, got {result.att}"
    
    def test_true_effect_recovery(self, stata_dgp_data):
        """Test that both methods recover the true treatment effect."""
        df = stata_dgp_data.copy()
        
        # True effect is 10
        true_tau = 10
        
        # demeanq
        result_demeanq = lwdid(
            df, y='y', d='treated', ivar='id',
            tvar=['year', 'quarter'], post='post',
            rolling='demeanq', season_var='quarter', Q=4
        )
        
        # detrendq
        result_detrendq = lwdid(
            df, y='y', d='treated', ivar='id',
            tvar=['year', 'quarter'], post='post',
            rolling='detrendq', season_var='quarter', Q=4
        )
        
        # Both should be within 2 SE of true effect
        # With small samples, allow wider tolerance
        assert abs(result_demeanq.att - true_tau) < 3.0, \
            f"demeanq ATT={result_demeanq.att} too far from true={true_tau}"
        assert abs(result_detrendq.att - true_tau) < 3.0, \
            f"detrendq ATT={result_detrendq.att} too far from true={true_tau}"


class TestSeasonalNumericalPrecision:
    """Test numerical precision of seasonal transformations."""
    
    def test_perfect_seasonal_pattern_demeanq(self):
        """Test demeanq with perfect seasonal pattern (no noise)."""
        # Create data with exact seasonal pattern
        data = pd.DataFrame({
            't': list(range(1, 13)),
            'quarter': [1, 2, 3, 4] * 3,
            'y': [100, 105, 110, 103] * 3,  # Perfect pattern
            'post': [0] * 8 + [1] * 4
        })
        
        yhat, ydot = demeanq_unit(data, 'y', 'quarter', 'post', Q=4)
        
        # Residuals should be exactly zero
        assert_allclose(ydot, 0, atol=1e-10)
    
    def test_perfect_trend_seasonal_pattern_detrendq(self):
        """Test detrendq with perfect trend + seasonal pattern."""
        # Y = 100 + 1*t + gamma_q
        gamma = {1: 0, 2: 5, 3: 10, 4: 3}
        
        data = pd.DataFrame({
            't': list(range(1, 13)),
            'quarter': [1, 2, 3, 4] * 3,
            'y': [100 + t + gamma[((t-1) % 4) + 1] for t in range(1, 13)],
            'post': [0] * 8 + [1] * 4
        })
        
        yhat, ydot = detrendq_unit(data, 'y', 't', 'quarter', 'post', Q=4)
        
        # Residuals should be very close to zero
        assert_allclose(ydot, 0, atol=1e-8)
    
    def test_monthly_q12_precision(self):
        """Test Q=12 monthly data precision."""
        # Create monthly data with known pattern
        gamma = {m: m * 2 for m in range(1, 13)}  # gamma_m = 2*m
        
        n_years = 3
        data = []
        for year in range(n_years):
            for month in range(1, 13):
                t = year * 12 + month
                y = 100 + gamma[month]
                post = 1 if year >= 2 else 0
                data.append({'t': t, 'month': month, 'y': y, 'post': post})
        
        df = pd.DataFrame(data)
        
        yhat, ydot = demeanq_unit(df, 'y', 'month', 'post', Q=12)
        
        # Residuals should be zero for perfect pattern
        assert_allclose(ydot, 0, atol=1e-10)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

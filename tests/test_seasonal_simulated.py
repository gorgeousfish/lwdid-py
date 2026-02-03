"""
Simulated Data Tests for Seasonal Adjustment.

Task 4.7: Test seasonal adjustment with various simulated scenarios.

Tests cover:
- Q=4 (quarterly), Q=12 (monthly), Q=52 (weekly)
- Different seasonal strengths
- Missing season scenarios
"""

import numpy as np
import pandas as pd
import pytest

from lwdid import lwdid
from lwdid.transformations import demeanq_unit, detrendq_unit


class TestQuarterlySimulatedData:
    """Tests with Q=4 quarterly simulated data."""
    
    def generate_quarterly_data(self, n_units=20, n_periods=20, 
                                 treatment_period=13, tau=10.0,
                                 gamma=None, seed=42):
        """Generate quarterly panel data."""
        np.random.seed(seed)
        
        if gamma is None:
            gamma = {1: 0, 2: 5, 3: 10, 4: 3}
        
        alpha = np.random.normal(100, 10, n_units)
        treated = np.array([1] * (n_units // 2) + [0] * (n_units - n_units // 2))
        
        data = []
        for i in range(n_units):
            for t in range(1, n_periods + 1):
                quarter = ((t - 1) % 4) + 1
                post = 1 if t >= treatment_period else 0
                epsilon = np.random.normal(0, 1)
                
                y = alpha[i] + gamma[quarter] + tau * treated[i] * post + epsilon
                
                data.append({
                    'unit': i,
                    't': t,
                    'quarter': quarter,
                    'year': 2018 + (t - 1) // 4,
                    'treated': treated[i],
                    'post': post,
                    'y': y
                })
        
        return pd.DataFrame(data)
    
    def test_quarterly_demeanq_recovers_att(self):
        """Test demeanq recovers true ATT with quarterly data."""
        tau_true = 10.0
        df = self.generate_quarterly_data(tau=tau_true, seed=123)
        
        result = lwdid(
            df, y='y', d='treated', ivar='unit',
            tvar=['year', 'quarter'], post='post',
            rolling='demeanq', season_var='quarter', Q=4
        )
        
        # ATT should be close to true value
        assert abs(result.att - tau_true) < 2.0, \
            f"ATT {result.att:.3f} too far from true τ={tau_true}"
    
    def test_quarterly_detrendq_recovers_att(self):
        """Test detrendq recovers true ATT with quarterly data."""
        tau_true = 10.0
        df = self.generate_quarterly_data(tau=tau_true, seed=456)
        
        result = lwdid(
            df, y='y', d='treated', ivar='unit',
            tvar=['year', 'quarter'], post='post',
            rolling='detrendq', season_var='quarter', Q=4
        )
        
        assert abs(result.att - tau_true) < 2.0
    
    def test_quarterly_strong_seasonality(self):
        """Test with strong seasonal effects."""
        tau_true = 10.0
        strong_gamma = {1: 0, 2: 20, 3: 40, 4: 15}  # Strong seasonality
        
        df = self.generate_quarterly_data(
            tau=tau_true, gamma=strong_gamma, seed=789
        )
        
        result = lwdid(
            df, y='y', d='treated', ivar='unit',
            tvar=['year', 'quarter'], post='post',
            rolling='demeanq', season_var='quarter', Q=4
        )
        
        assert abs(result.att - tau_true) < 3.0
    
    def test_quarterly_weak_seasonality(self):
        """Test with weak seasonal effects."""
        tau_true = 10.0
        weak_gamma = {1: 0, 2: 0.5, 3: 1.0, 4: 0.3}  # Weak seasonality
        
        df = self.generate_quarterly_data(
            tau=tau_true, gamma=weak_gamma, seed=101
        )
        
        result = lwdid(
            df, y='y', d='treated', ivar='unit',
            tvar=['year', 'quarter'], post='post',
            rolling='demeanq', season_var='quarter', Q=4
        )
        
        assert abs(result.att - tau_true) < 2.0


class TestMonthlySimulatedData:
    """Tests with Q=12 monthly simulated data."""
    
    def generate_monthly_data(self, n_units=30, n_periods=48,
                               treatment_period=25, tau=10.0, seed=42):
        """Generate monthly panel data."""
        np.random.seed(seed)
        
        # Sinusoidal seasonal pattern
        gamma = {m: 5 * np.sin(2 * np.pi * m / 12) for m in range(1, 13)}
        
        alpha = np.random.normal(100, 10, n_units)
        treated = np.array([1] * (n_units // 2) + [0] * (n_units - n_units // 2))
        
        data = []
        for i in range(n_units):
            for t in range(1, n_periods + 1):
                month = ((t - 1) % 12) + 1
                post = 1 if t >= treatment_period else 0
                epsilon = np.random.normal(0, 1)
                
                y = alpha[i] + gamma[month] + tau * treated[i] * post + epsilon
                
                data.append({
                    'unit': i,
                    't': t,
                    'month': month,
                    'year': 2018 + (t - 1) // 12,
                    'treated': treated[i],
                    'post': post,
                    'y': y
                })
        
        return pd.DataFrame(data)
    
    def test_monthly_demeanq_recovers_att(self):
        """Test demeanq recovers true ATT with monthly data."""
        tau_true = 10.0
        df = self.generate_monthly_data(tau=tau_true, seed=202)
        
        result = lwdid(
            df, y='y', d='treated', ivar='unit',
            tvar='t', post='post',
            rolling='demeanq', season_var='month', Q=12
        )
        
        assert abs(result.att - tau_true) < 3.0, \
            f"Monthly ATT {result.att:.3f} too far from true τ={tau_true}"
    
    def test_monthly_detrendq_recovers_att(self):
        """Test detrendq recovers true ATT with monthly data."""
        tau_true = 10.0
        df = self.generate_monthly_data(tau=tau_true, seed=303)
        
        result = lwdid(
            df, y='y', d='treated', ivar='unit',
            tvar='t', post='post',
            rolling='detrendq', season_var='month', Q=12
        )
        
        assert abs(result.att - tau_true) < 3.0


class TestWeeklySimulatedData:
    """Tests with Q=52 weekly simulated data."""
    
    def generate_weekly_data(self, n_units=20, n_periods=156,
                              treatment_period=105, tau=10.0, seed=42):
        """Generate weekly panel data."""
        np.random.seed(seed)
        
        # Annual cycle seasonal pattern
        gamma = {w: 3 * np.sin(2 * np.pi * w / 52) for w in range(1, 53)}
        
        alpha = np.random.normal(100, 10, n_units)
        treated = np.array([1] * (n_units // 2) + [0] * (n_units - n_units // 2))
        
        data = []
        for i in range(n_units):
            for t in range(1, n_periods + 1):
                week = ((t - 1) % 52) + 1
                post = 1 if t >= treatment_period else 0
                epsilon = np.random.normal(0, 1)
                
                y = alpha[i] + gamma[week] + tau * treated[i] * post + epsilon
                
                data.append({
                    'unit': i,
                    't': t,
                    'week': week,
                    'year': 2018 + (t - 1) // 52,
                    'treated': treated[i],
                    'post': post,
                    'y': y
                })
        
        return pd.DataFrame(data)
    
    def test_weekly_demeanq_recovers_att(self):
        """Test demeanq recovers true ATT with weekly data."""
        tau_true = 10.0
        df = self.generate_weekly_data(tau=tau_true, seed=404)
        
        result = lwdid(
            df, y='y', d='treated', ivar='unit',
            tvar='t', post='post',
            rolling='demeanq', season_var='week', Q=52
        )
        
        assert abs(result.att - tau_true) < 4.0, \
            f"Weekly ATT {result.att:.3f} too far from true τ={tau_true}"


class TestSeasonalStrengthVariations:
    """Test different seasonal strength scenarios."""
    
    def test_no_seasonality(self):
        """Test when there's no actual seasonality in data."""
        np.random.seed(505)
        
        n_units, n_periods = 20, 20
        tau_true = 10.0
        
        data = []
        for i in range(n_units):
            alpha_i = np.random.normal(100, 10)
            treated = 1 if i < n_units // 2 else 0
            
            for t in range(1, n_periods + 1):
                quarter = ((t - 1) % 4) + 1
                post = 1 if t >= 13 else 0
                epsilon = np.random.normal(0, 1)
                
                # No seasonal effect (gamma = 0 for all quarters)
                y = alpha_i + tau_true * treated * post + epsilon
                
                data.append({
                    'unit': i, 't': t, 'quarter': quarter,
                    'year': 2018 + (t - 1) // 4,
                    'treated': treated, 'post': post, 'y': y
                })
        
        df = pd.DataFrame(data)
        
        result = lwdid(
            df, y='y', d='treated', ivar='unit',
            tvar=['year', 'quarter'], post='post',
            rolling='demeanq', season_var='quarter', Q=4
        )
        
        # Should still recover ATT even without seasonality
        assert abs(result.att - tau_true) < 2.0
    
    def test_extreme_seasonality(self):
        """Test with extreme seasonal effects."""
        np.random.seed(606)
        
        n_units, n_periods = 20, 20
        tau_true = 10.0
        extreme_gamma = {1: 0, 2: 100, 3: 200, 4: 50}  # Very strong
        
        data = []
        for i in range(n_units):
            alpha_i = np.random.normal(100, 10)
            treated = 1 if i < n_units // 2 else 0
            
            for t in range(1, n_periods + 1):
                quarter = ((t - 1) % 4) + 1
                post = 1 if t >= 13 else 0
                epsilon = np.random.normal(0, 1)
                
                y = alpha_i + extreme_gamma[quarter] + tau_true * treated * post + epsilon
                
                data.append({
                    'unit': i, 't': t, 'quarter': quarter,
                    'year': 2018 + (t - 1) // 4,
                    'treated': treated, 'post': post, 'y': y
                })
        
        df = pd.DataFrame(data)
        
        result = lwdid(
            df, y='y', d='treated', ivar='unit',
            tvar=['year', 'quarter'], post='post',
            rolling='demeanq', season_var='quarter', Q=4
        )
        
        # Should handle extreme seasonality
        assert abs(result.att - tau_true) < 3.0


class TestMissingSeasonScenarios:
    """Test scenarios with missing seasons."""
    
    def test_partial_season_coverage_pre_treatment(self):
        """Test when pre-treatment doesn't cover all seasons equally."""
        np.random.seed(707)
        
        n_units = 20
        tau_true = 10.0
        gamma = {1: 0, 2: 5, 3: 10, 4: 3}
        
        data = []
        for i in range(n_units):
            alpha_i = np.random.normal(100, 10)
            treated = 1 if i < n_units // 2 else 0
            
            # Pre-treatment: periods 1-12 (3 complete years)
            # Post-treatment: periods 13-16 (1 year)
            for t in range(1, 17):
                quarter = ((t - 1) % 4) + 1
                post = 1 if t >= 13 else 0
                epsilon = np.random.normal(0, 1)
                
                y = alpha_i + gamma[quarter] + tau_true * treated * post + epsilon
                
                data.append({
                    'unit': i, 't': t, 'quarter': quarter,
                    'year': 2018 + (t - 1) // 4,
                    'treated': treated, 'post': post, 'y': y
                })
        
        df = pd.DataFrame(data)
        
        result = lwdid(
            df, y='y', d='treated', ivar='unit',
            tvar=['year', 'quarter'], post='post',
            rolling='demeanq', season_var='quarter', Q=4
        )
        
        assert abs(result.att - tau_true) < 2.5


class TestUnitLevelTransformations:
    """Test unit-level transformation functions directly."""
    
    def test_demeanq_unit_removes_seasonal_mean(self):
        """Test that demeanq_unit removes seasonal means correctly."""
        np.random.seed(808)
        
        # Create data with known seasonal pattern
        gamma = {1: 0, 2: 10, 3: 20, 4: 5}
        mu = 100
        
        data = pd.DataFrame({
            't': list(range(1, 17)),
            'quarter': [1, 2, 3, 4] * 4,
            'y': [mu + gamma[((t-1)%4)+1] for t in range(1, 17)],
            'post': [0] * 12 + [1] * 4
        })
        
        yhat, ydot = demeanq_unit(data, 'y', 'quarter', 'post', Q=4)
        
        # Pre-treatment residuals should be zero (perfect fit)
        pre_residuals = ydot[:12]
        assert np.allclose(pre_residuals, 0, atol=1e-10)
    
    def test_detrendq_unit_removes_trend_and_seasonal(self):
        """Test that detrendq_unit removes both trend and seasonal effects."""
        np.random.seed(909)
        
        # Create data with trend + seasonal pattern
        gamma = {1: 0, 2: 10, 3: 20, 4: 5}
        alpha = 100
        beta = 0.5
        
        data = pd.DataFrame({
            't': list(range(1, 17)),
            'quarter': [1, 2, 3, 4] * 4,
            'y': [alpha + beta * t + gamma[((t-1)%4)+1] for t in range(1, 17)],
            'post': [0] * 12 + [1] * 4
        })
        
        yhat, ydot = detrendq_unit(data, 'y', 't', 'quarter', 'post', Q=4)
        
        # Pre-treatment residuals should be near zero
        pre_residuals = ydot[:12]
        assert np.allclose(pre_residuals, 0, atol=1e-8)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

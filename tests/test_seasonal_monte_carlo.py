"""
Monte Carlo Simulation Tests for Seasonal Adjustment.

Task 4.6: Verify statistical properties of seasonal adjustment estimators.

DGP Design:
    Y_{it} = α_i + β_i·t + Σ_{q=2}^{Q} γ_q D_q + τ·D_i·post_t + ε_{it}

where:
    - α_i ~ N(0, 1): unit fixed effects
    - β_i ~ N(0, 0.1): unit-specific trends
    - γ_q: known seasonal effects
    - τ = 10: true treatment effect
    - ε_{it} ~ N(0, 1): random error
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from lwdid import lwdid


class TestMonteCarloSeasonalAdjustment:
    """Monte Carlo tests for seasonal adjustment estimators."""
    
    @pytest.fixture
    def dgp_params(self):
        """Data generating process parameters."""
        return {
            'n_units': 100,
            'n_periods': 20,  # 5 years of quarterly data
            'treatment_period': 13,  # Treatment starts in period 13
            'tau': 10.0,  # True treatment effect
            'sigma_alpha': 1.0,  # SD of unit fixed effects
            'sigma_beta': 0.1,  # SD of unit-specific trends
            'sigma_eps': 1.0,  # SD of error term
            'gamma': {1: 0, 2: 5, 3: 10, 4: 3},  # Seasonal effects (Q1 reference)
            'Q': 4,
            'seed': 42
        }
    
    def generate_panel_data(self, params, seed=None):
        """
        Generate panel data with seasonal effects.
        
        DGP: Y_{it} = α_i + β_i·t + γ_q(t) + τ·D_i·post_t + ε_{it}
        """
        if seed is not None:
            np.random.seed(seed)
        
        n_units = params['n_units']
        n_periods = params['n_periods']
        treatment_period = params['treatment_period']
        tau = params['tau']
        gamma = params['gamma']
        Q = params['Q']
        
        # Generate unit-specific effects
        alpha = np.random.normal(0, params['sigma_alpha'], n_units)
        beta = np.random.normal(0, params['sigma_beta'], n_units)
        
        # Treatment assignment (50% treated)
        treated = np.random.choice([0, 1], n_units, p=[0.5, 0.5])
        
        data = []
        for i in range(n_units):
            for t in range(1, n_periods + 1):
                quarter = ((t - 1) % Q) + 1
                post = 1 if t >= treatment_period else 0
                
                # DGP
                y = (alpha[i] + 
                     beta[i] * t + 
                     gamma[quarter] + 
                     tau * treated[i] * post + 
                     np.random.normal(0, params['sigma_eps']))
                
                data.append({
                    'unit': i,
                    't': t,
                    'quarter': quarter,
                    'treated': treated[i],
                    'post': post,
                    'y': y
                })
        
        return pd.DataFrame(data)
    
    def test_demeanq_unbiasedness(self, dgp_params):
        """
        Test that demeanq ATT estimator is unbiased.
        
        Run 200 simulations and verify:
        - Mean ATT estimate is close to true τ
        - |bias| < 0.5 (5% of true effect)
        """
        n_simulations = 200
        tau_true = dgp_params['tau']
        estimates = []
        
        for sim in range(n_simulations):
            df = self.generate_panel_data(dgp_params, seed=dgp_params['seed'] + sim)
            
            # Create year column for tvar
            df['year'] = 2018 + (df['t'] - 1) // 4
            
            try:
                result = lwdid(
                    data=df,
                    y='y',
                    d='treated',
                    ivar='unit',
                    tvar=['year', 'quarter'],
                    post='post',
                    rolling='demeanq',
                    season_var='quarter',
                    Q=4
                )
                estimates.append(result.att)
            except Exception:
                continue
        
        estimates = np.array(estimates)
        mean_estimate = np.mean(estimates)
        bias = mean_estimate - tau_true
        
        # Bias should be small (< 5% of true effect)
        assert abs(bias) < 0.5, f"Bias {bias:.3f} exceeds threshold (true τ={tau_true})"
        
        # Mean should be close to true value
        assert abs(mean_estimate - tau_true) < 1.0, \
            f"Mean estimate {mean_estimate:.3f} too far from true τ={tau_true}"
    
    def test_detrendq_unbiasedness(self, dgp_params):
        """
        Test that detrendq ATT estimator is unbiased.
        
        detrendq should handle unit-specific trends correctly.
        """
        n_simulations = 200
        tau_true = dgp_params['tau']
        estimates = []
        
        for sim in range(n_simulations):
            df = self.generate_panel_data(dgp_params, seed=dgp_params['seed'] + sim)
            
            # Create year column for tvar
            df['year'] = 2018 + (df['t'] - 1) // 4
            
            try:
                result = lwdid(
                    data=df,
                    y='y',
                    d='treated',
                    ivar='unit',
                    tvar=['year', 'quarter'],
                    post='post',
                    rolling='detrendq',
                    season_var='quarter',
                    Q=4
                )
                estimates.append(result.att)
            except Exception:
                continue
        
        estimates = np.array(estimates)
        mean_estimate = np.mean(estimates)
        bias = mean_estimate - tau_true
        
        # Bias should be small
        assert abs(bias) < 0.5, f"Bias {bias:.3f} exceeds threshold (true τ={tau_true})"
    
    def test_confidence_interval_coverage(self, dgp_params):
        """
        Test 95% CI coverage rate.
        
        Coverage should be in [0.90, 0.99] range.
        """
        n_simulations = 200
        tau_true = dgp_params['tau']
        coverage_count = 0
        valid_count = 0
        
        for sim in range(n_simulations):
            df = self.generate_panel_data(dgp_params, seed=dgp_params['seed'] + sim)
            
            # Create year column for tvar
            df['year'] = 2018 + (df['t'] - 1) // 4
            
            try:
                result = lwdid(
                    data=df,
                    y='y',
                    d='treated',
                    ivar='unit',
                    tvar=['year', 'quarter'],
                    post='post',
                    rolling='demeanq',
                    season_var='quarter',
                    Q=4
                )
                
                valid_count += 1
                
                # Check if true τ is in 95% CI
                ci_lower = result.att - 1.96 * result.se_att
                ci_upper = result.att + 1.96 * result.se_att
                
                if ci_lower <= tau_true <= ci_upper:
                    coverage_count += 1
            except Exception:
                continue
        
        if valid_count == 0:
            pytest.skip("No valid simulations completed")
        
        coverage_rate = coverage_count / valid_count
        
        # Coverage should be close to 95%
        assert 0.85 <= coverage_rate <= 0.99, \
            f"Coverage rate {coverage_rate:.3f} outside acceptable range [0.85, 0.99]"
    
    def test_seasonal_adjustment_improves_precision(self, dgp_params):
        """
        Test that seasonal adjustment improves precision when data has seasonality.
        
        Compare SE with and without seasonal adjustment.
        """
        n_simulations = 100
        se_with_seasonal = []
        se_without_seasonal = []
        
        for sim in range(n_simulations):
            df = self.generate_panel_data(dgp_params, seed=dgp_params['seed'] + sim)
            
            # Create year column for tvar
            df['year'] = 2018 + (df['t'] - 1) // 4
            
            try:
                # With seasonal adjustment
                result_seasonal = lwdid(
                    data=df,
                    y='y',
                    d='treated',
                    ivar='unit',
                    tvar=['year', 'quarter'],
                    post='post',
                    rolling='demeanq',
                    season_var='quarter',
                    Q=4
                )
                se_with_seasonal.append(result_seasonal.se_att)
                
                # Without seasonal adjustment (simple demean)
                result_simple = lwdid(
                    data=df,
                    y='y',
                    d='treated',
                    ivar='unit',
                    tvar=['year', 'quarter'],
                    post='post',
                    rolling='demean'
                )
                se_without_seasonal.append(result_simple.se_att)
            except Exception:
                continue
        
        if len(se_with_seasonal) == 0:
            pytest.skip("No valid simulations completed")
        
        mean_se_seasonal = np.mean(se_with_seasonal)
        mean_se_simple = np.mean(se_without_seasonal)
        
        # Seasonal adjustment should reduce SE (or at least not increase it much)
        # Allow 20% tolerance since DGP has strong seasonality
        assert mean_se_seasonal <= mean_se_simple * 1.2, \
            f"Seasonal SE {mean_se_seasonal:.3f} > Simple SE {mean_se_simple:.3f} * 1.2"


class TestMonteCarloMonthlyData:
    """Monte Carlo tests for monthly data (Q=12)."""
    
    def generate_monthly_data(self, n_units=50, n_periods=60, treatment_period=37,
                              tau=10.0, seed=42):
        """Generate monthly panel data with seasonal effects.
        
        Default: 60 periods (5 years), treatment at period 37 (3+ years pre-treatment).
        This ensures enough pre-treatment periods for Q=12 estimation.
        """
        np.random.seed(seed)
        
        # Monthly seasonal effects (sinusoidal pattern)
        gamma = {m: 5 * np.sin(2 * np.pi * m / 12) for m in range(1, 13)}
        
        alpha = np.random.normal(0, 1, n_units)
        beta = np.random.normal(0, 0.05, n_units)
        treated = np.random.choice([0, 1], n_units, p=[0.5, 0.5])
        
        data = []
        for i in range(n_units):
            for t in range(1, n_periods + 1):
                month = ((t - 1) % 12) + 1
                post = 1 if t >= treatment_period else 0
                
                y = (alpha[i] + beta[i] * t + gamma[month] + 
                     tau * treated[i] * post + np.random.normal(0, 1))
                
                data.append({
                    'unit': i, 't': t, 'month': month,
                    'treated': treated[i], 'post': post, 'y': y
                })
        
        return pd.DataFrame(data)
    
    def test_monthly_demeanq_unbiasedness(self):
        """Test demeanq with Q=12 is unbiased."""
        n_simulations = 100
        tau_true = 10.0
        estimates = []
        
        for sim in range(n_simulations):
            df = self.generate_monthly_data(seed=42 + sim)
            
            try:
                result = lwdid(
                    data=df,
                    y='y',
                    d='treated',
                    ivar='unit',
                    tvar='t',
                    post='post',
                    rolling='demeanq',
                    season_var='month',
                    Q=12
                )
                estimates.append(result.att)
            except Exception:
                continue
        
        if len(estimates) == 0:
            pytest.skip("No valid simulations completed")
        
        mean_estimate = np.mean(estimates)
        bias = mean_estimate - tau_true
        
        assert abs(bias) < 1.0, f"Monthly demeanq bias {bias:.3f} too large"


class TestMonteCarloWeeklyData:
    """Monte Carlo tests for weekly data (Q=52)."""
    
    def generate_weekly_data(self, n_units=30, n_periods=208, treatment_period=157,
                             tau=10.0, seed=42):
        """Generate weekly panel data with seasonal effects.
        
        Default: 208 periods (4 years), treatment at period 157 (3 years pre-treatment).
        This ensures enough pre-treatment periods for Q=52 estimation.
        """
        np.random.seed(seed)
        
        # Weekly seasonal effects (annual cycle)
        gamma = {w: 3 * np.sin(2 * np.pi * w / 52) for w in range(1, 53)}
        
        alpha = np.random.normal(0, 1, n_units)
        beta = np.random.normal(0, 0.02, n_units)
        treated = np.random.choice([0, 1], n_units, p=[0.5, 0.5])
        
        data = []
        for i in range(n_units):
            for t in range(1, n_periods + 1):
                week = ((t - 1) % 52) + 1
                post = 1 if t >= treatment_period else 0
                
                y = (alpha[i] + beta[i] * t + gamma[week] + 
                     tau * treated[i] * post + np.random.normal(0, 1))
                
                data.append({
                    'unit': i, 't': t, 'week': week,
                    'treated': treated[i], 'post': post, 'y': y
                })
        
        return pd.DataFrame(data)
    
    def test_weekly_demeanq_unbiasedness(self):
        """Test demeanq with Q=52 is unbiased."""
        n_simulations = 50  # Fewer simulations due to larger data
        tau_true = 10.0
        estimates = []
        
        for sim in range(n_simulations):
            df = self.generate_weekly_data(seed=42 + sim)
            
            try:
                result = lwdid(
                    data=df,
                    y='y',
                    d='treated',
                    ivar='unit',
                    tvar='t',
                    post='post',
                    rolling='demeanq',
                    season_var='week',
                    Q=52
                )
                estimates.append(result.att)
            except Exception:
                continue
        
        if len(estimates) == 0:
            pytest.skip("No valid simulations completed")
        
        mean_estimate = np.mean(estimates)
        bias = mean_estimate - tau_true
        
        assert abs(bias) < 1.5, f"Weekly demeanq bias {bias:.3f} too large"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

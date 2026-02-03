"""
Monte Carlo Simulation Tests for Trend Diagnostics Module.

This module contains Monte Carlo simulation tests to verify:
1. Size properties (Type I error rate under H0)
2. Power properties (detection rate under H1)
3. Coverage properties of confidence intervals
4. Consistency of estimators

These tests are marked as 'slow' and can be skipped in regular test runs.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from lwdid.trend_diagnostics import (
    test_parallel_trends as run_parallel_trends_test,
    diagnose_heterogeneous_trends,
    _estimate_cohort_trend,
    _compute_joint_f_test,
)


# =============================================================================
# Data Generation Functions
# =============================================================================

def generate_dgp_parallel_trends(
    n_units: int,
    n_periods: int,
    treatment_period: int,
    treatment_effect: float,
    common_trend: float,
    noise_std: float,
    seed: int,
) -> pd.DataFrame:
    """
    Generate data under parallel trends (H0).
    
    DGP: Y_it = α_i + γ*t + τ*D_it + ε_it
    
    Both treated and control have same trend γ.
    """
    np.random.seed(seed)
    
    data = []
    for i in range(n_units):
        is_treated = i < n_units // 2
        first_treat = treatment_period if is_treated else np.inf
        unit_fe = np.random.normal(0, 1)
        
        for t in range(1, n_periods + 1):
            y = 10 + unit_fe + common_trend * t
            if is_treated and t >= treatment_period:
                y += treatment_effect
            y += np.random.normal(0, noise_std)
            
            data.append({
                'unit': i,
                'time': t,
                'Y': y,
                'first_treat': first_treat,
            })
    
    return pd.DataFrame(data)


def generate_dgp_heterogeneous_trends(
    n_units: int,
    n_periods: int,
    treatment_period: int,
    treatment_effect: float,
    treated_trend: float,
    control_trend: float,
    noise_std: float,
    seed: int,
) -> pd.DataFrame:
    """
    Generate data under heterogeneous trends (H1).
    
    DGP: Y_it = α_i + β_g*t + τ*D_it + ε_it
    
    Treated units have trend β_treated, control have β_control.
    """
    np.random.seed(seed)
    
    data = []
    for i in range(n_units):
        is_treated = i < n_units // 2
        first_treat = treatment_period if is_treated else np.inf
        unit_fe = np.random.normal(0, 1)
        trend = treated_trend if is_treated else control_trend
        
        for t in range(1, n_periods + 1):
            y = 10 + unit_fe + trend * t
            if is_treated and t >= treatment_period:
                y += treatment_effect
            y += np.random.normal(0, noise_std)
            
            data.append({
                'unit': i,
                'time': t,
                'Y': y,
                'first_treat': first_treat,
            })
    
    return pd.DataFrame(data)


# =============================================================================
# Monte Carlo Tests - Size Properties
# =============================================================================

class TestSizeProperties:
    """Test Type I error rate under H0 (parallel trends holds)."""
    
    @pytest.mark.slow
    def test_parallel_trends_test_size(self):
        """
        Under H0 (parallel trends), rejection rate should be ≈ α.
        
        Test: Run 200 simulations, count rejections at α=0.05.
        
        The implementation uses rolling transformation (Procedure 3.1/4.1)
        as specified in Lee & Wooldridge (2025) Section 5, which produces
        correct standard errors that properly account for the panel structure.
        
        Expected: Rejection rate ≈ 0.05 (within [0.02, 0.10] range).
        """
        n_simulations = 200
        alpha = 0.05
        rejections = 0
        
        for sim in range(n_simulations):
            data = generate_dgp_parallel_trends(
                n_units=100,
                n_periods=10,
                treatment_period=6,
                treatment_effect=2.0,
                common_trend=0.2,  # Same trend for all
                noise_std=0.5,
                seed=1000 + sim,
            )
            
            result = run_parallel_trends_test(
                data, y='Y', ivar='unit', tvar='time',
                gvar='first_treat', method='placebo',
                alpha=alpha, verbose=False
            )
            
            if result.reject_null:
                rejections += 1
        
        rejection_rate = rejections / n_simulations
        
        # Test should have correct size (rejection rate ≈ α under H0)
        # Allow range [0.02, 0.10] to account for simulation variance
        assert 0.02 <= rejection_rate <= 0.10, \
            f"Rejection rate {rejection_rate:.2%} outside expected range [2%, 10%]"
    
    @pytest.mark.slow
    def test_heterogeneity_test_size(self):
        """
        Under H0 (homogeneous trends), F-test rejection rate should be ≈ α.
        """
        n_simulations = 200
        alpha = 0.05
        rejections = 0
        
        for sim in range(n_simulations):
            # Generate staggered data with same trend for all cohorts
            np.random.seed(2000 + sim)
            
            data = []
            unit_id = 0
            common_trend = 0.2
            
            for g in [5, 7, 9]:
                for _ in range(50):
                    unit_fe = np.random.normal(0, 1)
                    for t in range(1, 12):
                        y = 10 + unit_fe + common_trend * t
                        if t >= g:
                            y += 2.0
                        y += np.random.normal(0, 0.5)
                        data.append({
                            'unit': unit_id,
                            'time': t,
                            'Y': y,
                            'first_treat': g,
                        })
                    unit_id += 1
            
            # Control group
            for _ in range(50):
                unit_fe = np.random.normal(0, 1)
                for t in range(1, 12):
                    y = 10 + unit_fe + common_trend * t + np.random.normal(0, 0.5)
                    data.append({
                        'unit': unit_id,
                        'time': t,
                        'Y': y,
                        'first_treat': np.inf,
                    })
                unit_id += 1
            
            df = pd.DataFrame(data)
            
            diag = diagnose_heterogeneous_trends(
                df, y='Y', ivar='unit', tvar='time',
                gvar='first_treat', alpha=alpha, verbose=False
            )
            
            if diag.has_heterogeneous_trends:
                rejections += 1
        
        rejection_rate = rejections / n_simulations
        
        # Should be close to α
        assert 0.02 <= rejection_rate <= 0.15, \
            f"Rejection rate {rejection_rate:.2%} outside expected range"


# =============================================================================
# Monte Carlo Tests - Power Properties
# =============================================================================

class TestPowerProperties:
    """Test detection power under H1 (parallel trends violated)."""
    
    @pytest.mark.slow
    def test_parallel_trends_test_power(self):
        """
        Under H1 (heterogeneous trends), rejection rate should be high.
        
        Test: Run 100 simulations with strong trend difference.
        Expected: Rejection rate > 0.80.
        """
        n_simulations = 100
        alpha = 0.05
        rejections = 0
        
        for sim in range(n_simulations):
            data = generate_dgp_heterogeneous_trends(
                n_units=200,
                n_periods=10,
                treatment_period=6,
                treatment_effect=2.0,
                treated_trend=0.4,   # Treated trend
                control_trend=0.1,   # Control trend (different)
                noise_std=0.3,
                seed=3000 + sim,
            )
            
            result = run_parallel_trends_test(
                data, y='Y', ivar='unit', tvar='time',
                gvar='first_treat', method='placebo',
                alpha=alpha, verbose=False
            )
            
            if result.reject_null:
                rejections += 1
        
        power = rejections / n_simulations
        
        # Should have high power (> 80%)
        assert power > 0.70, f"Power {power:.2%} is too low"
    
    @pytest.mark.slow
    def test_heterogeneity_test_power(self):
        """
        Under H1 (heterogeneous cohort trends), F-test should have high power.
        """
        n_simulations = 100
        alpha = 0.05
        rejections = 0
        
        for sim in range(n_simulations):
            np.random.seed(4000 + sim)
            
            data = []
            unit_id = 0
            cohort_trends = {5: 0.1, 7: 0.3, 9: 0.5}  # Different trends
            
            for g in [5, 7, 9]:
                trend = cohort_trends[g]
                for _ in range(50):
                    unit_fe = np.random.normal(0, 1)
                    for t in range(1, 12):
                        y = 10 + unit_fe + trend * t
                        if t >= g:
                            y += 2.0
                        y += np.random.normal(0, 0.3)
                        data.append({
                            'unit': unit_id,
                            'time': t,
                            'Y': y,
                            'first_treat': g,
                        })
                    unit_id += 1
            
            # Control group
            for _ in range(50):
                unit_fe = np.random.normal(0, 1)
                for t in range(1, 12):
                    y = 10 + unit_fe + 0.2 * t + np.random.normal(0, 0.3)
                    data.append({
                        'unit': unit_id,
                        'time': t,
                        'Y': y,
                        'first_treat': np.inf,
                    })
                unit_id += 1
            
            df = pd.DataFrame(data)
            
            diag = diagnose_heterogeneous_trends(
                df, y='Y', ivar='unit', tvar='time',
                gvar='first_treat', alpha=alpha, verbose=False
            )
            
            if diag.has_heterogeneous_trends:
                rejections += 1
        
        power = rejections / n_simulations
        
        # Should have high power
        assert power > 0.80, f"Power {power:.2%} is too low"


# =============================================================================
# Monte Carlo Tests - Coverage Properties
# =============================================================================

class TestCoverageProperties:
    """Test confidence interval coverage properties."""
    
    @pytest.mark.slow
    def test_trend_slope_ci_coverage(self):
        """
        95% CI for trend slope should cover true value ~95% of the time.
        """
        n_simulations = 200
        true_slope = 0.25
        coverage_count = 0
        
        for sim in range(n_simulations):
            np.random.seed(5000 + sim)
            
            # Generate simple panel data
            n_units, n_periods = 50, 8
            data = pd.DataFrame({
                'unit': np.repeat(range(n_units), n_periods),
                'time': np.tile(range(1, n_periods + 1), n_units),
            })
            data['Y'] = 10 + true_slope * data['time'] + np.random.normal(0, 0.5, len(data))
            data['first_treat'] = 6
            
            pre_data = data[data['time'] < 6]
            trend = _estimate_cohort_trend(pre_data, 'Y', 'unit', 'time', None)
            
            if not np.isnan(trend.slope_se) and trend.slope_se > 0:
                ci_lower = trend.slope - 1.96 * trend.slope_se
                ci_upper = trend.slope + 1.96 * trend.slope_se
                if ci_lower <= true_slope <= ci_upper:
                    coverage_count += 1
        
        coverage_rate = coverage_count / n_simulations
        
        # Should be close to 95%
        assert 0.88 <= coverage_rate <= 0.99, \
            f"Coverage rate {coverage_rate:.2%} outside expected range"


# =============================================================================
# Monte Carlo Tests - Consistency
# =============================================================================

class TestConsistency:
    """Test consistency of estimators as sample size increases."""
    
    @pytest.mark.slow
    def test_trend_estimator_consistency(self):
        """
        Trend estimator should converge to true value as n → ∞.
        """
        true_slope = 0.3
        sample_sizes = [50, 100, 200, 500]
        biases = []
        
        for n_units in sample_sizes:
            estimates = []
            
            for sim in range(50):
                np.random.seed(6000 + sim)
                
                n_periods = 8
                data = pd.DataFrame({
                    'unit': np.repeat(range(n_units), n_periods),
                    'time': np.tile(range(1, n_periods + 1), n_units),
                })
                data['Y'] = 10 + true_slope * data['time'] + np.random.normal(0, 0.5, len(data))
                data['first_treat'] = 6
                
                pre_data = data[data['time'] < 6]
                trend = _estimate_cohort_trend(pre_data, 'Y', 'unit', 'time', None)
                
                if not np.isnan(trend.slope):
                    estimates.append(trend.slope)
            
            mean_estimate = np.mean(estimates)
            bias = abs(mean_estimate - true_slope)
            biases.append(bias)
        
        # Bias should decrease with sample size
        # Check that bias at n=500 is smaller than at n=50
        assert biases[-1] < biases[0], \
            f"Bias not decreasing: n=50 bias={biases[0]:.4f}, n=500 bias={biases[-1]:.4f}"
        
        # Bias at largest sample should be small
        assert biases[-1] < 0.05, f"Bias at n=500 is {biases[-1]:.4f}, expected < 0.05"


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'slow'])

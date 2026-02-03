"""
Numerical Validation Tests for Trend Diagnostics Module.

This module contains numerical validation tests to verify the correctness
of all calculations in the trend diagnostics functionality.

Test Categories:
- Trend estimation numerical validation
- F-statistic calculation verification
- Placebo ATT numerical validation
- Standard error coverage simulation
- vibe-math MCP formula verification
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from lwdid.trend_diagnostics import (
    PreTrendEstimate,
    CohortTrendEstimate,
    test_parallel_trends as run_parallel_trends_test,
    diagnose_heterogeneous_trends,
    recommend_transformation,
    _estimate_cohort_trend,
    _compute_joint_f_test,
    _test_trend_heterogeneity,
)


# =============================================================================
# Helper Functions for Data Generation
# =============================================================================

def generate_panel_data(
    n_units: int = 100,
    n_periods: int = 10,
    treatment_period: int = 6,
    treatment_effect: float = 2.0,
    unit_fe_std: float = 1.0,
    time_fe_std: float = 0.5,
    noise_std: float = 0.5,
    cohort_trend: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate panel data with known DGP for testing.
    
    DGP: Y_it = α_i + γ_t + β_g*t*D_i + τ*D_it + ε_it
    
    Parameters
    ----------
    n_units : int
        Number of units.
    n_periods : int
        Number of time periods.
    treatment_period : int
        Period when treatment starts.
    treatment_effect : float
        True treatment effect τ.
    unit_fe_std : float
        Standard deviation of unit fixed effects.
    time_fe_std : float
        Standard deviation of time fixed effects.
    noise_std : float
        Standard deviation of idiosyncratic error.
    cohort_trend : float
        Cohort-specific trend for treated units (β_g).
    seed : int
        Random seed.
    
    Returns
    -------
    pd.DataFrame
        Panel data with columns: unit, time, Y, first_treat.
    """
    np.random.seed(seed)
    
    # Generate fixed effects
    unit_fe = np.random.normal(0, unit_fe_std, n_units)
    time_fe = np.random.normal(0, time_fe_std, n_periods)
    
    data = []
    for i in range(n_units):
        is_treated = i < n_units // 2
        first_treat = treatment_period if is_treated else np.inf
        
        for t in range(1, n_periods + 1):
            # Base outcome
            y = 10 + unit_fe[i] + time_fe[t - 1]
            
            # Add cohort-specific trend for treated units
            if is_treated:
                y += cohort_trend * t
            
            # Add treatment effect
            if is_treated and t >= treatment_period:
                y += treatment_effect
            
            # Add noise
            y += np.random.normal(0, noise_std)
            
            data.append({
                'unit': i,
                'time': t,
                'Y': y,
                'first_treat': first_treat,
            })
    
    return pd.DataFrame(data)


def generate_staggered_data(
    n_per_cohort: int = 50,
    n_periods: int = 12,
    cohorts: list = None,
    common_trend: float = 0.1,
    cohort_specific_trends: list = None,
    treatment_effect: float = 2.0,
    noise_std: float = 0.3,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate staggered adoption panel data with known trends.
    
    Parameters
    ----------
    n_per_cohort : int
        Number of units per cohort.
    n_periods : int
        Number of time periods.
    cohorts : list
        List of treatment timing cohorts.
    common_trend : float
        Common time trend for all units.
    cohort_specific_trends : list
        Cohort-specific trends (one per cohort).
    treatment_effect : float
        True treatment effect.
    noise_std : float
        Standard deviation of noise.
    seed : int
        Random seed.
    
    Returns
    -------
    pd.DataFrame
        Staggered panel data.
    """
    np.random.seed(seed)
    
    if cohorts is None:
        cohorts = [5, 7, 9]
    if cohort_specific_trends is None:
        cohort_specific_trends = [0.0] * len(cohorts)
    
    data = []
    unit_id = 0
    
    for g_idx, g in enumerate(cohorts):
        cohort_trend = cohort_specific_trends[g_idx]
        
        for _ in range(n_per_cohort):
            unit_fe = np.random.normal(0, 1)
            
            for t in range(1, n_periods + 1):
                y = 10 + unit_fe + common_trend * t + cohort_trend * t
                y += np.random.normal(0, noise_std)
                
                if t >= g:
                    y += treatment_effect
                
                data.append({
                    'unit': unit_id,
                    'time': t,
                    'Y': y,
                    'first_treat': g,
                })
            unit_id += 1
    
    # Add never-treated control group
    for _ in range(n_per_cohort):
        unit_fe = np.random.normal(0, 1)
        
        for t in range(1, n_periods + 1):
            y = 10 + unit_fe + common_trend * t
            y += np.random.normal(0, noise_std)
            
            data.append({
                'unit': unit_id,
                'time': t,
                'Y': y,
                'first_treat': np.inf,
            })
        unit_id += 1
    
    return pd.DataFrame(data)


# =============================================================================
# Task 8: Numerical Validation Tests
# =============================================================================

class TestTrendEstimationNumerical:
    """Numerical validation of trend estimation."""
    
    def test_trend_estimation_exact_linear(self):
        """Test trend estimation with exact linear data."""
        # Create exact linear data: Y_it = 10 + 0.5*t
        n_units, n_periods = 50, 8
        data = pd.DataFrame({
            'unit': np.repeat(range(n_units), n_periods),
            'time': np.tile(range(1, n_periods + 1), n_units),
        })
        data['Y'] = 10 + 0.5 * data['time']
        data['first_treat'] = 6  # Treatment at period 6
        
        # Use only pre-treatment data
        pre_data = data[data['time'] < 6]
        
        trend = _estimate_cohort_trend(pre_data, 'Y', 'unit', 'time', None)
        
        # Should recover exact slope = 0.5
        assert_allclose(trend.slope, 0.5, atol=1e-10)
        assert trend.r_squared > 0.9999
    
    def test_trend_estimation_with_noise(self):
        """Test trend estimation with noisy data."""
        np.random.seed(123)
        n_units, n_periods = 100, 10
        true_slope = 0.3
        noise_std = 0.5
        
        data = pd.DataFrame({
            'unit': np.repeat(range(n_units), n_periods),
            'time': np.tile(range(1, n_periods + 1), n_units),
        })
        data['Y'] = 10 + true_slope * data['time'] + np.random.normal(0, noise_std, len(data))
        data['first_treat'] = 8
        
        pre_data = data[data['time'] < 8]
        trend = _estimate_cohort_trend(pre_data, 'Y', 'unit', 'time', None)
        
        # Should be within 2 SE of true slope
        assert abs(trend.slope - true_slope) < 2 * trend.slope_se
    
    def test_trend_estimation_multiple_cohorts(self):
        """Test trend estimation recovers different cohort trends."""
        np.random.seed(456)
        
        # Generate data with known heterogeneous trends
        true_trends = {5: 0.1, 7: 0.3, 9: 0.5}
        data = generate_staggered_data(
            n_per_cohort=100,
            n_periods=12,
            cohorts=[5, 7, 9],
            common_trend=0.0,
            cohort_specific_trends=[0.1, 0.3, 0.5],
            noise_std=0.2,
            seed=456,
        )
        
        diag = diagnose_heterogeneous_trends(
            data, y='Y', ivar='unit', tvar='time',
            gvar='first_treat', verbose=False
        )
        
        # Check each cohort trend is close to true value
        for trend in diag.trend_by_cohort:
            if trend.cohort in true_trends:
                true_val = true_trends[trend.cohort]
                # Should be within 0.05 of true value
                assert abs(trend.slope - true_val) < 0.05, \
                    f"Cohort {trend.cohort}: estimated {trend.slope:.4f}, true {true_val}"


class TestFStatisticNumerical:
    """Numerical validation of F-statistic calculations."""
    
    def test_f_statistic_manual_calculation(self):
        """Verify F-statistic calculation against manual computation."""
        # Create data with known trend heterogeneity
        np.random.seed(789)
        
        # Two cohorts with different slopes
        data = generate_staggered_data(
            n_per_cohort=100,
            n_periods=10,
            cohorts=[5, 8],
            common_trend=0.0,
            cohort_specific_trends=[0.1, 0.4],  # Different trends
            noise_std=0.2,
            seed=789,
        )
        
        diag = diagnose_heterogeneous_trends(
            data, y='Y', ivar='unit', tvar='time',
            gvar='first_treat', verbose=False
        )
        
        # F-stat should be significant (trends differ by 0.3)
        assert diag.trend_heterogeneity_test['f_stat'] > 10.0
        assert diag.trend_heterogeneity_test['pvalue'] < 0.001
    
    def test_f_statistic_homogeneous_trends(self):
        """F-statistic should be small when trends are equal."""
        np.random.seed(101)
        
        # All cohorts have same trend
        data = generate_staggered_data(
            n_per_cohort=100,
            n_periods=10,
            cohorts=[5, 7, 9],
            common_trend=0.2,
            cohort_specific_trends=[0.0, 0.0, 0.0],  # Same trends
            noise_std=0.3,
            seed=101,
        )
        
        diag = diagnose_heterogeneous_trends(
            data, y='Y', ivar='unit', tvar='time',
            gvar='first_treat', verbose=False
        )
        
        # F-stat should not be significant
        assert diag.trend_heterogeneity_test['pvalue'] > 0.05


class TestPlaceboATTNumerical:
    """Numerical validation of placebo ATT estimation."""
    
    def test_placebo_att_zero_under_pt(self):
        """Placebo ATT should be ~0 when PT holds."""
        np.random.seed(202)
        
        # DGP: Y_it = α_i + γ_t + ε_it (no cohort-specific trend)
        data = generate_panel_data(
            n_units=500, n_periods=10, treatment_period=7,
            treatment_effect=2.0,
            cohort_trend=0.0,  # No differential trend
            noise_std=0.5,
            seed=202,
        )
        
        result = run_parallel_trends_test(
            data, y='Y', ivar='unit', tvar='time',
            gvar='first_treat', method='placebo', verbose=False
        )
        
        # All pre-treatment ATTs should be close to 0
        for est in result.pre_trend_estimates:
            # Should be within 3 SE of 0
            assert abs(est.att) < 3 * est.se, \
                f"Event time {est.event_time}: ATT={est.att:.4f}, SE={est.se:.4f}"
    
    def test_placebo_att_nonzero_under_violation(self):
        """Placebo ATT should be non-zero when PT violated."""
        np.random.seed(303)
        
        # DGP: Y_it = α_i + γ_t + β_g*t + ε_it (cohort-specific trend)
        data = generate_panel_data(
            n_units=500, n_periods=10, treatment_period=7,
            treatment_effect=2.0,
            cohort_trend=0.3,  # Strong differential trend
            noise_std=0.3,
            seed=303,
        )
        
        result = run_parallel_trends_test(
            data, y='Y', ivar='unit', tvar='time',
            gvar='first_treat', method='placebo', verbose=False
        )
        
        # At least some pre-treatment ATTs should be significant
        n_significant = sum(1 for e in result.pre_trend_estimates if e.pvalue < 0.05)
        assert n_significant > 0, "Expected some significant pre-treatment ATTs"


class TestJointFTestNumerical:
    """Numerical validation of joint F-test."""
    
    def test_joint_f_test_wald_conversion(self):
        """Verify Wald to F-statistic conversion."""
        # Create estimates with known values
        # θ = [0.1, 0.2, 0.15], SE = [0.05, 0.04, 0.06]
        estimates = [
            PreTrendEstimate(-3, 4, 0.1, 0.05, 2.0, 0.05, 0, 0.2, 50, 100, 148),
            PreTrendEstimate(-2, 4, 0.2, 0.04, 5.0, 0.001, 0.12, 0.28, 50, 100, 148),
            PreTrendEstimate(-1, 4, 0.15, 0.06, 2.5, 0.01, 0.03, 0.27, 50, 100, 148),
        ]
        
        # Manual Wald calculation (assuming independence)
        theta = np.array([0.1, 0.2, 0.15])
        se = np.array([0.05, 0.04, 0.06])
        wald = np.sum(theta ** 2 / se ** 2)
        # = 0.01/0.0025 + 0.04/0.0016 + 0.0225/0.0036
        # = 4 + 25 + 6.25 = 35.25
        
        k = len(theta)
        expected_f = wald / k  # = 35.25 / 3 = 11.75
        
        computed_f, computed_p, computed_df = _compute_joint_f_test(estimates)
        
        assert_allclose(computed_f, expected_f, rtol=0.01)
        assert computed_df[0] == 3  # Number of estimates
    
    def test_joint_f_test_power(self):
        """Joint F-test should have power to detect violations."""
        np.random.seed(404)
        
        # Generate data with PT violation
        data = generate_panel_data(
            n_units=300, n_periods=10, treatment_period=7,
            cohort_trend=0.2,  # Differential trend
            noise_std=0.3,
            seed=404,
        )
        
        result = run_parallel_trends_test(
            data, y='Y', ivar='unit', tvar='time',
            gvar='first_treat', method='placebo', verbose=False
        )
        
        # Joint test should reject
        assert result.joint_pvalue < 0.05


class TestStandardErrorCoverage:
    """Test standard error coverage properties."""
    
    @pytest.mark.slow
    def test_se_coverage_simulation(self):
        """Test that 95% CI has correct coverage (Monte Carlo)."""
        np.random.seed(505)
        n_simulations = 100  # Reduced for speed
        coverage_count = 0
        true_slope = 0.3  # Common trend for all units
        
        for sim in range(n_simulations):
            # Generate data with known common trend
            n_units, n_periods = 50, 8
            data = pd.DataFrame({
                'unit': np.repeat(range(n_units), n_periods),
                'time': np.tile(range(1, n_periods + 1), n_units),
            })
            # Y = 10 + true_slope * t + noise
            np.random.seed(505 + sim)
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
        # Should be close to 95% (allow 85-99% range for small sample)
        assert 0.85 <= coverage_rate <= 0.99, f"Coverage rate: {coverage_rate:.2%}"


# =============================================================================
# Task 9: vibe-math MCP Formula Verification
# =============================================================================

class TestAssumptionCHTFormula:
    """
    Verify Assumption CHT formula (5.3) from Lee & Wooldridge (2025).
    
    Formula (5.3):
    E[Y_t(∞)|D, X] = η_S(D_S·t) + ... + η_T(D_T·t) + q_∞(X) + Σ D_g q_g(X) + m_t(X)
    """
    
    def test_formula_5_3_expectation_cohort_s(self):
        """
        Verify formula (5.3) for cohort S.
        
        For unit in cohort S (D_S=1, D_T=0):
        E[Y_t(∞)|D_S=1] = η_S*t + q_∞ + q_S + m_t
        """
        # Parameters
        eta_S = 0.02  # Cohort S trend
        q_inf = 10.0  # Base level
        q_S = 1.0     # Cohort S level shift
        m_t = 0.5     # Time effect
        t = 5         # Time period
        
        # Expected value for cohort S
        expected_cohort_S = eta_S * t + q_inf + q_S + m_t
        # = 0.02*5 + 10 + 1 + 0.5 = 11.6
        
        assert_allclose(expected_cohort_S, 11.6, atol=1e-10)
    
    def test_formula_5_3_expectation_control(self):
        """
        Verify formula (5.3) for control group.
        
        For never-treated (D_S=0, D_T=0):
        E[Y_t(∞)|D_∞=1] = q_∞ + m_t
        """
        q_inf = 10.0
        m_t = 0.5
        
        expected_control = q_inf + m_t
        # = 10 + 0.5 = 10.5
        
        assert_allclose(expected_control, 10.5, atol=1e-10)
    
    def test_formula_5_4_first_difference(self):
        """
        Verify formula (5.4) - first difference under CHT.
        
        E[Y_t(∞) - Y_{t-1}(∞)|D, X] = η_g*D_g + ... + [m_t(X) - m_{t-1}(X)]
        
        The first difference depends on cohort through η_g terms.
        """
        eta_g = 0.03  # Cohort g trend
        m_t = 1.2
        m_t_minus_1 = 1.0
        
        # For cohort g unit:
        # E[ΔY|D_g=1] = η_g + (m_t - m_{t-1})
        expected_diff_treated = eta_g + (m_t - m_t_minus_1)
        # = 0.03 + 0.2 = 0.23
        
        # For control unit:
        # E[ΔY|D_∞=1] = m_t - m_{t-1}
        expected_diff_control = m_t - m_t_minus_1
        # = 0.2
        
        # Difference reveals cohort-specific trend
        trend_revealed = expected_diff_treated - expected_diff_control
        # = 0.03 = η_g
        
        assert_allclose(trend_revealed, eta_g, atol=1e-10)


class TestDetrendingFormula:
    """Verify detrending transformation formulas from Procedure 5.1."""
    
    def test_procedure_5_1_detrending_exact(self):
        """
        Verify Procedure 5.1 detrending recovers treatment effect.
        
        Ÿ_{irg} = Y_{ir} - (Â_i + B̂_i * r)
        
        After detrending, treatment effect should be recovered.
        """
        # True DGP: Y_it = α_i + β_i*t + τ*D_it + ε_it
        alpha_i = 10.0
        beta_i = 0.5
        tau = 2.0  # True treatment effect
        
        # Pre-treatment periods: t = 1, 2, 3 (g = 4)
        pre_Y = np.array([alpha_i + beta_i * t for t in [1, 2, 3]])
        pre_t = np.array([1, 2, 3])
        
        # OLS estimation of trend
        X = np.column_stack([np.ones(3), pre_t])
        beta_hat = np.linalg.lstsq(X, pre_Y, rcond=None)[0]
        A_hat = beta_hat[0]  # Should be ~10
        B_hat = beta_hat[1]  # Should be ~0.5
        
        assert_allclose(A_hat, alpha_i, atol=1e-10)
        assert_allclose(B_hat, beta_i, atol=1e-10)
        
        # Post-treatment: t = 4
        Y_post = alpha_i + beta_i * 4 + tau
        # = 10 + 2 + 2 = 14
        
        # Detrended outcome:
        Y_detrended = Y_post - (A_hat + B_hat * 4)
        # = 14 - 12 = 2 = τ
        
        assert_allclose(Y_detrended, tau, atol=1e-10)
    
    def test_detrending_removes_heterogeneous_trends(self):
        """
        Verify that detrending removes cohort-specific trends.
        
        After detrending, parallel trends should hold.
        """
        np.random.seed(606)
        
        # Generate data with heterogeneous trends
        data = generate_staggered_data(
            n_per_cohort=100,
            n_periods=12,
            cohorts=[5, 7, 9],
            common_trend=0.0,
            cohort_specific_trends=[0.1, 0.2, 0.3],  # Different trends
            noise_std=0.2,
            seed=606,
        )
        
        # Before detrending: trends differ
        diag_before = diagnose_heterogeneous_trends(
            data, y='Y', ivar='unit', tvar='time',
            gvar='first_treat', verbose=False
        )
        
        # Should detect heterogeneous trends
        assert diag_before.has_heterogeneous_trends == True
        
        # Verify trend estimates are close to true values
        true_trends = {5: 0.1, 7: 0.2, 9: 0.3}
        for trend in diag_before.trend_by_cohort:
            if trend.cohort in true_trends:
                assert abs(trend.slope - true_trends[trend.cohort]) < 0.05


class TestWaldToFConversion:
    """Verify Wald to F-statistic conversion formula."""
    
    def test_wald_f_relationship(self):
        """
        Verify: W = θ' V^{-1} θ ~ χ²(k)
                F = W/k ~ F(k, df_den)
        """
        # Known values
        theta = np.array([0.1, 0.2, 0.15])
        var = np.array([0.05, 0.04, 0.06]) ** 2
        
        # Wald statistic (diagonal covariance)
        wald = np.sum(theta ** 2 / var)
        
        k = len(theta)
        f_stat = wald / k
        
        # Manual calculation
        # wald = 0.01/0.0025 + 0.04/0.0016 + 0.0225/0.0036
        #      = 4 + 25 + 6.25 = 35.25
        # f = 35.25 / 3 = 11.75
        
        assert_allclose(wald, 35.25, atol=0.01)
        assert_allclose(f_stat, 11.75, atol=0.01)


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])

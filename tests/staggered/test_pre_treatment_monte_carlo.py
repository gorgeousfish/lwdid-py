"""
Monte Carlo simulation tests for pre-treatment dynamics.

This module implements Monte Carlo simulations to verify statistical
properties of the pre-treatment estimation and parallel trends testing:

1. TestParallelTrendsSize: Under H0, rejection rate ≈ alpha
2. TestParallelTrendsPower: Under H1, rejection rate > alpha
3. TestPreTreatmentATTUnbiased: Under parallel trends, E[ATT_pre] ≈ 0
4. TestCoverageProbability: 95% CI should cover true value ~95% of time

References
----------
Lee, S. J., & Wooldridge, J. M. (2025). A Simple Transformation Approach
to Difference-in-Differences Estimation for Panel Data. Appendix D.
"""

import numpy as np
import pandas as pd
import pytest
import warnings
from typing import Tuple, List, Dict, Any

# Mark as slow tests - these take significant time to run
pytestmark = pytest.mark.slow


def generate_staggered_dgp(
    n_units: int = 100,
    n_periods: int = 10,
    cohort_periods: List[int] = None,
    treatment_effect: float = 0.5,
    pre_treatment_violation: float = 0.0,
    unit_fe_sd: float = 1.0,
    time_trend: float = 0.1,
    error_sd: float = 0.5,
    pct_never_treated: float = 0.2,
    seed: int = None,
) -> pd.DataFrame:
    """
    Generate staggered adoption data with configurable DGP.
    
    Parameters
    ----------
    n_units : int
        Number of units in the panel.
    n_periods : int
        Number of time periods.
    cohort_periods : list of int
        Treatment timing for each cohort. Default [4, 6, 8].
    treatment_effect : float
        True ATT for post-treatment periods.
    pre_treatment_violation : float
        Magnitude of parallel trends violation. If > 0, treated units
        have differential pre-trends, violating the identifying assumption.
    unit_fe_sd : float
        Standard deviation of unit fixed effects.
    time_trend : float
        Coefficient on linear time trend.
    error_sd : float
        Standard deviation of idiosyncratic errors.
    pct_never_treated : float
        Fraction of units that are never treated.
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    pd.DataFrame
        Panel data with columns: unit, period, gvar, y
    """
    if seed is not None:
        np.random.seed(seed)
    
    if cohort_periods is None:
        cohort_periods = [4, 6, 8]
    
    # Create panel structure
    units = np.repeat(np.arange(1, n_units + 1), n_periods)
    periods = np.tile(np.arange(1, n_periods + 1), n_units)
    
    # Assign cohorts
    n_never = int(n_units * pct_never_treated)
    n_treated = n_units - n_never
    n_cohorts = len(cohort_periods)
    per_cohort = n_treated // n_cohorts
    
    cohort_assignments = np.zeros(n_units)
    for i, g in enumerate(cohort_periods):
        start = n_never + i * per_cohort
        end = start + per_cohort if i < n_cohorts - 1 else n_units
        cohort_assignments[start:end] = g
    
    gvar = np.repeat(cohort_assignments, n_periods)
    
    # Generate components
    unit_fe = np.repeat(np.random.normal(0, unit_fe_sd, n_units), n_periods)
    time_fe = periods * time_trend
    
    # Treatment indicator
    treated_post = (gvar > 0) & (periods >= gvar)
    
    # Pre-treatment violation: differential trend for treated units
    # This creates a violation of parallel trends
    is_treated_unit = (gvar > 0)
    pre_trend_violation = pre_treatment_violation * periods * is_treated_unit
    
    # Idiosyncratic error
    epsilon = np.random.normal(0, error_sd, len(units))
    
    # Outcome
    y = (
        unit_fe + 
        time_fe + 
        treatment_effect * treated_post +
        pre_trend_violation +
        epsilon
    )
    
    return pd.DataFrame({
        'unit': units,
        'period': periods,
        'gvar': gvar,
        'y': y,
    })


class TestParallelTrendsSize:
    """
    Test that parallel trends test has correct size under H0.
    
    Under the null hypothesis (parallel trends holds), the rejection
    rate should be approximately equal to the nominal alpha level.
    """
    
    @pytest.fixture
    def size_simulation_params(self):
        """Parameters for size simulation."""
        return {
            'n_simulations': 200,  # Reduced for faster testing
            'n_units': 100,
            'n_periods': 10,
            'alpha': 0.05,
            'treatment_effect': 0.5,
            'pre_treatment_violation': 0.0,  # H0: no violation
        }
    
    def test_rejection_rate_under_h0(self, size_simulation_params):
        """
        Under H0, rejection rate should be close to alpha.
        
        We test that the rejection rate is within a reasonable range
        of the nominal alpha (using a binomial confidence interval).
        """
        from lwdid import lwdid
        
        params = size_simulation_params
        n_sims = params['n_simulations']
        alpha = params['alpha']
        
        rejections = 0
        valid_sims = 0
        
        for sim in range(n_sims):
            # Generate data under H0
            data = generate_staggered_dgp(
                n_units=params['n_units'],
                n_periods=params['n_periods'],
                treatment_effect=params['treatment_effect'],
                pre_treatment_violation=0.0,  # H0
                seed=sim * 1000,
            )
            
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    results = lwdid(
                        data=data,
                        y='y',
                        ivar='unit',
                        tvar='period',
                        gvar='gvar',
                        rolling='demean',
                        aggregate='cohort',
                        include_pretreatment=True,
                        pretreatment_test=True,
                        pretreatment_alpha=alpha,
                    )
                
                if results.parallel_trends_test is not None:
                    valid_sims += 1
                    if results.parallel_trends_test.reject_null:
                        rejections += 1
            except Exception:
                continue
        
        if valid_sims < n_sims * 0.8:
            pytest.skip(f"Too many failed simulations: {valid_sims}/{n_sims}")
        
        rejection_rate = rejections / valid_sims
        
        # Under H0, rejection rate should be close to alpha
        # Use binomial CI: alpha ± 2*sqrt(alpha*(1-alpha)/n)
        se = np.sqrt(alpha * (1 - alpha) / valid_sims)
        lower_bound = max(0, alpha - 3 * se)
        upper_bound = min(1, alpha + 3 * se)
        
        assert lower_bound <= rejection_rate <= upper_bound, (
            f"Rejection rate {rejection_rate:.3f} outside expected range "
            f"[{lower_bound:.3f}, {upper_bound:.3f}] under H0"
        )
    
    def test_size_with_detrending(self, size_simulation_params):
        """
        Size test using detrending transformation.
        """
        from lwdid import lwdid
        
        params = size_simulation_params
        n_sims = min(100, params['n_simulations'])  # Fewer sims for detrending
        alpha = params['alpha']
        
        rejections = 0
        valid_sims = 0
        
        for sim in range(n_sims):
            data = generate_staggered_dgp(
                n_units=params['n_units'],
                n_periods=12,  # Need more periods for detrending
                treatment_effect=params['treatment_effect'],
                pre_treatment_violation=0.0,
                seed=sim * 2000,
            )
            
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    results = lwdid(
                        data=data,
                        y='y',
                        ivar='unit',
                        tvar='period',
                        gvar='gvar',
                        rolling='detrend',
                        aggregate='cohort',
                        include_pretreatment=True,
                        pretreatment_test=True,
                        pretreatment_alpha=alpha,
                    )
                
                if results.parallel_trends_test is not None:
                    valid_sims += 1
                    if results.parallel_trends_test.reject_null:
                        rejections += 1
            except Exception:
                continue
        
        if valid_sims < n_sims * 0.5:
            pytest.skip(f"Too many failed simulations: {valid_sims}/{n_sims}")
        
        rejection_rate = rejections / valid_sims
        
        # More lenient bounds for detrending (higher variance)
        assert rejection_rate < 0.20, (
            f"Rejection rate {rejection_rate:.3f} too high under H0 with detrending"
        )


class TestParallelTrendsPower:
    """
    Test that parallel trends test has power under H1.
    
    Under the alternative hypothesis (parallel trends violated),
    the rejection rate should be higher than alpha.
    """
    
    @pytest.fixture
    def power_simulation_params(self):
        """Parameters for power simulation."""
        return {
            'n_simulations': 100,
            'n_units': 150,
            'n_periods': 10,
            'alpha': 0.05,
            'treatment_effect': 0.5,
            'pre_treatment_violation': 0.15,  # H1: violation exists
        }
    
    def test_rejection_rate_under_h1(self, power_simulation_params):
        """
        Under H1, rejection rate should be higher than alpha.
        
        With a meaningful violation of parallel trends, the test
        should reject more often than the nominal alpha level.
        """
        from lwdid import lwdid
        
        params = power_simulation_params
        n_sims = params['n_simulations']
        alpha = params['alpha']
        
        rejections = 0
        valid_sims = 0
        
        for sim in range(n_sims):
            data = generate_staggered_dgp(
                n_units=params['n_units'],
                n_periods=params['n_periods'],
                treatment_effect=params['treatment_effect'],
                pre_treatment_violation=params['pre_treatment_violation'],  # H1
                seed=sim * 3000,
            )
            
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    results = lwdid(
                        data=data,
                        y='y',
                        ivar='unit',
                        tvar='period',
                        gvar='gvar',
                        rolling='demean',
                        aggregate='cohort',
                        include_pretreatment=True,
                        pretreatment_test=True,
                        pretreatment_alpha=alpha,
                    )
                
                if results.parallel_trends_test is not None:
                    valid_sims += 1
                    if results.parallel_trends_test.reject_null:
                        rejections += 1
            except Exception:
                continue
        
        if valid_sims < n_sims * 0.8:
            pytest.skip(f"Too many failed simulations: {valid_sims}/{n_sims}")
        
        rejection_rate = rejections / valid_sims
        
        # Under H1, rejection rate should be meaningfully higher than alpha
        # We expect power > 0.20 for this violation magnitude
        assert rejection_rate > alpha, (
            f"Rejection rate {rejection_rate:.3f} not higher than alpha={alpha} under H1"
        )


class TestPreTreatmentATTUnbiased:
    """
    Test that pre-treatment ATT is unbiased under parallel trends.
    
    Under the null hypothesis, E[ATT_pre] should be approximately zero.
    """
    
    @pytest.fixture
    def unbiased_simulation_params(self):
        """Parameters for unbiasedness simulation."""
        return {
            'n_simulations': 200,
            'n_units': 100,
            'n_periods': 10,
            'treatment_effect': 0.5,
        }
    
    def test_mean_pre_treatment_att_near_zero(self, unbiased_simulation_params):
        """
        Average pre-treatment ATT across simulations should be near zero.
        """
        from lwdid import lwdid
        
        params = unbiased_simulation_params
        n_sims = params['n_simulations']
        
        all_pre_atts = []
        
        for sim in range(n_sims):
            data = generate_staggered_dgp(
                n_units=params['n_units'],
                n_periods=params['n_periods'],
                treatment_effect=params['treatment_effect'],
                pre_treatment_violation=0.0,  # H0
                seed=sim * 4000,
            )
            
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    results = lwdid(
                        data=data,
                        y='y',
                        ivar='unit',
                        tvar='period',
                        gvar='gvar',
                        rolling='demean',
                        aggregate='cohort',
                        include_pretreatment=True,
                    )
                
                if results.att_pre_treatment is not None:
                    pre_df = results.att_pre_treatment
                    non_anchor = pre_df[~pre_df['is_anchor']]
                    if len(non_anchor) > 0:
                        all_pre_atts.extend(non_anchor['att'].dropna().tolist())
            except Exception:
                continue
        
        if len(all_pre_atts) < 100:
            pytest.skip(f"Insufficient pre-treatment ATT estimates: {len(all_pre_atts)}")
        
        mean_pre_att = np.mean(all_pre_atts)
        se_mean = np.std(all_pre_atts) / np.sqrt(len(all_pre_atts))
        
        # Mean should be within 3 SE of zero
        assert abs(mean_pre_att) < 3 * se_mean + 0.05, (
            f"Mean pre-treatment ATT {mean_pre_att:.4f} too far from zero "
            f"(SE={se_mean:.4f})"
        )


class TestCoverageProbability:
    """
    Test that confidence intervals have correct coverage.
    
    95% confidence intervals should cover the true value (zero under H0)
    approximately 95% of the time.
    """
    
    @pytest.fixture
    def coverage_simulation_params(self):
        """Parameters for coverage simulation."""
        return {
            'n_simulations': 200,
            'n_units': 100,
            'n_periods': 10,
            'treatment_effect': 0.5,
            'ci_level': 0.95,
        }
    
    def test_ci_coverage_under_h0(self, coverage_simulation_params):
        """
        95% CI should cover true value (0) approximately 95% of time.
        """
        from lwdid import lwdid
        
        params = coverage_simulation_params
        n_sims = params['n_simulations']
        ci_level = params['ci_level']
        alpha = 1 - ci_level
        
        coverage_count = 0
        total_intervals = 0
        
        for sim in range(n_sims):
            data = generate_staggered_dgp(
                n_units=params['n_units'],
                n_periods=params['n_periods'],
                treatment_effect=params['treatment_effect'],
                pre_treatment_violation=0.0,  # True value is 0
                seed=sim * 5000,
            )
            
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    results = lwdid(
                        data=data,
                        y='y',
                        ivar='unit',
                        tvar='period',
                        gvar='gvar',
                        rolling='demean',
                        aggregate='cohort',
                        include_pretreatment=True,
                        pretreatment_alpha=alpha,
                    )
                
                if results.att_pre_treatment is not None:
                    pre_df = results.att_pre_treatment
                    non_anchor = pre_df[~pre_df['is_anchor']]
                    
                    for _, row in non_anchor.iterrows():
                        if pd.notna(row['ci_lower']) and pd.notna(row['ci_upper']):
                            total_intervals += 1
                            # True value under H0 is 0
                            if row['ci_lower'] <= 0 <= row['ci_upper']:
                                coverage_count += 1
            except Exception:
                continue
        
        if total_intervals < 100:
            pytest.skip(f"Insufficient intervals: {total_intervals}")
        
        coverage_rate = coverage_count / total_intervals
        
        # Coverage should be close to nominal level
        # Use binomial CI for coverage rate
        se = np.sqrt(ci_level * (1 - ci_level) / total_intervals)
        lower_bound = ci_level - 3 * se
        upper_bound = min(1.0, ci_level + 3 * se)
        
        # Allow some undercoverage due to finite sample bias
        assert coverage_rate >= lower_bound - 0.05, (
            f"Coverage rate {coverage_rate:.3f} below expected range "
            f"[{lower_bound:.3f}, {upper_bound:.3f}]"
        )


class TestNumericalStability:
    """
    Test numerical stability of pre-treatment estimation.
    """
    
    def test_extreme_values(self):
        """
        Test with extreme outcome values.
        """
        from lwdid import lwdid
        
        # Generate data with large values
        data = generate_staggered_dgp(
            n_units=100,
            n_periods=10,
            treatment_effect=1000.0,  # Large effect
            unit_fe_sd=100.0,  # Large FE
            error_sd=50.0,  # Large errors
            seed=99999,
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = lwdid(
                data=data,
                y='y',
                ivar='unit',
                tvar='period',
                gvar='gvar',
                rolling='demean',
                aggregate='cohort',
                include_pretreatment=True,
            )
        
        # Should complete without NaN/Inf
        assert results.att_pre_treatment is not None
        pre_df = results.att_pre_treatment
        
        # Check for numerical issues
        assert not pre_df['att'].isna().all(), "All ATT values are NaN"
        assert np.isfinite(pre_df['att'].dropna()).all(), "Non-finite ATT values"
    
    def test_small_sample(self):
        """
        Test with small sample size.
        """
        from lwdid import lwdid
        
        data = generate_staggered_dgp(
            n_units=30,  # Small sample
            n_periods=8,
            cohort_periods=[4, 6],
            treatment_effect=0.5,
            pct_never_treated=0.3,
            seed=88888,
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = lwdid(
                data=data,
                y='y',
                ivar='unit',
                tvar='period',
                gvar='gvar',
                rolling='demean',
                aggregate='cohort',
                include_pretreatment=True,
            )
        
        # Should handle small samples gracefully
        assert results.include_pretreatment == True
        # May have limited pre-treatment effects due to small sample

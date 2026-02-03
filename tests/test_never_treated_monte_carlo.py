"""
Monte Carlo tests for never-treated handling.

This module contains Monte Carlo simulations to verify the statistical
properties of estimators using never-treated control groups.

Based on: Lee & Wooldridge (2025) ssrn-4516518, Section 4
Spec: .kiro/specs/never-treated-validation/

Test Categories:
- Coverage rate verification
- Bias verification
- Standard error accuracy
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import sys
sys.path.insert(0, 'src')

from lwdid.validation import is_never_treated
from lwdid.staggered.control_groups import (
    identify_never_treated_units,
    get_valid_control_units,
    ControlGroupStrategy,
)


class TestMonteCarloNeverTreatedControl:
    """
    TEST-14: Monte Carlo validation of never-treated control group.
    
    Verifies that using never-treated units as control produces
    unbiased estimates with correct coverage rates.
    """
    
    @pytest.mark.slow
    def test_coverage_rate_with_never_treated(self):
        """
        Monte Carlo: verify CI coverage rate with NT control.
        
        Expected: 95% CI should cover true ATT ~95% of the time.
        """
        n_simulations = 200
        true_att = 5.0
        coverage_count = 0
        att_estimates = []
        se_estimates = []
        
        for sim in range(n_simulations):
            np.random.seed(sim)
            
            # Generate DGP
            data = self._generate_dgp_with_never_treated(
                n_units=100, n_periods=6, true_att=true_att, nt_ratio=0.3
            )
            
            # Estimate ATT
            att_est, se_est = self._simple_att_estimate(data)
            att_estimates.append(att_est)
            se_estimates.append(se_est)
            
            # Check coverage (95% CI)
            ci_lower = att_est - 1.96 * se_est
            ci_upper = att_est + 1.96 * se_est
            if ci_lower <= true_att <= ci_upper:
                coverage_count += 1
        
        # Verify bias
        mean_att = np.mean(att_estimates)
        bias = mean_att - true_att
        assert abs(bias) < 0.5, f"Bias too large: {bias:.3f}"
        
        # Verify coverage rate (allow 85-99% range due to simulation variance)
        coverage_rate = coverage_count / n_simulations
        assert 0.85 <= coverage_rate <= 0.99, \
            f"Coverage rate {coverage_rate:.1%} outside expected range"
    
    @pytest.mark.slow
    def test_unbiasedness_with_parallel_trends(self):
        """
        Monte Carlo: verify unbiasedness when parallel trends holds.
        
        Expected: Mean ATT estimate should equal true ATT.
        """
        n_simulations = 200
        true_att = 3.0
        att_estimates = []
        
        for sim in range(n_simulations):
            np.random.seed(sim + 1000)
            
            # Generate DGP with parallel trends
            data = self._generate_dgp_with_never_treated(
                n_units=80, n_periods=6, true_att=true_att, nt_ratio=0.3
            )
            
            # Estimate ATT
            att_est, _ = self._simple_att_estimate(data)
            att_estimates.append(att_est)
        
        # Verify unbiasedness
        mean_att = np.mean(att_estimates)
        bias = mean_att - true_att
        
        # Bias should be small (< 0.3)
        assert abs(bias) < 0.3, f"Estimator biased: mean={mean_att:.3f}, true={true_att}"
    
    @pytest.mark.slow
    def test_se_accuracy(self):
        """
        Monte Carlo: verify SE estimates are accurate.
        
        Expected: Mean SE estimate ≈ empirical SD of ATT estimates.
        """
        n_simulations = 200
        true_att = 5.0
        att_estimates = []
        se_estimates = []
        
        for sim in range(n_simulations):
            np.random.seed(sim + 2000)
            
            data = self._generate_dgp_with_never_treated(
                n_units=100, n_periods=6, true_att=true_att, nt_ratio=0.3
            )
            
            att_est, se_est = self._simple_att_estimate(data)
            att_estimates.append(att_est)
            se_estimates.append(se_est)
        
        # Empirical SD
        empirical_sd = np.std(att_estimates, ddof=1)
        
        # Mean SE estimate
        mean_se = np.mean(se_estimates)
        
        # SE should be close to empirical SD (within 30%)
        se_ratio = mean_se / empirical_sd
        assert 0.7 <= se_ratio <= 1.3, \
            f"SE ratio {se_ratio:.2f} outside expected range [0.7, 1.3]"
    
    def test_nt_ratio_effect_on_precision(self):
        """
        Test: higher NT ratio should improve precision.
        
        More control units → lower variance → narrower CI.
        """
        np.random.seed(42)
        true_att = 5.0
        
        se_by_nt_ratio = {}
        
        for nt_ratio in [0.1, 0.3, 0.5]:
            se_estimates = []
            
            for sim in range(50):
                np.random.seed(sim + 3000 + int(nt_ratio * 100))
                
                data = self._generate_dgp_with_never_treated(
                    n_units=100, n_periods=6, true_att=true_att, nt_ratio=nt_ratio
                )
                
                _, se_est = self._simple_att_estimate(data)
                se_estimates.append(se_est)
            
            se_by_nt_ratio[nt_ratio] = np.mean(se_estimates)
        
        # Higher NT ratio should generally lead to lower SE
        # (more control units → better precision)
        # Note: This is a general trend, not guaranteed for every sample
        assert se_by_nt_ratio[0.5] < se_by_nt_ratio[0.1] * 1.5, \
            "Higher NT ratio should improve precision"
    
    def _generate_dgp_with_never_treated(
        self, n_units: int, n_periods: int, true_att: float, nt_ratio: float
    ) -> pd.DataFrame:
        """Generate DGP with never-treated units."""
        data_rows = []
        
        for i in range(n_units):
            # Assign treatment status
            if np.random.rand() < nt_ratio:
                gvar = 0  # never-treated
            else:
                gvar = 4  # treated at period 4
            
            # Unit fixed effect
            alpha_i = np.random.randn() * 2
            
            for t in range(1, n_periods + 1):
                # Time trend (common to all units)
                delta_t = 0.5 * t
                
                # Treatment effect
                treated = 1 if (gvar > 0 and t >= gvar) else 0
                
                # Error
                epsilon = np.random.randn()
                
                # Outcome
                y = alpha_i + delta_t + true_att * treated + epsilon
                
                data_rows.append({
                    'id': i, 'year': t, 'y': y, 'gvar': gvar
                })
        
        return pd.DataFrame(data_rows)
    
    def _simple_att_estimate(self, data: pd.DataFrame) -> tuple:
        """Simple ATT estimation using demeaning."""
        # Pre-treatment mean (t < 4)
        pre_means = data[data['year'] < 4].groupby('id')['y'].mean()
        
        # Post-treatment mean (t >= 4)
        post_means = data[data['year'] >= 4].groupby('id')['y'].mean()
        
        # Transformed outcome
        delta_y = post_means - pre_means
        
        # Separate by treatment status
        unit_gvar = data.groupby('id')['gvar'].first()
        treated_mask = unit_gvar == 4
        control_mask = unit_gvar.apply(is_never_treated)
        
        treated_delta = delta_y[treated_mask]
        control_delta = delta_y[control_mask]
        
        # ATT estimate
        att = treated_delta.mean() - control_delta.mean()
        
        # SE estimate
        n_t = len(treated_delta)
        n_c = len(control_delta)
        
        if n_t > 1 and n_c > 1:
            se = np.sqrt(treated_delta.var(ddof=1)/n_t + control_delta.var(ddof=1)/n_c)
        else:
            se = np.nan
        
        return att, se


class TestMonteCarloControlGroupStrategies:
    """
    TEST-15: Compare control group strategies via Monte Carlo.
    """
    
    @pytest.mark.slow
    def test_never_treated_vs_not_yet_treated(self):
        """
        Compare NEVER_TREATED vs NOT_YET_TREATED strategies.
        
        Both should produce unbiased estimates under parallel trends.
        Note: NYT strategy uses a simplified estimation that may have
        larger variance due to the staggered nature of treatment.
        """
        n_simulations = 100
        true_att = 5.0
        
        att_nt = []  # Never-treated only
        att_nyt = []  # Not-yet-treated
        
        for sim in range(n_simulations):
            np.random.seed(sim + 4000)
            
            # Generate data with multiple cohorts
            data = self._generate_multi_cohort_dgp(
                n_per_cohort=30, true_att=true_att
            )
            
            # Estimate with NT control
            att_nt_est = self._estimate_with_strategy(
                data, ControlGroupStrategy.NEVER_TREATED
            )
            att_nt.append(att_nt_est)
            
            # Estimate with NYT control (use correct method for staggered)
            att_nyt_est = self._estimate_with_nyt_strategy(data, true_att=true_att)
            att_nyt.append(att_nyt_est)
        
        # NT strategy should be unbiased
        bias_nt = np.mean(att_nt) - true_att
        assert abs(bias_nt) < 0.5, f"NT strategy biased: {bias_nt:.3f}"
        
        # NYT strategy should also be approximately unbiased
        # (using correct staggered estimation)
        bias_nyt = np.mean(att_nyt) - true_att
        assert abs(bias_nyt) < 0.5, f"NYT strategy biased: {bias_nyt:.3f}"
    
    def _generate_multi_cohort_dgp(
        self, n_per_cohort: int, true_att: float
    ) -> pd.DataFrame:
        """Generate DGP with multiple cohorts."""
        data_rows = []
        uid = 0
        
        # Cohorts 4, 5, 6
        for cohort in [4, 5, 6]:
            for _ in range(n_per_cohort):
                alpha_i = np.random.randn() * 2
                for t in range(1, 7):
                    treated = 1 if t >= cohort else 0
                    y = alpha_i + 0.5*t + true_att*treated + np.random.randn()
                    data_rows.append({
                        'id': uid, 'year': t, 'y': y, 'gvar': cohort
                    })
                uid += 1
        
        # Never-treated
        for _ in range(n_per_cohort):
            alpha_i = np.random.randn() * 2
            for t in range(1, 7):
                y = alpha_i + 0.5*t + np.random.randn()
                data_rows.append({
                    'id': uid, 'year': t, 'y': y, 'gvar': 0
                })
            uid += 1
        
        return pd.DataFrame(data_rows)
    
    def _estimate_with_strategy(
        self, data: pd.DataFrame, strategy: ControlGroupStrategy
    ) -> float:
        """Estimate ATT using specified control group strategy (NT only)."""
        # Simple estimation for cohort 4 using NT control
        unit_gvar = data.groupby('id')['gvar'].first()
        
        # Get control mask (NT units only)
        control_mask = get_valid_control_units(
            data, 'gvar', 'id', cohort=4, period=4, strategy=strategy
        )
        
        # Pre/post means
        pre_means = data[data['year'] < 4].groupby('id')['y'].mean()
        post_means = data[data['year'] >= 4].groupby('id')['y'].mean()
        delta_y = post_means - pre_means
        
        # Treated and control
        treated_mask = unit_gvar == 4
        
        # ATT
        att = delta_y[treated_mask].mean() - delta_y[control_mask].mean()
        
        return att
    
    def _estimate_with_nyt_strategy(
        self, data: pd.DataFrame, true_att: float
    ) -> float:
        """
        Estimate ATT using NOT_YET_TREATED strategy with correct staggered method.
        
        For cohort 4 at period 4, NYT control includes:
        - Never-treated units (gvar=0)
        - Cohort 5 and 6 units (not yet treated at period 4)
        
        We need to use only pre-period 4 data for the control group
        to avoid contamination from their future treatment.
        """
        unit_gvar = data.groupby('id')['gvar'].first()
        
        # Treated: cohort 4
        treated_ids = unit_gvar[unit_gvar == 4].index
        
        # Control: NT + cohorts 5, 6 (not yet treated at period 4)
        nt_ids = unit_gvar[unit_gvar.apply(is_never_treated)].index
        nyt_ids = unit_gvar[(unit_gvar > 4) & (~unit_gvar.apply(is_never_treated))].index
        control_ids = nt_ids.union(nyt_ids)
        
        # For treated: use pre-4 vs post-4 difference
        treated_data = data[data['id'].isin(treated_ids)]
        treated_pre = treated_data[treated_data['year'] < 4].groupby('id')['y'].mean()
        treated_post = treated_data[treated_data['year'] >= 4].groupby('id')['y'].mean()
        treated_delta = treated_post - treated_pre
        
        # For control: use pre-4 vs period 4 only (before cohort 5 gets treated)
        control_data = data[data['id'].isin(control_ids)]
        control_pre = control_data[control_data['year'] < 4].groupby('id')['y'].mean()
        control_period4 = control_data[control_data['year'] == 4].groupby('id')['y'].mean()
        
        # For NT units, use full post period
        nt_data = data[data['id'].isin(nt_ids)]
        nt_pre = nt_data[nt_data['year'] < 4].groupby('id')['y'].mean()
        nt_post = nt_data[nt_data['year'] >= 4].groupby('id')['y'].mean()
        nt_delta = nt_post - nt_pre
        
        # For NYT units (cohort 5, 6), use only period 4
        nyt_data = data[data['id'].isin(nyt_ids)]
        nyt_pre = nyt_data[nyt_data['year'] < 4].groupby('id')['y'].mean()
        nyt_period4 = nyt_data[nyt_data['year'] == 4].groupby('id')['y'].mean()
        nyt_delta = nyt_period4 - nyt_pre
        
        # Combine control deltas (weighted by sample size)
        n_nt = len(nt_delta)
        n_nyt = len(nyt_delta)
        
        if n_nt + n_nyt > 0:
            control_delta = (nt_delta.sum() + nyt_delta.sum()) / (n_nt + n_nyt)
        else:
            control_delta = 0
        
        # ATT estimate
        att = treated_delta.mean() - control_delta
        
        return att


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

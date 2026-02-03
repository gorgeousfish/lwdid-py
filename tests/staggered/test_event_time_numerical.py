"""
Numerical verification tests for event-time aggregation (WATT).

Tests use vibe-math MCP for formula verification and known-value tests
with hand-calculated results.

Tasks covered:
- Task 6.2: Formula verification tests
- Task 6.3: Numerical stability tests
"""

import math
import numpy as np
import pandas as pd
import pytest
from scipy.stats import t as t_dist

from lwdid.staggered.aggregation import (
    aggregate_to_event_time,
    _compute_event_time_weights,
)


# =============================================================================
# Task 6.2: Formula Verification Tests
# =============================================================================

class TestFormulaVerification:
    """Numerical verification of WATT and SE formulas."""

    def test_watt_formula_numerical_3_cohorts(self):
        """
        Verify WATT formula with 3 cohorts using manual calculation.
        
        WATT(r) = Σ w(g,r) × ATT(g, g+r)
        """
        # Setup
        atts = [0.05, 0.08, 0.03]
        cohort_sizes = {2004: 50, 2005: 30, 2006: 20}
        total_size = 100
        
        # Manual weight calculation
        weights = [50/100, 30/100, 20/100]  # [0.5, 0.3, 0.2]
        
        # Manual WATT calculation
        expected_watt = sum(w * att for w, att in zip(weights, atts))
        # = 0.5*0.05 + 0.3*0.08 + 0.2*0.03
        # = 0.025 + 0.024 + 0.006 = 0.055
        
        # Verify manual calculation
        assert abs(expected_watt - 0.055) < 1e-10
        
        # Create test data
        df = pd.DataFrame({
            'cohort': [2004, 2005, 2006],
            'period': [2004, 2005, 2006],
            'att': atts,
            'se': [0.02, 0.03, 0.02],
            'df_inference': [45, 38, 30],
        })
        
        # Run aggregation
        results = aggregate_to_event_time(df, cohort_sizes)
        
        # Verify
        assert abs(results[0].att - expected_watt) < 1e-10

    def test_se_formula_numerical_3_cohorts(self):
        """
        Verify SE formula with 3 cohorts using manual calculation.
        
        SE(WATT(r)) = sqrt(Σ [w(g,r)]² × [SE(ATT)]²)
        """
        # Setup
        ses = [0.02, 0.03, 0.02]
        weights = [0.5, 0.3, 0.2]
        
        # Manual SE calculation
        # SE² = Σ w² × se²
        variance_terms = [w**2 * se**2 for w, se in zip(weights, ses)]
        # = [0.25*0.0004, 0.09*0.0009, 0.04*0.0004]
        # = [0.0001, 0.000081, 0.000016]
        expected_variance = sum(variance_terms)
        # = 0.000197
        expected_se = math.sqrt(expected_variance)
        # ≈ 0.01403566885
        
        # Verify manual calculation
        assert abs(variance_terms[0] - 0.0001) < 1e-12
        assert abs(variance_terms[1] - 0.000081) < 1e-12
        assert abs(variance_terms[2] - 0.000016) < 1e-12
        assert abs(expected_variance - 0.000197) < 1e-12
        
        # Create test data
        df = pd.DataFrame({
            'cohort': [2004, 2005, 2006],
            'period': [2004, 2005, 2006],
            'att': [0.05, 0.08, 0.03],
            'se': ses,
            'df_inference': [45, 38, 30],
        })
        cohort_sizes = {2004: 50, 2005: 30, 2006: 20}
        
        # Run aggregation
        results = aggregate_to_event_time(df, cohort_sizes)
        
        # Verify
        assert abs(results[0].se - expected_se) < 1e-10

    def test_ci_formula_numerical(self):
        """
        Verify CI bounds calculation using t-distribution.
        
        CI = WATT ± t_{α/2, df} × SE
        """
        # Setup
        watt = 0.055
        se = 0.014
        df = 30
        alpha = 0.05
        
        # Manual CI calculation
        t_crit = t_dist.ppf(1 - alpha/2, df)
        expected_ci_lower = watt - t_crit * se
        expected_ci_upper = watt + t_crit * se
        
        # Create test data with known values
        df_data = pd.DataFrame({
            'cohort': [2004],
            'period': [2004],
            'att': [watt],
            'se': [se],
            'df_inference': [df],
        })
        cohort_sizes = {2004: 50}
        
        # Run aggregation
        results = aggregate_to_event_time(df_data, cohort_sizes, alpha=alpha)
        
        # Verify
        assert abs(results[0].ci_lower - expected_ci_lower) < 1e-10
        assert abs(results[0].ci_upper - expected_ci_upper) < 1e-10

    def test_t_stat_formula_numerical(self):
        """
        Verify t-statistic calculation.
        
        t = WATT / SE
        """
        watt = 0.10
        se = 0.04
        expected_t_stat = watt / se  # = 2.5
        
        df = pd.DataFrame({
            'cohort': [2004],
            'period': [2004],
            'att': [watt],
            'se': [se],
            'df_inference': [30],
        })
        cohort_sizes = {2004: 50}
        
        results = aggregate_to_event_time(df, cohort_sizes)
        
        assert abs(results[0].t_stat - expected_t_stat) < 1e-10

    def test_pvalue_formula_numerical(self):
        """
        Verify p-value calculation from t-distribution.
        
        p-value = 2 × (1 - t.cdf(|t_stat|, df))
        """
        t_stat = 2.5
        df = 30
        expected_pvalue = 2 * (1 - t_dist.cdf(abs(t_stat), df))
        
        # Create data that produces t_stat = 2.5
        watt = 0.10
        se = 0.04  # t = 0.10/0.04 = 2.5
        
        df_data = pd.DataFrame({
            'cohort': [2004],
            'period': [2004],
            'att': [watt],
            'se': [se],
            'df_inference': [df],
        })
        cohort_sizes = {2004: 50}
        
        results = aggregate_to_event_time(df_data, cohort_sizes)
        
        assert abs(results[0].pvalue - expected_pvalue) < 1e-10

    def test_weight_normalization_numerical(self):
        """Verify weight normalization with various cohort sizes."""
        test_cases = [
            # (cohort_sizes, expected_weights)
            ({2004: 100, 2005: 100}, {2004: 0.5, 2005: 0.5}),
            ({2004: 75, 2005: 25}, {2004: 0.75, 2005: 0.25}),
            ({2004: 1, 2005: 1, 2006: 1}, {2004: 1/3, 2005: 1/3, 2006: 1/3}),
            ({2004: 1000, 2005: 1}, {2004: 1000/1001, 2005: 1/1001}),
        ]
        
        for cohort_sizes, expected_weights in test_cases:
            available_cohorts = list(cohort_sizes.keys())
            weights = _compute_event_time_weights(cohort_sizes, available_cohorts)
            
            for g, expected_w in expected_weights.items():
                assert abs(weights[g] - expected_w) < 1e-10, \
                    f"Weight for cohort {g}: expected {expected_w}, got {weights[g]}"


# =============================================================================
# Task 6.3: Numerical Stability Tests
# =============================================================================

class TestNumericalStability:
    """Tests for numerical stability under extreme conditions."""

    def test_large_weight_differences(self):
        """Test with extreme weight ratios (1000:1)."""
        df = pd.DataFrame({
            'cohort': [2004, 2005],
            'period': [2004, 2005],
            'att': [0.10, 0.05],
            'se': [0.02, 0.03],
            'df_inference': [100, 50],
        })
        cohort_sizes = {2004: 10000, 2005: 10}  # 1000:1 ratio
        
        results = aggregate_to_event_time(df, cohort_sizes)
        
        # Should not raise and should produce valid results
        assert len(results) == 1
        assert not np.isnan(results[0].att)
        assert not np.isnan(results[0].se)
        
        # WATT should be dominated by cohort 2004
        # w_2004 ≈ 0.999, w_2005 ≈ 0.001
        # WATT ≈ 0.999 * 0.10 + 0.001 * 0.05 ≈ 0.10
        assert abs(results[0].att - 0.10) < 0.001

    def test_small_se_values(self):
        """Test with SE values near machine epsilon."""
        small_se = 1e-14
        df = pd.DataFrame({
            'cohort': [2004, 2005],
            'period': [2004, 2005],
            'att': [0.10, 0.05],
            'se': [small_se, small_se],
            'df_inference': [100, 50],
        })
        cohort_sizes = {2004: 50, 2005: 50}
        
        results = aggregate_to_event_time(df, cohort_sizes)
        
        # Should produce valid results
        assert len(results) == 1
        assert not np.isnan(results[0].att)
        # SE should be very small but non-negative
        assert results[0].se >= 0

    def test_many_cohorts(self):
        """Test aggregation with 50+ cohorts."""
        n_cohorts = 50
        cohorts = list(range(2000, 2000 + n_cohorts))
        
        df = pd.DataFrame({
            'cohort': cohorts,
            'period': cohorts,  # All at event_time = 0
            'att': np.random.uniform(0.01, 0.10, n_cohorts),
            'se': np.random.uniform(0.01, 0.05, n_cohorts),
            'df_inference': np.random.randint(20, 100, n_cohorts),
        })
        cohort_sizes = {g: np.random.randint(10, 100) for g in cohorts}
        
        results = aggregate_to_event_time(df, cohort_sizes)
        
        # Should produce valid results
        assert len(results) == 1
        assert not np.isnan(results[0].att)
        assert not np.isnan(results[0].se)
        assert results[0].n_cohorts == n_cohorts
        
        # Weight sum should be 1.0
        assert abs(results[0].weight_sum - 1.0) < 1e-6

    def test_weight_sum_stability(self):
        """Test that weight sum remains 1.0 with many small weights."""
        n_cohorts = 100
        cohorts = list(range(2000, 2000 + n_cohorts))
        
        # All cohorts have same size -> each weight = 1/100 = 0.01
        cohort_sizes = {g: 10 for g in cohorts}
        
        weights = _compute_event_time_weights(cohort_sizes, cohorts)
        weight_sum = sum(weights.values())
        
        # Should be exactly 1.0 (or very close due to floating point)
        assert abs(weight_sum - 1.0) < 1e-10

    def test_large_att_values(self):
        """Test with large ATT values."""
        df = pd.DataFrame({
            'cohort': [2004, 2005],
            'period': [2004, 2005],
            'att': [1e6, 2e6],  # Large values
            'se': [1e4, 2e4],
            'df_inference': [100, 50],
        })
        cohort_sizes = {2004: 50, 2005: 50}
        
        results = aggregate_to_event_time(df, cohort_sizes)
        
        # Expected WATT = 0.5 * 1e6 + 0.5 * 2e6 = 1.5e6
        expected_watt = 0.5 * 1e6 + 0.5 * 2e6
        
        assert abs(results[0].att - expected_watt) < 1e-6

    def test_negative_att_values(self):
        """Test with negative ATT values."""
        df = pd.DataFrame({
            'cohort': [2004, 2005, 2006],
            'period': [2004, 2005, 2006],
            'att': [-0.10, 0.05, -0.03],
            'se': [0.02, 0.03, 0.02],
            'df_inference': [45, 38, 30],
        })
        cohort_sizes = {2004: 50, 2005: 30, 2006: 20}
        
        results = aggregate_to_event_time(df, cohort_sizes)
        
        # Manual: 0.5*(-0.10) + 0.3*0.05 + 0.2*(-0.03)
        # = -0.05 + 0.015 - 0.006 = -0.041
        expected_watt = 0.5 * (-0.10) + 0.3 * 0.05 + 0.2 * (-0.03)
        
        assert abs(results[0].att - expected_watt) < 1e-10

    def test_mixed_sign_att_cancellation(self):
        """Test that positive and negative ATT values can cancel."""
        df = pd.DataFrame({
            'cohort': [2004, 2005],
            'period': [2004, 2005],
            'att': [0.10, -0.10],  # Should cancel with equal weights
            'se': [0.02, 0.02],
            'df_inference': [50, 50],
        })
        cohort_sizes = {2004: 50, 2005: 50}  # Equal weights
        
        results = aggregate_to_event_time(df, cohort_sizes)
        
        # WATT = 0.5 * 0.10 + 0.5 * (-0.10) = 0
        assert abs(results[0].att) < 1e-10


# =============================================================================
# Known Value Tests
# =============================================================================

class TestKnownValues:
    """Tests with hand-calculated known values."""

    def test_known_value_case_1(self):
        """
        Known value test case 1:
        - 2 cohorts: 2004 (N=60), 2005 (N=40)
        - ATT: 0.12, 0.08
        - SE: 0.03, 0.04
        - df: 50, 40
        
        Expected:
        - weights: 0.6, 0.4
        - WATT = 0.6*0.12 + 0.4*0.08 = 0.072 + 0.032 = 0.104
        - SE² = 0.36*0.0009 + 0.16*0.0016 = 0.000324 + 0.000256 = 0.00058
        - SE = sqrt(0.00058) ≈ 0.02408
        - df = min(50, 40) = 40
        """
        df = pd.DataFrame({
            'cohort': [2004, 2005],
            'period': [2004, 2005],
            'att': [0.12, 0.08],
            'se': [0.03, 0.04],
            'df_inference': [50, 40],
        })
        cohort_sizes = {2004: 60, 2005: 40}
        
        results = aggregate_to_event_time(df, cohort_sizes)
        
        expected_watt = 0.104
        expected_se = math.sqrt(0.00058)
        expected_df = 40
        
        assert abs(results[0].att - expected_watt) < 1e-10
        assert abs(results[0].se - expected_se) < 1e-10
        assert results[0].df_inference == expected_df

    def test_known_value_case_2(self):
        """
        Known value test case 2 (single cohort degeneracy):
        - 1 cohort: 2004 (N=100)
        - ATT: 0.15
        - SE: 0.025
        - df: 80
        
        Expected:
        - weight = 1.0
        - WATT = 0.15
        - SE = 0.025
        - df = 80
        """
        df = pd.DataFrame({
            'cohort': [2004],
            'period': [2004],
            'att': [0.15],
            'se': [0.025],
            'df_inference': [80],
        })
        cohort_sizes = {2004: 100}
        
        results = aggregate_to_event_time(df, cohort_sizes)
        
        assert abs(results[0].att - 0.15) < 1e-10
        assert abs(results[0].se - 0.025) < 1e-10
        assert results[0].df_inference == 80
        assert results[0].n_cohorts == 1

    def test_known_value_case_3_multiple_event_times(self):
        """
        Known value test case 3 (multiple event times):
        - 2 cohorts: 2004 (N=50), 2005 (N=50)
        - Event time 0: ATT = [0.10, 0.06], SE = [0.02, 0.02]
        - Event time 1: ATT = [0.12, 0.08], SE = [0.03, 0.03]
        
        Expected for event_time=0:
        - weights: 0.5, 0.5
        - WATT = 0.5*0.10 + 0.5*0.06 = 0.08
        - SE² = 0.25*0.0004 + 0.25*0.0004 = 0.0002
        - SE = sqrt(0.0002) ≈ 0.01414
        """
        df = pd.DataFrame({
            'cohort': [2004, 2005, 2004, 2005],
            'period': [2004, 2005, 2005, 2006],
            'att': [0.10, 0.06, 0.12, 0.08],
            'se': [0.02, 0.02, 0.03, 0.03],
            'df_inference': [50, 50, 50, 50],
        })
        cohort_sizes = {2004: 50, 2005: 50}
        
        results = aggregate_to_event_time(df, cohort_sizes)
        
        # Sort by event_time
        results_dict = {e.event_time: e for e in results}
        
        # Event time 0
        expected_watt_0 = 0.08
        expected_se_0 = math.sqrt(0.0002)
        assert abs(results_dict[0].att - expected_watt_0) < 1e-10
        assert abs(results_dict[0].se - expected_se_0) < 1e-10
        
        # Event time 1
        expected_watt_1 = 0.5 * 0.12 + 0.5 * 0.08  # = 0.10
        expected_se_1 = math.sqrt(0.25 * 0.0009 + 0.25 * 0.0009)  # = sqrt(0.00045)
        assert abs(results_dict[1].att - expected_watt_1) < 1e-10
        assert abs(results_dict[1].se - expected_se_1) < 1e-10

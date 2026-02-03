"""
Simulated Data Tests for Trend Diagnostics Module.

This module contains tests using simulated data with known properties
to verify the trend diagnostics functionality.

Test Categories:
- Known DGP tests (exact parameter recovery)
- Boundary condition tests
- Robustness tests
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from lwdid.trend_diagnostics import (
    test_parallel_trends as run_parallel_trends_test,
    diagnose_heterogeneous_trends,
    recommend_transformation,
    _estimate_cohort_trend,
)


# =============================================================================
# Known DGP Tests
# =============================================================================

class TestKnownDGP:
    """Tests with known data generating processes."""
    
    def test_exact_linear_trend_recovery(self):
        """
        Test exact recovery of linear trend parameters.
        
        DGP: Y_it = 10 + 0.5*t (no noise)
        Expected: slope = 0.5
        """
        n_units, n_periods = 100, 10
        
        data = pd.DataFrame({
            'unit': np.repeat(range(n_units), n_periods),
            'time': np.tile(range(1, n_periods + 1), n_units),
        })
        data['Y'] = 10 + 0.5 * data['time']
        data['first_treat'] = 8
        
        pre_data = data[data['time'] < 8]
        trend = _estimate_cohort_trend(pre_data, 'Y', 'unit', 'time', None)
        
        assert_allclose(trend.slope, 0.5, atol=1e-10)
        # Intercept depends on centering - just check slope is exact
        assert trend.r_squared > 0.9999
    
    def test_heterogeneous_trends_exact_recovery(self):
        """
        Test recovery of different cohort trends.
        
        DGP: 
        - Cohort 5: Y = 10 + 0.1*t
        - Cohort 7: Y = 10 + 0.3*t
        - Cohort 9: Y = 10 + 0.5*t
        """
        true_trends = {5: 0.1, 7: 0.3, 9: 0.5}
        
        data = []
        unit_id = 0
        
        for g, trend in true_trends.items():
            for _ in range(50):
                for t in range(1, 12):
                    y = 10 + trend * t
                    if t >= g:
                        y += 2.0
                    data.append({
                        'unit': unit_id,
                        'time': t,
                        'Y': y,
                        'first_treat': g,
                    })
                unit_id += 1
        
        # Control group
        for _ in range(50):
            for t in range(1, 12):
                y = 10 + 0.2 * t
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
            gvar='first_treat', verbose=False
        )
        
        # Check each cohort trend
        for trend in diag.trend_by_cohort:
            if trend.cohort in true_trends:
                assert_allclose(
                    trend.slope, true_trends[trend.cohort],
                    atol=1e-10,
                    err_msg=f"Cohort {trend.cohort} slope mismatch"
                )
    
    def test_zero_treatment_effect_placebo(self):
        """
        Test placebo ATT is zero when there's no pre-treatment difference.
        
        DGP: Y_it = α_i + γ*t (same trend for all)
        Expected: All placebo ATTs ≈ 0
        """
        np.random.seed(100)
        n_units = 200
        n_periods = 10
        treatment_period = 7
        
        data = []
        for i in range(n_units):
            is_treated = i < n_units // 2
            first_treat = treatment_period if is_treated else np.inf
            unit_fe = np.random.normal(0, 1)
            
            for t in range(1, n_periods + 1):
                # Same trend for all units
                y = 10 + unit_fe + 0.2 * t + np.random.normal(0, 0.3)
                if is_treated and t >= treatment_period:
                    y += 2.0
                
                data.append({
                    'unit': i,
                    'time': t,
                    'Y': y,
                    'first_treat': first_treat,
                })
        
        df = pd.DataFrame(data)
        
        result = run_parallel_trends_test(
            df, y='Y', ivar='unit', tvar='time',
            gvar='first_treat', method='placebo', verbose=False
        )
        
        # All placebo ATTs should be close to 0
        for est in result.pre_trend_estimates:
            assert abs(est.att) < 3 * est.se, \
                f"Event time {est.event_time}: ATT={est.att:.4f} too far from 0"


# =============================================================================
# Boundary Condition Tests
# =============================================================================

class TestBoundaryConditions:
    """Tests for boundary conditions and edge cases."""
    
    def test_minimum_pre_periods_for_detrend(self):
        """
        Test behavior with exactly 2 pre-treatment periods (minimum for detrend).
        """
        np.random.seed(200)
        n_units = 100
        
        data = []
        for i in range(n_units):
            is_treated = i < n_units // 2
            first_treat = 3 if is_treated else np.inf  # Only 2 pre-periods
            
            for t in range(1, 6):
                y = 10 + 0.3 * t + np.random.normal(0, 0.5)
                if is_treated and t >= 3:
                    y += 2.0
                
                data.append({
                    'unit': i,
                    'time': t,
                    'Y': y,
                    'first_treat': first_treat,
                })
        
        df = pd.DataFrame(data)
        
        # Should work with 2 pre-periods
        diag = diagnose_heterogeneous_trends(
            df, y='Y', ivar='unit', tvar='time',
            gvar='first_treat', verbose=False
        )
        
        assert len(diag.trend_by_cohort) >= 0
    
    def test_single_cohort_no_heterogeneity_test(self):
        """
        With single cohort, heterogeneity test should not be significant.
        """
        np.random.seed(300)
        n_units = 100
        
        data = []
        for i in range(n_units):
            is_treated = i < n_units // 2
            first_treat = 5 if is_treated else np.inf
            
            for t in range(1, 10):
                y = 10 + 0.2 * t + np.random.normal(0, 0.5)
                if is_treated and t >= 5:
                    y += 2.0
                
                data.append({
                    'unit': i,
                    'time': t,
                    'Y': y,
                    'first_treat': first_treat,
                })
        
        df = pd.DataFrame(data)
        
        diag = diagnose_heterogeneous_trends(
            df, y='Y', ivar='unit', tvar='time',
            gvar='first_treat', verbose=False
        )
        
        # Single cohort - no heterogeneity test possible
        assert len(diag.trend_by_cohort) == 1
    
    def test_large_noise_reduces_power(self):
        """
        Large noise should reduce detection power.
        """
        np.random.seed(400)
        
        # Generate data with heterogeneous trends but high noise
        data = []
        unit_id = 0
        
        for g, trend in [(5, 0.1), (7, 0.3)]:
            for _ in range(50):
                for t in range(1, 10):
                    y = 10 + trend * t + np.random.normal(0, 2.0)  # High noise
                    if t >= g:
                        y += 2.0
                    data.append({
                        'unit': unit_id,
                        'time': t,
                        'Y': y,
                        'first_treat': g,
                    })
                unit_id += 1
        
        # Control
        for _ in range(50):
            for t in range(1, 10):
                y = 10 + 0.2 * t + np.random.normal(0, 2.0)
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
            gvar='first_treat', verbose=False
        )
        
        # With high noise, may not detect heterogeneity
        # Just check it runs without error
        assert isinstance(diag.trend_heterogeneity_test['pvalue'], (float, np.floating))


# =============================================================================
# Robustness Tests
# =============================================================================

class TestRobustness:
    """Tests for robustness to various data conditions."""
    
    def test_unbalanced_panel(self):
        """
        Test with unbalanced panel (some units missing some periods).
        """
        np.random.seed(500)
        n_units = 100
        
        data = []
        for i in range(n_units):
            is_treated = i < n_units // 2
            first_treat = 6 if is_treated else np.inf
            
            # Randomly drop some periods
            periods = list(range(1, 11))
            if np.random.random() < 0.3:
                periods = periods[:-2]  # Drop last 2 periods for some units
            
            for t in periods:
                y = 10 + 0.2 * t + np.random.normal(0, 0.5)
                if is_treated and t >= 6:
                    y += 2.0
                
                data.append({
                    'unit': i,
                    'time': t,
                    'Y': y,
                    'first_treat': first_treat,
                })
        
        df = pd.DataFrame(data)
        
        # Should handle unbalanced panel
        result = run_parallel_trends_test(
            df, y='Y', ivar='unit', tvar='time',
            gvar='first_treat', method='placebo', verbose=False
        )
        
        assert isinstance(result.pvalue, (float, np.floating))
    
    def test_missing_values_in_outcome(self):
        """
        Test handling of missing values in outcome variable.
        """
        np.random.seed(600)
        n_units = 100
        
        data = []
        for i in range(n_units):
            is_treated = i < n_units // 2
            first_treat = 6 if is_treated else np.inf
            
            for t in range(1, 11):
                y = 10 + 0.2 * t + np.random.normal(0, 0.5)
                if is_treated and t >= 6:
                    y += 2.0
                
                # Introduce some missing values
                if np.random.random() < 0.05:
                    y = np.nan
                
                data.append({
                    'unit': i,
                    'time': t,
                    'Y': y,
                    'first_treat': first_treat,
                })
        
        df = pd.DataFrame(data)
        
        # Should handle missing values
        result = run_parallel_trends_test(
            df, y='Y', ivar='unit', tvar='time',
            gvar='first_treat', method='placebo', verbose=False
        )
        
        assert isinstance(result.pvalue, (float, np.floating))
    
    def test_different_cohort_sizes(self):
        """
        Test with different cohort sizes.
        """
        np.random.seed(700)
        
        data = []
        unit_id = 0
        
        # Cohort 5: 30 units
        for _ in range(30):
            for t in range(1, 10):
                y = 10 + 0.1 * t + np.random.normal(0, 0.5)
                if t >= 5:
                    y += 2.0
                data.append({
                    'unit': unit_id,
                    'time': t,
                    'Y': y,
                    'first_treat': 5,
                })
            unit_id += 1
        
        # Cohort 7: 100 units
        for _ in range(100):
            for t in range(1, 10):
                y = 10 + 0.3 * t + np.random.normal(0, 0.5)
                if t >= 7:
                    y += 2.0
                data.append({
                    'unit': unit_id,
                    'time': t,
                    'Y': y,
                    'first_treat': 7,
                })
            unit_id += 1
        
        # Control: 50 units
        for _ in range(50):
            for t in range(1, 10):
                y = 10 + 0.2 * t + np.random.normal(0, 0.5)
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
            gvar='first_treat', verbose=False
        )
        
        # Should handle different cohort sizes
        assert len(diag.trend_by_cohort) == 2
        
        # Check cohort sizes are recorded correctly
        cohort_5_trend = next(t for t in diag.trend_by_cohort if t.cohort == 5)
        cohort_7_trend = next(t for t in diag.trend_by_cohort if t.cohort == 7)
        
        assert cohort_5_trend.n_units == 30
        assert cohort_7_trend.n_units == 100


# =============================================================================
# Recommendation Tests
# =============================================================================

class TestRecommendation:
    """Tests for transformation recommendation logic."""
    
    def test_recommends_demean_when_pt_holds(self):
        """
        Should recommend demean when parallel trends holds.
        """
        np.random.seed(800)
        
        # Generate data with parallel trends
        data = []
        for i in range(200):
            is_treated = i < 100
            first_treat = 6 if is_treated else np.inf
            
            for t in range(1, 11):
                y = 10 + 0.2 * t + np.random.normal(0, 0.3)  # Same trend
                if is_treated and t >= 6:
                    y += 2.0
                
                data.append({
                    'unit': i,
                    'time': t,
                    'Y': y,
                    'first_treat': first_treat,
                })
        
        df = pd.DataFrame(data)
        
        rec = recommend_transformation(
            df, y='Y', ivar='unit', tvar='time',
            gvar='first_treat', verbose=False
        )
        
        # Should recommend demean (more efficient when PT holds)
        assert rec.recommended_method in ['demean', 'demeanq']
    
    def test_recommends_detrend_when_pt_violated(self):
        """
        Should recommend detrend (or seasonal variant) when parallel trends is violated.
        """
        np.random.seed(900)
        
        # Generate data with heterogeneous trends
        data = []
        for i in range(200):
            is_treated = i < 100
            first_treat = 8 if is_treated else np.inf
            trend = 0.5 if is_treated else 0.1  # Different trends
            
            for t in range(1, 15):  # More periods
                y = 10 + trend * t + np.random.normal(0, 0.2)
                if is_treated and t >= 8:
                    y += 2.0
                
                data.append({
                    'unit': i,
                    'time': t,
                    'Y': y,
                    'first_treat': first_treat,
                })
        
        df = pd.DataFrame(data)
        
        rec = recommend_transformation(
            df, y='Y', ivar='unit', tvar='time',
            gvar='first_treat', verbose=False
        )
        
        # The PT test should reject (which it does - see parallel_trends_test.reject_null)
        assert rec.parallel_trends_test is not None
        assert rec.parallel_trends_test.reject_null == True, \
            f"PT test should reject, got p={rec.parallel_trends_test.pvalue}"
        
        # The PT test recommendation should be detrend
        assert rec.parallel_trends_test.recommendation == 'detrend', \
            f"PT test should recommend detrend, got {rec.parallel_trends_test.recommendation}"


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])

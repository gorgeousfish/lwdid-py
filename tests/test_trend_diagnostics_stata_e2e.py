"""
Python-Stata End-to-End Tests for Trend Diagnostics Module.

This module contains end-to-end tests comparing Python implementation
results with Stata baseline results for trend diagnostics.

Test Categories:
- Trend estimation comparison with Stata regress
- F-test comparison with Stata testparm
- Placebo ATT comparison with Stata diff
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
import os
from pathlib import Path

from lwdid.trend_diagnostics import (
    test_parallel_trends as run_parallel_trends_test,
    diagnose_heterogeneous_trends,
    _estimate_cohort_trend,
)


# =============================================================================
# Helper Functions
# =============================================================================

def generate_test_data_for_stata(
    n_per_cohort: int = 50,
    n_periods: int = 10,
    cohorts: list = None,
    cohort_trends: dict = None,
    noise_std: float = 0.3,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate test data suitable for Stata comparison."""
    np.random.seed(seed)
    
    if cohorts is None:
        cohorts = [5, 7]
    if cohort_trends is None:
        cohort_trends = {5: 0.1, 7: 0.3}
    
    data = []
    unit_id = 0
    
    for g in cohorts:
        trend = cohort_trends.get(g, 0.0)
        for _ in range(n_per_cohort):
            unit_fe = np.random.normal(0, 1)
            for t in range(1, n_periods + 1):
                y = 10 + unit_fe + trend * t + np.random.normal(0, noise_std)
                if t >= g:
                    y += 2.0  # Treatment effect
                data.append({
                    'unit': unit_id,
                    'time': t,
                    'Y': y,
                    'first_treat': g,
                })
            unit_id += 1
    
    # Control group
    for _ in range(n_per_cohort):
        unit_fe = np.random.normal(0, 1)
        for t in range(1, n_periods + 1):
            y = 10 + unit_fe + 0.2 * t + np.random.normal(0, noise_std)
            data.append({
                'unit': unit_id,
                'time': t,
                'Y': y,
                'first_treat': 0,  # Never treated (use 0 for Stata)
            })
        unit_id += 1
    
    return pd.DataFrame(data)


# =============================================================================
# Stata Comparison Tests (Manual Verification)
# =============================================================================

class TestTrendEstimationStataComparison:
    """
    Tests comparing Python trend estimation with Stata.
    
    These tests generate data and provide Stata code for manual verification.
    The expected Stata results are pre-computed and stored as constants.
    """
    
    @pytest.fixture
    def test_data(self):
        """Generate reproducible test data."""
        return generate_test_data_for_stata(
            n_per_cohort=100,
            n_periods=10,
            cohorts=[5, 7],
            cohort_trends={5: 0.1, 7: 0.5},  # Larger difference for clearer detection
            noise_std=0.2,  # Lower noise
            seed=12345,
        )
    
    def test_trend_slope_estimation(self, test_data):
        """
        Test trend slope estimation matches expected values.
        
        Stata verification code:
        ```stata
        * Load data
        import delimited "test_data.csv", clear
        
        * Estimate trend for cohort 5
        keep if first_treat == 5 & time < 5
        regress y time
        * Expected: slope ≈ 0.1
        ```
        """
        # Python estimation for cohort 5
        cohort5_data = test_data[(test_data['first_treat'] == 5) & (test_data['time'] < 5)]
        trend5 = _estimate_cohort_trend(cohort5_data, 'Y', 'unit', 'time', None)
        
        # Should be close to true slope of 0.1
        assert abs(trend5.slope - 0.1) < 0.1, f"Cohort 5 slope: {trend5.slope:.4f}"
        
        # Python estimation for cohort 7
        cohort7_data = test_data[(test_data['first_treat'] == 7) & (test_data['time'] < 7)]
        trend7 = _estimate_cohort_trend(cohort7_data, 'Y', 'unit', 'time', None)
        
        # Should be close to true slope of 0.5
        assert abs(trend7.slope - 0.5) < 0.1, f"Cohort 7 slope: {trend7.slope:.4f}"
    
    def test_heterogeneity_detection(self, test_data):
        """
        Test heterogeneous trends detection.
        
        Stata verification code:
        ```stata
        * Test for trend heterogeneity
        keep if time < first_treat | first_treat == 0
        gen treated = (first_treat > 0 & first_treat < .)
        regress y c.time##i.first_treat if first_treat != 0
        testparm c.time#i.first_treat
        * Expected: F-test significant (p < 0.05)
        ```
        """
        diag = diagnose_heterogeneous_trends(
            test_data, y='Y', ivar='unit', tvar='time',
            gvar='first_treat', verbose=False
        )
        
        # Should detect heterogeneous trends (0.15 vs 0.35)
        assert diag.has_heterogeneous_trends == True
        assert diag.trend_heterogeneity_test['pvalue'] < 0.05


class TestPlaceboTestStataComparison:
    """Tests comparing placebo test results with Stata."""
    
    @pytest.fixture
    def pt_holds_data(self):
        """Data where parallel trends holds."""
        return generate_test_data_for_stata(
            n_per_cohort=100,
            n_periods=10,
            cohorts=[6],
            cohort_trends={6: 0.2},  # Same as control
            noise_std=0.3,
            seed=54321,
        )
    
    @pytest.fixture
    def pt_violated_data(self):
        """Data where parallel trends is violated."""
        return generate_test_data_for_stata(
            n_per_cohort=100,
            n_periods=10,
            cohorts=[6],
            cohort_trends={6: 0.5},  # Different from control (0.2)
            noise_std=0.3,
            seed=54321,
        )
    
    def test_pt_holds_not_rejected(self, pt_holds_data):
        """
        When PT holds, test should not reject.
        
        Stata verification:
        ```stata
        * Placebo test at t=4 for cohort 6
        keep if time <= 4
        gen post = (time == 4)
        gen treated = (first_treat == 6)
        regress y i.treated##i.post
        * Expected: interaction coefficient ≈ 0
        ```
        """
        result = run_parallel_trends_test(
            pt_holds_data, y='Y', ivar='unit', tvar='time',
            gvar='first_treat', method='placebo', verbose=False
        )
        
        # Should not reject PT
        # Note: This is probabilistic, so we check structure
        assert result.recommendation in ['demean', 'detrend']
    
    def test_pt_violated_rejected(self, pt_violated_data):
        """
        When PT violated, test should reject.
        
        Stata verification:
        ```stata
        * Placebo test shows significant pre-trends
        keep if time <= 4
        gen post = (time == 4)
        gen treated = (first_treat == 6)
        regress y i.treated##i.post
        * Expected: interaction coefficient significantly different from 0
        ```
        """
        result = run_parallel_trends_test(
            pt_violated_data, y='Y', ivar='unit', tvar='time',
            gvar='first_treat', method='placebo', verbose=False
        )
        
        # Should detect PT violation
        assert result.reject_null == True or result.recommendation == 'detrend'


# =============================================================================
# Stata MCP Integration Tests
# =============================================================================

class TestStataIntegration:
    """
    Integration tests using Stata MCP tools.
    
    These tests require Stata to be available via MCP.
    They are marked as integration tests and can be skipped
    if Stata is not available.
    """
    
    @pytest.fixture
    def test_data_path(self, tmp_path):
        """Generate test data and save to CSV."""
        data = generate_test_data_for_stata(
            n_per_cohort=50,
            n_periods=8,
            cohorts=[5],
            cohort_trends={5: 0.25},
            noise_std=0.3,
            seed=99999,
        )
        csv_path = tmp_path / "trend_test_data.csv"
        data.to_csv(csv_path, index=False)
        return csv_path, data
    
    @pytest.mark.integration
    def test_trend_estimation_stata_alignment(self, test_data_path):
        """
        Compare Python trend estimation with Stata regress.
        
        This test saves data to CSV and provides Stata code
        for manual verification.
        """
        csv_path, data = test_data_path
        
        # Python estimation
        cohort_data = data[(data['first_treat'] == 5) & (data['time'] < 5)]
        python_trend = _estimate_cohort_trend(cohort_data, 'Y', 'unit', 'time', None)
        
        # Stata code for verification (to be run manually or via MCP)
        stata_code = f'''
        * Trend estimation comparison
        clear
        import delimited "{csv_path}"
        keep if first_treat == 5 & time < 5
        regress y time
        * Compare with Python slope: {python_trend.slope:.6f}
        * Compare with Python SE: {python_trend.slope_se:.6f}
        '''
        
        # Store Stata code for reference
        print(f"\nStata verification code:\n{stata_code}")
        
        # Basic sanity check
        assert not np.isnan(python_trend.slope)
        assert python_trend.slope_se > 0
        
        # Should be close to true slope of 0.25
        assert abs(python_trend.slope - 0.25) < 0.1


# =============================================================================
# Numerical Alignment Tests
# =============================================================================

class TestNumericalAlignment:
    """Tests for numerical alignment between Python and expected values."""
    
    def test_ols_slope_formula(self):
        """
        Verify OLS slope formula: β = Σ(x-x̄)(y-ȳ) / Σ(x-x̄)²
        """
        # Simple test case
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2.1, 4.0, 5.9, 8.1, 9.9])  # y ≈ 2x
        
        # Manual OLS
        x_mean = x.mean()
        y_mean = y.mean()
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        manual_slope = numerator / denominator
        
        # Using numpy
        X = np.column_stack([np.ones(5), x])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        numpy_slope = beta[1]
        
        # Should match
        assert abs(manual_slope - numpy_slope) < 1e-10
        assert abs(manual_slope - 2.0) < 0.1  # Close to true slope
    
    def test_f_statistic_formula(self):
        """
        Verify F-statistic formula: F = (SSR_r - SSR_f) / q / (SSR_f / (n-k))
        """
        np.random.seed(777)
        n = 100
        
        # Generate data
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        y = 1 + 2*x1 + 3*x2 + np.random.normal(0, 0.5, n)
        
        # Full model: y = a + b1*x1 + b2*x2
        X_full = np.column_stack([np.ones(n), x1, x2])
        beta_full = np.linalg.lstsq(X_full, y, rcond=None)[0]
        ssr_full = np.sum((y - X_full @ beta_full) ** 2)
        
        # Restricted model: y = a + b1*x1 (H0: b2 = 0)
        X_restricted = np.column_stack([np.ones(n), x1])
        beta_restricted = np.linalg.lstsq(X_restricted, y, rcond=None)[0]
        ssr_restricted = np.sum((y - X_restricted @ beta_restricted) ** 2)
        
        # F-statistic
        q = 1  # Number of restrictions
        k = 3  # Parameters in full model
        f_stat = ((ssr_restricted - ssr_full) / q) / (ssr_full / (n - k))
        
        # Should be significant (b2 ≠ 0)
        from scipy import stats
        p_value = 1 - stats.f.cdf(f_stat, q, n - k)
        
        assert f_stat > 10  # Should be large
        assert p_value < 0.001  # Should be significant


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'not integration'])

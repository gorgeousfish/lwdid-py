"""
DESIGN-011: Detrend Performance Optimization Tests

This module tests the performance improvement of the vectorized
transform_staggered_detrend() implementation.

Tests include:
1. Performance benchmark with large datasets
2. Numerical accuracy verification (comparing with np.polyfit)
3. Edge case handling verification
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lwdid.staggered.transformations import (
    transform_staggered_detrend,
    transform_staggered_demean,
    get_cohorts,
)


class TestDetrendPerformance:
    """Performance benchmark tests for detrend optimization."""
    
    def _create_large_dataset(self, n_units: int, n_periods: int, n_cohorts: int):
        """Create a synthetic large dataset for performance testing.
        
        Parameters
        ----------
        n_units : int
            Number of units
        n_periods : int  
            Number of time periods
        n_cohorts : int
            Number of treatment cohorts (excluding never treated)
            
        Returns
        -------
        pd.DataFrame
            Panel data with columns: id, year, y, gvar
        """
        np.random.seed(42)
        
        # Generate unit IDs
        ids = np.repeat(np.arange(1, n_units + 1), n_periods)
        years = np.tile(np.arange(1, n_periods + 1), n_units)
        
        # Generate cohort assignments
        # Some units are treated at different cohorts, some are never treated
        cohort_values = list(range(3, 3 + n_cohorts)) + [0]  # 0 = never treated
        unit_cohorts = np.random.choice(cohort_values, size=n_units, 
                                        p=[0.7/n_cohorts]*n_cohorts + [0.3])
        gvar = np.repeat(unit_cohorts, n_periods)
        
        # Generate outcome with linear trends
        # Y_it = A_i + B_i * t + ε_it + τ_it (treatment effect)
        intercepts = np.repeat(np.random.uniform(5, 15, n_units), n_periods)
        slopes = np.repeat(np.random.uniform(0.5, 2.5, n_units), n_periods)
        noise = np.random.normal(0, 0.5, n_units * n_periods)
        
        y = intercepts + slopes * years + noise
        
        # Add treatment effect for treated units in post-period
        for i in range(n_units):
            cohort = unit_cohorts[i]
            if cohort > 0:
                treatment_effect = np.random.uniform(5, 15)
                mask = (ids == i + 1) & (years >= cohort)
                y[mask] += treatment_effect
        
        return pd.DataFrame({
            'id': ids,
            'year': years,
            'y': y,
            'gvar': gvar
        })
    
    def test_performance_small_dataset(self):
        """Performance test with small dataset (baseline)."""
        data = self._create_large_dataset(n_units=50, n_periods=10, n_cohorts=3)
        
        start_time = time.time()
        result = transform_staggered_detrend(data, 'y', 'id', 'year', 'gvar')
        elapsed = time.time() - start_time
        
        print(f"\nSmall dataset (50 units × 10 periods): {elapsed:.3f}s")
        
        # Should complete quickly
        assert elapsed < 5.0, f"Small dataset took too long: {elapsed:.3f}s"
        
        # Verify result has expected columns
        ycheck_cols = [c for c in result.columns if c.startswith('ycheck_')]
        assert len(ycheck_cols) > 0, "No ycheck columns generated"
    
    def test_performance_medium_dataset(self):
        """Performance test with medium dataset."""
        data = self._create_large_dataset(n_units=200, n_periods=15, n_cohorts=5)
        
        start_time = time.time()
        result = transform_staggered_detrend(data, 'y', 'id', 'year', 'gvar')
        elapsed = time.time() - start_time
        
        print(f"\nMedium dataset (200 units × 15 periods): {elapsed:.3f}s")
        
        # Should complete in reasonable time
        assert elapsed < 10.0, f"Medium dataset took too long: {elapsed:.3f}s"
        
        ycheck_cols = [c for c in result.columns if c.startswith('ycheck_')]
        assert len(ycheck_cols) > 0
    
    def test_performance_large_dataset(self):
        """Performance test with large dataset."""
        data = self._create_large_dataset(n_units=500, n_periods=20, n_cohorts=5)
        
        start_time = time.time()
        result = transform_staggered_detrend(data, 'y', 'id', 'year', 'gvar')
        elapsed = time.time() - start_time
        
        print(f"\nLarge dataset (500 units × 20 periods): {elapsed:.3f}s")
        
        # Vectorized implementation should complete in reasonable time
        # Old nested loop would take minutes, vectorized should be seconds
        assert elapsed < 30.0, f"Large dataset took too long: {elapsed:.3f}s"
        
        ycheck_cols = [c for c in result.columns if c.startswith('ycheck_')]
        assert len(ycheck_cols) > 0
    
    def test_demean_vs_detrend_performance_ratio(self):
        """Compare performance ratio between demean and detrend.
        
        Both should have similar performance characteristics after optimization.
        """
        data = self._create_large_dataset(n_units=200, n_periods=12, n_cohorts=4)
        
        # Measure demean (already vectorized)
        start_time = time.time()
        _ = transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        demean_time = time.time() - start_time
        
        # Measure detrend (now vectorized)
        start_time = time.time()
        _ = transform_staggered_detrend(data, 'y', 'id', 'year', 'gvar')
        detrend_time = time.time() - start_time
        
        ratio = detrend_time / demean_time if demean_time > 0 else float('inf')
        
        print(f"\nDemean: {demean_time:.3f}s, Detrend: {detrend_time:.3f}s")
        print(f"Ratio (detrend/demean): {ratio:.2f}x")
        
        # Detrend should not be more than 5x slower than demean
        # (Before optimization, it could be 50x or more slower)
        assert ratio < 5.0, f"Detrend too slow compared to demean: {ratio:.2f}x"


class TestDetrendNumericalAccuracy:
    """Numerical accuracy tests comparing vectorized vs np.polyfit."""
    
    def test_perfect_linear_trend(self):
        """Test with perfectly linear data (no noise).
        
        For perfect linear data Y = A + B*t, residuals should be exactly 0
        in pre-treatment period and equal to treatment effect in post-period.
        """
        # Create data with perfect linear trend
        data = pd.DataFrame({
            'id': [1]*6 + [2]*6,
            'year': [1,2,3,4,5,6]*2,
            'y': [10, 12, 14, 26, 28, 30,   # unit 1: A=8, B=2, effect=10 at t>=4
                  5, 7, 9, 11, 13, 15],     # unit 2: A=3, B=2, no treatment (control)
            'gvar': [4]*6 + [0]*6
        })
        
        result = transform_staggered_detrend(data, 'y', 'id', 'year', 'gvar')
        
        # Unit 1 verification
        # OLS on t=1,2,3: Y = 10,12,14 -> A=8, B=2
        # Predicted t=4: 8 + 2*4 = 16, actual=26, residual=10 ✓
        # Predicted t=5: 8 + 2*5 = 18, actual=28, residual=10 ✓
        # Predicted t=6: 8 + 2*6 = 20, actual=30, residual=10 ✓
        
        ycheck_u1_r4 = result.loc[(result['id']==1) & (result['year']==4), 'ycheck_g4_r4'].iloc[0]
        ycheck_u1_r5 = result.loc[(result['id']==1) & (result['year']==5), 'ycheck_g4_r5'].iloc[0]
        ycheck_u1_r6 = result.loc[(result['id']==1) & (result['year']==6), 'ycheck_g4_r6'].iloc[0]
        
        assert np.isclose(ycheck_u1_r4, 10.0, atol=1e-10), f"Expected 10.0, got {ycheck_u1_r4}"
        assert np.isclose(ycheck_u1_r5, 10.0, atol=1e-10), f"Expected 10.0, got {ycheck_u1_r5}"
        assert np.isclose(ycheck_u1_r6, 10.0, atol=1e-10), f"Expected 10.0, got {ycheck_u1_r6}"
        
        # Unit 2 (control) verification
        # OLS on t=1,2,3: Y = 5,7,9 -> A=3, B=2
        # Predicted t=4: 3 + 2*4 = 11, actual=11, residual=0 ✓
        
        ycheck_u2_r4 = result.loc[(result['id']==2) & (result['year']==4), 'ycheck_g4_r4'].iloc[0]
        assert np.isclose(ycheck_u2_r4, 0.0, atol=1e-10), f"Expected 0.0, got {ycheck_u2_r4}"
    
    def test_numerical_precision_with_noise(self):
        """Test numerical precision with noisy data.
        
        Compare vectorized implementation output with manual np.polyfit calculation.
        """
        np.random.seed(123)
        
        data = pd.DataFrame({
            'id': [1]*5 + [2]*5,
            'year': [1,2,3,4,5]*2,
            'y': [10.1, 12.3, 13.8, 25.2, 27.1,   # unit 1 with noise
                  5.2, 7.1, 8.9, 10.8, 13.0],     # unit 2 with noise
            'gvar': [4]*5 + [0]*5
        })
        
        result = transform_staggered_detrend(data, 'y', 'id', 'year', 'gvar')
        
        # Manual calculation using np.polyfit for unit 1
        t_pre_u1 = np.array([1, 2, 3], dtype=float)
        y_pre_u1 = np.array([10.1, 12.3, 13.8], dtype=float)
        B_u1, A_u1 = np.polyfit(t_pre_u1, y_pre_u1, 1)
        
        # Compare residual at t=4
        predicted_u1_t4 = A_u1 + B_u1 * 4
        expected_residual_t4 = 25.2 - predicted_u1_t4
        actual_residual_t4 = result.loc[(result['id']==1) & (result['year']==4), 'ycheck_g4_r4'].iloc[0]
        
        assert np.isclose(actual_residual_t4, expected_residual_t4, atol=1e-10), \
            f"Expected {expected_residual_t4}, got {actual_residual_t4}"
        
        # Manual calculation for unit 2
        t_pre_u2 = np.array([1, 2, 3], dtype=float)
        y_pre_u2 = np.array([5.2, 7.1, 8.9], dtype=float)
        B_u2, A_u2 = np.polyfit(t_pre_u2, y_pre_u2, 1)
        
        predicted_u2_t4 = A_u2 + B_u2 * 4
        expected_residual_u2_t4 = 10.8 - predicted_u2_t4
        actual_residual_u2_t4 = result.loc[(result['id']==2) & (result['year']==4), 'ycheck_g4_r4'].iloc[0]
        
        assert np.isclose(actual_residual_u2_t4, expected_residual_u2_t4, atol=1e-10), \
            f"Expected {expected_residual_u2_t4}, got {actual_residual_u2_t4}"
    
    def test_multiple_cohorts_accuracy(self):
        """Test accuracy with multiple cohorts."""
        data = pd.DataFrame({
            'id': [1]*6 + [2]*6 + [3]*6,
            'year': [1,2,3,4,5,6]*3,
            'y': [10, 12, 14, 26, 28, 30,   # unit 1: cohort 4
                  20, 22, 24, 26, 38, 40,   # unit 2: cohort 5
                  5, 7, 9, 11, 13, 15],     # unit 3: never treated
            'gvar': [4]*6 + [5]*6 + [0]*6
        })
        
        result = transform_staggered_detrend(data, 'y', 'id', 'year', 'gvar')
        
        # Verify cohort 4 columns exist
        assert 'ycheck_g4_r4' in result.columns
        assert 'ycheck_g4_r5' in result.columns
        assert 'ycheck_g4_r6' in result.columns
        
        # Verify cohort 5 columns exist
        assert 'ycheck_g5_r5' in result.columns
        assert 'ycheck_g5_r6' in result.columns
        
        # Unit 1 at t=4 for cohort g=4: effect = 10
        ycheck_u1_g4_r4 = result.loc[(result['id']==1) & (result['year']==4), 'ycheck_g4_r4'].iloc[0]
        assert np.isclose(ycheck_u1_g4_r4, 10.0, atol=1e-10)
        
        # Unit 2 at t=5 for cohort g=5: effect = 10
        # Pre-treatment: t=1,2,3,4 -> Y=20,22,24,26 -> A=18, B=2
        # Predicted t=5: 18 + 2*5 = 28, actual=38, residual=10
        ycheck_u2_g5_r5 = result.loc[(result['id']==2) & (result['year']==5), 'ycheck_g5_r5'].iloc[0]
        assert np.isclose(ycheck_u2_g5_r5, 10.0, atol=1e-10)


class TestDetrendEdgeCases:
    """Edge case tests for vectorized detrend implementation."""
    
    def test_exactly_two_pre_periods(self):
        """Test with exactly 2 pre-treatment periods (minimum required)."""
        data = pd.DataFrame({
            'id': [1]*4 + [2]*4,
            'year': [1,2,3,4]*2,
            'y': [10, 12, 24, 26,   # unit 1: A=8, B=2, effect=10
                  5, 7, 9, 11],     # unit 2: A=3, B=2, no effect
            'gvar': [3]*4 + [0]*4
        })
        
        result = transform_staggered_detrend(data, 'y', 'id', 'year', 'gvar')
        
        # With only 2 pre-periods, trend fit is exact
        ycheck_u1_r3 = result.loc[(result['id']==1) & (result['year']==3), 'ycheck_g3_r3'].iloc[0]
        assert np.isclose(ycheck_u1_r3, 10.0, atol=1e-10)
    
    def test_missing_pre_period_data(self):
        """Test with missing pre-treatment data for some units."""
        data = pd.DataFrame({
            'id': [1,1,1,1, 2,2,2,2],
            'year': [1,2,3,4, 1,2,3,4],
            'y': [10, np.nan, 14, 20,   # unit 1: missing t=2
                  5, 6, 7, 8],           # unit 2: complete
            'gvar': [3,3,3,3, 0,0,0,0]
        })
        
        result = transform_staggered_detrend(data, 'y', 'id', 'year', 'gvar')
        
        # Unit 1 has only 1 valid pre-period (t=1), should be NaN
        ycheck_u1_r3 = result.loc[(result['id']==1) & (result['year']==3), 'ycheck_g3_r3'].iloc[0]
        assert np.isnan(ycheck_u1_r3), "Should be NaN with insufficient pre-periods"
        
        # Unit 2 should have valid result
        ycheck_u2_r3 = result.loc[(result['id']==2) & (result['year']==3), 'ycheck_g3_r3'].iloc[0]
        assert not np.isnan(ycheck_u2_r3), "Should have valid result"
    
    def test_all_pre_periods_same_value(self):
        """Test when all pre-period Y values are the same (zero variance in t not relevant here)."""
        data = pd.DataFrame({
            'id': [1]*4 + [2]*4,
            'year': [1,2,3,4]*2,
            'y': [10, 10, 10, 20,   # unit 1: constant pre-period
                  5, 6, 7, 8],      # unit 2: normal trend
            'gvar': [3,3,3,3, 0,0,0,0]
        })
        
        result = transform_staggered_detrend(data, 'y', 'id', 'year', 'gvar')
        
        # Unit 1 has constant Y in pre-period, slope=0, intercept=10
        # Predicted at t=3: 10, actual=10, residual=0 (in pre-period)
        # Predicted at t=4: 10, actual=20, residual=10
        ycheck_u1_r3 = result.loc[(result['id']==1) & (result['year']==3), 'ycheck_g3_r3'].iloc[0]
        ycheck_u1_r4 = result.loc[(result['id']==1) & (result['year']==4), 'ycheck_g3_r4'].iloc[0]
        
        # Note: even with constant Y, the regression is valid (slope=0)
        assert np.isclose(ycheck_u1_r4, 10.0, atol=1e-10)
    
    def test_single_time_point_all_same(self):
        """Test edge case where all units have same single time point.
        
        This should be caught by the 'at least 2 pre-periods' validation.
        """
        data = pd.DataFrame({
            'id': [1,1, 2,2],
            'year': [1,2, 1,2],
            'y': [10, 20, 5, 8],
            'gvar': [2,2, 0,0]  # cohort=2, pre-period only t=1
        })
        
        with pytest.raises(ValueError, match="at least 2"):
            transform_staggered_detrend(data, 'y', 'id', 'year', 'gvar')


class TestVibeMathVerification:
    """Tests using analytical verification of linear regression formulas."""
    
    def test_ols_formula_equivalence(self):
        """Verify that vectorized formula matches OLS closed-form solution.
        
        For simple linear regression:
        B = Σ(x-x̄)(y-ȳ) / Σ(x-x̄)² = [Σxy - n*x̄*ȳ] / [Σx² - n*x̄²]
        A = ȳ - B*x̄
        """
        # Simple test case with known solution
        t = np.array([1, 2, 3], dtype=float)
        y = np.array([10, 12, 14], dtype=float)
        
        # Manual calculation
        n = len(t)
        t_mean = t.mean()  # 2
        y_mean = y.mean()  # 12
        
        # Vectorized formula (as implemented)
        t2_sum = np.sum(t ** 2)  # 14
        ty_sum = np.sum(t * y)   # 10 + 24 + 42 = 76
        var_t = t2_sum / n - t_mean ** 2  # 14/3 - 4 = 2/3
        cov_ty = ty_sum / n - t_mean * y_mean  # 76/3 - 2*12 = 76/3 - 24 = 4/3
        
        B_vectorized = cov_ty / var_t  # (4/3) / (2/3) = 2
        A_vectorized = y_mean - B_vectorized * t_mean  # 12 - 2*2 = 8
        
        # np.polyfit for comparison
        B_polyfit, A_polyfit = np.polyfit(t, y, 1)
        
        assert np.isclose(B_vectorized, B_polyfit, atol=1e-10), \
            f"Slope mismatch: vectorized={B_vectorized}, polyfit={B_polyfit}"
        assert np.isclose(A_vectorized, A_polyfit, atol=1e-10), \
            f"Intercept mismatch: vectorized={A_vectorized}, polyfit={A_polyfit}"
        
        # Expected values for this specific case
        assert np.isclose(B_vectorized, 2.0, atol=1e-10)
        assert np.isclose(A_vectorized, 8.0, atol=1e-10)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

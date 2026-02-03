"""
Numerical validation tests for sensitivity analysis module.

Verifies correctness of all numerical computations including:
- Sensitivity ratio calculation
- Robustness level determination
- Data filtering logic
"""

import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal

from lwdid.sensitivity import (
    RobustnessLevel,
    _compute_sensitivity_ratio,
    _determine_robustness_level,
    _filter_to_n_pre_periods,
    _filter_excluding_periods,
    _auto_detect_pre_period_range,
)


class TestSensitivityRatioNumerical:
    """Numerical validation of sensitivity ratio calculation."""
    
    def test_sensitivity_ratio_exact_positive(self):
        """Test sensitivity ratio with known positive values."""
        # ATT estimates: [1.0, 1.1, 1.2, 1.3, 1.4]
        # Baseline: 1.4
        # Range: 1.4 - 1.0 = 0.4
        # Ratio: 0.4 / 1.4 = 0.2857142857...
        
        atts = [1.0, 1.1, 1.2, 1.3, 1.4]
        baseline = 1.4
        
        ratio = _compute_sensitivity_ratio(atts, baseline)
        expected = 0.4 / 1.4
        
        assert_allclose(ratio, expected, rtol=1e-10)
    
    def test_sensitivity_ratio_exact_negative(self):
        """Test with negative baseline ATT."""
        # ATT estimates: [-1.0, -1.1, -1.2]
        # Baseline: -1.2
        # Range: -1.0 - (-1.2) = 0.2
        # Ratio: 0.2 / |-1.2| = 0.1666...
        
        atts = [-1.0, -1.1, -1.2]
        baseline = -1.2
        
        ratio = _compute_sensitivity_ratio(atts, baseline)
        expected = 0.2 / 1.2
        
        assert_allclose(ratio, expected, rtol=1e-10)
    
    def test_sensitivity_ratio_mixed_signs(self):
        """Test with mixed sign ATT estimates."""
        # ATT estimates: [-0.5, 0.0, 0.5]
        # Baseline: 0.5
        # Range: 0.5 - (-0.5) = 1.0
        # Ratio: 1.0 / 0.5 = 2.0
        
        atts = [-0.5, 0.0, 0.5]
        baseline = 0.5
        
        ratio = _compute_sensitivity_ratio(atts, baseline)
        expected = 1.0 / 0.5
        
        assert_allclose(ratio, expected, rtol=1e-10)
    
    def test_sensitivity_ratio_zero_baseline(self):
        """Test with baseline near zero."""
        atts = [0.01, 0.02, 0.03]
        baseline = 0.0
        
        ratio = _compute_sensitivity_ratio(atts, baseline)
        
        # Should return inf when baseline is zero and range is non-zero
        assert ratio == float('inf')
    
    def test_sensitivity_ratio_zero_range(self):
        """Test with zero range (identical estimates)."""
        atts = [1.5, 1.5, 1.5]
        baseline = 1.5
        
        ratio = _compute_sensitivity_ratio(atts, baseline)
        
        assert ratio == 0.0
    
    def test_sensitivity_ratio_single_value(self):
        """Test with single ATT value."""
        atts = [2.0]
        baseline = 2.0
        
        ratio = _compute_sensitivity_ratio(atts, baseline)
        
        assert ratio == 0.0
    
    def test_sensitivity_ratio_empty_list(self):
        """Test with empty ATT list."""
        ratio = _compute_sensitivity_ratio([], 1.0)
        assert ratio == 0.0
    
    @pytest.mark.parametrize("atts,baseline,expected", [
        ([1.0, 1.5, 2.0], 2.0, 0.5),      # (2.0-1.0)/2.0 = 0.5
        ([0.8, 0.9, 1.0], 1.0, 0.2),      # (1.0-0.8)/1.0 = 0.2
        ([-2.0, -1.5, -1.0], -1.0, 1.0),  # (1.0)/1.0 = 1.0
        ([0.5, 0.5, 0.5], 0.5, 0.0),      # zero range
    ])
    def test_sensitivity_ratio_parametrized(self, atts, baseline, expected):
        """Parametrized test for various scenarios."""
        ratio = _compute_sensitivity_ratio(atts, baseline)
        assert_allclose(ratio, expected, rtol=1e-10)


class TestRobustnessLevelNumerical:
    """Numerical validation of robustness level determination."""
    
    @pytest.mark.parametrize("ratio,expected_level", [
        # Highly robust: < 10%
        (0.00, RobustnessLevel.HIGHLY_ROBUST),
        (0.05, RobustnessLevel.HIGHLY_ROBUST),
        (0.09, RobustnessLevel.HIGHLY_ROBUST),
        (0.099, RobustnessLevel.HIGHLY_ROBUST),
        
        # Moderately robust: 10% <= ratio < 25%
        (0.10, RobustnessLevel.MODERATELY_ROBUST),
        (0.15, RobustnessLevel.MODERATELY_ROBUST),
        (0.20, RobustnessLevel.MODERATELY_ROBUST),
        (0.24, RobustnessLevel.MODERATELY_ROBUST),
        (0.249, RobustnessLevel.MODERATELY_ROBUST),
        
        # Sensitive: 25% <= ratio < 50%
        (0.25, RobustnessLevel.SENSITIVE),
        (0.30, RobustnessLevel.SENSITIVE),
        (0.40, RobustnessLevel.SENSITIVE),
        (0.49, RobustnessLevel.SENSITIVE),
        (0.499, RobustnessLevel.SENSITIVE),
        
        # Highly sensitive: >= 50%
        (0.50, RobustnessLevel.HIGHLY_SENSITIVE),
        (0.75, RobustnessLevel.HIGHLY_SENSITIVE),
        (1.00, RobustnessLevel.HIGHLY_SENSITIVE),
        (2.00, RobustnessLevel.HIGHLY_SENSITIVE),
        (10.0, RobustnessLevel.HIGHLY_SENSITIVE),
    ])
    def test_robustness_level_thresholds(self, ratio, expected_level):
        """Test robustness level determination at boundary values."""
        level = _determine_robustness_level(ratio)
        assert level == expected_level
    
    def test_boundary_precision(self):
        """Test boundary precision at exact thresholds."""
        # Just below 10%
        assert _determine_robustness_level(0.09999999) == RobustnessLevel.HIGHLY_ROBUST
        # Exactly 10%
        assert _determine_robustness_level(0.10) == RobustnessLevel.MODERATELY_ROBUST
        
        # Just below 25%
        assert _determine_robustness_level(0.24999999) == RobustnessLevel.MODERATELY_ROBUST
        # Exactly 25%
        assert _determine_robustness_level(0.25) == RobustnessLevel.SENSITIVE
        
        # Just below 50%
        assert _determine_robustness_level(0.49999999) == RobustnessLevel.SENSITIVE
        # Exactly 50%
        assert _determine_robustness_level(0.50) == RobustnessLevel.HIGHLY_SENSITIVE


class TestDataFilteringNumerical:
    """Numerical validation of data filtering logic."""
    
    def test_filter_common_timing_correct_periods(self):
        """Verify correct periods are selected for common timing."""
        # Create data with periods 1-10, treatment at 6
        data = pd.DataFrame({
            'unit': np.repeat(range(10), 10),
            'time': np.tile(range(1, 11), 10),
            'post': np.tile([0]*5 + [1]*5, 10),
            'Y': np.random.randn(100),
        })
        
        # Filter to 3 pre-periods (should keep periods 3, 4, 5)
        filtered = _filter_to_n_pre_periods(
            data, ivar='unit', tvar='time', gvar=None, d=None, post='post',
            n_pre_periods=3, exclude_periods=0
        )
        
        pre_times = set(filtered[filtered['post'] == 0]['time'].unique())
        assert pre_times == {3, 4, 5}
        
        # Post-treatment periods should be unchanged
        post_times = set(filtered[filtered['post'] == 1]['time'].unique())
        assert post_times == {6, 7, 8, 9, 10}
    
    def test_filter_with_exclusion_correct_periods(self):
        """Verify exclusion of periods before treatment."""
        data = pd.DataFrame({
            'unit': np.repeat(range(10), 10),
            'time': np.tile(range(1, 11), 10),
            'post': np.tile([0]*5 + [1]*5, 10),
            'Y': np.random.randn(100),
        })
        
        # Filter to 3 pre-periods, excluding 1 period before treatment
        # Should keep periods 2, 3, 4 (not 5)
        filtered = _filter_to_n_pre_periods(
            data, ivar='unit', tvar='time', gvar=None, d=None, post='post',
            n_pre_periods=3, exclude_periods=1
        )
        
        pre_times = set(filtered[filtered['post'] == 0]['time'].unique())
        assert pre_times == {2, 3, 4}
        assert 5 not in pre_times  # Period 5 excluded
    
    def test_filter_excluding_periods_common_timing(self):
        """Test _filter_excluding_periods for common timing."""
        data = pd.DataFrame({
            'unit': np.repeat(range(5), 8),
            'time': np.tile(range(1, 9), 5),
            'post': np.tile([0]*4 + [1]*4, 5),
            'Y': np.random.randn(40),
        })
        
        # Exclude 2 periods before treatment (periods 3, 4)
        filtered = _filter_excluding_periods(
            data, ivar='unit', tvar='time', gvar=None, post='post',
            exclude_periods=2
        )
        
        pre_times = set(filtered[filtered['post'] == 0]['time'].unique())
        assert pre_times == {1, 2}
        assert 3 not in pre_times
        assert 4 not in pre_times
    
    def test_filter_preserves_all_units(self):
        """Verify filtering preserves all units."""
        n_units = 20
        data = pd.DataFrame({
            'unit': np.repeat(range(n_units), 8),
            'time': np.tile(range(1, 9), n_units),
            'post': np.tile([0]*4 + [1]*4, n_units),
            'Y': np.random.randn(n_units * 8),
        })
        
        filtered = _filter_to_n_pre_periods(
            data, ivar='unit', tvar='time', gvar=None, d=None, post='post',
            n_pre_periods=2, exclude_periods=0
        )
        
        # All units should still be present
        assert filtered['unit'].nunique() == n_units
    
    def test_filter_observation_count(self):
        """Verify correct observation count after filtering."""
        n_units = 10
        n_periods = 10
        n_pre = 5
        n_post = 5
        
        data = pd.DataFrame({
            'unit': np.repeat(range(n_units), n_periods),
            'time': np.tile(range(1, n_periods + 1), n_units),
            'post': np.tile([0]*n_pre + [1]*n_post, n_units),
            'Y': np.random.randn(n_units * n_periods),
        })
        
        # Filter to 3 pre-periods
        filtered = _filter_to_n_pre_periods(
            data, ivar='unit', tvar='time', gvar=None, d=None, post='post',
            n_pre_periods=3, exclude_periods=0
        )
        
        # Expected: 3 pre + 5 post = 8 periods per unit
        expected_obs = n_units * (3 + n_post)
        assert len(filtered) == expected_obs


class TestAutoDetectRangeNumerical:
    """Numerical validation of auto-detection logic."""
    
    def test_common_timing_range_detection(self):
        """Test range detection for common timing."""
        # 5 pre-treatment periods, 5 post-treatment periods
        data = pd.DataFrame({
            'unit': np.repeat(range(10), 10),
            'time': np.tile(range(1, 11), 10),
            'post': np.tile([0]*5 + [1]*5, 10),
            'Y': np.random.randn(100),
        })
        
        min_pre, max_pre = _auto_detect_pre_period_range(
            data, ivar='unit', tvar='time',
            gvar=None, d=None, post='post', rolling='demean'
        )
        
        assert min_pre == 1  # demean minimum
        assert max_pre == 5  # 5 pre-treatment periods
    
    def test_detrend_minimum_requirement(self):
        """Test minimum requirement for detrend."""
        data = pd.DataFrame({
            'unit': np.repeat(range(10), 10),
            'time': np.tile(range(1, 11), 10),
            'post': np.tile([0]*5 + [1]*5, 10),
            'Y': np.random.randn(100),
        })
        
        min_pre, max_pre = _auto_detect_pre_period_range(
            data, ivar='unit', tvar='time',
            gvar=None, d=None, post='post', rolling='detrend'
        )
        
        assert min_pre == 2  # detrend requires at least 2
        assert max_pre == 5
    
    def test_staggered_range_detection(self):
        """Test range detection for staggered design."""
        # Cohorts at periods 5, 7, 9
        data = []
        for i in range(30):
            cohort = [5, 7, 9][i % 3] if i < 20 else 0  # 20 treated, 10 never-treated
            for t in range(1, 12):
                data.append({
                    'unit': i,
                    'time': t,
                    'first_treat': cohort,
                    'Y': np.random.randn(),
                })
        data = pd.DataFrame(data)
        
        min_pre, max_pre = _auto_detect_pre_period_range(
            data, ivar='unit', tvar='time',
            gvar='first_treat', d=None, post=None, rolling='demean'
        )
        
        assert min_pre == 1
        # Max should be limited by earliest cohort (5 - 1 = 4 pre-periods)
        assert max_pre == 4


class TestNumericalPrecision:
    """Test numerical precision and edge cases."""
    
    def test_very_small_values(self):
        """Test with very small ATT values."""
        atts = [1e-10, 2e-10, 3e-10]
        baseline = 3e-10
        
        ratio = _compute_sensitivity_ratio(atts, baseline)
        expected = (3e-10 - 1e-10) / 3e-10
        
        assert_allclose(ratio, expected, rtol=1e-6)
    
    def test_very_large_values(self):
        """Test with very large ATT values."""
        atts = [1e10, 1.1e10, 1.2e10]
        baseline = 1.2e10
        
        ratio = _compute_sensitivity_ratio(atts, baseline)
        expected = (1.2e10 - 1e10) / 1.2e10
        
        assert_allclose(ratio, expected, rtol=1e-6)
    
    def test_nan_handling(self):
        """Test handling of NaN values."""
        # NaN in ATT list should be handled
        atts = [1.0, np.nan, 2.0]
        baseline = 2.0
        
        # Filter out NaN before computing
        valid_atts = [a for a in atts if not np.isnan(a)]
        ratio = _compute_sensitivity_ratio(valid_atts, baseline)
        
        expected = (2.0 - 1.0) / 2.0
        assert_allclose(ratio, expected, rtol=1e-10)

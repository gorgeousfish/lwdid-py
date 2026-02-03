# -*- coding: utf-8 -*-
"""
Edge case and boundary condition tests.

Based on Lee & Wooldridge (2026) ssrn-5325686, Section 5.

These tests verify proper handling of edge cases and boundary conditions.
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

# Add fixtures paths
fixtures_path = Path(__file__).parent / 'fixtures'
parent_fixtures = Path(__file__).parent.parent / 'test_common_timing' / 'fixtures'
sys.path.insert(0, str(fixtures_path))
sys.path.insert(0, str(parent_fixtures))

from dgp_small_sample import (
    generate_small_sample_dgp,
    SmallSampleDGPParams,
)
from monte_carlo_runner import (
    _estimate_manual_demean,
    _estimate_manual_detrend,
)


class TestMinimumSampleSize:
    """Tests for minimum sample size requirements."""
    
    def test_minimum_units_n3(self):
        """
        Test with minimum number of units (N=3).
        
        Paper requirement: N0 >= 1, N1 >= 1, N >= 3
        """
        data, params = generate_small_sample_dgp(
            n_units=3,
            n_periods=10,
            treatment_start=6,
            scenario=1,
            seed=42,
        )
        
        # Should have at least 1 treated and 1 control
        assert params['n_treated'] >= 1
        assert params['n_control'] >= 1
        assert params['n_treated'] + params['n_control'] == 3
        
        # Estimation should still work
        result = _estimate_manual_demean(data, treatment_start=6)
        
        # May have NaN due to small sample, but should not crash
        assert 'att' in result
    
    def test_minimum_periods_n3(self):
        """
        Test with minimum number of periods (T=3).
        """
        data, params = generate_small_sample_dgp(
            n_units=20,
            n_periods=3,
            treatment_start=2,
            scenario=1,
            seed=42,
        )
        
        assert len(data) == 20 * 3
        assert params['n_pre_periods'] == 1
        assert params['n_post_periods'] == 2
    
    def test_single_pre_treatment_period(self):
        """
        Test with single pre-treatment period.
        
        Demeaning requires at least 1 pre-treatment period.
        """
        data, params = generate_small_sample_dgp(
            n_units=20,
            n_periods=10,
            treatment_start=2,  # Only period 1 is pre-treatment
            scenario=1,
            seed=42,
        )
        
        assert params['n_pre_periods'] == 1
        
        # Demeaning should work with 1 pre-treatment period
        result = _estimate_manual_demean(data, treatment_start=2)
        assert 'att' in result
    
    def test_two_pre_treatment_periods_for_detrend(self):
        """
        Test with two pre-treatment periods (minimum for detrending).
        
        Detrending requires at least 2 pre-treatment periods for trend estimation.
        """
        data, params = generate_small_sample_dgp(
            n_units=20,
            n_periods=10,
            treatment_start=3,  # Periods 1-2 are pre-treatment
            scenario=1,
            seed=42,
        )
        
        assert params['n_pre_periods'] == 2
        
        # Detrending should work with 2 pre-treatment periods
        result = _estimate_manual_detrend(data, treatment_start=3)
        assert 'att' in result


class TestDegenerateScenarios:
    """Tests for degenerate scenarios."""
    
    def test_all_units_same_treatment_status_forced(self):
        """
        Test that DGP forces at least 1 treated and 1 control.
        """
        # Even with extreme probabilities, should have both groups
        for seed in range(20):
            data, params = generate_small_sample_dgp(
                scenario=3,  # Low treatment probability
                seed=seed,
            )
            
            assert params['n_treated'] >= 1
            assert params['n_control'] >= 1
    
    def test_zero_variance_components(self):
        """
        Test with zero variance in some components.
        """
        # Zero variance in unit fixed effects
        data, params = generate_small_sample_dgp(
            sigma_c=0.0,  # No unit fixed effects
            sigma_g=1.0,
            seed=42,
        )
        
        assert len(data) == 20 * 20
        
        # Estimation should still work
        result = _estimate_manual_demean(data, treatment_start=11)
        assert 'att' in result
    
    def test_zero_trend_variance(self):
        """
        Test with zero variance in unit trends.
        """
        data, params = generate_small_sample_dgp(
            sigma_c=2.0,
            sigma_g=0.0,  # No heterogeneous trends
            seed=42,
        )
        
        # All units should have same trend slope (mean = 1)
        if 'G' in params:
            assert np.allclose(params['G'], 1.0)
        
        result = _estimate_manual_detrend(data, treatment_start=11)
        assert 'att' in result


class TestNumericalStability:
    """Tests for numerical stability."""
    
    def test_large_outcome_values(self):
        """
        Test with large outcome values.
        """
        data, params = generate_small_sample_dgp(seed=42)
        
        # Scale outcomes by large factor
        data = data.copy()
        data['y'] = data['y'] * 1e6
        
        result = _estimate_manual_demean(data, treatment_start=11)
        
        # Should still produce valid results
        assert np.isfinite(result['att'])
        assert np.isfinite(result['se_ols']) or np.isnan(result['se_ols'])
    
    def test_small_outcome_values(self):
        """
        Test with small outcome values.
        """
        data, params = generate_small_sample_dgp(seed=42)
        
        # Scale outcomes by small factor
        data = data.copy()
        data['y'] = data['y'] * 1e-6
        
        result = _estimate_manual_demean(data, treatment_start=11)
        
        # Should still produce valid results
        assert np.isfinite(result['att'])
    
    def test_outcome_with_outliers(self):
        """
        Test with outliers in outcome.
        """
        data, params = generate_small_sample_dgp(seed=42)
        
        # Add outliers
        data = data.copy()
        outlier_idx = data.sample(5).index
        data.loc[outlier_idx, 'y'] = data.loc[outlier_idx, 'y'] * 100
        
        result = _estimate_manual_demean(data, treatment_start=11)
        
        # Should still produce results (may be affected by outliers)
        assert 'att' in result


class TestParameterBoundaries:
    """Tests for parameter boundary conditions."""
    
    def test_treatment_start_at_period_2(self):
        """
        Test with treatment starting at period 2 (minimum valid).
        """
        data, params = generate_small_sample_dgp(
            n_periods=10,
            treatment_start=2,
            seed=42,
        )
        
        assert params['n_pre_periods'] == 1
        assert params['n_post_periods'] == 9
    
    def test_treatment_start_at_last_period(self):
        """
        Test with treatment starting at last period.
        """
        data, params = generate_small_sample_dgp(
            n_periods=10,
            treatment_start=10,
            seed=42,
        )
        
        assert params['n_pre_periods'] == 9
        assert params['n_post_periods'] == 1
    
    def test_rho_near_boundary(self):
        """
        Test with AR(1) coefficient near boundary.
        """
        # High persistence
        data, params = generate_small_sample_dgp(
            rho=0.95,
            seed=42,
        )
        
        assert len(data) == 20 * 20
        
        # Low persistence
        data, params = generate_small_sample_dgp(
            rho=0.1,
            seed=42,
        )
        
        assert len(data) == 20 * 20


class TestDataIntegrity:
    """Tests for data integrity."""
    
    def test_no_missing_values(self):
        """
        Generated data should have no missing values.
        """
        data, params = generate_small_sample_dgp(seed=42)
        
        assert not data['id'].isna().any()
        assert not data['year'].isna().any()
        assert not data['y'].isna().any()
        assert not data['d'].isna().any()
        assert not data['post'].isna().any()
    
    def test_correct_data_types(self):
        """
        Data should have correct types.
        """
        data, params = generate_small_sample_dgp(seed=42)
        
        assert data['id'].dtype in [np.int64, np.int32, int]
        assert data['year'].dtype in [np.int64, np.int32, int]
        assert data['y'].dtype in [np.float64, np.float32, float]
        assert data['d'].dtype in [np.int64, np.int32, int]
        assert data['post'].dtype in [np.int64, np.int32, int]
    
    def test_treatment_binary(self):
        """
        Treatment should be binary (0 or 1).
        """
        data, params = generate_small_sample_dgp(seed=42)
        
        assert set(data['d'].unique()).issubset({0, 1})
    
    def test_post_binary(self):
        """
        Post indicator should be binary (0 or 1).
        """
        data, params = generate_small_sample_dgp(seed=42)
        
        assert set(data['post'].unique()).issubset({0, 1})


class TestReproducibility:
    """Tests for reproducibility."""
    
    def test_same_seed_same_data(self):
        """
        Same seed should produce identical data.
        """
        data1, params1 = generate_small_sample_dgp(seed=12345)
        data2, params2 = generate_small_sample_dgp(seed=12345)
        
        pd.testing.assert_frame_equal(data1, data2)
        assert params1['n_treated'] == params2['n_treated']
    
    def test_different_seed_different_data(self):
        """
        Different seeds should produce different data.
        """
        data1, _ = generate_small_sample_dgp(seed=12345)
        data2, _ = generate_small_sample_dgp(seed=12346)
        
        assert not data1['y'].equals(data2['y'])
    
    def test_seed_none_produces_random_data(self):
        """
        Seed=None should produce different data each time.
        """
        data1, _ = generate_small_sample_dgp(seed=None)
        data2, _ = generate_small_sample_dgp(seed=None)
        
        # Very unlikely to be identical
        # (Could fail with extremely low probability)
        assert not data1['y'].equals(data2['y'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

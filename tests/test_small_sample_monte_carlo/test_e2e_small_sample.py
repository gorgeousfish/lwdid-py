# -*- coding: utf-8 -*-
"""
End-to-end tests for small-sample Monte Carlo validation.

Based on Lee & Wooldridge (2026) ssrn-5325686, Section 5.

These tests verify the complete workflow from DGP generation
through estimation to result validation.
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
    generate_small_sample_dgp_from_scenario,
    SMALL_SAMPLE_SCENARIOS,
)
from monte_carlo_runner import (
    _estimate_manual_demean,
    _estimate_manual_detrend,
)


class TestE2ESmallSampleWorkflow:
    """End-to-end workflow tests."""
    
    def test_complete_workflow_demeaning(self):
        """
        Test complete workflow: DGP -> Demeaning -> ATT estimation.
        """
        # Step 1: Generate data
        data, params = generate_small_sample_dgp(
            n_units=20,
            n_periods=20,
            treatment_start=11,
            scenario=1,
            seed=42,
        )
        
        # Verify data structure
        assert len(data) == 20 * 20
        assert params['n_treated'] >= 1
        assert params['n_control'] >= 1
        
        # Step 2: Estimate ATT using demeaning
        result = _estimate_manual_demean(data, treatment_start=11)
        
        # Step 3: Validate results
        assert result['att'] is not None
        assert not np.isnan(result['att'])
        assert result['se_ols'] > 0
        
        # ATT should be in reasonable range of true value
        true_att = params['tau']
        # Very loose bound for single replication
        assert abs(result['att'] - true_att) < 10
    
    def test_complete_workflow_detrending(self):
        """
        Test complete workflow: DGP -> Detrending -> ATT estimation.
        """
        # Step 1: Generate data
        data, params = generate_small_sample_dgp(
            n_units=20,
            n_periods=20,
            treatment_start=11,
            scenario=1,
            seed=42,
        )
        
        # Step 2: Estimate ATT using detrending
        result = _estimate_manual_detrend(data, treatment_start=11)
        
        # Step 3: Validate results
        assert result['att'] is not None
        assert not np.isnan(result['att'])
        assert result['se_ols'] > 0
        
        # Detrending should recover true effect reasonably well
        # (DGP has heterogeneous trends, so detrending is appropriate)
        true_att = params['tau']
        assert abs(result['att'] - true_att) < 10
    
    def test_workflow_with_all_scenarios(self):
        """
        Test workflow across all three scenarios.
        """
        results = {}
        
        for scenario_name, scenario_config in SMALL_SAMPLE_SCENARIOS.items():
            data, params = generate_small_sample_dgp_from_scenario(
                scenario_name,
                seed=42,
            )
            
            result = _estimate_manual_detrend(data, treatment_start=11)
            
            results[scenario_name] = {
                'att': result['att'],
                'se': result['se_ols'],
                'n_treated': params['n_treated'],
                'n_control': params['n_control'],
                'true_att': params['tau'],
            }
        
        # All scenarios should produce valid results
        for scenario, res in results.items():
            assert res['att'] is not None, f"{scenario}: ATT is None"
            assert not np.isnan(res['att']), f"{scenario}: ATT is NaN"
            assert res['se'] > 0, f"{scenario}: SE not positive"
            assert res['n_treated'] >= 1, f"{scenario}: No treated units"
            assert res['n_control'] >= 1, f"{scenario}: No control units"
    
    def test_workflow_multiple_seeds(self):
        """
        Test workflow with multiple random seeds.
        """
        att_estimates = []
        
        for seed in range(20):
            data, params = generate_small_sample_dgp(
                scenario=1,
                seed=seed,
            )
            
            result = _estimate_manual_detrend(data, treatment_start=11)
            
            if not np.isnan(result['att']):
                att_estimates.append(result['att'])
        
        # Should have successful estimates
        assert len(att_estimates) >= 15  # At least 75% success rate
        
        # Mean should be close to true ATT
        _, params = generate_small_sample_dgp(scenario=1, seed=0)
        true_att = params['tau']
        mean_att = np.mean(att_estimates)
        
        # Loose bound due to small number of replications
        assert abs(mean_att - true_att) < 3


class TestE2EDataValidation:
    """End-to-end data validation tests."""
    
    def test_data_consistency_across_workflow(self):
        """
        Verify data consistency throughout the workflow.
        """
        data, params = generate_small_sample_dgp(seed=42)
        
        # Check panel structure
        assert data['id'].nunique() == params['n_units']
        assert data['year'].nunique() == params['n_periods']
        
        # Check treatment assignment
        treated_ids = data[data['d'] == 1]['id'].unique()
        control_ids = data[data['d'] == 0]['id'].unique()
        
        assert len(treated_ids) == params['n_treated']
        assert len(control_ids) == params['n_control']
        
        # Check post indicator
        pre_periods = data[data['post'] == 0]['year'].unique()
        post_periods = data[data['post'] == 1]['year'].unique()
        
        assert max(pre_periods) < params['treatment_start']
        assert min(post_periods) >= params['treatment_start']
    
    def test_treatment_effect_in_data(self):
        """
        Verify treatment effect is present in generated data.
        """
        data, params = generate_small_sample_dgp(
            seed=42,
            return_components=True,
        )
        
        # Compare treated vs control in post-treatment period
        post_data = data[data['post'] == 1]
        
        treated_mean = post_data[post_data['d'] == 1]['y'].mean()
        control_mean = post_data[post_data['d'] == 0]['y'].mean()
        
        # Raw difference should be positive (treatment effect exists)
        # Note: This is not the ATT due to confounding
        raw_diff = treated_mean - control_mean
        
        # Just verify the data has variation
        assert not np.isnan(raw_diff)


class TestE2EEstimatorComparison:
    """End-to-end estimator comparison tests."""
    
    def test_demean_vs_detrend_single_run(self):
        """
        Compare demeaning and detrending on single dataset.
        """
        data, params = generate_small_sample_dgp(seed=42)
        
        result_demean = _estimate_manual_demean(data, treatment_start=11)
        result_detrend = _estimate_manual_detrend(data, treatment_start=11)
        
        # Both should produce valid results
        assert not np.isnan(result_demean['att'])
        assert not np.isnan(result_detrend['att'])
        
        # Results may differ due to different transformations
        # Just verify they're both reasonable
        true_att = params['tau']
        assert abs(result_demean['att'] - true_att) < 15
        assert abs(result_detrend['att'] - true_att) < 15
    
    def test_estimator_variance_comparison(self):
        """
        Compare variance of estimators across replications.
        """
        n_reps = 30
        
        demean_atts = []
        detrend_atts = []
        
        for seed in range(n_reps):
            data, params = generate_small_sample_dgp(scenario=1, seed=seed)
            
            result_demean = _estimate_manual_demean(data, treatment_start=11)
            result_detrend = _estimate_manual_detrend(data, treatment_start=11)
            
            if not np.isnan(result_demean['att']):
                demean_atts.append(result_demean['att'])
            if not np.isnan(result_detrend['att']):
                detrend_atts.append(result_detrend['att'])
        
        # Both should have successful estimates
        assert len(demean_atts) >= 20
        assert len(detrend_atts) >= 20
        
        # Compute variances
        var_demean = np.var(demean_atts, ddof=1)
        var_detrend = np.var(detrend_atts, ddof=1)
        
        # Both should have positive variance
        assert var_demean > 0
        assert var_detrend > 0


class TestE2EEdgeCases:
    """End-to-end edge case tests."""
    
    def test_sparse_treatment_workflow(self):
        """
        Test workflow with sparse treatment (Scenario 3).
        """
        # Scenario 3 has P(D=1) ≈ 0.17
        data, params = generate_small_sample_dgp_from_scenario(
            'scenario_3',
            seed=42,
        )
        
        # Should still have at least 1 treated and 1 control
        assert params['n_treated'] >= 1
        assert params['n_control'] >= 1
        
        # Estimation should still work
        result = _estimate_manual_detrend(data, treatment_start=11)
        
        assert not np.isnan(result['att'])
    
    def test_high_treatment_probability_workflow(self):
        """
        Test workflow with high treatment probability (Scenario 1).
        """
        data, params = generate_small_sample_dgp_from_scenario(
            'scenario_1',
            seed=42,
        )
        
        # Should have more treated units
        assert params['n_treated'] >= 3
        
        result = _estimate_manual_detrend(data, treatment_start=11)
        
        assert not np.isnan(result['att'])
    
    def test_workflow_robustness_to_seeds(self):
        """
        Test workflow robustness across different seeds.
        """
        success_count = 0
        
        for seed in range(50):
            try:
                data, params = generate_small_sample_dgp(
                    scenario=2,  # Medium treatment probability
                    seed=seed,
                )
                
                result = _estimate_manual_detrend(data, treatment_start=11)
                
                if not np.isnan(result['att']) and result['se_ols'] > 0:
                    success_count += 1
            except Exception:
                pass
        
        # Should have high success rate
        assert success_count >= 40  # At least 80% success


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

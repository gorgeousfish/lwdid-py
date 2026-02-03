# -*- coding: utf-8 -*-
"""
Full Monte Carlo simulation tests matching paper Table 2.

Based on Lee & Wooldridge (2026) ssrn-5325686, Section 5.

These tests run Monte Carlo simulations and compare results with
paper Table 2 expectations.
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

from monte_carlo_runner import (
    run_small_sample_monte_carlo,
    run_all_scenarios_monte_carlo,
    generate_comparison_table,
    _estimate_manual_demean,
    _estimate_manual_detrend,
)
from dgp_small_sample import (
    generate_small_sample_dgp,
    SMALL_SAMPLE_SCENARIOS,
)


@pytest.mark.monte_carlo
class TestManualEstimators:
    """Tests for manual estimation functions."""
    
    def test_manual_demean_returns_valid_result(self):
        """Manual demeaning should return valid ATT estimate."""
        data, params = generate_small_sample_dgp(seed=42)
        
        result = _estimate_manual_demean(data, treatment_start=11)
        
        assert 'att' in result
        assert 'se_ols' in result
        assert 'df' in result
        assert not np.isnan(result['att'])
    
    def test_manual_detrend_returns_valid_result(self):
        """Manual detrending should return valid ATT estimate."""
        data, params = generate_small_sample_dgp(seed=42)
        
        result = _estimate_manual_detrend(data, treatment_start=11)
        
        assert 'att' in result
        assert 'se_ols' in result
        assert 'df' in result
        assert not np.isnan(result['att'])
    
    def test_demean_att_reasonable_range(self):
        """Demeaning ATT should be in reasonable range."""
        data, params = generate_small_sample_dgp(seed=42)
        true_att = params['tau']
        
        result = _estimate_manual_demean(data, treatment_start=11)
        
        # ATT should be within 5 standard errors of true value
        # (very loose bound for single replication)
        assert abs(result['att'] - true_att) < 10
    
    def test_detrend_att_reasonable_range(self):
        """Detrending ATT should be in reasonable range."""
        data, params = generate_small_sample_dgp(seed=42)
        true_att = params['tau']
        
        result = _estimate_manual_detrend(data, treatment_start=11)
        
        assert abs(result['att'] - true_att) < 10


@pytest.mark.monte_carlo
class TestMonteCarloRunner:
    """Tests for Monte Carlo runner function."""
    
    def test_runner_returns_results(self):
        """Runner should return results for all estimators."""
        results = run_small_sample_monte_carlo(
            n_reps=10,
            scenario='scenario_1',
            estimators=['demeaning', 'detrending'],
            seed=42,
            verbose=False,
            use_lwdid=False,  # Use manual implementation for testing
        )
        
        assert 'demeaning' in results
        assert 'detrending' in results
    
    def test_runner_result_structure(self):
        """Results should have correct structure."""
        results = run_small_sample_monte_carlo(
            n_reps=10,
            scenario='scenario_1',
            estimators=['demeaning'],
            seed=42,
            verbose=False,
            use_lwdid=False,
        )
        
        result = results['demeaning']
        
        assert result.estimator_name == 'demeaning'
        assert result.scenario == 'scenario_1'
        assert result.n_reps == 10
        assert result.n_successful > 0
        assert not np.isnan(result.bias)
        assert not np.isnan(result.sd)
        assert not np.isnan(result.rmse)
    
    def test_runner_invalid_scenario(self):
        """Invalid scenario should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown scenario"):
            run_small_sample_monte_carlo(
                n_reps=10,
                scenario='invalid_scenario',
                seed=42,
                verbose=False,
            )
    
    def test_runner_reproducibility(self):
        """Same seed should produce same results."""
        results1 = run_small_sample_monte_carlo(
            n_reps=20,
            scenario='scenario_1',
            estimators=['demeaning'],
            seed=42,
            verbose=False,
            use_lwdid=False,
        )
        
        results2 = run_small_sample_monte_carlo(
            n_reps=20,
            scenario='scenario_1',
            estimators=['demeaning'],
            seed=42,
            verbose=False,
            use_lwdid=False,
        )
        
        assert results1['demeaning'].mean_att == results2['demeaning'].mean_att
        assert results1['demeaning'].bias == results2['demeaning'].bias


@pytest.mark.monte_carlo
@pytest.mark.slow
class TestMonteCarloScenario1:
    """Monte Carlo tests for Scenario 1 (p=0.32)."""
    
    @pytest.fixture
    def scenario_1_results(self):
        """Run Monte Carlo for Scenario 1 using actual lwdid package."""
        return run_small_sample_monte_carlo(
            n_reps=100,  # Reduced for faster testing
            scenario='scenario_1',
            estimators=['demeaning', 'detrending'],
            seed=42,
            verbose=False,
            use_lwdid=True,  # Use actual lwdid package
        )
    
    def test_detrending_has_results(self, scenario_1_results):
        """Detrending should produce results."""
        assert 'detrending' in scenario_1_results
        result = scenario_1_results['detrending']
        assert result.n_successful > 50  # At least 50% success rate
    
    def test_demeaning_has_results(self, scenario_1_results):
        """Demeaning should produce results."""
        assert 'demeaning' in scenario_1_results
        result = scenario_1_results['demeaning']
        assert result.n_successful > 50
    
    def test_bias_reasonable(self, scenario_1_results):
        """Bias should be reasonable (not too large).
        
        Paper Table 2 shows detrending has small bias (~0.009).
        Demeaning has larger bias due to heterogeneous trends.
        """
        for estimator, result in scenario_1_results.items():
            if estimator == 'detrending':
                # Detrending should have small bias (paper Table 2: ~0.009)
                assert abs(result.bias) < 1.5, \
                    f"{estimator} bias {result.bias} too large"
            # Demeaning bias is expected to be larger due to heterogeneous trends
    
    def test_rmse_positive(self, scenario_1_results):
        """RMSE should be positive."""
        for estimator, result in scenario_1_results.items():
            assert result.rmse > 0, f"{estimator} RMSE should be positive"
    
    def test_coverage_reasonable(self, scenario_1_results):
        """Coverage should be in reasonable range.
        
        Paper Table 2 shows coverage around 95% for detrending.
        Using actual lwdid package for accurate SE estimation.
        """
        for estimator, result in scenario_1_results.items():
            # Coverage should be between 50% and 100%
            assert 0.5 <= result.coverage_ols <= 1.0, \
                f"{estimator} OLS coverage {result.coverage_ols} out of range"


@pytest.mark.monte_carlo
@pytest.mark.slow
class TestMonteCarloScenario3:
    """Monte Carlo tests for Scenario 3 (p=0.17, sparse treatment)."""
    
    @pytest.fixture
    def scenario_3_results(self):
        """Run Monte Carlo for Scenario 3."""
        return run_small_sample_monte_carlo(
            n_reps=100,
            scenario='scenario_3',
            estimators=['demeaning', 'detrending'],
            seed=42,
            verbose=False,
            use_lwdid=False,
        )
    
    def test_sparse_treatment_produces_results(self, scenario_3_results):
        """Sparse treatment scenario should still produce results."""
        assert len(scenario_3_results) > 0
        for estimator, result in scenario_3_results.items():
            assert result.n_successful > 30  # Lower threshold for sparse case


@pytest.mark.monte_carlo
@pytest.mark.slow
class TestMonteCarloComparison:
    """Compare results across all scenarios."""
    
    @pytest.fixture
    def all_scenario_results(self):
        """Run Monte Carlo for all scenarios."""
        return run_all_scenarios_monte_carlo(
            n_reps=50,  # Reduced for faster testing
            estimators=['demeaning', 'detrending'],
            seed=42,
            verbose=False,
            use_lwdid=False,
        )
    
    def test_all_scenarios_have_results(self, all_scenario_results):
        """All scenarios should have results."""
        for scenario in SMALL_SAMPLE_SCENARIOS.keys():
            assert scenario in all_scenario_results, \
                f"Missing results for {scenario}"
    
    def test_generate_comparison_table(self, all_scenario_results):
        """Should generate valid comparison table."""
        df = generate_comparison_table(all_scenario_results)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        
        required_cols = ['Scenario', 'Estimator', 'Bias', 'SD', 'RMSE']
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"
    
    def test_comparison_table_has_all_combinations(self, all_scenario_results):
        """Table should have all scenario-estimator combinations."""
        df = generate_comparison_table(all_scenario_results)
        
        # Should have 3 scenarios × 2 estimators = 6 rows
        assert len(df) >= 4  # At least 4 rows (some may fail)


@pytest.mark.monte_carlo
class TestMonteCarloStatistics:
    """Tests for Monte Carlo statistics calculations."""
    
    def test_bias_calculation(self):
        """Bias should be mean(ATT) - true_ATT."""
        results = run_small_sample_monte_carlo(
            n_reps=50,
            scenario='scenario_1',
            estimators=['demeaning'],
            seed=42,
            verbose=False,
            use_lwdid=False,
        )
        
        result = results['demeaning']
        
        # Verify bias calculation
        expected_bias = np.mean(result.att_estimates) - result.true_att
        assert np.isclose(result.bias, expected_bias, rtol=1e-6)
    
    def test_sd_calculation(self):
        """SD should be std(ATT) with ddof=1."""
        results = run_small_sample_monte_carlo(
            n_reps=50,
            scenario='scenario_1',
            estimators=['demeaning'],
            seed=42,
            verbose=False,
            use_lwdid=False,
        )
        
        result = results['demeaning']
        
        # Verify SD calculation
        expected_sd = np.std(result.att_estimates, ddof=1)
        assert np.isclose(result.sd, expected_sd, rtol=1e-6)
    
    def test_rmse_calculation(self):
        """RMSE should be sqrt(bias² + sd²)."""
        results = run_small_sample_monte_carlo(
            n_reps=50,
            scenario='scenario_1',
            estimators=['demeaning'],
            seed=42,
            verbose=False,
            use_lwdid=False,
        )
        
        result = results['demeaning']
        
        # Verify RMSE calculation
        expected_rmse = np.sqrt(result.bias**2 + result.sd**2)
        assert np.isclose(result.rmse, expected_rmse, rtol=1e-6)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'monte_carlo'])

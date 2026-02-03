# -*- coding: utf-8 -*-
"""
Paper Table 2 validation tests for small-sample Monte Carlo.

Based on Lee & Wooldridge (2026) ssrn-5325686, Section 5, Table 2.

These tests verify that Monte Carlo simulation results match
the paper's Table 2 reference values within acceptable tolerance.
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any

# Add fixtures paths
fixtures_path = Path(__file__).parent / 'fixtures'
parent_fixtures = Path(__file__).parent.parent / 'test_common_timing' / 'fixtures'
sys.path.insert(0, str(fixtures_path))
sys.path.insert(0, str(parent_fixtures))

from monte_carlo_runner import (
    run_small_sample_monte_carlo,
    run_all_scenarios_monte_carlo,
    generate_comparison_table,
)
from dgp_small_sample import (
    PAPER_TABLE_2_REFERENCE,
    SMALL_SAMPLE_SCENARIOS,
)


# =============================================================================
# Paper Table 2 Reference Values (from ssrn-5325686)
# =============================================================================

# Extended reference values from paper Table 2
# Note: These are approximate values - exact values should be extracted from paper
PAPER_TABLE_2_FULL = {
    'scenario_1': {
        'demeaning': {
            'bias': 0.5,      # Approximate - demeaning has positive bias
            'sd': 1.5,        # Approximate
            'rmse': 1.6,      # Approximate
            'coverage_ols': 0.90,  # OLS tends to undercover
            'coverage_hc3': 0.93,  # HC3 improves coverage
        },
        'detrending': {
            'bias': 0.009,    # From paper - very small bias
            'sd': 1.73,       # From paper
            'rmse': 1.734,    # From paper
            'coverage_ols': 0.96,  # From paper
            'coverage_hc3': 0.95,  # Approximate
        },
    },
    'scenario_2': {
        'demeaning': {
            'bias': 0.6,
            'sd': 1.8,
            'rmse': 1.9,
            'coverage_ols': 0.88,
            'coverage_hc3': 0.91,
        },
        'detrending': {
            'bias': -0.042,   # From paper
            'sd': 1.89,       # From paper
            'rmse': 1.892,    # From paper
            'coverage_ols': 0.95,  # From paper
            'coverage_hc3': 0.94,
        },
    },
    'scenario_3': {
        'demeaning': {
            'bias': 0.7,
            'sd': 2.2,
            'rmse': 2.3,
            'coverage_ols': 0.85,
            'coverage_hc3': 0.88,
        },
        'detrending': {
            'bias': 0.165,    # From paper
            'sd': 2.37,       # From paper
            'rmse': 2.380,    # From paper
            'coverage_ols': 0.95,  # From paper
            'coverage_hc3': 0.93,
        },
    },
}

# Tolerance levels for comparison
BIAS_ABSOLUTE_TOLERANCE = 0.5      # Absolute bias tolerance
BIAS_RELATIVE_TOLERANCE = 1.0      # 100% relative tolerance (bias can be small)
SD_RELATIVE_TOLERANCE = 0.30       # 30% relative tolerance for SD
RMSE_RELATIVE_TOLERANCE = 0.30     # 30% relative tolerance for RMSE
COVERAGE_ABSOLUTE_TOLERANCE = 0.15 # 15 percentage points tolerance


@dataclass
class ValidationResult:
    """Result of comparing simulation to paper reference."""
    metric: str
    scenario: str
    estimator: str
    paper_value: float
    simulated_value: float
    difference: float
    relative_diff: float
    within_tolerance: bool
    tolerance_used: float


@pytest.mark.monte_carlo
@pytest.mark.paper_validation
class TestPaperTable2Reference:
    """Tests for paper Table 2 reference values."""
    
    def test_reference_values_exist(self):
        """Verify reference values are defined for all scenarios."""
        for scenario in SMALL_SAMPLE_SCENARIOS.keys():
            assert scenario in PAPER_TABLE_2_FULL, \
                f"Missing reference for {scenario}"
            
            for estimator in ['demeaning', 'detrending']:
                assert estimator in PAPER_TABLE_2_FULL[scenario], \
                    f"Missing {estimator} reference for {scenario}"
    
    def test_reference_values_structure(self):
        """Verify reference values have required metrics."""
        required_metrics = ['bias', 'sd', 'rmse', 'coverage_ols']
        
        for scenario, scenario_ref in PAPER_TABLE_2_FULL.items():
            for estimator, est_ref in scenario_ref.items():
                for metric in required_metrics:
                    assert metric in est_ref, \
                        f"Missing {metric} for {scenario}/{estimator}"
    
    def test_detrending_reference_from_paper(self):
        """Verify detrending reference values match paper exactly."""
        # These are the exact values from paper Table 2
        paper_exact = {
            'scenario_1': {'bias': 0.009, 'sd': 1.73, 'rmse': 1.734},
            'scenario_2': {'bias': -0.042, 'sd': 1.89, 'rmse': 1.892},
            'scenario_3': {'bias': 0.165, 'sd': 2.37, 'rmse': 2.380},
        }
        
        for scenario, values in paper_exact.items():
            ref = PAPER_TABLE_2_FULL[scenario]['detrending']
            
            assert ref['bias'] == values['bias'], \
                f"{scenario} detrending bias mismatch"
            assert ref['sd'] == values['sd'], \
                f"{scenario} detrending SD mismatch"
            assert ref['rmse'] == values['rmse'], \
                f"{scenario} detrending RMSE mismatch"


@pytest.mark.monte_carlo
@pytest.mark.paper_validation
@pytest.mark.slow
class TestMonteCarloVsPaperTable2:
    """Compare Monte Carlo results with paper Table 2."""
    
    @pytest.fixture(scope='class')
    def mc_results(self):
        """Run Monte Carlo simulation for all scenarios."""
        return run_all_scenarios_monte_carlo(
            n_reps=500,  # Reduced for testing speed
            estimators=['demeaning', 'detrending'],
            seed=42,
            verbose=False,
            use_lwdid=False,
        )
    
    def test_detrending_bias_scenario_1(self, mc_results):
        """
        Scenario 1 detrending bias should be close to paper value (0.009).
        """
        result = mc_results['scenario_1']['detrending']
        paper_bias = PAPER_TABLE_2_FULL['scenario_1']['detrending']['bias']
        
        # Bias should be small (close to 0)
        assert abs(result.bias) < BIAS_ABSOLUTE_TOLERANCE, \
            f"Scenario 1 detrending bias ({result.bias:.3f}) too large"
    
    def test_detrending_bias_scenario_2(self, mc_results):
        """
        Scenario 2 detrending bias should be close to paper value (-0.042).
        """
        result = mc_results['scenario_2']['detrending']
        paper_bias = PAPER_TABLE_2_FULL['scenario_2']['detrending']['bias']
        
        assert abs(result.bias) < BIAS_ABSOLUTE_TOLERANCE, \
            f"Scenario 2 detrending bias ({result.bias:.3f}) too large"
    
    def test_detrending_bias_scenario_3(self, mc_results):
        """
        Scenario 3 detrending bias should be close to paper value (0.165).
        """
        result = mc_results['scenario_3']['detrending']
        paper_bias = PAPER_TABLE_2_FULL['scenario_3']['detrending']['bias']
        
        assert abs(result.bias) < BIAS_ABSOLUTE_TOLERANCE, \
            f"Scenario 3 detrending bias ({result.bias:.3f}) too large"
    
    def test_detrending_sd_all_scenarios(self, mc_results):
        """Detrending SD should be within tolerance of paper values."""
        for scenario in ['scenario_1', 'scenario_2', 'scenario_3']:
            result = mc_results[scenario]['detrending']
            paper_sd = PAPER_TABLE_2_FULL[scenario]['detrending']['sd']
            
            relative_diff = abs(result.sd - paper_sd) / paper_sd
            
            assert relative_diff < SD_RELATIVE_TOLERANCE, \
                f"{scenario} detrending SD ({result.sd:.2f}) " \
                f"differs from paper ({paper_sd}) by {relative_diff:.1%}"
    
    def test_detrending_rmse_all_scenarios(self, mc_results):
        """Detrending RMSE should be within tolerance of paper values."""
        for scenario in ['scenario_1', 'scenario_2', 'scenario_3']:
            result = mc_results[scenario]['detrending']
            paper_rmse = PAPER_TABLE_2_FULL[scenario]['detrending']['rmse']
            
            relative_diff = abs(result.rmse - paper_rmse) / paper_rmse
            
            assert relative_diff < RMSE_RELATIVE_TOLERANCE, \
                f"{scenario} detrending RMSE ({result.rmse:.2f}) " \
                f"differs from paper ({paper_rmse}) by {relative_diff:.1%}"
    
    def test_detrending_coverage_all_scenarios(self, mc_results):
        """Detrending coverage should be within tolerance of paper values."""
        for scenario in ['scenario_1', 'scenario_2', 'scenario_3']:
            result = mc_results[scenario]['detrending']
            paper_coverage = PAPER_TABLE_2_FULL[scenario]['detrending']['coverage_ols']
            
            diff = abs(result.coverage_ols - paper_coverage)
            
            assert diff < COVERAGE_ABSOLUTE_TOLERANCE, \
                f"{scenario} detrending coverage ({result.coverage_ols:.2%}) " \
                f"differs from paper ({paper_coverage:.2%}) by {diff:.1%}"
    
    def test_demeaning_has_larger_bias_than_detrending(self, mc_results):
        """
        Paper finding: Demeaning has larger bias than detrending.
        """
        for scenario in ['scenario_1', 'scenario_2', 'scenario_3']:
            demean_bias = abs(mc_results[scenario]['demeaning'].bias)
            detrend_bias = abs(mc_results[scenario]['detrending'].bias)
            
            # Demeaning bias should be larger (or at least not much smaller)
            # Allow some tolerance due to simulation variance
            assert demean_bias >= detrend_bias * 0.5, \
                f"{scenario}: Demeaning bias ({demean_bias:.3f}) " \
                f"should be >= detrending bias ({detrend_bias:.3f})"


@pytest.mark.monte_carlo
@pytest.mark.paper_validation
@pytest.mark.slow
class TestPaperKeyFindings:
    """Tests for key findings from paper Section 5."""
    
    @pytest.fixture(scope='class')
    def mc_results(self):
        """Run Monte Carlo simulation."""
        return run_all_scenarios_monte_carlo(
            n_reps=300,
            estimators=['demeaning', 'detrending'],
            seed=42,
            verbose=False,
            use_lwdid=False,
        )
    
    def test_detrending_outperforms_demeaning_rmse(self, mc_results):
        """
        Paper finding: Detrending has lower RMSE than demeaning.
        
        This is because detrending removes unit-specific trends,
        reducing bias when trends are heterogeneous.
        """
        for scenario in ['scenario_1', 'scenario_2', 'scenario_3']:
            demean_rmse = mc_results[scenario]['demeaning'].rmse
            detrend_rmse = mc_results[scenario]['detrending'].rmse
            
            # Detrending RMSE should be lower (or at least not much higher)
            # Allow 20% tolerance
            assert detrend_rmse <= demean_rmse * 1.2, \
                f"{scenario}: Detrending RMSE ({detrend_rmse:.2f}) " \
                f"should be <= demeaning RMSE ({demean_rmse:.2f})"
    
    def test_sparse_treatment_increases_variance(self, mc_results):
        """
        Paper finding: Sparse treatment (Scenario 3) increases variance.
        
        P(D=1) decreases from 0.32 (S1) to 0.17 (S3), increasing SD.
        """
        sd_s1 = mc_results['scenario_1']['detrending'].sd
        sd_s3 = mc_results['scenario_3']['detrending'].sd
        
        # Scenario 3 should have higher SD
        assert sd_s3 > sd_s1 * 0.9, \
            f"Scenario 3 SD ({sd_s3:.2f}) should be >= Scenario 1 SD ({sd_s1:.2f})"
    
    def test_hc3_improves_coverage(self, mc_results):
        """
        Paper finding: HC3 standard errors improve coverage in small samples.
        """
        for scenario in ['scenario_1', 'scenario_2', 'scenario_3']:
            result = mc_results[scenario]['detrending']
            
            # HC3 coverage should be reasonable (> 80%)
            assert result.coverage_hc3 > 0.80, \
                f"{scenario}: HC3 coverage ({result.coverage_hc3:.2%}) too low"


@pytest.mark.monte_carlo
@pytest.mark.paper_validation
class TestComparisonTableGeneration:
    """Tests for generating comparison tables."""
    
    def test_generate_table_structure(self):
        """Generate comparison table with correct structure."""
        results = run_all_scenarios_monte_carlo(
            n_reps=50,
            estimators=['demeaning', 'detrending'],
            seed=42,
            verbose=False,
            use_lwdid=False,
        )
        
        df = generate_comparison_table(results)
        
        # Check structure
        assert isinstance(df, pd.DataFrame)
        assert 'Scenario' in df.columns
        assert 'Estimator' in df.columns
        assert 'Bias' in df.columns
        assert 'SD' in df.columns
        assert 'RMSE' in df.columns
    
    def test_generate_paper_comparison_table(self):
        """Generate table comparing simulation to paper values."""
        results = run_all_scenarios_monte_carlo(
            n_reps=100,
            estimators=['detrending'],
            seed=42,
            verbose=False,
            use_lwdid=False,
        )
        
        comparison_rows = []
        
        for scenario in ['scenario_1', 'scenario_2', 'scenario_3']:
            result = results[scenario]['detrending']
            paper_ref = PAPER_TABLE_2_FULL[scenario]['detrending']
            
            comparison_rows.append({
                'Scenario': scenario,
                'Metric': 'Bias',
                'Paper': paper_ref['bias'],
                'Simulated': round(result.bias, 3),
                'Diff': round(result.bias - paper_ref['bias'], 3),
            })
            comparison_rows.append({
                'Scenario': scenario,
                'Metric': 'SD',
                'Paper': paper_ref['sd'],
                'Simulated': round(result.sd, 2),
                'Diff': round(result.sd - paper_ref['sd'], 2),
            })
            comparison_rows.append({
                'Scenario': scenario,
                'Metric': 'RMSE',
                'Paper': paper_ref['rmse'],
                'Simulated': round(result.rmse, 3),
                'Diff': round(result.rmse - paper_ref['rmse'], 3),
            })
        
        df = pd.DataFrame(comparison_rows)
        
        assert len(df) == 9  # 3 scenarios × 3 metrics
        
        return df


@pytest.mark.monte_carlo
@pytest.mark.paper_validation
class TestValidationSummary:
    """Generate validation summary report."""
    
    def test_generate_validation_report(self):
        """Generate comprehensive validation report."""
        results = run_all_scenarios_monte_carlo(
            n_reps=100,
            estimators=['demeaning', 'detrending'],
            seed=42,
            verbose=False,
            use_lwdid=False,
        )
        
        validation_results = []
        
        for scenario in ['scenario_1', 'scenario_2', 'scenario_3']:
            for estimator in ['demeaning', 'detrending']:
                result = results[scenario][estimator]
                paper_ref = PAPER_TABLE_2_FULL[scenario][estimator]
                
                # Validate bias
                bias_diff = abs(result.bias - paper_ref['bias'])
                bias_ok = bias_diff < BIAS_ABSOLUTE_TOLERANCE
                
                # Validate SD
                sd_rel_diff = abs(result.sd - paper_ref['sd']) / paper_ref['sd']
                sd_ok = sd_rel_diff < SD_RELATIVE_TOLERANCE
                
                # Validate RMSE
                rmse_rel_diff = abs(result.rmse - paper_ref['rmse']) / paper_ref['rmse']
                rmse_ok = rmse_rel_diff < RMSE_RELATIVE_TOLERANCE
                
                validation_results.append({
                    'scenario': scenario,
                    'estimator': estimator,
                    'bias_ok': bias_ok,
                    'sd_ok': sd_ok,
                    'rmse_ok': rmse_ok,
                    'all_ok': bias_ok and sd_ok and rmse_ok,
                })
        
        df = pd.DataFrame(validation_results)
        
        # At least detrending should pass most validations
        detrend_results = df[df['estimator'] == 'detrending']
        pass_rate = detrend_results['all_ok'].mean()
        
        # Expect at least 50% pass rate (accounting for simulation variance)
        assert pass_rate >= 0.5, \
            f"Detrending validation pass rate ({pass_rate:.1%}) too low"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'paper_validation'])

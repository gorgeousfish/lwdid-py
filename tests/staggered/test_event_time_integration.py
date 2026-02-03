"""
Integration tests for event-time aggregation (WATT).

Tests cover:
- Task 7.2: Pipeline integration tests
- Task 7.3: Empirical data tests
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from lwdid.staggered.aggregation import (
    aggregate_to_event_time,
    event_time_effects_to_dataframe,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_att_by_cohort_time():
    """Mock att_by_cohort_time DataFrame similar to StaggeredResult output."""
    return pd.DataFrame({
        'cohort': [2004, 2004, 2004, 2005, 2005, 2006],
        'period': [2004, 2005, 2006, 2005, 2006, 2006],
        'att': [0.05, 0.08, 0.10, 0.03, 0.06, 0.04],
        'se': [0.02, 0.025, 0.03, 0.015, 0.02, 0.018],
        'df_inference': [45, 45, 45, 38, 38, 30],
        't_stat': [2.5, 3.2, 3.33, 2.0, 3.0, 2.22],
        'pvalue': [0.016, 0.002, 0.002, 0.053, 0.005, 0.034],
    })


@pytest.fixture
def mock_cohort_sizes():
    """Mock cohort sizes."""
    return {2004: 50, 2005: 30, 2006: 20}


# =============================================================================
# Task 7.2: Pipeline Integration Tests
# =============================================================================

class TestPipelineIntegration:
    """Tests for integration with estimation pipeline."""

    def test_aggregate_from_att_by_cohort_time(
        self, mock_att_by_cohort_time, mock_cohort_sizes
    ):
        """Test aggregation from att_by_cohort_time DataFrame."""
        results = aggregate_to_event_time(
            mock_att_by_cohort_time, mock_cohort_sizes
        )
        
        # Should produce results for each unique event_time
        event_times = (
            mock_att_by_cohort_time['period'] - mock_att_by_cohort_time['cohort']
        ).unique()
        
        assert len(results) == len(event_times)
        
        # All results should have valid values
        for effect in results:
            assert not np.isnan(effect.att)
            assert not np.isnan(effect.se)
            assert effect.n_cohorts > 0

    def test_return_data_includes_df_inference(
        self, mock_att_by_cohort_time, mock_cohort_sizes
    ):
        """Test that returned DataFrame includes df_inference column."""
        results = aggregate_to_event_time(
            mock_att_by_cohort_time, mock_cohort_sizes
        )
        df = event_time_effects_to_dataframe(results)
        
        assert 'df_inference' in df.columns
        assert all(df['df_inference'] > 0)

    def test_backward_compatibility_columns(
        self, mock_att_by_cohort_time, mock_cohort_sizes
    ):
        """Test that output has expected columns for backward compatibility."""
        results = aggregate_to_event_time(
            mock_att_by_cohort_time, mock_cohort_sizes
        )
        df = event_time_effects_to_dataframe(results)
        
        expected_columns = [
            'event_time', 'att', 'se', 'ci_lower', 'ci_upper',
            't_stat', 'pvalue', 'df_inference', 'n_cohorts', 'weight_sum'
        ]
        
        for col in expected_columns:
            assert col in df.columns, f"Missing column: {col}"

    def test_event_time_ordering(
        self, mock_att_by_cohort_time, mock_cohort_sizes
    ):
        """Test that results are sorted by event_time."""
        results = aggregate_to_event_time(
            mock_att_by_cohort_time, mock_cohort_sizes
        )
        
        event_times = [e.event_time for e in results]
        assert event_times == sorted(event_times)

    def test_cohort_contributions_sum(
        self, mock_att_by_cohort_time, mock_cohort_sizes
    ):
        """Test that cohort contributions sum to WATT."""
        results = aggregate_to_event_time(
            mock_att_by_cohort_time, mock_cohort_sizes
        )
        
        for effect in results:
            if effect.n_cohorts > 0:
                contribution_sum = sum(effect.cohort_contributions.values())
                assert abs(contribution_sum - effect.att) < 1e-10

    def test_different_alpha_values(
        self, mock_att_by_cohort_time, mock_cohort_sizes
    ):
        """Test aggregation with different alpha values."""
        alpha_values = [0.01, 0.05, 0.10]
        
        results_by_alpha = {}
        for alpha in alpha_values:
            results = aggregate_to_event_time(
                mock_att_by_cohort_time, mock_cohort_sizes, alpha=alpha
            )
            results_by_alpha[alpha] = results
        
        # CI should be wider for smaller alpha (more confidence)
        for effect_idx in range(len(results_by_alpha[0.05])):
            ci_width_01 = (
                results_by_alpha[0.01][effect_idx].ci_upper -
                results_by_alpha[0.01][effect_idx].ci_lower
            )
            ci_width_05 = (
                results_by_alpha[0.05][effect_idx].ci_upper -
                results_by_alpha[0.05][effect_idx].ci_lower
            )
            ci_width_10 = (
                results_by_alpha[0.10][effect_idx].ci_upper -
                results_by_alpha[0.10][effect_idx].ci_lower
            )
            
            # 99% CI > 95% CI > 90% CI
            assert ci_width_01 > ci_width_05 > ci_width_10

    def test_df_strategy_comparison(
        self, mock_att_by_cohort_time, mock_cohort_sizes
    ):
        """Test different df_strategy options."""
        strategies = ['conservative', 'weighted', 'fallback']
        
        results_by_strategy = {}
        for strategy in strategies:
            results = aggregate_to_event_time(
                mock_att_by_cohort_time, mock_cohort_sizes,
                df_strategy=strategy
            )
            results_by_strategy[strategy] = results
        
        # Conservative should have smallest df (widest CI)
        # for event times with multiple cohorts
        for effect_idx in range(len(results_by_strategy['conservative'])):
            effect_cons = results_by_strategy['conservative'][effect_idx]
            effect_weighted = results_by_strategy['weighted'][effect_idx]
            
            if effect_cons.n_cohorts > 1:
                # Conservative df should be <= weighted df
                assert effect_cons.df_inference <= effect_weighted.df_inference


# =============================================================================
# Task 7.3: Empirical Data Tests
# =============================================================================

class TestEmpiricalData:
    """Tests with empirical datasets."""

    @pytest.fixture
    def cattaneo_data_path(self):
        """Path to cattaneo2.dta dataset."""
        # Try multiple possible locations
        possible_paths = [
            Path('cattaneo2.dta'),
            Path('../../cattaneo2.dta'),
            Path('../../../cattaneo2.dta'),
        ]
        for path in possible_paths:
            if path.exists():
                return path
        pytest.skip("cattaneo2.dta not found")

    @pytest.fixture
    def nlswork_data_path(self):
        """Path to nlswork_did.csv dataset."""
        possible_paths = [
            Path('nlswork_did.csv'),
            Path('../../nlswork_did.csv'),
            Path('../../../nlswork_did.csv'),
        ]
        for path in possible_paths:
            if path.exists():
                return path
        pytest.skip("nlswork_did.csv not found")

    def test_simulated_staggered_data(self):
        """Test with simulated staggered adoption data."""
        np.random.seed(42)
        
        # Simulate 3 cohorts with staggered adoption
        n_cohorts = 3
        cohorts = [2004, 2005, 2006]
        n_periods = 5
        
        # Generate cohort-time effects
        data = []
        for g in cohorts:
            for t in range(g, 2004 + n_periods):
                event_time = t - g
                # True effect increases with event time
                true_att = 0.05 + 0.02 * event_time
                # Add noise
                att = true_att + np.random.normal(0, 0.01)
                se = 0.02 + np.random.uniform(0, 0.01)
                df = 30 + np.random.randint(0, 20)
                
                data.append({
                    'cohort': g,
                    'period': t,
                    'att': att,
                    'se': se,
                    'df_inference': df,
                })
        
        df = pd.DataFrame(data)
        cohort_sizes = {2004: 50, 2005: 40, 2006: 30}
        
        results = aggregate_to_event_time(df, cohort_sizes)
        
        # Should have results for each event time
        assert len(results) > 0
        
        # WATT should generally increase with event time (given DGP)
        atts = [e.att for e in results]
        event_times = [e.event_time for e in results]
        
        # Check correlation is positive (effects increase over time)
        if len(atts) > 2:
            correlation = np.corrcoef(event_times, atts)[0, 1]
            assert correlation > 0, "Expected positive correlation with event time"

    def test_balanced_panel_simulation(self):
        """Test with balanced panel simulation."""
        np.random.seed(123)
        
        # 2 cohorts, each observed for 3 post-treatment periods
        df = pd.DataFrame({
            'cohort': [2004, 2004, 2004, 2005, 2005, 2005],
            'period': [2004, 2005, 2006, 2005, 2006, 2007],
            'att': [0.05, 0.07, 0.09, 0.04, 0.06, 0.08],
            'se': [0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
            'df_inference': [50, 50, 50, 45, 45, 45],
        })
        cohort_sizes = {2004: 100, 2005: 100}
        
        results = aggregate_to_event_time(df, cohort_sizes)
        
        # Event times 0, 1, 2 should have 2 cohorts each
        # Event time 3 should have 1 cohort (only 2005)
        results_dict = {e.event_time: e for e in results}
        
        assert results_dict[0].n_cohorts == 2
        assert results_dict[1].n_cohorts == 2
        # Event time 2 has both cohorts (2004: period 2006, 2005: period 2007)
        # But 2005's period 2007 is event_time = 2007-2005 = 2
        # And 2004's period 2006 is event_time = 2006-2004 = 2
        assert results_dict[2].n_cohorts == 2

    def test_unbalanced_panel_simulation(self):
        """Test with unbalanced panel (different observation windows)."""
        # Cohort 2004 observed for 3 periods, 2005 for 2 periods
        df = pd.DataFrame({
            'cohort': [2004, 2004, 2004, 2005, 2005],
            'period': [2004, 2005, 2006, 2005, 2006],
            'att': [0.05, 0.07, 0.09, 0.04, 0.06],
            'se': [0.02, 0.02, 0.02, 0.02, 0.02],
            'df_inference': [50, 50, 50, 45, 45],
        })
        cohort_sizes = {2004: 100, 2005: 80}
        
        results = aggregate_to_event_time(df, cohort_sizes)
        results_dict = {e.event_time: e for e in results}
        
        # Event time 0: both cohorts
        assert results_dict[0].n_cohorts == 2
        # Event time 1: both cohorts
        assert results_dict[1].n_cohorts == 2
        # Event time 2: only cohort 2004
        assert results_dict[2].n_cohorts == 1


# =============================================================================
# Regression Tests
# =============================================================================

class TestRegressionTests:
    """Regression tests to ensure no breaking changes."""

    def test_output_structure_unchanged(
        self, mock_att_by_cohort_time, mock_cohort_sizes
    ):
        """Test that output structure matches expected format."""
        results = aggregate_to_event_time(
            mock_att_by_cohort_time, mock_cohort_sizes
        )
        
        # Check that all expected attributes exist
        for effect in results:
            assert hasattr(effect, 'event_time')
            assert hasattr(effect, 'att')
            assert hasattr(effect, 'se')
            assert hasattr(effect, 'ci_lower')
            assert hasattr(effect, 'ci_upper')
            assert hasattr(effect, 't_stat')
            assert hasattr(effect, 'pvalue')
            assert hasattr(effect, 'df_inference')
            assert hasattr(effect, 'n_cohorts')
            assert hasattr(effect, 'cohort_contributions')
            assert hasattr(effect, 'weight_sum')
            assert hasattr(effect, 'alpha')

    def test_deterministic_output(
        self, mock_att_by_cohort_time, mock_cohort_sizes
    ):
        """Test that output is deterministic (same input -> same output)."""
        results1 = aggregate_to_event_time(
            mock_att_by_cohort_time, mock_cohort_sizes
        )
        results2 = aggregate_to_event_time(
            mock_att_by_cohort_time, mock_cohort_sizes
        )
        
        assert len(results1) == len(results2)
        
        for e1, e2 in zip(results1, results2):
            assert e1.event_time == e2.event_time
            assert e1.att == e2.att
            assert e1.se == e2.se
            assert e1.ci_lower == e2.ci_lower
            assert e1.ci_upper == e2.ci_upper

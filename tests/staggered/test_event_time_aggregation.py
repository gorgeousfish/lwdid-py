"""
Unit tests for event-time aggregation (Weighted Average Treatment on the Treated).

Validates weight normalization, the WATT formula, standard error computation,
t-distribution inference, and edge cases for the event-time aggregation
procedure.

Validates Section 7 (event-time aggregation) of the Lee-Wooldridge
Difference-in-Differences framework.

References
----------
Lee, S. & Wooldridge, J. M. (2025). A Simple Transformation Approach to
    Difference-in-Differences Estimation for Panel Data. SSRN 4516518.
Lee, S. & Wooldridge, J. M. (2026). Simple Approaches to Inference with
    DiD Estimators with Small Cross-Sectional Sample Sizes. SSRN 5325686.
"""

import math
import numpy as np
import pandas as pd
import pytest
from scipy.stats import t as t_dist

from lwdid.staggered.aggregation import (
    EventTimeEffect,
    aggregate_to_event_time,
    event_time_effects_to_dataframe,
    _compute_event_time_weights,
    _select_degrees_of_freedom,
    _validate_weight_sum,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_cohort_time_df():
    """Simple cohort-time effects DataFrame for testing."""
    return pd.DataFrame({
        'cohort': [2004, 2004, 2005, 2005, 2006],
        'period': [2004, 2005, 2005, 2006, 2006],
        'att': [0.05, 0.08, 0.03, 0.06, 0.04],
        'se': [0.02, 0.03, 0.02, 0.02, 0.015],
        'df_inference': [45, 45, 38, 38, 30],
    })


@pytest.fixture
def simple_cohort_sizes():
    """Simple cohort sizes for testing."""
    return {2004: 50, 2005: 30, 2006: 20}


@pytest.fixture
def single_cohort_df():
    """Single cohort effects for degeneracy testing."""
    return pd.DataFrame({
        'cohort': [2004, 2004, 2004],
        'period': [2004, 2005, 2006],
        'att': [0.05, 0.08, 0.10],
        'se': [0.02, 0.03, 0.025],
        'df_inference': [45, 45, 45],
    })


@pytest.fixture
def single_cohort_sizes():
    """Single cohort size."""
    return {2004: 50}


# =============================================================================
# Task 5.2: Weight Normalization Tests
# =============================================================================

class TestWeightNormalization:
    """Tests for weight computation and normalization."""

    def test_weight_normalization_basic(self, simple_cohort_sizes):
        """Test that weights sum to 1.0 for 3 cohorts."""
        available_cohorts = [2004, 2005, 2006]
        weights = _compute_event_time_weights(simple_cohort_sizes, available_cohorts)
        
        # Verify sum = 1.0
        weight_sum = sum(weights.values())
        assert abs(weight_sum - 1.0) < 1e-10, f"Weight sum {weight_sum} != 1.0"
        
        # Verify individual weights
        total = 50 + 30 + 20  # = 100
        assert abs(weights[2004] - 0.50) < 1e-10
        assert abs(weights[2005] - 0.30) < 1e-10
        assert abs(weights[2006] - 0.20) < 1e-10

    def test_weight_normalization_single_cohort(self):
        """Test that single cohort has weight = 1.0."""
        cohort_sizes = {2004: 50}
        available_cohorts = [2004]
        weights = _compute_event_time_weights(cohort_sizes, available_cohorts)
        
        assert len(weights) == 1
        assert abs(weights[2004] - 1.0) < 1e-10

    def test_weight_normalization_unequal_sizes(self):
        """Test weights with very unequal cohort sizes."""
        cohort_sizes = {2004: 1000, 2005: 10, 2006: 1}
        available_cohorts = [2004, 2005, 2006]
        weights = _compute_event_time_weights(cohort_sizes, available_cohorts)
        
        total = 1000 + 10 + 1  # = 1011
        assert abs(weights[2004] - 1000/1011) < 1e-10
        assert abs(weights[2005] - 10/1011) < 1e-10
        assert abs(weights[2006] - 1/1011) < 1e-10
        
        # Sum should still be 1.0
        assert abs(sum(weights.values()) - 1.0) < 1e-10

    def test_weight_normalization_subset_cohorts(self, simple_cohort_sizes):
        """Test weights when only subset of cohorts available."""
        available_cohorts = [2004, 2005]  # Exclude 2006
        weights = _compute_event_time_weights(simple_cohort_sizes, available_cohorts)
        
        # Should only have 2 cohorts
        assert len(weights) == 2
        assert 2006 not in weights
        
        # Weights should sum to 1.0 for available cohorts
        total = 50 + 30  # = 80
        assert abs(weights[2004] - 50/80) < 1e-10
        assert abs(weights[2005] - 30/80) < 1e-10
        assert abs(sum(weights.values()) - 1.0) < 1e-10

    def test_weight_empty_cohorts_raises(self, simple_cohort_sizes):
        """Test that empty available_cohorts raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            _compute_event_time_weights(simple_cohort_sizes, [])

    def test_validate_weight_sum_valid(self):
        """Test weight sum validation for valid weights."""
        weights = {2004: 0.5, 2005: 0.3, 2006: 0.2}
        is_valid, weight_sum = _validate_weight_sum(weights)
        assert is_valid
        assert abs(weight_sum - 1.0) < 1e-10

    def test_validate_weight_sum_invalid(self):
        """Test weight sum validation for invalid weights."""
        weights = {2004: 0.5, 2005: 0.3, 2006: 0.1}  # Sum = 0.9
        is_valid, weight_sum = _validate_weight_sum(weights)
        assert not is_valid
        assert abs(weight_sum - 0.9) < 1e-10


# =============================================================================
# Task 5.3: WATT Formula Tests
# =============================================================================

class TestWATTFormula:
    """Tests for WATT point estimate calculation."""

    def test_watt_formula_basic(self):
        """Verify WATT = Σ w × ATT with known values."""
        # Setup: 3 cohorts at event_time=0
        df = pd.DataFrame({
            'cohort': [2004, 2005, 2006],
            'period': [2004, 2005, 2006],
            'att': [0.05, 0.08, 0.03],
            'se': [0.02, 0.03, 0.02],
            'df_inference': [45, 38, 30],
        })
        cohort_sizes = {2004: 50, 2005: 30, 2006: 20}
        
        results = aggregate_to_event_time(df, cohort_sizes)
        
        # Manual calculation
        # weights: 50/100=0.5, 30/100=0.3, 20/100=0.2
        # WATT = 0.5*0.05 + 0.3*0.08 + 0.2*0.03 = 0.025 + 0.024 + 0.006 = 0.055
        expected_watt = 0.5 * 0.05 + 0.3 * 0.08 + 0.2 * 0.03
        
        assert len(results) == 1
        assert results[0].event_time == 0
        assert abs(results[0].att - expected_watt) < 1e-10

    def test_watt_single_cohort_degeneracy(self, single_cohort_df, single_cohort_sizes):
        """Test that WATT = ATT when only 1 cohort contributes."""
        results = aggregate_to_event_time(single_cohort_df, single_cohort_sizes)
        
        # Each event time has only 1 cohort, so WATT should equal ATT
        for effect in results:
            # Find corresponding ATT in original data
            original = single_cohort_df[
                single_cohort_df['period'] - single_cohort_df['cohort'] == effect.event_time
            ]
            expected_att = original['att'].values[0]
            expected_se = original['se'].values[0]
            
            assert abs(effect.att - expected_att) < 1e-10
            assert abs(effect.se - expected_se) < 1e-10
            assert effect.n_cohorts == 1
            assert abs(effect.weight_sum - 1.0) < 1e-10

    def test_watt_bounds(self):
        """Test that WATT is within [min(ATT), max(ATT)]."""
        # Create data with varying ATT values
        df = pd.DataFrame({
            'cohort': [2004, 2005, 2006],
            'period': [2004, 2005, 2006],
            'att': [-0.10, 0.20, 0.05],  # min=-0.10, max=0.20
            'se': [0.02, 0.03, 0.02],
            'df_inference': [45, 38, 30],
        })
        cohort_sizes = {2004: 50, 2005: 30, 2006: 20}
        
        results = aggregate_to_event_time(df, cohort_sizes)
        
        min_att = -0.10
        max_att = 0.20
        
        for effect in results:
            assert min_att <= effect.att <= max_att, \
                f"WATT {effect.att} not in [{min_att}, {max_att}]"

    def test_watt_weighted_average_property(self):
        """Test weighted average property with equal ATT values."""
        # If all ATT values are equal, WATT should equal that value
        df = pd.DataFrame({
            'cohort': [2004, 2005, 2006],
            'period': [2004, 2005, 2006],
            'att': [0.10, 0.10, 0.10],  # All equal
            'se': [0.02, 0.03, 0.02],
            'df_inference': [45, 38, 30],
        })
        cohort_sizes = {2004: 50, 2005: 30, 2006: 20}
        
        results = aggregate_to_event_time(df, cohort_sizes)
        
        assert len(results) == 1
        assert abs(results[0].att - 0.10) < 1e-10


# =============================================================================
# Task 5.4: SE Formula Tests
# =============================================================================

class TestSEFormula:
    """Tests for SE calculation."""

    def test_se_formula_basic(self):
        """Verify SE = sqrt(Σ w² × SE²) with known values."""
        df = pd.DataFrame({
            'cohort': [2004, 2005, 2006],
            'period': [2004, 2005, 2006],
            'att': [0.05, 0.08, 0.03],
            'se': [0.02, 0.03, 0.02],
            'df_inference': [45, 38, 30],
        })
        cohort_sizes = {2004: 50, 2005: 30, 2006: 20}
        
        results = aggregate_to_event_time(df, cohort_sizes)
        
        # Manual calculation
        # weights: 0.5, 0.3, 0.2
        # SE² = 0.5²×0.02² + 0.3²×0.03² + 0.2²×0.02²
        #     = 0.25×0.0004 + 0.09×0.0009 + 0.04×0.0004
        #     = 0.0001 + 0.000081 + 0.000016 = 0.000197
        # SE = sqrt(0.000197) ≈ 0.01403
        expected_var = 0.25 * 0.0004 + 0.09 * 0.0009 + 0.04 * 0.0004
        expected_se = math.sqrt(expected_var)
        
        assert abs(results[0].se - expected_se) < 1e-8

    def test_se_single_cohort(self, single_cohort_df, single_cohort_sizes):
        """Test that SE = cohort SE when only 1 cohort."""
        results = aggregate_to_event_time(single_cohort_df, single_cohort_sizes)
        
        for effect in results:
            original = single_cohort_df[
                single_cohort_df['period'] - single_cohort_df['cohort'] == effect.event_time
            ]
            expected_se = original['se'].values[0]
            assert abs(effect.se - expected_se) < 1e-10

    def test_se_monotonicity(self):
        """Test that SE ≤ max(cohort SE) due to weight normalization."""
        df = pd.DataFrame({
            'cohort': [2004, 2005, 2006],
            'period': [2004, 2005, 2006],
            'att': [0.05, 0.08, 0.03],
            'se': [0.02, 0.10, 0.05],  # max SE = 0.10
            'df_inference': [45, 38, 30],
        })
        cohort_sizes = {2004: 50, 2005: 30, 2006: 20}
        
        results = aggregate_to_event_time(df, cohort_sizes)
        
        max_se = 0.10
        # SE(WATT) = sqrt(Σ w² × SE²) ≤ sqrt(Σ w² × max_SE²) = max_SE × sqrt(Σ w²)
        # Since Σ w² ≤ 1 (equality when one weight = 1), SE(WATT) ≤ max_SE
        assert results[0].se <= max_se + 1e-10

    def test_se_zero_handling(self):
        """Test handling of zero SE values."""
        df = pd.DataFrame({
            'cohort': [2004, 2005],
            'period': [2004, 2005],
            'att': [0.05, 0.08],
            'se': [0.0, 0.03],  # One zero SE
            'df_inference': [45, 38],
        })
        cohort_sizes = {2004: 50, 2005: 30}
        
        results = aggregate_to_event_time(df, cohort_sizes)
        
        # Should still compute (zero SE contributes 0 to variance)
        assert len(results) == 1
        assert results[0].se >= 0


# =============================================================================
# Task 5.5: t-distribution Tests
# =============================================================================

class TestTDistribution:
    """Tests for t-distribution inference."""

    def test_ci_uses_t_distribution(self):
        """Test that CI width varies with df (not fixed z=1.96)."""
        # Create two datasets with same ATT/SE but different df
        df_high = pd.DataFrame({
            'cohort': [2004],
            'period': [2004],
            'att': [0.10],
            'se': [0.02],
            'df_inference': [100],  # High df
        })
        df_low = pd.DataFrame({
            'cohort': [2004],
            'period': [2004],
            'att': [0.10],
            'se': [0.02],
            'df_inference': [5],  # Low df
        })
        cohort_sizes = {2004: 50}
        
        results_high = aggregate_to_event_time(df_high, cohort_sizes)
        results_low = aggregate_to_event_time(df_low, cohort_sizes)
        
        # CI should be wider for low df
        ci_width_high = results_high[0].ci_upper - results_high[0].ci_lower
        ci_width_low = results_low[0].ci_upper - results_low[0].ci_lower
        
        assert ci_width_low > ci_width_high, \
            f"Low df CI width {ci_width_low} should be > high df CI width {ci_width_high}"
        
        # Verify using t-distribution critical values
        t_crit_high = t_dist.ppf(0.975, 100)
        t_crit_low = t_dist.ppf(0.975, 5)
        expected_width_high = 2 * t_crit_high * 0.02
        expected_width_low = 2 * t_crit_low * 0.02
        
        assert abs(ci_width_high - expected_width_high) < 1e-8
        assert abs(ci_width_low - expected_width_low) < 1e-8

    def test_pvalue_uses_t_distribution(self):
        """Test that p-value is computed from t-distribution."""
        df = pd.DataFrame({
            'cohort': [2004],
            'period': [2004],
            'att': [0.10],
            'se': [0.05],  # t-stat = 0.10/0.05 = 2.0
            'df_inference': [10],
        })
        cohort_sizes = {2004: 50}
        
        results = aggregate_to_event_time(df, cohort_sizes)
        
        # Expected p-value from t-distribution
        t_stat = 0.10 / 0.05  # = 2.0
        expected_pvalue = 2 * (1 - t_dist.cdf(abs(t_stat), 10))
        
        assert abs(results[0].t_stat - t_stat) < 1e-10
        assert abs(results[0].pvalue - expected_pvalue) < 1e-8

    def test_df_selection_conservative(self):
        """Test conservative df selection (min across cohorts)."""
        cohort_dfs = {2004: 45, 2005: 38, 2006: 30}
        weights = {2004: 0.5, 2005: 0.3, 2006: 0.2}
        
        df = _select_degrees_of_freedom(cohort_dfs, weights, 'conservative', 3)
        
        assert df == 30  # min(45, 38, 30)

    def test_df_selection_weighted(self):
        """Test weighted average df selection."""
        cohort_dfs = {2004: 100, 2005: 50, 2006: 25}
        weights = {2004: 0.5, 2005: 0.3, 2006: 0.2}
        
        df = _select_degrees_of_freedom(cohort_dfs, weights, 'weighted', 3)
        
        # Expected: 0.5*100 + 0.3*50 + 0.2*25 = 50 + 15 + 5 = 70
        expected_df = int(round(0.5 * 100 + 0.3 * 50 + 0.2 * 25))
        assert df == expected_df

    def test_df_selection_fallback(self):
        """Test fallback df selection (n_cohorts - 1)."""
        cohort_dfs = {2004: 45, 2005: 38, 2006: 30}
        weights = {2004: 0.5, 2005: 0.3, 2006: 0.2}
        
        df = _select_degrees_of_freedom(cohort_dfs, weights, 'fallback', 3)
        
        assert df == 2  # 3 - 1

    def test_df_selection_missing_df(self):
        """Test df selection when some cohorts have missing df."""
        cohort_dfs = {2004: 45, 2005: None, 2006: 30}
        weights = {2004: 0.5, 2005: 0.3, 2006: 0.2}
        
        # Conservative should use min of valid dfs
        df = _select_degrees_of_freedom(cohort_dfs, weights, 'conservative', 3)
        assert df == 30  # min(45, 30)

    def test_df_selection_all_missing(self):
        """Test df selection when all cohorts have missing df."""
        cohort_dfs = {2004: None, 2005: None, 2006: None}
        weights = {2004: 0.5, 2005: 0.3, 2006: 0.2}
        
        # Should fallback to n_cohorts - 1
        df = _select_degrees_of_freedom(cohort_dfs, weights, 'conservative', 3)
        assert df == 2  # 3 - 1


# =============================================================================
# Task 5.6: Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_input_raises(self):
        """Test that empty cohort_time_effects raises ValueError."""
        empty_df = pd.DataFrame(columns=['cohort', 'period', 'att', 'se'])
        cohort_sizes = {2004: 50}
        
        with pytest.raises(ValueError, match="cannot be empty"):
            aggregate_to_event_time(empty_df, cohort_sizes)

    def test_empty_list_input_raises(self):
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            aggregate_to_event_time([], {2004: 50})

    def test_invalid_cohort_size_raises(self):
        """Test that non-positive cohort size raises ValueError."""
        df = pd.DataFrame({
            'cohort': [2004],
            'period': [2004],
            'att': [0.05],
            'se': [0.02],
        })
        
        with pytest.raises(ValueError, match="non-positive"):
            aggregate_to_event_time(df, {2004: 0})
        
        with pytest.raises(ValueError, match="non-positive"):
            aggregate_to_event_time(df, {2004: -10})

    def test_empty_cohort_sizes_raises(self):
        """Test that empty cohort_sizes raises ValueError."""
        df = pd.DataFrame({
            'cohort': [2004],
            'period': [2004],
            'att': [0.05],
            'se': [0.02],
        })
        
        with pytest.raises(ValueError, match="cannot be empty"):
            aggregate_to_event_time(df, {})

    def test_missing_se_excluded(self):
        """Test that cohorts with NaN SE are excluded with warning."""
        df = pd.DataFrame({
            'cohort': [2004, 2005],
            'period': [2004, 2005],
            'att': [0.05, 0.08],
            'se': [0.02, np.nan],  # 2005 has NaN SE
            'df_inference': [45, 38],
        })
        cohort_sizes = {2004: 50, 2005: 30}
        
        with pytest.warns(UserWarning, match="excluded"):
            results = aggregate_to_event_time(df, cohort_sizes, verbose=True)
        
        # Should still produce results (using only 2004)
        assert len(results) == 1
        assert results[0].n_cohorts == 1

    def test_missing_att_excluded(self):
        """Test that cohorts with NaN ATT are excluded."""
        df = pd.DataFrame({
            'cohort': [2004, 2005],
            'period': [2004, 2005],
            'att': [np.nan, 0.08],  # 2004 has NaN ATT
            'se': [0.02, 0.03],
            'df_inference': [45, 38],
        })
        cohort_sizes = {2004: 50, 2005: 30}
        
        with pytest.warns(UserWarning, match="excluded"):
            results = aggregate_to_event_time(df, cohort_sizes, verbose=True)
        
        assert len(results) == 1
        assert results[0].n_cohorts == 1

    def test_all_nan_returns_nan(self):
        """Test that all invalid data returns NaN result."""
        df = pd.DataFrame({
            'cohort': [2004, 2005],
            'period': [2004, 2005],
            'att': [np.nan, np.nan],
            'se': [np.nan, np.nan],
            'df_inference': [45, 38],
        })
        cohort_sizes = {2004: 50, 2005: 30}
        
        with pytest.warns(UserWarning):
            results = aggregate_to_event_time(df, cohort_sizes, verbose=True)
        
        assert len(results) == 1
        assert np.isnan(results[0].att)
        assert np.isnan(results[0].se)
        assert results[0].n_cohorts == 0

    def test_missing_required_columns_raises(self):
        """Test that missing required columns raises ValueError."""
        df = pd.DataFrame({
            'cohort': [2004],
            'period': [2004],
            'att': [0.05],
            # Missing 'se' column
        })
        
        with pytest.raises(ValueError, match="Missing required columns"):
            aggregate_to_event_time(df, {2004: 50})

    def test_event_time_range_filter(self, simple_cohort_time_df, simple_cohort_sizes):
        """Test event_time_range filtering."""
        # Add event_time column
        df = simple_cohort_time_df.copy()
        df['event_time'] = df['period'] - df['cohort']
        
        # Filter to only event_time >= 0
        results = aggregate_to_event_time(
            df, simple_cohort_sizes, event_time_range=(0, 2)
        )
        
        for effect in results:
            assert 0 <= effect.event_time <= 2

    def test_event_time_range_empty_result(self, simple_cohort_time_df, simple_cohort_sizes):
        """Test that empty event_time_range returns empty list with warning."""
        with pytest.warns(UserWarning, match="No effects found"):
            results = aggregate_to_event_time(
                simple_cohort_time_df, simple_cohort_sizes, 
                event_time_range=(100, 200)  # No data in this range
            )
        
        assert results == []


# =============================================================================
# EventTimeEffect Dataclass Tests
# =============================================================================

class TestEventTimeEffectDataclass:
    """Tests for EventTimeEffect dataclass."""

    def test_dataclass_instantiation(self):
        """Test that EventTimeEffect can be instantiated."""
        effect = EventTimeEffect(
            event_time=0,
            att=0.05,
            se=0.02,
            ci_lower=0.01,
            ci_upper=0.09,
            t_stat=2.5,
            pvalue=0.02,
            df_inference=45,
            n_cohorts=3,
            cohort_contributions={2004: 0.025, 2005: 0.015, 2006: 0.010},
            weight_sum=1.0,
            alpha=0.05,
        )
        
        assert effect.event_time == 0
        assert effect.att == 0.05
        assert effect.se == 0.02
        assert effect.n_cohorts == 3

    def test_dataclass_default_alpha(self):
        """Test that alpha defaults to 0.05."""
        effect = EventTimeEffect(
            event_time=0,
            att=0.05,
            se=0.02,
            ci_lower=0.01,
            ci_upper=0.09,
            t_stat=2.5,
            pvalue=0.02,
            df_inference=45,
            n_cohorts=3,
            cohort_contributions={},
            weight_sum=1.0,
        )
        
        assert effect.alpha == 0.05


# =============================================================================
# DataFrame Conversion Tests
# =============================================================================

class TestDataFrameConversion:
    """Tests for event_time_effects_to_dataframe."""

    def test_to_dataframe_basic(self, simple_cohort_time_df, simple_cohort_sizes):
        """Test conversion to DataFrame."""
        results = aggregate_to_event_time(simple_cohort_time_df, simple_cohort_sizes)
        df = event_time_effects_to_dataframe(results)
        
        assert isinstance(df, pd.DataFrame)
        assert 'event_time' in df.columns
        assert 'att' in df.columns
        assert 'se' in df.columns
        assert 'ci_lower' in df.columns
        assert 'ci_upper' in df.columns
        assert len(df) == len(results)

    def test_to_dataframe_empty(self):
        """Test conversion of empty list."""
        df = event_time_effects_to_dataframe([])
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert 'event_time' in df.columns

    def test_to_dataframe_none_raises(self):
        """Test that None input raises TypeError."""
        with pytest.raises(TypeError, match="cannot be None"):
            event_time_effects_to_dataframe(None)

"""
Unit tests for pre-treatment period dynamics functionality.

This module contains comprehensive unit tests for pre-treatment demeaning
(formula D.1) and detrending (formula D.2) transformations, control group
selection, effect estimation, and parallel trends testing.

Tests verify correctness properties including:
- Anchor point convention (t = g-1 → 0)
- Rolling window computations
- Boundary condition handling
- Numerical stability

Validates the pre-treatment diagnostic procedures (Appendix D) of the
Lee-Wooldridge Difference-in-Differences framework.

References
----------
Lee, S. & Wooldridge, J. M. (2025). A Simple Transformation Approach to
    Difference-in-Differences Estimation for Panel Data. SSRN 4516518.
"""

import numpy as np
import pandas as pd
import pytest

from lwdid.staggered.transformations_pre import (
    transform_staggered_demean_pre,
    transform_staggered_detrend_pre,
    get_pre_treatment_periods_for_cohort,
    _compute_rolling_mean_future,
    _compute_rolling_trend_future,
)
from lwdid.staggered.control_groups import (
    get_valid_control_units,
    get_all_control_masks_pre,
    ControlGroupStrategy,
)
from lwdid.staggered.estimation_pre import (
    PreTreatmentEffect,
    estimate_pre_treatment_effects,
    pre_treatment_effects_to_dataframe,
)
from lwdid.staggered.parallel_trends import (
    ParallelTrendsTestResult,
    run_parallel_trends_test,
    parallel_trends_test,
    summarize_parallel_trends_test,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def simple_panel_data():
    """Create simple panel data for testing."""
    # 4 units, 6 time periods
    # Unit 1: cohort 4 (treated at t=4)
    # Unit 2: cohort 4 (treated at t=4)
    # Unit 3: cohort 6 (treated at t=6)
    # Unit 4: never treated (g=0)
    data = pd.DataFrame({
        'id': [1]*6 + [2]*6 + [3]*6 + [4]*6,
        'time': list(range(1, 7)) * 4,
        'y': [
            10, 12, 14, 16, 18, 20,  # Unit 1: linear trend
            20, 22, 24, 26, 28, 30,  # Unit 2: linear trend
            5, 7, 9, 11, 13, 15,     # Unit 3: linear trend
            15, 15, 15, 15, 15, 15,  # Unit 4: constant
        ],
        'g': [4]*6 + [4]*6 + [6]*6 + [0]*6,
    })
    return data


@pytest.fixture
def minimal_panel_data():
    """Create minimal panel data with 2 pre-treatment periods."""
    data = pd.DataFrame({
        'id': [1, 1, 1, 2, 2, 2],
        'time': [1, 2, 3, 1, 2, 3],
        'y': [10, 20, 30, 15, 25, 35],
        'g': [3, 3, 3, 0, 0, 0],
    })
    return data


# =============================================================================
# Test: get_pre_treatment_periods_for_cohort
# =============================================================================


class TestGetPreTreatmentPeriods:
    """Tests for get_pre_treatment_periods_for_cohort function."""

    def test_basic_period_list(self):
        """Test basic period list generation."""
        periods = get_pre_treatment_periods_for_cohort(cohort=6, T_min=1)
        assert periods == [1, 2, 3, 4, 5]

    def test_single_pre_period(self):
        """Test with single pre-treatment period."""
        periods = get_pre_treatment_periods_for_cohort(cohort=2, T_min=1)
        assert periods == [1]

    def test_no_pre_periods(self):
        """Test when cohort equals T_min (no pre-treatment periods)."""
        periods = get_pre_treatment_periods_for_cohort(cohort=1, T_min=1)
        assert periods == []

    def test_large_cohort(self):
        """Test with large cohort value."""
        periods = get_pre_treatment_periods_for_cohort(cohort=100, T_min=90)
        assert periods == list(range(90, 100))


# =============================================================================
# Test: Pre-treatment Demeaning Transformation
# =============================================================================


class TestPreTreatmentDemeaning:
    """Tests for transform_staggered_demean_pre function."""

    def test_anchor_point_is_zero(self, simple_panel_data):
        """Verify t=g-1 produces exactly 0.0 for all units."""
        result = transform_staggered_demean_pre(
            simple_panel_data, 'y', 'id', 'time', 'g'
        )

        # For cohort 4, anchor is t=3
        anchor_col = 'ydot_pre_g4_t3'
        assert anchor_col in result.columns

        # All values at t=3 should be exactly 0
        anchor_values = result[result['time'] == 3][anchor_col].dropna()
        assert len(anchor_values) > 0
        np.testing.assert_array_equal(anchor_values.values, 0.0)

    def test_single_future_period_computation(self, simple_panel_data):
        """Verify t=g-2 uses single future value correctly."""
        result = transform_staggered_demean_pre(
            simple_panel_data, 'y', 'id', 'time', 'g'
        )

        # For cohort 4, t=2 has single future period (t=3)
        # ẏ_{i,2,4} = Y_{i,2} - Y_{i,3}
        col = 'ydot_pre_g4_t2'
        assert col in result.columns

        # Unit 1: Y_2 = 12, Y_3 = 14 → ẏ = 12 - 14 = -2
        unit1_t2 = result[(result['id'] == 1) & (result['time'] == 2)][col].values[0]
        assert np.isclose(unit1_t2, 12 - 14)

        # Unit 2: Y_2 = 22, Y_3 = 24 → ẏ = 22 - 24 = -2
        unit2_t2 = result[(result['id'] == 2) & (result['time'] == 2)][col].values[0]
        assert np.isclose(unit2_t2, 22 - 24)

    def test_multiple_future_periods_computation(self, simple_panel_data):
        """Verify rolling mean computation for t≤g-3."""
        result = transform_staggered_demean_pre(
            simple_panel_data, 'y', 'id', 'time', 'g'
        )

        # For cohort 4, t=1 has future periods {2, 3}
        # ẏ_{i,1,4} = Y_{i,1} - mean(Y_{i,2}, Y_{i,3})
        col = 'ydot_pre_g4_t1'
        assert col in result.columns

        # Unit 1: Y_1 = 10, mean(Y_2, Y_3) = mean(12, 14) = 13 → ẏ = 10 - 13 = -3
        unit1_t1 = result[(result['id'] == 1) & (result['time'] == 1)][col].values[0]
        assert np.isclose(unit1_t1, 10 - 13)

    def test_all_units_transformed(self, simple_panel_data):
        """Verify transformation applies to all units including never-treated."""
        result = transform_staggered_demean_pre(
            simple_panel_data, 'y', 'id', 'time', 'g'
        )

        # Check that never-treated unit (id=4) also has transformed values
        col = 'ydot_pre_g4_t1'
        unit4_t1 = result[(result['id'] == 4) & (result['time'] == 1)][col].values[0]
        # Unit 4: Y_1 = 15, mean(Y_2, Y_3) = mean(15, 15) = 15 → ẏ = 15 - 15 = 0
        assert np.isclose(unit4_t1, 0.0)

    def test_nan_handling_in_rolling_window(self):
        """Verify NaN propagation in rolling window."""
        data = pd.DataFrame({
            'id': [1, 1, 1, 1],
            'time': [1, 2, 3, 4],
            'y': [10, np.nan, 14, 16],  # NaN at t=2
            'g': [4, 4, 4, 4],
        })

        result = transform_staggered_demean_pre(data, 'y', 'id', 'time', 'g')

        # At t=1, future periods are {2, 3}. Y_2 is NaN, so mean uses only Y_3=14
        col = 'ydot_pre_g4_t1'
        val = result[(result['id'] == 1) & (result['time'] == 1)][col].values[0]
        # mean of [NaN, 14] with dropna = 14
        assert np.isclose(val, 10 - 14)

    def test_column_naming_convention(self, simple_panel_data):
        """Verify output columns follow naming convention."""
        result = transform_staggered_demean_pre(
            simple_panel_data, 'y', 'id', 'time', 'g'
        )

        # Check column names for cohort 4
        expected_cols = ['ydot_pre_g4_t1', 'ydot_pre_g4_t2', 'ydot_pre_g4_t3']
        for col in expected_cols:
            assert col in result.columns

    def test_missing_columns_error(self, simple_panel_data):
        """Verify error on missing required columns."""
        with pytest.raises(ValueError, match="Missing required columns"):
            transform_staggered_demean_pre(
                simple_panel_data, 'nonexistent', 'id', 'time', 'g'
            )


# =============================================================================
# Test: Pre-treatment Detrending Transformation
# =============================================================================


class TestPreTreatmentDetrending:
    """Tests for transform_staggered_detrend_pre function."""

    def test_anchor_point_is_zero(self, simple_panel_data):
        """Verify t=g-1 produces exactly 0.0."""
        result = transform_staggered_detrend_pre(
            simple_panel_data, 'y', 'id', 'time', 'g'
        )

        # For cohort 6, anchor is t=5
        anchor_col = 'ycheck_pre_g6_t5'
        assert anchor_col in result.columns

        anchor_values = result[result['time'] == 5][anchor_col].dropna()
        assert len(anchor_values) > 0
        np.testing.assert_array_equal(anchor_values.values, 0.0)

    def test_single_future_period_returns_nan(self, simple_panel_data):
        """Verify t=g-2 returns NaN (need 2 points for OLS)."""
        result = transform_staggered_detrend_pre(
            simple_panel_data, 'y', 'id', 'time', 'g'
        )

        # For cohort 6, t=4 has only 1 future period (t=5)
        col = 'ycheck_pre_g6_t4'
        assert col in result.columns

        # Should be NaN because OLS needs at least 2 points
        val = result[(result['id'] == 3) & (result['time'] == 4)][col].values[0]
        assert np.isnan(val)

    def test_two_future_periods_ols_computation(self, simple_panel_data):
        """Verify OLS fit with exactly 2 future periods."""
        result = transform_staggered_detrend_pre(
            simple_panel_data, 'y', 'id', 'time', 'g'
        )

        # For cohort 6, t=3 has future periods {4, 5}
        # Unit 3: Y = [5, 7, 9, 11, 13, 15] at t = [1, 2, 3, 4, 5, 6]
        # OLS on (t=4, Y=11), (t=5, Y=13): slope=2, intercept=3
        # Ŷ_3 = 3 + 2*3 = 9
        # Ÿ_3 = Y_3 - Ŷ_3 = 9 - 9 = 0

        col = 'ycheck_pre_g6_t3'
        val = result[(result['id'] == 3) & (result['time'] == 3)][col].values[0]
        # For linear data, detrended value should be ~0
        assert np.isclose(val, 0.0, atol=1e-10)

    def test_multiple_future_periods_ols(self, simple_panel_data):
        """Verify OLS fit with >2 future periods."""
        result = transform_staggered_detrend_pre(
            simple_panel_data, 'y', 'id', 'time', 'g'
        )

        # For cohort 6, t=1 has future periods {2, 3, 4, 5}
        # Unit 3: perfectly linear, so detrended should be 0
        col = 'ycheck_pre_g6_t1'
        val = result[(result['id'] == 3) & (result['time'] == 1)][col].values[0]
        assert np.isclose(val, 0.0, atol=1e-10)

    def test_warning_for_insufficient_pre_periods(self, minimal_panel_data):
        """Verify warning when cohort has <3 pre-treatment periods."""
        with pytest.warns(UserWarning, match="only 2 pre-treatment period"):
            transform_staggered_detrend_pre(
                minimal_panel_data, 'y', 'id', 'time', 'g'
            )


# =============================================================================
# Test: Pre-treatment Control Groups
# =============================================================================


class TestPreTreatmentControlGroups:
    """Tests for pre-treatment control group selection."""

    def test_not_yet_treated_selection(self, simple_panel_data):
        """Verify gvar > t selection for pre-treatment periods."""
        # At t=2, for cohort 4:
        # - Unit 1 (g=4): gvar > 2 → True (control)
        # - Unit 2 (g=4): gvar > 2 → True (control)
        # - Unit 3 (g=6): gvar > 2 → True (control)
        # - Unit 4 (g=0): never-treated → True (control)

        period_data = simple_panel_data[simple_panel_data['time'] == 2]
        control_mask = get_valid_control_units(
            period_data, 'g', 'id',
            cohort=4, period=2,
            strategy=ControlGroupStrategy.NOT_YET_TREATED,
            is_pre_treatment=True,
        )

        # All units should be valid controls at t=2 for cohort 4
        assert control_mask.sum() == 4

    def test_never_treated_only_strategy(self, simple_panel_data):
        """Verify 'nevertreated' excludes not-yet-treated."""
        period_data = simple_panel_data[simple_panel_data['time'] == 2]
        control_mask = get_valid_control_units(
            period_data, 'g', 'id',
            cohort=4, period=2,
            strategy=ControlGroupStrategy.NEVER_TREATED,
            is_pre_treatment=True,
        )

        # Only unit 4 (never-treated) should be control
        assert control_mask.sum() == 1

    def test_pre_treatment_period_validation(self, simple_panel_data):
        """Verify error when period >= cohort for pre-treatment."""
        period_data = simple_panel_data[simple_panel_data['time'] == 4]
        with pytest.raises(ValueError, match="must be < cohort"):
            get_valid_control_units(
                period_data, 'g', 'id',
                cohort=4, period=4,
                strategy=ControlGroupStrategy.NOT_YET_TREATED,
                is_pre_treatment=True,
            )

    def test_batch_control_masks_pre(self, simple_panel_data):
        """Test batch control mask generation for pre-treatment."""
        masks = get_all_control_masks_pre(
            simple_panel_data, 'g', 'id',
            cohorts=[4, 6],
            T_min=1,
            strategy=ControlGroupStrategy.NOT_YET_TREATED,
        )

        # Should have masks for cohort 4: t=1,2,3 and cohort 6: t=1,2,3,4,5
        assert (4, 1) in masks
        assert (4, 2) in masks
        assert (4, 3) in masks
        assert (6, 1) in masks
        assert (6, 5) in masks


# =============================================================================
# Test: PreTreatmentEffect Dataclass
# =============================================================================


class TestPreTreatmentEffectDataclass:
    """Tests for PreTreatmentEffect dataclass."""

    def test_dataclass_instantiation(self):
        """Test basic dataclass creation."""
        effect = PreTreatmentEffect(
            cohort=4,
            period=2,
            event_time=-2,
            att=0.05,
            se=0.02,
            ci_lower=0.01,
            ci_upper=0.09,
            t_stat=2.5,
            pvalue=0.012,
            n_treated=50,
            n_control=100,
            is_anchor=False,
            rolling_window_size=2,
        )

        assert effect.cohort == 4
        assert effect.event_time == -2
        assert effect.is_anchor is False

    def test_anchor_point_identification(self):
        """Test anchor point flag."""
        anchor = PreTreatmentEffect(
            cohort=4,
            period=3,
            event_time=-1,
            att=0.0,
            se=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
            t_stat=np.nan,
            pvalue=np.nan,
            n_treated=50,
            n_control=100,
            is_anchor=True,
            rolling_window_size=0,
        )

        assert anchor.is_anchor is True
        assert anchor.att == 0.0
        assert anchor.se == 0.0

    def test_dataclass_immutability(self):
        """Test that dataclass is frozen (immutable)."""
        effect = PreTreatmentEffect(
            cohort=4, period=2, event_time=-2,
            att=0.05, se=0.02, ci_lower=0.01, ci_upper=0.09,
            t_stat=2.5, pvalue=0.012, n_treated=50, n_control=100,
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            effect.att = 0.10


# =============================================================================
# Test: Pre-treatment Effect Estimation
# =============================================================================


class TestPreTreatmentEstimation:
    """Tests for estimate_pre_treatment_effects function."""

    def test_anchor_point_handling(self, simple_panel_data):
        """Test that anchor points have ATT=0, SE=0."""
        # First transform the data
        transformed = transform_staggered_demean_pre(
            simple_panel_data, 'y', 'id', 'time', 'g'
        )

        results = estimate_pre_treatment_effects(
            transformed, 'g', 'id', 'time',
            transform_type='demean',
        )

        # Find anchor points
        anchors = [r for r in results if r.is_anchor]
        assert len(anchors) > 0

        for anchor in anchors:
            assert anchor.att == 0.0
            assert anchor.se == 0.0
            assert anchor.event_time == -1

    def test_results_to_dataframe(self, simple_panel_data):
        """Test DataFrame conversion."""
        transformed = transform_staggered_demean_pre(
            simple_panel_data, 'y', 'id', 'time', 'g'
        )

        results = estimate_pre_treatment_effects(
            transformed, 'g', 'id', 'time',
            transform_type='demean',
        )

        df = pre_treatment_effects_to_dataframe(results)

        assert isinstance(df, pd.DataFrame)
        assert 'cohort' in df.columns
        assert 'event_time' in df.columns
        assert 'is_anchor' in df.columns
        assert len(df) == len(results)


# =============================================================================
# Test: Parallel Trends Testing
# =============================================================================


class TestParallelTrendsTesting:
    """Tests for parallel trends test functions."""

    def test_test_result_dataclass(self):
        """Test ParallelTrendsTestResult dataclass."""
        result = ParallelTrendsTestResult(
            individual_tests=pd.DataFrame(),
            joint_f_stat=2.5,
            joint_pvalue=0.08,
            joint_df1=3,
            joint_df2=100,
            n_pre_periods=3,
            excluded_periods=[-1],
            reject_null=False,
            alpha=0.05,
        )

        assert result.joint_f_stat == 2.5
        assert result.reject_null is False

    def test_anchor_point_exclusion(self, simple_panel_data):
        """Test that anchor points are excluded from testing."""
        transformed = transform_staggered_demean_pre(
            simple_panel_data, 'y', 'id', 'time', 'g'
        )

        pre_effects = estimate_pre_treatment_effects(
            transformed, 'g', 'id', 'time',
            transform_type='demean',
        )

        test_result = run_parallel_trends_test(pre_effects, alpha=0.05)

        # Anchor points (e=-1) should be in excluded_periods
        assert -1 in test_result.excluded_periods

    def test_summary_generation(self, simple_panel_data):
        """Test summary string generation."""
        transformed = transform_staggered_demean_pre(
            simple_panel_data, 'y', 'id', 'time', 'g'
        )

        pre_effects = estimate_pre_treatment_effects(
            transformed, 'g', 'id', 'time',
            transform_type='demean',
        )

        test_result = run_parallel_trends_test(pre_effects, alpha=0.05)
        summary = summarize_parallel_trends_test(test_result)

        assert isinstance(summary, str)
        assert "Parallel Trends Test" in summary
        assert "Joint Test" in summary


# =============================================================================
# Test: Numerical Stability
# =============================================================================


class TestNumericalStability:
    """Tests for numerical stability of transformations."""

    def test_extreme_values(self):
        """Test with extreme outcome values."""
        data = pd.DataFrame({
            'id': [1, 1, 1, 1],
            'time': [1, 2, 3, 4],
            'y': [1e10, 1e10 + 1, 1e10 + 2, 1e10 + 3],
            'g': [4, 4, 4, 4],
        })

        result = transform_staggered_demean_pre(data, 'y', 'id', 'time', 'g')

        # Should still compute correctly
        col = 'ydot_pre_g4_t1'
        val = result[(result['id'] == 1) & (result['time'] == 1)][col].values[0]
        # Y_1 - mean(Y_2, Y_3) = 1e10 - mean(1e10+1, 1e10+2) = 1e10 - (1e10+1.5) = -1.5
        assert np.isclose(val, -1.5, rtol=1e-10)

    def test_constant_outcome(self):
        """Test with constant outcome values."""
        data = pd.DataFrame({
            'id': [1, 1, 1, 1],
            'time': [1, 2, 3, 4],
            'y': [5.0, 5.0, 5.0, 5.0],
            'g': [4, 4, 4, 4],
        })

        result = transform_staggered_demean_pre(data, 'y', 'id', 'time', 'g')

        # All demeaned values should be 0 for constant outcome
        for t in [1, 2, 3]:
            col = f'ydot_pre_g4_t{t}'
            val = result[(result['id'] == 1) & (result['time'] == t)][col].values[0]
            assert np.isclose(val, 0.0, atol=1e-10)
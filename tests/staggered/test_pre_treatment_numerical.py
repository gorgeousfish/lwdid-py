"""
Numerical verification tests for pre-treatment period dynamics.

This module contains numerical tests that verify the mathematical
correctness of the pre-treatment transformation formulas using
explicit calculations and the vibe-math MCP tool.

Tests verify:
- Formula D.1 (demeaning) numerical correctness
- Formula D.2 (detrending) OLS coefficients
- Numerical stability with extreme values
- Boundary condition handling
"""

import numpy as np
import pandas as pd
import pytest


# =============================================================================
# Test Data Generators
# =============================================================================


def generate_linear_panel(n_units: int, n_periods: int, cohort: int, seed: int = 42):
    """
    Generate panel data with linear trends for each unit.

    Y_{it} = a_i + b_i * t

    where a_i and b_i are unit-specific intercept and slope.
    """
    np.random.seed(seed)

    data = []
    for i in range(1, n_units + 1):
        a_i = np.random.uniform(0, 10)
        b_i = np.random.uniform(0.5, 2.0)

        for t in range(1, n_periods + 1):
            y = a_i + b_i * t
            data.append({
                'id': i,
                'time': t,
                'y': y,
                'g': cohort if i <= n_units // 2 else 0,  # Half treated, half never-treated
                'true_intercept': a_i,
                'true_slope': b_i,
            })

    return pd.DataFrame(data)


def generate_constant_panel(n_units: int, n_periods: int, cohort: int, constant: float = 5.0):
    """Generate panel data with constant outcomes."""
    data = []
    for i in range(1, n_units + 1):
        for t in range(1, n_periods + 1):
            data.append({
                'id': i,
                'time': t,
                'y': constant,
                'g': cohort if i <= n_units // 2 else 0,
            })
    return pd.DataFrame(data)


# =============================================================================
# Test: Formula D.1 Verification (Demeaning)
# =============================================================================


class TestFormulaD1Verification:
    """
    Verify formula D.1:
    ẏ_{itg} = Y_{it} - (1/(g-t-1)) × Σ_{q=t+1}^{g-1} Y_{iq}
    """

    def test_demeaning_formula_explicit(self):
        """Verify demeaning formula with explicit calculation."""
        from lwdid.staggered.transformations_pre import transform_staggered_demean_pre

        # Create simple test data
        # Unit 1: Y = [10, 20, 30, 40] at t = [1, 2, 3, 4], g = 4
        data = pd.DataFrame({
            'id': [1, 1, 1, 1],
            'time': [1, 2, 3, 4],
            'y': [10.0, 20.0, 30.0, 40.0],
            'g': [4, 4, 4, 4],
        })

        result = transform_staggered_demean_pre(data, 'y', 'id', 'time', 'g')

        # At t=1: future periods = {2, 3}
        # ẏ_{1,1,4} = Y_{1,1} - mean(Y_{1,2}, Y_{1,3}) = 10 - mean(20, 30) = 10 - 25 = -15
        expected_t1 = 10 - (20 + 30) / 2
        actual_t1 = result[(result['id'] == 1) & (result['time'] == 1)]['ydot_pre_g4_t1'].values[0]
        assert np.isclose(actual_t1, expected_t1, rtol=1e-10)

        # At t=2: future periods = {3}
        # ẏ_{1,2,4} = Y_{1,2} - Y_{1,3} = 20 - 30 = -10
        expected_t2 = 20 - 30
        actual_t2 = result[(result['id'] == 1) & (result['time'] == 2)]['ydot_pre_g4_t2'].values[0]
        assert np.isclose(actual_t2, expected_t2, rtol=1e-10)

        # At t=3 (anchor): ẏ = 0
        actual_t3 = result[(result['id'] == 1) & (result['time'] == 3)]['ydot_pre_g4_t3'].values[0]
        assert actual_t3 == 0.0

    def test_demeaning_rolling_window_size(self):
        """Verify rolling window size is correct for each period."""
        from lwdid.staggered.transformations_pre import transform_staggered_demean_pre

        # Cohort g=6, periods 1-5 are pre-treatment
        data = pd.DataFrame({
            'id': [1]*6,
            'time': list(range(1, 7)),
            'y': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            'g': [6]*6,
        })

        result = transform_staggered_demean_pre(data, 'y', 'id', 'time', 'g')

        # Window sizes: t=1 → 4, t=2 → 3, t=3 → 2, t=4 → 1, t=5 → 0 (anchor)
        # At t=1: mean(Y_2, Y_3, Y_4, Y_5) = mean(20, 30, 40, 50) = 35
        expected_t1 = 10 - 35
        actual_t1 = result[(result['id'] == 1) & (result['time'] == 1)]['ydot_pre_g6_t1'].values[0]
        assert np.isclose(actual_t1, expected_t1, rtol=1e-10)

    def test_demeaning_with_missing_values(self):
        """Verify demeaning handles NaN correctly."""
        from lwdid.staggered.transformations_pre import transform_staggered_demean_pre

        data = pd.DataFrame({
            'id': [1, 1, 1, 1],
            'time': [1, 2, 3, 4],
            'y': [10.0, np.nan, 30.0, 40.0],  # NaN at t=2
            'g': [4, 4, 4, 4],
        })

        result = transform_staggered_demean_pre(data, 'y', 'id', 'time', 'g')

        # At t=1: future periods = {2, 3}, but Y_2 is NaN
        # mean should use only Y_3 = 30
        expected_t1 = 10 - 30
        actual_t1 = result[(result['id'] == 1) & (result['time'] == 1)]['ydot_pre_g4_t1'].values[0]
        assert np.isclose(actual_t1, expected_t1, rtol=1e-10)


# =============================================================================
# Test: Formula D.2 Verification (Detrending)
# =============================================================================


class TestFormulaD2Verification:
    """
    Verify formula D.2:
    Ÿ_{itg} = Y_{it} - Ŷ_{itg}
    where Ŷ_{itg} is OLS-fitted from Y_{iq} on q for q ∈ {t+1, ..., g-1}
    """

    def test_detrending_ols_coefficients(self):
        """Verify OLS coefficients match analytical solution."""
        from lwdid.staggered.transformations_pre import _compute_rolling_trend_future

        # Create unit data with known linear relationship
        # Y = 5 + 2*t
        unit_data = pd.DataFrame({
            'time': [1, 2, 3, 4, 5],
            'y': [7.0, 9.0, 11.0, 13.0, 15.0],  # Y = 5 + 2*t
        })

        # For t=1, cohort=5: future periods = {2, 3, 4}
        A, B = _compute_rolling_trend_future(unit_data, 'y', 'time', period=1, cohort=5)

        # OLS on (2, 9), (3, 11), (4, 13) should give slope=2, intercept=5
        assert np.isclose(B, 2.0, rtol=1e-10)
        assert np.isclose(A, 5.0, rtol=1e-10)

    def test_detrending_perfect_linear(self):
        """Verify detrending produces zero for perfectly linear data."""
        from lwdid.staggered.transformations_pre import transform_staggered_detrend_pre

        # Perfect linear data: Y = 5 + 2*t
        data = pd.DataFrame({
            'id': [1]*6,
            'time': list(range(1, 7)),
            'y': [7.0, 9.0, 11.0, 13.0, 15.0, 17.0],
            'g': [6]*6,
        })

        result = transform_staggered_detrend_pre(data, 'y', 'id', 'time', 'g')

        # For perfectly linear data, detrended values should be ~0
        for t in [1, 2, 3]:  # t=4 has only 1 future period (NaN), t=5 is anchor
            col = f'ycheck_pre_g6_t{t}'
            val = result[(result['id'] == 1) & (result['time'] == t)][col].values[0]
            assert np.isclose(val, 0.0, atol=1e-10), f"t={t}: expected 0, got {val}"

    def test_detrending_with_deviation(self):
        """Verify detrending captures deviation from linear trend."""
        from lwdid.staggered.transformations_pre import transform_staggered_detrend_pre

        # Data with deviation at t=1: Y = 5 + 2*t, but Y_1 = 10 (deviation of +3)
        data = pd.DataFrame({
            'id': [1]*5,
            'time': list(range(1, 6)),
            'y': [10.0, 9.0, 11.0, 13.0, 15.0],  # Y_1 = 10 instead of 7
            'g': [5]*5,
        })

        result = transform_staggered_detrend_pre(data, 'y', 'id', 'time', 'g')

        # At t=1: OLS on {2, 3, 4} gives Y = 5 + 2*t
        # Ŷ_1 = 5 + 2*1 = 7
        # Ÿ_1 = Y_1 - Ŷ_1 = 10 - 7 = 3
        col = 'ycheck_pre_g5_t1'
        val = result[(result['id'] == 1) & (result['time'] == 1)][col].values[0]
        assert np.isclose(val, 3.0, rtol=1e-10)

    def test_detrending_insufficient_points(self):
        """Verify NaN when insufficient points for OLS."""
        from lwdid.staggered.transformations_pre import _compute_rolling_trend_future

        unit_data = pd.DataFrame({
            'time': [1, 2, 3],
            'y': [10.0, 20.0, 30.0],
        })

        # For t=2, cohort=4: future periods = {3} (only 1 point)
        A, B = _compute_rolling_trend_future(unit_data, 'y', 'time', period=2, cohort=4)

        assert np.isnan(A)
        assert np.isnan(B)


# =============================================================================
# Test: OLS Coefficients Analytical Verification
# =============================================================================


class TestOLSCoefficients:
    """Verify OLS coefficients match analytical formulas."""

    def test_ols_two_points(self):
        """Verify OLS with exactly 2 points."""
        from lwdid.staggered.transformations_pre import _compute_rolling_trend_future

        # Two points: (2, 10), (3, 16)
        # Slope = (16 - 10) / (3 - 2) = 6
        # Intercept = 10 - 6*2 = -2
        unit_data = pd.DataFrame({
            'time': [1, 2, 3, 4],
            'y': [0.0, 10.0, 16.0, 0.0],
        })

        A, B = _compute_rolling_trend_future(unit_data, 'y', 'time', period=1, cohort=4)

        assert np.isclose(B, 6.0, rtol=1e-10)
        assert np.isclose(A, -2.0, rtol=1e-10)

    def test_ols_three_points(self):
        """Verify OLS with 3 points using normal equations."""
        from lwdid.staggered.transformations_pre import _compute_rolling_trend_future

        # Three points: (2, 5), (3, 8), (4, 9)
        # Using normal equations: X'X β = X'y
        unit_data = pd.DataFrame({
            'time': [1, 2, 3, 4, 5],
            'y': [0.0, 5.0, 8.0, 9.0, 0.0],
        })

        A, B = _compute_rolling_trend_future(unit_data, 'y', 'time', period=1, cohort=5)

        # Manual calculation:
        # X = [[1, 2], [1, 3], [1, 4]]
        # y = [5, 8, 9]
        # X'X = [[3, 9], [9, 29]]
        # X'y = [22, 70]
        # β = (X'X)^{-1} X'y
        X = np.array([[1, 2], [1, 3], [1, 4]])
        y = np.array([5, 8, 9])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]

        assert np.isclose(A, beta[0], rtol=1e-10)
        assert np.isclose(B, beta[1], rtol=1e-10)


# =============================================================================
# Test: Numerical Stability
# =============================================================================


class TestNumericalStability:
    """Tests for numerical stability of transformations."""

    def test_large_values(self):
        """Test with large outcome values."""
        from lwdid.staggered.transformations_pre import transform_staggered_demean_pre

        data = pd.DataFrame({
            'id': [1, 1, 1, 1],
            'time': [1, 2, 3, 4],
            'y': [1e12, 1e12 + 100, 1e12 + 200, 1e12 + 300],
            'g': [4, 4, 4, 4],
        })

        result = transform_staggered_demean_pre(data, 'y', 'id', 'time', 'g')

        # At t=1: mean(Y_2, Y_3) = mean(1e12+100, 1e12+200) = 1e12 + 150
        # ẏ = 1e12 - (1e12 + 150) = -150
        expected = -150.0
        actual = result[(result['id'] == 1) & (result['time'] == 1)]['ydot_pre_g4_t1'].values[0]
        assert np.isclose(actual, expected, rtol=1e-8)

    def test_small_values(self):
        """Test with small outcome values."""
        from lwdid.staggered.transformations_pre import transform_staggered_demean_pre

        data = pd.DataFrame({
            'id': [1, 1, 1, 1],
            'time': [1, 2, 3, 4],
            'y': [1e-10, 2e-10, 3e-10, 4e-10],
            'g': [4, 4, 4, 4],
        })

        result = transform_staggered_demean_pre(data, 'y', 'id', 'time', 'g')

        # At t=1: mean(Y_2, Y_3) = mean(2e-10, 3e-10) = 2.5e-10
        # ẏ = 1e-10 - 2.5e-10 = -1.5e-10
        expected = 1e-10 - 2.5e-10
        actual = result[(result['id'] == 1) & (result['time'] == 1)]['ydot_pre_g4_t1'].values[0]
        assert np.isclose(actual, expected, rtol=1e-8)

    def test_mixed_sign_values(self):
        """Test with mixed positive and negative values."""
        from lwdid.staggered.transformations_pre import transform_staggered_demean_pre

        data = pd.DataFrame({
            'id': [1, 1, 1, 1],
            'time': [1, 2, 3, 4],
            'y': [-10.0, 5.0, -3.0, 8.0],
            'g': [4, 4, 4, 4],
        })

        result = transform_staggered_demean_pre(data, 'y', 'id', 'time', 'g')

        # At t=1: mean(Y_2, Y_3) = mean(5, -3) = 1
        # ẏ = -10 - 1 = -11
        expected = -10 - 1
        actual = result[(result['id'] == 1) & (result['time'] == 1)]['ydot_pre_g4_t1'].values[0]
        assert np.isclose(actual, expected, rtol=1e-10)


# =============================================================================
# Test: Boundary Conditions
# =============================================================================


class TestBoundaryConditions:
    """Tests for boundary condition handling."""

    def test_minimum_pre_treatment_periods_demean(self):
        """Test demeaning with minimum pre-treatment periods (2)."""
        from lwdid.staggered.transformations_pre import transform_staggered_demean_pre

        # Cohort 3, T_min=1: pre-treatment periods = {1, 2}
        data = pd.DataFrame({
            'id': [1, 1, 1],
            'time': [1, 2, 3],
            'y': [10.0, 20.0, 30.0],
            'g': [3, 3, 3],
        })

        result = transform_staggered_demean_pre(data, 'y', 'id', 'time', 'g')

        # t=2 is anchor → 0
        assert result[(result['time'] == 2)]['ydot_pre_g3_t2'].values[0] == 0.0

        # t=1: mean(Y_2) = 20, ẏ = 10 - 20 = -10
        assert np.isclose(
            result[(result['time'] == 1)]['ydot_pre_g3_t1'].values[0],
            -10.0
        )

    def test_single_pre_treatment_period(self):
        """Test with single pre-treatment period (only anchor)."""
        from lwdid.staggered.transformations_pre import transform_staggered_demean_pre

        # Cohort 2, T_min=1: pre-treatment periods = {1}
        data = pd.DataFrame({
            'id': [1, 1],
            'time': [1, 2],
            'y': [10.0, 20.0],
            'g': [2, 2],
        })

        result = transform_staggered_demean_pre(data, 'y', 'id', 'time', 'g')

        # t=1 is anchor → 0
        assert result[(result['time'] == 1)]['ydot_pre_g2_t1'].values[0] == 0.0

    def test_all_nan_future_window(self):
        """Test when all future values are NaN."""
        from lwdid.staggered.transformations_pre import transform_staggered_demean_pre

        data = pd.DataFrame({
            'id': [1, 1, 1, 1],
            'time': [1, 2, 3, 4],
            'y': [10.0, np.nan, np.nan, 40.0],  # NaN at t=2,3
            'g': [4, 4, 4, 4],
        })

        result = transform_staggered_demean_pre(data, 'y', 'id', 'time', 'g')

        # At t=1: future periods = {2, 3}, both NaN → result should be NaN
        val = result[(result['id'] == 1) & (result['time'] == 1)]['ydot_pre_g4_t1'].values[0]
        assert np.isnan(val)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

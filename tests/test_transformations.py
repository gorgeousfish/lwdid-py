"""Unit tests for the rolling transformation module (``transformations.py``).

This module validates the four rolling transformations that underpin the
Lee-Wooldridge Difference-in-Differences estimator:

- **demean**: Unit-specific demeaning using pre-treatment means
  (Procedure 2.1; Paper §4.12).
- **detrend**: Unit-specific linear detrending fitted on pre-treatment
  periods (Procedure 3.1; Paper §4.13).
- **demeanq**: Quarterly demeaning with seasonal-effect removal.
- **detrendq**: Quarterly detrending with joint trend and seasonal-effect
  removal.

Each transformation produces the residualized outcome ``ydot`` and its
post-treatment average ``ydot_postavg``, which serve as inputs to the
cross-sectional OLS regression for ATT estimation.

References
----------
Lee, S. & Wooldridge, J. M. (2026). Simple Approaches to Inference with
    DiD Estimators with Small Cross-Sectional Sample Sizes. SSRN 5325686.
Lee, S. & Wooldridge, J. M. (2025). A Simple Transformation Approach to
    DiD Estimation for Panel Data. SSRN 4516518.
"""

import numpy as np
import pandas as pd
import pytest

from lwdid.transformations import apply_rolling_transform


class TestDemeanTransform:
    """Tests for the demeaning transformation (Procedure 2.1, Paper §4.12).

    The demeaning transformation computes the residualized outcome as
    ŷ_{it} = y_{it} − ȳ_{i,pre}, where ȳ_{i,pre} is the arithmetic mean
    of unit i's outcomes over the pre-treatment periods (t = 1, …, g−1).
    This class verifies the numerical correctness of the transformation,
    the broadcasting of post-treatment averages, and the identification
    of the first post-treatment observation.
    """

    def test_demean_mve(self):
        """Manual verification example (MVE) for the demeaning transformation.

        Constructs a minimal balanced panel (N=3 units, T=3 periods with
        2 pre-treatment and 1 post-treatment period) and verifies the
        residualized outcomes against hand-computed expectations.

        Hand-computed expectations (Paper §4.12):
        - Unit 1 (treated): ȳ₁,pre = (5+6)/2 = 5.5
                   ydot₁ = [5−5.5, 6−5.5, 9−5.5] = [−0.5, 0.5, 3.5]
                   ydot_postavg₁ = 3.5
        - Unit 2 (control): ȳ₂,pre = (3+4)/2 = 3.5
                   ydot₂ = [−0.5, 0.5, 1.5]
                   ydot_postavg₂ = 1.5
        - Unit 3 (control): ȳ₃,pre = (7+8)/2 = 7.5
                   ydot₃ = [−0.5, 0.5, −1.5]
                   ydot_postavg₃ = −1.5
        """
        # Load MVE test data
        data = pd.read_csv('tests/data/mve_demean.csv')
        
        # Construct internal indicator columns (mirrors validation.py output)
        data['d_'] = (data['d'] != 0).astype(int)
        data['post_'] = (data['post'] != 0).astype(int)
        data['tindex'] = data['year']  # time index already coded as 1, 2, 3
        
        # Apply the demeaning transformation
        data_transformed = apply_rolling_transform(
            data=data,
            y='y',
            ivar='id',
            tindex='tindex',
            post='post_',
            rolling='demean',
            tpost1=3,
            quarter=None,
        )
        
        # Verify ydot for unit 1 (numerical tolerance < 1e-7)
        unit1_ydot_expected = np.array([-0.5, 0.5, 3.5])
        unit1_ydot_actual = data_transformed[data_transformed['id'] == 1]['ydot'].values
        assert np.allclose(unit1_ydot_actual, unit1_ydot_expected, atol=1e-7)
        
        # Verify ydot for unit 2
        unit2_ydot_expected = np.array([-0.5, 0.5, 1.5])
        unit2_ydot_actual = data_transformed[data_transformed['id'] == 2]['ydot'].values
        assert np.allclose(unit2_ydot_actual, unit2_ydot_expected, atol=1e-7)
        
        # Verify ydot for unit 3
        unit3_ydot_expected = np.array([-0.5, 0.5, -1.5])
        unit3_ydot_actual = data_transformed[data_transformed['id'] == 3]['ydot'].values
        assert np.allclose(unit3_ydot_actual, unit3_ydot_expected, atol=1e-7)
    
    def test_demean_pre_mean_precision(self):
        """Verify that pre-treatment residual means are numerically zero.

        After demeaning, the arithmetic mean of each unit's pre-treatment
        residuals must equal zero by construction (the in-sample mean is
        subtracted). This test asserts that the residual mean is below
        1e-10 for each unit, confirming numerical precision.
        """
        data = pd.read_csv('tests/data/mve_demean.csv')
        data['d_'] = (data['d'] != 0).astype(int)
        data['post_'] = (data['post'] != 0).astype(int)
        data['tindex'] = data['year']
        
        data_transformed = apply_rolling_transform(
            data, 'y', 'id', 'tindex', 'post_', 'demean', 3
        )
        
        # Pre-treatment means of ydot should be zero by construction
        for unit_id in [1, 2, 3]:
            unit_pre_ydot = data_transformed[
                (data_transformed['id'] == unit_id) & 
                (data_transformed['post_'] == 0)
            ]['ydot']
            pre_mean = unit_pre_ydot.mean()
            # After demeaning, the pre-treatment residual mean is zero
            assert abs(pre_mean) < 1e-10
    
    def test_ydot_postavg_broadcast(self):
        """Verify that ``ydot_postavg`` is broadcast as a unit-level constant.

        The variable ``ydot_postavg`` stores the post-treatment average of
        the residualized outcome for each unit. Two properties are verified:

        1. Within each unit, ``ydot_postavg`` is constant across all rows
           (both pre- and post-treatment periods).
        2. The value equals the arithmetic mean of that unit's post-treatment
           ``ydot`` values.
        """
        data = pd.read_csv('tests/data/mve_demean.csv')
        data['d_'] = (data['d'] != 0).astype(int)
        data['post_'] = (data['post'] != 0).astype(int)
        data['tindex'] = data['year']
        
        data_transformed = apply_rolling_transform(
            data, 'y', 'id', 'tindex', 'post_', 'demean', 3
        )
        
        # Unit 1: ydot_postavg should be constant = 3.5 across all rows
        unit1_postavg = data_transformed[data_transformed['id'] == 1]['ydot_postavg']
        assert len(unit1_postavg.unique()) == 1, "ydot_postavg must be constant within unit"
        assert abs(unit1_postavg.iloc[0] - 3.5) < 1e-7

        # Unit 2: ydot_postavg = 1.5
        unit2_postavg = data_transformed[data_transformed['id'] == 2]['ydot_postavg']
        assert len(unit2_postavg.unique()) == 1, "ydot_postavg must be constant within unit"
        assert abs(unit2_postavg.iloc[0] - 1.5) < 1e-7

        # Unit 3: ydot_postavg = −1.5
        unit3_postavg = data_transformed[data_transformed['id'] == 3]['ydot_postavg']
        assert len(unit3_postavg.unique()) == 1, "ydot_postavg must be constant within unit"
        assert abs(unit3_postavg.iloc[0] - (-1.5)) < 1e-7

    def test_firstpost_identification(self):
        """Verify correct identification of the first post-treatment observation.

        The ``firstpost`` indicator marks the row where ``tindex == tpost1``
        for each unit. This row is used to construct the cross-sectional
        sample for OLS estimation of the ATT (one observation per unit).

        Assertions:
        1. Exactly N rows are flagged across the entire panel.
        2. Each unit contributes exactly one flagged row.
        3. The flagged row corresponds to ``tindex == tpost1``.
        """
        data = pd.read_csv('tests/data/mve_demean.csv')
        data['d_'] = (data['d'] != 0).astype(int)
        data['post_'] = (data['post'] != 0).astype(int)
        data['tindex'] = data['year']

        data_transformed = apply_rolling_transform(
            data, 'y', 'id', 'tindex', 'post_', 'demean', tpost1=3
        )

        # Exactly N = 3 rows should be flagged (one per unit)
        n_firstpost = data_transformed['firstpost'].sum()
        assert n_firstpost == 3, f"Expected 3 firstpost rows, got {n_firstpost}"

        # Each unit must have exactly one flagged row at tindex == tpost1
        for unit_id in [1, 2, 3]:
            unit_firstpost = data_transformed[
                (data_transformed['id'] == unit_id) &
                (data_transformed['firstpost'])
            ]
            assert len(unit_firstpost) == 1, \
                f"Unit {unit_id} should have exactly 1 firstpost row"

            # The flagged row must correspond to the first post-treatment period
            assert unit_firstpost['tindex'].iloc[0] == 3, \
                f"Unit {unit_id} firstpost should be at tpost1=3"
    
    def test_ydot_all_periods(self):
        """Verify that ``ydot`` is computed for all periods (including pre).

        In the Stata implementation, residuals are computed for both pre-
        and post-treatment periods so that the residualized outcome can be
        plotted over the entire sample. The Python implementation mirrors
        this behavior.
        """
        data = pd.read_csv('tests/data/mve_demean.csv')
        data['d_'] = (data['d'] != 0).astype(int)
        data['post_'] = (data['post'] != 0).astype(int)
        data['tindex'] = data['year']
        
        data_transformed = apply_rolling_transform(
            data, 'y', 'id', 'tindex', 'post_', 'demean', 3
        )
        
        # All rows (including pre-treatment periods) should have ydot defined
        assert data_transformed['ydot'].notna().all()
        
        # Pre-treatment ydot values must also be present
        pre_ydot = data_transformed[data_transformed['post_'] == 0]['ydot']
        assert len(pre_ydot) == 6  # 3 units × 2 pre periods
        assert pre_ydot.notna().all()


class TestDetrendTransform:
    """Tests for the linear detrending transformation (Procedure 3.1, Paper §4.13).

    The detrending transformation fits a unit-specific OLS linear trend
    ŷ_{it} = α̂_i + β̂_i · t exclusively on pre-treatment periods and
    subtracts the predicted values from all periods. Post-treatment
    residuals thus capture deviations from the counterfactual trend.
    """
    
    def test_detrend_mve(self):
        """Manual verification example (MVE) for the detrending transformation.

        Constructs a balanced panel (N=3 units, T=5 periods with 3 pre-treatment
        and 2 post-treatment periods) where outcomes follow perfectly linear
        trends, and verifies residuals against hand-computed expectations.

        Data-generating process (Paper §4.13):
        - Unit 1 (treated): y = 3 + 2t + 5·post  (treatment effect = 5)
        - Unit 2 (control): y = 1 + t             (no treatment effect)
        - Unit 3 (control): y = 4 + 1.5t          (no treatment effect)

        Hand-computed expectations:
        - Unit 1: α̂₁=3, β̂₁=2 → ydot = [0, 0, 0, 5, 5], ydot_postavg = 5
        - Unit 2: α̂₂=1, β̂₂=1 → ydot = [0, 0, 0, 0, 0], ydot_postavg = 0
        - Unit 3: α̂₃=4, β̂₃=1.5 → ydot = [0, 0, 0, 0, 0], ydot_postavg = 0
        - Implied ATT = 5 − 0 = 5
        """
        # Load MVE test data
        data = pd.read_csv('tests/data/mve_detrend.csv')
        
        # Construct internal indicator columns (mirrors validation.py output)
        data['d_'] = (data['d'] != 0).astype(int)
        data['post_'] = (data['post'] != 0).astype(int)
        # tindex column already present with values 1, 2, 3, 4, 5
        
        # Apply the detrending transformation
        data_transformed = apply_rolling_transform(
            data=data,
            y='y',
            ivar='id',
            tindex='tindex',
            post='post_',
            rolling='detrend',
            tpost1=4,
            quarter=None,
        )
        
        # Validate pre-period residuals: for perfectly linear data,
        # the OLS fit is exact and residuals should be ≈ 0 (< 1e-12)
        for unit_id in [1, 2, 3]:
            unit_pre_ydot = data_transformed[
                (data_transformed['id'] == unit_id) & 
                (data_transformed['post_'] == 0)
            ]['ydot'].values
            assert np.allclose(unit_pre_ydot, 0, atol=1e-12), \
                f"Unit {unit_id}: pre-period residuals should be zero for linear DGP"
        
        # Validate post-period residuals against hand-computed expectations
        # Unit 1 (treated): post y = [16, 18], ŷ = [11, 13], ydot = [5, 5]
        unit1_post_ydot = data_transformed[
            (data_transformed['id'] == 1) & 
            (data_transformed['post_'] == 1)
        ]['ydot'].values
        assert np.allclose(unit1_post_ydot, [5.0, 5.0], atol=1e-7)
        
        # Unit 2 (control): post y = [5, 6], ŷ = [5, 6], ydot = [0, 0]
        unit2_post_ydot = data_transformed[
            (data_transformed['id'] == 2) & 
            (data_transformed['post_'] == 1)
        ]['ydot'].values
        assert np.allclose(unit2_post_ydot, [0.0, 0.0], atol=1e-7)
        
        # Unit 3 (control): post y = [10, 11.5], ŷ = [10, 11.5], ydot = [0, 0]
        unit3_post_ydot = data_transformed[
            (data_transformed['id'] == 3) & 
            (data_transformed['post_'] == 1)
        ]['ydot'].values
        assert np.allclose(unit3_post_ydot, [0.0, 0.0], atol=1e-7)
        
        # Validate ydot_postavg (post-treatment mean of residualized outcomes)
        # Unit 1: mean([5, 5]) = 5.0
        unit1_postavg = data_transformed[data_transformed['id'] == 1]['ydot_postavg'].iloc[0]
        assert abs(unit1_postavg - 5.0) < 1e-7
        
        # Unit 2: mean([0, 0]) = 0.0
        unit2_postavg = data_transformed[data_transformed['id'] == 2]['ydot_postavg'].iloc[0]
        assert abs(unit2_postavg - 0.0) < 1e-7
        
        # Unit 3: mean([0, 0]) = 0.0
        unit3_postavg = data_transformed[data_transformed['id'] == 3]['ydot_postavg'].iloc[0]
        assert abs(unit3_postavg - 0.0) < 1e-7
    
    def test_detrend_error_T0_insufficient(self):
        """Verify that detrending raises an error when T₀ < 2.

        Linear detrending requires at least two pre-treatment periods to
        identify the intercept and slope. When T₀ = 1, the system must
        raise ``InsufficientPrePeriodsError`` with a message indicating
        that at least 2 pre-treatment periods are required.
        """
        from lwdid.exceptions import InsufficientPrePeriodsError
        
        # Construct a panel with T₀ = 1 (N=3, T=3; 1 pre, 2 post)
        data = pd.DataFrame({
            'id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
            'tindex': [1, 2, 3, 1, 2, 3, 1, 2, 3],
            'year': [1, 2, 3, 1, 2, 3, 1, 2, 3],
            'y': [5.0, 6.0, 7.0, 3.0, 4.0, 5.0, 7.0, 8.0, 9.0],
            'd_': [1, 1, 1, 0, 0, 0, 0, 0, 0],
            'post_': [0, 1, 1, 0, 1, 1, 0, 1, 1],
        })
        
        # Detrending with T₀ = 1 must raise InsufficientPrePeriodsError
        with pytest.raises(
            InsufficientPrePeriodsError,
            match="rolling\\('detrend'\\) requires at least 2 pre-treatment periods"
        ):
            apply_rolling_transform(
                data, 'y', 'id', 'tindex', 'post_',
                rolling='detrend', tpost1=2
            )


class TestQuarterlyTransforms:
    """Tests for quarterly rolling transformations (``demeanq`` and ``detrendq``).

    Quarterly transformations extend the annual procedures by jointly
    removing seasonal (quarter-specific) effects alongside the mean or
    linear trend. The ``demeanq`` variant estimates an intercept plus
    three quarter dummies; ``detrendq`` adds a linear time trend,
    yielding five parameters in total.
    """
    
    def test_demeanq_mve(self):
        """Manual verification example (MVE) for the quarterly demeaning transformation.

        Constructs a panel (N=2 units, T=9 periods with 5 pre-treatment and
        4 post-treatment periods) exhibiting seasonal patterns. The ``demeanq``
        transformation estimates a unit-specific intercept and three quarter
        dummies from the pre-treatment sample, then removes these effects
        from all periods.

        Verification criteria:
        1. Pre-treatment residuals are near zero (perfect in-sample fit).
        2. Post-treatment residuals are non-null and reflect the treatment
           effect net of seasonality.
        3. ``ydot_postavg`` equals the post-treatment mean of ``ydot``.

        Note: ``demeanq`` requires at least 5 pre-treatment observations to
        ensure df ≥ 1 after estimating the intercept and 3 quarter dummies.
        """
        # Construct seasonal MVE data with 5 pre-treatment periods (demeanq requires ≥ 5)
        data = pd.DataFrame({
            'id': [1,1,1,1,1,1,1,1,1, 2,2,2,2,2,2,2,2,2],
            'year': [1,1,1,1,1,2,2,2,2, 1,1,1,1,1,2,2,2,2],
            'quarter': [1,2,3,4,1,2,3,4,1, 1,2,3,4,1,2,3,4,1],
            'y': [10.0, 12.0, 11.0, 13.0, 10.0, 15.0, 17.0, 16.0, 18.0,
                  5.0, 7.0, 6.0, 8.0, 5.0, 5.0, 7.0, 6.0, 8.0],
            'd_': [1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0],
            'post_': [0,0,0,0,0,1,1,1,1, 0,0,0,0,0,1,1,1,1],
            'tindex': [1,2,3,4,5,6,7,8,9, 1,2,3,4,5,6,7,8,9],
        })

        # Apply demeanq transformation (first post-treatment period at t = 6)
        data_transformed = apply_rolling_transform(
            data=data, y='y', ivar='id', tindex='tindex', post='post_',
            rolling='demeanq', tpost1=6, quarter='quarter'
        )

        # Validate pre-treatment residuals (perfect in-sample fit expected)
        unit1_pre = data_transformed[
            (data_transformed['id'] == 1) & (data_transformed['post_'] == 0)
        ]
        assert np.allclose(unit1_pre['ydot'].values, 0, atol=1e-10), \
            "Pre-treatment residuals should be near zero for perfect in-sample fit"

        # Validate that post-treatment residuals are computed for all observations
        unit1_post = data_transformed[
            (data_transformed['id'] == 1) & (data_transformed['post_'] == 1)
        ]
        assert len(unit1_post) == 4, \
            f"Expected 4 post-treatment observations, got {len(unit1_post)}"
        assert unit1_post['ydot'].notna().all(), \
            "All post-treatment ydot values should be non-null"
        
        # Verify that required output columns are present
        assert 'ydot_postavg' in data_transformed.columns
        assert 'firstpost' in data_transformed.columns
        
        # Unit 1: ydot_postavg = mean([5, 5, 5, 5]) = 5.0
        unit1_postavg = data_transformed[data_transformed['id']==1]['ydot_postavg'].iloc[0]
        assert abs(unit1_postavg - 5.0) < 1e-7
    
    def test_detrendq_mve(self):
        """Test detrendq transformation on data with linear trend and seasonality.

        Data structure:
        - N=2 units, T=10 periods (6 pre-periods + 4 post-periods)
        - Unit 1: y = 5 + 0.5*t + seasonal effects
        - Unit 2: y = 3 + 0.3*t + seasonal effects

        The detrendq transformation:
        1. Estimates unit-specific linear trend and quarterly effects from pre-period
        2. Removes these effects from all periods
        3. Pre-period residuals should be small (good fit)

        Note: detrendq requires at least 6 pre-period observations to ensure
        df = n - k >= 1 after estimating intercept (1), trend (1), and
        quarterly dummy coefficients (3), totaling k=5 parameters.
        """
        # Construct data with trend + seasonality (6 pre-periods for detrendq)
        # detrendq estimates 5 parameters: intercept + trend + 3 quarter dummies
        # Requires n >= k + 1 = 6 pre-period observations for df >= 1
        data = pd.DataFrame({
            'id': [1,1,1,1,1,1,1,1,1,1, 2,2,2,2,2,2,2,2,2,2],
            'year': [1,1,1,1,1,1,2,2,2,2, 1,1,1,1,1,1,2,2,2,2],
            'quarter': [1,2,3,4,1,2,3,4,1,2, 1,2,3,4,1,2,3,4,1,2],
            'tindex': [1,2,3,4,5,6,7,8,9,10, 1,2,3,4,5,6,7,8,9,10],
            # Unit 1: base=5, trend=0.5*t, seasonal=[0, 1, 0.5, 1.5]
            # y = 5 + 0.5*t + seasonal[q], seasonal = [0, 1, 0.5, 1.5] for q=1,2,3,4
            'y': [5.5, 7.0, 7.0, 8.5, 7.5, 9.0, 9.0, 10.5, 10.0, 11.5,
                  # Unit 2: base=3, trend=0.3*t, seasonal=[0, 1, 0.5, 1.5]
                  3.3, 4.6, 4.4, 5.7, 4.5, 5.8, 5.6, 6.9, 6.0, 7.3],
            'd_': [1,1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0,0],
            'post_': [0,0,0,0,0,0,1,1,1,1, 0,0,0,0,0,0,1,1,1,1],
        })
        
        # Apply detrendq transformation (tpost1=7 because first post-period is at t=7)
        data_transformed = apply_rolling_transform(
            data=data, y='y', ivar='id', tindex='tindex', post='post_',
            rolling='detrendq', tpost1=7, quarter='quarter'
        )

        # Verify required columns are present
        assert 'ydot' in data_transformed.columns, "ydot column should be present"
        assert 'ydot_postavg' in data_transformed.columns, \
            "ydot_postavg column should be present"
        assert 'firstpost' in data_transformed.columns, \
            "firstpost column should be present"

        # Verify pre-period residuals are small (trend and seasonal effects removed)
        unit1_pre = data_transformed[
            (data_transformed['id'] == 1) & (data_transformed['post_'] == 0)
        ]
        assert unit1_pre['ydot'].abs().max() < 0.5, \
            "Pre-period residuals should be small after detrending and seasonal adjustment"
    
    def test_quarter_diversity_insufficient(self):
        """Test error handling for insufficient pre-period observations in demeanq.

        demeanq requires at least 2 pre-period observations to ensure df ≥ 1
        after estimating the intercept and quarterly dummy coefficients.

        This test verifies that InsufficientPrePeriodsError is raised when
        a unit has only 1 pre-period observation.
        """
        from lwdid.exceptions import InsufficientPrePeriodsError

        # Construct data with only 1 pre-period observation for unit 1
        data = pd.DataFrame({
            'id': [1, 1, 2, 2, 2],
            'quarter': [1, 1, 1, 2, 1],
            'y': [10.0, 15.0, 5.0, 6.0, 7.0],
            'post_': [0, 1, 0, 0, 1],
            'tindex': [1, 5, 1, 2, 5],
        })

        # Should raise InsufficientPrePeriodsError for unit 1 (only 1 pre-period obs)
        with pytest.raises(
            InsufficientPrePeriodsError,
            match="requires at least 2 observations"
        ):
            apply_rolling_transform(
                data=data, y='y', ivar='id', tindex='tindex', post='post_',
                rolling='demeanq', tpost1=5, quarter='quarter'
            )

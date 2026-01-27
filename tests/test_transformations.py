"""Transformation Module Tests

Unit tests for rolling transformations implemented in transformations.py:
- demean: Unit-specific demeaning (Procedure 2.1)
- detrend: Unit-specific linear detrending (Procedure 3.1)
- demeanq: Quarterly demeaning with seasonal effects
- detrendq: Quarterly detrending with seasonal effects

Tests verify construction of ydot and ydot_postavg variables used by lwdid.
"""

import numpy as np
import pandas as pd
import pytest

from lwdid.transformations import apply_rolling_transform


class TestDemeanTransform:
    """Tests for the demean rolling transformation."""

    def test_demean_mve(self):
        """MVE test: hand-verified check of the demean transformation.

        Data: N=3 units, T=3 periods (2 pre, 1 post)
        - Unit 1 (treated): y=[5, 6, 9], post=[0, 0, 1]
        - Unit 2 (control): y=[3, 4, 5], post=[0, 0, 1]
        - Unit 3 (control): y=[7, 8, 6], post=[0, 0, 1]

        Manual expectations:
        - Unit 1: ȳ₁,pre = (5+6)/2 = 5.5
                   ydot₁ = [5-5.5, 6-5.5, 9-5.5] = [-0.5, 0.5, 3.5]
                   ydot_postavg₁ = 3.5
        - Unit 2: ȳ₂,pre = (3+4)/2 = 3.5
                   ydot₂ = [-0.5, 0.5, 1.5]
                   ydot_postavg₂ = 1.5
        - Unit 3: ȳ₃,pre = (7+8)/2 = 7.5
                   ydot₃ = [-0.5, 0.5, -1.5]
                   ydot_postavg₃ = -1.5
        """
        # Load MVE test data
        data = pd.read_csv('tests/data/mve_demean.csv')
        
        # Add required columns (mimic validation.py output)
        data['d_'] = (data['d'] != 0).astype(int)
        data['post_'] = (data['post'] != 0).astype(int)
        data['tindex'] = data['year']  # already 1,2,3
        
        # Apply demean transformation
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
        
        # Check ydot values (tolerance < 1e-7, numerical layer L2)
        # Expected ydot for unit 1
        unit1_ydot_expected = np.array([-0.5, 0.5, 3.5])
        unit1_ydot_actual = data_transformed[data_transformed['id'] == 1]['ydot'].values
        assert np.allclose(unit1_ydot_actual, unit1_ydot_expected, atol=1e-7)
        
        # Expected ydot for unit 2
        unit2_ydot_expected = np.array([-0.5, 0.5, 1.5])
        unit2_ydot_actual = data_transformed[data_transformed['id'] == 2]['ydot'].values
        assert np.allclose(unit2_ydot_actual, unit2_ydot_expected, atol=1e-7)
        
        # Expected ydot for unit 3
        unit3_ydot_expected = np.array([-0.5, 0.5, -1.5])
        unit3_ydot_actual = data_transformed[data_transformed['id'] == 3]['ydot'].values
        assert np.allclose(unit3_ydot_actual, unit3_ydot_expected, atol=1e-7)
    
    def test_demean_pre_mean_precision(self):
        """Check that pre-period means are numerically close to zero (<1e-10, L1)."""
        data = pd.read_csv('tests/data/mve_demean.csv')
        data['d_'] = (data['d'] != 0).astype(int)
        data['post_'] = (data['post'] != 0).astype(int)
        data['tindex'] = data['year']
        
        data_transformed = apply_rolling_transform(
            data, 'y', 'id', 'tindex', 'post_', 'demean', 3
        )
        
        # Pre-period means of ydot should be close to zero (in-sample fit)
        for unit_id in [1, 2, 3]:
            unit_pre_ydot = data_transformed[
                (data_transformed['id'] == unit_id) & 
                (data_transformed['post_'] == 0)
            ]['ydot']
            pre_mean = unit_pre_ydot.mean()
            # After residualization, the pre-period mean should be essentially zero
            assert abs(pre_mean) < 1e-10
    
    def test_ydot_postavg_broadcast(self):
        """Check that ``ydot_postavg`` is correctly broadcast at the unit level.

        Key properties:
        - Within each unit, all rows share the same ``ydot_postavg`` (constant over time).
        - The value equals the unit-specific post-treatment mean of ``ydot``.
        """
        data = pd.read_csv('tests/data/mve_demean.csv')
        data['d_'] = (data['d'] != 0).astype(int)
        data['post_'] = (data['post'] != 0).astype(int)
        data['tindex'] = data['year']
        
        data_transformed = apply_rolling_transform(
            data, 'y', 'id', 'tindex', 'post_', 'demean', 3
        )
        
        # Unit 1: all rows should share ydot_postavg = 3.5
        unit1_postavg = data_transformed[data_transformed['id'] == 1]['ydot_postavg']
        assert len(unit1_postavg.unique()) == 1  # constant within unit
        assert abs(unit1_postavg.iloc[0] - 3.5) < 1e-7

        # Unit 2
        unit2_postavg = data_transformed[data_transformed['id'] == 2]['ydot_postavg']
        assert len(unit2_postavg.unique()) == 1
        assert abs(unit2_postavg.iloc[0] - 1.5) < 1e-7

        # Unit 3
        unit3_postavg = data_transformed[data_transformed['id'] == 3]['ydot_postavg']
        assert len(unit3_postavg.unique()) == 1
        assert abs(unit3_postavg.iloc[0] - (-1.5)) < 1e-7

    def test_firstpost_identification(self):
        """Test that firstpost flag correctly identifies first post-treatment observation per unit.

        The 'firstpost' flag marks the first post-treatment observation for each unit
        (tindex == tpost1), which is used for cross-sectional OLS regression in ATT
        estimation. This creates a cross-sectional sample with one observation per unit.

        Verifies:
        1. Exactly N rows are flagged (one per unit)
        2. Each unit has exactly one firstpost row
        3. The flagged row is at tindex == tpost1 for that unit
        """
        data = pd.read_csv('tests/data/mve_demean.csv')
        data['d_'] = (data['d'] != 0).astype(int)
        data['post_'] = (data['post'] != 0).astype(int)
        data['tindex'] = data['year']

        data_transformed = apply_rolling_transform(
            data, 'y', 'id', 'tindex', 'post_', 'demean', tpost1=3
        )

        # Verify exactly N=3 rows are flagged (one per unit)
        n_firstpost = data_transformed['firstpost'].sum()
        assert n_firstpost == 3, f"Expected 3 firstpost rows, got {n_firstpost}"

        # Verify each unit has exactly one firstpost row at tindex == tpost1
        for unit_id in [1, 2, 3]:
            unit_firstpost = data_transformed[
                (data_transformed['id'] == unit_id) &
                (data_transformed['firstpost'])
            ]
            assert len(unit_firstpost) == 1, \
                f"Unit {unit_id} should have exactly 1 firstpost row"

            # Verify it's at tpost1 (the first post-treatment period)
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
    """Tests for the detrend rolling transformation."""
    
    def test_detrend_mve(self):
        """MVE test: detrend on perfectly linear trend data.

        Data: N=3 units, T=5 periods (3 pre, 2 post)
        - Unit 1 (treated): y = 3 + 2t + 5·post
        - Unit 2 (control): y = 1 + t
        - Unit 3 (control): y = 4 + 1.5t

        Manual expectations:
        - Unit 1: α̂₁=3, β̂₁=2, ydot₁=[0,0,0,5,5], ydot_postavg₁=5
        - Unit 2: α̂₂=1, β̂₂=1, ydot₂=[0,0,0,0,0], ydot_postavg₂=0
        - Unit 3: α̂₃=4, β̂₃=1.5, ydot₃=[0,0,0,0,0], ydot_postavg₃=0
        - Expected ATT: 5 - 0 = 5
        """
        # Load MVE test data
        data = pd.read_csv('tests/data/mve_detrend.csv')
        
        # Add required columns (mimic validation.py output)
        data['d_'] = (data['d'] != 0).astype(int)
        data['post_'] = (data['post'] != 0).astype(int)
        # tindex column already exists with values 1,2,3,4,5
        
        # Apply detrend transformation
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
        
        # === Validate trend fit via pre-period residuals ===
        # For perfectly linear data, pre-period residuals should be ≈0 (<1e-12)
        for unit_id in [1, 2, 3]:
            unit_pre_ydot = data_transformed[
                (data_transformed['id'] == unit_id) & 
                (data_transformed['post_'] == 0)
            ]['ydot'].values
            # Perfect pre-period fit: residuals should be near zero
            assert np.allclose(unit_pre_ydot, 0, atol=1e-12), \
                f"Unit {unit_id} pre-period residuals not near zero"
        
        # === Validate post-period residuals against manual expectations ===
        # Unit 1: post y=[16,18], yhat=[11,13], ydot=[5,5]
        unit1_post_ydot = data_transformed[
            (data_transformed['id'] == 1) & 
            (data_transformed['post_'] == 1)
        ]['ydot'].values
        assert np.allclose(unit1_post_ydot, [5.0, 5.0], atol=1e-7)
        
        # Unit 2: post y=[5,6], yhat=[5,6], ydot=[0,0]
        unit2_post_ydot = data_transformed[
            (data_transformed['id'] == 2) & 
            (data_transformed['post_'] == 1)
        ]['ydot'].values
        assert np.allclose(unit2_post_ydot, [0.0, 0.0], atol=1e-7)
        
        # Unit 3: post y=[10,11.5], yhat=[10,11.5], ydot=[0,0]
        unit3_post_ydot = data_transformed[
            (data_transformed['id'] == 3) & 
            (data_transformed['post_'] == 1)
        ]['ydot'].values
        assert np.allclose(unit3_post_ydot, [0.0, 0.0], atol=1e-7)
        
        # === Validate ydot_postavg (post-period mean) ===
        # Unit 1: mean([5,5]) = 5
        unit1_postavg = data_transformed[data_transformed['id'] == 1]['ydot_postavg'].iloc[0]
        assert abs(unit1_postavg - 5.0) < 1e-7
        
        # Unit 2: mean([0,0]) = 0
        unit2_postavg = data_transformed[data_transformed['id'] == 2]['ydot_postavg'].iloc[0]
        assert abs(unit2_postavg - 0.0) < 1e-7
        
        # Unit 3: mean([0,0]) = 0
        unit3_postavg = data_transformed[data_transformed['id'] == 3]['ydot_postavg'].iloc[0]
        assert abs(unit3_postavg - 0.0) < 1e-7
    
    def test_detrend_error_T0_insufficient(self):
        """Boundary test B003: detrend should fail when T₀=1.

        Verification points:
        - When the number of pre-treatment periods T₀=1, rolling='detrend'
          must raise InsufficientPrePeriodsError.
        - The error message should contain the phrase
          "at least 2 pre-treatment periods".
        """
        from lwdid.exceptions import InsufficientPrePeriodsError
        
        # Construct data with T₀=1 (N=3, T=3; 1 pre, 2 post)
        data = pd.DataFrame({
            'id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
            'tindex': [1, 2, 3, 1, 2, 3, 1, 2, 3],
            'year': [1, 2, 3, 1, 2, 3, 1, 2, 3],
            'y': [5.0, 6.0, 7.0, 3.0, 4.0, 5.0, 7.0, 8.0, 9.0],
            'd_': [1, 1, 1, 0, 0, 0, 0, 0, 0],
            'post_': [0, 1, 1, 0, 1, 1, 0, 1, 1],
        })
        
        # Expect InsufficientPrePeriodsError
        with pytest.raises(
            InsufficientPrePeriodsError,
            match="rolling\\('detrend'\\) requires at least 2 pre-treatment periods"
        ):
            apply_rolling_transform(
                data, 'y', 'id', 'tindex', 'post_',
                rolling='detrend', tpost1=2
            )


class TestQuarterlyTransforms:
    """Tests for quarterly rolling transformations (demeanq and detrendq)."""
    
    def test_demeanq_mve(self):
        """Test demeanq transformation on seasonal quarterly data.

        Data structure:
        - N=2 units, T=9 periods (5 pre-periods + 4 post-periods)
        - Unit 1 (treated): seasonal pattern with treatment effect in post-period
        - Unit 2 (control): pure seasonal pattern, no treatment effect

        The demeanq transformation:
        1. Estimates unit-specific mean and quarterly effects from pre-period
        2. Removes these effects from all periods
        3. Pre-period residuals should be near zero (in-sample fit)
        4. Post-period residuals reflect treatment effect with seasonality removed

        Note: demeanq requires at least 5 pre-period observations to ensure
        sufficient degrees of freedom (df ≥ 1) after estimating intercept and
        3 quarterly dummy coefficients.
        """
        # Construct seasonal MVE data with 5 pre-periods (demeanq requires ≥5)
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

        # Apply demeanq transformation (tpost1=6 because first post-period is at t=6)
        data_transformed = apply_rolling_transform(
            data=data, y='y', ivar='id', tindex='tindex', post='post_',
            rolling='demeanq', tpost1=6, quarter='quarter'
        )

        # Validate pre-period residuals (in-sample fit)
        unit1_pre = data_transformed[
            (data_transformed['id'] == 1) & (data_transformed['post_'] == 0)
        ]
        assert np.allclose(unit1_pre['ydot'].values, 0, atol=1e-10), \
            "Pre-period residuals should be near zero (perfect in-sample fit)"

        # Validate post-period residuals are computed
        unit1_post = data_transformed[
            (data_transformed['id'] == 1) & (data_transformed['post_'] == 1)
        ]
        assert len(unit1_post) == 4, \
            f"Expected 4 post-period observations, got {len(unit1_post)}"
        assert unit1_post['ydot'].notna().all(), \
            "All post-period ydot values should be non-null"
        
        # Check ydot_postavg construction
        assert 'ydot_postavg' in data_transformed.columns
        assert 'firstpost' in data_transformed.columns
        
        # For Unit 1: ydot_postavg = mean([5,5,5,5]) = 5
        unit1_postavg = data_transformed[data_transformed['id']==1]['ydot_postavg'].iloc[0]
        assert abs(unit1_postavg - 5.0) < 1e-7
    
    def test_detrendq_mve(self):
        """Test detrendq transformation on data with linear trend and seasonality.

        Data structure:
        - N=2 units, T=9 periods (5 pre-periods + 4 post-periods)
        - Unit 1: y = 5 + 0.5*t + seasonal effects
        - Unit 2: y = 3 + 0.3*t + seasonal effects

        The detrendq transformation:
        1. Estimates unit-specific linear trend and quarterly effects from pre-period
        2. Removes these effects from all periods
        3. Pre-period residuals should be small (good fit)

        Note: detrendq requires at least 5 pre-period observations to ensure
        sufficient degrees of freedom after estimating intercept, trend, and
        3 quarterly dummy coefficients.
        """
        # Construct data with trend + seasonality (5 pre-periods for detrendq)
        data = pd.DataFrame({
            'id': [1,1,1,1,1,1,1,1,1, 2,2,2,2,2,2,2,2,2],
            'year': [1,1,1,1,1,2,2,2,2, 1,1,1,1,1,2,2,2,2],
            'quarter': [1,2,3,4,1,2,3,4,1, 1,2,3,4,1,2,3,4,1],
            'tindex': [1,2,3,4,5,6,7,8,9, 1,2,3,4,5,6,7,8,9],
            # Unit 1: base=5, trend=0.5*t, seasonal=[0, 1, 0.5, 1.5]
            'y': [5.0, 6.5, 6.5, 7.5, 7.5, 9.0, 9.0, 10.0, 10.5,
                  # Unit 2: base=3, trend=0.3*t, seasonal=[0, 1, 0.5, 1.5]
                  3.0, 4.3, 4.4, 5.2, 4.5, 5.8, 5.9, 6.7, 7.0],
            'd_': [1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0],
            'post_': [0,0,0,0,0,1,1,1,1, 0,0,0,0,0,1,1,1,1],
        })
        
        # Apply detrendq transformation (tpost1=6 because first post-period is at t=6)
        data_transformed = apply_rolling_transform(
            data=data, y='y', ivar='id', tindex='tindex', post='post_',
            rolling='detrendq', tpost1=6, quarter='quarter'
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

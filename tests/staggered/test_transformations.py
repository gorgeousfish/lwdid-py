"""
Unit Tests for Staggered Data Transformations

Tests for transform_staggered_demean() and transform_staggered_detrend()
based on Story E1-S1 acceptance criteria.
"""

import warnings
import pytest
import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from lwdid.staggered.transformations import (
    transform_staggered_demean,
    transform_staggered_detrend,
    get_cohorts,
    get_valid_periods_for_cohort,
)


# =============================================================================
# Test 1: Basic Demeaning Transformation
# =============================================================================

class TestStaggeredDemeanBasic:
    """Basic tests for demeaning transformation."""
    
    def test_staggered_demean_basic(self):
        """Test basic demeaning transformation.
        
        Test data structure:
        - T=4 periods, cohorts = {3, 4, ∞}
        - unit 1: cohort 3, y = [10, 12, 14, 20]
        - unit 2: cohort 4, y = [15, 16, 17, 18]
        - unit 3: never treated, y = [5, 6, 7, 8]
        """
        data = pd.DataFrame({
            'id': [1,1,1,1, 2,2,2,2, 3,3,3,3],
            'year': [1,2,3,4, 1,2,3,4, 1,2,3,4],
            'y': [10,12,14,20, 15,16,17,18, 5,6,7,8],
            'gvar': [3,3,3,3, 4,4,4,4, 0,0,0,0]
        })
        
        result = transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        
        # === Cohort g=3 verification ===
        # pre-treatment periods: t=1,2 (g-1=2 periods)
        # unit 1 (cohort 3): pre_mean_g3 = (10+12)/2 = 11
        assert np.isclose(result.loc[(result['id']==1) & (result['year']==3), 'ydot_g3_r3'].iloc[0], 3)  # 14-11
        assert np.isclose(result.loc[(result['id']==1) & (result['year']==4), 'ydot_g3_r4'].iloc[0], 9)  # 20-11
        
        # unit 3 (never treated, also needs transformation for cohort 3): pre_mean_g3 = (5+6)/2 = 5.5
        assert np.isclose(result.loc[(result['id']==3) & (result['year']==3), 'ydot_g3_r3'].iloc[0], 1.5)  # 7-5.5
        assert np.isclose(result.loc[(result['id']==3) & (result['year']==4), 'ydot_g3_r4'].iloc[0], 2.5)  # 8-5.5
        
        # === Cohort g=4 verification ===
        # pre-treatment periods: t=1,2,3 (g-1=3 periods)
        # unit 2 (cohort 4): pre_mean_g4 = (15+16+17)/3 = 16
        assert np.isclose(result.loc[(result['id']==2) & (result['year']==4), 'ydot_g4_r4'].iloc[0], 2)  # 18-16
        
        # unit 3 (never treated): pre_mean_g4 = (5+6+7)/3 = 6
        assert np.isclose(result.loc[(result['id']==3) & (result['year']==4), 'ydot_g4_r4'].iloc[0], 2)  # 8-6
    
    def test_pre_treatment_mean_is_fixed(self):
        """⚠️ CRITICAL TEST: Verify pre-treatment mean is fixed across all post periods.
        
        This is the most common implementation mistake!
        """
        data = pd.DataFrame({
            'id': [1]*6,
            'year': [1,2,3,4,5,6],
            'y': [10, 12, 15, 20, 25, 30],  # pre={10,12,15}, post={20,25,30}
            'gvar': [4]*6
        })
        
        result = transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        
        # Pre-treatment mean should be (10+12+15)/3 = 12.333...
        expected_pre_mean = (10 + 12 + 15) / 3
        
        # Reverse-engineer pre-treatment mean from transformation results
        ydot_r4 = result.loc[result['year']==4, 'ydot_g4_r4'].iloc[0]  # 20 - pre_mean
        ydot_r5 = result.loc[result['year']==5, 'ydot_g4_r5'].iloc[0]  # 25 - pre_mean
        ydot_r6 = result.loc[result['year']==6, 'ydot_g4_r6'].iloc[0]  # 30 - pre_mean
        
        pre_mean_from_r4 = 20 - ydot_r4
        pre_mean_from_r5 = 25 - ydot_r5
        pre_mean_from_r6 = 30 - ydot_r6
        
        # CRITICAL: All three periods must derive the same pre_mean
        assert np.isclose(pre_mean_from_r4, expected_pre_mean)
        assert np.isclose(pre_mean_from_r5, expected_pre_mean)
        assert np.isclose(pre_mean_from_r6, expected_pre_mean)
        assert np.isclose(pre_mean_from_r4, pre_mean_from_r5)
        assert np.isclose(pre_mean_from_r5, pre_mean_from_r6)
    
    def test_all_units_get_transformed(self):
        """Verify ALL units (including controls) get transformation."""
        data = pd.DataFrame({
            'id': [1,1,1, 2,2,2, 3,3,3],
            'year': [1,2,3, 1,2,3, 1,2,3],
            'y': [10,12,20, 5,6,8, 15,16,17],
            'gvar': [3,3,3, 0,0,0, 0,0,0]  # unit 1: treated, unit 2,3: control
        })
        
        result = transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        
        # All units should have ydot_g3_r3 values at period 3
        assert 'ydot_g3_r3' in result.columns
        
        period_3_data = result[result['year'] == 3]
        assert not period_3_data['ydot_g3_r3'].isna().any()
        
        # Verify specific values
        # unit 1 (treated): pre_mean = (10+12)/2 = 11, ydot = 20-11 = 9
        # unit 2 (control): pre_mean = (5+6)/2 = 5.5, ydot = 8-5.5 = 2.5
        # unit 3 (control): pre_mean = (15+16)/2 = 15.5, ydot = 17-15.5 = 1.5
        assert np.isclose(period_3_data.loc[period_3_data['id']==1, 'ydot_g3_r3'].iloc[0], 9)
        assert np.isclose(period_3_data.loc[period_3_data['id']==2, 'ydot_g3_r3'].iloc[0], 2.5)
        assert np.isclose(period_3_data.loc[period_3_data['id']==3, 'ydot_g3_r3'].iloc[0], 1.5)


# =============================================================================
# Test 2: Detrending Transformation
# =============================================================================

class TestStaggeredDetrendBasic:
    """Basic tests for detrending transformation."""
    
    def test_staggered_detrend_basic(self):
        """Test basic detrending transformation with clear linear trend."""
        # Data with explicit linear trend
        data = pd.DataFrame({
            'id': [1]*6 + [2]*6,
            'year': [1,2,3,4,5,6]*2,
            'y': [12,14,16,28,32,36,  # unit 1: base=10, slope=2, effect=10 at t>=4
                  5,7,9,11,13,15],    # unit 2 (control): base=3, slope=2
            'gvar': [4]*6 + [0]*6
        })
        
        result = transform_staggered_detrend(data, 'y', 'id', 'year', 'gvar')
        
        # Unit 1's pre-treatment trend: Y = 10 + 2*t
        # At t=1,2,3: Y = 12, 14, 16 -> OLS gives A=10, B=2
        # Predicted at t=4: 10 + 2*4 = 18
        # Actual at t=4: 28
        # ycheck = 28 - 18 = 10 (treatment effect!)
        
        assert 'ycheck_g4_r4' in result.columns
        ycheck_unit1_r4 = result.loc[(result['id']==1) & (result['year']==4), 'ycheck_g4_r4'].iloc[0]
        assert np.isclose(ycheck_unit1_r4, 10, atol=0.1)
    
    def test_detrend_requires_two_pre_periods(self):
        """Test that detrending requires at least 2 pre-treatment periods."""
        # cohort=2 means only 1 pre-treatment period (t=1)
        data = pd.DataFrame({
            'id': [1,1,1],
            'year': [1,2,3],
            'y': [10,20,30],
            'gvar': [2,2,2]
        })
        
        with pytest.raises(ValueError, match="at least 2"):
            transform_staggered_detrend(data, 'y', 'id', 'year', 'gvar')
    
    def test_detrend_with_exactly_two_pre_periods(self):
        """Test detrending with exactly 2 pre-treatment periods (boundary case)."""
        # cohort=3, T_min=1, so pre_periods = {1, 2}, exactly 2 periods
        data = pd.DataFrame({
            'id': [1]*4 + [2]*4,
            'year': [1,2,3,4]*2,
            'y': [10,12,24,28,   # unit 1: trend=2, effect=10 at t>=3
                  5,7,9,11],     # unit 2: trend=2, no treatment
            'gvar': [3]*4 + [0]*4
        })
        
        # Should work without error
        result = transform_staggered_detrend(data, 'y', 'id', 'year', 'gvar')
        
        assert 'ycheck_g3_r3' in result.columns
        assert 'ycheck_g3_r4' in result.columns
        
        # Unit 1: OLS on t=1,2 gives Y = A + B*t
        # Y_1=10, Y_2=12 -> B = (12-10)/(2-1) = 2, A = 10 - 2*1 = 8
        # Predicted at t=3: 8 + 2*3 = 14
        # Actual at t=3: 24
        # ycheck = 24 - 14 = 10 (treatment effect)
        ycheck_unit1_r3 = result.loc[(result['id']==1) & (result['year']==3), 'ycheck_g3_r3'].iloc[0]
        assert np.isclose(ycheck_unit1_r3, 10, atol=0.1)


# =============================================================================
# Test 3: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_missing_values_handling(self):
        """Test missing value handling (partial pre-period missing)."""
        data = pd.DataFrame({
            'id': [1,1,1,1],
            'year': [1,2,3,4],
            'y': [10, np.nan, 14, 20],  # period 2 missing
            'gvar': [3,3,3,3]
        })
        
        result = transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        
        # pre-treatment periods: t=1,2, but t=2 is missing
        # Should use available data: pre_mean = 10 (only t=1)
        assert 'ydot_g3_r3' in result.columns
        # ydot_g3_r3 = 14 - 10 = 4
        assert np.isclose(result.loc[result['year']==3, 'ydot_g3_r3'].iloc[0], 4)
    
    def test_all_pre_periods_missing(self):
        """Test when all pre-periods are missing for a unit."""
        data = pd.DataFrame({
            'id': [1,1,1, 2,2,2],
            'year': [1,2,3, 1,2,3],
            'y': [np.nan, np.nan, 20, 10, 12, 14],  # unit 1: all pre missing
            'gvar': [3,3,3, 3,3,3]
        })
        
        result = transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        
        # unit 1: all pre periods missing -> transformation should be NaN
        unit1_ydot = result.loc[(result['id']==1) & (result['year']==3), 'ydot_g3_r3'].iloc[0]
        assert np.isnan(unit1_ydot)
        
        # unit 2: normal calculation (pre_mean = (10+12)/2 = 11, ydot = 14-11 = 3)
        unit2_ydot = result.loc[(result['id']==2) & (result['year']==3), 'ydot_g3_r3'].iloc[0]
        assert np.isclose(unit2_ydot, 3)
    
    def test_unbalanced_panel(self):
        """Test unbalanced panel support."""
        data = pd.DataFrame({
            'id': [1,1,1, 2,2],  # unit 2 only has 2 periods
            'year': [1,2,3, 2,3],
            'y': [10,12,20, 6,8],
            'gvar': [3,3,3, 0,0]
        })
        
        result = transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        
        # unit 1: normal processing
        assert np.isclose(result.loc[(result['id']==1) & (result['year']==3), 'ydot_g3_r3'].iloc[0], 9)
        
        # unit 2: only has period 2 pre-data, pre_mean = 6
        # ydot_g3_r3 = 8 - 6 = 2
        assert np.isclose(result.loc[(result['id']==2) & (result['year']==3), 'ydot_g3_r3'].iloc[0], 2)
    
    def test_never_treated_identification(self):
        """Test never treated unit identification."""
        data = pd.DataFrame({
            'id': [1,1, 2,2, 3,3, 4,4],
            'year': [1,2]*4,
            'y': [10,20]*4,
            'gvar': [0,0, np.inf,np.inf, np.nan,np.nan, 3,3]  # three NT representations
        })
        
        cohorts = get_cohorts(data, 'gvar', 'id')
        
        # Should only return cohort 3, excluding all never treated
        assert cohorts == [3]
    
    def test_cohort_at_or_before_t_min_raises_error(self):
        """Test that cohort <= T_min raises clear error."""
        # Case 1: cohort = T_min
        data = pd.DataFrame({
            'id': [1,1,1],
            'year': [1,2,3],  # T_min = 1
            'y': [10,20,30],
            'gvar': [1,1,1]   # cohort = 1 = T_min, no pre period
        })
        
        with pytest.raises(ValueError, match="no pre-treatment"):
            transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
    
    def test_t_min_dynamic_detection(self):
        """⚠️ CRITICAL TEST: Verify T_min is dynamically detected from data.
        
        T_min must NOT be hardcoded to 1!
        """
        data = pd.DataFrame({
            'id': [1,1,1,1,1, 2,2,2,2,2],
            'year': [2003,2004,2005,2006,2007]*2,
            'y': [10,12,20,22,24,  # unit 1: cohort 2005
                  5,6,8,9,10],     # unit 2: never treated
            'gvar': [2005]*5 + [0]*5
        })
        
        result = transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        
        # T_min = 2003, cohort = 2005
        # pre-treatment periods should be {2003, 2004}, cohort - T_min = 2 periods
        
        # unit 1: pre_mean = (10+12)/2 = 11 (only 2003 and 2004)
        # ydot_g2005_r2005 = 20 - 11 = 9
        assert np.isclose(
            result.loc[(result['id']==1) & (result['year']==2005), 'ydot_g2005_r2005'].iloc[0], 
            9.0
        )
        
        # unit 2: pre_mean = (5+6)/2 = 5.5
        # ydot_g2005_r2005 = 8 - 5.5 = 2.5
        assert np.isclose(
            result.loc[(result['id']==2) & (result['year']==2005), 'ydot_g2005_r2005'].iloc[0], 
            2.5
        )
    
    def test_float_cohort_column_naming(self):
        """⚠️ Test float cohort produces integer column names.
        
        Castle Law data has effyear as float (e.g., 2005.0).
        Column names should be 'ydot_g2005_r2005' not 'ydot_g2005.0_r2005'.
        """
        data = pd.DataFrame({
            'id': [1,1,1, 2,2,2],
            'year': [2003,2004,2005, 2003,2004,2005],
            'y': [10,12,20, 5,6,8],
            'gvar': [2005.0, 2005.0, 2005.0, 0.0, 0.0, 0.0]  # float format
        })
        
        result = transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        
        # Should have integer format column name
        assert 'ydot_g2005_r2005' in result.columns
        
        # Should NOT have float format
        assert 'ydot_g2005.0_r2005' not in result.columns
    
    def test_nan_as_never_treated(self):
        """Test NaN is automatically recognized as never treated."""
        data = pd.DataFrame({
            'id': [1,1,1, 2,2,2, 3,3,3],
            'year': [1,2,3, 1,2,3, 1,2,3],
            'y': [10,12,20, 5,6,8, 15,16,18],
            'gvar': [3,3,3, np.nan,np.nan,np.nan, np.nan,np.nan,np.nan]
        })
        
        # Should auto-recognize NaN as NT
        result = transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        
        cohorts = get_cohorts(data, 'gvar', 'id')
        assert cohorts == [3]
        
        # NT units should have transformation values
        nt_unit2 = result[(result['id']==2) & (result['year']==3)]
        assert not nt_unit2['ydot_g3_r3'].isna().all()
        
        # unit2 pre_mean = (5+6)/2 = 5.5, ydot = 8 - 5.5 = 2.5
        assert np.isclose(nt_unit2['ydot_g3_r3'].iloc[0], 2.5)
    
    def test_cohort_at_t_max(self):
        """⚠️ Test cohort exactly at T_max (only one post period)."""
        data = pd.DataFrame({
            'id': [1,1,1,1, 2,2,2,2],
            'year': [1,2,3,4, 1,2,3,4],
            'y': [10,12,14,50, 5,6,7,8],  # unit1: cohort 4, unit2: NT
            'gvar': [4,4,4,4, 0,0,0,0]
        })
        
        result = transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        
        # Should generate ydot_g4_r4 column
        assert 'ydot_g4_r4' in result.columns
        
        # Should NOT generate ydot_g4_r5 (T_max=4)
        assert 'ydot_g4_r5' not in result.columns
        
        # unit1: pre_mean = (10+12+14)/3 = 12
        # ydot_g4_r4 = 50 - 12 = 38
        unit1_ydot = result[(result['id']==1) & (result['year']==4)]['ydot_g4_r4'].iloc[0]
        assert np.isclose(unit1_ydot, 38)
    
    def test_transform_column_sparsity(self):
        """⚠️ Test that transformation columns are sparse (only filled at correct period)."""
        data = pd.DataFrame({
            'id': [1,1,1,1, 2,2,2,2],
            'year': [1,2,3,4, 1,2,3,4],
            'y': [10,12,14,20, 5,6,7,8],
            'gvar': [3,3,3,3, 0,0,0,0]
        })
        
        result = transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        
        # ydot_g3_r3 should only have values at year==3
        assert 'ydot_g3_r3' in result.columns
        
        # year==3 rows should have values
        year3_vals = result[result['year'] == 3]['ydot_g3_r3']
        assert not year3_vals.isna().all()
        assert len(year3_vals.dropna()) == 2  # 2 units at year==3
        
        # Other years should be NaN
        other_years = result[result['year'] != 3]['ydot_g3_r3']
        assert other_years.isna().all()
    
    def test_no_transform_in_pre_periods(self):
        """⚠️ Test that pre-periods have no transformation values."""
        data = pd.DataFrame({
            'id': [1,1,1,1,1, 2,2,2,2,2],
            'year': [1,2,3,4,5, 1,2,3,4,5],
            'y': [10,12,14,50,52, 5,6,7,8,9],
            'gvar': [4,4,4,4,4, 0,0,0,0,0]  # cohort 4: pre={1,2,3}, post={4,5}
        })
        
        result = transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        
        # Pre-period rows (year < 4) should be NaN in all ydot columns
        pre_rows = result[result['year'] < 4]
        
        ydot_cols = [c for c in result.columns if c.startswith('ydot_')]
        for col in ydot_cols:
            assert pre_rows[col].isna().all(), f"Pre-period rows should be NaN in {col}"
        
        # Post-period rows should have values
        post_rows = result[result['year'] >= 4]
        assert not post_rows['ydot_g4_r4'].isna().all()
    
    def test_cross_cohort_transformation(self):
        """⚠️ CRITICAL: Test cross-cohort transformation.
        
        Already-treated units also need transformation for later cohorts.
        """
        data = pd.DataFrame({
            'id': [1]*5 + [2]*5 + [3]*5,
            'year': [2003,2004,2005,2006,2007]*3,
            'y': [10,11,15,16,17,   # unit 1: cohort 2005 (already treated in 2005)
                  20,21,22,28,29,   # unit 2: cohort 2006
                  30,31,32,33,34],  # unit 3: never treated
            'gvar': [2005]*5 + [2006]*5 + [0]*5
        })
        
        result = transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        
        # Unit 1 (2005 cohort) should ALSO have cohort 2006 transformation
        # pre_mean_g2006 for Unit 1 = mean(Y_2003, Y_2004, Y_2005) = (10+11+15)/3 = 12
        # ydot_g2006_r2006 = Y_2006 - pre_mean = 16 - 12 = 4
        
        assert 'ydot_g2006_r2006' in result.columns
        
        unit1_2006 = result[(result['id']==1) & (result['year']==2006)]
        assert not unit1_2006['ydot_g2006_r2006'].isna().all()
        
        expected_pre_mean_g2006_u1 = (10 + 11 + 15) / 3  # = 12
        expected_ydot = 16 - expected_pre_mean_g2006_u1  # = 4
        actual_ydot = unit1_2006['ydot_g2006_r2006'].iloc[0]
        
        assert np.isclose(actual_ydot, expected_ydot, atol=1e-10)
    
    def test_same_unit_different_cohort_pre_means(self):
        """⚠️ CORE CONCEPT: Same unit has different pre_mean for different cohorts.
        
        For the same unit i, different cohorts g use different pre-treatment periods:
        - cohort g=3: pre(3) = {1, 2}, pre_mean = mean(Y_1, Y_2)
        - cohort g=4: pre(4) = {1, 2, 3}, pre_mean = mean(Y_1, Y_2, Y_3)
        """
        data = pd.DataFrame({
            'id': [1]*5 + [2]*5 + [3]*5,
            'year': [1,2,3,4,5]*3,
            'y': [10, 20, 30, 100, 110,   # unit 1: cohort 3
                  15, 25, 35, 45, 55,     # unit 2: cohort 4
                  5, 6, 7, 8, 9],         # unit 3: never treated
            'gvar': [3]*5 + [4]*5 + [0]*5
        })
        
        result = transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        
        # Unit 3 (NT) at year=4: different cohort definitions = different ydot values
        
        # For cohort g=3: pre(3) = {1, 2}
        # Unit 3's pre_mean_g3 = (5+6)/2 = 5.5
        # ydot_g3_r4 = 8 - 5.5 = 2.5
        
        # For cohort g=4: pre(4) = {1, 2, 3}
        # Unit 3's pre_mean_g4 = (5+6+7)/3 = 6
        # ydot_g4_r4 = 8 - 6 = 2
        
        unit3_year4 = result[(result['id']==3) & (result['year']==4)]
        
        ydot_g3_r4 = unit3_year4['ydot_g3_r4'].iloc[0]
        ydot_g4_r4 = unit3_year4['ydot_g4_r4'].iloc[0]
        
        # These should be DIFFERENT!
        assert not np.isclose(ydot_g3_r4, ydot_g4_r4), \
            f"Same unit should have different ydot for different cohorts! ydot_g3_r4={ydot_g3_r4}, ydot_g4_r4={ydot_g4_r4}"
        
        assert np.isclose(ydot_g3_r4, 2.5, atol=1e-10)
        assert np.isclose(ydot_g4_r4, 2.0, atol=1e-10)
    
    def test_all_eventually_treated_warning(self):
        """Test warning when no never-treated units exist."""
        data = pd.DataFrame({
            'id': [1,1,1, 2,2,2],
            'year': [1,2,3, 1,2,3],
            'y': [10,12,20, 15,16,25],
            'gvar': [2,2,2, 3,3,3]  # No NT (gvar=0)
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
            
            # Should have warning
            assert len(w) > 0
            assert "never-treated" in str(w[0].message).lower()
        
        # Transformation should still work
        assert 'ydot_g2_r2' in result.columns
        assert 'ydot_g3_r3' in result.columns


# =============================================================================
# Test 4: Detrending Numerical Verification
# =============================================================================

class TestDetrendingNumericalVerification:
    """Detailed numerical verification for detrending."""
    
    def test_detrending_numerical_verification(self):
        """Detailed numerical verification for detrending transformation.
        
        Test data:
        Unit 1: Y_t = 8 + 2*t (perfect linear trend)
                Pre: t=1,2,3 -> Y = 10, 12, 14
                Post: t=4,5 -> Y (observed) = 26, 32 (with treatment effect 10)
        """
        data = pd.DataFrame({
            'id': [1]*5 + [2]*5,
            'year': [1,2,3,4,5]*2,
            'y': [10, 12, 14, 26, 32,   # unit 1: trend=2, intercept=8, effect=10 at t>=4
                  5, 7, 9, 11, 13],     # unit 2 (control): trend=2, intercept=3
            'gvar': [4]*5 + [0]*5
        })
        
        result = transform_staggered_detrend(data, 'y', 'id', 'year', 'gvar')
        
        # Unit 1 verification
        # OLS on pre-treatment: Y_t = A + B*t
        # (10,1), (12,2), (14,3) -> B=2, A=8
        # Predicted t=4: 8 + 2*4 = 16
        # ycheck = 26 - 16 = 10
        ycheck_u1_r4 = result.loc[(result['id']==1) & (result['year']==4), 'ycheck_g4_r4'].iloc[0]
        assert np.isclose(ycheck_u1_r4, 10, atol=1e-6)
        
        # Predicted t=5: 8 + 2*5 = 18
        # ycheck = 32 - 18 = 14 (effect growing over time)
        ycheck_u1_r5 = result.loc[(result['id']==1) & (result['year']==5), 'ycheck_g4_r5'].iloc[0]
        assert np.isclose(ycheck_u1_r5, 14, atol=1e-6)
        
        # Unit 2 (control) verification
        # OLS: (5,1), (7,2), (9,3) -> B=2, A=3
        # Predicted t=4: 3 + 2*4 = 11
        # Actual t=4: 11
        # ycheck = 11 - 11 = 0 (no treatment effect)
        ycheck_u2_r4 = result.loc[(result['id']==2) & (result['year']==4), 'ycheck_g4_r4'].iloc[0]
        assert np.isclose(ycheck_u2_r4, 0, atol=1e-6)


# =============================================================================
# Test 5: Stata Comparison
# =============================================================================

class TestStataComparison:
    """Tests for Stata implementation comparison."""
    
    def test_stata_lag_logic_understanding(self):
        """Verify Stata Lag operation understanding.
        
        Stata: bysort id: gen y_44 = y - (L1.y + L2.y + L3.y)/3 if f04
        Key insight: At any calendar time, the lag offsets are designed
        to always get the same pre-treatment periods {1,2,3}.
        """
        data = pd.DataFrame({
            'id': [1]*6,
            'year': [1, 2, 3, 4, 5, 6],
            'y': [10, 20, 30, 100, 110, 120],  # pre={10,20,30}, post={100,110,120}
            'gvar': [4]*6
        })
        
        result = transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        
        # Pre-treatment mean: (10+20+30)/3 = 20
        expected_pre_mean = 20.0
        
        # All post periods should use the same pre_mean
        assert np.isclose(result.loc[result['year']==4, 'ydot_g4_r4'].iloc[0], 100 - 20)  # 80
        assert np.isclose(result.loc[result['year']==5, 'ydot_g4_r5'].iloc[0], 110 - 20)  # 90
        assert np.isclose(result.loc[result['year']==6, 'ydot_g4_r6'].iloc[0], 120 - 20)  # 100
    
    def test_stata_comparison_synthetic(self):
        """Synthetic data comparison with Stata logic."""
        data = pd.DataFrame({
            'id': [1,1,1,1,1,1,  2,2,2,2,2,2,  3,3,3,3,3,3,  4,4,4,4,4,4],
            'year': [1,2,3,4,5,6]*4,
            'y': [10,12,14,50,52,54,   # unit 1: cohort 4
                  20,22,24,26,60,62,   # unit 2: cohort 5
                  30,32,34,36,38,70,   # unit 3: cohort 6
                  40,42,44,46,48,50],  # unit 4: never treated
            'gvar': [4]*6 + [5]*6 + [6]*6 + [0]*6
        })
        
        result = transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        
        # === Cohort g=4 verification ===
        # unit 1: pre_mean_g4 = (10+12+14)/3 = 12
        assert np.isclose(result.loc[(result['id']==1) & (result['year']==4), 'ydot_g4_r4'].iloc[0], 50 - 12)  # 38
        assert np.isclose(result.loc[(result['id']==1) & (result['year']==5), 'ydot_g4_r5'].iloc[0], 52 - 12)  # 40
        assert np.isclose(result.loc[(result['id']==1) & (result['year']==6), 'ydot_g4_r6'].iloc[0], 54 - 12)  # 42
        
        # unit 4 (NT): pre_mean_g4 = (40+42+44)/3 = 42
        assert np.isclose(result.loc[(result['id']==4) & (result['year']==4), 'ydot_g4_r4'].iloc[0], 46 - 42)  # 4
        
        # === Cohort g=5 verification ===
        # unit 2: pre_mean_g5 = (20+22+24+26)/4 = 23
        assert np.isclose(result.loc[(result['id']==2) & (result['year']==5), 'ydot_g5_r5'].iloc[0], 60 - 23)  # 37
        
        # === Cohort g=6 verification ===
        # unit 3: pre_mean_g6 = (30+32+34+36+38)/5 = 34
        assert np.isclose(result.loc[(result['id']==3) & (result['year']==6), 'ydot_g6_r6'].iloc[0], 70 - 34)  # 36


# =============================================================================
# Test 6: Multiple Cohorts
# =============================================================================

class TestMultipleCohorts:
    """Tests for multiple cohort scenarios."""
    
    def test_multiple_cohorts(self):
        """Test multiple cohort scenario (castle law structure)."""
        data = pd.DataFrame({
            'id': [1]*5 + [2]*5 + [3]*5 + [4]*5,
            'year': [2003,2004,2005,2006,2007]*4,
            'y': [10,11,15,16,17,  # cohort 2005
                  20,21,22,28,29,  # cohort 2006
                  30,31,32,33,40,  # cohort 2007
                  40,41,42,43,44], # never treated
            'gvar': [2005]*5 + [2006]*5 + [2007]*5 + [0]*5
        })
        
        result = transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
        
        # Verify expected columns exist
        expected_cols = ['ydot_g2005_r2005', 'ydot_g2005_r2006', 'ydot_g2005_r2007',
                         'ydot_g2006_r2006', 'ydot_g2006_r2007',
                         'ydot_g2007_r2007']
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"


# =============================================================================
# Test 7: Helper Functions
# =============================================================================

class TestHelperFunctions:
    """Tests for helper functions."""
    
    def test_get_cohorts(self):
        """Test get_cohorts function."""
        data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'gvar': [2005, 2006, 0, np.nan, np.inf]
        })
        
        cohorts = get_cohorts(data, 'gvar', 'id')
        assert cohorts == [2005, 2006]
    
    def test_get_valid_periods_for_cohort(self):
        """Test get_valid_periods_for_cohort function."""
        periods = get_valid_periods_for_cohort(2005, 2010)
        assert periods == [2005, 2006, 2007, 2008, 2009, 2010]
        
        periods = get_valid_periods_for_cohort(4, 6)
        assert periods == [4, 5, 6]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

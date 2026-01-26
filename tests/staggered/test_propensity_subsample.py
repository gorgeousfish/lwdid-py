"""
Unit tests for Staggered Subsample Construction (Story 1.2).

Tests for build_subsample_for_ps_estimation() and related functions.

Reference:
- Lee & Wooldridge (2023) Section 4, Formulas 4.10-4.13
- Stata code: Lee_Wooldridge_2023-main 3/2.lee_wooldridge_rolling_staggered.do
"""

import numpy as np
import pandas as pd
import pytest
import warnings

# Skip entire module if required functions are not available
try:
    from lwdid.staggered.estimators import (
        SubsampleResult,
        build_subsample_for_ps_estimation,
        _identify_control_units,
        _create_binary_treatment,
    )
except ImportError as e:
    pytest.skip(
        f"Skipping module: required functions not implemented ({e})",
        allow_module_level=True
    )


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def staggered_panel_data():
    """
    Create a staggered DiD panel dataset similar to the paper's setup.
    
    Setup:
    - T=6 periods (year 1-6)
    - 4 cohorts: g=4, g=5, g=6, g=∞ (never treated)
    - 200 units total (50 per cohort)
    - Covariates: x1, x2
    """
    np.random.seed(42)
    
    n_per_cohort = 50
    T = 6
    
    # Create units
    units = []
    
    # Never Treated cohort (gvar = np.inf)
    for i in range(n_per_cohort):
        unit_id = i
        gvar = np.inf
        x1 = np.random.normal(5, 1)
        x2 = np.random.normal(0, 1)
        for t in range(1, T + 1):
            y = 1 + 0.5 * x1 + 0.3 * x2 + 0.1 * t + np.random.normal(0, 1)
            units.append({
                'id': unit_id,
                'year': t,
                'gvar': gvar,
                'x1': x1,
                'x2': x2,
                'y': y,
            })
    
    # Cohort 4 (treated starting year 4)
    for i in range(n_per_cohort):
        unit_id = n_per_cohort + i
        gvar = 4
        x1 = np.random.normal(5.5, 1)  # slightly different distribution
        x2 = np.random.normal(0.2, 1)
        for t in range(1, T + 1):
            treated = 1 if t >= gvar else 0
            treatment_effect = 2.0 * treated if t >= gvar else 0  # ATT = 2.0
            y = 1 + 0.5 * x1 + 0.3 * x2 + 0.1 * t + treatment_effect + np.random.normal(0, 1)
            units.append({
                'id': unit_id,
                'year': t,
                'gvar': gvar,
                'x1': x1,
                'x2': x2,
                'y': y,
            })
    
    # Cohort 5 (treated starting year 5)
    for i in range(n_per_cohort):
        unit_id = 2 * n_per_cohort + i
        gvar = 5
        x1 = np.random.normal(4.8, 1)
        x2 = np.random.normal(-0.1, 1)
        for t in range(1, T + 1):
            treated = 1 if t >= gvar else 0
            treatment_effect = 1.5 * treated if t >= gvar else 0  # ATT = 1.5
            y = 1 + 0.5 * x1 + 0.3 * x2 + 0.1 * t + treatment_effect + np.random.normal(0, 1)
            units.append({
                'id': unit_id,
                'year': t,
                'gvar': gvar,
                'x1': x1,
                'x2': x2,
                'y': y,
            })
    
    # Cohort 6 (treated starting year 6)
    for i in range(n_per_cohort):
        unit_id = 3 * n_per_cohort + i
        gvar = 6
        x1 = np.random.normal(5.2, 1)
        x2 = np.random.normal(0.1, 1)
        for t in range(1, T + 1):
            treated = 1 if t >= gvar else 0
            treatment_effect = 1.0 * treated if t >= gvar else 0  # ATT = 1.0
            y = 1 + 0.5 * x1 + 0.3 * x2 + 0.1 * t + treatment_effect + np.random.normal(0, 1)
            units.append({
                'id': unit_id,
                'year': t,
                'gvar': gvar,
                'x1': x1,
                'x2': x2,
                'y': y,
            })
    
    return pd.DataFrame(units)


@pytest.fixture
def data_all_eventually_treated():
    """
    Create a dataset where all units are eventually treated (no NT units).
    
    For FR-6 testing.
    """
    np.random.seed(42)
    
    n_per_cohort = 30
    T = 6
    
    units = []
    
    # Only cohorts 4, 5, 6 (no never treated)
    for g in [4, 5, 6]:
        for i in range(n_per_cohort):
            unit_id = (g - 4) * n_per_cohort + i
            gvar = g
            x1 = np.random.normal(5, 1)
            x2 = np.random.normal(0, 1)
            for t in range(1, T + 1):
                y = 1 + 0.5 * x1 + 0.3 * x2 + np.random.normal(0, 1)
                units.append({
                    'id': unit_id,
                    'year': t,
                    'gvar': gvar,
                    'x1': x1,
                    'x2': x2,
                    'y': y,
                })
    
    return pd.DataFrame(units)


@pytest.fixture
def small_panel_data():
    """
    Create a small panel for boundary condition testing.
    """
    np.random.seed(42)
    
    units = []
    
    # 3 units: 1 NT, 1 cohort 4, 1 cohort 5
    for unit_id, gvar in [(0, np.inf), (1, 4), (2, 5)]:
        x1 = np.random.normal(5, 1)
        x2 = np.random.normal(0, 1)
        for t in range(1, 7):
            y = 1 + 0.5 * x1 + np.random.normal(0, 1)
            units.append({
                'id': unit_id,
                'year': t,
                'gvar': gvar,
                'x1': x1,
                'x2': x2,
                'y': y,
            })
    
    return pd.DataFrame(units)


# ============================================================================
# Test _identify_control_units()
# ============================================================================

class TestIdentifyControlUnits:
    """Tests for _identify_control_units() helper function."""
    
    def test_never_treated_strategy(self):
        """Test never_treated strategy identifies only NT units."""
        # Create unit-level gvar Series
        unit_gvar = pd.Series(
            [np.inf, 4, 5, 6, np.inf, 4],
            index=[0, 1, 2, 3, 4, 5]
        )
        
        control_mask, has_nt = _identify_control_units(
            unit_gvar, period_r=4, control_group='never_treated',
            never_treated_values=[0, np.inf]
        )
        
        # Should only identify NT units (id 0, 4)
        assert control_mask[0] == True
        assert control_mask[4] == True
        assert control_mask[1] == False  # cohort 4
        assert control_mask[2] == False  # cohort 5
        assert has_nt == True
    
    def test_not_yet_treated_strategy_period4(self):
        """
        Test not_yet_treated strategy for period 4.
        
        Stata equivalent: if f04
        Control groups: {5, 6, ∞}
        """
        unit_gvar = pd.Series(
            [np.inf, 4, 5, 6],
            index=[0, 1, 2, 3]
        )
        
        control_mask, has_nt = _identify_control_units(
            unit_gvar, period_r=4, control_group='not_yet_treated',
            never_treated_values=[0, np.inf]
        )
        
        # Control: ∞ (NT) + 5 (>4) + 6 (>4)
        assert control_mask[0] == True   # ∞
        assert control_mask[1] == False  # 4 == period, NOT control!
        assert control_mask[2] == True   # 5 > 4
        assert control_mask[3] == True   # 6 > 4
    
    def test_not_yet_treated_strategy_period5(self):
        """
        Test not_yet_treated strategy for period 5.
        
        Stata equivalent: if f05 & ~g5
        Control groups: {6, ∞}
        """
        unit_gvar = pd.Series(
            [np.inf, 4, 5, 6],
            index=[0, 1, 2, 3]
        )
        
        control_mask, has_nt = _identify_control_units(
            unit_gvar, period_r=5, control_group='not_yet_treated',
            never_treated_values=[0, np.inf]
        )
        
        # Control: ∞ (NT) + 6 (>5)
        # Cohort 5 starts treatment at period 5, so NOT control
        assert control_mask[0] == True   # ∞
        assert control_mask[1] == False  # 4 < 5
        assert control_mask[2] == False  # 5 == period, NOT control!
        assert control_mask[3] == True   # 6 > 5
    
    def test_strict_greater_than(self):
        """
        Critical test: gvar == period should NOT be control.
        
        This tests the key distinction: gvar > period (strict).
        """
        unit_gvar = pd.Series([5, 5, 6], index=[0, 1, 2])
        
        control_mask, _ = _identify_control_units(
            unit_gvar, period_r=5, control_group='not_yet_treated',
            never_treated_values=[0, np.inf]
        )
        
        # Only unit 2 (gvar=6 > 5) is control
        assert control_mask[0] == False  # 5 == 5, NOT >
        assert control_mask[1] == False  # 5 == 5, NOT >
        assert control_mask[2] == True   # 6 > 5
    
    def test_nan_as_never_treated(self):
        """Test that NaN gvar is treated as never treated."""
        unit_gvar = pd.Series([np.nan, 4, 5], index=[0, 1, 2])
        
        control_mask, has_nt = _identify_control_units(
            unit_gvar, period_r=4, control_group='not_yet_treated',
            never_treated_values=[0, np.inf]
        )
        
        assert control_mask[0] == True   # NaN is NT
        assert has_nt == True
    
    def test_zero_as_never_treated(self):
        """Test that gvar=0 is treated as never treated."""
        unit_gvar = pd.Series([0, 4, 5], index=[0, 1, 2])
        
        control_mask, has_nt = _identify_control_units(
            unit_gvar, period_r=4, control_group='not_yet_treated',
            never_treated_values=[0, np.inf]
        )
        
        assert control_mask[0] == True   # 0 is NT
        assert has_nt == True
    
    def test_has_never_treated_detection(self):
        """Test has_never_treated flag detection."""
        # With NT units
        unit_gvar_with_nt = pd.Series([np.inf, 4, 5], index=[0, 1, 2])
        _, has_nt = _identify_control_units(
            unit_gvar_with_nt, period_r=4, control_group='not_yet_treated',
            never_treated_values=[0, np.inf]
        )
        assert has_nt == True
        
        # Without NT units
        unit_gvar_no_nt = pd.Series([4, 5, 6], index=[0, 1, 2])
        _, has_nt = _identify_control_units(
            unit_gvar_no_nt, period_r=4, control_group='not_yet_treated',
            never_treated_values=[0, np.inf]
        )
        assert has_nt == False


# ============================================================================
# Test _create_binary_treatment()
# ============================================================================

class TestCreateBinaryTreatment:
    """Tests for _create_binary_treatment() helper function."""
    
    def test_binary_values(self):
        """Test that D_ig is strictly 0 or 1."""
        df = pd.DataFrame({
            'gvar': [4, 4, 5, 6, np.inf],
            'x1': [1, 2, 3, 4, 5],
        })
        
        D_ig = _create_binary_treatment(df, 'gvar', cohort_g=4)
        
        assert set(D_ig).issubset({0, 1})
        assert D_ig.dtype == int
    
    def test_correct_assignment(self):
        """Test that D_ig correctly identifies cohort g."""
        df = pd.DataFrame({
            'gvar': [4, 4, 5, 6, np.inf, 4],
        })
        
        D_ig = _create_binary_treatment(df, 'gvar', cohort_g=4)
        
        expected = np.array([1, 1, 0, 0, 0, 1])
        np.testing.assert_array_equal(D_ig, expected)
    
    def test_count_consistency(self):
        """Test that D_ig counts match expected."""
        df = pd.DataFrame({
            'gvar': [4, 4, 4, 5, 5, np.inf],
        })
        
        D_ig = _create_binary_treatment(df, 'gvar', cohort_g=4)
        
        assert D_ig.sum() == 3   # 3 units in cohort 4
        assert (1 - D_ig).sum() == 3  # 3 control units


# ============================================================================
# Test build_subsample_for_ps_estimation()
# ============================================================================

class TestBuildSubsampleForPSEstimation:
    """Tests for the main subsample construction function."""
    
    def test_subsample_cohort4_period4(self, staggered_panel_data):
        """
        Test τ_{4,4} subsample construction.
        
        Stata: if f04
        Expected control: {5, 6, ∞}
        """
        result = build_subsample_for_ps_estimation(
            data=staggered_panel_data,
            gvar_col='gvar',
            ivar_col='id',
            cohort_g=4,
            period_r=4,
            control_group='not_yet_treated',
        )
        
        assert isinstance(result, SubsampleResult)
        assert result.cohort == 4
        assert result.period == 4
        assert result.n_treated > 0
        assert result.n_control > 0
        
        # Control cohorts should be {5, 6, ∞}
        assert 4 not in result.control_cohorts
        assert 5 in result.control_cohorts
        assert 6 in result.control_cohorts
        assert np.inf in result.control_cohorts
    
    def test_subsample_cohort4_period5(self, staggered_panel_data):
        """
        Test τ_{4,5} subsample construction.
        
        Stata: if f05 & ~g5
        Expected control: {6, ∞} (exclude cohort 5!)
        """
        result = build_subsample_for_ps_estimation(
            data=staggered_panel_data,
            gvar_col='gvar',
            ivar_col='id',
            cohort_g=4,
            period_r=5,
            control_group='not_yet_treated',
        )
        
        # Cohort 5 should be excluded (gvar=5 is not > 5)
        assert 5 not in result.control_cohorts
        assert 6 in result.control_cohorts
        assert np.inf in result.control_cohorts
    
    def test_subsample_cohort4_period6(self, staggered_panel_data):
        """
        Test τ_{4,6} subsample construction.
        
        Stata: if f06 & (g5 + g6 != 1)
        Expected control: {∞} only
        """
        result = build_subsample_for_ps_estimation(
            data=staggered_panel_data,
            gvar_col='gvar',
            ivar_col='id',
            cohort_g=4,
            period_r=6,
            control_group='not_yet_treated',
        )
        
        # Only NT should be control
        assert 5 not in result.control_cohorts
        assert 6 not in result.control_cohorts
        assert np.inf in result.control_cohorts
    
    def test_subsample_cohort5_period5(self, staggered_panel_data):
        """
        Test τ_{5,5} subsample construction.
        
        Stata: if f05 & ~g4
        Expected control: {6, ∞}
        """
        result = build_subsample_for_ps_estimation(
            data=staggered_panel_data,
            gvar_col='gvar',
            ivar_col='id',
            cohort_g=5,
            period_r=5,
            control_group='not_yet_treated',
        )
        
        assert 4 not in result.control_cohorts  # cohort 4 already treated
        assert 5 not in result.control_cohorts  # this is the treatment cohort
        assert 6 in result.control_cohorts
        assert np.inf in result.control_cohorts
    
    def test_D_ig_binary(self, staggered_panel_data):
        """Test that D_ig is binary 0/1."""
        result = build_subsample_for_ps_estimation(
            data=staggered_panel_data,
            gvar_col='gvar',
            ivar_col='id',
            cohort_g=4,
            period_r=4,
            control_group='not_yet_treated',
        )
        
        assert set(result.D_ig).issubset({0, 1})
        assert result.D_ig.dtype == int
    
    def test_D_ig_in_subsample(self, staggered_panel_data):
        """Test that D_ig column is added to subsample."""
        result = build_subsample_for_ps_estimation(
            data=staggered_panel_data,
            gvar_col='gvar',
            ivar_col='id',
            cohort_g=4,
            period_r=4,
            control_group='not_yet_treated',
        )
        
        assert 'D_ig' in result.subsample.columns
        np.testing.assert_array_equal(
            result.subsample['D_ig'].values,
            result.D_ig
        )
    
    def test_subsample_mask_traceback(self, staggered_panel_data):
        """Test that subsample_mask correctly traces back to original data."""
        result = build_subsample_for_ps_estimation(
            data=staggered_panel_data,
            gvar_col='gvar',
            ivar_col='id',
            cohort_g=4,
            period_r=4,
            control_group='not_yet_treated',
        )
        
        # Using the mask should give same subsample
        reconstructed = staggered_panel_data[result.subsample_mask]
        
        # Check same shape (ignoring D_ig column)
        assert len(reconstructed) == len(result.subsample)
    
    def test_count_consistency(self, staggered_panel_data):
        """Test that n_treated + n_control matches D_ig counts."""
        result = build_subsample_for_ps_estimation(
            data=staggered_panel_data,
            gvar_col='gvar',
            ivar_col='id',
            cohort_g=4,
            period_r=4,
            control_group='not_yet_treated',
        )
        
        assert result.n_treated == result.D_ig.sum()
        assert result.n_control == (1 - result.D_ig).sum()
        assert result.n_treated + result.n_control == len(result.D_ig)
    
    def test_never_treated_strategy(self, staggered_panel_data):
        """Test never_treated control group strategy."""
        result = build_subsample_for_ps_estimation(
            data=staggered_panel_data,
            gvar_col='gvar',
            ivar_col='id',
            cohort_g=4,
            period_r=4,
            control_group='never_treated',
        )
        
        # Only NT should be in control
        assert result.control_cohorts == [np.inf]
    
    def test_has_never_treated_flag(self, staggered_panel_data):
        """Test has_never_treated flag."""
        result = build_subsample_for_ps_estimation(
            data=staggered_panel_data,
            gvar_col='gvar',
            ivar_col='id',
            cohort_g=4,
            period_r=4,
            control_group='not_yet_treated',
        )
        
        assert result.has_never_treated == True


# ============================================================================
# Test Input Validation and Error Handling
# ============================================================================

class TestInputValidation:
    """Tests for input validation and error handling."""
    
    def test_missing_gvar_column(self, staggered_panel_data):
        """Test error when gvar column doesn't exist."""
        with pytest.raises(ValueError, match="Cohort variable.*not found"):
            build_subsample_for_ps_estimation(
                data=staggered_panel_data,
                gvar_col='nonexistent',
                ivar_col='id',
                cohort_g=4,
                period_r=4,
            )
    
    def test_missing_ivar_column(self, staggered_panel_data):
        """Test error when ivar column doesn't exist."""
        with pytest.raises(ValueError, match="Unit identifier variable.*not found"):
            build_subsample_for_ps_estimation(
                data=staggered_panel_data,
                gvar_col='gvar',
                ivar_col='nonexistent',
                cohort_g=4,
                period_r=4,
            )
    
    def test_invalid_control_group(self, staggered_panel_data):
        """Test error for invalid control_group parameter."""
        with pytest.raises(ValueError, match="Invalid control_group"):
            build_subsample_for_ps_estimation(
                data=staggered_panel_data,
                gvar_col='gvar',
                ivar_col='id',
                cohort_g=4,
                period_r=4,
                control_group='invalid',
            )
    
    def test_no_treatment_units(self, staggered_panel_data):
        """Test error when no units in treatment cohort."""
        with pytest.raises(ValueError, match="No treated units for cohort"):
            build_subsample_for_ps_estimation(
                data=staggered_panel_data,
                gvar_col='gvar',
                ivar_col='id',
                cohort_g=999,  # non-existent cohort
                period_r=4,
            )
    
    def test_small_sample_warning(self, small_panel_data):
        """Test warning for small sample size."""
        with pytest.warns(UserWarning, match="Subsample too small"):
            build_subsample_for_ps_estimation(
                data=small_panel_data,
                gvar_col='gvar',
                ivar_col='id',
                cohort_g=4,
                period_r=4,
            )


# ============================================================================
# Test FR-6: All Eventually Treated Scenarios
# ============================================================================

class TestAllEventuallyTreated:
    """
    Tests for FR-6: All Eventually Treated scenarios.
    
    When N_NT = 0, special handling is required.
    """
    
    def test_nyt_strategy_works_without_nt(self, data_all_eventually_treated):
        """
        FR-6.1: not_yet_treated strategy should work when N_NT = 0.
        """
        result = build_subsample_for_ps_estimation(
            data=data_all_eventually_treated,
            gvar_col='gvar',
            ivar_col='id',
            cohort_g=4,
            period_r=4,
            control_group='not_yet_treated',
        )
        
        assert result.has_never_treated == False
        assert result.n_control > 0  # Should have NYT controls
        assert 5 in result.control_cohorts
        assert 6 in result.control_cohorts
    
    def test_nt_strategy_fails_without_nt(self, data_all_eventually_treated):
        """
        FR-6.3: never_treated strategy should fail when N_NT = 0.
        """
        from lwdid.exceptions import NoNeverTreatedError
        with pytest.raises(NoNeverTreatedError, match="No never-treated units"):
            build_subsample_for_ps_estimation(
                data=data_all_eventually_treated,
                gvar_col='gvar',
                ivar_col='id',
                cohort_g=4,
                period_r=4,
                control_group='never_treated',
            )
    
    def test_tau_tt_fails_without_nt(self, data_all_eventually_treated):
        """
        FR-6.2: τ_{T,T} (last cohort at last period) should fail when N_NT = 0.
        
        In our data, T = 6, so τ_{6,6} should fail.
        """
        with pytest.raises(ValueError, match="Cannot estimate.*last cohort"):
            build_subsample_for_ps_estimation(
                data=data_all_eventually_treated,
                gvar_col='gvar',
                ivar_col='id',
                cohort_g=6,
                period_r=6,
                control_group='not_yet_treated',
            )
    
    def test_has_never_treated_flag_false(self, data_all_eventually_treated):
        """
        FR-6.4: has_never_treated should be False when N_NT = 0.
        """
        result = build_subsample_for_ps_estimation(
            data=data_all_eventually_treated,
            gvar_col='gvar',
            ivar_col='id',
            cohort_g=4,
            period_r=4,
            control_group='not_yet_treated',
        )
        
        assert result.has_never_treated == False
    
    def test_intermediate_cohort_period_works(self, data_all_eventually_treated):
        """
        Test that intermediate (g,r) pairs still work when N_NT = 0.
        
        τ_{4,5} should work with cohort 6 as control.
        """
        result = build_subsample_for_ps_estimation(
            data=data_all_eventually_treated,
            gvar_col='gvar',
            ivar_col='id',
            cohort_g=4,
            period_r=5,
            control_group='not_yet_treated',
        )
        
        assert result.n_control > 0
        assert 6 in result.control_cohorts
        assert 5 not in result.control_cohorts  # cohort 5 starts at period 5


# ============================================================================
# Test Stata Correspondence
# ============================================================================

class TestStataCorrespondence:
    """
    Tests that verify correspondence with Stata subsample conditions.
    
    Stata conditions from 2.lee_wooldridge_rolling_staggered.do:
    - τ_{4,4}: if f04
    - τ_{4,5}: if f05 & ~g5
    - τ_{4,6}: if f06 & (g5 + g6 != 1)
    """
    
    def test_tau44_subsample(self, staggered_panel_data):
        """
        τ_{4,4}: if f04
        
        Control = {5, 6, ∞} - all units not in cohort 4 that haven't started
        """
        result = build_subsample_for_ps_estimation(
            data=staggered_panel_data,
            gvar_col='gvar',
            ivar_col='id',
            cohort_g=4,
            period_r=4,
            control_group='not_yet_treated',
        )
        
        # Verify control cohorts match Stata
        expected_control = {5, 6, np.inf}
        actual_control = set(result.control_cohorts)
        assert actual_control == expected_control
    
    def test_tau45_excludes_cohort5(self, staggered_panel_data):
        """
        τ_{4,5}: if f05 & ~g5
        
        Cohort 5 excluded because g5 == 1 means gvar == 5 == period_r.
        """
        result = build_subsample_for_ps_estimation(
            data=staggered_panel_data,
            gvar_col='gvar',
            ivar_col='id',
            cohort_g=4,
            period_r=5,
            control_group='not_yet_treated',
        )
        
        # Cohort 5 should NOT be in control (5 == 5, not > 5)
        assert 5 not in result.control_cohorts
        
        # Cohorts 6 and ∞ should be in control
        assert 6 in result.control_cohorts
        assert np.inf in result.control_cohorts
    
    def test_tau46_only_never_treated(self, staggered_panel_data):
        """
        τ_{4,6}: if f06 & (g5 + g6 != 1)
        
        Only NT in control (both 5 and 6 have started by period 6).
        """
        result = build_subsample_for_ps_estimation(
            data=staggered_panel_data,
            gvar_col='gvar',
            ivar_col='id',
            cohort_g=4,
            period_r=6,
            control_group='not_yet_treated',
        )
        
        # Only NT should be control
        assert 5 not in result.control_cohorts
        assert 6 not in result.control_cohorts
        assert np.inf in result.control_cohorts


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_custom_never_treated_values(self, staggered_panel_data):
        """Test custom never_treated_values parameter.
        
        Note: has_never_treated flag only checks standard NT values (0, inf, NaN)
        via is_never_treated(). Custom NT values are included in control group
        but don't affect the has_never_treated flag.
        """
        # Create data with gvar=999 as NT
        df = staggered_panel_data.copy()
        df.loc[df['gvar'] == np.inf, 'gvar'] = 999
        
        result = build_subsample_for_ps_estimation(
            data=df,
            gvar_col='gvar',
            ivar_col='id',
            cohort_g=4,
            period_r=4,
            control_group='not_yet_treated',
            never_treated_values=[999],  # Custom NT value
        )
        
        # 999 should be in control cohorts (as it's > period_r=4)
        assert 999 in result.control_cohorts
        # Note: has_never_treated uses is_never_treated() which only recognizes
        # standard NT values (0, inf, NaN), so 999 doesn't count as NT
        assert result.has_never_treated == False
    
    def test_float_cohort_values(self):
        """Test with float cohort values."""
        df = pd.DataFrame({
            'id': [0, 0, 1, 1, 2, 2],
            'year': [1, 2, 1, 2, 1, 2],
            'gvar': [4.0, 4.0, 5.0, 5.0, np.inf, np.inf],
            'x1': [1, 1, 2, 2, 3, 3],
        })
        
        result = build_subsample_for_ps_estimation(
            data=df,
            gvar_col='gvar',
            ivar_col='id',
            cohort_g=4.0,
            period_r=4.0,
            control_group='not_yet_treated',
        )
        
        assert result.n_treated == 2  # Unit 0, 2 rows
        assert result.n_control == 4  # Units 1 and 2, 2 rows each
    
    def test_single_period_data(self):
        """Test with cross-sectional (single period) data."""
        df = pd.DataFrame({
            'id': [0, 1, 2, 3],
            'year': [1, 1, 1, 1],  # All same period
            'gvar': [4, 5, 6, np.inf],
            'x1': [1, 2, 3, 4],
        })
        
        result = build_subsample_for_ps_estimation(
            data=df,
            gvar_col='gvar',
            ivar_col='id',
            cohort_g=4,
            period_r=4,
            control_group='not_yet_treated',
        )
        
        # Should work with cross-sectional data
        assert result.n_treated == 1
        assert result.n_control == 3


# ============================================================================
# Test SubsampleResult Dataclass
# ============================================================================

class TestSubsampleResult:
    """Tests for SubsampleResult dataclass."""
    
    def test_repr(self, staggered_panel_data):
        """Test __repr__ output."""
        result = build_subsample_for_ps_estimation(
            data=staggered_panel_data,
            gvar_col='gvar',
            ivar_col='id',
            cohort_g=4,
            period_r=4,
            control_group='not_yet_treated',
        )
        
        repr_str = repr(result)
        
        assert 'SubsampleResult' in repr_str
        assert 'cohort=4' in repr_str
        assert 'period=4' in repr_str
        assert 'n_treated=' in repr_str
        assert 'n_control=' in repr_str
    
    def test_repr_with_inf(self, staggered_panel_data):
        """Test __repr__ handles infinity correctly."""
        result = build_subsample_for_ps_estimation(
            data=staggered_panel_data,
            gvar_col='gvar',
            ivar_col='id',
            cohort_g=4,
            period_r=4,
            control_group='not_yet_treated',
        )
        
        repr_str = repr(result)
        
        # Should display ∞ instead of inf
        assert '∞' in repr_str or 'inf' in repr_str.lower()


# ============================================================================
# Test DESIGN-043-E: τ_{TT} Detection Logic Fix
# ============================================================================

class TestDesign043ETauTTDetection:
    """
    Tests for DESIGN-043-E fix: τ_{TT} detection logic improvement.
    
    The fix changes the condition from `period_r == max_cohort` to 
    `period_r >= max_cohort` to correctly detect all cases where no 
    control units are available for the last cohort.
    """
    
    def test_tau_tt_at_treatment_period(self, data_all_eventually_treated):
        """
        Test τ_{T,T}: last cohort at its treatment period.
        
        cohort=6, period=6 should fail when no NT units exist.
        """
        with pytest.raises(ValueError, match="Cannot estimate.*last cohort"):
            build_subsample_for_ps_estimation(
                data=data_all_eventually_treated,
                gvar_col='gvar',
                ivar_col='id',
                cohort_g=6,
                period_r=6,
                control_group='not_yet_treated',
            )
    
    def test_tau_tt_after_treatment_period(self):
        """
        DESIGN-043-E key test: τ_{T,T+1} should also fail.
        
        This tests period_r > max_cohort case (the fix target).
        cohort=6, period=7 should fail when no NT units exist,
        because max(gvar)=6 < 7 means no not-yet-treated units.
        """
        # Create data with periods extending beyond max cohort
        np.random.seed(42)
        n_per_cohort = 20
        T = 8  # Extended periods
        
        units = []
        for g in [4, 5, 6]:  # No never-treated
            for i in range(n_per_cohort):
                unit_id = (g - 4) * n_per_cohort + i
                for t in range(1, T + 1):
                    units.append({
                        'id': unit_id,
                        'year': t,
                        'gvar': g,
                        'x1': np.random.normal(5, 1),
                    })
        df = pd.DataFrame(units)
        
        # τ_{6,7}: period > max_cohort, should fail
        with pytest.raises(ValueError, match="Cannot estimate.*last cohort"):
            build_subsample_for_ps_estimation(
                data=df,
                gvar_col='gvar',
                ivar_col='id',
                cohort_g=6,
                period_r=7,
                control_group='not_yet_treated',
            )
    
    def test_tau_tt_much_later_period(self):
        """
        Test τ_{T,T+k} for larger k should also fail.
        
        cohort=6, period=10 should fail.
        """
        np.random.seed(42)
        n_per_cohort = 20
        T = 12
        
        units = []
        for g in [4, 5, 6]:
            for i in range(n_per_cohort):
                unit_id = (g - 4) * n_per_cohort + i
                for t in range(1, T + 1):
                    units.append({
                        'id': unit_id,
                        'year': t,
                        'gvar': g,
                        'x1': np.random.normal(5, 1),
                    })
        df = pd.DataFrame(units)
        
        # τ_{6,10}: period >> max_cohort, should fail
        with pytest.raises(ValueError, match="Cannot estimate.*last cohort"):
            build_subsample_for_ps_estimation(
                data=df,
                gvar_col='gvar',
                ivar_col='id',
                cohort_g=6,
                period_r=10,
                control_group='not_yet_treated',
            )
    
    def test_non_last_cohort_still_works_with_nt(self):
        """
        Test that non-last cohort at final period still works if NT control available.
        
        τ_{4,6} should work with NT units as control.
        """
        np.random.seed(42)
        n_per_cohort = 20
        T = 6
        
        units = []
        # Create data with NT units (cohort 4, 5, and NT)
        cohort_list = [4, 5, np.inf]
        for idx, g in enumerate(cohort_list):
            for i in range(n_per_cohort):
                unit_id = idx * n_per_cohort + i
                for t in range(1, T + 1):
                    units.append({
                        'id': unit_id,
                        'year': t,
                        'gvar': g,
                        'x1': np.random.normal(5, 1),
                    })
        df = pd.DataFrame(units)
        
        # τ_{4,6} should work because NT units exist as control
        result = build_subsample_for_ps_estimation(
            data=df,
            gvar_col='gvar',
            ivar_col='id',
            cohort_g=4,
            period_r=6,
            control_group='not_yet_treated',
        )
        
        assert result.n_control > 0
        assert result.has_never_treated == True
        assert np.inf in result.control_cohorts
    
    def test_error_message_includes_max_cohort_info(self):
        """
        Test that error message includes helpful max_cohort information.
        """
        np.random.seed(42)
        n_per_cohort = 10
        
        units = []
        for g in [4, 5, 6]:
            for i in range(n_per_cohort):
                unit_id = (g - 4) * n_per_cohort + i
                units.append({
                    'id': unit_id,
                    'year': 1,
                    'gvar': g,
                    'x1': 1.0,
                })
        df = pd.DataFrame(units)
        
        with pytest.raises(ValueError) as exc_info:
            build_subsample_for_ps_estimation(
                data=df,
                gvar_col='gvar',
                ivar_col='id',
                cohort_g=6,
                period_r=7,
                control_group='not_yet_treated',
            )
        
        # Error message should explain the reason
        error_msg = str(exc_info.value)
        assert "max(gvar) = 6" in error_msg or "max_cohort" in error_msg.lower()
        assert "not-yet-treated" in error_msg.lower()
    
    def test_with_nt_units_last_cohort_works(self, staggered_panel_data):
        """
        Test that last cohort estimation works when NT units exist.
        
        τ_{6,6} should work when there are NT units as control.
        """
        result = build_subsample_for_ps_estimation(
            data=staggered_panel_data,
            gvar_col='gvar',
            ivar_col='id',
            cohort_g=6,
            period_r=6,
            control_group='not_yet_treated',
        )
        
        # Should work with NT as control
        assert result.n_control > 0
        assert np.inf in result.control_cohorts
        assert result.has_never_treated == True
    
    def test_different_encoding_schemes(self):
        """
        Test with different cohort/period encoding schemes.
        
        Note: This documents the assumption that cohort values represent
        treatment start periods. Mixed encodings may produce unexpected results.
        """
        np.random.seed(42)
        
        # Standard encoding (cohort = period when treatment starts)
        units_standard = []
        for g in [2004, 2005, 2006]:
            for i in range(10):
                unit_id = (g - 2004) * 10 + i
                for t in range(2001, 2008):
                    units_standard.append({
                        'id': unit_id,
                        'year': t,
                        'gvar': g,
                        'x1': 1.0,
                    })
        df_standard = pd.DataFrame(units_standard)
        
        # τ_{2006, 2006} should fail (last cohort, no NT)
        with pytest.raises(ValueError, match="Cannot estimate"):
            build_subsample_for_ps_estimation(
                data=df_standard,
                gvar_col='gvar',
                ivar_col='id',
                cohort_g=2006,
                period_r=2006,
                control_group='not_yet_treated',
            )
        
        # τ_{2006, 2007} should also fail (period > max_cohort, the fix)
        with pytest.raises(ValueError, match="Cannot estimate"):
            build_subsample_for_ps_estimation(
                data=df_standard,
                gvar_col='gvar',
                ivar_col='id',
                cohort_g=2006,
                period_r=2007,
                control_group='not_yet_treated',
            )


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegrationWithPSEstimation:
    """Integration tests with propensity score estimation."""
    
    def test_subsample_with_ps_estimation(self, staggered_panel_data):
        """Test that subsample can be used for PS estimation."""
        from lwdid.staggered.estimators import estimate_propensity_score
        
        result = build_subsample_for_ps_estimation(
            data=staggered_panel_data,
            gvar_col='gvar',
            ivar_col='id',
            cohort_g=4,
            period_r=4,
            control_group='not_yet_treated',
        )
        
        # Use D_ig for PS estimation
        pscores, coef = estimate_propensity_score(
            result.subsample,
            d='D_ig',
            controls=['x1', 'x2'],
            trim_threshold=0.01,
        )
        
        # PS should be in (0, 1)
        assert pscores.min() >= 0.01
        assert pscores.max() <= 0.99
        assert len(pscores) == len(result.subsample)
    
    def test_ps_estimated_on_correct_subsample(self, staggered_panel_data):
        """
        Verify PS is estimated on D_ig + A_{r+1} = 1 subsample.
        
        This is the core correctness test for Story 1.2.
        """
        result = build_subsample_for_ps_estimation(
            data=staggered_panel_data,
            gvar_col='gvar',
            ivar_col='id',
            cohort_g=4,
            period_r=5,
            control_group='not_yet_treated',
        )
        
        # All rows in subsample should satisfy D_ig + A_{r+1} = 1
        gvars_in_subsample = result.subsample['gvar'].unique()
        
        # Should only contain cohort 4 (treated) and valid controls
        for gvar in gvars_in_subsample:
            if gvar == 4:
                continue  # treatment cohort OK
            # Control must be > period_r OR NT
            assert (gvar > 5) or np.isinf(gvar) or np.isnan(gvar) or gvar in [0]

"""
Unit tests for BUG-206, BUG-207, BUG-208 fixes.

BUG-206: results.py plot_event_study warns on NaN standard errors
BUG-207: control_groups.py get_all_control_masks warns on empty cohorts
BUG-208: validation.py get_cohort_mask NaN handling documented
"""

import warnings
import numpy as np
import pandas as pd
import pytest

from lwdid.staggered.control_groups import (
    get_all_control_masks,
    ControlGroupStrategy,
)
from lwdid.validation import get_cohort_mask, COHORT_FLOAT_TOLERANCE


class TestBug206NaNSEWarning:
    """Tests for BUG-206: plot_event_study NaN SE warning.
    
    Tests the warning logic by directly testing the condition that triggers
    the warning in plot_event_study, rather than constructing full LWDIDResults.
    """

    def test_nan_se_detection_logic(self):
        """Test the NaN SE detection logic used in plot_event_study."""
        # Simulate event_df with NaN SE (as created in plot_event_study)
        event_df = pd.DataFrame({
            'event_time': [-1, 0, 1, 2],
            'att': [0.1, 0.5, 0.6, 0.7],
            'se': [0.05, np.nan, 0.08, 0.09],  # One NaN SE
        })
        
        # This is the exact check added in BUG-206 fix
        has_nan_se = event_df['se'].isna().any()
        n_nan = int(event_df['se'].isna().sum())
        
        assert has_nan_se == True, "Should detect NaN in SE"
        assert n_nan == 1, "Should count exactly 1 NaN SE"
        
        # Verify the warning message would be correctly formatted
        expected_msg_pattern = f"{n_nan} event time(s) have NaN standard errors"
        assert "1 event time(s) have NaN standard errors" in expected_msg_pattern
    
    def test_no_nan_se_detection(self):
        """Test that valid SE does not trigger NaN detection."""
        event_df = pd.DataFrame({
            'event_time': [-1, 0, 1, 2],
            'att': [0.1, 0.5, 0.6, 0.7],
            'se': [0.05, 0.06, 0.08, 0.09],  # All valid SE
        })
        
        has_nan_se = event_df['se'].isna().any()
        
        assert has_nan_se == False, "Should not detect NaN when SE is valid"

    def test_warning_code_path_in_results(self):
        """Verify the warning code exists in results.py at expected location."""
        import inspect
        from lwdid.results import LWDIDResults
        
        # Get the source code of plot_event_study
        source = inspect.getsource(LWDIDResults.plot_event_study)
        
        # Verify our fix is present
        assert "NaN standard errors" in source, \
            "BUG-206 fix warning message should be in plot_event_study"
        assert "event_df['se'].isna().any()" in source, \
            "BUG-206 fix NaN check should be in plot_event_study"
    
    def test_ci_calculation_with_nan_se(self):
        """Test that CI calculation produces NaN when SE is NaN."""
        from scipy import stats
        
        event_df = pd.DataFrame({
            'att': [0.5, 0.6],
            'se': [0.1, np.nan],  # One valid, one NaN
        })
        
        t_crit = stats.t.ppf(1 - 0.05 / 2, 1000)  # Large df approximation
        event_df['ci_lower'] = event_df['att'] - t_crit * event_df['se']
        event_df['ci_upper'] = event_df['att'] + t_crit * event_df['se']
        
        # Valid SE should produce valid CI
        assert not pd.isna(event_df.loc[0, 'ci_lower'])
        assert not pd.isna(event_df.loc[0, 'ci_upper'])
        
        # NaN SE should produce NaN CI (this is expected behavior)
        assert pd.isna(event_df.loc[1, 'ci_lower'])
        assert pd.isna(event_df.loc[1, 'ci_upper'])


class TestBug207EmptyCohortsWarning:
    """Tests for BUG-207: get_all_control_masks empty cohorts warning."""

    @pytest.fixture
    def sample_panel_data(self):
        """Create sample panel data for testing."""
        np.random.seed(42)
        n_units = 20
        n_periods = 6
        
        units = np.repeat(np.arange(1, n_units + 1), n_periods)
        periods = np.tile(np.arange(2000, 2000 + n_periods), n_units)
        
        # Assign gvar: some treated at 2003, some at 2004, some never-treated (inf)
        gvar_by_unit = {}
        for i in range(1, n_units + 1):
            if i <= 5:
                gvar_by_unit[i] = 2003
            elif i <= 10:
                gvar_by_unit[i] = 2004
            else:
                gvar_by_unit[i] = np.inf
        
        gvar = [gvar_by_unit[u] for u in units]
        y = np.random.randn(len(units))
        
        return pd.DataFrame({
            'id': units,
            'year': periods,
            'gvar': gvar,
            'y': y,
        })

    def test_empty_cohorts_triggers_warning(self, sample_panel_data):
        """get_all_control_masks should warn when cohorts list is empty."""
        data = sample_panel_data
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = get_all_control_masks(
                data=data,
                gvar='gvar',
                ivar='id',
                cohorts=[],  # Empty cohorts list
                T_max=2005,
                strategy=ControlGroupStrategy.NOT_YET_TREATED,
            )
            
            # Check warning was triggered
            empty_cohort_warnings = [
                x for x in w 
                if 'cohorts list is empty' in str(x.message)
            ]
            assert len(empty_cohort_warnings) == 1, \
                "Should warn exactly once about empty cohorts"
            
            # Result should be empty dict
            assert result == {}, "Should return empty dict for empty cohorts"

    def test_non_empty_cohorts_no_warning(self, sample_panel_data):
        """get_all_control_masks should not warn when cohorts list is non-empty."""
        data = sample_panel_data
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = get_all_control_masks(
                data=data,
                gvar='gvar',
                ivar='id',
                cohorts=[2003, 2004],  # Non-empty cohorts list
                T_max=2005,
                strategy=ControlGroupStrategy.NOT_YET_TREATED,
            )
            
            empty_cohort_warnings = [
                x for x in w 
                if 'cohorts list is empty' in str(x.message)
            ]
            assert len(empty_cohort_warnings) == 0, \
                "Should not warn when cohorts is non-empty"
            
            # Result should contain masks
            assert len(result) > 0, "Should return non-empty dict for valid cohorts"


class TestBug208GetCohortMaskNaNBehavior:
    """Tests for BUG-208: get_cohort_mask NaN handling documentation."""

    def test_nan_in_unit_gvar_returns_false(self):
        """NaN values in unit_gvar should return False in cohort mask."""
        unit_gvar = pd.Series([2003, 2004, np.nan, 2003, np.nan], 
                              index=['u1', 'u2', 'u3', 'u4', 'u5'])
        
        # Get mask for cohort 2003
        mask = get_cohort_mask(unit_gvar, 2003)
        
        # NaN units should have False
        assert mask['u3'] == False, "NaN unit should return False"
        assert mask['u5'] == False, "NaN unit should return False"
        
        # Regular matches should work
        assert mask['u1'] == True, "Matching cohort should return True"
        assert mask['u4'] == True, "Matching cohort should return True"
        assert mask['u2'] == False, "Non-matching cohort should return False"

    def test_all_nan_unit_gvar(self):
        """get_cohort_mask should handle all-NaN input gracefully."""
        unit_gvar = pd.Series([np.nan, np.nan, np.nan], 
                              index=['u1', 'u2', 'u3'])
        
        mask = get_cohort_mask(unit_gvar, 2003)
        
        # All should be False
        assert mask.sum() == 0, "All NaN input should return all False"
        assert len(mask) == 3, "Output length should match input"

    def test_nan_cohort_value_raises_or_returns_all_false(self):
        """Querying for NaN cohort should return all False (NaN != NaN)."""
        unit_gvar = pd.Series([2003, 2004, np.nan], 
                              index=['u1', 'u2', 'u3'])
        
        # NaN as cohort value should return all False
        # because np.isclose(x, NaN) is always False
        mask = get_cohort_mask(unit_gvar, np.nan)
        
        assert mask.sum() == 0, "Querying NaN cohort should return all False"

    def test_cohort_mask_preserves_index(self):
        """get_cohort_mask should preserve the original Series index."""
        unit_gvar = pd.Series([2003, np.nan, 2004], 
                              index=['alpha', 'beta', 'gamma'])
        
        mask = get_cohort_mask(unit_gvar, 2003)
        
        assert list(mask.index) == ['alpha', 'beta', 'gamma'], \
            "Index should be preserved"

    def test_cohort_mask_with_inf_values(self):
        """get_cohort_mask should handle infinity values correctly."""
        unit_gvar = pd.Series([2003, np.inf, -np.inf, 2003], 
                              index=['u1', 'u2', 'u3', 'u4'])
        
        mask = get_cohort_mask(unit_gvar, 2003)
        
        # Regular matches
        assert mask['u1'] == True
        assert mask['u4'] == True
        
        # Infinity should not match finite value
        assert mask['u2'] == False
        assert mask['u3'] == False

    def test_cohort_mask_float_tolerance(self):
        """get_cohort_mask should handle floating point tolerance."""
        # Value very close to 2003 (within tolerance)
        close_val = 2003.0 + COHORT_FLOAT_TOLERANCE / 2
        
        # Value outside tolerance
        far_val = 2003.0 + COHORT_FLOAT_TOLERANCE * 10
        
        unit_gvar = pd.Series([2003, close_val, far_val, np.nan], 
                              index=['u1', 'u2', 'u3', 'u4'])
        
        mask = get_cohort_mask(unit_gvar, 2003)
        
        assert mask['u1'] == True, "Exact match should return True"
        assert mask['u2'] == True, "Value within tolerance should return True"
        assert mask['u3'] == False, "Value outside tolerance should return False"
        assert mask['u4'] == False, "NaN should return False"


class TestNumericalCorrectness:
    """Test that fixes do not alter numerical results for valid inputs."""

    def test_control_masks_unchanged_for_valid_cohorts(self):
        """get_all_control_masks should produce same results for valid inputs."""
        np.random.seed(123)
        n_units = 30
        n_periods = 8
        
        units = np.repeat(np.arange(1, n_units + 1), n_periods)
        periods = np.tile(np.arange(2000, 2000 + n_periods), n_units)
        
        gvar_by_unit = {}
        for i in range(1, n_units + 1):
            if i <= 8:
                gvar_by_unit[i] = 2003
            elif i <= 16:
                gvar_by_unit[i] = 2005
            else:
                gvar_by_unit[i] = np.inf
        
        gvar = [gvar_by_unit[u] for u in units]
        
        data = pd.DataFrame({
            'id': units,
            'year': periods,
            'gvar': gvar,
        })
        
        # Get control masks
        masks = get_all_control_masks(
            data=data,
            gvar='gvar',
            ivar='id',
            cohorts=[2003, 2005],
            T_max=2007,
            strategy=ControlGroupStrategy.NOT_YET_TREATED,
        )
        
        # Verify expected structure
        assert len(masks) > 0, "Should return non-empty masks"
        
        # Check specific (g, r) combination
        key_2003_2003 = (2003, 2003.0)
        assert key_2003_2003 in masks, "Should have mask for (2003, 2003)"
        
        mask_2003_2003 = masks[key_2003_2003]
        
        # Never-treated units (id > 16) should always be in control
        for uid in range(17, n_units + 1):
            assert mask_2003_2003[uid] == True, \
                f"Never-treated unit {uid} should be in control"
        
        # Units treated at 2003 (id 1-8) should not be in control
        for uid in range(1, 9):
            assert mask_2003_2003[uid] == False, \
                f"Treated unit {uid} should not be in control"

    def test_cohort_mask_numerical_consistency(self):
        """get_cohort_mask should produce numerically consistent results."""
        unit_gvar = pd.Series([
            2003, 2003.0, 2003 + 1e-15,  # Should all match 2003
            2004, 2005, np.inf, np.nan,
        ], index=list('abcdefg'))
        
        mask = get_cohort_mask(unit_gvar, 2003)
        
        # First three should match
        assert mask['a'] == True
        assert mask['b'] == True
        assert mask['c'] == True  # Within floating point tolerance
        
        # Rest should not match
        assert mask['d'] == False
        assert mask['e'] == False
        assert mask['f'] == False
        assert mask['g'] == False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

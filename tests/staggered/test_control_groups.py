"""
Test suite for the control_groups module.

Validates control group selection logic for staggered Difference-in-Differences
estimation, including never-treated vs. not-yet-treated strategies, the critical
gvar == period exclusion rule, and sample construction per Equation (4.13).

References
----------
Lee, S. J. & Wooldridge, J. M. (2023). "Simple Difference-in-Differences
    Estimation in Fixed Effects Models." SSRN 5325686, Section 4.
Lee, S. J. & Wooldridge, J. M. (2025). "A Simple Transformation Approach
    to DiD Estimation for Panel Data." SSRN 4516518.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import time

from lwdid.staggered.control_groups import (
    ControlGroupStrategy,
    get_valid_control_units,
    get_all_control_masks,
    validate_control_group,
    identify_never_treated_units,
    has_never_treated_units,
    count_control_units_by_strategy,
)

TEST_DATA_DIR = Path(__file__).parent.parent.parent / 'data'
CASTLE_CSV_PATH = TEST_DATA_DIR / 'castle.csv'


class TestControlGroupStrategy:
    """Tests for ControlGroupStrategy enum values and string conversion."""

    def test_enum_values(self):
        """Verify enum members have the expected string values."""
        assert ControlGroupStrategy.NEVER_TREATED.value == 'never_treated'
        assert ControlGroupStrategy.NOT_YET_TREATED.value == 'not_yet_treated'
        assert ControlGroupStrategy.AUTO.value == 'auto'
    
    def test_from_string(self):
        """Verify enum can be constructed from its string value."""
        assert ControlGroupStrategy('never_treated') == ControlGroupStrategy.NEVER_TREATED


class TestGetValidControlUnits:
    """Tests for get_valid_control_units with never-treated and not-yet-treated strategies."""

    @pytest.fixture
    def simple_data(self):
        """Simple panel with cohorts {3, 4, 5} and two never-treated units."""
        return pd.DataFrame({'id': [1, 2, 3, 4, 5], 'gvar': [3, 4, 5, 0, 0]})
    
    def test_never_treated_strategy(self, simple_data):
        """Never-treated strategy should select only gvar=0 units as controls."""
        mask = get_valid_control_units(simple_data, 'gvar', 'id', cohort=3, period=4,
                                       strategy=ControlGroupStrategy.NEVER_TREATED)
        assert mask.loc[1] == False and mask.loc[4] == True and mask.sum() == 2
    
    def test_not_yet_treated_strategy_basic(self, simple_data):
        """Not-yet-treated strategy should exclude gvar <= period units from controls."""
        mask = get_valid_control_units(simple_data, 'gvar', 'id', cohort=3, period=4,
                                       strategy=ControlGroupStrategy.NOT_YET_TREATED)
        assert mask.loc[1] == False  # cohort 3 (treatment)
        assert mask.loc[2] == False  # gvar=4 == period=4, starts treatment!
        assert mask.loc[3] == True   # gvar=5 > 4
        assert mask.sum() == 3
    
    def test_gvar_equals_period_excluded(self):
        """Units with gvar == period must be excluded from the control group."""
        data = pd.DataFrame({'id': [1, 2, 3, 4, 5, 6], 'gvar': [4, 5, 6, 0, 0, 0]})
        mask = get_valid_control_units(data, 'gvar', 'id', cohort=4, period=6,
                                       strategy=ControlGroupStrategy.NOT_YET_TREATED)
        assert mask.loc[3] == False  # gvar=6 == period=6, starts NOW!
        assert mask.sum() == 3


class TestNeverTreatedIdentification:
    """Tests for never-treated unit recognition across different sentinel values."""

    def test_nan_always_recognized(self):
        """NaN gvar should always be recognized as never-treated."""
        data = pd.DataFrame({'id': [1, 2, 3], 'gvar': [4, np.nan, 5]})
        mask = get_valid_control_units(data, 'gvar', 'id', cohort=4, period=5,
                                       strategy=ControlGroupStrategy.NEVER_TREATED)
        assert mask.loc[2] == True
    
    def test_empty_list_never_treated_values(self):
        """Custom empty never_treated_values list should only recognize NaN as NT."""
        data = pd.DataFrame({'id': [1, 2, 3, 4], 'gvar': [4, 0, np.inf, np.nan]})
        mask = get_valid_control_units(data, 'gvar', 'id', cohort=4, period=5,
                                       strategy=ControlGroupStrategy.NEVER_TREATED,
                                       never_treated_values=[])
        assert mask.loc[4] == True and mask.sum() == 1


class TestEdgeCases:
    """Tests for edge cases in control group selection."""

    def test_no_control_units(self):
        """Empty control group when all non-treatment units have gvar <= period."""
        data = pd.DataFrame({'id': [1, 2, 3], 'gvar': [4, 4, 5]})
        mask = get_valid_control_units(data, 'gvar', 'id', cohort=4, period=6,
                                       strategy=ControlGroupStrategy.NOT_YET_TREATED)
        assert mask.sum() == 0
    
    def test_period_less_than_cohort(self):
        """Period < cohort should raise ValueError (pre-treatment query is invalid)."""
        data = pd.DataFrame({'id': [1, 2, 3], 'gvar': [5, 6, 0]})
        with pytest.raises(ValueError, match="period.*cohort"):
            get_valid_control_units(data, 'gvar', 'id', cohort=5, period=4,
                                    strategy=ControlGroupStrategy.NOT_YET_TREATED)


class TestValidateControlGroup:
    """Tests for validate_control_group post-selection checks."""

    def test_empty_control_group(self):
        """Empty control mask should be flagged as invalid."""
        mask = pd.Series([False, False, False], index=[1, 2, 3])
        is_valid, msg = validate_control_group(mask, cohort=3, period=4)
        assert is_valid == False
    
    def test_aggregate_requires_nt(self):
        """Overall aggregation without never-treated units should be invalid."""
        mask = pd.Series([True, True], index=[1, 2])
        is_valid, msg = validate_control_group(mask, cohort=3, period=4,
                                                aggregate_type='overall', has_never_treated=False)
        assert is_valid == False and "never-treated" in msg.lower()


class TestPanelDataHandling:
    """Tests for panel data structure handling in control group selection."""

    def test_returns_unit_level_mask(self):
        """Returned mask should be at the unit level, not the observation level."""
        data = pd.DataFrame({
            'id': [1,1,1,1, 2,2,2,2, 3,3,3,3],
            'year': [1,2,3,4, 1,2,3,4, 1,2,3,4],
            'gvar': [3,3,3,3, 4,4,4,4, 0,0,0,0]
        })
        mask = get_valid_control_units(data, 'gvar', 'id', cohort=3, period=4,
                                       strategy=ControlGroupStrategy.NOT_YET_TREATED)
        assert len(mask) == 3  # 3 units, not 12 rows


class TestCriticalGvarEqualsPeriodExclusion:
    """Critical tests: gvar == period exclusion logic."""
    
    def test_gvar_equals_period_basic(self):
        """Unit with gvar == period should be excluded from not-yet-treated controls."""
        data = pd.DataFrame({'id': [1, 2, 3, 4], 'gvar': [4, 5, 6, 0]})
        mask = get_valid_control_units(data, 'gvar', 'id', cohort=4, period=5,
                                       strategy=ControlGroupStrategy.NOT_YET_TREATED)
        assert mask.loc[2] == False  # gvar=5 == period=5, EXCLUDED!
        assert mask.loc[3] == True   # gvar=6 > 5, INCLUDED


class TestCastleLawEndToEnd:
    """End-to-end control group tests using the Castle Law dataset.
    
    Validates control group counts against known cohort structure:
    21 treated states across 5 cohorts (2005-2009), 29 never-treated.
    """

    @pytest.fixture
    def castle_data(self):
        """Load Castle Law CSV data, skip if unavailable."""
        if CASTLE_CSV_PATH.exists():
            return pd.read_csv(CASTLE_CSV_PATH)
        pytest.skip(f"Castle Law data file not found: {CASTLE_CSV_PATH}")
    
    def test_castle_data_structure(self, castle_data):
        """Verify Castle Law data has expected columns and 50 states."""
        assert 'sid' in castle_data.columns
        assert 'effyear' in castle_data.columns
        assert castle_data['sid'].nunique() == 50
    
    def test_castle_cohort_identification(self, castle_data):
        """Verify cohort 2006 has 13 states and 29 are never-treated."""
        unit_effyear = castle_data.groupby('sid')['effyear'].first()
        assert (unit_effyear == 2006).sum() == 13
        assert unit_effyear.isna().sum() == 29
    
    def test_castle_control_group_cohort_2006_period_2007(self, castle_data):
        """Cohort 2006 at period 2007 should have 32 control units."""
        mask = get_valid_control_units(castle_data, 'effyear', 'sid', 
                                       cohort=2006, period=2007,
                                       strategy=ControlGroupStrategy.NOT_YET_TREATED)
        assert mask.sum() == 32  # 2008(2) + 2009(1) + NT(29) = 32
    
    def test_castle_control_group_evolution(self, castle_data):
        """Control group size should shrink as later cohorts begin treatment."""
        expected = {2006: 36, 2007: 32, 2008: 30, 2009: 29, 2010: 29}
        for period, expected_count in expected.items():
            mask = get_valid_control_units(castle_data, 'effyear', 'sid',
                                           cohort=2006, period=period,
                                           strategy=ControlGroupStrategy.NOT_YET_TREATED)
            assert mask.sum() == expected_count, f"period={period}"
    
    def test_castle_performance_single_call(self, castle_data):
        """Single control group query should complete within 50 ms."""
        start = time.perf_counter()
        mask = get_valid_control_units(castle_data, 'effyear', 'sid',
                                       cohort=2006, period=2007,
                                       strategy=ControlGroupStrategy.NOT_YET_TREATED)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 50  # Relaxed threshold for system variability


class TestCountControlUnitsByStrategy:
    """Tests for the count_control_units_by_strategy summary utility."""

    def test_basic_count(self):
        """Verify correct counts for never-treated, not-yet-treated-only, and treatment cohort."""
        data = pd.DataFrame({'id': [1, 2, 3, 4, 5], 'gvar': [4, 5, 6, 0, 0]})
        counts = count_control_units_by_strategy(data, 'gvar', 'id', cohort=4, period=5)
        assert counts['never_treated'] == 2
        assert counts['not_yet_treated_only'] == 1  # cohort 6 only (cohort 5 starts at period 5!)
        assert counts['treatment_cohort'] == 1


class TestIdentifyNeverTreatedUnits:
    """Tests for the identify_never_treated_units function."""
    
    def test_identifies_nan_as_nt(self):
        """NaN gvar should be identified as never-treated."""
        data = pd.DataFrame({'id': [1, 2, 3], 'gvar': [4, np.nan, 5]})
        nt_mask = identify_never_treated_units(data, 'gvar', 'id')
        assert nt_mask.loc[2] == True
        assert nt_mask.sum() == 1
    
    def test_identifies_zero_and_inf_by_default(self):
        """Zero and infinity gvar should be identified as never-treated by default."""
        data = pd.DataFrame({'id': [1, 2, 3, 4], 'gvar': [4, 0, np.inf, np.nan]})
        nt_mask = identify_never_treated_units(data, 'gvar', 'id')
        assert nt_mask.sum() == 3  # 0, inf, nan
    
    def test_custom_nt_values(self):
        """Custom never_treated_values list should override default sentinel detection."""
        data = pd.DataFrame({'id': [1, 2, 3], 'gvar': [4, 9999, 0]})
        nt_mask = identify_never_treated_units(data, 'gvar', 'id', never_treated_values=[9999])
        assert nt_mask.loc[2] == True  # 9999 is NT
        assert nt_mask.loc[3] == False  # 0 is NOT NT with custom list


class TestHasNeverTreatedUnits:
    """Tests for the has_never_treated_units quick-check utility."""
    
    def test_returns_true_when_nt_exists(self):
        """Should return True when at least one never-treated unit exists."""
        data = pd.DataFrame({'id': [1, 2, 3], 'gvar': [4, 5, np.nan]})
        assert has_never_treated_units(data, 'gvar', 'id') == True
    
    def test_returns_false_when_no_nt(self):
        """Should return False when all units are eventually treated."""
        data = pd.DataFrame({'id': [1, 2, 3], 'gvar': [4, 5, 6]})
        assert has_never_treated_units(data, 'gvar', 'id') == False


class TestGvarTypeHandling:
    """Tests for gvar column type handling and type coercion."""
    
    def test_string_gvar_raises_typeerror(self):
        """String-typed gvar column should raise TypeError."""
        data = pd.DataFrame({'id': [1, 2], 'gvar': ['2005', 'never']})
        with pytest.raises(TypeError):
            get_valid_control_units(data, 'gvar', 'id', cohort=2005, period=2006,
                                    strategy=ControlGroupStrategy.NOT_YET_TREATED)
    
    def test_float_cohort_with_int_gvar(self):
        """Verifies compatibility between integer gvar and float cohort parameters."""
        data = pd.DataFrame({'id': [1, 2, 3], 'gvar': [4, 5, 0]})
        mask = get_valid_control_units(data, 'gvar', 'id', cohort=4.0, period=5.0,
                                       strategy=ControlGroupStrategy.NOT_YET_TREATED)
        assert mask.sum() == 1  # NT only


class TestSampleConstruction:
    """Validates sample construction logic per Equation (4.13).
    
    D_{ig} + A_{i,r+1} = 1
    Sample = treated | control (mutually exclusive)
    """
    
    def test_sample_excludes_other_treated_cohorts(self):
        """Verifies that other already-treated cohorts are excluded from the sample."""
        data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5, 6],
            'gvar': [3, 4, 5, 6, 0, 0]
        })
        unit_gvar = data.groupby('id')['gvar'].first()
        
        # cohort=3, period=5: exclude cohorts 4 and 5
        control_mask = get_valid_control_units(data, 'gvar', 'id', cohort=3, period=5,
                                               strategy=ControlGroupStrategy.NOT_YET_TREATED)
        treat_mask = (unit_gvar == 3)
        sample_mask = treat_mask | control_mask
        
        assert not sample_mask.loc[2], "Cohort 4 (gvar<period) should be excluded"
        assert not sample_mask.loc[3], "Cohort 5 (gvar==period) should be excluded"
        assert sample_mask.loc[4], "Cohort 6 (gvar>period) should be included"
    
    def test_treat_and_control_mutually_exclusive(self):
        """Verifies that treatment and control groups are mutually exclusive."""
        data = pd.DataFrame({'id': [1, 2, 3], 'gvar': [4, 5, 0]})
        unit_gvar = data.groupby('id')['gvar'].first()
        
        control_mask = get_valid_control_units(data, 'gvar', 'id', cohort=4, period=4,
                                               strategy=ControlGroupStrategy.NOT_YET_TREATED)
        treat_mask = (unit_gvar == 4)
        
        overlap = treat_mask & control_mask
        assert not overlap.any(), "Treatment and control should be mutually exclusive"


class TestAllEventuallyTreatedAggregation:
    """Tests aggregation constraints when all units are eventually treated."""
    
    def test_cohort_aggregate_requires_nt(self):
        """Cohort aggregation should fail without never-treated units."""
        data = pd.DataFrame({'id': [1, 2, 3], 'gvar': [4, 5, 6]})
        mask = get_valid_control_units(data, 'gvar', 'id', cohort=4, period=5,
                                       strategy=ControlGroupStrategy.NOT_YET_TREATED)
        
        is_valid, msg = validate_control_group(mask, cohort=4, period=5,
                                                aggregate_type='cohort', has_never_treated=False)
        assert is_valid == False
        assert 'never-treated' in msg.lower()
    
    def test_gr_specific_ok_without_nt(self):
        """(g,r)-specific effects are estimable without never-treated units."""
        data = pd.DataFrame({'id': [1, 2, 3], 'gvar': [4, 5, 6]})
        mask = get_valid_control_units(data, 'gvar', 'id', cohort=4, period=5,
                                       strategy=ControlGroupStrategy.NOT_YET_TREATED)
        
        is_valid, msg = validate_control_group(mask, cohort=4, period=5,
                                                aggregate_type=None, has_never_treated=False)
        assert is_valid == True


class TestAdditionalEdgeCases:
    """Additional edge case tests for control group selection."""
    
    def test_empty_dataframe_raises_error(self):
        """Empty DataFrame should raise ValueError."""
        data = pd.DataFrame(columns=['id', 'gvar'])
        with pytest.raises(ValueError):
            get_valid_control_units(data, 'gvar', 'id', cohort=3, period=4,
                                    strategy=ControlGroupStrategy.NOT_YET_TREATED)
    
    def test_missing_column_raises_keyerror(self):
        """Missing gvar column should raise KeyError."""
        data = pd.DataFrame({'id': [1, 2], 'other': [3, 4]})
        with pytest.raises(KeyError):
            get_valid_control_units(data, 'gvar', 'id', cohort=3, period=4,
                                    strategy=ControlGroupStrategy.NOT_YET_TREATED)
    
    def test_all_never_treated(self):
        """All units are never-treated."""
        data = pd.DataFrame({'id': [1, 2, 3], 'gvar': [0, np.nan, np.inf]})
        mask = get_valid_control_units(data, 'gvar', 'id', cohort=4, period=5,
                                       strategy=ControlGroupStrategy.NOT_YET_TREATED)
        assert mask.sum() == 3
    
    def test_single_unit(self):
        """Single unit in the dataset."""
        data = pd.DataFrame({'id': [1], 'gvar': [np.nan]})
        mask = get_valid_control_units(data, 'gvar', 'id', cohort=4, period=5,
                                       strategy=ControlGroupStrategy.NOT_YET_TREATED)
        assert len(mask) == 1
        assert mask.loc[1] == True
    
    def test_auto_strategy_fallback_to_nt(self):
        """AUTO strategy falls back to never-treated when no not-yet-treated units are available."""
        data = pd.DataFrame({'id': [1, 2, 3], 'gvar': [4, 0, 0]})
        mask = get_valid_control_units(data, 'gvar', 'id', cohort=4, period=5,
                                       strategy=ControlGroupStrategy.AUTO)
        # At period=5, only NT units are available (no units with gvar > 5)
        assert mask.sum() == 2

"""
Test suite for control_groups.py module.
Tests control group selection logic for staggered DiD estimation.
Reference: Lee & Wooldridge (2023) Section 4
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
    def test_enum_values(self):
        assert ControlGroupStrategy.NEVER_TREATED.value == 'never_treated'
        assert ControlGroupStrategy.NOT_YET_TREATED.value == 'not_yet_treated'
        assert ControlGroupStrategy.AUTO.value == 'auto'
    
    def test_from_string(self):
        assert ControlGroupStrategy('never_treated') == ControlGroupStrategy.NEVER_TREATED


class TestGetValidControlUnits:
    @pytest.fixture
    def simple_data(self):
        return pd.DataFrame({'id': [1, 2, 3, 4, 5], 'gvar': [3, 4, 5, 0, 0]})
    
    def test_never_treated_strategy(self, simple_data):
        mask = get_valid_control_units(simple_data, 'gvar', 'id', cohort=3, period=4,
                                       strategy=ControlGroupStrategy.NEVER_TREATED)
        assert mask.loc[1] == False and mask.loc[4] == True and mask.sum() == 2
    
    def test_not_yet_treated_strategy_basic(self, simple_data):
        mask = get_valid_control_units(simple_data, 'gvar', 'id', cohort=3, period=4,
                                       strategy=ControlGroupStrategy.NOT_YET_TREATED)
        assert mask.loc[1] == False  # cohort 3 (treatment)
        assert mask.loc[2] == False  # gvar=4 == period=4, starts treatment!
        assert mask.loc[3] == True   # gvar=5 > 4
        assert mask.sum() == 3
    
    def test_gvar_equals_period_excluded(self):
        data = pd.DataFrame({'id': [1, 2, 3, 4, 5, 6], 'gvar': [4, 5, 6, 0, 0, 0]})
        mask = get_valid_control_units(data, 'gvar', 'id', cohort=4, period=6,
                                       strategy=ControlGroupStrategy.NOT_YET_TREATED)
        assert mask.loc[3] == False  # gvar=6 == period=6, starts NOW!
        assert mask.sum() == 3


class TestNeverTreatedIdentification:
    def test_nan_always_recognized(self):
        data = pd.DataFrame({'id': [1, 2, 3], 'gvar': [4, np.nan, 5]})
        mask = get_valid_control_units(data, 'gvar', 'id', cohort=4, period=5,
                                       strategy=ControlGroupStrategy.NEVER_TREATED)
        assert mask.loc[2] == True
    
    def test_empty_list_never_treated_values(self):
        data = pd.DataFrame({'id': [1, 2, 3, 4], 'gvar': [4, 0, np.inf, np.nan]})
        mask = get_valid_control_units(data, 'gvar', 'id', cohort=4, period=5,
                                       strategy=ControlGroupStrategy.NEVER_TREATED,
                                       never_treated_values=[])
        assert mask.loc[4] == True and mask.sum() == 1


class TestEdgeCases:
    def test_no_control_units(self):
        data = pd.DataFrame({'id': [1, 2, 3], 'gvar': [4, 4, 5]})
        mask = get_valid_control_units(data, 'gvar', 'id', cohort=4, period=6,
                                       strategy=ControlGroupStrategy.NOT_YET_TREATED)
        assert mask.sum() == 0
    
    def test_period_less_than_cohort(self):
        data = pd.DataFrame({'id': [1, 2, 3], 'gvar': [5, 6, 0]})
        with pytest.raises(ValueError, match="period.*cohort"):
            get_valid_control_units(data, 'gvar', 'id', cohort=5, period=4,
                                    strategy=ControlGroupStrategy.NOT_YET_TREATED)


class TestValidateControlGroup:
    def test_empty_control_group(self):
        mask = pd.Series([False, False, False], index=[1, 2, 3])
        is_valid, msg = validate_control_group(mask, cohort=3, period=4)
        assert is_valid == False
    
    def test_aggregate_requires_nt(self):
        mask = pd.Series([True, True], index=[1, 2])
        is_valid, msg = validate_control_group(mask, cohort=3, period=4,
                                                aggregate_type='overall', has_never_treated=False)
        assert is_valid == False and "never treated" in msg.lower()


class TestPanelDataHandling:
    def test_returns_unit_level_mask(self):
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
        data = pd.DataFrame({'id': [1, 2, 3, 4], 'gvar': [4, 5, 6, 0]})
        mask = get_valid_control_units(data, 'gvar', 'id', cohort=4, period=5,
                                       strategy=ControlGroupStrategy.NOT_YET_TREATED)
        assert mask.loc[2] == False  # gvar=5 == period=5, EXCLUDED!
        assert mask.loc[3] == True   # gvar=6 > 5, INCLUDED


class TestCastleLawEndToEnd:
    @pytest.fixture
    def castle_data(self):
        if CASTLE_CSV_PATH.exists():
            return pd.read_csv(CASTLE_CSV_PATH)
        pytest.skip(f"Castle Law data file not found: {CASTLE_CSV_PATH}")
    
    def test_castle_data_structure(self, castle_data):
        assert 'sid' in castle_data.columns
        assert 'effyear' in castle_data.columns
        assert castle_data['sid'].nunique() == 50
    
    def test_castle_cohort_identification(self, castle_data):
        unit_effyear = castle_data.groupby('sid')['effyear'].first()
        assert (unit_effyear == 2006).sum() == 13
        assert unit_effyear.isna().sum() == 29
    
    def test_castle_control_group_cohort_2006_period_2007(self, castle_data):
        mask = get_valid_control_units(castle_data, 'effyear', 'sid', 
                                       cohort=2006, period=2007,
                                       strategy=ControlGroupStrategy.NOT_YET_TREATED)
        assert mask.sum() == 32  # 2008(2) + 2009(1) + NT(29) = 32
    
    def test_castle_control_group_evolution(self, castle_data):
        expected = {2006: 36, 2007: 32, 2008: 30, 2009: 29, 2010: 29}
        for period, expected_count in expected.items():
            mask = get_valid_control_units(castle_data, 'effyear', 'sid',
                                           cohort=2006, period=period,
                                           strategy=ControlGroupStrategy.NOT_YET_TREATED)
            assert mask.sum() == expected_count, f"period={period}"
    
    def test_castle_performance_single_call(self, castle_data):
        start = time.perf_counter()
        mask = get_valid_control_units(castle_data, 'effyear', 'sid',
                                       cohort=2006, period=2007,
                                       strategy=ControlGroupStrategy.NOT_YET_TREATED)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 10


class TestCountControlUnitsByStrategy:
    def test_basic_count(self):
        data = pd.DataFrame({'id': [1, 2, 3, 4, 5], 'gvar': [4, 5, 6, 0, 0]})
        counts = count_control_units_by_strategy(data, 'gvar', 'id', cohort=4, period=5)
        assert counts['never_treated'] == 2
        assert counts['not_yet_treated_only'] == 1  # cohort 6 only (cohort 5 starts at period 5!)
        assert counts['treatment_cohort'] == 1


class TestIdentifyNeverTreatedUnits:
    """测试identify_never_treated_units函数"""
    
    def test_identifies_nan_as_nt(self):
        data = pd.DataFrame({'id': [1, 2, 3], 'gvar': [4, np.nan, 5]})
        nt_mask = identify_never_treated_units(data, 'gvar', 'id')
        assert nt_mask.loc[2] == True
        assert nt_mask.sum() == 1
    
    def test_identifies_zero_and_inf_by_default(self):
        data = pd.DataFrame({'id': [1, 2, 3, 4], 'gvar': [4, 0, np.inf, np.nan]})
        nt_mask = identify_never_treated_units(data, 'gvar', 'id')
        assert nt_mask.sum() == 3  # 0, inf, nan
    
    def test_custom_nt_values(self):
        data = pd.DataFrame({'id': [1, 2, 3], 'gvar': [4, 9999, 0]})
        nt_mask = identify_never_treated_units(data, 'gvar', 'id', never_treated_values=[9999])
        assert nt_mask.loc[2] == True  # 9999 is NT
        assert nt_mask.loc[3] == False  # 0 is NOT NT with custom list


class TestHasNeverTreatedUnits:
    """测试has_never_treated_units快速检查函数"""
    
    def test_returns_true_when_nt_exists(self):
        data = pd.DataFrame({'id': [1, 2, 3], 'gvar': [4, 5, np.nan]})
        assert has_never_treated_units(data, 'gvar', 'id') == True
    
    def test_returns_false_when_no_nt(self):
        data = pd.DataFrame({'id': [1, 2, 3], 'gvar': [4, 5, 6]})
        assert has_never_treated_units(data, 'gvar', 'id') == False


class TestGvarTypeHandling:
    """测试gvar列类型处理"""
    
    def test_string_gvar_raises_typeerror(self):
        data = pd.DataFrame({'id': [1, 2], 'gvar': ['2005', 'never']})
        with pytest.raises(TypeError):
            get_valid_control_units(data, 'gvar', 'id', cohort=2005, period=2006,
                                    strategy=ControlGroupStrategy.NOT_YET_TREATED)
    
    def test_float_cohort_with_int_gvar(self):
        """整数gvar与浮点数cohort参数兼容"""
        data = pd.DataFrame({'id': [1, 2, 3], 'gvar': [4, 5, 0]})
        mask = get_valid_control_units(data, 'gvar', 'id', cohort=4.0, period=5.0,
                                       strategy=ControlGroupStrategy.NOT_YET_TREATED)
        assert mask.sum() == 1  # NT only


class TestSampleConstruction:
    """验证公式(4.13)的样本构建逻辑
    
    D_{ig} + A_{i,r+1} = 1
    样本 = 处理组 | 控制组 (互斥)
    """
    
    def test_sample_excludes_other_treated_cohorts(self):
        """验证排除其他已处理cohorts"""
        data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5, 6],
            'gvar': [3, 4, 5, 6, 0, 0]
        })
        unit_gvar = data.groupby('id')['gvar'].first()
        
        # cohort=3, period=5: 排除cohort 4和5
        control_mask = get_valid_control_units(data, 'gvar', 'id', cohort=3, period=5,
                                               strategy=ControlGroupStrategy.NOT_YET_TREATED)
        treat_mask = (unit_gvar == 3)
        sample_mask = treat_mask | control_mask
        
        assert not sample_mask.loc[2], "Cohort 4 (gvar<period) should be excluded"
        assert not sample_mask.loc[3], "Cohort 5 (gvar==period) should be excluded"
        assert sample_mask.loc[4], "Cohort 6 (gvar>period) should be included"
    
    def test_treat_and_control_mutually_exclusive(self):
        """处理组和控制组互斥"""
        data = pd.DataFrame({'id': [1, 2, 3], 'gvar': [4, 5, 0]})
        unit_gvar = data.groupby('id')['gvar'].first()
        
        control_mask = get_valid_control_units(data, 'gvar', 'id', cohort=4, period=4,
                                               strategy=ControlGroupStrategy.NOT_YET_TREATED)
        treat_mask = (unit_gvar == 4)
        
        overlap = treat_mask & control_mask
        assert not overlap.any(), "Treatment and control should be mutually exclusive"


class TestAllEventuallyTreatedAggregation:
    """测试All Eventually Treated情况下的聚合约束"""
    
    def test_cohort_aggregate_requires_nt(self):
        """无NT时cohort聚合应失败"""
        data = pd.DataFrame({'id': [1, 2, 3], 'gvar': [4, 5, 6]})
        mask = get_valid_control_units(data, 'gvar', 'id', cohort=4, period=5,
                                       strategy=ControlGroupStrategy.NOT_YET_TREATED)
        
        is_valid, msg = validate_control_group(mask, cohort=4, period=5,
                                                aggregate_type='cohort', has_never_treated=False)
        assert is_valid == False
        assert 'never treated' in msg.lower()
    
    def test_gr_specific_ok_without_nt(self):
        """无NT时(g,r)特定效应可估计"""
        data = pd.DataFrame({'id': [1, 2, 3], 'gvar': [4, 5, 6]})
        mask = get_valid_control_units(data, 'gvar', 'id', cohort=4, period=5,
                                       strategy=ControlGroupStrategy.NOT_YET_TREATED)
        
        is_valid, msg = validate_control_group(mask, cohort=4, period=5,
                                                aggregate_type=None, has_never_treated=False)
        assert is_valid == True


class TestAdditionalEdgeCases:
    """额外边界情况测试"""
    
    def test_empty_dataframe_raises_error(self):
        data = pd.DataFrame(columns=['id', 'gvar'])
        with pytest.raises(ValueError):
            get_valid_control_units(data, 'gvar', 'id', cohort=3, period=4,
                                    strategy=ControlGroupStrategy.NOT_YET_TREATED)
    
    def test_missing_column_raises_keyerror(self):
        data = pd.DataFrame({'id': [1, 2], 'other': [3, 4]})
        with pytest.raises(KeyError):
            get_valid_control_units(data, 'gvar', 'id', cohort=3, period=4,
                                    strategy=ControlGroupStrategy.NOT_YET_TREATED)
    
    def test_all_never_treated(self):
        """所有单位都是NT"""
        data = pd.DataFrame({'id': [1, 2, 3], 'gvar': [0, np.nan, np.inf]})
        mask = get_valid_control_units(data, 'gvar', 'id', cohort=4, period=5,
                                       strategy=ControlGroupStrategy.NOT_YET_TREATED)
        assert mask.sum() == 3
    
    def test_single_unit(self):
        """单个单位"""
        data = pd.DataFrame({'id': [1], 'gvar': [np.nan]})
        mask = get_valid_control_units(data, 'gvar', 'id', cohort=4, period=5,
                                       strategy=ControlGroupStrategy.NOT_YET_TREATED)
        assert len(mask) == 1
        assert mask.loc[1] == True
    
    def test_auto_strategy_fallback_to_nt(self):
        """AUTO策略在无NYT时回退到NT"""
        data = pd.DataFrame({'id': [1, 2, 3], 'gvar': [4, 0, 0]})
        mask = get_valid_control_units(data, 'gvar', 'id', cohort=4, period=5,
                                       strategy=ControlGroupStrategy.AUTO)
        # 在period=5，只有NT可用（无gvar > 5的单位）
        assert mask.sum() == 2

"""
Story 3.3: Staggered场景子样本构建验证测试

Phase 1 测试: 验证子样本构建与Stata筛选条件完全一致

关键验证点:
1. 子样本筛选条件与Stata命令一致
2. n_obs/n_treated/n_control与Stata完全一致
3. 控制组cohort分布正确
4. 严格使用 gvar > period (不是 >=)

Stata参考代码 (2.lee_wooldridge_rolling_staggered.do:47-53):
    teffects ipwra (y_44 x1 x2) (g4 x1 x2) if f04, atet
    teffects ipwra (y_45 x1 x2) (g4 x1 x2) if f05 & ~g5, atet
    teffects ipwra (y_46 x1 x2) (g4 x1 x2) if f06 & (g5 + g6 != 1), atet
    teffects ipwra (y_55 x1 x2) (g5 x1 x2) if f05 & ~g4, atet
    teffects ipwra (y_56 x1 x2) (g5 x1 x2) if f06 & (g4 + g6 != 1), atet
    teffects ipwra (y_66 x1 x2) (g6 x1 x2) if f06 & (g4 + g5 != 1), atet
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

# 添加tests/staggered到路径以便导入conftest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from conftest import (
    STATA_IPWRA_RESULTS,
    EXPECTED_CONTROL_COHORTS,
    build_subsample_for_gr,
    validate_subsample,
)


class TestSubsampleConstruction:
    """子样本构建正确性测试"""
    
    # Stata筛选条件对照表
    STATA_CONDITIONS = {
        (4, 4): "f04",                      # period==4
        (4, 5): "f05 & ~g5",                # period==5 & gvar!=5
        (4, 6): "f06 & (g5+g6!=1)",         # period==6 & ~gvar.isin([5,6])
        (5, 5): "f05 & ~g4",                # period==5 & gvar!=4
        (5, 6): "f06 & (g4+g6!=1)",         # period==6 & ~gvar.isin([4,6])
        (6, 6): "f06 & (g4+g5!=1)",         # period==6 & ~gvar.isin([4,5])
    }
    
    @pytest.mark.parametrize("g,r", [(4, 4), (4, 5), (4, 6), (5, 5), (5, 6), (6, 6)])
    def test_subsample_period_filter(self, staggered_data, g, r):
        """测试子样本只包含period r的观测"""
        subsample = build_subsample_for_gr(staggered_data, g, r)
        
        # 所有观测应该是period r
        assert (subsample['year'] == r).all(), \
            f"(g={g}, r={r}) 子样本包含非period {r}的观测"
    
    @pytest.mark.parametrize("g,r", [(4, 4), (4, 5), (4, 6), (5, 5), (5, 6), (6, 6)])
    def test_treatment_group_identification(self, staggered_data, g, r):
        """测试处理组识别正确（gvar == cohort_g）"""
        subsample = build_subsample_for_gr(staggered_data, g, r)
        
        # 处理组应该是gvar == g的单位
        treated = subsample[subsample['d'] == 1]
        assert (treated['gvar'] == g).all(), \
            f"(g={g}, r={r}) 处理组包含非cohort {g}的单位"
    
    @pytest.mark.parametrize("g,r", [(4, 4), (4, 5), (4, 6), (5, 5), (5, 6), (6, 6)])
    def test_control_group_strict_inequality(self, staggered_data, g, r):
        """测试控制组使用严格不等式 gvar > period"""
        subsample = build_subsample_for_gr(staggered_data, g, r)
        control = subsample[subsample['d'] == 0]
        
        # 控制组每个单位的gvar应该满足:
        # 1. gvar > r (严格大于，尚未处理), 或
        # 2. gvar == 0 (NT)
        for _, row in control.iterrows():
            gvar = row['gvar']
            assert gvar == 0 or gvar > r, \
                f"(g={g}, r={r}) 控制组包含无效单位: gvar={gvar}"
    
    @pytest.mark.parametrize("g,r", [(4, 4), (4, 5), (4, 6), (5, 5), (5, 6), (6, 6)])
    def test_no_same_period_treatment_in_control(self, staggered_data, g, r):
        """测试gvar == period的单位不在控制组中（除非是目标cohort）"""
        subsample = build_subsample_for_gr(staggered_data, g, r)
        control = subsample[subsample['d'] == 0]
        
        # 控制组中不应该有gvar == period的单位
        # 因为gvar == period意味着该期开始处理
        gvar_equals_period = (control['gvar'] == r)
        assert not gvar_equals_period.any(), \
            f"(g={g}, r={r}) 控制组错误包含gvar=={r}的单位（该期开始处理）"


class TestSubsampleCountsVsStata:
    """子样本计数与Stata一致性测试"""
    
    @pytest.mark.parametrize("g,r", [(4, 4), (4, 5), (4, 6), (5, 5), (5, 6), (6, 6)])
    def test_n_obs_vs_stata(self, staggered_data, g, r, stata_ipwra_results):
        """测试n_obs与Stata完全一致"""
        subsample = build_subsample_for_gr(staggered_data, g, r)
        stata = stata_ipwra_results[(g, r)]
        
        python_n_obs = len(subsample)
        stata_n_obs = stata['n_obs']
        
        assert python_n_obs == stata_n_obs, \
            f"(g={g}, r={r}) n_obs不匹配: Python={python_n_obs}, Stata={stata_n_obs}"
    
    @pytest.mark.parametrize("g,r", [(4, 4), (4, 5), (4, 6), (5, 5), (5, 6), (6, 6)])
    def test_n_treated_vs_stata(self, staggered_data, g, r, stata_ipwra_results):
        """测试n_treated与Stata完全一致"""
        subsample = build_subsample_for_gr(staggered_data, g, r)
        stata = stata_ipwra_results[(g, r)]
        
        python_n_treated = (subsample['d'] == 1).sum()
        stata_n_treated = stata['n_treated']
        
        assert python_n_treated == stata_n_treated, \
            f"(g={g}, r={r}) n_treated不匹配: Python={python_n_treated}, Stata={stata_n_treated}"
    
    @pytest.mark.parametrize("g,r", [(4, 4), (4, 5), (4, 6), (5, 5), (5, 6), (6, 6)])
    def test_n_control_vs_stata(self, staggered_data, g, r, stata_ipwra_results):
        """测试n_control与Stata完全一致"""
        subsample = build_subsample_for_gr(staggered_data, g, r)
        stata = stata_ipwra_results[(g, r)]
        
        python_n_control = (subsample['d'] == 0).sum()
        stata_n_control = stata['n_control']
        
        assert python_n_control == stata_n_control, \
            f"(g={g}, r={r}) n_control不匹配: Python={python_n_control}, Stata={stata_n_control}"
    
    @pytest.mark.parametrize("g,r", [(4, 4), (4, 5), (4, 6), (5, 5), (5, 6), (6, 6)])
    def test_all_counts_summary(self, staggered_data, g, r, stata_ipwra_results):
        """综合测试所有计数与Stata一致"""
        subsample = build_subsample_for_gr(staggered_data, g, r)
        stata = stata_ipwra_results[(g, r)]
        
        python_n_obs = len(subsample)
        python_n_treated = (subsample['d'] == 1).sum()
        python_n_control = (subsample['d'] == 0).sum()
        
        # 一次性验证所有计数
        assert python_n_obs == stata['n_obs'], f"n_obs不匹配"
        assert python_n_treated == stata['n_treated'], f"n_treated不匹配"
        assert python_n_control == stata['n_control'], f"n_control不匹配"
        
        # 额外验证: n_obs == n_treated + n_control
        assert python_n_obs == python_n_treated + python_n_control, \
            f"n_obs != n_treated + n_control"


class TestControlGroupCohortDistribution:
    """控制组cohort分布验证"""
    
    @pytest.mark.parametrize("g,r,expected_cohorts", [
        (4, 4, {5, 6, 0}),   # NYT(5,6) + NT
        (4, 5, {6, 0}),      # NYT(6) + NT
        (4, 6, {0}),         # 仅NT
        (5, 5, {6, 0}),      # NYT(6) + NT
        (5, 6, {0}),         # 仅NT
        (6, 6, {0}),         # 仅NT
    ])
    def test_control_group_cohorts(self, staggered_data, g, r, expected_cohorts):
        """验证控制组包含正确的cohorts"""
        subsample = build_subsample_for_gr(staggered_data, g, r)
        control = subsample[subsample['d'] == 0]
        
        actual_cohorts = set(control['gvar'].unique())
        
        assert actual_cohorts == expected_cohorts, \
            f"(g={g}, r={r}) 控制组cohorts不匹配: 实际={actual_cohorts}, 预期={expected_cohorts}"
    
    def test_no_treated_cohort_in_control_44(self, staggered_data):
        """(4,4): 控制组不应包含cohort 4（处理组本身）"""
        subsample = build_subsample_for_gr(staggered_data, 4, 4)
        control = subsample[subsample['d'] == 0]
        
        assert 4 not in control['gvar'].values, \
            "(4,4) 控制组错误包含cohort 4"
    
    def test_no_earlier_cohort_in_control_55(self, staggered_data):
        """(5,5): 控制组不应包含cohort 4（已处理）或cohort 5（处理组）"""
        subsample = build_subsample_for_gr(staggered_data, 5, 5)
        control = subsample[subsample['d'] == 0]
        
        # cohort 4 在 period 5 已经被处理，不应在控制组
        # cohort 5 是处理组，不应在控制组
        invalid_cohorts = control['gvar'].isin([4, 5])
        assert not invalid_cohorts.any(), \
            "(5,5) 控制组错误包含cohort 4或5"
    
    def test_control_only_nt_at_last_period(self, staggered_data):
        """r=T=6时，控制组应仅包含NT（所有cohort 4,5,6都已处理）"""
        for g in [4, 5, 6]:
            subsample = build_subsample_for_gr(staggered_data, g, 6)
            control = subsample[subsample['d'] == 0]
            
            # r=6时，所有非NT单位都已处理
            # 控制组应仅包含gvar=0（NT）
            assert (control['gvar'] == 0).all(), \
                f"(g={g}, r=6) 控制组应仅包含NT，但包含: {control['gvar'].unique()}"


class TestStrictInequalityBoundary:
    """严格不等式边界情况测试"""
    
    def test_gvar_equals_period_is_treated_not_control(self):
        """测试gvar==period的单位是处理组，不是控制组"""
        # 创建测试数据：
        # 在period=4，gvar=4的单位开始处理，应属于处理组（如果是目标cohort）
        data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5, 6],
            'year': [4, 4, 4, 4, 4, 4],
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'gvar': [4, 4, 5, 5, 0, 0],  # 2个cohort4, 2个cohort5, 2个NT
            'x1': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        })
        
        # 对于 (g=4, r=4):
        # - 处理组: gvar==4 (id 1,2)
        # - 控制组: gvar>4 (id 3,4: gvar=5) + NT (id 5,6: gvar=0)
        subsample = build_subsample_for_gr(data, cohort_g=4, period_r=4)
        
        # gvar=4 应该是处理组
        gvar4_units = subsample[subsample['gvar'] == 4]
        assert (gvar4_units['d'] == 1).all(), "gvar=4应该是处理组"
        
        # gvar=5 应该是控制组（5 > 4）
        gvar5_units = subsample[subsample['gvar'] == 5]
        assert (gvar5_units['d'] == 0).all(), "gvar=5应该是控制组"
        
        # gvar=0 (NT) 应该是控制组
        gvar0_units = subsample[subsample['gvar'] == 0]
        assert (gvar0_units['d'] == 0).all(), "gvar=0(NT)应该是控制组"
    
    def test_gvar_equals_period_excluded_from_control_when_not_target(self):
        """测试gvar==period但不是目标cohort的单位被排除"""
        # 对于 (g=4, r=5)，cohort 5 在 period 5 开始处理
        # cohort 5 不应该在控制组（因为它们在该期开始处理）
        data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5, 6],
            'year': [5, 5, 5, 5, 5, 5],
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'gvar': [4, 4, 5, 5, 6, 0],  # 2个g4, 2个g5, 1个g6, 1个NT
            'x1': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        })
        
        # 对于 (g=4, r=5):
        # - 处理组: gvar==4 (id 1,2)
        # - 控制组: gvar>5 (id 5: gvar=6) + NT (id 6: gvar=0)
        # - 排除: gvar==5 (id 3,4) 因为它们在period 5开始处理
        subsample = build_subsample_for_gr(data, cohort_g=4, period_r=5)
        
        # gvar=5 不应该在子样本中
        # (既不是处理组cohort 4，也不是控制组因为不满足gvar>5)
        gvar5_in_subsample = subsample['gvar'] == 5
        assert not gvar5_in_subsample.any(), \
            "gvar=5不应该在(g=4,r=5)的子样本中"
        
        # 验证控制组仅包含 gvar=6 和 gvar=0
        control = subsample[subsample['d'] == 0]
        assert set(control['gvar'].unique()) == {6, 0}, \
            f"控制组应仅包含gvar=6和gvar=0，实际: {control['gvar'].unique()}"
    
    def test_boundary_condition_comprehensive(self, staggered_data):
        """综合边界条件测试：验证所有(g,r)组合的gvar==r排除逻辑"""
        test_cases = [
            # (g, r, excluded_cohorts, reason)
            (4, 4, [], "r=g，无需排除"),
            (4, 5, [5], "cohort 5在period 5开始处理"),
            (4, 6, [5, 6], "cohort 5,6在period 6前已处理"),
            (5, 5, [4], "cohort 4在period 5前已处理"),
            (5, 6, [4, 6], "cohort 4在period 5前已处理，cohort 6在period 6开始处理"),
            (6, 6, [4, 5], "cohort 4,5在period 6前已处理"),
        ]
        
        for g, r, excluded, reason in test_cases:
            subsample = build_subsample_for_gr(staggered_data, g, r)
            control = subsample[subsample['d'] == 0]
            
            for excl in excluded:
                assert excl not in control['gvar'].values, \
                    f"(g={g}, r={r}) cohort {excl} 不应在控制组: {reason}"


class TestSubsampleValidation:
    """子样本验证函数测试"""
    
    def test_valid_subsample_passes(self, staggered_data):
        """测试有效子样本通过验证"""
        subsample = build_subsample_for_gr(staggered_data, 4, 4)
        is_valid, warnings = validate_subsample(subsample, 4, 4)
        
        assert is_valid, "有效子样本应该通过验证"
    
    def test_empty_control_raises(self):
        """测试控制组为空时抛出异常"""
        # 创建所有单位都在period 1被处理的数据
        data = pd.DataFrame({
            'id': [1, 2, 3],
            'year': [2, 2, 2],
            'y': [1.0, 2.0, 3.0],
            'gvar': [1, 1, 1],  # 所有人在period 1被处理
            'x1': [0.1, 0.2, 0.3],
        })
        
        # 在period 2，没有控制组（所有人都在period 1开始处理）
        with pytest.raises(ValueError, match="控制组为空"):
            build_subsample_for_gr(data, cohort_g=1, period_r=2)
    
    def test_empty_treatment_raises(self):
        """测试处理组为空时抛出异常"""
        data = pd.DataFrame({
            'id': [1, 2, 3],
            'year': [4, 4, 4],
            'y': [1.0, 2.0, 3.0],
            'gvar': [0, 0, 0],  # 所有人是NT
            'x1': [0.1, 0.2, 0.3],
        })
        
        # 尝试估计cohort 4，但数据中没有cohort 4
        with pytest.raises(ValueError, match="处理组为空"):
            build_subsample_for_gr(data, cohort_g=4, period_r=4)


class TestSubsampleConsistencyWithStataConditions:
    """子样本与Stata条件语句一致性测试"""
    
    def test_condition_44_equivalent(self, staggered_data):
        """测试(4,4)条件等价于Stata: if f04"""
        # Python实现
        subsample = build_subsample_for_gr(staggered_data, 4, 4)
        python_ids = set(subsample['id'].values)
        
        # Stata等价条件: if f04 (year == 4)
        # 但还需要满足子样本条件: D_g4 + A_5 = 1
        # 即: gvar==4 或 gvar>4 或 gvar==0(NT)
        stata_equiv = staggered_data[
            (staggered_data['year'] == 4) &
            ((staggered_data['gvar'] == 4) | 
             (staggered_data['gvar'] > 4) | 
             (staggered_data['gvar'] == 0))
        ]
        stata_ids = set(stata_equiv['id'].values)
        
        assert python_ids == stata_ids, \
            f"(4,4) ID集合不匹配: Python有{len(python_ids)}, Stata有{len(stata_ids)}"
    
    def test_condition_45_equivalent(self, staggered_data):
        """测试(4,5)条件等价于Stata: if f05 & ~g5"""
        subsample = build_subsample_for_gr(staggered_data, 4, 5)
        python_ids = set(subsample['id'].values)
        
        # Stata: if f05 & ~g5
        # 等价于: year==5 & gvar!=5
        # 再加子样本条件: gvar==4 或 gvar>5 或 gvar==0
        stata_equiv = staggered_data[
            (staggered_data['year'] == 5) &
            (staggered_data['gvar'] != 5) &
            ((staggered_data['gvar'] == 4) | 
             (staggered_data['gvar'] > 5) | 
             (staggered_data['gvar'] == 0))
        ]
        stata_ids = set(stata_equiv['id'].values)
        
        assert python_ids == stata_ids, \
            f"(4,5) ID集合不匹配"
    
    def test_condition_46_equivalent(self, staggered_data):
        """测试(4,6)条件等价于Stata: if f06 & (g5+g6!=1)"""
        subsample = build_subsample_for_gr(staggered_data, 4, 6)
        python_ids = set(subsample['id'].values)
        
        # Stata: if f06 & (g5+g6!=1)
        # g5+g6!=1 等价于 ~gvar.isin([5,6])
        # 子样本条件: gvar==4 或 gvar>6 或 gvar==0
        # 由于T=6，gvar>6不存在，所以控制组仅为gvar==0
        stata_equiv = staggered_data[
            (staggered_data['year'] == 6) &
            (~staggered_data['gvar'].isin([5, 6])) &
            ((staggered_data['gvar'] == 4) | (staggered_data['gvar'] == 0))
        ]
        stata_ids = set(stata_equiv['id'].values)
        
        assert python_ids == stata_ids, \
            f"(4,6) ID集合不匹配"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

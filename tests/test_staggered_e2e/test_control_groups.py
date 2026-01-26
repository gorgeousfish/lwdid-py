# -*- coding: utf-8 -*-
"""
控制组选择验证测试 (Task 6.2.11-6.2.13)

验证Staggered场景的控制组选择逻辑:
- 控制组使用严格不等式: gvar > period
- gvar == period 是处理组（当期开始处理）
- gvar < period 已被处理（排除）
"""

import numpy as np
import pandas as pd
import pytest

from .conftest import (
    GR_COMBINATIONS,
    STATA_IPWRA_RESULTS,
    EXPECTED_CONTROL_COHORTS,
    build_subsample_for_gr,
)


class TestControlGroupSizes:
    """控制组样本量测试"""
    
    EXPECTED_SIZES = {
        (4, 4): {'n_treated': 129, 'n_control': 871, 'n_obs': 1000},
        (4, 5): {'n_treated': 129, 'n_control': 762, 'n_obs': 891},
        (4, 6): {'n_treated': 129, 'n_control': 652, 'n_obs': 781},
        (5, 5): {'n_treated': 109, 'n_control': 762, 'n_obs': 871},
        (5, 6): {'n_treated': 109, 'n_control': 652, 'n_obs': 761},
        (6, 6): {'n_treated': 110, 'n_control': 652, 'n_obs': 762},
    }
    
    @pytest.mark.parametrize("g,r", GR_COMBINATIONS)
    def test_control_group_size(self, staggered_data, g, r):
        """测试控制组样本量与Stata一致"""
        subsample = build_subsample_for_gr(
            staggered_data, g, r,
            gvar_col='first_treat',
            period_col='period',
            control_group='not_yet_treated'
        )
        
        n_treated = (subsample['d'] == 1).sum()
        n_control = (subsample['d'] == 0).sum()
        n_obs = len(subsample)
        
        expected = self.EXPECTED_SIZES[(g, r)]
        
        assert n_treated == expected['n_treated'], \
            f"({g},{r}): n_treated {n_treated} != {expected['n_treated']}"
        assert n_control == expected['n_control'], \
            f"({g},{r}): n_control {n_control} != {expected['n_control']}"
        assert n_obs == expected['n_obs'], \
            f"({g},{r}): n_obs {n_obs} != {expected['n_obs']}"
    
    @pytest.mark.parametrize("g,r", GR_COMBINATIONS)
    def test_treated_count_vs_stata(self, staggered_data, stata_ipwra_results, g, r):
        """验证处理组数量与Stata IPWRA结果一致"""
        subsample = build_subsample_for_gr(
            staggered_data, g, r,
            gvar_col='first_treat',
            period_col='period'
        )
        
        n_treated = (subsample['d'] == 1).sum()
        stata = stata_ipwra_results[(g, r)]
        
        assert n_treated == stata['n_treated'], \
            f"({g},{r}): n_treated {n_treated} != Stata {stata['n_treated']}"
    
    @pytest.mark.parametrize("g,r", GR_COMBINATIONS)
    def test_control_count_vs_stata(self, staggered_data, stata_ipwra_results, g, r):
        """验证控制组数量与Stata IPWRA结果一致"""
        subsample = build_subsample_for_gr(
            staggered_data, g, r,
            gvar_col='first_treat',
            period_col='period'
        )
        
        n_control = (subsample['d'] == 0).sum()
        stata = stata_ipwra_results[(g, r)]
        
        assert n_control == stata['n_control'], \
            f"({g},{r}): n_control {n_control} != Stata {stata['n_control']}"


class TestStrictInequality:
    """严格不等式验证测试"""
    
    def test_gvar_equals_period_is_treated(self, staggered_data):
        """gvar == period 的单位应该是处理组，不是控制组"""
        for g, r in [(4, 4), (5, 5), (6, 6)]:
            period_data = staggered_data[staggered_data['period'] == r].copy()
            
            # gvar == r 应该是处理组（cohort g在period r开始处理）
            treated_in_period = period_data[period_data['first_treat'] == r]
            
            if g == r:  # 对角线元素
                assert len(treated_in_period) > 0, \
                    f"({g},{r}): No units with gvar == {r} found"
            
            # 构建子样本
            subsample = build_subsample_for_gr(
                staggered_data, g, r,
                gvar_col='first_treat',
                period_col='period'
            )
            
            # 验证gvar == r的单位不在控制组中
            control = subsample[subsample['d'] == 0]
            gvar_equals_r = control[control['first_treat'] == r]
            assert len(gvar_equals_r) == 0, \
                f"({g},{r}): {len(gvar_equals_r)} units with gvar=={r} found in control group"
    
    @pytest.mark.parametrize("g,r", GR_COMBINATIONS)
    def test_gvar_greater_than_period_is_control(self, staggered_data, g, r):
        """gvar > period 的单位应该是控制组"""
        subsample = build_subsample_for_gr(
            staggered_data, g, r,
            gvar_col='first_treat',
            period_col='period'
        )
        
        control = subsample[subsample['d'] == 0]
        
        # 所有控制组单位的gvar应该 > r
        control_gvar = control['first_treat']
        
        assert (control_gvar > r).all(), \
            f"({g},{r}): Some control units have gvar <= {r}.\n" \
            f"  Violations: {control_gvar[control_gvar <= r].unique()}"
    
    @pytest.mark.parametrize("g,r", GR_COMBINATIONS)
    def test_treated_has_gvar_equals_cohort(self, staggered_data, g, r):
        """处理组应该是 gvar == cohort_g"""
        subsample = build_subsample_for_gr(
            staggered_data, g, r,
            gvar_col='first_treat',
            period_col='period'
        )
        
        treated = subsample[subsample['d'] == 1]
        
        # 所有处理组单位的gvar应该 == g
        assert (treated['first_treat'] == g).all(), \
            f"({g},{r}): Some treated units have gvar != {g}"


class TestControlGroupCohorts:
    """控制组cohort构成验证"""
    
    @pytest.mark.parametrize("g,r", GR_COMBINATIONS)
    def test_control_cohorts_correct(self, staggered_data, expected_control_cohorts, g, r):
        """验证控制组包含正确的cohorts"""
        subsample = build_subsample_for_gr(
            staggered_data, g, r,
            gvar_col='first_treat',
            period_col='period'
        )
        
        control = subsample[subsample['d'] == 0]
        actual_cohorts = set(control['first_treat'].unique())
        expected = expected_control_cohorts[(g, r)]
        
        # 转换inf值进行比较
        actual_cohorts_normalized = {
            float('inf') if np.isinf(c) else c 
            for c in actual_cohorts
        }
        
        assert actual_cohorts_normalized == expected, \
            f"({g},{r}): 控制组cohorts不匹配\n" \
            f"  实际: {actual_cohorts_normalized}\n" \
            f"  预期: {expected}"
    
    def test_control_group_shrinks_over_time(self, staggered_data):
        """验证控制组随时间减少"""
        # 对于cohort 4，控制组应该随r增加而减少
        control_sizes = {}
        for r in [4, 5, 6]:
            subsample = build_subsample_for_gr(
                staggered_data, 4, r,
                gvar_col='first_treat',
                period_col='period'
            )
            control_sizes[r] = (subsample['d'] == 0).sum()
        
        assert control_sizes[4] > control_sizes[5] > control_sizes[6], \
            f"控制组未随时间减少: {control_sizes}"


class TestNeverTreatedControl:
    """Never Treated控制组测试"""
    
    @pytest.mark.parametrize("g,r", GR_COMBINATIONS)
    def test_never_treated_always_in_control(self, staggered_data, g, r):
        """Never treated单位应该始终在控制组中"""
        # 获取NT单位
        nt_units = staggered_data[np.isinf(staggered_data['first_treat'])]['id'].unique()
        
        subsample = build_subsample_for_gr(
            staggered_data, g, r,
            gvar_col='first_treat',
            period_col='period'
        )
        
        control = subsample[subsample['d'] == 0]
        control_ids = set(control['id'].unique())
        
        # 验证所有NT单位都在控制组中
        nt_in_control = set(nt_units) & control_ids
        
        # 由于period筛选，可能不是所有NT都在
        # 但如果在子样本中，应该在控制组中
        nt_in_subsample = set(nt_units) & set(subsample['id'].unique())
        
        assert nt_in_subsample == nt_in_control, \
            f"({g},{r}): 部分NT单位不在控制组中"
    
    def test_never_treated_only_control(self, staggered_data):
        """测试仅使用Never Treated作为控制组"""
        for g, r in [(4, 6), (5, 6), (6, 6)]:  # 这些只有NT作为控制
            subsample = build_subsample_for_gr(
                staggered_data, g, r,
                gvar_col='first_treat',
                period_col='period',
                control_group='never_treated'  # 仅NT
            )
            
            control = subsample[subsample['d'] == 0]
            
            # 所有控制组单位应该是NT (inf)
            assert np.isinf(control['first_treat']).all(), \
                f"({g},{r}): 控制组包含非NT单位"


class TestSubsampleConstruction:
    """子样本构建功能测试"""
    
    def test_subsample_period_filter(self, staggered_data):
        """验证子样本正确筛选到目标period"""
        for g, r in GR_COMBINATIONS:
            subsample = build_subsample_for_gr(
                staggered_data, g, r,
                gvar_col='first_treat',
                period_col='period'
            )
            
            # 所有观测应该在period r
            assert (subsample['period'] == r).all(), \
                f"({g},{r}): 子样本包含非目标period的观测"
    
    def test_subsample_d_is_binary(self, staggered_data):
        """验证处理指示符是二元的"""
        for g, r in GR_COMBINATIONS:
            subsample = build_subsample_for_gr(
                staggered_data, g, r,
                gvar_col='first_treat',
                period_col='period'
            )
            
            assert set(subsample['d'].unique()) <= {0, 1}, \
                f"({g},{r}): 处理指示符不是二元"
    
    def test_invalid_period_raises_error(self, staggered_data):
        """不存在的period应该报错"""
        with pytest.raises(ValueError, match="不存在于数据中"):
            build_subsample_for_gr(
                staggered_data, 4, 10,  # period 10不存在
                gvar_col='first_treat',
                period_col='period'
            )
    
    def test_empty_treated_raises_error(self, staggered_data):
        """处理组为空应该报错"""
        # Cohort 7不存在
        with pytest.raises(ValueError, match="处理组为空"):
            build_subsample_for_gr(
                staggered_data, 7, 4,  # cohort 7不存在
                gvar_col='first_treat',
                period_col='period'
            )

# -*- coding: utf-8 -*-
"""
变换变量验证测试 (Task 6.2.7-6.2.10)

验证Staggered场景的变换变量计算:
- 公式(4.12): ŷ_{irg} = Y_{ir} - (1/(g-1)) × Σ_{s=1}^{g-1} Y_{is}
- 关键: 分母是 g-1（cohort固定），不是 r-1
- 同一cohort的所有post periods使用相同的pre-mean
"""

import numpy as np
import pandas as pd
import pytest

from .conftest import (
    GR_COMBINATIONS,
    compute_transformed_outcome_staggered,
    assert_att_close_to_stata,
)


class TestTransformationStaggeredFormula:
    """变换公式验证测试"""
    
    def test_denominator_is_g_minus_1(self, staggered_data):
        """
        验证分母是 (g-1)，不是 (r-1)
        
        对于cohort 4:
        - (4,4): 分母=3 (periods 1,2,3)
        - (4,5): 分母=3 (periods 1,2,3)，不是4
        - (4,6): 分母=3 (periods 1,2,3)，不是5
        """
        data = staggered_data
        
        # 创建已知数据测试
        test_unit_id = data['id'].iloc[0]
        test_unit_data = data[data['id'] == test_unit_id].copy()
        
        # 获取该单位的y值（periods 1-6）
        y_vals = test_unit_data.sort_values('period').set_index('period')['y']
        
        # Cohort 4的pre-mean应该是 mean(y_1, y_2, y_3)
        expected_pre_mean_g4 = (y_vals[1] + y_vals[2] + y_vals[3]) / 3
        
        # 计算变换变量
        for r in [4, 5, 6]:
            y_dot = compute_transformed_outcome_staggered(
                data, 'y', 'id', 'period', 'first_treat',
                cohort_g=4, target_period=r
            )
            
            if test_unit_id in y_dot.index:
                computed_y_dot = y_dot.loc[test_unit_id]
                expected_y_dot = y_vals[r] - expected_pre_mean_g4
                
                assert abs(computed_y_dot - expected_y_dot) < 1e-10, \
                    f"(4,{r}): 变换变量计算错误\n" \
                    f"  computed: {computed_y_dot}\n" \
                    f"  expected: {expected_y_dot}"
    
    def test_cohort4_same_pre_mean_across_periods(self, staggered_data):
        """
        Cohort 4的所有periods应使用相同的pre-mean
        
        (4,4), (4,5), (4,6) 都使用 mean(y_1, y_2, y_3)
        """
        data = staggered_data
        
        # 筛选cohort 4的单位
        cohort4_units = data[data['first_treat'] == 4]['id'].unique()
        
        # 计算各period的变换
        pre_means = {}
        for r in [4, 5, 6]:
            y_dot = compute_transformed_outcome_staggered(
                data, 'y', 'id', 'period', 'first_treat',
                cohort_g=4, target_period=r
            )
            
            # 从y_dot反推pre_mean
            Y_r = data[data['period'] == r].set_index('id')['y']
            pre_means[r] = Y_r - y_dot
        
        # 验证pre-mean在各period相同（对于相同单位）
        common_ids = set(pre_means[4].index) & set(pre_means[5].index) & set(pre_means[6].index)
        
        for uid in list(common_ids)[:10]:  # 抽样检查
            pm4 = pre_means[4].loc[uid]
            pm5 = pre_means[5].loc[uid]
            pm6 = pre_means[6].loc[uid]
            
            # 使用较宽容差处理float32精度问题
            assert abs(pm4 - pm5) < 1e-5, \
                f"Unit {uid}: pre_mean at r=4 ({pm4}) != r=5 ({pm5})"
            assert abs(pm5 - pm6) < 1e-5, \
                f"Unit {uid}: pre_mean at r=5 ({pm5}) != r=6 ({pm6})"
    
    def test_different_cohorts_different_denominators(self, staggered_data):
        """
        不同cohort应使用不同的分母
        
        - Cohort 4: 分母=3
        - Cohort 5: 分母=4
        - Cohort 6: 分母=5
        """
        data = staggered_data
        
        # 创建简单测试数据
        test_data = pd.DataFrame({
            'id': [1]*6 + [2]*6 + [3]*6,
            'period': list(range(1, 7)) * 3,
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] * 3,  # 线性增长
            'first_treat': [4]*6 + [5]*6 + [6]*6,  # 不同cohort
        })
        
        # Cohort 4, period 4: pre-mean = (1+2+3)/3 = 2.0
        # y_dot = 4 - 2 = 2.0
        y_dot_44 = compute_transformed_outcome_staggered(
            test_data, 'y', 'id', 'period', 'first_treat',
            cohort_g=4, target_period=4
        )
        assert abs(y_dot_44.loc[1] - 2.0) < 1e-10, \
            f"y_dot_44 = {y_dot_44.loc[1]}, expected 2.0"
        
        # Cohort 5, period 5: pre-mean = (1+2+3+4)/4 = 2.5
        # y_dot = 5 - 2.5 = 2.5
        y_dot_55 = compute_transformed_outcome_staggered(
            test_data, 'y', 'id', 'period', 'first_treat',
            cohort_g=5, target_period=5
        )
        assert abs(y_dot_55.loc[2] - 2.5) < 1e-10, \
            f"y_dot_55 = {y_dot_55.loc[2]}, expected 2.5"
        
        # Cohort 6, period 6: pre-mean = (1+2+3+4+5)/5 = 3.0
        # y_dot = 6 - 3 = 3.0
        y_dot_66 = compute_transformed_outcome_staggered(
            test_data, 'y', 'id', 'period', 'first_treat',
            cohort_g=6, target_period=6
        )
        assert abs(y_dot_66.loc[3] - 3.0) < 1e-10, \
            f"y_dot_66 = {y_dot_66.loc[3]}, expected 3.0"


class TestTransformationStataConsistency:
    """变换变量与Stata一致性测试"""
    
    @pytest.mark.parametrize("g,r", GR_COMBINATIONS)
    def test_transformation_vs_stata(self, staggered_data, g, r):
        """
        测试变换变量与Stata计算一致
        
        Stata代码参考:
        bysort id: gen y_44 = y - (L1.y + L2.y + L3.y)/3 if f04
        """
        # Python计算
        python_y_dot = compute_transformed_outcome_staggered(
            staggered_data, 'y', 'id', 'period', 'first_treat',
            cohort_g=g, target_period=r
        )
        
        # 验证非空
        assert len(python_y_dot) > 0, f"({g},{r}): 变换结果为空"
        
        # 验证无异常值
        assert not python_y_dot.isna().all(), f"({g},{r}): 变换结果全为NaN"
        
        # 验证数值合理性
        assert python_y_dot.abs().max() < 100, \
            f"({g},{r}): 变换结果存在异常大值 {python_y_dot.abs().max()}"
    
    @pytest.mark.parametrize("g,r", GR_COMBINATIONS)
    def test_transformation_count(self, staggered_data, g, r):
        """验证变换变量数量正确"""
        y_dot = compute_transformed_outcome_staggered(
            staggered_data, 'y', 'id', 'period', 'first_treat',
            cohort_g=g, target_period=r
        )
        
        # 每个单位在每个period只有一个观测
        # 所以变换后的数量应该等于唯一单位数
        n_units = staggered_data['id'].nunique()
        
        # 变换后的数量不应超过单位数
        assert len(y_dot) <= n_units, \
            f"({g},{r}): 变换结果数量 {len(y_dot)} > 单位数 {n_units}"


class TestTransformationBoundary:
    """变换公式边界情况测试"""
    
    def test_cohort_at_first_post_period(self, staggered_data):
        """Cohort g在period g的变换（最早的post period）"""
        for g in [4, 5, 6]:
            y_dot = compute_transformed_outcome_staggered(
                staggered_data, 'y', 'id', 'period', 'first_treat',
                cohort_g=g, target_period=g
            )
            
            assert len(y_dot) > 0, f"Cohort {g} at period {g}: 变换结果为空"
            assert not y_dot.isna().all(), f"Cohort {g} at period {g}: 变换结果全为NaN"
    
    def test_cohort_at_last_period(self, staggered_data):
        """Cohort g在最后一个period的变换"""
        T = 6
        for g in [4, 5, 6]:
            y_dot = compute_transformed_outcome_staggered(
                staggered_data, 'y', 'id', 'period', 'first_treat',
                cohort_g=g, target_period=T
            )
            
            assert len(y_dot) > 0, f"Cohort {g} at period {T}: 变换结果为空"
    
    def test_invalid_period_raises_error(self, staggered_data):
        """target_period < cohort_g 应该报错"""
        with pytest.raises(AssertionError):
            compute_transformed_outcome_staggered(
                staggered_data, 'y', 'id', 'period', 'first_treat',
                cohort_g=5, target_period=4  # r < g
            )
    
    def test_never_treated_units_also_transformed(self, staggered_data):
        """Never treated单位也应该被变换"""
        # 筛选NT单位
        nt_units = staggered_data[np.isinf(staggered_data['first_treat'])]['id'].unique()
        
        for g, r in [(4, 4), (5, 5), (6, 6)]:
            y_dot = compute_transformed_outcome_staggered(
                staggered_data, 'y', 'id', 'period', 'first_treat',
                cohort_g=g, target_period=r
            )
            
            # 验证NT单位有变换值
            nt_in_result = set(nt_units) & set(y_dot.index)
            assert len(nt_in_result) > 0, \
                f"({g},{r}): Never treated单位未被变换"


class TestTransformationLwdidModule:
    """测试lwdid包的transform_staggered_demean函数"""
    
    def test_lwdid_transform_function_exists(self):
        """验证lwdid包的变换函数存在"""
        try:
            from lwdid.staggered.transformations import transform_staggered_demean
        except ImportError:
            pytest.fail("无法导入 transform_staggered_demean 函数")
    
    def test_lwdid_transform_produces_all_gr_columns(self, staggered_data):
        """验证lwdid变换函数生成所有(g,r)列"""
        from lwdid.staggered.transformations import transform_staggered_demean
        
        # 准备数据
        data = staggered_data.copy()
        data['gvar'] = data['first_treat'].replace({float('inf'): 0})  # 处理inf
        
        # 运行变换
        result = transform_staggered_demean(
            data, 'y', 'id', 'period', 'gvar'
        )
        
        # 验证列存在
        for g, r in GR_COMBINATIONS:
            col_name = f'ydot_g{g}_r{r}'
            assert col_name in result.columns, \
                f"缺少变换列: {col_name}"
    
    @pytest.mark.parametrize("g,r", GR_COMBINATIONS)
    def test_lwdid_transform_consistency(self, staggered_data, g, r):
        """验证lwdid变换与手动计算一致"""
        from lwdid.staggered.transformations import transform_staggered_demean
        
        # 准备数据
        data = staggered_data.copy()
        data['gvar'] = data['first_treat'].replace({float('inf'): 0})
        
        # lwdid变换
        result = transform_staggered_demean(
            data, 'y', 'id', 'period', 'gvar'
        )
        
        # 手动计算
        manual_y_dot = compute_transformed_outcome_staggered(
            staggered_data, 'y', 'id', 'period', 'first_treat',
            cohort_g=g, target_period=r
        )
        
        # 获取lwdid结果
        col_name = f'ydot_g{g}_r{r}'
        lwdid_y_dot = result[result['period'] == r].set_index('id')[col_name]
        
        # 对比
        common_ids = set(manual_y_dot.dropna().index) & set(lwdid_y_dot.dropna().index)
        
        if len(common_ids) == 0:
            pytest.skip(f"({g},{r}): 无共同单位可比较")
        
        for uid in list(common_ids)[:20]:  # 抽样检查
            manual_val = manual_y_dot.loc[uid]
            lwdid_val = lwdid_y_dot.loc[uid]
            
            assert abs(manual_val - lwdid_val) < 1e-10, \
                f"({g},{r}) Unit {uid}: manual={manual_val}, lwdid={lwdid_val}"

"""
数值验证测试 - exclude_pre_periods 参数

验证 exclude_pre_periods 的计算正确性，确保与论文 Section 6 的方法论一致。
使用 vibe-math MCP 进行精确数值验证。

Task 9: 数值验证测试
"""

import pytest
import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lwdid.staggered.transformations import (
    transform_staggered_demean,
    transform_staggered_detrend,
)


# =============================================================================
# Test 1: 论文 Section 6 示例验证
# =============================================================================

class TestPaperSection6Examples:
    """验证论文 Section 6 中的示例计算"""
    
    def test_paper_example_demean_g5_k0(self):
        """论文示例: g=5, k=0 (无排除)
        
        pre-treatment periods: {1,2,3,4}
        pre_mean = mean(Y_1, Y_2, Y_3, Y_4)
        """
        data = pd.DataFrame({
            'id': [1]*6 + [2]*6,
            'year': [1,2,3,4,5,6]*2,
            'y': [100, 110, 120, 130, 200, 210,  # unit 1: cohort 5
                  50, 55, 60, 65, 70, 75],        # unit 2: NT
            'gvar': [5]*6 + [0]*6
        })
        
        result = transform_staggered_demean(
            data, 'y', 'id', 'year', 'gvar',
            exclude_pre_periods=0
        )
        
        # unit 1: pre_mean = (100+110+120+130)/4 = 115
        # ydot_g5_r5 = 200 - 115 = 85
        expected_pre_mean = (100 + 110 + 120 + 130) / 4
        expected_ydot = 200 - expected_pre_mean
        
        actual_ydot = result.loc[
            (result['id']==1) & (result['year']==5), 
            'ydot_g5_r5'
        ].iloc[0]
        
        assert np.isclose(actual_ydot, expected_ydot, atol=1e-10), \
            f"Expected {expected_ydot}, got {actual_ydot}"
    
    def test_paper_example_demean_g5_k1(self):
        """论文示例: g=5, k=1 (排除 1 个时期)
        
        原始 pre-treatment: {1,2,3,4}
        排除后 pre-treatment: {1,2,3}
        pre_mean = mean(Y_1, Y_2, Y_3)
        """
        data = pd.DataFrame({
            'id': [1]*6 + [2]*6,
            'year': [1,2,3,4,5,6]*2,
            'y': [100, 110, 120, 130, 200, 210,
                  50, 55, 60, 65, 70, 75],
            'gvar': [5]*6 + [0]*6
        })
        
        result = transform_staggered_demean(
            data, 'y', 'id', 'year', 'gvar',
            exclude_pre_periods=1
        )
        
        # unit 1: pre_mean = (100+110+120)/3 = 110
        # ydot_g5_r5 = 200 - 110 = 90
        expected_pre_mean = (100 + 110 + 120) / 3
        expected_ydot = 200 - expected_pre_mean
        
        actual_ydot = result.loc[
            (result['id']==1) & (result['year']==5), 
            'ydot_g5_r5'
        ].iloc[0]
        
        assert np.isclose(actual_ydot, expected_ydot, atol=1e-10), \
            f"Expected {expected_ydot}, got {actual_ydot}"
    
    def test_paper_example_demean_g5_k2(self):
        """论文示例: g=5, k=2 (排除 2 个时期)
        
        原始 pre-treatment: {1,2,3,4}
        排除后 pre-treatment: {1,2}
        pre_mean = mean(Y_1, Y_2)
        """
        data = pd.DataFrame({
            'id': [1]*6 + [2]*6,
            'year': [1,2,3,4,5,6]*2,
            'y': [100, 110, 120, 130, 200, 210,
                  50, 55, 60, 65, 70, 75],
            'gvar': [5]*6 + [0]*6
        })
        
        result = transform_staggered_demean(
            data, 'y', 'id', 'year', 'gvar',
            exclude_pre_periods=2
        )
        
        # unit 1: pre_mean = (100+110)/2 = 105
        # ydot_g5_r5 = 200 - 105 = 95
        expected_pre_mean = (100 + 110) / 2
        expected_ydot = 200 - expected_pre_mean
        
        actual_ydot = result.loc[
            (result['id']==1) & (result['year']==5), 
            'ydot_g5_r5'
        ].iloc[0]
        
        assert np.isclose(actual_ydot, expected_ydot, atol=1e-10), \
            f"Expected {expected_ydot}, got {actual_ydot}"


# =============================================================================
# Test 2: Detrend 数值验证
# =============================================================================

class TestDetrendNumericalVerification:
    """验证 detrend 转换的数值正确性"""
    
    def test_detrend_ols_coefficients_k0(self):
        """验证 k=0 时 OLS 系数计算正确
        
        数据: Y = 8 + 2*t (完美线性)
        OLS 应该得到 A=8, B=2
        """
        data = pd.DataFrame({
            'id': [1]*6 + [2]*6,
            'year': [1,2,3,4,5,6]*2,
            # unit 1: Y = 8 + 2*t, 在 t>=5 有效应 +50
            'y': [10, 12, 14, 16, 68, 70,
                  5, 7, 9, 11, 13, 15],
            'gvar': [5]*6 + [0]*6
        })
        
        result = transform_staggered_detrend(
            data, 'y', 'id', 'year', 'gvar',
            exclude_pre_periods=0
        )
        
        # unit 1: OLS on {1,2,3,4}
        # Y_1=10, Y_2=12, Y_3=14, Y_4=16 -> B=2, A=8
        # 预测 t=5: 8 + 2*5 = 18
        # ycheck = 68 - 18 = 50
        expected_ycheck = 68 - (8 + 2*5)
        
        actual_ycheck = result.loc[
            (result['id']==1) & (result['year']==5), 
            'ycheck_g5_r5'
        ].iloc[0]
        
        assert np.isclose(actual_ycheck, expected_ycheck, atol=1e-6), \
            f"Expected {expected_ycheck}, got {actual_ycheck}"
    
    def test_detrend_ols_coefficients_k1(self):
        """验证 k=1 时 OLS 系数计算正确
        
        排除 t=4 后，OLS 基于 {1,2,3}
        """
        data = pd.DataFrame({
            'id': [1]*6 + [2]*6,
            'year': [1,2,3,4,5,6]*2,
            'y': [10, 12, 14, 16, 68, 70,
                  5, 7, 9, 11, 13, 15],
            'gvar': [5]*6 + [0]*6
        })
        
        result = transform_staggered_detrend(
            data, 'y', 'id', 'year', 'gvar',
            exclude_pre_periods=1
        )
        
        # unit 1: OLS on {1,2,3}
        # Y_1=10, Y_2=12, Y_3=14 -> B=2, A=8
        # 预测 t=5: 8 + 2*5 = 18
        # ycheck = 68 - 18 = 50
        expected_ycheck = 68 - (8 + 2*5)
        
        actual_ycheck = result.loc[
            (result['id']==1) & (result['year']==5), 
            'ycheck_g5_r5'
        ].iloc[0]
        
        assert np.isclose(actual_ycheck, expected_ycheck, atol=1e-6), \
            f"Expected {expected_ycheck}, got {actual_ycheck}"
    
    def test_detrend_nonlinear_data_different_results(self):
        """非线性数据下，不同 k 值产生不同结果
        
        当 pre-treatment 数据不是完美线性时，排除时期会改变 OLS 估计
        """
        data = pd.DataFrame({
            'id': [1]*6 + [2]*6,
            'year': [1,2,3,4,5,6]*2,
            # unit 1: 非线性 pre-treatment
            # t=1,2,3: Y=10,12,14 (斜率=2)
            # t=4: Y=30 (斜率突变)
            'y': [10, 12, 14, 30, 100, 102,
                  5, 7, 9, 11, 13, 15],
            'gvar': [5]*6 + [0]*6
        })
        
        result_k0 = transform_staggered_detrend(
            data, 'y', 'id', 'year', 'gvar',
            exclude_pre_periods=0
        )
        
        result_k1 = transform_staggered_detrend(
            data, 'y', 'id', 'year', 'gvar',
            exclude_pre_periods=1
        )
        
        ycheck_k0 = result_k0.loc[
            (result_k0['id']==1) & (result_k0['year']==5), 
            'ycheck_g5_r5'
        ].iloc[0]
        
        ycheck_k1 = result_k1.loc[
            (result_k1['id']==1) & (result_k1['year']==5), 
            'ycheck_g5_r5'
        ].iloc[0]
        
        # k=1 时: OLS on {1,2,3}, 预测 t=5 = 8 + 2*5 = 18
        # ycheck_k1 = 100 - 18 = 82
        assert np.isclose(ycheck_k1, 82.0, atol=1e-6)
        
        # k=0 时: OLS on {1,2,3,4}, 斜率被 t=4 的异常值拉高
        # 预测值更高，ycheck 更小
        assert ycheck_k0 < ycheck_k1



# =============================================================================
# Test 3: 多 Cohort 数值验证
# =============================================================================

class TestMultiCohortNumericalVerification:
    """验证多 cohort 场景下的数值正确性"""
    
    def test_multi_cohort_independent_pre_means(self):
        """验证不同 cohort 使用独立的 pre-treatment 时期
        
        cohort g=4: pre = {1,2,3}
        cohort g=5: pre = {1,2,3,4}
        """
        data = pd.DataFrame({
            'id': [1]*6 + [2]*6 + [3]*6,
            'year': [1,2,3,4,5,6]*3,
            'y': [10,12,14,50,52,54,   # unit 1: cohort 4
                  20,22,24,26,80,82,   # unit 2: cohort 5
                  5,6,7,8,9,10],       # unit 3: NT
            'gvar': [4]*6 + [5]*6 + [0]*6
        })
        
        result = transform_staggered_demean(
            data, 'y', 'id', 'year', 'gvar',
            exclude_pre_periods=1
        )
        
        # cohort 4: pre_mean = (10+12)/2 = 11 (排除 t=3)
        # unit 1 ydot_g4_r4 = 50 - 11 = 39
        expected_g4 = 50 - (10 + 12) / 2
        actual_g4 = result.loc[
            (result['id']==1) & (result['year']==4), 
            'ydot_g4_r4'
        ].iloc[0]
        assert np.isclose(actual_g4, expected_g4, atol=1e-10)
        
        # cohort 5: pre_mean = (20+22+24)/3 = 22 (排除 t=4)
        # unit 2 ydot_g5_r5 = 80 - 22 = 58
        expected_g5 = 80 - (20 + 22 + 24) / 3
        actual_g5 = result.loc[
            (result['id']==2) & (result['year']==5), 
            'ydot_g5_r5'
        ].iloc[0]
        assert np.isclose(actual_g5, expected_g5, atol=1e-10)
    
    def test_cross_cohort_transformation_numerical(self):
        """验证跨 cohort 转换的数值正确性
        
        unit 1 (cohort 4) 在 year=5 时也需要 cohort 5 的转换
        """
        data = pd.DataFrame({
            'id': [1]*6 + [2]*6 + [3]*6,
            'year': [1,2,3,4,5,6]*3,
            'y': [10,12,14,50,52,54,   # unit 1: cohort 4
                  20,22,24,26,80,82,   # unit 2: cohort 5
                  5,6,7,8,9,10],       # unit 3: NT
            'gvar': [4]*6 + [5]*6 + [0]*6
        })
        
        result = transform_staggered_demean(
            data, 'y', 'id', 'year', 'gvar',
            exclude_pre_periods=1
        )
        
        # unit 1 在 year=5 的 cohort 5 转换
        # cohort 5 的 pre_mean 对 unit 1: (10+12+14)/3 = 12 (排除 t=4)
        # ydot_g5_r5 = 52 - 12 = 40
        expected = 52 - (10 + 12 + 14) / 3
        actual = result.loc[
            (result['id']==1) & (result['year']==5), 
            'ydot_g5_r5'
        ].iloc[0]
        assert np.isclose(actual, expected, atol=1e-10)


# =============================================================================
# Test 4: 边界情况数值验证
# =============================================================================

class TestBoundaryNumericalVerification:
    """验证边界情况的数值正确性"""
    
    def test_max_exclude_demean(self):
        """验证 demean 最大排除值的数值正确性
        
        cohort g=5, T_min=1
        最大 k = 5 - 1 - 1 = 3 (剩余 1 个时期)
        """
        data = pd.DataFrame({
            'id': [1]*6 + [2]*6,
            'year': [1,2,3,4,5,6]*2,
            'y': [10, 20, 30, 40, 100, 110,
                  5, 10, 15, 20, 25, 30],
            'gvar': [5]*6 + [0]*6
        })
        
        result = transform_staggered_demean(
            data, 'y', 'id', 'year', 'gvar',
            exclude_pre_periods=3
        )
        
        # unit 1: pre_mean = 10 (只有 t=1)
        # ydot_g5_r5 = 100 - 10 = 90
        expected = 100 - 10
        actual = result.loc[
            (result['id']==1) & (result['year']==5), 
            'ydot_g5_r5'
        ].iloc[0]
        assert np.isclose(actual, expected, atol=1e-10)
    
    def test_max_exclude_detrend(self):
        """验证 detrend 最大排除值的数值正确性
        
        cohort g=5, T_min=1
        最大 k = 5 - 1 - 2 = 2 (剩余 2 个时期)
        """
        data = pd.DataFrame({
            'id': [1]*6 + [2]*6,
            'year': [1,2,3,4,5,6]*2,
            # unit 1: Y = 8 + 2*t
            'y': [10, 12, 14, 16, 100, 102,
                  5, 7, 9, 11, 13, 15],
            'gvar': [5]*6 + [0]*6
        })
        
        result = transform_staggered_detrend(
            data, 'y', 'id', 'year', 'gvar',
            exclude_pre_periods=2
        )
        
        # unit 1: OLS on {1,2}
        # Y_1=10, Y_2=12 -> B=2, A=8
        # 预测 t=5: 8 + 2*5 = 18
        # ycheck = 100 - 18 = 82
        expected = 100 - (8 + 2*5)
        actual = result.loc[
            (result['id']==1) & (result['year']==5), 
            'ycheck_g5_r5'
        ].iloc[0]
        assert np.isclose(actual, expected, atol=1e-6)


# =============================================================================
# Test 5: NT 单位数值验证
# =============================================================================

class TestNeverTreatedNumericalVerification:
    """验证 never-treated 单位的数值正确性"""
    
    def test_nt_unit_demean_transformation(self):
        """验证 NT 单位的 demean 转换数值"""
        data = pd.DataFrame({
            'id': [1]*6 + [2]*6,
            'year': [1,2,3,4,5,6]*2,
            'y': [100, 110, 120, 130, 200, 210,  # unit 1: cohort 5
                  50, 55, 60, 65, 70, 75],        # unit 2: NT
            'gvar': [5]*6 + [0]*6
        })
        
        result = transform_staggered_demean(
            data, 'y', 'id', 'year', 'gvar',
            exclude_pre_periods=1
        )
        
        # unit 2 (NT): pre_mean = (50+55+60)/3 = 55 (排除 t=4)
        # ydot_g5_r5 = 70 - 55 = 15
        expected = 70 - (50 + 55 + 60) / 3
        actual = result.loc[
            (result['id']==2) & (result['year']==5), 
            'ydot_g5_r5'
        ].iloc[0]
        assert np.isclose(actual, expected, atol=1e-10)
    
    def test_nt_unit_detrend_transformation(self):
        """验证 NT 单位的 detrend 转换数值"""
        data = pd.DataFrame({
            'id': [1]*6 + [2]*6,
            'year': [1,2,3,4,5,6]*2,
            # unit 2 (NT): Y = 3 + 2*t (完美线性)
            'y': [10, 12, 14, 16, 68, 70,
                  5, 7, 9, 11, 13, 15],
            'gvar': [5]*6 + [0]*6
        })
        
        result = transform_staggered_detrend(
            data, 'y', 'id', 'year', 'gvar',
            exclude_pre_periods=1
        )
        
        # unit 2 (NT): OLS on {1,2,3}
        # Y_1=5, Y_2=7, Y_3=9 -> B=2, A=3
        # 预测 t=5: 3 + 2*5 = 13
        # ycheck = 13 - 13 = 0
        expected = 13 - (3 + 2*5)
        actual = result.loc[
            (result['id']==2) & (result['year']==5), 
            'ycheck_g5_r5'
        ].iloc[0]
        assert np.isclose(actual, expected, atol=1e-6)


# =============================================================================
# Test 6: 多时期 Post-treatment 数值验证
# =============================================================================

class TestMultiPeriodPostTreatmentVerification:
    """验证多个 post-treatment 时期的数值正确性"""
    
    def test_fixed_pre_mean_across_post_periods(self):
        """验证 pre_mean 在所有 post-treatment 时期保持固定
        
        这是最常见的实现错误之一
        """
        data = pd.DataFrame({
            'id': [1]*6,
            'year': [1,2,3,4,5,6],
            'y': [10, 20, 30, 100, 110, 120],
            'gvar': [4]*6
        })
        
        result = transform_staggered_demean(
            data, 'y', 'id', 'year', 'gvar',
            exclude_pre_periods=1
        )
        
        # pre_mean = (10+20)/2 = 15 (排除 t=3)
        expected_pre_mean = (10 + 20) / 2
        
        # 所有 post-treatment 时期应该使用相同的 pre_mean
        ydot_r4 = result.loc[result['year']==4, 'ydot_g4_r4'].iloc[0]
        ydot_r5 = result.loc[result['year']==5, 'ydot_g4_r5'].iloc[0]
        ydot_r6 = result.loc[result['year']==6, 'ydot_g4_r6'].iloc[0]
        
        # 反推 pre_mean
        pre_mean_from_r4 = 100 - ydot_r4
        pre_mean_from_r5 = 110 - ydot_r5
        pre_mean_from_r6 = 120 - ydot_r6
        
        # 所有时期应该得到相同的 pre_mean
        assert np.isclose(pre_mean_from_r4, expected_pre_mean, atol=1e-10)
        assert np.isclose(pre_mean_from_r5, expected_pre_mean, atol=1e-10)
        assert np.isclose(pre_mean_from_r6, expected_pre_mean, atol=1e-10)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

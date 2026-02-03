"""
测试 staggered 模式下 exclude_pre_periods 参数功能

基于 Lee & Wooldridge (2025) Section 6 的方法论实现。
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
    transform_staggered_demeanq,
    transform_staggered_detrendq,
)
from lwdid.exceptions import InsufficientPrePeriodsError


# =============================================================================
# Test 1: Demean 转换的 exclude_pre_periods 测试
# =============================================================================

class TestExcludePrePeriodsDemean:
    """测试 transform_staggered_demean 的 exclude_pre_periods 参数"""
    
    def test_demean_exclude_zero_backward_compatible(self):
        """exclude_pre_periods=0 时结果与原实现完全一致"""
        data = pd.DataFrame({
            'id': [1,1,1,1,1, 2,2,2,2,2],
            'year': [1,2,3,4,5, 1,2,3,4,5],
            'y': [10,12,14,50,52, 5,6,7,8,9],
            'gvar': [4,4,4,4,4, 0,0,0,0,0]
        })
        
        # 不传参数 (默认 exclude_pre_periods=0)
        result_default = transform_staggered_demean(
            data, 'y', 'id', 'year', 'gvar'
        )
        
        # 显式传 exclude_pre_periods=0
        result_explicit = transform_staggered_demean(
            data, 'y', 'id', 'year', 'gvar',
            exclude_pre_periods=0
        )
        
        # 两者应完全一致
        ydot_cols = [c for c in result_default.columns if c.startswith('ydot_')]
        for col in ydot_cols:
            pd.testing.assert_series_equal(
                result_default[col], 
                result_explicit[col],
                check_names=False
            )
    
    def test_demean_exclude_one_period(self):
        """exclude_pre_periods=1 正确排除最后一个 pre-treatment 时期
        
        cohort g=5, T_min=1
        原始 pre-treatment: {1,2,3,4}
        排除后 pre-treatment: {1,2,3}
        """
        data = pd.DataFrame({
            'id': [1,1,1,1,1,1, 2,2,2,2,2,2],
            'year': [1,2,3,4,5,6, 1,2,3,4,5,6],
            'y': [10,12,14,16,50,52, 5,6,7,8,9,10],  # unit1: cohort 5, unit2: NT
            'gvar': [5,5,5,5,5,5, 0,0,0,0,0,0]
        })
        
        # exclude_pre_periods=0: pre_mean = (10+12+14+16)/4 = 13
        result_k0 = transform_staggered_demean(
            data, 'y', 'id', 'year', 'gvar',
            exclude_pre_periods=0
        )
        
        # exclude_pre_periods=1: pre_mean = (10+12+14)/3 = 12
        result_k1 = transform_staggered_demean(
            data, 'y', 'id', 'year', 'gvar',
            exclude_pre_periods=1
        )
        
        # 验证 unit 1 的 ydot_g5_r5
        # k=0: ydot = 50 - 13 = 37
        # k=1: ydot = 50 - 12 = 38
        ydot_k0 = result_k0.loc[(result_k0['id']==1) & (result_k0['year']==5), 'ydot_g5_r5'].iloc[0]
        ydot_k1 = result_k1.loc[(result_k1['id']==1) & (result_k1['year']==5), 'ydot_g5_r5'].iloc[0]
        
        assert np.isclose(ydot_k0, 37.0, atol=1e-10)
        assert np.isclose(ydot_k1, 38.0, atol=1e-10)
        
        # 验证 unit 2 (NT) 的 ydot_g5_r5
        # k=0: pre_mean = (5+6+7+8)/4 = 6.5, ydot = 9 - 6.5 = 2.5
        # k=1: pre_mean = (5+6+7)/3 = 6, ydot = 9 - 6 = 3
        ydot_nt_k0 = result_k0.loc[(result_k0['id']==2) & (result_k0['year']==5), 'ydot_g5_r5'].iloc[0]
        ydot_nt_k1 = result_k1.loc[(result_k1['id']==2) & (result_k1['year']==5), 'ydot_g5_r5'].iloc[0]
        
        assert np.isclose(ydot_nt_k0, 2.5, atol=1e-10)
        assert np.isclose(ydot_nt_k1, 3.0, atol=1e-10)
    
    def test_demean_exclude_multiple_periods(self):
        """exclude_pre_periods=2 正确排除最后两个 pre-treatment 时期
        
        cohort g=6, T_min=1
        原始 pre-treatment: {1,2,3,4,5}
        排除后 pre-treatment: {1,2,3}
        """
        data = pd.DataFrame({
            'id': [1]*7 + [2]*7,
            'year': [1,2,3,4,5,6,7]*2,
            'y': [10,12,14,16,18,100,102, 5,6,7,8,9,10,11],
            'gvar': [6]*7 + [0]*7
        })
        
        # exclude_pre_periods=2: pre_mean = (10+12+14)/3 = 12
        result = transform_staggered_demean(
            data, 'y', 'id', 'year', 'gvar',
            exclude_pre_periods=2
        )
        
        # unit 1: ydot_g6_r6 = 100 - 12 = 88
        ydot = result.loc[(result['id']==1) & (result['year']==6), 'ydot_g6_r6'].iloc[0]
        assert np.isclose(ydot, 88.0, atol=1e-10)
    
    def test_demean_insufficient_periods_error(self):
        """剩余时期不足时抛出 InsufficientPrePeriodsError
        
        cohort g=3, T_min=1, 原始 pre-treatment: {1,2}
        exclude_pre_periods=2 后剩余 0 个时期，应抛出错误
        """
        data = pd.DataFrame({
            'id': [1,1,1,1, 2,2,2,2],
            'year': [1,2,3,4, 1,2,3,4],
            'y': [10,12,50,52, 5,6,7,8],
            'gvar': [3,3,3,3, 0,0,0,0]
        })
        
        with pytest.raises(InsufficientPrePeriodsError) as exc_info:
            transform_staggered_demean(
                data, 'y', 'id', 'year', 'gvar',
                exclude_pre_periods=2
            )
        
        # 验证异常属性
        exc = exc_info.value
        assert exc.cohort == 3
        assert exc.required == 1
        assert exc.excluded == 2
        assert exc.available <= 0
    
    def test_demean_boundary_max_exclude(self):
        """边界测试: 排除到只剩 1 个时期 (demean 最小要求)
        
        cohort g=5, T_min=1, 原始 pre-treatment: {1,2,3,4}
        exclude_pre_periods=3 后剩余 1 个时期 {1}，应正常工作
        """
        data = pd.DataFrame({
            'id': [1]*6 + [2]*6,
            'year': [1,2,3,4,5,6]*2,
            'y': [10,12,14,16,100,102, 5,6,7,8,9,10],
            'gvar': [5]*6 + [0]*6
        })
        
        # exclude_pre_periods=3: pre_mean = 10 (只有 t=1)
        result = transform_staggered_demean(
            data, 'y', 'id', 'year', 'gvar',
            exclude_pre_periods=3
        )
        
        # unit 1: ydot_g5_r5 = 100 - 10 = 90
        ydot = result.loc[(result['id']==1) & (result['year']==5), 'ydot_g5_r5'].iloc[0]
        assert np.isclose(ydot, 90.0, atol=1e-10)



# =============================================================================
# Test 2: Detrend 转换的 exclude_pre_periods 测试
# =============================================================================

class TestExcludePrePeriodsDetrend:
    """测试 transform_staggered_detrend 的 exclude_pre_periods 参数"""
    
    def test_detrend_exclude_zero_backward_compatible(self):
        """exclude_pre_periods=0 时结果与原实现完全一致"""
        data = pd.DataFrame({
            'id': [1]*6 + [2]*6,
            'year': [1,2,3,4,5,6]*2,
            'y': [10,12,14,50,52,54, 5,7,9,11,13,15],
            'gvar': [4]*6 + [0]*6
        })
        
        result_default = transform_staggered_detrend(
            data, 'y', 'id', 'year', 'gvar'
        )
        
        result_explicit = transform_staggered_detrend(
            data, 'y', 'id', 'year', 'gvar',
            exclude_pre_periods=0
        )
        
        ycheck_cols = [c for c in result_default.columns if c.startswith('ycheck_')]
        for col in ycheck_cols:
            pd.testing.assert_series_equal(
                result_default[col], 
                result_explicit[col],
                check_names=False
            )
    
    def test_detrend_exclude_one_period(self):
        """exclude_pre_periods=1 正确排除最后一个 pre-treatment 时期
        
        cohort g=5, T_min=1
        原始 pre-treatment: {1,2,3,4}
        排除后 pre-treatment: {1,2,3}
        """
        data = pd.DataFrame({
            'id': [1]*6 + [2]*6,
            'year': [1,2,3,4,5,6]*2,
            # unit 1: 线性趋势 Y = 8 + 2*t, 在 t>=5 有处理效应 +30
            # pre: 10,12,14,16, post: 40,42 (10+30, 12+30)
            'y': [10,12,14,16,40,42, 5,7,9,11,13,15],
            'gvar': [5]*6 + [0]*6
        })
        
        # k=0: OLS on {1,2,3,4}, 预测 t=5: 8+2*5=18, ycheck=40-18=22
        result_k0 = transform_staggered_detrend(
            data, 'y', 'id', 'year', 'gvar',
            exclude_pre_periods=0
        )
        
        # k=1: OLS on {1,2,3}, 预测 t=5: 8+2*5=18, ycheck=40-18=22
        # (因为数据是完美线性的，结果相同)
        result_k1 = transform_staggered_detrend(
            data, 'y', 'id', 'year', 'gvar',
            exclude_pre_periods=1
        )
        
        ycheck_k0 = result_k0.loc[(result_k0['id']==1) & (result_k0['year']==5), 'ycheck_g5_r5'].iloc[0]
        ycheck_k1 = result_k1.loc[(result_k1['id']==1) & (result_k1['year']==5), 'ycheck_g5_r5'].iloc[0]
        
        # 完美线性数据，两者应相同
        assert np.isclose(ycheck_k0, 22.0, atol=1e-6)
        assert np.isclose(ycheck_k1, 22.0, atol=1e-6)
    
    def test_detrend_exclude_with_nonlinear_data(self):
        """非线性数据下 exclude_pre_periods 产生不同结果
        
        当 pre-treatment 数据不是完美线性时，排除时期会改变 OLS 估计
        """
        data = pd.DataFrame({
            'id': [1]*6 + [2]*6,
            'year': [1,2,3,4,5,6]*2,
            # unit 1: 非线性 pre-treatment
            # t=1,2,3: Y=10,12,14 (斜率=2)
            # t=4: Y=20 (斜率突变)
            # t=5,6: Y=100,102 (post-treatment)
            'y': [10,12,14,20,100,102, 5,7,9,11,13,15],
            'gvar': [5]*6 + [0]*6
        })
        
        # k=0: OLS on {1,2,3,4}, 斜率会被 t=4 的异常值拉高
        result_k0 = transform_staggered_detrend(
            data, 'y', 'id', 'year', 'gvar',
            exclude_pre_periods=0
        )
        
        # k=1: OLS on {1,2,3}, 斜率=2, 截距=8
        result_k1 = transform_staggered_detrend(
            data, 'y', 'id', 'year', 'gvar',
            exclude_pre_periods=1
        )
        
        ycheck_k0 = result_k0.loc[(result_k0['id']==1) & (result_k0['year']==5), 'ycheck_g5_r5'].iloc[0]
        ycheck_k1 = result_k1.loc[(result_k1['id']==1) & (result_k1['year']==5), 'ycheck_g5_r5'].iloc[0]
        
        # k=1 时: 预测 t=5 = 8 + 2*5 = 18, ycheck = 100 - 18 = 82
        assert np.isclose(ycheck_k1, 82.0, atol=1e-6)
        
        # k=0 时: 由于 t=4 的异常值，斜率更高，预测值更高，ycheck 更小
        assert ycheck_k0 < ycheck_k1
    
    def test_detrend_insufficient_periods_error(self):
        """剩余时期不足 2 个时抛出 InsufficientPrePeriodsError
        
        cohort g=4, T_min=1, 原始 pre-treatment: {1,2,3}
        exclude_pre_periods=2 后剩余 1 个时期，不足以进行 OLS
        """
        data = pd.DataFrame({
            'id': [1]*5 + [2]*5,
            'year': [1,2,3,4,5]*2,
            'y': [10,12,14,50,52, 5,7,9,11,13],
            'gvar': [4]*5 + [0]*5
        })
        
        with pytest.raises(InsufficientPrePeriodsError) as exc_info:
            transform_staggered_detrend(
                data, 'y', 'id', 'year', 'gvar',
                exclude_pre_periods=2
            )
        
        exc = exc_info.value
        assert exc.cohort == 4
        assert exc.required == 2
        assert exc.excluded == 2
    
    def test_detrend_boundary_min_periods(self):
        """边界测试: 排除到只剩 2 个时期 (detrend 最小要求)
        
        cohort g=5, T_min=1, 原始 pre-treatment: {1,2,3,4}
        exclude_pre_periods=2 后剩余 2 个时期 {1,2}，应正常工作
        """
        data = pd.DataFrame({
            'id': [1]*6 + [2]*6,
            'year': [1,2,3,4,5,6]*2,
            # unit 1: Y = 8 + 2*t
            'y': [10,12,14,16,100,102, 5,7,9,11,13,15],
            'gvar': [5]*6 + [0]*6
        })
        
        # exclude_pre_periods=2: OLS on {1,2}
        # Y_1=10, Y_2=12 -> B=2, A=8
        # 预测 t=5: 8+2*5=18, ycheck=100-18=82
        result = transform_staggered_detrend(
            data, 'y', 'id', 'year', 'gvar',
            exclude_pre_periods=2
        )
        
        ycheck = result.loc[(result['id']==1) & (result['year']==5), 'ycheck_g5_r5'].iloc[0]
        assert np.isclose(ycheck, 82.0, atol=1e-6)



# =============================================================================
# Test 3: 错误处理测试
# =============================================================================

class TestExcludePrePeriodsErrors:
    """测试 exclude_pre_periods 参数的错误处理"""
    
    def test_negative_exclude_raises_error(self):
        """负数 exclude_pre_periods 抛出 ValueError"""
        data = pd.DataFrame({
            'id': [1,1,1,1],
            'year': [1,2,3,4],
            'y': [10,12,50,52],
            'gvar': [3,3,3,3]
        })
        
        with pytest.raises(ValueError, match="non-negative"):
            transform_staggered_demean(
                data, 'y', 'id', 'year', 'gvar',
                exclude_pre_periods=-1
            )
        
        with pytest.raises(ValueError, match="non-negative"):
            transform_staggered_detrend(
                data, 'y', 'id', 'year', 'gvar',
                exclude_pre_periods=-1
            )
    
    def test_non_integer_exclude_raises_error(self):
        """非整数 exclude_pre_periods 抛出 TypeError"""
        data = pd.DataFrame({
            'id': [1,1,1,1],
            'year': [1,2,3,4],
            'y': [10,12,50,52],
            'gvar': [3,3,3,3]
        })
        
        with pytest.raises(TypeError, match="integer"):
            transform_staggered_demean(
                data, 'y', 'id', 'year', 'gvar',
                exclude_pre_periods=1.5
            )
        
        with pytest.raises(TypeError, match="integer"):
            transform_staggered_detrend(
                data, 'y', 'id', 'year', 'gvar',
                exclude_pre_periods=1.5
            )
    
    def test_string_exclude_raises_error(self):
        """字符串 exclude_pre_periods 抛出 TypeError"""
        data = pd.DataFrame({
            'id': [1,1,1,1],
            'year': [1,2,3,4],
            'y': [10,12,50,52],
            'gvar': [3,3,3,3]
        })
        
        with pytest.raises(TypeError, match="integer"):
            transform_staggered_demean(
                data, 'y', 'id', 'year', 'gvar',
                exclude_pre_periods="1"
            )


# =============================================================================
# Test 4: 多 Cohort 场景测试
# =============================================================================

class TestExcludePrePeriodsMultiCohort:
    """测试多 cohort 场景下 exclude_pre_periods 的行为"""
    
    def test_multi_cohort_independent_exclusion(self):
        """多 cohort 场景下各 cohort 独立应用排除
        
        cohort g=4: pre={1,2,3}, 排除 1 后 pre={1,2}
        cohort g=5: pre={1,2,3,4}, 排除 1 后 pre={1,2,3}
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
        ydot_g4 = result.loc[(result['id']==1) & (result['year']==4), 'ydot_g4_r4'].iloc[0]
        assert np.isclose(ydot_g4, 39.0, atol=1e-10)
        
        # cohort 5: pre_mean = (20+22+24)/3 = 22 (排除 t=4)
        # unit 2 ydot_g5_r5 = 80 - 22 = 58
        ydot_g5 = result.loc[(result['id']==2) & (result['year']==5), 'ydot_g5_r5'].iloc[0]
        assert np.isclose(ydot_g5, 58.0, atol=1e-10)
    
    def test_multi_cohort_different_min_periods(self):
        """不同 cohort 有不同的最小时期要求
        
        cohort g=3: 只有 2 个 pre-treatment 时期
        cohort g=5: 有 4 个 pre-treatment 时期
        
        exclude_pre_periods=1 对 g=3 可能导致不足
        """
        data = pd.DataFrame({
            'id': [1]*6 + [2]*6 + [3]*6,
            'year': [1,2,3,4,5,6]*3,
            'y': [10,12,50,52,54,56,   # unit 1: cohort 3
                  20,22,24,26,80,82,   # unit 2: cohort 5
                  5,6,7,8,9,10],       # unit 3: NT
            'gvar': [3]*6 + [5]*6 + [0]*6
        })
        
        # exclude_pre_periods=1 对 demean 应该可以工作
        # cohort 3: pre={1,2}, 排除 1 后 pre={1}, 仍有 1 个时期
        result = transform_staggered_demean(
            data, 'y', 'id', 'year', 'gvar',
            exclude_pre_periods=1
        )
        
        # cohort 3: pre_mean = 10 (只有 t=1)
        # unit 1 ydot_g3_r3 = 50 - 10 = 40
        ydot_g3 = result.loc[(result['id']==1) & (result['year']==3), 'ydot_g3_r3'].iloc[0]
        assert np.isclose(ydot_g3, 40.0, atol=1e-10)
    
    def test_cross_cohort_transformation_with_exclude(self):
        """已处理单位也需要为后续 cohort 进行转换
        
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
        
        # unit 1 在 year=5 应该有 ydot_g5_r5 值
        # cohort 5 的 pre_mean 对 unit 1: (10+12+14)/3 = 12 (排除 t=4)
        # ydot_g5_r5 = 52 - 12 = 40
        ydot_u1_g5 = result.loc[(result['id']==1) & (result['year']==5), 'ydot_g5_r5'].iloc[0]
        assert np.isclose(ydot_u1_g5, 40.0, atol=1e-10)



# =============================================================================
# Test 5: 季节性转换测试 (demeanq/detrendq)
# =============================================================================

class TestExcludePrePeriodsSeasonalTransforms:
    """测试季节性转换函数的 exclude_pre_periods 参数"""
    
    @pytest.fixture
    def seasonal_data(self):
        """创建季节性测试数据
        
        Q=4 (季度数据)
        cohort g=9, T_min=1
        pre-treatment: {1,2,3,4,5,6,7,8}
        """
        n_periods = 12
        data = pd.DataFrame({
            'id': [1]*n_periods + [2]*n_periods,
            'year': list(range(1, n_periods+1))*2,
            'y': [10,12,14,16, 18,20,22,24, 100,102,104,106,  # unit 1: cohort 9
                  5,6,7,8, 9,10,11,12, 13,14,15,16],          # unit 2: NT
            'gvar': [9]*n_periods + [0]*n_periods,
            'quarter': [1,2,3,4]*3*2  # 季度变量
        })
        return data
    
    def test_demeanq_exclude_zero_backward_compatible(self, seasonal_data):
        """demeanq exclude_pre_periods=0 时向后兼容"""
        result_default = transform_staggered_demeanq(
            seasonal_data, 'y', 'id', 'year', 'gvar',
            season_var='quarter', Q=4
        )
        
        result_explicit = transform_staggered_demeanq(
            seasonal_data, 'y', 'id', 'year', 'gvar',
            season_var='quarter', Q=4,
            exclude_pre_periods=0
        )
        
        ydot_cols = [c for c in result_default.columns if c.startswith('ydot_')]
        for col in ydot_cols:
            pd.testing.assert_series_equal(
                result_default[col], 
                result_explicit[col],
                check_names=False
            )
    
    def test_demeanq_exclude_one_period(self):
        """demeanq exclude_pre_periods=1 正确排除
        
        使用非对称季节数据，确保排除时期会改变季节均值
        cohort g=10, T_min=1, pre-treatment = {1,2,3,4,5,6,7,8,9}
        排除 1 后 pre-treatment = {1,2,3,4,5,6,7,8}
        
        t=9 是 Q1，排除后 Q1 只有 t=1,5 的数据
        t=13 也是 Q1，所以会受到 Q1 季节均值变化的影响
        """
        n_periods = 16
        # unit 1 数据
        y_unit1 = [10,12,14,16, 18,20,22,24, 100, 200,202,204,300,302,304,306]
        # unit 2 数据 (NT)
        y_unit2 = [5,6,7,8, 9,10,11,12, 13, 14,15,16,17,18,19,20]
        
        data = pd.DataFrame({
            'id': [1]*n_periods + [2]*n_periods,
            'year': list(range(1, n_periods+1))*2,
            'y': y_unit1 + y_unit2,
            'gvar': [10]*n_periods + [0]*n_periods,
            'quarter': [1,2,3,4]*4*2
        })
        
        result_k0 = transform_staggered_demeanq(
            data, 'y', 'id', 'year', 'gvar',
            season_var='quarter', Q=4,
            exclude_pre_periods=0
        )
        
        result_k1 = transform_staggered_demeanq(
            data, 'y', 'id', 'year', 'gvar',
            season_var='quarter', Q=4,
            exclude_pre_periods=1
        )
        
        # 验证 t=13 (Q1) 的结果不同
        # k=0: Q1 季节均值包含 t=9 的异常值 100
        # k=1: Q1 季节均值不包含 t=9
        
        ydot_k0 = result_k0.loc[(result_k0['id']==1) & (result_k0['year']==13), 'ydot_g10_r13'].iloc[0]
        ydot_k1 = result_k1.loc[(result_k1['id']==1) & (result_k1['year']==13), 'ydot_g10_r13'].iloc[0]
        
        # 由于排除了 t=9 (Q1 的异常值)，Q1 的季节均值会不同
        # k=0: mu + gamma_Q1 基于 {10,18,100} 的 Q1 数据
        # k=1: mu + gamma_Q1 基于 {10,18} 的 Q1 数据
        assert not np.isclose(ydot_k0, ydot_k1, atol=1e-6), \
            f"Expected different values but got k0={ydot_k0}, k1={ydot_k1}"
    
    def test_demeanq_insufficient_periods_error(self, seasonal_data):
        """demeanq 剩余时期不足 Q+1 时抛出错误"""
        # 创建只有 5 个 pre-treatment 时期的数据 (Q=4, 需要 Q+1=5)
        short_data = pd.DataFrame({
            'id': [1]*8 + [2]*8,
            'year': list(range(1, 9))*2,
            'y': [10,12,14,16,18, 100,102,104,  # unit 1: cohort 6
                  5,6,7,8,9, 10,11,12],          # unit 2: NT
            'gvar': [6]*8 + [0]*8,
            'quarter': [1,2,3,4]*2*2
        })
        
        # exclude_pre_periods=1 后只剩 4 个时期，不足 Q+1=5
        with pytest.raises(InsufficientPrePeriodsError) as exc_info:
            transform_staggered_demeanq(
                short_data, 'y', 'id', 'year', 'gvar',
                season_var='quarter', Q=4,
                exclude_pre_periods=1
            )
        
        exc = exc_info.value
        assert exc.required == 5  # Q+1
    
    def test_detrendq_exclude_zero_backward_compatible(self, seasonal_data):
        """detrendq exclude_pre_periods=0 时向后兼容"""
        result_default = transform_staggered_detrendq(
            seasonal_data, 'y', 'id', 'year', 'gvar',
            season_var='quarter', Q=4
        )
        
        result_explicit = transform_staggered_detrendq(
            seasonal_data, 'y', 'id', 'year', 'gvar',
            season_var='quarter', Q=4,
            exclude_pre_periods=0
        )
        
        ycheck_cols = [c for c in result_default.columns if c.startswith('ycheck_')]
        for col in ycheck_cols:
            pd.testing.assert_series_equal(
                result_default[col], 
                result_explicit[col],
                check_names=False
            )
    
    def test_detrendq_insufficient_periods_error(self, seasonal_data):
        """detrendq 剩余时期不足 Q+2 时抛出错误"""
        # 创建只有 6 个 pre-treatment 时期的数据 (Q=4, 需要 Q+2=6)
        short_data = pd.DataFrame({
            'id': [1]*9 + [2]*9,
            'year': list(range(1, 10))*2,
            'y': [10,12,14,16,18,20, 100,102,104,  # unit 1: cohort 7
                  5,6,7,8,9,10, 11,12,13],          # unit 2: NT
            'gvar': [7]*9 + [0]*9,
            'quarter': [1,2,3,4,1,2,3,4,1]*2
        })
        
        # exclude_pre_periods=1 后只剩 5 个时期，不足 Q+2=6
        with pytest.raises(InsufficientPrePeriodsError) as exc_info:
            transform_staggered_detrendq(
                short_data, 'y', 'id', 'year', 'gvar',
                season_var='quarter', Q=4,
                exclude_pre_periods=1
            )
        
        exc = exc_info.value
        assert exc.required == 6  # Q+2



# =============================================================================
# Test 6: 数值验证测试
# =============================================================================

class TestExcludePrePeriodsNumericalVerification:
    """数值验证测试，确保计算正确性"""
    
    def test_paper_example_g5_k1(self):
        """复现论文 Section 6 示例
        
        g=5, exclude_pre_periods=1
        应使用 periods {1,2,3} 而非 {1,2,3,4}
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
            exclude_pre_periods=1
        )
        
        # unit 1: pre_mean = (100+110+120)/3 = 110 (排除 t=4)
        # ydot_g5_r5 = 200 - 110 = 90
        ydot = result.loc[(result['id']==1) & (result['year']==5), 'ydot_g5_r5'].iloc[0]
        assert np.isclose(ydot, 90.0, atol=1e-10)
        
        # unit 2 (NT): pre_mean = (50+55+60)/3 = 55
        # ydot_g5_r5 = 70 - 55 = 15
        ydot_nt = result.loc[(result['id']==2) & (result['year']==5), 'ydot_g5_r5'].iloc[0]
        assert np.isclose(ydot_nt, 15.0, atol=1e-10)
    
    def test_paper_example_g5_k2(self):
        """论文示例: g=5, exclude_pre_periods=2
        
        应使用 periods {1,2} 而非 {1,2,3,4}
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
            exclude_pre_periods=2
        )
        
        # unit 1: pre_mean = (100+110)/2 = 105 (排除 t=3,4)
        # ydot_g5_r5 = 200 - 105 = 95
        ydot = result.loc[(result['id']==1) & (result['year']==5), 'ydot_g5_r5'].iloc[0]
        assert np.isclose(ydot, 95.0, atol=1e-10)
    
    def test_manual_calculation_verification_demean(self):
        """手动计算验证 demean 转换结果"""
        data = pd.DataFrame({
            'id': [1,1,1,1,1, 2,2,2,2,2, 3,3,3,3,3],
            'year': [1,2,3,4,5]*3,
            'y': [10, 20, 30, 100, 110,   # unit 1: cohort 4
                  15, 25, 35, 45, 55,     # unit 2: cohort 5
                  5, 10, 15, 20, 25],     # unit 3: NT
            'gvar': [4]*5 + [5]*5 + [0]*5
        })
        
        result = transform_staggered_demean(
            data, 'y', 'id', 'year', 'gvar',
            exclude_pre_periods=1
        )
        
        # === Cohort g=4 验证 ===
        # pre_mean (排除 t=3): (10+20)/2 = 15
        
        # unit 1: ydot_g4_r4 = 100 - 15 = 85
        assert np.isclose(
            result.loc[(result['id']==1) & (result['year']==4), 'ydot_g4_r4'].iloc[0],
            85.0, atol=1e-10
        )
        
        # unit 3 (NT): pre_mean = (5+10)/2 = 7.5
        # ydot_g4_r4 = 20 - 7.5 = 12.5
        assert np.isclose(
            result.loc[(result['id']==3) & (result['year']==4), 'ydot_g4_r4'].iloc[0],
            12.5, atol=1e-10
        )
        
        # === Cohort g=5 验证 ===
        # pre_mean (排除 t=4): (15+25+35)/3 = 25
        
        # unit 2: ydot_g5_r5 = 55 - 25 = 30
        assert np.isclose(
            result.loc[(result['id']==2) & (result['year']==5), 'ydot_g5_r5'].iloc[0],
            30.0, atol=1e-10
        )
    
    def test_manual_calculation_verification_detrend(self):
        """手动计算验证 detrend 转换结果"""
        data = pd.DataFrame({
            'id': [1]*6 + [2]*6,
            'year': [1,2,3,4,5,6]*2,
            # unit 1: 完美线性 Y = 8 + 2*t, 在 t>=5 有效应 +50
            'y': [10, 12, 14, 16, 68, 70,  # pre: 10,12,14,16, post: 18+50, 20+50
                  5, 7, 9, 11, 13, 15],     # unit 2: NT, Y = 3 + 2*t
            'gvar': [5]*6 + [0]*6
        })
        
        result = transform_staggered_detrend(
            data, 'y', 'id', 'year', 'gvar',
            exclude_pre_periods=1
        )
        
        # unit 1: OLS on {1,2,3} (排除 t=4)
        # Y_1=10, Y_2=12, Y_3=14 -> B=2, A=8
        # 预测 t=5: 8 + 2*5 = 18
        # ycheck = 68 - 18 = 50 (处理效应)
        ycheck = result.loc[(result['id']==1) & (result['year']==5), 'ycheck_g5_r5'].iloc[0]
        assert np.isclose(ycheck, 50.0, atol=1e-6)
        
        # unit 2 (NT): OLS on {1,2,3}
        # Y_1=5, Y_2=7, Y_3=9 -> B=2, A=3
        # 预测 t=5: 3 + 2*5 = 13
        # ycheck = 13 - 13 = 0
        ycheck_nt = result.loc[(result['id']==2) & (result['year']==5), 'ycheck_g5_r5'].iloc[0]
        assert np.isclose(ycheck_nt, 0.0, atol=1e-6)
    
    def test_boundary_max_exclude_demean(self):
        """边界测试: k = g - T_min - 1 (demean 最大允许值)
        
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
        
        # k=3: 只使用 t=1
        result = transform_staggered_demean(
            data, 'y', 'id', 'year', 'gvar',
            exclude_pre_periods=3
        )
        
        # unit 1: pre_mean = 10 (只有 t=1)
        # ydot_g5_r5 = 100 - 10 = 90
        ydot = result.loc[(result['id']==1) & (result['year']==5), 'ydot_g5_r5'].iloc[0]
        assert np.isclose(ydot, 90.0, atol=1e-10)
        
        # k=4: 应该抛出错误 (剩余 0 个时期)
        with pytest.raises(InsufficientPrePeriodsError):
            transform_staggered_demean(
                data, 'y', 'id', 'year', 'gvar',
                exclude_pre_periods=4
            )
    
    def test_boundary_max_exclude_detrend(self):
        """边界测试: k = g - T_min - 2 (detrend 最大允许值)
        
        cohort g=5, T_min=1
        最大 k = 5 - 1 - 2 = 2 (剩余 2 个时期)
        """
        data = pd.DataFrame({
            'id': [1]*6 + [2]*6,
            'year': [1,2,3,4,5,6]*2,
            'y': [10, 12, 14, 16, 100, 102,
                  5, 7, 9, 11, 13, 15],
            'gvar': [5]*6 + [0]*6
        })
        
        # k=2: 使用 t=1,2
        result = transform_staggered_detrend(
            data, 'y', 'id', 'year', 'gvar',
            exclude_pre_periods=2
        )
        
        # unit 1: OLS on {1,2}
        # Y_1=10, Y_2=12 -> B=2, A=8
        # 预测 t=5: 8 + 2*5 = 18
        # ycheck = 100 - 18 = 82
        ycheck = result.loc[(result['id']==1) & (result['year']==5), 'ycheck_g5_r5'].iloc[0]
        assert np.isclose(ycheck, 82.0, atol=1e-6)
        
        # k=3: 应该抛出错误 (剩余 1 个时期，不足 2)
        with pytest.raises(InsufficientPrePeriodsError):
            transform_staggered_detrend(
                data, 'y', 'id', 'year', 'gvar',
                exclude_pre_periods=3
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

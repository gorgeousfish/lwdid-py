# -*- coding: utf-8 -*-
"""
变换变量测试模块

测试Story 6.1的变换变量计算:
- 验证y_dot计算与Stata一致
- 验证pre-mean对所有period相同
- 验证分母使用(S-1)
- 边界情况测试
"""

import numpy as np
import pandas as pd
import pytest

from .conftest import (
    compute_transformed_outcome_common_timing,
    assert_att_close_to_stata,
)


class TestTransformationFormula:
    """变换公式测试"""
    
    def test_transformation_formula_known_values(self):
        """
        验证已知数据的变换计算
        
        公式: ŷ_{ir} = Y_{ir} - mean(Y_{i,pre})
        
        测试数据:
        - Y_pre = [1, 2, 3] for periods 1, 2, 3
        - pre_mean = 2
        - Y_4 = 10 → ŷ_4 = 10 - 2 = 8
        """
        # 创建测试数据
        data = pd.DataFrame({
            'id': [1, 1, 1, 1, 1, 1],
            'year': [1, 2, 3, 4, 5, 6],
            'y': [1.0, 2.0, 3.0, 10.0, 15.0, 20.0],
        })
        
        # S = 4, pre-periods = {1, 2, 3}
        # pre_mean = (1 + 2 + 3) / 3 = 2
        expected_pre_mean = 2.0
        
        # 计算变换
        for r in [4, 5, 6]:
            y_dot = compute_transformed_outcome_common_timing(
                data, 'y', 'id', 'year', 
                first_treat_period=4, 
                target_period=r
            )
            
            # 验证
            Y_r = data[data['year'] == r]['y'].values[0]
            expected_y_dot = Y_r - expected_pre_mean
            
            assert abs(y_dot.iloc[0] - expected_y_dot) < 1e-10, \
                f"Period {r}: expected {expected_y_dot}, got {y_dot.iloc[0]}"
    
    def test_pre_mean_constant_across_periods(self, common_timing_data):
        """
        验证pre-mean对所有r相同
        
        关键点: pre-mean仅依赖S，不依赖当前评估期r
        """
        S = 4
        
        # 计算不同r的变换 (使用period列)
        y_dots = {}
        for r in [4, 5, 6]:
            y_dots[r] = compute_transformed_outcome_common_timing(
                common_timing_data, 'y', 'id', 'period',
                first_treat_period=S,
                target_period=r
            )
        
        # 从y_dot反推pre_mean: Y_r - y_dot = pre_mean
        pre_means = {}
        for r in [4, 5, 6]:
            Y_r = common_timing_data[common_timing_data['period'] == r].set_index('id')['y']
            pre_means[r] = Y_r - y_dots[r]
        
        # 验证pre-mean对所有r相同 (使用atol=1e-5处理float32精度)
        assert np.allclose(pre_means[4], pre_means[5], atol=1e-5), \
            "pre_mean differs between period 4 and 5"
        assert np.allclose(pre_means[5], pre_means[6], atol=1e-5), \
            "pre_mean differs between period 5 and 6"
    
    def test_denominator_is_s_minus_1(self):
        """
        验证分母是(S-1)，不是(r-1)
        
        测试数据: S=4, pre_periods={1,2,3}
        分母应该是3，不是3,4,5
        """
        # 创建测试数据，让pre-period值便于验证
        data = pd.DataFrame({
            'id': [1, 1, 1, 1, 1, 1],
            'year': [1, 2, 3, 4, 5, 6],
            'y': [3.0, 6.0, 9.0, 100.0, 200.0, 300.0],  # pre_mean = (3+6+9)/3 = 6
        })
        
        S = 4
        expected_pre_mean = (3.0 + 6.0 + 9.0) / 3  # = 6.0
        
        for r in [4, 5, 6]:
            y_dot = compute_transformed_outcome_common_timing(
                data, 'y', 'id', 'year',
                first_treat_period=S,
                target_period=r
            )
            
            Y_r = data[data['year'] == r]['y'].values[0]
            computed_pre_mean = Y_r - y_dot.iloc[0]
            
            assert abs(computed_pre_mean - expected_pre_mean) < 1e-10, \
                f"Period {r}: expected pre_mean={expected_pre_mean}, got {computed_pre_mean}"
    
    def test_transformation_multiple_units(self, common_timing_data):
        """验证多单位数据的变换计算"""
        S = 4  # period索引
        r = 4  # period索引
        
        # 使用period列进行计算
        y_dot = compute_transformed_outcome_common_timing(
            common_timing_data, 'y', 'id', 'period',
            first_treat_period=S,
            target_period=r
        )
        
        # 验证返回的是Series且index是id
        assert isinstance(y_dot, pd.Series), "返回值应该是pd.Series"
        
        # 验证样本量
        n_expected = common_timing_data['id'].nunique()
        assert len(y_dot) == n_expected, \
            f"Expected {n_expected} units, got {len(y_dot)}"
        
        # 手动验证几个单位
        for unit_id in [1, 100, 500]:
            unit_data = common_timing_data[common_timing_data['id'] == unit_id]
            
            # 获取pre-period均值 (period < S)
            pre_data = unit_data[unit_data['period'] < S]
            pre_mean = pre_data['y'].mean()
            
            # 获取period r的Y
            Y_r = unit_data[unit_data['period'] == r]['y'].values[0]
            
            # 验证变换 (使用1e-5容差处理float32精度)
            expected = Y_r - pre_mean
            assert abs(y_dot[unit_id] - expected) < 1e-5, \
                f"Unit {unit_id}: expected {expected}, got {y_dot[unit_id]}"


class TestTransformationBoundary:
    """变换公式边界情况测试"""
    
    def test_all_same_pre_values(self):
        """测试pre-period值全部相同的情况"""
        data = pd.DataFrame({
            'id': [1, 1, 1, 1, 1, 1],
            'year': [1, 2, 3, 4, 5, 6],
            'y': [5.0, 5.0, 5.0, 10.0, 15.0, 20.0],  # pre全是5
        })
        
        y_dot = compute_transformed_outcome_common_timing(
            data, 'y', 'id', 'year',
            first_treat_period=4,
            target_period=4
        )
        
        # pre_mean = 5, Y_4 = 10, y_dot = 5
        assert abs(y_dot.iloc[0] - 5.0) < 1e-10
    
    def test_negative_y_values(self):
        """测试负值Y的情况"""
        data = pd.DataFrame({
            'id': [1, 1, 1, 1, 1, 1],
            'year': [1, 2, 3, 4, 5, 6],
            'y': [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0],
        })
        
        # pre_mean = (-3 + -2 + -1) / 3 = -2
        y_dot = compute_transformed_outcome_common_timing(
            data, 'y', 'id', 'year',
            first_treat_period=4,
            target_period=4
        )
        
        # Y_4 = 0, y_dot = 0 - (-2) = 2
        assert abs(y_dot.iloc[0] - 2.0) < 1e-10
    
    def test_large_y_values(self):
        """测试大数值Y的情况"""
        data = pd.DataFrame({
            'id': [1, 1, 1, 1, 1, 1],
            'year': [1, 2, 3, 4, 5, 6],
            'y': [1e8, 2e8, 3e8, 4e8, 5e8, 6e8],
        })
        
        # pre_mean = (1e8 + 2e8 + 3e8) / 3 = 2e8
        y_dot = compute_transformed_outcome_common_timing(
            data, 'y', 'id', 'year',
            first_treat_period=4,
            target_period=4
        )
        
        # Y_4 = 4e8, y_dot = 4e8 - 2e8 = 2e8
        assert abs(y_dot.iloc[0] - 2e8) < 1e-2  # 使用绝对误差
    
    def test_zero_pre_mean(self):
        """测试pre-mean为0的情况"""
        data = pd.DataFrame({
            'id': [1, 1, 1, 1, 1, 1],
            'year': [1, 2, 3, 4, 5, 6],
            'y': [-1.0, 0.0, 1.0, 5.0, 6.0, 7.0],  # pre_mean = 0
        })
        
        y_dot = compute_transformed_outcome_common_timing(
            data, 'y', 'id', 'year',
            first_treat_period=4,
            target_period=4
        )
        
        # Y_4 = 5, y_dot = 5 - 0 = 5
        assert abs(y_dot.iloc[0] - 5.0) < 1e-10


class TestTransformationDataIntegrity:
    """变换数据完整性测试"""
    
    def test_transformed_data_fixture(self, transformed_data, sample_info):
        """验证transformed_data fixture的数据完整性"""
        # 验证包含所有期
        periods = transformed_data['period'].unique()
        assert set(periods) == {4, 5, 6}, f"Expected periods {{4,5,6}}, got {set(periods)}"
        
        # 验证每期的样本量
        for period in [4, 5, 6]:
            period_data = transformed_data[transformed_data['period'] == period]
            assert len(period_data) == sample_info['n_obs'], \
                f"Period {period}: expected {sample_info['n_obs']} obs, got {len(period_data)}"
        
        # 验证y_dot列存在且无缺失
        assert 'y_dot' in transformed_data.columns, "Missing y_dot column"
        assert transformed_data['y_dot'].notna().all(), "y_dot contains NaN values"
    
    def test_y_dot_reasonable_values(self, transformed_data):
        """验证y_dot值在合理范围内"""
        y_dot = transformed_data['y_dot']
        
        # 统计特性
        assert not np.any(np.isinf(y_dot)), "y_dot contains infinite values"
        assert not np.any(np.isnan(y_dot)), "y_dot contains NaN values"
        
        # 验证值域 (变换后的值应该centered around 0或treatment effect)
        mean_y_dot = y_dot.mean()
        std_y_dot = y_dot.std()
        
        # 只检查不是异常值
        assert std_y_dot > 0, "y_dot has zero variance"
        assert std_y_dot < 100, f"y_dot variance seems too large: {std_y_dot}"


class TestTransformationConsistency:
    """变换计算一致性测试"""
    
    def test_transformation_consistent_with_stata_lag_structure(self, common_timing_data):
        """
        验证变换与Stata的lag结构一致
        
        Stata代码:
        - f04: L1.y + L2.y + L3.y = y3 + y2 + y1 (即y2003+y2002+y2001)
        - f05: L2.y + L3.y + L4.y = y3 + y2 + y1 (即y2003+y2002+y2001)
        - f06: L3.y + L4.y + L5.y = y3 + y2 + y1 (即y2003+y2002+y2001)
        
        关键: 不同period r时，使用不同的lag获取相同的pre-values
        年份映射: 2001=period1, ..., 2006=period6
        """
        S = 4
        
        # 取一个单位验证
        unit_id = 1
        unit_data = common_timing_data[common_timing_data['id'] == unit_id].sort_values('period')
        
        # 获取pre-period值 (period 1,2,3 对应年份 2001,2002,2003)
        y1 = unit_data[unit_data['period'] == 1]['y'].values[0]
        y2 = unit_data[unit_data['period'] == 2]['y'].values[0]
        y3 = unit_data[unit_data['period'] == 3]['y'].values[0]
        
        stata_pre_mean = (y1 + y2 + y3) / 3
        
        # 验证Python计算与Stata一致
        for r in [4, 5, 6]:
            y_dot = compute_transformed_outcome_common_timing(
                common_timing_data, 'y', 'id', 'period',
                first_treat_period=S,
                target_period=r
            )
            
            Y_r = unit_data[unit_data['period'] == r]['y'].values[0]
            python_y_dot = y_dot[unit_id]
            stata_y_dot = Y_r - stata_pre_mean
            
            # 使用1e-5容差处理float32精度
            assert abs(python_y_dot - stata_y_dot) < 1e-5, \
                f"Period {r}: Python y_dot={python_y_dot}, Stata y_dot={stata_y_dot}"


# =============================================================================
# 与Stata直接对比测试
# =============================================================================
# Stata计算方式:
#   xtset id year
#   bysort id: egen pre_mean = mean(y) if year >= 2001 & year <= 2003
#   bysort id: egen pre_mean_all = max(pre_mean)
#   gen y_dot = y - pre_mean_all if year >= 2004
# =============================================================================

class TestTransformationStataComparison:
    """
    与Stata直接对比的变换测试
    
    使用Stata MCP预计算的y_dot值进行验证。
    Stata命令: y_dot = y - mean(y) for pre-treatment periods (2001-2003)
    """
    
    # Stata计算的y_dot样本值 (通过Stata MCP获取，使用common timing数据)
    # 数据文件: 1.lee_wooldridge_common_data.dta
    # id: {period: (y, pre_mean, y_dot)}
    STATA_Y_DOT = {
        1: {
            4: (9.096133, 6.756263, 2.33987),
            5: (15.62343, 6.756263, 8.867171),
            6: (21.58159, 6.756263, 14.82533),
        },
        2: {
            4: (9.979717, 0.5568661, 9.422852),
            5: (13.32647, 0.5568661, 12.7696),
            6: (6.521822, 0.5568661, 5.964956),
        },
        3: {
            4: (1.47462, 5.932879, -4.458259),
            5: (6.666491, 5.932879, 0.7336116),
            6: (7.876863, 5.932879, 1.943984),
        },
        100: {
            4: (3.1842, 2.965106, 0.2190943),
            5: (8.533832, 2.965106, 5.568726),
            6: (16.02228, 2.965106, 13.05717),
        },
        500: {
            4: (10.89776, 8.197931, 2.699825),
            5: (23.50526, 8.197931, 15.30733),
            6: (14.21646, 8.197931, 6.018526),
        },
        1000: {
            4: (10.77844, 6.921341, 3.857095),
            5: (18.39667, 6.921341, 11.47533),
            6: (14.25148, 6.921341, 7.330143),
        },
    }
    
    # Stata计算的统计摘要 (common timing数据)
    STATA_SUMMARY = {
        4: {'mean': 3.131112, 'std': 4.812513, 'min': -11.41778, 'max': 19.36508},
        5: {'mean': 4.50372, 'std': 5.134296, 'min': -9.707735, 'max': 27.55585},
        6: {'mean': 6.320954, 'std': 5.770063, 'min': -14.76324, 'max': 38.6628},
    }
    
    def test_y_dot_vs_stata_f04(self, common_timing_data):
        """直接与Stata计算的y_dot对比 - Period 4"""
        y_dot = compute_transformed_outcome_common_timing(
            common_timing_data, 'y', 'id', 'period',
            first_treat_period=4,
            target_period=4
        )
        
        for unit_id, period_data in self.STATA_Y_DOT.items():
            if 4 in period_data:
                _, _, stata_y_dot = period_data[4]
                python_y_dot = y_dot.get(unit_id)
                
                assert python_y_dot is not None, f"Unit {unit_id} not found"
                assert abs(python_y_dot - stata_y_dot) < 1e-4, \
                    f"Unit {unit_id} f04: Python={python_y_dot:.6f}, Stata={stata_y_dot:.6f}"
    
    def test_y_dot_vs_stata_f05(self, common_timing_data):
        """直接与Stata计算的y_dot对比 - Period 5"""
        y_dot = compute_transformed_outcome_common_timing(
            common_timing_data, 'y', 'id', 'period',
            first_treat_period=4,
            target_period=5
        )
        
        for unit_id, period_data in self.STATA_Y_DOT.items():
            if 5 in period_data:
                _, _, stata_y_dot = period_data[5]
                python_y_dot = y_dot.get(unit_id)
                
                assert python_y_dot is not None, f"Unit {unit_id} not found"
                assert abs(python_y_dot - stata_y_dot) < 1e-4, \
                    f"Unit {unit_id} f05: Python={python_y_dot:.6f}, Stata={stata_y_dot:.6f}"
    
    def test_y_dot_vs_stata_f06(self, common_timing_data):
        """直接与Stata计算的y_dot对比 - Period 6"""
        y_dot = compute_transformed_outcome_common_timing(
            common_timing_data, 'y', 'id', 'period',
            first_treat_period=4,
            target_period=6
        )
        
        for unit_id, period_data in self.STATA_Y_DOT.items():
            if 6 in period_data:
                _, _, stata_y_dot = period_data[6]
                python_y_dot = y_dot.get(unit_id)
                
                assert python_y_dot is not None, f"Unit {unit_id} not found"
                assert abs(python_y_dot - stata_y_dot) < 1e-4, \
                    f"Unit {unit_id} f06: Python={python_y_dot:.6f}, Stata={stata_y_dot:.6f}"
    
    def test_y_dot_summary_statistics(self, common_timing_data):
        """验证y_dot的汇总统计与Stata一致"""
        for period in [4, 5, 6]:
            y_dot = compute_transformed_outcome_common_timing(
                common_timing_data, 'y', 'id', 'period',
                first_treat_period=4,
                target_period=period
            )
            
            # y_dot是pandas Series，使用.values获取numpy array
            y_dot_values = y_dot.values.astype(float)
            stata = self.STATA_SUMMARY[period]
            
            # 均值验证 (容差1e-4因为float32精度)
            assert abs(np.mean(y_dot_values) - stata['mean']) < 1e-4, \
                f"Period {period} mean: Python={np.mean(y_dot_values):.6f}, Stata={stata['mean']:.6f}"
            
            # 标准差验证
            assert abs(np.std(y_dot_values, ddof=1) - stata['std']) < 1e-3, \
                f"Period {period} std: Python={np.std(y_dot_values, ddof=1):.6f}, Stata={stata['std']:.6f}"

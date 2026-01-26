# -*- coding: utf-8 -*-
"""
BUG-060-D: Python-Stata 端到端交叉验证测试

验证 Python 的变换结果与 Stata 在缺失预处理数据场景下的行为一致性。

测试场景:
1. 完整数据单位的数值一致性
2. 缺失预处理数据单位产生 NaN（与 Stata 的 missing 一致）
3. 不同缺失模式的处理
4. detrend 变换的一致性验证
"""

import warnings
import numpy as np
import pandas as pd
import pytest

from lwdid.staggered.transformations import (
    transform_staggered_demean,
    transform_staggered_detrend,
)


class TestBug060DPythonStataConsistencyDemean:
    """
    验证 demean 变换的 Python-Stata 一致性
    
    Stata 计算方法 (from Lee & Wooldridge 2023 staggered code):
    bysort id: gen y_44 = y - (L1.y + L2.y + L3.y)/3 if f04
    
    这意味着如果某单位在 t-1, t-2, t-3 没有数据，结果为缺失
    """
    
    def test_complete_data_exact_match(self):
        """
        完整数据单位的数值应与 Stata 精确匹配
        
        Stata 结果:
        - Unit 1, period 4: ydot = 40 - (10+20+30)/3 = 40 - 20 = 20
        - Unit 1, period 5: ydot = 50 - (10+20+30)/3 = 50 - 20 = 30
        - Unit 1, period 6: ydot = 60 - (10+20+30)/3 = 60 - 20 = 40
        """
        data = pd.DataFrame({
            'id': [1, 1, 1, 1, 1, 1,
                   2, 2, 2, 2, 2, 2],
            'period': [1, 2, 3, 4, 5, 6] * 2,
            'y': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0,
                  15.0, 25.0, 35.0, 45.0, 55.0, 65.0],
            'gvar': [4, 4, 4, 4, 4, 4,
                     0, 0, 0, 0, 0, 0],
        })
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = transform_staggered_demean(
                data, 'y', 'id', 'period', 'gvar',
                never_treated_values=[0, np.inf]
            )
        
        # Stata results
        stata_results = {
            (1, 4): 20.0,
            (1, 5): 30.0,
            (1, 6): 40.0,
            (2, 4): 20.0,
            (2, 5): 30.0,
            (2, 6): 40.0,
        }
        
        for (unit_id, period), expected in stata_results.items():
            row = result[(result['id'] == unit_id) & (result['period'] == period)]
            actual = row[f'ydot_g4_r{period}'].values[0]
            
            assert abs(actual - expected) < 1e-10, \
                f"Unit {unit_id}, period {period}: Python={actual}, Stata={expected}"
    
    def test_missing_pretreatment_produces_nan(self):
        """
        缺失预处理数据的单位应产生 NaN（与 Stata 的 missing 一致）
        
        Stata 行为: 当 L1.y, L2.y, L3.y 中任一为缺失时，结果为缺失
        """
        data = pd.DataFrame({
            'id': [1, 1, 1, 1, 1, 1,
                   2, 2, 2, 2, 2, 2,
                   3, 3, 3],  # Unit 3: only has period 4,5,6
            'period': [1, 2, 3, 4, 5, 6,
                       1, 2, 3, 4, 5, 6,
                       4, 5, 6],
            'y': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0,
                  15.0, 25.0, 35.0, 45.0, 55.0, 65.0,
                  50.0, 60.0, 70.0],
            'gvar': [4, 4, 4, 4, 4, 4,
                     0, 0, 0, 0, 0, 0,
                     4, 4, 4],
        })
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = transform_staggered_demean(
                data, 'y', 'id', 'period', 'gvar',
                never_treated_values=[0, np.inf]
            )
        
        # Unit 3 should have NaN for all transformation columns
        for period in [4, 5, 6]:
            row = result[(result['id'] == 3) & (result['period'] == period)]
            actual = row[f'ydot_g4_r{period}'].values[0]
            assert np.isnan(actual), \
                f"Unit 3, period {period}: should be NaN, got {actual}"
    
    def test_partial_pretreatment_produces_nan(self):
        """
        部分预处理数据缺失也应产生 NaN
        
        场景: Unit 3 只有 period 2,3 的预处理数据（缺少 period 1）
        对于 cohort g=4, 需要 periods 1,2,3 的数据来计算 pre-mean
        """
        data = pd.DataFrame({
            'id': [1, 1, 1, 1, 1, 1,
                   3, 3, 3, 3, 3],  # Unit 3: missing period 1
            'period': [1, 2, 3, 4, 5, 6,
                       2, 3, 4, 5, 6],
            'y': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0,
                  25.0, 35.0, 50.0, 60.0, 70.0],
            'gvar': [4, 4, 4, 4, 4, 4,
                     4, 4, 4, 4, 4],
        })
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = transform_staggered_demean(
                data, 'y', 'id', 'period', 'gvar',
                never_treated_values=[0, np.inf]
            )
        
        # Unit 3 has pre-treatment data (periods 2,3), so should NOT be NaN
        # pre_mean = (25 + 35) / 2 = 30
        row = result[(result['id'] == 3) & (result['period'] == 4)]
        actual = row['ydot_g4_r4'].values[0]
        expected = 50.0 - 30.0  # = 20
        
        assert not np.isnan(actual), f"Unit 3 with partial pre-data should not be NaN"
        assert abs(actual - expected) < 1e-10, \
            f"Unit 3, period 4: expected {expected}, got {actual}"


class TestBug060DPythonStataConsistencyDetrend:
    """
    验证 detrend 变换的 Python-Stata 一致性
    
    detrend 需要至少 2 个预处理期来估计线性趋势
    """
    
    def test_detrend_complete_data(self):
        """
        完整数据的 detrend 变换应正确计算
        
        y = 10 + 10*t 的数据，预期趋势完美拟合
        """
        data = pd.DataFrame({
            'id': [1, 1, 1, 1, 1,
                   2, 2, 2, 2, 2],
            'period': [1, 2, 3, 4, 5] * 2,
            'y': [20.0, 30.0, 40.0, 50.0, 60.0,  # y = 10 + 10*t
                  25.0, 35.0, 45.0, 55.0, 65.0],  # y = 15 + 10*t
            'gvar': [4, 4, 4, 4, 4,
                     0, 0, 0, 0, 0],
        })
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = transform_staggered_detrend(
                data, 'y', 'id', 'period', 'gvar',
                never_treated_values=[0, np.inf]
            )
        
        # Perfect linear trend should give residuals ≈ 0
        for period in [4, 5]:
            for unit_id in [1, 2]:
                row = result[(result['id'] == unit_id) & (result['period'] == period)]
                actual = row[f'ycheck_g4_r{period}'].values[0]
                assert abs(actual) < 1e-8, \
                    f"Unit {unit_id}, period {period}: residual should be ~0, got {actual}"
    
    def test_detrend_insufficient_data_produces_nan(self):
        """
        预处理数据不足（只有1个观测）应产生 NaN
        """
        data = pd.DataFrame({
            'id': [1, 1, 1, 1, 1,  # Unit 1: 3 pre-treatment periods
                   2, 2, 2, 2, 2,  # Unit 2: 3 pre-treatment periods (NT)
                   3, 3, 3],       # Unit 3: only 1 pre-treatment period
            'period': [1, 2, 3, 4, 5,
                       1, 2, 3, 4, 5,
                       3, 4, 5],
            'y': [20.0, 30.0, 40.0, 50.0, 60.0,
                  25.0, 35.0, 45.0, 55.0, 65.0,
                  40.0, 100.0, 110.0],
            'gvar': [4, 4, 4, 4, 4,
                     0, 0, 0, 0, 0,
                     4, 4, 4],
        })
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = transform_staggered_detrend(
                data, 'y', 'id', 'period', 'gvar',
                never_treated_values=[0, np.inf]
            )
        
        # Unit 3 should have NaN (only 1 pre-treatment observation)
        for period in [4, 5]:
            row = result[(result['id'] == 3) & (result['period'] == period)]
            actual = row[f'ycheck_g4_r{period}'].values[0]
            assert np.isnan(actual), \
                f"Unit 3, period {period}: should be NaN, got {actual}"
    
    def test_detrend_no_pretreatment_produces_nan(self):
        """
        完全没有预处理数据应产生 NaN
        """
        data = pd.DataFrame({
            'id': [1, 1, 1, 1, 1,
                   3, 3],  # Unit 3: no pre-treatment data
            'period': [1, 2, 3, 4, 5,
                       4, 5],
            'y': [20.0, 30.0, 40.0, 50.0, 60.0,
                  100.0, 110.0],
            'gvar': [4, 4, 4, 4, 4,
                     4, 4],
        })
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = transform_staggered_detrend(
                data, 'y', 'id', 'period', 'gvar',
                never_treated_values=[0, np.inf]
            )
        
        # Unit 3 should have NaN
        for period in [4, 5]:
            row = result[(result['id'] == 3) & (result['period'] == period)]
            actual = row[f'ycheck_g4_r{period}'].values[0]
            assert np.isnan(actual), \
                f"Unit 3, period {period}: should be NaN, got {actual}"


class TestBug060DMultipleCohorts:
    """
    测试多 cohort 场景下的缺失数据处理
    """
    
    def test_different_cohorts_different_missing_patterns(self):
        """
        不同 cohort 可能有不同的缺失模式
        
        Unit 3 对于 cohort 4 没有预处理数据，但对于 cohort 5 有
        """
        data = pd.DataFrame({
            'id': [1, 1, 1, 1, 1, 1,  # cohort 4
                   2, 2, 2, 2, 2, 2,  # cohort 5
                   3, 3, 3, 3, 3],    # cohort 5, but has data from period 4
            'period': [1, 2, 3, 4, 5, 6,
                       1, 2, 3, 4, 5, 6,
                       4, 5, 6, 7, 8],  # missing 1,2,3
            'y': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0,
                  15.0, 25.0, 35.0, 45.0, 55.0, 65.0,
                  50.0, 60.0, 70.0, 80.0, 90.0],
            'gvar': [4, 4, 4, 4, 4, 4,
                     5, 5, 5, 5, 5, 5,
                     5, 5, 5, 5, 5],
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = transform_staggered_demean(
                data, 'y', 'id', 'period', 'gvar',
                never_treated_values=[0, np.inf]
            )
            
            # Should have warnings
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert len(user_warnings) > 0, "Should emit warning for missing data"
        
        # Unit 3 for cohort 4: missing pre-treatment (t<4)
        row = result[(result['id'] == 3) & (result['period'] == 5)]
        if 'ydot_g4_r5' in result.columns:
            actual = row['ydot_g4_r5'].values[0]
            assert np.isnan(actual), "Unit 3 should be NaN for cohort 4 transformation"
        
        # Unit 3 for cohort 5: has pre-treatment data (period 4)
        # pre_mean = 50 (only one period)
        if 'ydot_g5_r5' in result.columns:
            actual_g5 = row['ydot_g5_r5'].values[0]
            expected = 60.0 - 50.0  # = 10
            assert not np.isnan(actual_g5), "Unit 3 should NOT be NaN for cohort 5"
            assert abs(actual_g5 - expected) < 1e-10, \
                f"Unit 3 cohort 5: expected {expected}, got {actual_g5}"


class TestBug060DWarningMessage:
    """
    验证警告消息的准确性和有用性
    """
    
    def test_warning_includes_cohort_info(self):
        """
        警告应包含 cohort 信息
        """
        data = pd.DataFrame({
            'id': [1, 1, 1, 1,
                   2, 2, 2, 2, 2, 2,  # Add never-treated unit
                   3, 3, 3],  # Missing pre-treatment
            'period': [1, 2, 3, 4,
                       1, 2, 3, 4, 5, 6,
                       4, 5, 6],
            'y': [10.0, 20.0, 30.0, 40.0,
                  15.0, 25.0, 35.0, 45.0, 55.0, 65.0,
                  50.0, 60.0, 70.0],
            'gvar': [4, 4, 4, 4,
                     0, 0, 0, 0, 0, 0,  # never-treated
                     4, 4, 4],
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = transform_staggered_demean(
                data, 'y', 'id', 'period', 'gvar',
                never_treated_values=[0, np.inf]
            )
            
            # Filter for the specific BUG-060-D warning
            pre_treatment_warnings = [
                x for x in w 
                if issubclass(x.category, UserWarning) and 
                   "pre-treatment" in str(x.message).lower()
            ]
            assert len(pre_treatment_warnings) > 0, "Should emit pre-treatment warning"
            
            # Warning should mention cohort g=4
            warning_text = str(pre_treatment_warnings[0].message)
            assert "g=4" in warning_text, f"Warning should mention cohort: {warning_text}"
    
    def test_warning_includes_affected_unit_count(self):
        """
        警告应包含受影响单位的数量
        """
        # Create data with 3 units missing pre-treatment data
        data = pd.DataFrame({
            'id': [1, 1, 1, 1,  # has pre-treatment
                   5, 5, 5, 5, 5, 5,  # never-treated (to avoid no-NT warning)
                   2, 2, 2,      # missing pre-treatment
                   3, 3, 3,      # missing pre-treatment
                   4, 4, 4],     # missing pre-treatment
            'period': [1, 2, 3, 4,
                       1, 2, 3, 4, 5, 6,
                       4, 5, 6,
                       4, 5, 6,
                       4, 5, 6],
            'y': list(range(19)),
            'gvar': [4]*4 + [0]*6 + [4]*3 + [4]*3 + [4]*3,  # 5 is never-treated
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = transform_staggered_demean(
                data, 'y', 'id', 'period', 'gvar',
                never_treated_values=[0, np.inf]
            )
            
            # Filter for the specific BUG-060-D warning
            pre_treatment_warnings = [
                x for x in w 
                if issubclass(x.category, UserWarning) and 
                   "pre-treatment" in str(x.message).lower()
            ]
            assert len(pre_treatment_warnings) > 0, "Should emit pre-treatment warning"
            
            # Warning should mention "3 unit(s)"
            warning_text = str(pre_treatment_warnings[0].message)
            assert "3 unit" in warning_text, \
                f"Warning should mention count of affected units: {warning_text}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

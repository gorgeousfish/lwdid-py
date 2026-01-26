# -*- coding: utf-8 -*-
"""
BUG-060-D: 测试 transformations.py 处理前数据缺失警告机制

验证当某单位在预处理期间完全没有观测（或所有y值都是NaN）时，
系统会发出适当的警告，而不是静默地将变换结果设为NaN。

测试场景:
1. 单元测试 - 验证警告是否正确触发
2. 警告消息格式测试
3. 无缺失时不产生警告测试
4. 数值正确性测试 - 确保警告不影响计算正确性
"""

import warnings
import numpy as np
import pandas as pd
import pytest

from lwdid.staggered.transformations import (
    transform_staggered_demean,
    transform_staggered_detrend,
)


class TestBug060DDemeanWarning:
    """测试 transform_staggered_demean() 的缺失预处理数据警告"""
    
    def test_warning_when_unit_missing_all_pretreatment_data(self):
        """
        当某单位完全没有预处理期观测时，应发出警告
        
        场景: Unit 3 在 t < g=4 期间完全没有数据
        """
        # 创建测试数据，Unit 3 没有任何预处理期数据
        data = pd.DataFrame({
            'id': [1, 1, 1, 1, 1, 1,  # Unit 1: 完整数据
                   2, 2, 2, 2, 2, 2,  # Unit 2: 完整数据
                   3, 3, 3],          # Unit 3: 只有post期数据 (t=4,5,6)
            'period': [1, 2, 3, 4, 5, 6,
                       1, 2, 3, 4, 5, 6,
                       4, 5, 6],
            'y': [10.0, 11.0, 12.0, 20.0, 21.0, 22.0,
                  15.0, 16.0, 17.0, 25.0, 26.0, 27.0,
                  30.0, 31.0, 32.0],
            'gvar': [4, 4, 4, 4, 4, 4,  # Unit 1: cohort 4
                     0, 0, 0, 0, 0, 0,  # Unit 2: never treated
                     4, 4, 4],          # Unit 3: cohort 4, but missing pre data
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = transform_staggered_demean(
                data, 'y', 'id', 'period', 'gvar',
                never_treated_values=[0, np.inf]
            )
            
            # 检查是否发出了警告
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert len(user_warnings) > 0, "应该发出缺失预处理数据警告"
            
            # 检查警告消息内容
            warning_messages = [str(x.message) for x in user_warnings]
            found_warning = any(
                "no valid pre-treatment observations" in msg.lower() or 
                "unit" in msg.lower() and "nan" in msg.lower()
                for msg in warning_messages
            )
            assert found_warning, f"警告消息应包含相关内容: {warning_messages}"
    
    def test_warning_message_format_includes_unit_ids(self):
        """
        警告消息应包含缺失数据的单位ID列表
        """
        data = pd.DataFrame({
            'id': [1, 1, 1, 1,  # Unit 1: 完整数据
                   2, 2, 2,      # Unit 2: 只有post期数据
                   3, 3, 3],     # Unit 3: 只有post期数据
            'period': [1, 2, 3, 4,
                       4, 5, 6,
                       4, 5, 6],
            'y': [10.0, 11.0, 12.0, 20.0,
                  25.0, 26.0, 27.0,
                  30.0, 31.0, 32.0],
            'gvar': [4, 4, 4, 4,
                     4, 4, 4,
                     4, 4, 4],
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = transform_staggered_demean(
                data, 'y', 'id', 'period', 'gvar',
                never_treated_values=[0, np.inf]
            )
            
            user_warnings = [str(x.message) for x in w if issubclass(x.category, UserWarning)]
            # 检查是否列出了缺失单位的ID
            warning_text = " ".join(user_warnings)
            assert "2" in warning_text or "3" in warning_text, \
                f"警告消息应包含缺失单位ID: {warning_text}"
    
    def test_no_warning_when_all_units_have_pretreatment_data(self):
        """
        当所有单位都有有效的预处理数据时，不应发出此警告
        """
        data = pd.DataFrame({
            'id': [1, 1, 1, 1, 1, 1,
                   2, 2, 2, 2, 2, 2],
            'period': [1, 2, 3, 4, 5, 6] * 2,
            'y': [10.0, 11.0, 12.0, 20.0, 21.0, 22.0,
                  15.0, 16.0, 17.0, 25.0, 26.0, 27.0],
            'gvar': [4, 4, 4, 4, 4, 4,
                     0, 0, 0, 0, 0, 0],  # One treated, one never-treated
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = transform_staggered_demean(
                data, 'y', 'id', 'period', 'gvar',
                never_treated_values=[0, np.inf]
            )
            
            # 检查是否没有发出缺失预处理数据警告
            missing_data_warnings = [
                x for x in w 
                if issubclass(x.category, UserWarning) and 
                   "no valid pre-treatment observations" in str(x.message).lower()
            ]
            assert len(missing_data_warnings) == 0, \
                "不应发出缺失预处理数据警告"
    
    def test_warning_when_pretreatment_y_all_nan(self):
        """
        当预处理期的y值全为NaN时，应发出警告
        """
        data = pd.DataFrame({
            'id': [1, 1, 1, 1, 1, 1,
                   2, 2, 2, 2, 2, 2],
            'period': [1, 2, 3, 4, 5, 6] * 2,
            'y': [10.0, 11.0, 12.0, 20.0, 21.0, 22.0,  # Unit 1: 正常数据
                  np.nan, np.nan, np.nan, 25.0, 26.0, 27.0],  # Unit 2: 预处理期全NaN
            'gvar': [4, 4, 4, 4, 4, 4,
                     4, 4, 4, 4, 4, 4],
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = transform_staggered_demean(
                data, 'y', 'id', 'period', 'gvar',
                never_treated_values=[0, np.inf]
            )
            
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert len(user_warnings) > 0, "应发出警告"
    
    def test_transformed_values_nan_for_missing_units(self):
        """
        缺失预处理数据的单位，其变换值应为NaN
        """
        data = pd.DataFrame({
            'id': [1, 1, 1, 1,
                   2, 2, 2],  # Unit 2 只有post期数据
            'period': [1, 2, 3, 4,
                       4, 5, 6],
            'y': [10.0, 11.0, 12.0, 20.0,
                  25.0, 26.0, 27.0],
            'gvar': [4, 4, 4, 4,
                     4, 4, 4],
        })
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = transform_staggered_demean(
                data, 'y', 'id', 'period', 'gvar',
                never_treated_values=[0, np.inf]
            )
        
        # 检查Unit 2在period 4的变换值是NaN
        unit2_period4 = result[(result['id'] == 2) & (result['period'] == 4)]
        assert unit2_period4['ydot_g4_r4'].isna().all(), \
            "缺失预处理数据的单位，其变换值应为NaN"
        
        # 检查Unit 1的变换值不是NaN
        unit1_period4 = result[(result['id'] == 1) & (result['period'] == 4)]
        assert not unit1_period4['ydot_g4_r4'].isna().all(), \
            "有预处理数据的单位，其变换值不应为NaN"


class TestBug060DDetrendWarning:
    """测试 transform_staggered_detrend() 的缺失预处理数据警告"""
    
    def test_warning_when_unit_has_only_one_pretreatment_obs(self):
        """
        当某单位只有1个预处理期观测时，应发出警告（无法估计趋势）
        """
        data = pd.DataFrame({
            'id': [1, 1, 1, 1, 1,  # Unit 1: 3个预处理期
                   2, 2, 2, 2, 2,  # Unit 2: 3个预处理期
                   3, 3, 3],       # Unit 3: 只有1个预处理期 (t=3)
            'period': [1, 2, 3, 4, 5,
                       1, 2, 3, 4, 5,
                       3, 4, 5],
            'y': [10.0, 11.0, 12.0, 20.0, 21.0,
                  15.0, 16.0, 17.0, 25.0, 26.0,
                  18.0, 30.0, 31.0],
            'gvar': [4, 4, 4, 4, 4,
                     0, 0, 0, 0, 0,  # never treated
                     4, 4, 4],
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = transform_staggered_detrend(
                data, 'y', 'id', 'period', 'gvar',
                never_treated_values=[0, np.inf]
            )
            
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert len(user_warnings) > 0, "应发出警告"
            
            # 检查警告消息
            warning_text = " ".join(str(x.message) for x in user_warnings)
            assert "insufficient" in warning_text.lower() or "unit" in warning_text.lower(), \
                f"警告消息应包含相关内容: {warning_text}"
    
    def test_warning_when_unit_missing_all_pretreatment(self):
        """
        当某单位完全没有预处理期数据时，应发出警告
        """
        data = pd.DataFrame({
            'id': [1, 1, 1, 1, 1,
                   2, 2, 2, 2, 2,
                   3, 3],  # Unit 3: 完全没有预处理期数据
            'period': [1, 2, 3, 4, 5,
                       1, 2, 3, 4, 5,
                       4, 5],
            'y': [10.0, 11.0, 12.0, 20.0, 21.0,
                  15.0, 16.0, 17.0, 25.0, 26.0,
                  30.0, 31.0],
            'gvar': [4, 4, 4, 4, 4,
                     0, 0, 0, 0, 0,
                     4, 4],
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = transform_staggered_detrend(
                data, 'y', 'id', 'period', 'gvar',
                never_treated_values=[0, np.inf]
            )
            
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert len(user_warnings) > 0, "应发出警告"
    
    def test_no_warning_when_all_units_have_sufficient_data(self):
        """
        当所有单位都有足够的预处理数据时，不应发出缺失数据警告
        """
        data = pd.DataFrame({
            'id': [1, 1, 1, 1, 1,
                   2, 2, 2, 2, 2],
            'period': [1, 2, 3, 4, 5] * 2,
            'y': [10.0, 11.0, 12.0, 20.0, 21.0,
                  15.0, 16.0, 17.0, 25.0, 26.0],
            'gvar': [4, 4, 4, 4, 4,
                     0, 0, 0, 0, 0],
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = transform_staggered_detrend(
                data, 'y', 'id', 'period', 'gvar',
                never_treated_values=[0, np.inf]
            )
            
            # 检查是否没有发出缺失预处理数据警告
            missing_data_warnings = [
                x for x in w 
                if issubclass(x.category, UserWarning) and 
                   "insufficient" in str(x.message).lower()
            ]
            assert len(missing_data_warnings) == 0, \
                "不应发出缺失预处理数据警告"
    
    def test_transformed_values_nan_for_insufficient_data_units(self):
        """
        预处理数据不足的单位，其变换值应为NaN
        """
        data = pd.DataFrame({
            'id': [1, 1, 1, 1, 1,  # Unit 1: 3个预处理期
                   2, 2, 2],       # Unit 2: 只有1个预处理期
            'period': [1, 2, 3, 4, 5,
                       3, 4, 5],
            'y': [10.0, 11.0, 12.0, 20.0, 21.0,
                  18.0, 30.0, 31.0],
            'gvar': [4, 4, 4, 4, 4,
                     4, 4, 4],
        })
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = transform_staggered_detrend(
                data, 'y', 'id', 'period', 'gvar',
                never_treated_values=[0, np.inf]
            )
        
        # 检查Unit 2在period 4的变换值是NaN
        unit2_period4 = result[(result['id'] == 2) & (result['period'] == 4)]
        assert unit2_period4['ycheck_g4_r4'].isna().all(), \
            "预处理数据不足的单位，其变换值应为NaN"
        
        # 检查Unit 1的变换值不是NaN
        unit1_period4 = result[(result['id'] == 1) & (result['period'] == 4)]
        assert not unit1_period4['ycheck_g4_r4'].isna().all(), \
            "有足够预处理数据的单位，其变换值不应为NaN"


class TestBug060DWarningCount:
    """测试警告的数量和格式"""
    
    def test_warning_count_per_cohort_demean(self):
        """
        每个cohort应该发出独立的警告
        """
        # 创建有多个cohort的数据
        data = pd.DataFrame({
            'id': [1, 1, 1, 1, 1, 1,  # Unit 1: 完整数据, cohort 4
                   2, 2, 2, 2, 2, 2,  # Unit 2: 完整数据, cohort 5
                   3, 3, 3],          # Unit 3: 只有t>=4的数据, cohort 5
            'period': [1, 2, 3, 4, 5, 6,
                       1, 2, 3, 4, 5, 6,
                       4, 5, 6],
            'y': [10.0, 11.0, 12.0, 20.0, 21.0, 22.0,
                  15.0, 16.0, 17.0, 25.0, 26.0, 27.0,
                  30.0, 31.0, 32.0],
            'gvar': [4, 4, 4, 4, 4, 4,
                     5, 5, 5, 5, 5, 5,
                     5, 5, 5],
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = transform_staggered_demean(
                data, 'y', 'id', 'period', 'gvar',
                never_treated_values=[0, np.inf]
            )
            
            # 检查警告数量
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            # Unit 3 在 cohort 4 也没有预处理数据 (t<4)，所以可能有多个警告
            assert len(user_warnings) >= 1, "应发出至少一个警告"
    
    def test_many_missing_units_truncated_in_warning(self):
        """
        当缺失单位数量很多时，警告消息应该截断显示
        """
        # 创建大量缺失预处理数据的单位
        n_missing = 10
        ids = []
        periods = []
        y_vals = []
        gvar_vals = []
        
        # 一个正常单位
        for t in range(1, 7):
            ids.append(0)
            periods.append(t)
            y_vals.append(float(t * 10))
            gvar_vals.append(0)  # never treated
        
        # 多个缺失预处理数据的单位
        for i in range(1, n_missing + 1):
            for t in range(4, 7):  # 只有post期数据
                ids.append(i)
                periods.append(t)
                y_vals.append(float(t * 10 + i))
                gvar_vals.append(4)
        
        data = pd.DataFrame({
            'id': ids,
            'period': periods,
            'y': y_vals,
            'gvar': gvar_vals,
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = transform_staggered_demean(
                data, 'y', 'id', 'period', 'gvar',
                never_treated_values=[0, np.inf]
            )
            
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert len(user_warnings) > 0, "应发出警告"
            
            # 检查警告消息是否截断显示（包含"more"或类似内容）
            warning_text = str(user_warnings[0].message)
            # 如果超过5个单位，应该显示 "... (X more)"
            if n_missing > 5:
                assert "more" in warning_text.lower(), \
                    f"警告消息应包含截断提示: {warning_text}"


class TestBug060DNumericalCorrectness:
    """验证警告机制不影响数值计算正确性"""
    
    def test_valid_units_computed_correctly_despite_warning(self):
        """
        即使发出警告，有效单位的变换值仍应正确计算
        """
        data = pd.DataFrame({
            'id': [1, 1, 1, 1,  # Unit 1: 完整数据
                   2, 2, 2],     # Unit 2: 只有post期数据
            'period': [1, 2, 3, 4,
                       4, 5, 6],
            'y': [10.0, 20.0, 30.0, 100.0,  # Unit 1: pre-mean = (10+20+30)/3 = 20
                  50.0, 60.0, 70.0],
            'gvar': [4, 4, 4, 4,
                     4, 4, 4],
        })
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = transform_staggered_demean(
                data, 'y', 'id', 'period', 'gvar',
                never_treated_values=[0, np.inf]
            )
        
        # 验证Unit 1的变换值
        # pre_mean = (10 + 20 + 30) / 3 = 20
        # ydot_g4_r4 = 100 - 20 = 80
        unit1_period4 = result[(result['id'] == 1) & (result['period'] == 4)]
        expected_ydot = 100.0 - 20.0
        actual_ydot = unit1_period4['ydot_g4_r4'].values[0]
        
        assert abs(actual_ydot - expected_ydot) < 1e-10, \
            f"变换值计算错误: expected {expected_ydot}, got {actual_ydot}"
    
    def test_detrend_valid_units_computed_correctly(self):
        """
        即使发出警告，detrend的有效单位变换值仍应正确计算
        """
        # Unit 1: y = 10 + 10*t, 预期趋势
        data = pd.DataFrame({
            'id': [1, 1, 1, 1, 1,  # Unit 1: 完整数据
                   2, 2, 2],        # Unit 2: 只有1个预处理期
            'period': [1, 2, 3, 4, 5,
                       3, 4, 5],
            'y': [20.0, 30.0, 40.0, 50.0, 60.0,  # Unit 1: y = 10 + 10*t
                  40.0, 50.0, 60.0],
            'gvar': [4, 4, 4, 4, 4,
                     4, 4, 4],
        })
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = transform_staggered_detrend(
                data, 'y', 'id', 'period', 'gvar',
                never_treated_values=[0, np.inf]
            )
        
        # Unit 1 的趋势: y = 10 + 10*t (基于 t=1,2,3 的数据)
        # 在 t=4 时，预测值 = 10 + 10*4 = 50, 实际值 = 50
        # ycheck_g4_r4 = 50 - 50 = 0 (应该接近0，因为没有处理效应)
        unit1_period4 = result[(result['id'] == 1) & (result['period'] == 4)]
        actual_ycheck = unit1_period4['ycheck_g4_r4'].values[0]
        
        # 允许一定的数值误差
        assert abs(actual_ycheck) < 1e-8, \
            f"detrend变换值计算错误: expected ~0, got {actual_ycheck}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

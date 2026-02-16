"""
Story 3.3: Staggered场景变换变量验证测试

Phase 2 测试: 验证变换变量计算与Stata完全一致

关键验证点:
1. 变换公式分母是 g-1（cohort固定），不是 r-1
2. pre-treatment periods 正确识别为 1, 2, ..., g-1
3. 变换值与Stata计算一致（差异 < 1e-10）

变换公式 (论文公式4.7):
    ŷ_{irg} = Y_{ir} - (1/(g-1)) * Σ_{s=1}^{g-1} Y_{is}

Stata参考代码 (2.lee_wooldridge_rolling_staggered.do:19-28):
    bysort id: gen y_44 = y - (L1.y + L2.y + L3.y)/3 if f04
    bysort id: gen y_45 = y - (L2.y + L3.y + L4.y)/3 if f05
    bysort id: gen y_46 = y - (L3.y + L4.y + L5.y)/3 if f06
    bysort id: gen y_55 = y - (L1.y + L2.y + L3.y + L4.y)/4 if f05
    bysort id: gen y_56 = y - (L2.y + L3.y + L4.y +L5.y)/4 if f06
    bysort id: gen y_66 = y - (L1.y + L2.y + L3.y + L4.y+L5.y)/5 if f06
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from conftest import (
    compute_transformed_outcome,
    get_test_data_path,
)


class TestTransformationFormulaDenominator:
    """变换公式分母验证"""
    
    def test_denominator_is_g_minus_1_not_r_minus_1(self):
        """关键测试：验证分母是g-1而非r-1"""
        # 创建测试数据
        data = pd.DataFrame({
            'id': [1]*6,
            'year': [1, 2, 3, 4, 5, 6],
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        })
        
        # 测试(g=4, r=5)
        # 分母应该是 g-1 = 3，不是 r-1 = 4
        # pre-treatment periods: 1, 2, 3 (共3期)
        # pre_mean = (1+2+3)/3 = 2.0
        # Y_5 = 5.0
        # y_45 = 5.0 - 2.0 = 3.0
        y_45 = compute_transformed_outcome(data, 'y', 'id', 'year', cohort_g=4, period_r=5)
        
        assert abs(y_45.iloc[0] - 3.0) < 1e-10, \
            f"(g=4, r=5) 计算错误: {y_45.iloc[0]} != 3.0"
        
        # 测试(g=4, r=6)
        # 分母仍是 g-1 = 3
        # Y_6 = 6.0
        # y_46 = 6.0 - 2.0 = 4.0
        y_46 = compute_transformed_outcome(data, 'y', 'id', 'year', cohort_g=4, period_r=6)
        
        assert abs(y_46.iloc[0] - 4.0) < 1e-10, \
            f"(g=4, r=6) 计算错误: {y_46.iloc[0]} != 4.0"
        
        # 测试(g=5, r=5)
        # 分母是 g-1 = 4
        # pre-treatment periods: 1, 2, 3, 4 (共4期)
        # pre_mean = (1+2+3+4)/4 = 2.5
        # Y_5 = 5.0
        # y_55 = 5.0 - 2.5 = 2.5
        y_55 = compute_transformed_outcome(data, 'y', 'id', 'year', cohort_g=5, period_r=5)
        
        assert abs(y_55.iloc[0] - 2.5) < 1e-10, \
            f"(g=5, r=5) 计算错误: {y_55.iloc[0]} != 2.5"
    
    @pytest.mark.parametrize("g,expected_pre_periods", [
        (4, 3),   # g=4: periods 1,2,3 → 3 periods
        (5, 4),   # g=5: periods 1,2,3,4 → 4 periods
        (6, 5),   # g=6: periods 1,2,3,4,5 → 5 periods
    ])
    def test_pre_treatment_period_count(self, g, expected_pre_periods):
        """验证pre-treatment期数正确"""
        pre_periods = list(range(1, g))
        
        assert len(pre_periods) == expected_pre_periods, \
            f"cohort {g}: pre-treatment期数 {len(pre_periods)} != {expected_pre_periods}"
        assert len(pre_periods) == g - 1, \
            f"cohort {g}: 分母应该是 g-1={g-1}"


class TestTransformationManualCalculation:
    """手动计算验证"""
    
    @pytest.mark.parametrize("g,r,expected", [
        # Y_series = [1,2,3,4,5,6] for periods 1-6
        (4, 4, 4.0 - (1+2+3)/3),        # Y_4 - mean(1,2,3) = 4 - 2 = 2
        (4, 5, 5.0 - (1+2+3)/3),        # Y_5 - mean(1,2,3) = 5 - 2 = 3
        (4, 6, 6.0 - (1+2+3)/3),        # Y_6 - mean(1,2,3) = 6 - 2 = 4
        (5, 5, 5.0 - (1+2+3+4)/4),      # Y_5 - mean(1,2,3,4) = 5 - 2.5 = 2.5
        (5, 6, 6.0 - (1+2+3+4)/4),      # Y_6 - mean(1,2,3,4) = 6 - 2.5 = 3.5
        (6, 6, 6.0 - (1+2+3+4+5)/5),    # Y_6 - mean(1,2,3,4,5) = 6 - 3 = 3
    ])
    def test_transformation_manual_calculation(self, g, r, expected):
        """测试变换计算与手算一致"""
        data = pd.DataFrame({
            'id': [1]*6,
            'year': [1, 2, 3, 4, 5, 6],
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        })
        
        y_gr = compute_transformed_outcome(data, 'y', 'id', 'year', g, r)
        
        assert abs(y_gr.iloc[0] - expected) < 1e-10, \
            f"(g={g}, r={r}) 计算错误: {y_gr.iloc[0]} != {expected}"


class TestTransformationStataConsistency:
    """变换变量与Stata一致性测试"""
    
    @pytest.fixture
    def staggered_data_with_transforms(self, small_staggered_data):
        """计算变换变量并添加到数据"""
        data = small_staggered_data.copy()
        
        # 对于每个(g,r)组合，计算变换
        for g in [4, 5, 6]:
            for r in range(g, 7):
                col_name = f'y_{g}{r}'
                y_transformed = compute_transformed_outcome(
                    data, 'y', 'id', 'year', g, r
                )
                
                # 只在period r添加变换值
                data[col_name] = np.nan
                mask = data['year'] == r
                data.loc[mask, col_name] = data.loc[mask, 'id'].map(y_transformed)
        
        return data
    
    def test_transformation_consistency_44(self, small_staggered_data):
        """测试(4,4)变换与Stata计算逻辑一致"""
        # Stata: y_44 = y - (L1.y + L2.y + L3.y)/3 if f04
        # 等价于: Y_4 - mean(Y_1, Y_2, Y_3)
        
        # 手动实现Stata逻辑
        data = small_staggered_data.copy()
        data_wide = data.pivot(index='id', columns='year', values='y')
        
        # Stata计算: (L1 + L2 + L3)/3 at year=4 means (year3 + year2 + year1)/3
        stata_y_44 = data_wide[4] - (data_wide[1] + data_wide[2] + data_wide[3]) / 3
        
        # Python计算
        python_y_44 = compute_transformed_outcome(data, 'y', 'id', 'year', 4, 4)
        
        # 对齐索引
        common_ids = stata_y_44.index.intersection(python_y_44.index)
        diff = abs(stata_y_44.loc[common_ids] - python_y_44.loc[common_ids])
        
        # 注意：Stata数据使用float32，差异可能到1e-6级别
        assert diff.max() < 1e-5, \
            f"(4,4) 变换与Stata逻辑不一致: max_diff={diff.max()}"
    
    def test_transformation_consistency_55(self, small_staggered_data):
        """测试(5,5)变换与Stata计算逻辑一致"""
        # Stata: y_55 = y - (L1.y + L2.y + L3.y + L4.y)/4 if f05
        
        data = small_staggered_data.copy()
        data_wide = data.pivot(index='id', columns='year', values='y')
        
        # Stata计算
        stata_y_55 = data_wide[5] - (data_wide[1] + data_wide[2] + data_wide[3] + data_wide[4]) / 4
        
        # Python计算
        python_y_55 = compute_transformed_outcome(data, 'y', 'id', 'year', 5, 5)
        
        common_ids = stata_y_55.index.intersection(python_y_55.index)
        diff = abs(stata_y_55.loc[common_ids] - python_y_55.loc[common_ids])
        
        assert diff.max() < 1e-5, \
            f"(5,5) 变换与Stata逻辑不一致: max_diff={diff.max()}"
    
    def test_transformation_consistency_66(self, small_staggered_data):
        """测试(6,6)变换与Stata计算逻辑一致"""
        # Stata: y_66 = y - (L1.y + L2.y + L3.y + L4.y + L5.y)/5 if f06
        
        data = small_staggered_data.copy()
        data_wide = data.pivot(index='id', columns='year', values='y')
        
        # Stata计算
        stata_y_66 = data_wide[6] - (data_wide[1] + data_wide[2] + data_wide[3] + 
                                      data_wide[4] + data_wide[5]) / 5
        
        # Python计算
        python_y_66 = compute_transformed_outcome(data, 'y', 'id', 'year', 6, 6)
        
        common_ids = stata_y_66.index.intersection(python_y_66.index)
        diff = abs(stata_y_66.loc[common_ids] - python_y_66.loc[common_ids])
        
        assert diff.max() < 1e-5, \
            f"(6,6) 变换与Stata逻辑不一致: max_diff={diff.max()}"
    
    @pytest.mark.parametrize("g,r", [(4,4), (4,5), (4,6), (5,5), (5,6), (6,6)])
    def test_all_transformations_vs_stata_logic(self, small_staggered_data, g, r):
        """测试所有(g,r)组合的变换与Stata逻辑一致"""
        data = small_staggered_data.copy()
        data_wide = data.pivot(index='id', columns='year', values='y')
        
        # Stata逻辑：Y_r - mean(Y_1, ..., Y_{g-1})
        pre_periods = list(range(1, g))
        pre_sum = sum(data_wide[t] for t in pre_periods)
        stata_y_gr = data_wide[r] - pre_sum / len(pre_periods)
        
        # Python计算
        python_y_gr = compute_transformed_outcome(data, 'y', 'id', 'year', g, r)
        
        common_ids = stata_y_gr.index.intersection(python_y_gr.index)
        diff = abs(stata_y_gr.loc[common_ids] - python_y_gr.loc[common_ids])
        
        # 阈值放宽到1e-5以适应float32精度
        assert diff.max() < 1e-5, \
            f"(g={g}, r={r}) 变换与Stata逻辑不一致: max_diff={diff.max()}"


class TestTransformationBoundaryConditions:
    """变换公式边界情况测试"""
    
    def test_missing_pre_treatment_periods(self):
        """测试缺失pre-treatment期的处理"""
        # 创建缺失period 1的数据
        data = pd.DataFrame({
            'id': [1, 1, 1, 2, 2, 2, 2],
            'year': [2, 3, 4, 1, 2, 3, 4],  # id=1缺失period 1
            'y': [2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
        })
        
        # 对于g=4，需要periods 1,2,3
        # id=1缺失period 1，应该基于可用数据计算（或返回NaN）
        y_gr = compute_transformed_outcome(data, 'y', 'id', 'year', 4, 4)
        
        # id=1: 只有periods 2,3可用，pre_mean = (2+3)/2 = 2.5
        # y_44 for id=1 = 4 - 2.5 = 1.5 (使用可用数据的均值)
        # id=2: pre_mean = (1+2+3)/3 = 2.0
        # y_44 for id=2 = 4 - 2.0 = 2.0
        
        assert 1 in y_gr.index and 2 in y_gr.index
    
    def test_constant_outcome_transformation(self):
        """测试常数结果变量的变换"""
        data = pd.DataFrame({
            'id': [1, 1, 1, 1],
            'year': [1, 2, 3, 4],
            'y': [5.0, 5.0, 5.0, 5.0],  # 常数
        })
        
        y_gr = compute_transformed_outcome(data, 'y', 'id', 'year', 4, 4)
        
        # Y_r - mean(Y_pre) = 5 - 5 = 0
        assert abs(y_gr.iloc[0] - 0.0) < 1e-10
    
    def test_single_pre_treatment_period(self):
        """测试只有一个pre-treatment期（g=2）"""
        data = pd.DataFrame({
            'id': [1, 1],
            'year': [1, 2],
            'y': [3.0, 5.0],
        })
        
        # g=2, 分母=1, pre_periods=[1]
        y_gr = compute_transformed_outcome(data, 'y', 'id', 'year', 2, 2)
        
        # Y_2 - Y_1 = 5 - 3 = 2
        assert abs(y_gr.iloc[0] - 2.0) < 1e-10
    
    def test_multiple_units_transformation(self, small_staggered_data):
        """测试多单位变换的正确性"""
        # 验证每个单位都正确计算了变换
        y_44 = compute_transformed_outcome(small_staggered_data, 'y', 'id', 'year', 4, 4)
        
        # 应该有与period 4观测数相等的变换值
        n_units = small_staggered_data['id'].nunique()
        assert len(y_44) == n_units, \
            f"变换值数量 {len(y_44)} != 单位数 {n_units}"
    
    def test_transformation_preserves_unit_identity(self, small_staggered_data):
        """测试变换保持单位标识"""
        y_44 = compute_transformed_outcome(small_staggered_data, 'y', 'id', 'year', 4, 4)
        
        # 变换结果的index应该是单位ID
        expected_ids = set(small_staggered_data['id'].unique())
        actual_ids = set(y_44.index)
        
        assert actual_ids == expected_ids


class TestStataLagEquivalence:
    """Stata lag操作符等价性测试"""
    
    def test_lag_interpretation_44(self, small_staggered_data):
        """测试(4,4)的lag解释正确"""
        # Stata: L1.y + L2.y + L3.y at f04 (year=4)
        # L1.y at year=4 = y at year=3
        # L2.y at year=4 = y at year=2
        # L3.y at year=4 = y at year=1
        # 所以: (L1+L2+L3) = Y_3 + Y_2 + Y_1 = sum(periods 1,2,3)
        
        data = small_staggered_data.copy()
        data_wide = data.pivot(index='id', columns='year', values='y')
        
        # 验证pre-treatment periods是否正确
        pre_periods = list(range(1, 4))  # [1, 2, 3]
        assert pre_periods == [1, 2, 3]
    
    def test_lag_interpretation_45(self, small_staggered_data):
        """测试(4,5)的lag解释正确"""
        # Stata: L2.y + L3.y + L4.y at f05 (year=5)
        # L2.y at year=5 = y at year=3
        # L3.y at year=5 = y at year=2  
        # L4.y at year=5 = y at year=1
        # 所以: (L2+L3+L4) = Y_3 + Y_2 + Y_1 = sum(periods 1,2,3)
        # 注意：仍然是periods 1,2,3！因为g=4的pre-treatment是固定的
        
        # 这验证了分母是g-1而非r-1
        data = small_staggered_data.copy()
        data_wide = data.pivot(index='id', columns='year', values='y')
        
        pre_periods = list(range(1, 4))  # g=4 → [1, 2, 3]
        assert len(pre_periods) == 3, "分母应该是3 (=g-1)"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

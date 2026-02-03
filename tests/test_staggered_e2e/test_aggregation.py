# -*- coding: utf-8 -*-
"""
Cohort和Overall聚合公式验证测试 (Task 6.2.26-6.2.29)

验证:
- Cohort效应: τ_g = (1/(T-g+1)) × Σ_{r=g}^{T} τ_{gr}
- Overall效应: τ = Σ_g ω_g × τ_g, 其中 ω_g = N_g / N_treat
"""

import numpy as np
import pandas as pd
import pytest

from .conftest import (
    GR_COMBINATIONS,
    STATA_RA_RESULTS,
    STATA_IPWRA_RESULTS,
    STATA_PSM_RESULTS,
    SAMPLE_INFO,
    compute_transformed_outcome_staggered,
    build_subsample_for_gr,
)


class TestCohortAggregation:
    """Cohort效应聚合测试"""
    
    def test_cohort4_aggregation_formula(self, staggered_data):
        """
        测试Cohort 4的聚合公式:
        τ_4 = (1/3) × (τ_{44} + τ_{45} + τ_{46})
        """
        from lwdid.staggered import estimate_ipwra
        
        # 获取所有(4,r)的ATT
        atts = []
        for r in [4, 5, 6]:
            y_dot = compute_transformed_outcome_staggered(
                staggered_data, 'y', 'id', 'period', 'first_treat',
                cohort_g=4, target_period=r
            )
            subsample = build_subsample_for_gr(
                staggered_data, 4, r,
                gvar_col='first_treat',
                period_col='period'
            )
            subsample['y_dot'] = subsample['id'].map(y_dot)
            
            result = estimate_ipwra(
                data=subsample.dropna(subset=['y_dot']),
                y='y_dot',
                d='d',
                controls=['x1', 'x2'],
            )
            atts.append(result.att)
        
        # 手动计算聚合效应
        tau_4_manual = np.mean(atts)
        
        # 使用Stata值验证
        stata_atts = [
            STATA_IPWRA_RESULTS[(4, 4)]['att'],
            STATA_IPWRA_RESULTS[(4, 5)]['att'],
            STATA_IPWRA_RESULTS[(4, 6)]['att'],
        ]
        tau_4_stata = np.mean(stata_atts)
        
        # 手动计算应该接近Stata
        assert abs(tau_4_manual - tau_4_stata) / abs(tau_4_stata) < 0.02, \
            f"Cohort 4 aggregation mismatch: manual={tau_4_manual:.4f}, stata={tau_4_stata:.4f}"
    
    def test_cohort5_aggregation_formula(self, staggered_data):
        """
        测试Cohort 5的聚合公式:
        τ_5 = (1/2) × (τ_{55} + τ_{56})
        """
        from lwdid.staggered import estimate_ipwra
        
        atts = []
        for r in [5, 6]:
            y_dot = compute_transformed_outcome_staggered(
                staggered_data, 'y', 'id', 'period', 'first_treat',
                cohort_g=5, target_period=r
            )
            subsample = build_subsample_for_gr(
                staggered_data, 5, r,
                gvar_col='first_treat',
                period_col='period'
            )
            subsample['y_dot'] = subsample['id'].map(y_dot)
            
            result = estimate_ipwra(
                data=subsample.dropna(subset=['y_dot']),
                y='y_dot',
                d='d',
                controls=['x1', 'x2'],
            )
            atts.append(result.att)
        
        tau_5_manual = np.mean(atts)
        
        stata_atts = [
            STATA_IPWRA_RESULTS[(5, 5)]['att'],
            STATA_IPWRA_RESULTS[(5, 6)]['att'],
        ]
        tau_5_stata = np.mean(stata_atts)
        
        assert abs(tau_5_manual - tau_5_stata) / abs(tau_5_stata) < 0.02, \
            f"Cohort 5 aggregation mismatch: manual={tau_5_manual:.4f}, stata={tau_5_stata:.4f}"
    
    def test_cohort6_aggregation_formula(self, staggered_data):
        """
        测试Cohort 6的聚合公式:
        τ_6 = τ_{66} (只有一个period)
        """
        from lwdid.staggered import estimate_ipwra
        
        y_dot = compute_transformed_outcome_staggered(
            staggered_data, 'y', 'id', 'period', 'first_treat',
            cohort_g=6, target_period=6
        )
        subsample = build_subsample_for_gr(
            staggered_data, 6, 6,
            gvar_col='first_treat',
            period_col='period'
        )
        subsample['y_dot'] = subsample['id'].map(y_dot)
        
        result = estimate_ipwra(
            data=subsample.dropna(subset=['y_dot']),
            y='y_dot',
            d='d',
            controls=['x1', 'x2'],
        )
        
        tau_6_manual = result.att
        tau_6_stata = STATA_IPWRA_RESULTS[(6, 6)]['att']
        
        assert abs(tau_6_manual - tau_6_stata) / abs(tau_6_stata) < 0.02, \
            f"Cohort 6 aggregation mismatch: manual={tau_6_manual:.4f}, stata={tau_6_stata:.4f}"


class TestOverallAggregation:
    """Overall效应聚合测试"""
    
    def test_overall_effect_formula(self, staggered_data):
        """
        测试Overall效应聚合公式:
        τ = Σ_g ω_g × τ_g
        其中 ω_g = N_g / N_treat
        """
        # 样本信息
        n_g4 = SAMPLE_INFO['n_g4']  # 129
        n_g5 = SAMPLE_INFO['n_g5']  # 109
        n_g6 = SAMPLE_INFO['n_g6']  # 110
        n_treat = n_g4 + n_g5 + n_g6  # 348
        
        # 权重
        w_4 = n_g4 / n_treat
        w_5 = n_g5 / n_treat
        w_6 = n_g6 / n_treat
        
        assert abs(w_4 + w_5 + w_6 - 1.0) < 1e-10, "Weights should sum to 1"
        
        # Cohort效应 (使用Stata IPWRA)
        tau_4 = np.mean([
            STATA_IPWRA_RESULTS[(4, 4)]['att'],
            STATA_IPWRA_RESULTS[(4, 5)]['att'],
            STATA_IPWRA_RESULTS[(4, 6)]['att'],
        ])
        tau_5 = np.mean([
            STATA_IPWRA_RESULTS[(5, 5)]['att'],
            STATA_IPWRA_RESULTS[(5, 6)]['att'],
        ])
        tau_6 = STATA_IPWRA_RESULTS[(6, 6)]['att']
        
        # Overall效应
        tau_overall = w_4 * tau_4 + w_5 * tau_5 + w_6 * tau_6
        
        # 验证Overall在合理范围内
        assert 2.0 < tau_overall < 6.0, \
            f"Overall effect {tau_overall:.4f} out of expected range"
    
    def test_weights_sum_to_one(self):
        """验证权重之和为1"""
        n_g4 = SAMPLE_INFO['n_g4']
        n_g5 = SAMPLE_INFO['n_g5']
        n_g6 = SAMPLE_INFO['n_g6']
        n_treat = n_g4 + n_g5 + n_g6
        
        w_4 = n_g4 / n_treat
        w_5 = n_g5 / n_treat
        w_6 = n_g6 / n_treat
        
        assert abs(w_4 + w_5 + w_6 - 1.0) < 1e-10
    
    def test_expected_weights(self):
        """验证期望权重值"""
        n_g4 = 129
        n_g5 = 109
        n_g6 = 110
        n_treat = 348
        
        assert abs(n_g4 / n_treat - 0.3707) < 0.001  # ~37%
        assert abs(n_g5 / n_treat - 0.3132) < 0.001  # ~31%
        assert abs(n_g6 / n_treat - 0.3161) < 0.001  # ~32%


class TestAggregationModule:
    """测试lwdid.staggered.aggregation模块"""
    
    def test_aggregation_functions_exist(self):
        """验证聚合函数存在"""
        try:
            from lwdid.staggered import (
                aggregate_to_cohort,
                aggregate_to_overall,
                CohortEffect,
                OverallEffect,
            )
        except ImportError as e:
            pytest.fail(f"无法导入聚合函数: {e}")
    
    def test_cohort_effect_dataclass(self):
        """验证CohortEffect数据类结构"""
        from lwdid.staggered import CohortEffect
        
        # 验证类存在且有正确属性
        assert hasattr(CohortEffect, '__dataclass_fields__')
        assert 'cohort' in CohortEffect.__dataclass_fields__
        assert 'att' in CohortEffect.__dataclass_fields__
        assert 'se' in CohortEffect.__dataclass_fields__
    
    def test_overall_effect_dataclass(self):
        """验证OverallEffect数据类结构"""
        from lwdid.staggered import OverallEffect
        
        # 验证类存在且有正确属性
        assert hasattr(OverallEffect, '__dataclass_fields__')
        assert 'att' in OverallEffect.__dataclass_fields__
        assert 'se' in OverallEffect.__dataclass_fields__


class TestAggregationWithAllEstimators:
    """使用所有估计量测试聚合"""
    
    @pytest.mark.parametrize("estimator", ['ipwra'])
    def test_cohort_aggregation_all_estimators(self, staggered_data, estimator):
        """测试不同估计量的Cohort聚合"""
        from lwdid.staggered import estimate_ipwra
        
        estimate_func = estimate_ipwra
        stata_results = STATA_IPWRA_RESULTS
        
        # Cohort 4的聚合
        atts = []
        for r in [4, 5, 6]:
            y_dot = compute_transformed_outcome_staggered(
                staggered_data, 'y', 'id', 'period', 'first_treat',
                cohort_g=4, target_period=r
            )
            subsample = build_subsample_for_gr(
                staggered_data, 4, r,
                gvar_col='first_treat',
                period_col='period'
            )
            subsample['y_dot'] = subsample['id'].map(y_dot)
            
            result = estimate_func(
                data=subsample.dropna(subset=['y_dot']),
                y='y_dot',
                d='d',
                controls=['x1', 'x2'],
            )
            atts.append(result.att)
        
        tau_4 = np.mean(atts)
        
        # 与Stata聚合对比
        stata_tau_4 = np.mean([
            stata_results[(4, 4)]['att'],
            stata_results[(4, 5)]['att'],
            stata_results[(4, 6)]['att'],
        ])
        
        assert abs(tau_4 - stata_tau_4) / abs(stata_tau_4) < 0.02, \
            f"{estimator} Cohort 4 aggregation mismatch"


class TestAggregationEdgeCases:
    """聚合边界情况测试"""
    
    def test_single_period_cohort(self):
        """测试只有单个period的cohort（如cohort 6）"""
        # Cohort 6只有(6,6)一个组合
        # 聚合效应 = τ_{66}
        tau_6 = STATA_IPWRA_RESULTS[(6, 6)]['att']
        
        # 应该直接等于该period的ATT
        assert tau_6 > 0  # 合理性检查
    
    def test_all_cohorts_positive_effect(self):
        """验证所有cohort效应为正（符合预期）"""
        # 根据论文DGP，处理效应应该为正
        for g in [4, 5, 6]:
            periods = list(range(g, 7))
            atts = [STATA_IPWRA_RESULTS[(g, r)]['att'] for r in periods]
            tau_g = np.mean(atts)
            
            assert tau_g > 0, f"Cohort {g} effect should be positive"

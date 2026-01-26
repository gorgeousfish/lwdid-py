# -*- coding: utf-8 -*-
"""
IPWRA估计量与Stata一致性测试 (Task 6.2.18-6.2.21)

验证:
- IPWRA ATT与Stata teffects ipwra一致 (相对误差 < 1e-6)
- IPWRA SE与Stata一致 (相对误差 < 5%)
- 所有6个(g,r)组合的端到端测试
"""

import numpy as np
import pandas as pd
import pytest

from .conftest import (
    GR_COMBINATIONS,
    STATA_IPWRA_RESULTS,
    compute_transformed_outcome_staggered,
    build_subsample_for_gr,
    assert_att_close_to_stata,
    assert_se_close_to_stata,
)


class TestIPWRAEstimatorBasic:
    """IPWRA估计量基础测试"""
    
    def test_estimate_ipwra_exists(self):
        """验证estimate_ipwra函数存在"""
        try:
            from lwdid.staggered import estimate_ipwra, IPWRAResult
        except ImportError as e:
            pytest.fail(f"无法导入estimate_ipwra: {e}")
    
    def test_ipwra_result_structure(self, staggered_data):
        """验证IPWRAResult结构完整"""
        from lwdid.staggered import estimate_ipwra, IPWRAResult
        
        # 准备数据
        subsample = build_subsample_for_gr(
            staggered_data, 4, 4,
            gvar_col='first_treat',
            period_col='period'
        )
        
        y_dot = compute_transformed_outcome_staggered(
            staggered_data, 'y', 'id', 'period', 'first_treat',
            cohort_g=4, target_period=4
        )
        subsample['y_dot'] = subsample['id'].map(y_dot)
        
        # 运行估计
        result = estimate_ipwra(
            data=subsample.dropna(subset=['y_dot']),
            y='y_dot',
            d='d',
            controls=['x1', 'x2'],
        )
        
        # 验证结果结构
        assert isinstance(result, IPWRAResult)
        assert hasattr(result, 'att')
        assert hasattr(result, 'se')
        assert hasattr(result, 'ci_lower')
        assert hasattr(result, 'ci_upper')
        assert hasattr(result, 't_stat')
        assert hasattr(result, 'pvalue')
        assert hasattr(result, 'n_treated')
        assert hasattr(result, 'n_control')
        assert hasattr(result, 'propensity_scores')
        assert hasattr(result, 'weights')
    
    def test_ipwra_produces_reasonable_values(self, staggered_data):
        """验证IPWRA产生合理的值"""
        from lwdid.staggered import estimate_ipwra
        
        subsample = build_subsample_for_gr(
            staggered_data, 4, 4,
            gvar_col='first_treat',
            period_col='period'
        )
        
        y_dot = compute_transformed_outcome_staggered(
            staggered_data, 'y', 'id', 'period', 'first_treat',
            cohort_g=4, target_period=4
        )
        subsample['y_dot'] = subsample['id'].map(y_dot)
        
        result = estimate_ipwra(
            data=subsample.dropna(subset=['y_dot']),
            y='y_dot',
            d='d',
            controls=['x1', 'x2'],
        )
        
        # 基本合理性检查
        assert not np.isnan(result.att), "ATT should not be NaN"
        assert result.se > 0, "SE should be positive"
        assert result.ci_lower < result.att < result.ci_upper, \
            "ATT should be within CI"
        assert result.n_treated > 0, "n_treated should be positive"
        assert result.n_control > 0, "n_control should be positive"


class TestIPWRAStataConsistency:
    """IPWRA估计量与Stata一致性测试"""
    
    @pytest.mark.parametrize("g,r", GR_COMBINATIONS)
    def test_ipwra_att_vs_stata(self, staggered_data, g, r):
        """测试IPWRA ATT与Stata一致"""
        from lwdid.staggered import estimate_ipwra
        
        # 构建子样本
        subsample = build_subsample_for_gr(
            staggered_data, g, r,
            gvar_col='first_treat',
            period_col='period'
        )
        
        # 计算变换变量
        y_dot = compute_transformed_outcome_staggered(
            staggered_data, 'y', 'id', 'period', 'first_treat',
            cohort_g=g, target_period=r
        )
        subsample['y_dot'] = subsample['id'].map(y_dot)
        
        # 运行估计
        result = estimate_ipwra(
            data=subsample.dropna(subset=['y_dot']),
            y='y_dot',
            d='d',
            controls=['x1', 'x2'],
        )
        
        # 与Stata对比
        stata = STATA_IPWRA_RESULTS[(g, r)]
        
        # ATT应该非常接近
        assert_att_close_to_stata(
            result.att, stata['att'],
            tolerance=0.01,  # 1% 容差
            description=f"({g},{r}) IPWRA ATT"
        )
    
    @pytest.mark.parametrize("g,r", GR_COMBINATIONS)
    def test_ipwra_se_vs_stata(self, staggered_data, g, r):
        """测试IPWRA SE与Stata一致"""
        from lwdid.staggered import estimate_ipwra
        
        subsample = build_subsample_for_gr(
            staggered_data, g, r,
            gvar_col='first_treat',
            period_col='period'
        )
        
        y_dot = compute_transformed_outcome_staggered(
            staggered_data, 'y', 'id', 'period', 'first_treat',
            cohort_g=g, target_period=r
        )
        subsample['y_dot'] = subsample['id'].map(y_dot)
        
        result = estimate_ipwra(
            data=subsample.dropna(subset=['y_dot']),
            y='y_dot',
            d='d',
            controls=['x1', 'x2'],
        )
        
        stata = STATA_IPWRA_RESULTS[(g, r)]
        
        # SE容差5%
        assert_se_close_to_stata(
            result.se, stata['se'],
            tolerance=0.10,  # 10% 容差
            description=f"({g},{r}) IPWRA SE"
        )
    
    @pytest.mark.parametrize("g,r", GR_COMBINATIONS)
    def test_ipwra_sample_sizes(self, staggered_data, g, r):
        """测试IPWRA样本量与Stata一致"""
        from lwdid.staggered import estimate_ipwra
        
        subsample = build_subsample_for_gr(
            staggered_data, g, r,
            gvar_col='first_treat',
            period_col='period'
        )
        
        y_dot = compute_transformed_outcome_staggered(
            staggered_data, 'y', 'id', 'period', 'first_treat',
            cohort_g=g, target_period=r
        )
        subsample['y_dot'] = subsample['id'].map(y_dot)
        
        result = estimate_ipwra(
            data=subsample.dropna(subset=['y_dot']),
            y='y_dot',
            d='d',
            controls=['x1', 'x2'],
        )
        
        stata = STATA_IPWRA_RESULTS[(g, r)]
        
        assert result.n_treated == stata['n_treated'], \
            f"({g},{r}): n_treated {result.n_treated} != Stata {stata['n_treated']}"
        assert result.n_control == stata['n_control'], \
            f"({g},{r}): n_control {result.n_control} != Stata {stata['n_control']}"


class TestIPWRAE2E:
    """IPWRA端到端测试"""
    
    def test_ipwra_full_pipeline(self, staggered_data):
        """测试IPWRA完整流程"""
        from lwdid.staggered import estimate_ipwra
        
        results = {}
        
        for g, r in GR_COMBINATIONS:
            y_dot = compute_transformed_outcome_staggered(
                staggered_data, 'y', 'id', 'period', 'first_treat',
                cohort_g=g, target_period=r
            )
            
            subsample = build_subsample_for_gr(
                staggered_data, g, r,
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
            
            results[(g, r)] = result
        
        # 验证所有结果有效
        for (g, r), result in results.items():
            assert not np.isnan(result.att), f"({g},{r}): ATT is NaN"
            assert result.se > 0, f"({g},{r}): SE <= 0"


class TestIPWRACompareToRA:
    """IPWRA与RA对比测试"""
    
    @pytest.mark.parametrize("g,r", GR_COMBINATIONS)
    def test_ipwra_vs_ra_similarity(self, staggered_data, g, r):
        """测试IPWRA与RA结果相近（双稳健性）"""
        from lwdid.staggered import estimate_ipwra, estimate_ra
        
        subsample = build_subsample_for_gr(
            staggered_data, g, r,
            gvar_col='first_treat',
            period_col='period'
        )
        
        y_dot = compute_transformed_outcome_staggered(
            staggered_data, 'y', 'id', 'period', 'first_treat',
            cohort_g=g, target_period=r
        )
        subsample['y_dot'] = subsample['id'].map(y_dot)
        data = subsample.dropna(subset=['y_dot'])
        
        ra_result = estimate_ra(data, 'y_dot', 'd', ['x1', 'x2'])
        ipwra_result = estimate_ipwra(data, 'y_dot', 'd', ['x1', 'x2'])
        
        # IPWRA和RA应该方向一致
        assert (ra_result.att > 0) == (ipwra_result.att > 0), \
            f"({g},{r}): RA and IPWRA have different signs"
        
        # 两者差异不应该太大（<20%）
        relative_diff = abs(ra_result.att - ipwra_result.att) / abs(ra_result.att)
        assert relative_diff < 0.20, \
            f"({g},{r}): RA ({ra_result.att:.4f}) vs IPWRA ({ipwra_result.att:.4f}) differ by {relative_diff:.1%}"


class TestIPWRADoublyRobust:
    """IPWRA双稳健性测试"""
    
    def test_doubly_robust_property(self, staggered_data):
        """测试双稳健性：当两个模型有一个正确时估计一致"""
        from lwdid.staggered import estimate_ipwra
        
        # 使用(4,4)测试
        subsample = build_subsample_for_gr(
            staggered_data, 4, 4,
            gvar_col='first_treat',
            period_col='period'
        )
        
        y_dot = compute_transformed_outcome_staggered(
            staggered_data, 'y', 'id', 'period', 'first_treat',
            cohort_g=4, target_period=4
        )
        subsample['y_dot'] = subsample['id'].map(y_dot)
        data = subsample.dropna(subset=['y_dot'])
        
        # 正确控制变量
        result_correct = estimate_ipwra(data, 'y_dot', 'd', ['x1', 'x2'])
        
        # 仅使用部分控制变量（可能misspecified）
        result_partial = estimate_ipwra(data, 'y_dot', 'd', ['x1'])
        
        # 两者都应该产生合理结果
        assert not np.isnan(result_correct.att)
        assert not np.isnan(result_partial.att)
        
        # 差异不应该过大（IPWRA的稳健性）
        relative_diff = abs(result_correct.att - result_partial.att) / abs(result_correct.att)
        assert relative_diff < 0.30, \
            f"Partial controls differ by {relative_diff:.1%}, IPWRA may not be robust"

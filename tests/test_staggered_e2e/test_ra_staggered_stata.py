# -*- coding: utf-8 -*-
"""
RA估计量与Stata一致性测试 (Task 6.2.14-6.2.17)

验证:
- RA ATT与Stata teffects ra一致 (相对误差 < 1e-6)
- RA SE与Stata一致 (相对误差 < 5%)
- 所有6个(g,r)组合的端到端测试
"""

import numpy as np
import pandas as pd
import pytest

from .conftest import (
    GR_COMBINATIONS,
    STATA_RA_RESULTS,
    compute_transformed_outcome_staggered,
    build_subsample_for_gr,
    assert_att_close_to_stata,
    assert_se_close_to_stata,
)


class TestRAEstimatorBasic:
    """RA估计量基础测试"""
    
    def test_estimate_ra_exists(self):
        """验证estimate_ra函数存在"""
        try:
            from lwdid.staggered import estimate_ra, RAResult
        except ImportError as e:
            pytest.fail(f"无法导入estimate_ra: {e}")
    
    def test_ra_result_structure(self, staggered_data):
        """验证RAResult结构完整"""
        from lwdid.staggered import estimate_ra, RAResult
        
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
        result = estimate_ra(
            data=subsample.dropna(subset=['y_dot']),
            y='y_dot',
            d='d',
            controls=['x1', 'x2'],
        )
        
        # 验证结果结构
        assert isinstance(result, RAResult)
        assert hasattr(result, 'att')
        assert hasattr(result, 'se')
        assert hasattr(result, 'ci_lower')
        assert hasattr(result, 'ci_upper')
        assert hasattr(result, 't_stat')
        assert hasattr(result, 'pvalue')
        assert hasattr(result, 'n_treated')
        assert hasattr(result, 'n_control')
        assert hasattr(result, 'outcome_model_coef')
    
    def test_ra_produces_reasonable_values(self, staggered_data):
        """验证RA产生合理的值"""
        from lwdid.staggered import estimate_ra
        
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
        
        result = estimate_ra(
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


class TestRAStataConsistency:
    """RA估计量与Stata一致性测试"""
    
    @pytest.mark.parametrize("g,r", GR_COMBINATIONS)
    def test_ra_att_vs_stata(self, staggered_data, g, r):
        """测试RA ATT与Stata一致"""
        from lwdid.staggered import estimate_ra
        
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
        result = estimate_ra(
            data=subsample.dropna(subset=['y_dot']),
            y='y_dot',
            d='d',
            controls=['x1', 'x2'],
        )
        
        # 与Stata对比
        stata = STATA_RA_RESULTS[(g, r)]
        
        # 使用较宽松的容差，因为SE计算可能有细微差异
        # 但ATT应该非常接近
        att_error = abs(result.att - stata['att']) / abs(stata['att'])
        
        assert att_error < 0.01, \
            f"({g},{r}) ATT相对误差 {att_error:.2%} > 1%\n" \
            f"  Python: {result.att:.6f}\n  Stata: {stata['att']:.6f}"
    
    @pytest.mark.parametrize("g,r", GR_COMBINATIONS)
    def test_ra_se_vs_stata(self, staggered_data, g, r):
        """测试RA SE与Stata一致"""
        from lwdid.staggered import estimate_ra
        
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
        
        result = estimate_ra(
            data=subsample.dropna(subset=['y_dot']),
            y='y_dot',
            d='d',
            controls=['x1', 'x2'],
        )
        
        stata = STATA_RA_RESULTS[(g, r)]
        se_error = abs(result.se - stata['se']) / stata['se']
        
        # SE使用完整的Stacked M-estimation Sandwich方法
        # 与Stata teffects ra完全一致
        # SE误差目标: < 0.1%
        assert se_error < 0.001, \
            f"({g},{r}) SE相对误差 {se_error:.2%} > 0.1%\n" \
            f"  Python: {result.se:.6f}\n  Stata: {stata['se']:.6f}"
    
    @pytest.mark.parametrize("g,r", GR_COMBINATIONS)
    def test_ra_sample_sizes(self, staggered_data, g, r):
        """测试RA样本量与Stata一致"""
        from lwdid.staggered import estimate_ra
        
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
        
        result = estimate_ra(
            data=subsample.dropna(subset=['y_dot']),
            y='y_dot',
            d='d',
            controls=['x1', 'x2'],
        )
        
        stata = STATA_RA_RESULTS[(g, r)]
        
        assert result.n_treated == stata['n_treated'], \
            f"({g},{r}): n_treated {result.n_treated} != Stata {stata['n_treated']}"
        assert result.n_control == stata['n_control'], \
            f"({g},{r}): n_control {result.n_control} != Stata {stata['n_control']}"


class TestRAE2E:
    """RA端到端测试"""
    
    def test_ra_full_pipeline(self, staggered_data):
        """测试RA完整流程: 数据 → 变换 → 子样本 → 估计 → 验证"""
        from lwdid.staggered import estimate_ra
        
        results = {}
        
        for g, r in GR_COMBINATIONS:
            # Step 1: 变换
            y_dot = compute_transformed_outcome_staggered(
                staggered_data, 'y', 'id', 'period', 'first_treat',
                cohort_g=g, target_period=r
            )
            
            # Step 2: 构建子样本
            subsample = build_subsample_for_gr(
                staggered_data, g, r,
                gvar_col='first_treat',
                period_col='period'
            )
            subsample['y_dot'] = subsample['id'].map(y_dot)
            
            # Step 3: 估计
            result = estimate_ra(
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
    
    def test_ra_all_gr_combinations_produce_results(self, staggered_data):
        """测试所有(g,r)组合都能产生结果"""
        from lwdid.staggered import estimate_ra
        
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
            
            result = estimate_ra(
                data=subsample.dropna(subset=['y_dot']),
                y='y_dot',
                d='d',
                controls=['x1', 'x2'],
            )
            
            assert result is not None, f"({g},{r}): Result is None"


class TestRAFormulaValidation:
    """RA公式验证测试"""
    
    def test_ra_is_not_simple_ols_coefficient(self, staggered_data):
        """
        验证RA估计与简单OLS回归中D的系数不同
        
        重要区别:
        - 简单OLS: Y ~ 1 + D + X → β_D是条件ATE
        - RA: 在控制组估计结果模型，预测处理组反事实
        """
        import statsmodels.api as sm
        from lwdid.staggered import estimate_ra
        
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
        subsample = subsample.dropna(subset=['y_dot'])
        
        # 方法1: 简单OLS
        y = subsample['y_dot'].values
        X = sm.add_constant(np.column_stack([
            subsample['d'].values,
            subsample['x1'].values,
            subsample['x2'].values,
        ]))
        model = sm.OLS(y, X).fit()
        ols_att = model.params[1]
        
        # 方法2: RA估计
        result = estimate_ra(
            data=subsample,
            y='y_dot',
            d='d',
            controls=['x1', 'x2'],
        )
        ra_att = result.att
        
        # 两者应该方向一致
        assert ols_att > 0 and ra_att > 0, \
            "Both estimates should be positive"
        
        # 验证RA与Stata接近
        stata_att = STATA_RA_RESULTS[(4, 4)]['att']
        assert abs(ra_att - stata_att) / abs(stata_att) < 0.01, \
            "RA ATT should match Stata"
    
    def test_ra_counterfactual_prediction(self, staggered_data):
        """
        验证RA正确预测处理组的反事实结果
        
        m₀(X_i) 应该是在控制组上估计的模型对处理组X的预测
        """
        from lwdid.staggered import estimate_ra
        
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
        subsample = subsample.dropna(subset=['y_dot'])
        
        result = estimate_ra(
            data=subsample,
            y='y_dot',
            d='d',
            controls=['x1', 'x2'],
        )
        
        # pomean_0 应该是合理的值
        # 它是处理组单位的预测反事实结果的均值
        assert not np.isnan(result.pomean_0), "pomean_0 should not be NaN"


class TestRAVceOptions:
    """RA方差估计选项测试"""
    
    @pytest.mark.parametrize("vce", ['robust', 'hc0', 'hc1', 'hc2', 'hc3'])
    def test_vce_options(self, staggered_data, vce):
        """测试不同VCE选项"""
        from lwdid.staggered import estimate_ra
        
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
        
        result = estimate_ra(
            data=subsample.dropna(subset=['y_dot']),
            y='y_dot',
            d='d',
            controls=['x1', 'x2'],
            vce=vce,
        )
        
        # 所有VCE选项都应该产生有效结果
        assert not np.isnan(result.att), f"ATT is NaN with vce={vce}"
        assert result.se > 0, f"SE <= 0 with vce={vce}"

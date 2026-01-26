# -*- coding: utf-8 -*-
"""
RA估计量与Stata一致性测试

测试Story 6.1的RA估计量:
- 验证ATT与Stata teffects ra一致 (相对误差 < 1e-6)
- 验证SE与Stata一致 (相对误差 < 0.1%) - 使用Sandwich估计器精确对齐
- 端到端测试完整流程
"""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from .conftest import (
    compute_transformed_outcome_common_timing,
    assert_att_close_to_stata,
    assert_se_close_to_stata,
)
from lwdid.staggered.estimators import compute_ra_se_analytical


class TestRAAnalyticalSE:
    """
    RA SE精确对齐测试 - 使用Sandwich估计器
    
    使用compute_ra_se_analytical实现与Stata teffects ra完全一致的SE计算。
    基于联合估计方程(Stacked EE)的Sandwich方差估计器。
    
    目标: SE误差 < 0.1%
    """
    
    STATA_RESULTS = {
        'f04': {'att': 4.0830643, 'se': 0.41764992},
        'f05': {'att': 5.0768895, 'se': 0.45991541},
        'f06': {'att': 6.5960774, 'se': 0.48026091},
    }
    
    @pytest.mark.parametrize("period,period_key", [(4, 'f04'), (5, 'f05'), (6, 'f06')])
    def test_ra_se_analytical_vs_stata(self, transformed_data, period, period_key):
        """
        测试RA SE使用Sandwich估计器与Stata完全一致
        
        误差目标: < 0.1%
        """
        period_data = transformed_data[transformed_data['period'] == period].copy()
        
        # RA估计
        treated_mask = period_data['d'].values == 1
        control_mask = ~treated_mask
        y_vals = period_data['y_dot'].values.astype(float)
        X_controls = period_data[['x1', 'x2']].values.astype(float)
        
        # 控制组OLS
        X_control = sm.add_constant(X_controls[control_mask])
        outcome_model = sm.OLS(y_vals[control_mask], X_control).fit()
        
        # 预测处理组反事实
        X_treated = sm.add_constant(X_controls[treated_mask])
        y_counterfactual = outcome_model.predict(X_treated)
        
        # ATET
        att = np.mean(y_vals[treated_mask] - y_counterfactual)
        
        # 构建系数字典
        outcome_coef = {
            '_intercept': outcome_model.params[0],
            'x1': outcome_model.params[1],
            'x2': outcome_model.params[2],
        }
        
        # 使用Sandwich估计器计算SE
        se, ci_lower, ci_upper = compute_ra_se_analytical(
            data=period_data,
            y='y_dot',
            d='d',
            controls=['x1', 'x2'],
            att=att,
            outcome_coef=outcome_coef,
        )
        
        stata = self.STATA_RESULTS[period_key]
        
        # ATT验证 (< 1e-5)
        assert_att_close_to_stata(
            att,
            stata['att'],
            tolerance=1e-5,
            description=f"{period_key} RA ATT"
        )
        
        # SE验证 (< 0.1%)
        assert_se_close_to_stata(
            se,
            stata['se'],
            tolerance=0.001,  # 0.1%
            description=f"{period_key} RA SE (analytical)"
        )
    
    @pytest.mark.parametrize("period,period_key", [(4, 'f04'), (5, 'f05'), (6, 'f06')])
    def test_ra_ci_coverage(self, transformed_data, period, period_key):
        """测试置信区间合理性"""
        period_data = transformed_data[transformed_data['period'] == period].copy()
        
        # RA估计
        treated_mask = period_data['d'].values == 1
        control_mask = ~treated_mask
        y_vals = period_data['y_dot'].values.astype(float)
        X_controls = period_data[['x1', 'x2']].values.astype(float)
        
        X_control = sm.add_constant(X_controls[control_mask])
        outcome_model = sm.OLS(y_vals[control_mask], X_control).fit()
        X_treated = sm.add_constant(X_controls[treated_mask])
        y_counterfactual = outcome_model.predict(X_treated)
        att = np.mean(y_vals[treated_mask] - y_counterfactual)
        
        outcome_coef = {
            '_intercept': outcome_model.params[0],
            'x1': outcome_model.params[1],
            'x2': outcome_model.params[2],
        }
        
        se, ci_lower, ci_upper = compute_ra_se_analytical(
            data=period_data,
            y='y_dot',
            d='d',
            controls=['x1', 'x2'],
            att=att,
            outcome_coef=outcome_coef,
            alpha=0.05,
        )
        
        # 验证CI合理性
        assert ci_lower < att < ci_upper, "ATT should be within CI"
        assert ci_upper - ci_lower > 0, "CI width should be positive"
        
        # 验证CI宽度与SE一致 (95% CI = ATT ± 1.96*SE)
        expected_width = 2 * 1.96 * se
        actual_width = ci_upper - ci_lower
        assert abs(actual_width - expected_width) / expected_width < 0.01, \
            "CI width should be approximately 2*1.96*SE"


class TestRAStataConsistency:
    """RA估计量与Stata一致性测试 (旧版手动实现，用于对比)"""
    
    # Stata teffects ra验证结果 (从1.lee_wooldridge_rolling_common.do)
    STATA_RESULTS = {
        'f04': {'att': 4.0830643, 'se': 0.41764992, 'pomean_0': 2.622859},
        'f05': {'att': 5.0768895, 'se': 0.45991541, 'pomean_0': 3.901225},
        'f06': {'att': 6.5960774, 'se': 0.48026091, 'pomean_0': 5.795996},
    }
    
    def _estimate_ra_manual(
        self, 
        data: pd.DataFrame,
        y: str,
        d: str,
        controls: list,
    ) -> dict:
        """
        手动实现RA估计量 (模拟teffects ra, atet)
        
        teffects ra (outcome model) (treatment) [if], atet 的实现:
        
        步骤:
        1. 在控制组上估计结果模型: E[Y|X,D=0] = α + X'β
        2. 对处理组单位，预测其反事实结果: Ŷ(0)_i = α̂ + X_i'β̂
        3. ATET = (1/N_1) × Σ_{i:D=1} [Y_i - Ŷ(0)_i]
        
        标准误使用M-estimation (influence function approach)
        """
        # 分离处理组和控制组
        treated_mask = data[d].values == 1
        control_mask = ~treated_mask
        
        y_vals = data[y].values.astype(float)
        X_controls = data[controls].values.astype(float)
        
        # 控制组数据
        y_control = y_vals[control_mask]
        X_control = X_controls[control_mask]
        X_control_design = sm.add_constant(X_control)
        
        # Step 1: 在控制组上估计结果模型
        model_control = sm.OLS(y_control, X_control_design).fit()
        
        # 处理组数据
        y_treated = y_vals[treated_mask]
        X_treated = X_controls[treated_mask]
        X_treated_design = sm.add_constant(X_treated)
        
        # Step 2: 预测处理组的反事实结果
        y_counterfactual = model_control.predict(X_treated_design)
        
        # Step 3: 计算ATET
        individual_effects = y_treated - y_counterfactual
        att = np.mean(individual_effects)
        
        # Step 4: 计算SE (使用influence function approach)
        # 简化版本: 使用Delta method近似
        n_treated = int(treated_mask.sum())
        n_control = int(control_mask.sum())
        n = len(data)
        
        # SE计算 (M-estimation)
        # 参考: Stata teffects manual, Wooldridge (2010)
        # 简化: 使用样本方差的标准误
        se_simple = np.std(individual_effects, ddof=1) / np.sqrt(n_treated)
        
        # 需要调整预测误差的方差贡献
        # 使用完整的影响函数方法会更准确
        # 但这里使用近似方法
        resid_control = model_control.resid
        sigma2_control = np.var(resid_control, ddof=X_control_design.shape[1])
        
        # 预测方差贡献
        # Var(X'beta_hat) ≈ X' Var(beta_hat) X
        # 但对于ATET，主要的方差来自处理效应的异质性
        
        # 使用robust标准误的近似
        # 这是一个简化版本，完整版需要influence function
        se = se_simple * np.sqrt(1 + n_treated / n_control)
        
        return {
            'att': att,
            'se': se,
            'n_obs': n,
            'n_treated': n_treated,
            'n_control': n_control,
            'pomean_0': np.mean(y_counterfactual),  # 控制组潜在结果均值
        }
    
    @pytest.mark.parametrize("period,period_key", [(4, 'f04'), (5, 'f05'), (6, 'f06')])
    def test_ra_att_vs_stata(self, transformed_data, period, period_key):
        """测试RA ATT与Stata一致"""
        # 筛选到目标period
        period_data = transformed_data[transformed_data['period'] == period].copy()
        
        # RA估计
        result = self._estimate_ra_manual(
            data=period_data,
            y='y_dot',
            d='d',
            controls=['x1', 'x2'],
        )
        
        # 与Stata对比
        stata = self.STATA_RESULTS[period_key]
        
        assert_att_close_to_stata(
            result['att'],
            stata['att'],
            tolerance=1e-5,  # 放宽到1e-5由于float32精度
            description=f"{period_key} ATT"
        )
    
    @pytest.mark.parametrize("period,period_key", [(4, 'f04'), (5, 'f05'), (6, 'f06')])
    def test_ra_se_simplified_vs_stata(self, transformed_data, period, period_key):
        """
        测试简化版RA SE与Stata (用于对比，非精确对齐)
        
        注意: 这是简化版的influence function方法，与Stata有约3-4%差异。
        精确对齐请使用TestRAAnalyticalSE中的test_ra_se_analytical_vs_stata。
        """
        period_data = transformed_data[transformed_data['period'] == period].copy()
        
        result = self._estimate_ra_manual(
            data=period_data,
            y='y_dot',
            d='d',
            controls=['x1', 'x2'],
        )
        
        stata = self.STATA_RESULTS[period_key]
        
        assert_se_close_to_stata(
            result['se'],
            stata['se'],
            tolerance=0.05,  # 5%相对误差 (简化版)
            description=f"{period_key} SE (simplified)"
        )
    
    def test_ra_sample_sizes(self, transformed_data, sample_info):
        """测试样本量与预期一致"""
        for period in [4, 5, 6]:
            period_data = transformed_data[transformed_data['period'] == period]
            
            n_obs = len(period_data)
            n_treated = int(period_data['d'].sum())
            n_control = int((1 - period_data['d']).sum())
            
            assert n_obs == sample_info['n_obs'], \
                f"Period {period}: n_obs mismatch"
            assert n_treated == sample_info['n_treated'], \
                f"Period {period}: n_treated mismatch"
            assert n_control == sample_info['n_control'], \
                f"Period {period}: n_control mismatch"


class TestRAE2E:
    """RA端到端测试"""
    
    STATA_RESULTS = {
        'f04': {'att': 4.0830643, 'se': 0.41764992},
        'f05': {'att': 5.0768895, 'se': 0.45991541},
        'f06': {'att': 6.5960774, 'se': 0.48026091},
    }
    
    def test_ra_full_pipeline(self, common_timing_data, stata_results):
        """测试RA完整流程: 数据 → 变换 → 估计 → 验证"""
        results = {}
        S = 4
        
        for period in [4, 5, 6]:
            # Step 1: 变换
            y_dot = compute_transformed_outcome_common_timing(
                common_timing_data, 'y', 'id', 'period',
                first_treat_period=S,
                target_period=period
            )
            
            # Step 2: 准备数据
            period_data = common_timing_data[
                common_timing_data['period'] == period
            ].copy()
            period_data['y_dot'] = period_data['id'].map(y_dot)
            
            # Step 3: 使用正确的RA估计方法 (不是简单OLS)
            # RA方法: 在控制组估计结果模型，预测处理组反事实
            treated_mask = period_data['d'].values == 1
            control_mask = ~treated_mask
            
            y_vals = period_data['y_dot'].values.astype(float)
            X_controls = period_data[['x1', 'x2']].values.astype(float)
            
            # 控制组OLS
            y_control = y_vals[control_mask]
            X_control = sm.add_constant(X_controls[control_mask])
            model_control = sm.OLS(y_control, X_control).fit()
            
            # 预测处理组反事实
            X_treated = sm.add_constant(X_controls[treated_mask])
            y_counterfactual = model_control.predict(X_treated)
            
            # ATET
            y_treated = y_vals[treated_mask]
            att = np.mean(y_treated - y_counterfactual)
            
            results[period] = {'att': att}
        
        # 验证所有period的ATT
        for period in [4, 5, 6]:
            period_key = f'f{period:02d}'
            stata = self.STATA_RESULTS[period_key]
            result = results[period]
            
            # ATT验证 (相对误差 < 1e-5)
            att_error = abs(result['att'] - stata['att']) / abs(stata['att'])
            assert att_error < 1e-5, \
                f"Period {period} ATT error {att_error:.2e} > 1e-5"
    
    def test_ra_result_reasonable(self, transformed_data):
        """测试RA结果合理性"""
        period_data = transformed_data[transformed_data['period'] == 4].copy()
        
        # 简单OLS
        y_vals = period_data['y_dot'].values.astype(float)
        d_vals = period_data['d'].values.astype(float)
        X = sm.add_constant(np.column_stack([
            d_vals,
            period_data['x1'].values.astype(float),
            period_data['x2'].values.astype(float),
        ]))
        
        model = sm.OLS(y_vals, X).fit(cov_type='HC1')
        
        att = model.params[1]
        se = model.bse[1]
        
        # 验证结果合理
        assert att is not None and not np.isnan(att), "ATT should not be NaN"
        assert se > 0, "SE should be positive"
        assert abs(att) < 100, "ATT seems unreasonably large"
        assert se < 10, "SE seems unreasonably large"


class TestRAFormulaValidation:
    """RA公式验证测试"""
    
    def test_ra_is_not_simple_ols_coefficient(self, transformed_data):
        """
        验证RA估计与简单OLS回归中D的系数不同
        
        重要区别:
        - 简单OLS: Y ~ 1 + D + X → β_D是条件ATE
        - RA (teffects ra): 在控制组估计结果模型，预测处理组反事实
        
        两者只有在特定条件下（如线性模型+同质效应）才相等
        """
        period_data = transformed_data[transformed_data['period'] == 4].copy()
        
        # 方法1: 简单OLS
        y = period_data['y_dot'].values.astype(float)
        X = sm.add_constant(np.column_stack([
            period_data['d'].values.astype(float),
            period_data['x1'].values.astype(float),
            period_data['x2'].values.astype(float),
        ]))
        model = sm.OLS(y, X).fit()
        ols_att = model.params[1]
        
        # 方法2: RA估计
        treated_mask = period_data['d'].values == 1
        control_mask = ~treated_mask
        
        X_controls = period_data[['x1', 'x2']].values.astype(float)
        
        # 控制组OLS
        y_control = y[control_mask]
        X_control = sm.add_constant(X_controls[control_mask])
        model_control = sm.OLS(y_control, X_control).fit()
        
        # 预测处理组反事实
        X_treated = sm.add_constant(X_controls[treated_mask])
        y_counterfactual = model_control.predict(X_treated)
        
        # RA ATT
        ra_att = np.mean(y[treated_mask] - y_counterfactual)
        
        # 验证:
        # 1. RA ATT应该与Stata一致
        stata_att = 4.0830643
        assert abs(ra_att - stata_att) / abs(stata_att) < 1e-5, \
            "RA ATT should match Stata"
        
        # 2. 简单OLS与RA不完全相同 (但方向一致)
        assert ols_att > 0 and ra_att > 0, \
            "Both estimates should be positive"
    
    def test_ra_treated_control_means(self, transformed_data):
        """
        验证处理组和控制组均值差异
        
        未调整的均值差: mean(Y|D=1) - mean(Y|D=0)
        应该接近但不等于调整后的ATT (因为协变量分布不同)
        """
        period_data = transformed_data[transformed_data['period'] == 4].copy()
        
        treated = period_data[period_data['d'] == 1]['y_dot']
        control = period_data[period_data['d'] == 0]['y_dot']
        
        raw_diff = treated.mean() - control.mean()
        
        # 未调整差异应该与调整后ATT不完全相同
        # (如果完全相同，说明协变量没有作用)
        stata_att = 4.0830643
        
        # 只检查符号一致和数量级合理
        assert raw_diff > 0, "Raw difference should be positive"
        assert abs(raw_diff - stata_att) / abs(stata_att) < 0.5, \
            "Raw difference should be in same ballpark as adjusted ATT"


class TestRAWithLwdidFunction:
    """使用lwdid函数的RA测试"""
    
    @pytest.mark.skip(reason="lwdid common timing mode needs estimator parameter support")
    def test_lwdid_ra_common_timing(self, common_timing_data):
        """
        测试通过lwdid函数的Common Timing RA估计
        
        注意: 当前lwdid()的Common Timing模式不支持estimator参数
        这个测试将在Phase 4中启用
        """
        from lwdid import lwdid
        
        # 准备数据
        data = common_timing_data.copy()
        data['post'] = (data['period'] >= 4).astype(int)
        
        # 运行lwdid
        result = lwdid(
            data=data,
            y='y',
            d='d',
            ivar='id',
            tvar='year',
            post='post',
            rolling='demean',
            controls=['x1', 'x2'],
            vce='robust',
        )
        
        # 验证
        assert result.att is not None
        assert result.se_att > 0

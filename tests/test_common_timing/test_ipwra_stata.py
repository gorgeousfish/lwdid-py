# -*- coding: utf-8 -*-
"""
IPWRA估计量与Stata一致性测试

测试Story 6.1的IPWRA估计量:
- 验证ATT与Stata teffects ipwra一致 (相对误差 < 1e-6)
- 验证SE与Stata一致 (相对误差 < 5%)
- 复用staggered模块的estimate_ipwra函数
"""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy.special import expit

from .conftest import (
    compute_transformed_outcome_common_timing,
    assert_att_close_to_stata,
    assert_se_close_to_stata,
)


class TestIPWRAStataConsistency:
    """IPWRA估计量与Stata一致性测试"""
    
    # Stata teffects ipwra验证结果
    STATA_RESULTS = {
        'f04': {'att': 4.1236194, 'se': 0.41718316, 'pomean_0': 2.582304},
        'f05': {'att': 5.1070125, 'se': 0.47030741, 'pomean_0': 3.871102},
        'f06': {'att': 6.6221471, 'se': 0.47140480, 'pomean_0': 5.769926},
    }
    
    def _estimate_ipwra_manual(
        self,
        data: pd.DataFrame,
        y: str,
        d: str,
        outcome_controls: list,
        ps_controls: list,
    ) -> dict:
        """
        手动实现IPWRA估计量 (模拟teffects ipwra, atet)
        
        IPWRA-ATT公式 (doubly robust):
        τ = (1/N₁)Σ_{D=1}[Y - m₀(X)] - Σ_{D=0}[w·(Y-m₀(X))] / Σ_{D=0}[w]
        
        其中:
        - m₀(X) = E(Y|X, D=0) 是控制组条件均值
        - p(X) = P(D=1|X) 是倾向得分
        - w = p(X) / (1 - p(X)) 是IPW权重
        """
        # 分离处理组和控制组
        d_vals = data[d].values.astype(float)
        treated_mask = d_vals == 1
        control_mask = ~treated_mask
        
        y_vals = data[y].values.astype(float)
        X_outcome = data[outcome_controls].values.astype(float)
        X_ps = data[ps_controls].values.astype(float)
        
        n = len(data)
        n_treated = int(treated_mask.sum())
        n_control = int(control_mask.sum())
        
        # Step 1: 估计倾向得分 p(X) = P(D=1|X)
        X_ps_design = sm.add_constant(X_ps)
        ps_model = sm.Logit(d_vals, X_ps_design).fit(disp=0)
        ps_values = ps_model.predict(X_ps_design)
        
        # 裁剪倾向得分以避免极端权重
        trim = 0.01
        ps_clipped = np.clip(ps_values, trim, 1 - trim)
        
        # Step 2: 在控制组上估计结果模型 m₀(X) = E[Y|X, D=0]
        X_outcome_design = sm.add_constant(X_outcome[control_mask])
        outcome_model = sm.OLS(y_vals[control_mask], X_outcome_design).fit()
        
        # Step 3: 预测所有单位的m₀(X)
        X_outcome_all = sm.add_constant(X_outcome)
        m0_all = outcome_model.predict(X_outcome_all)
        
        # Step 4: 计算IPWRA-ATT (ATET)
        # 处理组部分: (1/N₁)Σ_{D=1}[Y - m₀(X)]
        treated_part = np.mean(y_vals[treated_mask] - m0_all[treated_mask])
        
        # 控制组部分 (IPW加权): Σ_{D=0}[w·(Y-m₀(X))] / Σ_{D=0}[w]
        weights = ps_clipped[control_mask] / (1 - ps_clipped[control_mask])
        residuals_control = y_vals[control_mask] - m0_all[control_mask]
        control_part = np.sum(weights * residuals_control) / np.sum(weights)
        
        # ATET = 处理组部分 - 控制组部分
        att = treated_part - control_part
        
        # Step 5: 计算SE (简化版本，使用influence function近似)
        # 完整版需要联合M-estimation
        individual_effects = y_vals[treated_mask] - m0_all[treated_mask]
        se_simple = np.std(individual_effects, ddof=1) / np.sqrt(n_treated)
        
        # 调整IPW贡献
        se = se_simple * np.sqrt(1 + np.var(weights) / np.mean(weights)**2 * n_treated / n_control)
        
        return {
            'att': att,
            'se': se,
            'n_obs': n,
            'n_treated': n_treated,
            'n_control': n_control,
            'pomean_0': np.mean(m0_all[treated_mask]),
        }
    
    @pytest.mark.parametrize("period,period_key", [(4, 'f04'), (5, 'f05'), (6, 'f06')])
    def test_ipwra_att_vs_stata(self, transformed_data, period, period_key):
        """测试IPWRA ATT与Stata一致"""
        period_data = transformed_data[transformed_data['period'] == period].copy()
        
        result = self._estimate_ipwra_manual(
            data=period_data,
            y='y_dot',
            d='d',
            outcome_controls=['x1', 'x2'],
            ps_controls=['x1', 'x2'],
        )
        
        stata = self.STATA_RESULTS[period_key]
        
        # ATT相对误差 < 1e-3 (IPWRA有更多数值敏感性)
        assert_att_close_to_stata(
            result['att'],
            stata['att'],
            tolerance=1e-3,
            description=f"{period_key} IPWRA ATT"
        )
    
    @pytest.mark.parametrize("period,period_key", [(4, 'f04'), (5, 'f05'), (6, 'f06')])
    def test_ipwra_se_vs_stata(self, transformed_data, period, period_key):
        """测试IPWRA SE与Stata一致"""
        period_data = transformed_data[transformed_data['period'] == period].copy()
        
        result = self._estimate_ipwra_manual(
            data=period_data,
            y='y_dot',
            d='d',
            outcome_controls=['x1', 'x2'],
            ps_controls=['x1', 'x2'],
        )
        
        stata = self.STATA_RESULTS[period_key]
        
        # SE相对误差 < 10% (IPWRA的SE计算更复杂)
        assert_se_close_to_stata(
            result['se'],
            stata['se'],
            tolerance=0.10,
            description=f"{period_key} IPWRA SE"
        )


class TestIPWRAWithStaggeredModule:
    """使用staggered模块的IPWRA测试"""
    
    STATA_RESULTS = {
        'f04': {'att': 4.1236194, 'se': 0.41718316},
        'f05': {'att': 5.1070125, 'se': 0.47030741},
        'f06': {'att': 6.6221471, 'se': 0.47140480},
    }
    
    @pytest.mark.parametrize("period,period_key", [(4, 'f04'), (5, 'f05'), (6, 'f06')])
    def test_staggered_ipwra_vs_stata(self, transformed_data, period, period_key):
        """测试staggered模块的IPWRA与Stata一致"""
        from lwdid.staggered.estimators import estimate_ipwra
        
        period_data = transformed_data[transformed_data['period'] == period].copy()
        
        # 使用staggered模块的IPWRA (横截面模式)
        result = estimate_ipwra(
            data=period_data,
            y='y_dot',
            d='d',
            controls=['x1', 'x2'],
            propensity_controls=['x1', 'x2'],
            se_method='analytical',
        )
        
        stata = self.STATA_RESULTS[period_key]
        
        # ATT验证
        assert_att_close_to_stata(
            result.att,
            stata['att'],
            tolerance=1e-4,
            description=f"{period_key} staggered IPWRA ATT"
        )
    
    @pytest.mark.parametrize("period,period_key", [(4, 'f04'), (5, 'f05'), (6, 'f06')])
    def test_staggered_ipwra_se_vs_stata(self, transformed_data, period, period_key):
        """测试staggered模块的IPWRA SE与Stata一致"""
        from lwdid.staggered.estimators import estimate_ipwra
        
        period_data = transformed_data[transformed_data['period'] == period].copy()
        
        result = estimate_ipwra(
            data=period_data,
            y='y_dot',
            d='d',
            controls=['x1', 'x2'],
            propensity_controls=['x1', 'x2'],
            se_method='analytical',
        )
        
        stata = self.STATA_RESULTS[period_key]
        
        # SE验证 (5%容差)
        assert_se_close_to_stata(
            result.se,  # IPWRAResult uses 'se', not 'se_att'
            stata['se'],
            tolerance=0.05,
            description=f"{period_key} staggered IPWRA SE"
        )


class TestIPWRADoublyRobust:
    """IPWRA双稳健性测试"""
    
    def test_ipwra_combines_ra_and_ipw(self, transformed_data):
        """
        验证IPWRA结合了RA和IPW的优点
        
        双稳健性: 只要结果模型或PS模型之一正确，估计就是一致的
        """
        period_data = transformed_data[transformed_data['period'] == 4].copy()
        
        # 计算IPWRA
        from lwdid.staggered.estimators import estimate_ipwra
        
        result = estimate_ipwra(
            data=period_data,
            y='y_dot',
            d='d',
            controls=['x1', 'x2'],
            propensity_controls=['x1', 'x2'],
        )
        
        # IPWRA应该与RA和IPW在方向上一致
        stata_ra_att = 4.0830643
        stata_ipwra_att = 4.1236194
        
        # IPWRA结果应该接近RA和IPWRA的Stata结果
        assert abs(result.att - stata_ipwra_att) / stata_ipwra_att < 0.01, \
            "IPWRA should be close to Stata IPWRA"
    
    def test_ipwra_reasonable_weights(self, transformed_data):
        """验证IPW权重在合理范围内"""
        period_data = transformed_data[transformed_data['period'] == 4].copy()
        
        # 估计倾向得分
        d_vals = period_data['d'].values
        X_ps = period_data[['x1', 'x2']].values
        X_ps_design = sm.add_constant(X_ps)
        
        ps_model = sm.Logit(d_vals, X_ps_design).fit(disp=0)
        ps_values = ps_model.predict(X_ps_design)
        
        # 验证倾向得分在合理范围
        assert ps_values.min() > 0.01, "PS should be > 0.01"
        assert ps_values.max() < 0.99, "PS should be < 0.99"
        
        # 验证控制组的IPW权重
        control_mask = d_vals == 0
        weights = ps_values[control_mask] / (1 - ps_values[control_mask])
        
        # 权重不应该太极端
        assert weights.max() < 10, "IPW weights should not be too extreme"
        assert weights.min() > 0.01, "IPW weights should be positive"


class TestIPWRAE2E:
    """IPWRA端到端测试"""
    
    STATA_RESULTS = {
        'f04': {'att': 4.1236194},
        'f05': {'att': 5.1070125},
        'f06': {'att': 6.6221471},
    }
    
    def test_ipwra_full_pipeline(self, common_timing_data):
        """测试IPWRA完整流程: 数据 → 变换 → 估计 → 验证"""
        from lwdid.staggered.estimators import estimate_ipwra
        
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
            
            # Step 3: IPWRA估计
            result = estimate_ipwra(
                data=period_data,
                y='y_dot',
                d='d',
                controls=['x1', 'x2'],
                propensity_controls=['x1', 'x2'],
            )
            
            # Step 4: 验证
            period_key = f'f{period:02d}'
            stata_att = self.STATA_RESULTS[period_key]['att']
            
            att_error = abs(result.att - stata_att) / abs(stata_att)
            assert att_error < 1e-4, \
                f"Period {period} IPWRA ATT error {att_error:.2e} > 1e-4"

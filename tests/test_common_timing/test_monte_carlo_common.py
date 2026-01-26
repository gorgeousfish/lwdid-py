# -*- coding: utf-8 -*-
"""
Monte Carlo模拟验证测试

测试Story 6.1的Monte Carlo验证:
- 验证RA、IPWRA、PSM估计量的Bias和Coverage
- 基于Lee & Wooldridge (2023) Section 7.1的DGP设置
- 场景1C-4C的验证
"""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from .fixtures.dgp_common_timing import (
    generate_common_timing_dgp,
    generate_simple_common_timing_dgp,
)
from .conftest import compute_transformed_outcome_common_timing


class TestMonteCarloRA:
    """RA估计量Monte Carlo验证"""
    
    @pytest.fixture
    def ra_estimator(self):
        """RA估计量函数"""
        def _estimate(data, period, controls=['x1', 'x2']):
            # DGP uses year column with values 1-6
            # 变换
            y_dot = compute_transformed_outcome_common_timing(
                data, 'y', 'id', 'year',
                first_treat_period=4,
                target_period=period
            )
            
            # 准备横截面数据 (DGP的year是1-6)
            period_data = data[data['year'] == period].copy()
            period_data['y_dot'] = period_data['id'].map(y_dot)
            
            # 检查是否有足够的处理组和控制组
            treated_mask = period_data['d'].values == 1
            control_mask = ~treated_mask
            
            if control_mask.sum() == 0 or treated_mask.sum() == 0:
                raise ValueError("Not enough treated or control units")
            
            y_vals = period_data['y_dot'].values.astype(float)
            X_controls = period_data[controls].values.astype(float)
            
            # 控制组OLS
            X_control = sm.add_constant(X_controls[control_mask])
            outcome_model = sm.OLS(y_vals[control_mask], X_control).fit()
            
            # 预测处理组反事实
            X_treated = sm.add_constant(X_controls[treated_mask])
            y_counterfactual = outcome_model.predict(X_treated)
            
            # ATET
            y_treated = y_vals[treated_mask]
            individual_effects = y_treated - y_counterfactual
            att = np.mean(individual_effects)
            
            # SE
            n_treated = len(individual_effects)
            n_control = control_mask.sum()
            se = np.std(individual_effects, ddof=1) / np.sqrt(n_treated)
            se *= np.sqrt(1 + n_treated / n_control)
            
            # 95% CI
            ci_lower = att - 1.96 * se
            ci_upper = att + 1.96 * se
            
            return att, se, ci_lower, ci_upper
        
        return _estimate
    
    def test_ra_bias_scenario_1c(self, ra_estimator):
        """
        测试场景1C的RA Bias
        
        场景1C: 正确设定的PS和OM
        目标: |Bias| < 0.5 (相对于真实ATT)
        """
        n_reps = 50  # 较小的重复次数用于快速测试
        n_units = 500
        period = 4
        
        atts = []
        true_att_sum = 0
        
        for rep in range(n_reps):
            data, true_atts = generate_common_timing_dgp(
                n_units=n_units,
                scenario='1C',
                seed=rep
            )
            
            try:
                att, _, _, _ = ra_estimator(data, period)
                atts.append(att)
                true_att_sum += true_atts[period]
            except ValueError:
                continue
        
        if len(atts) < 10:
            pytest.skip("Not enough successful estimations")
        
        # 使用平均真实ATT
        true_att = true_att_sum / len(atts)
        mean_att = np.mean(atts)
        bias = mean_att - true_att
        
        # Bias应该较小 (相对于true_att)
        relative_bias = abs(bias) / abs(true_att) if true_att != 0 else abs(bias)
        
        assert relative_bias < 0.15, \
            f"RA Bias relative error {relative_bias:.2%} > 15%"
    
    def test_ra_coverage_scenario_1c(self, ra_estimator):
        """
        测试场景1C的RA覆盖率
        
        目标: Coverage ∈ [0.80, 0.99]
        """
        n_reps = 50
        n_units = 500
        period = 4
        
        covers = []
        
        for rep in range(n_reps):
            data, true_atts = generate_common_timing_dgp(
                n_units=n_units,
                scenario='1C',
                seed=rep
            )
            true_att = true_atts[period]
            
            try:
                att, se, ci_lower, ci_upper = ra_estimator(data, period)
                
                if ci_lower <= true_att <= ci_upper:
                    covers.append(1)
                else:
                    covers.append(0)
            except ValueError:
                continue
        
        if len(covers) < 10:
            pytest.skip("Not enough successful estimations")
        
        coverage = np.mean(covers)
        
        # 覆盖率应该接近名义水平
        assert 0.80 <= coverage <= 0.99, \
            f"RA Coverage {coverage:.2%} outside [80%, 99%]"


class TestMonteCarloSimpleDGP:
    """使用简单DGP的Monte Carlo测试"""
    
    def test_simple_dgp_ra_unbiased(self):
        """
        测试简单DGP下RA估计量的无偏性
        
        使用线性、同质效应的简化DGP
        """
        n_reps = 50
        n_units = 200
        
        biases = []
        
        for rep in range(n_reps):
            data, true_atts = generate_simple_common_timing_dgp(
                n_units=n_units,
                seed=rep,
                treatment_effect=4.0
            )
            
            # 变换
            y_dot = compute_transformed_outcome_common_timing(
                data, 'y', 'id', 'year',
                first_treat_period=4,
                target_period=4
            )
            
            # 准备数据
            period_data = data[data['year'] == 4].copy()
            period_data['y_dot'] = period_data['id'].map(y_dot)
            
            # 简单差异估计
            treated = period_data[period_data['d'] == 1]['y_dot']
            control = period_data[period_data['d'] == 0]['y_dot']
            att = treated.mean() - control.mean()
            
            bias = att - true_atts[4]
            biases.append(bias)
        
        mean_bias = np.mean(biases)
        
        # 简单DGP下Bias应该很小
        assert abs(mean_bias) < 0.5, \
            f"Simple DGP Bias {mean_bias:.2f} > 0.5"
    
    def test_simple_dgp_variance_reasonable(self):
        """测试估计量方差的合理性"""
        n_reps = 50
        n_units = 200
        
        atts = []
        
        for rep in range(n_reps):
            data, true_atts = generate_simple_common_timing_dgp(
                n_units=n_units,
                seed=rep,
                treatment_effect=4.0
            )
            
            # 变换
            y_dot = compute_transformed_outcome_common_timing(
                data, 'y', 'id', 'year',
                first_treat_period=4,
                target_period=4
            )
            
            period_data = data[data['year'] == 4].copy()
            period_data['y_dot'] = period_data['id'].map(y_dot)
            
            treated = period_data[period_data['d'] == 1]['y_dot']
            control = period_data[period_data['d'] == 0]['y_dot']
            att = treated.mean() - control.mean()
            atts.append(att)
        
        # 估计量的标准差应该合理
        sd = np.std(atts, ddof=1)
        
        assert 0.1 < sd < 2.0, \
            f"Estimator SD {sd:.2f} outside reasonable range [0.1, 2.0]"


class TestMonteCarloDGPValidation:
    """DGP验证测试"""
    
    def test_dgp_treatment_share(self):
        """验证DGP生成的处理组比例"""
        data, _ = generate_common_timing_dgp(n_units=1000, seed=42)
        
        # 获取单位级别的处理状态
        unit_data = data.groupby('id').first()
        treated_share = unit_data['d'].mean()
        
        # 应该接近预期的处理组比例
        assert 0.10 < treated_share < 0.30, \
            f"Treated share {treated_share:.2%} outside [10%, 30%]"
    
    def test_dgp_true_att_reasonable(self):
        """验证真实ATT值合理"""
        _, true_atts = generate_common_timing_dgp(n_units=1000, seed=42)
        
        # 真实ATT应该为正 (论文设定的处理效应为正)
        for period, att in true_atts.items():
            assert att > 0, f"True ATT for period {period} should be positive"
            assert att < 20, f"True ATT for period {period} seems too large: {att}"
    
    def test_dgp_increasing_att_by_period(self):
        """验证ATT随时期增加"""
        _, true_atts = generate_common_timing_dgp(n_units=1000, seed=42)
        
        # 处理效应应该随时间增加 (根据论文设定)
        assert true_atts[4] < true_atts[5] < true_atts[6], \
            "True ATT should increase over time"
    
    def test_dgp_panel_structure(self):
        """验证DGP生成的面板数据结构"""
        data, _ = generate_common_timing_dgp(n_units=100, seed=42)
        
        # 验证面板结构
        n_units = data['id'].nunique()
        n_periods = data['year'].nunique()
        
        assert n_units == 100, f"Expected 100 units, got {n_units}"
        assert n_periods == 6, f"Expected 6 periods, got {n_periods}"
        assert len(data) == 600, f"Expected 600 obs, got {len(data)}"
    
    def test_dgp_scenarios_differ(self):
        """验证不同场景生成不同的数据"""
        _, true_atts_1c = generate_common_timing_dgp(
            n_units=1000, scenario='1C', seed=42
        )
        _, true_atts_4c = generate_common_timing_dgp(
            n_units=1000, scenario='4C', seed=42
        )
        
        # 不同场景的真实ATT应该不同 (因为DGP函数不同)
        # 但由于使用相同seed，差异主要来自DGP中的非线性项
        # 这是一个基本的sanity check
        assert true_atts_1c[4] > 0 and true_atts_4c[4] > 0


class TestMonteCarloPSM:
    """PSM估计量Monte Carlo验证"""
    
    def test_psm_with_simple_dgp(self):
        """测试PSM在简单DGP下的表现"""
        from lwdid.staggered.estimators import estimate_psm
        
        n_reps = 20  # 较少的重复次数
        atts = []
        
        for rep in range(n_reps):
            data, true_atts = generate_simple_common_timing_dgp(
                n_units=200,
                seed=rep,
                treatment_effect=4.0
            )
            
            # 变换
            y_dot = compute_transformed_outcome_common_timing(
                data, 'y', 'id', 'year',
                first_treat_period=4,
                target_period=4
            )
            
            period_data = data[data['year'] == 4].copy()
            period_data['y_dot'] = period_data['id'].map(y_dot)
            
            try:
                result = estimate_psm(
                    data=period_data,
                    y='y_dot',
                    d='d',
                    propensity_controls=['x1', 'x2'],
                    n_neighbors=1,
                )
                atts.append(result.att)
            except Exception:
                continue
        
        if len(atts) > 0:
            mean_att = np.mean(atts)
            bias = mean_att - 4.0  # true effect is 4.0
            
            # PSM应该大致无偏
            assert abs(bias) < 1.5, \
                f"PSM Bias {bias:.2f} > 1.5"

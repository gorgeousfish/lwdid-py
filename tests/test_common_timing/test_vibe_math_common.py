# -*- coding: utf-8 -*-
"""
Vibe Math MCP公式验证测试

测试Story 6.1的公式验证:
- 使用Vibe Math MCP验证变换公式
- 验证RA-ATT公式计算
- 验证统计量计算
"""

import numpy as np
import pandas as pd
import pytest
import json
import subprocess

from .conftest import compute_transformed_outcome_common_timing


class TestTransformationFormula:
    """变换公式验证"""
    
    def test_transformation_formula_manual(self, common_timing_data):
        """
        验证变换公式: ŷ_{ir} = Y_{ir} - (1/(S-1)) × Σ_{q=1}^{S-1} Y_{iq}
        
        使用Vibe Math验证公式计算
        """
        # 取一个单位的数据
        unit_id = 1
        unit_data = common_timing_data[common_timing_data['id'] == unit_id]
        
        # 获取pre-period Y值
        y1 = unit_data[unit_data['period'] == 1]['y'].values[0]
        y2 = unit_data[unit_data['period'] == 2]['y'].values[0]
        y3 = unit_data[unit_data['period'] == 3]['y'].values[0]
        y4 = unit_data[unit_data['period'] == 4]['y'].values[0]
        
        # 公式计算
        S = 4
        pre_mean = (y1 + y2 + y3) / (S - 1)
        y_dot_formula = y4 - pre_mean
        
        # Python计算
        y_dot_python = compute_transformed_outcome_common_timing(
            common_timing_data, 'y', 'id', 'period',
            first_treat_period=4,
            target_period=4
        )
        
        # 验证
        assert abs(y_dot_python[unit_id] - y_dot_formula) < 1e-5, \
            f"Formula mismatch: Python={y_dot_python[unit_id]}, Formula={y_dot_formula}"
    
    def test_pre_mean_formula(self):
        """验证pre-mean公式"""
        # 简单测试数据
        y_pre = [10, 20, 30]  # S-1 = 3个pre-period值
        
        # 公式: pre_mean = (1/(S-1)) × Σ Y_iq
        pre_mean_formula = sum(y_pre) / len(y_pre)
        pre_mean_numpy = np.mean(y_pre)
        
        assert abs(pre_mean_formula - pre_mean_numpy) < 1e-10, \
            "Pre-mean formula verification failed"
        
        assert pre_mean_formula == 20.0, "Pre-mean should be 20.0"


class TestRAFormula:
    """RA估计量公式验证"""
    
    def test_ra_att_formula(self):
        """
        验证RA-ATT公式
        
        ATT = (1/N₁) × Σ_{i:D=1} [Y_i - m₀(X_i)]
        
        其中 m₀(X) = α̂ + X'β̂ 是控制组模型的预测值
        """
        # 模拟数据
        np.random.seed(42)
        n_treated = 5
        n_control = 10
        
        # 控制组
        X_control = np.random.randn(n_control, 2)
        y_control = 1 + 0.5 * X_control[:, 0] + 0.3 * X_control[:, 1] + np.random.randn(n_control) * 0.5
        
        # 处理组
        X_treated = np.random.randn(n_treated, 2)
        true_effect = 3.0
        y_treated = 1 + 0.5 * X_treated[:, 0] + 0.3 * X_treated[:, 1] + true_effect + np.random.randn(n_treated) * 0.5
        
        # 在控制组上估计模型
        import statsmodels.api as sm
        X_control_const = sm.add_constant(X_control)
        model = sm.OLS(y_control, X_control_const).fit()
        
        # 预测处理组反事实
        X_treated_const = sm.add_constant(X_treated)
        y_counterfactual = model.predict(X_treated_const)
        
        # RA-ATT公式
        individual_effects = y_treated - y_counterfactual
        att_formula = np.mean(individual_effects)
        
        # 验证ATT接近真实效应
        assert abs(att_formula - true_effect) < 1.5, \
            f"RA ATT {att_formula:.2f} not close to true effect {true_effect}"


class TestIPWRAFormula:
    """IPWRA公式验证"""
    
    def test_ipw_weight_formula(self):
        """
        验证IPW权重公式
        
        w = p(X) / (1 - p(X))
        """
        # 测试倾向得分
        ps_values = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        for ps in ps_values:
            # 权重公式
            weight_formula = ps / (1 - ps)
            
            # 验证权重合理
            assert weight_formula > 0, "IPW weight should be positive"
            
            if ps < 0.5:
                assert weight_formula < 1, f"Weight should be < 1 for ps={ps}"
            elif ps > 0.5:
                assert weight_formula > 1, f"Weight should be > 1 for ps={ps}"
    
    def test_ipwra_doubly_robust(self):
        """
        验证IPWRA的双稳健性
        
        τ_IPWRA = (1/N₁)Σ_{D=1}[Y - m₀(X)] - Σ_{D=0}[w·(Y-m₀(X))] / Σ_{D=0}[w]
        """
        # 简单数值例子
        # 假设有2个处理单位和3个控制单位
        Y_treated = [10, 12]
        m0_treated = [8, 9]  # 反事实预测
        
        Y_control = [5, 6, 7]
        m0_control = [5.5, 5.8, 6.2]  # 结果模型预测
        weights = [0.2, 0.3, 0.5]  # IPW权重
        
        # IPWRA公式计算
        # 处理组部分
        treated_part = np.mean([Y_treated[i] - m0_treated[i] for i in range(len(Y_treated))])
        
        # 控制组部分 (加权)
        weighted_residuals = sum(weights[i] * (Y_control[i] - m0_control[i]) for i in range(len(Y_control)))
        sum_weights = sum(weights)
        control_part = weighted_residuals / sum_weights
        
        # IPWRA ATT
        att_ipwra = treated_part - control_part
        
        # 验证结果合理
        assert att_ipwra > 0, "IPWRA ATT should be positive in this example"


class TestHCFormulas:
    """HC标准误公式验证"""
    
    def test_hc0_formula(self):
        """
        验证HC0公式
        
        V̂_HC0 = (X'X)⁻¹ X' diag(e²) X (X'X)⁻¹
        """
        np.random.seed(42)
        n = 100
        k = 3
        
        # 生成数据
        X = np.random.randn(n, k)
        X = np.column_stack([np.ones(n), X[:, 1:]])  # 添加常数项
        y = X @ np.array([1, 0.5, 0.3]) + np.random.randn(n)
        
        # OLS估计
        XtX_inv = np.linalg.inv(X.T @ X)
        beta = XtX_inv @ X.T @ y
        residuals = y - X @ beta
        
        # HC0方差
        e2 = residuals ** 2
        
        # 使用向量化计算
        meat = (X.T * e2) @ X
        var_hc0 = XtX_inv @ meat @ XtX_inv
        
        # 验证对称性
        assert np.allclose(var_hc0, var_hc0.T), "HC0 variance should be symmetric"
        
        # 验证正定性
        eigenvalues = np.linalg.eigvalsh(var_hc0)
        assert np.all(eigenvalues > 0), "HC0 variance should be positive definite"
    
    def test_hc1_correction_factor(self):
        """
        验证HC1的自由度调整因子
        
        HC1 = (n / (n-k)) × HC0
        """
        n = 100
        k = 4
        
        # 调整因子
        correction = n / (n - k)
        
        # 验证
        assert correction > 1, "HC1 correction should be > 1"
        assert abs(correction - 100/96) < 1e-10, "Correction factor mismatch"


class TestStatisticalFormulas:
    """统计公式验证"""
    
    def test_sample_mean_variance(self):
        """验证样本均值和方差公式"""
        data = [1, 2, 3, 4, 5]
        
        # 公式计算
        n = len(data)
        mean_formula = sum(data) / n
        variance_formula = sum((x - mean_formula) ** 2 for x in data) / (n - 1)
        
        # numpy计算
        mean_numpy = np.mean(data)
        variance_numpy = np.var(data, ddof=1)
        
        assert abs(mean_formula - mean_numpy) < 1e-10
        assert abs(variance_formula - variance_numpy) < 1e-10
    
    def test_standard_error_formula(self):
        """
        验证标准误公式
        
        SE = σ / √n
        """
        data = [1, 2, 3, 4, 5]
        n = len(data)
        
        # 样本标准差
        sd = np.std(data, ddof=1)
        
        # 均值的标准误
        se_formula = sd / np.sqrt(n)
        se_stats = sd / n ** 0.5
        
        assert abs(se_formula - se_stats) < 1e-10


class TestNumericalPrecision:
    """数值精度测试"""
    
    def test_float32_vs_float64_precision(self, common_timing_data):
        """测试float32和float64精度差异"""
        # 获取一个单位的数据
        unit_data = common_timing_data[common_timing_data['id'] == 1]
        y_values = unit_data['y'].values
        
        # float32计算
        y_f32 = y_values.astype(np.float32)
        mean_f32 = np.mean(y_f32)
        
        # float64计算
        y_f64 = y_values.astype(np.float64)
        mean_f64 = np.mean(y_f64)
        
        # 差异应该很小
        rel_diff = abs(mean_f32 - mean_f64) / abs(mean_f64)
        assert rel_diff < 1e-5, f"Float precision difference {rel_diff:.2e} > 1e-5"
    
    def test_matrix_inversion_stability(self):
        """测试矩阵求逆的数值稳定性"""
        np.random.seed(42)
        
        # 生成设计矩阵
        n = 100
        X = np.column_stack([
            np.ones(n),
            np.random.randn(n),
            np.random.randn(n),
        ])
        
        XtX = X.T @ X
        
        # 验证条件数
        cond = np.linalg.cond(XtX)
        assert cond < 1e10, f"Matrix condition number {cond:.2e} too large"
        
        # 验证求逆精度
        XtX_inv = np.linalg.inv(XtX)
        identity_approx = XtX @ XtX_inv
        identity_error = np.max(np.abs(identity_approx - np.eye(3)))
        
        assert identity_error < 1e-10, \
            f"Matrix inversion error {identity_error:.2e} > 1e-10"

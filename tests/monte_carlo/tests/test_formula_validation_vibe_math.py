# -*- coding: utf-8 -*-
"""
公式验证器测试 (使用 vibe-math MCP)

Task 5: 实现公式验证器
- 5.1 验证DGP公式 (Gamma分布参数, PS公式计算)
- 5.2 验证ATT聚合公式 (加权平均)
- 5.3 验证标准误计算公式
- 5.4 验证置信区间临界值

Validates: Requirements 6.1, 6.2, 6.3, 6.4

References
----------
Lee & Wooldridge (2023, 2026)
ssrn-4516518 Appendix C
ssrn-5325686 Section 5
"""

import pytest
import numpy as np
from scipy import stats


class TestDGPFormulaValidation:
    """
    Task 5.1: 验证DGP公式
    Validates: Requirements 6.1
    """
    
    def test_gamma_distribution_parameters(self):
        """
        验证Gamma分布参数: X1 ~ Gamma(shape=2, scale=2)
        E(X1) = shape × scale = 2 × 2 = 4
        Var(X1) = shape × scale² = 2 × 4 = 8
        """
        # 理论值
        shape = 2
        scale = 2
        expected_mean = shape * scale  # = 4
        expected_var = shape * scale ** 2  # = 8
        
        # 验证
        assert expected_mean == 4, f"Expected mean 4, got {expected_mean}"
        assert expected_var == 8, f"Expected variance 8, got {expected_var}"
        
        # 使用scipy验证
        gamma_dist = stats.gamma(a=shape, scale=scale)
        assert np.isclose(gamma_dist.mean(), 4)
        assert np.isclose(gamma_dist.var(), 8)
    
    def test_ps_formula_correct_specification(self):
        """
        验证正确设定的PS公式 (C.3):
        Z₁γ₁ = -1.2 + (X₁-4)/2 - X₂
        
        测试点: X1=6, X2=0
        PS_index = -1.2 + (6-4)/2 - 0 = -1.2 + 1 = -0.2
        """
        X1, X2 = 6, 0
        ps_index = -1.2 + (X1 - 4) / 2 - X2
        
        expected = -0.2
        assert np.isclose(ps_index, expected), f"Expected {expected}, got {ps_index}"
    
    def test_ps_formula_misspecified(self):
        """
        验证错误设定的PS公式 (C.10):
        Z₂γ₂ = -1.2 + (X₁-4)/2 - X₂ + (X₁-4)²/2
        
        测试点: X1=6, X2=0
        PS_index = -1.2 + (6-4)/2 - 0 + (6-4)²/2
                 = -1.2 + 1 + 2 = 1.8
        """
        X1, X2 = 6, 0
        ps_index = -1.2 + (X1 - 4) / 2 - X2 + ((X1 - 4) ** 2) / 2
        
        expected = 1.8
        assert np.isclose(ps_index, expected), f"Expected {expected}, got {ps_index}"
    
    def test_ps_formula_difference(self):
        """
        验证正确和错误PS公式的差异
        差异 = (X₁-4)²/2 (二次项)
        
        测试点: X1=6
        差异 = (6-4)²/2 = 4/2 = 2
        """
        X1 = 6
        difference = ((X1 - 4) ** 2) / 2
        
        expected = 2.0
        assert np.isclose(difference, expected), f"Expected {expected}, got {difference}"
    
    def test_logistic_transformation(self):
        """
        验证Logistic变换: P(D=1|X) = 1 / (1 + exp(-PS_index))
        
        测试点:
        - PS_index = 0 → P = 0.5
        - PS_index = -0.2 → P ≈ 0.45
        - PS_index = 1.8 → P ≈ 0.858
        """
        # PS_index = 0
        ps_index = 0
        prob = 1 / (1 + np.exp(-ps_index))
        assert np.isclose(prob, 0.5)
        
        # PS_index = -0.2 (正确设定, X1=6, X2=0)
        ps_index = -0.2
        prob = 1 / (1 + np.exp(-ps_index))
        expected = 1 / (1 + np.exp(0.2))
        assert np.isclose(prob, expected)
        
        # PS_index = 1.8 (错误设定, X1=6, X2=0)
        ps_index = 1.8
        prob = 1 / (1 + np.exp(-ps_index))
        expected = 1 / (1 + np.exp(-1.8))
        assert np.isclose(prob, expected)
        assert prob > 0.85  # 应该接近0.858


class TestARProcessValidation:
    """
    验证AR(1)误差过程公式
    """
    
    def test_ar1_stationary_variance(self):
        """
        验证AR(1)平稳方差: Var(u) = σ_ε² / (1 - ρ²)
        
        参数: ρ = 0.75, σ_ε = √2
        Var(u) = 2 / (1 - 0.5625) = 2 / 0.4375 ≈ 4.571
        """
        rho = 0.75
        sigma_epsilon = np.sqrt(2)
        
        var_u = sigma_epsilon ** 2 / (1 - rho ** 2)
        expected = 2 / (1 - 0.75 ** 2)
        
        assert np.isclose(var_u, expected)
        assert np.isclose(var_u, 2 / 0.4375)
    
    def test_ar1_initial_std(self):
        """
        验证AR(1)初始标准差: σ_u = σ_ε / √(1 - ρ²)
        
        参数: ρ = 0.75, σ_ε = √2
        σ_u = √2 / √0.4375 ≈ 2.138
        """
        rho = 0.75
        sigma_epsilon = np.sqrt(2)
        
        sigma_u = sigma_epsilon / np.sqrt(1 - rho ** 2)
        expected = np.sqrt(2) / np.sqrt(1 - 0.75 ** 2)
        
        assert np.isclose(sigma_u, expected)


class TestATTAggregationFormula:
    """
    Task 5.2: 验证ATT聚合公式
    Validates: Requirements 6.2
    """
    
    def test_simple_average_att(self):
        """
        验证简单平均ATT: ATT_avg = (1/T_post) × Σ ATT_t
        
        测试: ATT = {11: 1, 12: 2, 13: 3, 14: 3, 15: 3, 16: 2, 17: 2, 18: 2, 19: 1, 20: 1}
        平均 = (1+2+3+3+3+2+2+2+1+1) / 10 = 20 / 10 = 2.0
        """
        att_by_period = {11: 1, 12: 2, 13: 3, 14: 3, 15: 3, 16: 2, 17: 2, 18: 2, 19: 1, 20: 1}
        
        avg_att = np.mean(list(att_by_period.values()))
        expected = 2.0
        
        assert np.isclose(avg_att, expected), f"Expected {expected}, got {avg_att}"
    
    def test_weighted_average_att(self):
        """
        验证加权平均ATT: ATT_weighted = Σ(w_t × ATT_t) / Σw_t
        
        测试: ATT = [1, 2, 3], weights = [0.5, 0.3, 0.2]
        加权平均 = (0.5×1 + 0.3×2 + 0.2×3) / 1.0 = 1.7
        """
        att_values = np.array([1, 2, 3])
        weights = np.array([0.5, 0.3, 0.2])
        
        weighted_avg = np.sum(weights * att_values) / np.sum(weights)
        expected = 1.7
        
        assert np.isclose(weighted_avg, expected), f"Expected {expected}, got {weighted_avg}"


class TestStandardErrorFormula:
    """
    Task 5.3: 验证标准误计算公式
    Validates: Requirements 6.3
    """
    
    def test_se_from_variance(self):
        """
        验证SE计算: SE = √(Var(ATT_hat))
        
        测试: Var = 4 → SE = 2
        """
        variance = 4.0
        se = np.sqrt(variance)
        
        assert np.isclose(se, 2.0)
    
    def test_se_ratio_formula(self):
        """
        验证SE Ratio: SE_Ratio = Mean_SE / SD
        
        测试: Mean_SE = 1.5, SD = 1.5 → SE_Ratio = 1.0
        """
        mean_se = 1.5
        sd = 1.5
        
        se_ratio = mean_se / sd
        expected = 1.0
        
        assert np.isclose(se_ratio, expected)
    
    def test_rmse_formula(self):
        """
        验证RMSE公式: RMSE = √(Bias² + SD²)
        
        测试: Bias = 0.3, SD = 0.4 → RMSE = 0.5
        """
        bias = 0.3
        sd = 0.4
        
        rmse = np.sqrt(bias ** 2 + sd ** 2)
        expected = 0.5
        
        assert np.isclose(rmse, expected)


class TestConfidenceIntervalFormula:
    """
    Task 5.4: 验证置信区间临界值
    Validates: Requirements 6.4
    """
    
    def test_normal_critical_value_95(self):
        """
        验证95%置信区间的正态临界值: z_0.975 ≈ 1.96
        """
        z_critical = stats.norm.ppf(0.975)
        
        assert np.isclose(z_critical, 1.96, atol=0.01)
    
    def test_normal_critical_value_90(self):
        """
        验证90%置信区间的正态临界值: z_0.95 ≈ 1.645
        """
        z_critical = stats.norm.ppf(0.95)
        
        assert np.isclose(z_critical, 1.645, atol=0.01)
    
    def test_t_critical_value_small_sample(self):
        """
        验证小样本t分布临界值
        
        df = 18 (N=20, 2 parameters)
        t_0.975 ≈ 2.101
        """
        df = 18
        t_critical = stats.t.ppf(0.975, df)
        
        assert np.isclose(t_critical, 2.101, atol=0.01)
    
    def test_confidence_interval_construction(self):
        """
        验证置信区间构造: CI = [ATT - z × SE, ATT + z × SE]
        
        测试: ATT = 2.0, SE = 0.5, z = 1.96
        CI = [2.0 - 0.98, 2.0 + 0.98] = [1.02, 2.98]
        """
        att = 2.0
        se = 0.5
        z = 1.96
        
        ci_lower = att - z * se
        ci_upper = att + z * se
        
        assert np.isclose(ci_lower, 1.02)
        assert np.isclose(ci_upper, 2.98)


class TestCoverageRateFormula:
    """
    验证覆盖率计算公式
    """
    
    def test_coverage_rate_calculation(self):
        """
        验证覆盖率计算: Coverage = (1/R) × Σ I(CI contains true ATT)
        
        测试: 100次模拟，95次覆盖 → Coverage = 0.95
        """
        n_reps = 100
        n_covers = 95
        
        coverage = n_covers / n_reps
        expected = 0.95
        
        assert np.isclose(coverage, expected)
    
    def test_coverage_indicator(self):
        """
        验证覆盖指示函数: I(ci_lower ≤ true_att ≤ ci_upper)
        """
        true_att = 2.0
        
        # 覆盖情况
        ci_lower, ci_upper = 1.5, 2.5
        covers = (ci_lower <= true_att <= ci_upper)
        assert covers == True
        
        # 不覆盖情况 (true_att在CI外)
        ci_lower, ci_upper = 2.5, 3.5
        covers = (ci_lower <= true_att <= ci_upper)
        assert covers == False


class TestBiasFormula:
    """
    验证Bias计算公式
    """
    
    def test_bias_calculation(self):
        """
        验证Bias计算: Bias = E[ATT_hat] - ATT_true
        
        测试: Mean_ATT = 2.1, True_ATT = 2.0 → Bias = 0.1
        """
        mean_att = 2.1
        true_att = 2.0
        
        bias = mean_att - true_att
        expected = 0.1
        
        assert np.isclose(bias, expected)
    
    def test_relative_bias(self):
        """
        验证相对Bias: Relative_Bias = Bias / True_ATT
        
        测试: Bias = 0.1, True_ATT = 2.0 → Relative_Bias = 0.05 (5%)
        """
        bias = 0.1
        true_att = 2.0
        
        relative_bias = bias / true_att
        expected = 0.05
        
        assert np.isclose(relative_bias, expected)


class TestTreatmentEffectFormula:
    """
    验证处理效应公式
    """
    
    def test_common_timing_treatment_effect(self):
        """
        验证Common Timing处理效应公式:
        τ_r(X) = (r - S + 1) × [θ + λ_r × h(X)]
        
        参数: S=4, θ=3, λ_4=0.5
        测试点: r=4, h(X)=1
        τ_4 = (4-4+1) × [3 + 0.5×1] = 1 × 3.5 = 3.5
        """
        S = 4
        theta = 3
        lambda_4 = 0.5
        r = 4
        h_X = 1
        
        tau_r = (r - S + 1) * (theta + lambda_4 * h_X)
        expected = 3.5
        
        assert np.isclose(tau_r, expected)
    
    def test_small_sample_treatment_effect(self):
        """
        验证Small Sample处理效应:
        Y_it(1) = Y_it(0) + δ_t + ν_it
        
        测试: Y(0) = 5, δ_t = 2, ν = 0.5
        Y(1) = 5 + 2 + 0.5 = 7.5
        """
        y0 = 5.0
        delta_t = 2.0
        nu = 0.5
        
        y1 = y0 + delta_t + nu
        expected = 7.5
        
        assert np.isclose(y1, expected)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

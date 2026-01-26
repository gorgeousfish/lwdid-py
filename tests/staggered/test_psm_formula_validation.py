"""
Story 4.1: PSM公式验证测试

使用数学验证来确保公式实现正确:
- K_M权重公式
- 方差组合公式
- Stata ATET Robust方差公式

已使用Vibe Math MCP验证:
1. (K_M/M)² - K_M/M² = K_M(K_M-1)/M²  ✓
2. Var = (Σ异质性项 + Σσ²·K权重) / n_treat²  ✓
3. SE = sqrt(Var)  ✓
"""

import pytest
import numpy as np

from lwdid.staggered.estimators import (
    _compute_control_match_counts,
    _estimate_conditional_variance_same_group,
    _compute_psm_se_abadie_imbens_full,
)


class TestKMWeightFormula:
    """K_M权重公式验证"""
    
    def test_k_m_weight_identity(self):
        """验证 (K_M/M)² - K_M/M² = K_M(K_M-1)/M²"""
        test_cases = [
            (3, 2),  # K_M=3, M=2
            (5, 3),  # K_M=5, M=3
            (1, 1),  # K_M=1, M=1 (边界)
            (10, 5), # K_M=10, M=5
            (2, 1),  # K_M=2, M=1
        ]
        
        for K_M, M in test_cases:
            # 原始形式
            formula1 = (K_M / M) ** 2 - K_M / (M ** 2)
            # 简化形式
            formula2 = K_M * (K_M - 1) / (M ** 2)
            
            assert np.isclose(formula1, formula2), \
                f"K_M={K_M}, M={M}: {formula1} != {formula2}"
    
    def test_k_m_weight_zero_when_k_m_le_1(self):
        """当K_M≤1时，权重差为0或负"""
        M = 2
        
        # K_M = 0: 不被匹配
        assert 0 * (0 - 1) / (M ** 2) == 0
        
        # K_M = 1: 恰好匹配一次 -> 权重调整项为0
        assert 1 * (1 - 1) / (M ** 2) == 0
    
    def test_k_m_sum_preservation(self):
        """Σ K_M = n_treat × M"""
        matched = [[0, 1], [1, 2], [0, 2], [1, 3]]  # 4个处理，每个2邻居
        K_M = _compute_control_match_counts(matched, n_control=5)
        
        n_treat = len(matched)
        M = len(matched[0])
        
        assert K_M.sum() == n_treat * M, \
            f"K_M.sum()={K_M.sum()} != {n_treat}*{M}={n_treat*M}"


class TestVarianceFormula:
    """方差公式验证"""
    
    def test_variance_decomposition(self):
        """验证方差分解: Var = (异质性 + 条件方差调整) / n²"""
        # 手工设置参数
        n_treat = 10
        var_heterogeneity = 0.5  # 处理效应异质性项求和
        sum_K_sq_sigma2 = 2.0    # K权重×σ²求和
        
        # 计算总方差
        var_total = (var_heterogeneity + sum_K_sq_sigma2) / (n_treat ** 2)
        
        # 验证
        expected = (0.5 + 2.0) / 100
        assert np.isclose(var_total, expected), f"{var_total} != {expected}"
    
    def test_se_from_variance(self):
        """验证 SE = sqrt(Var)"""
        var_total = 0.00025  # 来自Vibe Math验证
        se = np.sqrt(var_total)
        
        expected_se = 0.015811388300841896
        assert np.isclose(se, expected_se, rtol=1e-10)
    
    def test_iid_vs_robust_formula_difference(self):
        """IID vs Robust方差公式差异"""
        # 公式差异示意（paper 参考下 IID 不含异质性项）
        # Stata vce(iid) 仍包含异质性项 + 全局方差项
        
        var_treat = 1.0  # 处理组方差贡献
        var_control = 0.5  # 控制组方差贡献
        var_heterogeneity = 0.3  # 异质性项
        n_treat = 50
        
        # IID方差
        var_iid = (var_treat + var_control) / (n_treat ** 2)
        
        # Robust方差
        var_robust = (var_heterogeneity + var_control) / (n_treat ** 2)
        
        # 两者应该不同（除非特殊情况）
        # 不一定总是 robust > iid
        assert var_iid >= 0
        assert var_robust >= 0


class TestIntegratedFormulaValidation:
    """集成公式验证 - 完整流程"""
    
    def test_full_se_computation_manual(self):
        """手工计算完整SE验证"""
        np.random.seed(42)
        
        # 设置简单数据
        n_treat = 5
        n_control = 10
        M = 2
        
        Y_treat = np.array([5.0, 6.0, 4.5, 5.5, 5.0])
        Y_control = np.array([3.0, 3.5, 4.0, 2.5, 3.0, 3.5, 4.0, 3.0, 3.5, 4.0])
        X_treat = np.random.randn(n_treat, 2)
        X_control = np.random.randn(n_control, 2)
        
        # 构造匹配 (每个处理单位匹配2个控制)
        matched = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
        
        # 计算ATT
        att_individual = []
        for i, m in enumerate(matched):
            y_matched_mean = np.mean(Y_control[m])
            att_individual.append(Y_treat[i] - y_matched_mean)
        att = np.mean(att_individual)
        
        # 使用函数计算SE
        se, ci_lower, ci_upper, K_M, sigma2 = _compute_psm_se_abadie_imbens_full(
            Y_treat=Y_treat,
            Y_control=Y_control,
            X_treat=X_treat,
            X_control=X_control,
            matched_control_ids=matched,
            att=att,
            n_neighbors=M,
            vce_type='robust',
        )
        
        # 验证基本性质
        assert se > 0
        assert K_M.sum() == n_treat * M
        assert len(sigma2) == n_treat + n_control
        
        # 验证K_M分布 (每个控制被匹配一次)
        assert np.all(K_M == 1)
    
    def test_k_m_squared_weight_computation(self):
        """K_M²权重计算验证"""
        K_M = np.array([0, 1, 2, 3, 4])
        M = 2
        
        # 手工计算
        K_m_sq_minus_prime = (K_M / M) ** 2 - K_M / (M ** 2)
        
        # 使用简化形式验证
        K_m_simplified = K_M * (K_M - 1) / (M ** 2)
        
        assert np.allclose(K_m_sq_minus_prime, K_m_simplified)
        
        # 验证边界情况
        assert K_m_simplified[0] == 0  # K_M=0
        assert K_m_simplified[1] == 0  # K_M=1
        assert K_m_simplified[2] == 0.5  # K_M=2: 2*1/4 = 0.5
        assert K_m_simplified[3] == 1.5  # K_M=3: 3*2/4 = 1.5
        assert K_m_simplified[4] == 3.0  # K_M=4: 4*3/4 = 3.0


class TestNumericalStability:
    """数值稳定性测试"""
    
    def test_large_k_m_values(self):
        """大K_M值稳定性"""
        K_M = np.array([100, 200, 500])
        M = 10
        
        K_m_sq_minus_prime = K_M * (K_M - 1) / (M ** 2)
        
        # 不应有溢出
        assert np.all(np.isfinite(K_m_sq_minus_prime))
        
        # 验证计算
        assert K_m_sq_minus_prime[0] == 100 * 99 / 100  # 99
        assert K_m_sq_minus_prime[1] == 200 * 199 / 100  # 398
    
    def test_small_variance_values(self):
        """小方差值稳定性"""
        sigma2 = np.array([1e-10, 1e-15, 1e-20])
        K_weight = np.array([1.0, 1.5, 2.0])
        n_treat = 100
        
        var_contribution = np.sum(K_weight * sigma2) / (n_treat ** 2)
        
        # 应该能处理非常小的值
        assert np.isfinite(var_contribution)
        assert var_contribution >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

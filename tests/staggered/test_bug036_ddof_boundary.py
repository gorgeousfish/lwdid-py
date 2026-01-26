"""
BUG-036: PSM 条件方差估计中 ddof=1 样本量边界问题测试

测试 _estimate_conditional_variance_same_group() 和 
_estimate_conditional_variance_same_group_paper() 函数的边界情况处理。

主要验证：
1. n_group = 1 时返回 0（不产生 NaN）
2. J = 1 且 n_group = 2 时正常计算
3. J = 2 且 n_group = 2 时正常计算
4. 极端小样本场景不产生 NaN

理论背景:
- np.var(array, ddof=1) 在 len(array) == 1 时返回 NaN（因为分母 n-1=0）
- 修复方案：添加显式的样本量检查，当样本不足时返回 0.0

References:
- Abadie A, Imbens GW (2006). "Large Sample Properties of Matching
  Estimators for Average Treatment Effects." Econometrica 74(1):235-267.
"""

import numpy as np
import pandas as pd
import pytest
import warnings

from lwdid.staggered.estimators import (
    _estimate_conditional_variance_same_group,
    _estimate_conditional_variance_same_group_paper,
    estimate_psm,
)


# ============================================================================
# Unit Tests for _estimate_conditional_variance_same_group
# ============================================================================

class TestEstimateConditionalVarianceSameGroup:
    """测试 Stata 风格的条件方差估计函数"""
    
    def test_single_sample_per_group_returns_zero(self):
        """当每组只有1个样本时，应返回0而不是NaN"""
        # 只有1个处理组样本和1个控制组样本
        Y = np.array([1.0, 2.0])
        X = np.array([0.5, 0.6])
        W = np.array([0, 1])
        
        sigma2 = _estimate_conditional_variance_same_group(Y, X, W, J=2)
        
        # 应返回0而不是NaN
        assert not np.any(np.isnan(sigma2)), f"sigma2 包含 NaN: {sigma2}"
        assert np.all(sigma2 == 0), f"单样本组应返回0: {sigma2}"
    
    def test_two_samples_per_group_no_nan(self):
        """当每组有2个样本时，应正常计算且无NaN"""
        # 2个处理组样本和2个控制组样本
        Y = np.array([1.0, 2.0, 3.0, 4.0])
        X = np.array([0.5, 0.6, 0.4, 0.7])
        W = np.array([0, 0, 1, 1])
        
        sigma2 = _estimate_conditional_variance_same_group(Y, X, W, J=2)
        
        assert not np.any(np.isnan(sigma2)), f"sigma2 包含 NaN: {sigma2}"
        assert sigma2.shape == (4,), f"形状错误: {sigma2.shape}"
    
    def test_j_equals_1_with_two_samples(self):
        """J=1 且 n_group=2 时应正常工作"""
        Y = np.array([1.0, 2.0, 3.0, 4.0])
        X = np.array([0.5, 0.6, 0.4, 0.7])
        W = np.array([0, 0, 1, 1])
        
        # J=1 会进入 J_actual < 2 的分支
        sigma2 = _estimate_conditional_variance_same_group(Y, X, W, J=1)
        
        assert not np.any(np.isnan(sigma2)), f"sigma2 包含 NaN: {sigma2}"
        # 应该使用组内方差作为 fallback
        assert np.all(sigma2 >= 0), f"方差应非负: {sigma2}"
    
    def test_empty_treatment_group(self):
        """当处理组为空时应正确处理"""
        Y = np.array([1.0, 2.0, 3.0])
        X = np.array([0.5, 0.6, 0.4])
        W = np.array([0, 0, 0])  # 没有处理组
        
        sigma2 = _estimate_conditional_variance_same_group(Y, X, W, J=2)
        
        assert not np.any(np.isnan(sigma2)), f"sigma2 包含 NaN: {sigma2}"
    
    def test_empty_control_group(self):
        """当控制组为空时应正确处理"""
        Y = np.array([1.0, 2.0, 3.0])
        X = np.array([0.5, 0.6, 0.4])
        W = np.array([1, 1, 1])  # 没有控制组
        
        sigma2 = _estimate_conditional_variance_same_group(Y, X, W, J=2)
        
        assert not np.any(np.isnan(sigma2)), f"sigma2 包含 NaN: {sigma2}"
    
    def test_large_j_with_small_group(self):
        """J 大于组内样本数时应正确处理"""
        Y = np.array([1.0, 2.0, 3.0, 4.0])
        X = np.array([0.5, 0.6, 0.4, 0.7])
        W = np.array([0, 0, 1, 1])
        
        # J=10 远大于每组2个样本
        sigma2 = _estimate_conditional_variance_same_group(Y, X, W, J=10)
        
        assert not np.any(np.isnan(sigma2)), f"sigma2 包含 NaN: {sigma2}"
        assert np.all(sigma2 >= 0), f"方差应非负: {sigma2}"
    
    def test_2d_covariates(self):
        """多维协变量应正常工作"""
        Y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        X = np.array([
            [0.5, 0.1],
            [0.6, 0.2],
            [0.4, 0.3],
            [0.7, 0.4],
            [0.55, 0.15],
            [0.65, 0.25],
        ])
        W = np.array([0, 0, 0, 1, 1, 1])
        
        sigma2 = _estimate_conditional_variance_same_group(Y, X, W, J=2)
        
        assert not np.any(np.isnan(sigma2)), f"sigma2 包含 NaN: {sigma2}"
        assert sigma2.shape == (6,), f"形状错误: {sigma2.shape}"
    
    def test_variance_values_reasonable(self):
        """验证方差值在合理范围内"""
        # 创建有明显方差的数据
        np.random.seed(42)
        n = 50
        Y = np.random.normal(0, 2, n)  # 标准差为2
        X = np.random.normal(0, 1, n)
        W = np.array([0] * 25 + [1] * 25)
        
        sigma2 = _estimate_conditional_variance_same_group(Y, X, W, J=2)
        
        assert not np.any(np.isnan(sigma2)), f"sigma2 包含 NaN"
        # 方差应该在合理范围内（约为 2^2 = 4 的量级）
        assert np.mean(sigma2) > 0, "平均方差应为正"
        assert np.mean(sigma2) < 100, f"平均方差异常大: {np.mean(sigma2)}"


# ============================================================================
# Unit Tests for _estimate_conditional_variance_same_group_paper
# ============================================================================

class TestEstimateConditionalVarianceSameGroupPaper:
    """测试论文风格的条件方差估计函数"""
    
    def test_single_sample_per_group_returns_zero(self):
        """当每组只有1个样本时，应返回0而不是NaN"""
        Y = np.array([1.0, 2.0])
        X = np.array([0.5, 0.6])
        W = np.array([0, 1])
        
        sigma2 = _estimate_conditional_variance_same_group_paper(Y, X, W, J=2)
        
        assert not np.any(np.isnan(sigma2)), f"sigma2 包含 NaN: {sigma2}"
        assert np.all(sigma2 == 0), f"单样本组应返回0: {sigma2}"
    
    def test_two_samples_per_group_no_nan(self):
        """当每组有2个样本时，应正常计算且无NaN"""
        Y = np.array([1.0, 2.0, 3.0, 4.0])
        X = np.array([0.5, 0.6, 0.4, 0.7])
        W = np.array([0, 0, 1, 1])
        
        sigma2 = _estimate_conditional_variance_same_group_paper(Y, X, W, J=2)
        
        assert not np.any(np.isnan(sigma2)), f"sigma2 包含 NaN: {sigma2}"
        assert sigma2.shape == (4,), f"形状错误: {sigma2.shape}"
    
    def test_j_equals_1_with_two_samples(self):
        """J=1 且 n_group=2 时应正常工作"""
        Y = np.array([1.0, 2.0, 3.0, 4.0])
        X = np.array([0.5, 0.6, 0.4, 0.7])
        W = np.array([0, 0, 1, 1])
        
        sigma2 = _estimate_conditional_variance_same_group_paper(Y, X, W, J=1)
        
        assert not np.any(np.isnan(sigma2)), f"sigma2 包含 NaN: {sigma2}"
        assert np.all(sigma2 >= 0), f"方差应非负: {sigma2}"
    
    def test_empty_treatment_group(self):
        """当处理组为空时应正确处理"""
        Y = np.array([1.0, 2.0, 3.0])
        X = np.array([0.5, 0.6, 0.4])
        W = np.array([0, 0, 0])
        
        sigma2 = _estimate_conditional_variance_same_group_paper(Y, X, W, J=2)
        
        assert not np.any(np.isnan(sigma2)), f"sigma2 包含 NaN: {sigma2}"
    
    def test_invalid_j_raises_error(self):
        """J < 1 应抛出 ValueError"""
        Y = np.array([1.0, 2.0])
        X = np.array([0.5, 0.6])
        W = np.array([0, 1])
        
        with pytest.raises(ValueError, match="J must be >= 1"):
            _estimate_conditional_variance_same_group_paper(Y, X, W, J=0)
    
    def test_paper_formula_output(self):
        """验证论文公式(14)的输出格式正确"""
        # σ̂²(Xᵢ, Wᵢ) = (J/(J+1)) * (Yᵢ - mean(Y_neighbors))²
        np.random.seed(42)
        n = 30
        Y = np.random.normal(5, 2, n)
        X = np.random.normal(0, 1, n)
        W = np.array([0] * 15 + [1] * 15)
        
        sigma2 = _estimate_conditional_variance_same_group_paper(Y, X, W, J=2)
        
        assert not np.any(np.isnan(sigma2)), f"sigma2 包含 NaN"
        assert np.all(sigma2 >= 0), "方差应非负"


# ============================================================================
# Integration Tests with estimate_psm
# ============================================================================

class TestPSMBoundaryConditions:
    """PSM 估计量边界条件测试"""
    
    def test_psm_small_sample_no_nan(self):
        """小样本 PSM 估计不应产生 NaN SE"""
        np.random.seed(42)
        n = 20
        
        x1 = np.random.normal(0, 1, n)
        ps_index = -0.5 + 0.5 * x1
        ps_true = 1 / (1 + np.exp(-ps_index))
        d = np.random.binomial(1, ps_true)
        
        # 确保有足够的处理和控制单位
        d[:5] = 1
        d[5:10] = 0
        
        y0 = 1.0 + 0.5 * x1 + np.random.normal(0, 0.5, n)
        y1 = y0 + 2.0
        y = np.where(d == 1, y1, y0)
        
        data = pd.DataFrame({'y': y, 'd': d, 'x1': x1})
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = estimate_psm(
                data=data,
                y='y',
                d='d',
                propensity_controls=['x1'],
                se_method='abadie_imbens_full',
                n_neighbors=1,
            )
        
        assert not np.isnan(result.se), f"SE 为 NaN: {result.se}"
        assert result.se >= 0, f"SE 应非负: {result.se}"
    
    def test_psm_extreme_imbalance_no_nan(self):
        """极端不平衡数据的 PSM 估计不应产生 NaN SE"""
        np.random.seed(42)
        n = 50
        
        x1 = np.random.normal(0, 1, n)
        
        # 极端不平衡：只有约10%是处理组
        d = np.zeros(n, dtype=int)
        d[:5] = 1  # 只有5个处理单位
        
        y0 = 1.0 + 0.5 * x1 + np.random.normal(0, 0.5, n)
        y1 = y0 + 2.0
        y = np.where(d == 1, y1, y0)
        
        data = pd.DataFrame({'y': y, 'd': d, 'x1': x1})
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = estimate_psm(
                data=data,
                y='y',
                d='d',
                propensity_controls=['x1'],
                se_method='abadie_imbens_full',
                n_neighbors=1,
            )
        
        assert not np.isnan(result.se), f"SE 为 NaN: {result.se}"
    
    def test_psm_vce_robust_no_nan(self):
        """vce(robust) 模式不应产生 NaN SE"""
        np.random.seed(42)
        n = 40
        
        x1 = np.random.normal(0, 1, n)
        d = (x1 > np.median(x1)).astype(int)
        y = 1.0 + 0.5 * x1 + 2.0 * d + np.random.normal(0, 0.5, n)
        
        data = pd.DataFrame({'y': y, 'd': d, 'x1': x1})
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = estimate_psm(
                data=data,
                y='y',
                d='d',
                propensity_controls=['x1'],
                se_method='abadie_imbens_full',
                vce_type='robust',
                n_neighbors=1,
            )
        
        assert not np.isnan(result.se), f"vce(robust) SE 为 NaN: {result.se}"
    
    def test_psm_vce_iid_no_nan(self):
        """vce(iid) 模式不应产生 NaN SE"""
        np.random.seed(42)
        n = 40
        
        x1 = np.random.normal(0, 1, n)
        d = (x1 > np.median(x1)).astype(int)
        y = 1.0 + 0.5 * x1 + 2.0 * d + np.random.normal(0, 0.5, n)
        
        data = pd.DataFrame({'y': y, 'd': d, 'x1': x1})
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = estimate_psm(
                data=data,
                y='y',
                d='d',
                propensity_controls=['x1'],
                se_method='abadie_imbens_full',
                vce_type='iid',
                n_neighbors=1,
            )
        
        assert not np.isnan(result.se), f"vce(iid) SE 为 NaN: {result.se}"
    
    def test_psm_paper_reference_no_nan(self):
        """se_reference='paper' 模式不应产生 NaN SE"""
        np.random.seed(42)
        n = 40
        
        x1 = np.random.normal(0, 1, n)
        d = (x1 > np.median(x1)).astype(int)
        y = 1.0 + 0.5 * x1 + 2.0 * d + np.random.normal(0, 0.5, n)
        
        data = pd.DataFrame({'y': y, 'd': d, 'x1': x1})
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = estimate_psm(
                data=data,
                y='y',
                d='d',
                propensity_controls=['x1'],
                se_method='abadie_imbens_full',
                se_reference='paper',
                n_neighbors=1,
            )
        
        assert not np.isnan(result.se), f"paper reference SE 为 NaN: {result.se}"


# ============================================================================
# Numerical Validation Tests
# ============================================================================

class TestNumericalValidation:
    """数值验证测试"""
    
    def test_variance_formula_ddof1(self):
        """验证 ddof=1 样本方差公式"""
        # 样本方差公式: Σ(xᵢ - x̄)² / (n-1)
        Y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        
        # 手动计算
        mean_y = np.mean(Y)  # 6.0
        sum_sq_dev = np.sum((Y - mean_y) ** 2)  # 40.0
        expected_var = sum_sq_dev / (len(Y) - 1)  # 40 / 4 = 10.0
        
        actual_var = np.var(Y, ddof=1)
        
        assert np.isclose(actual_var, expected_var), (
            f"方差计算不一致: {actual_var} vs {expected_var}"
        )
    
    def test_variance_single_element_is_nan(self):
        """确认单元素数组的 ddof=1 方差确实是 NaN"""
        Y = np.array([5.0])
        var = np.var(Y, ddof=1)
        
        # 这是 BUG-036 修复的核心问题
        assert np.isnan(var), "单元素 ddof=1 方差应为 NaN"
    
    def test_variance_two_elements_is_finite(self):
        """确认两元素数组的 ddof=1 方差是有限值"""
        Y = np.array([3.0, 5.0])
        var = np.var(Y, ddof=1)
        
        assert np.isfinite(var), f"两元素方差应为有限值: {var}"
        # (3-4)² + (5-4)² = 1 + 1 = 2, var = 2/1 = 2.0
        assert np.isclose(var, 2.0), f"方差值错误: {var}"
    
    def test_conditional_variance_consistency(self):
        """验证两种条件方差估计器的一致性"""
        np.random.seed(42)
        n = 100
        Y = np.random.normal(5, 2, n)
        X = np.random.normal(0, 1, n)
        W = np.array([0] * 50 + [1] * 50)
        
        sigma2_stata = _estimate_conditional_variance_same_group(Y, X, W, J=2)
        sigma2_paper = _estimate_conditional_variance_same_group_paper(Y, X, W, J=2)
        
        # 两者应该都是有限的
        assert np.all(np.isfinite(sigma2_stata)), "Stata 方法应返回有限值"
        assert np.all(np.isfinite(sigma2_paper)), "Paper 方法应返回有限值"
        
        # 两者应该有相似的量级（但不必完全相等）
        ratio = np.mean(sigma2_stata) / np.mean(sigma2_paper) if np.mean(sigma2_paper) > 0 else 1
        assert 0.1 < ratio < 10, f"两种方法的比值异常: {ratio}"


# ============================================================================
# Monte Carlo Tests
# ============================================================================

class TestMonteCarlo:
    """蒙特卡洛模拟测试"""
    
    def test_boundary_conditions_multiple_seeds(self):
        """多种子测试边界条件"""
        for seed in range(20):
            np.random.seed(seed)
            
            # 故意创建边界情况数据
            n = 10
            Y = np.random.normal(0, 1, n)
            X = np.random.normal(0, 1, n)
            W = np.random.binomial(1, 0.5, n)
            
            # 确保至少有一些处理和控制单位
            if W.sum() == 0:
                W[0] = 1
            if W.sum() == n:
                W[0] = 0
            
            sigma2 = _estimate_conditional_variance_same_group(Y, X, W, J=2)
            
            assert not np.any(np.isnan(sigma2)), (
                f"seed={seed}: sigma2 包含 NaN"
            )
    
    def test_se_coverage_with_boundary_data(self):
        """验证边界数据的 SE 覆盖率"""
        true_att = 2.0
        coverage_count = 0
        n_simulations = 50
        
        for seed in range(n_simulations):
            np.random.seed(seed)
            
            # 小样本数据
            n = 30
            x1 = np.random.normal(0, 1, n)
            d = (x1 > np.median(x1)).astype(int)
            y = 1.0 + 0.5 * x1 + true_att * d + np.random.normal(0, 1, n)
            
            data = pd.DataFrame({'y': y, 'd': d, 'x1': x1})
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    result = estimate_psm(
                        data=data,
                        y='y',
                        d='d',
                        propensity_controls=['x1'],
                        se_method='abadie_imbens_full',
                        n_neighbors=1,
                    )
                    
                    if not np.isnan(result.se):
                        # 检查 95% CI 是否覆盖真值
                        if result.ci_lower <= true_att <= result.ci_upper:
                            coverage_count += 1
                except Exception:
                    pass  # 某些极端情况可能失败
        
        # 覆盖率应该接近 95%，但由于小样本，允许较大偏差
        coverage_rate = coverage_count / n_simulations
        assert coverage_rate > 0.5, f"覆盖率过低: {coverage_rate:.2%}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])

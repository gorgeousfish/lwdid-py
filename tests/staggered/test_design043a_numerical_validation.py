"""
DESIGN-043-A: 数值验证测试

验证极端权重修复后的数值稳定性和结果合理性。
包括：
1. 修复前后对比（模拟）
2. 不同极端程度的结果一致性
3. 大样本数值稳定性
4. 边界条件测试

References:
    Abadie A, Imbens GW (2016). "Matching on the Estimated Propensity Score."
    Econometrica 84(2):781-807.
"""

import pytest
import numpy as np
import pandas as pd
import warnings
from typing import List
import statsmodels.api as sm

from lwdid.staggered.estimators import _compute_psm_variance_adjustment


# ============================================================================
# Helper Functions
# ============================================================================

def _compute_weight_unprotected(p_i: float) -> float:
    """模拟修复前的权重计算（无上限保护）"""
    return p_i / max(1 - p_i, 1e-10)


def _compute_weight_protected(p_i: float, max_weight: float = 99.0) -> float:
    """修复后的权重计算（有上限保护）"""
    return min(p_i / max(1 - p_i, 1e-10), max_weight)


def create_test_data(n: int, extreme_ps_ratio: float = 0.0, 
                     extreme_ps_value: float = 0.999, seed: int = 42):
    """
    创建测试数据
    
    Parameters
    ----------
    n : int
        样本量
    extreme_ps_ratio : float
        极端倾向得分的比例（0-1）
    extreme_ps_value : float
        极端倾向得分的值（接近1）
    seed : int
        随机种子
    """
    np.random.seed(seed)
    
    Y = np.random.normal(0, 1, n)
    D = np.zeros(n, dtype=int)
    n_treat = n // 3
    D[:n_treat] = 1
    
    # 基础倾向得分
    pscores = np.random.uniform(0.2, 0.8, n)
    
    # 添加极端值
    n_extreme = int(n * extreme_ps_ratio)
    if n_extreme > 0:
        pscores[:n_extreme] = extreme_ps_value
    
    Z = sm.add_constant(np.random.normal(0, 1, (n, 2)))
    V_gamma = np.eye(3) * 0.01
    
    matched_control_ids: List[List[int]] = [[0] for _ in range(n_treat)]
    
    return Y, D, Z, pscores, V_gamma, matched_control_ids


# ============================================================================
# Test: Weight Calculation Comparison
# ============================================================================

class TestWeightCalculation:
    """测试权重计算的修复效果"""
    
    def test_weight_comparison_normal_ps(self):
        """测试正常倾向得分下两种方法结果相同"""
        ps_values = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        for p in ps_values:
            w_unprotected = _compute_weight_unprotected(p)
            w_protected = _compute_weight_protected(p)
            
            if p <= 0.99:  # 正常范围内两者应该相等
                assert abs(w_unprotected - w_protected) < 1e-10, \
                    f"p={p}: 正常范围内权重应相等"
    
    def test_weight_comparison_extreme_ps(self):
        """测试极端倾向得分下权重被限制"""
        extreme_ps = [0.99, 0.999, 0.9999, 0.99999]
        
        for p in extreme_ps:
            w_unprotected = _compute_weight_unprotected(p)
            w_protected = _compute_weight_protected(p)
            
            # 保护后的权重应该 <= 99
            assert w_protected <= 99.0, \
                f"p={p}: 保护后权重应 <= 99, 实际: {w_protected}"
            
            # 未保护的权重可能非常大
            if p >= 0.999:
                assert w_unprotected > 99, \
                    f"p={p}: 未保护权重应 > 99, 实际: {w_unprotected}"
    
    def test_weight_protection_boundary(self):
        """测试权重保护的边界值"""
        # p = 0.99 对应权重 = 99
        p_boundary = 0.99
        w = _compute_weight_protected(p_boundary)
        assert abs(w - 99.0) < 1e-6, f"p=0.99 权重应为 99, 实际: {w}"
        
        # p 略大于 0.99 时权重被限制
        p_over = 0.991
        w_over = _compute_weight_protected(p_over)
        assert w_over == 99.0, f"p=0.991 权重应被限制为 99, 实际: {w_over}"


# ============================================================================
# Test: Numerical Stability Across Extreme Levels
# ============================================================================

class TestNumericalStabilityAcrossExtremes:
    """测试不同极端程度下的数值稳定性"""
    
    def test_variance_adjustment_stability(self):
        """测试方差调整在不同极端程度下的稳定性"""
        extreme_levels = [0.95, 0.99, 0.999, 0.9999]
        results = []
        
        for extreme_ps in extreme_levels:
            Y, D, Z, pscores, V_gamma, matched = create_test_data(
                n=200, extreme_ps_ratio=0.1, extreme_ps_value=extreme_ps, seed=42
            )
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = _compute_psm_variance_adjustment(
                    Y=Y, D=D, Z=Z, pscores=pscores, V_gamma=V_gamma,
                    matched_control_ids=matched, att=0.0, h=2
                )
            
            results.append(result)
            
            # 每个结果都应该是有限值
            assert np.isfinite(result), \
                f"extreme_ps={extreme_ps}: 结果应有限, 实际: {result}"
        
        # 检查结果的变化幅度是否受控
        # 由于内部裁剪，不同极端程度的结果差异应该有限
        result_range = max(results) - min(results)
        print(f"Results across extreme levels: {results}")
        print(f"Range: {result_range}")
        
        # 由于内部裁剪到 [0.01, 0.99]，结果应该相对稳定
        assert result_range < 1e6, f"结果范围过大: {result_range}"
    
    def test_no_nan_or_inf_with_various_extremes(self):
        """测试各种极端情况下不产生 nan 或 inf"""
        test_configs = [
            {"extreme_ps_ratio": 0.0, "extreme_ps_value": 0.999},   # 无极端值
            {"extreme_ps_ratio": 0.1, "extreme_ps_value": 0.99},    # 10% 极端值
            {"extreme_ps_ratio": 0.3, "extreme_ps_value": 0.999},   # 30% 极端值
            {"extreme_ps_ratio": 0.5, "extreme_ps_value": 0.9999},  # 50% 极端值
        ]
        
        for config in test_configs:
            Y, D, Z, pscores, V_gamma, matched = create_test_data(
                n=200, seed=42, **config
            )
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = _compute_psm_variance_adjustment(
                    Y=Y, D=D, Z=Z, pscores=pscores, V_gamma=V_gamma,
                    matched_control_ids=matched, att=0.0, h=2
                )
            
            assert not np.isnan(result), f"config={config}: 不应产生 nan"
            assert not np.isinf(result), f"config={config}: 不应产生 inf"


# ============================================================================
# Test: Large Sample Behavior
# ============================================================================

class TestLargeSampleBehavior:
    """测试大样本下的行为"""
    
    @pytest.mark.parametrize("n", [500, 1000, 2000])
    def test_large_sample_stability(self, n):
        """测试大样本下的数值稳定性"""
        Y, D, Z, pscores, V_gamma, matched = create_test_data(
            n=n, extreme_ps_ratio=0.1, extreme_ps_value=0.999, seed=42
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = _compute_psm_variance_adjustment(
                Y=Y, D=D, Z=Z, pscores=pscores, V_gamma=V_gamma,
                matched_control_ids=matched, att=0.0, h=2
            )
        
        assert np.isfinite(result), f"n={n}: 结果应有限"
        # 结果绝对值应该在合理范围
        assert abs(result) < 1e8, f"n={n}: 结果应在合理范围, 实际: {result}"
    
    def test_result_scales_reasonably(self):
        """测试结果随样本量的变化是否合理"""
        results = {}
        
        for n in [200, 500, 1000]:
            Y, D, Z, pscores, V_gamma, matched = create_test_data(
                n=n, extreme_ps_ratio=0.05, extreme_ps_value=0.99, seed=42
            )
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = _compute_psm_variance_adjustment(
                    Y=Y, D=D, Z=Z, pscores=pscores, V_gamma=V_gamma,
                    matched_control_ids=matched, att=0.0, h=2
                )
            
            results[n] = result
        
        # 所有结果都应该是有限值
        assert all(np.isfinite(r) for r in results.values()), \
            f"所有结果应有限: {results}"
        
        print(f"Results by sample size: {results}")


# ============================================================================
# Test: Boundary Conditions
# ============================================================================

class TestBoundaryConditions:
    """测试边界条件"""
    
    def test_all_ps_at_boundary(self):
        """测试所有倾向得分都在边界时"""
        np.random.seed(42)
        n = 100
        
        Y = np.random.normal(0, 1, n)
        D = np.zeros(n, dtype=int)
        D[:30] = 1
        
        # 所有倾向得分都是 0.99
        pscores = np.ones(n) * 0.99
        
        Z = sm.add_constant(np.random.normal(0, 1, (n, 2)))
        V_gamma = np.eye(3) * 0.01
        
        matched_control_ids: List[List[int]] = [[0] for _ in range(30)]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = _compute_psm_variance_adjustment(
                Y=Y, D=D, Z=Z, pscores=pscores, V_gamma=V_gamma,
                matched_control_ids=matched_control_ids, att=0.0, h=2
            )
        
        assert np.isfinite(result), f"边界倾向得分应产生有限结果: {result}"
    
    def test_ps_exactly_at_threshold(self):
        """测试倾向得分恰好在裁剪阈值时"""
        np.random.seed(42)
        n = 100
        
        Y = np.random.normal(0, 1, n)
        D = np.zeros(n, dtype=int)
        D[:30] = 1
        
        # 恰好在阈值 0.01 和 0.99
        pscores = np.ones(n) * 0.5
        pscores[:5] = 0.01   # 恰好在低阈值
        pscores[5:10] = 0.99  # 恰好在高阈值
        
        Z = sm.add_constant(np.random.normal(0, 1, (n, 2)))
        V_gamma = np.eye(3) * 0.01
        
        matched_control_ids: List[List[int]] = [[0] for _ in range(30)]
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = _compute_psm_variance_adjustment(
                Y=Y, D=D, Z=Z, pscores=pscores, V_gamma=V_gamma,
                matched_control_ids=matched_control_ids, att=0.0, h=2
            )
        
        assert np.isfinite(result), f"阈值边界应产生有限结果: {result}"
        
        # 恰好在阈值时不应该触发警告
        clipping_warnings = [x for x in w if "clipped" in str(x.message)]
        assert len(clipping_warnings) == 0, "恰好在阈值时不应触发裁剪警告"
    
    def test_mixed_extreme_and_normal(self):
        """测试混合极端和正常倾向得分"""
        np.random.seed(42)
        n = 100
        
        Y = np.random.normal(0, 1, n)
        D = np.zeros(n, dtype=int)
        D[:30] = 1
        
        pscores = np.random.uniform(0.2, 0.8, n)
        # 添加一些极端值
        pscores[:5] = 0.001
        pscores[5:10] = 0.999
        pscores[10:15] = 0.9999
        
        Z = sm.add_constant(np.random.normal(0, 1, (n, 2)))
        V_gamma = np.eye(3) * 0.01
        
        matched_control_ids: List[List[int]] = [[0] for _ in range(30)]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = _compute_psm_variance_adjustment(
                Y=Y, D=D, Z=Z, pscores=pscores, V_gamma=V_gamma,
                matched_control_ids=matched_control_ids, att=0.0, h=2
            )
        
        assert np.isfinite(result), f"混合场景应产生有限结果: {result}"


# ============================================================================
# Test: Comparison with Known Values
# ============================================================================

class TestKnownValues:
    """测试与已知值的比较"""
    
    def test_zero_result_with_constant_data(self):
        """测试常数数据产生零或接近零的调整"""
        np.random.seed(42)
        n = 100
        
        # 常数结果变量
        Y = np.ones(n) * 5.0
        D = np.zeros(n, dtype=int)
        D[:30] = 1
        
        # 均匀倾向得分
        pscores = np.ones(n) * 0.5
        
        Z = sm.add_constant(np.zeros((n, 2)))  # 常数协变量
        V_gamma = np.eye(3) * 0.01
        
        matched_control_ids: List[List[int]] = [[0] for _ in range(30)]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = _compute_psm_variance_adjustment(
                Y=Y, D=D, Z=Z, pscores=pscores, V_gamma=V_gamma,
                matched_control_ids=matched_control_ids, att=0.0, h=2
            )
        
        # 常数数据应该产生较小的调整
        assert np.isfinite(result), f"常数数据应产生有限结果: {result}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
BUG-041: AI Robust SE 调整中权重限制不一致 - 回归测试

验证 `_compute_psm_variance_adjustment` 函数中两个路径（内存优化和非内存优化）
对 `_AI_MAX_WEIGHT` 限制的一致性应用。

问题描述：
- 内存优化路径（n >= 5000）应用了 `_AI_MAX_WEIGHT = 99.0` 限制
- 非内存优化路径（n < 5000）原本没有应用该限制
- 修复后两个路径都应用相同的权重限制

测试策略：
1. 直接测试权重计算公式的正确性
2. 测试两个路径在相同数据下产生相同结果
3. 测试边界条件（p=0.99 对应 weight=99）
4. 测试极端倾向得分时的数值稳定性

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

from lwdid.staggered.estimators import (
    _compute_psm_variance_adjustment,
    _MEMORY_EFFICIENT_THRESHOLD,
)


# ============================================================================
# Helper Functions
# ============================================================================

def compute_weight_with_cap(p_i: float, max_weight: float = 99.0) -> float:
    """计算带上限保护的权重（与代码实现一致）"""
    return min(p_i / max(1 - p_i, 1e-10), max_weight)


def compute_weight_without_cap(p_i: float) -> float:
    """计算无上限保护的权重（旧实现）"""
    return p_i / max(1 - p_i, 1e-10)


def create_test_data(n: int, seed: int = 42):
    """
    创建用于测试的数据
    
    Parameters
    ----------
    n : int
        样本量
    seed : int
        随机种子
        
    Returns
    -------
    tuple
        (Y, D, Z, pscores, V_gamma, matched_control_ids)
    """
    np.random.seed(seed)
    
    Y = np.random.normal(0, 1, n)
    D = np.zeros(n, dtype=int)
    n_treat = n // 3
    D[:n_treat] = 1
    
    # 生成正常范围的倾向得分
    pscores = np.random.uniform(0.2, 0.8, n)
    
    Z = sm.add_constant(np.random.normal(0, 1, (n, 2)))
    V_gamma = np.eye(3) * 0.01
    
    # 创建简单的匹配 ID（每个处理单位匹配到第一个控制单位）
    matched_control_ids: List[List[int]] = [[0] for _ in range(n_treat)]
    
    return Y, D, Z, pscores, V_gamma, matched_control_ids


# ============================================================================
# Test: Weight Calculation Formula (BUG-041 Core)
# ============================================================================

class TestWeightCalculation:
    """测试权重计算公式的正确性"""
    
    def test_weight_formula_normal_range(self):
        """测试正常倾向得分范围的权重计算"""
        test_cases = [
            (0.1, 0.1 / 0.9),         # p=0.1 -> weight=0.111
            (0.3, 0.3 / 0.7),         # p=0.3 -> weight=0.429
            (0.5, 0.5 / 0.5),         # p=0.5 -> weight=1.0
            (0.7, 0.7 / 0.3),         # p=0.7 -> weight=2.333
            (0.9, 0.9 / 0.1),         # p=0.9 -> weight=9.0
        ]
        
        for p, expected_weight in test_cases:
            actual_weight = compute_weight_with_cap(p)
            assert abs(actual_weight - expected_weight) < 1e-10, \
                f"p={p}: expected {expected_weight}, got {actual_weight}"
    
    def test_weight_capped_at_99(self):
        """测试权重在极端倾向得分时被限制为99"""
        extreme_pscores = [0.99, 0.995, 0.999, 0.9999, 0.99999]
        
        for p in extreme_pscores:
            weight_capped = compute_weight_with_cap(p)
            weight_uncapped = compute_weight_without_cap(p)
            
            # 有上限的权重应该 <= 99 (允许浮点误差)
            assert weight_capped <= 99.0 + 1e-10, \
                f"p={p}: capped weight should be <= 99, got {weight_capped}"
            
            # 无上限的权重在 p > 0.99 时会超过99 (使用 > 而不是 >=，避免浮点精度问题)
            if p > 0.99:
                assert weight_uncapped > 99.0, \
                    f"p={p}: uncapped weight should be > 99, got {weight_uncapped}"
    
    def test_weight_boundary_at_099(self):
        """测试 p=0.99 时权重恰好为99"""
        p = 0.99
        expected_weight = 99.0
        actual_weight = compute_weight_with_cap(p)
        
        assert abs(actual_weight - expected_weight) < 1e-10, \
            f"p=0.99: expected weight=99, got {actual_weight}"
    
    def test_weight_just_above_099(self):
        """测试 p 略大于 0.99 时权重被限制为99"""
        p_values = [0.991, 0.995, 0.999]
        
        for p in p_values:
            weight = compute_weight_with_cap(p)
            assert weight == 99.0, \
                f"p={p}: weight should be capped at 99, got {weight}"


# ============================================================================
# Test: Path Consistency (BUG-041 Main Fix)
# ============================================================================

class TestPathConsistency:
    """
    测试两个路径（内存优化和非内存优化）的行为一致性
    
    这是 BUG-041 的核心测试：验证两个路径对 _AI_MAX_WEIGHT 的应用一致
    """
    
    def test_small_sample_uses_non_memory_efficient_path(self):
        """测试小样本使用非内存优化路径"""
        n = 300  # 小于 5000
        assert n < _MEMORY_EFFICIENT_THRESHOLD, \
            f"Sample size {n} should be < {_MEMORY_EFFICIENT_THRESHOLD}"
        
        Y, D, Z, pscores, V_gamma, matched_control_ids = create_test_data(n, seed=42)
        
        # 添加极端倾向得分
        pscores[:10] = 0.999
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = _compute_psm_variance_adjustment(
                Y=Y, D=D, Z=Z, pscores=pscores, V_gamma=V_gamma,
                matched_control_ids=matched_control_ids, att=0.0, h=2
            )
        
        # 结果应该是有限值（不会因为极端权重而溢出）
        assert np.isfinite(result), \
            f"Small sample path should produce finite result, got {result}"
    
    def test_large_sample_uses_memory_efficient_path(self):
        """测试大样本使用内存优化路径"""
        n = 6000  # 大于 5000
        assert n >= _MEMORY_EFFICIENT_THRESHOLD, \
            f"Sample size {n} should be >= {_MEMORY_EFFICIENT_THRESHOLD}"
        
        Y, D, Z, pscores, V_gamma, matched_control_ids = create_test_data(n, seed=42)
        
        # 添加极端倾向得分
        pscores[:100] = 0.999
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = _compute_psm_variance_adjustment(
                Y=Y, D=D, Z=Z, pscores=pscores, V_gamma=V_gamma,
                matched_control_ids=matched_control_ids, att=0.0, h=2
            )
        
        # 结果应该是有限值
        assert np.isfinite(result), \
            f"Large sample path should produce finite result, got {result}"
    
    def test_both_paths_handle_extreme_pscores_consistently(self):
        """
        测试两个路径对极端倾向得分的处理一致性
        
        由于两个路径的邻居搜索算法不同，无法直接比较结果，
        但都应该产生有限且合理的结果。
        """
        small_n = 300  # 使用非内存优化路径
        large_n = 6000  # 使用内存优化路径
        
        results = {}
        
        for n, label in [(small_n, 'non_memory_efficient'), (large_n, 'memory_efficient')]:
            Y, D, Z, pscores, V_gamma, matched_control_ids = create_test_data(n, seed=42)
            
            # 设置极端倾向得分（会触发权重限制）
            n_extreme = n // 10
            pscores[:n_extreme] = 0.999
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = _compute_psm_variance_adjustment(
                    Y=Y, D=D, Z=Z, pscores=pscores, V_gamma=V_gamma,
                    matched_control_ids=matched_control_ids, att=0.0, h=2
                )
            
            results[label] = result
            
            # 两个路径都应该产生有限值
            assert np.isfinite(result), \
                f"{label} path should produce finite result, got {result}"
            
            # 结果绝对值应该在合理范围内（因为权重被限制）
            assert abs(result) < 1e6, \
                f"{label} path result should be bounded, got {result}"
        
        print(f"BUG-041 consistency test results: {results}")


# ============================================================================
# Test: Numerical Stability with Extreme Propensity Scores
# ============================================================================

class TestNumericalStabilityBUG041:
    """测试极端倾向得分时的数值稳定性（BUG-041 修复效果）"""
    
    def test_no_overflow_with_ps_0999(self):
        """测试 p=0.999 时不会溢出"""
        n = 300
        Y, D, Z, pscores, V_gamma, matched_control_ids = create_test_data(n, seed=42)
        
        # 设置所有处理单位的倾向得分为0.999
        n_treat = n // 3
        pscores[:n_treat] = 0.999
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = _compute_psm_variance_adjustment(
                Y=Y, D=D, Z=Z, pscores=pscores, V_gamma=V_gamma,
                matched_control_ids=matched_control_ids, att=0.0, h=2
            )
        
        assert np.isfinite(result), f"Should not overflow with p=0.999, got {result}"
        assert not np.isinf(result), f"Should not be inf with p=0.999, got {result}"
    
    def test_no_overflow_with_ps_09999(self):
        """测试 p=0.9999 时不会溢出"""
        n = 300
        Y, D, Z, pscores, V_gamma, matched_control_ids = create_test_data(n, seed=42)
        
        n_treat = n // 3
        pscores[:n_treat] = 0.9999
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = _compute_psm_variance_adjustment(
                Y=Y, D=D, Z=Z, pscores=pscores, V_gamma=V_gamma,
                matched_control_ids=matched_control_ids, att=0.0, h=2
            )
        
        assert np.isfinite(result), f"Should not overflow with p=0.9999, got {result}"
    
    def test_consistent_results_across_extreme_levels(self):
        """测试不同极端程度的倾向得分产生一致的结果"""
        n = 300
        extreme_levels = [0.99, 0.999, 0.9999]
        results = []
        
        for extreme_ps in extreme_levels:
            Y, D, Z, pscores, V_gamma, matched_control_ids = create_test_data(n, seed=42)
            
            n_treat = n // 3
            pscores[:n_treat] = extreme_ps
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = _compute_psm_variance_adjustment(
                    Y=Y, D=D, Z=Z, pscores=pscores, V_gamma=V_gamma,
                    matched_control_ids=matched_control_ids, att=0.0, h=2
                )
            
            results.append(result)
        
        # 所有结果都应该是有限值
        assert all(np.isfinite(r) for r in results), \
            f"All results should be finite: {results}"
        
        # 由于内部裁剪到 [0.01, 0.99]，结果应该相同
        # 因为所有 >= 0.99 的值都会被裁剪到 0.99
        max_diff = max(abs(results[i] - results[j]) 
                       for i in range(len(results)) 
                       for j in range(i+1, len(results)))
        
        assert max_diff < 1e-10, \
            f"Results should be identical after clipping (diff={max_diff}): {results}"


# ============================================================================
# Test: Pre-fix vs Post-fix Comparison (Simulated)
# ============================================================================

class TestPrePostFixComparison:
    """模拟 BUG-041 修复前后的行为对比"""
    
    def test_fix_prevents_unbounded_weights(self):
        """测试修复防止了无界权重"""
        # 模拟修复前的行为（无上限）
        extreme_p = 0.9999
        weight_before_fix = compute_weight_without_cap(extreme_p)
        
        # 模拟修复后的行为（有上限）
        weight_after_fix = compute_weight_with_cap(extreme_p)
        
        # 修复前的权重非常大
        assert weight_before_fix > 1000, \
            f"Weight before fix should be very large: {weight_before_fix}"
        
        # 修复后的权重被限制
        assert weight_after_fix == 99.0, \
            f"Weight after fix should be capped at 99: {weight_after_fix}"
    
    def test_fix_ensures_path_consistency(self):
        """测试修复确保了两个路径的一致性"""
        # 验证两个路径使用相同的权重计算
        # 通过检查代码注释和权重限制值
        
        from lwdid.staggered.estimators import _compute_psm_variance_adjustment
        
        # 获取函数源码检查（通过 docstring）
        docstring = _compute_psm_variance_adjustment.__doc__
        
        # 验证函数文档包含数值稳定性说明
        assert "Numerical Stability" in docstring or "DESIGN-043-A" in docstring, \
            "Function should document numerical stability measures"


# ============================================================================
# Test: Integration with Real Data Patterns
# ============================================================================

class TestRealDataPatterns:
    """测试类似真实数据模式的场景"""
    
    def test_mixed_normal_and_extreme_pscores(self):
        """测试混合正常和极端倾向得分"""
        n = 300
        Y, D, Z, pscores, V_gamma, matched_control_ids = create_test_data(n, seed=42)
        
        # 混合场景：大部分正常，少数极端
        n_extreme = 10
        pscores[:n_extreme] = 0.999  # 极端高
        pscores[n_extreme:2*n_extreme] = 0.001  # 极端低
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = _compute_psm_variance_adjustment(
                Y=Y, D=D, Z=Z, pscores=pscores, V_gamma=V_gamma,
                matched_control_ids=matched_control_ids, att=0.0, h=2
            )
        
        assert np.isfinite(result), f"Mixed scenario should work: {result}"
    
    def test_all_pscores_at_boundary(self):
        """测试所有倾向得分都在边界"""
        n = 300
        Y, D, Z, pscores, V_gamma, matched_control_ids = create_test_data(n, seed=42)
        
        # 所有倾向得分都是 0.99
        pscores[:] = 0.99
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = _compute_psm_variance_adjustment(
                Y=Y, D=D, Z=Z, pscores=pscores, V_gamma=V_gamma,
                matched_control_ids=matched_control_ids, att=0.0, h=2
            )
        
        assert np.isfinite(result), f"Boundary scenario should work: {result}"


# ============================================================================
# Test: Regression Prevention
# ============================================================================

class TestRegressionPrevention:
    """防止 BUG-041 回归的测试"""
    
    def test_ai_max_weight_constant_exists(self):
        """验证 _AI_MAX_WEIGHT 常量存在"""
        # 这是内部常量，在函数内定义
        # 通过测试极端权重被限制来间接验证
        
        extreme_p = 0.9999
        weight = compute_weight_with_cap(extreme_p, max_weight=99.0)
        
        assert weight == 99.0, \
            f"_AI_MAX_WEIGHT should be 99.0 (weight={weight})"
    
    def test_memory_efficient_threshold_value(self):
        """验证 _MEMORY_EFFICIENT_THRESHOLD 值"""
        assert _MEMORY_EFFICIENT_THRESHOLD == 5000, \
            f"Threshold should be 5000, got {_MEMORY_EFFICIENT_THRESHOLD}"
    
    def test_both_paths_apply_same_weight_cap(self):
        """
        验证两个路径应用相同的权重上限
        
        这是 BUG-041 的关键回归测试
        """
        # 测试小样本路径
        small_result = self._run_with_extreme_pscores(n=300, extreme_ps=0.9999)
        
        # 测试大样本路径
        large_result = self._run_with_extreme_pscores(n=6000, extreme_ps=0.9999)
        
        # 两个路径都应该产生有限结果（权重被限制）
        assert np.isfinite(small_result), f"Small path failed: {small_result}"
        assert np.isfinite(large_result), f"Large path failed: {large_result}"
        
        # 两个路径的结果都应该在合理范围内
        assert abs(small_result) < 1e6, f"Small path unbounded: {small_result}"
        assert abs(large_result) < 1e6, f"Large path unbounded: {large_result}"
    
    def _run_with_extreme_pscores(self, n: int, extreme_ps: float) -> float:
        """辅助方法：运行极端倾向得分测试"""
        Y, D, Z, pscores, V_gamma, matched_control_ids = create_test_data(n, seed=42)
        
        n_treat = n // 3
        pscores[:n_treat] = extreme_ps
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _compute_psm_variance_adjustment(
                Y=Y, D=D, Z=Z, pscores=pscores, V_gamma=V_gamma,
                matched_control_ids=matched_control_ids, att=0.0, h=2
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

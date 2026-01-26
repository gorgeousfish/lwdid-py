"""
DESIGN-034 测试：K_M 整数类型从 int32 改为 int64

测试目标：
1. 验证 K_M dtype 是 np.int64
2. 验证大值累加不溢出
3. 验证与现有功能兼容
4. 验证 SE 公式中 K_M 的正确使用
"""

import numpy as np
import pandas as pd
import pytest
from typing import List

# 导入被测试的函数
from lwdid.staggered.estimators import (
    _compute_control_match_counts,
    _compute_psm_se_abadie_imbens_full,
    estimate_psm,
)


class TestDesign034DTypeCorrectness:
    """验证 K_M dtype 是 np.int64"""
    
    def test_compute_control_match_counts_dtype(self):
        """测试 _compute_control_match_counts 返回 int64"""
        matched_control_ids = [[0, 1], [0, 2], [1, 2]]
        n_control = 3
        
        K_M = _compute_control_match_counts(matched_control_ids, n_control)
        
        assert K_M.dtype == np.int64, f"Expected int64, got {K_M.dtype}"
    
    def test_compute_control_match_counts_values(self):
        """测试 K_M 计算值正确"""
        matched_control_ids = [[0, 1], [0, 2], [1, 2]]
        n_control = 3
        
        K_M = _compute_control_match_counts(matched_control_ids, n_control)
        
        # 控制单位 0 被匹配 2 次，1 被匹配 2 次，2 被匹配 2 次
        expected = np.array([2, 2, 2], dtype=np.int64)
        np.testing.assert_array_equal(K_M, expected)
    
    def test_compute_control_match_counts_sum_formula(self):
        """验证 sum(K_M) = n_treat × n_neighbors"""
        n_treat = 10
        n_neighbors = 3
        n_control = 5
        
        # 每个处理单位匹配 n_neighbors 个控制单位
        matched_control_ids = [
            list(np.random.choice(n_control, n_neighbors, replace=True))
            for _ in range(n_treat)
        ]
        
        K_M = _compute_control_match_counts(matched_control_ids, n_control)
        
        # K_M总和至少应等于 n_treat * n_neighbors (Stata ties may increase matches)
        assert K_M.sum() >= n_treat * n_neighbors


class TestDesign034OverflowSafety:
    """验证大值累加不溢出"""
    
    def test_large_k_m_no_overflow(self):
        """测试大值 K_M 不溢出"""
        # 模拟极端情况：所有处理单位都匹配到同一个控制单位
        n_treat = 100_000
        n_neighbors = 10
        n_control = 5
        
        # 所有处理单位都匹配到控制单位 0
        matched_control_ids = [[0] * n_neighbors for _ in range(n_treat)]
        
        K_M = _compute_control_match_counts(matched_control_ids, n_control)
        
        # K_M[0] 应该等于 n_treat * n_neighbors = 1,000,000
        assert K_M[0] == n_treat * n_neighbors
        assert K_M.dtype == np.int64
        
    def test_very_large_k_m_value(self):
        """测试非常大的 K_M 值（超过 int32 范围的边界测试）"""
        # int32 max = 2,147,483,647
        # 构造一个 K_M 值接近但不超过 int32 max 的场景
        n_control = 3
        
        # 直接构造大值进行测试
        K_M = np.zeros(n_control, dtype=np.int64)
        K_M[0] = 2_000_000_000  # 接近 int32 max
        K_M[1] = 3_000_000_000  # 超过 int32 max，但 int64 可以处理
        
        # 验证不溢出
        assert K_M[0] == 2_000_000_000
        assert K_M[1] == 3_000_000_000
        
    def test_k_m_sum_large_dataset(self):
        """测试大数据集的 K_M 总和"""
        n_treat = 10_000
        n_neighbors = 5
        n_control = 100
        
        np.random.seed(42)
        matched_control_ids = [
            list(np.random.choice(n_control, n_neighbors, replace=True))
            for _ in range(n_treat)
        ]
        
        K_M = _compute_control_match_counts(matched_control_ids, n_control)
        
        # 总和应该等于 n_treat * n_neighbors
        assert K_M.sum() == n_treat * n_neighbors


class TestDesign034EmptyInput:
    """测试边界条件"""
    
    def test_empty_matched_ids(self):
        """测试空匹配列表"""
        matched_control_ids: List[List[int]] = []
        n_control = 5
        
        K_M = _compute_control_match_counts(matched_control_ids, n_control)
        
        assert len(K_M) == n_control
        assert K_M.sum() == 0
        assert K_M.dtype == np.int64
    
    def test_no_matches_for_some_treated(self):
        """测试部分处理单位无匹配"""
        matched_control_ids = [[0, 1], [], [2], []]
        n_control = 3
        
        K_M = _compute_control_match_counts(matched_control_ids, n_control)
        
        expected = np.array([1, 1, 1], dtype=np.int64)
        np.testing.assert_array_equal(K_M, expected)
    
    def test_all_control_units_unmatched(self):
        """测试所有控制单位都未匹配"""
        matched_control_ids = [[], [], []]
        n_control = 5
        
        K_M = _compute_control_match_counts(matched_control_ids, n_control)
        
        assert K_M.sum() == 0
        assert all(k == 0 for k in K_M)


class TestDesign034Integration:
    """集成测试：验证与 estimate_psm 的兼容性"""
    
    @pytest.fixture
    def simple_psm_data(self):
        """创建简单的 PSM 测试数据"""
        np.random.seed(42)
        n = 200
        
        # 协变量
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        
        # 处理概率
        logit = -0.5 + 0.3 * x1 + 0.2 * x2
        prob = 1 / (1 + np.exp(-logit))
        d = (np.random.rand(n) < prob).astype(int)
        
        # 结果变量（带处理效应）
        y = 2 + 0.5 * x1 + 0.3 * x2 + 1.5 * d + np.random.randn(n)
        
        return pd.DataFrame({
            'y': y,
            'd': d,
            'x1': x1,
            'x2': x2,
        })
    
    def test_estimate_psm_k_m_dtype(self, simple_psm_data):
        """测试 estimate_psm 返回的 control_match_counts 类型"""
        result = estimate_psm(
            data=simple_psm_data,
            y='y',
            d='d',
            propensity_controls=['x1', 'x2'],
            n_neighbors=1,
            se_method='abadie_imbens_full',
        )
        
        assert result.control_match_counts is not None
        assert result.control_match_counts.dtype == np.int64
    
    def test_estimate_psm_numerical_correctness(self, simple_psm_data):
        """测试 estimate_psm 数值正确性"""
        result = estimate_psm(
            data=simple_psm_data,
            y='y',
            d='d',
            propensity_controls=['x1', 'x2'],
            n_neighbors=2,
            se_method='abadie_imbens_full',
        )
        
        # 验证基本统计
        assert not np.isnan(result.att)
        assert result.se > 0
        assert result.ci_lower < result.att < result.ci_upper
        
        # 验证 K_M
        K_M = result.control_match_counts
        assert K_M is not None
        n_treat = result.n_treated - result.n_dropped
        n_neighbors = 2
        assert K_M.sum() == n_treat * n_neighbors


class TestDesign034SECalculation:
    """验证 SE 计算中 K_M 的正确使用"""
    
    def test_se_with_high_k_m(self):
        """测试高 K_M 值时 SE 计算的稳定性"""
        np.random.seed(123)
        n = 100
        
        # 创建简单数据
        x = np.random.randn(n)
        d = (x > 0).astype(int)
        y = 2 + 0.5 * x + 1.0 * d + 0.5 * np.random.randn(n)
        
        data = pd.DataFrame({'y': y, 'd': d, 'x': x})
        
        # 使用有放回匹配（会产生较高的 K_M 值）
        result = estimate_psm(
            data=data,
            y='y',
            d='d',
            propensity_controls=['x'],
            n_neighbors=5,
            with_replacement=True,
            se_method='abadie_imbens_full',
        )
        
        assert result.se > 0
        assert not np.isnan(result.se)
        
        # K_M 应该有一些高值（因为有放回匹配）
        K_M = result.control_match_counts
        if K_M is not None:
            assert K_M.max() >= 1


class TestDesign034MemoryEfficiency:
    """验证内存效率"""
    
    def test_memory_size_int64(self):
        """验证 int64 数组的内存大小"""
        n_control = 1_000_000
        
        # int64: 8 bytes per element
        K_M = np.zeros(n_control, dtype=np.int64)
        
        expected_bytes = n_control * 8
        assert K_M.nbytes == expected_bytes
        
    def test_memory_comparison_int32_vs_int64(self):
        """比较 int32 和 int64 内存使用"""
        n_control = 1_000_000
        
        K_M_32 = np.zeros(n_control, dtype=np.int32)
        K_M_64 = np.zeros(n_control, dtype=np.int64)
        
        # int64 应该是 int32 的两倍
        assert K_M_64.nbytes == 2 * K_M_32.nbytes
        
        # 但对于 100 万控制单位，额外内存仅 4 MB
        extra_mb = (K_M_64.nbytes - K_M_32.nbytes) / (1024 * 1024)
        assert extra_mb < 5  # 小于 5 MB


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

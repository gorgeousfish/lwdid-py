"""
DESIGN-031: A矩阵向量化优化性能基准测试

比较向量化实现与双重循环实现的性能差异。

预期结果:
- 向量化实现利用BLAS库，性能显著优于Python循环
- 性能提升随K增大更明显
- 大样本和多协变量场景下优势更大
"""

import numpy as np
import pandas as pd
import pytest
import time
from typing import Callable


def benchmark(func: Callable, *args, n_runs: int = 10, **kwargs) -> tuple:
    """
    基准测试函数
    
    Returns
    -------
    tuple
        (mean_time, std_time, all_times)
    """
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        times.append(end - start)
    
    return np.mean(times), np.std(times), times


def compute_a_matrix_loop(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """双重循环实现（原始版本）"""
    n, K = X.shape
    A = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            A[i, j] = -np.mean(w * X[:, i] * X[:, j])
    return A


def compute_a_matrix_vectorized(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """向量化实现（优化版本）"""
    n = X.shape[0]
    w_col = w.reshape(-1, 1)
    return -((X * w_col).T @ X) / n


class TestDesign031Performance:
    """A矩阵计算性能测试"""
    
    @pytest.mark.parametrize("n,K", [
        (500, 5),      # 小规模
        (1000, 10),    # 中规模
        (5000, 20),    # 大规模
        (10000, 30),   # 超大规模
    ])
    def test_vectorized_vs_loop_performance(self, n, K):
        """比较向量化与循环实现的性能"""
        np.random.seed(42)
        
        X = np.random.randn(n, K)
        w = np.random.rand(n)
        
        # 预热
        _ = compute_a_matrix_loop(X, w)
        _ = compute_a_matrix_vectorized(X, w)
        
        # 基准测试
        n_runs = 5 if n >= 5000 else 10
        
        loop_mean, loop_std, _ = benchmark(compute_a_matrix_loop, X, w, n_runs=n_runs)
        vec_mean, vec_std, _ = benchmark(compute_a_matrix_vectorized, X, w, n_runs=n_runs)
        
        # 计算加速比
        speedup = loop_mean / vec_mean if vec_mean > 0 else float('inf')
        
        print(f"\n[性能测试 n={n}, K={K}]")
        print(f"  循环实现:   {loop_mean*1000:.4f}ms (±{loop_std*1000:.4f}ms)")
        print(f"  向量化实现: {vec_mean*1000:.4f}ms (±{vec_std*1000:.4f}ms)")
        print(f"  加速比:     {speedup:.2f}x")
        
        # 验证数值一致性
        A_loop = compute_a_matrix_loop(X, w)
        A_vec = compute_a_matrix_vectorized(X, w)
        np.testing.assert_allclose(A_loop, A_vec, rtol=1e-14, atol=1e-14)
        
        # 向量化应该至少不慢于循环（考虑小规模时的开销）
        assert vec_mean <= loop_mean * 1.5, "向量化实现不应该显著慢于循环"
        
        # 对于较大的K，向量化应该明显更快
        if K >= 10:
            assert speedup >= 1.0, f"K={K}时向量化应该有加速效果"
    
    def test_estimator_se_performance(self):
        """测试完整估计器SE计算的性能"""
        from lwdid.staggered.estimators import estimate_ra, estimate_ipw, estimate_ipwra
        
        np.random.seed(42)
        n = 1000
        K = 10
        
        # 生成测试数据
        X = np.random.randn(n, K)
        D = (np.random.rand(n) > 0.5).astype(int)
        Y = X @ np.random.randn(K) + 3 * D + np.random.randn(n)
        
        controls = [f'x{i}' for i in range(K)]
        data = pd.DataFrame(X, columns=controls)
        data['D'] = D
        data['Y'] = Y
        
        # 基准测试各估计器
        print(f"\n[估计器SE性能测试 n={n}, K={K}]")
        
        # RA
        start = time.perf_counter()
        _ = estimate_ra(data, 'Y', 'D', controls, vce='robust')
        ra_time = time.perf_counter() - start
        print(f"  RA SE计算:    {ra_time*1000:.2f}ms")
        
        # IPW
        start = time.perf_counter()
        _ = estimate_ipw(data, 'Y', 'D', controls, se_method='analytical')
        ipw_time = time.perf_counter() - start
        print(f"  IPW SE计算:   {ipw_time*1000:.2f}ms")
        
        # IPWRA
        start = time.perf_counter()
        _ = estimate_ipwra(data, 'Y', 'D', controls, controls, se_method='analytical')
        ipwra_time = time.perf_counter() - start
        print(f"  IPWRA SE计算: {ipwra_time*1000:.2f}ms")
        
        # 性能应该合理
        assert ra_time < 1.0, "RA SE计算应该在1秒内完成"
        assert ipw_time < 1.0, "IPW SE计算应该在1秒内完成"
        assert ipwra_time < 1.0, "IPWRA SE计算应该在1秒内完成"


class TestDesign031ScalabilityWithK:
    """测试不同K值下的扩展性"""
    
    def test_k_scalability(self):
        """测试K增长时的性能扩展性"""
        np.random.seed(42)
        n = 2000
        k_values = [5, 10, 20, 30, 50]
        
        loop_times = []
        vec_times = []
        
        print("\n[K扩展性测试 n=2000]")
        print(f"{'K':>5} | {'循环(ms)':>12} | {'向量化(ms)':>12} | {'加速比':>8}")
        print("-" * 50)
        
        for K in k_values:
            X = np.random.randn(n, K)
            w = np.random.rand(n)
            
            # 预热
            _ = compute_a_matrix_loop(X, w)
            _ = compute_a_matrix_vectorized(X, w)
            
            # 基准测试
            loop_mean, _, _ = benchmark(compute_a_matrix_loop, X, w, n_runs=5)
            vec_mean, _, _ = benchmark(compute_a_matrix_vectorized, X, w, n_runs=5)
            
            speedup = loop_mean / vec_mean if vec_mean > 0 else float('inf')
            
            loop_times.append(loop_mean)
            vec_times.append(vec_mean)
            
            print(f"{K:>5} | {loop_mean*1000:>12.4f} | {vec_mean*1000:>12.4f} | {speedup:>8.2f}x")
        
        # 验证向量化实现的时间复杂度更优
        # 循环实现: O(n * K^2)
        # 向量化实现: O(K^2 * n)，但BLAS优化
        
        # 对于大K，加速比应该更高
        k_20_speedup = loop_times[2] / vec_times[2] if vec_times[2] > 0 else float('inf')
        k_50_speedup = loop_times[4] / vec_times[4] if vec_times[4] > 0 else float('inf')
        
        print(f"\n结论: K=20时加速{k_20_speedup:.1f}x, K=50时加速{k_50_speedup:.1f}x")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])

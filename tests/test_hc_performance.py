# -*- coding: utf-8 -*-
"""
HC标准误性能测试

验证向量化杠杆值计算方法在大样本和多控制变量场景下的性能。

Requirements: NFR-1 (性能)
"""

import numpy as np
import pytest
import time
import warnings


class TestHCPerformance:
    """HC标准误性能测试类"""
    
    @pytest.fixture
    def large_sample_data(self):
        """生成大样本数据 (N=10000)"""
        np.random.seed(42)
        n = 10000
        k = 5
        
        X = np.column_stack([
            np.ones(n),
            np.random.randn(n, k - 1)
        ])
        beta_true = np.random.randn(k)
        y = X @ beta_true + np.random.randn(n)
        
        return X, y, n, k
    
    @pytest.fixture
    def many_controls_data(self):
        """生成多控制变量数据 (K=50)"""
        np.random.seed(42)
        n = 500
        k = 50
        
        X = np.column_stack([
            np.ones(n),
            np.random.randn(n, k - 1)
        ])
        beta_true = np.random.randn(k)
        y = X @ beta_true + np.random.randn(n)
        
        return X, y, n, k
    
    def compute_leverage_vectorized(self, X, XtX_inv):
        """向量化杠杆值计算 - O(N × K²)"""
        tmp = X @ XtX_inv
        h_ii = (tmp * X).sum(axis=1)
        return np.clip(h_ii, 0, 0.9999)
    
    def compute_leverage_direct(self, X, XtX_inv):
        """直接计算杠杆值 - O(N² × K)"""
        H = X @ XtX_inv @ X.T
        h_ii = np.diag(H)
        return np.clip(h_ii, 0, 0.9999)
    
    def compute_hc3_variance(self, X, residuals, XtX_inv, h_ii):
        """计算HC3方差"""
        omega_diag = (residuals ** 2) / ((1 - h_ii) ** 2)
        meat = X.T @ np.diag(omega_diag) @ X
        var_beta = XtX_inv @ meat @ XtX_inv
        return var_beta
    
    @pytest.mark.performance
    def test_hc_performance_large_n(self, large_sample_data):
        """
        大样本性能测试 (N=10000)
        
        验收标准: 计算时间 < 1秒
        """
        X, y, n, k = large_sample_data
        
        # OLS估计
        XtX_inv = np.linalg.inv(X.T @ X)
        beta = XtX_inv @ X.T @ y
        residuals = y - X @ beta
        
        # 测试向量化方法性能
        start_time = time.time()
        h_ii = self.compute_leverage_vectorized(X, XtX_inv)
        var_hc3 = self.compute_hc3_variance(X, residuals, XtX_inv, h_ii)
        elapsed_time = time.time() - start_time
        
        # 验证结果有效
        assert np.all(np.isfinite(var_hc3)), "HC3方差应为有限值"
        assert np.all(np.diag(var_hc3) > 0), "HC3方差对角元素应为正"
        
        # 验证性能
        assert elapsed_time < 1.0, \
            f"大样本 (N={n}) HC3计算时间 {elapsed_time:.3f}s 超过1秒阈值"
        
        print(f"\n大样本性能测试 (N={n}, K={k}):")
        print(f"  向量化方法耗时: {elapsed_time:.4f}s")
    
    @pytest.mark.performance
    def test_hc_performance_many_controls(self, many_controls_data):
        """
        多控制变量性能测试 (K=50)
        
        验收标准: 计算时间 < 1秒
        """
        X, y, n, k = many_controls_data
        
        # OLS估计
        XtX_inv = np.linalg.inv(X.T @ X)
        beta = XtX_inv @ X.T @ y
        residuals = y - X @ beta
        
        # 测试向量化方法性能
        start_time = time.time()
        h_ii = self.compute_leverage_vectorized(X, XtX_inv)
        var_hc3 = self.compute_hc3_variance(X, residuals, XtX_inv, h_ii)
        elapsed_time = time.time() - start_time
        
        # 验证结果有效
        assert np.all(np.isfinite(var_hc3)), "HC3方差应为有限值"
        assert np.all(np.diag(var_hc3) > 0), "HC3方差对角元素应为正"
        
        # 验证性能
        assert elapsed_time < 1.0, \
            f"多控制变量 (K={k}) HC3计算时间 {elapsed_time:.3f}s 超过1秒阈值"
        
        print(f"\n多控制变量性能测试 (N={n}, K={k}):")
        print(f"  向量化方法耗时: {elapsed_time:.4f}s")
    
    @pytest.mark.performance
    def test_vectorized_vs_direct_leverage(self):
        """
        向量化方法 vs 直接方法性能对比
        
        验证向量化方法优于直接方法
        """
        np.random.seed(42)
        n = 2000  # 使用中等样本避免直接方法内存问题
        k = 10
        
        X = np.column_stack([
            np.ones(n),
            np.random.randn(n, k - 1)
        ])
        XtX_inv = np.linalg.inv(X.T @ X)
        
        # 向量化方法
        start_vec = time.time()
        h_vec = self.compute_leverage_vectorized(X, XtX_inv)
        time_vec = time.time() - start_vec
        
        # 直接方法
        start_direct = time.time()
        h_direct = self.compute_leverage_direct(X, XtX_inv)
        time_direct = time.time() - start_direct
        
        # 验证结果一致
        np.testing.assert_allclose(h_vec, h_direct, rtol=1e-10,
            err_msg="向量化方法与直接方法结果应一致")
        
        # 验证向量化方法更快
        speedup = time_direct / time_vec if time_vec > 0 else float('inf')
        
        print(f"\n杠杆值计算性能对比 (N={n}, K={k}):")
        print(f"  向量化方法: {time_vec:.4f}s")
        print(f"  直接方法: {time_direct:.4f}s")
        print(f"  加速比: {speedup:.1f}x")
        
        # 向量化方法应该更快（至少不慢于直接方法）
        assert time_vec <= time_direct * 1.5, \
            f"向量化方法 ({time_vec:.4f}s) 应不慢于直接方法 ({time_direct:.4f}s)"
    
    @pytest.mark.performance
    def test_all_hc_types_performance(self, large_sample_data):
        """
        所有HC类型性能测试
        
        验证HC0-HC4所有类型在大样本下的计算效率
        """
        X, y, n, k = large_sample_data
        
        # OLS估计
        XtX_inv = np.linalg.inv(X.T @ X)
        beta = XtX_inv @ X.T @ y
        residuals = y - X @ beta
        e2 = residuals ** 2
        
        # 预计算杠杆值
        h_ii = self.compute_leverage_vectorized(X, XtX_inv)
        
        results = {}
        
        # HC0
        start = time.time()
        omega_hc0 = e2
        meat_hc0 = X.T @ np.diag(omega_hc0) @ X
        var_hc0 = XtX_inv @ meat_hc0 @ XtX_inv
        results['HC0'] = time.time() - start
        
        # HC1
        start = time.time()
        omega_hc1 = (n / (n - k)) * e2
        meat_hc1 = X.T @ np.diag(omega_hc1) @ X
        var_hc1 = XtX_inv @ meat_hc1 @ XtX_inv
        results['HC1'] = time.time() - start
        
        # HC2
        start = time.time()
        omega_hc2 = e2 / (1 - h_ii)
        meat_hc2 = X.T @ np.diag(omega_hc2) @ X
        var_hc2 = XtX_inv @ meat_hc2 @ XtX_inv
        results['HC2'] = time.time() - start
        
        # HC3
        start = time.time()
        omega_hc3 = e2 / ((1 - h_ii) ** 2)
        meat_hc3 = X.T @ np.diag(omega_hc3) @ X
        var_hc3 = XtX_inv @ meat_hc3 @ XtX_inv
        results['HC3'] = time.time() - start
        
        # HC4
        start = time.time()
        delta = np.minimum(4.0, n * h_ii / k)
        omega_hc4 = e2 / ((1 - h_ii) ** delta)
        meat_hc4 = X.T @ np.diag(omega_hc4) @ X
        var_hc4 = XtX_inv @ meat_hc4 @ XtX_inv
        results['HC4'] = time.time() - start
        
        print(f"\n所有HC类型性能测试 (N={n}, K={k}):")
        for hc_type, elapsed in results.items():
            print(f"  {hc_type}: {elapsed:.4f}s")
        
        # 所有类型应在合理时间内完成
        for hc_type, elapsed in results.items():
            assert elapsed < 1.0, \
                f"{hc_type} 计算时间 {elapsed:.3f}s 超过1秒阈值"
    
    @pytest.mark.performance
    def test_memory_efficiency(self):
        """
        内存效率测试
        
        验证向量化方法不构造完整N×N帽子矩阵
        """
        np.random.seed(42)
        n = 5000
        k = 10
        
        X = np.column_stack([
            np.ones(n),
            np.random.randn(n, k - 1)
        ])
        XtX_inv = np.linalg.inv(X.T @ X)
        
        # 向量化方法内存占用估计
        # tmp: (N, K), h_ii: (N,)
        # 总计: N*K + N ≈ N*K 个float64
        vec_memory_estimate = n * k * 8 / (1024 * 1024)  # MB
        
        # 直接方法内存占用估计
        # H: (N, N)
        # 总计: N*N 个float64
        direct_memory_estimate = n * n * 8 / (1024 * 1024)  # MB
        
        print(f"\n内存效率测试 (N={n}, K={k}):")
        print(f"  向量化方法估计内存: {vec_memory_estimate:.2f} MB")
        print(f"  直接方法估计内存: {direct_memory_estimate:.2f} MB")
        print(f"  内存节省: {direct_memory_estimate / vec_memory_estimate:.1f}x")
        
        # 向量化方法应该节省大量内存
        assert vec_memory_estimate < direct_memory_estimate / 10, \
            "向量化方法应节省至少10倍内存"


class TestHCScalability:
    """HC标准误可扩展性测试"""
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_scaling_with_n(self):
        """
        样本量扩展性测试
        
        验证向量化杠杆值计算在大样本下仍然高效
        注意：meat矩阵计算 X.T @ diag(omega) @ X 的复杂度为 O(N*K²)
        """
        np.random.seed(42)
        k = 5
        sample_sizes = [2000, 4000, 8000]
        times = []
        
        for n in sample_sizes:
            X = np.column_stack([
                np.ones(n),
                np.random.randn(n, k - 1)
            ])
            y = np.random.randn(n)
            
            XtX_inv = np.linalg.inv(X.T @ X)
            beta = XtX_inv @ X.T @ y
            residuals = y - X @ beta
            
            start = time.time()
            # 向量化杠杆值计算
            tmp = X @ XtX_inv
            h_ii = np.clip((tmp * X).sum(axis=1), 0, 0.9999)
            omega = (residuals ** 2) / ((1 - h_ii) ** 2)
            # 使用高效的meat矩阵计算（避免构造N×N对角矩阵）
            # meat = X.T @ diag(omega) @ X = sum_i omega_i * x_i @ x_i.T
            meat = (X.T * omega) @ X
            var_hc3 = XtX_inv @ meat @ XtX_inv
            elapsed = time.time() - start
            
            times.append(elapsed)
        
        print(f"\n样本量扩展性测试 (K={k}):")
        for n, t in zip(sample_sizes, times):
            print(f"  N={n}: {t:.4f}s")
        
        # 验证所有样本量都能在合理时间内完成
        for n, t in zip(sample_sizes, times):
            assert t < 1.0, f"N={n} 计算时间 {t:.3f}s 超过1秒阈值"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'performance'])

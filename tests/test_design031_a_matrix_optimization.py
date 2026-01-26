"""
DESIGN-031: A矩阵计算向量化优化的数值一致性测试

验证向量化实现与原始双重循环实现产生相同的数值结果。
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats


class TestAMatrixVectorizationNumericalConsistency:
    """验证向量化与循环实现的数值一致性"""
    
    @pytest.fixture
    def sample_data(self):
        """生成测试数据"""
        np.random.seed(42)
        n = 500
        K = 5  # 协变量数量
        
        # 生成协变量
        X = np.random.randn(n, K)
        X_with_const = np.column_stack([np.ones(n), X])  # (n, K+1)
        
        # 生成处理指示符
        D = (np.random.rand(n) > 0.6).astype(float)
        
        # 生成结果变量
        Y = X @ np.random.randn(K) + D * 2 + np.random.randn(n)
        
        # 生成倾向得分
        pscores = 1 / (1 + np.exp(-X @ np.random.randn(K) * 0.5))
        pscores = np.clip(pscores, 0.01, 0.99)
        
        # 生成权重
        weights = pscores / (1 - pscores)
        
        return {
            'n': n,
            'K': K + 1,  # 包含截距
            'X': X_with_const,
            'D': D,
            'Y': Y,
            'pscores': pscores,
            'weights': weights,
        }
    
    def test_ra_se_core_a_matrix_block(self, sample_data):
        """测试 _compute_ra_se_core 中 A[1:,1:] 块的向量化一致性"""
        X = sample_data['X']
        D = sample_data['D']
        n = sample_data['n']
        K = sample_data['K']
        
        # 循环实现
        A_loop = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                A_loop[i, j] = -np.mean((1 - D) * X[:, i] * X[:, j])
        
        # 向量化实现
        control_weight = (1 - D).reshape(-1, 1)
        A_vec = -((X * control_weight).T @ X) / n
        
        # 验证数值一致性
        np.testing.assert_allclose(
            A_vec, A_loop,
            rtol=1e-14, atol=1e-14,
            err_msg="RA SE core A matrix block: vectorized != loop"
        )
        
        # 打印最大差异
        max_diff = np.max(np.abs(A_vec - A_loop))
        print(f"\n[RA A矩阵] 最大绝对差异: {max_diff:.2e}")
    
    def test_ipwra_ps_block_a_matrix(self, sample_data):
        """测试 compute_ipwra_se_analytical 中 PS 块的向量化一致性"""
        X = sample_data['X']
        pscores = sample_data['pscores']
        n = sample_data['n']
        K = sample_data['K']
        
        # 循环实现
        A_loop = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                A_loop[i, j] = -np.mean(pscores * (1 - pscores) * X[:, i] * X[:, j])
        
        # 向量化实现
        ps_weight = (pscores * (1 - pscores)).reshape(-1, 1)
        A_vec = -((X * ps_weight).T @ X) / n
        
        # 验证数值一致性
        np.testing.assert_allclose(
            A_vec, A_loop,
            rtol=1e-14, atol=1e-14,
            err_msg="IPWRA PS block A matrix: vectorized != loop"
        )
        
        max_diff = np.max(np.abs(A_vec - A_loop))
        print(f"\n[IPWRA PS块] 最大绝对差异: {max_diff:.2e}")
    
    def test_ipwra_om_block_a_matrix(self, sample_data):
        """测试 compute_ipwra_se_analytical 中 OM 块的向量化一致性"""
        X = sample_data['X']
        D = sample_data['D']
        weights = sample_data['weights']
        n = sample_data['n']
        K = sample_data['K']
        
        # 循环实现
        A_loop = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                A_loop[i, j] = -np.mean(weights * (1 - D) * X[:, i] * X[:, j])
        
        # 向量化实现
        om_weight = (weights * (1 - D)).reshape(-1, 1)
        A_vec = -((X * om_weight).T @ X) / n
        
        # 验证数值一致性
        np.testing.assert_allclose(
            A_vec, A_loop,
            rtol=1e-14, atol=1e-14,
            err_msg="IPWRA OM block A matrix: vectorized != loop"
        )
        
        max_diff = np.max(np.abs(A_vec - A_loop))
        print(f"\n[IPWRA OM块] 最大绝对差异: {max_diff:.2e}")
    
    def test_ipw_ps_block_a_matrix(self, sample_data):
        """测试 _compute_ipw_se_analytical 中 PS 块的向量化一致性"""
        X = sample_data['X']
        pscores = sample_data['pscores']
        n = sample_data['n']
        K = sample_data['K']
        
        # 循环实现
        A_loop = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                A_loop[i, j] = -np.mean(pscores * (1 - pscores) * X[:, i] * X[:, j])
        
        # 向量化实现
        ps_weight = (pscores * (1 - pscores)).reshape(-1, 1)
        A_vec = -((X * ps_weight).T @ X) / n
        
        # 验证数值一致性
        np.testing.assert_allclose(
            A_vec, A_loop,
            rtol=1e-14, atol=1e-14,
            err_msg="IPW PS block A matrix: vectorized != loop"
        )
        
        max_diff = np.max(np.abs(A_vec - A_loop))
        print(f"\n[IPW PS块] 最大绝对差异: {max_diff:.2e}")


class TestEstimatorSEConsistency:
    """测试优化后估计器SE计算的正确性"""
    
    @pytest.fixture
    def cross_sectional_data(self):
        """生成横截面测试数据"""
        np.random.seed(123)
        n = 300
        
        # 生成协变量
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        
        # 生成处理概率和处理指示符
        ps_true = 1 / (1 + np.exp(-(0.5 * x1 - 0.3 * x2)))
        D = (np.random.rand(n) < ps_true).astype(int)
        
        # 生成结果
        Y = 2 + 1.5 * x1 + 0.8 * x2 + 3 * D + np.random.randn(n)
        
        return pd.DataFrame({
            'Y': Y,
            'D': D,
            'x1': x1,
            'x2': x2,
        })
    
    def test_ra_estimator_se(self, cross_sectional_data):
        """测试RA估计器优化后SE计算"""
        from lwdid.staggered.estimators import estimate_ra
        
        result = estimate_ra(
            data=cross_sectional_data,
            y='Y',
            d='D',
            controls=['x1', 'x2'],
            vce='robust',
        )
        
        # 验证SE为正数且合理
        assert result.se > 0, "SE should be positive"
        assert result.se < 10, "SE should be reasonable"
        assert np.isfinite(result.se), "SE should be finite"
        
        # 验证置信区间合理
        assert result.ci_lower < result.att < result.ci_upper
        
        print(f"\n[RA估计器] ATT={result.att:.4f}, SE={result.se:.4f}")
        print(f"  95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
    
    def test_ipw_estimator_se(self, cross_sectional_data):
        """测试IPW估计器优化后SE计算"""
        from lwdid.staggered.estimators import estimate_ipw
        
        result = estimate_ipw(
            data=cross_sectional_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            se_method='analytical',
        )
        
        # 验证SE为正数且合理
        assert result.se > 0, "SE should be positive"
        assert result.se < 10, "SE should be reasonable"
        assert np.isfinite(result.se), "SE should be finite"
        
        # 验证置信区间合理
        assert result.ci_lower < result.att < result.ci_upper
        
        print(f"\n[IPW估计器] ATT={result.att:.4f}, SE={result.se:.4f}")
        print(f"  95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
    
    def test_ipwra_estimator_se(self, cross_sectional_data):
        """测试IPWRA估计器优化后SE计算"""
        from lwdid.staggered.estimators import estimate_ipwra
        
        result = estimate_ipwra(
            data=cross_sectional_data,
            y='Y',
            d='D',
            controls=['x1', 'x2'],
            propensity_controls=['x1', 'x2'],
            se_method='analytical',
        )
        
        # 验证SE为正数且合理
        assert result.se > 0, "SE should be positive"
        assert result.se < 10, "SE should be reasonable"
        assert np.isfinite(result.se), "SE should be finite"
        
        # 验证置信区间合理
        assert result.ci_lower < result.att < result.ci_upper
        
        print(f"\n[IPWRA估计器] ATT={result.att:.4f}, SE={result.se:.4f}")
        print(f"  95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
    
    def test_analytical_vs_bootstrap_consistency(self, cross_sectional_data):
        """测试解析SE与Bootstrap SE的一致性"""
        from lwdid.staggered.estimators import estimate_ipw
        
        # 解析SE
        result_analytical = estimate_ipw(
            data=cross_sectional_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            se_method='analytical',
        )
        
        # Bootstrap SE
        result_bootstrap = estimate_ipw(
            data=cross_sectional_data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            se_method='bootstrap',
            n_bootstrap=500,
            seed=42,
        )
        
        # ATT点估计应该相同
        np.testing.assert_allclose(
            result_analytical.att,
            result_bootstrap.att,
            rtol=1e-10,
            err_msg="ATT point estimates should match"
        )
        
        # SE应该接近（允许Bootstrap的随机性）
        se_ratio = result_analytical.se / result_bootstrap.se
        assert 0.7 < se_ratio < 1.4, f"SE ratio {se_ratio:.2f} is outside acceptable range"
        
        print(f"\n[解析vs Bootstrap] Analytical SE={result_analytical.se:.4f}")
        print(f"                    Bootstrap SE={result_bootstrap.se:.4f}")
        print(f"                    Ratio={se_ratio:.2f}")


class TestLargeKPerformance:
    """测试大K值情况下的数值稳定性"""
    
    def test_many_covariates(self):
        """测试多协变量情况"""
        np.random.seed(456)
        n = 500
        K = 20  # 较多协变量
        
        # 生成数据
        X = np.random.randn(n, K)
        D = (np.random.rand(n) > 0.5).astype(float)
        pscores = np.clip(1 / (1 + np.exp(-X @ np.random.randn(K) * 0.3)), 0.05, 0.95)
        weights = pscores / (1 - pscores)
        
        # 测试向量化计算不会产生数值问题
        X_with_const = np.column_stack([np.ones(n), X])
        K_full = K + 1
        
        # RA块
        control_weight = (1 - D).reshape(-1, 1)
        A_ra = -((X_with_const * control_weight).T @ X_with_const) / n
        assert np.all(np.isfinite(A_ra)), "RA A matrix should be finite"
        
        # IPWRA PS块
        ps_weight = (pscores * (1 - pscores)).reshape(-1, 1)
        A_ipwra_ps = -((X_with_const * ps_weight).T @ X_with_const) / n
        assert np.all(np.isfinite(A_ipwra_ps)), "IPWRA PS A matrix should be finite"
        
        # IPWRA OM块
        om_weight = (weights * (1 - D)).reshape(-1, 1)
        A_ipwra_om = -((X_with_const * om_weight).T @ X_with_const) / n
        assert np.all(np.isfinite(A_ipwra_om)), "IPWRA OM A matrix should be finite"
        
        print(f"\n[多协变量测试 K={K}] 所有A矩阵计算数值稳定")
        print(f"  RA块条件数: {np.linalg.cond(A_ra):.2e}")
        print(f"  PS块条件数: {np.linalg.cond(A_ipwra_ps):.2e}")
        print(f"  OM块条件数: {np.linalg.cond(A_ipwra_om):.2e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])

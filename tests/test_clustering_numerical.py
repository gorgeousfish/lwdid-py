"""
数值验证测试：聚类稳健标准误和 Wild Cluster Bootstrap

测试内容：
- Cluster-robust SE 公式验证
- G-1 自由度调整验证
- Bootstrap 权重分布验证
- Bootstrap 覆盖率验证
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats
import statsmodels.api as sm

from lwdid.inference.wild_bootstrap import (
    wild_cluster_bootstrap,
    _generate_bootstrap_weights,
)


# =============================================================================
# Cluster-Robust SE 数值验证
# =============================================================================

class TestClusterRobustSENumerical:
    """Cluster-robust 标准误的数值验证"""
    
    def test_cluster_se_formula(self):
        """
        验证 cluster-robust SE 公式：
        V = (X'X)^{-1} * (Σ_c X_c' û_c û_c' X_c) * (X'X)^{-1}
        """
        np.random.seed(42)
        
        # 生成简单的聚类数据
        n_clusters = 20
        obs_per_cluster = 50
        n = n_clusters * obs_per_cluster
        
        # 聚类 ID
        cluster_ids = np.repeat(range(n_clusters), obs_per_cluster)
        
        # 处理（在聚类级别变化）
        D = np.repeat(np.random.binomial(1, 0.5, n_clusters), obs_per_cluster)
        
        # 结果变量（带聚类级别误差）
        cluster_effects = np.repeat(np.random.normal(0, 2, n_clusters), obs_per_cluster)
        Y = 10 + 2 * D + cluster_effects + np.random.normal(0, 1, n)
        
        # 手动计算
        X = np.column_stack([np.ones(n), D])
        XtX_inv = np.linalg.inv(X.T @ X)
        beta = XtX_inv @ X.T @ Y
        residuals = Y - X @ beta
        
        # Cluster-robust meat 矩阵
        meat = np.zeros((2, 2))
        for c in range(n_clusters):
            mask = cluster_ids == c
            X_c = X[mask]
            u_c = residuals[mask]
            score_c = X_c.T @ u_c
            meat += np.outer(score_c, score_c)
        
        # 有限样本校正
        G = n_clusters
        k = 2
        correction = (G / (G - 1)) * ((n - 1) / (n - k))
        V_manual = correction * XtX_inv @ meat @ XtX_inv
        se_manual = np.sqrt(V_manual[1, 1])
        
        # 与 statsmodels 比较
        model = sm.OLS(Y, X)
        results = model.fit(cov_type='cluster', cov_kwds={'groups': cluster_ids})
        se_statsmodels = results.bse[1]
        
        # 应在数值容差内匹配
        assert abs(se_manual - se_statsmodels) / se_statsmodels < 0.01
    
    def test_df_adjustment_g_minus_1(self):
        """
        验证 cluster-robust 推断使用 G-1 自由度。
        
        t 临界值应使用 df = G - 1，而非 n - k。
        """
        G = 20  # 聚类数
        n = 1000  # 观测数
        k = 2  # 参数数
        alpha = 0.05
        
        # G-1 df（cluster-robust 正确）
        t_crit_cluster = stats.t.ppf(1 - alpha/2, G - 1)
        
        # n-k df（cluster-robust 不正确）
        t_crit_ols = stats.t.ppf(1 - alpha/2, n - k)
        
        # 聚类 df 应给出更宽的 CI（更保守）
        assert t_crit_cluster > t_crit_ols
        
        # 验证具体值
        assert abs(t_crit_cluster - 2.093) < 0.01  # t_{19, 0.975}
    
    def test_cluster_se_increases_with_correlation(self):
        """
        聚类内相关性增加时，cluster-robust SE 应增加。
        """
        np.random.seed(42)
        n_clusters = 30
        obs_per_cluster = 40
        n = n_clusters * obs_per_cluster
        
        cluster_ids = np.repeat(range(n_clusters), obs_per_cluster)
        D = np.repeat(np.random.binomial(1, 0.5, n_clusters), obs_per_cluster)
        
        # 低聚类内相关
        cluster_effects_low = np.repeat(np.random.normal(0, 0.5, n_clusters), obs_per_cluster)
        Y_low = 10 + 2 * D + cluster_effects_low + np.random.normal(0, 2, n)
        
        # 高聚类内相关
        cluster_effects_high = np.repeat(np.random.normal(0, 3, n_clusters), obs_per_cluster)
        Y_high = 10 + 2 * D + cluster_effects_high + np.random.normal(0, 0.5, n)
        
        X = sm.add_constant(D)
        
        model_low = sm.OLS(Y_low, X)
        results_low = model_low.fit(cov_type='cluster', cov_kwds={'groups': cluster_ids})
        
        model_high = sm.OLS(Y_high, X)
        results_high = model_high.fit(cov_type='cluster', cov_kwds={'groups': cluster_ids})
        
        # 高相关应有更大的 SE
        assert results_high.bse[1] > results_low.bse[1]


# =============================================================================
# Wild Cluster Bootstrap 数值验证
# =============================================================================

class TestWildClusterBootstrapNumerical:
    """Wild cluster bootstrap 的数值验证"""
    
    def test_rademacher_weights_distribution(self):
        """验证 Rademacher 权重具有正确的分布"""
        np.random.seed(42)
        n_samples = 10000
        n_clusters = 50
        
        weights = np.array([
            _generate_bootstrap_weights(n_clusters, 'rademacher')
            for _ in range(n_samples)
        ])
        
        # 应只有 +1 或 -1
        assert np.all(np.isin(weights, [-1, 1]))
        
        # 均值应约为 0
        assert abs(weights.mean()) < 0.02
        
        # 方差应约为 1
        assert abs(weights.var() - 1) < 0.02
    
    def test_mammen_weights_distribution(self):
        """验证 Mammen 权重具有正确的矩"""
        np.random.seed(42)
        n_samples = 10000
        n_clusters = 50
        
        weights = np.array([
            _generate_bootstrap_weights(n_clusters, 'mammen')
            for _ in range(n_samples)
        ])
        
        # E[w] = 0
        assert abs(weights.mean()) < 0.02
        
        # E[w^2] = 1
        assert abs((weights ** 2).mean() - 1) < 0.02
        
        # E[w^3] = 1（偏度匹配）
        assert abs((weights ** 3).mean() - 1) < 0.15
    
    def test_webb_weights_distribution(self):
        """验证 Webb 权重具有正确的矩"""
        np.random.seed(42)
        n_samples = 10000
        n_clusters = 50
        
        weights = np.array([
            _generate_bootstrap_weights(n_clusters, 'webb')
            for _ in range(n_samples)
        ])
        
        # E[w] = 0
        assert abs(weights.mean()) < 0.02
        
        # E[w^2] = 1
        assert abs((weights ** 2).mean() - 1) < 0.02
    
    def test_invalid_weight_type_raises(self):
        """无效的权重类型应引发错误"""
        with pytest.raises(ValueError, match="Unknown weight_type"):
            _generate_bootstrap_weights(10, 'invalid')
    
    def test_bootstrap_pvalue_bounds(self):
        """Bootstrap p 值应在 [0, 1] 范围内"""
        np.random.seed(42)
        n_clusters = 20
        obs_per_cluster = 30
        
        cluster_ids = np.repeat(range(n_clusters), obs_per_cluster)
        D = np.repeat(np.random.binomial(1, 0.5, n_clusters), obs_per_cluster)
        cluster_effects = np.repeat(np.random.normal(0, 2, n_clusters), obs_per_cluster)
        Y = 10 + 2 * D + cluster_effects + np.random.normal(0, 1, len(D))
        
        data = pd.DataFrame({
            'Y': Y, 'D': D, 'cluster': cluster_ids
        })
        
        result = wild_cluster_bootstrap(
            data, y_transformed='Y', d='D',
            cluster_var='cluster', n_bootstrap=199,
            seed=42
        )
        
        assert 0 <= result.pvalue <= 1
    
    def test_bootstrap_ci_contains_point_estimate(self):
        """Bootstrap CI 应合理（在 impose_null=False 时包含点估计）"""
        np.random.seed(42)
        n_clusters = 25
        obs_per_cluster = 40
        
        cluster_ids = np.repeat(range(n_clusters), obs_per_cluster)
        D = np.repeat(np.random.binomial(1, 0.5, n_clusters), obs_per_cluster)
        cluster_effects = np.repeat(np.random.normal(0, 1, n_clusters), obs_per_cluster)
        Y = 10 + 2 * D + cluster_effects + np.random.normal(0, 1, len(D))
        
        data = pd.DataFrame({
            'Y': Y, 'D': D, 'cluster': cluster_ids
        })
        
        # 使用 impose_null=False 时，CI 应包含点估计
        result = wild_cluster_bootstrap(
            data, y_transformed='Y', d='D',
            cluster_var='cluster', n_bootstrap=499,
            alpha=0.05, seed=42,
            impose_null=False  # 不施加零假设
        )
        
        # CI 应包含点估计
        assert result.ci_lower <= result.att <= result.ci_upper
    
    def test_bootstrap_reproducibility(self):
        """给定相同种子，结果应可复现"""
        np.random.seed(42)
        n_clusters = 15
        obs_per_cluster = 30
        
        cluster_ids = np.repeat(range(n_clusters), obs_per_cluster)
        D = np.repeat(np.random.binomial(1, 0.5, n_clusters), obs_per_cluster)
        Y = 10 + 2 * D + np.random.normal(0, 1, len(D))
        
        data = pd.DataFrame({
            'Y': Y, 'D': D, 'cluster': cluster_ids
        })
        
        result1 = wild_cluster_bootstrap(
            data, y_transformed='Y', d='D',
            cluster_var='cluster', n_bootstrap=99,
            seed=123
        )
        
        result2 = wild_cluster_bootstrap(
            data, y_transformed='Y', d='D',
            cluster_var='cluster', n_bootstrap=99,
            seed=123
        )
        
        assert result1.pvalue == result2.pvalue
        assert result1.se_bootstrap == result2.se_bootstrap


# =============================================================================
# 自由度公式验证
# =============================================================================

class TestDegreesOfFreedomFormula:
    """自由度公式验证"""
    
    def test_t_critical_values(self):
        """验证 t 分布临界值"""
        test_cases = [
            (10, 0.05, 2.262),   # G=10, α=0.05, t_{9,0.975}
            (20, 0.05, 2.093),   # G=20, α=0.05, t_{19,0.975}
            (30, 0.05, 2.045),   # G=30, α=0.05, t_{29,0.975}
            (50, 0.05, 2.010),   # G=50, α=0.05, t_{49,0.975}
        ]
        
        for G, alpha, expected_t in test_cases:
            df = G - 1
            t_crit = stats.t.ppf(1 - alpha/2, df)
            assert abs(t_crit - expected_t) < 0.01, \
                f"G={G}: expected {expected_t}, got {t_crit:.3f}"
    
    def test_ci_width_decreases_with_clusters(self):
        """聚类数增加时，CI 宽度应减小"""
        alpha = 0.05
        se = 1.0  # 固定 SE
        
        widths = []
        for G in [10, 20, 30, 50, 100]:
            df = G - 1
            t_crit = stats.t.ppf(1 - alpha/2, df)
            width = 2 * t_crit * se
            widths.append(width)
        
        # 宽度应单调递减
        for i in range(len(widths) - 1):
            assert widths[i] > widths[i + 1]


# =============================================================================
# Sandwich 估计量验证
# =============================================================================

class TestSandwichEstimator:
    """Sandwich 估计量验证"""
    
    def test_sandwich_positive_definite(self):
        """Sandwich 方差矩阵应为正定"""
        np.random.seed(42)
        n_clusters = 20
        obs_per_cluster = 30
        n = n_clusters * obs_per_cluster
        
        cluster_ids = np.repeat(range(n_clusters), obs_per_cluster)
        D = np.repeat(np.random.binomial(1, 0.5, n_clusters), obs_per_cluster)
        Y = 10 + 2 * D + np.random.normal(0, 1, n)
        
        X = sm.add_constant(D)
        model = sm.OLS(Y, X)
        results = model.fit(cov_type='cluster', cov_kwds={'groups': cluster_ids})
        
        # 方差矩阵应为正定（所有特征值 > 0）
        eigenvalues = np.linalg.eigvalsh(results.cov_params())
        assert np.all(eigenvalues > 0)
    
    def test_sandwich_symmetric(self):
        """Sandwich 方差矩阵应为对称"""
        np.random.seed(42)
        n_clusters = 20
        obs_per_cluster = 30
        n = n_clusters * obs_per_cluster
        
        cluster_ids = np.repeat(range(n_clusters), obs_per_cluster)
        D = np.repeat(np.random.binomial(1, 0.5, n_clusters), obs_per_cluster)
        Y = 10 + 2 * D + np.random.normal(0, 1, n)
        
        X = sm.add_constant(D)
        model = sm.OLS(Y, X)
        results = model.fit(cov_type='cluster', cov_kwds={'groups': cluster_ids})
        
        V = results.cov_params()
        assert np.allclose(V, V.T)

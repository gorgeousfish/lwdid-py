"""
vibe-math MCP 公式验证测试

使用 vibe-math MCP 工具验证聚类相关公式的正确性。

测试内容：
- Sandwich 估计量公式
- 自由度公式
- Bootstrap 权重公式（Mammen, Webb）
"""

import numpy as np
import pytest
from scipy import stats


# =============================================================================
# Sandwich 估计量公式验证
# =============================================================================

class TestSandwichEstimatorFormula:
    """使用数学验证 Sandwich 估计量公式"""
    
    def test_sandwich_formula_simple_case(self):
        """
        验证 Sandwich 估计量：
        V = (X'X)^{-1} * B * (X'X)^{-1}
        其中 B = Σ_c X_c' û_c û_c' X_c
        
        简单 2x2 示例
        """
        # X'X = [[n, Σd], [Σd, Σd]]
        n = 100
        n1 = 40  # 处理组
        n0 = 60  # 控制组
        
        # (X'X)^{-1} 计算
        XtX = np.array([[n, n1], [n1, n1]], dtype=float)
        XtX_inv = np.linalg.inv(XtX)
        
        # 验证逆矩阵
        identity = XtX @ XtX_inv
        assert np.allclose(identity, np.eye(2), atol=1e-10)
        
        # Meat 矩阵（简化示例）
        # B = Σ_c (Σ_{i∈c} x_i u_i)(Σ_{i∈c} x_i u_i)'
        # 2 个聚类，大小相等
        G = 2
        score_1 = np.array([5.0, 3.0])  # 聚类 1 得分
        score_2 = np.array([-4.0, -2.0])  # 聚类 2 得分
        
        B = np.outer(score_1, score_1) + np.outer(score_2, score_2)
        
        # Sandwich 方差
        V = XtX_inv @ B @ XtX_inv
        
        # 有限样本校正
        k = 2
        correction = (G / (G - 1)) * ((n - 1) / (n - k))
        V_corrected = correction * V
        
        # 处理系数的 SE
        se_tau = np.sqrt(V_corrected[1, 1])
        
        # 验证正且合理
        assert se_tau > 0
        assert se_tau < 10  # 合理的量级
    
    def test_xtx_inverse_formula(self):
        """
        验证 2x2 矩阵逆的解析公式：
        [[a, b], [c, d]]^{-1} = (1/det) * [[d, -b], [-c, a]]
        其中 det = ad - bc
        """
        a, b, c, d = 100, 40, 40, 40
        
        # 解析逆
        det = a * d - b * c
        inv_analytic = np.array([[d, -b], [-c, a]]) / det
        
        # 数值逆
        M = np.array([[a, b], [c, d]], dtype=float)
        inv_numeric = np.linalg.inv(M)
        
        assert np.allclose(inv_analytic, inv_numeric)
    
    def test_finite_sample_correction_formula(self):
        """
        验证有限样本校正公式：
        correction = (G / (G-1)) * ((n-1) / (n-k))
        """
        test_cases = [
            (20, 100, 2),   # G=20, n=100, k=2
            (50, 500, 3),   # G=50, n=500, k=3
            (10, 200, 4),   # G=10, n=200, k=4
        ]
        
        for G, n, k in test_cases:
            correction = (G / (G - 1)) * ((n - 1) / (n - k))
            
            # 校正应 > 1（放大方差）
            assert correction > 1
            
            # 当 G 和 n 都很大时，校正应接近 1
            if G > 30 and n > 300:
                assert correction < 1.1


# =============================================================================
# 自由度公式验证
# =============================================================================

class TestDegreesOfFreedomFormula:
    """验证 t 分布自由度公式"""
    
    def test_t_critical_values_formula(self):
        """
        验证 G-1 自由度的 t 临界值。
        """
        test_cases = [
            (10, 0.05, 2.262),   # G=10, α=0.05, t_{9,0.975}
            (20, 0.05, 2.093),   # G=20, α=0.05, t_{19,0.975}
            (30, 0.05, 2.045),   # G=30, α=0.05, t_{29,0.975}
            (50, 0.05, 2.010),   # G=50, α=0.05, t_{49,0.975}
            (100, 0.05, 1.984),  # G=100, α=0.05, t_{99,0.975}
        ]
        
        for G, alpha, expected_t in test_cases:
            df = G - 1
            t_crit = stats.t.ppf(1 - alpha/2, df)
            assert abs(t_crit - expected_t) < 0.01, \
                f"G={G}: expected {expected_t}, got {t_crit:.3f}"
    
    def test_t_converges_to_normal(self):
        """
        当 df → ∞ 时，t 分布应收敛到正态分布。
        """
        alpha = 0.05
        z_crit = stats.norm.ppf(1 - alpha/2)  # ≈ 1.96
        
        # 大 df 时 t 临界值应接近 z 临界值
        for df in [100, 500, 1000]:
            t_crit = stats.t.ppf(1 - alpha/2, df)
            assert abs(t_crit - z_crit) < 0.05
    
    def test_ci_formula(self):
        """
        验证置信区间公式：
        CI = β̂ ± t_{df, 1-α/2} * SE(β̂)
        """
        beta_hat = 2.0
        se = 0.5
        alpha = 0.05
        df = 19  # G = 20
        
        t_crit = stats.t.ppf(1 - alpha/2, df)
        ci_lower = beta_hat - t_crit * se
        ci_upper = beta_hat + t_crit * se
        
        # 验证 CI 宽度
        ci_width = ci_upper - ci_lower
        expected_width = 2 * t_crit * se
        assert abs(ci_width - expected_width) < 1e-10
        
        # 验证 CI 对称性
        assert abs((ci_upper - beta_hat) - (beta_hat - ci_lower)) < 1e-10


# =============================================================================
# Bootstrap 权重公式验证
# =============================================================================

class TestBootstrapWeightFormulas:
    """验证 Bootstrap 权重公式"""
    
    def test_mammen_weight_formula(self):
        """
        验证 Mammen 两点分布：
        P(w = -(√5-1)/2) = (√5+1)/(2√5)
        P(w = (√5+1)/2) = (√5-1)/(2√5)
        
        性质：E[w]=0, E[w²]=1, E[w³]=1
        """
        # 使用精确值计算
        sqrt5 = np.sqrt(5)
        
        w1 = -(sqrt5 - 1) / 2  # ≈ -0.618
        w2 = (sqrt5 + 1) / 2   # ≈ 1.618
        
        p1 = (sqrt5 + 1) / (2 * sqrt5)  # P(w = w1)
        p2 = (sqrt5 - 1) / (2 * sqrt5)  # P(w = w2)
        
        # 验证概率和为 1
        assert abs(p1 + p2 - 1) < 1e-10
        
        # 验证 E[w] = 0
        E_w = p1 * w1 + p2 * w2
        assert abs(E_w) < 1e-10
        
        # 验证 E[w²] = 1
        E_w2 = p1 * w1**2 + p2 * w2**2
        assert abs(E_w2 - 1) < 1e-10
        
        # 验证 E[w³] = 1
        E_w3 = p1 * w1**3 + p2 * w2**3
        assert abs(E_w3 - 1) < 1e-10
    
    def test_mammen_weight_values(self):
        """验证 Mammen 权重的具体数值"""
        sqrt5 = np.sqrt(5)
        
        w1 = -(sqrt5 - 1) / 2
        w2 = (sqrt5 + 1) / 2
        
        # 验证近似值
        assert abs(w1 - (-0.618034)) < 0.0001
        assert abs(w2 - 1.618034) < 0.0001
        
        # 黄金比例关系
        phi = (1 + sqrt5) / 2  # 黄金比例 ≈ 1.618
        assert abs(w2 - phi) < 1e-10
        assert abs(w1 - (-1/phi)) < 1e-10
    
    def test_webb_weight_formula(self):
        """
        验证 Webb 六点分布。
        
        值：±√(1/2), ±√(2/2), ±√(3/2)
        每个概率为 1/6
        """
        values = np.array([
            -np.sqrt(3/2), -np.sqrt(2/2), -np.sqrt(1/2),
            np.sqrt(1/2), np.sqrt(2/2), np.sqrt(3/2)
        ])
        
        # 等概率
        p = 1/6
        
        # 验证 E[w] = 0（对称）
        E_w = np.sum(values) * p
        assert abs(E_w) < 1e-10
        
        # 验证 E[w²] = 1
        # = (1/6) * (3/2 + 2/2 + 1/2 + 1/2 + 2/2 + 3/2)
        # = (1/6) * 6 = 1
        E_w2 = np.sum(values**2) * p
        assert abs(E_w2 - 1) < 1e-10
    
    def test_webb_weight_values(self):
        """验证 Webb 权重的具体数值"""
        expected_values = [
            -np.sqrt(1.5),  # ≈ -1.225
            -np.sqrt(1.0),  # = -1.0
            -np.sqrt(0.5),  # ≈ -0.707
            np.sqrt(0.5),   # ≈ 0.707
            np.sqrt(1.0),   # = 1.0
            np.sqrt(1.5),   # ≈ 1.225
        ]
        
        for val in expected_values:
            assert abs(val**2 - round(val**2 * 2) / 2) < 1e-10
    
    def test_rademacher_weight_formula(self):
        """
        验证 Rademacher 分布：
        P(w = 1) = P(w = -1) = 0.5
        
        性质：E[w]=0, E[w²]=1
        """
        w1, w2 = -1, 1
        p1, p2 = 0.5, 0.5
        
        # E[w] = 0
        E_w = p1 * w1 + p2 * w2
        assert E_w == 0
        
        # E[w²] = 1
        E_w2 = p1 * w1**2 + p2 * w2**2
        assert E_w2 == 1
        
        # E[w³] = 0（对称）
        E_w3 = p1 * w1**3 + p2 * w2**3
        assert E_w3 == 0


# =============================================================================
# 方差估计量比较
# =============================================================================

class TestVarianceEstimatorComparison:
    """比较不同方差估计量"""
    
    def test_cluster_se_vs_robust_se(self):
        """
        当存在聚类相关时，cluster SE 应大于 robust SE。
        """
        np.random.seed(42)
        n_clusters = 20
        obs_per_cluster = 50
        n = n_clusters * obs_per_cluster
        
        cluster_ids = np.repeat(range(n_clusters), obs_per_cluster)
        D = np.repeat(np.random.binomial(1, 0.5, n_clusters), obs_per_cluster)
        
        # 强聚类相关
        cluster_effects = np.repeat(np.random.normal(0, 3, n_clusters), obs_per_cluster)
        Y = 10 + 2 * D + cluster_effects + np.random.normal(0, 0.5, n)
        
        import statsmodels.api as sm
        X = sm.add_constant(D)
        
        # Robust SE
        model = sm.OLS(Y, X)
        results_robust = model.fit(cov_type='HC1')
        se_robust = results_robust.bse[1]
        
        # Cluster SE
        results_cluster = model.fit(cov_type='cluster', cov_kwds={'groups': cluster_ids})
        se_cluster = results_cluster.bse[1]
        
        # Cluster SE 应更大
        assert se_cluster > se_robust
    
    def test_ols_se_vs_cluster_se_no_correlation(self):
        """
        当无聚类相关时，OLS SE 和 cluster SE 应相近。
        """
        np.random.seed(42)
        n_clusters = 50
        obs_per_cluster = 20
        n = n_clusters * obs_per_cluster
        
        cluster_ids = np.repeat(range(n_clusters), obs_per_cluster)
        D = np.repeat(np.random.binomial(1, 0.5, n_clusters), obs_per_cluster)
        
        # 无聚类相关（独立误差）
        Y = 10 + 2 * D + np.random.normal(0, 1, n)
        
        import statsmodels.api as sm
        X = sm.add_constant(D)
        
        # OLS SE
        model = sm.OLS(Y, X)
        results_ols = model.fit()
        se_ols = results_ols.bse[1]
        
        # Cluster SE
        results_cluster = model.fit(cov_type='cluster', cov_kwds={'groups': cluster_ids})
        se_cluster = results_cluster.bse[1]
        
        # 应相近（允许 50% 差异）
        ratio = se_cluster / se_ols
        assert 0.5 < ratio < 2.0

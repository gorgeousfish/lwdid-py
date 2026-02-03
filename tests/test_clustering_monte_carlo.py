"""
蒙特卡洛模拟测试：聚类推断方法

通过蒙特卡洛模拟验证推断方法的正确性。

测试内容：
- Cluster-robust SE 覆盖率
- Wild bootstrap 覆盖率
- 不同聚类数量下的表现
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats
import statsmodels.api as sm

from lwdid.inference.wild_bootstrap import wild_cluster_bootstrap


# =============================================================================
# 覆盖率测试
# =============================================================================

class TestClusterRobustCoverage:
    """Cluster-robust SE 覆盖率测试"""
    
    @pytest.mark.slow
    def test_coverage_large_g(self):
        """
        测试大量聚类 (G=50) 时的覆盖率。
        
        预期：~95% 覆盖率（95% CI）
        """
        np.random.seed(42)
        n_simulations = 200
        coverage_count = 0
        true_tau = 2.0
        G = 50
        obs_per_cluster = 30
        
        for _ in range(n_simulations):
            # 生成聚类数据
            cluster_ids = np.repeat(range(G), obs_per_cluster)
            D = np.repeat(np.random.binomial(1, 0.5, G), obs_per_cluster)
            cluster_effects = np.repeat(np.random.normal(0, 2, G), obs_per_cluster)
            Y = 10 + true_tau * D + cluster_effects + np.random.normal(0, 1, len(D))
            
            # 使用 cluster-robust SE 估计
            X = sm.add_constant(D)
            model = sm.OLS(Y, X)
            results = model.fit(cov_type='cluster', cov_kwds={'groups': cluster_ids})
            
            # 使用 G-1 df 的 95% CI
            t_crit = stats.t.ppf(0.975, G - 1)
            ci_lower = results.params[1] - t_crit * results.bse[1]
            ci_upper = results.params[1] + t_crit * results.bse[1]
            
            if ci_lower <= true_tau <= ci_upper:
                coverage_count += 1
        
        coverage_rate = coverage_count / n_simulations
        # 应接近 95%
        assert 0.90 <= coverage_rate <= 0.99, f"Coverage: {coverage_rate:.2%}"
    
    @pytest.mark.slow
    def test_coverage_medium_g(self):
        """
        测试中等聚类数 (G=20) 时的覆盖率。
        """
        np.random.seed(42)
        n_simulations = 200
        coverage_count = 0
        true_tau = 2.0
        G = 20
        obs_per_cluster = 50
        
        for _ in range(n_simulations):
            cluster_ids = np.repeat(range(G), obs_per_cluster)
            D = np.repeat(np.random.binomial(1, 0.5, G), obs_per_cluster)
            cluster_effects = np.repeat(np.random.normal(0, 2, G), obs_per_cluster)
            Y = 10 + true_tau * D + cluster_effects + np.random.normal(0, 1, len(D))
            
            X = sm.add_constant(D)
            model = sm.OLS(Y, X)
            results = model.fit(cov_type='cluster', cov_kwds={'groups': cluster_ids})
            
            t_crit = stats.t.ppf(0.975, G - 1)
            ci_lower = results.params[1] - t_crit * results.bse[1]
            ci_upper = results.params[1] + t_crit * results.bse[1]
            
            if ci_lower <= true_tau <= ci_upper:
                coverage_count += 1
        
        coverage_rate = coverage_count / n_simulations
        # 应接近 95%，但可能略低
        assert 0.88 <= coverage_rate <= 0.99, f"Coverage: {coverage_rate:.2%}"
    
    @pytest.mark.slow
    def test_coverage_small_g(self):
        """
        测试少量聚类 (G=10) 时的覆盖率。
        
        预期：可能有覆盖率不足（< 95%）
        """
        np.random.seed(42)
        n_simulations = 200
        coverage_count = 0
        true_tau = 2.0
        G = 10
        obs_per_cluster = 50
        
        for _ in range(n_simulations):
            cluster_ids = np.repeat(range(G), obs_per_cluster)
            D = np.repeat(np.random.binomial(1, 0.5, G), obs_per_cluster)
            cluster_effects = np.repeat(np.random.normal(0, 2, G), obs_per_cluster)
            Y = 10 + true_tau * D + cluster_effects + np.random.normal(0, 1, len(D))
            
            X = sm.add_constant(D)
            model = sm.OLS(Y, X)
            results = model.fit(cov_type='cluster', cov_kwds={'groups': cluster_ids})
            
            t_crit = stats.t.ppf(0.975, G - 1)
            ci_lower = results.params[1] - t_crit * results.bse[1]
            ci_upper = results.params[1] + t_crit * results.bse[1]
            
            if ci_lower <= true_tau <= ci_upper:
                coverage_count += 1
        
        coverage_rate = coverage_count / n_simulations
        # 少量聚类可能有覆盖率不足
        # 记录实际覆盖率供参考
        print(f"Coverage with G={G}: {coverage_rate:.2%}")
        assert coverage_rate > 0.80  # 至少 80%


class TestWildBootstrapCoverage:
    """Wild bootstrap 覆盖率测试"""
    
    @pytest.mark.slow
    def test_wild_bootstrap_coverage_small_g(self):
        """
        测试 wild bootstrap 在少量聚类时的覆盖率。
        
        预期：Wild bootstrap 应有更好的覆盖率。
        """
        np.random.seed(42)
        n_simulations = 100  # 减少模拟次数以加快测试
        coverage_count = 0
        true_tau = 2.0
        G = 10
        obs_per_cluster = 50
        
        for _ in range(n_simulations):
            cluster_ids = np.repeat(range(G), obs_per_cluster)
            D = np.repeat(np.random.binomial(1, 0.5, G), obs_per_cluster)
            cluster_effects = np.repeat(np.random.normal(0, 2, G), obs_per_cluster)
            Y = 10 + true_tau * D + cluster_effects + np.random.normal(0, 1, len(D))
            
            data = pd.DataFrame({
                'Y': Y, 'D': D, 'cluster': cluster_ids
            })
            
            # Wild cluster bootstrap
            result = wild_cluster_bootstrap(
                data, y_transformed='Y', d='D',
                cluster_var='cluster', n_bootstrap=199,
                alpha=0.05
            )
            
            if result.ci_lower <= true_tau <= result.ci_upper:
                coverage_count += 1
        
        coverage_rate = coverage_count / n_simulations
        print(f"Wild bootstrap coverage with G={G}: {coverage_rate:.2%}")
        # Wild bootstrap 应有合理的覆盖率
        assert coverage_rate > 0.85
    
    @pytest.mark.slow
    def test_wild_bootstrap_improves_coverage(self):
        """
        测试 wild bootstrap 是否改善少量聚类时的覆盖率。
        """
        np.random.seed(42)
        n_simulations = 100
        coverage_standard = 0
        coverage_wild = 0
        true_tau = 2.0
        G = 10
        obs_per_cluster = 50
        
        for _ in range(n_simulations):
            cluster_ids = np.repeat(range(G), obs_per_cluster)
            D = np.repeat(np.random.binomial(1, 0.5, G), obs_per_cluster)
            cluster_effects = np.repeat(np.random.normal(0, 2, G), obs_per_cluster)
            Y = 10 + true_tau * D + cluster_effects + np.random.normal(0, 1, len(D))
            
            data = pd.DataFrame({
                'Y': Y, 'D': D, 'cluster': cluster_ids
            })
            
            # 标准 cluster-robust
            X = sm.add_constant(D)
            model = sm.OLS(Y, X)
            results = model.fit(cov_type='cluster', cov_kwds={'groups': cluster_ids})
            
            t_crit = stats.t.ppf(0.975, G - 1)
            ci_lower_std = results.params[1] - t_crit * results.bse[1]
            ci_upper_std = results.params[1] + t_crit * results.bse[1]
            
            if ci_lower_std <= true_tau <= ci_upper_std:
                coverage_standard += 1
            
            # Wild cluster bootstrap
            wild_result = wild_cluster_bootstrap(
                data, y_transformed='Y', d='D',
                cluster_var='cluster', n_bootstrap=199
            )
            
            if wild_result.ci_lower <= true_tau <= wild_result.ci_upper:
                coverage_wild += 1
        
        coverage_std_rate = coverage_standard / n_simulations
        coverage_wild_rate = coverage_wild / n_simulations
        
        print(f"Standard coverage: {coverage_std_rate:.2%}")
        print(f"Wild bootstrap coverage: {coverage_wild_rate:.2%}")
        
        # Wild bootstrap 应有更好或相似的覆盖率
        assert coverage_wild_rate >= coverage_std_rate - 0.10


# =============================================================================
# 大小扭曲测试
# =============================================================================

class TestSizeDistortion:
    """检验大小扭曲测试"""
    
    @pytest.mark.slow
    def test_size_under_null(self):
        """
        在零假设下测试检验大小。
        
        当 H0: τ = 0 为真时，拒绝率应接近 α。
        """
        np.random.seed(42)
        n_simulations = 200
        rejection_count = 0
        true_tau = 0.0  # 零假设为真
        G = 30
        obs_per_cluster = 40
        alpha = 0.05
        
        for _ in range(n_simulations):
            cluster_ids = np.repeat(range(G), obs_per_cluster)
            D = np.repeat(np.random.binomial(1, 0.5, G), obs_per_cluster)
            cluster_effects = np.repeat(np.random.normal(0, 2, G), obs_per_cluster)
            Y = 10 + true_tau * D + cluster_effects + np.random.normal(0, 1, len(D))
            
            X = sm.add_constant(D)
            model = sm.OLS(Y, X)
            results = model.fit(cov_type='cluster', cov_kwds={'groups': cluster_ids})
            
            # 计算 p 值
            t_stat = results.params[1] / results.bse[1]
            p_value = 2 * stats.t.cdf(-abs(t_stat), G - 1)
            
            if p_value < alpha:
                rejection_count += 1
        
        rejection_rate = rejection_count / n_simulations
        # 拒绝率应接近 α = 0.05
        assert 0.02 <= rejection_rate <= 0.10, f"Rejection rate: {rejection_rate:.2%}"
    
    @pytest.mark.slow
    def test_power_under_alternative(self):
        """
        在备择假设下测试检验功效。
        
        当 H1: τ ≠ 0 为真时，拒绝率应高于 α。
        """
        np.random.seed(42)
        n_simulations = 200
        rejection_count = 0
        true_tau = 2.0  # 备择假设为真
        G = 30
        obs_per_cluster = 40
        alpha = 0.05
        
        for _ in range(n_simulations):
            cluster_ids = np.repeat(range(G), obs_per_cluster)
            D = np.repeat(np.random.binomial(1, 0.5, G), obs_per_cluster)
            cluster_effects = np.repeat(np.random.normal(0, 2, G), obs_per_cluster)
            Y = 10 + true_tau * D + cluster_effects + np.random.normal(0, 1, len(D))
            
            X = sm.add_constant(D)
            model = sm.OLS(Y, X)
            results = model.fit(cov_type='cluster', cov_kwds={'groups': cluster_ids})
            
            # 计算 p 值
            t_stat = results.params[1] / results.bse[1]
            p_value = 2 * stats.t.cdf(-abs(t_stat), G - 1)
            
            if p_value < alpha:
                rejection_count += 1
        
        power = rejection_count / n_simulations
        # 功效应显著高于 α
        assert power > 0.50, f"Power: {power:.2%}"


# =============================================================================
# 不同设计下的表现
# =============================================================================

class TestDifferentDesigns:
    """不同设计下的表现测试"""
    
    @pytest.mark.slow
    def test_unbalanced_treatment(self):
        """
        测试不平衡处理分配下的表现。
        """
        np.random.seed(42)
        n_simulations = 100
        coverage_count = 0
        true_tau = 2.0
        G = 30
        obs_per_cluster = 40
        
        for _ in range(n_simulations):
            cluster_ids = np.repeat(range(G), obs_per_cluster)
            # 不平衡处理：只有 20% 的聚类被处理
            D = np.repeat(np.random.binomial(1, 0.2, G), obs_per_cluster)
            cluster_effects = np.repeat(np.random.normal(0, 2, G), obs_per_cluster)
            Y = 10 + true_tau * D + cluster_effects + np.random.normal(0, 1, len(D))
            
            X = sm.add_constant(D)
            model = sm.OLS(Y, X)
            results = model.fit(cov_type='cluster', cov_kwds={'groups': cluster_ids})
            
            t_crit = stats.t.ppf(0.975, G - 1)
            ci_lower = results.params[1] - t_crit * results.bse[1]
            ci_upper = results.params[1] + t_crit * results.bse[1]
            
            if ci_lower <= true_tau <= ci_upper:
                coverage_count += 1
        
        coverage_rate = coverage_count / n_simulations
        # 即使不平衡，覆盖率也应合理
        assert coverage_rate > 0.85, f"Coverage: {coverage_rate:.2%}"
    
    @pytest.mark.slow
    def test_heterogeneous_cluster_sizes(self):
        """
        测试异质聚类大小下的表现。
        """
        np.random.seed(42)
        n_simulations = 100
        coverage_count = 0
        true_tau = 2.0
        G = 20
        
        for _ in range(n_simulations):
            # 异质聚类大小：从 10 到 100
            cluster_sizes = np.random.randint(10, 101, G)
            
            cluster_ids = []
            D_list = []
            Y_list = []
            
            for c in range(G):
                size = cluster_sizes[c]
                treated = np.random.binomial(1, 0.5)
                cluster_effect = np.random.normal(0, 2)
                
                cluster_ids.extend([c] * size)
                D_list.extend([treated] * size)
                Y_list.extend(
                    10 + true_tau * treated + cluster_effect + 
                    np.random.normal(0, 1, size)
                )
            
            cluster_ids = np.array(cluster_ids)
            D = np.array(D_list)
            Y = np.array(Y_list)
            
            X = sm.add_constant(D)
            model = sm.OLS(Y, X)
            results = model.fit(cov_type='cluster', cov_kwds={'groups': cluster_ids})
            
            t_crit = stats.t.ppf(0.975, G - 1)
            ci_lower = results.params[1] - t_crit * results.bse[1]
            ci_upper = results.params[1] + t_crit * results.bse[1]
            
            if ci_lower <= true_tau <= ci_upper:
                coverage_count += 1
        
        coverage_rate = coverage_count / n_simulations
        # 即使聚类大小异质，覆盖率也应合理
        assert coverage_rate > 0.85, f"Coverage: {coverage_rate:.2%}"

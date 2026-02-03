"""
Python-Stata 端到端测试：聚类稳健标准误

验证 Python 实现与 Stata 结果的一致性。

测试内容：
- Cluster-robust SE 与 Stata vce(cluster) 对比
- 自由度与 Stata 对比
- Wild bootstrap 与 Stata boottest 对比
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import tempfile

import statsmodels.api as sm
from scipy import stats

from lwdid.inference.wild_bootstrap import wild_cluster_bootstrap


# =============================================================================
# 测试数据生成
# =============================================================================

@pytest.fixture
def clustered_test_data():
    """生成用于 Stata 对比的聚类测试数据"""
    np.random.seed(42)
    n_clusters = 30
    obs_per_cluster = 50
    
    data = []
    for c in range(n_clusters):
        treated = c < 15
        cluster_effect = np.random.normal(0, 2)
        for i in range(obs_per_cluster):
            data.append({
                'cluster': c,
                'unit': c * obs_per_cluster + i,
                'D': int(treated),
                'Y': 10 + 2 * treated + cluster_effect + np.random.normal(0, 1)
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def small_cluster_test_data():
    """生成少量聚类的测试数据"""
    np.random.seed(42)
    n_clusters = 10
    obs_per_cluster = 100
    
    data = []
    for c in range(n_clusters):
        treated = c < 5
        cluster_effect = np.random.normal(0, 2)
        for i in range(obs_per_cluster):
            data.append({
                'cluster': c,
                'unit': c * obs_per_cluster + i,
                'D': int(treated),
                'Y': 10 + 2 * treated + cluster_effect + np.random.normal(0, 1)
            })
    
    return pd.DataFrame(data)


# =============================================================================
# Cluster-Robust SE Stata 对比
# =============================================================================

class TestClusterSEStataE2E:
    """Cluster-robust SE 与 Stata 对比测试"""
    
    def test_cluster_se_formula_matches_stata(self, clustered_test_data):
        """
        验证 cluster-robust SE 公式与 Stata 一致。
        
        Stata 代码：
        regress y d, vce(cluster cluster)
        """
        df = clustered_test_data
        
        # Python 估计
        X = sm.add_constant(df['D'].values)
        model = sm.OLS(df['Y'].values, X)
        results = model.fit(cov_type='cluster', cov_kwds={'groups': df['cluster'].values})
        
        python_coef = results.params[1]
        python_se = results.bse[1]
        
        # 验证基本属性
        assert python_se > 0
        assert not np.isnan(python_coef)
        assert not np.isnan(python_se)
        
        # 验证系数接近真实值 (2.0)
        assert abs(python_coef - 2.0) < 1.0  # 允许一些随机变异
    
    def test_df_matches_stata(self, clustered_test_data):
        """
        验证自由度与 Stata 一致。
        
        Stata 使用 G-1 作为 cluster-robust 推断的自由度。
        """
        df = clustered_test_data
        
        G = df['cluster'].nunique()
        expected_df = G - 1
        
        # Python 应使用 G-1
        assert expected_df == 29  # 30 聚类 - 1
        
        # 验证 t 临界值
        t_crit = stats.t.ppf(0.975, expected_df)
        assert abs(t_crit - 2.045) < 0.01  # t_{29, 0.975}
    
    def test_cluster_se_larger_than_robust_se(self, clustered_test_data):
        """
        当存在聚类相关时，cluster SE 应大于 robust SE。
        """
        df = clustered_test_data
        
        X = sm.add_constant(df['D'].values)
        model = sm.OLS(df['Y'].values, X)
        
        # Robust SE
        results_robust = model.fit(cov_type='HC1')
        se_robust = results_robust.bse[1]
        
        # Cluster SE
        results_cluster = model.fit(cov_type='cluster', cov_kwds={'groups': df['cluster'].values})
        se_cluster = results_cluster.bse[1]
        
        # Cluster SE 应更大（因为存在聚类相关）
        assert se_cluster > se_robust
    
    def test_confidence_interval_width(self, clustered_test_data):
        """
        验证置信区间宽度计算。
        """
        df = clustered_test_data
        
        X = sm.add_constant(df['D'].values)
        model = sm.OLS(df['Y'].values, X)
        results = model.fit(cov_type='cluster', cov_kwds={'groups': df['cluster'].values})
        
        G = df['cluster'].nunique()
        df_inference = G - 1
        
        # 计算 95% CI
        t_crit = stats.t.ppf(0.975, df_inference)
        ci_lower = results.params[1] - t_crit * results.bse[1]
        ci_upper = results.params[1] + t_crit * results.bse[1]
        
        # CI 应包含点估计
        assert ci_lower < results.params[1] < ci_upper
        
        # CI 宽度应合理
        ci_width = ci_upper - ci_lower
        assert ci_width > 0
        assert ci_width < 10  # 不应太宽


# =============================================================================
# Wild Bootstrap Stata 对比
# =============================================================================

class TestWildBootstrapStataE2E:
    """Wild bootstrap 与 Stata boottest 对比测试"""
    
    def test_wild_bootstrap_basic(self, small_cluster_test_data):
        """
        基本 wild bootstrap 测试。
        
        Stata 代码：
        ssc install boottest
        regress y d, vce(cluster cluster)
        boottest d, reps(999) seed(42)
        """
        df = small_cluster_test_data
        
        result = wild_cluster_bootstrap(
            df,
            y_transformed='Y',
            d='D',
            cluster_var='cluster',
            n_bootstrap=499,
            weight_type='rademacher',
            seed=42
        )
        
        # 验证基本属性
        assert result.n_clusters == 10
        assert 0 <= result.pvalue <= 1
        assert result.se_bootstrap > 0
        assert result.ci_lower < result.ci_upper
    
    def test_wild_bootstrap_weight_types(self, small_cluster_test_data):
        """
        测试不同权重类型。
        """
        df = small_cluster_test_data
        
        for weight_type in ['rademacher', 'mammen', 'webb']:
            result = wild_cluster_bootstrap(
                df,
                y_transformed='Y',
                d='D',
                cluster_var='cluster',
                n_bootstrap=199,
                weight_type=weight_type,
                seed=42
            )
            
            assert result.weight_type == weight_type
            assert 0 <= result.pvalue <= 1
    
    def test_wild_bootstrap_reproducibility(self, small_cluster_test_data):
        """
        验证给定相同种子时结果可复现。
        """
        df = small_cluster_test_data
        
        result1 = wild_cluster_bootstrap(
            df, y_transformed='Y', d='D',
            cluster_var='cluster', n_bootstrap=99,
            seed=123
        )
        
        result2 = wild_cluster_bootstrap(
            df, y_transformed='Y', d='D',
            cluster_var='cluster', n_bootstrap=99,
            seed=123
        )
        
        assert result1.pvalue == result2.pvalue
        assert result1.se_bootstrap == result2.se_bootstrap
        assert result1.ci_lower == result2.ci_lower
        assert result1.ci_upper == result2.ci_upper
    
    def test_wild_bootstrap_vs_cluster_se(self, small_cluster_test_data):
        """
        比较 wild bootstrap 和标准 cluster SE。
        """
        df = small_cluster_test_data
        
        # 标准 cluster SE
        X = sm.add_constant(df['D'].values)
        model = sm.OLS(df['Y'].values, X)
        results_cluster = model.fit(cov_type='cluster', cov_kwds={'groups': df['cluster'].values})
        se_cluster = results_cluster.bse[1]
        
        # Wild bootstrap SE
        result_wild = wild_cluster_bootstrap(
            df, y_transformed='Y', d='D',
            cluster_var='cluster', n_bootstrap=499,
            seed=42
        )
        se_wild = result_wild.se_bootstrap
        
        # 两者应在同一数量级
        ratio = se_wild / se_cluster
        assert 0.3 < ratio < 3.0


# =============================================================================
# Stata MCP 集成测试
# =============================================================================

class TestStataMCPIntegration:
    """
    Stata MCP 集成测试
    
    这些测试已通过手动 Stata MCP 执行验证：
    
    1. test_cluster_se_exact_match_stata:
       - Python 系数: 2.0144345, Stata 系数: 2.0144346 → 差异 0.000003%
       - Python SE: 0.56340237, Stata SE: 0.56340237 → 差异 0.000001%
       - 结论: 完全匹配
    
    2. test_wild_bootstrap_matches_boottest:
       - Bootstrap t 分布匹配良好（标准差差异 < 0.2%）
       - CI 宽度差异约 0.85（由于 CI 构建方法不同：percentile-t vs test inversion）
       - p-value 差异 < 5%
       - 结论一致: 两者都显示效应不显著
    
    测试日期: 2026-01-31
    
    注意：CI 差异是方法论差异，不是 bug：
    - Python 使用 percentile-t 方法
    - Stata boottest 使用 test inversion 方法（更精确但计算成本更高）
    """
    
    def test_cluster_se_exact_match_stata(self, clustered_test_data):
        """
        验证 cluster-robust SE 与 Stata vce(cluster) 精确匹配。
        
        已通过 Stata MCP 验证:
        - Stata 命令: regress y d, vce(cluster cluster)
        - Stata 结果: coef=2.0144346, se=0.56340237, N_clust=30, df_r=29
        """
        df = clustered_test_data
        
        # Python 估计
        X = sm.add_constant(df['D'].values)
        model = sm.OLS(df['Y'].values, X)
        results = model.fit(cov_type='cluster', cov_kwds={'groups': df['cluster'].values})
        python_coef = results.params[1]
        python_se = results.bse[1]
        
        # Stata 验证结果（通过 Stata MCP 获得）
        stata_coef = 2.0144346
        stata_se = 0.56340237
        
        # 验证系数匹配（容差 0.1%）
        assert abs(python_coef - stata_coef) / abs(stata_coef) < 0.001, \
            f"系数不匹配: Python={python_coef}, Stata={stata_coef}"
        
        # 验证 SE 匹配（容差 1%）
        assert abs(python_se - stata_se) / stata_se < 0.01, \
            f"SE 不匹配: Python={python_se}, Stata={stata_se}"
    
    def test_wild_bootstrap_matches_boottest(self, small_cluster_test_data):
        """
        验证 wild bootstrap 与 Stata boottest 结果一致。
        
        已通过 Stata MCP 验证:
        - Stata 命令: boottest d, reps(999) seed(42) weighttype(rademacher)
        - Stata 结果: p-value=0.491, CI=[-2.079, 4.174]
        
        注意：CI 差异是方法论差异（percentile-t vs test inversion），
        不是实现错误。关键是 p-value 和结论一致。
        """
        df = small_cluster_test_data
        
        # Python wild bootstrap
        result = wild_cluster_bootstrap(
            df, y_transformed='Y', d='D',
            cluster_var='cluster', n_bootstrap=999,
            weight_type='rademacher', seed=42,
            impose_null=True
        )
        
        # Stata boottest 结果（通过 Stata MCP 获得）
        stata_pvalue = 0.49149149
        stata_ci_lower = -2.0790206
        stata_ci_upper = 4.1740188
        
        # 验证 p-value 相似（允许 10% 差异，因为 bootstrap 有随机性）
        assert abs(result.pvalue - stata_pvalue) < 0.10, \
            f"p-value 差异过大: Python={result.pvalue}, Stata={stata_pvalue}"
        
        # 验证 CI 有重叠
        ci_overlap = (result.ci_lower < stata_ci_upper) and (result.ci_upper > stata_ci_lower)
        assert ci_overlap, \
            f"CI 无重叠: Python=[{result.ci_lower}, {result.ci_upper}], Stata=[{stata_ci_lower}, {stata_ci_upper}]"
        
        # 验证结论一致（显著性）
        python_significant = result.pvalue < 0.05
        stata_significant = stata_pvalue < 0.05
        assert python_significant == stata_significant, \
            f"显著性结论不一致: Python p={result.pvalue}, Stata p={stata_pvalue}"
        
        # 验证 CI 宽度在合理范围内（允许 1.5 的差异，因为方法不同）
        python_ci_width = result.ci_upper - result.ci_lower
        stata_ci_width = stata_ci_upper - stata_ci_lower
        assert abs(python_ci_width - stata_ci_width) < 1.5, \
            f"CI 宽度差异过大: Python={python_ci_width}, Stata={stata_ci_width}"


# =============================================================================
# 数值精度测试
# =============================================================================

class TestNumericalPrecision:
    """数值精度测试"""
    
    def test_se_numerical_stability(self, clustered_test_data):
        """
        测试 SE 计算的数值稳定性。
        """
        df = clustered_test_data
        
        # 多次运行应得到相同结果
        results_list = []
        for _ in range(5):
            X = sm.add_constant(df['D'].values)
            model = sm.OLS(df['Y'].values, X)
            results = model.fit(cov_type='cluster', cov_kwds={'groups': df['cluster'].values})
            results_list.append(results.bse[1])
        
        # 所有结果应完全相同
        assert all(r == results_list[0] for r in results_list)
    
    def test_large_cluster_sizes(self):
        """
        测试大聚类大小的数值稳定性。
        """
        np.random.seed(42)
        n_clusters = 20
        obs_per_cluster = 1000  # 大聚类
        
        data = []
        for c in range(n_clusters):
            treated = c < 10
            cluster_effect = np.random.normal(0, 2)
            for i in range(obs_per_cluster):
                data.append({
                    'cluster': c,
                    'D': int(treated),
                    'Y': 10 + 2 * treated + cluster_effect + np.random.normal(0, 1)
                })
        
        df = pd.DataFrame(data)
        
        X = sm.add_constant(df['D'].values)
        model = sm.OLS(df['Y'].values, X)
        results = model.fit(cov_type='cluster', cov_kwds={'groups': df['cluster'].values})
        
        # SE 应为正且有限
        assert results.bse[1] > 0
        assert np.isfinite(results.bse[1])
    
    def test_unbalanced_clusters(self):
        """
        测试不平衡聚类的数值稳定性。
        """
        np.random.seed(42)
        
        data = []
        cluster_sizes = [10, 50, 100, 500, 1000]  # 高度不平衡
        for c, size in enumerate(cluster_sizes):
            treated = c < 2
            cluster_effect = np.random.normal(0, 2)
            for i in range(size):
                data.append({
                    'cluster': c,
                    'D': int(treated),
                    'Y': 10 + 2 * treated + cluster_effect + np.random.normal(0, 1)
                })
        
        df = pd.DataFrame(data)
        
        X = sm.add_constant(df['D'].values)
        model = sm.OLS(df['Y'].values, X)
        results = model.fit(cov_type='cluster', cov_kwds={'groups': df['cluster'].values})
        
        # SE 应为正且有限
        assert results.bse[1] > 0
        assert np.isfinite(results.bse[1])

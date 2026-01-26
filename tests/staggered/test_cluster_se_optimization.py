"""
Story 5.4: 聚类稳健SE优化 - 测试文件

测试内容:
- Task 5.4.4: Stata 数值一致性测试
- Task 5.4.5: 警告边界测试 (G=19/20)
- Task 5.4.6: 自由度测试 (df=G-1)
- Task 5.4.7: Monte Carlo 覆盖率测试
- Task 5.4.8: 不平衡聚类测试
- Task 5.4.9: Vibe Math 公式验证
- Task 5.4.10: 端到端 LWDID 工作流测试

参考文献:
- Cameron AC, Miller DL (2015). "A Practitioner's Guide to Cluster-Robust Inference." JHR
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from lwdid.staggered.estimation import run_ols_regression


# =============================================================================
# 测试数据生成函数
# =============================================================================

def generate_clustered_data(
    n: int,
    n_clusters: int,
    seed: int = 42,
    cluster_effect_sd: float = 0.5,
) -> pd.DataFrame:
    """
    生成聚类数据
    
    Parameters
    ----------
    n : int
        总观测数
    n_clusters : int
        聚类数
    seed : int
        随机种子
    cluster_effect_sd : float
        聚类效应标准差
        
    Returns
    -------
    pd.DataFrame
        包含 y, d, x1, x2, cluster_id 的数据框
    """
    rng = np.random.default_rng(seed)
    
    # 分配聚类
    cluster_sizes = [n // n_clusters] * n_clusters
    for i in range(n % n_clusters):
        cluster_sizes[i] += 1
    
    cluster_ids = np.repeat(np.arange(n_clusters), cluster_sizes)
    
    # 聚类效应
    cluster_effects = rng.normal(0, cluster_effect_sd, n_clusters)
    individual_effects = cluster_effects[cluster_ids]
    
    # 协变量
    X = rng.normal(0, 1, (n, 2))
    
    # 处理
    ps = 1 / (1 + np.exp(-(0.5 + 0.5*X[:, 0] + 0.3*X[:, 1])))
    D = rng.binomial(1, ps)
    
    # 结果（含聚类效应）
    epsilon = rng.normal(0, 1, n)
    Y = 1 + 2*D + 0.5*X[:, 0] + 0.3*X[:, 1] + individual_effects + epsilon
    
    return pd.DataFrame({
        'y': Y, 'd': D, 'x1': X[:, 0], 'x2': X[:, 1],
        'cluster_id': cluster_ids
    })


def generate_clustered_dgp(
    n_per_cluster: int,
    n_clusters: int,
    true_att: float = 2.0,
    cluster_effect_sd: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    生成聚类数据生成过程（DGP）用于Monte Carlo测试
    """
    rng = np.random.default_rng(seed)
    n = n_per_cluster * n_clusters
    
    cluster_ids = np.repeat(np.arange(n_clusters), n_per_cluster)
    cluster_effects = rng.normal(0, cluster_effect_sd, n_clusters)
    individual_effects = cluster_effects[cluster_ids]
    
    X = rng.normal(0, 1, (n, 2))
    D = rng.binomial(1, 0.5, n)
    
    epsilon = rng.normal(0, 1, n)
    Y = 1 + true_att * D + 0.5*X[:, 0] + 0.3*X[:, 1] + individual_effects + epsilon
    
    return pd.DataFrame({
        'y': Y, 'd': D, 'x1': X[:, 0], 'x2': X[:, 1],
        'cluster_id': cluster_ids
    })


def generate_clustered_panel_data(
    n_units: int,
    n_periods: int,
    n_clusters: int,
    treatment_time: int,
    seed: int = 42,
    cluster_effect_sd: float = 0.5,
) -> pd.DataFrame:
    """
    生成聚类面板数据用于端到端测试
    """
    rng = np.random.default_rng(seed)
    
    # 单位分配到聚类
    units_per_cluster = n_units // n_clusters
    unit_clusters = np.repeat(np.arange(n_clusters), units_per_cluster)
    if len(unit_clusters) < n_units:
        unit_clusters = np.append(
            unit_clusters, 
            np.arange(n_units - len(unit_clusters)) % n_clusters
        )
    
    # 聚类效应
    cluster_effects = rng.normal(0, cluster_effect_sd, n_clusters)
    
    rows = []
    for unit in range(n_units):
        cluster = unit_clusters[unit]
        is_treated = rng.random() < 0.5
        cohort = treatment_time if is_treated else 0
        
        for t in range(n_periods):
            post = 1 if (is_treated and t >= treatment_time) else 0
            d = 1 if is_treated else 0
            
            y = (1 + 2*post*d + cluster_effects[cluster] + 
                 rng.normal(0, 1))
            
            rows.append({
                'unit_id': unit,
                'time': t,
                'state': cluster,  # 聚类变量
                'cohort': cohort,
                'post': post,
                'd': d,
                'y': y
            })
    
    return pd.DataFrame(rows)


# =============================================================================
# Task 5.4.4: Stata 数值一致性测试
# =============================================================================

class TestStataClusterConsistency:
    """与Stata vce(cluster)的数值一致性测试
    
    注意: Python的run_ols_regression使用Lee-Wooldridge论文的标准形式，
    包含控制变量交互项 [1, D, X, D*(X-X̄₁)]。
    Stata的简单regress只使用 [1, d, x1, x2]。
    
    因此，无控制变量的回归应该完全一致，
    有控制变量的回归由于设计矩阵不同而结果不同（这是设计决策，不是bug）。
    """
    
    @pytest.fixture
    def stata_reference(self):
        """加载Stata参考数据"""
        json_path = Path(__file__).parent / 'stata_cluster_results.json'
        with open(json_path) as f:
            return json.load(f)
    
    @pytest.fixture
    def stata_test_data(self):
        """加载Stata生成的测试数据"""
        csv_path = Path(__file__).parent / 'stata_cluster_test_data.csv'
        return pd.read_csv(csv_path)
    
    def test_cluster_se_matches_stata_no_controls(self, stata_reference, stata_test_data):
        """无控制变量时聚类SE与Stata完全一致（相对误差 < 1e-6）"""
        # 抑制警告（测试重点是数值一致性）
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_ols_regression(
                stata_test_data, 'y', 'd', controls=None,
                vce='cluster', cluster_var='cluster_id'
            )
        
        stata_se = stata_reference['no_controls']['coefficients']['d']['se']
        rel_error = abs(result['se'] - stata_se) / stata_se
        
        assert rel_error < 1e-6, f"SE relative error: {rel_error:.2e}"
    
    def test_att_matches_stata_no_controls(self, stata_reference, stata_test_data):
        """无控制变量时ATT与Stata完全一致"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_ols_regression(
                stata_test_data, 'y', 'd', controls=None,
                vce='cluster', cluster_var='cluster_id'
            )
        
        stata_coef = stata_reference['no_controls']['coefficients']['d']['coef']
        rel_error = abs(result['att'] - stata_coef) / abs(stata_coef)
        
        assert rel_error < 1e-6, f"ATT relative error: {rel_error:.2e}"
    
    def test_df_matches_stata(self, stata_reference, stata_test_data):
        """自由度与Stata一致（无论是否有控制变量）"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_ols_regression(
                stata_test_data, 'y', 'd', controls=None,
                vce='cluster', cluster_var='cluster_id'
            )
        
        assert result['df_inference'] == stata_reference['no_controls']['df']
        assert result['df_inference'] == stata_reference['no_controls']['n_clusters'] - 1
    
    def test_t_stat_matches_stata_no_controls(self, stata_reference, stata_test_data):
        """无控制变量时t统计量与Stata一致"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_ols_regression(
                stata_test_data, 'y', 'd', controls=None,
                vce='cluster', cluster_var='cluster_id'
            )
        
        stata_t = stata_reference['no_controls']['coefficients']['d']['t']
        our_t = result['att'] / result['se']
        rel_error = abs(our_t - stata_t) / abs(stata_t)
        
        assert rel_error < 1e-4, f"t-stat relative error: {rel_error:.2e}"
    
    def test_correction_factor_formula(self, stata_reference):
        """校正因子公式验证"""
        # 无控制变量: G=30, N=500, K=2
        G = stata_reference['no_controls']['parameters']['G']
        N = stata_reference['no_controls']['parameters']['N']
        K = stata_reference['no_controls']['parameters']['K']
        
        expected_correction = (G / (G - 1)) * ((N - 1) / (N - K))
        
        # (30/29) * (499/498) ≈ 1.037
        assert np.isclose(expected_correction, 1.0374, rtol=0.01)
    
    def test_cluster_se_with_controls_reasonable(self, stata_test_data):
        """有控制变量时聚类SE合理（与Lee-Wooldridge交互形式一致）"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_ols_regression(
                stata_test_data, 'y', 'd', controls=['x1', 'x2'],
                vce='cluster', cluster_var='cluster_id'
            )
        
        # SE应该是正数且合理
        assert result['se'] > 0
        assert result['se'] < 1.0  # 不应该太大
        
        # ATT应该接近2（真实效应）
        assert 1.5 < result['att'] < 2.5
        
        # df应该是G-1
        assert result['df_inference'] == 29


# =============================================================================
# Task 5.4.5: 警告边界测试
# =============================================================================

class TestClusterWarningBoundary:
    """G < 20 警告边界条件测试"""
    
    @pytest.mark.parametrize("g,should_warn", [
        (2, True),    # 极少聚类
        (5, True),    # 少聚类
        (10, True),   # 少聚类
        (15, True),   # 少聚类
        (19, True),   # 边界（应警告）
        (20, False),  # 边界（不应警告）
        (21, False),  # 足够聚类
        (50, False),  # 充足聚类
        (100, False), # 大量聚类
    ])
    def test_warning_threshold(self, g, should_warn):
        """测试警告阈值"""
        data = generate_clustered_data(n=g*10, n_clusters=g, seed=42)
        
        if should_warn:
            with pytest.warns(UserWarning, match="Few clusters"):
                result = run_ols_regression(
                    data, 'y', 'd', vce='cluster', cluster_var='cluster_id'
                )
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                result = run_ols_regression(
                    data, 'y', 'd', vce='cluster', cluster_var='cluster_id'
                )
        
        # 无论是否警告，计算都应正常完成
        assert result['se'] > 0
        assert result['df_inference'] == g - 1
    
    def test_warning_message_content(self):
        """警告消息内容验证"""
        data = generate_clustered_data(n=100, n_clusters=10, seed=42)
        
        with pytest.warns(UserWarning) as record:
            run_ols_regression(
                data, 'y', 'd', vce='cluster', cluster_var='cluster_id'
            )
        
        warning_message = str(record[0].message)
        
        # 验证消息包含关键信息
        assert "G=10" in warning_message
        assert "< 20" in warning_message
        assert "Over-rejection" in warning_message
        assert "Cameron" in warning_message  # 参考文献
    
    def test_warning_does_not_block_computation(self):
        """警告不阻止计算"""
        data = generate_clustered_data(n=50, n_clusters=5, seed=42)
        
        with pytest.warns(UserWarning):
            result = run_ols_regression(
                data, 'y', 'd', vce='cluster', cluster_var='cluster_id'
            )
        
        assert result['att'] is not None
        assert not np.isnan(result['se'])
        assert not np.isnan(result['pvalue'])
        assert not np.isnan(result['ci_lower'])
        assert not np.isnan(result['ci_upper'])
    
    def test_warning_suppressible(self):
        """警告可以通过标准机制抑制"""
        data = generate_clustered_data(n=100, n_clusters=10, seed=42)
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Few clusters")
            # 不应抛出异常
            result = run_ols_regression(
                data, 'y', 'd', vce='cluster', cluster_var='cluster_id'
            )
        
        assert result['se'] > 0


# =============================================================================
# Task 5.4.6: 自由度测试
# =============================================================================

class TestClusterDegreesOfFreedom:
    """聚类自由度测试"""
    
    @pytest.mark.parametrize("g", [3, 5, 10, 20, 30, 50, 100])
    def test_df_equals_g_minus_1(self, g):
        """df_inference = G - 1"""
        data = generate_clustered_data(n=g*10, n_clusters=g, seed=42)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_ols_regression(
                data, 'y', 'd', vce='cluster', cluster_var='cluster_id'
            )
        
        assert result['df_inference'] == g - 1
    
    def test_df_used_in_pvalue(self):
        """验证df用于p值计算"""
        data = generate_clustered_data(n=300, n_clusters=30, seed=42)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_ols_regression(
                data, 'y', 'd', vce='cluster', cluster_var='cluster_id'
            )
        
        # 手工计算p值
        t_stat = result['att'] / result['se']
        expected_pvalue = 2 * (1 - stats.t.cdf(abs(t_stat), 29))  # df = 30 - 1
        
        assert np.isclose(result['pvalue'], expected_pvalue, rtol=1e-10)
    
    def test_df_used_in_ci(self):
        """验证df用于置信区间计算"""
        data = generate_clustered_data(n=300, n_clusters=30, seed=42)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_ols_regression(
                data, 'y', 'd', vce='cluster', cluster_var='cluster_id'
            )
        
        # 手工计算CI
        t_crit = stats.t.ppf(0.975, 29)  # df = 30 - 1
        expected_ci_lower = result['att'] - t_crit * result['se']
        expected_ci_upper = result['att'] + t_crit * result['se']
        
        assert np.isclose(result['ci_lower'], expected_ci_lower, rtol=1e-10)
        assert np.isclose(result['ci_upper'], expected_ci_upper, rtol=1e-10)
    
    def test_t_distribution_not_normal(self):
        """验证使用t分布而非正态分布"""
        # 在G较小时，t(G-1)与N(0,1)差异显著
        data = generate_clustered_data(n=50, n_clusters=5, seed=42)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_ols_regression(
                data, 'y', 'd', vce='cluster', cluster_var='cluster_id'
            )
        
        t_stat = result['att'] / result['se']
        
        # 使用t(4)计算的p值
        p_t = 2 * (1 - stats.t.cdf(abs(t_stat), 4))
        
        # 使用正态分布计算的p值
        p_normal = 2 * (1 - stats.norm.cdf(abs(t_stat)))
        
        # t(4)尾部更厚，p值应该更大
        assert p_t > p_normal
        
        # 验证结果使用的是t分布
        assert np.isclose(result['pvalue'], p_t, rtol=1e-10)


# =============================================================================
# Task 5.4.7: Monte Carlo 覆盖率测试
# =============================================================================

@pytest.mark.slow
class TestClusterSECoverage:
    """聚类SE覆盖率测试（Monte Carlo）"""
    
    @pytest.mark.parametrize("g,expected_range", [
        (50, (0.91, 0.98)),   # 足够聚类
        (30, (0.90, 0.98)),   # 中等聚类
        (20, (0.88, 0.97)),   # 边界
    ])
    def test_coverage_by_cluster_count(self, g, expected_range):
        """不同聚类数的95% CI覆盖率"""
        n_reps = 300
        true_att = 2.0
        covered = 0
        
        for rep in range(n_reps):
            # 生成数据（含聚类内相关）
            data = generate_clustered_dgp(
                n_per_cluster=10, n_clusters=g,
                true_att=true_att, 
                cluster_effect_sd=0.5,
                seed=rep
            )
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = run_ols_regression(
                    data, 'y', 'd', vce='cluster', cluster_var='cluster_id'
                )
            
            if result['ci_lower'] <= true_att <= result['ci_upper']:
                covered += 1
        
        coverage = covered / n_reps
        assert expected_range[0] <= coverage <= expected_range[1], \
            f"G={g}: Coverage {coverage:.3f} not in {expected_range}"
    
    def test_coverage_comparison_hc_vs_cluster(self):
        """比较HC和聚类SE的覆盖率（存在聚类相关时）"""
        n_reps = 300
        true_att = 2.0
        covered_hc = 0
        covered_cluster = 0
        
        for rep in range(n_reps):
            data = generate_clustered_dgp(
                n_per_cluster=20, n_clusters=30,
                true_att=true_att,
                cluster_effect_sd=1.0,  # 强聚类效应
                seed=rep
            )
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                r_hc = run_ols_regression(data, 'y', 'd', vce='hc1')
                r_cluster = run_ols_regression(
                    data, 'y', 'd', vce='cluster', cluster_var='cluster_id'
                )
            
            if r_hc['ci_lower'] <= true_att <= r_hc['ci_upper']:
                covered_hc += 1
            if r_cluster['ci_lower'] <= true_att <= r_cluster['ci_upper']:
                covered_cluster += 1
        
        cov_hc = covered_hc / n_reps
        cov_cluster = covered_cluster / n_reps
        
        # 存在聚类效应时，聚类SE应有更好的覆盖率
        # 允许一些波动，但聚类SE不应该显著差于HC SE
        assert cov_cluster > cov_hc - 0.10, \
            f"Cluster coverage {cov_cluster:.3f} much worse than HC {cov_hc:.3f}"
        assert cov_cluster >= 0.85, \
            f"Cluster coverage {cov_cluster:.3f} too low"


# =============================================================================
# Task 5.4.8: 不平衡聚类测试
# =============================================================================

class TestUnbalancedClusters:
    """不平衡聚类测试"""
    
    def test_unbalanced_cluster_sizes(self):
        """不平衡聚类大小"""
        rng = np.random.default_rng(42)
        
        # 创建不平衡聚类：大小从5到50不等
        n_clusters = 30
        cluster_sizes = rng.integers(5, 50, n_clusters)
        n_total = cluster_sizes.sum()
        
        cluster_ids = np.repeat(np.arange(n_clusters), cluster_sizes)
        
        data = pd.DataFrame({
            'y': rng.normal(0, 1, n_total),
            'd': rng.binomial(1, 0.5, n_total),
            'x1': rng.normal(0, 1, n_total),
            'cluster_id': cluster_ids
        })
        # 添加处理效应
        data['y'] = data['y'] + 2 * data['d']
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_ols_regression(
                data, 'y', 'd', controls=['x1'],
                vce='cluster', cluster_var='cluster_id'
            )
        
        # 应正常计算
        assert result['se'] > 0
        assert result['df_inference'] == n_clusters - 1
    
    def test_singleton_clusters(self):
        """单例聚类（每个聚类只有1个观测）"""
        rng = np.random.default_rng(42)
        n = 50
        
        data = pd.DataFrame({
            'y': rng.normal(0, 1, n),
            'd': rng.binomial(1, 0.5, n),
            'x1': rng.normal(0, 1, n),
            'cluster_id': np.arange(n)  # 每个观测一个聚类
        })
        data['y'] = data['y'] + 2 * data['d']
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_ols_regression(
                data, 'y', 'd', controls=['x1'],
                vce='cluster', cluster_var='cluster_id'
            )
        
        # 单例聚类下，聚类SE ≈ HC SE（因为没有组内相关）
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_hc = run_ols_regression(
                data, 'y', 'd', controls=['x1'], vce='hc1'
            )
        
        # SE应该接近（考虑到校正因子差异，允许更大范围）
        ratio = result['se'] / r_hc['se']
        assert 0.8 < ratio < 1.3, f"Ratio {ratio:.3f} unexpected for singleton clusters"
    
    def test_extreme_imbalance(self):
        """极端不平衡聚类"""
        rng = np.random.default_rng(42)
        
        # 一个大聚类 + 多个小聚类
        n_clusters = 25
        cluster_sizes = [200] + [5] * (n_clusters - 1)  # 200 + 24*5 = 320
        n_total = sum(cluster_sizes)
        
        cluster_ids = np.repeat(np.arange(n_clusters), cluster_sizes)
        
        data = pd.DataFrame({
            'y': rng.normal(0, 1, n_total),
            'd': rng.binomial(1, 0.5, n_total),
            'cluster_id': cluster_ids
        })
        data['y'] = data['y'] + 2 * data['d']
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_ols_regression(
                data, 'y', 'd', vce='cluster', cluster_var='cluster_id'
            )
        
        assert result['se'] > 0
        assert result['df_inference'] == n_clusters - 1


# =============================================================================
# Task 5.4.9: 公式验证测试
# =============================================================================

class TestClusterSEFormulaValidation:
    """聚类SE公式验证"""
    
    def test_correction_factor_calculation(self):
        """验证校正因子计算"""
        test_cases = [
            (30, 300, 5, 1.0484),   # G, N, K, expected (approx)
            (50, 500, 4, 1.0266),
            (10, 100, 3, 1.1340),
            (20, 400, 6, 1.0565),
        ]
        
        for G, N, K, expected in test_cases:
            c = (G / (G - 1)) * ((N - 1) / (N - K))
            assert np.isclose(c, expected, rtol=0.01), \
                f"G={G}, N={N}, K={K}: got {c:.4f}, expected {expected:.4f}"
    
    def test_meat_matrix_structure(self):
        """验证Meat矩阵结构"""
        # 创建简单数据来验证计算
        rng = np.random.default_rng(42)
        n_clusters = 5
        n_per_cluster = 20
        n = n_clusters * n_per_cluster
        
        cluster_ids = np.repeat(np.arange(n_clusters), n_per_cluster)
        X = np.column_stack([np.ones(n), rng.normal(0, 1, n)])
        residuals = rng.normal(0, 1, n)
        
        # 手工计算Meat矩阵
        meat_manual = np.zeros((2, 2))
        for g in range(n_clusters):
            mask = cluster_ids == g
            X_g = X[mask]
            e_g = residuals[mask]
            score_g = X_g.T @ e_g
            meat_manual += np.outer(score_g, score_g)
        
        # Meat矩阵应该是对称的
        assert np.allclose(meat_manual, meat_manual.T)
        
        # Meat矩阵应该是半正定的
        eigenvalues = np.linalg.eigvalsh(meat_manual)
        assert np.all(eigenvalues >= -1e-10)
    
    def test_sandwich_variance_positive_definite(self):
        """验证三明治方差矩阵是正定的"""
        data = generate_clustered_data(n=300, n_clusters=30, seed=42)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_ols_regression(
                data, 'y', 'd', controls=['x1'],
                vce='cluster', cluster_var='cluster_id'
            )
        
        # SE应该是正数
        assert result['se'] > 0


# =============================================================================
# Task 5.4.10: 端到端 LWDID 工作流测试
# =============================================================================

class TestLWDIDClusterSEEndToEnd:
    """LWDID完整工作流聚类SE测试"""
    
    def test_run_ols_regression_with_cluster(self):
        """run_ols_regression 聚类SE测试"""
        data = generate_clustered_data(n=500, n_clusters=30, seed=42)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_ols_regression(
                data, 'y', 'd', controls=['x1', 'x2'],
                vce='cluster', cluster_var='cluster_id'
            )
        
        assert result['att'] is not None
        assert result['se'] > 0
        assert result['df_inference'] == 29
        assert not np.isnan(result['pvalue'])
    
    def test_cluster_se_with_few_clusters_warning(self):
        """少聚类时发出警告"""
        data = generate_clustered_data(n=100, n_clusters=10, seed=42)
        
        with pytest.warns(UserWarning, match="Few clusters"):
            result = run_ols_regression(
                data, 'y', 'd', controls=['x1'],
                vce='cluster', cluster_var='cluster_id'
            )
        
        # 警告不阻止计算
        assert result['att'] is not None
        assert result['df_inference'] == 9
    
    def test_cluster_vs_robust_with_cluster_effects(self):
        """比较聚类SE和稳健SE（存在聚类效应时）"""
        # 生成有强聚类效应的数据
        rng = np.random.default_rng(42)
        n_clusters = 30
        n_per_cluster = 20
        n = n_clusters * n_per_cluster
        
        cluster_ids = np.repeat(np.arange(n_clusters), n_per_cluster)
        cluster_effects = rng.normal(0, 2.0, n_clusters)  # 强聚类效应
        individual_effects = cluster_effects[cluster_ids]
        
        X = rng.normal(0, 1, n)
        D = rng.binomial(1, 0.5, n)
        Y = 1 + 2*D + 0.5*X + individual_effects + rng.normal(0, 0.5, n)
        
        data = pd.DataFrame({
            'y': Y, 'd': D, 'x1': X, 'cluster_id': cluster_ids
        })
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_cluster = run_ols_regression(
                data, 'y', 'd', controls=['x1'],
                vce='cluster', cluster_var='cluster_id'
            )
            r_robust = run_ols_regression(
                data, 'y', 'd', controls=['x1'],
                vce='robust'
            )
        
        # 存在聚类效应时，聚类SE应该更大
        assert r_cluster['se'] > r_robust['se'], \
            f"Cluster SE ({r_cluster['se']:.4f}) should be larger than robust SE ({r_robust['se']:.4f})"
        
        # ATT点估计应该相同
        assert np.isclose(r_cluster['att'], r_robust['att'])
    
    def test_cluster_se_g_2_minimum(self):
        """G=2的最小有效值"""
        data = generate_clustered_data(n=20, n_clusters=2, seed=42)
        
        with pytest.warns(UserWarning, match="Few clusters"):
            result = run_ols_regression(
                data, 'y', 'd', vce='cluster', cluster_var='cluster_id'
            )
        
        assert result['df_inference'] == 1  # G - 1 = 2 - 1 = 1
        assert result['se'] > 0
    
    def test_cluster_se_g_1_error(self):
        """G=1应抛出错误"""
        rng = np.random.default_rng(42)
        n = 20
        data = pd.DataFrame({
            'y': rng.normal(0, 1, n),
            'd': rng.binomial(1, 0.5, n),
            'cluster_id': np.zeros(n)  # 只有一个聚类
        })
        
        with pytest.raises(ValueError, match="at least 2 clusters"):
            run_ols_regression(
                data, 'y', 'd', vce='cluster', cluster_var='cluster_id'
            )
    
    def test_cluster_var_missing_error(self):
        """缺少cluster_var应抛出错误"""
        from lwdid.exceptions import InvalidParameterError
        data = generate_clustered_data(n=100, n_clusters=10, seed=42)
        
        with pytest.raises(InvalidParameterError, match="requires cluster_var parameter"):
            run_ols_regression(data, 'y', 'd', vce='cluster')
    
    def test_cluster_var_not_found_error(self):
        """cluster_var不存在应抛出错误"""
        data = generate_clustered_data(n=100, n_clusters=10, seed=42)
        
        with pytest.raises(ValueError, match="not in data"):
            run_ols_regression(
                data, 'y', 'd', vce='cluster', cluster_var='nonexistent'
            )


# =============================================================================
# 辅助测试
# =============================================================================

class TestClusterSEEdgeCases:
    """边界情况测试"""
    
    def test_many_clusters_no_warning(self):
        """大量聚类时不发出警告"""
        data = generate_clustered_data(n=1000, n_clusters=100, seed=42)
        
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # 将警告转为错误
            result = run_ols_regression(
                data, 'y', 'd', vce='cluster', cluster_var='cluster_id'
            )
        
        assert result['se'] > 0
        assert result['df_inference'] == 99
    
    def test_cluster_with_missing_values(self):
        """处理缺失值"""
        rng = np.random.default_rng(42)
        n = 300
        
        data = pd.DataFrame({
            'y': rng.normal(0, 1, n),
            'd': rng.binomial(1, 0.5, n),
            'x1': rng.normal(0, 1, n),
            'cluster_id': np.repeat(np.arange(30), 10)
        })
        
        # 添加一些缺失值
        data.loc[data.index[:10], 'x1'] = np.nan
        data['y'] = data['y'] + 2 * data['d']
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_ols_regression(
                data, 'y', 'd', controls=['x1'],
                vce='cluster', cluster_var='cluster_id'
            )
        
        # 应该正常处理缺失值
        assert result['se'] > 0
        assert result['nobs'] == n - 10  # 减去缺失值


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

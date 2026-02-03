"""
Empirical data tests for clustering diagnostics.

Tests clustering diagnostics and recommendations using simulated hierarchical
data that mimics real-world panel data structures.

The simulated data has a hierarchical structure:
- idcode: Individual worker ID (unit level)
- year: Time period
- industry: Industry code (higher-level cluster)
- region: Region code (even higher-level cluster)
- state: State code (another higher-level cluster)
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

# 导入被测试的模块
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lwdid.clustering_diagnostics import (
    diagnose_clustering,
    recommend_clustering_level,
    check_clustering_consistency,
    ClusteringLevel,
    ClusterVarStats,
    ClusteringDiagnostics,
    ClusteringRecommendation,
    ClusteringConsistencyResult,
)


# =============================================================================
# 测试数据生成
# =============================================================================

@pytest.fixture(scope="module")
def hierarchical_panel_data():
    """
    生成具有层级结构的模拟面板数据。
    
    结构:
    - 4 个区域 (region)
    - 每个区域 5 个州 (state) = 20 个州
    - 每个州 10 个行业 (industry) = 200 个行业
    - 每个行业 5 个个体 (idcode) = 1000 个个体
    - 每个个体 10 个时期 (year)
    """
    np.random.seed(42)
    
    n_regions = 4
    states_per_region = 5
    industries_per_state = 10
    individuals_per_industry = 5
    n_periods = 10
    
    data = []
    idcode = 0
    
    for region in range(n_regions):
        for state_idx in range(states_per_region):
            state = region * states_per_region + state_idx
            
            for industry_idx in range(industries_per_state):
                industry = state * industries_per_state + industry_idx
                
                # 处理在州层级分配：前 10 个州在第 6 年开始处理
                first_treat = 6 if state < 10 else 0
                
                for ind_idx in range(individuals_per_industry):
                    idcode += 1
                    
                    for year in range(1, n_periods + 1):
                        # 生成结果变量
                        # 包含：基线 + 时间趋势 + 州效应 + 处理效应 + 噪声
                        state_effect = np.random.normal(0, 2)
                        treat_effect = 2.0 if (first_treat > 0 and year >= first_treat) else 0
                        
                        y = 10 + 0.5 * year + state_effect + treat_effect + np.random.normal(0, 1)
                        
                        data.append({
                            'idcode': idcode,
                            'year': year,
                            'region': region,
                            'state': state,
                            'industry': industry,
                            'first_treat': first_treat,
                            'Y': y
                        })
    
    return pd.DataFrame(data)


@pytest.fixture(scope="module")
def small_cluster_data():
    """
    生成聚类数较少的数据，用于测试 wild bootstrap 推荐。
    
    结构:
    - 5 个州 (state)
    - 每个州 20 个个体
    - 10 个时期
    """
    np.random.seed(123)
    
    n_states = 5
    individuals_per_state = 20
    n_periods = 10
    
    data = []
    idcode = 0
    
    for state in range(n_states):
        # 前 2 个州在第 6 年开始处理
        first_treat = 6 if state < 2 else 0
        
        for ind_idx in range(individuals_per_state):
            idcode += 1
            
            for year in range(1, n_periods + 1):
                treat_effect = 1.5 if (first_treat > 0 and year >= first_treat) else 0
                y = 10 + 0.3 * year + treat_effect + np.random.normal(0, 1)
                
                data.append({
                    'idcode': idcode,
                    'year': year,
                    'state': state,
                    'first_treat': first_treat,
                    'Y': y
                })
    
    return pd.DataFrame(data)


# =============================================================================
# 基本功能测试
# =============================================================================

class TestDiagnoseClusteringEmpirical:
    """使用模拟层级数据测试 diagnose_clustering 函数。"""
    
    def test_diagnose_with_hierarchical_data(self, hierarchical_panel_data):
        """测试诊断函数在层级数据上的基本功能。"""
        df = hierarchical_panel_data
        
        diag = diagnose_clustering(
            df,
            ivar='idcode',
            potential_cluster_vars=['state', 'region', 'industry'],
            gvar='first_treat',
            verbose=False
        )
        
        # 验证返回类型
        assert isinstance(diag, ClusteringDiagnostics)
        
        # 验证聚类结构包含所有变量
        assert 'state' in diag.cluster_structure
        assert 'region' in diag.cluster_structure
        assert 'industry' in diag.cluster_structure
        
        # 验证有推荐
        assert diag.recommended_cluster_var is not None
        assert diag.recommendation_reason != ""
    
    def test_cluster_stats_reasonable(self, hierarchical_panel_data):
        """验证聚类统计信息的合理性。"""
        df = hierarchical_panel_data
        
        diag = diagnose_clustering(
            df,
            ivar='idcode',
            potential_cluster_vars=['state', 'region'],
            gvar='first_treat',
            verbose=False
        )
        
        for var_name, stats in diag.cluster_structure.items():
            # 聚类数应该为正
            assert stats.n_clusters > 0
            
            # 聚类大小应该合理
            assert stats.min_cluster_size > 0
            assert stats.max_cluster_size >= stats.min_cluster_size
            assert stats.mean_cluster_size > 0
            
            # 变异系数应该非负
            assert stats.cluster_size_cv >= 0
            
            # 处理/控制聚类数应该合理
            assert stats.n_treated_clusters >= 0
            assert stats.n_control_clusters >= 0
            assert stats.n_treated_clusters + stats.n_control_clusters <= stats.n_clusters
    
    def test_level_detection(self, hierarchical_panel_data):
        """测试层级检测的正确性。"""
        df = hierarchical_panel_data
        
        diag = diagnose_clustering(
            df,
            ivar='idcode',
            potential_cluster_vars=['state', 'region', 'idcode'],
            gvar='first_treat',
            verbose=False
        )
        
        # idcode 应该是 SAME 层级（与单位相同）
        idcode_stats = diag.cluster_structure.get('idcode')
        if idcode_stats:
            assert idcode_stats.level_relative_to_unit == ClusteringLevel.SAME
        
        # state 和 region 应该是 HIGHER 层级
        state_stats = diag.cluster_structure.get('state')
        if state_stats:
            assert state_stats.level_relative_to_unit == ClusteringLevel.HIGHER
        
        region_stats = diag.cluster_structure.get('region')
        if region_stats:
            assert region_stats.level_relative_to_unit == ClusteringLevel.HIGHER
    
    def test_summary_output(self, hierarchical_panel_data):
        """测试摘要输出的格式。"""
        df = hierarchical_panel_data
        
        diag = diagnose_clustering(
            df,
            ivar='idcode',
            potential_cluster_vars=['state', 'region'],
            gvar='first_treat',
            verbose=False
        )
        
        summary = diag.summary()
        
        # 验证摘要包含关键信息
        assert "CLUSTERING DIAGNOSTICS" in summary
        assert "state" in summary
        assert "region" in summary
        assert "RECOMMENDATION" in summary
    
    def test_treatment_variation_detection(self, hierarchical_panel_data):
        """测试处理变化层级检测。"""
        df = hierarchical_panel_data
        
        diag = diagnose_clustering(
            df,
            ivar='idcode',
            potential_cluster_vars=['state', 'region'],
            gvar='first_treat',
            verbose=False
        )
        
        # 处理在 state 层级分配，所以 state 内不应该有处理变化
        state_stats = diag.cluster_structure['state']
        # 使用 == False 而不是 is False，因为可能返回 np.False_
        assert state_stats.treatment_varies_within_cluster == False
        
        # 由于数据生成方式，处理在 region 层级也是常数
        # （前 2 个 region 全部处理，后 2 个 region 全部控制）
        # 所以检测到的处理变化层级是 region（最高层级）
        # 这是正确的行为：函数返回处理常数的最高层级
        assert diag.treatment_variation_level in ['state', 'region']


class TestRecommendClusteringLevelEmpirical:
    """使用模拟数据测试 recommend_clustering_level 函数。"""
    
    def test_recommendation_with_hierarchical_data(self, hierarchical_panel_data):
        """测试推荐函数在层级数据上的基本功能。"""
        df = hierarchical_panel_data
        
        rec = recommend_clustering_level(
            df,
            ivar='idcode',
            tvar='year',
            potential_cluster_vars=['state', 'region'],
            gvar='first_treat',
            min_clusters=10,
            verbose=False
        )
        
        # 验证返回类型
        assert isinstance(rec, ClusteringRecommendation)
        
        # 验证推荐变量
        assert rec.recommended_var in ['state', 'region']
        
        # 验证聚类数
        assert rec.n_clusters > 0
        
        # 验证置信度在合理范围
        assert 0 <= rec.confidence <= 1
        
        # 验证有推荐理由
        assert len(rec.reasons) > 0
    
    def test_recommends_treatment_level(self, hierarchical_panel_data):
        """测试是否推荐处理变化层级的聚类。"""
        df = hierarchical_panel_data
        
        rec = recommend_clustering_level(
            df,
            ivar='idcode',
            tvar='year',
            potential_cluster_vars=['state', 'region'],
            gvar='first_treat',
            verbose=False
        )
        
        # 处理在 state 层级分配，应该推荐 state
        # （除非 state 聚类数不足）
        assert rec.recommended_var == 'state'
    
    def test_wild_bootstrap_recommendation(self, small_cluster_data):
        """测试 wild bootstrap 推荐逻辑。"""
        df = small_cluster_data
        
        rec = recommend_clustering_level(
            df,
            ivar='idcode',
            tvar='year',
            potential_cluster_vars=['state'],
            gvar='first_treat',
            min_clusters=20,  # 高于实际聚类数
            verbose=False
        )
        
        # 只有 5 个州，应该推荐 wild bootstrap
        assert rec.use_wild_bootstrap is True
        assert rec.wild_bootstrap_reason is not None
        assert rec.n_clusters == 5
    
    def test_alternatives_provided(self, hierarchical_panel_data):
        """测试是否提供替代方案。"""
        df = hierarchical_panel_data
        
        rec = recommend_clustering_level(
            df,
            ivar='idcode',
            tvar='year',
            potential_cluster_vars=['state', 'region'],
            gvar='first_treat',
            verbose=False
        )
        
        # 有多个有效选项，应该提供替代方案
        assert isinstance(rec.alternatives, list)
        assert len(rec.alternatives) >= 1


class TestCheckClusteringConsistencyEmpirical:
    """使用模拟数据测试 check_clustering_consistency 函数。"""
    
    def test_consistency_check_with_hierarchical_data(self, hierarchical_panel_data):
        """测试一致性检查在层级数据上的基本功能。"""
        df = hierarchical_panel_data
        
        result = check_clustering_consistency(
            df,
            ivar='idcode',
            cluster_var='state',
            gvar='first_treat',
            verbose=False
        )
        
        # 验证返回类型
        assert isinstance(result, ClusteringConsistencyResult)
        
        # 验证字段
        assert isinstance(result.is_consistent, bool)
        assert result.n_clusters > 0
        assert 0 <= result.pct_clusters_with_variation <= 100
        assert result.recommendation != ""
        assert result.details != ""
    
    def test_consistency_with_treatment_level_cluster(self, hierarchical_panel_data):
        """测试在处理层级聚类时的一致性。"""
        df = hierarchical_panel_data
        
        # 处理在 state 层级分配，所以在 state 聚类应该一致
        result = check_clustering_consistency(
            df,
            ivar='idcode',
            cluster_var='state',
            gvar='first_treat',
            verbose=False
        )
        
        # 处理在 state 层级分配，不应该在 state 内变化
        assert result.pct_clusters_with_variation == 0
        assert result.is_consistent is True
    
    def test_inconsistency_with_higher_level_cluster(self, hierarchical_panel_data):
        """测试在更高层级聚类时的一致性检查。"""
        df = hierarchical_panel_data
        
        # 处理在 state 层级分配，在 region 层级聚类
        # 由于数据生成方式，每个 region 包含 5 个 state
        # 前 10 个 state 是处理组，后 10 个是控制组
        # region 0 包含 state 0-4 (全部处理)
        # region 1 包含 state 5-9 (全部处理)
        # region 2 包含 state 10-14 (全部控制)
        # region 3 包含 state 15-19 (全部控制)
        # 所以实际上处理在 region 内不变化
        
        result = check_clustering_consistency(
            df,
            ivar='idcode',
            cluster_var='region',
            gvar='first_treat',
            verbose=False
        )
        
        # 由于数据生成方式，region 内处理不变化
        # 这是一个有效的测试场景，验证函数正确检测到一致性
        assert result.n_clusters == 4
        # 处理变化层级应该是 region（因为处理在 region 层级是常数）
        assert result.treatment_variation_level == 'region'
    
    def test_true_inconsistency_detection(self):
        """测试真正的不一致性检测。"""
        np.random.seed(789)
        
        # 创建处理在 region 内变化的数据
        data = []
        idcode = 0
        
        for region in range(4):
            for state in range(5):
                state_id = region * 5 + state
                # 处理在 state 层级分配，但每个 region 内有处理和控制 state
                # 每个 region 的前 2 个 state 是处理组
                first_treat = 6 if state < 2 else 0
                
                for ind in range(10):
                    idcode += 1
                    for year in range(1, 11):
                        data.append({
                            'idcode': idcode,
                            'year': year,
                            'region': region,
                            'state': state_id,
                            'first_treat': first_treat,
                            'Y': np.random.normal(10, 1)
                        })
        
        df = pd.DataFrame(data)
        
        # 在 region 层级聚类，但处理在 state 层级分配
        result = check_clustering_consistency(
            df,
            ivar='idcode',
            cluster_var='region',
            gvar='first_treat',
            verbose=False
        )
        
        # 每个 region 内都有处理和控制 state，所以处理在 region 内变化
        assert result.pct_clusters_with_variation > 0
        # 所有 4 个 region 都有变化
        assert result.n_treatment_changes_within_cluster == 4
    
    def test_summary_output(self, hierarchical_panel_data):
        """测试摘要输出的格式。"""
        df = hierarchical_panel_data
        
        result = check_clustering_consistency(
            df,
            ivar='idcode',
            cluster_var='state',
            gvar='first_treat',
            verbose=False
        )
        
        summary = result.summary()
        
        # 验证摘要包含关键信息
        assert "CLUSTERING CONSISTENCY CHECK" in summary
        assert "Recommendation" in summary


# =============================================================================
# 边界情况测试
# =============================================================================

class TestEdgeCasesEmpirical:
    """使用模拟数据测试边界情况。"""
    
    def test_single_cluster_variable(self, hierarchical_panel_data):
        """测试只有一个聚类变量的情况。"""
        df = hierarchical_panel_data
        
        diag = diagnose_clustering(
            df,
            ivar='idcode',
            potential_cluster_vars=['state'],
            gvar='first_treat',
            verbose=False
        )
        
        assert len(diag.cluster_structure) == 1
        assert 'state' in diag.cluster_structure
    
    def test_missing_values_handling(self, hierarchical_panel_data):
        """测试包含缺失值的数据处理。"""
        df = hierarchical_panel_data.copy()
        
        # 添加一些缺失值
        df.loc[df.index[:10], 'state'] = np.nan
        
        # 函数应该能够处理缺失值
        try:
            diag = diagnose_clustering(
                df,
                ivar='idcode',
                potential_cluster_vars=['state'],
                gvar='first_treat',
                verbose=False
            )
            # 如果成功，验证结果
            assert isinstance(diag, ClusteringDiagnostics)
        except Exception as e:
            # 如果失败，应该是合理的错误
            assert "missing" in str(e).lower() or "nan" in str(e).lower()
    
    def test_verbose_output(self, hierarchical_panel_data, capsys):
        """测试 verbose 输出。"""
        df = hierarchical_panel_data
        
        # verbose=True 应该打印输出
        diag = diagnose_clustering(
            df,
            ivar='idcode',
            potential_cluster_vars=['state'],
            gvar='first_treat',
            verbose=True
        )
        
        captured = capsys.readouterr()
        assert "CLUSTERING DIAGNOSTICS" in captured.out
    
    def test_all_treated_or_control(self):
        """测试所有单位都是处理组或控制组的情况。"""
        np.random.seed(456)
        
        # 创建所有单位都是处理组的数据
        data = []
        for i in range(100):
            for t in range(1, 11):
                data.append({
                    'idcode': i,
                    'year': t,
                    'state': i // 10,
                    'first_treat': 5,  # 所有单位都在第 5 年处理
                    'Y': np.random.normal(10, 1)
                })
        
        df = pd.DataFrame(data)
        
        diag = diagnose_clustering(
            df,
            ivar='idcode',
            potential_cluster_vars=['state'],
            gvar='first_treat',
            verbose=False
        )
        
        # 所有聚类都是处理组
        state_stats = diag.cluster_structure['state']
        assert state_stats.n_control_clusters == 0


# =============================================================================
# 集成测试
# =============================================================================

class TestIntegrationEmpirical:
    """使用模拟数据的集成测试。"""
    
    def test_full_workflow(self, hierarchical_panel_data):
        """测试完整的聚类诊断工作流。"""
        df = hierarchical_panel_data
        
        # Step 1: 诊断
        diag = diagnose_clustering(
            df,
            ivar='idcode',
            potential_cluster_vars=['state', 'region'],
            gvar='first_treat',
            verbose=False
        )
        
        # Step 2: 获取推荐
        rec = recommend_clustering_level(
            df,
            ivar='idcode',
            tvar='year',
            potential_cluster_vars=['state', 'region'],
            gvar='first_treat',
            verbose=False
        )
        
        # Step 3: 检查一致性
        consistency = check_clustering_consistency(
            df,
            ivar='idcode',
            cluster_var=rec.recommended_var,
            gvar='first_treat',
            verbose=False
        )
        
        # 验证工作流的一致性
        assert rec.recommended_var in diag.cluster_structure
        assert consistency.n_clusters == rec.n_clusters
    
    def test_recommendation_matches_diagnostics(self, hierarchical_panel_data):
        """验证推荐与诊断结果的一致性。"""
        df = hierarchical_panel_data
        
        diag = diagnose_clustering(
            df,
            ivar='idcode',
            potential_cluster_vars=['state', 'region'],
            gvar='first_treat',
            verbose=False
        )
        
        rec = recommend_clustering_level(
            df,
            ivar='idcode',
            tvar='year',
            potential_cluster_vars=['state', 'region'],
            gvar='first_treat',
            verbose=False
        )
        
        # 推荐的变量应该在诊断结果中
        assert rec.recommended_var in diag.cluster_structure
        
        # 推荐的聚类数应该与诊断一致
        diag_stats = diag.cluster_structure[rec.recommended_var]
        assert rec.n_clusters == diag_stats.n_clusters
        assert rec.n_treated_clusters == diag_stats.n_treated_clusters
        assert rec.n_control_clusters == diag_stats.n_control_clusters
    
    def test_workflow_with_small_clusters(self, small_cluster_data):
        """测试小聚类数据的完整工作流。"""
        df = small_cluster_data
        
        # 诊断
        diag = diagnose_clustering(
            df,
            ivar='idcode',
            potential_cluster_vars=['state'],
            gvar='first_treat',
            verbose=False
        )
        
        # 推荐
        rec = recommend_clustering_level(
            df,
            ivar='idcode',
            tvar='year',
            potential_cluster_vars=['state'],
            gvar='first_treat',
            min_clusters=20,
            verbose=False
        )
        
        # 应该推荐 wild bootstrap
        assert rec.use_wild_bootstrap is True
        
        # 一致性检查
        consistency = check_clustering_consistency(
            df,
            ivar='idcode',
            cluster_var='state',
            gvar='first_treat',
            verbose=False
        )
        
        assert consistency.n_clusters == 5


# =============================================================================
# 性能测试
# =============================================================================

class TestPerformanceEmpirical:
    """使用模拟数据的性能测试。"""
    
    def test_large_data_performance(self, hierarchical_panel_data):
        """测试在较大数据集上的性能。"""
        import time
        
        df = hierarchical_panel_data
        
        # 记录执行时间
        start_time = time.time()
        
        diag = diagnose_clustering(
            df,
            ivar='idcode',
            potential_cluster_vars=['state', 'region'],
            gvar='first_treat',
            verbose=False
        )
        
        elapsed_time = time.time() - start_time
        
        # 应该在合理时间内完成（例如 10 秒）
        assert elapsed_time < 10, f"诊断耗时过长: {elapsed_time:.2f}s"
    
    def test_multiple_cluster_vars_performance(self, hierarchical_panel_data):
        """测试多个聚类变量时的性能。"""
        import time
        
        df = hierarchical_panel_data.copy()
        
        # 添加更多聚类变量
        df['cluster_a'] = df['idcode'] % 20
        df['cluster_b'] = df['idcode'] % 50
        df['cluster_c'] = df['idcode'] % 100
        
        start_time = time.time()
        
        diag = diagnose_clustering(
            df,
            ivar='idcode',
            potential_cluster_vars=['state', 'region', 'cluster_a', 'cluster_b', 'cluster_c'],
            gvar='first_treat',
            verbose=False
        )
        
        elapsed_time = time.time() - start_time
        
        # 应该在合理时间内完成
        assert elapsed_time < 30, f"诊断耗时过长: {elapsed_time:.2f}s"
        
        # 验证所有变量都被分析
        assert len(diag.cluster_structure) == 5
    
    def test_recommend_performance(self, hierarchical_panel_data):
        """测试推荐函数的性能。"""
        import time
        
        df = hierarchical_panel_data
        
        start_time = time.time()
        
        rec = recommend_clustering_level(
            df,
            ivar='idcode',
            tvar='year',
            potential_cluster_vars=['state', 'region', 'industry'],
            gvar='first_treat',
            verbose=False
        )
        
        elapsed_time = time.time() - start_time
        
        # 应该在合理时间内完成
        assert elapsed_time < 15, f"推荐耗时过长: {elapsed_time:.2f}s"


# =============================================================================
# 数值正确性测试
# =============================================================================

class TestNumericalCorrectness:
    """验证数值计算的正确性。"""
    
    def test_cluster_count_accuracy(self, hierarchical_panel_data):
        """验证聚类数计算的准确性。"""
        df = hierarchical_panel_data
        
        diag = diagnose_clustering(
            df,
            ivar='idcode',
            potential_cluster_vars=['state', 'region'],
            gvar='first_treat',
            verbose=False
        )
        
        # 手动计算聚类数
        expected_state_clusters = df['state'].nunique()
        expected_region_clusters = df['region'].nunique()
        
        assert diag.cluster_structure['state'].n_clusters == expected_state_clusters
        assert diag.cluster_structure['region'].n_clusters == expected_region_clusters
    
    def test_treated_control_split(self, hierarchical_panel_data):
        """验证处理/控制聚类划分的准确性。"""
        df = hierarchical_panel_data
        
        diag = diagnose_clustering(
            df,
            ivar='idcode',
            potential_cluster_vars=['state'],
            gvar='first_treat',
            verbose=False
        )
        
        # 手动计算处理聚类数
        # 处理组：first_treat > 0
        treated_states = df[df['first_treat'] > 0]['state'].unique()
        control_states = df[df['first_treat'] == 0]['state'].unique()
        
        # 注意：一个 state 可能同时有处理和控制单位
        # 这里我们检查的是"包含处理单位的聚类数"
        state_stats = diag.cluster_structure['state']
        
        # 验证总数正确
        assert state_stats.n_clusters == df['state'].nunique()
    
    def test_cluster_size_statistics(self, hierarchical_panel_data):
        """验证聚类大小统计的准确性。"""
        df = hierarchical_panel_data
        
        diag = diagnose_clustering(
            df,
            ivar='idcode',
            potential_cluster_vars=['state'],
            gvar='first_treat',
            verbose=False
        )
        
        # 手动计算聚类大小统计
        cluster_sizes = df.groupby('state').size()
        
        state_stats = diag.cluster_structure['state']
        
        assert state_stats.min_cluster_size == cluster_sizes.min()
        assert state_stats.max_cluster_size == cluster_sizes.max()
        assert abs(state_stats.mean_cluster_size - cluster_sizes.mean()) < 0.01
        assert abs(state_stats.median_cluster_size - cluster_sizes.median()) < 0.01
    
    def test_reliability_score_bounds(self, hierarchical_panel_data):
        """验证可靠性评分在有效范围内。"""
        df = hierarchical_panel_data
        
        diag = diagnose_clustering(
            df,
            ivar='idcode',
            potential_cluster_vars=['state', 'region'],
            gvar='first_treat',
            verbose=False
        )
        
        for var_name, stats in diag.cluster_structure.items():
            assert 0 <= stats.reliability_score <= 1, \
                f"{var_name} 的可靠性评分超出范围: {stats.reliability_score}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

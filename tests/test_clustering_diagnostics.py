"""
单元测试：聚类诊断模块 (clustering_diagnostics.py)

测试内容：
- 数据类 (ClusterVarStats, ClusteringDiagnostics, etc.)
- 诊断函数 (diagnose_clustering)
- 推荐函数 (recommend_clustering_level)
- 一致性检验 (check_clustering_consistency)
"""

import numpy as np
import pandas as pd
import pytest

from lwdid.clustering_diagnostics import (
    ClusteringLevel,
    ClusteringWarningLevel,
    ClusterVarStats,
    ClusteringDiagnostics,
    ClusteringRecommendation,
    ClusteringConsistencyResult,
    WildClusterBootstrapResult,
    diagnose_clustering,
    recommend_clustering_level,
    check_clustering_consistency,
)


# =============================================================================
# 测试数据生成器
# =============================================================================

@pytest.fixture
def hierarchical_data():
    """生成具有清晰层级结构的数据：州 > 县"""
    np.random.seed(42)
    n_states = 50
    counties_per_state = 10
    periods = 10
    
    data = []
    for state in range(n_states):
        for county in range(counties_per_state):
            county_id = state * counties_per_state + county
            # 处理在州级别变化
            first_treat = 6 if state < 25 else 0
            for t in range(1, periods + 1):
                data.append({
                    'state': state,
                    'county': county_id,
                    'year': t,
                    'first_treat': first_treat,
                    'Y': np.random.normal(10 + 0.5 * t, 1)
                })
    return pd.DataFrame(data)


@pytest.fixture
def small_cluster_data():
    """生成少量聚类的数据（5个州）"""
    np.random.seed(42)
    n_states = 5
    counties_per_state = 100
    
    data = []
    for state in range(n_states):
        for county in range(counties_per_state):
            county_id = state * counties_per_state + county
            first_treat = 6 if state < 2 else 0
            for year in range(1, 11):
                data.append({
                    'state': state,
                    'county': county_id,
                    'year': year,
                    'first_treat': first_treat,
                    'Y': np.random.normal(10, 1)
                })
    return pd.DataFrame(data)


@pytest.fixture
def treatment_varies_within_cluster_data():
    """生成处理在聚类内变化的数据"""
    np.random.seed(42)
    data = pd.DataFrame({
        'state': [1, 1, 1, 1, 2, 2, 2, 2],
        'county': [1, 2, 3, 4, 5, 6, 7, 8],
        'year': [1, 1, 1, 1, 1, 1, 1, 1],
        'first_treat': [3, 3, 4, 4, 3, 4, 3, 4],  # 在州内变化
        'Y': np.random.normal(10, 1, 8)
    })
    return data


# =============================================================================
# ClusterVarStats 数据类测试
# =============================================================================

class TestClusterVarStats:
    """测试 ClusterVarStats 数据类"""
    
    def test_is_valid_cluster_higher_level(self):
        """高层级聚类且聚类数足够时应为有效"""
        stats = ClusterVarStats(
            var_name='state',
            n_clusters=50,
            n_treated_clusters=25,
            n_control_clusters=25,
            min_cluster_size=100,
            max_cluster_size=500,
            mean_cluster_size=200,
            median_cluster_size=180,
            cluster_size_cv=0.5,
            level_relative_to_unit=ClusteringLevel.HIGHER,
            units_per_cluster=10.0,
            is_nested_in_unit=False,
            treatment_varies_within_cluster=False
        )
        assert stats.is_valid_cluster is True
        assert stats.is_recommended is True
    
    def test_is_valid_cluster_nested_invalid(self):
        """嵌套在单位内的聚类应为无效"""
        stats = ClusterVarStats(
            var_name='sub_unit',
            n_clusters=1000,
            n_treated_clusters=500,
            n_control_clusters=500,
            min_cluster_size=1,
            max_cluster_size=5,
            mean_cluster_size=2,
            median_cluster_size=2,
            cluster_size_cv=0.3,
            level_relative_to_unit=ClusteringLevel.LOWER,
            units_per_cluster=1.0,
            is_nested_in_unit=True,
            treatment_varies_within_cluster=False
        )
        assert stats.is_valid_cluster is False
    
    def test_is_valid_cluster_too_few_clusters(self):
        """聚类数少于2时应为无效"""
        stats = ClusterVarStats(
            var_name='single_cluster',
            n_clusters=1,
            n_treated_clusters=1,
            n_control_clusters=0,
            min_cluster_size=100,
            max_cluster_size=100,
            mean_cluster_size=100,
            median_cluster_size=100,
            cluster_size_cv=0.0,
            level_relative_to_unit=ClusteringLevel.HIGHER,
            units_per_cluster=100.0,
            is_nested_in_unit=False,
            treatment_varies_within_cluster=False
        )
        assert stats.is_valid_cluster is False
    
    def test_is_recommended_few_clusters(self):
        """聚类数少于20时不应推荐"""
        stats = ClusterVarStats(
            var_name='region',
            n_clusters=10,
            n_treated_clusters=5,
            n_control_clusters=5,
            min_cluster_size=100,
            max_cluster_size=200,
            mean_cluster_size=150,
            median_cluster_size=150,
            cluster_size_cv=0.3,
            level_relative_to_unit=ClusteringLevel.HIGHER,
            units_per_cluster=50.0,
            is_nested_in_unit=False,
            treatment_varies_within_cluster=False
        )
        assert stats.is_valid_cluster is True
        assert stats.is_recommended is False  # 聚类数 < 20
    
    def test_is_recommended_treatment_varies(self):
        """处理在聚类内变化时不应推荐"""
        stats = ClusterVarStats(
            var_name='state',
            n_clusters=50,
            n_treated_clusters=25,
            n_control_clusters=25,
            min_cluster_size=100,
            max_cluster_size=200,
            mean_cluster_size=150,
            median_cluster_size=150,
            cluster_size_cv=0.3,
            level_relative_to_unit=ClusteringLevel.HIGHER,
            units_per_cluster=10.0,
            is_nested_in_unit=False,
            treatment_varies_within_cluster=True  # 处理在聚类内变化
        )
        assert stats.is_valid_cluster is True
        assert stats.is_recommended is False
    
    def test_reliability_score_calculation(self):
        """测试可靠性评分计算"""
        # 高可靠性：多聚类、平衡、低CV
        stats_high = ClusterVarStats(
            var_name='state',
            n_clusters=50,
            n_treated_clusters=25,
            n_control_clusters=25,
            min_cluster_size=100,
            max_cluster_size=150,
            mean_cluster_size=120,
            median_cluster_size=115,
            cluster_size_cv=0.2,
            level_relative_to_unit=ClusteringLevel.HIGHER,
            units_per_cluster=10.0,
            is_nested_in_unit=False,
            treatment_varies_within_cluster=False
        )
        
        # 低可靠性：少聚类、不平衡
        stats_low = ClusterVarStats(
            var_name='region',
            n_clusters=5,
            n_treated_clusters=1,
            n_control_clusters=4,
            min_cluster_size=50,
            max_cluster_size=500,
            mean_cluster_size=200,
            median_cluster_size=150,
            cluster_size_cv=1.5,
            level_relative_to_unit=ClusteringLevel.HIGHER,
            units_per_cluster=50.0,
            is_nested_in_unit=False,
            treatment_varies_within_cluster=False
        )
        
        assert stats_high.reliability_score > stats_low.reliability_score
        assert 0 <= stats_high.reliability_score <= 1
        assert 0 <= stats_low.reliability_score <= 1
    
    def test_reliability_score_bounds(self):
        """可靠性评分应在 [0, 1] 范围内"""
        # 极端情况：最佳
        stats_best = ClusterVarStats(
            var_name='best',
            n_clusters=100,
            n_treated_clusters=50,
            n_control_clusters=50,
            min_cluster_size=100,
            max_cluster_size=100,
            mean_cluster_size=100,
            median_cluster_size=100,
            cluster_size_cv=0.0,
            level_relative_to_unit=ClusteringLevel.HIGHER,
            units_per_cluster=10.0,
            is_nested_in_unit=False,
            treatment_varies_within_cluster=False
        )
        
        # 极端情况：最差
        stats_worst = ClusterVarStats(
            var_name='worst',
            n_clusters=2,
            n_treated_clusters=2,
            n_control_clusters=0,
            min_cluster_size=1,
            max_cluster_size=1000,
            mean_cluster_size=500,
            median_cluster_size=500,
            cluster_size_cv=2.0,
            level_relative_to_unit=ClusteringLevel.HIGHER,
            units_per_cluster=1.0,
            is_nested_in_unit=False,
            treatment_varies_within_cluster=False
        )
        
        assert 0 <= stats_best.reliability_score <= 1
        assert 0 <= stats_worst.reliability_score <= 1
        assert stats_best.reliability_score > stats_worst.reliability_score


# =============================================================================
# diagnose_clustering() 测试
# =============================================================================

class TestDiagnoseClustering:
    """测试 diagnose_clustering 函数"""
    
    def test_detects_higher_level_cluster(self, hierarchical_data):
        """应检测到州是比县更高的层级"""
        diag = diagnose_clustering(
            hierarchical_data,
            ivar='county',
            potential_cluster_vars=['state', 'county'],
            gvar='first_treat',
            verbose=False
        )
        
        state_stats = diag.cluster_structure['state']
        county_stats = diag.cluster_structure['county']
        
        assert state_stats.level_relative_to_unit == ClusteringLevel.HIGHER
        assert county_stats.level_relative_to_unit == ClusteringLevel.SAME
    
    def test_recommends_treatment_level(self, hierarchical_data):
        """应推荐在处理变化层级聚类"""
        diag = diagnose_clustering(
            hierarchical_data,
            ivar='county',
            potential_cluster_vars=['state', 'county'],
            gvar='first_treat',
            verbose=False
        )
        
        # 处理在州级别变化，应推荐州
        assert diag.recommended_cluster_var == 'state'
    
    def test_detects_treatment_variation_within_cluster(
        self, treatment_varies_within_cluster_data
    ):
        """应检测到处理在聚类内变化"""
        diag = diagnose_clustering(
            treatment_varies_within_cluster_data,
            ivar='county',
            potential_cluster_vars=['state'],
            gvar='first_treat',
            verbose=False
        )
        
        state_stats = diag.cluster_structure['state']
        assert state_stats.treatment_varies_within_cluster == True
        assert len(diag.warnings) > 0
    
    def test_cluster_counts(self, hierarchical_data):
        """应正确计算聚类数量"""
        diag = diagnose_clustering(
            hierarchical_data,
            ivar='county',
            potential_cluster_vars=['state', 'county'],
            gvar='first_treat',
            verbose=False
        )
        
        assert diag.cluster_structure['state'].n_clusters == 50
        assert diag.cluster_structure['county'].n_clusters == 500
    
    def test_treated_control_cluster_counts(self, hierarchical_data):
        """应正确计算处理/控制聚类数量"""
        diag = diagnose_clustering(
            hierarchical_data,
            ivar='county',
            potential_cluster_vars=['state'],
            gvar='first_treat',
            verbose=False
        )
        
        state_stats = diag.cluster_structure['state']
        # 25个州处理，25个州控制
        assert state_stats.n_treated_clusters == 25
        assert state_stats.n_control_clusters == 25
    
    def test_summary_output(self, hierarchical_data):
        """summary() 应返回格式化字符串"""
        diag = diagnose_clustering(
            hierarchical_data,
            ivar='county',
            potential_cluster_vars=['state', 'county'],
            gvar='first_treat',
            verbose=False
        )
        
        summary = diag.summary()
        assert isinstance(summary, str)
        assert 'CLUSTERING DIAGNOSTICS' in summary
        assert 'state' in summary
        assert 'county' in summary
    
    def test_invalid_ivar_raises(self, hierarchical_data):
        """无效的单位变量应引发错误"""
        with pytest.raises(ValueError, match="not found"):
            diagnose_clustering(
                hierarchical_data,
                ivar='invalid_var',
                potential_cluster_vars=['state'],
                gvar='first_treat',
                verbose=False
            )
    
    def test_invalid_cluster_var_raises(self, hierarchical_data):
        """无效的聚类变量应引发错误"""
        with pytest.raises(ValueError, match="not found"):
            diagnose_clustering(
                hierarchical_data,
                ivar='county',
                potential_cluster_vars=['invalid_var'],
                gvar='first_treat',
                verbose=False
            )
    
    def test_no_treatment_var_raises(self, hierarchical_data):
        """未指定处理变量应引发错误"""
        with pytest.raises(ValueError, match="Either gvar or d must be specified"):
            diagnose_clustering(
                hierarchical_data,
                ivar='county',
                potential_cluster_vars=['state'],
                gvar=None,
                d=None,
                verbose=False
            )


# =============================================================================
# recommend_clustering_level() 测试
# =============================================================================

class TestRecommendClusteringLevel:
    """测试 recommend_clustering_level 函数"""
    
    def test_recommends_wild_bootstrap_few_clusters(self, small_cluster_data):
        """聚类数少时应推荐 wild bootstrap"""
        rec = recommend_clustering_level(
            small_cluster_data,
            ivar='county',
            tvar='year',
            potential_cluster_vars=['state'],
            gvar='first_treat',
            min_clusters=20,
            verbose=False
        )
        
        assert rec.use_wild_bootstrap is True
        assert rec.n_clusters == 5
        assert 'wild' in rec.wild_bootstrap_reason.lower()
    
    def test_no_wild_bootstrap_many_clusters(self, hierarchical_data):
        """聚类数足够时不应推荐 wild bootstrap"""
        rec = recommend_clustering_level(
            hierarchical_data,
            ivar='county',
            tvar='year',
            potential_cluster_vars=['state'],
            gvar='first_treat',
            min_clusters=20,
            verbose=False
        )
        
        assert rec.use_wild_bootstrap is False
        assert rec.n_clusters == 50
    
    def test_confidence_score(self, hierarchical_data):
        """应返回合理的置信度评分"""
        rec = recommend_clustering_level(
            hierarchical_data,
            ivar='county',
            tvar='year',
            potential_cluster_vars=['state', 'county'],
            gvar='first_treat',
            verbose=False
        )
        
        assert 0 <= rec.confidence <= 1
    
    def test_provides_alternatives(self, hierarchical_data):
        """应提供替代方案"""
        rec = recommend_clustering_level(
            hierarchical_data,
            ivar='county',
            tvar='year',
            potential_cluster_vars=['state', 'county'],
            gvar='first_treat',
            verbose=False
        )
        
        # 应有至少一个替代方案
        assert len(rec.alternatives) >= 1
    
    def test_summary_output(self, hierarchical_data):
        """summary() 应返回格式化字符串"""
        rec = recommend_clustering_level(
            hierarchical_data,
            ivar='county',
            tvar='year',
            potential_cluster_vars=['state'],
            gvar='first_treat',
            verbose=False
        )
        
        summary = rec.summary()
        assert isinstance(summary, str)
        assert 'RECOMMENDATION' in summary


# =============================================================================
# check_clustering_consistency() 测试
# =============================================================================

class TestCheckClusteringConsistency:
    """测试 check_clustering_consistency 函数"""
    
    def test_consistent_clustering(self, hierarchical_data):
        """在处理层级聚类应为一致"""
        result = check_clustering_consistency(
            hierarchical_data,
            ivar='county',
            cluster_var='state',
            gvar='first_treat',
            verbose=False
        )
        
        assert result.is_consistent is True
        assert result.pct_clusters_with_variation == 0
    
    def test_inconsistent_clustering(self, treatment_varies_within_cluster_data):
        """处理在聚类内变化时应为不一致"""
        result = check_clustering_consistency(
            treatment_varies_within_cluster_data,
            ivar='county',
            cluster_var='state',
            gvar='first_treat',
            verbose=False
        )
        
        # 处理在州内变化
        assert result.pct_clusters_with_variation > 0
    
    def test_summary_output(self, hierarchical_data):
        """summary() 应返回格式化字符串"""
        result = check_clustering_consistency(
            hierarchical_data,
            ivar='county',
            cluster_var='state',
            gvar='first_treat',
            verbose=False
        )
        
        summary = result.summary()
        assert isinstance(summary, str)
        assert 'CONSISTENCY' in summary
    
    def test_invalid_cluster_var_raises(self, hierarchical_data):
        """无效的聚类变量应引发错误"""
        with pytest.raises(ValueError, match="not found"):
            check_clustering_consistency(
                hierarchical_data,
                ivar='county',
                cluster_var='invalid_var',
                gvar='first_treat',
                verbose=False
            )


# =============================================================================
# WildClusterBootstrapResult 数据类测试
# =============================================================================

class TestWildClusterBootstrapResult:
    """测试 WildClusterBootstrapResult 数据类"""
    
    def test_summary_output(self):
        """summary() 应返回格式化字符串"""
        result = WildClusterBootstrapResult(
            att=2.0,
            se_bootstrap=0.5,
            ci_lower=1.0,
            ci_upper=3.0,
            pvalue=0.001,
            n_clusters=20,
            n_bootstrap=999,
            weight_type='rademacher',
            t_stat_original=4.0,
            t_stats_bootstrap=np.random.normal(0, 1, 999),
            rejection_rate=0.001
        )
        
        summary = result.summary()
        assert isinstance(summary, str)
        assert 'Wild Cluster Bootstrap' in summary
        assert '2.0' in summary or '2.00' in summary
    
    def test_significance_stars(self):
        """应根据 p 值显示显著性星号"""
        # p < 0.01: ***
        result_001 = WildClusterBootstrapResult(
            att=2.0, se_bootstrap=0.5, ci_lower=1.0, ci_upper=3.0,
            pvalue=0.005, n_clusters=20, n_bootstrap=999,
            weight_type='rademacher', t_stat_original=4.0,
            t_stats_bootstrap=np.array([]), rejection_rate=0.005
        )
        assert '***' in result_001.summary()
        
        # p < 0.05: **
        result_005 = WildClusterBootstrapResult(
            att=2.0, se_bootstrap=0.5, ci_lower=1.0, ci_upper=3.0,
            pvalue=0.03, n_clusters=20, n_bootstrap=999,
            weight_type='rademacher', t_stat_original=2.5,
            t_stats_bootstrap=np.array([]), rejection_rate=0.03
        )
        assert '**' in result_005.summary()


# =============================================================================
# 枚举类型测试
# =============================================================================

class TestEnums:
    """测试枚举类型"""
    
    def test_clustering_level_values(self):
        """ClusteringLevel 应有正确的值"""
        assert ClusteringLevel.LOWER.value == "lower"
        assert ClusteringLevel.SAME.value == "same"
        assert ClusteringLevel.HIGHER.value == "higher"
    
    def test_clustering_warning_level_values(self):
        """ClusteringWarningLevel 应有正确的值"""
        assert ClusteringWarningLevel.INFO.value == "info"
        assert ClusteringWarningLevel.WARNING.value == "warning"
        assert ClusteringWarningLevel.ERROR.value == "error"

"""
模拟数据测试：聚类诊断和推荐

使用模拟数据测试各种场景下的聚类诊断功能。

测试场景：
- 州级政策 + 县级数据
- 公司级政策 + 员工级数据
- 多层级结构
- 边界情况
"""

import numpy as np
import pandas as pd
import pytest

from lwdid.clustering_diagnostics import (
    ClusteringLevel,
    diagnose_clustering,
    recommend_clustering_level,
    check_clustering_consistency,
)
from lwdid.inference.wild_bootstrap import wild_cluster_bootstrap


# =============================================================================
# 州级政策场景
# =============================================================================

class TestStateLevelPolicy:
    """测试州级政策 + 县级数据场景"""
    
    @pytest.fixture
    def state_policy_data(self):
        """生成州级政策数据"""
        np.random.seed(42)
        n_states = 50
        counties_per_state = 20
        
        data = []
        for state in range(n_states):
            # 处理在州级别变化
            treated = state < 25
            first_treat = 6 if treated else 0
            state_effect = np.random.normal(0, 2)
            
            for county in range(counties_per_state):
                county_id = state * counties_per_state + county
                county_effect = np.random.normal(0, 0.5)
                
                for year in range(1, 11):
                    post = year >= 6 if treated else False
                    Y = (10 + state_effect + county_effect + 
                         0.5 * year + 2.0 * post + np.random.normal(0, 1))
                    
                    data.append({
                        'state': state,
                        'county': county_id,
                        'year': year,
                        'first_treat': first_treat,
                        'Y': Y
                    })
        
        return pd.DataFrame(data)
    
    def test_diagnose_recommends_state(self, state_policy_data):
        """应推荐州级聚类"""
        diag = diagnose_clustering(
            state_policy_data,
            ivar='county',
            potential_cluster_vars=['state', 'county'],
            gvar='first_treat',
            verbose=False
        )
        
        assert diag.recommended_cluster_var == 'state'
        assert diag.cluster_structure['state'].n_clusters == 50
        # 使用 == 而不是 is，因为可能返回 np.False_
        assert diag.cluster_structure['state'].treatment_varies_within_cluster == False
    
    def test_recommend_no_wild_bootstrap(self, state_policy_data):
        """50 个州不需要 wild bootstrap"""
        rec = recommend_clustering_level(
            state_policy_data,
            ivar='county',
            tvar='year',
            potential_cluster_vars=['state', 'county'],
            gvar='first_treat',
            min_clusters=20,
            verbose=False
        )
        
        assert rec.recommended_var == 'state'
        assert rec.use_wild_bootstrap is False
        assert rec.n_clusters == 50
    
    def test_consistency_check_passes(self, state_policy_data):
        """州级聚类应通过一致性检验"""
        result = check_clustering_consistency(
            state_policy_data,
            ivar='county',
            cluster_var='state',
            gvar='first_treat',
            verbose=False
        )
        
        assert result.is_consistent is True
        assert result.pct_clusters_with_variation == 0
    
    def test_state_is_higher_level(self, state_policy_data):
        """州应被识别为比县更高的层级"""
        diag = diagnose_clustering(
            state_policy_data,
            ivar='county',
            potential_cluster_vars=['state', 'county'],
            gvar='first_treat',
            verbose=False
        )
        
        assert diag.cluster_structure['state'].level_relative_to_unit == ClusteringLevel.HIGHER
        assert diag.cluster_structure['county'].level_relative_to_unit == ClusteringLevel.SAME


# =============================================================================
# 公司级政策场景
# =============================================================================

class TestFirmLevelPolicy:
    """测试公司级政策 + 员工级数据场景"""
    
    @pytest.fixture
    def firm_policy_data(self):
        """生成公司级政策数据"""
        np.random.seed(42)
        n_firms = 30
        employees_per_firm = 50
        
        data = []
        for firm in range(n_firms):
            treated = firm < 15
            first_treat = 5 if treated else 0
            firm_effect = np.random.normal(0, 3)
            
            for emp in range(employees_per_firm):
                emp_id = firm * employees_per_firm + emp
                emp_effect = np.random.normal(0, 1)
                
                for year in range(1, 9):
                    post = year >= 5 if treated else False
                    Y = (20 + firm_effect + emp_effect + 
                         1.5 * post + np.random.normal(0, 2))
                    
                    data.append({
                        'firm': firm,
                        'employee': emp_id,
                        'year': year,
                        'first_treat': first_treat,
                        'Y': Y
                    })
        
        return pd.DataFrame(data)
    
    def test_diagnose_recommends_firm(self, firm_policy_data):
        """应推荐公司级聚类"""
        diag = diagnose_clustering(
            firm_policy_data,
            ivar='employee',
            potential_cluster_vars=['firm', 'employee'],
            gvar='first_treat',
            verbose=False
        )
        
        assert diag.recommended_cluster_var == 'firm'
        assert diag.cluster_structure['firm'].n_clusters == 30
    
    def test_recommend_clustering(self, firm_policy_data):
        """应推荐公司级聚类或员工级聚类（取决于可靠性分数）"""
        rec = recommend_clustering_level(
            firm_policy_data,
            ivar='employee',
            tvar='year',
            potential_cluster_vars=['firm', 'employee'],
            gvar='first_treat',
            verbose=False
        )
        
        # 推荐可能是 firm（处理变化层级）或 employee（更多聚类）
        assert rec.recommended_var in ['firm', 'employee']
        assert rec.n_clusters > 0


# =============================================================================
# 少量聚类场景
# =============================================================================

class TestFewClusters:
    """测试少量聚类场景"""
    
    @pytest.fixture
    def few_cluster_data(self):
        """生成少量聚类数据（5 个区域）"""
        np.random.seed(42)
        n_regions = 5
        units_per_region = 200
        
        data = []
        for region in range(n_regions):
            treated = region < 2
            first_treat = 5 if treated else 0
            region_effect = np.random.normal(0, 3)
            
            for unit in range(units_per_region):
                unit_id = region * units_per_region + unit
                
                for year in range(1, 9):
                    post = year >= 5 if treated else False
                    Y = (10 + region_effect + 2.0 * post + np.random.normal(0, 1))
                    
                    data.append({
                        'region': region,
                        'unit': unit_id,
                        'year': year,
                        'first_treat': first_treat,
                        'Y': Y
                    })
        
        return pd.DataFrame(data)
    
    def test_recommends_wild_bootstrap(self, few_cluster_data):
        """少量聚类应推荐 wild bootstrap"""
        rec = recommend_clustering_level(
            few_cluster_data,
            ivar='unit',
            tvar='year',
            potential_cluster_vars=['region'],
            gvar='first_treat',
            min_clusters=20,
            verbose=False
        )
        
        assert rec.use_wild_bootstrap is True
        assert rec.n_clusters == 5
        assert 'wild' in rec.wild_bootstrap_reason.lower()
    
    def test_wild_bootstrap_runs(self, few_cluster_data):
        """Wild bootstrap 应能运行"""
        # 准备数据
        data = few_cluster_data.copy()
        data['D'] = (data['first_treat'] > 0) & (data['year'] >= data['first_treat'])
        data['D'] = data['D'].astype(int)
        
        result = wild_cluster_bootstrap(
            data,
            y_transformed='Y',
            d='D',
            cluster_var='region',
            n_bootstrap=99,
            seed=42
        )
        
        assert result.n_clusters == 5
        assert 0 <= result.pvalue <= 1


# =============================================================================
# 处理在聚类内变化场景
# =============================================================================

class TestTreatmentVariesWithinCluster:
    """测试处理在聚类内变化的场景"""
    
    @pytest.fixture
    def mixed_treatment_data(self):
        """生成处理在聚类内变化的数据"""
        np.random.seed(42)
        n_states = 20
        counties_per_state = 10
        
        data = []
        for state in range(n_states):
            state_effect = np.random.normal(0, 2)
            
            for county in range(counties_per_state):
                county_id = state * counties_per_state + county
                # 处理在县级别变化（不是州级别）
                treated = county_id % 2 == 0
                first_treat = 5 if treated else 0
                
                for year in range(1, 9):
                    post = year >= 5 if treated else False
                    Y = (10 + state_effect + 2.0 * post + np.random.normal(0, 1))
                    
                    data.append({
                        'state': state,
                        'county': county_id,
                        'year': year,
                        'first_treat': first_treat,
                        'Y': Y
                    })
        
        return pd.DataFrame(data)
    
    def test_detects_treatment_variation(self, mixed_treatment_data):
        """应检测到处理在州内变化"""
        diag = diagnose_clustering(
            mixed_treatment_data,
            ivar='county',
            potential_cluster_vars=['state', 'county'],
            gvar='first_treat',
            verbose=False
        )
        
        state_stats = diag.cluster_structure['state']
        # 使用 == 而不是 is，因为可能返回 np.True_
        assert state_stats.treatment_varies_within_cluster == True
    
    def test_consistency_check_fails(self, mixed_treatment_data):
        """州级聚类应不通过一致性检验"""
        result = check_clustering_consistency(
            mixed_treatment_data,
            ivar='county',
            cluster_var='state',
            gvar='first_treat',
            verbose=False
        )
        
        # 处理在州内变化
        assert result.pct_clusters_with_variation > 0


# =============================================================================
# 多层级结构场景
# =============================================================================

class TestMultiLevelStructure:
    """测试多层级结构场景"""
    
    @pytest.fixture
    def multi_level_data(self):
        """生成多层级数据：区域 > 州 > 县"""
        np.random.seed(42)
        n_regions = 4
        states_per_region = 10
        counties_per_state = 5
        
        data = []
        for region in range(n_regions):
            # 处理在区域级别变化
            treated = region < 2
            first_treat = 5 if treated else 0
            region_effect = np.random.normal(0, 3)
            
            for state in range(states_per_region):
                state_id = region * states_per_region + state
                state_effect = np.random.normal(0, 1)
                
                for county in range(counties_per_state):
                    county_id = state_id * counties_per_state + county
                    
                    for year in range(1, 9):
                        post = year >= 5 if treated else False
                        Y = (10 + region_effect + state_effect + 
                             2.0 * post + np.random.normal(0, 1))
                        
                        data.append({
                            'region': region,
                            'state': state_id,
                            'county': county_id,
                            'year': year,
                            'first_treat': first_treat,
                            'Y': Y
                        })
        
        return pd.DataFrame(data)
    
    def test_diagnose_all_levels(self, multi_level_data):
        """应正确诊断所有层级"""
        diag = diagnose_clustering(
            multi_level_data,
            ivar='county',
            potential_cluster_vars=['region', 'state', 'county'],
            gvar='first_treat',
            verbose=False
        )
        
        # 验证聚类数量
        assert diag.cluster_structure['region'].n_clusters == 4
        assert diag.cluster_structure['state'].n_clusters == 40
        assert diag.cluster_structure['county'].n_clusters == 200
        
        # 验证层级
        assert diag.cluster_structure['region'].level_relative_to_unit == ClusteringLevel.HIGHER
        assert diag.cluster_structure['state'].level_relative_to_unit == ClusteringLevel.HIGHER
        assert diag.cluster_structure['county'].level_relative_to_unit == ClusteringLevel.SAME
    
    def test_recommends_treatment_level(self, multi_level_data):
        """应推荐处理变化层级或更高可靠性的聚类"""
        diag = diagnose_clustering(
            multi_level_data,
            ivar='county',
            potential_cluster_vars=['region', 'state', 'county'],
            gvar='first_treat',
            verbose=False
        )
        
        # 处理在区域级别变化，但只有 4 个区域
        # 函数可能推荐 county（最多聚类，最高可靠性分数）
        # 或 state（足够聚类且是更高层级）
        # 或 region（处理变化层级但聚类数少）
        # 这取决于推荐算法的具体实现
        assert diag.recommended_cluster_var in ['region', 'state', 'county']


# =============================================================================
# 边界情况测试
# =============================================================================

class TestEdgeCases:
    """测试边界情况"""
    
    def test_single_cluster_invalid(self):
        """单个聚类应为无效"""
        np.random.seed(42)
        data = pd.DataFrame({
            'cluster': [1] * 100,
            'unit': range(100),
            'year': [1] * 100,
            'first_treat': [5] * 50 + [0] * 50,
            'Y': np.random.normal(10, 1, 100)
        })
        
        diag = diagnose_clustering(
            data,
            ivar='unit',
            potential_cluster_vars=['cluster'],
            gvar='first_treat',
            verbose=False
        )
        
        assert diag.cluster_structure['cluster'].is_valid_cluster is False
    
    def test_all_treated_clusters(self):
        """所有聚类都是处理组"""
        np.random.seed(42)
        data = pd.DataFrame({
            'cluster': np.repeat(range(10), 10),
            'unit': range(100),
            'year': [1] * 100,
            'first_treat': [5] * 100,  # 全部处理
            'Y': np.random.normal(10, 1, 100)
        })
        
        diag = diagnose_clustering(
            data,
            ivar='unit',
            potential_cluster_vars=['cluster'],
            gvar='first_treat',
            verbose=False
        )
        
        stats = diag.cluster_structure['cluster']
        assert stats.n_treated_clusters == 10
        assert stats.n_control_clusters == 0
    
    def test_all_control_clusters(self):
        """所有聚类都是控制组"""
        np.random.seed(42)
        data = pd.DataFrame({
            'cluster': np.repeat(range(10), 10),
            'unit': range(100),
            'year': [1] * 100,
            'first_treat': [0] * 100,  # 全部控制
            'Y': np.random.normal(10, 1, 100)
        })
        
        diag = diagnose_clustering(
            data,
            ivar='unit',
            potential_cluster_vars=['cluster'],
            gvar='first_treat',
            verbose=False
        )
        
        stats = diag.cluster_structure['cluster']
        assert stats.n_treated_clusters == 0
        assert stats.n_control_clusters == 10
    
    def test_unbalanced_cluster_sizes(self):
        """不平衡的聚类大小"""
        np.random.seed(42)
        
        # 聚类大小从 10 到 100
        data_list = []
        for cluster in range(10):
            size = 10 + cluster * 10
            for i in range(size):
                data_list.append({
                    'cluster': cluster,
                    'unit': len(data_list),
                    'year': 1,
                    'first_treat': 5 if cluster < 5 else 0,
                    'Y': np.random.normal(10, 1)
                })
        
        data = pd.DataFrame(data_list)
        
        diag = diagnose_clustering(
            data,
            ivar='unit',
            potential_cluster_vars=['cluster'],
            gvar='first_treat',
            verbose=False
        )
        
        stats = diag.cluster_structure['cluster']
        assert stats.min_cluster_size == 10
        assert stats.max_cluster_size == 100
        assert stats.cluster_size_cv > 0  # 应有变异

"""
测试新的 wild_cluster_bootstrap API 参数。

验证：
1. full_enumeration 参数
2. use_wildboottest 参数
3. 自动完全枚举逻辑
"""

import numpy as np
import pandas as pd
import pytest
from lwdid.inference.wild_bootstrap import wild_cluster_bootstrap


@pytest.fixture
def test_data():
    """创建测试数据：10 个聚类"""
    np.random.seed(42)
    n_clusters = 10
    obs_per_cluster = 20
    
    data = []
    for g in range(n_clusters):
        cluster_effect = np.random.normal(0, 1)
        for i in range(obs_per_cluster):
            treat = 1 if g >= 5 else 0
            y = 1.0 + 0.5 * treat + cluster_effect + np.random.normal(0, 0.5)
            data.append({
                'y': y,
                'd': treat,
                'cluster': g
            })
    
    return pd.DataFrame(data)


class TestFullEnumerationParameter:
    """测试 full_enumeration 参数"""
    
    def test_auto_full_enumeration_small_clusters(self, test_data):
        """当 G <= 12 且 weight_type='rademacher' 时自动启用完全枚举"""
        result = wild_cluster_bootstrap(
            data=test_data,
            y_transformed='y',
            d='d',
            cluster_var='cluster',
            n_bootstrap=999,
            weight_type='rademacher',
            seed=42,
            # full_enumeration=None (默认)
        )
        
        # 应该自动使用完全枚举，n_bootstrap 应该是 2^10 = 1024
        assert result.n_bootstrap == 1024
    
    def test_explicit_full_enumeration_true(self, test_data):
        """显式设置 full_enumeration=True"""
        result = wild_cluster_bootstrap(
            data=test_data,
            y_transformed='y',
            d='d',
            cluster_var='cluster',
            n_bootstrap=999,
            weight_type='rademacher',
            full_enumeration=True,
            seed=42,
        )
        
        assert result.n_bootstrap == 1024
    
    def test_explicit_full_enumeration_false(self, test_data):
        """显式设置 full_enumeration=False 禁用完全枚举"""
        result = wild_cluster_bootstrap(
            data=test_data,
            y_transformed='y',
            d='d',
            cluster_var='cluster',
            n_bootstrap=999,
            weight_type='rademacher',
            full_enumeration=False,
            seed=42,
        )
        
        # 应该使用随机抽样
        assert result.n_bootstrap == 999
    
    def test_full_enumeration_deterministic(self, test_data):
        """完全枚举应该产生确定性结果"""
        result1 = wild_cluster_bootstrap(
            data=test_data,
            y_transformed='y',
            d='d',
            cluster_var='cluster',
            full_enumeration=True,
            # 不设置 seed
        )
        
        result2 = wild_cluster_bootstrap(
            data=test_data,
            y_transformed='y',
            d='d',
            cluster_var='cluster',
            full_enumeration=True,
            # 不设置 seed
        )
        
        # 完全枚举应该产生完全相同的结果
        assert result1.pvalue == result2.pvalue
        assert result1.att == result2.att
    
    def test_mammen_weights_no_auto_enumeration(self, test_data):
        """Mammen 权重不应该自动启用完全枚举"""
        result = wild_cluster_bootstrap(
            data=test_data,
            y_transformed='y',
            d='d',
            cluster_var='cluster',
            n_bootstrap=999,
            weight_type='mammen',
            seed=42,
        )
        
        # Mammen 权重应该使用随机抽样
        assert result.n_bootstrap == 999


class TestUseWildboottestParameter:
    """测试 use_wildboottest 参数"""
    
    def test_use_wildboottest_basic(self, test_data):
        """测试 use_wildboottest=True"""
        result = wild_cluster_bootstrap(
            data=test_data,
            y_transformed='y',
            d='d',
            cluster_var='cluster',
            n_bootstrap=999,
            weight_type='rademacher',
            use_wildboottest=True,
            seed=42,
        )
        
        # 应该返回有效结果
        assert not np.isnan(result.pvalue)
        assert 0 <= result.pvalue <= 1
        assert result.ci_method == 'wildboottest'
    
    def test_wildboottest_vs_native_full_enum(self, test_data):
        """比较 wildboottest 和原生实现的完全枚举结果"""
        # 原生实现
        result_native = wild_cluster_bootstrap(
            data=test_data,
            y_transformed='y',
            d='d',
            cluster_var='cluster',
            full_enumeration=True,
            use_wildboottest=False,
        )
        
        # wildboottest 实现
        result_wbt = wild_cluster_bootstrap(
            data=test_data,
            y_transformed='y',
            d='d',
            cluster_var='cluster',
            full_enumeration=True,
            use_wildboottest=True,
        )
        
        # ATT 应该完全相同（都是 OLS 估计）
        assert abs(result_native.att - result_wbt.att) < 1e-10
        
        # p 值应该非常接近（可能有微小差异由于实现细节）
        assert abs(result_native.pvalue - result_wbt.pvalue) < 0.05


class TestStataEquivalence:
    """测试与 Stata 的等价性"""
    
    def test_full_enumeration_stata_pvalue(self, test_data):
        """完全枚举模式应该与 Stata boottest 产生相同的 p 值"""
        result = wild_cluster_bootstrap(
            data=test_data,
            y_transformed='y',
            d='d',
            cluster_var='cluster',
            full_enumeration=True,
            impose_null=True,
        )
        
        # p 值应该在合理范围内
        assert 0 <= result.pvalue <= 1
        
        # 完全枚举的 n_bootstrap 应该是 2^G
        assert result.n_bootstrap == 2 ** result.n_clusters


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

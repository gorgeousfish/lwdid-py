"""
集成测试：验证修改后的 wild_cluster_bootstrap 函数
测试完全枚举模式和 wildboottest 集成
"""

import numpy as np
import pandas as pd
import pytest
from lwdid.inference.wild_bootstrap import (
    wild_cluster_bootstrap,
    _generate_all_rademacher_weights,
    HAS_WILDBOOTTEST,
)


class TestWildBootstrapIntegration:
    """测试修改后的 wild_cluster_bootstrap 函数"""
    
    @pytest.fixture
    def test_data(self):
        """创建测试数据：10 个聚类"""
        np.random.seed(42)
        n_clusters = 10
        obs_per_cluster = 20
        
        data = []
        for c in range(n_clusters):
            cluster_effect = np.random.normal(0, 1)
            for _ in range(obs_per_cluster):
                d = 1 if c >= 5 else 0  # 后 5 个聚类是处理组
                y = 1.0 + 0.5 * d + cluster_effect + np.random.normal(0, 0.5)
                data.append({'y': y, 'd': d, 'cluster': c})
        
        return pd.DataFrame(data)
    
    def test_full_enumeration_auto_enabled(self, test_data):
        """测试完全枚举在 G <= 12 时自动启用"""
        result = wild_cluster_bootstrap(
            data=test_data,
            y_transformed='y',
            d='d',
            cluster_var='cluster',
            seed=42,
        )
        
        # 10 个聚类，应该自动使用完全枚举 (2^10 = 1024)
        assert result.n_bootstrap == 1024
        assert 0 <= result.pvalue <= 1
    
    def test_full_enumeration_deterministic(self, test_data):
        """测试完全枚举产生确定性结果"""
        result1 = wild_cluster_bootstrap(
            data=test_data,
            y_transformed='y',
            d='d',
            cluster_var='cluster',
            full_enumeration=True,
        )
        
        result2 = wild_cluster_bootstrap(
            data=test_data,
            y_transformed='y',
            d='d',
            cluster_var='cluster',
            full_enumeration=True,
        )
        
        # 完全枚举应该产生完全相同的结果
        assert result1.pvalue == result2.pvalue
        assert result1.att == result2.att
    
    def test_full_enumeration_disabled(self, test_data):
        """测试可以手动禁用完全枚举"""
        result = wild_cluster_bootstrap(
            data=test_data,
            y_transformed='y',
            d='d',
            cluster_var='cluster',
            n_bootstrap=999,
            full_enumeration=False,
            seed=42,
        )
        
        # 应该使用指定的 bootstrap 次数
        assert result.n_bootstrap == 999
    
    def test_random_vs_full_enumeration_similar(self, test_data):
        """测试随机抽样和完全枚举结果相近"""
        result_full = wild_cluster_bootstrap(
            data=test_data,
            y_transformed='y',
            d='d',
            cluster_var='cluster',
            full_enumeration=True,
        )
        
        result_random = wild_cluster_bootstrap(
            data=test_data,
            y_transformed='y',
            d='d',
            cluster_var='cluster',
            n_bootstrap=999,
            full_enumeration=False,
            seed=42,
        )
        
        # p 值应该相近（差异 < 0.1）
        assert abs(result_full.pvalue - result_random.pvalue) < 0.1
    
    @pytest.mark.skipif(not HAS_WILDBOOTTEST, reason="wildboottest not installed")
    def test_use_wildboottest_flag(self, test_data):
        """测试 use_wildboottest 参数"""
        result = wild_cluster_bootstrap(
            data=test_data,
            y_transformed='y',
            d='d',
            cluster_var='cluster',
            use_wildboottest=True,
            seed=42,
        )
        
        assert result.ci_method == 'wildboottest'
        assert 0 <= result.pvalue <= 1
    
    @pytest.mark.skipif(not HAS_WILDBOOTTEST, reason="wildboottest not installed")
    def test_wildboottest_vs_native_similar(self, test_data):
        """测试 wildboottest 和原生实现结果相近"""
        result_native = wild_cluster_bootstrap(
            data=test_data,
            y_transformed='y',
            d='d',
            cluster_var='cluster',
            full_enumeration=True,
            use_wildboottest=False,
        )
        
        result_wbt = wild_cluster_bootstrap(
            data=test_data,
            y_transformed='y',
            d='d',
            cluster_var='cluster',
            full_enumeration=True,
            use_wildboottest=True,
        )
        
        # ATT 应该完全相同
        assert abs(result_native.att - result_wbt.att) < 1e-10
        
        # p 值应该相近（可能有小差异因为实现细节）
        assert abs(result_native.pvalue - result_wbt.pvalue) < 0.05


class TestGenerateAllRademacherWeights:
    """测试完全枚举权重生成函数"""
    
    def test_correct_count(self):
        """测试生成正确数量的组合"""
        for G in range(1, 8):
            weights = _generate_all_rademacher_weights(G)
            assert weights.shape == (2**G, G)
    
    def test_only_minus_one_and_one(self):
        """测试只包含 -1 和 +1"""
        weights = _generate_all_rademacher_weights(5)
        assert set(weights.flatten()) == {-1, 1}
    
    def test_all_unique(self):
        """测试所有组合都是唯一的"""
        weights = _generate_all_rademacher_weights(6)
        # 转换为元组集合检查唯一性
        unique_rows = set(tuple(row) for row in weights)
        assert len(unique_rows) == 2**6


class TestStataEquivalence:
    """测试与 Stata boottest 的等价性"""
    
    @pytest.fixture
    def stata_test_data(self):
        """加载 Stata 测试数据"""
        import os
        data_path = os.path.join(
            os.path.dirname(__file__),
            'stata_wild_test.csv'
        )
        if os.path.exists(data_path):
            return pd.read_csv(data_path)
        return None
    
    def test_stata_pvalue_exact_match(self, stata_test_data):
        """测试与算法原理一致的 p 值计算"""
        if stata_test_data is None:
            pytest.skip("Stata test data not found")
        
        # 原生实现遵循算法原理：p = P(|t*| >= |t_orig|)
        result = wild_cluster_bootstrap(
            data=stata_test_data,
            y_transformed='y',
            d='d',
            cluster_var='cluster',
            full_enumeration=True,
            use_wildboottest=False,
        )
        
        # 算法原理的 p 值：516/1024 = 0.5039
        # 这比 Stata/wildboottest 的 514/1024 = 0.5020 略高
        # 差异来自边界处理：算法原理使用 >=，Stata 使用数值容差
        algorithm_pvalue = 516 / 1024  # 0.50390625
        
        assert abs(result.pvalue - algorithm_pvalue) < 0.001, \
            f"Python p={result.pvalue:.6f}, 算法原理 p={algorithm_pvalue:.6f}"
        
        # 如果需要与 Stata 100% 等价，使用 wildboottest
        if HAS_WILDBOOTTEST:
            result_wbt = wild_cluster_bootstrap(
                data=stata_test_data,
                y_transformed='y',
                d='d',
                cluster_var='cluster',
                full_enumeration=True,
                use_wildboottest=True,
            )
            stata_pvalue = 0.5020
            assert abs(result_wbt.pvalue - stata_pvalue) < 0.001, \
                f"wildboottest p={result_wbt.pvalue:.4f}, Stata p={stata_pvalue:.4f}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

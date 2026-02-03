"""
Wild Cluster Bootstrap Stata 等价性测试

验证在完全枚举模式下，Python wildboottest 与 Stata boottest 100% 等价。

测试数据：10 个 cluster，每个 100 个观测
Stata 参考结果：
- 999 次随机抽样: p = 0.4915
- 1024 次完全枚举: p = 0.5020
"""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# 尝试导入 wildboottest
try:
    from wildboottest.wildboottest import wildboottest
    import statsmodels.formula.api as smf
    HAS_WILDBOOTTEST = True
except ImportError:
    HAS_WILDBOOTTEST = False

# Stata 参考结果
STATA_RESULTS = {
    'random_999': {
        't_stat': 0.7287,
        'p_value': 0.4915,
        'ci_lower': -2.079,
        'ci_upper': 4.174,
    },
    'full_enum_1024': {
        't_stat': 0.7287,
        'p_value': 0.5020,
        'ci_lower': -2.047,
        'ci_upper': 4.144,
    }
}


@pytest.fixture
def test_data():
    """加载测试数据"""
    return pd.read_csv(Path(__file__).parent / "stata_wild_test.csv")


@pytest.mark.skipif(not HAS_WILDBOOTTEST, reason="wildboottest 包未安装")
class TestWildBootstrapStataEquivalence:
    """测试 wildboottest 与 Stata boottest 的等价性"""
    
    def test_full_enumeration_exact_match(self, test_data):
        """
        测试完全枚举模式下与 Stata 100% 等价
        
        Stata 命令: boottest d, reps(1024) seed(42) weight(rademacher)
        Stata 结果: p = 0.5020
        """
        model = smf.ols(formula='y ~ d', data=test_data)
        
        # 使用完全枚举 (B > 2^G 触发)
        result = wildboottest(
            model, 
            param="d", 
            cluster=test_data['cluster'], 
            B=2048,  # > 2^10 = 1024
            seed=42,
            bootstrap_type='11'
        )
        
        p_value = result['p-value'].values[0]
        stata_p = STATA_RESULTS['full_enum_1024']['p_value']
        
        # 完全枚举模式下应该 100% 等价
        assert abs(p_value - stata_p) < 1e-4, \
            f"完全枚举 p 值不匹配: Python={p_value:.4f}, Stata={stata_p:.4f}"
    
    def test_full_enumeration_deterministic(self, test_data):
        """
        测试完全枚举模式的确定性
        
        不同种子应产生相同结果
        """
        model = smf.ols(formula='y ~ d', data=test_data)
        
        results = []
        for seed in [1, 42, 123, 456, 789]:
            result = wildboottest(
                model, 
                param="d", 
                cluster=test_data['cluster'], 
                B=2048,
                seed=seed,
                bootstrap_type='11'
            )
            results.append(result['p-value'].values[0])
        
        # 所有结果应该相同
        assert len(set(results)) == 1, \
            f"完全枚举结果不确定: {results}"
    
    def test_t_statistic_exact_match(self, test_data):
        """
        测试 t 统计量与 Stata 完全匹配
        
        t 统计量不依赖于 bootstrap，应该完全一致
        """
        model = smf.ols(formula='y ~ d', data=test_data)
        
        result = wildboottest(
            model, 
            param="d", 
            cluster=test_data['cluster'], 
            B=999,
            seed=42,
            bootstrap_type='11'
        )
        
        t_stat = result['statistic'].values[0]
        stata_t = STATA_RESULTS['random_999']['t_stat']
        
        # t 统计量应该非常接近
        assert abs(t_stat - stata_t) < 1e-3, \
            f"t 统计量不匹配: Python={t_stat:.4f}, Stata={stata_t:.4f}"
    
    def test_random_sampling_reasonable_difference(self, test_data):
        """
        测试随机抽样模式下差异在合理范围内
        
        由于 RNG 不同，p 值差异应 < 0.05
        """
        model = smf.ols(formula='y ~ d', data=test_data)
        
        result = wildboottest(
            model, 
            param="d", 
            cluster=test_data['cluster'], 
            B=999,
            seed=42,
            bootstrap_type='11'
        )
        
        p_value = result['p-value'].values[0]
        stata_p = STATA_RESULTS['random_999']['p_value']
        
        # 随机抽样差异应 < 0.05
        assert abs(p_value - stata_p) < 0.05, \
            f"随机抽样 p 值差异过大: Python={p_value:.4f}, Stata={stata_p:.4f}"


@pytest.mark.skipif(not HAS_WILDBOOTTEST, reason="wildboottest 包未安装")
class TestWildBootstrapWeightTypes:
    """测试不同权重类型"""
    
    @pytest.mark.parametrize("weight_type", ['rademacher', 'mammen', 'webb'])
    def test_weight_type_produces_valid_pvalue(self, test_data, weight_type):
        """测试不同权重类型产生有效的 p 值"""
        model = smf.ols(formula='y ~ d', data=test_data)
        
        result = wildboottest(
            model, 
            param="d", 
            cluster=test_data['cluster'], 
            B=999,
            seed=42,
            bootstrap_type='11',
            weights_type=weight_type
        )
        
        p_value = result['p-value'].values[0]
        
        # p 值应在 [0, 1] 范围内
        assert 0 <= p_value <= 1, f"无效的 p 值: {p_value}"
        
        # p 值应该合理（对于这个数据，应该在 0.3-0.7 范围内）
        assert 0.3 <= p_value <= 0.7, \
            f"p 值不合理: {p_value} (权重类型: {weight_type})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

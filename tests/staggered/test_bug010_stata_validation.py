"""
BUG-010 修复后的数值验证测试

验证目标：
1. RA/IPWRA 估计器在修复后的数值正确性
2. 使用模拟数据验证估计器行为
3. 验证负方差修复没有破坏现有功能
"""

import warnings
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

# 测试数据路径
TEST_DATA_DIR = Path(__file__).parent.parent / "data"


class TestRAEstimatorValidation:
    """RA 估计器验证测试"""
    
    def test_ra_simulated_known_effect(self):
        """使用已知真实效应的模拟数据测试 RA"""
        from lwdid.staggered.estimators import estimate_ra
        
        np.random.seed(42)
        n = 200
        x = np.random.randn(n)
        d = np.array([1] * 100 + [0] * 100)
        
        # 真实效应 = 3.0
        true_effect = 3.0
        y = 1.0 + 0.5 * x + np.random.randn(n) * 0.5 + d * true_effect
        
        df = pd.DataFrame({'y': y, 'd': d, 'x': x})
        result = estimate_ra(df, 'y', 'd', ['x'])
        
        # 验证 ATT 在合理范围内（真实值 ± 1 SE 范围应包含真实效应）
        assert result.se > 0, "SE 应为正数"
        assert abs(result.att - true_effect) < 3 * result.se, f"ATT={result.att:.3f} 应接近真实效应 {true_effect}"
        
        # 验证置信区间合理
        assert result.ci_lower < result.att < result.ci_upper
    
    def test_ra_different_sample_sizes(self):
        """测试不同样本量下 RA 的稳定性"""
        from lwdid.staggered.estimators import estimate_ra
        
        results = {}
        for n in [50, 100, 200]:
            np.random.seed(123)
            x = np.random.randn(n)
            d = np.array([1] * (n // 2) + [0] * (n // 2))
            y = 1.0 + x + np.random.randn(n) * 0.5 + d * 2.0
            
            df = pd.DataFrame({'y': y, 'd': d, 'x': x})
            result = estimate_ra(df, 'y', 'd', ['x'])
            results[n] = {'att': result.att, 'se': result.se}
        
        # 验证所有 SE 都是正数
        for n, res in results.items():
            assert res['se'] > 0, f"n={n}: SE 应为正数"
        
        # 验证 SE 随样本量增加而减小（大致趋势）
        assert results[200]['se'] < results[50]['se'], "更大样本量应有更小 SE"


class TestIPWRAEstimatorValidation:
    """IPWRA 估计器验证测试"""
    
    def test_ipwra_simulated_known_effect(self):
        """使用已知真实效应的模拟数据测试 IPWRA"""
        from lwdid.staggered.estimators import estimate_ipwra
        
        np.random.seed(2024)
        n = 200
        x = np.random.randn(n)
        
        # 真实倾向得分模型
        ps_true = 1 / (1 + np.exp(-0.5 * x))
        d = (np.random.uniform(0, 1, n) < ps_true).astype(int)
        
        # 真实效应 = 2.5
        true_effect = 2.5
        y = 1.0 + 0.5 * x + np.random.randn(n) * 0.5 + d * true_effect
        
        df = pd.DataFrame({'y': y, 'd': d, 'x': x})
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = estimate_ipwra(df, 'y', 'd', ['x'])
        
        # 验证基本属性
        assert result.se > 0, "SE 应为正数"
        assert abs(result.att - true_effect) < 3 * result.se, f"ATT={result.att:.3f} 应接近真实效应 {true_effect}"


class TestNegativeVarianceHandling:
    """验证负方差处理正确性"""
    
    def test_se_always_positive_ra(self):
        """验证 RA 估计器的 SE 总是正数"""
        from lwdid.staggered.estimators import estimate_ra
        
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        d = np.array([1] * 50 + [0] * 50)
        y = 1 + x + np.random.randn(n) * 0.5 + d * 2
        
        df = pd.DataFrame({'y': y, 'd': d, 'x': x})
        
        result = estimate_ra(df, 'y', 'd', ['x'])
        assert result.se > 0
    
    def test_se_always_positive_ipwra(self):
        """验证 IPWRA 估计器的 SE 总是正数"""
        from lwdid.staggered.estimators import estimate_ipwra
        
        np.random.seed(123)
        n = 100
        x = np.random.randn(n)
        d = np.array([1] * 50 + [0] * 50)
        y = 1 + x + np.random.randn(n) * 0.5 + d * 2
        
        df = pd.DataFrame({'y': y, 'd': d, 'x': x})
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = estimate_ipwra(df, 'y', 'd', ['x'])
        
        assert result.se > 0
    
    def test_warning_function_integration(self):
        """Verify warning function is correctly integrated in estimators"""
        from lwdid.staggered.estimators import _warn_negative_variance
        
        # Test warning function
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_negative_variance(
                var_value=-0.001,
                estimator_type='RA',
                n=50,
                n_treat=20,
                n_control=30,
                condition_number=1e8,
            )
            
            assert len(w) == 1
            msg = str(w[0].message)
            assert "RA estimator" in msg
            assert "condition number" in msg


class TestEstimatorComparisonRA_IPWRA:
    """比较 RA 和 IPWRA 估计器的结果"""
    
    def test_att_comparable_ra_ipwra(self):
        """验证 RA 和 IPWRA 的 ATT 估计在合理范围内"""
        from lwdid.staggered.estimators import estimate_ra, estimate_ipwra
        
        # 生成已知真实效应的模拟数据
        np.random.seed(2024)
        n = 300
        x = np.random.randn(n)
        # 倾向得分基于 x
        ps_true = 1 / (1 + np.exp(-0.5 * x))
        d = (np.random.uniform(0, 1, n) < ps_true).astype(int)
        # 真实 ATT = 2
        y = 1 + 0.5 * x + np.random.randn(n) * 0.5 + d * 2
        
        df = pd.DataFrame({'y': y, 'd': d, 'x': x})
        
        # 运行估计器
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ra_result = estimate_ra(df, 'y', 'd', ['x'])
            ipwra_result = estimate_ipwra(df, 'y', 'd', ['x'])
        
        # 验证两个估计器的 ATT 相差不大
        att_diff = abs(ra_result.att - ipwra_result.att)
        max_se = max(ra_result.se, ipwra_result.se)
        assert att_diff < 3 * max_se, f"RA ATT={ra_result.att:.3f}, IPWRA ATT={ipwra_result.att:.3f} 差异过大"
        
        # 验证所有 SE 都是正数
        assert ra_result.se > 0
        assert ipwra_result.se > 0


class TestMultipleSeedRobustness:
    """多种子鲁棒性测试"""
    
    def test_ra_multiple_seeds(self):
        """测试 RA 在多个随机种子下的稳定性"""
        from lwdid.staggered.estimators import estimate_ra
        
        results = []
        for seed in [1, 42, 123, 456, 789]:
            np.random.seed(seed)
            n = 150
            x = np.random.randn(n)
            d = np.array([1] * 75 + [0] * 75)
            y = 1 + x + np.random.randn(n) * 0.5 + d * 2
            
            df = pd.DataFrame({'y': y, 'd': d, 'x': x})
            result = estimate_ra(df, 'y', 'd', ['x'])
            results.append({
                'seed': seed,
                'att': result.att,
                'se': result.se
            })
        
        # 验证所有 SE 都是正数
        for res in results:
            assert res['se'] > 0, f"seed={res['seed']}: SE 应为正数"
        
        # 验证 ATT 估计的均值接近真实效应
        mean_att = np.mean([r['att'] for r in results])
        assert abs(mean_att - 2.0) < 0.5, f"平均 ATT={mean_att:.3f} 应接近真实效应 2.0"
    
    def test_ipwra_multiple_seeds(self):
        """测试 IPWRA 在多个随机种子下的稳定性"""
        from lwdid.staggered.estimators import estimate_ipwra
        
        results = []
        for seed in [1, 42, 123]:
            np.random.seed(seed)
            n = 150
            x = np.random.randn(n)
            ps_true = 1 / (1 + np.exp(-0.3 * x))
            d = (np.random.uniform(0, 1, n) < ps_true).astype(int)
            y = 1 + 0.5 * x + np.random.randn(n) * 0.5 + d * 2
            
            df = pd.DataFrame({'y': y, 'd': d, 'x': x})
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = estimate_ipwra(df, 'y', 'd', ['x'])
            
            results.append({
                'seed': seed,
                'att': result.att,
                'se': result.se
            })
        
        # 验证所有 SE 都是正数
        for res in results:
            assert res['se'] > 0, f"seed={res['seed']}: SE 应为正数"


class TestConditionNumberDiagnostics:
    """矩阵条件数诊断测试"""
    
    def test_condition_number_calculated(self):
        """验证条件数被正确计算"""
        from lwdid.staggered.estimators import _warn_negative_variance
        
        # 测试高条件数警告
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_negative_variance(
                var_value=-0.001,
                estimator_type='IPWRA',
                n=100,
                n_treat=50,
                n_control=50,
                weights_cv=2.5,
                condition_number=1e10,  # 非常高的条件数
            )
            
            msg = str(w[0].message)
            # 验证高条件数警告
            assert "1e+06" in msg or "条件数" in msg
            assert "1.00e+10" in msg or "1e+10" in msg


class TestSmallSampleBehavior:
    """小样本行为测试"""
    
    def test_ra_small_sample_warning(self):
        """测试小样本时 RA 仍能正确运行"""
        from lwdid.staggered.estimators import estimate_ra
        
        np.random.seed(42)
        n = 40  # 小样本
        x = np.random.randn(n)
        d = np.array([1] * 20 + [0] * 20)
        y = 1 + x + np.random.randn(n) * 0.5 + d * 2
        
        df = pd.DataFrame({'y': y, 'd': d, 'x': x})
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = estimate_ra(df, 'y', 'd', ['x'])
        
        assert result.se > 0
        assert not np.isnan(result.att)
    
    def test_ipwra_small_sample_warning(self):
        """测试小样本时 IPWRA 仍能正确运行"""
        from lwdid.staggered.estimators import estimate_ipwra
        
        np.random.seed(123)
        n = 50  # 小样本
        x = np.random.randn(n)
        d = np.array([1] * 25 + [0] * 25)
        y = 1 + x + np.random.randn(n) * 0.5 + d * 2
        
        df = pd.DataFrame({'y': y, 'd': d, 'x': x})
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = estimate_ipwra(df, 'y', 'd', ['x'])
        
        assert result.se > 0
        assert not np.isnan(result.att)

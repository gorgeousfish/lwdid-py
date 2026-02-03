# -*- coding: utf-8 -*-
"""
Monte Carlo框架属性测试

Property 3: 相同种子产生相同结果
Validates: Requirements 3.5

References
----------
Lee & Wooldridge (2025, 2026)
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add paths
fixtures_path = Path(__file__).parent.parent.parent / 'test_common_timing' / 'fixtures'
framework_path = Path(__file__).parent.parent / 'framework'
sys.path.insert(0, str(fixtures_path))
sys.path.insert(0, str(framework_path))


def simple_estimator(data, period, **kwargs):
    """
    简单估计器用于测试框架
    返回处理组和控制组post-treatment均值差
    """
    post_data = data[data['post'] == 1]
    treated = post_data[post_data['d'] == 1]['y'].mean()
    control = post_data[post_data['d'] == 0]['y'].mean()
    
    att = treated - control
    se = 1.0  # 简化的SE
    ci_lower = att - 1.96 * se
    ci_upper = att + 1.96 * se
    pvalue = 0.05
    
    return att, se, ci_lower, ci_upper, pvalue


class TestMonteCarloFrameworkReproducibility:
    """
    Property 3: 相同种子产生相同结果
    Validates: Requirements 3.5
    """
    
    def test_same_seed_produces_identical_results(self):
        """
        验证相同种子产生完全相同的Monte Carlo结果
        """
        from dgp_small_sample import generate_small_sample_dgp
        from runner import run_monte_carlo
        
        seed = 42
        n_reps = 10  # 小规模测试
        
        # 运行两次Monte Carlo
        result1 = run_monte_carlo(
            estimator_func=simple_estimator,
            dgp_func=generate_small_sample_dgp,
            n_reps=n_reps,
            dgp_kwargs={'scenario': 1},
            seed=seed,
            parallel=False,
            verbose=False,
            estimator_name='simple',
            dgp_name='small_sample',
            scenario='scenario_1',
        )
        
        result2 = run_monte_carlo(
            estimator_func=simple_estimator,
            dgp_func=generate_small_sample_dgp,
            n_reps=n_reps,
            dgp_kwargs={'scenario': 1},
            seed=seed,
            parallel=False,
            verbose=False,
            estimator_name='simple',
            dgp_name='small_sample',
            scenario='scenario_1',
        )
        
        # 验证结果相同
        assert np.isclose(result1.bias, result2.bias)
        assert np.isclose(result1.sd, result2.sd)
        assert np.isclose(result1.rmse, result2.rmse)
        assert np.isclose(result1.mean_att, result2.mean_att)
        assert np.allclose(result1.estimates, result2.estimates)
    
    def test_different_seeds_produce_different_results(self):
        """
        验证不同种子产生不同的结果
        """
        from dgp_small_sample import generate_small_sample_dgp
        from runner import run_monte_carlo
        
        n_reps = 10
        
        result1 = run_monte_carlo(
            estimator_func=simple_estimator,
            dgp_func=generate_small_sample_dgp,
            n_reps=n_reps,
            dgp_kwargs={'scenario': 1},
            seed=42,
            parallel=False,
            verbose=False,
            estimator_name='simple',
            dgp_name='small_sample',
            scenario='scenario_1',
        )
        
        result2 = run_monte_carlo(
            estimator_func=simple_estimator,
            dgp_func=generate_small_sample_dgp,
            n_reps=n_reps,
            dgp_kwargs={'scenario': 1},
            seed=43,
            parallel=False,
            verbose=False,
            estimator_name='simple',
            dgp_name='small_sample',
            scenario='scenario_1',
        )
        
        # 验证结果不同
        assert not np.allclose(result1.estimates, result2.estimates)


class TestMonteCarloFrameworkMetrics:
    """
    验证Monte Carlo框架的评估指标计算
    """
    
    def test_bias_calculation(self):
        """
        验证Bias计算: Bias = E[ATT_hat] - ATT_true
        """
        from dgp_small_sample import generate_small_sample_dgp
        from runner import run_monte_carlo
        
        result = run_monte_carlo(
            estimator_func=simple_estimator,
            dgp_func=generate_small_sample_dgp,
            n_reps=20,
            dgp_kwargs={'scenario': 1},
            seed=42,
            parallel=False,
            verbose=False,
            estimator_name='simple',
            dgp_name='small_sample',
            scenario='scenario_1',
        )
        
        # 验证Bias计算
        expected_bias = result.mean_att - result.true_att
        assert np.isclose(result.bias, expected_bias)
    
    def test_rmse_calculation(self):
        """
        验证RMSE计算: RMSE = sqrt(Bias² + SD²)
        """
        from dgp_small_sample import generate_small_sample_dgp
        from runner import run_monte_carlo
        
        result = run_monte_carlo(
            estimator_func=simple_estimator,
            dgp_func=generate_small_sample_dgp,
            n_reps=20,
            dgp_kwargs={'scenario': 1},
            seed=42,
            parallel=False,
            verbose=False,
            estimator_name='simple',
            dgp_name='small_sample',
            scenario='scenario_1',
        )
        
        # 验证RMSE计算
        expected_rmse = np.sqrt(result.bias**2 + result.sd**2)
        assert np.isclose(result.rmse, expected_rmse)
    
    def test_se_ratio_calculation(self):
        """
        验证SE Ratio计算: SE_Ratio = Mean_SE / SD
        """
        from dgp_small_sample import generate_small_sample_dgp
        from runner import run_monte_carlo
        
        result = run_monte_carlo(
            estimator_func=simple_estimator,
            dgp_func=generate_small_sample_dgp,
            n_reps=20,
            dgp_kwargs={'scenario': 1},
            seed=42,
            parallel=False,
            verbose=False,
            estimator_name='simple',
            dgp_name='small_sample',
            scenario='scenario_1',
        )
        
        # 验证SE Ratio计算
        if result.sd > 0:
            expected_se_ratio = result.mean_se / result.sd
            assert np.isclose(result.se_ratio, expected_se_ratio)


class TestMonteCarloFrameworkFailureHandling:
    """
    验证Monte Carlo框架的失败处理
    Validates: Requirements 3.4
    """
    
    def test_handles_estimation_failures(self):
        """
        验证框架正确处理估计失败
        """
        from dgp_small_sample import generate_small_sample_dgp
        from runner import run_monte_carlo
        
        def failing_estimator(data, period, **kwargs):
            """有时失败的估计器"""
            import random
            if random.random() < 0.3:  # 30%失败率
                raise ValueError("Estimation failed")
            return simple_estimator(data, period, **kwargs)
        
        result = run_monte_carlo(
            estimator_func=failing_estimator,
            dgp_func=generate_small_sample_dgp,
            n_reps=20,
            dgp_kwargs={'scenario': 1},
            seed=42,
            parallel=False,
            verbose=False,
            estimator_name='failing',
            dgp_name='small_sample',
            scenario='scenario_1',
        )
        
        # 验证记录了失败次数
        assert result.n_failed >= 0
        assert result.n_valid + result.n_failed == 20
        # 验证仍然产生了有效结果
        assert result.n_valid > 0


class TestMonteCarloResultsDataClass:
    """
    验证MonteCarloResults数据类
    Validates: Requirements 3.1
    """
    
    def test_summary_method(self):
        """
        验证summary()方法
        """
        from results import MonteCarloResults
        
        result = MonteCarloResults(
            estimator_name='test',
            dgp_type='small_sample',
            scenario='scenario_1',
            n_reps=100,
            n_units=20,
            bias=0.1,
            sd=1.5,
            rmse=1.503,
            coverage=0.95,
            mean_se=1.4,
            se_ratio=0.93,
            n_valid=100,
            n_failed=0,
            true_att=2.0,
            mean_att=2.1,
        )
        
        summary = result.summary()
        assert 'test' in summary
        assert 'Bias' in summary
        assert 'Coverage' in summary
    
    def test_to_dict_method(self):
        """
        验证to_dict()方法
        """
        from results import MonteCarloResults
        
        result = MonteCarloResults(
            estimator_name='test',
            dgp_type='small_sample',
            scenario='scenario_1',
            n_reps=100,
            n_units=20,
            bias=0.1,
            sd=1.5,
            rmse=1.503,
            coverage=0.95,
            mean_se=1.4,
            se_ratio=0.93,
        )
        
        d = result.to_dict()
        assert d['estimator'] == 'test'
        assert d['bias'] == 0.1
        assert d['coverage'] == 0.95


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

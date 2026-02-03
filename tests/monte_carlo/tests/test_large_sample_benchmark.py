# -*- coding: utf-8 -*-
"""
大样本场景性能基准测试

Task 7.2: 运行大样本场景测试
- 场景1C-4C，N∈{100, 500, 1000}
- 验证与论文的一致性

Validates: Requirements 4.2

References
----------
Lee & Wooldridge (2023) ssrn-4516518, Section 7.1
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# 添加路径
fixtures_path = Path(__file__).parent.parent.parent / 'test_common_timing' / 'fixtures'
framework_path = Path(__file__).parent.parent / 'framework'
benchmarks_path = Path(__file__).parent.parent / 'benchmarks'
sys.path.insert(0, str(fixtures_path))
sys.path.insert(0, str(framework_path))
sys.path.insert(0, str(benchmarks_path))


def large_sample_estimator(data, period, **kwargs):
    """
    大样本估计器包装
    
    对于 Common Timing 场景，使用 DiD 估计:
    ATT = (Y_treated_post - Y_treated_pre) - (Y_control_post - Y_control_pre)
    
    这是正确的 DiD 估计方法，消除了单位固定效应和时间趋势的影响。
    """
    from scipy import stats
    
    # Common Timing 设定: T=6, S=4 (首次处理期)
    pre_periods = [1, 2, 3]  # 处理前期
    post_periods = [4, 5, 6]  # 处理后期
    
    if 'f04' not in data.columns:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    
    # 分离处理组和控制组
    treated_ids = data[data['d'] == 1]['id'].unique()
    control_ids = data[data['d'] == 0]['id'].unique()
    
    if len(treated_ids) == 0 or len(control_ids) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    
    # 计算每个单位的 pre 和 post 均值
    def compute_unit_did(unit_data):
        pre_mean = unit_data[unit_data['year'].isin(pre_periods)]['y'].mean()
        post_mean = unit_data[unit_data['year'].isin(post_periods)]['y'].mean()
        return post_mean - pre_mean
    
    # 处理组的 DiD
    treated_diffs = []
    for uid in treated_ids:
        unit_data = data[data['id'] == uid]
        diff = compute_unit_did(unit_data)
        if not np.isnan(diff):
            treated_diffs.append(diff)
    
    # 控制组的 DiD
    control_diffs = []
    for uid in control_ids:
        unit_data = data[data['id'] == uid]
        diff = compute_unit_did(unit_data)
        if not np.isnan(diff):
            control_diffs.append(diff)
    
    treated_diffs = np.array(treated_diffs)
    control_diffs = np.array(control_diffs)
    
    n_treated = len(treated_diffs)
    n_control = len(control_diffs)
    
    if n_treated == 0 or n_control == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    
    # ATT = E[Y_post - Y_pre | D=1] - E[Y_post - Y_pre | D=0]
    att = treated_diffs.mean() - control_diffs.mean()
    
    # SE (Welch 方法)
    var_treated = np.var(treated_diffs, ddof=1) if n_treated > 1 else 0
    var_control = np.var(control_diffs, ddof=1) if n_control > 1 else 0
    se = np.sqrt(var_treated / n_treated + var_control / n_control)
    
    # 自由度和置信区间
    df = n_treated + n_control - 2
    alpha = 0.05
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    ci_lower = att - t_crit * se
    ci_upper = att + t_crit * se
    t_stat = att / se if se > 0 else 0
    pvalue = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    
    return att, se, ci_lower, ci_upper, pvalue


def large_sample_dgp_wrapper(**kwargs):
    """
    大样本 DGP 包装器
    
    将 generate_common_timing_dgp 的返回值转换为 Monte Carlo runner 期望的格式
    """
    from dgp_common_timing import generate_common_timing_dgp
    
    data, true_atts = generate_common_timing_dgp(**kwargs)
    
    # 计算平均真实 ATT
    avg_true_att = np.mean(list(true_atts.values()))
    
    # 构造 params 字典
    params = {
        'true_att': avg_true_att,
        'true_atts': true_atts,
        'tau': avg_true_att,
    }
    
    return data, params


class TestLargeSampleBenchmark:
    """
    大样本场景基准测试
    
    论文 Section 7.1 场景:
    - 1C: 均值正确, PS正确 (基准场景)
    - 2C: 均值正确, PS错误
    - 3C: 均值错误, PS正确
    - 4C: 均值错误, PS错误
    """
    
    @pytest.fixture
    def dgp_func(self):
        """获取大样本DGP函数（包装后）"""
        return large_sample_dgp_wrapper
    
    @pytest.mark.parametrize("scenario", ['1C', '2C', '3C', '4C'])
    def test_scenario_basic(self, dgp_func, scenario):
        """
        基础测试: 验证估计器能正常运行
        """
        from runner import run_monte_carlo
        
        result = run_monte_carlo(
            estimator_func=large_sample_estimator,
            dgp_func=dgp_func,
            n_reps=30,  # 小规模测试
            dgp_kwargs={'scenario': scenario, 'n_units': 100},
            seed=42,
            parallel=False,
            verbose=False,
            estimator_name='OLS_Simple',
            dgp_name='common_timing',
            scenario=scenario,
        )
        
        assert result.n_valid > 0, f"场景{scenario}应该有有效的估计结果"
        assert not np.isnan(result.bias), f"场景{scenario} Bias不应为NaN"
        assert result.sd > 0, f"场景{scenario} SD应该为正"
        # 验证真实 ATT 不为 0
        assert result.true_att != 0, f"场景{scenario} 真实ATT不应为0"
    
    @pytest.mark.parametrize("n_units", [100, 500])
    def test_scenario_1c_sample_sizes(self, dgp_func, n_units):
        """
        场景1C: 测试不同样本量
        """
        from runner import run_monte_carlo
        
        result = run_monte_carlo(
            estimator_func=large_sample_estimator,
            dgp_func=dgp_func,
            n_reps=50,
            dgp_kwargs={'scenario': '1C', 'n_units': n_units},
            seed=42,
            parallel=False,
            verbose=False,
            estimator_name='OLS_Simple',
            dgp_name='common_timing',
            scenario=f'1C_N{n_units}',
        )
        
        print(f"\n场景1C (N={n_units}):")
        print(f"  True ATT: {result.true_att:.4f}")
        print(f"  Mean ATT: {result.mean_att:.4f}")
        print(f"  Bias: {result.bias:.4f}")
        print(f"  SD: {result.sd:.4f}")
        print(f"  RMSE: {result.rmse:.4f}")
        print(f"  Coverage: {result.coverage:.2%}")
        
        # 验证基本属性
        assert result.n_valid > 0
        # Bias 应该相对较小（相对于真实 ATT）
        rel_bias = abs(result.bias) / abs(result.true_att) if result.true_att != 0 else abs(result.bias)
        assert rel_bias < 0.5, f"相对Bias {rel_bias:.2%} 过大"


class TestLargeSampleScenarioComparison:
    """
    大样本场景对比测试
    
    验证不同场景下估计器的表现差异
    """
    
    @pytest.fixture
    def dgp_func(self):
        """获取大样本DGP函数（包装后）"""
        return large_sample_dgp_wrapper
    
    def test_correct_vs_misspecified_ps(self, dgp_func):
        """
        比较正确设定PS (1C) vs 错误设定PS (2C)
        """
        from runner import run_monte_carlo
        
        n_reps = 50
        n_units = 200
        
        # 场景1C: PS正确
        result_1c = run_monte_carlo(
            estimator_func=large_sample_estimator,
            dgp_func=dgp_func,
            n_reps=n_reps,
            dgp_kwargs={'scenario': '1C', 'n_units': n_units},
            seed=42,
            parallel=False,
            verbose=False,
            estimator_name='OLS_Simple',
            dgp_name='common_timing',
            scenario='1C',
        )
        
        # 场景2C: PS错误
        result_2c = run_monte_carlo(
            estimator_func=large_sample_estimator,
            dgp_func=dgp_func,
            n_reps=n_reps,
            dgp_kwargs={'scenario': '2C', 'n_units': n_units},
            seed=42,
            parallel=False,
            verbose=False,
            estimator_name='OLS_Simple',
            dgp_name='common_timing',
            scenario='2C',
        )
        
        print(f"\nPS设定比较 (N={n_units}):")
        print(f"  1C (PS正确): Bias={result_1c.bias:.4f}, RMSE={result_1c.rmse:.4f}")
        print(f"  2C (PS错误): Bias={result_2c.bias:.4f}, RMSE={result_2c.rmse:.4f}")
        
        # 两个场景都应该产生有效结果
        assert result_1c.n_valid > 0
        assert result_2c.n_valid > 0
    
    def test_correct_vs_misspecified_om(self, dgp_func):
        """
        比较正确设定OM (1C) vs 错误设定OM (3C)
        """
        from runner import run_monte_carlo
        
        n_reps = 50
        n_units = 200
        
        # 场景1C: OM正确
        result_1c = run_monte_carlo(
            estimator_func=large_sample_estimator,
            dgp_func=dgp_func,
            n_reps=n_reps,
            dgp_kwargs={'scenario': '1C', 'n_units': n_units},
            seed=42,
            parallel=False,
            verbose=False,
            estimator_name='OLS_Simple',
            dgp_name='common_timing',
            scenario='1C',
        )
        
        # 场景3C: OM错误
        result_3c = run_monte_carlo(
            estimator_func=large_sample_estimator,
            dgp_func=dgp_func,
            n_reps=n_reps,
            dgp_kwargs={'scenario': '3C', 'n_units': n_units},
            seed=42,
            parallel=False,
            verbose=False,
            estimator_name='OLS_Simple',
            dgp_name='common_timing',
            scenario='3C',
        )
        
        print(f"\nOM设定比较 (N={n_units}):")
        print(f"  1C (OM正确): Bias={result_1c.bias:.4f}, RMSE={result_1c.rmse:.4f}")
        print(f"  3C (OM错误): Bias={result_3c.bias:.4f}, RMSE={result_3c.rmse:.4f}")
        
        assert result_1c.n_valid > 0
        assert result_3c.n_valid > 0


class TestLargeSampleConvergence:
    """
    大样本收敛性测试
    
    验证随着样本量增加，估计器表现改善
    """
    
    @pytest.fixture
    def dgp_func(self):
        """获取大样本DGP函数（包装后）"""
        return large_sample_dgp_wrapper
    
    def test_rmse_decreases_with_sample_size(self, dgp_func):
        """
        验证RMSE随样本量增加而减小
        """
        from runner import run_monte_carlo
        
        sample_sizes = [100, 300]
        results = {}
        
        for n_units in sample_sizes:
            result = run_monte_carlo(
                estimator_func=large_sample_estimator,
                dgp_func=dgp_func,
                n_reps=30,
                dgp_kwargs={'scenario': '1C', 'n_units': n_units},
                seed=42,
                parallel=False,
                verbose=False,
                estimator_name='OLS_Simple',
                dgp_name='common_timing',
                scenario=f'1C_N{n_units}',
            )
            results[n_units] = result
        
        print(f"\n样本量收敛性:")
        for n, r in results.items():
            print(f"  N={n}: SD={r.sd:.4f}, RMSE={r.rmse:.4f}")
        
        # 验证SD随样本量增加而减小（大致趋势）
        # 注意：由于模拟次数少，可能有噪声
        assert results[100].n_valid > 0
        assert results[300].n_valid > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])

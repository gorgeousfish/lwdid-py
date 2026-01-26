"""
DESIGN-027: Monte Carlo 覆盖率验证

测试目标：
1. 验证不同 alpha 值下 CI 的实际覆盖率
2. 确认 95%, 90%, 99% CI 达到预期覆盖率
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats


class TestMonteCarloCoverage:
    """Monte Carlo 模拟验证 CI 覆盖率"""
    
    @pytest.fixture
    def true_effect(self):
        """真实的处理效应"""
        return 0.5
    
    def _simulate_single_experiment(self, true_att, n_treated, n_control, se_true, seed):
        """模拟单次实验"""
        np.random.seed(seed)
        
        # 生成模拟的 ATT 估计（假设正态分布）
        att_hat = true_att + np.random.normal(0, se_true)
        
        # 生成模拟的 SE 估计（使用 chi-squared 分布模拟样本方差）
        # SE_hat^2 ~ se_true^2 * chi^2(df) / df
        df = n_treated + n_control - 2
        se_hat = se_true * np.sqrt(np.random.chisquare(df) / df)
        
        return att_hat, se_hat
    
    @pytest.mark.parametrize("alpha,expected_coverage", [
        (0.05, 0.95),
        (0.10, 0.90),
        (0.01, 0.99),
        (0.20, 0.80),
    ])
    def test_ci_coverage_monte_carlo(self, true_effect, alpha, expected_coverage):
        """Monte Carlo 模拟验证 CI 覆盖率"""
        n_simulations = 2000
        n_treated = 100
        n_control = 100
        se_true = 0.1
        
        coverage_count = 0
        
        for seed in range(n_simulations):
            att_hat, se_hat = self._simulate_single_experiment(
                true_effect, n_treated, n_control, se_true, seed
            )
            
            # 计算 CI
            z = stats.norm.ppf(1 - alpha / 2)
            ci_lower = att_hat - z * se_hat
            ci_upper = att_hat + z * se_hat
            
            # 检查真实效应是否在 CI 内
            if ci_lower <= true_effect <= ci_upper:
                coverage_count += 1
        
        actual_coverage = coverage_count / n_simulations
        
        # 允许 3% 的误差范围（Monte Carlo 模拟的随机性）
        tolerance = 0.03
        assert abs(actual_coverage - expected_coverage) < tolerance, (
            f"Coverage for alpha={alpha}: expected ≈{expected_coverage:.2f}, "
            f"got {actual_coverage:.4f}"
        )
    
    def test_coverage_increases_with_smaller_alpha(self, true_effect):
        """验证更小的 alpha 产生更高的覆盖率"""
        n_simulations = 1000
        n_treated = 100
        n_control = 100
        se_true = 0.1
        
        coverages = {}
        
        for alpha in [0.01, 0.05, 0.10, 0.20]:
            coverage_count = 0
            
            for seed in range(n_simulations):
                att_hat, se_hat = self._simulate_single_experiment(
                    true_effect, n_treated, n_control, se_true, seed
                )
                
                z = stats.norm.ppf(1 - alpha / 2)
                ci_lower = att_hat - z * se_hat
                ci_upper = att_hat + z * se_hat
                
                if ci_lower <= true_effect <= ci_upper:
                    coverage_count += 1
            
            coverages[alpha] = coverage_count / n_simulations
        
        # 验证 coverage 随 alpha 减小而增加
        assert coverages[0.01] > coverages[0.05] > coverages[0.10] > coverages[0.20], (
            f"Coverage should increase with smaller alpha: {coverages}"
        )


class TestMonteCarloCIWidth:
    """Monte Carlo 验证 CI 宽度与 alpha 的关系"""
    
    def test_ci_width_ratio(self):
        """验证不同 alpha 下 CI 宽度的比例关系"""
        se = 0.1
        
        width_95 = 2 * stats.norm.ppf(0.975) * se
        width_90 = 2 * stats.norm.ppf(0.95) * se
        width_99 = 2 * stats.norm.ppf(0.995) * se
        width_80 = 2 * stats.norm.ppf(0.90) * se
        
        # 验证相对宽度
        assert width_99 > width_95 > width_90 > width_80
        
        # 验证具体比例（允许小误差）
        ratio_99_95 = width_99 / width_95
        expected_ratio = stats.norm.ppf(0.995) / stats.norm.ppf(0.975)
        assert abs(ratio_99_95 - expected_ratio) < 0.001


class TestStaggeredEstimatorMonteCarlo:
    """对 staggered 估计器进行简化的 Monte Carlo 验证"""
    
    def test_ra_coverage_simple(self):
        """简单的 RA 估计器覆盖率测试"""
        from lwdid.staggered.transformations import transform_staggered_demean
        from lwdid.staggered.estimation import estimate_cohort_time_effects
        
        n_simulations = 50  # 减少模拟次数以加快测试
        true_effect = 0.5
        alpha = 0.05
        
        coverage_count = 0
        
        for seed in range(n_simulations):
            np.random.seed(seed)
            
            # 生成模拟数据
            n_units = 100
            n_periods = 6
            data = []
            
            for i in range(n_units):
                gvar = 0 if i < 50 else 2003
                
                for t in range(2000, 2000 + n_periods):
                    treated = (gvar > 0) and (t >= gvar)
                    effect = true_effect if treated else 0.0
                    y = 2.0 + 0.1 * t + effect + np.random.normal(0, 0.5)
                    
                    data.append({
                        'unit_id': i,
                        'year': t,
                        'y': y,
                        'gvar': gvar,
                    })
            
            df = pd.DataFrame(data)
            
            try:
                # 变换和估计
                df_transformed = transform_staggered_demean(
                    data=df, y='y', gvar='gvar', tvar='year', ivar='unit_id'
                )
                
                results = estimate_cohort_time_effects(
                    data_transformed=df_transformed,
                    gvar='gvar', tvar='year', ivar='unit_id',
                    estimator='ra', alpha=alpha
                )
                
                # 检查 e=0 时刻的估计
                for r in results:
                    if r.event_time == 0:
                        if r.ci_lower <= true_effect <= r.ci_upper:
                            coverage_count += 1
                        break
            except Exception:
                # 某些模拟可能因为样本量问题失败
                continue
        
        actual_coverage = coverage_count / n_simulations
        
        # 允许较大的容差（小样本 Monte Carlo）
        assert actual_coverage >= 0.80, (
            f"RA estimator coverage too low: {actual_coverage:.2f}"
        )
        assert actual_coverage <= 1.0, (
            f"RA estimator coverage invalid: {actual_coverage:.2f}"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

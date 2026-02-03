"""
测试 IPW 估计器使用正态分布进行推断。

验证修改后的 IPW 估计器：
1. p-value 使用正态分布计算
2. 置信区间使用正态分布计算
3. 与 Stata teffects ipw 行为一致
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import pytest

from lwdid.staggered.estimators import estimate_ipw


def create_test_data(n=200, seed=42):
    """创建测试数据集。"""
    np.random.seed(seed)
    
    # 协变量
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    
    # 倾向得分模型: logit(p) = -0.5 + 0.5*x1 + 0.3*x2
    logit_p = -0.5 + 0.5 * x1 + 0.3 * x2
    p = 1 / (1 + np.exp(-logit_p))
    
    # 处理分配
    d = (np.random.uniform(0, 1, n) < p).astype(int)
    
    # 结果变量: y = 2 + 1*x1 + 0.5*x2 + 3*d + epsilon
    # ATT = 3
    y = 2 + 1 * x1 + 0.5 * x2 + 3 * d + np.random.normal(0, 1, n)
    
    return pd.DataFrame({
        'y': y,
        'd': d,
        'x1': x1,
        'x2': x2,
    })


class TestIPWNormalDistribution:
    """测试 IPW 使用正态分布进行推断。"""
    
    def test_pvalue_uses_normal_distribution(self):
        """验证 p-value 使用正态分布计算。"""
        data = create_test_data(n=500, seed=123)
        
        result = estimate_ipw(
            data=data,
            y='y',
            d='d',
            propensity_controls=['x1', 'x2'],
            alpha=0.05,
        )
        
        # 手动计算正态分布 p-value
        z_stat = result.att / result.se
        expected_pvalue = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        # 验证 p-value 与正态分布计算一致
        assert np.isclose(result.pvalue, expected_pvalue, rtol=1e-10), \
            f"p-value 不一致: {result.pvalue} vs {expected_pvalue}"
        
        # 验证不是 t 分布（除非样本量很大时两者接近）
        # 对于中等样本，t 分布和正态分布应该有可测量的差异
        df = result.n_treated + result.n_control - 2
        t_pvalue = 2 * stats.t.sf(abs(z_stat), df)
        
        # 打印诊断信息
        print(f"\nATT: {result.att:.4f}")
        print(f"SE: {result.se:.4f}")
        print(f"z-stat: {z_stat:.4f}")
        print(f"Normal p-value: {expected_pvalue:.6f}")
        print(f"t p-value (df={df}): {t_pvalue:.6f}")
        print(f"Actual p-value: {result.pvalue:.6f}")
    
    def test_ci_uses_normal_distribution(self):
        """验证置信区间使用正态分布计算。"""
        data = create_test_data(n=500, seed=456)
        
        for alpha in [0.05, 0.10, 0.01]:
            result = estimate_ipw(
                data=data,
                y='y',
                d='d',
                propensity_controls=['x1', 'x2'],
                alpha=alpha,
            )
            
            # 手动计算正态分布置信区间
            z_crit = stats.norm.ppf(1 - alpha / 2)
            expected_ci_lower = result.att - z_crit * result.se
            expected_ci_upper = result.att + z_crit * result.se
            
            # 验证置信区间与正态分布计算一致
            assert np.isclose(result.ci_lower, expected_ci_lower, rtol=1e-10), \
                f"CI lower 不一致 (alpha={alpha}): {result.ci_lower} vs {expected_ci_lower}"
            assert np.isclose(result.ci_upper, expected_ci_upper, rtol=1e-10), \
                f"CI upper 不一致 (alpha={alpha}): {result.ci_upper} vs {expected_ci_upper}"
            
            print(f"\nalpha={alpha}:")
            print(f"  z_crit: {z_crit:.4f}")
            print(f"  Expected CI: [{expected_ci_lower:.4f}, {expected_ci_upper:.4f}]")
            print(f"  Actual CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
    
    def test_att_estimate_reasonable(self):
        """验证 ATT 估计值合理（真实值为 3）。"""
        data = create_test_data(n=1000, seed=789)
        
        result = estimate_ipw(
            data=data,
            y='y',
            d='d',
            propensity_controls=['x1', 'x2'],
            alpha=0.05,
        )
        
        # ATT 应该接近真实值 3
        assert 2.0 < result.att < 4.0, \
            f"ATT 估计值 {result.att} 偏离真实值 3 太远"
        
        # 置信区间应该包含真实值
        assert result.ci_lower < 3.0 < result.ci_upper, \
            f"95% CI [{result.ci_lower}, {result.ci_upper}] 不包含真实值 3"
        
        print(f"\n真实 ATT: 3.0")
        print(f"估计 ATT: {result.att:.4f}")
        print(f"95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])

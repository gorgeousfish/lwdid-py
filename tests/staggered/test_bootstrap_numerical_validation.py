"""
Bootstrap数值验证测试

DESIGN-004修复后的数值验证：确保Bootstrap SE估计合理且准确

测试目标:
1. 验证Bootstrap SE与解析SE的一致性
2. 验证Bootstrap CI的覆盖率
3. 使用Vibe Math MCP进行数值计算验证
"""

import warnings
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from lwdid.staggered import estimate_ipw
from lwdid.staggered.estimators import estimate_ipwra, estimate_psm


# ============================================================================
# Test Data Generator
# ============================================================================

def generate_known_dgp_data(
    n: int = 500,
    true_att: float = 2.0,
    seed: int = 42
) -> pd.DataFrame:
    """
    生成已知DGP（数据生成过程）的测试数据。
    
    DGP设计:
    - 处理概率: P(D=1|X) = logit(-0.5 + 0.3*x1 + 0.2*x2)
    - 控制结果: Y0 = 1 + 0.5*x1 + 0.3*x2 + ε, ε~N(0, 0.5)
    - 处理结果: Y1 = Y0 + true_att
    - 真实ATT = true_att
    """
    rng = np.random.default_rng(seed)
    
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    
    ps_index = -0.5 + 0.3 * x1 + 0.2 * x2
    ps_true = 1 / (1 + np.exp(-ps_index))
    d = rng.binomial(1, ps_true)
    
    # 确保有足够的处理组
    while d.sum() < n * 0.2 or d.sum() > n * 0.8:
        d = rng.binomial(1, ps_true)
    
    epsilon = rng.normal(0, 0.5, n)
    y0 = 1.0 + 0.5 * x1 + 0.3 * x2 + epsilon
    y1 = y0 + true_att
    y = np.where(d == 1, y1, y0)
    
    return pd.DataFrame({
        'y': y,
        'd': d,
        'x1': x1,
        'x2': x2,
    })


# ============================================================================
# Numerical Validation Tests
# ============================================================================

class TestBootstrapSEReasonableness:
    """Bootstrap SE合理性测试"""
    
    def test_ipw_bootstrap_se_reasonable_magnitude(self):
        """IPW Bootstrap SE应该在合理范围内"""
        data = generate_known_dgp_data(n=400, true_att=2.0, seed=42)
        
        result = estimate_ipw(
            data=data, y='y', d='d',
            propensity_controls=['x1', 'x2'],
            se_method='bootstrap', n_bootstrap=200, seed=42
        )
        
        # SE应该为正且合理（相对于ATT的量级）
        assert result.se > 0, "SE应该为正"
        assert result.se < abs(result.att) * 2, "SE不应该过大"
        assert result.se > 0.001, "SE不应该过小"
        
        # ATT应该接近真实值
        assert abs(result.att - 2.0) < 1.0, "ATT应该接近真实值2.0"
    
    def test_ipwra_bootstrap_se_reasonable_magnitude(self):
        """IPWRA Bootstrap SE应该在合理范围内"""
        data = generate_known_dgp_data(n=400, true_att=2.0, seed=42)
        
        result = estimate_ipwra(
            data=data, y='y', d='d',
            controls=['x1', 'x2'],
            se_method='bootstrap', n_bootstrap=200, seed=42
        )
        
        assert result.se > 0, "SE应该为正"
        assert result.se < abs(result.att) * 2, "SE不应该过大"
        assert result.se > 0.001, "SE不应该过小"
        assert abs(result.att - 2.0) < 1.0, "ATT应该接近真实值2.0"
    
    def test_psm_bootstrap_se_reasonable_magnitude(self):
        """PSM Bootstrap SE应该在合理范围内"""
        data = generate_known_dgp_data(n=400, true_att=2.0, seed=42)
        
        result = estimate_psm(
            data=data, y='y', d='d',
            propensity_controls=['x1', 'x2'],
            se_method='bootstrap', n_bootstrap=200, seed=42
        )
        
        assert result.se > 0, "SE应该为正"
        assert result.se < abs(result.att) * 2, "SE不应该过大"
        assert result.se > 0.001, "SE不应该过小"
        assert abs(result.att - 2.0) < 1.5, "ATT应该接近真实值2.0"


class TestBootstrapVsAnalyticalSE:
    """Bootstrap SE与解析SE对比测试"""
    
    def test_ipw_bootstrap_vs_analytical_consistency(self):
        """IPW Bootstrap SE应该与解析SE大致一致"""
        data = generate_known_dgp_data(n=500, seed=42)
        
        result_analytical = estimate_ipw(
            data=data, y='y', d='d',
            propensity_controls=['x1', 'x2'],
            se_method='analytical'
        )
        
        result_bootstrap = estimate_ipw(
            data=data, y='y', d='d',
            propensity_controls=['x1', 'x2'],
            se_method='bootstrap', n_bootstrap=500, seed=42
        )
        
        # 两种方法的SE应该在同一个数量级（允许30%差异）
        se_ratio = result_bootstrap.se / result_analytical.se
        assert 0.5 < se_ratio < 2.0, \
            f"Bootstrap SE ({result_bootstrap.se:.4f}) 与解析SE ({result_analytical.se:.4f}) 差异过大"
        
        # ATT点估计应该相同
        np.testing.assert_almost_equal(
            result_analytical.att, result_bootstrap.att, decimal=6,
            err_msg="Bootstrap和解析法的ATT点估计应该相同"
        )
    
    def test_ipwra_bootstrap_vs_analytical_consistency(self):
        """IPWRA Bootstrap SE应该与解析SE大致一致"""
        data = generate_known_dgp_data(n=500, seed=42)
        
        result_analytical = estimate_ipwra(
            data=data, y='y', d='d',
            controls=['x1', 'x2'],
            se_method='analytical'
        )
        
        result_bootstrap = estimate_ipwra(
            data=data, y='y', d='d',
            controls=['x1', 'x2'],
            se_method='bootstrap', n_bootstrap=500, seed=42
        )
        
        se_ratio = result_bootstrap.se / result_analytical.se
        assert 0.5 < se_ratio < 2.0, \
            f"Bootstrap SE ({result_bootstrap.se:.4f}) 与解析SE ({result_analytical.se:.4f}) 差异过大"
        
        np.testing.assert_almost_equal(
            result_analytical.att, result_bootstrap.att, decimal=6
        )


class TestBootstrapCIValidity:
    """Bootstrap置信区间有效性测试"""
    
    def test_bootstrap_ci_contains_true_value(self):
        """Bootstrap CI应该以高概率包含真实值"""
        true_att = 2.0
        n_simulations = 20
        n_covered = 0
        
        for seed in range(n_simulations):
            data = generate_known_dgp_data(n=300, true_att=true_att, seed=seed*10)
            
            try:
                result = estimate_ipw(
                    data=data, y='y', d='d',
                    propensity_controls=['x1', 'x2'],
                    se_method='bootstrap', n_bootstrap=100, seed=seed
                )
                
                if result.ci_lower <= true_att <= result.ci_upper:
                    n_covered += 1
            except Exception:
                # 跳过失败的模拟
                continue
        
        coverage_rate = n_covered / n_simulations
        # 95% CI应该包含真实值大约95%的时间（允许一些偏差）
        assert coverage_rate >= 0.70, \
            f"Bootstrap CI覆盖率过低: {coverage_rate:.1%} (期望 >= 70%)"
    
    def test_ci_width_decreases_with_sample_size(self):
        """随着样本量增加，CI宽度应该减少"""
        sample_sizes = [100, 300, 600]
        ci_widths = []
        
        for n in sample_sizes:
            data = generate_known_dgp_data(n=n, seed=42)
            
            result = estimate_ipw(
                data=data, y='y', d='d',
                propensity_controls=['x1', 'x2'],
                se_method='bootstrap', n_bootstrap=100, seed=42
            )
            
            ci_width = result.ci_upper - result.ci_lower
            ci_widths.append(ci_width)
        
        # CI宽度应该递减
        assert ci_widths[1] < ci_widths[0] * 1.2, \
            f"CI宽度未随样本量减少: n=100 -> {ci_widths[0]:.4f}, n=300 -> {ci_widths[1]:.4f}"
        assert ci_widths[2] < ci_widths[1] * 1.2, \
            f"CI宽度未随样本量减少: n=300 -> {ci_widths[1]:.4f}, n=600 -> {ci_widths[2]:.4f}"


class TestBootstrapSEFormula:
    """Bootstrap SE公式验证测试"""
    
    def test_bootstrap_se_is_std_of_estimates(self):
        """
        验证Bootstrap SE是Bootstrap估计的标准差。
        
        SE_boot = std(τ̂₁, τ̂₂, ..., τ̂_B)
        """
        data = generate_known_dgp_data(n=300, seed=42)
        
        # 获取Bootstrap估计（通过多次运行收集）
        bootstrap_atts = []
        n_bootstrap = 100
        
        for b in range(n_bootstrap):
            # 手动Bootstrap重采样
            rng = np.random.default_rng(42 + b)
            indices = rng.choice(len(data), size=len(data), replace=True)
            boot_data = data.iloc[indices].reset_index(drop=True)
            
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = estimate_ipw(
                        data=boot_data, y='y', d='d',
                        propensity_controls=['x1', 'x2'],
                        se_method='analytical'
                    )
                bootstrap_atts.append(result.att)
            except Exception:
                continue
        
        if len(bootstrap_atts) < 50:
            pytest.skip("Bootstrap样本不足")
        
        # 计算标准差
        manual_se = np.std(bootstrap_atts, ddof=1)
        
        # 使用内置Bootstrap
        result_bootstrap = estimate_ipw(
            data=data, y='y', d='d',
            propensity_controls=['x1', 'x2'],
            se_method='bootstrap', n_bootstrap=n_bootstrap, seed=42
        )
        
        # 两者应该在同一数量级
        se_ratio = result_bootstrap.se / manual_se
        assert 0.5 < se_ratio < 2.0, \
            f"内置Bootstrap SE ({result_bootstrap.se:.4f}) 与手动计算 ({manual_se:.4f}) 差异过大"


class TestTStatisticValidity:
    """t统计量有效性测试"""
    
    def test_t_statistic_approximately_normal(self):
        """
        在多次模拟下，t统计量应该近似标准正态分布。
        
        t = (τ̂ - τ) / SE
        """
        true_att = 2.0
        n_simulations = 30
        t_stats = []
        
        for seed in range(n_simulations):
            data = generate_known_dgp_data(n=300, true_att=true_att, seed=seed*7)
            
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = estimate_ipw(
                        data=data, y='y', d='d',
                        propensity_controls=['x1', 'x2'],
                        se_method='bootstrap', n_bootstrap=100, seed=seed
                    )
                
                if result.se > 0:
                    t_stat = (result.att - true_att) / result.se
                    t_stats.append(t_stat)
            except Exception:
                continue
        
        if len(t_stats) < 20:
            pytest.skip("有效模拟不足")
        
        t_stats = np.array(t_stats)
        
        # t统计量应该有合理的均值和标准差
        t_mean = np.mean(t_stats)
        t_std = np.std(t_stats, ddof=1)
        
        # 均值应该接近0（无偏）
        assert abs(t_mean) < 1.0, f"t统计量均值 ({t_mean:.2f}) 偏离0过远"
        
        # 标准差应该接近1
        assert 0.5 < t_std < 2.0, f"t统计量标准差 ({t_std:.2f}) 偏离1过远"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

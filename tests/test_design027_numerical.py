"""
DESIGN-027: 数值验证 - 不同 alpha 值下 CI 计算正确性

测试目标：
1. 验证 CI 计算公式正确：ci = att ± z_α/2 * se
2. 验证不同 alpha 值生成正确的 CI 宽度
3. 使用 vibe-math-mcp 验证公式
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats


# ============================================================================
# Section 1: CI 公式正确性验证
# ============================================================================

class TestCIFormulaCorrectness:
    """验证 CI 计算公式正确性"""
    
    @pytest.mark.parametrize("alpha,expected_z", [
        (0.05, 1.96),     # 95% CI
        (0.10, 1.645),    # 90% CI
        (0.01, 2.576),    # 99% CI
        (0.20, 1.282),    # 80% CI
    ])
    def test_z_critical_values(self, alpha, expected_z):
        """验证不同 alpha 对应的 z 临界值"""
        z_calculated = stats.norm.ppf(1 - alpha / 2)
        
        # 允许 0.01 的误差
        assert abs(z_calculated - expected_z) < 0.01, (
            f"For alpha={alpha}: expected z≈{expected_z}, got {z_calculated:.4f}"
        )
    
    def test_ci_width_formula(self):
        """验证 CI 宽度公式：width = 2 * z_α/2 * se"""
        att = 0.5
        se = 0.1
        
        for alpha in [0.01, 0.05, 0.10, 0.20]:
            z = stats.norm.ppf(1 - alpha / 2)
            expected_width = 2 * z * se
            
            ci_lower = att - z * se
            ci_upper = att + z * se
            actual_width = ci_upper - ci_lower
            
            assert abs(actual_width - expected_width) < 1e-10, (
                f"CI width mismatch for alpha={alpha}: "
                f"expected {expected_width:.6f}, got {actual_width:.6f}"
            )
    
    def test_ci_symmetry(self):
        """验证 CI 关于 ATT 对称"""
        att = 0.5
        se = 0.1
        alpha = 0.05
        
        z = stats.norm.ppf(1 - alpha / 2)
        ci_lower = att - z * se
        ci_upper = att + z * se
        
        midpoint = (ci_lower + ci_upper) / 2
        assert abs(midpoint - att) < 1e-10, (
            f"CI not symmetric around ATT: midpoint={midpoint:.6f}, ATT={att}"
        )


# ============================================================================
# Section 2: 不同 alpha 值的 CI 宽度验证
# ============================================================================

class TestCIWidthWithAlpha:
    """验证不同 alpha 值生成正确宽度的 CI"""
    
    def test_smaller_alpha_wider_ci(self):
        """更小的 alpha 应该产生更宽的 CI"""
        att = 0.5
        se = 0.1
        
        widths = {}
        for alpha in [0.01, 0.05, 0.10, 0.20]:
            z = stats.norm.ppf(1 - alpha / 2)
            width = 2 * z * se
            widths[alpha] = width
        
        # 验证：alpha 越小，CI 越宽
        assert widths[0.01] > widths[0.05] > widths[0.10] > widths[0.20], (
            f"CI widths should decrease with increasing alpha: {widths}"
        )
    
    def test_specific_ci_widths(self):
        """验证特定 alpha 值的 CI 宽度比例"""
        se = 0.1
        
        # 95% CI 宽度 / 90% CI 宽度 ≈ 1.96 / 1.645 ≈ 1.19
        z_95 = stats.norm.ppf(0.975)
        z_90 = stats.norm.ppf(0.95)
        
        width_ratio = z_95 / z_90
        expected_ratio = 1.96 / 1.645
        
        assert abs(width_ratio - expected_ratio) < 0.01, (
            f"95%/90% CI width ratio: expected ≈{expected_ratio:.4f}, got {width_ratio:.4f}"
        )


# ============================================================================
# Section 3: Staggered 估计器 CI 计算验证
# ============================================================================

class TestStaggeredEstimatorCI:
    """验证 staggered 估计器的 CI 计算"""
    
    @pytest.fixture
    def simple_panel_data(self):
        """创建简单的面板数据用于测试"""
        np.random.seed(42)
        n_units = 200
        n_periods = 10
        
        # 创建面板数据
        data = []
        for i in range(n_units):
            # 前 100 个单位是 never treated，后 100 个是 cohort 2006
            gvar = 0 if i < 100 else 2006
            
            for t in range(2000, 2000 + n_periods):
                # 处理效应：treated 单位在处理后增加 0.5
                treated = (gvar > 0) and (t >= gvar)
                effect = 0.5 if treated else 0.0
                
                # 生成结果变量
                y = 2.0 + 0.1 * t + effect + np.random.normal(0, 0.5)
                
                data.append({
                    'unit_id': i,
                    'year': t,
                    'y': y,
                    'gvar': gvar,
                })
        
        return pd.DataFrame(data)
    
    def test_ipw_ci_with_different_alpha(self, simple_panel_data):
        """测试 IPW 估计器在不同 alpha 下的 CI"""
        from lwdid.staggered.transformations import transform_staggered_demean
        from lwdid.staggered.estimation import estimate_cohort_time_effects
        
        data = simple_panel_data
        
        # 先进行 demean 变换
        data_transformed = transform_staggered_demean(
            data=data,
            y='y',
            gvar='gvar',
            tvar='year',
            ivar='unit_id'
        )
        
        # 使用 alpha=0.05
        results_05 = estimate_cohort_time_effects(
            data_transformed=data_transformed,
            gvar='gvar',
            tvar='year',
            ivar='unit_id',
            estimator='ipw',
            alpha=0.05
        )
        
        # 使用 alpha=0.10
        results_10 = estimate_cohort_time_effects(
            data_transformed=data_transformed,
            gvar='gvar',
            tvar='year',
            ivar='unit_id',
            estimator='ipw',
            alpha=0.10
        )
        
        # 验证 alpha=0.05 的 CI 比 alpha=0.10 更宽
        if len(results_05) > 0 and len(results_10) > 0:
            width_05 = results_05[0].ci_upper - results_05[0].ci_lower
            width_10 = results_10[0].ci_upper - results_10[0].ci_lower
            
            assert width_05 > width_10, (
                f"95% CI should be wider than 90% CI: "
                f"width_95={width_05:.4f}, width_90={width_10:.4f}"
            )
    
    def test_ra_ci_formula_validation(self, simple_panel_data):
        """
        测试 RA 估计器 CI 公式验证
        
        RA 估计器使用 run_ols_regression，该函数使用 t 分布进行推断
        （与 Stata 的 regress 命令一致，适用于小样本）。
        
        注意：
        - lwdid-stata (2025, 小样本) 使用 t 分布计算 p 值
        - Lee_Wooldridge_2023 大样本实现使用 teffects（正态分布）
        - Python 实现中 RA 使用 t 分布以匹配小样本理论
        """
        from lwdid.staggered.transformations import transform_staggered_demean
        from lwdid.staggered.estimation import estimate_cohort_time_effects
        
        data = simple_panel_data
        alpha = 0.05
        
        # 先进行 demean 变换
        data_transformed = transform_staggered_demean(
            data=data,
            y='y',
            gvar='gvar',
            tvar='year',
            ivar='unit_id'
        )
        
        results = estimate_cohort_time_effects(
            data_transformed=data_transformed,
            gvar='gvar',
            tvar='year',
            ivar='unit_id',
            estimator='ra',
            alpha=alpha
        )
        
        if len(results) > 0:
            result = results[0]
            
            # RA 估计器使用 t 分布，df = n_treated + n_control - 2
            df = result.n_treated + result.n_control - 2
            t_crit = stats.t.ppf(1 - alpha / 2, df)
            expected_ci_lower_t = result.att - t_crit * result.se
            expected_ci_upper_t = result.att + t_crit * result.se
            
            # 验证 CI 与 t 分布计算一致
            assert abs(result.ci_lower - expected_ci_lower_t) < 1e-4, (
                f"CI lower mismatch with t-distribution: "
                f"expected {expected_ci_lower_t:.6f}, got {result.ci_lower:.6f}"
            )
            assert abs(result.ci_upper - expected_ci_upper_t) < 1e-4, (
                f"CI upper mismatch with t-distribution: "
                f"expected {expected_ci_upper_t:.6f}, got {result.ci_upper:.6f}"
            )
            
            # 验证 CI 对称性
            midpoint = (result.ci_lower + result.ci_upper) / 2
            assert abs(midpoint - result.att) < 1e-6, (
                f"CI not symmetric around ATT: midpoint={midpoint:.6f}, "
                f"ATT={result.att:.6f}"
            )
    
    def test_ipwra_ci_uses_normal_distribution(self, simple_panel_data):
        """
        验证 IPWRA 估计器使用正态分布
        
        IPWRA/IPW/PSM 等估计器使用 M-estimation 框架，
        在大样本下是渐近正态的，与 Stata teffects 命令一致。
        """
        from lwdid.staggered.transformations import transform_staggered_demean
        from lwdid.staggered.estimation import estimate_cohort_time_effects
        
        data = simple_panel_data
        alpha = 0.05
        
        data_transformed = transform_staggered_demean(
            data=data,
            y='y',
            gvar='gvar',
            tvar='year',
            ivar='unit_id'
        )
        
        results = estimate_cohort_time_effects(
            data_transformed=data_transformed,
            gvar='gvar',
            tvar='year',
            ivar='unit_id',
            estimator='ipwra',
            alpha=alpha
        )
        
        if len(results) > 0:
            result = results[0]
            
            # IPWRA 使用正态分布
            z_crit = stats.norm.ppf(1 - alpha / 2)
            expected_ci_lower_z = result.att - z_crit * result.se
            expected_ci_upper_z = result.att + z_crit * result.se
            
            # 验证 CI 与正态分布计算一致
            assert abs(result.ci_lower - expected_ci_lower_z) < 1e-4, (
                f"IPWRA CI lower should use normal distribution: "
                f"expected {expected_ci_lower_z:.6f}, got {result.ci_lower:.6f}"
            )
            assert abs(result.ci_upper - expected_ci_upper_z) < 1e-4, (
                f"IPWRA CI upper should use normal distribution: "
                f"expected {expected_ci_upper_z:.6f}, got {result.ci_upper:.6f}"
            )


# ============================================================================
# Section 4: 使用 vibe-math-mcp 公式验证
# ============================================================================

class TestVibeMathValidation:
    """使用数学公式验证 CI 计算"""
    
    def test_normal_quantile_calculation(self):
        """验证正态分位数计算"""
        from scipy.stats import norm
        
        # 验证常用分位数
        test_cases = [
            (0.975, 1.96),   # 对应 95% CI
            (0.95, 1.645),   # 对应 90% CI
            (0.995, 2.576),  # 对应 99% CI
            (0.90, 1.282),   # 对应 80% CI
        ]
        
        for prob, expected_z in test_cases:
            calculated_z = norm.ppf(prob)
            assert abs(calculated_z - expected_z) < 0.01, (
                f"Normal quantile at {prob}: expected ≈{expected_z}, "
                f"got {calculated_z:.4f}"
            )
    
    def test_ci_coverage_probability(self):
        """验证 CI 覆盖概率计算"""
        # 对于标准正态分布，(1-α) CI 应该覆盖 (1-α) 的样本
        from scipy.stats import norm
        
        for alpha in [0.01, 0.05, 0.10, 0.20]:
            z = norm.ppf(1 - alpha / 2)
            coverage = 2 * norm.cdf(z) - 1
            expected_coverage = 1 - alpha
            
            assert abs(coverage - expected_coverage) < 1e-10, (
                f"Coverage probability for alpha={alpha}: "
                f"expected {expected_coverage}, got {coverage:.6f}"
            )
    
    def test_ci_formula_derivation(self):
        """验证 CI 公式推导正确性"""
        # CI 公式: θ̂ ± z_{1-α/2} × SE(θ̂)
        # 对于双侧 CI，两侧概率相等
        
        att = 1.0
        se = 0.2
        alpha = 0.05
        
        z = stats.norm.ppf(1 - alpha / 2)
        
        # 方法1: 直接计算
        ci_lower_1 = att - z * se
        ci_upper_1 = att + z * se
        
        # 方法2: 使用 scipy.stats.norm.interval
        ci_lower_2, ci_upper_2 = stats.norm.interval(1 - alpha, loc=att, scale=se)
        
        # 两种方法应该给出相同结果
        assert abs(ci_lower_1 - ci_lower_2) < 1e-10
        assert abs(ci_upper_1 - ci_upper_2) < 1e-10


# ============================================================================
# Section 5: 边界条件测试
# ============================================================================

class TestCIBoundaryConditions:
    """测试 CI 计算的边界条件"""
    
    def test_zero_se_ci(self):
        """SE=0 时 CI 应该退化为点估计"""
        att = 0.5
        se = 0.0
        alpha = 0.05
        
        z = stats.norm.ppf(1 - alpha / 2)
        ci_lower = att - z * se
        ci_upper = att + z * se
        
        assert ci_lower == ci_upper == att
    
    def test_very_small_alpha_ci(self):
        """非常小的 alpha 应该产生很宽的 CI"""
        att = 0.5
        se = 0.1
        alpha = 0.001  # 99.9% CI
        
        z = stats.norm.ppf(1 - alpha / 2)
        width = 2 * z * se
        
        # 99.9% CI 应该比 99% CI 更宽
        z_99 = stats.norm.ppf(0.995)
        width_99 = 2 * z_99 * se
        
        assert width > width_99
    
    def test_negative_att_ci(self):
        """负的 ATT 应该有正确的 CI"""
        att = -0.5
        se = 0.1
        alpha = 0.05
        
        z = stats.norm.ppf(1 - alpha / 2)
        ci_lower = att - z * se
        ci_upper = att + z * se
        
        # CI 应该包含 ATT
        assert ci_lower < att < ci_upper
        # 对于负的 ATT，CI 上界可能为正
        assert ci_lower < 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

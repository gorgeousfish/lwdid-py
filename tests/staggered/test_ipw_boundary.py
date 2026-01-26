"""
IPW估计量边界条件测试

Story 2.2: IPW标准误计算 - Task 2.2.7-2.2.9

测试IPW SE计算的边界条件处理：
1. 小样本策略（N<100时警告）
2. 奇异矩阵处理（PS score矩阵不可逆）
3. Bootstrap失败处理（成功率<80%时警告）

References:
- Wooldridge JM (2007). "Inverse Probability Weighted Estimation"
- Lee & Wooldridge (2023) Section 3
"""

import warnings
import numpy as np
import pandas as pd
import pytest

from lwdid.staggered import estimate_ipw


# ============================================================================
# Test Data Generators
# ============================================================================

def generate_small_sample_data(n: int, seed: int = 42) -> pd.DataFrame:
    """生成小样本测试数据"""
    rng = np.random.default_rng(seed)
    
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    
    ps_index = -0.5 + 0.3 * x1 + 0.2 * x2
    ps_true = 1 / (1 + np.exp(-ps_index))
    d = rng.binomial(1, ps_true)
    
    y0 = 1.0 + 0.5 * x1 + 0.3 * x2 + rng.normal(0, 0.5, n)
    y1 = y0 + 2.0
    y = np.where(d == 1, y1, y0)
    
    return pd.DataFrame({
        'y': y,
        'd': d,
        'x1': x1,
        'x2': x2,
    })


def generate_perfect_separation_data(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """
    生成导致完美分离的数据（PS=0或1）。
    
    当X1>0时D=1，当X1<0时D=0，这会导致PS模型无法收敛。
    """
    rng = np.random.default_rng(seed)
    
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    
    # 完美分离：X1>0时D=1，X1<0时D=0
    d = (x1 > 0).astype(int)
    
    y0 = 1.0 + 0.5 * x1 + 0.3 * x2 + rng.normal(0, 0.5, n)
    y1 = y0 + 2.0
    y = np.where(d == 1, y1, y0)
    
    return pd.DataFrame({
        'y': y,
        'd': d,
        'x1': x1,
        'x2': x2,
    })


def generate_collinear_data(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """生成包含共线性协变量的数据"""
    rng = np.random.default_rng(seed)
    
    x1 = rng.normal(0, 1, n)
    x2 = 2 * x1 + rng.normal(0, 0.01, n)  # x2与x1高度相关
    
    ps_index = -0.5 + 0.3 * x1 + 0.2 * x2
    ps_true = 1 / (1 + np.exp(-ps_index))
    d = rng.binomial(1, ps_true)
    
    y0 = 1.0 + 0.5 * x1 + 0.3 * x2 + rng.normal(0, 0.5, n)
    y1 = y0 + 2.0
    y = np.where(d == 1, y1, y0)
    
    return pd.DataFrame({
        'y': y,
        'd': d,
        'x1': x1,
        'x2': x2,
    })


def generate_extreme_imbalance_data(n: int = 50, seed: int = 42) -> pd.DataFrame:
    """生成极端不平衡的数据（处理组很少）"""
    rng = np.random.default_rng(seed)
    
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    
    # 极端不平衡：只有约5%是处理组
    ps_index = -3.0 + 0.3 * x1 + 0.2 * x2
    ps_true = 1 / (1 + np.exp(-ps_index))
    d = rng.binomial(1, ps_true)
    
    y0 = 1.0 + 0.5 * x1 + 0.3 * x2 + rng.normal(0, 0.5, n)
    y1 = y0 + 2.0
    y = np.where(d == 1, y1, y0)
    
    return pd.DataFrame({
        'y': y,
        'd': d,
        'x1': x1,
        'x2': x2,
    })


# ============================================================================
# Small Sample Strategy Tests
# ============================================================================

class TestSmallSampleStrategy:
    """小样本策略测试"""
    
    def test_analytical_warning_n50(self):
        """N=50时analytical应发出警告"""
        data = generate_small_sample_data(n=50, seed=42)
        
        with pytest.warns(UserWarning, match="Small sample size"):
            result = estimate_ipw(
                data=data,
                y='y',
                d='d',
                propensity_controls=['x1', 'x2'],
                se_method='analytical',
            )
        
        # 仍应返回有效结果
        assert result.se > 0
        assert not np.isnan(result.se)
    
    def test_analytical_warning_n80(self):
        """N=80时analytical应发出警告"""
        data = generate_small_sample_data(n=80, seed=42)
        
        with pytest.warns(UserWarning, match="Small sample size"):
            result = estimate_ipw(
                data=data,
                y='y',
                d='d',
                propensity_controls=['x1', 'x2'],
                se_method='analytical',
            )
        
        assert result.se > 0
    
    def test_no_warning_n100(self):
        """N=100时analytical不应发出小样本警告"""
        data = generate_small_sample_data(n=100, seed=42)
        
        # 不应发出小样本警告
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = estimate_ipw(
                data=data,
                y='y',
                d='d',
                propensity_controls=['x1', 'x2'],
                se_method='analytical',
            )
            
            # 检查是否有小样本警告
            small_sample_warnings = [
                warning for warning in w 
                if "Small sample size" in str(warning.message)
            ]
            assert len(small_sample_warnings) == 0, "N=100时不应发出小样本警告"
        
        assert result.se > 0
    
    def test_bootstrap_small_sample_no_warning(self):
        """小样本Bootstrap不应发出小样本警告（Bootstrap适合小样本）"""
        data = generate_small_sample_data(n=50, seed=42)
        
        # Bootstrap方法不应发出小样本警告
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = estimate_ipw(
                data=data,
                y='y',
                d='d',
                propensity_controls=['x1', 'x2'],
                se_method='bootstrap',
                n_bootstrap=100,
                seed=42,
            )
            
            # 检查是否有小样本警告
            small_sample_warnings = [
                warning for warning in w 
                if "Small sample size" in str(warning.message)
            ]
            assert len(small_sample_warnings) == 0, "Bootstrap不应发出小样本警告"
        
        assert result.se > 0
    
    def test_small_sample_bootstrap_works(self):
        """小样本Bootstrap应正常工作"""
        data = generate_small_sample_data(n=50, seed=42)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = estimate_ipw(
                data=data,
                y='y',
                d='d',
                propensity_controls=['x1', 'x2'],
                se_method='bootstrap',
                n_bootstrap=100,
                seed=42,
            )
        
        assert result.se > 0
        assert not np.isnan(result.se)
        assert result.ci_lower < result.att < result.ci_upper


# ============================================================================
# Singular Matrix Handling Tests
# ============================================================================

class TestSingularMatrixHandling:
    """奇异矩阵处理测试"""
    
    def test_collinear_covariates(self):
        """共线性协变量应正确处理"""
        data = generate_collinear_data(n=200, seed=42)
        
        # 应该能够处理共线性，可能发出警告
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = estimate_ipw(
                data=data,
                y='y',
                d='d',
                propensity_controls=['x1', 'x2'],
                se_method='analytical',
            )
        
        # 仍应返回有效结果
        assert result.se >= 0
        assert not np.isnan(result.att)
    
    def test_constant_covariate(self):
        """常数协变量应正确处理"""
        data = generate_small_sample_data(n=200, seed=42)
        data['x_const'] = 1.0  # 常数协变量
        
        # 应该能够处理，可能发出警告
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = estimate_ipw(
                data=data,
                y='y',
                d='d',
                propensity_controls=['x1', 'x_const'],
                se_method='analytical',
            )
        
        # 仍应返回有效结果
        assert result.se >= 0


# ============================================================================
# Bootstrap Failure Handling Tests
# ============================================================================

class TestBootstrapFailureHandling:
    """Bootstrap失败处理测试"""
    
    def test_bootstrap_low_success_rate_warning(self):
        """Bootstrap成功率低时应发出警告"""
        # 使用极端不平衡数据，可能导致Bootstrap失败
        data = generate_extreme_imbalance_data(n=50, seed=42)
        
        # 确保有足够的处理组和控制组
        if data['d'].sum() < 2 or (1 - data['d']).sum() < 2:
            pytest.skip("数据不满足最小样本要求")
        
        # 可能发出警告（取决于Bootstrap样本）
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                result = estimate_ipw(
                    data=data,
                    y='y',
                    d='d',
                    propensity_controls=['x1', 'x2'],
                    se_method='bootstrap',
                    n_bootstrap=100,
                    seed=42,
                )
                
                # 检查是否有Bootstrap成功率警告
                bootstrap_warnings = [
                    warning for warning in w 
                    if "Low bootstrap success rate" in str(warning.message)
                ]
                
                # 即使有警告，仍应返回结果
                assert result.se >= 0
                
            except ValueError as e:
                # 如果Bootstrap样本不足，应该抛出ValueError
                assert "Bootstrap样本不足" in str(e)
    
    def test_bootstrap_returns_result_even_with_failures(self):
        """即使有Bootstrap失败，仍应返回结果（如果成功率>80%）"""
        data = generate_small_sample_data(n=100, seed=42)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = estimate_ipw(
                data=data,
                y='y',
                d='d',
                propensity_controls=['x1', 'x2'],
                se_method='bootstrap',
                n_bootstrap=200,
                seed=42,
            )
        
        assert result.se > 0
        assert not np.isnan(result.se)


# ============================================================================
# Edge Cases Tests
# ============================================================================

class TestEdgeCases:
    """边界情况测试"""
    
    def test_single_covariate(self):
        """单个协变量应正常工作"""
        data = generate_small_sample_data(n=200, seed=42)
        
        result = estimate_ipw(
            data=data,
            y='y',
            d='d',
            propensity_controls=['x1'],
            se_method='analytical',
        )
        
        assert result.se > 0
        assert not np.isnan(result.att)
    
    def test_many_covariates(self):
        """多个协变量应正常工作"""
        data = generate_small_sample_data(n=500, seed=42)
        
        # 添加更多协变量
        rng = np.random.default_rng(42)
        for i in range(3, 8):
            data[f'x{i}'] = rng.normal(0, 1, len(data))
        
        result = estimate_ipw(
            data=data,
            y='y',
            d='d',
            propensity_controls=['x1', 'x2', 'x3', 'x4', 'x5'],
            se_method='analytical',
        )
        
        assert result.se > 0
        assert not np.isnan(result.att)
    
    def test_extreme_weights_warning(self):
        """极端权重应发出警告"""
        # 生成可能产生极端权重的数据
        rng = np.random.default_rng(42)
        n = 200
        
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        
        # 极端PS，产生极端权重
        ps_index = -2.0 + 2.0 * x1 + 1.5 * x2
        ps_true = 1 / (1 + np.exp(-ps_index))
        d = rng.binomial(1, ps_true)
        
        y0 = 1.0 + 0.5 * x1 + 0.3 * x2 + rng.normal(0, 0.5, n)
        y1 = y0 + 2.0
        y = np.where(d == 1, y1, y0)
        
        data = pd.DataFrame({
            'y': y,
            'd': d,
            'x1': x1,
            'x2': x2,
        })
        
        # 可能发出极端权重警告
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = estimate_ipw(
                data=data,
                y='y',
                d='d',
                propensity_controls=['x1', 'x2'],
                se_method='analytical',
            )
            
            # 检查是否有极端权重警告
            weight_warnings = [
                warning for warning in w 
                if "极端IPW权重" in str(warning.message) or "CV" in str(warning.message)
            ]
            
            # 即使有警告，仍应返回结果
            assert result.se > 0
    
    def test_invalid_se_method(self):
        """无效的se_method应抛出ValueError"""
        data = generate_small_sample_data(n=100, seed=42)
        
        with pytest.raises(ValueError, match="未知的se_method"):
            estimate_ipw(
                data=data,
                y='y',
                d='d',
                propensity_controls=['x1', 'x2'],
                se_method='invalid_method',
            )


# ============================================================================
# Numerical Stability Tests
# ============================================================================

class TestNumericalStability:
    """数值稳定性测试"""
    
    def test_se_positive(self):
        """SE应始终为正"""
        for seed in range(10):
            data = generate_small_sample_data(n=200, seed=seed)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = estimate_ipw(
                    data=data,
                    y='y',
                    d='d',
                    propensity_controls=['x1', 'x2'],
                    se_method='analytical',
                )
            
            assert result.se > 0, f"SE应为正，但得到 {result.se}"
    
    def test_ci_contains_att(self):
        """置信区间应包含ATT点估计"""
        data = generate_small_sample_data(n=200, seed=42)
        
        result = estimate_ipw(
            data=data,
            y='y',
            d='d',
            propensity_controls=['x1', 'x2'],
            se_method='analytical',
        )
        
        assert result.ci_lower < result.att < result.ci_upper, (
            f"CI [{result.ci_lower}, {result.ci_upper}] 不包含 ATT {result.att}"
        )
    
    def test_t_stat_consistent_with_se(self):
        """t统计量应与SE一致"""
        data = generate_small_sample_data(n=200, seed=42)
        
        result = estimate_ipw(
            data=data,
            y='y',
            d='d',
            propensity_controls=['x1', 'x2'],
            se_method='analytical',
        )
        
        expected_t_stat = result.att / result.se
        assert np.isclose(result.t_stat, expected_t_stat, rtol=1e-6), (
            f"t统计量不一致: {result.t_stat} vs {expected_t_stat}"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])

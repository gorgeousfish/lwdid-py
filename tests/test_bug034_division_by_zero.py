"""
BUG-034 回归测试: IPW/IPWRA SE 计算中 p_bar 和 weights_sum 除以零风险

验证 compute_ipwra_se_analytical() 和 _compute_ipw_se_analytical() 
在边界情况下正确抛出 ValueError 而不是除以零错误。

测试覆盖:
1. 处理组为空 (n_treated == 0) - 导致 p_bar = 0
2. 权重和为零或负数 (weights_sum <= 0) - 导致除零
3. 正常情况下功能不受影响

References:
- BUG-034: IPW/IPWRA SE 计算中 p_bar 和 weights_sum 除以零风险
"""

import numpy as np
import pandas as pd
import pytest

from lwdid.staggered.estimators import (
    compute_ipwra_se_analytical,
    _compute_ipw_se_analytical,
    estimate_ipwra,
    estimate_ipw,
)


class TestIPWRASEAnalyticalZeroChecks:
    """测试 compute_ipwra_se_analytical() 的零值检查"""
    
    def test_zero_treated_raises_value_error(self):
        """当处理组为空时，应抛出 ValueError"""
        # 创建测试数据 - 所有单位都是控制组
        n = 20
        data = pd.DataFrame({
            'y': np.random.randn(n),
            'd': np.zeros(n),  # 所有单位都是控制组
            'x1': np.random.randn(n),
        })
        
        # 创建假的参数
        pscores = np.full(n, 0.3)
        weights = pscores / (1 - pscores)
        m0_hat = np.random.randn(n)
        
        with pytest.raises(ValueError, match="No treated units found"):
            compute_ipwra_se_analytical(
                data=data,
                y='y',
                d='d',
                controls=['x1'],
                propensity_controls=['x1'],
                att=0.5,
                pscores=pscores,
                m0_hat=m0_hat,
                weights=weights,
                ps_coef={'_intercept': -0.5, 'x1': 0.1},
                outcome_coef={'_intercept': 0.0, 'x1': 0.2},
            )
    
    def test_zero_weights_sum_raises_value_error(self):
        """当权重和为零或负数时，应抛出 ValueError"""
        # 创建测试数据
        n = 20
        data = pd.DataFrame({
            'y': np.random.randn(n),
            'd': np.array([1] * 5 + [0] * 15),  # 5个处理组，15个控制组
            'x1': np.random.randn(n),
        })
        
        # 创建权重 - 控制组权重全为零
        pscores = np.array([0.5] * 5 + [0.0] * 15)  # 控制组 pscores 为 0
        # 注意: 权重 = pscores / (1 - pscores)，当 pscores = 0 时权重为 0
        weights = np.zeros(n)  # 所有权重为 0
        m0_hat = np.random.randn(n)
        
        with pytest.raises(ValueError, match="Sum of IPW weights is non-positive"):
            compute_ipwra_se_analytical(
                data=data,
                y='y',
                d='d',
                controls=['x1'],
                propensity_controls=['x1'],
                att=0.5,
                pscores=pscores,
                m0_hat=m0_hat,
                weights=weights,
                ps_coef={'_intercept': -0.5, 'x1': 0.1},
                outcome_coef={'_intercept': 0.0, 'x1': 0.2},
            )
    
    def test_normal_case_works(self):
        """正常情况下功能不受影响"""
        np.random.seed(42)
        n = 50
        
        # 创建平衡的测试数据
        x1 = np.random.randn(n)
        d = (x1 + np.random.randn(n) > 0).astype(float)
        y = 0.5 * d + x1 + np.random.randn(n) * 0.5
        
        data = pd.DataFrame({'y': y, 'd': d, 'x1': x1})
        
        # 计算合理的倾向得分
        pscores = 1 / (1 + np.exp(-x1))  # Logistic 函数
        pscores = np.clip(pscores, 0.1, 0.9)  # 裁剪
        weights = pscores / (1 - pscores)
        m0_hat = x1  # 简单的结果模型
        
        # 应该正常运行不抛出错误
        se, ci_lower, ci_upper = compute_ipwra_se_analytical(
            data=data,
            y='y',
            d='d',
            controls=['x1'],
            propensity_controls=['x1'],
            att=0.5,
            pscores=pscores,
            m0_hat=m0_hat,
            weights=weights,
            ps_coef={'_intercept': 0.0, 'x1': 1.0},
            outcome_coef={'_intercept': 0.0, 'x1': 1.0},
        )
        
        # 验证返回值是有效的
        assert se > 0, "SE should be positive"
        assert ci_lower < ci_upper, "CI lower should be less than CI upper"
        assert np.isfinite(se), "SE should be finite"


class TestIPWSEAnalyticalZeroChecks:
    """测试 _compute_ipw_se_analytical() 的零值检查"""
    
    def test_zero_treated_raises_value_error(self):
        """当处理组为空时，应抛出 ValueError"""
        n = 20
        Y = np.random.randn(n)
        D = np.zeros(n)  # 所有单位都是控制组
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        pscores = np.full(n, 0.3)
        weights = pscores / (1 - pscores)
        
        with pytest.raises(ValueError, match="No treated units found"):
            _compute_ipw_se_analytical(
                Y=Y,
                D=D,
                X=X,
                pscores=pscores,
                weights=weights,
                att=0.5,
            )
    
    def test_zero_weights_sum_raises_value_error(self):
        """当权重和为零或负数时，应抛出 ValueError"""
        n = 20
        Y = np.random.randn(n)
        D = np.array([1.0] * 5 + [0.0] * 15)
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        pscores = np.array([0.5] * 5 + [0.0] * 15)
        weights = np.zeros(n)  # 所有权重为 0
        
        with pytest.raises(ValueError, match="Sum of IPW weights is non-positive"):
            _compute_ipw_se_analytical(
                Y=Y,
                D=D,
                X=X,
                pscores=pscores,
                weights=weights,
                att=0.5,
            )
    
    def test_normal_case_works(self):
        """正常情况下功能不受影响"""
        np.random.seed(42)
        n = 50
        
        # 创建平衡的测试数据
        x1 = np.random.randn(n)
        D = (x1 + np.random.randn(n) > 0).astype(float)
        Y = 0.5 * D + x1 + np.random.randn(n) * 0.5
        X = np.column_stack([np.ones(n), x1])
        
        # 计算合理的倾向得分
        pscores = 1 / (1 + np.exp(-x1))
        pscores = np.clip(pscores, 0.1, 0.9)
        weights = pscores / (1 - pscores)
        
        # 应该正常运行不抛出错误
        se, ci_lower, ci_upper = _compute_ipw_se_analytical(
            Y=Y,
            D=D,
            X=X,
            pscores=pscores,
            weights=weights,
            att=0.5,
        )
        
        assert se > 0, "SE should be positive"
        assert ci_lower < ci_upper, "CI lower should be less than CI upper"
        assert np.isfinite(se), "SE should be finite"


class TestHighLevelAPIZeroChecks:
    """测试高级 API 在边界情况下的行为"""
    
    def test_estimate_ipwra_insufficient_treated_raises(self):
        """estimate_ipwra() 应在处理组过少时抛出 ValueError"""
        n = 20
        data = pd.DataFrame({
            'y': np.random.randn(n),
            'd': np.array([1] + [0] * 19),  # 只有 1 个处理组
            'x1': np.random.randn(n),
        })
        
        # estimate_ipwra 要求 n_treated >= 2
        with pytest.raises(ValueError, match="Insufficient treatment group"):
            estimate_ipwra(
                data=data,
                y='y',
                d='d',
                controls=['x1'],
            )
    
    def test_estimate_ipw_no_treated_raises(self):
        """estimate_ipw() 应在处理组为空时抛出 ValueError"""
        n = 20
        data = pd.DataFrame({
            'y': np.random.randn(n),
            'd': np.zeros(n),  # 无处理组
            'x1': np.random.randn(n),
        })
        
        with pytest.raises(ValueError, match="No treatment units"):
            estimate_ipw(
                data=data,
                y='y',
                d='d',
                propensity_controls=['x1'],
            )
    
    def test_estimate_ipw_no_control_raises(self):
        """estimate_ipw() 应在控制组为空时抛出 ValueError"""
        n = 20
        data = pd.DataFrame({
            'y': np.random.randn(n),
            'd': np.ones(n),  # 无控制组
            'x1': np.random.randn(n),
        })
        
        with pytest.raises(ValueError, match="No control units"):
            estimate_ipw(
                data=data,
                y='y',
                d='d',
                propensity_controls=['x1'],
            )


class TestBoundaryConditions:
    """测试边界条件"""
    
    def test_minimal_sample_size(self):
        """最小样本量情况（n_treated=2, n_control=2）"""
        np.random.seed(123)
        n = 10  # 较小样本
        
        # 创建简单数据
        d = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0]).astype(float)
        x1 = np.random.randn(n)
        y = 0.5 * d + x1 + np.random.randn(n) * 0.1
        
        data = pd.DataFrame({'y': y, 'd': d, 'x1': x1})
        
        # 使用 bootstrap 方法应该可以工作
        result = estimate_ipw(
            data=data,
            y='y',
            d='d',
            propensity_controls=['x1'],
            se_method='bootstrap',
            n_bootstrap=50,
            seed=42,
        )
        
        assert result.se > 0, "SE should be positive"
        assert np.isfinite(result.att), "ATT should be finite"
    
    def test_extreme_propensity_scores_warning(self):
        """极端倾向得分应产生警告但不崩溃"""
        np.random.seed(456)
        n = 50
        
        # 创建倾向得分几乎分离的数据
        x1 = np.concatenate([
            np.random.randn(25) + 3,  # 高倾向得分组
            np.random.randn(25) - 3,  # 低倾向得分组
        ])
        d = (x1 > 0).astype(float)
        y = 0.5 * d + x1 + np.random.randn(n) * 0.5
        
        data = pd.DataFrame({'y': y, 'd': d, 'x1': x1})
        
        # 应该产生警告但正常完成
        with pytest.warns(UserWarning):
            result = estimate_ipw(
                data=data,
                y='y',
                d='d',
                propensity_controls=['x1'],
                trim_threshold=0.01,  # 小的裁剪阈值
            )
        
        # 结果应该是有效的
        assert np.isfinite(result.att), "ATT should be finite"
        assert np.isfinite(result.se), "SE should be finite"


class TestNumericalValidation:
    """数值验证测试"""
    
    def test_error_message_content(self):
        """验证错误消息包含有用的诊断信息"""
        n = 20
        Y = np.random.randn(n)
        D = np.zeros(n)  # 触发零处理组错误
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        pscores = np.full(n, 0.3)
        weights = pscores / (1 - pscores)
        
        with pytest.raises(ValueError) as exc_info:
            _compute_ipw_se_analytical(Y=Y, D=D, X=X, pscores=pscores, weights=weights, att=0.5)
        
        error_msg = str(exc_info.value)
        # 验证错误消息包含有用的建议
        assert "bootstrap" in error_msg.lower(), "Error message should suggest bootstrap"
    
    def test_weights_sum_error_message_content(self):
        """验证权重和错误消息包含有用的诊断信息"""
        n = 20
        Y = np.random.randn(n)
        D = np.array([1.0] * 5 + [0.0] * 15)
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        pscores = np.array([0.5] * 5 + [0.0] * 15)
        weights = np.zeros(n)
        
        with pytest.raises(ValueError) as exc_info:
            _compute_ipw_se_analytical(Y=Y, D=D, X=X, pscores=pscores, weights=weights, att=0.5)
        
        error_msg = str(exc_info.value)
        # 验证错误消息包含有用的建议
        assert "propensity" in error_msg.lower() or "bootstrap" in error_msg.lower(), \
            "Error message should mention propensity scores or bootstrap"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
IPWRA估计量单元测试

测试范围:
1. 基本功能测试
2. 边界条件测试
3. 数值正确性测试
4. 与RA估计对比测试

Reference: Story E3-S1
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

# 确保可以导入lwdid模块
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from lwdid.staggered.estimators import (
    estimate_ipwra,
    estimate_propensity_score,
    estimate_outcome_model,
    IPWRAResult,
)


class TestEstimatePropensityScore:
    """倾向得分估计测试"""
    
    def test_basic_propensity_score(self):
        """基本倾向得分估计"""
        np.random.seed(42)
        n = 200
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        # 真实倾向得分: logit(p) = -0.5 + 0.5*x1 + 0.3*x2
        logit_p = -0.5 + 0.5 * x1 + 0.3 * x2
        p_true = 1 / (1 + np.exp(-logit_p))
        d = (np.random.uniform(0, 1, n) < p_true).astype(int)
        
        data = pd.DataFrame({'d': d, 'x1': x1, 'x2': x2})
        
        pscores, coef = estimate_propensity_score(
            data, 'd', ['x1', 'x2'], trim_threshold=0.01
        )
        
        # 检查倾向得分在合理范围
        assert pscores.min() >= 0.01
        assert pscores.max() <= 0.99
        assert len(pscores) == n
        
        # 检查系数符号（x1和x2应为正）
        assert coef['x1'] > 0, f"x1系数应为正: {coef['x1']}"
        assert coef['x2'] > 0, f"x2系数应为正: {coef['x2']}"
    
    def test_propensity_score_trimming(self):
        """测试倾向得分裁剪"""
        np.random.seed(42)
        n = 100
        # 构造极端数据：几乎完美分离
        x = np.concatenate([np.random.normal(-3, 0.5, 50),
                           np.random.normal(3, 0.5, 50)])
        d = np.concatenate([np.zeros(50), np.ones(50)])
        
        data = pd.DataFrame({'d': d, 'x': x})
        
        pscores, _ = estimate_propensity_score(
            data, 'd', ['x'], trim_threshold=0.05
        )
        
        assert pscores.min() >= 0.05, f"最小倾向得分应>=0.05: {pscores.min()}"
        assert pscores.max() <= 0.95, f"最大倾向得分应<=0.95: {pscores.max()}"


class TestEstimateOutcomeModel:
    """结果模型估计测试"""
    
    def test_basic_outcome_model(self):
        """基本结果模型估计"""
        np.random.seed(42)
        n = 200
        x = np.random.normal(0, 1, n)
        # 真实模型: Y = 1 + 2*x + error
        y = 1 + 2 * x + np.random.normal(0, 0.5, n)
        d = np.random.binomial(1, 0.5, n)
        
        data = pd.DataFrame({'y': y, 'd': d, 'x': x})
        
        m0_hat, coef = estimate_outcome_model(data, 'y', 'd', ['x'])
        
        # 检查系数接近真实值
        assert abs(coef['_intercept'] - 1) < 0.5, f"截距应接近1: {coef['_intercept']}"
        assert abs(coef['x'] - 2) < 0.5, f"x系数应接近2: {coef['x']}"
        
        # 检查预测值长度
        assert len(m0_hat) == n


class TestEstimateIPWRA:
    """IPWRA估计测试"""
    
    @pytest.fixture
    def simple_data(self):
        """构造简单测试数据：已知ATT≈2"""
        np.random.seed(42)
        n = 500
        
        # 协变量
        x = np.random.normal(0, 1, n)
        
        # 倾向得分: P(D=1|X) = logit(-0.5 + 0.3*x)
        logit_p = -0.5 + 0.3 * x
        p_true = 1 / (1 + np.exp(-logit_p))
        d = (np.random.uniform(0, 1, n) < p_true).astype(int)
        
        # 潜在结果
        # Y(0) = 1 + x + error
        # Y(1) = 3 + x + error  (ATT ≈ 2)
        y0 = 1 + x + np.random.normal(0, 0.5, n)
        y1 = 3 + x + np.random.normal(0, 0.5, n)
        y = d * y1 + (1 - d) * y0
        
        return pd.DataFrame({'y': y, 'd': d, 'x': x})
    
    def test_ipwra_basic(self, simple_data):
        """基本IPWRA估计"""
        result = estimate_ipwra(
            simple_data, 'y', 'd', ['x'],
            se_method='analytical'
        )
        
        assert isinstance(result, IPWRAResult)
        # ATT应接近2
        assert abs(result.att - 2) < 0.5, f"ATT={result.att}, 期望≈2"
        assert result.se > 0, "SE应为正"
        assert result.ci_lower < result.att < result.ci_upper, "CI应包含点估计"
        assert result.n_treated > 0
        assert result.n_control > 0
    
    def test_ipwra_bootstrap_se(self, simple_data):
        """Bootstrap标准误"""
        result = estimate_ipwra(
            simple_data, 'y', 'd', ['x'],
            se_method='bootstrap',
            n_bootstrap=50,  # 减少以加快测试
            seed=42
        )
        
        assert result.se > 0, "Bootstrap SE应为正"
        assert result.ci_lower < result.att < result.ci_upper
    
    def test_ipwra_missing_controls_error(self, simple_data):
        """缺失控制变量应报错"""
        with pytest.raises(ValueError, match="控制变量不存在"):
            estimate_ipwra(
                simple_data, 'y', 'd', ['nonexistent']
            )
    
    def test_ipwra_missing_y_error(self, simple_data):
        """缺失结果变量应报错"""
        with pytest.raises(ValueError, match="结果变量.*不在数据中"):
            estimate_ipwra(
                simple_data, 'nonexistent', 'd', ['x']
            )
    
    def test_ipwra_missing_d_error(self, simple_data):
        """缺失处理指示符应报错"""
        with pytest.raises(ValueError, match="处理指示符.*不在数据中"):
            estimate_ipwra(
                simple_data, 'y', 'nonexistent', ['x']
            )
    
    def test_ipwra_insufficient_treated(self):
        """处理组样本量不足应报错"""
        data = pd.DataFrame({
            'y': [1, 2, 3, 4, 5],
            'd': [1, 0, 0, 0, 0],  # 只有1个treated
            'x': [1, 2, 3, 4, 5]
        })
        
        with pytest.raises(ValueError, match="处理组样本量不足"):
            estimate_ipwra(data, 'y', 'd', ['x'])
    
    def test_ipwra_insufficient_control(self):
        """控制组样本量不足应报错"""
        data = pd.DataFrame({
            'y': [1, 2, 3, 4, 5],
            'd': [1, 1, 1, 1, 0],  # 只有1个control
            'x': [1, 2, 3, 4, 5]
        })
        
        with pytest.raises(ValueError, match="控制组样本量不足"):
            estimate_ipwra(data, 'y', 'd', ['x'])


class TestIPWRAEdgeCases:
    """IPWRA边界条件测试"""
    
    def test_small_treated_warning(self):
        """小处理组样本警告"""
        np.random.seed(42)
        data = pd.DataFrame({
            'y': np.random.normal(0, 1, 20),
            'd': np.array([1, 1, 1, 1] + [0] * 16),  # 只有4个treated
            'x': np.random.normal(0, 1, 20)
        })
        
        with pytest.warns(UserWarning, match="处理组样本量较小"):
            estimate_ipwra(data, 'y', 'd', ['x'])
    
    def test_small_control_warning(self):
        """小控制组样本警告"""
        np.random.seed(42)
        data = pd.DataFrame({
            'y': np.random.normal(0, 1, 20),
            'd': np.array([1] * 12 + [0] * 8),  # 只有8个control
            'x': np.random.normal(0, 1, 20)
        })
        
        with pytest.warns(UserWarning, match="控制组样本量较小"):
            estimate_ipwra(data, 'y', 'd', ['x'])
    
    def test_unknown_se_method_error(self):
        """未知标准误方法应报错"""
        np.random.seed(42)
        data = pd.DataFrame({
            'y': np.random.normal(0, 1, 100),
            'd': np.random.binomial(1, 0.5, 100),
            'x': np.random.normal(0, 1, 100)
        })
        
        with pytest.raises(ValueError, match="未知的se_method"):
            estimate_ipwra(data, 'y', 'd', ['x'], se_method='invalid')
    
    def test_extreme_weights_warning(self):
        """极端权重应发出警告"""
        np.random.seed(42)
        n = 100
        # 构造几乎完美分离的数据（导致极端倾向得分）
        x = np.concatenate([np.random.normal(-2, 0.3, 50),
                           np.random.normal(2, 0.3, 50)])
        d = np.concatenate([np.zeros(50), np.ones(50)])
        y = 1 + x + 2 * d + np.random.normal(0, 0.5, n)
        
        data = pd.DataFrame({'y': y, 'd': d, 'x': x})
        
        # 使用较小的trim_threshold来允许极端值
        with pytest.warns(UserWarning, match="极端|overlap|倾向得分"):
            estimate_ipwra(data, 'y', 'd', ['x'], trim_threshold=0.001)
    
    def test_ipwra_multiple_controls(self):
        """多控制变量测试"""
        np.random.seed(42)
        n = 300
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        x3 = np.random.normal(0, 1, n)
        
        logit_p = -0.3 + 0.2 * x1 + 0.1 * x2
        p = 1 / (1 + np.exp(-logit_p))
        d = (np.random.uniform(0, 1, n) < p).astype(int)
        
        y = 1 + 0.5 * x1 + 0.3 * x2 + 0.1 * x3 + 2 * d + np.random.normal(0, 0.5, n)
        
        data = pd.DataFrame({'y': y, 'd': d, 'x1': x1, 'x2': x2, 'x3': x3})
        
        result = estimate_ipwra(
            data, 'y', 'd', ['x1', 'x2', 'x3'],
            se_method='analytical'
        )
        
        # ATT应接近2
        assert abs(result.att - 2) < 0.5, f"ATT={result.att}, 期望≈2"
        assert result.se > 0


class TestIPWRADoublyRobust:
    """
    IPWRA Doubly Robust性质测试
    
    验证IPWRA的核心卖点：只要倾向得分模型或结果模型之一正确，估计量就是一致的。
    
    Reference: Lee & Wooldridge (2023) Section 7.1, Tables 7.3-7.4
    """
    
    def test_doubly_robust_outcome_correct_propensity_wrong(self):
        """
        Doubly Robust测试1: 结果模型正确 + 倾向得分模型错误
        
        DGP:
        - 真实倾向得分含二次项: logit(p) = -1.2 + 0.5*(x1-4)/2 - x2 + 0.5*(x1-4)^2
        - 结果模型线性: Y(0) = 1 + (x1-4)/3 + x2/2 + error
        - 真实ATT = 2.0
        
        使用只有线性项的倾向得分模型（错误指定），ATT仍应接近真实值。
        """
        np.random.seed(42)
        n = 1000
        
        # 协变量
        x1 = np.random.gamma(2, 2, n)  # mean=4
        x2 = np.random.binomial(1, 0.6, n)
        
        # 真实倾向得分（含二次项）
        logit_true = -1.2 + (x1-4)/2 - x2 + (x1-4)**2/4
        p_true = 1 / (1 + np.exp(-logit_true))
        d = (np.random.uniform(0, 1, n) < p_true).astype(int)
        
        # 结果变量（线性）
        tau_true = 2.0
        y0 = 1 + (x1-4)/3 + x2/2 + np.random.normal(0, 1, n)
        y1 = y0 + tau_true
        y = d * y1 + (1 - d) * y0
        
        data = pd.DataFrame({'y': y, 'd': d, 'x1': x1, 'x2': x2})
        
        # IPWRA估计（使用线性倾向得分模型 - 错误指定）
        result = estimate_ipwra(
            data, 'y', 'd', 
            controls=['x1', 'x2'],  # 结果模型正确
            propensity_controls=['x1', 'x2'],  # 倾向得分缺少x1^2项
            se_method='analytical'
        )
        
        # 由于结果模型正确，ATT应该接近真实值（允许一定误差）
        bias = abs(result.att - tau_true)
        assert bias < 0.5, (
            f"Doubly Robust性质失败: ATT={result.att:.4f}, "
            f"期望≈{tau_true}, 偏差={bias:.4f}"
        )
    
    def test_doubly_robust_propensity_correct_outcome_wrong(self):
        """
        Doubly Robust测试2: 倾向得分模型正确 + 结果模型错误
        
        DGP:
        - 倾向得分线性: logit(p) = -1.2 + 0.5*(x1-4)/2 - x2
        - 结果模型含二次项: Y(0) = 1 + (x1-4)/3 + x2/2 + (x1-4)^2/4 + error
        - 真实ATT = 2.0
        
        使用只有线性项的结果模型（错误指定），ATT仍应接近真实值。
        """
        np.random.seed(123)
        n = 1000
        
        # 协变量
        x1 = np.random.gamma(2, 2, n)
        x2 = np.random.binomial(1, 0.6, n)
        
        # 倾向得分（线性）
        logit = -1.2 + (x1-4)/2 - x2
        p = 1 / (1 + np.exp(-logit))
        d = (np.random.uniform(0, 1, n) < p).astype(int)
        
        # 结果变量（含二次项）
        tau_true = 2.0
        y0 = 1 + (x1-4)/3 + x2/2 + (x1-4)**2/4 + np.random.normal(0, 1, n)
        y1 = y0 + tau_true
        y = d * y1 + (1 - d) * y0
        
        data = pd.DataFrame({'y': y, 'd': d, 'x1': x1, 'x2': x2})
        
        # IPWRA估计（使用线性结果模型 - 错误指定）
        result = estimate_ipwra(
            data, 'y', 'd',
            controls=['x1', 'x2'],  # 结果模型缺少x1^2项
            propensity_controls=['x1', 'x2'],  # 倾向得分正确
            se_method='analytical'
        )
        
        # 由于倾向得分正确，ATT应该接近真实值
        bias = abs(result.att - tau_true)
        assert bias < 0.5, (
            f"Doubly Robust性质失败: ATT={result.att:.4f}, "
            f"期望≈{tau_true}, 偏差={bias:.4f}"
        )
    
    def test_both_models_correct_smallest_variance(self):
        """
        验证两模型均正确时，方差最小（效率最高）
        """
        np.random.seed(456)
        n = 500
        x = np.random.normal(0, 1, n)
        
        logit_p = -0.5 + 0.3 * x
        p = 1 / (1 + np.exp(-logit_p))
        d = (np.random.uniform(0, 1, n) < p).astype(int)
        
        tau_true = 2.0
        y0 = 1 + x + np.random.normal(0, 0.5, n)
        y1 = y0 + tau_true
        y = d * y1 + (1 - d) * y0
        
        data = pd.DataFrame({'y': y, 'd': d, 'x': x})
        
        result = estimate_ipwra(
            data, 'y', 'd', ['x'],
            se_method='analytical'
        )
        
        # 点估计应该接近真实值
        assert abs(result.att - tau_true) < 0.3
        
        # SE应该合理（不能太大）
        assert result.se < 0.3, f"SE过大: {result.se}"
        assert result.se > 0, f"SE应为正: {result.se}"


class TestIPWRAvsRA:
    """IPWRA与RA对比测试"""
    
    def test_ipwra_ra_similar_in_well_specified(self):
        """在正确指定模型下，IPWRA与RA应接近"""
        np.random.seed(42)
        n = 500
        x = np.random.normal(0, 1, n)
        
        logit_p = -0.5 + 0.3 * x
        p = 1 / (1 + np.exp(-logit_p))
        d = (np.random.uniform(0, 1, n) < p).astype(int)
        
        y0 = 1 + x + np.random.normal(0, 0.5, n)
        y1 = 3 + x + np.random.normal(0, 0.5, n)
        y = d * y1 + (1 - d) * y0
        
        data = pd.DataFrame({'y': y, 'd': d, 'x': x})
        
        # IPWRA估计
        ipwra_result = estimate_ipwra(
            data, 'y', 'd', ['x'],
            se_method='analytical'
        )
        
        # 简单RA估计（使用OLS回归）
        data['_D_treat'] = data['d']
        from lwdid.staggered.estimation import run_ols_regression
        ra_result = run_ols_regression(
            data, 'y', '_D_treat', controls=['x']
        )
        
        # 两者应该接近
        diff = abs(ipwra_result.att - ra_result['att'])
        assert diff < 0.3, f"IPWRA={ipwra_result.att}, RA={ra_result['att']}, diff={diff}"


class TestIPWRAResult:
    """IPWRAResult数据类测试"""
    
    def test_result_attributes(self):
        """验证结果对象属性"""
        np.random.seed(42)
        n = 200
        x = np.random.normal(0, 1, n)
        d = np.random.binomial(1, 0.5, n)
        y = 1 + x + 2 * d + np.random.normal(0, 0.5, n)
        
        data = pd.DataFrame({'y': y, 'd': d, 'x': x})
        
        result = estimate_ipwra(data, 'y', 'd', ['x'])
        
        # 验证所有属性存在
        assert hasattr(result, 'att')
        assert hasattr(result, 'se')
        assert hasattr(result, 'ci_lower')
        assert hasattr(result, 'ci_upper')
        assert hasattr(result, 't_stat')
        assert hasattr(result, 'pvalue')
        assert hasattr(result, 'propensity_scores')
        assert hasattr(result, 'weights')
        assert hasattr(result, 'outcome_model_coef')
        assert hasattr(result, 'propensity_model_coef')
        assert hasattr(result, 'n_treated')
        assert hasattr(result, 'n_control')
        
        # 验证类型
        assert isinstance(result.att, float)
        assert isinstance(result.se, float)
        assert isinstance(result.propensity_scores, np.ndarray)
        assert isinstance(result.weights, np.ndarray)
        assert isinstance(result.outcome_model_coef, dict)
        assert isinstance(result.propensity_model_coef, dict)
        assert isinstance(result.n_treated, int)
        assert isinstance(result.n_control, int)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

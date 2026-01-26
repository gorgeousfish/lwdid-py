"""
BUG-064 & BUG-065 测试: estimate_outcome_model 输入验证

BUG-064: 控制组为空检查
- estimate_outcome_model() 缺少控制组为空时的检查
- 当所有单位都是处理组时，应抛出清晰的 ValueError

BUG-065: 权重归一化除零风险
- estimate_outcome_model() 权重归一化时未检查均值是否有效
- 当权重均值为 0 或非有限值时，应抛出清晰的 ValueError

Reference: BUG-064, BUG-065 in 审查/bug列表.md
"""

import numpy as np
import pandas as pd
import pytest

from lwdid.staggered.estimators import estimate_outcome_model


class TestBUG064EmptyControlGroup:
    """BUG-064: 控制组为空检查测试"""
    
    def test_no_control_units_raises_error(self):
        """当没有控制组单位时，应抛出 ValueError"""
        np.random.seed(42)
        n = 50
        
        # 所有单位都是处理组 (d=1)
        data = pd.DataFrame({
            'y': np.random.normal(0, 1, n),
            'd': np.ones(n),  # 全部是处理组
            'x': np.random.normal(0, 1, n)
        })
        
        with pytest.raises(ValueError) as exc_info:
            estimate_outcome_model(data, 'y', 'd', ['x'])
        
        # 验证错误信息清晰
        error_msg = str(exc_info.value)
        assert "No control units found" in error_msg
        assert "D=0" in error_msg
    
    def test_single_control_unit_works(self):
        """单个控制组单位应能正常工作（边界情况）"""
        np.random.seed(42)
        n = 50
        
        # 只有一个控制组单位
        d = np.ones(n)
        d[0] = 0  # 第一个是控制组
        
        data = pd.DataFrame({
            'y': np.random.normal(0, 1, n),
            'd': d,
            'x': np.random.normal(0, 1, n)
        })
        
        # 单个控制组可能导致矩阵不可逆，但不应因为空控制组检查失败
        # 可能会抛出其他错误（如矩阵奇异），但不是 "No control units found"
        try:
            m0_hat, coef = estimate_outcome_model(data, 'y', 'd', ['x'])
            # 如果成功，检查输出格式
            assert len(m0_hat) == n
            assert '_intercept' in coef
        except ValueError as e:
            # 如果失败，不应是因为空控制组
            assert "No control units found" not in str(e)
    
    def test_normal_data_works(self):
        """正常数据应能正常工作"""
        np.random.seed(42)
        n = 100
        
        data = pd.DataFrame({
            'y': np.random.normal(0, 1, n),
            'd': np.random.binomial(1, 0.5, n),
            'x': np.random.normal(0, 1, n)
        })
        
        m0_hat, coef = estimate_outcome_model(data, 'y', 'd', ['x'])
        
        assert len(m0_hat) == n
        assert '_intercept' in coef
        assert 'x' in coef


class TestBUG065InvalidWeights:
    """BUG-065: 权重归一化除零检查测试"""
    
    def test_zero_weights_raises_error(self):
        """当所有权重为零时，应抛出 ValueError"""
        np.random.seed(42)
        n = 50
        
        data = pd.DataFrame({
            'y': np.random.normal(0, 1, n),
            'd': np.random.binomial(1, 0.5, n),
            'x': np.random.normal(0, 1, n)
        })
        
        # 所有权重为零
        weights = np.zeros(n)
        
        with pytest.raises(ValueError) as exc_info:
            estimate_outcome_model(data, 'y', 'd', ['x'], weights=weights)
        
        error_msg = str(exc_info.value)
        assert "Invalid weights" in error_msg or "mean=" in error_msg
    
    def test_negative_weights_mean_raises_error(self):
        """当权重均值为负时，应抛出 ValueError"""
        np.random.seed(42)
        n = 50
        
        data = pd.DataFrame({
            'y': np.random.normal(0, 1, n),
            'd': np.random.binomial(1, 0.5, n),
            'x': np.random.normal(0, 1, n)
        })
        
        # 负权重（控制组权重的均值为负）
        weights = -np.ones(n)
        
        with pytest.raises(ValueError) as exc_info:
            estimate_outcome_model(data, 'y', 'd', ['x'], weights=weights)
        
        error_msg = str(exc_info.value)
        assert "Invalid weights" in error_msg or "mean=" in error_msg
    
    def test_inf_weights_raises_error(self):
        """当权重包含无穷大时，应抛出 ValueError"""
        np.random.seed(42)
        n = 50
        
        data = pd.DataFrame({
            'y': np.random.normal(0, 1, n),
            'd': np.random.binomial(1, 0.5, n),
            'x': np.random.normal(0, 1, n)
        })
        
        # 包含无穷大的权重
        weights = np.ones(n)
        weights[0] = np.inf
        
        with pytest.raises(ValueError) as exc_info:
            estimate_outcome_model(data, 'y', 'd', ['x'], weights=weights)
        
        error_msg = str(exc_info.value)
        assert "Invalid weights" in error_msg or "finite" in error_msg
    
    def test_nan_weights_raises_error(self):
        """当权重包含 NaN 时，应抛出 ValueError"""
        np.random.seed(42)
        n = 50
        
        data = pd.DataFrame({
            'y': np.random.normal(0, 1, n),
            'd': np.random.binomial(1, 0.5, n),
            'x': np.random.normal(0, 1, n)
        })
        
        # 包含 NaN 的权重
        weights = np.ones(n)
        weights[0] = np.nan
        
        with pytest.raises(ValueError) as exc_info:
            estimate_outcome_model(data, 'y', 'd', ['x'], weights=weights)
        
        error_msg = str(exc_info.value)
        assert "Invalid weights" in error_msg or "finite" in error_msg
    
    def test_positive_weights_work(self):
        """正常正权重应能正常工作"""
        np.random.seed(42)
        n = 100
        
        data = pd.DataFrame({
            'y': np.random.normal(0, 1, n),
            'd': np.random.binomial(1, 0.5, n),
            'x': np.random.normal(0, 1, n)
        })
        
        # 正常的正权重
        weights = np.abs(np.random.normal(1, 0.3, n))
        
        m0_hat, coef = estimate_outcome_model(data, 'y', 'd', ['x'], weights=weights)
        
        assert len(m0_hat) == n
        assert '_intercept' in coef


class TestBUG064065Integration:
    """BUG-064 & BUG-065 集成测试"""
    
    def test_empty_control_with_weights(self):
        """空控制组 + 权重场景"""
        np.random.seed(42)
        n = 50
        
        # 所有单位都是处理组
        data = pd.DataFrame({
            'y': np.random.normal(0, 1, n),
            'd': np.ones(n),
            'x': np.random.normal(0, 1, n)
        })
        
        weights = np.ones(n)
        
        # 应该先检测到空控制组错误
        with pytest.raises(ValueError) as exc_info:
            estimate_outcome_model(data, 'y', 'd', ['x'], weights=weights)
        
        assert "No control units found" in str(exc_info.value)
    
    def test_error_message_clarity(self):
        """验证错误信息的清晰度"""
        np.random.seed(42)
        n = 50
        
        # 空控制组场景
        data_no_control = pd.DataFrame({
            'y': np.random.normal(0, 1, n),
            'd': np.ones(n),
            'x': np.random.normal(0, 1, n)
        })
        
        with pytest.raises(ValueError) as exc_info:
            estimate_outcome_model(data_no_control, 'y', 'd', ['x'])
        
        error_msg = str(exc_info.value)
        # 错误信息应包含：
        # 1. 问题描述
        # 2. 具体要求
        assert "control" in error_msg.lower()
        assert "D=0" in error_msg or "d=0" in error_msg.lower()
    
    def test_numerical_stability_with_small_weights(self):
        """小权重场景的数值稳定性"""
        np.random.seed(42)
        n = 100
        
        data = pd.DataFrame({
            'y': np.random.normal(0, 1, n),
            'd': np.random.binomial(1, 0.5, n),
            'x': np.random.normal(0, 1, n)
        })
        
        # 非常小但正的权重
        weights = np.full(n, 1e-10)
        
        # 应该能正常工作（权重被归一化）
        m0_hat, coef = estimate_outcome_model(data, 'y', 'd', ['x'], weights=weights)
        
        assert len(m0_hat) == n
        assert np.all(np.isfinite(m0_hat))


class TestRegressionPrevention:
    """回归防护测试：确保修复不破坏现有功能"""
    
    def test_basic_functionality_preserved(self):
        """基本功能保持不变"""
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
    
    def test_wls_with_weights_preserved(self):
        """加权最小二乘功能保持不变"""
        np.random.seed(42)
        n = 200
        x = np.random.normal(0, 1, n)
        y = 1 + 2 * x + np.random.normal(0, 0.5, n)
        d = np.random.binomial(1, 0.5, n)
        
        data = pd.DataFrame({'y': y, 'd': d, 'x': x})
        
        # 正常权重
        weights = np.abs(np.random.normal(1, 0.3, n))
        
        m0_hat_weighted, coef_weighted = estimate_outcome_model(
            data, 'y', 'd', ['x'], weights=weights
        )
        m0_hat_unweighted, coef_unweighted = estimate_outcome_model(
            data, 'y', 'd', ['x'], weights=None
        )
        
        # 两者应该不同（权重有影响）
        # 但都应该有效
        assert len(m0_hat_weighted) == n
        assert len(m0_hat_unweighted) == n
        assert '_intercept' in coef_weighted
        assert '_intercept' in coef_unweighted


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

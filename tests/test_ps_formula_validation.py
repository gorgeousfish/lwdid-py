"""
倾向得分诊断公式验证测试

验证诊断计算中的数学公式正确性：
1. 权重公式: w = p / (1 - p)
2. CV 公式: CV = σ(w) / μ(w)
3. 裁剪逻辑: p ∈ [trim, 1-trim]
4. 分位数计算

Reference: Story 1.1 - 倾向得分诊断增强
"""

import numpy as np
import pytest

from lwdid.staggered.estimators import _compute_ps_diagnostics


class TestWeightFormula:
    """测试权重公式: w = p / (1 - p)"""
    
    def test_weight_formula_basic(self):
        """基础权重公式验证"""
        # 测试向量
        p_values = np.array([0.1, 0.2, 0.3, 0.5, 0.7])
        
        # Python 计算
        weights = p_values / (1 - p_values)
        
        # 预期结果（手动计算验证）
        expected = np.array([
            0.1 / 0.9,    # 0.1111...
            0.2 / 0.8,    # 0.25
            0.3 / 0.7,    # 0.4286...
            0.5 / 0.5,    # 1.0
            0.7 / 0.3,    # 2.3333...
        ])
        
        # 验证
        assert np.allclose(weights, expected)
    
    def test_weight_formula_boundary(self):
        """边界情况权重公式验证"""
        # p = 0.01 (裁剪下界)
        p_low = 0.01
        w_low = p_low / (1 - p_low)
        assert abs(w_low - 0.010101) < 1e-4
        
        # p = 0.99 (裁剪上界)
        p_high = 0.99
        w_high = p_high / (1 - p_high)
        assert abs(w_high - 99.0) < 1e-4
        
        # p = 0.5 (对称点)
        p_mid = 0.5
        w_mid = p_mid / (1 - p_mid)
        assert w_mid == 1.0
    
    def test_weight_formula_monotonicity(self):
        """权重公式单调性验证"""
        # 权重应该随 p 单调递增
        p_values = np.linspace(0.01, 0.99, 100)
        weights = p_values / (1 - p_values)
        
        # 验证单调递增
        diffs = np.diff(weights)
        assert np.all(diffs > 0), "权重应该随倾向得分单调递增"


class TestCVFormula:
    """测试 CV 公式: CV = σ(w) / μ(w)"""
    
    def test_cv_formula_basic(self):
        """基础 CV 公式验证"""
        # 已知权重向量
        weights = np.array([0.5, 1.0, 1.5, 2.0])
        
        # 手动计算
        mean_w = np.mean(weights)  # 1.25
        std_w = np.std(weights, ddof=1)  # sqrt(var)
        cv_manual = std_w / mean_w
        
        # 验证均值
        assert abs(mean_w - 1.25) < 1e-10
        
        # 验证标准差 (无偏估计)
        # var = [(0.5-1.25)² + (1.0-1.25)² + (1.5-1.25)² + (2.0-1.25)²] / (4-1)
        # var = [0.5625 + 0.0625 + 0.0625 + 0.5625] / 3 = 1.25 / 3 = 0.4167
        expected_var = (0.5625 + 0.0625 + 0.0625 + 0.5625) / 3
        expected_std = np.sqrt(expected_var)
        assert abs(std_w - expected_std) < 1e-10
        
        # 验证 CV
        expected_cv = expected_std / 1.25
        assert abs(cv_manual - expected_cv) < 1e-10
    
    def test_cv_uniform_weights(self):
        """所有权重相同时 CV = 0"""
        weights_uniform = np.array([1.0, 1.0, 1.0])
        
        mean_w = np.mean(weights_uniform)
        std_w = np.std(weights_uniform, ddof=1)
        cv = std_w / mean_w
        
        assert cv == 0.0
    
    def test_cv_high_variability(self):
        """高变异权重的 CV"""
        # 构造高变异权重
        weights_high = np.array([0.1, 1.0, 10.0])
        
        mean_w = np.mean(weights_high)
        std_w = np.std(weights_high, ddof=1)
        cv = std_w / mean_w
        
        # 高变异情况 CV 应该大于 1
        assert cv > 1.0
    
    def test_cv_in_diagnostics(self):
        """验证诊断函数中的 CV 计算"""
        # 构造已知控制组倾向得分
        pscores_control = np.array([0.2, 0.3, 0.4])  # 控制组
        pscores_treat = np.array([0.5, 0.6])  # 处理组
        pscores = np.concatenate([pscores_treat, pscores_control])
        D = np.array([1, 1, 0, 0, 0], dtype=float)
        
        # 计算诊断
        diag = _compute_ps_diagnostics(pscores, pscores, D, 0.01)
        
        # 手动计算控制组权重 CV
        weights_control = pscores_control / (1 - pscores_control)
        expected_cv = np.std(weights_control, ddof=1) / np.mean(weights_control)
        
        # 验证
        assert abs(diag.weights_cv - expected_cv) < 1e-10


class TestTrimmingLogic:
    """测试裁剪逻辑: p ∈ [trim, 1-trim]"""
    
    def test_trimming_basic(self):
        """基础裁剪逻辑验证"""
        # 原始倾向得分（含极端值）
        pscores_raw = np.array([0.001, 0.05, 0.5, 0.95, 0.999])
        trim = 0.01
        
        # 裁剪
        pscores_trimmed = np.clip(pscores_raw, trim, 1 - trim)
        
        # 验证
        expected = np.array([0.01, 0.05, 0.5, 0.95, 0.99])
        assert np.allclose(pscores_trimmed, expected)
    
    def test_trimming_boundaries(self):
        """裁剪边界验证"""
        pscores_raw = np.array([0.0, 0.005, 0.01, 0.99, 0.995, 1.0])
        trim = 0.01
        
        pscores_trimmed = np.clip(pscores_raw, trim, 1 - trim)
        
        # 验证范围
        assert np.all(pscores_trimmed >= trim)
        assert np.all(pscores_trimmed <= 1 - trim)
    
    def test_extreme_detection_counts(self):
        """极端值检测计数验证"""
        # 原始倾向得分
        pscores_raw = np.array([
            0.001, 0.005,  # 2个低端极端值 (< 0.01)
            0.1, 0.5, 0.9,  # 3个正常值
            0.995, 0.999,   # 2个高端极端值 (> 0.99)
        ])
        trim = 0.01
        
        # 计算极端值
        n_low = np.sum(pscores_raw < trim)  # 2
        n_high = np.sum(pscores_raw > 1 - trim)  # 2
        
        assert n_low == 2
        assert n_high == 2
        assert n_low + n_high == 4
    
    def test_extreme_detection_in_diagnostics(self):
        """诊断函数中的极端值检测"""
        # 原始倾向得分
        pscores_raw = np.array([
            0.001, 0.005,  # 低端极端
            0.1, 0.5, 0.9,  # 正常
            0.995, 0.999,   # 高端极端
        ])
        D = np.array([1, 1, 1, 0, 0, 0, 0], dtype=float)
        pscores_trimmed = np.clip(pscores_raw, 0.01, 0.99)
        
        diag = _compute_ps_diagnostics(pscores_raw, pscores_trimmed, D, 0.01)
        
        # 验证
        assert diag.n_trimmed == 4
        assert diag.extreme_low_pct == 2/7
        assert diag.extreme_high_pct == 2/7


class TestPercentileCalculation:
    """测试分位数计算"""
    
    def test_percentile_basic(self):
        """基础分位数计算验证"""
        # 已知数据
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        
        # NumPy 分位数
        q25, q50, q75 = np.percentile(data, [25, 50, 75])
        
        # 验证 (NumPy 默认使用线性插值)
        assert q25 == 3.25  # 25% 分位数
        assert q50 == 5.5   # 中位数
        assert q75 == 7.75  # 75% 分位数
    
    def test_percentile_monotonicity(self):
        """分位数单调性验证"""
        np.random.seed(42)
        data = np.random.uniform(0, 1, 100)
        
        q25, q50, q75 = np.percentile(data, [25, 50, 75])
        
        # 分位数应该单调递增
        assert q25 <= q50 <= q75
    
    def test_percentile_in_diagnostics(self):
        """诊断函数中的分位数计算"""
        pscores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        D = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=float)
        
        diag = _compute_ps_diagnostics(pscores, pscores, D, 0.01)
        
        # 验证分位数结构
        assert '25%' in diag.ps_quantiles
        assert '50%' in diag.ps_quantiles
        assert '75%' in diag.ps_quantiles
        
        # 验证单调性
        assert diag.ps_quantiles['25%'] <= diag.ps_quantiles['50%']
        assert diag.ps_quantiles['50%'] <= diag.ps_quantiles['75%']
        
        # 验证数值
        expected_q25, expected_q50, expected_q75 = np.percentile(pscores, [25, 50, 75])
        assert abs(diag.ps_quantiles['25%'] - expected_q25) < 1e-10
        assert abs(diag.ps_quantiles['50%'] - expected_q50) < 1e-10
        assert abs(diag.ps_quantiles['75%'] - expected_q75) < 1e-10


class TestMathematicalProperties:
    """测试数学性质"""
    
    def test_weight_inverse_relationship(self):
        """权重与 (1-p) 的反比关系"""
        # w = p/(1-p) => w * (1-p) = p => w - wp = p => w = p(1+w) => p = w/(1+w)
        p = 0.3
        w = p / (1 - p)
        
        # 反推 p
        p_recovered = w / (1 + w)
        assert abs(p - p_recovered) < 1e-10
    
    def test_cv_scale_invariance(self):
        """CV 的尺度不变性"""
        # CV 应该对乘法常数保持不变
        weights = np.array([1.0, 2.0, 3.0])
        
        cv1 = np.std(weights, ddof=1) / np.mean(weights)
        cv2 = np.std(weights * 10, ddof=1) / np.mean(weights * 10)
        cv3 = np.std(weights * 0.1, ddof=1) / np.mean(weights * 0.1)
        
        assert abs(cv1 - cv2) < 1e-10
        assert abs(cv1 - cv3) < 1e-10
    
    def test_extreme_detection_threshold_sensitivity(self):
        """极端值检测对阈值的敏感性"""
        pscores_raw = np.array([0.005, 0.02, 0.5, 0.98, 0.995])
        D = np.ones(5, dtype=float)
        D[2:] = 0
        
        # 阈值 = 0.01
        pscores_trim_01 = np.clip(pscores_raw, 0.01, 0.99)
        diag_01 = _compute_ps_diagnostics(pscores_raw, pscores_trim_01, D, 0.01)
        
        # 阈值 = 0.05
        pscores_trim_05 = np.clip(pscores_raw, 0.05, 0.95)
        diag_05 = _compute_ps_diagnostics(pscores_raw, pscores_trim_05, D, 0.05)
        
        # 更大的阈值应该检测到更多极端值
        assert diag_05.n_trimmed >= diag_01.n_trimmed


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

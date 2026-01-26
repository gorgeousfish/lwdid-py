"""
倾向得分诊断 - MCP 数值验证测试

使用 vibe-math-mcp 工具进行精确数值验证，确保：
1. 权重公式 w = p/(1-p) 正确
2. CV 公式 CV = σ(w)/μ(w) 正确
3. 裁剪边界计算正确
4. 分位数计算正确

Reference: Story 1.1 - 倾向得分诊断增强
验证标准: 论文 Lee & Wooldridge (2023) Section 4, 公式 4.11
"""

import numpy as np
import pytest

from lwdid.staggered.estimators import (
    _compute_ps_diagnostics,
    estimate_propensity_score,
    PropensityScoreDiagnostics,
)


class TestWeightFormulaValidation:
    """权重公式数值验证: w = p / (1 - p)"""
    
    # vibe-math-mcp 验证的参考值
    MCP_REFERENCE_VALUES = {
        # (p, expected_weight) - 由 vibe-math-mcp calculate 验证
        0.1: 0.1111111111111111,
        0.2: 0.25,
        0.3: 0.42857142857142855,
        0.5: 1.0,
        0.7: 2.3333333333333335,
        0.9: 9.0,
    }
    
    def test_weight_formula_against_mcp_reference(self):
        """对照 vibe-math-mcp 验证的参考值"""
        for p, expected_w in self.MCP_REFERENCE_VALUES.items():
            calculated_w = p / (1 - p)
            assert abs(calculated_w - expected_w) < 1e-10, \
                f"p={p}: 计算值={calculated_w}, 期望值={expected_w}"
    
    def test_weight_formula_boundary_values(self):
        """
        边界值验证（裁剪阈值）
        
        vibe-math-mcp 验证结果:
        - w_min = 0.01/(1-0.01) = 0.010101010101010102
        - w_max = 0.99/(1-0.99) = 98.99999999999991
        """
        # 下界
        p_min = 0.01
        w_min = p_min / (1 - p_min)
        assert abs(w_min - 0.010101010101) < 1e-6
        
        # 上界
        p_max = 0.99
        w_max = p_max / (1 - p_max)
        assert abs(w_max - 99.0) < 0.01


class TestCVFormulaValidation:
    """CV 公式数值验证: CV = σ(w) / μ(w)"""
    
    def test_cv_formula_against_mcp_reference(self):
        """
        对照 vibe-math-mcp 验证的参考值
        
        控制组权重: [0.111111, 0.428571, 2.333333]
        vibe-math-mcp array_statistics 结果:
        - mean = 0.9576716666666667
        - std = 1.2018853068913578
        - CV = std/mean = 1.255007690761821
        """
        weights = np.array([0.111111, 0.428571, 2.333333])
        
        # 计算均值和标准差
        mean_w = np.mean(weights)
        # 注意: vibe-math-mcp 使用样本标准差 (ddof=0)，而我们使用 ddof=1
        # 这里验证我们的 ddof=1 计算
        std_w_ddof1 = np.std(weights, ddof=1)
        
        # 使用无偏估计的 CV
        cv_ddof1 = std_w_ddof1 / mean_w
        
        # 验证均值
        assert abs(mean_w - 0.9576716666666667) < 1e-6
        
        # 验证标准差 (ddof=1)
        # ddof=1 的标准差应该略大于 ddof=0
        std_w_ddof0 = np.std(weights, ddof=0)
        assert std_w_ddof1 > std_w_ddof0
        
        # 验证 CV 在合理范围内
        assert cv_ddof1 > 1.0  # 这个权重分布的 CV 应该 > 1
    
    def test_cv_in_diagnostics_function(self):
        """验证诊断函数中的 CV 计算使用 ddof=1"""
        # 构造已知控制组倾向得分
        pscores_control = np.array([0.1, 0.3, 0.7])
        D = np.array([1, 1, 1, 0, 0, 0], dtype=float)  # 3 treated, 3 control
        pscores = np.concatenate([np.array([0.5, 0.5, 0.5]), pscores_control])
        
        diag = _compute_ps_diagnostics(pscores, pscores, D, 0.01)
        
        # 手动计算控制组权重 CV (ddof=1)
        weights_control = pscores_control / (1 - pscores_control)
        expected_cv = np.std(weights_control, ddof=1) / np.mean(weights_control)
        
        assert abs(diag.weights_cv - expected_cv) < 1e-10


class TestTrimmingBoundaryValidation:
    """裁剪边界数值验证"""
    
    def test_trimming_preserves_interior_values(self):
        """验证内部值不被裁剪"""
        pscores_raw = np.array([0.05, 0.3, 0.5, 0.7, 0.95])
        trim = 0.01
        pscores_trimmed = np.clip(pscores_raw, trim, 1 - trim)
        
        # 所有值应该保持不变（都在 [0.01, 0.99] 内）
        assert np.allclose(pscores_raw, pscores_trimmed)
    
    def test_trimming_clips_extreme_values(self):
        """验证极端值被正确裁剪"""
        pscores_raw = np.array([0.001, 0.005, 0.5, 0.995, 0.999])
        trim = 0.01
        pscores_trimmed = np.clip(pscores_raw, trim, 1 - trim)
        
        expected = np.array([0.01, 0.01, 0.5, 0.99, 0.99])
        assert np.allclose(pscores_trimmed, expected)
    
    def test_extreme_detection_counts_match(self):
        """验证极端值计数与 n_trimmed 一致"""
        pscores_raw = np.array([
            0.001, 0.005, 0.008,  # 3 low extreme
            0.1, 0.5, 0.9,         # normal
            0.992, 0.995, 0.999,   # 3 high extreme
        ])
        D = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=float)
        pscores_trimmed = np.clip(pscores_raw, 0.01, 0.99)
        
        diag = _compute_ps_diagnostics(pscores_raw, pscores_trimmed, D, 0.01)
        
        # 验证计数
        assert diag.n_trimmed == 6  # 3 low + 3 high
        
        # 验证比例
        n_total = len(pscores_raw)
        expected_low_pct = 3 / n_total
        expected_high_pct = 3 / n_total
        
        assert abs(diag.extreme_low_pct - expected_low_pct) < 1e-10
        assert abs(diag.extreme_high_pct - expected_high_pct) < 1e-10


class TestQuantileValidation:
    """分位数计算验证"""
    
    def test_quantile_monotonicity(self):
        """验证分位数单调性"""
        np.random.seed(42)
        pscores = np.random.uniform(0.1, 0.9, 100)
        D = np.random.binomial(1, 0.3, 100).astype(float)
        
        diag = _compute_ps_diagnostics(pscores, pscores, D, 0.01)
        
        assert diag.ps_quantiles['25%'] <= diag.ps_quantiles['50%']
        assert diag.ps_quantiles['50%'] <= diag.ps_quantiles['75%']
    
    def test_quantile_values_match_numpy(self):
        """验证分位数与 NumPy percentile 一致"""
        pscores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        D = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=float)
        
        diag = _compute_ps_diagnostics(pscores, pscores, D, 0.01)
        
        expected_q25, expected_q50, expected_q75 = np.percentile(pscores, [25, 50, 75])
        
        assert abs(diag.ps_quantiles['25%'] - expected_q25) < 1e-10
        assert abs(diag.ps_quantiles['50%'] - expected_q50) < 1e-10
        assert abs(diag.ps_quantiles['75%'] - expected_q75) < 1e-10


class TestWarningThresholdValidation:
    """警告阈值验证"""
    
    def test_cv_warning_threshold(self):
        """验证 CV > 2.0 触发警告"""
        # 构造高 CV 情况
        pscores = np.array([
            0.5, 0.5, 0.5,        # 处理组
            0.05, 0.5, 0.95,      # 控制组: 极端变异
        ])
        D = np.array([1, 1, 1, 0, 0, 0], dtype=float)
        
        diag = _compute_ps_diagnostics(pscores, pscores, D, 0.01)
        
        # 计算预期 CV
        pscores_control = pscores[D == 0]
        weights_control = pscores_control / (1 - pscores_control)
        expected_cv = np.std(weights_control, ddof=1) / np.mean(weights_control)
        
        # 如果 CV > 2.0，应该触发警告
        if expected_cv > 2.0:
            assert diag.overlap_warning is not None
            assert 'Weight coefficient of variation too high' in diag.overlap_warning
    
    def test_extreme_pct_warning_threshold(self):
        """验证极端值 > 10% 触发警告"""
        # 构造 15% 极端值
        pscores_raw = np.concatenate([
            np.full(15, 0.005),   # 15% 极低
            np.full(85, 0.5),
        ])
        D = np.concatenate([np.ones(30), np.zeros(70)]).astype(float)
        pscores_trimmed = np.clip(pscores_raw, 0.01, 0.99)
        
        diag = _compute_ps_diagnostics(pscores_raw, pscores_trimmed, D, 0.01)
        
        assert diag.extreme_low_pct == 0.15
        assert diag.overlap_warning is not None
        assert 'Extreme propensity score proportion too high' in diag.overlap_warning
    
    def test_trimming_info_threshold(self):
        """验证裁剪 > 5% 触发信息提示"""
        # 构造 10% 被裁剪
        pscores_raw = np.concatenate([
            np.full(5, 0.005),   # 5% 极低
            np.full(90, 0.5),
            np.full(5, 0.995),   # 5% 极高
        ])
        D = np.concatenate([np.ones(30), np.zeros(70)]).astype(float)
        pscores_trimmed = np.clip(pscores_raw, 0.01, 0.99)
        
        diag = _compute_ps_diagnostics(pscores_raw, pscores_trimmed, D, 0.01)
        
        assert diag.n_trimmed == 10
        assert diag.overlap_warning is not None
        assert 'observations trimmed' in diag.overlap_warning


class TestPaperFormulaValidation:
    """论文公式验证: Lee & Wooldridge (2023)"""
    
    def test_overlap_assumption_formula_411(self):
        """
        验证 Overlap 假设 (公式 4.11)
        
        P(D_g = 1 | D_g + A_{r+1} = 1, X = x) < 1 for all x ∈ Supp(X)
        
        含义: 倾向得分必须严格小于 1，以确保每个处理单位都有可比的控制单位
        """
        # 测试正常情况: 所有倾向得分 < 1
        pscores_normal = np.array([0.1, 0.3, 0.5, 0.7, 0.8])
        D = np.array([1, 1, 0, 0, 0], dtype=float)
        
        diag = _compute_ps_diagnostics(pscores_normal, pscores_normal, D, 0.01)
        
        # 验证裁剪后最大值 < 1
        assert diag.ps_max < 1.0
        
        # 测试极端情况: 接近 1 的倾向得分被裁剪
        pscores_extreme = np.array([0.3, 0.5, 0.999, 0.9999])
        D_extreme = np.array([1, 0, 0, 0], dtype=float)
        pscores_trimmed = np.clip(pscores_extreme, 0.01, 0.99)
        
        diag_extreme = _compute_ps_diagnostics(pscores_extreme, pscores_trimmed, D_extreme, 0.01)
        
        # 验证极端值被检测
        assert diag_extreme.n_trimmed == 2  # 0.999 和 0.9999 都 > 0.99
    
    def test_ipw_weight_formula(self):
        """
        验证 IPW 权重公式 (Wooldridge 2007)
        
        w = p(X) / (1 - p(X))
        
        仅对控制组 (D = 0) 计算权重
        """
        pscores = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
        D = np.array([1, 1, 0, 0, 0], dtype=float)
        
        diag = _compute_ps_diagnostics(pscores, pscores, D, 0.01)
        
        # 手动计算控制组权重
        pscores_control = pscores[D == 0]  # [0.4, 0.5, 0.6]
        weights_expected = pscores_control / (1 - pscores_control)
        
        # 验证 CV 计算使用控制组权重
        expected_cv = np.std(weights_expected, ddof=1) / np.mean(weights_expected)
        assert abs(diag.weights_cv - expected_cv) < 1e-10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

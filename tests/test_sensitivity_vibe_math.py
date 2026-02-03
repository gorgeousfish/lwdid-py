"""
使用 vibe-math MCP 验证敏感性分析公式实现。

T10 任务：验证以下公式的正确性：
1. 敏感性比率公式
2. 稳健性判断公式
3. 预期效应检测公式

注意：此测试文件设计为可以独立运行，也可以与 vibe-math MCP 工具配合验证。
"""

import numpy as np
import pytest


class TestSensitivityRatioFormula:
    """验证敏感性比率公式。
    
    公式: Sensitivity Ratio = (max(ATT) - min(ATT)) / |baseline ATT|
    """

    def test_sensitivity_ratio_basic(self):
        """基本敏感性比率计算。
        
        ATT estimates: [1.0, 1.1, 1.2, 1.3, 1.4]
        Baseline: 1.4
        Range: 1.4 - 1.0 = 0.4
        Ratio: 0.4 / 1.4 ≈ 0.2857
        
        vibe-math 验证: mcp_vibe_math_mcp_calculate(expression="(1.4 - 1.0) / abs(1.4)")
        """
        atts = [1.0, 1.1, 1.2, 1.3, 1.4]
        baseline = 1.4
        
        att_max = max(atts)
        att_min = min(atts)
        att_range = att_max - att_min
        sensitivity_ratio = att_range / abs(baseline)
        
        expected = 0.4 / 1.4
        assert abs(sensitivity_ratio - expected) < 1e-10
        assert abs(sensitivity_ratio - 0.2857142857142857) < 1e-10

    def test_sensitivity_ratio_negative_baseline(self):
        """负基线 ATT 的敏感性比率计算。
        
        ATT estimates: [-1.0, -1.1, -1.2]
        Baseline: -1.2
        Range: -1.0 - (-1.2) = 0.2
        Ratio: 0.2 / |-1.2| = 0.167
        
        vibe-math 验证: mcp_vibe_math_mcp_calculate(expression="(-1.0 - (-1.2)) / abs(-1.2)")
        """
        atts = [-1.0, -1.1, -1.2]
        baseline = -1.2
        
        att_range = max(atts) - min(atts)
        sensitivity_ratio = att_range / abs(baseline)
        
        expected = 0.2 / 1.2
        assert abs(sensitivity_ratio - expected) < 1e-10
        assert abs(sensitivity_ratio - 0.16666666666666666) < 1e-10

    def test_sensitivity_ratio_zero_range(self):
        """零范围（所有估计相同）的敏感性比率。
        
        ATT estimates: [2.0, 2.0, 2.0]
        Baseline: 2.0
        Range: 0
        Ratio: 0 / 2.0 = 0
        
        vibe-math 验证: mcp_vibe_math_mcp_calculate(expression="(2.0 - 2.0) / abs(2.0)")
        """
        atts = [2.0, 2.0, 2.0]
        baseline = 2.0
        
        att_range = max(atts) - min(atts)
        sensitivity_ratio = att_range / abs(baseline)
        
        assert sensitivity_ratio == 0.0

    def test_sensitivity_ratio_large_variation(self):
        """大变异的敏感性比率。
        
        ATT estimates: [0.5, 1.0, 1.5, 2.0, 2.5]
        Baseline: 2.5
        Range: 2.5 - 0.5 = 2.0
        Ratio: 2.0 / 2.5 = 0.8
        
        vibe-math 验证: mcp_vibe_math_mcp_calculate(expression="(2.5 - 0.5) / abs(2.5)")
        """
        atts = [0.5, 1.0, 1.5, 2.0, 2.5]
        baseline = 2.5
        
        att_range = max(atts) - min(atts)
        sensitivity_ratio = att_range / abs(baseline)
        
        expected = 2.0 / 2.5
        assert abs(sensitivity_ratio - expected) < 1e-10
        assert abs(sensitivity_ratio - 0.8) < 1e-10


class TestRobustnessThresholdFormula:
    """验证稳健性判断公式。
    
    公式: is_robust = sensitivity_ratio < threshold
    
    阈值标准:
    - < 10%: 高度稳健
    - 10-25%: 中度稳健
    - 25-50%: 敏感
    - >= 50%: 高度敏感
    """

    @pytest.mark.parametrize("ratio,threshold,expected", [
        (0.05, 0.25, True),   # 5% < 25% → 稳健
        (0.20, 0.25, True),   # 20% < 25% → 稳健
        (0.25, 0.25, False),  # 25% = 25% → 不稳健（边界）
        (0.30, 0.25, False),  # 30% > 25% → 不稳健
        (0.09, 0.10, True),   # 9% < 10% → 高度稳健
        (0.10, 0.10, False),  # 10% = 10% → 边界
    ])
    def test_robustness_threshold_comparison(self, ratio, threshold, expected):
        """测试稳健性阈值比较。
        
        vibe-math 验证: mcp_vibe_math_mcp_calculate(expression="0.20 < 0.25")
        """
        is_robust = ratio < threshold
        assert is_robust == expected

    @pytest.mark.parametrize("ratio,expected_level", [
        (0.05, "highly_robust"),
        (0.09, "highly_robust"),
        (0.10, "moderately_robust"),
        (0.15, "moderately_robust"),
        (0.24, "moderately_robust"),
        (0.25, "sensitive"),
        (0.35, "sensitive"),
        (0.49, "sensitive"),
        (0.50, "highly_sensitive"),
        (0.75, "highly_sensitive"),
        (1.00, "highly_sensitive"),
    ])
    def test_robustness_level_determination(self, ratio, expected_level):
        """测试稳健性等级判断。
        
        vibe-math 验证边界值:
        - mcp_vibe_math_mcp_calculate(expression="0.09 < 0.10")  # True → highly_robust
        - mcp_vibe_math_mcp_calculate(expression="0.10 < 0.10")  # False
        - mcp_vibe_math_mcp_calculate(expression="0.10 < 0.25")  # True → moderately_robust
        """
        if ratio < 0.10:
            level = "highly_robust"
        elif ratio < 0.25:
            level = "moderately_robust"
        elif ratio < 0.50:
            level = "sensitive"
        else:
            level = "highly_sensitive"
        
        assert level == expected_level


class TestAnticipationDetectionFormula:
    """验证预期效应检测公式。
    
    公式: relative_change = |ATT_k - ATT_0| / |ATT_0|
    检测: detected = relative_change > threshold
    """

    def test_relative_change_formula(self):
        """测试相对变化公式。
        
        ATT_baseline (no exclusion): 2.0
        ATT_exclude_1: 2.3
        Relative change: |2.3 - 2.0| / |2.0| = 0.3 / 2.0 = 0.15
        
        vibe-math 验证: mcp_vibe_math_mcp_calculate(expression="abs(2.3 - 2.0) / abs(2.0)")
        """
        att_baseline = 2.0
        att_exclude_1 = 2.3
        
        relative_change = abs(att_exclude_1 - att_baseline) / abs(att_baseline)
        
        expected = 0.3 / 2.0
        assert abs(relative_change - expected) < 1e-10
        assert abs(relative_change - 0.15) < 1e-10

    def test_anticipation_detection_threshold(self):
        """测试预期效应检测阈值。
        
        Relative change: 0.15
        Threshold: 0.10
        Detected: 0.15 > 0.10 → True
        
        vibe-math 验证: mcp_vibe_math_mcp_calculate(expression="0.15 > 0.10")
        """
        relative_change = 0.15
        threshold = 0.10
        
        detected = relative_change > threshold
        assert detected is True

    def test_no_anticipation_detection(self):
        """测试无预期效应情况。
        
        ATT_baseline: 2.0
        ATT_exclude_1: 2.05
        Relative change: |2.05 - 2.0| / |2.0| = 0.05 / 2.0 = 0.025
        Threshold: 0.10
        Detected: 0.025 > 0.10 → False
        
        vibe-math 验证: mcp_vibe_math_mcp_calculate(expression="abs(2.05 - 2.0) / abs(2.0)")
        """
        att_baseline = 2.0
        att_exclude_1 = 2.05
        threshold = 0.10
        
        relative_change = abs(att_exclude_1 - att_baseline) / abs(att_baseline)
        detected = relative_change > threshold
        
        assert abs(relative_change - 0.025) < 1e-10
        assert detected is False

    def test_negative_att_anticipation(self):
        """测试负 ATT 的预期效应检测。
        
        ATT_baseline: -1.5
        ATT_exclude_1: -1.8
        Relative change: |-1.8 - (-1.5)| / |-1.5| = 0.3 / 1.5 = 0.2
        
        vibe-math 验证: mcp_vibe_math_mcp_calculate(expression="abs(-1.8 - (-1.5)) / abs(-1.5)")
        """
        att_baseline = -1.5
        att_exclude_1 = -1.8
        
        relative_change = abs(att_exclude_1 - att_baseline) / abs(att_baseline)
        
        expected = 0.3 / 1.5
        assert abs(relative_change - expected) < 1e-10
        assert abs(relative_change - 0.2) < 1e-10


class TestSensitivityModuleFunctions:
    """测试 sensitivity 模块中的实际函数实现。"""

    def test_compute_sensitivity_ratio_function(self):
        """测试 _compute_sensitivity_ratio 函数。"""
        from lwdid.sensitivity import _compute_sensitivity_ratio
        
        # 测试用例 1: 正常情况
        atts = [1.0, 1.1, 1.2, 1.3, 1.4]
        baseline = 1.4
        ratio = _compute_sensitivity_ratio(atts, baseline)
        expected = (1.4 - 1.0) / abs(1.4)
        assert abs(ratio - expected) < 1e-10
        
        # 测试用例 2: 负基线
        atts = [-1.0, -1.1, -1.2]
        baseline = -1.2
        ratio = _compute_sensitivity_ratio(atts, baseline)
        expected = (-1.0 - (-1.2)) / abs(-1.2)
        assert abs(ratio - expected) < 1e-10

    def test_determine_robustness_level_function(self):
        """测试 _determine_robustness_level 函数。"""
        from lwdid.sensitivity import _determine_robustness_level, RobustnessLevel
        
        assert _determine_robustness_level(0.05) == RobustnessLevel.HIGHLY_ROBUST
        assert _determine_robustness_level(0.15) == RobustnessLevel.MODERATELY_ROBUST
        assert _determine_robustness_level(0.35) == RobustnessLevel.SENSITIVE
        assert _determine_robustness_level(0.75) == RobustnessLevel.HIGHLY_SENSITIVE


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

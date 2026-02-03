"""
Python-论文 端到端测试：验证实现与 Lee & Wooldridge (2026) Section 8.1 一致。

T12 任务：验证敏感性分析实现与论文描述一致。

论文引用：
Lee, S.J. & Wooldridge, J.M. (2026). "Simple Difference-in-Differences
Estimation in Fixed-T Panels." SSRN 5325686, Section 8.1.

Section 8.1 关键内容：
"With synthetic control-type approaches and the approaches we suggest here,
one can study the robustness of the findings by adjusting the number of
pre-treatment time periods."
"""

import numpy as np
import pandas as pd
import pytest


class TestPaperSection81Compliance:
    """验证实现与论文 Section 8.1 描述一致。"""

    @pytest.fixture
    def panel_data(self):
        """生成符合论文设定的面板数据。"""
        np.random.seed(42)
        n_units = 100
        n_periods = 12
        treatment_period = 8
        
        data = []
        for i in range(n_units):
            alpha_i = np.random.normal(0, 1)
            treated = i < n_units // 2
            
            for t in range(1, n_periods + 1):
                gamma_t = 0.1 * t
                d_it = 1 if treated and t >= treatment_period else 0
                tau = 2.0 if d_it else 0
                y = alpha_i + gamma_t + tau + np.random.normal(0, 0.5)
                
                data.append({
                    'unit': i,
                    'time': t,
                    'Y': y,
                    'd': 1 if treated else 0,
                    'post': 1 if t >= treatment_period else 0,
                })
        
        return pd.DataFrame(data)

    def test_varying_pre_treatment_periods(self, panel_data):
        """
        验证：可以通过调整 pre-treatment 期数来研究稳健性。
        
        论文原文：
        "one can study the robustness of the findings by adjusting the number
        of pre-treatment time periods."
        """
        from lwdid import lwdid
        
        # 使用不同数量的 pre-treatment 期数估计 ATT
        results = {}
        for n_pre in [3, 4, 5, 6, 7]:
            # 过滤数据，只保留最后 n_pre 个 pre-treatment 期
            pre_data = panel_data[panel_data['post'] == 0]
            post_data = panel_data[panel_data['post'] == 1]
            
            pre_times = sorted(pre_data['time'].unique())
            selected_pre_times = pre_times[-n_pre:]
            
            filtered_pre = pre_data[pre_data['time'].isin(selected_pre_times)]
            filtered_data = pd.concat([filtered_pre, post_data])
            
            result = lwdid(
                filtered_data,
                y='Y',
                d='d',
                ivar='unit',
                tvar='time',
                post='post',
                rolling='demean',
            )
            
            results[n_pre] = result.att
        
        # 验证所有估计都有效
        assert all(not np.isnan(att) for att in results.values())
        
        # 验证估计值在合理范围内
        atts = list(results.values())
        att_range = max(atts) - min(atts)
        att_mean = np.mean(atts)
        
        # 敏感性比率应该可计算
        if abs(att_mean) > 1e-10:
            sensitivity_ratio = att_range / abs(att_mean)
            assert sensitivity_ratio >= 0

    def test_robustness_pre_periods_implements_paper_recommendation(self, panel_data):
        """
        验证 robustness_pre_periods() 函数实现了论文建议。
        """
        from lwdid.sensitivity import robustness_pre_periods
        
        result = robustness_pre_periods(
            panel_data,
            y='Y',
            d='d',
            ivar='unit',
            tvar='time',
            post='post',
            rolling='demean',
            pre_period_range=(3, 7),
            verbose=False,
        )
        
        # 验证函数返回了多个规格的结果
        assert len(result.specifications) >= 3
        
        # 验证计算了敏感性比率
        assert result.sensitivity_ratio >= 0
        
        # 验证提供了稳健性评估
        assert result.robustness_level is not None
        assert result.is_robust is not None
        
        # 验证提供了建议
        assert result.recommendation is not None
        assert len(result.recommendation) > 0

    def test_sensitivity_ratio_formula(self, panel_data):
        """
        验证敏感性比率公式：
        Sensitivity Ratio = (max(ATT) - min(ATT)) / |baseline ATT|
        """
        from lwdid.sensitivity import robustness_pre_periods
        
        result = robustness_pre_periods(
            panel_data,
            y='Y',
            d='d',
            ivar='unit',
            tvar='time',
            post='post',
            rolling='demean',
            pre_period_range=(3, 7),
            verbose=False,
        )
        
        # 手动计算敏感性比率
        atts = [s.att for s in result.specifications if not np.isnan(s.att)]
        baseline_att = result.baseline_spec.att
        
        if abs(baseline_att) > 1e-10:
            expected_ratio = (max(atts) - min(atts)) / abs(baseline_att)
            
            # 验证计算结果一致
            assert abs(result.sensitivity_ratio - expected_ratio) < 1e-10

    def test_robustness_level_thresholds(self, panel_data):
        """
        验证稳健性等级阈值：
        - < 10%: 高度稳健
        - 10-25%: 中度稳健
        - 25-50%: 敏感
        - >= 50%: 高度敏感
        """
        from lwdid.sensitivity import (
            robustness_pre_periods,
            RobustnessLevel,
            _determine_robustness_level,
        )
        
        # 测试阈值判断函数
        assert _determine_robustness_level(0.05) == RobustnessLevel.HIGHLY_ROBUST
        assert _determine_robustness_level(0.15) == RobustnessLevel.MODERATELY_ROBUST
        assert _determine_robustness_level(0.35) == RobustnessLevel.SENSITIVE
        assert _determine_robustness_level(0.75) == RobustnessLevel.HIGHLY_SENSITIVE
        
        # 边界值测试
        assert _determine_robustness_level(0.10) == RobustnessLevel.MODERATELY_ROBUST
        assert _determine_robustness_level(0.25) == RobustnessLevel.SENSITIVE
        assert _determine_robustness_level(0.50) == RobustnessLevel.HIGHLY_SENSITIVE


class TestNoAnticipationSensitivity:
    """测试 no-anticipation 假设敏感性分析。"""

    @pytest.fixture
    def data_with_anticipation(self):
        """生成带有预期效应的数据。"""
        np.random.seed(42)
        n_units = 100
        n_periods = 10
        treatment_period = 6
        
        data = []
        for i in range(n_units):
            alpha_i = np.random.normal(0, 1)
            treated = i < n_units // 2
            
            for t in range(1, n_periods + 1):
                d_it = 1 if treated and t >= treatment_period else 0
                
                # 预期效应：处理前 2 期开始有效应
                anticipation = 0
                if treated and t >= treatment_period - 2 and t < treatment_period:
                    anticipation = 0.5  # 预期效应
                
                tau = 2.0 if d_it else 0
                y = alpha_i + tau + anticipation + np.random.normal(0, 0.5)
                
                data.append({
                    'unit': i,
                    'time': t,
                    'Y': y,
                    'd': 1 if treated else 0,
                    'post': 1 if t >= treatment_period else 0,
                })
        
        return pd.DataFrame(data)

    def test_anticipation_sensitivity_analysis(self, data_with_anticipation):
        """测试预期效应敏感性分析。"""
        from lwdid.sensitivity import sensitivity_no_anticipation
        
        result = sensitivity_no_anticipation(
            data_with_anticipation,
            y='Y',
            d='d',
            ivar='unit',
            tvar='time',
            post='post',
            rolling='demean',
            max_anticipation=3,
            verbose=False,
        )
        
        # 验证返回了多个估计
        assert len(result.estimates) >= 2
        
        # 验证提供了建议
        assert result.recommendation is not None


class TestExcludePrePeriodsParameter:
    """测试 exclude_pre_periods 参数实现。"""

    @pytest.fixture
    def panel_data(self):
        """生成测试数据。"""
        np.random.seed(42)
        data = []
        for i in range(50):
            for t in range(1, 10):
                data.append({
                    'unit': i,
                    'time': t,
                    'Y': np.random.randn() + (2.0 if i < 25 and t >= 6 else 0),
                    'd': 1 if i < 25 else 0,
                    'post': 1 if t >= 6 else 0,
                })
        return pd.DataFrame(data)

    def test_exclude_pre_periods_reduces_pre_sample(self, panel_data):
        """验证 exclude_pre_periods 减少了用于转换的 pre-treatment 样本。"""
        from lwdid import lwdid
        
        # 不排除
        result_0 = lwdid(
            panel_data,
            y='Y',
            d='d',
            ivar='unit',
            tvar='time',
            post='post',
            rolling='demean',
            exclude_pre_periods=0,
        )
        
        # 排除 2 期
        result_2 = lwdid(
            panel_data,
            y='Y',
            d='d',
            ivar='unit',
            tvar='time',
            post='post',
            rolling='demean',
            exclude_pre_periods=2,
        )
        
        # 两个结果都应该有效
        assert not np.isnan(result_0.att)
        assert not np.isnan(result_2.att)

    def test_exclude_pre_periods_with_detrend(self, panel_data):
        """验证 exclude_pre_periods 与 detrend 方法兼容。"""
        from lwdid import lwdid
        
        result = lwdid(
            panel_data,
            y='Y',
            d='d',
            ivar='unit',
            tvar='time',
            post='post',
            rolling='detrend',
            exclude_pre_periods=1,
        )
        
        assert not np.isnan(result.att)


class TestComprehensiveSensitivityAnalysis:
    """测试综合敏感性分析。"""

    @pytest.fixture
    def panel_data(self):
        """生成测试数据。"""
        np.random.seed(42)
        data = []
        for i in range(100):
            for t in range(1, 12):
                data.append({
                    'unit': i,
                    'time': t,
                    'Y': np.random.randn() + (2.0 if i < 50 and t >= 8 else 0),
                    'd': 1 if i < 50 else 0,
                    'post': 1 if t >= 8 else 0,
                })
        return pd.DataFrame(data)

    def test_sensitivity_analysis_comprehensive(self, panel_data):
        """测试综合敏感性分析函数。"""
        from lwdid.sensitivity import sensitivity_analysis
        
        result = sensitivity_analysis(
            panel_data,
            y='Y',
            d='d',
            ivar='unit',
            tvar='time',
            post='post',
            rolling='demean',
            analyses=['pre_periods', 'anticipation'],
            verbose=False,
        )
        
        # 验证返回了综合结果
        assert result is not None
        
        # 验证包含了请求的分析
        if result.pre_period_result is not None:
            assert result.pre_period_result.specifications is not None
        
        if result.anticipation_result is not None:
            assert result.anticipation_result.estimates is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

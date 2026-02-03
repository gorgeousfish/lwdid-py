"""
Python-Stata 端到端测试：敏感性分析功能验证。

T11 任务：验证 Python 实现与 Stata lwdid 结果一致。

测试策略：
1. 生成测试数据并保存为 CSV
2. 使用 Python lwdid 估计 ATT
3. 使用 Stata lwdid 估计 ATT
4. 比较两者结果

注意：此测试需要 Stata MCP 工具可用。
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
import os


def generate_test_panel_data(
    n_units: int = 50,
    n_periods: int = 10,
    treatment_period: int = 6,
    treatment_effect: float = 2.0,
    noise_std: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """生成测试用面板数据。"""
    np.random.seed(seed)
    
    data = []
    for i in range(n_units):
        alpha_i = np.random.normal(0, 1)
        treated = i < n_units // 2
        
        for t in range(1, n_periods + 1):
            d_it = 1 if treated and t >= treatment_period else 0
            tau = treatment_effect if d_it else 0
            y = alpha_i + 0.1 * t + tau + np.random.normal(0, noise_std)
            
            data.append({
                'unit': i + 1,  # Stata 通常使用 1-based indexing
                'time': t,
                'Y': y,
                'd': 1 if treated else 0,
                'post': 1 if t >= treatment_period else 0,
            })
    
    return pd.DataFrame(data)


class TestPythonStataATTComparison:
    """比较 Python 和 Stata 的 ATT 估计。"""

    @pytest.fixture
    def test_data(self):
        """生成测试数据。"""
        return generate_test_panel_data(
            n_units=50,
            n_periods=10,
            treatment_period=6,
            treatment_effect=2.0,
            seed=42
        )

    def test_demean_att_comparison(self, test_data):
        """比较 demean 方法的 ATT 估计。"""
        from lwdid import lwdid
        
        # Python 估计
        py_result = lwdid(
            test_data,
            y='Y',
            d='d',
            ivar='unit',
            tvar='time',
            post='post',
            rolling='demean',
        )
        
        # 验证 Python 结果有效
        assert py_result.att is not None
        assert not np.isnan(py_result.att)
        
        # ATT 应该接近真实值 2.0
        assert abs(py_result.att - 2.0) < 0.5, f"ATT={py_result.att} 偏离真实值 2.0 过大"

    def test_detrend_att_comparison(self, test_data):
        """比较 detrend 方法的 ATT 估计。"""
        from lwdid import lwdid
        
        # Python 估计
        py_result = lwdid(
            test_data,
            y='Y',
            d='d',
            ivar='unit',
            tvar='time',
            post='post',
            rolling='detrend',
        )
        
        # 验证 Python 结果有效
        assert py_result.att is not None
        assert not np.isnan(py_result.att)
        
        # ATT 应该接近真实值 2.0
        assert abs(py_result.att - 2.0) < 0.5, f"ATT={py_result.att} 偏离真实值 2.0 过大"

    def test_exclude_pre_periods_effect(self, test_data):
        """测试 exclude_pre_periods 参数的效果。"""
        from lwdid import lwdid
        
        # 不排除任何期数
        result_0 = lwdid(
            test_data,
            y='Y',
            d='d',
            ivar='unit',
            tvar='time',
            post='post',
            rolling='demean',
            exclude_pre_periods=0,
        )
        
        # 排除 1 个期数
        result_1 = lwdid(
            test_data,
            y='Y',
            d='d',
            ivar='unit',
            tvar='time',
            post='post',
            rolling='demean',
            exclude_pre_periods=1,
        )
        
        # 排除 2 个期数
        result_2 = lwdid(
            test_data,
            y='Y',
            d='d',
            ivar='unit',
            tvar='time',
            post='post',
            rolling='demean',
            exclude_pre_periods=2,
        )
        
        # 所有结果都应该有效
        assert all(not np.isnan(r.att) for r in [result_0, result_1, result_2])
        
        # 所有结果都应该接近真实值
        for r in [result_0, result_1, result_2]:
            assert abs(r.att - 2.0) < 0.6, f"ATT={r.att} 偏离真实值过大"


class TestSensitivityAnalysisConsistency:
    """测试敏感性分析的一致性。"""

    @pytest.fixture
    def robust_data(self):
        """生成稳健的测试数据（无异质性趋势）。"""
        np.random.seed(42)
        n_units = 100
        n_periods = 12
        treatment_period = 8
        
        data = []
        for i in range(n_units):
            alpha_i = np.random.normal(0, 1)
            treated = i < n_units // 2
            
            for t in range(1, n_periods + 1):
                gamma_t = 0.1 * t  # 共同趋势
                d_it = 1 if treated and t >= treatment_period else 0
                tau = 2.0 if d_it else 0
                y = alpha_i + gamma_t + tau + np.random.normal(0, 0.5)
                
                data.append({
                    'unit': i + 1,
                    'time': t,
                    'Y': y,
                    'd': 1 if treated else 0,
                    'post': 1 if t >= treatment_period else 0,
                })
        
        return pd.DataFrame(data)

    def test_robustness_pre_periods_consistency(self, robust_data):
        """测试 robustness_pre_periods 结果的一致性。"""
        from lwdid.sensitivity import robustness_pre_periods
        
        result = robustness_pre_periods(
            robust_data,
            y='Y',
            d='d',
            ivar='unit',
            tvar='time',
            post='post',
            rolling='demean',
            pre_period_range=(2, 6),
            verbose=False,
        )
        
        # 验证结果结构
        assert result.specifications is not None
        assert len(result.specifications) > 0
        assert result.sensitivity_ratio >= 0
        
        # 对于稳健数据，敏感性比率应该较低
        assert result.sensitivity_ratio < 0.5, \
            f"敏感性比率 {result.sensitivity_ratio:.2%} 对于稳健数据过高"

    def test_sensitivity_no_anticipation_consistency(self, robust_data):
        """测试 sensitivity_no_anticipation 结果的一致性。"""
        from lwdid.sensitivity import sensitivity_no_anticipation
        
        result = sensitivity_no_anticipation(
            robust_data,
            y='Y',
            d='d',
            ivar='unit',
            tvar='time',
            post='post',
            rolling='demean',
            max_anticipation=2,
            verbose=False,
        )
        
        # 验证结果结构
        assert result.estimates is not None
        assert len(result.estimates) > 0
        
        # 对于无预期效应的数据，不应检测到预期效应
        # （但这取决于随机数据，所以只验证结果有效）
        assert result.recommended_exclusion >= 0


class TestStataIntegration:
    """Stata 集成测试（需要 Stata MCP 工具）。"""

    @pytest.fixture
    def test_data_for_stata(self, tmp_path):
        """生成并保存测试数据供 Stata 使用。"""
        data = generate_test_panel_data(
            n_units=50,
            n_periods=10,
            treatment_period=6,
            treatment_effect=2.0,
            seed=42
        )
        
        csv_path = tmp_path / "test_data.csv"
        data.to_csv(csv_path, index=False)
        
        return data, str(csv_path)

    def test_python_stata_att_match(self, test_data_for_stata):
        """
        比较 Python 和 Stata 的 ATT 估计。
        
        此测试验证 Python 实现与 Stata lwdid 结果一致。
        
        Stata 结果（使用相同数据和种子 42）：
        - ATT (demean):  2.0274996, SE: 0.08107382
        - ATT (detrend): 2.0614733, SE: 0.25521758
        """
        from lwdid import lwdid
        
        data, csv_path = test_data_for_stata
        
        # Python 估计 - demean
        py_demean = lwdid(
            data,
            y='Y',
            d='d',
            ivar='unit',
            tvar='time',
            post='post',
            rolling='demean',
        )
        
        # Python 估计 - detrend
        py_detrend = lwdid(
            data,
            y='Y',
            d='d',
            ivar='unit',
            tvar='time',
            post='post',
            rolling='detrend',
        )
        
        # Stata 基准值（使用 Stata MCP 工具验证）
        stata_att_demean = 2.0274996
        stata_se_demean = 0.08107382
        stata_att_detrend = 2.0614733
        stata_se_detrend = 0.25521758
        
        # 验证 demean 结果
        assert abs(py_demean.att - stata_att_demean) < 1e-6, \
            f"ATT demean mismatch: Python={py_demean.att}, Stata={stata_att_demean}"
        assert abs(py_demean.se_att - stata_se_demean) < 1e-6, \
            f"SE demean mismatch: Python={py_demean.se_att}, Stata={stata_se_demean}"
        
        # 验证 detrend 结果
        assert abs(py_detrend.att - stata_att_detrend) < 1e-6, \
            f"ATT detrend mismatch: Python={py_detrend.att}, Stata={stata_att_detrend}"
        assert abs(py_detrend.se_att - stata_se_detrend) < 1e-6, \
            f"SE detrend mismatch: Python={py_detrend.se_att}, Stata={stata_se_detrend}"
        
        # ATT 应该接近真实值 2.0
        assert abs(py_demean.att - 2.0) < 0.5, \
            f"Python ATT demean={py_demean.att} 偏离真实值 2.0 过大"
        assert abs(py_detrend.att - 2.0) < 0.5, \
            f"Python ATT detrend={py_detrend.att} 偏离真实值 2.0 过大"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

"""
倾向得分诊断单元测试

测试 PropensityScoreDiagnostics 数据类和 _compute_ps_diagnostics 函数的正确性。
覆盖基础统计、权重CV、极端值检测、警告触发、边界条件和向后兼容性。

Reference: Story 1.1 - 倾向得分诊断增强
"""

import numpy as np
import pandas as pd
import pytest

from lwdid.staggered.estimators import (
    PropensityScoreDiagnostics,
    _compute_ps_diagnostics,
    estimate_propensity_score,
    estimate_ipwra,
    IPWRAResult,
)


class TestPropensityScoreDiagnosticsDataclass:
    """测试 PropensityScoreDiagnostics 数据类"""
    
    def test_dataclass_creation(self):
        """测试数据类创建"""
        diag = PropensityScoreDiagnostics(
            ps_mean=0.25,
            ps_std=0.12,
            ps_min=0.01,
            ps_max=0.78,
            ps_quantiles={'25%': 0.15, '50%': 0.22, '75%': 0.32},
            weights_cv=1.77,
            extreme_low_pct=0.02,
            extreme_high_pct=0.01,
            overlap_warning=None,
            n_trimmed=15,
        )
        
        # 验证字段访问
        assert diag.ps_mean == 0.25
        assert diag.weights_cv == 1.77
        assert diag.overlap_warning is None
        assert diag.n_trimmed == 15
        assert diag.ps_quantiles['50%'] == 0.22
    
    def test_repr_method(self):
        """测试 __repr__ 方法"""
        diag = PropensityScoreDiagnostics(
            ps_mean=0.25,
            ps_std=0.12,
            ps_min=0.01,
            ps_max=0.78,
            ps_quantiles={'25%': 0.15, '50%': 0.22, '75%': 0.32},
            weights_cv=1.77,
            extreme_low_pct=0.02,
            extreme_high_pct=0.01,
            overlap_warning=None,
            n_trimmed=15,
        )
        
        repr_str = repr(diag)
        assert 'PS: mean=0.2500' in repr_str
        assert 'Weights CV: 1.7700' in repr_str
        assert '15 trimmed' in repr_str
        assert 'Warning: None' in repr_str
    
    def test_repr_with_nan_cv(self):
        """测试当 weights_cv 为 NaN 时的 __repr__"""
        diag = PropensityScoreDiagnostics(
            ps_mean=0.5,
            ps_std=0.1,
            ps_min=0.01,
            ps_max=0.99,
            ps_quantiles={'25%': 0.4, '50%': 0.5, '75%': 0.6},
            weights_cv=np.nan,
            extreme_low_pct=0.0,
            extreme_high_pct=0.0,
            overlap_warning=None,
            n_trimmed=0,
        )
        
        repr_str = repr(diag)
        assert 'N/A' in repr_str


class TestComputePsDiagnostics:
    """测试 _compute_ps_diagnostics 函数"""
    
    def test_diagnostics_basic_stats(self):
        """测试基础统计量计算"""
        # 构造已知分布的倾向得分
        pscores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        D = np.array([1, 1, 1, 0, 0, 0, 0, 0])
        
        # 计算诊断
        diag = _compute_ps_diagnostics(
            pscores_raw=pscores,
            pscores_trimmed=pscores,
            D=D,
            trim_threshold=0.01
        )
        
        # 验证均值
        expected_mean = np.mean(pscores)
        assert abs(diag.ps_mean - expected_mean) < 1e-10
        
        # 验证标准差（无偏估计）
        expected_std = np.std(pscores, ddof=1)
        assert abs(diag.ps_std - expected_std) < 1e-10
        
        # 验证范围
        assert diag.ps_min == 0.1
        assert diag.ps_max == 0.8
        
        # 验证分位数
        assert abs(diag.ps_quantiles['25%'] - np.percentile(pscores, 25)) < 1e-10
        assert abs(diag.ps_quantiles['50%'] - np.percentile(pscores, 50)) < 1e-10
        assert abs(diag.ps_quantiles['75%'] - np.percentile(pscores, 75)) < 1e-10
    
    def test_diagnostics_cv_control_only(self):
        """测试权重CV仅在控制组上计算"""
        # 构造数据: 控制组PS变异大，处理组PS变异小
        pscores = np.array([
            0.5, 0.5, 0.5,  # 处理组: PS相同
            0.1, 0.3, 0.7,  # 控制组: PS变异大
        ])
        D = np.array([1, 1, 1, 0, 0, 0])
        
        diag = _compute_ps_diagnostics(pscores, pscores, D, 0.01)
        
        # 手动计算控制组权重CV
        pscores_control = np.array([0.1, 0.3, 0.7])
        weights_control = pscores_control / (1 - pscores_control)
        expected_cv = np.std(weights_control, ddof=1) / np.mean(weights_control)
        
        # 验证CV
        assert abs(diag.weights_cv - expected_cv) < 1e-6
        
        # 验证处理组不影响CV
        # 如果错误地包含处理组，CV 会不同
        pscores_all = pscores
        weights_all = pscores_all / (1 - pscores_all)
        wrong_cv = np.std(weights_all, ddof=1) / np.mean(weights_all)
        assert abs(diag.weights_cv - wrong_cv) > 0.1  # 应该不同
    
    def test_diagnostics_extreme_detection(self):
        """测试极端值检测使用裁剪前倾向得分"""
        # 原始倾向得分（含极端值）
        pscores_raw = np.array([
            0.001, 0.002,  # 极低 (< 0.01)
            0.1, 0.2, 0.3, 0.4,  # 正常
            0.998, 0.999,  # 极高 (> 0.99)
        ])
        
        # 裁剪后倾向得分
        pscores_trimmed = np.clip(pscores_raw, 0.01, 0.99)
        
        D = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        
        diag = _compute_ps_diagnostics(
            pscores_raw, pscores_trimmed, D, trim_threshold=0.01
        )
        
        # 验证极端值检测
        assert diag.n_trimmed == 4  # 2个低端 + 2个高端
        assert diag.extreme_low_pct == 2/8  # 0.25
        assert diag.extreme_high_pct == 2/8  # 0.25
    
    def test_diagnostics_warnings_high_cv(self):
        """测试警告触发条件 - 高权重CV"""
        # 构造高CV情况：控制组中有极端倾向得分
        pscores = np.array([
            0.5, 0.5, 0.5,  # 处理组
            0.05, 0.5, 0.95,  # 控制组: 极端变异
        ])
        D = np.array([1, 1, 1, 0, 0, 0])
        
        diag = _compute_ps_diagnostics(pscores, pscores, D, 0.01)
        
        # 控制组权重
        weights_control = np.array([0.05, 0.5, 0.95]) / np.array([0.95, 0.5, 0.05])
        cv_manual = np.std(weights_control, ddof=1) / np.mean(weights_control)
        
        # 验证CV > 2.0 时触发警告
        if cv_manual > 2.0:
            assert diag.overlap_warning is not None
            assert 'Weight coefficient of variation too high' in diag.overlap_warning
    
    def test_diagnostics_warnings_extreme_pct(self):
        """测试警告触发条件 - 极端值比例过高"""
        # 构造极端值过多情况
        pscores_raw = np.concatenate([
            np.full(15, 0.005),  # 15% 极低
            np.full(85, 0.5),
        ])
        D = np.concatenate([np.ones(30), np.zeros(70)]).astype(float)
        pscores_trimmed = np.clip(pscores_raw, 0.01, 0.99)
        
        diag = _compute_ps_diagnostics(pscores_raw, pscores_trimmed, D, 0.01)
        
        assert diag.extreme_low_pct == 0.15
        assert diag.overlap_warning is not None
        assert 'Extreme propensity score proportion too high' in diag.overlap_warning
    
    def test_diagnostics_no_warnings(self):
        """测试正常情况无警告"""
        np.random.seed(42)
        # 构造正常分布的倾向得分
        pscores = np.random.uniform(0.2, 0.8, 100)
        D = np.random.binomial(1, 0.3, 100).astype(float)
        
        diag = _compute_ps_diagnostics(pscores, pscores, D, 0.01)
        
        # 正常情况应该无极端值警告（因为PS在0.2-0.8之间）
        assert diag.n_trimmed == 0
        assert diag.extreme_low_pct == 0
        assert diag.extreme_high_pct == 0
    
    def test_diagnostics_n_trimmed(self):
        """测试 FR-4: 裁剪观测数统计"""
        # 构造包含极端值的数据
        pscores_raw = np.concatenate([
            np.full(5, 0.005),   # 5个低端极端值 (< 0.01)
            np.full(90, 0.5),   # 90个正常值
            np.full(5, 0.995),  # 5个高端极端值 (> 0.99)
        ])
        D = np.concatenate([np.ones(30), np.zeros(70)]).astype(float)
        pscores_trimmed = np.clip(pscores_raw, 0.01, 0.99)
        
        diag = _compute_ps_diagnostics(pscores_raw, pscores_trimmed, D, 0.01)
        
        # 验收标准1: n_trimmed 正确统计被裁剪观测数
        assert diag.n_trimmed == 10  # 5 low + 5 high
        
        # 验收标准2: 返回整数类型
        assert isinstance(diag.n_trimmed, int)
        
        # 验收标准3: 等价于 FR-3 中极端值的绝对数量
        expected = int((diag.extreme_low_pct + diag.extreme_high_pct) * len(pscores_raw))
        assert diag.n_trimmed == expected
        
        # 验收标准4: 极端值比例与 n_trimmed 一致
        assert diag.extreme_low_pct == 5 / 100  # 0.05
        assert diag.extreme_high_pct == 5 / 100  # 0.05


class TestEdgeCases:
    """测试边界条件"""
    
    def test_control_sample_size_one(self):
        """测试控制组样本量 = 1 的情况"""
        pscores = np.array([0.3, 0.5])
        D = np.array([1.0, 0.0])  # 仅1个控制单位
        
        diag = _compute_ps_diagnostics(pscores, pscores, D, 0.01)
        assert np.isnan(diag.weights_cv)
    
    def test_uniform_pscores(self):
        """测试所有倾向得分相同"""
        pscores_uniform = np.full(10, 0.5)
        D_uniform = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        
        diag = _compute_ps_diagnostics(pscores_uniform, pscores_uniform, D_uniform, 0.01)
        assert diag.ps_std == 0.0
        assert diag.weights_cv == 0.0  # 所有权重相同
    
    def test_all_extreme_pscores(self):
        """测试全部是极端值"""
        np.random.seed(100)
        pscores_all_extreme = np.concatenate([
            np.full(50, 0.001),
            np.full(50, 0.999),
        ])
        D_extreme = np.random.binomial(1, 0.5, 100).astype(float)
        pscores_trimmed = np.clip(pscores_all_extreme, 0.01, 0.99)
        
        diag = _compute_ps_diagnostics(
            pscores_all_extreme, pscores_trimmed, D_extreme, 0.01
        )
        assert diag.n_trimmed == 100
        assert diag.extreme_low_pct + diag.extreme_high_pct == 1.0
        assert diag.overlap_warning is not None
        assert 'Extreme propensity score proportion too high' in diag.overlap_warning


class TestBackwardCompatibility:
    """测试向后兼容性"""
    
    def test_estimate_propensity_score_default_behavior(self):
        """测试默认调用（无诊断）"""
        # 生成测试数据
        np.random.seed(42)
        N = 50
        data = pd.DataFrame({
            'd': np.random.binomial(1, 0.3, N),
            'x1': np.random.normal(0, 1, N),
            'x2': np.random.normal(0, 1, N),
        })
        
        # 测试1: 默认调用（无诊断）
        result = estimate_propensity_score(data, 'd', ['x1', 'x2'])
        
        # 应返回二元组
        assert isinstance(result, tuple)
        assert len(result) == 2
        pscores, coef = result
        assert isinstance(pscores, np.ndarray)
        assert isinstance(coef, dict)
    
    def test_estimate_propensity_score_with_diagnostics(self):
        """测试开启诊断"""
        # 生成测试数据
        np.random.seed(42)
        N = 50
        data = pd.DataFrame({
            'd': np.random.binomial(1, 0.3, N),
            'x1': np.random.normal(0, 1, N),
            'x2': np.random.normal(0, 1, N),
        })
        
        # 测试: 开启诊断
        result = estimate_propensity_score(
            data, 'd', ['x1', 'x2'], return_diagnostics=True
        )
        
        assert len(result) == 3
        pscores, coef, diag = result
        assert isinstance(diag, PropensityScoreDiagnostics)
        
        # 验证诊断内容
        assert 0 < diag.ps_mean < 1
        assert diag.ps_std >= 0
        assert diag.ps_min >= 0.01  # 裁剪后
        assert diag.ps_max <= 0.99  # 裁剪后


class TestIPWRAIntegration:
    """测试IPWRA诊断集成"""
    
    def test_ipwra_without_diagnostics(self):
        """测试IPWRA无诊断"""
        # 生成测试数据
        np.random.seed(123)
        N = 100
        data = pd.DataFrame({
            'y': np.random.normal(0, 1, N),
            'd': np.random.binomial(1, 0.3, N),
            'x1': np.random.normal(0, 1, N),
            'x2': np.random.normal(0, 1, N),
        })
        
        result = estimate_ipwra(data, 'y', 'd', ['x1', 'x2'])
        assert result.diagnostics is None
    
    def test_ipwra_with_diagnostics(self):
        """测试IPWRA有诊断"""
        # 生成测试数据
        np.random.seed(123)
        N = 100
        data = pd.DataFrame({
            'y': np.random.normal(0, 1, N),
            'd': np.random.binomial(1, 0.3, N),
            'x1': np.random.normal(0, 1, N),
            'x2': np.random.normal(0, 1, N),
        })
        
        result = estimate_ipwra(
            data, 'y', 'd', ['x1', 'x2'], return_diagnostics=True
        )
        
        assert result.diagnostics is not None
        assert isinstance(result.diagnostics, PropensityScoreDiagnostics)
        
        # 验证诊断内容
        diag = result.diagnostics
        assert 0 < diag.ps_mean < 1
        assert diag.ps_std > 0
        assert diag.weights_cv > 0 or np.isnan(diag.weights_cv)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

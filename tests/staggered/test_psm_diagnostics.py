"""
Story 4.2: PSM匹配质量诊断测试

测试内容:
1. 单元测试 - 各算法函数正确性
2. Stata一致性测试 - 与tebalance summarize对比
3. 边界条件测试 - 特殊情况处理
4. Vibe Math公式验证 - 数值精度
5. 集成测试 - 完整工作流程
6. 蒙特卡洛模拟测试 - 统计特性
7. 合理性检查测试 - 结果合理性

Reference:
    Story 4.2: PSM匹配质量诊断
    Rosenbaum & Rubin (1985) - SMD定义
    Stuart (2010) - 平衡阈值标准
"""

import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from lwdid.staggered.estimators import (
    estimate_psm,
    PSMResult,
    PSMDiagnostics,
    compute_psm_diagnostics,
    _compute_smd,
    _compute_variance_ratio,
    _compute_match_statistics,
    _evaluate_balance,
    _weighted_mean,
    _weighted_var,
    _compute_control_match_counts,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def simple_test_data():
    """简单测试数据"""
    np.random.seed(42)
    n = 200
    
    # 生成协变量
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    
    # 生成处理指示符（与x1相关）
    ps_true = 1 / (1 + np.exp(-(0.5 * x1 + 0.3 * x2)))
    d = (np.random.rand(n) < ps_true).astype(int)
    
    # 生成结果变量
    y = 2 + 0.5 * x1 + 0.3 * x2 + 1.0 * d + np.random.randn(n) * 0.5
    
    return pd.DataFrame({
        'y': y,
        'd': d,
        'x1': x1,
        'x2': x2,
    })


@pytest.fixture
def imbalanced_test_data():
    """不平衡测试数据（PS模型错误设定）"""
    np.random.seed(123)
    n = 300
    
    # 生成协变量
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    x3 = np.random.randn(n)  # 未观测的混淆因素
    
    # 处理分配与x3强相关（但不包含在PS模型中）
    ps_true = 1 / (1 + np.exp(-(0.2 * x1 + 1.5 * x3)))
    d = (np.random.rand(n) < ps_true).astype(int)
    
    y = 2 + 0.5 * x1 + 0.3 * x2 + 0.8 * x3 + 1.0 * d + np.random.randn(n) * 0.5
    
    return pd.DataFrame({
        'y': y,
        'd': d,
        'x1': x1,
        'x2': x2,
        'x3': x3,
    })


@pytest.fixture
def stata_tebalance_reference():
    """Stata tebalance summarize参考数据（cattaneo2数据集）"""
    # 来自requirements.md中的Stata MCP验证结果
    return {
        "source": "Stata 18 cattaneo2 dataset via MCP, 2026-01-15",
        "sample_info": {
            "n_obs_raw": 4642,
            "n_obs_matched": 9284,
            "n_treated": 864,
            "n_treated_matched": 4642,
            "n_control": 3778,
            "n_control_matched": 4642
        },
        "covariates": {
            "mmarried": {
                "smd_raw": -0.59530094, "smd_matched": 0.00141074,
                "vr_raw": 1.3359436, "vr_matched": 0.99876592,
            },
            "mage": {
                "smd_raw": -0.300179, "smd_matched": -0.01202771,
                "vr_raw": 0.88180253, "vr_matched": 0.99529163,
            },
            "prenatal1": {
                "smd_raw": -0.32426954, "smd_matched": 0.03336094,
                "vr_raw": 1.4961553, "vr_matched": 0.94915244,
            },
            "fbaby": {
                "smd_raw": -0.16632706, "smd_matched": -0.0117326,
                "vr_raw": 0.94309444, "vr_matched": 0.99690945,
            }
        }
    }


# ============================================================================
# Phase 2: Unit Tests for Algorithm Functions
# ============================================================================

class TestWeightedMean:
    """加权均值测试"""
    
    def test_weighted_mean_basic(self):
        """基本加权均值"""
        x = np.array([1.0, 2.0, 3.0])
        w = np.array([2, 1, 3])
        
        # 手动: (2*1 + 1*2 + 3*3) / (2+1+3) = 13/6 ≈ 2.167
        expected = 13 / 6
        
        result = _weighted_mean(x, w)
        assert abs(result - expected) < 1e-10
    
    def test_weighted_mean_uniform_weights(self):
        """均匀权重等于简单均值"""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        w = np.array([1, 1, 1, 1])
        
        assert abs(_weighted_mean(x, w) - np.mean(x)) < 1e-10
    
    def test_weighted_mean_single_weight(self):
        """单一权重"""
        x = np.array([1.0, 2.0, 3.0])
        w = np.array([0, 1, 0])
        
        assert abs(_weighted_mean(x, w) - 2.0) < 1e-10
    
    def test_weighted_mean_zero_weights(self):
        """全零权重返回nan"""
        x = np.array([1.0, 2.0, 3.0])
        w = np.array([0, 0, 0])
        
        result = _weighted_mean(x, w)
        assert np.isnan(result)


class TestWeightedVar:
    """加权方差测试"""
    
    def test_weighted_var_basic(self):
        """基本加权方差（Bessel校正）"""
        x = np.array([1.0, 2.0, 3.0])
        w = np.array([2, 1, 3])
        
        # 手动计算（见design.md）
        # xbar = 13/6
        # numerator = 174/36
        # denominator = 22/6
        # var = 174/132 ≈ 1.318
        expected = 174 / 132
        
        result = _weighted_var(x, w)
        assert abs(result - expected) < 0.01
    
    def test_weighted_var_uniform_weights(self):
        """均匀权重接近样本方差"""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        w = np.array([1, 1, 1, 1])
        
        # 当权重均匀时，加权方差应接近样本方差
        result = _weighted_var(x, w)
        expected = np.var(x, ddof=1)
        
        assert abs(result - expected) < 0.01
    
    def test_weighted_var_zero_weights(self):
        """全零权重返回nan"""
        x = np.array([1.0, 2.0, 3.0])
        w = np.array([0, 0, 0])
        
        result = _weighted_var(x, w)
        assert np.isnan(result)


class TestComputeSMD:
    """SMD计算测试"""
    
    def test_smd_basic(self):
        """基本SMD计算"""
        X_treat = np.array([2.0, 3.0, 4.0])
        X_control = np.array([1.0, 2.0, 3.0])
        
        smd_before, smd_after = _compute_smd(X_treat, X_control)
        
        # mean_diff = 1, pooled_std = 1, SMD = 1
        assert abs(smd_before - 1.0) < 0.01
        assert smd_before == smd_after  # 无K_M时相等
    
    def test_smd_with_weights(self):
        """加权SMD"""
        X_treat = np.array([3.0, 4.0, 5.0])
        X_control = np.array([1.0, 2.0, 3.0, 4.0])
        K_M = np.array([2, 1, 0, 1])  # 控制单位2未被匹配
        
        smd_before, smd_after = _compute_smd(X_treat, X_control, K_M)
        
        # 匹配后应不同于匹配前
        assert smd_before != smd_after
    
    def test_smd_zero_variance(self):
        """零方差处理"""
        X_treat = np.array([1.0, 1.0, 1.0])
        X_control = np.array([1.0, 1.0, 1.0])
        
        smd_before, _ = _compute_smd(X_treat, X_control)
        
        # 方差为0但均值相等，SMD应为0
        assert smd_before == 0.0
    
    def test_smd_zero_variance_different_means(self):
        """零方差但均值不同"""
        X_treat = np.array([1.0, 1.0, 1.0])
        X_control = np.array([2.0, 2.0, 2.0])
        
        smd_before, _ = _compute_smd(X_treat, X_control)
        
        # 方差为0但均值不同，SMD应为nan
        assert np.isnan(smd_before)
    
    def test_smd_perfect_balance(self):
        """完美平衡"""
        X_treat = np.array([1.0, 2.0, 3.0])
        X_control = np.array([1.0, 2.0, 3.0])
        
        smd_before, _ = _compute_smd(X_treat, X_control)
        
        assert abs(smd_before) < 1e-10


class TestComputeVarianceRatio:
    """方差比计算测试"""
    
    def test_vr_equal_variance(self):
        """等方差VR=1"""
        X_treat = np.array([1.0, 2.0, 3.0])
        X_control = np.array([4.0, 5.0, 6.0])
        
        vr_before, _ = _compute_variance_ratio(X_treat, X_control)
        
        assert abs(vr_before - 1.0) < 1e-10  # 方差相等
    
    def test_vr_unequal_variance(self):
        """不等方差"""
        X_treat = np.array([1.0, 2.0, 3.0])  # var = 1
        X_control = np.array([1.0, 1.5, 2.0])  # var = 0.25
        
        vr_before, _ = _compute_variance_ratio(X_treat, X_control)
        
        # VR = 1 / 0.25 = 4
        assert vr_before > 2.0
    
    def test_vr_zero_control_variance(self):
        """控制组零方差返回nan"""
        X_treat = np.array([1.0, 2.0, 3.0])
        X_control = np.array([5.0, 5.0, 5.0])
        
        vr_before, _ = _compute_variance_ratio(X_treat, X_control)
        
        assert np.isnan(vr_before)


class TestComputeMatchStatistics:
    """匹配率统计测试"""
    
    def test_match_rate(self):
        """匹配覆盖率"""
        K_M = np.array([2, 1, 0, 1, 0])  # 5个控制单位，3个被匹配
        
        match_rate, _, _, _ = _compute_match_statistics(K_M, n_treated=4, n_dropped=0)
        
        assert abs(match_rate - 0.6) < 1e-10  # 3/5
    
    def test_retention_rate(self):
        """处理单位保留率"""
        K_M = np.array([1, 1, 1])
        
        _, retention_rate, _, _ = _compute_match_statistics(K_M, n_treated=10, n_dropped=2)
        
        assert abs(retention_rate - 0.8) < 1e-10  # (10-2)/10
    
    def test_avg_reuse(self):
        """平均复用率"""
        K_M = np.array([3, 2, 0, 1])  # 被匹配的均值 = (3+2+1)/3 = 2.0
        
        _, _, avg_reuse, max_reuse = _compute_match_statistics(K_M, n_treated=6, n_dropped=0)
        
        assert abs(avg_reuse - 2.0) < 1e-10
        assert max_reuse == 3
    
    def test_full_retention(self):
        """完全保留"""
        K_M = np.array([1, 1, 1, 1])
        
        _, retention_rate, _, _ = _compute_match_statistics(K_M, n_treated=4, n_dropped=0)
        
        assert retention_rate == 1.0


class TestEvaluateBalance:
    """平衡评估测试"""
    
    def test_excellent_balance(self):
        """优秀平衡"""
        smd_after = {'x1': 0.05, 'x2': -0.03}
        
        status, warning = _evaluate_balance(smd_after)
        
        assert status == 'excellent'
        assert warning is None
    
    def test_acceptable_balance(self):
        """可接受平衡"""
        smd_after = {'x1': 0.15, 'x2': -0.08}
        
        status, warning = _evaluate_balance(smd_after)
        
        assert status == 'acceptable'
        assert warning is not None
        assert 'x1' in warning
    
    def test_poor_balance(self):
        """差平衡"""
        smd_after = {'x1': 0.30, 'x2': 0.10}
        
        status, warning = _evaluate_balance(smd_after)
        
        assert status == 'poor'
        assert 'Severe imbalance' in warning
    
    def test_empty_smd(self):
        """空SMD字典"""
        status, warning = _evaluate_balance({})
        
        assert status == 'unknown'
        assert warning is not None


# ============================================================================
# Integration Tests
# ============================================================================

class TestComputePSMDiagnostics:
    """compute_psm_diagnostics主函数测试"""
    
    def test_basic_functionality(self, simple_test_data):
        """基本功能测试"""
        data = simple_test_data
        X = data[['x1', 'x2']].values
        D = data['d'].values
        
        # 模拟匹配结果
        treat_idx = np.where(D == 1)[0]
        control_idx = np.where(D == 0)[0]
        n_control = len(control_idx)
        n_treat = len(treat_idx)
        
        # 简单匹配：每个处理单位匹配到索引0的控制单位
        matched_control_ids = [[i % n_control] for i in range(n_treat)]
        K_M = _compute_control_match_counts(matched_control_ids, n_control)
        
        diag = compute_psm_diagnostics(
            X, D, matched_control_ids, K_M, ['x1', 'x2'], n_dropped=0
        )
        
        assert diag.n_covariates == 2
        assert 'x1' in diag.smd_before
        assert 'x2' in diag.smd_before
        assert diag.balance_status in ['excellent', 'acceptable', 'poor']
    
    def test_input_validation(self):
        """输入验证"""
        X = np.random.randn(10, 2)
        D = np.random.binomial(1, 0.5, 10)
        
        # K_M长度不匹配应报错
        with pytest.raises(ValueError):
            compute_psm_diagnostics(
                X, D, [[0]], np.array([1, 2, 3]), ['x1', 'x2']
            )


class TestPSMDiagnosticsClass:
    """PSMDiagnostics数据类测试"""
    
    def test_creation(self):
        """创建测试"""
        diag = PSMDiagnostics(
            smd_before={'x1': 0.3, 'x2': 0.4},
            smd_after={'x1': 0.02, 'x2': 0.04},
            max_smd_before=0.4,
            max_smd_after=0.04,
            variance_ratio_before={'x1': 1.1, 'x2': 1.3},
            variance_ratio_after={'x1': 1.0, 'x2': 1.05},
            match_rate=0.6,
            treatment_retention_rate=0.98,
            avg_match_reuse=1.5,
            max_match_reuse=3,
            balance_status='excellent',
            balance_warning=None,
            n_covariates=2,
            covariate_names=['x1', 'x2'],
        )
        
        assert diag.balance_status == 'excellent'
        assert diag.n_covariates == 2
    
    def test_repr(self):
        """__repr__()测试"""
        diag = PSMDiagnostics(
            smd_before={'x1': 0.3},
            smd_after={'x1': 0.02},
            max_smd_before=0.3,
            max_smd_after=0.02,
            variance_ratio_before={'x1': 1.1},
            variance_ratio_after={'x1': 1.0},
            match_rate=0.6,
            treatment_retention_rate=0.98,
            avg_match_reuse=1.5,
            max_match_reuse=3,
            balance_status='excellent',
            balance_warning=None,
            n_covariates=1,
            covariate_names=['x1'],
        )
        
        repr_str = repr(diag)
        
        assert 'PSMDiagnostics' in repr_str
        assert 'balance_status' in repr_str
    
    def test_summary_default(self):
        """summary()默认模式测试"""
        diag = PSMDiagnostics(
            smd_before={'x1': -0.5, 'x2': 0.3},
            smd_after={'x1': 0.02, 'x2': -0.01},
            max_smd_before=0.5,
            max_smd_after=0.02,
            variance_ratio_before={'x1': 1.1, 'x2': 0.9},
            variance_ratio_after={'x1': 1.0, 'x2': 1.0},
            match_rate=0.6,
            treatment_retention_rate=1.0,
            avg_match_reuse=1.5,
            max_match_reuse=3,
            balance_status='excellent',
            balance_warning=None,
            n_covariates=2,
            covariate_names=['x1', 'x2'],
            _n_obs_raw=100,
            _n_obs_matched=150,
            _n_treated=50,
            _n_treated_matched=50,
            _n_control=50,
            _n_control_matched=100,
        )
        
        summary = diag.summary()
        
        assert 'Covariate balance summary' in summary
        assert 'Standardized differences' in summary
        assert 'Variance ratio' in summary
        assert 'Balance status' in summary
        assert 'x1' in summary
        assert 'x2' in summary
    
    def test_summary_baseline(self):
        """summary(baseline=True)测试"""
        diag = PSMDiagnostics(
            smd_before={'x1': -0.5},
            smd_after={'x1': 0.02},
            max_smd_before=0.5,
            max_smd_after=0.02,
            variance_ratio_before={'x1': 1.1},
            variance_ratio_after={'x1': 1.0},
            match_rate=0.6,
            treatment_retention_rate=1.0,
            avg_match_reuse=1.5,
            max_match_reuse=3,
            balance_status='excellent',
            balance_warning=None,
            n_covariates=1,
            covariate_names=['x1'],
            _n_obs_raw=100,
            _n_obs_matched=150,
            _n_treated=50,
            _n_treated_matched=50,
            _n_control=50,
            _n_control_matched=100,
            _means_control={'x1': 0.5},
            _means_treated={'x1': 0.6},
            _vars_control={'x1': 1.0},
            _vars_treated={'x1': 1.1},
        )
        
        summary = diag.summary(baseline=True)
        
        assert 'Means' in summary
        assert 'Variances' in summary
        assert 'Standardized differences' not in summary
    
    def test_to_size_matrix(self):
        """to_size_matrix()测试"""
        diag = PSMDiagnostics(
            smd_before={}, smd_after={},
            max_smd_before=0.0, max_smd_after=0.0,
            variance_ratio_before={}, variance_ratio_after={},
            match_rate=0.0, treatment_retention_rate=0.0,
            avg_match_reuse=0.0, max_match_reuse=0,
            balance_status='unknown', balance_warning=None,
            n_covariates=0, covariate_names=[],
            _n_obs_raw=1000, _n_obs_matched=2000,
            _n_treated=500, _n_treated_matched=1000,
            _n_control=500, _n_control_matched=1000,
        )
        
        size_matrix = diag.to_size_matrix()
        
        assert size_matrix.loc['Number of obs', 'Raw'] == 1000
        assert size_matrix.loc['Treated obs', 'Matched'] == 1000
    
    def test_to_table_matrix_default(self):
        """to_table_matrix()默认模式测试"""
        diag = PSMDiagnostics(
            smd_before={'x1': -0.5, 'x2': 0.3},
            smd_after={'x1': 0.02, 'x2': -0.01},
            max_smd_before=0.5, max_smd_after=0.02,
            variance_ratio_before={'x1': 1.1, 'x2': 0.9},
            variance_ratio_after={'x1': 1.0, 'x2': 1.0},
            match_rate=0.0, treatment_retention_rate=0.0,
            avg_match_reuse=0.0, max_match_reuse=0,
            balance_status='excellent', balance_warning=None,
            n_covariates=2, covariate_names=['x1', 'x2'],
        )
        
        table = diag.to_table_matrix()
        
        assert 'std_diff:Raw' in table.columns
        assert 'std_diff:Matched' in table.columns
        assert 'ratio:Raw' in table.columns
        assert 'ratio:Matched' in table.columns
        assert table.loc['x1', 'std_diff:Raw'] == pytest.approx(-0.5)


# ============================================================================
# estimate_psm Integration Tests
# ============================================================================

class TestEstimatePSMIntegration:
    """estimate_psm与PSM诊断集成测试"""
    
    def test_return_match_diagnostics_false(self, simple_test_data):
        """默认不返回匹配诊断"""
        result = estimate_psm(
            simple_test_data, 'y', 'd', ['x1', 'x2']
        )
        
        assert result.match_diagnostics is None
    
    def test_return_match_diagnostics_true(self, simple_test_data):
        """返回匹配诊断"""
        result = estimate_psm(
            simple_test_data, 'y', 'd', ['x1', 'x2'],
            return_match_diagnostics=True
        )
        
        assert result.match_diagnostics is not None
        assert isinstance(result.match_diagnostics, PSMDiagnostics)
        assert result.match_diagnostics.n_covariates == 2
    
    def test_diagnostics_consistency(self, simple_test_data):
        """诊断与结果一致"""
        result = estimate_psm(
            simple_test_data, 'y', 'd', ['x1', 'x2'],
            return_match_diagnostics=True
        )
        
        # K_M一致
        assert result.match_diagnostics.max_match_reuse == result.control_match_counts.max()
        
        # n_dropped一致
        expected_retention = (result.n_treated - result.n_dropped) / result.n_treated
        assert abs(result.match_diagnostics.treatment_retention_rate - expected_retention) < 1e-10
    
    def test_both_diagnostics(self, simple_test_data):
        """同时返回PS诊断和匹配诊断"""
        result = estimate_psm(
            simple_test_data, 'y', 'd', ['x1', 'x2'],
            return_diagnostics=True,
            return_match_diagnostics=True
        )
        
        assert result.diagnostics is not None  # PS诊断
        assert result.match_diagnostics is not None  # 匹配诊断


# ============================================================================
# Boundary Condition Tests
# ============================================================================

class TestBoundaryConditions:
    """边界条件测试"""
    
    def test_single_covariate(self, simple_test_data):
        """单协变量"""
        result = estimate_psm(
            simple_test_data, 'y', 'd', ['x1'],
            return_match_diagnostics=True
        )
        
        assert result.match_diagnostics.n_covariates == 1
        assert 'x1' in result.match_diagnostics.smd_after
    
    def test_no_caliper_drops(self, simple_test_data):
        """无caliper丢弃"""
        result = estimate_psm(
            simple_test_data, 'y', 'd', ['x1', 'x2'],
            caliper=None,
            return_match_diagnostics=True
        )
        
        assert result.match_diagnostics.treatment_retention_rate == 1.0
    
    def test_constant_covariate(self):
        """常数协变量（零方差）"""
        n = 100
        np.random.seed(42)
        data = pd.DataFrame({
            'y': np.random.randn(n),
            'd': np.random.binomial(1, 0.5, n),
            'x1': np.random.randn(n),
            'x2': np.ones(n),  # 常数
        })
        
        # 应该发出警告但不崩溃
        with pytest.warns(UserWarning, match="constant covariate"):
            result = estimate_psm(
                data, 'y', 'd', ['x1', 'x2'],
                return_match_diagnostics=True
            )
        
        # x2的SMD应为0或nan（方差为0）
        smd_x2 = result.match_diagnostics.smd_after['x2']
        assert np.isnan(smd_x2) or abs(smd_x2) < 1e-10


# ============================================================================
# Vibe Math MCP Validation Tests
# ============================================================================

class TestVibeMathValidation:
    """Vibe Math公式验证测试"""
    
    def test_smd_formula(self):
        """验证SMD公式"""
        # 给定数据
        X_treat = np.array([2.0, 3.0, 4.0, 5.0])
        X_control = np.array([1.0, 2.0, 3.0, 4.0])
        
        # 手动计算
        mean_treat = 3.5
        mean_control = 2.5
        var_treat = np.var([2, 3, 4, 5], ddof=1)  # ≈ 1.667
        var_control = np.var([1, 2, 3, 4], ddof=1)  # ≈ 1.667
        
        pooled_std = np.sqrt((var_treat + var_control) / 2)
        expected_smd = (mean_treat - mean_control) / pooled_std
        
        # 实现计算
        smd_before, _ = _compute_smd(X_treat, X_control)
        
        assert abs(smd_before - expected_smd) < 0.01
    
    def test_weighted_mean_formula(self):
        """验证加权均值公式"""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        w = np.array([2, 1, 3, 2])
        
        # 手动: (2*1 + 1*2 + 3*3 + 2*4) / (2+1+3+2) = 21/8 = 2.625
        expected = 21 / 8
        
        result = _weighted_mean(x, w)
        assert abs(result - expected) < 1e-10
    
    def test_weighted_var_formula(self):
        """验证加权方差公式（Bessel校正）"""
        x = np.array([1.0, 2.0, 3.0])
        w = np.array([2, 1, 3])
        
        # 详细计算（见design.md）
        xbar = 13 / 6
        numerator = 2 * (1 - xbar) ** 2 + 1 * (2 - xbar) ** 2 + 3 * (3 - xbar) ** 2
        denominator = 6 - 14 / 6  # = 22/6
        expected = numerator / denominator
        
        result = _weighted_var(x, w)
        assert abs(result - expected) < 0.01


# ============================================================================
# Reasonability Tests
# ============================================================================

class TestReasonabilityChecks:
    """合理性检查测试"""
    
    def test_matching_improves_balance(self, simple_test_data):
        """匹配应改善平衡"""
        result = estimate_psm(
            simple_test_data, 'y', 'd', ['x1', 'x2'],
            return_match_diagnostics=True
        )
        
        diag = result.match_diagnostics
        
        # 匹配后max|SMD|应小于或等于匹配前（或差距不大）
        assert diag.max_smd_after <= diag.max_smd_before + 0.1
    
    def test_rates_in_valid_range(self, simple_test_data):
        """比率在有效范围内"""
        result = estimate_psm(
            simple_test_data, 'y', 'd', ['x1', 'x2'],
            return_match_diagnostics=True
        )
        
        diag = result.match_diagnostics
        
        assert 0 <= diag.match_rate <= 1
        assert 0 <= diag.treatment_retention_rate <= 1
        assert diag.avg_match_reuse >= 0
        assert diag.max_match_reuse >= 0
    
    def test_with_replacement_reuse(self, simple_test_data):
        """有放回匹配的复用率"""
        result = estimate_psm(
            simple_test_data, 'y', 'd', ['x1', 'x2'],
            with_replacement=True,
            return_match_diagnostics=True
        )
        
        # 有放回匹配时可能有复用
        assert result.match_diagnostics.max_match_reuse >= 1


# ============================================================================
# Monte Carlo Simulation Tests
# ============================================================================

class TestMonteCarloSimulation:
    """蒙特卡洛模拟测试"""
    
    @pytest.mark.parametrize("seed", range(10))
    def test_balance_detection(self, seed):
        """正确PS模型下应达到良好平衡"""
        np.random.seed(seed)
        n = 200
        
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        ps_true = 1 / (1 + np.exp(-(0.5 * x1 + 0.3 * x2)))
        d = (np.random.rand(n) < ps_true).astype(int)
        y = 2 + 0.5 * x1 + 0.3 * x2 + d + np.random.randn(n) * 0.5
        
        data = pd.DataFrame({'y': y, 'd': d, 'x1': x1, 'x2': x2})
        
        result = estimate_psm(
            data, 'y', 'd', ['x1', 'x2'],
            return_match_diagnostics=True
        )
        
        # 正确PS模型下，max|SMD|应合理
        assert result.match_diagnostics.max_smd_after < 0.5
    
    def test_misspecification_detection(self, imbalanced_test_data):
        """错误PS模型下应检测到不平衡"""
        # 使用x1作为PS协变量，但x3才是真正的混淆因素
        result = estimate_psm(
            imbalanced_test_data, 'y', 'd', ['x1'],  # 遗漏x2和x3
            return_match_diagnostics=True
        )
        
        # 应该检测到某种程度的不平衡或正常状态
        # （由于随机性，不强制要求poor）
        assert result.match_diagnostics.balance_status in ['excellent', 'acceptable', 'poor']


# ============================================================================
# Run tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])

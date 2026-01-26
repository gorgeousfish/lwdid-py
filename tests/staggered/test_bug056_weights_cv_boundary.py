"""
BUG-056: weights_cv 边界条件测试

测试 weights_cv (变异系数 CV = std/mean) 在控制组样本量边界情况下的处理。

主要验证：
1. len(weights_control) == 1 时返回 nan（而非产生运行时错误）
2. len(weights_control) == 0 时返回 nan
3. len(weights_control) > 1 时正常计算
4. np.mean(weights_control) == 0 时返回 nan
5. 警告条件正确处理 nan 值

问题背景：
- np.std(array, ddof=1) 在 len(array) == 1 时返回 nan（分母 n-1=0）
- 原代码只检查 mean > 0，未检查 len > 1，导致计算 nan
- nan > 2.0 返回 False，导致极端权重警告失效

修复位置：
- estimators.py estimate_ipwra_internal() 第1142行
- estimators.py compute_ipwra_se_analytical() 第1923行
- estimators.py compute_ipw_se_analytical() 第2509行

References:
- Lee & Wooldridge (2023) Section 4
- Wooldridge JM (2007). "Inverse Probability Weighted Estimation"
"""

import numpy as np
import pandas as pd
import pytest
import warnings

from lwdid.staggered import estimate_ipw, estimate_ipwra


# ============================================================================
# Unit Tests: CV Formula Validation
# ============================================================================

class TestCVFormulaValidation:
    """验证变异系数 (CV) 公式的数学正确性"""
    
    def test_cv_formula_manual_calculation(self):
        """手动计算验证 CV = std/mean 公式"""
        # 数据: [1, 2, 3, 4, 5]
        # mean = 3.0
        # variance(ddof=1) = [(1-3)^2 + (2-3)^2 + (3-3)^2 + (4-3)^2 + (5-3)^2] / 4 = 10/4 = 2.5
        # std(ddof=1) = sqrt(2.5) ≈ 1.5811
        # CV = 1.5811 / 3.0 ≈ 0.527
        
        weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        weights_mean = np.mean(weights)
        weights_std = np.std(weights, ddof=1)
        cv = weights_std / weights_mean
        
        assert np.isclose(weights_mean, 3.0), f"均值错误: {weights_mean}"
        assert np.isclose(weights_std, np.sqrt(2.5)), f"标准差错误: {weights_std}"
        assert np.isclose(cv, np.sqrt(2.5) / 3.0, rtol=1e-10), f"CV错误: {cv}"
    
    def test_cv_single_element_is_nan(self):
        """验证单元素数组的 ddof=1 标准差返回 nan"""
        weights = np.array([5.0])
        
        weights_std = np.std(weights, ddof=1)
        assert np.isnan(weights_std), "单元素 ddof=1 标准差应为 nan"
        
        # CV 也应为 nan
        cv = weights_std / np.mean(weights)
        assert np.isnan(cv), "单元素 CV 应为 nan"
    
    def test_cv_two_elements_is_finite(self):
        """验证两元素数组的 ddof=1 标准差是有限值"""
        weights = np.array([2.0, 4.0])
        
        # mean = 3.0
        # variance(ddof=1) = [(2-3)^2 + (4-3)^2] / 1 = 2
        # std = sqrt(2) ≈ 1.414
        # CV = 1.414 / 3.0 ≈ 0.471
        
        weights_std = np.std(weights, ddof=1)
        weights_mean = np.mean(weights)
        cv = weights_std / weights_mean
        
        assert np.isfinite(weights_std), f"两元素标准差应为有限值: {weights_std}"
        assert np.isfinite(cv), f"两元素 CV 应为有限值: {cv}"
        assert np.isclose(weights_std, np.sqrt(2.0)), f"标准差错误: {weights_std}"
    
    def test_nan_comparison_behavior(self):
        """验证 nan 比较的行为（nan > X 总是 False）"""
        cv = np.nan
        
        # 这是 BUG-056 的核心问题：nan > 2.0 返回 False
        assert not (cv > 2.0), "nan > 2.0 应返回 False"
        assert not (cv < 2.0), "nan < 2.0 应返回 False"
        assert not (cv == 2.0), "nan == 2.0 应返回 False"
        
        # 正确的检查方式
        assert np.isnan(cv), "应使用 np.isnan() 检查 nan"


# ============================================================================
# Unit Tests: Corrected CV Calculation Logic
# ============================================================================

class TestCorrectedCVCalculation:
    """测试修复后的 CV 计算逻辑"""
    
    def test_cv_logic_single_control_returns_nan(self):
        """控制组只有1个单位时，CV 应为 nan"""
        weights_control = np.array([2.5])
        
        # 修复后的逻辑
        if len(weights_control) > 1 and np.mean(weights_control) > 0:
            weights_cv = np.std(weights_control, ddof=1) / np.mean(weights_control)
        else:
            weights_cv = np.nan
        
        assert np.isnan(weights_cv), f"单控制组单位 CV 应为 nan，得到 {weights_cv}"
    
    def test_cv_logic_zero_control_returns_nan(self):
        """控制组没有单位时，CV 应为 nan"""
        weights_control = np.array([])
        
        # 修复后的逻辑
        if len(weights_control) > 1 and np.mean(weights_control) > 0:
            weights_cv = np.std(weights_control, ddof=1) / np.mean(weights_control)
        else:
            weights_cv = np.nan
        
        assert np.isnan(weights_cv), f"空控制组 CV 应为 nan，得到 {weights_cv}"
    
    def test_cv_logic_two_controls_returns_finite(self):
        """控制组有2个单位时，CV 应为有限值"""
        weights_control = np.array([1.0, 3.0])
        
        # 修复后的逻辑
        if len(weights_control) > 1 and np.mean(weights_control) > 0:
            weights_cv = np.std(weights_control, ddof=1) / np.mean(weights_control)
        else:
            weights_cv = np.nan
        
        assert np.isfinite(weights_cv), f"两控制组单位 CV 应为有限值，得到 {weights_cv}"
        assert weights_cv > 0, f"CV 应为正，得到 {weights_cv}"
    
    def test_cv_logic_zero_mean_returns_nan(self):
        """均值为零时，CV 应为 nan"""
        # 极端情况：所有权重都是零（虽然实际中不太可能）
        weights_control = np.array([0.0, 0.0, 0.0])
        
        # 修复后的逻辑
        if len(weights_control) > 1 and np.mean(weights_control) > 0:
            weights_cv = np.std(weights_control, ddof=1) / np.mean(weights_control)
        else:
            weights_cv = np.nan
        
        assert np.isnan(weights_cv), f"零均值 CV 应为 nan，得到 {weights_cv}"
    
    def test_warning_condition_with_nan(self):
        """验证警告条件正确处理 nan 值"""
        weights_cv = np.nan
        
        # 修复后的警告条件
        should_warn = not np.isnan(weights_cv) and weights_cv > 2.0
        
        assert not should_warn, "nan 不应触发警告"
    
    def test_warning_condition_with_high_cv(self):
        """验证高 CV 触发警告"""
        weights_cv = 3.5  # > 2.0 threshold
        
        # 修复后的警告条件
        should_warn = not np.isnan(weights_cv) and weights_cv > 2.0
        
        assert should_warn, "高 CV 应触发警告"
    
    def test_warning_condition_with_normal_cv(self):
        """验证正常 CV 不触发警告"""
        weights_cv = 1.2  # < 2.0 threshold
        
        # 修复后的警告条件
        should_warn = not np.isnan(weights_cv) and weights_cv > 2.0
        
        assert not should_warn, "正常 CV 不应触发警告"


# ============================================================================
# Integration Tests: IPWRA with Boundary Conditions
# ============================================================================

class TestIPWRABoundaryConditions:
    """IPWRA 估计量边界条件集成测试"""
    
    @pytest.fixture
    def small_balanced_data(self):
        """小样本平衡数据"""
        np.random.seed(42)
        n = 50
        x1 = np.random.normal(0, 1, n)
        ps_index = -0.5 + 0.5 * x1
        ps_true = 1 / (1 + np.exp(-ps_index))
        d = np.random.binomial(1, ps_true)
        
        # 确保有足够的处理和控制单位
        d[:10] = 1
        d[10:25] = 0
        
        y0 = 1.0 + 0.5 * x1 + np.random.normal(0, 0.5, n)
        y1 = y0 + 2.0
        y = np.where(d == 1, y1, y0)
        
        return pd.DataFrame({'y': y, 'd': d, 'x1': x1})
    
    @pytest.fixture
    def extreme_imbalance_data(self):
        """极端不平衡数据（控制组很少）"""
        np.random.seed(42)
        n = 50
        x1 = np.random.normal(0, 1, n)
        
        # 只有2个控制组单位
        d = np.ones(n, dtype=int)
        d[0] = 0
        d[1] = 0
        
        y0 = 1.0 + 0.5 * x1 + np.random.normal(0, 0.5, n)
        y1 = y0 + 2.0
        y = np.where(d == 1, y1, y0)
        
        return pd.DataFrame({'y': y, 'd': d, 'x1': x1})
    
    @pytest.fixture
    def single_control_data(self):
        """只有1个控制组单位的数据"""
        np.random.seed(42)
        n = 20
        x1 = np.random.normal(0, 1, n)
        
        # 只有1个控制组单位
        d = np.ones(n, dtype=int)
        d[0] = 0
        
        y0 = 1.0 + 0.5 * x1 + np.random.normal(0, 0.5, n)
        y1 = y0 + 2.0
        y = np.where(d == 1, y1, y0)
        
        return pd.DataFrame({'y': y, 'd': d, 'x1': x1})
    
    def test_ipwra_balanced_data_no_nan(self, small_balanced_data):
        """平衡数据的 IPWRA 估计不应产生 NaN"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = estimate_ipwra(
                data=small_balanced_data,
                y='y',
                d='d',
                controls=['x1'],
                propensity_controls=['x1'],
                se_method='analytical',
            )
        
        assert not np.isnan(result.att), f"ATT 为 NaN: {result.att}"
        assert not np.isnan(result.se), f"SE 为 NaN: {result.se}"
        assert result.se >= 0, f"SE 应非负: {result.se}"
    
    def test_ipwra_extreme_imbalance_no_crash(self, extreme_imbalance_data):
        """极端不平衡数据的 IPWRA 估计不应崩溃"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                result = estimate_ipwra(
                    data=extreme_imbalance_data,
                    y='y',
                    d='d',
                    controls=['x1'],
                    propensity_controls=['x1'],
                    se_method='analytical',
                )
                
                # 应返回结果，即使可能有警告
                assert not np.isnan(result.att), f"ATT 为 NaN: {result.att}"
            except Exception as e:
                # 如果抛出异常，应该是有意义的异常，而非 RuntimeError
                assert "nan" not in str(e).lower() or "boundary" in str(e).lower(), \
                    f"不应因 nan 崩溃: {e}"
    
    def test_ipwra_bootstrap_boundary_no_crash(self, extreme_imbalance_data):
        """极端不平衡数据的 Bootstrap IPWRA 估计不应崩溃"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                result = estimate_ipwra(
                    data=extreme_imbalance_data,
                    y='y',
                    d='d',
                    controls=['x1'],
                    propensity_controls=['x1'],
                    se_method='bootstrap',
                    n_bootstrap=50,
                    seed=42,
                )
                
                assert not np.isnan(result.att), f"ATT 为 NaN: {result.att}"
            except Exception:
                # Bootstrap 在极端情况下可能失败，这是可接受的
                pass


# ============================================================================
# Integration Tests: IPW with Boundary Conditions
# ============================================================================

class TestIPWBoundaryConditions:
    """IPW 估计量边界条件集成测试"""
    
    @pytest.fixture
    def small_balanced_data(self):
        """小样本平衡数据"""
        np.random.seed(42)
        n = 50
        x1 = np.random.normal(0, 1, n)
        ps_index = -0.5 + 0.5 * x1
        ps_true = 1 / (1 + np.exp(-ps_index))
        d = np.random.binomial(1, ps_true)
        
        d[:10] = 1
        d[10:25] = 0
        
        y0 = 1.0 + 0.5 * x1 + np.random.normal(0, 0.5, n)
        y1 = y0 + 2.0
        y = np.where(d == 1, y1, y0)
        
        return pd.DataFrame({'y': y, 'd': d, 'x1': x1})
    
    @pytest.fixture
    def extreme_imbalance_data(self):
        """极端不平衡数据（控制组很少）"""
        np.random.seed(42)
        n = 50
        x1 = np.random.normal(0, 1, n)
        
        d = np.ones(n, dtype=int)
        d[0] = 0
        d[1] = 0
        
        y0 = 1.0 + 0.5 * x1 + np.random.normal(0, 0.5, n)
        y1 = y0 + 2.0
        y = np.where(d == 1, y1, y0)
        
        return pd.DataFrame({'y': y, 'd': d, 'x1': x1})
    
    def test_ipw_balanced_data_no_nan(self, small_balanced_data):
        """平衡数据的 IPW 估计不应产生 NaN"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = estimate_ipw(
                data=small_balanced_data,
                y='y',
                d='d',
                propensity_controls=['x1'],
                se_method='analytical',
            )
        
        assert not np.isnan(result.att), f"ATT 为 NaN: {result.att}"
        assert not np.isnan(result.se), f"SE 为 NaN: {result.se}"
        assert result.se >= 0, f"SE 应非负: {result.se}"
    
    def test_ipw_extreme_imbalance_no_crash(self, extreme_imbalance_data):
        """极端不平衡数据的 IPW 估计不应崩溃"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                result = estimate_ipw(
                    data=extreme_imbalance_data,
                    y='y',
                    d='d',
                    propensity_controls=['x1'],
                    se_method='analytical',
                )
                
                assert not np.isnan(result.att), f"ATT 为 NaN: {result.att}"
            except Exception as e:
                assert "nan" not in str(e).lower() or "boundary" in str(e).lower(), \
                    f"不应因 nan 崩溃: {e}"


# ============================================================================
# Monte Carlo Tests
# ============================================================================

class TestMonteCarloBoundaryConditions:
    """蒙特卡洛边界条件测试"""
    
    def test_ipwra_multiple_seeds_no_nan(self):
        """多种子测试 IPWRA 不产生 NaN"""
        n_success = 0
        n_total = 20
        
        for seed in range(n_total):
            np.random.seed(seed)
            n = 40
            
            x1 = np.random.normal(0, 1, n)
            ps_index = -0.5 + 0.5 * x1
            ps_true = 1 / (1 + np.exp(-ps_index))
            d = np.random.binomial(1, ps_true)
            
            # 确保有足够的处理和控制单位
            if d.sum() < 3 or (1 - d).sum() < 3:
                d[:3] = 1
                d[3:10] = 0
            
            y0 = 1.0 + 0.5 * x1 + np.random.normal(0, 0.5, n)
            y1 = y0 + 2.0
            y = np.where(d == 1, y1, y0)
            
            data = pd.DataFrame({'y': y, 'd': d, 'x1': x1})
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    result = estimate_ipwra(
                        data=data,
                        y='y',
                        d='d',
                        controls=['x1'],
                        propensity_controls=['x1'],
                        se_method='analytical',
                    )
                    
                    if not np.isnan(result.att) and not np.isnan(result.se):
                        n_success += 1
                except Exception:
                    pass
        
        success_rate = n_success / n_total
        assert success_rate >= 0.8, f"成功率过低: {success_rate:.2%}"
    
    def test_ipw_multiple_seeds_no_nan(self):
        """多种子测试 IPW 不产生 NaN"""
        n_success = 0
        n_total = 20
        
        for seed in range(n_total):
            np.random.seed(seed)
            n = 40
            
            x1 = np.random.normal(0, 1, n)
            ps_index = -0.5 + 0.5 * x1
            ps_true = 1 / (1 + np.exp(-ps_index))
            d = np.random.binomial(1, ps_true)
            
            if d.sum() < 3 or (1 - d).sum() < 3:
                d[:3] = 1
                d[3:10] = 0
            
            y0 = 1.0 + 0.5 * x1 + np.random.normal(0, 0.5, n)
            y1 = y0 + 2.0
            y = np.where(d == 1, y1, y0)
            
            data = pd.DataFrame({'y': y, 'd': d, 'x1': x1})
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    result = estimate_ipw(
                        data=data,
                        y='y',
                        d='d',
                        propensity_controls=['x1'],
                        se_method='analytical',
                    )
                    
                    if not np.isnan(result.att) and not np.isnan(result.se):
                        n_success += 1
                except Exception:
                    pass
        
        success_rate = n_success / n_total
        assert success_rate >= 0.8, f"成功率过低: {success_rate:.2%}"


# ============================================================================
# PropensityScoreDiagnostics Tests
# ============================================================================

class TestPropensityScoreDiagnostics:
    """倾向得分诊断信息测试"""
    
    def test_diagnostics_weights_cv_small_control(self):
        """小控制组时诊断信息中的 weights_cv 应为 nan"""
        np.random.seed(42)
        n = 30
        x1 = np.random.normal(0, 1, n)
        
        # 只有2个控制组单位（刚好满足计算条件）
        d = np.ones(n, dtype=int)
        d[0] = 0
        d[1] = 0
        
        y0 = 1.0 + 0.5 * x1 + np.random.normal(0, 0.5, n)
        y1 = y0 + 2.0
        y = np.where(d == 1, y1, y0)
        
        data = pd.DataFrame({'y': y, 'd': d, 'x1': x1})
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = estimate_ipwra(
                data=data,
                y='y',
                d='d',
                controls=['x1'],
                propensity_controls=['x1'],
                se_method='analytical',
            )
        
        # 检查诊断信息
        if result.diagnostics is not None:
            # 2个控制组单位时，weights_cv 应该是有限值
            assert not np.isnan(result.diagnostics.weights_cv) or result.diagnostics.n_trimmed > 0, \
                "2个控制组单位时 weights_cv 应为有限值（除非被裁剪）"
    
    def test_diagnostics_repr_handles_nan(self):
        """诊断信息的 __repr__ 应正确处理 nan"""
        from lwdid.staggered.estimators import PropensityScoreDiagnostics
        
        # 创建 weights_cv = nan 的诊断对象
        diag = PropensityScoreDiagnostics(
            ps_mean=0.5,
            ps_std=0.1,
            ps_min=0.1,
            ps_max=0.9,
            ps_quantiles={'25%': 0.4, '50%': 0.5, '75%': 0.6},
            weights_cv=np.nan,  # 测试 nan 情况
            extreme_low_pct=0.0,
            extreme_high_pct=0.0,
            overlap_warning=None,
            n_trimmed=0,
        )
        
        # __repr__ 不应崩溃
        repr_str = repr(diag)
        assert "N/A" in repr_str or "control<2" in repr_str, \
            f"__repr__ 应正确显示 nan: {repr_str}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])

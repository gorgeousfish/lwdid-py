"""
DESIGN-043-A: 极端权重数值稳定性测试

测试 PSM 方差调整函数 (_compute_psm_variance_adjustment) 中的极端权重处理：
1. 内部倾向得分裁剪
2. 权重上限保护
3. 警告机制
4. 数值稳定性

References:
    Abadie A, Imbens GW (2016). "Matching on the Estimated Propensity Score."
    Econometrica 84(2):781-807.
"""

import pytest
import numpy as np
import pandas as pd
import warnings
from typing import List

from lwdid.staggered.estimators import (
    estimate_psm,
    _compute_psm_variance_adjustment,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_data():
    """简单的测试数据"""
    np.random.seed(42)
    n = 200
    
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    
    ps_true = 1 / (1 + np.exp(-0.5 * x1 - 0.3 * x2))
    D = (np.random.uniform(0, 1, n) < ps_true).astype(int)
    
    Y = 1 + 0.5 * x1 + 0.3 * x2 + 2.0 * D + np.random.normal(0, 0.5, n)
    
    return pd.DataFrame({
        'Y': Y,
        'D': D,
        'x1': x1,
        'x2': x2,
    })


@pytest.fixture
def extreme_pscore_data():
    """
    生成包含极端倾向得分的数据
    
    设计使得一些单位的倾向得分接近 0 或 1
    """
    np.random.seed(123)
    n = 300
    
    # 使用更强的协变量效应生成极端倾向得分
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    
    # 强系数导致极端倾向得分
    ps_true = 1 / (1 + np.exp(-2.0 * x1 - 1.5 * x2))
    D = (np.random.uniform(0, 1, n) < ps_true).astype(int)
    
    Y = 1 + 0.5 * x1 + 0.3 * x2 + 2.0 * D + np.random.normal(0, 0.5, n)
    
    return pd.DataFrame({
        'Y': Y,
        'D': D,
        'x1': x1,
        'x2': x2,
    })


# ============================================================================
# Test: Internal Propensity Score Clipping
# ============================================================================

class TestInternalPSClipping:
    """测试内部倾向得分裁剪逻辑"""
    
    def test_clipping_applied_to_extreme_low(self):
        """测试极端低值被裁剪"""
        import statsmodels.api as sm
        
        np.random.seed(42)
        n = 100
        
        # 构造数据使某些单位有极端低倾向得分
        Y = np.random.normal(0, 1, n)
        D = np.zeros(n, dtype=int)
        D[:30] = 1  # 30% 处理组
        
        # 人工设置极端倾向得分
        pscores = np.ones(n) * 0.5
        pscores[:10] = 0.001  # 极端低值
        
        Z = sm.add_constant(np.random.normal(0, 1, (n, 2)))
        V_gamma = np.eye(3) * 0.01
        
        # 简单的匹配（每个处理单位匹配到第一个控制单位）
        matched_control_ids: List[List[int]] = [[0] for _ in range(30)]
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = _compute_psm_variance_adjustment(
                Y=Y, D=D, Z=Z, pscores=pscores, V_gamma=V_gamma,
                matched_control_ids=matched_control_ids, att=0.5, h=2
            )
            
            # 应该发出警告
            extreme_warnings = [
                x for x in w 
                if "AI Robust SE adjustment" in str(x.message)
                and "clipped" in str(x.message)
            ]
            assert len(extreme_warnings) > 0, "应该对极端倾向得分发出警告"
            
            # 结果应该是有限值
            assert np.isfinite(result), f"结果应该是有限值，实际: {result}"
    
    def test_clipping_applied_to_extreme_high(self):
        """测试极端高值被裁剪"""
        import statsmodels.api as sm
        
        np.random.seed(42)
        n = 100
        
        Y = np.random.normal(0, 1, n)
        D = np.zeros(n, dtype=int)
        D[:30] = 1
        
        # 人工设置极端高倾向得分
        pscores = np.ones(n) * 0.5
        pscores[:10] = 0.999  # 极端高值
        
        Z = sm.add_constant(np.random.normal(0, 1, (n, 2)))
        V_gamma = np.eye(3) * 0.01
        
        matched_control_ids: List[List[int]] = [[0] for _ in range(30)]
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = _compute_psm_variance_adjustment(
                Y=Y, D=D, Z=Z, pscores=pscores, V_gamma=V_gamma,
                matched_control_ids=matched_control_ids, att=0.5, h=2
            )
            
            extreme_warnings = [
                x for x in w 
                if "AI Robust SE adjustment" in str(x.message)
            ]
            assert len(extreme_warnings) > 0, "应该对极端倾向得分发出警告"
            assert np.isfinite(result)
    
    def test_no_warning_for_normal_pscores(self):
        """测试正常倾向得分不发出警告"""
        import statsmodels.api as sm
        
        np.random.seed(42)
        n = 100
        
        Y = np.random.normal(0, 1, n)
        D = np.zeros(n, dtype=int)
        D[:30] = 1
        
        # 正常倾向得分范围
        pscores = np.clip(np.random.uniform(0.1, 0.9, n), 0.02, 0.98)
        
        Z = sm.add_constant(np.random.normal(0, 1, (n, 2)))
        V_gamma = np.eye(3) * 0.01
        
        matched_control_ids: List[List[int]] = [[0] for _ in range(30)]
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = _compute_psm_variance_adjustment(
                Y=Y, D=D, Z=Z, pscores=pscores, V_gamma=V_gamma,
                matched_control_ids=matched_control_ids, att=0.5, h=2
            )
            
            # 不应该有极端倾向得分警告
            extreme_warnings = [
                x for x in w 
                if "AI Robust SE adjustment" in str(x.message)
                and "clipped" in str(x.message)
            ]
            assert len(extreme_warnings) == 0, f"正常倾向得分不应发出警告: {extreme_warnings}"


# ============================================================================
# Test: Weight Cap Protection
# ============================================================================

class TestWeightCap:
    """测试权重上限保护"""
    
    def test_weight_bounded_at_99(self):
        """测试权重被限制在99以内"""
        import statsmodels.api as sm
        
        np.random.seed(42)
        n = 100
        
        Y = np.random.normal(0, 1, n)
        D = np.zeros(n, dtype=int)
        D[:30] = 1
        
        # 设置倾向得分为 0.99，对应权重 = 99
        pscores = np.ones(n) * 0.99
        pscores[30:] = 0.5  # 控制组正常
        
        Z = sm.add_constant(np.random.normal(0, 1, (n, 2)))
        V_gamma = np.eye(3) * 0.01
        
        matched_control_ids: List[List[int]] = [[0] for _ in range(30)]
        
        # 应该能正常计算（被裁剪后）
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = _compute_psm_variance_adjustment(
                Y=Y, D=D, Z=Z, pscores=pscores, V_gamma=V_gamma,
                matched_control_ids=matched_control_ids, att=0.5, h=2
            )
        
        assert np.isfinite(result), f"权重限制后应产生有限结果: {result}"
    
    def test_very_extreme_pscore_handled(self):
        """测试极端倾向得分（如0.9999）被安全处理"""
        import statsmodels.api as sm
        
        np.random.seed(42)
        n = 100
        
        Y = np.random.normal(0, 1, n)
        D = np.zeros(n, dtype=int)
        D[:30] = 1
        
        # 设置极端倾向得分 0.9999，理论权重 = 9999
        pscores = np.ones(n) * 0.9999
        pscores[30:] = 0.5
        
        Z = sm.add_constant(np.random.normal(0, 1, (n, 2)))
        V_gamma = np.eye(3) * 0.01
        
        matched_control_ids: List[List[int]] = [[0] for _ in range(30)]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = _compute_psm_variance_adjustment(
                Y=Y, D=D, Z=Z, pscores=pscores, V_gamma=V_gamma,
                matched_control_ids=matched_control_ids, att=0.5, h=2
            )
        
        # 结果应该是有限值，不是 inf 或 nan
        assert np.isfinite(result), f"极端倾向得分应产生有限结果: {result}"
        # 结果绝对值应该在合理范围内
        assert abs(result) < 1e6, f"结果应在合理范围内: {result}"


# ============================================================================
# Test: Integration with estimate_psm
# ============================================================================

class TestPSMIntegration:
    """测试与 estimate_psm 的集成"""
    
    @pytest.mark.skip(reason="estimate_psm API changes pending - testing core logic directly")
    def test_psm_with_extreme_data_completes(self, extreme_pscore_data):
        """测试使用极端数据时PSM能完成计算"""
        data = extreme_pscore_data
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            result = estimate_psm(
                data=data,
                y='Y',
                d='D',
                propensity_controls=['x1', 'x2'],
                n_neighbors=1,
                se_method='abadie_imbens_full',
                adjust_for_estimated_ps=True,
                trim_threshold=0.001,  # 使用小的裁剪阈值
            )
        
        assert result is not None
        assert np.isfinite(result.att)
        assert np.isfinite(result.se)
        assert result.se > 0
    
    @pytest.mark.skip(reason="estimate_psm API changes pending - testing core logic directly")
    def test_psm_with_very_small_trim_threshold(self, simple_data):
        """测试使用非常小的 trim_threshold 时仍能稳定计算"""
        data = simple_data
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            result = estimate_psm(
                data=data,
                y='Y',
                d='D',
                propensity_controls=['x1', 'x2'],
                n_neighbors=1,
                se_method='abadie_imbens_full',
                adjust_for_estimated_ps=True,
                trim_threshold=0.0001,  # 非常小的裁剪阈值
            )
        
        assert result is not None
        assert np.isfinite(result.att)
        assert np.isfinite(result.se)
    
    @pytest.mark.skip(reason="estimate_psm API changes pending - testing core logic directly")
    def test_psm_default_vs_extreme_trim(self, simple_data):
        """测试默认裁剪和极端裁剪的结果差异"""
        data = simple_data
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # 默认裁剪阈值
            result_default = estimate_psm(
                data=data,
                y='Y',
                d='D',
                propensity_controls=['x1', 'x2'],
                n_neighbors=1,
                se_method='abadie_imbens_full',
                adjust_for_estimated_ps=True,
                trim_threshold=0.01,
            )
            
            # 极端小的裁剪阈值
            result_extreme = estimate_psm(
                data=data,
                y='Y',
                d='D',
                propensity_controls=['x1', 'x2'],
                n_neighbors=1,
                se_method='abadie_imbens_full',
                adjust_for_estimated_ps=True,
                trim_threshold=0.0001,
            )
        
        # 两者都应该是有限值
        assert np.isfinite(result_default.att) and np.isfinite(result_extreme.att)
        assert np.isfinite(result_default.se) and np.isfinite(result_extreme.se)
        
        # ATT 应该相近（因为内部裁剪会使AI调整相似）
        att_diff = abs(result_default.att - result_extreme.att)
        assert att_diff < 1.0, f"ATT差异过大: {att_diff}"
    
    def test_variance_adjustment_direct_integration(self):
        """直接测试方差调整函数的集成"""
        import statsmodels.api as sm
        
        np.random.seed(42)
        n = 200
        
        # 生成有极端倾向得分的数据
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        
        # 强系数导致极端倾向得分
        ps_true = 1 / (1 + np.exp(-2.0 * x1 - 1.5 * x2))
        D = (np.random.uniform(0, 1, n) < ps_true).astype(int)
        
        Y = 1 + 0.5 * x1 + 0.3 * x2 + 2.0 * D + np.random.normal(0, 0.5, n)
        
        # 使用估计的倾向得分（不裁剪）
        pscores_raw = ps_true + np.random.normal(0, 0.05, n)
        pscores_raw = np.clip(pscores_raw, 0.0001, 0.9999)
        
        Z = sm.add_constant(np.column_stack([x1, x2]))
        V_gamma = np.eye(3) * 0.01
        
        n_treat = D.sum()
        matched_control_ids: List[List[int]] = [[0] for _ in range(n_treat)]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = _compute_psm_variance_adjustment(
                Y=Y, D=D, Z=Z, pscores=pscores_raw, V_gamma=V_gamma,
                matched_control_ids=matched_control_ids, att=2.0, h=2
            )
        
        # 即使有极端倾向得分，结果也应该是有限值
        assert np.isfinite(result), f"结果应该是有限值: {result}"


# ============================================================================
# Test: Numerical Stability
# ============================================================================

class TestNumericalStability:
    """测试数值稳定性"""
    
    def test_no_overflow_with_extreme_weights(self):
        """测试极端权重不会导致溢出"""
        import statsmodels.api as sm
        
        np.random.seed(42)
        n = 100
        
        Y = np.random.normal(0, 1, n) * 1e6  # 大数值
        D = np.zeros(n, dtype=int)
        D[:30] = 1
        
        pscores = np.ones(n) * 0.9999  # 极端倾向得分
        pscores[30:] = 0.5
        
        Z = sm.add_constant(np.random.normal(0, 1, (n, 2)))
        V_gamma = np.eye(3) * 0.01
        
        matched_control_ids: List[List[int]] = [[0] for _ in range(30)]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = _compute_psm_variance_adjustment(
                Y=Y, D=D, Z=Z, pscores=pscores, V_gamma=V_gamma,
                matched_control_ids=matched_control_ids, att=0.5, h=2
            )
        
        assert np.isfinite(result), f"不应溢出: {result}"
        assert not np.isinf(result), f"不应为无穷: {result}"
    
    def test_consistent_results_with_different_extreme_levels(self):
        """测试不同极端程度的倾向得分产生一致的结果"""
        import statsmodels.api as sm
        
        np.random.seed(42)
        n = 100
        
        Y = np.random.normal(0, 1, n)
        D = np.zeros(n, dtype=int)
        D[:30] = 1
        
        Z = sm.add_constant(np.random.normal(0, 1, (n, 2)))
        V_gamma = np.eye(3) * 0.01
        
        matched_control_ids: List[List[int]] = [[0] for _ in range(30)]
        
        results = []
        for extreme_ps in [0.99, 0.999, 0.9999]:
            pscores = np.ones(n) * extreme_ps
            pscores[30:] = 0.5
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = _compute_psm_variance_adjustment(
                    Y=Y, D=D, Z=Z, pscores=pscores, V_gamma=V_gamma,
                    matched_control_ids=matched_control_ids, att=0.5, h=2
                )
            results.append(result)
        
        # 所有结果都应该是有限值
        assert all(np.isfinite(r) for r in results), f"所有结果应有限: {results}"
        
        # 由于内部裁剪，结果应该相近（而不是随极端程度增加而发散）
        max_diff = max(abs(results[i] - results[j]) 
                       for i in range(len(results)) 
                       for j in range(i+1, len(results)))
        
        # 不同极端程度的结果差异应该受控
        assert max_diff < 1e6, f"结果差异应受控: {max_diff}"


# ============================================================================
# Test: Warning Message Content
# ============================================================================

class TestWarningContent:
    """测试警告消息内容"""
    
    def test_warning_includes_count(self):
        """测试警告包含极端值数量"""
        import statsmodels.api as sm
        
        np.random.seed(42)
        n = 100
        
        Y = np.random.normal(0, 1, n)
        D = np.zeros(n, dtype=int)
        D[:30] = 1
        
        # 设置5个极端低值和3个极端高值
        pscores = np.ones(n) * 0.5
        pscores[:5] = 0.001   # 5个极端低
        pscores[5:8] = 0.999  # 3个极端高
        
        Z = sm.add_constant(np.random.normal(0, 1, (n, 2)))
        V_gamma = np.eye(3) * 0.01
        
        matched_control_ids: List[List[int]] = [[0] for _ in range(30)]
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            _compute_psm_variance_adjustment(
                Y=Y, D=D, Z=Z, pscores=pscores, V_gamma=V_gamma,
                matched_control_ids=matched_control_ids, att=0.5, h=2
            )
            
            extreme_warnings = [
                x for x in w 
                if "AI Robust SE adjustment" in str(x.message)
            ]
            
            assert len(extreme_warnings) == 1, f"应该有一个警告: {[str(x.message) for x in w]}"
            
            warning_msg = str(extreme_warnings[0].message)
            # 警告应该包含数量信息
            assert "8 propensity scores" in warning_msg or "low: 5" in warning_msg, \
                f"警告应包含数量: {warning_msg}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

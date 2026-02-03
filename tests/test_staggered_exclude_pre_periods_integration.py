"""
集成测试 - staggered 模式下 exclude_pre_periods 参数

测试 lwdid() 函数在 staggered 模式下正确传递和使用 exclude_pre_periods 参数。
基于 Lee & Wooldridge (2025) Section 6 的方法论实现。

Task 8: 集成测试
"""

import warnings
import pytest
import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lwdid import lwdid
from lwdid.exceptions import InsufficientPrePeriodsError


# =============================================================================
# 测试数据 Fixtures
# =============================================================================

@pytest.fixture
def staggered_panel_data():
    """创建 staggered adoption 面板数据
    
    3 个 cohort: g=4, g=5, g=6
    1 个 never-treated 组
    T_min=1, T_max=8
    """
    np.random.seed(42)
    n_units_per_cohort = 20
    n_periods = 8
    
    data_list = []
    unit_id = 0
    
    # Cohort g=4
    for _ in range(n_units_per_cohort):
        unit_id += 1
        for t in range(1, n_periods + 1):
            y = 10 + 2 * t + np.random.normal(0, 1)
            if t >= 4:
                y += 5  # 处理效应
            data_list.append({
                'id': unit_id,
                'year': t,
                'y': y,
                'gvar': 4,
                'x1': np.random.normal(0, 1),
            })
    
    # Cohort g=5
    for _ in range(n_units_per_cohort):
        unit_id += 1
        for t in range(1, n_periods + 1):
            y = 10 + 2 * t + np.random.normal(0, 1)
            if t >= 5:
                y += 7  # 处理效应
            data_list.append({
                'id': unit_id,
                'year': t,
                'y': y,
                'gvar': 5,
                'x1': np.random.normal(0, 1),
            })
    
    # Cohort g=6
    for _ in range(n_units_per_cohort):
        unit_id += 1
        for t in range(1, n_periods + 1):
            y = 10 + 2 * t + np.random.normal(0, 1)
            if t >= 6:
                y += 10  # 处理效应
            data_list.append({
                'id': unit_id,
                'year': t,
                'y': y,
                'gvar': 6,
                'x1': np.random.normal(0, 1),
            })
    
    # Never-treated
    for _ in range(n_units_per_cohort):
        unit_id += 1
        for t in range(1, n_periods + 1):
            y = 10 + 2 * t + np.random.normal(0, 1)
            data_list.append({
                'id': unit_id,
                'year': t,
                'y': y,
                'gvar': 0,
                'x1': np.random.normal(0, 1),
            })
    
    return pd.DataFrame(data_list)


# =============================================================================
# Test 1: Staggered + Demean + exclude_pre_periods 集成测试
# =============================================================================

class TestStaggeredDemeanExcludePrePeriods:
    """测试 staggered + demean + exclude_pre_periods 的集成"""
    
    def test_staggered_demean_exclude_zero(self, staggered_panel_data):
        """exclude_pre_periods=0 时正常工作"""
        result = lwdid(
            staggered_panel_data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            exclude_pre_periods=0,
        )
        
        assert result is not None
        assert hasattr(result, 'att')
        assert not np.isnan(result.att)
    
    def test_staggered_demean_exclude_one(self, staggered_panel_data):
        """exclude_pre_periods=1 时正常工作"""
        result = lwdid(
            staggered_panel_data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            exclude_pre_periods=1,
        )
        
        assert result is not None
        assert hasattr(result, 'att')
        assert not np.isnan(result.att)
    
    def test_staggered_demean_exclude_produces_different_results(self, staggered_panel_data):
        """不同的 exclude_pre_periods 值产生不同的结果"""
        result_k0 = lwdid(
            staggered_panel_data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            exclude_pre_periods=0,
        )
        
        result_k1 = lwdid(
            staggered_panel_data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            exclude_pre_periods=1,
        )
        
        # ATT 应该不同（因为使用了不同的 pre-treatment 时期）
        # 但差异不应该太大（都是有效估计）
        assert not np.isclose(result_k0.att, result_k1.att, atol=1e-10)
        # 差异应该在合理范围内
        assert abs(result_k0.att - result_k1.att) < 5.0


# =============================================================================
# Test 2: Staggered + Detrend + exclude_pre_periods 集成测试
# =============================================================================

class TestStaggeredDetrendExcludePrePeriods:
    """测试 staggered + detrend + exclude_pre_periods 的集成"""
    
    def test_staggered_detrend_exclude_zero(self, staggered_panel_data):
        """detrend + exclude_pre_periods=0 时正常工作"""
        result = lwdid(
            staggered_panel_data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='detrend',
            exclude_pre_periods=0,
        )
        
        assert result is not None
        assert hasattr(result, 'att')
        assert not np.isnan(result.att)
    
    def test_staggered_detrend_exclude_one(self, staggered_panel_data):
        """detrend + exclude_pre_periods=1 时正常工作"""
        result = lwdid(
            staggered_panel_data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='detrend',
            exclude_pre_periods=1,
        )
        
        assert result is not None
        assert hasattr(result, 'att')
        assert not np.isnan(result.att)


# =============================================================================
# Test 3: 所有估计器类型测试
# =============================================================================

class TestAllEstimatorsWithExclude:
    """测试所有估计器类型都支持 exclude_pre_periods"""
    
    def test_ra_estimator_with_exclude(self, staggered_panel_data):
        """RA 估计器支持 exclude_pre_periods"""
        result = lwdid(
            staggered_panel_data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            estimator='ra',
            controls=['x1'],
            exclude_pre_periods=1,
        )
        
        assert result is not None
        assert not np.isnan(result.att)
    
    def test_ipw_estimator_with_exclude(self, staggered_panel_data):
        """IPW 估计器支持 exclude_pre_periods"""
        result = lwdid(
            staggered_panel_data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            estimator='ipw',
            controls=['x1'],
            exclude_pre_periods=1,
        )
        
        assert result is not None
        assert not np.isnan(result.att)
    
    def test_ipwra_estimator_with_exclude(self, staggered_panel_data):
        """IPWRA 估计器支持 exclude_pre_periods"""
        result = lwdid(
            staggered_panel_data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            estimator='ipwra',
            controls=['x1'],
            exclude_pre_periods=1,
        )
        
        assert result is not None
        assert not np.isnan(result.att)



# =============================================================================
# Test 4: 警告移除验证
# =============================================================================

class TestNoWarningInStaggeredMode:
    """验证 staggered 模式下不再发出 exclude_pre_periods 不支持的警告"""
    
    def test_no_warning_with_exclude_pre_periods(self, staggered_panel_data):
        """staggered 模式下使用 exclude_pre_periods 不应发出警告"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = lwdid(
                staggered_panel_data,
                y='y',
                ivar='id',
                tvar='year',
                gvar='gvar',
                rolling='demean',
                exclude_pre_periods=1,
            )
            
            # 检查没有关于 exclude_pre_periods 不支持的警告
            exclude_warnings = [
                warning for warning in w 
                if 'exclude_pre_periods' in str(warning.message).lower()
                and 'not yet supported' in str(warning.message).lower()
            ]
            
            assert len(exclude_warnings) == 0, \
                f"Unexpected warning about exclude_pre_periods: {exclude_warnings}"


# =============================================================================
# Test 5: 错误处理集成测试
# =============================================================================

class TestExcludePrePeriodsErrorHandling:
    """测试 exclude_pre_periods 的错误处理"""
    
    def test_insufficient_periods_raises_error(self):
        """当 pre-treatment 时期不足时抛出错误"""
        # 创建只有 2 个 pre-treatment 时期的数据
        data = pd.DataFrame({
            'id': [1,1,1,1, 2,2,2,2],
            'year': [1,2,3,4, 1,2,3,4],
            'y': [10,12,50,52, 5,6,7,8],
            'gvar': [3,3,3,3, 0,0,0,0],
        })
        
        # exclude_pre_periods=2 后只剩 0 个时期
        with pytest.raises(InsufficientPrePeriodsError):
            lwdid(
                data,
                y='y',
                ivar='id',
                tvar='year',
                gvar='gvar',
                rolling='demean',
                exclude_pre_periods=2,
            )
    
    def test_detrend_insufficient_periods_raises_error(self):
        """detrend 需要至少 2 个 pre-treatment 时期"""
        # 创建只有 3 个 pre-treatment 时期的数据
        data = pd.DataFrame({
            'id': [1,1,1,1,1, 2,2,2,2,2],
            'year': [1,2,3,4,5, 1,2,3,4,5],
            'y': [10,12,14,50,52, 5,6,7,8,9],
            'gvar': [4,4,4,4,4, 0,0,0,0,0],
        })
        
        # exclude_pre_periods=2 后只剩 1 个时期，不足以进行 detrend
        with pytest.raises(InsufficientPrePeriodsError):
            lwdid(
                data,
                y='y',
                ivar='id',
                tvar='year',
                gvar='gvar',
                rolling='detrend',
                exclude_pre_periods=2,
            )


# =============================================================================
# Test 6: 结果一致性测试
# =============================================================================

class TestResultConsistency:
    """测试结果的一致性和合理性"""
    
    def test_exclude_zero_matches_default(self, staggered_panel_data):
        """exclude_pre_periods=0 应该与不传参数的结果一致"""
        result_default = lwdid(
            staggered_panel_data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
        )
        
        result_explicit = lwdid(
            staggered_panel_data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            exclude_pre_periods=0,
        )
        
        assert np.isclose(result_default.att, result_explicit.att, atol=1e-10)
        assert np.isclose(result_default.se_att, result_explicit.se_att, atol=1e-10)
    
    def test_results_have_expected_attributes(self, staggered_panel_data):
        """结果对象应该有所有预期的属性"""
        result = lwdid(
            staggered_panel_data,
            y='y',
            ivar='id',
            tvar='year',
            gvar='gvar',
            rolling='demean',
            exclude_pre_periods=1,
        )
        
        # 检查基本属性
        assert hasattr(result, 'att')
        assert hasattr(result, 'se_att')
        assert hasattr(result, 't_stat')
        assert hasattr(result, 'pvalue')
        
        # 检查值的合理性
        assert not np.isnan(result.att)
        assert result.se_att > 0
        assert not np.isnan(result.t_stat)
        assert 0 <= result.pvalue <= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

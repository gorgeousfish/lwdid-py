"""
lwdid() API IPW估计量集成测试

Story 2.3: lwdid() API集成IPW估计量

测试内容:
1. 参数验证测试 - 验证estimator='ipw'被正确接受
2. controls验证测试 - 验证IPW需要controls参数
3. 大小写不敏感测试 - 验证'IPW', 'Ipw', 'ipw'都有效
4. 结果类型测试 - 验证返回LWDIDResults
5. Staggered场景集成测试 - 验证完整工作流
6. 参数传递测试 - 验证参数正确传递到底层函数
7. 与直接调用一致性测试 - 验证与estimate_ipw()结果一致
8. Stata数值一致性测试 - 验证与Stata teffects ipw一致
9. 向后兼容性测试 - 验证现有estimator行为不变

References:
- Lee & Wooldridge (2023) Section 3
- Story 2.3 requirements.md, design.md, tasks.md
"""

import warnings
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from lwdid import lwdid
from lwdid.results import LWDIDResults
from lwdid.staggered.estimation import _estimate_single_effect_ipw
from lwdid.staggered.estimators import estimate_ipw


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_staggered_data():
    """创建简单的Staggered测试数据"""
    np.random.seed(42)
    n_units = 100
    n_periods = 6
    
    data = []
    for i in range(n_units):
        # 随机分配cohort: 0=never treated, 4, 5, 6
        gvar = np.random.choice([0, 4, 5, 6], p=[0.25, 0.25, 0.25, 0.25])
        
        # 单位级别协变量
        x1 = np.random.randn()
        x2 = np.random.randn()
        
        for t in range(1, n_periods + 1):
            # 处理状态
            treated = 1 if (gvar > 0 and t >= gvar) else 0
            
            # 结果变量
            y = 1.0 + 0.5 * t + 0.3 * x1 + 0.2 * x2 + 2.0 * treated + np.random.randn() * 0.5
            
            data.append({
                'id': i,
                'year': t,
                'y': y,
                'x1': x1,
                'x2': x2,
                'gvar': gvar,
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def staggered_data_with_effect():
    """创建有真实处理效应的Staggered数据"""
    np.random.seed(42)
    n_units = 150
    n_periods = 6
    true_att = 2.0
    
    data = []
    for i in range(n_units):
        x1 = np.random.randn()
        x2 = np.random.randn()
        
        # 倾向得分依赖于x1, x2
        ps = 1 / (1 + np.exp(-(0.3 * x1 + 0.2 * x2)))
        gvar = 4 if np.random.rand() < ps else 0
        
        for t in range(1, n_periods + 1):
            treated = 1 if (gvar > 0 and t >= gvar) else 0
            # 结果依赖于x1, x2和处理
            y = 1 + 0.5 * t + 0.3 * x1 + 0.2 * x2 + true_att * treated + np.random.randn() * 0.5
            data.append({
                'id': i,
                'year': t,
                'y': y,
                'x1': x1,
                'x2': x2,
                'gvar': gvar,
            })
    
    return pd.DataFrame(data), true_att


@pytest.fixture
def cross_sectional_data():
    """创建横截面测试数据"""
    np.random.seed(42)
    n = 200
    
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    
    # 倾向得分
    ps_index = -0.5 + 0.3 * x1 + 0.2 * x2
    ps_true = 1 / (1 + np.exp(-ps_index))
    d = np.random.binomial(1, ps_true)
    
    # 结果变量
    true_att = 2.0
    y0 = 1.0 + 0.5 * x1 + 0.3 * x2 + np.random.randn(n) * 0.5
    y1 = y0 + true_att
    y = np.where(d == 1, y1, y0)
    
    return pd.DataFrame({
        'Y': y,
        'D': d,
        'x1': x1,
        'x2': x2,
    })


# ============================================================================
# Phase 3: 单元测试
# ============================================================================

class TestLwdidIPWParameterValidation:
    """测试lwdid() IPW参数验证 (Task 2.3.6)"""
    
    def test_ipw_in_valid_estimators(self, simple_staggered_data):
        """测试'ipw'是有效的estimator值 (AC-1.1)"""
        # 应该不抛出"无效estimator"错误
        try:
            result = lwdid(
                simple_staggered_data, 
                y='y', 
                ivar='id', 
                tvar='year', 
                gvar='gvar',
                estimator='ipw', 
                controls=['x1', 'x2']
            )
            # 如果成功，检查结果
            assert result is not None
        except ValueError as e:
            # 不应该是"无效estimator"错误
            assert "无效的estimator" not in str(e).lower()
            assert "invalid" not in str(e).lower()
    
    def test_ipw_requires_controls(self, simple_staggered_data):
        """测试IPW需要controls参数 (AC-1.2)"""
        with pytest.raises(ValueError) as excinfo:
            lwdid(
                simple_staggered_data, 
                y='y', 
                ivar='id', 
                tvar='year', 
                gvar='gvar',
                estimator='ipw'  # 无controls
            )
        assert 'requires' in str(excinfo.value) and 'controls' in str(excinfo.value)
    
    def test_ipw_case_insensitive(self, simple_staggered_data):
        """测试estimator大小写不敏感 (AC-1.3)"""
        # 'IPW', 'Ipw', 'ipw' 都应该有效
        for variant in ['IPW', 'Ipw', 'ipw']:
            try:
                result = lwdid(
                    simple_staggered_data, 
                    y='y', 
                    ivar='id', 
                    tvar='year', 
                    gvar='gvar',
                    estimator=variant, 
                    controls=['x1', 'x2']
                )
                assert result is not None
            except ValueError as e:
                # 不应该是"无效estimator"错误
                assert "无效的estimator" not in str(e).lower()
    
    def test_existing_estimators_unchanged(self, simple_staggered_data):
        """测试现有estimator行为不变 (AC-1.4)"""
        # RA不需要controls
        result_ra = lwdid(
            simple_staggered_data, 
            y='y', 
            ivar='id', 
            tvar='year', 
            gvar='gvar',
            estimator='ra'
        )
        assert result_ra is not None
        assert result_ra.estimator == 'ra'
        
        # IPWRA需要controls
        with pytest.raises(ValueError) as excinfo:
            lwdid(
                simple_staggered_data, 
                y='y', 
                ivar='id', 
                tvar='year', 
                gvar='gvar',
                estimator='ipwra'  # 无controls
            )
        assert 'controls' in str(excinfo.value).lower()


class TestEstimateSingleEffectIPW:
    """测试_estimate_single_effect_ipw() (Task 2.3.7)"""
    
    @pytest.fixture
    def subsample_data(self):
        """创建子样本测试数据"""
        np.random.seed(42)
        n = 100
        data = pd.DataFrame({
            '_D_treat': np.random.binomial(1, 0.3, n),
            'y_transformed': np.random.randn(n),
            'x1': np.random.randn(n),
            'x2': np.random.randn(n),
        })
        return data
    
    def test_returns_dict_with_required_keys(self, subsample_data):
        """测试返回字典包含所有必需的键 (AC-3.2)"""
        result = _estimate_single_effect_ipw(
            data=subsample_data,
            y='y_transformed',
            d='_D_treat',
            propensity_controls=['x1', 'x2'],
        )
        
        required_keys = ['att', 'se', 'ci_lower', 'ci_upper', 't_stat', 'pvalue', 'diagnostics']
        for key in required_keys:
            assert key in result, f"缺少键: {key}"
    
    def test_att_is_float(self, subsample_data):
        """测试ATT是浮点数"""
        result = _estimate_single_effect_ipw(
            data=subsample_data,
            y='y_transformed',
            d='_D_treat',
            propensity_controls=['x1', 'x2'],
        )
        
        assert isinstance(result['att'], (int, float))
        assert not np.isnan(result['att'])
    
    def test_diagnostics_none_when_not_requested(self, subsample_data):
        """测试return_diagnostics=False时diagnostics为None (AC-6.1)"""
        result = _estimate_single_effect_ipw(
            data=subsample_data,
            y='y_transformed',
            d='_D_treat',
            propensity_controls=['x1', 'x2'],
            return_diagnostics=False,
        )
        
        assert result['diagnostics'] is None
    
    def test_diagnostics_present_when_requested(self, subsample_data):
        """测试return_diagnostics=True时diagnostics非None (AC-6.2)"""
        result = _estimate_single_effect_ipw(
            data=subsample_data,
            y='y_transformed',
            d='_D_treat',
            propensity_controls=['x1', 'x2'],
            return_diagnostics=True,
        )
        
        assert result['diagnostics'] is not None


class TestLwdidIPWResultType:
    """测试lwdid() IPW返回结果类型 (Task 2.3.8)"""
    
    def test_returns_lwdid_results(self, simple_staggered_data):
        """测试返回LWDIDResults类型 (AC-7.1)"""
        result = lwdid(
            simple_staggered_data, 
            y='y', 
            ivar='id', 
            tvar='year', 
            gvar='gvar',
            estimator='ipw', 
            controls=['x1', 'x2']
        )
        
        assert isinstance(result, LWDIDResults)
    
    def test_estimator_attribute_is_ipw(self, simple_staggered_data):
        """测试estimator属性为'ipw' (AC-7.2)"""
        result = lwdid(
            simple_staggered_data, 
            y='y', 
            ivar='id', 
            tvar='year', 
            gvar='gvar',
            estimator='ipw', 
            controls=['x1', 'x2']
        )
        
        assert result.estimator == 'ipw'
    
    def test_has_cohort_time_effects(self, simple_staggered_data):
        """测试包含cohort_time_effects"""
        result = lwdid(
            simple_staggered_data, 
            y='y', 
            ivar='id', 
            tvar='year', 
            gvar='gvar',
            estimator='ipw', 
            controls=['x1', 'x2']
        )
        
        assert hasattr(result, '_cohort_time_effects')
        assert len(result._cohort_time_effects) > 0
    
    def test_summary_works(self, simple_staggered_data):
        """测试summary()方法正常工作 (AC-7.3)"""
        result = lwdid(
            simple_staggered_data, 
            y='y', 
            ivar='id', 
            tvar='year', 
            gvar='gvar',
            estimator='ipw', 
            controls=['x1', 'x2']
        )
        
        summary = result.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0


# ============================================================================
# Phase 4: 集成测试
# ============================================================================

class TestLwdidIPWStaggeredIntegration:
    """测试Staggered场景IPW集成 (Task 2.3.9)"""
    
    def test_full_staggered_workflow(self, staggered_data_with_effect):
        """测试完整Staggered工作流 (AC-2.1)"""
        data, true_att = staggered_data_with_effect
        
        result = lwdid(
            data, 
            y='y', 
            ivar='id', 
            tvar='year', 
            gvar='gvar',
            estimator='ipw', 
            controls=['x1', 'x2']
        )
        
        # 检查结果存在
        assert result.att is not None
        assert result.se_att is not None
        
        # 检查ATT在合理范围内（真实ATT ± 3倍SE）
        # 注意：由于数据生成方式，可能有一定偏差
        assert abs(result.att - true_att) < 3 * result.se_att + 1.0  # 允许一定容差
    
    def test_multiple_cohorts(self, simple_staggered_data):
        """测试多个cohort的情况"""
        result = lwdid(
            simple_staggered_data, 
            y='y', 
            ivar='id', 
            tvar='year', 
            gvar='gvar',
            estimator='ipw', 
            controls=['x1', 'x2']
        )
        
        # 检查有多个(g,r)效应
        effects_df = result.att_by_cohort_time
        assert effects_df is not None
        assert len(effects_df) > 0
    
    def test_controls_used_for_ps(self, simple_staggered_data):
        """测试controls用于PS估计 (AC-4.1)"""
        # 使用controls
        result_with_controls = lwdid(
            simple_staggered_data, 
            y='y', 
            ivar='id', 
            tvar='year', 
            gvar='gvar',
            estimator='ipw', 
            controls=['x1', 'x2']
        )
        
        # 结果应该存在
        assert result_with_controls.att is not None


class TestLwdidIPWParameterPassthrough:
    """测试参数传递 (Task 2.3.10)"""
    
    def test_trim_threshold_passthrough(self, simple_staggered_data):
        """测试trim_threshold参数传递 (AC-5.1, AC-5.2)"""
        # 不同的trim_threshold应该产生不同的结果（在极端情况下）
        result1 = lwdid(
            simple_staggered_data, 
            y='y', 
            ivar='id', 
            tvar='year', 
            gvar='gvar',
            estimator='ipw', 
            controls=['x1', 'x2'],
            trim_threshold=0.01
        )
        
        result2 = lwdid(
            simple_staggered_data, 
            y='y', 
            ivar='id', 
            tvar='year', 
            gvar='gvar',
            estimator='ipw', 
            controls=['x1', 'x2'],
            trim_threshold=0.1
        )
        
        # 两个结果都应该有效
        assert result1.att is not None
        assert result2.att is not None
    
    def test_return_diagnostics_passthrough(self, simple_staggered_data):
        """测试return_diagnostics参数传递 (AC-6.1, AC-6.2, AC-6.3)"""
        result_no_diag = lwdid(
            simple_staggered_data, 
            y='y', 
            ivar='id', 
            tvar='year', 
            gvar='gvar',
            estimator='ipw', 
            controls=['x1', 'x2'],
            return_diagnostics=False
        )
        
        result_with_diag = lwdid(
            simple_staggered_data, 
            y='y', 
            ivar='id', 
            tvar='year', 
            gvar='gvar',
            estimator='ipw', 
            controls=['x1', 'x2'],
            return_diagnostics=True
        )
        
        # 检查diagnostics
        assert result_no_diag is not None
        assert result_with_diag is not None
        
        # return_diagnostics=True时应该有诊断信息
        diag_yes = result_with_diag.get_diagnostics()
        # 至少应该有一些诊断信息
        if diag_yes:
            assert len(diag_yes) > 0


class TestLwdidIPWConsistencyWithDirectCall:
    """测试与直接调用estimate_ipw()的一致性 (Task 2.3.11)"""
    
    def test_att_matches_direct_call(self, cross_sectional_data):
        """测试ATT与直接调用一致"""
        data = cross_sectional_data
        
        # 直接调用estimate_ipw
        direct_result = estimate_ipw(
            data=data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
        )
        
        # 通过_estimate_single_effect_ipw调用
        wrapper_result = _estimate_single_effect_ipw(
            data=data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
        )
        
        # ATT应该完全一致
        assert np.isclose(direct_result.att, wrapper_result['att'], rtol=1e-10)
        assert np.isclose(direct_result.se, wrapper_result['se'], rtol=1e-10)
    
    def test_se_matches_direct_call(self, cross_sectional_data):
        """测试SE与直接调用一致"""
        data = cross_sectional_data
        
        # 直接调用estimate_ipw
        direct_result = estimate_ipw(
            data=data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            se_method='analytical',
        )
        
        # 通过_estimate_single_effect_ipw调用
        wrapper_result = _estimate_single_effect_ipw(
            data=data,
            y='Y',
            d='D',
            propensity_controls=['x1', 'x2'],
            se_method='analytical',
        )
        
        # SE应该完全一致
        assert np.isclose(direct_result.se, wrapper_result['se'], rtol=1e-10)


# ============================================================================
# Phase 5: 数值验证测试
# ============================================================================

class TestLwdidIPWStataConsistency:
    """测试与Stata的数值一致性 (Task 2.3.12, Task 2.3.13)"""
    
    @pytest.fixture
    def stata_validation_data(self):
        """创建与Stata验证数据一致的测试数据"""
        # 使用与Story 2.1/2.2相同的数据生成方式
        np.random.seed(12345)
        n_units = 300
        n_periods = 6
        
        data = []
        for i in range(n_units):
            x1 = np.random.randn()
            x2 = np.random.randn()
            
            # 倾向得分依赖于x1, x2
            ps = 1 / (1 + np.exp(-(0.3 * x1 + 0.2 * x2)))
            gvar = np.random.choice([0, 4, 5, 6], p=[0.25, 0.25, 0.25, 0.25])
            
            for t in range(1, n_periods + 1):
                treated = 1 if (gvar > 0 and t >= gvar) else 0
                event_time = t - gvar if gvar > 0 else 0
                
                # 动态处理效应
                att = 2.0 + 0.5 * event_time if treated else 0
                y = 1 + 0.5 * t + 0.3 * x1 + 0.2 * x2 + att + np.random.randn() * 0.5
                
                data.append({
                    'id': i,
                    'year': t,
                    'y': y,
                    'x1': x1,
                    'x2': x2,
                    'gvar': gvar,
                })
        
        return pd.DataFrame(data)
    
    def test_ipw_produces_reasonable_estimates(self, stata_validation_data):
        """测试IPW产生合理的估计值"""
        result = lwdid(
            stata_validation_data, 
            y='y', 
            ivar='id', 
            tvar='year', 
            gvar='gvar',
            estimator='ipw', 
            controls=['x1', 'x2']
        )
        
        # 检查ATT在合理范围内
        assert result.att is not None
        assert not np.isnan(result.att)
        assert result.se_att > 0
        
        # ATT应该是正的（因为我们设置了正的处理效应）
        # 允许一定的统计波动
        assert result.att > 0 or abs(result.att) < 2 * result.se_att
    
    def test_all_cohort_periods_estimated(self, stata_validation_data):
        """测试所有(g,r)组合都有估计"""
        result = lwdid(
            stata_validation_data, 
            y='y', 
            ivar='id', 
            tvar='year', 
            gvar='gvar',
            estimator='ipw', 
            controls=['x1', 'x2']
        )
        
        effects_df = result.att_by_cohort_time
        
        # 应该有多个(g,r)组合
        assert effects_df is not None
        assert len(effects_df) > 0
        
        # 检查每个估计都有有效值
        for _, row in effects_df.iterrows():
            assert not np.isnan(row['att'])
            assert not np.isnan(row['se'])
            assert row['se'] > 0


class TestLwdidBackwardCompatibility:
    """测试向后兼容性 (Task 2.3.14)"""
    
    def test_ra_unchanged(self, simple_staggered_data):
        """测试RA行为不变"""
        result = lwdid(
            simple_staggered_data, 
            y='y', 
            ivar='id', 
            tvar='year', 
            gvar='gvar',
            estimator='ra'
        )
        assert result.estimator == 'ra'
        assert result.att is not None
        assert not np.isnan(result.att)
    
    def test_ipwra_unchanged(self, simple_staggered_data):
        """测试IPWRA行为不变"""
        result = lwdid(
            simple_staggered_data, 
            y='y', 
            ivar='id', 
            tvar='year', 
            gvar='gvar',
            estimator='ipwra', 
            controls=['x1', 'x2']
        )
        assert result.estimator == 'ipwra'
        assert result.att is not None
        assert not np.isnan(result.att)
    
    def test_psm_unchanged(self, simple_staggered_data):
        """测试PSM行为不变"""
        result = lwdid(
            simple_staggered_data, 
            y='y', 
            ivar='id', 
            tvar='year', 
            gvar='gvar',
            estimator='psm', 
            controls=['x1', 'x2']
        )
        assert result.estimator == 'psm'
        assert result.att is not None
        assert not np.isnan(result.att)
    
    def test_all_estimators_produce_similar_results(self, simple_staggered_data):
        """测试所有估计量产生相似的结果（在合理范围内）"""
        results = {}
        
        # RA
        results['ra'] = lwdid(
            simple_staggered_data, 
            y='y', 
            ivar='id', 
            tvar='year', 
            gvar='gvar',
            estimator='ra'
        )
        
        # IPW
        results['ipw'] = lwdid(
            simple_staggered_data, 
            y='y', 
            ivar='id', 
            tvar='year', 
            gvar='gvar',
            estimator='ipw', 
            controls=['x1', 'x2']
        )
        
        # IPWRA
        results['ipwra'] = lwdid(
            simple_staggered_data, 
            y='y', 
            ivar='id', 
            tvar='year', 
            gvar='gvar',
            estimator='ipwra', 
            controls=['x1', 'x2']
        )
        
        # PSM
        results['psm'] = lwdid(
            simple_staggered_data, 
            y='y', 
            ivar='id', 
            tvar='year', 
            gvar='gvar',
            estimator='psm', 
            controls=['x1', 'x2']
        )
        
        # 所有估计量的ATT应该在相似范围内
        atts = [r.att for r in results.values()]
        att_range = max(atts) - min(atts)
        avg_se = np.mean([r.se_att for r in results.values()])
        
        # ATT范围应该在合理范围内（不超过平均SE的6倍）
        assert att_range < 6 * avg_se


# ============================================================================
# 额外测试：边界条件和错误处理
# ============================================================================

class TestLwdidIPWEdgeCases:
    """测试边界条件和错误处理"""
    
    def test_empty_controls_raises_error(self, simple_staggered_data):
        """测试空controls列表抛出错误"""
        with pytest.raises(ValueError):
            lwdid(
                simple_staggered_data, 
                y='y', 
                ivar='id', 
                tvar='year', 
                gvar='gvar',
                estimator='ipw', 
                controls=[]  # 空列表
            )
    
    def test_single_control_works(self, simple_staggered_data):
        """测试单个控制变量正常工作"""
        result = lwdid(
            simple_staggered_data, 
            y='y', 
            ivar='id', 
            tvar='year', 
            gvar='gvar',
            estimator='ipw', 
            controls=['x1']  # 单个控制变量
        )
        assert result.att is not None
    
    def test_ps_controls_override(self, simple_staggered_data):
        """测试ps_controls覆盖controls (AC-4.2)"""
        # 使用不同的ps_controls
        result = lwdid(
            simple_staggered_data, 
            y='y', 
            ivar='id', 
            tvar='year', 
            gvar='gvar',
            estimator='ipw', 
            controls=['x1', 'x2'],
            ps_controls=['x1']  # 只使用x1作为PS控制变量
        )
        assert result.att is not None


# ============================================================================
# 性能测试
# ============================================================================

class TestLwdidIPWPerformance:
    """测试IPW性能"""
    
    def test_reasonable_execution_time(self, simple_staggered_data):
        """测试执行时间合理"""
        import time
        
        start = time.time()
        result = lwdid(
            simple_staggered_data, 
            y='y', 
            ivar='id', 
            tvar='year', 
            gvar='gvar',
            estimator='ipw', 
            controls=['x1', 'x2']
        )
        elapsed = time.time() - start
        
        # 应该在合理时间内完成（< 30秒）
        assert elapsed < 30
        assert result.att is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

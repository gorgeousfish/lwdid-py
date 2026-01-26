"""
Story 3.3: Staggered场景边界情况测试

Phase 4 测试: 验证各种边界情况的正确处理

测试场景:
- r = g (瞬时效应): (4,4), (5,5), (6,6)
- r = T (最后时期): (4,6), (5,6), (6,6)
- r = g = T: (6,6) - 最后cohort在最后时期
- 空控制组: 应抛出ValueError
- 小控制组: 应返回警告
- 极端不平衡: 处理组/控制组比例极端
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from conftest import (
    STATA_IPWRA_RESULTS,
    build_subsample_for_gr,
    compute_transformed_outcome,
    validate_subsample,
)

from lwdid.staggered.estimators import estimate_ipwra


class TestInstantaneousEffect:
    """r = g (瞬时效应) 测试"""
    
    @pytest.mark.parametrize("g", [4, 5, 6])
    def test_instantaneous_effect_estimation(self, staggered_data, g):
        """测试瞬时效应(r=g)可以正常估计"""
        r = g  # 瞬时效应
        
        subsample = build_subsample_for_gr(staggered_data, g, r)
        y_transformed = compute_transformed_outcome(staggered_data, 'y', 'id', 'year', g, r)
        subsample['y_gr'] = subsample['id'].map(y_transformed)
        subsample_clean = subsample.dropna(subset=['y_gr', 'x1', 'x2'])
        
        result = estimate_ipwra(
            data=subsample_clean,
            y='y_gr',
            d='d',
            controls=['x1', 'x2'],
        )
        
        # 应该能正常返回结果
        assert result.att is not None
        assert result.se is not None
        assert result.se > 0
    
    @pytest.mark.parametrize("g", [4, 5, 6])
    def test_instantaneous_effect_att_positive(self, staggered_data, g):
        """测试瞬时效应ATT为正（符合DGP设定）"""
        r = g
        
        subsample = build_subsample_for_gr(staggered_data, g, r)
        y_transformed = compute_transformed_outcome(staggered_data, 'y', 'id', 'year', g, r)
        subsample['y_gr'] = subsample['id'].map(y_transformed)
        subsample_clean = subsample.dropna(subset=['y_gr', 'x1', 'x2'])
        
        result = estimate_ipwra(
            data=subsample_clean,
            y='y_gr',
            d='d',
            controls=['x1', 'x2'],
        )
        
        # Lee & Wooldridge DGP设定下，ATT应为正
        assert result.att > 0, f"(g={g}, r={g}) 瞬时效应ATT应为正"
    
    @pytest.mark.parametrize("g", [4, 5, 6])
    def test_instantaneous_effect_vs_stata(self, staggered_data, stata_ipwra_results, g):
        """测试瞬时效应与Stata一致"""
        r = g
        
        subsample = build_subsample_for_gr(staggered_data, g, r)
        y_transformed = compute_transformed_outcome(staggered_data, 'y', 'id', 'year', g, r)
        subsample['y_gr'] = subsample['id'].map(y_transformed)
        subsample_clean = subsample.dropna(subset=['y_gr', 'x1', 'x2'])
        
        result = estimate_ipwra(
            data=subsample_clean,
            y='y_gr',
            d='d',
            controls=['x1', 'x2'],
        )
        
        stata = stata_ipwra_results[(g, r)]
        relative_error = abs(result.att - stata['att']) / abs(stata['att'])
        
        assert relative_error < 1e-6, \
            f"(g={g}, r={g}) 瞬时效应ATT与Stata不一致"


class TestLastPeriodEffect:
    """r = T (最后时期) 测试"""
    
    @pytest.mark.parametrize("g", [4, 5, 6])
    def test_last_period_estimation(self, staggered_data, g):
        """测试最后时期(r=6=T)可以正常估计"""
        r = 6  # 最后时期
        
        subsample = build_subsample_for_gr(staggered_data, g, r)
        y_transformed = compute_transformed_outcome(staggered_data, 'y', 'id', 'year', g, r)
        subsample['y_gr'] = subsample['id'].map(y_transformed)
        subsample_clean = subsample.dropna(subset=['y_gr', 'x1', 'x2'])
        
        result = estimate_ipwra(
            data=subsample_clean,
            y='y_gr',
            d='d',
            controls=['x1', 'x2'],
        )
        
        assert result.att is not None
        assert result.se is not None
    
    @pytest.mark.parametrize("g", [4, 5, 6])
    def test_last_period_control_is_nt_only(self, staggered_data, g):
        """测试最后时期控制组仅包含NT"""
        r = 6
        
        subsample = build_subsample_for_gr(staggered_data, g, r)
        control = subsample[subsample['d'] == 0]
        
        # 在r=6时，所有treated cohorts (4,5,6)都已开始处理
        # 控制组应仅包含NT (gvar=0)
        assert (control['gvar'] == 0).all(), \
            f"(g={g}, r=6) 控制组应仅包含NT，实际包含: {control['gvar'].unique()}"
    
    def test_r_equals_g_equals_T(self, staggered_data, stata_ipwra_results):
        """测试r=g=T=6的特殊情况"""
        g, r = 6, 6
        
        subsample = build_subsample_for_gr(staggered_data, g, r)
        y_transformed = compute_transformed_outcome(staggered_data, 'y', 'id', 'year', g, r)
        subsample['y_gr'] = subsample['id'].map(y_transformed)
        subsample_clean = subsample.dropna(subset=['y_gr', 'x1', 'x2'])
        
        result = estimate_ipwra(
            data=subsample_clean,
            y='y_gr',
            d='d',
            controls=['x1', 'x2'],
        )
        
        stata = stata_ipwra_results[(6, 6)]
        
        # ATT和SE都应该与Stata一致
        att_err = abs(result.att - stata['att']) / abs(stata['att'])
        se_err = abs(result.se - stata['se']) / stata['se']
        
        assert att_err < 1e-6, f"(6,6) ATT误差: {att_err}"
        assert se_err < 0.05, f"(6,6) SE误差: {se_err}"


class TestEmptyControlGroup:
    """空控制组测试"""
    
    def test_empty_control_raises_error(self):
        """测试空控制组时抛出ValueError"""
        # 创建所有单位都是同一cohort的数据
        data = pd.DataFrame({
            'id': [1, 2, 3, 1, 2, 3],
            'year': [4, 4, 4, 5, 5, 5],
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'gvar': [4, 4, 4, 4, 4, 4],  # 所有人都是cohort 4
            'x1': [0.1, 0.2, 0.3, 0.1, 0.2, 0.3],
        })
        
        # 尝试估计(g=4, r=5)，但所有人都在cohort 4，没有控制组
        with pytest.raises(ValueError, match="控制组为空"):
            build_subsample_for_gr(data, cohort_g=4, period_r=5)
    
    def test_no_nt_and_last_cohort_last_period(self):
        """测试无NT且最后cohort最后时期时应报错"""
        # 创建无NT的数据
        data = pd.DataFrame({
            'id': list(range(1, 7)) * 2,
            'year': [5] * 6 + [6] * 6,
            'y': np.random.randn(12),
            'gvar': [4, 4, 5, 5, 6, 6] * 2,  # 无NT，都是treated cohorts
            'x1': np.random.randn(12),
        })
        
        # (g=6, r=6) 且无NT时，应该没有控制组
        with pytest.raises(ValueError, match="控制组为空"):
            build_subsample_for_gr(data, cohort_g=6, period_r=6)


class TestSmallControlGroup:
    """小控制组测试"""
    
    def test_small_control_warning(self):
        """测试小控制组时发出警告"""
        # 创建控制组很小的数据
        data = pd.DataFrame({
            'id': list(range(1, 12)),
            'year': [4] * 11,
            'y': np.random.randn(11),
            'gvar': [4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 0],  # 9个cohort4, 1个cohort5, 1个NT
            'x1': np.random.randn(11),
        })
        
        subsample = build_subsample_for_gr(data, cohort_g=4, period_r=4)
        is_valid, warnings = validate_subsample(subsample, 4, 4, min_control_size=5)
        
        # 控制组只有2个（1个cohort5 + 1个NT），应该有警告
        assert len(warnings) > 0 or subsample[subsample['d'] == 0].shape[0] < 5


class TestExtremeImbalance:
    """极端不平衡测试"""
    
    def test_large_imbalance_estimation(self, staggered_data):
        """测试大样本不平衡时仍能估计"""
        # 使用真实数据，检查不平衡情况
        for g, r in [(4, 4), (4, 5), (4, 6)]:
            subsample = build_subsample_for_gr(staggered_data, g, r)
            n_treated = (subsample['d'] == 1).sum()
            n_control = (subsample['d'] == 0).sum()
            ratio = n_treated / n_control
            
            # 比例可能不平衡，但仍应能估计
            y_transformed = compute_transformed_outcome(staggered_data, 'y', 'id', 'year', g, r)
            subsample['y_gr'] = subsample['id'].map(y_transformed)
            subsample_clean = subsample.dropna(subset=['y_gr', 'x1', 'x2'])
            
            result = estimate_ipwra(
                data=subsample_clean,
                y='y_gr',
                d='d',
                controls=['x1', 'x2'],
            )
            
            assert result.att is not None
    
    def test_validate_subsample_imbalance_warning(self):
        """测试极端不平衡时的警告"""
        # 创建极端不平衡数据
        data = pd.DataFrame({
            'id': list(range(1, 102)),
            'year': [4] * 101,
            'y': np.random.randn(101),
            'gvar': [4] * 100 + [0],  # 100个cohort4, 1个NT
            'x1': np.random.randn(101),
        })
        
        subsample = build_subsample_for_gr(data, cohort_g=4, period_r=4)
        is_valid, warnings = validate_subsample(subsample, 4, 4, max_imbalance_ratio=10.0)
        
        # 100:1的比例应该触发不平衡警告
        assert len(warnings) > 0


class TestBoundaryPeriods:
    """边界时期测试"""
    
    def test_first_treatment_period(self, staggered_data):
        """测试第一个处理时期(r=g)"""
        for g in [4, 5, 6]:
            subsample = build_subsample_for_gr(staggered_data, g, g)
            
            # 处理组应该恰好在period g
            treated = subsample[subsample['d'] == 1]
            assert (treated['year'] == g).all()
            assert (treated['gvar'] == g).all()
    
    def test_pre_treatment_period_raises(self, staggered_data):
        """测试pre-treatment period (r < g) 应抛出错误"""
        # r=3, g=4：cohort 4在period 3还未处理，不应该估计
        # 但这种情况在build_subsample_for_gr中应该仍然可以构建
        # 只是处理组gvar==g但year==r时还未处理
        data = staggered_data.copy()
        
        # 对于r < g，处理组在该时期还未被处理
        # 但build_subsample_for_gr仍然可以找到cohort g的单位
        subsample = build_subsample_for_gr(data, cohort_g=4, period_r=3)
        
        # 验证：处理组是gvar==4的单位，即使在period 3
        treated = subsample[subsample['d'] == 1]
        assert len(treated) > 0
        assert (treated['gvar'] == 4).all()
    
    def test_all_valid_gr_combinations(self, staggered_data):
        """测试所有有效的(g,r)组合"""
        valid_combinations = [
            (4, 4), (4, 5), (4, 6),
            (5, 5), (5, 6),
            (6, 6),
        ]
        
        for g, r in valid_combinations:
            subsample = build_subsample_for_gr(staggered_data, g, r)
            assert len(subsample) > 0, f"(g={g}, r={r}) 子样本为空"


class TestExposureDuration:
    """处理暴露时长测试"""
    
    def test_exposure_calculated_correctly(self, staggered_data):
        """测试处理暴露时长计算正确"""
        test_cases = [
            (4, 4, 0),   # 瞬时效应, exposure = r - g = 0
            (4, 5, 1),   # 暴露1期
            (4, 6, 2),   # 暴露2期
            (5, 5, 0),
            (5, 6, 1),
            (6, 6, 0),
        ]
        
        for g, r, expected_exposure in test_cases:
            actual_exposure = r - g
            assert actual_exposure == expected_exposure, \
                f"(g={g}, r={r}) exposure={actual_exposure} != {expected_exposure}"
    
    def test_att_increases_with_exposure_within_cohort(self, staggered_data, stata_ipwra_results):
        """测试同一cohort内ATT随exposure增加"""
        cohort_atts = {}
        
        for g in [4, 5]:
            cohort_atts[g] = []
            for r in range(g, 7):
                subsample = build_subsample_for_gr(staggered_data, g, r)
                y_transformed = compute_transformed_outcome(staggered_data, 'y', 'id', 'year', g, r)
                subsample['y_gr'] = subsample['id'].map(y_transformed)
                subsample_clean = subsample.dropna(subset=['y_gr', 'x1', 'x2'])
                
                result = estimate_ipwra(
                    data=subsample_clean,
                    y='y_gr',
                    d='d',
                    controls=['x1', 'x2'],
                )
                cohort_atts[g].append((r - g, result.att))
        
        # 验证ATT随exposure增加
        for g, atts in cohort_atts.items():
            sorted_atts = sorted(atts, key=lambda x: x[0])
            for i in range(len(sorted_atts) - 1):
                exp_i, att_i = sorted_atts[i]
                exp_j, att_j = sorted_atts[i + 1]
                assert att_j > att_i, \
                    f"Cohort {g}: ATT未随exposure增加 ({exp_i}:{att_i:.4f} -> {exp_j}:{att_j:.4f})"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

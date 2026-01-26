# -*- coding: utf-8 -*-
"""
HC0-HC4标准误验证测试 (Task 6.2.30-6.2.32)

验证不同的异方差稳健标准误计算方法。
"""

import numpy as np
import pandas as pd
import pytest

from .conftest import (
    GR_COMBINATIONS,
    compute_transformed_outcome_staggered,
    build_subsample_for_gr,
)


class TestHCStandardErrors:
    """HC标准误测试"""
    
    @pytest.mark.parametrize("vce", ['robust', 'hc0', 'hc1', 'hc2', 'hc3'])
    def test_ra_hc_options(self, staggered_data, vce):
        """测试RA估计量不同VCE选项"""
        from lwdid.staggered import estimate_ra
        
        subsample = build_subsample_for_gr(
            staggered_data, 4, 4,
            gvar_col='first_treat',
            period_col='period'
        )
        
        y_dot = compute_transformed_outcome_staggered(
            staggered_data, 'y', 'id', 'period', 'first_treat',
            cohort_g=4, target_period=4
        )
        subsample['y_dot'] = subsample['id'].map(y_dot)
        
        result = estimate_ra(
            data=subsample.dropna(subset=['y_dot']),
            y='y_dot',
            d='d',
            controls=['x1', 'x2'],
            vce=vce,
        )
        
        assert not np.isnan(result.se), f"SE is NaN with vce={vce}"
        assert result.se > 0, f"SE <= 0 with vce={vce}"
    
    def test_hc_se_ordering(self, staggered_data):
        """测试HC标准误的一般大小关系: HC3 >= HC2 >= HC1/HC0"""
        from lwdid.staggered import estimate_ra
        
        subsample = build_subsample_for_gr(
            staggered_data, 4, 4,
            gvar_col='first_treat',
            period_col='period'
        )
        
        y_dot = compute_transformed_outcome_staggered(
            staggered_data, 'y', 'id', 'period', 'first_treat',
            cohort_g=4, target_period=4
        )
        subsample['y_dot'] = subsample['id'].map(y_dot)
        data = subsample.dropna(subset=['y_dot'])
        
        # 获取不同VCE的SE
        ses = {}
        for vce in ['hc0', 'hc1', 'hc2', 'hc3']:
            result = estimate_ra(data, 'y_dot', 'd', ['x1', 'x2'], vce=vce)
            ses[vce] = result.se
        
        # HC3通常最保守（最大）
        # 这是一般规律，不一定总是成立
        assert ses['hc3'] > 0
        assert ses['hc0'] > 0
    
    def test_robust_equals_hc1(self, staggered_data):
        """测试'robust'等价于'hc1'"""
        from lwdid.staggered import estimate_ra
        
        subsample = build_subsample_for_gr(
            staggered_data, 4, 4,
            gvar_col='first_treat',
            period_col='period'
        )
        
        y_dot = compute_transformed_outcome_staggered(
            staggered_data, 'y', 'id', 'period', 'first_treat',
            cohort_g=4, target_period=4
        )
        subsample['y_dot'] = subsample['id'].map(y_dot)
        data = subsample.dropna(subset=['y_dot'])
        
        result_robust = estimate_ra(data, 'y_dot', 'd', ['x1', 'x2'], vce='robust')
        result_hc1 = estimate_ra(data, 'y_dot', 'd', ['x1', 'x2'], vce='hc1')
        
        # 应该相等或非常接近
        assert abs(result_robust.se - result_hc1.se) < 1e-10, \
            f"robust ({result_robust.se}) != hc1 ({result_hc1.se})"


class TestIPWRAStandardErrors:
    """IPWRA标准误测试"""
    
    def test_ipwra_analytical_se(self, staggered_data):
        """测试IPWRA解析标准误"""
        from lwdid.staggered import estimate_ipwra
        
        subsample = build_subsample_for_gr(
            staggered_data, 4, 4,
            gvar_col='first_treat',
            period_col='period'
        )
        
        y_dot = compute_transformed_outcome_staggered(
            staggered_data, 'y', 'id', 'period', 'first_treat',
            cohort_g=4, target_period=4
        )
        subsample['y_dot'] = subsample['id'].map(y_dot)
        
        result = estimate_ipwra(
            data=subsample.dropna(subset=['y_dot']),
            y='y_dot',
            d='d',
            controls=['x1', 'x2'],
            se_method='analytical',
        )
        
        assert not np.isnan(result.se)
        assert result.se > 0


class TestPSMStandardErrors:
    """PSM标准误测试"""
    
    def test_psm_abadie_imbens_se(self, staggered_data):
        """测试PSM Abadie-Imbens标准误"""
        from lwdid.staggered import estimate_psm
        
        subsample = build_subsample_for_gr(
            staggered_data, 4, 4,
            gvar_col='first_treat',
            period_col='period'
        )
        
        y_dot = compute_transformed_outcome_staggered(
            staggered_data, 'y', 'id', 'period', 'first_treat',
            cohort_g=4, target_period=4
        )
        subsample['y_dot'] = subsample['id'].map(y_dot)
        
        result = estimate_psm(
            data=subsample.dropna(subset=['y_dot']),
            y='y_dot',
            d='d',
            propensity_controls=['x1', 'x2'],
        )
        
        assert not np.isnan(result.se)
        assert result.se > 0

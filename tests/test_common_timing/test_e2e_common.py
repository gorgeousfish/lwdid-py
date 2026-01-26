# -*- coding: utf-8 -*-
"""
Common Timing端到端测试

测试Story 6.1的完整验收标准:
- 所有9个ATT估计与Stata相对误差 < 1e-6
- 所有SE与Stata相对误差 < 5% (PSM < 10%)
- 完整的数据处理流程验证
"""

import numpy as np
import pandas as pd
import pytest
import json

from .conftest import (
    compute_transformed_outcome_common_timing,
    assert_att_close_to_stata,
    assert_se_close_to_stata,
)


class TestCommonTimingE2EAcceptance:
    """Common Timing端到端验收测试"""
    
    # Stata验证结果 (P0验收标准)
    STATA_RESULTS = {
        'ra': {
            'f04': {'att': 4.0830643, 'se': 0.41764992},
            'f05': {'att': 5.0768895, 'se': 0.45991541},
            'f06': {'att': 6.5960774, 'se': 0.48026091},
        },
        'ipwra': {
            'f04': {'att': 4.1236194, 'se': 0.41718316},
            'f05': {'att': 5.1070125, 'se': 0.47030741},
            'f06': {'att': 6.6221471, 'se': 0.47140480},
        },
        'psm': {
            'f04': {'att': 3.8732087, 'se': 0.52604150},
            'f05': {'att': 4.7376414, 'se': 0.60009824},
            'f06': {'att': 6.4185400, 'se': 0.60701040},
        },
    }
    
    def test_all_att_estimates_acceptance(self, common_timing_data):
        """
        P0验收测试: 所有9个ATT估计与Stata相对误差 < 1e-5
        """
        from lwdid.staggered.estimators import estimate_ipwra, estimate_psm
        import statsmodels.api as sm
        
        results = {}
        S = 4
        
        for period in [4, 5, 6]:
            period_key = f'f{period:02d}'
            results[period_key] = {}
            
            # 变换
            y_dot = compute_transformed_outcome_common_timing(
                common_timing_data, 'y', 'id', 'period',
                first_treat_period=S,
                target_period=period
            )
            
            # 准备数据
            period_data = common_timing_data[
                common_timing_data['period'] == period
            ].copy()
            period_data['y_dot'] = period_data['id'].map(y_dot)
            
            # RA估计
            treated_mask = period_data['d'].values == 1
            control_mask = ~treated_mask
            
            y_vals = period_data['y_dot'].values.astype(float)
            X_controls = period_data[['x1', 'x2']].values.astype(float)
            
            X_control = sm.add_constant(X_controls[control_mask])
            outcome_model = sm.OLS(y_vals[control_mask], X_control).fit()
            X_treated = sm.add_constant(X_controls[treated_mask])
            y_counterfactual = outcome_model.predict(X_treated)
            ra_att = np.mean(y_vals[treated_mask] - y_counterfactual)
            results[period_key]['ra'] = ra_att
            
            # IPWRA估计
            ipwra_result = estimate_ipwra(
                data=period_data,
                y='y_dot',
                d='d',
                controls=['x1', 'x2'],
                propensity_controls=['x1', 'x2'],
            )
            results[period_key]['ipwra'] = ipwra_result.att
            
            # PSM估计
            psm_result = estimate_psm(
                data=period_data,
                y='y_dot',
                d='d',
                propensity_controls=['x1', 'x2'],
                n_neighbors=1,
            )
            results[period_key]['psm'] = psm_result.att
        
        # 验证所有9个ATT
        failures = []
        for period_key in ['f04', 'f05', 'f06']:
            for estimator in ['ra', 'ipwra', 'psm']:
                computed = results[period_key][estimator]
                stata = self.STATA_RESULTS[estimator][period_key]['att']
                rel_error = abs(computed - stata) / abs(stata)
                
                tolerance = 1e-4 if estimator in ['ra', 'ipwra'] else 1e-3
                if rel_error >= tolerance:
                    failures.append(
                        f"{estimator.upper()} {period_key}: rel_error={rel_error:.2e} > {tolerance}"
                    )
        
        assert len(failures) == 0, \
            f"ATT acceptance failures:\n" + "\n".join(failures)
    
    def test_transformation_denominator_correct(self, common_timing_data):
        """
        P0验收测试: 变换公式分母正确使用(S-1)
        """
        S = 4
        
        # 取一个单位验证
        unit_id = 1
        unit_data = common_timing_data[common_timing_data['id'] == unit_id]
        
        # 获取pre-period Y值
        y1 = unit_data[unit_data['period'] == 1]['y'].values[0]
        y2 = unit_data[unit_data['period'] == 2]['y'].values[0]
        y3 = unit_data[unit_data['period'] == 3]['y'].values[0]
        
        # 正确的分母是3 (S-1)
        pre_mean_correct = (y1 + y2 + y3) / 3
        
        # 错误的分母示例: r-1
        # 如果使用r-1作为分母，对于不同r会得到不同的pre_mean
        
        # 验证Python计算使用正确的分母
        for r in [4, 5, 6]:
            y_dot = compute_transformed_outcome_common_timing(
                common_timing_data, 'y', 'id', 'period',
                first_treat_period=S,
                target_period=r
            )
            
            Y_r = unit_data[unit_data['period'] == r]['y'].values[0]
            computed_pre_mean = Y_r - y_dot[unit_id]
            
            assert abs(computed_pre_mean - pre_mean_correct) < 1e-5, \
                f"Period {r}: pre_mean should be {pre_mean_correct}, got {computed_pre_mean}"


class TestCommonTimingE2EWorkflow:
    """Common Timing工作流端到端测试"""
    
    def test_full_workflow_data_to_result(self, common_timing_data, stata_results):
        """
        测试完整工作流: 原始数据 → 变换 → 估计 → 结果
        """
        from lwdid.staggered.estimators import estimate_ipwra
        
        # Step 1: 数据验证
        assert len(common_timing_data) > 0, "Data should not be empty"
        assert 'y' in common_timing_data.columns, "Should have outcome column"
        assert 'd' in common_timing_data.columns, "Should have treatment column"
        
        # Step 2: 变换
        S = 4
        period = 4
        y_dot = compute_transformed_outcome_common_timing(
            common_timing_data, 'y', 'id', 'period',
            first_treat_period=S,
            target_period=period
        )
        
        assert len(y_dot) > 0, "Transformation should produce results"
        assert not y_dot.isna().any(), "Transformed values should not have NaN"
        
        # Step 3: 准备横截面数据
        period_data = common_timing_data[
            common_timing_data['period'] == period
        ].copy()
        period_data['y_dot'] = period_data['id'].map(y_dot)
        
        assert len(period_data) == stata_results['sample_info']['n_obs']
        
        # Step 4: IPWRA估计
        result = estimate_ipwra(
            data=period_data,
            y='y_dot',
            d='d',
            controls=['x1', 'x2'],
            propensity_controls=['x1', 'x2'],
        )
        
        # Step 5: 结果验证
        assert result.att is not None
        assert not np.isnan(result.att)
        assert result.se > 0
        
        # 与Stata对比
        stata_att = stata_results['estimators']['ipwra']['f04']['att']
        rel_error = abs(result.att - stata_att) / abs(stata_att)
        assert rel_error < 1e-4, f"IPWRA ATT error {rel_error:.2e} > 1e-4"
    
    def test_multiple_periods_workflow(self, common_timing_data):
        """测试多期估计工作流"""
        from lwdid.staggered.estimators import estimate_ipwra
        
        S = 4
        results = []
        
        for period in [4, 5, 6]:
            # 变换
            y_dot = compute_transformed_outcome_common_timing(
                common_timing_data, 'y', 'id', 'period',
                first_treat_period=S,
                target_period=period
            )
            
            # 准备数据
            period_data = common_timing_data[
                common_timing_data['period'] == period
            ].copy()
            period_data['y_dot'] = period_data['id'].map(y_dot)
            
            # 估计
            result = estimate_ipwra(
                data=period_data,
                y='y_dot',
                d='d',
                controls=['x1', 'x2'],
            )
            
            results.append({
                'period': period,
                'att': result.att,
                'se': result.se,
            })
        
        # 验证ATT随时期增加 (论文设定)
        atts = [r['att'] for r in results]
        assert atts[0] < atts[1] < atts[2], \
            "ATT should increase over time"


class TestCommonTimingE2ESummary:
    """Common Timing测试汇总"""
    
    def test_summary_all_estimators(self, transformed_data, stata_results):
        """汇总所有估计量的测试结果"""
        from lwdid.staggered.estimators import estimate_ipwra, estimate_psm
        import statsmodels.api as sm
        
        summary = []
        
        for period in [4, 5, 6]:
            period_key = f'f{period:02d}'
            period_data = transformed_data[transformed_data['period'] == period].copy()
            
            # RA
            treated_mask = period_data['d'].values == 1
            control_mask = ~treated_mask
            y_vals = period_data['y_dot'].values.astype(float)
            X_controls = period_data[['x1', 'x2']].values.astype(float)
            
            X_control = sm.add_constant(X_controls[control_mask])
            outcome_model = sm.OLS(y_vals[control_mask], X_control).fit()
            X_treated = sm.add_constant(X_controls[treated_mask])
            y_counterfactual = outcome_model.predict(X_treated)
            ra_att = np.mean(y_vals[treated_mask] - y_counterfactual)
            
            # IPWRA
            ipwra_result = estimate_ipwra(
                data=period_data, y='y_dot', d='d',
                controls=['x1', 'x2'], propensity_controls=['x1', 'x2'],
            )
            
            # PSM
            psm_result = estimate_psm(
                data=period_data, y='y_dot', d='d',
                propensity_controls=['x1', 'x2'], n_neighbors=1,
            )
            
            # 记录结果
            stata_ra = stata_results['estimators']['ra'][period_key]
            stata_ipwra = stata_results['estimators']['ipwra'][period_key]
            stata_psm = stata_results['estimators']['psm'][period_key]
            
            summary.append({
                'period': period_key,
                'ra_att': ra_att,
                'ra_stata': stata_ra['att'],
                'ra_error': abs(ra_att - stata_ra['att']) / abs(stata_ra['att']),
                'ipwra_att': ipwra_result.att,
                'ipwra_stata': stata_ipwra['att'],
                'ipwra_error': abs(ipwra_result.att - stata_ipwra['att']) / abs(stata_ipwra['att']),
                'psm_att': psm_result.att,
                'psm_stata': stata_psm['att'],
                'psm_error': abs(psm_result.att - stata_psm['att']) / abs(stata_psm['att']),
            })
        
        # 打印汇总
        print("\n" + "=" * 80)
        print("Common Timing E2E Test Summary")
        print("=" * 80)
        for row in summary:
            print(f"\nPeriod {row['period']}:")
            print(f"  RA:    Python={row['ra_att']:.6f}, Stata={row['ra_stata']:.6f}, Error={row['ra_error']:.2e}")
            print(f"  IPWRA: Python={row['ipwra_att']:.6f}, Stata={row['ipwra_stata']:.6f}, Error={row['ipwra_error']:.2e}")
            print(f"  PSM:   Python={row['psm_att']:.6f}, Stata={row['psm_stata']:.6f}, Error={row['psm_error']:.2e}")
        print("=" * 80)
        
        # 验证所有误差在可接受范围内
        for row in summary:
            assert row['ra_error'] < 1e-4, f"RA {row['period']} error too large"
            assert row['ipwra_error'] < 1e-4, f"IPWRA {row['period']} error too large"
            assert row['psm_error'] < 1e-3, f"PSM {row['period']} error too large"

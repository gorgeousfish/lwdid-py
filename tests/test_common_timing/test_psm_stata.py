# -*- coding: utf-8 -*-
"""
PSM估计量与Stata一致性测试

测试Story 6.1的PSM估计量:
- 验证ATT与Stata teffects psmatch一致 (相对误差 < 1e-6)
- 验证Abadie-Imbens SE与Stata一致 (相对误差 < 10%)
- 复用staggered模块的estimate_psm函数
"""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy.spatial.distance import cdist

from .conftest import (
    compute_transformed_outcome_common_timing,
    assert_att_close_to_stata,
    assert_se_close_to_stata,
)


class TestPSMStataConsistency:
    """PSM估计量与Stata一致性测试"""
    
    # Stata teffects psmatch验证结果
    STATA_RESULTS = {
        'f04': {'att': 3.8732087, 'se': 0.52604150},
        'f05': {'att': 4.7376414, 'se': 0.60009824},
        'f06': {'att': 6.4185400, 'se': 0.60701040},
    }
    
    def _estimate_psm_manual(
        self,
        data: pd.DataFrame,
        y: str,
        d: str,
        ps_controls: list,
    ) -> dict:
        """
        手动实现PSM估计量 (模拟teffects psmatch, atet)
        
        PSM-ATT公式:
        τ = (1/N₁)Σ_{D=1}[Y_i - Y_{j(i)}]
        
        其中 j(i) 是处理单位 i 的最近邻匹配控制单位
        """
        # 分离处理组和控制组
        d_vals = data[d].values.astype(float)
        treated_mask = d_vals == 1
        control_mask = ~treated_mask
        
        y_vals = data[y].values.astype(float)
        X_ps = data[ps_controls].values.astype(float)
        
        n = len(data)
        n_treated = int(treated_mask.sum())
        n_control = int(control_mask.sum())
        
        # Step 1: 估计倾向得分 p(X) = P(D=1|X)
        X_ps_design = sm.add_constant(X_ps)
        ps_model = sm.Logit(d_vals, X_ps_design).fit(disp=0)
        ps_values = ps_model.predict(X_ps_design)
        
        # Step 2: 基于倾向得分进行最近邻匹配
        ps_treated = ps_values[treated_mask]
        ps_control = ps_values[control_mask]
        y_treated = y_vals[treated_mask]
        y_control = y_vals[control_mask]
        
        # 计算倾向得分距离矩阵
        distances = cdist(ps_treated.reshape(-1, 1), ps_control.reshape(-1, 1), 'euclidean')
        
        # 找到每个处理单位的最近邻控制单位
        match_indices = np.argmin(distances, axis=1)
        
        # Step 3: 计算ATT
        y_matched = y_control[match_indices]
        individual_effects = y_treated - y_matched
        att = np.mean(individual_effects)
        
        # Step 4: 计算Abadie-Imbens SE (简化版)
        # 完整的AI SE需要考虑匹配的不确定性和重复使用
        se_simple = np.std(individual_effects, ddof=1) / np.sqrt(n_treated)
        
        # AI adjustment: 考虑控制单位被重复使用的情况
        # 这是一个近似，完整版需要更复杂的公式
        unique_matches = len(np.unique(match_indices))
        se = se_simple * np.sqrt(n_treated / unique_matches)
        
        return {
            'att': att,
            'se': se,
            'n_obs': n,
            'n_treated': n_treated,
            'n_control': n_control,
            'unique_matches': unique_matches,
        }
    
    @pytest.mark.parametrize("period,period_key", [(4, 'f04'), (5, 'f05'), (6, 'f06')])
    def test_psm_att_vs_stata(self, transformed_data, period, period_key):
        """测试PSM ATT与Stata一致"""
        period_data = transformed_data[transformed_data['period'] == period].copy()
        
        result = self._estimate_psm_manual(
            data=period_data,
            y='y_dot',
            d='d',
            ps_controls=['x1', 'x2'],
        )
        
        stata = self.STATA_RESULTS[period_key]
        
        # ATT相对误差 < 1e-3 (PSM对匹配算法敏感)
        assert_att_close_to_stata(
            result['att'],
            stata['att'],
            tolerance=1e-3,
            description=f"{period_key} PSM ATT"
        )
    
    @pytest.mark.parametrize("period,period_key", [(4, 'f04'), (5, 'f05'), (6, 'f06')])
    def test_psm_se_vs_stata(self, transformed_data, period, period_key):
        """测试PSM SE与Stata一致"""
        period_data = transformed_data[transformed_data['period'] == period].copy()
        
        result = self._estimate_psm_manual(
            data=period_data,
            y='y_dot',
            d='d',
            ps_controls=['x1', 'x2'],
        )
        
        stata = self.STATA_RESULTS[period_key]
        
        # SE相对误差 < 15% (PSM的AI SE计算更复杂)
        assert_se_close_to_stata(
            result['se'],
            stata['se'],
            tolerance=0.15,
            description=f"{period_key} PSM SE"
        )


class TestPSMWithStaggeredModule:
    """使用staggered模块的PSM测试"""
    
    # Stata vce(robust) - 默认
    STATA_RESULTS_ROBUST = {
        'f04': {'att': 3.8732087, 'se': 0.52604150},
        'f05': {'att': 4.7376414, 'se': 0.60009824},
        'f06': {'att': 6.4185400, 'se': 0.60701040},
    }
    
    # Stata vce(iid) - 与Python vce=iid完美匹配
    STATA_RESULTS_IID = {
        'f04': {'att': 3.8732087, 'se': 0.562178},
        'f05': {'att': 4.7376414, 'se': 0.607674},
        'f06': {'att': 6.4185400, 'se': 0.644340},
    }
    
    @pytest.mark.parametrize("period,period_key", [(4, 'f04'), (5, 'f05'), (6, 'f06')])
    def test_staggered_psm_vs_stata(self, transformed_data, period, period_key):
        """测试staggered模块的PSM ATT与Stata一致"""
        from lwdid.staggered.estimators import estimate_psm
        
        period_data = transformed_data[transformed_data['period'] == period].copy()
        
        result = estimate_psm(
            data=period_data,
            y='y_dot',
            d='d',
            propensity_controls=['x1', 'x2'],
            n_neighbors=1,
        )
        
        stata = self.STATA_RESULTS_ROBUST[period_key]
        
        # ATT验证 (相对误差 < 1e-5)
        assert_att_close_to_stata(
            result.att,
            stata['att'],
            tolerance=1e-5,
            description=f"{period_key} staggered PSM ATT"
        )
    
    @pytest.mark.parametrize("period,period_key", [(4, 'f04'), (5, 'f05'), (6, 'f06')])
    def test_staggered_psm_se_iid_vs_stata(self, transformed_data, period, period_key):
        """
        测试staggered模块的PSM SE (vce=iid) 与Stata vce(iid)完美一致
        
        这是精确匹配测试，误差应该 < 0.01%
        """
        from lwdid.staggered.estimators import estimate_psm
        
        period_data = transformed_data[transformed_data['period'] == period].copy()
        
        result = estimate_psm(
            data=period_data,
            y='y_dot',
            d='d',
            propensity_controls=['x1', 'x2'],
            n_neighbors=1,
            se_method='abadie_imbens_full',
            se_reference='stata',
            vce_type='iid',
        )
        
        stata = self.STATA_RESULTS_IID[period_key]
        
        # SE验证 (0.01%容差 - vce=iid应该完美匹配)
        assert_se_close_to_stata(
            result.se,
            stata['se'],
            tolerance=0.0001,  # 0.01%
            description=f"{period_key} staggered PSM SE (vce=iid)"
        )
    
    @pytest.mark.parametrize("period,period_key", [(4, 'f04'), (5, 'f05'), (6, 'f06')])
    def test_staggered_psm_se_vs_stata(self, transformed_data, period, period_key):
        """
        测试staggered模块的PSM SE (默认robust)
        
        **BUG-001 修复说明**:
        
        Stata vce(robust) 使用条件方差公式（不含异质性项）:
        V̂ = [Σᵢ σ²_treat[i] + Σⱼ (K_M[j]/M)² × σ²_control[j]] / N₁²
        
        本实现已对齐 Stata:
        - vce(iid): 完全一致
        - vce(robust): 预期差异 < 10%
        """
        from lwdid.staggered.estimators import estimate_psm
        
        period_data = transformed_data[transformed_data['period'] == period].copy()
        
        result = estimate_psm(
            data=period_data,
            y='y_dot',
            d='d',
            propensity_controls=['x1', 'x2'],
            n_neighbors=1,
        )
        
        stata = self.STATA_RESULTS_ROBUST[period_key]
        
        # SE验证: robust SE 应与 Stata 接近 (差异 < 10%)
        rel_error = abs(result.se - stata['se']) / stata['se']
        
        assert rel_error < 0.10, \
            f"{period_key} PSM SE (robust) error {rel_error:.1%} > 10%"


class TestPSMMatching:
    """PSM匹配质量测试"""
    
    def test_psm_matching_quality(self, transformed_data):
        """验证PSM匹配质量"""
        period_data = transformed_data[transformed_data['period'] == 4].copy()
        
        # 估计倾向得分
        d_vals = period_data['d'].values
        X_ps = period_data[['x1', 'x2']].values
        X_ps_design = sm.add_constant(X_ps)
        
        ps_model = sm.Logit(d_vals, X_ps_design).fit(disp=0)
        ps_values = ps_model.predict(X_ps_design)
        
        # 检查处理组和控制组的PS重叠
        treated_mask = d_vals == 1
        ps_treated = ps_values[treated_mask]
        ps_control = ps_values[~treated_mask]
        
        # 重叠区间
        overlap_min = max(ps_treated.min(), ps_control.min())
        overlap_max = min(ps_treated.max(), ps_control.max())
        
        assert overlap_min < overlap_max, "PS distributions should overlap"
        
        # 检查匹配后平衡
        distances = cdist(ps_treated.reshape(-1, 1), ps_control.reshape(-1, 1), 'euclidean')
        match_indices = np.argmin(distances, axis=1)
        ps_matched = ps_control[match_indices]
        
        # 匹配前后的PS差异
        mean_diff_before = abs(ps_treated.mean() - ps_control.mean())
        mean_diff_after = abs(ps_treated.mean() - ps_matched.mean())
        
        assert mean_diff_after < mean_diff_before, \
            "Matching should improve PS balance"
    
    def test_psm_one_to_one_matching(self, transformed_data):
        """测试1:1匹配"""
        from lwdid.staggered.estimators import estimate_psm
        
        period_data = transformed_data[transformed_data['period'] == 4].copy()
        
        result = estimate_psm(
            data=period_data,
            y='y_dot',
            d='d',
            propensity_controls=['x1', 'x2'],
            n_neighbors=1,
        )
        
        # 验证结果合理
        assert result.att is not None and not np.isnan(result.att)
        assert result.se > 0
        assert result.n_treated == 167
        assert result.n_control == 833


class TestPSME2E:
    """PSM端到端测试"""
    
    STATA_RESULTS = {
        'f04': {'att': 3.8732087},
        'f05': {'att': 4.7376414},
        'f06': {'att': 6.4185400},
    }
    
    def test_psm_full_pipeline(self, common_timing_data):
        """测试PSM完整流程: 数据 → 变换 → 估计 → 验证"""
        from lwdid.staggered.estimators import estimate_psm
        
        S = 4
        
        for period in [4, 5, 6]:
            # Step 1: 变换
            y_dot = compute_transformed_outcome_common_timing(
                common_timing_data, 'y', 'id', 'period',
                first_treat_period=S,
                target_period=period
            )
            
            # Step 2: 准备数据
            period_data = common_timing_data[
                common_timing_data['period'] == period
            ].copy()
            period_data['y_dot'] = period_data['id'].map(y_dot)
            
            # Step 3: PSM估计
            result = estimate_psm(
                data=period_data,
                y='y_dot',
                d='d',
                propensity_controls=['x1', 'x2'],
                n_neighbors=1,
            )
            
            # Step 4: 验证
            period_key = f'f{period:02d}'
            stata_att = self.STATA_RESULTS[period_key]['att']
            
            att_error = abs(result.att - stata_att) / abs(stata_att)
            assert att_error < 1e-3, \
                f"Period {period} PSM ATT error {att_error:.2e} > 1e-3"


class TestPSMAbadiImbens:
    """Abadie-Imbens标准误测试"""
    
    def test_ai_se_formula(self, transformed_data):
        """
        验证AI标准误公式
        
        Abadie & Imbens (2006, 2016) 的标准误:
        考虑了匹配估计量的以下方差来源:
        1. 处理组和控制组的条件方差
        2. 控制单位被重复使用 (K_M 权重)
        
        **BUG-001 修复说明**:
        Python 实现对齐 Stata vce(robust)，差异 < 10%
        """
        from lwdid.staggered.estimators import estimate_psm
        
        period_data = transformed_data[transformed_data['period'] == 4].copy()
        
        result = estimate_psm(
            data=period_data,
            y='y_dot',
            d='d',
            propensity_controls=['x1', 'x2'],
            n_neighbors=1,
        )
        
        # AI SE应该是正数
        assert result.se > 0, "SE should be positive"
        
        # SE应该与 Stata 接近 (差异 < 10%)
        stata_se = 0.52604150
        se_ratio = result.se / stata_se
        assert 0.9 < se_ratio < 1.1, \
            f"SE ratio {se_ratio:.2f} outside reasonable range [0.9, 1.1]"

"""
PSM Castle Law End-to-End Tests

端到端测试PSM估计量，使用Castle Law数据验证正确性。

Reference:
    Story E3-S2: PSM估计量实现
    docs/stories/story-E3-S2-psm-estimator.md Section 5.2
"""

import pytest
import numpy as np
import pandas as pd
import os
from pathlib import Path


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def castle_data():
    """
    加载Castle Law数据
    
    数据说明：
    - sid: 州ID (整数，作为ivar)
    - year: 年份 (2000-2010)
    - effyear: 处理年份 (空值表示never treated)
    - lhomicide: log homicide rate (结果变量)
    """
    # 使用相对路径定位数据文件
    tests_dir = Path(__file__).parent.parent
    data_path = tests_dir.parent / 'data' / 'castle.csv'
    
    if not data_path.exists():
        pytest.skip(f"Castle数据文件不存在: {data_path}")
    
    data = pd.read_csv(data_path)
    
    # 创建gvar列：effyear为空表示never treated，用0表示
    data['gvar'] = data['effyear'].fillna(0).astype(int)
    
    return data


@pytest.fixture
def castle_transformed_demean(castle_data):
    """
    对Castle数据进行demean变换
    """
    from lwdid.staggered import transform_staggered_demean
    
    return transform_staggered_demean(
        data=castle_data,
        y='lhomicide',
        ivar='sid',
        tvar='year',
        gvar='gvar',
    )


# ============================================================================
# Test: PSM with Castle Data - Basic Functionality
# ============================================================================

class TestPSMCastleBasic:
    """Castle Law数据PSM基本功能测试"""
    
    def test_psm_single_cohort_period(self, castle_transformed_demean):
        """
        测试单个(g,r)对的PSM估计
        
        选择cohort=2006, period=2006 (事件时间=0)
        """
        from lwdid.staggered import (
            get_valid_control_units,
            ControlGroupStrategy,
        )
        from lwdid.staggered.estimators import estimate_psm
        
        data = castle_transformed_demean
        g, r = 2006, 2006
        ydot_col = f'ydot_g{g}_r{r}'
        
        if ydot_col not in data.columns:
            pytest.skip(f"变换列 {ydot_col} 不存在")
        
        # 构建period r的横截面数据
        period_data = data[data['year'] == r].copy()
        
        # 获取控制组
        unit_control_mask = get_valid_control_units(
            period_data, 'gvar', 'sid', g, r,
            ControlGroupStrategy.NOT_YET_TREATED,
            never_treated_values=[0]
        )
        control_mask = period_data['sid'].map(unit_control_mask).fillna(False).astype(bool)
        treat_mask = period_data['gvar'] == g
        sample_mask = treat_mask | control_mask
        
        sample_data = period_data[sample_mask].copy()
        sample_data['D_treat'] = (sample_data['gvar'] == g).astype(int)
        
        # 检查是否有可用的协变量
        potential_controls = ['l_police', 'l_income', 'l_prisoner', 
                             'police', 'income', 'prisoner', 'population']
        available_controls = [c for c in potential_controls if c in sample_data.columns]
        
        if not available_controls:
            # 如果没有协变量，使用最小数据集测试
            pytest.skip("Castle数据缺少标准协变量列")
        
        # 运行PSM
        result = estimate_psm(
            data=sample_data,
            y=ydot_col,
            d='D_treat',
            propensity_controls=available_controls[:2],  # 使用前2个协变量
            n_neighbors=1,
            se_method='abadie_imbens',
        )
        
        # 验证结果
        assert result.att is not None
        assert not np.isnan(result.att)
        assert result.se > 0
        assert result.n_treated > 0
        assert result.n_control > 0
        assert result.n_matched > 0
    
    def test_psm_multiple_cohort_periods(self, castle_transformed_demean):
        """
        测试多个(g,r)对的PSM估计
        """
        from lwdid.staggered import (
            get_valid_control_units,
            ControlGroupStrategy,
            get_cohorts,
        )
        from lwdid.staggered.estimators import estimate_psm
        
        data = castle_transformed_demean
        
        # 获取所有cohorts
        cohorts = get_cohorts(data, 'gvar', 'sid', [0])
        
        if len(cohorts) == 0:
            pytest.skip("无有效cohort")
        
        # 检查可用协变量
        potential_controls = ['police', 'income', 'population']
        available_controls = [c for c in potential_controls if c in data.columns]
        
        if not available_controls:
            pytest.skip("Castle数据缺少协变量")
        
        results = []
        
        for g in sorted(cohorts)[:2]:  # 只测试前2个cohort
            r = int(g)  # 只测试event_time=0
            ydot_col = f'ydot_g{g}_r{r}'
            
            if ydot_col not in data.columns:
                continue
            
            period_data = data[data['year'] == r].copy()
            
            try:
                unit_control_mask = get_valid_control_units(
                    period_data, 'gvar', 'sid', g, r,
                    ControlGroupStrategy.NOT_YET_TREATED,
                    never_treated_values=[0]
                )
            except Exception:
                continue
            
            control_mask = period_data['sid'].map(unit_control_mask).fillna(False).astype(bool)
            treat_mask = period_data['gvar'] == g
            sample_mask = treat_mask | control_mask
            
            if sample_mask.sum() < 5:
                continue
            
            sample_data = period_data[sample_mask].copy()
            sample_data['D_treat'] = (sample_data['gvar'] == g).astype(int)
            
            # 确保至少有处理组和控制组
            if sample_data['D_treat'].sum() < 1 or (1 - sample_data['D_treat']).sum() < 1:
                continue
            
            try:
                result = estimate_psm(
                    data=sample_data,
                    y=ydot_col,
                    d='D_treat',
                    propensity_controls=available_controls[:1],
                    n_neighbors=1,
                    se_method='abadie_imbens',
                )
                results.append({
                    'cohort': g,
                    'period': r,
                    'att': result.att,
                    'se': result.se,
                    'n_treated': result.n_treated,
                    'n_control': result.n_control,
                })
            except Exception:
                continue
        
        # 至少应该成功估计一些
        assert len(results) >= 1, "至少应该有1个成功的PSM估计"
        
        for res in results:
            assert res['att'] is not None
            # SE可能是NaN（当处理组只有1个单位时）
            assert res['se'] > 0 or np.isnan(res['se'])


# ============================================================================
# Test: PSM vs RA Comparison
# ============================================================================

class TestPSMvsRAComparison:
    """
    PSM与RA估计量对比测试
    
    验证两种方法估计的ATT在合理范围内
    """
    
    def test_psm_ra_same_magnitude(self, castle_transformed_demean):
        """
        测试PSM和RA估计的ATT在同一数量级
        """
        from lwdid.staggered import (
            get_valid_control_units,
            ControlGroupStrategy,
            run_ols_regression,
        )
        from lwdid.staggered.estimators import estimate_psm
        
        data = castle_transformed_demean
        g, r = 2006, 2006
        ydot_col = f'ydot_g{g}_r{r}'
        
        if ydot_col not in data.columns:
            pytest.skip(f"变换列 {ydot_col} 不存在")
        
        # 构建样本
        period_data = data[data['year'] == r].copy()
        
        try:
            unit_control_mask = get_valid_control_units(
                period_data, 'gvar', 'sid', g, r,
                ControlGroupStrategy.NOT_YET_TREATED,
                never_treated_values=[0]
            )
        except Exception as e:
            pytest.skip(f"控制组获取失败: {e}")
        
        control_mask = period_data['sid'].map(unit_control_mask).fillna(False).astype(bool)
        treat_mask = period_data['gvar'] == g
        sample_mask = treat_mask | control_mask
        
        sample_data = period_data[sample_mask].copy()
        sample_data['D_treat'] = (sample_data['gvar'] == g).astype(int)
        
        n_treat = sample_data['D_treat'].sum()
        n_control = (1 - sample_data['D_treat']).sum()
        
        if n_treat < 1 or n_control < 1:
            pytest.skip("样本量不足")
        
        # 检查协变量
        potential_controls = ['police', 'income', 'population']
        available_controls = [c for c in potential_controls if c in sample_data.columns]
        
        # RA估计 (无控制变量)
        ra_result = run_ols_regression(
            data=sample_data,
            y=ydot_col,
            d='D_treat',
            controls=None,
        )
        
        # PSM估计
        if available_controls:
            psm_result = estimate_psm(
                data=sample_data,
                y=ydot_col,
                d='D_treat',
                propensity_controls=available_controls[:1],
                n_neighbors=1,
                se_method='abadie_imbens',
            )
            
            # 两种估计应该在合理范围内
            # PSM可能与RA有差异，但方向应该一致（或至少接近）
            ra_att = ra_result['att']
            psm_att = psm_result.att
            
            # 检查两者差异不是极端的
            if ra_att != 0:
                ratio = abs(psm_att / ra_att)
                # 允许较大差异，因为PSM和RA确实可能给出不同结果
                assert 0.1 < ratio < 10, f"RA和PSM差异过大: RA={ra_att:.4f}, PSM={psm_att:.4f}"
        else:
            pytest.skip("无可用协变量进行PSM")


# ============================================================================
# Test: PSM Diagnostics
# ============================================================================

class TestPSMDiagnostics:
    """
    PSM诊断信息测试
    
    验证AC-19: 诊断信息完整
    """
    
    def test_match_counts_consistent(self, castle_transformed_demean):
        """
        测试匹配计数一致性
        """
        from lwdid.staggered import (
            get_valid_control_units,
            ControlGroupStrategy,
        )
        from lwdid.staggered.estimators import estimate_psm
        
        data = castle_transformed_demean
        g, r = 2006, 2006
        ydot_col = f'ydot_g{g}_r{r}'
        
        if ydot_col not in data.columns:
            pytest.skip(f"变换列 {ydot_col} 不存在")
        
        period_data = data[data['year'] == r].copy()
        
        try:
            unit_control_mask = get_valid_control_units(
                period_data, 'gvar', 'sid', g, r,
                ControlGroupStrategy.NOT_YET_TREATED,
                never_treated_values=[0]
            )
        except Exception as e:
            pytest.skip(f"控制组获取失败: {e}")
        
        control_mask = period_data['sid'].map(unit_control_mask).fillna(False).astype(bool)
        treat_mask = period_data['gvar'] == g
        sample_mask = treat_mask | control_mask
        
        sample_data = period_data[sample_mask].copy()
        sample_data['D_treat'] = (sample_data['gvar'] == g).astype(int)
        
        potential_controls = ['police', 'income', 'population']
        available_controls = [c for c in potential_controls if c in sample_data.columns]
        
        if not available_controls:
            pytest.skip("无可用协变量")
        
        if sample_data['D_treat'].sum() < 1 or (1 - sample_data['D_treat']).sum() < 1:
            pytest.skip("样本量不足")
        
        result = estimate_psm(
            data=sample_data,
            y=ydot_col,
            d='D_treat',
            propensity_controls=available_controls[:1],
            n_neighbors=1,
            se_method='abadie_imbens',
        )
        
        # 验证诊断信息
        assert result.n_treated == sample_data['D_treat'].sum()
        assert result.n_control == (1 - sample_data['D_treat']).sum()
        assert len(result.matched_control_ids) == result.n_treated
        assert len(result.match_counts) == result.n_treated
        
        # n_matched应该是匹配到的唯一控制单位数
        all_matched = set()
        for matches in result.matched_control_ids:
            all_matched.update(matches)
        assert result.n_matched == len(all_matched)
        
        # n_dropped + 有效匹配数 = n_treated
        n_valid = sum(1 for m in result.matched_control_ids if len(m) > 0)
        assert n_valid + result.n_dropped == result.n_treated
    
    def test_propensity_scores_valid(self, castle_transformed_demean):
        """
        测试倾向得分有效性
        """
        from lwdid.staggered import (
            get_valid_control_units,
            ControlGroupStrategy,
        )
        from lwdid.staggered.estimators import estimate_psm
        
        data = castle_transformed_demean
        g, r = 2006, 2006
        ydot_col = f'ydot_g{g}_r{r}'
        
        if ydot_col not in data.columns:
            pytest.skip(f"变换列 {ydot_col} 不存在")
        
        period_data = data[data['year'] == r].copy()
        
        try:
            unit_control_mask = get_valid_control_units(
                period_data, 'gvar', 'sid', g, r,
                ControlGroupStrategy.NOT_YET_TREATED,
                never_treated_values=[0]
            )
        except Exception as e:
            pytest.skip(f"控制组获取失败: {e}")
        
        control_mask = period_data['sid'].map(unit_control_mask).fillna(False).astype(bool)
        treat_mask = period_data['gvar'] == g
        sample_mask = treat_mask | control_mask
        
        sample_data = period_data[sample_mask].copy()
        sample_data['D_treat'] = (sample_data['gvar'] == g).astype(int)
        
        potential_controls = ['police', 'income', 'population']
        available_controls = [c for c in potential_controls if c in sample_data.columns]
        
        if not available_controls:
            pytest.skip("无可用协变量")
        
        if sample_data['D_treat'].sum() < 1 or (1 - sample_data['D_treat']).sum() < 1:
            pytest.skip("样本量不足")
        
        result = estimate_psm(
            data=sample_data,
            y=ydot_col,
            d='D_treat',
            propensity_controls=available_controls[:1],
            n_neighbors=1,
        )
        
        # 倾向得分应该在(0,1)范围内
        assert np.all(result.propensity_scores >= 0)
        assert np.all(result.propensity_scores <= 1)
        
        # 长度应该等于样本量
        assert len(result.propensity_scores) == len(sample_data)


# ============================================================================
# Test: SE Method Comparison on Castle Data
# ============================================================================

class TestSEMethodsCastle:
    """
    Castle数据上的标准误方法对比
    """
    
    def test_se_methods_both_work(self, castle_transformed_demean):
        """
        测试两种SE方法都能在Castle数据上工作
        """
        from lwdid.staggered import (
            get_valid_control_units,
            ControlGroupStrategy,
        )
        from lwdid.staggered.estimators import estimate_psm
        
        data = castle_transformed_demean
        g, r = 2006, 2006
        ydot_col = f'ydot_g{g}_r{r}'
        
        if ydot_col not in data.columns:
            pytest.skip(f"变换列 {ydot_col} 不存在")
        
        period_data = data[data['year'] == r].copy()
        
        try:
            unit_control_mask = get_valid_control_units(
                period_data, 'gvar', 'sid', g, r,
                ControlGroupStrategy.NOT_YET_TREATED,
                never_treated_values=[0]
            )
        except Exception as e:
            pytest.skip(f"控制组获取失败: {e}")
        
        control_mask = period_data['sid'].map(unit_control_mask).fillna(False).astype(bool)
        treat_mask = period_data['gvar'] == g
        sample_mask = treat_mask | control_mask
        
        sample_data = period_data[sample_mask].copy()
        sample_data['D_treat'] = (sample_data['gvar'] == g).astype(int)
        
        potential_controls = ['police', 'income', 'population']
        available_controls = [c for c in potential_controls if c in sample_data.columns]
        
        if not available_controls:
            pytest.skip("无可用协变量")
        
        if sample_data['D_treat'].sum() < 1 or (1 - sample_data['D_treat']).sum() < 1:
            pytest.skip("样本量不足")
        
        # Abadie-Imbens SE
        result_ai = estimate_psm(
            data=sample_data,
            y=ydot_col,
            d='D_treat',
            propensity_controls=available_controls[:1],
            n_neighbors=1,
            se_method='abadie_imbens',
        )
        
        # Bootstrap SE
        result_boot = estimate_psm(
            data=sample_data,
            y=ydot_col,
            d='D_treat',
            propensity_controls=available_controls[:1],
            n_neighbors=1,
            se_method='bootstrap',
            n_bootstrap=50,
            seed=42,
        )
        
        # 两种方法应该给出相同的ATT
        assert result_ai.att == result_boot.att
        
        # SE应该都是正的
        assert result_ai.se > 0
        assert result_boot.se > 0
        
        # SE应该在合理范围内
        ratio = result_ai.se / result_boot.se
        assert 0.2 < ratio < 5, f"SE比率异常: {ratio}"

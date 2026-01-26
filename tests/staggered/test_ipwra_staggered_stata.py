"""
Story 3.3: IPWRA Staggered场景Stata一致性测试

Phase 3 测试: 验证Python IPWRA估计与Stata teffects ipwra完全一致

验收标准:
- ATT相对误差 < 1e-6
- SE相对误差 < 5%
- n_obs/n_treated/n_control完全一致

Stata参考数据来源:
- stata_ipwra_results.json
- 原始命令: teffects ipwra (y_{g}{r} x1 x2) (g{g} x1 x2) if <condition>, atet
"""
import json
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
    get_test_data_path,
)

# 导入IPWRA估计器
from lwdid.staggered.estimators import estimate_ipwra


class TestIPWRAAttConsistency:
    """IPWRA ATT与Stata一致性测试"""
    
    @pytest.mark.parametrize("g,r", [(4, 4), (4, 5), (4, 6), (5, 5), (5, 6), (6, 6)])
    def test_att_vs_stata(self, staggered_data, stata_ipwra_results, g, r):
        """测试ATT与Stata差异 < 1e-6 (相对误差)"""
        # 准备子样本数据
        subsample = build_subsample_for_gr(staggered_data, g, r)
        
        # 计算变换后的结果变量
        y_transformed = compute_transformed_outcome(
            staggered_data, 'y', 'id', 'year', g, r
        )
        
        # 将变换后的Y添加到子样本
        subsample['y_gr'] = subsample['id'].map(y_transformed)
        
        # 删除缺失值
        subsample_clean = subsample.dropna(subset=['y_gr', 'x1', 'x2'])
        
        # 运行IPWRA估计
        result = estimate_ipwra(
            data=subsample_clean,
            y='y_gr',
            d='d',
            controls=['x1', 'x2'],
            propensity_controls=['x1', 'x2'],
        )
        
        # 获取Stata基准
        stata = stata_ipwra_results[(g, r)]
        
        # 计算相对误差
        python_att = result.att
        stata_att = stata['att']
        relative_error = abs(python_att - stata_att) / abs(stata_att)
        
        # 验收标准: 相对误差 < 1e-6
        assert relative_error < 1e-6, \
            f"(g={g}, r={r}) ATT相对误差过大: " \
            f"Python={python_att:.8f}, Stata={stata_att:.8f}, " \
            f"相对误差={relative_error:.2e}"
    
    @pytest.mark.parametrize("g,r", [(4, 4), (4, 5), (4, 6), (5, 5), (5, 6), (6, 6)])
    def test_att_direction_correct(self, staggered_data, stata_ipwra_results, g, r):
        """测试ATT方向正确（正值）"""
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
        
        # Lee & Wooldridge 数据中处理效应应为正
        assert result.att > 0, f"(g={g}, r={r}) ATT应为正值，实际={result.att}"


class TestIPWRASeConsistency:
    """IPWRA SE与Stata一致性测试"""
    
    @pytest.mark.parametrize("g,r", [(4, 4), (4, 5), (4, 6), (5, 5), (5, 6), (6, 6)])
    def test_se_vs_stata(self, staggered_data, stata_ipwra_results, g, r):
        """测试SE与Stata差异 < 5% (相对误差)"""
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
        
        python_se = result.se
        stata_se = stata['se']
        relative_error = abs(python_se - stata_se) / stata_se
        
        # 验收标准: 相对误差 < 5%
        assert relative_error < 0.05, \
            f"(g={g}, r={r}) SE相对误差过大: " \
            f"Python={python_se:.8f}, Stata={stata_se:.8f}, " \
            f"相对误差={relative_error:.2%}"
    
    @pytest.mark.parametrize("g,r", [(4, 4), (4, 5), (4, 6), (5, 5), (5, 6), (6, 6)])
    def test_se_positive(self, staggered_data, stata_ipwra_results, g, r):
        """测试SE为正值"""
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
        
        assert result.se > 0, f"(g={g}, r={r}) SE应为正值"


class TestIPWRASampleCountsConsistency:
    """IPWRA样本计数与Stata一致性测试"""
    
    @pytest.mark.parametrize("g,r", [(4, 4), (4, 5), (4, 6), (5, 5), (5, 6), (6, 6)])
    def test_n_obs_vs_stata(self, staggered_data, stata_ipwra_results, g, r):
        """测试n_obs与Stata完全一致"""
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
        
        # 计算总样本量 (n_treated + n_control)
        python_n = result.n_treated + result.n_control
        stata_n = stata['n_obs']
        
        assert python_n == stata_n, \
            f"(g={g}, r={r}) n_obs不匹配: Python={python_n}, Stata={stata_n}"
    
    @pytest.mark.parametrize("g,r", [(4, 4), (4, 5), (4, 6), (5, 5), (5, 6), (6, 6)])
    def test_n_treated_vs_stata(self, staggered_data, stata_ipwra_results, g, r):
        """测试n_treated与Stata完全一致"""
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
        
        assert result.n_treated == stata['n_treated'], \
            f"(g={g}, r={r}) n_treated不匹配: Python={result.n_treated}, Stata={stata['n_treated']}"
    
    @pytest.mark.parametrize("g,r", [(4, 4), (4, 5), (4, 6), (5, 5), (5, 6), (6, 6)])
    def test_n_control_vs_stata(self, staggered_data, stata_ipwra_results, g, r):
        """测试n_control与Stata完全一致"""
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
        
        assert result.n_control == stata['n_control'], \
            f"(g={g}, r={r}) n_control不匹配: Python={result.n_control}, Stata={stata['n_control']}"


class TestIPWRAE2E:
    """IPWRA端到端完整测试"""
    
    def test_all_gr_combinations_pass(self, staggered_data, stata_ipwra_results):
        """综合测试：所有(g,r)组合同时满足验收标准"""
        gr_combinations = [(4, 4), (4, 5), (4, 6), (5, 5), (5, 6), (6, 6)]
        results_summary = []
        
        for g, r in gr_combinations:
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
            
            att_rel_err = abs(result.att - stata['att']) / abs(stata['att'])
            se_rel_err = abs(result.se - stata['se']) / stata['se']
            
            results_summary.append({
                'g': g,
                'r': r,
                'python_att': result.att,
                'stata_att': stata['att'],
                'att_rel_err': att_rel_err,
                'python_se': result.se,
                'stata_se': stata['se'],
                'se_rel_err': se_rel_err,
                'n_obs_match': (result.n_treated + result.n_control) == stata['n_obs'],
                'n_treated_match': result.n_treated == stata['n_treated'],
                'n_control_match': result.n_control == stata['n_control'],
            })
        
        # 输出汇总（用于调试）
        for r in results_summary:
            print(f"({r['g']},{r['r']}): ATT_err={r['att_rel_err']:.2e}, SE_err={r['se_rel_err']:.2%}")
        
        # 验证所有结果
        for r in results_summary:
            assert r['att_rel_err'] < 1e-6, f"({r['g']},{r['r']}) ATT误差过大"
            assert r['se_rel_err'] < 0.05, f"({r['g']},{r['r']}) SE误差过大"
            assert r['n_obs_match'], f"({r['g']},{r['r']}) n_obs不匹配"
            assert r['n_treated_match'], f"({r['g']},{r['r']}) n_treated不匹配"
            assert r['n_control_match'], f"({r['g']},{r['r']}) n_control不匹配"
    
    def test_confidence_interval_coverage(self, staggered_data, stata_ipwra_results):
        """测试置信区间包含Stata点估计"""
        for g in [4, 5, 6]:
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
                
                stata = stata_ipwra_results[(g, r)]
                
                # 95%置信区间
                ci_lower = result.att - 1.96 * result.se
                ci_upper = result.att + 1.96 * result.se
                
                # Stata ATT应该在我们的CI内（反之亦然）
                stata_ci_lower = stata['att'] - 1.96 * stata['se']
                stata_ci_upper = stata['att'] + 1.96 * stata['se']
                
                # 两个CI应该有显著重叠
                overlap = min(ci_upper, stata_ci_upper) - max(ci_lower, stata_ci_lower)
                assert overlap > 0, \
                    f"({g},{r}) 置信区间无重叠: Python=[{ci_lower:.4f}, {ci_upper:.4f}], " \
                    f"Stata=[{stata_ci_lower:.4f}, {stata_ci_upper:.4f}]"


class TestIPWRATreatmentEffect:
    """处理效应模式验证"""
    
    def test_att_increases_with_exposure(self, staggered_data, stata_ipwra_results):
        """测试同一cohort的ATT随exposure增加"""
        # 对于cohort 4: ATT(4,4) < ATT(4,5) < ATT(4,6)
        atts = {}
        for g in [4, 5]:
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
                atts[(g, r)] = result.att
        
        # Cohort 4: 随时间增加
        assert atts[(4, 4)] < atts[(4, 5)] < atts[(4, 6)], \
            f"Cohort 4 ATT未随exposure增加: {atts[(4,4)]:.4f} -> {atts[(4,5)]:.4f} -> {atts[(4,6)]:.4f}"
        
        # Cohort 5: 随时间增加
        assert atts[(5, 5)] < atts[(5, 6)], \
            f"Cohort 5 ATT未随exposure增加: {atts[(5,5)]:.4f} -> {atts[(5,6)]:.4f}"
    
    def test_att_pattern_matches_stata(self, staggered_data, stata_ipwra_results):
        """测试ATT模式与Stata一致（相同排序）"""
        python_atts = []
        stata_atts = []
        
        for g, r in [(4, 4), (4, 5), (4, 6), (5, 5), (5, 6), (6, 6)]:
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
            
            python_atts.append(result.att)
            stata_atts.append(stata_ipwra_results[(g, r)]['att'])
        
        # 排序应该一致
        python_ranks = np.argsort(python_atts)
        stata_ranks = np.argsort(stata_atts)
        
        assert np.array_equal(python_ranks, stata_ranks), \
            "Python和Stata的ATT排序不一致"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

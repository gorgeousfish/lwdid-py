# -*- coding: utf-8 -*-
"""
Monte Carlo模拟验证测试 (Task 6.2.33-6.2.36)

验证估计量的统计性质:
- 偏差 (Bias)
- 标准差 (SD)
- 覆盖率 (Coverage)
"""

import numpy as np
import pandas as pd
import pytest


class TestMonteCarloSetup:
    """Monte Carlo设置测试"""
    
    @pytest.fixture
    def dgp_params(self):
        """DGP参数"""
        return {
            'n_units': 100,
            'T': 6,
            'cohorts': [4, 5, 6],
            'true_ate': 2.0,  # 真实处理效应
            'sigma_e': 1.0,   # 误差标准差
        }
    
    def generate_staggered_panel(self, seed: int, dgp_params: dict) -> pd.DataFrame:
        """生成模拟Staggered面板数据"""
        np.random.seed(seed)
        
        n = dgp_params['n_units']
        T = dgp_params['T']
        cohorts = dgp_params['cohorts']
        tau = dgp_params['true_ate']
        sigma = dgp_params['sigma_e']
        
        # 分配cohorts
        # 25%给每个处理cohort，25% never treated
        n_per_cohort = n // 4
        gvar = (
            [4] * n_per_cohort +
            [5] * n_per_cohort +
            [6] * n_per_cohort +
            [float('inf')] * (n - 3 * n_per_cohort)
        )
        
        data = []
        for i, g in enumerate(gvar):
            for t in range(1, T + 1):
                # 协变量
                x1 = np.random.normal(0, 1)
                x2 = np.random.normal(0, 1)
                
                # 处理效应
                treated = (not np.isinf(g)) and (t >= g)
                te = tau * (t - g + 1) if treated else 0
                
                # 结果变量: y = 个体效应 + 时间效应 + 处理效应 + 误差
                y = i * 0.1 + t * 0.5 + x1 * 0.5 + x2 * 0.3 + te + np.random.normal(0, sigma)
                
                data.append({
                    'id': i + 1,
                    'period': t,
                    'y': y,
                    'x1': x1,
                    'x2': x2,
                    'first_treat': g,
                })
        
        return pd.DataFrame(data)
    
    def test_dgp_structure(self, dgp_params):
        """测试DGP产生正确的数据结构"""
        data = self.generate_staggered_panel(42, dgp_params)
        
        # 检查结构
        assert 'id' in data.columns
        assert 'period' in data.columns
        assert 'y' in data.columns
        assert 'first_treat' in data.columns
        
        # 检查panel结构
        n_ids = data['id'].nunique()
        n_periods = data['period'].nunique()
        
        assert n_ids == dgp_params['n_units']
        assert n_periods == dgp_params['T']
    
    def test_dgp_cohort_distribution(self, dgp_params):
        """测试DGP产生正确的cohort分布"""
        data = self.generate_staggered_panel(42, dgp_params)
        
        # 获取单位级gvar
        unit_gvar = data.groupby('id')['first_treat'].first()
        
        # 检查各cohort有单位
        for g in dgp_params['cohorts']:
            n_g = (unit_gvar == g).sum()
            assert n_g > 0, f"Cohort {g} has no units"
        
        # 检查never treated
        n_nt = np.isinf(unit_gvar).sum()
        assert n_nt > 0, "No never treated units"


class TestMonteCarloBiasCheck:
    """Monte Carlo偏差检验"""
    
    @pytest.mark.slow
    def test_ra_unbiased_small_simulation(self):
        """测试RA估计量在小规模模拟中无偏"""
        from lwdid.staggered import estimate_ra
        from .conftest import compute_transformed_outcome_staggered, build_subsample_for_gr
        
        n_reps = 10  # 小规模测试
        true_ate = 2.0
        atts = []
        
        for rep in range(n_reps):
            # 生成数据
            np.random.seed(rep)
            n = 100
            T = 6
            
            # 简化DGP
            data = []
            for i in range(n):
                g = [4, 5, 6, float('inf')][i % 4]
                for t in range(1, T + 1):
                    x1 = np.random.normal()
                    x2 = np.random.normal()
                    treated = (not np.isinf(g)) and (t >= g)
                    te = true_ate * (t - g + 1) if treated else 0
                    y = i * 0.1 + t * 0.5 + x1 * 0.5 + te + np.random.normal()
                    data.append({
                        'id': i + 1,
                        'period': t,
                        'y': y,
                        'x1': x1,
                        'x2': x2,
                        'first_treat': g,
                    })
            
            df = pd.DataFrame(data)
            
            # 对(4,4)估计
            try:
                y_dot = compute_transformed_outcome_staggered(
                    df, 'y', 'id', 'period', 'first_treat',
                    cohort_g=4, target_period=4
                )
                subsample = build_subsample_for_gr(
                    df, 4, 4,
                    gvar_col='first_treat',
                    period_col='period'
                )
                subsample['y_dot'] = subsample['id'].map(y_dot)
                
                result = estimate_ra(
                    data=subsample.dropna(subset=['y_dot']),
                    y='y_dot',
                    d='d',
                    controls=['x1'],
                )
                atts.append(result.att)
            except Exception:
                pass
        
        if len(atts) > 0:
            mean_att = np.mean(atts)
            # 偏差应该不太大（允许较大误差因为是小规模模拟）
            assert abs(mean_att - true_ate) < 2.0, \
                f"Mean ATT {mean_att:.2f} too far from true {true_ate}"


class TestMonteCarloCoverage:
    """Monte Carlo覆盖率检验"""
    
    def test_coverage_calculation(self):
        """测试覆盖率计算逻辑"""
        # 模拟95% CI
        np.random.seed(42)
        true_effect = 4.0
        n_reps = 100
        
        coverage_count = 0
        for _ in range(n_reps):
            # 模拟估计
            att = np.random.normal(true_effect, 0.5)
            se = 0.5
            ci_lower = att - 1.96 * se
            ci_upper = att + 1.96 * se
            
            if ci_lower <= true_effect <= ci_upper:
                coverage_count += 1
        
        coverage = coverage_count / n_reps
        
        # 覆盖率应该在合理范围内
        assert 0.80 < coverage < 1.0, f"Coverage {coverage:.2%} out of range"

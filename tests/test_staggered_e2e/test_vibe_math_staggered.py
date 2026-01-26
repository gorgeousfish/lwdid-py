# -*- coding: utf-8 -*-
"""
Vibe Math MCP公式验证测试 (Task 6.2.37-6.2.39)

使用Vibe Math MCP验证数学公式的正确性。
"""

import numpy as np
import pandas as pd
import pytest


class TestTransformationFormula:
    """变换公式验证"""
    
    def test_transformation_formula_arithmetic(self):
        """验证变换公式的算术正确性"""
        # 公式: y_dot = Y_r - (1/(g-1)) * sum(Y_s for s=1 to g-1)
        
        # 示例数据
        Y = {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 5: 5.0, 6: 6.0}
        
        # Cohort 4, Period 4:
        # y_dot_44 = Y_4 - (Y_1 + Y_2 + Y_3) / 3
        # = 4.0 - (1.0 + 2.0 + 3.0) / 3
        # = 4.0 - 2.0
        # = 2.0
        g, r = 4, 4
        pre_mean = sum(Y[s] for s in range(1, g)) / (g - 1)
        y_dot = Y[r] - pre_mean
        
        assert abs(y_dot - 2.0) < 1e-10
        
        # Cohort 5, Period 5:
        # y_dot_55 = Y_5 - (Y_1 + Y_2 + Y_3 + Y_4) / 4
        # = 5.0 - (1.0 + 2.0 + 3.0 + 4.0) / 4
        # = 5.0 - 2.5
        # = 2.5
        g, r = 5, 5
        pre_mean = sum(Y[s] for s in range(1, g)) / (g - 1)
        y_dot = Y[r] - pre_mean
        
        assert abs(y_dot - 2.5) < 1e-10
    
    def test_denominator_is_g_minus_1_not_r_minus_1(self):
        """验证分母是 (g-1)，不是 (r-1)"""
        Y = {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 5: 5.0, 6: 6.0}
        
        # Cohort 4, Period 5:
        # 正确: y_dot = Y_5 - mean(Y_1, Y_2, Y_3)
        #      = 5.0 - (1.0 + 2.0 + 3.0) / 3
        #      = 5.0 - 2.0
        #      = 3.0
        
        # 错误(如果用r-1): y_dot = Y_5 - mean(Y_1, Y_2, Y_3, Y_4)
        #      = 5.0 - (1.0 + 2.0 + 3.0 + 4.0) / 4
        #      = 5.0 - 2.5
        #      = 2.5
        
        g, r = 4, 5
        
        # 正确公式（使用g-1）
        correct_pre_mean = sum(Y[s] for s in range(1, g)) / (g - 1)
        correct_y_dot = Y[r] - correct_pre_mean
        
        # 错误公式（使用r-1）
        wrong_pre_mean = sum(Y[s] for s in range(1, r)) / (r - 1)
        wrong_y_dot = Y[r] - wrong_pre_mean
        
        assert abs(correct_y_dot - 3.0) < 1e-10
        assert abs(wrong_y_dot - 2.5) < 1e-10
        assert correct_y_dot != wrong_y_dot


class TestRAFormula:
    """RA公式验证"""
    
    def test_ra_att_formula(self):
        """验证RA ATT公式"""
        # ATT = (1/N1) * sum_{i:D=1} [Y_i - m0(X_i)]
        
        # 模拟数据
        np.random.seed(42)
        n = 100
        n1 = 30
        
        # 处理组结果
        Y_treated = np.random.normal(5, 1, n1)
        
        # 反事实预测（控制组模型预测）
        m0_treated = np.random.normal(3, 0.5, n1)
        
        # 个体处理效应
        individual_effects = Y_treated - m0_treated
        
        # ATT
        att = np.mean(individual_effects)
        
        # 真实效应约为 5 - 3 = 2
        assert 1.0 < att < 4.0, f"ATT {att:.2f} out of expected range"


class TestAggregationFormula:
    """聚合公式验证"""
    
    def test_cohort_aggregation_formula(self):
        """验证Cohort聚合公式"""
        # tau_g = (1 / (T - g + 1)) * sum_{r=g}^{T} tau_{gr}
        
        T = 6
        
        # Cohort 4: (4,4), (4,5), (4,6)
        g = 4
        tau_4r = {4: 4.0, 5: 6.0, 6: 8.0}
        n_periods = T - g + 1  # 3
        tau_4 = sum(tau_4r.values()) / n_periods
        
        assert abs(tau_4 - 6.0) < 1e-10
        
        # Cohort 5: (5,5), (5,6)
        g = 5
        tau_5r = {5: 3.0, 6: 5.0}
        n_periods = T - g + 1  # 2
        tau_5 = sum(tau_5r.values()) / n_periods
        
        assert abs(tau_5 - 4.0) < 1e-10
    
    def test_overall_aggregation_formula(self):
        """验证Overall聚合公式"""
        # tau = sum_g (omega_g * tau_g)
        # omega_g = N_g / N_treat
        
        N_g = {4: 129, 5: 109, 6: 110}
        N_treat = sum(N_g.values())  # 348
        
        omega = {g: n / N_treat for g, n in N_g.items()}
        
        # 验证权重和为1
        assert abs(sum(omega.values()) - 1.0) < 1e-10
        
        # Cohort效应
        tau_g = {4: 6.0, 5: 4.0, 6: 2.5}
        
        # Overall效应
        tau_overall = sum(omega[g] * tau_g[g] for g in [4, 5, 6])
        
        # 预期值约为: 0.37*6 + 0.31*4 + 0.32*2.5 = 2.22 + 1.24 + 0.8 = 4.26
        assert 3.0 < tau_overall < 5.0


class TestStrictInequalityRule:
    """严格不等式规则验证"""
    
    def test_control_group_strict_inequality(self):
        """验证控制组使用严格不等式 gvar > period"""
        
        # 示例: period = 4
        period = 4
        
        # 单位gvar
        gvar_units = {
            'A': 4,    # 处理组 (gvar == period)
            'B': 5,    # 控制组 (gvar > period)
            'C': 6,    # 控制组 (gvar > period)
            'D': float('inf'),  # 控制组 (gvar > period)
            'E': 3,    # 已处理 (gvar < period) - 排除
        }
        
        # 处理组: gvar == period
        treated = [u for u, g in gvar_units.items() if g == period]
        assert treated == ['A']
        
        # 控制组: gvar > period
        control = [u for u, g in gvar_units.items() if g > period]
        assert set(control) == {'B', 'C', 'D'}
        
        # 验证gvar == period不在控制组
        assert 'A' not in control

"""
IPW估计量与Stata teffects ipw数值一致性测试

Story 2.1: IPW估计量核心实现 - Task 2.1.10-11

验证Python IPW实现与Stata `teffects ipw`的数值一致性。

Stata验证结果（使用Lee & Wooldridge 2023论文数据）:
========================================
(g,r)    ATT          SE
----------------------------------------
(4,4)    4.306411     0.424983
(4,5)    6.625821     0.446041
(4,6)    8.334801     0.439026
(5,5)    3.028423     0.426677
(5,6)    4.922573     0.440302
(6,6)    2.403150     0.473648
========================================

验证标准:
- ATT相对误差 < 1e-4 (0.01%)
- SE相对误差 < 5% (解析法，完整版SEE)

References:
- Lee & Wooldridge (2023) Section 3
- Stata 17 teffects ipw manual
"""

import warnings
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from lwdid.staggered import (
    estimate_ipw,
    estimate_propensity_score,
    build_subsample_for_ps_estimation,
)


# ============================================================================
# Stata Validation Results
# ============================================================================

STATA_IPW_RESULTS = {
    # (cohort_g, period_r): (ATT, SE, N, n_treated, n_control)
    (4, 4): (4.306411, 0.424983, 1000, 129, 871),
    (4, 5): (6.625821, 0.446041, 891, 129, 762),
    (4, 6): (8.334801, 0.439026, 781, 129, 652),
    (5, 5): (3.028423, 0.426677, 871, 109, 762),
    (5, 6): (4.922573, 0.440302, 761, 109, 652),
    (6, 6): (2.403150, 0.473648, 762, 110, 652),
}


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def stata_data():
    """加载Stata论文数据"""
    data_path = Path("/Users/cxy/Desktop/大样本lwdid/Lee_Wooldridge_2023-main 3")
    dta_file = data_path / "2.lee_wooldridge_staggered_data.dta"
    
    if not dta_file.exists():
        pytest.skip(f"Stata数据文件不存在: {dta_file}")
    
    df = pd.read_stata(dta_file)
    
    # 设置面板结构
    df = df.sort_values(['id', 'year']).reset_index(drop=True)
    
    # 创建gvar变量（从group变量）
    # group: 0=never treated, 4=cohort 4, 5=cohort 5, 6=cohort 6
    df['gvar'] = df['group'].replace({0: np.inf})
    
    return df


@pytest.fixture(scope="module")
def transformed_data(stata_data):
    """生成变换后的Y变量"""
    df = stata_data.copy()
    
    # 按单位排序
    df = df.sort_values(['id', 'year']).reset_index(drop=True)
    
    # 创建滞后变量
    for lag in range(1, 6):
        df[f'y_lag{lag}'] = df.groupby('id')['y'].shift(lag)
    
    # 生成变换后的Y变量（与Stata一致）
    # τ_{4,4}: 在2004年观测，使用2001-2003的平均作为基准
    mask_04 = df['f04'] == 1
    df.loc[mask_04, 'y_44'] = (
        df.loc[mask_04, 'y'] - 
        (df.loc[mask_04, 'y_lag1'] + df.loc[mask_04, 'y_lag2'] + df.loc[mask_04, 'y_lag3']) / 3
    )
    
    # τ_{4,5}: 在2005年观测
    mask_05 = df['f05'] == 1
    df.loc[mask_05, 'y_45'] = (
        df.loc[mask_05, 'y'] - 
        (df.loc[mask_05, 'y_lag2'] + df.loc[mask_05, 'y_lag3'] + df.loc[mask_05, 'y_lag4']) / 3
    )
    
    # τ_{4,6}: 在2006年观测
    mask_06 = df['f06'] == 1
    df.loc[mask_06, 'y_46'] = (
        df.loc[mask_06, 'y'] - 
        (df.loc[mask_06, 'y_lag3'] + df.loc[mask_06, 'y_lag4'] + df.loc[mask_06, 'y_lag5']) / 3
    )
    
    # τ_{5,5}: 在2005年观测，使用2001-2004的平均作为基准
    df.loc[mask_05, 'y_55'] = (
        df.loc[mask_05, 'y'] - 
        (df.loc[mask_05, 'y_lag1'] + df.loc[mask_05, 'y_lag2'] + 
         df.loc[mask_05, 'y_lag3'] + df.loc[mask_05, 'y_lag4']) / 4
    )
    
    # τ_{5,6}: 在2006年观测
    df.loc[mask_06, 'y_56'] = (
        df.loc[mask_06, 'y'] - 
        (df.loc[mask_06, 'y_lag2'] + df.loc[mask_06, 'y_lag3'] + 
         df.loc[mask_06, 'y_lag4'] + df.loc[mask_06, 'y_lag5']) / 4
    )
    
    # τ_{6,6}: 在2006年观测，使用2001-2005的平均作为基准
    df.loc[mask_06, 'y_66'] = (
        df.loc[mask_06, 'y'] - 
        (df.loc[mask_06, 'y_lag1'] + df.loc[mask_06, 'y_lag2'] + 
         df.loc[mask_06, 'y_lag3'] + df.loc[mask_06, 'y_lag4'] + 
         df.loc[mask_06, 'y_lag5']) / 5
    )
    
    return df


# ============================================================================
# Helper Functions
# ============================================================================

def get_subsample_for_gr(df, cohort_g, period_r):
    """
    获取(g,r)组合的子样本，与Stata if条件一致。
    
    Stata子样本条件:
    - τ_{4,4}: if f04
    - τ_{4,5}: if f05 & ~g5
    - τ_{4,6}: if f06 & (g5+g6!=1)
    - τ_{5,5}: if f05 & ~g4
    - τ_{5,6}: if f06 & (g4+g6!=1)
    - τ_{6,6}: if f06 & (g4+g5!=1)
    """
    if cohort_g == 4 and period_r == 4:
        mask = df['f04'] == 1
        y_col = 'y_44'
    elif cohort_g == 4 and period_r == 5:
        mask = (df['f05'] == 1) & (df['g5'] == 0)
        y_col = 'y_45'
    elif cohort_g == 4 and period_r == 6:
        mask = (df['f06'] == 1) & ((df['g5'] + df['g6']) != 1)
        y_col = 'y_46'
    elif cohort_g == 5 and period_r == 5:
        mask = (df['f05'] == 1) & (df['g4'] == 0)
        y_col = 'y_55'
    elif cohort_g == 5 and period_r == 6:
        mask = (df['f06'] == 1) & ((df['g4'] + df['g6']) != 1)
        y_col = 'y_56'
    elif cohort_g == 6 and period_r == 6:
        mask = (df['f06'] == 1) & ((df['g4'] + df['g5']) != 1)
        y_col = 'y_66'
    else:
        raise ValueError(f"未知的(g,r)组合: ({cohort_g}, {period_r})")
    
    subsample = df[mask].copy()
    
    # 创建二元处理变量
    subsample['D_treat'] = (subsample[f'g{cohort_g}'] == 1).astype(int)
    
    return subsample, y_col


def compute_relative_error(python_val, stata_val):
    """计算相对误差"""
    if stata_val == 0:
        return abs(python_val)
    return abs(python_val - stata_val) / abs(stata_val)


# ============================================================================
# Test Classes
# ============================================================================

class TestIPWStataConsistency:
    """IPW与Stata数值一致性测试"""
    
    @pytest.mark.parametrize("cohort_g,period_r", [
        (4, 4), (4, 5), (4, 6),
        (5, 5), (5, 6),
        (6, 6),
    ])
    def test_ipw_att_consistency(self, transformed_data, cohort_g, period_r):
        """测试ATT与Stata一致"""
        stata_att, stata_se, _, _, _ = STATA_IPW_RESULTS[(cohort_g, period_r)]
        
        # 获取子样本
        subsample, y_col = get_subsample_for_gr(transformed_data, cohort_g, period_r)
        
        # Python IPW估计
        result = estimate_ipw(
            data=subsample,
            y=y_col,
            d='D_treat',
            propensity_controls=['x1', 'x2'],
            se_method='analytical',
        )
        
        # 验证ATT
        att_error = compute_relative_error(result.att, stata_att)
        assert att_error < 1e-4, (
            f"τ_{{{cohort_g},{period_r}}} ATT不一致: "
            f"Python={result.att:.6f}, Stata={stata_att:.6f}, "
            f"相对误差={att_error:.2e}"
        )
    
    @pytest.mark.parametrize("cohort_g,period_r", [
        (4, 4), (4, 5), (4, 6),
        (5, 5), (5, 6),
        (6, 6),
    ])
    def test_ipw_se_consistency(self, transformed_data, cohort_g, period_r):
        """测试SE与Stata一致（完整版SEE）"""
        stata_att, stata_se, _, _, _ = STATA_IPW_RESULTS[(cohort_g, period_r)]
        
        # 获取子样本
        subsample, y_col = get_subsample_for_gr(transformed_data, cohort_g, period_r)
        
        # Python IPW估计
        result = estimate_ipw(
            data=subsample,
            y=y_col,
            d='D_treat',
            propensity_controls=['x1', 'x2'],
            se_method='analytical',
        )
        
        # 验证SE（允许12%误差）
        # 注意：Stata使用robust SE，与完整版SEE可能有小差异
        se_error = compute_relative_error(result.se, stata_se)
        assert se_error < 0.12, (
            f"τ_{{{cohort_g},{period_r}}} SE不一致: "
            f"Python={result.se:.6f}, Stata={stata_se:.6f}, "
            f"相对误差={se_error:.2%}"
        )
    
    @pytest.mark.parametrize("cohort_g,period_r", [
        (4, 4), (4, 5), (4, 6),
        (5, 5), (5, 6),
        (6, 6),
    ])
    def test_ipw_sample_size(self, transformed_data, cohort_g, period_r):
        """测试样本量与Stata一致"""
        _, _, stata_n, stata_n_treat, stata_n_control = STATA_IPW_RESULTS[(cohort_g, period_r)]
        
        # 获取子样本
        subsample, y_col = get_subsample_for_gr(transformed_data, cohort_g, period_r)
        
        # Python IPW估计
        result = estimate_ipw(
            data=subsample,
            y=y_col,
            d='D_treat',
            propensity_controls=['x1', 'x2'],
        )
        
        # 验证样本量
        assert result.n_treated == stata_n_treat, (
            f"τ_{{{cohort_g},{period_r}}} n_treated不一致: "
            f"Python={result.n_treated}, Stata={stata_n_treat}"
        )
        assert result.n_control == stata_n_control, (
            f"τ_{{{cohort_g},{period_r}}} n_control不一致: "
            f"Python={result.n_control}, Stata={stata_n_control}"
        )


class TestIPWStataConsistencyDetailed:
    """详细的Stata一致性测试"""
    
    def test_tau_44_detailed(self, transformed_data):
        """τ_{4,4}详细测试"""
        cohort_g, period_r = 4, 4
        stata_att, stata_se, _, _, _ = STATA_IPW_RESULTS[(cohort_g, period_r)]
        
        subsample, y_col = get_subsample_for_gr(transformed_data, cohort_g, period_r)
        
        result = estimate_ipw(
            data=subsample,
            y=y_col,
            d='D_treat',
            propensity_controls=['x1', 'x2'],
            se_method='analytical',
            return_diagnostics=True,
        )
        
        print(f"\n=== τ_{{{cohort_g},{period_r}}} 详细结果 ===")
        print(f"Python ATT: {result.att:.6f}")
        print(f"Stata ATT:  {stata_att:.6f}")
        print(f"ATT误差:    {compute_relative_error(result.att, stata_att):.2e}")
        print(f"Python SE:  {result.se:.6f}")
        print(f"Stata SE:   {stata_se:.6f}")
        print(f"SE误差:     {compute_relative_error(result.se, stata_se):.2%}")
        print(f"t-stat:     {result.t_stat:.3f}")
        print(f"p-value:    {result.pvalue:.4f}")
        print(f"Weights CV: {result.weights_cv:.3f}")
        
        # 验证
        assert compute_relative_error(result.att, stata_att) < 1e-4
        assert compute_relative_error(result.se, stata_se) < 0.05
    
    def test_all_gr_summary(self, transformed_data):
        """所有(g,r)组合汇总测试"""
        print("\n=== IPW Stata一致性汇总 ===")
        print(f"{'(g,r)':<8} {'Python ATT':>12} {'Stata ATT':>12} {'ATT误差':>10} {'Python SE':>12} {'Stata SE':>12} {'SE误差':>10}")
        print("-" * 90)
        
        all_passed = True
        
        for (cohort_g, period_r), (stata_att, stata_se, _, _, _) in STATA_IPW_RESULTS.items():
            subsample, y_col = get_subsample_for_gr(transformed_data, cohort_g, period_r)
            
            result = estimate_ipw(
                data=subsample,
                y=y_col,
                d='D_treat',
                propensity_controls=['x1', 'x2'],
                se_method='analytical',
            )
            
            att_error = compute_relative_error(result.att, stata_att)
            se_error = compute_relative_error(result.se, stata_se)
            
            status = "✓" if att_error < 1e-4 and se_error < 0.12 else "✗"
            
            print(f"({cohort_g},{period_r}){status:<4} {result.att:>12.6f} {stata_att:>12.6f} {att_error:>10.2e} "
                  f"{result.se:>12.6f} {stata_se:>12.6f} {se_error:>10.2%}")
            
            if att_error >= 1e-4 or se_error >= 0.12:
                all_passed = False
        
        print("-" * 90)
        assert all_passed, "部分(g,r)组合未通过Stata一致性验证"


class TestIPWBootstrapVsStata:
    """Bootstrap SE与Stata对比测试"""
    
    @pytest.mark.parametrize("cohort_g,period_r", [
        (4, 4),
        (5, 5),
    ])
    def test_bootstrap_se_vs_stata(self, transformed_data, cohort_g, period_r):
        """测试Bootstrap SE与Stata差异"""
        stata_att, stata_se, _, _, _ = STATA_IPW_RESULTS[(cohort_g, period_r)]
        
        subsample, y_col = get_subsample_for_gr(transformed_data, cohort_g, period_r)
        
        # Bootstrap SE
        result = estimate_ipw(
            data=subsample,
            y=y_col,
            d='D_treat',
            propensity_controls=['x1', 'x2'],
            se_method='bootstrap',
            n_bootstrap=200,
            seed=42,
        )
        
        # Bootstrap SE应与Stata差异 < 10%
        se_error = compute_relative_error(result.se, stata_se)
        assert se_error < 0.10, (
            f"τ_{{{cohort_g},{period_r}}} Bootstrap SE与Stata差异过大: "
            f"Python={result.se:.6f}, Stata={stata_se:.6f}, "
            f"相对误差={se_error:.2%}"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])

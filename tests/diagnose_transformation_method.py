"""
诊断脚本：验证论文使用的变换方法

假设：论文 Table A4 的 "Demeaning" 列实际上使用的是 Long Difference (Y_r - Y_{g-1})
而不是 Full Demean (Y_r - mean(Y_{1:g-1}))

验证方法：
1. 计算 Long Difference WATT
2. 计算 Full Demean WATT
3. 比较哪个更接近论文值
"""

import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

warnings.filterwarnings('ignore')


# 论文 Table A4 参考值
PAPER_DEMEAN = {
    0: 0.018, 1: 0.045, 2: 0.038, 3: 0.032, 4: 0.031,
    5: 0.036, 6: 0.040, 7: 0.054, 8: 0.062, 9: 0.063,
    10: 0.081, 11: 0.083, 12: 0.080, 13: 0.107,
}

PAPER_DETREND = {
    0: 0.007, 1: 0.032, 2: 0.025, 3: 0.021, 4: 0.018,
    5: 0.017, 6: 0.019, 7: 0.036, 8: 0.041, 9: 0.041,
    10: 0.037, 11: 0.018, 12: 0.017, 13: 0.047,
}


def load_data():
    """加载 Walmart 数据"""
    data_path = Path(__file__).parent.parent / 'data' / 'walmart.csv'
    return pd.read_csv(data_path)


def transform_long_difference(df, y, ivar, tvar, gvar):
    """
    Long Difference 变换 (Callaway-Sant'Anna 风格)
    
    公式: Ŷ_{irg} = Y_{ir} - Y_{i,g-1}
    
    只使用 g-1 期作为基准，而不是所有 pre-treatment 期的均值
    """
    result = df.copy()
    result[tvar] = pd.to_numeric(result[tvar], errors='coerce')
    
    cohorts = sorted([g for g in df[gvar].unique() if pd.notna(g) and g != np.inf])
    T_max = int(result[tvar].max())
    
    for g in cohorts:
        g = int(g)
        base_period = g - 1  # 只用 g-1 期
        
        # 获取基准期的值
        base_data = result[result[tvar] == base_period].set_index(ivar)[y]
        
        for r in range(g, T_max + 1):
            col_name = f'yld_g{g}_r{r}'
            result[col_name] = np.nan
            
            period_mask = result[tvar] == r
            result.loc[period_mask, col_name] = (
                result.loc[period_mask, y].values -
                result.loc[period_mask, ivar].map(base_data).values
            )
    
    return result


def transform_full_demean(df, y, ivar, tvar, gvar):
    """
    Full Demean 变换 (Lee-Wooldridge 风格)
    
    公式: Ŷ_{irg} = Y_{ir} - (1/(g-1)) * Σ_{s=1}^{g-1} Y_{is}
    
    使用所有 pre-treatment 期的均值
    """
    from lwdid.staggered.transformations import transform_staggered_demean
    return transform_staggered_demean(df, y, ivar, tvar, gvar)


def compute_simple_watt(df_transformed, df_original, gvar, col_prefix, control_group='never_treated'):
    """计算简单差分 WATT (无协变量)"""
    cohorts = sorted([g for g in df_original[gvar].unique() if pd.notna(g) and g != np.inf])
    cohort_sizes = df_original[df_original[gvar] != np.inf].groupby(gvar)['fips'].nunique().to_dict()
    T_max = int(df_original['year'].max())
    
    att_results = []
    
    for g in cohorts:
        g = int(g)
        for r in range(g, T_max + 1):
            col_name = f'{col_prefix}_g{g}_r{r}'
            
            if col_name not in df_transformed.columns:
                continue
            
            period_data = df_transformed[df_transformed['year'] == r].copy()
            
            # 处理组
            treated_mask = period_data[gvar] == g
            
            # 控制组
            if control_group == 'never_treated':
                control_mask = period_data[gvar] == np.inf
            else:
                control_mask = (period_data[gvar] > r) | (period_data[gvar] == np.inf)
            
            treated_vals = period_data[treated_mask][col_name].dropna()
            control_vals = period_data[control_mask][col_name].dropna()
            
            if len(treated_vals) == 0 or len(control_vals) == 0:
                continue
            
            att = treated_vals.mean() - control_vals.mean()
            
            att_results.append({
                'cohort': g,
                'period': r,
                'event_time': r - g,
                'att': att,
            })
    
    att_df = pd.DataFrame(att_results)
    
    # 计算 WATT
    watt_list = []
    for event_time in sorted(att_df['event_time'].unique()):
        subset = att_df[att_df['event_time'] == event_time].copy()
        subset = subset[subset['att'].notna()]
        
        if len(subset) == 0:
            continue
        
        subset['weight'] = subset['cohort'].map(cohort_sizes)
        total_weight = subset['weight'].sum()
        subset['norm_weight'] = subset['weight'] / total_weight
        
        watt = (subset['att'] * subset['norm_weight']).sum()
        watt_list.append({
            'event_time': int(event_time),
            'watt': watt,
        })
    
    return pd.DataFrame(watt_list)


def compute_ipwra_watt(df, rolling_method, controls, control_group='never_treated'):
    """使用 IPWRA 估计器计算 WATT"""
    from lwdid import lwdid
    
    results = lwdid(
        data=df,
        y='log_retail_emp',
        ivar='fips',
        tvar='year',
        gvar='g',
        rolling=rolling_method,
        estimator='ipwra',
        controls=controls,
        control_group=control_group,
        aggregate='none',
    )
    
    att_ct = results.att_by_cohort_time.copy()
    cohort_sizes = df[df['g'] != np.inf].groupby('g')['fips'].nunique().to_dict()
    
    watt_list = []
    for event_time in sorted(att_ct['event_time'].unique()):
        subset = att_ct[att_ct['event_time'] == event_time].copy()
        subset = subset[subset['att'].notna()]
        
        if len(subset) == 0:
            continue
        
        subset['weight'] = subset['cohort'].map(cohort_sizes)
        total_weight = subset['weight'].sum()
        subset['norm_weight'] = subset['weight'] / total_weight
        
        watt = (subset['att'] * subset['norm_weight']).sum()
        watt_list.append({
            'event_time': int(event_time),
            'watt': watt,
        })
    
    return pd.DataFrame(watt_list)


def main():
    """运行诊断"""
    print("=" * 80)
    print("诊断：论文使用的变换方法")
    print("=" * 80)
    
    df = load_data()
    controls = ['share_pop_poverty_78_above', 'share_pop_ind_manuf', 'share_school_some_hs']
    
    # 1. Long Difference (简单差分)
    print("\n计算 Long Difference (简单差分)...")
    df_ld = transform_long_difference(df, 'log_retail_emp', 'fips', 'year', 'g')
    watt_ld_simple = compute_simple_watt(df_ld, df, 'g', 'yld', 'never_treated')
    
    # 2. Full Demean (简单差分)
    print("计算 Full Demean (简单差分)...")
    df_dm = transform_full_demean(df, 'log_retail_emp', 'fips', 'year', 'g')
    watt_dm_simple = compute_simple_watt(df_dm, df, 'g', 'ydot', 'never_treated')
    
    # 3. Full Demean (IPWRA)
    print("计算 Full Demean (IPWRA)...")
    watt_dm_ipwra = compute_ipwra_watt(df, 'demean', controls, 'never_treated')
    
    # 4. Detrend (IPWRA)
    print("计算 Detrend (IPWRA)...")
    watt_dt_ipwra = compute_ipwra_watt(df, 'detrend', controls, 'never_treated')
    
    # 比较结果
    print("\n" + "=" * 80)
    print("结果比较")
    print("=" * 80)
    
    print("\n--- Demean 方法比较 ---")
    print(f"{'r':>3} | {'LD Simple':>12} | {'DM Simple':>12} | {'DM IPWRA':>12} | {'Paper':>8} | {'LD Ratio':>10} | {'DM Ratio':>10}")
    print("-" * 90)
    
    ld_ratios = []
    dm_ratios = []
    
    for event_time in range(14):
        ld_val = watt_ld_simple[watt_ld_simple['event_time'] == event_time]['watt'].values
        dm_val = watt_dm_simple[watt_dm_simple['event_time'] == event_time]['watt'].values
        dm_ipwra_val = watt_dm_ipwra[watt_dm_ipwra['event_time'] == event_time]['watt'].values
        paper_val = PAPER_DEMEAN.get(event_time, np.nan)
        
        ld_str = f"{ld_val[0]:.4f}" if len(ld_val) > 0 else "N/A"
        dm_str = f"{dm_val[0]:.4f}" if len(dm_val) > 0 else "N/A"
        dm_ipwra_str = f"{dm_ipwra_val[0]:.4f}" if len(dm_ipwra_val) > 0 else "N/A"
        
        if len(ld_val) > 0 and paper_val > 0:
            ld_ratio = ld_val[0] / paper_val
            ld_ratios.append(ld_ratio)
            ld_ratio_str = f"{ld_ratio:.2f}x"
        else:
            ld_ratio_str = "N/A"
        
        if len(dm_val) > 0 and paper_val > 0:
            dm_ratio = dm_val[0] / paper_val
            dm_ratios.append(dm_ratio)
            dm_ratio_str = f"{dm_ratio:.2f}x"
        else:
            dm_ratio_str = "N/A"
        
        print(f"{event_time:>3} | {ld_str:>12} | {dm_str:>12} | {dm_ipwra_str:>12} | {paper_val:>8.3f} | {ld_ratio_str:>10} | {dm_ratio_str:>10}")
    
    print("-" * 90)
    print(f"平均比率: Long Diff = {np.mean(ld_ratios):.2f}x, Full Demean = {np.mean(dm_ratios):.2f}x")
    
    print("\n--- Detrend 方法比较 ---")
    print(f"{'r':>3} | {'DT IPWRA':>12} | {'Paper':>8} | {'Ratio':>10}")
    print("-" * 50)
    
    dt_ratios = []
    for event_time in range(14):
        dt_val = watt_dt_ipwra[watt_dt_ipwra['event_time'] == event_time]['watt'].values
        paper_val = PAPER_DETREND.get(event_time, np.nan)
        
        dt_str = f"{dt_val[0]:.4f}" if len(dt_val) > 0 else "N/A"
        
        if len(dt_val) > 0 and paper_val > 0:
            dt_ratio = dt_val[0] / paper_val
            dt_ratios.append(dt_ratio)
            dt_ratio_str = f"{dt_ratio:.2f}x"
        else:
            dt_ratio_str = "N/A"
        
        print(f"{event_time:>3} | {dt_str:>12} | {paper_val:>8.3f} | {dt_ratio_str:>10}")
    
    print("-" * 50)
    print(f"平均比率: Detrend = {np.mean(dt_ratios):.2f}x")
    
    # 结论
    print("\n" + "=" * 80)
    print("结论")
    print("=" * 80)
    
    print(f"\n1. Long Difference (简单差分) 平均比率: {np.mean(ld_ratios):.2f}x")
    print(f"2. Full Demean (简单差分) 平均比率: {np.mean(dm_ratios):.2f}x")
    print(f"3. Detrend (IPWRA) 平均比率: {np.mean(dt_ratios):.2f}x")
    
    if np.mean(ld_ratios) < np.mean(dm_ratios):
        print("\n结论: Long Difference 更接近论文值")
        print("这表明论文可能使用的是 Long Difference (Y_r - Y_{g-1}) 而不是 Full Demean")
    else:
        print("\n结论: Full Demean 更接近论文值")
    
    # 检查是否有其他可能的差异来源
    print("\n" + "=" * 80)
    print("进一步分析：可能的差异来源")
    print("=" * 80)
    
    print("\n1. 变换方法:")
    print("   - 论文可能使用 Long Difference (Y_r - Y_{g-1})")
    print("   - 当前实现使用 Full Demean (Y_r - mean(Y_{1:g-1}))")
    
    print("\n2. 控制组选择:")
    print("   - 论文可能使用 not_yet_treated + never_treated")
    print("   - 当前实现使用 never_treated only")
    
    print("\n3. IPWRA 估计器:")
    print("   - 论文可能使用不同的倾向得分模型规范")
    print("   - 论文可能使用不同的权重归一化方式")
    
    print("\n4. 标准误估计:")
    print("   - 论文使用 Bootstrap (100 次重复)")
    print("   - 当前实现使用解析标准误")


if __name__ == '__main__':
    main()

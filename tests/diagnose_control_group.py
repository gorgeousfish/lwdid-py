"""
诊断脚本：控制组选择的影响

测试不同控制组选择对结果的影响：
1. never_treated only
2. not_yet_treated + never_treated
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


def load_data():
    """加载 Walmart 数据"""
    data_path = Path(__file__).parent.parent / 'data' / 'walmart.csv'
    return pd.read_csv(data_path)


def transform_long_difference(df, y, ivar, tvar, gvar):
    """Long Difference 变换"""
    result = df.copy()
    result[tvar] = pd.to_numeric(result[tvar], errors='coerce')
    
    cohorts = sorted([g for g in df[gvar].unique() if pd.notna(g) and g != np.inf])
    T_max = int(result[tvar].max())
    
    for g in cohorts:
        g = int(g)
        base_period = g - 1
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


def compute_watt_with_control_group(df_transformed, df_original, gvar, col_prefix, control_group):
    """计算 WATT，使用指定的控制组"""
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
            else:  # not_yet_treated
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
                'n_treated': len(treated_vals),
                'n_control': len(control_vals),
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


def main():
    """运行诊断"""
    print("=" * 80)
    print("诊断：控制组选择的影响")
    print("=" * 80)
    
    df = load_data()
    
    # Long Difference 变换
    print("\n计算 Long Difference 变换...")
    df_ld = transform_long_difference(df, 'log_retail_emp', 'fips', 'year', 'g')
    
    # 两种控制组
    print("计算 never_treated 控制组...")
    watt_nt = compute_watt_with_control_group(df_ld, df, 'g', 'yld', 'never_treated')
    
    print("计算 not_yet_treated 控制组...")
    watt_nyt = compute_watt_with_control_group(df_ld, df, 'g', 'yld', 'not_yet_treated')
    
    # 比较结果
    print("\n" + "=" * 80)
    print("Long Difference + 不同控制组")
    print("=" * 80)
    
    print(f"\n{'r':>3} | {'NT':>10} | {'NYT':>10} | {'Paper':>8} | {'NT Ratio':>10} | {'NYT Ratio':>10}")
    print("-" * 70)
    
    nt_ratios = []
    nyt_ratios = []
    
    for event_time in range(14):
        nt_val = watt_nt[watt_nt['event_time'] == event_time]['watt'].values
        nyt_val = watt_nyt[watt_nyt['event_time'] == event_time]['watt'].values
        paper_val = PAPER_DEMEAN.get(event_time, np.nan)
        
        nt_str = f"{nt_val[0]:.4f}" if len(nt_val) > 0 else "N/A"
        nyt_str = f"{nyt_val[0]:.4f}" if len(nyt_val) > 0 else "N/A"
        
        if len(nt_val) > 0 and paper_val > 0:
            nt_ratio = nt_val[0] / paper_val
            nt_ratios.append(nt_ratio)
            nt_ratio_str = f"{nt_ratio:.2f}x"
        else:
            nt_ratio_str = "N/A"
        
        if len(nyt_val) > 0 and paper_val > 0:
            nyt_ratio = nyt_val[0] / paper_val
            nyt_ratios.append(nyt_ratio)
            nyt_ratio_str = f"{nyt_ratio:.2f}x"
        else:
            nyt_ratio_str = "N/A"
        
        print(f"{event_time:>3} | {nt_str:>10} | {nyt_str:>10} | {paper_val:>8.3f} | {nt_ratio_str:>10} | {nyt_ratio_str:>10}")
    
    print("-" * 70)
    print(f"平均比率: never_treated = {np.mean(nt_ratios):.2f}x, not_yet_treated = {np.mean(nyt_ratios):.2f}x")
    
    # 检查控制组样本量
    print("\n" + "=" * 80)
    print("控制组样本量分析")
    print("=" * 80)
    
    # 统计 never_treated 和 not_yet_treated 的样本量
    n_never_treated = (df['g'] == np.inf).sum() // 23  # 除以年数
    print(f"\nNever-treated 县数量: {n_never_treated}")
    
    # 检查各 cohort 的 not_yet_treated 样本量
    cohorts = sorted([g for g in df['g'].unique() if pd.notna(g) and g != np.inf])
    
    print("\n各 cohort 在 event_time=0 时的控制组样本量:")
    print(f"{'Cohort':>8} | {'NT':>8} | {'NYT':>8}")
    print("-" * 30)
    
    for g in cohorts[:10]:
        g = int(g)
        r = g  # event_time = 0
        
        period_data = df[df['year'] == r]
        n_nt = (period_data['g'] == np.inf).sum()
        n_nyt = ((period_data['g'] > r) | (period_data['g'] == np.inf)).sum()
        
        print(f"{g:>8} | {n_nt:>8} | {n_nyt:>8}")
    
    # 结论
    print("\n" + "=" * 80)
    print("结论")
    print("=" * 80)
    
    print(f"\n1. never_treated 控制组平均比率: {np.mean(nt_ratios):.2f}x")
    print(f"2. not_yet_treated 控制组平均比率: {np.mean(nyt_ratios):.2f}x")
    
    if np.mean(nyt_ratios) < np.mean(nt_ratios):
        print("\n结论: not_yet_treated 控制组更接近论文值")
    else:
        print("\n结论: never_treated 控制组更接近论文值")


if __name__ == '__main__':
    main()

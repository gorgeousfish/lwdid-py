"""
诊断：测试不同的控制组定义

假设：论文可能只使用从未处理组作为控制组，而不是 "从未处理 + 尚未处理"

控制组定义选项：
1. never_only: 只使用从未处理组 (g == inf)
2. not_yet: 使用从未处理 + 尚未处理 (g > r | g == inf)
3. not_yet_strict: 使用尚未处理但不包括从未处理 (g > r & g != inf)
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


def compute_watt_with_control_group(df, method='demean', control_type='not_yet'):
    """
    使用指定的控制组定义计算 WATT
    
    Parameters:
    -----------
    method : str
        'demean' - Full Demean (Y_ir - mean(Y_{i,1:g-1}))
        'long_diff' - Long Difference (Y_ir - Y_{i,g-1})
    control_type : str
        'never_only' - 只使用从未处理组
        'not_yet' - 从未处理 + 尚未处理
        'not_yet_strict' - 只使用尚未处理（不包括从未处理）
    """
    T_min = int(df['year'].min())
    T_max = int(df['year'].max())
    cohorts = sorted([int(g) for g in df['g'].unique() if pd.notna(g) and g != np.inf])
    cohort_sizes = {int(k): v for k, v in df[df['g'] != np.inf].groupby('g')['fips'].nunique().to_dict().items()}
    
    # 预计算
    if method == 'demean':
        # 计算每个 cohort 的预处理期均值
        unit_pre_means = {}
        for g in cohorts:
            pre_mask = df['year'] < g
            pre_means = df[pre_mask].groupby('fips')['log_retail_emp'].mean()
            unit_pre_means[g] = pre_means
    elif method == 'long_diff':
        # 计算每个单元在每个时期的值
        unit_year_values = df.pivot_table(index='fips', columns='year', values='log_retail_emp')
    
    # 计算 ATT
    att_results = []
    
    for g in cohorts:
        for r in range(g, T_max + 1):
            period_data = df[df['year'] == r].copy()
            
            # 计算变换
            if method == 'demean':
                period_data['y_transformed'] = period_data.apply(
                    lambda row: row['log_retail_emp'] - unit_pre_means[g].get(row['fips'], np.nan),
                    axis=1
                )
            elif method == 'long_diff':
                base_period = g - 1
                if base_period not in unit_year_values.columns:
                    continue
                period_data['y_transformed'] = period_data.apply(
                    lambda row: row['log_retail_emp'] - unit_year_values.loc[row['fips'], base_period] 
                    if row['fips'] in unit_year_values.index and base_period in unit_year_values.columns 
                    else np.nan,
                    axis=1
                )
            
            # 定义处理组
            treated_mask = period_data['g'] == g
            
            # 定义控制组
            if control_type == 'never_only':
                control_mask = period_data['g'] == np.inf
            elif control_type == 'not_yet':
                control_mask = (period_data['g'] > r) | (period_data['g'] == np.inf)
            elif control_type == 'not_yet_strict':
                control_mask = (period_data['g'] > r) & (period_data['g'] != np.inf)
            else:
                raise ValueError(f"Unknown control_type: {control_type}")
            
            treated_vals = period_data[treated_mask]['y_transformed'].dropna()
            control_vals = period_data[control_mask]['y_transformed'].dropna()
            
            if len(treated_vals) == 0 or len(control_vals) == 0:
                continue
            
            att = treated_vals.mean() - control_vals.mean()
            
            att_results.append({
                'cohort': g, 'period': r, 'event_time': r - g, 'att': att,
                'n_treated': len(treated_vals), 'n_control': len(control_vals)
            })
    
    att_df = pd.DataFrame(att_results)
    
    # 计算 WATT
    watt_results = {}
    for event_time in sorted(att_df['event_time'].unique()):
        if event_time < 0:
            continue
        subset = att_df[att_df['event_time'] == event_time].copy()
        subset = subset[subset['att'].notna()]
        if len(subset) == 0:
            continue
        subset['weight'] = subset['cohort'].map(cohort_sizes)
        total_weight = subset['weight'].sum()
        subset['norm_weight'] = subset['weight'] / total_weight
        watt = (subset['att'] * subset['norm_weight']).sum()
        watt_results[int(event_time)] = watt
    
    return watt_results


def main():
    print("=" * 120)
    print("诊断：测试不同的控制组定义")
    print("=" * 120)
    
    df = load_data()
    
    # 检查从未处理组的数量
    n_never = df[df['g'] == np.inf]['fips'].nunique()
    n_total = df['fips'].nunique()
    print(f"\n数据信息:")
    print(f"  总单元数: {n_total}")
    print(f"  从未处理单元数: {n_never}")
    print(f"  处理单元数: {n_total - n_never}")
    
    # 测试不同的控制组定义
    print("\n" + "=" * 120)
    print("测试 Full Demean 方法")
    print("=" * 120)
    
    watt_demean_never = compute_watt_with_control_group(df, method='demean', control_type='never_only')
    watt_demean_notyet = compute_watt_with_control_group(df, method='demean', control_type='not_yet')
    
    print(f"\n{'r':>3} | {'Never Only':>12} | {'Not Yet':>12} | {'Paper':>8} | {'Never/Paper':>12} | {'NotYet/Paper':>12}")
    print("-" * 80)
    
    for r in range(14):
        wn = watt_demean_never.get(r, np.nan)
        wny = watt_demean_notyet.get(r, np.nan)
        paper = PAPER_DEMEAN.get(r, np.nan)
        
        ratio_n = wn / paper if paper > 0 else np.nan
        ratio_ny = wny / paper if paper > 0 else np.nan
        
        print(f"{r:>3} | {wn:>12.4f} | {wny:>12.4f} | {paper:>8.3f} | {ratio_n:>12.2f}x | {ratio_ny:>12.2f}x")
    
    # 测试 Long Difference
    print("\n" + "=" * 120)
    print("测试 Long Difference 方法")
    print("=" * 120)
    
    watt_ld_never = compute_watt_with_control_group(df, method='long_diff', control_type='never_only')
    watt_ld_notyet = compute_watt_with_control_group(df, method='long_diff', control_type='not_yet')
    
    print(f"\n{'r':>3} | {'Never Only':>12} | {'Not Yet':>12} | {'Paper':>8} | {'Never/Paper':>12} | {'NotYet/Paper':>12}")
    print("-" * 80)
    
    for r in range(14):
        wn = watt_ld_never.get(r, np.nan)
        wny = watt_ld_notyet.get(r, np.nan)
        paper = PAPER_DEMEAN.get(r, np.nan)
        
        ratio_n = wn / paper if paper > 0 else np.nan
        ratio_ny = wny / paper if paper > 0 else np.nan
        
        print(f"{r:>3} | {wn:>12.4f} | {wny:>12.4f} | {paper:>8.3f} | {ratio_n:>12.2f}x | {ratio_ny:>12.2f}x")
    
    # 计算平均比率
    print("\n" + "=" * 120)
    print("平均比率总结")
    print("=" * 120)
    
    def calc_avg_ratio(watt_dict):
        ratios = [watt_dict.get(r, np.nan) / PAPER_DEMEAN.get(r, 1) 
                  for r in range(14) if r in watt_dict and r in PAPER_DEMEAN]
        return np.nanmean(ratios)
    
    print(f"\nFull Demean:")
    print(f"  Never Only: {calc_avg_ratio(watt_demean_never):.2f}x")
    print(f"  Not Yet: {calc_avg_ratio(watt_demean_notyet):.2f}x")
    
    print(f"\nLong Difference:")
    print(f"  Never Only: {calc_avg_ratio(watt_ld_never):.2f}x")
    print(f"  Not Yet: {calc_avg_ratio(watt_ld_notyet):.2f}x")


if __name__ == '__main__':
    main()

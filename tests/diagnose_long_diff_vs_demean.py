"""
诊断：验证论文 Table A4 的 "Demean" 列是否实际上是 Long Difference

假设：论文的 "Demean" 列可能使用的是 Long Difference (Y_ir - Y_{i,g-1})
而不是 Full Demean (Y_ir - mean(Y_{i,1:g-1}))

这可以解释为什么我们的 Full Demean 结果比论文高 2.37 倍
"""

import sys
from pathlib import Path
import warnings
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
warnings.filterwarnings('ignore')

# 论文 Table A4 参考值
PAPER_VALUES = {
    'demean': {
        0: 0.018, 1: 0.045, 2: 0.038, 3: 0.032, 4: 0.031,
        5: 0.036, 6: 0.040, 7: 0.054, 8: 0.062, 9: 0.063,
        10: 0.081, 11: 0.083, 12: 0.080, 13: 0.107,
    },
    'detrend': {
        0: 0.007, 1: 0.032, 2: 0.025, 3: 0.021, 4: 0.018,
        5: 0.017, 6: 0.019, 7: 0.036, 8: 0.041, 9: 0.041,
        10: 0.037, 11: 0.018, 12: 0.017, 13: 0.047,
    }
}


def load_data():
    """加载 Walmart 数据"""
    data_path = Path(__file__).parent.parent / 'data' / 'walmart.csv'
    return pd.read_csv(data_path)


def compute_long_difference_watt():
    """
    计算 Long Difference 方法的 WATT
    
    Long Difference: Y_ir - Y_{i,g-1}
    即只使用处理前一期作为基准
    """
    df = load_data()
    
    T_min = int(df['year'].min())
    T_max = int(df['year'].max())
    cohorts = sorted([int(g) for g in df['g'].unique() if pd.notna(g) and g != np.inf])
    cohort_sizes = {int(k): v for k, v in df[df['g'] != np.inf].groupby('g')['fips'].nunique().to_dict().items()}
    
    # 计算每个单元在每个时期的值
    unit_year_values = df.pivot_table(index='fips', columns='year', values='log_retail_emp')
    
    # 计算 ATT
    att_results = []
    
    for g in cohorts:
        base_period = g - 1  # Long difference 使用 g-1 期作为基准
        
        if base_period not in unit_year_values.columns:
            continue
        
        for r in range(g, T_max + 1):
            if r not in unit_year_values.columns:
                continue
            
            period_data = df[df['year'] == r].copy()
            
            # Long Difference: Y_ir - Y_{i,g-1}
            def compute_long_diff(row):
                unit_id = row['fips']
                if unit_id not in unit_year_values.index:
                    return np.nan
                y_r = unit_year_values.loc[unit_id, r]
                y_base = unit_year_values.loc[unit_id, base_period]
                if pd.isna(y_r) or pd.isna(y_base):
                    return np.nan
                return y_r - y_base
            
            period_data['y_long_diff'] = period_data.apply(compute_long_diff, axis=1)
            
            # 处理组和控制组
            treated_mask = period_data['g'] == g
            control_mask = (period_data['g'] > r) | (period_data['g'] == np.inf)
            
            treated_vals = period_data[treated_mask]['y_long_diff'].dropna()
            control_vals = period_data[control_mask]['y_long_diff'].dropna()
            
            if len(treated_vals) == 0 or len(control_vals) == 0:
                continue
            
            att = treated_vals.mean() - control_vals.mean()
            
            att_results.append({
                'cohort': g, 'period': r, 'event_time': r - g, 'att': att
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


def compute_full_demean_watt():
    """
    计算 Full Demean 方法的 WATT
    
    Full Demean: Y_ir - mean(Y_{i,1:g-1})
    使用所有预处理期的均值
    """
    df = load_data()
    
    T_min = int(df['year'].min())
    T_max = int(df['year'].max())
    cohorts = sorted([int(g) for g in df['g'].unique() if pd.notna(g) and g != np.inf])
    cohort_sizes = {int(k): v for k, v in df[df['g'] != np.inf].groupby('g')['fips'].nunique().to_dict().items()}
    
    # 计算每个单元的预处理期均值（对于每个 cohort g）
    unit_pre_means = {}
    for g in cohorts:
        pre_mask = df['year'] < g
        pre_means = df[pre_mask].groupby('fips')['log_retail_emp'].mean()
        unit_pre_means[g] = pre_means
    
    # 计算 ATT
    att_results = []
    
    for g in cohorts:
        for r in range(g, T_max + 1):
            period_data = df[df['year'] == r].copy()
            
            period_data['ydot'] = period_data.apply(
                lambda row: row['log_retail_emp'] - unit_pre_means[g].get(row['fips'], np.nan),
                axis=1
            )
            
            treated_mask = period_data['g'] == g
            control_mask = (period_data['g'] > r) | (period_data['g'] == np.inf)
            
            treated_vals = period_data[treated_mask]['ydot'].dropna()
            control_vals = period_data[control_mask]['ydot'].dropna()
            
            if len(treated_vals) == 0 or len(control_vals) == 0:
                continue
            
            att = treated_vals.mean() - control_vals.mean()
            
            att_results.append({
                'cohort': g, 'period': r, 'event_time': r - g, 'att': att
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
    print("=" * 100)
    print("诊断：验证论文 Table A4 的 'Demean' 列是否实际上是 Long Difference")
    print("=" * 100)
    
    print("\n计算 Long Difference WATT...")
    long_diff_watt = compute_long_difference_watt()
    
    print("计算 Full Demean WATT...")
    full_demean_watt = compute_full_demean_watt()
    
    print("\n" + "=" * 100)
    print("对比结果")
    print("=" * 100)
    
    print(f"\n{'r':>3} | {'Long Diff':>10} | {'Full Demean':>11} | {'Paper Demean':>12} | {'LD/Paper':>10} | {'FD/Paper':>10}")
    print("-" * 80)
    
    for r in range(14):
        ld = long_diff_watt.get(r, np.nan)
        fd = full_demean_watt.get(r, np.nan)
        paper = PAPER_VALUES['demean'].get(r, np.nan)
        
        ratio_ld = ld / paper if paper > 0 else np.nan
        ratio_fd = fd / paper if paper > 0 else np.nan
        
        print(f"{r:>3} | {ld:>10.4f} | {fd:>11.4f} | {paper:>12.3f} | {ratio_ld:>10.2f}x | {ratio_fd:>10.2f}x")
    
    print("\n" + "=" * 100)
    print("分析")
    print("=" * 100)
    
    # 计算平均比率
    ld_ratios = [long_diff_watt.get(r, np.nan) / PAPER_VALUES['demean'].get(r, 1) 
                 for r in range(14) if r in long_diff_watt and r in PAPER_VALUES['demean']]
    fd_ratios = [full_demean_watt.get(r, np.nan) / PAPER_VALUES['demean'].get(r, 1) 
                 for r in range(14) if r in full_demean_watt and r in PAPER_VALUES['demean']]
    
    print(f"\nLong Difference 平均比率: {np.nanmean(ld_ratios):.2f}x")
    print(f"Full Demean 平均比率: {np.nanmean(fd_ratios):.2f}x")
    
    print("\n关键发现:")
    if abs(np.nanmean(ld_ratios) - 1.0) < abs(np.nanmean(fd_ratios) - 1.0):
        print("✓ Long Difference 更接近论文的 'Demean' 列")
        print("  这表明论文的 'Demean' 实际上是 Long Difference (Y_ir - Y_{i,g-1})")
        print("  而不是 Full Demean (Y_ir - mean(Y_{i,1:g-1}))")
    else:
        print("✓ Full Demean 更接近论文的 'Demean' 列")
        print("  这表明论文的 'Demean' 确实是 Full Demean")
        print("  需要进一步调查差异原因")


if __name__ == '__main__':
    main()

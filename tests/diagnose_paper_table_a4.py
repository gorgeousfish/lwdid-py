"""
精确诊断：验证论文 Table A4 的实际内容

论文 Table A4 标题: "Rolling IPWRA with Heterogeneous Trends"
- 这表明使用的是 Detrend 方法，而非 Demean

需要验证：
1. 论文 Table A4 报告的是 Demean 还是 Detrend 结果
2. 我们的 Detrend 实现是否正确
"""

import sys
from pathlib import Path
import warnings
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
warnings.filterwarnings('ignore')

# 论文 Table A4 参考值 - 需要确认这是 Demean 还是 Detrend
# 根据需求文档，Table A4 标题是 "Rolling IPWRA with Heterogeneous Trends"
# "Heterogeneous Trends" 暗示使用的是 Detrend 方法

# 从之前的诊断结果中提取的论文值
PAPER_VALUES = {
    # 这些值来自论文 Table A4
    # 需要确认是 Demean 还是 Detrend
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


def compute_detrend_watt():
    """计算 Detrend 方法的 WATT"""
    df = load_data()
    
    T_min = int(df['year'].min())
    T_max = int(df['year'].max())
    cohorts = sorted([int(g) for g in df['g'].unique() if pd.notna(g) and g != np.inf])
    cohort_sizes = {int(k): v for k, v in df[df['g'] != np.inf].groupby('g')['fips'].nunique().to_dict().items()}
    
    # 计算每个单元的线性趋势参数
    unit_trends = {}  # {(unit_id, cohort): (intercept, slope)}
    
    for g in cohorts:
        for unit_id in df['fips'].unique():
            unit_data = df[df['fips'] == unit_id].sort_values('year')
            pre_data = unit_data[unit_data['year'] < g].dropna(subset=['log_retail_emp'])
            
            if len(pre_data) < 2:
                continue
            
            t_vals = pre_data['year'].values.astype(float)
            y_vals = pre_data['log_retail_emp'].values.astype(float)
            
            try:
                B, A = np.polyfit(t_vals, y_vals, deg=1)
                unit_trends[(unit_id, g)] = (A, B)
            except:
                continue
    
    # 计算 ATT
    att_results = []
    
    for g in cohorts:
        for r in range(g, T_max + 1):
            period_data = df[df['year'] == r].copy()
            
            # 计算 detrend 变换
            def compute_ycheck(row):
                key = (row['fips'], g)
                if key not in unit_trends:
                    return np.nan
                A, B = unit_trends[key]
                predicted = A + B * r
                return row['log_retail_emp'] - predicted
            
            period_data['ycheck'] = period_data.apply(compute_ycheck, axis=1)
            
            # 处理组和控制组
            treated_mask = period_data['g'] == g
            control_mask = (period_data['g'] > r) | (period_data['g'] == np.inf)
            
            treated_vals = period_data[treated_mask]['ycheck'].dropna()
            control_vals = period_data[control_mask]['ycheck'].dropna()
            
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


def compute_demean_watt():
    """计算 Demean 方法的 WATT"""
    df = load_data()
    
    T_min = int(df['year'].min())
    T_max = int(df['year'].max())
    cohorts = sorted([int(g) for g in df['g'].unique() if pd.notna(g) and g != np.inf])
    cohort_sizes = {int(k): v for k, v in df[df['g'] != np.inf].groupby('g')['fips'].nunique().to_dict().items()}
    
    # 计算每个单元的预处理期均值
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
    print("=" * 80)
    print("精确诊断：验证论文 Table A4 的实际内容")
    print("=" * 80)
    
    print("\n计算 Demean WATT...")
    demean_watt = compute_demean_watt()
    
    print("计算 Detrend WATT...")
    detrend_watt = compute_detrend_watt()
    
    print("\n" + "=" * 80)
    print("对比结果")
    print("=" * 80)
    
    print(f"\n{'r':>3} | {'Demean':>10} | {'Detrend':>10} | {'Paper Demean':>12} | {'Paper Detrend':>13} | {'Demean/PD':>10} | {'Detrend/PT':>10}")
    print("-" * 95)
    
    for r in range(14):
        dm = demean_watt.get(r, np.nan)
        dt = detrend_watt.get(r, np.nan)
        pd_val = PAPER_VALUES['demean'].get(r, np.nan)
        pt_val = PAPER_VALUES['detrend'].get(r, np.nan)
        
        ratio_dm = dm / pd_val if pd_val > 0 else np.nan
        ratio_dt = dt / pt_val if pt_val > 0 else np.nan
        
        print(f"{r:>3} | {dm:>10.4f} | {dt:>10.4f} | {pd_val:>12.3f} | {pt_val:>13.3f} | {ratio_dm:>10.2f}x | {ratio_dt:>10.2f}x")
    
    print("\n" + "=" * 80)
    print("分析")
    print("=" * 80)
    
    # 计算平均比率
    demean_ratios = [demean_watt.get(r, np.nan) / PAPER_VALUES['demean'].get(r, 1) 
                    for r in range(14) if r in demean_watt and r in PAPER_VALUES['demean']]
    detrend_ratios = [detrend_watt.get(r, np.nan) / PAPER_VALUES['detrend'].get(r, 1) 
                     for r in range(14) if r in detrend_watt and r in PAPER_VALUES['detrend']]
    
    print(f"\nDemean 平均比率: {np.nanmean(demean_ratios):.2f}x")
    print(f"Detrend 平均比率: {np.nanmean(detrend_ratios):.2f}x")
    
    print("\n关键发现:")
    print("1. 如果 Detrend 比率接近 1.0，说明论文 Table A4 报告的是 Detrend 结果")
    print("2. 如果 Demean 比率接近 1.0，说明论文 Table A4 报告的是 Demean 结果")
    print("3. 如果两者都不接近 1.0，说明存在其他差异")


if __name__ == '__main__':
    main()

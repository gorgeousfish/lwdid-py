"""
诊断：论文公式的正确解释

论文公式 2.2:
Y_bar_{i,pre(g)} = (1/(g-1)) * sum_{s=1}^{g-1} Y_{is}

问题：
- 论文假设时间从 t=1 开始
- Walmart 数据时间从 t=1977 开始
- 分母应该是 (g - T_min) 而不是 (g - 1)

但实际上，pandas 的 mean() 函数会自动计算正确的均值，
所以这不应该是问题。

让我检查其他可能的问题。
"""

import sys
from pathlib import Path
import warnings
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
warnings.filterwarnings('ignore')

PAPER_DEMEAN = {
    0: 0.018, 1: 0.045, 2: 0.038, 3: 0.032, 4: 0.031,
    5: 0.036, 6: 0.040, 7: 0.054, 8: 0.062, 9: 0.063,
    10: 0.081, 11: 0.083, 12: 0.080, 13: 0.107,
}


def load_data():
    data_path = Path(__file__).parent.parent / 'data' / 'walmart.csv'
    return pd.read_csv(data_path)


def check_pre_treatment_periods():
    """检查预处理期的数量"""
    df = load_data()
    
    T_min = int(df['year'].min())
    T_max = int(df['year'].max())
    
    print("=" * 80)
    print("检查预处理期的数量")
    print("=" * 80)
    
    print(f"\n时间范围: {T_min} - {T_max}")
    
    cohorts = sorted([int(g) for g in df['g'].unique() if pd.notna(g) and g != np.inf])
    
    print(f"\n{'Cohort':>8} | {'Pre-periods':>12} | {'论文公式 (g-1)':>15}")
    print("-" * 45)
    
    for g in cohorts:
        n_pre = g - T_min  # 实际预处理期数量
        paper_formula = g - 1  # 论文公式中的分母
        print(f"{g:>8} | {n_pre:>12} | {paper_formula:>15}")
    
    print("\n结论: 论文公式假设 T_min = 1，但 Walmart 数据 T_min = 1977")
    print("这不影响均值计算，因为 pandas mean() 会自动处理")


def check_simple_difference():
    """检查简单差分（不使用变换）"""
    df = load_data()
    
    print("\n" + "=" * 80)
    print("检查简单差分（不使用变换）")
    print("=" * 80)
    
    T_max = int(df['year'].max())
    cohorts = sorted([int(g) for g in df['g'].unique() if pd.notna(g) and g != np.inf])
    cohort_sizes = {int(k): v for k, v in df[df['g'] != np.inf].groupby('g')['fips'].nunique().to_dict().items()}
    
    # 计算简单差分 ATT（不使用任何变换）
    att_results = []
    
    for g in cohorts:
        for r in range(g, T_max + 1):
            event_time = r - g
            period_data = df[df['year'] == r].copy()
            
            # 处理组和控制组的原始 Y 值
            treated_mask = period_data['g'] == g
            control_mask = (period_data['g'] > r) | (period_data['g'] == np.inf)
            
            treated_y = period_data[treated_mask]['log_retail_emp'].mean()
            control_y = period_data[control_mask]['log_retail_emp'].mean()
            
            att = treated_y - control_y
            
            att_results.append({
                'cohort': g, 'period': r, 'event_time': event_time, 
                'att': att, 'treated_y': treated_y, 'control_y': control_y
            })
    
    att_df = pd.DataFrame(att_results)
    
    # 计算 WATT
    watt_results = {}
    for event_time in sorted(att_df['event_time'].unique()):
        if event_time < 0:
            continue
        subset = att_df[att_df['event_time'] == event_time].copy()
        subset['weight'] = subset['cohort'].map(cohort_sizes)
        total_weight = subset['weight'].sum()
        subset['norm_weight'] = subset['weight'] / total_weight
        watt = (subset['att'] * subset['norm_weight']).sum()
        watt_results[int(event_time)] = watt
    
    print("\n简单差分 WATT（不使用变换）:")
    print(f"{'r':>3} | {'WATT':>10} | {'Paper':>8}")
    print("-" * 30)
    
    for r in range(14):
        watt = watt_results.get(r, np.nan)
        paper = PAPER_DEMEAN.get(r, np.nan)
        print(f"{r:>3} | {watt:>10.4f} | {paper:>8.3f}")
    
    return watt_results


def check_long_difference_detailed():
    """详细检查 Long Difference 方法"""
    df = load_data()
    
    print("\n" + "=" * 80)
    print("详细检查 Long Difference 方法")
    print("=" * 80)
    
    T_max = int(df['year'].max())
    cohorts = sorted([int(g) for g in df['g'].unique() if pd.notna(g) and g != np.inf])
    cohort_sizes = {int(k): v for k, v in df[df['g'] != np.inf].groupby('g')['fips'].nunique().to_dict().items()}
    
    # Long Difference: Y_ir - Y_{i,g-1}
    # 计算每个单元在 g-1 期的值
    unit_base = {}
    for g in cohorts:
        base_period = g - 1
        base_data = df[df['year'] == base_period].set_index('fips')['log_retail_emp']
        unit_base[g] = base_data
    
    # 计算 ATT
    att_results = []
    
    for g in cohorts:
        for r in range(g, T_max + 1):
            event_time = r - g
            period_data = df[df['year'] == r].copy()
            
            # Long Difference 变换
            period_data['yld'] = period_data.apply(
                lambda row: row['log_retail_emp'] - unit_base[g].get(row['fips'], np.nan),
                axis=1
            )
            
            treated_mask = period_data['g'] == g
            control_mask = (period_data['g'] > r) | (period_data['g'] == np.inf)
            
            treated_vals = period_data[treated_mask]['yld'].dropna()
            control_vals = period_data[control_mask]['yld'].dropna()
            
            if len(treated_vals) == 0 or len(control_vals) == 0:
                continue
            
            att = treated_vals.mean() - control_vals.mean()
            
            att_results.append({
                'cohort': g, 'period': r, 'event_time': event_time, 'att': att,
                'treated_mean': treated_vals.mean(), 'control_mean': control_vals.mean(),
            })
    
    att_df = pd.DataFrame(att_results)
    
    # 计算 WATT
    watt_results = {}
    for event_time in sorted(att_df['event_time'].unique()):
        if event_time < 0:
            continue
        subset = att_df[att_df['event_time'] == event_time].copy()
        subset['weight'] = subset['cohort'].map(cohort_sizes)
        total_weight = subset['weight'].sum()
        subset['norm_weight'] = subset['weight'] / total_weight
        watt = (subset['att'] * subset['norm_weight']).sum()
        watt_results[int(event_time)] = watt
    
    print("\nLong Difference WATT:")
    print(f"{'r':>3} | {'WATT':>10} | {'Paper':>8} | {'Ratio':>8}")
    print("-" * 40)
    
    for r in range(14):
        watt = watt_results.get(r, np.nan)
        paper = PAPER_DEMEAN.get(r, np.nan)
        ratio = watt / paper if paper > 0 else np.nan
        print(f"{r:>3} | {watt:>10.4f} | {paper:>8.3f} | {ratio:>8.2f}x")
    
    # 显示 r=0 的详细信息
    print("\n--- r=0 详细信息 ---")
    r0_data = att_df[att_df['event_time'] == 0]
    for _, row in r0_data.head(5).iterrows():
        print(f"  g={int(row['cohort'])}: ATT={row['att']:.4f}, treated_mean={row['treated_mean']:.4f}, control_mean={row['control_mean']:.4f}")
    
    return watt_results


def main():
    check_pre_treatment_periods()
    check_simple_difference()
    check_long_difference_detailed()
    
    print("\n" + "=" * 80)
    print("诊断结论")
    print("=" * 80)
    print("""
关键发现：
1. Long Difference 方法的比率约为 1.2x - 1.8x
2. Full Demean 方法的比率约为 2.0x - 2.8x
3. 简单差分（不使用变换）的结果与 Full Demean 类似

可能的原因：
1. 论文可能使用了不同的变换方法
2. 论文可能使用了不同的样本选择
3. 论文可能使用了不同的权重计算方式
""")


if __name__ == '__main__':
    main()

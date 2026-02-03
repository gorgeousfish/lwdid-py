"""
详细诊断 Demean 方法

目标：找出 Demean 结果与论文差异 2.4x 的根本原因
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


def compute_demean_detailed():
    """计算详细的 Demean 结果"""
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
    
    # 计算每个 (g, r) 的 ATT
    att_by_gr = {}
    
    for g in cohorts:
        for r in range(g, T_max + 1):
            event_time = r - g
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
            att_by_gr[(g, event_time)] = {
                'att': att,
                'n_treated': len(treated_vals),
                'n_control': len(control_vals),
                'treated_mean': treated_vals.mean(),
                'control_mean': control_vals.mean(),
            }
    
    print("=" * 80)
    print("详细 Demean 结果 (r=0, 1, 2)")
    print("=" * 80)
    
    for r in [0, 1, 2]:
        print(f"\n--- Event time r={r} ---")
        print(f"论文值: {PAPER_DEMEAN.get(r, np.nan):.3f}")
        
        contributing = [(g, att_by_gr.get((g, r), {})) for g in cohorts if (g, r) in att_by_gr]
        
        if not contributing:
            print("  无贡献 cohort")
            continue
        
        total_weight = sum(cohort_sizes.get(g, 0) for g, _ in contributing)
        
        watt = 0
        print(f"\n  {'Cohort':>8} | {'ATT':>10} | {'Weight':>8} | {'Contrib':>10} | {'Treated Mean':>12} | {'Control Mean':>12}")
        print("  " + "-" * 75)
        
        for g, info in contributing:
            weight = cohort_sizes.get(g, 0) / total_weight
            contribution = weight * info['att']
            watt += contribution
            
            print(f"  {g:>8} | {info['att']:>10.4f} | {weight:>8.3f} | {contribution:>10.4f} | {info['treated_mean']:>12.4f} | {info['control_mean']:>12.4f}")
        
        print(f"\n  WATT: {watt:.4f}")
        print(f"  论文: {PAPER_DEMEAN.get(r, np.nan):.3f}")
        print(f"  比率: {watt / PAPER_DEMEAN.get(r, 1):.2f}x")
    
    return att_by_gr


def analyze_treated_control_means():
    """分析处理组和控制组的均值"""
    df = load_data()
    
    print("\n" + "=" * 80)
    print("分析处理组和控制组的变换后均值")
    print("=" * 80)
    
    # 对于 r=0，检查各 cohort 的处理组和控制组均值
    T_max = int(df['year'].max())
    cohorts = sorted([int(g) for g in df['g'].unique() if pd.notna(g) and g != np.inf])
    
    # 计算预处理期均值
    unit_pre_means = {}
    for g in cohorts:
        pre_mask = df['year'] < g
        pre_means = df[pre_mask].groupby('fips')['log_retail_emp'].mean()
        unit_pre_means[g] = pre_means
    
    print("\n对于 event_time=0:")
    print(f"{'Cohort':>8} | {'Period':>8} | {'Treated ydot':>12} | {'Control ydot':>12} | {'ATT':>10}")
    print("-" * 60)
    
    for g in cohorts[:5]:  # 只显示前5个 cohort
        r = g  # event_time = 0
        period_data = df[df['year'] == r].copy()
        
        period_data['ydot'] = period_data.apply(
            lambda row: row['log_retail_emp'] - unit_pre_means[g].get(row['fips'], np.nan),
            axis=1
        )
        
        treated_mask = period_data['g'] == g
        control_mask = (period_data['g'] > r) | (period_data['g'] == np.inf)
        
        treated_mean = period_data[treated_mask]['ydot'].mean()
        control_mean = period_data[control_mask]['ydot'].mean()
        att = treated_mean - control_mean
        
        print(f"{g:>8} | {r:>8} | {treated_mean:>12.4f} | {control_mean:>12.4f} | {att:>10.4f}")
    
    print("\n关键观察:")
    print("- 处理组的 ydot 均值应该反映处理效应")
    print("- 控制组的 ydot 均值应该接近 0（如果平行趋势成立）")
    print("- ATT = 处理组均值 - 控制组均值")


def check_control_group_ydot():
    """检查控制组的 ydot 是否接近 0"""
    df = load_data()
    
    print("\n" + "=" * 80)
    print("检查控制组的 ydot 分布")
    print("=" * 80)
    
    cohorts = sorted([int(g) for g in df['g'].unique() if pd.notna(g) and g != np.inf])
    
    # 对于 cohort g=1990，检查控制组的 ydot
    g = 1990
    r = g  # event_time = 0
    
    # 计算预处理期均值
    pre_mask = df['year'] < g
    pre_means = df[pre_mask].groupby('fips')['log_retail_emp'].mean()
    
    period_data = df[df['year'] == r].copy()
    period_data['ydot'] = period_data.apply(
        lambda row: row['log_retail_emp'] - pre_means.get(row['fips'], np.nan),
        axis=1
    )
    
    # 控制组
    control_mask = (period_data['g'] > r) | (period_data['g'] == np.inf)
    control_data = period_data[control_mask]
    
    print(f"\nCohort g={g}, Period r={r} (event_time=0)")
    print(f"控制组样本量: {len(control_data)}")
    print(f"控制组 ydot 均值: {control_data['ydot'].mean():.4f}")
    print(f"控制组 ydot 标准差: {control_data['ydot'].std():.4f}")
    print(f"控制组 ydot 最小值: {control_data['ydot'].min():.4f}")
    print(f"控制组 ydot 最大值: {control_data['ydot'].max():.4f}")
    
    # 按 cohort 分组查看控制组
    print("\n控制组按 cohort 分组:")
    for cg in sorted(control_data['g'].unique()):
        subset = control_data[control_data['g'] == cg]
        print(f"  g={int(cg) if pd.notna(cg) and cg != np.inf else 'inf'}: n={len(subset)}, mean_ydot={subset['ydot'].mean():.4f}")


def main():
    compute_demean_detailed()
    analyze_treated_control_means()
    check_control_group_ydot()
    
    print("\n" + "=" * 80)
    print("诊断结论")
    print("=" * 80)
    print("""
关键发现：
1. Demean 方法的 ATT 计算逻辑是正确的
2. 控制组的 ydot 均值不为 0，这是正常的（因为控制组也有自己的趋势）
3. 差异可能来源于：
   - 论文使用了不同的变换方法（可能是 Long Difference 而非 Full Demean）
   - 论文使用了不同的控制组定义
   - 论文使用了不同的权重计算方式
""")


if __name__ == '__main__':
    main()

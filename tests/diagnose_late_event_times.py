"""
诊断：为什么 r=10, 11, 12, 13 的 Detrend 结果偏离论文

可能的原因：
1. 这些 event time 只有少数 cohort 贡献
2. 权重计算可能有问题
3. 论文可能使用了不同的处理方式
"""

import sys
from pathlib import Path
import warnings
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
warnings.filterwarnings('ignore')

PAPER_DETREND = {
    0: 0.007, 1: 0.032, 2: 0.025, 3: 0.021, 4: 0.018,
    5: 0.017, 6: 0.019, 7: 0.036, 8: 0.041, 9: 0.041,
    10: 0.037, 11: 0.018, 12: 0.017, 13: 0.047,
}


def load_data():
    data_path = Path(__file__).parent.parent / 'data' / 'walmart.csv'
    return pd.read_csv(data_path)


def analyze_event_time_contributions():
    """分析每个 event time 的 cohort 贡献"""
    df = load_data()
    
    T_min = int(df['year'].min())
    T_max = int(df['year'].max())
    cohorts = sorted([int(g) for g in df['g'].unique() if pd.notna(g) and g != np.inf])
    cohort_sizes = {int(k): v for k, v in df[df['g'] != np.inf].groupby('g')['fips'].nunique().to_dict().items()}
    
    print("=" * 80)
    print("分析每个 event time 的 cohort 贡献")
    print("=" * 80)
    
    print(f"\nCohorts: {cohorts}")
    print(f"T_max: {T_max}")
    
    # 对于每个 event time，找出哪些 cohort 可以贡献
    for r in range(14):
        contributing_cohorts = []
        for g in cohorts:
            # cohort g 在 event_time r 时的 calendar time 是 g + r
            calendar_time = g + r
            if calendar_time <= T_max:
                contributing_cohorts.append(g)
        
        total_weight = sum(cohort_sizes.get(g, 0) for g in contributing_cohorts)
        
        print(f"\nEvent time r={r}:")
        print(f"  贡献 cohorts: {contributing_cohorts}")
        print(f"  cohort 数量: {len(contributing_cohorts)}")
        print(f"  总权重: {total_weight}")
        
        if len(contributing_cohorts) <= 3:
            print(f"  ⚠️ 只有 {len(contributing_cohorts)} 个 cohort 贡献，结果可能不稳定")
            for g in contributing_cohorts:
                print(f"    - g={g}: {cohort_sizes.get(g, 0)} 县, calendar_time={g+r}")


def compute_detailed_detrend():
    """计算详细的 Detrend 结果，包括每个 cohort 的贡献"""
    df = load_data()
    
    T_min = int(df['year'].min())
    T_max = int(df['year'].max())
    cohorts = sorted([int(g) for g in df['g'].unique() if pd.notna(g) and g != np.inf])
    cohort_sizes = {int(k): v for k, v in df[df['g'] != np.inf].groupby('g')['fips'].nunique().to_dict().items()}
    
    # 计算每个单元的线性趋势参数
    unit_trends = {}
    
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
    
    # 计算每个 (g, r) 的 ATT
    att_by_gr = {}
    
    for g in cohorts:
        for r in range(g, T_max + 1):
            event_time = r - g
            period_data = df[df['year'] == r].copy()
            
            def compute_ycheck(row):
                key = (row['fips'], g)
                if key not in unit_trends:
                    return np.nan
                A, B = unit_trends[key]
                predicted = A + B * r
                return row['log_retail_emp'] - predicted
            
            period_data['ycheck'] = period_data.apply(compute_ycheck, axis=1)
            
            treated_mask = period_data['g'] == g
            control_mask = (period_data['g'] > r) | (period_data['g'] == np.inf)
            
            treated_vals = period_data[treated_mask]['ycheck'].dropna()
            control_vals = period_data[control_mask]['ycheck'].dropna()
            
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
    
    print("\n" + "=" * 80)
    print("详细 Detrend 结果 (r=10, 11, 12, 13)")
    print("=" * 80)
    
    for r in [10, 11, 12, 13]:
        print(f"\n--- Event time r={r} ---")
        print(f"论文值: {PAPER_DETREND.get(r, np.nan):.3f}")
        
        # 找出贡献的 cohorts
        contributing = [(g, att_by_gr.get((g, r), {})) for g in cohorts if (g, r) in att_by_gr]
        
        if not contributing:
            print("  无贡献 cohort")
            continue
        
        total_weight = sum(cohort_sizes.get(g, 0) for g, _ in contributing)
        
        watt = 0
        for g, info in contributing:
            weight = cohort_sizes.get(g, 0) / total_weight
            contribution = weight * info['att']
            watt += contribution
            
            print(f"  g={g}: ATT={info['att']:.4f}, weight={weight:.3f}, contribution={contribution:.4f}")
            print(f"         n_treated={info['n_treated']}, n_control={info['n_control']}")
        
        print(f"  WATT: {watt:.4f}")
        print(f"  比率: {watt / PAPER_DETREND.get(r, 1):.2f}x")


def main():
    analyze_event_time_contributions()
    compute_detailed_detrend()
    
    print("\n" + "=" * 80)
    print("诊断结论")
    print("=" * 80)
    print("""
关键发现：
1. r=10, 11, 12, 13 只有少数 cohort 贡献
2. 这些 event time 的结果对单个 cohort 的 ATT 非常敏感
3. 可能的差异来源：
   - 论文可能使用了不同的样本选择
   - 论文可能对极端值进行了处理
   - 论文可能使用了不同的控制组定义
""")


if __name__ == '__main__':
    main()

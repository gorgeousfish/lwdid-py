"""
诊断：控制组定义是否正确

论文使用的控制组定义：
- not_yet_treated: 在时期 r 尚未被处理的单元 (g > r)
- never_treated: 从未被处理的单元 (g = inf)

关键问题：
- 我们是否正确实现了控制组选择？
- 论文是否使用了 never_treated 还是 not_yet_treated？
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


def compute_watt_with_different_control_groups():
    """使用不同的控制组定义计算 WATT"""
    df = load_data()
    
    T_min = int(df['year'].min())
    T_max = int(df['year'].max())
    cohorts = sorted([int(g) for g in df['g'].unique() if pd.notna(g) and g != np.inf])
    cohort_sizes = {int(k): v for k, v in df[df['g'] != np.inf].groupby('g')['fips'].nunique().to_dict().items()}
    
    # 计算预处理期均值
    unit_pre_means = {}
    for g in cohorts:
        pre_mask = df['year'] < g
        pre_means = df[pre_mask].groupby('fips')['log_retail_emp'].mean()
        unit_pre_means[g] = pre_means
    
    def compute_att(control_type):
        """计算 ATT，使用指定的控制组类型"""
        att_results = []
        
        for g in cohorts:
            for r in range(g, T_max + 1):
                event_time = r - g
                period_data = df[df['year'] == r].copy()
                
                period_data['ydot'] = period_data.apply(
                    lambda row: row['log_retail_emp'] - unit_pre_means[g].get(row['fips'], np.nan),
                    axis=1
                )
                
                treated_mask = period_data['g'] == g
                
                if control_type == 'never_treated':
                    control_mask = period_data['g'] == np.inf
                elif control_type == 'not_yet_treated':
                    control_mask = (period_data['g'] > r) | (period_data['g'] == np.inf)
                elif control_type == 'not_yet_treated_strict':
                    # 严格的 not_yet_treated：只包括 g > r，不包括 never_treated
                    control_mask = period_data['g'] > r
                else:
                    raise ValueError(f"Unknown control type: {control_type}")
                
                treated_vals = period_data[treated_mask]['ydot'].dropna()
                control_vals = period_data[control_mask]['ydot'].dropna()
                
                if len(treated_vals) == 0 or len(control_vals) == 0:
                    continue
                
                att = treated_vals.mean() - control_vals.mean()
                att_results.append({
                    'cohort': g, 'period': r, 'event_time': event_time, 'att': att,
                    'n_treated': len(treated_vals), 'n_control': len(control_vals),
                })
        
        return pd.DataFrame(att_results)
    
    def compute_watt(att_df):
        """计算 WATT"""
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
    
    # 计算三种控制组定义的结果
    print("=" * 80)
    print("使用不同控制组定义计算 WATT")
    print("=" * 80)
    
    control_types = ['never_treated', 'not_yet_treated', 'not_yet_treated_strict']
    results = {}
    
    for ct in control_types:
        att_df = compute_att(ct)
        watt = compute_watt(att_df)
        results[ct] = watt
        print(f"\n{ct}:")
        print(f"  r=0: {watt.get(0, np.nan):.4f} (论文: {PAPER_DEMEAN[0]:.3f}, 比率: {watt.get(0, np.nan)/PAPER_DEMEAN[0]:.2f}x)")
        print(f"  r=1: {watt.get(1, np.nan):.4f} (论文: {PAPER_DEMEAN[1]:.3f}, 比率: {watt.get(1, np.nan)/PAPER_DEMEAN[1]:.2f}x)")
    
    # 对比表格
    print("\n" + "=" * 80)
    print("对比表格")
    print("=" * 80)
    
    print(f"\n{'r':>3} | {'never_treated':>14} | {'not_yet_treated':>16} | {'nyt_strict':>12} | {'Paper':>8}")
    print("-" * 70)
    
    for r in range(14):
        nt = results['never_treated'].get(r, np.nan)
        nyt = results['not_yet_treated'].get(r, np.nan)
        nyts = results['not_yet_treated_strict'].get(r, np.nan)
        paper = PAPER_DEMEAN.get(r, np.nan)
        
        print(f"{r:>3} | {nt:>14.4f} | {nyt:>16.4f} | {nyts:>12.4f} | {paper:>8.3f}")
    
    return results


def main():
    compute_watt_with_different_control_groups()
    
    print("\n" + "=" * 80)
    print("诊断结论")
    print("=" * 80)
    print("""
关键发现：
1. 不同的控制组定义会产生不同的结果
2. 需要确认论文使用的是哪种控制组定义
3. 如果所有控制组定义都比论文高，说明差异不在控制组选择
""")


if __name__ == '__main__':
    main()

"""
WATT 详细诊断

检查 WATT 计算的每个步骤，确保与论文一致。

论文 Section 6.2 WATT 公式:
WATT(r) = Σ_{g∈G_r} w(g,r) × ATT(g, g+r)
w(g,r) = N_g / N_{G_r}

其中:
- G_r = {g : g + r ≤ T} 是在 event time r 有观测的 cohort 集合
- N_g = cohort g 的单位数
- N_{G_r} = Σ_{g∈G_r} N_g
"""

import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

warnings.filterwarnings('ignore')


# 论文参考值
PAPER_DEMEAN = {
    0: 0.018, 1: 0.045, 2: 0.038, 3: 0.032, 4: 0.031,
    5: 0.036, 6: 0.040, 7: 0.054, 8: 0.062, 9: 0.063,
    10: 0.081, 11: 0.083, 12: 0.080, 13: 0.107,
}


def load_data():
    """加载 Walmart 数据"""
    data_path = Path(__file__).parent.parent / 'data' / 'walmart.csv'
    return pd.read_csv(data_path)


def main():
    """运行 WATT 详细诊断"""
    print("=" * 80)
    print("WATT 详细诊断")
    print("=" * 80)
    
    from lwdid import lwdid
    
    df = load_data()
    
    controls = [
        'share_pop_poverty_78_above',
        'share_pop_ind_manuf',
        'share_school_some_hs',
    ]
    
    # 运行 lwdid
    print("\n运行 lwdid (demean + not_yet_treated + ipwra)...")
    result = lwdid(
        data=df,
        y='log_retail_emp',
        ivar='fips',
        tvar='year',
        gvar='g',
        rolling='demean',
        estimator='ipwra',
        controls=controls,
        control_group='not_yet_treated',
        aggregate='none',
        alpha=0.05,
    )
    
    att_ct = result.att_by_cohort_time.copy()
    
    # 获取 cohort 大小
    cohort_sizes = df[df['g'] != np.inf].groupby('g')['fips'].nunique().to_dict()
    
    print(f"\nCohort 大小:")
    for g, n in sorted(cohort_sizes.items()):
        print(f"  g={int(g)}: {n} 单位")
    
    print(f"\n总处理单位数: {sum(cohort_sizes.values())}")
    
    # 检查 event_time = 0 的详细计算
    print("\n" + "=" * 80)
    print("Event Time = 0 的详细计算")
    print("=" * 80)
    
    event_time = 0
    subset = att_ct[att_ct['event_time'] == event_time].copy()
    
    print(f"\n参与 event_time={event_time} 的 cohort:")
    print(f"{'cohort':>8} | {'period':>8} | {'ATT':>10} | {'SE':>10} | {'N_g':>6} | {'w(g,r)':>8}")
    print("-" * 60)
    
    # 计算权重
    available_cohorts = subset['cohort'].astype(int).tolist()
    total_units = sum(cohort_sizes.get(g, 0) for g in available_cohorts)
    
    weighted_att = 0
    weighted_var = 0
    
    for _, row in subset.iterrows():
        g = int(row['cohort'])
        att = row['att']
        se = row['se']
        n_g = cohort_sizes.get(g, 0)
        w_g = n_g / total_units if total_units > 0 else 0
        
        weighted_att += w_g * att
        weighted_var += (w_g ** 2) * (se ** 2)
        
        print(f"{g:>8} | {int(row['period']):>8} | {att:>10.6f} | {se:>10.6f} | {n_g:>6} | {w_g:>8.4f}")
    
    watt = weighted_att
    watt_se = np.sqrt(weighted_var)
    
    print("-" * 60)
    print(f"{'WATT':>8} | {'-':>8} | {watt:>10.6f} | {watt_se:>10.6f} | {total_units:>6} | {'1.0000':>8}")
    
    paper_val = PAPER_DEMEAN.get(event_time, np.nan)
    print(f"\n论文值: {paper_val:.3f}")
    print(f"比率: {watt / paper_val:.2f}x")
    
    # 计算所有 event_time 的 WATT
    print("\n" + "=" * 80)
    print("所有 Event Time 的 WATT")
    print("=" * 80)
    
    print(f"\n{'r':>3} | {'WATT':>10} | {'SE':>10} | {'Paper':>8} | {'Ratio':>8} | {'N_cohorts':>10}")
    print("-" * 60)
    
    for event_time in range(14):
        subset = att_ct[att_ct['event_time'] == event_time].copy()
        subset = subset[subset['att'].notna()]
        
        if len(subset) == 0:
            continue
        
        available_cohorts = subset['cohort'].astype(int).tolist()
        total_units = sum(cohort_sizes.get(g, 0) for g in available_cohorts)
        
        weighted_att = 0
        weighted_var = 0
        
        for _, row in subset.iterrows():
            g = int(row['cohort'])
            att = row['att']
            se = row['se']
            n_g = cohort_sizes.get(g, 0)
            w_g = n_g / total_units if total_units > 0 else 0
            
            weighted_att += w_g * att
            weighted_var += (w_g ** 2) * (se ** 2)
        
        watt = weighted_att
        watt_se = np.sqrt(weighted_var)
        paper_val = PAPER_DEMEAN.get(event_time, np.nan)
        ratio = watt / paper_val if paper_val > 0 else np.nan
        
        print(f"{event_time:>3} | {watt:>10.6f} | {watt_se:>10.6f} | {paper_val:>8.3f} | {ratio:>8.2f}x | {len(subset):>10}")
    
    # 检查单个 cohort 的 ATT
    print("\n" + "=" * 80)
    print("单个 Cohort 的 ATT 检查 (event_time=0)")
    print("=" * 80)
    
    # 选择一个具体的 cohort 进行详细检查
    cohort = 1995
    period = 1995
    
    row = att_ct[(att_ct['cohort'] == cohort) & (att_ct['period'] == period)]
    if len(row) > 0:
        att = row['att'].values[0]
        se = row['se'].values[0]
        n_treated = row['n_treated'].values[0]
        n_control = row['n_control'].values[0]
        
        print(f"\nCohort {cohort}, Period {period}:")
        print(f"  ATT: {att:.6f}")
        print(f"  SE: {se:.6f}")
        print(f"  N_treated: {n_treated}")
        print(f"  N_control: {n_control}")


if __name__ == '__main__':
    main()

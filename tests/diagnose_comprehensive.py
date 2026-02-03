"""
综合诊断：深入分析 Python 实现与论文 Table A4 的差异

关键发现总结：
1. Python Full Demean + not_yet_treated: ~2.37x 论文
2. Python Long Difference + not_yet_treated: ~1.63x 论文
3. Stata csdid (Long Difference): ~1.43x 论文

所有方法都比论文高，说明差异不在变换方法或估计器，而在更基础的层面。

本脚本将深入检查：
1. 数据是否完全一致
2. 样本选择是否一致
3. 权重计算是否一致
4. 是否有遗漏的数据处理步骤
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


def check_data_consistency(df):
    """检查数据与论文 Table 2 的一致性"""
    print("\n" + "=" * 80)
    print("数据一致性检查 (vs 论文 Table 2)")
    print("=" * 80)
    
    # 论文 Table 2 的描述性统计
    paper_stats = {
        'N': 29440,
        'n_counties': 1280,
        'log_retail_emp_mean': 7.754502,
        'share_pop_poverty_78_above_mean': 0.8470385,
        'share_pop_ind_manuf_mean': 0.0998018,
        'share_school_some_hs_mean': 0.092258,
    }
    
    # 计算实际统计
    actual_stats = {
        'N': len(df),
        'n_counties': df['fips'].nunique(),
        'log_retail_emp_mean': df['log_retail_emp'].mean(),
        'share_pop_poverty_78_above_mean': df['share_pop_poverty_78_above'].mean(),
        'share_pop_ind_manuf_mean': df['share_pop_ind_manuf'].mean(),
        'share_school_some_hs_mean': df['share_school_some_hs'].mean(),
    }
    
    print(f"\n{'统计量':<35} | {'论文':>12} | {'实际':>12} | {'差异':>10}")
    print("-" * 75)
    
    all_match = True
    for key in paper_stats:
        paper_val = paper_stats[key]
        actual_val = actual_stats[key]
        diff = abs(paper_val - actual_val)
        match = "✓" if diff < 0.0001 else "✗"
        if diff >= 0.0001:
            all_match = False
        print(f"{key:<35} | {paper_val:>12.6f} | {actual_val:>12.6f} | {match:>10}")
    
    return all_match


def check_cohort_distribution(df):
    """检查 cohort 分布"""
    print("\n" + "=" * 80)
    print("Cohort 分布检查")
    print("=" * 80)
    
    cohort_counts = df.groupby('g')['fips'].nunique().sort_index()
    
    print(f"\n{'Cohort':>10} | {'单位数':>10} | {'占比':>10}")
    print("-" * 35)
    
    total_treated = 0
    for g, count in cohort_counts.items():
        if g != np.inf:
            total_treated += count
            pct = count / df['fips'].nunique() * 100
            print(f"{int(g) if g != np.inf else 'Never':>10} | {count:>10} | {pct:>9.1f}%")
    
    # Never-treated
    nt_count = cohort_counts.get(np.inf, 0)
    nt_pct = nt_count / df['fips'].nunique() * 100
    print(f"{'Never':>10} | {nt_count:>10} | {nt_pct:>9.1f}%")
    
    print("-" * 35)
    print(f"{'Total':>10} | {df['fips'].nunique():>10} | {'100.0':>9}%")
    print(f"\n处理单位总数: {total_treated}")
    print(f"从未处理单位数: {nt_count}")


def analyze_event_time_contributions(df):
    """分析各 event time 的 cohort 贡献"""
    print("\n" + "=" * 80)
    print("Event Time 的 Cohort 贡献分析")
    print("=" * 80)
    
    cohorts = sorted([g for g in df['g'].unique() if pd.notna(g) and g != np.inf])
    T_max = int(df['year'].max())
    
    # 计算 cohort 大小
    cohort_sizes = df[df['g'] != np.inf].groupby('g')['fips'].nunique().to_dict()
    
    print(f"\n{'Event Time':>10} | {'贡献 Cohorts':>15} | {'总权重':>10} | {'最大权重 Cohort':>20}")
    print("-" * 60)
    
    for event_time in range(14):
        contributing_cohorts = []
        for g in cohorts:
            g = int(g)
            calendar_r = g + event_time
            if calendar_r <= T_max:
                contributing_cohorts.append(g)
        
        if not contributing_cohorts:
            continue
        
        total_weight = sum(cohort_sizes.get(g, 0) for g in contributing_cohorts)
        max_cohort = max(contributing_cohorts, key=lambda g: cohort_sizes.get(g, 0))
        max_weight = cohort_sizes.get(max_cohort, 0)
        max_weight_pct = max_weight / total_weight * 100 if total_weight > 0 else 0
        
        print(f"{event_time:>10} | {len(contributing_cohorts):>15} | {total_weight:>10} | g={max_cohort} ({max_weight_pct:.1f}%)")


def compute_detailed_att(df, transform='demean', control_group='not_yet_treated'):
    """计算详细的 ATT，包括中间步骤"""
    print(f"\n" + "=" * 80)
    print(f"详细 ATT 计算 (transform={transform}, control={control_group})")
    print("=" * 80)
    
    y = 'log_retail_emp'
    ivar = 'fips'
    tvar = 'year'
    gvar = 'g'
    
    cohorts = sorted([g for g in df[gvar].unique() if pd.notna(g) and g != np.inf])
    T_min = int(df[tvar].min())
    T_max = int(df[tvar].max())
    
    # 计算 cohort 大小
    cohort_sizes = df[df[gvar] != np.inf].groupby(gvar)[ivar].nunique().to_dict()
    
    # 存储 cohort-time ATT
    att_results = []
    
    for g in cohorts:
        g = int(g)
        
        # 计算变换
        if transform == 'demean':
            # Full Demean: 使用所有 t < g 的时期
            pre_mask = df[tvar] < g
            pre_means = df[pre_mask].groupby(ivar)[y].mean()
        else:  # long_difference
            # Long Difference: 只使用 t = g-1
            base_period = g - 1
            pre_means = df[df[tvar] == base_period].set_index(ivar)[y]
        
        for r in range(g, T_max + 1):
            event_time = r - g
            
            # 获取该时期的数据
            period_data = df[df[tvar] == r].copy()
            
            # 计算变换后的值
            period_data['y_transformed'] = (
                period_data[y].values -
                period_data[ivar].map(pre_means).values
            )
            
            # 处理组
            treated_mask = period_data[gvar] == g
            
            # 控制组
            if control_group == 'never_treated':
                control_mask = period_data[gvar] == np.inf
            else:  # not_yet_treated
                control_mask = (period_data[gvar] > r) | (period_data[gvar] == np.inf)
            
            # 计算 ATT
            treated_vals = period_data[treated_mask]['y_transformed'].dropna()
            control_vals = period_data[control_mask]['y_transformed'].dropna()
            
            if len(treated_vals) == 0 or len(control_vals) == 0:
                continue
            
            att = treated_vals.mean() - control_vals.mean()
            
            att_results.append({
                'cohort': g,
                'period': r,
                'event_time': event_time,
                'att': att,
                'n_treated': len(treated_vals),
                'n_control': len(control_vals),
                'treated_mean': treated_vals.mean(),
                'control_mean': control_vals.mean(),
            })
    
    att_df = pd.DataFrame(att_results)
    
    # 计算 WATT
    watt_results = []
    for event_time in sorted(att_df['event_time'].unique()):
        subset = att_df[att_df['event_time'] == event_time].copy()
        subset = subset[subset['att'].notna()]
        
        if len(subset) == 0:
            continue
        
        subset['weight'] = subset['cohort'].map(cohort_sizes)
        total_weight = subset['weight'].sum()
        subset['norm_weight'] = subset['weight'] / total_weight
        
        watt = (subset['att'] * subset['norm_weight']).sum()
        
        watt_results.append({
            'event_time': int(event_time),
            'watt': watt,
            'n_cohorts': len(subset),
            'total_weight': total_weight,
        })
    
    watt_df = pd.DataFrame(watt_results)
    
    # 显示结果
    # 注意：Long Difference 对应论文的 Demean 列（因为 CS 方法使用 Long Difference）
    paper_ref = PAPER_DEMEAN
    
    print(f"\n{'r':>3} | {'WATT':>10} | {'Paper':>10} | {'Ratio':>8} | {'N Cohorts':>10}")
    print("-" * 50)
    
    ratios = []
    for _, row in watt_df.iterrows():
        r = int(row['event_time'])
        watt = row['watt']
        paper = paper_ref.get(r, np.nan)
        ratio = watt / paper if paper > 0 else np.nan
        ratios.append(ratio)
        
        print(f"{r:>3} | {watt:>10.4f} | {paper:>10.3f} | {ratio:>8.2f}x | {int(row['n_cohorts']):>10}")
    
    print("-" * 50)
    print(f"平均比率: {np.nanmean(ratios):.2f}x")
    
    return att_df, watt_df


def main():
    df = load_data()
    
    # 1. 数据一致性检查
    data_ok = check_data_consistency(df)
    
    # 2. Cohort 分布检查
    check_cohort_distribution(df)
    
    # 3. Event Time 贡献分析
    analyze_event_time_contributions(df)
    
    # 4. 详细 ATT 计算
    print("\n\n" + "=" * 80)
    print("方法 1: Full Demean + not_yet_treated")
    print("=" * 80)
    att_dm, watt_dm = compute_detailed_att(df, 'demean', 'not_yet_treated')
    
    print("\n\n" + "=" * 80)
    print("方法 2: Long Difference + not_yet_treated")
    print("=" * 80)
    att_ld, watt_ld = compute_detailed_att(df, 'long_difference', 'not_yet_treated')
    
    # 5. 结论
    print("\n\n" + "=" * 80)
    print("诊断结论")
    print("=" * 80)
    
    print("""
关键发现：

1. 数据一致性: """ + ("✓ 完全匹配" if data_ok else "✗ 存在差异") + """

2. 所有方法都比论文高出 ~1.4-2.4x:
   - Full Demean + not_yet_treated: ~2.37x
   - Long Difference + not_yet_treated: ~1.63x
   - Stata csdid: ~1.43x

3. 可能的原因:
   a) 论文可能使用了不同的 WATT 权重计算方式
   b) 论文可能有未公开的数据预处理步骤
   c) 论文可能使用了不同的样本选择标准
   d) 论文结果可能存在计算误差

4. 建议:
   - 联系论文作者确认方法细节
   - 检查论文的复现代码（如果有）
   - 记录差异并在文档中说明
""")


if __name__ == '__main__':
    main()

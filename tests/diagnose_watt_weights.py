"""
诊断脚本：WATT 权重计算方式

论文公式 6.3-6.4:
    w(g,r) = N_g / N_{G_r}
    WATT(r) = Σ_g w(g,r) × ATT(g, g+r)

其中 N_{G_r} 是在 event time r 有贡献的所有 cohort 的总处理单位数。

问题：当前实现是否正确计算了 N_{G_r}？
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


def compute_cohort_time_att(df_transformed, df_original, gvar, col_prefix, control_group='not_yet_treated'):
    """计算 cohort-time ATT"""
    cohorts = sorted([g for g in df_original[gvar].unique() if pd.notna(g) and g != np.inf])
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
            else:
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
            })
    
    return pd.DataFrame(att_results)


def compute_watt_method1(att_df, cohort_sizes):
    """
    方法 1: 使用所有 cohort 的总处理单位数作为分母
    
    w(g,r) = N_g / Σ_h N_h (所有 cohort)
    """
    total_treated = sum(cohort_sizes.values())
    
    watt_list = []
    for event_time in sorted(att_df['event_time'].unique()):
        subset = att_df[att_df['event_time'] == event_time].copy()
        subset = subset[subset['att'].notna()]
        
        if len(subset) == 0:
            continue
        
        subset['weight'] = subset['cohort'].map(cohort_sizes)
        subset['norm_weight'] = subset['weight'] / total_treated
        
        watt = (subset['att'] * subset['norm_weight']).sum()
        watt_list.append({
            'event_time': int(event_time),
            'watt': watt,
            'method': 'all_cohorts',
        })
    
    return pd.DataFrame(watt_list)


def compute_watt_method2(att_df, cohort_sizes):
    """
    方法 2: 使用在该 event time 有贡献的 cohort 的总处理单位数作为分母
    
    w(g,r) = N_g / N_{G_r}
    其中 N_{G_r} = Σ_{h: h+r <= T_max} N_h
    """
    watt_list = []
    for event_time in sorted(att_df['event_time'].unique()):
        subset = att_df[att_df['event_time'] == event_time].copy()
        subset = subset[subset['att'].notna()]
        
        if len(subset) == 0:
            continue
        
        # 只使用在该 event time 有贡献的 cohort
        subset['weight'] = subset['cohort'].map(cohort_sizes)
        total_weight = subset['weight'].sum()  # N_{G_r}
        subset['norm_weight'] = subset['weight'] / total_weight
        
        watt = (subset['att'] * subset['norm_weight']).sum()
        watt_list.append({
            'event_time': int(event_time),
            'watt': watt,
            'method': 'contributing_cohorts',
        })
    
    return pd.DataFrame(watt_list)


def compute_watt_method3(att_df, cohort_sizes):
    """
    方法 3: 简单平均 (不加权)
    
    WATT(r) = (1/|G_r|) × Σ_g ATT(g, g+r)
    """
    watt_list = []
    for event_time in sorted(att_df['event_time'].unique()):
        subset = att_df[att_df['event_time'] == event_time].copy()
        subset = subset[subset['att'].notna()]
        
        if len(subset) == 0:
            continue
        
        watt = subset['att'].mean()
        watt_list.append({
            'event_time': int(event_time),
            'watt': watt,
            'method': 'simple_average',
        })
    
    return pd.DataFrame(watt_list)


def main():
    """运行诊断"""
    print("=" * 80)
    print("诊断：WATT 权重计算方式")
    print("=" * 80)
    
    df = load_data()
    
    # Long Difference 变换
    print("\n计算 Long Difference 变换...")
    df_ld = transform_long_difference(df, 'log_retail_emp', 'fips', 'year', 'g')
    
    # 计算 cohort-time ATT
    print("计算 cohort-time ATT (not_yet_treated 控制组)...")
    att_df = compute_cohort_time_att(df_ld, df, 'g', 'yld', 'not_yet_treated')
    
    # Cohort 大小
    cohort_sizes = df[df['g'] != np.inf].groupby('g')['fips'].nunique().to_dict()
    
    # 三种 WATT 计算方法
    print("\n计算三种 WATT 方法...")
    watt_m1 = compute_watt_method1(att_df, cohort_sizes)
    watt_m2 = compute_watt_method2(att_df, cohort_sizes)
    watt_m3 = compute_watt_method3(att_df, cohort_sizes)
    
    # 比较结果
    print("\n" + "=" * 80)
    print("WATT 权重方法比较")
    print("=" * 80)
    
    print(f"\n{'r':>3} | {'M1 (All)':>10} | {'M2 (Contrib)':>12} | {'M3 (Simple)':>12} | {'Paper':>8}")
    print("-" * 60)
    
    m1_ratios = []
    m2_ratios = []
    m3_ratios = []
    
    for event_time in range(14):
        m1_val = watt_m1[watt_m1['event_time'] == event_time]['watt'].values
        m2_val = watt_m2[watt_m2['event_time'] == event_time]['watt'].values
        m3_val = watt_m3[watt_m3['event_time'] == event_time]['watt'].values
        paper_val = PAPER_DEMEAN.get(event_time, np.nan)
        
        m1_str = f"{m1_val[0]:.4f}" if len(m1_val) > 0 else "N/A"
        m2_str = f"{m2_val[0]:.4f}" if len(m2_val) > 0 else "N/A"
        m3_str = f"{m3_val[0]:.4f}" if len(m3_val) > 0 else "N/A"
        
        if len(m1_val) > 0 and paper_val > 0:
            m1_ratios.append(m1_val[0] / paper_val)
        if len(m2_val) > 0 and paper_val > 0:
            m2_ratios.append(m2_val[0] / paper_val)
        if len(m3_val) > 0 and paper_val > 0:
            m3_ratios.append(m3_val[0] / paper_val)
        
        print(f"{event_time:>3} | {m1_str:>10} | {m2_str:>12} | {m3_str:>12} | {paper_val:>8.3f}")
    
    print("-" * 60)
    print(f"平均比率: M1={np.mean(m1_ratios):.2f}x, M2={np.mean(m2_ratios):.2f}x, M3={np.mean(m3_ratios):.2f}x")
    
    # 检查各 event time 有多少 cohort 贡献
    print("\n" + "=" * 80)
    print("各 Event Time 的 Cohort 贡献")
    print("=" * 80)
    
    print(f"\n{'r':>3} | {'N Cohorts':>10} | {'Total N_g':>10} | {'Cohorts':>40}")
    print("-" * 70)
    
    for event_time in range(14):
        subset = att_df[att_df['event_time'] == event_time]
        n_cohorts = len(subset)
        total_ng = subset['cohort'].map(cohort_sizes).sum()
        cohorts_str = ', '.join([str(int(c)) for c in sorted(subset['cohort'].unique())])
        if len(cohorts_str) > 40:
            cohorts_str = cohorts_str[:37] + '...'
        
        print(f"{event_time:>3} | {n_cohorts:>10} | {total_ng:>10} | {cohorts_str:>40}")
    
    # 结论
    print("\n" + "=" * 80)
    print("结论")
    print("=" * 80)
    
    best_method = min(
        [('M1 (All Cohorts)', np.mean(m1_ratios)),
         ('M2 (Contributing)', np.mean(m2_ratios)),
         ('M3 (Simple Avg)', np.mean(m3_ratios))],
        key=lambda x: abs(x[1] - 1)
    )
    
    print(f"\n最接近论文的方法: {best_method[0]} (比率 {best_method[1]:.2f}x)")
    
    print("\n注意: 即使使用最佳方法，结果仍然是论文的 ~1.6x")
    print("这表明可能还有其他差异来源，例如:")
    print("  1. 论文可能使用了不同的估计器 (RA, IPW, IPWRA)")
    print("  2. 论文可能使用了不同的协变量处理方式")
    print("  3. 论文可能使用了不同的样本选择")


if __name__ == '__main__':
    main()

"""
Long Difference vs Demean 对比诊断

根据 LWDID_算法公式提取.md 第 9 节：

CS (2021) 方法 (Long Difference):
    Ŷ_{irg} = Y_{ir} - Y_{i,g-1}  (只用最后一个 pre-treatment 期)

LWDID 方法 (Demean):
    Ŷ_{irg} = Y_{ir} - mean(Y_{i,1:g-1})  (用所有 pre-treatment 期的均值)

假设：论文 Table A4 可能使用了 Long Difference 而不是 Demean
"""

import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

warnings.filterwarnings('ignore')


def load_data():
    """加载 Walmart 数据"""
    data_path = Path(__file__).parent.parent / 'data' / 'walmart.csv'
    return pd.read_csv(data_path)


# 论文参考值
PAPER_DEMEAN = {
    0: 0.018, 1: 0.045, 2: 0.038, 3: 0.032, 4: 0.031,
    5: 0.036, 6: 0.040, 7: 0.054, 8: 0.062, 9: 0.063,
    10: 0.081, 11: 0.083, 12: 0.080, 13: 0.107,
}


def transform_long_difference(df, y, ivar, tvar, gvar):
    """
    应用 Long Difference 变换 (CS 2021 方法)
    
    Ŷ_{irg} = Y_{ir} - Y_{i,g-1}
    """
    result = df.copy()
    result[tvar] = pd.to_numeric(result[tvar], errors='coerce')
    
    # 获取 cohorts
    cohorts = sorted([g for g in df[gvar].unique() if pd.notna(g) and g != np.inf])
    T_max = int(result[tvar].max())
    
    for g in cohorts:
        g = int(g)
        base_period = g - 1  # 最后一个 pre-treatment 期
        
        # 获取每个单位在 base_period 的值
        base_values = result[result[tvar] == base_period].set_index(ivar)[y]
        
        # 对每个 post-treatment 期间应用变换
        for r in range(g, T_max + 1):
            col_name = f'yld_g{g}_r{r}'
            result[col_name] = np.nan
            
            period_mask = result[tvar] == r
            result.loc[period_mask, col_name] = (
                result.loc[period_mask, y].values -
                result.loc[period_mask, ivar].map(base_values).values
            )
    
    return result


def compute_watt_long_diff(df, controls):
    """使用 Long Difference 变换计算 WATT"""
    from lwdid.staggered.estimators import estimate_ipwra
    
    df_transformed = transform_long_difference(
        df, y='log_retail_emp', ivar='fips', tvar='year', gvar='g'
    )
    
    cohorts = sorted([g for g in df['g'].unique() if pd.notna(g) and g != np.inf])
    cohort_sizes = df[df['g'] != np.inf].groupby('g')['fips'].nunique().to_dict()
    T_max = int(df['year'].max())
    
    # 收集所有 (cohort, period) 的 ATT
    att_results = []
    
    for g in cohorts:
        g = int(g)
        for r in range(g, T_max + 1):
            col_name = f'yld_g{g}_r{r}'
            
            if col_name not in df_transformed.columns:
                continue
            
            # 提取数据
            period_data = df_transformed[df_transformed['year'] == r].copy()
            period_data['d'] = (period_data['g'] == g).astype(int)
            
            # 只保留处理组和 never-treated 控制组
            sample = period_data[(period_data['g'] == g) | (period_data['g'] == np.inf)].copy()
            sample = sample.dropna(subset=[col_name] + controls)
            
            n_treated = (sample['d'] == 1).sum()
            n_control = (sample['d'] == 0).sum()
            
            if n_treated < 2 or n_control < 2:
                continue
            
            try:
                result = estimate_ipwra(
                    data=sample,
                    y=col_name,
                    d='d',
                    controls=controls,
                    propensity_controls=controls,
                    trim_threshold=0.01,
                    alpha=0.05,
                )
                
                att_results.append({
                    'cohort': g,
                    'period': r,
                    'event_time': r - g,
                    'att': result.att,
                    'se': result.se,
                })
            except Exception as e:
                print(f"  Cohort {g}, Period {r} 失败: {e}")
    
    att_df = pd.DataFrame(att_results)
    
    # 计算 WATT
    watt_list = []
    for event_time in sorted(att_df['event_time'].unique()):
        subset = att_df[att_df['event_time'] == event_time].copy()
        subset = subset[subset['att'].notna()]
        
        if len(subset) == 0:
            continue
        
        subset['weight'] = subset['cohort'].map(cohort_sizes)
        total_weight = subset['weight'].sum()
        subset['norm_weight'] = subset['weight'] / total_weight
        
        watt = (subset['att'] * subset['norm_weight']).sum()
        watt_list.append({
            'event_time': int(event_time),
            'watt': watt,
        })
    
    return pd.DataFrame(watt_list)


def main():
    """比较 Long Difference 和 Demean"""
    print("=" * 70)
    print("Long Difference vs Demean 对比")
    print("=" * 70)
    
    df = load_data()
    controls = ['share_pop_poverty_78_above', 'share_pop_ind_manuf', 'share_school_some_hs']
    
    # 1. 使用 Long Difference
    print("\n计算 Long Difference WATT...")
    watt_ld = compute_watt_long_diff(df, controls)
    
    # 2. 使用 Demean (已有结果)
    from lwdid import lwdid
    
    print("计算 Demean WATT...")
    results_dm = lwdid(
        data=df, y='log_retail_emp', ivar='fips', tvar='year', gvar='g',
        rolling='demean', estimator='ipwra', controls=controls,
        control_group='never_treated', aggregate='none',
    )
    
    att_ct = results_dm.att_by_cohort_time.copy()
    cohort_sizes = df[df['g'] != np.inf].groupby('g')['fips'].nunique().to_dict()
    
    watt_dm_list = []
    for event_time in range(14):
        subset = att_ct[att_ct['event_time'] == event_time].copy()
        subset = subset[subset['att'].notna()]
        
        if len(subset) == 0:
            continue
        
        subset['weight'] = subset['cohort'].map(cohort_sizes)
        total_weight = subset['weight'].sum()
        subset['norm_weight'] = subset['weight'] / total_weight
        
        watt = (subset['att'] * subset['norm_weight']).sum()
        watt_dm_list.append({
            'event_time': int(event_time),
            'watt': watt,
        })
    
    watt_dm = pd.DataFrame(watt_dm_list)
    
    # 3. 比较结果
    print("\n" + "=" * 70)
    print("结果比较")
    print("=" * 70)
    
    print(f"\n{'Event Time':>10} | {'Long Diff':>12} | {'Demean':>12} | {'Paper':>8} | {'LD Ratio':>10} | {'DM Ratio':>10}")
    print("-" * 80)
    
    for event_time in range(14):
        ld_val = watt_ld[watt_ld['event_time'] == event_time]['watt'].values
        dm_val = watt_dm[watt_dm['event_time'] == event_time]['watt'].values
        paper_val = PAPER_DEMEAN.get(event_time, np.nan)
        
        ld_str = f"{ld_val[0]:.6f}" if len(ld_val) > 0 else "N/A"
        dm_str = f"{dm_val[0]:.6f}" if len(dm_val) > 0 else "N/A"
        
        ld_ratio = f"{ld_val[0]/paper_val:.2f}x" if len(ld_val) > 0 and paper_val > 0 else "N/A"
        dm_ratio = f"{dm_val[0]/paper_val:.2f}x" if len(dm_val) > 0 and paper_val > 0 else "N/A"
        
        print(f"{event_time:>10} | {ld_str:>12} | {dm_str:>12} | {paper_val:>8.3f} | {ld_ratio:>10} | {dm_ratio:>10}")
    
    print("\n" + "=" * 70)
    print("分析完成")
    print("=" * 70)


if __name__ == '__main__':
    main()

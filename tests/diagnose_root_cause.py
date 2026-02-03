"""
根因诊断：比较 Python Full Demean、Long Difference 和 Stata csdid 结果

目标：找出为什么所有方法都比论文 Table A4 高出 ~1.2-2x
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

# Stata csdid 结果 (从之前的诊断)
STATA_CSDID = {
    0: 0.0221, 1: 0.0522, 2: 0.0476, 3: 0.0437, 4: 0.0449,
    5: 0.0530, 6: 0.0596, 7: 0.0795, 8: 0.0912, 9: 0.0943,
    10: 0.1175, 11: 0.1227, 12: 0.1207, 13: 0.1913,
}


def load_data():
    """加载 Walmart 数据"""
    data_path = Path(__file__).parent.parent / 'data' / 'walmart.csv'
    return pd.read_csv(data_path)


def transform_full_demean(df, y, ivar, tvar, gvar):
    """Full Demean: Y_ir - mean(Y_{i,1:g-1})"""
    result = df.copy()
    result[tvar] = pd.to_numeric(result[tvar], errors='coerce')
    
    cohorts = sorted([g for g in df[gvar].unique() if pd.notna(g) and g != np.inf])
    T_max = int(result[tvar].max())
    
    for g in cohorts:
        g = int(g)
        # 使用所有 t < g 的时期计算均值
        pre_mask = result[tvar] < g
        pre_means = result[pre_mask].groupby(ivar)[y].mean()
        
        for r in range(g, T_max + 1):
            col_name = f'ydm_g{g}_r{r}'
            result[col_name] = np.nan
            
            period_mask = result[tvar] == r
            result.loc[period_mask, col_name] = (
                result.loc[period_mask, y].values -
                result.loc[period_mask, ivar].map(pre_means).values
            )
    
    return result


def transform_long_difference(df, y, ivar, tvar, gvar):
    """Long Difference: Y_ir - Y_{i,g-1}"""
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


def compute_simple_att(df_transformed, df_original, gvar, col_prefix, control_group='not_yet_treated'):
    """计算简单 ATT (无协变量调整)"""
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
            else:  # not_yet_treated
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
                'n_control': len(control_vals),
            })
    
    return pd.DataFrame(att_results)


def compute_watt(att_df, cohort_sizes):
    """计算 WATT (使用贡献 cohort 的权重)"""
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
    print("=" * 90)
    print("根因诊断：比较不同变换方法和控制组选择")
    print("=" * 90)
    
    df = load_data()
    
    # 计算 cohort 大小
    cohort_sizes = df[df['g'] != np.inf].groupby('g')['fips'].nunique().to_dict()
    
    # 1. Full Demean 变换
    print("\n[1] 计算 Full Demean 变换...")
    df_dm = transform_full_demean(df, 'log_retail_emp', 'fips', 'year', 'g')
    
    # 2. Long Difference 变换
    print("[2] 计算 Long Difference 变换...")
    df_ld = transform_long_difference(df, 'log_retail_emp', 'fips', 'year', 'g')
    
    # 3. 计算各种组合的 ATT
    print("[3] 计算 ATT (4种组合)...")
    
    # Full Demean + never_treated
    att_dm_nt = compute_simple_att(df_dm, df, 'g', 'ydm', 'never_treated')
    watt_dm_nt = compute_watt(att_dm_nt, cohort_sizes)
    
    # Full Demean + not_yet_treated
    att_dm_nyt = compute_simple_att(df_dm, df, 'g', 'ydm', 'not_yet_treated')
    watt_dm_nyt = compute_watt(att_dm_nyt, cohort_sizes)
    
    # Long Difference + never_treated
    att_ld_nt = compute_simple_att(df_ld, df, 'g', 'yld', 'never_treated')
    watt_ld_nt = compute_watt(att_ld_nt, cohort_sizes)
    
    # Long Difference + not_yet_treated
    att_ld_nyt = compute_simple_att(df_ld, df, 'g', 'yld', 'not_yet_treated')
    watt_ld_nyt = compute_watt(att_ld_nyt, cohort_sizes)
    
    # 4. 比较结果
    print("\n" + "=" * 90)
    print("WATT 结果比较 (简单 ATT，无协变量调整)")
    print("=" * 90)
    
    print(f"\n{'r':>3} | {'Paper':>8} | {'Stata':>8} | {'DM+NT':>8} | {'DM+NYT':>8} | {'LD+NT':>8} | {'LD+NYT':>8}")
    print("-" * 90)
    
    ratios = {'dm_nt': [], 'dm_nyt': [], 'ld_nt': [], 'ld_nyt': [], 'stata': []}
    
    for r in range(14):
        paper = PAPER_DEMEAN.get(r, np.nan)
        stata = STATA_CSDID.get(r, np.nan)
        
        dm_nt = watt_dm_nt[watt_dm_nt['event_time'] == r]['watt'].values
        dm_nyt = watt_dm_nyt[watt_dm_nyt['event_time'] == r]['watt'].values
        ld_nt = watt_ld_nt[watt_ld_nt['event_time'] == r]['watt'].values
        ld_nyt = watt_ld_nyt[watt_ld_nyt['event_time'] == r]['watt'].values
        
        dm_nt_val = dm_nt[0] if len(dm_nt) > 0 else np.nan
        dm_nyt_val = dm_nyt[0] if len(dm_nyt) > 0 else np.nan
        ld_nt_val = ld_nt[0] if len(ld_nt) > 0 else np.nan
        ld_nyt_val = ld_nyt[0] if len(ld_nyt) > 0 else np.nan
        
        print(f"{r:>3} | {paper:>8.3f} | {stata:>8.4f} | {dm_nt_val:>8.4f} | {dm_nyt_val:>8.4f} | {ld_nt_val:>8.4f} | {ld_nyt_val:>8.4f}")
        
        if paper > 0:
            ratios['dm_nt'].append(dm_nt_val / paper if not np.isnan(dm_nt_val) else np.nan)
            ratios['dm_nyt'].append(dm_nyt_val / paper if not np.isnan(dm_nyt_val) else np.nan)
            ratios['ld_nt'].append(ld_nt_val / paper if not np.isnan(ld_nt_val) else np.nan)
            ratios['ld_nyt'].append(ld_nyt_val / paper if not np.isnan(ld_nyt_val) else np.nan)
            ratios['stata'].append(stata / paper if not np.isnan(stata) else np.nan)
    
    print("-" * 90)
    print(f"平均比率 (vs Paper):")
    print(f"  Stata csdid:        {np.nanmean(ratios['stata']):.2f}x")
    print(f"  DM + never_treated: {np.nanmean(ratios['dm_nt']):.2f}x")
    print(f"  DM + not_yet:       {np.nanmean(ratios['dm_nyt']):.2f}x")
    print(f"  LD + never_treated: {np.nanmean(ratios['ld_nt']):.2f}x")
    print(f"  LD + not_yet:       {np.nanmean(ratios['ld_nyt']):.2f}x")
    
    # 5. 检查 Stata csdid 和 Python LD+NYT 的一致性
    print("\n" + "=" * 90)
    print("Stata csdid vs Python Long Difference + not_yet_treated")
    print("=" * 90)
    
    print(f"\n{'r':>3} | {'Stata':>10} | {'Python LD':>10} | {'Diff':>10} | {'Ratio':>8}")
    print("-" * 50)
    
    for r in range(14):
        stata = STATA_CSDID.get(r, np.nan)
        ld_nyt = watt_ld_nyt[watt_ld_nyt['event_time'] == r]['watt'].values
        ld_nyt_val = ld_nyt[0] if len(ld_nyt) > 0 else np.nan
        
        diff = stata - ld_nyt_val if not np.isnan(ld_nyt_val) else np.nan
        ratio = stata / ld_nyt_val if not np.isnan(ld_nyt_val) and ld_nyt_val != 0 else np.nan
        
        print(f"{r:>3} | {stata:>10.4f} | {ld_nyt_val:>10.4f} | {diff:>10.4f} | {ratio:>8.2f}x")
    
    # 6. 关键发现
    print("\n" + "=" * 90)
    print("关键发现")
    print("=" * 90)
    
    print("""
1. Stata csdid 和 Python Long Difference + not_yet_treated 应该产生相似结果
   (因为 csdid 使用 Long Difference 变换)

2. 如果两者一致但都比论文高，说明：
   - 论文可能使用了不同的估计方法 (如 IPWRA 而非简单差分)
   - 论文可能使用了不同的样本选择
   - 论文可能使用了不同的协变量处理

3. 下一步：
   - 使用 IPWRA 估计器而非简单差分
   - 检查论文是否使用了协变量调整
   - 检查论文的样本选择标准
""")


if __name__ == '__main__':
    main()

"""
Demean 变换详细诊断

检查 demean 变换的每个步骤，确保与论文公式一致。
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


def manual_demean_transform(df, y, ivar, tvar, gvar, cohort, period):
    """
    手动实现 demean 变换，严格按照论文公式
    
    论文公式 (Procedure 4.1):
    ẏ_{irg} = Y_{ir} - (1/(g-1)) * Σ_{s=1}^{g-1} Y_{is}
    
    其中 g 是 cohort，r 是 period
    """
    # 获取该 period 的数据
    period_data = df[df[tvar] == period].copy()
    
    # 对每个单位计算 pre-treatment mean
    pre_means = {}
    for unit in period_data[ivar].unique():
        unit_data = df[df[ivar] == unit]
        # 预处理期: t < g
        pre_data = unit_data[unit_data[tvar] < cohort]
        if len(pre_data) > 0:
            pre_means[unit] = pre_data[y].mean()
        else:
            pre_means[unit] = np.nan
    
    # 计算变换后的值
    period_data['ydot_manual'] = period_data.apply(
        lambda row: row[y] - pre_means.get(row[ivar], np.nan),
        axis=1
    )
    
    return period_data


def main():
    """运行详细诊断"""
    print("=" * 80)
    print("Demean 变换详细诊断")
    print("=" * 80)
    
    df = load_data()
    
    # 选择一个具体的 cohort 和 period 进行检查
    cohort = 1995
    period = 1995  # event_time = 0
    
    print(f"\n检查 cohort={cohort}, period={period} (event_time=0)")
    print("-" * 60)
    
    # 手动计算
    manual_result = manual_demean_transform(
        df, 'log_retail_emp', 'fips', 'year', 'g', cohort, period
    )
    
    # 使用 lwdid 的变换
    from lwdid.staggered.transformations import transform_staggered_demean
    
    df_transformed = transform_staggered_demean(
        df, y='log_retail_emp', ivar='fips', tvar='year', gvar='g'
    )
    
    col_name = f'ydot_g{cohort}_r{period}'
    
    # 比较结果
    period_data = df_transformed[df_transformed['year'] == period].copy()
    
    # 合并手动计算和 lwdid 计算的结果
    comparison = pd.merge(
        manual_result[['fips', 'ydot_manual']],
        period_data[['fips', col_name]],
        on='fips'
    )
    
    comparison['diff'] = comparison['ydot_manual'] - comparison[col_name]
    
    print(f"\n变换结果比较 (前 10 个单位):")
    print(comparison.head(10).to_string())
    
    print(f"\n差异统计:")
    print(f"  最大绝对差异: {comparison['diff'].abs().max():.10f}")
    print(f"  平均绝对差异: {comparison['diff'].abs().mean():.10f}")
    
    # 检查处理组和控制组的变换值
    print("\n" + "=" * 80)
    print("处理组 vs 控制组变换值分析")
    print("=" * 80)
    
    # 处理组: g == cohort
    treated_mask = period_data['g'] == cohort
    # 控制组 (never_treated): g == inf
    control_nt_mask = period_data['g'] == np.inf
    # 控制组 (not_yet_treated): g > period
    control_nyt_mask = (period_data['g'] > period) | (period_data['g'] == np.inf)
    
    treated_vals = period_data.loc[treated_mask, col_name].dropna()
    control_nt_vals = period_data.loc[control_nt_mask, col_name].dropna()
    control_nyt_vals = period_data.loc[control_nyt_mask, col_name].dropna()
    
    print(f"\n处理组 (g={cohort}):")
    print(f"  样本量: {len(treated_vals)}")
    print(f"  均值: {treated_vals.mean():.6f}")
    print(f"  标准差: {treated_vals.std():.6f}")
    
    print(f"\n控制组 (never_treated):")
    print(f"  样本量: {len(control_nt_vals)}")
    print(f"  均值: {control_nt_vals.mean():.6f}")
    print(f"  标准差: {control_nt_vals.std():.6f}")
    
    print(f"\n控制组 (not_yet_treated):")
    print(f"  样本量: {len(control_nyt_vals)}")
    print(f"  均值: {control_nyt_vals.mean():.6f}")
    print(f"  标准差: {control_nyt_vals.std():.6f}")
    
    # 简单差分 ATT
    att_nt = treated_vals.mean() - control_nt_vals.mean()
    att_nyt = treated_vals.mean() - control_nyt_vals.mean()
    
    print(f"\n简单差分 ATT:")
    print(f"  ATT (never_treated): {att_nt:.6f}")
    print(f"  ATT (not_yet_treated): {att_nyt:.6f}")
    
    # 检查多个 cohort 的情况
    print("\n" + "=" * 80)
    print("多个 Cohort 的 ATT 分析 (event_time=0)")
    print("=" * 80)
    
    cohorts = sorted([g for g in df['g'].unique() if pd.notna(g) and g != np.inf])
    
    att_results = []
    for g in cohorts[:10]:  # 只检查前 10 个 cohort
        g = int(g)
        r = g  # event_time = 0
        col = f'ydot_g{g}_r{r}'
        
        if col not in df_transformed.columns:
            continue
        
        period_data = df_transformed[df_transformed['year'] == r].copy()
        
        treated_mask = period_data['g'] == g
        control_nyt_mask = (period_data['g'] > r) | (period_data['g'] == np.inf)
        
        treated_vals = period_data.loc[treated_mask, col].dropna()
        control_vals = period_data.loc[control_nyt_mask, col].dropna()
        
        if len(treated_vals) > 0 and len(control_vals) > 0:
            att = treated_vals.mean() - control_vals.mean()
            att_results.append({
                'cohort': g,
                'n_treated': len(treated_vals),
                'n_control': len(control_vals),
                'treated_mean': treated_vals.mean(),
                'control_mean': control_vals.mean(),
                'att': att,
            })
    
    att_df = pd.DataFrame(att_results)
    print(att_df.to_string())
    
    # 检查原始数据的 pre-treatment 均值
    print("\n" + "=" * 80)
    print("原始数据 Pre-treatment 均值检查")
    print("=" * 80)
    
    # 选择一个具体的单位进行检查
    sample_unit = df[df['g'] == 1995]['fips'].iloc[0]
    unit_data = df[df['fips'] == sample_unit].sort_values('year')
    
    print(f"\n单位 {sample_unit} (cohort=1995) 的时间序列:")
    print(unit_data[['year', 'log_retail_emp', 'g']].to_string())
    
    pre_mean = unit_data[unit_data['year'] < 1995]['log_retail_emp'].mean()
    print(f"\nPre-treatment 均值 (year < 1995): {pre_mean:.6f}")
    
    y_1995 = unit_data[unit_data['year'] == 1995]['log_retail_emp'].values[0]
    print(f"Y_1995: {y_1995:.6f}")
    print(f"变换后值 (Y_1995 - pre_mean): {y_1995 - pre_mean:.6f}")


if __name__ == '__main__':
    main()

"""
深度诊断：追根溯源找出 Walmart 复现差异的根本原因

关键问题：Python 结果与论文 Table A4 差异 2-3 倍

诊断策略：
1. 验证 demean 变换公式是否正确
2. 验证控制组选择是否正确
3. 验证 ATT 计算是否正确
4. 验证 WATT 聚合是否正确
5. 对比 Stata csdid 的中间结果
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


def diagnose_step1_data_structure():
    """步骤1：验证数据结构"""
    print("=" * 80)
    print("步骤 1：验证数据结构")
    print("=" * 80)
    
    df = load_data()
    
    print(f"\n数据维度: {df.shape}")
    print(f"观测数: {len(df)}")
    print(f"县数量: {df['fips'].nunique()}")
    print(f"年份范围: {df['year'].min()} - {df['year'].max()}")
    
    # 检查 cohort 分布
    cohort_dist = df.groupby('g')['fips'].nunique()
    print(f"\nCohort 分布 (县数量):")
    for g, n in cohort_dist.items():
        if pd.notna(g) and g != np.inf:
            print(f"  g={int(g)}: {n} 县")
        elif g == np.inf:
            print(f"  Never-treated: {n} 县")
    
    # 检查面板平衡性
    obs_per_unit = df.groupby('fips').size()
    print(f"\n每个县的观测数: min={obs_per_unit.min()}, max={obs_per_unit.max()}, mean={obs_per_unit.mean():.1f}")
    
    return df


def diagnose_step2_demean_formula():
    """步骤2：验证 demean 变换公式"""
    print("\n" + "=" * 80)
    print("步骤 2：验证 demean 变换公式")
    print("=" * 80)
    
    df = load_data()
    
    # 选择一个具体的 cohort 和单元进行验证
    test_cohort = 2000
    test_units = df[df['g'] == test_cohort]['fips'].unique()[:3]
    
    print(f"\n测试 cohort g={test_cohort}")
    print(f"预处理期: t = 1977, ..., {test_cohort - 1}")
    print(f"预处理期数量: {test_cohort - 1977} 个时期")
    
    for unit_id in test_units:
        unit_data = df[df['fips'] == unit_id].sort_values('year')
        
        # 手动计算预处理期均值
        pre_data = unit_data[unit_data['year'] < test_cohort]
        pre_mean = pre_data['log_retail_emp'].mean()
        
        # 计算 r=0 (即 year=2000) 的变换值
        y_2000 = unit_data[unit_data['year'] == test_cohort]['log_retail_emp'].values[0]
        ydot_manual = y_2000 - pre_mean
        
        print(f"\n单元 {unit_id}:")
        print(f"  预处理期均值: {pre_mean:.6f}")
        print(f"  Y(2000): {y_2000:.6f}")
        print(f"  ydot(g=2000, r=2000): {ydot_manual:.6f}")
    
    return df


def diagnose_step3_control_group():
    """步骤3：验证控制组选择"""
    print("\n" + "=" * 80)
    print("步骤 3：验证控制组选择")
    print("=" * 80)
    
    df = load_data()
    
    # 对于 cohort g=2000, period r=2000 (event_time=0)
    test_cohort = 2000
    test_period = 2000
    
    print(f"\n测试 cohort g={test_cohort}, period r={test_period}")
    
    # 处理组
    treated_units = df[(df['g'] == test_cohort)]['fips'].unique()
    n_treated = len(treated_units)
    
    # 控制组选项
    # 1. never_treated
    never_treated_units = df[df['g'] == np.inf]['fips'].unique()
    n_never_treated = len(never_treated_units)
    
    # 2. not_yet_treated (g > r)
    not_yet_treated_units = df[(df['g'] > test_period) | (df['g'] == np.inf)]['fips'].unique()
    n_not_yet_treated = len(not_yet_treated_units)
    
    print(f"\n处理组 (g={test_cohort}): {n_treated} 县")
    print(f"控制组 (never_treated): {n_never_treated} 县")
    print(f"控制组 (not_yet_treated, g > {test_period}): {n_not_yet_treated} 县")
    
    # 论文使用的是 not_yet_treated
    print(f"\n论文使用 not_yet_treated 控制组")
    
    return df


def diagnose_step4_att_calculation():
    """步骤4：验证 ATT 计算"""
    print("\n" + "=" * 80)
    print("步骤 4：验证 ATT 计算 (简单差分)")
    print("=" * 80)
    
    df = load_data()
    
    # 手动计算 demean 变换
    T_min = int(df['year'].min())
    T_max = int(df['year'].max())
    
    cohorts = sorted([int(g) for g in df['g'].unique() if pd.notna(g) and g != np.inf])
    
    print(f"\n时间范围: {T_min} - {T_max}")
    print(f"Cohorts: {cohorts}")
    
    # 计算每个单元的预处理期均值
    unit_pre_means = {}
    for g in cohorts:
        pre_mask = df['year'] < g
        pre_means = df[pre_mask].groupby('fips')['log_retail_emp'].mean()
        unit_pre_means[g] = pre_means
    
    # 计算 ATT(g, r) 使用简单差分
    att_results = []
    
    for g in cohorts:
        for r in range(g, T_max + 1):
            # 获取 period r 的数据
            period_data = df[df['year'] == r].copy()
            
            # 计算变换后的 Y
            period_data['ydot'] = period_data.apply(
                lambda row: row['log_retail_emp'] - unit_pre_means[g].get(row['fips'], np.nan),
                axis=1
            )
            
            # 处理组
            treated_mask = period_data['g'] == g
            
            # 控制组 (not_yet_treated)
            control_mask = (period_data['g'] > r) | (period_data['g'] == np.inf)
            
            treated_vals = period_data[treated_mask]['ydot'].dropna()
            control_vals = period_data[control_mask]['ydot'].dropna()
            
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
                'treated_mean': treated_vals.mean(),
                'control_mean': control_vals.mean(),
            })
    
    att_df = pd.DataFrame(att_results)
    
    # 显示部分结果
    print(f"\n部分 ATT(g, r) 结果:")
    print(att_df[att_df['event_time'] == 0][['cohort', 'period', 'att', 'n_treated', 'n_control']].head(10))
    
    return att_df


def diagnose_step5_watt_aggregation():
    """步骤5：验证 WATT 聚合"""
    print("\n" + "=" * 80)
    print("步骤 5：验证 WATT 聚合")
    print("=" * 80)
    
    df = load_data()
    att_df = diagnose_step4_att_calculation()
    
    # 计算 cohort 大小
    cohort_sizes = df[df['g'] != np.inf].groupby('g')['fips'].nunique().to_dict()
    
    print(f"\nCohort 大小:")
    for g, n in sorted(cohort_sizes.items()):
        print(f"  g={int(g)}: {n} 县")
    
    # 计算 WATT
    watt_results = {}
    
    for event_time in sorted(att_df['event_time'].unique()):
        if event_time < 0:
            continue
        
        subset = att_df[att_df['event_time'] == event_time].copy()
        subset = subset[subset['att'].notna()]
        
        if len(subset) == 0:
            continue
        
        # 计算权重 (使用贡献 cohort 的大小)
        subset['weight'] = subset['cohort'].map(cohort_sizes)
        total_weight = subset['weight'].sum()
        subset['norm_weight'] = subset['weight'] / total_weight
        
        # 计算 WATT
        watt = (subset['att'] * subset['norm_weight']).sum()
        watt_results[int(event_time)] = watt
        
        if event_time in [0, 1, 13]:
            print(f"\nEvent time r={int(event_time)}:")
            print(f"  贡献 cohorts: {len(subset)}")
            print(f"  WATT: {watt:.6f}")
            print(f"  论文: {PAPER_DEMEAN.get(int(event_time), np.nan):.3f}")
            print(f"  比率: {watt / PAPER_DEMEAN.get(int(event_time), 1):.2f}x")
    
    return watt_results


def diagnose_step6_compare_methods():
    """步骤6：对比不同方法"""
    print("\n" + "=" * 80)
    print("步骤 6：对比不同变换方法")
    print("=" * 80)
    
    df = load_data()
    
    # 方法1: Full Demean (使用所有预处理期)
    # 方法2: Long Difference (只使用 g-1 期)
    
    T_min = int(df['year'].min())
    T_max = int(df['year'].max())
    cohorts = sorted([int(g) for g in df['g'].unique() if pd.notna(g) and g != np.inf])
    cohort_sizes = df[df['g'] != np.inf].groupby('g')['fips'].nunique().to_dict()
    cohort_sizes = {int(k): v for k, v in cohort_sizes.items()}
    
    # 计算两种方法的预处理期值
    unit_full_demean = {}  # 使用所有预处理期的均值
    unit_long_diff = {}    # 只使用 g-1 期的值
    
    for g in cohorts:
        # Full demean: 使用 t < g 的所有期
        pre_mask = df['year'] < g
        unit_full_demean[g] = df[pre_mask].groupby('fips')['log_retail_emp'].mean()
        
        # Long difference: 只使用 t = g-1
        base_mask = df['year'] == (g - 1)
        unit_long_diff[g] = df[base_mask].set_index('fips')['log_retail_emp']
    
    # 计算两种方法的 ATT
    def compute_att(unit_baseline, method_name):
        att_results = []
        for g in cohorts:
            for r in range(g, T_max + 1):
                period_data = df[df['year'] == r].copy()
                
                period_data['ydot'] = period_data.apply(
                    lambda row: row['log_retail_emp'] - unit_baseline[g].get(row['fips'], np.nan),
                    axis=1
                )
                
                treated_mask = period_data['g'] == g
                control_mask = (period_data['g'] > r) | (period_data['g'] == np.inf)
                
                treated_vals = period_data[treated_mask]['ydot'].dropna()
                control_vals = period_data[control_mask]['ydot'].dropna()
                
                if len(treated_vals) == 0 or len(control_vals) == 0:
                    continue
                
                att = treated_vals.mean() - control_vals.mean()
                att_results.append({
                    'cohort': g, 'period': r, 'event_time': r - g, 'att': att
                })
        
        return pd.DataFrame(att_results)
    
    att_full = compute_att(unit_full_demean, "Full Demean")
    att_long = compute_att(unit_long_diff, "Long Difference")
    
    # 计算 WATT
    def compute_watt(att_df):
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
    
    watt_full = compute_watt(att_full)
    watt_long = compute_watt(att_long)
    
    print(f"\n{'r':>3} | {'Full Demean':>12} | {'Long Diff':>12} | {'Paper':>10} | {'Full/Paper':>10} | {'Long/Paper':>10}")
    print("-" * 75)
    
    for r in range(14):
        full = watt_full.get(r, np.nan)
        long = watt_long.get(r, np.nan)
        paper = PAPER_DEMEAN.get(r, np.nan)
        
        ratio_full = full / paper if paper > 0 else np.nan
        ratio_long = long / paper if paper > 0 else np.nan
        
        print(f"{r:>3} | {full:>12.4f} | {long:>12.4f} | {paper:>10.3f} | {ratio_full:>10.2f}x | {ratio_long:>10.2f}x")


def diagnose_step7_check_paper_method():
    """步骤7：检查论文可能使用的方法"""
    print("\n" + "=" * 80)
    print("步骤 7：检查论文可能使用的方法")
    print("=" * 80)
    
    df = load_data()
    
    # 论文 Section 6 提到使用 "rolling" 方法
    # 可能的解释：
    # 1. 论文使用的是 "pre-treatment" 变换，而不是 "full" 变换
    # 2. 即：对于 event_time r，使用 t = g-1 到 g+r-1 的均值作为基准
    
    print("\n假设1: 论文使用 'pre-treatment' 变换")
    print("即：对于 event_time r，基准期是 t = g-1 (锚点)")
    print("这等价于 Long Difference 方法")
    
    print("\n假设2: 论文使用不同的权重计算")
    print("可能使用观测数而非县数作为权重")
    
    # 检查使用观测数作为权重
    T_max = int(df['year'].max())
    cohorts = sorted([g for g in df['g'].unique() if pd.notna(g) and g != np.inf])
    
    # 使用观测数作为权重
    cohort_obs = df[df['g'] != np.inf].groupby('g').size().to_dict()
    cohort_units = df[df['g'] != np.inf].groupby('g')['fips'].nunique().to_dict()
    
    print(f"\n权重对比:")
    print(f"{'Cohort':>8} | {'县数':>8} | {'观测数':>8}")
    print("-" * 30)
    for g in sorted(cohort_units.keys()):
        print(f"{int(g):>8} | {cohort_units[g]:>8} | {cohort_obs[g]:>8}")


def main():
    """主诊断流程"""
    print("=" * 80)
    print("深度诊断：Walmart 复现差异根本原因分析")
    print("=" * 80)
    
    diagnose_step1_data_structure()
    diagnose_step2_demean_formula()
    diagnose_step3_control_group()
    diagnose_step4_att_calculation()
    diagnose_step5_watt_aggregation()
    diagnose_step6_compare_methods()
    diagnose_step7_check_paper_method()
    
    print("\n" + "=" * 80)
    print("诊断总结")
    print("=" * 80)
    print("""
关键发现：
1. Full Demean 和 Long Difference 都比论文高 ~2-3x
2. 两种方法的结果接近，说明变换方法不是主要差异来源
3. 需要进一步检查：
   - 论文是否使用了不同的样本选择
   - 论文是否使用了不同的权重计算
   - 论文是否使用了不同的控制组定义
""")


if __name__ == '__main__':
    main()

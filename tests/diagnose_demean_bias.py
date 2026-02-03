"""
Demean 偏差诊断脚本

分析 Python 实现与论文 Table A4 之间 demean 估计值差异的根因。

可能的差异来源:
1. 控制组选择 (never_treated vs not_yet_treated)
2. IPWRA 权重归一化方式
3. 倾向得分裁剪阈值
4. 结果模型规范
5. WATT 聚合权重计算
"""

import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

# 添加包路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from lwdid import lwdid

warnings.filterwarnings('ignore')


def load_data():
    """加载 Walmart 数据"""
    data_path = Path(__file__).parent.parent / 'data' / 'walmart.csv'
    return pd.read_csv(data_path)


def compute_watt(results, df):
    """计算 WATT"""
    att_ct = results.att_by_cohort_time.copy()
    if att_ct is None or len(att_ct) == 0:
        return pd.DataFrame()
    
    cohort_sizes = df[df['g'] != np.inf].groupby('g')['fips'].nunique().to_dict()
    att_ct['weight'] = att_ct['cohort'].map(cohort_sizes).fillna(0)
    
    watt_list = []
    for event_time in sorted(att_ct['event_time'].unique()):
        subset = att_ct[att_ct['event_time'] == event_time].copy()
        subset = subset[subset['att'].notna()]
        if len(subset) == 0:
            continue
        total_weight = subset['weight'].sum()
        if total_weight == 0:
            continue
        subset['norm_weight'] = subset['weight'] / total_weight
        watt = (subset['att'] * subset['norm_weight']).sum()
        watt_se = np.sqrt((subset['se']**2 * subset['norm_weight']**2).sum())
        watt_list.append({
            'event_time': int(event_time),
            'watt': watt,
            'se': watt_se,
        })
    return pd.DataFrame(watt_list)


# 论文参考值
PAPER_DEMEAN = {
    0: 0.018, 1: 0.045, 2: 0.038, 3: 0.032, 4: 0.031,
    5: 0.036, 6: 0.040, 7: 0.054, 8: 0.062, 9: 0.063,
    10: 0.081, 11: 0.083, 12: 0.080, 13: 0.107,
}


def diagnose_control_group():
    """诊断 1: 控制组选择的影响"""
    print("\n" + "=" * 70)
    print("诊断 1: 控制组选择的影响")
    print("=" * 70)
    
    df = load_data()
    controls = ['share_pop_poverty_78_above', 'share_pop_ind_manuf', 'share_school_some_hs']
    
    # never_treated 控制组
    results_nt = lwdid(
        data=df, y='log_retail_emp', ivar='fips', tvar='year', gvar='g',
        rolling='demean', estimator='ipwra', controls=controls,
        control_group='never_treated', aggregate='none',
    )
    watt_nt = compute_watt(results_nt, df)
    
    # not_yet_treated 控制组
    results_nyt = lwdid(
        data=df, y='log_retail_emp', ivar='fips', tvar='year', gvar='g',
        rolling='demean', estimator='ipwra', controls=controls,
        control_group='not_yet_treated', aggregate='none',
    )
    watt_nyt = compute_watt(results_nyt, df)
    
    print(f"\n{'r':>3} | {'never_treated':>14} | {'not_yet_treated':>15} | {'Paper':>8} | {'NT diff':>10} | {'NYT diff':>10}")
    print("-" * 80)
    
    for r in range(14):
        nt_val = watt_nt[watt_nt['event_time'] == r]['watt'].values
        nyt_val = watt_nyt[watt_nyt['event_time'] == r]['watt'].values
        paper_val = PAPER_DEMEAN.get(r, np.nan)
        
        nt_str = f"{nt_val[0]:.4f}" if len(nt_val) > 0 else "N/A"
        nyt_str = f"{nyt_val[0]:.4f}" if len(nyt_val) > 0 else "N/A"
        
        nt_diff = f"{nt_val[0] - paper_val:+.4f}" if len(nt_val) > 0 else "N/A"
        nyt_diff = f"{nyt_val[0] - paper_val:+.4f}" if len(nyt_val) > 0 else "N/A"
        
        print(f"{r:>3} | {nt_str:>14} | {nyt_str:>15} | {paper_val:>8.3f} | {nt_diff:>10} | {nyt_diff:>10}")
    
    # 计算平均差异
    nt_diffs = []
    nyt_diffs = []
    for r in range(14):
        nt_val = watt_nt[watt_nt['event_time'] == r]['watt'].values
        nyt_val = watt_nyt[watt_nyt['event_time'] == r]['watt'].values
        paper_val = PAPER_DEMEAN.get(r, np.nan)
        if len(nt_val) > 0:
            nt_diffs.append(abs(nt_val[0] - paper_val))
        if len(nyt_val) > 0:
            nyt_diffs.append(abs(nyt_val[0] - paper_val))
    
    print("-" * 80)
    print(f"平均绝对差异: never_treated={np.mean(nt_diffs):.4f}, not_yet_treated={np.mean(nyt_diffs):.4f}")
    
    return watt_nt, watt_nyt


def diagnose_estimator_comparison():
    """诊断 2: 不同估计器的比较"""
    print("\n" + "=" * 70)
    print("诊断 2: 不同估计器的比较 (RA vs IPW vs IPWRA)")
    print("=" * 70)
    
    df = load_data()
    controls = ['share_pop_poverty_78_above', 'share_pop_ind_manuf', 'share_school_some_hs']
    
    estimators = ['ra', 'ipw', 'ipwra']
    results_dict = {}
    
    for est in estimators:
        print(f"运行 {est.upper()} 估计器...")
        try:
            results = lwdid(
                data=df, y='log_retail_emp', ivar='fips', tvar='year', gvar='g',
                rolling='demean', estimator=est, controls=controls,
                control_group='never_treated', aggregate='none',
            )
            watt = compute_watt(results, df)
            if len(watt) > 0:
                results_dict[est] = watt
            else:
                print(f"  {est.upper()} 返回空结果")
                results_dict[est] = pd.DataFrame({'event_time': [], 'watt': []})
        except Exception as e:
            print(f"  {est.upper()} 失败: {e}")
            results_dict[est] = pd.DataFrame({'event_time': [], 'watt': []})
    
    print(f"\n{'r':>3} | {'RA':>10} | {'IPW':>10} | {'IPWRA':>10} | {'Paper':>8}")
    print("-" * 60)
    
    for r in range(14):
        vals = []
        for est in estimators:
            watt = results_dict[est]
            if 'event_time' in watt.columns:
                val = watt[watt['event_time'] == r]['watt'].values
                vals.append(f"{val[0]:.4f}" if len(val) > 0 else "N/A")
            else:
                vals.append("N/A")
        paper_val = PAPER_DEMEAN.get(r, np.nan)
        print(f"{r:>3} | {vals[0]:>10} | {vals[1]:>10} | {vals[2]:>10} | {paper_val:>8.3f}")
    
    return results_dict


def diagnose_single_cohort():
    """诊断 3: 单个 cohort 的详细分析"""
    print("\n" + "=" * 70)
    print("诊断 3: 单个 Cohort (g=1990) 的详细分析")
    print("=" * 70)
    
    df = load_data()
    controls = ['share_pop_poverty_78_above', 'share_pop_ind_manuf', 'share_school_some_hs']
    
    # 运行估计
    results = lwdid(
        data=df, y='log_retail_emp', ivar='fips', tvar='year', gvar='g',
        rolling='demean', estimator='ipwra', controls=controls,
        control_group='never_treated', aggregate='none',
    )
    
    att_ct = results.att_by_cohort_time
    
    # 分析 cohort 1990
    cohort_1990 = att_ct[att_ct['cohort'] == 1990]
    
    print("\nCohort 1990 的 ATT 估计:")
    print(cohort_1990[['period', 'event_time', 'att', 'se', 'n_treated', 'n_control']].to_string(index=False))
    
    # 检查 cohort 大小
    cohort_sizes = df[df['g'] != np.inf].groupby('g')['fips'].nunique()
    print(f"\nCohort 1990 的单位数: {cohort_sizes.get(1990, 'N/A')}")
    print(f"Never-treated 单位数: {(df['g'] == np.inf).sum() // 23}")  # 23 年
    
    return cohort_1990


def diagnose_transformation():
    """诊断 4: 变换后数据的检查"""
    print("\n" + "=" * 70)
    print("诊断 4: Demean 变换后数据的检查")
    print("=" * 70)
    
    df = load_data()
    
    # 手动应用 demean 变换
    from lwdid.staggered.transformations import transform_staggered_demean
    
    df_transformed = transform_staggered_demean(
        df, y='log_retail_emp', ivar='fips', tvar='year', gvar='g'
    )
    
    # 检查 cohort 1990, period 1990 (event_time=0) 的变换结果
    col_name = 'ydot_g1990_r1990'
    if col_name in df_transformed.columns:
        transformed_vals = df_transformed[df_transformed['year'] == 1990][col_name].dropna()
        print(f"\n{col_name} 统计:")
        print(f"  样本数: {len(transformed_vals)}")
        print(f"  均值: {transformed_vals.mean():.6f}")
        print(f"  标准差: {transformed_vals.std():.6f}")
        print(f"  最小值: {transformed_vals.min():.6f}")
        print(f"  最大值: {transformed_vals.max():.6f}")
        
        # 分组统计
        df_1990 = df_transformed[df_transformed['year'] == 1990].copy()
        df_1990['is_treated'] = df_1990['g'] == 1990
        df_1990['is_never_treated'] = df_1990['g'] == np.inf
        
        treated_vals = df_1990[df_1990['is_treated']][col_name].dropna()
        nt_vals = df_1990[df_1990['is_never_treated']][col_name].dropna()
        
        print(f"\n  处理组 (g=1990) 均值: {treated_vals.mean():.6f} (n={len(treated_vals)})")
        print(f"  控制组 (never-treated) 均值: {nt_vals.mean():.6f} (n={len(nt_vals)})")
        print(f"  差值 (ATT 近似): {treated_vals.mean() - nt_vals.mean():.6f}")


def main():
    """运行所有诊断"""
    print("=" * 70)
    print("Demean 偏差诊断报告")
    print("=" * 70)
    
    # 诊断 1: 控制组选择
    diagnose_control_group()
    
    # 诊断 2: 估计器比较
    diagnose_estimator_comparison()
    
    # 诊断 3: 单个 cohort 分析
    diagnose_single_cohort()
    
    # 诊断 4: 变换检查
    diagnose_transformation()
    
    print("\n" + "=" * 70)
    print("诊断完成")
    print("=" * 70)


if __name__ == '__main__':
    main()

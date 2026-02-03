"""
深度诊断：分析 Demean 方法与论文差异的根本原因

检查以下可能的原因：
1. IPWRA vs 简单均值差
2. 控制组的变换方式
3. 权重计算方式
4. 数据预处理差异
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


def compute_att_simple_diff(treated_vals, control_vals):
    """简单均值差"""
    return treated_vals.mean() - control_vals.mean()


def compute_att_ipwra(treated_data, control_data, y_col, x_cols):
    """IPWRA 估计量"""
    from sklearn.linear_model import LogisticRegression, LinearRegression
    
    # 合并数据
    treated_data = treated_data.copy()
    control_data = control_data.copy()
    treated_data['_D'] = 1
    control_data['_D'] = 0
    all_data = pd.concat([treated_data, control_data], ignore_index=True)
    
    # 检查是否有足够的数据
    if len(treated_data) < 2 or len(control_data) < 2:
        return np.nan
    
    # 检查协变量是否有变异
    X = all_data[x_cols].values
    if X.std(axis=0).min() < 1e-10:
        # 如果协变量没有变异，退化为简单均值差
        return treated_data[y_col].mean() - control_data[y_col].mean()
    
    try:
        # 估计倾向得分
        ps_model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
        ps_model.fit(X, all_data['_D'].values)
        pscores = ps_model.predict_proba(X)[:, 1]
        pscores = np.clip(pscores, 0.01, 0.99)
        
        # 估计结果模型（在控制组上）
        X_control = control_data[x_cols].values
        y_control = control_data[y_col].values
        outcome_model = LinearRegression()
        outcome_model.fit(X_control, y_control)
        
        # 预测所有单元的潜在结果
        m0_hat = outcome_model.predict(X)
        
        # IPWRA 估计量
        D = all_data['_D'].values
        Y = all_data[y_col].values
        
        treat_mask = D == 1
        control_mask = D == 0
        
        # 处理组部分
        treat_term = (Y[treat_mask] - m0_hat[treat_mask]).mean()
        
        # 控制组部分（加权）
        weights = pscores / (1 - pscores)
        weights_control = weights[control_mask]
        residuals_control = Y[control_mask] - m0_hat[control_mask]
        
        weights_sum = weights_control.sum()
        if weights_sum <= 0:
            return np.nan
        
        control_term = (weights_control * residuals_control).sum() / weights_sum
        
        return treat_term - control_term
        
    except Exception as e:
        # 如果 IPWRA 失败，返回简单均值差
        return treated_data[y_col].mean() - control_data[y_col].mean()


def analyze_single_cohort_period(df, g, r, method='demean'):
    """分析单个 (g, r) 对的详细信息"""
    T_min = int(df['year'].min())
    
    # 获取该时期的数据
    period_data = df[df['year'] == r].copy()
    
    # 计算变换
    if method == 'demean':
        # 计算预处理期均值
        pre_mask = df['year'] < g
        pre_means = df[pre_mask].groupby('fips')['log_retail_emp'].mean()
        period_data['y_transformed'] = period_data.apply(
            lambda row: row['log_retail_emp'] - pre_means.get(row['fips'], np.nan),
            axis=1
        )
    elif method == 'long_diff':
        # Long difference: Y_ir - Y_{i,g-1}
        base_period = g - 1
        base_data = df[df['year'] == base_period].set_index('fips')['log_retail_emp']
        period_data['y_transformed'] = period_data.apply(
            lambda row: row['log_retail_emp'] - base_data.get(row['fips'], np.nan),
            axis=1
        )
    
    # 定义处理组和控制组
    treated_mask = period_data['g'] == g
    control_mask = (period_data['g'] > r) | (period_data['g'] == np.inf)
    
    treated_data = period_data[treated_mask].dropna(subset=['y_transformed'])
    control_data = period_data[control_mask].dropna(subset=['y_transformed'])
    
    n_treated = len(treated_data)
    n_control = len(control_data)
    
    if n_treated == 0 or n_control == 0:
        return None
    
    # 计算 ATT
    att_simple = compute_att_simple_diff(
        treated_data['y_transformed'], 
        control_data['y_transformed']
    )
    
    # 尝试 IPWRA（如果有协变量）
    x_cols = ['pop']  # 使用人口作为协变量
    if all(col in period_data.columns for col in x_cols):
        att_ipwra = compute_att_ipwra(treated_data, control_data, 'y_transformed', x_cols)
    else:
        att_ipwra = np.nan
    
    return {
        'cohort': g,
        'period': r,
        'event_time': r - g,
        'n_treated': n_treated,
        'n_control': n_control,
        'att_simple': att_simple,
        'att_ipwra': att_ipwra,
        'treated_mean': treated_data['y_transformed'].mean(),
        'control_mean': control_data['y_transformed'].mean(),
    }


def main():
    print("=" * 100)
    print("深度诊断：分析 Demean 方法与论文差异的根本原因")
    print("=" * 100)
    
    df = load_data()
    
    T_min = int(df['year'].min())
    T_max = int(df['year'].max())
    cohorts = sorted([int(g) for g in df['g'].unique() if pd.notna(g) and g != np.inf])
    cohort_sizes = {int(k): v for k, v in df[df['g'] != np.inf].groupby('g')['fips'].nunique().to_dict().items()}
    
    print(f"\n数据信息:")
    print(f"  时间范围: {T_min} - {T_max}")
    print(f"  Cohorts: {cohorts}")
    print(f"  Cohort 大小: {cohort_sizes}")
    
    # 分析每个 (g, r) 对
    print("\n" + "=" * 100)
    print("分析 Demean 方法的每个 (g, r) 对")
    print("=" * 100)
    
    results = []
    for g in cohorts:
        for r in range(g, T_max + 1):
            result = analyze_single_cohort_period(df, g, r, method='demean')
            if result:
                results.append(result)
    
    results_df = pd.DataFrame(results)
    
    # 计算 WATT
    print("\n" + "=" * 100)
    print("WATT 计算（使用简单均值差）")
    print("=" * 100)
    
    watt_simple = {}
    watt_ipwra = {}
    
    for event_time in sorted(results_df['event_time'].unique()):
        if event_time < 0:
            continue
        subset = results_df[results_df['event_time'] == event_time].copy()
        subset = subset[subset['att_simple'].notna()]
        if len(subset) == 0:
            continue
        
        # 计算权重
        subset['weight'] = subset['cohort'].map(cohort_sizes)
        total_weight = subset['weight'].sum()
        subset['norm_weight'] = subset['weight'] / total_weight
        
        # 简单均值差的 WATT
        watt_simple[int(event_time)] = (subset['att_simple'] * subset['norm_weight']).sum()
        
        # IPWRA 的 WATT
        subset_ipwra = subset[subset['att_ipwra'].notna()]
        if len(subset_ipwra) > 0:
            watt_ipwra[int(event_time)] = (subset_ipwra['att_ipwra'] * subset_ipwra['norm_weight']).sum()
    
    print(f"\n{'r':>3} | {'WATT Simple':>12} | {'WATT IPWRA':>12} | {'Paper':>8} | {'Simple/Paper':>12} | {'IPWRA/Paper':>12}")
    print("-" * 80)
    
    for r in range(14):
        ws = watt_simple.get(r, np.nan)
        wi = watt_ipwra.get(r, np.nan)
        paper = PAPER_DEMEAN.get(r, np.nan)
        
        ratio_s = ws / paper if paper > 0 else np.nan
        ratio_i = wi / paper if paper > 0 else np.nan
        
        print(f"{r:>3} | {ws:>12.4f} | {wi:>12.4f} | {paper:>8.3f} | {ratio_s:>12.2f}x | {ratio_i:>12.2f}x")
    
    # 分析 Long Difference
    print("\n" + "=" * 100)
    print("分析 Long Difference 方法")
    print("=" * 100)
    
    results_ld = []
    for g in cohorts:
        for r in range(g, T_max + 1):
            result = analyze_single_cohort_period(df, g, r, method='long_diff')
            if result:
                results_ld.append(result)
    
    results_ld_df = pd.DataFrame(results_ld)
    
    watt_ld = {}
    for event_time in sorted(results_ld_df['event_time'].unique()):
        if event_time < 0:
            continue
        subset = results_ld_df[results_ld_df['event_time'] == event_time].copy()
        subset = subset[subset['att_simple'].notna()]
        if len(subset) == 0:
            continue
        
        subset['weight'] = subset['cohort'].map(cohort_sizes)
        total_weight = subset['weight'].sum()
        subset['norm_weight'] = subset['weight'] / total_weight
        
        watt_ld[int(event_time)] = (subset['att_simple'] * subset['norm_weight']).sum()
    
    print(f"\n{'r':>3} | {'WATT LD':>12} | {'Paper':>8} | {'LD/Paper':>12}")
    print("-" * 50)
    
    for r in range(14):
        wl = watt_ld.get(r, np.nan)
        paper = PAPER_DEMEAN.get(r, np.nan)
        ratio = wl / paper if paper > 0 else np.nan
        print(f"{r:>3} | {wl:>12.4f} | {paper:>8.3f} | {ratio:>12.2f}x")
    
    # 深入分析：检查单个 cohort 的结果
    print("\n" + "=" * 100)
    print("深入分析：检查单个 cohort 的 ATT")
    print("=" * 100)
    
    # 选择 r=0 进行详细分析
    r = 0
    print(f"\n分析 event_time = {r} 的各 cohort ATT:")
    print(f"{'Cohort':>8} | {'N_treated':>10} | {'N_control':>10} | {'ATT':>10} | {'Weight':>8}")
    print("-" * 60)
    
    subset = results_df[results_df['event_time'] == r].copy()
    for _, row in subset.iterrows():
        g = int(row['cohort'])
        weight = cohort_sizes.get(g, 0)
        print(f"{g:>8} | {int(row['n_treated']):>10} | {int(row['n_control']):>10} | {row['att_simple']:>10.4f} | {weight:>8}")
    
    # 计算加权平均
    subset['weight'] = subset['cohort'].map(cohort_sizes)
    total_weight = subset['weight'].sum()
    subset['norm_weight'] = subset['weight'] / total_weight
    watt_r0 = (subset['att_simple'] * subset['norm_weight']).sum()
    print(f"\nWATT(r=0) = {watt_r0:.4f}")
    print(f"Paper(r=0) = {PAPER_DEMEAN[0]:.3f}")
    print(f"Ratio = {watt_r0 / PAPER_DEMEAN[0]:.2f}x")


if __name__ == '__main__':
    main()

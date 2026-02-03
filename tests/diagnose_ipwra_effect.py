"""
诊断：IPWRA 估计器对结果的影响

论文 Section 6 明确使用了 IPWRA 估计器和协变量：
- share_pop_poverty_78_above
- share_pop_ind_manuf
- share_school_some_hs

本脚本比较简单差分和 IPWRA 估计器的结果差异。
"""

import sys
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

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


def compute_simple_att(period_data, y_col, gvar, g, control_group='not_yet_treated'):
    """计算简单 ATT (无协变量调整)"""
    treated_mask = period_data[gvar] == g
    
    if control_group == 'never_treated':
        control_mask = period_data[gvar] == np.inf
    else:
        r = period_data['year'].iloc[0]
        control_mask = (period_data[gvar] > r) | (period_data[gvar] == np.inf)
    
    treated_vals = period_data[treated_mask][y_col].dropna()
    control_vals = period_data[control_mask][y_col].dropna()
    
    if len(treated_vals) == 0 or len(control_vals) == 0:
        return np.nan
    
    return treated_vals.mean() - control_vals.mean()


def compute_ra_att(period_data, y_col, gvar, g, covariates, control_group='not_yet_treated'):
    """计算 Regression Adjustment ATT"""
    treated_mask = period_data[gvar] == g
    
    if control_group == 'never_treated':
        control_mask = period_data[gvar] == np.inf
    else:
        r = period_data['year'].iloc[0]
        control_mask = (period_data[gvar] > r) | (period_data[gvar] == np.inf)
    
    # 准备数据
    treated_data = period_data[treated_mask].dropna(subset=[y_col] + covariates)
    control_data = period_data[control_mask].dropna(subset=[y_col] + covariates)
    
    if len(treated_data) == 0 or len(control_data) == 0:
        return np.nan
    
    # 在控制组上拟合回归模型
    X_control = sm.add_constant(control_data[covariates])
    y_control = control_data[y_col]
    
    try:
        model = sm.OLS(y_control, X_control).fit()
    except Exception:
        return np.nan
    
    # 预测处理组的反事实结果
    X_treated = sm.add_constant(treated_data[covariates])
    y_treated = treated_data[y_col]
    y_counterfactual = model.predict(X_treated)
    
    # ATT = mean(Y_treated - Y_counterfactual)
    att = (y_treated.values - y_counterfactual.values).mean()
    
    return att


def compute_ipw_att(period_data, y_col, gvar, g, covariates, control_group='not_yet_treated'):
    """计算 IPW ATT"""
    treated_mask = period_data[gvar] == g
    
    if control_group == 'never_treated':
        control_mask = period_data[gvar] == np.inf
    else:
        r = period_data['year'].iloc[0]
        control_mask = (period_data[gvar] > r) | (period_data[gvar] == np.inf)
    
    # 准备数据
    sample_mask = treated_mask | control_mask
    sample_data = period_data[sample_mask].dropna(subset=[y_col] + covariates).copy()
    
    if len(sample_data) == 0:
        return np.nan
    
    sample_data['D'] = (sample_data[gvar] == g).astype(int)
    
    # 估计倾向得分
    X = sm.add_constant(sample_data[covariates])
    y = sample_data['D']
    
    try:
        ps_model = sm.Logit(y, X).fit(disp=0)
        ps = ps_model.predict(X)
    except Exception:
        return np.nan
    
    # 裁剪倾向得分
    ps = np.clip(ps, 0.01, 0.99)
    
    # IPW 估计
    treated_data = sample_data[sample_data['D'] == 1]
    control_data = sample_data[sample_data['D'] == 0]
    
    ps_treated = ps[sample_data['D'] == 1]
    ps_control = ps[sample_data['D'] == 0]
    
    # 处理组均值
    y_treated_mean = treated_data[y_col].mean()
    
    # 加权控制组均值
    weights = ps_control / (1 - ps_control)
    y_control_weighted = (control_data[y_col] * weights).sum() / weights.sum()
    
    att = y_treated_mean - y_control_weighted
    
    return att


def compute_ipwra_att(period_data, y_col, gvar, g, covariates, control_group='not_yet_treated'):
    """计算 IPWRA (Doubly Robust) ATT"""
    treated_mask = period_data[gvar] == g
    
    if control_group == 'never_treated':
        control_mask = period_data[gvar] == np.inf
    else:
        r = period_data['year'].iloc[0]
        control_mask = (period_data[gvar] > r) | (period_data[gvar] == np.inf)
    
    # 准备数据
    sample_mask = treated_mask | control_mask
    sample_data = period_data[sample_mask].dropna(subset=[y_col] + covariates).copy()
    
    if len(sample_data) == 0:
        return np.nan
    
    sample_data['D'] = (sample_data[gvar] == g).astype(int)
    
    # 1. 估计倾向得分
    X_ps = sm.add_constant(sample_data[covariates])
    y_ps = sample_data['D']
    
    try:
        ps_model = sm.Logit(y_ps, X_ps).fit(disp=0)
        ps = ps_model.predict(X_ps)
    except Exception:
        return np.nan
    
    ps = np.clip(ps, 0.01, 0.99)
    
    # 2. 在控制组上估计结果模型
    control_data = sample_data[sample_data['D'] == 0]
    X_outcome = sm.add_constant(control_data[covariates])
    y_outcome = control_data[y_col]
    
    try:
        outcome_model = sm.OLS(y_outcome, X_outcome).fit()
    except Exception:
        return np.nan
    
    # 3. 预测所有单位的反事实结果
    X_all = sm.add_constant(sample_data[covariates])
    m0 = outcome_model.predict(X_all)
    
    # 4. IPWRA 估计
    D = sample_data['D'].values
    Y = sample_data[y_col].values
    
    # 处理组部分
    treated_part = D * (Y - m0)
    
    # 控制组部分 (加权)
    weights = ps / (1 - ps)
    control_part = (1 - D) * weights * (Y - m0)
    
    # 归一化
    n_treated = D.sum()
    weight_sum = ((1 - D) * weights).sum()
    
    att = treated_part.sum() / n_treated + m0[D == 1].mean() - (
        control_part.sum() / weight_sum + m0[D == 0].mean() * (weight_sum / weight_sum)
    )
    
    # 简化版 IPWRA
    att = (D * Y).sum() / n_treated - (
        (1 - D) * weights * Y
    ).sum() / weight_sum
    
    return att


def main():
    print("=" * 90)
    print("诊断：IPWRA 估计器对结果的影响")
    print("=" * 90)
    
    df = load_data()
    
    # 协变量
    covariates = ['share_pop_poverty_78_above', 'share_pop_ind_manuf', 'share_school_some_hs']
    
    # Long Difference 变换
    print("\n[1] 计算 Long Difference 变换...")
    df_ld = transform_long_difference(df, 'log_retail_emp', 'fips', 'year', 'g')
    
    # 计算 cohort 大小
    cohort_sizes = df[df['g'] != np.inf].groupby('g')['fips'].nunique().to_dict()
    cohorts = sorted([g for g in df['g'].unique() if pd.notna(g) and g != np.inf])
    T_max = int(df['year'].max())
    
    # 计算各种估计器的 ATT
    print("[2] 计算各种估计器的 cohort-time ATT...")
    
    results = []
    for g in cohorts:
        g = int(g)
        for r in range(g, T_max + 1):
            col_name = f'yld_g{g}_r{r}'
            
            if col_name not in df_ld.columns:
                continue
            
            period_data = df_ld[df_ld['year'] == r].copy()
            
            # 简单差分
            att_simple = compute_simple_att(period_data, col_name, 'g', g, 'not_yet_treated')
            
            # RA
            att_ra = compute_ra_att(period_data, col_name, 'g', g, covariates, 'not_yet_treated')
            
            # IPW
            att_ipw = compute_ipw_att(period_data, col_name, 'g', g, covariates, 'not_yet_treated')
            
            # IPWRA
            att_ipwra = compute_ipwra_att(period_data, col_name, 'g', g, covariates, 'not_yet_treated')
            
            results.append({
                'cohort': g,
                'period': r,
                'event_time': r - g,
                'att_simple': att_simple,
                'att_ra': att_ra,
                'att_ipw': att_ipw,
                'att_ipwra': att_ipwra,
            })
    
    att_df = pd.DataFrame(results)
    
    # 计算 WATT
    print("[3] 计算 WATT...")
    
    def compute_watt_for_col(att_df, col):
        watt_list = []
        for event_time in sorted(att_df['event_time'].unique()):
            subset = att_df[att_df['event_time'] == event_time].copy()
            subset = subset[subset[col].notna()]
            
            if len(subset) == 0:
                continue
            
            subset['weight'] = subset['cohort'].map(cohort_sizes)
            total_weight = subset['weight'].sum()
            subset['norm_weight'] = subset['weight'] / total_weight
            
            watt = (subset[col] * subset['norm_weight']).sum()
            watt_list.append({
                'event_time': int(event_time),
                'watt': watt,
            })
        
        return pd.DataFrame(watt_list)
    
    watt_simple = compute_watt_for_col(att_df, 'att_simple')
    watt_ra = compute_watt_for_col(att_df, 'att_ra')
    watt_ipw = compute_watt_for_col(att_df, 'att_ipw')
    watt_ipwra = compute_watt_for_col(att_df, 'att_ipwra')
    
    # 比较结果
    print("\n" + "=" * 90)
    print("WATT 结果比较 (Long Difference + not_yet_treated)")
    print("=" * 90)
    
    print(f"\n{'r':>3} | {'Paper':>8} | {'Simple':>8} | {'RA':>8} | {'IPW':>8} | {'IPWRA':>8}")
    print("-" * 60)
    
    ratios = {'simple': [], 'ra': [], 'ipw': [], 'ipwra': []}
    
    for r in range(14):
        paper = PAPER_DEMEAN.get(r, np.nan)
        
        simple = watt_simple[watt_simple['event_time'] == r]['watt'].values
        ra = watt_ra[watt_ra['event_time'] == r]['watt'].values
        ipw = watt_ipw[watt_ipw['event_time'] == r]['watt'].values
        ipwra = watt_ipwra[watt_ipwra['event_time'] == r]['watt'].values
        
        simple_val = simple[0] if len(simple) > 0 else np.nan
        ra_val = ra[0] if len(ra) > 0 else np.nan
        ipw_val = ipw[0] if len(ipw) > 0 else np.nan
        ipwra_val = ipwra[0] if len(ipwra) > 0 else np.nan
        
        print(f"{r:>3} | {paper:>8.3f} | {simple_val:>8.4f} | {ra_val:>8.4f} | {ipw_val:>8.4f} | {ipwra_val:>8.4f}")
        
        if paper > 0:
            ratios['simple'].append(simple_val / paper if not np.isnan(simple_val) else np.nan)
            ratios['ra'].append(ra_val / paper if not np.isnan(ra_val) else np.nan)
            ratios['ipw'].append(ipw_val / paper if not np.isnan(ipw_val) else np.nan)
            ratios['ipwra'].append(ipwra_val / paper if not np.isnan(ipwra_val) else np.nan)
    
    print("-" * 60)
    print(f"平均比率 (vs Paper):")
    print(f"  Simple:  {np.nanmean(ratios['simple']):.2f}x")
    print(f"  RA:      {np.nanmean(ratios['ra']):.2f}x")
    print(f"  IPW:     {np.nanmean(ratios['ipw']):.2f}x")
    print(f"  IPWRA:   {np.nanmean(ratios['ipwra']):.2f}x")
    
    # 关键发现
    print("\n" + "=" * 90)
    print("关键发现")
    print("=" * 90)
    
    best_method = min(ratios.items(), key=lambda x: abs(np.nanmean(x[1]) - 1))
    print(f"\n最接近论文的方法: {best_method[0]} (比率 {np.nanmean(best_method[1]):.2f}x)")
    
    print("""
如果所有方法都比论文高，可能的原因：
1. 论文使用了不同的变换方法 (Full Demean 而非 Long Difference)
2. 论文使用了不同的控制组定义
3. 论文使用了不同的协变量处理方式
4. 论文可能有数据处理差异 (如样本选择)
""")


if __name__ == '__main__':
    main()

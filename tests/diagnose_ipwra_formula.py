"""
IPWRA 公式诊断

手动实现论文的 IPWRA 公式，与当前实现进行比较。

论文公式 4.8 (Staggered IPWRA):
τ̂_{gr}^{IPWRA} = (1/N_g) Σ_{D_ig=1} [Ŷ_{irg}] 
                 - (1/N_g) Σ_{D_ig=1} [m̂_0(X_i) + (p̂/(1-p̂)) × (A_{i,r+1}(Ŷ_{irg} - m̂_0(X_i))) / Σ_{A_{j,r+1}} (p̂/(1-p̂))]

简化形式:
τ̂ = Ȳ_1 - [m̂_0(X̄_1) + weighted_control_residual]

其中:
- Ȳ_1 = 处理组变换后结果的均值
- m̂_0(X) = 控制组结果模型的预测值
- p̂ = 倾向得分
- weighted_control_residual = IPW 加权的控制组残差
"""

import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression, LinearRegression

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

warnings.filterwarnings('ignore')


def load_data():
    """加载 Walmart 数据"""
    data_path = Path(__file__).parent.parent / 'data' / 'walmart.csv'
    return pd.read_csv(data_path)


def manual_ipwra(data, y, d, controls, trim_threshold=0.01):
    """
    手动实现 IPWRA 估计器，严格按照论文公式
    
    论文公式 4.7 (Common Timing IPWRA):
    τ̂^{IPWRA} = (1/N_1) Σ_{D=1} Y_i 
                - (1/N_1) Σ_{D=1} [m̂_0(X_i) + (p̂/(1-p̂)) × ((1-D)(Y - m̂_0(X))) / Σ_{D=0} (p̂/(1-p̂))]
    """
    # 准备数据
    all_vars = [y, d] + controls
    data_clean = data[all_vars].dropna().copy()
    
    D = data_clean[d].values
    Y = data_clean[y].values
    X = data_clean[controls].values
    
    n_treated = int(D.sum())
    n_control = int((1 - D).sum())
    
    treat_mask = D == 1
    control_mask = D == 0
    
    # 1. 估计倾向得分 P(D=1|X)
    ps_model = LogisticRegression(solver='lbfgs', max_iter=1000)
    ps_model.fit(X, D)
    pscores = ps_model.predict_proba(X)[:, 1]
    
    # 裁剪倾向得分
    pscores = np.clip(pscores, trim_threshold, 1 - trim_threshold)
    
    # 2. 估计结果模型 E[Y|X, D=0]
    outcome_model = LinearRegression()
    outcome_model.fit(X[control_mask], Y[control_mask])
    m0_hat = outcome_model.predict(X)
    
    # 3. 计算 IPWRA-ATT
    # 处理组部分: (1/N_1) Σ_{D=1} Y_i
    treat_mean = Y[treat_mask].mean()
    
    # 控制组加权残差部分
    weights = pscores / (1 - pscores)
    weights_control = weights[control_mask]
    residuals_control = Y[control_mask] - m0_hat[control_mask]
    
    weights_sum = weights_control.sum()
    weighted_residual = (weights_control * residuals_control).sum() / weights_sum
    
    # 处理组的 m̂_0(X) 均值
    m0_treated_mean = m0_hat[treat_mask].mean()
    
    # IPWRA-ATT = Ȳ_1 - [m̂_0(X̄_1) + weighted_control_residual]
    # 但论文公式是: (1/N_1) Σ_{D=1} [Y - m̂_0(X)] - weighted_control_residual
    # 等价于: Ȳ_1 - m̂_0(X̄_1) - weighted_control_residual
    
    att_ipwra = treat_mean - m0_treated_mean - weighted_residual
    
    return {
        'att': att_ipwra,
        'treat_mean': treat_mean,
        'm0_treated_mean': m0_treated_mean,
        'weighted_residual': weighted_residual,
        'pscores_mean': pscores.mean(),
        'pscores_std': pscores.std(),
        'n_treated': n_treated,
        'n_control': n_control,
    }


def main():
    """运行诊断"""
    print("=" * 80)
    print("IPWRA 公式诊断")
    print("=" * 80)
    
    from lwdid.staggered.transformations import transform_staggered_demean
    from lwdid.staggered.estimators import estimate_ipwra
    
    df = load_data()
    
    controls = [
        'share_pop_poverty_78_above',
        'share_pop_ind_manuf',
        'share_school_some_hs',
    ]
    
    # 变换数据
    df_transformed = transform_staggered_demean(
        df, y='log_retail_emp', ivar='fips', tvar='year', gvar='g'
    )
    
    # 选择一个具体的 cohort 和 period 进行详细分析
    cohort = 1995
    period = 1995  # event_time = 0
    
    print(f"\n分析 cohort={cohort}, period={period} (event_time=0)")
    print("-" * 60)
    
    col_name = f'ydot_g{cohort}_r{period}'
    
    # 获取该 period 的数据
    period_data = df_transformed[df_transformed['year'] == period].copy()
    
    # 定义处理组和控制组 (not_yet_treated)
    treated_mask = period_data['g'] == cohort
    control_mask = (period_data['g'] > period) | (period_data['g'] == np.inf)
    
    period_data['D'] = 0
    period_data.loc[treated_mask, 'D'] = 1
    
    # 只保留处理组和控制组
    analysis_data = period_data[treated_mask | control_mask].copy()
    
    print(f"\n样本量:")
    print(f"  处理组: {analysis_data['D'].sum()}")
    print(f"  控制组: {(1 - analysis_data['D']).sum()}")
    
    # 1. 手动 IPWRA
    manual_result = manual_ipwra(
        analysis_data, col_name, 'D', controls, trim_threshold=0.01
    )
    
    print(f"\n1. 手动 IPWRA (论文公式):")
    print(f"   ATT: {manual_result['att']:.6f}")
    print(f"   处理组均值: {manual_result['treat_mean']:.6f}")
    print(f"   m̂_0(X̄_1): {manual_result['m0_treated_mean']:.6f}")
    print(f"   加权残差: {manual_result['weighted_residual']:.6f}")
    print(f"   倾向得分均值: {manual_result['pscores_mean']:.4f}")
    
    # 2. lwdid IPWRA
    try:
        lwdid_result = estimate_ipwra(
            data=analysis_data,
            y=col_name,
            d='D',
            controls=controls,
            trim_threshold=0.01,
            se_method='analytical',
            alpha=0.05,
        )
        
        print(f"\n2. lwdid IPWRA:")
        print(f"   ATT: {lwdid_result.att:.6f}")
        print(f"   SE: {lwdid_result.se:.6f}")
        
    except Exception as e:
        print(f"\n2. lwdid IPWRA 失败: {e}")
    
    # 3. 简单差分
    treated_vals = analysis_data[analysis_data['D'] == 1][col_name].dropna()
    control_vals = analysis_data[analysis_data['D'] == 0][col_name].dropna()
    att_simple = treated_vals.mean() - control_vals.mean()
    
    print(f"\n3. 简单差分:")
    print(f"   ATT: {att_simple:.6f}")
    
    # 4. 比较多个 cohort
    print("\n" + "=" * 80)
    print("多个 Cohort 的 ATT 比较 (event_time=0)")
    print("=" * 80)
    
    cohorts = sorted([g for g in df['g'].unique() if pd.notna(g) and g != np.inf])
    
    results = []
    for g in cohorts[:10]:
        g = int(g)
        r = g  # event_time = 0
        col = f'ydot_g{g}_r{r}'
        
        if col not in df_transformed.columns:
            continue
        
        period_data = df_transformed[df_transformed['year'] == r].copy()
        
        treated_mask = period_data['g'] == g
        control_mask = (period_data['g'] > r) | (period_data['g'] == np.inf)
        
        period_data['D'] = 0
        period_data.loc[treated_mask, 'D'] = 1
        
        analysis_data = period_data[treated_mask | control_mask].copy()
        
        # 简单差分
        treated_vals = analysis_data[analysis_data['D'] == 1][col].dropna()
        control_vals = analysis_data[analysis_data['D'] == 0][col].dropna()
        
        if len(treated_vals) == 0 or len(control_vals) == 0:
            continue
        
        att_simple = treated_vals.mean() - control_vals.mean()
        
        # 手动 IPWRA
        try:
            manual_result = manual_ipwra(analysis_data, col, 'D', controls)
            att_manual = manual_result['att']
        except Exception:
            att_manual = np.nan
        
        # lwdid IPWRA
        try:
            lwdid_result = estimate_ipwra(
                data=analysis_data,
                y=col,
                d='D',
                controls=controls,
                trim_threshold=0.01,
            )
            att_lwdid = lwdid_result.att
        except Exception:
            att_lwdid = np.nan
        
        results.append({
            'cohort': g,
            'att_simple': att_simple,
            'att_manual': att_manual,
            'att_lwdid': att_lwdid,
        })
    
    results_df = pd.DataFrame(results)
    print(f"\n{'cohort':>8} | {'Simple':>10} | {'Manual':>10} | {'lwdid':>10}")
    print("-" * 50)
    for _, row in results_df.iterrows():
        print(f"{row['cohort']:>8} | {row['att_simple']:>10.6f} | {row['att_manual']:>10.6f} | {row['att_lwdid']:>10.6f}")
    
    # 检查手动和 lwdid 的差异
    results_df['diff_manual_lwdid'] = results_df['att_manual'] - results_df['att_lwdid']
    print(f"\n手动 vs lwdid 差异:")
    print(f"  最大绝对差异: {results_df['diff_manual_lwdid'].abs().max():.10f}")
    print(f"  平均绝对差异: {results_df['diff_manual_lwdid'].abs().mean():.10f}")


if __name__ == '__main__':
    main()

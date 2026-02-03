"""
IPWRA vs 简单差分诊断

比较 IPWRA 估计器和简单差分的结果，找出偏差来源。
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


def main():
    """运行诊断"""
    print("=" * 80)
    print("IPWRA vs 简单差分诊断")
    print("=" * 80)
    
    from lwdid.staggered.transformations import transform_staggered_demean
    from lwdid.staggered.estimators import estimate_ipwra
    
    df = load_data()
    
    controls = [
        'share_pop_poverty_78_above',
        'share_pop_ind_manuf',
        'share_school_some_hs',
    ]
    
    # 选择一个具体的 cohort 和 period 进行详细分析
    cohort = 1995
    period = 1995  # event_time = 0
    
    print(f"\n分析 cohort={cohort}, period={period} (event_time=0)")
    print("-" * 60)
    
    # 变换数据
    df_transformed = transform_staggered_demean(
        df, y='log_retail_emp', ivar='fips', tvar='year', gvar='g'
    )
    
    col_name = f'ydot_g{cohort}_r{period}'
    
    # 获取该 period 的数据
    period_data = df_transformed[df_transformed['year'] == period].copy()
    
    # 定义处理组和控制组
    # 处理组: g == cohort
    treated_mask = period_data['g'] == cohort
    # 控制组 (not_yet_treated): g > period 或 g == inf
    control_mask = (period_data['g'] > period) | (period_data['g'] == np.inf)
    
    # 创建处理指标
    period_data['D'] = 0
    period_data.loc[treated_mask, 'D'] = 1
    
    # 只保留处理组和控制组
    analysis_data = period_data[treated_mask | control_mask].copy()
    
    print(f"\n样本量:")
    print(f"  处理组: {analysis_data['D'].sum()}")
    print(f"  控制组: {(1 - analysis_data['D']).sum()}")
    
    # 1. 简单差分 (无协变量)
    treated_vals = analysis_data[analysis_data['D'] == 1][col_name].dropna()
    control_vals = analysis_data[analysis_data['D'] == 0][col_name].dropna()
    
    att_simple = treated_vals.mean() - control_vals.mean()
    
    print(f"\n1. 简单差分 (无协变量):")
    print(f"   处理组均值: {treated_vals.mean():.6f}")
    print(f"   控制组均值: {control_vals.mean():.6f}")
    print(f"   ATT: {att_simple:.6f}")
    
    # 2. IPWRA 估计
    try:
        ipwra_result = estimate_ipwra(
            data=analysis_data,
            y=col_name,
            d='D',
            controls=controls,
            trim_threshold=0.01,
            se_method='analytical',
            alpha=0.05,
        )
        
        print(f"\n2. IPWRA 估计:")
        print(f"   ATT: {ipwra_result.att:.6f}")
        print(f"   SE: {ipwra_result.se:.6f}")
        print(f"   处理组样本: {ipwra_result.n_treated}")
        print(f"   控制组样本: {ipwra_result.n_control}")
        
        # 检查倾向得分分布
        ps = ipwra_result.propensity_scores
        print(f"\n   倾向得分分布:")
        print(f"     均值: {ps.mean():.4f}")
        print(f"     标准差: {ps.std():.4f}")
        print(f"     最小值: {ps.min():.4f}")
        print(f"     最大值: {ps.max():.4f}")
        
    except Exception as e:
        print(f"\n2. IPWRA 估计失败: {e}")
    
    # 3. 检查协变量分布
    print(f"\n3. 协变量分布:")
    for ctrl in controls:
        treated_ctrl = analysis_data[analysis_data['D'] == 1][ctrl]
        control_ctrl = analysis_data[analysis_data['D'] == 0][ctrl]
        
        print(f"\n   {ctrl}:")
        print(f"     处理组均值: {treated_ctrl.mean():.4f}")
        print(f"     控制组均值: {control_ctrl.mean():.4f}")
        print(f"     差异: {treated_ctrl.mean() - control_ctrl.mean():.4f}")
    
    # 4. 比较多个 cohort 的结果
    print("\n" + "=" * 80)
    print("多个 Cohort 的 ATT 比较 (event_time=0)")
    print("=" * 80)
    
    cohorts = sorted([g for g in df['g'].unique() if pd.notna(g) and g != np.inf])
    
    results = []
    for g in cohorts[:10]:  # 只检查前 10 个 cohort
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
        
        # IPWRA
        try:
            ipwra_result = estimate_ipwra(
                data=analysis_data,
                y=col,
                d='D',
                controls=controls,
                trim_threshold=0.01,
                se_method='analytical',
                alpha=0.05,
            )
            att_ipwra = ipwra_result.att
        except Exception:
            att_ipwra = np.nan
        
        results.append({
            'cohort': g,
            'att_simple': att_simple,
            'att_ipwra': att_ipwra,
            'ratio': att_ipwra / att_simple if att_simple != 0 else np.nan,
        })
    
    results_df = pd.DataFrame(results)
    print(f"\n{'cohort':>8} | {'ATT_simple':>12} | {'ATT_IPWRA':>12} | {'Ratio':>8}")
    print("-" * 50)
    for _, row in results_df.iterrows():
        print(f"{row['cohort']:>8} | {row['att_simple']:>12.6f} | {row['att_ipwra']:>12.6f} | {row['ratio']:>8.2f}")
    
    print(f"\n平均比率 (IPWRA/Simple): {results_df['ratio'].mean():.3f}")


if __name__ == '__main__':
    main()

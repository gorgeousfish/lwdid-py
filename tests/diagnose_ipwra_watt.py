"""
IPWRA WATT 诊断脚本

使用 IPWRA 估计器（而非简单差分）计算 WATT，与论文 Table A4 进行定量比较。

论文参考值 (Rolling IPWRA without detrending):
- r=0: 0.018, r=1: 0.045, r=2: 0.038, r=3: 0.032, r=4: 0.031
- r=5: 0.036, r=6: 0.040, r=7: 0.054, r=8: 0.062, r=9: 0.063
- r=10: 0.081, r=11: 0.083, r=12: 0.080, r=13: 0.107
"""

import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

warnings.filterwarnings('ignore')


# 论文参考值
PAPER_DEMEAN = {
    0: 0.018, 1: 0.045, 2: 0.038, 3: 0.032, 4: 0.031,
    5: 0.036, 6: 0.040, 7: 0.054, 8: 0.062, 9: 0.063,
    10: 0.081, 11: 0.083, 12: 0.080, 13: 0.107,
}

PAPER_DETREND = {
    0: 0.007, 1: 0.032, 2: 0.025, 3: 0.021, 4: 0.018,
    5: 0.017, 6: 0.019, 7: 0.036, 8: 0.041, 9: 0.041,
    10: 0.037, 11: 0.018, 12: 0.017, 13: 0.047,
}


def load_data():
    """加载 Walmart 数据"""
    data_path = Path(__file__).parent.parent / 'data' / 'walmart.csv'
    return pd.read_csv(data_path)


def main():
    """运行 IPWRA WATT 诊断"""
    print("=" * 80)
    print("IPWRA WATT 诊断: 使用 lwdid 包的 IPWRA 估计器")
    print("=" * 80)
    
    from lwdid import lwdid
    
    df = load_data()
    
    controls = [
        'share_pop_poverty_78_above',
        'share_pop_ind_manuf',
        'share_school_some_hs',
    ]
    
    # 测试配置
    configs = [
        ('demean', 'never_treated', 'Demean + Never Treated'),
        ('demean', 'not_yet_treated', 'Demean + Not Yet Treated'),
        ('detrend', 'never_treated', 'Detrend + Never Treated'),
        ('detrend', 'not_yet_treated', 'Detrend + Not Yet Treated'),
    ]
    
    results = {}
    
    for rolling, control_group, label in configs:
        print(f"\n计算 {label}...")
        
        try:
            result = lwdid(
                data=df,
                y='log_retail_emp',
                ivar='fips',
                tvar='year',
                gvar='g',
                rolling=rolling,
                estimator='ipwra',
                controls=controls,
                control_group=control_group,
                aggregate='none',
                alpha=0.05,
            )
            
            # 计算 WATT
            watt = compute_watt(result, df)
            results[label] = watt
            
        except Exception as e:
            print(f"  错误: {e}")
            results[label] = pd.DataFrame()
    
    # 比较结果
    print("\n" + "=" * 80)
    print("WATT 结果比较 (IPWRA 估计器)")
    print("=" * 80)
    
    # 打印 Demean 结果
    print("\n--- Demean 方法 ---")
    print(f"{'r':>3} | {'NT':>10} | {'NYT':>10} | {'Paper':>8} | {'NT/Paper':>10} | {'NYT/Paper':>10}")
    print("-" * 70)
    
    for r in range(14):
        paper_val = PAPER_DEMEAN.get(r, np.nan)
        
        nt_val = get_watt_value(results.get('Demean + Never Treated'), r)
        nyt_val = get_watt_value(results.get('Demean + Not Yet Treated'), r)
        
        nt_ratio = f"{nt_val/paper_val:.2f}x" if not np.isnan(nt_val) and paper_val > 0 else "N/A"
        nyt_ratio = f"{nyt_val/paper_val:.2f}x" if not np.isnan(nyt_val) and paper_val > 0 else "N/A"
        
        nt_str = f"{nt_val:.4f}" if not np.isnan(nt_val) else "N/A"
        nyt_str = f"{nyt_val:.4f}" if not np.isnan(nyt_val) else "N/A"
        
        print(f"{r:>3} | {nt_str:>10} | {nyt_str:>10} | {paper_val:>8.3f} | {nt_ratio:>10} | {nyt_ratio:>10}")
    
    # 打印 Detrend 结果
    print("\n--- Detrend 方法 ---")
    print(f"{'r':>3} | {'NT':>10} | {'NYT':>10} | {'Paper':>8} | {'NT/Paper':>10} | {'NYT/Paper':>10}")
    print("-" * 70)
    
    for r in range(14):
        paper_val = PAPER_DETREND.get(r, np.nan)
        
        nt_val = get_watt_value(results.get('Detrend + Never Treated'), r)
        nyt_val = get_watt_value(results.get('Detrend + Not Yet Treated'), r)
        
        nt_ratio = f"{nt_val/paper_val:.2f}x" if not np.isnan(nt_val) and paper_val > 0 else "N/A"
        nyt_ratio = f"{nyt_val/paper_val:.2f}x" if not np.isnan(nyt_val) and paper_val > 0 else "N/A"
        
        nt_str = f"{nt_val:.4f}" if not np.isnan(nt_val) else "N/A"
        nyt_str = f"{nyt_val:.4f}" if not np.isnan(nyt_val) else "N/A"
        
        print(f"{r:>3} | {nt_str:>10} | {nyt_str:>10} | {paper_val:>8.3f} | {nt_ratio:>10} | {nyt_ratio:>10}")
    
    # 计算平均比率
    print("\n" + "=" * 80)
    print("平均比率统计")
    print("=" * 80)
    
    for label in results.keys():
        watt_df = results[label]
        if watt_df.empty:
            continue
        
        paper_ref = PAPER_DEMEAN if 'Demean' in label else PAPER_DETREND
        
        ratios = []
        for r in range(14):
            val = get_watt_value(watt_df, r)
            paper_val = paper_ref.get(r, np.nan)
            if not np.isnan(val) and paper_val > 0:
                ratios.append(val / paper_val)
        
        if ratios:
            avg_ratio = np.mean(ratios)
            std_ratio = np.std(ratios)
            print(f"{label}: 平均比率 = {avg_ratio:.3f}x (std = {std_ratio:.3f})")


def compute_watt(results, df):
    """计算 WATT"""
    att_ct = results.att_by_cohort_time.copy()
    
    if att_ct is None or len(att_ct) == 0:
        return pd.DataFrame()
    
    # 获取 cohort 大小用于加权
    cohort_sizes = df[df['g'] != np.inf].groupby('g')['fips'].nunique().to_dict()
    
    att_ct['weight'] = att_ct['cohort'].map(cohort_sizes)
    att_ct['weight'] = att_ct['weight'].fillna(0)
    
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


def get_watt_value(watt_df, event_time):
    """获取特定 event_time 的 WATT 值"""
    if watt_df is None or watt_df.empty:
        return np.nan
    
    row = watt_df[watt_df['event_time'] == event_time]
    if len(row) == 0:
        return np.nan
    
    return row['watt'].values[0]


if __name__ == '__main__':
    main()

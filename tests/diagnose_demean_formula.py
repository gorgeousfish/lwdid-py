"""
诊断：验证 Demean 公式实现

论文公式 2.2 (Staggered Demeaning):
    Ŷ_{irg} = Y_{ir} - (1/(g-1)) × Σ_{s=1}^{g-1} Y_{is}

关键问题：
1. 预处理期的定义是否正确？
2. 均值计算是否正确？
3. 变换是否应用于所有单位？
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


def verify_demean_formula(df, y, ivar, tvar, gvar):
    """验证 demean 公式的实现"""
    print("\n" + "=" * 80)
    print("验证 Demean 公式实现")
    print("=" * 80)
    
    # 选择一个示例单位和 cohort
    cohorts = sorted([g for g in df[gvar].unique() if pd.notna(g) and g != np.inf])
    
    # 选择第一个 cohort 作为示例
    g = int(cohorts[0])
    
    # 选择一个处理单位
    treated_units = df[df[gvar] == g][ivar].unique()
    unit_id = treated_units[0]
    
    print(f"\n示例: cohort g={g}, unit={unit_id}")
    
    # 获取该单位的数据
    unit_data = df[df[ivar] == unit_id].sort_values(tvar)
    
    print(f"\n该单位的时间序列:")
    print(unit_data[[tvar, y]].to_string(index=False))
    
    # 计算预处理期均值
    T_min = int(df[tvar].min())
    pre_periods = list(range(T_min, g))
    
    print(f"\n预处理期: {pre_periods}")
    print(f"预处理期数量: {len(pre_periods)}")
    
    pre_data = unit_data[unit_data[tvar] < g]
    pre_mean = pre_data[y].mean()
    
    print(f"\n预处理期 Y 值: {pre_data[y].values}")
    print(f"预处理期均值: {pre_mean:.6f}")
    
    # 计算变换后的值
    print(f"\n变换后的值 (Y_ir - pre_mean):")
    for r in range(g, int(df[tvar].max()) + 1):
        y_r = unit_data[unit_data[tvar] == r][y].values
        if len(y_r) > 0:
            y_transformed = y_r[0] - pre_mean
            print(f"  r={r}: Y={y_r[0]:.6f}, Ŷ={y_transformed:.6f}")
    
    return pre_mean


def check_data_structure(df, y, ivar, tvar, gvar):
    """检查数据结构"""
    print("\n" + "=" * 80)
    print("数据结构检查")
    print("=" * 80)
    
    T_min = int(df[tvar].min())
    T_max = int(df[tvar].max())
    
    print(f"\n时间范围: {T_min} - {T_max}")
    print(f"时间周期数: {T_max - T_min + 1}")
    
    # Cohort 分布
    cohorts = sorted([g for g in df[gvar].unique() if pd.notna(g) and g != np.inf])
    print(f"\n处理 Cohorts: {cohorts}")
    print(f"Cohort 数量: {len(cohorts)}")
    
    # 每个 cohort 的预处理期数量
    print(f"\n各 Cohort 的预处理期数量:")
    for g in cohorts:
        n_pre = g - T_min
        print(f"  g={g}: {n_pre} 个预处理期 (t={T_min} to {g-1})")
    
    # Never-treated 单位
    nt_mask = df[gvar] == np.inf
    n_nt = df[nt_mask][ivar].nunique()
    print(f"\nNever-treated 单位数: {n_nt}")
    
    # 各 cohort 的处理单位数
    print(f"\n各 Cohort 的处理单位数:")
    for g in cohorts:
        n_treated = df[df[gvar] == g][ivar].nunique()
        print(f"  g={g}: {n_treated} 个单位")


def check_control_group_definition(df, gvar, tvar):
    """检查控制组定义"""
    print("\n" + "=" * 80)
    print("控制组定义检查")
    print("=" * 80)
    
    cohorts = sorted([g for g in df[gvar].unique() if pd.notna(g) and g != np.inf])
    T_max = int(df[tvar].max())
    
    print("\n论文公式 2.3 控制组定义:")
    print("  A_{r+1} = D_{r+1} + D_{r+2} + ... + D_T + D_∞")
    print("  即: 在时期 r 尚未被处理的单位 (g > r) 或从未被处理的单位 (g = ∞)")
    
    # 检查各 event time 的控制组大小
    print(f"\n各 Event Time 的控制组大小 (not_yet_treated):")
    
    for r in range(14):
        # 对于 event_time = r，calendar time 取决于 cohort
        # 我们检查最早的 cohort (g=1985) 在 r=0 时的控制组
        g = cohorts[0]
        calendar_r = g + r
        
        if calendar_r > T_max:
            continue
        
        # 控制组: g > calendar_r 或 g = inf
        period_data = df[df[tvar] == calendar_r]
        control_mask = (period_data[gvar] > calendar_r) | (period_data[gvar] == np.inf)
        n_control = period_data[control_mask][df.columns[0]].nunique() if df.columns[0] != gvar else len(period_data[control_mask])
        
        print(f"  r={r} (calendar={calendar_r}): {control_mask.sum()} 个控制组观测")


def main():
    df = load_data()
    
    # 1. 检查数据结构
    check_data_structure(df, 'log_retail_emp', 'fips', 'year', 'g')
    
    # 2. 验证 demean 公式
    verify_demean_formula(df, 'log_retail_emp', 'fips', 'year', 'g')
    
    # 3. 检查控制组定义
    check_control_group_definition(df, 'g', 'year')
    
    # 4. 关键问题总结
    print("\n" + "=" * 80)
    print("关键问题总结")
    print("=" * 80)
    
    print("""
根据诊断结果，可能的差异来源：

1. **变换方法差异**:
   - 论文 Full Demean: Ŷ = Y_r - mean(Y_{1:g-1})
   - CS Long Difference: Ŷ = Y_r - Y_{g-1}
   - 两者都比论文结果高 ~1.6x

2. **可能的解释**:
   a) 论文可能使用了不同的样本选择
   b) 论文可能使用了不同的权重计算方式
   c) 论文可能有数据预处理步骤未公开
   d) 论文结果可能有误差

3. **下一步**:
   - 检查论文的 Stata 复现代码（如果有）
   - 联系论文作者确认方法细节
   - 接受 ~1.6x 的系统性差异，记录为已知问题
""")


if __name__ == '__main__':
    main()

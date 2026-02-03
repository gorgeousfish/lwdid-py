"""
测试完全枚举模式 (Full Enumeration)

当 cluster 数量较少时（G ≤ 10-12），可以枚举所有可能的 Rademacher 权重组合。
这样可以消除随机性，实现确定性的结果。

对于 G 个 cluster，共有 2^G 种可能的权重组合。
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from wildboottest.wildboottest import wildboottest
    HAS_WILDBOOTTEST = True
except ImportError:
    HAS_WILDBOOTTEST = False
    print("警告: wildboottest 包未安装")

# Stata 结果
STATA_RESULTS = {
    't_stat': 0.7287,
    'p_value': 0.4915,
    'ci_lower': -2.079,
    'ci_upper': 4.174,
}


def load_test_data():
    return pd.read_csv(Path(__file__).parent / "stata_wild_test.csv")


def test_full_enumeration():
    """测试完全枚举模式"""
    if not HAS_WILDBOOTTEST:
        print("跳过测试：wildboottest 未安装")
        return
    
    df = load_test_data()
    model = smf.ols(formula='y ~ d', data=df)
    
    n_clusters = df['cluster'].nunique()
    n_combinations = 2 ** n_clusters
    
    print("=" * 70)
    print("完全枚举模式测试")
    print("=" * 70)
    print(f"Cluster 数量: {n_clusters}")
    print(f"可能的权重组合数: {n_combinations}")
    print()
    
    # 使用完全枚举
    # 当 B >= 2^G 时，wildboottest 自动使用完全枚举
    result = wildboottest(
        model, 
        param="d", 
        cluster=df['cluster'], 
        B=n_combinations * 2,  # 设置为 > 2^G 触发完全枚举
        seed=42,
        bootstrap_type='11'
    )
    
    p_value = result['p-value'].values[0]
    t_stat = result['statistic'].values[0]
    
    print(f"wildboottest (完全枚举) 结果:")
    print(f"  t 统计量: {t_stat:.4f}")
    print(f"  p 值: {p_value:.4f}")
    print()
    print(f"Stata boottest 结果:")
    print(f"  t 统计量: {STATA_RESULTS['t_stat']:.4f}")
    print(f"  p 值: {STATA_RESULTS['p_value']:.4f}")
    print()
    print(f"差异:")
    print(f"  t 统计量差异: {abs(t_stat - STATA_RESULTS['t_stat']):.6f}")
    print(f"  p 值差异: {abs(p_value - STATA_RESULTS['p_value']):.6f}")
    
    return p_value


def test_full_enumeration_multiple_runs():
    """验证完全枚举的确定性"""
    if not HAS_WILDBOOTTEST:
        print("跳过测试：wildboottest 未安装")
        return
    
    df = load_test_data()
    model = smf.ols(formula='y ~ d', data=df)
    
    print()
    print("=" * 70)
    print("验证完全枚举的确定性")
    print("=" * 70)
    
    results = []
    for seed in [1, 42, 123, 456, 789]:
        result = wildboottest(
            model, 
            param="d", 
            cluster=df['cluster'], 
            B=2048,  # > 2^10 = 1024
            seed=seed,
            bootstrap_type='11'
        )
        p = result['p-value'].values[0]
        results.append(p)
        print(f"  seed={seed}: p = {p:.6f}")
    
    print()
    if len(set(results)) == 1:
        print("✓ 所有运行产生相同的 p 值（确定性）")
    else:
        print(f"✗ p 值有差异: {set(results)}")
    
    return results


def compare_with_stata_full_enum():
    """
    比较 Python 完全枚举与 Stata 完全枚举
    
    Stata 命令：
    boottest d, reps(1024) weight(rademacher) 
    
    注意：Stata 可能不会自动使用完全枚举，需要检查
    """
    print()
    print("=" * 70)
    print("与 Stata 完全枚举比较")
    print("=" * 70)
    print()
    print("要在 Stata 中使用完全枚举，需要运行：")
    print("  boottest d, reps(1024) weight(rademacher)")
    print()
    print("或者使用 boottest 的 full 选项（如果支持）")
    print()
    print("注意：Stata boottest 的默认行为可能不同")


def main():
    test_full_enumeration()
    test_full_enumeration_multiple_runs()
    compare_with_stata_full_enum()
    
    print()
    print("=" * 70)
    print("结论")
    print("=" * 70)
    print("""
1. 完全枚举模式消除了随机性，产生确定性结果
2. 但 Python 和 Stata 的完全枚举结果仍可能不同
3. 这是因为：
   - 权重组合的枚举顺序可能不同
   - p 值计算方式可能有细微差异
   - 数值精度差异

4. 要实现 100% 等价，需要：
   - 确保 Stata 也使用完全枚举
   - 验证两者的枚举顺序和计算方式完全一致
""")


if __name__ == "__main__":
    main()

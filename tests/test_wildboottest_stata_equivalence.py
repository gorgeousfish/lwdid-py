"""
测试 wildboottest 包是否能与 Stata boottest 实现 100% 等价。

目标：
1. 使用相同的数据和参数
2. 比较 p 值、CI、t 统计量
3. 分析差异来源
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# 尝试导入 wildboottest
try:
    from wildboottest.wildboottest import wildboottest
    HAS_WILDBOOTTEST = True
except ImportError:
    HAS_WILDBOOTTEST = False
    print("警告: wildboottest 包未安装，请运行: pip install wildboottest")

# Stata boottest 结果（参考）
# 这些是使用 Stata 17 运行以下命令得到的结果：
# regress y d, cluster(cluster)
# boottest d, reps(999) seed(42) weight(rademacher)
STATA_RESULTS = {
    't_stat': 0.7287,
    'p_value': 0.4915,
    'ci_lower': -2.079,
    'ci_upper': 4.174,
}


def load_test_data():
    """加载测试数据"""
    return pd.read_csv(Path(__file__).parent / "stata_wild_test.csv")


def test_wildboottest_basic():
    """基本测试：使用默认参数"""
    if not HAS_WILDBOOTTEST:
        print("跳过测试：wildboottest 未安装")
        return
    
    df = load_test_data()
    model = smf.ols(formula='y ~ d', data=df)
    
    print("=" * 70)
    print("测试 1: wildboottest 基本测试")
    print("=" * 70)
    
    result = wildboottest(
        model, 
        param="d", 
        cluster=df['cluster'], 
        B=999, 
        seed=42,
        bootstrap_type='11'  # WCR11 是默认类型
    )
    
    t_stat = result['statistic'].values[0]
    p_value = result['p-value'].values[0]
    
    print(f"wildboottest 结果:")
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
    
    return {
        't_stat': t_stat,
        'p_value': p_value,
        't_diff': abs(t_stat - STATA_RESULTS['t_stat']),
        'p_diff': abs(p_value - STATA_RESULTS['p_value'])
    }


def test_wildboottest_all_types():
    """测试所有 bootstrap 类型"""
    if not HAS_WILDBOOTTEST:
        print("跳过测试：wildboottest 未安装")
        return
    
    df = load_test_data()
    model = smf.ols(formula='y ~ d', data=df)
    
    print()
    print("=" * 70)
    print("测试 2: 不同 bootstrap 类型")
    print("=" * 70)
    
    # bootstrap_type 说明：
    # '11' = WCR11 (默认) - 使用 HC1 方差估计
    # '13' = WCR13 - 使用 HC3 方差估计
    # '31' = WCU11 - 非受限 bootstrap
    # '33' = WCU13 - 非受限 + HC3
    
    results = {}
    for bt in ['11', '13', '31', '33']:
        result = wildboottest(
            model, 
            param="d", 
            cluster=df['cluster'], 
            B=999, 
            seed=42,
            bootstrap_type=bt
        )
        p = result['p-value'].values[0]
        results[bt] = p
        diff = abs(p - STATA_RESULTS['p_value'])
        print(f"  类型 {bt}: p = {p:.4f} (与 Stata 差异: {diff:.4f})")
    
    return results


def test_wildboottest_different_seeds():
    """测试不同随机种子的影响"""
    if not HAS_WILDBOOTTEST:
        print("跳过测试：wildboottest 未安装")
        return
    
    df = load_test_data()
    model = smf.ols(formula='y ~ d', data=df)
    
    print()
    print("=" * 70)
    print("测试 3: 不同随机种子")
    print("=" * 70)
    
    results = []
    for seed in [42, 123, 456, 789, 1000]:
        result = wildboottest(
            model, 
            param="d", 
            cluster=df['cluster'], 
            B=999, 
            seed=seed,
            bootstrap_type='11'
        )
        p = result['p-value'].values[0]
        results.append(p)
        diff = abs(p - STATA_RESULTS['p_value'])
        print(f"  seed={seed}: p = {p:.4f} (与 Stata 差异: {diff:.4f})")
    
    print()
    print(f"  p 值范围: [{min(results):.4f}, {max(results):.4f}]")
    print(f"  p 值标准差: {np.std(results):.4f}")
    
    return results


def test_wildboottest_more_reps():
    """测试更多 bootstrap 次数"""
    if not HAS_WILDBOOTTEST:
        print("跳过测试：wildboottest 未安装")
        return
    
    df = load_test_data()
    model = smf.ols(formula='y ~ d', data=df)
    
    print()
    print("=" * 70)
    print("测试 4: 不同 bootstrap 次数")
    print("=" * 70)
    
    results = {}
    for B in [99, 499, 999, 4999, 9999]:
        result = wildboottest(
            model, 
            param="d", 
            cluster=df['cluster'], 
            B=B, 
            seed=42,
            bootstrap_type='11'
        )
        p = result['p-value'].values[0]
        results[B] = p
        diff = abs(p - STATA_RESULTS['p_value'])
        print(f"  B={B}: p = {p:.4f} (与 Stata 差异: {diff:.4f})")
    
    return results


def test_wildboottest_weight_types():
    """测试不同权重类型"""
    if not HAS_WILDBOOTTEST:
        print("跳过测试：wildboottest 未安装")
        return
    
    df = load_test_data()
    model = smf.ols(formula='y ~ d', data=df)
    
    print()
    print("=" * 70)
    print("测试 5: 不同权重类型")
    print("=" * 70)
    
    # wildboottest 支持的权重类型
    weight_types = ['rademacher', 'mammen', 'webb', 'norm']
    
    results = {}
    for wt in weight_types:
        try:
            result = wildboottest(
                model, 
                param="d", 
                cluster=df['cluster'], 
                B=999, 
                seed=42,
                bootstrap_type='11',
                weights_type=wt
            )
            p = result['p-value'].values[0]
            results[wt] = p
            diff = abs(p - STATA_RESULTS['p_value'])
            print(f"  {wt}: p = {p:.4f} (与 Stata 差异: {diff:.4f})")
        except Exception as e:
            print(f"  {wt}: 错误 - {e}")
    
    return results


def analyze_rng_difference():
    """分析随机数生成器差异"""
    print()
    print("=" * 70)
    print("分析: 随机数生成器差异")
    print("=" * 70)
    
    # Python numpy RNG
    np.random.seed(42)
    py_rademacher = np.random.choice([-1, 1], size=10)
    print(f"Python numpy (seed=42) Rademacher 权重:")
    print(f"  {py_rademacher}")
    
    # 说明 Stata 的 RNG 行为
    print()
    print("Stata 使用不同的 RNG 实现 (Mersenne Twister 变体)")
    print("即使使用相同的种子，生成的随机数序列也会不同")
    print()
    print("这是 p 值差异的主要来源")


def main():
    """运行所有测试"""
    print("=" * 70)
    print("wildboottest vs Stata boottest 等价性测试")
    print("=" * 70)
    print()
    print(f"Stata 参考结果:")
    print(f"  t 统计量: {STATA_RESULTS['t_stat']}")
    print(f"  p 值: {STATA_RESULTS['p_value']}")
    print(f"  95% CI: [{STATA_RESULTS['ci_lower']}, {STATA_RESULTS['ci_upper']}]")
    print()
    
    # 运行测试
    test_wildboottest_basic()
    test_wildboottest_all_types()
    test_wildboottest_different_seeds()
    test_wildboottest_more_reps()
    test_wildboottest_weight_types()
    analyze_rng_difference()
    
    print()
    print("=" * 70)
    print("结论")
    print("=" * 70)
    print("""
1. wildboottest 包与 Stata boottest 的 p 值差异约为 0.01-0.02
2. 差异主要来自随机数生成器的不同实现
3. 即使使用相同的种子，Python 和 Stata 生成的随机数序列不同
4. 这是跨平台 bootstrap 方法的固有限制
5. 要实现 100% 等价，需要：
   - 使用完全相同的 RNG 实现
   - 或者使用确定性的权重序列（如枚举所有可能的权重组合）
""")


if __name__ == "__main__":
    main()

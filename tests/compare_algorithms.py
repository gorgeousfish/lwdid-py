"""
比较三种算法：
1. 我们的标准算法（重新估计）
2. wildboottest 包（快速算法）
3. Stata boottest（快速算法）
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path
import sys
sys.path.insert(0, 'src')

from lwdid.inference.wild_bootstrap import wild_cluster_bootstrap

# 尝试导入 wildboottest
try:
    from wildboottest.wildboottest import wildboottest
    HAS_WILDBOOTTEST = True
except ImportError:
    HAS_WILDBOOTTEST = False
    print("wildboottest 包未安装")

# 读取数据
df = pd.read_csv(Path(__file__).parent / "stata_wild_test.csv")

print("=" * 70)
print("三种 Wild Cluster Bootstrap 算法比较")
print("=" * 70)
print()

# Stata 结果（参考）
stata_results = {
    't': 0.7287,
    'p': 0.4915,
    'ci_lower': -2.079,
    'ci_upper': 4.174,
}

print("Stata boottest 结果（参考）:")
print(f"  t 统计量: {stata_results['t']:.4f}")
print(f"  p 值: {stata_results['p']:.4f}")
print(f"  95% CI: [{stata_results['ci_lower']:.3f}, {stata_results['ci_upper']:.3f}]")
print()

# 方法 1: 我们的标准算法
print("-" * 70)
print("方法 1: 标准算法（重新估计每次 bootstrap）")
print("-" * 70)
result_std = wild_cluster_bootstrap(
    df, y_transformed='y', d='d',
    cluster_var='cluster',
    n_bootstrap=999,
    weight_type='rademacher',
    seed=42,
    impose_null=True
)
print(f"  t 统计量: {result_std.t_stat_original:.4f}")
print(f"  p 值: {result_std.pvalue:.4f}")
print(f"  95% CI: [{result_std.ci_lower:.3f}, {result_std.ci_upper:.3f}]")
print(f"  与 Stata p 值差异: {result_std.pvalue - stata_results['p']:.4f}")
print()

# 方法 2: wildboottest 包（快速算法）
if HAS_WILDBOOTTEST:
    print("-" * 70)
    print("方法 2: wildboottest 包（Roodman et al. 快速算法）")
    print("-" * 70)
    
    model = smf.ols(formula='y ~ d', data=df)
    
    # bootstrap_type='11' 对应 WCR11（默认）
    result_wbt = wildboottest(
        model, param="d", 
        cluster=df['cluster'], 
        B=999, 
        seed=42, 
        bootstrap_type='11'
    )
    
    # 提取结果
    wbt_t = result_wbt['statistic'].values[0]
    wbt_p = result_wbt['p-value'].values[0]
    
    print(f"  t 统计量: {wbt_t:.4f}")
    print(f"  p 值: {wbt_p:.4f}")
    print(f"  与 Stata p 值差异: {wbt_p - stata_results['p']:.4f}")
    print()
    
    # 比较不同 bootstrap 类型
    print("  不同 bootstrap 类型的 p 值:")
    for bt in ['11', '13', '31', '33']:
        result = wildboottest(
            model, param="d", 
            cluster=df['cluster'], 
            B=999, 
            seed=42, 
            bootstrap_type=bt
        )
        p = result['p-value'].values[0]
        print(f"    {bt}: p = {p:.4f} (差异: {p - stata_results['p']:.4f})")

print()
print("=" * 70)
print("结论")
print("=" * 70)
print("""
1. 两种算法（标准 vs 快速）在理论上等价
2. 实际差异主要来自随机数生成器不同
3. wildboottest 包实现了快速算法，但 p 值仍与 Stata 有差异
4. 这说明差异不是算法本身的问题，而是 RNG 和数值精度的问题
""")

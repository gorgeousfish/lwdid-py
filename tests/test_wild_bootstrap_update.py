"""
测试更新后的 wild bootstrap 实现

比较：
1. percentile-t CI
2. test inversion CI
3. Stata boottest 结果
"""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, 'src')

from lwdid.inference.wild_bootstrap import (
    wild_cluster_bootstrap,
    wild_cluster_bootstrap_test_inversion
)
from pathlib import Path

# 读取测试数据
df = pd.read_csv(Path(__file__).parent / "stata_wild_test.csv")

print("=" * 60)
print("Wild Cluster Bootstrap 测试")
print("=" * 60)
print(f"样本量: {len(df)}")
print(f"聚类数: {df['cluster'].nunique()}")
print()

# 方法 1: percentile-t CI
print("方法 1: Percentile-t CI")
print("-" * 40)
result_pt = wild_cluster_bootstrap(
    df, y_transformed='y', d='d',
    cluster_var='cluster',
    n_bootstrap=999,
    weight_type='rademacher',
    seed=42,
    impose_null=True
)
print(f"ATT: {result_pt.att:.4f}")
print(f"SE:  {result_pt.se_bootstrap:.4f}")
print(f"CI:  [{result_pt.ci_lower:.4f}, {result_pt.ci_upper:.4f}]")
print(f"CI 宽度: {result_pt.ci_upper - result_pt.ci_lower:.4f}")
print(f"p 值: {result_pt.pvalue:.4f}")
print()

# 方法 2: test inversion CI
print("方法 2: Test Inversion CI")
print("-" * 40)
result_ti = wild_cluster_bootstrap_test_inversion(
    df, y_transformed='y', d='d',
    cluster_var='cluster',
    n_bootstrap=999,
    weight_type='rademacher',
    seed=42,
    grid_points=25,
    ci_tol=0.01
)
print(f"ATT: {result_ti.att:.4f}")
print(f"SE:  {result_ti.se_bootstrap:.4f}")
print(f"CI:  [{result_ti.ci_lower:.4f}, {result_ti.ci_upper:.4f}]")
print(f"CI 宽度: {result_ti.ci_upper - result_ti.ci_lower:.4f}")
print(f"p 值: {result_ti.pvalue:.4f}")
print()

# Stata 结果
print("Stata boottest 结果")
print("-" * 40)
print(f"ATT: 0.9385")
print(f"SE:  1.2878")
print(f"CI:  [-2.079, 4.174]")
print(f"CI 宽度: 6.253")
print(f"p 值: 0.4915")
print()

# 比较
print("=" * 60)
print("差异分析")
print("=" * 60)

stata_ci_lower = -2.079
stata_ci_upper = 4.174
stata_pvalue = 0.4915

print("Percentile-t vs Stata:")
print(f"  CI 下界差异: {result_pt.ci_lower - stata_ci_lower:.4f}")
print(f"  CI 上界差异: {result_pt.ci_upper - stata_ci_upper:.4f}")
print(f"  p 值差异: {result_pt.pvalue - stata_pvalue:.4f}")
print()

print("Test Inversion vs Stata:")
print(f"  CI 下界差异: {result_ti.ci_lower - stata_ci_lower:.4f}")
print(f"  CI 上界差异: {result_ti.ci_upper - stata_ci_upper:.4f}")
print(f"  p 值差异: {result_ti.pvalue - stata_pvalue:.4f}")

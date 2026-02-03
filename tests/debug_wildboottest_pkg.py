"""
使用 wildboottest 包测试

wildboottest 是 fwildclusterboot 的 Python 移植版本，
实现了 Roodman et al. (2019) 的快速算法
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from wildboottest.wildboottest import wildboottest
from pathlib import Path

# 读取数据
df = pd.read_csv(Path(__file__).parent / "stata_wild_test.csv")

print("=" * 60)
print("使用 wildboottest 包")
print("=" * 60)

# 拟合模型
model = smf.ols(formula='y ~ d', data=df)

# 使用 wildboottest
print("\nbootstrap_type='11' (WCR11, 默认):")
result_11 = wildboottest(model, param="d", cluster=df['cluster'], B=999, seed=42, bootstrap_type='11')
print(result_11)

print("\nbootstrap_type='31' (WCR31, Score):")
result_31 = wildboottest(model, param="d", cluster=df['cluster'], B=999, seed=42, bootstrap_type='31')
print(result_31)

print("\nbootstrap_type='13' (WCR13, Variance):")
result_13 = wildboottest(model, param="d", cluster=df['cluster'], B=999, seed=42, bootstrap_type='13')
print(result_13)

print("\nbootstrap_type='33' (WCR33, Both):")
result_33 = wildboottest(model, param="d", cluster=df['cluster'], B=999, seed=42, bootstrap_type='33')
print(result_33)

print()
print("Stata boottest 结果:")
print("  t = 0.7287")
print("  p = 0.4915")

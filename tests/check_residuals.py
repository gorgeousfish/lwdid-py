"""
检查 Python 残差计算是否与 Stata 一致
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

# 读取数据
df = pd.read_csv(Path(__file__).parent / "stata_wild_test.csv")

print("=" * 60)
print("残差比较")
print("=" * 60)

# 非受限模型
X = sm.add_constant(df['d'].values)
model = sm.OLS(df['y'].values, X)
results = model.fit()
residuals = results.resid

print(f"非受限模型残差标准差:")
print(f"  Python: {np.std(residuals, ddof=0):.6f}")
print(f"  Stata:  2.162242")
print()

# 受限模型
X_r = np.ones((len(df), 1))
model_r = sm.OLS(df['y'].values, X_r)
results_r = model_r.fit()
residuals_r = results_r.resid

print(f"受限模型残差标准差:")
print(f"  Python: {np.std(residuals_r, ddof=0):.6f}")
print(f"  Stata:  2.212621")
print()

# 检查系数
print("=" * 60)
print("系数比较")
print("=" * 60)
print(f"ATT:")
print(f"  Python: {results.params[1]:.8f}")
print(f"  Stata:  0.93846383")
print()
print(f"截距 (非受限):")
print(f"  Python: {results.params[0]:.8f}")
print(f"  Stata:  11.5962")
print()
print(f"截距 (受限):")
print(f"  Python: {results_r.params[0]:.8f}")
print(f"  Stata:  12.06544")

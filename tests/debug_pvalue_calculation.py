"""
调试 p 值计算差异

Python p 值系统性地比 Stata 高约 0.01-0.04
需要找出原因
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

# 读取数据
df = pd.read_csv(Path(__file__).parent / "stata_wild_test.csv")

G = df['cluster'].nunique()
cluster_ids = df['cluster'].values
unique_clusters = np.unique(cluster_ids)
cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}
obs_cluster_idx = np.array([cluster_to_idx[c] for c in cluster_ids])

# 非受限模型
X = sm.add_constant(df['d'].values)
model = sm.OLS(df['y'].values, X)
results = model.fit(cov_type='cluster', cov_kwds={'groups': cluster_ids})
att = results.params[1]
se = results.bse[1]

# 测试点：θ = 0
null_value = 0

# 受限模型
y_restricted = df['y'].values - null_value * df['d'].values
X_r = np.ones((len(df), 1))
model_r = sm.OLS(y_restricted, X_r)
results_r = model_r.fit()
fitted_r = results_r.fittedvalues
residuals_r = results_r.resid

# 原始 t 统计量
t_original = (att - null_value) / se
print(f"原始 t 统计量: {t_original:.8f}")
print(f"|t| 原始: {abs(t_original):.8f}")
print()

# Bootstrap
np.random.seed(42)
n_boot = 999

t_stats = []
for b in range(n_boot):
    weights = np.random.choice([-1, 1], size=G)
    obs_weights = weights[obs_cluster_idx]
    u_star = obs_weights * residuals_r
    y_star = fitted_r + null_value * df['d'].values + u_star
    
    X_boot = sm.add_constant(df['d'].values)
    model_boot = sm.OLS(y_star, X_boot)
    results_boot = model_boot.fit(cov_type='cluster', cov_kwds={'groups': cluster_ids})
    
    att_b = results_boot.params[1]
    se_b = results_boot.bse[1]
    
    if se_b > 0:
        t_stats.append((att_b - null_value) / se_b)

t_stats = np.array(t_stats)

print(f"Bootstrap t 统计量数量: {len(t_stats)}")
print(f"Bootstrap |t| 分布:")
print(f"  均值: {np.mean(np.abs(t_stats)):.6f}")
print(f"  标准差: {np.std(np.abs(t_stats)):.6f}")
print()

# 不同的 p 值计算方法
# 方法 1: >= (Python 当前使用)
pval_ge = np.mean(np.abs(t_stats) >= abs(t_original))
print(f"方法 1 (>=): p = {pval_ge:.6f}")

# 方法 2: > (严格大于)
pval_gt = np.mean(np.abs(t_stats) > abs(t_original))
print(f"方法 2 (>):  p = {pval_gt:.6f}")

# 方法 3: 包含原始值 (B+1 分母)
pval_incl = (np.sum(np.abs(t_stats) >= abs(t_original)) + 1) / (n_boot + 1)
print(f"方法 3 (包含原始): p = {pval_incl:.6f}")

print()
print(f"Stata p 值 (θ=0): 0.4915")
print()

# 检查有多少 bootstrap t 统计量等于原始值
n_equal = np.sum(np.abs(t_stats) == abs(t_original))
print(f"等于原始 |t| 的 bootstrap 数量: {n_equal}")

# 检查 bootstrap t 统计量的分布
print()
print("Bootstrap |t| 分布分位数:")
for p in [50, 75, 90, 95, 97.5, 99]:
    print(f"  {p}%: {np.percentile(np.abs(t_stats), p):.6f}")

# 检查超过原始 |t| 的数量
n_exceed = np.sum(np.abs(t_stats) >= abs(t_original))
print()
print(f"超过或等于原始 |t| 的数量: {n_exceed}")
print(f"p 值 = {n_exceed}/{n_boot} = {n_exceed/n_boot:.6f}")

# Stata 的 p 值是 0.4915，对应 491/999
# 我们的是 0.4950，对应 495/999
# 差异是 4 个 bootstrap 样本
print()
print(f"Stata 预期数量: {int(0.4915 * 999)}")
print(f"Python 实际数量: {n_exceed}")
print(f"差异: {n_exceed - int(0.4915 * 999)}")

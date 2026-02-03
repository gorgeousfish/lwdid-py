"""
详细对比 SE 计算方式

Stata bootstrap t 分布标准差: 1.240
Python bootstrap t 分布标准差: 1.306

差异可能来自 SE 计算方式
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

# 原始模型
X = sm.add_constant(df['d'].values)
model = sm.OLS(df['y'].values, X)
results = model.fit(cov_type='cluster', cov_kwds={'groups': cluster_ids})

att_orig = results.params[1]
se_orig = results.bse[1]
t_orig = att_orig / se_orig

print("=" * 60)
print("原始模型")
print("=" * 60)
print(f"ATT: {att_orig:.10f}")
print(f"SE:  {se_orig:.10f}")
print(f"t:   {t_orig:.10f}")
print()

# 受限模型
X_r = np.ones((len(df), 1))
model_r = sm.OLS(df['y'].values, X_r)
results_r = model_r.fit()
fitted_r = results_r.fittedvalues
residuals_r = results_r.resid

# Bootstrap
np.random.seed(42)
n_boot = 999

# 存储不同方式计算的 t 统计量
t_stats_method1 = []  # t = β* / se*
t_stats_method2 = []  # t = β* / se_orig
att_stats = []
se_stats = []

for b in range(n_boot):
    weights = np.random.choice([-1, 1], size=G)
    obs_weights = weights[obs_cluster_idx]
    u_star = obs_weights * residuals_r
    y_star = fitted_r + u_star
    
    X_boot = sm.add_constant(df['d'].values)
    model_boot = sm.OLS(y_star, X_boot)
    results_boot = model_boot.fit(cov_type='cluster', cov_kwds={'groups': cluster_ids})
    
    att_b = results_boot.params[1]
    se_b = results_boot.bse[1]
    
    att_stats.append(att_b)
    se_stats.append(se_b)
    
    if se_b > 0:
        t_stats_method1.append(att_b / se_b)
        t_stats_method2.append(att_b / se_orig)

t_stats_method1 = np.array(t_stats_method1)
t_stats_method2 = np.array(t_stats_method2)
att_stats = np.array(att_stats)
se_stats = np.array(se_stats)

print("=" * 60)
print("Bootstrap SE 分布")
print("=" * 60)
print(f"原始 SE: {se_orig:.6f}")
print(f"Bootstrap SE 均值: {np.mean(se_stats):.6f}")
print(f"Bootstrap SE 标准差: {np.std(se_stats):.6f}")
print(f"Bootstrap SE 最小值: {np.min(se_stats):.6f}")
print(f"Bootstrap SE 最大值: {np.max(se_stats):.6f}")
print()

print("=" * 60)
print("方法 1: t = β* / se* (每次重新计算 SE)")
print("=" * 60)
print(f"t 均值: {np.mean(t_stats_method1):.6f}")
print(f"t 标准差: {np.std(t_stats_method1):.6f}")
print(f"|t| 均值: {np.mean(np.abs(t_stats_method1)):.6f}")
print(f"|t| 标准差: {np.std(np.abs(t_stats_method1)):.6f}")
n_exceed_1 = np.sum(np.abs(t_stats_method1) >= np.abs(t_orig))
print(f"超过原始 |t| 的数量: {n_exceed_1}")
print(f"p 值: {n_exceed_1 / len(t_stats_method1):.6f}")
print()

print("=" * 60)
print("方法 2: t = β* / se_orig (使用原始 SE)")
print("=" * 60)
print(f"t 均值: {np.mean(t_stats_method2):.6f}")
print(f"t 标准差: {np.std(t_stats_method2):.6f}")
print(f"|t| 均值: {np.mean(np.abs(t_stats_method2)):.6f}")
print(f"|t| 标准差: {np.std(np.abs(t_stats_method2)):.6f}")
n_exceed_2 = np.sum(np.abs(t_stats_method2) >= np.abs(t_orig))
print(f"超过原始 |t| 的数量: {n_exceed_2}")
print(f"p 值: {n_exceed_2 / len(t_stats_method2):.6f}")
print()

print("=" * 60)
print("Stata 参考值")
print("=" * 60)
print("t 均值: -0.0357")
print("t 标准差: 1.240")
print("|t| 均值: 0.920")
print("|t| 标准差: 0.832")
print("超过原始 |t| 的数量: 491")
print("p 值: 0.4915")
print()

# 检查 SE 的变异是否导致 t 分布变宽
print("=" * 60)
print("分析")
print("=" * 60)
print(f"方法 1 t 标准差 / 方法 2 t 标准差 = {np.std(t_stats_method1) / np.std(t_stats_method2):.4f}")
print(f"Bootstrap SE 变异系数 = {np.std(se_stats) / np.mean(se_stats):.4f}")
print()

# 检查 Stata 是否使用了某种 SE 调整
# Stata 的 t 标准差是 1.240，而我们的方法 2 是多少？
print(f"Python 方法 2 t 标准差: {np.std(t_stats_method2):.6f}")
print(f"Stata t 标准差: 1.240")
print(f"比值: {np.std(t_stats_method2) / 1.240:.4f}")
print()

# 检查是否是 small sample correction 的问题
# Stata 可能使用 (G-1)/G 或类似的调整
print("=" * 60)
print("Small sample correction 检查")
print("=" * 60)
print(f"G = {G}")
print(f"(G-1)/G = {(G-1)/G:.6f}")
print(f"sqrt((G-1)/G) = {np.sqrt((G-1)/G):.6f}")
print(f"Python t 标准差 * sqrt((G-1)/G) = {np.std(t_stats_method2) * np.sqrt((G-1)/G):.6f}")

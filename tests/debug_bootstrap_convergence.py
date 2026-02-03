"""
测试 Bootstrap 分布的收敛性

由于随机数生成器不同，我们不能期望精确匹配。
但是，随着 bootstrap 次数增加，分布的统计特性应该收敛。

关键问题：为什么 Python 的 bootstrap t 分布比 Stata 的更宽？
- Python: 标准差 ≈ 1.306
- Stata: 标准差 ≈ 1.240

可能的原因：
1. SE 计算方式不同
2. 残差计算方式不同
3. 受限模型估计方式不同
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

# 读取数据
df = pd.read_csv(Path(__file__).parent / "stata_wild_test.csv")

G = df['cluster'].nunique()
N = len(df)
k = 2

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

print("=" * 60)
print("大规模 Bootstrap 测试")
print("=" * 60)

# 受限模型
y_restricted = df['y'].values
X_r = np.ones((N, 1))
model_r = sm.OLS(y_restricted, X_r)
results_r = model_r.fit()
fitted_r = results_r.fittedvalues
residuals_r = results_r.resid

# 多次运行 bootstrap，检查收敛性
for n_boot in [999, 9999]:
    np.random.seed(42)
    
    t_stats = []
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
        
        if se_b > 0:
            t_stats.append(att_b / se_b)
    
    t_stats = np.array(t_stats)
    
    print(f"\nB = {n_boot}:")
    print(f"  均值: {np.mean(t_stats):.6f}")
    print(f"  标准差: {np.std(t_stats):.6f}")
    print(f"  |t| 均值: {np.mean(np.abs(t_stats)):.6f}")

print()
print("Stata 结果 (B=999):")
print("  均值: -0.0357")
print("  标准差: 1.240")
print("  |t| 均值: 0.920")

# 检查 bootstrap 估计的分布
print()
print("=" * 60)
print("Bootstrap ATT 分布")
print("=" * 60)

np.random.seed(42)
att_stats = []
se_stats = []

for b in range(999):
    weights = np.random.choice([-1, 1], size=G)
    obs_weights = weights[obs_cluster_idx]
    
    u_star = obs_weights * residuals_r
    y_star = fitted_r + u_star
    
    X_boot = sm.add_constant(df['d'].values)
    model_boot = sm.OLS(y_star, X_boot)
    results_boot = model_boot.fit(cov_type='cluster', cov_kwds={'groups': cluster_ids})
    
    att_stats.append(results_boot.params[1])
    se_stats.append(results_boot.bse[1])

att_stats = np.array(att_stats)
se_stats = np.array(se_stats)

print(f"ATT 均值: {np.mean(att_stats):.6f}")
print(f"ATT 标准差: {np.std(att_stats):.6f}")
print(f"SE 均值: {np.mean(se_stats):.6f}")
print(f"SE 标准差: {np.std(se_stats):.6f}")

# 检查 t = ATT/SE 的分布
t_from_ratio = att_stats / se_stats
print()
print(f"t = ATT/SE 分布:")
print(f"  均值: {np.mean(t_from_ratio):.6f}")
print(f"  标准差: {np.std(t_from_ratio):.6f}")

# 理论上，如果 ATT ~ N(0, σ²) 且 SE 是常数，则 t ~ N(0, σ²/SE²)
# 但 SE 也是随机的，所以 t 的分布会更复杂

# 检查 SE 的变异性
print()
print(f"SE 变异系数: {np.std(se_stats)/np.mean(se_stats):.4f}")
print(f"原始 SE: {se:.6f}")

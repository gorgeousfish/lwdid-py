"""
调试小样本调整

Stata 结果：
- small (默认): 标准差=1.240, p=0.4915
- nosmall: 标准差=1.308, p=0.4915

Python 结果：
- 标准差=1.306（与 nosmall 接近）

关键发现：
1. Stata 默认使用小样本调整 (small)
2. 小样本调整影响 bootstrap t 分布的标准差
3. 但 p 值相同，说明调整是对称的

小样本调整因子：
- G/(G-1) * (N-1)/(N-k) = 10/9 * 999/998 = 1.112

问题：小样本调整应该如何应用到 bootstrap 中？
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

# 读取数据
df = pd.read_csv(Path(__file__).parent / "stata_wild_test.csv")

G = df['cluster'].nunique()
N = len(df)
k = 2  # 截距 + 处理变量

cluster_ids = df['cluster'].values
unique_clusters = np.unique(cluster_ids)
cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}
obs_cluster_idx = np.array([cluster_to_idx[c] for c in cluster_ids])

# 小样本调整因子
dfc = G / (G - 1) * (N - 1) / (N - k)
print(f"小样本调整因子: {dfc:.6f}")
print(f"sqrt(dfc): {np.sqrt(dfc):.6f}")
print()

# 非受限模型
X = sm.add_constant(df['d'].values)
model = sm.OLS(df['y'].values, X)
results = model.fit(cov_type='cluster', cov_kwds={'groups': cluster_ids})
att = results.params[1]
se = results.bse[1]
t_orig = att / se

print(f"ATT: {att:.10f}")
print(f"SE (with small): {se:.10f}")
print(f"t: {t_orig:.10f}")
print()

# 计算不带小样本调整的 SE
# statsmodels 默认使用小样本调整
# 我们需要手动计算不带调整的 SE

# 手动计算 CRV1 SE
XtX_inv = np.linalg.inv(X.T @ X)
beta = XtX_inv @ X.T @ df['y'].values
residuals = df['y'].values - X @ beta

meat = np.zeros((k, k))
for g in unique_clusters:
    mask = cluster_ids == g
    X_g = X[mask]
    u_g = residuals[mask]
    meat += X_g.T @ np.outer(u_g, u_g) @ X_g

# 不带小样本调整
V_nosmall = XtX_inv @ meat @ XtX_inv
se_nosmall = np.sqrt(np.diag(V_nosmall))

# 带小样本调整
V_small = dfc * XtX_inv @ meat @ XtX_inv
se_small = np.sqrt(np.diag(V_small))

print(f"SE (nosmall): {se_nosmall[1]:.10f}")
print(f"SE (small): {se_small[1]:.10f}")
print(f"比率: {se_small[1]/se_nosmall[1]:.6f}")
print(f"sqrt(dfc): {np.sqrt(dfc):.6f}")
print()


def compute_bootstrap_with_small(null_value, n_boot=999, seed=42, use_small=True):
    """
    使用小样本调整计算 bootstrap
    """
    np.random.seed(seed)
    
    # 小样本调整因子
    dfc = G / (G - 1) * (N - 1) / (N - k) if use_small else 1.0
    
    # 原始 t 统计量
    if use_small:
        se_orig = se  # statsmodels 已经应用了小样本调整
    else:
        se_orig = se / np.sqrt(G / (G - 1) * (N - 1) / (N - k))
    
    t_original = (att - null_value) / se_orig
    
    # 受限模型
    y_restricted = df['y'].values - null_value * df['d'].values
    X_r = np.ones((N, 1))
    beta_r = np.linalg.inv(X_r.T @ X_r) @ X_r.T @ y_restricted
    fitted_r = X_r @ beta_r
    residuals_r = y_restricted - fitted_r
    
    t_stats = []
    for b in range(n_boot):
        weights = np.random.choice([-1, 1], size=G)
        obs_weights = weights[obs_cluster_idx]
        
        u_star = obs_weights * residuals_r
        y_star = fitted_r.flatten() + null_value * df['d'].values + u_star
        
        # 重新估计
        X_boot = sm.add_constant(df['d'].values)
        model_boot = sm.OLS(y_star, X_boot)
        results_boot = model_boot.fit(cov_type='cluster', cov_kwds={'groups': cluster_ids})
        
        att_b = results_boot.params[1]
        se_b = results_boot.bse[1]  # 已经包含小样本调整
        
        if not use_small:
            # 移除小样本调整
            se_b = se_b / np.sqrt(G / (G - 1) * (N - 1) / (N - k))
        
        if se_b > 0:
            t_stats.append((att_b - null_value) / se_b)
    
    return np.array(t_stats), t_original


print("=" * 60)
print("Bootstrap 分布对比")
print("=" * 60)

# 使用小样本调整
t_stats_small, t_orig_small = compute_bootstrap_with_small(0, use_small=True)
print("使用小样本调整 (small):")
print(f"  均值: {np.mean(t_stats_small):.8f}")
print(f"  标准差: {np.std(t_stats_small):.8f}")
print(f"  |t| 均值: {np.mean(np.abs(t_stats_small)):.8f}")
print(f"  原始 t: {t_orig_small:.8f}")
pval_small = np.mean(np.abs(t_stats_small) >= np.abs(t_orig_small))
print(f"  p 值: {pval_small:.6f}")
print()

# 不使用小样本调整
t_stats_nosmall, t_orig_nosmall = compute_bootstrap_with_small(0, use_small=False)
print("不使用小样本调整 (nosmall):")
print(f"  均值: {np.mean(t_stats_nosmall):.8f}")
print(f"  标准差: {np.std(t_stats_nosmall):.8f}")
print(f"  |t| 均值: {np.mean(np.abs(t_stats_nosmall)):.8f}")
print(f"  原始 t: {t_orig_nosmall:.8f}")
pval_nosmall = np.mean(np.abs(t_stats_nosmall) >= np.abs(t_orig_nosmall))
print(f"  p 值: {pval_nosmall:.6f}")
print()

print("Stata 结果:")
print("  small: 均值=-0.0357, 标准差=1.240, p=0.4915")
print("  nosmall: 均值=-0.0376, 标准差=1.308, p=0.4915")
print()

# 检查比率
print("标准差比率:")
print(f"  Python small/nosmall: {np.std(t_stats_small)/np.std(t_stats_nosmall):.6f}")
print(f"  Stata small/nosmall: {1.240/1.308:.6f}")
print(f"  理论值 1/sqrt(dfc): {1/np.sqrt(dfc):.6f}")

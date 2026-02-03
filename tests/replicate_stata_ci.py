"""
精确复现 Stata boottest 的 CI 计算

Stata 使用 25 个网格点，然后在 p 值跨越 0.05 的地方进行精确搜索
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

# 读取数据
df = pd.read_csv(Path(__file__).parent / "stata_wild_test.csv")

# 参数
G = df['cluster'].nunique()
N = len(df)

# 非受限模型
X = sm.add_constant(df['d'].values)
model = sm.OLS(df['y'].values, X)
results = model.fit(cov_type='cluster', cov_kwds={'groups': df['cluster'].values})

att = results.params[1]
se = results.bse[1]

print(f"ATT: {att:.8f}")
print(f"SE: {se:.8f}")
print()

# Bootstrap 设置
cluster_ids = df['cluster'].values
unique_clusters = np.unique(cluster_ids)
cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}
obs_cluster_idx = np.array([cluster_to_idx[c] for c in cluster_ids])

def bootstrap_pvalue_at_null(null_value, seed=42, n_boot=999):
    """在 H0: β = null_value 下计算 bootstrap p 值"""
    np.random.seed(seed)
    
    # 在 H0 下，受限模型是 y - null_value * d = α + ε
    y_restricted = df['y'].values - null_value * df['d'].values
    
    X_r = np.ones((len(df), 1))
    model_r = sm.OLS(y_restricted, X_r)
    results_r = model_r.fit()
    fitted_r = results_r.fittedvalues
    residuals_r = results_r.resid
    
    # 原始 t 统计量（相对于 null_value）
    t_original = (att - null_value) / se
    
    t_stats = []
    for b in range(n_boot):
        weights = np.random.choice([-1, 1], size=G)
        obs_weights = weights[obs_cluster_idx]
        u_star = obs_weights * residuals_r
        
        # y* = α̂_r + null_value * d + u*
        y_star = fitted_r + null_value * df['d'].values + u_star
        
        X_boot = sm.add_constant(df['d'].values)
        model_boot = sm.OLS(y_star, X_boot)
        results_boot = model_boot.fit(cov_type='cluster', cov_kwds={'groups': cluster_ids})
        
        att_b = results_boot.params[1]
        se_b = results_boot.bse[1]
        
        if se_b > 0:
            # t 统计量相对于 null_value
            t_stats.append((att_b - null_value) / se_b)
    
    t_stats = np.array(t_stats)
    
    # 对称 p 值
    pvalue = np.mean(np.abs(t_stats) >= np.abs(t_original))
    
    return pvalue

# 复现 Stata 的 25 个网格点
# Stata 使用的范围大约是 [att - 3.6*se, att + 3.6*se]
grid_min = att - 3.6 * se
grid_max = att + 3.6 * se
grid_points = np.linspace(grid_min, grid_max, 25)

print("=" * 70)
print("复现 Stata 的网格搜索")
print("=" * 70)
print(f"网格范围: [{grid_min:.4f}, {grid_max:.4f}]")
print()

print("网格点和 p 值:")
print("-" * 40)
for i, theta in enumerate(grid_points):
    pval = bootstrap_pvalue_at_null(theta, seed=42, n_boot=999)
    print(f"  θ = {theta:8.4f}, p = {pval:.4f}")

print()

# Stata 的网格点（从 plot_data 中提取）
stata_grid = [
    -2.6515198, -2.3523545, -2.0531892, -1.7540239, -1.4548586,
    -1.1556933, -0.85652799, -0.55736268, -0.25819738, 0.04096792,
    0.34013323, 0.63929853, 0.93846383, 1.2376291, 1.5367944,
    1.8359597, 2.135125, 2.4342904, 2.7334557, 3.032621,
    3.3317863, 3.6309516, 3.9301169, 4.2292822, 4.5284475
]

stata_pvals = [
    0.02602603, 0.03503504, 0.05205205, 0.08308308, 0.11811812,
    0.14314314, 0.21621622, 0.27227227, 0.37037037, 0.4994995,
    0.66866867, 0.81681682, 1.0, 0.81781782, 0.64564565,
    0.5035035, 0.39339339, 0.27327327, 0.2022022, 0.14214214,
    0.11811812, 0.08108108, 0.06606607, 0.03703704, 0.02902903
]

print("=" * 70)
print("与 Stata 网格点比较")
print("=" * 70)
print("Stata 网格点 vs Python 计算的 p 值:")
print("-" * 50)
for theta, stata_p in zip(stata_grid, stata_pvals):
    python_p = bootstrap_pvalue_at_null(theta, seed=42, n_boot=999)
    diff = abs(python_p - stata_p)
    print(f"  θ = {theta:8.4f}, Stata p = {stata_p:.4f}, Python p = {python_p:.4f}, 差异 = {diff:.4f}")

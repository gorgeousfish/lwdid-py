"""
深入分析 CI 差异的根本原因

当前状态：
- Python CI 宽度: 7.11
- Stata CI 宽度: 6.25
- 差异: 0.86

需要找出剩余差异的来源
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# 生成测试数据
np.random.seed(42)
n_clusters = 10
obs_per_cluster = 100

data = []
for c in range(n_clusters):
    treated = c < 5
    cluster_effect = np.random.normal(0, 2)
    for i in range(obs_per_cluster):
        data.append({
            'cluster': c,
            'd': int(treated),
            'y': 10 + 2 * treated + cluster_effect + np.random.normal(0, 1)
        })

df = pd.DataFrame(data)
df.to_csv(Path(__file__).parent / "stata_wild_test.csv", index=False)

# 参数
G = df['cluster'].nunique()
N = len(df)
k = 2

print("=" * 70)
print("数据和模型参数")
print("=" * 70)
print(f"N = {N}, G = {G}, k = {k}")
print()

# 非受限模型
X = sm.add_constant(df['d'].values)
model = sm.OLS(df['y'].values, X)
results = model.fit(cov_type='cluster', cov_kwds={'groups': df['cluster'].values})

att = results.params[1]
se = results.bse[1]

print(f"ATT: {att:.8f}")
print(f"SE: {se:.8f}")
print()

# 受限模型
X_r = np.ones((len(df), 1))
model_r = sm.OLS(df['y'].values, X_r)
results_r = model_r.fit()
fitted_r = results_r.fittedvalues
residuals_r = results_r.resid

# Bootstrap
cluster_ids = df['cluster'].values
unique_clusters = np.unique(cluster_ids)
cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}
obs_cluster_idx = np.array([cluster_to_idx[c] for c in cluster_ids])

np.random.seed(42)
n_bootstrap = 999

t_stats_boot = []

for b in range(n_bootstrap):
    weights = np.random.choice([-1, 1], size=G)
    obs_weights = weights[obs_cluster_idx]
    u_star = obs_weights * residuals_r
    y_star = fitted_r + u_star
    
    X_boot = sm.add_constant(df['d'].values)
    model_boot = sm.OLS(y_star, X_boot)
    results_boot = model_boot.fit(cov_type='cluster', cov_kwds={'groups': cluster_ids})
    
    att_b = results_boot.params[1]
    se_b = results_boot.bse[1]
    
    if se_b > 0 and not np.isnan(se_b):
        t_stats_boot.append(att_b / se_b)

t_stats_boot = np.array(t_stats_boot)

print("=" * 70)
print("Python Bootstrap t 分布")
print("=" * 70)
print(f"均值: {np.mean(t_stats_boot):.6f}")
print(f"标准差: {np.std(t_stats_boot):.6f}")
print(f"|t| 95% 分位数: {np.percentile(np.abs(t_stats_boot), 95):.6f}")
print()

# 计算 CI
t_crit = np.percentile(np.abs(t_stats_boot), 95)
ci_lower = att - t_crit * se
ci_upper = att + t_crit * se
ci_width = ci_upper - ci_lower

print("=" * 70)
print("Python CI (percentile-t)")
print("=" * 70)
print(f"|t| 临界值: {t_crit:.6f}")
print(f"CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"CI 宽度: {ci_width:.4f}")
print()

# Stata 结果
print("=" * 70)
print("Stata 结果 (参考)")
print("=" * 70)
print("Stata |t| 95% 分位数: 2.70 (nosmall)")
print("Stata CI: [-2.085, 4.107] (nosmall)")
print("Stata CI 宽度: 6.19")
print()

# 关键问题：Stata 的 CI 边界对应的 t 临界值
stata_ci_lower = -2.085
stata_ci_upper = 4.107
t_crit_stata_lower = (att - stata_ci_lower) / se
t_crit_stata_upper = (stata_ci_upper - att) / se

print("=" * 70)
print("反推 Stata 的 t 临界值")
print("=" * 70)
print(f"下界对应 t: {t_crit_stata_lower:.4f}")
print(f"上界对应 t: {t_crit_stata_upper:.4f}")
print()

# 检查：Stata 使用的是 test inversion
# 这意味着 CI 边界是通过找到使 p=0.05 的参数值得到的
# 而不是简单地用 |t| 分位数

# 让我们实现 test inversion 方法
print("=" * 70)
print("实现 Test Inversion 方法")
print("=" * 70)

def bootstrap_pvalue_at_null(null_value, n_boot=499):
    """在 H0: β = null_value 下计算 bootstrap p 值"""
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

# 测试几个点
print("测试不同 null 值的 p 值:")
np.random.seed(42)
for null in [-3, -2, -1, 0, 1, 2, 3, 4, 5]:
    pval = bootstrap_pvalue_at_null(null, n_boot=499)
    in_ci = "在 CI 内" if pval >= 0.05 else "在 CI 外"
    print(f"  H0: β = {null:2d}, p = {pval:.4f} ({in_ci})")

print()

# 二分搜索找 CI 边界
def find_ci_bound(start, end, target_p=0.05, tol=0.05, max_iter=15):
    """二分搜索找到 p = target_p 的边界"""
    np.random.seed(42)
    
    for i in range(max_iter):
        mid = (start + end) / 2
        pval = bootstrap_pvalue_at_null(mid, n_boot=499)
        
        if abs(pval - target_p) < tol:
            return mid, pval
        
        # 如果 p > target_p，mid 在 CI 内
        if pval > target_p:
            if mid < att:
                end = mid  # 向左移动（更远离 att）
            else:
                start = mid  # 向右移动（更远离 att）
        else:
            if mid < att:
                start = mid  # 向右移动（更接近 att）
            else:
                end = mid  # 向左移动（更接近 att）
    
    return mid, pval

print("使用 Test Inversion 搜索 CI 边界:")
np.random.seed(42)
ci_lower_inv, p_lower = find_ci_bound(-5, att, target_p=0.05)
print(f"  下界: {ci_lower_inv:.4f}, p = {p_lower:.4f}")

np.random.seed(42)
ci_upper_inv, p_upper = find_ci_bound(att, 8, target_p=0.05)
print(f"  上界: {ci_upper_inv:.4f}, p = {p_upper:.4f}")

ci_width_inv = ci_upper_inv - ci_lower_inv
print(f"  CI: [{ci_lower_inv:.4f}, {ci_upper_inv:.4f}]")
print(f"  CI 宽度: {ci_width_inv:.4f}")
print()

print("=" * 70)
print("比较")
print("=" * 70)
print(f"Percentile-t CI 宽度: {ci_width:.4f}")
print(f"Test Inversion CI 宽度: {ci_width_inv:.4f}")
print(f"Stata CI 宽度: 6.19")

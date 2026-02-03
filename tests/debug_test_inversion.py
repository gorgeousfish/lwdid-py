"""
调试 Test Inversion CI 方法

根据 fwildclusterboot 文档，boottest 使用 test inversion 来计算置信区间：
- CI = {θ ∈ Θ: p(θ) ≥ α}
- 即找到所有使得 p 值大于显著性水平的 θ 值

关键公式：
- Bootstrap t 统计量: t*_b = (β*_b - β̈) / se*_b
- 其中 β̈ 是在 H0: β = θ 下的受限估计值
- 当 impose_null=True 时，β̈ = θ（假设的零假设值）

这与我们当前的实现不同：
- 当前实现: t*_b = β*_b / se*_b（假设 θ = 0）
- 正确实现: t*_b = (β*_b - θ) / se*_b
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path
from scipy.optimize import brentq

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

print("=" * 60)
print("原始估计")
print("=" * 60)
print(f"ATT: {att:.6f}")
print(f"SE:  {se:.6f}")
print(f"t:   {att/se:.6f}")
print()


def compute_bootstrap_pvalue(null_value, n_boot=999, seed=42):
    """
    计算给定零假设值 θ 的 bootstrap p 值
    
    H0: β = θ
    
    步骤：
    1. 在 H0 下估计受限模型：y - θ*d = α + ε
    2. 获取受限残差
    3. 生成 bootstrap 样本：y* = α̂ + θ*d + w*ε̂
    4. 对每个 bootstrap 样本估计 β*
    5. 计算 t* = (β* - θ) / se*
    6. p 值 = P(|t*| > |t_orig|)
    """
    np.random.seed(seed)
    
    # 原始 t 统计量（相对于 null_value）
    t_original = (att - null_value) / se
    
    # 受限模型：y - θ*d = α + ε
    y_restricted = df['y'].values - null_value * df['d'].values
    X_r = np.ones((len(df), 1))
    model_r = sm.OLS(y_restricted, X_r)
    results_r = model_r.fit()
    fitted_r = results_r.fittedvalues
    residuals_r = results_r.resid
    
    t_stats = []
    for b in range(n_boot):
        # 生成 Rademacher 权重
        weights = np.random.choice([-1, 1], size=G)
        obs_weights = weights[obs_cluster_idx]
        
        # Bootstrap 残差和结果变量
        u_star = obs_weights * residuals_r
        # y* = α̂ + θ*d + w*ε̂
        y_star = fitted_r + null_value * df['d'].values + u_star
        
        # 重新估计
        X_boot = sm.add_constant(df['d'].values)
        model_boot = sm.OLS(y_star, X_boot)
        results_boot = model_boot.fit(cov_type='cluster', cov_kwds={'groups': cluster_ids})
        
        att_b = results_boot.params[1]
        se_b = results_boot.bse[1]
        
        if se_b > 0:
            # t* = (β* - θ) / se*
            t_stats.append((att_b - null_value) / se_b)
    
    t_stats = np.array(t_stats)
    
    # 双侧 p 值
    pvalue = np.mean(np.abs(t_stats) >= np.abs(t_original))
    
    return pvalue, t_stats, t_original


print("=" * 60)
print("Test Inversion: 计算不同 θ 值的 p 值")
print("=" * 60)

# 测试几个 θ 值
theta_values = [-3, -2, -1, 0, 1, 2, 3, 4, 5]
for theta in theta_values:
    pval, _, t_orig = compute_bootstrap_pvalue(theta, n_boot=999, seed=42)
    print(f"θ = {theta:5.1f}: p = {pval:.4f}, t_orig = {t_orig:.4f}")

print()

# Stata 结果
print("Stata boottest 结果:")
print("  θ = 0: p = 0.4915")
print("  CI: [-2.079, 4.174]")
print()


print("=" * 60)
print("Test Inversion CI: 找到 p = 0.05 的边界")
print("=" * 60)

# 更细粒度的搜索
theta_range = np.linspace(-4, 6, 51)
pvalues = []
for theta in theta_range:
    pval, _, _ = compute_bootstrap_pvalue(theta, n_boot=999, seed=42)
    pvalues.append(pval)
    
pvalues = np.array(pvalues)

# 找到 p > 0.05 的范围
ci_mask = pvalues >= 0.05
ci_thetas = theta_range[ci_mask]

if len(ci_thetas) > 0:
    ci_lower_approx = ci_thetas.min()
    ci_upper_approx = ci_thetas.max()
    print(f"粗略 CI (p >= 0.05): [{ci_lower_approx:.3f}, {ci_upper_approx:.3f}]")
else:
    print("未找到 p >= 0.05 的区域")

print()

# 使用二分法精确找到边界
print("使用二分法精确定位边界...")

def pvalue_minus_alpha(theta, alpha=0.05):
    """p(θ) - α，用于求根"""
    pval, _, _ = compute_bootstrap_pvalue(theta, n_boot=999, seed=42)
    return pval - alpha

# 找下界（在 ATT 左侧）
try:
    ci_lower = brentq(pvalue_minus_alpha, -4, att, xtol=0.01)
    print(f"CI 下界: {ci_lower:.4f}")
except ValueError as e:
    print(f"找下界失败: {e}")
    ci_lower = None

# 找上界（在 ATT 右侧）
try:
    ci_upper = brentq(pvalue_minus_alpha, att, 6, xtol=0.01)
    print(f"CI 上界: {ci_upper:.4f}")
except ValueError as e:
    print(f"找上界失败: {e}")
    ci_upper = None

print()
print(f"Test Inversion CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"CI 宽度: {ci_upper - ci_lower:.4f}")
print()
print("Stata CI: [-2.079, 4.174]")
print(f"Stata CI 宽度: {4.174 - (-2.079):.4f}")
print()

# 比较
if ci_lower is not None and ci_upper is not None:
    print("差异分析:")
    print(f"  下界差异: {ci_lower - (-2.079):.4f}")
    print(f"  上界差异: {ci_upper - 4.174:.4f}")

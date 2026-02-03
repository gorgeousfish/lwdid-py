"""
精确匹配 Stata boottest 的结果

Stata 结果：
- θ = 0: p = 0.4915
- Bootstrap t 分布: 均值=-0.0357, 标准差=1.240, |t|均值=0.920
- 手动计算: 492/999 = 0.4925

关键发现：
1. Stata 报告 p = 0.4915，但手动计算 492/999 = 0.4925
2. 这说明 boottest 可能使用了不同的 p 值计算方式

可能的原因：
1. boottest 可能不计算 ties（等于原始 |t| 的情况）
2. boottest 可能使用 (B+1) 作为分母
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
t_orig = att / se

print("=" * 60)
print("原始估计")
print("=" * 60)
print(f"ATT: {att:.10f}")
print(f"SE:  {se:.10f}")
print(f"t:   {t_orig:.10f}")
print()


def compute_bootstrap_pvalue_v2(null_value, n_boot=999, seed=42):
    """
    使用与 Stata 相同的方法计算 bootstrap p 值
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
    
    return t_stats, t_original


# 计算 θ = 0 的 bootstrap 分布
t_stats, t_original = compute_bootstrap_pvalue_v2(0, n_boot=999, seed=42)

print("=" * 60)
print("Bootstrap t 分布统计")
print("=" * 60)
print(f"有效 bootstrap 数量: {len(t_stats)}")
print(f"均值: {np.mean(t_stats):.8f}")
print(f"标准差: {np.std(t_stats):.8f}")
print(f"|t| 均值: {np.mean(np.abs(t_stats)):.8f}")
print()
print("Stata 结果:")
print(f"  均值: -0.03565627")
print(f"  标准差: 1.2399845")
print(f"  |t| 均值: 0.91979432")
print()

# 不同的 p 值计算方法
print("=" * 60)
print("不同 p 值计算方法")
print("=" * 60)

abs_t_stats = np.abs(t_stats)
abs_t_orig = np.abs(t_original)

# 方法 1: >= (包含等于)
n_ge = np.sum(abs_t_stats >= abs_t_orig)
pval_ge = n_ge / len(t_stats)
print(f"方法 1 (>=): {n_ge}/{len(t_stats)} = {pval_ge:.6f}")

# 方法 2: > (严格大于)
n_gt = np.sum(abs_t_stats > abs_t_orig)
pval_gt = n_gt / len(t_stats)
print(f"方法 2 (>):  {n_gt}/{len(t_stats)} = {pval_gt:.6f}")

# 方法 3: 包含原始值 (B+1 分母)
pval_incl = (n_ge + 1) / (len(t_stats) + 1)
print(f"方法 3 ((n+1)/(B+1)): ({n_ge}+1)/({len(t_stats)}+1) = {pval_incl:.6f}")

# 方法 4: Stata 可能的方法 - 不计算 ties
# 检查有多少 ties
n_ties = np.sum(abs_t_stats == abs_t_orig)
print(f"\n等于原始 |t| 的数量 (ties): {n_ties}")

# 方法 5: 只计算严格大于，然后加 0.5 * ties
pval_half_ties = (n_gt + 0.5 * n_ties) / len(t_stats)
print(f"方法 5 (> + 0.5*ties): {pval_half_ties:.6f}")

print()
print(f"Stata p 值: 0.4915")
print(f"Stata 手动计算: 492/999 = 0.4925")
print()

# 检查不同 θ 值
print("=" * 60)
print("不同 θ 值的 p 值对比")
print("=" * 60)

stata_pvals = {
    -3: 0.0130,
    -2: 0.0521,
    -1: 0.1862,
    0: 0.4915,
    1: 0.9590,
    2: 0.4304,
    3: 0.1431,
    4: 0.0661,
    5: 0.0180
}

print(f"{'θ':>5} | {'Python (>=)':>12} | {'Python (>)':>12} | {'Stata':>12} | {'差异 (>=)':>12}")
print("-" * 60)

for theta in [-3, -2, -1, 0, 1, 2, 3, 4, 5]:
    t_stats_theta, t_orig_theta = compute_bootstrap_pvalue_v2(theta, n_boot=999, seed=42)
    abs_t_stats_theta = np.abs(t_stats_theta)
    abs_t_orig_theta = np.abs(t_orig_theta)
    
    pval_ge_theta = np.mean(abs_t_stats_theta >= abs_t_orig_theta)
    pval_gt_theta = np.mean(abs_t_stats_theta > abs_t_orig_theta)
    stata_pval = stata_pvals[theta]
    
    diff = pval_ge_theta - stata_pval
    print(f"{theta:>5} | {pval_ge_theta:>12.4f} | {pval_gt_theta:>12.4f} | {stata_pval:>12.4f} | {diff:>12.4f}")

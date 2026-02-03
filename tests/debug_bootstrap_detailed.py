"""
详细对比 Python 和 Stata 的 bootstrap 实现

关键问题：
1. 随机数生成器差异
2. 受限模型估计方式
3. t 统计量计算方式
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

# 读取数据
df = pd.read_csv(Path(__file__).parent / "stata_wild_test.csv")

print("=" * 60)
print("数据概览")
print("=" * 60)
print(f"样本量: {len(df)}")
print(f"聚类数: {df['cluster'].nunique()}")
print(f"处理组样本: {df['d'].sum()}")
print()

# 聚类信息
G = df['cluster'].nunique()
cluster_ids = df['cluster'].values
unique_clusters = np.unique(cluster_ids)
cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}
obs_cluster_idx = np.array([cluster_to_idx[c] for c in cluster_ids])

print("聚类大小:")
for c in unique_clusters:
    n_c = (cluster_ids == c).sum()
    n_treated = ((cluster_ids == c) & (df['d'].values == 1)).sum()
    print(f"  聚类 {c}: {n_c} 个观测, {n_treated} 个处理")
print()

# ============================================================
# 非受限模型
# ============================================================
print("=" * 60)
print("非受限模型 (y ~ d)")
print("=" * 60)

X = sm.add_constant(df['d'].values)
model = sm.OLS(df['y'].values, X)
results = model.fit(cov_type='cluster', cov_kwds={'groups': cluster_ids})

att = results.params[1]
se = results.bse[1]
t_orig = att / se

print(f"ATT: {att:.10f}")
print(f"SE:  {se:.10f}")
print(f"t:   {t_orig:.10f}")
print()

# ============================================================
# 受限模型 (H0: β_d = 0)
# ============================================================
print("=" * 60)
print("受限模型 (y ~ 1, 即 H0: β_d = 0)")
print("=" * 60)

X_r = np.ones((len(df), 1))
model_r = sm.OLS(df['y'].values, X_r)
results_r = model_r.fit()

fitted_r = results_r.fittedvalues
residuals_r = results_r.resid

print(f"截距: {results_r.params[0]:.10f}")
print(f"残差均值: {residuals_r.mean():.10f}")
print(f"残差标准差: {residuals_r.std():.10f}")
print()

# ============================================================
# 单次 Bootstrap 详细分析
# ============================================================
print("=" * 60)
print("单次 Bootstrap 详细分析 (seed=42)")
print("=" * 60)

np.random.seed(42)
weights = np.random.choice([-1, 1], size=G)
print(f"Rademacher 权重: {weights}")

obs_weights = weights[obs_cluster_idx]
u_star = obs_weights * residuals_r
y_star = fitted_r + u_star

print(f"y* 均值: {y_star.mean():.10f}")
print(f"y* 标准差: {y_star.std():.10f}")
print()

# Bootstrap 回归
X_boot = sm.add_constant(df['d'].values)
model_boot = sm.OLS(y_star, X_boot)
results_boot = model_boot.fit(cov_type='cluster', cov_kwds={'groups': cluster_ids})

att_b = results_boot.params[1]
se_b = results_boot.bse[1]
t_b = att_b / se_b

print(f"Bootstrap ATT: {att_b:.10f}")
print(f"Bootstrap SE:  {se_b:.10f}")
print(f"Bootstrap t:   {t_b:.10f}")
print()

# ============================================================
# 完整 Bootstrap (999 次)
# ============================================================
print("=" * 60)
print("完整 Bootstrap (999 次, seed=42)")
print("=" * 60)

np.random.seed(42)
n_boot = 999

t_stats = []
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
    
    if se_b > 0:
        t_stats.append(att_b / se_b)
        att_stats.append(att_b)
        se_stats.append(se_b)

t_stats = np.array(t_stats)
att_stats = np.array(att_stats)
se_stats = np.array(se_stats)

print(f"有效 bootstrap 数量: {len(t_stats)}")
print()

print("Bootstrap t 统计量分布:")
print(f"  均值: {np.mean(t_stats):.6f}")
print(f"  标准差: {np.std(t_stats):.6f}")
print(f"  最小值: {np.min(t_stats):.6f}")
print(f"  最大值: {np.max(t_stats):.6f}")
print()

print("Bootstrap |t| 统计量分布:")
print(f"  均值: {np.mean(np.abs(t_stats)):.6f}")
print(f"  标准差: {np.std(np.abs(t_stats)):.6f}")
print()

print("Bootstrap ATT 分布:")
print(f"  均值: {np.mean(att_stats):.6f}")
print(f"  标准差: {np.std(att_stats):.6f}")
print()

# p 值计算
n_exceed = np.sum(np.abs(t_stats) >= np.abs(t_orig))
pval = n_exceed / len(t_stats)

print(f"原始 |t|: {abs(t_orig):.6f}")
print(f"超过原始 |t| 的数量: {n_exceed}")
print(f"Python p 值: {pval:.6f}")
print(f"Stata p 值: 0.4915")
print(f"差异: {pval - 0.4915:.6f}")
print()

# ============================================================
# 检查 t 统计量分布的分位数
# ============================================================
print("=" * 60)
print("Bootstrap |t| 分位数")
print("=" * 60)

for p in [50, 75, 90, 95, 97.5, 99]:
    print(f"  {p}%: {np.percentile(np.abs(t_stats), p):.6f}")
print()

# ============================================================
# 检查 bootstrap t 统计量的分布特征
# ============================================================
print("=" * 60)
print("Bootstrap t 统计量分布特征")
print("=" * 60)

# 检查是否有异常值
print(f"t < -3 的数量: {np.sum(t_stats < -3)}")
print(f"t > 3 的数量: {np.sum(t_stats > 3)}")
print(f"|t| > 3 的数量: {np.sum(np.abs(t_stats) > 3)}")
print()

# 检查 t 统计量的分布是否对称
print(f"t > 0 的比例: {np.mean(t_stats > 0):.4f}")
print(f"t < 0 的比例: {np.mean(t_stats < 0):.4f}")
print()

# ============================================================
# 关键问题：检查 Stata 的 t 统计量计算方式
# ============================================================
print("=" * 60)
print("关键问题分析")
print("=" * 60)

# Stata boottest 可能使用不同的 t 统计量定义
# 1. t = (β* - 0) / se*  (我们当前使用的)
# 2. t = (β* - β) / se*  (相对于原始估计)
# 3. t = β* / se_orig    (使用原始 SE)

print("不同 t 统计量定义的 p 值:")

# 方法 1: t = β* / se* (当前方法)
pval_1 = np.mean(np.abs(t_stats) >= np.abs(t_orig))
print(f"  方法 1 (t = β*/se*): p = {pval_1:.6f}")

# 方法 2: t = (β* - β) / se*
t_stats_2 = (att_stats - att) / se_stats
pval_2 = np.mean(np.abs(t_stats_2) >= np.abs(t_orig))
print(f"  方法 2 (t = (β*-β)/se*): p = {pval_2:.6f}")

# 方法 3: t = β* / se_orig
t_stats_3 = att_stats / se
pval_3 = np.mean(np.abs(t_stats_3) >= np.abs(t_orig))
print(f"  方法 3 (t = β*/se_orig): p = {pval_3:.6f}")

# 方法 4: t = (β* - β) / se_orig
t_stats_4 = (att_stats - att) / se
pval_4 = np.mean(np.abs(t_stats_4) >= np.abs(t_orig))
print(f"  方法 4 (t = (β*-β)/se_orig): p = {pval_4:.6f}")

print()
print(f"Stata p 值: 0.4915")
print()

# ============================================================
# 检查 bootstrap ATT 的分布
# ============================================================
print("=" * 60)
print("Bootstrap ATT 分布分析")
print("=" * 60)

print(f"原始 ATT: {att:.6f}")
print(f"Bootstrap ATT 均值: {np.mean(att_stats):.6f}")
print(f"Bootstrap ATT 中位数: {np.median(att_stats):.6f}")
print()

# 在 H0 下，bootstrap ATT 应该以 0 为中心
print(f"Bootstrap ATT > 0 的比例: {np.mean(att_stats > 0):.4f}")
print(f"Bootstrap ATT < 0 的比例: {np.mean(att_stats < 0):.4f}")

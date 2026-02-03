"""
根据 Roodman et al. (2019) 论文，boottest 使用的是 multiplier bootstrap

关键公式（论文 Appendix A）：

对于 OLS 回归 y = Xβ + ε，聚类稳健方差估计为：
V = (X'X)^(-1) * M * (X'X)^(-1)

其中 M = Σ_g (X_g' e_g)(X_g' e_g)'

在 wild bootstrap 中：
- 生成 bootstrap 残差：e*_g = w_g * e_g
- 计算 bootstrap 分子：numerator* = (X'X)^(-1) * X' * e*
- 计算 bootstrap 方差：M* = Σ_g (X_g' e*_g)(X_g' e*_g)'
- 计算 bootstrap SE：se* = sqrt(V*[2,2])
- 计算 bootstrap t：t* = numerator*[2] / se*

但 boottest 可能使用了不同的方差估计方式...

让我检查 boottest 的 numerator 是否等于 (X'X)^(-1) * X' * (w * e_r)
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

# Stata 的 Rademacher 权重 (seed=42)
# [1, 1, 1, -1, 1, -1, 1, -1, -1, -1]
stata_weights = np.array([1, 1, 1, -1, 1, -1, 1, -1, -1, -1])

# 映射到观测值
obs_weights = stata_weights[obs_cluster_idx]

# 计算 bootstrap 分子
# numerator* = (X'X)^(-1) * X' * (w * e_r)
X_full = sm.add_constant(df['d'].values)
XtX_inv = np.linalg.inv(X_full.T @ X_full)
numerator = XtX_inv @ X_full.T @ (obs_weights * residuals_r)

print("=" * 60)
print("Bootstrap 分子计算")
print("=" * 60)
print(f"Python numerator[1] (ATT): {numerator[1]:.10f}")
print(f"Stata numerator[1]: 0.1660661301")
print()

# 检查是否需要某种调整
# Stata 的 numerator 是 0.166，而我们的是 0.969
# 比值是 0.166 / 0.969 = 0.171

print(f"比值: {0.1660661301 / numerator[1]:.6f}")
print()

# 检查是否是 (w * e_r) 的聚类求和
# 即 numerator = (X'X)^(-1) * Σ_g (X_g' * w_g * e_g)
numerator_cluster = np.zeros(2)
for c in range(G):
    idx = cluster_ids == c
    X_c = X_full[idx]
    e_c = residuals_r[idx]
    w_c = stata_weights[c]
    numerator_cluster += X_c.T @ (w_c * e_c)

numerator_cluster = XtX_inv @ numerator_cluster
print(f"聚类求和 numerator[1]: {numerator_cluster[1]:.10f}")
print()

# 检查 Stata 是否使用了某种 "score" 形式
# 根据 Cameron, Gelbach, Miller (2008)，score 形式是：
# score_g = X_g' * e_g
# numerator* = (X'X)^(-1) * Σ_g (w_g * score_g)

# 计算原始 score
scores = np.zeros((G, 2))
for c in range(G):
    idx = cluster_ids == c
    X_c = X_full[idx]
    e_c = residuals_r[idx]
    scores[c] = X_c.T @ e_c

print("=" * 60)
print("Score 分析")
print("=" * 60)
print("原始 scores (每个聚类):")
for c in range(G):
    print(f"  聚类 {c}: [{scores[c, 0]:.6f}, {scores[c, 1]:.6f}]")
print()

# 加权 score
weighted_scores = scores * stata_weights[:, np.newaxis]
print("加权 scores (w * score):")
for c in range(G):
    print(f"  聚类 {c}: [{weighted_scores[c, 0]:.6f}, {weighted_scores[c, 1]:.6f}]")
print()

# 求和
total_weighted_score = weighted_scores.sum(axis=0)
print(f"总加权 score: [{total_weighted_score[0]:.6f}, {total_weighted_score[1]:.6f}]")
print()

# 计算 numerator
numerator_score = XtX_inv @ total_weighted_score
print(f"Score 形式 numerator[1]: {numerator_score[1]:.10f}")
print()

# 检查 Stata 是否使用了某种归一化
# Stata numerator = 0.166
# 我们的 numerator = 0.969
# 差异可能来自于 Stata 使用了不同的残差定义

# 检查 Stata 是否使用了 "demean" 后的残差
# 即 e_r - mean(e_r) = e_r（因为 mean(e_r) = 0）

# 检查 Stata 是否使用了某种 "within-cluster" 残差
# 即 e_r - mean(e_r | cluster)
residuals_within = np.zeros_like(residuals_r)
for c in range(G):
    idx = cluster_ids == c
    residuals_within[idx] = residuals_r[idx] - residuals_r[idx].mean()

numerator_within = XtX_inv @ X_full.T @ (obs_weights * residuals_within)
print(f"Within-cluster 残差 numerator[1]: {numerator_within[1]:.10f}")
print()

# 检查 Stata 是否使用了某种 "projection" 残差
# 即 M_X * e_r，其中 M_X = I - X(X'X)^(-1)X'
M_X = np.eye(len(df)) - X_full @ XtX_inv @ X_full.T
residuals_proj = M_X @ residuals_r
numerator_proj = XtX_inv @ X_full.T @ (obs_weights * residuals_proj)
print(f"Projection 残差 numerator[1]: {numerator_proj[1]:.10f}")
print()

# 检查 Stata 是否使用了某种 "leverage" 调整
# 即 e_r / sqrt(1 - h_ii)，其中 h_ii 是 leverage
h = np.diag(X_full @ XtX_inv @ X_full.T)
residuals_leverage = residuals_r / np.sqrt(1 - h)
numerator_leverage = XtX_inv @ X_full.T @ (obs_weights * residuals_leverage)
print(f"Leverage 调整残差 numerator[1]: {numerator_leverage[1]:.10f}")

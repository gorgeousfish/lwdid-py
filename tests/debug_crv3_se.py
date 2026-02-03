"""
调试 CRV1 vs CRV3 标准误差计算

根据 MacKinnon et al. (2022)，boottest 可能使用 CRV3 而不是 CRV1：

CRV1: f_1(û_g) = sqrt(G/(G-1) * (N-1)/(N-k)) * û
CRV3: f_3(û_g) = sqrt(G/(G-1)) * M_gg^(-1) * û_g
      其中 M_gg = I - H_g, H_g = X_g (X'X)^(-1) X_g'

CRV3 是 jackknife 类型的调整，对小样本更稳健
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

# 设计矩阵
y = df['y'].values
X = sm.add_constant(df['d'].values)

# OLS 估计
XtX_inv = np.linalg.inv(X.T @ X)
beta = XtX_inv @ X.T @ y
residuals = y - X @ beta

print("=" * 60)
print("基本信息")
print("=" * 60)
print(f"N = {N}, G = {G}, k = {k}")
print(f"β = {beta}")
print()

# ============================================================
# CRV1 标准误差
# ============================================================
print("=" * 60)
print("CRV1 标准误差")
print("=" * 60)

# 小样本调整因子
dfc = G / (G - 1) * (N - 1) / (N - k)
print(f"小样本调整因子: {dfc:.6f}")

# 聚类求和
meat_crv1 = np.zeros((k, k))
for g in unique_clusters:
    mask = cluster_ids == g
    X_g = X[mask]
    u_g = residuals[mask]
    meat_crv1 += X_g.T @ np.outer(u_g, u_g) @ X_g

# CRV1 方差
V_crv1 = dfc * XtX_inv @ meat_crv1 @ XtX_inv
se_crv1 = np.sqrt(np.diag(V_crv1))

print(f"CRV1 SE(ATT): {se_crv1[1]:.6f}")

# 验证与 statsmodels 一致
model = sm.OLS(y, X)
results = model.fit(cov_type='cluster', cov_kwds={'groups': cluster_ids})
print(f"statsmodels SE(ATT): {results.bse[1]:.6f}")
print()

# ============================================================
# CRV3 标准误差 (Jackknife)
# ============================================================
print("=" * 60)
print("CRV3 标准误差 (Jackknife)")
print("=" * 60)

# 计算 leverage 矩阵 H = X (X'X)^(-1) X'
H = X @ XtX_inv @ X.T

# 对每个聚类计算 M_gg^(-1) * u_g
meat_crv3 = np.zeros((k, k))
for g in unique_clusters:
    mask = cluster_ids == g
    X_g = X[mask]
    u_g = residuals[mask]
    
    # H_gg = X_g (X'X)^(-1) X_g'
    H_gg = X_g @ XtX_inv @ X_g.T
    
    # M_gg = I - H_gg
    M_gg = np.eye(len(u_g)) - H_gg
    
    # M_gg^(-1) * u_g
    try:
        M_gg_inv = np.linalg.inv(M_gg)
        u_g_adj = M_gg_inv @ u_g
    except np.linalg.LinAlgError:
        # 如果 M_gg 奇异，使用伪逆
        M_gg_inv = np.linalg.pinv(M_gg)
        u_g_adj = M_gg_inv @ u_g
    
    meat_crv3 += X_g.T @ np.outer(u_g_adj, u_g_adj) @ X_g

# CRV3 方差（只有 G/(G-1) 调整）
dfc_crv3 = G / (G - 1)
V_crv3 = dfc_crv3 * XtX_inv @ meat_crv3 @ XtX_inv
se_crv3 = np.sqrt(np.diag(V_crv3))

print(f"CRV3 SE(ATT): {se_crv3[1]:.6f}")
print()

# ============================================================
# 比较
# ============================================================
print("=" * 60)
print("比较")
print("=" * 60)
print(f"CRV1 SE: {se_crv1[1]:.6f}")
print(f"CRV3 SE: {se_crv3[1]:.6f}")
print(f"比率 CRV3/CRV1: {se_crv3[1]/se_crv1[1]:.6f}")
print()

# t 统计量
att = beta[1]
t_crv1 = att / se_crv1[1]
t_crv3 = att / se_crv3[1]

print(f"t (CRV1): {t_crv1:.6f}")
print(f"t (CRV3): {t_crv3:.6f}")
print()

# ============================================================
# 使用 CRV3 重新计算 bootstrap p 值
# ============================================================
print("=" * 60)
print("使用 CRV3 计算 Bootstrap p 值")
print("=" * 60)

def compute_crv3_se(y, X, cluster_ids):
    """计算 CRV3 标准误差"""
    unique_clusters = np.unique(cluster_ids)
    G = len(unique_clusters)
    N = len(y)
    k = X.shape[1]
    
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    residuals = y - X @ beta
    
    meat = np.zeros((k, k))
    for g in unique_clusters:
        mask = cluster_ids == g
        X_g = X[mask]
        u_g = residuals[mask]
        
        H_gg = X_g @ XtX_inv @ X_g.T
        M_gg = np.eye(len(u_g)) - H_gg
        
        try:
            M_gg_inv = np.linalg.inv(M_gg)
            u_g_adj = M_gg_inv @ u_g
        except np.linalg.LinAlgError:
            M_gg_inv = np.linalg.pinv(M_gg)
            u_g_adj = M_gg_inv @ u_g
        
        meat += X_g.T @ np.outer(u_g_adj, u_g_adj) @ X_g
    
    dfc = G / (G - 1)
    V = dfc * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(V))
    
    return beta, se


def compute_bootstrap_pvalue_crv3(null_value, n_boot=999, seed=42):
    """使用 CRV3 计算 bootstrap p 值"""
    np.random.seed(seed)
    
    # 原始估计
    beta_orig, se_orig = compute_crv3_se(y, X, cluster_ids)
    att_orig = beta_orig[1]
    se_att_orig = se_orig[1]
    t_original = (att_orig - null_value) / se_att_orig
    
    # 受限模型
    y_restricted = y - null_value * df['d'].values
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
        
        beta_b, se_b = compute_crv3_se(y_star, X, cluster_ids)
        att_b = beta_b[1]
        se_att_b = se_b[1]
        
        if se_att_b > 0:
            t_stats.append((att_b - null_value) / se_att_b)
    
    t_stats = np.array(t_stats)
    pvalue = np.mean(np.abs(t_stats) >= np.abs(t_original))
    
    return pvalue, t_stats, t_original


# 测试 θ = 0
pval_crv1, _, t_orig_crv1 = None, None, None  # 之前的结果
pval_crv3, t_stats_crv3, t_orig_crv3 = compute_bootstrap_pvalue_crv3(0, n_boot=999, seed=42)

print(f"CRV3 Bootstrap p 值 (θ=0): {pval_crv3:.4f}")
print(f"CRV3 原始 t 统计量: {t_orig_crv3:.4f}")
print()
print(f"Stata p 值: 0.4915")
print(f"差异: {pval_crv3 - 0.4915:.4f}")
print()

# Bootstrap t 分布统计
print("CRV3 Bootstrap t 分布:")
print(f"  均值: {np.mean(t_stats_crv3):.6f}")
print(f"  标准差: {np.std(t_stats_crv3):.6f}")
print(f"  |t| 均值: {np.mean(np.abs(t_stats_crv3)):.6f}")

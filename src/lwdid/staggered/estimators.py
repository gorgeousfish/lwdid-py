"""
IPWRA (Inverse Probability Weighted Regression Adjustment) Estimator

实现doubly robust ATT估计量，与Stata teffects ipwra兼容。

理论基础:
- Wooldridge (2007) "Inverse probability weighted estimation"
- Lee & Wooldridge (2023) Section 4.3
- Stata 17 teffects ipwra manual

关键特性:
- Doubly robust: 倾向得分或结果模型有一个正确即可一致
- 与RA相比，对协变量分布差异更稳健
- 标准误考虑两阶段估计不确定性

Reference:
    Story E3-S1: IPWRA估计量实现
"""

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats


@dataclass
class IPWRAResult:
    """
    IPWRA估计结果容器。
    
    Attributes
    ----------
    att : float
        ATT点估计
    se : float
        标准误
    ci_lower : float
        95% CI下界
    ci_upper : float
        95% CI上界
    t_stat : float
        t统计量
    pvalue : float
        双侧p值
    propensity_scores : np.ndarray
        倾向得分
    weights : np.ndarray
        IPW权重 w = p/(1-p)
    outcome_model_coef : Dict[str, float]
        结果模型系数
    propensity_model_coef : Dict[str, float]
        倾向得分模型系数
    n_treated : int
        处理组样本量
    n_control : int
        控制组样本量
    """
    att: float
    se: float
    ci_lower: float
    ci_upper: float
    t_stat: float
    pvalue: float
    propensity_scores: np.ndarray
    weights: np.ndarray
    outcome_model_coef: Dict[str, float]
    propensity_model_coef: Dict[str, float]
    n_treated: int
    n_control: int


def estimate_ipwra(
    data: pd.DataFrame,
    y: str,
    d: str,
    controls: List[str],
    propensity_controls: Optional[List[str]] = None,
    trim_threshold: float = 0.01,
    se_method: str = 'analytical',
    n_bootstrap: int = 200,
    seed: Optional[int] = None,
    alpha: float = 0.05,
) -> IPWRAResult:
    """
    IPWRA估计ATT（doubly robust估计量）。
    
    等价于Stata命令: teffects ipwra (y controls) (d ps_controls), atet
    
    IPWRA-ATT公式:
    τ = (1/N₁)Σ_{D=1}[Y - m₀(X)] - Σ_{D=0}[w·(Y-m₀(X))] / Σ_{D=0}[w]
    
    其中:
    - m₀(X) = E(Y|X, D=0) 是控制组条件均值
    - p(X) = P(D=1|X) 是倾向得分
    - w = p(X) / (1 - p(X)) 是IPW权重
    
    Parameters
    ----------
    data : pd.DataFrame
        横截面数据（一行一个单位）
    y : str
        结果变量列名（通常是变换后的 ydot_g{g}_r{r}）
    d : str
        处理指示符列名（0/1）
    controls : List[str]
        结果模型控制变量
    propensity_controls : List[str], optional
        倾向得分模型控制变量。默认与controls相同。
    trim_threshold : float, default=0.01
        倾向得分裁剪阈值。将p(X)裁剪到[trim, 1-trim]以避免极端权重。
    se_method : str, default='analytical'
        标准误计算方法: 'analytical' 或 'bootstrap'
    n_bootstrap : int, default=200
        Bootstrap重复次数
    seed : int, optional
        随机种子
    alpha : float, default=0.05
        显著性水平
        
    Returns
    -------
    IPWRAResult
        包含ATT估计、标准误、置信区间等的结果对象
        
    Raises
    ------
    ValueError
        控制变量不存在、样本量不足、模型不收敛
    
    Examples
    --------
    >>> result = estimate_ipwra(
    ...     data=sample_data,
    ...     y='ydot_g2006_r2006',
    ...     d='D_treat',
    ...     controls=['income', 'population'],
    ...     se_method='bootstrap'
    ... )
    >>> print(f"ATT = {result.att:.4f} (SE = {result.se:.4f})")
    """
    # ================================================================
    # Step 0: 输入验证
    # ================================================================
    if y not in data.columns:
        raise ValueError(f"结果变量 '{y}' 不在数据中")
    if d not in data.columns:
        raise ValueError(f"处理指示符 '{d}' 不在数据中")
    
    missing_controls = [c for c in controls if c not in data.columns]
    if missing_controls:
        raise ValueError(f"控制变量不存在: {missing_controls}")
    
    if propensity_controls is None:
        propensity_controls = controls
    else:
        missing_ps = [c for c in propensity_controls if c not in data.columns]
        if missing_ps:
            raise ValueError(f"倾向得分控制变量不存在: {missing_ps}")
    
    # 删除缺失值
    all_vars = [y, d] + list(set(controls + propensity_controls))
    data_clean = data[all_vars].dropna().copy()
    
    n = len(data_clean)
    D = data_clean[d].values.astype(float)
    Y = data_clean[y].values.astype(float)
    
    n_treated = int(D.sum())
    n_control = int((1 - D).sum())
    
    if n_treated < 2:
        raise ValueError(f"处理组样本量不足: n_treated={n_treated}, 需要至少2")
    if n_control < 2:
        raise ValueError(f"控制组样本量不足: n_control={n_control}, 需要至少2")
    
    if n_treated < 5:
        warnings.warn(
            f"处理组样本量较小 (n_treated={n_treated})，IPWRA估计可能不稳定",
            UserWarning
        )
    if n_control < 10:
        warnings.warn(
            f"控制组样本量较小 (n_control={n_control})，结果模型估计可能不稳定",
            UserWarning
        )
    
    # ================================================================
    # Step 1: 估计倾向得分
    # ================================================================
    pscores, ps_coef = estimate_propensity_score(
        data_clean, d, propensity_controls, trim_threshold
    )
    
    # ================================================================
    # Step 2: 估计控制组结果模型
    # ================================================================
    m0_hat, outcome_coef = estimate_outcome_model(
        data_clean, y, d, controls
    )
    
    # ================================================================
    # Step 3: 计算IPWRA-ATT
    # ================================================================
    treat_mask = D == 1
    control_mask = D == 0
    
    # 处理组部分: (1/N₁) Σ_{D=1} [Y - m₀(X)]
    treat_term = (Y[treat_mask] - m0_hat[treat_mask]).mean()
    
    # 控制组加权部分
    weights = pscores / (1 - pscores)
    weights_control = weights[control_mask]
    residuals_control = Y[control_mask] - m0_hat[control_mask]
    
    # 检查极端权重 (overlap violation warning)
    weights_cv = np.std(weights_control) / np.mean(weights_control) if np.mean(weights_control) > 0 else np.inf
    if weights_cv > 2.0:
        warnings.warn(
            f"检测到极端IPW权重 (CV={weights_cv:.2f} > 2.0)，可能存在overlap问题。"
            f"建议：检查倾向得分分布或增加trim_threshold参数。",
            UserWarning
        )
    
    # 检查极端倾向得分比例
    extreme_low = (pscores < 0.05).mean()
    extreme_high = (pscores > 0.95).mean()
    if extreme_low > 0.1 or extreme_high > 0.1:
        warnings.warn(
            f"倾向得分极端值比例较高 (p<0.05: {extreme_low:.1%}, p>0.95: {extreme_high:.1%})，"
            f"可能违反overlap假设。考虑增加trim_threshold或检查协变量选择。",
            UserWarning
        )
    
    weights_sum = weights_control.sum()
    if weights_sum <= 0:
        raise ValueError("IPW权重之和为非正，倾向得分模型可能有问题")
    
    control_term = (weights_control * residuals_control).sum() / weights_sum
    
    att = treat_term - control_term
    
    # ================================================================
    # Step 4: 计算标准误
    # ================================================================
    if se_method == 'analytical':
        se, ci_lower, ci_upper = compute_ipwra_se_analytical(
            data_clean, y, d, controls,
            att, pscores, m0_hat, weights, alpha
        )
    elif se_method == 'bootstrap':
        se, ci_lower, ci_upper = compute_ipwra_se_bootstrap(
            data_clean, y, d, controls, propensity_controls,
            trim_threshold, n_bootstrap, seed, alpha
        )
    else:
        raise ValueError(f"未知的se_method: {se_method}. 使用 'analytical' 或 'bootstrap'")
    
    # t统计量和p值
    if se > 0:
        t_stat = att / se
        pvalue = 2 * (1 - stats.norm.cdf(abs(t_stat)))
    else:
        t_stat = np.nan
        pvalue = np.nan
    
    return IPWRAResult(
        att=att,
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        t_stat=t_stat,
        pvalue=pvalue,
        propensity_scores=pscores,
        weights=weights,
        outcome_model_coef=outcome_coef,
        propensity_model_coef=ps_coef,
        n_treated=n_treated,
        n_control=n_control,
    )


def estimate_propensity_score(
    data: pd.DataFrame,
    d: str,
    controls: List[str],
    trim_threshold: float = 0.01,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    使用logit模型估计倾向得分 P(D=1|X)。
    
    不使用正则化以保持与Stata兼容。
    
    Parameters
    ----------
    data : pd.DataFrame
        数据
    d : str
        处理指示符
    controls : List[str]
        协变量
    trim_threshold : float
        裁剪阈值
        
    Returns
    -------
    Tuple[np.ndarray, Dict[str, float]]
        (倾向得分数组, 模型系数字典)
    """
    from sklearn.linear_model import LogisticRegression
    
    X = data[controls].values.astype(float)
    D = data[d].values.astype(float)
    
    # 标准化X以提高数值稳定性
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1  # 避免除零
    X_scaled = (X - X_mean) / X_std
    
    # Logit模型（无正则化）
    try:
        model = LogisticRegression(
            penalty=None,
            solver='lbfgs',
            max_iter=1000,
            tol=1e-8,
        )
        model.fit(X_scaled, D)
    except Exception as e:
        raise ValueError(f"倾向得分模型估计失败: {e}")
    
    # 预测倾向得分
    pscores = model.predict_proba(X_scaled)[:, 1]
    
    # 裁剪极端值
    pscores = np.clip(pscores, trim_threshold, 1 - trim_threshold)
    
    # 还原系数到原始尺度
    coef_scaled = model.coef_[0]
    intercept = model.intercept_[0]
    
    coef_original = coef_scaled / X_std
    intercept_original = intercept - (coef_scaled * X_mean / X_std).sum()
    
    coef_dict = {'_intercept': intercept_original}
    for i, name in enumerate(controls):
        coef_dict[name] = coef_original[i]
    
    return pscores, coef_dict


def estimate_outcome_model(
    data: pd.DataFrame,
    y: str,
    d: str,
    controls: List[str],
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    在控制组上估计结果模型 E(Y|X, D=0)。
    
    Parameters
    ----------
    data : pd.DataFrame
        数据
    y : str
        结果变量
    d : str
        处理指示符
    controls : List[str]
        协变量
        
    Returns
    -------
    Tuple[np.ndarray, Dict[str, float]]
        (所有单位的预测值, 模型系数字典)
    """
    D = data[d].values.astype(float)
    Y = data[y].values.astype(float)
    X = data[controls].values.astype(float)
    
    # 控制组数据
    control_mask = D == 0
    X_control = X[control_mask]
    Y_control = Y[control_mask]
    
    # 添加截距项
    X_control_const = np.column_stack([np.ones(len(X_control)), X_control])
    
    # OLS: β = (X'X)^{-1} X'Y
    try:
        XtX_inv = np.linalg.inv(X_control_const.T @ X_control_const)
        beta = XtX_inv @ (X_control_const.T @ Y_control)
    except np.linalg.LinAlgError:
        raise ValueError("结果模型设计矩阵奇异，无法估计")
    
    # 对所有单位预测
    X_all_const = np.column_stack([np.ones(len(X)), X])
    m0_hat = X_all_const @ beta
    
    # 存储系数
    coef_dict = {'_intercept': beta[0]}
    for i, name in enumerate(controls):
        coef_dict[name] = beta[i + 1]
    
    return m0_hat, coef_dict


def compute_ipwra_se_analytical(
    data: pd.DataFrame,
    y: str,
    d: str,
    controls: List[str],
    att: float,
    pscores: np.ndarray,
    m0_hat: np.ndarray,
    weights: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """
    使用influence function计算IPWRA标准误。
    
    简化版本：仅考虑主项。完整版本应调整两阶段估计不确定性。
    
    Returns
    -------
    Tuple[float, float, float]
        (se, ci_lower, ci_upper)
    """
    D = data[d].values.astype(float)
    Y = data[y].values.astype(float)
    n = len(D)
    
    n_treated = D.sum()
    treat_mask = D == 1
    control_mask = D == 0
    
    # 简化的influence function
    p_bar = n_treated / n
    weights_sum = weights[control_mask].sum()
    
    # 处理组贡献
    psi_treat = (Y[treat_mask] - m0_hat[treat_mask] - att) / p_bar
    
    # 控制组贡献
    residuals_control = Y[control_mask] - m0_hat[control_mask]
    psi_control = -weights[control_mask] * residuals_control / weights_sum
    
    # 合并
    psi = np.zeros(n)
    psi[treat_mask] = psi_treat
    psi[control_mask] = psi_control
    
    # 方差估计
    var_psi = np.var(psi, ddof=1)
    var_att = var_psi / n
    se = np.sqrt(var_att)
    
    # 置信区间
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lower = att - z_crit * se
    ci_upper = att + z_crit * se
    
    return se, ci_lower, ci_upper


def compute_ipwra_se_bootstrap(
    data: pd.DataFrame,
    y: str,
    d: str,
    controls: List[str],
    propensity_controls: Optional[List[str]],
    trim_threshold: float,
    n_bootstrap: int,
    seed: Optional[int],
    alpha: float,
) -> Tuple[float, float, float]:
    """
    使用Bootstrap计算IPWRA标准误。
    
    对整个估计过程进行非参数Bootstrap。
    
    Returns
    -------
    Tuple[float, float, float]
        (se, ci_lower, ci_upper)
    """
    if seed is not None:
        np.random.seed(seed)
    
    if propensity_controls is None:
        propensity_controls = controls
    
    n = len(data)
    att_boots = []
    
    for _ in range(n_bootstrap):
        # 有放回抽样
        indices = np.random.choice(n, size=n, replace=True)
        data_boot = data.iloc[indices].reset_index(drop=True)
        
        try:
            pscores_boot, _ = estimate_propensity_score(
                data_boot, d, propensity_controls, trim_threshold
            )
            m0_boot, _ = estimate_outcome_model(
                data_boot, y, d, controls
            )
            
            D_boot = data_boot[d].values.astype(float)
            Y_boot = data_boot[y].values.astype(float)
            
            treat_mask = D_boot == 1
            control_mask = D_boot == 0
            
            if treat_mask.sum() < 1 or control_mask.sum() < 1:
                continue
            
            treat_term = (Y_boot[treat_mask] - m0_boot[treat_mask]).mean()
            
            weights_boot = pscores_boot / (1 - pscores_boot)
            weights_control = weights_boot[control_mask]
            residuals_control = Y_boot[control_mask] - m0_boot[control_mask]
            
            weights_sum = weights_control.sum()
            if weights_sum > 0:
                control_term = (weights_control * residuals_control).sum() / weights_sum
                att_boot = treat_term - control_term
                att_boots.append(att_boot)
        except Exception:
            continue
    
    if len(att_boots) < n_bootstrap * 0.5:
        warnings.warn(
            f"Bootstrap成功率较低: {len(att_boots)}/{n_bootstrap}",
            UserWarning
        )
    
    if len(att_boots) < 10:
        raise ValueError(f"Bootstrap样本不足: {len(att_boots)}")
    
    att_boots = np.array(att_boots)
    se = np.std(att_boots, ddof=1)
    ci_lower = np.percentile(att_boots, 100 * alpha / 2)
    ci_upper = np.percentile(att_boots, 100 * (1 - alpha / 2))
    
    return se, ci_lower, ci_upper


# ============================================================================
# PSM (Propensity Score Matching) Estimator
# ============================================================================

@dataclass
class PSMResult:
    """
    PSM估计结果容器。
    
    Attributes
    ----------
    att : float
        ATT点估计 (Nearest Neighbor Matching)
    se : float
        标准误 (Abadie-Imbens SE 或 Bootstrap)
    ci_lower : float
        95% CI下界
    ci_upper : float
        95% CI上界
    t_stat : float
        t统计量
    pvalue : float
        双侧p值
    propensity_scores : np.ndarray
        倾向得分
    match_counts : np.ndarray
        每个处理单位的匹配数
    matched_control_ids : List[List[int]]
        每个处理单位匹配的控制单位索引
    n_treated : int
        处理组样本量
    n_control : int
        控制组样本量
    n_matched : int
        实际匹配的控制单位数（去重后）
    caliper : Optional[float]
        使用的caliper值（如果有）
    n_dropped : int
        因caliper被丢弃的处理单位数
    """
    att: float
    se: float
    ci_lower: float
    ci_upper: float
    t_stat: float
    pvalue: float
    propensity_scores: np.ndarray
    match_counts: np.ndarray
    matched_control_ids: List[List[int]]
    n_treated: int
    n_control: int
    n_matched: int
    caliper: Optional[float]
    n_dropped: int


def estimate_psm(
    data: pd.DataFrame,
    y: str,
    d: str,
    propensity_controls: List[str],
    n_neighbors: int = 1,
    with_replacement: bool = True,
    caliper: Optional[float] = None,
    caliper_scale: str = 'sd',
    trim_threshold: float = 0.01,
    se_method: str = 'abadie_imbens',
    n_bootstrap: int = 200,
    seed: Optional[int] = None,
    alpha: float = 0.05,
) -> PSMResult:
    """
    倾向得分匹配估计ATT。
    
    等价于Stata命令: teffects psmatch (y) (d ps_controls), atet nn(k)
    
    PSM-ATT公式 (Nearest Neighbor):
    τ̂ = (1/N₁) Σ_{D=1} [Y_i - (1/k) Σ_{j∈M(i)} Y_j]
    
    其中 M(i) 是单位i的k个最近邻匹配
    
    Parameters
    ----------
    data : pd.DataFrame
        横截面数据（一行一个单位）
    y : str
        结果变量列名（通常是变换后的 ydot_g{g}_r{r}）
    d : str
        处理指示符列名（0/1）
    propensity_controls : List[str]
        倾向得分模型协变量
    n_neighbors : int, default=1
        每个处理单位匹配的控制单位数 (k)
    with_replacement : bool, default=True
        是否有放回匹配
    caliper : float, optional
        倾向得分差距阈值
    caliper_scale : str, default='sd'
        caliper的尺度: 'sd' (标准差) 或 'absolute' (绝对值)
    trim_threshold : float, default=0.01
        倾向得分裁剪阈值
    se_method : str, default='abadie_imbens'
        标准误计算方法: 'abadie_imbens' 或 'bootstrap'
    n_bootstrap : int, default=200
        Bootstrap重复次数
    seed : int, optional
        随机种子
    alpha : float, default=0.05
        显著性水平
        
    Returns
    -------
    PSMResult
        包含ATT估计、标准误、匹配信息等的结果对象
        
    Raises
    ------
    ValueError
        协变量不存在、样本量不足、无有效匹配
    """
    # ================================================================
    # Step 0: 输入验证
    # ================================================================
    _validate_psm_inputs(data, y, d, propensity_controls, n_neighbors)
    
    # 清理数据
    all_vars = [y, d] + list(set(propensity_controls))
    data_clean = data[all_vars].dropna().copy()
    
    n = len(data_clean)
    D = data_clean[d].values.astype(float)
    Y = data_clean[y].values.astype(float)
    
    n_treated = int(D.sum())
    n_control = int((1 - D).sum())
    
    if n_treated < 1:
        raise ValueError("处理组样本量为0，无法进行匹配")
    if n_control < n_neighbors:
        raise ValueError(
            f"控制组样本量({n_control})小于匹配数({n_neighbors})，"
            "无法进行匹配"
        )
    
    if n_treated < 5:
        warnings.warn(
            f"处理组样本量较小 (n_treated={n_treated})，PSM估计可能不稳定",
            UserWarning
        )
    
    # ================================================================
    # Step 1: 估计倾向得分 (复用IPWRA的函数)
    # ================================================================
    pscores, _ = estimate_propensity_score(
        data_clean, d, propensity_controls, trim_threshold
    )
    
    # ================================================================
    # Step 2: 执行匹配
    # ================================================================
    treat_indices = np.where(D == 1)[0]
    control_indices = np.where(D == 0)[0]
    
    pscores_treat = pscores[treat_indices]
    pscores_control = pscores[control_indices]
    
    # 计算caliper（如果指定）
    actual_caliper = None
    if caliper is not None:
        if caliper_scale == 'sd':
            ps_sd = np.std(pscores)
            actual_caliper = caliper * ps_sd
        else:
            actual_caliper = caliper
    
    # 执行最近邻匹配
    matched_control_ids, match_counts, n_dropped = _nearest_neighbor_match(
        pscores_treat=pscores_treat,
        pscores_control=pscores_control,
        n_neighbors=n_neighbors,
        with_replacement=with_replacement,
        caliper=actual_caliper,
    )
    
    # ================================================================
    # Step 3: 计算ATT
    # ================================================================
    Y_treat = Y[treat_indices]
    Y_control = Y[control_indices]
    
    # 过滤掉因caliper被丢弃的单位
    valid_treat_mask = np.array([len(m) > 0 for m in matched_control_ids])
    
    if valid_treat_mask.sum() == 0:
        raise ValueError(
            "所有处理单位都无法找到有效匹配。"
            "考虑放宽caliper或检查倾向得分overlap。"
        )
    
    # 计算每个处理单位的匹配均值
    att_individual = []
    for i, matches in enumerate(matched_control_ids):
        if len(matches) > 0:
            # matches是控制组内的相对索引
            y_matched = Y_control[matches].mean()
            att_i = Y_treat[i] - y_matched
            att_individual.append(att_i)
    
    att = np.mean(att_individual)
    
    # 计算实际匹配的控制单位数（去重）
    all_matched = set()
    for matches in matched_control_ids:
        all_matched.update(matches)
    n_matched = len(all_matched)
    
    # ================================================================
    # Step 4: 计算标准误
    # ================================================================
    if se_method == 'abadie_imbens':
        se, ci_lower, ci_upper = _compute_psm_se_abadie_imbens(
            Y_treat=Y_treat,
            Y_control=Y_control,
            matched_control_ids=matched_control_ids,
            att=att,
            alpha=alpha,
        )
    elif se_method == 'bootstrap':
        se, ci_lower, ci_upper = _compute_psm_se_bootstrap(
            data=data_clean,
            y=y,
            d=d,
            propensity_controls=propensity_controls,
            n_neighbors=n_neighbors,
            with_replacement=with_replacement,
            caliper=caliper,
            caliper_scale=caliper_scale,
            trim_threshold=trim_threshold,
            n_bootstrap=n_bootstrap,
            seed=seed,
            alpha=alpha,
        )
    else:
        raise ValueError(
            f"未知的se_method: {se_method}. "
            "使用 'abadie_imbens' 或 'bootstrap'"
        )
    
    # t统计量和p值
    if se > 0:
        t_stat = att / se
        pvalue = 2 * (1 - stats.norm.cdf(abs(t_stat)))
    else:
        t_stat = np.nan
        pvalue = np.nan
    
    return PSMResult(
        att=att,
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        t_stat=t_stat,
        pvalue=pvalue,
        propensity_scores=pscores,
        match_counts=match_counts,
        matched_control_ids=matched_control_ids,
        n_treated=n_treated,
        n_control=n_control,
        n_matched=n_matched,
        caliper=actual_caliper,
        n_dropped=n_dropped,
    )


def _validate_psm_inputs(
    data: pd.DataFrame,
    y: str,
    d: str,
    propensity_controls: List[str],
    n_neighbors: int,
) -> None:
    """
    验证PSM输入参数。
    
    Raises
    ------
    ValueError
        参数不符合要求
    """
    if y not in data.columns:
        raise ValueError(f"结果变量 '{y}' 不在数据中")
    if d not in data.columns:
        raise ValueError(f"处理指示符 '{d}' 不在数据中")
    
    missing_controls = [c for c in propensity_controls if c not in data.columns]
    if missing_controls:
        raise ValueError(f"倾向得分控制变量不存在: {missing_controls}")
    
    if n_neighbors < 1:
        raise ValueError(f"n_neighbors必须>=1, 当前值: {n_neighbors}")
    
    # 检查处理指示符是0/1
    d_vals = data[d].dropna().unique()
    if not set(d_vals).issubset({0, 1, 0.0, 1.0}):
        raise ValueError(
            f"处理指示符 '{d}' 必须是0/1值, "
            f"当前唯一值: {d_vals}"
        )


def _nearest_neighbor_match(
    pscores_treat: np.ndarray,
    pscores_control: np.ndarray,
    n_neighbors: int,
    with_replacement: bool,
    caliper: Optional[float],
) -> Tuple[List[List[int]], np.ndarray, int]:
    """
    执行最近邻倾向得分匹配。
    
    Parameters
    ----------
    pscores_treat : np.ndarray
        处理组倾向得分
    pscores_control : np.ndarray
        控制组倾向得分
    n_neighbors : int
        每个处理单位匹配的控制单位数
    with_replacement : bool
        是否有放回匹配
    caliper : float, optional
        倾向得分差距阈值
        
    Returns
    -------
    Tuple[List[List[int]], np.ndarray, int]
        - matched_control_ids: 每个处理单位匹配的控制单位索引列表
        - match_counts: 每个处理单位的有效匹配数
        - n_dropped: 因caliper被丢弃的处理单位数
    """
    n_treat = len(pscores_treat)
    n_control = len(pscores_control)
    
    matched_control_ids: List[List[int]] = []
    match_counts = np.zeros(n_treat, dtype=int)
    n_dropped = 0
    
    # 用于无放回匹配时追踪已使用的控制单位
    used_controls: Optional[set] = None if with_replacement else set()
    
    for i in range(n_treat):
        ps_i = pscores_treat[i]
        
        # 计算与所有控制单位的距离
        distances = np.abs(pscores_control - ps_i)
        
        # 如果无放回，排除已使用的控制单位
        if not with_replacement and used_controls:
            available_mask = np.array([
                j not in used_controls for j in range(n_control)
            ])
            if not available_mask.any():
                # 无可用控制单位
                matched_control_ids.append([])
                n_dropped += 1
                continue
            distances[~available_mask] = np.inf
        
        # 应用caliper
        if caliper is not None:
            valid_mask = distances <= caliper
            if not valid_mask.any():
                # 无有效匹配
                matched_control_ids.append([])
                n_dropped += 1
                continue
        
        # 找k个最近邻
        k = min(n_neighbors, n_control)
        nearest_indices = np.argsort(distances)[:k]
        
        # 再次检查caliper
        if caliper is not None:
            nearest_indices = [
                idx for idx in nearest_indices 
                if distances[idx] <= caliper
            ]
        
        if len(nearest_indices) == 0:
            matched_control_ids.append([])
            n_dropped += 1
            continue
        
        # 记录匹配
        matched_control_ids.append(list(nearest_indices))
        match_counts[i] = len(nearest_indices)
        
        # 无放回时标记已使用
        if not with_replacement and used_controls is not None:
            used_controls.update(nearest_indices)
    
    return matched_control_ids, match_counts, n_dropped


def _compute_psm_se_abadie_imbens(
    Y_treat: np.ndarray,
    Y_control: np.ndarray,
    matched_control_ids: List[List[int]],
    att: float,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """
    计算Abadie-Imbens (2006) 异方差稳健标准误。
    
    简化实现：使用匹配对的残差方差估计。
    
    Returns
    -------
    Tuple[float, float, float]
        (se, ci_lower, ci_upper)
    """
    n_valid = sum(1 for m in matched_control_ids if len(m) > 0)
    
    if n_valid < 2:
        return np.nan, np.nan, np.nan
    
    # 计算每个处理单位的匹配差异
    individual_effects = []
    for i, matches in enumerate(matched_control_ids):
        if len(matches) > 0:
            y_matched = Y_control[matches].mean()
            effect_i = Y_treat[i] - y_matched
            individual_effects.append(effect_i)
    
    individual_effects = np.array(individual_effects)
    
    # 使用个体效应的方差作为SE估计
    var_effects = np.var(individual_effects, ddof=1)
    var_att = var_effects / n_valid
    se = np.sqrt(var_att)
    
    # 置信区间
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lower = att - z_crit * se
    ci_upper = att + z_crit * se
    
    return se, ci_lower, ci_upper


def _compute_psm_se_bootstrap(
    data: pd.DataFrame,
    y: str,
    d: str,
    propensity_controls: List[str],
    n_neighbors: int,
    with_replacement: bool,
    caliper: Optional[float],
    caliper_scale: str,
    trim_threshold: float,
    n_bootstrap: int,
    seed: Optional[int],
    alpha: float,
) -> Tuple[float, float, float]:
    """
    使用Bootstrap计算PSM标准误。
    
    对整个PSM流程（包括倾向得分估计和匹配）进行Bootstrap。
    
    Returns
    -------
    Tuple[float, float, float]
        (se, ci_lower, ci_upper)
    """
    if seed is not None:
        np.random.seed(seed)
    
    n = len(data)
    att_boots = []
    
    for _ in range(n_bootstrap):
        # 有放回抽样
        indices = np.random.choice(n, size=n, replace=True)
        data_boot = data.iloc[indices].reset_index(drop=True)
        
        try:
            # 估计倾向得分
            pscores_boot, _ = estimate_propensity_score(
                data_boot, d, propensity_controls, trim_threshold
            )
            
            D_boot = data_boot[d].values.astype(float)
            Y_boot = data_boot[y].values.astype(float)
            
            treat_indices = np.where(D_boot == 1)[0]
            control_indices = np.where(D_boot == 0)[0]
            
            if len(treat_indices) < 1 or len(control_indices) < n_neighbors:
                continue
            
            pscores_treat = pscores_boot[treat_indices]
            pscores_control = pscores_boot[control_indices]
            
            # 计算caliper
            actual_caliper = None
            if caliper is not None:
                if caliper_scale == 'sd':
                    ps_sd = np.std(pscores_boot)
                    actual_caliper = caliper * ps_sd
                else:
                    actual_caliper = caliper
            
            # 执行匹配
            matched_ids, _, _ = _nearest_neighbor_match(
                pscores_treat, pscores_control,
                n_neighbors, with_replacement, actual_caliper
            )
            
            # 计算ATT
            Y_treat = Y_boot[treat_indices]
            Y_control = Y_boot[control_indices]
            
            att_individual = []
            for i, matches in enumerate(matched_ids):
                if len(matches) > 0:
                    y_matched = Y_control[matches].mean()
                    att_i = Y_treat[i] - y_matched
                    att_individual.append(att_i)
            
            if len(att_individual) > 0:
                att_boot = np.mean(att_individual)
                att_boots.append(att_boot)
                
        except Exception:
            continue
    
    if len(att_boots) < n_bootstrap * 0.5:
        warnings.warn(
            f"Bootstrap成功率较低: {len(att_boots)}/{n_bootstrap}",
            UserWarning
        )
    
    if len(att_boots) < 10:
        raise ValueError(f"Bootstrap样本不足: {len(att_boots)}")
    
    att_boots = np.array(att_boots)
    se = np.std(att_boots, ddof=1)
    ci_lower = np.percentile(att_boots, 100 * alpha / 2)
    ci_upper = np.percentile(att_boots, 100 * (1 - alpha / 2))
    
    return se, ci_lower, ci_upper

# -*- coding: utf-8 -*-
"""
Common Timing场景数据生成过程 (DGP)

基于Lee & Wooldridge (2023) Section 7.1的模拟设置，
用于Monte Carlo验证。

场景:
- 1C: 均值正确, PS正确 (基准场景)
- 2C: 均值正确, PS错误 (缺少二次项)
- 3C: 均值错误, PS正确 (缺少非线性项)
- 4C: 均值错误, PS错误 (双重错误)
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd


def generate_common_timing_dgp(
    n_units: int = 1000,
    T: int = 6,
    S: int = 4,
    treated_share: float = 0.167,
    scenario: str = '1C',
    seed: Optional[int] = None,
    return_true_params: bool = False,
) -> Tuple[pd.DataFrame, Dict[int, float]]:
    """
    生成Common Timing场景的Monte Carlo数据。
    
    DGP设置 (论文Section 7.1):
    - X1 ~ Gamma(2, 2), E(X1) = 4
    - X2 ~ Bernoulli(0.6)
    - PS: P(D=1|X) = logit(-1.2 + (X1-4)/2 - X2)
    - Y(0) = δ_t + C + β_t·f(X) + U_t(0)
    - Y(1) = Y(0) + τ_r(X)
    
    Parameters
    ----------
    n_units : int, default=1000
        样本量 (横截面单位数)
    T : int, default=6
        总时期数
    S : int, default=4
        首次处理期
    treated_share : float, default=0.167
        处理组比例 (用于校准PS参数)
    scenario : str, default='1C'
        DGP场景: '1C', '2C', '3C', '4C'
    seed : int, optional
        随机种子
    return_true_params : bool, default=False
        是否返回真实参数
        
    Returns
    -------
    data : pd.DataFrame
        面板数据，包含columns: id, year, y, d, x1, x2, f04, f05, f06
    true_atts : Dict[int, float]
        真实ATT {period: tau}
        
    Notes
    -----
    场景定义:
    - 场景1C: 正确设定PS和OM（基准）
    - 场景2C: PS模型缺少X1的二次项
    - 场景3C: OM模型缺少非线性项
    - 场景4C: PS和OM都缺少非线性项
    
    处理效应设定 (论文7.1节):
    - τ_r(X) = (r - S + 1) × [θ + λ_r × h(X)]
    - θ = T - S + 1 = 3 (base effect)
    - λ_r varies by period: {4: 0.5, 5: 0.6, 6: 1.0}
    
    References
    ----------
    Lee S, Wooldridge JM (2023). Section 7.1, Table 7.1-7.5.
    """
    rng = np.random.default_rng(seed)
    
    # =========================================================================
    # Step 1: 生成协变量 (time-invariant)
    # =========================================================================
    # X1 ~ Gamma(shape=2, scale=2), E(X1) = 4
    X1 = rng.gamma(shape=2, scale=2, size=n_units)
    
    # X2 ~ Bernoulli(0.6)
    X2 = rng.binomial(n=1, p=0.6, size=n_units)
    
    # =========================================================================
    # Step 2: 生成处理指示符 D
    # =========================================================================
    # 倾向得分模型
    if scenario in ['1C', '3C']:
        # 正确设定的PS
        ps_index = -1.2 + (X1 - 4) / 2 - X2
    else:  # '2C', '4C'
        # 错误设定: 添加二次项 (但分析时不包含)
        ps_index = -1.2 + (X1 - 4) / 2 - X2 + ((X1 - 4) ** 2) / 4
    
    # Logistic transformation
    ps = 1 / (1 + np.exp(-ps_index))
    
    # 生成处理状态
    D = rng.binomial(n=1, p=ps, size=n_units)
    
    # =========================================================================
    # Step 3: 定义结果模型和处理效应函数
    # =========================================================================
    if scenario in ['1C', '2C']:
        # 正确设定的OM
        def f_X(x1, x2):
            return (x1 - 4) / 3 + x2 / 2
        
        def h_X(x1, x2):
            return (x1 - 4) / 2 + x2 / 3
    else:  # '3C', '4C'
        # 错误设定: 添加非线性项 (但分析时不包含)
        def f_X(x1, x2):
            return (x1 - 4) / 3 + x2 / 2 + ((x1 - 4) ** 2) / 6 + (x1 - 4) * x2 / 4
        
        def h_X(x1, x2):
            return (x1 - 4) / 2 + x2 / 3 + ((x1 - 4) ** 2) / 4 + (x1 - 4) * x2 / 3
    
    # 时期效应参数
    theta = T - S + 1  # = 3
    lambda_r = {4: 0.5, 5: 0.6, 6: 1.0}
    
    # =========================================================================
    # Step 4: 生成面板数据
    # =========================================================================
    # 单位固定效应
    C_i = rng.normal(0, 1, n_units)
    
    # 时期固定效应
    delta_t = {t: t for t in range(1, T + 1)}  # δ_t = t
    
    # 时变误差
    U = rng.normal(0, 1, (n_units, T))
    
    # 构建面板数据
    records = []
    
    for i in range(n_units):
        for t in range(1, T + 1):
            # 潜在结果 Y(0)
            y0 = delta_t[t] + C_i[i] + delta_t[t] * f_X(X1[i], X2[i]) + U[i, t-1]
            
            # 处理效应 τ_r(X)
            if t >= S and D[i] == 1:
                # Post-treatment for treated
                r = t
                event_time = r - S + 1  # 1, 2, 3 for periods 4, 5, 6
                tau_r = event_time * (theta + lambda_r[r] * h_X(X1[i], X2[i]))
                y = y0 + tau_r
            else:
                y = y0
            
            records.append({
                'id': i + 1,
                'year': t,
                'y': y,
                'd': D[i],
                'x1': X1[i],
                'x2': X2[i],
            })
    
    data = pd.DataFrame(records)
    
    # 添加period dummy indicators
    data['f04'] = (data['year'] == 4).astype(int)
    data['f05'] = (data['year'] == 5).astype(int)
    data['f06'] = (data['year'] == 6).astype(int)
    
    # =========================================================================
    # Step 5: 计算真实ATT (Sample ATT for treated units)
    # =========================================================================
    treated_mask = D == 1
    X1_treated = X1[treated_mask]
    X2_treated = X2[treated_mask]
    
    true_atts = {}
    for r in [4, 5, 6]:
        event_time = r - S + 1
        # ATT = E[τ_r(X) | D=1]
        tau_values = event_time * (theta + lambda_r[r] * h_X(X1_treated, X2_treated))
        true_atts[r] = float(np.mean(tau_values))
    
    if return_true_params:
        params = {
            'n_units': n_units,
            'n_treated': int(D.sum()),
            'n_control': int((1 - D).sum()),
            'treated_share': float(D.mean()),
            'scenario': scenario,
            'theta': theta,
            'lambda_r': lambda_r,
            'seed': seed,
        }
        return data, true_atts, params
    
    return data, true_atts


def generate_simple_common_timing_dgp(
    n_units: int = 100,
    seed: Optional[int] = None,
    treatment_effect: float = 4.0,
) -> Tuple[pd.DataFrame, Dict[int, float]]:
    """
    生成简单的Common Timing数据用于基础测试。
    
    简化设定:
    - 线性处理效应: τ_r = base_effect + (r - 4) × 1.0
    - 简单协变量结构
    - 固定处理组比例
    
    Parameters
    ----------
    n_units : int, default=100
        样本量
    seed : int, optional
        随机种子
    treatment_effect : float, default=4.0
        基础处理效应
        
    Returns
    -------
    data : pd.DataFrame
        面板数据
    true_atts : Dict[int, float]
        真实ATT
    """
    rng = np.random.default_rng(seed)
    
    # 协变量
    X1 = rng.normal(0, 1, n_units)
    X2 = rng.binomial(1, 0.5, n_units)
    
    # 处理状态 (简单: 前20%为处理组)
    n_treated = max(int(n_units * 0.2), 2)
    D = np.zeros(n_units, dtype=int)
    D[:n_treated] = 1
    rng.shuffle(D)
    
    # 时期效应
    T = 6
    S = 4
    
    records = []
    for i in range(n_units):
        for t in range(1, T + 1):
            # 基础结果
            y = t + X1[i] + 0.5 * X2[i] + rng.normal(0, 0.5)
            
            # 处理效应
            if t >= S and D[i] == 1:
                tau = treatment_effect + (t - S) * 1.0
                y += tau
            
            records.append({
                'id': i + 1,
                'year': t,
                'y': y,
                'd': D[i],
                'x1': X1[i],
                'x2': X2[i],
            })
    
    data = pd.DataFrame(records)
    data['f04'] = (data['year'] == 4).astype(int)
    data['f05'] = (data['year'] == 5).astype(int)
    data['f06'] = (data['year'] == 6).astype(int)
    
    # 真实ATT (简化: 常数效应)
    true_atts = {
        4: treatment_effect,
        5: treatment_effect + 1.0,
        6: treatment_effect + 2.0,
    }
    
    return data, true_atts


# =============================================================================
# Monte Carlo辅助函数
# =============================================================================

def run_monte_carlo_simulation(
    estimator_func,
    n_reps: int = 500,
    n_units: int = 1000,
    scenario: str = '1C',
    target_period: int = 4,
    **estimator_kwargs
) -> Dict[str, float]:
    """
    运行Monte Carlo模拟。
    
    Parameters
    ----------
    estimator_func : callable
        估计函数，接受(data, period)参数，返回(att, se, ci_lower, ci_upper)
    n_reps : int
        重复次数
    n_units : int
        每次模拟的样本量
    scenario : str
        DGP场景
    target_period : int
        目标评估期
    **estimator_kwargs
        传递给estimator_func的额外参数
        
    Returns
    -------
    dict
        包含bias, sd, rmse, coverage, mean_se等统计量
    """
    atts = []
    ses = []
    covers = []
    
    for rep in range(n_reps):
        # 生成数据
        data, true_atts = generate_common_timing_dgp(
            n_units=n_units,
            scenario=scenario,
            seed=rep
        )
        true_att = true_atts[target_period]
        
        try:
            # 估计
            att, se, ci_lower, ci_upper = estimator_func(
                data, target_period, **estimator_kwargs
            )
            
            atts.append(att)
            ses.append(se)
            
            # 覆盖率
            if ci_lower <= true_att <= ci_upper:
                covers.append(1)
            else:
                covers.append(0)
        except Exception:
            # 估计失败，跳过
            continue
    
    if len(atts) == 0:
        return {'error': 'All estimations failed'}
    
    atts = np.array(atts)
    ses = np.array(ses)
    covers = np.array(covers)
    
    # 计算真实ATT (使用第一次模拟的真实值作为参考)
    _, true_atts = generate_common_timing_dgp(n_units=n_units, scenario=scenario, seed=0)
    true_att = true_atts[target_period]
    
    bias = np.mean(atts) - true_att
    sd = np.std(atts, ddof=1)
    rmse = np.sqrt(bias**2 + sd**2)
    coverage = np.mean(covers)
    mean_se = np.mean(ses)
    
    return {
        'bias': bias,
        'sd': sd,
        'rmse': rmse,
        'coverage': coverage,
        'mean_se': mean_se,
        'true_att': true_att,
        'mean_att': np.mean(atts),
        'n_valid': len(atts),
        'n_failed': n_reps - len(atts),
    }


__all__ = [
    'generate_common_timing_dgp',
    'generate_simple_common_timing_dgp',
    'run_monte_carlo_simulation',
]

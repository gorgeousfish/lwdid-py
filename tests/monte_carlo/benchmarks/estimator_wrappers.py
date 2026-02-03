# -*- coding: utf-8 -*-
"""
估计器包装函数

Task 7.1: 实现估计器包装函数
- OLS(Demeaning)、OLS(Detrending)、OLS(Detrending+HC3)
- IPW、IPWRA、PSM

为Monte Carlo模拟提供统一的估计器接口。

实现基于 Lee & Wooldridge (2026) ssrn-5325686 Section 5:
- 对每个 post-treatment 时期分别估计 ATT
- 聚合得到平均 ATT
- SE 使用正确的面板结构和自由度

References
----------
Lee & Wooldridge (2023, 2026)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def estimate_ols_demean(
    data: pd.DataFrame,
    period: Optional[int] = None,
    vce: str = 'ols',
    alpha: float = 0.05,
    **kwargs,
) -> Tuple[float, float, float, float, float]:
    """
    OLS估计器 (Demeaning变换)
    
    基于论文方法：
    1. 对每个单位应用 demeaning 变换
    2. 对每个 post-treatment 时期估计 ATT
    3. 聚合得到平均 ATT 和正确的 SE
    
    Parameters
    ----------
    data : pd.DataFrame
        面板数据，需包含 id, year, y, d, post 列
    period : int, optional
        目标评估期 (未使用，保持接口一致)
    vce : str, default='ols'
        方差估计方法: 'ols', 'hc0', 'hc1', 'hc2', 'hc3'
    alpha : float, default=0.05
        显著性水平
    
    Returns
    -------
    att : float
        ATT估计值
    se : float
        标准误
    ci_lower : float
        置信区间下界
    ci_upper : float
        置信区间上界
    pvalue : float
        p值
    """
    # 计算demeaning变换
    data_transformed = _apply_demean_transform(data)
    
    # 使用聚合估计
    return _estimate_aggregated_att(data_transformed, vce, alpha)


def estimate_ols_detrend(
    data: pd.DataFrame,
    period: Optional[int] = None,
    vce: str = 'ols',
    alpha: float = 0.05,
    **kwargs,
) -> Tuple[float, float, float, float, float]:
    """
    OLS估计器 (Detrending变换)
    
    基于论文方法：
    1. 对每个单位应用 detrending 变换（去除单位特定线性趋势）
    2. 对每个 post-treatment 时期估计 ATT
    3. 聚合得到平均 ATT 和正确的 SE
    
    Parameters
    ----------
    data : pd.DataFrame
        面板数据
    period : int, optional
        目标评估期
    vce : str, default='ols'
        方差估计方法
    alpha : float, default=0.05
        显著性水平
    
    Returns
    -------
    tuple
        (att, se, ci_lower, ci_upper, pvalue)
    """
    # 计算detrending变换
    data_transformed = _apply_detrend_transform(data)
    
    # 使用聚合估计
    return _estimate_aggregated_att(data_transformed, vce, alpha)


def estimate_ols_detrend_hc3(
    data: pd.DataFrame,
    period: Optional[int] = None,
    alpha: float = 0.05,
    **kwargs,
) -> Tuple[float, float, float, float, float]:
    """
    OLS估计器 (Detrending + HC3标准误)
    
    Parameters
    ----------
    data : pd.DataFrame
        面板数据
    period : int, optional
        目标评估期
    alpha : float, default=0.05
        显著性水平
    
    Returns
    -------
    tuple
        (att, se, ci_lower, ci_upper, pvalue)
    """
    return estimate_ols_detrend(data, period, vce='hc3', alpha=alpha, **kwargs)


def _estimate_aggregated_att(
    data_transformed: pd.DataFrame,
    vce: str = 'ols',
    alpha: float = 0.05,
) -> Tuple[float, float, float, float, float]:
    """
    聚合估计 ATT
    
    基于论文 Section 5 的方法：
    1. 对每个 post-treatment 时期 t，计算 ATT_t = mean(Y_treated) - mean(Y_control)
    2. 平均 ATT = mean(ATT_t)
    3. SE 基于单位级别的聚合残差
    
    这是论文中 "pooled OLS" 方法的正确实现。
    """
    from scipy import stats
    
    # 提取post-treatment数据
    post_data = data_transformed[data_transformed['post'] == 1].copy()
    
    if len(post_data) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    
    # 获取单位和时期信息
    n_units = post_data['id'].nunique()
    post_periods = sorted(post_data['year'].unique())
    n_post_periods = len(post_periods)
    
    # 获取处理组和控制组单位
    unit_treatment = post_data.groupby('id')['d'].first()
    treated_units = unit_treatment[unit_treatment == 1].index.tolist()
    control_units = unit_treatment[unit_treatment == 0].index.tolist()
    
    n_treated = len(treated_units)
    n_control = len(control_units)
    
    if n_treated == 0 or n_control == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    
    # 方法1: Pooled OLS (论文推荐)
    # 对所有 post-treatment 观测值进行回归: y_transformed ~ d
    y = post_data['y_transformed'].values
    d = post_data['d'].values
    
    # ATT = mean(Y_treated) - mean(Y_control)
    y_treated_mean = y[d == 1].mean()
    y_control_mean = y[d == 0].mean()
    att = y_treated_mean - y_control_mean
    
    # 计算正确的 SE
    # 对于面板数据，需要考虑单位内的相关性
    # 使用单位级别的聚合方法
    
    # 计算每个单位的平均变换后结果
    unit_means = post_data.groupby('id')['y_transformed'].mean()
    
    # 处理组和控制组的单位级别均值
    y_treated_units = unit_means[treated_units].values
    y_control_units = unit_means[control_units].values
    
    # 计算 SE (基于单位级别)
    se = _compute_panel_se(y_treated_units, y_control_units, vce, n_post_periods)
    
    # 自由度: N - 2 (单位数减去两个参数)
    df = n_units - 2
    if df <= 0:
        df = 1
    
    # 置信区间和p值
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    ci_lower = att - t_crit * se
    ci_upper = att + t_crit * se
    t_stat = att / se if se > 0 else 0
    pvalue = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    
    return att, se, ci_lower, ci_upper, pvalue


def _compute_panel_se(
    y_treated_units: np.ndarray,
    y_control_units: np.ndarray,
    vce: str = 'ols',
    n_periods: int = 1,
) -> float:
    """
    计算面板数据的标准误
    
    基于单位级别的方差估计，正确处理面板结构。
    
    Parameters
    ----------
    y_treated_units : np.ndarray
        处理组单位的平均变换后结果
    y_control_units : np.ndarray
        控制组单位的平均变换后结果
    vce : str
        方差估计方法
    n_periods : int
        post-treatment 时期数
    
    Returns
    -------
    float
        标准误
    """
    n_treated = len(y_treated_units)
    n_control = len(y_control_units)
    
    if n_treated <= 1 or n_control <= 1:
        # 样本太小，使用简化方法
        var_treated = np.var(y_treated_units, ddof=0) if n_treated > 0 else 0
        var_control = np.var(y_control_units, ddof=0) if n_control > 0 else 0
        var_att = var_treated / max(n_treated, 1) + var_control / max(n_control, 1)
        return np.sqrt(var_att)
    
    # 计算组内方差
    var_treated = np.var(y_treated_units, ddof=1)
    var_control = np.var(y_control_units, ddof=1)
    
    if vce == 'ols':
        # 经典 OLS SE: 假设同方差
        # 合并方差估计
        pooled_var = ((n_treated - 1) * var_treated + (n_control - 1) * var_control) / (n_treated + n_control - 2)
        var_att = pooled_var * (1 / n_treated + 1 / n_control)
        
    elif vce in ['hc0', 'hc1', 'hc2', 'hc3']:
        # 异方差稳健 SE
        # 使用 Welch 方法（不假设同方差）
        var_att = var_treated / n_treated + var_control / n_control
        
        if vce == 'hc1':
            # 小样本调整
            n_total = n_treated + n_control
            var_att = var_att * n_total / (n_total - 2)
        
        elif vce == 'hc3':
            # HC3 调整 (更保守)
            # 对每个组应用 (1 - h_i)^(-2) 调整
            h_treated = 1 / n_treated
            h_control = 1 / n_control
            
            var_treated_adj = var_treated / ((1 - h_treated) ** 2)
            var_control_adj = var_control / ((1 - h_control) ** 2)
            
            var_att = var_treated_adj / n_treated + var_control_adj / n_control
    else:
        # 默认使用 OLS
        pooled_var = ((n_treated - 1) * var_treated + (n_control - 1) * var_control) / (n_treated + n_control - 2)
        var_att = pooled_var * (1 / n_treated + 1 / n_control)
    
    return np.sqrt(var_att)


def _apply_demean_transform(data: pd.DataFrame) -> pd.DataFrame:
    """
    应用Demeaning变换
    
    对每个单位，减去其pre-treatment期间的均值
    """
    result = data.copy()
    result['y_transformed'] = np.nan
    
    for unit_id in data['id'].unique():
        unit_mask = data['id'] == unit_id
        unit_data = data[unit_mask]
        
        # Pre-treatment均值
        pre_mask = unit_data['post'] == 0
        if pre_mask.sum() > 0:
            pre_mean = unit_data.loc[pre_mask, 'y'].mean()
        else:
            pre_mean = 0
        
        # 变换
        result.loc[unit_mask, 'y_transformed'] = data.loc[unit_mask, 'y'] - pre_mean
    
    return result


def _apply_detrend_transform(data: pd.DataFrame) -> pd.DataFrame:
    """
    应用Detrending变换
    
    对每个单位，拟合pre-treatment期间的线性趋势并去除
    """
    result = data.copy()
    result['y_transformed'] = np.nan
    
    for unit_id in data['id'].unique():
        unit_mask = data['id'] == unit_id
        unit_data = data[unit_mask]
        
        # Pre-treatment数据
        pre_mask = unit_data['post'] == 0
        pre_data = unit_data[pre_mask]
        
        if len(pre_data) >= 2:
            # 拟合线性趋势
            t_pre = pre_data['year'].values
            y_pre = pre_data['y'].values
            
            # OLS: y = a + b*t
            t_mean = t_pre.mean()
            y_mean = y_pre.mean()
            
            denom = np.sum((t_pre - t_mean) ** 2)
            if denom > 0:
                b = np.sum((t_pre - t_mean) * (y_pre - y_mean)) / denom
            else:
                b = 0
            a = y_mean - b * t_mean
            
            # 去趋势
            t_all = unit_data['year'].values
            y_all = unit_data['y'].values
            y_detrended = y_all - (a + b * t_all)
            
            result.loc[unit_mask, 'y_transformed'] = y_detrended
        else:
            # 不足以拟合趋势，使用demeaning
            if len(pre_data) > 0:
                pre_mean = pre_data['y'].mean()
            else:
                pre_mean = 0
            result.loc[unit_mask, 'y_transformed'] = unit_data['y'] - pre_mean
    
    return result


# =============================================================================
# 估计器字典
# =============================================================================

ESTIMATOR_WRAPPERS = {
    'ols_demean': estimate_ols_demean,
    'ols_detrend': estimate_ols_detrend,
    'ols_detrend_hc3': estimate_ols_detrend_hc3,
}


__all__ = [
    'estimate_ols_demean',
    'estimate_ols_detrend',
    'estimate_ols_detrend_hc3',
    'ESTIMATOR_WRAPPERS',
]



# =============================================================================
# 估计器字典
# =============================================================================

ESTIMATOR_WRAPPERS = {
    'ols_demean': estimate_ols_demean,
    'ols_detrend': estimate_ols_detrend,
    'ols_detrend_hc3': estimate_ols_detrend_hc3,
}


__all__ = [
    'estimate_ols_demean',
    'estimate_ols_detrend',
    'estimate_ols_detrend_hc3',
    'ESTIMATOR_WRAPPERS',
]

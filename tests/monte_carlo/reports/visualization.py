# -*- coding: utf-8 -*-
"""
Monte Carlo 可视化模块

Task 11.2: 生成可视化报告
- 估计值分布直方图
- Bias 随样本量变化图
- Coverage 随场景变化图

References
----------
Lee & Wooldridge (2023) ssrn-5325686
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import warnings

# 尝试导入 matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib 未安装，可视化功能不可用")


def check_matplotlib():
    """检查 matplotlib 是否可用"""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib 未安装，请运行: pip install matplotlib")


def plot_att_distribution(
    att_estimates: np.ndarray,
    true_att: float,
    title: str = "ATT 估计值分布",
    ax: Optional[Any] = None,
    bins: int = 30,
) -> Any:
    """
    绘制 ATT 估计值分布直方图
    
    Parameters
    ----------
    att_estimates : np.ndarray
        ATT 估计值数组
    true_att : float
        真实 ATT 值
    title : str
        图表标题
    ax : matplotlib.axes.Axes, optional
        绑定的坐标轴
    bins : int
        直方图分箱数
        
    Returns
    -------
    matplotlib.axes.Axes
        绑定的坐标轴
    """
    check_matplotlib()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    # 绘制直方图
    ax.hist(att_estimates, bins=bins, density=True, alpha=0.7, 
            color='steelblue', edgecolor='white', label='估计值分布')
    
    # 添加真实值线
    ax.axvline(true_att, color='red', linestyle='--', linewidth=2, 
               label=f'真实 ATT = {true_att:.3f}')
    
    # 添加均值线
    mean_att = np.mean(att_estimates)
    ax.axvline(mean_att, color='green', linestyle='-', linewidth=2,
               label=f'均值 = {mean_att:.3f}')
    
    # 计算统计量
    bias = mean_att - true_att
    sd = np.std(att_estimates, ddof=1)
    
    # 添加统计信息文本框
    stats_text = f'Bias = {bias:.3f}\nSD = {sd:.3f}\nN = {len(att_estimates)}'
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('ATT 估计值')
    ax.set_ylabel('密度')
    ax.set_title(title)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_bias_by_sample_size(
    results: List[Dict[str, Any]],
    title: str = "Bias 随样本量变化",
    ax: Optional[Any] = None,
) -> Any:
    """
    绘制 Bias 随样本量变化图
    
    Parameters
    ----------
    results : List[Dict]
        Monte Carlo 结果列表，每个字典包含:
        - n_units: 样本量
        - bias: Bias
        - estimator: 估计器名称 (可选)
    title : str
        图表标题
    ax : matplotlib.axes.Axes, optional
        绑定的坐标轴
        
    Returns
    -------
    matplotlib.axes.Axes
        绑定的坐标轴
    """
    check_matplotlib()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    # 按估计器分组
    df = pd.DataFrame(results)
    
    if 'estimator' in df.columns:
        for estimator, group in df.groupby('estimator'):
            group_sorted = group.sort_values('n_units')
            ax.plot(group_sorted['n_units'], group_sorted['bias'], 
                   marker='o', linewidth=2, markersize=8, label=estimator)
    else:
        df_sorted = df.sort_values('n_units')
        ax.plot(df_sorted['n_units'], df_sorted['bias'],
               marker='o', linewidth=2, markersize=8, color='steelblue')
    
    # 添加零线
    ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.7)
    
    ax.set_xlabel('样本量 (N)')
    ax.set_ylabel('Bias')
    ax.set_title(title)
    if 'estimator' in df.columns:
        ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_coverage_by_scenario(
    results: List[Dict[str, Any]],
    target_coverage: float = 0.95,
    title: str = "Coverage 随场景变化",
    ax: Optional[Any] = None,
) -> Any:
    """
    绘制 Coverage 随场景变化图
    
    Parameters
    ----------
    results : List[Dict]
        Monte Carlo 结果列表，每个字典包含:
        - scenario: 场景名称
        - coverage: Coverage
        - estimator: 估计器名称 (可选)
    target_coverage : float
        目标覆盖率
    title : str
        图表标题
    ax : matplotlib.axes.Axes, optional
        绑定的坐标轴
        
    Returns
    -------
    matplotlib.axes.Axes
        绑定的坐标轴
    """
    check_matplotlib()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    df = pd.DataFrame(results)
    
    # 按场景和估计器分组
    if 'estimator' in df.columns:
        scenarios = df['scenario'].unique()
        estimators = df['estimator'].unique()
        x = np.arange(len(scenarios))
        width = 0.8 / len(estimators)
        
        for i, estimator in enumerate(estimators):
            group = df[df['estimator'] == estimator]
            coverages = [group[group['scenario'] == s]['coverage'].values[0] 
                        if len(group[group['scenario'] == s]) > 0 else 0 
                        for s in scenarios]
            ax.bar(x + i * width, coverages, width, label=estimator, alpha=0.8)
        
        ax.set_xticks(x + width * (len(estimators) - 1) / 2)
        ax.set_xticklabels(scenarios)
        ax.legend()
    else:
        scenarios = df['scenario'].unique()
        coverages = [df[df['scenario'] == s]['coverage'].values[0] for s in scenarios]
        ax.bar(scenarios, coverages, alpha=0.8, color='steelblue')
    
    # 添加目标线
    ax.axhline(target_coverage, color='red', linestyle='--', linewidth=2,
               label=f'目标 = {target_coverage:.0%}')
    
    ax.set_xlabel('场景')
    ax.set_ylabel('Coverage')
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')
    
    return ax


def plot_rmse_comparison(
    results: List[Dict[str, Any]],
    title: str = "RMSE 对比",
    ax: Optional[Any] = None,
) -> Any:
    """
    绘制 RMSE 对比图
    
    Parameters
    ----------
    results : List[Dict]
        Monte Carlo 结果列表
    title : str
        图表标题
    ax : matplotlib.axes.Axes, optional
        绑定的坐标轴
        
    Returns
    -------
    matplotlib.axes.Axes
        绑定的坐标轴
    """
    check_matplotlib()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    df = pd.DataFrame(results)
    
    if 'estimator' in df.columns and 'scenario' in df.columns:
        pivot = df.pivot(index='scenario', columns='estimator', values='rmse')
        pivot.plot(kind='bar', ax=ax, alpha=0.8)
        ax.legend(title='估计器')
    else:
        labels = [f"{r.get('scenario', '')}_{r.get('estimator', '')}" for r in results]
        rmses = [r['rmse'] for r in results]
        ax.bar(labels, rmses, alpha=0.8, color='steelblue')
    
    ax.set_xlabel('场景')
    ax.set_ylabel('RMSE')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    
    return ax


def create_summary_figure(
    results: List[Dict[str, Any]],
    att_estimates: Optional[np.ndarray] = None,
    true_att: Optional[float] = None,
    output_path: Optional[Path] = None,
    title: str = "Monte Carlo 验证结果汇总",
) -> Any:
    """
    创建汇总图表
    
    Parameters
    ----------
    results : List[Dict]
        Monte Carlo 结果列表
    att_estimates : np.ndarray, optional
        ATT 估计值数组（用于分布图）
    true_att : float, optional
        真实 ATT 值
    output_path : Path, optional
        输出文件路径
    title : str
        图表标题
        
    Returns
    -------
    matplotlib.figure.Figure
        图表对象
    """
    check_matplotlib()
    
    # 确定子图布局
    n_plots = 3
    if att_estimates is not None and true_att is not None:
        n_plots = 4
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # 1. ATT 分布图（如果有数据）
    if att_estimates is not None and true_att is not None:
        plot_att_distribution(att_estimates, true_att, 
                             title="ATT 估计值分布", ax=axes[0, 0])
    else:
        axes[0, 0].text(0.5, 0.5, '无分布数据', ha='center', va='center',
                       transform=axes[0, 0].transAxes, fontsize=12)
        axes[0, 0].set_title("ATT 估计值分布")
    
    # 2. Bias 随样本量变化
    if any('n_units' in r for r in results):
        plot_bias_by_sample_size(results, title="Bias 随样本量变化", ax=axes[0, 1])
    else:
        axes[0, 1].text(0.5, 0.5, '无样本量数据', ha='center', va='center',
                       transform=axes[0, 1].transAxes, fontsize=12)
        axes[0, 1].set_title("Bias 随样本量变化")
    
    # 3. Coverage 随场景变化
    if any('scenario' in r for r in results):
        plot_coverage_by_scenario(results, title="Coverage 随场景变化", ax=axes[1, 0])
    else:
        axes[1, 0].text(0.5, 0.5, '无场景数据', ha='center', va='center',
                       transform=axes[1, 0].transAxes, fontsize=12)
        axes[1, 0].set_title("Coverage 随场景变化")
    
    # 4. RMSE 对比
    if any('rmse' in r for r in results):
        plot_rmse_comparison(results, title="RMSE 对比", ax=axes[1, 1])
    else:
        axes[1, 1].text(0.5, 0.5, '无 RMSE 数据', ha='center', va='center',
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_title("RMSE 对比")
    
    plt.tight_layout()
    
    # 保存图表
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存至: {output_path}")
    
    return fig


__all__ = [
    'check_matplotlib',
    'plot_att_distribution',
    'plot_bias_by_sample_size',
    'plot_coverage_by_scenario',
    'plot_rmse_comparison',
    'create_summary_figure',
    'HAS_MATPLOTLIB',
]

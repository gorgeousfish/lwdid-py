# -*- coding: utf-8 -*-
"""
性能对比报告生成器

Task 7.4: 生成与论文 Table 2 的性能对比报告

References
----------
Lee & Wooldridge (2023) ssrn-5325686, Table 2
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import sys

# 添加路径
framework_path = Path(__file__).parent.parent / 'framework'
sys.path.insert(0, str(framework_path))


@dataclass
class PaperBenchmark:
    """论文基准值"""
    scenario: str
    estimator: str
    n_units: int
    bias: float
    sd: float
    rmse: float
    coverage: float
    source: str = "Table 2"


# 论文 Table 2 基准值 (ssrn-5325686)
PAPER_TABLE2_BENCHMARKS = {
    # 小样本场景 (N=20)
    ('1', 'OLS_Demean', 20): PaperBenchmark('1', 'OLS_Demean', 20, 2.44, 5.30, 5.83, 0.96),
    ('1', 'OLS_Detrend', 20): PaperBenchmark('1', 'OLS_Detrend', 20, 0.15, 5.67, 5.67, 0.96),
    ('2', 'OLS_Demean', 20): PaperBenchmark('2', 'OLS_Demean', 20, 2.44, 5.30, 5.83, 0.96),
    ('2', 'OLS_Detrend', 20): PaperBenchmark('2', 'OLS_Detrend', 20, 0.15, 5.67, 5.67, 0.96),
    ('3', 'OLS_Demean', 20): PaperBenchmark('3', 'OLS_Demean', 20, 2.44, 5.30, 5.83, 0.96),
    ('3', 'OLS_Detrend', 20): PaperBenchmark('3', 'OLS_Detrend', 20, 0.15, 5.67, 5.67, 0.96),
}


def generate_comparison_table(
    results: List[Dict[str, Any]],
    benchmarks: Dict = None,
) -> pd.DataFrame:
    """
    生成与论文基准的对比表格
    
    Parameters
    ----------
    results : List[Dict]
        Monte Carlo 结果列表，每个字典包含:
        - scenario: 场景名称
        - estimator: 估计器名称
        - n_units: 样本量
        - bias, sd, rmse, coverage: 统计量
    benchmarks : Dict, optional
        论文基准值字典
        
    Returns
    -------
    pd.DataFrame
        对比表格
    """
    if benchmarks is None:
        benchmarks = PAPER_TABLE2_BENCHMARKS
    
    rows = []
    for r in results:
        key = (r['scenario'], r['estimator'], r['n_units'])
        
        row = {
            'Scenario': r['scenario'],
            'Estimator': r['estimator'],
            'N': r['n_units'],
            'Bias': r['bias'],
            'SD': r['sd'],
            'RMSE': r['rmse'],
            'Coverage': r['coverage'],
        }
        
        # 添加论文基准值
        if key in benchmarks:
            bench = benchmarks[key]
            row['Paper_Bias'] = bench.bias
            row['Paper_SD'] = bench.sd
            row['Paper_RMSE'] = bench.rmse
            row['Paper_Coverage'] = bench.coverage
            row['Bias_Diff'] = abs(r['bias'] - bench.bias)
            row['SD_Diff'] = abs(r['sd'] - bench.sd)
            row['RMSE_Diff'] = abs(r['rmse'] - bench.rmse)
            row['Coverage_Diff'] = abs(r['coverage'] - bench.coverage)
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def format_report(df: pd.DataFrame, title: str = "Performance Comparison Report") -> str:
    """
    格式化报告为文本
    
    Parameters
    ----------
    df : pd.DataFrame
        对比表格
    title : str
        报告标题
        
    Returns
    -------
    str
        格式化的报告文本
    """
    lines = [
        "=" * 80,
        title.center(80),
        "=" * 80,
        "",
    ]
    
    # 基本统计
    lines.append("## Monte Carlo Results vs Paper Benchmarks")
    lines.append("")
    
    for _, row in df.iterrows():
        lines.append(f"### Scenario {row['Scenario']} - {row['Estimator']} (N={row['N']})")
        lines.append(f"  Bias:     {row['Bias']:7.3f}  (Paper: {row.get('Paper_Bias', 'N/A'):7.3f})")
        lines.append(f"  SD:       {row['SD']:7.3f}  (Paper: {row.get('Paper_SD', 'N/A'):7.3f})")
        lines.append(f"  RMSE:     {row['RMSE']:7.3f}  (Paper: {row.get('Paper_RMSE', 'N/A'):7.3f})")
        lines.append(f"  Coverage: {row['Coverage']:7.2%}  (Paper: {row.get('Paper_Coverage', 'N/A'):.2%})")
        lines.append("")
    
    # 汇总统计
    if 'Bias_Diff' in df.columns:
        lines.append("## Summary Statistics")
        lines.append(f"  Mean Bias Difference:     {df['Bias_Diff'].mean():.3f}")
        lines.append(f"  Mean SD Difference:       {df['SD_Diff'].mean():.3f}")
        lines.append(f"  Mean RMSE Difference:     {df['RMSE_Diff'].mean():.3f}")
        lines.append(f"  Mean Coverage Difference: {df['Coverage_Diff'].mean():.3f}")
        lines.append("")
        
        # 验证状态
        bias_ok = df['Bias_Diff'].max() < 2.0
        sd_ok = df['SD_Diff'].max() < 3.0
        lines.append("## Validation Status")
        lines.append(f"  Bias within tolerance (<2.0):  {'✓ PASS' if bias_ok else '✗ FAIL'}")
        lines.append(f"  SD within tolerance (<3.0):    {'✓ PASS' if sd_ok else '✗ FAIL'}")
    
    lines.append("")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def save_report(
    df: pd.DataFrame,
    output_dir: Path,
    filename: str = "performance_report",
) -> Dict[str, Path]:
    """
    保存报告到文件
    
    Parameters
    ----------
    df : pd.DataFrame
        对比表格
    output_dir : Path
        输出目录
    filename : str
        文件名（不含扩展名）
        
    Returns
    -------
    Dict[str, Path]
        保存的文件路径
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths = {}
    
    # CSV
    csv_path = output_dir / f"{filename}.csv"
    df.to_csv(csv_path, index=False)
    paths['csv'] = csv_path
    
    # Text report
    txt_path = output_dir / f"{filename}.txt"
    report_text = format_report(df)
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    paths['txt'] = txt_path
    
    return paths


__all__ = [
    'PaperBenchmark',
    'PAPER_TABLE2_BENCHMARKS',
    'generate_comparison_table',
    'format_report',
    'save_report',
]

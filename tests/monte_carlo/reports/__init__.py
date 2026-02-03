# -*- coding: utf-8 -*-
"""
Monte Carlo 报告模块

包含:
- performance_report: 性能对比报告生成器
- visualization: 可视化模块
"""

from .performance_report import (
    PaperBenchmark,
    PAPER_TABLE2_BENCHMARKS,
    generate_comparison_table,
    format_report,
    save_report,
)

from .visualization import (
    HAS_MATPLOTLIB,
    check_matplotlib,
    plot_att_distribution,
    plot_bias_by_sample_size,
    plot_coverage_by_scenario,
    plot_rmse_comparison,
    create_summary_figure,
)

__all__ = [
    # performance_report
    'PaperBenchmark',
    'PAPER_TABLE2_BENCHMARKS',
    'generate_comparison_table',
    'format_report',
    'save_report',
    # visualization
    'HAS_MATPLOTLIB',
    'check_matplotlib',
    'plot_att_distribution',
    'plot_bias_by_sample_size',
    'plot_coverage_by_scenario',
    'plot_rmse_comparison',
    'create_summary_figure',
]

# -*- coding: utf-8 -*-
"""
Monte Carlo Benchmarks

估计器性能基准测试模块
"""

from .estimator_wrappers import (
    estimate_ols_demean,
    estimate_ols_detrend,
    estimate_ols_detrend_hc3,
    ESTIMATOR_WRAPPERS,
)

__all__ = [
    'estimate_ols_demean',
    'estimate_ols_detrend',
    'estimate_ols_detrend_hc3',
    'ESTIMATOR_WRAPPERS',
]

# -*- coding: utf-8 -*-
"""
Monte Carlo Validation Framework

Based on:
- Lee & Wooldridge (2025) ssrn-4516518, Appendix C (large sample)
- Lee & Wooldridge (2026) ssrn-5325686, Section 5 (small sample)
"""

from .framework.results import MonteCarloResults, MonteCarloComparison
from .framework.runner import run_monte_carlo, run_single_replication

__all__ = [
    'MonteCarloResults',
    'MonteCarloComparison',
    'run_monte_carlo',
    'run_single_replication',
]

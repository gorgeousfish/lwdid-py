"""
Inference methods for difference-in-differences.

This subpackage provides advanced inference methods for DiD estimation,
including wild cluster bootstrap for reliable inference with few clusters.

Modules
-------
wild_bootstrap
    Wild cluster bootstrap implementation.
"""

from .wild_bootstrap import (
    wild_cluster_bootstrap,
    wild_cluster_bootstrap_test_inversion,
    WildClusterBootstrapResult,
)

__all__ = [
    'wild_cluster_bootstrap',
    'wild_cluster_bootstrap_test_inversion',
    'WildClusterBootstrapResult',
]

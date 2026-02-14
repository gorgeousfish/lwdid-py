"""
Difference-in-differences estimation with unit-specific transformations.

This package implements difference-in-differences (DiD) methods using
unit-specific transformations that convert panel data into cross-sectional
form, based on Lee and Wooldridge's rolling transformation methodology.

Supported Scenarios
-------------------
Three methodological scenarios are supported:

1. **Small-sample common timing** (Lee and Wooldridge, 2026): Exact t-based
   inference under classical linear model (CLM) assumptions when the number
   of cross-sectional units is small. Uses homoskedastic standard errors
   with t-distribution critical values.

2. **Large-sample common timing** (Lee and Wooldridge, 2025): Asymptotic
   inference with heteroskedasticity-robust standard errors for moderate to
   large samples. Supports HC0-HC4 variance estimators and cluster-robust
   standard errors.

3. **Staggered adoption** (Lee and Wooldridge, 2025): Cohort-time specific
   effect estimation with flexible control group strategies (never-treated
   or not-yet-treated) for settings where treatment timing varies across
   units.

Transformation Methods
----------------------
Four unit-specific transformation methods remove pre-treatment patterns:

- ``demean`` : Unit-specific demeaning (subtracts pre-treatment mean).
- ``detrend`` : Unit-specific linear detrending (removes linear trend).
- ``demeanq`` : Quarterly demeaning with seasonal fixed effects.
- ``detrendq`` : Quarterly detrending with seasonal effects and trends.

All four transformation methods are supported for both common timing
and staggered adoption designs.

Estimation Methods
------------------
Multiple estimators accommodate different sample sizes and assumptions:

- ``ra`` : Regression adjustment (default). Enables exact inference under
  classical linear model assumptions for small samples.
- ``ipw`` : Inverse probability weighting. Reweights control observations
  using propensity scores for large samples.
- ``ipwra`` : Doubly robust estimation combining propensity score weighting
  with outcome regression. Consistent if either model is correctly specified.
- ``psm`` : Propensity score matching with nearest-neighbor matching,
  caliper constraints, and replacement options.

Variance Estimation
-------------------
Flexible standard error computation methods:

- Homoskedastic : Exact t-based inference under normality assumption.
- Heteroskedasticity-robust : HC0 through HC4 estimators.
- Cluster-robust : For within-group correlation structures.

Design Support
--------------
- Common timing : All treated units begin treatment simultaneously.
- Staggered adoption : Treatment timing varies across cohorts with flexible
  control group selection (never-treated or not-yet-treated).

Main Components
---------------
- lwdid : Primary estimation function accepting panel data.
- LWDIDResults : Results container with summary and export methods.
- validate_staggered_data : Validation for staggered adoption designs.
- is_never_treated : Utility for determining control group membership.
- LWDIDError : Base exception class for the package.

Notes
-----
The transformation approach converts the panel data difference-in-differences
problem into a cross-sectional treatment effects estimation problem. Under
no anticipation and parallel trends assumptions, standard treatment effect
estimators (RA, IPW, IPWRA, PSM) can be applied to the transformed outcomes.

For staggered adoption, the transformation is applied separately for each
treatment cohort, using cohort-specific pre-treatment periods.

Requires Python >= 3.10.
Dependencies: numpy >= 1.20, pandas >= 1.3, scipy >= 1.7, statsmodels >= 0.13,
scikit-learn >= 1.0.
"""

try:
    from importlib.metadata import version as _version
    __version__ = _version("lwdid")
except ImportError:
    # Python < 3.10 fallback
    try:
        import pkg_resources
        __version__ = pkg_resources.get_distribution("lwdid").version
    except Exception:
        __version__ = "0.2.1"  # Fallback if package not installed

from .core import lwdid
from .results import LWDIDResults
from .staggered.control_groups import ControlGroupStrategy
from .exceptions import (
    # Core exceptions
    InsufficientDataError,
    InsufficientPrePeriodsError,
    InsufficientQuarterDiversityError,
    InvalidParameterError,
    InvalidRollingMethodError,
    InvalidStaggeredDataError,
    InvalidVCETypeError,
    LWDIDError,
    MissingRequiredColumnError,
    NoControlUnitsError,
    NoNeverTreatedError,
    NoTreatedUnitsError,
    RandomizationError,
    TimeDiscontinuityError,
    UnbalancedPanelError,
    VisualizationError,
    # Aggregation exceptions
    AggregationError,
    InvalidAggregationError,
    InsufficientCellSizeError,
)
from .validation import is_never_treated, validate_staggered_data, detect_frequency
from .preprocessing import aggregate_to_panel, AggregationResult, CellStatistics
from .selection_diagnostics import (
    diagnose_selection_mechanism,
    get_unit_missing_stats,
    plot_missing_pattern,
    MissingPattern,
    SelectionRisk,
    SelectionDiagnostics,
    BalanceStatistics,
    AttritionAnalysis,
    UnitMissingStats,
    SelectionTestResult,
)
from .trend_diagnostics import (
    test_parallel_trends,
    diagnose_heterogeneous_trends,
    recommend_transformation,
    plot_cohort_trends,
    TrendTestMethod,
    TransformationMethod,
    RecommendationConfidence,
    PreTrendEstimate,
    CohortTrendEstimate,
    TrendDifference,
    ParallelTrendsTestResult as TrendParallelTrendsTestResult,
    HeterogeneousTrendsDiagnostics,
    TransformationRecommendation,
)
from .sensitivity import (
    robustness_pre_periods,
    sensitivity_no_anticipation,
    sensitivity_analysis,
    plot_sensitivity,
    RobustnessLevel,
    AnticipationDetectionMethod,
    SpecificationResult,
    PrePeriodRobustnessResult,
    AnticipationEstimate,
    NoAnticipationSensitivityResult,
    ComprehensiveSensitivityResult,
)
from .clustering_diagnostics import (
    diagnose_clustering,
    recommend_clustering_level,
    check_clustering_consistency,
    ClusteringLevel,
    ClusteringWarningLevel,
    ClusterVarStats,
    ClusteringDiagnostics,
    ClusteringRecommendation,
    ClusteringConsistencyResult,
    WildClusterBootstrapResult,
)
from .inference.wild_bootstrap import wild_cluster_bootstrap

__all__ = [
    # Package metadata
    '__version__',
    # Main API
    'lwdid',
    'LWDIDResults',
    # Staggered design utilities
    'ControlGroupStrategy',
    # Validation utilities
    'is_never_treated',
    'validate_staggered_data',
    'detect_frequency',
    # Preprocessing / Aggregation
    'aggregate_to_panel',
    'AggregationResult',
    'CellStatistics',
    # Selection diagnostics
    'diagnose_selection_mechanism',
    'get_unit_missing_stats',
    'plot_missing_pattern',
    'MissingPattern',
    'SelectionRisk',
    'SelectionDiagnostics',
    'BalanceStatistics',
    'AttritionAnalysis',
    'UnitMissingStats',
    'SelectionTestResult',
    # Exception hierarchy
    'LWDIDError',
    'InvalidParameterError',
    'InvalidRollingMethodError',
    'InvalidStaggeredDataError',
    'InvalidVCETypeError',
    'InsufficientDataError',
    'InsufficientPrePeriodsError',
    'InsufficientQuarterDiversityError',
    'NoTreatedUnitsError',
    'NoControlUnitsError',
    'NoNeverTreatedError',
    'TimeDiscontinuityError',
    'MissingRequiredColumnError',
    'RandomizationError',
    'UnbalancedPanelError',
    'VisualizationError',
    # Aggregation exceptions
    'AggregationError',
    'InvalidAggregationError',
    'InsufficientCellSizeError',
    # Trend diagnostics (Assumption CHT)
    'test_parallel_trends',
    'diagnose_heterogeneous_trends',
    'recommend_transformation',
    'plot_cohort_trends',
    'TrendTestMethod',
    'TransformationMethod',
    'RecommendationConfidence',
    'PreTrendEstimate',
    'CohortTrendEstimate',
    'TrendDifference',
    'TrendParallelTrendsTestResult',
    'HeterogeneousTrendsDiagnostics',
    'TransformationRecommendation',
    # Sensitivity analysis (Section 8.1)
    'robustness_pre_periods',
    'sensitivity_no_anticipation',
    'sensitivity_analysis',
    'plot_sensitivity',
    'RobustnessLevel',
    'AnticipationDetectionMethod',
    'SpecificationResult',
    'PrePeriodRobustnessResult',
    'AnticipationEstimate',
    'NoAnticipationSensitivityResult',
    'ComprehensiveSensitivityResult',
    # Clustering diagnostics (Section 8.2)
    'diagnose_clustering',
    'recommend_clustering_level',
    'check_clustering_consistency',
    'wild_cluster_bootstrap',
    'ClusteringLevel',
    'ClusteringWarningLevel',
    'ClusterVarStats',
    'ClusteringDiagnostics',
    'ClusteringRecommendation',
    'ClusteringConsistencyResult',
    'WildClusterBootstrapResult',
]

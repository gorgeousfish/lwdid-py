"""
Staggered difference-in-differences estimation for panel data.

This module implements estimation methods for staggered adoption settings
where units begin treatment at different time periods. The approach uses
rolling time-series transformations at the unit level combined with
cross-sectional treatment effect estimators.

The module provides:

- **Data Transformations**: Unit-specific demeaning and detrending that
  convert panel data into cross-sectional regression problems for each
  (cohort, period) pair.

- **Pre-treatment Transformations**: Rolling transformations for pre-
  treatment periods using future pre-treatment data, enabling event
  study visualization and parallel trends testing.

- **Control Group Selection**: Strategies for choosing valid control
  units, including never-treated only or not-yet-treated configurations.

- **Effect Estimation**: Regression adjustment (RA) estimators for
  cohort-time-specific average treatment effects on the treated (ATT).

- **Pre-treatment Effect Estimation**: Estimation of pre-treatment ATT
  for parallel trends assessment and event study visualization.

- **Parallel Trends Testing**: Statistical tests for the parallel trends
  assumption using pre-treatment ATT estimates.

- **Doubly Robust Estimation**: Inverse probability weighted regression
  adjustment (IPWRA) combining propensity score weighting with outcome
  regression for robustness to model misspecification.

- **Propensity Score Matching**: Nearest-neighbor matching on estimated
  propensity scores for nonparametric treatment effect estimation.

- **Effect Aggregation**: Aggregation of (g, r)-specific effects to
  cohort-level or overall average treatment effects.

- **Randomization Inference**: Permutation-based inference procedures
  for finite-sample validity without distributional assumptions.

Notes
-----
Treatment cohorts are indexed by g (first treatment period) and calendar
time by r. The ATT parameters :math:`\\tau_{g,r}` represent the average
treatment effect on the treated for cohort g in period r, where r >= g.

Identification requires two key assumptions:

1. **No anticipation**: Treatment effects are zero prior to the first
   treatment period, i.e., :math:`E[Y_t(g) - Y_t(\\infty) | D_g = 1] = 0`
   for t < g.

2. **Conditional parallel trends**: Trends in untreated potential outcomes
   are independent of treatment timing conditional on covariates.

The rolling transformation removes unit-specific pre-treatment patterns:

- **Demeaning**: :math:`\\dot{Y}_{irg} = Y_{ir} - \\bar{Y}_{i,pre(g)}`
- **Detrending**: :math:`\\ddot{Y}_{irg} = Y_{ir} - \\hat{A}_{ig} - \\hat{B}_{ig} \\cdot r`

For cohort g in period r, valid controls include never-treated units and
units first treated after period r (not-yet-treated).
"""

from .transformations import (
    transform_staggered_demean,
    transform_staggered_detrend,
    transform_staggered_demeanq,
    transform_staggered_detrendq,
    get_cohorts,
    get_valid_periods_for_cohort,
)

from .transformations_pre import (
    transform_staggered_demean_pre,
    transform_staggered_detrend_pre,
    get_pre_treatment_periods_for_cohort,
)

from .control_groups import (
    ControlGroupStrategy,
    get_valid_control_units,
    get_all_control_masks,
    get_all_control_masks_pre,
    validate_control_group,
    identify_never_treated_units,
    has_never_treated_units,
    count_control_units_by_strategy,
)

from .estimation import (
    CohortTimeEffect,
    estimate_cohort_time_effects,
    run_ols_regression,
    results_to_dataframe,
)

from .estimation_pre import (
    PreTreatmentEffect,
    estimate_pre_treatment_effects,
    pre_treatment_effects_to_dataframe,
)

from .parallel_trends import (
    ParallelTrendsTestResult,
    run_parallel_trends_test,
    parallel_trends_test,
    summarize_parallel_trends_test,
)

from .aggregation import (
    CohortEffect,
    OverallEffect,
    EventTimeEffect,
    aggregate_to_cohort,
    aggregate_to_overall,
    aggregate_to_event_time,
    construct_aggregated_outcome,
    cohort_effects_to_dataframe,
)

from .estimators import (
    IPWResult,
    estimate_ipw,
    IPWRAResult,
    estimate_ipwra,
    estimate_propensity_score,
    estimate_outcome_model,
    PSMResult,
    estimate_psm,
)

from .randomization import (
    StaggeredRIResult,
    randomization_inference_staggered,
    ri_overall_effect,
    ri_cohort_effect,
)

__all__ = [
    # -------------------------------------------------------------------------
    # Data Transformations (Post-treatment)
    # -------------------------------------------------------------------------
    'transform_staggered_demean',
    'transform_staggered_detrend',
    'transform_staggered_demeanq',
    'transform_staggered_detrendq',
    'get_cohorts',
    'get_valid_periods_for_cohort',
    # -------------------------------------------------------------------------
    # Data Transformations (Pre-treatment)
    # -------------------------------------------------------------------------
    'transform_staggered_demean_pre',
    'transform_staggered_detrend_pre',
    'get_pre_treatment_periods_for_cohort',
    # -------------------------------------------------------------------------
    # Control Group Selection
    # -------------------------------------------------------------------------
    'ControlGroupStrategy',
    'get_valid_control_units',
    'get_all_control_masks',
    'get_all_control_masks_pre',
    'validate_control_group',
    'identify_never_treated_units',
    'has_never_treated_units',
    'count_control_units_by_strategy',
    # -------------------------------------------------------------------------
    # Cohort-Time Effect Estimation (Post-treatment)
    # -------------------------------------------------------------------------
    'CohortTimeEffect',
    'estimate_cohort_time_effects',
    'run_ols_regression',
    'results_to_dataframe',
    # -------------------------------------------------------------------------
    # Pre-treatment Effect Estimation
    # -------------------------------------------------------------------------
    'PreTreatmentEffect',
    'estimate_pre_treatment_effects',
    'pre_treatment_effects_to_dataframe',
    # -------------------------------------------------------------------------
    # Parallel Trends Testing
    # -------------------------------------------------------------------------
    'ParallelTrendsTestResult',
    'run_parallel_trends_test',
    'parallel_trends_test',
    'summarize_parallel_trends_test',
    # -------------------------------------------------------------------------
    # Effect Aggregation
    # -------------------------------------------------------------------------
    'CohortEffect',
    'OverallEffect',
    'EventTimeEffect',
    'aggregate_to_cohort',
    'aggregate_to_overall',
    'aggregate_to_event_time',
    'construct_aggregated_outcome',
    'cohort_effects_to_dataframe',
    # -------------------------------------------------------------------------
    # IPW Estimation
    # -------------------------------------------------------------------------
    'IPWResult',
    'estimate_ipw',
    # -------------------------------------------------------------------------
    # Doubly Robust Estimation (IPWRA)
    # -------------------------------------------------------------------------
    'IPWRAResult',
    'estimate_ipwra',
    'estimate_propensity_score',
    'estimate_outcome_model',
    # -------------------------------------------------------------------------
    # Propensity Score Matching (PSM)
    # -------------------------------------------------------------------------
    'PSMResult',
    'estimate_psm',
    # -------------------------------------------------------------------------
    # Randomization Inference
    # -------------------------------------------------------------------------
    'StaggeredRIResult',
    'randomization_inference_staggered',
    'ri_overall_effect',
    'ri_cohort_effect',
]

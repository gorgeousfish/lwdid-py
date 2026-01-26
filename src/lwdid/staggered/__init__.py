"""
Staggered difference-in-differences estimation for panel data.

This module implements estimation methods for staggered adoption settings
where units begin treatment at different time periods. The approach uses
rolling time-series transformations at the unit level combined with
cross-sectional treatment effect estimators.

The module provides tools for:

- **Data Transformations**: Unit-specific demeaning and detrending that
  remove pre-treatment averages or linear trends, converting panel data
  into cross-sectional regression problems for each (cohort, period) pair.

- **Control Group Selection**: Flexible strategies for choosing valid
  control units, including never-treated only or not-yet-treated units.

- **Effect Estimation**: Regression adjustment (RA) estimators for
  cohort-time-specific average treatment effects on the treated (ATT).

- **Doubly Robust Estimation**: Inverse probability weighted regression
  adjustment (IPWRA) combining propensity score weighting with outcome
  regression for robustness to model misspecification.

- **Propensity Score Matching**: Nearest-neighbor matching on estimated
  propensity scores for nonparametric treatment effect estimation.

- **Effect Aggregation**: Aggregation of (g, r)-specific effects to
  cohort-level or overall average treatment effects with appropriate
  weighting schemes.

- **Randomization Inference**: Permutation-based inference procedures
  for finite-sample validity without distributional assumptions.

Notes
-----
In staggered designs, treatment cohorts are indexed by g (the first
treatment period) and calendar time by r. The ATT parameters of interest
are tau_{g,r} for each cohort g in periods r >= g. The key identification
assumptions are conditional parallel trends (across all cohort assignments)
and no anticipation (pre-treatment potential outcomes are identical to the
never-treated state).

The rolling transformation approach uses all pre-treatment periods for
each cohort to maximize efficiency while maintaining identification under
standard assumptions. For cohort g in period r, the control group can
include never-treated units and units first treated in periods after r.
"""

from .transformations import (
    transform_staggered_demean,
    transform_staggered_detrend,
    get_cohorts,
    get_valid_periods_for_cohort,
)

from .control_groups import (
    ControlGroupStrategy,
    get_valid_control_units,
    get_all_control_masks,
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

from .aggregation import (
    CohortEffect,
    OverallEffect,
    aggregate_to_cohort,
    aggregate_to_overall,
    construct_aggregated_outcome,
    cohort_effects_to_dataframe,
)

from .estimators import (
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
    # Transformations
    'transform_staggered_demean',
    'transform_staggered_detrend',
    'get_cohorts',
    'get_valid_periods_for_cohort',
    # Control groups
    'ControlGroupStrategy',
    'get_valid_control_units',
    'get_all_control_masks',
    'validate_control_group',
    'identify_never_treated_units',
    'has_never_treated_units',
    'count_control_units_by_strategy',
    # Estimation
    'CohortTimeEffect',
    'estimate_cohort_time_effects',
    'run_ols_regression',
    'results_to_dataframe',
    # Aggregation
    'CohortEffect',
    'OverallEffect',
    'aggregate_to_cohort',
    'aggregate_to_overall',
    'construct_aggregated_outcome',
    'cohort_effects_to_dataframe',
    # IPWRA estimator
    'IPWRAResult',
    'estimate_ipwra',
    'estimate_propensity_score',
    'estimate_outcome_model',
    # PSM estimator
    'PSMResult',
    'estimate_psm',
    # Randomization inference
    'StaggeredRIResult',
    'randomization_inference_staggered',
    'ri_overall_effect',
    'ri_cohort_effect',
]

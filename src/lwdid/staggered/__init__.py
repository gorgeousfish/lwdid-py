"""
Staggered DiD Module

Implements staggered difference-in-differences estimation based on 
Lee and Wooldridge (2023, 2025).

This module provides:
- Data transformation functions for staggered settings
- Control group selection utilities
- (g,r)-specific effect estimation (RA and IPWRA)
- Effect aggregation
- IPWRA doubly robust estimation
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
    # PSM estimator
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

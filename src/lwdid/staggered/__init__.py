r"""
Staggered difference-in-differences estimation for panel data.

This module implements estimation methods for staggered adoption designs
where treatment timing varies across units. The approach applies unit-level
rolling time-series transformations combined with cross-sectional treatment
effect estimators to identify cohort-time-specific average treatment effects
on the treated (ATT).

The module supports data transformations (demeaning, detrending), flexible
control group selection (never-treated, not-yet-treated), multiple estimators
(RA, IPW, IPWRA, PSM), effect aggregation, parallel trends testing, and
permutation-based randomization inference.

Notes
-----
Treatment cohorts are indexed by g (first treatment period) and calendar
time by r. The ATT parameters :math:`\\tau_{g,r}` represent the effect
for cohort g in period r, where r >= g.

Identification requires two key assumptions:

1. **No anticipation**: Treatment effects are zero prior to treatment onset.
2. **Conditional parallel trends**: Untreated outcome trends are independent
   of treatment timing conditional on covariates.

The rolling transformation removes unit-specific pre-treatment patterns:

.. math::

    \\dot{Y}_{irg} = Y_{ir} - \\bar{Y}_{i,pre(g)} \\quad \\text{(demeaning)}

.. math::

    \\ddot{Y}_{irg} = Y_{ir} - \\hat{A}_{ig} - \\hat{B}_{ig} r

Valid control units for cohort g in period r include never-treated units
and units first treated after period r (not-yet-treated).
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

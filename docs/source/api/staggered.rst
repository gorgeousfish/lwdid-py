Staggered DiD Module (staggered)
================================

The staggered module implements difference-in-differences estimation for settings
with staggered treatment adoption, based on Lee and Wooldridge (2025).

Overview
--------

In staggered settings, different units begin treatment at different times (cohorts).
This module provides:

- **Data transformations**: Cohort-specific demeaning and detrending
- **Control group selection**: Never-treated or not-yet-treated units
- **Effect estimation**: (g,r)-specific, cohort, and overall effects
- **Multiple estimators**: RA (regression adjustment), IPW, IPWRA, PSM
- **Randomization inference**: Bootstrap and permutation tests

Key Concepts
------------

Cohort (g)
    The period when a unit first receives treatment. Units that are never treated
    can be encoded as gvar=0, gvar=NaN, or gvar=inf (all internally mapped to
    :math:`\infty`).

(g, r) Effect
    The treatment effect :math:`\tau_{gr}` for cohort :math:`g` at calendar time
    :math:`r`, where :math:`r \geq g`. This is the ATT for units first treated
    in period :math:`g`, evaluated at time :math:`r`.

Control Group Strategies
    Lee and Wooldridge (2025) establishes that under no anticipation and
    conditional parallel trends, both never-treated and not-yet-treated units
    provide valid counterfactuals:

    - ``never_treated``: Only units with :math:`D_\infty = 1` (never treated during
      observation).
      Required when using ``aggregate='cohort'`` or ``aggregate='overall'``.
    - ``not_yet_treated``: Never-treated plus cohorts h > r (units first treated
      after period r). Uses more control observations, potentially improving
      efficiency.
    - ``all_others``: All units not in the treated cohort, including units that
      were already treated in earlier periods. This option is primarily intended
      for replication and diagnostics; it may introduce forbidden comparisons
      under the no-anticipation assumption.

    The theoretical justification shows that the cohort assignments are
    unconfounded with respect to the transformed potential outcome conditional
    on covariates.

Aggregation Levels
    - ``none``: Returns all :math:`(g, r)`-specific effects
    - ``cohort``: Averages effects within each cohort:
      :math:`\tau_g = \frac{1}{T-g+1} \sum_{r=g}^{T} \tau_{gr}`
    - ``overall``: Cohort-share weighted average:
      :math:`\tau_\omega = \sum_g \omega_g \tau_g` where
      :math:`\omega_g = N_g/N_{treat}`

All Units Eventually Treated
    When no units remain untreated through period :math:`T` (no never-treated
    group), treatment effects are defined relative to :math:`Y_t(T)` instead of
    :math:`Y_t(\infty)`. Effects can be estimated for cohorts
    :math:`g \in \{S, \ldots, T-1\}`; the final cohort (:math:`g = T`) serves as
    the control for all earlier cohorts in period :math:`T`. See
    :doc:`../methodological_notes` for theoretical details.

Transformations
---------------

.. autofunction:: lwdid.staggered.transform_staggered_demean

.. autofunction:: lwdid.staggered.transform_staggered_detrend

.. autofunction:: lwdid.staggered.transform_staggered_demeanq

.. autofunction:: lwdid.staggered.transform_staggered_detrendq

.. note::

   All four transformation methods (``demean``, ``detrend``, ``demeanq``,
   ``detrendq``) are available through the main ``lwdid()`` function in
   staggered mode. The seasonal transformations (``demeanq``, ``detrendq``)
   require the ``season_var`` and ``Q`` parameters.

.. autofunction:: lwdid.staggered.get_cohorts

.. autofunction:: lwdid.staggered.get_valid_periods_for_cohort

Control Groups
--------------

.. autoclass:: lwdid.staggered.ControlGroupStrategy
   :members:
   :undoc-members:
   :no-index:

.. autofunction:: lwdid.staggered.get_valid_control_units

.. autofunction:: lwdid.staggered.get_all_control_masks

.. autofunction:: lwdid.staggered.get_all_control_masks_pre

.. autofunction:: lwdid.staggered.validate_control_group

.. autofunction:: lwdid.staggered.identify_never_treated_units

.. autofunction:: lwdid.staggered.has_never_treated_units

.. autofunction:: lwdid.staggered.count_control_units_by_strategy

Estimation
----------

.. autoclass:: lwdid.staggered.CohortTimeEffect
   :members:
   :no-index:

.. autofunction:: lwdid.staggered.estimate_cohort_time_effects

.. autofunction:: lwdid.staggered.run_ols_regression

.. autofunction:: lwdid.staggered.results_to_dataframe

Aggregation
-----------

.. autoclass:: lwdid.staggered.CohortEffect
   :members:
   :no-index:

.. autoclass:: lwdid.staggered.OverallEffect
   :members:
   :no-index:

.. autoclass:: lwdid.staggered.EventTimeEffect
   :members:
   :no-index:

.. autofunction:: lwdid.staggered.aggregate_to_cohort

.. autofunction:: lwdid.staggered.aggregate_to_overall

.. autofunction:: lwdid.staggered.aggregate_to_event_time

.. autofunction:: lwdid.staggered.construct_aggregated_outcome

.. autofunction:: lwdid.staggered.cohort_effects_to_dataframe

IPW Estimator
-------------

Inverse probability weighting estimates treatment effects by weighting observations
based on their propensity scores.

.. autoclass:: lwdid.staggered.IPWResult
   :members:
   :no-index:

.. autofunction:: lwdid.staggered.estimate_ipw

IPWRA Estimator
---------------

The doubly robust IPWRA estimator combines regression adjustment and inverse
probability weighting, providing consistent estimates when either the outcome
model or the propensity score model is correctly specified.

.. autoclass:: lwdid.staggered.IPWRAResult
   :members:
   :no-index:

.. autofunction:: lwdid.staggered.estimate_ipwra

.. autofunction:: lwdid.staggered.estimate_propensity_score

.. autofunction:: lwdid.staggered.estimate_outcome_model

PSM Estimator
-------------

.. autoclass:: lwdid.staggered.PSMResult
   :members:
   :no-index:

.. autofunction:: lwdid.staggered.estimate_psm

Inference Distribution by Estimator
-----------------------------------

The reference distribution used for constructing confidence intervals and
computing p-values varies by estimator. The following table summarizes the
inference approach for each estimator in the staggered module:

**Summary Table:**

- **RA (Regression Adjustment)**: t-distribution with df = N_treated + N_control - k
- **IPW (Inverse Probability Weighting)**: Normal distribution (asymptotic inference)
- **IPWRA (Doubly Robust)**: Normal distribution (asymptotic inference)
- **PSM (Propensity Score Matching)**: Normal distribution (asymptotic inference)

**Detailed Explanation:**

RA (Regression Adjustment)
~~~~~~~~~~~~~~~~~~~~~~~~~~

The regression adjustment estimator uses the t-distribution for inference,
following Lee and Wooldridge (2026). Under classical linear model
assumptions (normality and homoskedasticity), this provides exact finite-sample
inference. With heteroskedasticity-robust standard errors (HC0-HC4), the
t-distribution provides a conservative approximation that improves upon normal
approximations in small samples.

IPW, IPWRA, and PSM
~~~~~~~~~~~~~~~~~~~

The IPW, IPWRA, and PSM estimators use the normal distribution for asymptotic
inference because:

1. These estimators rely on influence function-based variance estimation
2. Asymptotic theory justifies normal approximations for these methods
3. For large samples, the normal distribution provides valid inference

For small samples where exact inference is desired, consider using RA with
``vce=None`` instead of IPW/IPWRA/PSM.

**Practical Recommendations:**

1. **Small samples** (N < 50): Use RA with ``vce=None`` for exact t-based
   inference, or use randomization inference (``ri=True``) for assumption-free
   testing.

2. **Moderate samples** (50 ≤ N < 200): Use RA or IPWRA with HC3 standard errors
   (``vce='hc3'``).

3. **Large samples** (N ≥ 200): All estimators with asymptotic inference are
   appropriate; IPWRA is recommended when functional form assumptions are
   uncertain due to its double robustness property.

See :doc:`../methodological_notes` for detailed theoretical foundations.

Randomization Inference
-----------------------

.. autoclass:: lwdid.staggered.StaggeredRIResult
   :members:
   :no-index:

.. autofunction:: lwdid.staggered.randomization_inference_staggered

.. autofunction:: lwdid.staggered.ri_overall_effect

.. autofunction:: lwdid.staggered.ri_cohort_effect

Pre-treatment Dynamics
----------------------

The pre-treatment dynamics module implements estimation and testing for
pre-treatment periods, following Lee and Wooldridge (2025).

Pre-treatment Transformations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: lwdid.staggered.transform_staggered_demean_pre

.. autofunction:: lwdid.staggered.transform_staggered_detrend_pre

.. autofunction:: lwdid.staggered.get_pre_treatment_periods_for_cohort

Pre-treatment Estimation
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lwdid.staggered.PreTreatmentEffect
   :members:
   :no-index:

.. autofunction:: lwdid.staggered.estimate_pre_treatment_effects

.. autofunction:: lwdid.staggered.pre_treatment_effects_to_dataframe

Parallel Trends Testing
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lwdid.staggered.ParallelTrendsTestResult
   :members:
   :no-index:

.. autofunction:: lwdid.staggered.run_parallel_trends_test

.. autofunction:: lwdid.staggered.summarize_parallel_trends_test

Examples
--------

Basic Staggered Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   from lwdid import lwdid
   
   # Load Castle Law data
   data = pd.read_csv('castle.csv')
   
   # Create gvar from effyear (NaN = never treated -> 0)
   data['gvar'] = data['effyear'].fillna(0).astype(int)
   
   # Run staggered DiD
   results = lwdid(
       data=data,
       y='lhomicide',        # Log homicide rate
       ivar='sid',           # State ID (integer)
       tvar='year',          # Year
       gvar='gvar',          # First treatment year
       rolling='demean',     # Demeaning transformation
       control_group='never_treated',  # Use only never-treated as controls
       aggregate='overall',  # Get overall weighted effect
       vce='hc3'            # HC3 standard errors
   )
   
   # View results
   print(results.summary())
   print(f"Overall ATT: {results.att_overall:.4f}")
   print(f"SE: {results.se_overall:.4f}")

Cohort-Specific Effects
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get cohort-specific effects
   results = lwdid(
       data=data,
       y='lhomicide',
       ivar='sid',
       tvar='year',
       gvar='gvar',
       rolling='demean',
       aggregate='cohort',   # Aggregate within cohorts
   )
   
   # Access cohort effects
   print(results.att_by_cohort)
   # Columns: cohort, att, se, ci_lower, ci_upper, t_stat, pvalue, n_units, n_periods

All (g, r) Effects
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get all cohort-time specific effects
   results = lwdid(
       data=data,
       y='lhomicide',
       ivar='sid',
       tvar='year',
       gvar='gvar',
       rolling='demean',
       aggregate='none',     # No aggregation
   )
   
   # Access all (g,r) effects
   print(results.att_by_cohort_time)
   # Columns: cohort, period, event_time, att, se, ci_lower, ci_upper, t_stat, pvalue, n_treated, n_control

Event Study Plot
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Generate event study plot
   results = lwdid(
       data=data,
       y='lhomicide',
       ivar='sid',
       tvar='year',
       gvar='gvar',
       aggregate='none',
   )
   
   # Plot event study
   fig = results.plot_event_study(
       title='Castle Doctrine Effect',
       ylabel='Effect on Log Homicide Rate'
   )
   fig.savefig('event_study.png', dpi=300)

Event Time Aggregation (WATT)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lwdid.staggered import aggregate_to_event_time
   
   # Get all cohort-time specific effects
   results = lwdid(
       data=data,
       y='lhomicide',
       ivar='sid',
       tvar='year',
       gvar='gvar',
       aggregate='none',
   )
   
   # Aggregate to event time using WATT (Weighted ATT)
   # WATT(r) = Σ w(g,r) × ATT(g, g+r) where w(g,r) = N_g / Σ N_g'
   watt_effects = aggregate_to_event_time(
       cohort_time_effects=results.att_by_cohort_time,
       cohort_sizes=results.cohort_sizes,
       alpha=0.05,
       df_strategy='conservative',  # Use min(df) across cohorts
   )
   
   # Access event-time aggregated effects
   for e in watt_effects:
       print(f"Event time {e.event_time}: WATT={e.att:.4f}, SE={e.se:.4f}, "
             f"CI=[{e.ci_lower:.4f}, {e.ci_upper:.4f}], p={e.pvalue:.4f}")
   
   # Or use plot_event_study with weighted aggregation
   fig, ax = results.plot_event_study(
       aggregation='weighted',  # Use WATT aggregation
       title='Event Study with WATT Aggregation',
   )

Pre-treatment Dynamics and Parallel Trends Testing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Estimate with pre-treatment dynamics for parallel trends assessment
   results = lwdid(
       data=data,
       y='lhomicide',
       ivar='sid',
       tvar='year',
       gvar='gvar',
       rolling='demean',
       aggregate='cohort',
       include_pretreatment=True,    # Enable pre-treatment estimation
       pretreatment_test=True,       # Run parallel trends test
       pretreatment_alpha=0.05,      # Significance level
   )
   
   # View summary with pre-treatment results
   print(results.summary())
   
   # Access pre-treatment ATT estimates
   print(results.att_pre_treatment)
   # Columns: cohort, period, event_time, att, se, ci_lower, ci_upper,
   #          t_stat, pvalue, n_treated, n_control, is_anchor, rolling_window_size
   
   # Access parallel trends test results
   pt = results.parallel_trends_test
   print(f"Joint F-stat: {pt.joint_f_stat:.4f}")
   print(f"P-value: {pt.joint_pvalue:.4f}")
   print(f"Reject H0: {pt.reject_null}")
   
   # Plot event study with pre-treatment effects
   fig, ax = results.plot_event_study(
       include_pre_treatment=True,
       title='Event Study with Pre-treatment Effects',
       pre_treatment_color='gray',
       post_treatment_color='blue',
   )
   fig.savefig('event_study_pretreatment.png', dpi=300)

Low-Level API Usage
~~~~~~~~~~~~~~~~~~~

For more control, the staggered module can be used directly:

.. code-block:: python

   from lwdid.staggered import (
       transform_staggered_demean,
       estimate_cohort_time_effects,
       aggregate_to_overall,
       ControlGroupStrategy
   )
   
   # Step 1: Transform data
   data_transformed = transform_staggered_demean(
       data=data,
       y='lhomicide',
       ivar='sid',
       tvar='year',
       gvar='gvar'
   )
   
   # Step 2: Estimate (g,r) effects
   effects = estimate_cohort_time_effects(
       data_transformed=data_transformed,
       gvar='gvar',
       ivar='sid',
       tvar='year',
       control_strategy='never_treated',
       vce='hc3'
   )
   
   # Step 3: Aggregate to overall effect
   overall = aggregate_to_overall(
       data_transformed=data_transformed,
       gvar='gvar',
       ivar='sid',
       tvar='year',
       cohorts=[2005, 2006, 2007, 2008, 2009],
       T_max=2010,
       vce='hc3'
   )
   
   print(f"Overall ATT: {overall.att:.4f} (SE: {overall.se:.4f})")

See Also
--------

- :func:`lwdid.lwdid` - Main estimation function with staggered support
- :doc:`../user_guide` - Comprehensive usage guide
- :doc:`../methodological_notes` - Theoretical foundations
- :doc:`../examples/index` - Examples including Castle Law analysis

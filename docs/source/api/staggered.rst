Staggered DiD Module (staggered)
================================

The staggered module implements difference-in-differences estimation for settings
with staggered treatment adoption, based on Lee and Wooldridge (2023, 2025).

Overview
--------

In staggered settings, different units begin treatment at different times (cohorts).
This module provides:

- **Data transformations**: Cohort-specific demeaning and detrending
- **Control group selection**: Never-treated or not-yet-treated units
- **Effect estimation**: (g,r)-specific, cohort, and overall effects
- **Multiple estimators**: RA (regression adjustment), IPWRA, PSM
- **Randomization inference**: Bootstrap and permutation tests

Key Concepts
------------

Cohort (g)
    The period when a unit first receives treatment. Units that are never treated
    have gvar=0 or gvar=NaN.

(g, r) Effect
    The treatment effect for cohort g at calendar time r, where r >= g.

Control Group Strategies
    - ``never_treated``: Only units that are never treated (required for aggregation)
    - ``not_yet_treated``: Units not yet treated at time r (gvar > r)

Aggregation Levels
    - ``none``: Returns all (g, r)-specific effects
    - ``cohort``: Averages effects within each cohort
    - ``overall``: Single weighted average across all cohorts

Transformations
---------------

.. autofunction:: lwdid.staggered.transform_staggered_demean

.. autofunction:: lwdid.staggered.transform_staggered_detrend

.. autofunction:: lwdid.staggered.get_cohorts

.. autofunction:: lwdid.staggered.get_valid_periods_for_cohort

Control Groups
--------------

.. autoclass:: lwdid.staggered.ControlGroupStrategy
   :members:
   :undoc-members:

.. autofunction:: lwdid.staggered.get_valid_control_units

.. autofunction:: lwdid.staggered.get_all_control_masks

.. autofunction:: lwdid.staggered.validate_control_group

.. autofunction:: lwdid.staggered.identify_never_treated_units

.. autofunction:: lwdid.staggered.has_never_treated_units

.. autofunction:: lwdid.staggered.count_control_units_by_strategy

Estimation
----------

.. autoclass:: lwdid.staggered.CohortTimeEffect
   :members:

.. autofunction:: lwdid.staggered.estimate_cohort_time_effects

.. autofunction:: lwdid.staggered.run_ols_regression

.. autofunction:: lwdid.staggered.results_to_dataframe

Aggregation
-----------

.. autoclass:: lwdid.staggered.CohortEffect
   :members:

.. autoclass:: lwdid.staggered.OverallEffect
   :members:

.. autofunction:: lwdid.staggered.aggregate_to_cohort

.. autofunction:: lwdid.staggered.aggregate_to_overall

.. autofunction:: lwdid.staggered.construct_aggregated_outcome

.. autofunction:: lwdid.staggered.cohort_effects_to_dataframe

IPWRA Estimator
---------------

.. autoclass:: lwdid.staggered.IPWRAResult
   :members:

.. autofunction:: lwdid.staggered.estimate_ipwra

.. autofunction:: lwdid.staggered.estimate_propensity_score

.. autofunction:: lwdid.staggered.estimate_outcome_model

PSM Estimator
-------------

.. autoclass:: lwdid.staggered.PSMResult
   :members:

.. autofunction:: lwdid.staggered.estimate_psm

Randomization Inference
-----------------------

.. autoclass:: lwdid.staggered.StaggeredRIResult
   :members:

.. autofunction:: lwdid.staggered.randomization_inference_staggered

.. autofunction:: lwdid.staggered.ri_overall_effect

.. autofunction:: lwdid.staggered.ri_cohort_effect

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

Low-Level API Usage
~~~~~~~~~~~~~~~~~~~

For more control, you can use the staggered module directly:

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
- :doc:`../examples/castle_law` - Castle Law analysis example

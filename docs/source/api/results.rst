Results Module
==============

Container class for difference-in-differences estimation outputs.

This module provides the ``LWDIDResults`` class for encapsulating estimation
outputs from both common timing and staggered adoption DiD designs. The
container stores point estimates, standard errors, confidence intervals,
period-specific effects, and diagnostic information. Multiple export formats
are supported for reproducible research and integration with statistical
reporting workflows.

The ``LWDIDResults`` object is returned by the main :func:`lwdid.lwdid`
function and provides a unified interface for accessing estimation outputs
regardless of whether the design uses common timing or staggered adoption.
For staggered designs, additional attributes provide cohort-specific and
cohort-time specific effect estimates.

.. contents:: Contents
   :local:
   :depth: 2

.. automodule:: lwdid.results
   :no-members:

Attributes Reference
--------------------

The ``LWDIDResults`` object provides read-only access to estimation outputs
through properties. Attributes are organized into categories based on their
purpose.

Core Estimates
~~~~~~~~~~~~~~

These attributes provide the primary treatment effect estimates and inference
statistics.

- ``att`` (float): Average treatment effect on the treated (ATT).
- ``se_att`` (float): Standard error of ATT.
- ``t_stat`` (float): t-statistic for H0: ATT = 0.
- ``pvalue`` (float): Two-sided p-value for the t-test.
- ``ci_lower`` (float): Lower bound of 95% confidence interval.
- ``ci_upper`` (float): Upper bound of 95% confidence interval.

Sample Information
~~~~~~~~~~~~~~~~~~

These attributes describe the estimation sample.

- ``nobs`` (int): Number of observations in the regression.
- ``n_treated`` (int): Number of treated units in the sample.
- ``n_control`` (int): Number of control units in the sample.
- ``K`` (int): Last pre-treatment period index.
- ``tpost1`` (int): First post-treatment period index.

Inference Configuration
~~~~~~~~~~~~~~~~~~~~~~~

These attributes describe the variance estimation and inference settings.

- ``df_resid`` (int): Residual degrees of freedom from the regression.
- ``df_inference`` (int): Degrees of freedom used for inference. For
  cluster-robust SE, this is G - 1; otherwise equals ``df_resid``.
- ``vce_type`` (str or None): Variance estimator: 'ols', 'hc0', 'robust'/'hc1',
  'hc2', 'hc3', 'hc4', or 'cluster'.
- ``cluster_var`` (str or None): Clustering variable name (if
  ``vce='cluster'``).
- ``n_clusters`` (int or None): Number of clusters (if ``vce='cluster'``).

Transformation and Controls
~~~~~~~~~~~~~~~~~~~~~~~~~~~

These attributes describe the transformation method and control variables used.

- ``rolling`` (str): Transformation method: 'demean', 'detrend', 'demeanq',
  or 'detrendq'.
- ``controls_used`` (bool): Whether control variables were included.
- ``controls`` (list): List of control variable names used.

Period-Specific Effects
~~~~~~~~~~~~~~~~~~~~~~~

- ``att_by_period`` (pd.DataFrame or None): Period-specific ATT estimates with
  columns: period, tindex, beta, se, ci_lower, ci_upper, tstat, pval, N. First
  row contains the overall average effect.

Randomization Inference
~~~~~~~~~~~~~~~~~~~~~~~

These attributes are available only when ``ri=True`` is specified in the
``lwdid()`` call.

- ``ri_pvalue`` (float or None): Randomization inference p-value.
- ``ri_seed`` (int or None): Random seed used for RI.
- ``rireps`` (int or None): Number of RI replications.
- ``ri_method`` (str or None): RI method: 'bootstrap' or 'permutation'.
- ``ri_valid`` (int or None): Number of valid (successful) RI replications.
- ``ri_failed`` (int or None): Number of failed RI replications.

Regression Coefficients
~~~~~~~~~~~~~~~~~~~~~~~

These attributes provide access to the full regression output.

- ``params`` (array-like): Full vector of regression coefficients.
- ``bse`` (array-like): Standard errors of all coefficients.
- ``vcov`` (array-like): Variance-covariance matrix.

Staggered-Specific Attributes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These attributes are available only for staggered adoption designs (when
``gvar`` is specified instead of ``post``).

**Design Information:**

- ``is_staggered`` (bool): Whether this is a staggered DiD design.
- ``cohorts`` (list): Sorted list of treatment cohorts (first treatment
  periods).
- ``cohort_sizes`` (dict): Number of units in each cohort.
- ``n_never_treated`` (int or None): Number of never-treated units.
- ``control_group`` (str or None): User-specified control group strategy.
- ``control_group_used`` (str or None): Actual control group strategy used
  (may differ due to auto-switching).
- ``aggregate`` (str or None): Aggregation level: 'none', 'cohort', or
  'overall'.
- ``estimator`` (str or None): Estimation method: 'ra', 'ipw', 'ipwra', or
  'psm'.

**Effect Estimates:**

- ``att_by_cohort_time`` (pd.DataFrame or None): (g,r)-specific ATT estimates
  with columns: cohort, period, event_time, att, se, ci_lower, ci_upper,
  t_stat, pvalue, n_treated, n_control.
- ``att_by_cohort`` (pd.DataFrame or None): Cohort-specific ATT estimates (if
  ``aggregate='cohort'`` or ``'overall'``).
- ``att_overall`` (float or None): Overall weighted ATT (if
  ``aggregate='overall'``).
- ``se_overall`` (float or None): Standard error of overall ATT.
- ``ci_overall_lower`` (float or None): 95% CI lower bound for overall ATT.
- ``ci_overall_upper`` (float or None): 95% CI upper bound for overall ATT.
- ``t_stat_overall`` (float or None): t-statistic for overall ATT.
- ``pvalue_overall`` (float or None): p-value for overall ATT.
- ``cohort_weights`` (dict): Cohort weights for overall effect
  (:math:`\omega_g = N_g / N_{treat}`).

Methods Reference
-----------------

Display Methods
~~~~~~~~~~~~~~~

**summary()**
    Generate a formatted summary of estimation results. For staggered designs,
    automatically dispatches to ``summary_staggered()``. Returns a string
    suitable for console output.

    .. code-block:: python

       print(results.summary())

**summary_staggered()**
    Generate a formatted summary for staggered DiD results. Displays treatment
    cohorts, sample sizes, control group strategy, overall weighted effect (if
    ``aggregate='overall'``), and cohort-specific effects. Raises ValueError
    if called on non-staggered results.

Visualization Methods
~~~~~~~~~~~~~~~~~~~~~

**plot(gid=None, graph_options=None)**
    Generate a time series plot of residualized outcomes for treated and
    control groups. A vertical line indicates the treatment start period.

    Parameters:

    - ``gid`` (str or int, optional): Specific unit ID to highlight.
    - ``graph_options`` (dict, optional): Matplotlib customization options.

    Returns: matplotlib.figure.Figure

    .. code-block:: python

       fig = results.plot()
       fig.savefig('trends.png')

**plot_event_study(...)**
    Generate an event study diagram for staggered DiD results. Aggregates
    cohort-time specific effects by event time (e = r - g) and visualizes
    dynamic treatment effects relative to a reference period.

    Key parameters:

    - ``ref_period`` (int, optional): Reference period for normalization. Default 0.
    - ``show_ci`` (bool): Display 95% confidence interval shading. Default True.
    - ``aggregation`` ({'mean', 'weighted'}): Cross-cohort aggregation method.
    - ``include_pre_treatment`` (bool): Include pre-treatment periods. Default True.
    - ``return_data`` (bool): Also return the aggregated event study DataFrame.

    Returns: (fig, ax) or (fig, ax, event_df) if ``return_data=True``.

    .. code-block:: python

       fig, ax = results.plot_event_study(
           title='Policy Effect',
           ylabel='Treatment Effect'
       )
       fig.savefig('event_study.png', dpi=300)

Export Methods
~~~~~~~~~~~~~~

**to_excel(path)**
    Export results to an Excel file with multiple sheets. For common timing:
    Summary sheet with core statistics, ByPeriod sheet with period-specific
    effects, and RI sheet if randomization inference was performed. For
    staggered designs: Summary, Overall, Cohort, CohortTime, Weights, and
    Metadata sheets.

    .. code-block:: python

       results.to_excel('results.xlsx')

**to_csv(path)**
    Export period-specific effects (``att_by_period``) to CSV format.

    .. code-block:: python

       results.to_csv('period_effects.csv')

**to_latex(path)**
    Export results to a LaTeX table file. Generates publication-ready tables
    containing summary statistics and period-specific effects.

    .. code-block:: python

       results.to_latex('table.tex')

Usage by Design Type
--------------------

Common Timing Design
~~~~~~~~~~~~~~~~~~~~

For common timing designs (specified with the ``post`` parameter), the
following attributes are available:

.. code-block:: python

   from lwdid import lwdid

   results = lwdid(data, y='outcome', d='treated', ivar='unit',
                   tvar='year', post='post', rolling='detrend')

   # Core estimates
   print(f"ATT: {results.att:.4f}")
   print(f"SE: {results.se_att:.4f}")
   print(f"t-stat: {results.t_stat:.3f}")
   print(f"p-value: {results.pvalue:.4f}")
   print(f"95% CI: [{results.ci_lower:.4f}, {results.ci_upper:.4f}]")

   # Sample information
   print(f"N = {results.nobs}, df = {results.df_inference}")
   print(f"Treated: {results.n_treated}, Control: {results.n_control}")

   # Period-specific effects
   print(results.att_by_period)

   # Staggered-specific attributes are None
   assert results.is_staggered == False
   assert results.att_overall is None

Staggered Adoption Design
~~~~~~~~~~~~~~~~~~~~~~~~~

For staggered adoption designs (specified with the ``gvar`` parameter),
additional attributes are available:

.. code-block:: python

   from lwdid import lwdid

   results = lwdid(data, y='outcome', ivar='unit', tvar='year',
                   gvar='first_treat_year', rolling='demean',
                   aggregate='overall', control_group='not_yet_treated')

   # Staggered design indicator
   assert results.is_staggered == True

   # Overall weighted effect (when aggregate='overall')
   print(f"Overall ATT: {results.att_overall:.4f}")
   print(f"SE: {results.se_overall:.4f}")
   print(f"95% CI: [{results.ci_overall_lower:.4f}, {results.ci_overall_upper:.4f}]")

   # Cohort information
   print(f"Cohorts: {results.cohorts}")
   print(f"Cohort sizes: {results.cohort_sizes}")
   print(f"Cohort weights: {results.cohort_weights}")

   # Cohort-specific effects (when aggregate='cohort' or 'overall')
   print(results.att_by_cohort)

   # (g,r)-specific effects (always available)
   print(results.att_by_cohort_time)

   # Control group information
   print(f"Control group requested: {results.control_group}")
   print(f"Control group used: {results.control_group_used}")

   # Event study visualization
   fig, ax = results.plot_event_study()

Attribute Availability Summary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following table summarizes attribute availability by design type:

- ``att``: Common timing = Yes, Staggered = Yes
- ``se_att``: Common timing = Yes, Staggered = Yes
- ``att_by_period``: Common timing = Yes, Staggered = No
- ``is_staggered``: Common timing = False, Staggered = True
- ``att_overall``: Common timing = No, Staggered = when ``aggregate='overall'``
- ``att_by_cohort``: Common timing = No, Staggered = when ``aggregate='cohort'``
  or ``'overall'``
- ``att_by_cohort_time``: Common timing = No, Staggered = Yes (always)
- ``cohorts``: Common timing = Empty list, Staggered = Yes
- ``cohort_weights``: Common timing = Empty dict, Staggered = Yes

LWDIDResults Class
------------------

.. autoclass:: lwdid.results.LWDIDResults
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Example Usage
-------------

Basic Example
~~~~~~~~~~~~~

.. code-block:: python

   from lwdid import lwdid

   # Run estimation
   results = lwdid(data, y='outcome', d='treated', ivar='unit',
                   tvar='year', post='post', rolling='detrend')

   # Display summary
   print(results.summary())

   # Access attributes
   print(f"ATT: {results.att:.4f} (SE: {results.se_att:.4f})")
   print(f"95% CI: [{results.ci_lower:.4f}, {results.ci_upper:.4f}]")

   # Export results
   results.to_excel('results.xlsx')

Staggered Design Example
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lwdid import lwdid

   # Estimate with staggered design
   results = lwdid(data, y='l_homicide', ivar='state', tvar='year',
                   gvar='effyear', rolling='demean',
                   aggregate='overall', control_group='not_yet_treated')

   # Display staggered-specific summary
   print(results.summary())

   # Access overall effect
   print(f"Overall ATT: {results.att_overall:.4f}")

   # Access cohort-level effects
   print(results.att_by_cohort)

   # Generate event study plot
   fig, ax = results.plot_event_study()
   fig.savefig('event_study.png')

   # Export with all sheets
   results.to_excel('staggered_results.xlsx')

See Also
--------

- :func:`lwdid.lwdid` : Main estimation function that produces LWDIDResults objects.
- :doc:`../user_guide` : Comprehensive usage guide.
- :doc:`../quickstart` : Quick start tutorial.
- :doc:`staggered` : Staggered module API for low-level access.

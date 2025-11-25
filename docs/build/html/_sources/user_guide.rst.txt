User Guide
==========

This guide covers all aspects of using the ``lwdid`` package for difference-in-differences
estimation with small samples.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

The ``lwdid`` package implements the Lee and Wooldridge (2025) method for
difference-in-differences estimation with small cross-sectional sample sizes.

Key features:

- Designed for settings with small numbers of treated or control units
- Exact t-based inference under classical linear model assumptions (normality and homoskedasticity)
- Works best with large time dimensions
- Heteroskedasticity-robust standard errors
- Randomization inference
- Four transformation methods

Quick Reference
---------------

For a quick start, see :doc:`quickstart`. For complete API details, see :doc:`api/index`.

Transformation Methods
----------------------

Four transformation methods convert panel data into cross-sectional form. The choice
depends on data structure and assumptions.

demean: Standard DiD with Unit Fixed Effects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**When to use**:

- Annual data or data without seasonal patterns
- Control for time-invariant unit characteristics
- At least 1 pre-treatment period (T₀ ≥ 1)

**What it does**:

Removes unit-specific pre-treatment means from each observation.

**Mathematical form** (Lee and Wooldridge 2025, Procedure 2.1):

For each unit i, compute the pre-treatment mean ȳᵢ₀ and transform:

ỹᵢₜ = yᵢₜ - ȳᵢ₀ for all t

Then estimate a cross-sectional regression on the transformed data.

Example:

.. code-block:: python

   results = lwdid(
       data, y='outcome', d='treated', ivar='unit', tvar='year',
       post='post', rolling='demean'
   )

detrend: DiD with Unit-Specific Linear Trends
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**When to use**:

- Units exhibit different linear trends in the pre-treatment period
- Control for both unit fixed effects and unit-specific trends
- At least 2 pre-treatment periods (T₀ ≥ 2)

**What it does**:

Estimates and removes unit-specific linear trends from the pre-treatment data.

**Mathematical form** (Lee and Wooldridge 2025, Procedure 3.1):

For each unit i, estimate linear trend from pre-treatment data:

yᵢₜ = αᵢ + βᵢ·t + εᵢₜ (for t in pre-treatment period)

Then detrend all observations using the estimated trend.

Example:

.. code-block:: python

   results = lwdid(
       data, y='outcome', d='treated', ivar='unit', tvar='year',
       post='post', rolling='detrend'
   )

demeanq: Quarterly Data with Seasonal Fixed Effects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**When to use**:

- Quarterly data with seasonal patterns
- Control for both unit and quarter-of-year effects
- Pre-treatment sample rich enough to estimate quarter fixed effects for each unit:
  for each unit, the number of pre-treatment observations must be at least the
  number of distinct pre-treatment quarters plus one (n_pre ≥ #quarters_pre + 1)
- For each unit, every quarter that appears in the post-treatment period must also
  appear in its pre-treatment period (quarter-coverage condition).

**What it does**:

Removes unit-specific pre-treatment means and quarter-of-year effects.

**Data requirements**:

Pass ``tvar`` as a list: ``tvar=['year', 'quarter']``

Example:

.. code-block:: python

   results = lwdid(
       data_q, y='sales', d='treated', ivar='store',
       tvar=['year', 'quarter'],  # Composite time variable
       post='post', rolling='demeanq'
   )

detrendq: Quarterly Data with Trends and Seasonal Effects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**When to use**:

- Quarterly data with both trends and seasonal patterns
- Control for unit trends and quarter-of-year effects
- At least 2 pre-treatment periods overall, and for each unit enough pre-treatment
  observations to estimate a linear trend with quarter fixed effects
  (n_pre ≥ 1 + #quarters_pre)
- For each unit, every quarter that appears in the post-treatment period must also
  appear in its pre-treatment period (quarter-coverage condition).

**What it does**:

Combines detrending with quarterly seasonal adjustment.

Example:

.. code-block:: python

   results = lwdid(
       data_q, y='sales', d='treated', ivar='store',
       tvar=['year', 'quarter'],
       post='post', rolling='detrendq'
   )

Variance Estimation
-------------------

The package supports multiple variance estimators for different assumptions
about the error structure.

OLS (Homoskedastic)
~~~~~~~~~~~~~~~~~~~

**When to use**:

- Errors are homoskedastic and normally distributed
- Exact t-based inference is desired

**Specification**:

.. code-block:: python

   results = lwdid(..., vce=None)  # Default

**Degrees of freedom**:

- Non-clustered: df equals the residual degrees of freedom from the
  cross-sectional regression (df_resid). In a simple specification
  without controls this is N - 2 (intercept and treatment indicator);
  with controls and interactions it is N minus the total number of
  estimated parameters.
- Clustered: df = G - 1 (G clusters)

HC1 (Heteroskedasticity-Robust)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**When to use**:

- Heteroskedasticity is suspected
- Sample size is moderate to large, so asymptotic approximations are more reliable

**Specification**:

.. code-block:: python

   results = lwdid(..., vce='hc1')
   # or equivalently:
   results = lwdid(..., vce='robust')

Note: ``'hc1'`` and ``'robust'`` are aliases for the same estimator.

HC3 (Small-Sample Adjusted Robust)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**When to use**:

- Heteroskedasticity is suspected
- Sample size is small or moderate

**Specification**:

.. code-block:: python

   results = lwdid(..., vce='hc3')

**Advantage**: HC3 provides better finite-sample performance than HC1 in small samples.

Cluster-Robust
~~~~~~~~~~~~~~

**When to use**:

- Errors are correlated within clusters (e.g., states, regions)
- Multiple units per cluster

**Specification**:

.. code-block:: python

   results = lwdid(..., vce='cluster', cluster_var='state')

**Degrees of freedom**: G - 1, where G is the number of clusters.

Randomization Inference
-----------------------

Randomization inference provides an alternative to t-based inference without
distributional assumptions.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   results = lwdid(
       data, y='outcome', d='treated', ivar='unit', tvar='year',
       post='post', rolling='demean',
       ri=True,        # Enable RI
       rireps=1000,    # Number of randomization replications (default: 1000)
       seed=42         # Random seed for reproducibility
   )

   print(f"RI p-value: {results.ri_pvalue:.4f}")

RI Methods
~~~~~~~~~~

The package supports two RI methods:

Permutation (Recommended):

.. code-block:: python

   results = lwdid(..., ri=True, ri_method='permutation')

Fisher randomization inference without replacement.

Bootstrap:

.. code-block:: python

   results = lwdid(..., ri=True, ri_method='bootstrap')

With-replacement sampling.

Accessing RI Results
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   results.ri_pvalue          # RI p-value (available if ri=True)
   results.ri_method          # RI method ('permutation' or 'bootstrap')
   results.rireps             # Number of RI replications
   results.ri_valid           # Number of valid RI replications
   results.ri_failed          # Number of failed RI replications
   results.ri_seed            # Random seed actually used for RI (auto-generated if not provided)

Control Variables
-----------------

You can include time-invariant control variables in the estimation.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   results = lwdid(
       data, y='outcome', d='treated', ivar='unit', tvar='year',
       post='post', rolling='demean',
       controls=['population', 'income', 'education']
   )

Requirements
~~~~~~~~~~~~

- Controls must be time-invariant (constant within each unit)
- Controls are automatically centered (demeaned)
- Missing values in controls are handled at the estimation stage: observations with
  missing controls may be dropped if doing so still leaves enough treated and
  control units to satisfy the N1>K+1 and N0>K+1 conditions; otherwise controls
  are omitted and the full regression sample is retained (with a warning)

Effect on Inference
~~~~~~~~~~~~~~~~~~~

Including controls:

- Reduces residual variance (potentially increasing power)
- Reduces degrees of freedom by the number of added regressors (controls and their interactions)
- Controls are centered before inclusion in the cross-sectional regression
- Controls are included only if both the treated and control groups satisfy N1>K+1 and N0>K+1 (where K is the number of controls); otherwise, controls are automatically excluded and a warning is issued

Working with Results
--------------------

The ``lwdid()`` function returns a ``LWDIDResults`` object with comprehensive
estimation results and export capabilities.

Core Estimates
~~~~~~~~~~~~~~

.. code-block:: python

   results.att           # Average treatment effect on treated
   results.se_att        # Standard error
   results.t_stat        # t-statistic
   results.pvalue        # Two-sided p-value
   results.ci_lower      # 95% CI lower bound
   results.ci_upper      # 95% CI upper bound
   results.df_inference  # Degrees of freedom

Period-Specific Effects
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Access period-by-period estimates
   df_periods = results.att_by_period

   # DataFrame with columns:
   # - period: Time period label
   # - tindex: Numeric time index
   # - beta: Treatment effect estimate
   # - se: Standard error
   # - ci_lower, ci_upper: 95% confidence interval
   # - tstat: t-statistic
   # - pval: p-value
   # - N: Sample size
   # The first row summarizes the overall average effect ('period' == 'average').

Sample Information
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   results.nobs          # Total observations in regression
   results.n_treated     # Number of treated units
   results.n_control     # Number of control units
   results.K             # Number of pre-treatment periods
   results.tpost1        # First post-treatment period index

Visualization
~~~~~~~~~~~~~

.. code-block:: python

   # Plot treatment effects over time
   results.plot()

   # Customize the plot
   results.plot(graph_options={'figsize': (10, 6), 'title': 'Treatment Effects Over Time', 'ylabel': 'Effect Size'})

Export Results
~~~~~~~~~~~~~~

Excel:

.. code-block:: python

   results.to_excel('results.xlsx')
   # Creates sheets: Summary, ByPeriod, and RI (if available)

CSV:

.. code-block:: python

   results.to_csv('results.csv')
   # Exports period-by-period estimates

LaTeX:

.. code-block:: python

   results.to_latex('table.tex')
   # Creates publication-ready LaTeX table

Print Summary
~~~~~~~~~~~~~

.. code-block:: python

   results.summary()  # Prints formatted summary to console

Data Requirements and Validation
---------------------------------

Panel Structure
~~~~~~~~~~~~~~~

**Required format**:

- Long format: one row per (unit, time) observation
- Unique (unit, time) combinations
- Continuous time sequence (no gaps in time index)

**Validation**:

The package automatically validates:

- No duplicate (unit, time) pairs
- Time index forms continuous sequence
- Sufficient pre-treatment periods for chosen transformation

Treatment Structure
~~~~~~~~~~~~~~~~~~~

**Common timing assumption**:

All treated units must start treatment in the same period. The ``post``
variable must be a function of time only.

**Treatment persistence**:

Once ``post`` switches from 0 to 1, it must remain 1 for all subsequent
periods (no reversals).

**Validation**:

The package checks:

- ``post`` is binary (0 or 1)
- ``post`` is time-invariant across units in each period
- No treatment reversals

Treatment Indicator
~~~~~~~~~~~~~~~~~~~

The ``d`` parameter must be a time-invariant treatment group indicator:

- ``d = 1`` for all periods if unit is in treatment group
- ``d = 0`` for all periods if unit is in control group
- ``d`` should NOT vary over time

Common mistake: Passing time-varying treatment status instead of group indicator.

Missing Data
~~~~~~~~~~~~

**Allowed**:

- Missing values in outcome variable ``y`` (observations dropped)
- Unbalanced panels (different units can have different numbers of periods)

**Not allowed**:

- Rows with missing values in ``d``, ``ivar``, ``tvar``, or ``post`` (these observations are dropped during validation)
- Gaps in time sequence

Best Practices
--------------

Choosing Transformation Method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Start with ``demean``** for annual data
2. **Use ``detrend``** if pre-treatment trends differ across units
3. **Use ``demeanq``/``detrendq``** for quarterly data with seasonality
4. **Check pre-treatment period requirements** (T₀ ≥ 1 for demean, T₀ ≥ 2 for detrend)

Choosing Variance Estimator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Default (``vce=None``)**: Use if you believe homoskedasticity holds
2. **``vce='hc3'``**: Recommended for small to moderate sample sizes when you suspect heteroskedasticity
3. **``vce='cluster'``**: Use if errors are clustered
4. **Randomization inference**: Use as robustness check or when normality is doubtful

Sample Size Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Theoretical minimum:** N ≥ 3 units (requires strong CLM assumptions: normality and homoskedasticity)
- **Practical recommendation:** As a rough guideline, having at least 10 units improves stability, especially with heteroskedasticity-robust standard errors
- **Time dimension:** Method works best with large T (many time periods), where the central limit theorem across time supports normality
- **Large samples:** For very large cross-sectional samples (for example, N in the hundreds or more), standard difference-in-differences implementations designed for large N are also available; ``lwdid`` remains valid but is primarily motivated by small-N applications

Diagnostic Checks
~~~~~~~~~~~~~~~~~

1. Visual inspection: Plot outcome trends for treated vs. control units
2. Pre-treatment balance: Check if treated and control units are similar pre-treatment
3. Parallel trends: Examine pre-treatment period-specific effects (should be near zero)
4. Sensitivity: Try different transformation methods and variance estimators

Troubleshooting
---------------

Common Errors
~~~~~~~~~~~~~

**"Insufficient pre-treatment periods"**:

- ``demean``/``demeanq`` require T₀ ≥ 1
- ``detrend``/``detrendq`` require T₀ ≥ 2
- Solution: Check your ``post`` variable or use a different transformation

**"Treatment timing is not common across units"**:

- ``post`` must be the same for all units in each period
- Solution: Verify your ``post`` variable is time-based, not unit-specific

**"Control variable varies within unit"**:

- Controls must be time-invariant
- Solution: Use only time-invariant controls or create unit-level averages

**"Duplicate (unit, time) observations"**:

- Each (unit, time) combination must appear only once
- Solution: Check for data duplication or aggregation issues

Performance Tips
~~~~~~~~~~~~~~~~

- For large ``rireps`` (e.g., 10,000), RI can be slow
- Use ``seed`` parameter for reproducible RI results
- Export results to Excel/CSV for further analysis in other tools

Further Reading
---------------

- :doc:`quickstart` - Quick start tutorial
- :doc:`methodological_notes` - Theoretical background
- :doc:`api/index` - Complete API reference
- :doc:`examples/index` - Usage examples

Reference:

Lee, S. J., and Wooldridge, J. M. (2025). *Simple Approaches to Inference with
Difference-in-Differences Estimators with Small Cross-Sectional Sample Sizes*.
Available at SSRN 5325686.

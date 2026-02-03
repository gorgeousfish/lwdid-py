User Guide
==========

This guide covers all aspects of using the ``lwdid`` package for difference-in-differences
estimation with rolling transformations.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

The ``lwdid`` package implements the Lee and Wooldridge methods for difference-in-
differences estimation with panel data, covering three main scenarios:

**Small-Sample Common Timing** (Lee and Wooldridge, 2026):

- Exact t-based inference under CLM assumptions (normality and homoskedasticity)
- Designed for settings with small numbers of treated or control units
- Works best with large time dimensions

**Large-Sample Common Timing** (Lee and Wooldridge, 2025):

- Asymptotic inference with robust standard errors
- Supports heteroskedasticity-robust (HC0-HC4) and cluster-robust options
- Multiple estimators: RA, IPW, IPWRA, PSM

**Staggered Adoption** (Lee and Wooldridge, 2025):

- Units treated at different times
- Cohort-time specific effect estimation
- Flexible control group strategies and aggregation options

Key features:

- Four transformation methods: demean, detrend, demeanq, detrendq
- Multiple variance estimators for different assumptions
- Randomization inference for small samples
- Event study visualization for staggered designs

Scenario Selection Guide
------------------------

Use this guide to choose the appropriate scenario for your analysis:

- **Small-sample common timing**: Use when N is small with common treatment
  timing. Key parameters: ``post``, ``vce=None``.
- **Large-sample common timing**: Use when N is larger with common treatment
  timing. Key parameters: ``post``, ``vce='hc3'``.
- **Staggered adoption**: Use when units are treated at different times. Key
  parameters: ``gvar``, ``aggregate``.

**Decision Flowchart:**

.. code-block:: text

                    ┌─────────────────────────────────┐
                    │  Treatment timing varies        │
                    │  across units?                  │
                    └───────────────┬─────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
               ┌────────┐                     ┌─────────┐
               │  YES   │                     │   NO    │
               └────┬───┘                     └────┬────┘
                    │                              │
                    ▼                              ▼
    ┌───────────────────────────┐   ┌─────────────────────────────┐
    │  STAGGERED ADOPTION       │   │  Is cross-sectional N       │
    │  Use: gvar parameter      │   │  small (< 50)?              │
    │  Estimator: ra/ipwra      │   └───────────────┬─────────────┘
    │  Aggregate: cohort/overall│                   │
    └───────────────────────────┘   ┌───────────────┴───────────────┐
                                    │                               │
                                    ▼                               ▼
                               ┌────────┐                     ┌─────────┐
                               │  YES   │                     │   NO    │
                               └────┬───┘                     └────┬────┘
                                    │                              │
                                    ▼                              ▼
            ┌─────────────────────────────────────┐   ┌─────────────────────────────────┐
            │  SMALL-SAMPLE COMMON TIMING         │   │  LARGE-SAMPLE COMMON TIMING     │
            │  Use: post parameter                │   │  Use: post parameter            │
            │  VCE: None (exact t-inference)      │   │  VCE: hc3 or robust             │
            │  Alt: ri=True for RI                │   │  Estimator: ra/ipwra            │
            │  Reference: Lee and Wooldridge      │   │  Reference: Lee and Wooldridge  │
            │  (2026)                             │   │  (2025)                         │
            └─────────────────────────────────────┘   └─────────────────────────────────┘

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
- At least 1 pre-treatment period (:math:`T_0 \geq 1`)

**What it does**:

Removes unit-specific pre-treatment means from each observation.

**Mathematical form** (Lee and Wooldridge, 2026, Procedure 2.1):

For each unit :math:`i`, compute the pre-treatment mean :math:`\bar{Y}_{i,pre}` and
transform:

.. math::

   \dot{Y}_{it} = Y_{it} - \bar{Y}_{i,pre} \quad \text{for all } t

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
- At least 2 pre-treatment periods (:math:`T_0 \geq 2`)

**What it does**:

Estimates and removes unit-specific linear trends from the pre-treatment data.

**Mathematical form** (Lee and Wooldridge, 2026, Procedure 3.1):

For each unit :math:`i`, estimate linear trend from pre-treatment data:

.. math::

   Y_{it} = \alpha_i + \beta_i t + \varepsilon_{it} \quad \text{(for } t \text{ in pre-treatment period)}

Then detrend all observations using the estimated trend.

Example:

.. code-block:: python

   results = lwdid(
       data, y='outcome', d='treated', ivar='unit', tvar='year',
       post='post', rolling='detrend'
   )

demeanq: Seasonal Data with Seasonal Fixed Effects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**When to use**:

- Periodic data with seasonal patterns (quarterly, monthly, or weekly)
- Control for both unit and season-of-year effects
- Pre-treatment sample rich enough to estimate seasonal fixed effects for each unit:
  for each unit, the number of pre-treatment observations must be at least Q + 1
  (where Q is the number of seasons per year)
- For each unit, every season that appears in the post-treatment period must also
  appear in its pre-treatment period (season-coverage condition).

**What it does**:

Removes unit-specific pre-treatment means and season-of-year effects.

**Supported seasonal periods (Q parameter)**:

- **Q=4** (default): Quarterly data (4 seasons per year)
- **Q=12**: Monthly data (12 seasons per year)
- **Q=52**: Weekly data (52 seasons per year)

**Data requirements**:

- For quarterly data: Pass ``tvar`` as a list: ``tvar=['year', 'quarter']``
- For monthly/weekly data: Use a single time variable with ``Q`` and ``season_var`` parameters

**Examples**:

Quarterly data (Q=4, default):

.. code-block:: python

   results = lwdid(
       data_q, y='sales', d='treated', ivar='store',
       tvar=['year', 'quarter'],  # Composite time variable
       post='post', rolling='demeanq'
   )

Monthly data (Q=12):

.. code-block:: python

   results = lwdid(
       data_m, y='sales', d='treated', ivar='store',
       tvar='time',              # Single time index
       post='post', rolling='demeanq',
       Q=12,                     # 12 seasons per year
       season_var='month'        # Month indicator (1-12)
   )

Weekly data (Q=52):

.. code-block:: python

   results = lwdid(
       data_w, y='sales', d='treated', ivar='store',
       tvar='time',              # Single time index
       post='post', rolling='demeanq',
       Q=52,                     # 52 seasons per year
       season_var='week'         # Week indicator (1-52)
   )

detrendq: Seasonal Data with Trends and Seasonal Effects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**When to use**:

- Periodic data with both trends and seasonal patterns
- Control for unit trends and season-of-year effects
- At least Q + 2 pre-treatment observations per unit (where Q is the number of
  seasons per year)
- For each unit, every season that appears in the post-treatment period must also
  appear in its pre-treatment period (season-coverage condition).

**What it does**:

Combines detrending with seasonal adjustment.

**Supported seasonal periods (Q parameter)**:

- **Q=4** (default): Quarterly data (4 seasons per year)
- **Q=12**: Monthly data (12 seasons per year)
- **Q=52**: Weekly data (52 seasons per year)

**Examples**:

Quarterly data (Q=4, default):

.. code-block:: python

   results = lwdid(
       data_q, y='sales', d='treated', ivar='store',
       tvar=['year', 'quarter'],
       post='post', rolling='detrendq'
   )

Monthly data (Q=12):

.. code-block:: python

   results = lwdid(
       data_m, y='sales', d='treated', ivar='store',
       tvar='time',
       post='post', rolling='detrendq',
       Q=12,
       season_var='month'
   )

Weekly data (Q=52):

.. code-block:: python

   results = lwdid(
       data_w, y='sales', d='treated', ivar='store',
       tvar='time',
       post='post', rolling='detrendq',
       Q=52,
       season_var='week'
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

HC0 (White's Heteroskedasticity-Consistent)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**When to use**:

- Heteroskedasticity is suspected
- Sample size is large

**Specification**:

.. code-block:: python

   results = lwdid(..., vce='hc0')

**Note**: HC0 is the original White (1980) heteroskedasticity-consistent estimator
without finite-sample adjustments. Tends to underestimate standard errors in small
samples.

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

Note: ``'hc1'`` and ``'robust'`` are aliases for the same estimator. HC1 applies
a degrees-of-freedom correction (n/(n-k)) to HC0.

HC2 (Leverage-Adjusted)
~~~~~~~~~~~~~~~~~~~~~~~~

**When to use**:

- Heteroskedasticity is suspected
- Sample size is small to moderate
- Observations have varying leverage

**Specification**:

.. code-block:: python

   results = lwdid(..., vce='hc2')

**Note**: HC2 divides squared residuals by (1 - h_ii) where h_ii is the diagonal
of the hat matrix. Provides better performance than HC1 when leverage varies
across observations.

HC3 (Small-Sample Adjusted Robust)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**When to use**:

- Heteroskedasticity is suspected
- Sample size is small or moderate

**Specification**:

.. code-block:: python

   results = lwdid(..., vce='hc3')

**Advantage**: HC3 divides squared residuals by (1 - h_ii)^2, providing better
finite-sample performance than HC1 and HC2 in small samples.

HC4 (High-Leverage Adjusted)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**When to use**:

- Heteroskedasticity is suspected
- Data contains high-leverage observations

**Specification**:

.. code-block:: python

   results = lwdid(..., vce='hc4')

**Note**: HC4 uses an adaptive exponent that increases the adjustment for
observations with high leverage. Recommended when the design matrix contains
potentially influential observations.

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

**Bootstrap** (default):

.. code-block:: python

   results = lwdid(..., ri=True, ri_method='bootstrap')

With-replacement resampling.

**Permutation**:

.. code-block:: python

   results = lwdid(..., ri=True, ri_method='permutation')

Fisher randomization inference without replacement.

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

Never-Treated Unit Data Preparation
------------------------------------

In staggered adoption designs, never-treated units serve as the control group.
Proper encoding of never-treated units is essential for correct estimation.

Valid Never-Treated Encodings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``gvar`` column can encode never-treated units using any of these values:

1. **Zero (0)**: Common in Stata and other software

   .. code-block:: python

      data['gvar'] = data['gvar'].fillna(0)  # Convert NaN to 0

2. **Positive infinity (np.inf)**: Represents "treated at infinity"

   .. code-block:: python

      import numpy as np
      data.loc[data['never_treated'], 'gvar'] = np.inf

3. **NaN/NA/None**: Missing treatment time indicates never-treated

   .. code-block:: python

      # NaN is automatically recognized as never-treated
      data['gvar'] = data['first_treat_year']  # NaN for never-treated

All three encodings are equivalent and produce identical results.

Converting from Stata Data
~~~~~~~~~~~~~~~~~~~~~~~~~~

When importing data from Stata, never-treated units may be encoded differently:

.. code-block:: python

   import pandas as pd
   import numpy as np

   # Load Stata data
   data = pd.read_stata('staggered_data.dta')

   # Common Stata encodings for never-treated:
   # - Missing values (.) → automatically become NaN in pandas
   # - Zero (0) → recognized as never-treated
   # - Large values (9999) → need manual conversion

   # Convert custom encoding to standard format
   data['gvar'] = data['gvar'].replace({9999: 0, -1: 0})

   # Verify never-treated identification
   from lwdid.validation import is_never_treated
   unit_gvar = data.groupby('unit')['gvar'].first()
   n_nt = unit_gvar.apply(is_never_treated).sum()
   print(f"Never-treated units: {n_nt}")

Checking Never-Treated Status
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the ``is_never_treated()`` function to verify encoding:

.. code-block:: python

   from lwdid.validation import is_never_treated
   import numpy as np

   # Check individual values
   print(is_never_treated(0))        # True
   print(is_never_treated(np.inf))   # True
   print(is_never_treated(np.nan))   # True
   print(is_never_treated(2005))     # False

   # Check DataFrame
   unit_gvar = data.groupby('unit')['gvar'].first()
   nt_mask = unit_gvar.apply(is_never_treated)

   print(f"Total units: {len(unit_gvar)}")
   print(f"Never-treated: {nt_mask.sum()}")
   print(f"Treated: {(~nt_mask).sum()}")

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Issue: No never-treated units found**

.. code-block:: python

   # Check gvar values
   print(data['gvar'].unique())

   # If using custom encoding (e.g., 9999 for never-treated)
   data['gvar'] = data['gvar'].replace({9999: 0})

**Issue: Negative gvar values**

.. code-block:: python

   # Negative values are invalid
   if (data['gvar'] < 0).any():
       print("Warning: Negative gvar values found")
       # Convert to valid encoding
       data.loc[data['gvar'] < 0, 'gvar'] = 0

**Issue: gvar varies within unit**

.. code-block:: python

   # gvar must be time-invariant
   gvar_varies = data.groupby('unit')['gvar'].nunique() > 1
   if gvar_varies.any():
       print(f"Units with varying gvar: {gvar_varies.sum()}")
       # Use first observation's gvar
       first_gvar = data.groupby('unit')['gvar'].first()
       data['gvar'] = data['unit'].map(first_gvar)

Best Practices
--------------

Choosing Transformation Method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Start with ``demean``** for annual data
2. **Use ``detrend``** if pre-treatment trends differ across units
3. **Use ``demeanq``/``detrendq``** for quarterly data with seasonality
4. **Check pre-treatment period requirements** (:math:`T_0 \geq 1` for demean,
   :math:`T_0 \geq 2` for detrend)

Choosing Variance Estimator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Default (``vce=None``)**: Use if you believe homoskedasticity holds
2. **``vce='hc3'``**: Recommended for small to moderate sample sizes when you suspect heteroskedasticity
3. **``vce='cluster'``**: Use if errors are clustered
4. **Randomization inference**: Use as robustness check or when normality is doubtful

Sample Size Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Small samples (N < 50):** Use exact t-based inference (``vce=None``) under
  CLM assumptions (normality and homoskedasticity). Minimum requirement is
  :math:`N \geq 3` units. Having at least 10 units improves stability.
- **Moderate to large samples (N >= 50):** Use heteroskedasticity-robust
  standard errors (``vce='hc3'``) for valid asymptotic inference.
- **Large samples with controls:** Consider IPWRA (``estimator='ipwra'``) for
  doubly robust estimation when functional form assumptions are uncertain.
- **Time dimension:** Method works best with large T (many time periods), where
  the central limit theorem across time supports normality of the transformed
  outcome.
- **Asymptotic theory:** Lee and Wooldridge (2025) develops the theoretical
  foundation for large-sample inference, supporting HC0-HC4 and cluster-robust
  standard errors.

Estimator Selection (Large-Sample Common Timing)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For large cross-sectional samples in common timing settings, multiple estimators
are available beyond regression adjustment. Lee and Wooldridge (2025) shows
that the rolling transformation approach enables application of various treatment
effects estimators, including doubly robust methods.

**Available estimators for common timing:**

- ``'ra'`` (default): Regression adjustment via OLS. Consistent when the outcome
  model is correctly specified. Efficient under correct specification.

- ``'ipw'``: Inverse probability weighting. Consistent when the propensity score
  model is correctly specified. Requires ``controls`` parameter.

- ``'ipwra'``: Doubly robust estimator combining regression and IPW. Consistent
  when either the outcome model or propensity score is correctly specified.
  Recommended when functional form assumptions are uncertain.

- ``'psm'``: Propensity score matching. Matches treated to control units based
  on propensity scores. Requires adequate sample sizes for matching.

**Example (IPWRA in common timing):**

.. code-block:: python

   results = lwdid(
       data, y='outcome', d='treated', ivar='unit', tvar='year',
       post='post', rolling='demean',
       estimator='ipwra',           # Doubly robust estimator
       controls=['x1', 'x2'],       # Controls for outcome and PS models
       vce='hc3'                    # Robust standard errors
   )

**Estimator selection guidelines:**

1. **Start with RA** for simplicity when N is moderate
2. **Use IPWRA** as a robustness check or when model misspecification is a concern
3. **Use IPW/PSM** when propensity score weighting/matching is preferred

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

- ``demean``/``demeanq`` require :math:`T_0 \geq 1`
- ``detrend``/``detrendq`` require :math:`T_0 \geq 2`
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

Staggered Adoption Design
-------------------------

When units are treated at different times, use the staggered adoption framework.

.. note::

   **Staggered mode limitations**: The ``demeanq`` and ``detrendq``
   transformations are only available for common timing designs. In staggered
   mode, only ``demean`` and ``detrend`` are supported.

Basic Usage
~~~~~~~~~~~

Instead of ``post`` (common timing), specify ``gvar`` (first treatment period):

.. code-block:: python

   results = lwdid(
       data,
       y='outcome',
       ivar='unit',
       tvar='year',
       gvar='first_treat_year',  # First treatment period for each unit
       rolling='demean',
       aggregate='overall',      # Aggregate to overall effect
   )

Staggered-Specific Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**gvar** : str
    Column name indicating the first treatment period for each unit.
    Units with ``gvar = inf`` or ``gvar = NaN`` are never treated.

**aggregate** : {'none', 'cohort', 'overall'}
    - ``'none'``: Return (g,r)-specific effects only
    - ``'cohort'``: Average effects within each cohort (default)
    - ``'overall'``: Weighted average across cohorts

**control_group** : {'not_yet_treated', 'never_treated'}
    - ``'not_yet_treated'``: Use never-treated plus not-yet-treated units (default)
    - ``'never_treated'``: Use only never-treated units

**estimator** : {'ra', 'ipw', 'ipwra', 'psm'}
    - ``'ra'``: Regression adjustment (default)
    - ``'ipw'``: Inverse probability weighting
    - ``'ipwra'``: Doubly robust (recommended)
    - ``'psm'``: Propensity score matching

**exclude_pre_periods** : int, default 0
    Number of periods immediately before treatment to exclude from pre-treatment
    calculations. Used for robustness checks when the no-anticipation assumption
    may be violated.

    For cohort g, the pre-treatment mean/trend is computed using periods
    {T_min, ..., g-1-exclude_pre_periods} instead of {T_min, ..., g-1}.

    See Lee & Wooldridge (2025) Section 6 for methodological details.

Robustness to No-Anticipation Violations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the no-anticipation assumption may be violated (e.g., units adjust behavior
before formal treatment), you can use the ``exclude_pre_periods`` parameter to
perform robustness checks.

**Example**:

.. code-block:: python

   # Baseline estimation
   result_baseline = lwdid(
       data, y='outcome', ivar='unit', tvar='year',
       gvar='cohort', rolling='demean'
   )

   # Robustness check: exclude 1 period before treatment
   result_robust1 = lwdid(
       data, y='outcome', ivar='unit', tvar='year',
       gvar='cohort', rolling='demean',
       exclude_pre_periods=1
   )

   # Robustness check: exclude 2 periods before treatment
   result_robust2 = lwdid(
       data, y='outcome', ivar='unit', tvar='year',
       gvar='cohort', rolling='demean',
       exclude_pre_periods=2
   )

   # Compare results
   print(f"Baseline ATT: {result_baseline.att_overall:.4f}")
   print(f"Exclude 1 period: {result_robust1.att_overall:.4f}")
   print(f"Exclude 2 periods: {result_robust2.att_overall:.4f}")

**Minimum pre-treatment period requirements**:

- ``demean``: At least 1 pre-treatment period remaining after exclusion
- ``detrend``: At least 2 pre-treatment periods remaining after exclusion
- ``demeanq``: At least Q+1 pre-treatment periods remaining (Q = number of seasons)
- ``detrendq``: At least Q+2 pre-treatment periods remaining

If any cohort has insufficient pre-treatment periods after applying
``exclude_pre_periods``, an ``InsufficientPrePeriodsError`` is raised.

Accessing Staggered Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Overall weighted effect
   results.att_overall       # ATT estimate
   results.se_overall        # Standard error

   # Cohort-specific effects
   df_cohorts = results.att_by_cohort

   # Cohort-time specific effects
   df_gt = results.att_by_cohort_time

   # Event study visualization
   fig, ax = results.plot_event_study()

Control Group Strategies
~~~~~~~~~~~~~~~~~~~~~~~~

**Not-yet-treated (default)**:

Uses both never-treated units and units that will be treated in future periods.
More efficient but requires stronger no-anticipation assumption.

.. code-block:: python

   results = lwdid(..., control_group='not_yet_treated')

**Never-treated only**:

Uses only units that are never treated during the observation period.
More robust but potentially less efficient.

.. code-block:: python

   results = lwdid(..., control_group='never_treated')

Estimator Selection
~~~~~~~~~~~~~~~~~~~

**Regression Adjustment (RA)**:

- Consistent when conditional mean is correctly specified
- Default estimator, efficient under correct specification

**Doubly Robust (IPWRA)**:

- Consistent if either outcome model or propensity score is correct
- Recommended when functional form assumptions are uncertain

.. code-block:: python

   results = lwdid(..., estimator='ipwra')

**Inverse Probability Weighting (IPW)**:

- Weights by propensity score
- Consistent when propensity score is correctly specified

**Propensity Score Matching (PSM)**:

- Matches treated to control units by propensity score
- Use when sample sizes are adequate for matching

Staggered Example
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   from lwdid import lwdid

   # Load castle law data
   data = pd.read_csv('castle.csv')

   # Estimate with staggered design
   results = lwdid(
       data,
       y='l_homicide',
       ivar='state',
       tvar='year',
       gvar='effyear',
       rolling='demean',
       aggregate='overall',
       control_group='not_yet_treated',
       estimator='ipwra',
   )

   # Display results
   print(results.summary())

   # Event study plot
   fig, ax = results.plot_event_study(
       include_pre_treatment=True,
       show_ci=True
   )

Pre-treatment Dynamics and Parallel Trends Testing
--------------------------------------------------

Pre-treatment dynamics analysis allows you to assess the validity of the parallel
trends assumption by estimating treatment effects in pre-treatment periods. Under
the null hypothesis of parallel trends, these pre-treatment effects should be zero.

This feature implements the methodology from Lee & Wooldridge (2025) Appendix D,
using rolling transformations that look forward to future pre-treatment periods
rather than backward.

Why Pre-treatment Dynamics Matter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The parallel trends assumption is fundamental to difference-in-differences
identification. While this assumption cannot be directly tested (it concerns
counterfactual outcomes), examining pre-treatment dynamics provides indirect
evidence:

- **Under parallel trends**: Pre-treatment ATT estimates should be approximately
  zero (within sampling variation)
- **Violation of parallel trends**: Systematic non-zero pre-treatment effects
  suggest the control group may not provide a valid counterfactual

The anchor point convention sets the effect at event time e = -1 (the period
immediately before treatment) to exactly zero, providing a reference point for
interpreting other pre-treatment effects.

Basic Usage
~~~~~~~~~~~

Enable pre-treatment dynamics by setting ``include_pretreatment=True``:

.. code-block:: python

   from lwdid import lwdid

   results = lwdid(
       data,
       y='outcome',
       ivar='unit',
       tvar='year',
       gvar='first_treat',
       rolling='demean',
       aggregate='cohort',
       include_pretreatment=True,    # Enable pre-treatment estimation
       pretreatment_test=True,       # Run parallel trends test (default)
       pretreatment_alpha=0.05,      # Significance level (default)
   )

   # View complete summary including pre-treatment results
   print(results.summary())

Accessing Pre-treatment Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Pre-treatment ATT estimates as DataFrame
   pre_df = results.att_pre_treatment
   print(pre_df)
   # Columns: cohort, period, event_time, att, se, ci_lower, ci_upper,
   #          t_stat, pvalue, n_treated, n_control, is_anchor, rolling_window_size

   # Parallel trends test results
   pt = results.parallel_trends_test
   print(f"Joint F-statistic: {pt.joint_f_stat:.4f}")
   print(f"Joint p-value: {pt.joint_pvalue:.4f}")
   print(f"Reject parallel trends: {pt.reject_null}")
   print(f"Number of pre-treatment periods tested: {pt.n_periods}")

Interpreting the Parallel Trends Test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The parallel trends test performs a joint F-test of the null hypothesis that all
pre-treatment ATT estimates (excluding the anchor point) are jointly zero:

- **H0**: All pre-treatment ATT = 0 (parallel trends holds)
- **H1**: At least one pre-treatment ATT ≠ 0 (parallel trends violated)

**Interpretation guidelines**:

- ``reject_null=False``: No statistical evidence against parallel trends. This
  does not prove parallel trends holds, but the data are consistent with it.
- ``reject_null=True``: Evidence of pre-treatment differences. Consider:

  - Using ``detrend`` instead of ``demean`` to allow for heterogeneous trends
  - Examining which cohorts or periods show violations
  - Investigating potential anticipation effects or data issues

Event Study Visualization with Pre-treatment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The event study plot can display both pre-treatment and post-treatment effects:

.. code-block:: python

   # Plot with pre-treatment effects
   fig, ax = results.plot_event_study(
       include_pre_treatment=True,      # Show pre-treatment effects
       pre_treatment_color='gray',      # Color for pre-treatment points
       post_treatment_color='blue',     # Color for post-treatment points
       show_anchor_line=True,           # Vertical line at e=-1
       title='Event Study with Pre-treatment Dynamics',
       ylabel='Treatment Effect',
   )
   fig.savefig('event_study_pretreatment.png', dpi=300)

The plot shows:

- Pre-treatment effects (e < -1) in gray with confidence intervals
- Anchor point (e = -1) at zero
- Post-treatment effects (e ≥ 0) in blue with confidence intervals
- Optional vertical dashed line at the anchor point

Complete Workflow Example
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   from lwdid import lwdid

   # Load data
   data = pd.read_csv('castle.csv')

   # Step 1: Estimate with pre-treatment dynamics
   results = lwdid(
       data,
       y='l_homicide',
       ivar='state',
       tvar='year',
       gvar='effyear',
       rolling='demean',
       aggregate='cohort',
       include_pretreatment=True,
       pretreatment_test=True,
   )

   # Step 2: Check parallel trends test
   pt = results.parallel_trends_test
   if pt.reject_null:
       print("Warning: Evidence against parallel trends!")
       print(f"F-stat: {pt.joint_f_stat:.3f}, p-value: {pt.joint_pvalue:.4f}")
       print("Consider using rolling='detrend' to allow heterogeneous trends")
   else:
       print("No evidence against parallel trends")
       print(f"F-stat: {pt.joint_f_stat:.3f}, p-value: {pt.joint_pvalue:.4f}")

   # Step 3: Examine pre-treatment effects
   pre_df = results.att_pre_treatment
   non_anchor = pre_df[~pre_df['is_anchor']]
   print("\nPre-treatment effects (excluding anchor):")
   print(non_anchor[['event_time', 'att', 'se', 'pvalue']].to_string(index=False))

   # Step 4: Visualize
   fig, ax = results.plot_event_study(
       include_pre_treatment=True,
       title='Castle Doctrine: Event Study with Pre-treatment',
   )
   fig.savefig('castle_event_study.png', dpi=300, bbox_inches='tight')

   # Step 5: Report main results
   print(f"\nOverall ATT: {results.att_overall:.4f} (SE: {results.se_overall:.4f})")

Transformation Methods for Pre-treatment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pre-treatment dynamics support both ``demean`` and ``detrend`` transformations:

**Demean (default)**:

Uses rolling demeaning with future pre-treatment periods. For period t < g:

.. math::

   \dot{Y}_{itg} = Y_{it} - \frac{1}{g-1-t} \sum_{q=t+1}^{g-1} Y_{iq}

**Detrend**:

Uses rolling OLS detrending with future pre-treatment periods. Fits a linear
trend using periods {t+1, ..., g-1} and computes the residual.

Choose ``detrend`` when you suspect cohorts have different pre-treatment trends.

Further Reading
---------------

- :doc:`quickstart` - Quick start tutorial
- :doc:`methodological_notes` - Theoretical background
- :doc:`api/index` - Complete API reference
- :doc:`api/staggered` - Staggered module API
- :doc:`examples/index` - Usage examples

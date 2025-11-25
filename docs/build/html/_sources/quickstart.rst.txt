Quick Start
===========

This guide demonstrates basic usage of the lwdid package for difference-in-differences
estimation with small cross-sectional samples.

Basic Example
-------------

Simplest Usage
~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   from lwdid import lwdid

   # Load data
   data = pd.read_csv('smoking.csv')

   # Run estimation
   results = lwdid(
       data,
       y='lcigsale',      # outcome variable
       d='d',             # treatment indicator (0 = control, 1 = treated)
       ivar='state',      # unit identifier
       tvar='year',       # time variable
       post='post',       # post-treatment indicator
       rolling='demean',  # transformation method
   )

   # View results
   print(results.summary())

Output
~~~~~~

.. code-block:: text

   ================================================================================
                             lwdid Results
   ================================================================================
   Transformation: demean
   Variance Type: OLS (Homoskedastic)
   Dependent Variable: lcigsale

   Number of observations: 39
   Number of treated units: 1
   Number of control units: 38
   Pre-treatment periods: 19 (K=19)
   Post-treatment periods: 20 to end (tpost1=20)

   --------------------------------------------------------------------------------
   Average Treatment Effect on the Treated
   --------------------------------------------------------------------------------
   ATT:           -0.4222
   Std. Err.:      0.1208  (ols)
   t-stat:          -3.49
   P>|t|:           0.001
   df:                 37
   [95% Conf. Interval]:   -0.6669    -0.1774
   ================================================================================

   === Period-by-period post-treatment effects ===
    period tindex      beta       se  ci_lower  ci_upper     tstat     pval  N
   average      - -0.422175 0.120800 -0.666938 -0.177412 -3.494837 0.001249 39
      1989     20 -0.168195 0.095788 -0.362280  0.025890 -1.755906 0.087380 39
      1990     21 -0.187484 0.111675 -0.413759  0.038792 -1.678829 0.101613 39
      1991     22 -0.302473 0.116699 -0.538927 -0.066018 -2.591908 0.013584 39
      1992     23 -0.310131 0.128026 -0.569536 -0.050727 -2.422417 0.020432 39
   ... (8 more periods)

   Use results.att_by_period to view all period-specific estimates

Key Parameters
--------------

Required Parameters
~~~~~~~~~~~~~~~~~~~

- ``data``: pandas DataFrame in long format (panel data)
- ``y``: Outcome variable column name (string)
- ``d``: Treatment indicator column name (0=control, non-zero=treated)
- ``ivar``: Unit identifier column name
- ``tvar``: Time variable column name (string for annual data, list for quarterly data)
- ``post``: Post-treatment indicator column name (1=post-treatment, 0=pre-treatment)
- ``rolling``: Transformation method, options:

  - ``'demean'``: Standard DiD (unit fixed effects) — requires T₀ ≥ 1
  - ``'detrend'``: DiD with unit-specific linear trends — requires T₀ ≥ 2
  - ``'demeanq'``: Quarterly data with seasonal effects — requires T₀ ≥ 1
  - ``'detrendq'``: Quarterly data with trends and seasonal effects — requires T₀ ≥ 2

.. note::

   **Pre-treatment period requirements**:

   - ``demean/demeanq``: Requires at least 1 pre-treatment period (T₀ ≥ 1)
   - ``detrend/detrendq``: Requires at least 2 pre-treatment periods (T₀ ≥ 2)

   The ``detrend`` method estimates unit-specific linear trends using pre-treatment
   data via regression yᵢₜ = Aᵢ + Bᵢ·t + εᵢₜ, which requires at least 2 observations
   to identify both intercept and slope. This is a transformation step as described
   in Lee and Wooldridge (2025) Procedure 3.1; statistical inference is conducted in
   the subsequent cross-sectional regression.

Common Options
--------------

Heteroskedasticity-Robust Standard Errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   results = lwdid(
       data, 'lcigsale', 'd', 'state', 'year', 'post', 'detrend',
       vce='hc3'  # Use HC3 robust standard errors
   )

Randomization Inference
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   results = lwdid(
       data, 'lcigsale', 'd', 'state', 'year', 'post', 'detrend',
       ri=True,        # Enable randomization inference
       rireps=1000,    # Number of permutations
       seed=42         # Random seed
   )
   print(f"RI p-value: {results.ri_pvalue:.4f}")

Adding Control Variables
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Note: Controls must be time-invariant (constant within each unit)
   results = lwdid(
       data, 'outcome', 'treated', 'unit', 'year', 'post', 'detrend',
       controls=['baseline_x1', 'baseline_x2']  # Time-invariant controls
   )

Quarterly Data
~~~~~~~~~~~~~~

.. code-block:: python

   results = lwdid(
       data_q, 'outcome', 'd', 'unit',
       tvar=['year', 'quarter'],  # Composite time variable
       post='post',
       rolling='detrendq'         # Quarterly detrending
   )

Results Object
--------------

``lwdid()`` returns a ``LWDIDResults`` object containing:

Core Attributes
~~~~~~~~~~~~~~~

.. code-block:: python

   results.att           # ATT estimate
   results.se_att        # Standard error
   results.t_stat        # t-statistic
   results.pvalue        # p-value
   results.ci_lower      # 95% confidence interval lower bound
   results.ci_upper      # 95% confidence interval upper bound
   results.df_inference  # Degrees of freedom used for t-based inference

Sample Information
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   results.nobs          # Number of observations in the regression sample
   results.n_treated     # Number of treated units in the regression sample
   results.n_control     # Number of control units in the regression sample

Period-Specific Effects
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   results.att_by_period  # DataFrame: treatment effects by period

Methods
~~~~~~~

.. code-block:: python

   results.summary()              # Print formatted results
   results.plot()                 # Visualize time series
   results.to_excel('output.xlsx')  # Export to Excel
   results.to_latex('table.tex')    # Export to LaTeX

Data Format Requirements
------------------------

Panel Data Structure
~~~~~~~~~~~~~~~~~~~~

Data must be in **long format** (one row per observation):

.. code-block:: text

   ivar    tvar    y       d    post
   1       2000    10.5    1    0
   1       2001    10.8    1    0
   1       2002    11.2    1    1    <- Unit 1 is treated (d=1), receives treatment in 2002 (post=1)
   2       2000    9.3     0    0
   2       2001    9.5     0    0
   2       2002    9.7     0    0    <- Unit 2 is control (d=0), never treated

.. note::

   **Important**: The ``d`` column must be a time-invariant treatment group indicator (Dᵢ):

   - ``d = 1`` indicates the unit belongs to the treatment group (constant across all periods)
   - ``d = 0`` indicates the unit belongs to the control group (constant across all periods)
   - ``d`` must NOT vary over time. Do not pass a time-varying treatment indicator Wᵢₜ = Dᵢ × postₜ as the ``d`` parameter

Key Requirements
~~~~~~~~~~~~~~~~

1. **Panel structure**:

   - Data must be in long format: one row per unit-period observation
   - Each (unit, period) combination must be unique (no duplicate rows)
   - After constructing the internal time index (``tindex``) from ``tvar``, the
     set of time indices must form a continuous sequence with no gaps
   - Panels may be balanced or unbalanced across units; some units may have
     fewer periods, but each unit must satisfy the pre-treatment requirements
     for the chosen ``rolling`` method (see above)

2. **Common treatment timing**: All treated units must start treatment in the same
   period (``post`` is a pure function of time)
3. **Treatment persistence**: Once units enter the post-treatment regime
   (``post`` switches from 0 to 1), they remain in it for all subsequent periods
   (no policy reversals)

Next Steps
----------

- See :doc:`user_guide` for detailed usage
- Browse :doc:`examples/index` for complete examples
- Read :doc:`api/index` for API details

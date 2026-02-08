Quick Start
===========

This guide demonstrates basic usage of the lwdid package for difference-in-differences
estimation with rolling transformations. The package supports three scenarios:

- **Common timing (small sample)**: Exact t-based inference under CLM assumptions
- **Common timing (large sample)**: Asymptotic inference with robust standard errors
- **Staggered adoption**: Units treated at different times

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

  - ``'demean'``: Standard DiD (unit fixed effects) — requires :math:`T_0 \geq 1`
  - ``'detrend'``: DiD with unit-specific linear trends — requires :math:`T_0 \geq 2`
  - ``'demeanq'``: Quarterly data with seasonal effects — requires :math:`T_0 \geq 1`
  - ``'detrendq'``: Quarterly data with trends and seasonal effects — requires :math:`T_0 \geq 2`

.. note::

   **Pre-treatment period requirements**:

   - ``demean/demeanq``: Requires at least 1 pre-treatment period (:math:`T_0 \geq 1`)
   - ``detrend/detrendq``: Requires at least 2 pre-treatment periods (:math:`T_0 \geq 2`)

   The ``demean`` method subtracts the unit-specific pre-treatment mean, as
   described in Lee and Wooldridge (2026). The ``detrend`` method estimates
   unit-specific linear trends using pre-treatment data via regression
   :math:`Y_{it} = A_i + B_i t + \varepsilon_{it}`, which requires at least 2
   observations to identify both intercept and slope (Lee and Wooldridge, 2026).

Common Options
--------------

Small-Sample Exact Inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For settings with small numbers of treated or control units, exact t-based inference
is available under classical linear model assumptions:

.. code-block:: python

   results = lwdid(
       data, 'lcigsale', 'd', 'state', 'year', 'post', 'detrend',
       vce=None  # Default: exact inference under CLM assumptions
   )

Large-Sample Inference
~~~~~~~~~~~~~~~~~~~~~~

For large cross-sectional samples, heteroskedasticity-robust standard errors
provide valid asymptotic inference:

.. code-block:: python

   results = lwdid(
       data, 'lcigsale', 'd', 'state', 'year', 'post', 'detrend',
       vce='hc3'  # HC3 robust standard errors for large samples
   )

Doubly Robust Estimation
~~~~~~~~~~~~~~~~~~~~~~~~

The IPWRA estimator combines regression adjustment with propensity score weighting,
providing consistent estimates when either the outcome model or propensity score
model is correctly specified:

.. code-block:: python

   results = lwdid(
       data, 'outcome', 'd', 'unit', 'year', 'post', 'demean',
       estimator='ipwra',
       controls=['x1', 'x2'],
       vce='hc3'
   )

.. note::

   IPWRA is particularly recommended for large-sample settings (N >= 50) where
   functional form assumptions are uncertain. See :doc:`user_guide` for detailed
   estimator selection guidelines.

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
   # For time-varying variables, create time-invariant versions first
   
   # Example: Create pre-treatment mean for time-varying controls
   data_prep = data.copy()
   for var in ['retprice']:
       pre_mean = data[data['post']==0].groupby('state')[var].mean()
       data_prep[f'{var}_pre'] = data_prep['state'].map(pre_mean)
   
   results = lwdid(
       data_prep, 'lcigsale', 'd', 'state', 'year', 'post', 'detrend',
       controls=['retprice_pre']  # Time-invariant controls
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

   **Important**: The ``d`` column must be a time-invariant treatment group indicator (:math:`D_i`):

   - ``d = 1`` indicates the unit belongs to the treatment group (constant across all periods)
   - ``d = 0`` indicates the unit belongs to the control group (constant across all periods)
   - ``d`` must NOT vary over time. Do not pass a time-varying treatment indicator
     :math:`W_{it} = D_i \times post_t` as the ``d`` parameter

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

Staggered Adoption
------------------

When units are treated at different times, use the ``gvar`` parameter:

.. code-block:: python

   import pandas as pd
   from lwdid import lwdid

   # Load staggered adoption data
   data = pd.read_csv('castle.csv')

   # Estimate with staggered design
   # Create gvar (first treatment year, NaN -> 0 for never-treated)
   data['gvar'] = data['effyear'].fillna(0).astype(int)

   results = lwdid(
       data,
       y='lhomicide',            # outcome variable
       ivar='sid',               # unit identifier (state ID)
       tvar='year',              # time variable
       gvar='gvar',              # first treatment period (0 = never treated)
       rolling='demean',         # transformation method
       aggregate='overall',      # aggregate to overall effect
       control_group='not_yet_treated',  # control group strategy
   )

   # View results
   print(results.summary())

   # Event study visualization
   fig, ax = results.plot_event_study()

Staggered Results
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Overall weighted effect
   results.att_overall       # ATT estimate
   results.se_overall        # Standard error

   # Cohort-specific effects
   results.att_by_cohort     # DataFrame with cohort-level effects

   # Cohort-time specific effects
   results.att_by_cohort_time  # DataFrame with (g,r)-specific effects

Key Staggered Parameters
~~~~~~~~~~~~~~~~~~~~~~~~

- ``gvar``: Column indicating first treatment period (0, inf, or NaN for never-treated)
- ``aggregate``: ``'none'``, ``'cohort'``, or ``'overall'``
- ``control_group``: ``'not_yet_treated'`` or ``'never_treated'``
- ``estimator``: ``'ra'``, ``'ipw'``, ``'ipwra'``, or ``'psm'``

Next Steps
----------

- See :doc:`user_guide` for detailed usage including staggered designs
- Browse :doc:`examples/index` for complete examples
- Read :doc:`api/index` for API details
- Read :doc:`api/staggered` for staggered module details

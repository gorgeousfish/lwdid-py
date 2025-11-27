Core Module (core)
==================

The core module provides the main user-facing function for difference-in-differences
estimation.

Main Function
-------------

.. autofunction:: lwdid.lwdid

Examples
--------

Basic Usage
~~~~~~~~~~~

Simplest DiD estimation with default settings:

.. code-block:: python

   from lwdid import lwdid
   import pandas as pd

   # Load data
   data = pd.read_csv('smoking.csv')

   # Run estimation
   results = lwdid(
       data,
       y='lcigsale',      # Outcome variable
       d='d',             # Treatment indicator
       ivar='state',      # Unit ID
       tvar='year',       # Time variable
       post='post',       # Post-treatment indicator
       rolling='demean'   # Transformation method
   )

   # View results
   print(results.summary())
   print(f"ATT: {results.att:.4f}")
   print(f"SE: {results.se_att:.4f}")
   print(f"p-value: {results.pvalue:.4f}")

With Robust Standard Errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using HC3 heteroskedasticity-robust standard errors:

.. code-block:: python

   results = lwdid(
       data, 'lcigsale', 'd', 'state', 'year', 'post', 'detrend',
       vce='hc3'  # HC3 robust standard errors
   )

With Randomization Inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Adding randomization inference for non-parametric testing:

.. code-block:: python

   results = lwdid(
       data, 'lcigsale', 'd', 'state', 'year', 'post', 'demean',
       ri=True,           # Enable randomization inference
       rireps=1000,       # Number of permutations
       ri_method='permutation',  # Use permutation (recommended)
       seed=42            # For reproducibility
   )

   print(f"t-based p-value: {results.pvalue:.4f}")
   print(f"RI p-value: {results.ri_pvalue:.4f}")

With Control Variables
~~~~~~~~~~~~~~~~~~~~~~~

Including time-invariant control variables:

.. code-block:: python

   # Prepare time-invariant controls
   # Use pre-treatment mean for time-varying variables
   data_prep = data.copy()
   for var in ['retprice', 'beer']:
       pre_mean = data[data['post']==0].groupby('state')[var].mean()
       data_prep[f'{var}_pre'] = data_prep['state'].map(pre_mean)
   
   results = lwdid(
       data_prep, 'lcigsale', 'd', 'state', 'year', 'post', 'detrend',
       controls=['retprice_pre', 'beer_pre'],  # Time-invariant controls
       vce='hc3'
   )

.. warning::

   Control variables must be time-invariant (constant within each unit).
   If you have time-varying variables, you must first aggregate them to
   create unit-level constants. Common approaches:
   
   - Pre-treatment mean: ``data[data['post']==0].groupby('unit')[var].mean()``
   - First period value: ``data.groupby('unit')[var].first()``
   - Overall mean: ``data.groupby('unit')[var].mean()``

Cluster-Robust Standard Errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When errors are correlated within clusters:

.. code-block:: python

   results = lwdid(
       data, 'outcome', 'd', 'unit', 'year', 'post', 'demean',
       vce='cluster',
       cluster_var='state'  # Cluster by state
   )

Quarterly Data
~~~~~~~~~~~~~~

Handling quarterly data with seasonal patterns:

.. code-block:: python

   # Data has columns: unit, year, quarter, outcome, d, post
   results = lwdid(
       data_q,
       y='sales',
       d='d',
       ivar='store',
       tvar=['year', 'quarter'],  # Composite time variable
       post='post',
       rolling='detrendq'  # Quarterly detrending with seasonality
   )

Complete Example with All Options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   results = lwdid(
       data,
       y='outcome',
       d='d',
       ivar='unit',
       tvar='year',
       post='post_treatment',
       rolling='detrend',
       controls=['baseline_x1', 'baseline_x2'],
       vce='hc3',
       ri=True,
       rireps=2000,
       ri_method='permutation',
       seed=12345
   )

   # Access results
   print(f"ATT: {results.att:.3f} ({results.ci_lower:.3f}, {results.ci_upper:.3f})")
   print(f"t-stat: {results.t_stat:.3f}, p-value: {results.pvalue:.4f}")
   print(f"RI p-value: {results.ri_pvalue:.4f}")
   print(f"N = {results.nobs}, df = {results.df_inference}")

   # Export results
   results.to_excel('results.xlsx')
   results.plot()

See Also
--------

- :class:`lwdid.LWDIDResults` - Results object returned by lwdid()
- :doc:`../user_guide` - Comprehensive usage guide
- :doc:`../quickstart` - Quick start tutorial

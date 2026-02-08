Exceptions Module (exceptions)
===============================

The exceptions module defines custom exception classes for clear error reporting
in the ``lwdid`` package.

.. automodule:: lwdid.exceptions
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The package uses custom exceptions to provide informative error messages when
data or parameters do not meet requirements. All custom exceptions inherit from
``LWDIDError``, making it easy to catch all package-specific errors.

Exception Hierarchy
-------------------

.. code-block:: text

   LWDIDError (base class)
   ├── InvalidParameterError
   │   ├── InvalidRollingMethodError
   │   └── InvalidVCETypeError
   ├── InsufficientDataError
   │   ├── NoTreatedUnitsError
   │   ├── NoControlUnitsError
   │   ├── InsufficientPrePeriodsError
   │   ├── InsufficientQuarterDiversityError
   │   └── NoNeverTreatedError
   ├── InvalidStaggeredDataError
   ├── TimeDiscontinuityError
   ├── MissingRequiredColumnError
   ├── RandomizationError
   ├── VisualizationError
   ├── UnbalancedPanelError
   └── AggregationError
       ├── InvalidAggregationError
       └── InsufficientCellSizeError

Exception Classes
-----------------

LWDIDError
~~~~~~~~~~

**Base class** for all lwdid exceptions.

**Usage:**

.. code-block:: python

   from lwdid import lwdid
   from lwdid.exceptions import LWDIDError

   try:
       results = lwdid(data, 'y', 'd', 'unit', 'year', 'post', 'demean')
   except LWDIDError as e:
       print(f"LWDID error: {e}")

**When raised:** Never raised directly; use specific subclasses.

InsufficientDataError
~~~~~~~~~~~~~~~~~~~~~

**Raised when:** Data does not meet minimum sample size or period requirements.

**Common causes:**

- Too few units (N < 3)
- Insufficient pre-treatment periods for chosen transformation
- No post-treatment periods
- Empty dataset after filtering

**Example:**

.. code-block:: python

   from lwdid import lwdid
   from lwdid.exceptions import InsufficientDataError

   try:
       results = lwdid(data, 'y', 'd', 'unit', 'year', 'post', 'detrend')
   except InsufficientDataError as e:
       print(f"Insufficient data: {e}")
       # Use different transformation or collect more data

**Typical error messages:**

.. code-block:: text

   InsufficientPrePeriodsError: Insufficient pre-treatment periods for 'detrend'.
   All units must have at least 2 pre-treatment periods (T0 >= 2).
   Found units with fewer periods: ['unit_3', 'unit_7']

.. code-block:: text

   InsufficientDataError: Sample size too small.
   Need at least 3 units for estimation, found 2.

.. code-block:: text

   InsufficientDataError: No post-treatment periods found.
   The 'post' variable is 0 for all observations.

InvalidParameterError
~~~~~~~~~~~~~~~~~~~~~

**Raised when:** Input parameter validation fails.

**Common causes:**

- Invalid ``rolling`` method name (see ``InvalidRollingMethodError``)
- Invalid ``vce`` option (see ``InvalidVCETypeError``)
- ``cluster_var`` missing or incompatible when ``vce='cluster'``
- Treatment indicator or controls not time-invariant
- Non-numeric outcome or control variables
- Time variables not convertible to valid numeric year/quarter values

**Typical error messages (illustrative):**

.. code-block:: text

   InvalidParameterError: rolling() must be one of: demean, detrend, demeanq, detrendq. Got: 'invalid_method'

.. code-block:: text

   InvalidParameterError: vce='cluster' requires cluster_var parameter to be specified.

.. code-block:: text

   InvalidParameterError: Treatment indicator 'd' must be time-invariant (constant within each unit).

InvalidRollingMethodError
~~~~~~~~~~~~~~~~~~~~~~~~~

Specialized subclass of :class:`InvalidParameterError` raised when the
``rolling`` argument does not match one of the supported transformation
methods (``'demean'``, ``'detrend'``, ``'demeanq'``, ``'detrendq'``).

InvalidVCETypeError
~~~~~~~~~~~~~~~~~~~

Specialized subclass of :class:`InvalidParameterError` raised when the
``vce`` argument is not one of ``None``, ``'robust'``, ``'hc0'``,
``'hc1'``, ``'hc2'``, ``'hc3'``, ``'hc4'``, or ``'cluster'``.

InvalidStaggeredDataError
~~~~~~~~~~~~~~~~~~~~~~~~~

**Raised when:** Staggered adoption data validation fails.

**Common causes:**

- ``gvar`` column contains invalid values (negative numbers or non-numeric types)
- No valid treatment cohorts (all units are never-treated)
- ``gvar`` is not time-invariant within units (same unit has different gvar
  values across time periods)

**Valid gvar values:**

- Positive integer: Treatment cohort (first treatment period)
- 0: Never treated
- np.inf: Never treated
- NaN/None: Never treated

**Example:**

.. code-block:: python

   from lwdid import lwdid
   from lwdid.exceptions import InvalidStaggeredDataError

   try:
       results = lwdid(
           data, y='outcome', ivar='unit', tvar='year',
           gvar='first_treat', rolling='demean'
       )
   except InvalidStaggeredDataError as e:
       print(f"Staggered data error: {e}")
       # Check gvar column for invalid values

**Typical error messages:**

.. code-block:: text

   InvalidStaggeredDataError: gvar column contains negative values.

.. code-block:: text

   InvalidStaggeredDataError: No valid treatment cohorts found.
   All units are never-treated.

.. code-block:: text

   InvalidStaggeredDataError: gvar is not time-invariant within unit 'unit_5'.

NoNeverTreatedError
~~~~~~~~~~~~~~~~~~~

**Raised when:** Never-treated units are required but absent.

This is a subclass of :class:`InsufficientDataError` raised in staggered
adoption settings when cohort-level or overall aggregation is requested
but no never-treated units exist in the data.

**Common causes:**

- ``aggregate='cohort'`` specified but no never-treated units
- ``aggregate='overall'`` specified but no never-treated units

**Why never-treated units are required:**

For cohort and overall effect aggregation, different cohorts use different
pre-treatment periods for transformation. Only never-treated units can
serve as a consistent reference across cohorts.

For (g,r)-specific effects with ``aggregate='none'``, not-yet-treated units
can serve as controls, so this exception is not raised.

**Example:**

.. code-block:: python

   from lwdid import lwdid
   from lwdid.exceptions import NoNeverTreatedError

   try:
       results = lwdid(
           data, y='outcome', ivar='unit', tvar='year',
           gvar='first_treat', rolling='demean',
           aggregate='overall'  # Requires never-treated units
       )
   except NoNeverTreatedError as e:
       print(f"No never-treated units: {e}")
       # Use aggregate='none' or add never-treated units to data

**Typical error message:**

.. code-block:: text

   NoNeverTreatedError: aggregate='overall' requires never-treated units,
   but none were found in the data.

Data Validation Errors
~~~~~~~~~~~~~~~~~~~~~~

**Raised as:** ``InvalidParameterError``, ``InsufficientDataError``,
``TimeDiscontinuityError``, or ``MissingRequiredColumnError`` when
input data fail validation checks.

**Common causes:**

- Invalid ``rolling`` method name (see ``InvalidRollingMethodError``)
- Invalid ``vce`` option (see ``InvalidVCETypeError``)
- ``cluster_var`` missing or incompatible when ``vce='cluster'``
- Treatment indicator or controls not time-invariant
- Non-numeric outcome or control variables
- Time variables not convertible to valid numeric year/quarter values

**Typical issues (conceptual):**

- Singular matrix or near-singular design matrix (perfect or near-perfect
  collinearity among regressors)
- Insufficient variation in key variables (for example, all units have
  the same treatment status in the regression sample)

Estimation Errors
~~~~~~~~~~~~~~~~~

**Raised as:** subclasses of ``LWDIDError`` (for example
``InsufficientDataError`` or ``InvalidParameterError``) when estimation
fails due to data or parameter issues. Low-level numerical failures from
underlying libraries (for example, singular matrix errors in
``statsmodels``) may instead surface as their native exceptions.

**Common causes:**

- Singular matrix or near-singular design matrix (perfect or near-perfect
  collinearity among regressors)
- Insufficient variation in key variables (for example, all units have
  the same treatment status in the regression sample)

**Example:**

.. code-block:: python

   from lwdid import lwdid
   from lwdid.exceptions import LWDIDError

   try:
       results = lwdid(data, 'y', 'd', 'unit', 'year', 'post', 'demean')
   except LWDIDError as e:
       print(f"Estimation failed: {e}")
       # Check for perfect collinearity, insufficient variation, or other issues
   except Exception as e:
       print(f"Unexpected error: {e}")
       # Handle other errors

Error Handling Best Practices
-----------------------------

Catch Specific Exceptions
~~~~~~~~~~~~~~~~~~~~~~~~~

Catch specific exceptions for targeted error handling:

.. code-block:: python

   from lwdid import lwdid
   from lwdid.exceptions import (
       InvalidParameterError,
       InsufficientDataError,
       InvalidStaggeredDataError,
       NoNeverTreatedError,
       TimeDiscontinuityError,
       MissingRequiredColumnError,
       RandomizationError,
       VisualizationError,
       UnbalancedPanelError,
       AggregationError,
   )

   try:
       results = lwdid(data, 'y', 'd', 'unit', 'year', 'post', 'demean')

   except MissingRequiredColumnError as e:
       print(f"Missing columns: {e}")
       # Fix data and retry

   except TimeDiscontinuityError as e:
       print(f"Time structure issue: {e}")
       # Fix time index or post indicator

   except InvalidStaggeredDataError as e:
       print(f"Staggered data error: {e}")
       # Check gvar column for valid values

   except NoNeverTreatedError as e:
       print(f"No never-treated units: {e}")
       # Use aggregate='none' or add never-treated units

   except InsufficientDataError as e:
       print(f"Not enough data: {e}")
       # Use different method or collect more data

   except InvalidParameterError as e:
       print(f"Parameter error: {e}")
       # Fix parameters and retry

   except RandomizationError as e:
       print(f"Randomization inference failed: {e}")

   except VisualizationError as e:
       print(f"Plotting failed: {e}")

Catch All Package Errors
~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``LWDIDError`` to catch all package-specific errors:

.. code-block:: python

   from lwdid import lwdid
   from lwdid.exceptions import LWDIDError

   try:
       results = lwdid(data, 'y', 'd', 'unit', 'year', 'post', 'demean')
   except LWDIDError as e:
       print(f"LWDID error: {e}")
       # Handle any package error
   except Exception as e:
       print(f"Unexpected error: {e}")
       # Handle other errors

Logging Errors
~~~~~~~~~~~~~~

Log errors for debugging:

.. code-block:: python

   import logging
   from lwdid import lwdid
   from lwdid.exceptions import LWDIDError

   logging.basicConfig(level=logging.INFO)
   logger = logging.getLogger(__name__)

   try:
       results = lwdid(data, 'y', 'd', 'unit', 'year', 'post', 'demean')
   except LWDIDError as e:
       logger.error(f"LWDID estimation failed: {e}", exc_info=True)
       raise

Graceful Degradation
~~~~~~~~~~~~~~~~~~~~

Try alternative specifications when estimation fails:

.. code-block:: python

   from lwdid import lwdid
   from lwdid.exceptions import InsufficientDataError

   # Try detrend first
   try:
       results = lwdid(data, 'y', 'd', 'unit', 'year', 'post', 'detrend')
       print("Using detrend transformation")

   except InsufficientDataError:
       # Fall back to demean if insufficient pre-treatment periods
       results = lwdid(data, 'y', 'd', 'unit', 'year', 'post', 'demean')
       print("Insufficient data for detrend, using demean instead")

Common Error Scenarios
----------------------

Scenario 1: Insufficient Pre-Treatment Periods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Error:**

.. code-block:: text

   InsufficientPrePeriodsError: Insufficient pre-treatment periods for 'detrend'.

**Diagnosis:**

.. code-block:: python

   # Check pre-treatment periods by unit
   pre_periods = data[data['post'] == 0].groupby('unit').size()
   print(pre_periods[pre_periods < 2])

**Solution:**

.. code-block:: python

   # Option 1: Use demean instead
   results = lwdid(data, 'y', 'd', 'unit', 'year', 'post', 'demean')

   # Option 2: Drop units with insufficient periods
   units_ok = pre_periods[pre_periods >= 2].index
   data = data[data['unit'].isin(units_ok)]

Scenario 2: Time-Varying Controls
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Error:**

.. code-block:: text

   InvalidParameterError: Control variable 'income' must be time-invariant.

**Diagnosis:**

.. code-block:: python

   # Check which controls vary
   for control in ['income', 'population']:
       varying = data.groupby('unit')[control].nunique()
       print(f"{control}: {(varying > 1).sum()} units vary")

**Solution:**

.. code-block:: python

   # Use baseline (first period) value
   baseline = data.groupby('unit')['income'].first().reset_index()
   baseline.columns = ['unit', 'income_baseline']
   data = data.drop('income', axis=1).merge(baseline, on='unit')

See Also
--------

- :doc:`validation` - Validation functions that raise these exceptions
- :func:`lwdid.lwdid` - Main function that may raise exceptions
- :doc:`../user_guide` - Comprehensive usage guide with troubleshooting

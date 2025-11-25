Validation Module (validation)
===============================

The validation module ensures data quality and assumption compliance before
estimation. It checks panel structure, treatment timing, and data requirements.

.. automodule:: lwdid.validation
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

This module performs comprehensive validation of input data to ensure:

1. **Panel structure** is correct (unique unit-time pairs, continuous time)
2. **Treatment timing** follows common timing assumption
3. **Sample size** meets minimum requirements

4. **Control variables** are time-invariant
5. **Data types** are appropriate

All validation functions raise informative exceptions when requirements are
violated, helping users identify and fix data issues quickly.

Validation Checks
-----------------

Panel Structure Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Checks performed:**

- No duplicate (unit, time) observations
- Time index forms a continuous sequence (no gaps)
- Sufficient observations for estimation (N ≥ 3)
- At least one treated unit (d = 1) and one control unit (d = 0)

**Why it matters:**

- Duplicate observations indicate data errors
- Time gaps violate the continuous panel assumption
- Too few units make inference unreliable
- Need at least one treated and one control unit for DiD estimation

**Example error (conceptual):**

.. code-block:: text

   InvalidParameterError indicating duplicate (unit, time) observations.
   Each (unit, time) combination must appear at most once.

Treatment Timing Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Checks performed:**

- ``post`` indicator is binarized internally as 0/1 (non-zero values are
  treated as 1)
- ``post`` is the same for all units in each time period (common timing)
- ``post`` is monotone (no treatment reversals)
- At least one pre-treatment and one post-treatment period exist
  (``post != 0``) exist

**Why it matters:**

- Common timing is a core assumption of the method
- Treatment reversals violate the persistence assumption
- Need both pre- and post-treatment periods for DiD

**Example errors (conceptual):**

.. code-block:: text

   InvalidParameterError indicating that the common timing assumption is violated
   because 'post' varies across units within the same period.

.. code-block:: text

   TimeDiscontinuityError indicating that 'post' is not monotone in time
   (treatment reversals or suspensions).

Pre-Treatment Period Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Checks performed:**

- Each unit has sufficient pre-treatment periods for the chosen transformation:
  - ``demean``: at least 1 pre-treatment observation per unit
  - ``detrend``: at least 2 pre-treatment observations per unit
  - ``demeanq``: at least 1 pre-treatment observation per unit, and enough
    pre-period observations to estimate quarterly fixed effects
    (number of pre-period observations ≥ number of distinct pre-period
    quarters + 1)
  - ``detrendq``: at least 2 pre-treatment observations per unit, and enough
    pre-period observations to estimate a linear trend plus quarterly fixed
    effects (number of pre-period observations ≥ 1 + number of distinct
    pre-period quarters)

**Why it matters:**

- Demean requires at least 1 pre-treatment observation per unit to compute the
  pre-treatment mean
- Detrend requires at least 2 pre-treatment observations per unit to estimate a
  linear trend
- Quarterly methods additionally require sufficient pre-treatment observations
  within each unit to estimate quarterly fixed effects without rank
  deficiency

**Example error (conceptual):**

.. code-block:: text

   InsufficientPrePeriodsError indicating that some units have fewer
   pre-treatment periods than required by the chosen transformation.

Control Variables Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Checks performed:**

- Control variables exist in the data
- Controls are time-invariant (constant within each unit)
- Control variables are numeric; missing values (if any) are handled at
  the estimation stage rather than in the validation step

**Why it matters:**

- Time-varying controls can be endogenous to treatment
- Missing controls can lead to dropped observations when controls are included in the regression

**Example error (conceptual):**

.. code-block:: text

   InvalidParameterError indicating that control variable 'income' is
   not time-invariant within units.
   For example, 'income' varies within unit 'unit_42'.

Data Type Validation
~~~~~~~~~~~~~~~~~~~~~

**Checks performed:**

- Outcome variable is numeric
- Treatment indicator can be converted to numeric and is interpreted as
  0 vs non-zero (non-zero values are treated as 1)
- Unit and time identifiers are present
- Rows with missing values in required variables (outcome, treatment,
  unit identifier, time variable(s), or post) are dropped with a warning

**Why it matters:**

- Non-numeric outcomes cannot be used in regression
- Missing values in key variables change the effective sample after
  dropping affected rows

**Example error (conceptual):**

.. code-block:: text

   InvalidParameterError indicating that outcome variable 'y' is not
   numeric.

Validation Functions
--------------------

In practice, structural validation of panel layout, treatment timing,
control variables, and data types is performed internally by
``validate_and_prepare_data()``, which is called at the beginning of
``lwdid()``. Additional pre-treatment period requirements for each
``rolling`` method and quarterly coverage checks are enforced in the
transformation step (see :mod:`lwdid.transformations`). The earlier
sections (panel structure, treatment timing, pre-treatment periods,
controls, data types) describe **conceptual checks** rather than public
helper functions.

The pandas-based snippets in the following sections illustrate how to
diagnose and fix common problems yourself before calling ``lwdid()``,
but there are no separate public functions named
``validate_panel_structure``, ``validate_treatment_timing``, or
``validate_controls`` in the current implementation.

validate_and_prepare_data()
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lwdid.validation import validate_and_prepare_data
   import pandas as pd

   data = pd.read_csv('panel_data.csv')

   data_clean, metadata = validate_and_prepare_data(
       data=data,
       y='outcome',
       d='treated',
       ivar='unit',
       tvar='year',          # or ['year', 'quarter'] for quarterly data
       post='post',
       rolling='demean',     # required rolling method: 'demean', 'detrend', 'demeanq', or 'detrendq'
       controls=['x1', 'x2'] # optional time-invariant controls
   )

   print(metadata['N'], metadata['T'], metadata['K'])

 Quarterly helper checks
 ~~~~~~~~~~~~~~~~~~~~~~~

 For quarterly data, the module also provides helper functions such as
 ``validate_quarter_coverage`` that can be used in advanced workflows to
 pre-check seasonal coverage requirements for ``demeanq``/``detrendq``.
 In typical usage these helpers are called indirectly by
 :func:`lwdid.lwdid` via the transformation module rather than being
 used directly.

Common Validation Errors and Solutions
---------------------------------------

Error: Duplicate Observations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem (conceptual):**

.. code-block:: text

   InvalidParameterError indicating duplicate (unit, time) observations.

**Cause:** Multiple rows with the same (unit, time) combination.

**Solution:**

.. code-block:: python

   # Check for duplicates
   duplicates = data[data.duplicated(subset=['unit', 'year'], keep=False)]
   print(duplicates)

   # Remove duplicates (if appropriate)
   data = data.drop_duplicates(subset=['unit', 'year'], keep='first')

   # Or aggregate duplicates
   data = data.groupby(['unit', 'year']).mean().reset_index()

Error: Non-Common Treatment Timing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem (conceptual):**

.. code-block:: text

   InvalidParameterError indicating violation of the common timing assumption.

**Cause:** ``post`` varies across units in the same time period.

**Solution:**

.. code-block:: python

   # Check if post is time-based
   post_by_time = data.groupby('year')['post'].nunique()
   print(post_by_time[post_by_time > 1])  # Periods with varying post

   # Create time-based post indicator
   treatment_year = 2020
   data['post'] = (data['year'] >= treatment_year).astype(int)

Error: Insufficient Pre-Treatment Periods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem (conceptual):**

.. code-block:: text

   InsufficientPrePeriodsError indicating that some units lack enough
   pre-treatment periods for the chosen transformation.

**Cause:** For methods that require at least two pre-treatment periods (for

example ``detrend``/``detrendq``), some units have fewer than 2 pre-treatment
periods, or more generally fewer pre-treatment periods than required by the
chosen transformation.

**Solution:**

.. code-block:: python

   # Check pre-treatment periods by unit
   pre_periods = data[data['post'] == 0].groupby('unit').size()
   print(pre_periods[pre_periods < 2])  # Units with < 2 pre-periods

   # Option 1: Use 'demean' instead (requires T0 >= 1)
   results = lwdid(..., rolling='demean')

   # Option 2: Drop units with insufficient pre-treatment periods
   units_to_keep = pre_periods[pre_periods >= 2].index
   data = data[data['unit'].isin(units_to_keep)]

Error: Time-Varying Controls
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem (conceptual):**

.. code-block:: text

   InvalidParameterError indicating that control variable 'income' is
   not time-invariant.
 (constant within each unit).

**Cause:** Control variable changes over time for some units.

**Solution:**

.. code-block:: python

   # Check which controls vary
   for control in ['income', 'population']:
       varying = data.groupby('unit')[control].nunique()
       print(f"{control} varies in {(varying > 1).sum()} units")

   # Option 1: Use baseline (first period) value
   baseline = data.groupby('unit')['income'].first().reset_index()
   baseline.columns = ['unit', 'income_baseline']
   data = data.drop('income', axis=1).merge(baseline, on='unit')

   # Option 2: Use pre-treatment average
   pre_avg = data[data['post'] == 0].groupby('unit')['income'].mean()
   pre_avg = pre_avg.reset_index()
   pre_avg.columns = ['unit', 'income_pre_avg']
   data = data.drop('income', axis=1).merge(pre_avg, on='unit')

Error: Time Gaps
~~~~~~~~~~~~~~~~

**Problem (conceptual):**

.. code-block:: text

   TimeDiscontinuityError indicating that the time index is not continuous.

**Cause:** Missing time periods in the data.

**Solution:**

.. code-block:: python

   # Check time sequence
   time_values = sorted(data['year'].unique())
   gaps = [time_values[i+1] - time_values[i]
           for i in range(len(time_values)-1) if time_values[i+1] - time_values[i] > 1]
   print(f"Gaps found: {gaps}")

   # Option 1: Fill gaps with missing values (if appropriate)
   # Create complete panel
   units = data['unit'].unique()
   years = range(data['year'].min(), data['year'].max() + 1)
   complete_index = pd.MultiIndex.from_product([units, years],
                                                names=['unit', 'year'])
   data = data.set_index(['unit', 'year']).reindex(complete_index).reset_index()

   # Option 2: Restrict to continuous sub-period
   data = data[data['year'] >= 2015]  # Use only recent years

Best Practices
--------------

Pre-Validation Checks
~~~~~~~~~~~~~~~~~~~~~~

Before running ``lwdid()``, perform these checks:

.. code-block:: python

   import pandas as pd

   # 1. Check for duplicates
   assert not data.duplicated(subset=['unit', 'year']).any(), "Duplicates found"

   # 2. Check time continuity
   time_seq = sorted(data['year'].unique())
   assert all(time_seq[i+1] - time_seq[i] == 1
              for i in range(len(time_seq)-1)), "Time gaps found"

   # 3. Check post is time-based
   assert data.groupby('year')['post'].nunique().max() == 1, "Post varies by unit"

   # 4. Check controls are time-invariant
   for control in ['x1', 'x2']:
       assert data.groupby('unit')[control].nunique().max() == 1, \
              f"{control} varies within units"

   # 5. Check sample size
   n_units = data['unit'].nunique()
   assert n_units >= 3, f"Too few units: {n_units}"

   print("All pre-validation checks passed!")

Data Preparation Checklist
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before using ``lwdid()``, ensure:

1. ☑ Data is in long format (one row per unit-time observation)
2. ☑ No duplicate (unit, time) pairs
3. ☑ Time variable forms continuous sequence

4. ☑ ``post`` is binary (0/1) and time-based
5. ☑ ``d`` is time-invariant treatment group indicator
6. ☑ Control variables (if any) are time-invariant
7. ☑ No missing values in required variables
8. ☑ Sufficient pre-treatment periods for chosen transformation
9. ☑ At least N ≥ 3 units

See Also
--------

- :func:`lwdid.lwdid` - Main estimation function
- :doc:`../user_guide` - Comprehensive usage guide
- :doc:`exceptions` - Exception classes raised by validation

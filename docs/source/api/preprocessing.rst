Preprocessing Module (preprocessing)
====================================

The preprocessing module provides functionality to aggregate repeated
cross-sectional data to panel format for use with lwdid estimation methods.

.. automodule:: lwdid.preprocessing
   :no-members:

Overview
--------

When treatment is assigned at the unit level (e.g., state, county) but data
is collected at a lower level (e.g., individuals, firms), it is common to
aggregate outcomes to the unit-by-period level before applying DiD methods.

This module implements the aggregation methodology described in Lee and
Wooldridge (2026), using the weighted average formula:

.. math::

   \bar{Y}_{st} = \sum_{i \in (s,t)} w_{ist} Y_{ist}

where the weights :math:`w_{ist}` are normalized within each (unit, period)
cell to sum to one.

Main Functions
--------------

aggregate_to_panel
~~~~~~~~~~~~~~~~~~

.. autofunction:: lwdid.preprocessing.aggregate_to_panel
   :no-index:

Result Classes
--------------

AggregationResult
~~~~~~~~~~~~~~~~~

.. autoclass:: lwdid.preprocessing.AggregationResult
   :members:
   :undoc-members:
   :no-index:

CellStatistics
~~~~~~~~~~~~~~

.. autoclass:: lwdid.preprocessing.CellStatistics
   :members:
   :undoc-members:
   :no-index:

Usage Examples
--------------

Basic Aggregation
~~~~~~~~~~~~~~~~~

Aggregate individual-level survey data to state-year panel:

.. code-block:: python

   from lwdid.preprocessing import aggregate_to_panel

   # Aggregate to state-year panel
   result = aggregate_to_panel(
       data=survey_data,
       unit_var='state',
       time_var='year',
       outcome_var='income',
       treatment_var='treated'
   )

   # Access the aggregated panel data
   panel_data = result.panel_data

   # View summary statistics
   print(result.summary())

Weighted Aggregation
~~~~~~~~~~~~~~~~~~~~

Use survey weights for proper aggregation:

.. code-block:: python

   result = aggregate_to_panel(
       data=survey_data,
       unit_var='state',
       time_var='year',
       outcome_var='income',
       weight_var='survey_weight',
       treatment_var='treated',
       gvar='first_treat_year'
   )

   # Check effective sample sizes
   print(f"Min cell size: {result.min_cell_size}")
   print(f"Mean cell size: {result.mean_cell_size:.1f}")

High-Frequency Aggregation
~~~~~~~~~~~~~~~~~~~~~~~~~~

Aggregate to quarterly or monthly panels:

.. code-block:: python

   # Quarterly aggregation
   result = aggregate_to_panel(
       data=survey_data,
       unit_var='state',
       time_var=['year', 'quarter'],
       outcome_var='income',
       frequency='quarterly'
   )

   # Monthly aggregation
   result = aggregate_to_panel(
       data=survey_data,
       unit_var='county',
       time_var=['year', 'month'],
       outcome_var='employment',
       frequency='monthly'
   )

Aggregating Control Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Include time-invariant controls in the aggregation:

.. code-block:: python

   result = aggregate_to_panel(
       data=survey_data,
       unit_var='state',
       time_var='year',
       outcome_var='income',
       controls=['population', 'median_age', 'urban_pct'],
       treatment_var='treated'
   )

   # Control variables are included in the output
   print(result.panel_data.columns.tolist())

Minimum Cell Size Requirement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Exclude cells with insufficient observations:

.. code-block:: python

   result = aggregate_to_panel(
       data=survey_data,
       unit_var='state',
       time_var='year',
       outcome_var='income',
       min_cell_size=30,  # Require at least 30 observations per cell
       compute_variance=True
   )

   # Check excluded cells
   print(f"Excluded {result.n_excluded_cells} cells")
   for info in result.excluded_cells_info:
       print(f"  {info}")

Integration with lwdid
~~~~~~~~~~~~~~~~~~~~~~

After aggregation, use the panel data with lwdid:

.. code-block:: python

   import lwdid
   from lwdid.preprocessing import aggregate_to_panel

   # Step 1: Aggregate repeated cross-sectional data
   agg_result = aggregate_to_panel(
       data=survey_data,
       unit_var='state',
       time_var='year',
       outcome_var='income',
       treatment_var='treated',
       gvar='first_treat_year'
   )

   # Step 2: Estimate treatment effects on aggregated panel
   results = lwdid.lwdid(
       data=agg_result.panel_data,
       y='income',
       ivar='state',
       tvar='year',
       gvar='first_treat_year',
       rolling='demean',
       estimator='ipwra'
   )

   print(results.summary())

Methodological Notes
--------------------

Weight Normalization
~~~~~~~~~~~~~~~~~~~~

Survey weights are normalized within each (unit, period) cell to sum to one:

.. math::

   w_{ist}^* = \frac{w_{ist}}{\sum_{j \in (s,t)} w_{jst}}

This ensures that each cell contributes equally to the estimation regardless
of sample size differences across cells.

Variance Computation
~~~~~~~~~~~~~~~~~~~~

When ``compute_variance=True``, the weighted variance within each cell is
computed for diagnostic purposes:

.. math::

   Var(\bar{Y}_{st}) = \frac{\sum_{i} w_{ist}^* (Y_{ist} - \bar{Y}_{st})^2}{1 - \sum_{i} (w_{ist}^*)^2}

This uses the Bessel correction adjusted for weights.

Treatment Consistency
~~~~~~~~~~~~~~~~~~~~~

The function validates that treatment status is consistent within each unit
across all time periods. If treatment varies within a unit, a warning is
issued and the modal treatment value is used.

Data Requirements
-----------------

Input data must satisfy:

1. **Unit identifier**: Column identifying the aggregation unit (e.g., state)
2. **Time identifier**: Column(s) identifying the time period
3. **Outcome variable**: Numeric column to be aggregated
4. **No duplicate observations**: Each row should represent a unique lower-level
   observation (e.g., individual in a specific state-year)

Optional inputs:

- **Survey weights**: For weighted aggregation
- **Treatment indicator**: Binary treatment status (validated for consistency)
- **Cohort variable**: First treatment period for staggered designs
- **Control variables**: Time-invariant covariates to include in output

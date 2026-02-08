Trend Diagnostics Module (trend_diagnostics)
=============================================

The trend diagnostics module provides tools for assessing the parallel trends
assumption and detecting heterogeneous trends across treatment cohorts in
difference-in-differences settings.

.. automodule:: lwdid.trend_diagnostics
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

This module supports three main diagnostic workflows:

1. **Parallel trends testing**: Assess whether treated and control groups
   exhibit similar outcome trends prior to treatment.

2. **Heterogeneous trends diagnosis**: Detect cohort-specific linear trends
   that may violate the standard parallel trends assumption.

3. **Transformation recommendation**: Provide data-driven guidance on whether
   to use demeaning or detrending based on diagnostic results.

The conditional heterogeneous trends (CHT) framework from Lee and Wooldridge
(2025) allows each treatment cohort to have its own linear trend,
relaxing the standard parallel trends assumption. When CHT holds but parallel
trends fails, detrending removes cohort-specific linear trends and restores
consistency.

Testing Functions
-----------------

test_parallel_trends
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: lwdid.trend_diagnostics.test_parallel_trends

diagnose_heterogeneous_trends
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: lwdid.trend_diagnostics.diagnose_heterogeneous_trends

recommend_transformation
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: lwdid.trend_diagnostics.recommend_transformation

Visualization
-------------

plot_cohort_trends
~~~~~~~~~~~~~~~~~~

.. autofunction:: lwdid.trend_diagnostics.plot_cohort_trends

Enumerations
------------

TrendTestMethod
~~~~~~~~~~~~~~~

.. autoclass:: lwdid.trend_diagnostics.TrendTestMethod
   :members:
   :undoc-members:

TransformationMethod
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lwdid.trend_diagnostics.TransformationMethod
   :members:
   :undoc-members:

RecommendationConfidence
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lwdid.trend_diagnostics.RecommendationConfidence
   :members:
   :undoc-members:

Result Classes
--------------

ParallelTrendsTestResult
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lwdid.trend_diagnostics.ParallelTrendsTestResult
   :members:
   :undoc-members:

HeterogeneousTrendsDiagnostics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lwdid.trend_diagnostics.HeterogeneousTrendsDiagnostics
   :members:
   :undoc-members:

TransformationRecommendation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lwdid.trend_diagnostics.TransformationRecommendation
   :members:
   :undoc-members:

Supporting Data Classes
-----------------------

PreTrendEstimate
~~~~~~~~~~~~~~~~

.. autoclass:: lwdid.trend_diagnostics.PreTrendEstimate
   :members:
   :undoc-members:

CohortTrendEstimate
~~~~~~~~~~~~~~~~~~~

.. autoclass:: lwdid.trend_diagnostics.CohortTrendEstimate
   :members:
   :undoc-members:

TrendDifference
~~~~~~~~~~~~~~~

.. autoclass:: lwdid.trend_diagnostics.TrendDifference
   :members:
   :undoc-members:

Usage Examples
--------------

Testing Parallel Trends
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import lwdid
   import pandas as pd

   # Test parallel trends with placebo method
   pt_result = lwdid.test_parallel_trends(
       data=df,
       y='outcome',
       ivar='unit_id',
       tvar='year',
       gvar='first_treat',
       method='placebo',
       alpha=0.05
   )

   # Display summary
   print(pt_result.summary())

   # Check recommendation
   if pt_result.reject_null:
       print("Parallel trends rejected, consider detrending")
   else:
       print("Parallel trends not rejected, demeaning appropriate")

Diagnosing Heterogeneous Trends
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Diagnose trend heterogeneity across cohorts
   ht_diag = lwdid.diagnose_heterogeneous_trends(
       data=df,
       y='outcome',
       ivar='unit_id',
       tvar='year',
       gvar='first_treat'
   )

   # Display summary
   print(ht_diag.summary())

   # Check if heterogeneous trends detected
   if ht_diag.has_heterogeneous_trends:
       print(f"Recommendation: {ht_diag.recommendation}")

Getting Transformation Recommendation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get automated transformation recommendation
   rec = lwdid.recommend_transformation(
       data=df,
       y='outcome',
       ivar='unit_id',
       tvar='year',
       gvar='first_treat',
       verbose=True
   )

   print(f"Recommended method: {rec.recommended_method}")
   print(f"Confidence: {rec.confidence:.1%}")
   for reason in rec.reasons:
       print(f"  - {reason}")

Visualizing Cohort Trends
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Plot outcome trajectories by cohort
   fig = lwdid.plot_cohort_trends(
       data=df,
       y='outcome',
       ivar='unit_id',
       tvar='year',
       gvar='first_treat',
       normalize=True
   )
   fig.savefig('cohort_trends.png')

Methodological Background
-------------------------

The parallel trends assumption requires that, in the absence of treatment,
treated and control units would have followed similar outcome trajectories.
Formally, for all post-treatment periods :math:`t \geq S`:

.. math::

   E[Y_t(0) - Y_1(0) | D = 1] = E[Y_t(0) - Y_1(0) | D = 0]

When this assumption fails but the conditional heterogeneous trends (CHT)
assumption holds, detrending can restore identification. Under CHT, each
cohort :math:`g` may have its own linear trend :math:`\eta_g`, but the
detrended outcomes satisfy:

.. math::

   E[\ddot{Y}_{ir}(\infty) | \mathbf{D}, \mathbf{X}] = E[\ddot{Y}_{ir}(\infty) | \mathbf{X}]

where :math:`\ddot{Y}_{ir}` denotes the detrended outcome.

Decision Framework
~~~~~~~~~~~~~~~~~~

The recommended workflow for transformation selection:

1. Run ``test_parallel_trends()`` with ``method='placebo'``
2. If parallel trends not rejected: use ``rolling='demean'``
3. If parallel trends rejected: run ``diagnose_heterogeneous_trends()``
4. If heterogeneous trends detected: use ``rolling='detrend'``
5. Use ``recommend_transformation()`` for automated guidance

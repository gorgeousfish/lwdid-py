Clustering Diagnostics Module (clustering_diagnostics)
=======================================================

Clustering diagnostics and recommendations for difference-in-differences.

This module provides tools for analyzing clustering structure in panel data
and recommending appropriate clustering levels for standard error estimation.
When treatment varies at a level higher than the observation unit, standard
errors should be clustered at the policy variation level.

Overview
--------

Proper clustering is essential for valid inference in DiD settings. This module
helps:

- **Analyze hierarchical relationships**: Between potential clustering variables
- **Detect treatment variation level**: Identify where treatment assignment varies
- **Recommend clustering variables**: With sufficient cluster counts
- **Check consistency**: Between clustering choice and treatment variation

For reliable cluster-robust inference, a minimum of 20-30 clusters is generally
recommended. When clusters are fewer, wild cluster bootstrap methods provide
more accurate inference.

Enums
-----

.. autoclass:: lwdid.clustering_diagnostics.ClusteringLevel
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: lwdid.clustering_diagnostics.ClusteringWarningLevel
   :members:
   :undoc-members:
   :no-index:

Data Classes
------------

.. autoclass:: lwdid.clustering_diagnostics.ClusterVarStats
   :members:
   :no-index:

.. autoclass:: lwdid.clustering_diagnostics.ClusteringDiagnostics
   :members:
   :no-index:

.. autoclass:: lwdid.clustering_diagnostics.ClusteringRecommendation
   :members:
   :no-index:

.. autoclass:: lwdid.clustering_diagnostics.ClusteringConsistencyResult
   :members:
   :no-index:

Main Functions
--------------

.. autofunction:: lwdid.clustering_diagnostics.diagnose_clustering

.. autofunction:: lwdid.clustering_diagnostics.recommend_clustering_level

.. autofunction:: lwdid.clustering_diagnostics.check_clustering_consistency

Example Usage
-------------

Diagnosing Clustering Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lwdid import diagnose_clustering
   
   # Analyze potential clustering variables
   diagnostics = diagnose_clustering(
       data=panel_data,
       ivar='county',
       potential_cluster_vars=['state', 'region'],
       treatment_var='treated'
   )
   
   # Review cluster counts
   for var_stats in diagnostics.cluster_var_stats:
       print(f"{var_stats.var_name}: {var_stats.n_clusters} clusters")
   
   # Check warnings
   for warning in diagnostics.warnings:
       print(f"[{warning.level}] {warning.message}")

Getting Clustering Recommendations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lwdid import recommend_clustering_level
   
   # Get recommendation with minimum cluster count
   recommendation = recommend_clustering_level(
       data=panel_data,
       ivar='county',
       potential_cluster_vars=['state', 'region'],
       treatment_var='treated',
       min_clusters=20
   )
   
   print(f"Recommended: {recommendation.recommended_var}")
   print(f"Reason: {recommendation.reason}")
   
   # Use in estimation
   results = lwdid(
       data, y='outcome', d='treated', ivar='county', tvar='year',
       post='post', rolling='demean',
       vce='cluster', cluster_var=recommendation.recommended_var
   )

Checking Consistency
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lwdid import check_clustering_consistency
   
   # Verify clustering choice is appropriate
   consistency = check_clustering_consistency(
       data=panel_data,
       ivar='county',
       cluster_var='state',
       treatment_var='treated'
   )
   
   print(f"Consistent: {consistency.is_consistent}")
   if not consistency.is_consistent:
       print(f"Issue: {consistency.message}")

Wild Cluster Bootstrap
~~~~~~~~~~~~~~~~~~~~~~

When cluster counts are small (< 20), use wild cluster bootstrap:

.. code-block:: python

   from lwdid import wild_cluster_bootstrap
   
   # Run wild cluster bootstrap
   bootstrap_result = wild_cluster_bootstrap(
       data=transformed_data,
       y_transformed='y_dot',
       d='treated',
       cluster_var='state',
       n_bootstrap=999,
       weight_type='rademacher'
   )
   
   print(f"Bootstrap SE: {bootstrap_result.se_bootstrap:.4f}")
   print(f"Bootstrap p-value: {bootstrap_result.pvalue:.4f}")

Guidelines
----------

**Minimum Cluster Counts:**

- **20-30 clusters**: Generally sufficient for cluster-robust standard errors
- **10-20 clusters**: Use wild cluster bootstrap for improved inference
- **< 10 clusters**: Wild cluster bootstrap essential; consider randomization inference

**Clustering Level Selection:**

1. Cluster at the level where treatment varies
2. If treatment varies at multiple levels, cluster at the highest level
3. Never cluster below the unit level

See Also
--------

- :doc:`inference` - Wild cluster bootstrap implementation
- :doc:`../methodological_notes` - Theoretical foundations of clustering
- :func:`lwdid.lwdid` - Main estimation with ``vce='cluster'``

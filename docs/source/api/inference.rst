Inference Module (inference)
=============================

Advanced inference methods for difference-in-differences estimation.

This module provides inference methods beyond standard asymptotic approaches,
including wild cluster bootstrap for reliable inference with few clusters.

Overview
--------

Standard cluster-robust standard errors require a large number of clusters
for valid asymptotic inference. When the number of clusters is small (typically
fewer than 20-30), wild cluster bootstrap methods provide more accurate
inference by resampling cluster-level weights rather than relying on asymptotic
approximations.

This module implements the wild cluster bootstrap procedure with extensions for
difference-in-differences settings.

Wild Cluster Bootstrap
----------------------

.. autoclass:: lwdid.inference.WildClusterBootstrapResult
   :members:
   :no-index:

.. autofunction:: lwdid.inference.wild_cluster_bootstrap

.. autofunction:: lwdid.inference.wild_cluster_bootstrap_test_inversion

Example Usage
-------------

Basic Wild Cluster Bootstrap
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lwdid.inference import wild_cluster_bootstrap
   
   # Run wild cluster bootstrap on transformed data
   result = wild_cluster_bootstrap(
       data=transformed_data,
       y_transformed='y_dot',
       d='treated',
       cluster_var='state',
       n_bootstrap=999,
       weight_type='rademacher'
   )
   
   print(f"Original ATT: {result.att:.4f}")
   print(f"Bootstrap SE: {result.se_bootstrap:.4f}")
   print(f"Bootstrap p-value: {result.pvalue:.4f}")
   print(f"95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")

Weight Types
~~~~~~~~~~~~

Three bootstrap weight distributions are available:

.. code-block:: python

   # Rademacher weights (+1 or -1 with equal probability)
   result_rad = wild_cluster_bootstrap(
       data, y_transformed='y_dot', d='treated',
       cluster_var='state', weight_type='rademacher'
   )
   
   # Mammen weights (two-point distribution)
   result_mam = wild_cluster_bootstrap(
       data, y_transformed='y_dot', d='treated',
       cluster_var='state', weight_type='mammen'
   )
   
   # Webb weights (six-point distribution)
   result_web = wild_cluster_bootstrap(
       data, y_transformed='y_dot', d='treated',
       cluster_var='state', weight_type='webb'
   )

Test Inversion for Confidence Intervals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For more accurate confidence intervals with few clusters:

.. code-block:: python

   from lwdid.inference import wild_cluster_bootstrap_test_inversion
   
   # Construct CI via test inversion
   result = wild_cluster_bootstrap_test_inversion(
       data=transformed_data,
       y_transformed='y_dot',
       d='treated',
       cluster_var='state',
       alpha=0.05,
       n_bootstrap=999
   )
   
   print(f"CI via test inversion: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")

Integration with lwdid
~~~~~~~~~~~~~~~~~~~~~~

Wild cluster bootstrap is used as a standalone function after running the main
estimation. First estimate the model using ``lwdid()``, then apply wild cluster
bootstrap to the transformed data:

.. code-block:: python

   from lwdid import lwdid
   from lwdid.inference import wild_cluster_bootstrap
   
   # Step 1: Run standard estimation with cluster-robust SE
   results = lwdid(
       data, y='outcome', d='treated', ivar='unit', tvar='year',
       post='post', rolling='demean',
       vce='cluster', cluster_var='state'
   )
   
   # Step 2: Apply wild cluster bootstrap to the transformed data
   boot_result = wild_cluster_bootstrap(
       data=results.data,
       y_transformed='ydot_postavg',
       d='d_',
       cluster_var='state',
       n_bootstrap=999
   )
   
   print(f"Bootstrap p-value: {boot_result.pvalue:.4f}")

Complete Enumeration
~~~~~~~~~~~~~~~~~~~~

With small numbers of clusters (G <= 12), exact enumeration of all possible
weight combinations provides exact p-values:

.. code-block:: python

   # Complete enumeration with few clusters
   result = wild_cluster_bootstrap(
       data=transformed_data,
       y_transformed='y_dot',
       d='treated',
       cluster_var='state',  # If G <= 12, complete enumeration is used
       weight_type='rademacher'
   )
   
   # Check if complete enumeration was used
   if result.n_bootstrap == 2**result.n_clusters:
       print("Complete enumeration used - exact p-value")

Methodological Notes
--------------------

**Algorithm:**

1. Estimate the original model and obtain residuals :math:`\hat{u}_{ic}`
2. Generate cluster-level weights :math:`w_c` from chosen distribution
3. Construct bootstrap residuals: :math:`u^*_{ic} = w_c \times \hat{u}_{ic}`
4. Form bootstrap outcomes: :math:`Y^*_{ic} = \hat{Y}_{ic} + u^*_{ic}`
5. Re-estimate the model and compute t-statistic
6. Repeat B times to obtain bootstrap distribution
7. Compute p-value as proportion of bootstrap t-statistics exceeding observed

**Weight Distributions:**

- **Rademacher**: :math:`P(w = 1) = P(w = -1) = 0.5`
- **Mammen**: Two-point distribution with :math:`E[w] = 0`, :math:`E[w^2] = 1`, :math:`E[w^3] = 1`
- **Webb**: Six-point distribution for improved performance with few clusters

**Null Imposition:**

The ``impose_null`` parameter determines whether bootstrap samples are
constructed under the null hypothesis (H0: treatment effect = 0). Imposing
the null generally improves power but may be conservative.

Guidelines
----------

**When to Use Wild Cluster Bootstrap:**

- Number of clusters < 30
- Treatment varies at cluster level
- Standard cluster-robust SEs may be unreliable

**Recommended Settings:**

- ``n_bootstrap=999`` or ``n_bootstrap=9999`` for publication
- ``weight_type='rademacher'`` is standard choice
- ``impose_null=True`` for testing H0: effect = 0

See Also
--------

- :doc:`clustering_diagnostics` - Clustering diagnostics and recommendations
- :doc:`../methodological_notes` - Theoretical foundations

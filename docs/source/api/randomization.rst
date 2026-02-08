Randomization Inference Module
==============================

Hypothesis testing via randomization inference for small and large samples.

This module implements randomization inference (RI) for testing the sharp null
hypothesis of no treatment effect in difference-in-differences settings. The
approach provides valid p-values without relying on asymptotic distributional
assumptions.

**Applicability**: RI is applicable to both small-sample and large-sample
scenarios. While it is particularly valuable for small samples where t-based
inference may be unreliable, it also serves as a robust alternative for large
samples when normality assumptions are questionable. See Lee and Wooldridge
(2026) for discussion of RI in the small-sample context.

.. automodule:: lwdid.randomization
   :no-members:

Main Function
-------------

.. autofunction:: lwdid.randomization.randomization_inference
   :no-index:

Resampling Methods
------------------

The implementation supports two resampling methods:

**Permutation (Classical Fisher Randomization Inference)**

Permutes treatment labels without replacement, preserving the original number
of treated and control units in each replication. This is the classical Fisher
randomization approach and is generally recommended for design-based
randomization inference.

**Bootstrap (Resampling with Replacement)**

Resamples treatment labels with replacement. May produce degenerate draws
(all treated or all control) which are excluded from p-value computation.
This method is the default for backward compatibility.

Parameters
----------

firstpost_df : pd.DataFrame
    Cross-sectional data for the first post-treatment period, containing one
    observation per unit.

y_col : str, default 'ydot_postavg'
    Column name of the transformed outcome variable.

d_col : str, default ``'d_'``
    Column name of the binary treatment indicator.

ivar : str, default 'ivar'
    Column name of the unit identifier.

rireps : int, default 1000
    Number of randomization replications. Higher values provide more precise
    p-values but increase computation time.

seed : int or None, optional
    Random seed for reproducibility.

ri_method : {'bootstrap', 'permutation'}, default 'bootstrap'
    Resampling method for generating the null distribution.

controls : list of str or None, optional
    Control variables to include in the regression model.

Returns
-------

dict
    Dictionary containing:

    - ``p_value``: Two-sided p-value (proportion with \|ATT_perm\| >= \|ATT_obs\|)
    - ``ri_method``: Resampling method used
    - ``ri_reps``: Total replications requested
    - ``ri_valid``: Number of valid replications
    - ``ri_failed``: Number of failed replications
    - ``ri_failure_rate``: Proportion of failed replications

Example Usage
-------------

.. code-block:: python

   from lwdid import lwdid

   # Estimation with randomization inference
   results = lwdid(
       data,
       y='outcome',
       d='treated',
       ivar='unit',
       tvar='year',
       post='post',
       rolling='detrend',
       ri=True,
       rireps=1000,
       ri_method='permutation',
       seed=42
   )

   # Access RI results
   print(f"RI p-value: {results.ri_pvalue:.4f}")
   print(f"Method: {results.ri_method}")
   print(f"Valid replications: {results.ri_valid}/{results.rireps}")

Methodological Notes
--------------------

Randomization inference tests the sharp null hypothesis that all unit-level
treatment effects are exactly zero. Under this null, treatment assignment is
uninformative about potential outcomes, and permuting treatment labels generates
the null distribution of the test statistic.

**Advantages**

- Does not rely on normality or homoskedasticity assumptions
- Naturally accommodates heteroskedasticity and non-normality
- Provides exact p-values under the randomization model

**Limitations**

- Computationally intensive for large numbers of replications
- Tests only the sharp null hypothesis (zero effect for all units)
- Bootstrap method may have high failure rates with extreme treatment proportions

See Also
--------

:func:`lwdid.lwdid` : Main estimation function with ``ri=True`` option.
:doc:`../methodological_notes` : Theoretical foundations.

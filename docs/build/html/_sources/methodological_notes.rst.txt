Methodological Notes
====================

This document provides the theoretical foundation and methodological details for the
Lee and Wooldridge (2025) difference-in-differences method implemented in ``lwdid``.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

The Lee and Wooldridge (2025) method addresses inference in difference-in-differences
estimation when the number of cross-sectional units is small. Traditional DiD methods
rely on asymptotic approximations that require large samples. This method provides
an alternative approach:

1. Transform panel data into cross-sectional form via unit-specific time-series operations
2. Estimate treatment effects from a cross-sectional regression on transformed data
3. Under classical linear model assumptions (including normality and homoskedasticity),
   conduct exact t-based inference from the cross-sectional regression

The method exploits an algebraic equivalence: by removing unit-specific pre-treatment
patterns (means or trends), the panel DiD estimator can be obtained from a cross-sectional
regression where exact finite-sample inference is available under the classical linear
model assumptions, particularly normality and homoskedasticity of the error term, when
homoskedastic OLS standard errors are used.

The Panel-to-Cross-Section Transformation
------------------------------------------

Core Principle
~~~~~~~~~~~~~~

The method removes unit-specific time-series patterns using only pre-treatment data,
then pools the transformed outcomes across units. This proceeds in two steps:

1. **Transformation step**: Apply a unit-specific time-series transformation using
   pre-treatment periods only
2. **Regression step**: Estimate treatment effects from a cross-sectional regression
   on the transformed outcomes

The transformation uses only pre-treatment information, preserving the treatment
variation for estimation in the second step.

Demean Transformation (Procedure 2.1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Setup**

Conceptually, unit i is observed over T periods (t = 1, ..., T). Treatment begins at period S,
where S ∈ {2, ..., T}. Pre-treatment periods are t = 1, ..., S-1 (T₀ = S-1 periods).
Post-treatment periods are t = S, ..., T (T₁ = T-S+1 periods).

In the notation of Lee and Wooldridge (2025), this description uses a balanced-panel
setup where each unit is observed in all T periods. The ``lwdid`` implementation also
accommodates unbalanced panels: units need not appear in every period. However, each
unit included in the data must have at least one pre-treatment observation so that its
pre-treatment mean can be computed. Units without any post-treatment observations
remain in the panel but do not contribute to the main ATT regression because their
post-treatment average is undefined.

**Procedure**

1. Compute the pre-treatment mean for each unit i:

   ȳᵢ,pre = (1/(S-1)) Σₜ₌₁^(S-1) yᵢₜ

2. Compute the post-treatment mean for each unit i:

   ȳᵢ,post = (1/(T-S+1)) Σₜ₌S^T yᵢₜ

3. Construct the transformed outcome:

   Δȳᵢ = ȳᵢ,post - ȳᵢ,pre

4. Estimate the cross-sectional regression:

   Δȳᵢ = α + τ·Dᵢ + Uᵢ,  i = 1, ..., N

   where Dᵢ = 1 for treated units and Dᵢ = 0 for control units.

**Estimand**: τ is the average treatment effect on the treated (ATT).

**Equivalence**: This transformation yields the same point estimate as the two-way
fixed effects (TWFE) DiD estimator. The cross-sectional representation enables
exact t-based inference under classical linear model assumptions.

Detrend Transformation (Procedure 3.1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Motivation**: When units exhibit heterogeneous linear trends in pre-treatment
periods, the standard parallel trends assumption (constant trends across units)
may be too restrictive. Detrending allows for unit-specific linear trends by
removing them before estimating treatment effects.

**Procedure**

1. For each unit i, estimate a linear trend using pre-treatment data:

   yᵢₜ = Aᵢ + Bᵢ·t + εᵢₜ  for t = 1, ..., S-1

   This requires at least two pre-treatment periods (T₀ ≥ 2).

2. Compute predicted values for all periods using the estimated trend:

   ŷᵢₜ = Âᵢ + B̂ᵢ·t  for t = 1, ..., T

3. For post-treatment periods, compute out-of-sample residuals:

   ÿᵢₜ = yᵢₜ - ŷᵢₜ  for t = S, ..., T

4. Average the residuals over post-treatment periods:

   ȳ̈ᵢ = (1/(T-S+1)) Σₜ₌S^T ÿᵢₜ

5. Estimate the cross-sectional regression:

   ȳ̈ᵢ = α + τ_DT·Dᵢ + Uᵢ,  i = 1, ..., N

**Estimand**: τ_DT is the ATT after removing unit-specific linear trends.

**Note**: The trend is estimated using only pre-treatment data, so the treatment
variation remains available for estimation in the cross-sectional regression. As in
the demean case, the implementation allows unbalanced panels: units need not appear
in every period, but each unit included in the data must have at least two
pre-treatment observations so that its trend can be estimated. Units without any
post-treatment observations remain in the panel but do not contribute to the main
ATT regression because their post-treatment average is undefined.

Quarterly Transformations
~~~~~~~~~~~~~~~~~~~~~~~~~~

**demeanq**: Extends Procedure 2.1 to include quarter-of-year fixed effects,
removing seasonal patterns in quarterly data.

**detrendq**: Extends Procedure 3.1 to include both linear trends and
quarter-of-year fixed effects.

Both methods include quarter dummies in the pre-treatment regression to remove
seasonal variation before computing post-treatment residuals.

Inference Under CLM Assumptions
--------------------------------

Classical Linear Model Assumptions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When homoskedastic OLS standard errors are used (``vce=None`` in ``lwdid``),
exact finite-sample inference is available under the classical linear model (CLM)
assumptions for the cross-sectional regression. For the demean transformation,
the model is Δȳᵢ = α + τ·Dᵢ + Uᵢ (for detrending, replace Δȳᵢ with ȳ̈ᵢ). The CLM
assumptions are:

1. **Linearity**: E[Uᵢ | Dᵢ] = 0 (zero conditional mean)
2. **Random sampling**: Units are independently sampled
3. **No perfect collinearity**: Treatment indicator varies across units
4. **Homoskedasticity**: Var(Uᵢ | Dᵢ) = σ² (constant variance)
5. **Normality**: Uᵢ | Dᵢ ~ N(0, σ²) (conditional normality)

Under these assumptions and with homoskedastic OLS standard errors, the
t-statistic (τ̂ - τ)/se(τ̂) follows an exact
t-distribution with residual degrees of freedom equal to N - k, where k is the
number of estimated parameters in the cross-sectional regression (k = 2 without
controls: intercept and treatment indicator).
The normality and homoskedasticity assumptions are critical for exact inference.

Degrees of Freedom
~~~~~~~~~~~~~~~~~~

**Homoskedastic standard errors**:

df = N - k

where N is the number of cross-sectional units and k is the number of parameters
(k = 2 without controls: intercept and treatment indicator).

**Cluster-robust standard errors**:

df = G - 1

where G is the number of clusters.

The cross-sectional regression has N observations (one per unit), not N×T.
With cluster-robust standard errors, degrees of freedom equal the number of
clusters minus one.

Period-Specific Effects
~~~~~~~~~~~~~~~~~~~~~~~

In addition to an overall post-treatment average effect, the Lee and Wooldridge
framework allows estimation of period-specific treatment effects by running
separate cross-sectional regressions for each post-treatment period, using the
transformed outcome in that period as the dependent variable (see equation (2.20)
in Lee and Wooldridge, 2025). In ``lwdid``, these regressions use the same
variance estimator (``vce``) and control-variable specification as the main ATT
regression.

Robust Inference
----------------

Heteroskedasticity-Robust Standard Errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When homoskedasticity is violated, heteroskedasticity-robust standard errors
provide asymptotically valid inference:

- **HC1**: Heteroskedasticity-consistent (White) estimator
- **HC3**: Small-sample adjusted version

Robust standard errors rely on asymptotic approximations and may be less accurate
in very small samples. Recent simulation evidence (Simonsohn 2021) suggests HC3
can perform reasonably well even with small sample sizes (e.g., N₀ = 18, N₁ = 2
in the cited study), though caution is warranted with very small samples.
Exact inference is not available under heteroskedasticity.

Cluster-Robust Standard Errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When errors are correlated within clusters, cluster-robust standard errors account
for within-cluster correlation.

These standard errors rely on large-sample approximations in the number of
clusters. Inference is more reliable when the number of clusters is not too
small; ``lwdid`` issues a warning when the cluster count is very low.
Degrees of freedom for cluster-robust inference are taken as G - 1. The cluster
variable must be nested within the unit identifier (each unit belongs to exactly
one cluster across all time periods), reflecting the usual assumption that
clusters are independent.

Randomization Inference
~~~~~~~~~~~~~~~~~~~~~~~

Randomization inference constructs a reference distribution for the test statistic
under the sharp null hypothesis without relying on normality or homoskedasticity
assumptions.

**Procedure**:

1. Compute the observed test statistic. In ``lwdid``, this is the ATT estimate τ̂ from the cross-sectional regression.
2. Randomly reassign treatment status across units
3. Re-estimate the model with reassigned treatment
4. Repeat steps 2-3 R times (e.g., R = 1000)
5. Compute the p-value as the proportion of replications with the test statistic
   at least as extreme (in absolute value) as the observed value

**Methods**:

- **Bootstrap** (with replacement): Treatment group size may vary across replications
- **Permutation** (without replacement): Fisher randomization inference; treatment
  group size is fixed across permutations

In ``lwdid``, randomization inference always re-estimates the cross-sectional
regression by OLS (ignoring the ``vce`` option used for the main estimate).
The default ``ri_method`` is ``'bootstrap'`` for backward compatibility, while
``'permutation'`` corresponds to the classical Fisher randomization approach and
is generally recommended for design-based randomization inference.

**Advantages**:

- Does not rely on normality or homoskedasticity assumptions
- Naturally accommodates heteroskedasticity and non-normality under the maintained
  randomization scheme

**Limitations**:

- Computationally intensive for large numbers of replications
- Tests only the sharp null hypothesis (zero treatment effect for all units)

Identification Assumptions
---------------------------

No Anticipation
~~~~~~~~~~~~~~~

Units do not change behavior in anticipation of future treatment. Pre-treatment
outcomes are unaffected by future treatment status.

Example violation: Firms may alter behavior before a regulation takes effect.

Parallel Trends
~~~~~~~~~~~~~~~

**Demean (Procedure 2.1)**: In the absence of treatment, the average change in
outcomes from pre- to post-treatment periods would be the same for treated and
control units. Formally, E[Yᵢₜ(0) - Yᵢ₁(0) | Dᵢ] is constant across treatment
groups for all t. This is the standard parallel trends assumption.

**Detrend (Procedure 3.1)**: In the absence of treatment, the average change in
outcomes after removing unit-specific linear trends would be the same for treated
and control units. This allows for heterogeneous linear trends across units,
relaxing the standard parallel trends assumption.

The parallel trends assumption is not directly testable because the counterfactual
outcome (what would have happened to treated units without treatment) is unobserved.
Researchers can examine pre-treatment trends visually, conduct placebo tests on
pre-treatment periods, or use detrending when units exhibit different linear trends
in the pre-treatment period.

Common Treatment Timing
~~~~~~~~~~~~~~~~~~~~~~~

All treated units begin treatment in the same period. This implementation does not
support staggered adoption (units treated at different times). See Lee and Wooldridge
(2025, Section 7) for methods accommodating staggered rollouts.

Treatment Persistence
~~~~~~~~~~~~~~~~~~~~~

Once treated, units remain treated. At the implementation level, the post-treatment
indicator must be monotone non-decreasing in the time index (once ``post`` switches
from 0 to 1, it cannot revert to 0). Treatment reversals or temporary treatments
are therefore not permitted.

Comparison with Other DiD Methods
----------------------------------

Two-Way Fixed Effects (TWFE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The demean transformation is algebraically equivalent to TWFE with unit and time
fixed effects. The Lee and Wooldridge method makes the cross-sectional structure
explicit, enabling exact t-based inference in small samples under the CLM
assumptions when homoskedastic OLS standard errors are used. In contrast, TWFE
inference is usually based on large-sample approximations with cluster-robust
standard errors.

Callaway and Sant'Anna (2021)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Callaway and Sant'Anna (2021) develop DiD estimators for settings with staggered
adoption and group-time average treatment effects, with inference based on
large-sample approximations in panels with many groups and time periods. Lee and
Wooldridge (2025) instead focuses on common treatment timing and uses the
panel-to-cross-section representation to obtain exact small-sample t-based
inference under classical linear model assumptions.



When to Use This Method
~~~~~~~~~~~~~~~~~~~~~~~

**This method is particularly useful when**:

- The number of treated and/or control units is modest (small cross-sectional N)
- All treated units begin treatment in the same period (common timing)
- The researcher is willing to assume normality and homoskedasticity for exact
  inference, or to use randomization inference as an alternative

**This implementation does not cover the following cases**:

- Treatment adoption is staggered (units treated at different times). See Lee
  and Wooldridge (2025, Section 7) for methods that accommodate staggered designs
- Treatment is reversed or temporary, which violates the treatment persistence
  assumption imposed by this implementation

Practical Considerations
------------------------

Sample Size Requirements
~~~~~~~~~~~~~~~~~~~~~~~~

**Minimum (panel-level checks)**: ``lwdid`` requires at least N ≥ 3 units in the
panel, with at least one treated unit (N₁ ≥ 1) and at least one control unit
(N₀ ≥ 1).

In addition, the firstpost cross-sectional regression used for ATT estimation
must contain at least three units in total and at least one treated and one
control unit. If some units drop out of the panel before the post-treatment
period, the effective cross-sectional sample entering the main regression can be
smaller than the panel N; in such cases ``lwdid`` raises an error rather than
reporting an ATT based on an under-identified regression.

**Practical considerations**:

- Exact t-based inference under CLM assumptions is technically available at this
  minimum, but in practice inference is more stable when the cross-sectional
  sample is meaningfully larger than N = 3.
- Cluster-robust standard errors rely on having a non-trivial number of clusters.
  A commonly used rule of thumb in applied work is to have around G ≥ 10 clusters
  for more reliable cluster-robust inference, although there is no universal
  threshold.

Time Index and Panel Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- The time index used internally by ``lwdid`` must form a contiguous sequence
  without gaps (e.g., years 2000, 2001, 2002, ...). If the original time
  variables have gaps, the function raises an error rather than silently
  interpolating.
- The post-treatment indicator ``post`` must be a pure function of time (common
  timing): at any given time period, all units are either pre-treatment or
  post-treatment. This aligns with the common timing assumption described above.

Pre-Treatment Periods
~~~~~~~~~~~~~~~~~~~~~~

**Minimum requirements in this implementation**:

- ``demean``: At least one pre-treatment period overall (T₀ ≥ 1), and each unit
  must have at least one pre-treatment observation so that its pre-treatment
  mean can be computed.
- ``detrend``: At least two pre-treatment periods overall (T₀ ≥ 2), and each
  unit must have at least two pre-treatment observations so that a unit-specific
  linear trend can be estimated.
- ``demeanq`` and ``detrendq``: The same minimum numbers of pre-treatment
  periods as above, plus additional quarterly coverage conditions. For each
  unit, every quarter that appears in the post-treatment period must also appear
  in the pre-treatment period; otherwise seasonal effects for that quarter
  cannot be removed. To avoid rank-deficient seasonal/trend regressions, the
  implementation also enforces simple per-unit degrees-of-freedom checks based
  on the number of distinct quarters observed in the pre-treatment period.

**Practical recommendations**:

- T₀ ≥ 3 for detrend/detrendq (more stable trend estimation)
- More pre-treatment periods improve statistical power and facilitate visual
  assessment of parallel trends

Control Variables
~~~~~~~~~~~~~~~~~

Time-invariant controls (e.g., baseline characteristics) are permitted.

**Effects**:

- Reduces residual variance, increasing power
- Reduces degrees of freedom
- Does not affect the transformation step

Including many controls relative to sample size can lead to overfitting and
unstable estimates. In this implementation, time-invariant controls are included
only when both groups are sufficiently large relative to the number of controls:
controls enter the regression if N₁ > K+1 and N₀ > K+1, where K is the number of
control variables. Otherwise, ``lwdid`` issues a warning and estimates the model
without controls to avoid under-identified or extremely fragile regressions.
When controls are included, observations with missing values in any included
control variable are dropped from the main regression sample; ``lwdid`` reports
how many observations were removed. If dropping those observations would cause
either group to violate the N₁ > K+1 or N₀ > K+1 conditions, controls are
omitted instead and the full sample is retained.

Computational Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Speed**: The transformation and cross-sectional regression steps are computationally
efficient for typical panel datasets. Randomization inference requires R replications
and is more computationally intensive (computation time scales linearly with R).

**Memory**: Memory requirements are modest for typical panel datasets. Randomization
inference stores R test statistics for computing the empirical p-value.

Limitations
-----------

This implementation has the following limitations:

1. **Common timing**: All treated units must begin treatment in the same period
2. **Binary treatment**: Treatment is either on or off (no continuous treatment intensity)
3. **Time-invariant controls**: Controls must not vary over time
4. **Treatment persistence**: Once treated, units must remain treated

Lee and Wooldridge (2025, Section 7) discusses extensions to staggered adoption,
where units are treated at different times. These extensions are not implemented
in this version.

References
----------

**Primary Reference:**

Lee, S. J., and Wooldridge, J. M. (2025). Simple Approaches to Inference with
Difference-in-Differences Estimators with Small Cross-Sectional Sample Sizes.
*Available at SSRN 5325686*.

**Related Literature:**

- Callaway, B., and Sant'Anna, P. H. (2021). Difference-in-differences with
  multiple time periods. *Journal of Econometrics*, 225(2), 200-230.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel
  Data* (2nd ed.). MIT Press.

Authors
-------

Xuanyu Cai, Wenli Xu

Further Reading
---------------

- :doc:`user_guide` - Comprehensive usage guide
- :doc:`quickstart` - Quick start tutorial
- :doc:`api/index` - Complete API reference

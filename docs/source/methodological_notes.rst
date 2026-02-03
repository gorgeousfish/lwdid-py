Methodological Notes
====================

This document provides the theoretical foundation and methodological details for the
Lee and Wooldridge difference-in-differences methods implemented in ``lwdid``.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

The ``lwdid`` package implements the Lee and Wooldridge methods for difference-in-
differences estimation with panel data, covering three scenarios:

1. **Small-sample common timing** (Lee and Wooldridge, 2026): Exact t-based inference
   under classical linear model assumptions when the number of cross-sectional units
   is small.

2. **Large-sample common timing** (Lee and Wooldridge, 2025): Rolling transformation
   approach with asymptotic inference using robust standard errors.

3. **Staggered adoption** (Lee and Wooldridge, 2025): Extension to settings where
   units are treated at different times, with cohort-time specific effect estimation.

The core method transforms panel data into cross-sectional form via unit-specific
time-series operations:

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

Conceptually, unit :math:`i` is observed over :math:`T` periods
(:math:`t = 1, \ldots, T`). Treatment begins at period :math:`S`, where
:math:`S \in \{2, \ldots, T\}`. Pre-treatment periods are
:math:`t = 1, \ldots, S-1` (:math:`T_0 = S-1` periods). Post-treatment periods
are :math:`t = S, \ldots, T` (:math:`T_1 = T-S+1` periods).

In the notation of Lee and Wooldridge (2026), this description uses a balanced-panel
setup where each unit is observed in all T periods. The ``lwdid`` implementation also
accommodates unbalanced panels: units need not appear in every period. However, each
unit included in the data must have at least one pre-treatment observation so that its
pre-treatment mean can be computed. Units without any post-treatment observations
remain in the panel but do not contribute to the main ATT regression because their
post-treatment average is undefined.

**Procedure**

1. Compute the pre-treatment mean for each unit i:

   .. math::

      \bar{Y}_{i,pre} = \frac{1}{S-1} \sum_{t=1}^{S-1} Y_{it}

2. Compute the post-treatment mean for each unit i:

   .. math::

      \bar{Y}_{i,post} = \frac{1}{T-S+1} \sum_{t=S}^{T} Y_{it}

3. Construct the transformed outcome:

   .. math::

      \Delta\bar{Y}_i = \bar{Y}_{i,post} - \bar{Y}_{i,pre}

4. Estimate the cross-sectional regression:

   .. math::

      \Delta\bar{Y}_i = \alpha + \tau D_i + U_i, \quad i = 1, \ldots, N

   where :math:`D_i = 1` for treated units and :math:`D_i = 0` for control units.

**Estimand**: :math:`\tau` is the average treatment effect on the treated (ATT).

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

   .. math::

      Y_{it} = A_i + B_i t + \varepsilon_{it} \quad \text{for } t = 1, \ldots, S-1

   This requires at least two pre-treatment periods (:math:`T_0 \geq 2`).

2. Compute predicted values for all periods using the estimated trend:

   .. math::

      \hat{Y}_{it} = \hat{A}_i + \hat{B}_i t \quad \text{for } t = 1, \ldots, T

3. For post-treatment periods, compute out-of-sample residuals:

   .. math::

      \ddot{Y}_{it} = Y_{it} - \hat{Y}_{it} \quad \text{for } t = S, \ldots, T

4. Average the residuals over post-treatment periods:

   .. math::

      \bar{\ddot{Y}}_i = \frac{1}{T-S+1} \sum_{t=S}^{T} \ddot{Y}_{it}

5. Estimate the cross-sectional regression:

   .. math::

      \bar{\ddot{Y}}_i = \alpha + \tau_{DT} D_i + U_i, \quad i = 1, \ldots, N

**Estimand**: :math:`\tau_{DT}` is the ATT after removing unit-specific linear trends.

**Note**: The trend is estimated using only pre-treatment data, so the treatment
variation remains available for estimation in the cross-sectional regression. As in
the demean case, the implementation allows unbalanced panels: units need not appear
in every period, but each unit included in the data must have at least two
pre-treatment observations so that its trend can be estimated. Units without any
post-treatment observations remain in the panel but do not contribute to the main
ATT regression because their post-treatment average is undefined.

Seasonal Transformations (demeanq/detrendq)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**demeanq**: Extends Procedure 2.1 to include seasonal fixed effects,
removing seasonal patterns in periodic data.

**detrendq**: Extends Procedure 3.1 to include both linear trends and
seasonal fixed effects.

Both methods include seasonal dummies in the pre-treatment regression to remove
seasonal variation before computing post-treatment residuals.

**Generalized Q Parameter**

The seasonal transformations support arbitrary seasonal periods through the ``Q``
parameter:

- **Q=4** (default): Quarterly data with 4 seasons per year
- **Q=12**: Monthly data with 12 seasons per year
- **Q=52**: Weekly data with 52 seasons per year

The mathematical formulation generalizes to Q seasons. For demeanq, the
pre-treatment regression for each unit i is:

.. math::

   Y_{it} = \alpha_i + \sum_{q=2}^{Q} \gamma_q S_{itq} + \varepsilon_{it}

where :math:`S_{itq}` is a dummy variable equal to 1 if observation t falls in
season q (with season 1 as the reference category).

For detrendq, the pre-treatment regression includes both trend and seasonal terms:

.. math::

   Y_{it} = \alpha_i + \beta_i t + \sum_{q=2}^{Q} \gamma_q S_{itq} + \varepsilon_{it}

**Minimum Pre-Treatment Requirements**

The minimum number of pre-treatment observations per unit depends on Q:

- **demeanq**: :math:`n_{pre} \geq Q + 1` (to estimate intercept + Q-1 seasonal dummies)
- **detrendq**: :math:`n_{pre} \geq Q + 2` (to estimate intercept + trend + Q-1 seasonal dummies)

For monthly data (Q=12), this means at least 13 pre-treatment observations for
demeanq and 14 for detrendq. For weekly data (Q=52), at least 53 and 54
observations respectively.

**Season Coverage Requirement**

For each unit, every season that appears in the post-treatment period must also
appear in the pre-treatment period. This ensures that seasonal effects can be
properly removed from post-treatment observations.

**Numerical Stability**

For high-dimensional seasonal adjustments (especially Q=52), the implementation
includes numerical stability checks:

- Condition number monitoring for the design matrix
- Warnings when the design matrix approaches singularity
- Robust OLS estimation using QR decomposition

Inference Under CLM Assumptions
--------------------------------

Classical Linear Model Assumptions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When homoskedastic OLS standard errors are used (``vce=None`` in ``lwdid``),
exact finite-sample inference is available under the classical linear model (CLM)
assumptions for the cross-sectional regression. For the demean transformation,
the model is :math:`\Delta\bar{Y}_i = \alpha + \tau D_i + U_i` (for detrending,
replace :math:`\Delta\bar{Y}_i` with :math:`\bar{\ddot{Y}}_i`). The CLM
assumptions are:

1. **Linearity**: :math:`E[U_i | D_i] = 0` (zero conditional mean)
2. **Random sampling**: Units are independently sampled
3. **No perfect collinearity**: Treatment indicator varies across units
4. **Homoskedasticity**: :math:`\text{Var}(U_i | D_i) = \sigma^2` (constant variance)
5. **Normality**: :math:`U_i | D_i \sim N(0, \sigma^2)` (conditional normality)

Under these assumptions and with homoskedastic OLS standard errors, the
t-statistic :math:`(\hat{\tau} - \tau)/\text{se}(\hat{\tau})` follows an exact
t-distribution with residual degrees of freedom equal to :math:`N - k`, where
:math:`k` is the number of estimated parameters in the cross-sectional
regression (:math:`k = 2` without controls: intercept and treatment indicator).
The normality and homoskedasticity assumptions are critical for exact inference.

Degrees of Freedom
~~~~~~~~~~~~~~~~~~

**Homoskedastic standard errors**:

.. math::

   df = N - k

where :math:`N` is the number of cross-sectional units and :math:`k` is the
number of parameters (:math:`k = 2` without controls: intercept and treatment
indicator).

**Cluster-robust standard errors**:

.. math::

   df = G - 1

where :math:`G` is the number of clusters.

The cross-sectional regression has :math:`N` observations (one per unit), not
:math:`N \times T`. With cluster-robust standard errors, degrees of freedom
equal the number of clusters minus one.

Period-Specific Effects
~~~~~~~~~~~~~~~~~~~~~~~

In addition to an overall post-treatment average effect, the Lee and Wooldridge
framework allows estimation of period-specific treatment effects by running
separate cross-sectional regressions for each post-treatment period, using the
transformed outcome in that period as the dependent variable (see equation (2.20)
in Lee and Wooldridge (2026)). In ``lwdid``, these regressions use the same
variance estimator (``vce``) and control-variable specification as the main ATT
regression.

Robust Inference
----------------

Heteroskedasticity-Robust Standard Errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When homoskedasticity is violated, heteroskedasticity-robust standard errors
provide asymptotically valid inference. The ``lwdid`` package supports the
following HC estimators:

- **HC0**: White's original heteroskedasticity-consistent estimator. Tends to
  underestimate standard errors in small samples.
- **HC1**: Degrees-of-freedom adjusted version of HC0 (applies :math:`n/(n-k)`
  correction). Equivalent to ``vce='robust'``.
- **HC2**: Leverage-adjusted estimator that divides squared residuals by
  :math:`(1 - h_{ii})` where :math:`h_{ii}` is the diagonal of the hat matrix.
- **HC3**: Small-sample adjusted version that divides squared residuals by
  :math:`(1 - h_{ii})^2`. Recommended for small to moderate samples.
- **HC4**: High-leverage adjusted estimator with adaptive exponent based on
  leverage. Use when data contains influential observations.

Robust standard errors rely on asymptotic approximations and may be less accurate
in very small samples. Recent simulation evidence (Simonsohn 2021) suggests HC3
can perform reasonably well even with small sample sizes (e.g.,
:math:`N_0 = 18`, :math:`N_1 = 2` in the cited study), though caution is
warranted with very small samples.
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

1. Compute the observed test statistic. In ``lwdid``, this is the ATT estimate
   :math:`\hat{\tau}` from the cross-sectional regression.
2. Randomly reassign treatment status across units
3. Re-estimate the model with reassigned treatment
4. Repeat steps 2-3 :math:`R` times (e.g., :math:`R = 1000`)
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

Clustering at Higher Levels
---------------------------

When the policy or treatment varies at a level higher than the unit of observation,
cluster standard errors at the policy variation level (Lee & Wooldridge 2026,
Section 8.2). This section provides guidance on choosing the appropriate clustering
level and tools for diagnosing clustering structure.

When to Cluster at Higher Levels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Principle**: Cluster at the level where treatment is assigned or varies.

**Example**: If studying a state-level policy using county-level data, cluster at
the state level because the policy varies across states, not counties::

    result = lwdid(
        data, y='outcome', ivar='county', tvar='year',
        gvar='first_treat',
        vce='cluster', cluster_var='state'
    )

**Rationale**: When treatment is assigned at a higher level (e.g., state), errors
are likely correlated within that level. Clustering at the unit level (county)
would understate standard errors because it ignores this correlation.

Minimum Cluster Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The reliability of cluster-robust inference depends on the number of clusters:

- **Recommended**: ≥ 20-30 clusters for reliable asymptotic inference
- **Warning threshold**: < 10 clusters may produce unreliable inference
- **Alternative**: Use wild cluster bootstrap when clusters are few

``lwdid`` automatically warns when the number of clusters is small:

- **< 10 clusters**: Warning with recommendation to use wild cluster bootstrap
- **10-19 clusters**: Informational message suggesting wild cluster bootstrap
- **Cluster size imbalance** (CV > 1.0): Warning about potential reliability issues

Degrees of Freedom
~~~~~~~~~~~~~~~~~~

With cluster-robust standard errors, degrees of freedom are:

.. math::

   df = G - 1

where :math:`G` is the number of clusters. This is more conservative than the
usual :math:`N - k` degrees of freedom, reflecting the effective sample size
being the number of clusters rather than the number of observations.

Diagnosing Clustering Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``diagnose_clustering()`` to analyze potential clustering options and get
recommendations::

    from lwdid import diagnose_clustering

    diag = diagnose_clustering(
        data, ivar='county',
        potential_cluster_vars=['state', 'region', 'county'],
        gvar='first_treat'
    )
    print(diag.summary())

The diagnostics include:

- **Cluster counts**: Total, treated, and control clusters for each variable
- **Cluster sizes**: Min, max, mean, median, and coefficient of variation
- **Level detection**: Whether each variable is at a higher/same/lower level than unit
- **Treatment variation**: Whether treatment varies within clusters
- **Recommendation**: Suggested clustering variable with explanation

Getting a Clustering Recommendation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a detailed recommendation with confidence scores and alternatives::

    from lwdid import recommend_clustering_level

    rec = recommend_clustering_level(
        data, ivar='county', tvar='year',
        potential_cluster_vars=['state', 'region', 'county'],
        gvar='first_treat',
        min_clusters=20
    )
    print(rec.summary())

    if rec.use_wild_bootstrap:
        print("Consider using wild_cluster_bootstrap() for inference")

The recommendation considers:

- Treatment variation level (cluster at this level when possible)
- Number of clusters (more is better, ≥20 recommended)
- Balance between treated and control clusters
- Cluster size variation (lower CV is better)

Checking Clustering Consistency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Verify that your chosen clustering level is consistent with the treatment
assignment mechanism::

    from lwdid import check_clustering_consistency

    result = check_clustering_consistency(
        data, ivar='county', cluster_var='state',
        gvar='first_treat'
    )

    if not result.is_consistent:
        print(f"Warning: {result.recommendation}")

A clustering choice is consistent when:

- Treatment does not vary within clusters (or varies minimally, < 5%)
- Cluster level is at or above the treatment variation level

If treatment varies within clusters, standard errors may be conservative
(too large), leading to under-rejection of the null hypothesis.

Wild Cluster Bootstrap
~~~~~~~~~~~~~~~~~~~~~~

When the number of clusters is small (< 20), the wild cluster bootstrap provides
more reliable inference than asymptotic cluster-robust standard errors. This
method, developed by Cameron, Gelbach, and Miller (2008), constructs a bootstrap
distribution by:

1. Estimating the original model and obtaining residuals
2. Generating cluster-level random weights
3. Creating bootstrap residuals by multiplying original residuals by weights
4. Re-estimating the model with bootstrap outcomes
5. Computing the bootstrap distribution of t-statistics

**Basic usage**::

    from lwdid.inference import wild_cluster_bootstrap

    result = wild_cluster_bootstrap(
        data, y_transformed='ydot', d='d_',
        cluster_var='state', n_bootstrap=999
    )
    print(f"ATT: {result.att:.4f}")
    print(f"Bootstrap SE: {result.se_bootstrap:.4f}")
    print(f"Bootstrap p-value: {result.pvalue:.4f}")
    print(f"95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")

**Weight types**:

- **rademacher** (default): P(w=1) = P(w=-1) = 0.5. Simplest and most common.
- **mammen**: Two-point distribution matching first three moments. Better for
  asymmetric error distributions.
- **webb**: Six-point distribution (Webb 2014). Recommended for very few clusters
  (G < 10).

**Example with Webb weights for few clusters**::

    result = wild_cluster_bootstrap(
        data, y_transformed='ydot', d='d_',
        cluster_var='state',
        weight_type='webb',  # Better for very few clusters
        n_bootstrap=999,
        seed=42  # For reproducibility
    )

**Null hypothesis imposition**:

By default (``impose_null=True``), the bootstrap constructs samples under the
null hypothesis that the treatment effect is zero. This provides better size
control for hypothesis testing. Set ``impose_null=False`` for confidence interval
construction when the null may not hold.

Clustering Workflow
~~~~~~~~~~~~~~~~~~~

A recommended workflow for choosing and validating clustering:

1. **Diagnose**: Use ``diagnose_clustering()`` to understand the data structure
2. **Recommend**: Use ``recommend_clustering_level()`` to get a recommendation
3. **Check**: Use ``check_clustering_consistency()`` to validate the choice
4. **Estimate**: Run ``lwdid()`` with the chosen ``cluster_var``
5. **Bootstrap**: If clusters < 20, use ``wild_cluster_bootstrap()`` for inference

**Complete example**::

    import pandas as pd
    from lwdid import (
        lwdid, diagnose_clustering, recommend_clustering_level,
        check_clustering_consistency
    )
    from lwdid.inference import wild_cluster_bootstrap

    # Step 1-2: Diagnose and get recommendation
    rec = recommend_clustering_level(
        data, ivar='county', tvar='year',
        potential_cluster_vars=['state', 'region'],
        gvar='first_treat'
    )

    # Step 3: Check consistency
    consistency = check_clustering_consistency(
        data, ivar='county', cluster_var=rec.recommended_var,
        gvar='first_treat'
    )

    # Step 4: Estimate with cluster-robust SE
    result = lwdid(
        data, y='outcome', ivar='county', tvar='year',
        gvar='first_treat',
        vce='cluster', cluster_var=rec.recommended_var
    )

    # Step 5: Wild bootstrap if few clusters
    if rec.use_wild_bootstrap:
        boot_result = wild_cluster_bootstrap(
            result.data_transformed, y_transformed='ydot', d='d_',
            cluster_var=rec.recommended_var, n_bootstrap=999
        )
        print(f"Bootstrap p-value: {boot_result.pvalue:.4f}")

References for Clustering
~~~~~~~~~~~~~~~~~~~~~~~~~

- Cameron, A.C. & Miller, D.L. (2015). "A Practitioner's Guide to Cluster-Robust
  Inference." *Journal of Human Resources*, 50(2), 317-372.

- Cameron, A.C., Gelbach, J.B., & Miller, D.L. (2008). "Bootstrap-based
  improvements for inference with clustered errors." *Review of Economics and
  Statistics*, 90(3), 414-427.

- Webb, M.D. (2014). "Reworking wild bootstrap based inference for clustered
  errors." *Queen's Economics Department Working Paper No. 1315*.

Large-Sample Asymptotic Inference
---------------------------------

When the cross-sectional sample size N is moderately large, asymptotic inference
based on robust standard errors becomes appropriate. Lee and Wooldridge (2025)
develops the theoretical foundation for large-sample inference in the rolling
transformation framework.

Asymptotic Theory
~~~~~~~~~~~~~~~~~

Under standard regularity conditions (independent sampling across units, finite
moments, and either correct specification of the outcome model or propensity
score model for doubly robust estimators), the ATT estimator is asymptotically
normal:

.. math::

   \sqrt{N} (\hat{\tau} - \tau) \xrightarrow{d} N(0, V)

where :math:`V` is the asymptotic variance. This justifies the use of
heteroskedasticity-robust standard errors (HC0-HC4) or cluster-robust standard
errors for constructing confidence intervals and hypothesis tests.

The key insight from Lee and Wooldridge (2025) is that the rolling transformation
converts the panel DiD problem into a cross-sectional treatment effects problem,
enabling application of standard large-sample theory from the treatment effects
literature.

Doubly Robust Estimation
~~~~~~~~~~~~~~~~~~~~~~~~

Lee and Wooldridge (2025) shows that the rolling transformation approach enables
application of doubly robust estimators, which provide consistent estimates when
either the outcome model or the propensity score model is correctly specified.

**IPWRA (Inverse Probability Weighted Regression Adjustment)**:

The doubly robust IPWRA estimator combines regression adjustment and inverse
probability weighting. For cohort :math:`g` in period :math:`r`, the estimator
takes the form:

.. math::

   \hat{\tau}_{IPWRA} = N_1^{-1} \sum_i D_i [\hat{Y}_{ir} - \hat{m}_0(X_i)]
   - N_1^{-1} \sum_i (1-D_i) \frac{\hat{p}(X_i)}{1-\hat{p}(X_i)} [\hat{Y}_{ir} - \hat{m}_0(X_i)]

where :math:`N_1` is the number of treated units, :math:`\hat{m}_0(\cdot)` is the
estimated conditional mean for control units, and :math:`\hat{p}(\cdot)` is the
estimated propensity score.

**Double robustness property**: The IPWRA estimator is consistent if:

1. The outcome model :math:`E[Y|D=0, X]` is correctly specified, OR
2. The propensity score :math:`P(D=1|X)` is correctly specified

This property makes IPWRA particularly attractive when functional form
assumptions are uncertain.

**IPW (Inverse Probability Weighting)**:

The IPW estimator weights observations by the inverse of their propensity scores:

.. math::

   \hat{\tau}_{IPW} = N_1^{-1} \sum_i D_i \hat{Y}_{ir}
   - \frac{\sum_i (1-D_i)\hat{w}_i\hat{Y}_{ir}}{\sum_i (1-D_i)\hat{w}_i}

where :math:`\hat{w}_i = \hat{p}(X_i)/(1-\hat{p}(X_i))`. IPW is consistent when
the propensity score is correctly specified.

**PSM (Propensity Score Matching)**:

Propensity score matching estimates treatment effects by matching treated units
to control units with similar propensity scores. Nearest-neighbor matching finds
the closest control unit(s) for each treated unit based on the estimated
propensity score.

Efficiency Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~

Under correct specification of all models, Lee and Wooldridge (2025) shows that:

1. **Regression adjustment (RA)** is both best linear unbiased (BLUE) and
   asymptotically efficient under standard assumptions
2. **IPWRA** achieves efficiency close to RA while providing robustness to
   model misspecification
3. **Long differencing methods** (e.g., Callaway and Sant'Anna 2021) can be
   considerably less efficient because they use only the period just prior to
   intervention

Monte Carlo Simulation Evidence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Monte Carlo simulations in Lee and Wooldridge (2025, Section 7) provide
quantitative evidence for these theoretical results. Key findings include:

**Efficiency comparison under correct specification** (Tables 7.2-7.10):

- **RA/POLS**: Relative SD = 1.00 (baseline), RMSE ratio to RA = 1.00. Best
  linear unbiased estimator.
- **IPWRA**: Relative SD = 1.03-1.05, RMSE ratio to RA ≈ 1.03. Doubly robust
  property.
- **IPW**: Relative SD = 1.08-1.15, RMSE ratio to RA ≈ 1.10. Propensity
  score-based weighting.
- **PSM**: Relative SD = 1.25-1.40, RMSE ratio to RA ≈ 1.30. Matching-based
  estimator.
- **CS(2021)**: Relative SD = 1.25-1.40, RMSE ratio to RA ≈ 1.30. Long
  differencing approach.

**Rolling vs. long differencing efficiency**:

The rolling transformation uses all pre-treatment periods to estimate
unit-specific means or trends, whereas long differencing methods (e.g.,
Callaway and Sant'Anna, 2021) use only the period immediately prior to
treatment. This difference has substantial efficiency implications:

- Rolling transformation: Uses :math:`T_0` pre-treatment periods per unit
- Long differencing: Uses only 1 pre-treatment period per unit
- Efficiency gain: Rolling achieves standard deviation approximately
  :math:`\sqrt{T_0}` times smaller than long differencing when
  :math:`T_0 > 1`, under homoskedastic errors

In the simulation designs with :math:`T_0 = 4`, the rolling approach achieves
approximately 25-40% smaller standard deviations than long differencing
methods (CS 2021), translating to substantially more precise estimates.

**Robustness to misspecification** (Tables 7.4-7.5, 7.9-7.10):

When outcome models are misspecified but propensity scores are correct:

- IPWRA maintains small bias due to double robustness property
- RA may exhibit larger bias under outcome model misspecification
- IPWRA RMSE becomes comparable to or smaller than RA RMSE

These findings support using IPWRA as the primary estimator when functional
form assumptions are uncertain.

When to Use Large-Sample Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Large-sample asymptotic inference is appropriate when:

- The cross-sectional sample size :math:`N` is moderately large (e.g.,
  :math:`N \geq 50`)
- Homoskedasticity and normality assumptions may not hold
- Doubly robust estimation is desired for robustness to model misspecification

For small samples where CLM assumptions (normality and homoskedasticity) are
plausible, exact t-based inference with ``vce=None`` remains appropriate.

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
control units. Formally, :math:`E[Y_{it}(0) - Y_{i1}(0) | D_i]` is constant
across treatment groups for all :math:`t`. This is the standard parallel trends
assumption.

**Detrend (Procedure 3.1)**: In the absence of treatment, the average change in
outcomes after removing unit-specific linear trends would be the same for treated
and control units. This allows for heterogeneous linear trends across units,
relaxing the standard parallel trends assumption.

The parallel trends assumption is not directly testable because the counterfactual
outcome (what would have happened to treated units without treatment) is unobserved.
Researchers can examine pre-treatment trends visually, conduct placebo tests on
pre-treatment periods, or use detrending when units exhibit different linear trends
in the pre-treatment period.

Common Treatment Timing (Common Timing Mode)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In common timing mode, all treated units begin treatment in the same period. This is
specified via the ``post`` parameter, which indicates post-treatment periods.

For staggered adoption settings where units are treated at different times, use the
``gvar`` parameter instead of ``post``. See the Staggered Adoption section below.

Treatment Persistence
~~~~~~~~~~~~~~~~~~~~~

Once treated, units remain treated. At the implementation level, the post-treatment
indicator must be monotone non-decreasing in the time index (once ``post`` switches
from 0 to 1, it cannot revert to 0). Treatment reversals or temporary treatments
are therefore not permitted.

Heterogeneous Trends and Assumption CHT
----------------------------------------

Lee and Wooldridge (2025, Section 5) introduces Assumption CHT (Cohort-specific
Heterogeneous Trends), which relaxes the standard parallel trends assumption to
allow for cohort-specific linear trends.

Assumption CHT
~~~~~~~~~~~~~~

Under Assumption CHT, the potential outcome without treatment follows:

.. math::

   E[Y_t(\infty)|D, X] = \eta_S(D_S \cdot t) + \cdots + \eta_T(D_T \cdot t) + q_\infty(X) + \sum_g D_g q_g(X) + m_t(X)

where:

- :math:`\eta_g` is the cohort-specific linear trend for cohort :math:`g`
- :math:`D_g = 1` indicates membership in cohort :math:`g`
- :math:`q_\infty(X)` is the baseline level for never-treated units
- :math:`q_g(X)` is the cohort-specific level shift
- :math:`m_t(X)` is the common time effect

The key implication is that different cohorts may have different pre-treatment
trends, violating the standard parallel trends assumption but still allowing
valid causal inference through detrending.

First Difference Under CHT
~~~~~~~~~~~~~~~~~~~~~~~~~~

Taking first differences under Assumption CHT:

.. math::

   E[Y_t(\infty) - Y_{t-1}(\infty)|D, X] = \eta_g D_g + \cdots + [m_t(X) - m_{t-1}(X)]

The first difference depends on cohort membership through the :math:`\eta_g` terms.
This means that if cohorts have different trends (:math:`\eta_g \neq \eta_h`),
the standard parallel trends assumption fails, but detrending can remove these
cohort-specific trends.

Choosing Between Demean and Detrend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The choice between ``demean`` and ``detrend`` depends on whether the parallel
trends assumption holds:

**Use demean when**:

- Pre-treatment trends are parallel across treatment groups
- You have limited pre-treatment periods (:math:`T_0 < 3`)
- You want more efficient estimates (demean is more efficient when PT holds)

**Use detrend when**:

- Pre-treatment trends differ across treatment groups
- You have sufficient pre-treatment periods (:math:`T_0 \geq 2`, preferably :math:`\geq 3`)
- Visual inspection or formal tests suggest heterogeneous trends

**Decision procedure**:

1. Examine pre-treatment trends visually using ``plot_cohort_trends()``
2. Test parallel trends formally using ``test_parallel_trends()``
3. Diagnose trend heterogeneity using ``diagnose_heterogeneous_trends()``
4. Get a recommendation using ``recommend_transformation()``

Procedure 5.1: Detrending Under CHT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Lee and Wooldridge (2025, Procedure 5.1) describes the detrending procedure for
staggered adoption under Assumption CHT:

**Step 1**: For each cohort :math:`g \in \{S, \ldots, T\}`, run unit-specific
regressions using only pre-treatment periods:

.. math::

   Y_{it} \text{ on } 1, t \quad \text{for } t = 1, \ldots, g-1

This estimates unit-specific intercept :math:`\hat{A}_i` and slope :math:`\hat{B}_i`.

**Step 2**: For post-treatment periods :math:`r \in \{g, \ldots, T\}`, compute
out-of-sample predictions:

.. math::

   \hat{Y}_{irg} = \hat{A}_i + \hat{B}_i \cdot r

**Step 3**: Compute detrended residuals:

.. math::

   \ddot{Y}_{irg} = Y_{ir} - \hat{Y}_{irg}

The detrended outcome :math:`\ddot{Y}_{irg}` removes the unit-specific linear
trend, allowing valid estimation of treatment effects even when cohorts have
different pre-treatment trends.

Testing Parallel Trends
~~~~~~~~~~~~~~~~~~~~~~~

The ``lwdid`` package provides tools for testing the parallel trends assumption:

**Placebo test**: Estimate treatment effects in pre-treatment periods. Under
parallel trends, these "placebo" effects should be zero:

.. code-block:: python

   from lwdid.trend_diagnostics import test_parallel_trends

   result = test_parallel_trends(
       data, y='outcome', ivar='unit', tvar='time',
       gvar='first_treat', method='placebo'
   )
   print(result.summary())

**Interpretation**:

- If ``reject_null=False``: No evidence against parallel trends; ``demean`` is appropriate
- If ``reject_null=True``: Evidence of pre-treatment differences; consider ``detrend``

Diagnosing Heterogeneous Trends
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To diagnose whether cohorts have different pre-treatment trends:

.. code-block:: python

   from lwdid.trend_diagnostics import diagnose_heterogeneous_trends

   diag = diagnose_heterogeneous_trends(
       data, y='outcome', ivar='unit', tvar='time',
       gvar='first_treat'
   )
   print(diag.summary())

The diagnostics include:

- Estimated trend slope for each cohort
- F-test for trend heterogeneity across cohorts
- Pairwise trend difference tests
- Recommendation for transformation method

Getting a Transformation Recommendation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For an automated recommendation:

.. code-block:: python

   from lwdid.trend_diagnostics import recommend_transformation

   rec = recommend_transformation(
       data, y='outcome', ivar='unit', tvar='time',
       gvar='first_treat'
   )
   print(rec.summary())

The recommendation considers:

- Parallel trends test results
- Trend heterogeneity diagnostics
- Number of pre-treatment periods
- Panel balance
- Seasonal patterns (for quarterly data)

Comparison with Other DiD Methods
----------------------------------

Pre-treatment Period Dynamics
-----------------------------

Lee & Wooldridge (2025) Appendix D develops a methodology for estimating treatment
effects in pre-treatment periods to assess the validity of the parallel trends
assumption. This section describes the theoretical foundation and implementation.

Motivation
~~~~~~~~~~

The parallel trends assumption is fundamental to difference-in-differences
identification but cannot be directly tested because it concerns counterfactual
outcomes. However, examining pre-treatment dynamics provides indirect evidence:

- Under parallel trends, the transformed outcome difference between treated and
  control groups should be zero in all pre-treatment periods
- Systematic non-zero pre-treatment effects suggest potential violations

Rolling Transformation for Pre-treatment Periods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For pre-treatment periods, the transformation uses future pre-treatment periods
rather than past periods. This ensures the transformation is well-defined for
all pre-treatment periods.

**Pre-treatment Demeaning (Formula D.1)**

For cohort :math:`g` and pre-treatment period :math:`t < g`, the demeaned outcome is:

.. math::

   \dot{Y}_{itg} = Y_{it} - \frac{1}{g-1-t} \sum_{q=t+1}^{g-1} Y_{iq}

where the sum is over future pre-treatment periods :math:`\{t+1, \ldots, g-1\}`.

Key properties:

- Uses only pre-treatment data (periods before :math:`g`)
- Rolling window looks forward, not backward
- Window size decreases as :math:`t` approaches :math:`g-1`

**Pre-treatment Detrending (Formula D.2)**

For cohort :math:`g` and pre-treatment period :math:`t < g`, fit an OLS regression:

.. math::

   Y_{iq} = A + B \cdot q + \varepsilon_{iq} \quad \text{for } q \in \{t+1, \ldots, g-1\}

Then compute the detrended outcome:

.. math::

   \ddot{Y}_{itg} = Y_{it} - (\hat{A} + \hat{B} \cdot t)

This requires at least 2 future pre-treatment periods for OLS estimation.

Anchor Point Convention
~~~~~~~~~~~~~~~~~~~~~~~

The period immediately before treatment (:math:`t = g-1`, event time :math:`e = -1`)
serves as the anchor point:

.. math::

   \dot{Y}_{i,g-1,g} = 0 \quad \text{(by construction)}

This occurs because the rolling window :math:`\{g, \ldots, g-1\}` is empty, and
by convention the transformation returns zero.

The anchor point provides:

1. A reference point for interpreting other pre-treatment effects
2. Normalization that facilitates comparison across cohorts
3. Consistency with event study visualization conventions

Pre-treatment ATT Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each cohort :math:`g` and pre-treatment period :math:`t < g`, the pre-treatment
ATT is estimated by regressing the transformed outcome on treatment status:

.. math::

   \dot{Y}_{itg} = \alpha + \tau_{gt}^{pre} D_i + U_i

where :math:`D_i = 1` for units in cohort :math:`g` and :math:`D_i = 0` for control
units (never-treated or not-yet-treated at time :math:`t`).

Under parallel trends, :math:`E[\tau_{gt}^{pre}] = 0` for all :math:`t < g`.

Control Group for Pre-treatment Periods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For pre-treatment period :math:`t < g`, valid control units are:

- **Never-treated**: Units with :math:`D_\infty = 1`
- **Not-yet-treated at t**: Units with first treatment after :math:`t`
  (cohorts :math:`h > t`)

This differs from post-treatment control groups because we need units that are
untreated at time :math:`t`, not just at time :math:`r \geq g`.

Joint Test for Parallel Trends
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The parallel trends test performs a joint F-test of the null hypothesis:

.. math::

   H_0: \tau_{gt}^{pre} = 0 \quad \text{for all } t < g-1

The anchor point (:math:`t = g-1`) is excluded because it is zero by construction.

**Test statistic**:

.. math::

   F = \frac{1}{K} \sum_{k=1}^{K} \left(\frac{\hat{\tau}_k}{\text{SE}(\hat{\tau}_k)}\right)^2

where :math:`K` is the number of pre-treatment periods tested (excluding anchor).

Under :math:`H_0`, the F-statistic follows an :math:`F(K, \nu)` distribution,
where :math:`\nu` is the degrees of freedom from the estimation.

**Interpretation**:

- Rejection suggests evidence against parallel trends
- Non-rejection does not prove parallel trends holds
- Consider effect sizes, not just statistical significance

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
large-sample approximations in panels with many groups and time periods.
Lee and Wooldridge (2025) extends the rolling transformation approach to
staggered adoption settings and develops large-sample asymptotic inference
using doubly robust estimators (IPWRA). Lee and Wooldridge (2026) focuses
on small cross-sectional sample sizes in common timing settings, showing
that exact t-based inference is available under classical linear model
assumptions (normality and homoskedasticity).



When to Use This Method
~~~~~~~~~~~~~~~~~~~~~~~

**Small-sample exact inference** (common timing with ``vce=None``):

- Small numbers of treated and/or control units
- Willingness to assume normality and homoskedasticity
- Alternative: use randomization inference (``ri=True``)

**Large-sample asymptotic inference** (common timing or staggered):

- Moderately large cross-sectional samples
- Use robust standard errors (``vce='robust'`` or ``vce='hc3'``)
- For staggered designs, use ``gvar`` to specify first treatment period

**Staggered adoption**:

- Units treated at different times
- Supports cohort-time specific effect estimation
- Multiple aggregation options (none, cohort, overall)
- Flexible control group strategies (never-treated or not-yet-treated)

**This implementation does not cover**:

- Treatment reversals or temporary treatments (violates persistence assumption)
- Continuous treatment intensity (treatment must be binary)

Practical Considerations
------------------------

Sample Size Requirements
~~~~~~~~~~~~~~~~~~~~~~~~

**Minimum (panel-level checks)**: ``lwdid`` requires at least :math:`N \geq 3`
units in the panel, with at least one treated unit (:math:`N_1 \geq 1`) and at
least one control unit (:math:`N_0 \geq 1`).

In addition, the firstpost cross-sectional regression used for ATT estimation
must contain at least three units in total and at least one treated and one
control unit. If some units drop out of the panel before the post-treatment
period, the effective cross-sectional sample entering the main regression can be
smaller than the panel :math:`N`; in such cases ``lwdid`` raises an error rather
than reporting an ATT based on an under-identified regression.

**Practical considerations**:

- Exact t-based inference under CLM assumptions is technically available at this
  minimum, but in practice inference is more stable when the cross-sectional
  sample is meaningfully larger than :math:`N = 3`.
- Cluster-robust standard errors rely on having a non-trivial number of clusters.
  A commonly used rule of thumb in applied work is to have around
  :math:`G \geq 10` clusters for more reliable cluster-robust inference,
  although there is no universal threshold.

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

- ``demean``: At least one pre-treatment period overall (:math:`T_0 \geq 1`),
  and each unit must have at least one pre-treatment observation so that its
  pre-treatment mean can be computed.
- ``detrend``: At least two pre-treatment periods overall (:math:`T_0 \geq 2`),
  and each unit must have at least two pre-treatment observations so that a
  unit-specific linear trend can be estimated.
- ``demeanq`` and ``detrendq``: The minimum number of pre-treatment observations
  depends on the seasonal period Q:

  - **demeanq**: :math:`n_{pre} \geq Q + 1` per unit (e.g., 5 for quarterly, 13 for monthly, 53 for weekly)
  - **detrendq**: :math:`n_{pre} \geq Q + 2` per unit (e.g., 6 for quarterly, 14 for monthly, 54 for weekly)

  Additionally, for each unit, every season that appears in the post-treatment
  period must also appear in the pre-treatment period; otherwise seasonal effects
  for that season cannot be removed.

**Practical recommendations**:

- :math:`T_0 \geq 3` for detrend/detrendq (more stable trend estimation)
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
controls enter the regression if :math:`N_1 > K+1` and :math:`N_0 > K+1`, where
:math:`K` is the number of control variables. Otherwise, ``lwdid`` issues a
warning and estimates the model without controls to avoid under-identified or
extremely fragile regressions. When controls are included, observations with
missing values in any included control variable are dropped from the main
regression sample; ``lwdid`` reports how many observations were removed. If
dropping those observations would cause either group to violate the
:math:`N_1 > K+1` or :math:`N_0 > K+1` conditions, controls are omitted instead
and the full sample is retained.

Computational Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Speed**: The transformation and cross-sectional regression steps are
computationally efficient for typical panel datasets. Randomization inference
requires :math:`R` replications and is more computationally intensive
(computation time scales linearly with :math:`R`).

**Memory**: Memory requirements are modest for typical panel datasets.
Randomization inference stores :math:`R` test statistics for computing the
empirical p-value.

Limitations
-----------

This implementation has the following limitations:

1. **Binary treatment**: Treatment is either on or off (no continuous treatment intensity)
2. **Time-invariant controls**: Controls must not vary over time
3. **Treatment persistence**: Once treated, units must remain treated
4. **Quarterly methods in staggered mode**: The ``demeanq`` and ``detrendq``
   transformations are only available for common timing designs

Staggered Adoption
------------------

When units are treated at different times, the staggered adoption framework applies.
This is activated by specifying the ``gvar`` parameter (first treatment period for
each unit) instead of ``post``.

Identification
~~~~~~~~~~~~~~

For treatment cohort :math:`g` (units first treated in period :math:`g`) and
calendar time :math:`r \geq g`, the ATT is identified under:

1. **No anticipation**: :math:`E[Y_t(g) - Y_t(\infty) | D_g = 1] = 0` for
   :math:`t < g`
2. **Conditional parallel trends**:
   :math:`E[Y_t(\infty) - Y_1(\infty) | D, X] = E[Y_t(\infty) - Y_1(\infty) | X]`

These assumptions ensure that never-treated and not-yet-treated units provide
valid counterfactuals for estimating treatment effects.

Transformation
~~~~~~~~~~~~~~

For each cohort :math:`g` and period :math:`r \geq g`, the transformed outcome
is:

**Demean**:

.. math::

   \dot{Y}_{irg} = Y_{ir} - \frac{1}{g-1} \sum_{s=1}^{g-1} Y_{is}

**Detrend**:

.. math::

   \ddot{Y}_{irg} = Y_{ir} - \hat{A}_i - \hat{B}_i r

where the trend coefficients are estimated from pre-treatment periods
:math:`\{1, \ldots, g-1\}`.

Control Group Strategies
~~~~~~~~~~~~~~~~~~~~~~~~

Lee and Wooldridge (2025, Section 4.1) establishes that, under the conditional
parallel trends assumption (CPTS), the cohort treatment indicators are
unconfounded with respect to the transformed potential outcome. This implies:

- For estimating :math:`\tau_{gr}` (effect for cohort :math:`g` at time
  :math:`r`), cohorts :math:`h > r` (not-yet-treated at time :math:`r`) can
  serve as valid controls in addition to never-treated units
- Equation (4.8) in the paper shows:
  :math:`E[\dot{Y}_{rg}(\infty)|D_\infty=1, X] = E[\dot{Y}_{rg}(\infty)|D_h=1, X]`
  for all :math:`h`, meaning the conditional expectation is the same across all
  cohorts
- Equation (4.9) further shows that, by no anticipation,
  :math:`E[\dot{Y}_{rg}(\infty)|D_h=1, X] = E[\dot{Y}_{rg}(h)|D_h=1, X]` for
  :math:`h > r`, so not-yet-treated units can substitute for never-treated units

**Strategies**:

- **never_treated**: Only units with :math:`D_\infty = 1` (never treated during
  observation period)
- **not_yet_treated**: Never treated plus units with first treatment after
  period :math:`r` (i.e., cohorts :math:`h \in \{r+1, \ldots, T, \infty\}`)

The not-yet-treated strategy uses more control observations, potentially improving
efficiency, while the never-treated strategy may be more robust to violations of
no anticipation.

All Units Eventually Treated
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Lee and Wooldridge (2025, Section 4.2) addresses the case where no units remain
untreated through period :math:`T` (no never-treated group). In this setting:

1. Treatment effects are defined relative to :math:`Y_t(T)`, the state of being
   treated only in the final period, rather than :math:`Y_t(\infty)`
2. Effects can only be estimated for cohorts :math:`g \in \{S, \ldots, T-1\}`;
   no effect can be estimated for the final cohort (:math:`g = T`) because there
   are no control units
3. By no anticipation, for :math:`r < T`:
   :math:`E[Y_r(g) - Y_r(T)|D_g = 1] = E[Y_r(g) - Y_r(\infty)|D_g = 1]`, so
   except for the final period the ATTs have the same interpretation as when a
   never-treated group exists
4. When :math:`r = T`, the only available control is cohort :math:`g = T` (units
   first treated in the final period)

The implementation handles this automatically: when aggregating to cohort or overall
effects, only cohorts with valid control groups are included.

Estimation Methods
~~~~~~~~~~~~~~~~~~

In staggered mode, multiple estimators are available:

- **ra** (Regression Adjustment): OLS on transformed outcome
- **ipw** (Inverse Probability Weighting): Propensity score weighting
- **ipwra** (Doubly Robust): Combines regression and IPW
- **psm** (Propensity Score Matching): Nearest neighbor matching

The doubly robust IPWRA estimator is consistent if either the outcome model or
propensity score model is correctly specified, making it particularly attractive
when functional form assumptions are uncertain.

Inference Distribution by Estimator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Different estimators use different reference distributions for inference:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Estimator
     - Distribution
     - Rationale
   * - RA (OLS)
     - t-distribution
     - Exact inference under CLM assumptions; df = N - k
   * - IPW
     - t-distribution
     - Structure similar to OLS; conservative in small samples
   * - IPWRA
     - Normal
     - Asymptotic inference based on influence functions
   * - PSM
     - Normal
     - Asymptotic inference based on Abadie-Imbens SE

**Comparison with Stata**

Stata's ``teffects`` commands use the normal distribution for all estimators
(``teffects ipw``, ``teffects ipwra``, ``teffects psmatch``). This implementation
differs for IPW:

- **IPW**: Uses t-distribution (differs from Stata)
- **IPWRA**: Uses normal distribution (matches Stata)
- **PSM**: Uses normal distribution (matches Stata)

The IPW design decision is based on:

1. Lee and Wooldridge (2026) equation (2.10) establishes t-distribution inference
   for cross-sectional regressions under CLM assumptions
2. The IPW estimator structure resembles a simple mean difference, analogous to
   OLS regression
3. In small samples, the t-distribution provides more conservative inference with
   better confidence interval coverage
4. As sample size increases, the t-distribution converges to the normal
   distribution, preserving asymptotic properties

For large samples (N > 50), the difference between t and normal distributions
is negligible. For small samples, the t-distribution provides wider confidence
intervals and larger p-values, which may be more appropriate given the
uncertainty in variance estimation.

Aggregation
~~~~~~~~~~~

Cohort-time specific effects :math:`\tau_{gr}` can be aggregated:

- **none**: Report :math:`(g,r)`-specific effects only
- **cohort**: Average effects within each cohort:

  .. math::

     \tau_g = \frac{1}{T-g+1} \sum_{r=g}^{T} \tau_{gr}

- **overall**: Cohort-share weighted average:

  .. math::

     \tau_\omega = \sum_g \omega_g \tau_g \quad \text{where } \omega_g = N_g/N_{treat}

Robustness to Pre-treatment Period Selection
--------------------------------------------

Lee and Wooldridge (2026) Section 8.1 recommends studying the robustness of DiD
estimates by varying the number of pre-treatment periods used in the transformation.
This section describes the sensitivity analysis tools implemented in ``lwdid``.

Motivation
~~~~~~~~~~

The rolling transformation approach uses pre-treatment periods to estimate
unit-specific means or trends. The choice of how many pre-treatment periods to
include can affect the estimates:

- **Too few periods**: May not adequately capture unit-specific patterns
- **Too many periods**: May include periods where parallel trends do not hold

From Lee and Wooldridge (2026):

    "With synthetic control-type approaches and the approaches we suggest here,
    one can study the robustness of the findings by adjusting the number of
    pre-treatment time periods."

Sensitivity Ratio
~~~~~~~~~~~~~~~~~

The sensitivity ratio measures how much ATT estimates vary across specifications
with different numbers of pre-treatment periods:

.. math::

   \text{Sensitivity Ratio} = \frac{\max_k \hat{\tau}_k - \min_k \hat{\tau}_k}{|\hat{\tau}_{\text{baseline}}|}

where :math:`\hat{\tau}_k` is the ATT estimate using :math:`k` pre-treatment periods,
and :math:`\hat{\tau}_{\text{baseline}}` is the estimate using all available
pre-treatment periods.

**Interpretation**:

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Sensitivity Ratio
     - Robustness Level
     - Interpretation
   * - < 10%
     - Highly Robust
     - Estimates stable across specifications
   * - 10-25%
     - Moderately Robust
     - Some sensitivity, generally acceptable
   * - 25-50%
     - Sensitive
     - Interpret with caution
   * - ≥ 50%
     - Highly Sensitive
     - Results depend heavily on specification

Using robustness_pre_periods()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``robustness_pre_periods()`` function implements this sensitivity analysis::

    from lwdid import robustness_pre_periods

    result = robustness_pre_periods(
        data, y='outcome', ivar='unit', tvar='year',
        gvar='first_treat', rolling='detrend',
        pre_period_range=(3, 10)
    )

    print(result.summary())
    result.plot()

The function:

1. Estimates ATT using different numbers of pre-treatment periods
2. Computes the sensitivity ratio
3. Assesses robustness level
4. Generates recommendations

No-Anticipation Sensitivity
~~~~~~~~~~~~~~~~~~~~~~~~~~~

When policy is announced before implementation, units may adjust behavior in
anticipation. The ``sensitivity_no_anticipation()`` function tests robustness
by excluding periods immediately before treatment::

    from lwdid import sensitivity_no_anticipation

    result = sensitivity_no_anticipation(
        data, y='outcome', ivar='unit', tvar='year',
        gvar='first_treat', max_anticipation=3
    )

    if result.anticipation_detected:
        print(f"Consider excluding {result.recommended_exclusion} periods")

The function:

1. Estimates ATT excluding 0, 1, 2, ... periods before treatment
2. Detects significant changes in estimates
3. Recommends how many periods to exclude

Using exclude_pre_periods Parameter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Based on sensitivity analysis results, you can exclude periods in the main
estimation using the ``exclude_pre_periods`` parameter::

    from lwdid import lwdid

    # Exclude 2 periods immediately before treatment
    result = lwdid(
        data, y='outcome', d='d', ivar='unit', tvar='year', post='post',
        rolling='demean', exclude_pre_periods=2
    )

This parameter:

- Excludes the specified number of pre-treatment periods from transformation
- Addresses potential anticipation effects
- Implements the robustness check from Lee and Wooldridge (2026) Section 8.1

Comprehensive Sensitivity Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``sensitivity_analysis()`` function provides a comprehensive assessment::

    from lwdid import sensitivity_analysis

    result = sensitivity_analysis(
        data, y='outcome', ivar='unit', tvar='year',
        gvar='first_treat',
        analyses=['pre_periods', 'anticipation']
    )

    print(result.summary())
    result.plot_all()

This function combines multiple sensitivity analyses and provides an overall
assessment of estimate robustness.

References
----------

Lee, S. J., and Wooldridge, J. M. (2026). Simple Approaches to Inference with
Difference-in-Differences Estimators with Small Cross-Sectional Sample Sizes.
*Available at SSRN 5325686*.

Lee, S. J., and Wooldridge, J. M. (2025). A Simple Transformation Approach to
Difference-in-Differences Estimation for Panel Data.
*Available at SSRN 4516518*.

Authors
-------

Xuanyu Cai, Wenli Xu

Further Reading
---------------

- :doc:`user_guide` - Comprehensive usage guide
- :doc:`quickstart` - Quick start tutorial
- :doc:`api/index` - Complete API reference

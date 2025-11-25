Estimation Module (estimation)
===============================

The estimation module implements the core regression and inference procedures
for the Lee and Wooldridge (2025) difference-in-differences method.

.. automodule:: lwdid.estimation
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

This module provides two main estimation functions:

1. ``estimate_att``: Estimates the average treatment effect on the treated (ATT)
2. ``estimate_period_effects``: Estimates period-specific treatment effects

Both functions run OLS regressions on transformed data and compute standard
errors using various variance estimators.

Estimation Functions
--------------------

estimate_att()
~~~~~~~~~~~~~~

**Purpose:** Estimate the overall average treatment effect on the treated (ATT)
from the cross-sectional representation of the Lee and Wooldridge (2025)
estimator.

**Regression specification (conceptual):**

.. code-block:: text

   y_i = α + τ·D_i + Z_i'β + ε_i

where:

- y_i: Transformed outcome for unit i (typically the post-treatment
  average of the residualized outcome constructed by the transformation
  module)
- D_i: Treatment indicator (1 = treated, 0 = control)
- Z_i: Optional time-invariant controls and their interactions constructed
  following Lee and Wooldridge (2025, Section 2.2)
- ε_i: Regression error term

**Estimand:** τ is the ATT.

**Returns:**

- ATT estimate
- Standard error
- t-statistic
- p-value
- Confidence interval
- Degrees of freedom

estimate_period_effects()
~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:** Estimate treatment effects separately for each post-treatment
period using cross-sectional regressions.

**Regression specification (for each post-treatment period t):**

.. code-block:: text

   y_it = α_t + τ_t·D_i + Z_i'β_t + ε_it

where:

- y_it: Transformed outcome for unit i in period t
- D_i: Treatment indicator (1 = treated, 0 = control)
- Z_i: Optional time-invariant controls (and their interactions) re-used
  from the main regression
- τ_t: Treatment effect in period t (period-specific ATT)

**Returns:** DataFrame with period-specific estimates, standard errors,
t-statistics, p-values, and confidence intervals.

Variance Estimators
-------------------

The module supports multiple variance estimators for different assumptions
about the error structure.

OLS (Homoskedastic)
~~~~~~~~~~~~~~~~~~~

**Assumption:** Errors are homoskedastic and normally distributed.

**Formula:**

.. code-block:: text

   Var(β̂) = σ̂² (X'X)⁻¹

where σ̂² = RSS / (n - k)

**Degrees of freedom:**

- Non-clustered: df = n - k

**When to use:** When homoskedasticity and normality are plausible and exact t-based inference is desired.

HC1 (Heteroskedasticity-Robust)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Assumption:** Errors may be heteroskedastic.

**Formula:**

.. code-block:: text

   Var(β̂) = (X'X)⁻¹ (Σᵢ x̂ᵢx̂ᵢ' ε̂ᵢ²) (X'X)⁻¹ × n/(n-k)

**Degrees of freedom:** Same as OLS.

**When to use:** Medium to large samples with suspected heteroskedasticity.

HC3 (Small-Sample Adjusted)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Assumption:** Errors may be heteroskedastic.

**Formula:**

.. code-block:: text

   Var(β̂) = (X'X)⁻¹ (Σᵢ x̂ᵢx̂ᵢ' ε̂ᵢ²/(1-hᵢᵢ)²) (X'X)⁻¹

where hᵢᵢ is the i-th diagonal element of the hat matrix.

**Degrees of freedom:** Same as OLS.

**When to use:** Small or moderate samples with suspected heteroskedasticity.
Simulation evidence suggests HC3 can perform reasonably well in some small-sample designs, but results can still be sensitive when the number of treated or control units is very small. See :doc:`../methodological_notes` for further discussion.

Cluster-Robust
~~~~~~~~~~~~~~

**Assumption:** Errors are correlated within clusters but independent across
clusters.

**Formula:**

.. code-block:: text

   Var(β̂) = (X'X)⁻¹ (Σ_g X_g' ε̂_g ε̂_g' X_g) (X'X)⁻¹

where g indexes clusters.

**Degrees of freedom:** G - 1 (number of clusters minus 1).

**When to use:** Errors are clustered (e.g., students within schools).

Implementation Details
----------------------

Regression Procedure
~~~~~~~~~~~~~~~~~~~~

1. **Prepare design matrix:** For the main regression, construct a
   cross-sectional design matrix with an intercept, the treatment
   indicator, and (when applicable) time-invariant controls and their
   interactions with treatment as in Lee and Wooldridge (2025, Section 2.2).
   The same control specification is reused for period-by-period
   regressions.
2. **Run OLS:** Compute β̂ = (X'X)⁻¹ X'y
3. **Compute residuals:** ε̂ = y - Xβ̂
4. **Compute variance:** Use appropriate variance estimator
5. **Inference:** Compute t-statistics and p-values using t-distribution

Degrees of Freedom
~~~~~~~~~~~~~~~~~~

**Non-clustered:**

df = n - k

where:

- n: Number of observations
- k: Number of parameters (intercept + regressors)

**Clustered:**

df = G - 1

where G is the number of clusters.

**Rationale:** With cluster-robust SEs, the effective sample size is the
number of clusters, not observations.

Confidence Intervals
~~~~~~~~~~~~~~~~~~~~

95% confidence intervals are computed as:

.. code-block:: text

   CI = β̂ ± t_{α/2, df} × SE(β̂)

where t_{α/2, df} is the critical value from the t-distribution with df
degrees of freedom.

Technical Notes
---------------

Numerical Stability
~~~~~~~~~~~~~~~~~~~

The module relies on the numerically stable OLS implementation in ``statsmodels``:

- OLS estimation and variance–covariance computation are delegated to ``statsmodels``
- Robust variance estimators with small-sample adjustments (HC1/HC3)
- Singular or ill-conditioned designs raise errors or warnings rather than failing silently

Missing Data
~~~~~~~~~~~~

- Observations with missing values in required variables (outcome,
  treatment indicator, unit identifier, time variables, post indicator)
  are dropped during validation.
- For control variables, missing values are handled at the estimation
  stage: if dropping observations with missing controls still leaves
  enough treated and control units to satisfy the N₁ > K+1 and
  N₀ > K+1 conditions, those observations are removed and controls are
  included; otherwise, controls are omitted and the full regression
  sample is retained. In both cases, informative warnings are issued.
- The effective sample size (n) reported in the results corresponds to
  the cross-sectional regression sample used for ATT estimation (the
  ``firstpost`` cross-section).

Perfect Collinearity
~~~~~~~~~~~~~~~~~~~~~

Regressors must not be perfectly collinear for the OLS problem to be
identified. Common sources of exact collinearity include:

- Including both a variable and its exact linear transformation
- Including dummy variables for all categories (no omitted category)
- Including controls that are exact linear combinations of other regressors

Example Usage
-------------

These functions are used internally by :func:`lwdid.lwdid` after the
transformation step has constructed the transformed outcomes and main
regression sample. They are not part of the typical user-facing API. For
most applications, you should call :func:`lwdid.lwdid` directly and rely on
its high-level interface. Advanced users who need low-level access can
consult the docstrings and source code in :mod:`lwdid.estimation` to see the
exact function signatures and required inputs.

See Also
--------

- :func:`lwdid.lwdid` - Main function that calls these estimation functions
- :doc:`transformations` - Transformation functions applied before estimation
- :doc:`../methodological_notes` - Theoretical background on inference
- :doc:`../user_guide` - Comprehensive usage guide

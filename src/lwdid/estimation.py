"""
Cross-sectional OLS estimation for difference-in-differences analysis.

This module implements OLS regression on transformed outcome variables for
estimating average treatment effects on the treated (ATT) in common timing
difference-in-differences designs. After unit-specific time-series
transformations remove pre-treatment patterns, the estimation problem
reduces to standard cross-sectional regression.

The module provides three main functions:

- ``prepare_controls`` : Constructs centered controls and interaction terms
  for regression adjustment.
- ``estimate_att`` : Estimates the ATT using cross-sectional OLS on the
  first post-treatment period.
- ``estimate_period_effects`` : Estimates period-specific ATTs via
  independent cross-sectional regressions for each post-treatment period.

Supported variance-covariance estimators include homoskedastic OLS,
heteroskedasticity-robust (HC1, HC3), and cluster-robust standard errors.

Notes
-----
The transformation-based approach converts the panel data DiD problem into
a cross-sectional treatment effects problem. Under no anticipation and
parallel trends assumptions, the ATT is identified as the coefficient on
the treatment indicator in OLS regression of the transformed outcome.

For small samples, homoskedastic standard errors with t-distribution
critical values provide exact inference under normality. HC3 standard
errors offer improved finite-sample performance over HC1 when sample sizes
are moderate but asymptotic theory may not apply.

Cluster-robust inference uses G-1 degrees of freedom (where G is the number
of clusters) rather than the residual degrees of freedom, which provides
more conservative inference when the number of clusters is small.
"""

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

from .exceptions import InvalidParameterError, InvalidVCETypeError, InsufficientDataError


def prepare_controls(
    data: pd.DataFrame,
    d: str,
    ivar: str,
    controls: list[str],
    N_treated: int,
    N_control: int,
    data_sample: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Construct centered controls and interactions for regression adjustment.

    Verifies that sample sizes satisfy N_treated > K+1 and N_control > K+1,
    where K is the number of control variables. If conditions are met,
    computes the mean of control variables for the treated group, centers
    the controls by subtracting these means, and creates interaction terms
    between the treatment indicator and the centered controls.

    Parameters
    ----------
    data : pd.DataFrame
        Transformed panel data containing control variables and treatment
        indicators.
    d : str
        Name of the treatment indicator column.
    ivar : str
        Name of the unit identifier column.
    controls : list[str]
        List of control variable names. Must be numeric and time-invariant.
    N_treated : int
        Number of treated units in the regression sample.
    N_control : int
        Number of control units in the regression sample.
    data_sample : pd.DataFrame | None, optional
        Specific sample to use for computing means (e.g., first-post-treatment
        cross-section). If None, uses the full ``data`` filtered by treatment
        status.

    Returns
    -------
    dict[str, Any]
        A dictionary containing:

        - 'include' : bool
            Whether controls can be included in regression.
        - 'X_centered' : pd.DataFrame or None
            Centered control variables (X - X_mean_treated).
        - 'interactions' : pd.DataFrame or None
            Interaction terms (D * X_centered).
        - 'X_mean_treated' : dict
            Means of controls for the treated group.
        - 'RHS_varnames' : list of str
            Names of all right-hand side control variables.

    See Also
    --------
    estimate_att : Uses prepared controls for ATT estimation.
    estimate_period_effects : Uses prepared controls for period-specific ATTs.

    Notes
    -----
    Centering controls at the treated-group mean ensures that the coefficient
    on the treatment indicator directly estimates the ATT at the mean covariate
    values of the treated group. The regression model is:

    .. math::

        \\hat{Y}_i = \\alpha + \\tau D_i + X_i \\beta +
        D_i (X_i - \\bar{X}_1) \\delta + U_i

    where :math:`\\bar{X}_1 = N_1^{-1} \\sum_{i: D_i=1} X_i` is the
    treated-group mean of covariates.

    The sample size conditions (N_treated > K+1 and N_control > K+1) ensure
    that both treated and control subsamples have sufficient degrees of
    freedom for separate slope estimation.
    """
    K_ctrl = len(controls)
    nk = K_ctrl + 1
    
    # Check sample size conditions to ensure sufficient df for separate slopes.
    if N_treated > nk and N_control > nk:
        # Compute unit-weighted treated-group mean (not row-weighted).
        # Using groupby().first() avoids double-counting units with multiple
        # time periods in the sample, since X is time-invariant by assumption.
        X_mean_treated = {}
        for x in controls:
            if data_sample is not None:
                X_mean_treated[x] = (
                    data_sample[data_sample[d] == 1]
                    .groupby(ivar)[x]
                    .first()
                    .mean()
                )
            else:
                X_mean_treated[x] = (
                    data[data[d] == 1]
                    .groupby(ivar)[x]
                    .first()
                    .mean()
                )
        
        # Center controls at treated-group mean so the treatment coefficient
        # directly estimates ATT evaluated at mean X of treated units.
        X_centered = pd.DataFrame({
            f'{x}_c': data[x] - X_mean_treated[x]
            for x in controls
        })
        
        # Interaction terms allow heterogeneous treatment effects across X.
        # D * (X - X_bar_1) captures how the treatment effect varies with X.
        interactions = pd.DataFrame({
            f'd_{x}_c': data[d] * X_centered[f'{x}_c']
            for x in controls
        })
        
        RHS_varnames = controls + [f'd_{x}_c' for x in controls]
        
        return {
            'include': True,
            'X_centered': X_centered,
            'interactions': interactions,
            'X_mean_treated': X_mean_treated,
            'RHS_varnames': RHS_varnames
        }
    else:
        warnings.warn(
            f"Controls not applied: sample does not satisfy N_1 > K+1 and N_0 > K+1\n"
            f"  (where K={K_ctrl} is the number of control variables)\n"
            f"  Found: N_1={N_treated}, K+1={nk}, condition N_1 > K+1: {N_treated > nk}\n"
            f"         N_0={N_control}, K+1={nk}, condition N_0 > K+1: {N_control > nk}\n"
            f"  Controls will be ignored in the regression.",
            UserWarning,
            stacklevel=3
        )
        
        return {
            'include': False,
            'X_centered': None,
            'interactions': None,
            'X_mean_treated': {},
            'RHS_varnames': []
        }


def estimate_att(
    data: pd.DataFrame,
    y_transformed: str,
    d: str,
    ivar: str,
    controls: list[str] | None,
    vce: str | None,
    cluster_var: str | None,
    sample_filter: pd.Series,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Estimate Average Treatment Effect on the Treated (ATT) via cross-sectional OLS.

    Performs an OLS regression on the specified cross-section (typically the
    first post-treatment period) using the transformed outcome variable.
    Supports homoskedastic, heteroskedasticity-robust (HC1, HC3), and
    cluster-robust standard errors.

    Parameters
    ----------
    data : pd.DataFrame
        Transformed panel data containing the dependent variable and
        regressors.
    y_transformed : str
        Name of the transformed dependent variable (e.g., 'ydot_postavg').
    d : str
        Name of the treatment indicator column (typically ``d_``).
    ivar : str
        Name of the unit identifier column.
    controls : list[str] | None
        List of time-invariant control variables. Controls are included only
        if sample size conditions (N_1 > K+1 and N_0 > K+1) are met.
    vce : str | None
        Type of variance-covariance estimator:

        - None : Homoskedastic standard errors (OLS).
        - 'robust' or 'hc1' : Heteroskedasticity-robust (HC1).
        - 'hc3' : Heteroskedasticity-robust (HC3), better for small samples.
        - 'cluster' : Cluster-robust. Requires ``cluster_var``.
    cluster_var : str | None
        Name of the variable to use for clustering. Required if
        ``vce='cluster'``.
    sample_filter : pd.Series
        Boolean mask indicating the regression sample (e.g., first
        post-treatment period).
    alpha : float, default=0.05
        Significance level for confidence intervals (0.05 gives 95% CI).

    Returns
    -------
    dict[str, Any]
        A dictionary containing estimation results:

        - 'att' : float
            Estimated ATT.
        - 'se_att' : float
            Standard error of the ATT.
        - 't_stat' : float
            t-statistic for H0: ATT = 0.
        - 'pvalue' : float
            Two-sided p-value.
        - 'ci_lower', 'ci_upper' : float
            Confidence interval bounds.
        - 'params' : pd.Series
            All regression coefficients.
        - 'bse' : pd.Series
            Standard errors of all coefficients.
        - 'vcov' : pd.DataFrame
            Variance-covariance matrix.
        - 'resid' : np.ndarray
            Residuals.
        - 'nobs' : int
            Number of observations.
        - 'df_resid' : int
            Residual degrees of freedom.
        - 'df_inference' : int
            Degrees of freedom used for inference.
        - 'vce_type' : str
            Type of VCE used.
        - 'cluster_var' : str or None
            Clustering variable if used.
        - 'n_clusters' : int or None
            Number of clusters if clustered.
        - 'controls_used' : bool
            Whether controls were included.
        - 'controls' : list of str
            Controls used in regression.
        - 'controls_spec' : dict
            Details of control preparation.
        - 'n_treated_sample', 'n_control_sample' : int
            Sample sizes by treatment status.

    Raises
    ------
    InsufficientDataError
        If sample size is less than 3, or if no treated or control units
        exist in the regression sample.
    InvalidParameterError
        If cluster_var is missing when vce='cluster', or if cluster variable
        is not nested within units.
    InvalidVCETypeError
        If an unrecognized vce type is specified.

    See Also
    --------
    prepare_controls : Prepares centered controls for regression adjustment.
    estimate_period_effects : Estimates period-specific ATTs.

    Notes
    -----
    The ATT is estimated as the coefficient on the treatment indicator D in
    the OLS regression:

    .. math::

        \\hat{Y}_i = \\alpha + \\tau D_i + X_i \\beta +
        D_i (X_i - \\bar{X}_1) \\delta + U_i

    Confidence intervals use t-distribution critical values:

    .. math::

        CI = \\hat{\\tau} \\pm t_{df, 1-\\alpha/2} \\cdot SE(\\hat{\\tau})

    Degrees of freedom selection:

    - For non-clustered standard errors: df = n - k (residual df).
    - For cluster-robust standard errors: df = G - 1, where G is the number
      of clusters. This provides more conservative inference when the number
      of clusters is small.

    Under homoskedasticity and normality, the t-statistic has an exact
    t-distribution, enabling valid inference even with small samples. HC3
    is recommended over HC1 for moderate sample sizes as it provides better
    finite-sample performance.
    """
    data_sample = data[sample_filter].copy()

    if len(data_sample) < 3:
        raise InsufficientDataError(
            f"Insufficient sample size for regression: N={len(data_sample)} (need >= 3)"
        )

    N_treated_sample = int(data_sample[d].sum())
    N_control_sample = int(len(data_sample) - N_treated_sample)

    if N_treated_sample == 0:
        raise InsufficientDataError(
            f"Firstpost cross-section contains no treated units (N_treated=0, N_control={N_control_sample}). "
            f"ATT estimation requires at least one treated unit. "
            f"This typically occurs when all treated units exit the panel before the post-treatment period."
        )

    if N_control_sample == 0:
        raise InsufficientDataError(
            f"Firstpost cross-section contains no control units (N_treated={N_treated_sample}, N_control=0). "
            f"ATT estimation requires at least one control unit for comparison. "
            f"This typically occurs when all control units exit the panel before the post-treatment period."
        )

    controls_spec = None
    controls_used = False

    if controls is not None and len(controls) > 0:
        controls_missing = data_sample[controls].isna().any(axis=1)
        n_missing = controls_missing.sum()

        if n_missing > 0:
            data_sample_clean = data_sample[~controls_missing].copy()

            N_treated_clean = int(data_sample_clean[d].sum())
            N_control_clean = int(len(data_sample_clean) - N_treated_clean)

            K_ctrl = len(controls)
            nk = K_ctrl + 1

            if N_treated_clean > nk and N_control_clean > nk:
                warnings.warn(
                    f"Dropped {n_missing} observations from the regression sample due to missing control variables.\n"
                    f"  Original sample size: {len(data_sample)}\n"
                    f"  Final sample size: {len(data_sample_clean)}\n"
                    f"  Controls will be included in the regression.",
                    UserWarning,
                    stacklevel=3
                )

                data_sample = data_sample_clean
                N_treated = N_treated_clean
                N_control = N_control_clean

                controls_spec = prepare_controls(data, d, ivar, controls, N_treated, N_control, data_sample)

                X_centered_sample = controls_spec['X_centered'].loc[data_sample.index]
                interactions_sample = controls_spec['interactions'].loc[data_sample.index]

                X_parts = [
                    data_sample[[d]],
                    data_sample[controls],
                    interactions_sample
                ]
                X_df = pd.concat(X_parts, axis=1)
                X = sm.add_constant(X_df.astype(float).values)
                controls_used = True
            else:
                warnings.warn(
                    f"Control variables not included: after dropping {n_missing} observations with missing controls, "
                    f"the sample would not satisfy N_1 > K+1 and N_0 > K+1.\n"
                    f"  Would have: N_1={N_treated_clean}, N_0={N_control_clean}, K+1={nk}\n"
                    f"  Controls will be ignored, and observations with missing controls will be retained.",
                    UserWarning,
                    stacklevel=3
                )

                d_vals = data_sample[d].astype(float).values
                X = sm.add_constant(d_vals)
                controls_used = False
        else:
            N_treated = int(data_sample[d].sum())
            N_control = int(len(data_sample) - N_treated)

            controls_spec = prepare_controls(data, d, ivar, controls, N_treated, N_control, data_sample)

            if controls_spec['include']:
                X_centered_sample = controls_spec['X_centered'].loc[data_sample.index]
                interactions_sample = controls_spec['interactions'].loc[data_sample.index]

                X_parts = [
                    data_sample[[d]],
                    data_sample[controls],
                    interactions_sample
                ]
                X_df = pd.concat(X_parts, axis=1)
                X = sm.add_constant(X_df.astype(float).values)
                controls_used = True
            else:
                d_vals = data_sample[d].astype(float).values
                X = sm.add_constant(d_vals)
                controls_used = False
    else:
        d_vals = data_sample[d].astype(float).values
        X = sm.add_constant(d_vals)
        controls_used = False

    y_vals = data_sample[y_transformed].values

    model = sm.OLS(y_vals, X)
    
    # Normalize vce parameter (case-insensitive)
    vce_lower = vce.lower() if vce else None
    
    if vce_lower is None:
        results = model.fit()
    elif vce_lower == 'hc0':
        # HC0: basic heteroskedasticity-robust, no small-sample adjustment
        results = model.fit(cov_type='HC0')
    elif vce_lower in ('hc1', 'robust'):
        # HC1: degrees-of-freedom correction n/(n-k)
        n_treated = int((data_sample[d].astype(float).values == 1).sum())
        n_control = int((data_sample[d].astype(float).values == 0).sum())

        if n_treated < 2 or n_control < 2:
            warnings.warn(
                f"HC1 (robust) may be unstable with very small samples "
                f"(N_treated={n_treated}, N_control={n_control}). "
                f"Consider using vce=None for exact inference under normality.",
                UserWarning,
                stacklevel=3
            )

        results = model.fit(cov_type='HC1')
    elif vce_lower == 'hc2':
        # HC2: leverage-adjusted
        results = model.fit(cov_type='HC2')
    elif vce_lower == 'hc3':
        # HC3: small-sample adjusted, suitable for moderate N
        n_treated = int((data_sample[d].astype(float).values == 1).sum())
        n_control = int((data_sample[d].astype(float).values == 0).sum())
        
        if n_treated < 2 or n_control < 2:
            warnings.warn(
                f"HC3 may be unstable with very small samples "
                f"(N_treated={n_treated}, N_control={n_control}). "
                f"Consider using vce=None for exact inference under normality.",
                UserWarning,
                stacklevel=3
            )
        
        results = model.fit(cov_type='HC3')
    elif vce_lower == 'hc4':
        # HC4: adaptive leverage correction for high-leverage observations.
        # statsmodels does not support HC4; compute manually.
        results_ols = model.fit()
        
        # Get OLS estimates
        n_obs = len(y_vals)
        k = X.shape[1]
        residuals = results_ols.resid
        
        # Compute leverage values efficiently
        # Use pinv (pseudo-inverse) to handle singular/near-singular matrices
        try:
            XtX_inv = np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            # Fall back to pseudo-inverse for singular matrices
            XtX_inv = np.linalg.pinv(X.T @ X)
            warnings.warn(
                "Design matrix is singular or near-singular. "
                "Using pseudo-inverse for HC4 variance estimation. "
                "Results may be unreliable due to collinearity.",
                UserWarning,
                stacklevel=3
            )
        tmp = X @ XtX_inv
        h_ii = (tmp * X).sum(axis=1)
        h_ii = np.clip(h_ii, 0, 0.9999)
        
        # HC4 adjustment: δ_i = min(4, n·h_ii/k)
        delta = np.minimum(4.0, n_obs * h_ii / k)
        delta = np.maximum(delta, 0.0)
        
        # Omega diagonal
        omega_diag = (residuals ** 2) / ((1 - h_ii) ** delta)
        
        # Sandwich variance
        meat = X.T @ np.diag(omega_diag) @ X
        var_beta = XtX_inv @ meat @ XtX_inv
        
        # Construct a results wrapper with HC4 standard errors.
        class HC4Results:
            def __init__(self, ols_results, var_beta):
                self.params = ols_results.params
                # Handle both Series and ndarray for params
                if hasattr(ols_results.params, 'index'):
                    self.bse = pd.Series(np.sqrt(np.diag(var_beta)), index=ols_results.params.index)
                else:
                    self.bse = np.sqrt(np.diag(var_beta))
                self.tvalues = self.params / self.bse
                self.pvalues = 2 * (1 - stats.t.cdf(np.abs(self.tvalues), n_obs - k))
                self.df_resid = ols_results.df_resid
                self.nobs = ols_results.nobs
                self.rsquared = ols_results.rsquared
                self.rsquared_adj = ols_results.rsquared_adj
                self.fvalue = ols_results.fvalue
                self.f_pvalue = ols_results.f_pvalue
                self.resid = ols_results.resid
                self._cov_params = var_beta
                
            def conf_int(self, alpha=0.05):
                t_crit = stats.t.ppf(1 - alpha/2, self.df_resid)
                ci_lower = self.params - t_crit * self.bse
                ci_upper = self.params + t_crit * self.bse
                return pd.DataFrame({'lower': ci_lower, 'upper': ci_upper})
            
            def cov_params(self):
                return self._cov_params
        
        results = HC4Results(results_ols, var_beta)
    elif vce_lower == 'cluster':
        if cluster_var is None:
            raise InvalidParameterError(
                "vce='cluster' requires cluster_var parameter to be specified."
            )
        
        if cluster_var not in data_sample.columns:
            raise InvalidParameterError(
                f"Cluster variable '{cluster_var}' not found in data."
            )
        
        n_clusters = data_sample[cluster_var].nunique()
        if n_clusters < 2:
            raise InvalidParameterError(
                f"Cluster variable '{cluster_var}' must have at least 2 unique values. "
                f"Found: {n_clusters}"
            )

        if n_clusters < 10:
            warnings.warn(
                f"Cluster-robust inference with only {n_clusters} clusters may be unreliable. "
                f"Statistical literature typically recommends >= 20-30 clusters for reliable inference. "
                f"Consider using wild_cluster_bootstrap() from lwdid.inference for more reliable inference.",
                UserWarning,
                stacklevel=3
            )
        elif n_clusters < 20:
            # Info-level warning for 10-19 clusters
            warnings.warn(
                f"Cluster-robust inference with {n_clusters} clusters. "
                f"For improved reliability, consider using wild_cluster_bootstrap() from lwdid.inference.",
                UserWarning,
                stacklevel=3
            )
        
        # Check cluster size imbalance (CV > 1.0)
        cluster_sizes = data_sample.groupby(cluster_var).size()
        if cluster_sizes.mean() > 0:
            cluster_size_cv = cluster_sizes.std() / cluster_sizes.mean()
            if cluster_size_cv > 1.0:
                warnings.warn(
                    f"Highly unbalanced cluster sizes (CV={cluster_size_cv:.2f}). "
                    f"Cluster sizes range from {cluster_sizes.min()} to {cluster_sizes.max()} "
                    f"(mean={cluster_sizes.mean():.1f}). "
                    f"This may affect the reliability of cluster-robust inference.",
                    UserWarning,
                    stacklevel=3
                )

        if data_sample[cluster_var].isna().any():
            raise InvalidParameterError(
                f"Cluster variable '{cluster_var}' contains missing values."
            )

        # Verify cluster is nested within units (each unit belongs to one cluster).
        # Cluster-robust SE assumes independence across clusters; if a unit
        # belongs to multiple clusters, this assumption is violated.
        ivar_cluster_map = data.groupby(ivar)[cluster_var].nunique()
        if (ivar_cluster_map > 1).any():
            violating_units = ivar_cluster_map[ivar_cluster_map > 1]
            n_violating = len(violating_units)
            example_unit = violating_units.index[0]
            example_clusters = data[data[ivar] == example_unit][cluster_var].unique()
            raise InvalidParameterError(
                f"Cluster variable '{cluster_var}' is not nested within unit variable '{ivar}'. "
                f"{n_violating} unit(s) belong to multiple clusters across time periods. "
                f"In panel data, each unit must belong to exactly one cluster. "
                f"This is required for cluster-robust SE (assumes cluster independence). "
                f"Violating units: {list(violating_units.index[:5])}{'...' if n_violating > 5 else ''}. "
                f"Example: Unit {example_unit} belongs to {len(example_clusters)} different clusters {list(example_clusters)}."
            )
        
        cluster_groups = data_sample[cluster_var].values
        
        model = sm.OLS(y_vals, X, missing='drop')
        results = model.fit(
            cov_type='cluster',
            cov_kwds={'groups': cluster_groups}
        )
        
        G = len(np.unique(cluster_groups))
    else:
        raise InvalidVCETypeError(
            f"Invalid vce type: '{vce}'. "
            f"Valid options: None, 'hc0', 'hc1', 'hc2', 'hc3', 'hc4', 'robust', 'cluster'"
        )
    
    # Extract parameter estimates as numpy arrays for consistent indexing.
    # statsmodels may return Series or ndarray depending on input format.
    params_vals = results.params.values if hasattr(results.params, 'values') else results.params
    bse_vals = results.bse.values if hasattr(results.bse, 'values') else results.bse
    
    # Require at least 2 parameters: intercept (index 0) and treatment (index 1).
    if len(params_vals) < 2:
        raise InsufficientDataError(
            f"Insufficient variation in treatment indicator (d_) in firstpost cross-section. "
            f"Regression model has only {len(params_vals)} parameter(s) (expected >= 2). "
            f"This typically occurs when all units in the firstpost sample have the same "
            f"treatment status (all d_=0 or all d_=1). "
            f"Please check that your data contains both treated and control units in the "
            f"first post-treatment period."
        )
    
    # ATT is the coefficient on the treatment indicator (index 1).
    # Index 0 is the intercept; indices 2+ are control variables if included.
    att = params_vals[1]
    se_att = bse_vals[1]

    # Warn about near-zero SE which indicates potential numerical issues.
    if se_att < 1e-10:
        warnings.warn(
            f"Standard error is extremely small (SE={se_att:.2e}). "
            f"This may indicate perfect fit, numerical issues, or data problems. "
            f"Results (t-statistic={att/se_att:.2e}, p-value, CI) may be unreliable. "
            f"Consider checking: (1) model fit diagnostics (R², residuals), "
            f"(2) data quality (duplicates, errors), (3) model specification.",
            UserWarning,
            stacklevel=3
        )

    # Degrees of freedom for t-distribution inference.
    # For clustered SE, use G-1 (number of clusters minus 1) for more
    # conservative inference with few clusters. This accounts for the
    # fact that cluster-robust SE estimation effectively estimates G
    # cluster-level quantities, reducing effective degrees of freedom.
    if vce == 'cluster':
        df = G - 1
    else:
        df = results.df_resid

    t_stat = att / se_att

    # Two-sided p-value using the t-distribution CDF.
    # Multiply by 2 for two-sided test under H0: ATT = 0.
    pvalue = 2 * stats.t.cdf(-abs(t_stat), df)

    # Confidence interval using t critical value.
    # Under homoskedasticity and normality, this provides exact coverage.
    # With HC/cluster SE, coverage is asymptotically valid.
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    ci_lower = att - t_crit * se_att
    ci_upper = att + t_crit * se_att
    
    N_treated_sample = int(data_sample[d].sum())
    N_control_sample = int(len(data_sample) - N_treated_sample)

    return {
        'att': att,
        'se_att': se_att,
        't_stat': t_stat,
        'pvalue': pvalue,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'params': results.params,
        'bse': results.bse,
        'vcov': results.cov_params(),
        'resid': results.resid,
        'nobs': int(results.nobs),
        'df_resid': int(results.df_resid),
        'df_inference': int(df),  # actual df used for inference
        'vce_type': vce if vce is not None else 'ols',
        'cluster_var': cluster_var if vce == 'cluster' else None,
        'n_clusters': G if vce == 'cluster' else None,
        'controls_used': controls_used,
        'controls': controls if controls_used else [],
        'controls_spec': controls_spec,
        'n_treated_sample': N_treated_sample,
        'n_control_sample': N_control_sample,
    }


def estimate_period_effects(
    data: pd.DataFrame,
    ydot: str,
    d: str,
    tindex: str,
    tpost1: int,
    Tmax: int,
    controls_spec: dict[str, Any] | None,
    vce: str | None,
    cluster_var: str | None,
    period_labels: dict[int, str],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Estimate period-specific ATTs via independent cross-sectional regressions.

    Iterates through post-treatment periods and estimates the treatment effect
    for each period using the residualized outcome variable. Each period is
    estimated via a separate OLS regression, enabling examination of treatment
    effect dynamics over time.

    Parameters
    ----------
    data : pd.DataFrame
        Transformed panel data containing the outcome variable.
    ydot : str
        Name of the residualized outcome variable.
    d : str
        Name of the treatment indicator column.
    tindex : str
        Name of the time index column.
    tpost1 : int
        First post-treatment period index.
    Tmax : int
        Last period index.
    controls_spec : dict[str, Any] | None
        Control variable specification dictionary returned by
        ``prepare_controls``. If None or if 'include' is False, regression
        includes only the treatment indicator.
    vce : str | None
        Type of variance-covariance estimator. Same options as
        ``estimate_att``: None (homoskedastic), 'robust'/'hc1', 'hc3',
        or 'cluster'.
    cluster_var : str | None
        Name of the clustering variable. Required if ``vce='cluster'``.
    period_labels : dict[int, str]
        Mapping from time index to human-readable period labels for output.
    alpha : float, default=0.05
        Significance level for confidence intervals.

    Returns
    -------
    pd.DataFrame
        A DataFrame with one row per post-treatment period. Columns:

        - 'period' : str
            Human-readable period label.
        - 'tindex' : int
            Time index value.
        - 'beta' : float
            Estimated ATT for the period.
        - 'se' : float
            Standard error.
        - 'ci_lower', 'ci_upper' : float
            Confidence interval bounds.
        - 'tstat' : float
            t-statistic for H0: ATT = 0.
        - 'pval' : float
            Two-sided p-value.
        - 'N' : int
            Number of observations in the period.

    See Also
    --------
    estimate_att : Estimates average ATT across post-treatment periods.
    prepare_controls : Prepares control specification for regression.

    Notes
    -----
    Each period is estimated independently via cross-sectional OLS:

    .. math::

        \\dot{Y}_{it} = \\alpha_t + \\tau_t D_i + X_i \\beta_t +
        D_i (X_i - \\bar{X}_1) \\delta_t + U_{it}

    where :math:`\\dot{Y}_{it}` is the transformed outcome and
    :math:`\\tau_t` is the period-specific ATT.

    The estimators are consistent under no anticipation and parallel trends.
    Period-specific estimates allow examination of treatment effect dynamics,
    which is useful for detecting delayed effects or effect decay over time.

    Periods with insufficient variation in the treatment indicator (all
    treated or all control) produce NaN estimates with a warning. This can
    occur due to panel attrition or unbalanced designs.
    """
    results_list = []

    for t in range(tpost1, Tmax + 1):
        mask_t = (data[tindex] == t)
        data_t = data[mask_t]

        if controls_spec is not None and controls_spec['include']:
            X_centered_t = controls_spec['X_centered'].loc[data_t.index]
            interactions_t = controls_spec['interactions'].loc[data_t.index]
            controls_list = list(controls_spec['X_mean_treated'].keys())
            
            X_parts = [
                data_t[[d]],
                data_t[controls_list],
                interactions_t
            ]
            X_df = pd.concat(X_parts, axis=1)
            X_t = sm.add_constant(X_df.astype(float).values)
        else:
            d_vals = data_t[d].astype(float).values
            X_t = sm.add_constant(d_vals)
        
        # Normalize vce parameter (case-insensitive)
        vce_lower = vce.lower() if vce else None
        
        try:
            if vce_lower is None:
                model_t = sm.OLS(data_t[ydot], X_t, missing='drop').fit()
            elif vce_lower == 'hc0':
                model_t = sm.OLS(data_t[ydot], X_t, missing='drop').fit(cov_type='HC0')
            elif vce_lower in ('hc1', 'robust'):
                model_t = sm.OLS(data_t[ydot], X_t, missing='drop').fit(cov_type='HC1')
            elif vce_lower == 'hc2':
                model_t = sm.OLS(data_t[ydot], X_t, missing='drop').fit(cov_type='HC2')
            elif vce_lower == 'hc3':
                model_t = sm.OLS(data_t[ydot], X_t, missing='drop').fit(cov_type='HC3')
            elif vce_lower == 'hc4':
                # HC4: adaptive leverage correction for high-leverage observations.
                # statsmodels does not support HC4; compute manually.
                model_ols_t = sm.OLS(data_t[ydot], X_t, missing='drop').fit()
                
                n_obs_t = int(model_ols_t.nobs)
                k_t = X_t.shape[1]
                residuals_t = model_ols_t.resid
                
                # Compute leverage values
                XtX_inv_t = np.linalg.inv(X_t.T @ X_t)
                tmp_t = X_t @ XtX_inv_t
                h_ii_t = (tmp_t * X_t).sum(axis=1)
                h_ii_t = np.clip(h_ii_t, 0, 0.9999)
                
                # HC4 adjustment
                delta_t = np.minimum(4.0, n_obs_t * h_ii_t / k_t)
                delta_t = np.maximum(delta_t, 0.0)
                omega_diag_t = (residuals_t ** 2) / ((1 - h_ii_t) ** delta_t)
                
                # Sandwich variance
                meat_t = X_t.T @ np.diag(omega_diag_t) @ X_t
                var_beta_t = XtX_inv_t @ meat_t @ XtX_inv_t
                
                # Construct a results wrapper with HC4 standard errors.
                class HC4ResultsT:
                    def __init__(self, ols_results, var_beta):
                        self.params = ols_results.params
                        # Handle both Series and ndarray for params
                        if hasattr(ols_results.params, 'index'):
                            self.bse = pd.Series(np.sqrt(np.diag(var_beta)), index=ols_results.params.index)
                        else:
                            self.bse = np.sqrt(np.diag(var_beta))
                        self.df_resid = ols_results.df_resid
                        self.nobs = ols_results.nobs
                
                model_t = HC4ResultsT(model_ols_t, var_beta_t)
            elif vce_lower == 'cluster':
                if cluster_var is None:
                    raise InvalidParameterError(
                        "vce='cluster' requires cluster_var parameter to be specified."
                    )

                if cluster_var not in data_t.columns:
                    raise InvalidParameterError(
                        f"Cluster variable '{cluster_var}' not found in data for period t={t}."
                    )

                if data_t[cluster_var].isna().any():
                    raise InvalidParameterError(
                        f"Cluster variable '{cluster_var}' contains missing values in period t={t}."
                    )

                n_clusters_t = data_t[cluster_var].nunique()
                if n_clusters_t < 2:
                    raise InvalidParameterError(
                        f"Cluster variable '{cluster_var}' must have at least 2 unique values in period t={t}. "
                        f"Found: {n_clusters_t}"
                    )

                cluster_groups_t = data_t[cluster_var].values

                model_t = sm.OLS(data_t[ydot], X_t, missing='drop').fit(
                    cov_type='cluster',
                    cov_kwds={'groups': cluster_groups_t}
                )
            else:
                raise InvalidVCETypeError(
                    f"Invalid vce type: '{vce}'. "
                    f"Valid options: None, 'hc0', 'hc1', 'hc2', 'hc3', 'hc4', 'robust', 'cluster'"
                )
            
            params_vals = model_t.params.values if hasattr(model_t.params, 'values') else model_t.params
            bse_vals = model_t.bse.values if hasattr(model_t.bse, 'values') else model_t.bse

            beta_t = params_vals[1]
            se_t = bse_vals[1]

            if se_t == 0 or np.isnan(se_t) or np.isinf(se_t):
                period_label = period_labels.get(t, str(t))
                warnings.warn(
                    f"Period {period_label} (t={t}) regression produced degenerate results "
                    f"(se={se_t}). This likely indicates perfect separation (all d=0 or all d=1) "
                    f"or insufficient variation in treatment variable. Setting results to NaN.",
                    category=UserWarning,
                    stacklevel=2
                )
                beta_t = se_t = t_stat = p_val = np.nan
                ci_lower = ci_upper = np.nan
                N_t = int(model_t.nobs)
            else:
                # Select degrees of freedom based on variance estimator.
                # Cluster-robust uses G-1 for conservative inference.
                if vce_lower == 'cluster':
                    n_clusters_t = len(np.unique(cluster_groups_t))
                    df_t = n_clusters_t - 1
                else:
                    df_t = model_t.df_resid

                t_stat = beta_t / se_t
                # Two-sided p-value for H0: tau_t = 0.
                p_val = 2 * stats.t.cdf(-abs(t_stat), df_t)
                N_t = int(model_t.nobs)

                # Confidence interval with t critical value.
                t_crit_t = stats.t.ppf(1 - alpha / 2, df_t)
                ci_lower = beta_t - t_crit_t * se_t
                ci_upper = beta_t + t_crit_t * se_t
        except Exception as e:
            period_label = period_labels.get(t, str(t))
            warnings.warn(
                f"Period {period_label} (t={t}) regression failed: {str(e)}. "
                f"Setting results to NaN. This may indicate insufficient sample size, "
                f"perfect separation (all d=0 or all d=1), or numerical issues. "
                f"Check data quality for this period.",
                category=UserWarning,
                stacklevel=2
            )

            beta_t = se_t = t_stat = p_val = np.nan
            ci_lower = ci_upper = np.nan

            valid_mask = data_t[ydot].notna() & data_t[d].notna()
            if controls_spec is not None and 'X_centered' in controls_spec:
                for col in controls_spec['X_centered'].columns:
                    if col in data_t.columns:
                        valid_mask &= data_t[col].notna()
            N_t = int(valid_mask.sum())
        
        results_list.append({
            'period': period_labels.get(t, str(t)),
            'tindex': t,
            'beta': beta_t,
            'se': se_t,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'tstat': t_stat,
            'pval': p_val,
            'N': N_t
        })
    
    return pd.DataFrame(results_list)

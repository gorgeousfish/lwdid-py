"""
Treatment effect estimators for difference-in-differences settings.

This module provides inverse probability weighting (IPW), inverse probability
weighted regression adjustment (IPWRA), and propensity score matching (PSM)
estimators for ATT estimation. These estimators handle covariate imbalance
between treated and control groups through propensity score methods.

The IPW estimator reweights control observations to match the covariate
distribution of treated units, providing consistent ATT estimates when the
propensity score model is correctly specified.

The IPWRA estimator combines propensity score weighting with outcome regression
to achieve double robustness: consistency requires correct specification of
either the propensity score model or the outcome regression model, but not
necessarily both. This property makes IPWRA particularly valuable when there
is uncertainty about functional form assumptions.

The PSM estimator uses nearest-neighbor matching on estimated propensity scores,
with options for caliper constraints, replacement, and multiple matches per
treated unit. Standard errors use either a heteroskedasticity-robust formula
or bootstrap resampling.

Notes
-----
All estimators take transformed outcome variables (after unit-specific
demeaning or detrending) as inputs and return ATT estimates for specific
cohort-time pairs in staggered adoption designs. The propensity score model
uses logistic regression without regularization for unbiased coefficient
estimates.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats


# ============================================================================
# IPW (Inverse Probability Weighting) Estimator
# ============================================================================

@dataclass
class IPWResult:
    """
    Result container for inverse probability weighting ATT estimation.

    Attributes
    ----------
    att : float
        ATT point estimate.
    se : float
        Standard error.
    ci_lower : float
        Lower bound of confidence interval.
    ci_upper : float
        Upper bound of confidence interval.
    t_stat : float
        t-statistic.
    pvalue : float
        Two-sided p-value.
    propensity_scores : np.ndarray
        Estimated propensity scores.
    weights : np.ndarray
        IPW weights w = p/(1-p) for control units.
    propensity_model_coef : dict[str, float]
        Propensity score model coefficients.
    n_treated : int
        Number of treated units.
    n_control : int
        Number of control units.
    df_resid : int
        Residual degrees of freedom.
    df_inference : int
        Degrees of freedom for inference.
    weights_cv : float
        Coefficient of variation of IPW weights.
    diagnostics : dict[str, Any] or None
        Additional diagnostic information.
    """
    att: float
    se: float
    ci_lower: float
    ci_upper: float
    t_stat: float
    pvalue: float
    propensity_scores: np.ndarray
    weights: np.ndarray
    propensity_model_coef: dict[str, float]
    n_treated: int
    n_control: int
    df_resid: int = 0
    df_inference: int = 0
    weights_cv: float = 0.0
    diagnostics: dict[str, Any] | None = None


# ============================================================================
# IPWRA (Doubly Robust) Estimator
# ============================================================================

@dataclass
class IPWRAResult:
    """
    Result container for doubly robust IPWRA ATT estimation.

    Attributes
    ----------
    att : float
        ATT point estimate.
    se : float
        Standard error.
    ci_lower : float
        Lower bound of confidence interval.
    ci_upper : float
        Upper bound of confidence interval.
    t_stat : float
        t-statistic.
    pvalue : float
        Two-sided p-value.
    propensity_scores : np.ndarray
        Estimated propensity scores.
    weights : np.ndarray
        IPW weights w = p/(1-p) for control units.
    outcome_model_coef : dict[str, float]
        Outcome model coefficients.
    propensity_model_coef : dict[str, float]
        Propensity score model coefficients.
    n_treated : int
        Number of treated units.
    n_control : int
        Number of control units.
    """
    att: float
    se: float
    ci_lower: float
    ci_upper: float
    t_stat: float
    pvalue: float
    propensity_scores: np.ndarray
    weights: np.ndarray
    outcome_model_coef: dict[str, float]
    propensity_model_coef: dict[str, float]
    n_treated: int
    n_control: int


def estimate_ipw(
    data: pd.DataFrame,
    y: str,
    d: str,
    propensity_controls: list[str],
    trim_threshold: float = 0.01,
    alpha: float = 0.05,
    vce: str | None = None,
    return_diagnostics: bool = False,
    gvar_col: str | None = None,
    ivar_col: str | None = None,
    cohort_g: int | None = None,
    period_r: int | None = None,
) -> IPWResult:
    """
    Estimate ATT using inverse probability weighting.

    IPW estimates the average treatment effect on the treated by reweighting
    control observations to match the covariate distribution of treated units.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset containing outcome, treatment, and control variables.
    y : str
        Outcome variable column name.
    d : str
        Treatment indicator column name (1=treated, 0=control).
    propensity_controls : list[str]
        Variables for propensity score model.
    trim_threshold : float, default=0.01
        Propensity score trimming threshold. Observations with propensity
        scores outside [trim_threshold, 1-trim_threshold] are excluded.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    vce : str, optional
        Variance estimator type ('robust', 'hc0', 'hc1', 'hc2', 'hc3').
    return_diagnostics : bool, default=False
        Whether to return additional diagnostic information.
    gvar_col : str, optional
        Column name for cohort indicator (for staggered designs).
    ivar_col : str, optional
        Column name for unit identifier (for staggered designs).
    cohort_g : int, optional
        Cohort value (for staggered designs).
    period_r : int, optional
        Period value (for staggered designs).

    Returns
    -------
    IPWResult
        Estimation results including ATT, standard error, and diagnostics.

    Raises
    ------
    ValueError
        If no propensity controls are specified, no treated or control units
        exist, or all observations are trimmed.

    Notes
    -----
    The IPW estimator for ATT is:

    .. math::

        \\hat{\\tau}_{IPW} = \\frac{1}{N_1} \\sum_{i:D_i=1} Y_i
                           - \\frac{1}{N_1} \\sum_{i:D_i=0} \\frac{\\hat{p}(X_i)}{1-\\hat{p}(X_i)} Y_i

    where :math:`\\hat{p}(X_i)` is the estimated propensity score.

    Inference uses the normal distribution based on influence function
    asymptotics, consistent with IPWRA and PSM estimators in this package.
    """
    # Validate inputs.
    if not propensity_controls:
        raise ValueError("propensity_controls must be specified for IPW estimation.")

    # Extract data.
    y_values = data[y].values
    d_values = data[d].values.astype(int)
    X_ps = data[propensity_controls].values

    treated_mask = d_values == 1
    control_mask = d_values == 0
    n_treated = treated_mask.sum()
    n_control = control_mask.sum()

    if n_treated == 0:
        raise ValueError("No treated units in the data.")
    if n_control == 0:
        raise ValueError("No control units in the data.")

    # Estimate propensity scores.
    ps, ps_coef = estimate_propensity_score(
        data=data,
        d=d,
        controls=propensity_controls,
        trim_threshold=trim_threshold,
    )
    # Compute trimmed mask based on threshold.
    trimmed_mask = (ps < trim_threshold) | (ps > 1 - trim_threshold)

    # Apply trimming.
    valid_mask = ~trimmed_mask
    if valid_mask.sum() == 0:
        raise ValueError("All observations were trimmed. Adjust trim_threshold.")

    y_valid = y_values[valid_mask]
    d_valid = d_values[valid_mask]
    ps_valid = ps[valid_mask]

    treated_valid = d_valid == 1
    control_valid = d_valid == 0
    n_treated_valid = treated_valid.sum()
    n_control_valid = control_valid.sum()

    if n_treated_valid == 0:
        raise ValueError("No treated units remain after trimming.")
    if n_control_valid == 0:
        raise ValueError("No control units remain after trimming.")

    # IPW weights w = p/(1-p) reweight controls to match treated covariate distribution.
    weights = np.zeros(len(y_valid))
    weights[control_valid] = ps_valid[control_valid] / (1 - ps_valid[control_valid])

    # Normalize weights so they sum to n_treated for proper ATT estimation.
    weight_sum = weights[control_valid].sum()
    if weight_sum > 0:
        weights[control_valid] = weights[control_valid] * n_treated_valid / weight_sum

    # Compute ATT.
    y_treated_mean = y_valid[treated_valid].mean()
    y_control_weighted = np.sum(weights[control_valid] * y_valid[control_valid]) / n_treated_valid
    att = y_treated_mean - y_control_weighted

    # Influence function approach provides asymptotically valid variance estimation.
    psi = np.zeros(len(y_valid))

    # For treated: psi_i = (Y_i - tau) / N_1.
    psi[treated_valid] = (y_valid[treated_valid] - att) / n_treated_valid

    # For control: psi_i = -w_i * Y_i / N_1, where mu_0 is the weighted control mean.
    psi[control_valid] = -weights[control_valid] * y_valid[control_valid] / n_treated_valid

    # Variance is the sum of squared influence functions.
    var_att = np.sum(psi**2)
    se = np.sqrt(var_att)

    # Degrees of freedom retained for interface compatibility; inference uses normal distribution.
    df_resid = n_treated_valid + n_control_valid - 2
    df_resid = max(1, df_resid)
    df_inference = df_resid

    # z-statistic and p-value using normal distribution (asymptotic inference).
    if se > 0:
        t_stat = att / se
        pvalue = 2 * (1 - stats.norm.cdf(abs(t_stat)))
    else:
        t_stat = np.nan
        pvalue = np.nan

    # Confidence interval using normal distribution (asymptotic inference).
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lower = att - z_crit * se
    ci_upper = att + z_crit * se

    # Coefficient of variation diagnoses overlap violations; high CV indicates extreme weights.
    weights_control = weights[control_valid]
    if len(weights_control) > 1:
        weights_mean = np.mean(weights_control)
        weights_std = np.std(weights_control, ddof=1)
        weights_cv = weights_std / weights_mean if weights_mean > 0 else np.nan
    else:
        weights_cv = 0.0

    return IPWResult(
        att=att,
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        t_stat=t_stat,
        pvalue=pvalue,
        propensity_scores=ps,
        weights=weights,
        propensity_model_coef=ps_coef,
        n_treated=n_treated_valid,
        n_control=n_control_valid,
        df_resid=df_resid,
        df_inference=df_inference,
        weights_cv=weights_cv,
    )


def estimate_ipwra(
    data: pd.DataFrame,
    y: str,
    d: str,
    controls: list[str],
    propensity_controls: list[str] | None = None,
    trim_threshold: float = 0.01,
    se_method: str = 'analytical',
    n_bootstrap: int = 200,
    seed: int | None = None,
    alpha: float = 0.05,
    return_diagnostics: bool = False,
    gvar_col: str | None = None,
    ivar_col: str | None = None,
    cohort_g: int | None = None,
    period_r: int | None = None,
) -> IPWRAResult:
    """
    Estimate ATT using inverse probability weighted regression adjustment.

    IPWRA is a doubly robust estimator that combines propensity score weighting
    with outcome regression. The estimator remains consistent if either the
    propensity score model or the outcome regression model is correctly
    specified, providing protection against model misspecification.

    The IPWRA-ATT estimator is computed as:

    .. math::

        \\hat{\\tau} = \\frac{1}{N_1} \\sum_{D_i=1} [Y_i - \\hat{m}_0(X_i)]
                      - \\frac{\\sum_{D_i=0} w_i (Y_i - \\hat{m}_0(X_i))}
                             {\\sum_{D_i=0} w_i}

    where :math:`\\hat{m}_0(X)` is the estimated outcome regression for controls,
    and :math:`w_i = \\hat{p}(X_i) / (1 - \\hat{p}(X_i))` are IPW weights.

    Parameters
    ----------
    data : pd.DataFrame
        Cross-sectional data with one row per unit.
    y : str
        Outcome variable column name (typically transformed outcome).
    d : str
        Treatment indicator column name (1=treated, 0=control).
    controls : list[str]
        Control variables for the outcome regression model.
    propensity_controls : list[str], optional
        Control variables for the propensity score model.
        If None, uses the same variables as ``controls``.
    trim_threshold : float, default=0.01
        Propensity score trimming threshold. Scores are clipped to
        [trim_threshold, 1-trim_threshold] to prevent extreme weights.
    se_method : str, default='analytical'
        Standard error computation method: 'analytical' uses influence
        functions, 'bootstrap' uses nonparametric bootstrap.
    n_bootstrap : int, default=200
        Number of bootstrap replications when se_method='bootstrap'.
    seed : int, optional
        Random seed for bootstrap resampling.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    return_diagnostics : bool, default=False
        Whether to return additional diagnostic information.
    gvar_col : str, optional
        Cohort indicator column name (for staggered designs).
    ivar_col : str, optional
        Unit identifier column name (for staggered designs).
    cohort_g : int, optional
        Cohort value (for staggered designs).
    period_r : int, optional
        Period value (for staggered designs).

    Returns
    -------
    IPWRAResult
        Estimation results containing ATT estimate, standard error,
        confidence interval, propensity scores, and model coefficients.

    Raises
    ------
    ValueError
        If required columns are missing, sample sizes are insufficient,
        or model estimation fails to converge.

    See Also
    --------
    estimate_ipw : Pure IPW estimator without outcome regression.
    estimate_psm : Propensity score matching estimator.
    """
    # Validate inputs.
    if y not in data.columns:
        raise ValueError(f"Outcome variable '{y}' not found in data.")
    if d not in data.columns:
        raise ValueError(f"Treatment indicator '{d}' not found in data.")
    
    missing_controls = [c for c in controls if c not in data.columns]
    if missing_controls:
        raise ValueError(f"Control variables not found: {missing_controls}")
    
    if propensity_controls is None:
        propensity_controls = controls
    else:
        missing_ps = [c for c in propensity_controls if c not in data.columns]
        if missing_ps:
            raise ValueError(f"Propensity score controls not found: {missing_ps}")
    
    # Remove observations with missing values.
    all_vars = [y, d] + list(set(controls + propensity_controls))
    data_clean = data[all_vars].dropna().copy()
    
    n = len(data_clean)
    D = data_clean[d].values.astype(float)
    Y = data_clean[y].values.astype(float)
    
    n_treated = int(D.sum())
    n_control = int((1 - D).sum())
    
    if n_treated < 2:
        raise ValueError(f"Insufficient treated units: n_treated={n_treated}, requires at least 2.")
    if n_control < 2:
        raise ValueError(f"Insufficient control units: n_control={n_control}, requires at least 2.")
    
    if n_treated < 5:
        warnings.warn(
            f"Small treated sample (n_treated={n_treated}); IPWRA estimates may be unstable.",
            UserWarning
        )
    if n_control < 10:
        warnings.warn(
            f"Small control sample (n_control={n_control}); outcome model may be unstable.",
            UserWarning
        )
    
    # Estimate propensity scores.
    pscores, ps_coef = estimate_propensity_score(
        data_clean, d, propensity_controls, trim_threshold
    )
    
    # Compute ATT weights for WLS outcome model estimation.
    # IPWRA uses WLS with w_i = 1 for treated units and w_i = p(X)/(1-p(X)) for
    # controls. This reweights the control sample to match the treated covariate
    # distribution, ensuring the outcome model targets the ATT estimand.
    att_weights = np.where(D == 1, 1.0, pscores / (1 - pscores))
    
    # Estimate outcome model on control units using WLS with ATT weights.
    m0_hat, outcome_coef = estimate_outcome_model(
        data_clean, y, d, controls, sample_weights=att_weights
    )
    
    # Compute IPWRA-ATT estimator.
    treat_mask = D == 1
    control_mask = D == 0
    
    # Treated component: (1/N_1) sum_{D=1} [Y - m_0(X)].
    treat_term = (Y[treat_mask] - m0_hat[treat_mask]).mean()
    
    # Weighted control component.
    weights = pscores / (1 - pscores)
    weights_control = weights[control_mask]
    residuals_control = Y[control_mask] - m0_hat[control_mask]
    
    # Check for extreme weights indicating overlap violation.
    weights_cv = np.std(weights_control) / np.mean(weights_control) if np.mean(weights_control) > 0 else np.inf
    if weights_cv > 2.0:
        warnings.warn(
            f"Extreme IPW weights detected (CV={weights_cv:.2f} > 2.0), indicating potential "
            f"overlap violation. Consider checking propensity score distribution or increasing "
            f"trim_threshold.",
            UserWarning
        )
    
    # Check proportion of extreme propensity scores.
    extreme_low = (pscores < 0.05).mean()
    extreme_high = (pscores > 0.95).mean()
    if extreme_low > 0.1 or extreme_high > 0.1:
        warnings.warn(
            f"High proportion of extreme propensity scores (p<0.05: {extreme_low:.1%}, "
            f"p>0.95: {extreme_high:.1%}), suggesting overlap assumption may be violated. "
            f"Consider increasing trim_threshold or reviewing covariate selection.",
            UserWarning
        )
    
    weights_sum = weights_control.sum()
    if weights_sum <= 0:
        raise ValueError("Sum of IPW weights is non-positive; propensity score model may be misspecified.")
    
    control_term = (weights_control * residuals_control).sum() / weights_sum
    
    att = treat_term - control_term
    
    # Compute standard errors.
    if se_method == 'analytical':
        se, ci_lower, ci_upper = compute_ipwra_se_analytical(
            data_clean, y, d, controls,
            att, pscores, m0_hat, weights, alpha
        )
    elif se_method == 'bootstrap':
        se, ci_lower, ci_upper = compute_ipwra_se_bootstrap(
            data_clean, y, d, controls, propensity_controls,
            trim_threshold, n_bootstrap, seed, alpha
        )
    else:
        raise ValueError(f"Unknown se_method: {se_method}. Use 'analytical' or 'bootstrap'.")
    
    # Compute t-statistic and p-value.
    if se > 0:
        t_stat = att / se
        pvalue = 2 * (1 - stats.norm.cdf(abs(t_stat)))
    else:
        t_stat = np.nan
        pvalue = np.nan
    
    return IPWRAResult(
        att=att,
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        t_stat=t_stat,
        pvalue=pvalue,
        propensity_scores=pscores,
        weights=weights,
        outcome_model_coef=outcome_coef,
        propensity_model_coef=ps_coef,
        n_treated=n_treated,
        n_control=n_control,
    )


def estimate_propensity_score(
    data: pd.DataFrame,
    d: str,
    controls: list[str],
    trim_threshold: float = 0.01,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Estimate propensity scores using logistic regression.

    Fits a logit model P(D=1|X) without regularization, using maximum
    likelihood estimation for unbiased coefficient estimates.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset containing treatment and control variables.
    d : str
        Treatment indicator column name.
    controls : list[str]
        Covariate column names for the propensity score model.
    trim_threshold : float, default=0.01
        Trimming threshold for extreme propensity scores.

    Returns
    -------
    propensity_scores : np.ndarray
        Estimated propensity scores, trimmed to [trim_threshold, 1-trim_threshold].
    coefficients : dict[str, float]
        Dictionary mapping variable names to estimated coefficients,
        including '_intercept' for the constant term.

    Raises
    ------
    ValueError
        If the logistic regression fails to converge.
    """
    from sklearn.linear_model import LogisticRegression
    
    X = data[controls].values.astype(float)
    D = data[d].values.astype(float)
    
    # Standardize covariates for numerical stability.
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1  # Avoid division by zero for constant columns.
    X_scaled = (X - X_mean) / X_std
    
    # No regularization ensures unbiased coefficient estimates.
    try:
        model = LogisticRegression(
            penalty=None,
            solver='lbfgs',
            max_iter=1000,
            tol=1e-8,
        )
        model.fit(X_scaled, D)
    except Exception as e:
        raise ValueError(f"Propensity score model estimation failed: {e}")
    
    # Predict propensity scores.
    pscores = model.predict_proba(X_scaled)[:, 1]
    
    # Trimming prevents extreme IPW weights that inflate variance.
    pscores = np.clip(pscores, trim_threshold, 1 - trim_threshold)
    
    # Transform coefficients back to original scale.
    coef_scaled = model.coef_[0]
    intercept = model.intercept_[0]
    
    coef_original = coef_scaled / X_std
    intercept_original = intercept - (coef_scaled * X_mean / X_std).sum()
    
    coef_dict = {'_intercept': intercept_original}
    for i, name in enumerate(controls):
        coef_dict[name] = coef_original[i]
    
    return pscores, coef_dict


def estimate_outcome_model(
    data: pd.DataFrame,
    y: str,
    d: str,
    controls: list[str],
    sample_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Estimate the outcome regression model on control units.

    Fits a linear regression E(Y|X, D=0) using control observations only,
    then generates predicted values for all units. Optionally uses weighted
    least squares (WLS) when sample_weights are provided.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset containing outcome, treatment, and control variables.
    y : str
        Outcome variable column name.
    d : str
        Treatment indicator column name.
    controls : list[str]
        Covariate column names for the outcome model.
    sample_weights : np.ndarray, optional
        Weights for weighted least squares estimation. If provided, must have
        the same length as the data. Only weights for control units (D=0) are
        used in fitting. For IPWRA with ATT targeting, weights should be
        w_i = p(X_i) / (1 - p(X_i)) to reweight controls to the treated
        covariate distribution.

    Returns
    -------
    predictions : np.ndarray
        Predicted outcome values for all units based on control regression.
    coefficients : dict[str, float]
        Dictionary mapping variable names to estimated coefficients,
        including '_intercept' for the constant term.

    Raises
    ------
    ValueError
        If the design matrix is singular and cannot be inverted.

    Notes
    -----
    When sample_weights are provided, the outcome model is estimated using WLS:
    
    .. math::
    
        \\hat{\\beta} = (X'WX)^{-1} X'WY
    
    where W is the diagonal weight matrix. For IPWRA with ATT targeting, the
    weights for control units should be w_i = p(X_i) / (1 - p(X_i)), which
    reweights the control sample to match the treated covariate distribution.
    """
    D = data[d].values.astype(float)
    Y = data[y].values.astype(float)
    X = data[controls].values.astype(float)
    
    # Extract control group data.
    control_mask = D == 0
    X_control = X[control_mask]
    Y_control = Y[control_mask]
    
    # Add intercept term.
    X_control_const = np.column_stack([np.ones(len(X_control)), X_control])
    
    if sample_weights is not None:
        # Weighted Least Squares (WLS) estimation.
        # Extract weights for control units only.
        w_control = sample_weights[control_mask]
        
        # Ensure weights are positive.
        w_control = np.maximum(w_control, 1e-10)
        
        # WLS via sqrt(W) transformation: beta = (X̃'X̃)^{-1} X̃'Ỹ.
        sqrt_w = np.sqrt(w_control)
        X_weighted = X_control_const * sqrt_w[:, np.newaxis]
        Y_weighted = Y_control * sqrt_w
        
        try:
            XtWX_inv = np.linalg.inv(X_weighted.T @ X_weighted)
            beta = XtWX_inv @ (X_weighted.T @ Y_weighted)
        except np.linalg.LinAlgError:
            raise ValueError("Weighted outcome model design matrix is singular; cannot estimate coefficients.")
    else:
        # Standard OLS estimation.
        try:
            XtX_inv = np.linalg.inv(X_control_const.T @ X_control_const)
            beta = XtX_inv @ (X_control_const.T @ Y_control)
        except np.linalg.LinAlgError:
            raise ValueError("Outcome model design matrix is singular; cannot estimate coefficients.")
    
    # Generate predictions for all units.
    X_all_const = np.column_stack([np.ones(len(X)), X])
    m0_hat = X_all_const @ beta
    
    # Store coefficients.
    coef_dict = {'_intercept': beta[0]}
    for i, name in enumerate(controls):
        coef_dict[name] = beta[i + 1]
    
    return m0_hat, coef_dict


def compute_ipwra_se_analytical(
    data: pd.DataFrame,
    y: str,
    d: str,
    controls: list[str],
    att: float,
    pscores: np.ndarray,
    m0_hat: np.ndarray,
    weights: np.ndarray,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """
    Compute IPWRA standard error using the influence function approach.

    Uses a simplified influence function that accounts for the primary
    estimation uncertainty. The full doubly robust influence function
    would additionally adjust for uncertainty in the propensity score
    and outcome model estimation.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset used for estimation.
    y : str
        Outcome variable column name.
    d : str
        Treatment indicator column name.
    controls : list[str]
        Control variable column names.
    att : float
        Estimated ATT value.
    pscores : np.ndarray
        Estimated propensity scores.
    m0_hat : np.ndarray
        Predicted outcomes from control regression.
    weights : np.ndarray
        IPW weights for all observations.
    alpha : float, default=0.05
        Significance level for confidence interval.

    Returns
    -------
    se : float
        Standard error estimate.
    ci_lower : float
        Lower bound of confidence interval.
    ci_upper : float
        Upper bound of confidence interval.
    """
    D = data[d].values.astype(float)
    Y = data[y].values.astype(float)
    n = len(D)
    
    n_treated = D.sum()
    treat_mask = D == 1
    control_mask = D == 0
    
    # Simplified influence function.
    p_bar = n_treated / n
    weights_sum = weights[control_mask].sum()
    
    # Treated unit contribution.
    psi_treat = (Y[treat_mask] - m0_hat[treat_mask] - att) / p_bar
    
    # Control unit contribution.
    residuals_control = Y[control_mask] - m0_hat[control_mask]
    psi_control = -weights[control_mask] * residuals_control / weights_sum
    
    # Combine influence functions.
    psi = np.zeros(n)
    psi[treat_mask] = psi_treat
    psi[control_mask] = psi_control
    
    # Variance estimation.
    var_psi = np.var(psi, ddof=1)
    var_att = var_psi / n
    se = np.sqrt(var_att)
    
    # Confidence interval.
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lower = att - z_crit * se
    ci_upper = att + z_crit * se
    
    return se, ci_lower, ci_upper


def compute_ipwra_se_bootstrap(
    data: pd.DataFrame,
    y: str,
    d: str,
    controls: list[str],
    propensity_controls: list[str] | None,
    trim_threshold: float,
    n_bootstrap: int,
    seed: int | None,
    alpha: float,
) -> tuple[float, float, float]:
    """
    Compute IPWRA standard error using nonparametric bootstrap.

    Resamples the entire estimation procedure including propensity score
    estimation, outcome model fitting, and ATT computation. This approach
    properly accounts for all sources of estimation uncertainty.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset used for estimation.
    y : str
        Outcome variable column name.
    d : str
        Treatment indicator column name.
    controls : list[str]
        Control variable column names for outcome model.
    propensity_controls : list[str] or None
        Control variable column names for propensity score model.
    trim_threshold : float
        Propensity score trimming threshold.
    n_bootstrap : int
        Number of bootstrap replications.
    seed : int or None
        Random seed for reproducibility.
    alpha : float
        Significance level for confidence interval.

    Returns
    -------
    se : float
        Bootstrap standard error.
    ci_lower : float
        Percentile confidence interval lower bound.
    ci_upper : float
        Percentile confidence interval upper bound.
    """
    if seed is not None:
        np.random.seed(seed)
    
    if propensity_controls is None:
        propensity_controls = controls
    
    n = len(data)
    att_boots = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement.
        indices = np.random.choice(n, size=n, replace=True)
        data_boot = data.iloc[indices].reset_index(drop=True)
        
        try:
            pscores_boot, _ = estimate_propensity_score(
                data_boot, d, propensity_controls, trim_threshold
            )
            
            # Compute ATT weights for WLS (same as main estimation).
            D_boot = data_boot[d].values.astype(float)
            att_weights_boot = np.where(D_boot == 1, 1.0, pscores_boot / (1 - pscores_boot))
            
            m0_boot, _ = estimate_outcome_model(
                data_boot, y, d, controls, sample_weights=att_weights_boot
            )
            
            Y_boot = data_boot[y].values.astype(float)
            
            treat_mask = D_boot == 1
            control_mask = D_boot == 0
            
            if treat_mask.sum() < 1 or control_mask.sum() < 1:
                continue
            
            treat_term = (Y_boot[treat_mask] - m0_boot[treat_mask]).mean()
            
            weights_boot = pscores_boot / (1 - pscores_boot)
            weights_control = weights_boot[control_mask]
            residuals_control = Y_boot[control_mask] - m0_boot[control_mask]
            
            weights_sum = weights_control.sum()
            if weights_sum > 0:
                control_term = (weights_control * residuals_control).sum() / weights_sum
                att_boot = treat_term - control_term
                att_boots.append(att_boot)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError, ZeroDivisionError):
            # Skip failed bootstrap iterations due to numerical issues.
            continue
    
    if len(att_boots) < n_bootstrap * 0.5:
        warnings.warn(
            f"Low bootstrap success rate: {len(att_boots)}/{n_bootstrap}",
            UserWarning
        )
    
    if len(att_boots) < 10:
        raise ValueError(f"Insufficient bootstrap samples: {len(att_boots)}")
    
    att_boots = np.array(att_boots)
    se = np.std(att_boots, ddof=1)
    ci_lower = np.percentile(att_boots, 100 * alpha / 2)
    ci_upper = np.percentile(att_boots, 100 * (1 - alpha / 2))
    
    return se, ci_lower, ci_upper


# ============================================================================
# PSM (Propensity Score Matching) Estimator
# ============================================================================

@dataclass
class PSMResult:
    """
    Result container for propensity score matching ATT estimation.

    Attributes
    ----------
    att : float
        ATT point estimate from nearest neighbor matching.
    se : float
        Standard error (heteroskedasticity-robust or bootstrap).
    ci_lower : float
        Lower bound of confidence interval.
    ci_upper : float
        Upper bound of confidence interval.
    t_stat : float
        t-statistic for testing ATT=0.
    pvalue : float
        Two-sided p-value.
    propensity_scores : np.ndarray
        Estimated propensity scores for all observations.
    match_counts : np.ndarray
        Number of matches for each treated unit.
    matched_control_ids : list[list[int]]
        List of matched control unit indices for each treated unit.
    n_treated : int
        Number of treated units.
    n_control : int
        Number of control units.
    n_matched : int
        Number of unique control units used in matching.
    caliper : float or None
        Caliper value used for matching, if any.
    n_dropped : int
        Number of treated units dropped due to caliper constraint.
    """
    att: float
    se: float
    ci_lower: float
    ci_upper: float
    t_stat: float
    pvalue: float
    propensity_scores: np.ndarray
    match_counts: np.ndarray
    matched_control_ids: list[list[int]]
    n_treated: int
    n_control: int
    n_matched: int
    caliper: float | None
    n_dropped: int


def estimate_psm(
    data: pd.DataFrame,
    y: str,
    d: str,
    propensity_controls: list[str],
    n_neighbors: int = 1,
    with_replacement: bool = True,
    caliper: float | None = None,
    caliper_scale: str = 'sd',
    trim_threshold: float = 0.01,
    se_method: str = 'abadie_imbens',
    n_bootstrap: int = 200,
    seed: int | None = None,
    alpha: float = 0.05,
    match_order: str = 'data',
    return_diagnostics: bool = False,
    gvar_col: str | None = None,
    ivar_col: str | None = None,
    cohort_g: int | None = None,
    period_r: int | None = None,
) -> PSMResult:
    """
    Estimate ATT using propensity score matching.

    Uses nearest-neighbor matching on estimated propensity scores to find
    comparable control units for each treated unit. The ATT is estimated as
    the average difference between treated outcomes and matched control outcomes.

    The PSM-ATT estimator is:

    .. math::

        \\hat{\\tau} = \\frac{1}{N_1} \\sum_{D_i=1} \\left[ Y_i - \\frac{1}{k}
                       \\sum_{j \\in M(i)} Y_j \\right]

    where M(i) is the set of k nearest-neighbor matches for unit i.

    Parameters
    ----------
    data : pd.DataFrame
        Cross-sectional data with one row per unit.
    y : str
        Outcome variable column name (typically transformed outcome).
    d : str
        Treatment indicator column name (1=treated, 0=control).
    propensity_controls : list[str]
        Covariate column names for propensity score model.
    n_neighbors : int, default=1
        Number of control matches per treated unit (k).
    with_replacement : bool, default=True
        Whether to match with replacement.
    caliper : float, optional
        Maximum propensity score distance for valid matches.
    caliper_scale : str, default='sd'
        Scale for caliper: 'sd' (standard deviation units) or 'absolute'.
    trim_threshold : float, default=0.01
        Propensity score trimming threshold.
    se_method : str, default='abadie_imbens'
        Standard error method: 'abadie_imbens' uses heteroskedasticity-robust
        variance estimation, 'bootstrap' uses nonparametric bootstrap.
    n_bootstrap : int, default=200
        Number of bootstrap replications when se_method='bootstrap'.
    seed : int, optional
        Random seed for bootstrap resampling.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    match_order : str, default='data'
        Order in which to process treated units for matching.
    return_diagnostics : bool, default=False
        Whether to return additional diagnostic information.
    gvar_col : str, optional
        Cohort indicator column name (for staggered designs).
    ivar_col : str, optional
        Unit identifier column name (for staggered designs).
    cohort_g : int, optional
        Cohort value (for staggered designs).
    period_r : int, optional
        Period value (for staggered designs).

    Returns
    -------
    PSMResult
        Estimation results containing ATT estimate, standard error,
        confidence interval, and matching diagnostics.

    Raises
    ------
    ValueError
        If required columns are missing, sample sizes are insufficient,
        or no valid matches can be found.

    See Also
    --------
    estimate_ipwra : Doubly robust IPWRA estimator.
    estimate_ipw : Pure IPW estimator.
    """
    # Validate inputs.
    _validate_psm_inputs(data, y, d, propensity_controls, n_neighbors)
    
    # Remove observations with missing values.
    all_vars = [y, d] + list(set(propensity_controls))
    data_clean = data[all_vars].dropna().copy()
    
    n = len(data_clean)
    D = data_clean[d].values.astype(float)
    Y = data_clean[y].values.astype(float)
    
    n_treated = int(D.sum())
    n_control = int((1 - D).sum())
    
    if n_treated < 1:
        raise ValueError("No treated units; cannot perform matching.")
    if n_control < n_neighbors:
        raise ValueError(
            f"Control sample size ({n_control}) is less than number of neighbors "
            f"({n_neighbors}); cannot perform matching."
        )
    
    if n_treated < 5:
        warnings.warn(
            f"Small treated sample (n_treated={n_treated}); PSM estimates may be unstable.",
            UserWarning
        )
    
    # Estimate propensity scores.
    pscores, _ = estimate_propensity_score(
        data_clean, d, propensity_controls, trim_threshold
    )
    
    # Perform matching.
    treat_indices = np.where(D == 1)[0]
    control_indices = np.where(D == 0)[0]
    
    pscores_treat = pscores[treat_indices]
    pscores_control = pscores[control_indices]
    
    # Compute caliper if specified.
    actual_caliper = None
    if caliper is not None:
        if caliper_scale == 'sd':
            ps_sd = np.std(pscores)
            actual_caliper = caliper * ps_sd
        else:
            actual_caliper = caliper
    
    # Execute nearest neighbor matching.
    matched_control_ids, match_counts, n_dropped = _nearest_neighbor_match(
        pscores_treat=pscores_treat,
        pscores_control=pscores_control,
        n_neighbors=n_neighbors,
        with_replacement=with_replacement,
        caliper=actual_caliper,
    )
    
    # Compute ATT.
    Y_treat = Y[treat_indices]
    Y_control = Y[control_indices]
    
    # Filter out treated units dropped due to caliper.
    valid_treat_mask = np.array([len(m) > 0 for m in matched_control_ids])
    
    if valid_treat_mask.sum() == 0:
        raise ValueError(
            "No valid matches found for any treated unit. "
            "Consider relaxing the caliper or checking propensity score overlap."
        )
    
    # Compute matched mean for each treated unit.
    att_individual = []
    for i, matches in enumerate(matched_control_ids):
        if len(matches) > 0:
            # Matches contains indices within control group.
            y_matched = Y_control[matches].mean()
            att_i = Y_treat[i] - y_matched
            att_individual.append(att_i)
    
    att = np.mean(att_individual)
    
    # Count unique matched control units.
    all_matched = set()
    for matches in matched_control_ids:
        all_matched.update(matches)
    n_matched = len(all_matched)
    
    # Compute standard errors.
    if se_method == 'abadie_imbens':
        se, ci_lower, ci_upper = _compute_psm_se_abadie_imbens(
            Y_treat=Y_treat,
            Y_control=Y_control,
            matched_control_ids=matched_control_ids,
            att=att,
            alpha=alpha,
        )
    elif se_method == 'bootstrap':
        se, ci_lower, ci_upper = _compute_psm_se_bootstrap(
            data=data_clean,
            y=y,
            d=d,
            propensity_controls=propensity_controls,
            n_neighbors=n_neighbors,
            with_replacement=with_replacement,
            caliper=caliper,
            caliper_scale=caliper_scale,
            trim_threshold=trim_threshold,
            n_bootstrap=n_bootstrap,
            seed=seed,
            alpha=alpha,
        )
    else:
        raise ValueError(
            f"Unknown se_method: {se_method}. "
            "Use 'abadie_imbens' or 'bootstrap'."
        )
    
    # Compute t-statistic and p-value.
    if se > 0:
        t_stat = att / se
        pvalue = 2 * (1 - stats.norm.cdf(abs(t_stat)))
    else:
        t_stat = np.nan
        pvalue = np.nan
    
    return PSMResult(
        att=att,
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        t_stat=t_stat,
        pvalue=pvalue,
        propensity_scores=pscores,
        match_counts=match_counts,
        matched_control_ids=matched_control_ids,
        n_treated=n_treated,
        n_control=n_control,
        n_matched=n_matched,
        caliper=actual_caliper,
        n_dropped=n_dropped,
    )


def _validate_psm_inputs(
    data: pd.DataFrame,
    y: str,
    d: str,
    propensity_controls: list[str],
    n_neighbors: int,
) -> None:
    """
    Validate PSM input parameters.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset to validate.
    y : str
        Outcome variable column name.
    d : str
        Treatment indicator column name.
    propensity_controls : list[str]
        Propensity score control variable names.
    n_neighbors : int
        Number of nearest neighbors.

    Raises
    ------
    ValueError
        If any validation check fails.
    """
    if y not in data.columns:
        raise ValueError(f"Outcome variable '{y}' not found in data.")
    if d not in data.columns:
        raise ValueError(f"Treatment indicator '{d}' not found in data.")
    
    missing_controls = [c for c in propensity_controls if c not in data.columns]
    if missing_controls:
        raise ValueError(f"Propensity score controls not found: {missing_controls}")
    
    if n_neighbors < 1:
        raise ValueError(f"n_neighbors must be >= 1, got: {n_neighbors}")
    
    # Check that treatment indicator is binary.
    d_vals = data[d].dropna().unique()
    if not set(d_vals).issubset({0, 1, 0.0, 1.0}):
        raise ValueError(
            f"Treatment indicator '{d}' must be binary (0/1), "
            f"found unique values: {d_vals}"
        )


def _nearest_neighbor_match(
    pscores_treat: np.ndarray,
    pscores_control: np.ndarray,
    n_neighbors: int,
    with_replacement: bool,
    caliper: float | None,
) -> tuple[list[list[int]], np.ndarray, int]:
    """
    Perform nearest neighbor propensity score matching.

    For each treated unit, finds the k control units with the closest
    propensity scores, subject to optional caliper constraints.

    Parameters
    ----------
    pscores_treat : np.ndarray
        Propensity scores for treated units.
    pscores_control : np.ndarray
        Propensity scores for control units.
    n_neighbors : int
        Number of control units to match per treated unit.
    with_replacement : bool
        Whether control units can be matched to multiple treated units.
    caliper : float or None
        Maximum propensity score distance for valid matches.

    Returns
    -------
    matched_control_ids : list[list[int]]
        List of matched control unit indices for each treated unit.
    match_counts : np.ndarray
        Number of valid matches for each treated unit.
    n_dropped : int
        Number of treated units dropped due to caliper constraint.
    """
    n_treat = len(pscores_treat)
    n_control = len(pscores_control)
    
    matched_control_ids: list[list[int]] = []
    match_counts = np.zeros(n_treat, dtype=int)
    n_dropped = 0
    
    # Track used controls for matching without replacement.
    used_controls: set | None = None if with_replacement else set()
    
    for i in range(n_treat):
        ps_i = pscores_treat[i]
        
        # Compute distances to all control units.
        distances = np.abs(pscores_control - ps_i)
        
        # For matching without replacement, exclude already-used controls.
        if not with_replacement and used_controls:
            available_mask = np.array([
                j not in used_controls for j in range(n_control)
            ])
            if not available_mask.any():
                # No available control units.
                matched_control_ids.append([])
                n_dropped += 1
                continue
            distances[~available_mask] = np.inf
        
        # Apply caliper constraint.
        if caliper is not None:
            valid_mask = distances <= caliper
            if not valid_mask.any():
                # No valid matches within caliper.
                matched_control_ids.append([])
                n_dropped += 1
                continue
        
        # Find k nearest neighbors.
        k = min(n_neighbors, n_control)
        nearest_indices = np.argsort(distances)[:k]
        
        # Verify caliper constraint for selected neighbors.
        if caliper is not None:
            nearest_indices = [
                idx for idx in nearest_indices 
                if distances[idx] <= caliper
            ]
        
        if len(nearest_indices) == 0:
            matched_control_ids.append([])
            n_dropped += 1
            continue
        
        # Record matches.
        matched_control_ids.append(list(nearest_indices))
        match_counts[i] = len(nearest_indices)
        
        # Mark controls as used for matching without replacement.
        if not with_replacement and used_controls is not None:
            used_controls.update(nearest_indices)
    
    return matched_control_ids, match_counts, n_dropped


def _compute_psm_se_abadie_imbens(
    Y_treat: np.ndarray,
    Y_control: np.ndarray,
    matched_control_ids: list[list[int]],
    att: float,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """
    Compute heteroskedasticity-robust standard error for matching estimators.

    Uses the variance of individual treatment effects to estimate the
    standard error of the ATT, based on the asymptotic variance formula
    that accounts for heterogeneous treatment effects across matched pairs.

    Parameters
    ----------
    Y_treat : np.ndarray
        Outcome values for treated units.
    Y_control : np.ndarray
        Outcome values for control units.
    matched_control_ids : list[list[int]]
        Matched control indices for each treated unit.
    att : float
        Estimated ATT value.
    alpha : float, default=0.05
        Significance level for confidence interval.

    Returns
    -------
    se : float
        Standard error estimate.
    ci_lower : float
        Lower bound of confidence interval.
    ci_upper : float
        Upper bound of confidence interval.
    """
    n_valid = sum(1 for m in matched_control_ids if len(m) > 0)
    
    if n_valid < 2:
        return np.nan, np.nan, np.nan
    
    # Compute individual treatment effects for each matched pair.
    individual_effects = []
    for i, matches in enumerate(matched_control_ids):
        if len(matches) > 0:
            y_matched = Y_control[matches].mean()
            effect_i = Y_treat[i] - y_matched
            individual_effects.append(effect_i)
    
    individual_effects = np.array(individual_effects)
    
    # Use variance of individual effects for SE estimation.
    var_effects = np.var(individual_effects, ddof=1)
    var_att = var_effects / n_valid
    se = np.sqrt(var_att)
    
    # Confidence interval.
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lower = att - z_crit * se
    ci_upper = att + z_crit * se
    
    return se, ci_lower, ci_upper


def _compute_psm_se_bootstrap(
    data: pd.DataFrame,
    y: str,
    d: str,
    propensity_controls: list[str],
    n_neighbors: int,
    with_replacement: bool,
    caliper: float | None,
    caliper_scale: str,
    trim_threshold: float,
    n_bootstrap: int,
    seed: int | None,
    alpha: float,
) -> tuple[float, float, float]:
    """
    Compute PSM standard error using nonparametric bootstrap.

    Resamples the entire PSM procedure including propensity score estimation
    and nearest-neighbor matching. This approach properly accounts for all
    sources of estimation uncertainty.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset used for estimation.
    y : str
        Outcome variable column name.
    d : str
        Treatment indicator column name.
    propensity_controls : list[str]
        Propensity score control variable names.
    n_neighbors : int
        Number of nearest neighbors.
    with_replacement : bool
        Whether to match with replacement.
    caliper : float or None
        Caliper constraint value.
    caliper_scale : str
        Scale for caliper ('sd' or 'absolute').
    trim_threshold : float
        Propensity score trimming threshold.
    n_bootstrap : int
        Number of bootstrap replications.
    seed : int or None
        Random seed for reproducibility.
    alpha : float
        Significance level for confidence interval.

    Returns
    -------
    se : float
        Bootstrap standard error.
    ci_lower : float
        Percentile confidence interval lower bound.
    ci_upper : float
        Percentile confidence interval upper bound.
    """
    if seed is not None:
        np.random.seed(seed)
    
    n = len(data)
    att_boots = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement.
        indices = np.random.choice(n, size=n, replace=True)
        data_boot = data.iloc[indices].reset_index(drop=True)
        
        try:
            # Estimate propensity scores.
            pscores_boot, _ = estimate_propensity_score(
                data_boot, d, propensity_controls, trim_threshold
            )
            
            D_boot = data_boot[d].values.astype(float)
            Y_boot = data_boot[y].values.astype(float)
            
            treat_indices = np.where(D_boot == 1)[0]
            control_indices = np.where(D_boot == 0)[0]
            
            if len(treat_indices) < 1 or len(control_indices) < n_neighbors:
                continue
            
            pscores_treat = pscores_boot[treat_indices]
            pscores_control = pscores_boot[control_indices]
            
            # Compute caliper.
            actual_caliper = None
            if caliper is not None:
                if caliper_scale == 'sd':
                    ps_sd = np.std(pscores_boot)
                    actual_caliper = caliper * ps_sd
                else:
                    actual_caliper = caliper
            
            # Perform matching.
            matched_ids, _, _ = _nearest_neighbor_match(
                pscores_treat, pscores_control,
                n_neighbors, with_replacement, actual_caliper
            )
            
            # Compute ATT.
            Y_treat = Y_boot[treat_indices]
            Y_control = Y_boot[control_indices]
            
            att_individual = []
            for i, matches in enumerate(matched_ids):
                if len(matches) > 0:
                    y_matched = Y_control[matches].mean()
                    att_i = Y_treat[i] - y_matched
                    att_individual.append(att_i)
            
            if len(att_individual) > 0:
                att_boot = np.mean(att_individual)
                att_boots.append(att_boot)

        except (ValueError, np.linalg.LinAlgError, FloatingPointError, ZeroDivisionError):
            # Skip failed bootstrap iterations due to numerical issues.
            continue
    
    if len(att_boots) < n_bootstrap * 0.5:
        warnings.warn(
            f"Low bootstrap success rate: {len(att_boots)}/{n_bootstrap}",
            UserWarning
        )
    
    if len(att_boots) < 10:
        raise ValueError(f"Insufficient bootstrap samples: {len(att_boots)}")
    
    att_boots = np.array(att_boots)
    se = np.std(att_boots, ddof=1)
    ci_lower = np.percentile(att_boots, 100 * alpha / 2)
    ci_upper = np.percentile(att_boots, 100 * (1 - alpha / 2))
    
    return se, ci_lower, ci_upper

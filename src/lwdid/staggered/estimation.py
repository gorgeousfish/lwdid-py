"""
Cohort-time ATT estimation for staggered difference-in-differences designs.

This module implements treatment effect estimation for staggered adoption
settings where units begin treatment at different time periods. Each
(cohort, period) pair receives its own ATT estimate through cross-sectional
regression on transformed outcome data.

The estimation proceeds in three steps:

1. Transform outcomes by removing pre-treatment unit-specific patterns
   (demeaning or detrending) for each treatment cohort.
2. For each treated cohort g in post-treatment period r, restrict the
   sample to cohort g units plus valid control units.
3. Run cross-sectional regression of transformed outcomes on a treatment
   indicator to estimate the ATT.

Valid control units for cohort g at time r include never-treated units
and units first treated after r. This rolling approach efficiently uses
not-yet-treated observations while maintaining identification under
conditional parallel trends and no anticipation assumptions.

Notes
-----
Three estimation methods are supported:

- Regression adjustment (RA): OLS estimation with optional covariate
  adjustment. Enables exact t-based inference under classical linear
  model assumptions.
- Inverse probability weighted regression adjustment (IPWRA): Doubly
  robust estimation combining propensity score weighting with outcome
  regression. Consistent if either model is correctly specified.
- Propensity score matching (PSM): Nearest-neighbor matching on
  estimated propensity scores.

The cross-sectional regression for each (g, r) pair has the form:

.. math::

    \\hat{Y}_{irg} = \\alpha + \\tau_{gr} D_{ig} + X_i \\beta + u_i

where :math:`\\hat{Y}_{irg}` is the transformed outcome, :math:`D_{ig}`
is the treatment indicator for cohort g, and :math:`\\tau_{gr}` is the
ATT for cohort g in period r.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats

from .control_groups import (
    ControlGroupStrategy,
    get_valid_control_units,
)
from .transformations import get_cohorts, get_valid_periods_for_cohort
from .estimators import estimate_ipwra, estimate_psm, IPWRAResult, PSMResult


@dataclass
class CohortTimeEffect:
    """
    Container for a single cohort-time treatment effect estimate.

    Stores the ATT for treatment cohort g at calendar time r, along with
    inference statistics from cross-sectional regression.

    Attributes
    ----------
    cohort : int
        Treatment cohort identifier (first treatment period g).
    period : int
        Calendar time period r.
    event_time : int
        Event time relative to treatment onset (e = r - g).
    att : float
        Estimated average treatment effect on the treated.
    se : float
        Standard error of the ATT estimate.
    ci_lower : float
        Lower bound of the confidence interval.
    ci_upper : float
        Upper bound of the confidence interval.
    t_stat : float
        t-statistic for testing H0: ATT = 0.
    pvalue : float
        Two-sided p-value from t-distribution.
    n_treated : int
        Number of treated units in estimation sample.
    n_control : int
        Number of control units in estimation sample.
    n_total : int
        Total sample size (n_treated + n_control).
    df_resid : int
        Residual degrees of freedom. Default is 0.
    df_inference : int
        Degrees of freedom for inference. Default is 0.
    """
    cohort: int
    period: int
    event_time: int
    att: float
    se: float
    ci_lower: float
    ci_upper: float
    t_stat: float
    pvalue: float
    n_treated: int
    n_control: int
    n_total: int
    df_resid: int = 0
    df_inference: int = 0


# =============================================================================
# HC Standard Error Helper Functions
# =============================================================================

# Constants for numerical stability
_LEVERAGE_CLIP_MAX = 0.9999
_LEVERAGE_WARN_THRESHOLD = 0.9


def _compute_leverage(
    X: np.ndarray,
    XtX_inv: np.ndarray,
    clip_max: float = _LEVERAGE_CLIP_MAX,
    warn_threshold: float = _LEVERAGE_WARN_THRESHOLD,
) -> np.ndarray:
    """
    Compute leverage values (hat matrix diagonal elements).

    Uses an efficient vectorized method that avoids constructing the full
    N×N hat matrix, reducing time complexity from O(N²×K) to O(N×K²).

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (N, K).
    XtX_inv : np.ndarray
        Inverse of X'X matrix of shape (K, K).
    clip_max : float, default=0.9999
        Upper bound for leverage value clipping to prevent division by zero.
    warn_threshold : float, default=0.9
        Threshold above which to warn about extreme leverage values.

    Returns
    -------
    np.ndarray
        Leverage values of shape (N,), clipped to [0, clip_max].

    Notes
    -----
    The leverage value h_ii = x_i'(X'X)^{-1}x_i measures how much influence
    observation i has on its own fitted value. Properties:

    - 0 < h_ii < 1 for all i
    - sum(h_ii) = K (number of parameters)
    - Average leverage = K/N
    - High leverage threshold typically 2K/N or 3K/N

    The efficient computation uses:
    h_ii = diag(X @ XtX_inv @ X.T) = (X @ XtX_inv * X).sum(axis=1)
    """
    # Efficient computation avoiding N×N matrix
    # h_ii = diag(H) where H = X @ XtX_inv @ X.T
    # This is equivalent to: (X @ XtX_inv * X).sum(axis=1)
    tmp = X @ XtX_inv  # (N, K)
    h_ii = (tmp * X).sum(axis=1)  # (N,)

    # Check for extreme leverage values
    max_h = np.max(h_ii)
    if max_h > warn_threshold:
        warnings.warn(
            f"Extreme leverage detected: max(h_ii) = {max_h:.4f} > {warn_threshold}. "
            f"HC standard errors may be numerically unstable. "
            f"Consider using HC4 or checking for influential observations.",
            UserWarning
        )

    # Clip to ensure numerical stability (prevent division by zero in HC2-HC4)
    h_ii = np.clip(h_ii, 0, clip_max)

    return h_ii


def _compute_hc_variance(
    X: np.ndarray,
    residuals: np.ndarray,
    XtX_inv: np.ndarray,
    hc_type: str,
) -> np.ndarray:
    """
    Compute heteroskedasticity-consistent (HC) variance-covariance matrix.

    Implements the sandwich variance estimator for HC0 through HC4:
    V̂ = (X'X)^{-1} X' Ω X (X'X)^{-1}

    where Ω is a diagonal matrix with elements depending on the HC type.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (N, K).
    residuals : np.ndarray
        OLS residuals of shape (N,).
    XtX_inv : np.ndarray
        Inverse of X'X matrix of shape (K, K).
    hc_type : str
        HC type: 'hc0', 'hc1', 'hc2', 'hc3', 'hc4', or 'robust'.

    Returns
    -------
    np.ndarray
        Variance-covariance matrix of shape (K, K).

    Raises
    ------
    ValueError
        If hc_type is not a valid HC type.

    Notes
    -----
    HC formulas (Ω_ii diagonal elements):

    - HC0: e_i² (no adjustment)
    - HC1: (n/(n-k)) × e_i² (degrees-of-freedom correction)
    - HC2: e_i² / (1 - h_ii) (leverage adjustment)
    - HC3: e_i² / (1 - h_ii)² (small-sample adjustment)
    - HC4: e_i² / (1 - h_ii)^δ_i where δ_i = min(4, n·h_ii/k) (adaptive)

    Typical ordering: SE(HC0) ≤ SE(HC1) and SE(HC2) ≤ SE(HC3).
    """
    n, k = X.shape
    e2 = residuals ** 2

    # Case-insensitive matching for user convenience
    hc_lower = hc_type.lower() if hc_type else None

    if hc_lower == 'hc0':
        # HC0: Basic heteroskedasticity-robust estimator without adjustment.
        # Ω_ii = e_i²
        omega_diag = e2

    elif hc_lower in ('hc1', 'robust'):
        # HC1: Degrees-of-freedom adjustment for finite samples.
        # Ω_ii = (n/(n-k)) × e_i²
        correction = n / (n - k)
        omega_diag = correction * e2

    elif hc_lower == 'hc2':
        # HC2: Leverage-adjusted estimator using hat matrix diagonal.
        # Ω_ii = e_i² / (1 - h_ii)
        h_ii = _compute_leverage(X, XtX_inv)
        omega_diag = e2 / (1 - h_ii)

    elif hc_lower == 'hc3':
        # HC3: Small-sample adjustment using squared leverage correction.
        # Ω_ii = e_i² / (1 - h_ii)²
        # Recommended for small samples with few treated or control units.
        h_ii = _compute_leverage(X, XtX_inv)
        omega_diag = e2 / ((1 - h_ii) ** 2)

    elif hc_lower == 'hc4':
        # HC4: Adaptive leverage adjustment for high-influence observations.
        # Ω_ii = e_i² / (1 - h_ii)^δ_i where δ_i = min(4, n·h_ii/k)
        # Provides stronger adjustment for observations with extreme leverage.
        h_ii = _compute_leverage(X, XtX_inv)
        delta = np.minimum(4.0, n * h_ii / k)
        # Guard against numerical issues from negative leverage estimates
        delta = np.maximum(delta, 0.0)
        omega_diag = e2 / ((1 - h_ii) ** delta)

    else:
        valid_types = "'hc0', 'hc1', 'hc2', 'hc3', 'hc4', 'robust'"
        raise ValueError(
            f"Unknown HC type: '{hc_type}'. Valid options: {valid_types}"
        )

    # Sandwich variance: V̂ = (X'X)^{-1} X' Ω X (X'X)^{-1}
    # Using efficient computation: meat = X.T @ diag(omega) @ X
    meat = X.T @ np.diag(omega_diag) @ X
    var_beta = XtX_inv @ meat @ XtX_inv

    return var_beta


def _compute_hc1_variance(
    X: np.ndarray,
    residuals: np.ndarray,
    XtX_inv: np.ndarray,
) -> np.ndarray:
    """
    Compute HC1 variance-covariance matrix.

    This is a convenience wrapper for backward compatibility with existing
    test code that imports this function directly.

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (N, K).
    residuals : np.ndarray
        OLS residuals of shape (N,).
    XtX_inv : np.ndarray
        Inverse of X'X matrix of shape (K, K).

    Returns
    -------
    np.ndarray
        HC1 variance-covariance matrix of shape (K, K).

    Raises
    ------
    ValueError
        If n <= k (insufficient degrees of freedom).

    See Also
    --------
    _compute_hc_variance : General HC variance computation.
    """
    n, k = X.shape
    if n <= k:
        raise ValueError(
            f"HC1 variance requires n > k. Got n={n}, k={k}."
        )
    return _compute_hc_variance(X, residuals, XtX_inv, 'hc1')


def run_ols_regression(
    data: pd.DataFrame,
    y: str,
    d: str,
    controls: list[str] | None = None,
    vce: str | None = None,
    cluster_var: str | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Estimate the ATT via OLS regression on cross-sectional data.

    Regresses the transformed outcome on a constant and treatment indicator,
    optionally including control variables. When controls are included and
    sample sizes permit, interactions with the treatment indicator (centered
    at treated-group mean) are added to allow heterogeneous covariate effects.

    Parameters
    ----------
    data : pd.DataFrame
        Cross-sectional data with one row per unit.
    y : str
        Name of the dependent variable column (transformed outcome).
    d : str
        Name of the treatment indicator column.
    controls : list of str, optional
        Names of control variable columns.
    vce : {None, 'hc0', 'hc1', 'hc2', 'hc3', 'hc4', 'robust', 'cluster'}, optional
        Variance estimation method (case-insensitive):

        - None: Homoskedastic OLS standard errors. Enables exact t-based
          inference under normality assumption.
        - 'hc0': Basic heteroskedasticity-robust. No finite-sample adjustment.
        - 'hc1' or 'robust': HC1 with degrees-of-freedom correction n/(n-k).
        - 'hc2': Leverage-adjusted using (1 - h_ii)^{-1}.
        - 'hc3': Small-sample adjusted using (1 - h_ii)^{-2}. Recommended
          for small samples with few treated or control units.
        - 'hc4': Adaptive leverage correction using δ_i = min(4, n·h_ii/k).
          Recommended when extreme leverage exists.
        - 'cluster': Cluster-robust. Requires ``cluster_var``.

    cluster_var : str, optional
        Cluster variable column. Required when vce='cluster'.
    alpha : float, default=0.05
        Significance level for confidence interval construction.

    Returns
    -------
    dict
        Estimation results with keys: 'att', 'se', 'ci_lower', 'ci_upper',
        't_stat', 'pvalue', 'nobs', 'df_resid', 'df_inference'.

    Raises
    ------
    ValueError
        If required columns are missing, sample size is insufficient,
        or the design matrix is singular.

    See Also
    --------
    estimate_cohort_time_effects : Estimate ATT for all cohort-period pairs.
    _compute_hc_variance : HC variance computation details.

    Notes
    -----
    HC standard error ordering (typical):

    - SE(HC0) ≤ SE(HC1) due to degrees-of-freedom correction
    - SE(HC2) ≤ SE(HC3) due to stronger leverage adjustment
    - SE(HC4) adapts based on leverage; may exceed HC3 for high-leverage obs
    """
    if y not in data.columns:
        raise ValueError(f"Dependent variable '{y}' not in data")
    if d not in data.columns:
        raise ValueError(f"Treatment indicator '{d}' not in data")

    valid_mask = data[y].notna() & data[d].notna()
    data_clean = data[valid_mask].copy()

    n = len(data_clean)
    if n < 2:
        raise ValueError(f"Insufficient observations: n={n}, need at least 2")

    y_vals = data_clean[y].values.astype(float)
    d_vals = data_clean[d].values.astype(float)

    if controls is not None and len(controls) > 0:
        missing_controls = [c for c in controls if c not in data_clean.columns]
        if missing_controls:
            raise ValueError(f"Control variables not found: {missing_controls}")

        # Complete cases required for covariate adjustment to avoid bias
        control_valid = data_clean[controls].notna().all(axis=1)
        data_clean = data_clean[control_valid].copy()
        y_vals = data_clean[y].values.astype(float)
        d_vals = data_clean[d].values.astype(float)
        n = len(data_clean)

        if n < 2:
            raise ValueError(f"Insufficient observations after dropping missing controls: n={n}")

        treated_mask = d_vals == 1
        n_treated = treated_mask.sum()
        n_control = (~treated_mask).sum()

        K = len(controls)
        if n_treated > K + 1 and n_control > K + 1:
            # Centering covariates at treated-group mean ensures the coefficient
            # on D directly estimates ATT; interaction terms allow heterogeneous
            # slopes across treatment and control groups.
            X_controls = data_clean[controls].values.astype(float)
            X_mean_treated = X_controls[treated_mask].mean(axis=0)
            X_centered = X_controls - X_mean_treated
            X_interactions = d_vals.reshape(-1, 1) * X_centered

            X = np.column_stack([
                np.ones(n),
                d_vals,
                X_controls,
                X_interactions
            ])
        else:
            # Insufficient sample size for covariate adjustment
            warnings.warn(
                f"Controls not included: N_treated={n_treated}, N_control={n_control}, K={K}. "
                f"Need N_treated > K+1 and N_control > K+1.",
                UserWarning
            )
            X = np.column_stack([np.ones(n), d_vals])
    else:
        # Without covariates, simple difference-in-means identifies the ATT
        X = np.column_stack([np.ones(n), d_vals])

    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        raise ValueError("Design matrix is singular, cannot estimate")

    beta = XtX_inv @ (X.T @ y_vals)

    y_hat = X @ beta
    residuals = y_vals - y_hat
    df_resid = n - X.shape[1]

    if df_resid <= 0:
        att = beta[1]
        warnings.warn(
            f"No degrees of freedom (n={n}, k={X.shape[1]}). "
            f"Point estimate is valid but SE, CI, and p-value are unavailable.",
            UserWarning
        )
        return {
            'att': att,
            'se': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            't_stat': np.nan,
            'pvalue': np.nan,
            'nobs': n,
            'df_resid': df_resid,
            'df_inference': 0,
        }

    sigma2 = (residuals ** 2).sum() / df_resid

    # Case-insensitive matching for user convenience
    vce_lower = vce.lower() if vce else None

    if vce_lower is None:
        # Homoskedastic variance enables exact t-based inference
        var_beta = sigma2 * XtX_inv
        df_inference = df_resid

    elif vce_lower in ('hc0', 'hc1', 'hc2', 'hc3', 'hc4', 'robust'):
        # Heteroskedasticity-consistent standard errors
        var_beta = _compute_hc_variance(X, residuals, XtX_inv, vce_lower)
        df_inference = df_resid

    elif vce_lower == 'cluster':
        if cluster_var is None:
            raise ValueError("cluster_var required when vce='cluster'")
        if cluster_var not in data_clean.columns:
            raise ValueError(f"Cluster variable '{cluster_var}' not in data")

        cluster_ids = data_clean[cluster_var].values
        unique_clusters = np.unique(cluster_ids)
        G = len(unique_clusters)

        if G < 2:
            raise ValueError(f"Need at least 2 clusters, got {G}")

        # Sum of cluster-level outer products accounts for within-cluster
        # correlation of errors; this is the "meat" of the sandwich estimator.
        meat = np.zeros((X.shape[1], X.shape[1]))
        for cluster in unique_clusters:
            mask = cluster_ids == cluster
            X_g = X[mask]
            e_g = residuals[mask]
            score_g = X_g.T @ e_g
            meat += np.outer(score_g, score_g)

        # Finite-sample correction reduces bias with few clusters (G < 50)
        correction = (G / (G - 1)) * ((n - 1) / (n - X.shape[1]))
        var_beta = correction * XtX_inv @ meat @ XtX_inv
        df_inference = G - 1

    else:
        valid_options = "None, 'hc0', 'hc1', 'hc2', 'hc3', 'hc4', 'robust', 'cluster'"
        raise ValueError(
            f"Unknown vce type: '{vce}'. Valid options: {valid_options}"
        )

    # Treatment coefficient is in position 1 regardless of whether controls included
    att = beta[1]
    se = np.sqrt(var_beta[1, 1])

    if se > 0:
        t_stat = att / se
        pvalue = 2 * (1 - stats.t.cdf(abs(t_stat), df_inference))
    else:
        t_stat = np.nan
        pvalue = np.nan

    t_crit = stats.t.ppf(1 - alpha / 2, df_inference)
    ci_lower = att - t_crit * se
    ci_upper = att + t_crit * se

    return {
        'att': att,
        'se': se,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        't_stat': t_stat,
        'pvalue': pvalue,
        'nobs': n,
        'df_resid': df_resid,
        'df_inference': df_inference,
    }


def estimate_cohort_time_effects(
    data_transformed: pd.DataFrame,
    gvar: str,
    ivar: str,
    tvar: str,
    controls: list[str] | None = None,
    vce: str | None = None,
    cluster_var: str | None = None,
    control_strategy: str = 'not_yet_treated',
    never_treated_values: list | None = None,
    min_obs: int = 3,
    min_treated: int = 1,
    min_control: int = 1,
    alpha: float = 0.05,
    estimator: str = 'ra',
    transform_type: str = 'demean',
    propensity_controls: list[str] | None = None,
    trim_threshold: float = 0.01,
    se_method: str = 'analytical',
    n_neighbors: int = 1,
    with_replacement: bool = True,
    caliper: float | None = None,
    return_diagnostics: bool = False,
    match_order: str | None = None,
) -> list[CohortTimeEffect]:
    """
    Estimate treatment effects for all valid cohort-period pairs.

    Iterates over all treatment cohorts and their post-treatment periods,
    estimating the ATT for each (cohort, period) combination using the
    specified estimation method. For each pair, the estimation sample
    consists of the treatment cohort units plus valid control units
    determined by the control group strategy.

    Parameters
    ----------
    data_transformed : pd.DataFrame
        Panel data containing transformed outcome columns generated by
        ``transform_staggered_demean`` or ``transform_staggered_detrend``.
        Must include columns for gvar, ivar, tvar, and transformed
        outcomes named 'ydot_g{g}_r{r}' or 'ycheck_g{g}_r{r}'.
    gvar : str
        Name of the cohort variable column indicating first treatment period.
    ivar : str
        Name of the unit identifier column.
    tvar : str
        Name of the time variable column.
    controls : list of str, optional
        Names of time-invariant control variable columns.
    vce : str, optional
        Variance estimation type: None (homoskedastic), 'hc3'
        (heteroskedasticity-robust), or 'cluster' (cluster-robust).
    cluster_var : str, optional
        Name of the cluster variable column. Required when vce='cluster'.
    control_strategy : str, default='not_yet_treated'
        Control group selection strategy: 'never_treated' uses only
        never-treated units; 'not_yet_treated' includes units first
        treated after the current period; 'all_others' uses all units
        not in the treatment cohort (including already-treated units);
        'auto' selects based on data availability.
    never_treated_values : list, optional
        Values in gvar indicating never-treated units. Defaults to
        [0, np.inf] and NaN values.
    min_obs : int, default=3
        Minimum total sample size required for estimation.
    min_treated : int, default=1
        Minimum number of treated units required.
    min_control : int, default=1
        Minimum number of control units required.
    alpha : float, default=0.05
        Significance level for confidence interval construction.
    estimator : str, default='ra'
        Estimation method: 'ra' (regression adjustment), 'ipwra'
        (inverse probability weighted regression adjustment), or
        'psm' (propensity score matching).
    transform_type : str, default='demean'
        Transformation type applied to the data: 'demean' or 'detrend'.
        Determines the column prefix for transformed outcomes.
    propensity_controls : list of str, optional
        Control variables for the propensity score model. If None,
        uses the same variables as ``controls``.
    trim_threshold : float, default=0.01
        Propensity score trimming threshold for IPWRA and PSM.
        Observations with extreme propensity scores are excluded.
    se_method : str, default='analytical'
        Standard error method for IPWRA: 'analytical' or 'bootstrap'.
    n_neighbors : int, default=1
        Number of nearest neighbors for PSM matching.
    with_replacement : bool, default=True
        Whether PSM matching allows replacement.
    caliper : float, optional
        Maximum propensity score distance for PSM matching.
    return_diagnostics : bool, default=False
        Reserved for future use. Currently has no effect.
    match_order : str, optional
        Reserved for future use. Currently has no effect.

    Returns
    -------
    list of CohortTimeEffect
        Estimation results for all valid (cohort, period) pairs,
        sorted by cohort and period.

    Raises
    ------
    ValueError
        If required columns are missing, no valid treatment cohorts
        exist, or parameter values are invalid.

    See Also
    --------
    transform_staggered_demean : Demeaning transformation for staggered designs.
    transform_staggered_detrend : Detrending transformation for staggered designs.
    aggregate_to_cohort : Aggregate cohort-time effects to cohort-level effects.
    aggregate_to_overall : Aggregate effects to a single overall estimate.

    Notes
    -----
    The estimation proceeds separately for each (cohort, period) pair. For
    treatment cohort g in calendar time r, the control group consists of
    never-treated units and units first treated after period r. This rolling
    selection ensures that control units satisfy no anticipation at time r.

    Under the 'not_yet_treated' strategy, the control pool varies by period:
    earlier periods have more controls available since fewer cohorts have
    been treated. The 'never_treated' strategy uses a fixed control pool
    across all periods, which may be more restrictive but avoids potential
    confounding from units that eventually receive treatment.

    Sample size requirements (min_obs, min_treated, min_control) are checked
    before estimation. Pairs that fail these checks are silently skipped
    and reported in the warning message.
    """
    # =========================================================================
    # Input Validation
    # =========================================================================
    required_cols = [gvar, ivar, tvar]
    missing = [c for c in required_cols if c not in data_transformed.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if vce == 'cluster' and cluster_var is None:
        raise ValueError("cluster_var required when vce='cluster'")

    strategy_map = {
        'never_treated': ControlGroupStrategy.NEVER_TREATED,
        'not_yet_treated': ControlGroupStrategy.NOT_YET_TREATED,
        'all_others': ControlGroupStrategy.ALL_OTHERS,
        'auto': ControlGroupStrategy.AUTO,
    }
    if control_strategy not in strategy_map:
        raise ValueError(
            f"Invalid control_strategy: {control_strategy}. "
            f"Must be one of: {list(strategy_map.keys())}"
        )
    strategy = strategy_map[control_strategy]

    # =========================================================================
    # Cohort and Period Extraction
    # =========================================================================
    if never_treated_values is None:
        nt_values = [0, np.inf]
    else:
        nt_values = never_treated_values

    cohorts = get_cohorts(data_transformed, gvar, ivar, nt_values)

    if len(cohorts) == 0:
        raise ValueError("No valid treatment cohorts found in data.")

    T_max = int(data_transformed[tvar].max())

    # =========================================================================
    # Cohort-Time Effect Estimation
    # =========================================================================
    results = []
    skipped_pairs = []

    # Column prefix reflects transformation type: 'ydot' for demeaned, 'ycheck' for detrended
    prefix = 'ydot' if transform_type == 'demean' else 'ycheck'

    def _safe_int(value: Any) -> int:
        """Coerce possibly-missing numeric values to a safe int (NaN/None -> 0)."""
        if value is None:
            return 0
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            return 0
        if not np.isfinite(value_f):
            return 0
        return int(value_f)

    for g in cohorts:
        valid_periods = get_valid_periods_for_cohort(g, T_max)

        for r in valid_periods:
            ydot_col = f'{prefix}_g{g}_r{r}'

            if ydot_col not in data_transformed.columns:
                skipped_pairs.append((g, r, 'missing_transform_column'))
                continue

            # -------------------------------------------------------------------------
            # Extract Period Cross-Section
            # -------------------------------------------------------------------------
            period_data = data_transformed[data_transformed[tvar] == r].copy()

            if len(period_data) == 0:
                skipped_pairs.append((g, r, 'no_data_in_period'))
                continue

            # -------------------------------------------------------------------------
            # Identify Valid Control Units
            # -------------------------------------------------------------------------
            try:
                unit_control_mask = get_valid_control_units(
                    period_data, gvar, ivar,
                    cohort=g, period=r,
                    strategy=strategy,
                    never_treated_values=nt_values,
                )
            except Exception as e:
                skipped_pairs.append((g, r, f'control_mask_error: {e}'))
                continue

            # Map unit-level control status to observations; missing units default
            # to non-control to ensure conservative sample construction.
            control_mask = period_data[ivar].map(unit_control_mask).fillna(False).astype(bool)

            # -------------------------------------------------------------------------
            # Construct Estimation Sample
            # -------------------------------------------------------------------------
            treat_mask = (period_data[gvar] == g)
            sample_mask = treat_mask | control_mask

            n_treat = treat_mask.sum()
            n_control = control_mask.sum()
            n_total = sample_mask.sum()

            if n_total < min_obs:
                skipped_pairs.append((g, r, f'insufficient_total: {n_total}<{min_obs}'))
                continue
            if n_treat < min_treated:
                skipped_pairs.append((g, r, f'insufficient_treated: {n_treat}<{min_treated}'))
                continue
            if n_control < min_control:
                skipped_pairs.append((g, r, f'insufficient_control: {n_control}<{min_control}'))
                continue

            sample_data = period_data[sample_mask].copy()
            # Binary indicator needed for OLS: 1 for cohort g, 0 for controls
            sample_data['_D_treat'] = (sample_data[gvar] == g).astype(int)

            # -------------------------------------------------------------------------
            # Run Estimator
            # -------------------------------------------------------------------------
            try:
                if estimator == 'ra':
                    est_result = run_ols_regression(
                        data=sample_data,
                        y=ydot_col,
                        d='_D_treat',
                        controls=controls,
                        vce=vce,
                        cluster_var=cluster_var,
                        alpha=alpha,
                    )
                elif estimator == 'ipwra':
                    est_result = _estimate_single_effect_ipwra(
                        data=sample_data,
                        y=ydot_col,
                        d='_D_treat',
                        controls=controls or [],
                        propensity_controls=propensity_controls or controls or [],
                        trim_threshold=trim_threshold,
                        se_method=se_method,
                        alpha=alpha,
                    )
                elif estimator == 'psm':
                    est_result = _estimate_single_effect_psm(
                        data=sample_data,
                        y=ydot_col,
                        d='_D_treat',
                        propensity_controls=propensity_controls or controls or [],
                        n_neighbors=n_neighbors,
                        with_replacement=with_replacement,
                        caliper=caliper,
                        trim_threshold=trim_threshold,
                        alpha=alpha,
                    )
                else:
                    raise ValueError(f"Unknown estimator: {estimator}")
            except Exception as e:
                skipped_pairs.append((g, r, f'{estimator}_error: {e}'))
                continue

            results.append(CohortTimeEffect(
                cohort=int(g),
                period=int(r),
                event_time=int(r - g),
                att=est_result['att'],
                se=est_result['se'],
                ci_lower=est_result['ci_lower'],
                ci_upper=est_result['ci_upper'],
                t_stat=est_result['t_stat'],
                pvalue=est_result['pvalue'],
                n_treated=int(n_treat),
                n_control=int(n_control),
                n_total=int(n_total),
                df_resid=_safe_int(est_result.get('df_resid', 0)),
                df_inference=_safe_int(est_result.get('df_inference', 0)),
            ))

    # =========================================================================
    # Reporting
    # =========================================================================
    if skipped_pairs:
        n_skipped = len(skipped_pairs)
        n_total_pairs = sum(len(get_valid_periods_for_cohort(g, T_max)) for g in cohorts)
        warnings.warn(
            f"Skipped {n_skipped}/{n_total_pairs} (cohort, period) pairs due to "
            f"insufficient data or errors. Use verbose=True for details.",
            UserWarning
        )

    results.sort(key=lambda x: (x.cohort, x.period))

    return results


def results_to_dataframe(results: list[CohortTimeEffect]) -> pd.DataFrame:
    """
    Convert a list of CohortTimeEffect objects to a pandas DataFrame.

    Parameters
    ----------
    results : list of CohortTimeEffect
        Estimation results from ``estimate_cohort_time_effects``.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: cohort, period, event_time, att, se,
        ci_lower, ci_upper, t_stat, pvalue, n_treated, n_control, n_total.
        Returns an empty DataFrame with appropriate columns if the input
        list is empty.

    See Also
    --------
    estimate_cohort_time_effects : Estimate ATT for all cohort-period pairs.
    CohortTimeEffect : Container class for individual effect estimates.
    """
    if len(results) == 0:
        return pd.DataFrame(columns=[
            'cohort', 'period', 'event_time', 'att', 'se',
            'ci_lower', 'ci_upper', 't_stat', 'pvalue',
            'n_treated', 'n_control', 'n_total'
        ])

    return pd.DataFrame([
        {
            'cohort': r.cohort,
            'period': r.period,
            'event_time': r.event_time,
            'att': r.att,
            'se': r.se,
            'ci_lower': r.ci_lower,
            'ci_upper': r.ci_upper,
            't_stat': r.t_stat,
            'pvalue': r.pvalue,
            'n_treated': r.n_treated,
            'n_control': r.n_control,
            'n_total': r.n_total,
        }
        for r in results
    ])


# =============================================================================
# Internal Estimator Wrappers
# =============================================================================

def _estimate_single_effect_ipwra(
    data: pd.DataFrame,
    y: str,
    d: str,
    controls: list[str],
    propensity_controls: list[str],
    trim_threshold: float = 0.01,
    se_method: str = 'analytical',
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Estimate ATT using inverse probability weighted regression adjustment.

    Internal wrapper that calls the IPWRA estimator and returns results
    in a standardized dictionary format consistent with ``run_ols_regression``.

    Parameters
    ----------
    data : pd.DataFrame
        Cross-sectional sample data with one row per unit.
    y : str
        Name of the transformed outcome variable column.
    d : str
        Name of the treatment indicator column.
    controls : list of str
        Control variables for the outcome regression model.
    propensity_controls : list of str
        Control variables for the propensity score model.
    trim_threshold : float, default=0.01
        Threshold for trimming extreme propensity scores.
    se_method : str, default='analytical'
        Standard error computation method: 'analytical' or 'bootstrap'.
    alpha : float, default=0.05
        Significance level for confidence interval construction.

    Returns
    -------
    dict
        Estimation results with keys: 'att', 'se', 'ci_lower', 'ci_upper',
        't_stat', 'pvalue', 'nobs', 'df_resid', 'df_inference', 'estimator'.
        Returns NaN values if estimation fails.
    """
    try:
        result: IPWRAResult = estimate_ipwra(
            data=data,
            y=y,
            d=d,
            controls=controls,
            propensity_controls=propensity_controls,
            trim_threshold=trim_threshold,
            se_method=se_method,
            alpha=alpha
        )

        return {
            'att': result.att,
            'se': result.se,
            'ci_lower': result.ci_lower,
            'ci_upper': result.ci_upper,
            't_stat': result.t_stat,
            'pvalue': result.pvalue,
            'nobs': result.n_treated + result.n_control,
            'df_resid': result.n_treated + result.n_control - len(controls) - 1,
            'df_inference': result.n_treated + result.n_control - 2,
            'estimator': 'ipwra',
        }

    except Exception as e:
        warnings.warn(
            f"IPWRA estimation failed: {str(e)}. Returning NaN.",
            UserWarning
        )
        return {
            'att': np.nan,
            'se': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            't_stat': np.nan,
            'pvalue': np.nan,
            'nobs': len(data),
            'df_resid': np.nan,
            'df_inference': np.nan,
            'estimator': 'ipwra',
            'error': str(e),
        }


def _estimate_single_effect_psm(
    data: pd.DataFrame,
    y: str,
    d: str,
    propensity_controls: list[str],
    n_neighbors: int = 1,
    with_replacement: bool = True,
    caliper: float | None = None,
    trim_threshold: float = 0.01,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Estimate ATT using propensity score matching.

    Internal wrapper that calls the PSM estimator and returns results
    in a standardized dictionary format consistent with ``run_ols_regression``.

    Parameters
    ----------
    data : pd.DataFrame
        Cross-sectional sample data with one row per unit.
    y : str
        Name of the transformed outcome variable column.
    d : str
        Name of the treatment indicator column.
    propensity_controls : list of str
        Control variables for the propensity score model.
    n_neighbors : int, default=1
        Number of nearest neighbors for matching each treated unit.
    with_replacement : bool, default=True
        Whether control units can be matched to multiple treated units.
    caliper : float, optional
        Maximum propensity score distance for valid matches.
    trim_threshold : float, default=0.01
        Threshold for trimming extreme propensity scores.
    alpha : float, default=0.05
        Significance level for confidence interval construction.

    Returns
    -------
    dict
        Estimation results with keys: 'att', 'se', 'ci_lower', 'ci_upper',
        't_stat', 'pvalue', 'nobs', 'df_resid', 'df_inference', 'estimator',
        'n_matched', 'n_dropped'. Returns NaN values if estimation fails.
    """
    try:
        result: PSMResult = estimate_psm(
            data=data,
            y=y,
            d=d,
            propensity_controls=propensity_controls,
            n_neighbors=n_neighbors,
            with_replacement=with_replacement,
            caliper=caliper,
            trim_threshold=trim_threshold,
            alpha=alpha
        )

        return {
            'att': result.att,
            'se': result.se,
            'ci_lower': result.ci_lower,
            'ci_upper': result.ci_upper,
            't_stat': result.t_stat,
            'pvalue': result.pvalue,
            'nobs': result.n_treated + result.n_matched,
            'df_resid': result.n_treated + result.n_matched - 1,
            'df_inference': result.n_treated + result.n_matched - 2,
            'estimator': 'psm',
            'n_matched': result.n_matched,
            'n_dropped': result.n_dropped,
        }

    except Exception as e:
        warnings.warn(
            f"PSM estimation failed: {str(e)}. Returning NaN.",
            UserWarning
        )
        return {
            'att': np.nan,
            'se': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            't_stat': np.nan,
            'pvalue': np.nan,
            'nobs': len(data),
            'df_resid': np.nan,
            'df_inference': np.nan,
            'estimator': 'psm',
            'error': str(e),
        }

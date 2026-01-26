"""
Cohort-time treatment effect estimation for staggered adoption designs.

This module implements cohort-time-specific ATT estimation for staggered
difference-in-differences designs. Each (cohort, period) pair receives
its own average treatment effect on the treated (ATT) estimate through
cross-sectional regression on transformed outcome data.

The estimation approach proceeds as follows:

1. Transform outcome data by removing pre-treatment unit-specific patterns
   (demeaning or detrending) for each treatment cohort.
2. For each treated cohort g in each post-treatment period r, restrict
   the sample to cohort g units plus valid control units.
3. Run cross-sectional regression of transformed outcomes on a treatment
   indicator to estimate the ATT.

Valid control units for cohort g at time r include never-treated units and
units first treated in periods after r. This rolling approach efficiently
uses not-yet-treated observations while maintaining identification under
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

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats as stats

from .control_groups import (
    ControlGroupStrategy,
    get_valid_control_units,
    identify_never_treated_units,
)
from .transformations import get_cohorts, get_valid_periods_for_cohort
from .estimators import estimate_ipwra, estimate_psm, IPWRAResult, PSMResult


@dataclass
class CohortTimeEffect:
    """
    Container for a single cohort-time treatment effect estimate.

    Stores the average treatment effect on the treated (ATT) for a specific
    treatment cohort g at calendar time r, along with inference statistics.

    Attributes
    ----------
    cohort : int
        Treatment cohort identifier (first treatment period g).
    period : int
        Calendar time period r.
    event_time : int
        Event time relative to treatment onset (e = r - g). Zero indicates
        the instantaneous effect; positive values indicate dynamic effects.
    att : float
        Estimated average treatment effect on the treated.
    se : float
        Standard error of the ATT estimate.
    ci_lower : float
        Lower bound of the confidence interval.
    ci_upper : float
        Upper bound of the confidence interval.
    t_stat : float
        t-statistic (att / se).
    pvalue : float
        Two-sided p-value from t-distribution.
    n_treated : int
        Number of treated units in the estimation sample.
    n_control : int
        Number of control units in the estimation sample.
    n_total : int
        Total sample size (n_treated + n_control).
    df_resid : int
        Residual degrees of freedom from the regression.
    df_inference : int
        Degrees of freedom used for inference.
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


def run_ols_regression(
    data: pd.DataFrame,
    y: str,
    d: str,
    controls: Optional[List[str]] = None,
    vce: Optional[str] = None,
    cluster_var: Optional[str] = None,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Estimate the ATT via OLS regression on cross-sectional data.

    Regresses the transformed outcome on a constant and treatment indicator,
    optionally including control variables with treatment-control interactions.
    When controls are included and sample sizes permit, the regression includes
    interactions of controls with the treatment indicator centered at the
    treated group mean.

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
    vce : str, optional
        Variance estimation type. Options are None (homoskedastic), 'hc3'
        (heteroskedasticity-robust), or 'cluster' (cluster-robust).
    cluster_var : str, optional
        Name of the cluster variable column. Required when vce='cluster'.
    alpha : float, default=0.05
        Significance level for confidence interval construction.

    Returns
    -------
    dict
        Dictionary containing estimation results with keys:
        'att' (point estimate), 'se' (standard error), 'ci_lower' and
        'ci_upper' (confidence interval bounds), 't_stat' (t-statistic),
        'pvalue' (two-sided p-value), 'nobs' (number of observations),
        'df_resid' (residual degrees of freedom), 'df_inference'
        (degrees of freedom for inference).

    Raises
    ------
    ValueError
        If required columns are missing, sample size is insufficient,
        or the design matrix is singular.
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
    
    # Extract outcome and treatment vectors
    y_vals = data_clean[y].values.astype(float)
    d_vals = data_clean[d].values.astype(float)
    
    if controls is not None and len(controls) > 0:
        missing_controls = [c for c in controls if c not in data_clean.columns]
        if missing_controls:
            raise ValueError(f"Control variables not found: {missing_controls}")
        
        # Exclude observations with missing control values
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
            # Center controls at treated-group mean for interpretability
            X_controls = data_clean[controls].values.astype(float)
            X_mean_treated = X_controls[treated_mask].mean(axis=0)
            X_centered = X_controls - X_mean_treated
            X_interactions = d_vals.reshape(-1, 1) * X_centered
            
            # Full design: intercept, treatment, controls, interactions
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
        # Simple model: intercept and treatment indicator only
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
    
    # Variance estimation based on specified method
    if vce is None:
        # Classical homoskedastic variance
        var_beta = sigma2 * XtX_inv
        df_inference = df_resid
        
    elif vce == 'hc3':
        # HC3 heteroskedasticity-robust variance with leverage adjustment
        H = X @ XtX_inv @ X.T
        h_ii = np.diag(H)
        
        # Bound leverage to prevent numerical instability
        h_ii = np.clip(h_ii, 0, 0.9999)
        omega_diag = (residuals ** 2) / ((1 - h_ii) ** 2)
        
        meat = X.T @ np.diag(omega_diag) @ X
        var_beta = XtX_inv @ meat @ XtX_inv
        df_inference = df_resid
        
    elif vce == 'cluster':
        if cluster_var is None:
            raise ValueError("cluster_var required when vce='cluster'")
        if cluster_var not in data_clean.columns:
            raise ValueError(f"Cluster variable '{cluster_var}' not in data")
        
        cluster_ids = data_clean[cluster_var].values
        unique_clusters = np.unique(cluster_ids)
        G = len(unique_clusters)
        
        if G < 2:
            raise ValueError(f"Need at least 2 clusters, got {G}")
        
        # Cluster-robust variance with finite-sample correction
        meat = np.zeros((X.shape[1], X.shape[1]))
        for cluster in unique_clusters:
            mask = cluster_ids == cluster
            X_g = X[mask]
            e_g = residuals[mask]
            score_g = X_g.T @ e_g
            meat += np.outer(score_g, score_g)
        
        correction = (G / (G - 1)) * ((n - 1) / (n - X.shape[1]))
        var_beta = correction * XtX_inv @ meat @ XtX_inv
        df_inference = G - 1
        
    else:
        raise ValueError(f"Unknown vce type: {vce}. Use None, 'hc3', or 'cluster'.")
    
    # Treatment coefficient is at index 1 in the design matrix
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
    controls: Optional[List[str]] = None,
    vce: Optional[str] = None,
    cluster_var: Optional[str] = None,
    control_strategy: str = 'not_yet_treated',
    never_treated_values: Optional[List] = None,
    min_obs: int = 3,
    min_treated: int = 1,
    min_control: int = 1,
    alpha: float = 0.05,
    estimator: str = 'ra',
    transform_type: str = 'demean',
    propensity_controls: Optional[List[str]] = None,
    trim_threshold: float = 0.01,
    se_method: str = 'analytical',
    n_neighbors: int = 1,
    with_replacement: bool = True,
    caliper: Optional[float] = None,
    return_diagnostics: bool = False,
    match_order: Optional[str] = None,
) -> List[CohortTimeEffect]:
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
        treated after the current period; 'auto' selects based on
        data availability.
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
    
    prefix = 'ydot' if transform_type == 'demean' else 'ycheck'
    
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
            
            # Map unit-level mask to observation-level mask
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
                df_resid=int(est_result.get('df_resid', 0) or 0),
                df_inference=int(est_result.get('df_inference', 0) or 0),
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


def results_to_dataframe(results: List[CohortTimeEffect]) -> pd.DataFrame:
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
    controls: List[str],
    propensity_controls: List[str],
    trim_threshold: float = 0.01,
    se_method: str = 'analytical',
    alpha: float = 0.05,
) -> Dict[str, Any]:
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
    propensity_controls: List[str],
    n_neighbors: int = 1,
    with_replacement: bool = True,
    caliper: Optional[float] = None,
    trim_threshold: float = 0.01,
    alpha: float = 0.05,
) -> Dict[str, Any]:
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

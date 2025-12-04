"""
Staggered Effect Estimation Module

Implements (g,r)-specific treatment effect estimation for staggered DiD
based on Lee and Wooldridge (2023, 2025) Procedure 4.1.

Key concepts:
- Each (cohort, period) pair has its own ATT estimate τ_{gr}
- Estimation is cross-sectional regression on transformed data
- Sample restricted to: treatment cohort g + valid controls (gvar > r or NT)

Reference:
    Lee & Wooldridge (2023) Section 4, Procedure 4.1
    Lee & Wooldridge (2025) Section 7.1, Equation (7.8)
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
    Single (cohort, period) treatment effect estimate.
    
    Represents τ_{gr}, the ATT for cohort g at calendar time r.
    
    Attributes
    ----------
    cohort : int
        Treatment cohort (first treatment period g)
    period : int
        Calendar time (r)
    event_time : int
        Event time (e = r - g)
        - e = 0: instantaneous effect
        - e > 0: dynamic effects
    att : float
        Estimated ATT τ̂_{gr}
    se : float
        Standard error
    ci_lower : float
        95% confidence interval lower bound
    ci_upper : float
        95% confidence interval upper bound
    t_stat : float
        t-statistic = att / se
    pvalue : float
        Two-sided p-value
    n_treated : int
        Number of treated units in sample
    n_control : int
        Number of control units in sample
    n_total : int
        Total sample size
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
    Run OLS regression to estimate ATT.
    
    Implements the cross-sectional regression:
    Ŷ_{ir} on 1, D_i [, X_i, D_i·(X_i - X̄₁)]
    
    Parameters
    ----------
    data : pd.DataFrame
        Cross-sectional data (one row per unit)
    y : str
        Dependent variable (transformed Y)
    d : str
        Treatment indicator column name
    controls : list of str, optional
        Control variable names
    vce : str, optional
        Variance type: None (homoskedastic), 'hc3', 'cluster'
    cluster_var : str, optional
        Cluster variable (required if vce='cluster')
    alpha : float, default=0.05
        Significance level for confidence intervals
        
    Returns
    -------
    dict
        {'att': float, 'se': float, 'ci_lower': float, 'ci_upper': float,
         't_stat': float, 'pvalue': float, 'nobs': int, 'df_resid': int}
    """
    # Validate inputs
    if y not in data.columns:
        raise ValueError(f"Dependent variable '{y}' not in data")
    if d not in data.columns:
        raise ValueError(f"Treatment indicator '{d}' not in data")
    
    # Drop rows with missing y or d
    valid_mask = data[y].notna() & data[d].notna()
    data_clean = data[valid_mask].copy()
    
    n = len(data_clean)
    if n < 2:
        raise ValueError(f"Insufficient observations: n={n}, need at least 2")
    
    # Build design matrix
    y_vals = data_clean[y].values.astype(float)
    d_vals = data_clean[d].values.astype(float)
    
    if controls is not None and len(controls) > 0:
        # Check controls exist
        missing_controls = [c for c in controls if c not in data_clean.columns]
        if missing_controls:
            raise ValueError(f"Control variables not found: {missing_controls}")
        
        # Drop rows with missing controls
        control_valid = data_clean[controls].notna().all(axis=1)
        data_clean = data_clean[control_valid].copy()
        y_vals = data_clean[y].values.astype(float)
        d_vals = data_clean[d].values.astype(float)
        n = len(data_clean)
        
        if n < 2:
            raise ValueError(f"Insufficient observations after dropping missing controls: n={n}")
        
        # Compute treated-group mean of controls
        treated_mask = d_vals == 1
        n_treated = treated_mask.sum()
        n_control = (~treated_mask).sum()
        
        K = len(controls)
        if n_treated > K + 1 and n_control > K + 1:
            # Include controls with interactions
            X_controls = data_clean[controls].values.astype(float)
            X_mean_treated = X_controls[treated_mask].mean(axis=0)
            X_centered = X_controls - X_mean_treated
            X_interactions = d_vals.reshape(-1, 1) * X_centered
            
            # Design matrix: [1, D, X, D*(X - X̄₁)]
            X = np.column_stack([
                np.ones(n),
                d_vals,
                X_controls,
                X_interactions
            ])
        else:
            # Not enough observations to include controls
            warnings.warn(
                f"Controls not included: N_treated={n_treated}, N_control={n_control}, K={K}. "
                f"Need N_treated > K+1 and N_control > K+1.",
                UserWarning
            )
            X = np.column_stack([np.ones(n), d_vals])
    else:
        # No controls: X = [1, D]
        X = np.column_stack([np.ones(n), d_vals])
    
    # OLS estimation: β = (X'X)^{-1} X'y
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        raise ValueError("Design matrix is singular, cannot estimate")
    
    beta = XtX_inv @ (X.T @ y_vals)
    
    # Residuals and sigma^2
    y_hat = X @ beta
    residuals = y_vals - y_hat
    df_resid = n - X.shape[1]
    
    if df_resid <= 0:
        # No degrees of freedom - can estimate ATT but not SE
        # ATT is coefficient on D (index 1)
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
    
    # Compute variance of beta
    if vce is None:
        # Homoskedastic: Var(β) = σ² (X'X)^{-1}
        var_beta = sigma2 * XtX_inv
        df_inference = df_resid
        
    elif vce == 'hc3':
        # HC3 robust: Var(β) = (X'X)^{-1} X' Ω X (X'X)^{-1}
        # where Ω_ii = e_i² / (1 - h_ii)²
        H = X @ XtX_inv @ X.T  # Hat matrix
        h_ii = np.diag(H)
        
        # Avoid division by zero
        h_ii = np.clip(h_ii, 0, 0.9999)
        omega_diag = (residuals ** 2) / ((1 - h_ii) ** 2)
        
        # Meat matrix
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
        
        # Cluster-robust variance
        # Var(β) = (X'X)^{-1} (Σ_g X'_g e_g e'_g X_g) (X'X)^{-1} * G/(G-1) * (n-1)/(n-k)
        meat = np.zeros((X.shape[1], X.shape[1]))
        for cluster in unique_clusters:
            mask = cluster_ids == cluster
            X_g = X[mask]
            e_g = residuals[mask]
            score_g = X_g.T @ e_g
            meat += np.outer(score_g, score_g)
        
        # Small sample correction
        correction = (G / (G - 1)) * ((n - 1) / (n - X.shape[1]))
        var_beta = correction * XtX_inv @ meat @ XtX_inv
        df_inference = G - 1
        
    else:
        raise ValueError(f"Unknown vce type: {vce}. Use None, 'hc3', or 'cluster'.")
    
    # ATT is the coefficient on D (index 1)
    att = beta[1]
    se = np.sqrt(var_beta[1, 1])
    
    # t-statistic and p-value
    if se > 0:
        t_stat = att / se
        pvalue = 2 * (1 - stats.t.cdf(abs(t_stat), df_inference))
    else:
        t_stat = np.nan
        pvalue = np.nan
    
    # Confidence interval
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
    transform_type: str = 'demean',  # 'demean' or 'detrend'
    # IPWRA专用参数
    propensity_controls: Optional[List[str]] = None,
    trim_threshold: float = 0.01,
    se_method: str = 'analytical',
    # PSM专用参数
    n_neighbors: int = 1,
    with_replacement: bool = True,
    caliper: Optional[float] = None,
) -> List[CohortTimeEffect]:
    """
    Estimate treatment effects for all (cohort, period) pairs.
    
    Implements Lee & Wooldridge (2023) Procedure 4.1.
    
    Data Flow
    ---------
    1. Input: data_transformed (long panel with ydot_g{g}_r{r} columns)
    2. For each (g, r) pair:
       a. Extract period r cross-section (tvar == r)
       b. Apply control group mask (from get_valid_control_units)
       c. Build estimation sample (treatment + controls)
       d. Run OLS: ydot_g{g}_r{r} ~ 1 + D_g
       e. Extract ATT coefficient and SE
    3. Output: List[CohortTimeEffect]
    
    Parameters
    ----------
    data_transformed : pd.DataFrame
        Panel data with transformation columns from transform_staggered_demean/detrend.
        Must contain: gvar, ivar, tvar, ydot_g{g}_r{r} columns.
    gvar : str
        Cohort variable column name
    ivar : str
        Unit identifier column name
    tvar : str
        Time variable column name
    controls : list of str, optional
        Control variable names (time-invariant)
    vce : str, optional
        Standard error type: None, 'hc3', 'cluster'
    cluster_var : str, optional
        Cluster variable (required if vce='cluster')
    control_strategy : str, default='not_yet_treated'
        Control group strategy: 'never_treated', 'not_yet_treated', 'auto'
    never_treated_values : list, optional
        Values indicating never treated. Default: [0, np.inf, NaN]
    min_obs : int, default=3
        Minimum total sample size for estimation
    min_treated : int, default=1
        Minimum treated units required
    min_control : int, default=1
        Minimum control units required
    alpha : float, default=0.05
        Significance level for CIs
    estimator : str, default='ra'
        Estimation method:
        - 'ra': Regression Adjustment (OLS)
        - 'ipwra': Inverse Probability Weighted Regression Adjustment
        - 'psm': Propensity Score Matching
    propensity_controls : list of str, optional
        Controls for propensity score model (IPWRA/PSM). If None, uses `controls`.
    trim_threshold : float, default=0.01
        Propensity score trimming threshold (IPWRA/PSM).
    se_method : str, default='analytical'
        SE method for IPWRA: 'analytical' or 'bootstrap'.
    n_neighbors : int, default=1
        Number of matches per treated unit (PSM only).
    with_replacement : bool, default=True
        Whether to match with replacement (PSM only).
    caliper : float, optional
        Caliper for matching (PSM only).
        
    Returns
    -------
    List[CohortTimeEffect]
        Estimation results for all valid (g, r) pairs, sorted by cohort and period.
        
    Raises
    ------
    ValueError
        Missing required columns, no valid cohorts, or invalid parameters.
        
    Examples
    --------
    >>> results = estimate_cohort_time_effects(
    ...     data_transformed=transformed_data,
    ...     gvar='gvar', ivar='sid', tvar='year',
    ...     control_strategy='not_yet_treated'
    ... )
    >>> for r in results:
    ...     print(f"τ_{{{r.cohort},{r.period}}} = {r.att:.4f}")
    """
    # ================================================================
    # Step 1: Input validation
    # ================================================================
    required_cols = [gvar, ivar, tvar]
    missing = [c for c in required_cols if c not in data_transformed.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    if vce == 'cluster' and cluster_var is None:
        raise ValueError("cluster_var required when vce='cluster'")
    
    # Parse control strategy
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
    
    # ================================================================
    # Step 2: Extract cohorts and time range
    # ================================================================
    if never_treated_values is None:
        nt_values = [0, np.inf]
    else:
        nt_values = never_treated_values
    
    cohorts = get_cohorts(data_transformed, gvar, ivar, nt_values)
    
    if len(cohorts) == 0:
        raise ValueError("No valid treatment cohorts found in data.")
    
    T_max = int(data_transformed[tvar].max())
    
    # ================================================================
    # Step 3: Estimate effects for each (g, r) pair
    # ================================================================
    results = []
    skipped_pairs = []
    
    # Determine column prefix based on transform_type (once, outside loop)
    prefix = 'ydot' if transform_type == 'demean' else 'ycheck'
    
    for g in cohorts:
        # Valid periods for cohort g: {g, g+1, ..., T_max}
        valid_periods = get_valid_periods_for_cohort(g, T_max)
        
        for r in valid_periods:
            # Transformation column name
            ydot_col = f'{prefix}_g{g}_r{r}'
            
            # Check if transformation column exists
            if ydot_col not in data_transformed.columns:
                skipped_pairs.append((g, r, 'missing_transform_column'))
                continue
            
            # --------------------------------------------------------
            # Step 3a: Extract period r cross-section
            # --------------------------------------------------------
            period_data = data_transformed[data_transformed[tvar] == r].copy()
            
            if len(period_data) == 0:
                skipped_pairs.append((g, r, 'no_data_in_period'))
                continue
            
            # --------------------------------------------------------
            # Step 3b: Get control group mask
            # --------------------------------------------------------
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
            
            # Map unit-level mask to row-level mask
            # unit_control_mask index = ivar values
            control_mask = period_data[ivar].map(unit_control_mask).fillna(False).astype(bool)
            
            # --------------------------------------------------------
            # Step 3c: Build estimation sample
            # --------------------------------------------------------
            # Treatment group: cohort g units
            treat_mask = (period_data[gvar] == g)
            
            # Sample: treatment + controls (D_ig + A_{r+1} = 1)
            sample_mask = treat_mask | control_mask
            
            n_treat = treat_mask.sum()
            n_control = control_mask.sum()
            n_total = sample_mask.sum()
            
            # Check minimum requirements
            if n_total < min_obs:
                skipped_pairs.append((g, r, f'insufficient_total: {n_total}<{min_obs}'))
                continue
            if n_treat < min_treated:
                skipped_pairs.append((g, r, f'insufficient_treated: {n_treat}<{min_treated}'))
                continue
            if n_control < min_control:
                skipped_pairs.append((g, r, f'insufficient_control: {n_control}<{min_control}'))
                continue
            
            # Extract sample data
            sample_data = period_data[sample_mask].copy()
            
            # Create treatment indicator for this cohort
            sample_data['_D_treat'] = (sample_data[gvar] == g).astype(int)
            
            # --------------------------------------------------------
            # Step 3d: Run estimation (RA, IPWRA, or PSM)
            # --------------------------------------------------------
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
            
            # --------------------------------------------------------
            # Step 3e: Store result
            # --------------------------------------------------------
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
            ))
    
    # ================================================================
    # Step 4: Report skipped pairs if any
    # ================================================================
    if skipped_pairs:
        n_skipped = len(skipped_pairs)
        n_total_pairs = sum(len(get_valid_periods_for_cohort(g, T_max)) for g in cohorts)
        warnings.warn(
            f"Skipped {n_skipped}/{n_total_pairs} (cohort, period) pairs due to "
            f"insufficient data or errors. Use verbose=True for details.",
            UserWarning
        )
    
    # Sort by cohort and period
    results.sort(key=lambda x: (x.cohort, x.period))
    
    return results


def results_to_dataframe(results: List[CohortTimeEffect]) -> pd.DataFrame:
    """
    Convert list of CohortTimeEffect to DataFrame.
    
    Parameters
    ----------
    results : List[CohortTimeEffect]
        Estimation results from estimate_cohort_time_effects
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: cohort, period, event_time, att, se, 
        ci_lower, ci_upper, t_stat, pvalue, n_treated, n_control, n_total
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


# ============================================================================
# IPWRA and PSM Wrapper Functions
# ============================================================================

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
    Wrapper for estimate_ipwra() that returns dict matching run_ols_regression output.
    
    This ensures consistent return format across all estimators.
    
    Parameters
    ----------
    data : pd.DataFrame
        Cross-sectional sample data
    y : str
        Outcome variable (transformed)
    d : str
        Treatment indicator column name
    controls : List[str]
        Outcome model controls
    propensity_controls : List[str]
        Propensity score model controls
    trim_threshold : float
        Propensity score trimming threshold
    se_method : str
        'analytical' or 'bootstrap'
    alpha : float
        Significance level
        
    Returns
    -------
    Dict
        Estimation results matching run_ols_regression format
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
        # If IPWRA fails, return NaN result with warning
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
    Wrapper for estimate_psm() that returns dict matching run_ols_regression output.
    
    Parameters
    ----------
    data : pd.DataFrame
        Cross-sectional sample data
    y : str
        Outcome variable (transformed)
    d : str
        Treatment indicator column name
    propensity_controls : List[str]
        Propensity score model controls
    n_neighbors : int
        Number of nearest neighbors for matching
    with_replacement : bool
        Whether to match with replacement
    caliper : float, optional
        Maximum propensity score distance for matching
    trim_threshold : float
        Propensity score trimming threshold
    alpha : float
        Significance level
        
    Returns
    -------
    Dict
        Estimation results matching run_ols_regression format
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

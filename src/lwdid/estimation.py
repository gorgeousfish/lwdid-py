"""
Estimation module for the Lee and Wooldridge (2025) DiD estimator.

Implements cross-sectional OLS regressions for the Lee and Wooldridge (2025)
difference-in-differences estimator.
"""

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.api as sm

from .exceptions import InvalidParameterError, InvalidVCETypeError, InsufficientDataError


def prepare_controls(
    data: pd.DataFrame,
    d: str,
    ivar: str,
    controls: List[str],
    N_treated: int,
    N_control: int,
    data_sample: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Construct centered controls and interactions following Section 2.2

    Checks N₁ > K+1 and N₀ > K+1 conditions. If satisfied, computes X̄₁
    (treated-group mean), centered controls (X - X̄₁), and interactions
    D·(X - X̄₁) for the full regression as in equation (2.18).

    Parameters
    ----------
    data : pd.DataFrame
        Transformed panel data (N×T rows).
    d : str
        Treatment indicator column name.
    ivar : str
        Unit identifier column name.
    controls : list of str
        Control variable names (time-invariant, numeric).
    N_treated : int
        Number of treated units in regression sample.
    N_control : int
        Number of control units in regression sample.
    data_sample : pd.DataFrame, optional
        Regression sample for computing X̄₁ (firstpost cross-section).
    
    Returns
    -------
    dict
        'include' (bool), 'X_centered' (DataFrame), 'interactions' (DataFrame),
        'X_mean_treated' (dict), 'RHS_varnames' (list).
    """
    K_ctrl = len(controls)
    nk = K_ctrl + 1
    
    if N_treated > nk and N_control > nk:
        # Compute treated-group means X̄₁ = (1/N₁) Σ_{i:D_i=1} X_i
        # Use unit-weighted mean (not row-weighted) since X is time-invariant
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
        
        # Center controls: X - X̄₁
        X_centered = pd.DataFrame({
            f'{x}_c': data[x] - X_mean_treated[x]
            for x in controls
        })
        
        # Construct interactions: D·(X - X̄₁)
        interactions = pd.DataFrame({
            f'd_{x}_c': data[d] * X_centered[f'{x}_c']
            for x in controls
        })
        
        RHS_varnames = controls + [f'd_{x}_c' for x in controls]
        
        return {
            'include': True,
            'X_centered': X_centered,  # DataFrame; columns: '{x}_c'
            'interactions': interactions,  # DataFrame; columns: 'd_{x}_c'
            'X_mean_treated': X_mean_treated,  # dict: {variable_name: X̄₁ value}
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
    controls: Optional[list],
    vce: Optional[str],
    cluster_var: Optional[str],
    sample_filter: pd.Series,
) -> Dict[str, Any]:
    """
    Estimate ATT via cross-sectional OLS regression
    
    Runs OLS on the firstpost cross-section using transformed outcome ydot_postavg
    as dependent variable. Implements equation (2.13) or (2.18) with controls.
    Supports homoskedastic, HC1/HC3, and cluster-robust standard errors.
    
    Parameters
    ----------
    data : pd.DataFrame
        Transformed panel data with ydot_postavg column.
    y_transformed : str
        Dependent variable (typically 'ydot_postavg').
    d : str
        Treatment indicator (typically 'd_').
    ivar : str
        Unit identifier.
    controls : list of str or None
        Time-invariant controls. Included if N₁ > K+1 and N₀ > K+1.
    vce : {None, 'robust', 'hc1', 'hc3', 'cluster'}, optional
        Variance estimator (None=homoskedastic, 'hc3'=HC3, 'cluster'=clustered).
    cluster_var : str or None
        Clustering variable (required for vce='cluster').
    sample_filter : pd.Series
        Boolean indicating regression sample (firstpost).
    
    Returns
    -------
    dict
        'att', 'se_att', 't_stat', 'pvalue', 'ci_lower', 'ci_upper', 'params',
        'bse', 'vcov', 'resid', 'nobs', 'df_resid', 'df_inference', 'vce_type',
        'cluster_var', 'n_clusters', 'controls_used', 'controls', 'controls_spec',
        'n_treated_sample', 'n_control_sample'.
    
    Notes
    -----
    Confidence intervals use t-distribution with df = df_resid (non-clustered)
    or G-1 (clustered). See equations (2.10), (2.19) for exact inference under
    normality.
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
    
    if vce is None:
        results = model.fit()
    elif vce == 'hc3':
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
    elif vce == 'robust' or vce == 'hc1':
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
    elif vce == 'cluster':
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
                f"Cameron & Miller (2015) recommend ≥20-30 clusters.",
                UserWarning,
                stacklevel=3
            )

        if data_sample[cluster_var].isna().any():
            raise InvalidParameterError(
                f"Cluster variable '{cluster_var}' contains missing values."
            )

        # Verify cluster is nested within units (each unit belongs to one cluster)
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
            f"Must be one of: None, 'robust', 'hc1', 'hc3', 'cluster'"
        )
    
    params_vals = results.params.values if hasattr(results.params, 'values') else results.params
    bse_vals = results.bse.values if hasattr(results.bse, 'values') else results.bse
    
    if len(params_vals) < 2:
        raise InsufficientDataError(
            f"Insufficient variation in treatment indicator (d_) in firstpost cross-section. "
            f"Regression model has only {len(params_vals)} parameter(s) (expected >= 2). "
            f"This typically occurs when all units in the firstpost sample have the same "
            f"treatment status (all d_=0 or all d_=1). "
            f"Please check that your data contains both treated and control units in the "
            f"first post-treatment period."
        )
    
    att = params_vals[1]
    se_att = bse_vals[1]

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

    # Degrees of freedom: G-1 for clustered, df_resid otherwise
    if vce == 'cluster':
        df = G - 1
    else:
        df = results.df_resid

    t_stat = att / se_att

    pvalue = 2 * scipy.stats.t.cdf(-abs(t_stat), df)

    # 95% CI using t critical value (exact under normality, eq 2.10, 2.19)
    t_crit = scipy.stats.t.ppf(0.975, df)
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
    controls_spec: Optional[Dict],
    vce: Optional[str],
    cluster_var: Optional[str],
    period_labels: dict,
) -> pd.DataFrame:
    """
    Estimate period-specific ATTs via independent cross-sectional regressions
    
    Runs separate OLS for each t ∈ {tpost1, ..., Tmax} using residualized
    outcome ydot, implementing equation (2.20) or its analog with controls.
    
    Parameters
    ----------
    data : pd.DataFrame
        Transformed panel data with 'ydot' column.
    ydot : str
        Residualized outcome (typically 'ydot').
    d : str
        Treatment indicator.
    tindex : str
        Time index.
    tpost1 : int
        First post-treatment period.
    Tmax : int
        Last period.
    controls_spec : dict or None
        Control specification from prepare_controls().
    vce : {None, 'robust', 'hc1', 'hc3', 'cluster'}, optional
        Variance estimator.
    cluster_var : str or None
        Clustering variable.
    period_labels : dict
        Mapping tindex → period label.
    
    Returns
    -------
    pd.DataFrame
        Columns: 'period', 'tindex', 'beta', 'se', 'ci_lower', 'ci_upper',
        'tstat', 'pval', 'N'.
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
        
        try:
            if vce is None:
                model_t = sm.OLS(data_t[ydot], X_t, missing='drop').fit()
            elif vce == 'hc3':
                model_t = sm.OLS(data_t[ydot], X_t, missing='drop').fit(cov_type='HC3')
            elif vce == 'robust' or vce == 'hc1':
                model_t = sm.OLS(data_t[ydot], X_t, missing='drop').fit(cov_type='HC1')
            elif vce == 'cluster':
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
                    f"Must be one of: None, 'robust', 'hc1', 'hc3', 'cluster'"
                )
            
            params_vals = model_t.params.values if hasattr(model_t.params, 'values') else model_t.params
            bse_vals = model_t.bse.values if hasattr(model_t.bse, 'values') else model_t.bse

            beta_t = params_vals[1]
            se_t = bse_vals[1]

            if se_t == 0 or np.isnan(se_t) or np.isinf(se_t):
                import warnings
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
                if vce == 'cluster':
                    n_clusters_t = len(np.unique(cluster_groups_t))
                    df_t = n_clusters_t - 1
                else:
                    df_t = model_t.df_resid

                t_stat = beta_t / se_t
                p_val = 2 * scipy.stats.t.cdf(-abs(t_stat), df_t)
                N_t = int(model_t.nobs)

                t_crit_t = scipy.stats.t.ppf(0.975, df_t)
                ci_lower = beta_t - t_crit_t * se_t
                ci_upper = beta_t + t_crit_t * se_t
        except Exception as e:
            import warnings
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

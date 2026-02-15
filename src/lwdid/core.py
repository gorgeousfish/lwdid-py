"""
Core estimation interface for difference-in-differences with panel data.

This module provides the main entry point for LWDID estimation, supporting
three methodological scenarios based on the rolling transformation approach:

1. **Small-sample common timing**: Exact t-based inference under classical
   linear model (CLM) assumptions when the number of cross-sectional units
   is small.

2. **Large-sample common timing**: Asymptotic inference with
   heteroskedasticity-robust standard errors for moderate to large samples.

3. **Staggered adoption**: Cohort-time specific effect estimation with
   flexible control group strategies (never-treated or not-yet-treated)
   for settings where treatment timing varies.

The method applies unit-specific time-series transformations that remove
pre-treatment patterns, converting the panel DiD problem into a cross-sectional
treatment effects problem. Under no anticipation and parallel trends
assumptions, standard estimators can be applied to the transformed outcomes.

Notes
-----
Four transformation methods are available:

- **Demeaning** ('demean'): Subtracts the unit-specific pre-treatment mean.
  Requires at least one pre-treatment period.
- **Detrending** ('detrend'): Removes unit-specific linear time trends.
  Requires at least two pre-treatment periods.
- **Seasonal demeaning** ('demeanq'): Removes unit-specific mean with seasonal
  fixed effects. Requires sufficient pre-treatment observations for seasonal
  pattern estimation.
- **Seasonal detrending** ('detrendq'): Removes unit-specific linear trends
  with seasonal effects. Requires at least two pre-treatment periods plus
  adequate seasonal coverage.

Four estimation methods are supported:

- **RA**: Regression adjustment via OLS on transformed outcomes.
- **IPW**: Inverse probability weighting using propensity scores.
- **IPWRA**: Doubly robust combining IPW with regression adjustment.
- **PSM**: Propensity score matching with nearest-neighbor matching.

For staggered adoption, transformations are applied separately for each
treatment cohort using cohort-specific pre-treatment periods.
"""

from __future__ import annotations

import logging
import math
import random
import warnings

import numpy as np
import pandas as pd
import scipy.stats

from . import estimation, transformations, validation
from .exceptions import (
    InsufficientDataError,
    InvalidParameterError,
    NoNeverTreatedError,
    RandomizationError,
)
from .randomization import randomization_inference
from .results import LWDIDResults
from .warnings_categories import (
    DataWarning,
    NumericalWarning,
    SmallSampleWarning,
)
from .staggered.estimators import (
    IPWResult,
    IPWRAResult,
    PSMResult,
    estimate_ipw,
    estimate_ipwra,
    estimate_psm,
)
from .validation import is_never_treated, validate_staggered_data

logger = logging.getLogger('lwdid')


def _generate_ri_seed() -> int:
    """
    Generate a random seed for randomization inference.

    Combines two independent uniform random draws to produce a unique seed,
    reducing the probability of seed collisions across repeated calls.

    Returns
    -------
    int
        Random seed in the range [1, 1001000].
    """
    return math.ceil(random.random() * 1e6 + 1000 * random.random())


def _validate_psm_params(
    n_neighbors: int,
    caliper: float | None,
    with_replacement: bool,
    match_order: str = 'data',
) -> None:
    """
    Validate propensity score matching parameters.

    Checks type and value constraints for PSM-specific parameters before
    estimation proceeds.

    Parameters
    ----------
    n_neighbors : int
        Number of nearest neighbors for matching. Must be >= 1.
    caliper : float or None
        Maximum propensity score distance for valid matches. Must be positive
        and finite if specified.
    with_replacement : bool
        Whether control units can be matched to multiple treated units.
    match_order : {'data', 'random', 'largest', 'smallest'}, default='data'
        Order in which treated units are processed for matching without
        replacement.

    Raises
    ------
    TypeError
        If parameter types do not match expected types.
    ValueError
        If parameter values are outside valid ranges.
    """
    if not isinstance(n_neighbors, (int, np.integer)):
        raise TypeError(
            f"n_neighbors must be an integer, got {type(n_neighbors).__name__}"
        )
    if n_neighbors < 1:
        raise ValueError(
            f"n_neighbors must be >= 1, got {n_neighbors}"
        )

    if caliper is not None:
        if not isinstance(caliper, (int, float, np.number)):
            raise TypeError(
                f"caliper must be numeric or None, got {type(caliper).__name__}"
            )
        # Explicit finiteness check required because np.nan <= 0 returns False.
        if not np.isfinite(caliper) or caliper <= 0:
            raise ValueError(
                f"caliper must be a finite positive number, got {caliper}"
            )

    if not isinstance(with_replacement, bool):
        raise TypeError(
            f"with_replacement must be bool, got {type(with_replacement).__name__}"
        )

    valid_match_orders = {'data', 'random', 'largest', 'smallest'}
    if not isinstance(match_order, str):
        raise TypeError(
            f"match_order must be str, got {type(match_order).__name__}"
        )
    if match_order not in valid_match_orders:
        raise ValueError(
            f"match_order must be one of {valid_match_orders}, got '{match_order}'"
        )


def _convert_ipw_result_to_dict(
    ipw_result: IPWResult,
    alpha: float,
    vce: str | None,
    cluster_var: str | None,
    controls: list[str] | None,
    ps_controls: list[str] | None = None,
) -> dict:
    """
    Convert IPWResult to the standard results dictionary format.

    Transforms the IPW estimator output into the unified dictionary structure
    expected by LWDIDResults.

    Parameters
    ----------
    ipw_result : IPWResult
        Result object from estimate_ipw().
    alpha : float
        Significance level for confidence intervals.
    vce : str or None
        Variance estimator type for metadata storage.
    cluster_var : str or None
        Cluster variable name for metadata storage.
    controls : list of str or None
        Control variable names for outcome model.
    ps_controls : list of str or None
        Control variable names for propensity score model.

    Returns
    -------
    dict
        Results dictionary compatible with LWDIDResults constructor.
    """
    # IPW df uses n - 2: intercept + treatment indicator.
    # PS model parameters do not enter the ATT variance formula.
    df_val = ipw_result.n_treated + ipw_result.n_control - 2

    return {
        'att': ipw_result.att,
        'se_att': ipw_result.se,
        't_stat': ipw_result.t_stat,
        'pvalue': ipw_result.pvalue,
        'ci_lower': ipw_result.ci_lower,
        'ci_upper': ipw_result.ci_upper,
        'nobs': ipw_result.n_treated + ipw_result.n_control,
        'df_resid': df_val,
        'df_inference': df_val,
        'vce_type': vce if vce is not None else 'ipw',
        'cluster_var': cluster_var,
        'n_clusters': None,
        'controls_used': controls is not None and len(controls) > 0,
        'controls': controls if controls else [],
        'controls_spec': None,
        'n_treated_sample': ipw_result.n_treated,
        'n_control_sample': ipw_result.n_control,
        'params': None,
        'bse': None,
        'vcov': None,
        'resid': None,
        'diagnostics': ipw_result.diagnostics if hasattr(ipw_result, 'diagnostics') else None,
        'weights_cv': ipw_result.weights_cv,
        'propensity_scores': ipw_result.propensity_scores,
        'estimator': 'ipw',
    }


def _convert_ipwra_result_to_dict(
    ipwra_result: IPWRAResult,
    alpha: float,
    vce: str | None,
    cluster_var: str | None,
    controls: list[str] | None,
) -> dict:
    """
    Convert IPWRAResult to the standard results dictionary format.

    Transforms the doubly robust estimator output into the unified dictionary
    structure expected by LWDIDResults.

    Parameters
    ----------
    ipwra_result : IPWRAResult
        Result object from estimate_ipwra().
    alpha : float
        Significance level for confidence intervals.
    vce : str or None
        Variance estimator type for metadata storage.
    cluster_var : str or None
        Cluster variable name for metadata storage.
    controls : list of str or None
        Control variable names for outcome model.

    Returns
    -------
    dict
        Results dictionary compatible with LWDIDResults constructor.
    """
    # CV requires at least 2 observations for meaningful variance.
    weights = getattr(ipwra_result, 'weights', None)
    if weights is not None and len(weights) > 1:
        weights_mean = np.mean(weights)
        weights_std = np.std(weights, ddof=1)
        weights_cv = weights_std / weights_mean if weights_mean > 0 else np.nan
    elif weights is not None and len(weights) == 1:
        # Single observation has zero variance by definition.
        weights_cv = 0.0
    else:
        weights_cv = np.nan

    # IPWRA df: n - k where k = intercept + treatment + outcome model controls.
    n_controls = len(controls) if controls else 0
    n_params = 2 + n_controls
    df_val = ipwra_result.n_treated + ipwra_result.n_control - n_params

    return {
        'att': ipwra_result.att,
        'se_att': ipwra_result.se,
        't_stat': ipwra_result.t_stat,
        'pvalue': ipwra_result.pvalue,
        'ci_lower': ipwra_result.ci_lower,
        'ci_upper': ipwra_result.ci_upper,
        'nobs': ipwra_result.n_treated + ipwra_result.n_control,
        'df_resid': df_val,
        'df_inference': df_val,
        'vce_type': vce if vce is not None else 'ipwra',
        'cluster_var': cluster_var,
        'n_clusters': None,
        'controls_used': controls is not None and len(controls) > 0,
        'controls': controls if controls else [],
        'controls_spec': None,
        'n_treated_sample': ipwra_result.n_treated,
        'n_control_sample': ipwra_result.n_control,
        'params': None,
        'bse': None,
        'vcov': None,
        'resid': None,
        'diagnostics': ipwra_result.diagnostics if hasattr(ipwra_result, 'diagnostics') else None,
        'weights_cv': weights_cv,
        'propensity_scores': ipwra_result.propensity_scores,
        'estimator': 'ipwra',
    }


def _convert_psm_result_to_dict(
    psm_result: PSMResult,
    alpha: float,
    vce: str | None,
    cluster_var: str | None,
    controls: list[str] | None,
) -> dict:
    """
    Convert PSMResult to the standard results dictionary format.

    Transforms the propensity score matching output into the unified dictionary
    structure expected by LWDIDResults.

    Parameters
    ----------
    psm_result : PSMResult
        Result object from estimate_psm().
    alpha : float
        Significance level for confidence intervals.
    vce : str or None
        Variance estimator type for metadata storage.
    cluster_var : str or None
        Cluster variable name for metadata storage.
    controls : list of str or None
        Control variable names for propensity score model.

    Returns
    -------
    dict
        Results dictionary compatible with LWDIDResults constructor.
    """
    # Compute match rate, handling None n_matched by falling back to n_treated.
    n_matched_attr = getattr(psm_result, 'n_matched', None)
    n_matched = n_matched_attr if n_matched_attr is not None else psm_result.n_treated
    if psm_result.n_treated > 0:
        raw_rate = n_matched / psm_result.n_treated
        match_rate = max(0.0, min(1.0, raw_rate))
    else:
        match_rate = 0.0
    
    return {
        'att': psm_result.att,
        'se_att': psm_result.se,
        't_stat': psm_result.t_stat,
        'pvalue': psm_result.pvalue,
        'ci_lower': psm_result.ci_lower,
        'ci_upper': psm_result.ci_upper,
        'nobs': psm_result.n_treated + psm_result.n_control,
        'df_resid': psm_result.n_treated + psm_result.n_control - 2,
        'df_inference': psm_result.n_treated + psm_result.n_control - 2,
        'vce_type': vce if vce is not None else 'psm',
        'cluster_var': cluster_var,
        'n_clusters': None,
        'controls_used': controls is not None and len(controls) > 0,
        'controls': controls if controls else [],
        'controls_spec': None,
        'n_treated_sample': psm_result.n_treated,
        'n_control_sample': psm_result.n_control,
        'params': None,
        'bse': None,
        'vcov': None,
        'resid': None,
        'diagnostics': psm_result.diagnostics if hasattr(psm_result, 'diagnostics') else None,
        'propensity_scores': psm_result.propensity_scores,
        # PSM uses direct matching without IPW weights.
        'weights_cv': np.nan,
        'n_matched': n_matched,
        'match_rate': match_rate,
        'estimator': 'psm',
    }


def _estimate_period_effects_ipw(
    data: pd.DataFrame,
    ydot: str,
    d: str,
    tindex: str,
    tpost1: int,
    Tmax: int,
    estimator: str,
    controls: list[str] | None,
    ps_controls: list[str] | None,
    trim_threshold: float,
    n_neighbors: int,
    caliper: float | None,
    with_replacement: bool,
    match_order: str,
    period_labels: dict,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Estimate period-specific treatment effects using propensity score methods.

    Applies the specified estimator (IPW, IPWRA, or PSM) to each post-treatment
    period cross-section independently, producing period-specific ATT estimates.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data containing the transformed outcome variable.
    ydot : str
        Column name of the transformed outcome.
    d : str
        Column name of the binary treatment indicator.
    tindex : str
        Column name of the time period index.
    tpost1 : int
        Index of the first post-treatment period.
    Tmax : int
        Index of the last period in the panel.
    estimator : {'ipw', 'ipwra', 'psm'}
        Estimation method to apply at each period.
    controls : list of str or None
        Control variables for the outcome model (IPWRA only).
    ps_controls : list of str or None
        Control variables for the propensity score model.
    trim_threshold : float
        Propensity score trimming threshold in (0, 0.5).
    n_neighbors : int
        Number of nearest neighbors for PSM matching.
    caliper : float or None
        Maximum propensity score distance for PSM matches.
    with_replacement : bool
        Whether PSM allows control unit reuse.
    match_order : str
        Order for processing treated units in PSM without replacement.
    period_labels : dict
        Mapping from time index to human-readable period labels.
    alpha : float, default=0.05
        Significance level for confidence intervals.

    Returns
    -------
    pd.DataFrame
        Period-specific estimates with columns: 'period', 'tindex', 'beta',
        'se', 'ci_lower', 'ci_upper', 'tstat', 'pval', 'N'.
    """
    results_list = []

    # Return empty DataFrame if no post-treatment periods exist.
    if tpost1 > Tmax:
        warnings.warn(
            f"First post-treatment period ({tpost1}) is after the last period ({Tmax}). "
            f"No period-specific effects can be estimated. "
            f"This may indicate data issues or incorrect post indicator.",
            DataWarning,
            stacklevel=4
        )
        return pd.DataFrame(columns=[
            'period', 'tindex', 'beta', 'se', 'ci_lower', 'ci_upper', 'tstat', 'pval', 'N'
        ])

    # Track periods with estimation issues for consolidated warnings.
    empty_periods = []
    insufficient_periods = []
    failed_periods = []
    
    for t in range(tpost1, Tmax + 1):
        mask_t = (data[tindex] == t)
        data_t = data[mask_t].copy()
        period_label = period_labels.get(t, str(t))

        if len(data_t) == 0:
            empty_periods.append(period_label)
            results_list.append({
                'period': period_label,
                'tindex': t,
                'beta': np.nan,
                'se': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'tstat': np.nan,
                'pval': np.nan,
                'N': 0
            })
            continue

        n_treated_t = int(data_t[d].sum())
        n_control_t = int(len(data_t) - n_treated_t)
        
        if n_treated_t == 0 or n_control_t == 0:
            insufficient_periods.append(
                f"{period_label} (N_treated={n_treated_t}, N_control={n_control_t})"
            )
            results_list.append({
                'period': period_label,
                'tindex': t,
                'beta': np.nan,
                'se': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'tstat': np.nan,
                'pval': np.nan,
                'N': n_treated_t + n_control_t
            })
            continue
        
        try:
            if estimator == 'ipw':
                result_t = estimate_ipw(
                    data=data_t,
                    y=ydot,
                    d=d,
                    propensity_controls=ps_controls,
                    trim_threshold=trim_threshold,
                    alpha=alpha,
                    return_diagnostics=False,
                    gvar_col=None,
                    ivar_col=None,
                    cohort_g=None,
                    period_r=None,
                )
            elif estimator == 'ipwra':
                result_t = estimate_ipwra(
                    data=data_t,
                    y=ydot,
                    d=d,
                    controls=controls,
                    propensity_controls=ps_controls,
                    trim_threshold=trim_threshold,
                    alpha=alpha,
                    return_diagnostics=False,
                    gvar_col=None,
                    ivar_col=None,
                    cohort_g=None,
                    period_r=None,
                )
            elif estimator == 'psm':
                result_t = estimate_psm(
                    data=data_t,
                    y=ydot,
                    d=d,
                    propensity_controls=ps_controls,
                    n_neighbors=n_neighbors,
                    caliper=caliper,
                    with_replacement=with_replacement,
                    match_order=match_order,
                    alpha=alpha,
                    return_diagnostics=False,
                    gvar_col=None,
                    ivar_col=None,
                    cohort_g=None,
                    period_r=None,
                )
            
            results_list.append({
                'period': period_label,
                'tindex': t,
                'beta': result_t.att,
                'se': result_t.se,
                'ci_lower': result_t.ci_lower,
                'ci_upper': result_t.ci_upper,
                'tstat': result_t.t_stat,
                'pval': result_t.pvalue,
                'N': result_t.n_treated + result_t.n_control
            })
            
        except (ValueError, np.linalg.LinAlgError, RuntimeError, KeyError) as e:
            failed_periods.append(f"{period_label} ({type(e).__name__})")
            results_list.append({
                'period': period_label,
                'tindex': t,
                'beta': np.nan,
                'se': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'tstat': np.nan,
                'pval': np.nan,
                'N': n_treated_t + n_control_t
            })

    if empty_periods:
        warnings.warn(
            f"{len(empty_periods)} period(s) contain no observations: "
            f"{', '.join(empty_periods[:5])}"
            f"{' ...' if len(empty_periods) > 5 else ''}. "
            f"Results for these periods set to NaN.",
            DataWarning,
            stacklevel=4
        )
    
    if insufficient_periods:
        warnings.warn(
            f"{len(insufficient_periods)} period(s) have insufficient treated/control units: "
            f"{', '.join(insufficient_periods[:3])}"
            f"{' ...' if len(insufficient_periods) > 3 else ''}. "
            f"Results for these periods set to NaN.",
            SmallSampleWarning,
            stacklevel=4
        )
    
    if failed_periods:
        warnings.warn(
            f"{len(failed_periods)} period(s) failed {estimator.upper()} estimation: "
            f"{', '.join(failed_periods[:3])}"
            f"{' ...' if len(failed_periods) > 3 else ''}. "
            f"Results for these periods set to NaN.",
            NumericalWarning,
            stacklevel=4
        )
    
    return pd.DataFrame(results_list)


def lwdid(
    data: pd.DataFrame,
    y: str,
    d: str | None = None,
    ivar: str | None = None,
    tvar: str | list[str] | None = None,
    post: str | None = None,
    rolling: str = 'demean',
    *,
    gvar: str | None = None,
    control_group: str = 'not_yet_treated',
    estimator: str = 'ra',
    aggregate: str = 'cohort',
    balanced_panel: str = 'warn',
    ps_controls: list[str] | None = None,
    trim_threshold: float = 0.01,
    return_diagnostics: bool = False,
    n_neighbors: int = 1,
    caliper: float | None = None,
    with_replacement: bool = True,
    match_order: str = 'data',
    vce: str | None = None,
    controls: list[str] | None = None,
    cluster_var: str | None = None,
    alpha: float = 0.05,
    ri: bool = False,
    rireps: int = 1000,
    seed: int | None = None,
    ri_method: str = 'bootstrap',
    graph: bool = False,
    gid: str | int | None = None,
    graph_options: dict | None = None,
    season_var: str | None = None,
    Q: int = 4,
    auto_detect_frequency: bool = False,
    include_pretreatment: bool = False,
    pretreatment_test: bool = True,
    pretreatment_alpha: float = 0.05,
    exclude_pre_periods: int = 0,
    verbose: str = 'default',
    **kwargs,
) -> LWDIDResults:
    """
    Difference-in-differences estimator with unit-specific transformations.

    Implements the rolling transformation approach for DiD estimation,
    supporting three methodological scenarios:

    1. **Small-sample common timing**: Exact t-based inference under classical
       linear model assumptions.

    2. **Large-sample common timing**: Asymptotic inference with
       heteroskedasticity-robust standard errors.

    3. **Staggered adoption**: Cohort-time specific effect estimation with
       flexible control group strategies.

    The transformation removes unit-specific pre-treatment patterns, converting
    panel DiD into a cross-sectional treatment effects problem.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format with one row per unit-time observation.
        Each (unit, time) combination must be unique. Requires at least 3 units.
    y : str
        Column name of the outcome variable.
    d : str, optional
        Column name of the unit-level treatment indicator (required for common
        timing mode). Must be time-invariant: non-zero for treated units, zero
        for control units. Ignored in staggered mode.
    ivar : str
        Column name of the unit identifier.
    tvar : str or list of str
        Time variable specification. For annual data, a single column name.
        For quarterly data, a list of two column names [year_var, quarter_var]
        where quarter_var contains values in {1, 2, 3, 4}.
    post : str, optional
        Column name of the post-treatment indicator (required for common timing
        mode). Internally binarized: non-zero values indicate post-treatment
        periods. Must be monotone non-decreasing in time (no treatment reversals).
    rolling : {'demean', 'detrend', 'demeanq', 'detrendq'}, default='demean'
        Transformation method (case-insensitive):

        - 'demean': Remove unit-specific pre-treatment mean.
        - 'detrend': Remove unit-specific linear time trend.
        - 'demeanq': Demeaning with seasonal fixed effects. Requires ``season_var``
          and ``Q`` parameters. Supports quarterly (Q=4), monthly (Q=12), or
          weekly (Q=52) data.
        - 'detrendq': Detrending with seasonal fixed effects. Requires ``season_var``
          and ``Q`` parameters. Supports quarterly (Q=4), monthly (Q=12), or
          weekly (Q=52) data.

        All four transformation methods are supported for both common timing
        and staggered adoption designs.
    gvar : str, optional
        Column name indicating first treatment period for staggered adoption.
        If specified, activates staggered mode and ignores ``d`` and ``post``.
        Valid values: positive integers (treatment cohort), 0/inf/NaN (never-treated).
    control_group : {'not_yet_treated', 'never_treated', 'all_others'}, default='not_yet_treated'
        Control group composition for staggered adoption:

        - 'not_yet_treated': Never-treated plus not-yet-treated units.
        - 'never_treated': Only never-treated units.
        - 'all_others': All units not in the treated cohort (including already-treated units).
          This option is mainly intended for replication/diagnostics and may introduce
          forbidden comparisons under no-anticipation.

        Auto-switched to 'never_treated' for cohort/overall aggregation.
    estimator : {'ra', 'ipw', 'ipwra', 'psm'}, default='ra'
        Estimation method (case-insensitive):

        - 'ra': Regression adjustment via OLS on transformed outcomes.
        - 'ipw': Inverse probability weighting. Requires ``controls``.
        - 'ipwra': Doubly robust combining IPW with RA. Requires ``controls``.
        - 'psm': Propensity score matching. Requires ``controls``.
    aggregate : {'none', 'cohort', 'overall'}, default='cohort'
        Aggregation level for staggered adoption:

        - 'none': Return cohort-time specific effects only.
        - 'cohort': Aggregate to cohort-specific effects.
        - 'overall': Aggregate to a single weighted overall effect.
    balanced_panel : {'warn', 'error', 'ignore'}, default='warn'
        How to handle unbalanced panels (units with different observation counts):

        - 'warn': Issue a warning with selection mechanism diagnostics (default).
        - 'error': Raise UnbalancedPanelError if panel is unbalanced.
        - 'ignore': Silently proceed without warnings.

        Selection may depend on time-invariant heterogeneity but not on
        shocks to the untreated potential outcome. Use
        ``diagnose_selection_mechanism()`` for detailed diagnostics.
    ps_controls : list of str, optional
        Control variables for propensity score model. If None, uses ``controls``.
    trim_threshold : float, default=0.01
        Propensity score trimming threshold. Observations with propensity scores
        outside [trim_threshold, 1 - trim_threshold] are excluded.
    return_diagnostics : bool, default=False
        Whether to include propensity score diagnostics in results.
    n_neighbors : int, default=1
        Number of nearest neighbors for PSM matching.
    caliper : float, optional
        Maximum propensity score distance for PSM, in units of PS standard
        deviation. Treated units without valid matches are dropped.
    with_replacement : bool, default=True
        Whether PSM allows control units to be matched multiple times.
    match_order : {'data', 'random', 'largest', 'smallest'}, default='data'
        Order for processing treated units in without-replacement PSM:

        - 'data': Original data order.
        - 'random': Randomized order (use ``seed`` for reproducibility).
        - 'largest': Prioritize units with extreme propensity scores.
        - 'smallest': Prioritize units with propensity scores near 0.5.
    vce : {None, 'robust', 'hc0', 'hc1', 'hc2', 'hc3', 'hc4', 'cluster'}, optional
        Variance estimator (case-insensitive):

        - None: Homoskedastic OLS standard errors.
        - 'hc0': White heteroskedasticity-robust.
        - 'robust'/'hc1': HC1 with degrees-of-freedom correction N/(N-K).
        - 'hc2': Leverage-adjusted using (1 - h_ii)^{-1}.
        - 'hc3': Small-sample adjusted using (1 - h_ii)^{-2}.
        - 'hc4': Adaptive leverage correction.
        - 'cluster': Cluster-robust (requires ``cluster_var``).
    controls : list of str, optional
        Time-invariant control variables for outcome regression.
    cluster_var : str, optional
        Column name for clustering (required when vce='cluster').
    alpha : float, default=0.05
        Significance level for confidence intervals.
    ri : bool, default=False
        Whether to perform randomization inference for the null H0: ATT=0.
    rireps : int, default=1000
        Number of randomization inference replications.
    seed : int, optional
        Random seed for reproducibility in randomization inference.
    ri_method : {'bootstrap', 'permutation'}, default='bootstrap'
        Resampling method for randomization inference:

        - 'bootstrap': With-replacement resampling.
        - 'permutation': Fisher's exact permutation test.
    graph : bool, default=False
        Whether to generate a plot of transformed outcomes over time.
    gid : str or int, optional
        Specific unit identifier to highlight in the plot.
    graph_options : dict, optional
        Additional plotting options passed to the visualization function.
    season_var : str, optional
        Column name of seasonal indicator variable for seasonal transformations
        (demeanq, detrendq). Values should be integers from 1 to Q representing
        seasonal periods (e.g., quarters 1-4, months 1-12, or weeks 1-52).
        This parameter is preferred over the legacy ``quarter`` parameter in
        ``tvar`` for non-quarterly seasonal data.
    Q : int, default=4
        Number of seasonal periods per cycle. Used with seasonal transformations
        (demeanq, detrendq). Common values:

        - 4: Quarterly data (default)
        - 12: Monthly data
        - 52: Weekly data

        Must match the range of values in ``season_var`` (1 to Q).
    auto_detect_frequency : bool, default=False
        Whether to automatically detect data frequency and set Q accordingly.
        When True, the function analyzes the time variable to infer whether
        data is quarterly (Q=4), monthly (Q=12), or weekly (Q=52).

        - If detection succeeds with high confidence, Q is set automatically.
        - If detection fails or has low confidence, a warning is issued and
          the explicit Q value is used.
        - An explicit Q value always overrides auto-detection when both are
          specified (Q != 4 and auto_detect_frequency=True).

        This parameter is useful when working with datasets of unknown frequency
        or when building generic analysis pipelines.
    include_pretreatment : bool, default=False
        Whether to compute pre-treatment transformed outcomes and ATT estimates
        for parallel trends assessment. Only applicable in staggered mode.
        
        When True:
        
        - Applies rolling transformations to pre-treatment periods using
          future pre-treatment periods {t+1, ..., g-1} as reference.
        - Estimates pre-treatment ATT for each (cohort, period) pair.
        - Stores results in ``att_pre_treatment`` attribute of LWDIDResults.
        - Enables extended event study visualization with pre-treatment effects.
        
        Under the parallel trends assumption, pre-treatment ATT estimates
        should be statistically indistinguishable from zero.
    pretreatment_test : bool, default=True
        Whether to perform parallel trends statistical test when
        ``include_pretreatment=True``. The test includes:
        
        - Individual t-tests for each pre-treatment period ATT.
        - Joint F-test for H0: all pre-treatment ATT = 0.
        
        Results are stored in ``parallel_trends_test`` attribute.
    pretreatment_alpha : float, default=0.05
        Significance level for parallel trends test. Used for determining
        ``reject_null`` in the test results.
    exclude_pre_periods : int, default=0
        Number of pre-treatment periods to exclude immediately before treatment.
        Used to address potential anticipation effects when the no-anticipation
        assumption may be violated.
        
        When ``exclude_pre_periods > 0``:
        
        - The specified number of periods immediately before treatment are
          excluded from the pre-treatment sample used for transformation.
        - For common timing: excludes the last k pre-treatment periods.
        - For staggered adoption: excludes k periods before each cohort's
          treatment date.
        
        This implements a robustness check for testing sensitivity to
        anticipation effects.
        
        Example: If treatment occurs at t=6 and ``exclude_pre_periods=2``,
        periods t=4 and t=5 are excluded from the pre-treatment sample.
    verbose : {'quiet', 'default', 'verbose'}, default='default'
        Warning output verbosity level (case-insensitive):

        - 'quiet': Suppress informational warnings; emit only critical
          warnings (convergence failures, numerical instability).
        - 'default': Emit one aggregated summary per warning category.
          Repeated warnings from (cohort, period) loops are collected
          and reported as a single summary with affected-pair counts.
        - 'verbose': Emit every individual warning record without
          aggregation.

        Regardless of the verbosity setting, all warnings are always
        recorded in ``LWDIDResults.diagnostics`` for post-hoc inspection.

    Returns
    -------
    LWDIDResults
        Results object with the following key attributes:

        - att : Average treatment effect on the treated.
        - se_att : Standard error of ATT.
        - t_stat : t-statistic for H0: ATT=0.
        - pvalue : Two-sided p-value.
        - ci_lower, ci_upper : Confidence interval bounds.
        - df_inference : Degrees of freedom for inference.
        - nobs : Number of observations in estimation sample.
        - n_treated, n_control : Unit counts by treatment status.
        - att_by_period : Period-specific ATT estimates (DataFrame).
        - ri_pvalue : Randomization inference p-value (if ri=True).
        - att_pre_treatment : Pre-treatment ATT estimates (if include_pretreatment=True).
        - parallel_trends_test : Parallel trends test results (if include_pretreatment=True).
        - include_pretreatment : Whether pre-treatment dynamics were computed.

        Key methods: summary(), plot(), to_excel(), to_csv(), to_latex(),
        get_diagnostics(), plot_event_study().

    Raises
    ------
    MissingRequiredColumnError
        Required columns not found in data.
    InvalidRollingMethodError
        Invalid rolling method specified.
    InsufficientDataError
        Insufficient sample size or pre-/post-treatment observations.
    NoTreatedUnitsError
        No treated units in data.
    NoControlUnitsError
        No control units in data.
    InsufficientPrePeriodsError
        Insufficient pre-treatment periods for the chosen transformation.
    NoNeverTreatedError
        Cohort/overall aggregation requested but no never-treated units exist.

    Notes
    -----
    Mode selection:

    - **Common timing** (gvar=None): Requires ``d``, ``post``, ``rolling``.
    - **Staggered adoption** (gvar specified): Requires ``gvar``, ``rolling``.

    Confidence intervals use t-distribution critical values with degrees of
    freedom N-k (homoskedastic) or G-1 (cluster-robust).

    See Also
    --------
    LWDIDResults : Detailed documentation of the results container.
    """
    from .exceptions import UnbalancedPanelError
    from .warning_registry import WarningRegistry
    
    # Create centralized warning registry for deferred collection and aggregation.
    # The registry buffers warnings during (g, r) iteration loops and emits
    # aggregated summaries when flushed, preventing warning floods in staggered mode.
    registry = WarningRegistry(verbose=verbose)

    # Validate unknown kwargs to catch parameter typos.
    # Reserved for backward compatibility: riseed is an alias for seed.
    _KNOWN_KWARGS = {'riseed'}
    unknown_kwargs = set(kwargs.keys()) - _KNOWN_KWARGS
    if unknown_kwargs:
        warnings.warn(
            f"Unknown keyword argument(s) ignored: {sorted(unknown_kwargs)}. "
            f"Valid extra arguments: {sorted(_KNOWN_KWARGS)}.",
            DataWarning,
            stacklevel=2
        )

    # Validate balanced_panel parameter
    if balanced_panel not in ('warn', 'error', 'ignore'):
        raise ValueError(
            f"balanced_panel must be 'warn', 'error', or 'ignore', got '{balanced_panel}'"
        )

    if vce is not None:
        if not isinstance(vce, str):
            raise TypeError(
                f"Parameter 'vce' must be a string or None, got {type(vce).__name__}.\n"
                f"Valid values: None, 'robust', 'hc0', 'hc1', 'hc2', 'hc3', 'hc4', 'cluster'"
            )
        vce = vce.lower()

    _validate_psm_params(n_neighbors, caliper, with_replacement, match_order)

    # Validate Q parameter for seasonal transformations.
    if not isinstance(Q, (int, np.integer)):
        raise TypeError(
            f"Parameter 'Q' must be an integer, got {type(Q).__name__}.\n"
            f"Common values: 4 (quarterly), 12 (monthly), 52 (weekly)."
        )
    if Q < 2:
        raise ValueError(
            f"Parameter 'Q' must be >= 2, got {Q}.\n"
            f"Q represents the number of seasonal periods per cycle.\n"
            f"Common values: 4 (quarterly), 12 (monthly), 52 (weekly)."
        )

    # Validate season_var parameter.
    if season_var is not None and not isinstance(season_var, str):
        raise TypeError(
            f"Parameter 'season_var' must be a string or None, got {type(season_var).__name__}.\n"
            f"Specify the column name containing seasonal values (1 to Q)."
        )

    # Auto-detect frequency if requested and using seasonal transformations.
    # Detection only applies when rolling method requires seasonal adjustment.
    if auto_detect_frequency:
        if not isinstance(auto_detect_frequency, bool):
            raise TypeError(
                f"Parameter 'auto_detect_frequency' must be a boolean, "
                f"got {type(auto_detect_frequency).__name__}."
            )
        
        # Only auto-detect for seasonal transformations.
        rolling_lower = rolling.lower() if isinstance(rolling, str) else ''
        if rolling_lower in ('demeanq', 'detrendq'):
            # Determine time variable for detection.
            tvar_for_detection = tvar if isinstance(tvar, str) else (tvar[0] if tvar else None)
            
            if tvar_for_detection is not None and ivar is not None:
                try:
                    detection_result = validation.detect_frequency(
                        data, tvar=tvar_for_detection, ivar=ivar
                    )
                    
                    detected_Q = detection_result.get('Q')
                    confidence = detection_result.get('confidence', 0)
                    frequency = detection_result.get('frequency')
                    
                    # Check if user explicitly set Q (not default value).
                    user_specified_Q = Q != 4
                    
                    if detected_Q is not None and confidence >= 0.5:
                        if user_specified_Q and Q != detected_Q:
                            # User explicitly set Q, warn about mismatch but use user's value.
                            logger.warning(
                                f"Auto-detected frequency '{frequency}' (Q={detected_Q}) "
                                f"differs from explicit Q={Q}. Using explicit Q={Q}."
                            )
                        else:
                            # Use detected Q value.
                            Q = detected_Q
                            logger.info(
                                f"Auto-detected data frequency: {frequency} (Q={Q}, "
                                f"confidence={confidence:.2f})"
                            )
                    elif detected_Q is None or confidence < 0.5:
                        # Detection failed or low confidence.
                        warnings.warn(
                            f"Could not reliably detect data frequency "
                            f"(confidence={confidence:.2f}). Using Q={Q}. "
                            f"Consider setting Q explicitly for seasonal transformations.",
                            DataWarning,
                            stacklevel=2
                        )
                except Exception as e:
                    # Detection failed, use default Q.
                    warnings.warn(
                        f"Frequency auto-detection failed: {e}. Using Q={Q}.",
                        DataWarning,
                        stacklevel=2
                    )
            else:
                warnings.warn(
                    "Cannot auto-detect frequency: tvar or ivar not specified. Using Q={Q}.",
                    DataWarning,
                    stacklevel=2
                )

    # Check panel balance if balanced_panel='error'
    if balanced_panel == 'error' and ivar is not None:
        tvar_col = tvar if isinstance(tvar, str) else tvar[0]
        if tvar_col in data.columns and ivar in data.columns:
            panel_counts = data.groupby(ivar)[tvar_col].count()
            is_balanced = panel_counts.nunique() == 1
            
            if not is_balanced:
                min_obs = int(panel_counts.min())
                max_obs = int(panel_counts.max())
                n_incomplete = int((panel_counts < max_obs).sum())
                
                raise UnbalancedPanelError(
                    f"Unbalanced panel detected: {n_incomplete} units have incomplete "
                    f"observations (range: {min_obs}-{max_obs}). "
                    f"Set balanced_panel='warn' to proceed with warnings, or create "
                    f"a balanced subsample. Use diagnose_selection_mechanism() for "
                    f"detailed diagnostics on selection bias risk.",
                    min_obs=min_obs,
                    max_obs=max_obs,
                    n_incomplete_units=n_incomplete,
                )

    # Dispatch to staggered or common timing implementation.
    if gvar is not None:
        if d is not None or post is not None:
            warnings.warn(
                "Both gvar and d/post parameters provided. Staggered mode takes precedence. "
                "The d and post parameters will be ignored in staggered mode.",
                DataWarning,
                stacklevel=2
            )
        return _lwdid_staggered(
            data=data, y=y, ivar=ivar, tvar=tvar, gvar=gvar,
            rolling=rolling, control_group=control_group,
            estimator=estimator, aggregate=aggregate,
            ps_controls=ps_controls, trim_threshold=trim_threshold,
            return_diagnostics=return_diagnostics,
            n_neighbors=n_neighbors, caliper=caliper,
            with_replacement=with_replacement,
            match_order=match_order,
            vce=vce, controls=controls, cluster_var=cluster_var,
            alpha=alpha,
            ri=ri, rireps=rireps, seed=seed, ri_method=ri_method,
            graph=graph, gid=gid, graph_options=graph_options,
            season_var=season_var, Q=Q,
            include_pretreatment=include_pretreatment,
            pretreatment_test=pretreatment_test,
            pretreatment_alpha=pretreatment_alpha,
            exclude_pre_periods=exclude_pre_periods,
            warning_registry=registry,
            **kwargs
        )
    else:
        # Common timing mode: validate estimator parameter.
        # Type check required before calling .lower() to avoid AttributeError.
        if estimator is not None and not isinstance(estimator, str):
            raise TypeError(
                f"estimator must be a string, got {type(estimator).__name__}.\n"
                f"Valid values: ('ra', 'ipw', 'ipwra', 'psm').\n"
                f"Example: lwdid(..., estimator='ipwra')"
            )
        estimator_lower = estimator.lower() if estimator else 'ra'
        VALID_ESTIMATORS_COMMON = ('ra', 'ipw', 'ipwra', 'psm')
        if estimator_lower not in VALID_ESTIMATORS_COMMON:
            raise ValueError(
                f"Invalid estimator='{estimator}'.\n"
                f"Valid values for common timing mode: {VALID_ESTIMATORS_COMMON}"
            )
        
        # Check for staggered-only parameters (control_group, aggregate).
        ignored_staggered_params = []
        if control_group != 'not_yet_treated':
            ignored_staggered_params.append(f"control_group='{control_group}'")
        if aggregate != 'cohort':
            ignored_staggered_params.append(f"aggregate='{aggregate}'")
        
        # For RA estimator, IPW-related parameters are ignored.
        if estimator_lower == 'ra':
            if ps_controls is not None:
                ignored_staggered_params.append(f"ps_controls={ps_controls}")
            if trim_threshold != 0.01:
                ignored_staggered_params.append(f"trim_threshold={trim_threshold}")
            if return_diagnostics:
                ignored_staggered_params.append("return_diagnostics=True")
            if n_neighbors != 1:
                ignored_staggered_params.append(f"n_neighbors={n_neighbors}")
            if caliper is not None:
                ignored_staggered_params.append(f"caliper={caliper}")
            if not with_replacement:
                ignored_staggered_params.append("with_replacement=False")
            if match_order != 'data':
                ignored_staggered_params.append(f"match_order='{match_order}'")
        
        # Validate control variables based on estimator type.
        # IPWRA: requires controls (outcome model), ps_controls optional (defaults to controls).
        # IPW/PSM: requires controls OR ps_controls (propensity score model only).
        if estimator_lower == 'ipwra' and not controls:
            raise ValueError(
                f"estimator='ipwra' requires 'controls' parameter for outcome model.\n"
                f"IPWRA (doubly robust) uses controls in both outcome regression and "
                f"propensity score model.\n"
                f"  - controls: Variables for outcome model E[Y|X,D=0]\n"
                f"  - ps_controls: Variables for propensity score P(D=1|X), defaults to controls"
            )
        elif estimator_lower in ('ipw', 'psm') and not controls and not ps_controls:
            raise ValueError(
                f"estimator='{estimator}' requires 'controls' or 'ps_controls' parameter.\n"
                f"IPW and PSM estimators need control variables for propensity score model.\n"
                f"  - Use 'controls' to specify variables (also used as ps_controls by default)\n"
                f"  - Or use 'ps_controls' to specify propensity score model variables directly"
            )
        
        # Validate trim_threshold range for IPW/IPWRA/PSM estimators.
        # At threshold=0.5, scores are clipped to [0.5, 0.5], excluding all observations.
        if estimator_lower in ('ipw', 'ipwra', 'psm'):
            if not (0 < trim_threshold < 0.5):
                raise ValueError(
                    f"Invalid trim_threshold={trim_threshold}.\n"
                    f"trim_threshold must be in (0, 0.5) for valid propensity score trimming.\n"
                    f"  - trim_threshold=0.01 trims PS to [0.01, 0.99] (default)\n"
                    f"  - trim_threshold=0.05 trims PS to [0.05, 0.95]\n"
                    f"  - trim_threshold=0.5 is invalid as it would trim all observations"
                )
        
        # For non-PSM estimators, PSM-specific parameters are ignored.
        if estimator_lower in ('ipw', 'ipwra'):
            if n_neighbors != 1:
                ignored_staggered_params.append(f"n_neighbors={n_neighbors}")
            if caliper is not None:
                ignored_staggered_params.append(f"caliper={caliper}")
            if not with_replacement:
                ignored_staggered_params.append("with_replacement=False")
            if match_order != 'data':
                ignored_staggered_params.append(f"match_order='{match_order}'")
        
        if ignored_staggered_params:
            warnings.warn(
                f"Common timing mode (gvar=None): the following parameters "
                f"are ignored: {', '.join(ignored_staggered_params)}. "
                f"To use control_group/aggregate, specify the 'gvar' parameter for staggered adoption.",
                DataWarning,
                stacklevel=2
            )
        
        if d is None:
            raise ValueError(
                "Common timing mode requires 'd' parameter (unit-level treatment indicator).\n"
                "If your data has staggered adoption (units treated at different times), "
                "use the 'gvar' parameter to specify the first treatment period column."
            )
        if post is None:
            raise ValueError(
                "Common timing mode requires 'post' parameter (post-treatment period indicator).\n"
                "If your data has staggered adoption, use the 'gvar' parameter instead."
            )
        if ivar is None:
            raise ValueError("Parameter 'ivar' (unit identifier column) is required.")
        if tvar is None:
            raise ValueError("Parameter 'tvar' (time variable column) is required.")

        if rolling is None:
            raise ValueError(
                "Common timing mode requires 'rolling' parameter.\n"
                "Valid values: 'demean', 'detrend', 'demeanq', 'detrendq'.\n"
                "  - 'demean': Remove unit-specific pre-treatment mean\n"
                "  - 'detrend': Remove unit-specific linear time trend\n"
                "  - 'demeanq': Demeaning with quarterly fixed effects\n"
                "  - 'detrendq': Detrending with quarterly fixed effects"
            )

        if not isinstance(rolling, str):
            raise TypeError(
                f"Parameter 'rolling' must be a string, got {type(rolling).__name__}. "
                f"Valid values: 'demean', 'detrend', 'demeanq', 'detrendq'."
            )

        if vce is not None and vce.lower() == 'cluster' and cluster_var is None:
            raise InvalidParameterError(
                "vce='cluster' requires cluster_var parameter.\n"
                "Specify the column name for cluster-robust standard errors."
            )

        if not isinstance(alpha, (int, float, np.number)):
            raise TypeError(
                f"Parameter 'alpha' must be numeric, got {type(alpha).__name__}.\n"
                f"Example: alpha=0.05 for 95% confidence interval."
            )
        if hasattr(alpha, '__float__') and np.isnan(float(alpha)):
            raise ValueError("Parameter 'alpha' cannot be NaN.")
        if not (0 < alpha < 1):
            raise ValueError(
                f"Parameter 'alpha' must be between 0 and 1 (exclusive), got {alpha}.\n"
                "Common values: 0.05 (95% CI), 0.10 (90% CI), 0.01 (99% CI)."
            )

        # Validate exclude_pre_periods parameter.
        if not isinstance(exclude_pre_periods, (int, np.integer)):
            raise TypeError(
                f"Parameter 'exclude_pre_periods' must be an integer, "
                f"got {type(exclude_pre_periods).__name__}.\n"
                f"Example: exclude_pre_periods=2 to exclude 2 periods before treatment."
            )
        if exclude_pre_periods < 0:
            raise ValueError(
                f"Parameter 'exclude_pre_periods' must be non-negative, "
                f"got {exclude_pre_periods}.\n"
                f"Use 0 for no exclusion, or a positive integer to exclude periods."
            )

        if isinstance(tvar, (list, tuple)):
            if len(tvar) != 2:
                raise ValueError(
                    f"Parameter 'tvar' as a list must have exactly 2 elements "
                    f"[year_column, quarter_column], got {len(tvar)} elements: {tvar}"
                )

        if ri:
            # Accept both Python int and numpy integer types for rireps.
            if not isinstance(rireps, (int, np.integer)) or rireps < 1:
                raise ValueError(
                    f"Invalid rireps={rireps}.\n"
                    f"rireps must be a positive integer >= 1 when ri=True.\n"
                    f"Recommended: rireps >= 500 for reliable p-values."
                )

            # Validate ri_method type before calling .lower()
            if ri_method is not None and not isinstance(ri_method, str):
                raise TypeError(
                    f"Parameter 'ri_method' must be a string or None, "
                    f"got {type(ri_method).__name__}.\n"
                    f"Valid values: 'bootstrap', 'permutation'"
                )

            VALID_RI_METHODS = ('bootstrap', 'permutation')
            ri_method_lower = ri_method.lower() if ri_method else 'bootstrap'
            if ri_method_lower not in VALID_RI_METHODS:
                raise ValueError(
                    f"Invalid ri_method='{ri_method}'.\n"
                    f"Valid values: {VALID_RI_METHODS}\n"
                    f"  - 'bootstrap': Bootstrap resampling (with replacement)\n"
                    f"  - 'permutation': Fisher's exact permutation test (without replacement)"
                )
            # Normalize ri_method to lowercase for downstream use.
            ri_method = ri_method_lower

            # Validate seed type to ensure reproducibility.
            # Float values would be silently truncated by random.seed(),
            # potentially causing unexpected behavior.
            if seed is not None:
                if not isinstance(seed, (int, np.integer)):
                    raise TypeError(
                        f"Parameter 'seed' must be an integer or None, "
                        f"got {type(seed).__name__}.\n"
                        f"Example: lwdid(..., ri=True, seed=42)"
                    )
                if seed < 0:
                    raise ValueError(
                        f"Parameter 'seed' must be non-negative, got {seed}.\n"
                        f"Use a non-negative integer for reproducible results."
                    )

    # Data validation and transformation.
    data_clean, metadata = validation.validate_and_prepare_data(
        data=data,
        y=y,
        d=d,
        ivar=ivar,
        tvar=tvar,
        post=post,
        rolling=rolling,
        controls=controls,
        season_var=season_var,
    )

    rolling = metadata['rolling']

    # Resolve season_var: prefer explicit season_var over tvar[1] for backward compatibility.
    # If season_var is provided, use it; otherwise fall back to tvar[1] for quarterly data.
    effective_season_var = season_var
    if effective_season_var is None and not isinstance(tvar, str):
        # Legacy behavior: use tvar[1] as quarter variable when tvar is a list
        effective_season_var = tvar[1]

    data_transformed = transformations.apply_rolling_transform(
        data=data_clean,
        y=y,
        ivar=ivar,
        tindex='tindex',
        post='post_',
        rolling=rolling,
        tpost1=metadata['tpost1'],
        quarter=tvar[1] if not isinstance(tvar, str) else None,
        season_var=effective_season_var,
        Q=Q,
        exclude_pre_periods=exclude_pre_periods,
    )

    # Extract first post-treatment cross-section for ATT estimation.
    # Using firstpost ensures consistent sample across all estimators.
    firstpost_data = data_transformed[data_transformed['firstpost']].copy()
    
    # Verify first post-treatment period has observations.
    # Empty data would cause unclear errors in downstream estimators.
    if len(firstpost_data) == 0:
        raise InsufficientDataError(
            "No observations found for the first post-treatment period.\n\n"
            "Possible causes:\n"
            "  - No treated units in the data (all units have d=0)\n"
            "  - The 'post' indicator is never 1 (no post-treatment periods)\n"
            "  - All first-post observations were filtered during transformation\n"
            "  - Data filtering removed all eligible observations\n\n"
            "How to fix:\n"
            "  1. Check treatment indicator: data[d].value_counts()\n"
            "  2. Check post indicator: data[post].value_counts()\n"
            "  3. Verify data has post-treatment observations for treated units"
        )
    
    if estimator_lower == 'ra':
        # RA uses OLS on transformed outcomes for unbiased ATT under parallel trends.
        results_dict = estimation.estimate_att(
            data=data_transformed,
            y_transformed='ydot_postavg',
            d='d_',
            ivar=ivar,
            controls=controls,
            vce=vce,
            cluster_var=cluster_var,
            sample_filter=data_transformed['firstpost'],
            alpha=alpha,
        )
    else:
        # Non-RA estimators require propensity score model for weighting or matching.
        ps_controls_final = ps_controls if ps_controls is not None else controls
        
        if estimator_lower == 'ipw':
            ipw_result = estimate_ipw(
                data=firstpost_data,
                y='ydot_postavg',
                d='d_',
                propensity_controls=ps_controls_final,
                trim_threshold=trim_threshold,
                alpha=alpha,
                return_diagnostics=return_diagnostics,
                # Common timing mode bypasses cohort-specific logic.
                gvar_col=None,
                ivar_col=None,
                cohort_g=None,
                period_r=None,
            )
            results_dict = _convert_ipw_result_to_dict(
                ipw_result, alpha, vce, cluster_var, controls, ps_controls_final
            )
        elif estimator_lower == 'ipwra':
            ipwra_result = estimate_ipwra(
                data=firstpost_data,
                y='ydot_postavg',
                d='d_',
                controls=controls,
                propensity_controls=ps_controls_final,
                trim_threshold=trim_threshold,
                alpha=alpha,
                return_diagnostics=return_diagnostics,
                # Common timing mode bypasses cohort-specific logic.
                gvar_col=None,
                ivar_col=None,
                cohort_g=None,
                period_r=None,
            )
            results_dict = _convert_ipwra_result_to_dict(
                ipwra_result, alpha, vce, cluster_var, controls
            )
        elif estimator_lower == 'psm':
            psm_result = estimate_psm(
                data=firstpost_data,
                y='ydot_postavg',
                d='d_',
                propensity_controls=ps_controls_final,
                n_neighbors=n_neighbors,
                caliper=caliper,
                with_replacement=with_replacement,
                match_order=match_order,
                alpha=alpha,
                return_diagnostics=return_diagnostics,
                # Common timing mode bypasses cohort-specific logic.
                gvar_col=None,
                ivar_col=None,
                cohort_g=None,
                period_r=None,
            )
            results_dict = _convert_psm_result_to_dict(
                psm_result, alpha, vce, cluster_var, controls
            )
        
        # Store estimator-specific diagnostics if requested.
        if return_diagnostics:
            metadata['estimator_diagnostics'] = results_dict.get('diagnostics')

    # Construct human-readable period labels preserving numeric precision.
    # Integer years display without decimals for cleaner output formatting.
    if isinstance(tvar, str):
        period_labels = {}
        for t, year in data_transformed.groupby('tindex')[tvar].first().items():
            if pd.notna(year):
                if year == int(year):
                    period_labels[t] = str(int(year))
                else:
                    # Preserve decimal part for non-integer years.
                    period_labels[t] = str(year)
            else:
                period_labels[t] = f"T{t}"
    else:
        year_var, quarter_var = tvar[0], tvar[1]
        period_labels = {}
        for t in data_transformed['tindex'].unique():
            row = data_transformed[data_transformed['tindex'] == t].iloc[0]
            year_val = row[year_var]
            quarter_val = row[quarter_var]
            if pd.notna(year_val) and pd.notna(quarter_val):
                # Preserve non-integer values for display precision.
                year_str = str(int(year_val)) if year_val == int(year_val) else str(year_val)
                quarter_str = str(int(quarter_val)) if quarter_val == int(quarter_val) else str(quarter_val)
                period_labels[t] = f"{year_str}q{quarter_str}"
            else:
                period_labels[t] = f"T{t}"

    Tmax = int(data_transformed['tindex'].max())
    controls_spec = results_dict.get('controls_spec', None)

    # Estimate period-specific effects using the appropriate estimator.
    if estimator_lower == 'ra':
        # RA uses OLS-based period effect estimation.
        period_df = estimation.estimate_period_effects(
            data=data_transformed,
            ydot='ydot',
            d='d_',
            tindex='tindex',
            tpost1=metadata['tpost1'],
            Tmax=Tmax,
            controls_spec=controls_spec,
            vce=vce,
            cluster_var=cluster_var,
            period_labels=period_labels,
            alpha=alpha,
        )
    else:
        # IPW/IPWRA/PSM use propensity-based period effect estimation.
        period_df = _estimate_period_effects_ipw(
            data=data_transformed,
            ydot='ydot',
            d='d_',
            tindex='tindex',
            tpost1=metadata['tpost1'],
            Tmax=Tmax,
            estimator=estimator_lower,
            controls=controls,
            ps_controls=ps_controls_final,
            trim_threshold=trim_threshold,
            n_neighbors=n_neighbors,
            caliper=caliper,
            with_replacement=with_replacement,
            match_order=match_order,
            period_labels=period_labels,
            alpha=alpha,
        )

    # Combine average and period-specific effects into unified output.
    avg_row = pd.DataFrame([{
        'period': 'average',
        'tindex': '-',
        'beta': results_dict['att'],
        'se': results_dict['se_att'],
        'ci_lower': results_dict['ci_lower'],
        'ci_upper': results_dict['ci_upper'],
        'tstat': results_dict['t_stat'],
        'pval': results_dict['pvalue'],
        'N': results_dict['nobs']
    }])

    avg_row['is_avg'] = True
    period_df['is_avg'] = False

    att_by_period = pd.concat([avg_row, period_df], ignore_index=True)
    att_by_period = att_by_period.sort_values(
        ['is_avg', 'tindex'], ascending=[False, True]
    )

    att_by_period = att_by_period.drop(columns=['is_avg']).reset_index(drop=True)
    att_by_period['tindex'] = att_by_period['tindex'].astype(str)
    
    att_by_period = att_by_period[[
        'period', 'tindex', 'beta', 'se', 'ci_lower', 'ci_upper', 'tstat', 'pval', 'N'
    ]]

    # Initialize RI variables before conditional execution block.
    ri_result = None
    actual_seed = None
    
    if ri:
        # Handle legacy riseed parameter with explicit type validation.
        # Invalid input raises an error rather than silently using a random seed.
        if seed is None and 'riseed' in kwargs:
            riseed_val = kwargs['riseed']
            if riseed_val is not None:
                if isinstance(riseed_val, (int, np.integer)):
                    seed = int(riseed_val)
                elif isinstance(riseed_val, str):
                    try:
                        seed = int(riseed_val)
                    except ValueError:
                        raise TypeError(
                            f"riseed must be an integer or integer-convertible string, "
                            f"got '{riseed_val}'. Use seed parameter for explicit control."
                        )
                else:
                    raise TypeError(
                        f"riseed must be an integer, got {type(riseed_val).__name__}. "
                        f"Use seed parameter for explicit control."
                    )

        actual_seed = seed if seed is not None else _generate_ri_seed()

        firstpost_df = data_transformed.loc[data_transformed['firstpost']].copy()
        if metadata.get('id_mapping') is not None:
            firstpost_df.attrs['id_mapping'] = metadata['id_mapping']

        try:
            ri_result = randomization_inference(
                firstpost_df=firstpost_df,
                y_col='ydot_postavg',
                d_col='d_',
                ivar=ivar,
                rireps=rireps,
                seed=actual_seed,
                att_obs=results_dict['att'],
                ri_method=ri_method,
                controls=controls,
            )
        except (RandomizationError, ValueError, np.linalg.LinAlgError) as e:
            # RandomizationError: RI-specific failures (insufficient data, invalid params)
            # ValueError: Data issues during resampling
            # LinAlgError: Singular matrix in permuted regressions
            warnings.warn(
                f"Randomization inference failed: {type(e).__name__}: {e}. "
                f"ATT estimation results are still valid.",
                NumericalWarning,
                stacklevel=3
            )
            ri_result = {
                'p_value': np.nan,
                'ri_method': ri_method,
                'ri_valid': 0,
                'ri_failed': -1,
                'ri_error': str(e),
            }

    results_dict['alpha'] = alpha
    metadata['alpha'] = alpha

    results = LWDIDResults(results_dict, metadata, att_by_period)

    results.data = data_transformed
    if metadata.get('id_mapping') is not None:
        results.data.attrs['id_mapping'] = metadata['id_mapping']

    if ri:
        results.ri_pvalue = ri_result['p_value']
        results.rireps = int(rireps)
        results.ri_seed = int(actual_seed)
        results.ri_method = ri_result['ri_method']
        results.ri_valid = ri_result['ri_valid']
        results.ri_failed = ri_result['ri_failed']
        results.ri_error = ri_result.get('ri_error', None)

    if graph:
        try:
            results.plot(gid=gid, graph_options=graph_options)
        except Exception as e:
            warnings.warn(
                f"Plotting failed: {type(e).__name__}: {str(e)}. "
                f"The estimation results are unaffected.",
                DataWarning,
                stacklevel=3  # _lwdid_classic() is level 2, so +1 to point to user code
            )

    # Attach structured diagnostic records from the warning registry.
    # Common timing mode typically has few warnings, but the registry
    # still provides a consistent interface for post-hoc inspection.
    results._diagnostics = registry.get_diagnostics()

    return results


def _validate_control_group_for_aggregate(
    aggregate: str,
    control_group: str,
    has_never_treated: bool,
    n_never_treated: int = 0
) -> tuple:
    """
    Validate and adjust control group strategy based on aggregation level.

    Cohort and overall aggregation require never-treated units as a consistent
    reference group across different treatment cohorts. This function auto-
    switches to 'never_treated' when needed and validates that sufficient
    never-treated units exist.

    Parameters
    ----------
    aggregate : {'none', 'cohort', 'overall'}
        Aggregation level for effect estimates.
    control_group : {'never_treated', 'not_yet_treated', 'all_others'}
        Requested control group composition strategy.
    has_never_treated : bool
        Whether the data contains any never-treated units.
    n_never_treated : int
        Number of never-treated units in the data.

    Returns
    -------
    control_group_used : str
        The control group strategy that will be used.
    warning_msg : str or None
        Warning message if auto-switching occurred, None otherwise.

    Raises
    ------
    NoNeverTreatedError
        If aggregation requires never-treated units but none exist.
    """
    warning_msg = None
    control_group_used = control_group

    if aggregate in ('cohort', 'overall'):
        if control_group != 'never_treated':
            warning_msg = (
                f"{aggregate} effect estimation requires never_treated control group, "
                f"automatically switched from '{control_group}' to 'never_treated'."
            )
            logger.info(warning_msg)
            warnings.warn(warning_msg, DataWarning, stacklevel=4)
            control_group_used = 'never_treated'

        if not has_never_treated:
            raise NoNeverTreatedError(
                f"Cannot estimate {aggregate} effect: no never-treated units in data.\n"
                f"Reason: {aggregate} effect requires NT units as a unified reference baseline.\n"
                f"  - Cohort effect: Different cohorts' transformations use different pre-treatment periods, "
                f"only NT units can provide a consistent reference.\n"
                f"  - Overall effect: NT units are needed to compute weighted transformations across all cohorts.\n"
                f"Suggestion: Use aggregate='none' to estimate (g,r)-specific effects, which can use not-yet-treated control group."
            )

        if n_never_treated < 2:
            warnings.warn(
                f"Number of never-treated units is too few (N={n_never_treated}), "
                f"inference results may be unreliable. Recommended N_NT >= 2.",
                SmallSampleWarning,
                stacklevel=4
            )

    return control_group_used, warning_msg


def _lwdid_staggered(
    data: pd.DataFrame,
    y: str,
    ivar: str,
    tvar: str | list[str],
    gvar: str,
    rolling: str,
    control_group: str,
    estimator: str,
    aggregate: str,
    ps_controls: list[str] | None,
    trim_threshold: float,
    return_diagnostics: bool,
    n_neighbors: int,
    caliper: float | None,
    with_replacement: bool,
    match_order: str,
    vce: str | None,
    controls: list[str] | None,
    cluster_var: str | None,
    alpha: float,
    ri: bool,
    rireps: int,
    seed: int | None,
    ri_method: str,
    graph: bool,
    gid: str | int | None,
    graph_options: dict | None,
    season_var: str | None = None,
    Q: int = 4,
    include_pretreatment: bool = False,
    pretreatment_test: bool = True,
    pretreatment_alpha: float = 0.05,
    exclude_pre_periods: int = 0,
    warning_registry: 'WarningRegistry | None' = None,
    **kwargs
) -> LWDIDResults:
    """
    Estimate treatment effects under staggered adoption design.

    Internal dispatcher that applies cohort-specific transformations, estimates
    cohort-time ATT effects, and aggregates results according to the specified
    aggregation level. This function handles all staggered adoption logic when
    the ``gvar`` parameter is provided to ``lwdid()``.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with unit, time, treatment cohort, and outcome variables.
    y : str
        Column name of the outcome variable.
    ivar : str
        Column name of the unit identifier.
    tvar : str or list of str
        Column name(s) of the time variable.
    gvar : str
        Column name indicating the first treatment period for each unit.
    rolling : {'demean', 'detrend', 'demeanq', 'detrendq'}
        Transformation method for removing pre-treatment patterns.
    control_group : {'never_treated', 'not_yet_treated', 'all_others'}
        Control group composition strategy.
    estimator : {'ra', 'ipw', 'ipwra', 'psm'}
        Treatment effect estimation method.
    aggregate : {'none', 'cohort', 'overall'}
        Aggregation level for the effect estimates.
    ps_controls : list of str or None
        Variables for the propensity score model.
    trim_threshold : float
        Propensity score trimming bound in (0, 0.5).
    return_diagnostics : bool
        Whether to include estimation diagnostics in results.
    n_neighbors : int
        Number of nearest neighbors for PSM matching.
    caliper : float or None
        Maximum propensity score distance for PSM matches.
    with_replacement : bool
        Whether PSM allows control unit reuse.
    match_order : str
        Order for processing treated units in PSM without replacement.
    vce : str or None
        Variance-covariance estimator type.
    controls : list of str or None
        Control variables for the outcome model.
    cluster_var : str or None
        Variable name for cluster-robust standard errors.
    alpha : float
        Significance level for confidence intervals.
    ri : bool
        Whether to perform randomization inference.
    rireps : int
        Number of randomization inference replications.
    seed : int or None
        Random seed for reproducibility.
    ri_method : {'bootstrap', 'permutation'}
        Randomization inference resampling method.
    graph : bool
        Whether to generate visualization plots.
    gid : str, int, or None
        Specific unit identifier to highlight in plots.
    graph_options : dict or None
        Additional plotting configuration options.
    **kwargs
        Additional keyword arguments (reserved for future use).

    Returns
    -------
    LWDIDResults
        Results object containing ATT estimates, standard errors, confidence
        intervals, cohort-time specific effects, and aggregated effects.
    """
    from .staggered import (
        transformations as stag_trans,
        estimation as stag_est,
        aggregation as stag_agg
    )

    if graph:
        warnings.warn(
            "Parameter 'graph=True' is not yet supported in staggered mode.\n"
            "To visualize results, use the `plot_event_study()` method on the returned "
            "LWDIDResults object: `results.plot_event_study()`",
            DataWarning,
            stacklevel=4
        )

    # Parameter validation.
    if ivar is None:
        raise ValueError("Staggered mode requires 'ivar' parameter (unit identifier column).")
    if tvar is None:
        raise ValueError("Staggered mode requires 'tvar' parameter (time variable column).")

    # Validate tvar list/tuple format for quarterly data support.
    if isinstance(tvar, (list, tuple)):
        if len(tvar) == 0:
            raise ValueError(
                "Parameter 'tvar' cannot be an empty list/tuple.\n"
                "For annual data: tvar='year_column'\n"
                "For quarterly data: tvar=['year_column', 'quarter_column']"
            )
        if len(tvar) > 2:
            raise ValueError(
                f"Parameter 'tvar' as a list must have 1-2 elements "
                f"[year_column, quarter_column], got {len(tvar)} elements: {tvar}"
            )

    if rolling is None:
        raise ValueError(
            "Staggered mode requires 'rolling' parameter.\n"
            "Valid values: 'demean', 'detrend', 'demeanq', 'detrendq'."
        )
    
    if not isinstance(rolling, str):
        raise TypeError(
            f"Parameter 'rolling' must be a string, got {type(rolling).__name__}. "
            f"Valid values: 'demean', 'detrend', 'demeanq', 'detrendq'."
        )
    
    rolling_lower = rolling.lower()

    VALID_ROLLING_STAGGERED = ('demean', 'detrend', 'demeanq', 'detrendq')
    if rolling_lower not in VALID_ROLLING_STAGGERED:
        raise ValueError(
            f"Invalid rolling='{rolling}' for staggered mode.\n"
            f"Valid values: {VALID_ROLLING_STAGGERED}"
        )

    # Validate aggregate parameter type before string operations.
    if aggregate is not None and not isinstance(aggregate, str):
        raise TypeError(
            f"Parameter 'aggregate' must be a string or None, "
            f"got {type(aggregate).__name__}.\n"
            f"Valid values: 'none', 'cohort', 'overall'"
        )

    VALID_AGGREGATE = ('none', 'cohort', 'overall')
    aggregate_lower = aggregate.lower() if aggregate else 'none'
    if aggregate_lower not in VALID_AGGREGATE:
        raise ValueError(
            f"Invalid aggregate='{aggregate}'.\n"
            f"Valid values: {VALID_AGGREGATE}\n"
            f"  - 'none': Return (g,r)-specific effects only\n"
            f"  - 'cohort': Aggregate to cohort-specific effects (τ_g)\n"
            f"  - 'overall': Aggregate to overall weighted effect (τ_ω)"
        )

    # Validate control_group parameter type before string operations.
    if control_group is not None and not isinstance(control_group, str):
        raise TypeError(
            f"Parameter 'control_group' must be a string or None, "
            f"got {type(control_group).__name__}.\n"
            f"Valid values: 'never_treated', 'not_yet_treated', 'all_others'"
        )

    VALID_CONTROL_GROUPS = ('never_treated', 'not_yet_treated', 'all_others')
    control_group_lower = control_group.lower() if control_group else 'not_yet_treated'
    if control_group_lower not in VALID_CONTROL_GROUPS:
        raise ValueError(
            f"Invalid control_group='{control_group}'.\n"
            f"Valid values: {VALID_CONTROL_GROUPS}\n"
            f"  - 'never_treated': Use only never-treated units as control\n"
            f"  - 'not_yet_treated': Use never-treated + not-yet-treated units as control\n"
            f"  - 'all_others': Use all non-cohort units as control (includes already-treated)"
        )

    # Validate estimator type before string operations.
    # Default to 'ra' when estimator is None for consistency.
    if estimator is not None and not isinstance(estimator, str):
        raise TypeError(
            f"estimator must be a string, got {type(estimator).__name__}.\n"
            f"Valid values: ('ra', 'ipw', 'ipwra', 'psm').\n"
            f"Example: lwdid(..., estimator='ipwra')"
        )
    estimator_lower = estimator.lower() if estimator else 'ra'
    VALID_ESTIMATORS = ('ra', 'ipw', 'ipwra', 'psm')
    if estimator_lower not in VALID_ESTIMATORS:
        raise ValueError(
            f"Invalid estimator='{estimator}'.\n"
            f"Valid values: {VALID_ESTIMATORS}"
        )

    # Validate control variables based on estimator type.
    # IPWRA: requires controls (outcome model), ps_controls optional (defaults to controls).
    # IPW/PSM: requires controls OR ps_controls (propensity score model only).
    if estimator_lower == 'ipwra' and not controls:
        raise ValueError(
            f"estimator='ipwra' requires 'controls' parameter for outcome model.\n"
            f"IPWRA (doubly robust) uses controls in both outcome regression and "
            f"propensity score model.\n"
            f"  - controls: Variables for outcome model E[Y|X,D=0]\n"
            f"  - ps_controls: Variables for propensity score P(D=1|X), defaults to controls"
        )
    elif estimator_lower in ('ipw', 'psm') and not controls and not ps_controls:
        raise ValueError(
            f"estimator='{estimator}' requires 'controls' or 'ps_controls' parameter.\n"
            f"IPW and PSM estimators need control variables for propensity score model.\n"
            f"  - Use 'controls' to specify variables (also used as ps_controls by default)\n"
            f"  - Or use 'ps_controls' to specify propensity score model variables directly"
        )

    # Validate alpha parameter type for confidence interval calculation.
    if not isinstance(alpha, (int, float, np.number)):
        raise TypeError(
            f"Parameter 'alpha' must be numeric, got {type(alpha).__name__}.\n"
            f"Example: alpha=0.05 for 95% confidence interval."
        )
    if hasattr(alpha, '__float__') and np.isnan(float(alpha)):
        raise ValueError("Parameter 'alpha' cannot be NaN.")
    if not (0 < alpha < 1):
        raise ValueError(
            f"Invalid alpha={alpha}.\n"
            f"alpha must be in (0, 1) for valid confidence interval calculation.\n"
            f"  - alpha=0.05 gives 95% CI (default)\n"
            f"  - alpha=0.10 gives 90% CI\n"
            f"  - alpha=0.01 gives 99% CI"
        )

    if vce is not None and vce.lower() == 'cluster' and cluster_var is None:
        raise InvalidParameterError(
            "vce='cluster' requires cluster_var parameter.\n"
            "Specify the column name for cluster-robust standard errors."
        )

    if estimator_lower in ('ipw', 'ipwra', 'psm'):
        # Propensity score trimming requires threshold in (0, 0.5) exclusive.
        # At threshold=0.5, scores are clipped to [0.5, 0.5], excluding all observations.
        if not (0 < trim_threshold < 0.5):
            raise ValueError(
                f"Invalid trim_threshold={trim_threshold}.\n"
                f"trim_threshold must be in (0, 0.5) for valid propensity score trimming.\n"
                f"  - trim_threshold=0.01 trims PS to [0.01, 0.99] (default)\n"
                f"  - trim_threshold=0.05 trims PS to [0.05, 0.95]\n"
                f"  - trim_threshold=0.5 is invalid as it would trim all observations"
            )

    if ri:
        # Accept both Python int and numpy integer types for rireps.
        if not isinstance(rireps, (int, np.integer)) or rireps < 1:
            raise ValueError(
                f"Invalid rireps={rireps}.\n"
                f"rireps must be a positive integer >= 1 when ri=True.\n"
                f"Recommended: rireps >= 500 for reliable p-values."
            )

        # Validate ri_method type before calling .lower()
        if ri_method is not None and not isinstance(ri_method, str):
            raise TypeError(
                f"Parameter 'ri_method' must be a string or None, "
                f"got {type(ri_method).__name__}.\n"
                f"Valid values: 'bootstrap', 'permutation'"
            )

        VALID_RI_METHODS = ('bootstrap', 'permutation')
        ri_method_lower = ri_method.lower() if ri_method else 'bootstrap'
        if ri_method_lower not in VALID_RI_METHODS:
            raise ValueError(
                f"Invalid ri_method='{ri_method}'.\n"
                f"Valid values: {VALID_RI_METHODS}\n"
                f"  - 'bootstrap': Bootstrap resampling (with replacement)\n"
                f"  - 'permutation': Fisher's exact permutation test (without replacement)"
            )
        # Normalize ri_method to lowercase for downstream use.
        ri_method = ri_method_lower

        # Validate seed type to ensure reproducibility.
        # Float values would be silently truncated by random.seed().
        if seed is not None:
            if not isinstance(seed, (int, np.integer)):
                raise TypeError(
                    f"Parameter 'seed' must be an integer or None, "
                    f"got {type(seed).__name__}.\n"
                    f"Example: lwdid(..., ri=True, seed=42)"
                )
            if seed < 0:
                raise ValueError(
                    f"Parameter 'seed' must be non-negative, got {seed}.\n"
                    f"Use a non-negative integer for reproducible results."
                )

    # Data validation and preparation.
    validation_result = validate_staggered_data(
        data=data,
        gvar=gvar,
        ivar=ivar,
        tvar=tvar,
        y=y,
        controls=controls
    )
    cohorts = validation_result['cohorts']
    has_never_treated = validation_result['n_never_treated'] > 0
    n_never_treated = validation_result['n_never_treated']
    T_max = validation_result['T_max']
    T_min = validation_result['T_min']
    cohort_sizes = validation_result['cohort_sizes']

    for warning in validation_result.get('warnings', []):
        warnings.warn(warning, DataWarning, stacklevel=3)

    control_group_used, switch_warning = _validate_control_group_for_aggregate(
        aggregate=aggregate_lower,
        control_group=control_group_lower,
        has_never_treated=has_never_treated,
        n_never_treated=n_never_treated
    )

    # Apply transformation.
    tvar_str = tvar if isinstance(tvar, str) else tvar[0]
    
    # Select appropriate transformation function based on rolling method.
    if rolling_lower == 'demean':
        transform_func = stag_trans.transform_staggered_demean
        data_transformed = transform_func(
            data=data,
            y=y,
            ivar=ivar,
            tvar=tvar_str,
            gvar=gvar,
            exclude_pre_periods=exclude_pre_periods,
        )
    elif rolling_lower == 'detrend':
        transform_func = stag_trans.transform_staggered_detrend
        data_transformed = transform_func(
            data=data,
            y=y,
            ivar=ivar,
            tvar=tvar_str,
            gvar=gvar,
            exclude_pre_periods=exclude_pre_periods,
        )
    elif rolling_lower == 'demeanq':
        data_transformed = stag_trans.transform_staggered_demeanq(
            data=data,
            y=y,
            ivar=ivar,
            tvar=tvar_str,
            gvar=gvar,
            season_var=season_var,
            Q=Q,
            exclude_pre_periods=exclude_pre_periods,
        )
    elif rolling_lower == 'detrendq':
        data_transformed = stag_trans.transform_staggered_detrendq(
            data=data,
            y=y,
            ivar=ivar,
            tvar=tvar_str,
            gvar=gvar,
            season_var=season_var,
            Q=Q,
            exclude_pre_periods=exclude_pre_periods,
        )

    # Estimate cohort-time effects.
    ps_controls_final = ps_controls if ps_controls is not None else controls
    cohort_time_effects = stag_est.estimate_cohort_time_effects(
        data_transformed=data_transformed,
        gvar=gvar,
        ivar=ivar,
        tvar=tvar_str,
        controls=controls,
        vce=vce,
        cluster_var=cluster_var,
        control_strategy=control_group_used,
        estimator=estimator_lower,
        transform_type=rolling_lower,
        alpha=alpha,
        propensity_controls=ps_controls_final,
        trim_threshold=trim_threshold,
        return_diagnostics=return_diagnostics,
        n_neighbors=n_neighbors,
        caliper=caliper,
        with_replacement=with_replacement,
        match_order=match_order,
        warning_registry=warning_registry,
    )

    att_by_cohort_time = pd.DataFrame([
        {
            'cohort': e.cohort,
            'period': e.period,
            'event_time': e.event_time,
            'att': e.att,
            'se': e.se,
            'ci_lower': e.ci_lower,
            'ci_upper': e.ci_upper,
            't_stat': e.t_stat,
            'pvalue': e.pvalue,
            'n_treated': e.n_treated,
            'n_control': e.n_control,
            'n_total': e.n_total,
            'df_resid': e.df_resid,
            'df_inference': e.df_inference,
        }
        for e in cohort_time_effects
    ])

    # Initialize aggregation variables for consistent scope across all code paths.
    att_by_cohort = None
    att_overall = None
    se_overall = None
    cohort_weights = {}
    t_stat_overall = None
    pvalue_overall = None
    ci_overall = (None, None)
    overall_effect = None
    cohort_effects = []  # Populated when aggregate in ('cohort', 'overall').

    if aggregate_lower in ('cohort', 'overall'):
        cohort_effects = stag_agg.aggregate_to_cohort(
            data_transformed=data_transformed,
            gvar=gvar,
            ivar=ivar,
            tvar=tvar_str,
            cohorts=cohorts,
            T_max=T_max,
            transform_type='demean' if rolling_lower in ('demean', 'demeanq') else 'detrend',
            vce=vce,
            cluster_var=cluster_var,
            alpha=alpha,
        )
        
        att_by_cohort = pd.DataFrame([
            {
                'cohort': c.cohort,
                'att': c.att,
                'se': c.se,
                'ci_lower': c.ci_lower,
                'ci_upper': c.ci_upper,
                't_stat': c.t_stat,
                'pvalue': c.pvalue,
                'n_periods': c.n_periods,
                'n_units': c.n_units
            }
            for c in cohort_effects
        ])

    if aggregate_lower == 'overall':
        overall_effect = stag_agg.aggregate_to_overall(
            data_transformed=data_transformed,
            gvar=gvar,
            ivar=ivar,
            tvar=tvar_str,
            transform_type='demean' if rolling_lower == 'demean' else 'detrend',
            vce=vce,
            cluster_var=cluster_var,
            alpha=alpha,
        )
        
        att_overall = overall_effect.att
        se_overall = overall_effect.se
        cohort_weights = overall_effect.cohort_weights
        t_stat_overall = overall_effect.t_stat
        pvalue_overall = overall_effect.pvalue
        ci_overall = (overall_effect.ci_lower, overall_effect.ci_upper)

    # Build results object.
    n_treated = int(sum(cohort_sizes.values()))

    # Compute n_control based on actual control group strategy used.
    # For 'never_treated': control group consists only of never-treated units.
    # For 'not_yet_treated': control group varies by (g,r) and includes NT + NYT units.
    if control_group_used == 'never_treated':
        n_control = n_never_treated
    else:  # 'not_yet_treated'
        # Extract maximum n_control from cohort-time effects.
        # This reflects the largest control group actually used in estimation.
        if len(cohort_time_effects) > 0:
            n_control = max(e.n_control for e in cohort_time_effects)
        else:
            # Fallback: use never-treated count as control group size.
            n_control = n_never_treated

    # Compute df_resid and df_inference based on aggregation level.
    # The degrees of freedom should reflect the underlying regression.
    # Fallback df calculation accounts for number of control variables.
    n_controls = len(controls) if controls else 0
    estimator_for_df = estimator_lower if estimator_lower else 'ra'
    if estimator_for_df == 'ra':
        # RA: intercept + D + K controls + K interactions = 2 + 2K
        df_fallback = n_treated + n_control - 2 - 2 * n_controls
    elif estimator_for_df == 'ipwra':
        # IPWRA: intercept + D + K controls = 2 + K
        df_fallback = n_treated + n_control - 2 - n_controls
    else:
        # IPW/PSM: df = n - 2 (controls only affect PS model, not outcome).
        df_fallback = n_treated + n_control - 2
    df_fallback = max(1, df_fallback)  # Ensure valid t-distribution quantiles.

    if aggregate_lower == 'overall' and overall_effect is not None:
        # For overall aggregation: use df from the overall regression
        df_resid_val = overall_effect.df_resid
        df_inference_val = overall_effect.df_inference
    elif aggregate_lower == 'cohort' and cohort_effects:
        # For cohort aggregation: use median df across cohort regressions.
        # Use nanmedian to handle cases where some cohorts failed estimation.
        valid_df_resid = [c.df_resid for c in cohort_effects if np.isfinite(c.df_resid)]
        valid_df_inference = [c.df_inference for c in cohort_effects if np.isfinite(c.df_inference)]
        if valid_df_resid:
            df_resid_val = int(np.median(valid_df_resid))
        else:
            # Use controls-aware fallback when no valid cohort effects exist.
            df_resid_val = df_fallback
        if valid_df_inference:
            df_inference_val = int(np.median(valid_df_inference))
        else:
            # Use controls-aware fallback when no valid cohort effects exist.
            df_inference_val = df_fallback
    elif len(cohort_time_effects) > 0:
        # For none or fallback: use median df across cohort-time regressions.
        # Use nanmedian to handle cases where some effects failed estimation.
        valid_df_resid = [e.df_resid for e in cohort_time_effects if np.isfinite(e.df_resid)]
        valid_df_inference = [e.df_inference for e in cohort_time_effects if np.isfinite(e.df_inference)]
        if valid_df_resid:
            df_resid_val = int(np.median(valid_df_resid))
        else:
            # Use controls-aware fallback when no valid effects exist.
            df_resid_val = df_fallback
        if valid_df_inference:
            df_inference_val = int(np.median(valid_df_inference))
        else:
            # Use controls-aware fallback when no valid effects exist.
            df_inference_val = df_fallback
    else:
        # Fallback to controls-aware formula when no effects available.
        df_resid_val = df_fallback
        df_inference_val = df_fallback

    # Compute aggregated inference statistics for cohort-level aggregation.
    # Weighted average ATT and SE across cohorts uses n_units as weights.
    se_cohort_agg = None
    att_cohort_agg = None
    t_stat_cohort_agg = None
    pvalue_cohort_agg = None
    ci_cohort_agg = (None, None)

    if aggregate_lower == 'cohort' and cohort_effects:
        # Filter valid cohort effects with finite ATT and SE values.
        valid_cohorts = [c for c in cohort_effects if np.isfinite(c.att) and np.isfinite(c.se)]
        
        if valid_cohorts:
            # Compute n_units-weighted average ATT across cohorts.
            total_units = sum(c.n_units for c in valid_cohorts)
            if total_units > 0:
                weights = np.array([c.n_units / total_units for c in valid_cohorts])
                atts = np.array([c.att for c in valid_cohorts])
                ses = np.array([c.se for c in valid_cohorts])
                
                # Weighted average ATT: τ̂ = Σ(w_g × τ̂_g)
                att_cohort_agg = float(np.sum(weights * atts))
                
                # SE via delta method assuming independent cohort estimates:
                # Var(τ̂) = Σ(w_g² × Var(τ̂_g)) = Σ(w_g² × SE_g²)
                # SE(τ̂) = sqrt(Σ(w_g² × SE_g²))
                var_agg = np.sum(weights**2 * ses**2)
                se_cohort_agg = float(np.sqrt(var_agg))
                
                # Compute t-statistic and p-value using aggregated df.
                if se_cohort_agg > 0 and df_inference_val > 0:
                    t_stat_cohort_agg = att_cohort_agg / se_cohort_agg
                    pvalue_cohort_agg = 2 * scipy.stats.t.sf(abs(t_stat_cohort_agg), df_inference_val)
                    
                    # Confidence interval.
                    t_crit = scipy.stats.t.ppf(1 - alpha / 2, df_inference_val)
                    ci_cohort_agg = (
                        att_cohort_agg - t_crit * se_cohort_agg,
                        att_cohort_agg + t_crit * se_cohort_agg
                    )
        else:
            # All cohort-level ATT estimates are invalid (NaN or infinite).
            n_total_cohorts = len(cohort_effects)
            warnings.warn(
                f"All {n_total_cohorts} cohort-level ATT estimates are invalid (NaN or infinite). "
                f"Cohort-aggregated statistics (att_cohort_agg, se_cohort_agg, etc.) remain None. "
                f"Possible causes: insufficient sample size, numerical instability, or data quality issues.",
                NumericalWarning,
                stacklevel=3
            )

    # Compute fallback ATT from cohort-time effects if no higher-level aggregation.
    # This is used when aggregate='none' or when aggregation fails.
    att_cohort_time_fallback = None
    if att_overall is None and att_cohort_agg is None:
        if len(att_by_cohort_time) > 0 and att_by_cohort_time['att'].notna().any():
            valid_df = att_by_cohort_time.loc[att_by_cohort_time['att'].notna()]
            weights_sum = valid_df['n_treated'].sum()
            # ATT = E[Y(1) - Y(0) | D=1] requires D=1 observations.
            # If total n_treated is zero across all cohort-time effects, ATT is not identifiable.
            if weights_sum <= 0:
                raise ValueError(
                    "Cannot compute weighted average ATT: total n_treated across all "
                    "cohort-time effects is zero. ATT estimation requires at least one "
                    "treated unit with valid outcome. This may occur due to:\n"
                    "  1. Propensity score trimming excluded all treated units\n"
                    "  2. Missing outcome values for all treated units\n"
                    "  3. Data quality issues in treatment indicator or outcome variable\n"
                    "Check data quality or adjust trim_threshold parameter."
                )
            att_cohort_time_fallback = float(np.average(valid_df['att'], weights=valid_df['n_treated']))

    # =========================================================================
    # Pre-treatment Dynamics Estimation (when include_pretreatment=True)
    # =========================================================================
    att_pre_treatment_df = None
    parallel_trends_result = None
    
    if include_pretreatment:
        from .staggered.transformations_pre import (
            transform_staggered_demean_pre,
            transform_staggered_detrend_pre,
        )
        from .staggered.estimation_pre import (
            estimate_pre_treatment_effects,
            pre_treatment_effects_to_dataframe,
        )
        from .staggered.parallel_trends import run_parallel_trends_test
        
        # Apply pre-treatment transformation
        try:
            if rolling_lower in ('demean', 'demeanq'):
                data_pre_transformed = transform_staggered_demean_pre(
                    data=data_transformed,
                    y=y,
                    ivar=ivar,
                    tvar=tvar_str,
                    gvar=gvar,
                    never_treated_values=[0, np.inf],
                )
                pre_transform_type = 'demean'
            else:  # detrend, detrendq
                data_pre_transformed = transform_staggered_detrend_pre(
                    data=data_transformed,
                    y=y,
                    ivar=ivar,
                    tvar=tvar_str,
                    gvar=gvar,
                    never_treated_values=[0, np.inf],
                )
                pre_transform_type = 'detrend'
            
            # Estimate pre-treatment effects
            pre_treatment_effects = estimate_pre_treatment_effects(
                data_transformed=data_pre_transformed,
                gvar=gvar,
                ivar=ivar,
                tvar=tvar_str,
                controls=controls,
                vce=vce,
                cluster_var=cluster_var,
                control_strategy=control_group_used,
                never_treated_values=[0, np.inf],
                alpha=pretreatment_alpha,
                estimator=estimator_lower,
                transform_type=pre_transform_type,
                propensity_controls=ps_controls_final,
                trim_threshold=trim_threshold,
            )
            
            # Convert to DataFrame
            att_pre_treatment_df = pre_treatment_effects_to_dataframe(pre_treatment_effects)
            
            # Run parallel trends test if requested
            if pretreatment_test and len(pre_treatment_effects) > 0:
                parallel_trends_result = run_parallel_trends_test(
                    pre_treatment_effects=pre_treatment_effects,
                    alpha=pretreatment_alpha,
                    test_type='f',
                    min_pre_periods=2,
                )
            
            # Update data_transformed with pre-treatment columns for visualization
            data_transformed = data_pre_transformed
            
        except Exception as e:
            warnings.warn(
                f"Pre-treatment dynamics estimation failed: {e}. "
                f"Continuing without pre-treatment effects.",
                NumericalWarning,
                stacklevel=3
            )

    results_dict = {
        'is_staggered': True,
        'cohorts': cohorts,
        'cohort_sizes': cohort_sizes,
        'att_by_cohort_time': att_by_cohort_time,
        'att_by_cohort': att_by_cohort,
        'att_overall': att_overall,
        'se_overall': se_overall,
        'cohort_weights': cohort_weights,
        'ci_overall_lower': ci_overall[0],
        'ci_overall_upper': ci_overall[1],
        't_stat_overall': t_stat_overall,
        'pvalue_overall': pvalue_overall,
        'n_treated': n_treated,
        'n_control': n_control,
        'nobs': n_treated + n_control,
        'control_group': control_group,
        'control_group_used': control_group_used,
        'aggregate': aggregate,
        'estimator': estimator,
        'rolling': rolling,
        'n_never_treated': n_never_treated,
        'alpha': alpha,
        'never_treated_values': [0, np.inf],
        # Compute ATT with fallback hierarchy:
        # 1. att_overall (from aggregate='overall')
        # 2. att_cohort_agg (weighted average of cohort effects for aggregate='cohort')
        # 3. att_cohort_time_fallback (n_treated-weighted average across cohort-time effects)
        # Return None if no valid estimates exist (all NaN) to maintain type consistency
        # (None = "no estimate" vs np.nan = "undefined/failed estimate").
        'att': (
            att_overall if att_overall is not None
            else att_cohort_agg if att_cohort_agg is not None
            else att_cohort_time_fallback
        ),
        # SE with fallback: se_overall (overall aggregation) -> se_cohort_agg (cohort aggregation)
        'se_att': (
            se_overall if se_overall is not None
            else se_cohort_agg if se_cohort_agg is not None
            else np.nan
        ),
        't_stat': (
            t_stat_overall if t_stat_overall is not None
            else t_stat_cohort_agg if t_stat_cohort_agg is not None
            else np.nan
        ),
        'pvalue': (
            pvalue_overall if pvalue_overall is not None
            else pvalue_cohort_agg if pvalue_cohort_agg is not None
            else np.nan
        ),
        'ci_lower': (
            ci_overall[0] if ci_overall[0] is not None
            else ci_cohort_agg[0] if ci_cohort_agg[0] is not None
            else np.nan
        ),
        'ci_upper': (
            ci_overall[1] if ci_overall[1] is not None
            else ci_cohort_agg[1] if ci_cohort_agg[1] is not None
            else np.nan
        ),
        'df_resid': df_resid_val,
        'df_inference': df_inference_val,
        'vce_type': vce if vce else 'ols',
        'params': None,
        'bse': None,
        'vcov': None,
        'resid': None,
        # Pre-treatment dynamics
        'att_pre_treatment': att_pre_treatment_df,
        'parallel_trends_test': parallel_trends_result,
        'include_pretreatment': include_pretreatment,
    }

    metadata = {
        'is_staggered': True,
        'rolling': rolling,
        'control_group': control_group,
        'control_group_used': control_group_used,
        'aggregate': aggregate,
        'estimator': estimator,
        'cohorts': cohorts,
        'T_max': T_max,
        'T_min': T_min,
        'has_never_treated': has_never_treated,
        'n_never_treated': n_never_treated,
        'n_cohorts': len(cohorts),
        'vce': vce,
        'depvar': y,
        'K': 0,
        'tpost1': cohorts[0] if cohorts else 0,
        'N_treated': n_treated,
        'N_control': n_control,
        'alpha': alpha,
        'ivar': ivar,
        'gvar': gvar,
        'tvar': tvar,
    }

    results = LWDIDResults(
        results_dict, metadata, att_by_cohort_time,
        cohort_time_effects=cohort_time_effects,
    )
    results.data = data_transformed

    # Attach structured diagnostic records from the warning registry.
    # Diagnostics are always complete regardless of verbosity setting,
    # enabling post-hoc inspection via results.diagnostics.
    if warning_registry is not None:
        results._diagnostics = warning_registry.get_diagnostics()
    else:
        results._diagnostics = []

    # Randomization inference
    if ri:
        from .staggered.randomization import randomization_inference_staggered

        # Handle riseed parameter with type validation.
        # Invalid input raises explicit error rather than silently using random seed.
        if seed is None and 'riseed' in kwargs:
            riseed_val = kwargs['riseed']
            if riseed_val is not None:
                if isinstance(riseed_val, (int, np.integer)):
                    seed = int(riseed_val)
                elif isinstance(riseed_val, str):
                    try:
                        seed = int(riseed_val)
                    except ValueError:
                        raise TypeError(
                            f"riseed must be an integer or integer-convertible string, "
                            f"got '{riseed_val}'. Use seed parameter for explicit control."
                        )
                else:
                    raise TypeError(
                        f"riseed must be an integer, got {type(riseed_val).__name__}. "
                        f"Use seed parameter for explicit control."
                    )

        # Determine target statistic based on aggregation level.
        # For each aggregation type, select the first valid (non-NaN) ATT estimate
        # to ensure randomization inference uses a meaningful observed statistic.
        if aggregate_lower == 'overall' and att_overall is not None:
            ri_target = 'overall'
            ri_observed = att_overall
            target_cohort_ri = None
            target_period_ri = None
        elif aggregate_lower == 'cohort' and att_by_cohort is not None and len(att_by_cohort) > 0:
            # Select the first cohort with a valid (non-NaN) ATT estimate
            valid_cohorts = att_by_cohort[att_by_cohort['att'].notna()]
            if len(valid_cohorts) > 0:
                ri_target = 'cohort'
                ri_observed = valid_cohorts.iloc[0]['att']
                target_cohort_ri = int(valid_cohorts.iloc[0]['cohort'])
                target_period_ri = None
            else:
                warnings.warn(
                    "No valid cohort ATT estimates available for randomization inference. "
                    "All cohort-level ATT values are NaN.",
                    DataWarning,
                    stacklevel=3
                )
                # Set RI attributes to NaN instead of returning incomplete results.
                results.ri_pvalue = np.nan
                results.rireps = rireps
                results.ri_seed = seed if seed is not None else _generate_ri_seed()
                results.ri_method = ri_method
                results.ri_valid = 0
                results.ri_failed = -1
                results.ri_error = "No valid cohort ATT estimates available"
                results.ri_target = 'cohort'
                return results
        else:
            ri_target = 'cohort_time'
            if len(cohort_time_effects) > 0:
                # Select the first cohort-time effect with a valid (non-NaN) ATT
                valid_effects = [e for e in cohort_time_effects if pd.notna(e.att)]
                if len(valid_effects) > 0:
                    first_effect = valid_effects[0]
                    ri_observed = first_effect.att
                    target_cohort_ri = first_effect.cohort
                    target_period_ri = first_effect.period
                else:
                    warnings.warn(
                        "No valid cohort-time ATT estimates available for randomization inference. "
                        "All cohort-time ATT values are NaN.",
                        DataWarning,
                        stacklevel=3
                    )
                    # Set RI attributes to NaN instead of returning incomplete results.
                    results.ri_pvalue = np.nan
                    results.rireps = rireps
                    results.ri_seed = seed if seed is not None else _generate_ri_seed()
                    results.ri_method = ri_method
                    results.ri_valid = 0
                    results.ri_failed = -1
                    results.ri_error = "No valid cohort-time ATT estimates available"
                    results.ri_target = 'cohort_time'
                    return results
            else:
                warnings.warn(
                    "No available effect estimates, skipping randomization inference.",
                    DataWarning,
                    stacklevel=3
                )
                # Set RI attributes to NaN instead of returning incomplete results.
                results.ri_pvalue = np.nan
                results.rireps = rireps
                results.ri_seed = seed if seed is not None else _generate_ri_seed()
                results.ri_method = ri_method
                results.ri_valid = 0
                results.ri_failed = -1
                results.ri_error = "No available effect estimates"
                results.ri_target = 'cohort_time'
                return results

        actual_seed = seed if seed is not None else _generate_ri_seed()

        try:
            ri_result = randomization_inference_staggered(
                data=data,
                gvar=gvar,
                ivar=ivar,
                tvar=tvar_str,
                y=y,
                cohorts=cohorts,
                observed_att=ri_observed,
                target=ri_target,
                target_cohort=target_cohort_ri,
                target_period=target_period_ri,
                ri_method=ri_method,
                rireps=rireps,
                seed=actual_seed,
                rolling=rolling,
                controls=controls,
                vce=vce,
                cluster_var=cluster_var,
                n_never_treated=n_never_treated,
            )

            results.ri_pvalue = ri_result.p_value
            results.rireps = rireps
            results.ri_seed = actual_seed
            results.ri_method = ri_result.ri_method
            results.ri_valid = ri_result.ri_valid
            results.ri_failed = ri_result.ri_failed
            results.ri_target = ri_target

        except (RandomizationError, ValueError, np.linalg.LinAlgError) as e:
            # RandomizationError: RI-specific failures (insufficient data, invalid params)
            # ValueError: Data issues during resampling
            # LinAlgError: Singular matrix in permuted regressions
            warnings.warn(
                f"Randomization inference failed: {type(e).__name__}: {e}",
                NumericalWarning,
                stacklevel=3
            )
            results.ri_pvalue = np.nan
            results.ri_seed = actual_seed
            results.rireps = rireps
            results.ri_method = ri_method
            results.ri_valid = 0
            results.ri_failed = -1
            results.ri_error = str(e)
            results.ri_target = ri_target
    
    return results

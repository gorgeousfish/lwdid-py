"""
Unit-specific panel data transformations for difference-in-differences.

This module implements unit-specific outcome transformations that remove
pre-treatment heterogeneity from panel data. Transformation parameters are
estimated using only pre-treatment observations, then applied out-of-sample
to all periods including post-treatment.

The transformations convert panel difference-in-differences estimation into
cross-sectional treatment effects problems. Under no anticipation and parallel
trends assumptions, standard treatment effect estimators (regression adjustment,
inverse probability weighting, doubly robust, matching) can be applied to the
transformed outcomes.

Available Transformations
-------------------------
demean
    Removes unit-specific pre-treatment mean:

    .. math::

        \\dot{Y}_{it} = Y_{it} - \\bar{Y}_{i,pre}

    where :math:`\\bar{Y}_{i,pre} = T_0^{-1} \\sum_{s<g} Y_{is}`.
    Requires at least 1 pre-treatment period per unit.

detrend
    Removes unit-specific linear time trend:

    .. math::

        \\dot{Y}_{it} = Y_{it} - \\hat{\\alpha}_i - \\hat{\\beta}_i t

    where :math:`(\\hat{\\alpha}_i, \\hat{\\beta}_i)` are OLS estimates from
    pre-treatment data. Requires at least 2 pre-treatment periods per unit.

demeanq
    Removes unit-specific mean with quarterly seasonal fixed effects:

    .. math::

        \\dot{Y}_{it} = Y_{it} - \\hat{\\mu}_i - \\sum_{q=2}^{4} \\hat{\\gamma}_q D_q

    where :math:`D_q` are quarter dummies with the smallest observed quarter
    as reference category. Requires :math:`n_{pre} \\geq Q + 1` per unit.

detrendq
    Removes unit-specific linear trend with quarterly seasonal effects:

    .. math::

        \\dot{Y}_{it} = Y_{it} - \\hat{\\alpha}_i - \\hat{\\beta}_i t
                        - \\sum_{q=2}^{4} \\hat{\\gamma}_q D_q

    Requires :math:`n_{pre} \\geq Q + 2` per unit.

Notes
-----
The transformations eliminate unit-specific level differences or trends that
may be correlated with treatment assignment. By removing these pre-treatment
patterns, the parallel trends assumption becomes an assumption about the
transformed outcomes rather than the original levels.

Time centering is applied in detrending methods to improve numerical stability
of OLS estimation. This reduces the condition number of the design matrix
without affecting the final residuals, as centering is an affine transformation
that preserves predicted values.
"""

import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .exceptions import InsufficientPrePeriodsError
from .validation import validate_quarter_diversity, validate_quarter_coverage


# Threshold for detecting degenerate time variance in OLS estimation.
# Time series with variance below this value lack sufficient variation
# for reliable slope estimation and will produce numerically unstable results.
VARIANCE_THRESHOLD = 1e-10


def _compute_max_pre_tindex(
    data: pd.DataFrame, post: str, tindex: str, method: str
) -> int:
    """
    Compute the maximum pre-treatment time index.

    Identifies the last period before treatment onset, defined as
    :math:`K = \\max\\{t : \\text{post}_t = 0\\}`.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data containing post indicator and time index columns.
    post : str
        Column name of binary post-treatment indicator (0=pre, 1=post).
    tindex : str
        Column name of integer-valued time index.
    method : str
        Transformation method name for error message context.

    Returns
    -------
    int
        Maximum pre-treatment period index K.

    Raises
    ------
    InsufficientPrePeriodsError
        If no pre-treatment observations exist or all pre-treatment
        time index values are missing.
    """
    pre_tindex = data[data[post] == 0][tindex]

    if pre_tindex.empty:
        raise InsufficientPrePeriodsError(
            f"No pre-treatment observations found (post==0). "
            f"rolling('{method}') requires at least 1 pre-treatment period."
        )

    if pre_tindex.isna().all():
        raise InsufficientPrePeriodsError(
            f"All pre-treatment time index values are NaN. "
            f"rolling('{method}') requires valid time index values."
        )

    return int(pre_tindex.max())


def _validate_quarterly_transform_requirements(
    data: pd.DataFrame,
    ivar: str,
    tindex: str,
    post: str,
    quarter: str,
    y: str,
    transform_type: str,
    min_global_pre_periods: int,
) -> int:
    """
    Validate data requirements for quarterly seasonal transformations.

    Verifies that data satisfy the requirements for demeanq or detrendq
    transformations: sufficient global pre-treatment periods, adequate
    per-unit observations for model estimation, and complete quarter
    coverage in the pre-treatment period.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with unit, time, post indicator, and quarter columns.
    ivar : str
        Column name for unit identifier.
    tindex : str
        Column name for time index.
    post : str
        Column name for post-treatment indicator (0=pre, 1=post).
    quarter : str
        Column name for quarter variable (values in {1, 2, 3, 4}).
    y : str
        Column name for outcome variable.
    transform_type : str
        Transformation type: 'demeanq' or 'detrendq'.
    min_global_pre_periods : int
        Minimum required unique pre-treatment periods.

    Returns
    -------
    int
        Maximum pre-treatment time index K.

    Raises
    ------
    InsufficientPrePeriodsError
        If global pre-period count is below minimum, or if any unit has
        insufficient pre-period observations for reliable estimation.
    ValueError
        If quarter column contains values outside {1, 2, 3, 4}.

    Notes
    -----
    Parameter count differs by transformation type:

    - demeanq: :math:`Y \\sim 1 + \\text{quarter}` requires :math:`k = Q` parameters
    - detrendq: :math:`Y \\sim 1 + t + \\text{quarter}` requires :math:`k = Q + 1`

    where Q is the number of distinct quarters in the unit's pre-period.
    Reliable estimation requires :math:`n \\geq k + 1` to ensure at least
    one residual degree of freedom.
    """
    pre_data = data[data[post] == 0]
    n_pre_periods = pre_data[tindex].nunique()
    K = _compute_max_pre_tindex(data, post, tindex, transform_type)

    # Validate quarter values are in the expected range {1, 2, 3, 4}.
    # Non-standard values would create incorrect dummy variables in the model.
    quarter_values = data[quarter].dropna().unique()
    valid_quarters = {1, 2, 3, 4}

    # Handle both integer and float representations (e.g., 1.0, 2.0).
    try:
        quarter_int_values = {int(q) for q in quarter_values if q == int(q)}
        invalid_values = set(quarter_values) - {float(q) for q in quarter_int_values}
        out_of_range = quarter_int_values - valid_quarters
    except (ValueError, TypeError):
        invalid_values = set(quarter_values)
        out_of_range = set()
    
    if invalid_values or out_of_range:
        all_invalid = invalid_values | {float(q) for q in out_of_range}
        raise ValueError(
            f"Quarter column '{quarter}' contains invalid values: {sorted(all_invalid)}. "
            f"Expected integer values in {{1, 2, 3, 4}} representing calendar quarters.\n\n"
            f"If your quarter values use a different encoding (e.g., 0-3 or 'Q1'-'Q4'), "
            f"please recode them to 1-4 before calling lwdid()."
        )
    
    if n_pre_periods < min_global_pre_periods:
        raise InsufficientPrePeriodsError(
            f"rolling('{transform_type}') requires at least {min_global_pre_periods} "
            f"pre-treatment period(s). Found: {n_pre_periods} unique pre-treatment "
            f"period(s) (max tindex={K})."
        )
    
    # Model parameter offset: demeanq has k=Q, detrendq has k=Q+1 (adds trend).
    param_offset = 0 if transform_type == 'demeanq' else 1
    
    for unit_id in data[ivar].unique():
        unit_mask = (data[ivar] == unit_id)
        unit_data = data[unit_mask]
        unit_pre = unit_data[unit_data[post] == 0]
        unit_pre_count = len(unit_pre)
        
        if unit_pre_count < min_global_pre_periods:
            raise InsufficientPrePeriodsError(
                f"Unit {unit_id} has {'no' if unit_pre_count == 0 else f'only {unit_pre_count}'} "
                f"pre-period observation(s). rolling('{transform_type}') requires at least "
                f"{min_global_pre_periods} pre-treatment period(s) per unit."
            )
        
        # Build valid_mask to match OLS estimation (missing='drop').
        if transform_type == 'demeanq':
            valid_mask = unit_pre[y].notna() & unit_pre[quarter].notna()
        else:
            valid_mask = unit_pre[y].notna() & unit_pre[tindex].notna() & unit_pre[quarter].notna()
        
        n_valid = valid_mask.sum()
        n_unique_quarters = unit_pre.loc[valid_mask, quarter].nunique() if n_valid > 0 else 0
        n_params = n_unique_quarters + param_offset
        min_required = n_params + 1  # Require at least df = 1
        
        if unit_pre_count < min_required:
            model_desc = (
                f"y ~ 1 + i.quarter with k = {n_unique_quarters} parameters "
                f"(1 constant + {n_unique_quarters-1} quarter dummies)"
                if transform_type == 'demeanq' else
                f"y ~ 1 + tindex + i.quarter with k = 1 + 1 + ({n_unique_quarters}-1) = {n_params} parameters"
            )
            raise InsufficientPrePeriodsError(
                f"Unit {unit_id} has {unit_pre_count} pre-period observation(s) "
                f"with {n_unique_quarters} distinct quarter(s). "
                f"rolling('{transform_type}') requires at least {min_required} observations "
                f"to ensure df = n - k ≥ 1 for reliable statistical inference. "
                f"The {transform_type} method estimates a model {model_desc}."
            )
    
    validate_quarter_coverage(data, ivar, quarter, post)
    
    return K


def apply_rolling_transform(
    data: pd.DataFrame,
    y: str,
    ivar: str,
    tindex: str,
    post: str,
    rolling: str,
    tpost1: int,
    quarter: Optional[str] = None,
) -> pd.DataFrame:
    """
    Apply unit-specific transformation to panel data.

    Dispatches to the appropriate transformation method (demean, detrend,
    demeanq, or detrendq), computes post-treatment averages, and marks the
    cross-sectional regression sample.

    Transformation parameters are estimated from pre-treatment data only, then
    applied out-of-sample to all observations to obtain residualized outcomes.
    This out-of-sample application is essential for valid causal inference.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format with one row per unit-period observation.
    y : str
        Column name of outcome variable.
    ivar : str
        Column name of unit identifier.
    tindex : str
        Column name of integer-valued time index.
    post : str
        Column name of binary post-treatment indicator (0=pre, 1=post).
    rolling : {'demean', 'detrend', 'demeanq', 'detrendq'}
        Transformation method to apply.
    tpost1 : int
        First post-treatment period index. Identifies the cross-sectional
        regression sample (firstpost=True for observations at tindex==tpost1).
    quarter : str, optional
        Column name of quarter indicator (values in {1, 2, 3, 4}).
        Required for quarterly methods (demeanq, detrendq).

    Returns
    -------
    pd.DataFrame
        Copy of input data with added columns:

        - ``ydot``: Transformed (residualized) outcome for each observation.
        - ``ydot_postavg``: Post-treatment average of ydot per unit.
        - ``firstpost``: Boolean indicator for cross-sectional regression sample.

    Raises
    ------
    InsufficientPrePeriodsError
        If any unit has insufficient pre-treatment observations for the
        chosen transformation method.
    ValueError
        If rolling method is invalid or quarterly method lacks quarter column.

    See Also
    --------
    _demean_transform : Unit-specific demeaning implementation.
    _detrend_transform : Unit-specific detrending implementation.
    demeanq_unit : Seasonal demeaning for a single unit.
    detrendq_unit : Seasonal detrending for a single unit.
    """
    data = data.copy()
    
    if rolling == 'demean':
        data = _demean_transform(data, y, ivar, post)
    elif rolling == 'detrend':
        data = _detrend_transform(data, y, ivar, tindex, post)
    elif rolling == 'demeanq':
        if quarter is None:
            raise ValueError(
                "rolling='demeanq' requires the 'quarter' parameter. "
                "Please specify the column name containing quarter values (1-4).\n\n"
                "Example: lwdid(..., rolling='demeanq', quarter='qtr')"
            )

        # Seasonal dummy coding requires standard quarter values for proper identification.
        quarter_values = data[quarter].dropna().unique()
        valid_quarters = {1, 2, 3, 4}
        invalid_values = set(quarter_values) - valid_quarters
        if invalid_values:
            raise ValueError(
                f"Quarter column '{quarter}' contains invalid values: {sorted(invalid_values)}. "
                f"Expected values in {{1, 2, 3, 4}}.\n\n"
                f"If your quarter column uses different encoding (e.g., 0-3 or 'Q1'-'Q4'), "
                f"please recode it to 1-4 before calling lwdid()."
            )

        _validate_quarterly_transform_requirements(
            data, ivar, tindex, post, quarter, y,
            transform_type='demeanq', min_global_pre_periods=1
        )

        data['ydot'] = np.nan
        for unit_id in data[ivar].unique():
            unit_mask = (data[ivar] == unit_id)
            unit_data = data[unit_mask].copy()
            _, ydot = demeanq_unit(unit_data, y, quarter, post)
            data.loc[unit_mask, 'ydot'] = ydot

    elif rolling == 'detrendq':
        if quarter is None:
            raise ValueError(
                "rolling='detrendq' requires the 'quarter' parameter. "
                "Please specify the column name containing quarter values (1-4).\n\n"
                "Example: lwdid(..., rolling='detrendq', quarter='qtr')"
            )

        _validate_quarterly_transform_requirements(
            data, ivar, tindex, post, quarter, y,
            transform_type='detrendq', min_global_pre_periods=2
        )

        data['ydot'] = np.nan
        for unit_id in data[ivar].unique():
            unit_mask = (data[ivar] == unit_id)
            unit_data = data[unit_mask].copy()
            _, ydot = detrendq_unit(unit_data, y, tindex, quarter, post)
            data.loc[unit_mask, 'ydot'] = ydot

    else:
        raise ValueError(f"Invalid rolling method: {rolling}")

    # Post-treatment average aggregates period-specific effects into a single ATT.
    post_data = data[data[post] == 1]
    ydot_postavg_map = post_data.groupby(ivar)['ydot'].mean()
    data['ydot_postavg'] = data[ivar].map(ydot_postavg_map)

    # First post-treatment observation identifies cross-sectional regression sample.
    # One row per unit enables standard treatment effect estimation.
    data['firstpost'] = (
        (data[tindex] == tpost1) &
        data['ydot_postavg'].notna()
    )

    return data


def _demean_transform(
    data: pd.DataFrame, y: str, ivar: str, post: str
) -> pd.DataFrame:
    """
    Apply unit-specific demeaning transformation to all units.

    Computes :math:`\\dot{Y}_{it} = Y_{it} - \\bar{Y}_{i,pre}` for all
    observations, where :math:`\\bar{Y}_{i,pre}` is the pre-treatment mean:

    .. math::

        \\bar{Y}_{i,pre} = T_{0i}^{-1} \\sum_{t: \\text{post}=0} Y_{it}

    This transformation removes unit-specific level heterogeneity that may
    be correlated with treatment assignment.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data containing outcome and post indicator columns.
    y : str
        Column name of outcome variable.
    ivar : str
        Column name of unit identifier.
    post : str
        Column name of binary post-treatment indicator (0=pre, 1=post).

    Returns
    -------
    pd.DataFrame
        Input data with ``ydot`` column containing demeaned outcomes.

    Raises
    ------
    InsufficientPrePeriodsError
        If any unit has no pre-treatment observations.

    See Also
    --------
    _detrend_transform : Removes unit-specific linear trends.
    """
    data['ydot'] = np.nan

    for unit_id in data[ivar].unique():
        mask_pre = (data[ivar] == unit_id) & (data[post] == 0)
        unit_pre_count = mask_pre.sum()

        if unit_pre_count < 1:
            raise InsufficientPrePeriodsError(
                f"Unit {unit_id} has no pre-treatment observations. "
                f"rolling('demean') requires at least 1 pre-treatment period per unit."
            )

        y_pre_mean = data.loc[mask_pre, y].mean()

        # All-NaN outcomes would silently propagate to transformed values.
        if pd.isna(y_pre_mean):
            warnings.warn(
                f"Unit {unit_id}: all pre-treatment y values are NaN. "
                f"Transformed outcome (ydot) will be NaN for this unit.",
                UserWarning,
                stacklevel=3
            )

        mask_unit = (data[ivar] == unit_id)
        data.loc[mask_unit, 'ydot'] = data.loc[mask_unit, y] - y_pre_mean

    return data


def detrend_unit(
    unit_data: pd.DataFrame, y: str, tindex: str, post: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove unit-specific linear time trend for a single unit.

    Estimates :math:`Y_{it} = \\alpha + \\beta t + \\varepsilon` via OLS using
    pre-treatment observations only, then computes out-of-sample residuals
    for all periods:

    .. math::

        \\dot{Y}_{it} = Y_{it} - \\hat{\\alpha} - \\hat{\\beta} t

    This transformation removes unit-specific linear trends that may violate
    the parallel trends assumption in levels.

    Parameters
    ----------
    unit_data : pd.DataFrame
        Data for a single unit containing all time periods.
    y : str
        Column name of outcome variable.
    tindex : str
        Column name of time index.
    post : str
        Column name of binary post-treatment indicator (0=pre, 1=post).

    Returns
    -------
    yhat_all : ndarray
        Fitted values :math:`\\hat{\\alpha} + \\hat{\\beta} t` for all periods.
        Returns NaN array if estimation fails.
    ydot : ndarray
        Detrended outcomes for all periods. Returns NaN array if estimation
        fails due to numerical issues.

    Notes
    -----
    Time centering at the pre-treatment mean improves numerical stability by
    reducing the condition number of :math:`X'X`. Centering is an affine
    transformation that preserves predicted values.

    See Also
    --------
    _detrend_transform : Apply detrending to all units in panel data.
    """
    unit_pre = unit_data[unit_data[post] == 0].copy()
    n_obs = len(unit_data)

    # Slope estimation requires variation in time; identical values make X'X singular.
    t_variance = unit_pre[tindex].var(ddof=0)
    if t_variance < VARIANCE_THRESHOLD:
        warnings.warn(
            "Degenerate time variance detected: all pre-treatment time values are "
            "identical, making OLS slope estimation impossible. Returning NaN.",
            UserWarning, stacklevel=2
        )
        return np.full(n_obs, np.nan), np.full(n_obs, np.nan)

    # OLS requires n > k for positive residual degrees of freedom.
    valid_mask = unit_pre[y].notna() & unit_pre[tindex].notna()
    n_valid = valid_mask.sum()
    n_params = 2  # intercept + slope
    if n_valid <= n_params:
        warnings.warn(
            f"Insufficient valid pre-treatment observations for OLS detrending: "
            f"found {n_valid} valid observations, require at least {n_params + 1} "
            f"(more than {n_params} parameters to ensure df >= 1). Returning NaN.",
            UserWarning, stacklevel=2
        )
        return np.full(n_obs, np.nan), np.full(n_obs, np.nan)

    # Centering reduces condition number from O(t_max^2) to O(1).
    t_mean = unit_pre[tindex].mean()
    t_centered_pre = unit_pre[tindex] - t_mean

    X_pre = sm.add_constant(t_centered_pre.values)
    y_pre = unit_pre[y].values
    model = sm.OLS(y_pre, X_pre, missing='drop').fit()

    # Invalid coefficients indicate numerical failure (e.g., collinearity).
    if np.isnan(model.params).any() or np.isinf(model.params).any():
        warnings.warn(
            f"OLS detrending produced invalid coefficients (NaN or Inf). "
            f"This may indicate insufficient time variation or constant "
            f"outcome values in pre-treatment data. Returning NaN for "
            f"detrended values.",
            UserWarning, stacklevel=3
        )
        return np.full(n_obs, np.nan), np.full(n_obs, np.nan)

    # Out-of-sample prediction uses same centering; predicted values are affine-invariant.
    t_centered_all = unit_data[tindex] - t_mean
    X_all = sm.add_constant(t_centered_all.values)
    yhat_all = model.predict(X_all)
    ydot = unit_data[y].values - yhat_all

    return yhat_all, ydot


def demeanq_unit(
    unit_data: pd.DataFrame,
    y: str,
    quarter: str,
    post: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove unit-specific mean with quarterly seasonal fixed effects.

    Estimates a seasonal mean model using pre-treatment observations:

    .. math::

        Y_{it} = \\mu + \\sum_{q=2}^{Q} \\gamma_q D_q + \\varepsilon_{it}

    where :math:`D_q` are quarter dummies. The smallest observed quarter serves
    as the reference category for identification.

    Parameters
    ----------
    unit_data : pd.DataFrame
        Data for a single unit containing all time periods.
    y : str
        Column name of outcome variable.
    quarter : str
        Column name of quarter indicator (values in {1, 2, 3, 4}).
    post : str
        Column name of binary post-treatment indicator (0=pre, 1=post).

    Returns
    -------
    yhat_all : ndarray
        Fitted values :math:`\\hat{\\mu} + \\sum_q \\hat{\\gamma}_q D_q` for
        all periods. Returns NaN array if estimation fails.
    ydot : ndarray
        Seasonally-adjusted demeaned outcomes for all periods. Returns NaN
        array if estimation fails due to numerical issues.

    Notes
    -----
    Using observed quarters rather than all four quarters as categories
    prevents rank-deficient design matrices when some quarters are absent
    from pre-treatment data.

    See Also
    --------
    detrendq_unit : Combines seasonal adjustment with linear trend removal.
    """
    unit_pre = unit_data[unit_data[post] == 0].copy()
    n_obs = len(unit_data)

    # OLS requires n > k for positive residual degrees of freedom.
    valid_mask = unit_pre[y].notna() & unit_pre[quarter].notna()
    n_valid = valid_mask.sum()
    n_quarters = unit_pre.loc[valid_mask, quarter].nunique() if n_valid > 0 else 0
    n_params = n_quarters  # intercept + (n_quarters - 1) dummies

    if n_valid <= n_params:
        warnings.warn(
            f"Insufficient valid pre-treatment observations for OLS seasonal demeaning: "
            f"found {n_valid} valid observations, require at least {n_params + 1} "
            f"(more than number of parameters). Returning NaN.",
            UserWarning, stacklevel=2
        )
        return np.full(n_obs, np.nan), np.full(n_obs, np.nan)

    # Restrict dummy coding to observed quarters to avoid rank deficiency.
    valid_pre = unit_pre[valid_mask]
    observed_quarters_pre = sorted(valid_pre[quarter].unique())
    q_categorical = pd.Categorical(unit_pre[quarter], categories=observed_quarters_pre)
    q_dummies_pre = pd.get_dummies(q_categorical, drop_first=True, prefix='q', dtype=float)

    X_pre = sm.add_constant(q_dummies_pre.values)
    y_pre = unit_pre[y].values
    model = sm.OLS(y_pre, X_pre, missing='drop').fit()

    # Invalid coefficients indicate numerical failure (e.g., collinearity).
    if np.isnan(model.params).any() or np.isinf(model.params).any():
        warnings.warn(
            f"OLS seasonal demeaning produced invalid coefficients (NaN or Inf). "
            f"This may indicate constant outcome values or collinearity in "
            f"pre-treatment data. Returning NaN for transformed values.",
            UserWarning, stacklevel=3
        )
        return np.full(n_obs, np.nan), np.full(n_obs, np.nan)

    # Prediction design matrix must match estimation design matrix structure.
    q_categorical_all = pd.Categorical(unit_data[quarter], categories=observed_quarters_pre)
    q_dummies_all = pd.get_dummies(q_categorical_all, drop_first=True, prefix='q', dtype=float)
    
    # Handle quarter mismatch between pre and post periods:
    # - Post-period new quarters get zero coefficients (extrapolation from model).
    # - Post-period missing quarters require column alignment.
    missing_cols = set(q_dummies_pre.columns) - set(q_dummies_all.columns)
    for col in missing_cols:
        q_dummies_all[col] = 0.0
    extra_cols = set(q_dummies_all.columns) - set(q_dummies_pre.columns)
    if extra_cols:
        q_dummies_all = q_dummies_all.drop(columns=list(extra_cols))
    q_dummies_all = q_dummies_all[q_dummies_pre.columns]

    X_all = sm.add_constant(q_dummies_all.values)
    yhat_all = model.predict(X_all)
    ydot = unit_data[y].values - yhat_all

    return yhat_all, ydot


def detrendq_unit(
    unit_data: pd.DataFrame,
    y: str,
    tindex: str,
    quarter: str,
    post: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove unit-specific linear trend with quarterly seasonal fixed effects.

    Estimates a combined trend and seasonal model using pre-treatment data:

    .. math::

        Y_{it} = \\alpha + \\beta t + \\sum_{q=2}^{Q} \\gamma_q D_q + \\varepsilon_{it}

    The smallest observed quarter serves as the reference category. Time is
    centered at its pre-treatment mean for numerical stability.

    Parameters
    ----------
    unit_data : pd.DataFrame
        Data for a single unit containing all time periods.
    y : str
        Column name of outcome variable.
    tindex : str
        Column name of time index.
    quarter : str
        Column name of quarter indicator (values in {1, 2, 3, 4}).
    post : str
        Column name of binary post-treatment indicator (0=pre, 1=post).

    Returns
    -------
    yhat_all : ndarray
        Fitted values for all periods. Returns NaN array if estimation fails.
    ydot : ndarray
        Seasonally-adjusted detrended outcomes for all periods. Returns NaN
        array if estimation fails due to numerical issues.

    Notes
    -----
    This transformation combines trend removal and seasonal adjustment,
    accounting for both unit-specific growth patterns and quarterly cycles.
    Time centering reduces the condition number of the design matrix without
    affecting predicted values.

    See Also
    --------
    demeanq_unit : Seasonal adjustment without trend removal.
    detrend_unit : Linear trend removal without seasonal adjustment.
    """
    unit_pre = unit_data[unit_data[post] == 0].copy()
    n_obs = len(unit_data)

    # Slope estimation requires variation in time; identical values make X'X singular.
    t_variance = unit_pre[tindex].var(ddof=0)
    if t_variance < VARIANCE_THRESHOLD:
        warnings.warn(
            "Degenerate time variance detected: all pre-treatment time values are "
            "identical, making OLS slope estimation impossible. Returning NaN.",
            UserWarning, stacklevel=2
        )
        return np.full(n_obs, np.nan), np.full(n_obs, np.nan)

    # OLS requires n > k for positive residual degrees of freedom.
    valid_mask = unit_pre[y].notna() & unit_pre[tindex].notna() & unit_pre[quarter].notna()
    n_valid = valid_mask.sum()
    n_quarters = unit_pre.loc[valid_mask, quarter].nunique() if n_valid > 0 else 0
    n_params = 1 + n_quarters  # intercept + slope + (n_quarters - 1) dummies

    if n_valid <= n_params:
        warnings.warn(
            f"Insufficient valid pre-treatment observations for OLS seasonal detrending: "
            f"found {n_valid} valid observations, require at least {n_params + 1} "
            f"(more than {n_params} parameters to ensure df >= 1). Returning NaN.",
            UserWarning, stacklevel=2
        )
        return np.full(n_obs, np.nan), np.full(n_obs, np.nan)

    # Center time at pre-treatment mean to improve numerical stability.
    # Reduces condition number of X'X from O(t_max^2) to O(1).
    t_mean = unit_pre[tindex].mean()
    t_centered_pre = unit_pre[tindex] - t_mean

    # Restrict dummy coding to observed quarters to avoid rank deficiency.
    # Unobserved quarters would create all-zero columns in the design matrix.
    valid_pre = unit_pre[valid_mask]
    observed_quarters_pre = sorted(valid_pre[quarter].unique())
    q_categorical = pd.Categorical(unit_pre[quarter], categories=observed_quarters_pre)
    q_dummies_pre = pd.get_dummies(q_categorical, drop_first=True, prefix='q', dtype=float)

    X_pre = np.column_stack([
        np.ones(len(unit_pre)),
        t_centered_pre.values,
        q_dummies_pre.values
    ])
    y_pre = unit_pre[y].values
    model = sm.OLS(y_pre, X_pre, missing='drop').fit()

    # Invalid coefficients indicate numerical failure (e.g., collinearity).
    if np.isnan(model.params).any() or np.isinf(model.params).any():
        warnings.warn(
            f"OLS seasonal detrending produced invalid coefficients (NaN or Inf). "
            f"This may indicate insufficient time variation, constant outcome values, "
            f"or collinearity in pre-treatment data. Returning NaN for transformed "
            f"values.",
            UserWarning, stacklevel=3
        )
        return np.full(n_obs, np.nan), np.full(n_obs, np.nan)

    # Out-of-sample prediction uses same centering; predicted values are affine-invariant.
    t_centered_all = unit_data[tindex] - t_mean

    # Prediction design matrix must match estimation design matrix structure.
    q_categorical_all = pd.Categorical(unit_data[quarter], categories=observed_quarters_pre)
    q_dummies_all = pd.get_dummies(q_categorical_all, drop_first=True, prefix='q', dtype=float)
    
    # Handle quarter mismatch between pre and post periods:
    # - Post-period new quarters get zero coefficients (extrapolation from model).
    # - Post-period missing quarters require column alignment.
    missing_cols = set(q_dummies_pre.columns) - set(q_dummies_all.columns)
    for col in missing_cols:
        q_dummies_all[col] = 0.0
    extra_cols = set(q_dummies_all.columns) - set(q_dummies_pre.columns)
    if extra_cols:
        q_dummies_all = q_dummies_all.drop(columns=list(extra_cols))
    q_dummies_all = q_dummies_all[q_dummies_pre.columns]

    X_all = np.column_stack([
        np.ones(len(unit_data)),
        t_centered_all.values,
        q_dummies_all.values
    ])
    yhat_all = model.predict(X_all)
    ydot = unit_data[y].values - yhat_all

    return yhat_all, ydot


def _detrend_transform(
    data: pd.DataFrame, y: str, ivar: str, tindex: str, post: str
) -> pd.DataFrame:
    """
    Apply unit-specific linear detrending transformation to all units.

    For each unit i, estimates a linear trend from pre-treatment data and
    computes out-of-sample residuals:

    .. math::

        \\dot{Y}_{it} = Y_{it} - \\hat{\\alpha}_i - \\hat{\\beta}_i t

    This transformation removes unit-specific linear trends that may violate
    the parallel trends assumption when trends differ across treatment groups.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data containing outcome, time index, and post indicator columns.
    y : str
        Column name of outcome variable.
    ivar : str
        Column name of unit identifier.
    tindex : str
        Column name of integer-valued time index.
    post : str
        Column name of binary post-treatment indicator (0=pre, 1=post).

    Returns
    -------
    pd.DataFrame
        Input data with ``ydot`` column containing detrended outcomes.

    Raises
    ------
    InsufficientPrePeriodsError
        If any unit has fewer than 2 pre-treatment observations.

    See Also
    --------
    detrend_unit : Detrending implementation for a single unit.
    _demean_transform : Removes unit-specific means instead of trends.
    """
    # Linear trend requires at least 2 time points for slope identification.
    K = _compute_max_pre_tindex(data, post, tindex, 'detrend')
    n_pre_periods = data[data[post] == 0][tindex].nunique()
    if n_pre_periods < 2:
        raise InsufficientPrePeriodsError(
            f"rolling('detrend') requires at least 2 pre-treatment periods "
            f"to estimate linear trend. Found: {n_pre_periods} unique pre-period(s) "
            f"with max tindex={K}."
        )

    data['ydot'] = np.nan

    for unit_id in data[ivar].unique():
        unit_mask = (data[ivar] == unit_id)
        unit_data = data[unit_mask].copy()
        unit_pre_count = (unit_data[post] == 0).sum()

        if unit_pre_count < 2:
            raise InsufficientPrePeriodsError(
                f"Unit {unit_id} has only {unit_pre_count} pre-period observation(s). "
                f"rolling('detrend') requires at least 2 pre-treatment periods per unit."
            )

        _, ydot = detrend_unit(unit_data, y, tindex, post)
        data.loc[unit_mask, 'ydot'] = ydot

    return data

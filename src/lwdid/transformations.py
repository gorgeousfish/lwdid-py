"""
Transformation Module

Implements transformation methods for the Lee and Wooldridge (2025)
difference-in-differences estimator.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .exceptions import InsufficientPrePeriodsError
from .validation import validate_quarter_diversity, validate_quarter_coverage


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
    Apply unit-specific transformation

    Transforms panel data by removing unit-specific pre-treatment patterns
    (means, linear trends, or seasonal effects) to enable cross-sectional
    difference-in-differences estimation. Implements Procedures 2.1 and 3.1
    from Lee and Wooldridge (2025).

    Parameters
    ----------
    data : pd.DataFrame
        Cleaned panel data containing d_, post_, and tindex columns (and tq
        for quarterly data). Structural assumptions (common timing, time
        continuity) are enforced in the validation stage.
    y : str
        Outcome variable column name.
    ivar : str
        Unit identifier column name.
    tindex : str
        Time index column name.
    post : str
        Post-treatment indicator column name.
    rolling : {'demean', 'detrend', 'demeanq', 'detrendq'}
        Transformation method:

        - 'demean': Unit-specific demeaning (Procedure 2.1)
        - 'detrend': Unit-specific detrending (Procedure 3.1)
        - 'demeanq': Demeaning with quarterly fixed effects
        - 'detrendq': Detrending with quarterly fixed effects

    tpost1 : int
        First post-treatment period index.
    quarter : str, optional
        Quarter variable column name (required for demeanq and detrendq).
        Values must be in {1, 2, 3, 4}.

    Returns
    -------
    pd.DataFrame
        Input data with additional columns:

        - 'ydot': Residualized outcome for all periods
        - 'ydot_postavg': Unit-level average of ydot over post-treatment periods
        - 'firstpost': Indicator for regression sample (one row per unit)

    Raises
    ------
    InsufficientPrePeriodsError
        Insufficient pre-treatment periods for chosen method.
    InsufficientQuarterDiversityError
        Quarterly diversity/coverage requirements not met for demeanq or detrendq.
    """
    if rolling == 'demean':
        data = _demean_transform(data, y, ivar, post)
    elif rolling == 'detrend':
        data = _detrend_transform(data, y, ivar, tindex, post)
    elif rolling == 'demeanq':
        # Require at least 1 pre-treatment period globally
        K = int(data[data[post] == 0][tindex].max())
        if K < 1:
            raise InsufficientPrePeriodsError(
                f"rolling('demeanq') requires at least 1 pre-treatment period. "
                f"Found: K={K} (T0={K})"
            )

        # Check each unit has sufficient pre-period observations for df ≥ 1
        # Model: y ~ 1 + i.quarter has k = q parameters (1 constant + q-1 quarter dummies)
        # Require n ≥ q + 1 to ensure df = n - k ≥ 1
        for unit_id in data[ivar].unique():
            unit_mask = (data[ivar] == unit_id)
            unit_data = data[unit_mask]
            unit_pre = unit_data[unit_data[post] == 0]
            unit_pre_count = len(unit_pre)

            if unit_pre_count < 1:
                raise InsufficientPrePeriodsError(
                    f"Unit {unit_id} has no pre-treatment observations. "
                    f"rolling('demeanq') requires at least 1 pre-treatment period per unit."
                )

            n_unique_quarters = unit_pre[quarter].nunique()
            min_required = n_unique_quarters + 1

            if unit_pre_count < min_required:
                raise InsufficientPrePeriodsError(
                    f"Unit {unit_id} has {unit_pre_count} pre-period observation(s) "
                    f"with {n_unique_quarters} distinct quarter(s). "
                    f"rolling('demeanq') requires at least {min_required} observations "
                    f"to ensure df = n - k ≥ 1 for reliable statistical inference. "
                    f"The demeanq method estimates a model y ~ 1 + i.quarter "
                    f"with k = {n_unique_quarters} parameters (1 constant + {n_unique_quarters-1} quarter dummies)."
                )

        # Validate that post-period quarters appear in pre-period
        validate_quarter_coverage(data, ivar, quarter, post)

        data['ydot'] = np.nan

        for unit_id in data[ivar].unique():
            unit_mask = (data[ivar] == unit_id)
            unit_data = data[unit_mask].copy()

            yhat, ydot = demeanq_unit(unit_data, y, quarter, post)
            data.loc[unit_mask, 'ydot'] = ydot
    elif rolling == 'detrendq':
        # Require at least 2 pre-treatment periods globally
        K = int(data[data[post] == 0][tindex].max())
        if K < 2:
            raise InsufficientPrePeriodsError(
                f"rolling('detrendq') requires at least 2 pre-treatment periods. "
                f"Found: K={K} (T0={K})"
            )

        # Check each unit has sufficient pre-period observations for rank identification
        # Model: y ~ 1 + tindex + i.quarter has k = 1 + q parameters
        # Require n ≥ 1 + q to avoid rank deficiency (df = n - k ≥ 0)
        for unit_id in data[ivar].unique():
            unit_mask = (data[ivar] == unit_id)
            unit_data = data[unit_mask]
            unit_pre = unit_data[unit_data[post] == 0]
            unit_pre_count = len(unit_pre)

            if unit_pre_count < 2:
                raise InsufficientPrePeriodsError(
                    f"Unit {unit_id} has only {unit_pre_count} pre-period observation(s). "
                    f"rolling('detrendq') requires at least 2 pre-treatment periods per unit."
                )

            n_unique_quarters = unit_pre[quarter].nunique()
            min_required = 1 + n_unique_quarters

            if unit_pre_count < min_required:
                raise InsufficientPrePeriodsError(
                    f"Unit {unit_id} has {unit_pre_count} pre-period observation(s) "
                    f"with {n_unique_quarters} distinct quarter(s). "
                    f"rolling('detrendq') requires at least {min_required} observations "
                    f"to avoid rank deficiency (df = n - k ≥ 0). "
                    f"The detrendq method estimates a model y ~ 1 + tindex + i.quarter "
                    f"with k = 1 + {n_unique_quarters} = {1 + n_unique_quarters} parameters."
                )

        # Validate that post-period quarters appear in pre-period
        validate_quarter_coverage(data, ivar, quarter, post)

        data['ydot'] = np.nan

        for unit_id in data[ivar].unique():
            unit_mask = (data[ivar] == unit_id)
            unit_data = data[unit_mask].copy()
            
            yhat, ydot = detrendq_unit(unit_data, y, tindex, quarter, post)
            data.loc[unit_mask, 'ydot'] = ydot
    else:
        raise ValueError(f"Invalid rolling method: {rolling}")
    
    # Compute post-period average of residualized outcome for each unit
    post_data = data[data[post] == 1]
    ydot_postavg_map = post_data.groupby(ivar)['ydot'].mean()
    data['ydot_postavg'] = data[ivar].map(ydot_postavg_map)
    
    # Mark cross-sectional regression sample (one row per unit)
    # Implements equation (2.13): regress ȳ̇ᵢ on 1, Dᵢ for i=1,...,N
    data['firstpost'] = False
    valid_units = data[data['ydot_postavg'].notna()].groupby(ivar).head(1).index
    data.loc[valid_units, 'firstpost'] = True
    
    return data


def _demean_transform(
    data: pd.DataFrame, y: str, ivar: str, post: str
) -> pd.DataFrame:
    """
    Unit-specific demeaning transformation (Procedure 2.1)

    Removes unit-specific pre-treatment means from all periods.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with pre-treatment indicator.
    y : str
        Outcome variable column name.
    ivar : str
        Unit identifier column name.
    post : str
        Post-treatment indicator column name.

    Returns
    -------
    pd.DataFrame
        Data with 'ydot' column added (residualized outcome for all periods).

    Raises
    ------
    InsufficientPrePeriodsError
        If any unit has no pre-treatment observations.

    Notes
    -----
    Computes residuals for all periods to enable plotting. Each unit requires
    at least 1 pre-treatment observation to compute Ȳ_{i,pre}.
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

        mask_unit = (data[ivar] == unit_id)
        data.loc[mask_unit, 'ydot'] = data.loc[mask_unit, y] - y_pre_mean

    return data


def detrend_unit(
    unit_data: pd.DataFrame, y: str, tindex: str, post: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove unit-specific linear trend estimated from pre-treatment periods
    
    Algorithm
    ---------
    1. Estimate linear trend from pre-treatment data: y_it = α_i + β_i·t + ε_it
    2. Extrapolate trend to all periods: ŷ_it = α̂_i + β̂_i·t
    3. Compute residuals: ẏ_it = y_it − ŷ_it
    
    Parameters
    ----------
    unit_data : pandas.DataFrame
        Data for a single unit.
    y : str
        Outcome variable column name.
    tindex : str
        Time index column name.
    post : str
        Post-treatment indicator column name.
    
    Returns
    -------
    tuple of numpy.ndarray
        (yhat_all, ydot) where yhat_all contains predicted values α̂_i + β̂_i·t
        and ydot contains residuals y_it − ŷ_it for all periods.
    """
    unit_pre = unit_data[unit_data[post] == 0].copy()
    
    X_pre = sm.add_constant(unit_pre[tindex].values)
    y_pre = unit_pre[y].values
    
    model = sm.OLS(y_pre, X_pre).fit()
    
    alpha_i = model.params[0]
    beta_i = model.params[1]
    
    X_all = sm.add_constant(unit_data[tindex].values)
    yhat_all = model.predict(X_all)
    
    ydot = unit_data[y].values - yhat_all
    
    return (yhat_all, ydot)


def demeanq_unit(
    unit_data: pd.DataFrame,
    y: str,
    quarter: str,
    post: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove unit-specific quarterly fixed effects estimated from pre-treatment periods
    
    Parameters
    ----------
    unit_data : pd.DataFrame
        Single unit's data (all time periods).
    y : str
        Outcome variable column name.
    quarter : str
        Quarter variable column name (values in {1, 2, 3, 4}).
    post : str
        Post-treatment indicator column name.
    
    Returns
    -------
    yhat_all : np.ndarray
        Predicted values (α̂_i + Σ γ̂_q·1{quarter=q}).
    ydot : np.ndarray
        Residuals y_it − ŷ_it for all periods.
    
    Notes
    -----
    Estimates y_it = α_i + Σ_{q=2}^4 γ_q·1{quarter=q} + ε_it using pre-treatment
    data, with quarter=1 as baseline. Extrapolates to all periods.
    """
    pre_mask = (unit_data[post] == 0)
    unit_pre = unit_data[pre_mask].copy()
    
    q_categorical = pd.Categorical(unit_pre[quarter], categories=[1, 2, 3, 4])
    q_dummies_pre = pd.get_dummies(q_categorical, drop_first=True, prefix='q', dtype=float)
    
    X_pre = sm.add_constant(q_dummies_pre.values)
    y_pre = unit_pre[y].values
    
    model = sm.OLS(y_pre, X_pre).fit()
    
    q_categorical_all = pd.Categorical(unit_data[quarter], categories=[1, 2, 3, 4])
    q_dummies_all = pd.get_dummies(q_categorical_all, drop_first=True, prefix='q', dtype=float)
    q_dummies_all = q_dummies_all[q_dummies_pre.columns]
    
    X_all = sm.add_constant(q_dummies_all.values)
    yhat_all = model.predict(X_all)
    
    ydot = unit_data[y].values - yhat_all
    
    return (yhat_all, ydot)


def detrendq_unit(
    unit_data: pd.DataFrame,
    y: str,
    tindex: str,
    quarter: str,
    post: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove unit-specific linear trend and quarterly fixed effects from pre-treatment periods
    
    Parameters
    ----------
    unit_data : pd.DataFrame
        Single unit's data (all time periods).
    y : str
        Outcome variable column name.
    tindex : str
        Time index column name.
    quarter : str
        Quarter variable column name.
    post : str
        Post-treatment indicator column name.
    
    Returns
    -------
    yhat_all : np.ndarray
        Predicted values (α̂_i + β̂_i·t + Σ γ̂_q·1{quarter=q}).
    ydot : np.ndarray
        Residuals y_it − ŷ_it for all periods.
    
    Notes
    -----
    Estimates y_it = α_i + β_i·t + Σ_{q=2}^4 γ_q·1{quarter=q} + ε_it using
    pre-treatment data. Extrapolates trend and seasonal effects to all periods.
    """
    pre_mask = (unit_data[post] == 0)
    unit_pre = unit_data[pre_mask].copy()
    
    q_categorical = pd.Categorical(unit_pre[quarter], categories=[1, 2, 3, 4])
    q_dummies_pre = pd.get_dummies(q_categorical, drop_first=True, prefix='q', dtype=float)
    
    X_pre = np.column_stack([
        np.ones(len(unit_pre)),
        unit_pre[tindex].values,
        q_dummies_pre.values
    ])
    
    y_pre = unit_pre[y].values
    
    model = sm.OLS(y_pre, X_pre).fit()
    
    q_categorical_all = pd.Categorical(unit_data[quarter], categories=[1, 2, 3, 4])
    q_dummies_all = pd.get_dummies(q_categorical_all, drop_first=True, prefix='q', dtype=float)
    q_dummies_all = q_dummies_all[q_dummies_pre.columns]
    
    X_all = np.column_stack([
        np.ones(len(unit_data)),
        unit_data[tindex].values,
        q_dummies_all.values
    ])
    
    yhat_all = model.predict(X_all)
    
    ydot = unit_data[y].values - yhat_all
    
    return (yhat_all, ydot)


def _detrend_transform(
    data: pd.DataFrame, y: str, ivar: str, tindex: str, post: str
) -> pd.DataFrame:
    """
    Unit-specific linear detrending transformation (Procedure 3.1)

    Removes unit-specific linear time trends estimated from pre-treatment periods.

    Per Lee and Wooldridge (2025) Procedure 3.1, this is a TRANSFORMATION step:
    1. For each unit i, estimate linear trend from pre-treatment data: Y_it ~ 1 + t
    2. Remove the estimated trend from all periods (pre and post)
    3. Statistical inference occurs in the subsequent cross-sectional regression

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with time index and post-treatment indicator.
    y : str
        Outcome variable column name.
    ivar : str
        Unit identifier column name.
    tindex : str
        Time index column name.
    post : str
        Post-treatment indicator column name.

    Returns
    -------
    pd.DataFrame
        Data with 'ydot' column added (residualized outcome for all periods).

    Notes
    -----
    Requires T₀ ≥ 2 to identify linear trend. Computes residuals for all periods
    to enable plotting. See Procedure 3.1 in Lee and Wooldridge (2025).
    """
    K = int(data[data[post] == 0][tindex].max())
    if K < 2:
        raise InsufficientPrePeriodsError(
            f"rolling('detrend') requires at least 2 pre-treatment periods "
            f"to estimate linear trend. Found: T0={K}"
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
        
        yhat, ydot = detrend_unit(unit_data, y, tindex, post)
        
        data.loc[unit_mask, 'ydot'] = ydot
    
    return data

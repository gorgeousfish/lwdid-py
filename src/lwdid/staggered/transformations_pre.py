"""
Pre-treatment period transformations for staggered difference-in-differences.

This module implements rolling transformations for pre-treatment periods.
Unlike post-treatment transformations that use past pre-treatment periods,
these transformations use **future** pre-treatment periods {t+1, ..., g-1}
to compute baselines.

Two transformation methods are provided:

- **Demeaning**: Subtracts the mean of future pre-treatment outcomes from
  the current period outcome. The anchor point t = g-1 is set to 0.

- **Detrending**: Removes unit-specific linear trends estimated from future
  pre-treatment data. Requires at least 2 future periods for OLS.

These transformations enable:

1. Event study visualization with pre-treatment effects
2. Parallel trends assumption testing
3. Detection of anticipation effects or dynamic selection

Notes
-----
The demeaning transformation computes:

.. math::

    \\dot{Y}_{itg} = Y_{it} - \\frac{1}{g-t-1} \\sum_{q=t+1}^{g-1} Y_{iq}

The detrending transformation computes:

.. math::

    \\ddot{Y}_{itg} = Y_{it} - \\hat{Y}_{itg}

where :math:`\\hat{Y}_{itg}` is the OLS-fitted value from regressing
:math:`Y_{iq}` on :math:`q` for :math:`q \\in \\{t+1, \\ldots, g-1\\}`.

At the anchor point t = g-1 (event time e = -1), the future window
{g, ..., g-1} is empty. By convention, the transformed outcome is set
to exactly 0.0, serving as the reference point for pre-treatment dynamics.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence


# =============================================================================
# Utility Functions
# =============================================================================


def get_pre_treatment_periods_for_cohort(
    cohort: int,
    T_min: int,
) -> list[int]:
    """
    Return list of pre-treatment periods for a treatment cohort.

    Generates the sequence {T_min, T_min+1, ..., g-1} representing all
    calendar periods strictly before the first treatment period g.

    Parameters
    ----------
    cohort : int
        Treatment cohort identifier (first treatment period g).
    T_min : int
        Minimum time period in the panel data.

    Returns
    -------
    list of int
        Pre-treatment periods in ascending order.

    See Also
    --------
    get_valid_periods_for_cohort : Get post-treatment periods for a cohort.
    """
    return list(range(int(T_min), int(cohort)))


def _compute_rolling_mean_future(
    unit_data: pd.DataFrame,
    y: str,
    tvar: str,
    period: int,
    cohort: int,
) -> float:
    """
    Compute mean of Y_{iq} for q in {period+1, ..., cohort-1}.

    Implements the rolling mean using future pre-treatment periods for
    the demeaning transformation in pre-treatment event study analysis.

    Parameters
    ----------
    unit_data : pd.DataFrame
        Time series observations for a single unit.
    y : str
        Outcome variable column name.
    tvar : str
        Time variable column name.
    period : int
        Current period t for which to compute the transformation.
    cohort : int
        Treatment cohort g defining the pre-treatment cutoff.

    Returns
    -------
    float
        Mean of future pre-treatment outcomes.
        Returns NaN if no valid future periods exist (except anchor point).

    Notes
    -----
    For the anchor point (period = cohort - 1), the future window is empty.
    This function returns NaN for empty windows; the caller should handle
    the anchor point convention separately.
    """
    # Future pre-treatment periods: {period+1, ..., cohort-1}
    # Empty window when period >= cohort-1 (anchor point case).
    if period + 1 >= cohort:
        return np.nan

    future_periods = range(period + 1, cohort)

    future_data = unit_data[unit_data[tvar].isin(future_periods)]
    future_values = future_data[y].dropna()

    if len(future_values) == 0:
        return np.nan

    return float(future_values.mean())


def _compute_rolling_trend_future(
    unit_data: pd.DataFrame,
    y: str,
    tvar: str,
    period: int,
    cohort: int,
) -> tuple[float, float]:
    """
    Fit OLS Y_q = A + B*q for q in {period+1, ..., cohort-1}.

    Implements rolling OLS trend estimation using future pre-treatment
    periods for the detrending transformation in pre-treatment event
    study analysis.

    Parameters
    ----------
    unit_data : pd.DataFrame
        Time series observations for a single unit.
    y : str
        Outcome variable column name.
    tvar : str
        Time variable column name.
    period : int
        Current period t for which to compute the fitted value.
    cohort : int
        Treatment cohort g defining the pre-treatment cutoff.

    Returns
    -------
    tuple of float
        (intercept A, slope B) from OLS fit.
        Returns (NaN, NaN) if fewer than 2 future periods exist.

    Notes
    -----
    OLS requires at least 2 data points to estimate both intercept and slope.
    For numerical stability, time indices are used directly without centering
    (centering is applied at the transformation level if needed).
    """
    # Future pre-treatment periods: {period+1, ..., cohort-1}
    future_periods = list(range(period + 1, cohort))

    if len(future_periods) < 2:
        return (np.nan, np.nan)

    future_data = unit_data[unit_data[tvar].isin(future_periods)].dropna(subset=[y])

    if len(future_data) < 2:
        return (np.nan, np.nan)

    t_vals = future_data[tvar].values.astype(float)
    y_vals = future_data[y].values.astype(float)

    # Check for constant time values (would cause collinearity)
    if np.std(t_vals) < 1e-10:
        return (np.nan, np.nan)

    try:
        # np.polyfit returns [slope, intercept] for degree 1
        B, A = np.polyfit(t_vals, y_vals, deg=1)
        return (float(A), float(B))
    except (np.linalg.LinAlgError, ValueError):
        # Numerical issues in OLS
        return (np.nan, np.nan)


# =============================================================================
# Main Transformation Functions
# =============================================================================


def transform_staggered_demean_pre(
    data: pd.DataFrame,
    y: str,
    ivar: str,
    tvar: str,
    gvar: str,
    never_treated_values: Sequence | None = None,
) -> pd.DataFrame:
    """
    Apply cohort-specific rolling demeaning for pre-treatment periods.

    For each cohort g and pre-treatment period t < g, computes:

    .. math::

        \\dot{Y}_{itg} = Y_{it} - \\frac{1}{g-t-1} \\sum_{q=t+1}^{g-1} Y_{iq}

    The anchor point t = g-1 is set to 0 by convention, as the future
    window {g, ..., g-1} is empty.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel data with columns for outcome, unit identifier,
        time period, and cohort assignment.
    y : str
        Outcome variable column name.
    ivar : str
        Unit identifier column name.
    tvar : str
        Time variable column name (must be numeric or coercible to numeric).
    gvar : str
        Cohort variable column name indicating first treatment period.
    never_treated_values : sequence or None, optional
        Values in gvar indicating never-treated units. Default recognizes
        NaN, 0, and np.inf as never-treated indicators.

    Returns
    -------
    pd.DataFrame
        Copy of input data with additional columns named ``ydot_pre_g{g}_t{t}``
        containing the transformed outcome for each (cohort, period) pair.

    Raises
    ------
    ValueError
        If required columns are missing or no valid cohorts exist.

    See Also
    --------
    transform_staggered_detrend_pre : Apply linear detrending transformation.
    transform_staggered_demean : Post-treatment demeaning transformation.

    Notes
    -----
    The transformation is applied to ALL units (treated, not-yet-treated,
    and never-treated) for each cohort, enabling flexible control group
    selection during estimation.
    """
    # =========================================================================
    # Data Validation
    # =========================================================================
    required_cols = [y, ivar, tvar, gvar]
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    result = data.copy()
    result[tvar] = pd.to_numeric(result[tvar], errors='coerce')

    # =========================================================================
    # Extract Cohorts and Time Range
    # =========================================================================
    if never_treated_values is None:
        nt_values = [0, np.inf]
    else:
        nt_values = list(never_treated_values)

    from .transformations import get_cohorts

    cohorts = get_cohorts(result, gvar, ivar, nt_values)

    if len(cohorts) == 0:
        raise ValueError("No valid treatment cohorts found in data.")

    T_min = int(result[tvar].min())

    # Pre-treatment transformation requires at least one period before g
    for g in cohorts:
        if g <= T_min:
            raise ValueError(
                f"Cohort {g} has no pre-treatment periods: "
                f"earliest time period is {T_min}. Cohort must be > T_min."
            )

    # =========================================================================
    # Compute Pre-treatment Demeaning Transformations
    # =========================================================================
    all_units = result[ivar].unique()

    for g in cohorts:
        pre_periods = get_pre_treatment_periods_for_cohort(g, T_min)

        # Initialize columns with NaN; missing data remains NaN after iteration
        for t in pre_periods:
            col_name = f'ydot_pre_g{int(g)}_t{t}'
            result[col_name] = np.nan

        for unit_id in all_units:
            unit_mask = result[ivar] == unit_id
            unit_data = result[unit_mask]

            for t in pre_periods:
                col_name = f'ydot_pre_g{int(g)}_t{t}'
                period_mask = unit_mask & (result[tvar] == t)

                if not period_mask.any():
                    continue

                Y_it = result.loc[period_mask, y].values[0]

                if pd.isna(Y_it):
                    continue

                # Anchor point (t = g-1): empty future window, set to 0 by convention
                if t == g - 1:
                    result.loc[period_mask, col_name] = 0.0
                else:
                    # Subtract mean of future pre-treatment outcomes
                    rolling_mean = _compute_rolling_mean_future(
                        unit_data, y, tvar, t, g
                    )

                    if not np.isnan(rolling_mean):
                        # Transformed outcome = Y_it - mean of future pre-treatment outcomes.
                        result.loc[period_mask, col_name] = Y_it - rolling_mean
                    # else: leave as NaN

    return result


def transform_staggered_detrend_pre(
    data: pd.DataFrame,
    y: str,
    ivar: str,
    tvar: str,
    gvar: str,
    never_treated_values: Sequence | None = None,
) -> pd.DataFrame:
    """
    Apply cohort-specific rolling detrending for pre-treatment periods.

    For each cohort g and pre-treatment period t ≤ g-3, computes:

    .. math::

        \\ddot{Y}_{itg} = Y_{it} - \\hat{Y}_{itg}

    where Ŷ_{itg} is the OLS-fitted value from regressing Y_{iq} on q
    for q ∈ {t+1, ..., g-1}.

    The anchor points are:
    - t = g-1: Set to 0.0 (empty future window)
    - t = g-2: Set to NaN (only 1 future period, need 2 for OLS)

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel data with columns for outcome, unit identifier,
        time period, and cohort assignment.
    y : str
        Outcome variable column name.
    ivar : str
        Unit identifier column name.
    tvar : str
        Time variable column name (must be numeric or coercible to numeric).
    gvar : str
        Cohort variable column name indicating first treatment period.
    never_treated_values : sequence or None, optional
        Values in gvar indicating never-treated units. Default recognizes
        NaN, 0, and np.inf as never-treated indicators.

    Returns
    -------
    pd.DataFrame
        Copy of input data with additional columns named ``ycheck_pre_g{g}_t{t}``
        containing the detrended outcome for each (cohort, period) pair.

    Raises
    ------
    ValueError
        If required columns are missing or no valid cohorts exist.

    Warns
    -----
    UserWarning
        If a cohort has fewer than 3 pre-treatment periods (detrending
        requires at least 2 future periods for OLS).

    See Also
    --------
    transform_staggered_demean_pre : Apply demeaning transformation.
    transform_staggered_detrend : Post-treatment detrending transformation.

    Notes
    -----
    Detrending requires at least 2 future pre-treatment periods to estimate
    both intercept and slope. For t = g-2, only one future period exists,
    so the detrended value is NaN. For t = g-1 (anchor), the value is 0.

    The transformation is applied to ALL units for each cohort.
    """
    # =========================================================================
    # Data Validation
    # =========================================================================
    required_cols = [y, ivar, tvar, gvar]
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    result = data.copy()
    result[tvar] = pd.to_numeric(result[tvar], errors='coerce')

    # =========================================================================
    # Extract Cohorts and Time Range
    # =========================================================================
    if never_treated_values is None:
        nt_values = [0, np.inf]
    else:
        nt_values = list(never_treated_values)

    from .transformations import get_cohorts

    cohorts = get_cohorts(result, gvar, ivar, nt_values)

    if len(cohorts) == 0:
        raise ValueError("No valid treatment cohorts found in data.")

    T_min = int(result[tvar].min())

    # Detrending requires sufficient pre-treatment periods for OLS estimation
    for g in cohorts:
        n_pre_periods = g - T_min
        if n_pre_periods < 1:
            raise ValueError(
                f"Cohort {g} has no pre-treatment periods: "
                f"earliest time period is {T_min}."
            )
        if n_pre_periods < 3:
            warnings.warn(
                f"Cohort {g} has only {n_pre_periods} pre-treatment period(s). "
                f"Detrending requires at least 3 pre-treatment periods to have "
                f"valid detrended values (2 future periods for OLS). "
                f"Only anchor point will have value 0.",
                UserWarning
            )

    # =========================================================================
    # Compute Pre-treatment Detrending Transformations
    # =========================================================================
    all_units = result[ivar].unique()

    for g in cohorts:
        pre_periods = get_pre_treatment_periods_for_cohort(g, T_min)

        # Initialize columns with NaN; missing data remains NaN after iteration
        for t in pre_periods:
            col_name = f'ycheck_pre_g{int(g)}_t{t}'
            result[col_name] = np.nan

        for unit_id in all_units:
            unit_mask = result[ivar] == unit_id
            unit_data = result[unit_mask]

            for t in pre_periods:
                col_name = f'ycheck_pre_g{int(g)}_t{t}'
                period_mask = unit_mask & (result[tvar] == t)

                if not period_mask.any():
                    continue

                Y_it = result.loc[period_mask, y].values[0]

                if pd.isna(Y_it):
                    continue

                # Anchor point (t = g-1): empty future window, set to 0 by convention
                if t == g - 1:
                    result.loc[period_mask, col_name] = 0.0
                elif t == g - 2:
                    # Only 1 future period; OLS requires at least 2 points
                    pass
                else:
                    # Subtract OLS-fitted value from future pre-treatment trend
                    A, B = _compute_rolling_trend_future(
                        unit_data, y, tvar, t, g
                    )

                    if not np.isnan(A) and not np.isnan(B):
                        # Detrended outcome = Y_it - fitted value from future trend.
                        Y_hat = A + B * t
                        result.loc[period_mask, col_name] = Y_it - Y_hat
                    # else: leave as NaN

    return result

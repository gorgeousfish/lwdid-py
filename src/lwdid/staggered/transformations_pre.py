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

from ..warnings_categories import SmallSampleWarning

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
    # Compute Pre-treatment Demeaning Transformations (Vectorized)
    # =========================================================================
    # Build (unit, time) -> Y pivot table for efficient lookup.
    # Rows = units, columns = time periods.
    pivot = result.pivot_table(index=ivar, columns=tvar, values=y, aggfunc='first')
    # Ensure columns are sorted in ascending integer order.
    pivot = pivot.reindex(columns=sorted(pivot.columns))

    # Collect all new columns and concatenate once to avoid DataFrame fragmentation.
    new_cols = {}

    for g in cohorts:
        pre_periods = get_pre_treatment_periods_for_cohort(g, T_min)

        # Initialize all columns as NaN Series.
        for t in pre_periods:
            col_name = f'ydot_pre_g{int(g)}_t{t}'
            new_cols[col_name] = pd.Series(np.nan, index=result.index)

        # Anchor point (t = g-1): set to 0.0 for all units.
        anchor_t = g - 1
        if anchor_t in pre_periods:
            col_name = f'ydot_pre_g{int(g)}_t{anchor_t}'
            anchor_mask = result[tvar] == anchor_t
            new_cols[col_name].loc[anchor_mask] = 0.0

        # Compute vectorized transformations for non-anchor pre-treatment periods.
        for t in pre_periods:
            if t == g - 1:
                continue  # Anchor already handled.

            col_name = f'ydot_pre_g{int(g)}_t{t}'

            # Future periods: {t+1, ..., g-1}
            future_cols = [c for c in pivot.columns if t < c < g]
            if len(future_cols) == 0:
                continue

            # Mean of future periods for each unit (vectorized).
            future_mean = pivot[future_cols].mean(axis=1)  # Series: unit -> mean

            # Current period outcome Y_it.
            if t in pivot.columns:
                Y_it = pivot[t]  # Series: unit -> Y_it
            else:
                continue

            # Transformed value = Y_it - future_mean.
            transformed = Y_it - future_mean

            # Retain only observations where both values are non-NaN.
            valid = transformed.dropna()

            if len(valid) == 0:
                continue

            # Assign to corresponding (unit, t) rows.
            period_mask = result[tvar] == t
            period_data = result.loc[period_mask]
            unit_map = valid.to_dict()
            new_cols[col_name].loc[period_mask] = period_data[ivar].map(unit_map).values

    # Concatenate all new columns at once.
    if new_cols:
        new_df = pd.DataFrame(new_cols, index=result.index)
        result = pd.concat([result, new_df], axis=1)

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
    SmallSampleWarning
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
                SmallSampleWarning
            )

    # =========================================================================
    # Compute Pre-treatment Detrending Transformations (Vectorized)
    # =========================================================================
    # Build (unit, time) -> Y pivot table.
    pivot = result.pivot_table(index=ivar, columns=tvar, values=y, aggfunc='first')
    pivot = pivot.reindex(columns=sorted(pivot.columns))

    # Collect all new columns and concatenate once to avoid DataFrame fragmentation.
    new_cols = {}

    for g in cohorts:
        pre_periods = get_pre_treatment_periods_for_cohort(g, T_min)

        # Initialize all columns as NaN Series.
        for t in pre_periods:
            col_name = f'ycheck_pre_g{int(g)}_t{t}'
            new_cols[col_name] = pd.Series(np.nan, index=result.index)

        # Anchor point (t = g-1): set to 0.0.
        anchor_t = g - 1
        if anchor_t in pre_periods:
            col_name = f'ycheck_pre_g{int(g)}_t{anchor_t}'
            anchor_mask = result[tvar] == anchor_t
            new_cols[col_name].loc[anchor_mask] = 0.0

        for t in pre_periods:
            if t == g - 1:
                continue  # Anchor point already handled.
            if t == g - 2:
                continue  # Only 1 future period; OLS requires at least 2.

            col_name = f'ycheck_pre_g{int(g)}_t{t}'

            # Future periods: {t+1, ..., g-1}.
            future_cols = [c for c in pivot.columns if t < c < g]
            if len(future_cols) < 2:
                continue

            # Extract future-period data matrix (units × future_periods).
            Y_future = pivot[future_cols]
            t_vals = np.array(future_cols, dtype=float)

            # Vectorized OLS: Y = A + B*t.
            # Formulas: B = (n*Σ(tY) - Σt*ΣY) / (n*Σ(t²) - (Σt)²)
            #           A = (ΣY - B*Σt) / n

            # Count valid (non-NaN) observations per unit.
            valid_mask = Y_future.notna()
            n_valid = valid_mask.sum(axis=1)

            # Require at least 2 valid observations for OLS.
            enough_mask = n_valid >= 2

            if not enough_mask.any():
                continue

            # Replace NaN with 0 for summation (masked by valid_mask).
            Y_filled = Y_future.fillna(0.0)

            # Compute sufficient statistics over valid observations only.
            sum_Y = (Y_filled * valid_mask).sum(axis=1)
            sum_tY = (Y_filled * valid_mask * t_vals[np.newaxis, :]).sum(axis=1)
            sum_t = (valid_mask * t_vals[np.newaxis, :]).sum(axis=1)
            sum_t2 = (valid_mask * (t_vals ** 2)[np.newaxis, :]).sum(axis=1)

            # OLS normal equations.
            denom = n_valid * sum_t2 - sum_t ** 2

            # Guard against division by zero (constant time or insufficient data).
            safe_denom = denom.where(denom.abs() > 1e-10, np.nan)

            B_hat = (n_valid * sum_tY - sum_t * sum_Y) / safe_denom
            A_hat = (sum_Y - B_hat * sum_t) / n_valid

            # Fitted value: Y_hat = A + B * t.
            Y_hat_at_t = A_hat + B_hat * float(t)

            # Current-period outcome Y_{it}.
            if t in pivot.columns:
                Y_it = pivot[t]
            else:
                continue

            # Transformed value: Y_{it} - Y_hat_{it}.
            transformed = Y_it - Y_hat_at_t

            # Retain valid entries (sufficient data, non-NaN outcome, successful OLS).
            valid = transformed[enough_mask].dropna()

            if len(valid) == 0:
                continue

            # Map transformed values back to the corresponding rows.
            period_mask = result[tvar] == t
            period_data = result.loc[period_mask]
            unit_map = valid.to_dict()
            new_cols[col_name].loc[period_mask] = period_data[ivar].map(unit_map).values

    # Concatenate all new columns at once to avoid DataFrame fragmentation.
    if new_cols:
        new_df = pd.DataFrame(new_cols, index=result.index)
        result = pd.concat([result, new_df], axis=1)

    return result

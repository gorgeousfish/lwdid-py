"""
Cohort-specific data transformations for staggered difference-in-differences.

This module implements unit-level time-series transformations that convert
panel data into cross-sectional form for each cohort-time pair in staggered
adoption settings. The transformations remove pre-treatment patterns at the
unit level, enabling application of standard treatment effect estimators.

Two transformation methods are available:

- **Demeaning**: Subtracts the unit-specific pre-treatment mean from post-
  treatment outcomes. Requires at least one pre-treatment period per cohort.

- **Detrending**: Removes unit-specific linear trends estimated from pre-
  treatment data. Requires at least two pre-treatment periods per cohort
  to identify both intercept and slope parameters.

For cohort g in calendar time r, the transformation uses all periods prior
to g as the pre-treatment window. This cohort-specific definition ensures
that transformed outcomes reflect the appropriate counterfactual for each
treatment cohort.

Notes
-----
The pre-treatment transformation parameters (mean or trend coefficients)
are fixed for each cohort regardless of the calendar time r at which effects
are estimated. All units (including never-treated and not-yet-treated) are
transformed for each cohort to enable valid control group construction.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def get_cohorts(
    data: pd.DataFrame,
    gvar: str,
    ivar: str,
    never_treated_values: list | None = None,
) -> list[int]:
    """
    Extract valid treatment cohorts from panel data.

    Identifies all distinct first-treatment periods in the data, excluding
    units that are never treated. Never-treated status is determined by
    missing values (NaN) or explicit indicator values.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data containing unit and cohort identifiers.
    gvar : str
        Column name for the cohort variable indicating first treatment period.
    ivar : str
        Column name for the unit identifier.
    never_treated_values : list or None, optional
        Values in gvar that indicate never-treated units. Default is [0, np.inf].
        NaN values are always treated as never-treated regardless of this list.

    Returns
    -------
    list of int
        Sorted list of treatment cohorts, excluding never-treated indicators.

    Raises
    ------
    KeyError
        If gvar or ivar columns are not found in data.

    See Also
    --------
    get_valid_periods_for_cohort : Get post-treatment periods for a cohort.
    transform_staggered_demean : Apply demeaning transformation.
    """
    if never_treated_values is None:
        never_treated_values = [0, np.inf]

    # Deduplicate at unit level to ensure each unit contributes one cohort value
    unit_gvar = data.drop_duplicates(subset=[ivar])[gvar]

    valid_cohorts = []
    for g in unit_gvar.unique():
        # NaN always indicates never-treated regardless of explicit values
        if pd.isna(g):
            continue
        if g in never_treated_values:
            continue
        valid_cohorts.append(int(g))

    return sorted(valid_cohorts)


def get_valid_periods_for_cohort(
    cohort: int,
    T_max: int,
) -> list[int]:
    """
    Determine valid post-treatment periods for a treatment cohort.

    For cohort g, returns the set of calendar times {g, g+1, ..., T_max}
    during which the cohort is exposed to treatment and treatment effects
    can be estimated.

    Parameters
    ----------
    cohort : int
        Treatment cohort identifier (first treatment period).
    T_max : int
        Maximum time period available in the data.

    Returns
    -------
    list of int
        Consecutive integers from cohort through T_max inclusive.

    See Also
    --------
    get_cohorts : Extract treatment cohorts from data.
    """
    return list(range(int(cohort), int(T_max) + 1))


def _compute_pre_treatment_mean(
    unit_data: pd.DataFrame,
    y: str,
    tvar: str,
    cohort: int,
) -> float:
    """
    Compute pre-treatment outcome mean for a single unit.

    Parameters
    ----------
    unit_data : pd.DataFrame
        Time series data for a single unit.
    y : str
        Outcome variable column name.
    tvar : str
        Time variable column name.
    cohort : int
        Cohort value defining the pre-treatment cutoff.

    Returns
    -------
    float
        Mean of outcome values for periods strictly before cohort.
        Returns NaN if no valid pre-treatment observations.
    """
    pre_data = unit_data[unit_data[tvar] < cohort]
    y_pre = pre_data[y].dropna()

    if len(y_pre) == 0:
        return np.nan

    return y_pre.mean()


def _compute_pre_treatment_trend(
    unit_data: pd.DataFrame,
    y: str,
    tvar: str,
    cohort: int,
) -> tuple[float, float]:
    """
    Estimate pre-treatment linear trend parameters for a single unit.

    Fits the model Y_t = A + B * t using ordinary least squares on
    observations with t < cohort.

    Parameters
    ----------
    unit_data : pd.DataFrame
        Time series data for a single unit.
    y : str
        Outcome variable column name.
    tvar : str
        Time variable column name.
    cohort : int
        Cohort value defining the pre-treatment cutoff.

    Returns
    -------
    tuple of float
        (intercept, slope) parameters from the fitted linear trend.
        Returns (NaN, NaN) if fewer than 2 pre-treatment periods.
    """
    pre_data = unit_data[unit_data[tvar] < cohort].dropna(subset=[y])

    # Linear trend requires at least 2 points to identify intercept and slope
    if len(pre_data) < 2:
        return (np.nan, np.nan)

    t_vals = pre_data[tvar].values.astype(float)
    y_vals = pre_data[y].values.astype(float)

    # np.polyfit returns coefficients in descending order: [slope, intercept]
    B, A = np.polyfit(t_vals, y_vals, deg=1)

    return (A, B)


def transform_staggered_demean(
    data: pd.DataFrame,
    y: str,
    ivar: str,
    tvar: str,
    gvar: str,
    never_treated_values: list | None = None,
) -> pd.DataFrame:
    """
    Apply cohort-specific demeaning transformation for staggered designs.

    Computes the transformed outcome for each cohort g and post-treatment
    period r:

    .. math::

        \\dot{Y}_{irg} = Y_{ir} - \\bar{Y}_{i,pre(g)}

    where :math:`\\bar{Y}_{i,pre(g)}` is the mean of unit i's outcomes over
    all periods strictly before g.

    The transformation is applied to all units (treated, not-yet-treated,
    and never-treated) for each cohort, enabling flexible control group
    selection during estimation.

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
    never_treated_values : list or None, optional
        Values in gvar indicating never-treated units. Default recognizes
        NaN, 0, and np.inf as never-treated indicators.

    Returns
    -------
    pd.DataFrame
        Copy of input data with additional columns named ``ydot_g{g}_r{r}``
        containing the transformed outcome for each cohort-period pair.

    Raises
    ------
    ValueError
        If required columns are missing, no valid cohorts exist, or any
        cohort has no pre-treatment periods (cohort <= T_min).

    Warns
    -----
    UserWarning
        If no never-treated units are found in the data.

    See Also
    --------
    transform_staggered_detrend : Apply detrending transformation.
    get_cohorts : Extract treatment cohorts from data.

    Notes
    -----
    The pre-treatment mean is computed once per cohort and remains fixed
    across all post-treatment periods. This ensures consistency with the
    identification strategy where each cohort defines its own baseline.
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
        nt_values = never_treated_values

    cohorts = get_cohorts(result, gvar, ivar, nt_values)

    if len(cohorts) == 0:
        raise ValueError("No valid treatment cohorts found in data.")

    T_min = int(result[tvar].min())
    T_max = int(result[tvar].max())

    # Validate each cohort has pre-treatment periods
    for g in cohorts:
        if g <= T_min:
            raise ValueError(
                f"Cohort {g} has no pre-treatment periods: "
                f"earliest time period is {T_min}. Cohort must be > T_min."
            )

    # Check for never-treated units
    unit_gvar = result.drop_duplicates(subset=[ivar]).set_index(ivar)[gvar]
    is_nt = unit_gvar.isna() | unit_gvar.isin(nt_values)
    has_nt = is_nt.any()

    if not has_nt:
        warnings.warn(
            "No never-treated units found. Overall and cohort effects cannot "
            "be estimated. Only (g,r)-specific effects are available.",
            UserWarning
        )

    # =========================================================================
    # Compute Transformations
    # =========================================================================
    for g in cohorts:
        # Use strict inequality to ensure cohort g's first treatment period
        # is not included in its own pre-treatment baseline
        pre_mask = result[tvar] < g

        # Vectorized computation across units for efficiency with large panels
        pre_means = result[pre_mask].groupby(ivar)[y].mean()

        post_periods = get_valid_periods_for_cohort(g, T_max)

        for r in post_periods:
            col_name = f'ydot_g{int(g)}_r{r}'
            result[col_name] = np.nan

            period_mask = result[tvar] == r

            # Apply demeaning transformation to all units in period r
            result.loc[period_mask, col_name] = (
                result.loc[period_mask, y].values -
                result.loc[period_mask, ivar].map(pre_means).values
            )

    return result


def transform_staggered_detrend(
    data: pd.DataFrame,
    y: str,
    ivar: str,
    tvar: str,
    gvar: str,
    never_treated_values: list | None = None,
) -> pd.DataFrame:
    """
    Apply cohort-specific detrending transformation for staggered designs.

    Fits unit-specific linear trends using pre-treatment data and computes
    out-of-sample residuals for post-treatment periods:

    .. math::

        \\ddot{Y}_{irg} = Y_{ir} - (\\hat{A}_{ig} + \\hat{B}_{ig} \\cdot r)

    where :math:`\\hat{A}_{ig}` and :math:`\\hat{B}_{ig}` are OLS estimates
    from regressing :math:`Y_{it}` on a constant and :math:`t` for periods
    :math:`t < g`.

    This transformation removes unit-specific heterogeneous linear trends,
    relaxing the parallel trends assumption to allow treatment assignment
    correlated with trend slopes.

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
        Time variable column name.
    gvar : str
        Cohort variable column name indicating first treatment period.
    never_treated_values : list or None, optional
        Values in gvar indicating never-treated units.

    Returns
    -------
    pd.DataFrame
        Copy of input data with additional columns named ``ycheck_g{g}_r{r}``
        containing the detrended outcome for each cohort-period pair.

    Raises
    ------
    ValueError
        If required columns are missing, no valid cohorts exist, or any
        cohort has fewer than 2 pre-treatment periods.

    Warns
    -----
    UserWarning
        If no never-treated units are found in the data.

    See Also
    --------
    transform_staggered_demean : Apply demeaning transformation.

    Notes
    -----
    Detrending requires at least two pre-treatment periods per cohort to
    identify both intercept and slope parameters. Units with insufficient
    pre-treatment data receive NaN values for transformed outcomes.

    The detrending approach is appropriate when treatment and control groups
    may have different pre-treatment trends but those trends are linear. For
    more complex trend patterns, consider using only periods where trends
    appear approximately parallel.
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
    # Extract Cohorts and Validate
    # =========================================================================
    if never_treated_values is None:
        nt_values = [0, np.inf]
    else:
        nt_values = never_treated_values

    cohorts = get_cohorts(result, gvar, ivar, nt_values)

    if len(cohorts) == 0:
        raise ValueError("No valid treatment cohorts found in data.")

    T_min = int(result[tvar].min())
    T_max = int(result[tvar].max())

    # Detrending requires at least 2 pre-treatment periods
    for g in cohorts:
        n_pre_periods = g - T_min
        if n_pre_periods < 2:
            raise ValueError(
                f"Cohort {g} has only {n_pre_periods} pre-treatment period(s). "
                f"Detrending requires at least 2 pre-treatment periods to "
                f"estimate linear trend. Consider using demean instead."
            )

    # Check for never-treated units
    unit_gvar = result.drop_duplicates(subset=[ivar]).set_index(ivar)[gvar]
    is_nt = unit_gvar.isna() | unit_gvar.isin(nt_values)
    has_nt = is_nt.any()

    if not has_nt:
        warnings.warn(
            "No never-treated units found. Overall and cohort effects cannot "
            "be estimated. Only (g,r)-specific effects are available.",
            UserWarning
        )

    # =========================================================================
    # Compute Detrending Transformations
    # =========================================================================
    # Process unit by unit to allow different trend estimates per unit
    all_units = result[ivar].unique()

    for g in cohorts:
        post_periods = get_valid_periods_for_cohort(g, T_max)

        # Pre-allocate columns to avoid repeated DataFrame modifications
        for r in post_periods:
            col_name = f'ycheck_g{int(g)}_r{r}'
            result[col_name] = np.nan

        for unit_id in all_units:
            unit_mask = result[ivar] == unit_id
            unit_data = result[unit_mask]

            # Use strict inequality for consistent pre-treatment definition
            pre_data = unit_data[unit_data[tvar] < g].dropna(subset=[y])

            if len(pre_data) < 2:
                # Skip units without enough data for trend identification
                continue

            t_vals = pre_data[tvar].values.astype(float)
            y_vals = pre_data[y].values.astype(float)

            try:
                # np.polyfit returns [slope, intercept] for degree 1
                B, A = np.polyfit(t_vals, y_vals, deg=1)
            except (np.linalg.LinAlgError, ValueError):
                # Collinearity or other numerical issues prevent estimation
                continue

            # Project trend forward and compute out-of-sample residuals
            for r in post_periods:
                col_name = f'ycheck_g{int(g)}_r{r}'
                period_mask = unit_mask & (result[tvar] == r)

                if period_mask.any():
                    y_r = result.loc[period_mask, y].values[0]
                    predicted = A + B * r
                    result.loc[period_mask, col_name] = y_r - predicted

    return result

"""
Cohort-specific data transformations for staggered difference-in-differences.

This module implements unit-level time-series transformations that convert
panel data into cross-sectional form for each cohort-time pair in staggered
adoption settings. The transformations remove pre-treatment patterns at the
unit level, enabling application of standard treatment effect estimators.

Four transformation methods are provided:

Demeaning (``transform_staggered_demean``)
    Subtracts the unit-specific pre-treatment mean from post-treatment
    outcomes. Requires at least one pre-treatment period.

Detrending (``transform_staggered_detrend``)
    Removes unit-specific linear trends estimated from pre-treatment data.
    Requires at least two pre-treatment periods to identify intercept and
    slope parameters.

Seasonal demeaning (``transform_staggered_demeanq``)
    Extends demeaning to account for seasonal fixed effects. Suitable for
    quarterly, monthly, or weekly data with regular seasonal patterns.

Seasonal detrending (``transform_staggered_detrendq``)
    Combines linear detrending with seasonal adjustment. Requires sufficient
    pre-treatment observations to estimate trend and seasonal parameters.

For treatment cohort g in calendar time r, transformations use all periods
strictly prior to g as the pre-treatment window. This cohort-specific
definition ensures transformed outcomes reflect the appropriate counterfactual
for each treatment cohort.

Notes
-----
Pre-treatment transformation parameters (mean, trend coefficients, or seasonal
effects) are computed once per cohort and remain fixed across all post-treatment
periods. All units (treated, not-yet-treated, and never-treated) are transformed
for each cohort to enable flexible control group selection during estimation.
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

    Identifies all distinct first-treatment periods present in the data,
    excluding units that are never treated. Never-treated status is
    determined by missing values (NaN) or explicit indicator values
    specified by the user.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format containing unit and cohort identifiers.
    gvar : str
        Column name for the cohort variable indicating first treatment period.
    ivar : str
        Column name for the unit identifier.
    never_treated_values : list or None, optional
        Values in gvar that indicate never-treated units. Default is
        [0, np.inf]. NaN values are always treated as never-treated
        regardless of this parameter.

    Returns
    -------
    list of int
        Sorted list of unique treatment cohort values, excluding any
        never-treated indicators.

    Raises
    ------
    KeyError
        If gvar or ivar columns are not found in data.

    See Also
    --------
    get_valid_periods_for_cohort : Determine post-treatment periods for a cohort.
    transform_staggered_demean : Apply demeaning transformation.

    Notes
    -----
    In staggered adoption designs, a cohort comprises all units first treated
    in the same time period. The cohort identifier g also defines the pre-
    treatment window (periods 1 through g-1) used for baseline transformations.
    """
    if never_treated_values is None:
        never_treated_values = [0, np.inf]

    # Deduplicate at unit level to ensure each unit contributes one cohort value.
    unit_gvar = data.drop_duplicates(subset=[ivar])[gvar]

    valid_cohorts = []
    for g in unit_gvar.unique():
        # NaN always indicates never-treated regardless of explicit values.
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

    For treatment cohort g, returns the set of calendar times {g, g+1, ...,
    T_max} during which the cohort is exposed to treatment and cohort-time
    specific average treatment effects on the treated can be estimated.

    Parameters
    ----------
    cohort : int
        Treatment cohort identifier representing the first treatment period.
    T_max : int
        Maximum calendar time period available in the panel data.

    Returns
    -------
    list of int
        Consecutive integers from cohort through T_max inclusive,
        representing all periods where the cohort experiences treatment.

    See Also
    --------
    get_cohorts : Extract treatment cohorts from data.
    transform_staggered_demean : Apply demeaning transformation.

    Notes
    -----
    The event-time relative to treatment onset ranges from 0 (instantaneous
    effect at treatment start) to T_max - cohort (longest exposure duration).
    Each calendar time r corresponds to event-time e = r - g for cohort g.
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
        Time series observations for a single unit.
    y : str
        Outcome variable column name.
    tvar : str
        Time variable column name.
    cohort : int
        Treatment cohort defining the pre-treatment cutoff (t < cohort).

    Returns
    -------
    float
        Mean of outcome values for periods strictly before cohort.
        Returns NaN if no valid pre-treatment observations exist.
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

    Fits a linear model Y_t = A + B * t using ordinary least squares on
    observations with t < cohort.

    Parameters
    ----------
    unit_data : pd.DataFrame
        Time series observations for a single unit.
    y : str
        Outcome variable column name.
    tvar : str
        Time variable column name.
    cohort : int
        Treatment cohort defining the pre-treatment cutoff (t < cohort).

    Returns
    -------
    tuple of float
        (intercept, slope) parameters from the fitted linear trend.
        Returns (NaN, NaN) if fewer than two pre-treatment periods exist.

    Notes
    -----
    Uses numpy.polyfit for numerical stability with potentially ill-
    conditioned time indices. The function handles edge cases where
    collinearity prevents parameter estimation.
    """
    pre_data = unit_data[unit_data[tvar] < cohort].dropna(subset=[y])

    # Linear trend requires at least 2 points to identify intercept and slope.
    if len(pre_data) < 2:
        return (np.nan, np.nan)

    t_vals = pre_data[tvar].values.astype(float)
    y_vals = pre_data[y].values.astype(float)

    # np.polyfit returns coefficients in descending order: [slope, intercept].
    B, A = np.polyfit(t_vals, y_vals, deg=1)

    return (A, B)


def transform_staggered_demean(
    data: pd.DataFrame,
    y: str,
    ivar: str,
    tvar: str,
    gvar: str,
    never_treated_values: list | None = None,
    exclude_pre_periods: int = 0,
) -> pd.DataFrame:
    """
    Apply cohort-specific demeaning transformation for staggered designs.

    Computes the transformed outcome for each treatment cohort g and
    post-treatment calendar time r:

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
    exclude_pre_periods : int, default 0
        Number of periods immediately before treatment to exclude from
        pre-treatment mean calculation. Used for robustness checks when
        no-anticipation assumption may be violated.

        For cohort g, the pre-treatment mean is computed using periods
        {T_min, ..., g-1-exclude_pre_periods} instead of {T_min, ..., g-1}.

    Returns
    -------
    pd.DataFrame
        Copy of input data with additional columns named ``ydot_g{g}_r{r}``
        containing the transformed outcome for each (cohort, period) pair.

    Raises
    ------
    ValueError
        If required columns are missing, no valid cohorts exist, or any
        cohort has no pre-treatment periods (cohort <= T_min).
    InsufficientPrePeriodsError
        If any cohort has fewer than 1 pre-treatment period remaining
        after applying exclude_pre_periods.

    Warns
    -----
    UserWarning
        If no never-treated units are found in the data.

    See Also
    --------
    transform_staggered_detrend : Apply linear detrending transformation.
    get_cohorts : Extract treatment cohorts from data.

    Notes
    -----
    The pre-treatment mean is computed once per cohort and remains fixed
    across all post-treatment periods. This ensures consistency with the
    identification strategy where each cohort defines its own baseline.
    """
    from ..exceptions import InsufficientPrePeriodsError

    # =========================================================================
    # Data Validation
    # =========================================================================
    required_cols = [y, ivar, tvar, gvar]
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Validate exclude_pre_periods parameter
    if not isinstance(exclude_pre_periods, (int, np.integer)):
        raise TypeError(
            f"exclude_pre_periods must be an integer, got {type(exclude_pre_periods).__name__}"
        )
    if exclude_pre_periods < 0:
        raise ValueError(
            f"exclude_pre_periods must be non-negative, got {exclude_pre_periods}"
        )

    result = data.copy()
    result[tvar] = pd.to_numeric(result[tvar], errors='coerce')

    # Round the time variable to integer to avoid floating-point precision
    # issues (e.g., 4.0 + 1e-14 after CSV import fails to match the
    # integer 4 exactly)
    tvar_notna = result[tvar].notna()
    if tvar_notna.any():
        result.loc[tvar_notna, tvar] = result.loc[tvar_notna, tvar].round().astype(int)

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

    # Validate that each cohort has pre-treatment periods.
    for g in cohorts:
        if g <= T_min:
            raise ValueError(
                f"Cohort {g} has no pre-treatment periods: "
                f"earliest time period is {T_min}. Cohort must be > T_min."
            )

    # Validate that each cohort has sufficient pre-treatment periods after exclusion.
    # Demeaning requires at least 1 pre-treatment period.
    min_required = 1
    for g in cohorts:
        pre_upper_bound = g - exclude_pre_periods
        n_pre_periods = pre_upper_bound - T_min
        if n_pre_periods < min_required:
            raise InsufficientPrePeriodsError(
                f"Cohort {g} requires at least {min_required} pre-treatment period(s) "
                f"for demeaning, but only {max(0, n_pre_periods)} remain after "
                f"excluding {exclude_pre_periods} period(s). "
                f"Available periods: [{T_min}, {g-1}], "
                f"excluded: [{max(T_min, g-exclude_pre_periods)}, {g-1}].",
                cohort=g,
                available=max(0, n_pre_periods),
                required=min_required,
                excluded=exclude_pre_periods,
            )

    # Check for never-treated units.
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
        # Compute effective pre-treatment upper bound with exclusion.
        # Use strict inequality: t < (g - exclude_pre_periods)
        pre_upper_bound = g - exclude_pre_periods
        pre_mask = result[tvar] < pre_upper_bound

        # Vectorized computation across units for efficiency with large panels.
        pre_means = result[pre_mask].groupby(ivar)[y].mean()

        post_periods = get_valid_periods_for_cohort(g, T_max)

        for r in post_periods:
            col_name = f'ydot_g{int(g)}_r{r}'
            result[col_name] = np.nan

            period_mask = result[tvar] == r

            # Transform all units to enable both treated and control estimation.
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
    exclude_pre_periods: int = 0,
) -> pd.DataFrame:
    """
    Apply cohort-specific linear detrending transformation for staggered designs.

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
        Time variable column name (must be numeric or coercible to numeric).
    gvar : str
        Cohort variable column name indicating first treatment period.
    never_treated_values : list or None, optional
        Values in gvar indicating never-treated units. Default recognizes
        NaN, 0, and np.inf as never-treated indicators.
    exclude_pre_periods : int, default 0
        Number of periods immediately before treatment to exclude from
        pre-treatment trend estimation. Used for robustness checks when
        no-anticipation assumption may be violated.

        For cohort g, the trend is estimated using periods
        {T_min, ..., g-1-exclude_pre_periods} instead of {T_min, ..., g-1}.

    Returns
    -------
    pd.DataFrame
        Copy of input data with additional columns named ``ycheck_g{g}_r{r}``
        containing the detrended outcome for each (cohort, period) pair.

    Raises
    ------
    ValueError
        If required columns are missing, no valid cohorts exist, or any
        cohort has fewer than two pre-treatment periods.
    InsufficientPrePeriodsError
        If any cohort has fewer than 2 pre-treatment periods remaining
        after applying exclude_pre_periods.

    Warns
    -----
    UserWarning
        If no never-treated units are found in the data.

    See Also
    --------
    transform_staggered_demean : Apply demeaning transformation.
    get_cohorts : Extract treatment cohorts from data.

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
    from ..exceptions import InsufficientPrePeriodsError

    # =========================================================================
    # Data Validation
    # =========================================================================
    required_cols = [y, ivar, tvar, gvar]
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Validate exclude_pre_periods parameter
    if not isinstance(exclude_pre_periods, (int, np.integer)):
        raise TypeError(
            f"exclude_pre_periods must be an integer, got {type(exclude_pre_periods).__name__}"
        )
    if exclude_pre_periods < 0:
        raise ValueError(
            f"exclude_pre_periods must be non-negative, got {exclude_pre_periods}"
        )

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

    # Detrending requires at least 2 pre-treatment periods.
    min_required = 2
    for g in cohorts:
        pre_upper_bound = g - exclude_pre_periods
        n_pre_periods = pre_upper_bound - T_min
        if n_pre_periods < min_required:
            raise InsufficientPrePeriodsError(
                f"Cohort {g} requires at least {min_required} pre-treatment periods "
                f"for detrending, but only {max(0, n_pre_periods)} remain after "
                f"excluding {exclude_pre_periods} period(s). "
                f"Available periods: [{T_min}, {g-1}], "
                f"excluded: [{max(T_min, g-exclude_pre_periods)}, {g-1}]. "
                f"Consider using demean instead or reducing exclude_pre_periods.",
                cohort=g,
                available=max(0, n_pre_periods),
                required=min_required,
                excluded=exclude_pre_periods,
            )

    # Check for never-treated units.
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
    # Process unit by unit to allow different trend estimates per unit.
    all_units = result[ivar].unique()

    for g in cohorts:
        # Compute effective pre-treatment upper bound with exclusion.
        pre_upper_bound = g - exclude_pre_periods

        post_periods = get_valid_periods_for_cohort(g, T_max)

        # Pre-allocate columns to avoid repeated DataFrame resizing overhead.
        for r in post_periods:
            col_name = f'ycheck_g{int(g)}_r{r}'
            result[col_name] = np.nan

        for unit_id in all_units:
            unit_mask = result[ivar] == unit_id
            unit_data = result[unit_mask]

            # Strict inequality ensures pre-treatment window excludes treatment onset.
            pre_data = unit_data[unit_data[tvar] < pre_upper_bound].dropna(subset=[y])

            if len(pre_data) < 2:
                # OLS requires at least 2 points to identify intercept and slope.
                continue

            t_vals = pre_data[tvar].values.astype(float)
            y_vals = pre_data[y].values.astype(float)

            try:
                # polyfit returns [slope, intercept] in descending degree order.
                B, A = np.polyfit(t_vals, y_vals, deg=1)
            except (np.linalg.LinAlgError, ValueError):
                # Collinearity or ill-conditioned design matrix prevents estimation.
                continue

            # Out-of-sample residuals identify treatment effects under parallel trends.
            for r in post_periods:
                col_name = f'ycheck_g{int(g)}_r{r}'
                period_mask = unit_mask & (result[tvar] == r)

                if period_mask.any():
                    y_r = result.loc[period_mask, y].values[0]
                    predicted = A + B * r
                    result.loc[period_mask, col_name] = y_r - predicted

    return result


def _compute_pre_treatment_seasonal_mean(
    unit_data: pd.DataFrame,
    y: str,
    tvar: str,
    season_var: str,
    cohort: int,
    Q: int = 4,
    exclude_pre_periods: int = 0,
) -> tuple[float, np.ndarray, list]:
    """
    Compute seasonal mean parameters for a single unit using pre-treatment data.

    Estimates a seasonal mean model:

    .. math::

        Y_{it} = \\mu + \\sum_{q=2}^{Q} \\gamma_q D_q + \\varepsilon_{it}

    Parameters
    ----------
    unit_data : pd.DataFrame
        Time series observations for a single unit.
    y : str
        Outcome variable column name.
    tvar : str
        Time variable column name.
    season_var : str
        Seasonal indicator column name (values 1 to Q).
    cohort : int
        Treatment cohort defining the pre-treatment cutoff (t < cohort).
    Q : int, default 4
        Number of seasonal periods (4=quarterly, 12=monthly, 52=weekly).
    exclude_pre_periods : int, default 0
        Number of periods immediately before treatment to exclude.

    Returns
    -------
    mu : float
        Intercept (reference category mean).
    gamma : ndarray
        Seasonal dummy coefficients (length Q-1 or fewer if not all seasons observed).
    observed_seasons : list
        List of observed seasons in pre-treatment period (sorted).

    Notes
    -----
    Returns (NaN, empty array, empty list) if insufficient pre-treatment data.
    """
    import statsmodels.api as sm

    # Adjust pre-treatment window to exclude periods near treatment onset.
    pre_upper_bound = cohort - exclude_pre_periods
    pre_data = unit_data[unit_data[tvar] < pre_upper_bound].copy()

    # Require non-missing outcome and season for valid estimation.
    valid_mask = pre_data[y].notna() & pre_data[season_var].notna()
    n_valid = valid_mask.sum()

    if n_valid == 0:
        return np.nan, np.array([]), []

    valid_pre = pre_data[valid_mask]
    observed_seasons = sorted(valid_pre[season_var].unique())
    n_seasons = len(observed_seasons)
    # Total parameters: intercept + (n_seasons - 1) seasonal dummies.
    n_params = n_seasons

    # OLS requires residual degrees of freedom > 0 for valid inference.
    if n_valid <= n_params:
        return np.nan, np.array([]), []

    # Use smallest observed season as reference to avoid dummy trap.
    s_categorical = pd.Categorical(pre_data[season_var], categories=observed_seasons)
    s_dummies = pd.get_dummies(s_categorical, drop_first=True, prefix='s', dtype=float)

    X = sm.add_constant(s_dummies.values)
    y_vals = pre_data[y].values

    try:
        model = sm.OLS(y_vals, X, missing='drop').fit()

        if np.isnan(model.params).any() or np.isinf(model.params).any():
            return np.nan, np.array([]), []

        mu = model.params[0]
        gamma = model.params[1:] if len(model.params) > 1 else np.array([])

        return mu, gamma, observed_seasons

    except Exception:
        return np.nan, np.array([]), []


def _compute_pre_treatment_seasonal_trend(
    unit_data: pd.DataFrame,
    y: str,
    tvar: str,
    season_var: str,
    cohort: int,
    Q: int = 4,
    exclude_pre_periods: int = 0,
) -> tuple[float, float, np.ndarray, list, float]:
    """
    Compute seasonal trend parameters for a single unit using pre-treatment data.

    Estimates a combined trend and seasonal model:

    .. math::

        Y_{it} = \\alpha + \\beta t + \\sum_{q=2}^{Q} \\gamma_q D_q + \\varepsilon_{it}

    Parameters
    ----------
    unit_data : pd.DataFrame
        Time series observations for a single unit.
    y : str
        Outcome variable column name.
    tvar : str
        Time variable column name.
    season_var : str
        Seasonal indicator column name (values 1 to Q).
    cohort : int
        Treatment cohort defining the pre-treatment cutoff (t < cohort).
    Q : int, default 4
        Number of seasonal periods (4=quarterly, 12=monthly, 52=weekly).
    exclude_pre_periods : int, default 0
        Number of periods immediately before treatment to exclude.

    Returns
    -------
    alpha : float
        Intercept.
    beta : float
        Linear trend slope.
    gamma : ndarray
        Seasonal dummy coefficients.
    observed_seasons : list
        List of observed seasons in pre-treatment period (sorted).
    t_mean : float
        Mean of time variable in pre-treatment period (for centering).

    Notes
    -----
    Returns (NaN, NaN, empty array, empty list, NaN) if insufficient data.
    Time is centered at pre-treatment mean for numerical stability.
    """
    import statsmodels.api as sm

    # Adjust pre-treatment window to exclude periods near treatment onset.
    pre_upper_bound = cohort - exclude_pre_periods
    pre_data = unit_data[unit_data[tvar] < pre_upper_bound].copy()

    # Require non-missing outcome, time, and season for valid estimation.
    valid_mask = pre_data[y].notna() & pre_data[tvar].notna() & pre_data[season_var].notna()
    n_valid = valid_mask.sum()

    if n_valid == 0:
        return np.nan, np.nan, np.array([]), [], np.nan

    valid_pre = pre_data[valid_mask]
    observed_seasons = sorted(valid_pre[season_var].unique())
    n_seasons = len(observed_seasons)
    # Total parameters: intercept + slope + (n_seasons - 1) seasonal dummies.
    n_params = 1 + n_seasons

    # OLS requires residual degrees of freedom > 0 for valid inference.
    if n_valid <= n_params:
        return np.nan, np.nan, np.array([]), [], np.nan

    # Constant time values prevent trend identification.
    t_variance = pre_data[tvar].var(ddof=0)
    if t_variance < 1e-10:
        return np.nan, np.nan, np.array([]), [], np.nan

    # Center time at pre-treatment mean to reduce numerical condition number.
    t_mean = pre_data[tvar].mean()
    t_centered = pre_data[tvar] - t_mean

    # Use smallest observed season as reference to avoid dummy trap.
    s_categorical = pd.Categorical(pre_data[season_var], categories=observed_seasons)
    s_dummies = pd.get_dummies(s_categorical, drop_first=True, prefix='s', dtype=float)

    X = np.column_stack([
        np.ones(len(pre_data)),
        t_centered.values,
        s_dummies.values
    ])
    y_vals = pre_data[y].values

    try:
        model = sm.OLS(y_vals, X, missing='drop').fit()

        if np.isnan(model.params).any() or np.isinf(model.params).any():
            return np.nan, np.nan, np.array([]), [], np.nan

        alpha = model.params[0]
        beta = model.params[1]
        gamma = model.params[2:] if len(model.params) > 2 else np.array([])

        return alpha, beta, gamma, observed_seasons, t_mean

    except Exception:
        return np.nan, np.nan, np.array([]), [], np.nan


def transform_staggered_demeanq(
    data: pd.DataFrame,
    y: str,
    ivar: str,
    tvar: str,
    gvar: str,
    season_var: str,
    Q: int = 4,
    never_treated_values: list | None = None,
    exclude_pre_periods: int = 0,
) -> pd.DataFrame:
    """
    Apply cohort-specific seasonal demeaning for staggered designs.

    Computes transformed outcome for each cohort g and post-treatment
    calendar time r:

    .. math::

        \\dot{Y}_{irg} = Y_{ir} - \\hat{\\mu}_{ig} - \\sum_{q=2}^{Q} \\hat{\\gamma}_{qg} D_q

    where parameters are estimated from pre-treatment periods (t < g).

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel data with columns for outcome, unit identifier,
        time period, cohort assignment, and seasonal indicator.
    y : str
        Outcome variable column name.
    ivar : str
        Unit identifier column name.
    tvar : str
        Time variable column name (must be numeric or coercible to numeric).
    gvar : str
        Cohort variable column name indicating first treatment period.
    season_var : str
        Seasonal indicator column name. Values should be integers from 1 to Q
        representing seasonal periods (e.g., quarters 1-4, months 1-12).
    Q : int, default 4
        Number of seasonal periods per cycle. Common values:
        - 4: Quarterly data (default)
        - 12: Monthly data
        - 52: Weekly data
    never_treated_values : list or None, optional
        Values in gvar indicating never-treated units. Default recognizes
        NaN, 0, and np.inf as never-treated indicators.
    exclude_pre_periods : int, default 0
        Number of periods immediately before treatment to exclude from
        pre-treatment seasonal estimation. Used for robustness checks when
        no-anticipation assumption may be violated.

        For cohort g, the seasonal parameters are estimated using periods
        {T_min, ..., g-1-exclude_pre_periods} instead of {T_min, ..., g-1}.

    Returns
    -------
    pd.DataFrame
        Copy of input data with additional columns named ``ydot_g{g}_r{r}``
        containing the seasonally-adjusted transformed outcome for each
        (cohort, period) pair.

    Raises
    ------
    ValueError
        If required columns are missing, no valid cohorts exist, or any
        cohort has insufficient pre-treatment periods for seasonal estimation.
    InsufficientPrePeriodsError
        If any cohort has fewer than Q+1 pre-treatment periods remaining
        after applying exclude_pre_periods.

    Warns
    -----
    UserWarning
        If no never-treated units are found in the data.

    See Also
    --------
    transform_staggered_demean : Demeaning without seasonal adjustment.
    transform_staggered_detrendq : Seasonal detrending transformation.

    Notes
    -----
    The seasonal parameters are estimated once per cohort using all periods
    strictly before g. This ensures consistency with the identification
    strategy where each cohort defines its own baseline.

    Minimum required pre-treatment observations per unit is Q + 1 to ensure
    at least one residual degree of freedom for OLS estimation.
    """
    from ..exceptions import InsufficientPrePeriodsError

    # =========================================================================
    # Data Validation
    # =========================================================================
    required_cols = [y, ivar, tvar, gvar, season_var]
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Validate exclude_pre_periods parameter.
    if not isinstance(exclude_pre_periods, (int, np.integer)):
        raise TypeError(
            f"exclude_pre_periods must be an integer, got {type(exclude_pre_periods).__name__}"
        )
    if exclude_pre_periods < 0:
        raise ValueError(
            f"exclude_pre_periods must be non-negative, got {exclude_pre_periods}"
        )

    result = data.copy()
    result[tvar] = pd.to_numeric(result[tvar], errors='coerce')

    # Validate seasonal indicator values are within expected range.
    season_values = result[season_var].dropna().unique()
    valid_seasons = set(range(1, Q + 1))

    try:
        season_int_values = {int(s) for s in season_values if pd.notna(s) and s == int(s)}
        out_of_range = season_int_values - valid_seasons
    except (ValueError, TypeError):
        out_of_range = set(season_values)

    if out_of_range:
        freq_label = {4: 'quarters', 12: 'months', 52: 'weeks'}.get(Q, 'seasons')
        raise ValueError(
            f"Seasonal column '{season_var}' contains values outside valid range: "
            f"{sorted(out_of_range)}. Expected values in {{1, 2, ..., {Q}}} for {freq_label}."
        )

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

    # Seasonal demeaning requires at least Q+1 observations to estimate
    # intercept plus Q-1 seasonal dummies with positive degrees of freedom.
    min_required = Q + 1
    for g in cohorts:
        n_pre_periods_original = g - T_min
        if n_pre_periods_original < 1:
            raise ValueError(
                f"Cohort {g} has no pre-treatment periods: "
                f"earliest time period is {T_min}. Cohort must be > T_min."
            )

        # Check after exclusion
        pre_upper_bound = g - exclude_pre_periods
        n_pre_periods = pre_upper_bound - T_min
        if n_pre_periods < min_required:
            raise InsufficientPrePeriodsError(
                f"Cohort {g} requires at least {min_required} pre-treatment periods "
                f"for seasonal demeaning (Q={Q}), but only {max(0, n_pre_periods)} remain "
                f"after excluding {exclude_pre_periods} period(s). "
                f"Available periods: [{T_min}, {g-1}], "
                f"excluded: [{max(T_min, g-exclude_pre_periods)}, {g-1}]. "
                f"Consider using demean instead or reducing exclude_pre_periods.",
                cohort=g,
                available=max(0, n_pre_periods),
                required=min_required,
                excluded=exclude_pre_periods,
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
    # Compute Seasonal Demeaning Transformations
    # =========================================================================
    all_units = result[ivar].unique()

    for g in cohorts:
        post_periods = get_valid_periods_for_cohort(g, T_max)

        # Pre-allocate columns to avoid repeated DataFrame resizing overhead.
        for r in post_periods:
            col_name = f'ydot_g{int(g)}_r{r}'
            result[col_name] = np.nan

        for unit_id in all_units:
            unit_mask = result[ivar] == unit_id
            unit_data = result[unit_mask]

            # Estimate seasonal intercept and dummy coefficients from pre-treatment data.
            mu, gamma, observed_seasons = _compute_pre_treatment_seasonal_mean(
                unit_data, y, tvar, season_var, g, Q, exclude_pre_periods
            )

            if np.isnan(mu):
                # Insufficient pre-treatment data for seasonal parameter estimation.
                continue

            # Apply transformation using fixed pre-treatment parameters.
            for r in post_periods:
                col_name = f'ydot_g{int(g)}_r{r}'
                period_mask = unit_mask & (result[tvar] == r)

                if period_mask.any():
                    y_r = result.loc[period_mask, y].values[0]
                    season_r = result.loc[period_mask, season_var].values[0]

                    # Seasonal prediction: mu + gamma_q for season q.
                    predicted = mu
                    if len(gamma) > 0 and season_r in observed_seasons:
                        season_idx = observed_seasons.index(season_r)
                        # Reference season (index 0) has gamma=0 by construction.
                        if season_idx > 0:
                            predicted += gamma[season_idx - 1]

                    result.loc[period_mask, col_name] = y_r - predicted

    return result


def transform_staggered_detrendq(
    data: pd.DataFrame,
    y: str,
    ivar: str,
    tvar: str,
    gvar: str,
    season_var: str,
    Q: int = 4,
    never_treated_values: list | None = None,
    exclude_pre_periods: int = 0,
) -> pd.DataFrame:
    """
    Apply cohort-specific seasonal detrending for staggered designs.

    Computes transformed outcome for each cohort g and post-treatment
    calendar time r:

    .. math::

        \\ddot{Y}_{irg} = Y_{ir} - \\hat{\\alpha}_{ig} - \\hat{\\beta}_{ig} r
                         - \\sum_{q=2}^{Q} \\hat{\\gamma}_{qg} D_q

    where parameters are estimated from pre-treatment periods (t < g).

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel data with columns for outcome, unit identifier,
        time period, cohort assignment, and seasonal indicator.
    y : str
        Outcome variable column name.
    ivar : str
        Unit identifier column name.
    tvar : str
        Time variable column name (must be numeric or coercible to numeric).
    gvar : str
        Cohort variable column name indicating first treatment period.
    season_var : str
        Seasonal indicator column name. Values should be integers from 1 to Q
        representing seasonal periods (e.g., quarters 1-4, months 1-12).
    Q : int, default 4
        Number of seasonal periods per cycle. Common values:
        - 4: Quarterly data (default)
        - 12: Monthly data
        - 52: Weekly data
    never_treated_values : list or None, optional
        Values in gvar indicating never-treated units. Default recognizes
        NaN, 0, and np.inf as never-treated indicators.
    exclude_pre_periods : int, default 0
        Number of periods immediately before treatment to exclude from
        pre-treatment seasonal trend estimation. Used for robustness checks
        when no-anticipation assumption may be violated.

        For cohort g, the seasonal trend parameters are estimated using periods
        {T_min, ..., g-1-exclude_pre_periods} instead of {T_min, ..., g-1}.

    Returns
    -------
    pd.DataFrame
        Copy of input data with additional columns named ``ycheck_g{g}_r{r}``
        containing the seasonally-adjusted detrended outcome for each
        (cohort, period) pair.

    Raises
    ------
    ValueError
        If required columns are missing, no valid cohorts exist, or any
        cohort has insufficient pre-treatment periods for seasonal detrending.
    InsufficientPrePeriodsError
        If any cohort has fewer than Q+2 pre-treatment periods remaining
        after applying exclude_pre_periods.

    Warns
    -----
    UserWarning
        If no never-treated units are found in the data.

    See Also
    --------
    transform_staggered_detrend : Detrending without seasonal adjustment.
    transform_staggered_demeanq : Seasonal demeaning transformation.

    Notes
    -----
    The seasonal and trend parameters are estimated once per cohort using
    all periods strictly before g. Time is centered at the pre-treatment
    mean for numerical stability.

    Minimum required pre-treatment observations per unit is Q + 2 to ensure
    at least one residual degree of freedom for OLS estimation (intercept +
    slope + Q-1 seasonal dummies = Q+1 parameters).
    """
    from ..exceptions import InsufficientPrePeriodsError

    # =========================================================================
    # Data Validation
    # =========================================================================
    required_cols = [y, ivar, tvar, gvar, season_var]
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Validate exclude_pre_periods parameter.
    if not isinstance(exclude_pre_periods, (int, np.integer)):
        raise TypeError(
            f"exclude_pre_periods must be an integer, got {type(exclude_pre_periods).__name__}"
        )
    if exclude_pre_periods < 0:
        raise ValueError(
            f"exclude_pre_periods must be non-negative, got {exclude_pre_periods}"
        )

    result = data.copy()
    result[tvar] = pd.to_numeric(result[tvar], errors='coerce')

    # Validate seasonal indicator values are within expected range.
    season_values = result[season_var].dropna().unique()
    valid_seasons = set(range(1, Q + 1))

    try:
        season_int_values = {int(s) for s in season_values if pd.notna(s) and s == int(s)}
        out_of_range = season_int_values - valid_seasons
    except (ValueError, TypeError):
        out_of_range = set(season_values)

    if out_of_range:
        freq_label = {4: 'quarters', 12: 'months', 52: 'weeks'}.get(Q, 'seasons')
        raise ValueError(
            f"Seasonal column '{season_var}' contains values outside valid range: "
            f"{sorted(out_of_range)}. Expected values in {{1, 2, ..., {Q}}} for {freq_label}."
        )

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

    # Seasonal detrending requires at least Q+2 observations to estimate
    # intercept, slope, and Q-1 seasonal dummies with positive degrees of freedom.
    min_required = Q + 2
    for g in cohorts:
        n_pre_periods_original = g - T_min
        if n_pre_periods_original < 2:
            raise ValueError(
                f"Cohort {g} has only {n_pre_periods_original} pre-treatment period(s). "
                f"Seasonal detrending requires at least 2 pre-treatment "
                f"periods to estimate linear trend. Consider using demeanq instead."
            )

        # Check after exclusion
        pre_upper_bound = g - exclude_pre_periods
        n_pre_periods = pre_upper_bound - T_min
        if n_pre_periods < min_required:
            raise InsufficientPrePeriodsError(
                f"Cohort {g} requires at least {min_required} pre-treatment periods "
                f"for seasonal detrending (Q={Q}), but only {max(0, n_pre_periods)} remain "
                f"after excluding {exclude_pre_periods} period(s). "
                f"Available periods: [{T_min}, {g-1}], "
                f"excluded: [{max(T_min, g-exclude_pre_periods)}, {g-1}]. "
                f"Consider using demeanq instead or reducing exclude_pre_periods.",
                cohort=g,
                available=max(0, n_pre_periods),
                required=min_required,
                excluded=exclude_pre_periods,
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
    # Compute Seasonal Detrending Transformations
    # =========================================================================
    all_units = result[ivar].unique()

    for g in cohorts:
        post_periods = get_valid_periods_for_cohort(g, T_max)

        # Pre-allocate columns to avoid repeated DataFrame resizing overhead.
        for r in post_periods:
            col_name = f'ycheck_g{int(g)}_r{r}'
            result[col_name] = np.nan

        for unit_id in all_units:
            unit_mask = result[ivar] == unit_id
            unit_data = result[unit_mask]

            # Estimate trend and seasonal parameters from pre-treatment data.
            alpha, beta, gamma, observed_seasons, t_mean = _compute_pre_treatment_seasonal_trend(
                unit_data, y, tvar, season_var, g, Q, exclude_pre_periods
            )

            if np.isnan(alpha) or np.isnan(beta):
                # Insufficient pre-treatment data for trend parameter estimation.
                continue

            # Apply transformation using fixed pre-treatment parameters.
            for r in post_periods:
                col_name = f'ycheck_g{int(g)}_r{r}'
                period_mask = unit_mask & (result[tvar] == r)

                if period_mask.any():
                    y_r = result.loc[period_mask, y].values[0]
                    season_r = result.loc[period_mask, season_var].values[0]

                    # Center time at pre-treatment mean for numerical stability.
                    t_centered = r - t_mean
                    predicted = alpha + beta * t_centered

                    # Add seasonal effect: gamma_q for season q.
                    if len(gamma) > 0 and season_r in observed_seasons:
                        season_idx = observed_seasons.index(season_r)
                        # Reference season (index 0) has gamma=0 by construction.
                        if season_idx > 0:
                            predicted += gamma[season_idx - 1]

                    result.loc[period_mask, col_name] = y_r - predicted

    return result

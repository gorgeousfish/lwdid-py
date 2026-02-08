"""
Control group selection for staggered difference-in-differences.

This module provides functions for identifying and validating control
groups in staggered adoption designs. Three control group strategies
are supported:

- **Never-Treated (NT)**: Units that never receive treatment throughout
  the observation period. These form a stable comparison group whose
  composition does not change across calendar time periods.

- **Not-Yet-Treated (NYT)**: Units scheduled for future treatment but
  not yet treated at the current period. Under conditional parallel
  trends and no anticipation, these units provide valid controls before
  their own treatment begins, expanding the control pool for improved
  estimation efficiency.

- **All Others**: All units not in the focal treatment cohort, including
  already-treated units. This strategy may introduce forbidden
  comparisons and is generally not recommended for identification, but
  is provided for replication and diagnostic purposes.

For estimating the ATT of cohort g in period r, valid controls are
units with first treatment period strictly greater than r (gvar > r),
which includes both never-treated units and units first treated after
period r.

Notes
-----
The strict inequality criterion (gvar > r rather than gvar >= r) is
fundamental to correct identification. Units beginning treatment in
period r (i.e., gvar == r) belong to the treatment group for that
period, not the control group. This ensures that comparisons are made
only against units that remain untreated throughout period r.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd


class ControlGroupStrategy(Enum):
    """
    Enumeration of control group selection strategies.

    Defines which units are eligible to serve as controls when
    estimating treatment effects for a given cohort-period pair.
    The choice of strategy affects both the valid comparison group
    and the identifying assumptions required.

    Attributes
    ----------
    NEVER_TREATED : str
        Use only units that never receive treatment throughout the
        observation window. Provides a stable control group whose
        composition does not vary across periods. Required for
        aggregated effect estimation where controls must be consistent.
    NOT_YET_TREATED : str
        Use never-treated units plus units not yet treated at the
        current period. Expands the control pool by including future
        treatment cohorts as temporary controls, improving estimation
        efficiency under conditional parallel trends and no anticipation.
    ALL_OTHERS : str
        Use all units not in the focal treatment cohort, including
        already-treated units from earlier cohorts. May induce forbidden
        comparisons that violate identification assumptions. Provided
        primarily for replication and diagnostic purposes.
    AUTO : str
        Automatically select based on data availability. Prefers the
        not-yet-treated strategy when sufficient controls are available,
        falling back to never-treated only when necessary.

    See Also
    --------
    get_valid_control_units : Apply strategy to select control units.
    count_control_units_by_strategy : Compare control counts across strategies.
    """

    NEVER_TREATED = 'never_treated'
    NOT_YET_TREATED = 'not_yet_treated'
    ALL_OTHERS = 'all_others'
    AUTO = 'auto'


def identify_never_treated_units(
    data: pd.DataFrame,
    gvar: str,
    ivar: str,
    never_treated_values: list | None = None,
) -> pd.Series:
    """
    Identify units that never receive treatment.

    Creates a boolean mask indicating which units are classified as
    never-treated based on their treatment timing variable values.

    Parameters
    ----------
    data : pd.DataFrame
        Panel dataset containing unit and treatment timing information.
    gvar : str
        Column name indicating first treatment period for each unit.
    ivar : str
        Column name containing unit identifiers.
    never_treated_values : list, optional
        Values in gvar indicating never-treated status. Defaults to
        [0, np.inf]. Units with NaN in gvar are also classified as
        never-treated regardless of this parameter.

    Returns
    -------
    pd.Series
        Boolean Series indexed by ivar (unit ID). True indicates
        never-treated status.

    Raises
    ------
    ValueError
        If the input data is empty.
    KeyError
        If gvar or ivar column is not found in the data.

    See Also
    --------
    has_never_treated_units : Check presence of never-treated units.
    get_valid_control_units : Select control units for estimation.

    Notes
    -----
    Never-treated units are identified through three mechanisms:

    1. Missing values (NaN) in gvar, representing units with no
       recorded treatment date.
    2. Zero values, a common coding convention for never-treated.
    3. Infinity values, representing treatment dates beyond the
       observation window.
    """
    if len(data) == 0:
        raise ValueError("Input data is empty")
    if gvar not in data.columns:
        raise KeyError(f"Column '{gvar}' not found in data")
    if ivar not in data.columns:
        raise KeyError(f"Column '{ivar}' not found in data")

    # Extract first gvar value per unit; panel data may have repeated rows
    unit_gvar = data.groupby(ivar)[gvar].first()

    # Default sentinel values: 0 and infinity are common conventions
    if never_treated_values is None:
        nt_values = [0, np.inf]
    else:
        nt_values = never_treated_values

    # NaN always indicates never-treated (missing treatment date)
    never_treated_mask = unit_gvar.isna()

    # Include units matching any specified sentinel value
    if len(nt_values) > 0:
        never_treated_mask = never_treated_mask | unit_gvar.isin(nt_values)

    return never_treated_mask


def has_never_treated_units(
    data: pd.DataFrame,
    gvar: str,
    ivar: str,
    never_treated_values: list | None = None,
) -> bool:
    """
    Check whether the data contains any never-treated units.

    A convenience function for quickly determining if a never-treated
    control group is available for estimation. This is particularly
    useful for deciding whether aggregated effects can be estimated.

    Parameters
    ----------
    data : pd.DataFrame
        Panel dataset containing unit and treatment timing information.
    gvar : str
        Column name indicating first treatment period for each unit.
    ivar : str
        Column name containing unit identifiers.
    never_treated_values : list, optional
        Values in gvar indicating never-treated status. Defaults to
        [0, np.inf].

    Returns
    -------
    bool
        True if at least one never-treated unit exists.

    See Also
    --------
    identify_never_treated_units : Get full mask of never-treated units.
    validate_control_group : Validate control group for aggregation.
    """
    nt_mask = identify_never_treated_units(data, gvar, ivar, never_treated_values)
    return nt_mask.sum() > 0


def get_valid_control_units(
    data: pd.DataFrame,
    gvar: str,
    ivar: str,
    cohort: int | float,
    period: int | float,
    strategy: ControlGroupStrategy = ControlGroupStrategy.NOT_YET_TREATED,
    never_treated_values: list | None = None,
    is_pre_treatment: bool = False,
) -> pd.Series:
    """
    Determine valid control units for a specific cohort-period pair.

    For estimating the ATT of cohort g in period r, identifies which
    units can serve as valid controls based on the selected strategy
    and the fundamental strict inequality criterion.

    Parameters
    ----------
    data : pd.DataFrame
        Panel dataset containing unit and treatment timing information.
    gvar : str
        Column name indicating first treatment period for each unit.
    ivar : str
        Column name containing unit identifiers.
    cohort : int or float
        Treatment cohort (first treatment period g) of the treated group.
    period : int or float
        Calendar time period r for which to identify controls.
        For post-treatment: must satisfy period >= cohort.
        For pre-treatment: must satisfy period < cohort.
    strategy : ControlGroupStrategy, default NOT_YET_TREATED
        Strategy for selecting control units.
    never_treated_values : list, optional
        Values in gvar indicating never-treated status. Defaults to
        [0, np.inf].
    is_pre_treatment : bool, default False
        If True, selects control units for pre-treatment period estimation
        (parallel trends testing). For pre-treatment periods t < g, the
        control group includes all units not yet treated at period t.

    Returns
    -------
    pd.Series
        Boolean Series indexed by unit ID where True indicates valid control.

    Raises
    ------
    ValueError
        If data is empty, or period constraints are violated.
    KeyError
        If required columns are not found.
    TypeError
        If gvar column is not numeric.

    See Also
    --------
    get_all_control_masks : Batch computation for multiple cohort-period pairs.
    get_all_control_masks_pre : Batch computation for pre-treatment periods.
    validate_control_group : Validate control group size requirements.

    Notes
    -----
    The strict inequality criterion (gvar > period) is fundamental:

    - Units with gvar == period are beginning treatment in period r and
      thus belong to the treatment group, not the control group.
    - This ensures valid controls have not yet been exposed to treatment.

    For post-treatment estimation (period >= cohort):
        The treatment cohort is automatically excluded because cohort
        units have gvar == cohort <= period, failing the gvar > period
        criterion.

    For pre-treatment estimation (period < cohort):
        The treatment cohort is correctly included as controls because
        period < cohort implies gvar (== cohort) > period. At pre-treatment
        periods, these units are not yet treated and serve as valid
        comparisons for parallel trends assessment.
    """
    # -------------------------------------------------------------------------
    # Input Validation
    # -------------------------------------------------------------------------
    if len(data) == 0:
        raise ValueError("Input data is empty")
    if gvar not in data.columns:
        raise KeyError(f"Column '{gvar}' not found in data")
    if ivar not in data.columns:
        raise KeyError(f"Column '{ivar}' not found in data")

    # Require numeric gvar for comparison operations
    if not pd.api.types.is_numeric_dtype(data[gvar]):
        raise TypeError(
            f"gvar column '{gvar}' must be numeric, got {data[gvar].dtype}. "
            f"String values like 'never' or '2005' are not supported."
        )

    # Convert to float for consistent comparison across int/float inputs
    cohort_f = float(cohort)
    period_f = float(period)

    # Validate period constraints based on pre/post treatment context
    if is_pre_treatment:
        if period_f >= cohort_f:
            raise ValueError(
                f"For pre-treatment estimation, period ({period}) must be < cohort ({cohort}). "
                f"Pre-treatment effects are only defined for periods t < g."
            )
    else:
        if period_f < cohort_f:
            raise ValueError(
                f"period ({period}) must be >= cohort ({cohort}). "
                f"Treatment effects are only defined for periods r >= g."
            )

    # -------------------------------------------------------------------------
    # Identify Never-Treated Units
    # -------------------------------------------------------------------------
    # Extract first gvar value per unit from panel data
    unit_gvar = data.groupby(ivar)[gvar].first()

    # Default sentinel values for never-treated status
    if never_treated_values is None:
        nt_values = [0, np.inf]
    else:
        nt_values = never_treated_values

    # Build never-treated mask from NaN and sentinel values
    never_treated_mask = unit_gvar.isna()
    if len(nt_values) > 0:
        never_treated_mask = never_treated_mask | unit_gvar.isin(nt_values)

    # -------------------------------------------------------------------------
    # Build Control Mask by Strategy
    # -------------------------------------------------------------------------
    if strategy == ControlGroupStrategy.NEVER_TREATED:
        # Use only units that never receive treatment
        control_mask = never_treated_mask

    elif strategy == ControlGroupStrategy.NOT_YET_TREATED:
        # Include never-treated plus units first treated after current period
        # Strict inequality: gvar > period excludes units starting treatment now
        not_yet_treated_mask = (unit_gvar > period_f)
        control_mask = never_treated_mask | not_yet_treated_mask

    elif strategy == ControlGroupStrategy.ALL_OTHERS:
        # Include all units except the focal treatment cohort
        # Warning: may include already-treated units from earlier cohorts
        control_mask = (unit_gvar != cohort_f)

    else:  # AUTO strategy
        # Prefer not-yet-treated if available, fallback to never-treated
        not_yet_treated_mask = (unit_gvar > period_f)
        nyt_plus_nt_mask = never_treated_mask | not_yet_treated_mask

        if nyt_plus_nt_mask.sum() > 0:
            control_mask = nyt_plus_nt_mask
        else:
            control_mask = never_treated_mask

    return control_mask


def get_all_control_masks(
    data: pd.DataFrame,
    gvar: str,
    ivar: str,
    cohorts: list[int | float],
    T_max: int | float,
    T_min: int | float | None = None,
    strategy: ControlGroupStrategy = ControlGroupStrategy.NOT_YET_TREATED,
    never_treated_values: list | None = None,
) -> dict[tuple[int | float, int | float], pd.Series]:
    """
    Compute control group masks for all cohort-period combinations.

    Efficiently generates control masks for multiple cohort-period pairs
    by pre-computing shared data structures. This batch approach avoids
    redundant groupby operations when estimating effects across many
    (cohort, period) combinations.

    Parameters
    ----------
    data : pd.DataFrame
        Panel dataset containing unit and treatment timing information.
    gvar : str
        Column name indicating first treatment period for each unit.
    ivar : str
        Column name containing unit identifiers.
    cohorts : list of int or float
        Treatment cohorts for which to generate control masks.
    T_max : int or float
        Maximum time period to consider (inclusive).
    T_min : int or float, optional
        Minimum time period. Reserved for future extension.
    strategy : ControlGroupStrategy, default NOT_YET_TREATED
        Strategy for selecting control units.
    never_treated_values : list, optional
        Values in gvar indicating never-treated status. Defaults to
        [0, np.inf].

    Returns
    -------
    dict
        Dictionary mapping (cohort, period) tuples to boolean Series
        indexed by unit ID. True indicates valid control status.

    Raises
    ------
    ValueError
        If the input data is empty.

    See Also
    --------
    get_valid_control_units : Single cohort-period control mask.
    get_all_control_masks_pre : Batch computation for pre-treatment periods.

    Notes
    -----
    For each cohort g, masks are generated for post-treatment periods
    {g, g+1, ..., T_max}. The never-treated mask is computed once and
    reused across all cohort-period pairs, while not-yet-treated masks
    vary by period due to the strict inequality criterion.
    """
    if len(data) == 0:
        raise ValueError("Input data is empty")

    # -------------------------------------------------------------------------
    # Pre-compute Shared Data Structures
    # -------------------------------------------------------------------------
    # Extract unit-level gvar once to avoid repeated groupby operations
    unit_gvar = data.groupby(ivar)[gvar].first()

    # Build never-treated mask (constant across all periods)
    if never_treated_values is None:
        nt_values = [0, np.inf]
    else:
        nt_values = never_treated_values

    never_treated_mask = unit_gvar.isna()
    if len(nt_values) > 0:
        never_treated_mask = never_treated_mask | unit_gvar.isin(nt_values)

    T_max_f = float(T_max)

    # -------------------------------------------------------------------------
    # Generate Masks for Each (cohort, period) Pair
    # -------------------------------------------------------------------------
    results = {}

    for g in cohorts:
        g_f = float(g)
        # Iterate over post-treatment periods: g, g+1, ..., T_max
        r = g_f
        while r <= T_max_f:
            if strategy == ControlGroupStrategy.NEVER_TREATED:
                control_mask = never_treated_mask.copy()

            elif strategy == ControlGroupStrategy.NOT_YET_TREATED:
                # Not-yet-treated: units with first treatment after period r
                not_yet_treated_mask = (unit_gvar > r)
                control_mask = never_treated_mask | not_yet_treated_mask

            elif strategy == ControlGroupStrategy.ALL_OTHERS:
                # All non-cohort units (may include already-treated)
                control_mask = (unit_gvar != g_f)

            else:  # AUTO strategy
                not_yet_treated_mask = (unit_gvar > r)
                nyt_plus_nt_mask = never_treated_mask | not_yet_treated_mask
                if nyt_plus_nt_mask.sum() > 0:
                    control_mask = nyt_plus_nt_mask
                else:
                    control_mask = never_treated_mask.copy()

            results[(g, r)] = control_mask
            r += 1

    return results


def get_all_control_masks_pre(
    data: pd.DataFrame,
    gvar: str,
    ivar: str,
    cohorts: list[int | float],
    T_min: int | float,
    strategy: ControlGroupStrategy = ControlGroupStrategy.NOT_YET_TREATED,
    never_treated_values: list | None = None,
) -> dict[tuple[int | float, int | float], pd.Series]:
    """
    Compute control group masks for all pre-treatment cohort-period combinations.

    Efficiently generates control masks for pre-treatment periods by
    pre-computing shared data structures. Used for parallel trends
    testing and event study visualization where pre-treatment effects
    should be approximately zero under the identifying assumptions.

    Parameters
    ----------
    data : pd.DataFrame
        Panel dataset containing unit and treatment timing information.
    gvar : str
        Column name indicating first treatment period for each unit.
    ivar : str
        Column name containing unit identifiers.
    cohorts : list of int or float
        Treatment cohorts for which to generate control masks.
    T_min : int or float
        Minimum time period in the data (inclusive).
    strategy : ControlGroupStrategy, default NOT_YET_TREATED
        Strategy for selecting control units.
    never_treated_values : list, optional
        Values in gvar indicating never-treated status. Defaults to
        [0, np.inf].

    Returns
    -------
    dict
        Dictionary mapping (cohort, period) tuples to boolean Series
        indexed by unit ID. True indicates valid control status.

    Raises
    ------
    ValueError
        If the input data is empty.

    See Also
    --------
    get_valid_control_units : Single cohort-period control mask.
    get_all_control_masks : Batch computation for post-treatment periods.

    Notes
    -----
    For each cohort g, masks are generated for pre-treatment periods
    {T_min, T_min+1, ..., g-1}. At pre-treatment period t < g, the
    focal treatment cohort (gvar == g) is correctly included as
    controls because these units are not yet treated.
    """
    if len(data) == 0:
        raise ValueError("Input data is empty")

    # -------------------------------------------------------------------------
    # Pre-compute Shared Data Structures
    # -------------------------------------------------------------------------
    # Extract unit-level gvar once to avoid repeated groupby operations
    unit_gvar = data.groupby(ivar)[gvar].first()

    # Build never-treated mask (constant across all periods)
    if never_treated_values is None:
        nt_values = [0, np.inf]
    else:
        nt_values = never_treated_values

    never_treated_mask = unit_gvar.isna()
    if len(nt_values) > 0:
        never_treated_mask = never_treated_mask | unit_gvar.isin(nt_values)

    T_min_f = float(T_min)

    # -------------------------------------------------------------------------
    # Generate Masks for Each (cohort, period) Pair
    # -------------------------------------------------------------------------
    results = {}

    for g in cohorts:
        g_f = float(g)
        # Iterate over pre-treatment periods: T_min, T_min+1, ..., g-1
        t = T_min_f
        while t < g_f:
            if strategy == ControlGroupStrategy.NEVER_TREATED:
                control_mask = never_treated_mask.copy()

            elif strategy == ControlGroupStrategy.NOT_YET_TREATED:
                # Include all units not yet treated at period t
                not_yet_treated_mask = (unit_gvar > t)
                control_mask = never_treated_mask | not_yet_treated_mask

            elif strategy == ControlGroupStrategy.ALL_OTHERS:
                # All non-cohort units (may include already-treated)
                control_mask = (unit_gvar != g_f)

            else:  # AUTO strategy
                not_yet_treated_mask = (unit_gvar > t)
                nyt_plus_nt_mask = never_treated_mask | not_yet_treated_mask
                if nyt_plus_nt_mask.sum() > 0:
                    control_mask = nyt_plus_nt_mask
                else:
                    control_mask = never_treated_mask.copy()

            results[(g, t)] = control_mask
            t += 1

    return results


def validate_control_group(
    control_mask: pd.Series,
    cohort: int | float,
    period: int | float,
    min_control_units: int = 1,
    aggregate_type: str | None = None,
    has_never_treated: bool = True,
    strategy: ControlGroupStrategy | None = None,
) -> tuple[bool, str]:
    """
    Validate whether a control group meets estimation requirements.

    Checks control group suitability for treatment effect estimation,
    including minimum size requirements and aggregation constraints.

    Parameters
    ----------
    control_mask : pd.Series
        Boolean Series indexed by unit ID indicating control group membership.
    cohort : int or float
        Treatment cohort being estimated.
    period : int or float
        Time period being estimated.
    min_control_units : int, default 1
        Minimum number of control units required for estimation.
    aggregate_type : str, optional
        Type of aggregation ('cohort' or 'overall'). Aggregated effects
        require never-treated units because not-yet-treated controls
        vary across periods and cannot form a consistent comparison group.
    has_never_treated : bool, default True
        Whether the data contains any never-treated units.
    strategy : ControlGroupStrategy, optional
        Control group strategy being used. Generates warnings when
        aggregated estimation uses non-recommended strategies.

    Returns
    -------
    is_valid : bool
        True if the control group passes all validation checks.
    message : str
        Descriptive message indicating success or failure reason.

    See Also
    --------
    get_valid_control_units : Generate control group masks.
    has_never_treated_units : Check for never-treated unit availability.

    Notes
    -----
    Validation checks are applied in priority order:

    1. Non-empty control group (required for any estimation)
    2. Minimum size requirement (ensures sufficient degrees of freedom)
    3. Aggregation constraints: cohort-level and overall effects require
       never-treated units because they aggregate across multiple periods,
       and not-yet-treated units transition out of the control group as
       they become treated
    """
    n_controls = control_mask.sum()

    # Check 1: Non-empty control group
    if n_controls == 0:
        return False, f"No control units found for cohort={cohort}, period={period}."

    # Check 2: Minimum size requirement
    if n_controls < min_control_units:
        return False, (
            f"Insufficient control units for cohort={cohort}, period={period}. "
            f"Found {n_controls}, required {min_control_units}."
        )

    # Check 3: Aggregation constraints
    if aggregate_type in ('cohort', 'overall'):
        # Aggregated effects need consistent controls across periods
        if not has_never_treated:
            return False, (
                f"Cannot estimate {aggregate_type} effects without never-treated units. "
                f"When all units are eventually treated, only cohort-period-specific "
                f"effects can be estimated using not-yet-treated controls."
            )

        # Recommend never-treated strategy for aggregated estimation
        if strategy is not None and strategy != ControlGroupStrategy.NEVER_TREATED:
            return True, (
                f"Valid control group with {n_controls} units. "
                f"Warning: For {aggregate_type} effect estimation, it is recommended "
                f"to use 'never_treated' control group strategy for robustness."
            )

    return True, f"Valid control group with {n_controls} units for cohort={cohort}, period={period}."


def count_control_units_by_strategy(
    data: pd.DataFrame,
    gvar: str,
    ivar: str,
    cohort: int | float,
    period: int | float,
    never_treated_values: list | None = None,
) -> dict[str, int]:
    """
    Count available control units under different selection strategies.

    A diagnostic function to help users understand data structure and
    make informed decisions about control group selection.

    Parameters
    ----------
    data : pd.DataFrame
        Panel dataset containing unit and treatment timing information.
    gvar : str
        Column name indicating first treatment period for each unit.
    ivar : str
        Column name containing unit identifiers.
    cohort : int or float
        Treatment cohort of interest.
    period : int or float
        Time period of interest.
    never_treated_values : list, optional
        Values in gvar indicating never-treated status. Defaults to
        [0, np.inf].

    Returns
    -------
    dict
        Dictionary with keys:

        - ``'never_treated'``: Count of never-treated units.
        - ``'not_yet_treated_only'``: Count of units treated in future
          periods (excluding never-treated).
        - ``'not_yet_treated_total'``: Total valid controls under the
          not-yet-treated strategy.
        - ``'treatment_cohort'``: Count of units in the treatment cohort.

    Raises
    ------
    ValueError
        If the input data is empty.

    See Also
    --------
    get_valid_control_units : Generate control masks for estimation.
    ControlGroupStrategy : Available control group selection strategies.

    Notes
    -----
    The not-yet-treated count uses strict inequality (gvar > period) to
    exclude units beginning treatment in the current period.
    """
    if len(data) == 0:
        raise ValueError("Input data is empty")
    
    # Extract unit-level gvar
    unit_gvar = data.groupby(ivar)[gvar].first()
    
    # Handle never_treated_values
    if never_treated_values is None:
        nt_values = [0, np.inf]
    else:
        nt_values = never_treated_values
    
    # Convert to float for consistent numeric comparison
    cohort_f = float(cohort)
    period_f = float(period)
    
    # Identify never-treated units via NaN or sentinel values
    nt_mask = unit_gvar.isna()
    if len(nt_values) > 0:
        nt_mask = nt_mask | unit_gvar.isin(nt_values)
    
    # Not-yet-treated excludes both never-treated and currently-treated units
    nyt_only_mask = (unit_gvar > period_f) & ~nt_mask
    treat_mask = (unit_gvar == cohort_f)
    
    return {
        'never_treated': int(nt_mask.sum()),
        'not_yet_treated_only': int(nyt_only_mask.sum()),
        'not_yet_treated_total': int(nt_mask.sum() + nyt_only_mask.sum()),
        'treatment_cohort': int(treat_mask.sum())
    }

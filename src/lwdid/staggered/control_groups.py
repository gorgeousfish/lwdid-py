"""
Control group selection for staggered difference-in-differences.

This module provides functions for identifying and validating control
groups in staggered adoption designs. Two control unit types are
supported:

- **Never-Treated (NT)**: Units that never receive treatment throughout
  the observation period, serving as a stable comparison group.

- **Not-Yet-Treated (NYT)**: Units scheduled for future treatment but
  not yet treated at the current period. Under conditional parallel
  trends, these provide valid controls before their treatment begins.

In addition, an "all others" control strategy is supported for
replicating published rolling cross-sectional specifications that set
``D_{ig} = 1{g_i = g}`` and use *all* non-cohort units as controls in each
cross-section (including already-treated units). This strategy can
introduce forbidden comparisons and is generally **not recommended** for
identification under no-anticipation, but it is useful for replication
and diagnostics.

For estimating the ATT of cohort g in period r, valid controls are
units with first treatment time strictly greater than r (gvar > r),
including both never-treated units and units first treated after r.

Notes
-----
The strict inequality (gvar > period rather than gvar >= period) is
essential: units beginning treatment in period r belong to the
treatment group, not the control group.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd


class ControlGroupStrategy(Enum):
    """
    Enumeration of control group selection strategies.

    Defines the set of units eligible to serve as controls when
    estimating treatment effects for a given cohort-period pair.

    Attributes
    ----------
    NEVER_TREATED : str
        Use only units that never receive treatment. Most conservative
        choice; required for aggregated effect estimation.
    NOT_YET_TREATED : str
        Use never-treated plus not-yet-treated units. Expands the
        control pool for improved efficiency under conditional
        parallel trends.
    ALL_OTHERS : str
        Use all units not in the treatment cohort as controls (including
        already-treated units). This may violate no-anticipation / induce
        forbidden comparisons and is mainly intended for replication and
        diagnostic purposes.
    AUTO : str
        Automatically select based on data availability. Uses
        not-yet-treated when available, falls back to never-treated
        only if necessary.

    See Also
    --------
    get_valid_control_units : Apply strategy to select control units.
    count_control_units_by_strategy : Compare counts across strategies.
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
        never-treated.

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
    """
    if len(data) == 0:
        raise ValueError("Input data is empty")
    if gvar not in data.columns:
        raise KeyError(f"Column '{gvar}' not found in data")
    if ivar not in data.columns:
        raise KeyError(f"Column '{ivar}' not found in data")
    
    unit_gvar = data.groupby(ivar)[gvar].first()
    
    if never_treated_values is None:
        nt_values = [0, np.inf]
    else:
        nt_values = never_treated_values
    
    never_treated_mask = unit_gvar.isna()
    
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
    control group is available for estimation.

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

    For estimating the ATT of cohort g in period r, identifies which units
    can serve as valid controls based on the selected strategy.

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
        If True, selects control units for pre-treatment period estimation.
        For pre-treatment periods t < g, the control group consists of
        units with gvar > t plus never-treated units.

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
    For post-treatment periods (is_pre_treatment=False):
        Control units must have first treatment period strictly greater
        than the current period (gvar > period), ensuring units beginning
        treatment in the current period are correctly classified as
        treated, not controls.

    For pre-treatment periods (is_pre_treatment=True):
        The control group is defined as {units with gvar > t} plus
        never-treated units. This includes all units not yet treated
        at period t.
    """
    # Validate inputs
    if len(data) == 0:
        raise ValueError("Input data is empty")
    if gvar not in data.columns:
        raise KeyError(f"Column '{gvar}' not found in data")
    if ivar not in data.columns:
        raise KeyError(f"Column '{ivar}' not found in data")
    
    # Check gvar is numeric
    if not pd.api.types.is_numeric_dtype(data[gvar]):
        raise TypeError(
            f"gvar column '{gvar}' must be numeric, got {data[gvar].dtype}. "
            f"String values like 'never' or '2005' are not supported."
        )
    
    # Convert to float for consistent numeric comparison across int/float inputs
    cohort_f = float(cohort)
    period_f = float(period)
    
    # Validate period constraints based on pre/post treatment
    if is_pre_treatment:
        if period_f >= cohort_f:
            raise ValueError(
                f"For pre-treatment estimation, period ({period}) must be < cohort ({cohort}). "
                f"Pre-treatment effects are only defined for periods t < g."
            )
    else:
        # ATT is only defined for post-treatment periods (r >= g)
        if period_f < cohort_f:
            raise ValueError(
                f"period ({period}) must be >= cohort ({cohort}). "
                f"Treatment effects are only defined for periods r >= g."
            )
    
    # Extract unit-level gvar values
    unit_gvar = data.groupby(ivar)[gvar].first()
    
    # Handle never_treated_values parameter
    if never_treated_values is None:
        nt_values = [0, np.inf]
    else:
        nt_values = never_treated_values
    
    # Identify never treated units
    never_treated_mask = unit_gvar.isna()
    if len(nt_values) > 0:
        never_treated_mask = never_treated_mask | unit_gvar.isin(nt_values)
    
    # Calculate control mask based on strategy
    if strategy == ControlGroupStrategy.NEVER_TREATED:
        control_mask = never_treated_mask
        
    elif strategy == ControlGroupStrategy.NOT_YET_TREATED:
        if is_pre_treatment:
            # For pre-treatment: control = {gvar > t} ∪ {never-treated}
            # This includes all units not yet treated at period t
            not_yet_treated_mask = (unit_gvar > period_f)
            control_mask = never_treated_mask | not_yet_treated_mask
        else:
            # For post-treatment: use strict inequality
            # Units starting treatment in this period (gvar == period) are treated
            not_yet_treated_mask = (unit_gvar > period_f)
            control_mask = never_treated_mask | not_yet_treated_mask
    
    elif strategy == ControlGroupStrategy.ALL_OTHERS:
        # All units except those in the treatment cohort itself.
        # This includes never-treated, not-yet-treated, and already-treated units.
        control_mask = (unit_gvar != cohort_f)
        
    else:  # AUTO
        not_yet_treated_mask = (unit_gvar > period_f)
        nyt_plus_nt_mask = never_treated_mask | not_yet_treated_mask
        
        if nyt_plus_nt_mask.sum() > 0:
            control_mask = nyt_plus_nt_mask
        else:
            control_mask = never_treated_mask
    
    # For post-treatment: Treatment cohort is excluded automatically by the
    # strict inequality: cohort units have gvar == cohort <= period, so they
    # fail gvar > period
    
    # For pre-treatment: The treatment cohort (gvar == cohort) is included
    # in the control group since period < cohort implies gvar > period for
    # the cohort units. This is correct because at pre-treatment periods,
    # the cohort units are not yet treated.
    
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
    by pre-computing shared data structures to avoid redundant calculations.

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
        Maximum time period to consider.
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
    For each cohort g, masks are generated for periods {g, g+1, ...,
    T_max}. The never-treated mask is pre-computed once and reused
    across all cohort-period pairs for efficiency.
    """
    if len(data) == 0:
        raise ValueError("Input data is empty")
    
    # Pre-compute shared data
    unit_gvar = data.groupby(ivar)[gvar].first()
    
    if never_treated_values is None:
        nt_values = [0, np.inf]
    else:
        nt_values = never_treated_values
    
    never_treated_mask = unit_gvar.isna()
    if len(nt_values) > 0:
        never_treated_mask = never_treated_mask | unit_gvar.isin(nt_values)
    
    # Convert to float for consistent numeric comparison
    T_max_f = float(T_max)
    
    results = {}
    
    for g in cohorts:
        g_f = float(g)
        # Period range: from cohort g to T_max
        r = g_f
        while r <= T_max_f:
            if strategy == ControlGroupStrategy.NEVER_TREATED:
                control_mask = never_treated_mask.copy()
                
            elif strategy == ControlGroupStrategy.NOT_YET_TREATED:
                not_yet_treated_mask = (unit_gvar > r)
                control_mask = never_treated_mask | not_yet_treated_mask
            
            elif strategy == ControlGroupStrategy.ALL_OTHERS:
                control_mask = (unit_gvar != g_f)
                
            else:  # AUTO
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
    pre-computing shared data structures to avoid redundant calculations.

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
        Minimum time period in the data.
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
    {T_min, T_min+1, ..., g-1}. For pre-treatment period t, the control
    group is defined as {units with gvar > t} plus never-treated units.
    """
    if len(data) == 0:
        raise ValueError("Input data is empty")
    
    # Pre-compute shared data
    unit_gvar = data.groupby(ivar)[gvar].first()
    
    if never_treated_values is None:
        nt_values = [0, np.inf]
    else:
        nt_values = never_treated_values
    
    never_treated_mask = unit_gvar.isna()
    if len(nt_values) > 0:
        never_treated_mask = never_treated_mask | unit_gvar.isin(nt_values)
    
    # Convert to float for consistent numeric comparison
    T_min_f = float(T_min)
    
    results = {}
    
    for g in cohorts:
        g_f = float(g)
        # Pre-treatment period range: from T_min to g-1
        t = T_min_f
        while t < g_f:
            if strategy == ControlGroupStrategy.NEVER_TREATED:
                control_mask = never_treated_mask.copy()
                
            elif strategy == ControlGroupStrategy.NOT_YET_TREATED:
                # For pre-treatment: control = {gvar > t} ∪ {never-treated}
                not_yet_treated_mask = (unit_gvar > t)
                control_mask = never_treated_mask | not_yet_treated_mask
            
            elif strategy == ControlGroupStrategy.ALL_OTHERS:
                # All units except those in the treatment cohort itself.
                control_mask = (unit_gvar != g_f)
                
            else:  # AUTO
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
    including size requirements and aggregation constraints.

    Parameters
    ----------
    control_mask : pd.Series
        Boolean Series indexed by unit ID indicating control group membership.
    cohort : int or float
        Treatment cohort being estimated.
    period : int or float
        Time period being estimated.
    min_control_units : int, default 1
        Minimum number of control units required.
    aggregate_type : str, optional
        Type of aggregation ('cohort' or 'overall'). Aggregated effects
        require never-treated units for proper identification.
    has_never_treated : bool, default True
        Whether the data contains never-treated units.
    strategy : ControlGroupStrategy, optional
        Control group strategy being used. Generates warnings for
        aggregated estimation when appropriate.

    Returns
    -------
    is_valid : bool
        True if the control group passes all validation checks.
    message : str
        Descriptive message indicating success or failure reason.

    See Also
    --------
    get_valid_control_units : Generate control group masks.

    Notes
    -----
    Validation checks in priority order:

    1. Non-empty control group
    2. Minimum size requirement
    3. Aggregation constraints (cohort/overall effects require
       never-treated units since not-yet-treated controls vary
       across periods)
    """
    n_controls = control_mask.sum()
    
    # Check 1: Empty control group
    if n_controls == 0:
        return False, f"No control units found for cohort={cohort}, period={period}."
    
    # Check 2: Minimum size
    if n_controls < min_control_units:
        return False, (
            f"Insufficient control units for cohort={cohort}, period={period}. "
            f"Found {n_controls}, required {min_control_units}."
        )
    
    # Check 3: Aggregate constraints
    if aggregate_type in ('cohort', 'overall'):
        if not has_never_treated:
            return False, (
                f"Cannot estimate {aggregate_type} effects without never-treated units. "
                f"When all units are eventually treated, only cohort-period-specific "
                f"effects can be estimated using not-yet-treated controls."
            )
        
        # Warn if strategy is not NEVER_TREATED for aggregate estimation
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

"""
Control Group Selection Module for Staggered DiD.

This module implements control group selection logic for staggered
difference-in-differences estimation based on Lee and Wooldridge (2023, 2025).

Key concepts:
- Never Treated (NT): Units that never receive treatment
- Not-Yet-Treated (NYT): Units that will be treated in the future but haven't been yet

Reference:
    Lee and Wooldridge (2023) Section 4, Theorem 4.1 and Procedure 4.1
    
Critical Algorithm Rule:
    Control group A_{r+1} = D_{r+1} + D_{r+2} + ... + D_T + D_inf
    
    This means: gvar > period (STRICT greater than, NOT >=)
    
    - gvar == period: Unit starts treatment in period r (NOT control!)
    - gvar > period: Unit has not yet started treatment (IS control)
"""

from enum import Enum
from typing import Dict, Tuple, List, Optional, Union
import warnings

import pandas as pd
import numpy as np


class ControlGroupStrategy(Enum):
    """Control group selection strategy enumeration."""
    NEVER_TREATED = 'never_treated'
    NOT_YET_TREATED = 'not_yet_treated'
    AUTO = 'auto'


def identify_never_treated_units(
    data: pd.DataFrame,
    gvar: str,
    ivar: str,
    never_treated_values: Optional[List] = None,
) -> pd.Series:
    """Identify never treated units in the data."""
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
    never_treated_values: Optional[List] = None,
) -> bool:
    """Quick check if data contains never treated units."""
    nt_mask = identify_never_treated_units(data, gvar, ivar, never_treated_values)
    return nt_mask.sum() > 0


def get_valid_control_units(
    data: pd.DataFrame,
    gvar: str,
    ivar: str,
    cohort: Union[int, float],
    period: Union[int, float],
    strategy: ControlGroupStrategy = ControlGroupStrategy.NOT_YET_TREATED,
    never_treated_values: Optional[List] = None,
) -> pd.Series:
    """
    Get valid control unit mask for given cohort and period.
    
    Core Logic (from Lee and Wooldridge 2023 Section 4):
    - Control group A_{r+1} = D_{r+1} + D_{r+2} + ... + D_T + D_inf
    - gvar > period (STRICT) means not-yet-treated
    - gvar == period means starts treatment in period, NOT control!
    
    Returns unit-level mask (index = unit IDs), not row-level.
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
    
    # Convert to float for consistent comparison
    cohort_f = float(cohort)
    period_f = float(period)
    
    # Validate period >= cohort (paper definition: tau_gr only for r >= g)
    if period_f < cohort_f:
        raise ValueError(
            f"period ({period}) must be >= cohort ({cohort}). "
            f"Treatment effect tau_{{g,r}} is only defined for r >= g."
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
        # CRITICAL: Use strict > (not >=)
        # gvar == period means starts treatment in period, NOT control!
        not_yet_treated_mask = (unit_gvar > period_f)
        control_mask = never_treated_mask | not_yet_treated_mask
        
    else:  # AUTO
        not_yet_treated_mask = (unit_gvar > period_f)
        nyt_plus_nt_mask = never_treated_mask | not_yet_treated_mask
        
        if nyt_plus_nt_mask.sum() > 0:
            control_mask = nyt_plus_nt_mask
        else:
            control_mask = never_treated_mask
    
    # Control mask should NOT include treatment cohort itself
    # This is automatically handled:
    # - cohort's gvar == cohort, not NaN/0/inf, so not in NT
    # - cohort's gvar == cohort, not > period (since period >= cohort)
    
    return control_mask


def get_all_control_masks(
    data: pd.DataFrame,
    gvar: str,
    ivar: str,
    cohorts: List[Union[int, float]],
    T_max: Union[int, float],
    T_min: Optional[Union[int, float]] = None,
    strategy: ControlGroupStrategy = ControlGroupStrategy.NOT_YET_TREATED,
    never_treated_values: Optional[List] = None,
) -> Dict[Tuple[Union[int, float], Union[int, float]], pd.Series]:
    """
    Batch get control group masks for all (cohort, period) pairs.
    
    For each cohort g, generates masks for period in {g, g+1, ..., T_max}.
    
    Performance optimizations:
    1. Only compute unit_gvar once
    2. Only compute never_treated_mask once
    3. For each (g, r), only compute not_yet_treated part
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
    
    # Convert T_max to float for consistent comparison
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


def validate_control_group(
    control_mask: pd.Series,
    cohort: Union[int, float],
    period: Union[int, float],
    min_control_units: int = 1,
    aggregate_type: Optional[str] = None,
    has_never_treated: bool = True,
    strategy: Optional[ControlGroupStrategy] = None,
) -> Tuple[bool, str]:
    """
    Validate if control group meets estimation requirements.
    
    Validation logic (by priority):
    1. Is control group empty?
    2. Does control group meet minimum size requirement?
    3. Aggregate type constraint check:
       - If aggregate_type in {'cohort', 'overall'}, must have NT units
       - If strategy != NEVER_TREATED, should warn and suggest switching
    
    Returns
    -------
    Tuple[bool, str]
        (is_valid, message)
        - is_valid: whether validation passed
        - message: description (success or error reason)
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
                f"Cannot estimate {aggregate_type} effects without never treated units. "
                f"All units are eventually treated (All Eventually Treated case). "
                f"Only (g,r)-specific effects can be estimated using not-yet-treated controls. "
                f"See Lee & Wooldridge (2023) Section 4.2 for theoretical details."
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
    cohort: Union[int, float],
    period: Union[int, float],
    never_treated_values: Optional[List] = None,
) -> Dict[str, int]:
    """
    Count control units by different strategies (diagnostic tool).
    
    Helps users understand data structure and choose appropriate strategy.
    
    Returns
    -------
    Dict[str, int]
        Control counts by strategy:
        {
            'never_treated': NT unit count,
            'not_yet_treated_only': pure NYT count (excluding NT),
            'not_yet_treated_total': NYT + NT total,
            'treatment_cohort': treatment group unit count
        }
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
    
    # Convert for comparison
    cohort_f = float(cohort)
    period_f = float(period)
    
    # Compute masks
    nt_mask = unit_gvar.isna()
    if len(nt_values) > 0:
        nt_mask = nt_mask | unit_gvar.isin(nt_values)
    
    # CRITICAL: nyt_only uses strict > and excludes NT
    nyt_only_mask = (unit_gvar > period_f) & ~nt_mask
    treat_mask = (unit_gvar == cohort_f)
    
    return {
        'never_treated': int(nt_mask.sum()),
        'not_yet_treated_only': int(nyt_only_mask.sum()),
        'not_yet_treated_total': int(nt_mask.sum() + nyt_only_mask.sum()),
        'treatment_cohort': int(treat_mask.sum())
    }

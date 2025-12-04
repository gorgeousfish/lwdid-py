"""
Staggered Data Transformation Module

Implements cohort-specific data transformations for staggered DiD estimation
based on Lee and Wooldridge (2023, 2025).

Key concepts:
- Each cohort g defines its own pre-treatment period: pre(g) = {T_min, ..., g-1}
- All units (including never treated and other cohorts) need transformation
- Pre-treatment mean/trend is FIXED for a cohort, regardless of calendar time r
"""

import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def get_cohorts(
    data: pd.DataFrame,
    gvar: str,
    ivar: str,
    never_treated_values: Optional[List] = None,
) -> List[int]:
    """
    Extract all valid treatment cohorts from data.
    
    Excludes never treated units (gvar = NaN, 0, inf, or custom values).
    
    Parameters
    ----------
    data : pd.DataFrame
        Panel data.
    gvar : str
        Cohort variable column name (first treatment period).
    ivar : str
        Unit identifier column name.
    never_treated_values : list, optional
        Values indicating never treated. Default: [0, np.inf].
        NaN is always automatically included.
        
    Returns
    -------
    List[int]
        Sorted list of cohorts, excluding never treated.
        
    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'id': [1, 2, 3, 4],
    ...     'gvar': [2005, 2006, 0, np.nan]
    ... })
    >>> get_cohorts(data, 'gvar', 'id')
    [2005, 2006]
    """
    if never_treated_values is None:
        never_treated_values = [0, np.inf]
    
    # Get unique gvar values (at unit level)
    unit_gvar = data.drop_duplicates(subset=[ivar])[gvar]
    
    # Filter out never treated
    valid_cohorts = []
    for g in unit_gvar.unique():
        if pd.isna(g):
            continue  # NaN is always never treated
        if g in never_treated_values:
            continue
        valid_cohorts.append(int(g))
    
    return sorted(valid_cohorts)


def get_valid_periods_for_cohort(
    cohort: int,
    T_max: int,
) -> List[int]:
    """
    Get valid post-treatment periods for a given cohort.
    
    For cohort g, valid periods are {g, g+1, ..., T_max}.
    
    Parameters
    ----------
    cohort : int
        Treatment cohort (first treatment period).
    T_max : int
        Maximum time period in data.
        
    Returns
    -------
    List[int]
        List of valid post-treatment periods.
        
    Examples
    --------
    >>> get_valid_periods_for_cohort(2005, 2010)
    [2005, 2006, 2007, 2008, 2009, 2010]
    """
    return list(range(int(cohort), int(T_max) + 1))


def _compute_pre_treatment_mean(
    unit_data: pd.DataFrame,
    y: str,
    tvar: str,
    cohort: int,
) -> float:
    """
    Compute pre-treatment mean for a single unit.
    
    Parameters
    ----------
    unit_data : pd.DataFrame
        Single unit's time series data.
    y : str
        Outcome variable column name.
    tvar : str
        Time variable column name.
    cohort : int
        Cohort value (first treatment period).
        
    Returns
    -------
    float
        Mean of Y over pre-treatment periods {t: t < cohort}.
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
) -> Tuple[float, float]:
    """
    Compute pre-treatment linear trend parameters for a single unit.
    
    Uses OLS regression: Y_t = A + B * t + error, for t < cohort.
    
    Parameters
    ----------
    unit_data : pd.DataFrame
        Single unit's time series data.
    y : str
        Outcome variable column name.
    tvar : str
        Time variable column name.
    cohort : int
        Cohort value.
        
    Returns
    -------
    Tuple[float, float]
        (intercept A, slope B).
        Returns (NaN, NaN) if insufficient data.
        
    Raises
    ------
    ValueError
        If fewer than 2 pre-treatment periods available.
    """
    pre_data = unit_data[unit_data[tvar] < cohort].dropna(subset=[y])
    
    if len(pre_data) < 2:
        return (np.nan, np.nan)
    
    t_vals = pre_data[tvar].values.astype(float)
    y_vals = pre_data[y].values.astype(float)
    
    # np.polyfit returns [slope, intercept] for deg=1
    B, A = np.polyfit(t_vals, y_vals, deg=1)
    
    return (A, B)


def transform_staggered_demean(
    data: pd.DataFrame,
    y: str,
    ivar: str,
    tvar: str,
    gvar: str,
    never_treated_values: Optional[Union[List, None]] = None,
) -> pd.DataFrame:
    """
    Apply cohort-specific demeaning transformation for staggered settings.
    
    For each cohort g, computes:
    ydot_{irg} = Y_{ir} - mean(Y_{it} for t < g)
    
    This implements Procedure 4.1 from Lee & Wooldridge (2023) and 
    equation (7.4) from Lee & Wooldridge (2025).
    
    **Key Implementation Points:**
    
    1. Transformation is cohort-specific: each cohort g uses different pre-treatment periods
    2. ALL units (including controls) need transformation for each cohort
    3. For never treated units, compute transformation for each cohort (for overall effects)
    4. **CRITICAL**: Same pre-treatment mean is used for all post periods within a cohort
    
    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel data with columns: y, ivar, tvar, gvar.
    y : str
        Outcome variable column name.
    ivar : str
        Unit identifier column name.
    tvar : str
        Time variable column name (integer).
    gvar : str
        First treatment period column name.
    never_treated_values : list or None, optional
        Values indicating never treated. Default: NaN is always recognized.
        If None, also checks [0, np.inf].
        
    Returns
    -------
    pd.DataFrame
        Original data with added transformation columns:
        - ydot_g{cohort}_r{period}: transformed variable for each (g, r) pair
        
    Raises
    ------
    ValueError
        If no valid cohorts found, required columns missing, or cohort <= T_min.
        
    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'id': [1,1,1,1, 2,2,2,2],
    ...     'year': [1,2,3,4, 1,2,3,4],
    ...     'y': [10,12,14,20, 5,6,7,8],
    ...     'gvar': [3,3,3,3, 0,0,0,0]
    ... })
    >>> result = transform_staggered_demean(data, 'y', 'id', 'year', 'gvar')
    >>> # Cohort g=3: pre={1,2}, post={3,4}
    >>> # Unit 1: pre_mean = (10+12)/2 = 11
    >>> #   ydot_g3_r3 = 14 - 11 = 3
    >>> #   ydot_g3_r4 = 20 - 11 = 9
    """
    # ================================================================
    # Step 1: Data validation and preprocessing
    # ================================================================
    required_cols = [y, ivar, tvar, gvar]
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Create a copy to avoid modifying original data
    result = data.copy()
    
    # Ensure tvar is numeric
    result[tvar] = pd.to_numeric(result[tvar], errors='coerce')
    
    # ================================================================
    # Step 2: Extract cohorts and time range
    # ================================================================
    if never_treated_values is None:
        nt_values = [0, np.inf]
    else:
        nt_values = never_treated_values
    
    cohorts = get_cohorts(result, gvar, ivar, nt_values)
    
    if len(cohorts) == 0:
        raise ValueError("No valid treatment cohorts found in data.")
    
    T_min = int(result[tvar].min())
    T_max = int(result[tvar].max())
    
    # Validate cohorts have pre-treatment periods
    for g in cohorts:
        if g <= T_min:
            raise ValueError(
                f"Cohort {g} has no pre-treatment periods: "
                f"earliest time period is {T_min}. Cohort must be > T_min."
            )
    
    # Check for never treated units
    unit_gvar = result.drop_duplicates(subset=[ivar]).set_index(ivar)[gvar]
    is_nt = unit_gvar.isna() | unit_gvar.isin(nt_values)
    has_nt = is_nt.any()
    
    if not has_nt:
        warnings.warn(
            "No never-treated units found. Overall and cohort effects cannot be estimated. "
            "Only (g,r)-specific effects are available.",
            UserWarning
        )
    
    # ================================================================
    # Step 3: Compute pre-treatment means for each cohort (VECTORIZED)
    # ================================================================
    all_units = result[ivar].unique()
    
    for g in cohorts:
        # Pre-treatment mask: t < g
        pre_mask = result[tvar] < g
        
        # Compute pre-treatment mean for ALL units (vectorized)
        pre_means = result[pre_mask].groupby(ivar)[y].mean()
        
        # Get post-treatment periods for this cohort
        post_periods = get_valid_periods_for_cohort(g, T_max)
        
        # Create transformation columns for each post period
        for r in post_periods:
            col_name = f'ydot_g{int(g)}_r{r}'
            
            # Initialize column with NaN
            result[col_name] = np.nan
            
            # Get mask for period r
            period_mask = result[tvar] == r
            
            # Map pre_means to each row and compute transformation
            # Key: use ivar to map pre_means to corresponding rows
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
    never_treated_values: Optional[Union[List, None]] = None,
) -> pd.DataFrame:
    """
    Apply cohort-specific detrending transformation for staggered settings.
    
    For each cohort g, fits unit-specific linear trend Y_t = A + B*t using
    pre-treatment data, then computes out-of-sample residuals:
    ycheck_{irg} = Y_{ir} - (A_ig + B_ig * r)
    
    This implements Procedure 5.1 from Lee & Wooldridge (2023).
    
    **Prerequisites:**
    - Cohort g must have at least 2 pre-treatment periods (g >= T_min + 2)
    - Otherwise, cannot estimate linear trend
    
    **When to use:**
    - When parallel trends assumption may be violated
    - When treatment and control groups have different linear trends
    - As a robustness check
    
    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel data.
    y : str
        Outcome variable column name.
    ivar : str
        Unit identifier column name.
    tvar : str
        Time variable column name.
    gvar : str
        First treatment period column name.
    never_treated_values : list or None, optional
        Values indicating never treated.
        
    Returns
    -------
    pd.DataFrame
        Original data with added transformation columns:
        - ycheck_g{cohort}_r{period}: detrended variable for each (g, r) pair
        
    Raises
    ------
    ValueError
        If any cohort has fewer than 2 pre-treatment periods.
        
    Examples
    --------
    >>> # Data with clear linear trend
    >>> data = pd.DataFrame({
    ...     'id': [1]*5 + [2]*5,
    ...     'year': [1,2,3,4,5]*2,
    ...     'y': [12,14,16,28,32,   # unit 1: trend=2, effect=10 at t>=4
    ...           5,7,9,11,13],     # unit 2: trend=2, no treatment
    ...     'gvar': [4]*5 + [0]*5
    ... })
    >>> result = transform_staggered_detrend(data, 'y', 'id', 'year', 'gvar')
    >>> # Unit 1: OLS on t=1,2,3 gives A=10, B=2
    >>> # Predicted at t=4: 10 + 2*4 = 18
    >>> # ycheck_g4_r4 = 28 - 18 = 10 (treatment effect!)
    """
    # ================================================================
    # Step 1: Data validation and preprocessing
    # ================================================================
    required_cols = [y, ivar, tvar, gvar]
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    result = data.copy()
    result[tvar] = pd.to_numeric(result[tvar], errors='coerce')
    
    # ================================================================
    # Step 2: Extract cohorts and validate
    # ================================================================
    if never_treated_values is None:
        nt_values = [0, np.inf]
    else:
        nt_values = never_treated_values
    
    cohorts = get_cohorts(result, gvar, ivar, nt_values)
    
    if len(cohorts) == 0:
        raise ValueError("No valid treatment cohorts found in data.")
    
    T_min = int(result[tvar].min())
    T_max = int(result[tvar].max())
    
    # Validate: detrending requires at least 2 pre-treatment periods
    for g in cohorts:
        n_pre_periods = g - T_min
        if n_pre_periods < 2:
            raise ValueError(
                f"Cohort {g} has only {n_pre_periods} pre-treatment period(s). "
                f"Detrending requires at least 2 pre-treatment periods to estimate linear trend. "
                f"Consider using demean instead."
            )
    
    # Check for never treated units
    unit_gvar = result.drop_duplicates(subset=[ivar]).set_index(ivar)[gvar]
    is_nt = unit_gvar.isna() | unit_gvar.isin(nt_values)
    has_nt = is_nt.any()
    
    if not has_nt:
        warnings.warn(
            "No never-treated units found. Overall and cohort effects cannot be estimated. "
            "Only (g,r)-specific effects are available.",
            UserWarning
        )
    
    # ================================================================
    # Step 3: Compute detrending for each cohort
    # ================================================================
    all_units = result[ivar].unique()
    
    for g in cohorts:
        post_periods = get_valid_periods_for_cohort(g, T_max)
        
        # Initialize columns
        for r in post_periods:
            col_name = f'ycheck_g{int(g)}_r{r}'
            result[col_name] = np.nan
        
        # Compute trend coefficients for each unit
        for unit_id in all_units:
            unit_mask = result[ivar] == unit_id
            unit_data = result[unit_mask]
            
            # Get pre-treatment data for this unit
            pre_data = unit_data[unit_data[tvar] < g].dropna(subset=[y])
            
            if len(pre_data) < 2:
                # Not enough data for this unit - set to NaN (don't raise error)
                continue
            
            # Fit linear trend: Y = A + B*t
            t_vals = pre_data[tvar].values.astype(float)
            y_vals = pre_data[y].values.astype(float)
            
            try:
                B, A = np.polyfit(t_vals, y_vals, deg=1)
            except (np.linalg.LinAlgError, ValueError):
                # Fitting failed - set to NaN
                continue
            
            # Compute residuals for post periods
            for r in post_periods:
                col_name = f'ycheck_g{int(g)}_r{r}'
                period_mask = unit_mask & (result[tvar] == r)
                
                if period_mask.any():
                    y_r = result.loc[period_mask, y].values[0]
                    predicted = A + B * r
                    result.loc[period_mask, col_name] = y_r - predicted
    
    return result

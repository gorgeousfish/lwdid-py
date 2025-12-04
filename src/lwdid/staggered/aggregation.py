"""
Staggered Effect Aggregation Module

Implements cohort-specific and overall effect aggregation for staggered DiD
based on Lee and Wooldridge (2023, 2025) Section 7.

Key concepts:
- Cohort effect τ_g: average effect across post-treatment periods for cohort g
- Overall effect τ_ω: weighted average of cohort effects

Reference:
    Lee & Wooldridge (2025) Section 7, Equations (7.9)-(7.19)
    Lee & Wooldridge (2023) Section 4, Theorem 4.1
"""

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import scipy.stats as stats

from .control_groups import identify_never_treated_units
from .estimation import run_ols_regression
from .transformations import get_cohorts, get_valid_periods_for_cohort


# =============================================================================
# Error Message Constants
# =============================================================================

COHORT_NO_NT_ERROR = """
Cohort效应估计需要never treated单位作为控制组，但数据中没有NT单位。

原因：不同cohort的变换变量使用不同的pre-treatment时期，只有NT单位能提供一致的参照基准。

建议：使用aggregate='none'来估计(g,r)特定效应，或手动聚合这些效应。
"""

OVERALL_NO_NT_ERROR = """
整体效应估计需要never treated单位作为控制组，但数据中没有NT单位。

原因：整体效应估计需要never treated单位作为统一参照(论文公式7.18)。

建议：
1. 使用aggregate='cohort'估计cohort-specific效应
2. 使用aggregate='none'估计(g,r)特定效应
3. 手动对(g,r)效应进行加权聚合
"""

INSUFFICIENT_SAMPLE_ERROR = """
样本量不足，无法进行回归估计。

最小要求：
- 至少1个treated单位
- 至少1个控制组单位
- 总样本量 >= 3

当前状态：n_treated={n_treat}, n_control={n_control}, total={n_total}
"""


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CohortEffect:
    """
    Cohort-specific aggregated effect estimate.
    
    Represents τ_g, the average ATT for cohort g across all post-treatment periods.
    Implements equation (7.10) from Lee & Wooldridge (2025).
    
    Note: Control group must be never treated only.
    
    Attributes
    ----------
    cohort : int
        Treatment cohort (first treatment period g)
    att : float
        Estimated ATT τ̂_g
    se : float
        Standard error
    ci_lower : float
        Confidence interval lower bound
    ci_upper : float
        Confidence interval upper bound
    t_stat : float
        t-statistic = att / se
    pvalue : float
        Two-sided p-value
    n_periods : int
        Number of post-treatment periods (T-g+1)
    n_units : int
        Number of treated units (N_g)
    n_control : int
        Number of control units (N_∞, never treated only)
    """
    cohort: int
    att: float
    se: float
    ci_lower: float
    ci_upper: float
    t_stat: float
    pvalue: float
    n_periods: int
    n_units: int
    n_control: int


@dataclass
class OverallEffect:
    """
    Overall weighted effect estimate.
    
    Represents τ_ω, the weighted average of cohort effects.
    Implements equation (7.19) from Lee & Wooldridge (2025).
    
    Note: Control group must be never treated only.
    
    Attributes
    ----------
    att : float
        Estimated ATT τ̂_ω
    se : float
        Standard error
    ci_lower : float
        Confidence interval lower bound
    ci_upper : float
        Confidence interval upper bound
    t_stat : float
        t-statistic = att / se
    pvalue : float
        Two-sided p-value
    cohort_weights : Dict[int, float]
        {cohort: weight}, where weight ω_g = N_g/N_treat
    n_treated : int
        Total treated units (N_treat = N_S + ... + N_T)
    n_control : int
        Never treated units (N_∞)
    """
    att: float
    se: float
    ci_lower: float
    ci_upper: float
    t_stat: float
    pvalue: float
    cohort_weights: Dict[int, float]
    n_treated: int
    n_control: int


# =============================================================================
# Helper Functions
# =============================================================================

def _compute_cohort_aggregated_variable(
    data: pd.DataFrame,
    ivar: str,
    ydot_cols: List[str],
) -> pd.Series:
    """
    Compute cohort aggregated variable for all units (vectorized).
    
    For each unit i, computes:
    Y_bar_ig = (1/(T-g+1)) * Σ_{r=g}^T ydot_g{g}_r{r}
    
    Panel data structure (N units × T periods):
    
    id  year  y    gvar  ydot_g4_r4  ydot_g4_r5  ydot_g4_r6
    1   1     10   4     NaN         NaN         NaN
    1   2     12   4     NaN         NaN         NaN
    1   3     14   4     NaN         NaN         NaN
    1   4     20   4     8.0         NaN         NaN    <- only at year=4
    1   5     22   4     NaN         10.0        NaN    <- only at year=5
    1   6     24   4     NaN         NaN         12.0   <- only at year=6
    2   1     5    0     NaN         NaN         NaN
    2   2     6    0     NaN         NaN         NaN
    ...
    
    Aggregation logic:
    - For unit 1: Y_bar_i1_g4 = mean(8.0, 10.0, 12.0) = 10.0
    - Need to collect values from year=4,5,6 rows
    
    Parameters
    ----------
    data : DataFrame
        Transformed panel data
    ivar : str
        Unit identifier column name
    ydot_cols : list
        List of transformation column names to aggregate,
        e.g., ['ydot_g4_r4', 'ydot_g4_r5', 'ydot_g4_r6']
        
    Returns
    -------
    Series
        Index = unit IDs, values = aggregated variable Y_bar_ig
    """
    if not ydot_cols:
        raise ValueError("No transformation columns provided for aggregation")
    
    # Check columns exist
    missing_cols = [c for c in ydot_cols if c not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing transformation columns: {missing_cols}")
    
    # Vectorized approach:
    # Each ydot column only has non-NaN values at its corresponding period
    # Sum across columns (only one will be non-NaN per row), then average by unit
    
    ydot_values = data[ydot_cols].values  # shape: (N*T, n_cols)
    
    # Each row only has one non-NaN value, use nansum to get it
    row_values = np.nansum(ydot_values, axis=1)  # shape: (N*T,)
    
    # Mark valid rows (at least one non-NaN value)
    valid_mask = np.any(~np.isnan(ydot_values), axis=1)
    
    # Create temp DataFrame for grouping
    temp_df = pd.DataFrame({
        ivar: data[ivar].values,
        'ydot_value': np.where(valid_mask, row_values, np.nan)
    })
    
    # Group by unit and compute mean
    Y_bar = temp_df.groupby(ivar)['ydot_value'].mean()
    
    # Check for missing units
    n_valid = Y_bar.notna().sum()
    n_total = len(Y_bar)
    if n_valid < n_total:
        warnings.warn(
            f"聚合变量计算: {n_total - n_valid}个单位缺失数据将被排除",
            UserWarning
        )
    
    return Y_bar


def _get_unit_level_gvar(
    data: pd.DataFrame,
    gvar: str,
    ivar: str,
) -> pd.Series:
    """Get unit-level gvar values (one value per unit)."""
    return data.groupby(ivar)[gvar].first()


def _identify_nt_mask(
    unit_gvar: pd.Series,
    never_treated_values: Optional[List] = None,
) -> pd.Series:
    """
    Identify never treated units from unit-level gvar.
    
    Parameters
    ----------
    unit_gvar : Series
        Unit-level gvar values (index = unit IDs)
    never_treated_values : list, optional
        Values indicating never treated. Default: [0, np.inf]
        NaN is always automatically included.
        
    Returns
    -------
    Series
        Boolean mask, True = never treated
    """
    if never_treated_values is None:
        nt_values = [0, np.inf]
    else:
        nt_values = never_treated_values
    
    # Start with NaN mask
    nt_mask = unit_gvar.isna()
    
    # Add specified values
    for val in nt_values:
        if pd.isna(val):
            nt_mask |= unit_gvar.isna()
        elif np.isinf(val):
            nt_mask |= np.isinf(unit_gvar.astype(float))
        else:
            nt_mask |= (unit_gvar == val)
    
    return nt_mask


def _add_cluster_to_reg_data(
    reg_data: pd.DataFrame,
    original_data: pd.DataFrame,
    ivar: str,
    cluster_var: Optional[str],
) -> pd.DataFrame:
    """
    Add cluster variable to unit-level regression data.
    
    For aggregation regressions, data is at unit level (one row per unit).
    Need to extract cluster values from original panel data.
    
    Parameters
    ----------
    reg_data : DataFrame
        Unit-level regression data (index = unit IDs)
    original_data : DataFrame
        Original panel data with cluster variable
    ivar : str
        Unit identifier column name
    cluster_var : str or None
        Cluster variable column name
        
    Returns
    -------
    DataFrame
        reg_data with '_cluster' column added
    """
    if cluster_var is None:
        return reg_data
    
    if cluster_var not in original_data.columns:
        raise ValueError(f"Cluster variable '{cluster_var}' not found in data")
    
    # Extract unit-level cluster values
    unit_cluster = original_data.groupby(ivar)[cluster_var].first()
    
    # Map to regression data
    reg_data = reg_data.copy()
    reg_data['_cluster'] = reg_data.index.map(unit_cluster)
    
    # Check cluster count
    n_clusters = reg_data['_cluster'].nunique()
    if n_clusters < 20:
        warnings.warn(
            f"聚类数({n_clusters})较少，聚类标准误可能不可靠。"
            f"建议至少20个聚类。",
            UserWarning
        )
    
    return reg_data


# =============================================================================
# Main Aggregation Functions
# =============================================================================

def aggregate_to_cohort(
    data_transformed: pd.DataFrame,
    gvar: str,
    ivar: str,
    tvar: str,
    cohorts: List[int],
    T_max: int,
    never_treated_values: Optional[List] = None,
    transform_type: str = 'demean',
    vce: Optional[str] = None,
    cluster_var: Optional[str] = None,
    alpha: float = 0.05,
) -> List[CohortEffect]:
    """
    Aggregate (g,r) effects to cohort-specific effects.
    
    Implements equations (7.9)-(7.10) from Lee & Wooldridge (2025).
    
    For each cohort g:
    1. Compute aggregated variable: Y_bar_ig = mean(ydot_g{g}_r{r} for r in [g, T])
    2. Restrict sample: D_ig + D_i∞ = 1 (cohort g + NT only)
    3. Run regression: Y_bar_ig on 1, D_ig
    4. ATT coefficient = τ_g
    
    Parameters
    ----------
    data_transformed : DataFrame
        Panel data with transformation columns from transform_staggered_demean/detrend.
    gvar : str
        Cohort variable column name
    ivar : str
        Unit identifier column name
    tvar : str
        Time variable column name
    cohorts : List[int]
        List of cohorts to aggregate
    T_max : int
        Maximum time period
    never_treated_values : list, optional
        Values indicating never treated. Default: [0, np.inf].
        Castle Law data: use [0] after preprocessing effyear.
    transform_type : str
        'demean': search for ydot_g{g}_r{r} columns (default)
        'detrend': search for ycheck_g{g}_r{r} columns
    vce : str, optional
        Standard error type: None (homoskedastic), 'hc3', 'cluster'
    cluster_var : str, optional
        Cluster variable (required if vce='cluster')
    alpha : float
        Significance level, default 0.05 for 95% CI
        
    Returns
    -------
    List[CohortEffect]
        Aggregated effect estimates for each cohort
        
    Raises
    ------
    ValueError
        If no never treated units in data
        
    Notes
    -----
    **Must use never_treated control group**
    
    Different cohorts use different pre-treatment periods for transformation,
    so only NT units provide consistent baseline reference.
    """
    # ================================================================
    # Step 1: Get unit-level gvar and identify NT units
    # ================================================================
    unit_gvar = _get_unit_level_gvar(data_transformed, gvar, ivar)
    nt_mask = _identify_nt_mask(unit_gvar, never_treated_values)
    
    # Check for NT units
    if nt_mask.sum() == 0:
        raise ValueError(COHORT_NO_NT_ERROR)
    
    # Determine column prefix based on transform_type
    prefix = 'ydot' if transform_type == 'demean' else 'ycheck'
    
    results = []
    
    # ================================================================
    # Step 2: Process each cohort
    # ================================================================
    for g in cohorts:
        # Get post-treatment periods
        post_periods = get_valid_periods_for_cohort(int(g), int(T_max))
        
        # Get transformation columns
        ydot_cols = [f'{prefix}_g{int(g)}_r{r}' for r in post_periods 
                     if f'{prefix}_g{int(g)}_r{r}' in data_transformed.columns]
        
        if not ydot_cols:
            warnings.warn(f"Cohort {g}: 无有效变换列，跳过")
            continue
        
        # --------------------------------------------------------
        # Step 2a: Compute aggregated variable Y_bar_ig
        # --------------------------------------------------------
        try:
            Y_bar_ig = _compute_cohort_aggregated_variable(
                data_transformed, ivar, ydot_cols
            )
        except Exception as e:
            warnings.warn(f"Cohort {g}: 聚合变量计算失败 - {e}")
            continue
        
        # --------------------------------------------------------
        # Step 2b: Build regression sample (cohort g + NT only)
        # --------------------------------------------------------
        cohort_mask = (unit_gvar == g)
        sample_mask = cohort_mask | nt_mask
        
        # Get sample unit IDs
        sample_units = sample_mask[sample_mask].index
        
        # Build regression DataFrame
        reg_data = pd.DataFrame({
            'Y_bar_ig': Y_bar_ig.reindex(sample_units),
            'D_ig': cohort_mask.reindex(sample_units).astype(int),
        }, index=sample_units)
        
        # Drop missing Y_bar_ig
        reg_data = reg_data.dropna(subset=['Y_bar_ig'])
        
        n_treat = int(cohort_mask.sum())
        n_control = int(nt_mask.sum())
        n_total = len(reg_data)
        
        # Check sample size: minimum 2 units (1 treated + 1 control)
        if n_total < 2 or n_treat < 1 or n_control < 1:
            warnings.warn(
                f"Cohort {g}: 样本量不足 (total={n_total}, treat={n_treat}, control={n_control})"
            )
            continue
        
        # Warn for very small samples
        if n_total == 2:
            warnings.warn(
                f"Cohort {g}: 样本量仅有2个单位，标准误可能不可靠",
                UserWarning
            )
        
        # Add cluster variable if needed
        if vce == 'cluster':
            reg_data = _add_cluster_to_reg_data(
                reg_data, data_transformed, ivar, cluster_var
            )
            cluster_col = '_cluster'
        else:
            cluster_col = None
        
        # --------------------------------------------------------
        # Step 2c: Run regression
        # --------------------------------------------------------
        try:
            result = run_ols_regression(
                data=reg_data.reset_index(),
                y='Y_bar_ig',
                d='D_ig',
                vce=vce,
                cluster_var=cluster_col,
                alpha=alpha,
            )
        except Exception as e:
            warnings.warn(f"Cohort {g}: 回归估计失败 - {e}")
            continue
        
        # --------------------------------------------------------
        # Step 2d: Store result
        # --------------------------------------------------------
        results.append(CohortEffect(
            cohort=int(g),
            att=result['att'],
            se=result['se'],
            ci_lower=result['ci_lower'],
            ci_upper=result['ci_upper'],
            t_stat=result['t_stat'],
            pvalue=result['pvalue'],
            n_periods=len(ydot_cols),
            n_units=n_treat,
            n_control=n_control,
        ))
    
    # Sort by cohort
    results.sort(key=lambda x: x.cohort)
    
    return results


def construct_aggregated_outcome(
    data: pd.DataFrame,
    gvar: str,
    ivar: str,
    tvar: str,
    weights: Dict[int, float],
    cohorts: List[int],
    T_max: int,
    transform_type: str = 'demean',
    never_treated_values: Optional[List] = None,
) -> pd.Series:
    """
    Construct aggregated outcome Y_bar_i according to equation (7.18).
    
    CRITICAL: Treated and NT units are handled differently!
    
    For treated units (D_ig=1):
        Y_bar_i = Y_bar_ig = mean(ydot_g{g}_r{r} for r in [g, T])
        Use own cohort g's aggregated transformation
        
    For never treated units (D_i∞=1):
        Y_bar_i = Σ_g ω_g * Y_bar_ig
        Weighted average of ALL cohorts' transformations
        
    This construction ensures regression (7.19) coefficient = τ_ω.
    
    Parameters
    ----------
    data : DataFrame
        Transformed panel data with ydot_g{g}_r{r} or ycheck_g{g}_r{r} columns
    gvar : str
        Cohort variable column name
    ivar : str
        Unit identifier column name
    tvar : str
        Time variable column name
    weights : Dict[int, float]
        Cohort weights {g: omega_g}, must sum to 1
    cohorts : List[int]
        Cohort list [S, S+1, ..., T]
    T_max : int
        Maximum time period
    transform_type : str
        'demean': search for ydot columns (default)
        'detrend': search for ycheck columns
    never_treated_values : list, optional
        Values indicating never treated. Default: [0, np.inf]
        
    Returns
    -------
    Series
        Index = unit IDs, values = Y_bar_i
        
    Notes
    -----
    Common implementation error:
    - WRONG: Use single cohort transformation for NT units
    - CORRECT: NT units need weighted average of ALL cohorts' transformations
    """
    # Get unit-level gvar
    unit_gvar = _get_unit_level_gvar(data, gvar, ivar)
    nt_mask = _identify_nt_mask(unit_gvar, never_treated_values)
    
    # Determine column prefix
    prefix = 'ydot' if transform_type == 'demean' else 'ycheck'
    
    # Initialize output
    Y_bar = pd.Series(index=unit_gvar.index, dtype=float)
    Y_bar[:] = np.nan
    
    # Pre-compute aggregated variables for each cohort (for all units)
    cohort_Y_bar = {}
    for g in cohorts:
        post_periods = get_valid_periods_for_cohort(int(g), int(T_max))
        ydot_cols = [f'{prefix}_g{int(g)}_r{r}' for r in post_periods 
                     if f'{prefix}_g{int(g)}_r{r}' in data.columns]
        if ydot_cols:
            try:
                cohort_Y_bar[g] = _compute_cohort_aggregated_variable(data, ivar, ydot_cols)
            except Exception:
                continue
    
    # Process each unit
    for unit_id in unit_gvar.index:
        g_unit = unit_gvar[unit_id]
        
        if nt_mask[unit_id]:
            # Never treated: weighted average of all cohorts' transformations
            weighted_sum = 0.0
            valid_weights = 0.0
            
            for g in cohorts:
                if g not in cohort_Y_bar:
                    continue
                if unit_id not in cohort_Y_bar[g].index:
                    continue
                    
                Y_bar_ig = cohort_Y_bar[g][unit_id]
                if not np.isnan(Y_bar_ig):
                    weighted_sum += weights[g] * Y_bar_ig
                    valid_weights += weights[g]
            
            if valid_weights > 0:
                # Normalize if some cohorts missing
                Y_bar[unit_id] = weighted_sum / valid_weights * sum(weights.values())
            
        elif int(g_unit) in cohorts:
            # Treated: use own cohort's aggregated transformation
            g = int(g_unit)
            if g in cohort_Y_bar and unit_id in cohort_Y_bar[g].index:
                Y_bar[unit_id] = cohort_Y_bar[g][unit_id]
    
    return Y_bar


def aggregate_to_overall(
    data_transformed: pd.DataFrame,
    gvar: str,
    ivar: str,
    tvar: str,
    never_treated_values: Optional[List] = None,
    transform_type: str = 'demean',
    vce: Optional[str] = None,
    cluster_var: Optional[str] = None,
    alpha: float = 0.05,
) -> OverallEffect:
    """
    Estimate overall weighted effect τ_ω.
    
    Implements equations (7.18)-(7.19) from Lee & Wooldridge (2025).
    
    Algorithm:
    1. Compute cohort weights: ω_g = N_g / N_treat
    2. Construct aggregated outcome Y_bar_i (equation 7.18):
       - Treated: Y_bar_i = Y_bar_ig (own cohort)
       - NT: Y_bar_i = Σ_g ω_g Y_bar_ig (weighted average)
    3. Construct ever-treated indicator: D_i = D_{iS} + ... + D_{iT}
    4. Run regression: Y_bar_i on 1, D_i
    5. D_i coefficient = τ_ω
    
    Parameters
    ----------
    data_transformed : DataFrame
        Panel data with transformation columns
    gvar : str
        Cohort variable column name
    ivar : str
        Unit identifier column name
    tvar : str
        Time variable column name (for T_max detection)
    never_treated_values : list, optional
        Values indicating never treated. Default: [0, np.inf].
        Castle Law data: use [0] after preprocessing effyear.
    transform_type : str
        'demean': search for ydot columns (default)
        'detrend': search for ycheck columns
    vce : str, optional
        Standard error type: None (homoskedastic), 'hc3', 'cluster'
    cluster_var : str, optional
        Cluster variable (required if vce='cluster')
    alpha : float
        Significance level, default 0.05 for 95% CI
        
    Returns
    -------
    OverallEffect
        Overall weighted effect estimate
        
    Raises
    ------
    ValueError
        If no never treated units in data, or insufficient sample
        
    Notes
    -----
    **Control group must be Never Treated Only**
    
    Sample restriction: D_i + D_{i∞} = 1 (ever treated + never treated)
    
    Why not-yet-treated cannot be used:
    - NT units need transformations for all cohorts (weighted average)
    - NYT units have own cohort, cannot serve as control
    - This is a theoretical requirement, not optional
    
    Standard errors:
    - Obtained directly from regression (7.19)
    - Automatically handles cohort covariance
    """
    # ================================================================
    # Step 1: Get unit-level gvar and validate NT units exist
    # ================================================================
    unit_gvar = _get_unit_level_gvar(data_transformed, gvar, ivar)
    nt_mask = _identify_nt_mask(unit_gvar, never_treated_values)
    
    if nt_mask.sum() == 0:
        raise ValueError(OVERALL_NO_NT_ERROR)
    
    # ================================================================
    # Step 2: Get cohorts and compute weights
    # ================================================================
    cohorts = get_cohorts(data_transformed, gvar, ivar, never_treated_values)
    
    if len(cohorts) == 0:
        raise ValueError("No valid treatment cohorts found in data")
    
    # Count units per cohort
    cohort_sizes = {g: int((unit_gvar == g).sum()) for g in cohorts}
    N_treat = sum(cohort_sizes.values())
    
    if N_treat == 0:
        raise ValueError("No treated units found")
    
    # Compute weights: ω_g = N_g / N_treat
    weights = {g: n / N_treat for g, n in cohort_sizes.items()}
    
    T_max = int(data_transformed[tvar].max())
    
    # ================================================================
    # Step 3: Construct aggregated outcome (equation 7.18)
    # ================================================================
    Y_bar = construct_aggregated_outcome(
        data_transformed, gvar, ivar, tvar,
        weights, cohorts, T_max,
        transform_type, never_treated_values
    )
    
    # ================================================================
    # Step 4: Construct ever-treated indicator
    # ================================================================
    # D_i = 1 if ever treated, 0 if never treated
    D_ever = (~nt_mask).astype(int)
    
    # ================================================================
    # Step 5: Build regression data
    # ================================================================
    reg_data = pd.DataFrame({
        'Y_bar': Y_bar,
        'D_ever': D_ever,
    }, index=unit_gvar.index)
    
    # Drop missing Y_bar
    reg_data = reg_data.dropna(subset=['Y_bar'])
    
    n_treated = int(D_ever.sum())
    n_control = int(nt_mask.sum())
    n_total = len(reg_data)
    
    if n_total < 2 or n_treated < 1 or n_control < 1:
        raise ValueError(
            INSUFFICIENT_SAMPLE_ERROR.format(
                n_treat=n_treated, n_control=n_control, n_total=n_total
            )
        )
    
    if n_total == 2:
        warnings.warn(
            "样本量仅有2个单位，标准误可能不可靠",
            UserWarning
        )
    
    # Add cluster variable if needed
    if vce == 'cluster':
        reg_data = _add_cluster_to_reg_data(
            reg_data, data_transformed, ivar, cluster_var
        )
        cluster_col = '_cluster'
    else:
        cluster_col = None
    
    # ================================================================
    # Step 6: Run regression (equation 7.19)
    # ================================================================
    result = run_ols_regression(
        data=reg_data.reset_index(),
        y='Y_bar',
        d='D_ever',
        vce=vce,
        cluster_var=cluster_col,
        alpha=alpha,
    )
    
    return OverallEffect(
        att=result['att'],
        se=result['se'],
        ci_lower=result['ci_lower'],
        ci_upper=result['ci_upper'],
        t_stat=result['t_stat'],
        pvalue=result['pvalue'],
        cohort_weights=weights,
        n_treated=n_treated,
        n_control=n_control,
    )


# =============================================================================
# Utility Functions
# =============================================================================

def cohort_effects_to_dataframe(results: List[CohortEffect]) -> pd.DataFrame:
    """
    Convert list of CohortEffect to DataFrame.
    
    Parameters
    ----------
    results : List[CohortEffect]
        Cohort effect estimates from aggregate_to_cohort
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: cohort, att, se, ci_lower, ci_upper,
        t_stat, pvalue, n_periods, n_units, n_control
    """
    if len(results) == 0:
        return pd.DataFrame(columns=[
            'cohort', 'att', 'se', 'ci_lower', 'ci_upper',
            't_stat', 'pvalue', 'n_periods', 'n_units', 'n_control'
        ])
    
    return pd.DataFrame([
        {
            'cohort': r.cohort,
            'att': r.att,
            'se': r.se,
            'ci_lower': r.ci_lower,
            'ci_upper': r.ci_upper,
            't_stat': r.t_stat,
            'pvalue': r.pvalue,
            'n_periods': r.n_periods,
            'n_units': r.n_units,
            'n_control': r.n_control,
        }
        for r in results
    ])

"""
Aggregation of cohort-time ATT estimates to summary parameters.

This module aggregates (g, r)-specific treatment effect estimates into
cohort-level and overall summary parameters via cross-sectional OLS regression.

Aggregation Levels
------------------
Cohort Effect (τ_g)
    Time-averaged ATT for cohort g across post-treatment periods R_g:

        Ȳ_ig = α + τ_g · D_ig + ε_ig

    where Ȳ_ig = (1/|R_g|) Σ_{r∈R_g} ẏ_{irg} averages the transformed outcome
    over periods R_g = {g, g+1, ..., T}, and D_ig indicates cohort membership.

Overall Effect (τ_ω)
    Cohort-size-weighted average treatment effect:

        Ȳ_i = α + τ_ω · D_i + ε_i

    where D_i indicates ever-treated status. The aggregated outcome Ȳ_i is
    constructed differently by treatment status:

    - Treated units: Ȳ_i = Ȳ_ig (own cohort's time-averaged transformation)
    - Never-treated units: Ȳ_i = Σ_g ω_g · Ȳ_ig (cohort-weighted mixture)

    with cohort weights ω_g = N_g / Σ_h N_h proportional to cohort sizes.

Control Group Requirement
-------------------------
Both aggregation levels require never-treated units as controls. Each cohort g
uses a different pre-treatment reference period {T_min, ..., g-1} for the
outcome transformation, so only never-treated units provide a consistent
counterfactual baseline across cohorts. When no never-treated units exist,
use aggregate='none' to estimate (g, r)-specific effects with not-yet-treated
controls instead.
"""

import warnings
from dataclasses import dataclass
from math import fsum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .estimation import run_ols_regression
from .transformations import get_cohorts, get_valid_periods_for_cohort
from ..exceptions import NoNeverTreatedError
from ..validation import is_never_treated, COHORT_FLOAT_TOLERANCE, get_cohort_mask

# Tolerance for weight sum validation (more permissive than COHORT_FLOAT_TOLERANCE).
# When summing multiple floating-point weights, cumulative rounding errors may
# exceed the strict 1e-9 tolerance used for cohort comparisons. A tolerance of
# 1e-6 accommodates typical floating-point accumulation errors while still
# detecting meaningful weight calculation errors.
WEIGHT_SUM_TOLERANCE = 1e-6


_COHORT_NO_NT_ERROR = (
    "Cohort effect estimation requires never-treated control units, but none found. "
    "Each cohort uses a different pre-treatment reference period, so only "
    "never-treated units provide a consistent baseline across cohorts. "
    "Use aggregate='none' to estimate (g, r)-specific effects with not-yet-treated controls."
)

_OVERALL_NO_NT_ERROR = (
    "Overall effect estimation requires never-treated control units, but none found. "
    "Use aggregate='cohort' for cohort-specific effects or aggregate='none' for "
    "(g, r)-specific effects with not-yet-treated controls."
)

_INSUFFICIENT_SAMPLE_ERROR = (
    "Insufficient sample size for regression: "
    "total={n_total}, treated={n_treat}, control={n_control}. "
    "Requires at least 1 treated and 1 control unit."
)


@dataclass
class CohortEffect:
    """
    Cohort-specific time-averaged treatment effect estimate.

    Stores τ̂_g, the ATT for cohort g averaged over post-treatment periods
    r ∈ {g, g+1, ..., T}, estimated via cross-sectional OLS regression on
    time-averaged transformed outcomes with never-treated controls.

    Attributes
    ----------
    cohort : int
        Treatment cohort identifier g (first treatment period).
    att : float
        Point estimate τ̂_g.
    se : float
        Standard error of τ̂_g.
    ci_lower : float
        Lower bound of the confidence interval.
    ci_upper : float
        Upper bound of the confidence interval.
    t_stat : float
        t-statistic for H₀: τ_g = 0.
    pvalue : float
        Two-sided p-value.
    n_periods : int
        Number of post-treatment periods in the average.
    n_units : int
        Number of treated units in cohort g.
    n_control : int
        Number of never-treated control units.
    df_resid : int
        Residual degrees of freedom from the aggregation regression.
        Equals n - 2 for the simple regression Ȳ_ig = α + τ_g·D_ig + ε_ig.
    df_inference : int
        Degrees of freedom used for inference. Equals df_resid for
        homoskedastic/HC variance, or G-1 for cluster-robust variance.
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
    df_resid: int = 0
    df_inference: int = 0


@dataclass
class OverallEffect:
    """
    Overall cohort-size-weighted treatment effect estimate.

    Stores τ̂_ω, the weighted average ATT across all treated cohorts with
    weights ω_g = N_g / Σ_h N_h proportional to cohort sizes, estimated via
    cross-sectional OLS regression on aggregated transformed outcomes with
    never-treated controls.

    Attributes
    ----------
    att : float
        Point estimate τ̂_ω.
    se : float
        Standard error of τ̂_ω.
    ci_lower : float
        Lower bound of the confidence interval.
    ci_upper : float
        Upper bound of the confidence interval.
    t_stat : float
        t-statistic for H₀: τ_ω = 0.
    pvalue : float
        Two-sided p-value.
    cohort_weights : Dict[int, float]
        Cohort weight mapping {g: ω_g} where Σ_g ω_g = 1.
    n_treated : int
        Total number of ever-treated units.
    n_control : int
        Number of never-treated control units.
    df_resid : int
        Residual degrees of freedom from the overall aggregation regression.
        Equals n - 2 for the simple regression Ȳ_i = α + τ_ω·D_i + ε_i.
    df_inference : int
        Degrees of freedom used for inference. Equals df_resid for
        homoskedastic/HC variance, or G-1 for cluster-robust variance.
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
    df_resid: int = 0
    df_inference: int = 0


def _compute_cohort_aggregated_variable(
    data: pd.DataFrame,
    ivar: str,
    ydot_cols: List[str],
) -> pd.Series:
    """
    Compute unit-level time-averaged transformed outcome.

    Averages period-specific transformed outcomes into a single unit-level
    variable: Ȳ_ig = (1/|R_g|) Σ_{r∈R_g} ẏ_{irg}.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data containing transformation columns.
    ivar : str
        Unit identifier column name.
    ydot_cols : list of str
        Period-specific transformation column names to average
        (e.g., 'ydot_g2005_r2005', 'ydot_g2005_r2006').

    Returns
    -------
    pd.Series
        Unit-level aggregated outcome Ȳ_ig indexed by unit ID; NaN for units
        with no valid observations.

    Raises
    ------
    ValueError
        If ydot_cols is empty or contains missing columns.
    """
    if not ydot_cols:
        raise ValueError("No transformation columns provided for aggregation")

    missing_cols = [c for c in ydot_cols if c not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing transformation columns: {missing_cols}")

    # Use pd.isna() for compatibility with pandas nullable types (pd.NA)
    # np.isnan() may raise TypeError for pd.NA values in object arrays
    ydot_df = data[ydot_cols]
    non_nan_count = (~ydot_df.isna()).sum(axis=1).values
    multi_value_rows = non_nan_count > 1

    if np.any(multi_value_rows):
        n_problematic = int(np.sum(multi_value_rows))
        # stacklevel=3: _compute_cohort_aggregated_variable (2) -> caller function (3)
        warnings.warn(
            f"{n_problematic} rows have multiple non-NaN transformed values. "
            f"Using mean for aggregation.",
            UserWarning,
            stacklevel=3
        )

    # Suppress known empty slice warning from np.nanmean when all values are NaN
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Mean of empty slice')
        row_values = np.nanmean(ydot_df.values, axis=1)
    valid_mask = non_nan_count > 0
    
    temp_df = pd.DataFrame({
        ivar: data[ivar].values,
        'ydot_value': np.where(valid_mask, row_values, np.nan)
    })
    
    Y_bar = temp_df.groupby(ivar)['ydot_value'].mean()
    
    n_valid = Y_bar.notna().sum()
    n_total = len(Y_bar)
    if n_valid < n_total:
        # stacklevel=3: _compute_cohort_aggregated_variable (2) -> caller function (3)
        warnings.warn(
            f"Aggregation: {n_total - n_valid} units excluded due to missing data.",
            UserWarning,
            stacklevel=3
        )
    
    return Y_bar


def _get_unit_level_gvar(
    data: pd.DataFrame,
    gvar: str,
    ivar: str,
) -> pd.Series:
    """
    Extract unit-level cohort assignment from panel data.

    The cohort variable (first treatment period) is time-invariant within
    each unit; this extracts one observation per unit.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format.
    gvar : str
        Cohort variable column name.
    ivar : str
        Unit identifier column name.

    Returns
    -------
    pd.Series
        Unit-level gvar values indexed by unit ID.
    """
    return data.groupby(ivar)[gvar].first()


def _identify_nt_mask(unit_gvar: pd.Series) -> pd.Series:
    """
    Create boolean mask identifying never-treated units.

    A unit is never-treated if its gvar is 0, NaN, or +∞. Delegates to the
    canonical `is_never_treated()` function for consistent identification.

    Parameters
    ----------
    unit_gvar : pd.Series
        Unit-level gvar values indexed by unit ID.

    Returns
    -------
    pd.Series
        Boolean mask indexed by unit ID; True indicates never-treated.
    """
    return unit_gvar.apply(is_never_treated)


def _add_cluster_to_reg_data(
    reg_data: pd.DataFrame,
    original_data: pd.DataFrame,
    ivar: str,
    cluster_var: Optional[str],
) -> pd.DataFrame:
    """
    Merge cluster identifiers into unit-level regression data.

    Extracts cluster assignments from the original panel and joins them to
    the regression data. Warns for missing cluster values or few clusters.

    Parameters
    ----------
    reg_data : pd.DataFrame
        Unit-level regression data with unit IDs as index.
    original_data : pd.DataFrame
        Panel data containing the cluster variable.
    ivar : str
        Unit identifier column name.
    cluster_var : str or None
        Cluster variable column name; if None, returns reg_data unchanged.

    Returns
    -------
    pd.DataFrame
        Copy of reg_data with '_cluster' column added if cluster_var specified.

    Raises
    ------
    ValueError
        If cluster_var not found in original_data.
    """
    if cluster_var is None:
        return reg_data

    if cluster_var not in original_data.columns:
        raise ValueError(f"Cluster variable '{cluster_var}' not found in data")

    unit_cluster = original_data.groupby(ivar)[cluster_var].first()

    reg_data = reg_data.copy()
    reg_data['_cluster'] = reg_data.index.map(unit_cluster)

    # Distinguish between two sources of NaN cluster values:
    # 1. Original NaN: unit_cluster value was already NaN (from .first())
    # 2. Mapping NaN: unit ID in reg_data.index not found in unit_cluster.index
    n_missing_cluster = reg_data['_cluster'].isna().sum()
    if n_missing_cluster > 0:
        # Count original NaN values from the cluster extraction
        n_original_na = unit_cluster.isna().sum()
        # Count units not in the mapping (mapping produced NaN)
        units_in_reg = set(reg_data.index)
        units_in_cluster = set(unit_cluster.index)
        n_unmapped = len(units_in_reg - units_in_cluster)
        
        # stacklevel=3: _add_cluster_to_reg_data (2) -> caller function (3)
        warning_parts = [f"{n_missing_cluster} units have missing cluster values."]
        if n_original_na > 0:
            warning_parts.append(f"  - {n_original_na} unit(s) have NaN in original cluster column")
        if n_unmapped > 0:
            warning_parts.append(f"  - {n_unmapped} unit(s) in regression data not found in original data")
        warning_parts.append("These may cause errors in clustered SE calculation.")
        
        warnings.warn(
            "\n".join(warning_parts),
            UserWarning,
            stacklevel=3
        )

    n_clusters = reg_data['_cluster'].nunique()
    if n_clusters < 20:
        # stacklevel=3: _add_cluster_to_reg_data (2) -> caller function (3)
        warnings.warn(
            f"Number of clusters ({n_clusters}) is small, clustered standard errors may be unreliable. "
            f"At least 20 clusters are recommended.",
            UserWarning,
            stacklevel=3
        )

    return reg_data


def aggregate_to_cohort(
    data_transformed: pd.DataFrame,
    gvar: str,
    ivar: str,
    tvar: str,
    cohorts: List[int],
    T_max: int,
    transform_type: str = 'demean',
    vce: Optional[str] = None,
    cluster_var: Optional[str] = None,
    alpha: float = 0.05,
    controls: Optional[List[str]] = None,
    never_treated_values: Optional[List] = None,
) -> List[CohortEffect]:
    """
    Aggregate period-specific effects to cohort-level effects.

    For each cohort g, estimates τ_g via cross-sectional OLS:

        Ȳ_ig = α + τ_g · D_ig + X_i'γ + D_ig·(X_i - X̄₁)'δ + ε_ig

    where Ȳ_ig = (1/|R_g|) Σ_{r∈R_g} ẏ_{irg} is the time-averaged transformed
    outcome, D_ig indicates cohort g membership, and X_i are time-invariant
    controls centered at treated mean.

    Parameters
    ----------
    data_transformed : pd.DataFrame
        Panel data with transformation columns named 'ydot_g{g}_r{r}' (demean)
        or 'ycheck_g{g}_r{r}' (detrend).
    gvar : str
        Cohort variable column name.
    ivar : str
        Unit identifier column name.
    tvar : str
        Time variable column name.
    cohorts : list of int
        Treatment cohorts to estimate.
    T_max : int
        Maximum time period (determines R_g = {g, ..., T_max}).
    transform_type : {'demean', 'detrend'}, default='demean'
        Transformation type; determines column prefix ('ydot' or 'ycheck').
    vce : {None, 'robust', 'hc0', 'hc1', 'hc2', 'hc3', 'hc4', 'cluster'}, optional
        Variance-covariance estimator. See `run_ols_regression()`.
    cluster_var : str, optional
        Cluster variable for cluster-robust standard errors. Required when
        vce='cluster'.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    controls : list of str, optional
        Time-invariant control variables to include in the regression.
        Controls are only used when both treated and control groups satisfy
        n > k + 1, where k is the number of control variables.
    never_treated_values : list, optional
        Deprecated. This parameter is accepted for backward compatibility but
        is ignored. Never-treated units are automatically detected via the
        is_never_treated() function which identifies gvar values of 0, NaN,
        or +inf as never-treated.

    Returns
    -------
    list of CohortEffect
        Cohort effect estimates sorted by cohort. Cohorts with insufficient
        data are skipped with warnings.

    Raises
    ------
    NoNeverTreatedError
        If no never-treated control units exist.
    """
    # Accept never_treated_values for backward compatibility but ignore it.
    # Never-treated detection is automatic via is_never_treated().
    if never_treated_values is not None:
        warnings.warn(
            "The 'never_treated_values' parameter is deprecated and ignored. "
            "Never-treated units are automatically detected based on gvar values "
            "(0, NaN, or +inf). You can safely remove this parameter.",
            DeprecationWarning,
            stacklevel=2
        )

    unit_gvar = _get_unit_level_gvar(data_transformed, gvar, ivar)
    nt_mask = _identify_nt_mask(unit_gvar)

    if nt_mask.sum() == 0:
        raise NoNeverTreatedError(_COHORT_NO_NT_ERROR)

    prefix = 'ydot' if transform_type == 'demean' else 'ycheck'
    results = []

    for g in cohorts:
        post_periods = get_valid_periods_for_cohort(int(g), int(T_max))

        ydot_cols = [f'{prefix}_g{int(g)}_r{r}' for r in post_periods
                     if f'{prefix}_g{int(g)}_r{r}' in data_transformed.columns]

        if not ydot_cols:
            # stacklevel=4: aggregate_to_cohort (2) -> internal caller (3) -> user code (4)
            warnings.warn(
                f"Cohort {g}: No valid transformation columns, skipping",
                stacklevel=4
            )
            continue

        try:
            Y_bar_ig = _compute_cohort_aggregated_variable(
                data_transformed, ivar, ydot_cols
            )
        except (ValueError, KeyError, IndexError) as e:
            # stacklevel=4: aggregate_to_cohort (2) -> internal caller (3) -> user code (4)
            warnings.warn(
                f"Cohort {g}: Aggregated variable calculation failed - {e}",
                stacklevel=4
            )
            continue

        # Build regression sample (cohort g + never-treated only)
        cohort_mask = get_cohort_mask(unit_gvar, g)
        sample_mask = cohort_mask | nt_mask
        sample_units = sample_mask[sample_mask].index

        reg_data = pd.DataFrame({
            'Y_bar_ig': Y_bar_ig.reindex(sample_units),
            'D_ig': cohort_mask.reindex(sample_units).astype(int),
        }, index=sample_units)

        # Add control variables if specified (unit-level, time-invariant)
        if controls:
            unit_controls = data_transformed.groupby(ivar)[controls].first()
            for ctrl in controls:
                if ctrl in unit_controls.columns:
                    reg_data[ctrl] = unit_controls[ctrl].reindex(sample_units)

        # Drop observations with missing values in outcome or control variables
        # This matches Stata's regress behavior which automatically drops missing values
        dropna_cols = ['Y_bar_ig'] + (controls if controls else [])
        reg_data = reg_data.dropna(subset=dropna_cols)

        n_treat = int(reg_data['D_ig'].sum())
        n_control = len(reg_data) - n_treat
        n_total = len(reg_data)

        if n_total < 2 or n_treat < 1 or n_control < 1:
            # stacklevel=4: aggregate_to_cohort (2) -> internal caller (3) -> user code (4)
            warnings.warn(
                f"Cohort {g}: Insufficient sample size (total={n_total}, treat={n_treat}, control={n_control})",
                stacklevel=4
            )
            continue

        if n_total == 2:
            # stacklevel=4: aggregate_to_cohort (2) -> internal caller (3) -> user code (4)
            warnings.warn(
                f"Cohort {g}: Sample size is only 2 units, "
                f"standard errors may be unreliable",
                UserWarning,
                stacklevel=4
            )

        if vce == 'cluster':
            # Validate cluster_var is provided when vce='cluster'.
            if cluster_var is None:
                raise ValueError(
                    "Parameter 'cluster_var' is required when vce='cluster'. "
                    "Please specify the cluster variable name."
                )
            reg_data = _add_cluster_to_reg_data(
                reg_data, data_transformed, ivar, cluster_var
            )
            cluster_col = '_cluster'
        else:
            cluster_col = None

        try:
            result = run_ols_regression(
                data=reg_data.reset_index(),
                y='Y_bar_ig',
                d='D_ig',
                controls=controls,
                vce=vce,
                cluster_var=cluster_col,
                alpha=alpha,
            )
        except (ValueError, np.linalg.LinAlgError) as e:
            # stacklevel=4: aggregate_to_cohort (2) -> internal caller (3) -> user code (4)
            warnings.warn(
                f"Cohort {g}: Regression estimation failed - {e}",
                stacklevel=4
            )
            continue

        # Validate that regression result contains required degrees of freedom.
        # The OLS regression must return df_resid and df_inference; missing values
        # indicate an implementation error rather than a recoverable data issue.
        if 'df_resid' not in result or 'df_inference' not in result:
            raise ValueError(
                f"Cohort {g}: Regression result missing required degrees of freedom. "
                f"Expected 'df_resid' and 'df_inference' keys. "
                f"Available keys: {list(result.keys())}"
            )

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
            df_resid=int(result['df_resid']),
            df_inference=int(result['df_inference']),
        ))
    
    results.sort(key=lambda x: x.cohort)

    n_requested = len(cohorts)
    n_successful = len(results)
    
    if n_successful == 0:
        # stacklevel=4: aggregate_to_cohort (2) -> internal caller (3) -> user code (4)
        warnings.warn(
            f"All cohort effect estimations failed ({n_requested} cohorts attempted).\n"
            f"Possible causes:\n"
            f"  1. Missing required transformation columns ({prefix}_g{{g}}_r{{r}})\n"
            f"  2. Insufficient sample size (each cohort needs at least 1 treated + 1 NT unit)\n"
            f"  3. Aggregated variable calculation failed (data contains missing values)\n"
            f"Please check if the data transformation step was completed correctly.",
            UserWarning,
            stacklevel=4
        )
    elif n_successful < n_requested:
        failed_cohorts = set(cohorts) - {r.cohort for r in results}
        # stacklevel=4: aggregate_to_cohort (2) -> internal caller (3) -> user code (4)
        warnings.warn(
            f"Some cohort effect estimations failed: {n_successful}/{n_requested} succeeded.\n"
            f"Failed cohorts: {sorted(failed_cohorts)}",
            UserWarning,
            stacklevel=4
        )
    
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
    Construct aggregated outcome variable for overall effect estimation.

    Computes Ȳ_i with treatment-status-specific construction:

    - Treated units (cohort g): Ȳ_i = Ȳ_ig (own cohort's time-averaged
      transformation)
    - Never-treated units: Ȳ_i = Σ_g ω_g · Ȳ_ig (cohort-weighted mixture)
    
    Raises
    ------
    ValueError
        If cohorts list is empty.

    The cohort-weighted construction for never-treated units ensures the
    counterfactual reflects the same weighting scheme as treated units.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with transformation columns.
    gvar : str
        Cohort variable column name.
    ivar : str
        Unit identifier column name.
    tvar : str
        Time variable column name. Used to validate consistency between
        the provided T_max parameter and the actual data range.
    weights : Dict[int, float]
        Cohort weights {g: ω_g}; must sum to 1 and keys must match cohorts.
    cohorts : list of int
        Treatment cohorts to include.
    T_max : int
        Maximum time period.
    transform_type : {'demean', 'detrend'}, default='demean'
        Transformation type; determines column prefix.
    never_treated_values : list, optional
        Deprecated. This parameter is accepted for backward compatibility but
        is ignored. Never-treated units are automatically detected via the
        is_never_treated() function which identifies gvar values of 0, NaN,
        or +inf as never-treated.

    Returns
    -------
    pd.Series
        Aggregated outcome Ȳ_i indexed by unit ID; NaN for units where
        aggregation cannot be computed.

    Raises
    ------
    ValueError
        If cohorts list is empty or weights keys do not match cohorts.
    """
    # Accept never_treated_values for backward compatibility but ignore it.
    # Never-treated detection is automatic via is_never_treated().
    if never_treated_values is not None:
        warnings.warn(
            "The 'never_treated_values' parameter is deprecated and ignored. "
            "Never-treated units are automatically detected based on gvar values "
            "(0, NaN, or +inf). You can safely remove this parameter.",
            DeprecationWarning,
            stacklevel=2
        )

    # Validate that cohorts list is not empty.
    if not cohorts:
        raise ValueError(
            "cohorts list cannot be empty.\n"
            "At least one treatment cohort is required to construct aggregated outcomes.\n\n"
            "Possible causes:\n"
            "  1. No treated units found in data\n"
            "  2. All cohorts were filtered out due to data issues\n"
            "  3. Incorrect gvar column specification"
        )
    
    weights_keys = set(weights.keys())
    cohorts_set = set(cohorts)
    if weights_keys != cohorts_set:
        missing_in_weights = cohorts_set - weights_keys
        extra_in_weights = weights_keys - cohorts_set
        raise ValueError(
            f"weights keys must match cohorts.\n"
            f"  Cohorts: {sorted(cohorts)}\n"
            f"  Weights keys: {sorted(weights.keys())}\n"
            f"  Missing in weights: {sorted(missing_in_weights) if missing_in_weights else 'None'}\n"
            f"  Extra in weights: {sorted(extra_in_weights) if extra_in_weights else 'None'}"
        )

    # Validate T_max consistency with data's time variable
    # The tvar parameter enables cross-validation between the provided T_max
    # and the actual maximum time period in the data
    if tvar in data.columns:
        data_T_max = data[tvar].max()
        if pd.notna(data_T_max) and int(data_T_max) != T_max:
            warnings.warn(
                f"T_max parameter ({T_max}) differs from data's max time value "
                f"({int(data_T_max)}). Using provided T_max={T_max}. "
                f"This may result in missing transformation columns if T_max > data max.",
                UserWarning,
                stacklevel=5
            )

    # Use fsum for numerically stable weight summation.
    weights_sum = fsum(weights.values())
    if not np.isclose(weights_sum, 1.0, atol=WEIGHT_SUM_TOLERANCE):
        # stacklevel=5: construct_aggregated_outcome (2) -> aggregate_to_overall (3) ->
        #              internal caller (4) -> user code (5)
        warnings.warn(
            f"Cohort weights sum to {weights_sum:.10f}, expected 1.0. "
            f"This may indicate incorrect weight calculation.",
            UserWarning,
            stacklevel=5
        )

    unit_gvar = _get_unit_level_gvar(data, gvar, ivar)
    nt_mask = _identify_nt_mask(unit_gvar)
    prefix = 'ydot' if transform_type == 'demean' else 'ycheck'

    Y_bar = pd.Series(index=unit_gvar.index, dtype=float)
    Y_bar[:] = np.nan

    cohort_Y_bar = {}
    cohort_errors = {}

    for g in cohorts:
        post_periods = get_valid_periods_for_cohort(int(g), int(T_max))
        ydot_cols = [f'{prefix}_g{int(g)}_r{r}' for r in post_periods 
                     if f'{prefix}_g{int(g)}_r{r}' in data.columns]
        
        if not ydot_cols:
            cohort_errors[g] = (
                f"No valid transformation columns "
                f"(need {prefix}_g{int(g)}_r* columns)"
            )
            continue

        try:
            cohort_Y_bar[g] = _compute_cohort_aggregated_variable(
                data, ivar, ydot_cols
            )
        except (ValueError, KeyError, IndexError) as e:
            cohort_errors[g] = f"Aggregation failed: {type(e).__name__}: {e}"
            continue

    # Separate counters for two distinct NT unit situations:
    # 1. nt_normalized: partial cohort data, successfully included via weight normalization
    # 2. nt_excluded: all cohorts missing, cannot compute aggregated outcome
    nt_normalized_count = 0
    nt_normalized_units = []
    nt_excluded_count = 0
    nt_excluded_units = []
    
    # Collect non-integer units locally to avoid state pollution across function calls.
    non_integer_units = []

    for unit_id in unit_gvar.index:
        g_unit = unit_gvar[unit_id]
        
        if nt_mask[unit_id]:
            # Never-treated: weighted sum across cohorts
            # Use lists with fsum for numerically stable accumulation.
            weighted_products = []  # stores weights[g] * Y_bar_ig
            weight_values = []      # stores weights[g] for valid cohorts
            missing_cohorts = []

            for g in cohorts:
                if g not in cohort_Y_bar:
                    missing_cohorts.append(g)
                    continue
                if unit_id not in cohort_Y_bar[g].index:
                    missing_cohorts.append(g)
                    continue
                    
                Y_bar_ig = cohort_Y_bar[g][unit_id]
                # Use pd.isna() for compatibility with pandas nullable types (pd.NA)
                # np.isnan() may raise TypeError for pd.NA values
                if pd.isna(Y_bar_ig):
                    missing_cohorts.append(g)
                    continue
                
                weighted_products.append(weights[g] * Y_bar_ig)
                weight_values.append(weights[g])
            
            # Use math.fsum for numerically stable summation
            weighted_sum = fsum(weighted_products) if weighted_products else 0.0
            valid_weights = fsum(weight_values) if weight_values else 0.0
            
            # Normalize weights when some cohorts are missing.
            # Use WEIGHT_SUM_TOLERANCE (1e-6) for weight sum comparisons to accommodate
            # cumulative floating-point rounding errors from multiple weight additions.
            if valid_weights > WEIGHT_SUM_TOLERANCE:
                # Normalize: Y_bar = weighted_sum / valid_weights
                normalized_value = weighted_sum / valid_weights
                Y_bar[unit_id] = normalized_value
                
                # Track units with normalized weights (partial cohort data)
                if not np.isclose(valid_weights, 1.0, atol=WEIGHT_SUM_TOLERANCE):
                    nt_normalized_count += 1
                    if len(nt_normalized_units) < 5:
                        nt_normalized_units.append((unit_id, missing_cohorts, valid_weights))
            else:
                # All cohorts are missing - cannot compute aggregated outcome
                nt_excluded_count += 1
                if len(nt_excluded_units) < 5:
                    nt_excluded_units.append((unit_id, list(cohorts)))

        elif np.isfinite(g_unit):
            # Treated: own cohort's transformation (np.isfinite handles NaN implicitly)
            g_rounded = round(g_unit)
            if abs(g_unit - g_rounded) > COHORT_FLOAT_TOLERANCE:
                # Collect for batch warning instead of per-unit warning.
                non_integer_units.append((unit_id, g_unit))
                continue
            g = int(g_rounded)
            if g in cohorts and g in cohort_Y_bar and unit_id in cohort_Y_bar[g].index:
                Y_bar[unit_id] = cohort_Y_bar[g][unit_id]
    
    # Issue single consolidated warning for non-integer gvar units.
    if non_integer_units:
        n_non_int = len(non_integer_units)
        examples = [f"unit {uid} (gvar={gv:.4f})" for uid, gv in non_integer_units[:3]]
        example_str = ", ".join(examples)
        if n_non_int > 3:
            example_str += f" ... and {n_non_int - 3} more"
        # stacklevel=5: construct_aggregated_outcome (2) -> aggregate_to_overall (3) ->
        #              internal caller (4) -> user code (5)
        warnings.warn(
            f"{n_non_int} unit(s) have non-integer gvar values and were skipped from aggregation. "
            f"Examples: {example_str}",
            UserWarning,
            stacklevel=5
        )

    treated_nan_count = 0
    treated_nan_units = []
    for unit_id, y_val in Y_bar.items():
        if pd.isna(y_val):
            gvar_val = unit_gvar.get(unit_id, None)
            if gvar_val is not None and gvar_val > 0 and np.isfinite(gvar_val):
                treated_nan_count += 1
                if len(treated_nan_units) < 5:
                    treated_nan_units.append((unit_id, gvar_val))
    
    if treated_nan_count > 0:
        example_info = ""
        if treated_nan_units:
            examples = [f"unit {uid} (cohort {g})" for uid, g in treated_nan_units[:3]]
            example_info = f"\nExamples: {', '.join(examples)}"
            if treated_nan_count > 3:
                example_info += f" ... and {treated_nan_count - 3} more"

        warnings.warn(
            f"{treated_nan_count} treated unit(s) have NaN Y_bar values.\n"
            f"These units will be excluded from overall effect estimation.\n"
            f"Possible causes: missing data, insufficient observations.{example_info}",
            UserWarning,
            stacklevel=5
        )

    # Warning for NT units with partial cohort data (normalized and included)
    if nt_normalized_count > 0:
        example_info = ""
        if nt_normalized_units:
            examples = [
                f"unit {uid} (missing: {mc}, weight_sum={wt:.4f})"
                for uid, mc, wt in nt_normalized_units[:3]
            ]
            example_info = f"\nExamples: {', '.join(examples)}"
            if nt_normalized_count > 3:
                example_info += f" ... and {nt_normalized_count - 3} more"

        warnings.warn(
            f"{nt_normalized_count} never-treated unit(s) have incomplete cohort data.\n"
            f"Weights were renormalized to available cohorts (included in regression).\n"
            f"This may affect estimation if missingness is non-random."
            f"{example_info}",
            UserWarning,
            stacklevel=5
        )

    # Warning for NT units completely excluded (all cohorts missing)
    if nt_excluded_count > 0:
        example_info = ""
        if nt_excluded_units:
            examples = [
                f"unit {uid} (all {len(mc)} cohorts missing)"
                for uid, mc in nt_excluded_units[:3]
            ]
            example_info = f"\nExamples: {', '.join(examples)}"
            if nt_excluded_count > 3:
                example_info += f" ... and {nt_excluded_count - 3} more"

        warnings.warn(
            f"{nt_excluded_count} never-treated unit(s) excluded due to "
            f"missing data for all cohorts.\n"
            f"These units have no valid cohort observations and cannot be included.\n"
            f"They will be dropped from overall effect regression."
            f"{example_info}",
            UserWarning,
            stacklevel=5
        )

    n_valid = Y_bar.notna().sum()
    n_total = len(Y_bar)
    valid_ratio = n_valid / n_total if n_total > 0 else 0
    
    if n_valid == 0:
        error_details = []
        if cohort_errors:
            error_details.append("Cohort processing details:")
            for g, err in sorted(cohort_errors.items()):
                error_details.append(f"  - Cohort {g}: {err}")
        
        warnings.warn(
            f"All aggregated results are NaN. Possible causes:\n"
            f"  1. Missing required transformation columns ({prefix}_g{{g}}_r{{r}})\n"
            f"  2. All cohort aggregation calculations failed\n"
            f"  3. No valid treated or control units in data\n"
            f"Please check if the data transformation step was completed correctly.\n"
            f"Attempted cohorts: {list(cohorts)}\n"
            f"Successfully pre-computed cohorts: {list(cohort_Y_bar.keys())}\n"
            + ("\n".join(error_details) if error_details else ""),
            UserWarning,
            stacklevel=5
        )
    elif valid_ratio < 0.5:
        warnings.warn(
            f"Low aggregation success rate: {n_valid}/{n_total} ({valid_ratio:.1%}).\n"
            f"Some units' aggregated variables could not be computed, which may affect estimation reliability.\n"
            f"Successfully pre-computed cohorts: {list(cohort_Y_bar.keys())}"
            + (f"\nFailed cohorts: {list(cohort_errors.keys())}" if cohort_errors else ""),
            UserWarning,
            stacklevel=5
        )
    
    return Y_bar


def aggregate_to_overall(
    data_transformed: pd.DataFrame,
    gvar: str,
    ivar: str,
    tvar: str,
    transform_type: str = 'demean',
    vce: Optional[str] = None,
    cluster_var: Optional[str] = None,
    alpha: float = 0.05,
    controls: Optional[List[str]] = None,
    never_treated_values: Optional[List] = None,
) -> OverallEffect:
    """
    Estimate overall cohort-size-weighted treatment effect.

    Estimates τ_ω via cross-sectional OLS:

        Ȳ_i = α + τ_ω · D_i + X_i'γ + D_i·(X_i - X̄₁)'δ + ε_i

    where D_i indicates ever-treated status, Ȳ_i is the aggregated outcome
    with cohort weights ω_g = N_g / Σ_h N_h proportional to cohort sizes, and
    X_i are time-invariant controls centered at treated mean.

    Parameters
    ----------
    data_transformed : pd.DataFrame
        Panel data with transformation columns named 'ydot_g{g}_r{r}' (demean)
        or 'ycheck_g{g}_r{r}' (detrend).
    gvar : str
        Cohort variable column name.
    ivar : str
        Unit identifier column name.
    tvar : str
        Time variable column name.
    transform_type : {'demean', 'detrend'}, default='demean'
        Transformation type; determines column prefix.
    vce : {None, 'robust', 'hc0', 'hc1', 'hc2', 'hc3', 'hc4', 'cluster'}, optional
        Variance-covariance estimator. See `run_ols_regression()`.
    cluster_var : str, optional
        Cluster variable for cluster-robust standard errors. Required when
        vce='cluster'.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    controls : list of str, optional
        Time-invariant control variables to include in the regression.
        Controls are only used when both treated and control groups satisfy
        n > k + 1, where k is the number of control variables.
    never_treated_values : list, optional
        Deprecated. This parameter is accepted for backward compatibility but
        is ignored. Never-treated units are automatically detected via the
        is_never_treated() function which identifies gvar values of 0, NaN,
        or +inf as never-treated.

    Returns
    -------
    OverallEffect
        Overall effect estimate with point estimate, standard error,
        confidence interval, test statistics, cohort weights, and sample sizes.

    Raises
    ------
    NoNeverTreatedError
        If no never-treated control units exist.
    ValueError
        If sample size insufficient or no valid cohorts exist.
    """
    # Accept never_treated_values for backward compatibility but ignore it.
    # Never-treated detection is automatic via is_never_treated().
    if never_treated_values is not None:
        warnings.warn(
            "The 'never_treated_values' parameter is deprecated and ignored. "
            "Never-treated units are automatically detected based on gvar values "
            "(0, NaN, or +inf). You can safely remove this parameter.",
            DeprecationWarning,
            stacklevel=2
        )

    unit_gvar = _get_unit_level_gvar(data_transformed, gvar, ivar)
    nt_mask = _identify_nt_mask(unit_gvar)

    if nt_mask.sum() == 0:
        raise NoNeverTreatedError(_OVERALL_NO_NT_ERROR)

    cohorts = get_cohorts(data_transformed, gvar, ivar)

    if len(cohorts) == 0:
        raise ValueError("No valid treatment cohorts found in data")

    cohort_sizes = {g: int(get_cohort_mask(unit_gvar, g).sum()) for g in cohorts}

    zero_cohorts = [g for g, n in cohort_sizes.items() if n == 0]
    if zero_cohorts:
        # stacklevel=4: aggregate_to_overall (2) -> internal caller (3) -> user code (4)
        warnings.warn(
            f"Cohorts with 0 units detected and excluded: {sorted(zero_cohorts)}. "
            f"This may indicate data filtering issues.",
            UserWarning,
            stacklevel=4
        )
        cohort_sizes = {g: n for g, n in cohort_sizes.items() if n > 0}
        cohorts = [g for g in cohorts if g in cohort_sizes]

    if not cohort_sizes:
        raise ValueError(
            "No valid cohorts with units found after filtering. "
            "All cohorts have 0 units. Please check data integrity."
        )
    
    N_treat = sum(cohort_sizes.values())

    if N_treat == 0:
        raise ValueError("No treated units found")

    weights = {g: n / N_treat for g, n in cohort_sizes.items()}

    tvar_max = data_transformed[tvar].max()
    if pd.isna(tvar_max):
        raise ValueError(f"Time variable '{tvar}' contains no valid (non-NaN) values")
    T_max = int(tvar_max)

    Y_bar = construct_aggregated_outcome(
        data_transformed, gvar, ivar, tvar,
        weights, cohorts, T_max,
        transform_type
    )

    D_ever = (~nt_mask).astype(int)

    reg_data = pd.DataFrame({
        'Y_bar': Y_bar,
        'D_ever': D_ever,
    }, index=unit_gvar.index)

    # Add control variables if specified (unit-level, time-invariant)
    if controls:
        unit_controls = data_transformed.groupby(ivar)[controls].first()
        for ctrl in controls:
            if ctrl in unit_controls.columns:
                reg_data[ctrl] = unit_controls[ctrl].reindex(unit_gvar.index)

    # Drop observations with missing values in outcome or control variables
    # This matches Stata's regress behavior which automatically drops missing values
    dropna_cols = ['Y_bar'] + (controls if controls else [])
    reg_data = reg_data.dropna(subset=dropna_cols)

    n_treated = int(reg_data['D_ever'].sum())
    n_control = len(reg_data) - n_treated
    n_total = len(reg_data)
    
    # Compute effective cohort weights after dropna and warn if different.
    # The pre-dropna weights were used to construct Y_bar; check if actual sample differs.
    effective_gvar = unit_gvar.reindex(reg_data.index)
    effective_cohort_sizes = {
        g: int(get_cohort_mask(effective_gvar, g).sum()) 
        for g in cohorts
    }
    N_treat_effective = sum(effective_cohort_sizes.values())
    if N_treat_effective > 0:
        effective_weights = {g: n / N_treat_effective for g, n in effective_cohort_sizes.items()}
    else:
        effective_weights = weights.copy()
    
    # Check for significant weight differences due to differential NaN rates across cohorts
    weight_diffs = {g: abs(weights.get(g, 0) - effective_weights.get(g, 0)) for g in cohorts}
    max_diff = max(weight_diffs.values()) if weight_diffs else 0
    if max_diff > 0.01:  # More than 1% difference
        warnings.warn(
            f"Cohort weights differ after dropping missing values (max diff: {max_diff:.3f}). "
            f"This may occur when NaN rates vary across cohorts. "
            f"Intended weights: {weights}; Effective weights: {effective_weights}",
            UserWarning,
            stacklevel=4
        )
        # Update weights to reflect actual sample used in regression
        weights = effective_weights

    if n_total < 2 or n_treated < 1 or n_control < 1:
        raise ValueError(
            _INSUFFICIENT_SAMPLE_ERROR.format(
                n_treat=n_treated, n_control=n_control, n_total=n_total
            )
        )

    if n_total == 2:
        # stacklevel=4: aggregate_to_overall (2) -> internal caller (3) -> user code (4)
        warnings.warn(
            "Sample size is only 2 units, standard errors may be unreliable",
            UserWarning,
            stacklevel=4
        )

    if vce == 'cluster':
        # Validate cluster_var is provided when vce='cluster'.
        if cluster_var is None:
            raise ValueError(
                "Parameter 'cluster_var' is required when vce='cluster'. "
                "Please specify the cluster variable name."
            )
        reg_data = _add_cluster_to_reg_data(
            reg_data, data_transformed, ivar, cluster_var
        )
        cluster_col = '_cluster'
    else:
        cluster_col = None

    result = run_ols_regression(
        data=reg_data.reset_index(),
        y='Y_bar',
        d='D_ever',
        controls=controls,
        vce=vce,
        cluster_var=cluster_col,
        alpha=alpha,
    )

    # Validate that regression result contains required degrees of freedom.
    # The OLS regression must return df_resid and df_inference; missing values
    # indicate an implementation error rather than a recoverable data issue.
    if 'df_resid' not in result or 'df_inference' not in result:
        raise ValueError(
            "Overall effect regression result missing required degrees of freedom. "
            f"Expected 'df_resid' and 'df_inference' keys. "
            f"Available keys: {list(result.keys())}"
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
        df_resid=int(result['df_resid']),
        df_inference=int(result['df_inference']),
    )


def cohort_effects_to_dataframe(results: List[CohortEffect]) -> pd.DataFrame:
    """
    Convert CohortEffect objects to a pandas DataFrame.

    Parameters
    ----------
    results : list of CohortEffect
        Cohort effect estimates from `aggregate_to_cohort()`.

    Returns
    -------
    pd.DataFrame
        Tabular representation with columns: cohort, att, se, ci_lower,
        ci_upper, t_stat, pvalue, n_periods, n_units, n_control, df_resid,
        df_inference. Returns an empty DataFrame with expected schema if
        results is empty.

    Raises
    ------
    TypeError
        If results is None (programming error). Pass empty list [] for
        no cohort effects.
    """
    # Validate input type: None indicates programming error
    # Empty list is valid and returns empty DataFrame with correct schema
    if results is None:
        raise TypeError(
            "results cannot be None. Expected a list of CohortEffect objects. "
            "If no cohort effects were estimated, pass an empty list []."
        )

    if len(results) == 0:
        # Specify dtypes for empty DataFrame to match non-empty case.
        # Without explicit dtypes, all columns default to object type.
        return pd.DataFrame({
            'cohort': pd.Series(dtype='int64'),
            'att': pd.Series(dtype='float64'),
            'se': pd.Series(dtype='float64'),
            'ci_lower': pd.Series(dtype='float64'),
            'ci_upper': pd.Series(dtype='float64'),
            't_stat': pd.Series(dtype='float64'),
            'pvalue': pd.Series(dtype='float64'),
            'n_periods': pd.Series(dtype='int64'),
            'n_units': pd.Series(dtype='int64'),
            'n_control': pd.Series(dtype='int64'),
            'df_resid': pd.Series(dtype='int64'),
            'df_inference': pd.Series(dtype='int64'),
        })
    
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
            'df_resid': r.df_resid,
            'df_inference': r.df_inference,
        }
        for r in results
    ])

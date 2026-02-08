"""
Aggregation of cohort-time ATT estimates to summary parameters.

This module aggregates (g, r)-specific treatment effect estimates into
cohort-level, overall, and event-time summary parameters for staggered
difference-in-differences designs.

Three aggregation levels are supported:

- **Cohort Effect**: Time-averaged ATT for each cohort g across post-treatment
  periods, estimated via regression of time-averaged transformed outcomes on
  cohort membership indicators with never-treated controls.

- **Overall Effect**: Cohort-size-weighted average ATT across all cohorts,
  estimated via regression on ever-treated indicators with cohort weights
  proportional to cohort sizes.

- **Event-Time Effect (WATT)**: Weighted ATT aggregated by event time (relative
  time since treatment). Weights are proportional to cohort sizes, and
  t-distribution is used for proper inference with finite samples.

Key Classes
-----------
CohortEffect
    Cohort-specific time-averaged treatment effect estimate.
OverallEffect
    Overall cohort-size-weighted treatment effect estimate.
EventTimeEffect
    Event-time aggregated treatment effect estimate (WATT).

Key Functions
-------------
aggregate_to_cohort
    Aggregate period-specific effects to cohort-level effects.
aggregate_to_overall
    Estimate overall cohort-size-weighted treatment effect.
aggregate_to_event_time
    Aggregate cohort-time ATT estimates to event-time weighted ATT (WATT).

Notes
-----
Both cohort and overall aggregation levels require never-treated units as
controls. Each cohort uses a different pre-treatment reference period for
outcome transformation, so only never-treated units provide a consistent
counterfactual baseline. When no never-treated units exist, use aggregate='none'
to estimate (g, r)-specific effects with not-yet-treated controls.

The event-time aggregation computes:

.. math::

    \\text{WATT}(r) = \\sum_{g \\in G_r} w(g, r) \\cdot \\widehat{\\tau}_{g, g+r}

where :math:`G_r` is the set of cohorts observed at event time r, and weights
:math:`w(g, r) = N_g / \\sum_{g' \\in G_r} N_{g'}` are proportional to cohort
sizes. The standard error assumes independence across cohorts:

.. math::

    \\text{SE}(\\text{WATT}(r)) = \\sqrt{\\sum_{g \\in G_r} w(g, r)^2 \\cdot
    \\text{SE}(\\widehat{\\tau}_{g, g+r})^2}

Inference uses the t-distribution with conservative degrees of freedom
(minimum across contributing cohorts) for proper finite-sample coverage.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from math import fsum

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


# =============================================================================
# Data Classes: Result Containers
# =============================================================================


@dataclass
class CohortEffect:
    """
    Cohort-specific time-averaged treatment effect estimate.

    Stores the ATT estimate for a single treatment cohort, computed by
    averaging the transformed outcome across all post-treatment periods
    and regressing on cohort membership with never-treated controls.

    Attributes
    ----------
    cohort : int
        Treatment cohort identifier (first treatment period).
    att : float
        Point estimate of the cohort-level ATT.
    se : float
        Standard error of the ATT estimate.
    ci_lower : float
        Lower bound of the confidence interval.
    ci_upper : float
        Upper bound of the confidence interval.
    t_stat : float
        t-statistic for the null hypothesis of zero effect.
    pvalue : float
        Two-sided p-value.
    n_periods : int
        Number of post-treatment periods included in the average.
    n_units : int
        Number of treated units in this cohort.
    n_control : int
        Number of never-treated control units.
    df_resid : int
        Residual degrees of freedom from the aggregation regression.
    df_inference : int
        Degrees of freedom used for inference (equals df_resid for
        homoskedastic/HC variance, or number of clusters minus one for
        cluster-robust variance).

    See Also
    --------
    aggregate_to_cohort : Function that produces CohortEffect instances.
    cohort_effects_to_dataframe : Convert results to DataFrame format.
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

    Stores the aggregate ATT estimate across all treatment cohorts, with
    weights proportional to cohort sizes. The weight for cohort g is
    :math:`w(g) = N_g / \\sum_{g'} N_{g'}`, where :math:`N_g` is the number
    of treated units in cohort g.

    Attributes
    ----------
    att : float
        Point estimate of the overall weighted ATT.
    se : float
        Standard error of the ATT estimate.
    ci_lower : float
        Lower bound of the confidence interval.
    ci_upper : float
        Upper bound of the confidence interval.
    t_stat : float
        t-statistic for the null hypothesis of zero effect.
    pvalue : float
        Two-sided p-value.
    cohort_weights : dict[int, float]
        Mapping from cohort identifiers to weights (weights sum to one).
    n_treated : int
        Total number of ever-treated units.
    n_control : int
        Number of never-treated control units.
    df_resid : int
        Residual degrees of freedom from the aggregation regression.
    df_inference : int
        Degrees of freedom used for inference (equals df_resid for
        homoskedastic/HC variance, or number of clusters minus one for
        cluster-robust variance).

    See Also
    --------
    aggregate_to_overall : Function that produces OverallEffect instances.
    """
    att: float
    se: float
    ci_lower: float
    ci_upper: float
    t_stat: float
    pvalue: float
    cohort_weights: dict[int, float]
    n_treated: int
    n_control: int
    df_resid: int = 0
    df_inference: int = 0


@dataclass
class EventTimeEffect:
    """
    Event-time aggregated treatment effect estimate (WATT).

    Represents the weighted average treatment effect at event time r, computed
    as WATT(r) = Σ_{g∈G_r} w(g,r) · ATT(g, g+r), where weights are proportional
    to cohort sizes.

    Attributes
    ----------
    event_time : int
        Relative time since treatment (r = period - cohort). Negative values
        indicate pre-treatment periods, positive values indicate post-treatment.
    att : float
        Point estimate of the weighted ATT at this event time.
    se : float
        Standard error: SE(WATT(r)) = sqrt(Σ [w(g,r)]² × [SE(ATT)]²).
    ci_lower : float
        Lower bound of confidence interval using t-distribution.
    ci_upper : float
        Upper bound of confidence interval using t-distribution.
    t_stat : float
        t-statistic: WATT(r) / SE(WATT(r)).
    pvalue : float
        Two-sided p-value from t-distribution.
    df_inference : int
        Degrees of freedom for t-distribution inference.
        Uses min(df_g) across contributing cohorts as conservative choice.
    n_cohorts : int
        Number of cohorts contributing to this event time estimate.
    cohort_contributions : dict[int, float]
        Mapping from cohort g to its weighted contribution w(g,r) × ATT(g, g+r).
    weight_sum : float
        Sum of normalized weights (should equal 1.0 for validation).
    alpha : float
        Significance level used for confidence interval (default 0.05).

    See Also
    --------
    aggregate_to_event_time : Function that produces EventTimeEffect instances.
    event_time_effects_to_dataframe : Convert results to DataFrame format.
    """
    event_time: int
    att: float
    se: float
    ci_lower: float
    ci_upper: float
    t_stat: float
    pvalue: float
    df_inference: int
    n_cohorts: int
    cohort_contributions: dict[int, float]
    weight_sum: float
    alpha: float = 0.05


# =============================================================================
# Helper Functions: Internal Utilities
# =============================================================================


def _compute_cohort_aggregated_variable(
    data: pd.DataFrame,
    ivar: str,
    ydot_cols: list[str],
) -> pd.Series:
    """
    Compute unit-level time-averaged transformed outcome.

    For each unit, averages the transformed outcome values across all
    specified columns, handling the sparse structure where each row
    typically has only one non-NaN value corresponding to its cohort.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data containing transformation columns.
    ivar : str
        Unit identifier column name.
    ydot_cols : list of str
        Transformation column names to average.

    Returns
    -------
    pd.Series
        Unit-level aggregated outcome indexed by unit ID.

    Raises
    ------
    ValueError
        If ydot_cols is empty or contains columns not present in data.

    Warns
    -----
    UserWarning
        If rows contain multiple non-NaN values (unexpected for correctly
        transformed data) or if units are excluded due to missing data.

    See Also
    --------
    aggregate_to_cohort : Uses this function for cohort effect estimation.
    construct_aggregated_outcome : Uses this function for overall effect.

    Notes
    -----
    The transformed data structure places each (cohort, period) outcome
    in a separate column. For treated units, only their own cohort's
    column contains non-NaN values. This function aggregates across
    time periods within a unit.
    """
    if not ydot_cols:
        raise ValueError("No transformation columns provided for aggregation")

    missing_cols = [c for c in ydot_cols if c not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing transformation columns: {missing_cols}")

    # Each row should have at most one non-NaN value since each (g,r) pair
    # produces a separate column. Multiple non-NaN values indicate either
    # data corruption or incorrect column selection, warranting a warning.
    ydot_df = data[ydot_cols]
    non_nan_count = (~ydot_df.isna()).sum(axis=1).values
    multi_value_rows = non_nan_count > 1

    if np.any(multi_value_rows):
        n_problematic = int(np.sum(multi_value_rows))
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

    Deduplicates panel data to unit level by taking the first observed
    cohort value for each unit. Cohort assignment is time-invariant by
    construction, so any observation suffices.

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
        Unit-level cohort values indexed by unit ID.

    See Also
    --------
    _identify_nt_mask : Uses output to identify never-treated units.
    get_cohort_mask : Uses cohort values for treatment indicator construction.

    Notes
    -----
    Uses groupby().first() rather than drop_duplicates() for efficiency
    with large panels. The result is used for constructing treatment
    indicators and identifying never-treated units.
    """
    return data.groupby(ivar)[gvar].first()


def _identify_nt_mask(unit_gvar: pd.Series) -> pd.Series:
    """
    Create boolean mask identifying never-treated units.

    Applies the package's never-treated detection logic to each unit's
    cohort value, recognizing NaN, 0, and positive infinity as indicators
    of never-treated status.

    Parameters
    ----------
    unit_gvar : pd.Series
        Unit-level cohort values indexed by unit ID.

    Returns
    -------
    pd.Series
        Boolean mask indexed by unit ID; True indicates never-treated.

    See Also
    --------
    is_never_treated : Underlying detection function for individual values.
    """
    return unit_gvar.apply(is_never_treated)


def _add_cluster_to_reg_data(
    reg_data: pd.DataFrame,
    original_data: pd.DataFrame,
    ivar: str,
    cluster_var: str | None,
) -> pd.DataFrame:
    """
    Merge cluster identifiers into unit-level regression data.

    Extracts cluster assignments from panel data and maps them to the
    unit-level regression dataset for cluster-robust variance estimation.

    Parameters
    ----------
    reg_data : pd.DataFrame
        Unit-level regression data with unit IDs as index.
    original_data : pd.DataFrame
        Panel data containing the cluster variable.
    ivar : str
        Unit identifier column name.
    cluster_var : str or None
        Cluster variable column name. If None, returns reg_data unchanged.

    Returns
    -------
    pd.DataFrame
        Copy of regression data with '_cluster' column added.

    Raises
    ------
    ValueError
        If cluster_var is not found in original_data columns.

    Warns
    -----
    UserWarning
        If units have missing cluster values or if the number of clusters
        is below the recommended threshold for reliable inference.

    See Also
    --------
    aggregate_to_cohort : Calls this function for cluster-robust inference.
    aggregate_to_overall : Calls this function for cluster-robust inference.

    Notes
    -----
    Cluster-robust standard errors require sufficient clusters (typically
    20 or more) for asymptotic validity. Small cluster counts may produce
    unreliable inference even with many observations.
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
        warnings.warn(
            f"Number of clusters ({n_clusters}) is small, clustered standard errors may be unreliable. "
            f"At least 20 clusters are recommended.",
            UserWarning,
            stacklevel=3
        )

    return reg_data


# =============================================================================
# Cohort-Level Aggregation
# =============================================================================


def aggregate_to_cohort(
    data_transformed: pd.DataFrame,
    gvar: str,
    ivar: str,
    tvar: str,
    cohorts: list[int],
    T_max: int,
    transform_type: str = 'demean',
    vce: str | None = None,
    cluster_var: str | None = None,
    alpha: float = 0.05,
    controls: list[str] | None = None,
    never_treated_values: list | None = None,
) -> list[CohortEffect]:
    """
    Aggregate period-specific effects to cohort-level effects.

    For each cohort, estimates the time-averaged ATT via cross-sectional OLS
    regression of time-averaged transformed outcomes on cohort membership
    indicators with never-treated controls.

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
        Maximum time period.
    transform_type : {'demean', 'detrend'}, default='demean'
        Transformation type determining column prefix.
    vce : {None, 'robust', 'hc0', 'hc1', 'hc2', 'hc3', 'hc4', 'cluster'}, optional
        Variance-covariance estimator.
    cluster_var : str, optional
        Cluster variable for cluster-robust standard errors.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    controls : list of str, optional
        Time-invariant control variables to include.
    never_treated_values : list, optional
        Deprecated and ignored. Never-treated units are detected automatically.

    Returns
    -------
    list of CohortEffect
        Cohort effect estimates sorted by cohort.

    Raises
    ------
    NoNeverTreatedError
        If no never-treated control units exist.

    See Also
    --------
    CohortEffect : Result container for cohort-level estimates.
    aggregate_to_overall : Aggregate across cohorts to overall effect.
    cohort_effects_to_dataframe : Convert results to DataFrame format.

    Notes
    -----
    For each cohort g, the time-averaged transformed outcome is:

    .. math::

        \\bar{Y}_{ig} = \\frac{1}{T - g + 1} \\sum_{r=g}^{T} \\dot{Y}_{irg}

    where :math:`\\dot{Y}_{irg}` is the demeaned or detrended outcome for unit i
    in period r. The cohort ATT is estimated by regressing :math:`\\bar{Y}_{ig}`
    on cohort membership indicator :math:`D_{ig}` using never-treated controls:

    .. math::

        \\bar{Y}_{ig} = \\alpha + \\tau_g D_{ig} + \\varepsilon_i

    The coefficient :math:`\\tau_g` estimates the time-averaged treatment effect
    for cohort g.
    """
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
            warnings.warn(
                f"Cohort {g}: Aggregated variable calculation failed - {e}",
                stacklevel=4
            )
            continue

        # Restrict to cohort g and never-treated units. Not-yet-treated units
        # are excluded because each cohort uses different pre-treatment periods,
        # so only never-treated units provide consistent counterfactual baselines.
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

        # Drop observations with missing values to ensure complete cases for regression.
        dropna_cols = ['Y_bar_ig'] + (controls if controls else [])
        reg_data = reg_data.dropna(subset=dropna_cols)

        n_treat = int(reg_data['D_ig'].sum())
        n_control = len(reg_data) - n_treat
        n_total = len(reg_data)

        if n_total < 2 or n_treat < 1 or n_control < 1:
            warnings.warn(
                f"Cohort {g}: Insufficient sample size (total={n_total}, treat={n_treat}, control={n_control})",
                stacklevel=4
            )
            continue

        if n_total == 2:
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
    weights: dict[int, float],
    cohorts: list[int],
    T_max: int,
    transform_type: str = 'demean',
    never_treated_values: list | None = None,
) -> pd.Series:
    """
    Construct aggregated outcome variable for overall effect estimation.

    Computes unit-level aggregated outcomes with treatment-status-specific
    construction: treated units use their own cohort's time-averaged
    transformation, while never-treated units use a cohort-weighted mixture.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with transformation columns.
    gvar : str
        Cohort variable column name.
    ivar : str
        Unit identifier column name.
    tvar : str
        Time variable column name.
    weights : dict[int, float]
        Cohort weights (must sum to one, keys must match cohorts).
    cohorts : list of int
        Treatment cohorts to include.
    T_max : int
        Maximum time period.
    transform_type : {'demean', 'detrend'}, default='demean'
        Transformation type determining column prefix.
    never_treated_values : list, optional
        Deprecated and ignored. Never-treated units are detected automatically.

    Returns
    -------
    pd.Series
        Aggregated outcome indexed by unit ID.

    Raises
    ------
    ValueError
        If cohorts list is empty or weights keys do not match cohorts.

    See Also
    --------
    aggregate_to_overall : Uses this function to construct outcomes.
    _compute_cohort_aggregated_variable : Computes cohort-specific averages.

    Notes
    -----
    The aggregated outcome is constructed differently by treatment status:

    - **Treated units**: Use their own cohort's time-averaged transformation,
      :math:`\\bar{Y}_i = \\bar{Y}_{ig}` where g is the unit's cohort.

    - **Never-treated units**: Use a weighted average across all cohorts,
      :math:`\\bar{Y}_i = \\sum_g w(g) \\cdot \\bar{Y}_{ig}`.

    When some cohorts have missing data for a never-treated unit, weights
    are renormalized to available cohorts. This preserves unbiasedness if
    missingness is unrelated to potential outcomes.
    """
    if never_treated_values is not None:
        warnings.warn(
            "The 'never_treated_values' parameter is deprecated and ignored. "
            "Never-treated units are automatically detected based on gvar values "
            "(0, NaN, or +inf). You can safely remove this parameter.",
            DeprecationWarning,
            stacklevel=2
        )

    # Validate inputs.
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
                if pd.isna(Y_bar_ig):
                    missing_cohorts.append(g)
                    continue
                
                weighted_products.append(weights[g] * Y_bar_ig)
                weight_values.append(weights[g])
            
            # Use math.fsum for numerically stable summation
            weighted_sum = fsum(weighted_products) if weighted_products else 0.0
            valid_weights = fsum(weight_values) if weight_values else 0.0
            
            # Renormalize weights when some cohorts have missing data. This ensures
            # the weighted average reflects only available cohorts rather than being
            # biased downward by treating missing cohorts as zeros. The tolerance
            # threshold distinguishes genuine partial data from complete missingness.
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


# =============================================================================
# Overall Effect Aggregation
# =============================================================================


def aggregate_to_overall(
    data_transformed: pd.DataFrame,
    gvar: str,
    ivar: str,
    tvar: str,
    transform_type: str = 'demean',
    vce: str | None = None,
    cluster_var: str | None = None,
    alpha: float = 0.05,
    controls: list[str] | None = None,
    never_treated_values: list | None = None,
) -> OverallEffect:
    """
    Estimate overall cohort-size-weighted treatment effect.

    Estimates the weighted average ATT across all treated cohorts via
    cross-sectional OLS regression on aggregated transformed outcomes with
    cohort weights proportional to cohort sizes and never-treated controls.

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
        Transformation type determining column prefix.
    vce : {None, 'robust', 'hc0', 'hc1', 'hc2', 'hc3', 'hc4', 'cluster'}, optional
        Variance-covariance estimator.
    cluster_var : str, optional
        Cluster variable for cluster-robust standard errors.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    controls : list of str, optional
        Time-invariant control variables to include.
    never_treated_values : list, optional
        Deprecated and ignored. Never-treated units are detected automatically.

    Returns
    -------
    OverallEffect
        Overall effect estimate with inference statistics and cohort weights.

    Raises
    ------
    NoNeverTreatedError
        If no never-treated control units exist.
    ValueError
        If sample size insufficient or no valid cohorts exist.

    See Also
    --------
    OverallEffect : Result container for overall effect estimate.
    aggregate_to_cohort : Estimate cohort-specific effects.
    construct_aggregated_outcome : Helper function for outcome construction.

    Notes
    -----
    Cohort weights are proportional to cohort sizes:

    .. math::

        w(g) = \\frac{N_g}{\\sum_{g'} N_{g'}}

    For treated units, the aggregated outcome equals their cohort's
    time-averaged transformation. For never-treated units, it is a
    weighted average across all cohorts' transformations:

    .. math::

        \\bar{Y}_i = \\sum_g w(g) \\cdot \\bar{Y}_{ig}

    The overall ATT is estimated by regressing :math:`\\bar{Y}_i` on an
    ever-treated indicator with never-treated controls.
    """
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

    # Drop observations with missing values to ensure complete cases for regression.
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


# =============================================================================
# DataFrame Conversion Utilities
# =============================================================================


def cohort_effects_to_dataframe(results: list[CohortEffect]) -> pd.DataFrame:
    """
    Convert CohortEffect objects to a pandas DataFrame.

    Parameters
    ----------
    results : list of CohortEffect
        Cohort effect estimates from aggregate_to_cohort.

    Returns
    -------
    pd.DataFrame
        Tabular representation with one row per cohort.

    Raises
    ------
    TypeError
        If results is None.
    """
    if results is None:
        raise TypeError(
            "results cannot be None. Expected a list of CohortEffect objects. "
            "If no cohort effects were estimated, pass an empty list []."
        )

    if len(results) == 0:
        # Return empty DataFrame with correct dtypes.
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


# =============================================================================
# Event-Time Aggregation
# =============================================================================


def _compute_event_time_weights(
    cohort_sizes: dict[int, int],
    available_cohorts: list[int],
) -> dict[int, float]:
    """
    Compute normalized weights for available cohorts at a given event time.

    Computes cohort-size-proportional weights: w(g,r) = N_g / Σ_{g'∈G_r} N_{g'},
    where N_g is the number of treated units in cohort g.

    Parameters
    ----------
    cohort_sizes : dict[int, int]
        Mapping from cohort identifier to number of treated units.
    available_cohorts : list[int]
        Cohorts with valid ATT estimates at this event time.

    Returns
    -------
    dict[int, float]
        Mapping from cohort to normalized weight.

    Raises
    ------
    ValueError
        If available_cohorts is empty or total size is zero.

    See Also
    --------
    aggregate_to_event_time : Uses this function for weight computation.
    _validate_weight_sum : Validates computed weights sum to 1.0.

    Notes
    -----
    Uses math.fsum for numerically stable summation.
    """
    if not available_cohorts:
        raise ValueError("available_cohorts cannot be empty")

    # Get sizes for available cohorts only
    sizes = [cohort_sizes.get(g, 0) for g in available_cohorts]
    total_size = fsum(sizes)

    if total_size <= 0:
        raise ValueError(
            f"Total cohort size is {total_size}, cannot compute weights. "
            f"Available cohorts: {available_cohorts}, sizes: {sizes}"
        )

    weights = {g: cohort_sizes.get(g, 0) / total_size for g in available_cohorts}
    return weights


def _validate_weight_sum(
    weights: dict[int, float],
    tolerance: float = WEIGHT_SUM_TOLERANCE,
) -> tuple[bool, float]:
    """
    Validate that weights sum to 1.0 within tolerance.

    Parameters
    ----------
    weights : dict[int, float]
        Mapping from cohort to weight.
    tolerance : float, default=1e-6
        Acceptable deviation from 1.0.

    Returns
    -------
    tuple[bool, float]
        (is_valid, actual_sum) where is_valid indicates if sum is within tolerance.
    """
    weight_sum = fsum(weights.values())
    is_valid = abs(weight_sum - 1.0) <= tolerance
    return is_valid, weight_sum


def _select_degrees_of_freedom(
    cohort_dfs: dict[int, int | None],
    weights: dict[int, float],
    strategy: str,
    n_cohorts: int,
) -> int:
    """
    Select degrees of freedom for t-distribution inference.

    Parameters
    ----------
    cohort_dfs : dict[int, int | None]
        Mapping from cohort to degrees of freedom. None indicates unavailable.
    weights : dict[int, float]
        Cohort weights for weighted average strategy.
    strategy : {'conservative', 'weighted', 'fallback'}
        Selection strategy:
        - 'conservative': min(df_g) across cohorts (most conservative)
        - 'weighted': weighted average of df_g
        - 'fallback': n_cohorts - 1 (ignores individual df)
    n_cohorts : int
        Number of contributing cohorts.

    Returns
    -------
    int
        Selected degrees of freedom (minimum 1).

    Raises
    ------
    ValueError
        If strategy is invalid.

    See Also
    --------
    aggregate_to_event_time : Uses this function for df selection.

    Notes
    -----
    The 'conservative' strategy (default) uses the minimum df across cohorts,
    which produces wider confidence intervals but guards against under-coverage
    when cohort-specific df values vary substantially. The 'weighted' strategy
    may be appropriate when cohort df values are similar. The 'fallback' strategy
    ignores cohort-specific df and uses n_cohorts - 1, which may be suitable
    when cohort df values are unreliable or unavailable.
    """
    valid_strategies = {'conservative', 'weighted', 'fallback'}
    if strategy not in valid_strategies:
        raise ValueError(
            f"Invalid df_strategy '{strategy}'. Must be one of {valid_strategies}"
        )

    # Filter to valid (non-None, positive) df values
    valid_dfs = {g: df for g, df in cohort_dfs.items() if df is not None and df > 0}

    if not valid_dfs:
        # No valid df available, use fallback
        return max(1, n_cohorts - 1)

    if strategy == 'fallback':
        return max(1, n_cohorts - 1)

    if strategy == 'conservative':
        return max(1, min(valid_dfs.values()))

    if strategy == 'weighted':
        # Weighted average of df values
        weighted_sum = fsum(weights.get(g, 0) * df for g, df in valid_dfs.items())
        # Renormalize weights to valid cohorts only
        valid_weight_sum = fsum(weights.get(g, 0) for g in valid_dfs.keys())
        if valid_weight_sum > 0:
            weighted_df = weighted_sum / valid_weight_sum
            return max(1, int(round(weighted_df)))
        else:
            return max(1, n_cohorts - 1)

    # Should not reach here
    return max(1, n_cohorts - 1)


def aggregate_to_event_time(
    cohort_time_effects: pd.DataFrame | list,
    cohort_sizes: dict[int, int],
    alpha: float = 0.05,
    event_time_range: tuple[int, int] | None = None,
    df_strategy: str = 'conservative',
    verbose: bool = False,
) -> list[EventTimeEffect]:
    """
    Aggregate cohort-time ATT estimates to event-time weighted ATT (WATT).

    Computes WATT(r) = Σ_{g∈G_r} w(g,r) · ATT(g, g+r), where weights are
    proportional to cohort sizes. Uses t-distribution for proper inference.

    Parameters
    ----------
    cohort_time_effects : pd.DataFrame or list
        Cohort-time specific ATT estimates. DataFrame must contain columns:
        'cohort', 'period', 'att', 'se'. Optional: 'df_inference'.
        If list, must contain objects with these attributes.
    cohort_sizes : dict[int, int]
        Mapping from cohort identifier to number of treated units in that cohort.
        Used to compute weights w(g,r) = N_g / Σ N_g'.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    event_time_range : tuple[int, int] or None, optional
        If provided, only compute WATT for event times in [min, max].
        If None, compute for all available event times.
    df_strategy : {'conservative', 'weighted', 'fallback'}, default='conservative'
        Strategy for selecting degrees of freedom:
        - 'conservative': min(df_g) across contributing cohorts
        - 'weighted': weighted average of df_g
        - 'fallback': n_cohorts - 1 (ignores individual df)
    verbose : bool, default=False
        If True, log diagnostic information about aggregation.

    Returns
    -------
    list[EventTimeEffect]
        Event-time aggregated effects sorted by event_time.

    Raises
    ------
    ValueError
        If cohort_time_effects is empty or cohort_sizes contains invalid values.

    Warns
    -----
    UserWarning
        If weight sum deviates from 1.0, if cohorts are excluded due to
        missing data, or if df information is unavailable.

    See Also
    --------
    EventTimeEffect : Result container for event-time estimates.
    event_time_effects_to_dataframe : Convert results to DataFrame format.
    _compute_event_time_weights : Weight computation helper.
    _select_degrees_of_freedom : Degrees of freedom selection.

    Notes
    -----
    The weighted average treatment effect at event time r is:

    .. math::

        \\text{WATT}(r) = \\sum_{g \\in G_r} w(g, r) \\cdot
        \\widehat{\\tau}_{g, g+r}

    where :math:`G_r` is the set of cohorts observed at event time r, and
    weights :math:`w(g, r) = N_g / \\sum_{g' \\in G_r} N_{g'}` are proportional
    to cohort sizes. The standard error assumes independence across cohorts:

    .. math::

        \\text{SE}(\\text{WATT}(r)) = \\sqrt{\\sum_{g \\in G_r} w(g, r)^2
        \\cdot \\text{SE}(\\widehat{\\tau}_{g, g+r})^2}

    Confidence intervals use t-distribution rather than normal distribution,
    which provides proper inference for small samples.
    """
    from scipy.stats import t as t_dist

    # Convert to DataFrame if needed
    if isinstance(cohort_time_effects, list):
        if len(cohort_time_effects) == 0:
            raise ValueError("cohort_time_effects cannot be empty")
        # Convert list of objects to DataFrame
        df = pd.DataFrame([
            {
                'cohort': getattr(e, 'cohort', None),
                'period': getattr(e, 'period', None),
                'att': getattr(e, 'att', None),
                'se': getattr(e, 'se', None),
                'df_inference': getattr(e, 'df_inference', None),
            }
            for e in cohort_time_effects
        ])
    else:
        df = cohort_time_effects.copy()

    if df.empty:
        raise ValueError("cohort_time_effects cannot be empty")

    # Validate required columns
    required_cols = ['cohort', 'period', 'att', 'se']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Validate cohort_sizes
    if not cohort_sizes:
        raise ValueError("cohort_sizes cannot be empty")

    invalid_sizes = {g: n for g, n in cohort_sizes.items() if n <= 0}
    if invalid_sizes:
        raise ValueError(
            f"cohort_sizes contains invalid (non-positive) values: {invalid_sizes}"
        )

    # Compute event_time if not present
    if 'event_time' not in df.columns:
        df['event_time'] = df['period'] - df['cohort']

    # Filter by event_time_range if specified
    if event_time_range is not None:
        min_e, max_e = event_time_range
        df = df[(df['event_time'] >= min_e) & (df['event_time'] <= max_e)]
        if df.empty:
            warnings.warn(
                f"No effects found in event_time_range [{min_e}, {max_e}]",
                UserWarning,
                stacklevel=2
            )
            return []

    # Group by event_time
    event_times = sorted(df['event_time'].unique())
    results = []

    for r in event_times:
        event_df = df[df['event_time'] == r].copy()

        # Filter out rows with invalid ATT or SE
        valid_mask = event_df['att'].notna() & event_df['se'].notna()
        invalid_count = (~valid_mask).sum()
        if invalid_count > 0:
            excluded_cohorts = event_df.loc[~valid_mask, 'cohort'].tolist()
            if verbose:
                warnings.warn(
                    f"Event time {r}: {invalid_count} cohort(s) excluded due to "
                    f"missing ATT/SE: {excluded_cohorts}",
                    UserWarning,
                    stacklevel=2
                )
            event_df = event_df[valid_mask]

        if event_df.empty:
            # All cohorts invalid for this event time
            if verbose:
                warnings.warn(
                    f"Event time {r}: All cohorts have invalid data, returning NaN",
                    UserWarning,
                    stacklevel=2
                )
            results.append(EventTimeEffect(
                event_time=int(r),
                att=np.nan,
                se=np.nan,
                ci_lower=np.nan,
                ci_upper=np.nan,
                t_stat=np.nan,
                pvalue=np.nan,
                df_inference=0,
                n_cohorts=0,
                cohort_contributions={},
                weight_sum=0.0,
                alpha=alpha,
            ))
            continue

        # Get available cohorts for this event time
        available_cohorts = event_df['cohort'].astype(int).tolist()
        n_cohorts = len(available_cohorts)

        # Compute weights
        try:
            weights = _compute_event_time_weights(cohort_sizes, available_cohorts)
        except ValueError as e:
            warnings.warn(
                f"Event time {r}: Weight computation failed - {e}",
                UserWarning,
                stacklevel=2
            )
            results.append(EventTimeEffect(
                event_time=int(r),
                att=np.nan,
                se=np.nan,
                ci_lower=np.nan,
                ci_upper=np.nan,
                t_stat=np.nan,
                pvalue=np.nan,
                df_inference=0,
                n_cohorts=n_cohorts,
                cohort_contributions={},
                weight_sum=0.0,
                alpha=alpha,
            ))
            continue

        # Validate weight sum
        is_valid, weight_sum = _validate_weight_sum(weights)
        if not is_valid:
            warnings.warn(
                f"Event time {r}: Weight sum {weight_sum:.10f} deviates from 1.0",
                UserWarning,
                stacklevel=2
            )

        # Compute WATT(r) = Σ w(g,r) × ATT(g, g+r)
        cohort_contributions = {}
        weighted_att_terms = []
        weighted_var_terms = []

        for _, row in event_df.iterrows():
            g = int(row['cohort'])
            att_g = row['att']
            se_g = row['se']
            w_g = weights.get(g, 0)

            contribution = w_g * att_g
            cohort_contributions[g] = contribution
            weighted_att_terms.append(contribution)
            weighted_var_terms.append((w_g ** 2) * (se_g ** 2))

        # Use fsum for numerical stability
        watt = fsum(weighted_att_terms)
        variance = fsum(weighted_var_terms)
        se_watt = np.sqrt(variance) if variance > 0 else 0.0

        # Get df for each cohort
        cohort_dfs = {}
        if 'df_inference' in event_df.columns:
            for _, row in event_df.iterrows():
                g = int(row['cohort'])
                df_val = row.get('df_inference', None)
                if pd.notna(df_val):
                    cohort_dfs[g] = int(df_val)
                else:
                    cohort_dfs[g] = None
        else:
            # No df information available
            cohort_dfs = {g: None for g in available_cohorts}

        # Select df for inference
        df_inference = _select_degrees_of_freedom(
            cohort_dfs, weights, df_strategy, n_cohorts
        )

        if verbose and not cohort_dfs:
            warnings.warn(
                f"Event time {r}: No df_inference available, using fallback df={df_inference}",
                UserWarning,
                stacklevel=2
            )

        # Compute t-statistic
        if se_watt > 0:
            t_stat = watt / se_watt
        else:
            t_stat = np.inf if watt > 0 else (-np.inf if watt < 0 else np.nan)

        # Compute p-value using t-distribution
        if np.isfinite(t_stat) and df_inference > 0:
            pvalue = 2 * (1 - t_dist.cdf(abs(t_stat), df_inference))
        else:
            pvalue = 0.0 if np.isinf(t_stat) else np.nan

        # Compute CI using t-distribution
        if df_inference > 0:
            t_crit = t_dist.ppf(1 - alpha / 2, df_inference)
        else:
            # Fallback to z=1.96 if df is invalid (should not happen)
            t_crit = 1.96
            warnings.warn(
                f"Event time {r}: Invalid df_inference={df_inference}, using z=1.96",
                UserWarning,
                stacklevel=2
            )

        ci_lower = watt - t_crit * se_watt
        ci_upper = watt + t_crit * se_watt

        results.append(EventTimeEffect(
            event_time=int(r),
            att=watt,
            se=se_watt,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            t_stat=t_stat,
            pvalue=pvalue,
            df_inference=df_inference,
            n_cohorts=n_cohorts,
            cohort_contributions=cohort_contributions,
            weight_sum=weight_sum,
            alpha=alpha,
        ))

    # Sort by event_time
    results.sort(key=lambda x: x.event_time)

    if verbose:
        n_event_times = len(results)
        n_valid = sum(1 for e in results if not np.isnan(e.att))
        warnings.warn(
            f"aggregate_to_event_time: {n_valid}/{n_event_times} event times computed",
            UserWarning,
            stacklevel=2
        )

    return results


def event_time_effects_to_dataframe(results: list[EventTimeEffect]) -> pd.DataFrame:
    """
    Convert EventTimeEffect objects to a pandas DataFrame.

    Parameters
    ----------
    results : list[EventTimeEffect]
        Event-time effect estimates from aggregate_to_event_time.

    Returns
    -------
    pd.DataFrame
        Tabular representation with one row per event time.

    Raises
    ------
    TypeError
        If results is None.
    """
    if results is None:
        raise TypeError(
            "results cannot be None. Expected a list of EventTimeEffect objects. "
            "If no event-time effects were estimated, pass an empty list []."
        )

    if len(results) == 0:
        return pd.DataFrame({
            'event_time': pd.Series(dtype='int64'),
            'att': pd.Series(dtype='float64'),
            'se': pd.Series(dtype='float64'),
            'ci_lower': pd.Series(dtype='float64'),
            'ci_upper': pd.Series(dtype='float64'),
            't_stat': pd.Series(dtype='float64'),
            'pvalue': pd.Series(dtype='float64'),
            'df_inference': pd.Series(dtype='int64'),
            'n_cohorts': pd.Series(dtype='int64'),
            'weight_sum': pd.Series(dtype='float64'),
        })

    return pd.DataFrame([
        {
            'event_time': e.event_time,
            'att': e.att,
            'se': e.se,
            'ci_lower': e.ci_lower,
            'ci_upper': e.ci_upper,
            't_stat': e.t_stat,
            'pvalue': e.pvalue,
            'df_inference': e.df_inference,
            'n_cohorts': e.n_cohorts,
            'weight_sum': e.weight_sum,
        }
        for e in results
    ])

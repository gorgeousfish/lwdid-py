"""
Aggregation of repeated cross-sectional data to panel format.

This module provides functionality to aggregate lower-level repeated cross-sectional
data (e.g., individuals, counties) to the unit-by-period level (e.g., state-year)
for use with lwdid estimation methods.

The aggregation follows the methodology described in Lee & Wooldridge (2026),
Section 3, using the formula:

    Y_bar_st = sum_{i in (s,t)} w_ist * Y_ist,  where sum_{i in (s,t)} w_ist = 1

Key Functions
-------------
aggregate_to_panel
    Main function to aggregate repeated cross-sectional data to panel format.

Key Classes
-----------
AggregationResult
    Container for aggregation results and metadata.
CellStatistics
    Statistics for individual (unit, period) cells.

Notes
-----
When using repeated cross-sectional data with large numbers of observations within
each subgroup, it is common to collapse the data to a panel format. This involves
aggregating lower-level outcomes to the unit-by-period level when the treatment
or event of interest is assigned at the unit level.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from math import fsum
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..exceptions import (
    InvalidAggregationError,
    InsufficientCellSizeError,
    MissingRequiredColumnError,
)


# Tolerance for weight sum validation
WEIGHT_SUM_TOLERANCE = 1e-9


@dataclass
class CellStatistics:
    """
    Statistics for a single (unit, period) cell.

    Attributes
    ----------
    unit : Any
        Unit identifier value.
    period : Any
        Period identifier value (year, or tuple of year/quarter/month/week).
    n_obs : int
        Number of observations in the cell.
    outcome_mean : float
        Weighted mean of the outcome variable.
    outcome_variance : float or None
        Weighted variance of the outcome (None if not computed or n_obs == 1).
    effective_sample_size : float or None
        Effective sample size when survey weights are used.
    weight_type : {'equal', 'survey'}
        Type of weights used for aggregation.
    """
    unit: Any
    period: Any
    n_obs: int
    outcome_mean: float
    outcome_variance: float | None = None
    effective_sample_size: float | None = None
    weight_type: Literal['equal', 'survey'] = 'equal'


@dataclass
class AggregationResult:
    """
    Container for aggregation results and metadata.

    This class holds the aggregated panel data along with comprehensive
    metadata about the aggregation process, including cell statistics
    and configuration parameters.

    Attributes
    ----------
    panel_data : pd.DataFrame
        Aggregated panel data with one row per (unit, period) combination.
    n_original_obs : int
        Total number of observations in the original data.
    n_cells : int
        Number of (unit, period) cells in the output.
    n_units : int
        Number of unique units in the output.
    n_periods : int
        Number of unique periods in the output.
    cell_stats : pd.DataFrame
        DataFrame with statistics for each cell.
    min_cell_size : int
        Minimum cell size across all cells.
    max_cell_size : int
        Maximum cell size across all cells.
    mean_cell_size : float
        Mean cell size across all cells.
    median_cell_size : float
        Median cell size across all cells.
    unit_var : str
        Name of the unit variable column.
    time_var : str or list of str
        Name(s) of the time variable column(s).
    outcome_var : str
        Name of the outcome variable column.
    weight_var : str or None
        Name of the weight variable column (None if equal weights).
    frequency : str
        Aggregation frequency ('annual', 'quarterly', 'monthly', 'weekly').
    n_excluded_cells : int
        Number of cells excluded due to min_cell_size or all-NaN outcomes.
    excluded_cells_info : list of dict
        Information about excluded cells.
    """
    panel_data: pd.DataFrame
    n_original_obs: int
    n_cells: int
    n_units: int
    n_periods: int
    cell_stats: pd.DataFrame
    min_cell_size: int
    max_cell_size: int
    mean_cell_size: float
    median_cell_size: float
    unit_var: str
    time_var: str | list[str]
    outcome_var: str
    weight_var: str | None
    frequency: str
    n_excluded_cells: int = 0
    excluded_cells_info: list = field(default_factory=list)

    def summary(self) -> str:
        """
        Return formatted summary of aggregation.

        Returns
        -------
        str
            Multi-line string with aggregation statistics.

        Examples
        --------
        >>> result = aggregate_to_panel(data, 'state', 'year', 'income')
        >>> print(result.summary())
        Aggregation Summary
        ===================
        Original observations: 10000
        Output cells: 150
        Units: 50
        Periods: 3
        ...
        """
        lines = [
            "Aggregation Summary",
            "===================",
            f"Original observations: {self.n_original_obs:,}",
            f"Output cells: {self.n_cells:,}",
            f"Units: {self.n_units:,}",
            f"Periods: {self.n_periods:,}",
            "",
            "Cell Size Statistics",
            "--------------------",
            f"Minimum: {self.min_cell_size:,}",
            f"Maximum: {self.max_cell_size:,}",
            f"Mean: {self.mean_cell_size:.2f}",
            f"Median: {self.median_cell_size:.2f}",
            "",
            "Configuration",
            "-------------",
            f"Unit variable: {self.unit_var}",
            f"Time variable: {self.time_var}",
            f"Outcome variable: {self.outcome_var}",
            f"Weight variable: {self.weight_var or 'None (equal weights)'}",
            f"Frequency: {self.frequency}",
        ]
        if self.n_excluded_cells > 0:
            lines.extend([
                "",
                f"Excluded cells: {self.n_excluded_cells}",
            ])
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """
        Return aggregation parameters as dictionary.

        Returns
        -------
        dict
            Dictionary containing all aggregation parameters and statistics.

        Examples
        --------
        >>> result = aggregate_to_panel(data, 'state', 'year', 'income')
        >>> params = result.to_dict()
        >>> params['n_units']
        50
        """
        return {
            'n_original_obs': self.n_original_obs,
            'n_cells': self.n_cells,
            'n_units': self.n_units,
            'n_periods': self.n_periods,
            'min_cell_size': self.min_cell_size,
            'max_cell_size': self.max_cell_size,
            'mean_cell_size': self.mean_cell_size,
            'median_cell_size': self.median_cell_size,
            'unit_var': self.unit_var,
            'time_var': self.time_var,
            'outcome_var': self.outcome_var,
            'weight_var': self.weight_var,
            'frequency': self.frequency,
            'n_excluded_cells': self.n_excluded_cells,
        }

    def to_csv(self, path: str, include_metadata: bool = True) -> None:
        """
        Export panel data to CSV with optional metadata header.

        Parameters
        ----------
        path : str
            Output file path.
        include_metadata : bool, default=True
            Whether to include aggregation metadata as header comments.

        Examples
        --------
        >>> result = aggregate_to_panel(data, 'state', 'year', 'income')
        >>> result.to_csv('aggregated_panel.csv')
        """
        if include_metadata:
            with open(path, 'w') as f:
                f.write("# Aggregation Metadata\n")
                f.write(f"# Original observations: {self.n_original_obs}\n")
                f.write(f"# Output cells: {self.n_cells}\n")
                f.write(f"# Units: {self.n_units}\n")
                f.write(f"# Periods: {self.n_periods}\n")
                f.write(f"# Unit variable: {self.unit_var}\n")
                f.write(f"# Time variable: {self.time_var}\n")
                f.write(f"# Outcome variable: {self.outcome_var}\n")
                f.write(f"# Weight variable: {self.weight_var}\n")
                f.write(f"# Frequency: {self.frequency}\n")
                f.write("#\n")
            self.panel_data.to_csv(path, mode='a', index=False)
        else:
            self.panel_data.to_csv(path, index=False)


# =============================================================================
# Validation Functions
# =============================================================================

def _validate_aggregation_inputs(
    data: pd.DataFrame,
    unit_var: str,
    time_var: str | list[str],
    outcome_var: str,
    weight_var: str | None,
    controls: list[str] | None,
) -> None:
    """
    Validate input data and column existence.

    Parameters
    ----------
    data : pd.DataFrame
        Input repeated cross-sectional data.
    unit_var : str
        Column name for aggregation unit.
    time_var : str or list of str
        Time variable column name(s).
    outcome_var : str
        Outcome variable column name.
    weight_var : str or None
        Weight variable column name.
    controls : list of str or None
        Control variable column names.

    Raises
    ------
    TypeError
        If data is not a pandas DataFrame.
    ValueError
        If data is empty.
    MissingRequiredColumnError
        If required columns are missing.
    ValueError
        If outcome variable is not numeric.
    """
    # Check DataFrame type
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            f"Input data must be a pandas DataFrame. Got: {type(data).__name__}"
        )

    # Check empty data
    if len(data) == 0:
        raise ValueError("Input data is empty")

    # Build list of required columns
    required_cols = [unit_var, outcome_var]
    if isinstance(time_var, str):
        required_cols.append(time_var)
    else:
        required_cols.extend(time_var)

    if weight_var is not None:
        required_cols.append(weight_var)

    if controls:
        required_cols.extend(controls)

    # Check for missing columns
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise MissingRequiredColumnError(
            f"Required column(s) not found in data: {missing_cols}. "
            f"Available columns: {list(data.columns)}"
        )

    # Check outcome is numeric
    if not pd.api.types.is_numeric_dtype(data[outcome_var]):
        raise ValueError(
            f"Outcome variable '{outcome_var}' must be numeric type. "
            f"Found dtype: '{data[outcome_var].dtype}'"
        )


def _validate_weights(
    data: pd.DataFrame,
    weight_var: str,
) -> tuple[pd.DataFrame, int]:
    """
    Validate weights are non-negative and handle missing values.

    Parameters
    ----------
    data : pd.DataFrame
        Input data containing weight column.
    weight_var : str
        Weight variable column name.

    Returns
    -------
    data_clean : pd.DataFrame
        Data with missing weight rows removed.
    n_missing : int
        Number of rows with missing weights that were removed.

    Raises
    ------
    ValueError
        If any weight is negative.
    """
    weights = data[weight_var]

    # Check for negative weights
    negative_mask = weights < 0
    if negative_mask.any():
        n_negative = negative_mask.sum()
        negative_examples = data.loc[negative_mask, weight_var].head(5).tolist()
        raise ValueError(
            f"Weights must be non-negative. Found {n_negative} negative values. "
            f"Examples: {negative_examples}"
        )

    # Handle missing weights
    missing_mask = weights.isna()
    n_missing = missing_mask.sum()

    if n_missing > 0:
        warnings.warn(
            f"Excluded {n_missing} observations with missing weights.",
            UserWarning,
            stacklevel=3
        )
        data_clean = data[~missing_mask].copy()
    else:
        data_clean = data.copy()

    return data_clean, n_missing


def _validate_treatment_consistency(
    data: pd.DataFrame,
    unit_var: str,
    time_var: str | list[str],
    treatment_var: str,
) -> None:
    """
    Verify treatment status is constant within each cell.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    unit_var : str
        Column name for aggregation unit.
    time_var : str or list of str
        Time variable column name(s).
    treatment_var : str
        Treatment indicator column name.

    Raises
    ------
    InvalidAggregationError
        If treatment status varies within any cell.
    """
    # Build groupby columns
    if isinstance(time_var, str):
        group_cols = [unit_var, time_var]
    else:
        group_cols = [unit_var] + list(time_var)

    # Check within-cell variation
    within_cell_std = data.groupby(group_cols)[treatment_var].std()
    varying_cells = within_cell_std[within_cell_std > 1e-10]

    if len(varying_cells) > 0:
        # Get examples of varying cells
        examples = varying_cells.head(3).index.tolist()
        raise InvalidAggregationError(
            f"Treatment status varies within {len(varying_cells)} cell(s). "
            f"Treatment must be constant within each (unit, period) cell. "
            f"Examples of cells with varying treatment: {examples}"
        )


def _validate_gvar_consistency(
    data: pd.DataFrame,
    unit_var: str,
    gvar: str,
) -> None:
    """
    Verify treatment timing (gvar) is constant within each unit.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    unit_var : str
        Column name for aggregation unit.
    gvar : str
        Treatment timing variable column name.

    Raises
    ------
    InvalidAggregationError
        If gvar varies within any unit.
    """
    # Check within-unit variation
    within_unit_std = data.groupby(unit_var)[gvar].std()
    varying_units = within_unit_std[within_unit_std > 1e-10]

    if len(varying_units) > 0:
        # Get examples of varying units
        examples = varying_units.head(3).index.tolist()
        raise InvalidAggregationError(
            f"Treatment timing (gvar) varies within {len(varying_units)} unit(s). "
            f"gvar must be constant within each unit across all periods. "
            f"Examples of units with varying gvar: {examples}"
        )


# =============================================================================
# Core Aggregation Functions
# =============================================================================

def _normalize_weights(weights: pd.Series) -> pd.Series:
    """
    Normalize weights to sum to 1 within the series.

    Uses math.fsum() for numerically stable summation.

    Parameters
    ----------
    weights : pd.Series
        Raw weights (must be non-negative).

    Returns
    -------
    pd.Series
        Normalized weights summing to 1.

    Notes
    -----
    If all weights are zero, returns equal weights (1/n).
    """
    weight_sum = fsum(weights.values)

    if weight_sum == 0 or np.isclose(weight_sum, 0, atol=1e-15):
        # All weights are zero, use equal weights
        n = len(weights)
        return pd.Series(1.0 / n, index=weights.index)

    return weights / weight_sum


def _compute_cell_weighted_mean(
    cell_data: pd.DataFrame,
    outcome_var: str,
    weight_var: str | None,
    compute_variance: bool = False,
) -> tuple[float, float | None, float | None, int]:
    """
    Compute weighted mean and optionally variance for a single cell.

    Formula: Y_bar = sum(w_i * Y_i) where sum(w_i) = 1

    Parameters
    ----------
    cell_data : pd.DataFrame
        Data for a single (unit, period) cell.
    outcome_var : str
        Outcome variable column name.
    weight_var : str or None
        Weight variable column name (None for equal weights).
    compute_variance : bool, default=False
        Whether to compute weighted variance.

    Returns
    -------
    mean : float
        Weighted mean of the outcome.
    variance : float or None
        Weighted variance (None if not computed or n == 1).
    ess : float or None
        Effective sample size (None if equal weights).
    n_obs : int
        Number of observations in the cell.
    """
    # Get outcome values, excluding NaN
    outcomes = cell_data[outcome_var].dropna()
    n_obs = len(outcomes)

    if n_obs == 0:
        return np.nan, None, None, 0

    # Get or create weights
    if weight_var is not None:
        # Use survey weights, align with non-NaN outcomes
        raw_weights = cell_data.loc[outcomes.index, weight_var]
        weights = _normalize_weights(raw_weights)
        weight_type = 'survey'
    else:
        # Equal weights
        weights = pd.Series(1.0 / n_obs, index=outcomes.index)
        weight_type = 'equal'

    # Compute weighted mean using fsum for numerical stability
    weighted_products = weights.values * outcomes.values
    mean = fsum(weighted_products)

    # Compute variance if requested
    variance = None
    if compute_variance and n_obs > 1:
        # Weighted variance: sum(w_i * (Y_i - Y_bar)^2)
        deviations_sq = (outcomes.values - mean) ** 2
        weighted_deviations = weights.values * deviations_sq
        variance = fsum(weighted_deviations)

    # Compute effective sample size for survey weights
    ess = None
    if weight_var is not None:
        ess = _compute_effective_sample_size(raw_weights)

    return mean, variance, ess, n_obs


def _compute_effective_sample_size(weights: pd.Series) -> float:
    """
    Compute effective sample size for survey weights.

    ESS = (sum(w_i))^2 / sum(w_i^2)

    Parameters
    ----------
    weights : pd.Series
        Raw (unnormalized) weights.

    Returns
    -------
    float
        Effective sample size.
    """
    weight_sum = fsum(weights.values)
    weight_sq_sum = fsum(weights.values ** 2)

    if weight_sq_sum == 0 or np.isclose(weight_sq_sum, 0, atol=1e-15):
        return 0.0

    return (weight_sum ** 2) / weight_sq_sum


# =============================================================================
# Main Aggregation Function
# =============================================================================

def aggregate_to_panel(
    data: pd.DataFrame,
    unit_var: str,
    time_var: str | list[str],
    outcome_var: str,
    *,
    weight_var: str | None = None,
    controls: list[str] | None = None,
    treatment_var: str | None = None,
    gvar: str | None = None,
    frequency: Literal['annual', 'quarterly', 'monthly', 'weekly'] = 'annual',
    min_cell_size: int = 1,
    compute_variance: bool = False,
) -> AggregationResult:
    """
    Aggregate repeated cross-sectional data to panel format.

    This function aggregates lower-level repeated cross-sectional data
    (e.g., individuals, counties) to the unit-by-period level (e.g., state-year)
    using weighted means. The aggregation follows Lee & Wooldridge (2026), Section 3.

    Formula: Y_bar_st = sum_{i in (s,t)} w_ist * Y_ist, where sum w_ist = 1

    Parameters
    ----------
    data : pd.DataFrame
        Repeated cross-sectional data in long format.
    unit_var : str
        Column name for aggregation unit (e.g., 'state').
    time_var : str or list of str
        Time variable(s). Single string for annual data,
        list of [year, quarter/month/week] for high-frequency data.
    outcome_var : str
        Outcome variable column name.
    weight_var : str, optional
        Survey weight column name. If None, uses equal weights (1/n_st).
    controls : list of str, optional
        Control variable column names to aggregate.
    treatment_var : str, optional
        Treatment indicator column name. Must be constant within each cell.
    gvar : str, optional
        Treatment timing variable. Must be constant within each unit.
    frequency : {'annual', 'quarterly', 'monthly', 'weekly'}, default='annual'
        Aggregation frequency.
    min_cell_size : int, default=1
        Minimum observations per cell. Cells below threshold are excluded.
    compute_variance : bool, default=False
        Whether to compute within-cell variance estimates.

    Returns
    -------
    AggregationResult
        Container with aggregated panel data and metadata.

    Raises
    ------
    TypeError
        If data is not a pandas DataFrame.
    ValueError
        If input data is empty or outcome is not numeric.
    MissingRequiredColumnError
        If required columns are missing.
    InvalidAggregationError
        If treatment varies within cells or gvar varies within units.
    InsufficientCellSizeError
        If all cells are below min_cell_size threshold.

    Examples
    --------
    >>> import pandas as pd
    >>> from lwdid.preprocessing import aggregate_to_panel
    >>> # Create sample repeated cross-section data
    >>> data = pd.DataFrame({
    ...     'state': ['CA', 'CA', 'CA', 'TX', 'TX', 'TX'],
    ...     'year': [2000, 2000, 2001, 2000, 2001, 2001],
    ...     'income': [50000, 55000, 60000, 45000, 48000, 52000],
    ...     'weight': [1.0, 1.2, 0.8, 1.0, 1.1, 0.9],
    ... })
    >>> result = aggregate_to_panel(
    ...     data, 'state', 'year', 'income', weight_var='weight'
    ... )
    >>> print(result.panel_data)
    """
    # Step 1: Validate inputs
    _validate_aggregation_inputs(
        data, unit_var, time_var, outcome_var, weight_var, controls
    )

    # Step 2: Handle weights
    data_work = data.copy()
    n_original = len(data_work)

    if weight_var is not None:
        data_work, n_missing_weights = _validate_weights(data_work, weight_var)
    else:
        n_missing_weights = 0

    # Step 3: Validate treatment consistency if specified
    if treatment_var is not None:
        _validate_treatment_consistency(data_work, unit_var, time_var, treatment_var)

    # Step 4: Validate gvar consistency if specified
    if gvar is not None:
        _validate_gvar_consistency(data_work, unit_var, gvar)

    # Step 5: Build groupby columns based on frequency
    group_cols = _build_group_columns(unit_var, time_var, frequency)

    # Step 6: Perform aggregation
    aggregated_rows = []
    cell_stats_list = []
    excluded_cells = []

    for group_key, group_data in data_work.groupby(group_cols, dropna=False):
        # Compute weighted mean for outcome
        mean, variance, ess, n_obs = _compute_cell_weighted_mean(
            group_data, outcome_var, weight_var, compute_variance
        )

        # Check cell size
        if n_obs < min_cell_size:
            excluded_cells.append({
                'cell': group_key,
                'n_obs': n_obs,
                'reason': f'below min_cell_size ({min_cell_size})'
            })
            continue

        # Check for all-NaN outcome
        if np.isnan(mean):
            excluded_cells.append({
                'cell': group_key,
                'n_obs': n_obs,
                'reason': 'all-NaN outcome'
            })
            continue

        # Build row dictionary
        row = _build_aggregated_row(
            group_key, group_cols, mean, n_obs,
            group_data, outcome_var, weight_var, controls,
            treatment_var, gvar, variance, ess
        )
        aggregated_rows.append(row)

        # Build cell statistics
        cell_stat = CellStatistics(
            unit=group_key[0] if isinstance(group_key, tuple) else group_key,
            period=group_key[1:] if isinstance(group_key, tuple) and len(group_key) > 2 else (
                group_key[1] if isinstance(group_key, tuple) else None
            ),
            n_obs=n_obs,
            outcome_mean=mean,
            outcome_variance=variance,
            effective_sample_size=ess,
            weight_type='survey' if weight_var else 'equal',
        )
        cell_stats_list.append(cell_stat)

    # Step 7: Handle case where all cells are excluded
    if len(aggregated_rows) == 0:
        raise InsufficientCellSizeError(
            f"All cells have fewer than {min_cell_size} observations or all-NaN outcomes. "
            f"Total cells attempted: {len(excluded_cells)}. "
            f"Consider reducing min_cell_size parameter."
        )

    # Step 8: Issue warnings for excluded cells
    if len(excluded_cells) > 0:
        warnings.warn(
            f"Excluded {len(excluded_cells)} cells: "
            f"{sum(1 for c in excluded_cells if 'min_cell_size' in c['reason'])} below min_cell_size, "
            f"{sum(1 for c in excluded_cells if 'NaN' in c['reason'])} with all-NaN outcomes.",
            UserWarning,
            stacklevel=2
        )

    # Step 9: Build output DataFrame
    panel_data = pd.DataFrame(aggregated_rows)

    # Step 10: Build cell statistics DataFrame
    cell_stats_df = pd.DataFrame([
        {
            'unit': cs.unit,
            'period': cs.period,
            'n_obs': cs.n_obs,
            'outcome_mean': cs.outcome_mean,
            'outcome_variance': cs.outcome_variance,
            'effective_sample_size': cs.effective_sample_size,
            'weight_type': cs.weight_type,
        }
        for cs in cell_stats_list
    ])

    # Step 11: Compute summary statistics
    cell_sizes = [cs.n_obs for cs in cell_stats_list]
    n_units = panel_data[unit_var].nunique()
    n_periods = _count_periods(panel_data, time_var, frequency)

    # Step 12: Issue warning if fewer than 3 units
    if n_units < 3:
        warnings.warn(
            f"Aggregation resulted in {n_units} units. "
            f"lwdid requires at least 3 units for valid estimation.",
            UserWarning,
            stacklevel=2
        )

    # Step 13: Build and return result
    return AggregationResult(
        panel_data=panel_data,
        n_original_obs=n_original,
        n_cells=len(aggregated_rows),
        n_units=n_units,
        n_periods=n_periods,
        cell_stats=cell_stats_df,
        min_cell_size=min(cell_sizes),
        max_cell_size=max(cell_sizes),
        mean_cell_size=np.mean(cell_sizes),
        median_cell_size=np.median(cell_sizes),
        unit_var=unit_var,
        time_var=time_var,
        outcome_var=outcome_var,
        weight_var=weight_var,
        frequency=frequency,
        n_excluded_cells=len(excluded_cells),
        excluded_cells_info=excluded_cells,
    )


def _build_group_columns(
    unit_var: str,
    time_var: str | list[str],
    frequency: str,
) -> list[str]:
    """
    Build list of columns for groupby based on frequency.

    Parameters
    ----------
    unit_var : str
        Unit variable column name.
    time_var : str or list of str
        Time variable column name(s).
    frequency : str
        Aggregation frequency.

    Returns
    -------
    list of str
        Column names for groupby operation.
    """
    if isinstance(time_var, str):
        return [unit_var, time_var]
    else:
        # For high-frequency data, time_var is [year, quarter/month/week]
        return [unit_var] + list(time_var)


def _build_aggregated_row(
    group_key: tuple | Any,
    group_cols: list[str],
    mean: float,
    n_obs: int,
    group_data: pd.DataFrame,
    outcome_var: str,
    weight_var: str | None,
    controls: list[str] | None,
    treatment_var: str | None,
    gvar: str | None,
    variance: float | None,
    ess: float | None,
) -> dict:
    """
    Build a single row for the aggregated panel DataFrame.

    Parameters
    ----------
    group_key : tuple or Any
        Group key from groupby.
    group_cols : list of str
        Column names used for grouping.
    mean : float
        Weighted mean of outcome.
    n_obs : int
        Number of observations in cell.
    group_data : pd.DataFrame
        Data for this cell.
    outcome_var : str
        Outcome variable name.
    weight_var : str or None
        Weight variable name.
    controls : list of str or None
        Control variable names.
    treatment_var : str or None
        Treatment variable name.
    gvar : str or None
        Treatment timing variable name.
    variance : float or None
        Weighted variance.
    ess : float or None
        Effective sample size.

    Returns
    -------
    dict
        Row dictionary for DataFrame construction.
    """
    row = {}

    # Add group columns
    if isinstance(group_key, tuple):
        for col, val in zip(group_cols, group_key):
            row[col] = val
    else:
        row[group_cols[0]] = group_key

    # Add outcome mean
    row[outcome_var] = mean

    # Add cell size
    row['_n_obs'] = n_obs

    # Add variance if computed
    if variance is not None:
        row[f'{outcome_var}_var'] = variance

    # Add ESS if computed
    if ess is not None:
        row['_ess'] = ess

    # Aggregate control variables
    if controls:
        for ctrl in controls:
            ctrl_mean = _compute_control_weighted_mean(
                group_data, ctrl, weight_var
            )
            row[ctrl] = ctrl_mean

    # Add treatment variable (should be constant within cell)
    if treatment_var is not None:
        row[treatment_var] = group_data[treatment_var].iloc[0]

    # Add gvar (should be constant within unit)
    if gvar is not None:
        row[gvar] = group_data[gvar].iloc[0]

    return row


def _compute_control_weighted_mean(
    cell_data: pd.DataFrame,
    control_var: str,
    weight_var: str | None,
) -> float:
    """
    Compute weighted mean for a control variable.

    Parameters
    ----------
    cell_data : pd.DataFrame
        Data for a single cell.
    control_var : str
        Control variable column name.
    weight_var : str or None
        Weight variable column name.

    Returns
    -------
    float
        Weighted mean of the control variable.
    """
    # Get control values, excluding NaN
    values = cell_data[control_var].dropna()
    n_obs = len(values)

    if n_obs == 0:
        return np.nan

    # Get or create weights
    if weight_var is not None:
        raw_weights = cell_data.loc[values.index, weight_var]
        weights = _normalize_weights(raw_weights)
    else:
        weights = pd.Series(1.0 / n_obs, index=values.index)

    # Compute weighted mean
    weighted_products = weights.values * values.values
    return fsum(weighted_products)


def _count_periods(
    panel_data: pd.DataFrame,
    time_var: str | list[str],
    frequency: str,
) -> int:
    """
    Count unique periods in the panel data.

    Parameters
    ----------
    panel_data : pd.DataFrame
        Aggregated panel data.
    time_var : str or list of str
        Time variable column name(s).
    frequency : str
        Aggregation frequency.

    Returns
    -------
    int
        Number of unique periods.
    """
    if isinstance(time_var, str):
        return panel_data[time_var].nunique()
    else:
        # For high-frequency data, count unique combinations
        return panel_data[list(time_var)].drop_duplicates().shape[0]

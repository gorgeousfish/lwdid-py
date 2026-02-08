"""
Selection mechanism diagnostics for unbalanced panel data.

This module provides diagnostic tools for assessing potential selection bias
in unbalanced panel data for difference-in-differences estimation.

The key assumption is that selection (missing data) may depend on unobserved
time-invariant heterogeneity, but cannot systematically depend on outcome
shocks in the untreated state. This is analogous to the standard fixed effects
assumption and is removed by the rolling transformation.

Main Functions
--------------
diagnose_selection_mechanism : Comprehensive selection mechanism diagnostics.
get_unit_missing_stats : Per-unit missing data statistics.
plot_missing_pattern : Visualize missing data patterns.

Data Classes
------------
SelectionDiagnostics : Complete diagnostic results.
BalanceStatistics : Panel balance metrics.
AttritionAnalysis : Attrition pattern analysis.
UnitMissingStats : Per-unit statistics.
SelectionTestResult : Statistical test results.

Enums
-----
MissingPattern : Missing data pattern classification (MCAR, MAR, MNAR).
SelectionRisk : Selection bias risk level (LOW, MEDIUM, HIGH).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# Enums
# =============================================================================

class MissingPattern(Enum):
    """
    Missing data pattern classification based on Rubin's taxonomy.
    
    Attributes
    ----------
    MCAR : str
        Missing Completely At Random - missingness is independent of all data,
        both observed and unobserved. This is the most benign pattern.
    MAR : str
        Missing At Random - missingness depends only on observed data.
        Acceptable under the selection mechanism assumption when controls
        are included.
    MNAR : str
        Missing Not At Random - missingness depends on unobserved data.
        This may violate the selection mechanism assumption if missingness
        depends on outcome shocks in the untreated state.
    UNKNOWN : str
        Pattern could not be determined with available data.
    
    Notes
    -----
    The selection mechanism assumption requires that missingness may depend on
    unobserved time-invariant heterogeneity, but cannot systematically depend
    on time-varying outcome shocks.
    
    MCAR and MAR patterns are generally acceptable. MNAR patterns may be
    acceptable if missingness depends only on time-invariant factors (which
    are removed by the rolling transformation), but problematic if missingness
    depends on time-varying outcome shocks.
    """
    MCAR = "missing_completely_at_random"
    MAR = "missing_at_random"
    MNAR = "missing_not_at_random"
    UNKNOWN = "unknown"


class SelectionRisk(Enum):
    """
    Risk level for selection bias in ATT estimation.
    
    Attributes
    ----------
    LOW : str
        Low risk - selection mechanism assumption likely holds.
        Proceed with estimation.
    MEDIUM : str
        Medium risk - some indicators suggest potential issues.
        Consider using detrending and sensitivity analysis.
    HIGH : str
        High risk - strong evidence of problematic selection.
        Results should be interpreted with caution.
    UNKNOWN : str
        Risk could not be assessed with available data.
    
    Notes
    -----
    Risk assessment is based on multiple factors:
    
    - Missing data pattern (MCAR < MAR < MNAR)
    - Attrition rate (lower is better)
    - Differential attrition before/after treatment
    - Panel balance ratio
    
    The rolling transformation removes unit-specific averages, so selection
    is allowed to depend on unobserved time-constant heterogeneity, similar
    to the standard fixed effects assumption.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class AttritionAnalysis:
    """
    Analysis of unit dropout patterns in panel data.
    
    Attributes
    ----------
    n_units_complete : int
        Number of units with complete observations across all periods.
    n_units_partial : int
        Number of units with at least one missing period.
    attrition_rate : float
        Proportion of units with incomplete observations (n_partial / n_total).
    attrition_by_cohort : dict[int, float]
        Attrition rate by treatment cohort. Keys are cohort identifiers,
        values are attrition rates within each cohort.
    attrition_by_period : dict[int, float]
        Cumulative attrition rate by time period. Shows the proportion of
        units not observed at each time point.
    early_dropout_rate : float
        Rate of units that exit before the final period (last_obs < T_max).
    late_entry_rate : float
        Rate of units that enter after the first period (first_obs > T_min).
    dropout_before_treatment : int
        Number of treated units that dropout before their treatment period.
        High values may indicate anticipation effects.
    dropout_after_treatment : int
        Number of treated units that dropout after treatment starts.
        High values may indicate treatment-induced attrition.
    
    Notes
    -----
    Differential attrition patterns (e.g., more dropout after treatment than
    before) may indicate selection related to treatment effects, which would
    violate the selection mechanism assumption.
    """
    n_units_complete: int
    n_units_partial: int
    attrition_rate: float
    attrition_by_cohort: dict[int, float] = field(default_factory=dict)
    attrition_by_period: dict[int, float] = field(default_factory=dict)
    early_dropout_rate: float = 0.0
    late_entry_rate: float = 0.0
    dropout_before_treatment: int = 0
    dropout_after_treatment: int = 0


@dataclass
class BalanceStatistics:
    """
    Panel balance statistics.
    
    Attributes
    ----------
    is_balanced : bool
        True if all units have the same number of observations.
    n_units : int
        Total number of unique units in the panel.
    n_periods : int
        Total number of unique time periods in the panel.
    min_obs_per_unit : int
        Minimum observations across all units.
    max_obs_per_unit : int
        Maximum observations across all units.
    mean_obs_per_unit : float
        Average observations per unit.
    std_obs_per_unit : float
        Standard deviation of observations per unit.
    balance_ratio : float
        Ratio of min to max observations (1.0 = perfectly balanced).
        Lower values indicate more severe imbalance.
    units_below_demean_threshold : int
        Number of treated units with < 1 pre-treatment observation.
        These units cannot be used with demeaning.
    units_below_detrend_threshold : int
        Number of treated units with < 2 pre-treatment observations.
        These units cannot be used with detrending.
    pct_usable_demean : float
        Percentage of treated units usable for demeaning (0-100).
    pct_usable_detrend : float
        Percentage of treated units usable for detrending (0-100).
    
    Notes
    -----
    For treatment cohort g in period r, the transformed outcome can only be
    computed if there are enough observed pre-treatment periods (t < g):
    
    - Demeaning requires at least one pre-treatment period to compute the mean.
    - Detrending requires at least two pre-treatment periods to estimate a
      linear trend.
    
    Units with insufficient pre-treatment observations are excluded from the
    corresponding transformation method.
    """
    is_balanced: bool
    n_units: int
    n_periods: int
    min_obs_per_unit: int
    max_obs_per_unit: int
    mean_obs_per_unit: float
    std_obs_per_unit: float
    balance_ratio: float
    units_below_demean_threshold: int = 0
    units_below_detrend_threshold: int = 0
    pct_usable_demean: float = 100.0
    pct_usable_detrend: float = 100.0


@dataclass
class SelectionTestResult:
    """
    Result of a statistical test for selection mechanism.
    
    Attributes
    ----------
    test_name : str
        Name of the statistical test performed.
    statistic : float
        Test statistic value.
    pvalue : float
        P-value of the test.
    reject_null : bool
        Whether to reject the null hypothesis at alpha=0.05.
    interpretation : str
        Human-readable interpretation of the test result.
    details : dict[str, Any]
        Additional test-specific details (e.g., means, correlations).
    
    Notes
    -----
    Common tests include:
    
    - Little's MCAR Test: Tests if data is missing completely at random
    - Selection on Observables: Tests if missingness depends on controls
    - Lagged Outcome Test: Tests if missingness depends on past outcomes
    """
    test_name: str
    statistic: float
    pvalue: float
    reject_null: bool
    interpretation: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class UnitMissingStats:
    """
    Missing data statistics for a single unit.
    
    Attributes
    ----------
    unit_id : Any
        Unit identifier.
    cohort : int | None
        Treatment cohort (None for never-treated units).
    is_treated : bool
        Whether the unit is ever treated.
    n_total_periods : int
        Total periods in the panel.
    n_observed : int
        Number of observed periods for this unit.
    n_missing : int
        Number of missing periods for this unit.
    missing_rate : float
        Proportion of missing periods (n_missing / n_total_periods).
    first_observed : int
        First period with observation.
    last_observed : int
        Last period with observation.
    observation_span : int
        Span from first to last observation (last - first + 1).
    n_pre_treatment : int | None
        Pre-treatment observations (treated units only).
    n_post_treatment : int | None
        Post-treatment observations (treated units only).
    pre_treatment_missing_rate : float | None
        Missing rate in pre-treatment period.
    post_treatment_missing_rate : float | None
        Missing rate in post-treatment period.
    can_use_demean : bool
        Whether unit has sufficient data for demeaning (≥1 pre-treatment obs).
    can_use_detrend : bool
        Whether unit has sufficient data for detrending (≥2 pre-treatment obs).
    reason_if_excluded : str | None
        Reason for exclusion if unit cannot be used.
    """
    unit_id: Any
    cohort: int | None
    is_treated: bool
    n_total_periods: int
    n_observed: int
    n_missing: int
    missing_rate: float
    first_observed: int
    last_observed: int
    observation_span: int
    n_pre_treatment: int | None = None
    n_post_treatment: int | None = None
    pre_treatment_missing_rate: float | None = None
    post_treatment_missing_rate: float | None = None
    can_use_demean: bool = True
    can_use_detrend: bool = True
    reason_if_excluded: str | None = None


@dataclass
class SelectionDiagnostics:
    """
    Complete selection mechanism diagnostics for unbalanced panels.
    
    This class aggregates all diagnostic information about missing data
    patterns and potential selection bias in panel data for DiD estimation.
    
    Attributes
    ----------
    missing_pattern : MissingPattern
        Classified missing data pattern (MCAR, MAR, MNAR, UNKNOWN).
    missing_pattern_confidence : float
        Confidence level (0-1) in the pattern classification.
    selection_risk : SelectionRisk
        Assessed risk level for selection bias.
    attrition_analysis : AttritionAnalysis
        Detailed attrition pattern analysis.
    balance_statistics : BalanceStatistics
        Panel balance statistics.
    recommendations : list[str]
        Actionable recommendations based on diagnostics.
    warnings : list[str]
        Warning messages about potential issues.
    missing_rate_overall : float
        Overall missing rate across all unit-periods.
    missing_rate_by_period : dict[int, float]
        Missing rate by time period.
    missing_rate_by_cohort : dict[int, float]
        Missing rate by treatment cohort.
    selection_tests : List[SelectionTestResult]
        Results of statistical tests for selection.
    unit_stats : List[UnitMissingStats]
        Per-unit missing data statistics.
    
    Notes
    -----
    The selection mechanism assumption requires that selection may depend on
    unobserved time-invariant heterogeneity, but cannot systematically depend
    on time-varying outcome shocks.
    
    This is analogous to the standard fixed effects assumption. The rolling
    transformation removes unit-specific averages (or trends), which eliminates
    bias from selection on time-invariant factors.
    
    See Also
    --------
    diagnose_selection_mechanism : Function to create this diagnostics object.
    """
    missing_pattern: MissingPattern
    missing_pattern_confidence: float
    selection_risk: SelectionRisk
    attrition_analysis: AttritionAnalysis
    balance_statistics: BalanceStatistics
    recommendations: list[str]
    warnings: list[str]
    missing_rate_overall: float
    missing_rate_by_period: dict[int, float]
    missing_rate_by_cohort: dict[int, float]
    selection_tests: List[SelectionTestResult] = field(default_factory=list)
    unit_stats: List[UnitMissingStats] = field(default_factory=list)
    
    def summary(self) -> str:
        """
        Generate a human-readable summary of diagnostics.
        
        Returns
        -------
        str
            Formatted summary string containing key diagnostic information,
            warnings, and recommendations.
        """
        lines = [
            "=" * 70,
            "SELECTION MECHANISM DIAGNOSTICS",
            "=" * 70,
            "",
            "PANEL BALANCE:",
            f"  Status: {'Balanced' if self.balance_statistics.is_balanced else 'UNBALANCED'}",
            f"  Units: {self.balance_statistics.n_units}",
            f"  Periods: {self.balance_statistics.n_periods}",
            f"  Observations per unit: {self.balance_statistics.min_obs_per_unit} - {self.balance_statistics.max_obs_per_unit}",
            f"  Balance ratio: {self.balance_statistics.balance_ratio:.2%}",
            "",
            "MISSING DATA:",
            f"  Overall missing rate: {self.missing_rate_overall:.2%}",
            f"  Pattern classification: {self.missing_pattern.value}",
            f"  Classification confidence: {self.missing_pattern_confidence:.0%}",
            "",
            "ATTRITION:",
            f"  Attrition rate: {self.attrition_analysis.attrition_rate:.2%}",
            f"  Complete units: {self.attrition_analysis.n_units_complete}",
            f"  Partial units: {self.attrition_analysis.n_units_partial}",
            f"  Late entry rate: {self.attrition_analysis.late_entry_rate:.2%}",
            f"  Early dropout rate: {self.attrition_analysis.early_dropout_rate:.2%}",
            "",
            f"SELECTION RISK: {self.selection_risk.value.upper()}",
            "",
            "METHOD USABILITY:",
            f"  Demean (≥1 pre-period): {self.balance_statistics.pct_usable_demean:.1f}% of treated units",
            f"  Detrend (≥2 pre-periods): {self.balance_statistics.pct_usable_detrend:.1f}% of treated units",
        ]
        
        if self.selection_tests:
            lines.extend(["", "STATISTICAL TESTS:"])
            for test in self.selection_tests:
                status = "REJECT" if test.reject_null else "FAIL TO REJECT"
                lines.append(f"  {test.test_name}:")
                lines.append(f"    Statistic: {test.statistic:.4f}, p-value: {test.pvalue:.4f} ({status})")
        
        if self.warnings:
            lines.extend(["", "⚠️  WARNINGS:"])
            for w in self.warnings:
                lines.append(f"  • {w}")
        
        if self.recommendations:
            lines.extend(["", "📋 RECOMMENDATIONS:"])
            for r in self.recommendations:
                lines.append(f"  → {r}")
        
        lines.extend([
            "",
            "=" * 70,
            "SELECTION MECHANISM ASSUMPTION:",
            "  Selection may depend on unobserved time-invariant heterogeneity,",
            "  but cannot systematically depend on time-varying outcome shocks.",
            "=" * 70,
        ])
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict[str, Any]:
        """
        Convert diagnostics to dictionary format.
        
        Returns
        -------
        dict[str, Any]
            Dictionary containing all diagnostic information.
        """
        return {
            'missing_pattern': self.missing_pattern.value,
            'missing_pattern_confidence': self.missing_pattern_confidence,
            'selection_risk': self.selection_risk.value,
            'missing_rate_overall': self.missing_rate_overall,
            'missing_rate_by_period': self.missing_rate_by_period,
            'missing_rate_by_cohort': self.missing_rate_by_cohort,
            'balance_statistics': {
                'is_balanced': self.balance_statistics.is_balanced,
                'n_units': self.balance_statistics.n_units,
                'n_periods': self.balance_statistics.n_periods,
                'balance_ratio': self.balance_statistics.balance_ratio,
                'pct_usable_demean': self.balance_statistics.pct_usable_demean,
                'pct_usable_detrend': self.balance_statistics.pct_usable_detrend,
            },
            'attrition_analysis': {
                'attrition_rate': self.attrition_analysis.attrition_rate,
                'n_units_complete': self.attrition_analysis.n_units_complete,
                'n_units_partial': self.attrition_analysis.n_units_partial,
            },
            'recommendations': self.recommendations,
            'warnings': self.warnings,
        }


# =============================================================================
# Helper Functions
# =============================================================================

def _validate_diagnostic_inputs(
    data: pd.DataFrame,
    y: str,
    ivar: str,
    tvar: str,
    gvar: str | None,
) -> None:
    """
    Validate inputs for diagnostic functions.
    
    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format.
    y : str
        Outcome variable column name.
    ivar : str
        Unit identifier column name.
    tvar : str
        Time variable column name.
    gvar : str or None
        Cohort variable column name.
    
    Raises
    ------
    ValueError
        If required columns are missing or data is insufficient.
    TypeError
        If time variable is not numeric.
    """
    # Check required columns exist
    required = [y, ivar, tvar]
    if gvar is not None:
        required.append(gvar)
    
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Check data types
    if not pd.api.types.is_numeric_dtype(data[tvar]):
        raise TypeError(
            f"Time variable '{tvar}' must be numeric. "
            f"Found type: {data[tvar].dtype}"
        )
    
    # Check minimum data requirements
    if len(data) < 3:
        raise ValueError(
            "Insufficient data: need at least 3 observations for diagnostics."
        )
    
    if data[ivar].nunique() < 2:
        raise ValueError(
            "Need at least 2 unique units for meaningful diagnostics."
        )
    
    if data[tvar].nunique() < 2:
        raise ValueError(
            "Need at least 2 unique time periods for panel diagnostics."
        )


def _is_never_treated(gvar_value: Any, never_treated_values: List) -> bool:
    """
    Check if a gvar value indicates never-treated status.
    
    Parameters
    ----------
    gvar_value : Any
        Value from the gvar column.
    never_treated_values : List
        List of values indicating never-treated status.
    
    Returns
    -------
    bool
        True if the value indicates never-treated status.
    """
    if pd.isna(gvar_value):
        return True
    if gvar_value in never_treated_values:
        return True
    if isinstance(gvar_value, float) and np.isinf(gvar_value):
        return True
    return False


def _compute_balance_statistics(
    data: pd.DataFrame,
    y: str,
    ivar: str,
    tvar: str,
    gvar: str | None,
    never_treated_values: List,
) -> BalanceStatistics:
    """
    Compute panel balance statistics.
    
    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format.
    y : str
        Outcome variable column name.
    ivar : str
        Unit identifier column name.
    tvar : str
        Time variable column name.
    gvar : str or None
        Cohort variable column name.
    never_treated_values : List
        Values indicating never-treated status.
    
    Returns
    -------
    BalanceStatistics
        Panel balance statistics.
    """
    all_periods = sorted(data[tvar].unique())
    n_periods = len(all_periods)
    
    # Count observations per unit
    obs_per_unit = data.groupby(ivar).size()
    n_units = len(obs_per_unit)
    
    is_balanced = obs_per_unit.nunique() == 1
    min_obs = int(obs_per_unit.min())
    max_obs = int(obs_per_unit.max())
    mean_obs = float(obs_per_unit.mean())
    std_obs = float(obs_per_unit.std()) if len(obs_per_unit) > 1 else 0.0
    balance_ratio = min_obs / max_obs if max_obs > 0 else 0.0
    
    # Count units below method thresholds
    units_below_demean = 0
    units_below_detrend = 0
    n_treated_units = 0
    
    if gvar is not None:
        unit_gvar = data.drop_duplicates(subset=[ivar]).set_index(ivar)[gvar]
        
        for unit_id in data[ivar].unique():
            g = unit_gvar.get(unit_id)
            
            # Skip never-treated
            if _is_never_treated(g, never_treated_values):
                continue
            
            n_treated_units += 1
            
            # Count pre-treatment observations
            unit_data = data[data[ivar] == unit_id]
            n_pre = len(unit_data[unit_data[tvar] < g])
            
            if n_pre < 1:
                units_below_demean += 1
            if n_pre < 2:
                units_below_detrend += 1
    
    # Calculate usability percentages
    if n_treated_units > 0:
        pct_demean = 100.0 * (1 - units_below_demean / n_treated_units)
        pct_detrend = 100.0 * (1 - units_below_detrend / n_treated_units)
    else:
        pct_demean = 100.0
        pct_detrend = 100.0
    
    return BalanceStatistics(
        is_balanced=is_balanced,
        n_units=n_units,
        n_periods=n_periods,
        min_obs_per_unit=min_obs,
        max_obs_per_unit=max_obs,
        mean_obs_per_unit=mean_obs,
        std_obs_per_unit=std_obs,
        balance_ratio=balance_ratio,
        units_below_demean_threshold=units_below_demean,
        units_below_detrend_threshold=units_below_detrend,
        pct_usable_demean=pct_demean,
        pct_usable_detrend=pct_detrend,
    )


def _compute_attrition_analysis(
    data: pd.DataFrame,
    ivar: str,
    tvar: str,
    gvar: str | None,
    never_treated_values: List,
) -> AttritionAnalysis:
    """
    Compute attrition pattern analysis.
    
    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format.
    ivar : str
        Unit identifier column name.
    tvar : str
        Time variable column name.
    gvar : str or None
        Cohort variable column name.
    never_treated_values : List
        Values indicating never-treated status.
    
    Returns
    -------
    AttritionAnalysis
        Attrition pattern analysis.
    """
    all_periods = sorted(data[tvar].unique())
    T_min, T_max = min(all_periods), max(all_periods)
    n_periods = len(all_periods)
    
    # Count observations per unit
    obs_per_unit = data.groupby(ivar).size()
    n_units = len(obs_per_unit)
    
    # Identify complete vs partial units
    complete_units = obs_per_unit[obs_per_unit == n_periods].index
    n_complete = len(complete_units)
    n_partial = n_units - n_complete
    attrition_rate = n_partial / n_units if n_units > 0 else 0.0
    
    # Attrition by cohort
    attrition_by_cohort = {}
    if gvar is not None:
        unit_gvar = data.drop_duplicates(subset=[ivar]).set_index(ivar)[gvar]
        
        for g in data[gvar].dropna().unique():
            if _is_never_treated(g, never_treated_values):
                continue
            
            cohort_units = unit_gvar[unit_gvar == g].index
            cohort_complete = len([u for u in cohort_units if u in complete_units])
            cohort_total = len(cohort_units)
            
            if cohort_total > 0:
                attrition_by_cohort[int(g)] = 1 - cohort_complete / cohort_total
    
    # Attrition by period (proportion not observed at each time)
    attrition_by_period = {}
    for t in all_periods:
        units_observed_at_t = data[data[tvar] == t][ivar].nunique()
        attrition_by_period[int(t)] = 1 - units_observed_at_t / n_units
    
    # Early dropout / late entry
    first_obs = data.groupby(ivar)[tvar].min()
    last_obs = data.groupby(ivar)[tvar].max()
    
    late_entry_rate = float((first_obs > T_min).mean())
    early_dropout_rate = float((last_obs < T_max).mean())
    
    # Dropout before/after treatment
    dropout_before = 0
    dropout_after = 0
    
    if gvar is not None:
        unit_gvar = data.drop_duplicates(subset=[ivar]).set_index(ivar)[gvar]
        
        for unit_id in data[ivar].unique():
            g = unit_gvar.get(unit_id)
            
            if _is_never_treated(g, never_treated_values):
                continue
            
            unit_last = last_obs.get(unit_id)
            
            if unit_last < g:
                dropout_before += 1
            elif unit_last < T_max:
                dropout_after += 1
    
    return AttritionAnalysis(
        n_units_complete=n_complete,
        n_units_partial=n_partial,
        attrition_rate=attrition_rate,
        attrition_by_cohort=attrition_by_cohort,
        attrition_by_period=attrition_by_period,
        early_dropout_rate=early_dropout_rate,
        late_entry_rate=late_entry_rate,
        dropout_before_treatment=dropout_before,
        dropout_after_treatment=dropout_after,
    )


def _classify_missing_pattern(
    data: pd.DataFrame,
    y: str,
    ivar: str,
    tvar: str,
    controls: Optional[list[str]] = None,
) -> Tuple[MissingPattern, float, List[SelectionTestResult]]:
    """
    Classify missing data pattern using statistical tests.
    
    Implements a simplified version of Little's MCAR test and auxiliary
    regressions to classify the missing data mechanism.
    
    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format.
    y : str
        Outcome variable column name.
    ivar : str
        Unit identifier column name.
    tvar : str
        Time variable column name.
    controls : list[str] or None
        Control variable column names.
    
    Returns
    -------
    Tuple[MissingPattern, float, List[SelectionTestResult]]
        - Classified missing pattern
        - Confidence in classification (0-1)
        - List of test results
    """
    tests = []
    
    # Create full panel and missing indicator
    all_periods = sorted(data[tvar].unique())
    all_units = data[ivar].unique()
    
    full_index = pd.MultiIndex.from_product(
        [all_units, all_periods], names=[ivar, tvar]
    )
    full_panel = pd.DataFrame(index=full_index).reset_index()
    merged = full_panel.merge(data, on=[ivar, tvar], how='left')
    
    # M_it = 1 if missing
    merged['_missing'] = merged[y].isna().astype(int)
    
    # If no missing data, return MCAR with high confidence
    if merged['_missing'].sum() == 0:
        return MissingPattern.MCAR, 1.0, []
    
    # =========================================================================
    # Test 1: Simplified Little's MCAR Test
    # Compare mean outcome between complete and incomplete units
    # =========================================================================
    obs_per_unit = merged.groupby(ivar)['_missing'].sum()
    complete_units = obs_per_unit[obs_per_unit == 0].index
    incomplete_units = obs_per_unit[obs_per_unit > 0].index
    
    if len(complete_units) >= 5 and len(incomplete_units) >= 5:
        # Get observed Y values for each group
        complete_y = data[data[ivar].isin(complete_units)][y].dropna()
        incomplete_y = data[data[ivar].isin(incomplete_units)][y].dropna()
        
        if len(complete_y) > 1 and len(incomplete_y) > 1:
            complete_mean = complete_y.mean()
            incomplete_mean = incomplete_y.mean()
            complete_std = complete_y.std()
            incomplete_std = incomplete_y.std()
            
            n1, n2 = len(complete_units), len(incomplete_units)
            
            # Pooled standard error
            pooled_se = np.sqrt(
                complete_std**2 / n1 + incomplete_std**2 / n2
            )
            
            if pooled_se > 0:
                t_stat = (complete_mean - incomplete_mean) / pooled_se
                df = n1 + n2 - 2
                pvalue = 2 * (1 - stats.t.cdf(abs(t_stat), df))
                
                tests.append(SelectionTestResult(
                    test_name="Simplified Little's MCAR Test",
                    statistic=float(t_stat),
                    pvalue=float(pvalue),
                    reject_null=pvalue < 0.05,
                    interpretation=(
                        "Reject MCAR: significant difference in outcomes between "
                        "complete and incomplete units" if pvalue < 0.05 else
                        "Cannot reject MCAR: no significant difference detected"
                    ),
                    details={
                        'complete_mean': float(complete_mean),
                        'incomplete_mean': float(incomplete_mean),
                        'n_complete': n1,
                        'n_incomplete': n2,
                    }
                ))
    
    # =========================================================================
    # Test 2: Selection on Observables (MAR test)
    # Regress unit-level missing rate on controls
    # =========================================================================
    if controls and len(controls) > 0:
        # Get unit-level controls (first observation per unit)
        available_controls = [c for c in controls if c in data.columns]
        
        if available_controls:
            unit_controls = data.drop_duplicates(subset=[ivar])[
                available_controls + [ivar]
            ].dropna()
            
            # Compute unit-level missing rate
            unit_missing = merged.groupby(ivar)['_missing'].mean().reset_index()
            unit_missing.columns = [ivar, '_unit_missing_rate']
            
            test_data = unit_missing.merge(unit_controls, on=ivar)
            
            if len(test_data) > len(available_controls) + 2:
                X = test_data[available_controls].values
                X = np.column_stack([np.ones(len(X)), X])  # Add constant
                y_miss = test_data['_unit_missing_rate'].values
                
                try:
                    # OLS regression
                    beta, residuals, rank, s = np.linalg.lstsq(X, y_miss, rcond=None)
                    
                    # Compute R-squared
                    y_pred = X @ beta
                    SS_res = np.sum((y_miss - y_pred)**2)
                    SS_tot = np.sum((y_miss - y_miss.mean())**2)
                    R2 = 1 - SS_res / SS_tot if SS_tot > 0 else 0
                    
                    # F-test for joint significance
                    k = len(available_controls)
                    n = len(y_miss)
                    
                    if R2 < 1 and n > k + 1:
                        F_stat = (R2 / k) / ((1 - R2) / (n - k - 1))
                        pvalue = 1 - stats.f.cdf(F_stat, k, n - k - 1)
                        
                        tests.append(SelectionTestResult(
                            test_name="Selection on Observables (MAR) Test",
                            statistic=float(F_stat),
                            pvalue=float(pvalue),
                            reject_null=pvalue < 0.05,
                            interpretation=(
                                "Selection depends on observed controls (MAR)" 
                                if pvalue < 0.05 else
                                "No evidence of selection on observed controls"
                            ),
                            details={
                                'R2': float(R2),
                                'controls': available_controls,
                            }
                        ))
                except (np.linalg.LinAlgError, ValueError):
                    pass
    
    # =========================================================================
    # Test 3: Selection on Lagged Outcomes (MNAR indicator)
    # Test if missingness correlates with lagged Y
    # =========================================================================
    data_sorted = data.sort_values([ivar, tvar])
    data_sorted['_y_lag'] = data_sorted.groupby(ivar)[y].shift(1)
    
    # Merge with missing indicator
    lag_test = merged.merge(
        data_sorted[[ivar, tvar, '_y_lag']],
        on=[ivar, tvar],
        how='left'
    )
    lag_test = lag_test.dropna(subset=['_y_lag'])
    
    if len(lag_test) > 10:
        # Point-biserial correlation
        try:
            corr, pvalue = stats.pointbiserialr(
                lag_test['_missing'].values,
                lag_test['_y_lag'].values
            )
            
            tests.append(SelectionTestResult(
                test_name="Selection on Lagged Outcome (MNAR) Test",
                statistic=float(corr),
                pvalue=float(pvalue),
                reject_null=pvalue < 0.05,
                interpretation=(
                    "WARNING: Missingness correlates with lagged outcomes. "
                    "This suggests potential MNAR and may violate the selection "
                    "mechanism assumption." if pvalue < 0.05 else
                    "No evidence of selection on lagged outcomes"
                ),
                details={'correlation': float(corr)}
            ))
        except (ValueError, TypeError):
            pass
    
    # =========================================================================
    # Classify Pattern Based on Test Results
    # =========================================================================
    mcar_rejected = any(
        t.test_name.startswith("Simplified Little") and t.reject_null 
        for t in tests
    )
    mar_detected = any(
        t.test_name.startswith("Selection on Observables") and t.reject_null 
        for t in tests
    )
    mnar_detected = any(
        t.test_name.startswith("Selection on Lagged") and t.reject_null 
        for t in tests
    )
    
    if mnar_detected:
        pattern = MissingPattern.MNAR
        confidence = 0.7  # MNAR is hard to confirm definitively
    elif mar_detected:
        pattern = MissingPattern.MAR
        confidence = 0.8
    elif not mcar_rejected:
        pattern = MissingPattern.MCAR
        confidence = 0.9 if len(tests) > 0 else 0.5
    else:
        pattern = MissingPattern.UNKNOWN
        confidence = 0.3
    
    return pattern, confidence, tests


def _assess_selection_risk(
    missing_pattern: MissingPattern,
    attrition_analysis: AttritionAnalysis,
    balance_statistics: BalanceStatistics,
    selection_tests: List[SelectionTestResult],
) -> Tuple[SelectionRisk, list[str], list[str]]:
    """
    Assess overall selection bias risk based on multiple indicators.
    
    Risk Assessment Criteria:
    
    LOW Risk (acceptable):
    - Missing pattern is MCAR or MAR
    - Attrition rate < 10%
    - No significant selection on lagged outcomes
    - Balance ratio > 0.8
    
    MEDIUM Risk (caution):
    - Missing pattern is MAR with moderate attrition
    - Attrition rate 10-30%
    - Some evidence of differential attrition by cohort
    - Balance ratio 0.5-0.8
    
    HIGH Risk (problematic):
    - Missing pattern is MNAR
    - Attrition rate > 30%
    - Strong selection on lagged outcomes
    - Differential dropout before/after treatment
    - Balance ratio < 0.5
    
    Parameters
    ----------
    missing_pattern : MissingPattern
        Classified missing data pattern.
    attrition_analysis : AttritionAnalysis
        Attrition pattern analysis.
    balance_statistics : BalanceStatistics
        Panel balance statistics.
    selection_tests : List[SelectionTestResult]
        Statistical test results.
    
    Returns
    -------
    Tuple[SelectionRisk, list[str], list[str]]
        - Assessed risk level
        - List of recommendations
        - List of warnings
    """
    recommendations = []
    warnings = []
    risk_score = 0  # 0-100 scale
    
    # =========================================================================
    # Factor 1: Missing Pattern (weight: 30%)
    # =========================================================================
    if missing_pattern == MissingPattern.MCAR:
        risk_score += 0
    elif missing_pattern == MissingPattern.MAR:
        risk_score += 15
    elif missing_pattern == MissingPattern.MNAR:
        risk_score += 30
        warnings.append(
            "Missing data pattern suggests selection on unobservables. "
            "This may violate the selection mechanism assumption."
        )
    else:  # UNKNOWN
        risk_score += 10
    
    # =========================================================================
    # Factor 2: Attrition Rate (weight: 25%)
    # =========================================================================
    attrition = attrition_analysis.attrition_rate
    if attrition < 0.10:
        risk_score += 0
    elif attrition < 0.30:
        risk_score += 12
    else:
        risk_score += 25
        warnings.append(
            f"High attrition rate ({attrition:.1%}). Consider using "
            "detrending which is more robust to selection on trends."
        )
    
    # =========================================================================
    # Factor 3: Differential Attrition (weight: 25%)
    # =========================================================================
    dropout_before = attrition_analysis.dropout_before_treatment
    dropout_after = attrition_analysis.dropout_after_treatment
    
    if dropout_after > 0 and dropout_before > 0:
        if dropout_after > dropout_before * 2:
            risk_score += 25
            warnings.append(
                f"Significantly more dropout after treatment ({dropout_after}) "
                f"than before ({dropout_before}). This may indicate selection "
                "related to treatment effects."
            )
        elif dropout_after > dropout_before * 1.5:
            risk_score += 15
    
    # =========================================================================
    # Factor 4: Balance Ratio (weight: 20%)
    # =========================================================================
    balance = balance_statistics.balance_ratio
    if balance > 0.8:
        risk_score += 0
    elif balance > 0.5:
        risk_score += 10
    else:
        risk_score += 20
        warnings.append(
            f"Low balance ratio ({balance:.1%}). Some units have much fewer "
            "observations than others."
        )
    
    # =========================================================================
    # Determine Risk Level and Generate Recommendations
    # =========================================================================
    if risk_score < 25:
        risk = SelectionRisk.LOW
        recommendations.append(
            "Selection risk is low. Proceed with estimation. "
            "The selection mechanism assumption appears reasonable."
        )
    elif risk_score < 50:
        risk = SelectionRisk.MEDIUM
        recommendations.extend([
            "Moderate selection risk detected. Consider the following:",
            "1. Use rolling='detrend' for additional robustness to selection on trends",
            "2. Compare results with a balanced subsample as sensitivity check",
            "3. Report both demean and detrend results for transparency",
        ])
    else:
        risk = SelectionRisk.HIGH
        recommendations.extend([
            "High selection risk detected. Strongly recommend:",
            "1. Use rolling='detrend' method (more robust to selection on trends)",
            "2. Conduct sensitivity analysis with balanced subsample",
            "3. Report diagnostics and discuss potential selection bias",
            "4. Consider alternative identification strategies if possible",
        ])
    
    # Add method-specific recommendations
    if balance_statistics.pct_usable_detrend < 90:
        recommendations.append(
            f"Note: Only {balance_statistics.pct_usable_detrend:.1f}% of treated "
            "units have sufficient pre-treatment periods for detrending. "
            "Consider using demean if detrending excludes too many units."
        )
    
    return risk, recommendations, warnings


def _compute_missing_rates(
    data: pd.DataFrame,
    y: str,
    ivar: str,
    tvar: str,
    gvar: str | None,
    never_treated_values: List,
) -> Tuple[float, dict[int, float], dict[int, float]]:
    """
    Compute missing rates overall, by period, and by cohort.
    
    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format.
    y : str
        Outcome variable column name.
    ivar : str
        Unit identifier column name.
    tvar : str
        Time variable column name.
    gvar : str or None
        Cohort variable column name.
    never_treated_values : List
        Values indicating never-treated status.
    
    Returns
    -------
    Tuple[float, dict[int, float], dict[int, float]]
        - Overall missing rate
        - Missing rate by period
        - Missing rate by cohort
    """
    all_periods = sorted(data[tvar].unique())
    all_units = data[ivar].unique()
    
    # Create full panel index
    full_index = pd.MultiIndex.from_product(
        [all_units, all_periods], names=[ivar, tvar]
    )
    full_panel = pd.DataFrame(index=full_index).reset_index()
    merged = full_panel.merge(data[[ivar, tvar, y]], on=[ivar, tvar], how='left')
    
    # Overall missing rate
    missing_rate_overall = float(merged[y].isna().mean())
    
    # Missing rate by period
    missing_rate_by_period = {}
    for t in all_periods:
        period_data = merged[merged[tvar] == t]
        missing_rate_by_period[int(t)] = float(period_data[y].isna().mean())
    
    # Missing rate by cohort
    missing_rate_by_cohort = {}
    if gvar is not None:
        unit_gvar = data.drop_duplicates(subset=[ivar]).set_index(ivar)[gvar]
        
        for g in data[gvar].dropna().unique():
            if _is_never_treated(g, never_treated_values):
                continue
            
            cohort_units = unit_gvar[unit_gvar == g].index
            cohort_data = merged[merged[ivar].isin(cohort_units)]
            
            if len(cohort_data) > 0:
                missing_rate_by_cohort[int(g)] = float(cohort_data[y].isna().mean())
    
    return missing_rate_overall, missing_rate_by_period, missing_rate_by_cohort


def _compute_unit_stats(
    data: pd.DataFrame,
    y: str,
    ivar: str,
    tvar: str,
    gvar: str | None,
    never_treated_values: List,
) -> List[UnitMissingStats]:
    """
    Compute per-unit missing data statistics.
    
    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format.
    y : str
        Outcome variable column name.
    ivar : str
        Unit identifier column name.
    tvar : str
        Time variable column name.
    gvar : str or None
        Cohort variable column name.
    never_treated_values : List
        Values indicating never-treated status.
    
    Returns
    -------
    List[UnitMissingStats]
        Per-unit missing data statistics.
    """
    all_periods = sorted(data[tvar].unique())
    n_total_periods = len(all_periods)
    T_min, T_max = min(all_periods), max(all_periods)
    
    unit_stats = []
    
    # Get unit-level gvar if available
    unit_gvar = None
    if gvar is not None:
        unit_gvar = data.drop_duplicates(subset=[ivar]).set_index(ivar)[gvar]
    
    for unit_id in data[ivar].unique():
        unit_data = data[data[ivar] == unit_id]
        
        # Basic statistics
        observed_periods = unit_data[tvar].unique()
        n_observed = len(observed_periods)
        n_missing = n_total_periods - n_observed
        missing_rate = n_missing / n_total_periods if n_total_periods > 0 else 0.0
        
        first_observed = int(unit_data[tvar].min())
        last_observed = int(unit_data[tvar].max())
        observation_span = last_observed - first_observed + 1
        
        # Cohort information
        cohort = None
        is_treated = False
        n_pre_treatment = None
        n_post_treatment = None
        pre_missing_rate = None
        post_missing_rate = None
        can_use_demean = True
        can_use_detrend = True
        reason_if_excluded = None
        
        if unit_gvar is not None:
            g = unit_gvar.get(unit_id)
            
            if not _is_never_treated(g, never_treated_values):
                cohort = int(g)
                is_treated = True
                
                # Count pre/post treatment observations
                pre_periods = [t for t in all_periods if t < g]
                post_periods = [t for t in all_periods if t >= g]
                
                n_pre_treatment = len([t for t in observed_periods if t < g])
                n_post_treatment = len([t for t in observed_periods if t >= g])
                
                if len(pre_periods) > 0:
                    pre_missing_rate = 1 - n_pre_treatment / len(pre_periods)
                if len(post_periods) > 0:
                    post_missing_rate = 1 - n_post_treatment / len(post_periods)
                
                # Check method usability
                if n_pre_treatment < 1:
                    can_use_demean = False
                    reason_if_excluded = "No pre-treatment observations"
                if n_pre_treatment < 2:
                    can_use_detrend = False
                    if reason_if_excluded is None:
                        reason_if_excluded = "Fewer than 2 pre-treatment observations"
        
        unit_stats.append(UnitMissingStats(
            unit_id=unit_id,
            cohort=cohort,
            is_treated=is_treated,
            n_total_periods=n_total_periods,
            n_observed=n_observed,
            n_missing=n_missing,
            missing_rate=missing_rate,
            first_observed=first_observed,
            last_observed=last_observed,
            observation_span=observation_span,
            n_pre_treatment=n_pre_treatment,
            n_post_treatment=n_post_treatment,
            pre_treatment_missing_rate=pre_missing_rate,
            post_treatment_missing_rate=post_missing_rate,
            can_use_demean=can_use_demean,
            can_use_detrend=can_use_detrend,
            reason_if_excluded=reason_if_excluded,
        ))
    
    return unit_stats


# =============================================================================
# Main Public Functions
# =============================================================================

def diagnose_selection_mechanism(
    data: pd.DataFrame,
    y: str,
    ivar: str,
    tvar: str,
    gvar: str | None = None,
    controls: Optional[list[str]] = None,
    never_treated_values: Optional[List] = None,
    verbose: bool = True,
) -> SelectionDiagnostics:
    """
    Diagnose potential selection mechanism violations in unbalanced panels.
    
    This function implements diagnostic procedures to assess whether the
    selection mechanism assumption is likely to hold. The key assumption is
    that selection (missing data) may depend on time-invariant heterogeneity
    but not on time-varying outcome shocks.
    
    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format.
    y : str
        Outcome variable column name.
    ivar : str
        Unit identifier column name.
    tvar : str
        Time variable column name.
    gvar : str, optional
        Cohort variable for staggered designs. If None, assumes common timing
        and skips cohort-specific diagnostics.
    controls : list of str, optional
        Control variable column names for additional diagnostics.
    never_treated_values : list, optional
        Values in gvar indicating never-treated units. Default: [0, np.inf].
    verbose : bool, default True
        Whether to print diagnostic summary.
    
    Returns
    -------
    SelectionDiagnostics
        Comprehensive diagnostic results including:
        
        - missing_pattern: Classified pattern (MCAR, MAR, MNAR)
        - selection_risk: Risk level for selection bias
        - attrition_analysis: Detailed attrition patterns
        - balance_statistics: Panel balance metrics
        - recommendations: Actionable suggestions
        - selection_tests: Statistical test results
    
    Notes
    -----
    The function performs several diagnostic procedures:
    
    1. **Balance Statistics**: Computes panel balance metrics and identifies
       units that cannot be used for demeaning (< 1 pre-period) or
       detrending (< 2 pre-periods).
    
    2. **Attrition Analysis**: Analyzes dropout patterns by cohort and time,
       distinguishing between early dropout (before treatment) and late
       dropout (after treatment).
    
    3. **Missing Pattern Classification**: Uses Little's MCAR test and
       auxiliary regressions to classify the missing data mechanism.
    
    4. **Selection Risk Assessment**: Combines multiple indicators to
       assess the overall risk of selection bias.
    
    The selection mechanism assumption requires that selection may depend on
    unobserved time-constant heterogeneity (which is removed by the rolling
    transformation, similar to the fixed effects estimator), but cannot
    systematically depend on time-varying outcome shocks.
    
    See Also
    --------
    plot_missing_pattern : Visualize missing data patterns.
    get_unit_missing_stats : Get per-unit missing statistics as DataFrame.
    """
    # Validate inputs
    _validate_diagnostic_inputs(data, y, ivar, tvar, gvar)
    
    if never_treated_values is None:
        never_treated_values = [0, np.inf]
    
    # Step 1: Compute balance statistics
    balance_stats = _compute_balance_statistics(
        data, y, ivar, tvar, gvar, never_treated_values
    )
    
    # Step 2: Compute attrition analysis
    attrition = _compute_attrition_analysis(
        data, ivar, tvar, gvar, never_treated_values
    )
    
    # Step 3: Classify missing pattern
    pattern, confidence, tests = _classify_missing_pattern(
        data, y, ivar, tvar, controls
    )
    
    # Step 4: Assess selection risk
    risk, recommendations, warnings = _assess_selection_risk(
        pattern, attrition, balance_stats, tests
    )
    
    # Step 5: Compute missing rates
    missing_overall, missing_by_period, missing_by_cohort = _compute_missing_rates(
        data, y, ivar, tvar, gvar, never_treated_values
    )
    
    # Step 6: Compute unit statistics
    unit_stats = _compute_unit_stats(
        data, y, ivar, tvar, gvar, never_treated_values
    )
    
    # Step 7: Assemble result
    result = SelectionDiagnostics(
        missing_pattern=pattern,
        missing_pattern_confidence=confidence,
        selection_risk=risk,
        attrition_analysis=attrition,
        balance_statistics=balance_stats,
        recommendations=recommendations,
        warnings=warnings,
        missing_rate_overall=missing_overall,
        missing_rate_by_period=missing_by_period,
        missing_rate_by_cohort=missing_by_cohort,
        selection_tests=tests,
        unit_stats=unit_stats,
    )
    
    if verbose:
        print(result.summary())
    
    return result


def get_unit_missing_stats(
    data: pd.DataFrame,
    y: str,
    ivar: str,
    tvar: str,
    gvar: str | None = None,
    never_treated_values: Optional[List] = None,
) -> pd.DataFrame:
    """
    Compute per-unit missing data statistics as a DataFrame.
    
    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format.
    y : str
        Outcome variable column name.
    ivar : str
        Unit identifier column name.
    tvar : str
        Time variable column name.
    gvar : str, optional
        Cohort variable for staggered designs.
    never_treated_values : list, optional
        Values indicating never-treated units. Default: [0, np.inf].
    
    Returns
    -------
    pd.DataFrame
        DataFrame with one row per unit containing:
        
        - unit_id: Unit identifier
        - cohort: Treatment cohort (NaN for never-treated)
        - is_treated: Whether unit is ever treated
        - n_observed: Number of observed periods
        - n_missing: Number of missing periods
        - missing_rate: Proportion missing
        - n_pre_treatment: Pre-treatment observations
        - n_post_treatment: Post-treatment observations
        - can_use_demean: Sufficient data for demeaning
        - can_use_detrend: Sufficient data for detrending
    
    See Also
    --------
    diagnose_selection_mechanism : Comprehensive diagnostics.
    """
    _validate_diagnostic_inputs(data, y, ivar, tvar, gvar)
    
    if never_treated_values is None:
        never_treated_values = [0, np.inf]
    
    unit_stats = _compute_unit_stats(
        data, y, ivar, tvar, gvar, never_treated_values
    )
    
    # Convert to DataFrame
    records = []
    for us in unit_stats:
        records.append({
            'unit_id': us.unit_id,
            'cohort': us.cohort,
            'is_treated': us.is_treated,
            'n_total_periods': us.n_total_periods,
            'n_observed': us.n_observed,
            'n_missing': us.n_missing,
            'missing_rate': us.missing_rate,
            'first_observed': us.first_observed,
            'last_observed': us.last_observed,
            'observation_span': us.observation_span,
            'n_pre_treatment': us.n_pre_treatment,
            'n_post_treatment': us.n_post_treatment,
            'pre_treatment_missing_rate': us.pre_treatment_missing_rate,
            'post_treatment_missing_rate': us.post_treatment_missing_rate,
            'can_use_demean': us.can_use_demean,
            'can_use_detrend': us.can_use_detrend,
            'reason_if_excluded': us.reason_if_excluded,
        })
    
    return pd.DataFrame(records)


def plot_missing_pattern(
    data: pd.DataFrame,
    ivar: str,
    tvar: str,
    y: str | None = None,
    gvar: str | None = None,
    sort_by: str = 'cohort',
    figsize: Tuple[float, float] = (12, 8),
    cmap: str = 'RdYlGn',
    show_cohort_lines: bool = True,
    never_treated_values: Optional[List] = None,
    max_units: int = 200,
    ax: Optional[Any] = None,
) -> Any:
    """
    Visualize missing data patterns in panel data.
    
    Creates a heatmap showing observation availability across units and time.
    Units can be sorted by cohort, missing rate, or unit ID.
    
    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format.
    ivar : str
        Unit identifier column name.
    tvar : str
        Time variable column name.
    y : str, optional
        Outcome variable. If provided, checks for missing Y values.
        If None, checks for missing rows.
    gvar : str, optional
        Cohort variable. If provided, shows treatment timing.
    sort_by : str, default 'cohort'
        How to sort units: 'cohort', 'missing_rate', 'unit_id'.
    figsize : tuple, default (12, 8)
        Figure size in inches.
    cmap : str, default 'RdYlGn'
        Colormap for the heatmap (not used, custom colors applied).
    show_cohort_lines : bool, default True
        Whether to show treatment timing lines.
    never_treated_values : list, optional
        Values indicating never-treated units. Default: [0, np.inf].
    max_units : int, default 200
        Maximum number of units to display. If more units exist,
        a random sample is shown.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the missing pattern heatmap.
    
    Notes
    -----
    The heatmap uses the following color coding:
    
    - Green: Observed (Y value present)
    - Red: Missing (Y value missing or row absent)
    - Black line: Treatment timing (if gvar provided)
    
    See Also
    --------
    diagnose_selection_mechanism : Comprehensive diagnostics.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.colors import ListedColormap
    except ImportError:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install it with: pip install matplotlib"
        )
    
    if never_treated_values is None:
        never_treated_values = [0, np.inf]
    
    # Get all units and periods
    all_units = list(data[ivar].unique())
    all_periods = sorted(data[tvar].unique())
    
    # Sample units if too many
    if len(all_units) > max_units:
        import warnings
        warnings.warn(
            f"Panel has {len(all_units)} units. Showing random sample of {max_units}.",
            UserWarning
        )
        np.random.seed(42)
        all_units = list(np.random.choice(all_units, max_units, replace=False))
    
    # Get unit-level gvar for sorting
    unit_gvar = None
    if gvar is not None and gvar in data.columns:
        unit_gvar = data.drop_duplicates(subset=[ivar]).set_index(ivar)[gvar]
    
    # Build observation matrix
    obs_matrix = np.zeros((len(all_units), len(all_periods)))
    
    for i, unit in enumerate(all_units):
        unit_data = data[data[ivar] == unit]
        for j, period in enumerate(all_periods):
            period_data = unit_data[unit_data[tvar] == period]
            if len(period_data) > 0:
                if y is None or period_data[y].notna().any():
                    obs_matrix[i, j] = 1  # Observed
    
    # Sort units
    if sort_by == 'cohort' and unit_gvar is not None:
        sort_keys = []
        for u in all_units:
            g = unit_gvar.get(u)
            if _is_never_treated(g, never_treated_values):
                sort_keys.append(np.inf)
            else:
                sort_keys.append(g if pd.notna(g) else np.inf)
        sort_idx = np.argsort(sort_keys)
    elif sort_by == 'missing_rate':
        missing_rates = 1 - obs_matrix.mean(axis=1)
        sort_idx = np.argsort(missing_rates)
    else:  # 'unit_id'
        sort_idx = np.arange(len(all_units))
    
    obs_matrix = obs_matrix[sort_idx]
    sorted_units = [all_units[i] for i in sort_idx]
    
    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Plot heatmap with custom colors
    custom_cmap = ListedColormap(['#d73027', '#1a9850'])  # Red=missing, Green=observed
    im = ax.imshow(obs_matrix, aspect='auto', cmap=custom_cmap, vmin=0, vmax=1)
    
    # Add cohort lines
    if show_cohort_lines and unit_gvar is not None:
        for i, unit in enumerate(sorted_units):
            g = unit_gvar.get(unit)
            if not _is_never_treated(g, never_treated_values) and pd.notna(g):
                try:
                    j = list(all_periods).index(g)
                    ax.plot([j - 0.5, j - 0.5], [i - 0.5, i + 0.5], 
                           'k-', linewidth=0.8, alpha=0.7)
                except ValueError:
                    pass
    
    # Labels
    ax.set_xlabel('Time Period', fontsize=11)
    ax.set_ylabel('Unit', fontsize=11)
    ax.set_title('Panel Data Observation Pattern\n(Green=Observed, Red=Missing)', 
                fontsize=12)
    
    # X-axis ticks
    if len(all_periods) <= 25:
        ax.set_xticks(range(len(all_periods)))
        ax.set_xticklabels(all_periods, rotation=45, ha='right')
    else:
        # Show subset of ticks
        tick_step = max(1, len(all_periods) // 10)
        tick_positions = range(0, len(all_periods), tick_step)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([all_periods[i] for i in tick_positions], rotation=45, ha='right')
    
    # Y-axis: hide individual unit labels if too many
    if len(sorted_units) > 50:
        ax.set_yticks([])
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#1a9850', label='Observed'),
        mpatches.Patch(facecolor='#d73027', label='Missing'),
    ]
    if show_cohort_lines and unit_gvar is not None:
        legend_elements.append(
            plt.Line2D([0], [0], color='black', linewidth=1, label='Treatment Start')
        )
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    return fig

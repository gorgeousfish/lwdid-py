"""
Diagnostics for parallel trends and heterogeneous trends in DiD estimation.

This module provides tools for assessing the parallel trends assumption and
detecting heterogeneous trends across treatment cohorts in difference-in-
differences (DiD) settings. The main functions test parallel trends via
placebo effects, diagnose trend heterogeneity, recommend transformation
methods, and visualize cohort-specific trajectories.

The conditional heterogeneous trends (CHT) framework allows each treatment
cohort to have its own linear trend, relaxing the standard parallel trends
assumption. Under CHT, the expected outcome in the never-treated state
includes cohort-specific linear time trends in addition to common time
effects and covariate adjustments.

Notes
-----
When parallel trends holds, demeaning is more efficient. When CHT holds
but parallel trends fails, detrending removes cohort-specific linear
trends and restores consistency.

For placebo testing, rolling transformations applied to pre-treatment
periods produce proper standard errors that account for the panel structure.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

# Import staggered module functions for proper placebo testing.
# These implement rolling demeaning and detrending transformations.
try:
    from .staggered.transformations_pre import (
        transform_staggered_demean_pre,
        transform_staggered_detrend_pre,
    )
    from .staggered.estimation_pre import (
        estimate_pre_treatment_effects,
        PreTreatmentEffect,
    )
    from .staggered.parallel_trends import (
        run_parallel_trends_test as _run_staggered_pt_test,
    )
    _STAGGERED_AVAILABLE = True
except ImportError:
    _STAGGERED_AVAILABLE = False


# =============================================================================
# Enumerations
# =============================================================================

class TrendTestMethod(Enum):
    """
    Method for testing the parallel trends assumption.

    Attributes
    ----------
    VISUAL : str
        Visual inspection of pre-treatment trajectories.
    REGRESSION : str
        Formal regression-based test for trend differences.
    PLACEBO : str
        Estimate pre-treatment ATTs using rolling transformation.
    JOINT : str
        Combine placebo and regression tests.
    """

    VISUAL = "visual"
    REGRESSION = "regression"
    PLACEBO = "placebo"
    JOINT = "joint"


class TransformationMethod(Enum):
    """
    Unit-specific transformation methods for panel data.

    Attributes
    ----------
    DEMEAN : str
        Subtract pre-treatment mean from post-treatment outcomes.
    DETREND : str
        Remove unit-specific linear trend using pre-treatment periods.
    DEMEANQ : str
        Quarterly demeaning with seasonal fixed effects.
    DETRENDQ : str
        Quarterly detrending with seasonal effects and trends.
    """

    DEMEAN = "demean"
    DETREND = "detrend"
    DEMEANQ = "demeanq"
    DETRENDQ = "detrendq"


class RecommendationConfidence(Enum):
    """
    Confidence level for transformation method recommendation.

    Attributes
    ----------
    HIGH : str
        Confidence score above 0.8.
    MEDIUM : str
        Confidence score between 0.5 and 0.8.
    LOW : str
        Confidence score below 0.5.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PreTrendEstimate:
    """
    Pre-treatment ATT estimate for a single event time.

    Stores the estimated treatment effect for a pre-treatment period, used
    for placebo tests and parallel trends assessment. Under the null
    hypothesis of parallel trends, these estimates should be statistically
    indistinguishable from zero.

    Attributes
    ----------
    event_time : int
        Event time relative to treatment onset (negative for pre-treatment).
    cohort : int or None
        Treatment cohort identifier for staggered adoption designs.
    att : float
        Estimated average treatment effect on the treated.
    se : float
        Standard error of the ATT estimate.
    t_stat : float
        t-statistic computed as att / se.
    pvalue : float
        Two-sided p-value for testing H0: ATT = 0.
    ci_lower : float
        Lower bound of the confidence interval.
    ci_upper : float
        Upper bound of the confidence interval.
    n_treated : int
        Number of treated units in the estimation sample.
    n_control : int
        Number of control units in the estimation sample.
    df : int
        Degrees of freedom for t-distribution inference.
    """
    event_time: int
    cohort: int | None
    att: float
    se: float
    t_stat: float
    pvalue: float
    ci_lower: float
    ci_upper: float
    n_treated: int
    n_control: int
    df: int = 0

    @property
    def is_significant_05(self) -> bool:
        """Whether estimate is significant at 5% level."""
        return self.pvalue < 0.05

    @property
    def is_significant_10(self) -> bool:
        """Whether estimate is significant at 10% level."""
        return self.pvalue < 0.10


@dataclass
class CohortTrendEstimate:
    """
    Estimated linear trend for a single treatment cohort.

    Stores the pre-treatment linear trend estimate for a cohort, obtained
    by regressing outcomes on time for pre-treatment periods. Significant
    slopes indicate the presence of cohort-specific trends.

    Attributes
    ----------
    cohort : int
        Treatment cohort identifier (first treatment period).
    intercept : float
        Estimated intercept of the trend regression.
    intercept_se : float
        Standard error of the intercept estimate.
    slope : float
        Estimated slope representing the linear time trend.
    slope_se : float
        Standard error of the slope estimate.
    slope_pvalue : float
        Two-sided p-value for testing H0: slope = 0.
    n_units : int
        Number of units in this cohort.
    n_pre_periods : int
        Number of pre-treatment periods used in estimation.
    r_squared : float
        Coefficient of determination for the trend regression.
    residual_std : float
        Standard deviation of regression residuals.
    """
    cohort: int
    intercept: float
    intercept_se: float
    slope: float
    slope_se: float
    slope_pvalue: float
    n_units: int
    n_pre_periods: int
    r_squared: float
    residual_std: float = 0.0

    @property
    def has_significant_trend(self) -> bool:
        """Whether cohort has significant linear trend."""
        return self.slope_pvalue < 0.05


@dataclass
class TrendDifference:
    """
    Pairwise difference in linear trends between two cohorts.

    Stores the result of testing whether two cohorts have equal pre-treatment
    trends. Under the parallel trends assumption, all pairwise differences
    should be statistically indistinguishable from zero.

    Attributes
    ----------
    cohort_1 : int
        First cohort identifier.
    cohort_2 : int
        Second cohort identifier.
    slope_1 : float
        Estimated slope for the first cohort.
    slope_2 : float
        Estimated slope for the second cohort.
    slope_diff : float
        Difference in slopes (slope_1 - slope_2).
    slope_diff_se : float
        Standard error of the slope difference.
    t_stat : float
        t-statistic for testing equal slopes.
    pvalue : float
        Two-sided p-value for H0: slope_1 = slope_2.
    df : int
        Degrees of freedom for the test.
    """
    cohort_1: int
    cohort_2: int
    slope_1: float
    slope_2: float
    slope_diff: float
    slope_diff_se: float
    t_stat: float
    pvalue: float
    df: int

    @property
    def significant_at_05(self) -> bool:
        """Whether difference is significant at 5% level."""
        return self.pvalue < 0.05


@dataclass
class ParallelTrendsTestResult:
    """
    Results from testing the parallel trends assumption.

    Aggregates pre-treatment ATT estimates and joint test statistics to
    assess whether the parallel trends assumption is likely to hold.
    Includes a method recommendation based on the test outcome.

    Attributes
    ----------
    method : TrendTestMethod
        Testing method used (placebo, regression, visual, or joint).
    reject_null : bool
        Whether to reject H0 that parallel trends holds.
    pvalue : float
        P-value for the overall test.
    test_statistic : float
        Test statistic value (F-statistic for joint test).
    pre_trend_estimates : list of PreTrendEstimate
        Pre-treatment ATT estimates by event time.
    joint_f_stat : float or None
        F-statistic for the joint test H0: all pre-ATT = 0.
    joint_pvalue : float or None
        P-value for the joint F-test.
    joint_df : tuple of int
        Degrees of freedom (numerator, denominator) for the F-test.
    recommendation : str
        Recommended transformation method based on test results.
    recommendation_reason : str
        Explanation for the recommendation.
    figure : Any or None
        Pre-trends visualization figure object.
    warnings : list of str
        Warning messages from the testing procedure.
    """
    method: TrendTestMethod
    reject_null: bool
    pvalue: float
    test_statistic: float
    pre_trend_estimates: list[PreTrendEstimate]
    joint_f_stat: float | None = None
    joint_pvalue: float | None = None
    joint_df: tuple[int, int] = (0, 0)
    recommendation: str = "demean"
    recommendation_reason: str = ""
    figure: Any | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def n_significant_pre_trends(self) -> int:
        """Number of significant pre-treatment estimates at 5%."""
        return sum(1 for e in self.pre_trend_estimates if e.is_significant_05)

    @property
    def max_abs_pre_att(self) -> float:
        """Maximum absolute pre-treatment ATT."""
        if not self.pre_trend_estimates:
            return 0.0
        return max(abs(e.att) for e in self.pre_trend_estimates)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 70,
            "PARALLEL TRENDS TEST RESULTS",
            "=" * 70,
            "",
            f"Method: {self.method.value}",
            f"Test Statistic: {self.test_statistic:.4f}",
            f"P-value: {self.pvalue:.4f}",
            f"Reject H0 (Parallel Trends): {'YES ⚠️' if self.reject_null else 'NO ✓'}",
            "",
        ]

        if self.pre_trend_estimates:
            lines.append("Pre-treatment ATT Estimates:")
            lines.append("-" * 60)
            lines.append(f"{'Event Time':>12} {'ATT':>12} {'SE':>10} {'t-stat':>10} {'P-value':>10}")
            lines.append("-" * 60)
            for est in sorted(self.pre_trend_estimates, key=lambda x: x.event_time):
                sig = "**" if est.pvalue < 0.01 else ("*" if est.pvalue < 0.05 else "")
                lines.append(
                    f"{est.event_time:>12} {est.att:>12.4f} {est.se:>10.4f} "
                    f"{est.t_stat:>10.4f} {est.pvalue:>9.4f}{sig}"
                )
            lines.append("-" * 60)
            lines.append(f"Significant at 5%: {self.n_significant_pre_trends}/{len(self.pre_trend_estimates)}")

        if self.joint_f_stat is not None:
            lines.extend([
                "",
                f"Joint F-test (H0: all pre-ATT = 0):",
                f"  F({self.joint_df[0]}, {self.joint_df[1]}) = {self.joint_f_stat:.4f}",
                f"  P-value = {self.joint_pvalue:.4f}",
            ])

        lines.extend([
            "",
            "─" * 70,
            f"RECOMMENDATION: Use rolling='{self.recommendation}'",
            f"Reason: {self.recommendation_reason}",
            "─" * 70,
        ])

        if self.warnings:
            lines.extend(["", "WARNINGS:"])
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")

        lines.append("=" * 70)
        return "\n".join(lines)


@dataclass
class HeterogeneousTrendsDiagnostics:
    """
    Diagnostic results for heterogeneous trends across cohorts.

    Tests whether different treatment cohorts have different pre-treatment
    linear trends. Significant heterogeneity violates the standard parallel
    trends assumption but may be accommodated by detrending to remove
    cohort-specific trends under the CHT framework.

    Attributes
    ----------
    trend_by_cohort : list of CohortTrendEstimate
        Estimated linear trends for each treatment cohort.
    trend_heterogeneity_test : dict
        Results of the F-test for overall trend heterogeneity.
        Keys: 'f_stat', 'pvalue', 'df_num', 'df_den', 'reject_null'.
    trend_differences : list of TrendDifference
        Pairwise trend differences between all cohort pairs.
    control_group_trend : CohortTrendEstimate or None
        Trend estimate for the never-treated control group.
    has_heterogeneous_trends : bool
        Whether significant trend heterogeneity is detected.
    recommendation : str
        Recommended transformation method based on diagnostics.
    recommendation_confidence : float
        Confidence score for the recommendation (0 to 1).
    recommendation_reason : str
        Explanation for the recommendation.
    figure : Any or None
        Trend comparison visualization figure object.
    """
    trend_by_cohort: list[CohortTrendEstimate]
    trend_heterogeneity_test: dict[str, float]
    trend_differences: list[TrendDifference]
    control_group_trend: CohortTrendEstimate | None
    has_heterogeneous_trends: bool
    recommendation: str
    recommendation_confidence: float
    recommendation_reason: str
    figure: Any | None = None

    @property
    def n_cohorts(self) -> int:
        """Number of treatment cohorts analyzed."""
        return len(self.trend_by_cohort)

    @property
    def n_significant_differences(self) -> int:
        """Number of significant pairwise trend differences."""
        return sum(1 for d in self.trend_differences if d.significant_at_05)

    @property
    def max_trend_difference(self) -> float:
        """Maximum absolute trend difference."""
        if not self.trend_differences:
            return 0.0
        return max(abs(d.slope_diff) for d in self.trend_differences)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 70,
            "HETEROGENEOUS TRENDS DIAGNOSTICS",
            "=" * 70,
            "",
            f"Number of Cohorts Analyzed: {self.n_cohorts}",
            "",
            "Estimated Pre-treatment Trends by Cohort:",
            "-" * 60,
            f"{'Cohort':>8} {'Slope':>12} {'SE':>10} {'P-value':>10} {'N units':>8}",
            "-" * 60,
        ]

        for t in sorted(self.trend_by_cohort, key=lambda x: x.cohort):
            sig = "**" if t.slope_pvalue < 0.01 else ("*" if t.slope_pvalue < 0.05 else "")
            lines.append(
                f"{t.cohort:>8} {t.slope:>12.6f} {t.slope_se:>10.6f} "
                f"{t.slope_pvalue:>9.4f}{sig} {t.n_units:>8}"
            )

        if self.control_group_trend:
            t = self.control_group_trend
            lines.append(
                f"{'Control':>8} {t.slope:>12.6f} {t.slope_se:>10.6f} "
                f"{t.slope_pvalue:>9.4f} {t.n_units:>8}"
            )

        lines.extend([
            "",
            "Trend Heterogeneity Test (H0: all cohort trends are equal):",
            f"  F({self.trend_heterogeneity_test.get('df_num', 0)}, "
            f"{self.trend_heterogeneity_test.get('df_den', 0)}) = "
            f"{self.trend_heterogeneity_test.get('f_stat', 0):.4f}",
            f"  P-value = {self.trend_heterogeneity_test.get('pvalue', 1):.4f}",
            f"  Heterogeneous Trends Detected: "
            f"{'YES ⚠️' if self.has_heterogeneous_trends else 'NO ✓'}",
        ])

        if self.trend_differences:
            lines.extend([
                "",
                f"Significant Pairwise Differences: "
                f"{self.n_significant_differences}/{len(self.trend_differences)}",
                f"Maximum Trend Difference: {self.max_trend_difference:.6f}",
            ])

        lines.extend([
            "",
            "─" * 70,
            f"RECOMMENDATION: Use rolling='{self.recommendation}'",
            f"Confidence: {self.recommendation_confidence:.1%}",
            f"Reason: {self.recommendation_reason}",
            "─" * 70,
        ])

        lines.append("=" * 70)
        return "\n".join(lines)


@dataclass
class TransformationRecommendation:
    """
    Comprehensive recommendation for transformation method selection.

    Combines parallel trends test results, heterogeneous trends diagnostics,
    and data characteristics to provide an informed recommendation on
    whether to use demean, detrend, or their seasonal variants.

    Attributes
    ----------
    recommended_method : str
        Primary recommendation: 'demean', 'detrend', 'demeanq', or 'detrendq'.
    confidence : float
        Confidence score for the recommendation (0 to 1).
    confidence_level : RecommendationConfidence
        Categorical confidence level (HIGH, MEDIUM, or LOW).
    reasons : list of str
        List of reasons supporting the recommendation.
    parallel_trends_test : ParallelTrendsTestResult or None
        Results from the parallel trends test, if performed.
    heterogeneous_trends_diag : HeterogeneousTrendsDiagnostics or None
        Results from heterogeneous trends diagnostics, if performed.
    n_pre_periods_min : int
        Minimum number of pre-treatment periods across cohorts.
    n_pre_periods_max : int
        Maximum number of pre-treatment periods across cohorts.
    has_seasonal_pattern : bool
        Whether seasonal patterns are detected in the outcome.
    is_balanced_panel : bool
        Whether the panel is balanced.
    alternative_method : str or None
        Alternative recommendation if primary is not suitable.
    alternative_reason : str or None
        Explanation for the alternative recommendation.
    warnings : list of str
        Warning messages about data limitations or method constraints.
    """
    recommended_method: str
    confidence: float
    confidence_level: RecommendationConfidence
    reasons: list[str]
    parallel_trends_test: ParallelTrendsTestResult | None = None
    heterogeneous_trends_diag: HeterogeneousTrendsDiagnostics | None = None
    n_pre_periods_min: int = 0
    n_pre_periods_max: int = 0
    has_seasonal_pattern: bool = False
    is_balanced_panel: bool = True
    alternative_method: str | None = None
    alternative_reason: str | None = None
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 70,
            "TRANSFORMATION METHOD RECOMMENDATION",
            "=" * 70,
            "",
            f"Recommended Method: rolling='{self.recommended_method}'",
            f"Confidence: {self.confidence:.1%} ({self.confidence_level.value})",
            "",
            "Reasons:",
        ]
        for i, reason in enumerate(self.reasons, 1):
            lines.append(f"  {i}. {reason}")

        lines.extend([
            "",
            "Data Characteristics:",
            f"  - Pre-treatment periods: {self.n_pre_periods_min}-{self.n_pre_periods_max}",
            f"  - Balanced panel: {'Yes' if self.is_balanced_panel else 'No'}",
            f"  - Seasonal patterns: {'Detected' if self.has_seasonal_pattern else 'Not detected'}",
        ])

        if self.alternative_method:
            lines.extend([
                "",
                f"Alternative: rolling='{self.alternative_method}'",
                f"  Reason: {self.alternative_reason}",
            ])

        if self.warnings:
            lines.extend(["", "WARNINGS:"])
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")

        lines.append("=" * 70)
        return "\n".join(lines)


# =============================================================================
# Helper Functions
# =============================================================================

def _get_valid_cohorts(
    data: pd.DataFrame,
    gvar: str,
    ivar: str,
    never_treated_values: list | None = None,
) -> list[int]:
    """
    Get list of valid treatment cohorts from data.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data.
    gvar : str
        Cohort variable column name.
    ivar : str
        Unit identifier column name.
    never_treated_values : list, optional
        Values indicating never-treated units.

    Returns
    -------
    list of int
        Sorted list of valid treatment cohort values.
    """
    if never_treated_values is None:
        never_treated_values = [0, np.inf]

    # Extract unique cohort values, excluding missing.
    cohort_values = data[gvar].dropna().unique()

    # Keep only cohorts with valid treatment timing.
    valid_cohorts = [
        int(g) for g in cohort_values
        if g not in never_treated_values and not np.isinf(g)
    ]

    return sorted(valid_cohorts)


def _compute_pre_period_range(
    data: pd.DataFrame,
    tvar: str,
    gvar: str,
    never_treated_values: list | None = None,
) -> tuple[int, int]:
    """
    Compute the range of pre-treatment periods across cohorts.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data.
    tvar : str
        Time variable column name.
    gvar : str
        Cohort variable column name.
    never_treated_values : list, optional
        Values indicating never-treated units.

    Returns
    -------
    tuple of int
        (min_pre_periods, max_pre_periods) across all cohorts.
    """
    if never_treated_values is None:
        never_treated_values = [0, np.inf]

    T_min = int(data[tvar].min())

    cohorts = _get_valid_cohorts(data, gvar, 'unit', never_treated_values)

    if not cohorts:
        return 0, 0

    pre_periods = []
    for g in cohorts:
        n_pre = g - T_min
        if n_pre > 0:
            pre_periods.append(n_pre)

    if not pre_periods:
        return 0, 0

    return min(pre_periods), max(pre_periods)


def _check_panel_balance(
    data: pd.DataFrame,
    ivar: str,
    tvar: str,
) -> bool:
    """
    Check if panel is balanced.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data.
    ivar : str
        Unit identifier column name.
    tvar : str
        Time variable column name.

    Returns
    -------
    bool
        True if panel is balanced (all units observed in all periods).
    """
    n_units = data[ivar].nunique()
    n_periods = data[tvar].nunique()
    expected_obs = n_units * n_periods

    return len(data) == expected_obs


def _detect_seasonal_patterns(
    data: pd.DataFrame,
    y: str,
    ivar: str,
    tvar: str,
    threshold: float = 0.1,
) -> bool:
    """
    Detect seasonal patterns in the outcome variable.

    Uses autocorrelation at common seasonal lags (4, 12) to detect
    quarterly or monthly seasonality.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data.
    y : str
        Outcome variable column name.
    ivar : str
        Unit identifier column name.
    tvar : str
        Time variable column name.
    threshold : float, default=0.1
        Minimum autocorrelation to consider as seasonal.

    Returns
    -------
    bool
        True if seasonal patterns are detected.
    """
    try:
        # Compute average outcome by time period
        time_means = data.groupby(tvar)[y].mean()

        if len(time_means) < 8:
            return False

        # Check autocorrelation at lag 4 (quarterly) and lag 12 (monthly)
        values = time_means.values
        n = len(values)

        for lag in [4, 12]:
            if n > lag * 2:
                # Compute autocorrelation
                mean_val = np.mean(values)
                var_val = np.var(values)
                if var_val > 0:
                    autocorr = np.mean(
                        (values[:-lag] - mean_val) * (values[lag:] - mean_val)
                    ) / var_val
                    if abs(autocorr) > threshold:
                        return True

        return False
    except Exception:
        return False


# =============================================================================
# Core Functions - Trend Estimation
# =============================================================================

def _estimate_cohort_trend(
    data: pd.DataFrame,
    y: str,
    ivar: str,
    tvar: str,
    controls: list[str] | None = None,
) -> CohortTrendEstimate:
    """
    Estimate linear trend for a cohort using pooled OLS.

    Model: Y_it = α + β*t + X_i'γ + ε_it

    Parameters
    ----------
    data : pd.DataFrame
        Pre-treatment data for the cohort.
    y : str
        Outcome variable column name.
    ivar : str
        Unit identifier column name.
    tvar : str
        Time variable column name.
    controls : list of str, optional
        Control variable column names.

    Returns
    -------
    CohortTrendEstimate
        Estimated trend with slope, SE, and diagnostics.
    """
    # Prepare data
    y_vals = data[y].dropna().values
    t_vals = data.loc[data[y].notna(), tvar].values

    if len(y_vals) < 3:
        # Not enough data for trend estimation
        return CohortTrendEstimate(
            cohort=0,
            intercept=np.nan,
            intercept_se=np.nan,
            slope=np.nan,
            slope_se=np.nan,
            slope_pvalue=1.0,
            n_units=data[ivar].nunique(),
            n_pre_periods=data[tvar].nunique(),
            r_squared=0.0,
            residual_std=np.nan,
        )

    # Center time to reduce numerical instability in matrix inversion.
    t_mean = np.mean(t_vals)
    t_centered = t_vals - t_mean

    # Construct design matrix with intercept, centered time, and controls.
    X = np.column_stack([np.ones(len(t_centered)), t_centered])

    if controls:
        for c in controls:
            if c in data.columns:
                c_vals = data.loc[data[y].notna(), c].values
                X = np.column_stack([X, c_vals])

    # OLS estimation
    try:
        # (X'X)^{-1} X'y
        XtX = X.T @ X
        XtX_inv = np.linalg.inv(XtX)
        beta_hat = XtX_inv @ (X.T @ y_vals)

        # Residuals and variance
        y_hat = X @ beta_hat
        residuals = y_vals - y_hat
        n = len(y_vals)
        k = X.shape[1]
        df = n - k

        if df > 0:
            sigma2 = np.sum(residuals ** 2) / df
            var_beta = sigma2 * XtX_inv
            se_beta = np.sqrt(np.diag(var_beta))
        else:
            sigma2 = np.nan
            se_beta = np.full(k, np.nan)

        # Extract intercept and slope
        intercept = beta_hat[0]
        intercept_se = se_beta[0]
        slope = beta_hat[1]
        slope_se = se_beta[1]

        # T-test for slope
        if slope_se > 0 and df > 0:
            t_stat = slope / slope_se
            slope_pvalue = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        else:
            slope_pvalue = 1.0

        # R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_vals - np.mean(y_vals)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return CohortTrendEstimate(
            cohort=0,  # Will be set by caller
            intercept=intercept,
            intercept_se=intercept_se,
            slope=slope,
            slope_se=slope_se,
            slope_pvalue=slope_pvalue,
            n_units=data[ivar].nunique(),
            n_pre_periods=data[tvar].nunique(),
            r_squared=r_squared,
            residual_std=np.sqrt(sigma2) if not np.isnan(sigma2) else np.nan,
        )

    except np.linalg.LinAlgError:
        return CohortTrendEstimate(
            cohort=0,
            intercept=np.nan,
            intercept_se=np.nan,
            slope=np.nan,
            slope_se=np.nan,
            slope_pvalue=1.0,
            n_units=data[ivar].nunique(),
            n_pre_periods=data[tvar].nunique(),
            r_squared=0.0,
            residual_std=np.nan,
        )


def _test_trend_heterogeneity(
    data: pd.DataFrame,
    y: str,
    ivar: str,
    tvar: str,
    gvar: str,
    controls: list[str] | None,
    never_treated_values: list,
    alpha: float,
) -> dict[str, float]:
    """
    Test for trend heterogeneity across cohorts using F-test.

    Model: Y_it = α + β*t + Σ_g γ_g*(D_g*t) + X'δ + ε_it

    H0: γ_S = γ_{S+1} = ... = γ_T = 0 (all cohort trends equal)

    Parameters
    ----------
    data : pd.DataFrame
        Panel data.
    y : str
        Outcome variable column name.
    ivar : str
        Unit identifier column name.
    tvar : str
        Time variable column name.
    gvar : str
        Cohort variable column name.
    controls : list of str, optional
        Control variable column names.
    never_treated_values : list
        Values indicating never-treated units.
    alpha : float
        Significance level.

    Returns
    -------
    dict
        Test results with keys: f_stat, pvalue, df_num, df_den, reject_null
    """
    # Get cohorts
    cohorts = sorted([
        g for g in data[gvar].dropna().unique()
        if g not in never_treated_values and not np.isinf(g)
    ])

    if len(cohorts) < 2:
        return {
            'f_stat': 0.0,
            'pvalue': 1.0,
            'df_num': 0,
            'df_den': 0,
            'reject_null': False,
        }

    # Prepare pre-treatment data
    pre_data = data.copy()

    # Keep only pre-treatment observations
    def is_pre_treatment(row):
        g = row[gvar]
        t = row[tvar]
        if pd.isna(g) or g in never_treated_values or np.isinf(g):
            return True  # Control units - keep all
        return t < g

    pre_mask = pre_data.apply(is_pre_treatment, axis=1)
    pre_data = pre_data[pre_mask].copy()

    if len(pre_data) < 10:
        return {
            'f_stat': 0.0,
            'pvalue': 1.0,
            'df_num': 0,
            'df_den': 0,
            'reject_null': False,
        }

    # Build design matrices
    y_vals = pre_data[y].values
    t_vals = pre_data[tvar].values
    t_mean = t_vals.mean()
    t_centered = t_vals - t_mean

    # Restricted model: constant + time
    X_restricted = np.column_stack([np.ones(len(t_centered)), t_centered])

    # Full model: add cohort-time interactions
    X_full = X_restricted.copy()
    for g in cohorts[1:]:  # First cohort is reference
        cohort_indicator = (pre_data[gvar] == g).astype(float).values
        interaction = cohort_indicator * t_centered
        X_full = np.column_stack([X_full, interaction])

    # Add controls if specified
    if controls:
        for c in controls:
            if c in pre_data.columns:
                c_vals = pre_data[c].values
                X_restricted = np.column_stack([X_restricted, c_vals])
                X_full = np.column_stack([X_full, c_vals])

    # Fit models
    try:
        # Restricted model
        XtX_r = X_restricted.T @ X_restricted
        beta_r = np.linalg.solve(XtX_r, X_restricted.T @ y_vals)
        resid_r = y_vals - X_restricted @ beta_r
        ssr_restricted = np.sum(resid_r ** 2)

        # Full model
        XtX_f = X_full.T @ X_full
        beta_f = np.linalg.solve(XtX_f, X_full.T @ y_vals)
        resid_f = y_vals - X_full @ beta_f
        ssr_full = np.sum(resid_f ** 2)

        # F-test
        df_num = len(cohorts) - 1  # Number of restrictions
        df_den = len(y_vals) - X_full.shape[1]

        if ssr_full > 0 and df_den > 0:
            f_stat = ((ssr_restricted - ssr_full) / df_num) / (ssr_full / df_den)
            pvalue = 1 - stats.f.cdf(f_stat, df_num, df_den)
        else:
            f_stat = 0.0
            pvalue = 1.0

        return {
            'f_stat': f_stat,
            'pvalue': pvalue,
            'df_num': df_num,
            'df_den': df_den,
            'reject_null': pvalue < alpha,
        }

    except np.linalg.LinAlgError:
        return {
            'f_stat': 0.0,
            'pvalue': 1.0,
            'df_num': 0,
            'df_den': 0,
            'reject_null': False,
        }


def _compute_pairwise_trend_differences(
    trend_by_cohort: list[CohortTrendEstimate],
    control_group_trend: CohortTrendEstimate | None,
    alpha: float,
) -> list[TrendDifference]:
    """
    Compute pairwise trend differences between cohorts.

    Parameters
    ----------
    trend_by_cohort : list of CohortTrendEstimate
        Trend estimates for each cohort.
    control_group_trend : CohortTrendEstimate, optional
        Trend estimate for control group.
    alpha : float
        Significance level.

    Returns
    -------
    list of TrendDifference
        Pairwise trend differences.
    """
    differences = []

    # Include control group for comprehensive comparison.
    all_trends = list(trend_by_cohort)
    if control_group_trend is not None:
        all_trends.append(control_group_trend)

    # Test equal trends for all cohort pairs.
    for i, t1 in enumerate(all_trends):
        for t2 in all_trends[i + 1:]:
            if np.isnan(t1.slope) or np.isnan(t2.slope):
                continue
            if np.isnan(t1.slope_se) or np.isnan(t2.slope_se):
                continue
            if t1.slope_se <= 0 or t2.slope_se <= 0:
                continue

            slope_diff = t1.slope - t2.slope

            # Compute SE assuming independent cohort samples.
            slope_diff_se = np.sqrt(t1.slope_se ** 2 + t2.slope_se ** 2)

            # Perform two-sample t-test for trend equality.
            if slope_diff_se > 0:
                t_stat = slope_diff / slope_diff_se
                # Welch-Satterthwaite approximation accounts for unequal variances.
                df = (t1.slope_se ** 2 + t2.slope_se ** 2) ** 2 / (
                    t1.slope_se ** 4 / max(t1.n_units - 1, 1) +
                    t2.slope_se ** 4 / max(t2.n_units - 1, 1)
                )
                df = max(int(df), 1)
                pvalue = 2 * (1 - stats.t.cdf(abs(t_stat), df))
            else:
                t_stat = 0.0
                pvalue = 1.0
                df = 1

            differences.append(TrendDifference(
                cohort_1=t1.cohort,
                cohort_2=t2.cohort,
                slope_1=t1.slope,
                slope_2=t2.slope,
                slope_diff=slope_diff,
                slope_diff_se=slope_diff_se,
                t_stat=t_stat,
                pvalue=pvalue,
                df=df,
            ))

    return differences


# =============================================================================
# Core Functions - Placebo Test
# =============================================================================

def _compute_joint_f_test(
    estimates: list[PreTrendEstimate],
) -> tuple[float, float, tuple[int, int]]:
    """
    Compute joint F-test for H0: all pre-treatment ATTs = 0.

    Uses Wald test statistic: W = θ' * V^{-1} * θ
    where θ is vector of ATT estimates and V is covariance matrix.

    Under H0, W ~ χ²(k) or equivalently W/k ~ F(k, ∞).

    Parameters
    ----------
    estimates : list of PreTrendEstimate
        Pre-treatment ATT estimates.

    Returns
    -------
    tuple
        (f_stat, pvalue, (df_num, df_den))
    """
    if not estimates:
        return 0.0, 1.0, (0, 0)

    # Exclude estimates with invalid ATT or SE values.
    valid_estimates = [
        e for e in estimates
        if not np.isnan(e.att) and not np.isnan(e.se) and e.se > 0
    ]

    if not valid_estimates:
        return 0.0, 1.0, (0, 0)

    k = len(valid_estimates)
    atts = np.array([e.att for e in valid_estimates])
    ses = np.array([e.se for e in valid_estimates])

    # Diagonal covariance assumes independence across event times.
    # This yields conservative inference if estimates are positively correlated.
    var_diag = ses ** 2

    # Compute Wald statistic as sum of squared standardized effects.
    wald_stat = np.sum(atts ** 2 / var_diag)

    # Convert to F-statistic using minimum df for conservative inference.
    dfs = [e.df for e in valid_estimates if e.df > 0]
    df_den = min(dfs) if dfs else 100
    f_stat = wald_stat / k
    pvalue = 1 - stats.f.cdf(f_stat, k, df_den)

    return f_stat, pvalue, (k, df_den)


def _create_placebo_dataset(
    data: pd.DataFrame,
    y: str,
    ivar: str,
    tvar: str,
    gvar: str,
    cohort: int,
    placebo_period: int,
) -> pd.DataFrame:
    """
    Create dataset for placebo treatment effect estimation.

    For cohort g and placebo period t (where t < g-1):
    - "Treated" units: those in cohort g
    - "Control" units: those not yet treated by period t
    - Pre-treatment: periods < t
    - Post-treatment: period t

    Parameters
    ----------
    data : pd.DataFrame
        Original panel data.
    y : str
        Outcome variable column name.
    ivar : str
        Unit identifier column name.
    tvar : str
        Time variable column name.
    gvar : str
        Cohort variable column name.
    cohort : int
        Treatment cohort.
    placebo_period : int
        Placebo treatment period.

    Returns
    -------
    pd.DataFrame
        Dataset for placebo estimation with 'post_placebo' indicator.
    """
    # Select relevant periods: up to placebo_period
    placebo_data = data[data[tvar] <= placebo_period].copy()

    # Create placebo post indicator
    placebo_data['post_placebo'] = (placebo_data[tvar] == placebo_period).astype(int)

    # Create treatment indicator for this cohort
    placebo_data['treat_cohort'] = (placebo_data[gvar] == cohort).astype(int)

    return placebo_data


def _estimate_placebo_att(
    placebo_data: pd.DataFrame,
    y: str,
    ivar: str,
    tvar: str,
    controls: list[str] | None,
    estimator: str,
    n_bootstrap: int,
) -> tuple[float, float, int, int, int]:
    """
    Estimate placebo ATT using simple two-by-two DiD.

    Computes the difference-in-differences estimate using cell means. This
    approach produces conservative standard errors that include unit fixed
    effect variation. The rolling transformation approach accounts for the
    panel structure more accurately.

    Parameters
    ----------
    placebo_data : pd.DataFrame
        Placebo dataset with 'post_placebo' and 'treat_cohort' columns.
    y : str
        Outcome variable column name.
    ivar : str
        Unit identifier column name.
    tvar : str
        Time variable column name.
    controls : list of str, optional
        Control variable column names (not used in simple DiD).
    estimator : str
        Estimation method identifier (not used in simple DiD).
    n_bootstrap : int
        Number of bootstrap replications (not used in simple DiD).

    Returns
    -------
    tuple
        (att, se, n_treated, n_control, df) where att is the estimated
        treatment effect, se is the standard error, n_treated and n_control
        are sample sizes, and df is degrees of freedom.
    """
    try:
        # Simple 2x2 DiD estimation
        # ATT = (Y_treat_post - Y_treat_pre) - (Y_control_post - Y_control_pre)

        treat_mask = placebo_data['treat_cohort'] == 1
        post_mask = placebo_data['post_placebo'] == 1

        # Treated group means
        y_treat_post = placebo_data.loc[treat_mask & post_mask, y].mean()
        y_treat_pre = placebo_data.loc[treat_mask & ~post_mask, y].mean()

        # Control group means
        y_control_post = placebo_data.loc[~treat_mask & post_mask, y].mean()
        y_control_pre = placebo_data.loc[~treat_mask & ~post_mask, y].mean()

        # DiD estimate
        att = (y_treat_post - y_treat_pre) - (y_control_post - y_control_pre)

        # Sample sizes
        n_treated = treat_mask.sum() // 2  # Approximate (pre + post)
        n_control = (~treat_mask).sum() // 2

        # Standard error using pooled variance
        n_total = len(placebo_data)
        if n_total > 4:
            # Compute variance of each cell
            var_treat_post = placebo_data.loc[treat_mask & post_mask, y].var()
            var_treat_pre = placebo_data.loc[treat_mask & ~post_mask, y].var()
            var_control_post = placebo_data.loc[~treat_mask & post_mask, y].var()
            var_control_pre = placebo_data.loc[~treat_mask & ~post_mask, y].var()

            n_tp = (treat_mask & post_mask).sum()
            n_tr = (treat_mask & ~post_mask).sum()
            n_cp = (~treat_mask & post_mask).sum()
            n_cr = (~treat_mask & ~post_mask).sum()

            # SE of DiD
            se_sq = 0
            if n_tp > 0:
                se_sq += var_treat_post / n_tp if not np.isnan(var_treat_post) else 0
            if n_tr > 0:
                se_sq += var_treat_pre / n_tr if not np.isnan(var_treat_pre) else 0
            if n_cp > 0:
                se_sq += var_control_post / n_cp if not np.isnan(var_control_post) else 0
            if n_cr > 0:
                se_sq += var_control_pre / n_cr if not np.isnan(var_control_pre) else 0

            se = np.sqrt(se_sq) if se_sq > 0 else np.nan
            df = n_total - 4
        else:
            se = np.nan
            df = 0

        return att, se, n_treated, n_control, df

    except Exception:
        return np.nan, np.nan, 0, 0, 0


def _estimate_placebo_with_rolling_transformation(
    data: pd.DataFrame,
    y: str,
    ivar: str,
    tvar: str,
    gvar: str,
    controls: list[str] | None,
    estimator: str,
    rolling: str,
    never_treated_values: list,
    alpha: float,
    warnings_list: list[str],
) -> list[PreTrendEstimate]:
    """
    Estimate pre-treatment ATTs using rolling transformation.

    Applies rolling demeaning or detrending to pre-treatment periods to
    estimate placebo treatment effects. The rolling transformation uses
    future pre-treatment periods as the baseline, which properly accounts
    for the panel structure and produces correct standard errors.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data.
    y : str
        Outcome variable column name.
    ivar : str
        Unit identifier column name.
    tvar : str
        Time variable column name.
    gvar : str
        Cohort variable column name.
    controls : list of str, optional
        Control variable column names.
    estimator : str
        Estimation method: 'ra', 'ipwra', 'psm'.
    rolling : str
        Rolling transformation: 'demean' or 'detrend'.
    never_treated_values : list
        Values indicating never-treated units.
    alpha : float
        Significance level.
    warnings_list : list
        List to append warnings to.

    Returns
    -------
    list of PreTrendEstimate
        Pre-treatment ATT estimates.
    """
    pre_trend_estimates = []

    try:
        # Step 1: Apply rolling transformation to pre-treatment periods.
        # Demeaning subtracts pre-treatment means; detrending removes linear trends.
        if rolling == 'demean':
            data_transformed = transform_staggered_demean_pre(
                data, y, ivar, tvar, gvar, never_treated_values
            )
            transform_type = 'demean'
        else:
            data_transformed = transform_staggered_detrend_pre(
                data, y, ivar, tvar, gvar, never_treated_values
            )
            transform_type = 'detrend'

        # Step 2: Estimate pre-treatment effects using the transformed data
        pre_effects = estimate_pre_treatment_effects(
            data_transformed=data_transformed,
            gvar=gvar,
            ivar=ivar,
            tvar=tvar,
            controls=controls,
            vce=None,  # Use default variance estimation
            cluster_var=None,
            control_strategy='not_yet_treated',
            never_treated_values=never_treated_values,
            min_obs=3,
            min_treated=1,
            min_control=1,
            alpha=alpha,
            estimator=estimator,
            transform_type=transform_type,
        )

        # Step 3: Convert PreTreatmentEffect to PreTrendEstimate
        for effect in pre_effects:
            # Skip anchor points (they are 0 by construction)
            if effect.is_anchor:
                continue

            # Skip if SE is missing or zero
            if np.isnan(effect.se) or effect.se <= 0:
                continue

            pre_trend_estimates.append(PreTrendEstimate(
                event_time=effect.event_time,
                cohort=effect.cohort,
                att=effect.att,
                se=effect.se,
                t_stat=effect.t_stat,
                pvalue=effect.pvalue,
                ci_lower=effect.ci_lower,
                ci_upper=effect.ci_upper,
                n_treated=effect.n_treated,
                n_control=effect.n_control,
                df=effect.n_treated + effect.n_control - 2,
            ))

    except Exception as e:
        warnings_list.append(
            f"Rolling transformation approach failed: {e}. "
            f"This may indicate insufficient pre-treatment periods or data issues."
        )

    return pre_trend_estimates


def _estimate_placebo_with_simple_did(
    data: pd.DataFrame,
    y: str,
    ivar: str,
    tvar: str,
    gvar: str,
    controls: list[str] | None,
    estimator: str,
    never_treated_values: list,
    alpha: float,
    warnings_list: list[str],
    T_min: int,
    cohorts: list[int],
    n_bootstrap: int,
) -> list[PreTrendEstimate]:
    """
    Estimate pre-treatment ATTs using simple two-by-two DiD.

    Serves as a fallback when rolling transformation functions are unavailable.
    This approach uses simple cell-mean differences, which produces conservative
    standard errors that include unit fixed effect variation. The rolling
    transformation approach is preferred when available.

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
    gvar : str
        Cohort variable column name.
    controls : list of str, optional
        Control variable column names.
    estimator : str
        Estimation method identifier.
    never_treated_values : list
        Values indicating never-treated units.
    alpha : float
        Significance level for confidence intervals.
    warnings_list : list
        List to append warning messages to.
    T_min : int
        Minimum time period in the data.
    cohorts : list of int
        List of treatment cohort identifiers.
    n_bootstrap : int
        Number of bootstrap replications for SE estimation.

    Returns
    -------
    list of PreTrendEstimate
        Pre-treatment ATT estimates for each cohort and event time.
    """
    pre_trend_estimates = []

    for g in cohorts:
        # Need at least 2 pre-treatment periods for placebo test
        n_pre = g - T_min
        if n_pre < 2:
            warnings_list.append(
                f"Cohort {g} has only {n_pre} pre-treatment period(s). "
                f"Skipping placebo test for this cohort."
            )
            continue

        # For each potential placebo period
        for t in range(T_min + 1, g - 1):
            event_time = t - g  # Negative for pre-treatment

            # Create placebo dataset
            placebo_data = _create_placebo_dataset(
                data, y, ivar, tvar, gvar, g, t
            )

            # Estimate placebo ATT using simple DiD
            att, se, n_treated, n_control, df = _estimate_placebo_att(
                placebo_data, y, ivar, tvar, controls, estimator, n_bootstrap
            )

            if not np.isnan(att) and not np.isnan(se) and se > 0:
                t_stat = att / se
                pvalue = 2 * (1 - stats.t.cdf(abs(t_stat), max(df, 1)))
                t_crit = stats.t.ppf(1 - alpha / 2, max(df, 1))
                ci_lower = att - t_crit * se
                ci_upper = att + t_crit * se

                pre_trend_estimates.append(PreTrendEstimate(
                    event_time=event_time,
                    cohort=g,
                    att=att,
                    se=se,
                    t_stat=t_stat,
                    pvalue=pvalue,
                    ci_lower=ci_lower,
                    ci_upper=ci_upper,
                    n_treated=n_treated,
                    n_control=n_control,
                    df=df,
                ))

    return pre_trend_estimates


def _validate_trend_test_inputs(
    data: pd.DataFrame,
    y: str,
    ivar: str,
    tvar: str,
    gvar: str | None,
    method: str,
) -> None:
    """
    Validate inputs for parallel trends test.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data to validate.
    y : str
        Outcome variable column name.
    ivar : str
        Unit identifier column name.
    tvar : str
        Time variable column name.
    gvar : str or None
        Cohort variable column name.
    method : str
        Testing method to validate.

    Raises
    ------
    ValueError
        If required columns are missing or method is invalid.
    """
    required_cols = [y, ivar, tvar]
    if gvar is not None:
        required_cols.append(gvar)

    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    valid_methods = ['placebo', 'regression', 'visual', 'joint']
    if method not in valid_methods:
        raise ValueError(f"Unknown method: {method}. Must be one of {valid_methods}")


# =============================================================================
# Public API Functions
# =============================================================================

def test_parallel_trends(
    data: pd.DataFrame,
    y: str,
    ivar: str,
    tvar: str,
    gvar: str | None = None,
    controls: list[str] | None = None,
    method: str = 'placebo',
    estimator: str = 'ra',
    alpha: float = 0.05,
    n_bootstrap: int = 0,
    never_treated_values: list | None = None,
    rolling: str = 'demean',
    verbose: bool = True,
) -> ParallelTrendsTestResult:
    """
    Test the parallel trends assumption.

    Estimates placebo treatment effects in pre-treatment periods to assess
    whether the parallel trends assumption holds. Under the null hypothesis
    of parallel trends, all pre-treatment ATT estimates should be
    statistically indistinguishable from zero.

    This function uses rolling transformations (not simple 2x2 DiD) to
    properly estimate pre-treatment ATTs with correct standard errors
    that account for the panel structure.

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
        Cohort variable for staggered designs. If None, assumes common timing.
    controls : list of str, optional
        Control variable column names.
    method : str, default 'placebo'
        Testing method:
        - 'placebo': Estimate pre-treatment ATTs using rolling transformation
        - 'regression': Formal regression-based test for trend differences
        - 'visual': Generate pre-trends plot only
        - 'joint': Combine placebo and regression tests
    estimator : str, default 'ra'
        Estimator for ATT: 'ra', 'ipwra', 'psm'.
    alpha : float, default 0.05
        Significance level for hypothesis tests.
    n_bootstrap : int, default 0
        Number of bootstrap replications for SE. If 0, use analytical SE.
    never_treated_values : list, optional
        Values in gvar indicating never-treated units.
    rolling : str, default 'demean'
        Rolling transformation method: 'demean' or 'detrend'.
        Demeaning subtracts pre-treatment means; detrending removes
        unit-specific linear trends.
    verbose : bool, default True
        Whether to print summary.

    Returns
    -------
    ParallelTrendsTestResult
        Test results including pre-treatment estimates, joint test,
        and method recommendation.

    Notes
    -----
    The testing procedure:

    1. Apply rolling transformation (demeaning or detrending) to pre-treatment
       periods using future pre-treatment periods as the baseline.
    2. Estimate ATT for each pre-treatment event time using the transformed
       outcomes.
    3. Under H0 (parallel trends), all pre-treatment ATTs should be zero.
    4. Perform joint F-test for H0: all pre-treatment ATT = 0.

    The anchor point (event time e = -1) is set to 0 by construction and
    excluded from testing.

    See Also
    --------
    diagnose_heterogeneous_trends : Diagnose trend heterogeneity.
    recommend_transformation : Get method recommendation.
    """
    # Validate inputs
    _validate_trend_test_inputs(data, y, ivar, tvar, gvar, method)

    if never_treated_values is None:
        never_treated_values = [0, np.inf]

    warnings_list = []
    pre_trend_estimates = []

    # Get cohorts and time range
    if gvar is None:
        # Common timing case - infer treatment period
        T_min = int(data[tvar].min())
        T_max = int(data[tvar].max())
        treatment_period = (T_min + T_max) // 2
        cohorts = [treatment_period]
        warnings_list.append(
            f"No gvar specified. Assuming common timing with treatment at period {treatment_period}."
        )
    else:
        cohorts = _get_valid_cohorts(data, gvar, ivar, never_treated_values)

    if not cohorts:
        raise ValueError("No valid treatment cohorts found in data.")

    T_min = int(data[tvar].min())

    # For common timing without gvar, create a dummy gvar column
    if gvar is None:
        data = data.copy()
        unique_units = data[ivar].unique()
        n_units = len(unique_units)
        treated_units = set(unique_units[:n_units // 2])
        data['_gvar_dummy'] = data[ivar].apply(
            lambda x: treatment_period if x in treated_units else np.inf
        )
        gvar = '_gvar_dummy'

    # =========================================================================
    # Placebo Test Implementation using Rolling Transformation
    # =========================================================================
    if method in ('placebo', 'joint'):
        # Check if staggered module is available for proper implementation
        if _STAGGERED_AVAILABLE:
            # Use the correct rolling transformation approach
            pre_trend_estimates = _estimate_placebo_with_rolling_transformation(
                data=data,
                y=y,
                ivar=ivar,
                tvar=tvar,
                gvar=gvar,
                controls=controls,
                estimator=estimator,
                rolling=rolling,
                never_treated_values=never_treated_values,
                alpha=alpha,
                warnings_list=warnings_list,
            )
        else:
            # Fallback to simple 2x2 DiD (with warning)
            warnings_list.append(
                "Staggered module not available. Using simple 2x2 DiD for placebo test. "
                "This may produce conservative standard errors. For proper implementation, "
                "ensure the staggered module is importable."
            )
            pre_trend_estimates = _estimate_placebo_with_simple_did(
                data=data,
                y=y,
                ivar=ivar,
                tvar=tvar,
                gvar=gvar,
                controls=controls,
                estimator=estimator,
                never_treated_values=never_treated_values,
                alpha=alpha,
                warnings_list=warnings_list,
                T_min=T_min,
                cohorts=cohorts,
                n_bootstrap=n_bootstrap,
            )

    # =========================================================================
    # Compute Joint F-test
    # =========================================================================
    if pre_trend_estimates:
        joint_f_stat, joint_pvalue, joint_df = _compute_joint_f_test(pre_trend_estimates)
    else:
        joint_f_stat = np.nan
        joint_pvalue = 1.0
        joint_df = (0, 0)
        warnings_list.append(
            "No valid pre-treatment estimates computed. Cannot perform joint test."
        )

    # =========================================================================
    # Determine Recommendation
    # =========================================================================
    reject_null = joint_pvalue < alpha if not np.isnan(joint_pvalue) else False
    n_significant = sum(1 for e in pre_trend_estimates if e.pvalue < alpha)
    n_total = len(pre_trend_estimates)

    if reject_null or (n_total > 0 and n_significant > n_total * 0.2):
        recommendation = "detrend"
        recommendation_reason = (
            f"Parallel trends assumption appears violated: "
            f"joint F-test p={joint_pvalue:.4f}, "
            f"{n_significant}/{n_total} pre-treatment estimates significant. "
            f"Detrending removes unit-specific linear trends under Assumption CHT."
        )
    else:
        recommendation = "demean"
        recommendation_reason = (
            f"Parallel trends assumption appears to hold: "
            f"joint F-test p={joint_pvalue:.4f}, "
            f"{n_significant}/{n_total} pre-treatment estimates significant. "
            f"Demeaning is more efficient when PT holds."
        )

    result = ParallelTrendsTestResult(
        method=TrendTestMethod.PLACEBO if method == 'placebo' else TrendTestMethod.JOINT,
        reject_null=reject_null,
        pvalue=joint_pvalue,
        test_statistic=joint_f_stat,
        pre_trend_estimates=pre_trend_estimates,
        joint_f_stat=joint_f_stat,
        joint_pvalue=joint_pvalue,
        joint_df=joint_df,
        recommendation=recommendation,
        recommendation_reason=recommendation_reason,
        warnings=warnings_list,
    )

    if verbose:
        print(result.summary())

    return result


def diagnose_heterogeneous_trends(
    data: pd.DataFrame,
    y: str,
    ivar: str,
    tvar: str,
    gvar: str | None = None,
    controls: list[str] | None = None,
    never_treated_values: list | None = None,
    include_control_group: bool = True,
    alpha: float = 0.05,
    verbose: bool = True,
) -> HeterogeneousTrendsDiagnostics:
    """
    Diagnose heterogeneous trends across treatment cohorts.

    Estimates pre-treatment linear trends for each cohort and tests
    whether trends differ significantly. Under the conditional heterogeneous
    trends (CHT) assumption, different cohorts may have different linear
    trends, which can be removed by detrending.

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
        Cohort variable. If None, assumes common timing.
    controls : list of str, optional
        Control variable column names.
    never_treated_values : list, optional
        Values indicating never-treated units. Default: [0, np.inf].
    include_control_group : bool, default True
        Whether to include never-treated group in trend analysis.
    alpha : float, default 0.05
        Significance level for tests.
    verbose : bool, default True
        Whether to print summary.

    Returns
    -------
    HeterogeneousTrendsDiagnostics
        Diagnostic results including cohort trends, heterogeneity test,
        and method recommendation.

    Notes
    -----
    Under the CHT framework, the expected outcome in the never-treated state
    includes cohort-specific linear time trends. Each cohort g has its own
    trend coefficient, allowing for differential pre-treatment trajectories
    across cohorts.

    The heterogeneity test uses an F-test for the null hypothesis that
    all cohort trends are equal. Rejection suggests detrending may be
    appropriate to remove cohort-specific trends.

    See Also
    --------
    test_parallel_trends : Test parallel trends assumption.
    recommend_transformation : Get method recommendation.
    """
    if never_treated_values is None:
        never_treated_values = [0, np.inf]

    # Validate inputs
    required_cols = [y, ivar, tvar]
    if gvar is not None:
        required_cols.append(gvar)
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    trend_by_cohort = []
    T_min = int(data[tvar].min())

    # =========================================================================
    # Estimate Trends by Cohort
    # =========================================================================
    if gvar is not None:
        cohorts = _get_valid_cohorts(data, gvar, ivar, never_treated_values)
    else:
        # Common timing - treat all treated units as one cohort
        T_max = int(data[tvar].max())
        treatment_period = (T_min + T_max) // 2
        cohorts = [treatment_period]

    for g in cohorts:
        if gvar is not None:
            cohort_data = data[data[gvar] == g].copy()
        else:
            # For common timing, use all data
            cohort_data = data.copy()

        # Select pre-treatment periods
        pre_data = cohort_data[cohort_data[tvar] < g]

        if len(pre_data) < 3:
            # Not enough data for trend estimation
            continue

        # Estimate trend
        trend_est = _estimate_cohort_trend(pre_data, y, ivar, tvar, controls)
        trend_est = CohortTrendEstimate(
            cohort=int(g),
            intercept=trend_est.intercept,
            intercept_se=trend_est.intercept_se,
            slope=trend_est.slope,
            slope_se=trend_est.slope_se,
            slope_pvalue=trend_est.slope_pvalue,
            n_units=trend_est.n_units,
            n_pre_periods=trend_est.n_pre_periods,
            r_squared=trend_est.r_squared,
            residual_std=trend_est.residual_std,
        )
        trend_by_cohort.append(trend_est)

    # =========================================================================
    # Estimate Control Group Trend
    # =========================================================================
    control_group_trend = None
    if include_control_group and gvar is not None:
        # Get never-treated units
        control_mask = (
            data[gvar].isna() |
            data[gvar].isin(never_treated_values) |
            np.isinf(data[gvar])
        )
        control_data = data[control_mask]

        if len(control_data) >= 3:
            trend_est = _estimate_cohort_trend(control_data, y, ivar, tvar, controls)
            control_group_trend = CohortTrendEstimate(
                cohort=0,  # Use 0 for control
                intercept=trend_est.intercept,
                intercept_se=trend_est.intercept_se,
                slope=trend_est.slope,
                slope_se=trend_est.slope_se,
                slope_pvalue=trend_est.slope_pvalue,
                n_units=trend_est.n_units,
                n_pre_periods=trend_est.n_pre_periods,
                r_squared=trend_est.r_squared,
                residual_std=trend_est.residual_std,
            )

    # =========================================================================
    # Test for Trend Heterogeneity
    # =========================================================================
    if gvar is not None and len(trend_by_cohort) >= 2:
        trend_heterogeneity_test = _test_trend_heterogeneity(
            data, y, ivar, tvar, gvar, controls, never_treated_values, alpha
        )
    else:
        trend_heterogeneity_test = {
            'f_stat': 0.0,
            'pvalue': 1.0,
            'df_num': 0,
            'df_den': 0,
            'reject_null': False,
        }

    # =========================================================================
    # Compute Pairwise Trend Differences
    # =========================================================================
    trend_differences = _compute_pairwise_trend_differences(
        trend_by_cohort, control_group_trend, alpha
    )

    # =========================================================================
    # Determine Recommendation
    # =========================================================================
    has_heterogeneous_trends = trend_heterogeneity_test.get('reject_null', False)

    if has_heterogeneous_trends:
        recommendation = "detrend"
        confidence = min(0.95, 1 - trend_heterogeneity_test.get('pvalue', 1))
        recommendation_reason = (
            f"Significant trend heterogeneity detected (F-test p="
            f"{trend_heterogeneity_test.get('pvalue', 1):.4f}). "
            f"Detrending removes cohort-specific linear trends under Assumption CHT."
        )
    else:
        recommendation = "demean"
        confidence = trend_heterogeneity_test.get('pvalue', 0)
        recommendation_reason = (
            f"No significant trend heterogeneity detected (F-test p="
            f"{trend_heterogeneity_test.get('pvalue', 1):.4f}). "
            f"Demeaning is more efficient when trends are parallel."
        )

    result = HeterogeneousTrendsDiagnostics(
        trend_by_cohort=trend_by_cohort,
        trend_heterogeneity_test=trend_heterogeneity_test,
        trend_differences=trend_differences,
        control_group_trend=control_group_trend,
        has_heterogeneous_trends=has_heterogeneous_trends,
        recommendation=recommendation,
        recommendation_confidence=confidence,
        recommendation_reason=recommendation_reason,
    )

    if verbose:
        print(result.summary())

    return result


def recommend_transformation(
    data: pd.DataFrame,
    y: str,
    ivar: str,
    tvar: str,
    gvar: str | None = None,
    controls: list[str] | None = None,
    never_treated_values: list | None = None,
    run_all_diagnostics: bool = True,
    verbose: bool = True,
) -> TransformationRecommendation:
    """
    Recommend optimal transformation method based on data characteristics.

    Combines multiple diagnostic procedures to provide an informed
    recommendation on whether to use demean, detrend, or seasonal variants.

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
    controls : list of str, optional
        Control variable column names.
    never_treated_values : list, optional
        Values indicating never-treated units.
    run_all_diagnostics : bool, default True
        Whether to run all diagnostic tests. If False, uses heuristics only.
    verbose : bool, default True
        Whether to print summary.

    Returns
    -------
    TransformationRecommendation
        Recommendation with confidence level and supporting diagnostics.

    Notes
    -----
    The recommendation algorithm considers:

    1. **Data requirements**: detrend requires at least 2 pre-treatment periods
    2. **Parallel trends test**: If violated, recommend detrend
    3. **Trend heterogeneity**: If detected, recommend detrend
    4. **Panel balance**: Unbalanced panels favor detrend
    5. **Seasonal patterns**: If detected, recommend seasonal variants

    Decision tree::

        n_pre_periods < 2?
        |-- Yes -> demean (detrend not feasible)
        +-- No -> Run diagnostics
                |-- PT violated OR heterogeneous trends?
                |   |-- Yes -> detrend
                |   +-- No -> demean (more efficient)
                +-- Seasonal patterns?
                    |-- Yes -> demeanq/detrendq
                    +-- No -> demean/detrend

    See Also
    --------
    test_parallel_trends : Test parallel trends assumption.
    diagnose_heterogeneous_trends : Diagnose trend heterogeneity.
    """
    if never_treated_values is None:
        never_treated_values = [0, np.inf]

    warnings_list = []
    reasons = []

    # =========================================================================
    # Step 1: Check Data Requirements
    # =========================================================================
    n_pre_min, n_pre_max = _compute_pre_period_range(
        data, tvar, gvar if gvar else '_dummy', never_treated_values
    )

    if n_pre_min < 1:
        raise ValueError(
            "At least one pre-treatment period required for any transformation."
        )

    detrend_feasible = n_pre_min >= 2
    if not detrend_feasible:
        warnings_list.append(
            f"Detrending requires ≥2 pre-treatment periods. "
            f"Minimum found: {n_pre_min}. Only demeaning is feasible."
        )

    # =========================================================================
    # Step 2: Check Panel Balance
    # =========================================================================
    is_balanced = _check_panel_balance(data, ivar, tvar)
    if not is_balanced:
        reasons.append("Unbalanced panel detected - detrend is more robust to selection")

    # =========================================================================
    # Step 3: Check for Seasonal Patterns
    # =========================================================================
    has_seasonal = _detect_seasonal_patterns(data, y, ivar, tvar)

    # =========================================================================
    # Step 4: Run Diagnostics if Requested
    # =========================================================================
    pt_test_result = None
    ht_diag_result = None

    if run_all_diagnostics and gvar is not None:
        # Run parallel trends test
        try:
            pt_test_result = test_parallel_trends(
                data, y, ivar, tvar, gvar, controls,
                method='placebo', alpha=0.05, verbose=False,
                never_treated_values=never_treated_values,
            )
        except Exception as e:
            warnings_list.append(f"Parallel trends test failed: {e}")

        # Run heterogeneous trends diagnosis
        try:
            ht_diag_result = diagnose_heterogeneous_trends(
                data, y, ivar, tvar, gvar, controls,
                never_treated_values=never_treated_values,
                include_control_group=True, alpha=0.05, verbose=False,
            )
        except Exception as e:
            warnings_list.append(f"Heterogeneous trends diagnosis failed: {e}")

    # =========================================================================
    # Step 5: Combine Evidence
    # =========================================================================
    score_demean = 0.5  # Base score
    score_detrend = 0.5

    # Factor 1: PT test (weight 0.4)
    if pt_test_result is not None:
        if pt_test_result.reject_null:
            score_detrend += 0.4
            reasons.append(
                f"Parallel trends test rejected (p={pt_test_result.pvalue:.4f})"
            )
        else:
            score_demean += 0.4
            reasons.append(
                f"Parallel trends test not rejected (p={pt_test_result.pvalue:.4f})"
            )

    # Factor 2: Trend heterogeneity (weight 0.4)
    if ht_diag_result is not None:
        if ht_diag_result.has_heterogeneous_trends:
            score_detrend += 0.4
            reasons.append(
                f"Heterogeneous trends detected "
                f"(F-test p={ht_diag_result.trend_heterogeneity_test.get('pvalue', 1):.4f})"
            )
        else:
            score_demean += 0.4
            reasons.append(
                f"No heterogeneous trends detected "
                f"(F-test p={ht_diag_result.trend_heterogeneity_test.get('pvalue', 1):.4f})"
            )

    # Factor 3: Panel balance (weight 0.1)
    if not is_balanced:
        score_detrend += 0.1
    else:
        score_demean += 0.1

    # Factor 4: Data requirements (weight 0.1)
    if detrend_feasible:
        score_detrend += 0.05
        score_demean += 0.05
    else:
        score_demean += 0.1
        score_detrend = 0  # Not feasible

    # Normalize scores
    total = score_demean + score_detrend
    if total > 0:
        score_demean /= total
        score_detrend /= total

    # =========================================================================
    # Step 6: Determine Recommendation
    # =========================================================================
    if not detrend_feasible:
        recommended = "demean"
        confidence = 1.0
        reasons.insert(0, "Detrending not feasible (insufficient pre-treatment periods)")
    elif has_seasonal:
        if score_detrend > score_demean:
            recommended = "detrendq"
            confidence = score_detrend
        else:
            recommended = "demeanq"
            confidence = score_demean
        reasons.append("Seasonal patterns detected - using seasonal variant")
    elif score_detrend > score_demean:
        recommended = "detrend"
        confidence = score_detrend
    else:
        recommended = "demean"
        confidence = score_demean

    # Determine confidence level
    if confidence > 0.8:
        confidence_level = RecommendationConfidence.HIGH
    elif confidence > 0.5:
        confidence_level = RecommendationConfidence.MEDIUM
    else:
        confidence_level = RecommendationConfidence.LOW
        warnings_list.append(
            "Low confidence in recommendation. Consider running sensitivity "
            "analysis comparing demean and detrend results."
        )

    # Alternative recommendation
    if recommended in ("demean", "demeanq") and detrend_feasible:
        alternative = "detrend" if not has_seasonal else "detrendq"
        alternative_reason = "Use if parallel trends assumption is questionable"
    elif recommended in ("detrend", "detrendq"):
        alternative = "demean" if not has_seasonal else "demeanq"
        alternative_reason = "Use if parallel trends assumption is believed to hold (more efficient)"
    else:
        alternative = None
        alternative_reason = None

    result = TransformationRecommendation(
        recommended_method=recommended,
        confidence=confidence,
        confidence_level=confidence_level,
        reasons=reasons,
        parallel_trends_test=pt_test_result,
        heterogeneous_trends_diag=ht_diag_result,
        n_pre_periods_min=n_pre_min,
        n_pre_periods_max=n_pre_max,
        has_seasonal_pattern=has_seasonal,
        is_balanced_panel=is_balanced,
        alternative_method=alternative,
        alternative_reason=alternative_reason,
        warnings=warnings_list,
    )

    if verbose:
        print(result.summary())

    return result


def plot_cohort_trends(
    data: pd.DataFrame,
    y: str,
    ivar: str,
    tvar: str,
    gvar: str,
    controls: list[str] | None = None,
    never_treated_values: list | None = None,
    normalize: bool = True,
    normalize_period: int | None = None,
    show_treatment_lines: bool = True,
    show_trend_lines: bool = True,
    confidence_bands: bool = True,
    alpha: float = 0.05,
    figsize: tuple[float, float] = (12, 8),
    ax: Any | None = None,
) -> Any:
    """
    Visualize outcome trends by treatment cohort.

    Creates a plot showing average outcome trajectories for each cohort,
    with optional trend lines and treatment timing indicators.

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
    gvar : str
        Cohort variable indicating first treatment period.
    controls : list of str, optional
        Control variables for residualized outcomes.
    never_treated_values : list, optional
        Values indicating never-treated units.
    normalize : bool, default True
        Whether to normalize outcomes (subtract baseline).
    normalize_period : int, optional
        Period to use as baseline. Default: period before first treatment.
    show_treatment_lines : bool, default True
        Whether to show vertical lines at treatment timing.
    show_trend_lines : bool, default True
        Whether to show fitted linear trend lines.
    confidence_bands : bool, default True
        Whether to show confidence bands around means.
    alpha : float, default 0.05
        Significance level for confidence bands.
    figsize : tuple, default (12, 8)
        Figure size in inches.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the cohort trends plot.

    Notes
    -----
    This visualization helps assess:

    1. Whether pre-treatment trends are parallel across cohorts
    2. Whether treatment effects appear at the expected timing
    3. Whether trends are approximately linear (for detrending)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install matplotlib"
        )

    if never_treated_values is None:
        never_treated_values = [0, np.inf]

    # Get cohorts
    cohorts = _get_valid_cohorts(data, gvar, ivar, never_treated_values)

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Color palette
    colors = plt.cm.tab10.colors

    # Get time range
    T_min = int(data[tvar].min())
    T_max = int(data[tvar].max())
    time_range = range(T_min, T_max + 1)

    # Normalization period
    if normalize_period is None and cohorts:
        normalize_period = min(cohorts) - 1

    # Plot each cohort
    for i, g in enumerate(cohorts):
        color = colors[i % len(colors)]

        # Get cohort data
        cohort_data = data[data[gvar] == g]

        # Compute mean by time period
        means = cohort_data.groupby(tvar)[y].mean()
        stds = cohort_data.groupby(tvar)[y].std()
        counts = cohort_data.groupby(tvar)[y].count()

        # Normalize if requested
        if normalize and normalize_period in means.index:
            baseline = means[normalize_period]
            means = means - baseline

        # Standard error
        ses = stds / np.sqrt(counts)

        # Plot mean trajectory
        ax.plot(
            means.index, means.values,
            marker='o', color=color, linewidth=2,
            label=f'Cohort {g}'
        )

        # Confidence bands
        if confidence_bands:
            t_crit = stats.t.ppf(1 - alpha / 2, counts - 1)
            ci_lower = means - t_crit * ses
            ci_upper = means + t_crit * ses
            ax.fill_between(
                means.index, ci_lower, ci_upper,
                color=color, alpha=0.2
            )

        # Trend line (pre-treatment only)
        if show_trend_lines:
            pre_periods = [t for t in means.index if t < g]
            if len(pre_periods) >= 2:
                pre_means = means[pre_periods]
                # Fit linear trend
                X = np.column_stack([np.ones(len(pre_periods)), pre_periods])
                beta = np.linalg.lstsq(X, pre_means.values, rcond=None)[0]
                # Extend trend line
                trend_x = np.array(list(range(T_min, g + 2)))
                trend_y = beta[0] + beta[1] * trend_x
                if normalize and normalize_period is not None:
                    trend_y = trend_y - (beta[0] + beta[1] * normalize_period)
                ax.plot(
                    trend_x, trend_y,
                    linestyle='--', color=color, alpha=0.5, linewidth=1
                )

        # Treatment timing line
        if show_treatment_lines:
            ax.axvline(
                x=g - 0.5, color=color, linestyle=':',
                alpha=0.5, linewidth=1
            )

    # Plot control group
    control_mask = (
        data[gvar].isna() |
        data[gvar].isin(never_treated_values) |
        np.isinf(data[gvar])
    )
    control_data = data[control_mask]

    if len(control_data) > 0:
        means = control_data.groupby(tvar)[y].mean()
        if normalize and normalize_period in means.index:
            means = means - means[normalize_period]

        ax.plot(
            means.index, means.values,
            marker='s', color='gray', linewidth=2,
            linestyle='--', label='Never Treated'
        )

    # Labels and legend
    ax.set_xlabel('Time Period')
    ax.set_ylabel('Outcome' + (' (Normalized)' if normalize else ''))
    ax.set_title('Outcome Trends by Treatment Cohort')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Enums
    'TrendTestMethod',
    'TransformationMethod',
    'RecommendationConfidence',
    # Data classes
    'PreTrendEstimate',
    'CohortTrendEstimate',
    'TrendDifference',
    'ParallelTrendsTestResult',
    'HeterogeneousTrendsDiagnostics',
    'TransformationRecommendation',
    # Functions
    'test_parallel_trends',
    'diagnose_heterogeneous_trends',
    'recommend_transformation',
    'plot_cohort_trends',
]

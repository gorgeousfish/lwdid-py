"""
Sensitivity analysis for difference-in-differences estimation.

This module provides tools to assess the robustness of ATT estimates under
varying methodological choices and potential assumption violations. Three
types of sensitivity analysis are supported: pre-treatment period selection
(testing stability across different baseline period configurations),
no-anticipation assumption testing (evaluating robustness by excluding
periods immediately before treatment), and comprehensive analysis combining
multiple robustness checks including transformation method and estimator
comparisons.

Notes
-----
Results are classified into robustness levels based on the sensitivity ratio,
defined as the range of ATT estimates across specifications divided by the
absolute value of the baseline estimate. Thresholds for classification are:
highly robust (< 10%), moderately robust (10-25%), sensitive (25-50%), and
highly sensitive (>= 50%).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import warnings

import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# Enumerations
# =============================================================================

class RobustnessLevel(Enum):
    """
    Categorical assessment of estimate stability across specifications.

    The robustness level is determined by the sensitivity ratio, which measures
    the range of ATT estimates relative to the baseline estimate magnitude.

    Attributes
    ----------
    HIGHLY_ROBUST : str
        Sensitivity ratio below 10%. Estimates are very stable.
    MODERATELY_ROBUST : str
        Sensitivity ratio between 10% and 25%. Estimates show minor variation.
    SENSITIVE : str
        Sensitivity ratio between 25% and 50%. Estimates vary noticeably.
    HIGHLY_SENSITIVE : str
        Sensitivity ratio at or above 50%. Estimates are unstable.
    """
    HIGHLY_ROBUST = "highly_robust"
    MODERATELY_ROBUST = "moderately_robust"
    SENSITIVE = "sensitive"
    HIGHLY_SENSITIVE = "highly_sensitive"


class AnticipationDetectionMethod(Enum):
    """
    Detection method used to identify potential anticipation effects.

    Anticipation effects occur when units adjust behavior before formal
    treatment begins, violating the no-anticipation assumption.

    Attributes
    ----------
    TREND_BREAK : str
        Detected via structural break in pre-treatment trend.
    COEFFICIENT_CHANGE : str
        Detected via significant change in ATT when excluding periods.
    PLACEBO_TEST : str
        Detected via significant placebo effects in pre-treatment periods.
    NONE_DETECTED : str
        No anticipation effects identified by any method.
    INSUFFICIENT_DATA : str
        Insufficient pre-treatment periods to perform detection.
    """
    TREND_BREAK = "trend_break"
    COEFFICIENT_CHANGE = "coefficient_change"
    PLACEBO_TEST = "placebo_test"
    NONE_DETECTED = "none_detected"
    INSUFFICIENT_DATA = "insufficient_data"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SpecificationResult:
    """
    Result from a single specification in sensitivity analysis.
    
    Represents one point in the sensitivity analysis, corresponding to
    a specific configuration of pre-treatment periods.
    
    Attributes
    ----------
    specification_id : int
        Unique identifier for this specification.
    n_pre_periods : int
        Number of pre-treatment periods used.
    start_period : int
        First pre-treatment period included.
    end_period : int
        Last pre-treatment period included.
    excluded_periods : int
        Number of periods excluded before treatment.
    att : float
        Average treatment effect on the treated.
    se : float
        Standard error of ATT.
    t_stat : float
        t-statistic for H0: ATT=0.
    pvalue : float
        Two-sided p-value.
    ci_lower : float
        Lower bound of confidence interval.
    ci_upper : float
        Upper bound of confidence interval.
    n_treated : int
        Number of treated units.
    n_control : int
        Number of control units.
    df : int
        Degrees of freedom for inference.
    converged : bool
        Whether estimation converged successfully.
    spec_warnings : list[str]
        Warning messages from estimation.
    """
    specification_id: int
    n_pre_periods: int
    start_period: int
    end_period: int
    excluded_periods: int
    att: float
    se: float
    t_stat: float
    pvalue: float
    ci_lower: float
    ci_upper: float
    n_treated: int
    n_control: int
    df: int
    converged: bool = True
    spec_warnings: list[str] = field(default_factory=list)
    
    @property
    def is_significant_05(self) -> bool:
        """Whether estimate is significant at 5% level."""
        return self.pvalue < 0.05
    
    @property
    def is_significant_10(self) -> bool:
        """Whether estimate is significant at 10% level."""
        return self.pvalue < 0.10
    
    def to_dict(self) -> dict:
        """
        Convert specification result to dictionary for DataFrame construction.

        Returns
        -------
        dict
            Dictionary containing all specification attributes suitable for
            constructing a pandas DataFrame row.
        """
        return {
            'spec_id': self.specification_id,
            'n_pre_periods': self.n_pre_periods,
            'start_period': self.start_period,
            'end_period': self.end_period,
            'excluded_periods': self.excluded_periods,
            'att': self.att,
            'se': self.se,
            't_stat': self.t_stat,
            'pvalue': self.pvalue,
            'ci_lower': self.ci_lower,
            'ci_upper': self.ci_upper,
            'n_treated': self.n_treated,
            'n_control': self.n_control,
            'df': self.df,
            'significant_05': self.is_significant_05,
            'converged': self.converged,
        }


@dataclass
class AnticipationEstimate:
    """
    ATT estimate with specific anticipation exclusion.
    
    Attributes
    ----------
    excluded_periods : int
        Number of periods excluded before treatment.
    att : float
        Average treatment effect on the treated.
    se : float
        Standard error of ATT.
    t_stat : float
        t-statistic for H0: ATT=0.
    pvalue : float
        Two-sided p-value.
    ci_lower : float
        Lower bound of confidence interval.
    ci_upper : float
        Upper bound of confidence interval.
    n_pre_periods_used : int
        Number of pre-treatment periods actually used.
    """
    excluded_periods: int
    att: float
    se: float
    t_stat: float
    pvalue: float
    ci_lower: float
    ci_upper: float
    n_pre_periods_used: int
    
    @property
    def is_significant(self) -> bool:
        """Whether estimate is significant at 5% level."""
        return self.pvalue < 0.05
    
    def to_dict(self) -> dict:
        """
        Convert anticipation estimate to dictionary.

        Returns
        -------
        dict
            Dictionary containing all estimate attributes suitable for
            constructing a pandas DataFrame row.
        """
        return {
            'excluded_periods': self.excluded_periods,
            'att': self.att,
            'se': self.se,
            't_stat': self.t_stat,
            'pvalue': self.pvalue,
            'ci_lower': self.ci_lower,
            'ci_upper': self.ci_upper,
            'n_pre_periods_used': self.n_pre_periods_used,
            'significant': self.is_significant,
        }


@dataclass
class PrePeriodRobustnessResult:
    """
    Result of pre-treatment period robustness analysis.
    
    Assesses how ATT estimates vary when using different numbers of
    pre-treatment periods, helping identify whether findings are robust
    to this methodological choice.
    
    Attributes
    ----------
    specifications : list[SpecificationResult]
        ATT estimates for each pre-period configuration.
    baseline_spec : SpecificationResult
        Estimate using all available pre-treatment periods.
    att_range : tuple[float, float]
        (min ATT, max ATT) across all specifications.
    att_mean : float
        Mean ATT across specifications.
    att_std : float
        Standard deviation of ATT across specifications.
    sensitivity_ratio : float
        Ratio of range to baseline: (max - min) / abs(baseline).
    robustness_level : RobustnessLevel
        Categorical assessment of robustness.
    is_robust : bool
        Whether estimates are stable (ratio < threshold).
    robustness_threshold : float
        Threshold used for robustness determination.
    all_same_sign : bool
        Whether all estimates have the same sign.
    all_significant : bool
        Whether all estimates are significant at 5%.
    n_significant : int
        Number of significant specifications.
    n_sign_changes : int
        Number of specifications with sign different from baseline.
    rolling_method : str
        Transformation method used.
    estimator : str
        Estimation method used.
    n_specifications : int
        Total number of specifications tested.
    pre_period_range_tested : tuple[int, int]
        Range of pre-periods tested (min, max).
    recommendation : str
        Main recommendation based on analysis.
    detailed_recommendations : list[str]
        Detailed recommendations.
    result_warnings : list[str]
        Warning messages.
    figure : Any | None
        Matplotlib figure if plot was generated.
    """
    specifications: list[SpecificationResult]
    baseline_spec: SpecificationResult
    att_range: tuple[float, float]
    att_mean: float
    att_std: float
    sensitivity_ratio: float
    robustness_level: RobustnessLevel
    is_robust: bool
    robustness_threshold: float
    all_same_sign: bool
    all_significant: bool
    n_significant: int
    n_sign_changes: int
    rolling_method: str
    estimator: str
    n_specifications: int
    pre_period_range_tested: tuple[int, int]
    recommendation: str
    detailed_recommendations: list[str] = field(default_factory=list)
    result_warnings: list[str] = field(default_factory=list)
    figure: Any | None = None
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert all specification results to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with one row per specification containing ATT estimates,
            standard errors, p-values, and other diagnostic information.
        """
        return pd.DataFrame([s.to_dict() for s in self.specifications])
    
    def get_specification(self, n_pre: int) -> SpecificationResult | None:
        """
        Retrieve specification result for a specific pre-period count.

        Parameters
        ----------
        n_pre : int
            Number of pre-treatment periods to look up.

        Returns
        -------
        SpecificationResult or None
            The specification result if found, None otherwise.
        """
        for spec in self.specifications:
            if spec.n_pre_periods == n_pre:
                return spec
        return None
    
    def summary(self) -> str:
        """
        Generate a comprehensive human-readable summary report.

        Returns
        -------
        str
            Formatted text report containing configuration, baseline estimates,
            sensitivity metrics, robustness assessment, and recommendations.
        """
        lines = [
            "=" * 75,
            "PRE-TREATMENT PERIOD ROBUSTNESS ANALYSIS",
            "=" * 75,
            "",
            "CONFIGURATION:",
            f"  Transformation: {self.rolling_method}",
            f"  Estimator: {self.estimator}",
            f"  Pre-period range tested: {self.pre_period_range_tested[0]} - {self.pre_period_range_tested[1]}",
            f"  Number of specifications: {self.n_specifications}",
            "",
            "BASELINE ESTIMATE (all pre-periods):",
            f"  ATT = {self.baseline_spec.att:.4f} (SE = {self.baseline_spec.se:.4f})",
            f"  t-stat = {self.baseline_spec.t_stat:.3f}, p-value = {self.baseline_spec.pvalue:.4f}",
            f"  95% CI: [{self.baseline_spec.ci_lower:.4f}, {self.baseline_spec.ci_upper:.4f}]",
            "",
            "SENSITIVITY ANALYSIS:",
            f"  ATT Range: [{self.att_range[0]:.4f}, {self.att_range[1]:.4f}]",
            f"  ATT Mean: {self.att_mean:.4f}",
            f"  ATT Std Dev: {self.att_std:.4f}",
            f"  Sensitivity Ratio: {self.sensitivity_ratio:.1%}",
            "",
            "ROBUSTNESS ASSESSMENT:",
            f"  Level: {self.robustness_level.value.replace('_', ' ').title()}",
            f"  Is Robust (ratio < {self.robustness_threshold:.0%}): {'YES ✓' if self.is_robust else 'NO ⚠️'}",
            f"  All Same Sign: {'YES ✓' if self.all_same_sign else 'NO ⚠️'}",
            f"  All Significant: {'YES ✓' if self.all_significant else f'NO ({self.n_significant}/{self.n_specifications})'}",
            "",
        ]
        
        # Add specification table
        lines.extend([
            "SPECIFICATION DETAILS:",
            "-" * 70,
            f"{'N_Pre':>8} {'ATT':>12} {'SE':>10} {'P-value':>10} {'Sig':>6}",
            "-" * 70,
        ])
        
        for spec in sorted(self.specifications, key=lambda x: x.n_pre_periods):
            sig = "***" if spec.pvalue < 0.01 else ("**" if spec.pvalue < 0.05 else ("*" if spec.pvalue < 0.1 else ""))
            baseline_marker = " (baseline)" if spec.specification_id == self.baseline_spec.specification_id else ""
            lines.append(
                f"{spec.n_pre_periods:>8} {spec.att:>12.4f} {spec.se:>10.4f} "
                f"{spec.pvalue:>10.4f} {sig:>6}{baseline_marker}"
            )
        
        lines.extend([
            "",
            "─" * 75,
            "RECOMMENDATION:",
            f"  {self.recommendation}",
        ])
        
        if self.detailed_recommendations:
            lines.append("")
            lines.append("DETAILED RECOMMENDATIONS:")
            for i, rec in enumerate(self.detailed_recommendations, 1):
                lines.append(f"  {i}. {rec}")
        
        if self.result_warnings:
            lines.extend(["", "WARNINGS:"])
            for w in self.result_warnings:
                lines.append(f"  ⚠ {w}")
        
        lines.append("=" * 75)
        return "\n".join(lines)
    
    def plot(
        self,
        show_ci: bool = True,
        show_baseline: bool = True,
        figsize: tuple[float, float] = (10, 6),
        ax: Any = None,
    ) -> Any:
        """
        Generate sensitivity plot.
        
        Shows ATT estimates across different pre-period specifications
        with confidence intervals.
        
        Parameters
        ----------
        show_ci : bool, default True
            Whether to show confidence intervals.
        show_baseline : bool, default True
            Whether to show baseline reference line.
        figsize : tuple, default (10, 6)
            Figure size in inches.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
            
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        
        df = self.to_dataframe().sort_values('n_pre_periods')
        
        # Plot estimates with CI
        if show_ci:
            ax.errorbar(
                df['n_pre_periods'],
                df['att'],
                yerr=1.96 * df['se'],
                fmt='o-',
                capsize=4,
                label='ATT Estimate',
                color='steelblue',
                markersize=8,
            )
        else:
            ax.plot(
                df['n_pre_periods'],
                df['att'],
                'o-',
                label='ATT Estimate',
                color='steelblue',
                markersize=8,
            )
        
        # Highlight baseline
        if show_baseline:
            ax.axhline(
                self.baseline_spec.att,
                color='red',
                linestyle='--',
                alpha=0.7,
                label=f'Baseline ATT = {self.baseline_spec.att:.3f}',
            )
            
            # Robustness band (±25% of baseline)
            band_width = 0.25 * abs(self.baseline_spec.att)
            if band_width > 0:
                ax.axhspan(
                    self.baseline_spec.att - band_width,
                    self.baseline_spec.att + band_width,
                    alpha=0.1,
                    color='green',
                    label='±25% Robustness Band',
                )
        
        # Add zero line
        ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
        
        # Labels and title
        ax.set_xlabel('Number of Pre-treatment Periods', fontsize=12)
        ax.set_ylabel('ATT Estimate', fontsize=12)
        ax.set_title(
            f'Pre-treatment Period Robustness Analysis\n'
            f'Sensitivity Ratio: {self.sensitivity_ratio:.1%} '
            f'({self.robustness_level.value.replace("_", " ").title()})',
            fontsize=14,
        )
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.figure = fig
        return fig


@dataclass
class NoAnticipationSensitivityResult:
    """
    Result of no-anticipation sensitivity analysis.
    
    Tests robustness of ATT estimates to potential anticipation effects
    by excluding periods immediately before treatment.
    
    Attributes
    ----------
    estimates : list[AnticipationEstimate]
        ATT estimates for each exclusion configuration.
    baseline_estimate : AnticipationEstimate
        Estimate with no exclusion (excluded_periods=0).
    anticipation_detected : bool
        Whether anticipation effects are detected.
    recommended_exclusion : int
        Recommended number of periods to exclude.
    detection_method : AnticipationDetectionMethod
        Method used to detect anticipation.
    recommendation : str
        Interpretation and recommendations.
    result_warnings : list[str]
        Warning messages.
    figure : Any | None
        Matplotlib figure if plot was generated.
    """
    estimates: list[AnticipationEstimate]
    baseline_estimate: AnticipationEstimate
    anticipation_detected: bool
    recommended_exclusion: int
    detection_method: AnticipationDetectionMethod
    recommendation: str
    result_warnings: list[str] = field(default_factory=list)
    figure: Any | None = None
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert all anticipation estimates to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with one row per exclusion level containing ATT estimates,
            standard errors, p-values, and significance indicators.
        """
        return pd.DataFrame([e.to_dict() for e in self.estimates])
    
    def summary(self) -> str:
        """
        Generate a human-readable summary of the anticipation analysis.

        Returns
        -------
        str
            Formatted text report containing estimates by exclusion level,
            detection results, and recommendations.
        """
        lines = [
            "=" * 70,
            "NO-ANTICIPATION SENSITIVITY ANALYSIS",
            "=" * 70,
            "",
            f"Exclusion range tested: 0 - {max(e.excluded_periods for e in self.estimates)}",
            "",
            "Estimates by Exclusion:",
            "-" * 60,
            f"{'Excluded':>10} {'ATT':>12} {'SE':>10} {'P-value':>10} {'Sig':>6}",
            "-" * 60,
        ]
        
        for e in self.estimates:
            sig = "***" if e.pvalue < 0.01 else ("**" if e.pvalue < 0.05 else ("*" if e.pvalue < 0.1 else ""))
            lines.append(
                f"{e.excluded_periods:>10} {e.att:>12.4f} {e.se:>10.4f} "
                f"{e.pvalue:>10.4f} {sig:>6}"
            )
        
        lines.extend([
            "",
            f"Anticipation Detected: {'YES ⚠️' if self.anticipation_detected else 'NO ✓'}",
            f"Detection Method: {self.detection_method.value}",
        ])
        
        if self.anticipation_detected:
            lines.append(f"Recommended Exclusion: {self.recommended_exclusion} period(s)")
        
        lines.extend([
            "",
            "─" * 70,
            f"RECOMMENDATION: {self.recommendation}",
            "─" * 70,
        ])
        
        if self.result_warnings:
            lines.extend(["", "WARNINGS:"])
            for w in self.result_warnings:
                lines.append(f"  ⚠ {w}")
        
        lines.append("=" * 70)
        return "\n".join(lines)
    
    def plot(
        self,
        show_ci: bool = True,
        figsize: tuple[float, float] = (10, 6),
        ax: Any = None,
    ) -> Any:
        """
        Generate anticipation sensitivity plot.
        
        Parameters
        ----------
        show_ci : bool, default True
            Whether to show confidence intervals.
        figsize : tuple, default (10, 6)
            Figure size in inches.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.
            
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        
        df = self.to_dataframe().sort_values('excluded_periods')
        
        # Plot estimates
        if show_ci:
            ax.errorbar(
                df['excluded_periods'],
                df['att'],
                yerr=1.96 * df['se'],
                fmt='o-',
                capsize=4,
                color='steelblue',
                markersize=8,
                label='ATT Estimate',
            )
        else:
            ax.plot(
                df['excluded_periods'],
                df['att'],
                'o-',
                color='steelblue',
                markersize=8,
                label='ATT Estimate',
            )
        
        # Highlight recommended exclusion
        if self.anticipation_detected and self.recommended_exclusion > 0:
            rec_est = next(
                (e for e in self.estimates if e.excluded_periods == self.recommended_exclusion),
                None
            )
            if rec_est:
                ax.scatter(
                    [self.recommended_exclusion],
                    [rec_est.att],
                    s=200,
                    facecolors='none',
                    edgecolors='red',
                    linewidths=2,
                    label=f'Recommended (k={self.recommended_exclusion})',
                    zorder=5,
                )
        
        # Zero line
        ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
        
        # Labels
        ax.set_xlabel('Number of Excluded Periods Before Treatment', fontsize=12)
        ax.set_ylabel('ATT Estimate', fontsize=12)
        ax.set_title(
            f'No-Anticipation Sensitivity Analysis\n'
            f'Anticipation Detected: {"Yes" if self.anticipation_detected else "No"}',
            fontsize=14,
        )
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(df['excluded_periods'].values)
        
        plt.tight_layout()
        self.figure = fig
        return fig


@dataclass
class ComprehensiveSensitivityResult:
    """
    Combined results from comprehensive sensitivity analysis.
    
    Attributes
    ----------
    pre_period_result : PrePeriodRobustnessResult | None
        Results from pre-period robustness analysis.
    anticipation_result : NoAnticipationSensitivityResult | None
        Results from no-anticipation sensitivity analysis.
    transformation_comparison : dict | None
        Comparison of demean vs detrend results.
    estimator_comparison : dict | None
        Comparison across different estimators.
    overall_assessment : str
        Overall robustness assessment.
    recommendations : list[str]
        List of recommendations.
    """
    pre_period_result: PrePeriodRobustnessResult | None = None
    anticipation_result: NoAnticipationSensitivityResult | None = None
    transformation_comparison: dict | None = None
    estimator_comparison: dict | None = None
    overall_assessment: str = ""
    recommendations: list[str] = field(default_factory=list)
    
    def summary(self) -> str:
        """
        Generate a comprehensive summary of all sensitivity analyses.

        Returns
        -------
        str
            Formatted text report containing results from pre-period robustness,
            anticipation sensitivity, transformation comparison, estimator
            comparison, overall assessment, and recommendations.
        """
        lines = [
            "=" * 70,
            "COMPREHENSIVE SENSITIVITY ANALYSIS",
            "=" * 70,
            "",
        ]
        
        if self.pre_period_result:
            lines.extend([
                "1. Pre-treatment Period Robustness:",
                f"   Robust: {'YES' if self.pre_period_result.is_robust else 'NO'}",
                f"   Sensitivity Ratio: {self.pre_period_result.sensitivity_ratio:.2%}",
                "",
            ])
        
        if self.anticipation_result:
            lines.extend([
                "2. No-Anticipation Sensitivity:",
                f"   Anticipation Detected: {'YES' if self.anticipation_result.anticipation_detected else 'NO'}",
                "",
            ])
        
        if self.transformation_comparison:
            lines.extend([
                "3. Transformation Comparison (demean vs detrend):",
                f"   Demean ATT: {self.transformation_comparison.get('demean_att', 'N/A'):.4f}",
                f"   Detrend ATT: {self.transformation_comparison.get('detrend_att', 'N/A'):.4f}",
                f"   Difference: {self.transformation_comparison.get('difference', 'N/A'):.4f}",
                "",
            ])
        
        if self.estimator_comparison:
            lines.extend([
                "4. Estimator Comparison:",
            ])
            for est, att in self.estimator_comparison.items():
                if est != 'range':
                    lines.append(f"   {est.upper()}: {att:.4f}")
            lines.append("")
        
        lines.extend([
            "─" * 70,
            f"OVERALL ASSESSMENT: {self.overall_assessment}",
            "",
            "RECOMMENDATIONS:",
        ])
        
        for i, rec in enumerate(self.recommendations, 1):
            lines.append(f"  {i}. {rec}")
        
        lines.append("=" * 70)
        return "\n".join(lines)
    
    def plot_all(self, figsize: tuple[float, float] = (14, 10)) -> Any:
        """
        Generate combined visualization of all sensitivity analyses.

        Parameters
        ----------
        figsize : tuple of float, default (14, 10)
            Figure size in inches (width, height).

        Returns
        -------
        matplotlib.figure.Figure or None
            Combined figure with subplots for each available analysis,
            or None if no results are available to plot.
        """
        import matplotlib.pyplot as plt
        
        n_plots = sum([
            self.pre_period_result is not None,
            self.anticipation_result is not None,
        ])
        
        if n_plots == 0:
            warnings.warn("No results to plot")
            return None
        
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)
        if n_plots == 1:
            axes = [axes]
        
        idx = 0
        if self.pre_period_result:
            self.pre_period_result.plot(ax=axes[idx])
            idx += 1
        
        if self.anticipation_result:
            self.anticipation_result.plot(ax=axes[idx])
            idx += 1
        
        plt.tight_layout()
        return fig


# =============================================================================
# Helper Functions
# =============================================================================

def _validate_robustness_inputs(
    data: pd.DataFrame,
    y: str,
    ivar: str,
    tvar: str,
    gvar: str | None,
    d: str | None,
    post: str | None,
    rolling: str,
) -> None:
    """
    Validate inputs for robustness analysis.

    Performs three validation checks: presence of required columns in the
    DataFrame, validity of the transformation method specification, and
    consistency of the design mode parameters.

    Parameters
    ----------
    data : pd.DataFrame
        Input panel data.
    y : str
        Outcome variable column name.
    ivar : str
        Unit identifier column name.
    tvar : str
        Time variable column name.
    gvar : str or None
        Cohort variable for staggered designs.
    d : str or None
        Treatment indicator for common timing.
    post : str or None
        Post-treatment indicator for common timing.
    rolling : str
        Transformation method.

    Raises
    ------
    ValueError
        If required columns are missing, rolling method is invalid, or
        design mode parameters are inconsistent (must specify either gvar
        for staggered designs or both d and post for common timing).
    """
    # Check required columns
    required = [y, ivar, tvar]
    if gvar is not None:
        required.append(gvar)
    if d is not None:
        required.append(d)
    if post is not None:
        required.append(post)
    
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Check rolling method
    valid_rolling = {'demean', 'detrend', 'demeanq', 'detrendq'}
    if rolling.lower() not in valid_rolling:
        raise ValueError(f"rolling must be one of {valid_rolling}, got '{rolling}'")
    
    # Check mode consistency
    is_staggered = gvar is not None
    is_common = d is not None and post is not None
    
    if not is_staggered and not is_common:
        raise ValueError(
            "Must specify either gvar (staggered) or both d and post (common timing)"
        )


def _auto_detect_pre_period_range(
    data: pd.DataFrame,
    ivar: str,
    tvar: str,
    gvar: str | None,
    d: str | None,
    post: str | None,
    rolling: str,
) -> tuple[int, int]:
    """
    Automatically detect valid pre-treatment period range.
    
    Returns (min_pre, max_pre) where:
    - min_pre: Minimum required for the transformation method
    - max_pre: Maximum available in the data
    
    Parameters
    ----------
    data : pd.DataFrame
        Input panel data.
    ivar : str
        Unit identifier column name.
    tvar : str
        Time variable column name.
    gvar : str or None
        Cohort variable for staggered designs.
    d : str or None
        Treatment indicator for common timing.
    post : str or None
        Post-treatment indicator for common timing.
    rolling : str
        Transformation method.
        
    Returns
    -------
    tuple[int, int]
        (min_pre_periods, max_pre_periods)
    """
    # Minimum requirements by method
    min_required = {
        'demean': 1,
        'detrend': 2,
        'demeanq': 1,
        'detrendq': 2,
    }
    min_pre = min_required.get(rolling.lower(), 1)
    
    # Detect maximum available
    if gvar is not None:
        # Staggered: find minimum pre-periods across cohorts
        cohorts = data[gvar].dropna().unique()
        cohorts = [c for c in cohorts if c > 0 and np.isfinite(c)]
        
        if not cohorts:
            return (min_pre, min_pre)
        
        max_pre_by_cohort = []
        min_time = data[tvar].min()
        for cohort in cohorts:
            max_pre_by_cohort.append(int(cohort - min_time))
        
        max_pre = min(max_pre_by_cohort) if max_pre_by_cohort else min_pre
    else:
        # Common timing: count pre-treatment periods
        pre_data = data[data[post] == 0]
        max_pre = pre_data[tvar].nunique()
    
    # Ensure valid range
    max_pre = max(max_pre, min_pre)
    
    return (min_pre, max_pre)


def _get_max_pre_periods(
    data: pd.DataFrame,
    ivar: str,
    tvar: str,
    gvar: str | None,
    post: str | None,
) -> int:
    """
    Determine the maximum number of pre-treatment periods available.

    For staggered designs, returns the minimum across all cohorts to ensure
    all cohorts have sufficient pre-treatment data. For common timing,
    counts the number of unique pre-treatment time periods.

    Parameters
    ----------
    data : pd.DataFrame
        Input panel data.
    ivar : str
        Unit identifier column name.
    tvar : str
        Time variable column name.
    gvar : str or None
        Cohort variable for staggered designs.
    post : str or None
        Post-treatment indicator for common timing.

    Returns
    -------
    int
        Maximum number of pre-treatment periods available.
    """
    if gvar is not None:
        cohorts = data[gvar].dropna().unique()
        cohorts = [c for c in cohorts if c > 0 and np.isfinite(c)]
        
        if not cohorts:
            return 0
        
        min_time = data[tvar].min()
        max_pre_by_cohort = [int(c - min_time) for c in cohorts]
        return min(max_pre_by_cohort) if max_pre_by_cohort else 0
    else:
        pre_data = data[data[post] == 0]
        return pre_data[tvar].nunique()


def _filter_to_n_pre_periods(
    data: pd.DataFrame,
    ivar: str,
    tvar: str,
    gvar: str | None,
    d: str | None,
    post: str | None,
    n_pre_periods: int,
    exclude_periods: int,
) -> pd.DataFrame:
    """
    Filter data to use only specified number of pre-treatment periods.
    
    For staggered designs, this is done cohort-by-cohort.
    For common timing, this filters the entire dataset.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input panel data.
    ivar : str
        Unit identifier column name.
    tvar : str
        Time variable column name.
    gvar : str or None
        Cohort variable for staggered designs.
    d : str or None
        Treatment indicator for common timing.
    post : str or None
        Post-treatment indicator for common timing.
    n_pre_periods : int
        Number of pre-treatment periods to keep.
    exclude_periods : int
        Number of periods to exclude before treatment.
        
    Returns
    -------
    pd.DataFrame
        Filtered data with specified pre-treatment periods.
    """
    data = data.copy()
    
    if gvar is not None:
        # Staggered: filter by cohort
        filtered_dfs = []
        
        cohorts = data[gvar].dropna().unique()
        cohorts = [c for c in cohorts if c > 0 and np.isfinite(c)]
        
        for cohort in cohorts:
            cohort_mask = data[gvar] == cohort
            cohort_data = data[cohort_mask].copy()
            
            # Determine pre-treatment period range for this cohort
            treatment_period = cohort
            pre_end = treatment_period - 1 - exclude_periods
            pre_start = pre_end - n_pre_periods + 1
            
            # Keep only specified pre-periods and all post-periods
            time_mask = (
                (cohort_data[tvar] >= pre_start) & 
                (cohort_data[tvar] <= pre_end)
            ) | (cohort_data[tvar] >= treatment_period)
            
            filtered_dfs.append(cohort_data[time_mask])
        
        # Also include never-treated units (all their periods)
        never_treated_mask = (
            data[gvar].isna() | 
            (data[gvar] == 0) | 
            (data[gvar] == np.inf)
        )
        if never_treated_mask.any():
            # For never-treated, keep periods that align with treated cohorts
            never_data = data[never_treated_mask].copy()
            if cohorts:
                min_cohort = min(cohorts)
                pre_end = min_cohort - 1 - exclude_periods
                pre_start = pre_end - n_pre_periods + 1
                time_mask = (never_data[tvar] >= pre_start)
                filtered_dfs.append(never_data[time_mask])
            else:
                filtered_dfs.append(never_data)
        
        if filtered_dfs:
            return pd.concat(filtered_dfs, ignore_index=True)
        # Return empty DataFrame preserving schema for downstream compatibility
        return data.iloc[0:0]
    
    else:
        # Common timing: simpler filtering
        pre_data = data[data[post] == 0]
        post_data = data[data[post] != 0]
        
        # Get all pre-treatment times and select the last n_pre_periods
        pre_times = sorted(pre_data[tvar].unique())
        
        if exclude_periods > 0:
            pre_times = pre_times[:-exclude_periods]
        
        if len(pre_times) < n_pre_periods:
            # Not enough periods, use all available
            selected_pre_times = pre_times
        else:
            selected_pre_times = pre_times[-n_pre_periods:]
        
        # Filter pre-treatment data
        filtered_pre = pre_data[pre_data[tvar].isin(selected_pre_times)]
        
        return pd.concat([filtered_pre, post_data], ignore_index=True)


def _filter_excluding_periods(
    data: pd.DataFrame,
    ivar: str,
    tvar: str,
    gvar: str | None,
    post: str | None,
    exclude_periods: int,
) -> pd.DataFrame:
    """
    Filter data excluding specified periods before treatment.

    Removes the specified number of periods immediately preceding treatment
    from the pre-treatment baseline. For common timing designs, re-encodes
    the time variable to maintain continuity after filtering.

    Parameters
    ----------
    data : pd.DataFrame
        Input panel data.
    ivar : str
        Unit identifier column name.
    tvar : str
        Time variable column name.
    gvar : str or None
        Cohort variable for staggered designs.
    post : str or None
        Post-treatment indicator for common timing.
    exclude_periods : int
        Number of periods to exclude before treatment.

    Returns
    -------
    pd.DataFrame
        Filtered data with excluded periods removed and time re-encoded.

    Notes
    -----
    Time re-encoding for common timing designs prevents discontinuity errors
    that would otherwise occur when the excluded periods create gaps in the
    time sequence. For staggered designs, re-encoding is not performed due
    to the complexity of cohort-specific time structures.
    """
    if exclude_periods == 0:
        return data.copy()
    
    data = data.copy()
    
    if gvar is not None:
        # Staggered: exclude by cohort
        filtered_dfs = []
        
        cohorts = data[gvar].dropna().unique()
        cohorts = [c for c in cohorts if c > 0 and np.isfinite(c)]
        
        for cohort in cohorts:
            cohort_mask = data[gvar] == cohort
            cohort_data = data[cohort_mask].copy()
            
            # Exclude periods [g - exclude_periods, g - 1]
            excluded_times = list(range(int(cohort - exclude_periods), int(cohort)))
            time_mask = ~cohort_data[tvar].isin(excluded_times)
            filtered_dfs.append(cohort_data[time_mask])
        
        # Include never-treated units (no exclusion needed)
        never_treated_mask = (
            data[gvar].isna() | 
            (data[gvar] == 0) | 
            (data[gvar] == np.inf)
        )
        if never_treated_mask.any():
            filtered_dfs.append(data[never_treated_mask])
        
        if filtered_dfs:
            result = pd.concat(filtered_dfs, ignore_index=True)
            # Time re-encoding is deferred to the caller for staggered designs,
            # as cohort-specific period structures may introduce discontinuities.
            return result
        return data.iloc[0:0]
    
    else:
        # Common timing
        pre_data = data[data[post] == 0]
        post_data = data[data[post] != 0]
        
        # Get pre-treatment times and exclude the last `exclude_periods`
        pre_times = sorted(pre_data[tvar].unique())
        
        if exclude_periods >= len(pre_times):
            warnings.warn(
                f"Cannot exclude {exclude_periods} periods: only {len(pre_times)} "
                f"pre-treatment periods available."
            )
            return data.copy()
        
        excluded_times = pre_times[-exclude_periods:]
        filtered_pre = pre_data[~pre_data[tvar].isin(excluded_times)]
        
        # Combine filtered pre and post data
        result = pd.concat([filtered_pre, post_data], ignore_index=True)
        
        # Re-encode time variable to be continuous
        # This avoids TimeDiscontinuityError in lwdid()
        remaining_times = sorted(result[tvar].unique())
        time_mapping = {old_t: new_t for new_t, old_t in enumerate(remaining_times, start=1)}
        result[tvar] = result[tvar].map(time_mapping)
        
        return result


def _determine_robustness_level(sensitivity_ratio: float) -> RobustnessLevel:
    """
    Determine robustness level based on sensitivity ratio.
    
    Parameters
    ----------
    sensitivity_ratio : float
        Ratio of ATT range to baseline ATT.
        
    Returns
    -------
    RobustnessLevel
        Categorical robustness assessment.
    """
    if sensitivity_ratio < 0.10:
        return RobustnessLevel.HIGHLY_ROBUST
    elif sensitivity_ratio < 0.25:
        return RobustnessLevel.MODERATELY_ROBUST
    elif sensitivity_ratio < 0.50:
        return RobustnessLevel.SENSITIVE
    else:
        return RobustnessLevel.HIGHLY_SENSITIVE


def _compute_sensitivity_ratio(
    atts: list[float],
    baseline_att: float,
) -> float:
    """
    Compute sensitivity ratio measuring estimate variability.

    Parameters
    ----------
    atts : list[float]
        List of ATT estimates across specifications.
    baseline_att : float
        Baseline ATT estimate used for normalization.

    Returns
    -------
    float
        Sensitivity ratio, or infinity if baseline is near zero but range
        is positive, or zero if both baseline and range are near zero.

    Notes
    -----
    The sensitivity ratio is defined as the range of ATT estimates divided
    by the absolute value of the baseline estimate:

    .. math::

        \\text{ratio} = \\frac{\\max(ATT) - \\min(ATT)}{|ATT_{baseline}|}

    A ratio of 0.25 indicates the estimate range spans 25% of the baseline
    magnitude. Lower ratios indicate greater stability across specifications.
    """
    if not atts:
        return 0.0
    
    att_range = max(atts) - min(atts)
    
    if abs(baseline_att) > 1e-10:
        return att_range / abs(baseline_att)
    else:
        return float('inf') if att_range > 1e-10 else 0.0


def _generate_robustness_recommendations(
    specifications: list[SpecificationResult],
    baseline_spec: SpecificationResult,
    sensitivity_ratio: float,
    is_robust: bool,
    all_same_sign: bool,
    all_significant: bool,
    rolling: str,
) -> tuple[str, list[str], list[str]]:
    """
    Generate recommendations based on robustness analysis.
    
    Parameters
    ----------
    specifications : list[SpecificationResult]
        All specification results.
    baseline_spec : SpecificationResult
        Baseline specification result.
    sensitivity_ratio : float
        Computed sensitivity ratio.
    is_robust : bool
        Whether results are robust.
    all_same_sign : bool
        Whether all estimates have same sign.
    all_significant : bool
        Whether all estimates are significant.
    rolling : str
        Transformation method used.
        
    Returns
    -------
    tuple[str, list[str], list[str]]
        (main_recommendation, detailed_recommendations, warnings)
    """
    recommendations = []
    result_warnings = []
    
    # Main recommendation
    if is_robust and all_same_sign and all_significant:
        main_rec = (
            "Results are robust to pre-treatment period selection. "
            "The ATT estimate is stable across specifications."
        )
    elif is_robust and all_same_sign:
        main_rec = (
            "Results are moderately robust. Sign is consistent but "
            "significance varies across specifications."
        )
        recommendations.append(
            "Consider reporting the range of estimates for transparency."
        )
    elif not all_same_sign:
        main_rec = (
            "CAUTION: Results are sensitive to pre-treatment period selection. "
            "Sign changes detected across specifications."
        )
        result_warnings.append("Sign change detected - interpret results with caution.")
        recommendations.append(
            "Investigate why estimates change sign with different pre-periods."
        )
        recommendations.append(
            "Consider using detrend method if trends may be heterogeneous."
        )
    else:
        main_rec = (
            f"Results show moderate sensitivity (ratio = {sensitivity_ratio:.1%}). "
            "Consider additional robustness checks."
        )
    
    # Method-specific recommendations
    if rolling.lower() == 'demean' and sensitivity_ratio > 0.25:
        recommendations.append(
            "High sensitivity with demean suggests potential heterogeneous trends. "
            "Consider using rolling='detrend' instead."
        )
    
    if rolling.lower() == 'detrend' and sensitivity_ratio > 0.50:
        recommendations.append(
            "High sensitivity even with detrend suggests potential model "
            "misspecification or data quality issues."
        )
    
    # Monotonic pattern may indicate time-varying confounding
    converged_specs = [s for s in specifications if s.converged]
    atts = [s.att for s in sorted(converged_specs, key=lambda x: x.n_pre_periods)]
    if len(atts) >= 3:
        # Consistent increase or decrease across specifications is suspicious
        diffs = np.diff(atts)
        if len(diffs) > 0 and (all(d > 0 for d in diffs) or all(d < 0 for d in diffs)):
            result_warnings.append(
                "ATT estimates show monotonic trend with pre-period count. "
                "This may indicate time-varying confounding."
            )
    
    return main_rec, recommendations, result_warnings


def _run_single_specification(
    data: pd.DataFrame,
    y: str,
    ivar: str,
    tvar: str,
    gvar: str | None,
    d: str | None,
    post: str | None,
    rolling: str,
    estimator: str,
    controls: list[str] | None,
    vce: str | None,
    cluster_var: str | None,
    n_pre_periods: int,
    exclude_periods: int,
    alpha: float,
    spec_id: int,
) -> SpecificationResult:
    """
    Run estimation for a single specification.
    
    Filters data to use only the specified number of pre-treatment periods
    and runs lwdid estimation.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input panel data.
    y : str
        Outcome variable column name.
    ivar : str
        Unit identifier column name.
    tvar : str
        Time variable column name.
    gvar : str or None
        Cohort variable for staggered designs.
    d : str or None
        Treatment indicator for common timing.
    post : str or None
        Post-treatment indicator for common timing.
    rolling : str
        Transformation method.
    estimator : str
        Estimation method.
    controls : list[str] or None
        Control variables.
    vce : str or None
        Variance estimator type.
    cluster_var : str or None
        Cluster variable.
    n_pre_periods : int
        Number of pre-treatment periods to use.
    exclude_periods : int
        Number of periods to exclude before treatment.
    alpha : float
        Significance level.
    spec_id : int
        Specification identifier.
        
    Returns
    -------
    SpecificationResult
        Result for this specification.
    """
    from .core import lwdid
    
    try:
        # Filter data to specified pre-period range
        filtered_data = _filter_to_n_pre_periods(
            data=data,
            ivar=ivar,
            tvar=tvar,
            gvar=gvar,
            d=d,
            post=post,
            n_pre_periods=n_pre_periods,
            exclude_periods=exclude_periods,
        )
        
        if len(filtered_data) == 0:
            raise ValueError("No data remaining after filtering")
        
        # Determine start and end periods
        if gvar is not None:
            # Staggered: use minimum cohort
            cohorts = filtered_data[gvar].dropna().unique()
            cohorts = [c for c in cohorts if c > 0 and np.isfinite(c)]
            if cohorts:
                min_cohort = min(cohorts)
                end_period = int(min_cohort - 1 - exclude_periods)
                start_period = end_period - n_pre_periods + 1
            else:
                start_period = end_period = 0
        else:
            pre_mask = filtered_data[post] == 0
            pre_times = filtered_data.loc[pre_mask, tvar].unique()
            if len(pre_times) > 0:
                start_period = int(min(pre_times))
                end_period = int(max(pre_times))
            else:
                start_period = end_period = 0
        
        # Run estimation
        result = lwdid(
            data=filtered_data,
            y=y,
            d=d,
            ivar=ivar,
            tvar=tvar,
            post=post,
            gvar=gvar,
            rolling=rolling,
            estimator=estimator,
            controls=controls,
            vce=vce,
            cluster_var=cluster_var,
            alpha=alpha,
        )
        
        return SpecificationResult(
            specification_id=spec_id,
            n_pre_periods=n_pre_periods,
            start_period=start_period,
            end_period=end_period,
            excluded_periods=exclude_periods,
            att=result.att,
            se=result.se_att,
            t_stat=result.t_stat,
            pvalue=result.pvalue,
            ci_lower=result.ci_lower,
            ci_upper=result.ci_upper,
            n_treated=result.n_treated,
            n_control=result.n_control,
            df=result.df_inference,
            converged=True,
            spec_warnings=[],
        )
        
    except Exception as e:
        warnings.warn(f"Specification {spec_id} (n_pre={n_pre_periods}) failed: {e}")
        return SpecificationResult(
            specification_id=spec_id,
            n_pre_periods=n_pre_periods,
            start_period=0,
            end_period=0,
            excluded_periods=exclude_periods,
            att=np.nan,
            se=np.nan,
            t_stat=np.nan,
            pvalue=np.nan,
            ci_lower=np.nan,
            ci_upper=np.nan,
            n_treated=0,
            n_control=0,
            df=0,
            converged=False,
            spec_warnings=[str(e)],
        )


def _detect_anticipation_effects(
    estimates: list[AnticipationEstimate],
    baseline: AnticipationEstimate,
    threshold: float,
) -> tuple[bool, int, AnticipationDetectionMethod]:
    """
    Detect anticipation effects from sensitivity analysis results.

    Applies two detection methods sequentially to identify potential
    violations of the no-anticipation assumption.

    Parameters
    ----------
    estimates : list[AnticipationEstimate]
        Estimates for each exclusion level, ordered by exclusion count.
    baseline : AnticipationEstimate
        Baseline estimate with no period exclusion.
    threshold : float
        Detection threshold for relative change in ATT.

    Returns
    -------
    detected : bool
        Whether anticipation effects were detected.
    recommended_exclusion : int
        Recommended number of periods to exclude if detected.
    method : AnticipationDetectionMethod
        Detection method that identified the effect.

    Notes
    -----
    Two detection methods are applied:

    1. Coefficient change method: Flags anticipation if ATT increases in
       magnitude by more than the threshold when excluding periods. This
       pattern suggests pre-treatment periods were biasing estimates toward
       zero.

    2. Trend break method: Flags anticipation if ATT magnitude increases
       monotonically with exclusion count, then stabilizes. The recommended
       exclusion is set where the rate of increase drops by at least 50%.
    """
    valid_estimates = [e for e in estimates if not np.isnan(e.att)]
    
    if len(valid_estimates) < 2:
        return False, 0, AnticipationDetectionMethod.INSUFFICIENT_DATA
    
    # Method 1: Check for significant coefficient change
    baseline_att = baseline.att
    for est in valid_estimates[1:]:  # Skip baseline
        if abs(baseline_att) > 1e-10:
            relative_change = abs(est.att - baseline_att) / abs(baseline_att)
            if relative_change > threshold:
                # Check if change is in expected direction
                # (anticipation typically biases toward zero)
                if abs(est.att) > abs(baseline_att):
                    return True, est.excluded_periods, AnticipationDetectionMethod.COEFFICIENT_CHANGE
    
    # Method 2: Check for monotonic pattern suggesting anticipation
    atts = [e.att for e in valid_estimates]
    if len(atts) >= 3:
        # If ATT magnitude increases monotonically with exclusion,
        # suggests anticipation was biasing estimates toward zero
        abs_atts = [abs(a) for a in atts]
        if all(abs_atts[i] <= abs_atts[i+1] for i in range(len(abs_atts)-1)):
            # Find where the increase stabilizes
            for i in range(1, len(abs_atts)):
                if i < len(abs_atts) - 1:
                    current_increase = abs_atts[i] - abs_atts[i-1]
                    next_increase = abs_atts[i+1] - abs_atts[i]
                    if current_increase > 0 and next_increase < current_increase * 0.5:
                        return True, i, AnticipationDetectionMethod.TREND_BREAK
    
    return False, 0, AnticipationDetectionMethod.NONE_DETECTED


# =============================================================================
# Main Public Functions
# =============================================================================

def robustness_pre_periods(
    data: pd.DataFrame,
    y: str,
    ivar: str,
    tvar: str,
    gvar: str | None = None,
    d: str | None = None,
    post: str | None = None,
    rolling: str = 'demean',
    estimator: str = 'ra',
    controls: list[str] | None = None,
    vce: str | None = None,
    cluster_var: str | None = None,
    pre_period_range: tuple[int, int] | None = None,
    step: int = 1,
    exclude_periods_before_treatment: int = 0,
    robustness_threshold: float = 0.25,
    alpha: float = 0.05,
    verbose: bool = True,
) -> PrePeriodRobustnessResult:
    """
    Assess robustness of ATT estimates to pre-treatment period selection.
    
    Tests how ATT estimates vary when using different numbers of pre-treatment
    periods, allowing researchers to assess whether findings are robust to
    this methodological choice.
    
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
    d : str, optional
        Treatment indicator for common timing.
    post : str, optional
        Post-treatment indicator for common timing.
    rolling : {'demean', 'detrend'}, default 'demean'
        Transformation method.
    estimator : {'ra', 'ipw', 'ipwra', 'psm'}, default 'ra'
        Estimation method.
    controls : list of str, optional
        Control variable column names.
    vce : str, optional
        Variance estimator type.
    cluster_var : str, optional
        Cluster variable for clustered SE.
    pre_period_range : tuple of (int, int), optional
        Range of pre-treatment periods to test (min_periods, max_periods).
        If None, automatically determined from data.
    step : int, default 1
        Step size for varying pre-treatment periods.
    exclude_periods_before_treatment : int, default 0
        Number of periods to exclude immediately before treatment.
        Useful for testing robustness to no-anticipation violations.
    robustness_threshold : float, default 0.25
        Threshold for robustness determination. Results are considered
        robust if sensitivity_ratio < robustness_threshold.
    alpha : float, default 0.05
        Significance level for confidence intervals.
    verbose : bool, default True
        Whether to print progress and summary.
    
    Returns
    -------
    PrePeriodRobustnessResult
        Results containing:
        - specifications: ATT estimates for each pre-period count
        - sensitivity_ratio: Range of ATT estimates relative to baseline
        - is_robust: Whether estimates are stable across specifications
        - recommendation: Interpretation and recommendations
        - figure: Sensitivity plot (if plot() called)
    
    Notes
    -----
    The function varies the starting point of pre-treatment data and re-estimates
    ATT for each specification, allowing researchers to assess how sensitive
    their findings are to this methodological choice.
    
    In many applications, the policy intervention may be based on past outcomes.
    This analysis helps determine whether sufficient pre-treatment periods are
    being used to adequately control for selection into treatment.
    
    Robustness levels based on sensitivity ratio:
    - < 10%: Highly robust
    - 10-25%: Moderately robust
    - 25-50%: Sensitive
    - >= 50%: Highly sensitive
    
    See Also
    --------
    lwdid : Main estimation function.
    sensitivity_no_anticipation : Test robustness to anticipation effects.
    sensitivity_analysis : Comprehensive sensitivity analysis.
    """
    # 1. Validate inputs
    _validate_robustness_inputs(data, y, ivar, tvar, gvar, d, post, rolling)
    
    # 2. Determine pre-period range
    if pre_period_range is None:
        pre_period_range = _auto_detect_pre_period_range(
            data, ivar, tvar, gvar, d, post, rolling
        )
    
    min_pre, max_pre = pre_period_range
    
    if verbose:
        print(f"Pre-treatment period robustness analysis")
        print(f"Testing pre-period range: {min_pre} to {max_pre}")
        print("-" * 50)
    
    # 3. Generate specification list
    n_pre_values = list(range(min_pre, max_pre + 1, step))
    
    if len(n_pre_values) < 2:
        warnings.warn(
            f"Only {len(n_pre_values)} specification(s) possible. "
            "Consider expanding pre_period_range or reducing step."
        )
    
    # 4. Run estimations for each specification
    specifications = []
    for i, n_pre in enumerate(n_pre_values):
        if verbose:
            print(f"Running specification {i+1}/{len(n_pre_values)}: n_pre={n_pre}")
        
        spec_result = _run_single_specification(
            data=data,
            y=y, ivar=ivar, tvar=tvar,
            gvar=gvar, d=d, post=post,
            rolling=rolling, estimator=estimator,
            controls=controls, vce=vce, cluster_var=cluster_var,
            n_pre_periods=n_pre,
            exclude_periods=exclude_periods_before_treatment,
            alpha=alpha,
            spec_id=i,
        )
        specifications.append(spec_result)
    
    # 5. Identify baseline (maximum pre-periods)
    converged_specs = [s for s in specifications if s.converged]
    if not converged_specs:
        raise ValueError("All specifications failed to converge")
    
    baseline_spec = max(converged_specs, key=lambda x: x.n_pre_periods)
    
    # 6. Compute sensitivity metrics
    atts = [s.att for s in converged_specs]
    att_range = (min(atts), max(atts))
    att_mean = float(np.mean(atts))
    att_std = float(np.std(atts, ddof=1)) if len(atts) > 1 else 0.0
    
    # Sensitivity ratio: range / |baseline|
    sensitivity_ratio = _compute_sensitivity_ratio(atts, baseline_spec.att)
    
    # 7. Assess robustness
    robustness_level = _determine_robustness_level(sensitivity_ratio)
    is_robust = sensitivity_ratio < robustness_threshold
    
    # 8. Check sign and significance stability
    baseline_sign = np.sign(baseline_spec.att)
    all_same_sign = all(np.sign(s.att) == baseline_sign for s in converged_specs)
    n_significant = sum(1 for s in converged_specs if s.is_significant_05)
    all_significant = n_significant == len(converged_specs)
    n_sign_changes = sum(1 for s in converged_specs if np.sign(s.att) != baseline_sign)
    
    # 9. Generate recommendations
    recommendation, detailed_recs, result_warnings = _generate_robustness_recommendations(
        specifications=specifications,
        baseline_spec=baseline_spec,
        sensitivity_ratio=sensitivity_ratio,
        is_robust=is_robust,
        all_same_sign=all_same_sign,
        all_significant=all_significant,
        rolling=rolling,
    )
    
    # 10. Create result object
    result = PrePeriodRobustnessResult(
        specifications=specifications,
        baseline_spec=baseline_spec,
        att_range=att_range,
        att_mean=att_mean,
        att_std=att_std,
        sensitivity_ratio=sensitivity_ratio,
        robustness_level=robustness_level,
        is_robust=is_robust,
        robustness_threshold=robustness_threshold,
        all_same_sign=all_same_sign,
        all_significant=all_significant,
        n_significant=n_significant,
        n_sign_changes=n_sign_changes,
        rolling_method=rolling,
        estimator=estimator,
        n_specifications=len(specifications),
        pre_period_range_tested=pre_period_range,
        recommendation=recommendation,
        detailed_recommendations=detailed_recs,
        result_warnings=result_warnings,
    )
    
    # 11. Print summary if verbose
    if verbose:
        print()
        print(result.summary())
    
    return result


def sensitivity_no_anticipation(
    data: pd.DataFrame,
    y: str,
    ivar: str,
    tvar: str,
    gvar: str | None = None,
    d: str | None = None,
    post: str | None = None,
    rolling: str = 'demean',
    estimator: str = 'ra',
    controls: list[str] | None = None,
    vce: str | None = None,
    cluster_var: str | None = None,
    max_anticipation: int = 3,
    detection_threshold: float = 0.10,
    alpha: float = 0.05,
    verbose: bool = True,
) -> NoAnticipationSensitivityResult:
    """
    Test robustness of ATT estimates to potential anticipation effects.
    
    When the no-anticipation assumption may be violated (e.g., policy announced
    before implementation), units may adjust behavior before formal treatment.
    This function tests robustness by excluding periods immediately before
    treatment from the pre-treatment baseline.
    
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
    d : str, optional
        Treatment indicator for common timing.
    post : str, optional
        Post-treatment indicator for common timing.
    rolling : {'demean', 'detrend'}, default 'demean'
        Transformation method.
    estimator : {'ra', 'ipw', 'ipwra', 'psm'}, default 'ra'
        Estimation method.
    controls : list of str, optional
        Control variable column names.
    vce : str, optional
        Variance estimator type.
    cluster_var : str, optional
        Cluster variable for clustered SE.
    max_anticipation : int, default 3
        Maximum number of periods to test for anticipation effects.
        Tests excluding 0, 1, 2, ..., max_anticipation periods.
    detection_threshold : float, default 0.10
        Threshold for detecting anticipation effects. If relative change
        in ATT exceeds this threshold, anticipation is detected.
    alpha : float, default 0.05
        Significance level for confidence intervals.
    verbose : bool, default True
        Whether to print progress and summary.
    
    Returns
    -------
    NoAnticipationSensitivityResult
        Results containing:
        - estimates: ATT estimates for each exclusion count
        - anticipation_detected: Whether anticipation effects are detected
        - recommended_exclusion: Recommended number of periods to exclude
        - figure: Sensitivity plot (if plot() called)
    
    Notes
    -----
    The no-anticipation assumption requires that, prior to the first
    intervention period for a given treatment cohort, the potential outcomes
    are the same (on average) as in the never treated state.
    
    If policy is announced k periods before implementation, units may adjust
    behavior during periods {g-k, ..., g-1}. By excluding these periods from
    the pre-treatment baseline, we can test whether estimates are robust to
    such anticipation effects.
    
    See Also
    --------
    robustness_pre_periods : General pre-period robustness check.
    sensitivity_analysis : Comprehensive sensitivity analysis.
    """
    from .core import lwdid
    
    # Validate inputs
    _validate_robustness_inputs(data, y, ivar, tvar, gvar, d, post, rolling)
    
    # Determine maximum feasible exclusion
    min_required = 2 if rolling.lower() in ('detrend', 'detrendq') else 1
    max_available = _get_max_pre_periods(data, ivar, tvar, gvar, post)
    max_feasible_exclusion = max(0, max_available - min_required)
    max_anticipation = min(max_anticipation, max_feasible_exclusion)
    
    if max_anticipation < 1:
        warnings.warn(
            "Insufficient pre-treatment periods for anticipation analysis. "
            f"Need at least {min_required + 1} pre-periods, have {max_available}."
        )
    
    if verbose:
        print(f"No-anticipation sensitivity analysis")
        print(f"Testing exclusion range: 0 to {max_anticipation}")
        print("-" * 50)
    
    # Run estimations for each exclusion level
    estimates = []
    result_warnings = []
    
    for exclude in range(max_anticipation + 1):
        if verbose:
            print(f"Testing exclusion = {exclude} periods...")
        
        try:
            # Filter data
            filtered_data = _filter_excluding_periods(
                data, ivar, tvar, gvar, post, exclude
            )
            
            if len(filtered_data) == 0:
                raise ValueError("No data remaining after filtering")
            
            # Run estimation
            result = lwdid(
                data=filtered_data,
                y=y, d=d, ivar=ivar, tvar=tvar, post=post, gvar=gvar,
                rolling=rolling, estimator=estimator,
                controls=controls, vce=vce, cluster_var=cluster_var,
                alpha=alpha,
            )
            
            # Calculate n_pre_periods_used
            n_pre_used = max_available - exclude
            
            estimates.append(AnticipationEstimate(
                excluded_periods=exclude,
                att=result.att,
                se=result.se_att,
                t_stat=result.t_stat,
                pvalue=result.pvalue,
                ci_lower=result.ci_lower,
                ci_upper=result.ci_upper,
                n_pre_periods_used=n_pre_used,
            ))
            
        except Exception as e:
            warnings.warn(f"Exclusion {exclude} failed: {e}")
            result_warnings.append(f"Exclusion {exclude} failed: {e}")
            estimates.append(AnticipationEstimate(
                excluded_periods=exclude,
                att=np.nan, se=np.nan, t_stat=np.nan, pvalue=np.nan,
                ci_lower=np.nan, ci_upper=np.nan,
                n_pre_periods_used=0,
            ))
    
    # Identify baseline (no exclusion)
    baseline = estimates[0] if estimates else AnticipationEstimate(
        excluded_periods=0, att=np.nan, se=np.nan, t_stat=np.nan,
        pvalue=np.nan, ci_lower=np.nan, ci_upper=np.nan, n_pre_periods_used=0
    )
    
    # Detect anticipation effects
    anticipation_detected, recommended_exclusion, detection_method = \
        _detect_anticipation_effects(estimates, baseline, detection_threshold)
    
    # Generate recommendation
    if anticipation_detected:
        recommendation = (
            f"Anticipation effects detected. Consider excluding "
            f"{recommended_exclusion} period(s) before treatment. "
            f"Use lwdid(..., exclude_pre_periods={recommended_exclusion})."
        )
    else:
        recommendation = (
            "No significant anticipation effects detected. "
            "The no-anticipation assumption appears reasonable."
        )
    
    result = NoAnticipationSensitivityResult(
        estimates=estimates,
        baseline_estimate=baseline,
        anticipation_detected=anticipation_detected,
        recommended_exclusion=recommended_exclusion,
        detection_method=detection_method,
        recommendation=recommendation,
        result_warnings=result_warnings,
    )
    
    if verbose:
        print()
        print(result.summary())
    
    return result


def sensitivity_analysis(
    data: pd.DataFrame,
    y: str,
    ivar: str,
    tvar: str,
    gvar: str | None = None,
    d: str | None = None,
    post: str | None = None,
    rolling: str = 'demean',
    estimator: str = 'ra',
    controls: list[str] | None = None,
    vce: str | None = None,
    cluster_var: str | None = None,
    analyses: list[str] | None = None,
    alpha: float = 0.05,
    verbose: bool = True,
) -> ComprehensiveSensitivityResult:
    """
    Perform comprehensive sensitivity analysis for DiD estimation.

    Combines multiple robustness checks into a single analysis, providing
    an overall assessment of estimate reliability across different
    methodological choices.

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
    d : str, optional
        Treatment indicator for common timing.
    post : str, optional
        Post-treatment indicator for common timing.
    rolling : {'demean', 'detrend'}, default 'demean'
        Primary transformation method.
    estimator : {'ra', 'ipw', 'ipwra', 'psm'}, default 'ra'
        Primary estimation method.
    controls : list of str, optional
        Control variable column names.
    vce : str, optional
        Variance estimator type.
    cluster_var : str, optional
        Cluster variable for clustered SE.
    analyses : list of str, optional
        Which analyses to run. Default: all.
        Options: 'pre_periods', 'anticipation', 'transformation', 'estimator'
    alpha : float, default 0.05
        Significance level.
    verbose : bool, default True
        Whether to print progress and summary.

    Returns
    -------
    ComprehensiveSensitivityResult
        Combined results from all sensitivity analyses.

    Notes
    -----
    Four types of sensitivity analysis are available:

    1. **Pre-periods**: Tests stability across different numbers of
       pre-treatment periods used in the transformation.

    2. **Anticipation**: Tests robustness to potential anticipation effects
       by excluding periods immediately before treatment.

    3. **Transformation**: Compares demean and detrend methods to assess
       whether heterogeneous trends may be present.

    4. **Estimator**: Compares RA, IPW, and IPWRA estimators to check
       robustness to propensity score or outcome model misspecification.

    See Also
    --------
    robustness_pre_periods : Pre-period robustness check.
    sensitivity_no_anticipation : Anticipation sensitivity check.
    """
    from .core import lwdid
    
    # Default: run all analyses
    if analyses is None:
        analyses = ['pre_periods', 'anticipation', 'transformation', 'estimator']
    
    # Validate inputs
    _validate_robustness_inputs(data, y, ivar, tvar, gvar, d, post, rolling)
    
    if verbose:
        print("=" * 70)
        print("COMPREHENSIVE SENSITIVITY ANALYSIS")
        print("=" * 70)
        print()
    
    pre_period_result = None
    anticipation_result = None
    transformation_comparison = None
    estimator_comparison = None
    recommendations = []
    
    # 1. Pre-treatment period robustness
    if 'pre_periods' in analyses:
        if verbose:
            print("1. Running pre-treatment period robustness analysis...")
            print()
        try:
            pre_period_result = robustness_pre_periods(
                data=data, y=y, ivar=ivar, tvar=tvar,
                gvar=gvar, d=d, post=post,
                rolling=rolling, estimator=estimator,
                controls=controls, vce=vce, cluster_var=cluster_var,
                alpha=alpha, verbose=False,
            )
            if not pre_period_result.is_robust:
                recommendations.append(
                    f"Pre-period sensitivity detected (ratio={pre_period_result.sensitivity_ratio:.1%}). "
                    "Consider using detrend method or investigating data quality."
                )
        except Exception as e:
            warnings.warn(f"Pre-period analysis failed: {e}")
    
    # 2. No-anticipation sensitivity
    if 'anticipation' in analyses:
        if verbose:
            print("2. Running no-anticipation sensitivity analysis...")
            print()
        try:
            anticipation_result = sensitivity_no_anticipation(
                data=data, y=y, ivar=ivar, tvar=tvar,
                gvar=gvar, d=d, post=post,
                rolling=rolling, estimator=estimator,
                controls=controls, vce=vce, cluster_var=cluster_var,
                alpha=alpha, verbose=False,
            )
            if anticipation_result.anticipation_detected:
                recommendations.append(
                    f"Anticipation effects detected. Consider excluding "
                    f"{anticipation_result.recommended_exclusion} period(s) before treatment."
                )
        except Exception as e:
            warnings.warn(f"Anticipation analysis failed: {e}")
    
    # 3. Transformation comparison
    if 'transformation' in analyses:
        if verbose:
            print("3. Comparing transformation methods (demean vs detrend)...")
            print()
        try:
            # Check if detrend is feasible
            min_pre_detrend = 2
            max_pre = _get_max_pre_periods(data, ivar, tvar, gvar, post)
            
            if max_pre >= min_pre_detrend:
                result_demean = lwdid(
                    data=data, y=y, d=d, ivar=ivar, tvar=tvar, post=post, gvar=gvar,
                    rolling='demean', estimator=estimator,
                    controls=controls, vce=vce, cluster_var=cluster_var, alpha=alpha,
                )
                result_detrend = lwdid(
                    data=data, y=y, d=d, ivar=ivar, tvar=tvar, post=post, gvar=gvar,
                    rolling='detrend', estimator=estimator,
                    controls=controls, vce=vce, cluster_var=cluster_var, alpha=alpha,
                )
                
                transformation_comparison = {
                    'demean_att': result_demean.att,
                    'demean_se': result_demean.se_att,
                    'detrend_att': result_detrend.att,
                    'detrend_se': result_detrend.se_att,
                    'difference': abs(result_demean.att - result_detrend.att),
                }
                
                # Check if difference is substantial
                if abs(result_demean.att) > 1e-10:
                    rel_diff = abs(result_demean.att - result_detrend.att) / abs(result_demean.att)
                    if rel_diff > 0.25:
                        recommendations.append(
                            f"Substantial difference between demean and detrend ({rel_diff:.1%}). "
                            "This suggests heterogeneous trends may be present."
                        )
            else:
                warnings.warn(
                    f"Insufficient pre-periods for detrend comparison "
                    f"(need {min_pre_detrend}, have {max_pre})"
                )
        except Exception as e:
            warnings.warn(f"Transformation comparison failed: {e}")
    
    # 4. Estimator comparison
    if 'estimator' in analyses and controls is not None:
        if verbose:
            print("4. Comparing estimators (RA, IPW, IPWRA)...")
            print()
        try:
            estimator_comparison = {}
            
            for est in ['ra', 'ipw', 'ipwra']:
                try:
                    result_est = lwdid(
                        data=data, y=y, d=d, ivar=ivar, tvar=tvar, post=post, gvar=gvar,
                        rolling=rolling, estimator=est,
                        controls=controls, vce=vce, cluster_var=cluster_var, alpha=alpha,
                    )
                    estimator_comparison[est] = result_est.att
                except Exception:
                    pass
            
            if len(estimator_comparison) >= 2:
                atts = list(estimator_comparison.values())
                estimator_comparison['range'] = max(atts) - min(atts)
                
                # Check if range is substantial
                baseline_att = estimator_comparison.get('ra', atts[0])
                if abs(baseline_att) > 1e-10:
                    rel_range = estimator_comparison['range'] / abs(baseline_att)
                    if rel_range > 0.25:
                        recommendations.append(
                            f"Substantial variation across estimators ({rel_range:.1%}). "
                            "Consider which estimator assumptions are most appropriate."
                        )
        except Exception as e:
            warnings.warn(f"Estimator comparison failed: {e}")
    
    # Generate overall assessment
    issues = []
    if pre_period_result and not pre_period_result.is_robust:
        issues.append("pre-period sensitivity")
    if anticipation_result and anticipation_result.anticipation_detected:
        issues.append("anticipation effects")
    if transformation_comparison and transformation_comparison.get('difference', 0) > 0.25 * abs(transformation_comparison.get('demean_att', 1)):
        issues.append("transformation sensitivity")
    
    if not issues:
        overall_assessment = "Results appear robust across multiple sensitivity checks."
    elif len(issues) == 1:
        overall_assessment = f"Caution: {issues[0]} detected. See recommendations."
    else:
        overall_assessment = f"Multiple concerns: {', '.join(issues)}. Interpret with caution."
    
    if not recommendations:
        recommendations.append("No major robustness concerns identified.")
    
    result = ComprehensiveSensitivityResult(
        pre_period_result=pre_period_result,
        anticipation_result=anticipation_result,
        transformation_comparison=transformation_comparison,
        estimator_comparison=estimator_comparison,
        overall_assessment=overall_assessment,
        recommendations=recommendations,
    )
    
    if verbose:
        print()
        print(result.summary())
    
    return result


# =============================================================================
# Convenience function for plotting
# =============================================================================

def plot_sensitivity(
    result: PrePeriodRobustnessResult | NoAnticipationSensitivityResult,
    show_ci: bool = True,
    show_baseline: bool = True,
    highlight_significant: bool = True,
    figsize: tuple[float, float] = (10, 6),
    ax: Any = None,
) -> Any:
    """
    Visualize sensitivity analysis results.
    
    Creates a plot showing how ATT estimates vary across different
    specifications, with confidence intervals and significance indicators.
    
    Parameters
    ----------
    result : PrePeriodRobustnessResult or NoAnticipationSensitivityResult
        Result object from sensitivity analysis.
    show_ci : bool, default True
        Whether to show confidence intervals.
    show_baseline : bool, default True
        Whether to show baseline reference line.
    highlight_significant : bool, default True
        Whether to highlight significant estimates.
    figsize : tuple, default (10, 6)
        Figure size in inches.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
        
    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    if isinstance(result, PrePeriodRobustnessResult):
        return result.plot(show_ci=show_ci, show_baseline=show_baseline, figsize=figsize, ax=ax)
    elif isinstance(result, NoAnticipationSensitivityResult):
        return result.plot(show_ci=show_ci, figsize=figsize, ax=ax)
    else:
        raise TypeError(f"Unsupported result type: {type(result)}")

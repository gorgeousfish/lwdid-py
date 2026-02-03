"""
Parallel trends assumption testing for staggered difference-in-differences.

This module implements statistical tests for the parallel trends assumption
using pre-treatment ATT estimates. Under the null hypothesis of parallel
trends and no anticipation, pre-treatment ATT estimates should be
statistically indistinguishable from zero.

Two types of tests are provided:

1. **Individual t-tests**: Test H0: ATT_e = 0 for each pre-treatment
   event time e < -1 (excluding the anchor point).

2. **Joint F-test**: Test H0: all pre-treatment ATT = 0 simultaneously.
   This is the primary test for parallel trends.

The anchor point (e = -1) is excluded from testing because it is set to
zero by construction of the rolling transformation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy.stats as stats

if TYPE_CHECKING:
    from .estimation_pre import PreTreatmentEffect


@dataclass
class ParallelTrendsTestResult:
    """
    Container for parallel trends test results.

    Stores both individual t-tests for each pre-treatment period and
    the joint F-test for the null hypothesis that all pre-treatment
    ATT estimates equal zero.

    Attributes
    ----------
    individual_tests : pd.DataFrame
        DataFrame with columns: event_time, att, se, t_stat, pvalue.
        Contains test results for each pre-treatment period (excluding
        anchor point).
    joint_f_stat : float
        F-statistic for joint test H0: all pre-treatment ATT = 0.
    joint_pvalue : float
        P-value for joint F-test.
    joint_df1 : int
        Numerator degrees of freedom (number of pre-treatment periods
        included in the test).
    joint_df2 : int
        Denominator degrees of freedom.
    n_pre_periods : int
        Number of pre-treatment periods included in test.
    excluded_periods : list
        Event times excluded due to missing SE, anchor point, or
        other issues.
    reject_null : bool
        True if joint test rejects H0 at the specified alpha level.
    alpha : float
        Significance level used for the test.

    Notes
    -----
    Rejection of the null hypothesis suggests potential violation of
    the parallel trends assumption. However, failure to reject does
    not prove parallel trends holds - it may simply reflect low power.
    """
    individual_tests: pd.DataFrame
    joint_f_stat: float
    joint_pvalue: float
    joint_df1: int
    joint_df2: int
    n_pre_periods: int
    excluded_periods: list = field(default_factory=list)
    reject_null: bool = False
    alpha: float = 0.05


def _compute_individual_t_tests(
    pre_treatment_effects: list[PreTreatmentEffect],
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, list[int]]:
    """
    Compute individual t-tests for each pre-treatment ATT.

    Parameters
    ----------
    pre_treatment_effects : list of PreTreatmentEffect
        Pre-treatment effect estimates from estimate_pre_treatment_effects.
    alpha : float, default=0.05
        Significance level for individual tests.

    Returns
    -------
    individual_tests : pd.DataFrame
        DataFrame with columns: event_time, cohort, att, se, t_stat,
        pvalue, significant.
    excluded_periods : list of int
        Event times excluded from testing.
    """
    test_results = []
    excluded_periods = []

    for effect in pre_treatment_effects:
        # Skip anchor points (e = -1)
        if effect.is_anchor:
            excluded_periods.append(effect.event_time)
            continue

        # Skip if SE is missing or zero
        if np.isnan(effect.se) or effect.se <= 0:
            excluded_periods.append(effect.event_time)
            continue

        # Skip if ATT is missing
        if np.isnan(effect.att):
            excluded_periods.append(effect.event_time)
            continue

        # Compute t-statistic
        t_stat = effect.att / effect.se

        # Use t-distribution with appropriate df
        # If df_inference not available, use large-sample approximation
        df = effect.n_treated + effect.n_control - 2
        if df <= 0:
            df = 1000  # Large-sample approximation

        pvalue = 2 * (1 - stats.t.cdf(abs(t_stat), df))

        test_results.append({
            'event_time': effect.event_time,
            'cohort': effect.cohort,
            'period': effect.period,
            'att': effect.att,
            'se': effect.se,
            't_stat': t_stat,
            'pvalue': pvalue,
            'significant': pvalue < alpha,
            'n_treated': effect.n_treated,
            'n_control': effect.n_control,
        })

    if len(test_results) == 0:
        return pd.DataFrame(columns=[
            'event_time', 'cohort', 'period', 'att', 'se',
            't_stat', 'pvalue', 'significant', 'n_treated', 'n_control'
        ]), excluded_periods

    return pd.DataFrame(test_results), excluded_periods


def _compute_joint_f_test(
    individual_tests: pd.DataFrame,
    alpha: float = 0.05,
) -> tuple[float, float, int, int]:
    """
    Compute joint F-test for H0: all pre-treatment ATT = 0.

    Uses a Wald-type test statistic:
    F = (1/q) * sum((ATT_j / SE_j)^2)

    where q is the number of pre-treatment periods being tested.

    Parameters
    ----------
    individual_tests : pd.DataFrame
        DataFrame with columns: att, se, t_stat.
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    f_stat : float
        F-statistic.
    pvalue : float
        P-value from F-distribution.
    df1 : int
        Numerator degrees of freedom.
    df2 : int
        Denominator degrees of freedom.

    Notes
    -----
    This is a simplified F-test that assumes independence across
    pre-treatment periods. For a more rigorous test accounting for
    correlation, a full variance-covariance matrix would be needed.
    """
    if len(individual_tests) == 0:
        return np.nan, np.nan, 0, 0

    # Number of restrictions (pre-treatment periods)
    q = len(individual_tests)

    # Sum of squared t-statistics
    t_stats = individual_tests['t_stat'].values
    sum_t2 = np.sum(t_stats ** 2)

    # F-statistic: average of squared t-statistics
    f_stat = sum_t2 / q

    # Degrees of freedom
    df1 = q
    # Use average sample size for df2 approximation
    avg_n = individual_tests['n_treated'].mean() + individual_tests['n_control'].mean()
    df2 = max(int(avg_n - 2), 1)

    # P-value from F-distribution
    pvalue = 1 - stats.f.cdf(f_stat, df1, df2)

    return f_stat, pvalue, df1, df2


def _compute_joint_wald_test(
    individual_tests: pd.DataFrame,
    alpha: float = 0.05,
) -> tuple[float, float, int]:
    """
    Compute joint Wald test (chi-squared) for H0: all pre-treatment ATT = 0.

    Uses a Wald statistic:
    W = sum((ATT_j / SE_j)^2) ~ chi^2(q)

    under the null hypothesis.

    Parameters
    ----------
    individual_tests : pd.DataFrame
        DataFrame with columns: att, se, t_stat.
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    wald_stat : float
        Wald statistic.
    pvalue : float
        P-value from chi-squared distribution.
    df : int
        Degrees of freedom.
    """
    if len(individual_tests) == 0:
        return np.nan, np.nan, 0

    # Number of restrictions
    q = len(individual_tests)

    # Wald statistic: sum of squared t-statistics
    t_stats = individual_tests['t_stat'].values
    wald_stat = np.sum(t_stats ** 2)

    # P-value from chi-squared distribution
    pvalue = 1 - stats.chi2.cdf(wald_stat, q)

    return wald_stat, pvalue, q


def run_parallel_trends_test(
    pre_treatment_effects: list[PreTreatmentEffect],
    alpha: float = 0.05,
    test_type: str = 'f',
    min_pre_periods: int = 2,
) -> ParallelTrendsTestResult:
    """
    Test the parallel trends assumption using pre-treatment ATT estimates.

    Performs both individual t-tests for each pre-treatment period and
    a joint test for the null hypothesis that all pre-treatment ATT
    estimates equal zero.

    Parameters
    ----------
    pre_treatment_effects : list of PreTreatmentEffect
        Pre-treatment effect estimates from estimate_pre_treatment_effects.
    alpha : float, default=0.05
        Significance level for hypothesis tests.
    test_type : str, default='f'
        Type of joint test: 'f' for F-test, 'wald' for Wald/chi-squared test.
    min_pre_periods : int, default=2
        Minimum number of pre-treatment periods required for joint test.
        If fewer periods are available, a warning is issued.

    Returns
    -------
    ParallelTrendsTestResult
        Test results including individual t-tests and joint test.

    Raises
    ------
    ValueError
        If pre_treatment_effects is empty or test_type is invalid.

    See Also
    --------
    estimate_pre_treatment_effects : Estimate pre-treatment ATT.
    PreTreatmentEffect : Container for pre-treatment effect estimates.

    Notes
    -----
    The anchor point (event time e = -1) is excluded from testing because
    it is set to zero by construction of the rolling transformation.

    Interpretation:
    - If reject_null is True: Evidence against parallel trends assumption.
      Pre-treatment effects are significantly different from zero.
    - If reject_null is False: No evidence against parallel trends.
      However, this does not prove parallel trends holds - it may
      simply reflect low statistical power.
    """
    if len(pre_treatment_effects) == 0:
        raise ValueError("pre_treatment_effects is empty")

    if test_type not in ('f', 'wald'):
        raise ValueError(f"Invalid test_type: {test_type}. Must be 'f' or 'wald'.")

    # =========================================================================
    # Compute Individual t-tests
    # =========================================================================
    individual_tests, excluded_periods = _compute_individual_t_tests(
        pre_treatment_effects, alpha=alpha
    )

    n_pre_periods = len(individual_tests)

    # =========================================================================
    # Check Minimum Pre-treatment Periods
    # =========================================================================
    if n_pre_periods < min_pre_periods:
        warnings.warn(
            f"Only {n_pre_periods} pre-treatment period(s) available for testing "
            f"(minimum recommended: {min_pre_periods}). Joint test may have low power.",
            UserWarning
        )

    if n_pre_periods == 0:
        warnings.warn(
            "No valid pre-treatment periods for testing (all excluded). "
            "Cannot perform parallel trends test.",
            UserWarning
        )
        return ParallelTrendsTestResult(
            individual_tests=individual_tests,
            joint_f_stat=np.nan,
            joint_pvalue=np.nan,
            joint_df1=0,
            joint_df2=0,
            n_pre_periods=0,
            excluded_periods=excluded_periods,
            reject_null=False,
            alpha=alpha,
        )

    # =========================================================================
    # Compute Joint Test
    # =========================================================================
    if test_type == 'f':
        joint_stat, joint_pvalue, df1, df2 = _compute_joint_f_test(
            individual_tests, alpha=alpha
        )
    else:  # wald
        joint_stat, joint_pvalue, df1 = _compute_joint_wald_test(
            individual_tests, alpha=alpha
        )
        df2 = 0  # Not applicable for chi-squared

    # =========================================================================
    # Determine Rejection
    # =========================================================================
    reject_null = joint_pvalue < alpha if not np.isnan(joint_pvalue) else False

    return ParallelTrendsTestResult(
        individual_tests=individual_tests,
        joint_f_stat=joint_stat,
        joint_pvalue=joint_pvalue,
        joint_df1=df1,
        joint_df2=df2,
        n_pre_periods=n_pre_periods,
        excluded_periods=excluded_periods,
        reject_null=reject_null,
        alpha=alpha,
    )


def summarize_parallel_trends_test(
    test_result: ParallelTrendsTestResult,
) -> str:
    """
    Generate a human-readable summary of parallel trends test results.

    Parameters
    ----------
    test_result : ParallelTrendsTestResult
        Test results from test_parallel_trends.

    Returns
    -------
    str
        Formatted summary string.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("Parallel Trends Test Results")
    lines.append("=" * 60)
    lines.append("")

    # Joint test results
    lines.append("Joint Test (H0: all pre-treatment ATT = 0)")
    lines.append("-" * 40)
    lines.append(f"  F-statistic:     {test_result.joint_f_stat:.4f}")
    lines.append(f"  P-value:         {test_result.joint_pvalue:.4f}")
    lines.append(f"  DF (num, denom): ({test_result.joint_df1}, {test_result.joint_df2})")
    lines.append(f"  Alpha:           {test_result.alpha:.2f}")
    lines.append(f"  Reject H0:       {'Yes' if test_result.reject_null else 'No'}")
    lines.append("")

    # Individual tests
    lines.append("Individual Tests by Event Time")
    lines.append("-" * 40)

    if len(test_result.individual_tests) > 0:
        for _, row in test_result.individual_tests.iterrows():
            sig_marker = "*" if row['significant'] else ""
            lines.append(
                f"  e = {int(row['event_time']):3d}: "
                f"ATT = {row['att']:8.4f}, "
                f"SE = {row['se']:7.4f}, "
                f"t = {row['t_stat']:6.3f}, "
                f"p = {row['pvalue']:.4f} {sig_marker}"
            )
    else:
        lines.append("  No valid pre-treatment periods for testing.")

    lines.append("")

    # Excluded periods
    if test_result.excluded_periods:
        lines.append(f"Excluded event times: {test_result.excluded_periods}")
        lines.append("  (anchor points and periods with missing SE)")
    lines.append("")

    # Interpretation
    lines.append("Interpretation")
    lines.append("-" * 40)
    if test_result.reject_null:
        lines.append("  The joint test REJECTS the null hypothesis of parallel trends.")
        lines.append("  Pre-treatment effects are significantly different from zero,")
        lines.append("  suggesting potential violation of the parallel trends assumption.")
    else:
        lines.append("  The joint test FAILS TO REJECT the null hypothesis.")
        lines.append("  No significant evidence against parallel trends.")
        lines.append("  Note: This does not prove parallel trends holds.")

    lines.append("=" * 60)

    return "\n".join(lines)


# Alias for convenience - use parallel_trends_test instead of test_parallel_trends
# to avoid pytest collecting it as a test function
parallel_trends_test = run_parallel_trends_test

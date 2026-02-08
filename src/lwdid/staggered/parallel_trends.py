"""
Statistical tests for the parallel trends assumption in staggered DiD.

This module provides hypothesis tests to assess the parallel trends assumption
using pre-treatment ATT estimates. Under the null hypothesis of parallel
trends and no anticipation, pre-treatment ATT estimates should be
statistically indistinguishable from zero.

Two complementary testing approaches are implemented:

1. **Individual t-tests**: Test H0: ATT_e = 0 for each pre-treatment event
   time e < -1, excluding the anchor point which is zero by construction.

2. **Joint F-test (or Wald test)**: Test H0: all pre-treatment ATT = 0
   simultaneously. This is the primary diagnostic for parallel trends.

The anchor point at event time e = -1 is excluded from testing because
it is set to zero by construction of the rolling transformation.

Notes
-----
The joint F-test assumes independence across pre-treatment periods. For
settings with substantial serial correlation, the Wald (chi-squared) test
may be preferred. Both tests are asymptotically valid under standard
regularity conditions.

Rejection of the null hypothesis suggests potential violation of the
parallel trends assumption. However, failure to reject does not prove
parallel trends holds, as the test may simply lack statistical power.
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
    Compute individual t-tests for each pre-treatment ATT estimate.

    Tests H0: ATT_e = 0 for each pre-treatment event time, using a
    t-distribution with degrees of freedom based on sample sizes.

    Parameters
    ----------
    pre_treatment_effects : list of PreTreatmentEffect
        Pre-treatment effect estimates from estimate_pre_treatment_effects.
    alpha : float, default=0.05
        Significance level for determining the 'significant' flag.

    Returns
    -------
    individual_tests : pd.DataFrame
        DataFrame with columns: event_time, cohort, period, att, se,
        t_stat, pvalue, significant, n_treated, n_control. Each row
        represents test results for one pre-treatment period.
    excluded_periods : list of int
        Event times excluded from testing due to anchor point status,
        missing standard errors, or invalid ATT values.
    """
    test_results = []
    excluded_periods = []

    for effect in pre_treatment_effects:
        # Anchor points are zero by construction and provide no test information.
        if effect.is_anchor:
            excluded_periods.append(effect.event_time)
            continue

        # Invalid SE prevents reliable t-test computation.
        if np.isnan(effect.se) or effect.se <= 0:
            excluded_periods.append(effect.event_time)
            continue

        # Missing ATT indicates estimation failure for this period.
        if np.isnan(effect.att):
            excluded_periods.append(effect.event_time)
            continue

        t_stat = effect.att / effect.se

        # Approximate df using pooled sample size; default to 1000 for
        # asymptotic approximation when sample is too small.
        df = effect.n_treated + effect.n_control - 2
        if df <= 0:
            df = 1000

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

    Uses a Wald-type F-statistic computed as the average of squared
    t-statistics across pre-treatment periods.

    Parameters
    ----------
    individual_tests : pd.DataFrame
        DataFrame with columns: att, se, t_stat, n_treated, n_control.
    alpha : float, default=0.05
        Significance level for hypothesis testing. Currently unused
        but retained for API consistency with related functions.

    Returns
    -------
    f_stat : float
        F-statistic computed as (1/q) * sum(t_j^2).
    pvalue : float
        P-value from F(q, df2) distribution.
    df1 : int
        Numerator degrees of freedom (number of restrictions q).
    df2 : int
        Denominator degrees of freedom (based on average sample size).

    Notes
    -----
    This is a simplified F-test that assumes independence across
    pre-treatment periods. For settings with substantial correlation
    between period-specific estimates, a test based on the full
    variance-covariance matrix would provide more accurate inference.
    """
    if len(individual_tests) == 0:
        return np.nan, np.nan, 0, 0

    q = len(individual_tests)
    t_stats = individual_tests['t_stat'].values
    sum_t2 = np.sum(t_stats ** 2)

    # F-statistic as average of squared t-statistics follows F(q, df2).
    f_stat = sum_t2 / q

    df1 = q
    # Approximate denominator df using average sample size across periods.
    avg_n = individual_tests['n_treated'].mean() + individual_tests['n_control'].mean()
    df2 = max(int(avg_n - 2), 1)

    pvalue = 1 - stats.f.cdf(f_stat, df1, df2)

    return f_stat, pvalue, df1, df2


def _compute_joint_wald_test(
    individual_tests: pd.DataFrame,
    alpha: float = 0.05,
) -> tuple[float, float, int]:
    """
    Compute joint Wald test for H0: all pre-treatment ATT = 0.

    The Wald statistic is the sum of squared t-statistics, which follows
    a chi-squared distribution with q degrees of freedom under the null.

    Parameters
    ----------
    individual_tests : pd.DataFrame
        DataFrame with columns: att, se, t_stat.
    alpha : float, default=0.05
        Significance level for hypothesis testing. Currently unused
        but retained for API consistency with related functions.

    Returns
    -------
    wald_stat : float
        Wald statistic computed as sum(t_j^2).
    pvalue : float
        P-value from chi-squared(q) distribution.
    df : int
        Degrees of freedom (number of pre-treatment periods tested).

    Notes
    -----
    The Wald test is asymptotically equivalent to the F-test but uses
    the chi-squared distribution rather than the F-distribution. This
    test may be preferred when sample sizes are large or when comparing
    results with other software implementations.
    """
    if len(individual_tests) == 0:
        return np.nan, np.nan, 0

    q = len(individual_tests)
    t_stats = individual_tests['t_stat'].values

    # Sum of squared t-statistics follows chi-squared(q) under the null.
    wald_stat = np.sum(t_stats ** 2)
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
    The anchor point at event time e = -1 is excluded from testing because
    it is exactly zero by construction of the rolling transformation.

    Test interpretation guidelines:

    - **reject_null = True**: Evidence against parallel trends. The
      pre-treatment ATT estimates are jointly significantly different
      from zero, suggesting potential violation of the identifying
      assumption.

    - **reject_null = False**: No evidence against parallel trends.
      This does not prove the assumption holds; the test may lack
      sufficient power to detect violations, particularly with few
      pre-treatment periods or small sample sizes.
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

    Produces a formatted text report containing joint test statistics,
    individual period-specific test results, and interpretation guidance.

    Parameters
    ----------
    test_result : ParallelTrendsTestResult
        Test results from run_parallel_trends_test.

    Returns
    -------
    str
        Multi-line formatted summary including:
        - Joint test F-statistic, p-value, and degrees of freedom
        - Individual t-tests for each pre-treatment event time
        - List of excluded event times (anchor points and missing data)
        - Plain-language interpretation of the test outcome

    See Also
    --------
    run_parallel_trends_test : Compute parallel trends test statistics.
    ParallelTrendsTestResult : Container for test results.
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


# Alternative function name following common API naming conventions.
parallel_trends_test = run_parallel_trends_test

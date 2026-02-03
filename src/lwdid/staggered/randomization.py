"""
Randomization inference for staggered difference-in-differences estimation.

This module implements permutation-based inference procedures for staggered
DiD settings with multiple treatment cohorts. Randomization inference provides
finite-sample valid p-values without distributional assumptions by comparing
the observed test statistic to its null distribution generated through random
reassignment of treatment cohort labels.

Notes
-----
Randomization inference is particularly useful in staggered DiD settings with
a small number of treated or control units where asymptotic approximations may
be unreliable. The permutation distribution preserves the cohort structure by
shuffling which units belong to each cohort while maintaining the total number
of units per cohort.

For overall effect inference, never-treated units are required as a consistent
reference group across permutations. When testing cohort-specific or (g, r)-
specific effects, the target cohort must be present after each permutation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import warnings

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from ..exceptions import RandomizationError


@dataclass
class StaggeredRIResult:
    """
    Container for staggered randomization inference results.

    This dataclass stores the output from permutation or bootstrap-based
    randomization inference procedures, including the p-value, replication
    diagnostics, and the full distribution of resampled statistics.

    Attributes
    ----------
    p_value : float
        Two-sided p-value for the null hypothesis that the ATT equals zero.
    ri_method : str
        Resampling method used, either 'permutation' or 'bootstrap'.
    ri_reps : int
        Number of requested replications.
    ri_valid : int
        Number of valid replications that produced non-missing statistics.
    ri_failed : int
        Number of replications that failed due to estimation errors.
    observed_stat : float
        Observed ATT estimate being tested.
    permutation_stats : NDArray[np.float64]
        Array of ATT statistics from valid replications, excluding NaN
        values from failed replications.

    See Also
    --------
    randomization_inference_staggered : Main function that produces this result.
    ri_overall_effect : Convenience wrapper for overall effect inference.
    ri_cohort_effect : Convenience wrapper for cohort-specific inference.
    """

    p_value: float
    ri_method: str
    ri_reps: int
    ri_valid: int
    ri_failed: int
    observed_stat: float
    permutation_stats: NDArray[np.float64]

    def __repr__(self) -> str:
        """Return a concise string representation of the result."""
        return (
            f"StaggeredRIResult(p_value={self.p_value:.4f}, "
            f"method='{self.ri_method}', valid={self.ri_valid}/{self.ri_reps})"
        )


def randomization_inference_staggered(
    data: pd.DataFrame,
    gvar: str,
    ivar: str,
    tvar: str,
    y: str,
    cohorts: list[int],
    observed_att: float,
    target: Literal['overall', 'cohort', 'cohort_time'] = 'overall',
    target_cohort: int | None = None,
    target_period: int | None = None,
    ri_method: Literal['permutation', 'bootstrap'] = 'permutation',
    rireps: int = 1000,
    seed: int | None = None,
    rolling: str = 'demean',
    controls: list[str] | None = None,
    vce: str | None = None,
    cluster_var: str | None = None,
    n_never_treated: int = 0,
) -> StaggeredRIResult:
    """
    Perform randomization inference for staggered DiD estimation.

    Tests the null hypothesis that the average treatment effect on the treated
    equals zero by permuting or bootstrapping treatment cohort assignments and
    computing the empirical distribution of the test statistic.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format with unit, time, cohort, and outcome columns.
    gvar : str
        Column name for the treatment cohort variable. Units with missing,
        zero, or infinite values are treated as never-treated.
    ivar : str
        Column name for the unit identifier.
    tvar : str
        Column name for the time period variable.
    y : str
        Column name for the outcome variable.
    cohorts : list of int
        List of treatment cohort values present in the data.
    observed_att : float
        Observed ATT estimate to be tested against the null hypothesis.
    target : {'overall', 'cohort', 'cohort_time'}, default 'overall'
        Aggregation level for the target effect:

        - 'overall': Overall weighted average effect across all cohorts
        - 'cohort': Cohort-specific average effect (requires target_cohort)
        - 'cohort_time': Effect for a specific (g, r) pair (requires both
          target_cohort and target_period)

    target_cohort : int, optional
        Target cohort for 'cohort' or 'cohort_time' targets.
    target_period : int, optional
        Target time period for 'cohort_time' target.
    ri_method : {'permutation', 'bootstrap'}, default 'permutation'
        Resampling method for generating the null distribution:

        - 'permutation': Without-replacement permutation preserving cohort
          sizes (Fisher exact randomization inference)
        - 'bootstrap': With-replacement sampling from unit cohort assignments

    rireps : int, default 1000
        Number of resampling replications.
    seed : int, optional
        Random seed for reproducibility of the resampling procedure.
    rolling : {'demean', 'detrend'}, default 'demean'
        Transformation method for removing pre-treatment variation:

        - 'demean': Subtract pre-treatment mean from each unit
        - 'detrend': Remove unit-specific linear time trend

    controls : list of str, optional
        Column names for control variables to include in estimation.
    vce : str, optional
        Variance-covariance estimator type for standard errors.
    cluster_var : str, optional
        Column name for clustering standard errors.
    n_never_treated : int, default 0
        Number of never-treated units. Required for overall effect inference
        to ensure a consistent reference group across permutations.

    Returns
    -------
    StaggeredRIResult
        Dataclass containing the p-value, replication counts, observed
        statistic, and array of permutation statistics.

    Raises
    ------
    RandomizationError
        If input parameters are invalid, if there are insufficient units
        for inference, or if too few replications produce valid estimates.

    See Also
    --------
    ri_overall_effect : Convenience wrapper for overall effect inference.
    ri_cohort_effect : Convenience wrapper for cohort-specific inference.

    Notes
    -----
    The permutation procedure shuffles cohort assignments across units while
    preserving the marginal distribution of cohorts. This generates the null
    distribution under the sharp null hypothesis that treatment has no effect
    on any unit.

    For inference on overall effects, never-treated units must be present to
    serve as a consistent control group across all permutations. When testing
    cohort-specific effects, replications where the target cohort is absent
    after permutation are marked as invalid.

    The two-sided p-value is computed as:

    .. math::

        p = \\frac{1}{R} \\sum_{r=1}^{R} \\mathbf{1}\\{|\\hat{\\tau}^{(r)}| \\geq |\\hat{\\tau}^{obs}|\\}

    where :math:`R` is the number of valid replications and
    :math:`\\hat{\\tau}^{(r)}` is the ATT from replication :math:`r`.

    A minimum of 50 valid replications (or 10% of rireps, whichever is larger)
    is required to ensure reliable p-value computation.
    """
    # =========================================================================
    # Input Validation
    # =========================================================================
    if rireps <= 0:
        raise RandomizationError("rireps must be positive")

    if target == 'cohort' and target_cohort is None:
        raise RandomizationError("target_cohort required when target='cohort'")

    if target == 'cohort_time' and (target_cohort is None or target_period is None):
        raise RandomizationError(
            "target_cohort and target_period required when target='cohort_time'"
        )

    if ri_method not in ('permutation', 'bootstrap'):
        raise RandomizationError("ri_method must be 'permutation' or 'bootstrap'")

    # Overall effect requires never-treated units as consistent reference group
    # across all permutations; without them, the control group varies arbitrarily.
    if target == 'overall' and n_never_treated == 0:
        raise RandomizationError(
            "RI for overall effect requires never treated units. "
            "Use target='cohort_time' when no NT units exist."
        )

    rolling_lower = rolling.lower()
    if rolling_lower not in ('demean', 'detrend'):
        raise RandomizationError(
            f"rolling must be 'demean' or 'detrend', got '{rolling}'"
        )

    # Seed ensures reproducibility of permutation sequence across runs.
    rng = np.random.default_rng(seed)
    
    # Cohort assignment is time-invariant; extract once per unit for efficiency.
    unit_gvar = data.groupby(ivar)[gvar].first()
    unit_ids = unit_gvar.index.tolist()
    n_units = len(unit_ids)

    # Permutation distribution requires sufficient units for meaningful inference.
    if n_units < 4:
        raise RandomizationError(f"Too few units for RI: N={n_units}")

    # Deferred imports to avoid circular dependencies between submodules.
    from .transformations import transform_staggered_demean, transform_staggered_detrend
    from .estimation import estimate_cohort_time_effects
    from .aggregation import aggregate_to_cohort, aggregate_to_overall, get_cohorts

    transform_func = (
        transform_staggered_demean
        if rolling_lower == 'demean'
        else transform_staggered_detrend
    )

    T_max = int(data[tvar].max())
    
    # =========================================================================
    # Resampling Loop
    # =========================================================================
    # Pre-fill with NaN to distinguish failed replications from valid zeros;
    # NaN entries are excluded when computing the empirical p-value.
    perm_stats = np.empty(rireps, dtype=float)
    perm_stats.fill(np.nan)

    for rep in range(rireps):
        try:
            if ri_method == 'permutation':
                # Without-replacement shuffle preserves marginal cohort distribution.
                perm_idx = rng.permutation(n_units)
                perm_gvar = unit_gvar.values[perm_idx]
            else:
                # With-replacement bootstrap allows cohort frequency variation.
                boot_idx = rng.integers(0, n_units, size=n_units)
                perm_gvar = unit_gvar.values[boot_idx]

            perm_gvar_mapping = dict(zip(unit_ids, perm_gvar))

            data_perm = data.copy()
            data_perm[gvar] = data_perm[ivar].map(perm_gvar_mapping)

            try:
                data_transformed = transform_func(
                    data_perm, y, ivar, tvar, gvar
                )
            except (ValueError, KeyError):
                perm_stats[rep] = np.nan
                continue

            if target == 'overall':
                try:
                    result = aggregate_to_overall(
                        data_transformed, gvar, ivar, tvar,
                        transform_type=rolling_lower,
                        vce=vce,
                        cluster_var=cluster_var,
                    )
                    perm_stats[rep] = result.att
                except (ValueError, KeyError):
                    perm_stats[rep] = np.nan

            elif target == 'cohort':
                try:
                    perm_cohorts = get_cohorts(data_transformed, gvar, ivar)
                    # Permutation may reassign all units away from target cohort.
                    if target_cohort not in perm_cohorts:
                        perm_stats[rep] = np.nan
                        continue

                    results = aggregate_to_cohort(
                        data_transformed, gvar, ivar, tvar,
                        cohorts=[target_cohort], T_max=T_max,
                        transform_type=rolling_lower,
                        vce=vce,
                        cluster_var=cluster_var,
                    )
                    if results:
                        perm_stats[rep] = results[0].att
                    else:
                        perm_stats[rep] = np.nan
                except (ValueError, KeyError):
                    perm_stats[rep] = np.nan

            else:
                # target == 'cohort_time'
                try:
                    ct_results = estimate_cohort_time_effects(
                        data_transformed, gvar, ivar, tvar,
                        controls=controls,
                        vce=vce,
                        cluster_var=cluster_var,
                        transform_type=rolling_lower,
                    )
                    target_result = [
                        r for r in ct_results
                        if r.cohort == target_cohort and r.period == target_period
                    ]
                    if target_result:
                        perm_stats[rep] = target_result[0].att
                    else:
                        perm_stats[rep] = np.nan
                except (ValueError, KeyError):
                    perm_stats[rep] = np.nan

        except Exception:
            # Catch-all for unexpected failures; NaN exclusion handles these.
            perm_stats[rep] = np.nan
    
    # =========================================================================
    # P-Value Computation
    # =========================================================================
    valid_stats = perm_stats[~np.isnan(perm_stats)]
    n_valid = len(valid_stats)
    n_failed = rireps - n_valid

    # Insufficient valid replications yield unreliable p-value estimates.
    min_valid = max(50, int(0.1 * rireps))
    if n_valid < min_valid:
        raise RandomizationError(
            f"Insufficient valid RI replications: {n_valid}/{rireps} "
            f"(need at least {min_valid}). "
            f"Consider increasing rireps or checking data quality."
        )

    # Two-sided p-value under sharp null hypothesis of no treatment effect.
    p_value = float((np.abs(valid_stats) >= abs(observed_att)).mean())

    # High failure rate may indicate data quality issues or model misspecification.
    if n_failed / rireps > 0.1:
        warnings.warn(
            f"Staggered RI: {n_failed}/{rireps} replications failed "
            f"({n_failed/rireps:.1%}). P-value computed using {n_valid} "
            f"valid replications.",
            UserWarning
        )
    
    return StaggeredRIResult(
        p_value=p_value,
        ri_method=ri_method,
        ri_reps=rireps,
        ri_valid=n_valid,
        ri_failed=n_failed,
        observed_stat=observed_att,
        permutation_stats=valid_stats
    )


def ri_overall_effect(
    data: pd.DataFrame,
    gvar: str,
    ivar: str,
    tvar: str,
    y: str,
    observed_att: float,
    rolling: str = 'demean',
    ri_method: str = 'permutation',
    rireps: int = 1000,
    seed: int | None = None,
    vce: str | None = None,
    cluster_var: str | None = None,
) -> StaggeredRIResult:
    """
    Perform randomization inference for the overall weighted ATT.

    This is a convenience wrapper around `randomization_inference_staggered`
    for testing the aggregate effect across all treatment cohorts. The overall
    effect is a cohort-share-weighted average of cohort-specific ATTs.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format with unit, time, cohort, and outcome columns.
    gvar : str
        Column name for the treatment cohort variable.
    ivar : str
        Column name for the unit identifier.
    tvar : str
        Column name for the time period.
    y : str
        Column name for the outcome variable.
    observed_att : float
        Observed overall ATT estimate to test.
    rolling : {'demean', 'detrend'}, default 'demean'
        Transformation method for pre-treatment variation removal.
    ri_method : {'permutation', 'bootstrap'}, default 'permutation'
        Resampling method for null distribution generation.
    rireps : int, default 1000
        Number of resampling replications.
    seed : int, optional
        Random seed for reproducibility.
    vce : str, optional
        Variance-covariance estimator type.
    cluster_var : str, optional
        Column name for clustering standard errors.

    Returns
    -------
    StaggeredRIResult
        Randomization inference results including p-value and diagnostics.

    See Also
    --------
    randomization_inference_staggered : Full-featured inference function.
    ri_cohort_effect : Inference for cohort-specific effects.
    """
    from .transformations import get_cohorts
    from ..validation import is_never_treated

    unit_gvar = data.groupby(ivar)[gvar].first()
    cohorts = get_cohorts(data, gvar, ivar)

    # Identify never-treated units using the single source of truth function.
    nt_mask = unit_gvar.apply(is_never_treated)
    n_nt = int(nt_mask.sum())

    return randomization_inference_staggered(
        data=data,
        gvar=gvar,
        ivar=ivar,
        tvar=tvar,
        y=y,
        cohorts=cohorts,
        observed_att=observed_att,
        target='overall',
        ri_method=ri_method,
        rireps=rireps,
        seed=seed,
        rolling=rolling,
        n_never_treated=n_nt,
        vce=vce,
        cluster_var=cluster_var,
    )


def ri_cohort_effect(
    data: pd.DataFrame,
    gvar: str,
    ivar: str,
    tvar: str,
    y: str,
    target_cohort: int,
    observed_att: float,
    rolling: str = 'demean',
    ri_method: str = 'permutation',
    rireps: int = 1000,
    seed: int | None = None,
    vce: str | None = None,
    cluster_var: str | None = None,
) -> StaggeredRIResult:
    """
    Perform randomization inference for a cohort-specific ATT.

    This is a convenience wrapper around `randomization_inference_staggered`
    for testing the average effect for a specific treatment cohort. The
    cohort-specific ATT averages effects across all post-treatment periods
    for units first treated in the target cohort.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format with unit, time, cohort, and outcome columns.
    gvar : str
        Column name for the treatment cohort variable.
    ivar : str
        Column name for the unit identifier.
    tvar : str
        Column name for the time period.
    y : str
        Column name for the outcome variable.
    target_cohort : int
        Treatment cohort (first treatment period) to test.
    observed_att : float
        Observed cohort-specific ATT estimate to test.
    rolling : {'demean', 'detrend'}, default 'demean'
        Transformation method for pre-treatment variation removal.
    ri_method : {'permutation', 'bootstrap'}, default 'permutation'
        Resampling method for null distribution generation.
    rireps : int, default 1000
        Number of resampling replications.
    seed : int, optional
        Random seed for reproducibility.
    vce : str, optional
        Variance-covariance estimator type.
    cluster_var : str, optional
        Column name for clustering standard errors.

    Returns
    -------
    StaggeredRIResult
        Randomization inference results including p-value and diagnostics.

    See Also
    --------
    randomization_inference_staggered : Full-featured inference function.
    ri_overall_effect : Inference for overall weighted effect.
    """
    from .transformations import get_cohorts
    from ..validation import is_never_treated

    unit_gvar = data.groupby(ivar)[gvar].first()
    cohorts = get_cohorts(data, gvar, ivar)

    # Identify never-treated units using the single source of truth function.
    nt_mask = unit_gvar.apply(is_never_treated)
    n_nt = int(nt_mask.sum())

    return randomization_inference_staggered(
        data=data,
        gvar=gvar,
        ivar=ivar,
        tvar=tvar,
        y=y,
        cohorts=cohorts,
        observed_att=observed_att,
        target='cohort',
        target_cohort=target_cohort,
        ri_method=ri_method,
        rireps=rireps,
        seed=seed,
        rolling=rolling,
        n_never_treated=n_nt,
        vce=vce,
        cluster_var=cluster_var,
    )

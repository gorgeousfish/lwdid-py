"""
Randomization inference for staggered difference-in-differences estimation.

This module implements permutation-based inference for staggered DiD settings
with multiple treatment cohorts. Randomization inference provides finite-sample
valid p-values without distributional assumptions by comparing the observed
test statistic to its permutation distribution.

Notes
-----
Randomization inference is useful when the number of treated or control units
is small and asymptotic approximations may be unreliable. The permutation
procedure shuffles cohort assignments while preserving the marginal distribution.

For overall effect inference, never-treated units are required as a consistent
reference group. For cohort-specific inference, the target cohort must remain
present after each permutation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import warnings

import numpy as np
from numpy.random import SeedSequence, default_rng
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


def _single_staggered_ri_iteration(
    child_seed,
    ri_method: str,
    unit_ids: list,
    unit_gvar_values: np.ndarray,
    n_units: int,
    data: pd.DataFrame,
    gvar: str,
    ivar: str,
    tvar: str,
    y: str,
    target: str,
    target_cohort: int | None,
    target_period: int | None,
    rolling_lower: str,
    transform_func,
    controls: list[str] | None,
    vce: str | None,
    cluster_var: str | None,
    T_max: int,
) -> float:
    """
    Execute a single staggered RI iteration.

    Designed as a pure function with no shared mutable state, safe for
    parallel execution via joblib.

    Parameters
    ----------
    child_seed : SeedSequence or Generator
        Deterministic child seed for this iteration's RNG, or a pre-existing
        Generator instance (for legacy_seed mode).
    ri_method : str
        'permutation' or 'bootstrap'.
    unit_ids : list
        Unit identifiers.
    unit_gvar_values : np.ndarray
        Cohort assignment values per unit.
    n_units : int
        Number of units.
    data : pd.DataFrame
        Original panel data (read-only within this function).
    gvar, ivar, tvar, y : str
        Column names.
    target : str
        'overall', 'cohort', or 'cohort_time'.
    target_cohort, target_period : int or None
        Target cohort/period for cohort or cohort_time targets.
    rolling_lower : str
        Transformation type ('demean' or 'detrend').
    transform_func : callable
        Transformation function.
    controls : list of str or None
        Control variable names.
    vce : str or None
        Variance-covariance estimator type.
    cluster_var : str or None
        Cluster variable name.
    T_max : int
        Maximum time period.

    Returns
    -------
    float
        ATT statistic for this iteration, or np.nan on failure.
    """
    # Two seed modes are supported:
    # - SeedSequence: new mode (parallel-safe)
    # - Generator: legacy mode (pre-existing RNG passed in)
    if hasattr(child_seed, 'permutation'):
        # Already a Generator instance (legacy mode)
        rng_local = child_seed
    else:
        rng_local = default_rng(child_seed)

    try:
        if ri_method == 'permutation':
            perm_idx = rng_local.permutation(n_units)
            perm_gvar = unit_gvar_values[perm_idx]
        else:
            boot_idx = rng_local.integers(0, n_units, size=n_units)
            perm_gvar = unit_gvar_values[boot_idx]

        perm_gvar_mapping = dict(zip(unit_ids, perm_gvar))

        # Use assign() to avoid mutating the original data
        data_perm = data.assign(**{gvar: data[ivar].map(perm_gvar_mapping)})

        try:
            data_transformed = transform_func(data_perm, y, ivar, tvar, gvar)
        except (ValueError, KeyError):
            return np.nan

        # Deferred imports to avoid circular dependencies
        from .aggregation import aggregate_to_cohort, aggregate_to_overall
        from .transformations import get_cohorts
        from .estimation import estimate_cohort_time_effects

        if target == 'overall':
            try:
                result = aggregate_to_overall(
                    data_transformed, gvar, ivar, tvar,
                    transform_type=rolling_lower,
                    vce=vce,
                    cluster_var=cluster_var,
                )
                return result.att
            except (ValueError, KeyError):
                return np.nan

        elif target == 'cohort':
            try:
                perm_cohorts = get_cohorts(data_transformed, gvar, ivar)
                if target_cohort not in perm_cohorts:
                    return np.nan

                results = aggregate_to_cohort(
                    data_transformed, gvar, ivar, tvar,
                    cohorts=[target_cohort], T_max=T_max,
                    transform_type=rolling_lower,
                    vce=vce,
                    cluster_var=cluster_var,
                )
                return results[0].att if results else np.nan
            except (ValueError, KeyError):
                return np.nan

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
                return target_result[0].att if target_result else np.nan
            except (ValueError, KeyError):
                return np.nan

    except Exception:
        return np.nan


def randomization_inference_staggered(
    data: pd.DataFrame,
    gvar: str,
    ivar: str,
    tvar: str,
    y: str,
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
    n_jobs: int = 1,
    legacy_seed: bool = False,
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
    n_jobs : int, default 1
        Number of parallel cores for resampling iterations.

        - 1: Serial execution (default, backward compatible).
        - -1: Use all available CPU cores.
        - n > 1: Use n cores.

        Requires joblib package. Falls back to serial execution with a
        UserWarning if joblib is not installed.

        .. versionadded:: 0.2.0

    legacy_seed : bool, default False
        Whether to preserve the original single-RNG sequential seed behavior.
        Only effective when ``n_jobs=1``. When True, uses a single RNG
        instance for all iterations (reproducing pre-optimization results
        exactly). When False (default), uses SeedSequence-derived child
        seeds for each iteration, enabling parallel-safe determinism.

        .. versionadded:: 0.2.0

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
    The permutation procedure shuffles cohort assignments while preserving
    the marginal cohort distribution, generating the null distribution under
    the sharp null hypothesis of no treatment effect.

    The two-sided p-value is computed as:

    .. math::

        p = \\frac{1}{R} \\sum_{r=1}^{R} \\mathbf{1}\\{|\\hat{\\tau}^{(r)}| \\geq |\\hat{\\tau}^{obs}|\\}

    where :math:`R` is the number of valid replications and
    :math:`\\hat{\\tau}^{(r)}` is the ATT from replication :math:`r`.

    A minimum of 50 valid replications (or 10% of rireps, whichever is larger)
    is required for reliable p-value computation.

    Performance
    -----------
    Each resampling iteration is independent and can be parallelized via
    the ``n_jobs`` parameter (requires joblib). Seed determinism is
    maintained through ``numpy.random.SeedSequence`` child seeds, ensuring
    identical results regardless of execution order.

    .. versionchanged:: 0.2.0
       Added ``n_jobs`` for joblib-based parallelization and
       ``legacy_seed`` for backward-compatible seed behavior.
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

    # Cohort assignment is time-invariant; extract once per unit for efficiency.
    unit_gvar = data.groupby(ivar)[gvar].first()
    unit_ids = unit_gvar.index.tolist()
    n_units = len(unit_ids)

    # Permutation distribution requires sufficient units for meaningful inference.
    if n_units < 4:
        raise RandomizationError(f"Too few units for RI: N={n_units}")

    # Deferred imports to avoid circular dependencies between submodules.
    from .transformations import transform_staggered_demean, transform_staggered_detrend

    transform_func = (
        transform_staggered_demean
        if rolling_lower == 'demean'
        else transform_staggered_detrend
    )

    T_max = int(data[tvar].max())

    # =========================================================================
    # Seed Strategy
    # =========================================================================
    # legacy_seed=True + n_jobs=1: use the original single-RNG sequential
    # generation to exactly reproduce pre-optimization results.
    # Otherwise: use SeedSequence-derived deterministic child seeds,
    # enabling parallel execution.
    use_legacy = legacy_seed and n_jobs == 1

    # Common arguments (passed to _single_staggered_ri_iteration)
    common_args = dict(
        ri_method=ri_method,
        unit_ids=unit_ids,
        unit_gvar_values=unit_gvar.values.copy(),
        n_units=n_units,
        data=data,
        gvar=gvar,
        ivar=ivar,
        tvar=tvar,
        y=y,
        target=target,
        target_cohort=target_cohort,
        target_period=target_period,
        rolling_lower=rolling_lower,
        transform_func=transform_func,
        controls=controls,
        vce=vce,
        cluster_var=cluster_var,
        T_max=T_max,
    )

    # =========================================================================
    # Resampling Loop
    # =========================================================================
    if use_legacy:
        # Legacy mode: single RNG sequential generation, fully consistent
        # with pre-optimization behavior
        rng = default_rng(seed)
        perm_stats = np.array([
            _single_staggered_ri_iteration(rng, **common_args)
            for _ in range(rireps)
        ])
    else:
        # New mode: SeedSequence-derived deterministic child seeds
        ss = SeedSequence(seed if seed is not None else None)
        child_seeds = ss.spawn(rireps)

        if n_jobs == 1:
            # Serial execution (default)
            perm_stats = np.array([
                _single_staggered_ri_iteration(child_seeds[rep], **common_args)
                for rep in range(rireps)
            ])
        else:
            # Parallel execution
            try:
                from joblib import Parallel, delayed
                results = Parallel(n_jobs=n_jobs, backend='loky')(
                    delayed(_single_staggered_ri_iteration)(
                        child_seeds[rep], **common_args
                    )
                    for rep in range(rireps)
                )
                perm_stats = np.array(results)
            except ImportError:
                warnings.warn(
                    "joblib is not installed; falling back to serial execution. "
                    "Install joblib to enable parallelization: pip install joblib",
                    UserWarning,
                )
                perm_stats = np.array([
                    _single_staggered_ri_iteration(
                        child_seeds[rep], **common_args
                    )
                    for rep in range(rireps)
                ])

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
    n_jobs: int = 1,
    legacy_seed: bool = False,
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
    n_jobs : int, default 1
        Number of parallel cores. See `randomization_inference_staggered`.

        .. versionadded:: 0.2.0

    legacy_seed : bool, default False
        Preserve original seed behavior. See `randomization_inference_staggered`.

        .. versionadded:: 0.2.0

    Returns
    -------
    StaggeredRIResult
        Randomization inference results including p-value and diagnostics.

    See Also
    --------
    randomization_inference_staggered : Full-featured inference function.
    ri_cohort_effect : Inference for cohort-specific effects.
    """
    from ..validation import is_never_treated

    unit_gvar = data.groupby(ivar)[gvar].first()
    nt_mask = unit_gvar.apply(is_never_treated)
    n_nt = int(nt_mask.sum())

    return randomization_inference_staggered(
        data=data,
        gvar=gvar,
        ivar=ivar,
        tvar=tvar,
        y=y,
        observed_att=observed_att,
        target='overall',
        ri_method=ri_method,
        rireps=rireps,
        seed=seed,
        rolling=rolling,
        n_never_treated=n_nt,
        vce=vce,
        cluster_var=cluster_var,
        n_jobs=n_jobs,
        legacy_seed=legacy_seed,
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
    n_jobs: int = 1,
    legacy_seed: bool = False,
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
    n_jobs : int, default 1
        Number of parallel cores. See `randomization_inference_staggered`.

        .. versionadded:: 0.2.0

    legacy_seed : bool, default False
        Preserve original seed behavior. See `randomization_inference_staggered`.

        .. versionadded:: 0.2.0

    Returns
    -------
    StaggeredRIResult
        Randomization inference results including p-value and diagnostics.

    See Also
    --------
    randomization_inference_staggered : Full-featured inference function.
    ri_overall_effect : Inference for overall weighted effect.
    """
    from ..validation import is_never_treated

    unit_gvar = data.groupby(ivar)[gvar].first()
    nt_mask = unit_gvar.apply(is_never_treated)
    n_nt = int(nt_mask.sum())

    return randomization_inference_staggered(
        data=data,
        gvar=gvar,
        ivar=ivar,
        tvar=tvar,
        y=y,
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
        n_jobs=n_jobs,
        legacy_seed=legacy_seed,
    )

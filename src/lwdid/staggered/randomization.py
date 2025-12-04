"""
Staggered Randomization Inference Module

Implements randomization inference for staggered DiD estimation based on
Lee and Wooldridge (2023, 2025).

Key concepts:
- Permutes treatment cohort assignments (gvar) across units
- Re-transforms data and re-estimates effects for each permutation
- Computes two-sided p-value for H₀: ATT = 0

Reference:
    Lee & Wooldridge (2025) Section 7
    Lee & Wooldridge (2023) Section 6
"""

from dataclasses import dataclass
from typing import List, Literal, Optional
import warnings

import numpy as np
import pandas as pd

from ..exceptions import RandomizationError


@dataclass
class StaggeredRIResult:
    """
    Staggered randomization inference result.
    
    Attributes
    ----------
    p_value : float
        Two-sided p-value for H₀: ATT = 0
    ri_method : str
        Method used: 'permutation' or 'bootstrap'
    ri_reps : int
        Number of requested replications
    ri_valid : int
        Number of valid replications (non-NaN)
    ri_failed : int
        Number of failed replications
    observed_stat : float
        Observed ATT statistic
    permutation_stats : np.ndarray
        Array of permutation/bootstrap ATT statistics (valid only)
    """
    p_value: float
    ri_method: str
    ri_reps: int
    ri_valid: int
    ri_failed: int
    observed_stat: float
    permutation_stats: np.ndarray
    
    def __repr__(self) -> str:
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
    cohorts: List[int],
    observed_att: float,
    target: Literal['overall', 'cohort', 'cohort_time'] = 'overall',
    target_cohort: Optional[int] = None,
    target_period: Optional[int] = None,
    ri_method: Literal['permutation', 'bootstrap'] = 'permutation',
    rireps: int = 1000,
    seed: Optional[int] = None,
    rolling: str = 'demean',
    controls: Optional[List[str]] = None,
    vce: Optional[str] = None,
    cluster_var: Optional[str] = None,
    n_never_treated: int = 0,
) -> StaggeredRIResult:
    """
    Staggered DiD randomization inference.
    
    Tests H₀: ATT = 0 by permuting/bootstrapping treatment cohort assignments
    and re-estimating the target effect under each permutation.
    
    Algorithm:
    1. Validate parameters
    2. Get unit-level gvar assignments
    3. For each replication:
       a. Generate new gvar assignment (permutation or bootstrap)
       b. Re-transform data with permuted gvar
       c. Re-estimate target effect
    4. Compute two-sided p-value: P(|ATT_perm| >= |ATT_obs|)
    
    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format.
    gvar : str
        Cohort variable column name.
    ivar : str
        Unit identifier column name.
    tvar : str
        Time variable column name.
    y : str
        Outcome variable column name.
    cohorts : List[int]
        List of treatment cohorts.
    observed_att : float
        Observed ATT estimate to test.
    target : {'overall', 'cohort', 'cohort_time'}
        Target effect level:
        - 'overall': Overall weighted effect τ_ω
        - 'cohort': Cohort-specific effect τ_g (requires target_cohort)
        - 'cohort_time': (g,r)-specific effect (requires target_cohort and target_period)
    target_cohort : int, optional
        Target cohort for 'cohort' or 'cohort_time' targets.
    target_period : int, optional
        Target period for 'cohort_time' target.
    ri_method : {'permutation', 'bootstrap'}
        Resampling method:
        - 'permutation': Without-replacement permutation (Fisher RI)
        - 'bootstrap': With-replacement sampling
    rireps : int, default 1000
        Number of replications.
    seed : int, optional
        Random seed for reproducibility.
    rolling : {'demean', 'detrend'}
        Transformation method.
    controls : list of str, optional
        Control variables for estimation.
    vce : str, optional
        Variance estimator type.
    cluster_var : str, optional
        Cluster variable for clustered SE.
    n_never_treated : int
        Number of never-treated units (for validation).
        
    Returns
    -------
    StaggeredRIResult
        Randomization inference results.
        
    Raises
    ------
    RandomizationError
        If parameters are invalid or insufficient valid replications.
        
    Notes
    -----
    For overall effects, never-treated units are required to provide a
    consistent baseline reference across permutations.
    
    The permutation preserves the total number of units in each cohort,
    only shuffling which units belong to which cohort.
    """
    # === Parameter validation ===
    if rireps <= 0:
        raise RandomizationError("rireps must be positive")
    
    if target == 'cohort' and target_cohort is None:
        raise RandomizationError("target_cohort required when target='cohort'")
    
    if target == 'cohort_time' and (target_cohort is None or target_period is None):
        raise RandomizationError(
            "target_cohort and target_period required when target='cohort_time'"
        )
    
    if ri_method not in ('permutation', 'bootstrap'):
        raise RandomizationError(f"ri_method must be 'permutation' or 'bootstrap'")
    
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
    
    rng = np.random.default_rng(seed)
    
    # === Get unit-level gvar ===
    unit_gvar = data.groupby(ivar)[gvar].first()
    unit_ids = unit_gvar.index.tolist()
    n_units = len(unit_ids)
    
    if n_units < 4:
        raise RandomizationError(f"Too few units for RI: N={n_units}")
    
    # === Import required functions ===
    from .transformations import transform_staggered_demean, transform_staggered_detrend
    from .estimation import estimate_cohort_time_effects
    from .aggregation import aggregate_to_cohort, aggregate_to_overall, get_cohorts
    
    # Determine transform function
    transform_func = (
        transform_staggered_demean 
        if rolling_lower == 'demean' 
        else transform_staggered_detrend
    )
    
    # Get T_max for aggregation
    T_max = int(data[tvar].max())
    
    # === Permutation loop ===
    perm_stats = np.empty(rireps, dtype=float)
    perm_stats.fill(np.nan)
    
    for rep in range(rireps):
        try:
            # Generate permuted gvar assignment
            if ri_method == 'permutation':
                perm_idx = rng.permutation(n_units)
                perm_gvar = unit_gvar.values[perm_idx]
            else:  # bootstrap
                boot_idx = rng.integers(0, n_units, size=n_units)
                perm_gvar = unit_gvar.values[boot_idx]
            
            # Build permuted gvar mapping
            perm_gvar_mapping = dict(zip(unit_ids, perm_gvar))
            
            # Create permuted data
            data_perm = data.copy()
            data_perm[gvar] = data_perm[ivar].map(perm_gvar_mapping)
            
            # Re-transform with permuted gvar
            try:
                data_transformed = transform_func(
                    data_perm, y, ivar, tvar, gvar
                )
            except (ValueError, KeyError):
                # Transformation failed (e.g., no valid cohorts after permutation)
                perm_stats[rep] = np.nan
                continue
            
            # Re-estimate target effect
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
                    
            else:  # cohort_time
                try:
                    ct_results = estimate_cohort_time_effects(
                        data_transformed, gvar, ivar, tvar,
                        controls=controls,
                        vce=vce,
                        cluster_var=cluster_var,
                        transform_type=rolling_lower,
                    )
                    # Find target (g,r) effect
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
            # Catch-all for unexpected errors
            perm_stats[rep] = np.nan
    
    # === Compute p-value ===
    valid_stats = perm_stats[~np.isnan(perm_stats)]
    n_valid = len(valid_stats)
    n_failed = rireps - n_valid
    
    # Minimum valid replications threshold
    min_valid = max(50, int(0.1 * rireps))
    if n_valid < min_valid:
        raise RandomizationError(
            f"Insufficient valid RI replications: {n_valid}/{rireps} "
            f"(need at least {min_valid}). "
            f"Consider increasing rireps or checking data quality."
        )
    
    # Two-sided p-value: P(|ATT_perm| >= |ATT_obs|)
    p_value = float((np.abs(valid_stats) >= abs(observed_att)).mean())
    
    # Warn if many failures
    if n_failed / rireps > 0.1:
        warnings.warn(
            f"Staggered RI: {n_failed}/{rireps} replications failed ({n_failed/rireps:.1%}). "
            f"P-value computed using {n_valid} valid replications.",
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
    seed: Optional[int] = None,
    vce: Optional[str] = None,
    cluster_var: Optional[str] = None,
) -> StaggeredRIResult:
    """
    Convenience function for randomization inference on overall effect.
    
    Parameters
    ----------
    data : pd.DataFrame
        Panel data with gvar column.
    gvar : str
        Cohort variable column name.
    ivar : str
        Unit identifier column name.
    tvar : str
        Time variable column name.
    y : str
        Outcome variable column name.
    observed_att : float
        Observed overall ATT to test.
    rolling : {'demean', 'detrend'}
        Transformation method.
    ri_method : {'permutation', 'bootstrap'}
        Resampling method.
    rireps : int, default 1000
        Number of replications.
    seed : int, optional
        Random seed.
    vce : str, optional
        Variance estimator type.
    cluster_var : str, optional
        Cluster variable.
        
    Returns
    -------
    StaggeredRIResult
        Randomization inference result.
    """
    from .transformations import get_cohorts
    
    # Get cohorts and count NT units
    unit_gvar = data.groupby(ivar)[gvar].first()
    cohorts = get_cohorts(data, gvar, ivar)
    
    # Count never-treated: NaN, 0, or inf
    nt_mask = (
        unit_gvar.isna() | 
        (unit_gvar == 0) | 
        np.isinf(unit_gvar.astype(float))
    )
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
    seed: Optional[int] = None,
    vce: Optional[str] = None,
    cluster_var: Optional[str] = None,
) -> StaggeredRIResult:
    """
    Convenience function for randomization inference on cohort effect.
    
    Parameters
    ----------
    data : pd.DataFrame
        Panel data with gvar column.
    gvar : str
        Cohort variable column name.
    ivar : str
        Unit identifier column name.
    tvar : str
        Time variable column name.
    y : str
        Outcome variable column name.
    target_cohort : int
        Target cohort to test.
    observed_att : float
        Observed cohort ATT to test.
    rolling : {'demean', 'detrend'}
        Transformation method.
    ri_method : {'permutation', 'bootstrap'}
        Resampling method.
    rireps : int, default 1000
        Number of replications.
    seed : int, optional
        Random seed.
    vce : str, optional
        Variance estimator type.
    cluster_var : str, optional
        Cluster variable.
        
    Returns
    -------
    StaggeredRIResult
        Randomization inference result.
    """
    from .transformations import get_cohorts
    
    # Get cohorts and count NT units
    unit_gvar = data.groupby(ivar)[gvar].first()
    cohorts = get_cohorts(data, gvar, ivar)
    
    # Count never-treated
    nt_mask = (
        unit_gvar.isna() | 
        (unit_gvar == 0) | 
        np.isinf(unit_gvar.astype(float))
    )
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

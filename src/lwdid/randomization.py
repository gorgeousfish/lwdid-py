"""
Randomization Inference Module

Implements randomization inference for small-sample hypothesis testing by resampling
treatment labels. Provides p-values without distributional assumptions.

Authors: Xuanyu Cai, Wenli Xu
"""

from typing import Optional
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .exceptions import RandomizationError
from . import estimation


def randomization_inference(
    firstpost_df: pd.DataFrame,
    y_col: str = 'ydot_postavg',
    d_col: str = 'd_',
    ivar: str = 'ivar',
    rireps: int = 1000,
    seed: Optional[int] = None,
    att_obs: Optional[float] = None,
    ri_method: str = 'bootstrap',
    controls: Optional[list] = None,
) -> dict:
    """
    Randomization inference for H₀: ATT = 0

    Tests sharp null via resampling treatment labels. Computes Monte Carlo
    p-value as proportion of permutations with |ATT_perm| ≥ |ATT_obs|.

    Parameters
    ----------
    firstpost_df : pd.DataFrame
        Cross-section (N units).
    y_col : str, default 'ydot_postavg'
        Transformed outcome.
    d_col : str, default 'd_'
        Treatment indicator.
    ivar : str, default 'ivar'
        Unit identifier.
    rireps : int, default 1000
        Number of replications.
    seed : int, optional
        Random seed.
    att_obs : float, optional
        Observed ATT.
    ri_method : {'bootstrap', 'permutation'}, default 'bootstrap'
        'bootstrap' (with replacement) or 'permutation' (without replacement).
        Permutation is classical Fisher RI with fixed N₁.

    Returns
    -------
    dict
        'p_value', 'ri_method', 'ri_reps', 'ri_valid', 'ri_failed', 'ri_failure_rate'.

    Notes
    -----
    Uses plain OLS (vce=None) for all permutation regressions. Model specification
    matches main regression: includes controls if N₁ > K+1 and N₀ > K+1, with
    fixed centering point X̄₁ from original treatment group.
    """
    if rireps is None or rireps <= 0:
        raise RandomizationError('rireps must be positive')
    for col in (y_col, d_col):
        if col not in firstpost_df.columns:
            raise RandomizationError('firstpost_df missing required columns')

    # Validate ri_method
    if ri_method not in ('bootstrap', 'permutation'):
        raise RandomizationError(
            f"ri_method must be 'bootstrap' or 'permutation', got '{ri_method}'"
        )

    rng = np.random.default_rng(seed)

    N = int(len(firstpost_df))
    if N < 3:
        raise RandomizationError(f'firstpost_df too small for RI: N={N}')

    # Re-estimate ATT if not provided (always use plain OLS for RI)
    if att_obs is None:
        res0 = estimation.estimate_att(
            data=firstpost_df,
            y_transformed=y_col,
            d=d_col,
            ivar=ivar,
            controls=controls,
            vce=None,
            cluster_var=None,
            sample_filter=pd.Series(True, index=firstpost_df.index),
        )
        att_obs = float(res0['att'])

    # Pre-compute controls_spec with fixed X̄₁ from original treatment group
    # RI permutes D only, not X; centering point stays fixed
    controls_spec = None
    if controls is not None and len(controls) > 0:
        N_treated_orig = int((firstpost_df[d_col] == 1).sum())
        N_control_orig = int((firstpost_df[d_col] == 0).sum())

        controls_spec = estimation.prepare_controls(
            data=firstpost_df,
            d=d_col,
            ivar=ivar,
            controls=controls,
            N_treated=N_treated_orig,
            N_control=N_control_orig,
            data_sample=firstpost_df
        )

    d_values = firstpost_df[d_col].astype(int).values
    y_values = firstpost_df[y_col].values

    atts = np.empty(rireps, dtype=float)
    atts.fill(np.nan)

    n1_distribution = [] if ri_method == 'bootstrap' else None
    failure_reasons = [] if ri_method == 'bootstrap' else None

    for rep in range(rireps):
        try:
            if ri_method == 'permutation':
                perm_idx = rng.permutation(N)
                d_perm = d_values[perm_idx]
            else:
                draw_idx = rng.integers(low=0, high=N, size=N)
                d_perm = d_values[draw_idx]

                n1_perm = int(d_perm.sum())
                n1_distribution.append(n1_perm)

                if n1_perm == 0:
                    failure_reasons.append(('N1=0', rep))
                    atts[rep] = np.nan
                    continue
                if n1_perm == N:
                    failure_reasons.append(('N1=N', rep))
                    atts[rep] = np.nan
                    continue

            # Construct design matrix with fixed X̄₁ from original treatment group
            # Model: ydot ~ d_perm + X + d_perm·(X - X̄₁)
            if controls_spec is not None and controls_spec['include']:
                X_centered = controls_spec['X_centered']

                interactions_perm = pd.DataFrame({
                    f'd_{x}_c': d_perm * X_centered[f'{x}_c'].values
                    for x in controls
                }, index=firstpost_df.index)

                X_parts = [
                    pd.Series(d_perm, index=firstpost_df.index, name=d_col),
                    firstpost_df[controls],
                    interactions_perm
                ]
                X_df = pd.concat(X_parts, axis=1)
                X = sm.add_constant(X_df.astype(float).values)
            else:
                X = sm.add_constant(d_perm)

            model = sm.OLS(y_values, X)
            results = model.fit()
            atts[rep] = float(results.params[1])

        except Exception as e:
            if ri_method == 'bootstrap' and failure_reasons is not None:
                failure_reasons.append((type(e).__name__, rep))
            atts[rep] = np.nan

    valid = atts[~np.isnan(atts)]
    n_valid = len(valid)
    n_failed = rireps - n_valid
    failure_rate = n_failed / rireps

    # Theoretical failure rate for bootstrap: P(N1=0) + P(N1=N)
    theoretical_failure_rate = None
    if ri_method == 'bootstrap' and n1_distribution:
        N1_obs = int(d_values.sum())
        p = N1_obs / N
        theoretical_failure_rate = (1 - p) ** N + p ** N

    if ri_method == 'bootstrap':
        min_valid_reps = max(100, int(0.1 * rireps))
    else:
        min_valid_reps = max(10, int(0.1 * rireps))

    if n_valid < min_valid_reps:
        if ri_method == 'bootstrap':
            from collections import Counter
            failure_counts = Counter([r[0] for r in failure_reasons])
            raise RandomizationError(
                f'Insufficient valid RI replications: {n_valid}/{rireps} (need at least {min_valid_reps}).\n'
                f'  Failure rate: {failure_rate:.1%}\n'
                f'  Failure reasons: {dict(failure_counts)}\n'
                f'  N1 distribution: min={min(n1_distribution)}, max={max(n1_distribution)}\n'
                f'  Recommendation: Use ri_method="permutation" for classical Fisher RI, '
                f'or increase rireps to at least {int(min_valid_reps / (1 - failure_rate))}.'
            )
        else:
            raise RandomizationError(
                f'Insufficient valid RI replications: {n_valid}/{rireps} (need at least {min_valid_reps}).'
            )

    if ri_method == 'bootstrap' and failure_rate > 0.05:
        from collections import Counter
        failure_counts = Counter([r[0] for r in failure_reasons])

        if theoretical_failure_rate is not None and failure_rate <= theoretical_failure_rate * 1.5:
            warnings.warn(
                f"Randomization inference (bootstrap): {n_failed}/{rireps} "
                f"replications failed ({failure_rate:.1%}).\n"
                f"  This is expected for bootstrap with N1={int(d_values.sum())}/{N} "
                f"(theoretical failure rate ≈ {theoretical_failure_rate:.1%}).\n"
                f"  N1 distribution: min={min(n1_distribution)}, "
                f"max={max(n1_distribution)}, mean={np.mean(n1_distribution):.1f}\n"
                f"  Failure reasons: {dict(failure_counts)}\n"
                f"  The p-value is computed using {n_valid} valid replications.\n"
                f"  Consider using ri_method='permutation' for classical Fisher RI "
                f"to avoid failures.",
                UserWarning,
                stacklevel=2
            )
        else:
            warnings.warn(
                f"Randomization inference (bootstrap): {n_failed}/{rireps} "
                f"replications failed ({failure_rate:.1%}).\n"
                f"  This is higher than expected "
                f"(theoretical ≈ {theoretical_failure_rate:.1%} for N1={int(d_values.sum())}/{N}).\n"
                f"  N1 distribution: min={min(n1_distribution)}, "
                f"max={max(n1_distribution)}, mean={np.mean(n1_distribution):.1f}\n"
                f"  Failure reasons: {dict(failure_counts)}\n"
                f"  The p-value is computed using {n_valid} valid replications.\n"
                f"  Recommendation: Use ri_method='permutation' or investigate data issues.",
                UserWarning,
                stacklevel=2
            )

    pval = float((np.abs(valid) >= abs(att_obs)).mean())

    return {
        'p_value': pval,
        'ri_method': ri_method,
        'ri_reps': rireps,
        'ri_valid': n_valid,
        'ri_failed': n_failed,
        'ri_failure_rate': failure_rate,
    }

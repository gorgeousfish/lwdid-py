"""
Randomization inference for small-sample hypothesis testing.

This module implements randomization inference (RI) for testing the sharp null
hypothesis of no treatment effect in difference-in-differences settings. The
approach provides valid p-values without relying on asymptotic distributional
assumptions, making it particularly suitable for small sample sizes where
t-based inference may be unreliable.

The implementation supports two resampling methods: permutation (classical
Fisher randomization inference with fixed treatment group size) and bootstrap
(resampling with replacement). Both methods compute a Monte Carlo p-value as
the proportion of resampled test statistics at least as extreme as the observed
statistic.

Notes
-----
Randomization inference tests the sharp null hypothesis that all unit-level
treatment effects are exactly zero. Under this null, treatment assignment is
uninformative about potential outcomes, and permuting treatment labels generates
the null distribution of the test statistic.

The permutation method preserves the original number of treated units in each
replication, while the bootstrap method may produce varying treatment group
sizes. Bootstrap can yield degenerate draws (all treated or all control) which
are excluded from p-value computation.

When control variables are specified, the model maintains a fixed centering
point (mean of controls for the original treated group) across all replications.
This ensures that the randomization distribution reflects variation in treatment
assignment rather than in covariate adjustment.
"""

from collections import Counter
from collections.abc import Sequence
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import estimation
from .exceptions import RandomizationError


def randomization_inference(
    firstpost_df: pd.DataFrame,
    y_col: str = 'ydot_postavg',
    d_col: str = 'd_',
    ivar: str = 'ivar',
    rireps: int = 1000,
    seed: int | None = None,
    att_obs: float | None = None,
    ri_method: str = 'bootstrap',
    controls: Sequence[str] | None = None,
) -> dict[str, float | str | int]:
    """
    Perform randomization inference for the null hypothesis of zero treatment effect.

    Tests the sharp null hypothesis that all unit-level treatment effects equal
    zero by resampling treatment labels and computing a Monte Carlo p-value.
    Supports both classical Fisher permutation inference and bootstrap resampling.

    Parameters
    ----------
    firstpost_df : pd.DataFrame
        Cross-sectional data for the first post-treatment period, containing one
        observation per unit. Must include the transformed outcome, treatment
        indicator, and unit identifier columns.
    y_col : str, default 'ydot_postavg'
        Name of the column containing the transformed outcome variable (after
        unit-specific demeaning or detrending).
    d_col : str, default 'd_'
        Name of the column containing the binary treatment indicator (0 or 1).
    ivar : str, default 'ivar'
        Name of the column containing the unit identifier.
    rireps : int, default 1000
        Number of randomization replications for computing the p-value. Higher
        values provide more precise p-values but increase computation time.
    seed : int | None, optional
        Random seed for reproducibility of the randomization distribution.
        If None, results will vary across runs.
    att_obs : float | None, optional
        Observed ATT estimate from the original regression. If None, the ATT
        is re-estimated from the data using OLS.
    ri_method : {'bootstrap', 'permutation'}, default 'bootstrap'
        Resampling method for generating the null distribution:

        - 'permutation': Classical Fisher randomization inference. Permutes
          treatment labels without replacement, preserving the original number
          of treated and control units in each replication.
        - 'bootstrap': Resamples treatment labels with replacement. May produce
          degenerate draws (all treated or all control) which are excluded.

    controls : Sequence[str] | None, optional
        List of control variable names to include in the regression model.
        Controls are included only if sample size conditions are satisfied
        (N_treated > K+1 and N_control > K+1 where K is the number of controls).
        The centering point for interactions is fixed at the mean of controls
        for the original treatment group.

    Returns
    -------
    dict
        Dictionary containing randomization inference results:

        - 'p_value' : float
            Two-sided p-value computed as the proportion of replications with
            abs(ATT_perm) >= abs(ATT_obs).
        - 'ri_method' : str
            The resampling method used ('bootstrap' or 'permutation').
        - 'ri_reps' : int
            Total number of replications requested.
        - 'ri_valid' : int
            Number of valid (non-degenerate) replications used for p-value.
        - 'ri_failed' : int
            Number of failed or degenerate replications.
        - 'ri_failure_rate' : float
            Proportion of replications that failed (ri_failed / ri_reps).

    Raises
    ------
    RandomizationError
        Raised in the following cases:

        - ``rireps`` is not positive.
        - Required columns (``y_col``, ``d_col``) are missing from the data.
        - ``ri_method`` is not 'bootstrap' or 'permutation'.
        - Sample size is too small (N < 3).
        - Insufficient valid replications for reliable inference (fewer than
          10% of requested replications or minimum of 100 for bootstrap,
          10 for permutation).

    Warns
    -----
    UserWarning
        When bootstrap failure rate exceeds 5%, indicating potential issues
        with extreme treatment proportions or data quality.

    See Also
    --------
    estimate_att : Estimates ATT via cross-sectional OLS regression.
    lwdid : Main estimation function that can invoke randomization inference.

    Notes
    -----
    The randomization distribution is computed using plain OLS (homoskedastic
    standard errors) for computational efficiency. The model specification
    matches the main regression, including control variables if sample size
    conditions are met.

    For the permutation method, every replication is valid since treatment
    group sizes are preserved. For the bootstrap method, replications that
    produce degenerate treatment assignment (all N_1=0 or N_1=N) are excluded
    from the p-value computation, and a warning is issued if the failure rate
    is substantial.

    The theoretical failure rate for bootstrap with treatment proportion p and
    sample size N is approximately (1-p)^N + p^N, which can be substantial when
    either treatment or control groups are small.
    """
    # -------------------------------------------------------------------------
    # Input Validation
    # -------------------------------------------------------------------------
    if rireps is None or rireps <= 0:
        raise RandomizationError('rireps must be positive')

    for col in (y_col, d_col):
        if col not in firstpost_df.columns:
            raise RandomizationError('firstpost_df missing required columns')

    if ri_method not in ('bootstrap', 'permutation'):
        raise RandomizationError(
            f"ri_method must be 'bootstrap' or 'permutation', got '{ri_method}'"
        )

    rng = np.random.default_rng(seed)
    N = int(len(firstpost_df))

    if N < 3:
        raise RandomizationError(f'firstpost_df too small for RI: N={N}')

    # -------------------------------------------------------------------------
    # Observed ATT Estimation
    # -------------------------------------------------------------------------
    # Use plain OLS (vce=None) for consistency with permutation regressions
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

    # -------------------------------------------------------------------------
    # Control Variable Preparation
    # -------------------------------------------------------------------------
    # Pre-compute centering using original treatment group means.
    # The centering point remains fixed across all replications so that
    # randomization reflects variation in treatment assignment, not in
    # covariate adjustment.
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

    # Extract arrays for efficient computation in the loop
    d_values = firstpost_df[d_col].astype(int).values
    y_values = firstpost_df[y_col].values

    # Pre-allocate array for ATT estimates; NaN indicates failed replication
    atts = np.empty(rireps, dtype=float)
    atts.fill(np.nan)

    # Track bootstrap diagnostics for failure analysis
    n1_distribution = [] if ri_method == 'bootstrap' else None
    failure_reasons = [] if ri_method == 'bootstrap' else None

    # -------------------------------------------------------------------------
    # Randomization Loop
    # -------------------------------------------------------------------------
    for rep in range(rireps):
        try:
            if ri_method == 'permutation':
                # Permutation preserves N_1, ensuring valid regression
                perm_idx = rng.permutation(N)
                d_perm = d_values[perm_idx]
            else:
                # Bootstrap may produce degenerate draws (N_1=0 or N_1=N)
                draw_idx = rng.integers(low=0, high=N, size=N)
                d_perm = d_values[draw_idx]

                n1_perm = int(d_perm.sum())
                n1_distribution.append(n1_perm)

                # Skip degenerate draws where regression would be undefined
                if n1_perm == 0:
                    failure_reasons.append(('N1=0', rep))
                    atts[rep] = np.nan
                    continue
                if n1_perm == N:
                    failure_reasons.append(('N1=N', rep))
                    atts[rep] = np.nan
                    continue

            # Match the main regression specification to ensure comparable test statistics
            if controls_spec is not None and controls_spec['include']:
                X_centered = controls_spec['X_centered']

                # Interaction terms use permuted treatment with fixed centering
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

    # -------------------------------------------------------------------------
    # Results Aggregation
    # -------------------------------------------------------------------------
    valid = atts[~np.isnan(atts)]
    n_valid = len(valid)
    n_failed = rireps - n_valid
    failure_rate = n_failed / rireps

    # Compute theoretical failure rate to distinguish expected degeneracy from data issues.
    # Helps users understand whether high failure rates reflect small treatment proportions
    # (expected) versus unexpected problems requiring investigation.
    theoretical_failure_rate = None
    if ri_method == 'bootstrap' and n1_distribution:
        N1_obs = int(d_values.sum())
        p = N1_obs / N
        theoretical_failure_rate = (1 - p) ** N + p ** N

    # Require minimum valid replications for reliable p-value estimation
    if ri_method == 'bootstrap':
        min_valid_reps = max(100, int(0.1 * rireps))
    else:
        min_valid_reps = max(10, int(0.1 * rireps))

    if n_valid < min_valid_reps:
        if ri_method == 'bootstrap':
            failure_counts = Counter([r[0] for r in failure_reasons])
            # Protect against division by zero when failure_rate = 1.0
            if failure_rate >= 1.0:
                recommended_reps = "N/A (100% failure rate)"
            else:
                recommended_reps = str(int(min_valid_reps / (1 - failure_rate)))
            raise RandomizationError(
                f'Insufficient valid RI replications: {n_valid}/{rireps} (need at least {min_valid_reps}).\n'
                f'  Failure rate: {failure_rate:.1%}\n'
                f'  Failure reasons: {dict(failure_counts)}\n'
                f'  N1 distribution: min={min(n1_distribution)}, max={max(n1_distribution)}\n'
                f'  Recommendation: Use ri_method="permutation" for classical Fisher RI, '
                f'or increase rireps to at least {recommended_reps}.'
            )
        else:
            raise RandomizationError(
                f'Insufficient valid RI replications: {n_valid}/{rireps} (need at least {min_valid_reps}).'
            )

    # Warn if bootstrap failure rate is substantial
    if ri_method == 'bootstrap' and failure_rate > 0.05:
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

    # Two-sided p-value: proportion of abs(ATT_perm) >= abs(ATT_obs)
    pval = float((np.abs(valid) >= abs(att_obs)).mean())

    return {
        'p_value': pval,
        'ri_method': ri_method,
        'ri_reps': rireps,
        'ri_valid': n_valid,
        'ri_failed': n_failed,
        'ri_failure_rate': failure_rate,
    }

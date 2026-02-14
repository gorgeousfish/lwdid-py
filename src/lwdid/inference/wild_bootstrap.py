"""
Wild cluster bootstrap for inference with few clusters.

This module implements the wild cluster bootstrap method for more reliable
inference when the number of clusters is small. The method is particularly
useful in difference-in-differences settings where standard cluster-robust
standard errors may perform poorly.

The wild cluster bootstrap is recommended when:

- Number of clusters G < 30
- Cluster sizes are unbalanced
- Few treated clusters

Key features:

- Full enumeration mode for exact p-values when G <= 12
- Optional integration with wildboottest package for optimized computation
- Multiple weight distributions: Rademacher, Mammen, Webb

Notes
-----
The Rademacher weights are the simplest and most commonly used. Mammen weights
match the first three moments of the error distribution. Webb weights provide
better performance with very few clusters (G < 10).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import warnings
from itertools import product

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Check if wildboottest package is available.
try:
    from wildboottest.wildboottest import wildboottest as _wildboottest
    HAS_WILDBOOTTEST = True
except ImportError:
    HAS_WILDBOOTTEST = False
    _wildboottest = None


@dataclass
class WildClusterBootstrapResult:
    """
    Result of wild cluster bootstrap inference.
    
    This class contains the results of wild cluster bootstrap,
    which provides more reliable inference when the number of
    clusters is small.
    
    Attributes
    ----------
    att : float
        Point estimate of ATT.
    se_bootstrap : float
        Bootstrap standard error.
    ci_lower : float
        Lower bound of bootstrap confidence interval.
    ci_upper : float
        Upper bound of bootstrap confidence interval.
    pvalue : float
        Bootstrap p-value (two-sided).
    n_clusters : int
        Number of clusters.
    n_bootstrap : int
        Number of bootstrap replications.
    weight_type : str
        Type of bootstrap weights used.
    t_stat_original : float
        Original t-statistic.
    t_stats_bootstrap : np.ndarray
        Bootstrap t-statistics.
    rejection_rate : float
        Proportion of bootstrap t-stats exceeding original.
    ci_method : str
        Method used for CI construction ('percentile_t' or 'test_inversion').
    """
    att: float
    se_bootstrap: float
    ci_lower: float
    ci_upper: float
    pvalue: float
    n_clusters: int
    n_bootstrap: int
    weight_type: str
    t_stat_original: float
    t_stats_bootstrap: np.ndarray
    rejection_rate: float
    ci_method: str = 'percentile_t'
    
    def summary(self) -> str:
        """
        Generate human-readable summary of bootstrap results.
        
        Returns
        -------
        str
            Formatted summary string.
        """
        sig = "***" if self.pvalue < 0.01 else "**" if self.pvalue < 0.05 else "*" if self.pvalue < 0.1 else ""
        return (
            f"Wild Cluster Bootstrap Results\n"
            f"{'='*50}\n"
            f"ATT: {self.att:.4f} {sig}\n"
            f"Bootstrap SE: {self.se_bootstrap:.4f}\n"
            f"95% CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}]\n"
            f"P-value: {self.pvalue:.4f}\n"
            f"N clusters: {self.n_clusters}\n"
            f"N bootstrap: {self.n_bootstrap}\n"
            f"Weight type: {self.weight_type}\n"
            f"{'='*50}"
        )


def _generate_bootstrap_weights(n_clusters: int, weight_type: str) -> np.ndarray:
    """
    Generate bootstrap weights for each cluster.
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    weight_type : str
        Type of bootstrap weights:
        - 'rademacher': P(w=1) = P(w=-1) = 0.5
        - 'mammen': Two-point distribution matching skewness
        - 'webb': Six-point distribution (Webb 2014)
    
    Returns
    -------
    np.ndarray
        Array of bootstrap weights, one per cluster.
    
    Notes
    -----
    Rademacher weights are the simplest and most commonly used.
    Mammen weights match the first three moments of the error distribution.
    Webb weights provide better performance with very few clusters.
    """
    if weight_type == 'rademacher':
        # P(w=1) = P(w=-1) = 0.5
        # E[w] = 0, E[w²] = 1
        return np.random.choice([-1, 1], size=n_clusters)
    
    elif weight_type == 'mammen':
        # Two-point distribution matching first three moments:
        # P(w = -(√5-1)/2) = (√5+1)/(2√5)
        # P(w = (√5+1)/2) = (√5-1)/(2√5)
        # E[w] = 0, E[w²] = 1, E[w³] = 1
        sqrt5 = np.sqrt(5)
        p = (sqrt5 + 1) / (2 * sqrt5)
        w1 = -(sqrt5 - 1) / 2  # ≈ -0.618
        w2 = (sqrt5 + 1) / 2   # ≈ 1.618
        return np.where(np.random.random(n_clusters) < p, w1, w2)
    
    elif weight_type == 'webb':
        # Six-point distribution (Webb 2014)
        # Values: ±√(1/2), ±√(2/2), ±√(3/2)
        # Each with probability 1/6
        # E[w] = 0, E[w²] = 1
        values = np.array([
            -np.sqrt(3/2), -np.sqrt(2/2), -np.sqrt(1/2),
            np.sqrt(1/2), np.sqrt(2/2), np.sqrt(3/2)
        ])
        return np.random.choice(values, size=n_clusters)
    
    else:
        raise ValueError(
            f"Unknown weight_type: {weight_type}. "
            f"Must be one of: 'rademacher', 'mammen', 'webb'"
        )


def _generate_all_rademacher_weights(n_clusters: int) -> np.ndarray:
    """
    Generate all possible Rademacher weight combinations for full enumeration.
    
    For G clusters, there are 2^G possible combinations of {-1, +1} weights.
    Full enumeration eliminates randomness and produces deterministic results
    with exact p-values.
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    
    Returns
    -------
    np.ndarray
        Array of shape (2^G, G) containing all weight combinations.
    
    Notes
    -----
    This is only practical for G <= 12 (4096 combinations).
    For G > 12, use random sampling instead.
    """
    # Generate all {-1, +1}^G combinations.
    return np.array(list(product([-1, 1], repeat=n_clusters)))

def _precompute_bootstrap_matrices(
    data: pd.DataFrame,
    y_var: str,
    d_var: str,
    cluster_var: str,
    controls: list[str] | None = None,
) -> dict:
    """
    Precompute all matrices and indices required by the bootstrap loop.

    Performs all pandas-to-numpy conversions and matrix decompositions
    in a single pass so that subsequent bootstrap iterations operate
    entirely at the numpy level.

    Parameters
    ----------
    data : pd.DataFrame
        Regression data.
    y_var : str
        Name of the outcome variable.
    d_var : str
        Name of the treatment indicator.
    cluster_var : str
        Name of the cluster variable.
    controls : list[str], optional
        Names of control variables.

    Returns
    -------
    dict
        Dictionary with the following keys:
        y : ndarray, shape (N,) — original outcome vector
        X : ndarray, shape (N, k) — design matrix (with intercept)
        P : ndarray, shape (k, N) — projection matrix (X'X)⁻¹X'
        XtX_inv : ndarray, shape (k, k) — (X'X)⁻¹
        cluster_ids : ndarray, shape (N,) — cluster identifiers
        unique_clusters : ndarray, shape (G,) — unique cluster values
        obs_cluster_idx : ndarray, shape (N,) — cluster index for each obs
        cluster_masks : list[ndarray] — boolean mask for each cluster
        cluster_X : list[ndarray] — X sub-matrix for each cluster
        G : int — number of clusters
        N : int — number of observations
        k : int — number of parameters
        correction : float — finite-sample correction factor
    """
    # Extract numpy arrays
    y = data[y_var].values.astype(np.float64)

    X_vars = [d_var]
    if controls:
        X_vars.extend(controls)
    X_raw = data[X_vars].values.astype(np.float64)
    # Prepend intercept column (column order matches sm.add_constant: [intercept, d, controls...])
    X = np.column_stack([np.ones(len(data), dtype=np.float64), X_raw])

    N, k = X.shape

    # Compute (X'X) and its inverse
    XtX = X.T @ X

    # Condition number check (Design Doc §7.3.1)
    cond = np.linalg.cond(XtX)
    if cond > 1e10:
        warnings.warn(
            f"Design matrix condition number is large ({cond:.2e}). "
            f"Normal equations may lose ~{np.log10(cond):.0f} digits of "
            f"precision compared to SVD. Numerical accuracy may be reduced.",
            UserWarning,
            stacklevel=2,
        )

    XtX_inv = np.linalg.inv(XtX)
    # Projection matrix P = (X'X)⁻¹X', shape (k, N)
    P = XtX_inv @ X.T

    # Cluster structure
    cluster_ids = data[cluster_var].values
    unique_clusters = np.unique(cluster_ids)
    G = len(unique_clusters)

    cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}
    obs_cluster_idx = np.array([cluster_to_idx[c] for c in cluster_ids])

    # Precompute boolean mask and X sub-matrix for each cluster
    cluster_masks = []
    cluster_X = []
    for g in range(G):
        mask = obs_cluster_idx == g
        cluster_masks.append(mask)
        cluster_X.append(X[mask])

    # Finite-sample correction factor (Cameron & Miller, 2015)
    # c = (G/(G-1)) * ((N-1)/(N-K))
    correction = (G / (G - 1)) * ((N - 1) / (N - k))

    return {
        'y': y, 'X': X, 'P': P, 'XtX_inv': XtX_inv,
        'cluster_ids': cluster_ids, 'unique_clusters': unique_clusters,
        'obs_cluster_idx': obs_cluster_idx,
        'cluster_masks': cluster_masks, 'cluster_X': cluster_X,
        'G': G, 'N': N, 'k': k, 'correction': correction,
    }


def _fast_ols_cluster_se(
    y_star: np.ndarray,
    precomp: dict,
) -> tuple[float, float]:
    """
    Compute OLS coefficients and cluster-robust standard errors using
    precomputed matrices for fast bootstrap replication.

    Mathematical formulation:
        β = P @ y*                          (OLS coefficients)
        û = y* - X @ β                      (residuals)
        s_g = X_g' @ û_g                    (cluster score vector)
        B_clu = Σ_g s_g s_g'               (meat matrix)
        V_clu = c · (X'X)⁻¹ B_clu (X'X)⁻¹ (bias-corrected variance)
        SE(τ̂) = √(V_clu[1,1])              (treatment effect standard error)

    Parameters
    ----------
    y_star : ndarray, shape (N,)
        Outcome vector for the bootstrap sample.
    precomp : dict
        Return value from ``_precompute_bootstrap_matrices()``.

    Returns
    -------
    att : float
        Estimated treatment effect (coefficient on column 1 of X;
        column 0 is the intercept).
    se : float
        Cluster-robust standard error.
    """
    P = precomp['P']
    X = precomp['X']
    XtX_inv = precomp['XtX_inv']
    k = precomp['k']
    correction = precomp['correction']

    # OLS coefficients: β = P @ y*
    beta = P @ y_star  # shape (k,)
    att = beta[1]  # Treatment effect coefficient (column 0 is the intercept)

    # Residuals: û = y* - X @ β
    residuals = y_star - X @ beta  # shape (N,)

    # Cluster-robust SE (sandwich estimator)
    # meat = Σ_g (X_g' û_g)(X_g' û_g)'
    meat = np.zeros((k, k))
    for g_idx in range(precomp['G']):
        mask = precomp['cluster_masks'][g_idx]
        X_g = precomp['cluster_X'][g_idx]
        e_g = residuals[mask]
        score_g = X_g.T @ e_g  # shape (k,)
        meat += np.outer(score_g, score_g)

    # Bias-corrected variance: V = c · (X'X)⁻¹ · meat · (X'X)⁻¹
    var_beta = correction * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(var_beta[1, 1])

    return att, se









def _generate_all_bootstrap_weights(
    n_bootstrap: int,
    n_clusters: int,
    weight_type: str,
) -> np.ndarray:
    """
    Generate all bootstrap weights at once as a (B, G) matrix.

    This batch generation is numerically identical to calling
    _generate_bootstrap_weights() B times sequentially, because
    the underlying np.random calls consume the same RNG state
    in the same order.

    Parameters
    ----------
    n_bootstrap : int
        Number of bootstrap replications (B).
    n_clusters : int
        Number of clusters (G).
    weight_type : str
        Weight distribution: 'rademacher', 'mammen', or 'webb'.

    Returns
    -------
    ndarray, shape (B, G)
        All bootstrap weights.
    """
    if weight_type == 'rademacher':
        return np.random.choice([-1, 1], size=(n_bootstrap, n_clusters))
    elif weight_type == 'mammen':
        sqrt5 = np.sqrt(5)
        p = (sqrt5 + 1) / (2 * sqrt5)
        w1 = -(sqrt5 - 1) / 2
        w2 = (sqrt5 + 1) / 2
        return np.where(
            np.random.random((n_bootstrap, n_clusters)) < p, w1, w2
        )
    elif weight_type == 'webb':
        values = np.array([
            -np.sqrt(3 / 2), -np.sqrt(2 / 2), -np.sqrt(1 / 2),
            np.sqrt(1 / 2), np.sqrt(2 / 2), np.sqrt(3 / 2),
        ])
        return np.random.choice(values, size=(n_bootstrap, n_clusters))
    else:
        raise ValueError(
            f"Unknown weight_type: {weight_type}. "
            f"Must be one of: 'rademacher', 'mammen', 'webb'"
        )


# Maximum memory budget for batch bootstrap matrices (bytes).
# Three (B, N) float64 matrices: obs_weights_all, Y_star, Residuals.
_MAX_BATCH_MEMORY_BYTES: int = 500 * 1024 * 1024  # 500 MB


def _batch_bootstrap(
    all_weights: np.ndarray,
    precomp: dict,
    fitted: np.ndarray,
    residuals_base: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Batch-compute bootstrap ATT estimates and t-statistics.

    Replaces the Python for-loop with matrix operations for the
    coefficient and residual computation steps.  Cluster-robust SE
    still requires a per-sample loop because each bootstrap sample
    has a different meat matrix.

    Parameters
    ----------
    all_weights : ndarray, shape (B, G)
        Bootstrap cluster-level weights.
    precomp : dict
        Output of _precompute_bootstrap_matrices().
    fitted : ndarray, shape (N,)
        Fitted values from the restricted or unrestricted model.
    residuals_base : ndarray, shape (N,)
        Residuals from the restricted or unrestricted model.

    Returns
    -------
    att_bootstrap : ndarray, shape (B,)
        Bootstrap ATT estimates.
    t_stats_bootstrap : ndarray, shape (B,)
        Bootstrap t-statistics (NaN where SE is invalid).
    """
    B = all_weights.shape[0]
    N = precomp['N']
    obs_cluster_idx = precomp['obs_cluster_idx']

    # Memory check: estimate peak usage for 3 (B, N) float64 matrices
    element_bytes = 8  # float64
    required_bytes = 3 * B * N * element_bytes

    if required_bytes > _MAX_BATCH_MEMORY_BYTES:
        # Process in chunks to stay within memory budget
        chunk_size = max(1, _MAX_BATCH_MEMORY_BYTES // (3 * N * element_bytes))
        return _batch_bootstrap_chunked(
            all_weights, precomp, fitted, residuals_base, chunk_size
        )

    # --- Full batch path (fits in memory) ---
    return _batch_bootstrap_core(
        all_weights, precomp, fitted, residuals_base
    )


def _batch_bootstrap_core(
    all_weights: np.ndarray,
    precomp: dict,
    fitted: np.ndarray,
    residuals_base: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Core batch bootstrap computation without chunking.

    Parameters
    ----------
    all_weights : ndarray, shape (B, G)
        Bootstrap cluster-level weights.
    precomp : dict
        Precomputed matrices.
    fitted : ndarray, shape (N,)
        Fitted values.
    residuals_base : ndarray, shape (N,)
        Base residuals.

    Returns
    -------
    att_bootstrap : ndarray, shape (B,)
    t_stats_bootstrap : ndarray, shape (B,)
    """
    B = all_weights.shape[0]
    obs_cluster_idx = precomp['obs_cluster_idx']
    P = precomp['P']          # (k, N)
    X = precomp['X']          # (N, k)

    # Map cluster weights to observation level: (B, N)
    obs_weights_all = all_weights[:, obs_cluster_idx]

    # Batch construct Y*: (B, N)
    # y*_i = fitted_i + w_{g(i)} * residuals_i
    Y_star = fitted[np.newaxis, :] + obs_weights_all * residuals_base[np.newaxis, :]

    # Batch OLS coefficients: Beta = P @ Y*.T  -> (k, B)
    Beta = P @ Y_star.T

    # ATT is the coefficient on the treatment variable (index 1)
    att_bootstrap = Beta[1, :].copy()  # (B,)

    # Batch residuals: (B, N)
    # Residuals[b, :] = Y_star[b, :] - X @ Beta[:, b]
    Residuals = Y_star - (X @ Beta).T

    # Free Y_star and obs_weights_all to reduce peak memory
    del Y_star, obs_weights_all

    # Vectorized cluster-robust SE: replace B × G nested loop with
    # G iterations + einsum batch operations
    var_11 = _batch_cluster_se_variance_11(Residuals, precomp)
    se_all = np.sqrt(np.maximum(var_11, 0))
    valid = (se_all > 0) & ~np.isnan(se_all)

    t_stats_bootstrap = np.full(B, np.nan)
    t_stats_bootstrap[valid] = att_bootstrap[valid] / se_all[valid]
    # Mark ATT as NaN when SE is invalid (consistent with loop mode)
    att_bootstrap[~valid] = np.nan

    return att_bootstrap, t_stats_bootstrap

def _batch_cluster_se_variance_11(
    Residuals: np.ndarray,
    precomp: dict,
) -> np.ndarray:
    """
    Vectorized computation of cluster-robust variance V[1,1] for all
    bootstrap samples simultaneously.

    Replaces the B × G nested loop with G iterations plus einsum batch
    operations for substantial computational savings.

    Algorithm:
        For each cluster g:
            Scores_g = X_g.T @ Residuals[:, mask_g].T   # shape (k, B)
        Meat = sum_g einsum('ib,jb->ijb', Scores_g, Scores_g)  # shape (k, k, B)
        a = XtX_inv[1, :]  # row corresponding to the treatment variable
        var_11 = correction * einsum('i,ijb,j->b', a, Meat, a)  # shape (B,)

    Parameters
    ----------
    Residuals : ndarray, shape (B, N)
        Residuals for each bootstrap sample.
    precomp : dict
        Precomputed matrices containing cluster_masks, cluster_X,
        XtX_inv, correction, k, and G.

    Returns
    -------
    var_11 : ndarray, shape (B,)
        V[1,1] (variance of the treatment effect coefficient) for each
        bootstrap sample. Invalid values (negative or NaN) are preserved
        for the caller to handle.
    """
    B = Residuals.shape[0]
    k = precomp['k']
    G = precomp['G']
    XtX_inv = precomp['XtX_inv']
    correction = precomp['correction']
    a = XtX_inv[1, :]  # (k,) — row of XtX_inv for the treatment variable

    # Accumulate meat matrix across all clusters: shape (k, k, B)
    Meat = np.zeros((k, k, B))
    for g_idx in range(G):
        mask = precomp['cluster_masks'][g_idx]
        X_g = precomp['cluster_X'][g_idx]  # (n_g, k)
        # Batch score vectors: (k, B) = (k, n_g) @ (n_g, B)
        Scores_g = X_g.T @ Residuals[:, mask].T
        # Accumulate outer products: (k, k, B) += einsum('ib,jb->ijb', ...)
        Meat += np.einsum('ib,jb->ijb', Scores_g, Scores_g)

    # Batch compute V[1,1] = correction * a @ Meat[:,:,b] @ a
    var_11 = correction * np.einsum('i,ijb,j->b', a, Meat, a)

    return var_11




def _batch_bootstrap_chunked(
    all_weights: np.ndarray,
    precomp: dict,
    fitted: np.ndarray,
    residuals_base: np.ndarray,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Chunked batch bootstrap for large datasets that exceed memory budget.

    Splits the B bootstrap replications into chunks of size `chunk_size`,
    processes each chunk via _batch_bootstrap_core(), and concatenates.

    Parameters
    ----------
    all_weights : ndarray, shape (B, G)
        All bootstrap weights.
    precomp : dict
        Precomputed matrices.
    fitted : ndarray, shape (N,)
        Fitted values.
    residuals_base : ndarray, shape (N,)
        Base residuals.
    chunk_size : int
        Maximum number of bootstrap samples per chunk.

    Returns
    -------
    att_bootstrap : ndarray, shape (B,)
    t_stats_bootstrap : ndarray, shape (B,)
    """
    B = all_weights.shape[0]
    att_parts = []
    t_parts = []

    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        chunk_weights = all_weights[start:end]
        att_chunk, t_chunk = _batch_bootstrap_core(
            chunk_weights, precomp, fitted, residuals_base
        )
        att_parts.append(att_chunk)
        t_parts.append(t_chunk)

    return np.concatenate(att_parts), np.concatenate(t_parts)


def _estimate_ols_for_bootstrap(
    data: pd.DataFrame,
    y_var: str,
    d_var: str,
    cluster_var: str,
    controls: list[str] | None = None,
) -> dict:
    """
    Estimate OLS regression for bootstrap procedure.
    
    Parameters
    ----------
    data : pd.DataFrame
        Regression data.
    y_var : str
        Outcome variable name.
    d_var : str
        Treatment indicator variable name.
    cluster_var : str
        Clustering variable name.
    controls : list[str], optional
        Control variable names.
    
    Returns
    -------
    dict
        Dictionary containing:
        - att: Treatment effect estimate
        - se: Cluster-robust standard error
        - residuals: OLS residuals
        - fitted: Fitted values
        - X: Design matrix
        - beta: All coefficient estimates
    """
    import statsmodels.api as sm
    
    # Construct design matrix.
    y = data[y_var].values
    X_vars = [d_var]
    if controls:
        X_vars.extend(controls)
    
    X = data[X_vars].values
    X = sm.add_constant(X)
    
    # OLS estimation.
    model = sm.OLS(y, X)
    
    # Cluster-robust standard errors.
    cluster_ids = data[cluster_var].values
    results = model.fit(cov_type='cluster', cov_kwds={'groups': cluster_ids})
    
    # Extract results.
    att = results.params[1]  # Treatment effect coefficient.
    se = results.bse[1]
    residuals = results.resid
    fitted = results.fittedvalues
    
    return {
        'att': att,
        'se': se,
        'residuals': residuals,
        'fitted': fitted,
        'X': X,
        'beta': results.params
    }


def wild_cluster_bootstrap(
    data: pd.DataFrame,
    y_transformed: str,
    d: str,
    cluster_var: str,
    controls: list[str] | None = None,
    n_bootstrap: int = 999,
    weight_type: str = 'rademacher',
    alpha: float = 0.05,
    seed: int | None = None,
    impose_null: bool = True,
    full_enumeration: bool | None = None,
    use_wildboottest: bool = False,
    _force_loop: bool = False,
) -> WildClusterBootstrapResult:
    """
    Perform wild cluster bootstrap for inference with few clusters.
    
    Implements the wild cluster bootstrap of Cameron, Gelbach, and Miller (2008).
    This method provides more reliable inference when the number of clusters
    is small (< 20-30).
    
    **Algorithm:**
    
    1. Estimate original model and obtain residuals
    2. For each bootstrap replication: (a) Generate cluster-level weights
       (Rademacher/Mammen/Webb); (b) Construct bootstrap residuals u*_ic = w_c * u_ic;
       (c) Construct bootstrap outcome y*_ic = X_ic * beta_hat + u*_ic (if impose_null)
       or y*_ic = y_hat_ic + u*_ic (if not impose_null); (d) Re-estimate model and
       compute t-statistic
    3. Compute bootstrap p-value and confidence interval
    
    Parameters
    ----------
    data : pd.DataFrame
        Regression data.
    y_transformed : str
        Transformed outcome variable (e.g., after within-transformation).
    d : str
        Treatment indicator variable.
    cluster_var : str
        Clustering variable.
    controls : list[str], optional
        Control variables.
    n_bootstrap : int, default 999
        Number of bootstrap replications. Should be odd for symmetric
        p-value calculation.
    weight_type : str, default 'rademacher'
        Bootstrap weight distribution:
        - 'rademacher': P(w=1) = P(w=-1) = 0.5 (simplest, most common)
        - 'mammen': Two-point distribution matching skewness
        - 'webb': Six-point distribution (best for very few clusters)
    alpha : float, default 0.05
        Significance level for confidence interval.
    seed : int, optional
        Random seed for reproducibility.
    impose_null : bool, default True
        Whether to impose null hypothesis (H0: τ = 0) when constructing
        bootstrap samples. Recommended for hypothesis testing.
    full_enumeration : bool, optional
        Whether to use full enumeration of all 2^G Rademacher weight
        combinations instead of random sampling. If None (default),
        automatically enabled when G <= 12 and weight_type='rademacher'.
        Full enumeration produces deterministic results and is the most
        faithful implementation of the algorithm principle, as it computes
        the exact p-value without Monte Carlo error.
    use_wildboottest : bool, default False
        Whether to use the wildboottest package for fast algorithm.
        Requires wildboottest to be installed. When True, uses the
        optimized implementation that matches Stata boottest exactly.
        Note: wildboottest uses slightly different boundary handling,
        resulting in p-values about 0.002 lower than the algorithm
        principle definition.
    
    Returns
    -------
    WildClusterBootstrapResult
        Bootstrap results containing point estimate, standard error,
        confidence interval, p-value, and diagnostic information.
    
    Notes
    -----
    The wild cluster bootstrap is particularly useful when:
    - Number of clusters G < 30
    - Cluster sizes are unbalanced
    - Few treated clusters
    
    The method works by:
    1. Estimating the original model to get residuals
    2. Generating cluster-level random weights
    3. Creating bootstrap residuals by multiplying original residuals by weights
    4. Re-estimating the model with bootstrap outcomes
    5. Computing the bootstrap distribution of t-statistics
    
    When impose_null=True (recommended for hypothesis testing), the bootstrap
    outcome is constructed under the null hypothesis that the treatment effect
    is zero. This provides better size control.
    
    See Also
    --------
    diagnose_clustering : Diagnose clustering structure.
    recommend_clustering_level : Get recommendation for clustering level.

    Performance
    -----------
    Uses pre-computed projection matrices and batch matrix operations
    internally, avoiding per-iteration statsmodels overhead. The design
    matrix inverse ``(X'X)^{-1}`` and cluster structure are computed once
    before the bootstrap loop. Set ``_force_loop=True`` to use the
    single-iteration path (useful for numerical verification).

    .. versionchanged:: 0.2.0
       Vectorized OLS and batch matrix operations for >10x speedup.
    """
    # Set random seed.
    if seed is not None:
        np.random.seed(seed)
    
    # Validate inputs.
    if y_transformed not in data.columns:
        raise ValueError(f"Outcome variable '{y_transformed}' not found in data")
    if d not in data.columns:
        raise ValueError(f"Treatment variable '{d}' not found in data")
    if cluster_var not in data.columns:
        raise ValueError(f"Cluster variable '{cluster_var}' not found in data")
    if weight_type not in ['rademacher', 'mammen', 'webb']:
        raise ValueError(
            f"Unknown weight_type: {weight_type}. "
            f"Must be one of: 'rademacher', 'mammen', 'webb'"
        )
    
    # Get number of clusters.
    G = data[cluster_var].nunique()
    
    # Determine whether to use full enumeration.
    if full_enumeration is None:
        # Auto-enable when G <= 12 and using Rademacher weights.
        full_enumeration = (G <= 12 and weight_type == 'rademacher')
    
    # If wildboottest package is requested.
    if use_wildboottest:
        if not HAS_WILDBOOTTEST:
            raise ImportError(
                "wildboottest package is not installed. "
                "Install it with: pip install wildboottest"
            )
        return _wild_cluster_bootstrap_wildboottest(
            data=data,
            y_transformed=y_transformed,
            d=d,
            cluster_var=cluster_var,
            controls=controls,
            n_bootstrap=n_bootstrap,
            weight_type=weight_type,
            alpha=alpha,
            seed=seed,
            impose_null=impose_null,
            full_enumeration=full_enumeration,
        )
    
    # Step 1: Estimate original model.
    original_result = _estimate_ols_for_bootstrap(
        data, y_transformed, d, cluster_var, controls
    )
    
    att_original = original_result['att']
    se_original = original_result['se']
    
    # Handle degenerate cases where SE is zero or NaN.
    if se_original == 0 or np.isnan(se_original):
        # Return degenerate result.
        return WildClusterBootstrapResult(
            att=att_original,
            se_bootstrap=np.nan,
            ci_lower=np.nan,
            ci_upper=np.nan,
            pvalue=np.nan,
            n_clusters=data[cluster_var].nunique(),
            n_bootstrap=n_bootstrap,
            weight_type=weight_type,
            t_stat_original=np.nan,
            t_stats_bootstrap=np.array([]),
            rejection_rate=np.nan
        )
    
    t_stat_original = att_original / se_original
    
    # Precompute matrices and cluster structure for bootstrap replication
    precomp = _precompute_bootstrap_matrices(data, y_transformed, d, cluster_var, controls)
    
    obs_cluster_idx = precomp['obs_cluster_idx']
    G = precomp['G']
    
    # Fitted values and residuals from restricted/unrestricted model
    if impose_null:
        # Restricted model: y = α + ε (intercept only, no treatment effect)
        X_restricted = np.ones((precomp['N'], 1), dtype=np.float64)
        beta_r = np.linalg.lstsq(X_restricted, precomp['y'], rcond=None)[0]
        fitted_restricted = (X_restricted @ beta_r).ravel()
        residuals_restricted = precomp['y'] - fitted_restricted
    else:
        # Unrestricted model fitted values and residuals
        beta_u = precomp['P'] @ precomp['y']
        fitted_unrestricted = precomp['X'] @ beta_u
        residuals_unrestricted = precomp['y'] - fitted_unrestricted
    
    # Step 2: Bootstrap
    # Determine full enumeration vs random sampling
    if full_enumeration and weight_type == 'rademacher':
        # Full enumeration: generate all 2^G Rademacher weight combinations
        all_weights = _generate_all_rademacher_weights(G)
        actual_n_bootstrap = len(all_weights)
        use_full_enum = True
    else:
        actual_n_bootstrap = n_bootstrap
        use_full_enum = False
    
    # Select fitted values and residuals for y* construction
    if impose_null:
        fitted_base = fitted_restricted
        resid_base = residuals_restricted
    else:
        fitted_base = fitted_unrestricted
        resid_base = residuals_unrestricted

    # Decide between batch mode and loop mode
    if _force_loop:
        # --- Loop mode (reference implementation) ---
        t_stats_bootstrap = np.full(actual_n_bootstrap, np.nan)
        att_bootstrap = np.full(actual_n_bootstrap, np.nan)

        for b in range(actual_n_bootstrap):
            if use_full_enum:
                weights = all_weights[b]
            else:
                weights = _generate_bootstrap_weights(G, weight_type)

            obs_weights = weights[obs_cluster_idx]
            y_star = fitted_base + obs_weights * resid_base

            try:
                att_b, se_b = _fast_ols_cluster_se(y_star, precomp)
                if se_b > 0 and not np.isnan(se_b):
                    t_stats_bootstrap[b] = att_b / se_b
                    att_bootstrap[b] = att_b
                else:
                    t_stats_bootstrap[b] = np.nan
                    att_bootstrap[b] = np.nan
            except Exception:
                t_stats_bootstrap[b] = np.nan
                att_bootstrap[b] = np.nan
    else:
        # --- Batch mode: eliminate Python for-loop for OLS ---
        # Generate all weights at once: shape (B, G)
        if use_full_enum:
            # all_weights already generated above as (2^G, G)
            pass
        else:
            all_weights = _generate_all_bootstrap_weights(
                actual_n_bootstrap, G, weight_type
            )

        # Batch compute ATT and t-statistics via matrix operations
        att_bootstrap, t_stats_bootstrap = _batch_bootstrap(
            all_weights, precomp, fitted_base, resid_base
        )
    
    # Remove NaN values.
    valid_mask = ~np.isnan(t_stats_bootstrap)
    t_stats_valid = t_stats_bootstrap[valid_mask]
    att_valid = att_bootstrap[valid_mask]
    
    if len(t_stats_valid) == 0:
        # All bootstrap replications failed.
        return WildClusterBootstrapResult(
            att=att_original,
            se_bootstrap=np.nan,
            ci_lower=np.nan,
            ci_upper=np.nan,
            pvalue=np.nan,
            n_clusters=G,
            n_bootstrap=n_bootstrap,
            weight_type=weight_type,
            t_stat_original=t_stat_original,
            t_stats_bootstrap=t_stats_bootstrap,
            rejection_rate=np.nan
        )
    
    # Step 3: Compute bootstrap p-value and confidence interval.
    #
    # The p-value is computed as: p = P(|t*| >= |t_orig| | H0)
    #
    # Using >= instead of > ensures that:
    # 1. In full enumeration, the all +1 weights combination produces t* = t_orig.
    # 2. This combination is a valid realization under the null and should be included.
    # 3. Excluding any valid combination would bias the p-value estimate.
    pvalue = np.mean(np.abs(t_stats_valid) >= np.abs(t_stat_original))
    
    # Bootstrap SE (standard deviation of bootstrap estimates).
    se_bootstrap = np.std(att_valid)
    
    # Confidence interval construction.
    # Use symmetric CI based on |t*| distribution:
    #   CI = [att - t*_crit * se, att + t*_crit * se]
    # where t*_crit is the (1-alpha) quantile of |t*|.
    
    if impose_null:
        # Symmetric CI based on (1-alpha) quantile of |t*|.
        t_abs_crit = np.percentile(np.abs(t_stats_valid), 100 * (1 - alpha))
        ci_lower = att_original - t_abs_crit * se_original
        ci_upper = att_original + t_abs_crit * se_original
    else:
        # When not imposing null, use percentile CI.
        ci_lower = np.percentile(att_valid, 100 * alpha / 2)
        ci_upper = np.percentile(att_valid, 100 * (1 - alpha / 2))
    
    return WildClusterBootstrapResult(
        att=att_original,
        se_bootstrap=se_bootstrap,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        pvalue=pvalue,
        n_clusters=G,
        n_bootstrap=actual_n_bootstrap,
        weight_type=weight_type,
        t_stat_original=t_stat_original,
        t_stats_bootstrap=t_stats_bootstrap,
        rejection_rate=pvalue,
        ci_method='percentile_t'
    )


def _wild_cluster_bootstrap_wildboottest(
    data: pd.DataFrame,
    y_transformed: str,
    d: str,
    cluster_var: str,
    controls: list[str] | None = None,
    n_bootstrap: int = 999,
    weight_type: str = 'rademacher',
    alpha: float = 0.05,
    seed: int | None = None,
    impose_null: bool = True,
    full_enumeration: bool = False,
) -> WildClusterBootstrapResult:
    """
    Perform wild cluster bootstrap using the wildboottest package.
    
    This function wraps the wildboottest package to provide an optimized
    implementation of the wild cluster bootstrap algorithm.
    
    Parameters
    ----------
    data : pd.DataFrame
        Regression data.
    y_transformed : str
        Outcome variable name.
    d : str
        Treatment variable name.
    cluster_var : str
        Clustering variable name.
    controls : list[str], optional
        Control variables.
    n_bootstrap : int, default 999
        Number of bootstrap replications (ignored when using full enumeration).
    weight_type : str, default 'rademacher'
        Type of bootstrap weights.
    alpha : float, default 0.05
        Significance level.
    seed : int, optional
        Random seed for reproducibility.
    impose_null : bool, default True
        Whether to impose the null hypothesis.
    full_enumeration : bool, default False
        Whether to use full enumeration.
    
    Returns
    -------
    WildClusterBootstrapResult
        Bootstrap results.
    """
    if not HAS_WILDBOOTTEST:
        raise ImportError("wildboottest package is not installed")
    
    # Construct formula.
    if controls:
        formula = f"{y_transformed} ~ {d} + {' + '.join(controls)}"
    else:
        formula = f"{y_transformed} ~ {d}"
    
    # Create OLS model using statsmodels.
    # Note: wildboottest requires an unfitted model object.
    model = smf.ols(formula, data=data)
    
    # Get number of clusters.
    G = data[cluster_var].nunique()
    
    # Determine bootstrap type and number of replications.
    if full_enumeration and weight_type == 'rademacher':
        # Full enumeration: set B > 2^G to trigger automatic full enumeration.
        actual_n_bootstrap = 2 ** G
        B_param = actual_n_bootstrap * 2  # Set > 2^G to trigger full enumeration.
    else:
        actual_n_bootstrap = n_bootstrap
        B_param = n_bootstrap
    
    # Map weight type.
    weight_map = {
        'rademacher': 'rademacher',
        'mammen': 'mammen',
        'webb': 'webb'
    }
    
    # Call wildboottest.
    boot_result = _wildboottest(
        model=model,
        cluster=data[cluster_var],
        param=d,
        B=B_param,
        weights_type=weight_map[weight_type],
        impose_null=impose_null,
        bootstrap_type='11',  # WCR11 is the default type.
        seed=seed,
    )
    
    # Fit model to obtain coefficients.
    results = model.fit()
    
    # Extract results - wildboottest returns a DataFrame.
    att = results.params[d]
    se_original = results.bse[d]
    
    # Extract p-value and t-statistic from DataFrame.
    if isinstance(boot_result, pd.DataFrame):
        pvalue = boot_result['p-value'].values[0]
        t_stat = boot_result['statistic'].values[0]
    else:
        # Fallback for older versions.
        t_stat = boot_result.t_stat if hasattr(boot_result, 't_stat') else att / se_original
        pvalue = boot_result.pvalue if hasattr(boot_result, 'pvalue') else np.nan
    
    # Confidence interval using t-distribution approximation.
    from scipy import stats as scipy_stats
    t_crit = scipy_stats.t.ppf(1 - alpha/2, G - 1)
    ci_lower = att - t_crit * se_original
    ci_upper = att + t_crit * se_original
    
    # Bootstrap SE.
    se_bootstrap = se_original
    
    return WildClusterBootstrapResult(
        att=att,
        se_bootstrap=se_bootstrap,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        pvalue=pvalue,
        n_clusters=G,
        n_bootstrap=actual_n_bootstrap,
        weight_type=weight_type,
        t_stat_original=t_stat,
        t_stats_bootstrap=np.array([]),  # wildboottest does not return full distribution.
        rejection_rate=pvalue,
        ci_method='wildboottest'
    )


def _compute_bootstrap_pvalue_at_null(
    data: pd.DataFrame,
    y_var: str,
    d_var: str,
    cluster_var: str,
    null_value: float,
    att_original: float,
    se_original: float,
    n_bootstrap: int,
    weight_type: str,
    controls: list[str] | None = None,
    seed: int | None = None,
) -> float:
    """
    Compute bootstrap p-value at a given null hypothesis value.
    
    Internal function used for test inversion confidence intervals.
    This is the legacy slow path kept for reference and fallback.
    
    Parameters
    ----------
    data : pd.DataFrame
        Regression data.
    y_var : str
        Outcome variable name.
    d_var : str
        Treatment variable name.
    cluster_var : str
        Clustering variable name.
    null_value : float
        Null hypothesis value theta (H0: beta = theta).
    att_original : float
        Original ATT estimate.
    se_original : float
        Original cluster-robust standard error.
    n_bootstrap : int
        Number of bootstrap replications.
    weight_type : str
        Type of bootstrap weights.
    controls : list[str], optional
        Control variables.
    seed : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    float
        Bootstrap p-value.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Original t-statistic (relative to null_value).
    t_original = (att_original - null_value) / se_original
    
    # Cluster information.
    cluster_ids = data[cluster_var].values
    unique_clusters = np.unique(cluster_ids)
    G = len(unique_clusters)
    cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}
    obs_cluster_idx = np.array([cluster_to_idx[c] for c in cluster_ids])
    
    # Restricted model: y - theta*d = alpha + epsilon.
    y_restricted = data[y_var].values - null_value * data[d_var].values
    X_r = np.ones((len(data), 1))
    model_r = sm.OLS(y_restricted, X_r)
    results_r = model_r.fit()
    fitted_r = results_r.fittedvalues
    residuals_r = results_r.resid
    
    t_stats = []
    for b in range(n_bootstrap):
        # Generate weights.
        weights = _generate_bootstrap_weights(G, weight_type)
        obs_weights = weights[obs_cluster_idx]
        
        # Bootstrap sample: y* = alpha_hat + theta*d + w*epsilon_hat.
        u_star = obs_weights * residuals_r
        y_star = fitted_r + null_value * data[d_var].values + u_star
        
        # Re-estimate.
        data_boot = data.copy()
        data_boot[y_var] = y_star
        
        try:
            boot_result = _estimate_ols_for_bootstrap(
                data_boot, y_var, d_var, cluster_var, controls
            )
            if boot_result['se'] > 0 and not np.isnan(boot_result['se']):
                # t* = (beta* - theta) / se*
                t_stats.append((boot_result['att'] - null_value) / boot_result['se'])
        except Exception:
            pass
    
    if len(t_stats) == 0:
        return np.nan
    
    t_stats = np.array(t_stats)
    
    # Two-sided p-value.
    pvalue = np.mean(np.abs(t_stats) >= np.abs(t_original))
    
    return pvalue


def _compute_bootstrap_pvalue_at_null_fast(
    null_value: float,
    precomp: dict,
    d_values: np.ndarray,
    att_original: float,
    se_original: float,
    n_bootstrap: int,
    weight_type: str,
    seed: int | None = None,
) -> float:
    """
    Compute bootstrap p-value at a given null hypothesis value using precomputed matrices.

    Numerically equivalent to _compute_bootstrap_pvalue_at_null(), but achieves
    substantial performance gains by reusing precomputed projection matrices
    and leveraging batch matrix operations.

    Algorithm:
        1. Restricted model: y_r = y - θd, fit intercept model y_r = α + ε
        2. Batch-generate all bootstrap weights: W shape (B, G)
        3. Batch-construct y*: Y* = fitted_r + θd + W_obs * residuals_r
        4. Batch OLS: Beta = P @ Y*.T
        5. Compute cluster-robust SE per bootstrap sample (meat matrix differs per sample)
        6. t* = (att* - θ) / se*
        7. p-value = mean(|t*| >= |t_orig|)

    Parameters
    ----------
    null_value : float
        Null hypothesis value θ (H₀: β = θ).
    precomp : dict
        Return value of _precompute_bootstrap_matrices(), shared across all grid points.
    d_values : ndarray, shape (N,)
        Treatment indicator vector.
    att_original : float
        Original ATT point estimate.
    se_original : float
        Original cluster-robust standard error.
    n_bootstrap : int
        Number of bootstrap replications.
    weight_type : str
        Bootstrap weight type: 'rademacher', 'mammen', 'webb'.
    seed : int, optional
        Random seed. Resets the RNG state on each call to ensure identical
        bootstrap weights at every grid point (consistent with the original implementation).

    Returns
    -------
    float
        Bootstrap p-value.
    """
    if seed is not None:
        np.random.seed(seed)

    # Original t-statistic (relative to null_value)
    t_original = (att_original - null_value) / se_original

    N = precomp['N']
    G = precomp['G']
    obs_cluster_idx = precomp['obs_cluster_idx']
    P = precomp['P']          # (k, N)
    X = precomp['X']          # (N, k)

    # Restricted model: y - θd = α + ε (intercept-only model)
    y_restricted = precomp['y'] - null_value * d_values
    # OLS solution for the intercept model is the sample mean
    alpha_hat = y_restricted.mean()
    fitted_r = np.full(N, alpha_hat)
    residuals_r = y_restricted - alpha_hat

    # Batch-generate all bootstrap weights: (B, G)
    all_weights = _generate_all_bootstrap_weights(n_bootstrap, G, weight_type)

    # Map to observation level: (B, N)
    obs_weights_all = all_weights[:, obs_cluster_idx]

    # Batch-construct Y*: y*_i = fitted_r_i + θ * d_i + w_{g(i)} * residuals_r_i
    # Shape: (B, N)
    Y_star = (fitted_r[np.newaxis, :]
              + null_value * d_values[np.newaxis, :]
              + obs_weights_all * residuals_r[np.newaxis, :])

    # Batch OLS coefficients: Beta = P @ Y*.T -> (k, B)
    Beta = P @ Y_star.T
    att_all = Beta[1, :]  # (B,) — treatment effect coefficient

    # Batch residuals: (B, N)
    Residuals = Y_star - (X @ Beta).T

    # Release large matrices to reduce peak memory usage
    del Y_star, obs_weights_all

    # Vectorized cluster-robust SE: eliminates B × G double loop
    var_11 = _batch_cluster_se_variance_11(Residuals, precomp)
    se_all = np.sqrt(np.maximum(var_11, 0))
    valid = (se_all > 0) & ~np.isnan(se_all)

    t_stats = np.full(n_bootstrap, np.nan)
    t_stats[valid] = (att_all[valid] - null_value) / se_all[valid]

    # Valid t-statistics
    valid = ~np.isnan(t_stats)
    if not np.any(valid):
        return np.nan

    # Two-sided p-value
    pvalue = np.mean(np.abs(t_stats[valid]) >= np.abs(t_original))
    return pvalue


def wild_cluster_bootstrap_test_inversion(
    data: pd.DataFrame,
    y_transformed: str,
    d: str,
    cluster_var: str,
    controls: list[str] | None = None,
    n_bootstrap: int = 999,
    weight_type: str = 'rademacher',
    alpha: float = 0.05,
    seed: int | None = None,
    grid_points: int = 25,
    ci_tol: float = 0.01,
    _force_slow: bool = False,
) -> WildClusterBootstrapResult:
    """
    Compute wild cluster bootstrap confidence interval using test inversion.
    
    This method constructs confidence intervals by inverting hypothesis tests:
    CI = {theta : p(theta) >= alpha}
    
    That is, the CI consists of all null hypothesis values for which the
    bootstrap p-value exceeds the significance level.
    
    Parameters
    ----------
    data : pd.DataFrame
        Regression data.
    y_transformed : str
        Transformed outcome variable.
    d : str
        Treatment variable.
    cluster_var : str
        Clustering variable.
    controls : list[str], optional
        Control variables.
    n_bootstrap : int, default 999
        Number of bootstrap replications.
    weight_type : str, default 'rademacher'
        Type of bootstrap weights.
    alpha : float, default 0.05
        Significance level.
    seed : int, optional
        Random seed for reproducibility.
    grid_points : int, default 25
        Number of grid points for initial search.
    ci_tol : float, default 0.01
        Tolerance for CI boundary precision.
    _force_slow : bool, default False
        Internal parameter. When True, uses the original slow path (for regression testing).
    
    Returns
    -------
    WildClusterBootstrapResult
        Results containing test inversion CI.
    
    Notes
    -----
    Advantages of test inversion CI:
    
    1. Can be more accurate than percentile-t CI in some settings.
    2. Handles asymmetric distributions appropriately.
    
    Disadvantages:
    
    1. More computationally intensive (requires multiple bootstrap runs).
    2. Requires numerical optimization to find boundaries.

    Performance
    -----------
    The projection matrix ``(X'X)^{-1}X'`` is pre-computed once and shared
    across all grid points and bisection iterations, reducing the cost from
    ~35 full bootstrap runs to ~35 batch matrix operations. Set
    ``_force_slow=True`` to use the original per-grid-point path.

    .. versionchanged:: 0.2.0
       Shared pre-computation across grid points for >6x speedup.
    """
    # Set random seed.
    rng_state = np.random.get_state() if seed is None else None
    if seed is not None:
        np.random.seed(seed)
    
    # Validate inputs.
    if y_transformed not in data.columns:
        raise ValueError(f"Outcome variable '{y_transformed}' not found in data")
    if d not in data.columns:
        raise ValueError(f"Treatment variable '{d}' not found in data")
    if cluster_var not in data.columns:
        raise ValueError(f"Cluster variable '{cluster_var}' not found in data")
    
    # Step 1: Estimate original model.
    original_result = _estimate_ols_for_bootstrap(
        data, y_transformed, d, cluster_var, controls
    )
    
    att_original = original_result['att']
    se_original = original_result['se']
    
    if se_original == 0 or np.isnan(se_original):
        return WildClusterBootstrapResult(
            att=att_original,
            se_bootstrap=np.nan,
            ci_lower=np.nan,
            ci_upper=np.nan,
            pvalue=np.nan,
            n_clusters=data[cluster_var].nunique(),
            n_bootstrap=n_bootstrap,
            weight_type=weight_type,
            t_stat_original=np.nan,
            t_stats_bootstrap=np.array([]),
            rejection_rate=np.nan,
            ci_method='test_inversion'
        )
    
    t_stat_original = att_original / se_original
    G = data[cluster_var].nunique()
    
    # Precompute projection matrices and cluster structure (once, shared across all grid points)
    precomp = _precompute_bootstrap_matrices(
        data, y_transformed, d, cluster_var, controls
    )
    d_values = data[d].values.astype(np.float64)
    
    # Select p-value computation function
    if _force_slow:
        # Slow path: used for regression testing validation
        def _pvalue_func(theta):
            return _compute_bootstrap_pvalue_at_null(
                data, y_transformed, d, cluster_var,
                null_value=theta,
                att_original=att_original,
                se_original=se_original,
                n_bootstrap=n_bootstrap,
                weight_type=weight_type,
                controls=controls,
                seed=seed,
            )
    else:
        # Fast path: uses precomputed matrices and batch operations
        def _pvalue_func(theta):
            return _compute_bootstrap_pvalue_at_null_fast(
                null_value=theta,
                precomp=precomp,
                d_values=d_values,
                att_original=att_original,
                se_original=se_original,
                n_bootstrap=n_bootstrap,
                weight_type=weight_type,
                seed=seed,
            )
    
    # Step 2: Compute p-value at theta=0 (for reporting).
    pvalue_at_zero = _pvalue_func(0.0)
    
    # Step 3: Find CI boundaries using test inversion.
    # First perform coarse grid search.
    # Search range: ATT +/- 4 * SE.
    search_range = 4 * se_original
    theta_min = att_original - search_range
    theta_max = att_original + search_range
    
    theta_grid = np.linspace(theta_min, theta_max, grid_points)
    pvalues_grid = np.array([_pvalue_func(theta) for theta in theta_grid])
    
    # Find region where p >= alpha.
    ci_mask = pvalues_grid >= alpha
    
    if not np.any(ci_mask):
        # No region found with p >= alpha, use percentile-t as fallback.
        ci_lower = att_original - 1.96 * se_original
        ci_upper = att_original + 1.96 * se_original
    else:
        # Coarse boundaries.
        ci_thetas = theta_grid[ci_mask]
        ci_lower_approx = ci_thetas.min()
        ci_upper_approx = ci_thetas.max()
        
        # Use bisection to precisely locate boundaries.
        def pvalue_minus_alpha(theta):
            return _pvalue_func(theta) - alpha
        
        # Find lower bound.
        try:
            # Search for p < alpha point to the left of ATT.
            lower_search_min = theta_min
            lower_search_max = att_original
            ci_lower = brentq(pvalue_minus_alpha, lower_search_min, lower_search_max, xtol=ci_tol)
        except ValueError:
            ci_lower = ci_lower_approx
        
        # Find upper bound.
        try:
            # Search for p < alpha point to the right of ATT.
            upper_search_min = att_original
            upper_search_max = theta_max
            ci_upper = brentq(pvalue_minus_alpha, upper_search_min, upper_search_max, xtol=ci_tol)
        except ValueError:
            ci_upper = ci_upper_approx
    
    # Compute bootstrap SE (simplified using CI width estimate).
    se_bootstrap = (ci_upper - ci_lower) / (2 * 1.96)
    
    # Restore random state.
    if rng_state is not None:
        np.random.set_state(rng_state)
    
    return WildClusterBootstrapResult(
        att=att_original,
        se_bootstrap=se_bootstrap,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        pvalue=pvalue_at_zero,
        n_clusters=G,
        n_bootstrap=n_bootstrap,
        weight_type=weight_type,
        t_stat_original=t_stat_original,
        t_stats_bootstrap=np.array([]),  # Test inversion does not save t distribution.
        rejection_rate=pvalue_at_zero,
        ci_method='test_inversion'
    )

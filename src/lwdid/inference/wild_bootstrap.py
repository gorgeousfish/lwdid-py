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
    residuals = original_result['residuals']
    fitted = original_result['fitted']
    X = original_result['X']
    
    cluster_ids = data[cluster_var].values
    unique_clusters = np.unique(cluster_ids)
    G = len(unique_clusters)
    
    # Create cluster-to-index mapping.
    cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}
    obs_cluster_idx = np.array([cluster_to_idx[c] for c in cluster_ids])
    
    # When impose_null=True, re-estimate under H0 to obtain restricted residuals.
    if impose_null:
        # Restricted model: y = alpha + epsilon (intercept only, no treatment).
        y_values = data[y_transformed].values
        X_restricted = np.ones((len(data), 1))
        model_restricted = sm.OLS(y_values, X_restricted)
        results_restricted = model_restricted.fit()
        fitted_restricted = results_restricted.fittedvalues
        residuals_restricted = results_restricted.resid
    
    # Step 2: Bootstrap loop.
    # Determine whether to use full enumeration or random sampling.
    if full_enumeration and weight_type == 'rademacher':
        # Full enumeration: generate all 2^G Rademacher weight combinations.
        all_weights = _generate_all_rademacher_weights(G)
        actual_n_bootstrap = len(all_weights)
        use_full_enum = True
    else:
        actual_n_bootstrap = n_bootstrap
        use_full_enum = False
    
    t_stats_bootstrap = np.zeros(actual_n_bootstrap)
    att_bootstrap = np.zeros(actual_n_bootstrap)
    
    for b in range(actual_n_bootstrap):
        # Generate cluster-level weights.
        if use_full_enum:
            weights = all_weights[b]
        else:
            weights = _generate_bootstrap_weights(G, weight_type)
        
        # Map weights to observations.
        obs_weights = weights[obs_cluster_idx]
        
        # Construct bootstrap residuals and outcome variable.
        if impose_null:
            # Use restricted model residuals under the null hypothesis.
            # y* = alpha_hat_r + w * epsilon_hat_r
            u_star = obs_weights * residuals_restricted
            y_star = fitted_restricted + u_star
        else:
            # When not imposing null, use unrestricted model residuals.
            u_star = obs_weights * residuals
            y_star = fitted + u_star
        
        # Create bootstrap data.
        data_boot = data.copy()
        data_boot[y_transformed] = y_star
        
        # Re-estimate and compute t-statistic.
        try:
            boot_result = _estimate_ols_for_bootstrap(
                data_boot, y_transformed, d, cluster_var, controls
            )
            if boot_result['se'] > 0 and not np.isnan(boot_result['se']):
                t_stats_bootstrap[b] = boot_result['att'] / boot_result['se']
                att_bootstrap[b] = boot_result['att']
            else:
                t_stats_bootstrap[b] = np.nan
                att_bootstrap[b] = np.nan
        except Exception:
            t_stats_bootstrap[b] = np.nan
            att_bootstrap[b] = np.nan
    
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
    
    # Step 2: Compute p-value at theta=0 (for reporting).
    pvalue_at_zero = _compute_bootstrap_pvalue_at_null(
        data, y_transformed, d, cluster_var,
        null_value=0,
        att_original=att_original,
        se_original=se_original,
        n_bootstrap=n_bootstrap,
        weight_type=weight_type,
        controls=controls,
        seed=seed
    )
    
    # Step 3: Find CI boundaries using test inversion.
    # First perform coarse grid search.
    # Search range: ATT +/- 4 * SE.
    search_range = 4 * se_original
    theta_min = att_original - search_range
    theta_max = att_original + search_range
    
    theta_grid = np.linspace(theta_min, theta_max, grid_points)
    pvalues_grid = []
    
    for theta in theta_grid:
        pval = _compute_bootstrap_pvalue_at_null(
            data, y_transformed, d, cluster_var,
            null_value=theta,
            att_original=att_original,
            se_original=se_original,
            n_bootstrap=n_bootstrap,
            weight_type=weight_type,
            controls=controls,
            seed=seed
        )
        pvalues_grid.append(pval)
    
    pvalues_grid = np.array(pvalues_grid)
    
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
            pval = _compute_bootstrap_pvalue_at_null(
                data, y_transformed, d, cluster_var,
                null_value=theta,
                att_original=att_original,
                se_original=se_original,
                n_bootstrap=n_bootstrap,
                weight_type=weight_type,
                controls=controls,
                seed=seed
            )
            return pval - alpha
        
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

# -*- coding: utf-8 -*-
"""
Monte Carlo Simulation Runner

Provides the core functionality for running Monte Carlo simulations
with support for parallel execution and comprehensive result collection.
"""

from __future__ import annotations

import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# 支持相对导入和直接导入两种方式
try:
    from .results import MonteCarloResults
except ImportError:
    from results import MonteCarloResults


def run_single_replication(
    rep_id: int,
    dgp_func: Callable,
    estimator_func: Callable,
    dgp_kwargs: Dict[str, Any],
    estimator_kwargs: Dict[str, Any],
    target_period: Optional[int] = None,
) -> Tuple[int, Dict[str, Any]]:
    """
    Run a single Monte Carlo replication.
    
    Parameters
    ----------
    rep_id : int
        Replication identifier (used for seeding).
    dgp_func : Callable
        DGP function: (**kwargs) -> (data, params).
    estimator_func : Callable
        Estimator function: (data, period, **kwargs) -> result.
    dgp_kwargs : dict
        Arguments for DGP function.
    estimator_kwargs : dict
        Arguments for estimator function.
    target_period : int, optional
        Target evaluation period.
    
    Returns
    -------
    rep_id : int
        Replication identifier.
    result : dict
        Result dictionary with att, se, ci_lower, ci_upper, pvalue, 
        true_att, success, and optionally error.
    """
    try:
        # Set seed for this replication
        base_seed = dgp_kwargs.get('seed', 0) or 0
        rep_seed = base_seed + rep_id
        dgp_kwargs_rep = {**dgp_kwargs, 'seed': rep_seed}
        
        # Generate data
        dgp_output = dgp_func(**dgp_kwargs_rep)
        
        # Handle different DGP return formats
        if len(dgp_output) == 2:
            data, params = dgp_output
        elif len(dgp_output) == 3:
            data, true_atts, params = dgp_output
            params['true_atts'] = true_atts
        else:
            raise ValueError(f"Unexpected DGP output length: {len(dgp_output)}")

        # Extract true ATT
        true_att = _extract_true_att(params, target_period)
        
        # Run estimator
        est_result = estimator_func(data, target_period, **estimator_kwargs)
        
        # Parse estimator result
        att, se, ci_lower, ci_upper, pvalue = _parse_estimator_result(est_result)
        
        return rep_id, {
            'att': att,
            'se': se,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'pvalue': pvalue,
            'true_att': true_att,
            'success': True,
        }
    
    except Exception as e:
        return rep_id, {
            'att': np.nan,
            'se': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'pvalue': np.nan,
            'true_att': np.nan,
            'success': False,
            'error': str(e),
        }


def _extract_true_att(
    params: Dict[str, Any],
    target_period: Optional[int],
) -> float:
    """Extract true ATT from DGP parameters."""
    # Try different parameter names
    if 'true_att' in params:
        return float(params['true_att'])
    
    if 'tau' in params:
        return float(params['tau'])
    
    if 'true_atts' in params:
        true_atts = params['true_atts']
        if isinstance(true_atts, dict):
            if target_period is not None and target_period in true_atts:
                return float(true_atts[target_period])
            # Return average of all periods
            return float(np.mean(list(true_atts.values())))
        return float(true_atts)
    
    if 'att_by_period' in params:
        att_by_period = params['att_by_period']
        if target_period is not None and target_period in att_by_period:
            return float(att_by_period[target_period])
        return float(np.mean(list(att_by_period.values())))
    
    if 'average_effect' in params:
        return float(params['average_effect'])
    
    return 0.0


def _parse_estimator_result(est_result: Any) -> Tuple[float, float, float, float, float]:
    """Parse estimator result into standard format."""
    # Handle result object with attributes
    if hasattr(est_result, 'att'):
        att = float(est_result.att)
        se = float(getattr(est_result, 'se', np.nan))
        ci_lower = float(getattr(est_result, 'ci_lower', np.nan))
        ci_upper = float(getattr(est_result, 'ci_upper', np.nan))
        pvalue = float(getattr(est_result, 'pvalue', np.nan))
        return att, se, ci_lower, ci_upper, pvalue
    
    # Handle tuple result
    if isinstance(est_result, tuple):
        att = float(est_result[0])
        se = float(est_result[1]) if len(est_result) > 1 else np.nan
        ci_lower = float(est_result[2]) if len(est_result) > 2 else np.nan
        ci_upper = float(est_result[3]) if len(est_result) > 3 else np.nan
        pvalue = float(est_result[4]) if len(est_result) > 4 else np.nan
        return att, se, ci_lower, ci_upper, pvalue
    
    # Handle dict result
    if isinstance(est_result, dict):
        att = float(est_result.get('att', est_result.get('estimate', np.nan)))
        se = float(est_result.get('se', est_result.get('std_error', np.nan)))
        ci_lower = float(est_result.get('ci_lower', est_result.get('conf_int_lower', np.nan)))
        ci_upper = float(est_result.get('ci_upper', est_result.get('conf_int_upper', np.nan)))
        pvalue = float(est_result.get('pvalue', est_result.get('p_value', np.nan)))
        return att, se, ci_lower, ci_upper, pvalue
    
    # Handle scalar result
    return float(est_result), np.nan, np.nan, np.nan, np.nan


def run_monte_carlo(
    estimator_func: Callable,
    dgp_func: Callable,
    n_reps: int = 1000,
    dgp_kwargs: Dict[str, Any] | None = None,
    estimator_kwargs: Dict[str, Any] | None = None,
    target_period: int | None = None,
    alpha: float = 0.05,
    parallel: bool = True,
    n_jobs: int = -1,
    seed: int | None = None,
    verbose: bool = True,
    estimator_name: str = "Unknown",
    dgp_name: str = "Unknown",
    scenario: str = "Unknown",
) -> MonteCarloResults:
    """
    Run Monte Carlo simulation to evaluate estimator performance.
    
    Parameters
    ----------
    estimator_func : Callable
        Estimator function with signature: (data, period, **kwargs) -> result.
        Result should have att, se, ci_lower, ci_upper attributes or be a tuple.
    dgp_func : Callable
        DGP function with signature: (**kwargs) -> (data, params).
    n_reps : int, default=1000
        Number of Monte Carlo replications.
    dgp_kwargs : dict, optional
        Arguments passed to DGP function.
    estimator_kwargs : dict, optional
        Arguments passed to estimator function.
    target_period : int, optional
        Target evaluation period.
    alpha : float, default=0.05
        Significance level for hypothesis testing.
    parallel : bool, default=True
        Whether to use parallel execution.
    n_jobs : int, default=-1
        Number of parallel jobs (-1 for all CPUs).
    seed : int, optional
        Base random seed for reproducibility.
    verbose : bool, default=True
        Whether to show progress.
    estimator_name : str
        Name of the estimator (for reporting).
    dgp_name : str
        Name of the DGP (for reporting).
    scenario : str
        Scenario identifier (for reporting).
    
    Returns
    -------
    MonteCarloResults
        Complete Monte Carlo results including all metrics and raw data.
    
    Examples
    --------
    >>> from tests.monte_carlo.dgp.small_sample import generate_small_sample_dgp
    >>> 
    >>> def my_estimator(data, period, **kwargs):
    ...     # Your estimation logic
    ...     return att, se, ci_lower, ci_upper
    >>> 
    >>> results = run_monte_carlo(
    ...     estimator_func=my_estimator,
    ...     dgp_func=generate_small_sample_dgp,
    ...     n_reps=100,
    ...     dgp_kwargs={'scenario': 1},
    ...     estimator_name='My Estimator',
    ...     dgp_name='small_sample',
    ...     scenario='scenario_1',
    ... )
    >>> print(results.summary())
    """
    dgp_kwargs = dgp_kwargs or {}
    estimator_kwargs = estimator_kwargs or {}
    
    if seed is not None:
        dgp_kwargs['seed'] = seed
    
    # Get n_units from dgp_kwargs for reporting
    n_units = dgp_kwargs.get('n_units', 0)
    
    # Collect results
    results_list: List[Dict[str, Any]] = []
    
    if parallel and n_jobs != 1:
        results_list = _run_parallel(
            n_reps=n_reps,
            dgp_func=dgp_func,
            estimator_func=estimator_func,
            dgp_kwargs=dgp_kwargs,
            estimator_kwargs=estimator_kwargs,
            target_period=target_period,
            n_jobs=n_jobs,
            verbose=verbose,
            estimator_name=estimator_name,
        )
    else:
        results_list = _run_sequential(
            n_reps=n_reps,
            dgp_func=dgp_func,
            estimator_func=estimator_func,
            dgp_kwargs=dgp_kwargs,
            estimator_kwargs=estimator_kwargs,
            target_period=target_period,
            verbose=verbose,
            estimator_name=estimator_name,
        )
    
    # Aggregate results
    return _aggregate_results(
        results_list=results_list,
        n_reps=n_reps,
        n_units=n_units,
        alpha=alpha,
        estimator_name=estimator_name,
        dgp_name=dgp_name,
        scenario=scenario,
        target_period=target_period,
    )


def _run_parallel(
    n_reps: int,
    dgp_func: Callable,
    estimator_func: Callable,
    dgp_kwargs: Dict[str, Any],
    estimator_kwargs: Dict[str, Any],
    target_period: Optional[int],
    n_jobs: int,
    verbose: bool,
    estimator_name: str,
) -> List[Dict[str, Any]]:
    """Run Monte Carlo replications in parallel."""
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 4
    
    results_list = []
    
    # Note: ProcessPoolExecutor may have issues with closures
    # Using sequential execution as fallback for now
    if verbose:
        print(f"Running {n_reps} replications for {estimator_name}...")
    
    try:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(
                    run_single_replication,
                    rep_id=rep,
                    dgp_func=dgp_func,
                    estimator_func=estimator_func,
                    dgp_kwargs=dgp_kwargs,
                    estimator_kwargs=estimator_kwargs,
                    target_period=target_period,
                ): rep
                for rep in range(n_reps)
            }
            
            completed = 0
            for future in as_completed(futures):
                rep_id, result = future.result()
                results_list.append(result)
                completed += 1
                
                if verbose and completed % 100 == 0:
                    print(f"  Completed {completed}/{n_reps} replications")
    
    except Exception as e:
        if verbose:
            print(f"Parallel execution failed: {e}. Falling back to sequential.")
        return _run_sequential(
            n_reps=n_reps,
            dgp_func=dgp_func,
            estimator_func=estimator_func,
            dgp_kwargs=dgp_kwargs,
            estimator_kwargs=estimator_kwargs,
            target_period=target_period,
            verbose=verbose,
            estimator_name=estimator_name,
        )
    
    return results_list


def _run_sequential(
    n_reps: int,
    dgp_func: Callable,
    estimator_func: Callable,
    dgp_kwargs: Dict[str, Any],
    estimator_kwargs: Dict[str, Any],
    target_period: Optional[int],
    verbose: bool,
    estimator_name: str,
) -> List[Dict[str, Any]]:
    """Run Monte Carlo replications sequentially."""
    results_list = []
    
    if verbose:
        print(f"Running {n_reps} replications for {estimator_name}...")
    
    for rep in range(n_reps):
        _, result = run_single_replication(
            rep_id=rep,
            dgp_func=dgp_func,
            estimator_func=estimator_func,
            dgp_kwargs=dgp_kwargs,
            estimator_kwargs=estimator_kwargs,
            target_period=target_period,
        )
        results_list.append(result)
        
        if verbose and (rep + 1) % 100 == 0:
            print(f"  Completed {rep + 1}/{n_reps} replications")
    
    return results_list


def _aggregate_results(
    results_list: List[Dict[str, Any]],
    n_reps: int,
    n_units: int,
    alpha: float,
    estimator_name: str,
    dgp_name: str,
    scenario: str,
    target_period: Optional[int],
) -> MonteCarloResults:
    """Aggregate Monte Carlo results into summary statistics."""
    # Filter successful results
    valid_results = [r for r in results_list if r.get('success', False)]
    n_valid = len(valid_results)
    n_failed = n_reps - n_valid
    
    if n_valid == 0:
        return MonteCarloResults(
            estimator_name=estimator_name,
            dgp_type=dgp_name,
            scenario=scenario,
            n_reps=n_reps,
            n_units=n_units,
            target_period=target_period,
            n_valid=0,
            n_failed=n_failed,
        )
    
    # Extract arrays
    estimates = np.array([r['att'] for r in valid_results])
    standard_errors = np.array([r['se'] for r in valid_results])
    ci_lowers = np.array([r['ci_lower'] for r in valid_results])
    ci_uppers = np.array([r['ci_upper'] for r in valid_results])
    true_values = np.array([r['true_att'] for r in valid_results])
    p_values = np.array([r.get('pvalue', np.nan) for r in valid_results])
    
    # Compute true ATT (use first valid value or mean)
    true_att = float(np.nanmean(true_values))
    
    # Core metrics
    mean_att = float(np.nanmean(estimates))
    median_att = float(np.nanmedian(estimates))
    bias = mean_att - true_att
    sd = float(np.nanstd(estimates, ddof=1))
    rmse = float(np.sqrt(bias**2 + sd**2))
    
    # Coverage: proportion of CIs containing true ATT
    valid_ci_mask = ~np.isnan(ci_lowers) & ~np.isnan(ci_uppers)
    if valid_ci_mask.sum() > 0:
        covers = (ci_lowers[valid_ci_mask] <= true_att) & (true_att <= ci_uppers[valid_ci_mask])
        coverage = float(np.mean(covers))
    else:
        coverage = np.nan
    
    # SE metrics
    valid_se_mask = ~np.isnan(standard_errors)
    if valid_se_mask.sum() > 0:
        mean_se = float(np.nanmean(standard_errors))
        se_ratio = mean_se / sd if sd > 0 else np.nan
    else:
        mean_se = np.nan
        se_ratio = np.nan
    
    # Rejection rate (H0: ATT = 0)
    valid_pvalue_mask = ~np.isnan(p_values)
    if valid_pvalue_mask.sum() > 0:
        rejection_rate = float(np.mean(p_values[valid_pvalue_mask] < alpha))
    else:
        rejection_rate = np.nan
    
    return MonteCarloResults(
        estimator_name=estimator_name,
        dgp_type=dgp_name,
        scenario=scenario,
        n_reps=n_reps,
        n_units=n_units,
        target_period=target_period,
        bias=bias,
        sd=sd,
        rmse=rmse,
        coverage=coverage,
        mean_se=mean_se,
        se_ratio=se_ratio,
        rejection_rate=rejection_rate,
        mean_att=mean_att,
        true_att=true_att,
        median_att=median_att,
        n_valid=n_valid,
        n_failed=n_failed,
        estimates=estimates,
        standard_errors=standard_errors,
        ci_lowers=ci_lowers,
        ci_uppers=ci_uppers,
        true_values=true_values,
        p_values=p_values,
    )


__all__ = [
    'run_single_replication',
    'run_monte_carlo',
    'MonteCarloResults',
]

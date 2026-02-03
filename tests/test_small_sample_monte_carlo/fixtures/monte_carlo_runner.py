# -*- coding: utf-8 -*-
"""
Monte Carlo simulation runner for small-sample validation.

Based on Lee & Wooldridge (2026) ssrn-5325686, Section 5.

This module provides functions to run Monte Carlo simulations
comparing demeaning vs detrending estimators with OLS vs HC3 standard errors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy import stats

import sys
from pathlib import Path

# Add parent fixtures path
parent_fixtures = Path(__file__).parent.parent.parent / 'test_common_timing' / 'fixtures'
sys.path.insert(0, str(parent_fixtures))

from dgp_small_sample import (
    generate_small_sample_dgp,
    generate_small_sample_dgp_from_scenario,
    MonteCarloResult,
    SMALL_SAMPLE_SCENARIOS,
    DEFAULT_SMALL_SAMPLE_PARAMS,
)


# =============================================================================
# Estimation Helper Functions
# =============================================================================

def _estimate_with_lwdid(
    data: pd.DataFrame,
    rolling: str = 'demean',
    vce: str = 'ols',
) -> Dict[str, float]:
    """
    Estimate ATT using lwdid package.
    
    Parameters
    ----------
    data : pd.DataFrame
        Panel data with columns: id, year, y, d, post
    rolling : str
        Transformation method: 'demean' or 'detrend'
    vce : str
        Variance-covariance estimator: 'ols' or 'hc3'
    
    Returns
    -------
    dict
        Contains: att, se, t_stat, pvalue, df, ci_lower, ci_upper
    """
    try:
        from lwdid import lwdid
        import warnings
        
        # Suppress warnings about single treated unit during Monte Carlo
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = lwdid(
                data=data,
                y='y',
                d='d',
                ivar='id',
                tvar='year',
                post='post',  # Required for common timing mode
                rolling=rolling,
                vce=vce if vce != 'ols' else None,
            )
        
        return {
            'att': result.att,
            'se': result.se_att,  # Correct attribute name
            't_stat': result.t_stat if hasattr(result, 't_stat') else result.att / result.se_att,
            'pvalue': result.pvalue,
            'df': result.df_inference if hasattr(result, 'df_inference') else None,
            'ci_lower': result.ci_lower if hasattr(result, 'ci_lower') else None,
            'ci_upper': result.ci_upper if hasattr(result, 'ci_upper') else None,
        }
    except ImportError:
        raise ImportError("lwdid package not found. Please install it first.")
    except Exception as e:
        raise RuntimeError(f"Estimation failed: {e}")


def _estimate_manual_demean(
    data: pd.DataFrame,
    treatment_start: int = 11,
) -> Dict[str, float]:
    """
    Manual demeaning estimation matching Stata lwdid.
    
    Demeaning transformation:
    ẏ_it = Y_it - Ȳ_i,pre
    
    Stata lwdid procedure:
    1. Compute demeaned outcomes for all periods
    2. Compute post-treatment average of demeaned outcomes per unit: ẏ_i,post_avg
    3. Run cross-sectional OLS: ẏ_i,post_avg = α + τ × D_i + ε_i
    
    This matches Stata's "Number of obs = 20" (unit-level regression).
    """
    import statsmodels.api as sm
    
    # Compute pre-treatment means for each unit
    pre_data = data[data['year'] < treatment_start]
    pre_means = pre_data.groupby('id')['y'].mean()
    
    # Apply demeaning transformation to all periods
    data = data.copy()
    data['y_demean'] = data.apply(
        lambda row: row['y'] - pre_means.get(row['id'], 0), axis=1
    )
    
    # Post-treatment data only
    post_data = data[data['post'] == 1].copy()
    
    # Compute post-treatment average of demeaned outcomes per unit
    # This is what Stata lwdid does: ydot_postavg
    unit_post_avg = post_data.groupby('id').agg({
        'y_demean': 'mean',
        'd': 'first',  # Treatment is time-invariant
    }).reset_index()
    unit_post_avg.columns = ['id', 'ydot_postavg', 'd_']
    
    n_units = len(unit_post_avg)
    if n_units < 3:
        return {
            'att': np.nan,
            'se_ols': np.nan,
            'se_hc3': np.nan,
            'df': 0,
            'n_treated': 0,
            'n_control': 0,
        }
    
    # Cross-sectional OLS regression: ẏ_i,post_avg = α + τ × D_i + ε_i
    X = sm.add_constant(unit_post_avg['d_'].values)
    y = unit_post_avg['ydot_postavg'].values
    
    try:
        # OLS with homoskedastic SE
        model_ols = sm.OLS(y, X).fit()
        att = model_ols.params[1]  # Coefficient on D
        se_ols = model_ols.bse[1]
        
        # OLS with HC3 SE
        model_hc3 = sm.OLS(y, X).fit(cov_type='HC3')
        se_hc3 = model_hc3.bse[1]
        
        df = int(model_ols.df_resid)  # N - 2
    except Exception:
        att = np.nan
        se_ols = np.nan
        se_hc3 = np.nan
        df = 0
    
    n_treated = int((unit_post_avg['d_'] == 1).sum())
    n_control = int((unit_post_avg['d_'] == 0).sum())
    
    return {
        'att': att,
        'se_ols': se_ols,
        'se_hc3': se_hc3,
        'df': df,
        'n_treated': n_treated,
        'n_control': n_control,
    }


def _estimate_manual_detrend(
    data: pd.DataFrame,
    treatment_start: int = 11,
) -> Dict[str, float]:
    """
    Manual detrending estimation matching Stata lwdid.
    
    Detrending transformation:
    Ÿ_it = Y_it - (α̂_i + β̂_i × t)
    
    Where (α̂_i, β̂_i) are from OLS: Y_is on 1, s for pre-treatment periods.
    
    Stata lwdid procedure:
    1. Compute detrended outcomes for all periods
    2. Compute post-treatment average of detrended outcomes per unit: Ÿ_i,post_avg
    3. Run cross-sectional OLS: Ÿ_i,post_avg = α + τ × D_i + ε_i
    
    This matches Stata's "Number of obs = 20" (unit-level regression).
    """
    import statsmodels.api as sm
    
    data = data.copy()
    
    # Fit unit-specific linear trends using pre-treatment data
    unit_trends = {}
    for unit_id in data['id'].unique():
        unit_data = data[data['id'] == unit_id]
        pre_data = unit_data[unit_data['year'] < treatment_start]
        
        if len(pre_data) < 2:
            # Not enough data for trend estimation
            unit_trends[unit_id] = (0, 0)
            continue
        
        X = sm.add_constant(pre_data['year'].values)
        y = pre_data['y'].values
        
        try:
            model = sm.OLS(y, X).fit()
            alpha_hat, beta_hat = model.params
            unit_trends[unit_id] = (alpha_hat, beta_hat)
        except Exception:
            unit_trends[unit_id] = (0, 0)
    
    # Apply detrending transformation to all periods
    def detrend_outcome(row):
        alpha, beta = unit_trends.get(row['id'], (0, 0))
        return row['y'] - (alpha + beta * row['year'])
    
    data['y_detrend'] = data.apply(detrend_outcome, axis=1)
    
    # Post-treatment data only
    post_data = data[data['post'] == 1].copy()
    
    # Compute post-treatment average of detrended outcomes per unit
    # This is what Stata lwdid does: ydot_postavg
    unit_post_avg = post_data.groupby('id').agg({
        'y_detrend': 'mean',
        'd': 'first',  # Treatment is time-invariant
    }).reset_index()
    unit_post_avg.columns = ['id', 'ydot_postavg', 'd_']
    
    n_units = len(unit_post_avg)
    if n_units < 3:
        return {
            'att': np.nan,
            'se_ols': np.nan,
            'se_hc3': np.nan,
            'df': 0,
            'n_treated': 0,
            'n_control': 0,
        }
    
    # Cross-sectional OLS regression: Ÿ_i,post_avg = α + τ × D_i + ε_i
    X = sm.add_constant(unit_post_avg['d_'].values)
    y = unit_post_avg['ydot_postavg'].values
    
    try:
        # OLS with homoskedastic SE
        model_ols = sm.OLS(y, X).fit()
        att = model_ols.params[1]  # Coefficient on D
        se_ols = model_ols.bse[1]
        
        # OLS with HC3 SE
        model_hc3 = sm.OLS(y, X).fit(cov_type='HC3')
        se_hc3 = model_hc3.bse[1]
        
        df = int(model_ols.df_resid)  # N - 2
    except Exception:
        att = np.nan
        se_ols = np.nan
        se_hc3 = np.nan
        df = 0
    
    n_treated = int((unit_post_avg['d_'] == 1).sum())
    n_control = int((unit_post_avg['d_'] == 0).sum())
    
    return {
        'att': att,
        'se_ols': se_ols,
        'se_hc3': se_hc3,
        'df': df,
        'n_treated': n_treated,
        'n_control': n_control,
    }


# =============================================================================
# Monte Carlo Runner
# =============================================================================

def run_small_sample_monte_carlo(
    n_reps: int = 1000,
    scenario: str = 'scenario_1',
    estimators: List[str] = None,
    alpha: float = 0.05,
    seed: int = 42,
    verbose: bool = True,
    use_lwdid: bool = True,
) -> Dict[str, MonteCarloResult]:
    """
    Run Monte Carlo simulation for small-sample validation.
    
    Parameters
    ----------
    n_reps : int, default=1000
        Number of Monte Carlo replications.
    scenario : str, default='scenario_1'
        Scenario from SMALL_SAMPLE_SCENARIOS.
    estimators : list of str, optional
        Estimators to compare: 'demeaning', 'detrending'.
        Default: ['demeaning', 'detrending']
    alpha : float, default=0.05
        Significance level for coverage calculation.
    seed : int, default=42
        Base random seed.
    verbose : bool, default=True
        Print progress updates.
    use_lwdid : bool, default=True
        Use lwdid package for estimation. If False, use manual implementation.
    
    Returns
    -------
    Dict[str, MonteCarloResult]
        Results keyed by estimator name.
    """
    if estimators is None:
        estimators = ['demeaning', 'detrending']
    
    if scenario not in SMALL_SAMPLE_SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    
    scenario_config = SMALL_SAMPLE_SCENARIOS[scenario]
    scenario_num = scenario_config['scenario']
    
    results = {}
    
    for estimator in estimators:
        if verbose:
            print(f"Running {estimator} for {scenario}...")
        
        att_estimates = []
        se_ols_estimates = []
        se_hc3_estimates = []
        covers_ols = []
        covers_hc3 = []
        
        for rep in range(n_reps):
            # Generate data with unique seed
            rep_seed = seed + rep
            
            try:
                data, params = generate_small_sample_dgp(
                    scenario=scenario_num,
                    seed=rep_seed,
                    **{k: v for k, v in DEFAULT_SMALL_SAMPLE_PARAMS.items() 
                       if k != 'scenario'}
                )
                true_tau = params['tau']
                treatment_start = params['treatment_start']
                
                # Run estimation
                if use_lwdid:
                    # Use lwdid package
                    rolling = 'demean' if estimator == 'demeaning' else 'detrend'
                    
                    try:
                        result_ols = _estimate_with_lwdid(data, rolling=rolling, vce='ols')
                        result_hc3 = _estimate_with_lwdid(data, rolling=rolling, vce='hc3')
                        
                        att = result_ols['att']
                        se_ols = result_ols['se']
                        se_hc3 = result_hc3['se']
                        df = result_ols['df'] or (params['n_treated'] + params['n_control'] - 2)
                    except Exception as e:
                        if verbose and rep < 5:
                            print(f"  Rep {rep} lwdid failed: {e}")
                        continue
                else:
                    # Use manual implementation
                    if estimator == 'demeaning':
                        result = _estimate_manual_demean(data, treatment_start)
                    else:
                        result = _estimate_manual_detrend(data, treatment_start)
                    
                    att = result['att']
                    se_ols = result['se_ols']
                    se_hc3 = result['se_hc3']
                    df = result['df']
                
                if np.isnan(att) or np.isnan(se_ols):
                    continue
                
                att_estimates.append(att)
                se_ols_estimates.append(se_ols)
                se_hc3_estimates.append(se_hc3 if not np.isnan(se_hc3) else se_ols)
                
                # Coverage calculation using t-distribution
                t_crit = stats.t.ppf(1 - alpha/2, df) if df and df > 0 else 1.96
                
                # OLS coverage
                ci_lower_ols = att - t_crit * se_ols
                ci_upper_ols = att + t_crit * se_ols
                covers_ols.append(1 if ci_lower_ols <= true_tau <= ci_upper_ols else 0)
                
                # HC3 coverage
                se_hc3_val = se_hc3 if not np.isnan(se_hc3) else se_ols
                ci_lower_hc3 = att - t_crit * se_hc3_val
                ci_upper_hc3 = att + t_crit * se_hc3_val
                covers_hc3.append(1 if ci_lower_hc3 <= true_tau <= ci_upper_hc3 else 0)
                
            except Exception as e:
                if verbose and rep < 5:
                    print(f"  Rep {rep} failed: {e}")
                continue
            
            if verbose and (rep + 1) % 100 == 0:
                print(f"  Completed {rep + 1}/{n_reps} replications")
        
        # Compute summary statistics
        if len(att_estimates) == 0:
            if verbose:
                print(f"  Warning: All estimations failed for {estimator}")
            continue
        
        att_arr = np.array(att_estimates)
        se_ols_arr = np.array(se_ols_estimates)
        se_hc3_arr = np.array(se_hc3_estimates)
        
        # Get true ATT from first successful run
        _, params = generate_small_sample_dgp(scenario=scenario_num, seed=seed)
        true_tau = params['tau']
        
        mean_att = np.mean(att_arr)
        bias = mean_att - true_tau
        sd = np.std(att_arr, ddof=1)
        rmse = np.sqrt(bias**2 + sd**2)
        
        coverage_ols = np.mean(covers_ols) if covers_ols else 0.0
        coverage_hc3 = np.mean(covers_hc3) if covers_hc3 else 0.0
        
        mean_se_ols = np.mean(se_ols_arr)
        mean_se_hc3 = np.mean(se_hc3_arr)
        
        # SE ratio (mean SE / SD of estimates)
        se_ratio_ols = mean_se_ols / sd if sd > 0 else np.nan
        se_ratio_hc3 = mean_se_hc3 / sd if sd > 0 else np.nan
        
        results[estimator] = MonteCarloResult(
            estimator_name=estimator,
            scenario=scenario,
            n_reps=n_reps,
            n_successful=len(att_estimates),
            true_att=true_tau,
            mean_att=mean_att,
            bias=bias,
            sd=sd,
            rmse=rmse,
            coverage_ols=coverage_ols,
            coverage_hc3=coverage_hc3,
            mean_se_ols=mean_se_ols,
            mean_se_hc3=mean_se_hc3,
            se_ratio_ols=se_ratio_ols,
            se_ratio_hc3=se_ratio_hc3,
            att_estimates=att_estimates,
            se_ols_estimates=se_ols_estimates,
            se_hc3_estimates=se_hc3_estimates,
        )
        
        if verbose:
            print(f"  {estimator}: Bias={bias:.4f}, SD={sd:.4f}, RMSE={rmse:.4f}, "
                  f"Coverage(OLS)={coverage_ols:.2%}, Coverage(HC3)={coverage_hc3:.2%}")
    
    return results


def run_all_scenarios_monte_carlo(
    n_reps: int = 500,
    estimators: List[str] = None,
    seed: int = 42,
    verbose: bool = True,
    use_lwdid: bool = True,
) -> Dict[str, Dict[str, MonteCarloResult]]:
    """
    Run Monte Carlo simulation for all scenarios.
    
    Returns
    -------
    Dict[str, Dict[str, MonteCarloResult]]
        Nested dict: results[scenario][estimator] = MonteCarloResult
    """
    if estimators is None:
        estimators = ['demeaning', 'detrending']
    
    all_results = {}
    
    for scenario in SMALL_SAMPLE_SCENARIOS.keys():
        if verbose:
            print(f"\n{'='*60}")
            print(f"Scenario: {scenario}")
            print(f"{'='*60}")
        
        all_results[scenario] = run_small_sample_monte_carlo(
            n_reps=n_reps,
            scenario=scenario,
            estimators=estimators,
            seed=seed,
            verbose=verbose,
            use_lwdid=use_lwdid,
        )
    
    return all_results


def generate_comparison_table(
    results: Dict[str, Dict[str, MonteCarloResult]],
) -> pd.DataFrame:
    """
    Generate comparison table similar to paper Table 2.
    
    Parameters
    ----------
    results : Dict[str, Dict[str, MonteCarloResult]]
        Results from run_all_scenarios_monte_carlo.
    
    Returns
    -------
    pd.DataFrame
        Comparison table with columns: Scenario, Estimator, Bias, SD, RMSE, 
        Coverage(OLS), Coverage(HC3), SE Ratio(OLS), SE Ratio(HC3)
    """
    rows = []
    
    for scenario, scenario_results in results.items():
        for estimator, result in scenario_results.items():
            rows.append({
                'Scenario': scenario,
                'Estimator': estimator,
                'N_reps': result.n_successful,
                'True_ATT': round(result.true_att, 3),
                'Mean_ATT': round(result.mean_att, 3),
                'Bias': round(result.bias, 3),
                'SD': round(result.sd, 3),
                'RMSE': round(result.rmse, 3),
                'Coverage_OLS': round(result.coverage_ols, 3),
                'Coverage_HC3': round(result.coverage_hc3, 3),
                'SE_Ratio_OLS': round(result.se_ratio_ols, 3) if not np.isnan(result.se_ratio_ols) else np.nan,
                'SE_Ratio_HC3': round(result.se_ratio_hc3, 3) if not np.isnan(result.se_ratio_hc3) else np.nan,
            })
    
    return pd.DataFrame(rows)


__all__ = [
    'run_small_sample_monte_carlo',
    'run_all_scenarios_monte_carlo',
    'generate_comparison_table',
    '_estimate_manual_demean',
    '_estimate_manual_detrend',
]

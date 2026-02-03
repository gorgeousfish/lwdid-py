# -*- coding: utf-8 -*-
"""
Small-sample DGP for Monte Carlo validation.

Based on Lee & Wooldridge (2026) ssrn-5325686, Section 5.

This DGP is designed for validating LWDID methods in small-sample settings,
particularly comparing demeaning vs detrending and OLS vs HC3 standard errors.

Key features from paper Section 5:
1. N=20 units, T=20 periods (10 pre + 10 post)
2. Treatment starts at t=11
3. Unit fixed effects C_i ~ N(0, σ_C²), σ_C = 2
4. Unit-specific trends G_i ~ N(1, σ_G²), σ_G = 1
5. AR(1) error process with ρ = 0.75
6. Treatment assignment via latent index model with Logistic errors
7. Time-varying treatment effects δ_t (Table 1)
8. Time-varying period effects λ_t (Table 1)

References
----------
Lee S.J. & Wooldridge J.M. (2026). "Difference-in-Differences with a 
Single Treated Unit." SSRN 5325686, Section 5, Table 1-2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Paper Table 1 Parameters
# =============================================================================

# Time period effects λ_t (Table 1, row "λ_t")
# Periods 1-20, index 0-19
LAMBDA_T_TABLE1 = [
    0.0, 0.0, 0.0, 0.0, 0.2, 0.6, 0.7, 0.8, 0.6, 0.9,  # t=1-10 (pre)
    0.9, 1.0, 1.1, 1.3, 1.2, 1.5, 0.6, 1.4, 1.8, 1.9   # t=11-20 (post)
]

# Treatment effects δ_t (Table 1, row "δ_t")
# Periods 1-20, index 0-19
# Pre-treatment (t=1-10): δ_t = 0
# Post-treatment (t=11-20): δ_t varies
DELTA_T_TABLE1 = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # t=1-10 (pre): no treatment effect
    1, 2, 3, 3, 3, 2, 2, 2, 1, 1   # t=11-20 (post): time-varying effects
]

# Treatment rule parameters (Table 1)
# Scenario: (α₀, α₁, α₂) -> P(D=1)
TREATMENT_RULE_PARAMS = {
    1: {'alpha_0': -1.0, 'alpha_1': -1/3, 'alpha_2': 1/4, 'expected_prob': 0.32},
    2: {'alpha_0': -1.5, 'alpha_1': 1/3, 'alpha_2': 1/4, 'expected_prob': 0.24},
    3: {'alpha_0': -2.0, 'alpha_1': 1/3, 'alpha_2': 1/4, 'expected_prob': 0.17},
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SmallSampleDGPParams:
    """
    Parameters for small-sample DGP from ssrn-5325686 Section 5.
    
    Attributes
    ----------
    n_units : int
        Number of cross-sectional units (default: 20).
    n_periods : int
        Number of time periods (default: 20).
    treatment_start : int
        First treatment period (default: 11).
    scenario : int
        Scenario number (1, 2, or 3) controlling treatment probability.
    sigma_c : float
        Standard deviation of unit fixed effects (default: 2.0).
    sigma_g : float
        Standard deviation of unit-specific trends (default: 1.0).
    rho : float
        AR(1) coefficient for error process (default: 0.75).
    sigma_epsilon : float
        Standard deviation of AR(1) innovations (default: sqrt(2)).
    sigma_nu : float
        Standard deviation of treatment effect heterogeneity (default: sqrt(2)).
    seed : Optional[int]
        Random seed for reproducibility.
    """
    n_units: int = 20
    n_periods: int = 20
    treatment_start: int = 11
    scenario: int = 1
    sigma_c: float = 2.0
    sigma_g: float = 1.0
    rho: float = 0.75
    sigma_epsilon: float = field(default_factory=lambda: np.sqrt(2))
    sigma_nu: float = field(default_factory=lambda: np.sqrt(2))
    seed: Optional[int] = None

    def __post_init__(self):
        """Validate parameters."""
        if self.n_units < 3:
            raise ValueError("n_units must be at least 3")
        if self.n_periods < 3:
            raise ValueError("n_periods must be at least 3")
        if not 1 < self.treatment_start <= self.n_periods:
            raise ValueError(
                f"treatment_start must be in (1, n_periods], "
                f"got {self.treatment_start} with n_periods={self.n_periods}"
            )
        if self.scenario not in [1, 2, 3]:
            raise ValueError(f"scenario must be 1, 2, or 3, got {self.scenario}")
        if self.sigma_c < 0:
            raise ValueError(f"sigma_c must be non-negative, got {self.sigma_c}")
        if self.sigma_g < 0:
            raise ValueError(f"sigma_g must be non-negative, got {self.sigma_g}")
        if not -1 < self.rho < 1:
            raise ValueError(f"rho must be in (-1, 1), got {self.rho}")
        if self.sigma_epsilon < 0:
            raise ValueError(f"sigma_epsilon must be non-negative")
        if self.sigma_nu < 0:
            raise ValueError(f"sigma_nu must be non-negative")


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation."""
    estimator_name: str
    scenario: str
    n_reps: int
    n_successful: int
    true_att: float
    mean_att: float
    bias: float
    sd: float
    rmse: float
    coverage_ols: float = 0.0
    coverage_hc3: float = 0.0
    mean_se_ols: float = 0.0
    mean_se_hc3: float = 0.0
    se_ratio_ols: float = 0.0
    se_ratio_hc3: float = 0.0
    att_estimates: List[float] = field(default_factory=list)
    se_ols_estimates: List[float] = field(default_factory=list)
    se_hc3_estimates: List[float] = field(default_factory=list)
    
    def summary(self) -> str:
        """Generate summary string."""
        return (
            f"Monte Carlo Results: {self.estimator_name} ({self.scenario})\n"
            f"  Replications: {self.n_successful}/{self.n_reps}\n"
            f"  True ATT: {self.true_att:.4f}\n"
            f"  Mean ATT: {self.mean_att:.4f}\n"
            f"  Bias: {self.bias:.4f}\n"
            f"  SD: {self.sd:.4f}\n"
            f"  RMSE: {self.rmse:.4f}\n"
            f"  Coverage (OLS): {self.coverage_ols:.2%}\n"
            f"  Coverage (HC3): {self.coverage_hc3:.2%}\n"
            f"  SE Ratio (OLS): {self.se_ratio_ols:.4f}\n"
            f"  SE Ratio (HC3): {self.se_ratio_hc3:.4f}"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for tabulation."""
        return {
            'estimator': self.estimator_name,
            'scenario': self.scenario,
            'n_reps': self.n_reps,
            'n_successful': self.n_successful,
            'true_att': round(self.true_att, 4),
            'mean_att': round(self.mean_att, 4),
            'bias': round(self.bias, 4),
            'sd': round(self.sd, 4),
            'rmse': round(self.rmse, 4),
            'coverage_ols': round(self.coverage_ols, 4),
            'coverage_hc3': round(self.coverage_hc3, 4),
            'mean_se_ols': round(self.mean_se_ols, 4),
            'mean_se_hc3': round(self.mean_se_hc3, 4),
            'se_ratio_ols': round(self.se_ratio_ols, 4),
            'se_ratio_hc3': round(self.se_ratio_hc3, 4),
        }


# =============================================================================
# Scenario Configurations
# =============================================================================

SMALL_SAMPLE_SCENARIOS = {
    'scenario_1': {
        'scenario': 1,
        'prob_treated': 0.32,
        'description': 'High treatment probability (P(D=1) ≈ 0.32)',
        'alpha_0': -1.0,
        'alpha_1': -1/3,
        'alpha_2': 1/4,
    },
    'scenario_2': {
        'scenario': 2,
        'prob_treated': 0.24,
        'description': 'Medium treatment probability (P(D=1) ≈ 0.24)',
        'alpha_0': -1.5,
        'alpha_1': 1/3,
        'alpha_2': 1/4,
    },
    'scenario_3': {
        'scenario': 3,
        'prob_treated': 0.17,
        'description': 'Low treatment probability (P(D=1) ≈ 0.17)',
        'alpha_0': -2.0,
        'alpha_1': 1/3,
        'alpha_2': 1/4,
    },
}

DEFAULT_SMALL_SAMPLE_PARAMS = {
    'n_units': 20,
    'n_periods': 20,
    'treatment_start': 11,
    'sigma_c': 2.0,
    'sigma_g': 1.0,
    'rho': 0.75,
}


# =============================================================================
# Core DGP Function
# =============================================================================

def generate_small_sample_dgp(
    n_units: int = 20,
    n_periods: int = 20,
    treatment_start: int = 11,
    scenario: int = 1,
    sigma_c: float = 2.0,
    sigma_g: float = 1.0,
    rho: float = 0.75,
    sigma_epsilon: float | None = None,
    sigma_nu: float | None = None,
    seed: int | None = None,
    return_components: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generate small-sample DGP from ssrn-5325686 Section 5.
    
    DGP Structure (论文 Section 5.1):
    
    Step 1: Generate unit characteristics
        C_i ~ N(0, σ_C²), σ_C = 2  (unit fixed effects)
        G_i ~ N(1, σ_G²), σ_G = 1  (unit-specific trend slopes, mean=1!)
    
    Step 2: Generate AR(1) error process
        u_i1 ~ N(0, √(2/(1-ρ²)))  (stationary initial distribution)
        u_it = ρ·u_{i,t-1} + ε_it, ε_it ~ N(0, σ_ε), σ_ε = √2
    
    Step 3: Generate treatment assignment
        D_i = I(α₀ - α₁·C_i + α₂·G_i + e_i > 0)
        e_i ~ Logistic(0, 1)
        
        Scenario 1: (α₀, α₁, α₂) = (-1, -1/3, 1/4) → P(D=1) ≈ 0.32
        Scenario 2: (α₀, α₁, α₂) = (-1.5, 1/3, 1/4) → P(D=1) ≈ 0.24
        Scenario 3: (α₀, α₁, α₂) = (-2, 1/3, 1/4) → P(D=1) ≈ 0.17
    
    Step 4: Generate potential outcomes
        Y_it(0) = λ_t - C_i + G_i·t + u_it
        Y_it(1) = Y_it(0) + δ_t + ν_it, ν_it ~ N(0, σ_ν), σ_ν = √2
    
    Step 5: Generate observed outcomes
        Y_it = D_i·post_t·Y_it(1) + (1 - D_i·post_t)·Y_it(0)
    
    Parameters
    ----------
    n_units : int, default=20
        Number of cross-sectional units.
    n_periods : int, default=20
        Number of time periods.
    treatment_start : int, default=11
        First treatment period.
    scenario : int, default=1
        Scenario number (1, 2, or 3) controlling treatment probability.
    sigma_c : float, default=2.0
        Standard deviation of unit fixed effects.
    sigma_g : float, default=1.0
        Standard deviation of unit-specific trends.
    rho : float, default=0.75
        AR(1) coefficient for error process.
    sigma_epsilon : float, optional
        Standard deviation of AR(1) innovations (default: sqrt(2)).
    sigma_nu : float, optional
        Standard deviation of treatment effect heterogeneity (default: sqrt(2)).
    seed : int, optional
        Random seed for reproducibility.
    return_components : bool, default=False
        If True, return individual DGP components.
    
    Returns
    -------
    data : pd.DataFrame
        Panel data with columns: id, year, y, d, post
    params : Dict[str, Any]
        DGP parameters and true treatment effects.
    """
    # Set defaults for sigma parameters
    if sigma_epsilon is None:
        sigma_epsilon = np.sqrt(2)
    if sigma_nu is None:
        sigma_nu = np.sqrt(2)

    # Validate parameters
    _ = SmallSampleDGPParams(
        n_units=n_units,
        n_periods=n_periods,
        treatment_start=treatment_start,
        scenario=scenario,
        sigma_c=sigma_c,
        sigma_g=sigma_g,
        rho=rho,
        sigma_epsilon=sigma_epsilon,
        sigma_nu=sigma_nu,
        seed=seed,
    )
    
    rng = np.random.default_rng(seed)
    N = n_units
    T = n_periods
    S = treatment_start
    
    # Get treatment rule parameters for this scenario
    rule_params = TREATMENT_RULE_PARAMS[scenario]
    alpha_0 = rule_params['alpha_0']
    alpha_1 = rule_params['alpha_1']
    alpha_2 = rule_params['alpha_2']
    
    # =========================================================================
    # Step 1: Generate unit characteristics (time-invariant)
    # C_i ~ N(0, σ_C²): Unit fixed effects
    # G_i ~ N(1, σ_G²): Unit-specific trend slopes (mean=1, NOT 0!)
    # =========================================================================
    C_i = rng.normal(0, sigma_c, N)
    G_i = rng.normal(1, sigma_g, N)
    
    # =========================================================================
    # Step 2: Generate AR(1) error process
    # u_i1 ~ N(0, √(σ_ε²/(1-ρ²))) for stationarity
    # u_it = ρ·u_{i,t-1} + ε_it, ε_it ~ N(0, σ_ε)
    # =========================================================================
    # Stationary variance: Var(u) = σ_ε² / (1 - ρ²)
    sigma_u_stationary = sigma_epsilon / np.sqrt(1 - rho**2)
    
    U = np.zeros((N, T))
    U[:, 0] = rng.normal(0, sigma_u_stationary, N)
    
    for t in range(1, T):
        epsilon_t = rng.normal(0, sigma_epsilon, N)
        U[:, t] = rho * U[:, t-1] + epsilon_t
    
    # =========================================================================
    # Step 3: Generate treatment assignment
    # D_i = I(α₀ - α₁·C_i + α₂·G_i + e_i > 0)
    # e_i ~ Logistic(0, 1)
    # =========================================================================
    logistic_error = rng.logistic(0, 1, N)
    latent_index = alpha_0 - alpha_1 * C_i + alpha_2 * G_i + logistic_error
    D_i = (latent_index > 0).astype(int)
    
    # Ensure at least 1 treated and 1 control unit
    n_treated = int(D_i.sum())
    n_control = N - n_treated
    
    if n_treated == 0:
        # Force at least one treated unit (pick unit with highest latent index)
        idx = np.argmax(latent_index)
        D_i[idx] = 1
        n_treated = 1
        n_control = N - 1
    elif n_control == 0:
        # Force at least one control unit (pick unit with lowest latent index)
        idx = np.argmin(latent_index)
        D_i[idx] = 0
        n_treated = N - 1
        n_control = 1

    # =========================================================================
    # Step 4 & 5: Generate potential outcomes and observed outcomes
    # Y_it(0) = λ_t - C_i + G_i·t + u_it
    # Y_it(1) = Y_it(0) + δ_t + ν_it
    # =========================================================================
    # Get period effects from Table 1, extending if T > 20
    if T <= 20:
        lambda_t = LAMBDA_T_TABLE1[:T]
        delta_t = DELTA_T_TABLE1[:T]
    else:
        # Extend with last values for T > 20
        lambda_t = LAMBDA_T_TABLE1 + [LAMBDA_T_TABLE1[-1]] * (T - 20)
        # For delta_t, extend post-treatment effects
        delta_t = DELTA_T_TABLE1 + [DELTA_T_TABLE1[-1]] * (T - 20)
    
    records = []
    
    for i in range(N):
        for t in range(T):
            period = t + 1  # 1-indexed period
            
            # Post-treatment indicator
            post = 1 if period >= S else 0
            
            # Potential outcome Y(0)
            # Note: Paper uses -C_i (negative sign on fixed effect)
            y0 = lambda_t[t] - C_i[i] + G_i[i] * period + U[i, t]
            
            # Treatment effect heterogeneity
            nu_it = rng.normal(0, sigma_nu) if (post == 1 and D_i[i] == 1) else 0
            
            # Observed outcome
            if post == 1 and D_i[i] == 1:
                # Treated in post-treatment period
                y = y0 + delta_t[t] + nu_it
            else:
                y = y0
            
            records.append({
                'id': i + 1,
                'year': period,
                'y': y,
                'd': int(D_i[i]),
                'post': post,
            })
    
    data = pd.DataFrame(records)
    
    # =========================================================================
    # Step 6: Compute true treatment effects
    # =========================================================================
    # Average treatment effect on treated (ATT) for post-treatment periods
    post_periods = list(range(S, T + 1))
    true_att_by_period = {t: delta_t[t-1] for t in post_periods}
    
    # Overall average effect (average of δ_t for post-treatment periods)
    average_effect = np.mean([delta_t[t-1] for t in post_periods])
    
    # Sample ATT (based on actual treated units)
    # In this DGP, treatment effect is δ_t + ν_it, so sample ATT ≈ average δ_t
    sample_att = average_effect
    
    # =========================================================================
    # Step 7: Prepare return values
    # =========================================================================
    result_params: Dict[str, Any] = {
        'tau': average_effect,
        'true_att': average_effect,
        'att_by_period': true_att_by_period,
        'n_treated': n_treated,
        'n_control': n_control,
        'n_units': N,
        'n_periods': T,
        'treatment_start': S,
        'scenario': scenario,
        'alpha_0': alpha_0,
        'alpha_1': alpha_1,
        'alpha_2': alpha_2,
        'sigma_c': sigma_c,
        'sigma_g': sigma_g,
        'rho': rho,
        'sigma_epsilon': sigma_epsilon,
        'sigma_nu': sigma_nu,
        'seed': seed,
        'n_pre_periods': S - 1,
        'n_post_periods': T - S + 1,
        'treated_share': n_treated / N,
        'lambda_t': lambda_t,
        'delta_t': delta_t,
    }
    
    if return_components:
        result_params['C'] = C_i.copy()
        result_params['G'] = G_i.copy()
        result_params['D'] = D_i.copy()
        result_params['U'] = U.copy()
    
    return data, result_params


def generate_small_sample_dgp_from_scenario(
    scenario: str,
    seed: int | None = None,
    return_components: bool = False,
    **kwargs,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generate small-sample DGP using predefined scenario configuration.
    
    Parameters
    ----------
    scenario : str
        Scenario name: 'scenario_1', 'scenario_2', or 'scenario_3'.
    seed : int, optional
        Random seed for reproducibility.
    return_components : bool, default=False
        If True, return individual DGP components.
    **kwargs
        Additional parameters to override defaults.
    
    Returns
    -------
    data : pd.DataFrame
        Panel data.
    params : Dict[str, Any]
        DGP parameters including scenario info.
    """
    if scenario not in SMALL_SAMPLE_SCENARIOS:
        raise ValueError(
            f"Unknown scenario: {scenario}. "
            f"Available: {list(SMALL_SAMPLE_SCENARIOS.keys())}"
        )
    
    scenario_config = SMALL_SAMPLE_SCENARIOS[scenario]
    scenario_num = scenario_config['scenario']
    
    # Merge default params with user overrides
    dgp_params = {
        **DEFAULT_SMALL_SAMPLE_PARAMS,
        'scenario': scenario_num,
        'seed': seed,
        'return_components': return_components,
        **kwargs,
    }
    
    data, params = generate_small_sample_dgp(**dgp_params)
    
    # Add scenario info to params
    params['scenario_name'] = scenario
    params['scenario_description'] = scenario_config['description']
    
    return data, params


# =============================================================================
# Utility Functions
# =============================================================================

def compute_theoretical_variance_components(
    sigma_c: float = 2.0,
    sigma_g: float = 1.0,
    sigma_epsilon: float | None = None,
    rho: float = 0.75,
    n_periods: int = 20,
) -> Dict[str, float]:
    """
    Compute theoretical variance components of the DGP.
    
    For Y_it = λ_t - C_i + G_i × t + u_it (ignoring treatment):
    
    Var(Y_it) = σ_C² + t² × σ_G² + σ_u²
    where σ_u² = σ_ε² / (1 - ρ²) (stationary AR(1) variance)
    """
    if sigma_epsilon is None:
        sigma_epsilon = np.sqrt(2)
    
    var_c = sigma_c ** 2
    var_g = sigma_g ** 2
    var_u = sigma_epsilon ** 2 / (1 - rho ** 2)
    
    # Variance at different time points
    var_t1 = var_c + 1 * var_g + var_u
    var_t_mid = var_c + (n_periods // 2) ** 2 * var_g + var_u
    var_t_end = var_c + n_periods ** 2 * var_g + var_u
    
    return {
        'var_c': var_c,
        'var_g': var_g,
        'var_u': var_u,
        'var_epsilon': sigma_epsilon ** 2,
        'var_y_t1': var_t1,
        'var_y_t_mid': var_t_mid,
        'var_y_t_end': var_t_end,
        'mean_g': 1.0,  # G_i ~ N(1, σ_G²)
        'rho': rho,
    }


def validate_dgp_output(
    data: pd.DataFrame,
    params: Dict[str, Any],
) -> Dict[str, bool]:
    """Validate DGP output for correctness."""
    checks = {}
    
    # Check data structure
    required_cols = {'id', 'year', 'y', 'd', 'post'}
    checks['has_required_columns'] = required_cols.issubset(set(data.columns))
    
    # Check panel dimensions
    n_units = params['n_units']
    n_periods = params['n_periods']
    checks['correct_n_rows'] = len(data) == n_units * n_periods
    checks['correct_n_units'] = data['id'].nunique() == n_units
    checks['correct_n_periods'] = data['year'].nunique() == n_periods
    
    # Check balanced panel
    obs_per_unit = data.groupby('id').size()
    checks['balanced_panel'] = obs_per_unit.nunique() == 1
    
    # Check treatment is time-constant
    d_by_unit = data.groupby('id')['d'].nunique()
    checks['treatment_time_constant'] = (d_by_unit == 1).all()
    
    # Check post indicator
    treatment_start = params['treatment_start']
    pre_data = data[data['year'] < treatment_start]
    post_data = data[data['year'] >= treatment_start]
    checks['post_indicator_correct'] = (
        (pre_data['post'] == 0).all() and 
        (post_data['post'] == 1).all()
    )
    
    # Check at least 1 treated and 1 control
    n_treated = params['n_treated']
    n_control = params['n_control']
    checks['has_treated'] = n_treated >= 1
    checks['has_control'] = n_control >= 1
    checks['n_units_match'] = n_treated + n_control == n_units
    
    return checks


def compute_expected_treatment_probability(
    scenario: int,
    n_simulations: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Compute expected treatment probability via simulation.
    
    This validates that the treatment rule parameters produce
    the expected treatment probabilities from Table 1.
    """
    rng = np.random.default_rng(seed)
    
    rule_params = TREATMENT_RULE_PARAMS[scenario]
    alpha_0 = rule_params['alpha_0']
    alpha_1 = rule_params['alpha_1']
    alpha_2 = rule_params['alpha_2']
    expected_prob = rule_params['expected_prob']
    
    # Simulate treatment assignment
    C_i = rng.normal(0, 2.0, n_simulations)  # σ_C = 2
    G_i = rng.normal(1, 1.0, n_simulations)  # σ_G = 1, mean = 1
    e_i = rng.logistic(0, 1, n_simulations)
    
    latent = alpha_0 - alpha_1 * C_i + alpha_2 * G_i + e_i
    D_i = (latent > 0).astype(int)
    
    simulated_prob = D_i.mean()
    
    return {
        'scenario': scenario,
        'expected_prob': expected_prob,
        'simulated_prob': simulated_prob,
        'difference': abs(simulated_prob - expected_prob),
        'within_tolerance': abs(simulated_prob - expected_prob) < 0.02,
    }


# =============================================================================
# Paper Reference Values (Table 2)
# =============================================================================

# Reference values from paper Table 2 for validation
# These are the expected Monte Carlo results for detrending estimator
PAPER_TABLE_2_REFERENCE = {
    'scenario_1': {
        'detrending': {
            'bias': 0.009,
            'sd': 1.73,
            'rmse': 1.734,
            'coverage_ols': 0.96,
        },
    },
    'scenario_2': {
        'detrending': {
            'bias': -0.042,
            'sd': 1.89,
            'rmse': 1.892,
            'coverage_ols': 0.95,
        },
    },
    'scenario_3': {
        'detrending': {
            'bias': 0.165,
            'sd': 2.37,
            'rmse': 2.380,
            'coverage_ols': 0.95,
        },
    },
}


__all__ = [
    # Data classes
    'SmallSampleDGPParams',
    'MonteCarloResult',
    # Configurations
    'SMALL_SAMPLE_SCENARIOS',
    'DEFAULT_SMALL_SAMPLE_PARAMS',
    'TREATMENT_RULE_PARAMS',
    'LAMBDA_T_TABLE1',
    'DELTA_T_TABLE1',
    'PAPER_TABLE_2_REFERENCE',
    # Core functions
    'generate_small_sample_dgp',
    'generate_small_sample_dgp_from_scenario',
    # Utilities
    'compute_theoretical_variance_components',
    'validate_dgp_output',
    'compute_expected_treatment_probability',
]

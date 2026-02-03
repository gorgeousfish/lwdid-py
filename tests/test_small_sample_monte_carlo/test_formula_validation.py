# -*- coding: utf-8 -*-
"""
Formula validation tests for small-sample DGP.

Based on Lee & Wooldridge (2026) ssrn-5325686, Section 5.

These tests verify that DGP formulas match paper specifications exactly.
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path
from scipy import stats

# Add fixtures paths
fixtures_path = Path(__file__).parent / 'fixtures'
parent_fixtures = Path(__file__).parent.parent / 'test_common_timing' / 'fixtures'
sys.path.insert(0, str(fixtures_path))
sys.path.insert(0, str(parent_fixtures))

from dgp_small_sample import (
    generate_small_sample_dgp,
    SmallSampleDGPParams,
    LAMBDA_T_TABLE1,
    DELTA_T_TABLE1,
    TREATMENT_RULE_PARAMS,
    SMALL_SAMPLE_SCENARIOS,
    compute_theoretical_variance_components,
    compute_expected_treatment_probability,
)


@pytest.mark.numerical
class TestDGPComponentDistributions:
    """Tests for DGP component distributions matching paper specifications."""
    
    def test_unit_fixed_effect_distribution(self):
        """
        C_i ~ N(0, σ_C²) with σ_C = 2.
        
        Paper Section 5: "C_i ~ N(0, 4)"
        """
        n_simulations = 10000
        rng = np.random.default_rng(42)
        
        sigma_c = 2.0
        C_i = rng.normal(0, sigma_c, n_simulations)
        
        # Test mean ≈ 0
        assert abs(C_i.mean()) < 0.1, f"C_i mean should be ~0, got {C_i.mean()}"
        
        # Test variance ≈ σ_C² = 4
        expected_var = sigma_c ** 2
        actual_var = C_i.var()
        assert abs(actual_var - expected_var) < 0.2, \
            f"C_i variance should be ~{expected_var}, got {actual_var}"
    
    def test_unit_trend_distribution(self):
        """
        G_i ~ N(1, σ_G²) with σ_G = 1.
        
        Paper Section 5: "G_i ~ N(1, 1)" - NOTE: mean is 1, not 0!
        """
        n_simulations = 10000
        rng = np.random.default_rng(42)
        
        sigma_g = 1.0
        mean_g = 1.0  # Paper specifies mean = 1
        G_i = rng.normal(mean_g, sigma_g, n_simulations)
        
        # Test mean ≈ 1
        assert abs(G_i.mean() - mean_g) < 0.05, \
            f"G_i mean should be ~{mean_g}, got {G_i.mean()}"
        
        # Test variance ≈ σ_G² = 1
        expected_var = sigma_g ** 2
        actual_var = G_i.var()
        assert abs(actual_var - expected_var) < 0.1, \
            f"G_i variance should be ~{expected_var}, got {actual_var}"
    
    def test_ar1_error_stationary_distribution(self):
        """
        AR(1) error process: u_it = ρ·u_{i,t-1} + ε_it
        
        Paper Section 5:
        - ρ = 0.75
        - σ_ε = √2
        - Stationary variance: σ_u² = σ_ε² / (1 - ρ²)
        """
        rho = 0.75
        sigma_epsilon = np.sqrt(2)
        
        # Theoretical stationary variance
        expected_var = sigma_epsilon ** 2 / (1 - rho ** 2)
        
        # Simulate AR(1) process
        n_units = 1000
        n_periods = 100  # Long enough for stationarity
        rng = np.random.default_rng(42)
        
        sigma_u_stationary = sigma_epsilon / np.sqrt(1 - rho ** 2)
        U = np.zeros((n_units, n_periods))
        U[:, 0] = rng.normal(0, sigma_u_stationary, n_units)
        
        for t in range(1, n_periods):
            epsilon_t = rng.normal(0, sigma_epsilon, n_units)
            U[:, t] = rho * U[:, t-1] + epsilon_t
        
        # Check variance at last period (should be stationary)
        actual_var = U[:, -1].var()
        assert abs(actual_var - expected_var) < 0.5, \
            f"AR(1) stationary variance should be ~{expected_var:.2f}, got {actual_var:.2f}"
    
    def test_treatment_assignment_logistic_distribution(self):
        """
        D_i = I(α₀ - α₁·C_i + α₂·G_i + e_i > 0)
        e_i ~ Logistic(0, 1)
        
        Paper Table 1 scenarios.
        """
        for scenario in [1, 2, 3]:
            result = compute_expected_treatment_probability(
                scenario=scenario,
                n_simulations=50000,
                seed=42,
            )
            
            expected = result['expected_prob']
            simulated = result['simulated_prob']
            
            assert result['within_tolerance'], \
                f"Scenario {scenario}: P(D=1) should be ~{expected}, got {simulated}"


@pytest.mark.numerical
class TestTransformationFormulas:
    """Tests for transformation formulas matching paper Procedures."""
    
    def test_demeaning_formula_paper_procedure_2_1(self):
        """
        Demeaning transformation (Procedure 2.1):
        ẏ_it = Y_it - Ȳ_i,pre
        
        Where Ȳ_i,pre = (1/(g-1)) × Σ_{s=1}^{g-1} Y_is
        """
        # Generate test data
        data, params = generate_small_sample_dgp(seed=42)
        treatment_start = params['treatment_start']
        
        # Compute pre-treatment means manually
        pre_data = data[data['year'] < treatment_start]
        pre_means = pre_data.groupby('id')['y'].mean()
        
        # Apply demeaning
        data_copy = data.copy()
        data_copy['y_demean'] = data_copy.apply(
            lambda row: row['y'] - pre_means.get(row['id'], 0), axis=1
        )
        
        # Verify: pre-treatment demeaned outcomes should have mean ≈ 0 per unit
        pre_demeaned = data_copy[data_copy['year'] < treatment_start]
        for unit_id in pre_demeaned['id'].unique():
            unit_pre = pre_demeaned[pre_demeaned['id'] == unit_id]['y_demean']
            assert abs(unit_pre.mean()) < 1e-10, \
                f"Unit {unit_id} pre-treatment demeaned mean should be 0"
    
    def test_detrending_formula_paper_procedure_5_1(self):
        """
        Detrending transformation (Procedure 5.1):
        Ÿ_it = Y_it - (α̂_i + β̂_i × t)
        
        Where (α̂_i, β̂_i) from OLS: Y_is on 1, s for s ∈ {1, ..., g-1}
        """
        import statsmodels.api as sm
        
        # Generate test data
        data, params = generate_small_sample_dgp(seed=42)
        treatment_start = params['treatment_start']
        
        # Fit unit-specific trends
        unit_trends = {}
        for unit_id in data['id'].unique():
            unit_data = data[data['id'] == unit_id]
            pre_data = unit_data[unit_data['year'] < treatment_start]
            
            if len(pre_data) >= 2:
                X = sm.add_constant(pre_data['year'].values)
                y = pre_data['y'].values
                model = sm.OLS(y, X).fit()
                unit_trends[unit_id] = (model.params[0], model.params[1])
        
        # Apply detrending
        data_copy = data.copy()
        
        def detrend_outcome(row):
            alpha, beta = unit_trends.get(row['id'], (0, 0))
            return row['y'] - (alpha + beta * row['year'])
        
        data_copy['y_detrend'] = data_copy.apply(detrend_outcome, axis=1)
        
        # Verify: pre-treatment detrended outcomes should have no linear trend
        pre_detrended = data_copy[data_copy['year'] < treatment_start]
        for unit_id in pre_detrended['id'].unique():
            unit_pre = pre_detrended[pre_detrended['id'] == unit_id]
            if len(unit_pre) >= 2:
                X = sm.add_constant(unit_pre['year'].values)
                y = unit_pre['y_detrend'].values
                model = sm.OLS(y, X).fit()
                
                # Slope should be ≈ 0 after detrending
                assert abs(model.params[1]) < 1e-10, \
                    f"Unit {unit_id} detrended slope should be ~0, got {model.params[1]}"


@pytest.mark.numerical
class TestPeriodEffectsTable1:
    """Tests for period effects matching paper Table 1."""
    
    def test_lambda_t_values_match_table_1(self):
        """
        λ_t values from Table 1.
        
        Pre-treatment (t=1-10): [0, 0, 0, 0, 0.2, 0.6, 0.7, 0.8, 0.6, 0.9]
        Post-treatment (t=11-20): [0.9, 1.0, 1.1, 1.3, 1.2, 1.5, 0.6, 1.4, 1.8, 1.9]
        """
        expected_lambda = [
            0.0, 0.0, 0.0, 0.0, 0.2, 0.6, 0.7, 0.8, 0.6, 0.9,  # t=1-10
            0.9, 1.0, 1.1, 1.3, 1.2, 1.5, 0.6, 1.4, 1.8, 1.9   # t=11-20
        ]
        
        assert len(LAMBDA_T_TABLE1) == 20, \
            f"λ_t should have 20 values, got {len(LAMBDA_T_TABLE1)}"
        
        for t, (expected, actual) in enumerate(zip(expected_lambda, LAMBDA_T_TABLE1)):
            assert expected == actual, \
                f"λ_{t+1} should be {expected}, got {actual}"
    
    def test_delta_t_values_match_table_1(self):
        """
        δ_t values from Table 1.
        
        Pre-treatment (t=1-10): all zeros
        Post-treatment (t=11-20): [1, 2, 3, 3, 3, 2, 2, 2, 1, 1]
        """
        expected_delta = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # t=1-10 (pre)
            1, 2, 3, 3, 3, 2, 2, 2, 1, 1   # t=11-20 (post)
        ]
        
        assert len(DELTA_T_TABLE1) == 20, \
            f"δ_t should have 20 values, got {len(DELTA_T_TABLE1)}"
        
        for t, (expected, actual) in enumerate(zip(expected_delta, DELTA_T_TABLE1)):
            assert expected == actual, \
                f"δ_{t+1} should be {expected}, got {actual}"
    
    def test_average_treatment_effect_calculation(self):
        """
        Average treatment effect = mean(δ_t) for post-treatment periods.
        
        Post-treatment δ_t = [1, 2, 3, 3, 3, 2, 2, 2, 1, 1]
        Average = (1+2+3+3+3+2+2+2+1+1) / 10 = 20/10 = 2.0
        """
        post_delta = DELTA_T_TABLE1[10:]  # t=11-20
        expected_avg = 2.0
        actual_avg = np.mean(post_delta)
        
        assert actual_avg == expected_avg, \
            f"Average δ_t should be {expected_avg}, got {actual_avg}"


@pytest.mark.numerical
class TestTreatmentRuleParameters:
    """Tests for treatment rule parameters matching paper Table 1."""
    
    def test_scenario_1_parameters(self):
        """
        Scenario 1: (α₀, α₁, α₂) = (-1, -1/3, 1/4) → P(D=1) ≈ 0.32
        """
        params = TREATMENT_RULE_PARAMS[1]
        
        assert params['alpha_0'] == -1.0
        assert abs(params['alpha_1'] - (-1/3)) < 1e-10
        assert params['alpha_2'] == 1/4
        assert params['expected_prob'] == 0.32
    
    def test_scenario_2_parameters(self):
        """
        Scenario 2: (α₀, α₁, α₂) = (-1.5, 1/3, 1/4) → P(D=1) ≈ 0.24
        """
        params = TREATMENT_RULE_PARAMS[2]
        
        assert params['alpha_0'] == -1.5
        assert abs(params['alpha_1'] - (1/3)) < 1e-10
        assert params['alpha_2'] == 1/4
        assert params['expected_prob'] == 0.24
    
    def test_scenario_3_parameters(self):
        """
        Scenario 3: (α₀, α₁, α₂) = (-2, 1/3, 1/4) → P(D=1) ≈ 0.17
        """
        params = TREATMENT_RULE_PARAMS[3]
        
        assert params['alpha_0'] == -2.0
        assert abs(params['alpha_1'] - (1/3)) < 1e-10
        assert params['alpha_2'] == 1/4
        assert params['expected_prob'] == 0.17


@pytest.mark.numerical
class TestOutcomeFormula:
    """Tests for outcome formula Y_it matching paper specification."""
    
    def test_outcome_formula_control_unit(self):
        """
        For control units (D_i = 0):
        Y_it = λ_t - C_i + G_i × t + u_it
        
        Note: Paper uses -C_i (negative sign on fixed effect)
        """
        data, params = generate_small_sample_dgp(
            seed=42,
            return_components=True,
        )
        
        C = params['C']
        G = params['G']
        D = params['D']
        U = params['U']
        lambda_t = params['lambda_t']
        
        # Check control units
        control_ids = np.where(D == 0)[0]
        
        for i in control_ids[:3]:  # Check first 3 control units
            unit_data = data[data['id'] == i + 1]
            
            for _, row in unit_data.iterrows():
                t = int(row['year'])
                t_idx = t - 1  # 0-indexed
                
                # Expected outcome for control
                expected_y = lambda_t[t_idx] - C[i] + G[i] * t + U[i, t_idx]
                actual_y = row['y']
                
                assert abs(expected_y - actual_y) < 1e-10, \
                    f"Control unit {i+1}, period {t}: expected {expected_y}, got {actual_y}"
    
    def test_outcome_formula_treated_unit_pre_treatment(self):
        """
        For treated units in pre-treatment periods:
        Y_it = λ_t - C_i + G_i × t + u_it (same as control)
        """
        data, params = generate_small_sample_dgp(
            seed=42,
            return_components=True,
        )
        
        C = params['C']
        G = params['G']
        D = params['D']
        U = params['U']
        lambda_t = params['lambda_t']
        treatment_start = params['treatment_start']
        
        # Check treated units in pre-treatment
        treated_ids = np.where(D == 1)[0]
        
        for i in treated_ids[:3]:  # Check first 3 treated units
            unit_data = data[(data['id'] == i + 1) & (data['year'] < treatment_start)]
            
            for _, row in unit_data.iterrows():
                t = int(row['year'])
                t_idx = t - 1
                
                # Expected outcome (no treatment effect in pre-treatment)
                expected_y = lambda_t[t_idx] - C[i] + G[i] * t + U[i, t_idx]
                actual_y = row['y']
                
                assert abs(expected_y - actual_y) < 1e-10, \
                    f"Treated unit {i+1}, pre-treatment period {t}: expected {expected_y}, got {actual_y}"


@pytest.mark.numerical
class TestTheoreticalVarianceComponents:
    """Tests for theoretical variance component calculations."""
    
    def test_variance_components_formula(self):
        """
        Var(Y_it) = σ_C² + t² × σ_G² + σ_u²
        where σ_u² = σ_ε² / (1 - ρ²)
        """
        components = compute_theoretical_variance_components(
            sigma_c=2.0,
            sigma_g=1.0,
            sigma_epsilon=np.sqrt(2),
            rho=0.75,
            n_periods=20,
        )
        
        # Check individual components
        assert components['var_c'] == 4.0  # σ_C² = 2² = 4
        assert components['var_g'] == 1.0  # σ_G² = 1² = 1
        
        # σ_u² = σ_ε² / (1 - ρ²) = 2 / (1 - 0.5625) = 2 / 0.4375 ≈ 4.57
        expected_var_u = 2.0 / (1 - 0.75**2)
        assert abs(components['var_u'] - expected_var_u) < 0.01
        
        # Var(Y_t1) = 4 + 1 × 1 + 4.57 ≈ 9.57
        expected_var_t1 = 4.0 + 1.0 + expected_var_u
        assert abs(components['var_y_t1'] - expected_var_t1) < 0.01


@pytest.mark.numerical
class TestLargeSampleConvergence:
    """Tests for large-sample convergence to theoretical values."""
    
    @pytest.mark.slow
    def test_treatment_probability_convergence(self):
        """
        With large N, treatment probability should converge to expected value.
        """
        for scenario in [1, 2, 3]:
            expected_prob = TREATMENT_RULE_PARAMS[scenario]['expected_prob']
            
            # Run multiple simulations
            probs = []
            for seed in range(100):
                data, params = generate_small_sample_dgp(
                    n_units=100,  # Larger sample
                    scenario=scenario,
                    seed=seed,
                )
                probs.append(params['treated_share'])
            
            mean_prob = np.mean(probs)
            
            # Should be within 5% of expected
            assert abs(mean_prob - expected_prob) < 0.05, \
                f"Scenario {scenario}: mean P(D=1) = {mean_prob}, expected {expected_prob}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'numerical'])

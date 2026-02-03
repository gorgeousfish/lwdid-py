# -*- coding: utf-8 -*-
"""
Unit tests for small-sample DGP.

Based on Lee & Wooldridge (2026) ssrn-5325686, Section 5.

These tests verify:
1. Parameter validation
2. Data structure correctness
3. Treatment assignment properties
4. Reproducibility
5. Scenario configurations
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

# Add fixtures path
fixtures_path = Path(__file__).parent.parent / 'test_common_timing' / 'fixtures'
sys.path.insert(0, str(fixtures_path))

from dgp_small_sample import (
    SmallSampleDGPParams,
    generate_small_sample_dgp,
    generate_small_sample_dgp_from_scenario,
    validate_dgp_output,
    compute_theoretical_variance_components,
    compute_expected_treatment_probability,
    SMALL_SAMPLE_SCENARIOS,
    DEFAULT_SMALL_SAMPLE_PARAMS,
    TREATMENT_RULE_PARAMS,
    LAMBDA_T_TABLE1,
    DELTA_T_TABLE1,
)


class TestSmallSampleDGPParams:
    """Tests for DGP parameter validation."""
    
    def test_valid_params_default(self):
        """Default parameters should not raise."""
        params = SmallSampleDGPParams()
        assert params.n_units == 20
        assert params.n_periods == 20
        assert params.treatment_start == 11
        assert params.scenario == 1
        assert params.sigma_c == 2.0
        assert params.sigma_g == 1.0
        assert params.rho == 0.75
    
    def test_valid_params_custom(self):
        """Custom valid parameters should not raise."""
        params = SmallSampleDGPParams(
            n_units=50,
            n_periods=30,
            treatment_start=15,
            scenario=2,
            sigma_c=3.0,
            sigma_g=2.0,
            rho=0.5,
            seed=123,
        )
        assert params.n_units == 50
        assert params.scenario == 2
    
    def test_invalid_n_units_too_small(self):
        """n_units < 3 should raise ValueError."""
        with pytest.raises(ValueError, match="n_units must be at least 3"):
            SmallSampleDGPParams(n_units=2)
    
    def test_invalid_n_units_zero(self):
        """n_units = 0 should raise ValueError."""
        with pytest.raises(ValueError, match="n_units must be at least 3"):
            SmallSampleDGPParams(n_units=0)
    
    def test_invalid_n_periods_too_small(self):
        """n_periods < 3 should raise ValueError."""
        with pytest.raises(ValueError, match="n_periods must be at least 3"):
            SmallSampleDGPParams(n_periods=2)
    
    def test_invalid_scenario(self):
        """Invalid scenario should raise ValueError."""
        with pytest.raises(ValueError, match="scenario must be 1, 2, or 3"):
            SmallSampleDGPParams(scenario=4)
    
    def test_invalid_treatment_start_too_early(self):
        """treatment_start = 1 should raise ValueError."""
        with pytest.raises(ValueError, match="treatment_start must be in"):
            SmallSampleDGPParams(treatment_start=1)
    
    def test_invalid_treatment_start_too_late(self):
        """treatment_start > n_periods should raise ValueError."""
        with pytest.raises(ValueError, match="treatment_start must be in"):
            SmallSampleDGPParams(n_periods=20, treatment_start=25)
    
    def test_invalid_sigma_c_negative(self):
        """sigma_c < 0 should raise ValueError."""
        with pytest.raises(ValueError, match="sigma_c must be non-negative"):
            SmallSampleDGPParams(sigma_c=-1.0)
    
    def test_invalid_sigma_g_negative(self):
        """sigma_g < 0 should raise ValueError."""
        with pytest.raises(ValueError, match="sigma_g must be non-negative"):
            SmallSampleDGPParams(sigma_g=-0.5)
    
    def test_invalid_rho_out_of_range(self):
        """rho outside (-1, 1) should raise ValueError."""
        with pytest.raises(ValueError, match="rho must be in"):
            SmallSampleDGPParams(rho=1.0)
        with pytest.raises(ValueError, match="rho must be in"):
            SmallSampleDGPParams(rho=-1.0)
    
    def test_valid_sigma_zero(self):
        """sigma = 0 should be valid (deterministic case)."""
        params = SmallSampleDGPParams(sigma_c=0.0, sigma_g=0.0)
        assert params.sigma_c == 0.0
        assert params.sigma_g == 0.0


class TestGenerateSmallSampleDGP:
    """Tests for DGP generation function."""
    
    def test_output_types(self):
        """Verify output types."""
        data, params = generate_small_sample_dgp(seed=42)
        
        assert isinstance(data, pd.DataFrame)
        assert isinstance(params, dict)
    
    def test_output_structure(self):
        """Verify output DataFrame structure."""
        data, params = generate_small_sample_dgp(seed=42)
        
        required_cols = {'id', 'year', 'y', 'd', 'post'}
        assert required_cols.issubset(set(data.columns))
        assert len(data) == 20 * 20  # n_units × n_periods
    
    def test_panel_structure_balanced(self):
        """Verify balanced panel structure."""
        data, params = generate_small_sample_dgp(seed=42)
        
        assert data['id'].nunique() == 20
        assert data['year'].nunique() == 20
        
        # Each unit should have exactly n_periods observations
        obs_per_unit = data.groupby('id').size()
        assert obs_per_unit.nunique() == 1
        assert obs_per_unit.iloc[0] == 20
    
    def test_treatment_assignment_time_constant(self):
        """Verify treatment is time-constant within unit."""
        data, params = generate_small_sample_dgp(seed=42)
        
        # Treatment should be constant within unit
        d_by_unit = data.groupby('id')['d'].nunique()
        assert (d_by_unit == 1).all()
    
    def test_treatment_assignment_binary(self):
        """Verify treatment is binary."""
        data, params = generate_small_sample_dgp(seed=42)
        
        assert set(data['d'].unique()).issubset({0, 1})
    
    def test_post_indicator_correct(self):
        """Verify post indicator is correct."""
        data, params = generate_small_sample_dgp(
            treatment_start=11, seed=42
        )
        
        # post = 0 for year < 11
        assert (data[data['year'] < 11]['post'] == 0).all()
        # post = 1 for year >= 11
        assert (data[data['year'] >= 11]['post'] == 1).all()
    
    def test_post_indicator_different_treatment_start(self):
        """Verify post indicator with different treatment_start."""
        data, params = generate_small_sample_dgp(
            treatment_start=5, seed=42
        )
        
        assert (data[data['year'] < 5]['post'] == 0).all()
        assert (data[data['year'] >= 5]['post'] == 1).all()
    
    def test_at_least_one_treated_and_control(self):
        """Ensure at least 1 treated and 1 control unit."""
        # Test with various seeds
        for seed in range(50):
            for scenario in [1, 2, 3]:
                data, params = generate_small_sample_dgp(
                    scenario=scenario, seed=seed
                )
                assert params['n_treated'] >= 1, \
                    f"No treated units with scenario={scenario}, seed={seed}"
                assert params['n_control'] >= 1, \
                    f"No control units with scenario={scenario}, seed={seed}"
    
    def test_reproducibility_same_seed(self):
        """Same seed should produce same data."""
        data1, params1 = generate_small_sample_dgp(seed=42)
        data2, params2 = generate_small_sample_dgp(seed=42)
        
        pd.testing.assert_frame_equal(data1, data2)
        assert params1['n_treated'] == params2['n_treated']
        assert params1['n_control'] == params2['n_control']
    
    def test_reproducibility_different_seeds(self):
        """Different seeds should produce different data."""
        data1, _ = generate_small_sample_dgp(seed=42)
        data2, _ = generate_small_sample_dgp(seed=43)
        
        # Outcomes should differ
        assert not data1['y'].equals(data2['y'])
    
    def test_params_returned_correctly(self):
        """Verify all expected params are returned."""
        data, params = generate_small_sample_dgp(
            n_units=20,
            n_periods=20,
            treatment_start=11,
            scenario=1,
            seed=42,
        )
        
        assert params['n_units'] == 20
        assert params['n_periods'] == 20
        assert params['treatment_start'] == 11
        assert params['scenario'] == 1
        assert params['seed'] == 42
        assert params['n_pre_periods'] == 10
        assert params['n_post_periods'] == 10
        assert params['n_treated'] + params['n_control'] == 20
        assert 'tau' in params
        assert 'true_att' in params
        assert 'att_by_period' in params
    
    def test_return_components(self):
        """Verify components are returned when requested."""
        data, params = generate_small_sample_dgp(
            seed=42, return_components=True
        )
        
        assert 'C' in params
        assert 'G' in params
        assert 'D' in params
        assert 'U' in params
        
        assert len(params['C']) == 20
        assert len(params['G']) == 20
        assert len(params['D']) == 20
        assert params['U'].shape == (20, 20)
    
    def test_components_not_returned_by_default(self):
        """Verify components are not returned by default."""
        data, params = generate_small_sample_dgp(seed=42)
        
        assert 'C' not in params
        assert 'G' not in params
        assert 'D' not in params
        assert 'U' not in params
    
    def test_id_starts_from_one(self):
        """Verify unit IDs start from 1."""
        data, params = generate_small_sample_dgp(seed=42)
        
        assert data['id'].min() == 1
        assert data['id'].max() == 20
    
    def test_year_starts_from_one(self):
        """Verify years start from 1."""
        data, params = generate_small_sample_dgp(seed=42)
        
        assert data['year'].min() == 1
        assert data['year'].max() == 20
    
    def test_custom_dimensions(self):
        """Test with custom panel dimensions."""
        data, params = generate_small_sample_dgp(
            n_units=50,
            n_periods=30,
            treatment_start=15,
            seed=42,
        )
        
        assert len(data) == 50 * 30
        assert data['id'].nunique() == 50
        assert data['year'].nunique() == 30
        assert params['n_pre_periods'] == 14
        assert params['n_post_periods'] == 16


class TestScenarioConfigurations:
    """Tests for scenario configurations."""
    
    def test_scenario_1_config(self):
        """Verify Scenario 1 configuration."""
        config = SMALL_SAMPLE_SCENARIOS['scenario_1']
        assert config['scenario'] == 1
        assert config['prob_treated'] == 0.32
        assert 'description' in config
    
    def test_scenario_2_config(self):
        """Verify Scenario 2 configuration."""
        config = SMALL_SAMPLE_SCENARIOS['scenario_2']
        assert config['scenario'] == 2
        assert config['prob_treated'] == 0.24
    
    def test_scenario_3_config(self):
        """Verify Scenario 3 configuration."""
        config = SMALL_SAMPLE_SCENARIOS['scenario_3']
        assert config['scenario'] == 3
        assert config['prob_treated'] == 0.17
    
    def test_all_scenarios_have_required_keys(self):
        """All scenarios should have required keys."""
        required_keys = {'scenario', 'prob_treated', 'description'}
        
        for scenario_name, config in SMALL_SAMPLE_SCENARIOS.items():
            assert required_keys.issubset(set(config.keys())), \
                f"Scenario {scenario_name} missing required keys"
    
    def test_generate_from_scenario_1(self):
        """Test generating data from Scenario 1."""
        data, params = generate_small_sample_dgp_from_scenario(
            'scenario_1', seed=42
        )
        
        assert params['scenario'] == 1
        assert params['scenario_name'] == 'scenario_1'
        assert 'scenario_description' in params
    
    def test_generate_from_scenario_2(self):
        """Test generating data from Scenario 2."""
        data, params = generate_small_sample_dgp_from_scenario(
            'scenario_2', seed=42
        )
        
        assert params['scenario'] == 2
        assert params['scenario_name'] == 'scenario_2'
    
    def test_generate_from_scenario_3(self):
        """Test generating data from Scenario 3."""
        data, params = generate_small_sample_dgp_from_scenario(
            'scenario_3', seed=42
        )
        
        assert params['scenario'] == 3
        assert params['scenario_name'] == 'scenario_3'
    
    def test_generate_from_invalid_scenario(self):
        """Invalid scenario should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown scenario"):
            generate_small_sample_dgp_from_scenario('scenario_99', seed=42)
    
    def test_scenario_override_params(self):
        """Test overriding scenario parameters."""
        data, params = generate_small_sample_dgp_from_scenario(
            'scenario_1',
            seed=42,
            n_units=30,  # Override n_units
        )
        
        assert params['n_units'] == 30
        assert params['scenario'] == 1  # From scenario


class TestTreatmentProbabilities:
    """Tests for treatment probability calibration."""
    
    def test_scenario_1_treatment_probability(self):
        """Scenario 1 should have P(D=1) ≈ 0.32."""
        result = compute_expected_treatment_probability(scenario=1, seed=42)
        # Allow 5% tolerance due to simulation variance
        assert result['difference'] < 0.05, \
            f"Scenario 1: expected {result['expected_prob']}, got {result['simulated_prob']}"
    
    def test_scenario_2_treatment_probability(self):
        """Scenario 2 should have P(D=1) ≈ 0.24."""
        result = compute_expected_treatment_probability(scenario=2, seed=42)
        assert result['difference'] < 0.05, \
            f"Scenario 2: expected {result['expected_prob']}, got {result['simulated_prob']}"
    
    def test_scenario_3_treatment_probability(self):
        """Scenario 3 should have P(D=1) ≈ 0.17."""
        result = compute_expected_treatment_probability(scenario=3, seed=42)
        assert result['difference'] < 0.05, \
            f"Scenario 3: expected {result['expected_prob']}, got {result['simulated_prob']}"
    
    def test_treatment_rates_across_scenarios(self):
        """Treatment rates should decrease from Scenario 1 to 3."""
        n_reps = 100
        
        rates = {}
        for scenario in [1, 2, 3]:
            treated_counts = []
            for seed in range(n_reps):
                _, params = generate_small_sample_dgp(
                    scenario=scenario, seed=seed
                )
                treated_counts.append(params['n_treated'])
            rates[scenario] = np.mean(treated_counts) / 20
        
        # Scenario 1 > Scenario 2 > Scenario 3
        assert rates[1] > rates[2], f"Rate 1 ({rates[1]}) should > Rate 2 ({rates[2]})"
        assert rates[2] > rates[3], f"Rate 2 ({rates[2]}) should > Rate 3 ({rates[3]})"


class TestValidateDGPOutput:
    """Tests for DGP output validation function."""
    
    def test_valid_output_passes_all_checks(self):
        """Valid DGP output should pass all checks."""
        data, params = generate_small_sample_dgp(seed=42)
        checks = validate_dgp_output(data, params)
        
        for check_name, passed in checks.items():
            assert passed, f"Check '{check_name}' failed"
    
    def test_validation_detects_wrong_row_count(self):
        """Validation should detect wrong row count."""
        data, params = generate_small_sample_dgp(seed=42)
        data_bad = data.iloc[:-10]  # Remove some rows
        
        checks = validate_dgp_output(data_bad, params)
        assert not checks['correct_n_rows']


class TestTheoreticalVarianceComponents:
    """Tests for theoretical variance computation."""
    
    def test_default_variance_components(self):
        """Test variance components with default parameters."""
        var_comp = compute_theoretical_variance_components()
        
        # σ_C² = 4, σ_G² = 1
        assert var_comp['var_c'] == 4.0
        assert var_comp['var_g'] == 1.0
        
        # Mean of G is 1
        assert var_comp['mean_g'] == 1.0
        
        # AR(1) variance: σ_ε² / (1 - ρ²) = 2 / (1 - 0.5625) ≈ 4.57
        assert 'var_u' in var_comp
        assert var_comp['var_u'] > 0
    
    def test_variance_increases_with_time(self):
        """Variance should increase with time due to trend."""
        var_comp = compute_theoretical_variance_components()
        
        # Var(Y_t) = σ_C² + t² × σ_G² + σ_u²
        # Should increase with t
        assert var_comp['var_y_t1'] < var_comp['var_y_t_mid']
        assert var_comp['var_y_t_mid'] < var_comp['var_y_t_end']


class TestDGPFormulaCorrectness:
    """Tests verifying DGP formula matches paper."""
    
    def test_treatment_effect_time_varying(self):
        """Treatment effects should be time-varying as per Table 1."""
        data, params = generate_small_sample_dgp(seed=42)
        
        # Check that att_by_period matches DELTA_T_TABLE1
        for period, att in params['att_by_period'].items():
            expected = DELTA_T_TABLE1[period - 1]
            assert att == expected, \
                f"Period {period}: expected δ={expected}, got {att}"
    
    def test_average_treatment_effect(self):
        """Average treatment effect should be mean of post-treatment δ_t."""
        data, params = generate_small_sample_dgp(seed=42)
        
        # Post-treatment periods: 11-20
        post_deltas = DELTA_T_TABLE1[10:20]  # indices 10-19
        expected_avg = np.mean(post_deltas)
        
        assert np.isclose(params['tau'], expected_avg), \
            f"Expected tau={expected_avg}, got {params['tau']}"
    
    def test_period_effects_from_table1(self):
        """Period effects λ_t should match Table 1."""
        data, params = generate_small_sample_dgp(seed=42)
        
        assert params['lambda_t'] == LAMBDA_T_TABLE1[:20]
    
    def test_unit_trend_mean_is_one(self):
        """G_i should have mean ≈ 1 (not 0)."""
        n_reps = 100
        G_samples = []
        
        for seed in range(n_reps):
            _, params = generate_small_sample_dgp(
                seed=seed, return_components=True
            )
            G_samples.extend(params['G'])
        
        mean_G = np.mean(G_samples)
        # Mean should be close to 1
        assert abs(mean_G - 1.0) < 0.1, \
            f"Mean of G should be ≈ 1, got {mean_G}"
    
    def test_unit_fixed_effect_mean_is_zero(self):
        """C_i should have mean ≈ 0."""
        n_reps = 100
        C_samples = []
        
        for seed in range(n_reps):
            _, params = generate_small_sample_dgp(
                seed=seed, return_components=True
            )
            C_samples.extend(params['C'])
        
        mean_C = np.mean(C_samples)
        # Mean should be close to 0
        assert abs(mean_C) < 0.2, \
            f"Mean of C should be ≈ 0, got {mean_C}"


class TestTable1Parameters:
    """Tests for Table 1 parameter values."""
    
    def test_lambda_t_length(self):
        """λ_t should have 20 values."""
        assert len(LAMBDA_T_TABLE1) == 20
    
    def test_delta_t_length(self):
        """δ_t should have 20 values."""
        assert len(DELTA_T_TABLE1) == 20
    
    def test_delta_t_pre_treatment_zero(self):
        """δ_t should be 0 for pre-treatment periods (t=1-10)."""
        for t in range(10):
            assert DELTA_T_TABLE1[t] == 0, \
                f"δ_{t+1} should be 0, got {DELTA_T_TABLE1[t]}"
    
    def test_delta_t_post_treatment_positive(self):
        """δ_t should be positive for post-treatment periods (t=11-20)."""
        for t in range(10, 20):
            assert DELTA_T_TABLE1[t] > 0, \
                f"δ_{t+1} should be > 0, got {DELTA_T_TABLE1[t]}"
    
    def test_treatment_rule_params_complete(self):
        """All scenarios should have complete treatment rule parameters."""
        for scenario in [1, 2, 3]:
            params = TREATMENT_RULE_PARAMS[scenario]
            assert 'alpha_0' in params
            assert 'alpha_1' in params
            assert 'alpha_2' in params
            assert 'expected_prob' in params


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

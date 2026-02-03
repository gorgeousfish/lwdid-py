"""
Simulated data tests for sensitivity analysis module.

Tests various data scenarios including:
- Common timing design
- Staggered adoption design
- Small sample scenarios
- Unbalanced panels
"""

import pytest
import numpy as np
import pandas as pd

from lwdid.sensitivity import (
    robustness_pre_periods,
    sensitivity_no_anticipation,
    sensitivity_analysis,
    PrePeriodRobustnessResult,
    NoAnticipationSensitivityResult,
    ComprehensiveSensitivityResult,
)


def generate_common_timing_data(
    n_units: int = 100,
    n_periods: int = 10,
    treatment_period: int = 6,
    treatment_effect: float = 2.0,
    noise_std: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate common timing design data."""
    np.random.seed(seed)
    
    data = []
    n_treated = n_units // 2
    
    for i in range(n_units):
        alpha_i = np.random.normal(0, 1)
        treated = i < n_treated
        
        for t in range(1, n_periods + 1):
            d_it = 1 if treated and t >= treatment_period else 0
            tau = treatment_effect if d_it else 0
            y = alpha_i + tau + np.random.normal(0, noise_std)
            
            data.append({
                'unit': i,
                'time': t,
                'Y': y,
                'D': 1 if treated else 0,
                'post': 1 if t >= treatment_period else 0,
            })
    
    return pd.DataFrame(data)


def generate_staggered_data(
    n_units: int = 150,
    n_periods: int = 15,
    cohorts: list = None,
    cohort_sizes: list = None,
    treatment_effect: float = 2.0,
    noise_std: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate staggered adoption design data."""
    if cohorts is None:
        cohorts = [6, 8, 10]
    if cohort_sizes is None:
        cohort_sizes = [40, 40, 40]
    
    np.random.seed(seed)
    
    data = []
    unit_id = 0
    
    # Treated cohorts
    for cohort, size in zip(cohorts, cohort_sizes):
        for _ in range(size):
            alpha_i = np.random.normal(0, 1)
            
            for t in range(1, n_periods + 1):
                d_it = 1 if t >= cohort else 0
                tau = treatment_effect if d_it else 0
                y = alpha_i + tau + np.random.normal(0, noise_std)
                
                data.append({
                    'unit': unit_id,
                    'time': t,
                    'Y': y,
                    'first_treat': cohort,
                })
            
            unit_id += 1
    
    # Never-treated units
    n_never_treated = n_units - sum(cohort_sizes)
    for _ in range(n_never_treated):
        alpha_i = np.random.normal(0, 1)
        
        for t in range(1, n_periods + 1):
            y = alpha_i + np.random.normal(0, noise_std)
            
            data.append({
                'unit': unit_id,
                'time': t,
                'Y': y,
                'first_treat': 0,
            })
        
        unit_id += 1
    
    return pd.DataFrame(data)


def generate_unbalanced_panel_data(
    n_units: int = 100,
    n_periods: int = 10,
    treatment_period: int = 6,
    treatment_effect: float = 2.0,
    missing_rate: float = 0.1,
    noise_std: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate unbalanced panel data with random missing observations."""
    np.random.seed(seed)
    
    data = []
    n_treated = n_units // 2
    
    for i in range(n_units):
        alpha_i = np.random.normal(0, 1)
        treated = i < n_treated
        
        for t in range(1, n_periods + 1):
            # Randomly skip some observations
            if np.random.random() < missing_rate:
                continue
            
            d_it = 1 if treated and t >= treatment_period else 0
            tau = treatment_effect if d_it else 0
            y = alpha_i + tau + np.random.normal(0, noise_std)
            
            data.append({
                'unit': i,
                'time': t,
                'Y': y,
                'first_treat': treatment_period if treated else 0,
            })
    
    return pd.DataFrame(data)


class TestCommonTimingScenario:
    """Test with common timing design."""
    
    @pytest.fixture
    def common_timing_data(self):
        """Generate common timing data."""
        return generate_common_timing_data(
            n_units=100, n_periods=10, treatment_period=6,
            treatment_effect=2.0, noise_std=0.5, seed=42
        )
    
    def test_robustness_pre_periods(self, common_timing_data):
        """Test robustness analysis with common timing."""
        result = robustness_pre_periods(
            common_timing_data, y='Y', ivar='unit', tvar='time',
            d='D', post='post', rolling='demean',
            pre_period_range=(2, 4), verbose=False
        )
        
        assert isinstance(result, PrePeriodRobustnessResult)
        assert result.n_specifications >= 2
        assert not np.isnan(result.baseline_spec.att)
    
    def test_sensitivity_no_anticipation(self, common_timing_data):
        """Test anticipation analysis with common timing."""
        result = sensitivity_no_anticipation(
            common_timing_data, y='Y', ivar='unit', tvar='time',
            d='D', post='post', rolling='demean',
            max_anticipation=2, verbose=False
        )
        
        assert isinstance(result, NoAnticipationSensitivityResult)
        assert len(result.estimates) >= 2
    
    def test_comprehensive_analysis(self, common_timing_data):
        """Test comprehensive analysis with common timing."""
        result = sensitivity_analysis(
            common_timing_data, y='Y', ivar='unit', tvar='time',
            d='D', post='post', rolling='demean',
            analyses=['pre_periods', 'anticipation'], verbose=False
        )
        
        assert isinstance(result, ComprehensiveSensitivityResult)
        assert result.pre_period_result is not None


class TestStaggeredScenario:
    """Test with staggered adoption design."""
    
    @pytest.fixture
    def staggered_data(self):
        """Generate staggered data."""
        return generate_staggered_data(
            n_units=150, n_periods=15,
            cohorts=[6, 8, 10], cohort_sizes=[40, 40, 40],
            treatment_effect=2.0, noise_std=0.5, seed=42
        )
    
    def test_robustness_pre_periods(self, staggered_data):
        """Test robustness analysis with staggered design."""
        result = robustness_pre_periods(
            staggered_data, y='Y', ivar='unit', tvar='time',
            gvar='first_treat', rolling='demean',
            pre_period_range=(2, 4), verbose=False
        )
        
        assert isinstance(result, PrePeriodRobustnessResult)
        assert result.n_specifications >= 2
    
    def test_sensitivity_no_anticipation(self, staggered_data):
        """Test anticipation analysis with staggered design."""
        result = sensitivity_no_anticipation(
            staggered_data, y='Y', ivar='unit', tvar='time',
            gvar='first_treat', rolling='demean',
            max_anticipation=2, verbose=False
        )
        
        assert isinstance(result, NoAnticipationSensitivityResult)
        assert len(result.estimates) >= 2
    
    def test_multiple_cohorts_handled(self, staggered_data):
        """Test that multiple cohorts are handled correctly."""
        result = robustness_pre_periods(
            staggered_data, y='Y', ivar='unit', tvar='time',
            gvar='first_treat', rolling='demean',
            verbose=False
        )
        
        # Should complete without error
        assert result.n_specifications > 0
        
        # ATT should be close to true effect (2.0)
        assert 1.0 < result.baseline_spec.att < 3.0


class TestSmallSampleScenario:
    """Test with small sample sizes."""
    
    def test_small_n_units(self):
        """Test with small number of units (N=20)."""
        data = generate_common_timing_data(
            n_units=20, n_periods=8, treatment_period=5,
            treatment_effect=2.0, noise_std=0.5, seed=42
        )
        
        result = robustness_pre_periods(
            data, y='Y', ivar='unit', tvar='time',
            d='D', post='post', rolling='demean',
            pre_period_range=(2, 3), verbose=False
        )
        
        # Should handle small samples gracefully
        assert result.n_specifications >= 2
    
    def test_small_n_periods(self):
        """Test with small number of periods (T=6)."""
        data = generate_common_timing_data(
            n_units=100, n_periods=6, treatment_period=4,
            treatment_effect=2.0, noise_std=0.5, seed=42
        )
        
        result = robustness_pre_periods(
            data, y='Y', ivar='unit', tvar='time',
            d='D', post='post', rolling='demean',
            pre_period_range=(2, 3), verbose=False
        )
        
        assert result.n_specifications >= 2
    
    def test_minimum_pre_periods(self):
        """Test with minimum pre-treatment periods."""
        # Only 2 pre-treatment periods
        data = generate_common_timing_data(
            n_units=100, n_periods=5, treatment_period=3,
            treatment_effect=2.0, noise_std=0.5, seed=42
        )
        
        result = robustness_pre_periods(
            data, y='Y', ivar='unit', tvar='time',
            d='D', post='post', rolling='demean',
            pre_period_range=(1, 2), verbose=False
        )
        
        assert result.n_specifications >= 1


class TestUnbalancedPanelScenario:
    """Test with unbalanced panel data."""
    
    @pytest.fixture
    def unbalanced_data(self):
        """Generate unbalanced panel data."""
        return generate_unbalanced_panel_data(
            n_units=100, n_periods=10, treatment_period=6,
            treatment_effect=2.0, missing_rate=0.1, noise_std=0.5, seed=42
        )
    
    def test_robustness_with_missing(self, unbalanced_data):
        """Test robustness analysis handles missing data."""
        result = robustness_pre_periods(
            unbalanced_data, y='Y', ivar='unit', tvar='time',
            gvar='first_treat', rolling='demean',
            pre_period_range=(2, 4), verbose=False
        )
        
        # Should handle unbalanced panels
        assert result.n_specifications > 0
    
    def test_anticipation_with_missing(self, unbalanced_data):
        """Test anticipation analysis handles missing data."""
        result = sensitivity_no_anticipation(
            unbalanced_data, y='Y', ivar='unit', tvar='time',
            gvar='first_treat', rolling='demean',
            max_anticipation=2, verbose=False
        )
        
        assert len(result.estimates) > 0


class TestTransformationMethods:
    """Test different transformation methods."""
    
    @pytest.fixture
    def panel_data(self):
        """Generate panel data."""
        return generate_common_timing_data(
            n_units=100, n_periods=12, treatment_period=7,
            treatment_effect=2.0, noise_std=0.5, seed=42
        )
    
    def test_demean_method(self, panel_data):
        """Test with demean transformation."""
        result = robustness_pre_periods(
            panel_data, y='Y', ivar='unit', tvar='time',
            d='D', post='post', rolling='demean',
            pre_period_range=(2, 5), verbose=False
        )
        
        assert result.rolling_method == 'demean'
        assert result.n_specifications >= 2
    
    def test_detrend_method(self, panel_data):
        """Test with detrend transformation."""
        result = robustness_pre_periods(
            panel_data, y='Y', ivar='unit', tvar='time',
            d='D', post='post', rolling='detrend',
            pre_period_range=(2, 5), verbose=False
        )
        
        assert result.rolling_method == 'detrend'
        assert result.n_specifications >= 2
    
    def test_demean_vs_detrend_comparison(self, panel_data):
        """Compare demean and detrend results."""
        result_demean = robustness_pre_periods(
            panel_data, y='Y', ivar='unit', tvar='time',
            d='D', post='post', rolling='demean',
            pre_period_range=(3, 5), verbose=False
        )
        
        result_detrend = robustness_pre_periods(
            panel_data, y='Y', ivar='unit', tvar='time',
            d='D', post='post', rolling='detrend',
            pre_period_range=(3, 5), verbose=False
        )
        
        # Both should produce valid results
        assert not np.isnan(result_demean.baseline_spec.att)
        assert not np.isnan(result_detrend.baseline_spec.att)


class TestEstimatorMethods:
    """Test different estimator methods."""
    
    @pytest.fixture
    def panel_data_with_controls(self):
        """Generate panel data with control variables."""
        np.random.seed(42)
        n_units, n_periods = 100, 10
        treatment_period = 6
        
        data = []
        n_treated = n_units // 2
        
        for i in range(n_units):
            alpha_i = np.random.normal(0, 1)
            treated = i < n_treated
            x1 = np.random.normal(0, 1)  # Control variable
            
            for t in range(1, n_periods + 1):
                d_it = 1 if treated and t >= treatment_period else 0
                tau = 2.0 if d_it else 0
                y = alpha_i + 0.5 * x1 + tau + np.random.normal(0, 0.5)
                
                data.append({
                    'unit': i,
                    'time': t,
                    'Y': y,
                    'X1': x1,
                    'first_treat': treatment_period if treated else 0,
                })
        
        return pd.DataFrame(data)
    
    def test_ra_estimator(self, panel_data_with_controls):
        """Test with RA estimator."""
        result = robustness_pre_periods(
            panel_data_with_controls, y='Y', ivar='unit', tvar='time',
            gvar='first_treat', rolling='demean', estimator='ra',
            pre_period_range=(2, 4), verbose=False
        )
        
        assert result.estimator == 'ra'
        assert result.n_specifications >= 2
    
    def test_ipw_estimator(self, panel_data_with_controls):
        """Test with IPW estimator."""
        result = robustness_pre_periods(
            panel_data_with_controls, y='Y', ivar='unit', tvar='time',
            gvar='first_treat', rolling='demean', estimator='ipw',
            controls=['X1'], pre_period_range=(2, 4), verbose=False
        )
        
        assert result.estimator == 'ipw'


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_specification(self):
        """Test with only one possible specification."""
        data = generate_common_timing_data(
            n_units=50, n_periods=4, treatment_period=3,
            treatment_effect=2.0, noise_std=0.5, seed=42
        )
        
        # Only 2 pre-periods, so only one specification possible
        result = robustness_pre_periods(
            data, y='Y', ivar='unit', tvar='time',
            d='D', post='post', rolling='demean',
            pre_period_range=(2, 2), verbose=False
        )
        
        assert result.n_specifications >= 1
    
    def test_all_treated_same_cohort(self):
        """Test when all treated units are in same cohort."""
        data = generate_staggered_data(
            n_units=100, n_periods=10,
            cohorts=[6], cohort_sizes=[50],
            treatment_effect=2.0, noise_std=0.5, seed=42
        )
        
        result = robustness_pre_periods(
            data, y='Y', ivar='unit', tvar='time',
            gvar='first_treat', rolling='demean',
            pre_period_range=(2, 4), verbose=False
        )
        
        assert result.n_specifications >= 2
    
    def test_high_noise_data(self):
        """Test with high noise data."""
        data = generate_common_timing_data(
            n_units=100, n_periods=10, treatment_period=6,
            treatment_effect=2.0, noise_std=2.0,  # High noise
            seed=42
        )
        
        result = robustness_pre_periods(
            data, y='Y', ivar='unit', tvar='time',
            d='D', post='post', rolling='demean',
            pre_period_range=(2, 4), verbose=False
        )
        
        # Should still produce results
        assert result.n_specifications >= 2
    
    def test_zero_treatment_effect(self):
        """Test with zero treatment effect."""
        data = generate_common_timing_data(
            n_units=100, n_periods=10, treatment_period=6,
            treatment_effect=0.0,  # No effect
            noise_std=0.5, seed=42
        )
        
        result = robustness_pre_periods(
            data, y='Y', ivar='unit', tvar='time',
            d='D', post='post', rolling='demean',
            pre_period_range=(2, 4), verbose=False
        )
        
        # ATT should be close to zero
        assert abs(result.baseline_spec.att) < 0.5

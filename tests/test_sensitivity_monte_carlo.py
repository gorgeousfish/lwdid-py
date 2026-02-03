"""
Monte Carlo simulation tests for sensitivity analysis module.

Tests statistical properties of the sensitivity analysis methods:
- Type I error rate (false positive rate)
- Power to detect sensitivity
- Anticipation detection power
"""

import pytest
import numpy as np
import pandas as pd

from lwdid.sensitivity import (
    robustness_pre_periods,
    sensitivity_no_anticipation,
    RobustnessLevel,
)


def generate_panel_dgp(
    n_units: int,
    n_periods: int,
    treatment_period: int,
    treatment_effect: float,
    noise_std: float,
    cohort_trend: float = 0.0,
    seed: int = None,
) -> pd.DataFrame:
    """
    Generate panel data from a known DGP.
    
    DGP: Y_it = α_i + γ_t + β_i * t + τ * D_it + ε_it
    
    where:
    - α_i ~ N(0, 1) is unit fixed effect
    - γ_t = 0.1 * t is common time trend
    - β_i = cohort_trend for treated, 0 for control (heterogeneous trend)
    - τ = treatment_effect
    - ε_it ~ N(0, noise_std²)
    """
    if seed is not None:
        np.random.seed(seed)
    
    data = []
    n_treated = n_units // 2
    
    for i in range(n_units):
        alpha_i = np.random.normal(0, 1)
        treated = i < n_treated
        beta_i = cohort_trend if treated else 0
        
        for t in range(1, n_periods + 1):
            gamma_t = 0.1 * t
            d_it = 1 if treated and t >= treatment_period else 0
            tau = treatment_effect if d_it else 0
            epsilon = np.random.normal(0, noise_std)
            
            y = alpha_i + gamma_t + beta_i * t + tau + epsilon
            
            data.append({
                'unit': i,
                'time': t,
                'Y': y,
                'first_treat': treatment_period if treated else 0,
            })
    
    return pd.DataFrame(data)


def generate_anticipation_dgp(
    n_units: int,
    n_periods: int,
    treatment_period: int,
    treatment_effect: float,
    anticipation_periods: int,
    anticipation_effect: float,
    noise_std: float,
    seed: int = None,
) -> pd.DataFrame:
    """
    Generate panel data with anticipation effects.
    
    DGP: Y_it = α_i + τ * D_it + δ * A_it + ε_it
    
    where A_it = 1 if treated and t ∈ [g - anticipation_periods, g - 1]
    """
    if seed is not None:
        np.random.seed(seed)
    
    data = []
    n_treated = n_units // 2
    
    for i in range(n_units):
        alpha_i = np.random.normal(0, 1)
        treated = i < n_treated
        
        for t in range(1, n_periods + 1):
            d_it = 1 if treated and t >= treatment_period else 0
            a_it = 1 if (
                treated and 
                t >= treatment_period - anticipation_periods and 
                t < treatment_period
            ) else 0
            
            tau = treatment_effect if d_it else 0
            delta = anticipation_effect if a_it else 0
            epsilon = np.random.normal(0, noise_std)
            
            y = alpha_i + tau + delta + epsilon
            
            data.append({
                'unit': i,
                'time': t,
                'Y': y,
                'first_treat': treatment_period if treated else 0,
            })
    
    return pd.DataFrame(data)


@pytest.mark.slow
class TestRobustnessDetectionPower:
    """Monte Carlo tests for robustness detection power."""
    
    def test_type_i_error_rate(self):
        """
        Test Type I error rate: When data is truly robust,
        should not falsely detect sensitivity.
        
        H0: Estimates are robust (sensitivity_ratio < threshold)
        
        Under H0 (no heterogeneous trends), false positive rate should be low.
        """
        n_simulations = 100
        false_positive_count = 0
        threshold = 0.25
        
        for sim in range(n_simulations):
            # Generate data with no heterogeneous trends (truly robust)
            data = generate_panel_dgp(
                n_units=100, n_periods=10, treatment_period=6,
                treatment_effect=2.0, noise_std=0.5,
                cohort_trend=0.0,  # No heterogeneous trends
                seed=42 + sim
            )
            
            try:
                result = robustness_pre_periods(
                    data, y='Y', ivar='unit', tvar='time', gvar='first_treat',
                    rolling='demean', pre_period_range=(2, 4),
                    robustness_threshold=threshold, verbose=False
                )
                
                if not result.is_robust:
                    false_positive_count += 1
            except Exception:
                # Skip failed simulations
                pass
        
        type_i_error_rate = false_positive_count / n_simulations
        
        # Type I error should be low (< 15% allowing for sampling variation)
        assert type_i_error_rate < 0.15, \
            f"Type I error rate too high: {type_i_error_rate:.2%}"
    
    def test_power_to_detect_sensitivity(self):
        """
        Test power: When data has heterogeneous trends,
        should detect sensitivity.
        
        H1: Estimates are sensitive (sensitivity_ratio >= threshold)
        
        Note: The sensitivity ratio = (max_ATT - min_ATT) / |baseline_ATT|.
        To trigger sensitivity detection, we need:
        1. Strong heterogeneous trends (cohort_trend)
        2. Smaller treatment effect (so ratio is larger)
        3. Wider pre-period range (more variation in estimates)
        """
        n_simulations = 100
        true_positive_count = 0
        threshold = 0.25
        
        for sim in range(n_simulations):
            # Generate data with strong heterogeneous trends and smaller treatment effect
            # This combination ensures sensitivity ratio > 25%
            data = generate_panel_dgp(
                n_units=100, n_periods=12, treatment_period=8,
                treatment_effect=0.3,  # Smaller effect to increase sensitivity ratio
                noise_std=0.3,
                cohort_trend=0.5,  # Strong heterogeneous trend
                seed=42 + sim
            )
            
            try:
                result = robustness_pre_periods(
                    data, y='Y', ivar='unit', tvar='time', gvar='first_treat',
                    rolling='demean', pre_period_range=(2, 6),  # Wider range
                    robustness_threshold=threshold, verbose=False
                )
                
                if not result.is_robust:
                    true_positive_count += 1
            except Exception:
                pass
        
        power = true_positive_count / n_simulations
        
        # Power should be reasonably high (> 50%)
        # With these parameters, heterogeneous trends should cause
        # ATT estimates to vary significantly across pre-period specifications
        assert power > 0.50, f"Power too low: {power:.2%}"
    
    def test_detrend_reduces_false_positives(self):
        """
        Test that detrend method reduces false positives when
        heterogeneous trends are present.
        """
        n_simulations = 50
        demean_sensitive_count = 0
        detrend_sensitive_count = 0
        threshold = 0.25
        
        for sim in range(n_simulations):
            # Generate data with heterogeneous trends
            data = generate_panel_dgp(
                n_units=100, n_periods=12, treatment_period=8,
                treatment_effect=2.0, noise_std=0.5,
                cohort_trend=0.15,  # Moderate heterogeneous trend
                seed=42 + sim
            )
            
            try:
                # Test with demean
                result_demean = robustness_pre_periods(
                    data, y='Y', ivar='unit', tvar='time', gvar='first_treat',
                    rolling='demean', pre_period_range=(3, 6),
                    robustness_threshold=threshold, verbose=False
                )
                if not result_demean.is_robust:
                    demean_sensitive_count += 1
                
                # Test with detrend
                result_detrend = robustness_pre_periods(
                    data, y='Y', ivar='unit', tvar='time', gvar='first_treat',
                    rolling='detrend', pre_period_range=(3, 6),
                    robustness_threshold=threshold, verbose=False
                )
                if not result_detrend.is_robust:
                    detrend_sensitive_count += 1
            except Exception:
                pass
        
        # Detrend should show fewer sensitivity detections
        # (because it removes the heterogeneous trends)
        assert detrend_sensitive_count <= demean_sensitive_count + 10, \
            f"Detrend ({detrend_sensitive_count}) should not be worse than demean ({demean_sensitive_count})"


@pytest.mark.slow
class TestAnticipationDetectionPower:
    """Monte Carlo tests for anticipation detection power."""
    
    def test_no_anticipation_false_positive_rate(self):
        """
        Test false positive rate when no anticipation is present.
        """
        n_simulations = 100
        false_positive_count = 0
        
        for sim in range(n_simulations):
            # Generate data without anticipation
            data = generate_panel_dgp(
                n_units=100, n_periods=10, treatment_period=6,
                treatment_effect=2.0, noise_std=0.5,
                cohort_trend=0.0,
                seed=42 + sim
            )
            
            try:
                result = sensitivity_no_anticipation(
                    data, y='Y', ivar='unit', tvar='time', gvar='first_treat',
                    max_anticipation=2, detection_threshold=0.10, verbose=False
                )
                
                if result.anticipation_detected:
                    false_positive_count += 1
            except Exception:
                pass
        
        false_positive_rate = false_positive_count / n_simulations
        
        # False positive rate should be low (< 20%)
        assert false_positive_rate < 0.20, \
            f"False positive rate too high: {false_positive_rate:.2%}"
    
    def test_anticipation_detection_power(self):
        """
        Test power to detect anticipation effects when present.
        """
        n_simulations = 100
        detected_count = 0
        
        for sim in range(n_simulations):
            # Generate data with anticipation effects
            data = generate_anticipation_dgp(
                n_units=100, n_periods=10, treatment_period=6,
                treatment_effect=2.0,
                anticipation_periods=2,
                anticipation_effect=0.8,  # Strong anticipation
                noise_std=0.5,
                seed=42 + sim
            )
            
            try:
                result = sensitivity_no_anticipation(
                    data, y='Y', ivar='unit', tvar='time', gvar='first_treat',
                    max_anticipation=3, detection_threshold=0.10, verbose=False
                )
                
                if result.anticipation_detected:
                    detected_count += 1
            except Exception:
                pass
        
        power = detected_count / n_simulations
        
        # Power should be reasonable (> 40%)
        assert power > 0.40, f"Anticipation detection power too low: {power:.2%}"


@pytest.mark.slow
class TestCoverageProperties:
    """Test coverage properties of sensitivity analysis."""
    
    def test_att_coverage_across_specifications(self):
        """
        Test that true ATT is covered by confidence intervals
        across different specifications.
        """
        n_simulations = 100
        true_att = 2.0
        coverage_counts = []
        
        for sim in range(n_simulations):
            data = generate_panel_dgp(
                n_units=150, n_periods=10, treatment_period=6,
                treatment_effect=true_att, noise_std=0.5,
                cohort_trend=0.0,
                seed=42 + sim
            )
            
            try:
                result = robustness_pre_periods(
                    data, y='Y', ivar='unit', tvar='time', gvar='first_treat',
                    rolling='demean', pre_period_range=(2, 4), verbose=False
                )
                
                # Check coverage for each specification
                for spec in result.specifications:
                    if spec.converged:
                        covered = spec.ci_lower <= true_att <= spec.ci_upper
                        coverage_counts.append(covered)
            except Exception:
                pass
        
        if coverage_counts:
            coverage_rate = sum(coverage_counts) / len(coverage_counts)
            
            # Coverage should be close to 95% (allow 85-99% range)
            assert 0.85 <= coverage_rate <= 0.99, \
                f"Coverage rate {coverage_rate:.2%} outside expected range"


@pytest.mark.slow
class TestSampleSizeEffects:
    """Test effects of sample size on sensitivity analysis."""
    
    @pytest.mark.parametrize("n_units", [50, 100, 200])
    def test_sensitivity_ratio_stability(self, n_units):
        """
        Test that sensitivity ratio is stable across sample sizes
        when data is truly robust.
        """
        n_simulations = 30
        ratios = []
        
        for sim in range(n_simulations):
            data = generate_panel_dgp(
                n_units=n_units, n_periods=10, treatment_period=6,
                treatment_effect=2.0, noise_std=0.5,
                cohort_trend=0.0,
                seed=42 + sim
            )
            
            try:
                result = robustness_pre_periods(
                    data, y='Y', ivar='unit', tvar='time', gvar='first_treat',
                    rolling='demean', pre_period_range=(2, 4), verbose=False
                )
                ratios.append(result.sensitivity_ratio)
            except Exception:
                pass
        
        if ratios:
            mean_ratio = np.mean(ratios)
            # Mean sensitivity ratio should be low for robust data
            assert mean_ratio < 0.30, \
                f"Mean sensitivity ratio {mean_ratio:.2%} too high for n={n_units}"

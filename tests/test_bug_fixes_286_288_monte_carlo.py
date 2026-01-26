"""
Monte Carlo simulation tests for BUG-286, BUG-287, BUG-288.

This module uses Monte Carlo simulations to validate:
1. BUG-286: t_stat boundary handling (SE=0/Inf scenarios)
2. Type I error rate and coverage probability
3. Numerical stability under various DGPs
"""

import warnings
import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lwdid import lwdid
from lwdid.results import _compute_t_stat


# =============================================================================
# Data Generation Functions
# =============================================================================

def generate_dgp_common_timing(n_units=100, n_periods=6, true_effect=2.0, seed=None):
    """Generate data from common timing DiD DGP."""
    if seed is not None:
        np.random.seed(seed)
    
    data = []
    for i in range(1, n_units + 1):
        d = 1 if i <= n_units // 2 else 0
        for t in range(2000, 2000 + n_periods):
            post = 1 if t >= 2003 else 0
            y = 10 + true_effect * d * post + np.random.normal(0, 1)
            data.append({'unit': i, 'year': t, 'd': d, 'post': post, 'y': y})
    
    return pd.DataFrame(data)


def generate_dgp_with_degenerate_variance(n_units=50, seed=None):
    """Generate data where some groups have zero variance (edge case for SE)."""
    if seed is not None:
        np.random.seed(seed)
    
    data = []
    for i in range(1, n_units + 1):
        d = 1 if i <= n_units // 2 else 0
        for t in range(2000, 2006):
            post = 1 if t >= 2003 else 0
            # Make control group have constant outcome in post period
            if d == 0 and post == 1:
                y = 10.0  # Constant - no variation
            else:
                y = 10 + 2.0 * d * post + np.random.normal(0, 1)
            data.append({'unit': i, 'year': t, 'd': d, 'post': post, 'y': y})
    
    return pd.DataFrame(data)


# =============================================================================
# BUG-286: Monte Carlo Tests for t_stat Boundary Handling
# =============================================================================

class TestBug286MonteCarlo:
    """Monte Carlo tests for t_stat boundary handling."""
    
    def test_tstat_helper_function_vectorized(self):
        """Test _compute_t_stat with various SE values in batch."""
        test_cases = [
            # (att, se, expected_is_nan, expected_is_zero, expected_warning)
            (2.0, 1.0, False, False, False),   # Normal case
            (2.0, 0.0, True, False, True),     # SE=0 -> NaN + warning
            (2.0, np.inf, False, True, True),  # SE=Inf -> 0 + warning
            (2.0, np.nan, True, False, False), # SE=NaN -> NaN, no warning
            (np.nan, 1.0, True, False, False), # ATT=NaN -> NaN, no warning
            (0.0, 1.0, False, True, False),    # ATT=0 -> 0
        ]
        
        for att, se, expect_nan, expect_zero, expect_warning in test_cases:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = _compute_t_stat(att, se, warn_on_boundary=True)
                
                if expect_nan:
                    assert np.isnan(result), f"Expected NaN for att={att}, se={se}"
                elif expect_zero:
                    assert result == 0.0, f"Expected 0.0 for att={att}, se={se}"
                
                if expect_warning:
                    assert len(w) > 0, f"Expected warning for att={att}, se={se}"
                else:
                    # NaN inputs don't trigger warnings
                    if not (np.isnan(att) if isinstance(att, float) else False) and \
                       not (np.isnan(se) if isinstance(se, float) else False):
                        pass  # Other cases handled above
    
    def test_monte_carlo_coverage_probability(self):
        """Monte Carlo test: 95% CI should cover true effect ~95% of time."""
        n_simulations = 100
        true_effect = 2.0
        alpha = 0.05
        coverage_count = 0
        
        for sim in range(n_simulations):
            df = generate_dgp_common_timing(n_units=100, true_effect=true_effect, seed=42 + sim)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = lwdid(
                    data=df,
                    y='y',
                    d='d',
                    ivar='unit',
                    tvar='year',
                    post='post',
                    rolling='demean',
                    alpha=alpha,
                )
            
            # Check if true effect is within CI
            if result.ci_lower <= true_effect <= result.ci_upper:
                coverage_count += 1
        
        coverage_rate = coverage_count / n_simulations
        
        # Coverage should be approximately 95% (+/- 10% for simulation variance)
        assert 0.85 < coverage_rate < 1.0, f"Coverage rate {coverage_rate} outside expected range"
    
    def test_monte_carlo_type_i_error(self):
        """Monte Carlo test: Type I error rate under null hypothesis."""
        n_simulations = 100
        true_effect = 0.0  # Null hypothesis
        alpha = 0.05
        rejection_count = 0
        
        for sim in range(n_simulations):
            df = generate_dgp_common_timing(n_units=100, true_effect=true_effect, seed=123 + sim)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = lwdid(
                    data=df,
                    y='y',
                    d='d',
                    ivar='unit',
                    tvar='year',
                    post='post',
                    rolling='demean',
                    alpha=alpha,
                )
            
            # Check if null is rejected (p-value < alpha)
            if result.pvalue < alpha:
                rejection_count += 1
        
        type_i_error_rate = rejection_count / n_simulations
        
        # Type I error should be approximately 5% (+/- 5% for simulation variance)
        assert type_i_error_rate < 0.15, f"Type I error rate {type_i_error_rate} too high"
    
    def test_monte_carlo_estimate_bias(self):
        """Monte Carlo test: ATT estimate should be unbiased."""
        n_simulations = 100
        true_effect = 2.0
        estimates = []
        
        for sim in range(n_simulations):
            df = generate_dgp_common_timing(n_units=100, true_effect=true_effect, seed=456 + sim)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = lwdid(
                    data=df,
                    y='y',
                    d='d',
                    ivar='unit',
                    tvar='year',
                    post='post',
                    rolling='demean',
                )
            
            estimates.append(result.att)
        
        mean_estimate = np.mean(estimates)
        std_estimate = np.std(estimates)
        
        # Mean should be close to true effect (within 0.2)
        assert abs(mean_estimate - true_effect) < 0.2, \
            f"Mean estimate {mean_estimate} not close to true effect {true_effect}"
        
        # Standard deviation should be reasonable
        assert 0 < std_estimate < 1.0, f"Std estimate {std_estimate} unreasonable"


# =============================================================================
# BUG-287/288: Parameter Validation Monte Carlo Tests
# =============================================================================

class TestParameterValidationMonteCarlo:
    """Monte Carlo tests for parameter validation stability."""
    
    def test_ri_method_stability_across_simulations(self):
        """ri_method validation should be consistent across simulations."""
        n_simulations = 10
        
        for sim in range(n_simulations):
            df = generate_dgp_common_timing(n_units=50, seed=789 + sim)
            
            # Valid ri_method should always work
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = lwdid(
                    data=df,
                    y='y',
                    d='d',
                    ivar='unit',
                    tvar='year',
                    post='post',
                    rolling='demean',
                    ri=True,
                    rireps=50,
                    ri_method='bootstrap',
                    seed=42,
                )
            
            assert result.att is not None
            
            # Invalid ri_method should always fail
            with pytest.raises(TypeError):
                lwdid(
                    data=df,
                    y='y',
                    d='d',
                    ivar='unit',
                    tvar='year',
                    post='post',
                    rolling='demean',
                    ri=True,
                    rireps=50,
                    ri_method=123,  # Invalid type
                    seed=42,
                )
    
    def test_trim_threshold_stability_across_simulations(self):
        """trim_threshold validation should be consistent across simulations."""
        n_simulations = 5
        
        for sim in range(n_simulations):
            # Generate data with time-invariant controls
            np.random.seed(999 + sim)
            data = []
            for i in range(1, 51):
                d = 1 if i <= 25 else 0
                x1 = np.random.normal(0, 1)  # Time-invariant
                for t in range(2000, 2006):
                    post = 1 if t >= 2003 else 0
                    y = 10 + 2 * d * post + 0.5 * x1 + np.random.normal(0, 1)
                    data.append({'unit': i, 'year': t, 'd': d, 'post': post, 'y': y, 'x1': x1})
            df = pd.DataFrame(data)
            
            # Valid trim_threshold should work
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = lwdid(
                    data=df,
                    y='y',
                    d='d',
                    ivar='unit',
                    tvar='year',
                    post='post',
                    rolling='demean',
                    estimator='ipw',
                    controls=['x1'],
                    trim_threshold=0.05,
                )
            
            assert result.att is not None
            
            # Invalid trim_threshold should always fail
            with pytest.raises(ValueError):
                lwdid(
                    data=df,
                    y='y',
                    d='d',
                    ivar='unit',
                    tvar='year',
                    post='post',
                    rolling='demean',
                    estimator='ipw',
                    controls=['x1'],
                    trim_threshold=0.6,  # Invalid: > 0.5
                )


# =============================================================================
# Numerical Stability Monte Carlo Tests
# =============================================================================

class TestNumericalStabilityMonteCarlo:
    """Monte Carlo tests for numerical stability."""
    
    def test_no_nan_estimates_under_normal_dgp(self):
        """Under normal DGP, estimates should never be NaN."""
        n_simulations = 50
        nan_count = 0
        
        for sim in range(n_simulations):
            df = generate_dgp_common_timing(n_units=100, seed=111 + sim)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = lwdid(
                    data=df,
                    y='y',
                    d='d',
                    ivar='unit',
                    tvar='year',
                    post='post',
                    rolling='demean',
                )
            
            if np.isnan(result.att):
                nan_count += 1
        
        assert nan_count == 0, f"{nan_count} simulations produced NaN ATT"
    
    def test_se_always_positive_under_normal_dgp(self):
        """Under normal DGP, SE should always be positive."""
        n_simulations = 50
        negative_se_count = 0
        
        for sim in range(n_simulations):
            df = generate_dgp_common_timing(n_units=100, seed=222 + sim)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = lwdid(
                    data=df,
                    y='y',
                    d='d',
                    ivar='unit',
                    tvar='year',
                    post='post',
                    rolling='demean',
                )
            
            if result.se_att <= 0:
                negative_se_count += 1
        
        assert negative_se_count == 0, f"{negative_se_count} simulations produced non-positive SE"
    
    def test_tstat_consistency_with_manual_calculation(self):
        """t_stat should always equal att/se_att for valid SE."""
        n_simulations = 50
        inconsistency_count = 0
        
        for sim in range(n_simulations):
            df = generate_dgp_common_timing(n_units=100, seed=333 + sim)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = lwdid(
                    data=df,
                    y='y',
                    d='d',
                    ivar='unit',
                    tvar='year',
                    post='post',
                    rolling='demean',
                )
            
            expected_t_stat = result.att / result.se_att
            if abs(result.t_stat - expected_t_stat) > 1e-10:
                inconsistency_count += 1
        
        assert inconsistency_count == 0, \
            f"{inconsistency_count} simulations had inconsistent t_stat"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

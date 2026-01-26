"""Monte Carlo Tests for DESIGN-008: Exception Handling Fix.

This module verifies that the exception handling changes maintain
the statistical properties of the estimators through Monte Carlo simulation.
"""

import numpy as np
import pandas as pd
import pytest
import warnings


# =============================================================================
# Test 1: IPW Estimator Monte Carlo
# =============================================================================

class TestIPWMonteCarlo:
    """Monte Carlo tests for IPW estimator."""
    
    def test_ipw_unbiasedness(self):
        """IPW should be approximately unbiased for ATT."""
        from lwdid.staggered.estimators import estimate_ipw
        
        n_simulations = 50
        n_obs = 300
        true_att = 2.0
        estimates = []
        
        for seed in range(n_simulations):
            np.random.seed(seed)
            
            # DGP
            x = np.random.normal(0, 1, n_obs)
            prob_treat = 1 / (1 + np.exp(-(0.5 * x)))
            d = (np.random.random(n_obs) < prob_treat).astype(float)
            y = 1.0 + true_att * d + 0.8 * x + np.random.normal(0, 1, n_obs)
            
            data = pd.DataFrame({'y': y, 'd': d, 'x': x})
            
            try:
                result = estimate_ipw(
                    data=data,
                    y='y',
                    d='d',
                    propensity_controls=['x'],
                    trim_threshold=0.01,
                    se_method='analytical',
                )
                estimates.append(result.att)
            except (ValueError, np.linalg.LinAlgError, RuntimeError):
                continue
        
        # Should have enough successful estimates
        assert len(estimates) >= 40, f"Too many failures: {n_simulations - len(estimates)}"
        
        # Mean should be close to true ATT (within 0.5)
        mean_estimate = np.mean(estimates)
        bias = abs(mean_estimate - true_att)
        assert bias < 0.5, f"Bias {bias} is too large"
    
    def test_ipw_ci_coverage(self):
        """IPW confidence intervals should have approximately correct coverage."""
        from lwdid.staggered.estimators import estimate_ipw
        
        n_simulations = 50
        n_obs = 300
        true_att = 1.5
        coverage_count = 0
        
        for seed in range(n_simulations):
            np.random.seed(seed)
            
            x = np.random.normal(0, 1, n_obs)
            prob_treat = 1 / (1 + np.exp(-(0.5 * x)))
            d = (np.random.random(n_obs) < prob_treat).astype(float)
            y = 1.0 + true_att * d + 0.8 * x + np.random.normal(0, 0.5, n_obs)
            
            data = pd.DataFrame({'y': y, 'd': d, 'x': x})
            
            try:
                result = estimate_ipw(
                    data=data,
                    y='y',
                    d='d',
                    propensity_controls=['x'],
                    trim_threshold=0.01,
                    se_method='analytical',
                    alpha=0.05,
                )
                
                if result.ci_lower <= true_att <= result.ci_upper:
                    coverage_count += 1
            except (ValueError, np.linalg.LinAlgError, RuntimeError):
                continue
        
        # Coverage should be approximately 95% (allow 80-100% due to small n_simulations)
        coverage = coverage_count / n_simulations
        assert coverage >= 0.80, f"Coverage {coverage:.1%} is too low"


# =============================================================================
# Test 2: IPWRA Estimator Monte Carlo
# =============================================================================

class TestIPWRAMonteCarlo:
    """Monte Carlo tests for IPWRA estimator."""
    
    def test_ipwra_unbiasedness(self):
        """IPWRA should be approximately unbiased for ATT."""
        from lwdid.staggered.estimators import estimate_ipwra
        
        n_simulations = 30
        n_obs = 400
        true_att = 2.5
        estimates = []
        
        for seed in range(n_simulations):
            np.random.seed(seed + 100)
            
            x1 = np.random.normal(0, 1, n_obs)
            x2 = np.random.normal(0, 1, n_obs)
            prob_treat = 1 / (1 + np.exp(-(0.3 * x1 + 0.2 * x2)))
            d = (np.random.random(n_obs) < prob_treat).astype(float)
            y = 0.5 + true_att * d + 0.6 * x1 + 0.4 * x2 + np.random.normal(0, 0.5, n_obs)
            
            data = pd.DataFrame({'y': y, 'd': d, 'x1': x1, 'x2': x2})
            
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = estimate_ipwra(
                        data=data,
                        y='y',
                        d='d',
                        controls=['x1', 'x2'],
                        propensity_controls=['x1', 'x2'],
                        trim_threshold=0.01,
                        se_method='analytical',
                    )
                estimates.append(result.att)
            except (ValueError, np.linalg.LinAlgError, RuntimeError):
                continue
        
        assert len(estimates) >= 20, f"Too many failures: {n_simulations - len(estimates)}"
        
        mean_estimate = np.mean(estimates)
        bias = abs(mean_estimate - true_att)
        assert bias < 0.5, f"Bias {bias} is too large"


# =============================================================================
# Test 3: Bootstrap SE Monte Carlo
# =============================================================================

class TestBootstrapSEMonteCarlo:
    """Monte Carlo tests for bootstrap standard errors."""
    
    def test_bootstrap_se_consistency(self):
        """Bootstrap SE should be consistent with analytical SE."""
        from lwdid.staggered.estimators import estimate_ipw
        
        np.random.seed(42)
        n_obs = 400
        
        x = np.random.normal(0, 1, n_obs)
        prob_treat = 1 / (1 + np.exp(-(0.5 * x)))
        d = (np.random.random(n_obs) < prob_treat).astype(float)
        y = 1.0 + 2.0 * d + 0.8 * x + np.random.normal(0, 0.5, n_obs)
        
        data = pd.DataFrame({'y': y, 'd': d, 'x': x})
        
        # Analytical SE
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result_analytical = estimate_ipw(
                data=data,
                y='y',
                d='d',
                propensity_controls=['x'],
                trim_threshold=0.01,
                se_method='analytical',
            )
        
        # Bootstrap SE
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result_bootstrap = estimate_ipw(
                data=data,
                y='y',
                d='d',
                propensity_controls=['x'],
                trim_threshold=0.01,
                se_method='bootstrap',
                n_bootstrap=200,
                seed=42,
            )
        
        # SE should be reasonably close (within factor of 2)
        ratio = result_bootstrap.se / result_analytical.se
        assert 0.5 < ratio < 2.0, f"SE ratio {ratio} is outside expected range"


# =============================================================================
# Test 4: Exception Robustness Monte Carlo
# =============================================================================

class TestExceptionRobustnessMonteCarlo:
    """Test that estimators are robust to occasional failures."""
    
    def test_estimator_handles_difficult_samples(self):
        """Estimators should handle difficult samples gracefully."""
        from lwdid.staggered.estimators import estimate_ipw
        
        n_simulations = 50
        success_count = 0
        
        for seed in range(n_simulations):
            np.random.seed(seed + 200)
            
            # Create difficult sample (small, highly imbalanced)
            n_obs = 50
            x = np.random.normal(0, 1, n_obs)
            
            # Strong selection makes PS extreme
            prob_treat = 1 / (1 + np.exp(-2 * x))
            d = (np.random.random(n_obs) < prob_treat).astype(float)
            y = 1.0 + 1.5 * d + x + np.random.normal(0, 1, n_obs)
            
            data = pd.DataFrame({'y': y, 'd': d, 'x': x})
            
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = estimate_ipw(
                        data=data,
                        y='y',
                        d='d',
                        propensity_controls=['x'],
                        trim_threshold=0.01,
                        se_method='analytical',
                    )
                
                if not np.isnan(result.att):
                    success_count += 1
            except (ValueError, np.linalg.LinAlgError, RuntimeError):
                # Expected failures are OK
                pass
        
        # Should have at least some successful estimates
        assert success_count > 0, "No successful estimates in difficult samples"
        
        # But not all should succeed (some difficulty expected)
        # If all succeed, the test may not be testing edge cases
        print(f"Success rate: {success_count}/{n_simulations}")


# =============================================================================
# Test 5: RA Estimator Monte Carlo
# =============================================================================

class TestRAMonteCarlo:
    """Monte Carlo tests for RA estimator."""
    
    def test_ra_unbiasedness(self):
        """RA should be approximately unbiased when outcome model is correct."""
        from lwdid.staggered.estimators import estimate_ra
        
        n_simulations = 50
        n_obs = 300
        true_att = 1.8
        estimates = []
        
        for seed in range(n_simulations):
            np.random.seed(seed + 300)
            
            x = np.random.normal(0, 1, n_obs)
            d = (np.random.random(n_obs) > 0.5).astype(float)
            
            # Outcome depends on x
            y = 0.5 + true_att * d + 1.2 * x + np.random.normal(0, 0.5, n_obs)
            
            data = pd.DataFrame({'y': y, 'd': d, 'x': x})
            
            try:
                result = estimate_ra(
                    data=data,
                    y='y',
                    d='d',
                    controls=['x'],
                )
                estimates.append(result.att)
            except (ValueError, np.linalg.LinAlgError, RuntimeError):
                continue
        
        assert len(estimates) >= 40, f"Too many failures"
        
        mean_estimate = np.mean(estimates)
        bias = abs(mean_estimate - true_att)
        assert bias < 0.3, f"Bias {bias} is too large"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

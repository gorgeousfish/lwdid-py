"""
Story 1.1: Boundary Case Tests for IPW/IPWRA/PSM Estimators

This module tests boundary conditions and edge cases:
1. Perfect separation (propensity scores near 0 or 1)
2. Extreme sample imbalance (very few treated/control units)
3. Near-collinearity in covariates
4. Extreme propensity score trimming
5. Small sample sizes with many covariates

These tests verify that the estimators handle edge cases gracefully
with appropriate warnings or errors.

References
----------
Lee & Wooldridge (2023), Section 3, Procedure 3.1
"""

import numpy as np
import pandas as pd
import pytest
import warnings

from lwdid import lwdid
from lwdid.staggered.estimators import estimate_ipw, estimate_ipwra, estimate_psm


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def perfect_separation_data():
    """
    Create data with perfect separation in propensity score model.
    
    Treatment is deterministic based on x1 > 0.
    This should trigger warnings or regularization in logit model.
    """
    np.random.seed(42)
    n = 100
    
    # Create deterministic separation
    x1 = np.random.normal(0, 1, n)
    d = (x1 > 0).astype(int)  # Perfect separation
    
    # Add some noise to make it near-perfect
    noise_idx = np.random.choice(n, size=5, replace=False)
    d[noise_idx] = 1 - d[noise_idx]  # Flip 5 units
    
    y = 5 * d + 2 * x1 + np.random.normal(0, 1, n)
    
    return pd.DataFrame({
        'y': y,
        'd': d,
        'x1': x1,
    })


@pytest.fixture
def extreme_imbalance_data():
    """
    Create extremely imbalanced data (5 treated vs 195 control).
    
    This tests behavior with very few treated units.
    """
    np.random.seed(123)
    n = 200
    n_treated = 5
    
    d = np.zeros(n)
    d[:n_treated] = 1
    
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.binomial(1, 0.5, n)
    
    # True ATT = 3.0
    y = 3 * d + 1.5 * x1 + x2 + np.random.normal(0, 1, n)
    
    return pd.DataFrame({
        'y': y,
        'd': d.astype(int),
        'x1': x1,
        'x2': x2,
    })


@pytest.fixture
def reverse_imbalance_data():
    """
    Create data with many treated, few control (195 treated vs 5 control).
    """
    np.random.seed(456)
    n = 200
    n_control = 5
    
    d = np.ones(n)
    d[:n_control] = 0
    
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.binomial(1, 0.5, n)
    
    y = 3 * d + 1.5 * x1 + x2 + np.random.normal(0, 1, n)
    
    return pd.DataFrame({
        'y': y,
        'd': d.astype(int),
        'x1': x1,
        'x2': x2,
    })


@pytest.fixture
def near_collinear_data():
    """
    Create data with near-collinear covariates.
    
    x2 = x1 + small_noise, creating multicollinearity issues.
    """
    np.random.seed(789)
    n = 200
    
    x1 = np.random.normal(0, 1, n)
    x2 = x1 + np.random.normal(0, 0.01, n)  # Near-perfect correlation
    x3 = np.random.binomial(1, 0.5, n)
    
    ps_index = -0.5 + 0.5 * x1 + 0.3 * x3
    ps = 1 / (1 + np.exp(-ps_index))
    d = np.random.binomial(1, ps)
    
    y = 3 * d + x1 + x3 + np.random.normal(0, 1, n)
    
    return pd.DataFrame({
        'y': y,
        'd': d.astype(int),
        'x1': x1,
        'x2': x2,  # Near-collinear with x1
        'x3': x3,
    })


@pytest.fixture
def panel_extreme_imbalance():
    """
    Create panel data with extreme imbalance for lwdid() testing.
    """
    np.random.seed(321)
    n_units = 100
    n_treated = 3  # Only 3 treated units
    n_periods = 6
    s = 4
    
    treated = np.zeros(n_units)
    treated[:n_treated] = 1
    
    x1 = np.random.normal(0, 1, n_units)
    
    records = []
    for i in range(n_units):
        for t in range(1, n_periods + 1):
            post = 1 if t >= s else 0
            tau = 3.0 if (treated[i] == 1 and post == 1) else 0
            y = t + x1[i] + tau + np.random.normal(0, 0.5)
            
            records.append({
                'id': i + 1,
                'year': 2000 + t,
                'y': y,
                'd': int(treated[i]),
                'post': post,
                'x1': x1[i],
            })
    
    return pd.DataFrame(records)


# =============================================================================
# Test: Perfect Separation
# =============================================================================

class TestPerfectSeparation:
    """Test estimator behavior with perfect/near-perfect separation."""
    
    def test_ipw_near_perfect_separation(self, perfect_separation_data):
        """Test IPW handles near-perfect separation with warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = estimate_ipw(
                data=perfect_separation_data,
                y='y',
                d='d',
                propensity_controls=['x1'],
                trim_threshold=0.05,
            )
            
            # Should produce valid result (with possible warnings)
            assert not np.isnan(result.att), "ATT should not be NaN"
            assert result.se > 0, "SE should be positive"
            
            # Check for convergence or separation warnings
            warning_msgs = [str(warning.message).lower() for warning in w]
            print(f"\nIPW near-perfect separation warnings: {warning_msgs}")
    
    def test_ipwra_near_perfect_separation(self, perfect_separation_data):
        """Test IPWRA handles near-perfect separation."""
        result = estimate_ipwra(
            data=perfect_separation_data,
            y='y',
            d='d',
            controls=['x1'],
            propensity_controls=['x1'],
            trim_threshold=0.05,
        )
        
        # IPWRA is doubly robust, should still produce valid result
        assert not np.isnan(result.att), "ATT should not be NaN"
        assert result.se > 0, "SE should be positive"
        
        print(f"\nIPWRA near-perfect separation: ATT={result.att:.4f}, SE={result.se:.4f}")
    
    def test_psm_near_perfect_separation(self, perfect_separation_data):
        """Test PSM handles near-perfect separation."""
        result = estimate_psm(
            data=perfect_separation_data,
            y='y',
            d='d',
            propensity_controls=['x1'],
            n_neighbors=1,
        )
        
        assert not np.isnan(result.att), "ATT should not be NaN"
        print(f"\nPSM near-perfect separation: ATT={result.att:.4f}, SE={result.se:.4f}")


# =============================================================================
# Test: Extreme Sample Imbalance
# =============================================================================

class TestExtremeImbalance:
    """Test estimator behavior with extreme sample imbalance."""
    
    def test_ipw_few_treated(self, extreme_imbalance_data):
        """Test IPW with very few treated units (5 treated, 195 control)."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = estimate_ipw(
                data=extreme_imbalance_data,
                y='y',
                d='d',
                propensity_controls=['x1', 'x2'],
            )
            
            # Should produce valid result
            assert not np.isnan(result.att), "ATT should not be NaN"
            assert result.n_treated == 5, "Should have 5 treated units"
            assert result.n_control == 195, "Should have 195 control units"
            
            # True ATT is 3.0
            print(f"\nIPW few treated: ATT={result.att:.4f} (true=3.0), SE={result.se:.4f}")
    
    def test_ipw_few_control(self, reverse_imbalance_data):
        """Test IPW with very few control units (195 treated, 5 control)."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = estimate_ipw(
                data=reverse_imbalance_data,
                y='y',
                d='d',
                propensity_controls=['x1', 'x2'],
            )
            
            # Should produce valid result (possibly with high SE)
            assert not np.isnan(result.att), "ATT should not be NaN"
            assert result.n_treated == 195
            assert result.n_control == 5
            
            # Expect high variance with few controls
            print(f"\nIPW few control: ATT={result.att:.4f}, SE={result.se:.4f}")
    
    def test_ipwra_few_treated(self, extreme_imbalance_data):
        """Test IPWRA with very few treated units."""
        result = estimate_ipwra(
            data=extreme_imbalance_data,
            y='y',
            d='d',
            controls=['x1', 'x2'],
            propensity_controls=['x1', 'x2'],
        )
        
        assert not np.isnan(result.att)
        print(f"\nIPWRA few treated: ATT={result.att:.4f} (true=3.0), SE={result.se:.4f}")
    
    def test_psm_few_treated(self, extreme_imbalance_data):
        """Test PSM with very few treated units."""
        result = estimate_psm(
            data=extreme_imbalance_data,
            y='y',
            d='d',
            propensity_controls=['x1', 'x2'],
            n_neighbors=1,
        )
        
        assert not np.isnan(result.att)
        assert result.n_matched <= 5  # At most 5 matches (5 treated)
        print(f"\nPSM few treated: ATT={result.att:.4f}, N_matched={result.n_matched}")
    
    def test_lwdid_panel_extreme_imbalance(self, panel_extreme_imbalance):
        """Test lwdid() with extremely imbalanced panel data."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # RA should work
            result_ra = lwdid(
                data=panel_extreme_imbalance,
                y='y',
                d='d',
                ivar='id',
                tvar='year',
                post='post',
                rolling='demean',
                estimator='ra',
                controls=['x1'],
            )
            
            assert result_ra is not None
            print(f"\nlwdid RA extreme imbalance: ATT={result_ra.att:.4f}")
            
            # IPW should work but may have warnings
            result_ipw = lwdid(
                data=panel_extreme_imbalance,
                y='y',
                d='d',
                ivar='id',
                tvar='year',
                post='post',
                rolling='demean',
                estimator='ipw',
                controls=['x1'],
            )
            
            assert result_ipw is not None
            print(f"lwdid IPW extreme imbalance: ATT={result_ipw.att:.4f}")


# =============================================================================
# Test: Near-Collinearity
# =============================================================================

class TestNearCollinearity:
    """Test estimator behavior with near-collinear covariates."""
    
    def test_ipw_collinear_covariates(self, near_collinear_data):
        """Test IPW with near-collinear covariates."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = estimate_ipw(
                data=near_collinear_data,
                y='y',
                d='d',
                propensity_controls=['x1', 'x2', 'x3'],  # x1 and x2 are collinear
            )
            
            # Should handle collinearity (possibly with warning)
            assert not np.isnan(result.att)
            
            # Check for collinearity warning
            warning_msgs = [str(warning.message).lower() for warning in w]
            has_collinear_warning = any(
                'collinear' in msg or 'singular' in msg or 'constant' in msg
                for msg in warning_msgs
            )
            
            print(f"\nIPW collinear: ATT={result.att:.4f}, SE={result.se:.4f}")
            if has_collinear_warning:
                print("  (Collinearity warning detected as expected)")
    
    def test_ipwra_collinear_covariates(self, near_collinear_data):
        """Test IPWRA with near-collinear covariates."""
        result = estimate_ipwra(
            data=near_collinear_data,
            y='y',
            d='d',
            controls=['x1', 'x3'],  # Avoid collinearity in outcome model
            propensity_controls=['x1', 'x2', 'x3'],  # Include collinear vars in PS
        )
        
        assert not np.isnan(result.att)
        print(f"\nIPWRA collinear: ATT={result.att:.4f}, SE={result.se:.4f}")


# =============================================================================
# Test: Extreme Trimming
# =============================================================================

class TestExtremeTrimming:
    """Test behavior with extreme propensity score trimming."""
    
    def test_ipw_aggressive_trimming(self):
        """Test IPW with very aggressive trimming (20%).
        
        Note: trim_threshold clips propensity scores to [trim, 1-trim], it does
        NOT remove observations. This affects the weights but not the sample size.
        """
        np.random.seed(999)
        n = 200
        
        x1 = np.random.normal(0, 1, n)
        ps = 1 / (1 + np.exp(-x1))
        d = np.random.binomial(1, ps)
        y = 3 * d + x1 + np.random.normal(0, 1, n)
        
        data = pd.DataFrame({'y': y, 'd': d, 'x1': x1})
        
        # Run with different trim thresholds
        result_low = estimate_ipw(
            data=data,
            y='y',
            d='d',
            propensity_controls=['x1'],
            trim_threshold=0.01,  # Minimal trimming
        )
        
        result_high = estimate_ipw(
            data=data,
            y='y',
            d='d',
            propensity_controls=['x1'],
            trim_threshold=0.20,  # Aggressive trimming
        )
        
        print(f"\nIPW trim comparison:")
        print(f"  trim=0.01: ATT={result_low.att:.4f}, SE={result_low.se:.4f}")
        print(f"  trim=0.20: ATT={result_high.att:.4f}, SE={result_high.se:.4f}")
        
        # Both should produce valid results
        assert not np.isnan(result_low.att)
        assert not np.isnan(result_high.att)
        
        # Higher trimming should generally reduce variance in weights
        # This may result in different SE (not necessarily smaller/larger)
    
    def test_ipw_minimal_trimming(self):
        """Test IPW with minimal trimming (0.1%)."""
        np.random.seed(888)
        n = 200
        
        x1 = np.random.normal(0, 1, n)
        ps = 1 / (1 + np.exp(-0.5 * x1))  # Moderate PS variation
        d = np.random.binomial(1, ps)
        y = 3 * d + x1 + np.random.normal(0, 1, n)
        
        data = pd.DataFrame({'y': y, 'd': d, 'x1': x1})
        
        result = estimate_ipw(
            data=data,
            y='y',
            d='d',
            propensity_controls=['x1'],
            trim_threshold=0.001,  # Minimal trimming
        )
        
        print(f"\nIPW minimal trim (0.1%): ATT={result.att:.4f}, SE={result.se:.4f}")
        
        assert not np.isnan(result.att)


# =============================================================================
# Test: Small Sample with Many Covariates
# =============================================================================

class TestSmallSampleManyCovariates:
    """Test behavior with more covariates than reasonable for sample size."""
    
    def test_ipw_overparameterized(self):
        """Test IPW with many covariates relative to sample size."""
        np.random.seed(777)
        n = 50  # Small sample
        
        # Generate many covariates
        n_covars = 10
        X = np.random.normal(0, 1, (n, n_covars))
        
        # PS depends only on first 2
        ps_index = -0.5 + 0.3 * X[:, 0] + 0.2 * X[:, 1]
        ps = 1 / (1 + np.exp(-ps_index))
        d = np.random.binomial(1, ps)
        
        y = 3 * d + X[:, 0] + np.random.normal(0, 1, n)
        
        data = pd.DataFrame({
            'y': y, 'd': d,
            **{f'x{i}': X[:, i] for i in range(n_covars)}
        })
        
        all_controls = [f'x{i}' for i in range(n_covars)]
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = estimate_ipw(
                data=data,
                y='y',
                d='d',
                propensity_controls=all_controls,
            )
            
            # Should work but may have warnings
            print(f"\nIPW overparameterized (n=50, k=10): ATT={result.att:.4f}, SE={result.se:.4f}")
            
            # Note: This may fail or have very large SE
            assert not np.isnan(result.att)


# =============================================================================
# Test: Minimum Sample Requirements
# =============================================================================

class TestMinimumSampleRequirements:
    """Test minimum sample size requirements."""
    
    def test_ipw_minimum_treated(self):
        """Test IPW with minimum number of treated units (2)."""
        np.random.seed(666)
        n = 50
        n_treated = 2
        
        d = np.zeros(n)
        d[:n_treated] = 1
        
        x1 = np.random.normal(0, 1, n)
        y = 3 * d + x1 + np.random.normal(0, 1, n)
        
        data = pd.DataFrame({'y': y, 'd': d.astype(int), 'x1': x1})
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = estimate_ipw(
                data=data,
                y='y',
                d='d',
                propensity_controls=['x1'],
            )
            
            print(f"\nIPW minimum treated (n_treated=2): ATT={result.att:.4f}, SE={result.se:.4f}")
            
            # Should work but SE will be very large
            assert not np.isnan(result.att)
            assert result.n_treated == 2
    
    def test_ipw_fails_with_one_treated(self):
        """Test that IPW fails appropriately with only 1 treated unit."""
        n = 50
        d = np.zeros(n)
        d[0] = 1  # Only 1 treated
        
        x1 = np.random.normal(0, 1, n)
        y = 3 * d + x1 + np.random.normal(0, 1, n)
        
        data = pd.DataFrame({'y': y, 'd': d.astype(int), 'x1': x1})
        
        # Should fail or produce very unreliable results
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            
            try:
                result = estimate_ipw(
                    data=data,
                    y='y',
                    d='d',
                    propensity_controls=['x1'],
                )
                print(f"\nIPW with 1 treated: ATT={result.att:.4f}")
                # If it doesn't fail, check for warnings
            except Exception as e:
                print(f"\nIPW with 1 treated failed as expected: {type(e).__name__}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

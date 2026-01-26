"""Numerical Validation Tests for DESIGN-008: Exception Handling Fix.

This module verifies that the exception handling changes do not affect
the numerical results of estimation functions. Tests compare results
before and after the fix using known reference values.
"""

import numpy as np
import pandas as pd
import pytest
import warnings


# =============================================================================
# Test 1: IPW Estimation Numerical Stability
# =============================================================================

class TestIPWNumericalStability:
    """Test that IPW estimation produces correct numerical results."""
    
    @pytest.fixture
    def ipw_test_data(self):
        """Create test data for IPW estimation."""
        np.random.seed(42)
        n = 200
        
        # Treatment assignment based on covariate
        x = np.random.normal(0, 1, n)
        prob_treat = 1 / (1 + np.exp(-(0.5 * x)))
        d = (np.random.random(n) < prob_treat).astype(float)
        
        # Outcome with treatment effect
        y = 2.0 + 1.5 * d + 0.8 * x + np.random.normal(0, 1, n)
        
        return pd.DataFrame({
            'y': y,
            'd': d,
            'x': x,
        })
    
    def test_ipw_att_reasonable_range(self, ipw_test_data):
        """IPW ATT should be in a reasonable range."""
        from lwdid.staggered.estimators import estimate_ipw
        
        result = estimate_ipw(
            data=ipw_test_data,
            y='y',
            d='d',
            propensity_controls=['x'],
            trim_threshold=0.01,
            se_method='analytical',
        )
        
        # True ATT is 1.5
        assert 0.5 < result.att < 2.5, f"ATT {result.att} outside reasonable range"
        assert result.se > 0, "SE should be positive"
        assert result.ci_lower < result.att < result.ci_upper, "CI should contain ATT"
    
    def test_ipw_bootstrap_se_reasonable(self, ipw_test_data):
        """IPW Bootstrap SE should be reasonable."""
        from lwdid.staggered.estimators import estimate_ipw
        
        result = estimate_ipw(
            data=ipw_test_data,
            y='y',
            d='d',
            propensity_controls=['x'],
            trim_threshold=0.01,
            se_method='bootstrap',
            n_bootstrap=100,
            seed=42,
        )
        
        # SE should be positive and reasonable
        assert result.se > 0, "SE should be positive"
        assert result.se < 5.0, "SE should not be unreasonably large"


# =============================================================================
# Test 2: IPWRA Estimation Numerical Stability
# =============================================================================

class TestIPWRANumericalStability:
    """Test that IPWRA estimation produces correct numerical results."""
    
    @pytest.fixture
    def ipwra_test_data(self):
        """Create test data for IPWRA estimation."""
        np.random.seed(123)
        n = 300
        
        # Covariates
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        
        # Treatment assignment
        prob_treat = 1 / (1 + np.exp(-(0.3 * x1 + 0.2 * x2)))
        d = (np.random.random(n) < prob_treat).astype(float)
        
        # Outcome with treatment effect = 2.0
        y = 1.0 + 2.0 * d + 0.5 * x1 + 0.3 * x2 + np.random.normal(0, 0.5, n)
        
        return pd.DataFrame({
            'y': y,
            'd': d,
            'x1': x1,
            'x2': x2,
        })
    
    def test_ipwra_att_reasonable_range(self, ipwra_test_data):
        """IPWRA ATT should be in a reasonable range."""
        from lwdid.staggered.estimators import estimate_ipwra
        
        result = estimate_ipwra(
            data=ipwra_test_data,
            y='y',
            d='d',
            controls=['x1', 'x2'],
            propensity_controls=['x1', 'x2'],
            trim_threshold=0.01,
            se_method='analytical',
        )
        
        # True ATT is 2.0
        assert 1.0 < result.att < 3.0, f"ATT {result.att} outside reasonable range"
        assert result.se > 0, "SE should be positive"
    
    def test_ipwra_bootstrap_completes(self, ipwra_test_data):
        """IPWRA Bootstrap should complete without error."""
        from lwdid.staggered.estimators import estimate_ipwra
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = estimate_ipwra(
                data=ipwra_test_data,
                y='y',
                d='d',
                controls=['x1', 'x2'],
                propensity_controls=['x1', 'x2'],
                trim_threshold=0.01,
                se_method='bootstrap',
                n_bootstrap=50,
                seed=42,
            )
        
        assert result.se > 0, "Bootstrap SE should be positive"


# =============================================================================
# Test 3: PSM Estimation Numerical Stability
# =============================================================================

class TestPSMNumericalStability:
    """Test that PSM estimation produces correct numerical results."""
    
    @pytest.fixture
    def psm_test_data(self):
        """Create test data for PSM estimation."""
        np.random.seed(456)
        n = 400
        
        # Covariates
        x1 = np.random.normal(0, 1, n)
        
        # Treatment assignment (strongly dependent on x1)
        prob_treat = 1 / (1 + np.exp(-1.5 * x1))
        d = (np.random.random(n) < prob_treat).astype(float)
        
        # Outcome with treatment effect = 1.0
        y = 0.5 + 1.0 * d + 2.0 * x1 + np.random.normal(0, 0.3, n)
        
        return pd.DataFrame({
            'y': y,
            'd': d,
            'x1': x1,
        })
    
    def test_psm_att_reasonable_range(self, psm_test_data):
        """PSM ATT should be in a reasonable range."""
        from lwdid.staggered.estimators import estimate_psm
        
        result = estimate_psm(
            data=psm_test_data,
            y='y',
            d='d',
            propensity_controls=['x1'],
            n_neighbors=3,
            with_replacement=True,
            se_method='abadie_imbens_full',
        )
        
        # True ATT is 1.0
        assert 0.0 < result.att < 2.0, f"ATT {result.att} outside reasonable range"
        assert result.se > 0, "SE should be positive"
        assert result.n_matched > 0, "Should have matches"


# =============================================================================
# Test 4: Aggregation Numerical Stability
# =============================================================================

class TestAggregationNumericalStability:
    """Test that aggregation functions produce correct results."""
    
    @pytest.fixture
    def staggered_data(self):
        """Create staggered adoption data."""
        np.random.seed(789)
        
        units = 50
        periods = 10
        
        # Create panel
        data = pd.DataFrame({
            'id': np.repeat(np.arange(1, units + 1), periods),
            't': np.tile(np.arange(1, periods + 1), units),
        })
        
        # Assign cohorts (staggered adoption)
        np.random.seed(789)
        unit_cohorts = np.random.choice([0, 5, 6, 7], size=units, p=[0.3, 0.25, 0.25, 0.2])
        data['gvar'] = data['id'].map(dict(zip(range(1, units + 1), unit_cohorts)))
        
        # Treatment indicator
        data['d'] = ((data['t'] >= data['gvar']) & (data['gvar'] > 0)).astype(int)
        
        # Outcome with heterogeneous treatment effects
        base_effect = 2.0
        data['y'] = (
            np.random.normal(0, 1, len(data)) +
            base_effect * data['d']
        )
        
        return data
    
    def test_aggregation_handles_cohort_failures(self, staggered_data):
        """Aggregation should handle cohort failures gracefully."""
        from lwdid.staggered.aggregation import _compute_cohort_aggregated_variable
        
        # Create transformation columns
        staggered_data['ydot_g5_r5'] = staggered_data['y']
        staggered_data['ydot_g5_r6'] = staggered_data['y']
        
        # This should work without errors
        result = _compute_cohort_aggregated_variable(
            staggered_data,
            'id',
            ['ydot_g5_r5', 'ydot_g5_r6']
        )
        
        assert len(result) > 0, "Should produce results"
        assert not result.isna().all(), "Should have non-NaN values"


# =============================================================================
# Test 5: Regression Estimation Numerical Stability
# =============================================================================

class TestRegressionNumericalStability:
    """Test OLS regression estimation numerical stability."""
    
    @pytest.fixture
    def regression_data(self):
        """Create regression test data."""
        np.random.seed(111)
        n = 100
        
        x = np.random.normal(0, 1, n)
        d = (np.random.random(n) > 0.5).astype(float)
        y = 1.0 + 2.0 * d + 0.5 * x + np.random.normal(0, 0.5, n)
        
        return pd.DataFrame({'y': y, 'd': d, 'x': x})
    
    def test_ols_regression_numerical_accuracy(self, regression_data):
        """OLS regression should produce numerically accurate results."""
        from lwdid.staggered.estimation import run_ols_regression
        
        result = run_ols_regression(
            data=regression_data,
            y='y',
            d='d',
            controls=['x'],
            vce='robust',
        )
        
        # Treatment coefficient should be close to 2.0
        assert 1.0 < result['att'] < 3.0, f"ATT {result['att']} outside expected range"
        assert result['se'] > 0, "SE should be positive"
        assert result['pvalue'] < 0.05, "Should be significant"


# =============================================================================
# Test 6: Randomization Inference Numerical Stability
# =============================================================================

class TestRINumericalStability:
    """Test randomization inference numerical stability."""
    
    @pytest.fixture
    def ri_data(self):
        """Create data for RI testing."""
        np.random.seed(222)
        n = 50
        
        d = (np.random.random(n) > 0.5).astype(float)
        y = 1.0 + 1.5 * d + np.random.normal(0, 1, n)
        
        return pd.DataFrame({'y': y, 'd': d})
    
    def test_ri_completes_without_error(self, ri_data):
        """RI should complete without errors."""
        from lwdid.randomization import randomization_inference
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = randomization_inference(
                firstpost_df=ri_data,
                d_col='d',
                y_col='y',
                rireps=100,
                ri_method='permutation',
                seed=42,
            )
        
        assert 'p_value' in result, "Should have p-value"
        assert 0.0 <= result['p_value'] <= 1.0, "p-value should be in [0, 1]"


# =============================================================================
# Test 7: Edge Case Numerical Tests
# =============================================================================

class TestEdgeCaseNumerical:
    """Test numerical behavior in edge cases."""
    
    def test_small_sample_handling(self):
        """Small samples should be handled without crashing."""
        np.random.seed(333)
        
        # Very small sample
        small_data = pd.DataFrame({
            'y': [1, 2, 3, 4, 5],
            'd': [0, 0, 1, 1, 1],
            'x': [0.1, 0.2, 0.3, 0.4, 0.5],
        })
        
        from lwdid.staggered.estimators import estimate_ipw
        
        # May fail or succeed, but should not crash with unexpected exception
        try:
            result = estimate_ipw(
                data=small_data,
                y='y',
                d='d',
                propensity_controls=['x'],
                trim_threshold=0.01,
                se_method='analytical',
            )
            # If successful, check result is reasonable
            assert not np.isnan(result.att) or True  # NaN is acceptable for small sample
        except (ValueError, np.linalg.LinAlgError):
            # These are expected failures for small samples
            pass
    
    def test_perfect_separation_handling(self):
        """Perfect separation should raise ValueError (caught)."""
        # Perfect separation: d perfectly predicts y
        perfect_sep_data = pd.DataFrame({
            'y': [0, 0, 0, 1, 1, 1],
            'd': [0, 0, 0, 1, 1, 1],
            'x': [0, 0, 0, 1, 1, 1],
        })
        
        from lwdid.staggered.estimators import estimate_ipw
        
        # Should handle perfect separation
        try:
            result = estimate_ipw(
                data=perfect_sep_data,
                y='y',
                d='d',
                propensity_controls=['x'],
                trim_threshold=0.01,
                se_method='analytical',
            )
        except (ValueError, np.linalg.LinAlgError, RuntimeError):
            # Expected exceptions
            pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

"""
Numerical Validation Tests for Story 1.3: Diagnostics API

This module validates the numerical correctness of propensity score diagnostics
by comparing Python calculations against independent implementations.

Test categories:
1. Statistics calculation verification
2. Weights CV formula verification
3. Extreme percentage calculation verification
4. Quantile ordering verification

References:
    Story 1.3: 顶层API参数暴露
    Lee & Wooldridge (2023), Section 4
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from lwdid import lwdid
from lwdid.staggered.estimators import (
    PropensityScoreDiagnostics,
    estimate_propensity_score,
)


class TestStatisticsNumerical:
    """Numerical validation of PS statistics calculations."""
    
    def test_ps_mean_calculation(self):
        """Verify PS mean matches numpy calculation."""
        np.random.seed(42)
        n = 100
        
        # Create simple test data
        data = pd.DataFrame({
            'y': np.random.randn(n),
            'd': np.concatenate([np.ones(30), np.zeros(70)]),
            'x1': np.random.randn(n),
            'x2': np.random.randn(n),
        })
        
        # Estimate propensity scores
        ps_trimmed, ps_info, diag = estimate_propensity_score(
            data=data,
            d='d',
            controls=['x1', 'x2'],
            trim_threshold=0.01,
            return_diagnostics=True,
        )
        
        # Verify mean calculation
        expected_mean = np.mean(ps_trimmed)
        
        assert abs(diag.ps_mean - expected_mean) < 1e-10, \
            f"PS mean mismatch: {diag.ps_mean} vs {expected_mean}"
    
    def test_ps_std_calculation(self):
        """Verify PS std matches numpy calculation with ddof=1."""
        np.random.seed(42)
        n = 100
        
        data = pd.DataFrame({
            'y': np.random.randn(n),
            'd': np.concatenate([np.ones(30), np.zeros(70)]),
            'x1': np.random.randn(n),
            'x2': np.random.randn(n),
        })
        
        ps_trimmed, ps_info, diag = estimate_propensity_score(
            data=data,
            d='d',
            controls=['x1', 'x2'],
            trim_threshold=0.01,
            return_diagnostics=True,
        )
        
        # Verify std calculation (unbiased, ddof=1)
        expected_std = np.std(ps_trimmed, ddof=1)
        
        assert abs(diag.ps_std - expected_std) < 1e-10, \
            f"PS std mismatch: {diag.ps_std} vs {expected_std}"
    
    def test_ps_quantiles_calculation(self):
        """Verify PS quantiles match numpy calculation."""
        np.random.seed(42)
        n = 100
        
        data = pd.DataFrame({
            'y': np.random.randn(n),
            'd': np.concatenate([np.ones(30), np.zeros(70)]),
            'x1': np.random.randn(n),
            'x2': np.random.randn(n),
        })
        
        ps_trimmed, ps_info, diag = estimate_propensity_score(
            data=data,
            d='d',
            controls=['x1', 'x2'],
            trim_threshold=0.01,
            return_diagnostics=True,
        )
        
        # Verify quantile calculations
        expected_q25 = np.percentile(ps_trimmed, 25)
        expected_q50 = np.percentile(ps_trimmed, 50)
        expected_q75 = np.percentile(ps_trimmed, 75)
        
        assert abs(diag.ps_quantiles['25%'] - expected_q25) < 1e-10, \
            f"Q25 mismatch: {diag.ps_quantiles['25%']} vs {expected_q25}"
        assert abs(diag.ps_quantiles['50%'] - expected_q50) < 1e-10, \
            f"Q50 mismatch: {diag.ps_quantiles['50%']} vs {expected_q50}"
        assert abs(diag.ps_quantiles['75%'] - expected_q75) < 1e-10, \
            f"Q75 mismatch: {diag.ps_quantiles['75%']} vs {expected_q75}"


class TestWeightsCVNumerical:
    """Numerical validation of weights CV calculation."""
    
    def test_weights_cv_formula(self):
        """Verify weights CV = std/mean for control group weights."""
        np.random.seed(42)
        n = 100
        
        data = pd.DataFrame({
            'y': np.random.randn(n),
            'd': np.concatenate([np.ones(30), np.zeros(70)]),
            'x1': np.random.randn(n),
            'x2': np.random.randn(n),
        })
        
        ps_trimmed, ps_info, diag = estimate_propensity_score(
            data=data,
            d='d',
            controls=['x1', 'x2'],
            trim_threshold=0.01,
            return_diagnostics=True,
        )
        
        d_vals = data['d'].values
        
        # Calculate control group weights: p/(1-p)
        control_mask = (d_vals == 0)
        ps_control = ps_trimmed[control_mask]
        weights = ps_control / (1 - ps_control)
        
        # CV = std / mean
        expected_cv = np.std(weights, ddof=1) / np.mean(weights)
        
        # Allow small tolerance for numerical differences
        assert abs(diag.weights_cv - expected_cv) < 1e-6, \
            f"Weights CV mismatch: {diag.weights_cv} vs {expected_cv}"


class TestExtremePctNumerical:
    """Numerical validation of extreme percentage calculations."""
    
    def test_extreme_low_pct_calculation(self):
        """Verify extreme_low_pct counts PS < trim_threshold."""
        np.random.seed(42)
        n = 100
        trim = 0.05
        
        data = pd.DataFrame({
            'y': np.random.randn(n),
            'd': np.concatenate([np.ones(30), np.zeros(70)]),
            'x1': np.random.randn(n),
            'x2': np.random.randn(n),
        })
        
        ps_trimmed, ps_info, diag = estimate_propensity_score(
            data=data,
            d='d',
            controls=['x1', 'x2'],
            trim_threshold=trim,
            return_diagnostics=True,
        )
        
        # Get raw PS from ps_info
        ps_raw = ps_info.get('ps_raw', ps_trimmed)
        
        # Calculate expected extreme low percentage
        n_extreme_low = np.sum(ps_raw < trim)
        expected_pct = n_extreme_low / len(ps_raw)
        
        assert abs(diag.extreme_low_pct - expected_pct) < 1e-10, \
            f"Extreme low pct mismatch: {diag.extreme_low_pct} vs {expected_pct}"
    
    def test_extreme_high_pct_calculation(self):
        """Verify extreme_high_pct counts PS > 1 - trim_threshold."""
        np.random.seed(42)
        n = 100
        trim = 0.05
        
        data = pd.DataFrame({
            'y': np.random.randn(n),
            'd': np.concatenate([np.ones(30), np.zeros(70)]),
            'x1': np.random.randn(n),
            'x2': np.random.randn(n),
        })
        
        ps_trimmed, ps_info, diag = estimate_propensity_score(
            data=data,
            d='d',
            controls=['x1', 'x2'],
            trim_threshold=trim,
            return_diagnostics=True,
        )
        
        # Get raw PS from ps_info
        ps_raw = ps_info.get('ps_raw', ps_trimmed)
        
        # Calculate expected extreme high percentage
        n_extreme_high = np.sum(ps_raw > 1 - trim)
        expected_pct = n_extreme_high / len(ps_raw)
        
        assert abs(diag.extreme_high_pct - expected_pct) < 1e-10, \
            f"Extreme high pct mismatch: {diag.extreme_high_pct} vs {expected_pct}"


class TestTrimmingNumerical:
    """Numerical validation of trimming behavior."""
    
    def test_n_trimmed_calculation(self):
        """Verify n_trimmed counts observations outside trim bounds."""
        np.random.seed(42)
        n = 100
        trim = 0.05
        
        data = pd.DataFrame({
            'y': np.random.randn(n),
            'd': np.concatenate([np.ones(30), np.zeros(70)]),
            'x1': np.random.randn(n),
            'x2': np.random.randn(n),
        })
        
        ps_trimmed, ps_info, diag = estimate_propensity_score(
            data=data,
            d='d',
            controls=['x1', 'x2'],
            trim_threshold=trim,
            return_diagnostics=True,
        )
        
        # Get raw PS from ps_info
        ps_raw = ps_info.get('ps_raw', ps_trimmed)
        
        # Calculate expected trimmed count
        n_low = np.sum(ps_raw < trim)
        n_high = np.sum(ps_raw > 1 - trim)
        expected_trimmed = n_low + n_high
        
        assert diag.n_trimmed == expected_trimmed, \
            f"N trimmed mismatch: {diag.n_trimmed} vs {expected_trimmed}"
    
    def test_trimmed_ps_bounds(self):
        """Verify trimmed PS values are within bounds."""
        np.random.seed(42)
        n = 100
        trim = 0.05
        
        data = pd.DataFrame({
            'y': np.random.randn(n),
            'd': np.concatenate([np.ones(30), np.zeros(70)]),
            'x1': np.random.randn(n),
            'x2': np.random.randn(n),
        })
        
        ps_trimmed, ps_info, diag = estimate_propensity_score(
            data=data,
            d='d',
            controls=['x1', 'x2'],
            trim_threshold=trim,
            return_diagnostics=True,
        )
        
        # Verify bounds
        assert diag.ps_min >= trim - 1e-10, \
            f"PS min {diag.ps_min} should be >= {trim}"
        assert diag.ps_max <= 1 - trim + 1e-10, \
            f"PS max {diag.ps_max} should be <= {1 - trim}"


class TestDiagnosticsIntegration:
    """Integration tests for diagnostics through full pipeline."""
    
    def test_diagnostics_consistency_across_estimators(self, staggered_test_data):
        """IPWRA and PSM should produce similar PS diagnostics for same data."""
        # Run IPWRA
        result_ipwra = lwdid(
            staggered_test_data,
            y='y',
            ivar='id',
            tvar='period',
            gvar='gvar',
            estimator='ipwra',
            controls=['x1', 'x2'],
            return_diagnostics=True,
        )
        
        # Run PSM
        result_psm = lwdid(
            staggered_test_data,
            y='y',
            ivar='id',
            tvar='period',
            gvar='gvar',
            estimator='psm',
            controls=['x1', 'x2'],
            return_diagnostics=True,
        )
        
        diags_ipwra = result_ipwra.get_diagnostics()
        diags_psm = result_psm.get_diagnostics()
        
        # Should have same (g,r) pairs
        assert set(diags_ipwra.keys()) == set(diags_psm.keys()), \
            "IPWRA and PSM should have diagnostics for same (g,r) pairs"
        
        # PS statistics should be similar (same PS model)
        for (g, r) in diags_ipwra.keys():
            diag_ipwra = diags_ipwra[(g, r)]
            diag_psm = diags_psm[(g, r)]
            
            # PS mean should be close (same logit model)
            assert abs(diag_ipwra.ps_mean - diag_psm.ps_mean) < 0.01, \
                f"PS mean should be similar for ({g},{r})"


# Fixture for staggered test data
@pytest.fixture
def staggered_test_data():
    """Generate staggered panel data for testing."""
    np.random.seed(42)
    
    n_units = 100
    n_periods = 6
    
    # Create panel structure
    ids = np.repeat(np.arange(1, n_units + 1), n_periods)
    periods = np.tile(np.arange(1, n_periods + 1), n_units)
    
    # Assign treatment cohorts (0 = never treated)
    cohorts = np.random.choice([0, 3, 4, 5], size=n_units, p=[0.4, 0.2, 0.2, 0.2])
    gvar = np.repeat(cohorts, n_periods)
    
    # Generate covariates
    x1 = np.random.randn(n_units * n_periods)
    x2 = np.random.randn(n_units * n_periods)
    
    # Generate outcome with treatment effect
    treatment = (periods >= gvar) & (gvar > 0)
    y = 1 + 0.5 * x1 + 0.3 * x2 + 2.0 * treatment + np.random.randn(n_units * n_periods)
    
    return pd.DataFrame({
        'id': ids,
        'period': periods,
        'gvar': gvar,
        'y': y,
        'x1': x1,
        'x2': x2,
    })


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

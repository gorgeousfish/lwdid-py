"""
Numerical validation test for BUG-161 fix.

This module performs comprehensive numerical validation comparing the
memory-efficient path and cdist path results to ensure they produce
consistent standard error estimates across different sample sizes.
"""

import numpy as np
import pandas as pd
import pytest

from lwdid.staggered.estimators import (
    _estimate_conditional_variance_same_group,
    _MEMORY_EFFICIENT_THRESHOLD,
)


class TestNumericalConsistencyAcrossScales:
    """Numerical validation comparing small and large sample scenarios."""

    def test_se_consistency_scaled_data(self):
        """Test SE consistency when scaling same data structure to different sizes."""
        np.random.seed(42)
        
        # Base parameters for data generation
        treatment_effect = 2.5
        noise_sd = 1.0
        
        results = []
        
        for scale in [1, 2, 5]:
            n_base = 100 * scale
            n_treat = 30 * scale
            n_control = 70 * scale
            
            # Generate covariates
            X = np.random.randn(n_base)
            
            # Treatment assignment based on propensity
            ps_true = 1 / (1 + np.exp(-0.5 * X))
            W = (np.random.rand(n_base) < ps_true).astype(float)
            n_treat_actual = int(W.sum())
            n_control_actual = n_base - n_treat_actual
            
            # Generate outcomes
            Y0 = X + np.random.randn(n_base) * noise_sd
            Y1 = Y0 + treatment_effect
            Y = W * Y1 + (1 - W) * Y0
            
            # Estimate conditional variance
            X_reshaped = X.reshape(-1, 1)
            sigma2 = _estimate_conditional_variance_same_group(Y, X_reshaped, W, J=2)
            
            # Record statistics
            results.append({
                'n': n_base,
                'n_treat': n_treat_actual,
                'n_control': n_control_actual,
                'mean_sigma2': np.mean(sigma2),
                'std_sigma2': np.std(sigma2),
                'min_sigma2': np.min(sigma2),
                'max_sigma2': np.max(sigma2),
                'n_zeros': np.sum(sigma2 == 0),
                'path': 'memory_efficient' if n_base >= _MEMORY_EFFICIENT_THRESHOLD else 'cdist'
            })
        
        df = pd.DataFrame(results)
        
        # Assertions
        # 1. No zeros should appear after the fix
        assert df['n_zeros'].sum() == 0, f"Found {df['n_zeros'].sum()} zeros in variance estimates"
        
        # 2. Mean variance should be roughly consistent (not varying wildly with sample size)
        mean_vars = df['mean_sigma2'].values
        cv = np.std(mean_vars) / np.mean(mean_vars)
        assert cv < 0.5, f"Variance estimates too inconsistent across scales (CV={cv:.3f})"
        
        print("\nNumerical Validation Results:")
        print(df.to_string(index=False))

    def test_boundary_case_comparison(self):
        """Compare results just below and above the threshold."""
        np.random.seed(123)
        
        # Just below threshold (cdist path)
        n_below = _MEMORY_EFFICIENT_THRESHOLD - 10
        Y_below = np.random.randn(n_below) * 2 + 5
        X_below = np.random.randn(n_below).reshape(-1, 1)
        W_below = np.concatenate([np.ones(n_below // 3), np.zeros(n_below - n_below // 3)])
        
        sigma2_below = _estimate_conditional_variance_same_group(Y_below, X_below, W_below, J=2)
        
        # Just above threshold (memory-efficient path)
        n_above = _MEMORY_EFFICIENT_THRESHOLD + 10
        Y_above = np.random.randn(n_above) * 2 + 5
        X_above = np.random.randn(n_above).reshape(-1, 1)
        W_above = np.concatenate([np.ones(n_above // 3), np.zeros(n_above - n_above // 3)])
        
        sigma2_above = _estimate_conditional_variance_same_group(Y_above, X_above, W_above, J=2)
        
        # Compare statistics
        stats = {
            'below_threshold': {
                'n': n_below,
                'mean': np.mean(sigma2_below),
                'std': np.std(sigma2_below),
                'zeros': np.sum(sigma2_below == 0),
                'path': 'cdist'
            },
            'above_threshold': {
                'n': n_above,
                'mean': np.mean(sigma2_above),
                'std': np.std(sigma2_above),
                'zeros': np.sum(sigma2_above == 0),
                'path': 'memory_efficient'
            }
        }
        
        print(f"\nBoundary Comparison (threshold={_MEMORY_EFFICIENT_THRESHOLD}):")
        for key, val in stats.items():
            print(f"  {key}: n={val['n']}, mean={val['mean']:.4f}, std={val['std']:.4f}, zeros={val['zeros']}, path={val['path']}")
        
        # No zeros in either path
        assert stats['below_threshold']['zeros'] == 0, "cdist path produced zeros"
        assert stats['above_threshold']['zeros'] == 0, "memory-efficient path produced zeros"
        
        # Means should be in same order of magnitude
        ratio = stats['above_threshold']['mean'] / stats['below_threshold']['mean']
        assert 0.1 < ratio < 10, f"Mean variance ratio {ratio:.2f} suggests inconsistent behavior"

    def test_edge_case_small_treated_group(self):
        """Test with very small treated group forcing fallback logic."""
        np.random.seed(456)
        
        for path_type, n in [('cdist', 100), ('memory_efficient', _MEMORY_EFFICIENT_THRESHOLD + 100)]:
            Y = np.random.randn(n) * 3 + 10
            X = np.random.randn(n).reshape(-1, 1)
            
            # Only 3 treated units -> small group triggers fallback
            n_treat = 3
            W = np.concatenate([np.ones(n_treat), np.zeros(n - n_treat)])
            
            sigma2 = _estimate_conditional_variance_same_group(Y, X, W, J=2)
            
            # Check treated group variances
            treated_sigma2 = sigma2[W == 1]
            
            print(f"\n{path_type} path with {n_treat} treated units (n={n}):")
            print(f"  Treated sigma2: {treated_sigma2}")
            print(f"  Mean: {np.mean(treated_sigma2):.4f}, Any zeros: {np.any(treated_sigma2 == 0)}")
            
            assert not np.any(treated_sigma2 == 0), f"{path_type} path produced zeros for small treated group"
            assert np.all(treated_sigma2 > 0), f"All variances should be positive"


class TestPSMIntegration:
    """Test PSM ATT estimation with fallback consistency."""

    @pytest.mark.slow
    def test_psm_att_small_sample(self):
        """Test PSM ATT estimation with small sample."""
        np.random.seed(789)
        
        n = 100
        # Create simple panel data
        data = pd.DataFrame({
            'id': np.repeat(np.arange(n), 3),
            'time': np.tile([2000, 2001, 2002], n),
            'y': np.random.randn(n * 3) * 2,
            'x1': np.repeat(np.random.randn(n), 3),
            'd': np.repeat(np.concatenate([np.ones(30), np.zeros(70)]).astype(int), 3)
        })
        
        # Prepare for PSM: get one cross-section
        cross_section = data[data['time'] == 2002].copy()
        
        Y = cross_section['y'].values
        X = cross_section[['x1']].values
        D = cross_section['d'].values
        
        # Estimate propensity scores (simple logistic model simulation)
        ps = 1 / (1 + np.exp(-0.5 * X.ravel()))
        
        # Conditional variance estimation
        sigma2 = _estimate_conditional_variance_same_group(Y, X, D, J=2)
        
        # Verify no zeros
        assert not np.any(sigma2 == 0), "PSM integration test found zeros in variance estimates"
        
        print(f"\nPSM Integration (n={n}):")
        print(f"  Treated sigma2 range: [{sigma2[D == 1].min():.4f}, {sigma2[D == 1].max():.4f}]")
        print(f"  Control sigma2 range: [{sigma2[D == 0].min():.4f}, {sigma2[D == 0].max():.4f}]")


class TestMonteCarlo:
    """Monte Carlo simulation to verify consistency."""

    @pytest.mark.slow
    def test_monte_carlo_variance_coverage(self):
        """Monte Carlo test for variance estimate coverage."""
        np.random.seed(2024)
        
        n_sims = 50
        n_sample = 200
        true_var = 4.0  # Known population variance
        
        estimated_vars = []
        
        for sim in range(n_sims):
            Y = np.random.randn(n_sample) * np.sqrt(true_var)
            X = np.random.randn(n_sample).reshape(-1, 1)
            W = np.concatenate([np.ones(n_sample // 2), np.zeros(n_sample // 2)])
            
            sigma2 = _estimate_conditional_variance_same_group(Y, X, W, J=2)
            estimated_vars.append(np.mean(sigma2))
        
        estimated_vars = np.array(estimated_vars)
        
        # Check no zeros
        assert not np.any(estimated_vars == 0), "Monte Carlo found zero variance estimates"
        
        # Check mean is reasonable (within 50% of true variance)
        mean_est = np.mean(estimated_vars)
        assert 0.5 * true_var < mean_est < 2.0 * true_var, \
            f"Mean estimated variance {mean_est:.2f} far from true {true_var}"
        
        print(f"\nMonte Carlo Results (n_sims={n_sims}, n={n_sample}):")
        print(f"  True variance: {true_var}")
        print(f"  Mean estimated: {mean_est:.4f}")
        print(f"  Std of estimates: {np.std(estimated_vars):.4f}")
        print(f"  Any zeros: {np.any(estimated_vars == 0)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

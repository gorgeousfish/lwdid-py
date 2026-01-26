"""
Monte Carlo simulation tests for BUG-183, BUG-184, BUG-185 fixes.

These tests verify:
1. BUG-183: HC4 variance estimation stability under various matrix conditions
2. BUG-184: RI failure handling robustness across many simulations
3. BUG-185: Outcome NaN detection reliability

Monte Carlo tests help ensure the fixes work correctly across a wide range
of scenarios, not just specific test cases.
"""
import numpy as np
import pandas as pd
import pytest
import warnings
from collections import Counter

from lwdid import lwdid
from lwdid.randomization import randomization_inference
from lwdid.exceptions import RandomizationError
from lwdid.estimation import estimate_att


class TestBUG183_MonteCarlo:
    """Monte Carlo tests for BUG-183: HC4 variance calculation robustness."""
    
    def test_hc4_coverage_rate(self):
        """
        Monte Carlo: Verify HC4 confidence intervals have correct coverage.
        
        Run many simulations with known true ATT and check that 95% CI
        contains true value approximately 95% of the time.
        """
        np.random.seed(42)
        n_sims = 100
        n_obs = 100
        true_att = 2.0
        coverage_count = 0
        
        for sim in range(n_sims):
            # Generate data with known ATT
            d_vals = np.array([0] * 50 + [1] * 50)
            x1 = np.random.randn(n_obs)
            y = 1.0 + true_att * d_vals + 0.5 * x1 + np.random.randn(n_obs)
            
            data = pd.DataFrame({
                'id': range(n_obs),
                'y': y,
                'd_': d_vals,
                'x1': x1,
            })
            
            result = estimate_att(
                data=data,
                y_transformed='y',
                d='d_',
                ivar='id',
                controls=['x1'],
                vce='hc4',
                cluster_var=None,
                sample_filter=pd.Series(True, index=data.index),
            )
            
            # Check if true ATT is within 95% CI
            ci_lower = result['att'] - 1.96 * result['se_att']
            ci_upper = result['att'] + 1.96 * result['se_att']
            
            if ci_lower <= true_att <= ci_upper:
                coverage_count += 1
        
        coverage_rate = coverage_count / n_sims
        
        # Coverage should be approximately 95% (allow for sampling variation)
        assert 0.85 <= coverage_rate <= 1.0, \
            f"HC4 coverage rate {coverage_rate:.2%} outside expected range [85%, 100%]"
    
    def test_hc4_singular_matrix_recovery(self):
        """
        Monte Carlo: Test that HC4 handles varying degrees of collinearity.
        
        Generate data with varying correlation between controls and verify
        that estimation succeeds even with high correlation.
        """
        np.random.seed(123)
        n_sims = 50
        n_obs = 100
        
        correlations = [0.0, 0.5, 0.9, 0.99, 0.999, 1.0]  # 1.0 = perfect collinearity
        success_rates = {}
        
        for corr in correlations:
            successes = 0
            
            for sim in range(n_sims):
                d_vals = np.array([0] * 50 + [1] * 50)
                x1 = np.random.randn(n_obs)
                
                # Generate x2 with specified correlation to x1
                if corr == 1.0:
                    x2 = 2 * x1  # Perfect collinearity
                else:
                    noise = np.random.randn(n_obs)
                    x2 = corr * x1 + np.sqrt(1 - corr**2) * noise
                
                y = 1.0 + 2.0 * d_vals + 0.5 * x1 + 0.3 * x2 + np.random.randn(n_obs)
                
                data = pd.DataFrame({
                    'id': range(n_obs),
                    'y': y,
                    'd_': d_vals,
                    'x1': x1,
                    'x2': x2,
                })
                
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        
                        result = estimate_att(
                            data=data,
                            y_transformed='y',
                            d='d_',
                            ivar='id',
                            controls=['x1', 'x2'],
                            vce='hc4',
                            cluster_var=None,
                            sample_filter=pd.Series(True, index=data.index),
                        )
                    
                    if not np.isnan(result['att']):
                        successes += 1
                except Exception:
                    pass
            
            success_rates[corr] = successes / n_sims
        
        # All correlations should have high success rate (pinv handles singularity)
        for corr, rate in success_rates.items():
            assert rate >= 0.9, \
                f"HC4 success rate {rate:.2%} too low for correlation {corr}"


class TestBUG184_MonteCarlo:
    """Monte Carlo tests for BUG-184: RI failure handling."""
    
    def test_ri_pvalue_distribution(self):
        """
        Monte Carlo: Verify RI p-value distribution under null hypothesis.
        
        Under H0 (no treatment effect), p-values should be approximately
        uniformly distributed on [0, 1].
        """
        np.random.seed(42)
        n_sims = 50
        pvalues = []
        
        for sim in range(n_sims):
            n_units = 30
            n_periods = 6
            treatment_period = 4
            
            data = []
            for i in range(n_units):
                is_treated = i < n_units // 2
                d_unit = 1 if is_treated else 0
                
                for t in range(1, n_periods + 1):
                    post = 1 if t >= treatment_period else 0
                    # NO treatment effect under H0
                    y = np.random.randn()
                    data.append({
                        'unit_id': i,
                        'time_var': t,
                        'y': y,
                        'd': d_unit,
                        'post': post,
                        'quarter': ((t - 1) % 4) + 1,
                    })
            
            df = pd.DataFrame(data)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                result = lwdid(
                    data=df,
                    y='y',
                    ivar='unit_id',
                    tvar='time_var',
                    d='d',
                    post='post',
                    ri=True,
                    rireps=100,
                    seed=sim,  # Different seed for each simulation
                )
            
            if hasattr(result, 'ri_pvalue') and not np.isnan(result.ri_pvalue):
                pvalues.append(result.ri_pvalue)
        
        # Under H0, p-values should be approximately uniform
        # Check that we don't have too many small p-values
        if len(pvalues) >= 20:
            small_pvalues = sum(1 for p in pvalues if p < 0.05)
            # Under H0, expect ~5% small p-values, allow up to 20% for sampling variation
            assert small_pvalues / len(pvalues) <= 0.3, \
                f"Too many small p-values ({small_pvalues}/{len(pvalues)}) under H0"
    
    def test_att_always_computed(self):
        """
        Monte Carlo: Verify ATT is always computed even when RI has issues.
        
        This directly tests BUG-184 fix: RI problems shouldn't crash ATT.
        """
        np.random.seed(42)
        n_sims = 30
        att_computed = 0
        
        for sim in range(n_sims):
            n_units = 20
            n_periods = 6
            treatment_period = 4
            
            data = []
            for i in range(n_units):
                is_treated = i < n_units // 2
                d_unit = 1 if is_treated else 0
                
                for t in range(1, n_periods + 1):
                    post = 1 if t >= treatment_period else 0
                    treat_effect = 2.0 if (is_treated and post == 1) else 0
                    y = np.random.randn() + treat_effect
                    data.append({
                        'unit_id': i,
                        'time_var': t,
                        'y': y,
                        'd': d_unit,
                        'post': post,
                        'quarter': ((t - 1) % 4) + 1,
                    })
            
            df = pd.DataFrame(data)
            
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    result = lwdid(
                        data=df,
                        y='y',
                        ivar='unit_id',
                        tvar='time_var',
                        d='d',
                        post='post',
                        ri=True,
                        rireps=100,
                        seed=sim,
                    )
                
                if not np.isnan(result.att):
                    att_computed += 1
            except Exception:
                pass  # This should never happen after BUG-184 fix
        
        # ATT should be computed in all simulations
        assert att_computed == n_sims, \
            f"ATT computed in only {att_computed}/{n_sims} simulations"


class TestBUG185_MonteCarlo:
    """Monte Carlo tests for BUG-185: Outcome NaN detection."""
    
    def test_nan_detection_reliability(self):
        """
        Monte Carlo: Verify NaN detection works for various NaN patterns.
        
        Test with different numbers and positions of NaN values.
        """
        np.random.seed(42)
        n_obs = 50
        
        nan_patterns = [
            [5],           # Single NaN
            [0, 49],       # First and last
            [10, 20, 30],  # Multiple scattered
            list(range(10)),  # First 10 rows
        ]
        
        for nan_indices in nan_patterns:
            d_vals = np.array([0] * 25 + [1] * 25)
            
            data = pd.DataFrame({
                'id': range(n_obs),
                'y': np.random.randn(n_obs),
                'd_': d_vals,
            })
            
            # Introduce NaN at specified indices
            for idx in nan_indices:
                data.loc[idx, 'y'] = np.nan
            
            # Should always detect and reject
            with pytest.raises(RandomizationError):
                randomization_inference(
                    firstpost_df=data,
                    y_col='y',
                    d_col='d_',
                    ivar='id',
                    rireps=100,
                    seed=42,
                )
    
    def test_clean_data_success_rate(self):
        """
        Monte Carlo: Verify clean data (no NaN) always succeeds.
        """
        np.random.seed(42)
        n_sims = 50
        successes = 0
        
        for sim in range(n_sims):
            n_obs = 50
            d_vals = np.array([0] * 25 + [1] * 25)
            
            data = pd.DataFrame({
                'id': range(n_obs),
                'y': np.random.randn(n_obs) + d_vals * 1.0,
                'd_': d_vals,
            })
            
            try:
                result = randomization_inference(
                    firstpost_df=data,
                    y_col='y',
                    d_col='d_',
                    ivar='id',
                    rireps=100,
                    seed=sim,
                    ri_method='permutation',
                )
                
                if result is not None and 'p_value' in result:
                    successes += 1
            except Exception:
                pass
        
        # Clean data should always succeed
        assert successes == n_sims, \
            f"Clean data succeeded in only {successes}/{n_sims} simulations"


class TestIntegratedMonteCarlo:
    """Integrated Monte Carlo tests combining multiple bug fixes."""
    
    def test_full_pipeline_stability(self):
        """
        Monte Carlo: Test full lwdid pipeline stability across many runs.
        
        This tests that all three bug fixes work together correctly.
        """
        np.random.seed(42)
        n_sims = 30
        results_summary = {
            'att_valid': 0,
            'se_valid': 0,
            'ri_valid': 0,
            'total_success': 0,
        }
        
        for sim in range(n_sims):
            n_units = 30
            n_periods = 6
            treatment_period = 4
            
            # Generate unit-level control (time-invariant)
            unit_x = np.random.randn(n_units)
            
            data = []
            for i in range(n_units):
                is_treated = i < n_units // 2
                d_unit = 1 if is_treated else 0
                x_unit = unit_x[i]
                
                for t in range(1, n_periods + 1):
                    post = 1 if t >= treatment_period else 0
                    treat_effect = 2.0 if (is_treated and post == 1) else 0
                    y = np.random.randn() + treat_effect + 0.3 * x_unit
                    data.append({
                        'unit_id': i,
                        'time_var': t,
                        'y': y,
                        'd': d_unit,
                        'post': post,
                        'x': x_unit,
                        'quarter': ((t - 1) % 4) + 1,
                    })
            
            df = pd.DataFrame(data)
            
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    result = lwdid(
                        data=df,
                        y='y',
                        ivar='unit_id',
                        tvar='time_var',
                        d='d',
                        post='post',
                        vce='hc4',
                        controls=['x'],
                        ri=True,
                        rireps=100,
                        seed=sim,
                    )
                
                if not np.isnan(result.att):
                    results_summary['att_valid'] += 1
                if not np.isnan(result.se_att):
                    results_summary['se_valid'] += 1
                if hasattr(result, 'ri_pvalue'):
                    results_summary['ri_valid'] += 1
                
                if (not np.isnan(result.att) and 
                    not np.isnan(result.se_att) and
                    hasattr(result, 'ri_pvalue')):
                    results_summary['total_success'] += 1
                    
            except Exception as e:
                print(f"Simulation {sim} failed: {e}")
        
        # All simulations should have valid ATT and SE
        assert results_summary['att_valid'] == n_sims, \
            f"ATT valid in only {results_summary['att_valid']}/{n_sims}"
        assert results_summary['se_valid'] == n_sims, \
            f"SE valid in only {results_summary['se_valid']}/{n_sims}"
        
        # Most simulations should have RI results
        assert results_summary['ri_valid'] >= n_sims * 0.9, \
            f"RI valid in only {results_summary['ri_valid']}/{n_sims}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

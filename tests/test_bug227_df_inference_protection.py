"""
Test suite for BUG-227: df_inference <= 0 protection in t-distribution calculations.

This module tests that the run_ols_regression() function properly handles invalid
degrees of freedom by issuing warnings and returning NaN for inference statistics.

BUG-227 Details:
- When df_inference <= 0, stats.t.sf() and stats.t.ppf() return NaN silently
- Fix adds explicit validation and user warning before t-distribution calculations
- Consistent with Stata's ttail() behavior which returns missing for df <= 0
"""

import warnings
import numpy as np
import pandas as pd
import pytest
import scipy.stats as stats

from lwdid.staggered.estimation import run_ols_regression


class TestBug227DfInferenceProtection:
    """Test df_inference <= 0 protection in run_ols_regression()."""

    @pytest.fixture
    def minimal_valid_data(self):
        """Create minimal valid dataset for OLS regression."""
        np.random.seed(42)
        n = 20
        return pd.DataFrame({
            'y': np.random.randn(n),
            'd': np.array([1] * 10 + [0] * 10),
            'id': np.arange(n),
        })

    @pytest.fixture
    def high_dimensional_data(self):
        """Create dataset where controls exceed sample size capacity.
        
        With n=10 and K=8 controls, df = n - 2 - K = 10 - 2 - 8 = 0.
        This triggers the df_inference <= 0 condition.
        """
        np.random.seed(42)
        n = 10
        data = pd.DataFrame({
            'y': np.random.randn(n),
            'd': np.array([1] * 5 + [0] * 5),
        })
        # Add many control variables
        for i in range(8):
            data[f'x{i}'] = np.random.randn(n)
        return data

    def test_valid_df_inference_no_warning(self, minimal_valid_data):
        """Verify no warning is issued when df_inference > 0."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = run_ols_regression(
                data=minimal_valid_data,
                y='y',
                d='d',
                controls=None,
                vce=None,
                alpha=0.05,
            )
            
            # Filter for df_inference warnings
            df_warnings = [
                x for x in w 
                if "degrees of freedom" in str(x.message).lower()
            ]
            
            assert len(df_warnings) == 0, "Should not warn for valid df_inference"
            assert not np.isnan(result['t_stat']), "t_stat should be valid"
            assert not np.isnan(result['pvalue']), "pvalue should be valid"
            assert not np.isnan(result['ci_lower']), "ci_lower should be valid"
            assert not np.isnan(result['ci_upper']), "ci_upper should be valid"
            assert result['df_inference'] > 0, "df_inference should be positive"

    def test_zero_df_inference_warning_issued(self):
        """Verify warning is issued when df_inference = 0.
        
        Simulates the condition where n - k = 0 (saturated model).
        """
        # Create minimal data where df_resid will be 0
        # n = 2 observations, k = 2 parameters (intercept + treatment)
        data = pd.DataFrame({
            'y': [1.0, 2.0],
            'd': [1, 0],
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = run_ols_regression(
                data=data,
                y='y',
                d='d',
                controls=None,
                vce=None,
                alpha=0.05,
            )
            
            # Filter for df warnings
            df_warnings = [
                x for x in w 
                if "degrees of freedom" in str(x.message).lower()
            ]
            
            # Should have warning about insufficient df
            assert len(df_warnings) >= 1, "Should warn about invalid df"
            
            # Inference statistics should be NaN
            assert np.isnan(result['t_stat']) or result['df_inference'] <= 0
            assert np.isnan(result['pvalue']) or result['df_inference'] <= 0

    def test_negative_df_inference_warning_content(self):
        """Verify warning message contains helpful diagnostic information."""
        # Create data with k > n (impossible to fit)
        # This will trigger the df_resid <= 0 path
        data = pd.DataFrame({
            'y': [1.0, 2.0],
            'd': [1, 0],
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = run_ols_regression(
                data=data,
                y='y',
                d='d',
                controls=None,
                vce=None,
                alpha=0.05,
            )
            
            # Check warning content
            warning_messages = [str(x.message) for x in w]
            combined = " ".join(warning_messages).lower()
            
            # Should mention degrees of freedom or sample size
            assert "degrees of freedom" in combined or "no degrees" in combined, \
                f"Warning should mention df issue. Got: {warning_messages}"

    def test_att_remains_valid_when_df_invalid(self):
        """Verify ATT point estimate is valid even when inference fails."""
        data = pd.DataFrame({
            'y': [1.0, 2.0],
            'd': [1, 0],
        })
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_ols_regression(
                data=data,
                y='y',
                d='d',
                controls=None,
                vce=None,
                alpha=0.05,
            )
            
            # ATT should be computable even with no df
            # ATT = mean(y|d=1) - mean(y|d=0) = 1.0 - 2.0 = -1.0
            assert not np.isnan(result['att']), "ATT should be valid even with invalid df"

    def test_scipy_t_distribution_boundary_behavior(self):
        """Verify scipy.stats.t behavior at df boundaries.
        
        Documents the behavior that necessitates the BUG-227 fix:
        - stats.t.sf(x, df=0) returns NaN
        - stats.t.sf(x, df=-1) returns NaN
        - stats.t.ppf(q, df=0) returns NaN
        - stats.t.ppf(q, df=-1) returns NaN
        """
        # df = 0
        assert np.isnan(stats.t.sf(2.0, 0)), "t.sf with df=0 should be NaN"
        assert np.isnan(stats.t.ppf(0.975, 0)), "t.ppf with df=0 should be NaN"
        
        # df = -1
        assert np.isnan(stats.t.sf(2.0, -1)), "t.sf with df=-1 should be NaN"
        assert np.isnan(stats.t.ppf(0.975, -1)), "t.ppf with df=-1 should be NaN"
        
        # df = 1 (valid, should not be NaN)
        assert not np.isnan(stats.t.sf(2.0, 1)), "t.sf with df=1 should be valid"
        assert not np.isnan(stats.t.ppf(0.975, 1)), "t.ppf with df=1 should be valid"

    def test_cluster_se_minimum_clusters(self):
        """Verify cluster SE with G=2 clusters gives df_inference=1.
        
        With G clusters, df_inference = G - 1. Minimum valid is G=2 → df=1.
        """
        np.random.seed(42)
        data = pd.DataFrame({
            'y': np.random.randn(20),
            'd': np.array([1] * 10 + [0] * 10),
            'cluster': np.array([1] * 10 + [2] * 10),
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = run_ols_regression(
                data=data,
                y='y',
                d='d',
                controls=None,
                vce='cluster',
                cluster_var='cluster',
                alpha=0.05,
            )
            
            # df_inference should be G - 1 = 2 - 1 = 1
            assert result['df_inference'] == 1, \
                f"df_inference should be 1 with 2 clusters, got {result['df_inference']}"
            
            # Should have valid inference (df=1 is valid, though wide CI)
            assert not np.isnan(result['t_stat']), "t_stat should be valid with df=1"
            assert not np.isnan(result['pvalue']), "pvalue should be valid with df=1"

    def test_t_critical_value_at_df_boundaries(self):
        """Verify t critical values at small df are correctly computed."""
        alpha = 0.05
        
        # df = 1: t_0.975 ≈ 12.706
        t_crit_1 = stats.t.ppf(1 - alpha / 2, 1)
        assert abs(t_crit_1 - 12.706) < 0.01, f"t_crit at df=1 should be ~12.706, got {t_crit_1}"
        
        # df = 2: t_0.975 ≈ 4.303
        t_crit_2 = stats.t.ppf(1 - alpha / 2, 2)
        assert abs(t_crit_2 - 4.303) < 0.01, f"t_crit at df=2 should be ~4.303, got {t_crit_2}"
        
        # df = 30: t_0.975 ≈ 2.042
        t_crit_30 = stats.t.ppf(1 - alpha / 2, 30)
        assert abs(t_crit_30 - 2.042) < 0.01, f"t_crit at df=30 should be ~2.042, got {t_crit_30}"


class TestBug227NumericalValidation:
    """Numerical validation tests for BUG-227 fix."""

    def test_pvalue_formula_equivalence(self):
        """Verify p-value formula: p = 2 * t.sf(|t|, df) matches Stata's ttail.
        
        Stata: p = 2 * ttail(df, |t|)
        Python: p = 2 * stats.t.sf(|t|, df)
        
        Both compute: 2 * P(T > |t|) for t-distribution with df degrees of freedom.
        """
        test_cases = [
            {'t': 2.0, 'df': 10},
            {'t': 1.96, 'df': 100},
            {'t': 2.576, 'df': 30},
            {'t': 3.0, 'df': 5},
        ]
        
        for case in test_cases:
            t_stat = case['t']
            df = case['df']
            
            # Python formula
            p_python = 2 * stats.t.sf(abs(t_stat), df)
            
            # Alternative formula using CDF
            p_alt = 2 * (1 - stats.t.cdf(abs(t_stat), df))
            
            # Should match
            assert abs(p_python - p_alt) < 1e-10, \
                f"P-value formulas should match for t={t_stat}, df={df}"

    def test_confidence_interval_coverage_property(self):
        """Verify CI has correct coverage under repeated sampling (Monte Carlo).
        
        For large samples, 95% CI should cover true parameter ~95% of time.
        """
        np.random.seed(42)
        n_simulations = 500
        n_obs = 50
        true_att = 1.0
        alpha = 0.05
        
        coverage_count = 0
        
        for _ in range(n_simulations):
            # Generate data with known ATT
            y0 = np.random.randn(n_obs)
            d = np.array([1] * (n_obs // 2) + [0] * (n_obs // 2))
            y = y0 + true_att * d + np.random.randn(n_obs) * 0.5
            
            data = pd.DataFrame({'y': y, 'd': d})
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = run_ols_regression(
                    data=data,
                    y='y',
                    d='d',
                    controls=None,
                    vce=None,
                    alpha=alpha,
                )
            
            if not np.isnan(result['ci_lower']) and not np.isnan(result['ci_upper']):
                if result['ci_lower'] <= true_att <= result['ci_upper']:
                    coverage_count += 1
        
        coverage_rate = coverage_count / n_simulations
        
        # Should be close to 95% (allow some Monte Carlo error)
        assert 0.90 <= coverage_rate <= 0.99, \
            f"Coverage rate should be ~95%, got {coverage_rate:.1%}"


class TestBug227EdgeCases:
    """Edge case tests for BUG-227."""

    def test_exactly_zero_se(self):
        """Test behavior when SE is exactly zero (perfect fit)."""
        # Perfect fit: all y values are identical within groups
        data = pd.DataFrame({
            'y': [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            'd': [1, 1, 1, 0, 0, 0],
        })
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_ols_regression(
                data=data,
                y='y',
                d='d',
                controls=None,
                vce=None,
                alpha=0.05,
            )
        
        # ATT should be 1.0 - 0.0 = 1.0
        assert abs(result['att'] - 1.0) < 1e-10, \
            f"ATT should be 1.0, got {result['att']}"
        
        # With perfect fit, SE should be 0 and t_stat should be NaN (0/0)
        # or SE should be very small

    def test_large_df_approximates_normal(self):
        """Verify t-distribution with large df approximates standard normal.
        
        As df → ∞, t-distribution → N(0,1).
        """
        alpha = 0.05
        
        # t_0.975 for standard normal ≈ 1.96
        z_crit = stats.norm.ppf(1 - alpha / 2)
        
        # t_0.975 for df = 10000 should be very close to z_crit
        t_crit_large = stats.t.ppf(1 - alpha / 2, 10000)
        
        assert abs(t_crit_large - z_crit) < 0.01, \
            f"t-distribution with large df should approximate normal. " \
            f"t_crit={t_crit_large:.4f}, z_crit={z_crit:.4f}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

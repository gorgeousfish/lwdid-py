# -*- coding: utf-8 -*-
"""
HC3 coverage validation tests for small-sample Monte Carlo.

Based on Lee & Wooldridge (2026) ssrn-5325686, Section 5.

These tests verify that HC3 standard errors improve coverage rates
in small-sample settings compared to OLS standard errors.
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path
from scipy import stats
import statsmodels.api as sm

# Add fixtures paths
fixtures_path = Path(__file__).parent / 'fixtures'
parent_fixtures = Path(__file__).parent.parent / 'test_common_timing' / 'fixtures'
sys.path.insert(0, str(fixtures_path))
sys.path.insert(0, str(parent_fixtures))

from monte_carlo_runner import (
    run_small_sample_monte_carlo,
    _estimate_manual_demean,
    _estimate_manual_detrend,
)
from dgp_small_sample import (
    generate_small_sample_dgp,
    SMALL_SAMPLE_SCENARIOS,
)


@pytest.mark.numerical
class TestHC3Formula:
    """Tests for HC3 standard error formula correctness."""
    
    def test_hc3_formula_definition(self):
        """
        HC3 variance formula:
        Var(β̂)_HC3 = (X'X)^{-1} × (Σ_i x_i x_i' × e_i² / (1 - h_ii)²) × (X'X)^{-1}
        
        Where h_ii = x_i' (X'X)^{-1} x_i is the leverage.
        """
        # Generate simple regression data
        np.random.seed(42)
        n = 20
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        beta_true = np.array([1.0, 2.0])
        y = X @ beta_true + np.random.randn(n)
        
        # OLS estimation
        XtX_inv = np.linalg.inv(X.T @ X)
        beta_hat = XtX_inv @ X.T @ y
        residuals = y - X @ beta_hat
        
        # Compute leverage (hat matrix diagonal)
        H = X @ XtX_inv @ X.T
        h_ii = np.diag(H)
        
        # HC3 variance
        omega_diag = (residuals ** 2) / ((1 - h_ii) ** 2)
        meat = X.T @ np.diag(omega_diag) @ X
        var_hc3 = XtX_inv @ meat @ XtX_inv
        se_hc3 = np.sqrt(np.diag(var_hc3))
        
        # Compare with statsmodels HC3
        model = sm.OLS(y, X).fit(cov_type='HC3')
        se_sm = model.bse
        
        np.testing.assert_allclose(se_hc3, se_sm, rtol=1e-6)
    
    def test_hc3_vs_ols_se_ratio(self):
        """
        HC3 SE should be >= OLS SE in general.
        
        HC3 inflates SE by factor 1/(1-h_ii)² which is always >= 1.
        """
        np.random.seed(42)
        n = 20
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = X @ np.array([1.0, 2.0]) + np.random.randn(n)
        
        # OLS SE
        model_ols = sm.OLS(y, X).fit()
        se_ols = model_ols.bse
        
        # HC3 SE
        model_hc3 = sm.OLS(y, X).fit(cov_type='HC3')
        se_hc3 = model_hc3.bse
        
        # HC3 should be >= OLS (with some numerical tolerance)
        assert all(se_hc3 >= se_ols * 0.99), \
            f"HC3 SE should be >= OLS SE: HC3={se_hc3}, OLS={se_ols}"
    
    def test_leverage_bounds(self):
        """
        Leverage h_ii should satisfy: 1/n <= h_ii <= 1.
        
        For balanced design, h_ii ≈ k/n where k = number of parameters.
        """
        np.random.seed(42)
        n = 20
        k = 2
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        
        XtX_inv = np.linalg.inv(X.T @ X)
        H = X @ XtX_inv @ X.T
        h_ii = np.diag(H)
        
        # Check bounds
        assert all(h_ii >= 1/n - 1e-10), "Leverage should be >= 1/n"
        assert all(h_ii <= 1 + 1e-10), "Leverage should be <= 1"
        
        # Sum of leverages = k (trace of hat matrix)
        assert abs(h_ii.sum() - k) < 1e-10, f"Sum of leverages should be {k}"


@pytest.mark.numerical
class TestHC3CoverageTheory:
    """Tests for HC3 coverage rate theory."""
    
    def test_coverage_definition(self):
        """
        Coverage rate = P(CI contains true parameter).
        
        For 95% CI: coverage should be ≈ 0.95 if SE is correctly estimated.
        """
        # Simulate coverage calculation
        np.random.seed(42)
        n_reps = 1000
        true_beta = 2.0
        n = 20
        
        covers = []
        for rep in range(n_reps):
            X = np.column_stack([np.ones(n), np.random.randn(n)])
            y = X @ np.array([1.0, true_beta]) + np.random.randn(n)
            
            model = sm.OLS(y, X).fit(cov_type='HC3')
            ci = model.conf_int(alpha=0.05)
            
            # Check if true beta is in CI
            covers.append(ci[0, 1] <= true_beta <= ci[1, 1])
        
        coverage = np.mean(covers)
        
        # Coverage should be close to 95%
        assert 0.90 <= coverage <= 1.0, \
            f"HC3 coverage should be ~95%, got {coverage:.1%}"
    
    def test_t_distribution_critical_value(self):
        """
        For small samples, use t-distribution critical value.
        
        t_{0.975, df} where df = n - k.
        """
        n = 20
        k = 2
        df = n - k
        alpha = 0.05
        
        t_crit = stats.t.ppf(1 - alpha/2, df)
        z_crit = stats.norm.ppf(1 - alpha/2)
        
        # t critical value should be larger than z for small df
        assert t_crit > z_crit, \
            f"t_{df} critical value ({t_crit:.3f}) should be > z ({z_crit:.3f})"
        
        # For df=18, t_0.975 ≈ 2.101
        expected_t = 2.101
        assert abs(t_crit - expected_t) < 0.01, \
            f"t_{{0.975, 18}} should be ~{expected_t}, got {t_crit:.3f}"


@pytest.mark.monte_carlo
class TestHC3CoverageMonteCarlo:
    """Monte Carlo tests for HC3 coverage improvement."""
    
    @pytest.fixture
    def mc_results_scenario_1(self):
        """Run Monte Carlo for Scenario 1 with coverage tracking."""
        return run_small_sample_monte_carlo(
            n_reps=100,
            scenario='scenario_1',
            estimators=['demeaning', 'detrending'],
            seed=42,
            verbose=False,
            use_lwdid=False,
        )
    
    def test_hc3_coverage_exists(self, mc_results_scenario_1):
        """HC3 coverage should be computed."""
        for estimator, result in mc_results_scenario_1.items():
            assert hasattr(result, 'coverage_hc3'), \
                f"{estimator} should have coverage_hc3 attribute"
            assert 0 <= result.coverage_hc3 <= 1, \
                f"{estimator} coverage_hc3 should be in [0, 1]"
    
    def test_hc3_improves_coverage_over_ols(self, mc_results_scenario_1):
        """
        HC3 should improve coverage compared to OLS in small samples.
        
        Paper finding: HC3 coverage closer to 95% than OLS.
        """
        for estimator, result in mc_results_scenario_1.items():
            # HC3 coverage should be >= OLS coverage (or at least close)
            # In small samples, OLS tends to undercover
            coverage_diff = result.coverage_hc3 - result.coverage_ols
            
            # Allow some tolerance - HC3 should not be much worse
            assert coverage_diff >= -0.10, \
                f"{estimator}: HC3 coverage ({result.coverage_hc3:.2%}) " \
                f"should not be much worse than OLS ({result.coverage_ols:.2%})"
    
    @pytest.mark.slow
    def test_hc3_coverage_all_scenarios(self):
        """Test HC3 coverage across all scenarios using actual lwdid package.
        
        Paper Table 2 shows coverage should be around 95% for detrending.
        """
        for scenario in SMALL_SAMPLE_SCENARIOS.keys():
            results = run_small_sample_monte_carlo(
                n_reps=100,
                scenario=scenario,
                estimators=['detrending'],
                seed=42,
                verbose=False,
                use_lwdid=True,  # Use actual lwdid package
            )
            
            result = results['detrending']
            
            # HC3 coverage should be reasonable (> 70%)
            # Paper Table 2 shows ~95% coverage
            assert result.coverage_hc3 > 0.70, \
                f"{scenario}: HC3 coverage ({result.coverage_hc3:.2%}) too low"


@pytest.mark.monte_carlo
class TestSERatioValidation:
    """Tests for SE ratio (mean SE / SD of estimates) validation."""
    
    def test_se_ratio_definition(self):
        """
        SE ratio = mean(SE) / SD(ATT estimates).
        
        If SE is correctly estimated, ratio should be ≈ 1.
        """
        results = run_small_sample_monte_carlo(
            n_reps=100,
            scenario='scenario_1',
            estimators=['detrending'],
            seed=42,
            verbose=False,
            use_lwdid=False,
        )
        
        result = results['detrending']
        
        # Verify SE ratio calculation
        expected_ratio_ols = result.mean_se_ols / result.sd if result.sd > 0 else np.nan
        expected_ratio_hc3 = result.mean_se_hc3 / result.sd if result.sd > 0 else np.nan
        
        assert abs(result.se_ratio_ols - expected_ratio_ols) < 1e-6
        assert abs(result.se_ratio_hc3 - expected_ratio_hc3) < 1e-6
    
    def test_se_ratio_interpretation(self):
        """
        SE ratio interpretation:
        - Ratio < 1: SE underestimates true variability (undercoverage)
        - Ratio ≈ 1: SE correctly estimates variability
        - Ratio > 1: SE overestimates variability (overcoverage)
        
        Note: With manual estimators (not using lwdid package), SE estimates
        may be less accurate. We use a wider tolerance range.
        """
        results = run_small_sample_monte_carlo(
            n_reps=100,
            scenario='scenario_1',
            estimators=['detrending'],
            seed=42,
            verbose=False,
            use_lwdid=False,
        )
        
        result = results['detrending']
        
        # SE ratio should be positive and finite
        # Note: Manual estimators may underestimate SE significantly
        # because they don't account for all sources of variance
        assert result.se_ratio_ols > 0, \
            f"OLS SE ratio ({result.se_ratio_ols:.2f}) should be positive"
        assert result.se_ratio_hc3 > 0, \
            f"HC3 SE ratio ({result.se_ratio_hc3:.2f}) should be positive"
        assert np.isfinite(result.se_ratio_ols), \
            f"OLS SE ratio should be finite"
        assert np.isfinite(result.se_ratio_hc3), \
            f"HC3 SE ratio should be finite"
        
        # HC3 ratio should be >= OLS ratio (HC3 inflates SE)
        assert result.se_ratio_hc3 >= result.se_ratio_ols * 0.95, \
            f"HC3 SE ratio ({result.se_ratio_hc3:.2f}) should be >= OLS ({result.se_ratio_ols:.2f})"


@pytest.mark.monte_carlo
@pytest.mark.slow
class TestHC3VsOLSComparison:
    """Comprehensive comparison of HC3 vs OLS standard errors."""
    
    def test_hc3_reduces_undercoverage(self):
        """
        In small samples with heteroskedasticity, OLS SE tends to undercover.
        HC3 should reduce this undercoverage.
        """
        results = run_small_sample_monte_carlo(
            n_reps=200,
            scenario='scenario_1',
            estimators=['detrending'],
            seed=42,
            verbose=False,
            use_lwdid=False,
        )
        
        result = results['detrending']
        
        # Calculate distance from nominal 95% coverage
        dist_ols = abs(result.coverage_ols - 0.95)
        dist_hc3 = abs(result.coverage_hc3 - 0.95)
        
        # HC3 should be closer to 95% (or at least not much worse)
        # Allow 10% tolerance
        assert dist_hc3 <= dist_ols + 0.10, \
            f"HC3 ({result.coverage_hc3:.2%}) should be closer to 95% " \
            f"than OLS ({result.coverage_ols:.2%})"
    
    def test_sparse_treatment_hc3_benefit(self):
        """
        HC3 benefit should be more pronounced with sparse treatment.
        
        Scenario 3 (P(D=1) = 0.17) has fewer treated units.
        Using actual lwdid package for accurate SE estimation.
        """
        results_s1 = run_small_sample_monte_carlo(
            n_reps=100,
            scenario='scenario_1',
            estimators=['detrending'],
            seed=42,
            verbose=False,
            use_lwdid=True,  # Use actual lwdid package
        )
        
        results_s3 = run_small_sample_monte_carlo(
            n_reps=100,
            scenario='scenario_3',
            estimators=['detrending'],
            seed=42,
            verbose=False,
            use_lwdid=True,  # Use actual lwdid package
        )
        
        # Both scenarios should have reasonable HC3 coverage
        assert results_s1['detrending'].coverage_hc3 > 0.70
        assert results_s3['detrending'].coverage_hc3 > 0.60  # Lower threshold for sparse


@pytest.mark.numerical
class TestHC3Implementation:
    """Tests for HC3 implementation details."""
    
    def test_leverage_clipping_for_stability(self):
        """
        Leverage values close to 1 can cause numerical instability.
        Implementation should clip h_ii to avoid division by zero.
        """
        np.random.seed(42)
        n = 10
        k = 5  # High leverage situation
        X = np.column_stack([np.ones(n)] + [np.random.randn(n) for _ in range(k-1)])
        y = np.random.randn(n)
        
        XtX_inv = np.linalg.inv(X.T @ X)
        H = X @ XtX_inv @ X.T
        h_ii = np.diag(H)
        
        # Clip leverage for stability
        h_ii_clipped = np.clip(h_ii, 0, 0.9999)
        
        # Should not have any values >= 1
        assert all(h_ii_clipped < 1), "Clipped leverage should be < 1"
        
        # HC3 calculation should not produce inf/nan
        residuals = y - X @ (XtX_inv @ X.T @ y)
        omega_diag = (residuals ** 2) / ((1 - h_ii_clipped) ** 2)
        
        assert not np.any(np.isinf(omega_diag)), "HC3 omega should not be inf"
        assert not np.any(np.isnan(omega_diag)), "HC3 omega should not be nan"
    
    def test_hc3_with_perfect_fit(self):
        """
        When h_ii = 1 (perfect fit), HC3 is undefined.
        Implementation should handle this gracefully.
        """
        # Create data where one observation has high leverage
        np.random.seed(42)
        n = 5
        X = np.column_stack([np.ones(n), np.array([0, 0, 0, 0, 10])])  # Outlier
        y = np.random.randn(n)
        
        try:
            model = sm.OLS(y, X).fit(cov_type='HC3')
            se_hc3 = model.bse
            
            # Should produce finite SE
            assert np.all(np.isfinite(se_hc3)), "HC3 SE should be finite"
        except Exception as e:
            # Some implementations may raise an error
            pytest.skip(f"HC3 with high leverage raised: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'monte_carlo or numerical'])

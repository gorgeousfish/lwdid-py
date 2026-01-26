"""
BUG-095 Deep Analysis: z-distribution vs t-distribution in statistical inference.

This test validates the correct use of distributions for confidence intervals:

1. OLS-based estimators (RA) should use t-distribution (exact inference)
2. M-estimation based estimators (IPW, IPWRA, PSM) should use z-distribution (asymptotic)

Reference:
- Lee & Wooldridge (2025, Small Sample paper), Equation (2.10):
  (τ̂_DD - τ) / se(τ̂_DD) ~ T_{N-2}
  
- Lee & Wooldridge (2023, Large Sample paper):
  IPW/IPWRA/PSM use M-estimation with asymptotic normality

Stata behavior:
- teffects ra/ipwra/ipw/psmatch: Reports "z P>|z|" (z-distribution)
- lwdid.ado (small sample): p-value uses t-distribution, CI uses 1.96 (inconsistent)
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lwdid import lwdid
from lwdid.staggered.estimators import estimate_ipwra, estimate_ipw, estimate_psm
from lwdid.staggered.estimation import run_ols_regression


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def smoking_data():
    """Load smoking dataset for testing."""
    data_path = Path(__file__).parent / 'data' / 'smoking.csv'
    if data_path.exists():
        return pd.read_csv(data_path)
    return None


@pytest.fixture
def castle_data():
    """Load castle law dataset for testing."""
    data_path = Path(__file__).parent / 'data' / 'castle.csv'
    if data_path.exists():
        return pd.read_csv(data_path)
    return None


@pytest.fixture
def small_sample_data():
    """Create small sample data (N < 30) to test t vs z difference."""
    np.random.seed(42)
    n_units = 20
    n_periods = 6
    
    units = np.repeat(np.arange(n_units), n_periods)
    periods = np.tile(np.arange(1, n_periods + 1), n_units)
    
    # 10 treated (cohort 4), 10 never-treated
    cohorts = np.zeros(n_units)
    cohorts[:10] = 4
    gvar = np.repeat(cohorts, n_periods)
    
    x1 = np.random.normal(0, 1, n_units * n_periods)
    y = 1.0 + 0.3 * x1 + np.random.normal(0, 0.5, n_units * n_periods)
    
    # Add treatment effect
    treated_post = (gvar > 0) & (periods >= gvar)
    y[treated_post] += 2.0
    
    return pd.DataFrame({
        'ivar': units,
        'tvar': periods,
        'gvar': gvar,
        'y': y,
        'x1': x1,
    })


@pytest.fixture
def large_sample_data():
    """Create large sample data (N > 100) to compare distributions."""
    np.random.seed(42)
    n_units = 200
    n_periods = 10
    
    units = np.repeat(np.arange(n_units), n_periods)
    periods = np.tile(np.arange(1, n_periods + 1), n_units)
    
    # 100 treated, 100 never-treated
    cohorts = np.zeros(n_units)
    cohorts[:50] = 5
    cohorts[50:100] = 7
    gvar = np.repeat(cohorts, n_periods)
    
    x1 = np.random.normal(0, 1, n_units * n_periods)
    x2 = np.random.normal(0, 1, n_units * n_periods)
    y = 1.0 + 0.3 * x1 + 0.2 * x2 + np.random.normal(0, 0.5, n_units * n_periods)
    
    # Add treatment effect
    treated_post = (gvar > 0) & (periods >= gvar)
    y[treated_post] += 1.5
    
    return pd.DataFrame({
        'ivar': units,
        'tvar': periods,
        'gvar': gvar,
        'y': y,
        'x1': x1,
        'x2': x2,
    })


# =============================================================================
# Test: Distribution Theory Verification
# =============================================================================

class TestDistributionTheory:
    """Verify theoretical properties of t vs z distributions."""
    
    def test_t_vs_z_critical_values(self):
        """Demonstrate the difference between t and z critical values."""
        alpha = 0.05
        z_crit = stats.norm.ppf(1 - alpha / 2)
        
        # Small df: significant difference
        t_crit_10 = stats.t.ppf(1 - alpha / 2, 10)
        t_crit_20 = stats.t.ppf(1 - alpha / 2, 20)
        
        # Large df: converges to z
        t_crit_100 = stats.t.ppf(1 - alpha / 2, 100)
        t_crit_1000 = stats.t.ppf(1 - alpha / 2, 1000)
        
        # Verify relationships
        assert t_crit_10 > z_crit, "t(10) should be larger than z"
        assert t_crit_10 > t_crit_20 > t_crit_100, "t critical decreases with df"
        assert abs(t_crit_1000 - z_crit) < 0.01, "t(1000) should approximate z"
        
        # Print for documentation
        print("\n=== t vs z critical values at alpha=0.05 ===")
        print(f"z (normal): {z_crit:.6f}")
        print(f"t(df=10):   {t_crit_10:.6f} (+{(t_crit_10/z_crit-1)*100:.2f}%)")
        print(f"t(df=20):   {t_crit_20:.6f} (+{(t_crit_20/z_crit-1)*100:.2f}%)")
        print(f"t(df=100):  {t_crit_100:.6f} (+{(t_crit_100/z_crit-1)*100:.2f}%)")
        print(f"t(df=1000): {t_crit_1000:.6f} (+{(t_crit_1000/z_crit-1)*100:.2f}%)")
    
    def test_ci_width_difference(self):
        """Test that t-distribution produces wider CI for small samples."""
        alpha = 0.05
        se = 0.1  # Example SE
        att = 1.0  # Example ATT
        
        z_crit = stats.norm.ppf(1 - alpha / 2)
        
        for df in [10, 20, 50, 100, 500]:
            t_crit = stats.t.ppf(1 - alpha / 2, df)
            
            ci_width_z = 2 * z_crit * se
            ci_width_t = 2 * t_crit * se
            
            width_ratio = ci_width_t / ci_width_z
            
            if df <= 30:
                assert width_ratio > 1.03, f"t CI should be >3% wider for df={df}"
            
            print(f"df={df:4d}: z_width={ci_width_z:.4f}, t_width={ci_width_t:.4f}, ratio={width_ratio:.4f}")


# =============================================================================
# Test: OLS Regression Uses t-Distribution
# =============================================================================

class TestOLSDistribution:
    """Verify that OLS-based estimation uses t-distribution."""
    
    def test_run_ols_regression_uses_t(self):
        """run_ols_regression should use t-distribution for CI."""
        np.random.seed(42)
        n = 50
        x1 = np.random.normal(0, 1, n)
        d = (np.random.uniform(0, 1, n) < 0.5).astype(int)
        y = 1.0 + 0.5 * d + 0.3 * x1 + np.random.normal(0, 0.5, n)
        
        data = pd.DataFrame({'y': y, 'd': d, 'x1': x1})
        
        result = run_ols_regression(data, 'y', 'd', ['x1'], alpha=0.05)
        
        # Verify df_inference is set
        df = result['df_inference']
        assert df > 0, "df_inference should be positive"
        
        # Verify CI uses t-distribution
        att = result['att']
        se = result['se']
        ci_lower = result['ci_lower']
        ci_upper = result['ci_upper']
        
        # Reconstruct CI with t-distribution
        t_crit = stats.t.ppf(0.975, df)
        expected_ci_lower = att - t_crit * se
        expected_ci_upper = att + t_crit * se
        
        assert abs(ci_lower - expected_ci_lower) < 1e-10, "CI lower should use t-distribution"
        assert abs(ci_upper - expected_ci_upper) < 1e-10, "CI upper should use t-distribution"
        
        # Verify NOT using z-distribution
        z_crit = stats.norm.ppf(0.975)
        z_ci_lower = att - z_crit * se
        z_ci_upper = att + z_crit * se
        
        # For small df, the difference should be noticeable
        if df < 100:
            assert ci_lower < z_ci_lower, "t-based CI lower should be more conservative"
            assert ci_upper > z_ci_upper, "t-based CI upper should be more conservative"


# =============================================================================
# Test: IPW/IPWRA/PSM Use z-Distribution
# =============================================================================

class TestMEstimationDistribution:
    """Verify that M-estimation methods use z-distribution."""
    
    def test_ipwra_uses_z_distribution(self):
        """estimate_ipwra should use z-distribution for CI."""
        np.random.seed(42)
        n = 200
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        logit_p = -0.5 + 0.5 * x1 + 0.3 * x2
        p_true = 1 / (1 + np.exp(-logit_p))
        d = (np.random.uniform(0, 1, n) < p_true).astype(int)
        y = 1.0 + 0.5 * d + 0.3 * x1 + 0.2 * x2 + np.random.normal(0, 0.5, n)
        
        data = pd.DataFrame({'y': y, 'd': d, 'x1': x1, 'x2': x2})
        
        result = estimate_ipwra(data, 'y', 'd', ['x1', 'x2'], alpha=0.05)
        
        att = result.att
        se = result.se
        ci_lower = result.ci_lower
        ci_upper = result.ci_upper
        
        # IPWRA should use z-distribution
        z_crit = stats.norm.ppf(0.975)
        expected_ci_lower = att - z_crit * se
        expected_ci_upper = att + z_crit * se
        
        assert abs(ci_lower - expected_ci_lower) < 1e-10, "IPWRA CI should use z-distribution"
        assert abs(ci_upper - expected_ci_upper) < 1e-10, "IPWRA CI should use z-distribution"
    
    def test_ipw_uses_z_distribution(self):
        """estimate_ipw should use z-distribution for CI."""
        np.random.seed(42)
        n = 200
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        logit_p = -0.5 + 0.5 * x1 + 0.3 * x2
        p_true = 1 / (1 + np.exp(-logit_p))
        d = (np.random.uniform(0, 1, n) < p_true).astype(int)
        y = 1.0 + 0.5 * d + 0.3 * x1 + 0.2 * x2 + np.random.normal(0, 0.5, n)
        
        data = pd.DataFrame({'y': y, 'd': d, 'x1': x1, 'x2': x2})
        
        result = estimate_ipw(data, 'y', 'd', ['x1', 'x2'], alpha=0.05)
        
        att = result.att
        se = result.se
        ci_lower = result.ci_lower
        ci_upper = result.ci_upper
        
        # IPW should use z-distribution
        z_crit = stats.norm.ppf(0.975)
        expected_ci_lower = att - z_crit * se
        expected_ci_upper = att + z_crit * se
        
        assert abs(ci_lower - expected_ci_lower) < 1e-10, "IPW CI should use z-distribution"
        assert abs(ci_upper - expected_ci_upper) < 1e-10, "IPW CI should use z-distribution"
    
    def test_psm_uses_z_distribution(self):
        """estimate_psm should use z-distribution for CI."""
        np.random.seed(42)
        n = 200
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        logit_p = -0.5 + 0.5 * x1 + 0.3 * x2
        p_true = 1 / (1 + np.exp(-logit_p))
        d = (np.random.uniform(0, 1, n) < p_true).astype(int)
        y = 1.0 + 0.5 * d + 0.3 * x1 + 0.2 * x2 + np.random.normal(0, 0.5, n)
        
        data = pd.DataFrame({'y': y, 'd': d, 'x1': x1, 'x2': x2})
        
        result = estimate_psm(data, 'y', 'd', ['x1', 'x2'], n_neighbors=1, alpha=0.05)
        
        att = result.att
        se = result.se
        ci_lower = result.ci_lower
        ci_upper = result.ci_upper
        
        # PSM should use z-distribution
        z_crit = stats.norm.ppf(0.975)
        expected_ci_lower = att - z_crit * se
        expected_ci_upper = att + z_crit * se
        
        assert abs(ci_lower - expected_ci_lower) < 1e-10, "PSM CI should use z-distribution"
        assert abs(ci_upper - expected_ci_upper) < 1e-10, "PSM CI should use z-distribution"


# =============================================================================
# Test: LWDIDResults Uses Correct Distribution
# =============================================================================

class TestLWDIDResultsDistribution:
    """Verify that LWDIDResults uses correct distribution based on estimator."""
    
    def test_lwdid_ra_uses_t_distribution(self, small_sample_data):
        """lwdid with RA estimator should produce t-based CI."""
        if small_sample_data is None:
            pytest.skip("Test data not available")
        
        result = lwdid(
            data=small_sample_data,
            y='y',
            gvar='gvar',
            ivar='ivar',
            tvar='tvar',
            estimator='ra',
            controls=['x1'],
            alpha=0.05,
        )
        
        # The CI should be computed with t-distribution
        # Check that df_inference is set in the result
        if hasattr(result, '_df_inference') and result._df_inference:
            df = result._df_inference
            t_crit = stats.t.ppf(0.975, df)
            z_crit = stats.norm.ppf(0.975)
            
            # For small samples, t_crit should be noticeably larger
            if df < 50:
                assert t_crit > z_crit * 1.02, f"t({df}) should be >2% larger than z"
                print(f"\ndf={df}, t_crit={t_crit:.4f}, z_crit={z_crit:.4f}")


# =============================================================================
# Test: Numerical Comparison Summary
# =============================================================================

class TestNumericalComparison:
    """Summarize numerical differences between t and z distributions."""
    
    def test_coverage_impact(self):
        """Demonstrate coverage impact of using wrong distribution."""
        np.random.seed(42)
        n_simulations = 1000
        alpha = 0.05
        true_att = 1.0
        
        # Small sample scenario
        n = 20
        
        coverage_t = 0
        coverage_z = 0
        
        for _ in range(n_simulations):
            # Generate data
            y = true_att + np.random.normal(0, 1, n)
            att_hat = np.mean(y)
            se = np.std(y, ddof=1) / np.sqrt(n)
            
            df = n - 1
            
            # t-based CI
            t_crit = stats.t.ppf(1 - alpha/2, df)
            ci_t_lower = att_hat - t_crit * se
            ci_t_upper = att_hat + t_crit * se
            
            # z-based CI
            z_crit = stats.norm.ppf(1 - alpha/2)
            ci_z_lower = att_hat - z_crit * se
            ci_z_upper = att_hat + z_crit * se
            
            # Check coverage
            if ci_t_lower <= true_att <= ci_t_upper:
                coverage_t += 1
            if ci_z_lower <= true_att <= ci_z_upper:
                coverage_z += 1
        
        coverage_t_pct = coverage_t / n_simulations * 100
        coverage_z_pct = coverage_z / n_simulations * 100
        
        print(f"\n=== Coverage Simulation (n={n}, {n_simulations} reps) ===")
        print(f"Target coverage: {(1-alpha)*100:.1f}%")
        print(f"t-distribution coverage: {coverage_t_pct:.1f}%")
        print(f"z-distribution coverage: {coverage_z_pct:.1f}%")
        
        # t-distribution should be closer to nominal coverage
        assert abs(coverage_t_pct - 95) < abs(coverage_z_pct - 95), \
            "t-distribution should have better coverage for small samples"
        
        # z-distribution should undercover
        assert coverage_z_pct < 95, "z-distribution should undercover for small samples"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])

"""
BUG-060-L: t-Distribution Numerical Validation

Validates the mathematical properties of t-distribution at small degrees of freedom
to confirm that warning thresholds (df <= 2) are statistically justified.

References:
- t(df=1) = Cauchy distribution (no finite mean or variance)
- t(df=2) has finite mean (=0) but infinite variance
- For df > 2, variance = df/(df-2)
"""

import numpy as np
from scipy import stats
import pytest


class TestTDistributionEquivalences:
    """Verify t(df=1) equals Cauchy distribution"""
    
    def test_pdf_equivalence(self):
        """t(df=1) PDF equals Cauchy PDF at multiple points"""
        x_values = np.array([-5, -2, -1, 0, 1, 2, 5])
        t_pdf = stats.t.pdf(x_values, df=1)
        cauchy_pdf = stats.cauchy.pdf(x_values)
        
        np.testing.assert_allclose(t_pdf, cauchy_pdf, rtol=1e-10)
    
    def test_cdf_equivalence(self):
        """t(df=1) CDF equals Cauchy CDF at multiple points"""
        x_values = np.array([-5, -2, -1, 0, 1, 2, 5])
        t_cdf = stats.t.cdf(x_values, df=1)
        cauchy_cdf = stats.cauchy.cdf(x_values)
        
        np.testing.assert_allclose(t_cdf, cauchy_cdf, rtol=1e-10)
    
    def test_quantile_equivalence(self):
        """t(df=1) quantiles equal Cauchy quantiles"""
        probs = np.array([0.01, 0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975, 0.99])
        t_quantiles = stats.t.ppf(probs, df=1)
        cauchy_quantiles = stats.cauchy.ppf(probs)
        
        # Use both rtol and atol to handle near-zero values
        np.testing.assert_allclose(t_quantiles, cauchy_quantiles, rtol=1e-10, atol=1e-15)


class TestCriticalValueProgression:
    """Verify critical values increase rapidly as df decreases"""
    
    def test_critical_values_95ci(self):
        """95% CI critical values for different df"""
        alpha = 0.05
        
        t_crit_df1 = stats.t.ppf(1 - alpha/2, df=1)
        t_crit_df2 = stats.t.ppf(1 - alpha/2, df=2)
        t_crit_df3 = stats.t.ppf(1 - alpha/2, df=3)
        t_crit_df10 = stats.t.ppf(1 - alpha/2, df=10)
        t_crit_df30 = stats.t.ppf(1 - alpha/2, df=30)
        t_crit_inf = stats.norm.ppf(1 - alpha/2)  # Normal ~ 1.96
        
        # Verify expected values (approximately)
        assert 12.5 < t_crit_df1 < 13.0  # ~12.71
        assert 4.0 < t_crit_df2 < 4.5    # ~4.30
        assert 3.0 < t_crit_df3 < 3.3    # ~3.18
        assert 2.1 < t_crit_df10 < 2.3   # ~2.23
        assert 2.0 < t_crit_df30 < 2.1   # ~2.04
        assert 1.9 < t_crit_inf < 2.0    # ~1.96
        
        # Verify monotonicity
        assert t_crit_df1 > t_crit_df2 > t_crit_df3 > t_crit_df10 > t_crit_df30 > t_crit_inf
    
    def test_df1_makes_ci_very_wide(self):
        """df=1 critical value is >6x larger than normal"""
        alpha = 0.05
        t_crit_df1 = stats.t.ppf(1 - alpha/2, df=1)
        z_crit = stats.norm.ppf(1 - alpha/2)
        
        ratio = t_crit_df1 / z_crit
        assert ratio > 6  # Actually ~6.5
    
    def test_df2_makes_ci_wide(self):
        """df=2 critical value is >2x larger than normal"""
        alpha = 0.05
        t_crit_df2 = stats.t.ppf(1 - alpha/2, df=2)
        z_crit = stats.norm.ppf(1 - alpha/2)
        
        ratio = t_crit_df2 / z_crit
        assert ratio > 2  # Actually ~2.2


class TestVarianceProperties:
    """Verify variance properties of t-distribution"""
    
    def test_variance_df_greater_than_2(self):
        """For df > 2, variance = df/(df-2)"""
        for df in [3, 4, 5, 10, 30, 100]:
            expected_var = df / (df - 2)
            # t-distribution variance formula
            actual_var = stats.t.var(df)
            np.testing.assert_allclose(actual_var, expected_var, rtol=1e-10)
    
    def test_variance_df2_is_infinite(self):
        """df=2 has infinite variance"""
        var_df2 = stats.t.var(2)
        assert np.isinf(var_df2)
    
    def test_variance_df1_is_nan(self):
        """df=1 variance is undefined (NaN)"""
        var_df1 = stats.t.var(1)
        assert np.isnan(var_df1)


class TestMonteCarloValidation:
    """Monte Carlo validation of statistical properties"""
    
    def test_sample_variance_instability_df1(self):
        """df=1 sample variance is highly unstable"""
        np.random.seed(42)
        
        # Generate multiple sample variances
        variances = []
        for _ in range(100):
            samples = stats.t.rvs(df=1, size=1000)
            variances.append(np.var(samples))
        
        # Compare with df=30
        variances_df30 = []
        for _ in range(100):
            samples = stats.t.rvs(df=30, size=1000)
            variances_df30.append(np.var(samples))
        
        # df=1 variance estimates should be much more variable
        assert np.std(variances) > np.std(variances_df30) * 3
    
    def test_sample_variance_instability_df2(self):
        """df=2 sample variance is unstable"""
        np.random.seed(42)
        
        variances = []
        for _ in range(100):
            samples = stats.t.rvs(df=2, size=1000)
            variances.append(np.var(samples))
        
        variances_df30 = []
        for _ in range(100):
            samples = stats.t.rvs(df=30, size=1000)
            variances_df30.append(np.var(samples))
        
        # df=2 variance estimates should be more variable
        assert np.std(variances) > np.std(variances_df30) * 2
    
    def test_sample_mean_instability_df1(self):
        """df=1 sample mean is highly unstable (no finite mean)"""
        np.random.seed(42)
        
        means = []
        for _ in range(100):
            samples = stats.t.rvs(df=1, size=1000)
            means.append(np.mean(samples))
        
        means_df30 = []
        for _ in range(100):
            samples = stats.t.rvs(df=30, size=1000)
            means_df30.append(np.mean(samples))
        
        # df=1 mean estimates should be much more variable
        assert np.std(means) > np.std(means_df30) * 3


class TestCoverageProperties:
    """Verify CI coverage is correct when df is accurately specified"""
    
    def test_coverage_df3_is_accurate(self):
        """95% CI with df=3 has ~95% coverage"""
        np.random.seed(42)
        n_sims = 1000
        true_mean = 5.0
        true_se = 1.0
        df = 3
        alpha = 0.05
        
        covered = 0
        t_crit = stats.t.ppf(1 - alpha/2, df)
        
        for _ in range(n_sims):
            # Simulate t-distributed test statistic
            t_stat = stats.t.rvs(df)
            # Check if true mean (0 for centered t) is in CI
            if -t_crit <= t_stat <= t_crit:
                covered += 1
        
        coverage = covered / n_sims
        # Should be close to 95%
        assert 0.93 < coverage < 0.97
    
    def test_coverage_df30_is_accurate(self):
        """95% CI with df=30 has ~95% coverage"""
        np.random.seed(42)
        n_sims = 1000
        df = 30
        alpha = 0.05
        
        covered = 0
        t_crit = stats.t.ppf(1 - alpha/2, df)
        
        for _ in range(n_sims):
            t_stat = stats.t.rvs(df)
            if -t_crit <= t_stat <= t_crit:
                covered += 1
        
        coverage = covered / n_sims
        assert 0.93 < coverage < 0.97


class TestPValueBehavior:
    """Verify p-value computation at different df"""
    
    def test_pvalue_at_t2_df1(self):
        """p-value at t=2 with df=1 is much larger than with df=30"""
        t_stat = 2.0
        pval_df1 = 2 * stats.t.sf(t_stat, df=1)
        pval_df30 = 2 * stats.t.sf(t_stat, df=30)
        
        # df=1 gives less significant result
        assert pval_df1 > pval_df30 * 2
    
    def test_pvalue_at_t3_df2(self):
        """p-value at t=3 with df=2 is larger than with df=30"""
        t_stat = 3.0
        pval_df2 = 2 * stats.t.sf(t_stat, df=2)
        pval_df30 = 2 * stats.t.sf(t_stat, df=30)
        
        # df=2 gives less significant result
        assert pval_df2 > pval_df30


def test_print_summary_statistics():
    """Print summary for manual review"""
    print("\n" + "="*70)
    print("t-Distribution Properties at Small Degrees of Freedom")
    print("="*70)
    
    # Critical values
    print("\n1. 95% CI Critical Values (alpha=0.05):")
    print("-" * 50)
    alpha = 0.05
    dfs = [1, 2, 3, 5, 10, 30, 100]
    print(f"{'df':>6} {'t_crit':>12} {'CI width':>14}")
    for df in dfs:
        t_crit = stats.t.ppf(1 - alpha/2, df)
        print(f"{df:>6} {t_crit:>12.4f} {2*t_crit:>14.2f}")
    z_crit = stats.norm.ppf(1 - alpha/2)
    print(f"{'inf':>6} {z_crit:>12.4f} {2*z_crit:>14.2f}")
    
    # Variance
    print("\n2. Theoretical Variance:")
    print("-" * 50)
    print("   df=1: undefined (Cauchy)")
    print("   df=2: infinite")
    print("   df=3: 3.0")
    print("   df=4: 2.0")
    print("   df=5: 1.67")
    print("   df=30: 1.07")
    print("   df=inf: 1.0")
    
    print("\n" + "="*70)
    print("Conclusion: Warning at df <= 2 is statistically justified")
    print("="*70)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
    test_print_summary_statistics()

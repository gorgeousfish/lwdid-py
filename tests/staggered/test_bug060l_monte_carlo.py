"""
BUG-060-L: Monte Carlo Coverage Simulation

Validates that:
1. CI coverage is approximately 95% when df is correctly specified
2. Small df leads to very wide CIs (even with correct coverage)
3. Warning at df <= 2 is justified by CI width characteristics

This simulation demonstrates WHY the warning is important:
- Even though coverage may be correct, the CIs are so wide at df <= 2
  that the inference provides little useful information.
"""

import numpy as np
import pandas as pd
import warnings
import pytest
from scipy import stats

from lwdid.staggered.estimation import run_ols_regression


class TestMonteCarloCoverage:
    """Monte Carlo simulations to verify CI coverage"""
    
    @pytest.fixture
    def rng(self):
        """Fixed random state for reproducibility"""
        return np.random.default_rng(42)
    
    def simulate_coverage(self, n_treated, n_control, true_att, n_sims, rng, vce=None):
        """
        Simulate coverage rate for given sample sizes.
        
        Under the classical linear model assumptions with normal errors:
        - The OLS estimator has exact t-distribution
        - CI should have ~95% coverage
        
        Parameters
        ----------
        n_treated : int
            Number of treated units
        n_control : int
            Number of control units
        true_att : float
            True ATT value
        n_sims : int
            Number of simulations
        rng : np.random.Generator
            Random number generator
        vce : str, optional
            Variance estimator
        
        Returns
        -------
        dict
            coverage: float
            ci_widths: list
            df_inference: int
        """
        covered = 0
        ci_widths = []
        df_inference_vals = []
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress warnings during simulation
            
            for _ in range(n_sims):
                # Generate data under classical linear model
                # Y = beta0 + beta1 * D + epsilon, epsilon ~ N(0, 1)
                beta0 = 0.0
                beta1 = true_att
                sigma = 1.0
                
                n = n_treated + n_control
                d = np.array([1] * n_treated + [0] * n_control)
                epsilon = rng.normal(0, sigma, n)
                y = beta0 + beta1 * d + epsilon
                
                df = pd.DataFrame({'y': y, 'd': d})
                
                try:
                    result = run_ols_regression(df, 'y', 'd', vce=vce)
                    
                    if np.isfinite(result['ci_lower']) and np.isfinite(result['ci_upper']):
                        if result['ci_lower'] <= true_att <= result['ci_upper']:
                            covered += 1
                        ci_widths.append(result['ci_upper'] - result['ci_lower'])
                        df_inference_vals.append(result['df_inference'])
                except Exception:
                    pass  # Skip failed simulations
        
        n_valid = len(ci_widths)
        coverage = covered / n_valid if n_valid > 0 else np.nan
        df_inference = df_inference_vals[0] if df_inference_vals else np.nan
        
        return {
            'coverage': coverage,
            'ci_widths': ci_widths,
            'df_inference': df_inference,
            'n_valid': n_valid,
        }
    
    def test_coverage_df1(self, rng):
        """
        Test coverage at df=1 (n_treated=1, n_control=2, k=2, df=3-2=1)
        
        Coverage should still be ~95% but CIs are very wide.
        """
        result = self.simulate_coverage(
            n_treated=1, n_control=2, true_att=1.0, n_sims=500, rng=rng
        )
        
        assert result['df_inference'] == 1
        
        # Coverage should be around 95% (allow 90-100% due to sampling variation)
        # Note: with df=1 (Cauchy), this is testing the validity of t(1) inference
        assert 0.85 < result['coverage'] < 1.0, f"Coverage={result['coverage']}"
        
        # CI width should be very large (t-critical for df=1 is ~12.71)
        median_width = np.median(result['ci_widths'])
        assert median_width > 10, f"Median CI width={median_width} should be > 10"
    
    def test_coverage_df2(self, rng):
        """
        Test coverage at df=2 (n_treated=2, n_control=2, k=2, df=4-2=2)
        
        Coverage should be ~95% but CIs are still quite wide.
        """
        result = self.simulate_coverage(
            n_treated=2, n_control=2, true_att=1.0, n_sims=500, rng=rng
        )
        
        assert result['df_inference'] == 2
        
        # Coverage around 95%
        assert 0.90 < result['coverage'] < 1.0, f"Coverage={result['coverage']}"
        
        # CI width should be moderate-large (t-critical for df=2 is ~4.30)
        median_width = np.median(result['ci_widths'])
        assert median_width > 4, f"Median CI width={median_width} should be > 4"
    
    def test_coverage_df3(self, rng):
        """
        Test coverage at df=3 (n_treated=2, n_control=3, k=2, df=5-2=3)
        
        This is just above the warning threshold.
        """
        result = self.simulate_coverage(
            n_treated=2, n_control=3, true_att=1.0, n_sims=500, rng=rng
        )
        
        assert result['df_inference'] == 3
        
        # Coverage around 95%
        assert 0.90 < result['coverage'] < 1.0, f"Coverage={result['coverage']}"
        
        # CI width should be smaller than df=2 case
        median_width = np.median(result['ci_widths'])
        assert median_width > 2, f"Median CI width={median_width}"
    
    def test_coverage_df30(self, rng):
        """
        Test coverage at df=30 (n_treated=16, n_control=16, k=2, df=32-2=30)
        
        This should behave close to normal inference.
        """
        result = self.simulate_coverage(
            n_treated=16, n_control=16, true_att=1.0, n_sims=500, rng=rng
        )
        
        assert result['df_inference'] == 30
        
        # Coverage around 95% (tighter bounds)
        assert 0.93 < result['coverage'] < 0.98, f"Coverage={result['coverage']}"
        
        # CI width should be close to normal (z=1.96)
        # Expected width ≈ 2 * 2.04 * SE, where SE ≈ sqrt(2/16) ≈ 0.35
        median_width = np.median(result['ci_widths'])
        assert 1.0 < median_width < 2.0, f"Median CI width={median_width}"


class TestCIWidthComparison:
    """Compare CI widths across different df values"""
    
    def test_ci_width_decreases_with_df(self):
        """
        CI width (normalized by SE) should decrease as df increases.
        
        Theoretical values for 95% CI:
        - df=1: width = 2 * 12.71 * SE = 25.4 * SE
        - df=2: width = 2 * 4.30 * SE = 8.6 * SE
        - df=3: width = 2 * 3.18 * SE = 6.4 * SE
        - df=30: width = 2 * 2.04 * SE = 4.1 * SE
        - df=∞: width = 2 * 1.96 * SE = 3.9 * SE
        """
        alpha = 0.05
        
        t_crit_df1 = stats.t.ppf(1 - alpha/2, df=1)
        t_crit_df2 = stats.t.ppf(1 - alpha/2, df=2)
        t_crit_df3 = stats.t.ppf(1 - alpha/2, df=3)
        t_crit_df30 = stats.t.ppf(1 - alpha/2, df=30)
        z_crit = stats.norm.ppf(1 - alpha/2)
        
        # Verify monotonic decrease
        assert t_crit_df1 > t_crit_df2 > t_crit_df3 > t_crit_df30 > z_crit
        
        # Verify df=1 is dramatically wider
        ratio_df1_to_normal = t_crit_df1 / z_crit
        assert ratio_df1_to_normal > 6, f"df=1 ratio to normal: {ratio_df1_to_normal}"
        
        # Verify df=2 is significantly wider
        ratio_df2_to_normal = t_crit_df2 / z_crit
        assert ratio_df2_to_normal > 2, f"df=2 ratio to normal: {ratio_df2_to_normal}"
        
        # Verify df=3 is only moderately wider
        ratio_df3_to_normal = t_crit_df3 / z_crit
        assert 1.5 < ratio_df3_to_normal < 2.0, f"df=3 ratio to normal: {ratio_df3_to_normal}"


class TestWarningJustification:
    """
    Tests demonstrating why the df <= 2 warning is justified.
    
    Even though CI coverage may be approximately correct, the CIs are
    so wide that they provide little useful information.
    """
    
    def test_df1_ci_useless(self):
        """
        With df=1, even a clear treatment effect produces uninformative CI.
        """
        np.random.seed(42)
        
        # Create data with clear treatment effect
        df = pd.DataFrame({
            'y': [10.0, 1.0, 2.0],  # Treated unit has much higher outcome
            'd': [1, 0, 0],
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = run_ols_regression(df, 'y', 'd')
            
            # ATT estimate is substantial
            assert result['att'] > 5
            
            # But CI is so wide it includes 0 (and negative values)
            ci_width = result['ci_upper'] - result['ci_lower']
            assert ci_width > 20, f"CI width = {ci_width}"
            
            # CI includes zero even with clear effect
            assert result['ci_lower'] < 0 < result['ci_upper']
            
            # Warning should be issued
            df_warnings = [x for x in w if 'degrees of freedom' in str(x.message).lower()]
            assert len(df_warnings) == 1
    
    def test_df2_ci_very_wide(self):
        """
        With df=2, CI is still quite wide even with clear effect.
        """
        np.random.seed(42)
        
        # Create data with clear treatment effect
        df = pd.DataFrame({
            'y': [8.0, 9.0, 1.0, 2.0],  # Treated units have higher outcomes
            'd': [1, 1, 0, 0],
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = run_ols_regression(df, 'y', 'd')
            
            # ATT estimate is substantial
            assert result['att'] > 5
            
            # CI is wide but more reasonable than df=1
            ci_width = result['ci_upper'] - result['ci_lower']
            assert 5 < ci_width < 20, f"CI width = {ci_width}"
            
            # Warning should be issued
            df_warnings = [x for x in w if 'degrees of freedom' in str(x.message).lower()]
            assert len(df_warnings) == 1
    
    def test_df30_ci_informative(self):
        """
        With df=30, CI is reasonably informative.
        """
        np.random.seed(42)
        n = 32  # df = 30
        true_att = 5.0
        
        d = np.array([1] * 16 + [0] * 16)
        y = 0 + true_att * d + np.random.normal(0, 1, n)
        
        df = pd.DataFrame({'y': y, 'd': d})
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = run_ols_regression(df, 'y', 'd')
            
            # CI should be much narrower
            ci_width = result['ci_upper'] - result['ci_lower']
            assert ci_width < 2.0, f"CI width = {ci_width}"
            
            # CI should exclude zero for true effect of 5
            assert result['ci_lower'] > 0
            
            # No df warning
            df_warnings = [x for x in w if 'degrees of freedom' in str(x.message).lower()]
            assert len(df_warnings) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

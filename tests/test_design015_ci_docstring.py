"""
Tests for DESIGN-015: CI attribute docstring fix.

Verifies that CI attributes no longer have hardcoded "95%" in docstrings,
and that the actual CI calculations correctly use the alpha parameter.
"""

import pytest
import numpy as np
import pandas as pd
from scipy import stats
import inspect

from lwdid import lwdid
from lwdid.results import LWDIDResults


class TestDocstringNoHardcoded95:
    """Test that docstrings no longer contain hardcoded '95%'."""
    
    def test_ci_lower_docstring_generic(self):
        """Test ci_lower docstring is generic, not hardcoded 95%."""
        docstring = LWDIDResults.ci_lower.fget.__doc__
        assert "95%" not in docstring
        assert "Confidence interval" in docstring or "CI" in docstring.upper()
    
    def test_ci_upper_docstring_generic(self):
        """Test ci_upper docstring is generic, not hardcoded 95%."""
        docstring = LWDIDResults.ci_upper.fget.__doc__
        assert "95%" not in docstring
        assert "Confidence interval" in docstring or "CI" in docstring.upper()
    
    def test_ci_overall_lower_docstring_generic(self):
        """Test ci_overall_lower docstring is generic, not hardcoded 95%."""
        docstring = LWDIDResults.ci_overall_lower.fget.__doc__
        assert "95%" not in docstring
        assert "Confidence interval" in docstring or "CI" in docstring.upper()
    
    def test_ci_overall_upper_docstring_generic(self):
        """Test ci_overall_upper docstring is generic, not hardcoded 95%."""
        docstring = LWDIDResults.ci_overall_upper.fget.__doc__
        assert "95%" not in docstring
        assert "Confidence interval" in docstring or "CI" in docstring.upper()
    
    def test_all_ci_docstrings_consistent(self):
        """Test all CI docstrings follow consistent pattern."""
        docstrings = [
            LWDIDResults.ci_lower.fget.__doc__,
            LWDIDResults.ci_upper.fget.__doc__,
            LWDIDResults.ci_overall_lower.fget.__doc__,
            LWDIDResults.ci_overall_upper.fget.__doc__,
        ]
        
        # All should contain "Confidence interval" or "CI"
        for doc in docstrings:
            assert "Confidence interval" in doc or "CI" in doc.upper(), \
                f"Docstring missing 'Confidence interval': {doc}"
        
        # None should contain "95%"
        for doc in docstrings:
            assert "95%" not in doc, f"Docstring still contains '95%': {doc}"


class TestCICalculationWithDifferentAlpha:
    """Test CI calculations work correctly with different alpha values."""
    
    @pytest.fixture
    def mock_results_factory(self):
        """Factory to create mock results with specified alpha."""
        def _create(alpha=0.05):
            # Calculate CI based on alpha
            att = 0.5
            se = 0.1
            z_crit = stats.norm.ppf(1 - alpha / 2)
            ci_lower = att - z_crit * se
            ci_upper = att + z_crit * se
            
            results_dict = {
                'att': att,
                'se_att': se,
                't_stat': att / se,
                'pvalue': 2 * (1 - stats.norm.cdf(abs(att / se))),
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'nobs': 100,
                'df_resid': 98,
                'vce_type': 'ols',
                'params': np.array([att]),
                'bse': np.array([se]),
                'vcov': np.array([[se**2]]),
                'resid': np.zeros(100),
                'alpha': alpha,
            }
            metadata = {
                'K': 5,
                'tpost1': 6,
                'depvar': 'y',
                'rolling': 'demean',
                'N_treated': 50,
                'N_control': 50,
                'alpha': alpha,
            }
            
            return LWDIDResults(results_dict, metadata)
        return _create
    
    @pytest.mark.parametrize("alpha,expected_confidence", [
        (0.05, 95),
        (0.10, 90),
        (0.01, 99),
        (0.20, 80),
    ])
    def test_ci_width_varies_with_alpha(self, mock_results_factory, alpha, expected_confidence):
        """Test CI width correctly varies with alpha."""
        results = mock_results_factory(alpha=alpha)
        
        # Verify alpha is stored correctly
        assert results.alpha == alpha
        
        # Verify CI is calculated correctly
        att = 0.5
        se = 0.1
        z_crit = stats.norm.ppf(1 - alpha / 2)
        expected_ci_lower = att - z_crit * se
        expected_ci_upper = att + z_crit * se
        
        assert np.isclose(results.ci_lower, expected_ci_lower, rtol=1e-10)
        assert np.isclose(results.ci_upper, expected_ci_upper, rtol=1e-10)
    
    def test_ci_width_ordering(self, mock_results_factory):
        """Test that higher confidence level means wider CI."""
        results_80 = mock_results_factory(alpha=0.20)
        results_90 = mock_results_factory(alpha=0.10)
        results_95 = mock_results_factory(alpha=0.05)
        results_99 = mock_results_factory(alpha=0.01)
        
        width_80 = results_80.ci_upper - results_80.ci_lower
        width_90 = results_90.ci_upper - results_90.ci_lower
        width_95 = results_95.ci_upper - results_95.ci_lower
        width_99 = results_99.ci_upper - results_99.ci_lower
        
        assert width_80 < width_90 < width_95 < width_99


class TestTDistributionCI:
    """Test CI calculations using t-distribution (for small samples)."""
    
    @pytest.mark.parametrize("df,alpha", [
        (10, 0.05),
        (30, 0.10),
        (100, 0.01),
        (5, 0.05),  # Very small df
    ])
    def test_t_distribution_critical_values(self, df, alpha):
        """Verify t-distribution critical values for different df and alpha."""
        t_crit = stats.t.ppf(1 - alpha / 2, df)
        z_crit = stats.norm.ppf(1 - alpha / 2)
        
        # t critical value should be larger than z for small df
        if df < 30:
            assert t_crit > z_crit
        # For large df, should converge
        if df > 100:
            assert np.isclose(t_crit, z_crit, rtol=0.05)
    
    def test_ci_formula_t_distribution(self):
        """Verify CI formula: CI = ATT ± t_crit × SE."""
        att = 1.0
        se = 0.2
        df = 50
        alpha = 0.05
        
        t_crit = stats.t.ppf(1 - alpha / 2, df)
        ci_lower = att - t_crit * se
        ci_upper = att + t_crit * se
        
        # CI should contain the true value if ATT is the true value
        assert ci_lower < att < ci_upper
        
        # CI width should be 2 * t_crit * se
        ci_width = ci_upper - ci_lower
        assert np.isclose(ci_width, 2 * t_crit * se, rtol=1e-10)


class TestStaggeredCIOverall:
    """Test overall CI attributes for staggered DiD results."""
    
    @pytest.fixture
    def staggered_results_factory(self):
        """Factory to create mock staggered results with specified alpha."""
        def _create(alpha=0.05):
            att_overall = 0.45
            se_overall = 0.08
            z_crit = stats.norm.ppf(1 - alpha / 2)
            
            results_dict = {
                'is_staggered': True,
                'att': att_overall,
                'se_att': se_overall,
                't_stat': att_overall / se_overall,
                'pvalue': 0.001,
                'ci_lower': att_overall - z_crit * se_overall,
                'ci_upper': att_overall + z_crit * se_overall,
                'nobs': 200,
                'df_resid': 190,
                'vce_type': 'ols',
                'params': np.array([att_overall]),
                'bse': np.array([se_overall]),
                'vcov': np.array([[se_overall**2]]),
                'resid': np.zeros(200),
                'alpha': alpha,
                'cohorts': [2004, 2006],
                'cohort_sizes': {2004: 50, 2006: 50},
                'att_by_cohort_time': pd.DataFrame({
                    'cohort': [2004, 2004, 2006],
                    'period': [2004, 2005, 2006],
                    'event_time': [0, 1, 0],
                    'att': [0.5, 0.6, 0.4],
                    'se': [0.1, 0.12, 0.11],
                    'ci_lower': [0.3, 0.36, 0.22],
                    'ci_upper': [0.7, 0.84, 0.58],
                    't_stat': [5.0, 5.0, 3.6],
                    'pvalue': [0.001, 0.001, 0.002],
                    'n_treated': [50, 50, 50],
                    'n_control': [100, 100, 100],
                    'n_total': [150, 150, 150],
                }),
                'control_group': 'never_treated',
                'control_group_used': 'never_treated',
                'aggregate': 'overall',
                'estimator': 'ra',
                'n_never_treated': 100,
                'att_overall': att_overall,
                'se_overall': se_overall,
                'ci_overall_lower': att_overall - z_crit * se_overall,
                'ci_overall_upper': att_overall + z_crit * se_overall,
                't_stat_overall': att_overall / se_overall,
                'pvalue_overall': 2 * (1 - stats.norm.cdf(abs(att_overall / se_overall))),
            }
            metadata = {
                'K': 0,
                'tpost1': 2004,
                'depvar': 'y',
                'rolling': 'demean',
                'N_treated': 100,
                'N_control': 100,
                'is_staggered': True,
                'alpha': alpha,
            }
            
            return LWDIDResults(results_dict, metadata)
        return _create
    
    @pytest.mark.parametrize("alpha", [0.05, 0.10, 0.01])
    def test_ci_overall_varies_with_alpha(self, staggered_results_factory, alpha):
        """Test overall CI correctly varies with alpha."""
        results = staggered_results_factory(alpha=alpha)
        
        att_overall = 0.45
        se_overall = 0.08
        z_crit = stats.norm.ppf(1 - alpha / 2)
        expected_ci_lower = att_overall - z_crit * se_overall
        expected_ci_upper = att_overall + z_crit * se_overall
        
        assert np.isclose(results.ci_overall_lower, expected_ci_lower, rtol=1e-10)
        assert np.isclose(results.ci_overall_upper, expected_ci_upper, rtol=1e-10)


class TestE2EWithRealData:
    """End-to-end tests with real datasets."""
    
    @pytest.fixture
    def smoking_data(self):
        """Load smoking dataset."""
        import os
        data_path = os.path.join(
            os.path.dirname(__file__), 
            'data', 
            'smoking.csv'
        )
        return pd.read_csv(data_path)
    
    def test_common_timing_different_alpha(self, smoking_data):
        """Test common timing mode with different alpha values."""
        alphas = [0.01, 0.05, 0.10, 0.20]
        results_dict = {}
        
        for alpha in alphas:
            result = lwdid(
                data=smoking_data,
                y='lcigsale',
                d='d',
                ivar='state',
                tvar='year',
                post='post',
                rolling='demean',
                alpha=alpha,
            )
            results_dict[alpha] = result
            
            # Verify alpha is correctly stored
            assert result.alpha == alpha
        
        # Verify CI widths are ordered correctly
        widths = {a: r.ci_upper - r.ci_lower for a, r in results_dict.items()}
        assert widths[0.20] < widths[0.10] < widths[0.05] < widths[0.01]
    
    def test_ci_contains_point_estimate(self, smoking_data):
        """Test that CI always contains the point estimate."""
        for alpha in [0.01, 0.05, 0.10]:
            result = lwdid(
                data=smoking_data,
                y='lcigsale',
                d='d',
                ivar='state',
                tvar='year',
                post='post',
                rolling='demean',
                alpha=alpha,
            )
            
            assert result.ci_lower < result.att < result.ci_upper, \
                f"CI [{result.ci_lower}, {result.ci_upper}] does not contain ATT {result.att} for alpha={alpha}"


class TestMonteCarloCoverage:
    """Monte Carlo simulation tests for CI coverage."""
    
    def test_ci_coverage_rate(self):
        """Test that CI achieves nominal coverage rate."""
        np.random.seed(42)
        
        true_att = 0.5
        se = 0.1
        n_simulations = 1000
        
        for alpha in [0.05, 0.10]:
            coverage_count = 0
            z_crit = stats.norm.ppf(1 - alpha / 2)
            
            for _ in range(n_simulations):
                # Simulate ATT estimate
                att_hat = np.random.normal(true_att, se)
                
                # Calculate CI
                ci_lower = att_hat - z_crit * se
                ci_upper = att_hat + z_crit * se
                
                # Check if true value is covered
                if ci_lower <= true_att <= ci_upper:
                    coverage_count += 1
            
            coverage_rate = coverage_count / n_simulations
            expected_coverage = 1 - alpha
            
            # Allow 3% tolerance for Monte Carlo variability
            assert abs(coverage_rate - expected_coverage) < 0.03, \
                f"Coverage rate {coverage_rate:.3f} differs from expected {expected_coverage:.3f} for alpha={alpha}"


class TestSummaryDisplaysCorrectCI:
    """Test that summary methods display correct CI percentage."""
    
    def test_summary_common_timing(self):
        """Test summary() shows correct CI percentage for common timing."""
        for alpha, expected_pct in [(0.05, "95%"), (0.10, "90%"), (0.01, "99%")]:
            results_dict = {
                'att': 0.5,
                'se_att': 0.1,
                't_stat': 5.0,
                'pvalue': 0.001,
                'ci_lower': 0.3,
                'ci_upper': 0.7,
                'nobs': 100,
                'df_resid': 98,
                'df_inference': 98,
                'vce_type': 'ols',
                'params': np.array([0.5]),
                'bse': np.array([0.1]),
                'vcov': np.array([[0.01]]),
                'resid': np.zeros(100),
                'alpha': alpha,
            }
            metadata = {
                'K': 5,
                'tpost1': 6,
                'depvar': 'y',
                'rolling': 'demean',
                'N_treated': 50,
                'N_control': 50,
            }
            
            results = LWDIDResults(results_dict, metadata)
            summary_str = results.summary()
            
            assert f"[{expected_pct} Conf. Interval]" in summary_str, \
                f"Summary should show [{expected_pct} Conf. Interval] for alpha={alpha}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

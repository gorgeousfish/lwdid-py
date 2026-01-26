"""
Tests for alpha parameter and confidence interval calculations.

BUG-002 Fix: Verifies that confidence intervals are correctly calculated
using dynamic z-values based on the alpha parameter, not hardcoded 1.96.
"""

import pytest
import numpy as np
import pandas as pd
from scipy import stats

from lwdid import lwdid
from lwdid.results import LWDIDResults


class TestAlphaParameter:
    """Test alpha parameter in LWDIDResults class."""
    
    def test_alpha_property_default(self):
        """Test that default alpha is 0.05 (95% CI)."""
        # Create minimal mock data
        results_dict = {
            'att': 0.5,
            'se_att': 0.1,
            't_stat': 5.0,
            'pvalue': 0.001,
            'ci_lower': 0.3,
            'ci_upper': 0.7,
            'nobs': 100,
            'df_resid': 98,
            'vce_type': 'ols',
            'params': np.array([0.5]),
            'bse': np.array([0.1]),
            'vcov': np.array([[0.01]]),
            'resid': np.zeros(100),
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
        assert results.alpha == 0.05
    
    def test_alpha_property_custom(self):
        """Test that custom alpha is correctly stored."""
        results_dict = {
            'att': 0.5,
            'se_att': 0.1,
            't_stat': 5.0,
            'pvalue': 0.001,
            'ci_lower': 0.3,
            'ci_upper': 0.7,
            'nobs': 100,
            'df_resid': 98,
            'vce_type': 'ols',
            'params': np.array([0.5]),
            'bse': np.array([0.1]),
            'vcov': np.array([[0.01]]),
            'resid': np.zeros(100),
            'alpha': 0.10,  # 90% CI
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
        assert results.alpha == 0.10
    
    def test_alpha_from_metadata(self):
        """Test that alpha can be read from metadata."""
        results_dict = {
            'att': 0.5,
            'se_att': 0.1,
            't_stat': 5.0,
            'pvalue': 0.001,
            'ci_lower': 0.3,
            'ci_upper': 0.7,
            'nobs': 100,
            'df_resid': 98,
            'vce_type': 'ols',
            'params': np.array([0.5]),
            'bse': np.array([0.1]),
            'vcov': np.array([[0.01]]),
            'resid': np.zeros(100),
        }
        metadata = {
            'K': 5,
            'tpost1': 6,
            'depvar': 'y',
            'rolling': 'demean',
            'N_treated': 50,
            'N_control': 50,
            'alpha': 0.01,  # 99% CI
        }
        
        results = LWDIDResults(results_dict, metadata)
        assert results.alpha == 0.01


class TestZCriticalValues:
    """Test z critical value calculations for different alpha levels."""
    
    @pytest.mark.parametrize("alpha,expected_z", [
        (0.05, 1.959963984540054),  # 95% CI
        (0.10, 1.6448536269514722),  # 90% CI
        (0.01, 2.5758293035489004),  # 99% CI
        (0.20, 1.2815515655446004),  # 80% CI
    ])
    def test_z_critical_values(self, alpha, expected_z):
        """Test that z critical values are correctly calculated."""
        z_crit = stats.norm.ppf(1 - alpha / 2)
        assert np.isclose(z_crit, expected_z, rtol=1e-10)
    
    def test_ci_width_increases_with_confidence(self):
        """Test that CI width increases as confidence level increases."""
        att = 1.0
        se = 0.1
        
        # 90% CI (alpha=0.10)
        z_90 = stats.norm.ppf(1 - 0.10 / 2)
        ci_width_90 = 2 * z_90 * se
        
        # 95% CI (alpha=0.05)
        z_95 = stats.norm.ppf(1 - 0.05 / 2)
        ci_width_95 = 2 * z_95 * se
        
        # 99% CI (alpha=0.01)
        z_99 = stats.norm.ppf(1 - 0.01 / 2)
        ci_width_99 = 2 * z_99 * se
        
        assert ci_width_90 < ci_width_95 < ci_width_99


class TestSummaryDynamicCI:
    """Test that summary() shows correct CI percentage based on alpha."""
    
    def test_summary_shows_correct_ci_percentage_default(self):
        """Test that summary shows 95% CI by default."""
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
        
        assert "[95% Conf. Interval]" in summary_str
    
    def test_summary_shows_correct_ci_percentage_90(self):
        """Test that summary shows 90% CI when alpha=0.10."""
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
            'alpha': 0.10,
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
        
        assert "[90% Conf. Interval]" in summary_str
    
    def test_summary_shows_correct_ci_percentage_99(self):
        """Test that summary shows 99% CI when alpha=0.01."""
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
            'alpha': 0.01,
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
        
        assert "[99% Conf. Interval]" in summary_str


class TestCICalculationFormula:
    """Test that CI calculations use correct formula: ATT ± z_crit × SE."""
    
    def test_ci_formula_verification(self):
        """Verify CI formula: CI = ATT ± z_crit × SE."""
        att = 1.0
        se = 0.2
        alpha = 0.05
        
        z_crit = stats.norm.ppf(1 - alpha / 2)
        expected_ci_lower = att - z_crit * se
        expected_ci_upper = att + z_crit * se
        
        # Verify values
        assert np.isclose(expected_ci_lower, 1.0 - 1.96 * 0.2, rtol=0.01)
        assert np.isclose(expected_ci_upper, 1.0 + 1.96 * 0.2, rtol=0.01)
    
    def test_ci_formula_alpha_10(self):
        """Verify CI formula with alpha=0.10."""
        att = 1.0
        se = 0.2
        alpha = 0.10
        
        z_crit = stats.norm.ppf(1 - alpha / 2)  # ≈ 1.645
        expected_ci_lower = att - z_crit * se
        expected_ci_upper = att + z_crit * se
        
        # Verify values
        assert np.isclose(expected_ci_lower, 1.0 - 1.6449 * 0.2, rtol=0.01)
        assert np.isclose(expected_ci_upper, 1.0 + 1.6449 * 0.2, rtol=0.01)


class TestNoHardcoded196:
    """Regression tests to ensure no hardcoded 1.96 remains."""
    
    def test_summary_staggered_uses_dynamic_z(self):
        """Test that summary_staggered uses dynamic z value."""
        # Create staggered results with alpha=0.10
        results_dict = {
            'is_staggered': True,
            'att': 0.5,
            'se_att': 0.1,
            't_stat': 5.0,
            'pvalue': 0.001,
            'ci_lower': 0.3,
            'ci_upper': 0.7,
            'nobs': 100,
            'df_resid': 98,
            'vce_type': 'ols',
            'params': np.array([0.5]),
            'bse': np.array([0.1]),
            'vcov': np.array([[0.01]]),
            'resid': np.zeros(100),
            'alpha': 0.10,  # 90% CI
            'cohorts': [2005, 2006],
            'cohort_sizes': {2005: 10, 2006: 15},
            'att_by_cohort_time': pd.DataFrame({
                'cohort': [2005, 2005, 2006],
                'period': [2005, 2006, 2006],
                'event_time': [0, 1, 0],
                'att': [0.5, 0.6, 0.4],
                'se': [0.1, 0.12, 0.11],
                'ci_lower': [0.3, 0.36, 0.22],
                'ci_upper': [0.7, 0.84, 0.58],
                't_stat': [5.0, 5.0, 3.6],
                'pvalue': [0.001, 0.001, 0.002],
                'n_treated': [10, 10, 15],
                'n_control': [20, 20, 20],
                'n_total': [30, 30, 35],
            }),
            'control_group': 'never_treated',
            'control_group_used': 'never_treated',
            'aggregate': 'cohort',
            'estimator': 'ra',
            'n_never_treated': 20,
            'att_by_cohort': pd.DataFrame({
                'cohort': [2005, 2006],
                'att': [0.55, 0.4],
                'se': [0.08, 0.11],
                'n_units': [10, 15],
                'n_periods': [2, 1],
            }),
        }
        metadata = {
            'K': 0,
            'tpost1': 2005,
            'depvar': 'y',
            'rolling': 'demean',
            'N_treated': 25,
            'N_control': 20,
            'is_staggered': True,
            'alpha': 0.10,
        }
        
        results = LWDIDResults(results_dict, metadata)
        summary_str = results.summary_staggered()
        
        # Should show 90% CI, not 95%
        assert "[90% CI]" in summary_str
        assert "[95% CI]" not in summary_str


class TestAlphaE2E:
    """End-to-end tests for alpha parameter with real data."""
    
    @pytest.fixture
    def smoking_data(self):
        """Load smoking dataset for testing."""
        import os
        data_path = os.path.join(
            os.path.dirname(__file__), 
            'data', 
            'smoking.csv'
        )
        return pd.read_csv(data_path)
    
    @pytest.fixture
    def castle_data(self):
        """Create a simple Castle Law-like dataset for staggered testing."""
        np.random.seed(42)
        
        # Create panel data
        units = list(range(1, 21))  # 20 units
        years = list(range(2000, 2010))  # 10 years
        
        data = []
        for unit in units:
            # Assign treatment cohort: some never treated (0), others treated
            if unit <= 5:
                gvar = 0  # Never treated
            elif unit <= 10:
                gvar = 2004  # Treated in 2004
            else:
                gvar = 2006  # Treated in 2006
            
            for year in years:
                # Generate outcome
                treated = (gvar > 0) and (year >= gvar)
                y = 1.0 + 0.1 * year + (0.5 if treated else 0) + np.random.normal(0, 0.3)
                
                data.append({
                    'unit': unit,
                    'year': year,
                    'y': y,
                    'gvar': gvar,
                })
        
        return pd.DataFrame(data)
    
    def test_common_timing_alpha_default(self, smoking_data):
        """Test common timing with default alpha=0.05."""
        results = lwdid(
            data=smoking_data,
            y='lcigsale',
            d='d',  # Column name in smoking.csv
            ivar='state',
            tvar='year',
            post='post',
            rolling='demean',
        )
        
        assert results.alpha == 0.05
        assert "[95% Conf. Interval]" in results.summary()
    
    def test_common_timing_alpha_custom(self, smoking_data):
        """Test common timing with custom alpha=0.10."""
        results = lwdid(
            data=smoking_data,
            y='lcigsale',
            d='d',  # Column name in smoking.csv
            ivar='state',
            tvar='year',
            post='post',
            rolling='demean',
            alpha=0.10,
        )
        
        assert results.alpha == 0.10
        assert "[90% Conf. Interval]" in results.summary()
    
    def test_staggered_alpha_default(self, castle_data):
        """Test staggered DiD with default alpha=0.05."""
        results = lwdid(
            data=castle_data,
            y='y',
            ivar='unit',
            tvar='year',
            gvar='gvar',
            control_group='never_treated',
            aggregate='overall',
        )
        
        assert results.alpha == 0.05
        summary = results.summary()
        assert "95% CI:" in summary
    
    def test_staggered_alpha_custom(self, castle_data):
        """Test staggered DiD with custom alpha=0.10."""
        results = lwdid(
            data=castle_data,
            y='y',
            ivar='unit',
            tvar='year',
            gvar='gvar',
            control_group='never_treated',
            aggregate='overall',
            alpha=0.10,
        )
        
        assert results.alpha == 0.10
        summary = results.summary()
        assert "90% CI:" in summary
    
    def test_ci_width_comparison(self, smoking_data):
        """Test that different alpha values produce different CI widths."""
        results_95 = lwdid(
            data=smoking_data,
            y='lcigsale',
            d='d',  # Column name in smoking.csv
            ivar='state',
            tvar='year',
            post='post',
            rolling='demean',
            alpha=0.05,
        )
        
        results_90 = lwdid(
            data=smoking_data,
            y='lcigsale',
            d='d',  # Column name in smoking.csv
            ivar='state',
            tvar='year',
            post='post',
            rolling='demean',
            alpha=0.10,
        )
        
        # 95% CI should be wider than 90% CI
        ci_width_95 = results_95.ci_upper - results_95.ci_lower
        ci_width_90 = results_90.ci_upper - results_90.ci_lower
        
        assert ci_width_95 > ci_width_90


class TestPlotEventStudyAlpha:
    """Test plot_event_study with different alpha values."""
    
    @pytest.fixture
    def staggered_results(self):
        """Create mock staggered results for plot testing."""
        results_dict = {
            'is_staggered': True,
            'att': 0.5,
            'se_att': 0.1,
            't_stat': 5.0,
            'pvalue': 0.001,
            'ci_lower': 0.3,
            'ci_upper': 0.7,
            'nobs': 100,
            'df_resid': 98,
            'vce_type': 'ols',
            'params': np.array([0.5]),
            'bse': np.array([0.1]),
            'vcov': np.array([[0.01]]),
            'resid': np.zeros(100),
            'alpha': 0.10,  # 90% CI
            'cohorts': [2005, 2006],
            'cohort_sizes': {2005: 10, 2006: 15},
            'att_by_cohort_time': pd.DataFrame({
                'cohort': [2005, 2005, 2006, 2006],
                'period': [2005, 2006, 2006, 2007],
                'event_time': [0, 1, 0, 1],
                'att': [0.5, 0.6, 0.4, 0.55],
                'se': [0.1, 0.12, 0.11, 0.09],
                'ci_lower': [0.3, 0.36, 0.22, 0.40],
                'ci_upper': [0.7, 0.84, 0.58, 0.70],
                't_stat': [5.0, 5.0, 3.6, 6.1],
                'pvalue': [0.001, 0.001, 0.002, 0.0005],
                'n_treated': [10, 10, 15, 15],
                'n_control': [20, 20, 20, 20],
                'n_total': [30, 30, 35, 35],
            }),
            'control_group': 'never_treated',
            'control_group_used': 'never_treated',
            'aggregate': 'cohort',
            'estimator': 'ra',
            'n_never_treated': 20,
            'cohort_weights': {2005: 0.4, 2006: 0.6},
        }
        metadata = {
            'K': 0,
            'tpost1': 2005,
            'depvar': 'y',
            'rolling': 'demean',
            'N_treated': 25,
            'N_control': 20,
            'is_staggered': True,
            'alpha': 0.10,
        }
        
        return LWDIDResults(results_dict, metadata)
    
    def test_plot_event_study_returns_correct_ci(self, staggered_results):
        """Test that plot_event_study returns correct CI values based on alpha."""
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        fig, ax, event_df = staggered_results.plot_event_study(
            return_data=True,
            show_ci=True
        )
        
        # Check that CI is calculated using correct z value
        z_crit_90 = stats.norm.ppf(0.95)  # alpha=0.10
        
        for _, row in event_df.iterrows():
            expected_ci_lower = row['att'] - z_crit_90 * row['se']
            expected_ci_upper = row['att'] + z_crit_90 * row['se']
            
            assert np.isclose(row['ci_lower'], expected_ci_lower, rtol=0.01)
            assert np.isclose(row['ci_upper'], expected_ci_upper, rtol=0.01)
        
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestAlphaStataConsistency:
    """Test Python-Stata consistency for alpha parameter and CI calculations."""
    
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
    
    @pytest.fixture
    def mve_demean_data(self):
        """Load mve_demean dataset."""
        import os
        data_path = os.path.join(
            os.path.dirname(__file__), 
            'data', 
            'mve_demean.csv'
        )
        return pd.read_csv(data_path)
    
    @pytest.fixture
    def smoking_controls_small_data(self):
        """Load smoking_controls_small dataset."""
        import os
        data_path = os.path.join(
            os.path.dirname(__file__), 
            'data', 
            'smoking_controls_small.csv'
        )
        return pd.read_csv(data_path)
    
    # Stata baseline values (from lwdid with set level 95/90)
    # Dataset 1: smoking.csv
    STATA_ATT = -0.4221746150201265
    STATA_SE = 0.1207995238667734
    STATA_CI_L_95 = -0.6669377
    STATA_CI_U_95 = -0.1774115
    STATA_CI_L_90 = -0.6259747
    STATA_CI_U_90 = -0.2183745
    
    # Dataset 2: mve_demean.csv
    STATA_ATT_MVE = 3.5
    STATA_SE_MVE = 2.5980762
    STATA_CI_L_95_MVE = -29.51169
    STATA_CI_U_95_MVE = 36.51169
    STATA_CI_L_90_MVE = -12.90361
    STATA_CI_U_90_MVE = 19.90361
    
    # Dataset 3: smoking_controls_small.csv
    STATA_ATT_CTRL = -0.2535597
    STATA_SE_CTRL = 0.17414711
    STATA_CI_L_95_CTRL = -0.7370696
    STATA_CI_U_95_CTRL = 0.2299502
    STATA_CI_L_90_CTRL = -0.6248147
    STATA_CI_U_90_CTRL = 0.1176953
    
    def test_att_matches_stata(self, smoking_data):
        """Test that ATT matches Stata baseline."""
        result = lwdid(
            data=smoking_data,
            y='lcigsale',
            d='d',
            ivar='state',
            tvar='year',
            post='post',
            rolling='demean',
        )
        
        # ATT should match within 1e-6 relative tolerance
        assert np.isclose(result.att, self.STATA_ATT, rtol=1e-6)
    
    def test_se_matches_stata(self, smoking_data):
        """Test that SE matches Stata baseline."""
        result = lwdid(
            data=smoking_data,
            y='lcigsale',
            d='d',
            ivar='state',
            tvar='year',
            post='post',
            rolling='demean',
        )
        
        # SE should match within 1e-6 relative tolerance
        assert np.isclose(result.se_att, self.STATA_SE, rtol=1e-6)
    
    def test_95_ci_matches_stata(self, smoking_data):
        """Test that 95% CI matches Stata baseline."""
        result = lwdid(
            data=smoking_data,
            y='lcigsale',
            d='d',
            ivar='state',
            tvar='year',
            post='post',
            rolling='demean',
            alpha=0.05,
        )
        
        # CI should match within 1e-5 absolute tolerance
        assert np.isclose(result.ci_lower, self.STATA_CI_L_95, atol=1e-5)
        assert np.isclose(result.ci_upper, self.STATA_CI_U_95, atol=1e-5)
    
    def test_90_ci_matches_stata(self, smoking_data):
        """Test that 90% CI matches Stata baseline (set level 90)."""
        result = lwdid(
            data=smoking_data,
            y='lcigsale',
            d='d',
            ivar='state',
            tvar='year',
            post='post',
            rolling='demean',
            alpha=0.10,
        )
        
        # CI should match within 1e-5 absolute tolerance
        assert np.isclose(result.ci_lower, self.STATA_CI_L_90, atol=1e-5)
        assert np.isclose(result.ci_upper, self.STATA_CI_U_90, atol=1e-5)
    
    def test_ci_width_matches_stata(self, smoking_data):
        """Test that CI widths match Stata for different alpha values."""
        result_95 = lwdid(
            data=smoking_data,
            y='lcigsale',
            d='d',
            ivar='state',
            tvar='year',
            post='post',
            rolling='demean',
            alpha=0.05,
        )
        
        result_90 = lwdid(
            data=smoking_data,
            y='lcigsale',
            d='d',
            ivar='state',
            tvar='year',
            post='post',
            rolling='demean',
            alpha=0.10,
        )
        
        stata_width_95 = self.STATA_CI_U_95 - self.STATA_CI_L_95
        stata_width_90 = self.STATA_CI_U_90 - self.STATA_CI_L_90
        
        python_width_95 = result_95.ci_upper - result_95.ci_lower
        python_width_90 = result_90.ci_upper - result_90.ci_lower
        
        # Widths should match within 1e-5 absolute tolerance
        assert np.isclose(python_width_95, stata_width_95, atol=1e-5)
        assert np.isclose(python_width_90, stata_width_90, atol=1e-5)
        
        # 90% CI should be narrower than 95% CI
        assert python_width_90 < python_width_95
    
    # === Cross-validation with mve_demean dataset ===
    def test_mve_demean_att_matches_stata(self, mve_demean_data):
        """Test mve_demean ATT matches Stata baseline."""
        result = lwdid(
            data=mve_demean_data,
            y='y', d='d', ivar='id', tvar='year', post='post',
            rolling='demean',
        )
        assert np.isclose(result.att, self.STATA_ATT_MVE, rtol=1e-6)
    
    def test_mve_demean_se_matches_stata(self, mve_demean_data):
        """Test mve_demean SE matches Stata baseline."""
        result = lwdid(
            data=mve_demean_data,
            y='y', d='d', ivar='id', tvar='year', post='post',
            rolling='demean',
        )
        assert np.isclose(result.se_att, self.STATA_SE_MVE, rtol=1e-6)
    
    def test_mve_demean_95_ci_matches_stata(self, mve_demean_data):
        """Test mve_demean 95% CI matches Stata baseline."""
        result = lwdid(
            data=mve_demean_data,
            y='y', d='d', ivar='id', tvar='year', post='post',
            rolling='demean', alpha=0.05,
        )
        assert np.isclose(result.ci_lower, self.STATA_CI_L_95_MVE, atol=1e-4)
        assert np.isclose(result.ci_upper, self.STATA_CI_U_95_MVE, atol=1e-4)
    
    def test_mve_demean_90_ci_matches_stata(self, mve_demean_data):
        """Test mve_demean 90% CI matches Stata baseline."""
        result = lwdid(
            data=mve_demean_data,
            y='y', d='d', ivar='id', tvar='year', post='post',
            rolling='demean', alpha=0.10,
        )
        assert np.isclose(result.ci_lower, self.STATA_CI_L_90_MVE, atol=1e-4)
        assert np.isclose(result.ci_upper, self.STATA_CI_U_90_MVE, atol=1e-4)
    
    # === Cross-validation with smoking_controls_small dataset ===
    def test_smoking_controls_att_matches_stata(self, smoking_controls_small_data):
        """Test smoking_controls_small ATT matches Stata baseline."""
        result = lwdid(
            data=smoking_controls_small_data,
            y='lcigsale', d='d', ivar='state', tvar='year', post='post',
            rolling='demean',
        )
        assert np.isclose(result.att, self.STATA_ATT_CTRL, rtol=1e-6)
    
    def test_smoking_controls_se_matches_stata(self, smoking_controls_small_data):
        """Test smoking_controls_small SE matches Stata baseline."""
        result = lwdid(
            data=smoking_controls_small_data,
            y='lcigsale', d='d', ivar='state', tvar='year', post='post',
            rolling='demean',
        )
        assert np.isclose(result.se_att, self.STATA_SE_CTRL, rtol=1e-6)
    
    def test_smoking_controls_95_ci_matches_stata(self, smoking_controls_small_data):
        """Test smoking_controls_small 95% CI matches Stata baseline."""
        result = lwdid(
            data=smoking_controls_small_data,
            y='lcigsale', d='d', ivar='state', tvar='year', post='post',
            rolling='demean', alpha=0.05,
        )
        assert np.isclose(result.ci_lower, self.STATA_CI_L_95_CTRL, atol=1e-5)
        assert np.isclose(result.ci_upper, self.STATA_CI_U_95_CTRL, atol=1e-5)
    
    def test_smoking_controls_90_ci_matches_stata(self, smoking_controls_small_data):
        """Test smoking_controls_small 90% CI matches Stata baseline."""
        result = lwdid(
            data=smoking_controls_small_data,
            y='lcigsale', d='d', ivar='state', tvar='year', post='post',
            rolling='demean', alpha=0.10,
        )
        assert np.isclose(result.ci_lower, self.STATA_CI_L_90_CTRL, atol=1e-5)
        assert np.isclose(result.ci_upper, self.STATA_CI_U_90_CTRL, atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

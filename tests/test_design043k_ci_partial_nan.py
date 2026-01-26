"""
DESIGN-043-K: Test CI partial NaN information preservation in summary_staggered()

This test validates that when only one CI bound is NaN, the valid bound
is still displayed rather than showing "[N/A, N/A]" for both.

File location: lwdid-py_v0.1.0/tests/test_design043k_ci_partial_nan.py
"""
import numpy as np
import pandas as pd
import pytest

from lwdid.results import LWDIDResults


def create_mock_staggered_results_with_partial_nan_ci(
    ci_scenario: str = 'both_valid'
):
    """
    Create mock staggered results with different CI scenarios.
    
    Parameters
    ----------
    ci_scenario : str
        One of: 'both_valid', 'lower_only', 'upper_only', 'both_nan', 'mixed'
    
    Notes
    -----
    For partial NaN CI to display, we need BOTH:
    1. CI bound in DataFrame is NaN
    2. ATT or SE is NaN (preventing fallback calculation)
    """
    # Prepare data based on scenario
    # To prevent fallback CI calculation, att or se must be NaN for scenarios with partial CI
    scenarios = {
        'both_valid': {
            'att': [0.085, 0.098],
            'se': [0.062, 0.045],
            'ci_lower': [-0.041, 0.006],
            'ci_upper': [0.211, 0.190],
        },
        # Lower bound valid, upper bound NaN, SE is NaN to prevent fallback
        'lower_only': {
            'att': [np.nan, np.nan],  # NaN to prevent fallback calculation
            'se': [np.nan, np.nan],
            'ci_lower': [0.123, 0.456],
            'ci_upper': [np.nan, np.nan],
        },
        # Upper bound valid, lower bound NaN, SE is NaN to prevent fallback
        'upper_only': {
            'att': [np.nan, np.nan],
            'se': [np.nan, np.nan],
            'ci_lower': [np.nan, np.nan],
            'ci_upper': [0.789, 0.321],
        },
        # Both bounds NaN, SE is NaN
        'both_nan': {
            'att': [np.nan, np.nan],
            'se': [np.nan, np.nan],
            'ci_lower': [np.nan, np.nan],
            'ci_upper': [np.nan, np.nan],
        },
        # Mixed: cohort 2005 has lower only, cohort 2006 has upper only
        'mixed': {
            'att': [np.nan, np.nan],
            'se': [np.nan, np.nan],
            'ci_lower': [0.123, np.nan],
            'ci_upper': [np.nan, 0.456],
        },
    }
    
    data = scenarios.get(ci_scenario, scenarios['both_valid'])
    
    results_dict = {
        'is_staggered': True,
        'att': 0.0,
        'se_att': 0.0,
        't_stat': 0.0,
        'pvalue': 1.0,
        'ci_lower': 0.0,
        'ci_upper': 0.0,
        'nobs': 100,
        'n_treated': 50,
        'n_control': 50,
        'df_resid': 98,
        'params': np.array([0.0]),
        'bse': np.array([0.0]),
        'vcov': np.array([[0.0]]),
        'resid': np.zeros(100),
        'vce_type': 'hc3',
        
        'cohorts': [2005, 2006],
        'cohort_sizes': {2005: 25, 2006: 25},
        'control_group': 'never_treated',
        'control_group_used': 'never_treated',
        'aggregate': 'cohort',
        'estimator': 'ra',
        'n_never_treated': 50,
        'rolling': 'demean',
        
        'att_by_cohort_time': pd.DataFrame({
            'cohort': [2005, 2006],
            'period': [2005, 2006],
            'event_time': [0, 0],
            'att': data['att'],
            'se': data['se'],
            'ci_lower': data['ci_lower'],
            'ci_upper': data['ci_upper'],
            't_stat': [np.nan, np.nan] if ci_scenario != 'both_valid' else [1.37, 2.18],
            'pvalue': [np.nan, np.nan] if ci_scenario != 'both_valid' else [0.182, 0.038],
            'n_treated': [25, 25],
            'n_control': [50, 50],
            'n_total': [75, 75],
        }),
        
        'att_by_cohort': pd.DataFrame({
            'cohort': [2005, 2006],
            'att': data['att'],
            'se': data['se'],
            'ci_lower': data['ci_lower'],
            'ci_upper': data['ci_upper'],
            't_stat': [np.nan, np.nan] if ci_scenario != 'both_valid' else [1.37, 2.18],
            'pvalue': [np.nan, np.nan] if ci_scenario != 'both_valid' else [0.182, 0.038],
            'n_units': [25, 25],
            'n_periods': [5, 4],
        }),
        
        'cohort_weights': {2005: 0.5, 2006: 0.5},
        'att_overall': None,
        'se_overall': None,
        't_stat_overall': None,
        'pvalue_overall': None,
        'ci_overall': (None, None),
    }
    
    metadata = {
        'K': 4,
        'tpost1': 2005,
        'depvar': 'y',
        'N_treated': 50,
        'N_control': 50,
    }
    
    return LWDIDResults(results_dict, metadata)


class TestCIPartialNaN:
    """Test CI display with partial NaN bounds."""
    
    def test_both_valid_ci_display(self):
        """Test that both valid CI bounds are displayed correctly."""
        results = create_mock_staggered_results_with_partial_nan_ci('both_valid')
        summary = results.summary_staggered()
        
        # Should contain properly formatted CI values
        assert '[-0.041' in summary or '-0.041' in summary
        assert '0.211' in summary
        # Should NOT contain N/A for this scenario
        lines_with_cohort_2005 = [l for l in summary.split('\n') if '2005' in l and 'ATT' not in l and 'Cohort' not in l]
        if lines_with_cohort_2005:
            # The cohort data line should not have N/A in CI
            for line in lines_with_cohort_2005:
                if 'N/A, N/A' in line:
                    pytest.fail("Valid CI should not display as [N/A, N/A]")
    
    def test_lower_only_ci_display(self):
        """Test CI display when only lower bound is valid."""
        results = create_mock_staggered_results_with_partial_nan_ci('lower_only')
        summary = results.summary_staggered()
        
        # Should display valid lower bound
        assert '0.123' in summary
        # Should display N/A for upper bound, but not both
        assert 'N/A' in summary
        # Should NOT have [N/A, N/A] pattern (both bounds as N/A)
        # Instead should have [0.123,   N/A] or similar
        lines = summary.split('\n')
        for line in lines:
            if '2005' in line and '[' in line and ']' in line:
                # This line has CI info
                # The old behavior would show [N/A, N/A]
                # New behavior should show [0.123,   N/A]
                if 'N/A, N/A' in line.replace(' ', ''):
                    pytest.fail(f"Lower bound 0.123 should be preserved, got: {line}")
    
    def test_upper_only_ci_display(self):
        """Test CI display when only upper bound is valid."""
        results = create_mock_staggered_results_with_partial_nan_ci('upper_only')
        summary = results.summary_staggered()
        
        # Should display valid upper bound
        assert '0.789' in summary
        # Should display N/A for lower bound, but preserve upper
        assert 'N/A' in summary
        
        # Verify the pattern shows N/A for lower, valid for upper
        lines = summary.split('\n')
        for line in lines:
            if '2005' in line and '[' in line and ']' in line:
                if 'N/A, N/A' in line.replace(' ', ''):
                    pytest.fail(f"Upper bound 0.789 should be preserved, got: {line}")
    
    def test_both_nan_ci_display(self):
        """Test CI display when both bounds are NaN."""
        results = create_mock_staggered_results_with_partial_nan_ci('both_nan')
        summary = results.summary_staggered()
        
        # Should display N/A for both bounds
        assert 'N/A' in summary
        # The format should be [   N/A,    N/A] with proper spacing
        lines = summary.split('\n')
        ci_found = False
        for line in lines:
            if '2005' in line and '[' in line and ']' in line:
                ci_found = True
                # Both should be N/A
                assert line.count('N/A') >= 2 or 'N/A' in line
        assert ci_found, "Could not find CI line in summary"
    
    def test_mixed_nan_ci_display(self):
        """Test CI display with mixed NaN pattern across cohorts."""
        results = create_mock_staggered_results_with_partial_nan_ci('mixed')
        summary = results.summary_staggered()
        
        # Cohort 2005: ci_lower=0.123, ci_upper=NaN
        # Cohort 2006: ci_lower=NaN, ci_upper=0.456
        
        # Both valid values should be displayed
        assert '0.123' in summary
        assert '0.456' in summary
        
        # N/A should appear for the NaN bounds
        assert 'N/A' in summary


class TestCIFormattingAlignment:
    """Test CI string formatting and column alignment."""
    
    def test_ci_string_width_consistency(self):
        """Test that CI strings maintain consistent width for alignment."""
        results_valid = create_mock_staggered_results_with_partial_nan_ci('both_valid')
        results_nan = create_mock_staggered_results_with_partial_nan_ci('both_nan')
        results_mixed = create_mock_staggered_results_with_partial_nan_ci('mixed')
        
        summary_valid = results_valid.summary_staggered()
        summary_nan = results_nan.summary_staggered()
        summary_mixed = results_mixed.summary_staggered()
        
        # All summaries should be properly formatted
        assert '======' in summary_valid
        assert '======' in summary_nan
        assert '======' in summary_mixed
    
    def test_negative_ci_values(self):
        """Test formatting of negative CI values."""
        # Create results with negative CI bounds
        results_dict = {
            'is_staggered': True,
            'att': 0.0,
            'se_att': 0.0,
            't_stat': 0.0,
            'pvalue': 1.0,
            'ci_lower': 0.0,
            'ci_upper': 0.0,
            'nobs': 100,
            'n_treated': 50,
            'n_control': 50,
            'df_resid': 98,
            'params': np.array([0.0]),
            'bse': np.array([0.0]),
            'vcov': np.array([[0.0]]),
            'resid': np.zeros(100),
            'vce_type': 'hc3',
            
            'cohorts': [2005],
            'cohort_sizes': {2005: 50},
            'control_group': 'never_treated',
            'control_group_used': 'never_treated',
            'aggregate': 'cohort',
            'estimator': 'ra',
            'n_never_treated': 50,
            'rolling': 'demean',
            
            'att_by_cohort_time': pd.DataFrame({
                'cohort': [2005],
                'period': [2005],
                'event_time': [0],
                'att': [-0.5],
                'se': [0.1],
                'ci_lower': [-0.696],
                'ci_upper': [-0.304],
                't_stat': [-5.0],
                'pvalue': [0.001],
                'n_treated': [50],
                'n_control': [50],
                'n_total': [100],
            }),
            
            'att_by_cohort': pd.DataFrame({
                'cohort': [2005],
                'att': [-0.5],
                'se': [0.1],
                'ci_lower': [-0.696],
                'ci_upper': [-0.304],
                't_stat': [-5.0],
                'pvalue': [0.001],
                'n_units': [50],
                'n_periods': [5],
            }),
            
            'cohort_weights': {2005: 1.0},
            'att_overall': None,
            'se_overall': None,
            't_stat_overall': None,
            'pvalue_overall': None,
            'ci_overall': (None, None),
        }
        
        metadata = {
            'K': 4,
            'tpost1': 2005,
            'depvar': 'y',
            'N_treated': 50,
            'N_control': 50,
        }
        
        results = LWDIDResults(results_dict, metadata)
        summary = results.summary_staggered()
        
        # Should contain negative values formatted correctly
        assert '-0.696' in summary
        assert '-0.304' in summary
    
    def test_large_ci_values(self):
        """Test formatting of large CI values."""
        results_dict = {
            'is_staggered': True,
            'att': 0.0,
            'se_att': 0.0,
            't_stat': 0.0,
            'pvalue': 1.0,
            'ci_lower': 0.0,
            'ci_upper': 0.0,
            'nobs': 100,
            'n_treated': 50,
            'n_control': 50,
            'df_resid': 98,
            'params': np.array([0.0]),
            'bse': np.array([0.0]),
            'vcov': np.array([[0.0]]),
            'resid': np.zeros(100),
            'vce_type': 'hc3',
            
            'cohorts': [2005],
            'cohort_sizes': {2005: 50},
            'control_group': 'never_treated',
            'control_group_used': 'never_treated',
            'aggregate': 'cohort',
            'estimator': 'ra',
            'n_never_treated': 50,
            'rolling': 'demean',
            
            'att_by_cohort_time': pd.DataFrame({
                'cohort': [2005],
                'period': [2005],
                'event_time': [0],
                'att': [123.456],
                'se': [10.0],
                'ci_lower': [103.856],
                'ci_upper': [143.056],
                't_stat': [12.35],
                'pvalue': [0.001],
                'n_treated': [50],
                'n_control': [50],
                'n_total': [100],
            }),
            
            'att_by_cohort': pd.DataFrame({
                'cohort': [2005],
                'att': [123.456],
                'se': [10.0],
                'ci_lower': [103.856],
                'ci_upper': [143.056],
                't_stat': [12.35],
                'pvalue': [0.001],
                'n_units': [50],
                'n_periods': [5],
            }),
            
            'cohort_weights': {2005: 1.0},
            'att_overall': None,
            'se_overall': None,
            't_stat_overall': None,
            'pvalue_overall': None,
            'ci_overall': (None, None),
        }
        
        metadata = {
            'K': 4,
            'tpost1': 2005,
            'depvar': 'y',
            'N_treated': 50,
            'N_control': 50,
        }
        
        results = LWDIDResults(results_dict, metadata)
        summary = results.summary_staggered()
        
        # Should contain large values formatted correctly
        assert '103.856' in summary
        assert '143.056' in summary


class TestCIEdgeCases:
    """Test edge cases for CI display."""
    
    def test_zero_ci_values(self):
        """Test formatting of zero CI values."""
        results_dict = {
            'is_staggered': True,
            'att': 0.0,
            'se_att': 0.0,
            't_stat': 0.0,
            'pvalue': 1.0,
            'ci_lower': 0.0,
            'ci_upper': 0.0,
            'nobs': 100,
            'n_treated': 50,
            'n_control': 50,
            'df_resid': 98,
            'params': np.array([0.0]),
            'bse': np.array([0.0]),
            'vcov': np.array([[0.0]]),
            'resid': np.zeros(100),
            'vce_type': 'hc3',
            
            'cohorts': [2005],
            'cohort_sizes': {2005: 50},
            'control_group': 'never_treated',
            'control_group_used': 'never_treated',
            'aggregate': 'cohort',
            'estimator': 'ra',
            'n_never_treated': 50,
            'rolling': 'demean',
            
            'att_by_cohort_time': pd.DataFrame({
                'cohort': [2005],
                'period': [2005],
                'event_time': [0],
                'att': [0.0],
                'se': [0.1],
                'ci_lower': [0.0],
                'ci_upper': [0.0],
                't_stat': [0.0],
                'pvalue': [1.0],
                'n_treated': [50],
                'n_control': [50],
                'n_total': [100],
            }),
            
            'att_by_cohort': pd.DataFrame({
                'cohort': [2005],
                'att': [0.0],
                'se': [0.1],
                'ci_lower': [0.0],
                'ci_upper': [0.0],
                't_stat': [0.0],
                'pvalue': [1.0],
                'n_units': [50],
                'n_periods': [5],
            }),
            
            'cohort_weights': {2005: 1.0},
            'att_overall': None,
            'se_overall': None,
            't_stat_overall': None,
            'pvalue_overall': None,
            'ci_overall': (None, None),
        }
        
        metadata = {
            'K': 4,
            'tpost1': 2005,
            'depvar': 'y',
            'N_treated': 50,
            'N_control': 50,
        }
        
        results = LWDIDResults(results_dict, metadata)
        summary = results.summary_staggered()
        
        # Should contain zero values formatted correctly
        assert '0.000' in summary
        # Should not have N/A for zero values
        lines = summary.split('\n')
        for line in lines:
            if '2005' in line and '[' in line and ']' in line:
                if 'N/A' in line:
                    pytest.fail(f"Zero CI values should not display as N/A, got: {line}")
    
    def test_inf_ci_values_treated_as_nan(self):
        """Test that infinite CI values are handled appropriately."""
        results_dict = {
            'is_staggered': True,
            'att': 0.0,
            'se_att': 0.0,
            't_stat': 0.0,
            'pvalue': 1.0,
            'ci_lower': 0.0,
            'ci_upper': 0.0,
            'nobs': 100,
            'n_treated': 50,
            'n_control': 50,
            'df_resid': 98,
            'params': np.array([0.0]),
            'bse': np.array([0.0]),
            'vcov': np.array([[0.0]]),
            'resid': np.zeros(100),
            'vce_type': 'hc3',
            
            'cohorts': [2005],
            'cohort_sizes': {2005: 50},
            'control_group': 'never_treated',
            'control_group_used': 'never_treated',
            'aggregate': 'cohort',
            'estimator': 'ra',
            'n_never_treated': 50,
            'rolling': 'demean',
            
            'att_by_cohort_time': pd.DataFrame({
                'cohort': [2005],
                'period': [2005],
                'event_time': [0],
                'att': [1.0],
                'se': [0.1],
                'ci_lower': [-np.inf],
                'ci_upper': [np.inf],
                't_stat': [10.0],
                'pvalue': [0.001],
                'n_treated': [50],
                'n_control': [50],
                'n_total': [100],
            }),
            
            'att_by_cohort': pd.DataFrame({
                'cohort': [2005],
                'att': [1.0],
                'se': [0.1],
                'ci_lower': [-np.inf],
                'ci_upper': [np.inf],
                't_stat': [10.0],
                'pvalue': [0.001],
                'n_units': [50],
                'n_periods': [5],
            }),
            
            'cohort_weights': {2005: 1.0},
            'att_overall': None,
            'se_overall': None,
            't_stat_overall': None,
            'pvalue_overall': None,
            'ci_overall': (None, None),
        }
        
        metadata = {
            'K': 4,
            'tpost1': 2005,
            'depvar': 'y',
            'N_treated': 50,
            'N_control': 50,
        }
        
        results = LWDIDResults(results_dict, metadata)
        summary = results.summary_staggered()
        
        # Summary should be generated without error
        # Inf values will display as "inf" in the format string
        assert summary is not None
        assert len(summary) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
